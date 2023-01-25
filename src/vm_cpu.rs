// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::{mem::size_of, ptr::NonNull};
use drivers::{imsic::*, CpuId, CpuInfo, MAX_CPUS};
use memoffset::offset_of;
use page_tracking::collections::PageBox;
use page_tracking::TlbVersion;
use riscv_page_tables::GuestStagePagingMode;
use riscv_pages::{GuestPhysAddr, GuestVirtAddr, PageOwnerId, RawAddr};
use riscv_regs::*;
use sbi_rs::{self, api::tee_host::TsmShmemAreaRef, SbiMessage, SbiReturn, SbiReturnType};
use spin::{Mutex, MutexGuard, Once, RwLock};

use crate::smp::PerCpu;
use crate::vm::{MmioOpcode, MmioOperation, VmExitCause};
use crate::vm_id::VmId;
use crate::vm_interrupts::{self, VmCpuExtInterrupts};
use crate::vm_pages::{ActiveVmPages, FinalizedVmPages, PinnedPages};
use crate::vm_pmu::VmPmuState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BadVmCpuId,
    VmCpuExists,
    VmCpuNotFound,
    VmCpuRunning,
    VmCpuOff,
    VmCpuAlreadyPowered,
    VmCpuBlocked,
    WrongAddressSpace,
    InvalidSharedStatePtr,
    InsufficientSharedStatePages,
    NoImsicVirtualization,
    ImsicLocationAlreadySet,
    VmCpuNotBound,
    Binding(vm_interrupts::Error),
    Rebinding(vm_interrupts::Error),
    Unbinding(vm_interrupts::Error),
    AllowingInterrupt(vm_interrupts::Error),
    DenyingInterrupt(vm_interrupts::Error),
    InjectingInterrupt(vm_interrupts::Error),
    InvalidCsrAccess,
}

pub type Result<T> = core::result::Result<T, Error>;

/// The maximum number of vCPUs supported by a VM.
pub const VM_CPUS_MAX: usize = MAX_CPUS;

/// Hypervisor GPR and CSR state which must be saved/restored when entering/exiting virtualization.
#[derive(Default)]
#[repr(C)]
struct HypervisorCpuState {
    gprs: GeneralPurposeRegisters,
    sstatus: u64,
    hstatus: u64,
    scounteren: u64,
    stvec: u64,
    sscratch: u64,
}

/// Guest GPR and CSR state which must be saved/restored when exiting/entering virtualization.
#[derive(Default)]
#[repr(C)]
struct GuestCpuState {
    gprs: GeneralPurposeRegisters,
    fprs: FloatingPointRegisters,
    vprs: VectorRegisters,
    fcsr: u64,
    sstatus: u64,
    hstatus: u64,
    scounteren: u64,
    sepc: u64,

    vstart: u64,
    vcsr: u64,
    vtype: u64,
    vl: u64,
}

/// The CSRs that are only in effect when virtualization is enabled (V=1) and must be saved and
/// restored whenever we switch between VMs.
#[derive(Default)]
#[repr(C)]
pub struct GuestVsCsrs {
    htimedelta: u64,
    vsstatus: u64,
    vsie: u64,
    vstvec: u64,
    vsscratch: u64,
    vsepc: u64,
    vscause: u64,
    vstval: u64,
    vsatp: u64,
    vstimecmp: u64,
}

/// Virtualized HS-level CSRs that are used to emulate (part of) the hypervisor extension for the
/// guest.
#[derive(Default)]
#[repr(C)]
pub struct GuestVirtualHsCsrs {
    hie: u64,
    hgeie: u64,
}

/// CSRs written on an exit from virtualization that are used by the hypervisor to determine the cause
/// of the trap.
#[derive(Default, Clone)]
#[repr(C)]
pub struct VmCpuTrapState {
    pub scause: u64,
    pub stval: u64,
    pub htval: u64,
    pub htinst: u64,
}

/// (v)CPU register state that must be saved or restored when entering/exiting a VM or switching
/// between VMs.
#[derive(Default)]
#[repr(C)]
struct VmCpuRegisters {
    // CPU state that's shared between our's and the guest's execution environment. Saved/restored
    // when entering/exiting a VM.
    hyp_regs: HypervisorCpuState,
    guest_regs: GuestCpuState,

    // CPU state that only applies when V=1, e.g. the VS-level CSRs. Saved/restored on activation of
    // the vCPU.
    vs_csrs: GuestVsCsrs,

    // Virtualized HS-level CPU state.
    virtual_hs_csrs: GuestVirtualHsCsrs,

    // Read on VM exit.
    trap_csrs: VmCpuTrapState,
}

// The vCPU context switch, defined in guest.S
extern "C" {
    fn _run_guest(state: *mut VmCpuRegisters);
    fn _save_fp(state: *mut VmCpuRegisters);
    fn _restore_fp(state: *mut VmCpuRegisters);
}

extern "C" {
    fn _restore_vector(state: *mut VmCpuRegisters);
    fn _save_vector(state: *mut VmCpuRegisters);
}

#[allow(dead_code)]
const fn hyp_gpr_offset(index: GprIndex) -> usize {
    offset_of!(VmCpuRegisters, hyp_regs)
        + offset_of!(HypervisorCpuState, gprs)
        + (index as usize) * size_of::<u64>()
}

#[allow(dead_code)]
const fn guest_gpr_offset(index: GprIndex) -> usize {
    offset_of!(VmCpuRegisters, guest_regs)
        + offset_of!(GuestCpuState, gprs)
        + (index as usize) * size_of::<u64>()
}

const fn guest_fpr_offset(index: usize) -> usize {
    offset_of!(VmCpuRegisters, guest_regs)
        + offset_of!(GuestCpuState, fprs)
        + index * size_of::<u64>()
}

const fn guest_vpr_offset(index: usize) -> usize {
    offset_of!(VmCpuRegisters, guest_regs)
        + offset_of!(GuestCpuState, vprs)
        + index * size_of::<riscv_regs::VectorRegister>()
}

macro_rules! hyp_csr_offset {
    ($reg:tt) => {
        offset_of!(VmCpuRegisters, hyp_regs) + offset_of!(HypervisorCpuState, $reg)
    };
}

macro_rules! guest_csr_offset {
    ($reg:tt) => {
        offset_of!(VmCpuRegisters, guest_regs) + offset_of!(GuestCpuState, $reg)
    };
}

global_asm!(
    include_str!("guest.S"),
    hyp_ra = const hyp_gpr_offset(GprIndex::RA),
    hyp_gp = const hyp_gpr_offset(GprIndex::GP),
    hyp_tp = const hyp_gpr_offset(GprIndex::TP),
    hyp_s0 = const hyp_gpr_offset(GprIndex::S0),
    hyp_s1 = const hyp_gpr_offset(GprIndex::S1),
    hyp_a1 = const hyp_gpr_offset(GprIndex::A1),
    hyp_a2 = const hyp_gpr_offset(GprIndex::A2),
    hyp_a3 = const hyp_gpr_offset(GprIndex::A3),
    hyp_a4 = const hyp_gpr_offset(GprIndex::A4),
    hyp_a5 = const hyp_gpr_offset(GprIndex::A5),
    hyp_a6 = const hyp_gpr_offset(GprIndex::A6),
    hyp_a7 = const hyp_gpr_offset(GprIndex::A7),
    hyp_s2 = const hyp_gpr_offset(GprIndex::S2),
    hyp_s3 = const hyp_gpr_offset(GprIndex::S3),
    hyp_s4 = const hyp_gpr_offset(GprIndex::S4),
    hyp_s5 = const hyp_gpr_offset(GprIndex::S5),
    hyp_s6 = const hyp_gpr_offset(GprIndex::S6),
    hyp_s7 = const hyp_gpr_offset(GprIndex::S7),
    hyp_s8 = const hyp_gpr_offset(GprIndex::S8),
    hyp_s9 = const hyp_gpr_offset(GprIndex::S9),
    hyp_s10 = const hyp_gpr_offset(GprIndex::S10),
    hyp_s11 = const hyp_gpr_offset(GprIndex::S11),
    hyp_sp = const hyp_gpr_offset(GprIndex::SP),
    hyp_sstatus = const hyp_csr_offset!(sstatus),
    hyp_hstatus = const hyp_csr_offset!(hstatus),
    hyp_scounteren = const hyp_csr_offset!(scounteren),
    hyp_stvec = const hyp_csr_offset!(stvec),
    hyp_sscratch = const hyp_csr_offset!(sscratch),
    guest_ra = const guest_gpr_offset(GprIndex::RA),
    guest_gp = const guest_gpr_offset(GprIndex::GP),
    guest_tp = const guest_gpr_offset(GprIndex::TP),
    guest_s0 = const guest_gpr_offset(GprIndex::S0),
    guest_s1 = const guest_gpr_offset(GprIndex::S1),
    guest_a0 = const guest_gpr_offset(GprIndex::A0),
    guest_a1 = const guest_gpr_offset(GprIndex::A1),
    guest_a2 = const guest_gpr_offset(GprIndex::A2),
    guest_a3 = const guest_gpr_offset(GprIndex::A3),
    guest_a4 = const guest_gpr_offset(GprIndex::A4),
    guest_a5 = const guest_gpr_offset(GprIndex::A5),
    guest_a6 = const guest_gpr_offset(GprIndex::A6),
    guest_a7 = const guest_gpr_offset(GprIndex::A7),
    guest_s2 = const guest_gpr_offset(GprIndex::S2),
    guest_s3 = const guest_gpr_offset(GprIndex::S3),
    guest_s4 = const guest_gpr_offset(GprIndex::S4),
    guest_s5 = const guest_gpr_offset(GprIndex::S5),
    guest_s6 = const guest_gpr_offset(GprIndex::S6),
    guest_s7 = const guest_gpr_offset(GprIndex::S7),
    guest_s8 = const guest_gpr_offset(GprIndex::S8),
    guest_s9 = const guest_gpr_offset(GprIndex::S9),
    guest_s10 = const guest_gpr_offset(GprIndex::S10),
    guest_s11 = const guest_gpr_offset(GprIndex::S11),
    guest_t0 = const guest_gpr_offset(GprIndex::T0),
    guest_t1 = const guest_gpr_offset(GprIndex::T1),
    guest_t2 = const guest_gpr_offset(GprIndex::T2),
    guest_t3 = const guest_gpr_offset(GprIndex::T3),
    guest_t4 = const guest_gpr_offset(GprIndex::T4),
    guest_t5 = const guest_gpr_offset(GprIndex::T5),
    guest_t6 = const guest_gpr_offset(GprIndex::T6),
    guest_sp = const guest_gpr_offset(GprIndex::SP),
    guest_f0 = const guest_fpr_offset(0),
    guest_f1 = const guest_fpr_offset(1),
    guest_f2 = const guest_fpr_offset(2),
    guest_f3 = const guest_fpr_offset(3),
    guest_f4 = const guest_fpr_offset(4),
    guest_f5 = const guest_fpr_offset(5),
    guest_f6 = const guest_fpr_offset(6),
    guest_f7 = const guest_fpr_offset(7),
    guest_f8 = const guest_fpr_offset(8),
    guest_f9 = const guest_fpr_offset(9),
    guest_f10 = const guest_fpr_offset(10),
    guest_f11 = const guest_fpr_offset(11),
    guest_f12 = const guest_fpr_offset(12),
    guest_f13 = const guest_fpr_offset(13),
    guest_f14 = const guest_fpr_offset(14),
    guest_f15 = const guest_fpr_offset(15),
    guest_f16 = const guest_fpr_offset(16),
    guest_f17 = const guest_fpr_offset(17),
    guest_f18 = const guest_fpr_offset(18),
    guest_f19 = const guest_fpr_offset(19),
    guest_f20 = const guest_fpr_offset(20),
    guest_f21 = const guest_fpr_offset(21),
    guest_f22 = const guest_fpr_offset(22),
    guest_f23 = const guest_fpr_offset(23),
    guest_f24 = const guest_fpr_offset(24),
    guest_f25 = const guest_fpr_offset(25),
    guest_f26 = const guest_fpr_offset(26),
    guest_f27 = const guest_fpr_offset(27),
    guest_f28 = const guest_fpr_offset(28),
    guest_f29 = const guest_fpr_offset(29),
    guest_f30 = const guest_fpr_offset(30),
    guest_f31 = const guest_fpr_offset(31),
    guest_fcsr = const guest_csr_offset!(fcsr),
    guest_v0 = const guest_vpr_offset(0),
    guest_v8 = const guest_vpr_offset(8),
    guest_v16 = const guest_vpr_offset(16),
    guest_v24 = const guest_vpr_offset(24),
    guest_vstart = const guest_csr_offset!(vstart),
    guest_vcsr = const guest_csr_offset!(vcsr),
    guest_vtype = const guest_csr_offset!(vtype),
    guest_vl = const guest_csr_offset!(vl),
    guest_sstatus = const guest_csr_offset!(sstatus),
    guest_hstatus = const guest_csr_offset!(hstatus),
    guest_scounteren = const guest_csr_offset!(scounteren),
    guest_sepc = const guest_csr_offset!(sepc),
    sstatus_fs_dirty = const sstatus::fs::Dirty.value,
    sstatus_vs_enable = const sstatus::vs::Initial.value,
);

// Wrapper for a `NaclShmem` struct pinned in host shared memory.
struct PinnedTsmShmemArea {
    ptr: NonNull<sbi_rs::NaclShmem>,
    // Optional since we might be sharing with the hypervisor in the host VM case.
    _pin: Option<PinnedPages>,
}

impl PinnedTsmShmemArea {
    // Creates a new `PinnedTsmShmemArea` from a set of pinned shared pages.
    fn new(pages: PinnedPages) -> Result<Self> {
        // Make sure the pin actually covers the size of the structure.
        if pages.range().length_bytes() < size_of::<sbi_rs::NaclShmem>() as u64 {
            return Err(Error::InsufficientSharedStatePages);
        }
        let ptr = pages.range().base().bits() as *mut sbi_rs::NaclShmem;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::InvalidSharedStatePtr)?,
            _pin: Some(pages),
        })
    }

    // Returns the wrapped shared state buffer as a `TsmShmemAreaRef`.
    fn as_ref(&self) -> TsmShmemAreaRef {
        // Safety: We've validated at construction that self.ptr points to a valid `TsmShmemArea`.
        unsafe { TsmShmemAreaRef::new(self.ptr.as_ptr()) }
    }
}

/// Identifies the reason for a trap taken from a vCPU.
pub enum VmCpuTrap {
    /// ECALLs from VS mode.
    Ecall(Option<SbiMessage>),
    /// G-stage page faults.
    PageFault {
        exception: Exception,
        fault_addr: GuestPhysAddr,
        fault_pc: GuestVirtAddr,
        priv_level: PrivilegeLevel,
    },
    /// Instruction emulation trap.
    VirtualInstruction {
        fault_pc: GuestVirtAddr,
        priv_level: PrivilegeLevel,
    },
    /// An exception which we expected to handle directly at VS, but trapped to HS instead.
    DelegatedException { exception: Exception, stval: u64 },
    /// Everything other exception that we currently don't or can't handle.
    OtherException(VmCpuTrapState),
    /// An interrupt intended for the vCPU's host.
    HostInterrupt(Interrupt),
    /// An interrupt for the running vCPU that can't be delegated and must be injected. The
    /// interrupt is injected the vCPU is run.
    InterruptEmulation,
    /// Unknown / unexpected interupt.
    OtherInterrupt(Interrupt),
    // TODO: Add other exit causes as needed.
}

// The VMID and TLB version last used by a vCPU.
struct PrevTlb {
    cpu: CpuId,
    vmid: VmId,
    tlb_version: TlbVersion,
}

// An operation that's pending a return value from the vCPU's host.
enum PendingOperation {
    Mmio(MmioOperation),
    Ecall(SbiMessage),
}

// The architectural state of a vCPU.
struct VmCpuArchState {
    regs: VmCpuRegisters,
    pmu: VmPmuState,
    prev_tlb: Option<PrevTlb>,
    pending_op: Option<PendingOperation>,
    shmem_area: Option<PinnedTsmShmemArea>,
}

impl VmCpuArchState {
    // Sets up the architectural state for a new vCPU.
    fn new(guest_id: PageOwnerId) -> Self {
        let mut regs = VmCpuRegisters::default();

        let mut hstatus = LocalRegisterCopy::<u64, hstatus::Register>::new(0);
        hstatus.modify(hstatus::spv.val(1));
        hstatus.modify(hstatus::spvp::Supervisor);
        if !guest_id.is_host() {
            // Trap on WFI for non-host VMs. Trapping WFI for the host is pointless since all we'd do
            // in the hypervisor is WFI ourselves.
            hstatus.modify(hstatus::vtw.val(1));
        }
        regs.guest_regs.hstatus = hstatus.get();

        let mut sstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(0);
        sstatus.modify(sstatus::spp::Supervisor);
        sstatus.modify(sstatus::fs::Initial);
        if CpuInfo::get().has_vector() {
            sstatus.modify(sstatus::vs::Initial);
        }
        regs.guest_regs.sstatus = sstatus.get();

        let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
        scounteren.modify(scounteren::cycle.val(1));
        scounteren.modify(scounteren::time.val(1));
        scounteren.modify(scounteren::instret.val(1));
        regs.guest_regs.scounteren = scounteren.get();

        Self {
            regs,
            pmu: VmPmuState::default(),
            prev_tlb: None,
            pending_op: None,
            shmem_area: None,
        }
    }
}

// Sets the status on dropping depending on the state of next_status.
// Used by ActiveVmCpu to set the Vcpu's state on de-activation.
struct StatusSet<'vcpu> {
    vcpu: &'vcpu VmCpu,
    next_status: VmCpuStatus,
}

impl<'vcpu> StatusSet<'vcpu> {
    fn new(vcpu: &'vcpu VmCpu) -> Self {
        // vCPUs return to the "Runnable" state by default.
        Self {
            vcpu,
            next_status: VmCpuStatus::Runnable,
        }
    }
}

impl<'vcpu> Drop for StatusSet<'vcpu> {
    fn drop(&mut self) {
        let mut status = self.vcpu.status.write();
        assert_eq!(*status, VmCpuStatus::Running);
        *status = self.next_status;
    }
}

/// Interface for reporting the exit status of `VmCpu` to its host.
pub trait VmCpuExitReporting {
    /// Sets a CSR for the host (v)CPU.
    fn set_csr(&mut self, csr_num: u16, val: u64);

    /// Sets a guest GPR value for the host (v)CPU.
    fn set_guest_gpr(&mut self, index: GprIndex, val: u64);

    /// Gets a guest GPR value from the host (v)CPU.
    fn guest_gpr(&self, index: GprIndex) -> u64;
}

/// Interface to the host vCPU context when a `VmCpu` is being hosted by another `VmCpu`. Used to
/// save/restore state that's shared between host and guest vCPUs, e.g. VS-level CSRs.
///
/// Note that a trait is used here to avoid the lifetime recursion when holding an `ActiveVmCpu`
/// within another `ActiveVmCpu`.
pub trait HostVmCpu: VmCpuExitReporting {
    /// Saves host vCPU state. Called when the host activates a guest `VmCpu`.
    fn save(&mut self);

    /// Restores host vCPU state. Called when a guest `VmCpu` is deactivated, i.e. when the
    /// `ActiveVmCpu` is dropped.
    fn restore(&mut self);

    /// Returns a reference to the vCPU's VS-level CSRs.
    fn vs_csrs(&self) -> &GuestVsCsrs;

    /// Returns a reference to the vCPU's virtual HS-level CSRs.
    fn virtual_hs_csrs(&self) -> &GuestVirtualHsCsrs;

    /// Returns the guest interrupt file to which the vCPU is currently bound.
    fn bound_interrupt_file(&self) -> Option<ImsicFileId>;
}

/// The parent (host) context of a `VmCpu`.
pub enum VmCpuParent<'host> {
    /// The `VmCpu` is hosted by the TSM itself. This is the case for vCPUs of the host VM.
    Tsm(&'host mut dyn VmCpuExitReporting),
    /// The `VmCpu` is hosted by a vCPU of the host VM.
    HostVm(&'host mut dyn HostVmCpu),
}

impl VmCpuParent<'_> {
    fn set_csr(&mut self, csr_num: u16, val: u64) {
        match self {
            VmCpuParent::Tsm(host_runner) => host_runner.set_csr(csr_num, val),
            VmCpuParent::HostVm(host_vcpu) => host_vcpu.set_csr(csr_num, val),
        }
    }

    fn set_guest_gpr(&mut self, index: GprIndex, val: u64) {
        match self {
            VmCpuParent::Tsm(host_runner) => host_runner.set_guest_gpr(index, val),
            VmCpuParent::HostVm(host_vcpu) => host_vcpu.set_guest_gpr(index, val),
        }
    }

    fn guest_gpr(&self, index: GprIndex) -> u64 {
        match self {
            VmCpuParent::Tsm(host_runner) => host_runner.guest_gpr(index),
            VmCpuParent::HostVm(host_vcpu) => host_vcpu.guest_gpr(index),
        }
    }
}

/// An activated vCPU. A vCPU in this state has entered the VM's address space and is ready to run.
pub struct ActiveVmCpu<'vcpu, 'pages, 'host, T: GuestStagePagingMode> {
    vcpu: &'vcpu VmCpu,
    // We hold a lock on vcpu.arch for the lifetime of this object to avoid having to
    // repeatedly acquire it for every register modification. Must be dropped before `status_set`,
    // so declared first.
    arch: MutexGuard<'vcpu, VmCpuArchState>,
    vm_pages: FinalizedVmPages<'pages, T>,
    // `None` if this vCPU is itself running a child vCPU. Restored when the child vCPU exits.
    active_pages: Option<ActiveVmPages<'pages, T>>,
    // The context of the (v)CPU that is hosting this one.
    host_context: VmCpuParent<'host>,
    // Important drop order, status_set must come _after_ `arch` to maintain lock ordering.
    // on drop StatusSet takes the status lock.
    status_set: StatusSet<'vcpu>,
}

impl<'vcpu, 'pages, 'host, T: GuestStagePagingMode> ActiveVmCpu<'vcpu, 'pages, 'host, T> {
    // Restores and activates the vCPU state from `vcpu`, with the VM address space represented by
    // `vm_pages`.
    fn restore_from(
        vcpu: &'vcpu VmCpu,
        vm_pages: FinalizedVmPages<'pages, T>,
        mut host_context: VmCpuParent<'host>,
    ) -> Result<Self> {
        let mut arch = vcpu.arch.lock();
        // If we're running on a new CPU, then any previous VMID or TLB version we used is inavlid.
        if let Some(ref prev_tlb) = arch.prev_tlb && prev_tlb.cpu != PerCpu::this_cpu().cpu_id() {
            arch.prev_tlb = None;
        }

        // Save the state of the host vCPU (if any) and swap in ours.
        if let VmCpuParent::HostVm(ref mut host_vcpu) = host_context {
            host_vcpu.save();
        }
        let mut active_vcpu = Self {
            vcpu,
            arch,
            vm_pages,
            active_pages: None,
            host_context,
            status_set: StatusSet::new(vcpu),
        };
        active_vcpu.restore();

        Ok(active_vcpu)
    }

    /// Runs this vCPU until it traps.
    pub fn run(&mut self) -> VmCpuTrap {
        self.complete_pending_op();

        match self.host_context {
            VmCpuParent::HostVm(ref host_vcpu) => {
                // TODO: Consider bailing out early if any of the below host interrupts are pending.

                // Make sure we trap if the host has timer interrupts enabled.
                let host_sie = LocalRegisterCopy::new(host_vcpu.vs_csrs().vsie);
                if host_sie.read(sie::stimer) != 0 {
                    CSR.sie.read_and_set_field(sie::stimer);
                }
                // Also trap if the host has external or guest external interrupts enabled.
                // Both cases will trap as SG_EXT. We'll disambiguate later.
                let host_hie = LocalRegisterCopy::new(host_vcpu.virtual_hs_csrs().hie);
                if host_hie.read(hie::sgext) != 0 || host_sie.read(sie::sext) != 0 {
                    CSR.hie.read_and_set_field(hie::sgext);
                }
            }
            VmCpuParent::Tsm(_) => {
                let hie = LocalRegisterCopy::new(self.arch.regs.virtual_hs_csrs.hie);
                if hie.read(hie::sgext) != 0 {
                    if CSR.hip.read(hip::sgext) != 0 {
                        // SG_EXT is pending; inject it with HVICTL. Leave it disabled in HIE to avoid
                        // trapping immediately once we enter the host VM.
                        let mut hvictl = LocalRegisterCopy::new(0);
                        hvictl.modify(hvictl::iid.val(Interrupt::SupervisorGuestExternal as u64));
                        // We need VTI=1 to inject major interrupts, which in this case means we
                        // may get some unnecessary traps.
                        hvictl.modify(hvictl::vti.val(1));
                        // Set IPRIOM=1 and IPRIO=0 to get the default priority.
                        hvictl.modify(hvictl::ipriom.val(1));
                        CSR.hvictl.set(hvictl.get());
                    } else {
                        CSR.hie.read_and_set_field(hie::sgext);
                    }
                }
            }
        }

        let has_vector = CpuInfo::get().has_vector();
        let guest_id = self.vcpu.guest_id;
        let regs = &mut self.arch.regs;

        unsafe {
            // Safe since _restore_vector() only reads within the bounds of the vector register
            // state in VmCpuRegisters.
            if has_vector {
                _restore_vector(regs);
            }

            // Safe since _restore_fp() only reads within the bounds of the floating point
            // register state in VmCpuRegisters.
            _restore_fp(regs);

            // Safe to run the guest as it only touches memory assigned to it by being owned
            // by its page table.
            _run_guest(regs);
        }

        // Save off the trap information.
        regs.trap_csrs.scause = CSR.scause.get();
        regs.trap_csrs.stval = CSR.stval.get();
        regs.trap_csrs.htval = CSR.htval.get();
        regs.trap_csrs.htinst = CSR.htinst.get();

        CSR.sie.read_and_clear_field(sie::stimer);
        CSR.hie.read_and_clear_field(hie::sgext);
        CSR.hvictl.set(0);

        // Check if FPU state needs to be saved.
        let mut sstatus = LocalRegisterCopy::new(regs.guest_regs.sstatus);
        if sstatus.matches_all(sstatus::fs::Dirty) {
            // Safe since _save_fp() only writes within the bounds of the floating point register
            // state in VmCpuRegisters.
            unsafe { _save_fp(regs) };
            sstatus.modify(sstatus::fs::Clean);
        }
        // Check if vector state needs to be saved
        if has_vector && sstatus.matches_all(sstatus::vs::Dirty) {
            // Safe since _save_vector() only writes within the bounds of the vector register
            // state in VmCpuRegisters.
            unsafe { _save_vector(regs) };
            sstatus.modify(sstatus::vs::Clean)
        }
        regs.guest_regs.sstatus = sstatus.get();

        // Determine the exit cause from the trap CSRs.
        use Exception::*;
        use Interrupt::*;
        match Trap::from_scause(regs.trap_csrs.scause).unwrap() {
            Trap::Exception(VirtualSupervisorEnvCall) => {
                let sbi_msg = SbiMessage::from_regs(regs.guest_regs.gprs.a_regs()).ok();
                VmCpuTrap::Ecall(sbi_msg)
            }
            Trap::Exception(GuestInstructionPageFault)
            | Trap::Exception(GuestLoadPageFault)
            | Trap::Exception(GuestStorePageFault) => {
                let fault_addr = RawAddr::guest(
                    regs.trap_csrs.htval << 2 | regs.trap_csrs.stval & 0x03,
                    guest_id,
                );
                VmCpuTrap::PageFault {
                    exception: Exception::from_scause_reason(regs.trap_csrs.scause).unwrap(),
                    fault_addr,
                    // Note that this address is not necessarily guest virtual as the guest may or
                    // may not have 1st-stage translation enabled in VSATP. We still use GuestVirtAddr
                    // here though to distinguish it from addresses (e.g. in HTVAL, or passed via a
                    // TEECALL) which are exclusively guest-physical. Furthermore we only access guest
                    // instructions via the HLVX instruction, which will take the VSATP translation
                    // mode into account.
                    fault_pc: RawAddr::guest_virt(regs.guest_regs.sepc, guest_id),
                    priv_level: PrivilegeLevel::from_hstatus(regs.guest_regs.hstatus),
                }
            }
            Trap::Exception(VirtualInstruction) => {
                VmCpuTrap::VirtualInstruction {
                    // See above re: this address being guest virtual.
                    fault_pc: RawAddr::guest_virt(regs.guest_regs.sepc, guest_id),
                    priv_level: PrivilegeLevel::from_hstatus(regs.guest_regs.hstatus),
                }
            }
            Trap::Exception(e) => {
                if e.to_hedeleg_field()
                    .map_or(false, |f| CSR.hedeleg.matches_any(f))
                {
                    // Even if we intended to delegate this exception it might not be set in
                    // medeleg, in which case firmware may send it our way instead.
                    VmCpuTrap::DelegatedException {
                        exception: e,
                        stval: regs.trap_csrs.stval,
                    }
                } else {
                    VmCpuTrap::OtherException(regs.trap_csrs.clone())
                }
            }
            Trap::Interrupt(SupervisorTimer) => VmCpuTrap::HostInterrupt(SupervisorTimer),
            Trap::Interrupt(SupervisorGuestExternal) => {
                if let VmCpuParent::HostVm(ref host_vcpu) = self.host_context {
                    // We may have gotten an SG_EXT because of an external interrupt directed at
                    // our host VM. If so, that takes priority.
                    let host_sie = LocalRegisterCopy::new(host_vcpu.vs_csrs().vsie);
                    if let Some(host_vgein) = host_vcpu.bound_interrupt_file().map(|f| f.bits()) &&
                        host_sie.read(sie::sext) != 0 &&
                        (CSR.hgeip.get() & (1 << host_vgein)) != 0
                    {
                        VmCpuTrap::HostInterrupt(SupervisorExternal)
                    } else {
                        VmCpuTrap::HostInterrupt(SupervisorGuestExternal)
                    }
                } else {
                    // We trapped because we need to emulate an SG_EXT for the host VM.
                    VmCpuTrap::InterruptEmulation
                }
            }
            Trap::Interrupt(i) => VmCpuTrap::OtherInterrupt(i),
        }
    }

    // Rewrites `mmio_op` as a transformed load or store instruction to/from A0 as would be written
    // to the HTINST CSR.
    fn mmio_op_to_htinst(mmio_op: MmioOperation) -> u64 {
        use MmioOpcode::*;
        // Get the base instruction for the operation.
        let htinst_base = match mmio_op.opcode() {
            Store8 => MATCH_SB,
            Store16 => MATCH_SH,
            Store32 => MATCH_SW,
            Store64 => MATCH_SD,
            Load8 => MATCH_LB,
            Load16 => MATCH_LH,
            Load32 => MATCH_LW,
            Load8U => MATCH_LBU,
            Load16U => MATCH_LHU,
            Load32U => MATCH_LWU,
            Load64 => MATCH_LD,
        };
        // Set rd (for loads) or rs2 (for stores) to A0.
        let htinst = if mmio_op.opcode().is_load() {
            htinst_base | ((GprIndex::A0 as u32) << 20)
        } else {
            htinst_base | ((GprIndex::A0 as u32) << 7)
        };
        htinst as u64
    }

    fn report_ecall_exit(&mut self, msg: SbiMessage) {
        self.host_context.set_guest_gpr(GprIndex::A0, msg.a0());
        self.host_context.set_guest_gpr(GprIndex::A1, msg.a1());
        self.host_context.set_guest_gpr(GprIndex::A2, msg.a2());
        self.host_context.set_guest_gpr(GprIndex::A3, msg.a3());
        self.host_context.set_guest_gpr(GprIndex::A4, msg.a4());
        self.host_context.set_guest_gpr(GprIndex::A5, msg.a5());
        self.host_context.set_guest_gpr(GprIndex::A6, msg.a6());
        self.host_context.set_guest_gpr(GprIndex::A7, msg.a7());
        self.host_context.set_csr(CSR_STVAL, 0);
        self.host_context.set_csr(CSR_HTVAL, 0);
        self.host_context.set_csr(CSR_HTINST, 0);
        self.host_context
            .set_csr(CSR_SCAUSE, Exception::VirtualSupervisorEnvCall as u64);
    }

    fn report_pf_exit(&mut self, exception: Exception, addr: GuestPhysAddr) {
        self.host_context.set_csr(CSR_STVAL, addr.bits() & 0x3);
        self.host_context.set_csr(CSR_HTVAL, addr.bits() >> 2);
        self.host_context.set_csr(CSR_HTINST, 0);
        self.host_context.set_csr(CSR_SCAUSE, exception as u64);
    }

    fn report_vi_exit(&mut self, inst: u64) {
        // Note that this is technically not spec-compliant as the privileged spec only states
        // that illegal instruction exceptions may write the faulting instruction to *TVAL CSRs.
        self.host_context.set_csr(CSR_STVAL, inst);
        self.host_context.set_csr(CSR_HTVAL, 0);
        self.host_context.set_csr(CSR_HTINST, 0);
        self.host_context
            .set_csr(CSR_SCAUSE, Exception::VirtualInstruction as u64);
    }

    fn report_unhandled_exit(&mut self, scause: u64) {
        self.host_context.set_csr(CSR_STVAL, 0);
        self.host_context.set_csr(CSR_HTVAL, 0);
        self.host_context.set_csr(CSR_HTINST, 0);
        self.host_context.set_csr(CSR_SCAUSE, scause);
    }

    /// Reports the exit cause in `cause` back to the host and deactivates this vCPU. The vCPU is
    /// either returned to the `Available` or `PoweredOff` state, depending on if the exit cause is
    /// resumable.
    pub fn exit(mut self, cause: VmExitCause) {
        self.host_context
            .set_csr(CSR_VSTIMECMP, CSR.vstimecmp.get());
        self.host_context.set_csr(CSR_VSIE, CSR.vsie.get());

        use VmExitCause::*;
        match cause {
            ResumableEcall(msg) | FatalEcall(msg) | BlockingEcall(msg, _) => {
                self.report_ecall_exit(msg);
            }
            ForwardedEcall(msg) => {
                self.report_ecall_exit(msg);
                self.arch.pending_op = Some(PendingOperation::Ecall(msg));
            }
            PageFault(exception, page_addr) => {
                self.report_pf_exit(exception, page_addr.into());
            }
            MmioFault(mmio_op, addr) => {
                let exception = if mmio_op.opcode().is_load() {
                    Exception::GuestLoadPageFault
                } else {
                    Exception::GuestStorePageFault
                };
                self.report_pf_exit(exception, addr);

                // The MMIO instruction is transformed as an ordinary load/store to/from A0, so
                // update A0 with the value the vCPU wants to store.
                use MmioOpcode::*;
                let val = match mmio_op.opcode() {
                    Store8 => self.get_gpr(mmio_op.register()) as u8 as u64,
                    Store16 => self.get_gpr(mmio_op.register()) as u16 as u64,
                    Store32 => self.get_gpr(mmio_op.register()) as u32 as u64,
                    Store64 => self.get_gpr(mmio_op.register()),
                    _ => 0,
                };
                self.host_context
                    .set_csr(CSR_HTINST, Self::mmio_op_to_htinst(mmio_op));
                self.host_context.set_guest_gpr(GprIndex::A0, val);

                // We'll complete a load instruction the next time this vCPU is run.
                self.arch.pending_op = Some(PendingOperation::Mmio(mmio_op));
            }
            Wfi(inst) => {
                self.report_vi_exit(inst.raw() as u64);
            }
            UnhandledTrap(scause) => {
                self.report_unhandled_exit(scause);
            }
            HostInterrupt(i) => {
                // Just set up SCAUSE on an interrupt. The interrupt will pend (and trap, if
                // necessary) when the host is swapped in.
                self.host_context
                    .set_csr(CSR_SCAUSE, Trap::Interrupt(i).to_scause());
            }
        };

        if cause.is_fatal() {
            self.status_set.next_status = VmCpuStatus::PoweredOff;
        } else if let BlockingEcall(_, tlb_version) = cause {
            self.status_set.next_status = VmCpuStatus::Blocked(tlb_version);
        }
    }

    /// Delivers the given exception to the vCPU, setting up its register state to handle the trap
    /// the next time it is run.
    pub fn inject_exception(&mut self, exception: Exception, stval: u64) {
        // Update previous privelege level and interrupt state in VSSTATUS.
        let mut vsstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(CSR.vsstatus.get());
        if vsstatus.is_set(sstatus::sie) {
            vsstatus.modify(sstatus::spie.val(1));
        }
        vsstatus.modify(sstatus::sie.val(0));
        let sstatus =
            LocalRegisterCopy::<u64, sstatus::Register>::new(self.arch.regs.guest_regs.sstatus);
        vsstatus.modify(sstatus::spp.val(sstatus.read(sstatus::spp)));
        CSR.vsstatus.set(vsstatus.get());

        CSR.vscause.set(Trap::Exception(exception).to_scause());
        CSR.vstval.set(stval);
        CSR.vsepc.set(self.arch.regs.guest_regs.sepc);

        // Redirect the vCPU to its STVEC on entry.
        self.arch.regs.guest_regs.sepc = CSR.vstvec.get();
    }

    /// Gets one of the vCPU's general purpose registers.
    pub fn get_gpr(&self, gpr: GprIndex) -> u64 {
        self.arch.regs.guest_regs.gprs.reg(gpr)
    }

    /// Sets one of the vCPU's general-purpose registers.
    pub fn set_gpr(&mut self, gpr: GprIndex, value: u64) {
        self.arch.regs.guest_regs.gprs.set_reg(gpr, value);
    }

    /// Increments the current `sepc` CSR value by `value`.
    pub fn inc_sepc(&mut self, value: u64) {
        self.arch.regs.guest_regs.sepc += value;
    }

    /// Increments SEPC and Updates A0/A1 with the result of an SBI call.
    pub fn set_ecall_result(&mut self, result: SbiReturnType) {
        self.inc_sepc(4); // ECALL is always a 4-byte instruction.
        match result {
            SbiReturnType::Legacy(a0) => {
                self.set_gpr(GprIndex::A0, a0);
            }
            SbiReturnType::Standard(ret) => {
                self.set_gpr(GprIndex::A0, ret.error_code as u64);
                self.set_gpr(GprIndex::A1, ret.return_value);
            }
        }
    }

    /// Emulates a CSR read-modify-write operation taken from a virtual instruction trap. Returns
    /// the previous value of the (virtual) CSR.
    pub fn virtual_csr_rmw(&mut self, csr_num: u16, value: u64, mask: u64) -> Result<u64> {
        if (CSR_CYCLE..=CSR_HPMCOUNTER31).contains(&csr_num) && mask == 0 {
            self.pmu()
                .get_cached_csr_value(csr_num.into())
                .map_err(|_| Error::InvalidCsrAccess)
        } else if self.vcpu.guest_id.is_host() {
            // We emulate a subset of the H-CSRs for the host VM.
            match csr_num {
                CSR_HIE => {
                    let prev = self.arch.regs.virtual_hs_csrs.hie;
                    let mut valid = LocalRegisterCopy::new(0);
                    valid.modify(hie::vssoft.val(1));
                    valid.modify(hie::vstimer.val(1));
                    valid.modify(hie::vsext.val(1));
                    valid.modify(hie::sgext.val(1));
                    // Salus isn't interruptible (for now), so we don't set HIE.SG_EXT until we run
                    // the vCPU to avoid trapping within Salus unnecessarily, e.g. if we're running
                    // a U-mode task.
                    self.arch.regs.virtual_hs_csrs.hie =
                        (prev & !mask) | (valid.get() & mask & value);
                    Ok(prev)
                }
                CSR_HIP => {
                    // Only report SG_EXT. We can use the bit in HIP directly since we keep HGEIE
                    // programmed with the bits the host VM requested.
                    //
                    // We forcibly delegate VS_TIMER/VS_EXT and do not attempt to emulate them for
                    // the host VM; VS_SOFT is unused.
                    let mut valid = LocalRegisterCopy::new(0);
                    valid.write(hip::sgext.val(1));
                    Ok(CSR.hip.get() & valid.get())
                }
                CSR_HGEIE => {
                    if let Ok(ext_interrupts) = self.vcpu.ext_interrupts() {
                        let prev = self.arch.regs.virtual_hs_csrs.hgeie;
                        let valid = ((1 << ext_interrupts.lock().num_guests()) - 1) << 1;
                        let vhgeie = (prev & !mask) | (valid & mask & value);
                        self.arch.regs.virtual_hs_csrs.hgeie = vhgeie;

                        // Now update the real HGEIE. We assume our guests are the interrupt files
                        // immediately following ours. If we were to support another level of
                        // nesting, we'd need to instead remap based on the interrupt files that
                        // our host assigned us.
                        //
                        // Unwrap ok: We must be bound to have been activated with external interrupts.
                        let vgein = self.bound_interrupt_file().unwrap().bits();
                        CSR.hgeie.set((vhgeie << (vgein + 1)) >> 1);
                        Ok(prev)
                    } else {
                        Ok(0)
                    }
                }
                CSR_HGEIP if mask == 0 => {
                    if let Ok(ext_interrupts) = self.vcpu.ext_interrupts() {
                        let valid = ((1 << ext_interrupts.lock().num_guests()) - 1) << 1;
                        // As above, we assume our guests will be immediately following our bit in
                        // HGEIP.
                        //
                        // Unwrap ok: We must be bound to have been activated with external interrupts.
                        let vgein = self.bound_interrupt_file().unwrap().bits();
                        let vhgeip = (CSR.hgeip.get() >> (vgein + 1)) << 1;
                        Ok(vhgeip & valid)
                    } else {
                        Ok(0)
                    }
                }
                // If the vCPU is getting an interrupt injected via HVICTL, we'll trap on accesses to
                // SIE, SIP, and STIMECMP. Just forward the access to the appropriate VS-level CSR.
                CSR_SIE => {
                    let sie = CSR.vsie.get();
                    CSR.vsie.set((sie & !mask) | (value & mask));
                    Ok(sie)
                }
                CSR_SIP => {
                    let sip = CSR.vsip.get();
                    CSR.vsip.set((sip & !mask) | (value & mask));
                    Ok(sip)
                }
                CSR_STIMECMP => {
                    let stimecmp = CSR.vstimecmp.get();
                    CSR.vstimecmp.set((stimecmp & !mask) | (value & mask));
                    Ok(stimecmp)
                }
                _ => Err(Error::InvalidCsrAccess),
            }
        } else {
            Err(Error::InvalidCsrAccess)
        }
    }

    /// Returns the location of this vCPU's virtualized IMSIC.
    pub fn get_imsic_location(&self) -> Option<ImsicLocation> {
        self.vcpu.get_imsic_location()
    }

    /// Returns this active vCPU's `ActiveVmPages`.
    pub fn active_pages(&self) -> &ActiveVmPages<'pages, T> {
        // Unwrap ok since it's not possible to externally hold a reference to an `ActiveVmCpu` with
        // `self.active_pages` set to `None`. The only instance where `self.active_pages` is `None` is
        // when we're swapped out to run a child vCPU, in which case that child `ActiveVmCpu` holds
        // the exclusive reference to the parent `ActiveVmCpu`.
        self.active_pages.as_ref().unwrap()
    }

    /// Performs any pending TLB maintenance for this VM's address space.
    pub fn sync_tlb(&mut self) {
        // Exit and re-enter so that we pick up the new TLB version.
        self.active_pages = None;
        self.restore_vm_pages();
    }

    /// Returns a mutable reference to this active vCPU's PMU state.
    pub fn pmu(&mut self) -> &mut VmPmuState {
        &mut self.arch.pmu
    }

    /// Adds `id` to the list of injectable external interrupts.
    pub fn allow_ext_interrupt(&self, id: usize) -> Result<()> {
        self.vcpu
            .ext_interrupts()?
            .lock()
            .allow_interrupt(id)
            .map_err(Error::AllowingInterrupt)
    }

    /// Allows injection of all external interrupts.
    pub fn allow_all_ext_interrupts(&self) -> Result<()> {
        let ext_interrupts = self.vcpu.ext_interrupts()?;
        ext_interrupts.lock().allow_all_interrupts();
        Ok(())
    }

    /// Removes `id` from the list of injectable external interrupts.
    pub fn deny_ext_interrupt(&self, id: usize) -> Result<()> {
        self.vcpu
            .ext_interrupts()?
            .lock()
            .deny_interrupt(id)
            .map_err(Error::DenyingInterrupt)
    }

    /// Disables injection of all external interrupts.
    pub fn deny_all_ext_interrupts(&self) -> Result<()> {
        let ext_interrupts = self.vcpu.ext_interrupts()?;
        ext_interrupts.lock().deny_all_interrupts();
        Ok(())
    }

    /// Registers `pages` as the host <-> TSM shared-memory communication area for this vCPU.
    pub fn register_shmem_area(&mut self, pages: PinnedPages) -> Result<()> {
        self.arch.shmem_area = Some(PinnedTsmShmemArea::new(pages)?);
        Ok(())
    }

    /// Unregisters this vCPU's host <-> TSM shared-memory communication area.
    pub fn unregister_shmem_area(&mut self) {
        self.arch.shmem_area = None;
    }

    // Completes any pending MMIO or ECALL result from the host for this vCPU.
    fn complete_pending_op(&mut self) {
        match self.arch.pending_op {
            Some(PendingOperation::Mmio(mmio_op)) => {
                // Complete any pending load operations. The host is expected to have written the
                // value to complete the load to A0.
                let val = self.host_context.guest_gpr(GprIndex::A0);
                use MmioOpcode::*;
                // Write the value to the actual destination register.
                match mmio_op.opcode() {
                    Load8 => {
                        self.set_gpr(mmio_op.register(), val as i8 as u64);
                    }
                    Load8U => {
                        self.set_gpr(mmio_op.register(), val as u8 as u64);
                    }
                    Load16 => {
                        self.set_gpr(mmio_op.register(), val as i16 as u64);
                    }
                    Load16U => {
                        self.set_gpr(mmio_op.register(), val as u16 as u64);
                    }
                    Load32 => {
                        self.set_gpr(mmio_op.register(), val as i32 as u64);
                    }
                    Load32U => {
                        self.set_gpr(mmio_op.register(), val as u32 as u64);
                    }
                    Load64 => {
                        self.set_gpr(mmio_op.register(), val);
                    }
                    _ => (),
                };
                self.host_context.set_guest_gpr(GprIndex::A0, 0);

                // Advance SEPC past the faulting instruction.
                self.inc_sepc(mmio_op.len() as u64);
            }
            Some(PendingOperation::Ecall(msg)) => {
                // Forward the SBI call return value from the A0/A1 values provided by the host.
                let sbi_ret = match msg {
                    SbiMessage::PutChar(_) => {
                        SbiReturnType::Legacy(self.host_context.guest_gpr(GprIndex::A0))
                    }
                    _ => SbiReturnType::Standard(SbiReturn {
                        error_code: self.host_context.guest_gpr(GprIndex::A0) as i64,
                        return_value: self.host_context.guest_gpr(GprIndex::A1),
                    }),
                };

                self.set_ecall_result(sbi_ret);
            }
            None => (),
        }
        self.arch.pending_op = None;
    }

    fn save(&mut self) {
        self.active_pages = None;
        self.save_vs_csrs();
        self.pmu().save_counters();
    }

    // Saves the VS-level CSRs.
    fn save_vs_csrs(&mut self) {
        let vs_csrs = &mut self.arch.regs.vs_csrs;
        vs_csrs.htimedelta = CSR.htimedelta.get();
        vs_csrs.vsstatus = CSR.vsstatus.get();
        vs_csrs.vsie = CSR.vsie.get();
        vs_csrs.vstvec = CSR.vstvec.get();
        vs_csrs.vsscratch = CSR.vsscratch.get();
        vs_csrs.vsepc = CSR.vsepc.get();
        vs_csrs.vscause = CSR.vscause.get();
        vs_csrs.vstval = CSR.vstval.get();
        vs_csrs.vsatp = CSR.vsatp.get();
        vs_csrs.vstimecmp = CSR.vstimecmp.get();
    }

    fn restore(&mut self) {
        self.restore_vs_csrs();
        self.restore_vm_pages();
        self.pmu().restore_counters();

        match self.host_context {
            VmCpuParent::HostVm(ref host_vcpu) => {
                // Set our host's timer to the HS-level STIMECMP so that we'll trap if the host's
                // timer expires.
                CSR.stimecmp.set(
                    host_vcpu
                        .vs_csrs()
                        .vstimecmp
                        .wrapping_add(host_vcpu.vs_csrs().htimedelta),
                );

                // If our host has external interrupts enabled, set the bit for their interrupt file
                // in HGEIE so that we trap on any external interrupts they receive.
                let host_sie = LocalRegisterCopy::new(host_vcpu.vs_csrs().vsie);
                if let Some(host_vgein) = host_vcpu.bound_interrupt_file().map(|f| f.bits()) &&
                    host_sie.read(sie::sext) != 0
                {
                    CSR.hgeie.read_and_set_bits(1 << host_vgein);
                }
            }
            VmCpuParent::Tsm(_) => {
                // Clear the bit in HGEIE for our own interrupt file to prevent trapping back into
                // the TSM for own VS external interrupts.
                if let Some(vgein) = self.bound_interrupt_file().map(|f| f.bits()) {
                    CSR.hgeie.read_and_clear_bits(1 << vgein);
                }
            }
        }
    }

    // Restores the VS-level CSRs.
    fn restore_vs_csrs(&mut self) {
        let vs_csrs = &self.arch.regs.vs_csrs;
        // Safe as these don't take effect until V=1.
        CSR.htimedelta.set(vs_csrs.htimedelta);
        CSR.vsstatus.set(vs_csrs.vsstatus);
        CSR.vsie.set(vs_csrs.vsie);
        CSR.vstvec.set(vs_csrs.vstvec);
        CSR.vsscratch.set(vs_csrs.vsscratch);
        CSR.vsepc.set(vs_csrs.vsepc);
        CSR.vscause.set(vs_csrs.vscause);
        CSR.vstval.set(vs_csrs.vstval);
        CSR.vsatp.set(vs_csrs.vsatp);
        CSR.vstimecmp.set(vs_csrs.vstimecmp);
    }

    // Restores the VM's address space.
    fn restore_vm_pages(&mut self) {
        // Get the VMID to use for this VM's address space on this physical CPU.
        let this_cpu = PerCpu::this_cpu();
        let mut vmid_tracker = this_cpu.vmid_tracker_mut();
        // If VMIDs rolled over next_vmid() will do the necessary flush and the previous TLB version
        // doesn't matter.
        let (vmid, tlb_version) = match self.arch.prev_tlb {
            Some(ref prev_tlb) if prev_tlb.vmid.version() == vmid_tracker.current_version() => {
                (prev_tlb.vmid, Some(prev_tlb.tlb_version))
            }
            _ => (vmid_tracker.next_vmid(), None),
        };

        // enter_with_vmid() will fence if prev_tlb_version is stale.
        let active_pages = self.vm_pages.enter_with_vmid(vmid, tlb_version);

        // Record the TLB version we're using.
        self.arch.prev_tlb = Some(PrevTlb {
            cpu: this_cpu.cpu_id(),
            vmid,
            tlb_version: active_pages.tlb_version(),
        });

        self.active_pages = Some(active_pages);
    }
}

impl<T: GuestStagePagingMode> VmCpuExitReporting for ActiveVmCpu<'_, '_, '_, T> {
    fn set_csr(&mut self, csr_num: u16, val: u64) {
        // Supervisor CSRs get written directly to their VS-level equivalent, while HS and VS CSRs
        // are written to the shared memory area.
        match csr_num {
            CSR_SCAUSE => {
                self.arch.regs.vs_csrs.vscause = val;
            }
            CSR_STVAL => {
                self.arch.regs.vs_csrs.vstval = val;
            }
            _ => {
                if let Some(shmem) = self.arch.shmem_area.as_ref().map(|s| s.as_ref()) {
                    shmem.set_csr(csr_num, val);
                }
            }
        }
    }

    fn set_guest_gpr(&mut self, index: GprIndex, val: u64) {
        if let Some(shmem) = self.arch.shmem_area.as_ref().map(|s| s.as_ref()) {
            shmem.set_gpr(index as usize, val);
        }
    }

    fn guest_gpr(&self, index: GprIndex) -> u64 {
        self.arch
            .shmem_area
            .as_ref()
            .map(|s| s.as_ref().gpr(index as usize))
            .unwrap_or(0)
    }
}

impl<T: GuestStagePagingMode> HostVmCpu for ActiveVmCpu<'_, '_, '_, T> {
    fn save(&mut self) {
        self.save();
    }

    fn restore(&mut self) {
        self.restore();
    }

    fn vs_csrs(&self) -> &GuestVsCsrs {
        &self.arch.regs.vs_csrs
    }

    fn virtual_hs_csrs(&self) -> &GuestVirtualHsCsrs {
        &self.arch.regs.virtual_hs_csrs
    }

    fn bound_interrupt_file(&self) -> Option<ImsicFileId> {
        // Read VGEIN from HSTATUS instead of bouncing off the ext_interrupts lock. We must be
        // bound to that interrupt file if we're active.
        let hstatus = LocalRegisterCopy::new(self.arch.regs.guest_regs.hstatus);
        let vgein = hstatus.read(hstatus::vgein) as u32;
        if vgein != 0 {
            Some(ImsicFileId::from_index(vgein))
        } else {
            None
        }
    }
}

impl<T: GuestStagePagingMode> Drop for ActiveVmCpu<'_, '_, '_, T> {
    fn drop(&mut self) {
        // Context-switch back to the host (v)CPU.
        self.save();
        if let VmCpuParent::HostVm(ref mut host_vcpu) = self.host_context {
            host_vcpu.restore();
        }
    }
}

// ActiveVmCpu is used to guard CSR and TLB state which is local to the current physical CPU and
// thus cannot be safely shared between threads.
impl<T: GuestStagePagingMode> !Sync for ActiveVmCpu<'_, '_, '_, T> {}

/// The availability of vCPU in a VM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmCpuStatus {
    /// The vCPU is not powered on.
    PoweredOff,
    /// The vCPU is available to be run.
    Runnable,
    /// The vCPU has been claimed exclusively for running on a (physical) CPU.
    Running,
    /// The vCPU is blocked from running until the VM reaches the given TLB version.
    Blocked(TlbVersion),
}

/// Represents a single virtual CPU of a VM.
pub struct VmCpu {
    // Locking: status -> arch -> ext_interrupts.
    status: RwLock<VmCpuStatus>,
    arch: Mutex<VmCpuArchState>,
    ext_interrupts: Once<Mutex<VmCpuExtInterrupts>>,
    guest_id: PageOwnerId,
    vcpu_id: u64,
}

impl VmCpu {
    /// Creates a new `VmCpu` with ID `vcpu_id`. The returned `VmCpu` is initially powered off.
    pub fn new(vcpu_id: u64, guest_id: PageOwnerId) -> VmCpu {
        VmCpu {
            status: RwLock::new(VmCpuStatus::PoweredOff),
            arch: Mutex::new(VmCpuArchState::new(guest_id)),
            ext_interrupts: Once::new(),
            guest_id,
            vcpu_id,
        }
    }

    /// Powers on this vCPU and sets its entry point to the specified SEPC and A1 values.
    pub fn power_on(&self, sepc: u64, opaque: u64) -> Result<()> {
        let mut status = self.status.write();
        if *status != VmCpuStatus::PoweredOff {
            return Err(Error::VmCpuAlreadyPowered);
        }
        let mut arch = self.arch.lock();
        arch.regs.guest_regs.sepc = sepc;
        arch.regs
            .guest_regs
            .gprs
            .set_reg(GprIndex::A0, self.vcpu_id);
        arch.regs.guest_regs.gprs.set_reg(GprIndex::A1, opaque);
        *status = VmCpuStatus::Runnable;
        Ok(())
    }

    /// Returns the ID of the vCPU in the guest.
    pub fn vcpu_id(&self) -> u64 {
        self.vcpu_id
    }

    /// Returns the runnability status of this vCPU.
    pub fn status(&self) -> VmCpuStatus {
        *self.status.read()
    }

    /// Activates this vCPU, swapping in its register state. Takes exclusive ownership of the
    /// ability to run this vCPU. `host_context` is saved before this vCPU is activated and is restored
    /// when the returned `ActiveVmCpu` is dropped.
    pub fn activate<'vcpu, 'pages, 'host: 'vcpu + 'pages, T: GuestStagePagingMode>(
        &'vcpu self,
        vm_pages: FinalizedVmPages<'pages, T>,
        host_context: VmCpuParent<'host>,
    ) -> Result<ActiveVmCpu<'vcpu, 'pages, 'host, T>> {
        let mut status = self.status.write();
        use VmCpuStatus::*;
        match *status {
            Blocked(tlb_version) if tlb_version > vm_pages.min_tlb_version() => {
                Err(Error::VmCpuBlocked)
            }
            Runnable | Blocked(_) => {
                if self.guest_id != vm_pages.page_owner_id() {
                    return Err(Error::WrongAddressSpace);
                }

                // We must be bound to the current CPU if IMSIC virtualization is enabled.
                if let Some(ext_interrupts) = self.ext_interrupts.get() &&
                    !ext_interrupts.lock().is_bound_on_this_cpu()
                {
                    return Err(Error::VmCpuNotBound);
                }

                let active_vcpu = ActiveVmCpu::restore_from(self, vm_pages, host_context)?;
                *status = Running;
                Ok(active_vcpu)
            }
            Running => Err(Error::VmCpuRunning),
            PoweredOff => Err(Error::VmCpuOff),
        }
    }

    /// Enables IMSIC virtualization for this vCPU, setting the location of the virtualized ISMIC
    /// in guest physical address space.
    pub fn enable_imsic_virtualization(
        &self,
        imsic_location: ImsicLocation,
        num_guests: usize,
    ) -> Result<()> {
        self.ext_interrupts
            .call_once(|| Mutex::new(VmCpuExtInterrupts::new(imsic_location, num_guests)));
        if self.ext_interrupts.get().unwrap().lock().imsic_location() != imsic_location {
            return Err(Error::ImsicLocationAlreadySet);
        }
        Ok(())
    }

    fn ext_interrupts(&self) -> Result<&Mutex<VmCpuExtInterrupts>> {
        self.ext_interrupts
            .get()
            .ok_or(Error::NoImsicVirtualization)
    }

    /// Returns the location of this vCPU's virtualized IMSIC.
    pub fn get_imsic_location(&self) -> Option<ImsicLocation> {
        self.ext_interrupts()
            .ok()
            .map(|ei| ei.lock().imsic_location())
    }

    /// Prepares to bind this vCPU to `interrupt_file` on the current physical CPU.
    pub fn bind_imsic_prepare(&self, interrupt_file: ImsicFileId) -> Result<()> {
        // We skip the self.status check here (and similarly for the other bind/unbind calls)
        // since the only way for the vCPU to be presently running is to be bound to a different
        // physical CPU.
        self.ext_interrupts()?
            .lock()
            .bind_imsic_prepare(interrupt_file)
            .map_err(Error::Binding)
    }

    /// Completes the IMSIC bind operation started in `bind_imsic_prepare()`.
    pub fn bind_imsic_finish(&self) -> Result<()> {
        let interrupt_file = self
            .ext_interrupts()?
            .lock()
            .bind_imsic_finish()
            .map_err(Error::Binding)?;

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut arch = self.arch.lock();
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(arch.regs.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.bits() as u64));
        arch.regs.guest_regs.hstatus = hstatus.get();
        Ok(())
    }

    /// Prepares to rebind this vCPU to `interrupt_file` on the current physical CPU.
    pub fn rebind_imsic_prepare(&self, interrupt_file: ImsicFileId) -> Result<()> {
        self.ext_interrupts()?
            .lock()
            .rebind_imsic_prepare(interrupt_file)
            .map_err(Error::Rebinding)?;
        Ok(())
    }

    /// Copies the guest interrupt file state on the previous CPU.
    pub fn rebind_imsic_clone(&self) -> Result<()> {
        self.ext_interrupts()?
            .lock()
            .rebind_imsic_clone()
            .map_err(Error::Rebinding)?;
        Ok(())
    }

    /// Get the previous guest interrupt file's location.
    pub fn prev_imsic_location(&self) -> Result<ImsicLocation> {
        let prev_imsic_loc = self
            .ext_interrupts()?
            .lock()
            .prev_imsic_location()
            .map_err(Error::Rebinding)?;
        Ok(prev_imsic_loc)
    }

    /// Completes the IMSIC rebind operation started in `rebind_imsic_prepare()`.
    pub fn rebind_imsic_finish(&self) -> Result<()> {
        let interrupt_file = self
            .ext_interrupts()?
            .lock()
            .rebind_imsic_finish()
            .map_err(Error::Rebinding)?;

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut arch = self.arch.lock();
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(arch.regs.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.bits() as u64));
        arch.regs.guest_regs.hstatus = hstatus.get();
        Ok(())
    }

    /// Prepares to unbind this vCPU from its current interrupt file.
    pub fn unbind_imsic_prepare(&self) -> Result<()> {
        self.ext_interrupts()?
            .lock()
            .unbind_imsic_prepare()
            .map_err(Error::Unbinding)
    }

    /// Completes the IMSIC unbind operation started in `unbind_imsic_prepare()`.
    pub fn unbind_imsic_finish(&self) -> Result<()> {
        self.ext_interrupts()?
            .lock()
            .unbind_imsic_finish()
            .map_err(Error::Unbinding)
    }

    /// Injects the specified external interrupt ID into this vCPU, if allowed.
    pub fn inject_ext_interrupt(&self, id: usize) -> Result<()> {
        self.ext_interrupts()?
            .lock()
            .inject_interrupt(id)
            .map_err(Error::InjectingInterrupt)
    }
}

/// The set of vCPUs in a VM.
pub struct VmCpus {
    inner: [Once<PageBox<VmCpu>>; VM_CPUS_MAX],
}

impl VmCpus {
    /// Creates a new vCPU tracking structure.
    pub fn new() -> Self {
        Self {
            inner: [(); VM_CPUS_MAX].map(|_| Once::new()),
        }
    }

    /// Adds the given vCPU to the set of vCPUs.
    pub fn add_vcpu(&self, vcpu: PageBox<VmCpu>) -> Result<()> {
        let vcpu_id = vcpu.vcpu_id();
        let once_entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadVmCpuId)?;

        // CAVEAT: following code is tricky.
        //
        // What we need is a OnceLock or a way to check that we have
        // indeed set the value we passed. Since we don't have this
        // (for now), we determine if the `vcpu` has been set by
        // checking the underlying pointer of the `PageBox`.
        let ptr: *const VmCpu = &*vcpu;
        let once_val = once_entry.call_once(|| vcpu);
        let once_ptr: *const VmCpu = &**once_val;

        if once_ptr != ptr {
            Err(Error::VmCpuExists)
        } else {
            Ok(())
        }
    }

    /// Returns a reference to the vCPU with `vcpu_id` if it exists.
    pub fn get_vcpu(&self, vcpu_id: u64) -> Result<&VmCpu> {
        let vcpu = self
            .inner
            .get(vcpu_id as usize)
            .and_then(|once| once.get())
            .ok_or(Error::VmCpuNotFound)?;
        Ok(vcpu)
    }

    /// Returns the number of pages that must be donated to create a vCPU.
    pub const fn required_state_pages_per_vcpu() -> u64 {
        PageBox::<VmCpu>::required_pages()
    }
}

// Safety: Each VmCpu is wrapped with a Mutex to provide safe concurrent access to VmCpu and its
// composite structures from within the hypervisor, and the VmCpu API provides safe accessors to
// the shared-memory vCPU state structure.
unsafe impl Sync for VmCpus {}
unsafe impl Send for VmCpus {}
