// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::{marker::PhantomData, mem::size_of, ops::Deref, ops::DerefMut, ptr, ptr::NonNull};
use drivers::{imsic::ImsicFileId, imsic::ImsicLocation, CpuId, CpuInfo};
use memoffset::offset_of;
use page_tracking::collections::PageVec;
use page_tracking::{PageTracker, TlbVersion};
use riscv_page_tables::GuestStagePagingMode;
use riscv_pages::{
    GuestPhysAddr, GuestVirtAddr, InternalClean, PageOwnerId, PageSize, RawAddr, SequentialPages,
};
use riscv_regs::*;
use sbi::{self, SbiMessage, SbiReturnType, TvmMmioOpCode};
use spin::{Mutex, Once, RwLock, RwLockReadGuard};

use crate::smp::PerCpu;
use crate::vm::{MmioOperation, VmExitCause};
use crate::vm_id::VmId;
use crate::vm_pages::{ActiveVmPages, FinalizedVmPages, PinnedPages};
use crate::vm_pmu::VmPmuState;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Error {
    BadCpuId,
    VmCpuExists,
    VmCpuNotFound,
    VmCpuRunning,
    VmCpuOff,
    VmCpuAlreadyPowered,
    InsufficientVmCpuStorage,
    WrongAddressSpace,
    InvalidSharedStatePtr,
    InsufficientSharedStatePages,
}

pub type Result<T> = core::result::Result<T, Error>;

/// The number of bytes required to hold the state of a single vCPU. We include the overhead for the
/// `PageVec<>` itself to ensure enough bytes are donated for it as well.
pub const VM_CPU_BYTES: u64 = (size_of::<VmCpusInner>() + size_of::<PageVec<VmCpusInner>>()) as u64;

/// Host GPR and CSR state which must be saved/restored when entering/exiting virtualization.
#[derive(Default)]
#[repr(C)]
struct HostCpuState {
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
struct GuestVCpuState {
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

/// CSRs written on an exit from virtualization that are used by the host to determine the cause of
/// the trap.
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
struct VmCpuState {
    // CPU state that's shared between our's and the guest's execution environment. Saved/restored
    // when entering/exiting a VM.
    host_regs: HostCpuState,
    guest_regs: GuestCpuState,

    // CPU state that only applies when V=1, e.g. the VS-level CSRs. Saved/restored on activation of
    // the vCPU.
    guest_vcpu_csrs: GuestVCpuState,

    // Read on VM exit.
    trap_csrs: VmCpuTrapState,
}

// The vCPU context switch, defined in guest.S
extern "C" {
    fn _run_guest(state: *mut VmCpuState);
    fn _save_fp(state: *mut VmCpuState);
    fn _restore_fp(state: *mut VmCpuState);
}

#[cfg(target_feature = "v")]
extern "C" {
    fn _restore_vector(g: *mut VmCpuState);
    fn _save_vector(g: *mut VmCpuState);
}

#[allow(dead_code)]
const fn host_gpr_offset(index: GprIndex) -> usize {
    offset_of!(VmCpuState, host_regs)
        + offset_of!(HostCpuState, gprs)
        + (index as usize) * size_of::<u64>()
}

#[allow(dead_code)]
const fn guest_gpr_offset(index: GprIndex) -> usize {
    offset_of!(VmCpuState, guest_regs)
        + offset_of!(GuestCpuState, gprs)
        + (index as usize) * size_of::<u64>()
}

const fn guest_fpr_offset(index: usize) -> usize {
    offset_of!(VmCpuState, guest_regs) + offset_of!(GuestCpuState, fprs) + index * size_of::<u64>()
}

#[cfg(target_feature = "v")]
const fn guest_vpr_offset(index: usize) -> usize {
    offset_of!(VmCpuState, guest_regs)
        + offset_of!(GuestCpuState, vprs)
        + index * size_of::<riscv_regs::VectorRegister>()
}

macro_rules! host_csr_offset {
    ($reg:tt) => {
        offset_of!(VmCpuState, host_regs) + offset_of!(HostCpuState, $reg)
    };
}

macro_rules! guest_csr_offset {
    ($reg:tt) => {
        offset_of!(VmCpuState, guest_regs) + offset_of!(GuestCpuState, $reg)
    };
}

global_asm!(
    include_str!("guest.S"),
    host_ra = const host_gpr_offset(GprIndex::RA),
    host_gp = const host_gpr_offset(GprIndex::GP),
    host_tp = const host_gpr_offset(GprIndex::TP),
    host_s0 = const host_gpr_offset(GprIndex::S0),
    host_s1 = const host_gpr_offset(GprIndex::S1),
    host_a1 = const host_gpr_offset(GprIndex::A1),
    host_a2 = const host_gpr_offset(GprIndex::A2),
    host_a3 = const host_gpr_offset(GprIndex::A3),
    host_a4 = const host_gpr_offset(GprIndex::A4),
    host_a5 = const host_gpr_offset(GprIndex::A5),
    host_a6 = const host_gpr_offset(GprIndex::A6),
    host_a7 = const host_gpr_offset(GprIndex::A7),
    host_s2 = const host_gpr_offset(GprIndex::S2),
    host_s3 = const host_gpr_offset(GprIndex::S3),
    host_s4 = const host_gpr_offset(GprIndex::S4),
    host_s5 = const host_gpr_offset(GprIndex::S5),
    host_s6 = const host_gpr_offset(GprIndex::S6),
    host_s7 = const host_gpr_offset(GprIndex::S7),
    host_s8 = const host_gpr_offset(GprIndex::S8),
    host_s9 = const host_gpr_offset(GprIndex::S9),
    host_s10 = const host_gpr_offset(GprIndex::S10),
    host_s11 = const host_gpr_offset(GprIndex::S11),
    host_sp = const host_gpr_offset(GprIndex::SP),
    host_sstatus = const host_csr_offset!(sstatus),
    host_hstatus = const host_csr_offset!(hstatus),
    host_scounteren = const host_csr_offset!(scounteren),
    host_stvec = const host_csr_offset!(stvec),
    host_sscratch = const host_csr_offset!(sscratch),
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
    sstatus_fs_dirty = const sstatus::fs::Dirty.value,
    guest_sstatus = const guest_csr_offset!(sstatus),
    guest_hstatus = const guest_csr_offset!(hstatus),
    guest_scounteren = const guest_csr_offset!(scounteren),
    guest_sepc = const guest_csr_offset!(sepc),
);

#[cfg(target_feature = "v")]
global_asm!(
    include_str!("vectors.S"),
    guest_v0 =     const guest_vpr_offset(0),
    guest_v8 =     const guest_vpr_offset(8),
    guest_v16 =    const guest_vpr_offset(16),
    guest_v24 =    const guest_vpr_offset(24),
    guest_vstart = const guest_csr_offset!(vstart),
    guest_vcsr =   const guest_csr_offset!(vcsr),
    guest_vtype =  const guest_csr_offset!(vtype),
    guest_vl =     const guest_csr_offset!(vl),
    sstatus_vs_enable = const sstatus::vs::Initial.value,
);

/// Defines the structure of the vCPU shared-memory state area.
#[derive(Default)]
pub struct VmCpuSharedState {
    gprs: sbi::Gprs,
    s_csrs: sbi::SupervisorCsrs,
    hs_csrs: sbi::HypervisorCsrs,
}

/// Defines the layout of `VmCpuSharedState` in terms of `RegisterSetLocation`s.
pub const VM_CPU_SHARED_LAYOUT: &[sbi::RegisterSetLocation] = &[
    sbi::RegisterSetLocation {
        id: sbi::RegisterSetId::Gprs as u16,
        version: 0,
        offset: offset_of!(VmCpuSharedState, gprs) as u32,
    },
    sbi::RegisterSetLocation {
        id: sbi::RegisterSetId::SupervisorCsrs as u16,
        version: 0,
        offset: offset_of!(VmCpuSharedState, s_csrs) as u32,
    },
    sbi::RegisterSetLocation {
        id: sbi::RegisterSetId::HypervisorCsrs as u16,
        version: 0,
        offset: offset_of!(VmCpuSharedState, hs_csrs) as u32,
    },
];

/// The number of pages required for `VmCpuSharedState`.
pub const VM_CPU_SHARED_PAGES: u64 = PageSize::num_4k_pages(size_of::<VmCpuSharedState>() as u64);

/// Provides volatile accessors to a `VmCpuSharedState`.
pub struct VmCpuSharedStateRef<'a> {
    ptr: *mut VmCpuSharedState,
    _lifetime: PhantomData<&'a VmCpuSharedState>,
}

// Defines volatile accessors to idividual fields in `VmCpuSharedState`.
macro_rules! define_accessors {
    ($regset:ident, $field:ident, $get:ident, $set:ident) => {
        /// Gets $field in the shared-memory state area.
        #[allow(dead_code)]
        pub fn $get(&self) -> u64 {
            // Safety: The caller guaranteed at construction that `ptr` points to a valid
            // VmCpuSharedState.
            unsafe { ptr::addr_of!((*self.ptr).$regset.$field).read_volatile() }
        }

        /// Sets $field in the shared-memory state area.
        #[allow(dead_code)]
        pub fn $set(&self, val: u64) {
            // Safety: The caller guaranteed at construction that `ptr` points to a valid
            // VmCpuSharedState.
            unsafe { ptr::addr_of_mut!((*self.ptr).$regset.$field).write_volatile(val) };
        }
    };
}

impl<'a> VmCpuSharedStateRef<'a> {
    /// Creates a new `VvmCpuSharedStateRef` from a raw pointer to a `VmCpuSharedState`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is suitably aligned and points to a `TvmCpuSharedState`
    /// structure that is valid for the lifetime `'a`.
    pub unsafe fn new(ptr: *mut VmCpuSharedState) -> Self {
        Self {
            ptr,
            _lifetime: PhantomData,
        }
    }

    define_accessors! {s_csrs, sepc, sepc, set_sepc}
    define_accessors! {s_csrs, scause, scause, set_scause}
    define_accessors! {s_csrs, stval, stval, set_stval}
    define_accessors! {hs_csrs, htval, htval, set_htval}
    define_accessors! {hs_csrs, htinst, htinst, set_htinst}

    /// Reads the general purpose register at `index`.
    pub fn gpr(&self, index: GprIndex) -> u64 {
        // Safety: `index` is guaranteed to be a valid GPR index and the caller guaranteed at
        // construction that `ptr` points to a valid `VmCpuSharedState`.
        unsafe { ptr::addr_of!((*self.ptr).gprs.0[index as usize]).read_volatile() }
    }

    /// Writes the general purpose register at `index`.
    pub fn set_gpr(&self, index: GprIndex, val: u64) {
        // Safety: `index` is guaranteed to be a valid GPR index and the caller guaranteed at
        // construction that `ptr` points to a valid `VmCpuSharedState`.
        unsafe { ptr::addr_of_mut!((*self.ptr).gprs.0[index as usize]).write_volatile(val) };
    }
}

/// Wrapper for the shared-memory state area used to communicate vCPU state with the host.
pub struct VmCpuSharedArea {
    ptr: NonNull<VmCpuSharedState>,
    // Optional since we might be sharing with the hypervisor in the host VM case.
    _pin: Option<PinnedPages>,
}

impl VmCpuSharedArea {
    /// Creates a new `VmCpuSharedArea` using the `VmCpuSharedState` struct referred to by `ptr`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is suitably aligned, points to an allocated block of
    /// memory that is at least as large as `VmCpuSharedState`, and is valid for reads and writes
    /// for the lifetime of this structure. The caller must only access the memory referred to
    /// by `ptr` using volatile accessors.
    pub unsafe fn new(ptr: *mut VmCpuSharedState) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::InvalidSharedStatePtr)?,
            _pin: None,
        })
    }

    /// Creates a new `VmCpuSharedArea` using the pinned pages referred to by `pages`.
    pub fn from_pinned_pages(pages: PinnedPages) -> Result<Self> {
        // Make sure the pin actually covers the size of the structure.
        if pages.range().num_pages() < VM_CPU_SHARED_PAGES {
            return Err(Error::InsufficientSharedStatePages);
        }
        let ptr = pages.range().base().bits() as *mut VmCpuSharedState;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::InvalidSharedStatePtr)?,
            _pin: Some(pages),
        })
    }

    // Updates the shared state buffer for an ECALL exit from the given SBI message.
    fn update_with_ecall_exit(&self, msg: SbiMessage) {
        let shared = self.as_ref();
        shared.set_gpr(GprIndex::A0, msg.a0());
        shared.set_gpr(GprIndex::A1, msg.a1());
        shared.set_gpr(GprIndex::A2, msg.a2());
        shared.set_gpr(GprIndex::A3, msg.a3());
        shared.set_gpr(GprIndex::A4, msg.a4());
        shared.set_gpr(GprIndex::A5, msg.a5());
        shared.set_gpr(GprIndex::A6, msg.a6());
        shared.set_gpr(GprIndex::A7, msg.a7());
        shared.set_stval(0);
        shared.set_htval(0);
        shared.set_htinst(0);
        shared.set_scause(Exception::VirtualSupervisorEnvCall as u64);
    }

    // Updates the shared state buffer for a guest page fault exit at the given address.
    fn update_with_pf_exit(&self, exception: Exception, addr: GuestPhysAddr) {
        let shared = self.as_ref();
        shared.set_stval(addr.bits() & 0x3);
        shared.set_htval(addr.bits() >> 2);
        shared.set_htinst(0);
        shared.set_scause(exception as u64);
    }

    // Updates the shared state buffer for a virtual instruction exception caused by the given
    // instruction.
    fn update_with_vi_exit(&self, inst: u64) {
        let shared = self.as_ref();
        // Note that this is technically not spec-compliant as the privileged spec only states
        // that illegal instruction exceptions may write the faulting instruction to *TVAL CSRs.
        shared.set_stval(inst);
        shared.set_htval(0);
        shared.set_htinst(0);
        shared.set_scause(Exception::VirtualInstruction as u64);
    }

    // Updates the shared state buffer for an unhandled exit due to `scause`.
    fn update_with_unhandled_exit(&self, scause: u64) {
        let shared = self.as_ref();
        shared.set_stval(0);
        shared.set_htval(0);
        shared.set_htinst(0);
        shared.set_scause(scause);
    }

    // Returns the wrapped shared state buffer as a `VmCpuSharedRef`.
    fn as_ref(&self) -> VmCpuSharedStateRef {
        // Safety: We've validated at construction that self.ptr points to a valid
        // `VmCpuSharedState`.
        unsafe { VmCpuSharedStateRef::new(self.ptr.as_ptr()) }
    }
}

/// Identifies the exit cause for a vCPU.
pub enum VmCpuExit {
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
    /// Everything else that we currently don't or can't handle.
    Other(VmCpuTrapState),
    // TODO: Add other exit causes as needed.
}

// Interface for vCPU state save/restore on context switch.
trait VmCpuSaveState {
    // Saves this `ActiveVmCpu`'s state, effectively de-activating it until a corresponding call to
    // `restore()`.
    fn save(&mut self);

    // Re-activates this `ActiveVmCpu` by restoring the state that was saved with `save()`.
    fn restore(&mut self);
}

/// An activated vCPU. A vCPU in this state has entered the VM's address space and is ready to run.
pub struct ActiveVmCpu<'vcpu, 'pages, 'prev, T: GuestStagePagingMode> {
    vcpu: &'vcpu mut VmCpu,
    vm_pages: FinalizedVmPages<'pages, T>,
    // `None` if this vCPU is itself running a child vCPU. Restored when the child vCPU exits.
    active_pages: Option<ActiveVmPages<'pages, T>>,
    // The parent vCPU which activated us.
    parent_vcpu: Option<&'prev mut dyn VmCpuSaveState>,
}

#[cfg(target_feature = "v")]
fn restore_vector(state: *mut VmCpuState) {
    unsafe {
        // Safe because the only memory it touches is known offsets of state
        _restore_vector(state);
    }
}

#[cfg(target_feature = "v")]
fn save_vector(state: *mut VmCpuState) {
    unsafe {
        // Safe because the only memory it touches is known offsets of state
        _save_vector(state);
    }
}

#[cfg(not(target_feature = "v"))]
fn restore_vector(_state: *mut VmCpuState) {}
#[cfg(not(target_feature = "v"))]
fn save_vector(_state: *mut VmCpuState) {}

impl<'vcpu, 'pages, 'prev, T: GuestStagePagingMode> ActiveVmCpu<'vcpu, 'pages, 'prev, T> {
    // Restores and activates the vCPU state from `vcpu`, with the VM address space represented by
    // `vm_pages`.
    fn restore_from(
        vcpu: &'vcpu mut VmCpu,
        vm_pages: FinalizedVmPages<'pages, T>,
        parent_vcpu: Option<&'prev mut ActiveVmCpu<T>>,
    ) -> Self {
        let this_cpu = PerCpu::this_cpu();
        if let Some(ref c) = vcpu.current_cpu && c.cpu != this_cpu.cpu_id() {
            // If we've changed CPUs, then any per-CPU state is invalid.
            //
            // TODO: Migration between physical CPUs needs to be done explicitly via TEECALL.
            vcpu.current_cpu = None;
        }

        // We store the parent vCPU as a trait object as a means of type-erasure for ActiveVmCpu's
        // inner lifetimes.
        let parent_vcpu = parent_vcpu.map(|p| p as &mut dyn VmCpuSaveState);

        let mut active_vcpu = Self {
            vcpu,
            vm_pages,
            active_pages: None,
            parent_vcpu,
        };
        active_vcpu.restore();
        active_vcpu
    }

    /// Runs this vCPU until it exits.
    pub fn run_to_exit(&mut self) -> VmCpuExit {
        self.complete_pending_mmio_op();

        // TODO, HGEIE programinng:
        //  - Track which guests the host wants interrupts from (by trapping HGEIE accesses from
        //    VS level) and update HGEIE[2:] appropriately.
        //  - If this is the host: clear HGEIE[1] on entry; inject SGEI into host VM if we receive
        //    any SGEI at HS level.
        //  - If this is a guest: set HGEIE[1] on entry; switch to the host VM for any SGEI that
        //    occur, injecting an SEI for the host interrupts and SGEI for guest VM interrupts.

        // TODO: Enforce that the vCPU has an assigned interrupt file before running.

        let has_vector = CpuInfo::get().has_vector();

        if has_vector {
            restore_vector(&mut self.state);
        }

        unsafe {
            // Safe since _restore_fp() only reads within the bounds of the floating point
            // register state in VmCpuState.
            _restore_fp(&mut self.state);

            // Safe to run the guest as it only touches memory assigned to it by being owned
            // by its page table.
            _run_guest(&mut self.state);
        }

        // Save off the trap information.
        self.state.trap_csrs.scause = CSR.scause.get();
        self.state.trap_csrs.stval = CSR.stval.get();
        self.state.trap_csrs.htval = CSR.htval.get();
        self.state.trap_csrs.htinst = CSR.htinst.get();

        // Check if FPU state needs to be saved.
        let mut sstatus = LocalRegisterCopy::new(self.state.guest_regs.sstatus);
        if sstatus.matches_all(sstatus::fs::Dirty) {
            // Safe since _save_fp() only writes within the bounds of the floating point register
            // state in VmCpuState.
            unsafe { _save_fp(&mut self.state) };

            sstatus.modify(sstatus::fs::Clean);
            self.state.guest_regs.sstatus = sstatus.get();
        }

        // Check if vector state needs to be saved
        if has_vector && sstatus.matches_all(sstatus::vs::Dirty) {
            save_vector(&mut self.state);
            sstatus.modify(sstatus::vs::Clean)
        }

        // Determine the exit cause from the trap CSRs.
        use Exception::*;
        match Trap::from_scause(self.state.trap_csrs.scause).unwrap() {
            Trap::Exception(VirtualSupervisorEnvCall) => {
                let sbi_msg = SbiMessage::from_regs(self.state.guest_regs.gprs.a_regs()).ok();
                VmCpuExit::Ecall(sbi_msg)
            }
            Trap::Exception(GuestInstructionPageFault)
            | Trap::Exception(GuestLoadPageFault)
            | Trap::Exception(GuestStorePageFault) => {
                let fault_addr = RawAddr::guest(
                    self.state.trap_csrs.htval << 2 | self.state.trap_csrs.stval & 0x03,
                    self.guest_id,
                );
                VmCpuExit::PageFault {
                    exception: Exception::from_scause_reason(self.state.trap_csrs.scause).unwrap(),
                    fault_addr,
                    // Note that this address is not necessarily guest virtual as the guest may or
                    // may not have 1st-stage translation enabled in VSATP. We still use GuestVirtAddr
                    // here though to distinguish it from addresses (e.g. in HTVAL, or passed via a
                    // TEECALL) which are exclusively guest-physical. Furthermore we only access guest
                    // instructions via the HLVX instruction, which will take the VSATP translation
                    // mode into account.
                    fault_pc: RawAddr::guest_virt(self.state.guest_regs.sepc, self.guest_id),
                    priv_level: PrivilegeLevel::from_hstatus(self.state.guest_regs.hstatus),
                }
            }
            Trap::Exception(VirtualInstruction) => {
                VmCpuExit::VirtualInstruction {
                    // See above re: this address being guest virtual.
                    fault_pc: RawAddr::guest_virt(self.state.guest_regs.sepc, self.guest_id),
                    priv_level: PrivilegeLevel::from_hstatus(self.state.guest_regs.hstatus),
                }
            }
            Trap::Exception(e) => {
                if e.to_hedeleg_field()
                    .map_or(false, |f| CSR.hedeleg.matches_any(f))
                {
                    // Even if we intended to delegate this exception it might not be set in
                    // medeleg, in which case firmware may send it our way instead.
                    VmCpuExit::DelegatedException {
                        exception: e,
                        stval: self.state.trap_csrs.stval,
                    }
                } else {
                    VmCpuExit::Other(self.state.trap_csrs.clone())
                }
            }
            _ => VmCpuExit::Other(self.state.trap_csrs.clone()),
        }
    }

    // Rewrites `mmio_op` as a transformed load or store instruction to/from A0 as would be written
    // to the HTINST CSR.
    fn mmio_op_to_htinst(mmio_op: MmioOperation) -> u64 {
        use TvmMmioOpCode::*;
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
        let htinst = match mmio_op.opcode() {
            Store8 | Store16 | Store32 | Store64 => htinst_base | ((GprIndex::A0 as u32) << 7),
            Load8 | Load16 | Load32 | Load8U | Load16U | Load32U | Load64 => {
                htinst_base | ((GprIndex::A0 as u32) << 20)
            }
        };
        htinst as u64
    }

    /// Sets up this vCPU's shared-memory state and virtual registers for reporting `exit` back
    /// to this vCPU's host.
    pub fn set_exit_cause(&mut self, exit: VmExitCause) {
        use VmExitCause::*;
        match exit {
            PowerOff(reset_type, reason) => {
                let msg = SbiMessage::Reset(sbi::ResetFunction::Reset { reset_type, reason });
                self.shared_area().update_with_ecall_exit(msg);
            }
            CpuStart(hart_id) => {
                let msg = SbiMessage::HartState(sbi::StateFunction::HartStart {
                    hart_id,
                    start_addr: 0,
                    opaque: 0,
                });
                self.shared_area().update_with_ecall_exit(msg);
            }
            CpuStop => {
                let msg = SbiMessage::HartState(sbi::StateFunction::HartStop);
                self.shared_area().update_with_ecall_exit(msg);
            }
            ConfidentialPageFault(exception, page_addr) => {
                self.shared_area()
                    .update_with_pf_exit(exception, page_addr.into());
            }
            SharedPageFault(exception, page_addr) => {
                self.shared_area()
                    .update_with_pf_exit(exception, page_addr.into());
            }
            MmioPageFault(exception, addr, mmio_op) => {
                self.shared_area().update_with_pf_exit(exception, addr);

                // The MMIO instruction is transformed as an ordinary load/store to/from A0, so
                // update A0 with the value the vCPU wants to store.
                use TvmMmioOpCode::*;
                let val = match mmio_op.opcode() {
                    Store8 => self.get_gpr(mmio_op.register()) as u8 as u64,
                    Store16 => self.get_gpr(mmio_op.register()) as u16 as u64,
                    Store32 => self.get_gpr(mmio_op.register()) as u32 as u64,
                    Store64 => self.get_gpr(mmio_op.register()),
                    _ => 0,
                };
                let shared = self.shared_area().as_ref();
                shared.set_htinst(Self::mmio_op_to_htinst(mmio_op));
                shared.set_gpr(GprIndex::A0, val);

                // We'll complete a load instruction the next time this vCPU is run.
                self.pending_mmio_op = Some(mmio_op);
            }
            Wfi(inst) => {
                self.shared_area().update_with_vi_exit(inst.raw() as u64);
            }
            UnhandledTrap(scause) => {
                self.shared_area().update_with_unhandled_exit(scause);
            }
        };

        if let Some(val) = exit.cause0() {
            self.set_virt_reg(VirtualRegister::Cause0, val);
        }
        if let Some(val) = exit.cause1() {
            self.set_virt_reg(VirtualRegister::Cause1, val);
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
            LocalRegisterCopy::<u64, sstatus::Register>::new(self.state.guest_regs.sstatus);
        vsstatus.modify(sstatus::spp.val(sstatus.read(sstatus::spp)));
        CSR.vsstatus.set(vsstatus.get());

        CSR.vscause.set(Trap::Exception(exception).to_scause());
        CSR.vstval.set(stval);
        CSR.vsepc.set(self.state.guest_regs.sepc);

        // Redirect the vCPU to its STVEC on entry.
        self.state.guest_regs.sepc = CSR.vstvec.get();
    }

    /// Increments the current `sepc` CSR value by `value`.
    pub fn inc_sepc(&mut self, value: u64) {
        self.state.guest_regs.sepc += value;
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
                self.set_gpr(GprIndex::A1, ret.return_value as u64);
            }
        }
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

    // Completes any pending MMIO operation for this CPU.
    fn complete_pending_mmio_op(&mut self) {
        // Complete any pending load operations. The host is expected to have written the value
        // to complete the load to A0.
        if let Some(mmio_op) = self.pending_mmio_op {
            let val = self.shared_area().as_ref().gpr(GprIndex::A0);
            use TvmMmioOpCode::*;
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

            self.pending_mmio_op = None;
            self.shared_area().as_ref().set_gpr(GprIndex::A0, 0);

            // Advance SEPC past the faulting instruction.
            self.inc_sepc(mmio_op.len() as u64);
        }
    }

    // Saves the VS-level CSRs.
    fn save_vcpu_csrs(&mut self) {
        self.state.guest_vcpu_csrs.htimedelta = CSR.htimedelta.get();
        self.state.guest_vcpu_csrs.vsstatus = CSR.vsstatus.get();
        self.state.guest_vcpu_csrs.vsie = CSR.vsie.get();
        self.state.guest_vcpu_csrs.vstvec = CSR.vstvec.get();
        self.state.guest_vcpu_csrs.vsscratch = CSR.vsscratch.get();
        self.state.guest_vcpu_csrs.vsepc = CSR.vsepc.get();
        self.state.guest_vcpu_csrs.vscause = CSR.vscause.get();
        self.state.guest_vcpu_csrs.vstval = CSR.vstval.get();
        self.state.guest_vcpu_csrs.vsatp = CSR.vsatp.get();
        if CpuInfo::get().has_sstc() {
            self.state.guest_vcpu_csrs.vstimecmp = CSR.vstimecmp.get();
        }
    }

    // Restores the VS-level CSRs.
    fn restore_vcpu_csrs(&mut self) {
        // Safe as these don't take effect until V=1.
        CSR.htimedelta.set(self.state.guest_vcpu_csrs.htimedelta);
        CSR.vsstatus.set(self.state.guest_vcpu_csrs.vsstatus);
        CSR.vsie.set(self.state.guest_vcpu_csrs.vsie);
        CSR.vstvec.set(self.state.guest_vcpu_csrs.vstvec);
        CSR.vsscratch.set(self.state.guest_vcpu_csrs.vsscratch);
        CSR.vsepc.set(self.state.guest_vcpu_csrs.vsepc);
        CSR.vscause.set(self.state.guest_vcpu_csrs.vscause);
        CSR.vstval.set(self.state.guest_vcpu_csrs.vstval);
        CSR.vsatp.set(self.state.guest_vcpu_csrs.vsatp);
        if CpuInfo::get().has_sstc() {
            CSR.vstimecmp.set(self.state.guest_vcpu_csrs.vstimecmp);
        }
    }

    // Restores the VM's address space.
    fn restore_vm_pages(&mut self) {
        // Get the VMID to use for this VM's address space on this physical CPU.
        let this_cpu = PerCpu::this_cpu();
        let mut vmid_tracker = this_cpu.vmid_tracker_mut();
        // If VMIDs rolled over next_vmid() will do the necessary flush and the previous TLB version
        // doesn't matter.
        let (vmid, prev_tlb_version) = match self.current_cpu {
            Some(ref c) if c.vmid.version() == vmid_tracker.current_version() => {
                (c.vmid, Some(c.tlb_version))
            }
            _ => (vmid_tracker.next_vmid(), None),
        };

        // enter_with_vmid() will fence if prev_tlb_version is stale.
        let active_pages = self.vm_pages.enter_with_vmid(vmid, prev_tlb_version);

        // Update our per-CPU state in case VMID or TLB version changed.
        if let Some(ref mut c) = self.current_cpu {
            c.vmid = vmid;
            c.tlb_version = active_pages.tlb_version();
        } else {
            self.current_cpu = Some(CurrentCpu {
                cpu: this_cpu.cpu_id(),
                vmid,
                tlb_version: active_pages.tlb_version(),
            });
        }

        self.active_pages = Some(active_pages);
    }
}

impl<T: GuestStagePagingMode> VmCpuSaveState for ActiveVmCpu<'_, '_, '_, T> {
    fn save(&mut self) {
        self.active_pages = None;
        self.save_vcpu_csrs();
        self.pmu_state.save_counters();
    }

    fn restore(&mut self) {
        self.restore_vcpu_csrs();
        self.restore_vm_pages();
        self.pmu_state.restore_counters();
    }
}

impl<T: GuestStagePagingMode> Deref for ActiveVmCpu<'_, '_, '_, T> {
    type Target = VmCpu;

    fn deref(&self) -> &VmCpu {
        self.vcpu
    }
}

impl<T: GuestStagePagingMode> DerefMut for ActiveVmCpu<'_, '_, '_, T> {
    fn deref_mut(&mut self) -> &mut VmCpu {
        self.vcpu
    }
}

impl<T: GuestStagePagingMode> Drop for ActiveVmCpu<'_, '_, '_, T> {
    fn drop(&mut self) {
        self.save();
        if let Some(ref mut p) = self.parent_vcpu {
            p.restore();
        }
    }
}

/// Used to store any per-physical-CPU state for a virtual CPU of a VM.
struct CurrentCpu {
    cpu: CpuId,
    vmid: VmId,
    tlb_version: TlbVersion,
}

/// Virtual CPU registers that are used to store vCPU state accessible to the VM's host without
/// giving the host access to internal register state.
pub enum VirtualRegister {
    /// 1st detailed exit cause register. Usage depends on the exit code.
    Cause0,
    /// 2nd detailed exit cause register. Usage depends on the exit code.
    Cause1,
    /// Result of an emulated MMIO load.
    MmioLoad,
    /// Source value of an emulated MMIO store.
    MmioStore,
}

/// Virtual register state of a vCPU.
#[derive(Default)]
struct VirtualRegisters {
    cause0: u64,
    cause1: u64,
}

/// Represents a single virtual CPU of a VM.
pub struct VmCpu {
    state: VmCpuState,
    // Initialized in add_vcpu().
    shared_area: Once<VmCpuSharedArea>,
    virt_regs: VirtualRegisters,
    imsic_location: Option<ImsicLocation>,
    pmu_state: VmPmuState,
    current_cpu: Option<CurrentCpu>,
    // TODO: interrupt_file should really be part of CurrentCpu, but we have no way to migrate it
    // at present.
    interrupt_file: Option<ImsicFileId>,
    pending_mmio_op: Option<MmioOperation>,
    guest_id: PageOwnerId,
}

impl VmCpu {
    /// Creates a new vCPU.
    pub fn new(guest_id: PageOwnerId) -> Self {
        let mut state = VmCpuState::default();

        let mut hstatus = LocalRegisterCopy::<u64, hstatus::Register>::new(0);
        hstatus.modify(hstatus::spv.val(1));
        hstatus.modify(hstatus::spvp::Supervisor);
        if !guest_id.is_host() {
            // Trap on WFI for non-host VMs. Trapping WFI for the host is pointless since all we'd do
            // in the hypervisor is WFI ourselves.
            hstatus.modify(hstatus::vtw.val(1));
        }
        state.guest_regs.hstatus = hstatus.get();

        let mut sstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(0);
        sstatus.modify(sstatus::spp::Supervisor);
        sstatus.modify(sstatus::fs::Initial);
        #[cfg(target_feature = "v")]
        sstatus.modify(sstatus::vs::Initial);
        state.guest_regs.sstatus = sstatus.get();

        let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
        scounteren.modify(scounteren::cycle.val(1));
        scounteren.modify(scounteren::time.val(1));
        scounteren.modify(scounteren::instret.val(1));
        state.guest_regs.scounteren = scounteren.get();

        Self {
            state,
            shared_area: Once::new(),
            virt_regs: VirtualRegisters::default(),
            imsic_location: None,
            current_cpu: None,
            pending_mmio_op: None,
            interrupt_file: None,
            guest_id,
            pmu_state: VmPmuState::default(),
        }
    }

    /// Sets the `sepc` CSR, or the PC value the vCPU will jump to when it is run.
    pub fn set_sepc(&mut self, sepc: u64) {
        self.state.guest_regs.sepc = sepc;
    }

    /// Sets one of the vCPU's general-purpose registers.
    pub fn set_gpr(&mut self, gpr: GprIndex, value: u64) {
        self.state.guest_regs.gprs.set_reg(gpr, value);
    }

    /// Gets one of the vCPU's general purpose registers.
    pub fn get_gpr(&mut self, gpr: GprIndex) -> u64 {
        self.state.guest_regs.gprs.reg(gpr)
    }

    /// Sets the initial SEPC value in the vCPU's shared-state buffer.
    pub fn set_entry_sepc(&mut self, sepc: u64) {
        self.shared_area().as_ref().set_sepc(sepc);
    }

    /// Gets the initial SEPC value in the vCPU's shared-state buffer.
    pub fn get_entry_sepc(&self) -> u64 {
        self.shared_area().as_ref().sepc()
    }

    /// Sets the initial opaque boot argument (A1) in the vCPU's shared-state buffer.
    pub fn set_entry_arg(&mut self, arg: u64) {
        self.shared_area().as_ref().set_gpr(GprIndex::A1, arg);
    }

    /// Gets the initial opaque boot argument (A1) in the vCPU's shared-state buffer.
    pub fn get_entry_arg(&self) -> u64 {
        self.shared_area().as_ref().gpr(GprIndex::A1)
    }

    /// Latches the entry point of this vCPU from the shared-memory state buffer, returning the
    /// (SEPC, A1) pair. Should only be called for the boot vCPU.
    pub fn latch_entry_args(&mut self) -> (u64, u64) {
        let sepc = self.get_entry_sepc();
        let arg = self.get_entry_arg();
        self.set_sepc(sepc);
        self.set_gpr(GprIndex::A1, arg);
        (sepc, arg)
    }

    /// Set one of the vCPU's virtual registers.
    pub fn set_virt_reg(&mut self, reg: VirtualRegister, value: u64) {
        use VirtualRegister::*;
        match reg {
            Cause0 => {
                self.virt_regs.cause0 = value;
            }
            Cause1 => {
                self.virt_regs.cause1 = value;
            }
            MmioLoad | MmioStore => {
                // MMIO loads/stores are always from A0 in the shared-state buffer.
                self.shared_area().as_ref().set_gpr(GprIndex::A0, value);
            }
        }
    }

    /// Gets one of the vCPU's virtual registers.
    pub fn get_virt_reg(&mut self, reg: VirtualRegister) -> u64 {
        use VirtualRegister::*;
        match reg {
            Cause0 => self.virt_regs.cause0,
            Cause1 => self.virt_regs.cause1,
            MmioLoad | MmioStore => {
                // MMIO loads/stores are always from A0 in the shared-state buffer.
                self.shared_area().as_ref().gpr(GprIndex::A0)
            }
        }
    }

    /// Sets the location of this vCPU's virtualized IMSIC.
    pub fn set_imsic_location(&mut self, imsic_location: ImsicLocation) {
        self.imsic_location = Some(imsic_location);
    }

    /// Returns the location of this vCPU's virtualized IMSIC.
    pub fn get_imsic_location(&self) -> Option<ImsicLocation> {
        self.imsic_location
    }

    /// Sets the interrupt file for this vCPU.
    pub fn set_interrupt_file(&mut self, interrupt_file: ImsicFileId) {
        self.interrupt_file = Some(interrupt_file);

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(self.state.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.bits() as u64));
        self.state.guest_regs.hstatus = hstatus.get();
    }

    /// Activates `vcpu` with the VM address space in `vm_pages`, returning a reference to it as an
    /// `ActiveVmCpu`. If `parent_vcpu` is not `None`, its state is saved before this vCPU is
    /// activated and is restored when the returned `ActiveVmCpu` is dropped.
    pub fn activate<'vcpu, 'pages, 'prev: 'vcpu + 'pages, T: GuestStagePagingMode>(
        &'vcpu mut self,
        vm_pages: FinalizedVmPages<'pages, T>,
        mut parent_vcpu: Option<&'prev mut ActiveVmCpu<T>>,
    ) -> Result<ActiveVmCpu<'vcpu, 'pages, 'prev, T>> {
        if self.guest_id != vm_pages.page_owner_id() {
            return Err(Error::WrongAddressSpace);
        }

        if let Some(ref mut p) = parent_vcpu {
            p.save();
        }

        Ok(ActiveVmCpu::restore_from(self, vm_pages, parent_vcpu))
    }

    pub fn pmu(&mut self) -> &mut VmPmuState {
        &mut self.pmu_state
    }

    // Returns a reference to the shared-memory state area for this vCPU.
    fn shared_area(&self) -> &VmCpuSharedArea {
        // Unwrap ok: shared_area must've been initialized for this vCPU to have been activated.
        self.shared_area.get().unwrap()
    }
}

/// Represents the state of a vCPU in a VM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmCpuStatus {
    /// There is no vCPU with this ID in the VM.
    NotPresent,
    /// The vCPU is present, but not powered on.
    PoweredOff,
    /// The vCPU is available to be run.
    Runnable,
    /// The vCPU has been claimed exclusively for running on a (physical) CPU.
    Running,
}

struct VmCpusInner {
    // Locking: status must be locked before vcpu.
    status: RwLock<VmCpuStatus>,
    vcpu: Mutex<VmCpu>,
}

/// A reference to an "Available" (idle) `VmCpu`. The `VmCpu` is guaranteed not to change states
/// while this reference is held.
pub struct IdleVmCpu<'a> {
    _status: RwLockReadGuard<'a, VmCpuStatus>,
    vcpu: &'a Mutex<VmCpu>,
}

impl<'a> Deref for IdleVmCpu<'a> {
    type Target = Mutex<VmCpu>;

    fn deref(&self) -> &Mutex<VmCpu> {
        self.vcpu
    }
}

/// A reference to an exclusively-owned `VmCpu` in the "Running" state. The `VmCpu` transitions
/// back to idle when this reference is dropped.
pub struct RunningVmCpu<'a> {
    parent: &'a VmCpus,
    vcpu: &'a Mutex<VmCpu>,
    id: u64,
    power_off: bool,
}

impl<'a> RunningVmCpu<'a> {
    /// Mark this vCPU as powered off when it is returned.
    pub fn power_off(&mut self) {
        self.power_off = true;
    }
}

impl<'a> Deref for RunningVmCpu<'a> {
    type Target = Mutex<VmCpu>;

    fn deref(&self) -> &Mutex<VmCpu> {
        self.vcpu
    }
}

impl<'a> Drop for RunningVmCpu<'a> {
    fn drop(&mut self) {
        let entry = self.parent.inner.get(self.id as usize).unwrap();
        let mut status = entry.status.write();
        assert_eq!(*status, VmCpuStatus::Running);
        *status = if self.power_off {
            VmCpuStatus::PoweredOff
        } else {
            VmCpuStatus::Runnable
        };
    }
}

/// The set of vCPUs in a VM.
pub struct VmCpus {
    inner: PageVec<VmCpusInner>,
}

impl VmCpus {
    /// Creates a new vCPU tracking structure backed by `pages`.
    pub fn new(
        guest_id: PageOwnerId,
        pages: SequentialPages<InternalClean>,
        page_tracker: PageTracker,
    ) -> Result<Self> {
        let num_vcpus = pages.length_bytes() / VM_CPU_BYTES;
        if num_vcpus == 0 {
            return Err(Error::InsufficientVmCpuStorage);
        }
        let mut inner = PageVec::new(pages, page_tracker);
        for _ in 0..num_vcpus {
            let entry = VmCpusInner {
                status: RwLock::new(VmCpuStatus::NotPresent),
                vcpu: Mutex::new(VmCpu::new(guest_id)),
            };
            inner.push(entry);
        }
        Ok(Self { inner })
    }

    /// Returns the number of vCPUs in this `VmCpus`.
    pub fn num_vcpus(&self) -> usize {
        self.inner.len()
    }

    /// Adds the vCPU at `vcpu_id` as an available vCPU using `shared_area` as the vCPU's shared
    /// state-memory state area, returning a reference to it.
    pub fn add_vcpu(&self, vcpu_id: u64, shared_area: VmCpuSharedArea) -> Result<IdleVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let mut status = entry.status.write();
        if *status != VmCpuStatus::NotPresent {
            return Err(Error::VmCpuExists);
        }
        entry.vcpu.lock().shared_area.call_once(|| shared_area);
        *status = VmCpuStatus::PoweredOff;
        Ok(IdleVmCpu {
            _status: status.downgrade(),
            vcpu: &entry.vcpu,
        })
    }

    /// Returns a reference to the vCPU with `vcpu_id` if it exists and is not currently running.
    /// The returned `IdleVmCpu` is guaranteed not to change state until it is dropped.
    pub fn get_vcpu(&self, vcpu_id: u64) -> Result<IdleVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let status = entry.status.read();
        match *status {
            VmCpuStatus::PoweredOff | VmCpuStatus::Runnable => Ok(IdleVmCpu {
                _status: status,
                vcpu: &entry.vcpu,
            }),
            VmCpuStatus::Running => Err(Error::VmCpuRunning),
            VmCpuStatus::NotPresent => Err(Error::VmCpuNotFound),
        }
    }

    /// Marks the vCPU with `vcpu_id` as powered on and runnable and returns a reference to it.
    /// The returned `IdleVmCpu` is guaranteed not to change state until it is dropped.
    pub fn power_on_vcpu(&self, vcpu_id: u64) -> Result<IdleVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let mut status = entry.status.write();
        match *status {
            VmCpuStatus::PoweredOff => {
                *status = VmCpuStatus::Runnable;
                Ok(IdleVmCpu {
                    _status: status.downgrade(),
                    vcpu: &entry.vcpu,
                })
            }
            VmCpuStatus::Running | VmCpuStatus::Runnable => Err(Error::VmCpuAlreadyPowered),
            VmCpuStatus::NotPresent => Err(Error::VmCpuNotFound),
        }
    }

    /// Takes exclusive ownership of the vCPU with `vcpu_id`, marking it as running. The vCPU is
    /// returned to the "Available" state when the returned `RunningVmCpu` is dropped.
    pub fn take_vcpu(&self, vcpu_id: u64) -> Result<RunningVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let mut status = entry.status.write();
        match *status {
            VmCpuStatus::Runnable => {
                *status = VmCpuStatus::Running;
                Ok(RunningVmCpu {
                    parent: self,
                    vcpu: &entry.vcpu,
                    id: vcpu_id,
                    power_off: false,
                })
            }
            VmCpuStatus::Running => Err(Error::VmCpuRunning),
            VmCpuStatus::PoweredOff => Err(Error::VmCpuOff),
            VmCpuStatus::NotPresent => Err(Error::VmCpuNotFound),
        }
    }

    /// Returns the status of the specified vCPU.
    pub fn get_vcpu_status(&self, vcpu_id: u64) -> Result<VmCpuStatus> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        Ok(*entry.status.read())
    }
}

// Safety: Each VmCpu is wrapped with a Mutex to provide safe concurrent access to VmCpu and its
// composite structures from within the hypervisor, and the VmCpu API provides safe accessors to
// the shared-memory vCPU state structure.
unsafe impl Sync for VmCpus {}
unsafe impl Send for VmCpus {}
