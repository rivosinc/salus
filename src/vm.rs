// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::mem::size_of;
use drivers::{CpuInfo, ImsicGuestId};
use memoffset::offset_of;
use page_collections::page_box::PageBox;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{PageState, PlatformPageTable};
use riscv_pages::{GuestPhysAddr, PageAddr, PageOwnerId, Pfn, RawAddr, SequentialPages};
use riscv_regs::{hgatp, hstatus, scounteren, sstatus};
use riscv_regs::{
    Exception, GeneralPurposeRegisters, GprIndex, LocalRegisterCopy, Readable, Trap, Writeable, CSR,
};
use sbi::Error as SbiError;
use sbi::{self, MeasurementFunction, ResetFunction, SbiMessage, SbiReturn, TeeFunction};
use spin::{Mutex, RwLock};

use crate::print_util::*;
use crate::sha256_measure::SHA256_DIGEST_BYTES;
use crate::vm_pages::{self, GuestRootBuilder, HostRootPages, VmPages};
use crate::{print, println};

const GUEST_ID_SELF_MEASUREMENT: u64 = 0;

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
    sstatus: u64,
    hstatus: u64,
    scounteren: u64,
    sepc: u64,
}

/// The CSRs that are only in effect when virtualization is enabled (V=1) and must be saved and
/// restored whenever we switch between VMs.
#[derive(Default)]
#[repr(C)]
struct GuestVCpuState {
    hgatp: u64,
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
pub struct TrapState {
    scause: u64,
    stval: u64,
    htval: u64,
    htinst: u64,
}

/// (v)CPU register state that must be saved or restored when entering/exiting a VM or switching
/// between VMs.
#[derive(Default)]
#[repr(C)]
struct VmCpuState {
    host_regs: HostCpuState,
    guest_regs: GuestCpuState,
    guest_vcpu_csrs: GuestVCpuState,
    trap_csrs: TrapState,
}

// The vCPU context switch, defined in guest.S
extern "C" {
    fn _run_guest(g: *mut VmCpuState);
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
    guest_sstatus = const guest_csr_offset!(sstatus),
    guest_hstatus = const guest_csr_offset!(hstatus),
    guest_scounteren = const guest_csr_offset!(scounteren),
    guest_sepc = const guest_csr_offset!(sepc),
);

struct Guests<T: PlatformPageTable> {
    inner: PageVec<PageBox<GuestState<T>>>,
    phys_pages: PageState,
}

impl<T: PlatformPageTable> Guests<T> {
    fn add(&mut self, guest_state: PageBox<GuestState<T>>) -> sbi::Result<()> {
        self.inner
            .try_reserve(1)
            .map_err(|_| SbiError::InvalidParam)?;
        self.inner.push(guest_state);
        Ok(())
    }

    // Returns the index in to the guest array of the given guest id. If the guest ID isn't valid or
    // isn't owned by this VM, then return an error.
    fn get_guest_index(&self, guest_id: u64) -> sbi::Result<usize> {
        let page_owner_id = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        self.inner
            .iter()
            .enumerate()
            .find(|(_i, g)| page_owner_id == g.page_owner_id())
            .map(|(guest_index, _guest)| guest_index)
            .ok_or(SbiError::InvalidParam)
    }

    fn remove(&mut self, guest_id: u64) -> sbi::Result<()> {
        let to_remove = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        self.inner.retain(|g| {
            if g.page_owner_id() == to_remove {
                self.phys_pages.rm_active_guest(g.page_owner_id());
                true
            } else {
                false
            }
        });
        Ok(())
    }

    // Returns a mutable reference to the guest for the given ID if it exists, otherwise None.
    fn guest_mut(&mut self, guest_id: u64) -> sbi::Result<&mut PageBox<GuestState<T>>> {
        let guest_index = self.get_guest_index(guest_id)?;
        self.inner
            .get_mut(guest_index)
            .ok_or(SbiError::InvalidParam)
    }

    // Returns an immutable reference to the guest for the given ID if it exists, otherwise None.
    fn guest(&self, guest_id: u64) -> sbi::Result<&PageBox<GuestState<T>>> {
        let guest_index = self.get_guest_index(guest_id)?;
        self.inner.get(guest_index).ok_or(SbiError::InvalidParam)
    }

    // returns the initializing guest if it's present and runnable, otherwise none
    fn initializing_guest(&self, guest_id: u64) -> sbi::Result<&GuestRootBuilder<T>> {
        self.guest(guest_id)
            .and_then(|g| g.as_guest_root_builder().ok_or(SbiError::InvalidParam))
    }

    // Returns the runnable guest if it's present and runnable, otherwise None
    fn running_guest(&self, guest_id: u64) -> sbi::Result<&Vm<T>> {
        self.guest(guest_id)
            .and_then(|g| g.as_vm().ok_or(SbiError::InvalidParam))
    }
}

enum GuestState<T: PlatformPageTable> {
    Init(GuestRootBuilder<T>),
    Running(Vm<T>),
    Temp,
}

impl<T: PlatformPageTable> GuestState<T> {
    fn page_owner_id(&self) -> PageOwnerId {
        match self {
            Self::Init(grb) => grb.page_owner_id(),
            Self::Running(v) => v.vm_pages.page_owner_id(),
            Self::Temp => unreachable!(),
        }
    }

    fn as_guest_root_builder(&self) -> Option<&GuestRootBuilder<T>> {
        match self {
            Self::Init(ref grb) => Some(grb),
            Self::Running(_) => None,
            Self::Temp => unreachable!(),
        }
    }

    fn as_vm(&self) -> Option<&Vm<T>> {
        match self {
            Self::Init(_) => None,
            Self::Running(ref v) => Some(v),
            Self::Temp => unreachable!(),
        }
    }
}

/// Identifies the exit cause for a vCPU.
pub enum VmCpuExit {
    /// ECALLs from VS mode.
    Ecall(Option<SbiMessage>),
    /// G-stage page faults.
    PageFault(GuestPhysAddr),
    /// Everything else that we currently don't or can't handle.
    Other(TrapState),
    // TODO: Add other exit causes as needed.
}

/// Represents a single virtual CPU of a VM.
pub struct VmCpu {
    state: VmCpuState,
    interrupt_file: Option<ImsicGuestId>,
    page_owner_id: PageOwnerId,
}

impl VmCpu {
    /// Creates a new vCPU using the address space of `vm_pages`.
    fn new<T: PlatformPageTable>(vm_pages: &VmPages<T>) -> Self {
        let mut state = VmCpuState::default();

        let mut hgatp = LocalRegisterCopy::<u64, hgatp::Register>::new(0);
        hgatp.modify(hgatp::vmid.val(1)); // TODO: VMID assignments.
        hgatp.modify(hgatp::ppn.val(Pfn::from(vm_pages.root_address()).bits()));
        hgatp.modify(hgatp::mode.val(T::HGATP_VALUE));
        state.guest_vcpu_csrs.hgatp = hgatp.get();

        let mut hstatus = LocalRegisterCopy::<u64, hstatus::Register>::new(0);
        hstatus.modify(hstatus::spv.val(1));
        hstatus.modify(hstatus::spvp::Supervisor);
        state.guest_regs.hstatus = hstatus.get();

        let mut sstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(0);
        sstatus.modify(sstatus::spie.val(1));
        sstatus.modify(sstatus::spp::Supervisor);
        state.guest_regs.sstatus = sstatus.get();

        let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
        scounteren.modify(scounteren::cycle.val(1));
        scounteren.modify(scounteren::time.val(1));
        scounteren.modify(scounteren::instret.val(1));
        state.guest_regs.scounteren = scounteren.get();

        // set the hart ID - TODO other hart IDs when multi-threaded
        state.guest_regs.gprs.set_reg(GprIndex::A0, 0);

        Self {
            state,
            interrupt_file: None,
            page_owner_id: vm_pages.page_owner_id(),
        }
    }

    /// Sets the launch arguments (entry point and A1) for this vCPU.
    fn set_launch_args(&mut self, entry_addr: GuestPhysAddr, a1: u64) {
        self.state.guest_regs.sepc = entry_addr.bits();
        self.state.guest_regs.gprs.set_reg(GprIndex::A1, a1);
    }

    /// Updates A0/A1 with the result of an SBI call.
    fn set_ecall_result(&mut self, result: SbiReturn) {
        self.state
            .guest_regs
            .gprs
            .set_reg(GprIndex::A0, result.error_code as u64);
        if result.error_code == sbi::SBI_SUCCESS {
            self.state
                .guest_regs
                .gprs
                .set_reg(GprIndex::A1, result.return_value as u64);
        }
    }

    /// Sets the interrupt file for this vCPU.
    fn set_interrupt_file(&mut self, interrupt_file: ImsicGuestId) {
        self.interrupt_file = Some(interrupt_file);

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(self.state.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.to_raw_index() as u64));
        self.state.guest_regs.hstatus = hstatus.get();
    }

    /// Runs this vCPU until it exits.
    fn run_to_exit(&mut self) -> VmCpuExit {
        // Load the vCPU CSRs. Safe as these don't take effect until V=1.
        CSR.hgatp.set(self.state.guest_vcpu_csrs.hgatp);
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

        // TO DO: This assumes that we'll never have a VM with sepc
        // deliberately set to 0. This is probably generally true
        // but we can set the start explicitly via an interface
        if self.state.guest_regs.sepc == 0 {
            self.state.guest_regs.sepc = 0x8020_0000;
        }

        // TODO, HGEIE programinng:
        //  - Track which guests the host wants interrupts from (by trapping HGEIE accesses from
        //    VS level) and update HGEIE[2:] appropriately.
        //  - If this is the host: clear HGEIE[1] on entry; inject SGEI into host VM if we receive
        //    any SGEI at HS level.
        //  - If this is a guest: set HGEIE[1] on entry; switch to the host VM for any SGEI that
        //    occur, injecting an SEI for the host interrupts and SGEI for guest VM interrupts.

        // TODO: Enforce that the vCPU has an assigned interrupt file before running.

        unsafe {
            // Safe to run the guest as it only touches memory assigned to it by being owned
            // by its page table.
            _run_guest(&mut self.state as *mut VmCpuState);
        }

        // Save off the trap information.
        self.state.trap_csrs.scause = CSR.scause.get();
        self.state.trap_csrs.stval = CSR.stval.get();
        self.state.trap_csrs.htval = CSR.htval.get();
        self.state.trap_csrs.htinst = CSR.htinst.get();

        // Save the vCPU state.
        self.state.guest_vcpu_csrs.hgatp = CSR.hgatp.get();
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

        // Determine the exit cause from the trap CSRs.
        use Exception::*;
        match Trap::from_scause(self.state.trap_csrs.scause).unwrap() {
            Trap::Exception(VirtualSupervisorEnvCall) => {
                let sbi_msg = SbiMessage::from_regs(&self.state.guest_regs.gprs).ok();
                self.state.guest_regs.sepc += 4;
                VmCpuExit::Ecall(sbi_msg)
            }
            Trap::Exception(GuestInstructionPageFault)
            | Trap::Exception(GuestLoadPageFault)
            | Trap::Exception(GuestStorePageFault) => {
                let fault_addr = RawAddr::guest(
                    self.state.trap_csrs.htval << 2 | self.state.trap_csrs.stval & 0x03,
                    self.page_owner_id,
                );
                VmCpuExit::PageFault(fault_addr)
            }
            _ => VmCpuExit::Other(self.state.trap_csrs.clone()),
        }
    }
}

/// A VM that is being run.
pub struct Vm<T: PlatformPageTable> {
    // TODO: Support multiple vCPUs.
    vcpu: Mutex<VmCpu>,
    vm_pages: VmPages<T>,
    guests: Option<RwLock<Guests<T>>>,
}

impl<T: PlatformPageTable> Vm<T> {
    /// Create a new guest using the given initial page table and pool of initial pages.
    fn new(vm_pages: VmPages<T>) -> Self {
        Self {
            vcpu: Mutex::new(VmCpu::new(&vm_pages)),
            vm_pages,
            guests: None,
        }
    }

    /// Sets the launch arguments (entry point and A1) for vCPU0.
    fn set_launch_args(&self, entry_addr: GuestPhysAddr, a1: u64) {
        let mut vcpu = self.vcpu.lock();
        vcpu.set_launch_args(entry_addr, a1);
    }

    /// Sets the interrupt file for vCPU0.
    fn set_interrupt_file(&self, interrupt_file: ImsicGuestId) {
        let mut vcpu = self.vcpu.lock();
        vcpu.set_interrupt_file(interrupt_file);
    }

    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    fn add_guest_tracking_pages(&mut self, pages: SequentialPages) {
        let guests = PageVec::from(pages);
        self.guests = Some(RwLock::new(Guests {
            inner: guests,
            phys_pages: self.vm_pages.phys_pages(),
        }));
    }

    /// Run this guest until an unhandled exit is encountered.
    fn run(&self, _vcpu_id: u64) -> VmCpuExit {
        loop {
            let mut vcpu = self.vcpu.lock();
            let exit = vcpu.run_to_exit();
            match exit {
                VmCpuExit::Ecall(Some(sbi_msg)) => {
                    match self.handle_ecall(sbi_msg) {
                        Ok(Some(sbi_ret)) => {
                            vcpu.set_ecall_result(sbi_ret);
                        }
                        Ok(None) => {
                            // for legacy, leave the a0 and a1 registers as-is.
                        }
                        Err(error_code) => {
                            vcpu.set_ecall_result(SbiReturn::from(error_code));
                        }
                    }
                }
                VmCpuExit::Ecall(None) => {
                    // Unrecognized ECALL, return an error.
                    vcpu.set_ecall_result(SbiReturn::from(sbi::Error::NotSupported));
                }
                VmCpuExit::PageFault(addr) => {
                    if self.handle_guest_fault(addr).is_err() {
                        return exit;
                    }
                }
                VmCpuExit::Other(ref trap_csrs) => {
                    println!("Unhandled guest exit, SCAUSE = 0x{:08x}", trap_csrs.scause);
                    return exit;
                }
            }
        }
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&self, msg: SbiMessage) -> sbi::Result<Option<SbiReturn>> {
        match msg {
            SbiMessage::PutChar(c) => {
                // put char - legacy command
                print!("{}", c as u8 as char);
                Ok(None)
            }
            SbiMessage::Reset(r) => {
                match r {
                    ResetFunction::Reset {
                        reset_type: _,
                        reason: _,
                    } => {
                        // TODO do shutdown of VM or system if from primary host VM
                        println!("Vm shutdown/reboot request");
                        crate::poweroff();
                    }
                }
            }
            SbiMessage::Base(_) => Err(SbiError::NotSupported), // TODO
            SbiMessage::HartState(_) => Err(SbiError::NotSupported), // TODO
            SbiMessage::Tee(tee_func) => Ok(Some(self.handle_tee_msg(tee_func))),
            SbiMessage::Measurement(measurement_func) => {
                Ok(Some(self.handle_measurement_msg(measurement_func)))
            }
        }
    }

    fn handle_tee_msg(&self, tee_func: TeeFunction) -> SbiReturn {
        use TeeFunction::*;
        match tee_func {
            TvmCreate(state_page) => self.add_guest(state_page).into(),
            TvmDestroy { guest_id } => self.destroy_guest(guest_id).into(),
            AddPageTablePages {
                guest_id,
                page_addr,
                num_pages,
            } => self
                .guest_add_page_table_pages(guest_id, page_addr, num_pages)
                .into(),
            AddPages {
                guest_id,
                page_addr,
                page_type,
                num_pages,
                gpa,
                measure_preserve,
            } => self
                .guest_add_pages(
                    guest_id,
                    page_addr,
                    page_type,
                    num_pages,
                    gpa,
                    measure_preserve,
                )
                .into(),
            Finalize { guest_id } => self.guest_finalize(guest_id).into(),
            Run { guest_id } => self.guest_run(guest_id).into(),
            RemovePages {
                guest_id,
                gpa,
                remap_addr: _, // TODO - remove
                num_pages,
            } => self.guest_rm_pages(guest_id, gpa, num_pages).into(),
            GetGuestMeasurement {
                measurement_version,
                measurement_type,
                page_addr,
                guest_id,
            } => self
                .guest_get_measurement(measurement_version, measurement_type, page_addr, guest_id)
                .into(),
        }
    }

    fn handle_measurement_msg(&self, measurement_func: MeasurementFunction) -> SbiReturn {
        use MeasurementFunction::*;
        match measurement_func {
            GetSelfMeasurement {
                measurement_version,
                measurement_type,
                page_addr,
            } => self
                .guest_get_measurement(
                    measurement_version,
                    measurement_type,
                    page_addr,
                    GUEST_ID_SELF_MEASUREMENT,
                )
                .into(),
        }
    }

    // Handle access faults. For example, when a returned page needs to be demand-faulted back to
    // the page table.
    fn handle_guest_fault(&self, fault_addr: GuestPhysAddr) -> vm_pages::Result<()> {
        self.vm_pages.handle_page_fault(fault_addr)?;

        // Get instruction that caused the fault
        //   - disable ints
        //   - load hstatus with value from guest
        //   - set stvec to catch traps during access
        //   - read instruction using HLV.HU (or tow for 32 bit).
        //   - reset stvec
        //   - reset hstatus
        //   - re-enable ints

        // Determine width of faulting access
        // determine destination/source register
        // Check how to service access (device emulation for example) and run.
        // if load, set destination register

        Ok(())
    }

    fn add_guest(&self, donor_pages_addr: u64) -> sbi::Result<u64> {
        println!("Add guest {:x}", donor_pages_addr);
        if self.guests.is_none() {
            return Err(SbiError::InvalidParam); // TODO different error
        }

        let from_page_addr = PageAddr::new(RawAddr::guest(
            donor_pages_addr,
            self.vm_pages.page_owner_id(),
        ))
        .ok_or(SbiError::InvalidAddress)?;

        let (guest_builder, state_page) = self
            .vm_pages
            .create_guest_root_builder(from_page_addr)
            .map_err(|_| SbiError::InvalidParam)?;
        // unwrap can't fail because a valid guest must have a valid guest id.
        let id = guest_builder.page_owner_id();

        // create a boxpage for builder state and add it to the list of vms.
        let guest_state: PageBox<GuestState<T>> =
            PageBox::new_with(GuestState::Init(guest_builder), state_page);

        let mut guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.write();
        guests.add(guest_state)?;

        Ok(id.raw())
    }

    fn destroy_guest(&self, guest_id: u64) -> sbi::Result<u64> {
        let mut guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.write();
        guests.remove(guest_id)?;
        Ok(0)
    }

    // converts the given guest from init to running
    fn guest_finalize(&self, guest_id: u64) -> sbi::Result<u64> {
        let mut guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.write();
        guests.guest_mut(guest_id).map(|g| {
            let mut temp = GuestState::Temp;
            core::mem::swap(&mut **g, &mut temp);
            let mut running = match temp {
                GuestState::Init(gbr) => GuestState::Running(Vm::new(gbr.create_pages())),
                t => t,
            };
            core::mem::swap(&mut **g, &mut running);
        })?;
        Ok(0)
    }

    fn guest_run(&self, guest_id: u64) -> sbi::Result<u64> {
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.running_guest(guest_id)?;
        guest_vm.run(0); // TODO: Take vCPU ID.
        Ok(0) // TODO: Return the exit reason to the host.
    }

    fn guest_add_page_table_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        num_pages: u64,
    ) -> sbi::Result<u64> {
        let from_page_addr =
            PageAddr::new(RawAddr::guest(from_addr, self.vm_pages.page_owner_id()))
                .ok_or(SbiError::InvalidAddress)?;

        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let grb = guests.initializing_guest(guest_id)?;
        self.vm_pages
            .add_pte_pages_builder(from_page_addr, num_pages, grb)
            .map_err(|e| {
                println!("Salus - pte_pages_builder error {e:?}");
                SbiError::InvalidAddress
            })?;

        Ok(0)
    }

    fn guest_rm_pages(&self, guest_id: u64, gpa: u64, num_pages: u64) -> sbi::Result<u64> {
        println!("Salus - Rm pages {guest_id:x} gpa:{gpa:x} num_pages:{num_pages}",);
        let from_page_addr = PageAddr::new(RawAddr::guest(gpa, self.vm_pages.page_owner_id()))
            .ok_or(SbiError::InvalidAddress)?;

        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        // TODO: Enforce that pages can't be removed while the guest is alive.
        let guest_vm = guests.running_guest(guest_id)?;
        guest_vm
            .vm_pages
            .remove_4k_pages(from_page_addr, num_pages)
            .map_err(|e| {
                println!("Salus - remove_4k_pages error {e:?}");
                SbiError::InvalidAddress
            })
    }

    /// page_type: 0 => 4K, 1=> 2M, 2=> 1G, 3=512G
    fn guest_add_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        page_type: u64,
        num_pages: u64,
        to_addr: u64,
        measure_preserve: bool,
    ) -> sbi::Result<u64> {
        println!(
            "Add pages {from_addr:x} page_type:{page_type} num_pages:{num_pages} to_addr:{to_addr:x}",
        );
        if page_type != 0 {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(SbiError::InvalidParam);
        }

        let from_page_addr =
            PageAddr::new(RawAddr::guest(from_addr, self.vm_pages.page_owner_id()))
                .ok_or(SbiError::InvalidAddress)?;
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let grb = guests.initializing_guest(guest_id)?;
        let to_page_addr = PageAddr::new(RawAddr::guest(to_addr, grb.page_owner_id()))
            .ok_or(SbiError::InvalidAddress)?;
        self.vm_pages
            .add_4k_pages_builder(
                from_page_addr,
                num_pages,
                grb,
                to_page_addr,
                measure_preserve,
            )
            .map_err(|_| SbiError::InvalidParam)?;

        Ok(num_pages)
    }

    // TODO: Add code to return actual measurements
    fn guest_get_measurement(
        &self,
        measurement_version: u64,
        measurement_type: u64,
        page_addr: u64,
        guest_id: u64,
    ) -> sbi::Result<u64> {
        let gpa = RawAddr::guest(page_addr, self.vm_pages.page_owner_id());
        if (measurement_version != 1) || (measurement_type != 1) || PageAddr::new(gpa).is_none() {
            return Err(SbiError::InvalidParam);
        }

        // The guest_id of 0 is a special identifier used to retrieve
        // measurements for self. Note that since we are borrowing
        // measurements from vm_pages, we can't take another mutable
        // reference to vm_pages to write to the GPA, so we have to
        // call a helper method to retrieve the measurements and write
        // them using the same mutable reference
        let result = if guest_id == GUEST_ID_SELF_MEASUREMENT {
            self.vm_pages.write_measurements_to_guest_owned_page(gpa)
        } else {
            // TODO: Define a compile-time constant for the maximum length of any measurement we
            // would conceivably use.
            let mut bytes = [0u8; SHA256_DIGEST_BYTES];
            let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
            let _ = guests.get_guest_index(guest_id)?;
            if let Ok(running_guest) = guests.running_guest(guest_id) {
                running_guest
                    .vm_pages
                    .get_measurement(&mut bytes)
                    .map_err(|_| SbiError::Failed)?;
            } else {
                guests
                    .initializing_guest(guest_id)
                    .unwrap()
                    .get_measurement(&mut bytes)
                    .map_err(|_| SbiError::Failed)?;
            }

            self.vm_pages.write_to_guest_owned_page(gpa, &bytes)
        };

        result
            .map(|bytes| bytes as u64)
            .map_err(|_| SbiError::InvalidAddress)
    }
}

/// Represents the special VM that serves as the host for the system.
pub struct Host<T: PlatformPageTable> {
    inner: Vm<T>,
}

impl<T: PlatformPageTable> Host<T> {
    /// Creates a new `Host` using the given initial page table root.
    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    pub fn new(page_root: HostRootPages<T>, pages: SequentialPages) -> Self {
        let mut inner = Vm::new(page_root.into_inner());
        inner.add_guest_tracking_pages(pages);
        inner.set_interrupt_file(ImsicGuestId::HostVm);
        Self { inner }
    }

    /// Sets the launch arguments (entry point and FDT) for the host vCPU.
    pub fn set_launch_args(&self, entry_addr: GuestPhysAddr, fdt_addr: GuestPhysAddr) {
        self.inner.set_launch_args(entry_addr, fdt_addr.bits());
    }

    /// Run the host. Only returns for system shutdown
    pub fn run(&mut self, vcpu_id: u64) -> VmCpuExit {
        // TODO - return value need to be host specific
        self.inner.run(vcpu_id)
    }
}
