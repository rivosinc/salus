// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::mem::size_of;
use drivers::{CpuInfo, ImsicGuestId};
use memoffset::offset_of;
use page_collections::page_box::PageBox;
use page_collections::page_vec::PageVec;
use riscv_page_tables::PlatformPageTable;
use riscv_pages::{PageAddr, PageOwnerId, RawAddr, SequentialPages};
use riscv_regs::{hgatp, hstatus, scounteren, sstatus, HgatpHelpers};
use riscv_regs::{
    Exception, GeneralPurposeRegisters, GprIndex, LocalRegisterCopy, Readable, Trap, Writeable, CSR,
};
use sbi::Error as SbiError;
use sbi::{self, ResetFunction, SbiMessage, SbiReturn, TeeFunction};

use crate::print_util::*;
use crate::vm_pages::{self, GuestRootBuilder, HostRootPages, VmPages};
use crate::{print, println};

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
#[derive(Default)]
#[repr(C)]
struct TrapState {
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
        self.inner.retain(|g| g.page_owner_id() != to_remove);
        Ok(())
    }

    // Returns the guest for the given ID if it exists, otherwise None.
    fn guest_mut(&mut self, guest_id: u64) -> sbi::Result<&mut PageBox<GuestState<T>>> {
        let guest_index = self.get_guest_index(guest_id)?;
        self.inner
            .get_mut(guest_index)
            .ok_or(SbiError::InvalidParam)
    }

    // returns the initializing guest if it's present and runnable, otherwise none
    fn initializing_guest_mut(&mut self, guest_id: u64) -> sbi::Result<&mut GuestRootBuilder<T>> {
        self.guest_mut(guest_id)
            .and_then(|g| g.init_mut().ok_or(SbiError::InvalidParam))
    }

    // Returns the runnable guest if it's present and runnable, otherwise None
    fn running_guest_mut(&mut self, guest_id: u64) -> sbi::Result<&mut Vm<T>> {
        self.guest_mut(guest_id)
            .and_then(|g| g.vm_mut().ok_or(SbiError::InvalidParam))
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

    fn init_mut(&mut self) -> Option<&mut GuestRootBuilder<T>> {
        match self {
            Self::Init(ref mut grb) => Some(grb),
            Self::Running(_) => None,
            Self::Temp => unreachable!(),
        }
    }

    fn vm_mut(&mut self) -> Option<&mut Vm<T>> {
        match self {
            Self::Init(_) => None,
            Self::Running(ref mut v) => Some(v),
            Self::Temp => unreachable!(),
        }
    }
}

/// A Vm VM that is being run.
pub struct Vm<T: PlatformPageTable> {
    // TODO, info should be per-hart.
    info: VmCpuState,
    vm_pages: VmPages<T>,
    guests: Option<Guests<T>>,
    interrupt_file: Option<ImsicGuestId>, // TODO: Should be per-hart
    has_run: bool, // TODO - different Vm type for different life cycle stages.
}

impl<T: PlatformPageTable> Vm<T> {
    /// Create a new guest using the given initial page table and pool of initial pages.
    fn new(vm_pages: VmPages<T>) -> Self {
        let mut info = VmCpuState::default();

        let mut hgatp = LocalRegisterCopy::<u64, hgatp::Register>::new(0);
        hgatp.set_from(vm_pages.root(), 1);
        info.guest_vcpu_csrs.hgatp = hgatp.get();

        let mut hstatus = LocalRegisterCopy::<u64, hstatus::Register>::new(0);
        hstatus.modify(hstatus::spv.val(1));
        hstatus.modify(hstatus::spvp::Supervisor);
        info.guest_regs.hstatus = hstatus.get();

        let mut sstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(0);
        sstatus.modify(sstatus::spie.val(1));
        sstatus.modify(sstatus::spp::Supervisor);
        info.guest_regs.sstatus = sstatus.get();

        let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
        scounteren.modify(scounteren::cycle.val(1));
        scounteren.modify(scounteren::time.val(1));
        scounteren.modify(scounteren::instret.val(1));
        info.guest_regs.scounteren = scounteren.get();

        // set the hart ID - TODO other hart IDs when multi-threaded
        info.guest_regs.gprs.set_reg(GprIndex::A0, 0);

        Vm {
            info,
            vm_pages,
            guests: None,
            interrupt_file: None,
            has_run: false,
        }
    }

    fn set_entry_address(&mut self, entry_addr: u64) {
        self.info.guest_regs.sepc = entry_addr;
    }

    fn set_interrupt_file(&mut self, interrupt_file: ImsicGuestId) {
        self.interrupt_file = Some(interrupt_file);

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(self.info.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.to_raw_index() as u64));
        self.info.guest_regs.hstatus = hstatus.get();
    }

    // TODO - also pass the DT here and copy it?
    fn add_device_tree(&mut self, dt_addr: u64) {
        // set the DT address to the one passed in.
        self.info.guest_regs.gprs.set_reg(GprIndex::A1, dt_addr);
    }

    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    fn add_guest_tracking_pages(&mut self, pages: SequentialPages) {
        let guests = PageVec::from(pages);
        self.guests = Some(Guests { inner: guests });
    }

    /// Run this VM until the guest exits
    fn run_to_exit(&mut self, _hart_id: u64) {
        // Load the vCPU CSRs. Safe as these don't take effect until V=1.
        CSR.hgatp.set(self.info.guest_vcpu_csrs.hgatp);
        CSR.htimedelta.set(self.info.guest_vcpu_csrs.htimedelta);
        CSR.vsstatus.set(self.info.guest_vcpu_csrs.vsstatus);
        CSR.vsie.set(self.info.guest_vcpu_csrs.vsie);
        CSR.vstvec.set(self.info.guest_vcpu_csrs.vstvec);
        CSR.vsscratch.set(self.info.guest_vcpu_csrs.vsscratch);
        CSR.vsepc.set(self.info.guest_vcpu_csrs.vsepc);
        CSR.vscause.set(self.info.guest_vcpu_csrs.vscause);
        CSR.vstval.set(self.info.guest_vcpu_csrs.vstval);
        CSR.vsatp.set(self.info.guest_vcpu_csrs.vsatp);
        if CpuInfo::get().has_sstc() {
            CSR.vstimecmp.set(self.info.guest_vcpu_csrs.vstimecmp);
        }

        // TO DO: This assumes that we'll never have a VM with sepc
        // deliberately set to 0. This is probably generally true
        // but we can set the start explicitly via an interface
        if self.info.guest_regs.sepc == 0 {
            self.info.guest_regs.sepc = 0x8020_0000;
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
            _run_guest(&mut self.info as *mut VmCpuState);
        }

        // Save off the trap information.
        self.info.trap_csrs.scause = CSR.scause.get();
        self.info.trap_csrs.stval = CSR.stval.get();
        self.info.trap_csrs.htval = CSR.htval.get();
        self.info.trap_csrs.htinst = CSR.htinst.get();

        // Save the vCPU state.
        self.info.guest_vcpu_csrs.hgatp = CSR.hgatp.get();
        self.info.guest_vcpu_csrs.htimedelta = CSR.htimedelta.get();
        self.info.guest_vcpu_csrs.vsstatus = CSR.vsstatus.get();
        self.info.guest_vcpu_csrs.vsie = CSR.vsie.get();
        self.info.guest_vcpu_csrs.vstvec = CSR.vstvec.get();
        self.info.guest_vcpu_csrs.vsscratch = CSR.vsscratch.get();
        self.info.guest_vcpu_csrs.vsepc = CSR.vsepc.get();
        self.info.guest_vcpu_csrs.vscause = CSR.vscause.get();
        self.info.guest_vcpu_csrs.vstval = CSR.vstval.get();
        self.info.guest_vcpu_csrs.vsatp = CSR.vsatp.get();
        if CpuInfo::get().has_sstc() {
            self.info.guest_vcpu_csrs.vstimecmp = CSR.vstimecmp.get();
        }
    }

    /// Run this guest until it requests an exit or an interrupt is received for the host.
    fn run(&mut self, hart_id: u64) -> Trap {
        use Exception::*;
        self.has_run = true;
        loop {
            self.run_to_exit(hart_id);
            match Trap::from_scause(self.info.trap_csrs.scause).unwrap() {
                Trap::Exception(VirtualSupervisorEnvCall) => {
                    self.handle_ecall();
                    self.inc_sepc_ecall(); // must return to _after_ the ecall.
                }
                Trap::Exception(GuestInstructionPageFault) => {
                    if self.handle_guest_fault(/*Instruction*/).is_err() {
                        return Trap::Exception(GuestInstructionPageFault);
                    }
                }
                Trap::Exception(GuestLoadPageFault) => {
                    if self.handle_guest_fault(/*Load*/).is_err() {
                        return Trap::Exception(GuestLoadPageFault);
                    }
                }
                Trap::Exception(GuestStorePageFault) => {
                    if self.handle_guest_fault(/*Store*/).is_err() {
                        return Trap::Exception(GuestStorePageFault);
                    }
                }
                e => return e, // TODO
            }
        }
    }

    /// Advances the sepc past the ecall instruction that caused the exit.
    fn inc_sepc_ecall(&mut self) {
        self.info.guest_regs.sepc += 4;
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&mut self) {
        // determine the call from a7, a6, and a2-5, put error code in a0 and return value in a1.
        // a0 and a1 aren't set by legacy extensions so the block below yields an `Option` that is
        // written when set to `Some(val)`.
        let result = SbiMessage::from_regs(&self.info.guest_regs.gprs).and_then(|msg| {
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
            }
        });

        match result {
            Ok(Some(sbi_ret)) => {
                self.info
                    .guest_regs
                    .gprs
                    .set_reg(GprIndex::A0, sbi_ret.error_code as u64);
                self.info
                    .guest_regs
                    .gprs
                    .set_reg(GprIndex::A1, sbi_ret.return_value as u64);
            }
            Ok(None) => {
                // for legacy, leave the a0 and a1 registers as-is.
            }
            Err(error_code) => {
                self.info
                    .guest_regs
                    .gprs
                    .set_reg(GprIndex::A0, SbiReturn::from(error_code).error_code as u64);
            }
        }
    }

    fn handle_tee_msg(&mut self, tee_func: TeeFunction) -> SbiReturn {
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
                guest_id,
                measurement_version,
                measurement_type,
                page_addr,
            } => self
                .guest_get_measurement(guest_id, measurement_version, measurement_type, page_addr)
                .into(),
        }
    }

    // Handle access faults. For example, when a returned page needs to be demand-faulted back to
    // the page table.
    fn handle_guest_fault(&mut self) -> core::result::Result<(), vm_pages::Error> {
        let fault_addr = RawAddr::guest(
            self.info.trap_csrs.htval << 2 | self.info.trap_csrs.stval & 0x03,
            self.vm_pages.page_owner_id(),
        );
        println!(
            "got fault stval: {:x} htval: {:x} sepc: {:x} address: {:x}",
            self.info.trap_csrs.stval,
            self.info.trap_csrs.htval,
            self.info.guest_regs.sepc,
            fault_addr.bits(),
        );

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

    fn add_guest(&mut self, donor_pages_addr: u64) -> sbi::Result<u64> {
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
        self.guests
            .as_mut()
            .ok_or(SbiError::InvalidParam)
            .and_then(|g| g.add(guest_state))?;

        Ok(id.raw())
    }

    fn destroy_guest(&mut self, guest_id: u64) -> sbi::Result<u64> {
        self.guests
            .as_mut()
            .ok_or(SbiError::InvalidParam)
            .and_then(|guests| guests.remove(guest_id))?;
        Ok(0)
    }

    // converts the given guest from init to running
    fn guest_finalize(&mut self, guest_id: u64) -> sbi::Result<u64> {
        let guests = self.guests.as_mut().ok_or(SbiError::InvalidParam)?;
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

    fn guest_run(&mut self, guest_id: u64) -> sbi::Result<u64> {
        self.guests
            .as_mut()
            .ok_or(SbiError::InvalidParam)
            .and_then(|guests| guests.running_guest_mut(guest_id))
            .map(|v| v.run(0))?; // TODO take hart id
        Ok(0)
    }

    fn guest_add_page_table_pages(
        &mut self,
        guest_id: u64,
        from_addr: u64,
        num_pages: u64,
    ) -> sbi::Result<u64> {
        let from_page_addr =
            PageAddr::new(RawAddr::guest(from_addr, self.vm_pages.page_owner_id()))
                .ok_or(SbiError::InvalidAddress)?;

        self.guests
            .as_mut()
            .ok_or(SbiError::InvalidParam)
            .and_then(|guests| guests.initializing_guest_mut(guest_id))
            .and_then(|grb| {
                self.vm_pages
                    .add_pte_pages_builder(from_page_addr, num_pages, grb)
                    .map_err(|e| {
                        println!("Salus - pte_pages_builder error {e:?}");
                        SbiError::InvalidAddress
                    })
            })?;

        Ok(0)
    }

    fn guest_rm_pages(&mut self, guest_id: u64, gpa: u64, num_pages: u64) -> sbi::Result<u64> {
        println!("Salus - Rm pages {guest_id:x} gpa:{gpa:x} num_pages:{num_pages}",);
        let from_page_addr = PageAddr::new(RawAddr::guest(gpa, self.vm_pages.page_owner_id()))
            .ok_or(SbiError::InvalidAddress)?;

        self.guests
            .as_mut()
            .ok_or(SbiError::InvalidParam)
            .and_then(|guests| guests.running_guest_mut(guest_id))
            .and_then(|g| {
                g.vm_pages
                    .remove_4k_pages(from_page_addr, num_pages)
                    .map_err(|e| {
                        println!("Salus - remove_4k_pages error {e:?}");
                        SbiError::InvalidAddress
                    })
            })
    }

    /// page_type: 0 => 4K, 1=> 2M, 2=> 1G, 3=512G
    fn guest_add_pages(
        &mut self,
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
        self.guests
            .as_mut()
            .ok_or(SbiError::InvalidParam)
            .and_then(|guests| guests.initializing_guest_mut(guest_id))
            .and_then(|grb| {
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
                    .map_err(|_| SbiError::InvalidParam)
            })?;

        Ok(num_pages)
    }

    // TODO: Add code to return actual measurements
    fn guest_get_measurement(
        &mut self,
        guest_id: u64,
        measurement_version: u64,
        measurement_type: u64,
        page_addr: u64,
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
        let result = if guest_id == 0 {
            self.vm_pages.write_measurements_to_guest_owned_page(gpa)
        } else {
            let guests = self.guests.as_mut().ok_or(SbiError::InvalidParam)?;
            let _ = guests.get_guest_index(guest_id)?;
            let measurements = if let Ok(running_guest) = guests.running_guest_mut(guest_id) {
                running_guest.vm_pages.get_measurement()
            } else {
                guests
                    .initializing_guest_mut(guest_id)
                    .unwrap()
                    .get_measurement()
            };

            self.vm_pages.write_to_guest_owned_page(gpa, measurements)
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
    /* TODO
    /// Creates from the system memory pool
    pub fn from_mem_pool(HypMemMap?) -> Self{}
    */

    /// Creates a new `Host` using the given initial page table root.
    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    pub fn new(page_root: HostRootPages<T>, pages: SequentialPages) -> Self {
        let mut inner = Vm::new(page_root.into_inner());
        inner.add_guest_tracking_pages(pages);
        inner.set_interrupt_file(ImsicGuestId::HostVm);
        Self { inner }
    }

    // TODO - also pass the DT here and copy it?
    pub fn add_device_tree(&mut self, dt_addr: u64) {
        self.inner.add_device_tree(dt_addr)
    }

    /// Set the address we should 'sret' to upon entering the VM.
    pub fn set_entry_address(&mut self, entry_addr: u64) {
        self.inner.set_entry_address(entry_addr);
    }

    /// Run the host. Only returns for system shutdown
    //TODO - return value need to be host specific
    pub fn run(&mut self, hart_id: u64) -> Trap {
        self.inner.run(hart_id)
    }
}
