// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::vm_pages::VmPages;
use core::arch::global_asm;
use core::mem::size_of;
use drivers::{CpuInfo, ImsicGuestId};
use memoffset::offset_of;
use riscv_page_tables::GuestStagePageTable;
use riscv_pages::{GuestPhysAddr, PageOwnerId, Pfn, RawAddr};
use riscv_regs::{hgatp, hstatus, scounteren, sstatus};
use riscv_regs::{
    Exception, GeneralPurposeRegisters, GprIndex, LocalRegisterCopy, Readable, Trap, Writeable, CSR,
};
use sbi::{SbiMessage, SbiReturn};

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
    host_regs: HostCpuState,
    guest_regs: GuestCpuState,
    guest_vcpu_csrs: GuestVCpuState,
    trap_csrs: VmCpuTrapState,
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

/// Identifies the exit cause for a vCPU.
pub enum VmCpuExit {
    /// ECALLs from VS mode.
    Ecall(Option<SbiMessage>),
    /// G-stage page faults.
    PageFault(GuestPhysAddr),
    /// Everything else that we currently don't or can't handle.
    Other(VmCpuTrapState),
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
    pub fn new<T: GuestStagePageTable>(vm_pages: &VmPages<T>) -> Self {
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
    pub fn set_launch_args(&mut self, entry_addr: GuestPhysAddr, a1: u64) {
        self.state.guest_regs.sepc = entry_addr.bits();
        self.state.guest_regs.gprs.set_reg(GprIndex::A1, a1);
    }

    /// Updates A0/A1 with the result of an SBI call.
    pub fn set_ecall_result(&mut self, result: SbiReturn) {
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
    pub fn set_interrupt_file(&mut self, interrupt_file: ImsicGuestId) {
        self.interrupt_file = Some(interrupt_file);

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(self.state.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.to_raw_index() as u64));
        self.state.guest_regs.hstatus = hstatus.get();
    }

    /// Runs this vCPU until it exits.
    pub fn run_to_exit(&mut self) -> VmCpuExit {
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
