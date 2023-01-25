// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::smp::PerCpu;

use core::arch::global_asm;
use core::fmt;
use core::mem::size_of;
use core::ops::ControlFlow;
use memoffset::offset_of;
use riscv_elf::ElfMap;
use riscv_regs::{Exception::UserEnvCall, GeneralPurposeRegisters, GprIndex, Readable, Trap, CSR};
use s_mode_utils::print::*;
use spin::Once;
use u_mode_api::{Error as UmodeApiError, HypCall, TryIntoRegisters, UmodeRequest};

/// Host GPR and which must be saved/restored when entering/exiting U-mode.
#[derive(Default)]
#[repr(C)]
struct HostCpuRegs {
    gprs: GeneralPurposeRegisters,
    sstatus: u64,
    stvec: u64,
    sscratch: u64,
}

/// Umode GPR and CSR state which must be saved/restored when exiting/entering U-mode.
#[derive(Default)]
#[repr(C)]
struct UmodeCpuRegs {
    gprs: GeneralPurposeRegisters,
    sepc: u64,
    sstatus: u64,
}

/// CSRs written on an exit from virtualization that are used by the host to determine the cause of
/// the trap.
#[derive(Default)]
#[repr(C)]
struct TrapRegs {
    scause: u64,
    stval: u64,
}

/// CPU register state that must be saved or restored when entering/exiting U-mode.
#[derive(Default)]
#[repr(C)]
struct UmodeCpuArchState {
    hyp_regs: HostCpuRegs,
    umode_regs: UmodeCpuRegs,
    trap_csrs: TrapRegs,
}

impl fmt::Display for UmodeCpuArchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let uregs = &self.umode_regs;
        writeln!(
            f,
            "SEPC: 0x{:016x}, SSTATUS: 0x{:016x}",
            uregs.sepc, uregs.sstatus,
        )?;
        writeln!(
            f,
            "SCAUSE: 0x{:016x}, STVAL: 0x{:016x}",
            self.trap_csrs.scause, self.trap_csrs.stval,
        )?;
        use GprIndex::*;
        writeln!(
            f,
            "RA:  0x{:016x}, GP:  0x{:016x}, TP:  0x{:016x}, S0:  0x{:016x}",
            uregs.gprs.reg(RA),
            uregs.gprs.reg(GP),
            uregs.gprs.reg(TP),
            uregs.gprs.reg(S0)
        )?;
        writeln!(
            f,
            "S1:  0x{:016x}, A0:  0x{:016x}, A1:  0x{:016x}, A2:  0x{:016x}",
            uregs.gprs.reg(S1),
            uregs.gprs.reg(A0),
            uregs.gprs.reg(A1),
            uregs.gprs.reg(A2)
        )?;
        writeln!(
            f,
            "A3:  0x{:016x}, A4:  0x{:016x}, A5:  0x{:016x}, A6:  0x{:016x}",
            uregs.gprs.reg(A3),
            uregs.gprs.reg(A4),
            uregs.gprs.reg(A5),
            uregs.gprs.reg(A6)
        )?;
        writeln!(
            f,
            "A7:  0x{:016x}, S2:  0x{:016x}, S3:  0x{:016x}, S4:  0x{:016x}",
            uregs.gprs.reg(A7),
            uregs.gprs.reg(S2),
            uregs.gprs.reg(S3),
            uregs.gprs.reg(S4)
        )?;
        writeln!(
            f,
            "S5:  0x{:016x}, S6:  0x{:016x}, S7:  0x{:016x}, S8:  0x{:016x}",
            uregs.gprs.reg(S5),
            uregs.gprs.reg(S6),
            uregs.gprs.reg(S7),
            uregs.gprs.reg(S8)
        )?;
        writeln!(
            f,
            "S9:  0x{:016x}, S10: 0x{:016x}, S11: 0x{:016x}, T0:  0x{:016x}",
            uregs.gprs.reg(S9),
            uregs.gprs.reg(S10),
            uregs.gprs.reg(S11),
            uregs.gprs.reg(T0)
        )?;
        writeln!(
            f,
            "T1:  0x{:016x}, T2:  0x{:016x}, T3:  0x{:016x}, T4:  0x{:016x}",
            uregs.gprs.reg(T1),
            uregs.gprs.reg(T2),
            uregs.gprs.reg(T3),
            uregs.gprs.reg(T4)
        )?;
        writeln!(
            f,
            "T5:  0x{:016x}, T6:  0x{:016x}, SP:  0x{:016x}",
            uregs.gprs.reg(T5),
            uregs.gprs.reg(T6),
            uregs.gprs.reg(SP)
        )
    }
}

extern "C" {
    // umode context switch. Defined in umode.S
    fn _run_umode(g: *mut UmodeCpuArchState);
}

#[allow(dead_code)]
const fn hyp_gpr_offset(index: GprIndex) -> usize {
    offset_of!(UmodeCpuArchState, hyp_regs)
        + offset_of!(HostCpuRegs, gprs)
        + (index as usize) * size_of::<u64>()
}

#[allow(dead_code)]
const fn umode_gpr_offset(index: GprIndex) -> usize {
    offset_of!(UmodeCpuArchState, umode_regs)
        + offset_of!(UmodeCpuRegs, gprs)
        + (index as usize) * size_of::<u64>()
}

macro_rules! hyp_csr_offset {
    ($reg:tt) => {
        offset_of!(UmodeCpuArchState, hyp_regs) + offset_of!(HostCpuRegs, $reg)
    };
}

macro_rules! umode_csr_offset {
    ($reg:tt) => {
        offset_of!(UmodeCpuArchState, umode_regs) + offset_of!(UmodeCpuRegs, $reg)
    };
}

global_asm!(
    include_str!("umode.S"),
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
    hyp_stvec = const hyp_csr_offset!(stvec),
    hyp_sscratch = const hyp_csr_offset!(sscratch),
    umode_ra = const umode_gpr_offset(GprIndex::RA),
    umode_gp = const umode_gpr_offset(GprIndex::GP),
    umode_tp = const umode_gpr_offset(GprIndex::TP),
    umode_s0 = const umode_gpr_offset(GprIndex::S0),
    umode_s1 = const umode_gpr_offset(GprIndex::S1),
    umode_a0 = const umode_gpr_offset(GprIndex::A0),
    umode_a1 = const umode_gpr_offset(GprIndex::A1),
    umode_a2 = const umode_gpr_offset(GprIndex::A2),
    umode_a3 = const umode_gpr_offset(GprIndex::A3),
    umode_a4 = const umode_gpr_offset(GprIndex::A4),
    umode_a5 = const umode_gpr_offset(GprIndex::A5),
    umode_a6 = const umode_gpr_offset(GprIndex::A6),
    umode_a7 = const umode_gpr_offset(GprIndex::A7),
    umode_s2 = const umode_gpr_offset(GprIndex::S2),
    umode_s3 = const umode_gpr_offset(GprIndex::S3),
    umode_s4 = const umode_gpr_offset(GprIndex::S4),
    umode_s5 = const umode_gpr_offset(GprIndex::S5),
    umode_s6 = const umode_gpr_offset(GprIndex::S6),
    umode_s7 = const umode_gpr_offset(GprIndex::S7),
    umode_s8 = const umode_gpr_offset(GprIndex::S8),
    umode_s9 = const umode_gpr_offset(GprIndex::S9),
    umode_s10 = const umode_gpr_offset(GprIndex::S10),
    umode_s11 = const umode_gpr_offset(GprIndex::S11),
    umode_t0 = const umode_gpr_offset(GprIndex::T0),
    umode_t1 = const umode_gpr_offset(GprIndex::T1),
    umode_t2 = const umode_gpr_offset(GprIndex::T2),
    umode_t3 = const umode_gpr_offset(GprIndex::T3),
    umode_t4 = const umode_gpr_offset(GprIndex::T4),
    umode_t5 = const umode_gpr_offset(GprIndex::T5),
    umode_t6 = const umode_gpr_offset(GprIndex::T6),
    umode_sp = const umode_gpr_offset(GprIndex::SP),
    umode_sepc = const umode_csr_offset!(sepc),
    umode_sstatus = const umode_csr_offset!(sstatus),
);

/// Errors returned by U-mode operations.
#[derive(Debug)]
pub enum Error {
    /// Received an unexpected trap while running Umode.
    UnexpectedTrap,
    /// Umode called panic.
    Panic,
    /// Error in umode.
    Umode(UmodeApiError),
}

// Entry for umode task.
static UMODE_ENTRY: Once<u64> = Once::new();

/// Represents a per-CPU U-mode task.
pub struct UmodeTask {
    arch: UmodeCpuArchState,
}

impl UmodeTask {
    /// Initialize U-mode tasks. Must be called once before `setup_this_cpu()`.
    pub fn init(umode_elf: ElfMap) {
        UMODE_ENTRY.call_once(|| umode_elf.entry());
        // Consumes the ElfMap.
    }

    /// Initialize this CPU's U-mode task. Must be called once on each physical CPU.
    pub fn setup_this_cpu() -> Result<(), Error> {
        let arch = UmodeCpuArchState::default();
        let mut task = UmodeTask { arch };
        task.reset()?;
        // Install umode cpu state in the current cpu.
        PerCpu::this_cpu().set_umode_task(task);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        // Initialize umode CPU state to run at ELF entry.
        let mut arch = UmodeCpuArchState::default();
        // Unwrap okay: this is called after `Self::init()`.
        arch.umode_regs.sepc = *UMODE_ENTRY.get().unwrap();
        // Set cpu id as a0.
        arch.umode_regs
            .gprs
            .set_reg(GprIndex::A0, PerCpu::this_cpu().cpu_id().raw() as u64);
        // sstatus set to 0 (by default) is actually okay.
        self.arch = arch;
        // Run task until it initializes itself and calls HypCall::NextOp().
        self.run()?;
        Ok(())
    }

    /// Send a request and execute U-mode until an error is returned.
    pub fn send_req(req: UmodeRequest) -> Result<(), Error> {
        let mut task = PerCpu::this_cpu().umode_task_mut();
        req.to_registers(task.arch.umode_regs.gprs.a_regs_mut());
        let ret = task.run();
        if let Err(e) = &ret {
            match e {
                Error::UnexpectedTrap | Error::Panic => {
                    // Reset on unrecoverable u-mode error.
                    // 1. restore memory to original state
                    PerCpu::this_cpu().page_table().restore_umode();
                    // 2. setup umode again.
                    // Panic if we can't restore umode: the hypervisor is in an unrecoverable state.
                    task.reset().expect("Failed to recover U-mode.");
                }
                Error::Umode(_) => {}
            }
        }
        ret
    }

    fn handle_ecall(&mut self) -> ControlFlow<Result<(), Error>> {
        let regs = self.arch.umode_regs.gprs.a_regs();
        let cflow = match HypCall::try_from_registers(regs) {
            Ok(hypercall) => match hypercall {
                HypCall::Panic => {
                    println!("U-mode panic!");
                    println!("{}", self.arch);
                    ControlFlow::Break(Err(Error::Panic))
                }
                HypCall::PutChar(byte) => {
                    if let Some(c) = char::from_u32(byte as u32) {
                        print!("{}", c);
                    }
                    // No return. Leave umode registers untouched and return.
                    ControlFlow::Continue(())
                }
                HypCall::NextOp(result) => ControlFlow::Break(result.map_err(Error::Umode)),
            },
            Err(err) => {
                // Hypervisor could not parse the hypercall. Stop running umode and return error.
                ControlFlow::Break(Err(Error::Umode(err)))
            }
        };
        // Increase SEPC to skip ecall on entry.
        self.arch.umode_regs.sepc += 4;
        cflow
    }

    /// Run until it exits
    fn run_to_exit(&mut self) {
        unsafe {
            // Safe to run umode code as it only touches memory assigned to it through umode mappings.
            _run_umode(&mut self.arch as *mut UmodeCpuArchState);
        }
        // Save off the trap information.
        self.arch.trap_csrs.scause = CSR.scause.get();
        self.arch.trap_csrs.stval = CSR.stval.get();
    }

    // Run `umode` until result is returned.
    fn run(&mut self) -> Result<(), Error> {
        loop {
            self.run_to_exit();
            match Trap::from_scause(self.arch.trap_csrs.scause).unwrap() {
                Trap::Exception(UserEnvCall) => match self.handle_ecall() {
                    ControlFlow::Continue(_) => continue,
                    ControlFlow::Break(res) => break res,
                },
                _ => {
                    println!("Unexpected U-mode Trap:");
                    println!("{}", self.arch);
                    break Err(Error::UnexpectedTrap);
                }
            }
        }
    }
}
