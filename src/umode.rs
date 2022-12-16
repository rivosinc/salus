// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::smp::PerCpu;

use core::cell::{RefCell, RefMut};
use riscv_elf::ElfMap;
use riscv_regs::GeneralPurposeRegisters;
use spin::Once;

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

/// Errors returned by U-mode runs.
#[derive(Debug)]
pub enum Error {
    /// Task already active.
    TaskBusy,
}

// Entry for umode task.
static UMODE_ENTRY: Once<u64> = Once::new();

/// Represents a U-mode state with its running context.
pub struct UmodeTask {
    arch: RefCell<UmodeCpuArchState>,
}

impl UmodeTask {
    /// Initialize U-mode tasks. Must be called once bofore `setup_this_cpu()`.
    pub fn init(umode_elf: ElfMap) {
        UMODE_ENTRY.call_once(|| umode_elf.entry());
        // Consumes the ElfMap.
    }

    /// Initialize a new U-mode task. Must be called once on each physical CPU.
    pub fn setup_this_cpu() {
        let mut arch = UmodeCpuArchState::default();
        // sstatus set to 0 (by default) is actually okay.
        // Unwrap okay: this is called after `Self::init()`.
        arch.umode_regs.sepc = *UMODE_ENTRY.get().unwrap();
        let task = UmodeTask {
            arch: RefCell::new(arch),
        };
        // Install umode in the current cpu.
        PerCpu::this_cpu().set_umode_task(task);
    }

    /// Return this CPU's task. Must be call after `Self::setup_this_cpu()`.
    pub fn get() -> &'static UmodeTask {
        PerCpu::this_cpu().umode_task()
    }

    /// Activate this umode in order to run it.
    pub fn activate(&self) -> Result<ActiveUmodeTask, Error> {
        let arch = self.arch.try_borrow_mut().map_err(|_| Error::TaskBusy)?;
        Ok(ActiveUmodeTask { arch })
    }
}

/// Represents a U-mode that is running or runnable. Not at initial state.
pub struct ActiveUmodeTask<'act> {
    arch: RefMut<'act, UmodeCpuArchState>,
}

impl<'act> ActiveUmodeTask<'act> {
    /// Run `umode` until completion or error.
    pub fn run(&mut self) -> Result<(), Error> {
        // Dummy write.
        self.arch.trap_csrs.stval = 0;
        Ok(())
    }
}
