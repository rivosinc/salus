// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::{hedeleg, hideleg, hie, hip, hvip, reason, scause, sie, sip};
use tock_registers::fields::FieldValue;
use tock_registers::LocalRegisterCopy;

/// Errors as a result of converting to/from CSR values and Trap enums.
#[derive(Copy, Clone, Debug)]
pub enum Error {
    /// Unknwon cause value in CSR.
    UnknownCause(u64),

    /// Exception/interrupt cause can't be used with this CSR.
    InvalidCause,
}

pub type Result<T> = core::result::Result<T, Error>;

/// Trap causes.
#[derive(Copy, Clone, Debug)]
pub enum Trap {
    Interrupt(Interrupt),
    Exception(Exception),
}

impl Trap {
    /// Returns the Trap corresponding to the raw scause value.
    pub fn from_scause(csr: u64) -> Result<Self> {
        Self::try_from(LocalRegisterCopy::<u64, scause::Register>::new(csr))
    }
}

impl TryFrom<LocalRegisterCopy<u64, scause::Register>> for Trap {
    type Error = Error;

    fn try_from(val: LocalRegisterCopy<u64, scause::Register>) -> Result<Self> {
        if val.is_set(scause::is_interrupt) {
            Ok(Trap::Interrupt(Interrupt::from_scause_reason(
                val.read(scause::reason),
            )?))
        } else {
            Ok(Trap::Exception(Exception::from_scause_reason(
                val.read(scause::reason),
            )?))
        }
    }
}

/// Interrupt causes.
#[derive(Copy, Clone, Debug)]
pub enum Interrupt {
    UserSoft,
    SupervisorSoft,
    VirtualSupervisorSoft,
    MachineSoft,
    UserTimer,
    SupervisorTimer,
    VirtualSupervisorTimer,
    MachineTimer,
    UserExternal,
    SupervisorExternal,
    VirtualSupervisorExternal,
    MachineExternal,
    SupervisorGuestExternal,
}

/// Exception causes.
#[derive(Copy, Clone, Debug)]
pub enum Exception {
    InstructionMisaligned,
    InstructionFault,
    IllegalInstruction,
    Breakpoint,
    LoadMisaligned,
    LoadFault,
    StoreMisaligned,
    StoreFault,
    UserEnvCall,
    SupervisorEnvCall,
    VirtualSupervisorEnvCall,
    MachineEnvCall,
    InstructionPageFault,
    LoadPageFault,
    StorePageFault,
    GuestInstructionPageFault,
    GuestLoadPageFault,
    VirtualInstruction,
    GuestStorePageFault,
}

impl Interrupt {
    pub fn from_scause_reason(val: u64) -> Result<Self> {
        let reason = LocalRegisterCopy::<u64, reason::Register>::new(val);
        match reason.read(reason::std) {
            0 => Ok(Interrupt::UserSoft),
            1 => Ok(Interrupt::SupervisorSoft),
            2 => Ok(Interrupt::VirtualSupervisorSoft),
            3 => Ok(Interrupt::MachineSoft),
            4 => Ok(Interrupt::UserTimer),
            5 => Ok(Interrupt::SupervisorTimer),
            6 => Ok(Interrupt::VirtualSupervisorTimer),
            7 => Ok(Interrupt::MachineTimer),
            8 => Ok(Interrupt::UserExternal),
            9 => Ok(Interrupt::SupervisorExternal),
            10 => Ok(Interrupt::VirtualSupervisorExternal),
            11 => Ok(Interrupt::MachineExternal),
            12 => Ok(Interrupt::SupervisorGuestExternal),
            v => Err(Error::UnknownCause(v)),
        }
    }

    pub fn to_sie_field(&self) -> Result<FieldValue<u64, sie::Register>> {
        match self {
            Interrupt::SupervisorSoft => Ok(sie::ssoft.val(1)),
            Interrupt::SupervisorTimer => Ok(sie::stimer.val(1)),
            Interrupt::SupervisorExternal => Ok(sie::sext.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }

    pub fn to_sip_field(&self) -> Result<FieldValue<u64, sip::Register>> {
        match self {
            Interrupt::SupervisorSoft => Ok(sip::ssoft.val(1)),
            Interrupt::SupervisorTimer => Ok(sip::stimer.val(1)),
            Interrupt::SupervisorExternal => Ok(sip::sext.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }

    pub fn to_hideleg_field(&self) -> Result<FieldValue<u64, hideleg::Register>> {
        match self {
            Interrupt::VirtualSupervisorSoft => Ok(hideleg::vssoft.val(1)),
            Interrupt::VirtualSupervisorTimer => Ok(hideleg::vstimer.val(1)),
            Interrupt::VirtualSupervisorExternal => Ok(hideleg::vsext.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }

    pub fn to_hie_field(&self) -> Result<FieldValue<u64, hie::Register>> {
        match self {
            Interrupt::VirtualSupervisorSoft => Ok(hie::vssoft.val(1)),
            Interrupt::VirtualSupervisorTimer => Ok(hie::vstimer.val(1)),
            Interrupt::VirtualSupervisorExternal => Ok(hie::vsext.val(1)),
            Interrupt::SupervisorGuestExternal => Ok(hie::sgext.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }

    pub fn to_hip_field(&self) -> Result<FieldValue<u64, hip::Register>> {
        match self {
            Interrupt::VirtualSupervisorSoft => Ok(hip::vssoft.val(1)),
            Interrupt::VirtualSupervisorTimer => Ok(hip::vstimer.val(1)),
            Interrupt::VirtualSupervisorExternal => Ok(hip::vsext.val(1)),
            Interrupt::SupervisorGuestExternal => Ok(hip::sgext.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }

    pub fn to_hvip_field(&self) -> Result<FieldValue<u64, hvip::Register>> {
        match self {
            Interrupt::VirtualSupervisorSoft => Ok(hvip::vssoft.val(1)),
            Interrupt::VirtualSupervisorTimer => Ok(hvip::vstimer.val(1)),
            Interrupt::VirtualSupervisorExternal => Ok(hvip::vsext.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }
}

impl Exception {
    pub fn from_scause_reason(val: u64) -> Result<Self> {
        let reason = LocalRegisterCopy::<u64, reason::Register>::new(val);
        match reason.read(reason::std) {
            0 => Ok(Exception::InstructionMisaligned),
            1 => Ok(Exception::InstructionFault),
            2 => Ok(Exception::IllegalInstruction),
            3 => Ok(Exception::Breakpoint),
            4 => Ok(Exception::LoadMisaligned),
            5 => Ok(Exception::LoadFault),
            6 => Ok(Exception::StoreMisaligned),
            7 => Ok(Exception::StoreFault),
            8 => Ok(Exception::UserEnvCall),
            9 => Ok(Exception::SupervisorEnvCall),
            10 => Ok(Exception::VirtualSupervisorEnvCall),
            11 => Ok(Exception::MachineEnvCall),
            12 => Ok(Exception::InstructionPageFault),
            13 => Ok(Exception::LoadPageFault),
            15 => Ok(Exception::StorePageFault),
            20 => Ok(Exception::GuestInstructionPageFault),
            21 => Ok(Exception::GuestLoadPageFault),
            22 => Ok(Exception::VirtualInstruction),
            23 => Ok(Exception::GuestStorePageFault),
            v => Err(Error::UnknownCause(v)),
        }
    }

    pub fn to_hedeleg_field(&self) -> Result<FieldValue<u64, hedeleg::Register>> {
        match self {
            Exception::InstructionMisaligned => Ok(hedeleg::instr_misaligned.val(1)),
            Exception::InstructionFault => Ok(hedeleg::instr_fault.val(1)),
            Exception::IllegalInstruction => Ok(hedeleg::illegal_instr.val(1)),
            Exception::Breakpoint => Ok(hedeleg::breakpoint.val(1)),
            Exception::LoadMisaligned => Ok(hedeleg::load_misaligned.val(1)),
            Exception::LoadFault => Ok(hedeleg::load_fault.val(1)),
            Exception::StoreMisaligned => Ok(hedeleg::store_misaligned.val(1)),
            Exception::StoreFault => Ok(hedeleg::store_fault.val(1)),
            Exception::UserEnvCall => Ok(hedeleg::u_ecall.val(1)),
            Exception::InstructionPageFault => Ok(hedeleg::instr_page_fault.val(1)),
            Exception::LoadPageFault => Ok(hedeleg::load_page_fault.val(1)),
            Exception::StorePageFault => Ok(hedeleg::store_page_fault.val(1)),
            _ => Err(Error::InvalidCause),
        }
    }
}
