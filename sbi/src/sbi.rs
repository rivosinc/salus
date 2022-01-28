// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Rust SBI message parsing.
//! `SbiMessage` is an enum of all the SBI extensions.
//! For each extension, a function enum is defined to contain the SBI function data.
#![no_std]

use riscv_regs::{GeneralPurposeRegisters, GprIndex};

const EXT_PUT_CHAR: u64 = 0x01;
const EXT_BASE: u64 = 0x10;
const EXT_HART_STATE: u64 = 0x48534D;
const EXT_RESET: u64 = 0x53525354;

/// Error constants from the sbi [spec](https://github.com/riscv-non-isa/riscv-sbi-doc/releases)
pub const SBI_SUCCESS: i64 = 0;
pub const SBI_ERR_FAILED: i64 = -1;
pub const SBI_ERR_NOT_SUPPORTED: i64 = -2;
pub const SBI_ERR_INVALID_PARAM: i64 = -3;
pub const SBI_ERR_DENIED: i64 = -4;
pub const SBI_ERR_INVALID_ADDRESS: i64 = -5;
pub const SBI_ERR_ALREADY_AVAILABLE: i64 = -6;
pub const SBI_ERR_ALREADY_STARTED: i64 = -7;
pub const SBI_ERR_ALREADY_STOPPED: i64 = -8;

/// Errors passed over the SBI protocol
#[derive(Debug)]
pub enum Error {
    InvalidAddress,
    InvalidParam,
    Failed,
    NotSupported,
    UnknownSbiExtension,
}

impl Error {
    /// Parse the given error code to an `Error` enum.
    pub fn from_code(e: i64) -> Self {
        use Error::*;
        match e {
            SBI_ERR_INVALID_ADDRESS => InvalidAddress,
            SBI_ERR_INVALID_PARAM => InvalidParam,
            SBI_ERR_NOT_SUPPORTED => NotSupported,
            _ => Failed,
        }
    }

    /// Convert `Self` to a 64bit error code to be returned over SBI.
    pub fn to_code(&self) -> i64 {
        use Error::*;
        match self {
            InvalidAddress => SBI_ERR_INVALID_ADDRESS,
            InvalidParam => SBI_ERR_INVALID_PARAM,
            Failed => SBI_ERR_FAILED,
            NotSupported => SBI_ERR_NOT_SUPPORTED,
            UnknownSbiExtension => SBI_ERR_INVALID_PARAM,
        }
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// Functions defined for the Base extension
pub enum BaseFunction {
    GetSpecificationVersion,
    GetImplementationID,
    GetImplementationVersion,
    GetMachineVendorID,
    GetMachineArchitectureID,
    GetMachineImplementationID,
}

impl BaseFunction {
    fn from_func_id(a6: u64) -> Result<Self> {
        use BaseFunction::*;

        Ok(match a6 {
            0 => GetSpecificationVersion,
            1 => GetImplementationID,
            2 => GetImplementationVersion,
            3 => GetMachineVendorID,
            4 => GetMachineArchitectureID,
            5 => GetMachineImplementationID,
            _ => return Err(Error::InvalidParam),
        })
    }
}

/// Functions defined for the State extension
pub enum StateFunction {
    HartStart,
    HartStop,
    HartStatus,
    HartSuspend,
}

impl StateFunction {
    fn from_func_id(a6: u64) -> Result<Self> {
        use StateFunction::*;

        Ok(match a6 {
            0 => HartStart,
            1 => HartStop,
            2 => HartStatus,
            3 => HartSuspend,
            _ => return Err(Error::InvalidParam),
        })
    }
}

/// Funcions for the Reset extension
#[derive(Copy, Clone)]
pub enum ResetFunction {
    Reset {
        reset_type: ResetType,
        reason: ResetReason,
    },
}

#[derive(Copy, Clone)]
pub enum ResetType {
    Shutdown,
    ColdReset,
    WarmReset,
}

impl ResetType {
    fn from_reg(a0: u64) -> Result<Self> {
        use ResetType::*;
        Ok(match a0 {
            0 => Shutdown,
            1 => ColdReset,
            2 => WarmReset,
            _ => return Err(Error::InvalidParam),
        })
    }
}

#[derive(Copy, Clone)]
pub enum ResetReason {
    NoReason,
    SystemFailure,
}

impl ResetReason {
    fn from_reg(a1: u64) -> Result<Self> {
        use ResetReason::*;
        Ok(match a1 {
            0 => NoReason,
            2 => SystemFailure,
            _ => return Err(Error::InvalidParam),
        })
    }
}

impl ResetFunction {
    pub fn shutdown() -> Self {
        ResetFunction::Reset {
            reset_type: ResetType::Shutdown,
            reason: ResetReason::NoReason,
        }
    }

    fn from_regs(a6: u64, a0: u64, a1: u64) -> Result<Self> {
        use ResetFunction::*;

        Ok(match a6 {
            0 => Reset {
                reset_type: ResetType::from_reg(a0)?,
                reason: ResetReason::from_reg(a1)?,
            },
            _ => return Err(Error::InvalidParam),
        })
    }

    fn get_a0(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type: _,
                reason,
            } => *reason as u64,
        }
    }

    fn get_a1(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type,
                reason: _,
            } => *reset_type as u64,
        }
    }
}

/// Return value for an SBI call.
pub struct SbiReturn {
    pub error_code: i64,
    pub return_value: u64,
}

impl SbiReturn {
    pub fn success(return_value: u64) -> Self {
        Self {
            error_code: SBI_SUCCESS,
            return_value,
        }
    }
}

impl From<Result<u64>> for SbiReturn {
    fn from(result: Result<u64>) -> SbiReturn {
        match result {
            Ok(rv) => Self::success(rv),
            Err(e) => Self::from(e),
        }
    }
}

impl From<Error> for SbiReturn {
    fn from(error: Error) -> SbiReturn {
        SbiReturn {
            error_code: error.to_code(),
            return_value: 0,
        }
    }
}

/// SBI Message used to invoke the specified SBI extension in the firmware.
pub enum SbiMessage {
    Base(BaseFunction),
    PutChar(u64),
    HartState(StateFunction),
    Reset(ResetFunction),
}

impl SbiMessage {
    /// Creates an SbiMessage struct from the given GPRs. Intended for use from the ECALL handler
    /// and passed the saved register state from the calling OS. A7 must contain a valid SBI
    /// extension and the other A* registers will be interpreted based on the extension A7 selects.
    pub fn from_regs(gprs: &GeneralPurposeRegisters) -> Result<Self> {
        use GprIndex::*;
        match gprs.reg(A7) {
            EXT_PUT_CHAR => Ok(SbiMessage::PutChar(gprs.reg(A0))),
            EXT_BASE => BaseFunction::from_func_id(gprs.reg(A6)).map(SbiMessage::Base),
            EXT_HART_STATE => StateFunction::from_func_id(gprs.reg(A6)).map(SbiMessage::HartState),
            EXT_RESET => ResetFunction::from_regs(gprs.reg(A6), gprs.reg(A0), gprs.reg(A1))
                .map(SbiMessage::Reset),
            _ => Err(Error::UnknownSbiExtension),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a7(&self) -> u64 {
        match self {
            SbiMessage::Base(_) => EXT_BASE,
            SbiMessage::PutChar(_) => EXT_PUT_CHAR,
            SbiMessage::HartState(_) => EXT_HART_STATE,
            SbiMessage::Reset(_) => EXT_RESET,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a6(&self) -> u64 {
        match self {
            SbiMessage::Base(_) => 0,      //TODO
            SbiMessage::HartState(_) => 0, //TODO
            SbiMessage::PutChar(_) => 0,
            SbiMessage::Reset(_) => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a5(&self) -> u64 {
        match self {
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a4(&self) -> u64 {
        match self {
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a3(&self) -> u64 {
        match self {
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a2(&self) -> u64 {
        match self {
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a1(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.get_a1(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a0(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.get_a0(),
            SbiMessage::PutChar(c) => *c,
            _ => 0,
        }
    }

    /// Returns the result returned in the SbiMessage. Intended for use after an SbiMessage has been
    /// handled by the firmware. Interprets the given registers based on the extension and function
    /// and returns the approprate result.
    ///
    /// # Example
    ///
    /// ```rust
    /// pub fn ecall_send(msg: &SbiMessage) -> Result<u64> {
    ///     let mut a0 = msg.a0(); // error code
    ///     let mut a1 = msg.a1(); // return value
    ///     unsafe {
    ///         // Safe, but relies on trusting the hypervisor or firmware.
    ///         asm!("ecall", inout("a0") a0, inout("a1")a1,
    ///                 in("a2")msg.a2(), in("a3") msg.a3(),
    ///                 in("a4")msg.a4(), in("a5") msg.a5(),
    ///                 in("a6")msg.a6(), in("a7") msg.a7());
    ///     }
    ///
    ///     msg.result(a0, a1)
    /// }
    /// ```
    pub fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        match self {
            SbiMessage::Base(_) => {
                if a0 == 0 {
                    Ok(a1)
                } else {
                    Err(Error::InvalidParam) // TODO - set error
                }
            } //TODO
            SbiMessage::HartState(_) => Ok(a1), //TODO
            SbiMessage::PutChar(_) => Ok(0),
            SbiMessage::Reset(_) => Err(Error::InvalidParam),
        }
    }
}
