// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Rust SBI message parsing.
//! `SbiMessage` is an enum of all the SBI extensions.
//! For each extension, a function enum is defined to contain the SBI function data.
#![no_std]

mod consts;
pub use consts::*;
mod error;
pub use error::*;
mod function;
pub use function::*;
// The Attestation SBI extension
mod attestation;
pub use attestation::*;
// The Base SBI extension
mod base;
pub use base::*;
// The reset SBI extension
mod reset;
pub use reset::*;
// The State SBI extension
mod state;
pub use state::*;
// The TSM SBI extension
mod tsm;
pub use tsm::*;
// The TSM-AIA SBI extension.
mod tsm_aia;
pub use tsm_aia::*;
// The PMU SBI extension
mod pmu;
pub use pmu::*;

/// Interfaces for invoking SBI functionality.
pub mod api;

#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;

/// The values returned from an SBI function call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SbiReturn {
    /// The error code (0 for success).
    pub error_code: i64,
    /// The return value if the operation is successful.
    pub return_value: u64,
}

impl SbiReturn {
    /// Returns an `SbiReturn` that indicates success.
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
            error_code: error as i64,
            return_value: 0,
        }
    }
}

impl From<SbiReturn> for Result<u64> {
    fn from(ret: SbiReturn) -> Result<u64> {
        match ret.error_code {
            SBI_SUCCESS => Ok(ret.return_value),
            e => Err(Error::from_code(e)),
        }
    }
}

/// SBI return value conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SbiReturnType {
    /// Legacy (v0.1) extensions return a single value in A0, usually with the convention that 0
    /// is success and < 0 is an implementation defined error code.
    Legacy(u64),
    /// Modern extensions use the standard error code values enumerated above.
    Standard(SbiReturn),
}

/// SBI Message used to invoke the specified SBI extension in the firmware.
#[derive(Clone, Copy)]
pub enum SbiMessage {
    /// The base SBI extension functions.
    Base(BaseFunction),
    /// The legacy PutChar extension.
    PutChar(u64),
    /// The extension for getting/setting the state of CPUs.
    HartState(StateFunction),
    /// Handles system reset.
    Reset(ResetFunction),
    /// Provides capabilities for starting confidential virtual machines.
    Tee(TeeFunction),
    /// The extension for getting attestation evidences and extending measurements.
    Attestation(AttestationFunction),
    /// Provides interrupt virtualization for confidential virtual machines.
    TeeAia(TeeAiaFunction),
    /// The extension for getting performance counter state.
    Pmu(PmuFunction),
}

impl SbiMessage {
    /// Creates an SbiMessage struct from the given GPRs. Intended for use from the ECALL handler
    /// and passed the saved register state from the calling OS. A7 must contain a valid SBI
    /// extension and the other A* registers will be interpreted based on the extension A7 selects.
    pub fn from_regs(args: &[u64]) -> Result<Self> {
        match args[7] {
            EXT_PUT_CHAR => Ok(SbiMessage::PutChar(args[0])),
            EXT_BASE => BaseFunction::from_regs(args).map(SbiMessage::Base),
            EXT_HART_STATE => StateFunction::from_regs(args).map(SbiMessage::HartState),
            EXT_RESET => ResetFunction::from_regs(args).map(SbiMessage::Reset),
            EXT_TEE => TeeFunction::from_regs(args).map(SbiMessage::Tee),
            EXT_ATTESTATION => AttestationFunction::from_regs(args).map(SbiMessage::Attestation),
            EXT_TEE_AIA => TeeAiaFunction::from_regs(args).map(SbiMessage::TeeAia),
            EXT_PMU => PmuFunction::from_regs(args).map(SbiMessage::Pmu),
            _ => Err(Error::NotSupported),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a7(&self) -> u64 {
        match self {
            SbiMessage::Base(_) => EXT_BASE,
            SbiMessage::PutChar(_) => EXT_PUT_CHAR,
            SbiMessage::HartState(_) => EXT_HART_STATE,
            SbiMessage::Reset(_) => EXT_RESET,
            SbiMessage::Tee(_) => EXT_TEE,
            SbiMessage::Attestation(_) => EXT_ATTESTATION,
            SbiMessage::TeeAia(_) => EXT_TEE_AIA,
            SbiMessage::Pmu(_) => EXT_PMU,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a6(&self) -> u64 {
        match self {
            SbiMessage::Base(_) => 0, //TODO
            SbiMessage::HartState(f) => f.a6(),
            SbiMessage::PutChar(_) => 0,
            SbiMessage::Reset(_) => 0,
            SbiMessage::Tee(f) => f.a6(),
            SbiMessage::Attestation(f) => f.a6(),
            SbiMessage::TeeAia(f) => f.a6(),
            SbiMessage::Pmu(f) => f.a6(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a5(&self) -> u64 {
        match self {
            SbiMessage::Tee(f) => f.a5(),
            SbiMessage::TeeAia(f) => f.a5(),
            SbiMessage::Pmu(f) => f.a5(),
            SbiMessage::Attestation(f) => f.a5(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a4(&self) -> u64 {
        match self {
            SbiMessage::Tee(f) => f.a4(),
            SbiMessage::TeeAia(f) => f.a4(),
            SbiMessage::Pmu(f) => f.a4(),
            SbiMessage::Attestation(f) => f.a4(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a3(&self) -> u64 {
        match self {
            SbiMessage::Tee(f) => f.a3(),
            SbiMessage::Attestation(f) => f.a3(),
            SbiMessage::Pmu(f) => f.a3(),
            SbiMessage::TeeAia(f) => f.a3(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a2(&self) -> u64 {
        match self {
            SbiMessage::HartState(f) => f.a2(),
            SbiMessage::Tee(f) => f.a2(),
            SbiMessage::Attestation(f) => f.a2(),
            SbiMessage::Pmu(f) => f.a2(),
            SbiMessage::TeeAia(f) => f.a2(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a1(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.a1(),
            SbiMessage::HartState(f) => f.a1(),
            SbiMessage::Tee(f) => f.a1(),
            SbiMessage::Attestation(f) => f.a1(),
            SbiMessage::TeeAia(f) => f.a1(),
            SbiMessage::Pmu(f) => f.a1(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a0(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.a0(),
            SbiMessage::PutChar(c) => *c,
            SbiMessage::HartState(f) => f.a0(),
            SbiMessage::Tee(f) => f.a0(),
            SbiMessage::Attestation(f) => f.a0(),
            SbiMessage::TeeAia(f) => f.a0(),
            SbiMessage::Pmu(f) => f.a0(),
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
        let ret = SbiReturn {
            error_code: a0 as i64,
            return_value: a1,
        };
        match self {
            // For legacy messages, a0 is 0 on success and an implementation-defined error value on
            // failure. Nothing is returned in a1.
            SbiMessage::PutChar(_) => match a0 as i64 {
                SBI_SUCCESS => Ok(0),
                _ => Err(Error::Failed),
            },
            _ => ret.into(),
        }
    }
}

/// Send an ecall to the firmware or hypervisor.
///
/// # Safety
///
/// The caller must verify that any memory references contained in `msg` obey rust's memory
/// safety rules. For example, any pointers to memory that will be modified in the handling of
/// the ecall must be uniquely owned. Similarly any pointers read by the ecall must not be
/// mutably borrowed.
///
/// In addition the caller is placing trust in the firmware or hypervisor to maintain the promises
/// of the interface w.r.t. reading and writing only within the provided bounds.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub unsafe fn ecall_send(msg: &SbiMessage) -> Result<u64> {
    // normally error code
    let mut a0;
    // normally return value
    let mut a1;
    asm!("ecall", inlateout("a0") msg.a0()=>a0, inlateout("a1")msg.a1()=>a1,
                in("a2")msg.a2(), in("a3") msg.a3(),
                in("a4")msg.a4(), in("a5") msg.a5(),
                in("a6")msg.a6(), in("a7") msg.a7(), options(nostack));

    msg.result(a0, a1)
}

#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
unsafe fn ecall_send(_msg: &SbiMessage) -> Result<u64> {
    panic!("ecall_send called");
}
