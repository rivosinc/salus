// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Rust SBI message parsing.
//! `SbiMessage` is an enum of all the SBI extensions.
//! For each extension, a function enum is defined to contain the SBI function data.
#![no_std]

mod consts;
pub use consts::*;
mod debug_console;
pub use debug_console::*;
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
// The Nested Virtualization Acceleration (NACL) SBI extension
mod nacl;
pub use nacl::*;
// The reset SBI extension
mod reset;
pub use reset::*;
// The State SBI extension
mod state;
pub use state::*;
// The TEE host SBI extension
mod tee_host;
pub use tee_host::*;
// The TEE interrupt SBI extension
mod tee_interrupt;
pub use tee_interrupt::*;
// The TEE guest SBI extension
mod tee_guest;
pub use tee_guest::*;
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
#[derive(Clone, Copy, Debug)]
pub enum SbiMessage {
    /// The base SBI extension functions.
    Base(BaseFunction),
    /// The legacy PutChar extension.
    PutChar(u64),
    /// The extension for getting/setting the state of CPUs.
    HartState(StateFunction),
    /// Handles system reset.
    Reset(ResetFunction),
    /// Handles output to the console for debug.
    DebugConsole(DebugConsoleFunction),
    /// Provides functions for accelerating nested virtualization.
    Nacl(NaclFunction),
    /// Provides capabilities for starting confidential virtual machines.
    TeeHost(TeeHostFunction),
    /// Provides interrupt virtualization for confidential virtual machines.
    TeeInterrupt(TeeInterruptFunction),
    /// Provides capabilities for enlightened confidential virtual machines.
    TeeGuest(TeeGuestFunction),
    /// The extension for getting attestation evidences and extending measurements.
    Attestation(AttestationFunction),
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
            EXT_DBCN => DebugConsoleFunction::from_regs(args).map(SbiMessage::DebugConsole),
            EXT_NACL => NaclFunction::from_regs(args).map(SbiMessage::Nacl),
            EXT_TEE_HOST => TeeHostFunction::from_regs(args).map(SbiMessage::TeeHost),
            EXT_TEE_INTERRUPT => {
                TeeInterruptFunction::from_regs(args).map(SbiMessage::TeeInterrupt)
            }
            EXT_TEE_GUEST => TeeGuestFunction::from_regs(args).map(SbiMessage::TeeGuest),
            EXT_ATTESTATION => AttestationFunction::from_regs(args).map(SbiMessage::Attestation),
            EXT_PMU => PmuFunction::from_regs(args).map(SbiMessage::Pmu),
            _ => Err(Error::NotSupported),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a7(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => EXT_PUT_CHAR,
            Base(_) => EXT_BASE,
            HartState(_) => EXT_HART_STATE,
            Reset(_) => EXT_RESET,
            DebugConsole(_) => EXT_DBCN,
            Nacl(_) => EXT_NACL,
            TeeHost(_) => EXT_TEE_HOST,
            TeeInterrupt(_) => EXT_TEE_INTERRUPT,
            TeeGuest(_) => EXT_TEE_GUEST,
            Attestation(_) => EXT_ATTESTATION,
            Pmu(_) => EXT_PMU,
        }
    }

    // TODO: Consider using enum_dispatch to avoid the repetition.

    /// Returns the register value for this `SbiMessage`.
    pub fn a6(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => 0,
            Base(f) => f.a6(),
            HartState(f) => f.a6(),
            Reset(f) => f.a6(),
            DebugConsole(f) => f.a6(),
            Nacl(f) => f.a6(),
            TeeHost(f) => f.a6(),
            TeeInterrupt(f) => f.a6(),
            TeeGuest(f) => f.a6(),
            Attestation(f) => f.a6(),
            Pmu(f) => f.a6(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a5(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => 0,
            Base(f) => f.a5(),
            HartState(f) => f.a5(),
            Reset(f) => f.a5(),
            DebugConsole(f) => f.a5(),
            Nacl(f) => f.a5(),
            TeeHost(f) => f.a5(),
            TeeInterrupt(f) => f.a5(),
            TeeGuest(f) => f.a5(),
            Attestation(f) => f.a5(),
            Pmu(f) => f.a5(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a4(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => 0,
            Base(f) => f.a4(),
            HartState(f) => f.a4(),
            Reset(f) => f.a4(),
            DebugConsole(f) => f.a4(),
            Nacl(f) => f.a4(),
            TeeHost(f) => f.a4(),
            TeeInterrupt(f) => f.a4(),
            TeeGuest(f) => f.a4(),
            Attestation(f) => f.a4(),
            Pmu(f) => f.a4(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a3(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => 0,
            Base(f) => f.a3(),
            HartState(f) => f.a3(),
            Reset(f) => f.a3(),
            DebugConsole(f) => f.a3(),
            Nacl(f) => f.a3(),
            TeeHost(f) => f.a3(),
            TeeInterrupt(f) => f.a3(),
            TeeGuest(f) => f.a3(),
            Attestation(f) => f.a3(),
            Pmu(f) => f.a3(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a2(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => 0,
            Base(f) => f.a2(),
            HartState(f) => f.a2(),
            Reset(f) => f.a2(),
            DebugConsole(f) => f.a2(),
            Nacl(f) => f.a2(),
            TeeHost(f) => f.a2(),
            TeeInterrupt(f) => f.a2(),
            TeeGuest(f) => f.a2(),
            Attestation(f) => f.a2(),
            Pmu(f) => f.a2(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a1(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(_) => 0,
            Base(f) => f.a1(),
            HartState(f) => f.a1(),
            Reset(f) => f.a1(),
            DebugConsole(f) => f.a1(),
            Nacl(f) => f.a1(),
            TeeHost(f) => f.a1(),
            TeeInterrupt(f) => f.a1(),
            TeeGuest(f) => f.a1(),
            Attestation(f) => f.a1(),
            Pmu(f) => f.a1(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a0(&self) -> u64 {
        use SbiMessage::*;
        match self {
            PutChar(c) => *c,
            Base(f) => f.a0(),
            Reset(f) => f.a0(),
            DebugConsole(f) => f.a0(),
            HartState(f) => f.a0(),
            Nacl(f) => f.a0(),
            TeeHost(f) => f.a0(),
            TeeInterrupt(f) => f.a0(),
            TeeGuest(f) => f.a0(),
            Attestation(f) => f.a0(),
            Pmu(f) => f.a0(),
        }
    }

    /// Returns the result returned in the SbiMessage. Intended for use after an SbiMessage has been
    /// handled by the firmware. Interprets the given registers based on the extension and function
    /// and returns the approprate result.
    ///
    /// # Example
    ///
    /// ```rust
    /// #[cfg(all(target_arch = "riscv64", target_os = "none"))]
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
/// The caller must verify that any memory references contained in `msg` obey Rust's memory
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
