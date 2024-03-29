// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

#![no_std]

//! # Salus U-mode API.
//!
//! This library contains data structures that are passed between
//! hypervisor and user mode.
//!
//! All data is passed through `hypcalls`, calls from U-mode to
//! hypervisor, implemented using an `ecall`/`sret` pair.
//!
//! `hypcalls` originate in user mode. They are used to ask the
//! hypervisor for specific services or for signalling end of
//! execution.
//!
//! There are two ways to pass data between the two components:
//! registers and memory.
//!
//! ## Passing Data through Registers.
//!
//! During `ecall` or `sret`, registers `A0`-`A7` (A-registers) are
//! used to pass information between the two components.
//!
//! Details of the specific `hypcall` to run are specified in the
//! A-registers at the moment of the `ecall`. If the `hypcalls`
//! implementation in the hypervisor returns some result then, when
//! `sret` is executed, the A-registers will contain the information
//! returned.
//!
//! This library defines two traits, `IntoRegisters` and
//! `TryIntoRegisters`, that must implemented to allow a type to be
//! passed through registers.
//!
//! ## Passing Data through Memory.
//!
//! TBD.
//!
//! ## Entry of user mode.
//!
//! The user mode process expect to be run from the entry point
//! specified in the ELF file with register `A0` containing a unique
//! u64 ID (the CPU ID).

/// Attestation-related data structures.
pub mod cert;

/// The Error type returned returned from this library.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum Error {
    /// Generic failure in execution.
    Failed = 1,
    /// Invalid arguments passed.
    InvalidArgument = 2,
    /// Ecall not supported. From hypervisor to umode.
    EcallNotSupported = 3,
    /// Request not supported. From umode to hypervisor.
    RequestNotSupported = 4,
}

impl From<u64> for Error {
    fn from(val: u64) -> Error {
        match val {
            1 => Error::Failed,
            2 => Error::InvalidArgument,
            3 => Error::EcallNotSupported,
            4 => Error::RequestNotSupported,
            _ => Error::Failed,
        }
    }
}

// All types that can be passed in registers must implement `IntoRegisters` or `TryIntoRegisters`.

/// Trait to transform a type into A-registers when a set of registers will always transform into
/// this type.
pub trait IntoRegisters {
    /// Get current type from a set of registers.
    fn from_registers(regs: &[u64]) -> Self;
    /// Write `self` into a set of registers.
    fn to_registers(&self, regs: &mut [u64]);
}

/// Trait to transform a type into A-registers when a set of registers might not be able to be
/// transformed into this type, returning an error.
pub trait TryIntoRegisters: Sized {
    /// Get current type from a set of registers or return an error.
    fn try_from_registers(regs: &[u64]) -> Result<Self, Error>;
    /// Write `self` into a set of registers.
    fn to_registers(&self, regs: &mut [u64]);
}

// Result<u64, Error> is passed through registers. Implement trait.

// Error code for success.
const HYPC_SUCCESS: u64 = 0;

impl IntoRegisters for Result<u64, Error> {
    fn from_registers(regs: &[u64]) -> Result<u64, Error> {
        match regs[0] {
            HYPC_SUCCESS => Ok(regs[1]),
            e => Err(e.into()),
        }
    }

    fn to_registers(&self, regs: &mut [u64]) {
        match self {
            Ok(val) => {
                regs[0] = HYPC_SUCCESS;
                regs[1] = *val;
            }
            Err(e) => {
                regs[0] = *e as u64;
            }
        }
    }
}

// UmodeRequest: calls from hypervisor to Umode requesting an operation.

/// An operation requested by the hypervisor and executed by umode.
#[derive(Debug)]
pub enum UmodeRequest {
    /// Do nothing.
    Nop,
    /// Get Attestation Evidence.
    ///
    /// Umode Input Region: contains `MeasurementRegisters`.
    GetEvidence {
        /// starting address of the Certificate Signing Request.
        csr_addr: u64,
        /// size of the Certificate Signing Request.
        csr_len: usize,
        /// starting address of the output Certificate.
        certout_addr: u64,
        /// size of the output Certificate.
        certout_len: usize,
    },
}

// Mappings of A0 register to U-mode operation.
const UMOP_NOP: u64 = 0;
const UMOP_GET_EVIDENCE: u64 = 1;

impl TryIntoRegisters for UmodeRequest {
    fn try_from_registers(regs: &[u64]) -> Result<UmodeRequest, Error> {
        match regs[0] {
            UMOP_NOP => Ok(UmodeRequest::Nop),
            UMOP_GET_EVIDENCE => Ok(UmodeRequest::GetEvidence {
                csr_addr: regs[1],
                csr_len: regs[2] as usize,
                certout_addr: regs[3],
                certout_len: regs[3] as usize,
            }),
            _ => Err(Error::RequestNotSupported),
        }
    }

    fn to_registers(&self, regs: &mut [u64]) {
        match *self {
            UmodeRequest::Nop => {
                regs[0] = UMOP_NOP;
            }
            UmodeRequest::GetEvidence {
                csr_addr,
                csr_len,
                certout_addr,
                certout_len,
            } => {
                regs[0] = UMOP_GET_EVIDENCE;
                regs[1] = csr_addr;
                regs[2] = csr_len as u64;
                regs[3] = certout_addr;
                regs[4] = certout_len as u64;
            }
        }
    }
}

// HypCall: calls from umode to hypervisor.

/// Result returned from Umode Request execution (U-mode -> Hypervisor)
pub type OpResult = Result<u64, Error>;

/// CDI Selector. This lists all possible CDis presents in the attestation manager.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum CdiSel {
    /// Attestation Current CDI.
    AttestationCurrent = 0,
    /// Attestation Next CDI.
    AttestationNext = 1,
    /// Sealing Current CDI.
    SealingCurrent = 2,
    /// Sealing Next CDI.
    SealingNext = 3,
}

impl TryFrom<u64> for CdiSel {
    type Error = Error;

    fn try_from(val: u64) -> Result<CdiSel, Error> {
        match val {
            0 => Ok(CdiSel::AttestationCurrent),
            1 => Ok(CdiSel::AttestationNext),
            2 => Ok(CdiSel::SealingCurrent),
            3 => Ok(CdiSel::SealingNext),
            _ => Err(Error::InvalidArgument),
        }
    }
}

/// Maximum size of a message to sign.
pub const CDIOP_SIGN_MAXMSG: usize = 2048;

/// CDI Operation
#[derive(Debug, Clone, Copy)]
pub enum CdiOp {
    /// CDI ID.
    Id {
        /// Address where ID will be stored.
        idout_addr: u64,
        /// Length of the buffer for storing the ID.
        idout_len: u64,
    },
    /// Sign a message with the CDI private key.
    Sign {
        /// Address where the message to be sign starts.
        msg_addr: u64,
        /// Length of the message to be signed.
        msg_len: u64,
        /// Address where the signature will be stored.
        signout_addr: u64,
        /// Length of the buffer for storing the signature.
        signout_len: u64,
    },
}

const HYPC_CDI_ID: u64 = 0;
const HYPC_CDI_SIGN: u64 = 1;

impl TryIntoRegisters for CdiOp {
    fn try_from_registers(regs: &[u64]) -> Result<Self, Error> {
        match regs[0] {
            HYPC_CDI_ID => Ok(CdiOp::Id {
                idout_addr: regs[1],
                idout_len: regs[2],
            }),
            HYPC_CDI_SIGN => Ok(CdiOp::Sign {
                msg_addr: regs[1],
                msg_len: regs[2],
                signout_addr: regs[3],
                signout_len: regs[4],
            }),
            _ => Err(Error::EcallNotSupported),
        }
    }

    fn to_registers(&self, regs: &mut [u64]) {
        match *self {
            CdiOp::Id {
                idout_addr,
                idout_len,
            } => {
                regs[0] = HYPC_CDI_ID;
                regs[1] = idout_addr;
                regs[2] = idout_len;
            }
            CdiOp::Sign {
                msg_addr,
                msg_len,
                signout_addr,
                signout_len,
            } => {
                regs[0] = HYPC_CDI_SIGN;
                regs[1] = msg_addr;
                regs[2] = msg_len;
                regs[3] = signout_addr;
                regs[4] = signout_len;
            }
        }
    }
}

/// Calls from umode to the hypervisors.
#[derive(Debug)]
pub enum HypCall {
    /// Panic and exit immediately.
    Panic,
    /// Print a character for debug.
    PutChar(u8),
    /// Return result of previous request and wait for next operation.
    NextOp(OpResult),
    /// Attestation Manager CDI operations.
    Cdi {
        /// CDI Selector.
        cdi_sel: CdiSel,
        /// CDI Operation.
        cdi_op: CdiOp,
    },
}

const HYPC_PANIC: u64 = 0;
const HYPC_PUTCHAR: u64 = 1;
const HYPC_NEXTOP: u64 = 2;
const HYPC_CDI: u64 = 0x10;

impl TryIntoRegisters for HypCall {
    fn try_from_registers(regs: &[u64]) -> Result<Self, Error> {
        match regs[7] {
            HYPC_PANIC => Ok(HypCall::Panic),
            HYPC_PUTCHAR => Ok(HypCall::PutChar(regs[0] as u8)),
            HYPC_NEXTOP => Ok(HypCall::NextOp(Result::from_registers(regs))),
            HYPC_CDI => Ok(HypCall::Cdi {
                cdi_sel: regs[6].try_into()?,
                cdi_op: CdiOp::try_from_registers(regs)?,
            }),
            _ => Err(Error::EcallNotSupported),
        }
    }

    fn to_registers(&self, regs: &mut [u64]) {
        match *self {
            HypCall::Panic => {
                regs[7] = HYPC_PANIC;
            }
            HypCall::PutChar(byte) => {
                regs[0] = byte as u64;
                regs[7] = HYPC_PUTCHAR;
            }
            HypCall::NextOp(result) => {
                result.to_registers(regs);
                regs[7] = HYPC_NEXTOP;
            }
            HypCall::Cdi { cdi_sel, cdi_op } => {
                cdi_op.to_registers(regs);
                regs[6] = cdi_sel as u64;
                regs[7] = HYPC_CDI;
            }
        }
    }
}
