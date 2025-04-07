// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![allow(missing_docs)]

//! Crate for handling RV64 registers.
//! inst - auto-generated register definitions
//! regs - RV64 General Purpose Registers (GPRs), 0-31.
//! csrs - (H)S-mode CSRs
//! decode - basic RV64 instruction decoding
//! fence - memory fence instructions

mod csrs;
mod decode;
mod fence;
mod inst;
mod regs;

pub use csrs::*;
pub use decode::*;
pub use fence::*;
pub use inst::*;
pub use regs::*;
