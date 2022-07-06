// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![feature(asm_const)]
#![allow(missing_docs)]

//! Crate for handling RV64 registers.
//! inst - auto-generated register definitions
//! regs - RV64 General Purpose Registers (GPRs), 0-31.
//! csrs - (H)S-mode CSRs
//! decode - basic RV64 instruction decoding

mod csrs;
mod decode;
mod inst;
mod regs;

pub use csrs::*;
pub use decode::*;
pub use inst::*;
pub use regs::*;
