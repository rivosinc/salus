// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;

/// A Trait for an SbiFunction. Implementers use this trait to specify how to parse from and
/// serialize into the a0-a7 registers used to make SBI calls.
pub trait SbiFunction {
    /// Returns the `u64` value that should be stored in register a6 before making the ecall for
    /// this function.
    fn a6(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a5 before making the ecall for
    /// this function.
    fn a5(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a4 before making the ecall for
    /// this function.
    fn a4(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a3 before making the ecall for
    /// this function.
    fn a3(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a2 before making the ecall for
    /// this function.
    fn a2(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a1 before making the ecall for
    /// this function.
    fn a1(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a0 before making the ecall for
    /// this function.
    fn a0(&self) -> u64 {
        0
    }
    /// Returns a result parsed from the a0 and a1 return value registers.
    fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        match a0 {
            0 => Ok(a1),
            e => Err(Error::from_code(e as i64)),
        }
    }
}
