// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Functions for the Debug Console extension
#[derive(Copy, Clone, Debug)]
pub enum DebugConsoleFunction {
    /// Prints the given string to the system console.
    PutString {
        /// The length of the string to print.
        len: u64,
        /// The address of the string.
        addr: u64,
    },
}

impl DebugConsoleFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        Ok(match args[6] {
            0 => DebugConsoleFunction::PutString {
                len: args[0],
                addr: args[1],
            },
            _ => return Err(Error::NotSupported),
        })
    }
}

impl SbiFunction for DebugConsoleFunction {
    fn a0(&self) -> u64 {
        match self {
            DebugConsoleFunction::PutString { len, addr: _ } => *len,
        }
    }

    fn a1(&self) -> u64 {
        match self {
            DebugConsoleFunction::PutString { len: _, addr } => *addr,
        }
    }
}
