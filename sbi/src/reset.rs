// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Functions for the Reset extension
#[derive(Copy, Clone)]
pub enum ResetFunction {
    /// Performs a system reset.
    Reset {
        /// Determines the type of reset to perform.
        reset_type: ResetType,
        /// Represents the reason for system reset.
        reason: ResetReason,
    },
}

/// The types of reset a supervisor can request.
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResetType {
    /// Powers down the system.
    Shutdown = 0,
    /// Powers down, then reboots.
    ColdReset = 1,
    /// Reboots, doesn't power down.
    WarmReset = 2,
}

impl ResetType {
    // Creates a reset type from the a0 register value or returns an error if no mapping is
    // known for the given value.
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

/// Reasons why a supervisor requests a reset.
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResetReason {
    /// Used for normal resets.
    NoReason = 0,
    /// Used when the system has failed.
    SystemFailure = 1,
}

impl ResetReason {
    // Creates a reset reason from the a1 register value or returns an error if no mapping is
    // known for the given value.
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
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use ResetFunction::*;

        Ok(match args[6] {
            0 => Reset {
                reset_type: ResetType::from_reg(args[0])?,
                reason: ResetReason::from_reg(args[1])?,
            },
            _ => return Err(Error::NotSupported),
        })
    }

    /// Creates an operation to shutdown the machine.
    pub fn shutdown() -> Self {
        ResetFunction::Reset {
            reset_type: ResetType::Shutdown,
            reason: ResetReason::NoReason,
        }
    }
}

impl SbiFunction for ResetFunction {
    fn a0(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type: _,
                reason,
            } => *reason as u64,
        }
    }

    fn a1(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type,
                reason: _,
            } => *reset_type as u64,
        }
    }
}
