// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Functions defined for the State extension
#[derive(Clone, Copy)]
pub enum StateFunction {
    /// Starts the given hart.
    HartStart {
        /// a0 - hart id to start.
        hart_id: u64,
        /// a1 - address to start the hart.
        start_addr: u64,
        /// a2 - value to be set in a1 when starting the hart.
        opaque: u64,
    },
    /// Stops the current hart.
    HartStop,
    /// Returns the status of the given hart.
    HartStatus {
        /// a0 - ID of the hart to check.
        hart_id: u64,
    },
    /// Requests that the calling hart be suspended.
    HartSuspend {
        /// a0 - Specifies the type of suspend to initiate.
        suspend_type: u32,
        /// a1 - The address to jump to on resume.
        resume_addr: u64,
        /// a2 - An opaque value to load in a1 when resuming the hart.
        opaque: u64,
    },
}

/// Return value for the HartStatus SBI call.
#[repr(u64)]
pub enum HartState {
    /// The hart is physically powered-up and executing normally.
    Started = 0,
    /// The hart is not executing in supervisor-mode or any lower privilege mode.
    Stopped = 1,
    /// Some other hart has requested to start, operation still in progress.
    StartPending = 2,
    /// Some other hart has requested to stop, operation still in progress.
    StopPending = 3,
    /// This hart is in a platform specific suspend (or low power) state.
    Suspended = 4,
    /// The hart has requested to put itself in a platform specific low power state, in progress.
    SuspendPending = 5,
    /// An interrupt or platform specific hardware event has caused the hart to resume normal
    /// execution. Resuming is ongoing.
    ResumePending = 6,
}

impl StateFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use StateFunction::*;
        match args[6] {
            0 => Ok(HartStart {
                hart_id: args[0],
                start_addr: args[1],
                opaque: args[2],
            }),
            1 => Ok(HartStop),
            2 => Ok(HartStatus { hart_id: args[0] }),
            3 => Ok(HartSuspend {
                suspend_type: args[0] as u32,
                resume_addr: args[1],
                opaque: args[2],
            }),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for StateFunction {
    fn a6(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart { .. } => 0,
            HartStop => 1,
            HartStatus { .. } => 2,
            HartSuspend { .. } => 3,
        }
    }

    fn a0(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart {
                hart_id,
                start_addr: _,
                opaque: _,
            } => *hart_id,
            HartStatus { hart_id } => *hart_id,
            HartSuspend {
                suspend_type,
                resume_addr: _,
                opaque: _,
            } => *suspend_type as u64,
            _ => 0,
        }
    }

    fn a1(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart {
                hart_id: _,
                start_addr,
                opaque: _,
            } => *start_addr,
            HartSuspend {
                suspend_type: _,
                resume_addr,
                opaque: _,
            } => *resume_addr,
            _ => 0,
        }
    }

    fn a2(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart {
                hart_id: _,
                start_addr: _,
                opaque,
            } => *opaque,
            HartSuspend {
                suspend_type: _,
                resume_addr: _,
                opaque,
            } => *opaque,
            _ => 0,
        }
    }
}
