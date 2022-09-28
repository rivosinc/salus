// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;
use crate::TeeMemoryRegion;

/// Functions provided by the TEE Guest extension to TVM guests.
#[derive(Copy, Clone, Debug)]
pub enum TeeGuestFunction {
    /// Adds a memory region to the calling TVM at the specified range of guest physical address
    /// space. Both `addr` and `len` must be 4kB-aligned and must not overlap with any
    /// previously-added regions.
    ///
    /// Only `Shared` and `EmulatedMmio` regions may be added by the TVM.
    ///
    /// a6 = 0
    AddMemoryRegion {
        /// a0 = type of memory region
        region_type: TeeMemoryRegion,
        /// a1 = start of the region
        addr: u64,
        /// a2 = length of the region
        len: u64,
    },
}

impl TeeGuestFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use TeeGuestFunction::*;
        match args[6] {
            0 => Ok(AddMemoryRegion {
                region_type: TeeMemoryRegion::from_reg(args[0])?,
                addr: args[1],
                len: args[2],
            }),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for TeeGuestFunction {
    fn a6(&self) -> u64 {
        use TeeGuestFunction::*;
        match self {
            AddMemoryRegion { .. } => 0,
        }
    }

    fn a0(&self) -> u64 {
        use TeeGuestFunction::*;
        match self {
            AddMemoryRegion {
                region_type,
                addr: _,
                len: _,
            } => *region_type as u64,
        }
    }

    fn a1(&self) -> u64 {
        use TeeGuestFunction::*;
        match self {
            AddMemoryRegion {
                region_type: _,
                addr,
                len: _,
            } => *addr,
        }
    }

    fn a2(&self) -> u64 {
        use TeeGuestFunction::*;
        match self {
            AddMemoryRegion {
                region_type: _,
                addr: _,
                len,
            } => *len,
        }
    }
}
