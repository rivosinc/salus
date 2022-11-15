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
    /// Allows injection of the specified external interrupt ID into the calling TVM vCPU. Passing
    /// an ID of -1 allows injection of all external interrupts. TVM vCPUs are started with
    /// injection of external interrupts completely disabled by default.
    ///
    /// Returns an error if the specified external interrupt ID is invalid.
    ///
    /// a6 = 1
    AllowExternalInterrupt {
        /// a0 = interrupt ID
        id: i64,
    },
    /// Denies injection of the specified external interrupt ID into the calling TVM vCPU. Passing
    /// an ID of -1 denies injection of all external interrupts.
    ///
    /// Returns an error if the specified external interrupt ID is invalid.
    ///
    /// a6 = 2
    DenyExternalInterrupt {
        /// a0 = interrupt ID
        id: i64,
    },
    /// Requests conversion of the specified range of guest physical address space from confidential
    /// to shared. The caller is blocked until the host has completed the invalidation and removal
    /// of any confidential pages that were mapped into the region. Upon return, all accesses by
    /// the TVM within the range are guaranteed to be to shared memory.
    ///
    /// Both `addr` and `len` must be 4kB-aligned, and the range must lie within an existing region
    /// of confidential memory. Returns 0 on success.
    ///
    /// a6 = 3
    ShareMemory {
        /// a0 = start address of the region
        addr: u64,
        /// a1 = length of the region
        len: u64,
    },
    /// Requests conversion of the specified range of guest physical address space from shared to
    /// confidential. The caller is blocked until the host has completed the invalidation and
    /// removal of any shared pages that were mapped into the region. Upon return, all accesses by
    /// the TVM within the range are guaranteed to be to confidential memory.
    ///
    /// Both `addr` and `len` must be 4kB-aligned, and the range must lie within an existing region
    /// of shared memory. Returns 0 on success.
    ///
    /// a6 = 4
    UnshareMemory {
        /// a0 = start address of the region
        addr: u64,
        /// a1 = length of the region
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
            1 => Ok(AllowExternalInterrupt { id: args[0] as i64 }),
            2 => Ok(DenyExternalInterrupt { id: args[0] as i64 }),
            3 => Ok(ShareMemory {
                addr: args[0],
                len: args[1],
            }),
            4 => Ok(UnshareMemory {
                addr: args[0],
                len: args[1],
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
            AllowExternalInterrupt { .. } => 1,
            DenyExternalInterrupt { .. } => 2,
            ShareMemory { .. } => 3,
            UnshareMemory { .. } => 4,
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
            AllowExternalInterrupt { id } => *id as u64,
            DenyExternalInterrupt { id } => *id as u64,
            ShareMemory { addr, len: _ } => *addr,
            UnshareMemory { addr, len: _ } => *addr,
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
            ShareMemory { addr: _, len } => *len,
            UnshareMemory { addr: _, len } => *len,
            _ => 0,
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
            _ => 0,
        }
    }
}
