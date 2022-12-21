// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Number of bytes in the `NaclShmem` scratch area.
pub const NACL_SCRATCH_BYTES: usize = 2048;

/// Layout of the shared-memory area registered with `SetShmem`.
pub struct NaclShmem {
    /// Scratch space. The layout of this scratch space is defined by the particular function being
    /// invoked.
    ///
    /// For the `TvmCpuRun` function in the TEE-Host extension, the layout of this scratch space
    /// matches the `TsmShmemScratch` struct.
    pub scratch: [u64; NACL_SCRATCH_BYTES / 8],
    _reserved: [u64; 240],
    /// Bitmap indicating which CSRs in `csrs` the host wishes to sync.
    ///
    /// Currently unused in the TEE-related extensions and will not be read or written by the TSM.
    pub dirty_bitmap: [u64; 16],
    /// Hypervisor and virtual-supervisor CSRs. The 12-bit CSR number is transformed into a 10-bit
    /// index by extracting bits `{csr[11:10], csr[8:0]}` since `csr[9:8]` is always 2'b10 for HS
    /// and VS CSRs.
    ///
    /// These CSRs may be updated by `TvmCpuRun` in the TEE-Host extension. See the documentation
    /// of `TvmCpuRun` for more detials.
    pub csrs: [u64; 1024],
}

impl NaclShmem {
    /// Returns the index in `csrs` of the HS or VS CSR at `csr_num`.
    pub fn csr_index(csr_num: u16) -> usize {
        (((csr_num & 0xc00) >> 2) | (csr_num & 0xff)) as usize
    }
}

impl Default for NaclShmem {
    fn default() -> Self {
        Self {
            scratch: [0; 256],
            _reserved: [0; 240],
            dirty_bitmap: [0; 16],
            csrs: [0; 1024],
        }
    }
}

/// Functions provided by the Nested Virtualization Acceleration (NACL) extension.
#[derive(Copy, Clone, Debug)]
pub enum NaclFunction {
    /// Registers the nested hypervisor <-> host hypervisor shared memory area for the calling CPU.
    /// `shmem_pfn` is the base PFN of where the `NaclShmem` struct will be placed in the caller's
    /// physical address space. The entire range of memory occupied by the `NaclShmem` struct must
    /// remain accessible to the caller until the `NaclShmem` strucutre is unregistered by calling
    /// this function with `shmem_pfn` set to -1. In particular this means that, in the presence of
    /// the TEE-Host extension, the memory occupied by the `NaclShmem` structure is "pinned" in
    /// the non-confidential state and cannot be converted.
    ///
    /// a6 = 0
    SetShmem {
        /// a0 = PFN of shared memory area
        shmem_pfn: u64,
    },
    // There are other functions in the proposed NACL extension, but we ignore them as they aren't
    // relevant to the TEE extensions. Note that this violates SBI policy, but since both the TEE and
    // NACL extensions are in active development, we let it go for now.
}

impl NaclFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use NaclFunction::*;
        match args[6] {
            0 => Ok(SetShmem { shmem_pfn: args[0] }),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for NaclFunction {
    fn a6(&self) -> u64 {
        use NaclFunction::*;
        match self {
            SetShmem { .. } => 0,
        }
    }

    fn a0(&self) -> u64 {
        use NaclFunction::*;
        match self {
            SetShmem { shmem_pfn } => *shmem_pfn,
        }
    }
}
