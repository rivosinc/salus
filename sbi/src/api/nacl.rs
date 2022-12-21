// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::NaclFunction::*;
use crate::NaclShmem;
use crate::{ecall_send, Result, SbiMessage};

const PFN_SHIFT: u64 = 12;

/// Registers the nested hypervisor <-> host hypervisor shared memory area for the calling CPU.
/// `shmem_ptr` must be page-aligned and refer to a sufficient number of accessible, contiguous
/// pages to hold a `NaclShmem` struct. The pages must remain accessible until the shared-memory area
/// is unregistered by calling `unregister_shmem()`.
///
/// # Safety
///
/// The caller must own the pages referenced by `shmem_ptr`, for the number of pages sufficient to
/// hold the `NaclShmem` structure. Since memory within the shared-memory communication area may be
/// read or written by the host hypervisor at any time, the caller must treat the memory as volatile
/// until it is unregistered.
pub unsafe fn register_shmem(shmem_ptr: *mut NaclShmem) -> Result<()> {
    let msg = SbiMessage::Nacl(SetShmem {
        shmem_pfn: (shmem_ptr as u64) >> PFN_SHIFT,
    });
    ecall_send(&msg)?;
    Ok(())
}

/// Unregisters the nested hypervisor <-> host hypervisor  shared memory area for the calling CPU.
pub fn unregister_shmem() -> Result<()> {
    let msg = SbiMessage::Nacl(SetShmem {
        shmem_pfn: u64::MAX,
    });
    // Safety: Doesn't access host memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}
