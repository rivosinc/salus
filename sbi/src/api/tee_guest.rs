// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::TeeGuestFunction::*;
use crate::{ecall_send, Result, SbiMessage, TeeMemoryRegion};

/// Registers an emulated MMIO region in a previously-unused range of guest physical address space.
/// Future accesses in the specified address range will trap to the host, allowing it to emulate
/// the access.
pub fn add_emulated_mmio_region(addr: u64, len: u64) -> Result<()> {
    let msg = SbiMessage::TeeGuest(AddMemoryRegion {
        region_type: TeeMemoryRegion::EmulatedMmio,
        addr,
        len,
    });
    // Safety: AddMemoryRegion does not directly access our memory. The specified range of
    // address space must have been previously inaccessible for the call to succeed, after which
    // accesses to that range have well-defined behavior.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Registers a shared memory region in a previously-unused range of guest physical address space.
/// Future accesses in the specified address range will trap to to the host, allowing it to insert
/// pages of shared memory.
pub fn add_shared_memory_region(addr: u64, len: u64) -> Result<()> {
    let msg = SbiMessage::TeeGuest(AddMemoryRegion {
        region_type: TeeMemoryRegion::Shared,
        addr,
        len,
    });
    // Safety: AddMemoryRegion does not directly access our memory. The specified range of
    // address space must have been previously inaccessible for the call to succeed, after which
    // accesses to that range have well-defined behavior.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}
