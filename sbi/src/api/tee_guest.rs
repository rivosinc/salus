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

/// Converts the specified range of address space from confidential to shared.
///
/// # Safety
///
/// This operation is destructive; the contents of memory in the range to be converted are lost.
/// The calling VM must not access the memory in this range on any other CPUs until this call
/// returns, at which point accesses within the range are guaranteed to be to memory shared with
/// the host.
pub unsafe fn share_memory(addr: u64, len: u64) -> Result<()> {
    let msg = SbiMessage::TeeGuest(ShareMemory { addr, len });
    ecall_send(&msg)?;
    Ok(())
}

/// Converts the specified range of address space from shared to confidential.
///
/// # Safety
///
/// This operation is destructive; the contents of memory in the range to be converted are lost.
/// The calling VM must not access the memory in this range on any other CPUs until this call
/// returns, at which point accesses within the range are guaranteed to be to memory that is
/// confidential to the calling VM.
pub unsafe fn unshare_memory(addr: u64, len: u64) -> Result<()> {
    let msg = SbiMessage::TeeGuest(UnshareMemory { addr, len });
    ecall_send(&msg)?;
    Ok(())
}

/// Allows injection of the specified external interrupt ID by the host to the calling CPU.
pub fn allow_external_interrupt(id: u64) -> Result<()> {
    let msg = SbiMessage::TeeGuest(AllowExternalInterrupt { id: id as i64 });
    // Safety: AllowExternalInterrupt doesn't access our memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Allows injection of all external interrupts by the host to the calling CPU.
pub fn allow_all_external_interrupts() -> Result<()> {
    let msg = SbiMessage::TeeGuest(AllowExternalInterrupt { id: -1 });
    // Safety: AllowExternalInterrupt doesn't access our memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Denies injection of the specified external interrupt ID by the host to the calling CPU.
pub fn deny_external_interrupt(id: u64) -> Result<()> {
    let msg = SbiMessage::TeeGuest(DenyExternalInterrupt { id: id as i64 });
    // Safety: DenyExternalInterrupt doesn't access our memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Denies injection of all external interrupts by the host to the calling CPU.
pub fn deny_all_external_interrupts() -> Result<()> {
    let msg = SbiMessage::TeeGuest(DenyExternalInterrupt { id: -1 });
    // Safety: DenyExternalInterrupt doesn't access our memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}
