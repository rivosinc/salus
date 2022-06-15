// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Low-level TLB management operations.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;

/// A TLB version number.
///
/// We use TLB versions to track the progress of a coordinated TLB shootdown across multiple CPUs.
/// A shootdown is considered completed if the TLB version at which the shootdown was initiated is
/// older (less than) the current TLB version.
///
/// TODO: Overflow. A simple '!=' check against the current version should be sufficient since a
/// page cannot have been invalidated at a future version. Let's do the more defensive thing for now
/// though given how long it'll take a 64-bit counter to overflow.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TlbVersion(u64);

impl TlbVersion {
    /// Creates a new TLB version number, starting from 0.
    pub fn new() -> Self {
        TlbVersion(0)
    }

    /// Increments this TLB version number.
    pub fn increment(self) -> Self {
        TlbVersion(self.0 + 1)
    }
}

/// Executes an SFENCE.VMA instruction.
///
/// If `vaddr` is not None only translations mapping the specified virtual address are invalidated,
/// otherwise translations for all virtual addresses are invalidated.
///
/// If 'asid' is not None only translations using the specified ASID are invalidated, otherwise
/// translations for all ASIDs are invalidated.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub fn sfence_vma(vaddr: Option<u64>, asid: Option<u64>) {
    match (vaddr, asid) {
        // Safety: SFENCE.VMA's behavior is well-defined and its only side effect is to invalidate
        // address translation caches.
        (Some(addr), Some(id)) => unsafe {
            asm!("sfence.vma {rs1}, {rs2}", rs1 = in(reg) addr, rs2 = in(reg) id);
        },
        (Some(addr), None) => unsafe {
            asm!("sfence.vma {rs1}, zero", rs1 = in(reg) addr);
        },
        (None, Some(id)) => unsafe {
            asm!("sfence.vma zero, {rs2}", rs2 = in(reg) id);
        },
        (None, None) => unsafe {
            asm!("sfence.vma");
        },
    }
}

/// Executes an HFENCE.GVMA instruction.
///
/// If `gaddr` is not None only 2nd-stage translations mapping the specified guest physical address
/// are invalidated, otherwise translations for guest physical addresses are invalidated.
///
/// If 'vmid' is not None only 2nd-stage translations using the specified VMID are invalidated,
/// otherwise translations for all VMIDs are invalidated.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub fn hfence_gvma(gaddr: Option<u64>, vmid: Option<u64>) {
    // LLVM 14.x doesn't support hypervisor instruction mnemonics :(
    //
    // HFENCE.VMA encoding: 011001 rs2[4:0] rs1[4:0] 000 00000 1110011
    match (gaddr, vmid) {
        // Safety: HFENCE.GVMA's behavior is well-defined and its only side effect is to invalidate
        // address translation caches.
        (Some(addr), Some(id)) => unsafe {
            // hfence.gvma a0, a1
            asm!(".word 0x62b50073", in("a0") addr >> 2, in("a1") id);
        },
        (Some(addr), None) => unsafe {
            // hfence.gvma a0, zero
            asm!(".word 0x62050073", in("a0") addr >> 2);
        },
        (None, Some(id)) => unsafe {
            // hfence.gvma zero, a0
            asm!(".word 0x62a00073", in("a0") id);
        },
        (None, None) => unsafe {
            // hfence.gvma zero, zero
            asm!(".word 0x62000073");
        },
    }
}

// Make fence instructions a no-op for testing.
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn sfence_vma(_vaddr: Option<u64>, _asid: Option<u64>) {}
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn hfence_gvma(_gaddr: Option<u64>, _vmid: Option<u64>) {}
