// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Low-level TLB management operations.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;

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
    match (gaddr, vmid) {
        // Safety: HFENCE.GVMA's behavior is well-defined and its only side effect is to invalidate
        // address translation caches.
        (Some(addr), Some(id)) => unsafe {
            asm!("hfence.gvma {rs1}, {rs2}", rs1 = in(reg) addr >> 2, rs2 = in(reg) id);
        },
        (Some(addr), None) => unsafe {
            asm!("hfence.gvma {rs1}, zero", rs1 = in(reg) addr >> 2);
        },
        (None, Some(id)) => unsafe {
            asm!("hfence.gvma zero, {rs2}", rs2 = in(reg) id);
        },
        (None, None) => unsafe {
            asm!("hfence.gvma");
        },
    }
}

// Make fence instructions a no-op for testing.
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn sfence_vma(_vaddr: Option<u64>, _asid: Option<u64>) {}
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub fn hfence_gvma(_gaddr: Option<u64>, _vmid: Option<u64>) {}
