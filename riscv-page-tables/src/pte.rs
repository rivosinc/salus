// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{Pfn, PhysPage, SupervisorPfn};

// Both Sv39 and Sv48 use 44 bits for the page frame number.
const PFN_BITS: u64 = 44;
const PFN_MASK: u64 = (1 << PFN_BITS) - 1;
// Risc-V PTEs keep the PFN starting at bit 10. The first 10 bits are for the `PteFieldBits1` and
// two bits reserved for the supervisor `RSW` in the privileged spec.
const PFN_SHIFT: u64 = 10;

/// Bits from a Risc-V PTE.
#[derive(Copy, Clone)]
pub enum PteFieldBit {
    Valid = 0,
    Read = 1,
    Write = 2,
    Execute = 3,
    User = 4,
    Global = 5,
    Accessed = 6,
    Dirty = 7,
}

impl PteFieldBit {
    pub const fn shift(&self) -> u64 {
        *self as u64
    }

    pub const fn mask(&self) -> u64 {
        1 << self.shift()
    }

    pub const fn is_set(&self, val: u64) -> bool {
        val & self.mask() != 0
    }
}

/// Permissions for a leaf page entry.
pub enum PteLeafPerms {
    R = PteFieldBit::Read.mask() as isize,
    RW = (PteFieldBit::Read.mask() | PteFieldBit::Write.mask()) as isize,
    X = PteFieldBit::Execute.mask() as isize,
    RX = (PteFieldBit::Read.mask() | PteFieldBit::Execute.mask()) as isize,
    RWX = (PteFieldBit::Read.mask() | PteFieldBit::Write.mask() | PteFieldBit::Execute.mask())
        as isize,
}

const MASK_RWX: u64 = (1 << PteFieldBit::Read.shift())
    | (1 << PteFieldBit::Write.shift())
    | (1 << PteFieldBit::Execute.shift());

/// Represents a PTE in memory. Never instantiated. Only used as a reference to entries in a page
/// table.
pub(crate) struct Pte(u64);

impl Pte {
    /// Writes the mapping for the given page with that config bits in `status` and marks the entry
    /// as valid.
    pub fn set<P: PhysPage>(&mut self, page: P, status: &PteFieldBits) {
        self.0 = (page.pfn().bits() << PFN_SHIFT) | status.bits | PteFieldBit::Valid.mask();
    }

    /// Returns the raw bits the make up the PTE.
    pub fn bits(&self) -> u64 {
        self.0
    }

    /// Returns `true` if the entry is valid.
    pub fn valid(&self) -> bool {
        PteFieldBit::Valid.is_set(self.bits())
    }

    /// Marks the entry as invalid
    pub fn invalidate(&mut self) {
        self.0 &= !PteFieldBit::Valid.mask();
    }

    /// Marks the entry as valid
    pub fn mark_valid(&mut self) {
        self.0 |= PteFieldBit::Valid.mask();
    }

    /// Clears everything including valid bit.
    pub fn clear(&mut self) {
        self.0 = 0;
    }

    /// Returns `true` if the entry is a leaf.
    pub fn leaf(&self) -> bool {
        self.bits() & MASK_RWX != 0
    }

    /// Returns the pfn of this entry.
    pub fn pfn(&self) -> SupervisorPfn {
        Pfn::supervisor((self.bits() >> PFN_SHIFT) & PFN_MASK)
    }
}

/// The status bits that define PTE state.
#[derive(Default, Copy, Clone)]
pub struct PteFieldBits {
    bits: u64,
}

impl PteFieldBits {
    /// Returns the raw bits that make up the PTE status.
    pub fn bits(&self) -> u64 {
        self.bits
    }

    /// Sets the given bit in the PTE.
    pub fn set_bit(&mut self, bit: PteFieldBit) {
        self.bits |= 1 << bit.shift();
    }

    /// Clears the given bit in the PTE.
    pub fn clear_bit(&mut self, bit: PteFieldBit) {
        self.bits &= !(1 << bit.shift());
    }

    /// Creates a new status for a leaf entry with the given `perms`.
    pub fn leaf_with_perms(perms: PteLeafPerms) -> Self {
        let mut ret = Self::default();
        ret.bits |= perms as u64;
        ret
    }

    /// Creates a new status for a non-leaf entry.
    /// Used for intermeidate levels of page tables.
    pub fn non_leaf() -> Self {
        Self::default()
    }
}
