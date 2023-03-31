// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Allow unused code until all features are added to the owning crate.
#![allow(dead_code)]

use riscv_pages::{Pfn, SupervisorPfn};

// Both Sv39 and Sv48 use 44 bits for the page frame number.
const PFN_BITS: u64 = 44;
const PFN_MASK: u64 = (1 << PFN_BITS) - 1;
// Risc-V PTEs keep the PFN starting at bit 10. The first 10 bits are for the `PteFieldBits1` and
// two bits reserved for the supervisor `RSW` in the privileged spec.
const PFN_SHIFT: u64 = 10;

/// Bits from a Risc-V PTE.
#[derive(Copy, Clone)]
pub enum PteFieldBit {
    /// This PTE is valid when set.
    Valid = 0,
    /// Reading allowed when set.
    Read = 1,
    /// Writing allowed when set.
    Write = 2,
    /// Executing allowed when set.
    Execute = 3,
    /// Access from U mode is allowed when set.
    User = 4,
    /// When set, indicates this is a global mapping.
    Global = 5,
    /// The page has been accessed.
    Accessed = 6,
    /// The page has been written.
    Dirty = 7,
    /// The page has been locked by software.
    Locked = 8,
}

impl PteFieldBit {
    /// Returns the bit position of this bit field in the PTE.
    pub const fn shift(&self) -> u64 {
        *self as u64
    }

    /// Returns the mask covering all bits of this PTE field.
    pub const fn mask(&self) -> u64 {
        1 << self.shift()
    }

    /// Returns true if the field is non-zero.
    pub const fn is_set(&self, val: u64) -> bool {
        val & self.mask() != 0
    }
}

/// Permissions for a leaf page entry.
#[allow(clippy::upper_case_acronyms)]
pub enum PteLeafPerms {
    /// Read only
    R = PteFieldBit::Read.mask() as isize,
    /// Read/Write
    RW = (PteFieldBit::Read.mask() | PteFieldBit::Write.mask()) as isize,
    /// Execute only
    X = PteFieldBit::Execute.mask() as isize,
    /// Read/Execute
    RX = (PteFieldBit::Read.mask() | PteFieldBit::Execute.mask()) as isize,
    /// Read/Write/Execute
    RWX = (PteFieldBit::Read.mask() | PteFieldBit::Write.mask() | PteFieldBit::Execute.mask())
        as isize,
    /// User Read/Execute
    URX = (PteFieldBit::User.mask() | PteFieldBit::Read.mask() | PteFieldBit::Execute.mask())
        as isize,
    /// User Read Only
    UR = (PteFieldBit::User.mask() | PteFieldBit::Read.mask()) as isize,
    /// User Read/Write
    URW =
        (PteFieldBit::User.mask() | PteFieldBit::Read.mask() | PteFieldBit::Write.mask()) as isize,
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
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `pfn` references a page that is uniquely owned and doesn't
    /// create an alias.
    pub unsafe fn set(&mut self, pfn: SupervisorPfn, status: &PteFieldBits) {
        self.0 = (pfn.bits() << PFN_SHIFT) | status.bits | PteFieldBit::Valid.mask();
    }

    /// Updates the pfn part of the entry, keeping everything else same. Returns the old pfn value.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `pfn` references a page that is uniquely owned and doesn't
    /// create an alias. Old SupervisorPageAddr is returned and the caller must make sure
    /// to remove it from `GuestStagePageTable::page_tracker`.
    pub unsafe fn update_pfn(&mut self, pfn: SupervisorPfn) -> SupervisorPfn {
        let prev = Pfn::supervisor((self.0 >> PFN_SHIFT) & PFN_MASK);
        self.0 = (self.0 & !(PFN_MASK << PFN_SHIFT)) | (pfn.bits() << PFN_SHIFT);
        prev
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
    #[allow(dead_code)]
    pub fn mark_valid(&mut self) {
        self.0 |= PteFieldBit::Valid.mask();
    }

    /// Returns if the entry is marked as locked.
    pub fn locked(&self) -> bool {
        PteFieldBit::Locked.is_set(self.bits())
    }

    /// Marks the entry as locked.
    pub fn lock(&mut self) {
        self.0 |= PteFieldBit::Locked.mask()
    }

    /// Marks the entry as unlocked.
    pub fn unlock(&mut self) {
        self.0 &= !PteFieldBit::Locked.mask()
    }

    /// Clears everything including valid bit.
    pub fn clear(&mut self) {
        self.0 = 0;
    }

    /// Returns `true` if the entry is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.bits() & MASK_RWX != 0
    }

    /// Returns the pfn of this entry.
    pub fn pfn(&self) -> SupervisorPfn {
        Pfn::supervisor((self.bits() >> PFN_SHIFT) & PFN_MASK)
    }
}

/// The status bits that define PTE state.
#[derive(Default, Copy, Clone, PartialEq)]
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

    /// Creates a new status for a leaf entry with the given `perms`.
    pub fn user_leaf_with_perms(perms: PteLeafPerms) -> Self {
        let mut ret = Self::default();
        ret.bits |= perms as u64;
        ret.set_bit(PteFieldBit::User);
        ret
    }

    /// Creates a new status for a non-leaf entry.
    /// Used for intermeidate levels of page tables.
    pub fn non_leaf() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        pte::{Pte, PteFieldBit},
        PteFieldBits, PteLeafPerms,
    };

    #[test]
    fn pte() {
        let mut pte = Pte(0x1000);
        assert_eq!(pte.bits(), 0x1000);
        pte.clear();
        assert_eq!(pte.bits(), 0);
        assert!(!pte.valid());
        assert!(!pte.locked());
        pte.mark_valid();
        assert!(pte.valid());
        pte.invalidate();
        assert!(!pte.valid());
        pte.lock();
        assert!(pte.locked());
        pte.unlock();
        assert!(!pte.locked());
        assert!(!pte.is_leaf());
        let pfn = pte.pfn();
        let status = PteFieldBits::leaf_with_perms(PteLeafPerms::RWX);
        assert!(!PteFieldBit::User.is_set(status.bits()));
        let status = PteFieldBits::user_leaf_with_perms(PteLeafPerms::RWX);
        assert!(PteFieldBit::User.is_set(status.bits()));
        unsafe { pte.set(pfn, &status) };
        assert!(pte.valid());
        assert!(pte.is_leaf());
        let status = PteFieldBits::non_leaf();
        assert_eq!(status.bits(), 0);
        unsafe { pte.set(pfn, &status) };
        assert!(!pte.is_leaf());
    }

    #[test]
    fn pte_field_bits() {
        let mut status = PteFieldBits::default();
        assert_eq!(status.bits(), 0);
        for pte_field_bit in vec![
            PteFieldBit::Valid,
            PteFieldBit::Read,
            PteFieldBit::Write,
            PteFieldBit::Execute,
            PteFieldBit::User,
            PteFieldBit::Global,
            PteFieldBit::Accessed,
            PteFieldBit::Dirty,
            PteFieldBit::Locked,
        ] {
            assert!(!pte_field_bit.is_set(status.bits()));
            status.set_bit(pte_field_bit);
            assert!(pte_field_bit.is_set(status.bits()));
            status.clear_bit(pte_field_bit);
            assert!(!pte_field_bit.is_set(status.bits()));
        }
    }
}
