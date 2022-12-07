// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use super::core::MAX_INTERRUPT_IDS;

// A single EIE/EIP pair.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SwFileEntry {
    pending: u64,
    enable: u64,
}

/// The number of 64-bit EIE/EIP pairs in an interrupt file, as mandated by the AIA specification.
pub const SW_FILE_ENTRIES: usize = MAX_INTERRUPT_IDS / 64;

/// Holds the software-visible state of an IMSIC guest interrupt file. Used when a guest interrupt
/// file is swapped out.
///
/// This is meant to resemble the memory-resident interrupt file (MRIF) structure described in chapter
/// 9 of the AIA specification. Since we don't yet support those on the IOMMU side, this is simply
/// a for-CPU-access-only version of that structure.
///
/// TODO: Actual MRIFs need to be 512-byte aligned and use atomic ops when manipulating the EIP bits.
#[repr(C)]
#[derive(Default)]
pub struct SwFile {
    entries: [SwFileEntry; SW_FILE_ENTRIES],
    eidelivery: u64,
    eithreshold: u64,
}

impl SwFile {
    /// Creates an empty `SwFile`.
    pub fn new() -> Self {
        Self {
            entries: [SwFileEntry::default(); SW_FILE_ENTRIES],
            eidelivery: 0,
            eithreshold: 0,
        }
    }

    /// Returns the saved value of the EIDEILVERY register.
    pub fn eidelivery(&self) -> u64 {
        self.eidelivery
    }

    /// Sets the saved value of the EIDELIVERY register.
    pub fn set_eidelivery(&mut self, val: u64) {
        self.eidelivery = val;
    }

    /// Returns the saved value of the EITHRESHOLD register.
    pub fn eithreshold(&self) -> u64 {
        self.eithreshold
    }

    /// Sets the saved value of the EITHRESHOLD register.
    pub fn set_eithreshold(&mut self, val: u64) {
        self.eithreshold = val;
    }

    /// Returns the saved value of the EIP register at `index`.
    pub fn eip(&self, index: usize) -> u64 {
        self.entries[index].pending
    }

    /// Sets the saved value of the EIP register at `index`.
    pub fn set_eip(&mut self, index: usize, val: u64) {
        self.entries[index].pending = val;
    }

    /// Sets the bit corresponding to `id` in the EIP register array.
    pub fn set_eip_bit(&mut self, id: usize) {
        self.entries[id / 64].pending |= 1 << (id % 64);
    }

    /// Returns the saved value of the EIE register at `index`.
    pub fn eie(&self, index: usize) -> u64 {
        self.entries[index].enable
    }

    /// Sets the saved value of the EIE register at `index`.
    pub fn set_eie(&mut self, index: usize, val: u64) {
        self.entries[index].enable = val;
    }
}
