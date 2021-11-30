// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::slice;

use crate::page::*;
use crate::page_table::*;
use crate::pte::{Pte, PteLeafPerms};

pub enum L1Table {}
pub enum L2Table {}
pub enum L3Table {}
pub enum L4Table {}

// Setup the page table levels for sv48x4.

impl PageTableLevel for L1Table {
    type LeafPageSize = PageSize4k;
    type NextLevel = L4Table; // Invalid
    const ADDR_SHIFT: u64 = 12;
    const ADDR_WIDTH: u64 = 9;
    const TABLE_PAGES: usize = 1;
}

impl PageTableLevel for L2Table {
    type LeafPageSize = PageSize2MB;
    type NextLevel = L1Table;
    const ADDR_SHIFT: u64 = 21;
    const ADDR_WIDTH: u64 = 9;
    const TABLE_PAGES: usize = 1;
}
impl UpperLevel for L2Table {}

impl PageTableLevel for L3Table {
    type LeafPageSize = PageSize1GB;
    type NextLevel = L2Table;
    const ADDR_SHIFT: u64 = 30;
    const ADDR_WIDTH: u64 = 9;
    const TABLE_PAGES: usize = 1;
}
impl UpperLevel for L3Table {}

impl PageTableLevel for L4Table {
    type LeafPageSize = PageSize512GB;
    type NextLevel = L3Table;
    const ADDR_SHIFT: u64 = 39;
    const ADDR_WIDTH: u64 = 11;
    const TABLE_PAGES: usize = 4;
}
impl UpperLevel for L4Table {}

/// An Sv48x4 set of mappings for second stage translation.
pub struct Sv48x4 {
    pages: [Page4k; 4],
}

impl Sv48x4 {
    /// Creates a new `Sv48x4` from the provided `pages` that provide the 4 pages for the top level
    /// page table directory.
    pub fn new(pages: [Page4k; 4]) -> Self {
        Self { pages }
    }

    /// Returns the top level (L4) page table.
    fn top_level_directory(&mut self) -> PageTable<L4Table> {
        unsafe {
            // Safe to create an array of mutable ptes from the owned pages because the mut
            // reference guarantees the mut slice will be the only owner for the lifetime of the
            // `PageTable` that is returned.
            PageTable::from_slice(slice::from_raw_parts_mut(
                self.pages[0].addr().bits() as *mut Pte,
                L4Table::TABLE_PAGES * ENTRIES_PER_PAGE,
            ))
        }
    }
}

impl PlatformPageTable for Sv48x4 {
    type TLD = L4Table;

    fn map_page_4k<I>(
        &mut self,
        guest_phys_addr: u64,
        page_to_map: Page4k,
        free_pages: &mut I,
    ) -> core::result::Result<(), ()>
    where
        I: Iterator<Item = Page4k>,
    {
        let mut l4 = self.top_level_directory();
        let mut l3 = l4.next_level_or_fill(guest_phys_addr, free_pages)?;
        let mut l2 = l3.next_level_or_fill(guest_phys_addr, free_pages)?;
        let mut l1 = l2.next_level_or_fill(guest_phys_addr, free_pages)?;
        let _entry = l1.map_leaf(guest_phys_addr, page_to_map, PteLeafPerms::RWX);
        Ok(())
    }

    fn map_page_2mb<I>(
        &mut self,
        guest_phys_addr: u64,
        page_to_map: Page<PageSize2MB>,
        free_pages: &mut I,
    ) -> core::result::Result<(), ()>
    where
        I: Iterator<Item = Page4k>,
    {
        let mut l4 = self.top_level_directory();
        let mut l3 = l4.next_level_or_fill(guest_phys_addr, free_pages)?;
        let mut l2 = l3.next_level_or_fill(guest_phys_addr, free_pages)?;
        let _entry = l2.map_leaf(guest_phys_addr, page_to_map, PteLeafPerms::RWX);
        Ok(())
    }

    fn unmap_page(&mut self, guest_phys_addr: u64) -> Option<UnmappedPage> {
        use TableEntryMut::*;
        use ValidTableEntryMut::*;
        let mut l4 = self.top_level_directory();
        let mut l3 = match l4.entry_for_addr_mut(guest_phys_addr) {
            Invalid(_) => return None,
            Valid(Table(t)) => t,
            Valid(valid_leaf) => return valid_leaf.take_page().map(UnmappedPage::Tera),
        };
        let mut l2 = match l3.entry_for_addr_mut(guest_phys_addr) {
            Invalid(_) => return None,
            Valid(Table(t)) => t,
            Valid(valid_leaf) => return valid_leaf.take_page().map(UnmappedPage::Giga),
        };
        let mut l1 = match l2.entry_for_addr_mut(guest_phys_addr) {
            Invalid(_) => return None,
            Valid(Table(t)) => t,
            Valid(valid_leaf) => return valid_leaf.take_page().map(UnmappedPage::Mega),
        };
        match l1.entry_for_addr_mut(guest_phys_addr) {
            Valid(valid_leaf) => return valid_leaf.take_page().map(UnmappedPage::Page),
            _ => None,
        }
    }

    fn get_root_address(&self) -> PageAddr4k {
        self.pages[0].addr()
    }
}
