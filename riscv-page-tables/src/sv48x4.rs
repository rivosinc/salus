// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use data_measure::data_measure::DataMeasure;
use riscv_pages::*;

use crate::page_table::Result;
use crate::page_table::*;
use crate::page_tracking::PageState;
use crate::pte::PteLeafPerms;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sv48x4Level {
    L1Table,
    L2Table,
    L3Table,
    L4Table,
}

impl PageTableLevel for Sv48x4Level {
    fn leaf_page_size(&self) -> PageSize {
        match self {
            Sv48x4Level::L1Table => PageSize::Size4k,
            Sv48x4Level::L2Table => PageSize::Size2M,
            Sv48x4Level::L3Table => PageSize::Size1G,
            Sv48x4Level::L4Table => PageSize::Size512G,
        }
    }

    fn next(&self) -> Option<Self> {
        match self {
            Sv48x4Level::L1Table => None,
            Sv48x4Level::L2Table => Some(Sv48x4Level::L1Table),
            Sv48x4Level::L3Table => Some(Sv48x4Level::L2Table),
            Sv48x4Level::L4Table => Some(Sv48x4Level::L3Table),
        }
    }

    fn addr_shift(&self) -> u64 {
        match self {
            Sv48x4Level::L1Table => 12,
            Sv48x4Level::L2Table => 21,
            Sv48x4Level::L3Table => 30,
            Sv48x4Level::L4Table => 39,
        }
    }

    fn addr_width(&self) -> u64 {
        match self {
            Sv48x4Level::L1Table => 9,
            Sv48x4Level::L2Table => 9,
            Sv48x4Level::L3Table => 9,
            Sv48x4Level::L4Table => 11,
        }
    }

    fn table_pages(&self) -> usize {
        match self {
            Sv48x4Level::L1Table => 1,
            Sv48x4Level::L2Table => 1,
            Sv48x4Level::L3Table => 1,
            Sv48x4Level::L4Table => 4,
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Sv48x4Level::L1Table)
    }
}

/// An Sv48x4 set of mappings for second stage translation.
pub struct Sv48x4 {
    root: SequentialPages,
    owner: PageOwnerId,
    phys_pages: PageState,
}

impl Sv48x4 {
    /// Walks the page table to a valid leaf entry mapping `gpa`. Returns `None` if `gpa` is not
    /// mapped.
    fn walk_to_valid_leaf(&mut self, gpa: GuestPhysAddr) -> Option<ValidTableEntryMut<Self>> {
        use TableEntryMut::*;
        use ValidTableEntryMut::*;

        let mut table = PageTable::from_root(self);
        while let Valid(Table(t)) = table.entry_for_addr_mut(gpa) {
            table = t;
        }
        match table.entry_for_addr_mut(gpa) {
            Valid(valid_leaf) => Some(valid_leaf),
            _ => None,
        }
    }

    /// Walks the page table until an invalid entry that would map `gpa` is encountered. Returns
    /// `None` if `gpa` is mapped.
    fn walk_to_invalid_leaf(&mut self, gpa: GuestPhysAddr) -> Option<TableEntryMut<Self>> {
        use TableEntryMut::*;
        use ValidTableEntryMut::*;

        let mut table = PageTable::from_root(self);
        while let Valid(Table(t)) = table.entry_for_addr_mut(gpa) {
            table = t;
        }
        match table.entry_for_addr_mut(gpa) {
            Valid(_) => None,
            invalid => Some(invalid),
        }
    }

    /// Creates a translation for `gpa` to `spa` with the given permissions, filling in any
    /// intermediate page tables using `get_pte_page` as necessary.
    fn fill_and_map(
        &mut self,
        gpa: GuestPhysAddr,
        spa: SupervisorPageAddr,
        perms: PteLeafPerms,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        let mut table = PageTable::from_root(self);
        while table.level().leaf_page_size() != spa.size() {
            table = table.next_level_or_fill_fn(gpa, get_pte_page)?;
        }
        table.map_leaf(gpa, spa, perms)
    }

    /// Returns if the given guest physical address is mapped in the page table as a 4kB page.
    fn addr_mapped_as_4k_page(&mut self, gpa: GuestPhysAddr) -> bool {
        use ValidTableEntryMut::*;

        let entry = self.walk_to_valid_leaf(gpa);
        if let Some(Leaf(pte, level)) = entry {
            if !level.is_leaf() {
                return false;
            }
            let addr = PageAddr::from_pfn(pte.pfn(), level.leaf_page_size()).unwrap();
            self.phys_pages.owner(addr).unwrap() == self.owner
        } else {
            false
        }
    }
}

// TODO: Support non-4k page sizes.
impl PlatformPageTable for Sv48x4 {
    type Level = Sv48x4Level;
    const HGATP_VALUE: u64 = 9;
    const TOP_LEVEL_ALIGN: u64 = 16 * 1024;

    fn page_owner_id(&self) -> PageOwnerId {
        self.owner
    }

    fn root_level(&self) -> Self::Level {
        Sv48x4Level::L4Table
    }

    fn max_pte_pages(num_pages: u64) -> u64 {
        // Determine how much ram is needed for host sv48x4 mappings; 512 8-byte ptes per page
        let num_l1_pages = num_pages / ENTRIES_PER_PAGE + 1;
        let num_l2_pages = num_l1_pages / ENTRIES_PER_PAGE + 1;
        let num_l3_pages = num_l2_pages / ENTRIES_PER_PAGE + 1;
        let num_l4_pages = 4;
        num_l1_pages + num_l2_pages + num_l3_pages + num_l4_pages
    }

    fn new(root: SequentialPages, owner: PageOwnerId, phys_pages: PageState) -> Result<Self> {
        // TODO: Verify ownership of root PT pages.
        if root.page_size().is_huge() {
            return Err(Error::PageSizeNotSupported(root.page_size()));
        }
        if root.base().bits() & (Self::TOP_LEVEL_ALIGN - 1) != 0 {
            return Err(Error::MisalignedPages(root));
        }
        if root.len() < Sv48x4Level::L4Table.table_pages() as u64 {
            return Err(Error::InsufficientPages(root));
        }
        Ok(Self {
            root,
            owner,
            phys_pages,
        })
    }

    fn phys_pages(&self) -> PageState {
        self.phys_pages.clone()
    }

    fn map_page(
        &mut self,
        gpa: GuestPhysAddr,
        page_to_map: Page,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
        data_measure: Option<&mut dyn DataMeasure>,
    ) -> Result<()> {
        let page_addr = page_to_map.addr();
        if page_addr.size().is_huge() {
            return Err(Error::PageSizeNotSupported(page_addr.size()));
        }
        let owner = self
            .phys_pages
            .owner(page_addr)
            .ok_or(Error::PageNotOwned)?;
        if owner != self.owner {
            return Err(Error::PageNotOwned);
        }
        self.fill_and_map(gpa, page_to_map.addr(), PteLeafPerms::RWX, get_pte_page)?;
        if let Some(data_measure) = data_measure {
            data_measure.add_page(gpa.bits(), page_to_map.as_bytes());
        }
        Ok(())
    }

    fn unmap_page(&mut self, gpa: GuestPhysAddr) -> Result<CleanPage> {
        let entry = self.walk_to_valid_leaf(gpa).ok_or(Error::PageNotOwned)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        Ok(CleanPage::from(
            entry.take_page().map(UnmappedPage::new).unwrap(),
        ))
    }

    fn invalidate_page(&mut self, gpa: GuestPhysAddr) -> Result<UnmappedPage> {
        let entry = self.walk_to_valid_leaf(gpa).ok_or(Error::PageNotOwned)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        Ok(entry.invalidate_page().map(UnmappedPage::new).unwrap())
    }

    fn unmap_range(&mut self, addr: GuestPageAddr, num_pages: u64) -> Result<UnmapIter<Self>> {
        if addr.size().is_huge() {
            return Err(Error::PageSizeNotSupported(addr.size()));
        }
        if addr
            .iter_from()
            .take(num_pages as usize)
            .all(|a| self.addr_mapped_as_4k_page(RawAddr::from(a)))
        {
            Ok(UnmapIter::new(self, addr, num_pages))
        } else {
            Err(Error::PageNotOwned)
        }
    }

    fn invalidate_range(
        &mut self,
        addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<InvalidateIter<Self>> {
        if addr.size().is_huge() {
            return Err(Error::PageSizeNotSupported(addr.size()));
        }
        if addr
            .iter_from()
            .take(num_pages as usize)
            .all(|a| self.addr_mapped_as_4k_page(RawAddr::from(a)))
        {
            Ok(InvalidateIter::new(self, addr, num_pages))
        } else {
            Err(Error::PageNotOwned)
        }
    }

    fn get_root_address(&self) -> SupervisorPageAddr {
        self.root.base()
    }

    fn do_guest_fault(&mut self, gpa: GuestPhysAddr) -> bool {
        // avoid double self borrow, by cloning the pages, each layer borrows self, so the borrow
        // checked can't tell that phys_pages is only borrowed once.
        let phys_pages = self.phys_pages.clone();
        let owner = self.owner;
        let entry = self.walk_to_invalid_leaf(gpa);
        if let Some(TableEntryMut::Invalid(pte, level)) = entry {
            if !level.is_leaf() {
                return false;
            }
            let addr = PageAddr::from_pfn(pte.pfn(), level.leaf_page_size()).unwrap();
            if phys_pages.owner(addr) != Some(owner) {
                return false;
            }
            // Zero the page before mapping it back to this VM.
            unsafe {
                // Safe because this table uniquely owns the page and it isn't mapped to a
                // guest.
                core::ptr::write_bytes(addr.bits() as *mut u8, 0, level.leaf_page_size() as usize);
            }
            pte.mark_valid();
            true
        } else {
            false
        }
    }

    fn write_guest_owned_page(
        &mut self,
        gpa: GuestPhysAddr,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        let entry = self.walk_to_valid_leaf(gpa).ok_or(Error::PageNotOwned)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        entry.write_to_page(offset, bytes)
    }
}
