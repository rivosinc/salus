// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use data_measure::data_measure::DataMeasure;
use riscv_pages::*;

use crate::page_table::Result;
use crate::page_table::{TableEntryMut::*, ValidTableEntryMut::*, *};
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
    /// Walks the page table from the root for `gpa` until `pred` returns true. Returns `None` if
    /// a leaf is reached without `pred` being met.
    fn walk_until<P>(&mut self, gpa: GuestPhysAddr, mut pred: P) -> Option<TableEntryMut<Self>>
    where
        P: FnMut(&TableEntryMut<Self>) -> bool,
    {
        let mut entry = PageTable::from_root(self).entry_for_addr_mut(gpa);
        while !pred(&entry) {
            if let Valid(Table(mut t)) = entry {
                entry = t.entry_for_addr_mut(gpa);
            } else {
                return None;
            }
        }
        Some(entry)
    }

    /// Walks the page table to a valid leaf entry mapping `gpa`. Returns `None` if `gpa` is not
    /// mapped.
    fn walk_to_leaf(&mut self, gpa: GuestPhysAddr) -> Option<ValidTableEntryMut<Self>> {
        self.walk_until(gpa, |e| matches!(e, Valid(Leaf(..))))
            .and_then(TableEntryMut::as_valid_entry)
    }

    /// Walks the page table until an invalid entry that would map `gpa` is encountered. Returns
    /// `None` if `gpa` is mapped.
    fn walk_until_invalid(&mut self, gpa: GuestPhysAddr) -> Option<TableEntryMut<Self>> {
        self.walk_until(gpa, |e| matches!(e, Invalid(..)))
    }

    /// Checks the ownership and typing of the page at `page_addr` and then creates a translation
    /// for `gpa` to `spa` with the given permissions, filling in any intermediate page tables
    /// using `get_pte_page` as necessary.
    fn do_map_page<P: PhysPage>(
        &mut self,
        gpa: GuestPhysAddr,
        spa: SupervisorPageAddr,
        perms: PteLeafPerms,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        if spa.size().is_huge() {
            return Err(Error::PageSizeNotSupported(spa.size()));
        }
        if self.phys_pages.owner(spa) != Some(self.owner) {
            return Err(Error::PageNotOwned);
        }
        if self.phys_pages.mem_type(spa) != Some(P::mem_type()) {
            return Err(Error::PageTypeMismatch);
        }

        let mut table = PageTable::from_root(self);
        while table.level().leaf_page_size() != spa.size() {
            table = table.next_level_or_fill_fn(gpa, get_pte_page)?;
        }
        unsafe {
            // Safe since we've verified ownership of the page.
            table.map_leaf(gpa, spa, perms)?
        };
        Ok(())
    }

    /// Returns the valid 4kB leaf PTE mapping `gpa` if the mapped page matches the specified
    /// `mem_type`.
    fn get_4k_leaf_with_type(
        &mut self,
        gpa: GuestPhysAddr,
        mem_type: MemType,
    ) -> Result<ValidTableEntryMut<Self>> {
        let phys_pages = self.phys_pages.clone();
        let entry = self.walk_to_leaf(gpa).ok_or(Error::PageNotOwned)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        // Unwrap ok, must be a leaf entry.
        let spa = entry.page_addr().unwrap();
        if phys_pages.mem_type(spa) != Some(mem_type) {
            return Err(Error::PageTypeMismatch);
        }
        Ok(entry)
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

    fn map_page<P: PhysPage>(
        &mut self,
        gpa: GuestPhysAddr,
        page_to_map: P,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        self.do_map_page::<P>(gpa, page_to_map.addr(), PteLeafPerms::RWX, get_pte_page)
    }

    fn map_page_with_measurement(
        &mut self,
        gpa: GuestPhysAddr,
        page_to_map: Page,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
        data_measure: &mut dyn DataMeasure,
    ) -> Result<()> {
        self.do_map_page::<Page>(gpa, page_to_map.addr(), PteLeafPerms::RWX, get_pte_page)?;
        data_measure.add_page(gpa.bits(), page_to_map.as_bytes());
        Ok(())
    }

    fn unmap_page<P: PhysPage>(&mut self, gpa: GuestPhysAddr) -> Result<UnmappedPhysPage<P>> {
        let entry = self.get_4k_leaf_with_type(gpa, P::mem_type())?;
        let page = unsafe {
            // Safe since we've verified the typing of the page.
            entry.take_page().unwrap()
        };
        Ok(UnmappedPhysPage::new(page))
    }

    fn invalidate_page<P: PhysPage>(&mut self, gpa: GuestPhysAddr) -> Result<UnmappedPhysPage<P>> {
        let entry = self.get_4k_leaf_with_type(gpa, P::mem_type())?;
        let page = unsafe {
            // Safe since we've verified the typing of the page.
            entry.invalidate_page().unwrap()
        };
        Ok(UnmappedPhysPage::new(page))
    }

    fn unmap_range<P: PhysPage>(
        &mut self,
        addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<UnmapIter<Self, P>> {
        if addr.size().is_huge() {
            return Err(Error::PageSizeNotSupported(addr.size()));
        }
        if addr.iter_from().take(num_pages as usize).all(|a| {
            self.get_4k_leaf_with_type(RawAddr::from(a), P::mem_type())
                .is_ok()
        }) {
            Ok(UnmapIter::new(self, addr, num_pages))
        } else {
            Err(Error::PageNotOwned)
        }
    }

    fn invalidate_range<P: PhysPage>(
        &mut self,
        addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<InvalidateIter<Self, P>> {
        if addr.size().is_huge() {
            return Err(Error::PageSizeNotSupported(addr.size()));
        }
        if addr.iter_from().take(num_pages as usize).all(|a| {
            self.get_4k_leaf_with_type(RawAddr::from(a), P::mem_type())
                .is_ok()
        }) {
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
        if let Some(TableEntryMut::Invalid(pte, level)) = self.walk_until_invalid(gpa) {
            if !level.is_leaf() {
                // We don't support huge pages right now so we shouldn't be faulting them in.
                return false;
            }
            // Unwrap ok, this must be a 4kB page.
            let addr = PageAddr::from_pfn(pte.pfn(), level.leaf_page_size()).unwrap();
            if phys_pages.owner(addr) != Some(owner)
                || phys_pages.mem_type(addr) != Some(MemType::Ram)
            {
                // We shouldn't be faulting in MMIO or pages we don't own.
                return false;
            }
            // Zero the page before mapping it back to this VM.
            unsafe {
                // Safe since we've verified ownership and typing of this page.
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
        let entry = self.get_4k_leaf_with_type(gpa, MemType::Ram)?;
        unsafe {
            // Safe since we've verified that this is a RAM page.
            entry.write_to_page(offset, bytes)
        }
    }
}
