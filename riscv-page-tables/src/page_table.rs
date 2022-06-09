// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;

use data_measure::data_measure::DataMeasure;
use riscv_pages::{
    AddressSpace, GuestPhys, MemType, Page, PageAddr, PageOwnerId, PageSize, PhysPage, RawAddr,
    SequentialPages, SupervisorPageAddr, SupervisorVirt, UnmappedPhysPage,
};

use crate::page_tracking::PageTracker;
use crate::pte::{Pte, PteFieldBit, PteFieldBits, PteLeafPerms};

pub(crate) const ENTRIES_PER_PAGE: u64 = 4096 / 8;

#[derive(Debug)]
pub enum Error {
    InsufficientPages(SequentialPages),
    InsufficientPtePages,
    InvalidOffset,
    LeafEntryNotTable,
    MisalignedPages(SequentialPages),
    OutOfBounds,
    PageNotOwned,
    TableEntryNotLeaf,
    PageSizeNotSupported(PageSize),
    MappingExists,
    PageTypeMismatch,
}
pub type Result<T> = core::result::Result<T, Error>;

/// Defines the structure of a multi-level page table.
pub trait PageTableLevel: Sized + Clone + Copy + PartialEq {
    /// Returns the page size of leaf pages mapped by this page table level.
    fn leaf_page_size(&self) -> PageSize;

    /// Returns the next level (in order of decreasing page size) in the hierarchy. Returns `None`
    /// if this is a leaf level.
    fn next(&self) -> Option<Self>;

    /// Returns the position of the table index selected from the input address at this level.
    fn addr_shift(&self) -> u64;

    /// Returns the width of the table index selected from the input address at this level.
    fn addr_width(&self) -> u64;

    /// Returns the number of pages that make up a page table at this level. Must be 1 for all but
    /// the root level.
    fn table_pages(&self) -> usize;

    /// Returns if this is a leaf level.
    fn is_leaf(&self) -> bool;
}

/// A mutable reference to an entry of a page table, either valid or invalid.
pub(crate) enum TableEntryMut<'a, T: PlatformPageTable> {
    Valid(ValidTableEntryMut<'a, T>),
    Invalid(&'a mut Pte, T::Level),
}

impl<'a, T: PlatformPageTable> TableEntryMut<'a, T> {
    /// Creates a `TableEntryMut` by inspecting the passed `pte` and determining its type.
    pub fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        if !pte.valid() {
            TableEntryMut::Invalid(pte, level)
        } else {
            TableEntryMut::Valid(ValidTableEntryMut::from_pte(pte, level))
        }
    }

    /// Returns the entry as a `ValidTableEntryMut` if it's a valid PTE.
    pub fn as_valid_entry(entry: TableEntryMut<'a, T>) -> Option<ValidTableEntryMut<'a, T>> {
        if let TableEntryMut::Valid(v) = entry {
            Some(v)
        } else {
            None
        }
    }
}

/// A valid entry that contains either a `Leaf` with the host address of the page, or a `Table` with
/// a nested `PageTable`.
pub(crate) enum ValidTableEntryMut<'a, T: PlatformPageTable> {
    Leaf(&'a mut Pte, T::Level),
    Table(PageTable<'a, T>),
}

impl<'a, T: PlatformPageTable> ValidTableEntryMut<'a, T> {
    /// Creates a valid entry from a raw `Pte` at the given level. Asserts if the PTE is invalid.
    pub fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        assert!(pte.valid()); // TODO -valid pte type to eliminate this runtime assert.
        if pte.leaf() {
            ValidTableEntryMut::Leaf(pte, level)
        } else {
            // Safe to create a `PageTable` from the page pointed to by this entry since:
            //  - all valid, non-leaf PTEs must point to an intermediate page table which must
            //    consume exactly one page, and
            //  - all pages pointed to by PTEs in this paging hierarchy are owned by the root
            //    `PlatformPageTable` and have their lifetime bound to the root.
            let table = unsafe { PageTable::from_pte(pte, level.next().unwrap()) };
            ValidTableEntryMut::Table(table)
        }
    }

    /// Returns the `PageTableLevel` this entry is at.
    pub fn level(&self) -> T::Level {
        match self {
            ValidTableEntryMut::Leaf(_, level) => *level,
            ValidTableEntryMut::Table(t) => t.level(),
        }
    }

    /// Returns the page table pointed to by this entry or `None` if it is a leaf.
    pub fn table(self) -> Option<PageTable<'a, T>> {
        match self {
            ValidTableEntryMut::Table(t) => Some(t),
            ValidTableEntryMut::Leaf(..) => None,
        }
    }

    /// Returns the supervisor physical address mapped by this entry or `None` if it is a table
    /// pointer.
    pub fn page_addr(&self) -> Option<SupervisorPageAddr> {
        if let ValidTableEntryMut::Leaf(pte, level) = self {
            PageAddr::from_pfn(pte.pfn(), level.leaf_page_size())
        } else {
            None
        }
    }

    /// Take the page out of the page table that owns it and return it.
    /// Returns the page if a valid leaf, otherwise, None.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that this page table entry points to a page of memory of the
    /// same type as `P`.
    pub unsafe fn take_page<P: PhysPage>(self) -> Option<P> {
        let addr = self.page_addr()?;
        if let ValidTableEntryMut::Leaf(pte, _) = self {
            // The page is guaranteed to be owned by this page table (and by extension the
            // root `PlatformPageTable`), so the safety requirements of `P::new()` are partially
            // met. It's up to the caller of `take_page()` to ensure the page is of the correct
            // type.
            let page = P::new(addr);
            pte.clear();
            Some(page)
        } else {
            None
        }
    }

    /// Mark the page invalid in the page table that owns it and return it.
    /// Returns the page if a valid leaf, otherwise, None.
    ///
    /// # Safety
    ///
    /// See the safety requirements for `take_page()`.
    pub unsafe fn invalidate_page<P: PhysPage>(self) -> Option<P> {
        let addr = self.page_addr()?;
        if let ValidTableEntryMut::Leaf(pte, _) = self {
            // See comments above in `take_page()`.
            let page = P::new(addr);
            pte.invalidate();
            Some(page)
        } else {
            None
        }
    }
}

/// Holds the address of a page table for a given level in the paging structure.
/// `PageTable`s are loaned by top level pages translation schemes such as `Sv48x4` and `Sv48`
/// (implementors of `PlatformPageTable`).
pub(crate) struct PageTable<'a, T: PlatformPageTable> {
    table_addr: SupervisorPageAddr,
    level: T::Level,
    // Bind our lifetime to that of the top-level `PlatformPageTable`.
    phantom: PhantomData<&'a mut T>,
}

impl<'a, T: PlatformPageTable> PageTable<'a, T> {
    /// Creates a `PageTable` from the root of a `PlatformPageTable`.
    pub fn from_root(owner: &'a mut T) -> Self {
        Self {
            table_addr: owner.get_root_address(),
            level: owner.root_level(),
            phantom: PhantomData,
        }
    }

    /// Creates a `PageTable` from a raw `Pte` at the given level.
    ///
    /// # Safety
    ///
    /// The given `Pte` must be valid and point to an intermediate paging structure at the specified
    /// level. The pointed-to page table must be owned by the same `PlatformPageTable` that owns the
    /// `Pte`.
    pub unsafe fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        assert!(pte.valid());
        // Beyond the root, every level must be only one 4kB page.
        assert_eq!(level.table_pages(), 1);
        Self {
            // Unwrap ok, PFNs are always 4kB-aligned.
            table_addr: PageAddr::from_pfn(pte.pfn(), PageSize::Size4k).unwrap(),
            level,
            phantom: PhantomData,
        }
    }

    /// Returns the `PageTableLevel` this table is at.
    pub fn level(&self) -> T::Level {
        self.level
    }

    /// Returns a mutable reference to the entry at the given guest address.
    fn entry_mut(&mut self, addr: RawAddr<T::MappedAddressSpace>) -> &'a mut Pte {
        let index = self.index_from_addr(addr).index(); // Guaranteed to be in range.
        let pte_addr = self.table_addr.bits() + index * (core::mem::size_of::<Pte>() as u64);
        let pte = unsafe { (pte_addr as *mut Pte).as_mut().unwrap() };
        pte
    }

    /// Map a leaf entry such that the given `addr` will map to `spa` after translation, with the
    /// specified permissions.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `spa` references a page uniquely owned by the root
    /// `PlatformPageTabel`.
    pub unsafe fn map_leaf(
        &mut self,
        addr: PageAddr<T::MappedAddressSpace>,
        spa: SupervisorPageAddr,
        perms: PteLeafPerms,
    ) -> Result<()> {
        let level = self.level;
        let entry = self.entry_mut(RawAddr::from(addr));
        if entry.valid() {
            return Err(Error::MappingExists);
        }
        assert_eq!(spa.size(), level.leaf_page_size());

        let status = {
            let mut s = PteFieldBits::leaf_with_perms(perms);
            s.set_bit(PteFieldBit::Valid);
            s.set_bit(PteFieldBit::User);
            s
        };
        entry.set(spa.pfn(), &status);
        Ok(())
    }

    fn index_from_addr(&self, addr: RawAddr<T::MappedAddressSpace>) -> PageTableIndex<T> {
        PageTableIndex::from_addr(addr.bits(), self.level)
    }

    /// Returns a mutable reference to the entry at this level for the address being translated.
    pub fn entry_for_addr_mut(
        &mut self,
        addr: RawAddr<T::MappedAddressSpace>,
    ) -> TableEntryMut<'a, T> {
        let level = self.level;
        TableEntryMut::from_pte(self.entry_mut(addr), level)
    }

    /// Returns the next page table level for the given address to translate.
    /// If the next level isn't yet filled, consumes a `free_page` and uses it to map those entries.
    pub fn next_level_or_fill_fn(
        &mut self,
        addr: RawAddr<T::MappedAddressSpace>,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<PageTable<'a, T>> {
        let v = match self.entry_for_addr_mut(addr) {
            TableEntryMut::Valid(v) => v,
            TableEntryMut::Invalid(pte, level) => {
                // TODO: Verify ownership of PTE pages.
                pte.set(
                    get_pte_page().ok_or(Error::InsufficientPtePages)?.pfn(),
                    &PteFieldBits::non_leaf(),
                );
                ValidTableEntryMut::from_pte(pte, level)
            }
        };
        v.table().ok_or(Error::LeafEntryNotTable)
    }
}

/// An index to an entry in a page table.
pub(crate) trait PteIndex {
    /// Returns the offset in bytes of the index
    fn offset(&self) -> u64 {
        self.index() * core::mem::size_of::<u64>() as u64
    }

    /// get the underlying index
    fn index(&self) -> u64;
}

/// Guarantees that the contained index is within the range of the page table type it is constructed
/// for.
#[derive(Copy, Clone)]
pub(crate) struct PageTableIndex<T: PlatformPageTable> {
    index: u64,
    level: PhantomData<T::Level>,
}

impl<T: PlatformPageTable> PageTableIndex<T> {
    /// Get an index from the address to be translated
    pub fn from_addr(addr: u64, level: T::Level) -> Self {
        let addr_bit_mask = (1 << level.addr_width()) - 1;
        let index = (addr >> level.addr_shift()) & addr_bit_mask;
        Self {
            index,
            level: PhantomData,
        }
    }
}

impl<T: PlatformPageTable> PteIndex for PageTableIndex<T> {
    fn index(&self) -> u64 {
        self.index
    }
}

/// A page table for a S or U mode. It's enabled by storing its root address in `satp`.
/// Examples include `Sv39`, `Sv48`, or `Sv57`
pub trait FirstStagePageTable: PlatformPageTable<MappedAddressSpace = SupervisorVirt> {
    /// `SATP_VALUE` must be set to the paging mode stored in register satp.
    const SATP_VALUE: u64;
}

/// A page table for a VM. It's enabled by storing its root address in `hgatp`.
/// Examples include `Sv39x4`, `Sv48x4`, or `Sv57x4`
pub trait GuestStagePageTable: PlatformPageTable<MappedAddressSpace = GuestPhys> {
    /// `HGATP_VALUE` must be set to the paging mode stored in register hgatp.
    const HGATP_VALUE: u64;
}

/// A page table for a given addressing type.
pub trait PlatformPageTable: Sized {
    type Level: PageTableLevel;
    type MappedAddressSpace: AddressSpace;

    /// The alignement requirement of the top level page table.
    const TOP_LEVEL_ALIGN: u64;

    /// Creates a new page table root from the provided `pages` that must be at least
    /// `root_level().table_pages()` in length and aligned to `T::TOP_LEVEL_ALIGN`.
    fn new(pages: SequentialPages, owner: PageOwnerId, page_tracker: PageTracker) -> Result<Self>;

    /// Returns an ref to the systems physical pages map.
    fn page_tracker(&self) -> PageTracker;

    /// Returns the owner Id for this page table.
    fn page_owner_id(&self) -> PageOwnerId;

    /// Returns the address of the top level page table.
    /// This is the value that should be written to satp/hgatp to start using the page tables.
    fn get_root_address(&self) -> SupervisorPageAddr;

    /// Returns the root `PageTableLevel`.
    fn root_level(&self) -> Self::Level;

    /// Calculates the number of PTE pages that are needed to map all pages for `num_pages` maped
    /// pages.
    fn max_pte_pages(num_pages: u64) -> u64;

    /// Handles a fault from the owner of this page table. Until page permissions are added,
    /// this will only happen when a page has been loaned to another guest. That is valid if the
    /// guest has exited, in which case this fixed the PTE entry and returns true. False will be
    /// returned if the page is still owned by the guest it was loaned to, or if the entry is
    /// invalid.
    fn do_fault(&mut self, addr: RawAddr<Self::MappedAddressSpace>) -> bool;

    /// Maps a page for translation with address `addr`, using `get_pte_page` to fetch new page-table
    /// pages if necessary.
    ///
    /// TODO: Page permissions.
    fn map_page<P: PhysPage>(
        &mut self,
        addr: PageAddr<Self::MappedAddressSpace>,
        page_to_map: P,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        self.do_map_page::<P>(addr, page_to_map.addr(), PteLeafPerms::RWX, get_pte_page)
    }

    /// Same as `map_page()`, but also extends `data_measure` with the address and contents of the
    /// page to be mapped.
    fn map_page_with_measurement(
        &mut self,
        addr: PageAddr<Self::MappedAddressSpace>,
        page_to_map: Page,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
        data_measure: &mut dyn DataMeasure,
    ) -> Result<()> {
        self.do_map_page::<Page>(addr, page_to_map.addr(), PteLeafPerms::RWX, get_pte_page)?;
        data_measure.add_page(addr.bits(), page_to_map.as_bytes());
        Ok(())
    }

    /// Invalidates and clears the PTE mapping `addr`, returning the mapped host page.
    fn unmap_page<P: PhysPage>(
        &mut self,
        addr: PageAddr<Self::MappedAddressSpace>,
    ) -> Result<UnmappedPhysPage<P>> {
        let entry = self.get_4k_leaf_with_type(RawAddr::from(addr), P::mem_type())?;
        let page = unsafe {
            // Safe since we've verified the typing of the page.
            entry.take_page().unwrap()
        };
        Ok(UnmappedPhysPage::new(page))
    }

    /// Like `unmap_page` but leaves the PFN in the PTE intact.
    fn invalidate_page<P: PhysPage>(
        &mut self,
        addr: PageAddr<Self::MappedAddressSpace>,
    ) -> Result<UnmappedPhysPage<P>> {
        let entry = self.get_4k_leaf_with_type(RawAddr::from(addr), P::mem_type())?;
        let page = unsafe {
            // Safe since we've verified the typing of the page.
            entry.invalidate_page().unwrap()
        };
        Ok(UnmappedPhysPage::new(page))
    }

    /// Returns an iterator to invalidated pages for the given range.
    /// Guarantees that the full range of pages can be unmapped.
    fn invalidate_range<P: PhysPage>(
        &mut self,
        addr: PageAddr<Self::MappedAddressSpace>,
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
}

pub(crate) trait PlatformPageTableHelpers: PlatformPageTable {
    /// Walks the page table from the root for `vaddr` until `pred` returns true. Returns `None` if
    /// a leaf is reached without `pred` being met.
    fn walk_until<P>(
        &mut self,
        vaddr: RawAddr<Self::MappedAddressSpace>,
        mut pred: P,
    ) -> Option<TableEntryMut<Self>>
    where
        P: FnMut(&TableEntryMut<Self>) -> bool,
    {
        use TableEntryMut::*;
        use ValidTableEntryMut::*;
        let mut entry = PageTable::from_root(self).entry_for_addr_mut(vaddr);
        while !pred(&entry) {
            if let Valid(Table(mut t)) = entry {
                entry = t.entry_for_addr_mut(vaddr);
            } else {
                return None;
            }
        }
        Some(entry)
    }

    /// Walks the page table to a valid leaf entry mapping `mapped_addr`. Returns `None` if
    /// `mapped_addr` is not mapped.
    fn walk_to_leaf(
        &mut self,
        mapped_addr: RawAddr<Self::MappedAddressSpace>,
    ) -> Option<ValidTableEntryMut<Self>> {
        use TableEntryMut::*;
        use ValidTableEntryMut::*;
        self.walk_until(mapped_addr, |e| matches!(e, Valid(Leaf(..))))
            .and_then(TableEntryMut::as_valid_entry)
    }

    /// Walks the page table until an invalid entry that would map `mapped_addr` is encountered.
    /// Returns `None` if `mapped_addr` is mapped.
    fn walk_until_invalid(
        &mut self,
        mapped_addr: RawAddr<Self::MappedAddressSpace>,
    ) -> Option<TableEntryMut<Self>> {
        use TableEntryMut::*;
        self.walk_until(mapped_addr, |e| matches!(e, Invalid(..)))
    }

    /// Checks the ownership and typing of the page at `page_addr` and then creates a translation
    /// for `vaddr` to `spa` with the given permissions, filling in any intermediate page tables
    /// using `get_pte_page` as necessary.
    fn do_map_page<P: PhysPage>(
        &mut self,
        vaddr: PageAddr<Self::MappedAddressSpace>,
        spa: SupervisorPageAddr,
        perms: PteLeafPerms,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        if spa.size().is_huge() {
            return Err(Error::PageSizeNotSupported(spa.size()));
        }
        if self.page_tracker().owner(spa) != Some(self.page_owner_id()) {
            return Err(Error::PageNotOwned);
        }
        if self.page_tracker().mem_type(spa) != Some(P::mem_type()) {
            return Err(Error::PageTypeMismatch);
        }

        let mut table = PageTable::from_root(self);
        while table.level().leaf_page_size() != spa.size() {
            table = table.next_level_or_fill_fn(RawAddr::from(vaddr), get_pte_page)?;
        }
        unsafe {
            // Safe since we've verified ownership of the page.
            table.map_leaf(vaddr, spa, perms)?
        };
        Ok(())
    }

    /// Returns the valid 4kB leaf PTE mapping `vaddr` if the mapped page matches the specified
    /// `mem_type`.
    fn get_4k_leaf_with_type(
        &mut self,
        vaddr: RawAddr<Self::MappedAddressSpace>,
        mem_type: MemType,
    ) -> Result<ValidTableEntryMut<Self>> {
        let page_tracker = self.page_tracker();
        let entry = self.walk_to_leaf(vaddr).ok_or(Error::PageNotOwned)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        // Unwrap ok, must be a leaf entry.
        let spa = entry.page_addr().unwrap();
        if page_tracker.mem_type(spa) != Some(mem_type) {
            return Err(Error::PageTypeMismatch);
        }
        Ok(entry)
    }
}

impl<T: PlatformPageTable> PlatformPageTableHelpers for T {}

pub struct InvalidateIter<'a, T: PlatformPageTable, P: PhysPage> {
    owner: &'a mut T,
    curr: PageAddr<T::MappedAddressSpace>,
    count: u64,
    phantom: PhantomData<P>,
}

impl<'a, T: PlatformPageTable, P: PhysPage> InvalidateIter<'a, T, P> {
    pub fn new(owner: &'a mut T, curr: PageAddr<T::MappedAddressSpace>, count: u64) -> Self {
        Self {
            owner,
            curr,
            count,
            phantom: PhantomData,
        }
    }
}

impl<'a, T: PlatformPageTable, P: PhysPage> Iterator for InvalidateIter<'a, T, P> {
    type Item = UnmappedPhysPage<P>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }

        let this_page = self.curr;
        self.curr = self.curr.checked_add_pages(1).unwrap();
        self.count -= 1;
        self.owner.invalidate_page(this_page).ok()
    }
}
