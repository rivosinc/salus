// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use core::slice;

use riscv_pages::{
    CleanPage, Page, Page4k, AlignedPageAddr, AlignedPageAddr4k, PageOwnerId, PageSize, PageSize2MB, PageSize4k,
    SequentialPages, UnmappedPage,
};

use crate::page_tracking::PageState;
use crate::pte::{Pte, PteFieldBit, PteFieldBits, PteLeafPerms};

pub(crate) const ENTRIES_PER_PAGE: usize = 4096 / 8;

#[derive(Debug)]
pub enum Error {
    InsufficientPages(SequentialPages<PageSize4k>),
    InsufficientPtePages,
    LeafEntryNotTable,
    MisalignedPages(SequentialPages<PageSize4k>),
    SettingOwner(crate::page_tracking::Error),
}
pub type Result<T> = core::result::Result<T, Error>;

/// Represents a level in a multi-level page table.
/// `LeafPageSize`: Page size pointed to by leaf entries at this page level.
/// `NextLevel`: Next level of page table type pointed to by non-leaf entries.
/// ADDR_WIDTH: width in bits of the ppn[i]/vpn[i] field for this index in a virtual/guest physical address.
/// ADDR_SHIFT: starting bit location of the ppn[i]/vpn[i] field in the address.
/// TABLE_PAGES: number of 4k pages used for PTEs.
pub trait PageTableLevel {
    type LeafPageSize: PageSize;
    type NextLevel: PageTableLevel;
    const ADDR_SHIFT: u64;
    const ADDR_WIDTH: u64;
    const TABLE_PAGES: usize;
}

/// A trait for upper levels of the page tables that can hold non-leaf entries (L4-2).
pub trait UpperLevel {}

/// A mutable reference to an entry of a page table, either valid or invalid.
pub(crate) enum TableEntryMut<'a, L: PageTableLevel> {
    Valid(ValidTableEntryMut<'a, L>),
    Invalid(&'a mut Pte),
}

impl<'a, L: PageTableLevel> TableEntryMut<'a, L> {
    /// Creates a `TableEntryMut` by inspecting the passed `pte` and determining its type.
    pub fn from_pte(pte: &'a mut Pte) -> Self {
        if !pte.valid() {
            TableEntryMut::Invalid(pte)
        } else {
            TableEntryMut::Valid(ValidTableEntryMut::from_pte(pte))
        }
    }
}

/// A valid entry that contains either a `Leaf` with the host address of the page, or a `Table` with
/// a nested `PageTable`.
pub(crate) enum ValidTableEntryMut<'a, L: PageTableLevel> {
    Leaf(&'a mut Pte),
    Table(PageTable<'a, L::NextLevel>),
}

impl<'a, L: PageTableLevel> ValidTableEntryMut<'a, L> {
    /// Asserts if the pte entry is invalid.
    pub fn from_pte(pte: &'a mut Pte) -> Self {
        assert!(pte.valid()); // TODO -valid pte type to eliminate this runtime assert.
        if pte.leaf() {
            ValidTableEntryMut::Leaf(pte)
        } else {
            // Safe to create a "Page" of PTEs from the entry as all pages pointed to by table
            // entries are owned by the table and have their lifetime bound to the table.  Because
            // this is an intermediate page table, and it's not a leaf entry, the  entry must point
            // to a 4 kilobyte page of PTES for the next level.
            let ptes: &'a mut [Pte] = unsafe {
                slice::from_raw_parts_mut(
                    AlignedPageAddr::<PageSize4k>::try_from(pte.pfn()).unwrap().bits() as *mut Pte,
                    L::NextLevel::TABLE_PAGES * 4096,
                )
            };
            ValidTableEntryMut::Table(PageTable {
                ptes,
                phantom_level: PhantomData,
            })
        }
    }

    /// Returns the page table pointed to by this entry or `None` if it is a leaf.
    pub fn table(self) -> Option<PageTable<'a, L::NextLevel>> {
        match self {
            ValidTableEntryMut::Table(t) => Some(t),
            ValidTableEntryMut::Leaf(_) => None,
        }
    }

    /// Take the page out of the page table that owns it and return it.
    /// Returns the page if a balid leaf, otherwise, None.
    pub fn take_page(self) -> Option<Page<L::LeafPageSize>> {
        if let ValidTableEntryMut::Leaf(pte) = self {
            let page = unsafe {
                // Safe because the page table ownes this page and is giving up ownership by
                // returning it.
                Page::new(AlignedPageAddr::try_from(pte.pfn()).ok()?)
            };
            pte.clear();
            Some(page)
        } else {
            None
        }
    }

    /// Mark the page invalid in the page table that owns it and return it.
    /// Returns the page if a valid leaf, otherwise, None.
    pub fn invalidate_page(self) -> Option<Page<L::LeafPageSize>> {
        if let ValidTableEntryMut::Leaf(pte) = self {
            let page = unsafe {
                // Safe because the page table owns this page and is giving up ownership by
                // returning it.
                Page::new(AlignedPageAddr::try_from(pte.pfn()).ok()?)
            };
            pte.invalidate();
            Some(page)
        } else {
            None
        }
    }
}

/// Holds a reference to entries for the given level of paging structures.
/// `PageTable`s are loaned by top level pages translation schemes such as `Sv48x4` and `Sv48`
/// (implementors of `PlatformPageTable`).
pub struct PageTable<'a, LEVEL>
where
    LEVEL: PageTableLevel,
{
    ptes: &'a mut [Pte],
    phantom_level: PhantomData<LEVEL>,
}

impl<'a, LEVEL> PageTable<'a, LEVEL>
where
    LEVEL: PageTableLevel,
{
    /// Creates a page table from the given slice or PTEs.
    /// `ptes` must be the `LEVEL::TABLE_PAGES` long and be aligned to that number of pages.
    pub(crate) fn from_slice(ptes: &'a mut [Pte]) -> Self {
        assert!(ptes.len() == LEVEL::TABLE_PAGES * ENTRIES_PER_PAGE);
        assert!(((ptes.as_ptr() as usize) & (LEVEL::TABLE_PAGES * 4096 - 1)) == 0);
        Self {
            ptes,
            phantom_level: PhantomData,
        }
    }

    /// Returns a mutable reference to the entry at the given guest address.
    fn entry_mut(&mut self, guest_phys_addr: u64) -> &mut Pte {
        let index = self.index_from_addr(guest_phys_addr); // Guaranteed to be in range.

        // Note - This can be changed to an unchecked index as the index is guaranteed to be in
        // range above.
        &mut self.ptes[index.index() as usize]
    }

    /// Map a leaf entry such that the given `guest_phys_addr` will map to `page` after translation.
    pub(crate) fn map_leaf(
        &mut self,
        guest_phys_addr: u64,
        page: Page<LEVEL::LeafPageSize>,
        perms: PteLeafPerms,
    ) {
        let entry = self.entry_mut(guest_phys_addr);
        assert!(!entry.valid()); // Panic if already mapped - TODO - type help

        let status = {
            let mut s = PteFieldBits::leaf_with_perms(perms);
            s.set_bit(PteFieldBit::Valid);
            s.set_bit(PteFieldBit::User);
            s
        };
        entry.set(page, &status);
    }

    fn index_from_addr(&self, addr: u64) -> PageTableIndex<LEVEL> {
        PageTableIndex::from_addr(addr)
    }

    /// Returns a mutable reference to the entry at this level for the address being translated.
    pub(crate) fn entry_for_addr_mut(&mut self, guest_phys_addr: u64) -> TableEntryMut<LEVEL>
    where
        LEVEL: PageTableLevel,
    {
        TableEntryMut::from_pte(self.entry_mut(guest_phys_addr))
    }

    /// Returns the next page table level for the given address to translate.
    /// If the next level isn't yet filled, consumes a `free_page` and uses it to map those entries.
    pub(crate) fn next_level_or_fill_fn<F>(
        &mut self,
        guest_phys_addr: u64,
        get_pte_page: &mut F,
    ) -> Result<PageTable<LEVEL::NextLevel>>
    where
        LEVEL: PageTableLevel + UpperLevel,
        F: FnMut() -> Option<Page4k>,
    {
        let v = match self.entry_for_addr_mut(guest_phys_addr) {
            TableEntryMut::Valid(v) => v,
            TableEntryMut::Invalid(pte) => {
                pte.set(
                    get_pte_page().ok_or(Error::InsufficientPtePages)?,
                    &PteFieldBits::non_leaf(),
                );
                ValidTableEntryMut::from_pte(pte)
            }
        };
        v.table().ok_or(Error::LeafEntryNotTable)
    }
}

/// An index to an entry in a page table.
pub trait PteIndex {
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
pub struct PageTableIndex<LEVEL>
where
    LEVEL: PageTableLevel,
{
    index: u64,
    level: PhantomData<LEVEL>,
}

impl<LEVEL> PageTableIndex<LEVEL>
where
    LEVEL: PageTableLevel,
{
    /// Get an index from the address to be translated
    pub fn from_addr(addr: u64) -> Self
    where
        Self: Sized,
    {
        let addr_bit_mask = (1 << LEVEL::ADDR_WIDTH) - 1;
        let index = (addr >> LEVEL::ADDR_SHIFT) & addr_bit_mask;
        Self {
            index,
            level: PhantomData,
        }
    }
}

impl<LEVEL: PageTableLevel> PteIndex for PageTableIndex<LEVEL> {
    fn index(&self) -> u64 {
        self.index
    }
}

/// A page table for a given addressing type.
/// `HGATP_VALUE` must be set to the paging mode stored in register hgatp.
pub trait PlatformPageTable {
    type TLD: PageTableLevel;
    const HGATP_VALUE: u64;
    /// The alignement requirement of the top level page table.
    const TOP_LEVEL_ALIGN: u64;

    /// Creates a new page table from the provided `pages` that provide the 4 pages for the top level
    /// page table directory.
    fn new(
        pages: SequentialPages<PageSize4k>,
        owner: PageOwnerId,
        phys_pages: PageState,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Returns an ref to the systems physical pages map.
    fn phys_pages(&self) -> PageState;

    /// Returns the owner Id for this page table.
    fn page_owner_id(&self) -> PageOwnerId;

    // TODO - page permissions
    // TODO - generic enough to work with satp in addition to hgatp
    /// Maps a 4k page for translation with address `guest_phys_addr`.
    fn map_page_4k<F>(
        &mut self,
        guest_phys_addr: u64,
        page_to_map: Page4k,
        get_pte_page: &mut F,
    ) -> Result<()>
    where
        F: FnMut() -> Option<Page4k>;

    // TODO - page permissions
    // TODO - generic enough to work with satp in addition to hgatp
    /// Maps a 2MB page for translation with address `addr`.
    fn map_page_2mb<F>(
        &mut self,
        addr: u64,
        page_to_map: Page<PageSize2MB>,
        get_pte_page: &mut F,
    ) -> Result<()>
    where
        F: FnMut() -> Option<Page4k>;

    /// Unmaps, wipes clean, and returns the host page of the given guest address if that address is
    /// mapped.
    fn unmap_page(&mut self, guest_phys_addr: u64) -> Option<UnmappedPage>;

    /// Like `unmap_page` but leaves the entry in the PTE, marking it as invalid.
    fn invalidate_page(&mut self, guest_phys_addr: u64) -> Option<UnmappedPage>;

    /// Returns an iterator to unmapped pages for the given range.
    /// Guarantees that the full range of pages can be unmapped.
    fn unmap_range<S: PageSize>(
        &mut self,
        addr: AlignedPageAddr<S>,
        num_pages: u64,
    ) -> Option<UnmapIter<Self>>
    where
        Self: Sized;

    /// Returns an iterator to unmapped pages for the given range.
    /// Guarantees that the full range of pages can be unmapped.
    fn invalidate_range<S: PageSize>(
        &mut self,
        addr: AlignedPageAddr<S>,
        num_pages: u64,
    ) -> Option<InvalidateIter<Self>>
    where
        Self: Sized;

    /// Returns the address of the top level page table.
    /// This is the value that should be written to satp/hgatp to start using the page tables.
    fn get_root_address(&self) -> AlignedPageAddr4k;

    /// Calculates the number of PTE pages that are needed to map all pages for `num_pages` maped
    /// pages.
    fn max_pte_pages(num_pages: u64) -> u64;

    /// Handles a fault from the guest that owns this page table. Until page permissions are added,
    /// this will only happen when a page has been loaned to another guest. That is valid if the
    /// guest has exited, in which case this fixed the PTE entry and returns true. False will be
    /// returned if the page is still owned by the guest it was loaned to, or if the entry is
    /// invalid.
    fn do_guest_fault(&mut self, guest_phys_addr: u64) -> bool;
}

pub struct UnmapIter<'a, T: PlatformPageTable> {
    owner: &'a mut T,
    curr: u64,
    end: u64,
    page_size: u64,
}

impl<'a, T: PlatformPageTable> UnmapIter<'a, T> {
    pub fn new(owner: &'a mut T, curr: u64, count: u64, page_size: u64) -> Self {
        Self {
            owner,
            curr,
            end: curr + page_size * count,
            page_size,
        }
    }
}

impl<'a, T: PlatformPageTable> Iterator for UnmapIter<'a, T> {
    type Item = CleanPage;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == self.end {
            return None;
        }

        let this_page = self.curr;
        self.curr += self.page_size;
        self.owner.unmap_page(this_page).map(CleanPage::from)
    }
}

pub struct InvalidateIter<'a, T: PlatformPageTable> {
    owner: &'a mut T,
    curr: u64,
    end: u64,
    page_size: u64,
}

impl<'a, T: PlatformPageTable> InvalidateIter<'a, T> {
    pub fn new(owner: &'a mut T, curr: u64, count: u64, page_size: u64) -> Self {
        Self {
            owner,
            curr,
            end: curr + page_size * count,
            page_size,
        }
    }
}

impl<'a, T: PlatformPageTable> Iterator for InvalidateIter<'a, T> {
    type Item = UnmappedPage;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == self.end {
            return None;
        }

        let this_page = self.curr;
        self.curr += self.page_size;
        self.owner.invalidate_page(this_page)
    }
}
