// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use core::slice;

use data_measure::data_measure::DataMeasure;
use riscv_pages::{
    CleanPage, GuestPageAddr, GuestPhysAddr, Page, PageAddr, PageOwnerId, PageSize, RawAddr,
    SequentialPages, SupervisorPageAddr, UnmappedPage,
};

use crate::page_tracking::PageState;
use crate::pte::{Pte, PteFieldBit, PteFieldBits, PteLeafPerms};

pub(crate) const ENTRIES_PER_PAGE: usize = 4096 / 8;

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
}

/// A valid entry that contains either a `Leaf` with the host address of the page, or a `Table` with
/// a nested `PageTable`.
pub(crate) enum ValidTableEntryMut<'a, T: PlatformPageTable> {
    Leaf(&'a mut Pte, T::Level),
    Table(PageTable<'a, T>),
}

impl<'a, T: PlatformPageTable> ValidTableEntryMut<'a, T> {
    /// Asserts if the pte entry is invalid.
    pub fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        assert!(pte.valid()); // TODO -valid pte type to eliminate this runtime assert.
        if pte.leaf() {
            ValidTableEntryMut::Leaf(pte, level)
        } else {
            // Safe to create a 4kB slice of PTEs from the page pointed to by this entry since:
            //  - all valid, non-leaf PTEs must point to an intermediate page table which must
            //    consume exactly one page, and
            //  - all pages pointed to by PTEs are owned by the table and have their lifetime bound
            //    to the table.
            let next_level = level.next().unwrap();
            assert_eq!(next_level.table_pages(), 1);
            let ptes: &'a mut [Pte] = unsafe {
                slice::from_raw_parts_mut(
                    PageAddr::from_pfn(pte.pfn(), PageSize::Size4k)
                        .unwrap()
                        .bits() as *mut Pte,
                    next_level.table_pages() * 4096,
                )
            };
            ValidTableEntryMut::Table(PageTable {
                ptes,
                level: next_level,
            })
        }
    }

    /// Returns the page table pointed to by this entry or `None` if it is a leaf.
    pub fn table(self) -> Option<PageTable<'a, T>> {
        match self {
            ValidTableEntryMut::Table(t) => Some(t),
            ValidTableEntryMut::Leaf(..) => None,
        }
    }

    /// Take the page out of the page table that owns it and return it.
    /// Returns the page if a valid leaf, otherwise, None.
    pub fn take_page(self) -> Option<Page> {
        if let ValidTableEntryMut::Leaf(pte, level) = self {
            let page = unsafe {
                // Safe because the page table ownes this page and is giving up ownership by
                // returning it.
                Page::new(PageAddr::from_pfn(pte.pfn(), level.leaf_page_size())?)
            };
            pte.clear();
            Some(page)
        } else {
            None
        }
    }

    /// Mark the page invalid in the page table that owns it and return it.
    /// Returns the page if a valid leaf, otherwise, None.
    pub fn invalidate_page(self) -> Option<Page> {
        if let ValidTableEntryMut::Leaf(pte, level) = self {
            let page = unsafe {
                // Safe because the page table owns this page and is giving up ownership by
                // returning it.
                Page::new(PageAddr::from_pfn(pte.pfn(), level.leaf_page_size())?)
            };
            pte.invalidate();
            Some(page)
        } else {
            None
        }
    }

    /// Writes the given data at `offset` of the mapped page.
    /// Uses volatile access as the VM where this page is mapped may modify it at any time.
    pub fn write_to_page(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        if let ValidTableEntryMut::Leaf(pte, level) = self {
            let last_offset = offset
                .checked_add(bytes.len() as u64)
                .ok_or(Error::InvalidOffset)?;
            if last_offset <= level.leaf_page_size() as u64 {
                // unwrap is ok because the address at the entry must be correctly aligned.
                let spa = PageAddr::from_pfn(pte.pfn(), level.leaf_page_size())
                    .unwrap()
                    .bits() as *mut u8;
                for (i, c) in bytes.iter().enumerate() {
                    unsafe {
                        // Safe because the page table owns this page and bounds checks were done
                        // above.
                        core::ptr::write_volatile(spa.offset(offset as isize + i as isize), *c);
                    }
                }
                Ok(())
            } else {
                Err(Error::OutOfBounds)
            }
        } else {
            Err(Error::TableEntryNotLeaf)
        }
    }
}

/// Holds a reference to entries for the given level of paging structures.
/// `PageTable`s are loaned by top level pages translation schemes such as `Sv48x4` and `Sv48`
/// (implementors of `PlatformPageTable`).
pub struct PageTable<'a, T: PlatformPageTable> {
    ptes: &'a mut [Pte],
    level: T::Level,
}

impl<'a, T: PlatformPageTable> PageTable<'a, T> {
    /// Creates a page table from the given slice or PTEs.
    /// `ptes` must be `level.table_pages()` long and be aligned to that number of pages.
    pub(crate) fn from_slice(ptes: &'a mut [Pte], level: T::Level) -> Self {
        assert!(ptes.len() == level.table_pages() * ENTRIES_PER_PAGE);
        assert!(((ptes.as_ptr() as usize) & (level.table_pages() * 4096 - 1)) == 0);
        Self { ptes, level }
    }

    /// Returns a mutable reference to the entry at the given guest address.
    fn entry_mut(&mut self, gpa: GuestPhysAddr) -> &mut Pte {
        let index = self.index_from_addr(gpa); // Guaranteed to be in range.

        // Note - This can be changed to an unchecked index as the index is guaranteed to be in
        // range above.
        &mut self.ptes[index.index() as usize]
    }

    /// Map a leaf entry such that the given `guest_phys_addr` will map to `page` after translation.
    pub(crate) fn map_leaf(&mut self, gpa: GuestPhysAddr, page: Page, perms: PteLeafPerms) {
        let level = self.level;
        let entry = self.entry_mut(gpa);
        assert!(!entry.valid()); // Panic if already mapped - TODO - type help
        assert_eq!(page.addr().size(), level.leaf_page_size());

        let status = {
            let mut s = PteFieldBits::leaf_with_perms(perms);
            s.set_bit(PteFieldBit::Valid);
            s.set_bit(PteFieldBit::User);
            s
        };
        entry.set(page, &status);
    }

    fn index_from_addr(&self, gpa: GuestPhysAddr) -> PageTableIndex<T> {
        PageTableIndex::from_addr(gpa.bits(), self.level)
    }

    /// Returns a mutable reference to the entry at this level for the address being translated.
    pub(crate) fn entry_for_addr_mut(&mut self, gpa: GuestPhysAddr) -> TableEntryMut<T> {
        let level = self.level;
        TableEntryMut::from_pte(self.entry_mut(gpa), level)
    }

    /// Returns the next page table level for the given address to translate.
    /// If the next level isn't yet filled, consumes a `free_page` and uses it to map those entries.
    pub(crate) fn next_level_or_fill_fn(
        &mut self,
        gpa: GuestPhysAddr,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<PageTable<T>> {
        let v = match self.entry_for_addr_mut(gpa) {
            TableEntryMut::Valid(v) => v,
            TableEntryMut::Invalid(pte, level) => {
                // TODO: Verify ownership of PTE pages.
                pte.set(
                    get_pte_page().ok_or(Error::InsufficientPtePages)?,
                    &PteFieldBits::non_leaf(),
                );
                ValidTableEntryMut::from_pte(pte, level)
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
pub struct PageTableIndex<T: PlatformPageTable> {
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

/// A page table for a given addressing type.
/// `HGATP_VALUE` must be set to the paging mode stored in register hgatp.
pub trait PlatformPageTable: Sized {
    type Level: PageTableLevel;

    const HGATP_VALUE: u64;
    /// The alignement requirement of the top level page table.
    const TOP_LEVEL_ALIGN: u64;

    /// Creates a new page table root from the provided `pages` that must be at least
    /// `root_level().table_pages()` in length and aligned to `T::TOP_LEVEL_ALIGN`.
    fn new(pages: SequentialPages, owner: PageOwnerId, phys_pages: PageState) -> Result<Self>;

    /// Returns an ref to the systems physical pages map.
    fn phys_pages(&self) -> PageState;

    /// Returns the owner Id for this page table.
    fn page_owner_id(&self) -> PageOwnerId;

    // TODO - page permissions
    // TODO - generic enough to work with satp in addition to hgatp
    /// Maps a page for translation with address `gpa`
    /// Optionally extends measurements
    fn map_page(
        &mut self,
        gpa: GuestPhysAddr,
        page_to_map: Page,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
        data_measure: Option<&mut dyn DataMeasure>,
    ) -> Result<()>;

    /// Unmaps, wipes clean, and returns the host page of the given guest address if that address is
    /// mapped.
    fn unmap_page(&mut self, gpa: GuestPhysAddr) -> Result<CleanPage>;

    /// Like `unmap_page` but leaves the entry in the PTE, marking it as invalid.
    fn invalidate_page(&mut self, gpa: GuestPhysAddr) -> Result<UnmappedPage>;

    /// Returns an iterator to unmapped pages for the given range.
    /// Guarantees that the full range of pages can be unmapped.
    fn unmap_range(&mut self, addr: GuestPageAddr, num_pages: u64) -> Result<UnmapIter<Self>>;

    /// Returns an iterator to unmapped pages for the given range.
    /// Guarantees that the full range of pages can be unmapped.
    fn invalidate_range(
        &mut self,
        addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<InvalidateIter<Self>>;

    /// Returns the address of the top level page table.
    /// This is the value that should be written to satp/hgatp to start using the page tables.
    fn get_root_address(&self) -> SupervisorPageAddr;

    /// Returns the root `PageTableLevel`.
    fn root_level(&self) -> Self::Level;

    /// Calculates the number of PTE pages that are needed to map all pages for `num_pages` maped
    /// pages.
    fn max_pte_pages(num_pages: u64) -> u64;

    /// Handles a fault from the guest that owns this page table. Until page permissions are added,
    /// this will only happen when a page has been loaned to another guest. That is valid if the
    /// guest has exited, in which case this fixed the PTE entry and returns true. False will be
    /// returned if the page is still owned by the guest it was loaned to, or if the entry is
    /// invalid.
    fn do_guest_fault(&mut self, gpa: GuestPhysAddr) -> bool;

    /// Translates the GPA -> SPA, and writes the specified bytes at the given offset
    /// The function will fail if there's no valid GPA -> SPA mapping, or if the
    /// bounds of the page would have been exceeded
    /// Presently supports only 4K pages
    fn write_guest_owned_page(
        &mut self,
        gpa: GuestPhysAddr,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()>;
}

pub struct UnmapIter<'a, T: PlatformPageTable> {
    owner: &'a mut T,
    curr: GuestPageAddr,
    count: u64,
}

impl<'a, T: PlatformPageTable> UnmapIter<'a, T> {
    pub fn new(owner: &'a mut T, curr: GuestPageAddr, count: u64) -> Self {
        Self { owner, curr, count }
    }
}

impl<'a, T: PlatformPageTable> Iterator for UnmapIter<'a, T> {
    type Item = CleanPage;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }

        let this_page = self.curr;
        self.curr = self.curr.checked_add_pages(1).unwrap();
        self.count -= 1;
        self.owner.unmap_page(RawAddr::from(this_page)).ok()
    }
}

pub struct InvalidateIter<'a, T: PlatformPageTable> {
    owner: &'a mut T,
    curr: GuestPageAddr,
    count: u64,
}

impl<'a, T: PlatformPageTable> InvalidateIter<'a, T> {
    pub fn new(owner: &'a mut T, curr: GuestPageAddr, count: u64) -> Self {
        Self { owner, curr, count }
    }
}

impl<'a, T: PlatformPageTable> Iterator for InvalidateIter<'a, T> {
    type Item = UnmappedPage;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }

        let this_page = self.curr;
        self.curr = self.curr.checked_add_pages(1).unwrap();
        self.count -= 1;
        self.owner.invalidate_page(RawAddr::from(this_page)).ok()
    }
}
