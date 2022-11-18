// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::pte::{Pte, PteFieldBits, PteLeafPerms};
use core::marker::PhantomData;
use page_tracking::PageTracker;
use riscv_pages::*;
use spin::Mutex;

pub(crate) const ENTRIES_PER_PAGE: u64 = 4096 / 8;

/// Error in creating or modifying a page table.
#[derive(Debug)]
pub enum Error {
    /// Failure to create a root page table because the root requires more pages.
    InsufficientPages(SequentialPages<InternalClean>),
    /// Failure to allocate a page to hold the PTEs for the given mapping.
    InsufficientPtePages,
    /// Attempt to access a middle-level page table, but found a leaf.
    LeafEntryNotTable,
    /// Attempt to access a leaf page table, but found a middle-level.
    TableEntryNotLeaf,
    /// Failure creating a root page table at an address that isn't aligned as required.
    MisalignedPages(SequentialPages<InternalClean>),
    /// Failure creating a root page table with an unsupported page size.
    UnsupportedPageSizePages(SequentialPages<InternalClean>, PageSize),
    /// The requested page size isn't (yet) handled by the hypervisor.
    PageSizeNotSupported(PageSize),
    /// The requested size doesn't match what is found
    PageSizeMismatch(PageSize, PageSize),
    /// Attempt to create a mapping over an existing one.
    MappingExists,
    /// The requested range isn't mapped.
    PageNotMapped,
    /// The requested range couldn't be removed from the page table.
    PageNotUnmappable,
    /// The requested range isn't converted.
    PageNotConverted,
    /// Attempt to lock a PTE that is already locked.
    PteLocked,
    /// Attempt to unlock a PTE that is not locked.
    PteNotLocked,
    /// At least one of the pages for building a table root were not owned by the table's owner.
    RootPageNotOwned(SequentialPages<InternalClean>),
    /// The page was not in the range that the `GuestStageMapper` covers.
    OutOfMapRange,
    /// The page cannot be shared
    PageNotShareable,
    /// Address is not page aligned
    AddressMisaligned(u64),
    /// The user-supplied predicate failed.
    PredicateFailed,
    /// An address range causes overflow.
    AddressOverflow,
}
/// Hold the result of page table operations.
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

/// An invalid page table entry that is not being used for any purpose.
enum UnusedEntry {}

/// A page table entry that was valid but was later invalidated (e.g. for conversion). The PFN of
/// the page table entry holds the PFN of the page that was previously mapped.
enum InvalidatedEntry {}

/// An invalid page table entry that has been locked in prepartion for mapping.
enum LockedEntry {}

/// A valid page table entry that provides translation for a page of memory.
enum LeafEntry {}

/// A valid page table entry that points to a next level page table.
enum NextTableEntry {}

// Convenience aliases for the various types of PTEs.
type UnusedPte<'a, T> = TableEntryMut<'a, T, UnusedEntry>;
type InvalidatedPte<'a, T> = TableEntryMut<'a, T, InvalidatedEntry>;
type LockedPte<'a, T> = TableEntryMut<'a, T, LockedEntry>;
type LeafPte<'a, T> = TableEntryMut<'a, T, LeafEntry>;
type PageTablePte<'a, T> = TableEntryMut<'a, T, NextTableEntry>;

enum TableEntryType<'a, T: PagingMode> {
    Unused(UnusedPte<'a, T>),
    Invalidated(InvalidatedPte<'a, T>),
    Locked(LockedPte<'a, T>),
    Leaf(LeafPte<'a, T>),
    Table(PageTablePte<'a, T>),
}

impl<'a, T: PagingMode> TableEntryType<'a, T> {
    /// Creates a `TableEntryType` by inspecting the passed `pte` and determining its type.
    fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        use TableEntryType::*;
        if !pte.valid() {
            if pte.locked() {
                Locked(LockedPte::new(pte, level))
            } else if pte.pfn().bits() != 0 {
                Invalidated(InvalidatedPte::new(pte, level))
            } else {
                Unused(UnusedPte::new(pte, level))
            }
        } else if !pte.leaf() {
            Table(PageTablePte::new(pte, level))
        } else {
            Leaf(LeafPte::new(pte, level))
        }
    }
}

/// A mutable reference to a page table entry of a particular type.
struct TableEntryMut<'a, T: PagingMode, S> {
    pte: &'a mut Pte,
    level: T::Level,
    state: PhantomData<S>,
}

impl<'a, T: PagingMode, S> TableEntryMut<'a, T, S> {
    /// Creates a new `TableEntryMut` from the raw `pte` at `level`.
    fn new(pte: &'a mut Pte, level: T::Level) -> Self {
        Self {
            pte,
            level,
            state: PhantomData,
        }
    }

    /// Returns the `PageTableLevel` this entry is at.
    fn level(&self) -> T::Level {
        self.level
    }
}

impl<'a, T: PagingMode> UnusedPte<'a, T> {
    /// Marks this invalid PTE as valid and maps it to a next-level page table at `table_paddr`.
    /// Returns this entry as a valid table entry.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `table_paddr` references a page-table page uniquely owned by
    /// the root `GuestStagePageTable`.
    unsafe fn map_table(self, table_paddr: SupervisorPageAddr) -> PageTablePte<'a, T> {
        self.pte.set(table_paddr.pfn(), &PteFieldBits::non_leaf());
        PageTablePte::new(self.pte, self.level)
    }

    /// Locks the PTE for mapping.
    fn lock(self) -> LockedPte<'a, T> {
        self.pte.lock();
        LockedPte::new(self.pte, self.level)
    }
}

impl<'a, T: PagingMode> InvalidatedPte<'a, T> {
    /// Returns the physical address of the page this PTE would map if it were valid.
    fn page_addr(&self) -> SupervisorPageAddr {
        // Unwrap ok since this must have been a valid PTE at some point, in which case the PFN must
        // be properly aligned for the level.
        PageAddr::from_pfn(self.pte.pfn(), self.level.leaf_page_size()).unwrap()
    }

    /// Locks the PTE for mapping.
    fn lock(self) -> LockedPte<'a, T> {
        self.pte.lock();
        LockedPte::new(self.pte, self.level)
    }

    /// Clears the PTE.
    fn clear(self) {
        self.pte.clear();
    }
}

impl<'a, T: PagingMode> LockedPte<'a, T> {
    /// Marks this PTE as valid and maps it to `paddr` with the specified permissions. Returns this
    /// entry as a valid leaf entry.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `paddr` references a page uniquely owned by the root
    /// `GuestStagePageTable`.
    unsafe fn map_leaf(self, paddr: SupervisorPageAddr, perms: PteFieldBits) -> LeafPte<'a, T> {
        assert!(paddr.is_aligned(self.level.leaf_page_size()));
        self.pte.set(paddr.pfn(), &perms);
        LeafPte::new(self.pte, self.level)
    }

    /// Unlocks this PTE, returning it to either an unused (zero) PTE, or invalidated PTE.
    fn unlock(self) -> TableEntryType<'a, T> {
        self.pte.unlock();
        use TableEntryType::*;
        if self.pte.pfn().bits() == 0 {
            Unused(UnusedPte::new(self.pte, self.level))
        } else {
            Invalidated(InvalidatedPte::new(self.pte, self.level))
        }
    }
}

impl<'a, T: PagingMode> LeafPte<'a, T> {
    /// Returns the physical address of the page this PTE maps.
    fn page_addr(&self) -> SupervisorPageAddr {
        // Unwrap ok since a valid PTE must contain a valid PFN for this level.
        PageAddr::from_pfn(self.pte.pfn(), self.level.leaf_page_size()).unwrap()
    }

    /// Inavlidates this PTE, returning it as an invalid entry.
    fn invalidate(self) -> InvalidatedPte<'a, T> {
        self.pte.invalidate();
        InvalidatedPte::new(self.pte, self.level)
    }
}

impl<'a, T: PagingMode> PageTablePte<'a, T> {
    /// Returns the base address of the page table this PTE points to.
    fn table_addr(&self) -> SupervisorPageAddr {
        // Unwrap ok, PFNs are always 4kB-aligned.
        PageAddr::from_pfn(self.pte.pfn(), PageSize::Size4k).unwrap()
    }

    /// Returns the `PageTable` that this PTE points to.
    fn table(self) -> PageTable<'a, T> {
        // Safe to create a `PageTable` from the page pointed to by this entry since:
        //  - all valid, non-leaf PTEs must point to an intermediate page table which must
        //    consume exactly one page, and
        //  - all pages pointed to by PTEs in this paging hierarchy are owned by the root
        //    `GuestStagePageTable` and have their lifetime bound to the root.
        unsafe { PageTable::from_pte(self.pte, self.level.next().unwrap()) }
    }
}

/// Holds the address of a page table for a given level in the paging structure.
/// `PageTable`s are loaned by top level pages translation schemes such as `Sv48x4` and `Sv48`
/// (implementors of `PagingMode`).
struct PageTable<'a, T: PagingMode> {
    table_addr: SupervisorPageAddr,
    level: T::Level,
    // Bind our lifetime to that of the top-level `GuestStagePageTable`.
    phantom: PhantomData<&'a mut PageTableInner<T>>,
}

impl<'a, T: PagingMode> PageTable<'a, T> {
    /// Creates a `PageTable` from the root of a `GuestStagePageTable`.
    fn from_root(owner: &'a mut PageTableInner<T>) -> Self {
        Self {
            table_addr: owner.root.base(),
            level: T::root_level(),
            phantom: PhantomData,
        }
    }

    /// Creates a `PageTable` from a raw `Pte` at the given level.
    ///
    /// # Safety
    ///
    /// The given `Pte` must be valid and point to an intermediate paging structure at the specified
    /// level. The pointed-to page table must be owned by the same `GuestStagePageTable` that owns
    /// the `Pte`.
    unsafe fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
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

    /// Returns a mutable reference to the raw PTE at the given guest address.
    fn entry_mut(&mut self, index: PageTableIndex<T>) -> &'a mut Pte {
        let pte_addr =
            self.table_addr.bits() + index.index() * (core::mem::size_of::<Pte>() as u64);
        let pte = unsafe { (pte_addr as *mut Pte).as_mut().unwrap() };
        pte
    }

    /// Returns the index of the page table entry mapping `addr`.
    fn index_from_addr(&self, addr: RawAddr<T::MappedAddressSpace>) -> PageTableIndex<T> {
        PageTableIndex::from_addr(addr.bits(), self.level)
    }

    /// Returns a mutable reference to the entry at this level for the address being translated.
    fn entry_for_addr_mut(
        &mut self,
        addr: RawAddr<T::MappedAddressSpace>,
    ) -> TableEntryType<'a, T> {
        let level = self.level;
        TableEntryType::from_pte(self.entry_mut(self.index_from_addr(addr)), level)
    }

    /// Returns a mutable reference to the entry at this level for the specified index.
    fn entry_for_index_mut(&mut self, index: PageTableIndex<T>) -> TableEntryType<'a, T> {
        let level = self.level;
        TableEntryType::from_pte(self.entry_mut(index), level)
    }

    /// Returns the next page table level for the given address to translate.
    /// If the next level isn't yet filled, consumes a `free_page` and uses it to map those entries.
    fn next_level_or_fill_fn(
        &mut self,
        addr: RawAddr<T::MappedAddressSpace>,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<PageTable<'a, T>> {
        use TableEntryType::*;
        let table_pte = match self.entry_for_addr_mut(addr) {
            Table(t) => t,
            Unused(u) => {
                // TODO: Verify ownership of PTE pages.
                let pt_page = get_pte_page().ok_or(Error::InsufficientPtePages)?;
                unsafe {
                    // Safe since we have unique ownership of `pt_page`.
                    u.map_table(pt_page.addr())
                }
            }
            _ => {
                return Err(Error::MappingExists);
            }
        };
        Ok(table_pte.table())
    }

    /// Releases the pages mapped by this page table, recursing through the paging hierarchy if any
    /// next-level table pointers are encountered.
    fn release_pages(&mut self, page_tracker: PageTracker, owner: PageOwnerId) {
        let iter = PageTableIndexIter::new(self.level);
        for index in iter {
            let entry = self.entry_for_index_mut(index);
            use TableEntryType::*;
            match entry {
                Table(t) => {
                    let table_addr = t.table_addr();
                    // Recursively release the pages in the pointed-to table before dropping the
                    // table itself.
                    t.table().release_pages(page_tracker.clone(), owner);
                    // Safe since we must uniquely own the page if we're using it as a page-table page.
                    let table_page: Page<InternalDirty> = unsafe { Page::new(table_addr) };
                    // Unwrap ok since the page must have been assigned to us.
                    page_tracker.release_page(table_page).unwrap();
                }
                Leaf(l) => {
                    // Unwrap ok since by virtue of being mapped into this page table, we must
                    // uniquely own the page and it must be in a releasable state.
                    page_tracker
                        .release_page_by_addr(l.page_addr(), owner)
                        .unwrap();
                }
                Invalidated(i) => {
                    // Unwrap ok since the only usage of invalid PTEs we currently have is for
                    // converted pages.
                    page_tracker
                        .release_page_by_addr(i.page_addr(), owner)
                        .unwrap();
                }
                _ => (),
            }
        }
    }
}

/// An index to an entry in a page table.
trait PteIndex {
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
struct PageTableIndex<T: PagingMode> {
    index: u64,
    level: PhantomData<T::Level>,
}

impl<T: PagingMode> PageTableIndex<T> {
    /// Get an index from the address to be translated
    fn from_addr(addr: u64, level: T::Level) -> Self {
        let addr_bit_mask = (1 << level.addr_width()) - 1;
        let index = (addr >> level.addr_shift()) & addr_bit_mask;
        Self {
            index,
            level: PhantomData,
        }
    }
}

impl<T: PagingMode> PteIndex for PageTableIndex<T> {
    fn index(&self) -> u64 {
        self.index
    }
}

struct PageTableIndexIter<T: PagingMode> {
    index: u64,
    end: u64,
    level: PhantomData<T::Level>,
}

impl<T: PagingMode> PageTableIndexIter<T> {
    fn new(level: T::Level) -> Self {
        Self {
            index: 0,
            end: 1 << level.addr_width(),
            level: PhantomData,
        }
    }
}

impl<T: PagingMode> Iterator for PageTableIndexIter<T> {
    type Item = PageTableIndex<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.end {
            let index = self.index;
            self.index += 1;
            Some(PageTableIndex {
                index,
                level: PhantomData,
            })
        } else {
            None
        }
    }
}

/// Defines the structure of a particular paging mode.
pub trait PagingMode {
    /// The levels used by this paging mode.
    type Level: PageTableLevel;
    /// The address space that is mapped by this page table.
    type MappedAddressSpace: AddressSpace;

    /// The alignement requirement of the top level page table.
    const TOP_LEVEL_ALIGN: u64;

    /// Returns the root `PageTableLevel` for this type of page table.
    fn root_level() -> Self::Level;

    /// Calculates the number of PTE pages that are needed to map all pages for `num_pages` mapped
    /// pages for this type of page table.
    fn max_pte_pages(num_pages: u64) -> u64;
}

/// A page table for a S or U mode. It's enabled by storing its root address in `satp`.
/// Examples include `Sv39`, `Sv48`, or `Sv57`
pub trait FirstStagePagingMode: PagingMode<MappedAddressSpace = SupervisorVirt> {
    /// `SATP_VALUE` must be set to the paging mode stored in register satp.
    const SATP_VALUE: u64;
}

/// A page table for a VM. It's enabled by storing its root address in `hgatp`.
/// Examples include `Sv39x4`, `Sv48x4`, or `Sv57x4`
pub trait GuestStagePagingMode: PagingMode<MappedAddressSpace = GuestPhys> {
    /// `HGATP_VALUE` must be set to the paging mode stored in register hgatp.
    const HGATP_VALUE: u64;
}

/// The internal state of a paging hierarchy.
#[derive(Debug)]
struct PageTableInner<T: PagingMode> {
    root: SequentialPages<InternalClean>,
    table_type: PhantomData<T>,
}

impl<T: PagingMode> PageTableInner<T> {
    /// Creates a new `PageTableInner` from the pages in `root`.
    fn new(root: SequentialPages<InternalClean>) -> Result<Self> {
        let page_size = root.page_size();
        if page_size.is_huge() {
            return Err(Error::UnsupportedPageSizePages(root, page_size));
        }
        if root.base().bits() & (T::TOP_LEVEL_ALIGN - 1) != 0 {
            return Err(Error::MisalignedPages(root));
        }
        if root.len() < T::root_level().table_pages() as u64 {
            return Err(Error::InsufficientPages(root));
        }

        Ok(Self {
            root,
            table_type: PhantomData,
        })
    }

    /// Walks the page table from the root for `vaddr` until an invalid entry or a valid leaf entry is
    /// encountered.
    fn walk(&mut self, vaddr: RawAddr<T::MappedAddressSpace>) -> TableEntryType<T> {
        let mut entry = PageTable::from_root(self).entry_for_addr_mut(vaddr);
        use TableEntryType::*;
        while let Table(t) = entry {
            entry = t.table().entry_for_addr_mut(vaddr);
        }
        entry
    }

    /// Creates a translation for `vaddr` to `paddr` with the given permissions.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `paddr` references a page uniquely owned by the root
    /// `GuestStagePageTable`.
    unsafe fn map_leaf(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        paddr: SupervisorPageAddr,
        page_size: PageSize,
        perms: PteFieldBits,
    ) -> Result<()> {
        let entry = self.walk(RawAddr::from(vaddr));
        use TableEntryType::*;
        match entry {
            Locked(l) => {
                if l.level().leaf_page_size() != page_size {
                    return Err(Error::PageSizeMismatch(
                        page_size,
                        l.level().leaf_page_size(),
                    ));
                }
                l.map_leaf(paddr, perms);
                Ok(())
            }
            Unused(_) | Invalidated(_) => Err(Error::PteNotLocked),
            Leaf(_) => Err(Error::MappingExists),
            Table(_) => unreachable!(),
        }
    }

    /// Locks the invalid leaf PTE mapping `vaddr`, filling in any missing intermediate page tables
    /// using `get_pte_page`.
    fn lock_leaf_for_mapping(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        page_size: PageSize,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        let mut table = PageTable::from_root(self);
        // match the level to the page size
        while table.level.leaf_page_size() != page_size {
            table = table.next_level_or_fill_fn(RawAddr::from(vaddr), get_pte_page)?;
        }
        let entry = table.entry_for_addr_mut(RawAddr::from(vaddr));
        use TableEntryType::*;
        match entry {
            Invalidated(i) => {
                i.lock();
                Ok(())
            }
            Unused(u) => {
                u.lock();
                Ok(())
            }
            Locked(_) => Err(Error::PteLocked),
            Leaf(_) => Err(Error::MappingExists),
            Table(_) => Err(Error::TableEntryNotLeaf),
        }
    }

    /// Unlocks the leaf PTE mapping `vaddr`.
    fn unlock_leaf(&mut self, vaddr: PageAddr<T::MappedAddressSpace>) -> Result<()> {
        let entry = self.walk(RawAddr::from(vaddr));
        use TableEntryType::*;
        match entry {
            Locked(l) => {
                l.unlock();
                Ok(())
            }
            _ => Err(Error::PteNotLocked),
        }
    }

    /// Returns the valid 4kB leaf PTE mapping `vaddr` if it exists.
    fn get_mapped_4k_leaf(&mut self, vaddr: PageAddr<T::MappedAddressSpace>) -> Result<LeafPte<T>> {
        let entry = self.walk(RawAddr::from(vaddr));
        use TableEntryType::*;
        match entry {
            Leaf(l) => {
                if !l.level().is_leaf() {
                    return Err(Error::PageSizeNotSupported(l.level().leaf_page_size()));
                }
                Ok(l)
            }
            _ => Err(Error::PageNotMapped),
        }
    }

    /// Returns the invalid 4kB leaf PTE mapping `vaddr` if the PFN the PTE references is a
    /// page that was invalidated.
    fn get_invalidated_4k_leaf(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
    ) -> Result<InvalidatedPte<T>> {
        let entry = self.walk(RawAddr::from(vaddr));
        use TableEntryType::*;
        match entry {
            Invalidated(i) => {
                if !i.level().is_leaf() {
                    return Err(Error::PageSizeNotSupported(i.level().leaf_page_size()));
                }
                Ok(i)
            }
            _ => Err(Error::PageNotConverted),
        }
    }
}

/// A paging hierarchy for a given addressing type.
/// This is used for HS and HU paging modes such as Sv48.
/// TODO: Support non-4k page sizes.
pub struct FirstStagePageTable<T: FirstStagePagingMode> {
    inner: Mutex<PageTableInner<T>>,
}

impl<T: FirstStagePagingMode> FirstStagePageTable<T> {
    /// Creates a new page table root from the provided `root` that must be at least
    /// `T::root_level().table_pages()` in length and aligned to `T::TOP_LEVEL_ALIGN`.
    pub fn new(root: Page<InternalClean>) -> Result<Self> {
        let inner = PageTableInner::new(root.into())?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Returns the address of the top level page table. The PFN of this address is what should be
    /// written to the SATP or HGATP CSR to start using the translations provided by this page
    /// table.
    pub fn get_root_address(&self) -> SupervisorPageAddr {
        self.inner.lock().root.base()
    }

    /// Prepares for mapping `num_pages` pages of size `page_size` starting at `addr` in the mapped
    /// address space by locking the target PTEs and populating any intermediate page tables using
    /// `get_pte_page`. Upon success, returns a `PageTableMapper` that is guaranteed to be able to
    /// map the specified range.
    pub fn map_range(
        &self,
        addr: PageAddr<T::MappedAddressSpace>,
        page_size: PageSize,
        num_pages: u64,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<FirstStageMapper<T>> {
        if !addr.is_aligned(page_size) {
            return Err(Error::AddressMisaligned(addr.bits()));
        }
        let iterator = addr
            .iter_from_with_size(page_size)
            .ok_or_else(|| Error::AddressMisaligned(addr.bits()))?;
        addr.checked_add_pages_with_size(num_pages, page_size)
            .ok_or(Error::OutOfMapRange)?;

        let mut inner = self.inner.lock();
        for a in iterator.take(num_pages as usize) {
            inner.lock_leaf_for_mapping(a, page_size, get_pte_page)?;
        }

        Ok(FirstStageMapper::new(self, addr, page_size, num_pages))
    }
}

/// A range of mapped address space that has been locked for mapping. The PTEs are unlocked when
/// this struct is dropped. Mapping a page in this range is guaranteed to succeed as long as the
/// address hasn't already been mapped by this `FirstStageMapper`.
pub struct FirstStageMapper<'a, T: FirstStagePagingMode> {
    owner: &'a FirstStagePageTable<T>,
    vaddr: PageAddr<T::MappedAddressSpace>,
    page_size: PageSize,
    num_pages: u64,
}

impl<'a, T: FirstStagePagingMode> FirstStageMapper<'a, T> {
    /// Creates a new `PageTableMapper` for `num_pages` starting at `vaddr`.
    fn new(
        owner: &'a FirstStagePageTable<T>,
        vaddr: PageAddr<T::MappedAddressSpace>,
        page_size: PageSize,
        num_pages: u64,
    ) -> Self {
        Self {
            owner,
            vaddr,
            page_size,
            num_pages,
        }
    }

    /// Maps `vaddr` to `paddr`, The caller must guarantee that paddr points to a page it is safe to
    /// map in this page table.
    ///
    /// # Safety
    ///
    /// Don't create aliases.
    pub unsafe fn map_addr(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        paddr: PageAddr<SupervisorPhys>,
        pte_perms: PteFieldBits,
    ) -> Result<()> {
        if !paddr.is_aligned(self.page_size) {
            return Err(Error::AddressMisaligned(paddr.bits()));
        }

        if vaddr < self.vaddr
            || vaddr.bits() >= self.vaddr.bits() + self.num_pages * self.page_size as u64
        {
            return Err(Error::OutOfMapRange);
        }
        let mut inner = self.owner.inner.lock();

        inner.map_leaf(vaddr, paddr, self.page_size, pte_perms)
    }
}

/// A paging hierarchy for a given addressing type.
///
/// TODO: Support non-4k page sizes.
pub struct GuestStagePageTable<T: PagingMode> {
    inner: Mutex<PageTableInner<T>>,
    page_tracker: PageTracker,
    owner: PageOwnerId,
}

impl<T: PagingMode> GuestStagePageTable<T> {
    /// Creates a new page table root from the provided `root` that must be at least
    /// `T::root_level().table_pages()` in length and aligned to `T::TOP_LEVEL_ALIGN`.
    pub fn new(
        root: SequentialPages<InternalClean>,
        owner: PageOwnerId,
        page_tracker: PageTracker,
    ) -> Result<Self> {
        // Check that all pages in `root` are owned by `owner`.
        if !root
            .page_addrs()
            .all(|paddr| page_tracker.is_owned(paddr, owner))
        {
            return Err(Error::RootPageNotOwned(root));
        }

        let inner = PageTableInner::new(root)?;
        Ok(Self {
            inner: Mutex::new(inner),
            page_tracker,
            owner,
        })
    }

    /// Returns a reference to the systems physical pages map.
    pub fn page_tracker(&self) -> PageTracker {
        self.page_tracker.clone()
    }

    /// Returns the owner Id for this page table.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.owner
    }

    /// Returns the address of the top level page table. The PFN of this address is what should be
    /// written to the SATP or HGATP CSR to start using the translations provided by this page table.
    pub fn get_root_address(&self) -> SupervisorPageAddr {
        self.inner.lock().root.base()
    }

    /// Handles a fault from the owner of this page table.
    pub fn do_fault(&self, _addr: RawAddr<T::MappedAddressSpace>) -> bool {
        // At the moment we have no reason to take a page fault.
        false
    }

    /// Prepares for mapping `num_pages` pages of size `page_size` starting at `addr` in the mapped
    /// address space by locking the target PTEs and populating any intermediate page tables using
    /// `get_pte_page`. Upon success, returns a `GuestStageMapper` that is guaranteed to be able to
    /// map the specified range.
    pub fn map_range(
        &self,
        addr: PageAddr<T::MappedAddressSpace>,
        page_size: PageSize,
        num_pages: u64,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<GuestStageMapper<T>> {
        if page_size.is_huge() {
            return Err(Error::PageSizeNotSupported(page_size));
        }

        let mut mapper = GuestStageMapper::new(self, addr, 0);
        let mut inner = self.inner.lock();
        for a in addr.iter_from().take(num_pages as usize) {
            inner.lock_leaf_for_mapping(a, page_size, get_pte_page)?;
            mapper.num_pages += 1;
        }

        Ok(mapper)
    }

    fn do_invalidate_range<F>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        len: u64,
        sparse: bool,
        mut pred: F,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_>
    where
        F: FnMut(SupervisorPageAddr) -> bool,
    {
        let num_pages = PageSize::num_4k_pages(len);
        vaddr
            .checked_add_pages(num_pages)
            .ok_or(Error::AddressOverflow)?;
        let mut inner = self.inner.lock();
        // TODO: Support huge pages, making sure we're not invalidating pages that are partially
        // covered by the range.
        for va in vaddr.iter_from().take(num_pages as usize) {
            use TableEntryType::*;
            match inner.walk(va.into()) {
                Leaf(pte) => {
                    if !pte.level().is_leaf() {
                        return Err(Error::PageSizeNotSupported(pte.level().leaf_page_size()));
                    }
                    if !pred(pte.page_addr()) {
                        return Err(Error::PredicateFailed);
                    }
                }
                Unused(_) if sparse => {
                    continue;
                }
                _ => {
                    return Err(Error::PageNotMapped);
                }
            }
        }

        Ok(vaddr
            .iter_from()
            .take(num_pages as usize)
            .filter_map(move |va| {
                // Skip over unmapped PTEs -- we verified there were no invalidated or locked PTEs
                // above.
                Some(inner.get_mapped_4k_leaf(va).ok()?.invalidate().page_addr())
            }))
    }

    /// Verifies the entire virtual address range is mapped and that `pred` returns true for
    /// each page, returning an iterator that yields the pages after invalidating them.
    pub fn invalidate_range<F>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        len: u64,
        pred: F,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_>
    where
        F: FnMut(SupervisorPageAddr) -> bool,
    {
        self.do_invalidate_range(vaddr, len, false, pred)
    }

    /// Like `invalidate_range()`, but the virtual address range need not be entirely populated
    /// with mapped pages. Any holes in the address range must be completely unmapped, i.e. there
    /// must not be invalidated or locked PTEs.
    pub fn invalidate_range_sparse<F>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        len: u64,
        pred: F,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_>
    where
        F: FnMut(SupervisorPageAddr) -> bool,
    {
        self.do_invalidate_range(vaddr, len, true, pred)
    }

    /// Verifies the entire virtual address range is invalidated (or unpopulated) and that `pred`
    /// returns true for each invalidated page, returning an iterator that yields the pages after
    /// clearing their PTEs.
    pub fn unmap_range<F>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        len: u64,
        mut pred: F,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_>
    where
        F: FnMut(SupervisorPageAddr) -> bool,
    {
        let num_pages = PageSize::num_4k_pages(len);
        vaddr
            .checked_add_pages(num_pages)
            .ok_or(Error::AddressOverflow)?;
        let mut inner = self.inner.lock();
        // TODO: Support huge pages, making sure we're not clearing PTEs that only partially cover
        // the address range.
        for va in vaddr.iter_from().take(num_pages as usize) {
            use TableEntryType::*;
            match inner.walk(va.into()) {
                Invalidated(pte) => {
                    if !pte.level().is_leaf() {
                        return Err(Error::PageSizeNotSupported(pte.level().leaf_page_size()));
                    }
                    if !pred(pte.page_addr()) {
                        return Err(Error::PredicateFailed);
                    }
                }
                Unused(_) => {
                    continue;
                }
                _ => {
                    return Err(Error::PageNotUnmappable);
                }
            }
        }

        Ok(vaddr
            .iter_from()
            .take(num_pages as usize)
            .filter_map(move |va| {
                // Skip over non-invalidated PTEs -- we verified there were no mapped or locked PTEs
                // above.
                let pte = inner.get_invalidated_4k_leaf(va).ok()?;
                let paddr = pte.page_addr();
                pte.clear();
                Some(paddr)
            }))
    }

    /// Verifies the entire virtual address range is mapped and that `pred` returns true for
    /// each page, returning an iterator that yields the pages.
    pub fn get_mapped_pages<F>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        len: u64,
        mut pred: F,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_>
    where
        F: FnMut(SupervisorPageAddr) -> bool,
    {
        let num_pages = PageSize::num_4k_pages(len);
        vaddr
            .checked_add_pages(num_pages)
            .ok_or(Error::AddressOverflow)?;
        let mut inner = self.inner.lock();
        // TODO: Support huge pages, making sure mappings aren't partially covering the address
        // range.
        for va in vaddr.iter_from().take(num_pages as usize) {
            let paddr = inner.get_mapped_4k_leaf(va)?.page_addr();
            if !pred(paddr) {
                return Err(Error::PredicateFailed);
            }
        }

        Ok(vaddr.iter_from().take(num_pages as usize).map(move |va| {
            // Unwrap ok: We checked that the entire range was mapped above.
            inner.get_mapped_4k_leaf(va).unwrap().page_addr()
        }))
    }

    /// Verifies the entire virtual address range is invalidated and that `pred` returns true for
    /// each page, returning an iterator that yields the pages.
    pub fn get_invalidated_pages<F>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        len: u64,
        mut pred: F,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_>
    where
        F: FnMut(SupervisorPageAddr) -> bool,
    {
        let num_pages = PageSize::num_4k_pages(len);
        vaddr
            .checked_add_pages(num_pages)
            .ok_or(Error::AddressOverflow)?;
        let mut inner = self.inner.lock();
        // TODO: Support huge pages, making sure mappings aren't partially covering the address
        // range.
        for va in vaddr.iter_from().take(num_pages as usize) {
            let paddr = inner.get_invalidated_4k_leaf(va)?.page_addr();
            if !pred(paddr) {
                return Err(Error::PredicateFailed);
            }
        }

        Ok(vaddr.iter_from().take(num_pages as usize).map(move |va| {
            // Unwrap ok: We checked that the entire range was invalidated above.
            inner.get_invalidated_4k_leaf(va).unwrap().page_addr()
        }))
    }

    /// Returns true if the specified range is completely unpopulated, including pages that are
    /// converted or in the process of conversion.
    pub fn range_is_empty(&self, vaddr: PageAddr<T::MappedAddressSpace>, len: u64) -> bool {
        let mut inner = self.inner.lock();
        vaddr
            .iter_from()
            .take(PageSize::num_4k_pages(len) as usize)
            .all(|va| matches!(inner.walk(va.into()), TableEntryType::Unused(_)))
    }
}

impl<T: PagingMode> Drop for GuestStagePageTable<T> {
    fn drop(&mut self) {
        let mut inner = self.inner.lock();
        // Walk the page table starting from the root, freeing any referenced pages.
        let page_tracker = self.page_tracker.clone();
        let mut table = PageTable::from_root(&mut inner);
        table.release_pages(page_tracker, self.owner);

        // Safe since we uniquely own the pages in self.inner.root.
        let root_pages: SequentialPages<InternalDirty> = unsafe {
            SequentialPages::from_mem_range(inner.root.base(), PageSize::Size4k, inner.root.len())
        }
        .unwrap();
        for p in root_pages {
            // Unwrap ok, the page must've been assigned to us to begin with.
            self.page_tracker.release_page(p).unwrap();
        }
    }
}

/// A range of mapped address space that has been locked for mapping. The PTEs are unlocked when
/// this struct is dropped. Mapping a page in this range is guaranteed to succeed as long as the
/// address hasn't already been mapped by this `GuestStageMapper`.
pub struct GuestStageMapper<'a, T: PagingMode> {
    owner: &'a GuestStagePageTable<T>,
    vaddr: PageAddr<T::MappedAddressSpace>,
    num_pages: u64,
}

impl<'a, T: PagingMode> GuestStageMapper<'a, T> {
    /// Creates a new `GuestStageMapper` for `num_pages` starting at `vaddr`.
    fn new(
        owner: &'a GuestStagePageTable<T>,
        vaddr: PageAddr<T::MappedAddressSpace>,
        num_pages: u64,
    ) -> Self {
        Self {
            owner,
            vaddr,
            num_pages,
        }
    }

    /// Maps `vaddr` to `page_to_map`, consuming `page_to_map`.
    ///
    /// TODO: Page permissions.
    pub fn map_page<P: MappablePhysPage<M>, M: MeasureRequirement>(
        &self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        page_to_map: P,
    ) -> Result<()> {
        if page_to_map.size().is_huge() {
            return Err(Error::PageSizeNotSupported(page_to_map.size()));
        }
        let end_vaddr = self.vaddr.checked_add_pages(self.num_pages).unwrap();
        if vaddr < self.vaddr || vaddr >= end_vaddr {
            return Err(Error::OutOfMapRange);
        }

        let mut inner = self.owner.inner.lock();
        let pte_fields = PteFieldBits::user_leaf_with_perms(PteLeafPerms::RWX);
        unsafe {
            // Safe since we uniquely own page_to_map.
            inner.map_leaf(vaddr, page_to_map.addr(), PageSize::Size4k, pte_fields)
        }
    }
}

impl<'a, T: PagingMode> Drop for GuestStageMapper<'a, T> {
    fn drop(&mut self) {
        let mut inner = self.owner.inner.lock();
        for a in self.vaddr.iter_from().take(self.num_pages as usize) {
            // Ignore the return value since this is expected to fail if the PTE was successfully
            // mapped (which will unlock the PTE), but may succeed if the holder of the
            // GuestStageMapper bailed before having filled the entire range (e.g. because of
            // another failure).
            let _ = inner.unlock_leaf(a);
        }
    }
}
