// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;

use data_measure::data_measure::DataMeasure;
use riscv_pages::*;
use spin::Mutex;

use crate::page_list::{LockedPageList, PageList};
use crate::page_tracking::PageTracker;
use crate::pte::{Pte, PteFieldBit, PteFieldBits, PteLeafPerms};
use crate::TlbVersion;

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
    /// Failure creating a root page table at an address that isn't aligned as required.
    MisalignedPages(SequentialPages<InternalClean>),
    /// The requested page size isn't (yet) handled by the hypervisor.
    PageSizeNotSupported(PageSize),
    /// Attempt to create a mapping over an existing one.
    MappingExists,
    /// The requested range couldn't be removed from the page table.
    PageNotUnmappable,
    /// Attempt to access a non-converted page as confidential.
    PageNotConverted,
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

/// A mutable reference to an entry of a page table, either valid or invalid.
enum TableEntryMut<'a, T: PagingMode> {
    Valid(ValidTableEntryMut<'a, T>),
    Invalid(&'a mut Pte, T::Level),
}

impl<'a, T: PagingMode> TableEntryMut<'a, T> {
    /// Creates a `TableEntryMut` by inspecting the passed `pte` and determining its type.
    fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        if !pte.valid() {
            TableEntryMut::Invalid(pte, level)
        } else {
            TableEntryMut::Valid(ValidTableEntryMut::from_pte(pte, level))
        }
    }

    /// Returns the entry as a `ValidTableEntryMut` if it's a valid PTE.
    fn as_valid_entry(entry: TableEntryMut<'a, T>) -> Option<ValidTableEntryMut<'a, T>> {
        if let TableEntryMut::Valid(v) = entry {
            Some(v)
        } else {
            None
        }
    }

    /// Returns the `PageTableLevel` this entry is at.
    fn level(&self) -> T::Level {
        match self {
            TableEntryMut::Valid(v) => v.level(),
            TableEntryMut::Invalid(_, level) => *level,
        }
    }

    /// Returns the physical address of the page this entry maps if it's a valid leaf, or would
    /// map if it were valid.
    fn page_addr(&self) -> Option<SupervisorPageAddr> {
        match self {
            TableEntryMut::Valid(v) => v.page_addr(),
            TableEntryMut::Invalid(pte, level) => {
                PageAddr::from_pfn(pte.pfn(), level.leaf_page_size())
            }
        }
    }
}

/// A valid entry that contains either a `Leaf` with the host address of the page, or a `Table` with
/// a nested `PageTable`.
enum ValidTableEntryMut<'a, T: PagingMode> {
    Leaf(&'a mut Pte, T::Level),
    Table(PageTable<'a, T>),
}

impl<'a, T: PagingMode> ValidTableEntryMut<'a, T> {
    /// Creates a valid entry from a raw `Pte` at the given level. Asserts if the PTE is invalid.
    fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
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
    fn level(&self) -> T::Level {
        match self {
            ValidTableEntryMut::Leaf(_, level) => *level,
            ValidTableEntryMut::Table(t) => t.level(),
        }
    }

    /// Returns the page table pointed to by this entry or `None` if it is a leaf.
    fn table(self) -> Option<PageTable<'a, T>> {
        match self {
            ValidTableEntryMut::Table(t) => Some(t),
            ValidTableEntryMut::Leaf(..) => None,
        }
    }

    /// Returns the supervisor physical address mapped by this entry or `None` if it is a table
    /// pointer.
    fn page_addr(&self) -> Option<SupervisorPageAddr> {
        if let ValidTableEntryMut::Leaf(pte, level) = self {
            PageAddr::from_pfn(pte.pfn(), level.leaf_page_size())
        } else {
            None
        }
    }

    /// Mark the page invalid in the page table that owns it and return it.
    /// Returns the page if a valid leaf, otherwise, None.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that this page table entry points to a page of memory of the
    /// same type as `P`.
    unsafe fn invalidate_page<P: InvalidatedPhysPage>(self) -> Option<P> {
        let addr = self.page_addr()?;
        if let ValidTableEntryMut::Leaf(pte, level) = self {
            // See comments above in `take_page()`.
            let page = P::new_with_size(addr, level.leaf_page_size());
            pte.invalidate();
            Some(page)
        } else {
            None
        }
    }
}

/// Holds the address of a page table for a given level in the paging structure.
/// `PageTable`s are loaned by top level pages translation schemes such as `Sv48x4` and `Sv48`
/// (implementors of `PagingMode`).
struct PageTable<'a, T: PagingMode> {
    table_addr: SupervisorPageAddr,
    level: T::Level,
    // Bind our lifetime to that of the top-level `PlatformPageTable`.
    phantom: PhantomData<&'a mut PageTableInner<T>>,
}

impl<'a, T: PagingMode> PageTable<'a, T> {
    /// Creates a `PageTable` from the root of a `PlatformPageTable`.
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
    /// level. The pointed-to page table must be owned by the same `PlatformPageTable` that owns the
    /// `Pte`.
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

    /// Returns the `PageTableLevel` this table is at.
    fn level(&self) -> T::Level {
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
    /// `PlatformPageTable`.
    unsafe fn map_leaf(
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
        assert!(spa.is_aligned(level.leaf_page_size()));

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
    fn entry_for_addr_mut(&mut self, addr: RawAddr<T::MappedAddressSpace>) -> TableEntryMut<'a, T> {
        let level = self.level;
        TableEntryMut::from_pte(self.entry_mut(addr), level)
    }

    /// Returns the next page table level for the given address to translate.
    /// If the next level isn't yet filled, consumes a `free_page` and uses it to map those entries.
    fn next_level_or_fill_fn(
        &mut self,
        addr: RawAddr<T::MappedAddressSpace>,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
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
pub trait FirstStagePageTable: PagingMode<MappedAddressSpace = SupervisorVirt> {
    /// `SATP_VALUE` must be set to the paging mode stored in register satp.
    const SATP_VALUE: u64;
}

/// A page table for a VM. It's enabled by storing its root address in `hgatp`.
/// Examples include `Sv39x4`, `Sv48x4`, or `Sv57x4`
pub trait GuestStagePageTable: PagingMode<MappedAddressSpace = GuestPhys> {
    /// `HGATP_VALUE` must be set to the paging mode stored in register hgatp.
    const HGATP_VALUE: u64;
}

/// The internal state of a paging hierarchy.
struct PageTableInner<T: PagingMode> {
    root: SequentialPages<InternalClean>,
    owner: PageOwnerId,
    page_tracker: PageTracker,
    table_type: PhantomData<T>,
}

impl<T: PagingMode> PageTableInner<T> {
    /// Creates a new `PageTableInner` from the pages in `root`.
    fn new(
        root: SequentialPages<InternalClean>,
        owner: PageOwnerId,
        page_tracker: PageTracker,
    ) -> Result<Self> {
        // TODO: Verify ownership of root PT pages.
        if root.page_size().is_huge() {
            return Err(Error::PageSizeNotSupported(root.page_size()));
        }
        if root.base().bits() & (T::TOP_LEVEL_ALIGN - 1) != 0 {
            return Err(Error::MisalignedPages(root));
        }
        if root.len() < T::root_level().table_pages() as u64 {
            return Err(Error::InsufficientPages(root));
        }

        Ok(Self {
            root,
            owner,
            page_tracker,
            table_type: PhantomData,
        })
    }

    /// Walks the page table from the root for `vaddr` until `pred` returns true. Returns `None` if
    /// a leaf is reached without `pred` being met.
    fn walk_until<P>(
        &mut self,
        vaddr: RawAddr<T::MappedAddressSpace>,
        mut pred: P,
    ) -> Option<TableEntryMut<T>>
    where
        P: FnMut(&TableEntryMut<T>) -> bool,
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
        mapped_addr: RawAddr<T::MappedAddressSpace>,
    ) -> Option<ValidTableEntryMut<T>> {
        use TableEntryMut::*;
        use ValidTableEntryMut::*;
        self.walk_until(mapped_addr, |e| matches!(e, Valid(Leaf(..))))
            .and_then(TableEntryMut::as_valid_entry)
    }

    /// Walks the page table until an invalid entry that would map `mapped_addr` is encountered.
    /// Returns `None` if `mapped_addr` is mapped.
    fn walk_until_invalid(
        &mut self,
        mapped_addr: RawAddr<T::MappedAddressSpace>,
    ) -> Option<TableEntryMut<T>> {
        use TableEntryMut::*;
        self.walk_until(mapped_addr, |e| matches!(e, Invalid(..)))
    }

    /// Checks the ownership and typing of the page at `page_addr` and then creates a translation
    /// for `vaddr` to `spa` with the given permissions, filling in any intermediate page tables
    /// using `get_pte_page` as necessary.
    fn do_map_page(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        spa: SupervisorPageAddr,
        page_size: PageSize,
        perms: PteLeafPerms,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        if page_size.is_huge() {
            return Err(Error::PageSizeNotSupported(page_size));
        }
        let mut table = PageTable::from_root(self);
        while table.level().leaf_page_size() != page_size {
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
    fn get_mapped_4k_leaf(
        &mut self,
        vaddr: RawAddr<T::MappedAddressSpace>,
        mem_type: MemType,
    ) -> Result<ValidTableEntryMut<T>> {
        let page_tracker = self.page_tracker.clone();
        let owner = self.owner;
        let entry = self.walk_to_leaf(vaddr).ok_or(Error::PageNotUnmappable)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        // Unwrap ok since we've already verified this a valid leaf.
        let paddr = entry.page_addr().unwrap();
        if !page_tracker.is_mapped_page(paddr, owner, mem_type) {
            return Err(Error::PageNotUnmappable);
        }
        Ok(entry)
    }

    /// Returns the invalid 4kB leaf PTE mapping `vaddr` if the PFN the PTE references is a
    /// page that was converted at a TLB version older than `tlb_version`.
    fn get_converted_4k_leaf(
        &mut self,
        vaddr: RawAddr<T::MappedAddressSpace>,
        mem_type: MemType,
        tlb_version: TlbVersion,
    ) -> Result<TableEntryMut<T>> {
        let page_tracker = self.page_tracker.clone();
        let owner = self.owner;
        let entry = self
            .walk_until_invalid(vaddr)
            .ok_or(Error::PageNotConverted)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        let paddr = entry.page_addr().ok_or(Error::PageNotConverted)?;
        if !page_tracker.is_converted_page(paddr, owner, mem_type, tlb_version) {
            return Err(Error::PageNotConverted);
        }
        Ok(entry)
    }
}

/// A paging hierarchy for a given addressing type.
///
/// TODO: Support non-4k page sizes.
pub struct PlatformPageTable<T: PagingMode> {
    inner: Mutex<PageTableInner<T>>,
}

impl<T: PagingMode> PlatformPageTable<T> {
    /// Creates a new page table root from the provided `root` that must be at least
    /// `T::root_level().table_pages()` in length and aligned to `T::TOP_LEVEL_ALIGN`.
    pub fn new(
        root: SequentialPages<InternalClean>,
        owner: PageOwnerId,
        page_tracker: PageTracker,
    ) -> Result<Self> {
        let inner = PageTableInner::new(root, owner, page_tracker)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Returns a reference to the systems physical pages map.
    pub fn page_tracker(&self) -> PageTracker {
        self.inner.lock().page_tracker.clone()
    }

    /// Returns the owner Id for this page table.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.inner.lock().owner
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

    /// Maps a page for translation with address `addr`, using `get_pte_page` to fetch new page-table
    /// pages if necessary.
    ///
    /// TODO: Page permissions.
    pub fn map_page<P: MappablePhysPage<MeasureOptional>>(
        &self,
        addr: PageAddr<T::MappedAddressSpace>,
        page_to_map: P,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.do_map_page(
            addr,
            page_to_map.addr(),
            page_to_map.size(),
            PteLeafPerms::RWX,
            get_pte_page,
        )
    }

    /// Same as `map_page()`, but also extends `data_measure` with the address and contents of the
    /// page to be mapped.
    pub fn map_page_with_measurement<S: Mappable<M>, M: MeasureRequirement>(
        &self,
        addr: PageAddr<T::MappedAddressSpace>,
        page_to_map: Page<S>,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
        data_measure: &mut dyn DataMeasure,
    ) -> Result<()> {
        {
            let mut inner = self.inner.lock();
            inner.do_map_page(
                addr,
                page_to_map.addr(),
                page_to_map.size(),
                PteLeafPerms::RWX,
                get_pte_page,
            )?;
        }
        data_measure.add_page(addr.bits(), page_to_map.as_bytes());
        Ok(())
    }

    /// Returns a list of invalidated pages for the given range.
    pub fn invalidate_range<P: InvalidatedPhysPage>(
        &self,
        addr: PageAddr<T::MappedAddressSpace>,
        page_size: PageSize,
        num_pages: u64,
    ) -> Result<PageList<P>> {
        if page_size.is_huge() {
            return Err(Error::PageSizeNotSupported(page_size));
        }

        let mut inner = self.inner.lock();
        // First make sure the entire range can be unmapped before we start invalidating things.
        if !addr.iter_from().take(num_pages as usize).all(|a| {
            inner
                .get_mapped_4k_leaf(RawAddr::from(a), P::mem_type())
                .is_ok()
        }) {
            return Err(Error::PageNotUnmappable);
        }

        let mut pages = PageList::new(inner.page_tracker.clone());
        for a in addr.iter_from().take(num_pages as usize) {
            // We verified above that we can safely unwrap here.
            let entry = inner
                .get_mapped_4k_leaf(RawAddr::from(a), P::mem_type())
                .unwrap();
            let page = unsafe {
                // Safe since we've verified the typing of the page.
                entry.invalidate_page().unwrap()
            };
            // Unwrap ok, a just-invalidated page can't be on any other PageList.
            pages.push(page).unwrap();
        }

        Ok(pages)
    }

    /// Returns a list of converted pages that were previously mapped in this page table if they were
    /// invalidated a TLB version older than `tlb_version`. Guarantees that the full range of pages
    /// are converted pages.
    pub fn get_converted_range<P: ConvertedPhysPage>(
        &self,
        addr: PageAddr<T::MappedAddressSpace>,
        page_size: PageSize,
        num_pages: u64,
        tlb_version: TlbVersion,
    ) -> Result<LockedPageList<P::DirtyPage>> {
        if page_size.is_huge() {
            return Err(Error::PageSizeNotSupported(page_size));
        }

        let mut inner = self.inner.lock();
        let page_tracker = inner.page_tracker.clone();
        let mut pages = LockedPageList::new(inner.page_tracker.clone());
        for a in addr.iter_from().take(num_pages as usize) {
            let entry =
                inner.get_converted_4k_leaf(RawAddr::from(a), P::mem_type(), tlb_version)?;
            // Unwrap ok since we've already verified that this page is owned and converted.
            let page = page_tracker
                .get_converted_page::<P>(entry.page_addr().unwrap(), inner.owner, tlb_version)
                .unwrap();
            // Unwrap ok since we have unique ownership of the page and therefore it can't be on
            // any other list.
            pages.push(page).unwrap();
        }

        Ok(pages)
    }
}
