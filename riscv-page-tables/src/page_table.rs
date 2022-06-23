// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use page_tracking::{LockedPageList, PageList, PageTracker, TlbVersion};
use riscv_pages::*;
use spin::Mutex;

use crate::pte::{Pte, PteFieldBit, PteFieldBits, PteLeafPerms};

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
    /// Attempt to lock a PTE that is already locked.
    PteLocked,
    /// Attempt to unlock a PTE that is not locked.
    PteNotLocked,
    /// The page was not in the range that the `PageTableMapper` covers.
    OutOfMapRange,
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

// The possible states of a page table entry.
enum InvalidEntry {}
enum LeafEntry {}
enum NextTableEntry {}

// Convenience aliases for the various types of PTEs.
type InvalidPte<'a, T> = TableEntryMut<'a, T, InvalidEntry>;
type LeafPte<'a, T> = TableEntryMut<'a, T, LeafEntry>;
type PageTablePte<'a, T> = TableEntryMut<'a, T, NextTableEntry>;

enum TableEntryType<'a, T: PagingMode> {
    Invalid(InvalidPte<'a, T>),
    Leaf(LeafPte<'a, T>),
    Table(PageTablePte<'a, T>),
}

impl<'a, T: PagingMode> TableEntryType<'a, T> {
    /// Creates a `TableEntryType` by inspecting the passed `pte` and determining its type.
    fn from_pte(pte: &'a mut Pte, level: T::Level) -> Self {
        use TableEntryType::*;
        if !pte.valid() {
            Invalid(InvalidPte::new(pte, level))
        } else if !pte.leaf() {
            Table(PageTablePte::new(pte, level))
        } else {
            Leaf(LeafPte::new(pte, level))
        }
    }

    /// Returns the entry as an inavlid `TableEntryMut` if it's an invalid PTE.
    fn as_invalid_pte(entry: TableEntryType<'a, T>) -> Option<InvalidPte<'a, T>> {
        if let TableEntryType::Invalid(i) = entry {
            Some(i)
        } else {
            None
        }
    }

    /// Returns the entry as a valid leaf `TableEntryMut` if it's a valid leaf PTE.
    fn as_leaf_pte(entry: TableEntryType<'a, T>) -> Option<LeafPte<'a, T>> {
        if let TableEntryType::Leaf(l) = entry {
            Some(l)
        } else {
            None
        }
    }

    /// Returns the `PageTableLevel` this entry is at.
    fn level(&self) -> T::Level {
        use TableEntryType::*;
        match self {
            Invalid(i) => i.level(),
            Leaf(l) => l.level(),
            Table(t) => t.level(),
        }
    }

    /// Unlocks the page table entry.
    fn unlock(&mut self) -> Result<()> {
        use TableEntryType::*;
        match self {
            Invalid(i) => i.unlock(),
            Leaf(l) => l.unlock(),
            Table(t) => t.unlock(),
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

    /// Returns the physical address of the page this entry maps if it's a valid leaf, or would
    /// map if it were valid.
    fn page_addr(&self) -> Option<SupervisorPageAddr> {
        PageAddr::from_pfn(self.pte.pfn(), self.level.leaf_page_size())
    }

    /// Marks the page table entry as locked if it isn't locked already.
    fn lock(&mut self) -> Result<()> {
        if self.pte.locked() {
            return Err(Error::PteLocked);
        }
        self.pte.lock();
        Ok(())
    }

    /// Unlocks the page table entry.
    fn unlock(&mut self) -> Result<()> {
        if !self.pte.locked() {
            return Err(Error::PteNotLocked);
        }
        self.pte.unlock();
        Ok(())
    }
}

impl<'a, T: PagingMode> InvalidPte<'a, T> {
    /// Marks this PTE as valid and maps it to `paddr` with the specified permissions. Returns this
    /// entry as a valid leaf entry.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `paddr` references a page uniquely owned by the root
    /// `PlatformPageTable`.
    unsafe fn map_leaf(self, paddr: SupervisorPageAddr, perms: PteLeafPerms) -> LeafPte<'a, T> {
        assert!(paddr.is_aligned(self.level.leaf_page_size()));
        let status = {
            let mut s = PteFieldBits::leaf_with_perms(perms);
            s.set_bit(PteFieldBit::User);
            s
        };
        self.pte.set(paddr.pfn(), &status);
        LeafPte::new(self.pte, self.level)
    }

    /// Marks this invalid PTE as valid and maps it to a next-level page table at `table_paddr`.
    /// Returns this entry as a valid table entry.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `table_paddr` references a page-table page uniquely owned by
    /// the root `PlatformPageTable`.
    unsafe fn map_table(self, table_paddr: SupervisorPageAddr) -> PageTablePte<'a, T> {
        self.pte.set(table_paddr.pfn(), &PteFieldBits::non_leaf());
        PageTablePte::new(self.pte, self.level)
    }
}

impl<'a, T: PagingMode> LeafPte<'a, T> {
    /// Inavlidates this PTE, returning it as an invalid entry.
    fn invalidate(self) -> InvalidPte<'a, T> {
        self.pte.invalidate();
        InvalidPte::new(self.pte, self.level)
    }
}

impl<'a, T: PagingMode> PageTablePte<'a, T> {
    /// Returns the `PageTable` that this PTE points to.
    fn table(self) -> PageTable<'a, T> {
        // Safe to create a `PageTable` from the page pointed to by this entry since:
        //  - all valid, non-leaf PTEs must point to an intermediate page table which must
        //    consume exactly one page, and
        //  - all pages pointed to by PTEs in this paging hierarchy are owned by the root
        //    `PlatformPageTable` and have their lifetime bound to the root.
        unsafe { PageTable::from_pte(self.pte, self.level.next().unwrap()) }
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
        TableEntryType::from_pte(self.entry_mut(addr), level)
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
            Leaf(_) => return Err(Error::LeafEntryNotTable),
            Invalid(i) => {
                // TODO: Verify ownership of PTE pages.
                let pt_page = get_pte_page().ok_or(Error::InsufficientPtePages)?;
                unsafe {
                    // Safe since we have unique ownership of `pt_page`.
                    i.map_table(pt_page.addr())
                }
            }
        };
        Ok(table_pte.table())
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
    ) -> Option<TableEntryType<T>>
    where
        P: FnMut(&TableEntryType<T>) -> bool,
    {
        use TableEntryType::*;
        let mut entry = PageTable::from_root(self).entry_for_addr_mut(vaddr);
        while !pred(&entry) {
            if let Table(t) = entry {
                entry = t.table().entry_for_addr_mut(vaddr);
            } else {
                return None;
            }
        }
        Some(entry)
    }

    /// Walks the page table to a valid leaf entry mapping `vaddr`. Returns `None` if `vaddr` is not
    /// mapped.
    fn walk_to_leaf(&mut self, vaddr: RawAddr<T::MappedAddressSpace>) -> Option<LeafPte<T>> {
        use TableEntryType::*;
        self.walk_until(vaddr, |e| matches!(e, Leaf(_)))
            .and_then(TableEntryType::as_leaf_pte)
    }

    /// Walks the page table until an invalid entry that would map `vaddr` is encountered. Returns
    /// `None` if `vaddr` is mapped.
    fn walk_until_invalid(
        &mut self,
        vaddr: RawAddr<T::MappedAddressSpace>,
    ) -> Option<InvalidPte<T>> {
        use TableEntryType::*;
        self.walk_until(vaddr, |e| matches!(e, Invalid(_)))
            .and_then(TableEntryType::as_invalid_pte)
    }

    /// Creates a translation for `vaddr` to `paddr` with the given permissions.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `paddr` references a page uniquely owned by the root
    /// `PlatformPageTable`.
    unsafe fn map_4k_leaf(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        paddr: SupervisorPageAddr,
        perms: PteLeafPerms,
    ) -> Result<()> {
        let entry = self
            .walk_until_invalid(RawAddr::from(vaddr))
            .ok_or(Error::MappingExists)?;
        if !entry.level().is_leaf() {
            return Err(Error::PageSizeNotSupported(entry.level().leaf_page_size()));
        }
        entry.map_leaf(paddr, perms);
        Ok(())
    }

    /// Locks the invalid leaf PTE mapping `vaddr`, filling in any missing intermediate page tables
    /// using `get_pte_page`.
    fn lock_4k_leaf_for_mapping(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        let mut table = PageTable::from_root(self);
        while !table.level().is_leaf() {
            table = table.next_level_or_fill_fn(RawAddr::from(vaddr), get_pte_page)?;
        }
        let entry = table.entry_for_addr_mut(RawAddr::from(vaddr));
        TableEntryType::as_invalid_pte(entry)
            .ok_or(Error::MappingExists)?
            .lock()
    }

    /// Unlocks the leaf PTE mapping `vaddr`.
    fn unlock_4k_leaf(&mut self, vaddr: PageAddr<T::MappedAddressSpace>) -> Result<()> {
        let mut entry = self
            .walk_until(RawAddr::from(vaddr), |e| e.level().is_leaf())
            .ok_or(Error::PteNotLocked)?;
        entry.unlock()
    }

    /// Returns the valid 4kB leaf PTE mapping `vaddr` if the mapped page matches the specified
    /// `mem_type`.
    fn get_mapped_4k_leaf(
        &mut self,
        vaddr: PageAddr<T::MappedAddressSpace>,
        mem_type: MemType,
    ) -> Result<LeafPte<T>> {
        let page_tracker = self.page_tracker.clone();
        let owner = self.owner;
        let entry = self
            .walk_to_leaf(RawAddr::from(vaddr))
            .ok_or(Error::PageNotUnmappable)?;
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
        vaddr: PageAddr<T::MappedAddressSpace>,
        mem_type: MemType,
        tlb_version: TlbVersion,
    ) -> Result<InvalidPte<T>> {
        let page_tracker = self.page_tracker.clone();
        let owner = self.owner;
        let entry = self
            .walk_until_invalid(RawAddr::from(vaddr))
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
    ) -> Result<PageTableMapper<T>> {
        if page_size.is_huge() {
            return Err(Error::PageSizeNotSupported(page_size));
        }

        let mut mapper = PageTableMapper::new(self, addr, 0);
        let mut inner = self.inner.lock();
        for a in addr.iter_from().take(num_pages as usize) {
            inner.lock_4k_leaf_for_mapping(a, get_pte_page)?;
            mapper.num_pages += 1;
        }

        Ok(mapper)
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
        if !addr
            .iter_from()
            .take(num_pages as usize)
            .all(|a| inner.get_mapped_4k_leaf(a, P::mem_type()).is_ok())
        {
            return Err(Error::PageNotUnmappable);
        }

        let mut pages = PageList::new(inner.page_tracker.clone());
        for a in addr.iter_from().take(num_pages as usize) {
            // We verified above that we can safely unwrap here.
            let entry = inner.get_mapped_4k_leaf(a, P::mem_type()).unwrap();
            // Unwrap ok, PFN must have been properly aligned in order to have been mapped.
            let paddr = entry.page_addr().unwrap();
            entry.invalidate();
            let page = unsafe {
                // Safe since we've verified the typing of the page.
                P::new(paddr)
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
            let paddr = inner
                .get_converted_4k_leaf(a, P::mem_type(), tlb_version)?
                .page_addr()
                .unwrap();
            // Unwrap ok since we've already verified that this page is owned and converted.
            let page = page_tracker
                .get_converted_page::<P>(paddr, inner.owner, tlb_version)
                .unwrap();
            // Unwrap ok since we have unique ownership of the page and therefore it can't be on
            // any other list.
            pages.push(page).unwrap();
        }

        Ok(pages)
    }
}

/// A range of mapped address space that has been locked for mapping. The PTEs are unlocked when
/// this struct is dropped. Mapping a page in this range is guaranteed to succeed as long as the
/// address hasn't already been mapped by this `PageTableMapper`.
pub struct PageTableMapper<'a, T: PagingMode> {
    owner: &'a PlatformPageTable<T>,
    vaddr: PageAddr<T::MappedAddressSpace>,
    num_pages: u64,
}

impl<'a, T: PagingMode> PageTableMapper<'a, T> {
    /// Creates a new `PageTableMapper` for `num_pages` starting at `vaddr`.
    fn new(
        owner: &'a PlatformPageTable<T>,
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
        unsafe {
            // Safe since we uniquely own page_to_map.
            inner.map_4k_leaf(vaddr, page_to_map.addr(), PteLeafPerms::RWX)
        }
    }
}

impl<'a, T: PagingMode> Drop for PageTableMapper<'a, T> {
    fn drop(&mut self) {
        let mut inner = self.owner.inner.lock();
        for a in self.vaddr.iter_from().take(self.num_pages as usize) {
            // Unwrap ok, the PTEs in this range must have been locked by construction.
            inner.unlock_4k_leaf(a).unwrap();
        }
    }
}
