// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::num::NonZeroU64;
use riscv_pages::*;

use crate::collections::RawPageVec;
use crate::{
    HwMemMap, HwMemRegionType, HwReservedMemType, PageTrackingError, PageTrackingResult, TlbVersion,
};

/// Tracks the owners of a page. Ownership is nested in order to establish a "chain-of-custody"
/// for the page.
type PageOwnerVec = ArrayVec<PageOwnerId, MAX_PAGE_OWNERS>;

/// `PageState` holds the current ownership status of a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    /// Not present, reserved, or otherwise not usable.
    Reserved,

    /// Page is unowned. No pages should be in this state after startup: they must either
    /// be reserved, or owned by the hypervisor / VMs.
    Free,

    /// Page is mapped into the address space of the current owner.
    Mapped,

    /// Page is mapped into the address space of the current owner, but has been shared with
    /// a child-VM. Shared pages cannot be converted, and are reverted to the Mapped state when
    /// the reference count maintained by the inner u64 drops to 0.
    Shared(u64),

    /// Page is used to store hypervisor-internal state for the current owner, e.g. to back per-VM
    /// data structures or as a page-table page for the VM.
    VmState,

    /// Page is used to store hypervisor-internal state for the hypervisor. These pages cannot be
    /// reassigned.
    HypState,

    /// Page has been invalidated and started the conversion operation at the given TLB version in
    /// the current owner's address space.
    Converting(TlbVersion),

    /// Page has been invalidated and started the unassignment operation at the given TLB version in
    /// the current owner's address space. Upon completion, the page is returned to the previous owner
    /// in the `Converted` state.
    Unassigning(TlbVersion),

    /// Page has completed the conversion or unassignment operation and is eligible for assignment or
    /// to be reclaimed. The page must be locked exclusively before it can be assigned or reclaimed.
    Converted,

    /// A Converted page that has been locked exclusively for assignment or reclaim.
    ConvertedLocked,
}

/// The maximum length for an ownership chain. Enough for the host VM to assign to a guest VM
/// without further nesting. An empty owners vector indicates that the page is hypervisor-owned.
pub const MAX_PAGE_OWNERS: usize = 2;

/// Holds ownership and typing details about a particular page in the system memory map.
#[derive(Clone, Debug)]
pub struct PageInfo {
    mem_type: MemType,
    state: PageState,
    owners: PageOwnerVec,
    // Address of the next page in the list if != None.
    link: Option<NonZeroU64>,
}

impl PageInfo {
    /// Creates a new `PageInfo` representing a free RAM page.
    pub fn new() -> Self {
        Self {
            mem_type: MemType::Ram,
            state: PageState::Free,
            owners: PageOwnerVec::new(),
            link: None,
        }
    }

    /// Creates a new `PageInfo` representing a RAM page that is initially owned by the hypervisor.
    pub fn new_hypervisor_owned() -> Self {
        Self {
            mem_type: MemType::Ram,
            state: PageState::ConvertedLocked,
            owners: PageOwnerVec::new(),
            link: None,
        }
    }

    /// Creates a new `PageInfo` representing a RAM page that is forever reserved.
    pub fn new_reserved() -> Self {
        Self {
            mem_type: MemType::Ram,
            state: PageState::Reserved,
            owners: PageOwnerVec::new(),
            link: None,
        }
    }

    /// Creates a new `PageInfo` representing an MMIO page.
    pub fn new_mmio(dev_type: DeviceMemType) -> Self {
        Self {
            mem_type: MemType::Mmio(dev_type),
            state: PageState::ConvertedLocked,
            owners: PageOwnerVec::new(),
            link: None,
        }
    }

    /// Returns the current owner, if it exists.
    pub fn owner(&self) -> Option<PageOwnerId> {
        use PageState::*;
        match self.state {
            Converting(_) | Unassigning(_) | Converted | ConvertedLocked | Mapped | VmState
            | Shared(_) => {
                if !self.owners.is_empty() {
                    Some(self.owners[self.owners.len() - 1])
                } else {
                    Some(PageOwnerId::hypervisor())
                }
            }
            _ => None,
        }
    }

    /// Returns the page's current state.
    pub fn state(&self) -> PageState {
        self.state
    }

    /// Returns if the page is free.
    pub fn is_free(&self) -> bool {
        matches!(self.state, PageState::Free)
    }

    /// Returns if the page is marked reserved.
    #[allow(dead_code)]
    pub fn is_reserved(&self) -> bool {
        matches!(self.state, PageState::Reserved)
    }

    /// Returns the page type.
    pub fn mem_type(&self) -> MemType {
        self.mem_type
    }

    /// Pops the current owner if there is one, returning the page to the previous owner.
    pub fn release(&mut self) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Mapped | VmState | Converted | Converting(_) | Unassigning(_) => {
                if self.owners.is_empty() {
                    Err(PageTrackingError::OwnerUnderflow) // Can't pop the last owner.
                } else {
                    self.owners.pop().unwrap();
                    self.state = Converted;
                    Ok(())
                }
            }
            ConvertedLocked => Err(PageTrackingError::PageLocked),
            Shared(rc) => {
                // Shared pages start with a RC of 1, so RC is always > 0 here
                if let Some(rc) = rc.checked_sub(1) {
                    self.state = if rc == 0 { Mapped } else { Shared(rc) };
                    // Unwrap ok: Shared pages must have an owner
                    Ok(())
                } else {
                    Err(PageTrackingError::RefCountUnderflow)
                }
            }
            HypState => Err(PageTrackingError::HypervisorStatePage),
            Reserved => Err(PageTrackingError::ReservedPage),
            Free => Err(PageTrackingError::UnownedPage),
        }
    }

    /// Assigns a "ConvertedLocked" page to `owner` with state `new_state`.
    pub fn assign(&mut self, owner: PageOwnerId, new_state: PageState) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Free => {
                if !matches!(new_state, ConvertedLocked | HypState) {
                    // Free pages can either be assigned to 'ConvertedLocked', ready to be
                    // reassigned, or to `HypState` for hypervisor allocation.
                    return Err(PageTrackingError::InvalidStateTransition);
                }
                // We need not be "locked" here since Free is a startup-only state when the hypervisor
                // has exclusive ownership over all memory.
                if owner != PageOwnerId::hypervisor() {
                    self.owners.push(owner);
                }
                self.state = new_state;
                Ok(())
            }
            Converted => Err(PageTrackingError::PageNotLocked),
            ConvertedLocked => {
                if !matches!(new_state, Mapped | VmState) {
                    // Going back to free/reserved or converted isn't allowed, nor does it make
                    // sense for a page to immediately enter the "Converting" or "Unassigning"
                    // state.
                    return Err(PageTrackingError::InvalidStateTransition);
                }
                self.owners
                    .try_push(owner)
                    .map_err(|_| PageTrackingError::OwnerOverflow)?;
                self.state = new_state;
                Ok(())
            }
            _ => Err(PageTrackingError::PageNotAssignable),
        }
    }

    /// Transitions the page to Converting at `tlb_version` if it is currently Mapped.
    pub fn begin_conversion(&mut self, tlb_version: TlbVersion) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Mapped => {
                self.state = Converting(tlb_version);
                Ok(())
            }
            _ => Err(PageTrackingError::PageNotConvertible),
        }
    }

    /// Transitions the page to Converted if it is Converting with a TLB version older than
    /// `tlb_version`. After the page is converted it may be assigned to child VMs.
    pub fn complete_conversion(&mut self, tlb_version: TlbVersion) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Converting(version) => {
                if version < tlb_version {
                    self.state = Converted;
                    Ok(())
                } else {
                    Err(PageTrackingError::PageNotConvertible)
                }
            }
            _ => Err(PageTrackingError::PageNotConvertible),
        }
    }

    /// Returns if the page is Converting and can complete conversion at the given `tlb_version`.
    pub fn is_convertible(&self, tlb_version: TlbVersion) -> bool {
        use PageState::*;
        match self.state {
            Converting(version) => version < tlb_version,
            _ => false,
        }
    }

    /// Transitions the page to Unassigning at `tlb_version` if it is currently Mapped.
    pub fn begin_unassignment(&mut self, tlb_version: TlbVersion) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Mapped => {
                self.state = Unassigning(tlb_version);
                Ok(())
            }
            _ => Err(PageTrackingError::PageNotUnassignable),
        }
    }

    /// Transitions the page to Converted and returns it to the previous owner if it is Unassigning
    /// with a TLB version older than `tlb_version`. The page may then be reassigned to child VMs.
    pub fn complete_unassignment(&mut self, tlb_version: TlbVersion) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Unassigning(version) => {
                if version < tlb_version {
                    if self.owners.is_empty() {
                        Err(PageTrackingError::OwnerUnderflow)
                    } else {
                        self.owners.pop().unwrap();
                        self.state = Converted;
                        Ok(())
                    }
                } else {
                    Err(PageTrackingError::PageNotUnassignable)
                }
            }
            _ => Err(PageTrackingError::PageNotUnassignable),
        }
    }

    /// Returns if the page is Unassigning and can complete unassignment at the given `tlb_version`.
    pub fn is_unassignable(&self, tlb_version: TlbVersion) -> bool {
        use PageState::*;
        match self.state {
            Unassigning(version) => version < tlb_version,
            _ => false,
        }
    }

    /// Returns if the page can be Shared.
    pub fn is_shareable(&self) -> bool {
        use PageState::*;
        match self.state() {
            Shared(rc) if rc == u64::MAX => false,
            Shared(_) | PageState::Mapped => true,
            _ => false,
        }
    }

    /// Returns if the page is Shared
    pub fn is_shared(&self) -> bool {
        use PageState::*;
        matches!(self.state, Shared(_))
    }

    /// Obtains an exclusive reference to a "Converted" page in preparation for assignment or
    /// reclaim.
    pub fn lock_for_assignment(&mut self) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Converted => {
                self.state = ConvertedLocked;
                Ok(())
            }
            ConvertedLocked => Err(PageTrackingError::PageLocked),
            _ => Err(PageTrackingError::PageNotAssignable),
        }
    }

    /// Drops the exclusive lock on a "ConvertedLocked" page.
    pub fn unlock(&mut self) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            ConvertedLocked => {
                self.state = Converted;
                Ok(())
            }
            _ => Err(PageTrackingError::PageNotLocked),
        }
    }

    /// Transitions a page from "Mapped" to "Shared", or increments the reference count if the
    /// page is already shared.
    pub fn share(&mut self) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            Mapped => {
                self.state = Shared(1);
                Ok(())
            }
            Shared(rc) => {
                if let Some(rc) = rc.checked_add(1) {
                    self.state = Shared(rc);
                    Ok(())
                } else {
                    Err(PageTrackingError::RefCountOverflow)
                }
            }
            _ => Err(PageTrackingError::PageNotShareable),
        }
    }

    /// Reclaims the "ConvertedLocked" page as a "Mapped" page for the current owner.
    pub fn reclaim(&mut self) -> PageTrackingResult<()> {
        use PageState::*;
        match self.state {
            ConvertedLocked => {
                self.state = Mapped;
                Ok(())
            }
            // TODO: Reclaim pages that are converting but not yet fully converted?
            _ => Err(PageTrackingError::PageNotReclaimable),
        }
    }

    /// Links this page to the page with the given address. Returns an error if the page is already
    /// linked.
    pub fn link(&mut self, next: SupervisorPageAddr) -> PageTrackingResult<()> {
        if self.link.is_some() {
            Err(PageTrackingError::PageAlreadyLinked)
        } else {
            let addr = NonZeroU64::new(next.bits()).ok_or(PageTrackingError::InvalidPage(next))?;
            self.link = Some(addr);
            Ok(())
        }
    }

    /// Unlinks this page.
    pub fn unlink(&mut self) {
        self.link = None;
    }

    /// Returns the next page in the list, if any.
    pub fn next(&self) -> Option<SupervisorPageAddr> {
        self.link
            .and_then(|n| PageAddr::new(RawAddr::supervisor(n.get())))
    }
}

impl Default for PageInfo {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_SPARSE_MAP_ENTRIES: usize = 16;

/// Maps a contiguous range of memory to a subset of the `PageMap`.
#[derive(Clone, Copy, Debug)]
struct SparseMapEntry {
    base_pfn: usize,
    num_pages: usize,
    page_map_index: usize,
}

/// Keeps information for all physical pages in the system.
pub struct PageMap {
    pages: RawPageVec<PageInfo>,
    sparse_map: ArrayVec<SparseMapEntry, MAX_SPARSE_MAP_ENTRIES>,
}

impl PageMap {
    /// Builds a new `PageMap` from a populated `HwMemMap`. It will track ownership information
    /// for each page in the system.
    pub fn build_from(mem_map: &mut HwMemMap) -> Self {
        // Determine how many pages we'll need for the page map.
        let total_pages = mem_map
            .regions()
            .fold(0, |pages, r| pages + r.size() / PageSize::Size4k as u64);
        let page_map_size =
            PageSize::Size4k.round_up(total_pages * core::mem::size_of::<PageInfo>() as u64);
        let page_map_pages = page_map_size / PageSize::Size4k as u64;

        // Find a space for the page map.
        let page_map_region = mem_map
            .regions()
            .find(|r| r.region_type() == HwMemRegionType::Available && r.size() >= page_map_size)
            .expect("No free space for PageMap");
        let page_map_base = page_map_region.base();

        // Safe to create pages from this memory as `HwMemMap` guarantees that this range is
        // valid and free to use. Safe to unwrap since pages are always 4kB-aligned.
        let seq_pages: SequentialPages<InternalDirty> = unsafe {
            SequentialPages::from_mem_range(page_map_base, PageSize::Size4k, page_map_pages)
                .unwrap()
        };
        let struct_pages = RawPageVec::from(seq_pages.clean());

        // Reserve the memory consumed by the pagemap itself.
        mem_map
            .reserve_region(
                HwReservedMemType::PageMap,
                RawAddr::from(page_map_base),
                page_map_size,
            )
            .expect("Failed to reserve page map");

        let mut page_map = Self::new(struct_pages);
        page_map.populate_from(mem_map);
        page_map
    }

    /// Constructs an empty `PageMap` from an existing vector of `PageInfo` structs.
    fn new(pages: RawPageVec<PageInfo>) -> Self {
        Self {
            pages,
            sparse_map: ArrayVec::new(),
        }
    }

    /// Populates an already-constructed `PageMap` with the memory map information from the given
    /// `HwMemMap`. This `PageMap` must be empty and must have been constructed with enough space
    /// for all the pages in the `HwMemMap`.
    fn populate_from(&mut self, mem_map: &HwMemMap) {
        // Populate the page map with the regions in the memory map.
        //
        // All pages in available RAM regions are initially free and will later become
        // allocated by the hypervisor (and for most pages, further deligated to the host VM).
        //
        // Pages in reserved regions are marked reserved, except for those containing the
        // host VM images, which are considered to be initially hypervisor-owned.
        //
        // MMIO regions are considered to be hyperviosr owned, though they may be further delegated
        // to VMs.
        let mut current_entry = SparseMapEntry {
            base_pfn: mem_map.regions().next().unwrap().base().index(),
            num_pages: 0,
            page_map_index: 0,
        };
        for r in mem_map.regions() {
            let base = r.base();
            if current_entry.base_pfn + current_entry.num_pages != base.index() {
                let next_entry = SparseMapEntry {
                    base_pfn: base.index(),
                    num_pages: 0,
                    page_map_index: current_entry.page_map_index + current_entry.num_pages,
                };
                self.sparse_map.push(current_entry);
                current_entry = next_entry;
            }

            let end = r.end();
            for _ in base.iter_from().take_while(|&a| a != end) {
                match r.region_type() {
                    HwMemRegionType::Available => {
                        self.pages.push(PageInfo::new());
                    }
                    HwMemRegionType::Reserved(HwReservedMemType::HostKernelImage)
                    | HwMemRegionType::Reserved(HwReservedMemType::HostInitramfsImage) => {
                        self.pages.push(PageInfo::new_hypervisor_owned());
                    }
                    HwMemRegionType::Mmio(d) => {
                        self.pages.push(PageInfo::new_mmio(d));
                    }
                    _ => {
                        self.pages.push(PageInfo::new_reserved());
                    }
                }
                current_entry.num_pages += 1;
            }
            // Make sure we won't overflow later.
            assert!(current_entry
                .base_pfn
                .checked_add(current_entry.num_pages)
                .is_some());
        }
        self.sparse_map.push(current_entry);
    }

    /// Returns a reference to the `PageInfo` struct for the 4k page at `addr`.
    pub fn get(&self, addr: SupervisorPageAddr) -> Option<&PageInfo> {
        let index = self.get_map_index(addr)?;
        self.pages.get(index)
    }

    /// Returns a mutable reference to the `PageInfo` struct for the 4k page at `addr`.
    pub fn get_mut(&mut self, addr: SupervisorPageAddr) -> Option<&mut PageInfo> {
        let index = self.get_map_index(addr)?;
        self.pages.get_mut(index)
    }

    /// Returns the number of pages after the page at `addr`
    pub fn num_after(&self, addr: SupervisorPageAddr) -> Option<usize> {
        let index = self.get_map_index(addr)?;
        self.pages.len().checked_sub(index)
    }

    /// Returns an iterator over the `PageInfo`s starting at `addr`. Returns `None` if `addr` is
    /// not in the memory map or is a huge page.
    pub fn iter_from(&self, addr: SupervisorPageAddr) -> Option<PageMapIter> {
        PageMapIter::new(self, addr)
    }

    /// Returns the index in the `PageMap` for the given address.
    fn get_map_index(&self, addr: SupervisorPageAddr) -> Option<usize> {
        self.sparse_map
            .iter()
            .find(|s| s.base_pfn <= addr.index() && addr.index() < s.base_pfn + s.num_pages)
            .map(|entry| entry.page_map_index + addr.index() - entry.base_pfn)
    }
}

/// An iterator over `PageMap` in (PageInfo, address) pairs.
pub struct PageMapIter<'a> {
    page_map: &'a PageMap,
    cur_sparse_entry: usize,
    cur_index: usize,
}

impl<'a> PageMapIter<'a> {
    /// Creates a new iterator from `page_map` starting at `start_addr`.
    pub fn new(page_map: &'a PageMap, start_addr: SupervisorPageAddr) -> Option<Self> {
        let (cur_sparse_entry, entry) = page_map.sparse_map.iter().enumerate().find(|(_, s)| {
            s.base_pfn <= start_addr.index() && start_addr.index() < s.base_pfn + s.num_pages
        })?;
        let cur_index = entry.page_map_index + start_addr.index() - entry.base_pfn;
        Some(Self {
            page_map,
            cur_sparse_entry,
            cur_index,
        })
    }
}

/// A (`PageInfo`, address) pair returned by `PageMapIter`.
pub struct PageInfoWithAddr<'a> {
    pub page: &'a PageInfo,
    pub addr: SupervisorPageAddr,
}

impl<'a> Iterator for PageMapIter<'a> {
    type Item = PageInfoWithAddr<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.page_map.sparse_map.get(self.cur_sparse_entry)?;
        let page = self.page_map.pages.get(self.cur_index).unwrap();
        let pfn = Pfn::supervisor((entry.base_pfn + self.cur_index - entry.page_map_index) as u64);
        let addr = SupervisorPageAddr::from_pfn(pfn, PageSize::Size4k).unwrap();

        self.cur_index += 1;
        if self.cur_index >= entry.num_pages + entry.page_map_index {
            self.cur_sparse_entry += 1;
        }

        Some(Self::Item { page, addr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::HwMemMapBuilder;
    use riscv_pages::{Page, PageAddr, PhysPage, RawAddr, SequentialPages};

    fn stub_page_vec() -> RawPageVec<PageInfo> {
        let backing_mem = vec![0u8; 8192];
        let aligned_pointer = unsafe {
            // Not safe - just a test
            backing_mem
                .as_ptr()
                .add(backing_mem.as_ptr().align_offset(4096))
        };
        let addr = PageAddr::new(RawAddr::supervisor(aligned_pointer as u64)).unwrap();
        let page = unsafe {
            // Test-only: safe because the backing memory is leaked so the memory used for this page
            // will live until the test exits.
            Page::new(addr)
        };
        RawPageVec::from(SequentialPages::from(page))
    }

    #[test]
    fn indexing() {
        let pages = stub_page_vec();
        let num_pages = 10;
        let base_addr = PageAddr::new(RawAddr::supervisor(0x1000_0000)).unwrap();
        let mem_map = unsafe {
            // Not safe - just a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(
                    RawAddr::from(base_addr),
                    num_pages * PageSize::Size4k as u64,
                )
                .unwrap()
                .build()
        };
        let mut pages = PageMap::new(pages);
        pages.populate_from(&mem_map);

        let before_addr = PageAddr::new(RawAddr::supervisor(base_addr.bits() - 4096)).unwrap();
        let last_addr = base_addr.checked_add_pages(num_pages - 1).unwrap();
        let after_addr = last_addr.checked_add_pages(1).unwrap();

        assert!(pages.get(before_addr).is_none());
        assert!(pages.get(base_addr).is_some());
        assert!(pages.get(last_addr).is_some());
        assert!(pages.get(after_addr).is_none());
    }

    #[test]
    fn page_map_building() {
        let pages = stub_page_vec();
        let mut mem_map = unsafe {
            // Not safe - just a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(RawAddr::supervisor(0x1000_0000), 0x2_0000)
                .unwrap()
                .add_mmio_region(
                    DeviceMemType::Imsic,
                    RawAddr::supervisor(0x4000_0000),
                    0x2000,
                )
                .unwrap()
                .build()
        };
        mem_map
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x1000_4000),
                0x1000,
            )
            .unwrap();
        mem_map
            .reserve_region(
                HwReservedMemType::HostKernelImage,
                RawAddr::supervisor(0x1001_0000),
                0x2000,
            )
            .unwrap();
        let mut pages = PageMap::new(pages);
        pages.populate_from(&mem_map);

        let free_addr = PageAddr::new(RawAddr::supervisor(0x1000_1000)).unwrap();
        let reserved_addr = PageAddr::new(RawAddr::supervisor(0x1000_4000)).unwrap();
        let used_addr = PageAddr::new(RawAddr::supervisor(0x1001_1000)).unwrap();
        let mmio_addr = PageAddr::new(RawAddr::supervisor(0x4000_1000)).unwrap();

        let free_page = pages.get(free_addr).unwrap();
        assert!(free_page.is_free());
        assert_eq!(free_page.mem_type(), MemType::Ram);
        assert!(pages.get(reserved_addr).unwrap().is_reserved());
        assert!(pages.get(used_addr).unwrap().owner().is_some());
        let mmio_page = pages.get(mmio_addr).unwrap();
        assert!(mmio_page.owner().is_some());
        assert_eq!(mmio_page.mem_type(), MemType::Mmio(DeviceMemType::Imsic));
    }

    #[test]
    fn sparse_map() {
        let pages = stub_page_vec();
        const TOTAL_SIZE: u64 = 0x4_0000;
        let mem_map = unsafe {
            // Not safe - just a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(RawAddr::supervisor(0x1000_0000), TOTAL_SIZE / 2)
                .unwrap()
                .add_memory_region(RawAddr::supervisor(0x2000_0000), TOTAL_SIZE / 2)
                .unwrap()
                .build()
        };
        let mut pages = PageMap::new(pages);
        pages.populate_from(&mem_map);

        let base_addr = PageAddr::new(RawAddr::supervisor(0x1000_0000)).unwrap();
        let r0_addr = PageAddr::new(RawAddr::supervisor(0x1000_8000)).unwrap();
        let r1_addr = PageAddr::new(RawAddr::supervisor(0x2000_3000)).unwrap();

        assert!(pages.get(base_addr).unwrap().is_free());
        assert!(pages.get(r0_addr).unwrap().is_free());
        assert!(pages.get(r1_addr).unwrap().is_free());
        assert_eq!(
            pages.num_after(base_addr).unwrap(),
            (TOTAL_SIZE / PageSize::Size4k as u64) as usize
        );
    }

    #[test]
    fn page_ownership() {
        let mut page = PageInfo::new();
        assert!(page.is_free());
        assert!(page
            .assign(PageOwnerId::hypervisor(), PageState::ConvertedLocked)
            .is_ok());
        assert!(page.assign(PageOwnerId::host(), PageState::Mapped).is_ok());
        assert_eq!(page.owner().unwrap(), PageOwnerId::host());
        let version = TlbVersion::new();
        assert!(page.begin_conversion(version).is_ok());
        assert_eq!(page.state(), PageState::Converting(version));
        assert!(!page.is_convertible(version));
        let version = version.increment();
        assert!(page.complete_conversion(version).is_ok());
        assert_eq!(page.state(), PageState::Converted);
        assert!(page.lock_for_assignment().is_ok());
        let guest_id = PageOwnerId::new(2).unwrap();
        assert!(page.assign(guest_id, PageState::Mapped).is_ok());
        let guest_version = TlbVersion::new();
        assert!(page.begin_unassignment(guest_version).is_ok());
        assert_eq!(page.state(), PageState::Unassigning(guest_version));
        let guest_version = version.increment();
        assert!(page.is_unassignable(guest_version));
        assert!(page.complete_unassignment(guest_version).is_ok());
        assert_eq!(page.state(), PageState::Converted);
        assert!(page.lock_for_assignment().is_ok());
        assert!(page.reclaim().is_ok());

        let mut page = PageInfo::new_reserved();
        assert!(!page.is_free());
        assert!(page
            .assign(PageOwnerId::hypervisor(), PageState::ConvertedLocked)
            .is_err());
    }
}
