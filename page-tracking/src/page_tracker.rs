// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// TODO - move to a riscv-specific mutex implementation when ready.
use riscv_pages::*;
use spin::Mutex;

use crate::collections::{RawPageVec, StaticPageRef};
use crate::page_info::{PageInfo, PageMap, PageState};
use crate::{HwMemMap, PageList, TlbVersion};

/// Errors related to managing physical page information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Too many guests started by the host at once.
    GuestOverflow,
    /// Too many guests per system(u64 overflow).
    IdOverflow,
    /// The given page isn't physically present.
    InvalidPage(SupervisorPageAddr),
    /// The ownership chain is too long to add another owner.
    OwnerOverflow,
    /// The page would become unowned as a result of popping its current owner.
    OwnerUnderflow,
    /// Attempt to pop the owner of an unowned page.
    UnownedPage,
    /// Attempt to modify the owner of a reserved page.
    ReservedPage,
    /// The page is not owned by the specified owner.
    OwnerMismatch,
    /// Attempt to modify the owner of a shared page.
    /// At present, all shared pages are owned by the host
    SharedPage,
    /// The page is not in a state where it can be converted.
    PageNotConvertible,
    /// The page is not in a state where it can be assigned.
    PageNotAssignable,
    /// The page cannot be mapped back into the owner's address space.
    PageNotReclaimable,
    /// The page cannot be shared.
    PageNotShareable,
    /// The page isn't in Shared state.
    PageNotShared,
    /// Attempt to intiate an invalid page state transition.
    InvalidStateTransition,
    /// Attempt to assign or reclaim a page that is not locked.
    PageNotLocked,
    /// Attempt to lock a page for assignment that is already locked.
    PageLocked,
    /// Attempt to create a link from a page that is already linked.
    PageAlreadyLinked,
    /// The ref count was at u64::MAX.
    RefCountOverflow,
    /// The ref count was already 0.
    RefCountUnderflow,
}

/// Holds the result of page tracking operations.
pub type Result<T> = core::result::Result<T, Error>;

// Inner struct that is wrapped in a mutex by `PageTracker`.
struct PageTrackerInner {
    next_owner_id: u64,
    active_guests: RawPageVec<PageOwnerId>,
    pages: PageMap,
}

impl PageTrackerInner {
    fn get_mut(&mut self, addr: SupervisorPageAddr) -> Result<&mut PageInfo> {
        self.pages.get_mut(addr).ok_or(Error::InvalidPage(addr))
    }

    fn get(&mut self, addr: SupervisorPageAddr) -> Result<&PageInfo> {
        self.pages.get(addr).ok_or(Error::InvalidPage(addr))
    }
}

/// This struct wraps the list of all memory pages and active guests. It can be cloned and passed to
/// other compontents that need access to page state. Once created, there is no way to free the
/// backing page list. That page list is needed for the lifetime of the system.
#[derive(Clone)]
pub struct PageTracker {
    inner: StaticPageRef<Mutex<PageTrackerInner>>,
}

impl PageTracker {
    /// Creates a new PageTracker representing all pages in the system and returns all pages that are
    /// available for the primary host to use, starting at the next `host_alignment`-aligned chunk.
    pub fn from(
        mut hyp_mem: HypPageAlloc,
        host_alignment: u64,
    ) -> (Self, PageList<Page<ConvertedClean>>) {
        // TODO - hard coded to two pages worth of guests. - really dumb if page size is 1G
        let mut active_guests = RawPageVec::from(hyp_mem.take_pages_for_host_state(2));
        active_guests.push(PageOwnerId::host());

        let state_storage_page = hyp_mem
            .take_pages_for_host_state(1)
            .into_iter()
            .next()
            .unwrap();

        // Discard a host_alignment sized chunk to align ourselves.
        let _ = hyp_mem.take_pages(
            (host_alignment / PageSize::Size4k as u64)
                .try_into()
                .unwrap(),
            host_alignment,
        );

        let (page_map, head_addr) = hyp_mem.drain();

        let inner = StaticPageRef::new_with(
            Mutex::new(PageTrackerInner {
                // Start at two for owners as host and hypervisor reserve 0 and 1.
                next_owner_id: 2,
                active_guests,
                pages: page_map,
            }),
            state_storage_page,
        );
        let page_tracker = Self { inner };

        let host_pages = unsafe {
            // Safe since we trust that HypPageAlloc::drain() properly created a linked-list starting
            // at head_addr and that all the pages in the list were cleaned.
            PageList::from_raw_parts(page_tracker.clone(), head_addr)
        };

        (page_tracker, host_pages)
    }

    /// Creates a `PageTracker` and a list of the pages it contains for use in test environments.
    #[cfg(test)]
    pub(crate) fn new_in_test() -> (Self, PageList<Page<ConvertedClean>>) {
        use crate::HwMemMapBuilder;

        const ONE_MEG: usize = 1024 * 1024;
        const MEM_ALIGN: usize = 2 * ONE_MEG;
        const MEM_SIZE: usize = 256 * ONE_MEG;
        let backing_mem = vec![0u8; MEM_SIZE + MEM_ALIGN];
        let aligned_pointer = unsafe {
            // Not safe - just a test
            backing_mem
                .as_ptr()
                .add(backing_mem.as_ptr().align_offset(MEM_ALIGN))
        };
        let start_pa = RawAddr::supervisor(aligned_pointer as u64);
        let hw_map = unsafe {
            // Not safe - just a test
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(start_pa, MEM_SIZE.try_into().unwrap())
                .unwrap()
                .build()
        };
        let hyp_mem = HypPageAlloc::new(hw_map);
        let (page_tracker, host_pages) = PageTracker::from(hyp_mem, PageSize::Size4k as u64);
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        (page_tracker, host_pages)
    }

    /// Adds a new guest to the system, giving it the next ID.
    pub fn add_active_guest(&self) -> Result<PageOwnerId> {
        let mut page_tracker = self.inner.lock();
        // unwrap is fine as next_owner_id is guaranteed to be valid.
        let id = PageOwnerId::new(page_tracker.next_owner_id).unwrap();
        // TODO handle very rare roll over cleaner.
        page_tracker.next_owner_id = page_tracker
            .next_owner_id
            .checked_add(1)
            .ok_or(Error::IdOverflow)?;

        page_tracker
            .active_guests
            .try_reserve(1)
            .map_err(|_| Error::GuestOverflow)?;
        page_tracker.active_guests.push(id);
        Ok(id)
    }

    /// Removes an active guest previously added by `add_active_guest`.
    pub fn rm_active_guest(&self, remove_id: PageOwnerId) {
        let mut page_tracker = self.inner.lock();
        page_tracker.active_guests.retain(|&id| id != remove_id);
    }

    /// Assigns `page` as a mapped page for `owner`, returning a page that can then be mapped into
    /// a page table.
    pub fn assign_page_for_mapping<P, M>(
        &self,
        page: P,
        owner: PageOwnerId,
    ) -> Result<P::MappablePage>
    where
        P: AssignablePhysPage<M>,
        M: MeasureRequirement,
    {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr()).unwrap();
        info.assign(owner, PageState::Mapped)?;
        // Safe since we own the page and have updated its state.
        Ok(unsafe { P::MappablePage::new_with_size(page.addr(), page.size()) })
    }

    /// Consumes the shared `page`, returning a page that can then be mapped into
    /// a page table.
    pub fn share_page<P>(&self, page: P, owner: PageOwnerId) -> Result<P::MappablePage>
    where
        P: ShareablePhysPage,
    {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr()).unwrap();
        if info.owner() != Some(owner) || info.mem_type() != P::mem_type() || !info.is_shareable() {
            return Err(Error::PageNotShareable);
        }
        info.share()?;
        // Safe since we own the page and have updated its state.
        Ok(unsafe { P::MappablePage::new_with_size(page.addr(), page.size()) })
    }

    /// Assigns `page` as an internal state page for `owner`, returning a page that is eligible to
    /// be used with various internal collection types (e.g. `PageBox<>`).
    pub fn assign_page_for_internal_state(
        &self,
        page: Page<ConvertedClean>,
        owner: PageOwnerId,
    ) -> Result<Page<InternalClean>> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr()).unwrap();
        info.assign(owner, PageState::VmState)?;
        // Safe since we own the page and have updated its state.
        Ok(unsafe { Page::new_with_size(page.addr(), page.size()) })
    }

    /// Relases `page` back to its previous owner.
    pub fn release_page<P: PhysPage>(&self, page: P) -> Result<()> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr()).unwrap();
        info.release()?;
        Ok(())
    }

    /// Releases the page at `addr` back to its previous owner if it's currently owned by `owner`
    /// and is in a releasable state.
    pub fn release_page_by_addr(&self, addr: SupervisorPageAddr, owner: PageOwnerId) -> Result<()> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(addr)?;
        // Shared pages might be owned by the parent
        if info.owner() != Some(owner) && !info.is_shared() {
            return Err(Error::OwnerMismatch);
        }
        info.release()?;
        Ok(())
    }

    /// Marks the invalidated page as having started conversion at `tlb_version`.
    pub fn convert_page<P: InvalidatedPhysPage>(
        &self,
        page: P,
        tlb_version: TlbVersion,
    ) -> Result<()> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr()).unwrap();
        info.begin_conversion(tlb_version)
    }

    /// Reclaims the converted, but unassigned, `page` back to a mapped page for the current owner.
    /// Returns a page that can then be mapped in a page table.
    pub fn reclaim_page<P: ReclaimablePhysPage>(&self, page: P) -> Result<P::MappablePage> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr()).unwrap();
        info.reclaim()?;
        // Safe since we own the page and have verified that it can be reclaimed.
        Ok(unsafe { P::MappablePage::new_with_size(page.addr(), page.size()) })
    }

    /// Acquires an exclusive reference to the Converted page at `addr` if it's unassigned and owned
    /// by `owner`. Completes conversion if the page was Converting at a TLB version older than
    /// `tlb_version`.
    pub fn get_converted_page<P: ConvertedPhysPage>(
        &self,
        addr: SupervisorPageAddr,
        owner: PageOwnerId,
        tlb_version: TlbVersion,
    ) -> Result<P::DirtyPage> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(addr)?;
        if info.owner() != Some(owner)
            || info.mem_type() != P::mem_type()
            || (info.state() != PageState::Converted
                && info.complete_conversion(tlb_version).is_err())
        {
            return Err(Error::PageNotConvertible);
        }
        info.lock_for_assignment()?;
        // Safe since we've taken exclusive ownership of the page, verified its typing, and that it is
        // converted as of `tlb_version`.
        //
        // TODO: Page size
        Ok(unsafe { P::DirtyPage::new(addr) })
    }

    /// Releases an exclusive reference to a locked page
    pub fn unlock_page<P: PhysPage>(&self, page: P) -> Result<()> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(page.addr())?;
        info.unlock()
    }

    /// Returns true if and only if `addr` is a page owned by `owner`.
    pub fn is_owned(&self, addr: SupervisorPageAddr, owner: PageOwnerId) -> bool {
        let mut page_tracker = self.inner.lock();
        if let Ok(info) = page_tracker.get(addr) {
            info.owner() == Some(owner)
        } else {
            false
        }
    }

    /// Returns true if and only if `addr` is a "Mapped" page owned by `owner` with type `mem_type`.
    pub fn is_mapped_page(
        &self,
        addr: SupervisorPageAddr,
        owner: PageOwnerId,
        mem_type: MemType,
    ) -> bool {
        let mut page_tracker = self.inner.lock();
        if let Ok(info) = page_tracker.get(addr) {
            info.owner() == Some(owner)
                && info.mem_type() == mem_type
                && info.state() == PageState::Mapped
        } else {
            false
        }
    }

    /// Acquires an exclusive reference to the shareable page at `addr` if it's owned by owner,
    /// and in Mapped or Shared state.
    pub fn get_shareable_page<P: ShareablePhysPage>(
        &self,
        addr: SupervisorPageAddr,
        owner: PageOwnerId,
    ) -> Result<P> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(addr)?;
        if info.owner() != Some(owner) || info.mem_type() != P::mem_type() || !info.is_shareable() {
            return Err(Error::PageNotShareable);
        }

        info.lock_for_assignment()?;
        // Safe since we've taken exclusive ownership of the page, verified its typing.
        // TODO: Page size
        Ok(unsafe { P::new(addr) })
    }

    /// Returns true if and only if `addr` is a page owned by `owner` with type `mem_type` and
    /// was converted at a TLB version older than `tlb_version`.
    pub fn is_converted_page(
        &self,
        addr: SupervisorPageAddr,
        owner: PageOwnerId,
        mem_type: MemType,
        tlb_version: TlbVersion,
    ) -> bool {
        let mut page_tracker = self.inner.lock();
        if let Ok(info) = page_tracker.get(addr) {
            info.owner() == Some(owner)
                && info.mem_type() == mem_type
                && (info.state() == PageState::Converted || info.is_convertible(tlb_version))
        } else {
            false
        }
    }

    /// Returns true if and only if `addr` is a `VmState` page owned by `owner`.
    pub fn is_internal_state_page(&self, addr: SupervisorPageAddr, owner: PageOwnerId) -> bool {
        let mut page_tracker = self.inner.lock();
        if let Ok(info) = page_tracker.get(addr) {
            info.owner() == Some(owner)
                && info.mem_type() == MemType::Ram
                && info.state() == PageState::VmState
        } else {
            false
        }
    }

    /// Creates a link from page `a` to `b` if neither is already linked.
    pub(crate) fn link_pages(&self, a: SupervisorPageAddr, b: SupervisorPageAddr) -> Result<()> {
        let mut page_tracker = self.inner.lock();
        // We don't need to touch the destination, but we need to make sure it isn't linked.
        let dst = page_tracker.get(b)?;
        if dst.next().is_some() {
            return Err(Error::PageAlreadyLinked);
        }
        let src = page_tracker.get_mut(a)?;
        src.link(b)
    }

    /// Unlinks the page at `addr`, returning the address of the page to which it was pointing.
    pub(crate) fn unlink_page(&self, addr: SupervisorPageAddr) -> Option<SupervisorPageAddr> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(addr).ok()?;
        let next = info.next();
        info.unlink();
        next
    }

    /// Returns the address of the page linked to the page at `addr`, if any.
    pub(crate) fn linked_page(&self, addr: SupervisorPageAddr) -> Option<SupervisorPageAddr> {
        let mut page_tracker = self.inner.lock();
        let info = page_tracker.get_mut(addr).ok()?;
        info.next()
    }
}

/// `HypPageAlloc` is created from the hardware memory map and builds the array of PageInfo
/// structs for all pages in the system. It is used to allocate pages for the hypervisor at
/// startup for building the host VM and other local data. Once the hypervisor has taken the
/// pages it needs, `HypPageAlloc` should be converted to the list of remaining free memory
/// regions to be mapped into the host with `drain()`.
#[derive(Debug)]
pub struct HypPageAlloc {
    next_page: Option<SupervisorPageAddr>,
    pages: PageMap,
}

impl HypPageAlloc {
    /// Creates a new `HypPageAlloc`. The memory map passed in contains information about what
    /// physical memory can be used by the machine.
    pub fn new(mem_map: HwMemMap) -> Self {
        let first_page = mem_map.regions().next().unwrap().base();
        let mut hyp_pages = Self {
            next_page: None,
            pages: PageMap::build_from(mem_map),
        };
        hyp_pages.next_page = hyp_pages.next_free_page(first_page);
        hyp_pages
    }

    /// Takes ownership of the remaining free pages, cleaning them and linking them together. Returns
    /// the global `PageMap` structure and the head of the free page list.
    fn drain(mut self) -> (PageMap, SupervisorPageAddr) {
        let head = self.next_page;
        let mut tail: Option<SupervisorPageAddr> = None;
        while let Some(next) = self.next_page {
            let info = self.pages.get_mut(next).unwrap();
            info.assign(PageOwnerId::hypervisor(), PageState::Converted)
                .unwrap();
            info.lock_for_assignment().unwrap();
            // Safe to create this page as it was previously free and we just took ownership.
            let page: Page<ConvertedDirty> = unsafe { Page::new(next) };
            page.clean();

            // Link ourselves to the previous page.
            if let Some(tail_addr) = tail {
                let tail_info = self.pages.get_mut(tail_addr).unwrap();
                tail_info.link(next).unwrap();
            }
            tail = Some(next);

            // Skip until the next free page or we reach the end of memory.
            self.next_page = self.next_free_page(next);
        }

        (self.pages, head.unwrap())
    }

    /// Returns the number of pages remaining in the system. Note that this may include reserved
    /// pages.
    pub fn pages_remaining(&self) -> u64 {
        if let Some(addr) = self.next_page {
            // Ok to unwrap because next page must be in range.
            self.pages.num_after(addr).unwrap() as u64
        } else {
            0
        }
    }

    /// Takes and cleans `count` contiguous Pages with the requested alignment from the system map.
    /// Sets the hypervisor as the owner of the pages, and any pages consumed up until that point,
    /// in the system page map. Allows passing ranges of pages around without a mutable
    /// reference to the global owners list. Panics if there are not `count` pages available. The
    /// returned pages are eligible to be mapped into the host's address space.
    pub fn take_pages(&mut self, count: usize, align: u64) -> SequentialPages<ConvertedClean> {
        // Helper to test whether a contiguous range of `count` pages is free and aligned.
        let range_is_free_and_aligned = |start: SupervisorPageAddr| {
            let end = start.checked_add_pages(count as u64).unwrap();
            if start.bits() & (align - 1) != 0 {
                return false;
            }
            start
                .iter_from()
                .take_while(|&a| a != end)
                .all(|a| self.pages.get(a).map_or(false, |p| p.is_free()))
        };

        // Find the free page rage and mark it, and any free pages we skipped in between,
        // as hypervisor-owned.
        let start_page = self.next_page.unwrap();
        let first_page = self
            .pages
            .iter_from(start_page)
            .and_then(|mut i| i.find(|p| range_is_free_and_aligned(p.addr)))
            .map(|p| p.addr)
            .unwrap();
        let last_page = first_page.checked_add_pages(count as u64).unwrap();
        for page in start_page.iter_from().take_while(|&a| a != last_page) {
            if let Some(page_info) = self.pages.get_mut(page) {
                if page_info.is_free() {
                    // OK to unwrap as this struct is new and must have space for one owner.
                    // Free pages can be assigned without locking
                    page_info
                        .assign(PageOwnerId::hypervisor(), PageState::Converted)
                        .unwrap();
                    page_info.lock_for_assignment().unwrap();
                }
            }
        }

        // Move self's next page past these taken pages.
        self.next_page = self.next_free_page(last_page);

        let dirty_pages: SequentialPages<ConvertedDirty> = unsafe {
            // It's safe to create a page range of the memory that `self` forfeited ownership of
            // above and the new `SequentialPages` is now the unique owner. Ok to unwrap here simce
            // all pages are trivially aligned to 4kB.
            SequentialPages::from_page_range(first_page, last_page, PageSize::Size4k).unwrap()
        };
        dirty_pages.clean()
    }

    /// Same as above, but the returned pages are assigned for internal use.
    pub fn take_pages_for_host_state_with_alignment(
        &mut self,
        count: usize,
        align: u64,
    ) -> SequentialPages<InternalClean> {
        let assignable_pages = self.take_pages(count, align);
        SequentialPages::from_pages(assignable_pages.into_iter().map(|p| {
            let page_info = self.pages.get_mut(p.addr()).unwrap();
            page_info
                .assign(PageOwnerId::host(), PageState::VmState)
                .unwrap();
            // Safety: We uniquely own this memory and we've updated its state in page_info.
            unsafe { Page::<InternalClean>::new(p.addr()) }
        }))
        .unwrap()
    }

    /// Same as above, but with no alignment requirement.
    pub fn take_pages_for_host_state(&mut self, count: usize) -> SequentialPages<InternalClean> {
        self.take_pages_for_host_state_with_alignment(count, PageSize::Size4k as u64)
    }

    /// Returns the address of the next free page at or after `addr`, or `None` if there are no free
    /// pages left.
    fn next_free_page(&self, addr: SupervisorPageAddr) -> Option<SupervisorPageAddr> {
        self.pages
            .iter_from(addr)
            .and_then(|mut i| i.find(|p| p.page.is_free()))
            .map(|p| p.addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HwMemMapBuilder;
    use riscv_pages::RawAddr;

    fn stub_hyp_mem() -> HypPageAlloc {
        const ONE_MEG: usize = 1024 * 1024;
        const MEM_ALIGN: usize = 2 * ONE_MEG;
        const MEM_SIZE: usize = 256 * ONE_MEG;
        let backing_mem = vec![0u8; MEM_SIZE + MEM_ALIGN];
        let aligned_pointer = unsafe {
            // Not safe - just a test
            backing_mem
                .as_ptr()
                .add(backing_mem.as_ptr().align_offset(MEM_ALIGN))
        };
        let start_pa = RawAddr::supervisor(aligned_pointer as u64);
        let hw_map = unsafe {
            // Not safe - just a test
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(start_pa, MEM_SIZE.try_into().unwrap())
                .unwrap()
                .build()
        };
        let hyp_mem = HypPageAlloc::new(hw_map);
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        hyp_mem
    }

    fn stub_page_tracker() -> (PageTracker, PageList<Page<ConvertedClean>>) {
        let hyp_mem = stub_hyp_mem();
        PageTracker::from(hyp_mem, PageSize::Size4k as u64)
    }

    #[test]
    fn hyp_mem_take_pages() {
        let mut hyp_mem = stub_hyp_mem();
        let first = hyp_mem
            .take_pages(1, PageSize::Size4k as u64)
            .into_iter()
            .next()
            .unwrap();
        let mut taken = hyp_mem.take_pages(2, PageSize::Size4k as u64).into_iter();
        let after_taken = hyp_mem
            .take_pages(1, PageSize::Size4k as u64)
            .into_iter()
            .next()
            .unwrap();

        assert_eq!(
            after_taken.addr().bits(),
            first.addr().bits() + (PageSize::Size4k as u64 * 3)
        );
        assert_eq!(
            taken.next().unwrap().addr().bits(),
            first.addr().bits() + PageSize::Size4k as u64
        );
        assert_eq!(
            taken.next().unwrap().addr().bits(),
            first.addr().bits() + (PageSize::Size4k as u64 * 2)
        );
    }

    #[test]
    fn hyp_mem_take_aligned() {
        let mut hyp_mem = stub_hyp_mem();
        let range = hyp_mem.take_pages(4, 16 * 1024);
        assert_eq!(range.base().bits() & (16 * 1024 - 1), 0);
    }

    #[test]
    fn hyp_mem_drain() {
        let hyp_mem = stub_hyp_mem();
        let remaining = hyp_mem.pages_remaining();
        let (_, host_pages) = PageTracker::from(hyp_mem, PageSize::Size4k as u64);
        assert!(host_pages.len() > 0);
        assert!((host_pages.len() as u64) < remaining);
    }

    #[test]
    fn drop_one_page_tracker_ref() {
        let (page_tracker, _host_mem) = stub_page_tracker();
        let new_id = {
            let c = page_tracker.clone();
            c.add_active_guest().unwrap()
        };
        assert_eq!(page_tracker.inner.lock().active_guests.len(), 2);

        page_tracker.rm_active_guest(new_id);

        assert_eq!(page_tracker.inner.lock().active_guests.len(), 1);
    }
}
