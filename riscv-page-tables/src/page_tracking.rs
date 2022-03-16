// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// TODO - move to a riscv-specific mutex implementation when ready.
use spin::Mutex;

use page_collections::page_box::PageBox;
use page_collections::page_vec::PageVec;
use riscv_pages::{
    Page, PageAddr, PageAddr4k, PageOwnerId, PageSize, PageSize4k, PhysAddr, SequentialPages,
};

use crate::page_info::{PageInfo, Pages};
use crate::{HwMemMap, PageRange};

/// Errors related to managing physical page information.
#[derive(Debug)]
pub enum Error {
    /// Too many guests started by the host at once.
    GuestOverflow,
    /// Too many guests per system(u64 overflow).
    IdOverflow,
    /// The given page isn't physically present.
    InvalidPage(PageAddr4k),
    /// Reserved RAM amount given isn't aligned to a page boundary or overflows end of ram.
    InvalidReservedSize(u64),
    /// The ownership chain is too long to add another owner.
    OwnerTooDeep,
    /// Ram size and base don't fit in the address space.
    RamRangeInvalid,
    /// Address given for memory start isn't aligned to a page boundary.
    UnalignedMemory(u64),
}

pub type Result<T> = core::result::Result<T, Error>;

// Inner struct that is wrapped in a mutex by `PageState`.
struct PageStateInner {
    next_owner_id: u64,
    active_guests: PageVec<PageOwnerId>,
    pages: Pages,
}

impl PageStateInner {
    // pops any owners that have exited.
    // Remove owners of the page that have since terminated. This is done lazily as needed to
    // prevent a long running operation on guest exit.
    fn pop_exited_owners(&mut self, addr: PageAddr4k) {
        if let Some(info) = self.pages.get_mut(addr) {
            info.pop_owners_while(|id| !self.active_guests.contains(id));
        }
    }

    // Pop the current owner returning the page to the previous owner. Returns the removed owner ID.
    fn pop_owner_internal(&mut self, addr: PageAddr4k) -> PageOwnerId {
        let page_info = self.pages.get_mut(addr).unwrap();

        page_info.pop_owner()
    }

    // Sets the owner of the page at `addr` to `owner`
    fn set_page_owner(&mut self, addr: PageAddr4k, owner: PageOwnerId) -> Result<()> {
        self.pop_exited_owners(addr);

        let page_info = self.pages.get_mut(addr).ok_or(Error::InvalidPage(addr))?;
        page_info.push_owner(owner)
    }

    // Returns the current owner of the the page ad `addr`.
    fn owner(&self, addr: PageAddr4k) -> PageOwnerId {
        if let Some(info) = self.pages.get(addr) {
            info.find_owner(|id| self.active_guests.contains(id))
        } else {
            // Default owner of all pages is the host.
            PageOwnerId::host()
        }
    }
}

/// This struct wraps the list of all memory pages and active guests. It can be cloned and passed to
/// other compontents that need access to page state. Once created, there is no way to free the
/// backing page list. That page list is needed for the lifetime of the system.
#[derive(Clone)]
pub struct PageState {
    inner: &'static Mutex<PageStateInner>,
}

impl PageState {
    /// Creates a new PageState representing all pages in the system and returns all pages that are
    /// available for the primary host to use. The pages for the host will begin at the next 2MB
    /// boundary.
    pub fn from(mut hyp_mem: HypMemoryPages) -> (Self, PageRange) {
        let active_guests = Self::make_active_guest_vec(&mut hyp_mem);

        let state_storage_page = hyp_mem.next_page();

        // 2MB align the host's pages
        hyp_mem.discard_to_align(2 * 1024 * 1024);

        let (host_pages, pages) = hyp_mem.split_host_range();
        let mutex_box = PageBox::new_with(
            Mutex::new(PageStateInner {
                // Start at two for owners as host and hypervisor reserve 0 and 1.
                next_owner_id: 2,
                active_guests,
                pages,
            }),
            state_storage_page,
        );

        (
            Self {
                inner: PageBox::leak(mutex_box),
            },
            host_pages,
        )
    }

    // Creates the PageVec for holding all active guests
    fn make_active_guest_vec(hyp_mem: &mut HypMemoryPages) -> PageVec<PageOwnerId> {
        // TODO - hard coded to two pages worth of guests. - really dumb if page size is 1G
        let pages = [hyp_mem.next_page(), hyp_mem.next_page()];
        let seq_pages = match SequentialPages::from_pages(pages) {
            Err(_) => unreachable!(),
            Ok(sp) => sp,
        };
        PageVec::from(seq_pages)
    }

    /// Adds a new guest to the system, giving it the next ID.
    pub fn add_active_guest(&mut self) -> Result<PageOwnerId> {
        let mut phys_pages = self.inner.lock();
        // unwrap is fine as next_owner_id is guaranteed to be valid.
        let id = PageOwnerId::new(phys_pages.next_owner_id).unwrap();
        // TODO handle very rare roll over cleaner.
        phys_pages.next_owner_id = phys_pages
            .next_owner_id
            .checked_add(1)
            .ok_or(Error::IdOverflow)?;

        phys_pages
            .active_guests
            .try_reserve(1)
            .map_err(|_| Error::GuestOverflow)?;
        phys_pages.active_guests.push(id);
        Ok(id)
    }

    /// Removes an active guest previously added by `add_active_guest`.
    pub fn rm_active_guest(&mut self, remove_id: PageOwnerId) {
        let mut phys_pages = self.inner.lock();
        phys_pages.active_guests.retain(|&id| id != remove_id);
    }

    /// Sets the owner of the page at the given `addr` to `owner`.
    pub fn set_page_owner(&mut self, addr: PageAddr4k, owner: PageOwnerId) -> Result<()> {
        let mut phys_pages = self.inner.lock();
        phys_pages.set_page_owner(addr, owner)
    }

    /// Removes the current owner of the page at `addr` and returns it.
    pub fn pop_owner(&mut self, addr: PageAddr4k) -> PageOwnerId {
        let mut phys_pages = self.inner.lock();
        phys_pages.pop_owner_internal(addr)
    }

    /// Returns the current owner of the page. Pages without an owner set are owned by the host.
    pub fn owner(&self, addr: PageAddr4k) -> PageOwnerId {
        let phys_pages = self.inner.lock();
        phys_pages.owner(addr)
    }
}

/// `HypMemoryPages` is created with reserved memory and memory for all struct pages assigned. It is
/// used to allocate pages for the hypervisor to use for other local data.
/// Once the hypervisor has taken the pages it needs, `HypMemoryPages` should be converted to
/// `PageRange` for the host to allocate from.
pub struct HypMemoryPages {
    next_page: PageAddr<PageSize4k>,
    pages: Pages,
}

impl HypMemoryPages {
    /// Creates a new `HypMemoryPages`. The memory map passed in contains information about what
    /// physical memory can be used by the machine.
    pub fn new(mmap: HwMemMap) -> Self {
        let structs_per_page = PageSize4k::SIZE_BYTES / core::mem::size_of::<PageInfo>() as u64;
        let total_pages = mmap.ram_size() / PageSize4k::SIZE_BYTES;
        let pages_for_structs = total_pages / structs_per_page;
        let base_page_index = mmap.ram_base().index() as usize;

        // Safe to create pages from this memory as `HwMemMap` guarantees all ranges are valid and
        // free to use.
        let seq_pages = unsafe {
            SequentialPages::<PageSize4k>::from_mem_range(mmap.usable_ram_base(), pages_for_structs)
        };

        // track the next available page for hypervisor use.
        let first_avail_page: PageAddr<PageSize4k> = PageAddr::new(PhysAddr::new(
            mmap.usable_ram_base().bits() + pages_for_structs * PageSize4k::SIZE_BYTES,
        ))
        .unwrap();

        let mut struct_pages = PageVec::from(seq_pages);

        // Mark all pages used as owned by the host, the hypervisor will steal some before starting the
        // host.
        for _i in 0..total_pages {
            struct_pages.push(PageInfo::new_host_owned());
        }

        // Mark all reserved memory as used by hypervisor and inaccessible from VMs.
        for page_addr in mmap
            .ram_base()
            .iter_from()
            .take_while(|&a| a != first_avail_page)
        {
            // OK to unwrap as this struct is new and must have space for one owner.
            struct_pages[page_addr.index() - base_page_index]
                .push_owner(PageOwnerId::hypervisor())
                .unwrap();
        }

        Self {
            next_page: first_avail_page,
            pages: Pages::new(struct_pages, base_page_index),
        }
    }

    /// Takes the rest of the pages contained in `self` and converts them to a `PageRange` with all
    /// the page's owners set to the host. It also returns the global page info structs as `Pages`.
    pub fn split_host_range(self) -> (PageRange, Pages) {
        let pages_remaining = self.pages_remaining();
        let range = unsafe {
            // Safe to create a range of pages as the range was previously owned by `self` and
            // that memory ownership is being moved to the new `PageRange`.
            PageRange::new(
                self.next_page,
                self.next_page.checked_add_pages(pages_remaining).unwrap(),
            )
        };
        (range, self.pages)
    }

    /// Returns the number of pages remaining in the system.
    pub fn pages_remaining(&self) -> u64 {
        // Ok to unwrap because next page must be in range.
        self.pages.num_after(self.next_page).unwrap() as u64
    }

    /// Returns the next 4k page for the hypervisor to use.
    /// Asserts if out of memory. If there aren't enough pages to set up hypervisor state, there is
    /// no point in continuing.
    fn next_page(&mut self) -> Page<PageSize4k> {
        // OK to unwrap as next_page is guaranteed to be in range.
        match self
            .pages
            .get_mut(self.next_page)
            .unwrap()
            .push_owner(PageOwnerId::hypervisor())
        {
            Ok(_) => (),
            Err(_) => {
                panic!("already owned");
            }
        }

        let page = unsafe {
            // Safe to create a page here as all memory from next_page to end of ram is owned by
            // self and the ownership of memory backing the new page is uniquely assigned to the
            // page.
            Page::new(self.next_page)
        };
        // unwrap here because if physical memory runs out before setting up basic hypervisor
        // structures, the system can't continue.
        self.next_page = self.next_page.checked_add_pages(1).unwrap();
        page
    }

    /// Takes `count` Pages from the system map after setting their owner in the global list.
    /// Allows passing ranges of pages around without a mutable reference to the global owners list.
    /// Panics if there are not `count` pages available.
    pub fn take_pages(&mut self, count: usize) -> PageRange {
        // mark them all as owned by the hypervisor, then return an iterator across the pages.
        let first_page = self.next_page;
        // Move self's next page past these taken pages.
        self.next_page = self.next_page.checked_add_pages(count as u64).unwrap();

        for page in first_page.iter_from().take(count) {
            // OK to unwrap as this struct is new and must have space for one owner.
            self.pages
                .get_mut(page)
                .unwrap()
                .push_owner(PageOwnerId::hypervisor())
                .unwrap();
        }

        unsafe {
            // It's safe to create a page range of the memory that `self` forfeited ownership of
            // above and the new `PageRange` is now the unique owner.
            PageRange::new(first_page, self.next_page)
        }
    }

    /// Skips over pages until the alignment requirement is met
    pub fn discard_to_align(&mut self, align: usize) {
        while (self.next_page.bits() as *const u64).align_offset(align) != 0 {
            let _ = self.next_page();
        }
    }
}

impl Iterator for HypMemoryPages {
    type Item = Page<PageSize4k>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_page())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn stub_hyp_mem() -> HypMemoryPages {
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
        let start_page = PageAddr::new(PhysAddr::new(aligned_pointer as u64)).unwrap();
        let hw_map = unsafe {
            // not safe, but this is only a test...
            HwMemMap::new(start_page, MEM_SIZE as u64, start_page)
        };
        let hyp_mem = HypMemoryPages::new(hw_map);
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        hyp_mem
    }

    fn stub_phys_pages() -> (PageState, crate::PageRange) {
        let hyp_mem = stub_hyp_mem();
        let (phys_pages, host_mem) = PageState::from(hyp_mem);
        (phys_pages, host_mem)
    }

    #[test]
    fn hyp_mem_take_pages() {
        let mut hyp_mem = stub_hyp_mem();
        let first = hyp_mem.next_page();
        let mut taken = hyp_mem.take_pages(2);
        let after_taken = hyp_mem.next_page();

        assert_eq!(
            after_taken.addr().bits(),
            first.addr().bits() + (PageSize4k::SIZE_BYTES * 3)
        );
        assert_eq!(
            taken.next().unwrap().addr().bits(),
            first.addr().bits() + PageSize4k::SIZE_BYTES
        );
        assert_eq!(
            taken.next().unwrap().addr().bits(),
            first.addr().bits() + (PageSize4k::SIZE_BYTES * 2)
        );
    }

    #[test]
    fn hyp_mem_take_by_ref() {
        let mut hyp_mem = stub_hyp_mem();
        let first = hyp_mem.next_page();
        let mut taken = hyp_mem.by_ref().take_pages(2);
        let after_taken = hyp_mem.next_page();

        assert_eq!(
            after_taken.addr().bits(),
            first.addr().bits() + (PageSize4k::SIZE_BYTES * 3)
        );
        assert_eq!(
            taken.next().unwrap().addr().bits(),
            first.addr().bits() + PageSize4k::SIZE_BYTES
        );
        assert_eq!(
            taken.next().unwrap().addr().bits(),
            first.addr().bits() + (PageSize4k::SIZE_BYTES * 2)
        );
    }

    #[test]
    fn drop_one_phys_pages_ref() {
        let (mut phys_pages, _host_mem) = stub_phys_pages();
        let new_id = {
            let mut c = phys_pages.clone();
            c.add_active_guest().unwrap()
        };
        assert_eq!(phys_pages.inner.lock().active_guests.len(), 1);

        phys_pages.rm_active_guest(new_id);

        assert_eq!(phys_pages.inner.lock().active_guests.len(), 0);
    }
}
