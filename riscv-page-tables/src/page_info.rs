// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use page_collections::page_vec::PageVec;
use riscv_pages::{PageAddr, PageOwnerId, PageSize4k};

use crate::{Error, Result};

/// `PageInfo` holds the owner and previous owner of the referenced page.
#[derive(Clone, Copy, Debug)]
pub struct PageInfo {
    owner: PageOwnerId,
    prev_owner: PageOwnerId,
}

impl PageInfo {
    /// Creates a new `PageInfo` that is owned by the host(The primary VM running in VS mode).
    pub fn new_host_owned() -> Self {
        Self {
            owner: PageOwnerId::host(),
            prev_owner: PageOwnerId::host(),
        }
    }

    /// Pops the current owner, returning the page to the previous owner.
    pub fn pop_owner(&mut self) -> PageOwnerId {
        let result = self.owner;
        self.owner = self.prev_owner;
        result
    }

    /// Pops owners while the provided `check` function returns true or there are no more owners.
    pub fn pop_owners_while<F>(&mut self, check: F)
    where
        F: Fn(&PageOwnerId) -> bool,
    {
        loop {
            let o = self.owner;
            if o == PageOwnerId::host() || !check(&o) {
                break;
            }
            self.pop_owner();
        }
    }

    /// Finds the first owner for which `check` returns true or host if none.
    pub fn find_owner<F>(&self, check: F) -> PageOwnerId
    where
        F: Fn(&PageOwnerId) -> bool,
    {
        if check(&self.owner) {
            return self.owner;
        }
        if check(&self.prev_owner) {
            return self.prev_owner;
        }
        PageOwnerId::host()
    }

    /// Sets the current owner of the page while maintaining a "chain of custody" so the previous
    /// owner is known when the new owner abandons the page.
    pub fn push_owner(&mut self, owner: PageOwnerId) -> Result<()> {
        if self.owner.is_host() {
            self.owner = owner;
            return Ok(());
        }

        if self.prev_owner.is_host() {
            self.prev_owner = self.owner;
            self.owner = owner;
            return Ok(());
        }

        Err(Error::OwnerTooDeep)
    }
}

/// Keeps information for all physical pages in the system.
pub struct Pages {
    pages: PageVec<PageInfo>,
    base_page_index: usize,
}

impl Pages {
    /// Creates a new `Pages`. It will track the information for each page.
    /// `pages`: A vec of `PageInfo` for each page in the system.
    /// `base_page_index`: The index from 0 of the first physical page in the system. Base address
    /// divided by the page size.
    pub fn new(pages: PageVec<PageInfo>, base_page_index: usize) -> Self {
        Self {
            pages,
            base_page_index,
        }
    }

    /// Returns a reference to the `PageInfo` struct for the 4k page at `addr`.
    pub fn get(&self, addr: PageAddr<PageSize4k>) -> Option<&PageInfo> {
        let index = addr.index().checked_sub(self.base_page_index)?;
        self.pages.get(index)
    }

    /// Returns a mutable reference to the `PageInfo` struct for the 4k page at `addr`.
    pub fn get_mut(&mut self, addr: PageAddr<PageSize4k>) -> Option<&mut PageInfo> {
        let index = addr.index().checked_sub(self.base_page_index)?;
        self.pages.get_mut(index)
    }

    /// Returns the number of pages after the page at `addr`
    pub fn num_after(&self, addr: PageAddr<PageSize4k>) -> Option<usize> {
        let offset = addr.index().checked_sub(self.base_page_index)?;
        self.pages.len().checked_sub(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use riscv_pages::{Page, PhysAddr, SequentialPages};

    fn stub_page_vec() -> PageVec<PageInfo> {
        let backing_mem = vec![0u8; 8192];
        let aligned_pointer = unsafe {
            // Not safe - just a test
            backing_mem
                .as_ptr()
                .add(backing_mem.as_ptr().align_offset(4096))
        };
        let addr: PageAddr<PageSize4k> =
            PageAddr::new(PhysAddr::new(aligned_pointer as u64)).unwrap();
        let page = unsafe {
            // Test-only: safe because the backing memory is leaked so the memory used for this page
            // will live until the test exits.
            Page::new(addr)
        };
        PageVec::from(SequentialPages::from(page))
    }

    #[test]
    fn indexing() {
        let mut pages = stub_page_vec();
        let num_pages = 10;
        for _i in 0..num_pages {
            pages.push(PageInfo::new_host_owned());
        }
        let first_index = 1000u64;
        let pages = Pages::new(pages, first_index as usize);

        let before_addr: PageAddr<PageSize4k> =
            PageAddr::new(PhysAddr::new((first_index - 1) * 4096)).unwrap();
        let first_addr: PageAddr<PageSize4k> =
            PageAddr::new(PhysAddr::new(first_index * 4096)).unwrap();
        let last_addr = first_addr.checked_add_pages(num_pages - 1).unwrap();
        let after_addr = last_addr.checked_add_pages(1).unwrap();

        assert!(pages.get(before_addr).is_none());
        assert!(pages.get(first_addr).is_some());
        assert!(pages.get(last_addr).is_some());
        assert!(pages.get(after_addr).is_none());
    }
}
