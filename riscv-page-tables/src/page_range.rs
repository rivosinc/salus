// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{AlignedPageAddr, Page, PageSize4k};

/// A range of pages with the owner set that can be consumed as an iterator.
pub struct PageRange {
    next_page: AlignedPageAddr<PageSize4k>,
    end_page: AlignedPageAddr<PageSize4k>,
}

impl PageRange {
    /// Creates a new PageRange spanning the from `start` to `end`.
    /// # Safety
    /// The caller must guarantee that the memory in the range is uniquely owned. Passing a range of
    /// pages to `PageRange::new` assigns ownership of all memory in that range.
    pub(crate) unsafe fn new(
        start: AlignedPageAddr<PageSize4k>,
        end: AlignedPageAddr<PageSize4k>,
    ) -> Self {
        Self {
            next_page: start,
            end_page: end,
        }
    }

    /// Get the next 4k page available to the host.
    pub fn next_addr(&self) -> AlignedPageAddr<PageSize4k> {
        self.next_page
    }

    /// Returns the amount of memory that remains available to the host.
    pub fn remaining_size(&self) -> u64 {
        self.end_page.bits() - self.next_page.bits()
    }
}

impl Iterator for PageRange {
    type Item = Page<PageSize4k>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next_page == self.end_page {
            return None;
        }
        let page = unsafe {
            // Safe to create a page here as all memory from next_page to end of ram is owned by
            // self and the ownership of memory backing the new page is uniquely assigned to the
            // page.
            Page::new(self.next_page)
        };
        // Safe to unwrap as if physical memory runs out before setting up basic hypervisor
        // structures, the system can't continue.
        self.next_page = self.next_page.checked_add_pages(1)?;
        Some(page)
    }
}
