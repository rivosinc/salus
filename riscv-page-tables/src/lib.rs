// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # Page table management for HS mode on Risc-V.
//! ## Overview
//!
//! - `Page` is the basic building block, representing pages of host supervisor memory. Provided by
//!   the `riscv-pages` crate.
//! - `Sv48x4`, `Sv48` etc are top level page table structures used to manipulate address translation
//! and protection.
//! - `PageTable` provides a generic implementation of a single level of multi-level translation.
//!
//! ## Safety
//!
//! Safe interfaces are exposed by giving each page table (such as `Sv48x4`) ownership of the pages
//! used to construct the page tables. In this way the pages can be manipulated as needed, but only
//! by the owning page table. The details of managing the pages are contained in the page table.
//!
//! Note that leaf pages mapped into the table are assumed to never be safe to "own", they are
//! implicitly shared with the user of the page table (the entity on the other end of that stage of
//! address translation). Don't make a slice out of guest memory after the guest is started with
//! access to the pages.
#![no_std]

mod hw_mem_map;
mod page_table;
pub mod sv48x4;

pub use page_table::PlatformPageTable;
pub use sv48x4::Sv48x4;

pub mod pte;

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
mod tests {
    use riscv_pages::*;

    use super::page_table::*;
    use super::sv48x4::Sv48x4;

    struct PagePool4k {
        mem: std::vec::Vec<u8>,
        next: u64,
    }

    impl PagePool4k {
        fn new(num_pages: usize) -> Self {
            let mem_raw = vec![0u8; 4096 * (num_pages + 4)]; // Add four for alignment
            let mem_start = mem_raw.as_ptr() as u64;
            let align_start = (mem_start + 16384 - 1) & !(16384 - 1);
            Self {
                mem: mem_raw,
                next: align_start,
            }
        }

        fn next_page(&mut self) -> Option<Page4k> {
            let this_page = self.next;
            let mem_start = self.mem.as_ptr() as u64;
            let mem_end = mem_start + self.mem.len() as u64;
            if this_page >= mem_end {
                return None;
            }
            self.next = self.next + 4096;
            // Not safe, but it's a test so don't drop the Pool before the pages...
            unsafe {
                Some(Page4k::new(
                    PageAddr::new(PhysAddr::new(this_page)).unwrap(),
                ))
            }
        }
    }

    const PAGE_SIZE_2MB: u64 = 1024 * 1024 * 2;

    struct PagePool2M {
        mem: std::vec::Vec<u8>,
        next: u64,
    }

    impl PagePool2M {
        fn new(num_pages: usize) -> Self {
            let mem_raw = vec![0u8; PAGE_SIZE_2MB as usize * (num_pages + 1)]; // Add one for alignment
            let mem_start = mem_raw.as_ptr() as u64;
            let align_start = (mem_start + PAGE_SIZE_2MB - 1) & !(PAGE_SIZE_2MB - 1);
            Self {
                mem: mem_raw,
                next: align_start,
            }
        }

        fn next_page(&mut self) -> Option<Page<PageSize2MB>> {
            let this_page = self.next;
            let mem_start = self.mem.as_ptr() as u64;
            let mem_end = mem_start + self.mem.len() as u64;
            if this_page >= mem_end {
                return None;
            }
            self.next = self.next + PAGE_SIZE_2MB;
            // Not safe, but it's a test so don't drop the Pool before the pages...
            unsafe {
                Some(Page::<PageSize2MB>::new(
                    PageAddr::new(PhysAddr::new(this_page)).unwrap(),
                ))
            }
        }
    }

    #[test]
    fn map_one_4k() {
        // pages needed: 4 for L4, 1 for each L1-3, and 1 for the guest.
        let mut mem = PagePool4k::new(16 + 3 + 1);
        let pgd_pages = [
            mem.next_page().unwrap(),
            mem.next_page().unwrap(),
            mem.next_page().unwrap(),
            mem.next_page().unwrap(),
        ];
        let free_pages = [
            mem.next_page().unwrap(),
            mem.next_page().unwrap(),
            mem.next_page().unwrap(),
        ];
        let guest_page = mem.next_page().unwrap();
        let guest_page_addr = guest_page.addr();

        let mut guest_page_table = Sv48x4::new(pgd_pages);
        let guest_addr = 0x8000_0000;
        assert!(guest_page_table
            .map_page_4k(guest_addr, guest_page, &mut free_pages.into_iter())
            .is_ok());
        // check that fetching the address from 0x8000_0000 returns the mapped page.
        let returned_page = guest_page_table.unmap_page(guest_addr).unwrap().unwrap_4k();
        assert!(returned_page.addr().bits() == guest_page_addr.bits());
    }

    #[test]
    fn map_one_2mb() {
        // pages needed: 4 for L4, 1 for each L1-2.
        let mut pte_mem = PagePool4k::new(16 + 3);
        let pgd_pages = [
            pte_mem.next_page().unwrap(),
            pte_mem.next_page().unwrap(),
            pte_mem.next_page().unwrap(),
            pte_mem.next_page().unwrap(),
        ];
        let free_pages = [pte_mem.next_page().unwrap(), pte_mem.next_page().unwrap()];

        let mut guest_mem = PagePool2M::new(1);

        let guest_page = guest_mem.next_page().unwrap();
        let guest_page_addr = guest_page.addr();

        let mut guest_page_table = Sv48x4::new(pgd_pages);
        let guest_addr = 0x8000_0000;
        assert!(guest_page_table
            .map_page_2mb(guest_addr, guest_page, &mut free_pages.into_iter())
            .is_ok());
        // check that fetching the address from 0x8000_0000 returns the mapped page.
        let returned_page = guest_page_table
            .unmap_page(guest_addr)
            .unwrap()
            .unwrap_2mb();
        assert!(returned_page.addr().bits() == guest_page_addr.bits());
    }
}
