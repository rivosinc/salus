// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # Page table management for HS mode on Risc-V.
//!
//! ## Key types
//!
//! - `Page` is the basic building block, representing pages of host supervisor memory. Provided by
//!   the `riscv-pages` crate.
//! - `Sv48x4`, `Sv48` etc are top level page table structures used to manipulate address
//! translation and protection.
//! - `PageTable` provides a generic implementation of a single level of multi-level translation.
//! - `PageState` - Contains information about active VMs (page owners), manages allocation of
//! unique owner IDs, and per-page state such as current and previous owner. This is system-wide
//! state updated whenever a page owner changes or a VM starts or stops.
//! - `PageMap` - Per-page state (tracks the owner).
//! - `HypPageAlloc` - Initial manager of physical memory. The hypervisor allocates pages from
//! here to store local state. It's turned in to a `PageState` and a pool of ram for the host VM.
//! - `HwMemMap` - Map of system memory, used to determine address ranges to create `Page`s from.
//!
//! ## Initialization
//!
//! `HwMemMap` -> `HypPageAlloc` ---> `PageState`
//!                                 \
//!                                  -------> `SequentialPages`
//!
//! ## Safety
//!
//! Safe interfaces are exposed by giving each page table (such as `Sv48x4`) ownership of the pages
//! used to construct the page tables. In this way the pages can be manipulated as needed, but only
//! by the owning page table. The details of managing the pages are contained in the page table.
//!
//! Note that leaf pages mapped into the table are assumed to never be safe to "own", they are
//! implicitly shared with the user of the page table (the entity on the other end of that stage of
//! address translation). Interacting directly with memory currently mapped to a VM will lead to
//! pain so the interfaces don't support that.
#![no_std]
#![feature(allocator_api)]

extern crate alloc;

mod hw_mem_map;
mod page_info;
mod page_table;
pub mod page_tracking;
pub mod sv48x4;

pub use hw_mem_map::Error as MemMapError;
pub use hw_mem_map::Result as MemMapResult;
pub use hw_mem_map::{HwMemMap, HwMemMapBuilder, HwMemRegion, HwMemRegionType, HwReservedMemType};
pub use page_table::Error as PageTableError;
pub use page_table::Result as PageTableResult;
pub use page_table::{GuestStagePageTable, PlatformPageTable};
pub use page_tracking::Error as PageTrackingError;
pub use page_tracking::Result as PageTrackingResult;
pub use page_tracking::{HypPageAlloc, PageState};
pub use sv48x4::Sv48x4;

pub mod pte;

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
mod tests {
    use alloc::alloc::Global;
    use alloc::vec::Vec;
    use riscv_pages::*;
    use std::{mem, slice};

    use super::page_table::*;
    use super::sv48x4::Sv48x4;
    use super::*;

    fn stub_sys_memory() -> (PageState, Vec<SequentialPages, Global>) {
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
            HwMemMapBuilder::new(Sv48x4::TOP_LEVEL_ALIGN)
                .add_memory_region(start_pa, MEM_SIZE.try_into().unwrap())
                .unwrap()
                .build()
        };
        let hyp_mem = HypPageAlloc::new(hw_map, Global);
        let (phys_pages, host_mem) = PageState::from(hyp_mem, Sv48x4::TOP_LEVEL_ALIGN);
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        (phys_pages, host_mem)
    }

    #[test]
    fn map_and_unmap() {
        let (phys_pages, host_mem) = stub_sys_memory();

        let mut host_pages = host_mem.into_iter().flatten();
        let seq_pages = SequentialPages::from_pages(host_pages.by_ref().take(4)).unwrap();
        let id = phys_pages.add_active_guest().unwrap();
        let mut guest_page_table =
            Sv48x4::new(seq_pages, id, phys_pages.clone()).expect("creating sv48x4");

        let pages_to_map = [host_pages.next().unwrap(), host_pages.next().unwrap()];
        let page_addrs: Vec<SupervisorPageAddr> = pages_to_map.iter().map(|p| p.addr()).collect();
        let mut pte_pages = host_pages.by_ref().take(3);
        let gpa_base = PageAddr::new(RawAddr::guest(0x8000_0000, PageOwnerId::host())).unwrap();
        for (page, gpa) in pages_to_map.into_iter().zip(gpa_base.iter_from()) {
            // Write to the page so that we can test if it's retained later.
            unsafe {
                // Not safe - just a test
                let slice = slice::from_raw_parts_mut(
                    page.addr().bits() as *mut u64,
                    page.addr().size() as usize / mem::size_of::<u64>(),
                );
                slice[0] = 0xdeadbeef;
            }
            assert!(phys_pages
                .set_page_owner(page.addr(), guest_page_table.page_owner_id())
                .is_ok());
            assert!(guest_page_table
                .map_page(gpa, page, &mut || pte_pages.next())
                .is_ok());
        }
        let dirty_page: Page = guest_page_table.unmap_page(gpa_base).unwrap().to_page();
        assert_eq!(dirty_page.addr(), page_addrs[0]);
        assert_eq!(dirty_page.get_u64(0).unwrap(), 0xdeadbeef);
        let clean_page = Page::from(CleanPage::from(
            guest_page_table
                .unmap_page(gpa_base.checked_add_pages(1).unwrap())
                .unwrap(),
        ));
        assert_eq!(clean_page.addr(), page_addrs[1]);
        assert_eq!(clean_page.get_u64(0).unwrap(), 0);
    }
}
