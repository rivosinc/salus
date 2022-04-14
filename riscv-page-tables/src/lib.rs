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
//! - `PageRange` - Provides an iterator of `Page`s. Used to pass chunks of memory to the various
//! stages of system initialization.
//! - `HwMemMap` - Map of system memory, used to determine address ranges to create `Page`s from.
//!
//! ## Initialization
//!
//! `HwMemMap` -> `HypPageAlloc` ---> `PageState`
//!                                 \
//!                                  -------> `PageRange`
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

mod hw_mem_map;
mod page_info;
pub mod page_range;
mod page_table;
pub mod page_tracking;
pub mod sv48x4;

pub use hw_mem_map::Error as MemMapError;
pub use hw_mem_map::Result as MemMapResult;
pub use hw_mem_map::{HwMemMap, HwMemMapBuilder, HwMemRegion, HwMemType, HwReservedMemType};
pub use page_range::PageRange;
pub use page_table::Error as PageTableError;
pub use page_table::PlatformPageTable;
pub use page_table::Result as PageTableResult;
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
    use riscv_pages::*;

    use super::page_table::*;
    use super::sv48x4::Sv48x4;
    use super::*;

    fn stub_sys_memory() -> (PageState, crate::PageRange) {
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
        let start_pa = PhysAddr::new(aligned_pointer as u64);
        let hw_map = unsafe {
            // Not safe - just a test
            HwMemMapBuilder::new(Sv48x4::TOP_LEVEL_ALIGN)
                .add_memory_region(start_pa, MEM_SIZE.try_into().unwrap())
                .unwrap()
                .build()
        };
        let hyp_mem = HypPageAlloc::new(hw_map);
        let (phys_pages, host_mem) = PageState::from(hyp_mem);
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        (phys_pages, host_mem)
    }

    #[test]
    fn map_one_4k() {
        let (phys_pages, mut host_mem) = stub_sys_memory();

        let seq_pages = match SequentialPages::from_pages(host_mem.by_ref().take(4)) {
            Ok(s) => s,
            Err(_) => panic!("setting up seq pages"),
        };
        let mut guest_page_table = Sv48x4::new(seq_pages, PageOwnerId::new(2).unwrap(), phys_pages)
            .expect("creating sv48x4");

        let guest_page = host_mem.next().unwrap();
        let guest_page_addr = guest_page.addr();
        let mut free_pages = host_mem.by_ref().take(3);
        let guest_addr = 0x8000_0000;
        assert!(guest_page_table
            .map_page_4k(guest_addr, guest_page, &mut || free_pages.next())
            .is_ok());
        // check that fetching the address from 0x8000_0000 returns the mapped page.
        let returned_page = guest_page_table.unmap_page(guest_addr).unwrap().unwrap_4k();
        assert!(returned_page.addr().bits() == guest_page_addr.bits());
    }
}
