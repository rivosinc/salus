// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;

use crate::page_table::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sv48Level {
    L1,
    L2,
    L3,
    L4,
}

impl PageTableLevel for Sv48Level {
    fn leaf_page_size(&self) -> PageSize {
        match self {
            Sv48Level::L1 => PageSize::Size4k,
            Sv48Level::L2 => PageSize::Size2M,
            Sv48Level::L3 => PageSize::Size1G,
            Sv48Level::L4 => PageSize::Size512G,
        }
    }

    fn next(&self) -> Option<Self> {
        match self {
            Sv48Level::L1 => None,
            Sv48Level::L2 => Some(Sv48Level::L1),
            Sv48Level::L3 => Some(Sv48Level::L2),
            Sv48Level::L4 => Some(Sv48Level::L3),
        }
    }

    fn addr_shift(&self) -> u64 {
        match self {
            Sv48Level::L1 => 12,
            Sv48Level::L2 => 21,
            Sv48Level::L3 => 30,
            Sv48Level::L4 => 39,
        }
    }

    fn addr_width(&self) -> u64 {
        9
    }

    fn table_pages(&self) -> usize {
        1
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Sv48Level::L1)
    }
}

/// The `Sv48` addressing mode for 1st-stage translation tables.
pub enum Sv48 {}

impl FirstStagePagingMode for Sv48 {
    const SATP_MODE: u64 = 9;
}

impl PagingMode for Sv48 {
    type Level = Sv48Level;
    type MappedAddressSpace = SupervisorVirt;
    const TOP_LEVEL_ALIGN: u64 = 4 * 1024;

    fn root_level() -> Self::Level {
        Sv48Level::L4
    }

    fn max_pte_pages(num_pages: u64) -> u64 {
        // Determine how much ram is needed for host sv48 mappings; 512 8-byte ptes per page
        let num_l1_pages = num_pages / ENTRIES_PER_PAGE + 1;
        let num_l2_pages = num_l1_pages / ENTRIES_PER_PAGE + 1;
        let num_l3_pages = num_l2_pages / ENTRIES_PER_PAGE + 1;
        let num_l4_pages = 1;
        num_l1_pages + num_l2_pages + num_l3_pages + num_l4_pages
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::test_stubs::*;
    use crate::*;

    use std::{mem, slice};

    #[test]
    fn map_and_unmap() {
        let state = stub_sys_memory();

        let mut host_pages = state.host_pages;
        let hyp_page_table: FirstStagePageTable<Sv48> =
            FirstStagePageTable::new(state.root_pages.into_iter().next().unwrap())
                .expect("creating sv48");

        let pages_to_map = [host_pages.next().unwrap(), host_pages.next().unwrap()];
        let mut pte_pages = state.pte_pages.into_iter();
        let gpa_base = PageAddr::new(RawAddr::supervisor_virt(0x8000_0000)).unwrap();
        let pte_fields = PteFieldBits::leaf_with_perms(PteLeafPerms::RW);
        let mapper = hyp_page_table
            .map_range(gpa_base, PageSize::Size4k, 2, &mut || pte_pages.next())
            .unwrap();
        for (page, gpa) in pages_to_map.into_iter().zip(gpa_base.iter_from()) {
            // Write to the page so that we can test if it's retained later.
            unsafe {
                // Not safe - just a test
                let slice = slice::from_raw_parts_mut(
                    page.addr().bits() as *mut u64,
                    page.size() as usize / mem::size_of::<u64>(),
                );
                slice[0] = 0xdeadbeef;
                assert!(mapper.map_addr(gpa, page.addr(), pte_fields).is_ok());
            }
        }
    }
}
