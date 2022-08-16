// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;

use crate::page_table::*;

/// The levels of the four-level Sv48x4 page table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sv48x4Level {
    /// Level 1 table - references 4k pages.
    L1Table,
    /// Level 2 table - references L1 tables or 2M pages.
    L2Table,
    /// Level 3 table - references L2 tables or 1G pages.
    L3Table,
    /// Level 4 table - references L3 tables or 512G pages.
    L4Table,
}

impl PageTableLevel for Sv48x4Level {
    fn leaf_page_size(&self) -> PageSize {
        match self {
            Sv48x4Level::L1Table => PageSize::Size4k,
            Sv48x4Level::L2Table => PageSize::Size2M,
            Sv48x4Level::L3Table => PageSize::Size1G,
            Sv48x4Level::L4Table => PageSize::Size512G,
        }
    }

    fn next(&self) -> Option<Self> {
        match self {
            Sv48x4Level::L1Table => None,
            Sv48x4Level::L2Table => Some(Sv48x4Level::L1Table),
            Sv48x4Level::L3Table => Some(Sv48x4Level::L2Table),
            Sv48x4Level::L4Table => Some(Sv48x4Level::L3Table),
        }
    }

    fn addr_shift(&self) -> u64 {
        match self {
            Sv48x4Level::L1Table => 12,
            Sv48x4Level::L2Table => 21,
            Sv48x4Level::L3Table => 30,
            Sv48x4Level::L4Table => 39,
        }
    }

    fn addr_width(&self) -> u64 {
        match self {
            Sv48x4Level::L1Table => 9,
            Sv48x4Level::L2Table => 9,
            Sv48x4Level::L3Table => 9,
            Sv48x4Level::L4Table => 11,
        }
    }

    fn table_pages(&self) -> usize {
        match self {
            Sv48x4Level::L1Table => 1,
            Sv48x4Level::L2Table => 1,
            Sv48x4Level::L3Table => 1,
            Sv48x4Level::L4Table => 4,
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Sv48x4Level::L1Table)
    }
}

/// The `Sv48x4` addressing mode for 2nd-stage translation tables.
pub enum Sv48x4 {}

impl GuestStagePagingMode for Sv48x4 {
    const HGATP_VALUE: u64 = 9;
}

impl PagingMode for Sv48x4 {
    type Level = Sv48x4Level;
    type MappedAddressSpace = GuestPhys;

    const TOP_LEVEL_ALIGN: u64 = 16 * 1024;

    fn root_level() -> Self::Level {
        Sv48x4Level::L4Table
    }

    fn max_pte_pages(num_pages: u64) -> u64 {
        // Determine how much ram is needed for host sv48x4 mappings; 512 8-byte ptes per page
        let num_l1_pages = num_pages / ENTRIES_PER_PAGE + 1;
        let num_l2_pages = num_l1_pages / ENTRIES_PER_PAGE + 1;
        let num_l3_pages = num_l2_pages / ENTRIES_PER_PAGE + 1;
        let num_l4_pages = 4;
        num_l1_pages + num_l2_pages + num_l3_pages + num_l4_pages
    }
}
