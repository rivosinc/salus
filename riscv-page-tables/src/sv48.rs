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

impl FirstStagePageTable for Sv48 {
    const SATP_VALUE: u64 = 9;
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
