// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;

use crate::page_table::Result;
use crate::page_table::*;
use crate::page_tracking::PageTracker;

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

/// An Sv48 set of mappings for address translation.
pub struct Sv48 {
    root: SequentialPages,
    owner: PageOwnerId,
    page_tracker: PageTracker,
}

impl FirstStagePageTable for Sv48 {
    const SATP_VALUE: u64 = 9;
}

// TODO: Support non-4k page sizes.
impl PlatformPageTable for Sv48 {
    type Level = Sv48Level;
    type MappedAddressSpace = SupervisorVirt;
    const TOP_LEVEL_ALIGN: u64 = 4 * 1024;

    fn page_owner_id(&self) -> PageOwnerId {
        self.owner
    }

    fn root_level(&self) -> Self::Level {
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

    fn new(root: SequentialPages, owner: PageOwnerId, page_tracker: PageTracker) -> Result<Self> {
        // TODO: Verify ownership of root PT pages.
        if root.page_size().is_huge() {
            return Err(Error::PageSizeNotSupported(root.page_size()));
        }
        if root.base().bits() & (Self::TOP_LEVEL_ALIGN - 1) != 0 {
            return Err(Error::MisalignedPages(root));
        }
        if root.len() < Sv48Level::L4.table_pages() as u64 {
            return Err(Error::InsufficientPages(root));
        }
        Ok(Self {
            root,
            owner,
            page_tracker,
        })
    }

    fn page_tracker(&self) -> PageTracker {
        self.page_tracker.clone()
    }

    fn get_root_address(&self) -> SupervisorPageAddr {
        self.root.base()
    }

    fn do_fault(&mut self, _vaddr: RawAddr<Self::MappedAddressSpace>) -> bool {
        false
    }
}
