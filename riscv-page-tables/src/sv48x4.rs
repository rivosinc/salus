// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;

use crate::page_table::Result;
use crate::page_table::*;
use crate::page_tracking::PageTracker;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sv48x4Level {
    L1Table,
    L2Table,
    L3Table,
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

/// An Sv48x4 set of mappings for second stage translation.
pub struct Sv48x4 {
    root: SequentialPages<InternalClean>,
    owner: PageOwnerId,
    page_tracker: PageTracker,
}

impl GuestStagePageTable for Sv48x4 {
    const HGATP_VALUE: u64 = 9;
}

// TODO: Support non-4k page sizes.
impl PlatformPageTable for Sv48x4 {
    type Level = Sv48x4Level;
    type MappedAddressSpace = GuestPhys;

    const TOP_LEVEL_ALIGN: u64 = 16 * 1024;

    fn page_owner_id(&self) -> PageOwnerId {
        self.owner
    }

    fn root_level(&self) -> Self::Level {
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

    fn new(
        root: SequentialPages<InternalClean>,
        owner: PageOwnerId,
        page_tracker: PageTracker,
    ) -> Result<Self> {
        // TODO: Verify ownership of root PT pages.
        if root.page_size().is_huge() {
            return Err(Error::PageSizeNotSupported(root.page_size()));
        }
        if root.base().bits() & (Self::TOP_LEVEL_ALIGN - 1) != 0 {
            return Err(Error::MisalignedPages(root));
        }
        if root.len() < Sv48x4Level::L4Table.table_pages() as u64 {
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

    fn do_fault(&mut self, gpa: RawAddr<Self::MappedAddressSpace>) -> bool {
        // avoid double self borrow, by cloning the pages, each layer borrows self, so the borrow
        // checked can't tell that page_tracker is only borrowed once.
        let page_tracker = self.page_tracker.clone();
        let owner = self.owner;
        if let Some(TableEntryMut::Invalid(pte, level)) = self.walk_until_invalid(gpa) {
            if !level.is_leaf() {
                // We don't support huge pages right now so we shouldn't be faulting them in.
                return false;
            }
            // Unwrap ok, this must be a 4kB page.
            let addr = PageAddr::from_pfn(pte.pfn(), level.leaf_page_size()).unwrap();
            let page: Page<ConvertedDirty> = match page_tracker.get_converted_page(addr, owner) {
                Ok(p) => p,
                Err(_) => {
                    // We don't own the page, or it's not reclaimable.
                    return false;
                }
            };
            page_tracker.reclaim_page(page.clean()).unwrap();
            pte.mark_valid();
            true
        } else {
            false
        }
    }
}
