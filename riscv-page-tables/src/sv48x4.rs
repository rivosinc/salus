// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
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
    const HGATP_MODE: u64 = 9;
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

#[cfg(test)]
mod tests {
    use crate::test_stubs::*;
    use alloc::vec::Vec;
    use page_tracking::*;
    use riscv_pages::*;
    use std::{mem, slice};

    use crate::page_table::*;
    use crate::sv48x4::Sv48x4;

    #[test]
    fn ownership_root_pages() {
        let state = stub_sys_memory();

        let page_tracker = state.page_tracker;
        let id = page_tracker.add_active_guest().unwrap();

        // Should fail as root_pages owner is not set.
        assert!(
            GuestStagePageTable::<Sv48x4>::new(state.root_pages, id, page_tracker.clone()).is_err()
        );
    }

    fn map_and_unmap_sv48x4(page_size: PageSize) {
        let state = stub_sys_memory();

        let page_tracker = state.page_tracker;
        let host_pages = state.host_pages;
        let id = PageOwnerId::host();
        let guest_page_table: GuestStagePageTable<Sv48x4> =
            GuestStagePageTable::new(state.root_pages, id, page_tracker.clone())
                .expect("creating sv48x4");

        let mut pages_to_map = Vec::new();
        for page in host_pages
            .take(2 * PageSize::num_4k_pages(page_size as u64) as usize)
            .filter(|p| p.addr().is_aligned(page_size))
        {
            // Safety: Not safe - just a test
            let page_to_map: Page<ConvertedClean> =
                unsafe { Page::new_with_size(page.addr(), page_size) };
            pages_to_map.push(page_to_map);
        }
        let page_addrs: Vec<SupervisorPageAddr> = pages_to_map.iter().map(|p| p.addr()).collect();
        let mut pte_pages = state.pte_pages.into_iter();
        let gpa_base = PageAddr::new(RawAddr::guest(0x8000_0000, PageOwnerId::host())).unwrap();
        let mapper = guest_page_table
            .map_range(gpa_base, page_size, 2, &mut || pte_pages.next())
            .unwrap();
        for (page, gpa) in pages_to_map
            .into_iter()
            .zip(gpa_base.iter_from_with_size(page_size).unwrap())
        {
            // Write to the page so that we can test if it's retained later.
            unsafe {
                // Not safe - just a test
                let slice = slice::from_raw_parts_mut(
                    page.addr().bits() as *mut u64,
                    page.size() as usize / mem::size_of::<u64>(),
                );
                slice[0] = 0xdeadbeef;
            }
            let mappable = page_tracker.assign_page_for_mapping(page, id).unwrap();
            assert!(mapper.map_page(gpa, mappable).is_ok());
        }
        let version = TlbVersion::new();
        let invalidated = guest_page_table
            .invalidate_range(gpa_base, 2 * page_size as u64, |addr, ps| {
                if ps != page_size {
                    return false;
                }
                page_tracker.is_mapped_page(addr, ps, id, MemType::Ram)
            })
            .unwrap();
        for (paddr, ps) in invalidated {
            assert_eq!(ps, page_size);
            // Safety: Not safe - just a test
            let page: Page<Invalidated> = unsafe { Page::new_with_size(paddr, ps) };
            page_tracker.convert_page(page, version).unwrap();
        }
        let version = version.increment();
        let converted = guest_page_table
            .get_invalidated_pages(gpa_base, 2 * page_size as u64, |addr, ps| {
                if ps != page_size {
                    return false;
                }
                page_tracker.is_converted_page(addr, ps, id, MemType::Ram, version)
            })
            .unwrap();
        let mut locked_pages = LockedPageList::new(page_tracker.clone(), page_size);
        for (paddr, ps) in converted {
            assert_eq!(ps, page_size);
            let page = page_tracker
                .get_converted_page::<Page<ConvertedDirty>>(paddr, ps, id, version)
                .unwrap();
            locked_pages.push(page).unwrap();
        }
        let dirty_page = locked_pages.next().unwrap();
        assert_eq!(dirty_page.addr(), page_addrs[0]);
        assert_eq!(dirty_page.get_u64(0).unwrap(), 0xdeadbeef);
        page_tracker.unlock_page(dirty_page).unwrap();
        let clean_page = locked_pages.next().unwrap().clean();
        assert_eq!(clean_page.addr(), page_addrs[1]);
        assert_eq!(clean_page.get_u64(0).unwrap(), 0);
        page_tracker.unlock_page(clean_page).unwrap();
    }

    #[test]
    fn map_and_unmap_4k_page_sv48x4() {
        map_and_unmap_sv48x4(PageSize::Size4k)
    }

    #[test]
    fn map_and_unmap_2m_pages_sv48x4() {
        map_and_unmap_sv48x4(PageSize::Size2M)
    }
}
