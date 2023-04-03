// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use page_tracking::*;
use riscv_pages::*;

use super::page_table::*;
use super::sv48x4::Sv48x4;
use super::*;

pub struct StubState {
    pub root_pages: SequentialPages<InternalClean>,
    pub pte_pages: SequentialPages<InternalClean>,
    pub page_tracker: PageTracker,
    pub host_pages: PageList<Page<ConvertedClean>>,
}

pub fn stub_sys_memory() -> StubState {
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
    let mut hw_map = unsafe {
        // Not safe - just a test
        HwMemMapBuilder::new(MEM_ALIGN as u64)
            .add_memory_region(start_pa, MEM_SIZE.try_into().unwrap())
            .unwrap()
            .build()
    };
    let mut hyp_mem = HypPageAlloc::new(&mut hw_map).unwrap();
    let root_pages = hyp_mem.take_pages_for_host_state_with_alignment(4, Sv48x4::TOP_LEVEL_ALIGN);
    let pte_pages = hyp_mem.take_pages_for_host_state(3);
    let (page_tracker, host_pages) = PageTracker::from(hyp_mem, MEM_ALIGN as u64);
    // Leak the backing ram so it doesn't get freed
    std::mem::forget(backing_mem);
    StubState {
        root_pages,
        pte_pages,
        page_tracker,
        host_pages,
    }
}
