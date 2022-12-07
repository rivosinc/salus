// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use page_tracking::{HwMemMap, HwMemRegion, HwMemRegionType, HwReservedMemType, HypPageAlloc};
use riscv_page_tables::{FirstStagePageTable, PteFieldBits, PteLeafPerms, Sv48};
use riscv_pages::{InternalClean, Page, PageAddr, PageSize, RawAddr, SupervisorPhys};
use riscv_regs::{satp, LocalRegisterCopy, SatpHelpers};

// Returns the base, size, and permission pair for the given region if that region type should be
// mapped in the hypervisor's virtual address space.
fn hyp_map_params(r: &HwMemRegion) -> Option<(PageAddr<SupervisorPhys>, u64, PteLeafPerms)> {
    match r.region_type() {
        HwMemRegionType::Available => {
            // map available memory as rwx - unser what it'll be used for.
            Some((r.base(), r.size(), PteLeafPerms::RWX))
        }
        HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved) => {
            // No need to map regions reserved for firmware use
            None
        }
        HwMemRegionType::Reserved(HwReservedMemType::HypervisorImage)
        | HwMemRegionType::Reserved(HwReservedMemType::HostKernelImage)
        | HwMemRegionType::Reserved(HwReservedMemType::HostInitramfsImage) => {
            Some((r.base(), r.size(), PteLeafPerms::RWX))
        }
        HwMemRegionType::Reserved(HwReservedMemType::HypervisorHeap)
        | HwMemRegionType::Reserved(HwReservedMemType::HypervisorPerCpu)
        | HwMemRegionType::Reserved(HwReservedMemType::PageMap) => {
            Some((r.base(), r.size(), PteLeafPerms::RW))
        }
        HwMemRegionType::Mmio(_) => Some((r.base(), r.size(), PteLeafPerms::RW)),
    }
}

// Adds an identity mapping to the given Sv48 table for the specified address range.
fn hyp_map_region(
    sv48: &FirstStagePageTable<Sv48>,
    base: PageAddr<SupervisorPhys>,
    size: u64,
    perms: PteLeafPerms,
    get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
) {
    let region_page_count = PageSize::num_4k_pages(size);
    // Pass through mappings, vaddr=paddr.
    let vaddr = PageAddr::new(RawAddr::supervisor_virt(base.bits())).unwrap();
    // Add mapping for this region to the page table
    let mapper = sv48
        .map_range(vaddr, PageSize::Size4k, region_page_count, get_pte_page)
        .unwrap();
    let pte_fields = PteFieldBits::leaf_with_perms(perms);
    for (virt, phys) in vaddr
        .iter_from()
        .zip(base.iter_from())
        .take(region_page_count as usize)
    {
        // Safe as we will create exactly one mapping to each page and will switch to
        // using that mapping exclusively.
        unsafe {
            mapper.map_addr(virt, phys, pte_fields).unwrap();
        }
    }
}

/// A page table that contains hypervisor mappings.
pub struct HypPageTable {
    inner: FirstStagePageTable<Sv48>,
}

impl HypPageTable {
    /// Return the value of the SATP register for this page table.
    pub fn satp(&self) -> u64 {
        let mut satp = LocalRegisterCopy::<u64, satp::Register>::new(0);
        satp.set_from(&self.inner, 0);
        satp.get()
    }
}

/// A set of global mappings of the hypervisor that can be used to create page tables.
pub struct HypMap {
    mem_map: HwMemMap,
}

impl HypMap {
    /// Create a new hypervisor map from a hardware memory mem map.
    pub fn new(mem_map: HwMemMap) -> HypMap {
        HypMap { mem_map }
    }

    /// Create a new page table based on this memory map.
    pub fn new_page_table(&self, hyp_mem: &mut HypPageAlloc) -> HypPageTable {
        // Create empty sv48 page table
        // Unwrap okay: we expect to have at least one page free or not much will happen anyway.
        let root_page = hyp_mem
            .take_pages_for_hyp_state(1)
            .into_iter()
            .next()
            .unwrap();
        let sv48: FirstStagePageTable<Sv48> =
            FirstStagePageTable::new(root_page).expect("creating first sv48");

        // Map all the regions in the memory map that the hypervisor could need.
        for (base, size, perms) in self.mem_map.regions().filter_map(hyp_map_params) {
            hyp_map_region(&sv48, base, size, perms, &mut || {
                hyp_mem.take_pages_for_hyp_state(1).into_iter().next()
            });
        }
        HypPageTable { inner: sv48 }
    }
}
