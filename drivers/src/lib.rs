// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # Hardware drivers

#![no_std]
#![feature(allocator_api, int_log, result_option_inspect, iter_advance_by)]

extern crate alloc;

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;

/// Provides access to topology and static properties of the CPU the hypervisor is running on.
pub mod cpu;
/// Provides the driver for the IMSIC from the AIA spec.
pub mod imsic;
/// Provides the driver for the IOMMU from the proposed RISCV-IOMMU spec.
pub mod iommu;
/// Provides PCI bus scanning and device discovery.
pub mod pci;
/// Caches information about platform hardware and firmware PMU counters.
pub mod pmu;

pub use cpu::{CpuId, CpuInfo, MAX_CPUS};

#[cfg(test)]
mod tests {
    use super::imsic::Imsic;
    use super::*;
    use alloc::vec::Vec;
    use device_tree::DeviceTree;
    use page_tracking::{HwMemMap, HwMemMapBuilder, HwMemRegionType};
    use riscv_pages::{DeviceMemType, PageAddr, PageSize, RawAddr};

    const NUM_CPUS: u32 = 4;
    const HART_ID_BASE: u32 = 4;
    const PHANDLE_BASE: u32 = 10;
    const IMSIC_PHANDLE: u32 = 99;

    const GUEST_BITS: u32 = 3;
    const GROUP_SHIFT: u32 = 24;

    fn stub_tree() -> DeviceTree {
        // Create a tree with a couple of CPUs.
        let mut tree = DeviceTree::new();
        let root = tree.add_node("", None).unwrap();
        let root_node = tree.get_mut_node(root).unwrap();
        root_node
            .add_prop("#address-cells")
            .unwrap()
            .set_value_u32(&[2])
            .unwrap();
        root_node
            .add_prop("#size-cells")
            .unwrap()
            .set_value_u32(&[2])
            .unwrap();
        let cpu_node_id = tree.add_node("cpus", Some(root)).unwrap();
        let cpu_node = tree.get_mut_node(cpu_node_id).unwrap();
        cpu_node
            .add_prop("#address-cells")
            .unwrap()
            .set_value_u32(&[1])
            .unwrap();
        cpu_node
            .add_prop("#size-cells")
            .unwrap()
            .set_value_u32(&[0])
            .unwrap();
        cpu_node
            .add_prop("timebase-frequency")
            .unwrap()
            .set_value_u32(&[100000])
            .unwrap();

        // Assign non-standard hart IDs to test CpuID <-> hart ID translation.
        for i in 0..NUM_CPUS {
            let id = tree
                .add_node(format!("cpu@{}", i).as_str(), Some(cpu_node_id))
                .unwrap();
            let node = tree.get_mut_node(id).unwrap();
            node.add_prop("device_type")
                .unwrap()
                .set_value_str("cpu")
                .unwrap();
            node.add_prop("reg")
                .unwrap()
                .set_value_u32(&[HART_ID_BASE + i])
                .unwrap();
            node.add_prop("riscv,isa")
                .unwrap()
                .set_value_str("rv64imafdcvsuh_sstc")
                .unwrap();
            node.add_prop("mmu-type")
                .unwrap()
                .set_value_str("riscv,sv48")
                .unwrap();

            let intc_id = tree.add_node("interrupt-controller", Some(id)).unwrap();
            let intc_node = tree.get_mut_node(intc_id).unwrap();
            intc_node
                .add_prop("#interrupt-cells")
                .unwrap()
                .set_value_u32(&[1])
                .unwrap();
            intc_node.add_prop("interrupt-controller").unwrap();
            intc_node
                .add_prop("compatible")
                .unwrap()
                .set_value_str("riscv,cpu-intc-aia\0riscv,cpu-intc")
                .unwrap();
            intc_node
                .add_prop("phandle")
                .unwrap()
                .set_value_u32(&[PHANDLE_BASE + i])
                .unwrap();
        }

        let soc_node_id = tree.add_node("soc", Some(root)).unwrap();
        let soc_node = tree.get_mut_node(soc_node_id).unwrap();
        soc_node
            .add_prop("#address-cells")
            .unwrap()
            .set_value_u32(&[2])
            .unwrap();
        soc_node
            .add_prop("#size-cells")
            .unwrap()
            .set_value_u32(&[2])
            .unwrap();
        soc_node.add_prop("ranges").unwrap();

        // Create an IMSIC node with two groups of two CPUs.
        let imsic_node_id = tree.add_node("imsic@4000000", Some(soc_node_id)).unwrap();
        let imsic_node = tree.get_mut_node(imsic_node_id).unwrap();
        imsic_node
            .add_prop("compatible")
            .unwrap()
            .set_value_str("riscv,imsics")
            .unwrap();
        imsic_node
            .add_prop("phandle")
            .unwrap()
            .set_value_u32(&[IMSIC_PHANDLE])
            .unwrap();
        imsic_node.add_prop("msi-controller").unwrap();
        imsic_node.add_prop("interrupt-controller").unwrap();
        imsic_node
            .add_prop("#interrupt-cells")
            .unwrap()
            .set_value_u32(&[0])
            .unwrap();
        imsic_node
            .add_prop("riscv,guest-index-bits")
            .unwrap()
            .set_value_u32(&[GUEST_BITS])
            .unwrap();
        imsic_node
            .add_prop("riscv,hart-index-bits")
            .unwrap()
            .set_value_u32(&[1])
            .unwrap();
        imsic_node
            .add_prop("riscv,num-ids")
            .unwrap()
            .set_value_u32(&[127])
            .unwrap();
        imsic_node
            .add_prop("riscv,group-index-bits")
            .unwrap()
            .set_value_u32(&[1])
            .unwrap();
        imsic_node
            .add_prop("riscv,group-index-shift")
            .unwrap()
            .set_value_u32(&[GROUP_SHIFT])
            .unwrap();
        imsic_node
            .add_prop("reg")
            .unwrap()
            .set_value_u64(&[0x4000_0000, 0x1_0000, 0x4100_0000, 0x1_0000])
            .unwrap();
        let mut interrupts = Vec::new();
        for i in 0..NUM_CPUS {
            interrupts.push(PHANDLE_BASE + i);
            interrupts.push(9);
        }
        imsic_node
            .add_prop("interrupts-extended")
            .unwrap()
            .set_value_u32(&interrupts)
            .unwrap();

        tree
    }

    fn stub_mem_map() -> HwMemMap {
        let builder = unsafe {
            // Not safe -- it's a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(RawAddr::supervisor(0x8000_0000), 0x4000_0000)
                .unwrap()
        };
        builder.build()
    }

    #[test]
    fn build_cpu_info() {
        let tree = stub_tree();
        CpuInfo::parse_from(&tree);

        let cpu_info = CpuInfo::get();
        assert!(cpu_info.has_sstc());
        assert_eq!(cpu_info.num_cpus(), 4);
        for i in 0..cpu_info.num_cpus() {
            let hart_id = cpu_info.cpu_to_hart_id(CpuId::new(i)).unwrap();
            assert_eq!(hart_id, i as u32 + HART_ID_BASE);
            let cpu_id = cpu_info.hart_id_to_cpu(hart_id).unwrap();
            assert_eq!(i, cpu_id.raw());

            assert_eq!(
                i as u32 + PHANDLE_BASE,
                cpu_info.cpu_to_intc_phandle(CpuId::new(i)).unwrap()
            );
            assert_eq!(
                cpu_id,
                cpu_info
                    .intc_phandle_to_cpu(i as u32 + PHANDLE_BASE)
                    .unwrap()
            );
        }
    }

    #[test]
    fn probe_imsic() {
        let tree = stub_tree();
        CpuInfo::parse_from(&tree);
        let mut mem_map = stub_mem_map();
        Imsic::probe_from(&tree, &mut mem_map).unwrap();

        let imsic = Imsic::get();
        let geometry = imsic.phys_geometry();
        assert_eq!(geometry.guests_per_hart(), (1 << GUEST_BITS as usize) - 1);
        assert_eq!(imsic.phandle(), IMSIC_PHANDLE);

        // Make sure the interrupt file addresses are correct.
        let group0_addr = geometry.base_addr();
        assert_eq!(group0_addr.bits(), 0x4000_0000);
        let per_hart_pages = 1 << GUEST_BITS as u64;
        let cpu1_loc = imsic.supervisor_file_location(CpuId::new(1)).unwrap();
        assert_eq!(
            geometry.location_to_addr(cpu1_loc).unwrap(),
            group0_addr.checked_add_pages(per_hart_pages).unwrap()
        );
        let group1_addr =
            PageAddr::new(RawAddr::supervisor(group0_addr.bits() + (1 << GROUP_SHIFT))).unwrap();
        let cpu2_loc = imsic.supervisor_file_location(CpuId::new(2)).unwrap();
        assert_eq!(geometry.location_to_addr(cpu2_loc).unwrap(), group1_addr);

        // Make sure `HwMemMap` got updated.
        let mut iter = mem_map
            .regions()
            .filter(|r| r.region_type() == HwMemRegionType::Mmio(DeviceMemType::Imsic));
        let group0 = iter.next().unwrap();
        assert_eq!(group0.base(), group0_addr);
        let group_size = 2 * per_hart_pages * PageSize::Size4k as u64;
        assert_eq!(group0.size(), group_size);
        let group1 = iter.next().unwrap();
        assert_eq!(group1.base(), group1_addr);
        assert_eq!(group1.size(), group_size);
    }
}
