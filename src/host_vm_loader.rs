// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use arrayvec::ArrayString;
use core::{alloc::Allocator, fmt, slice};
use device_tree::{DeviceTree, DeviceTreeResult, DeviceTreeSerializer};
use drivers::CpuInfo;
use riscv_page_tables::{HwMemRegion, HypPageAlloc, PlatformPageTable};
use riscv_pages::{GuestPhysAddr, PageAddr, PageOwnerId, PageSize, RawAddr, SequentialPages};

use crate::print_util::*;
use crate::println;
use crate::vm::Host;
use crate::vm_pages::HostRootBuilder;

// Where the kernel, initramfs, and FDT will be located in the guest physical address space.
//
// TODO: Kernel offset should be pulled from the header in the kernel image.
const KERNEL_OFFSET: u64 = 0x20_0000;
const INITRAMFS_OFFSET: u64 = KERNEL_OFFSET + 0x800_0000;
// Assuming RAM base at 2GB, ends up at 3GB - 16MB which is consistent with QEMU.
const FDT_OFFSET: u64 = 0x3f00_0000;

/// A builder for the host VM's device-tree. Starting with the hypervisor's device-tree, makes the
/// necessary modifications to create a device-tree that reflects the hardware available to the
/// host VM.
struct HostDtBuilder<A: Allocator + Clone> {
    tree: DeviceTree<A>,
}

impl<A: Allocator + Clone> HostDtBuilder<A> {
    /// Creates a new builder from the hypervisor device-tree.
    pub fn new(hyp_dt: &DeviceTree<A>) -> DeviceTreeResult<Self> {
        let mut host_dt = DeviceTree::new(hyp_dt.alloc());
        let hyp_root = hyp_dt.get_node(hyp_dt.root().unwrap()).unwrap();
        let host_root_id = host_dt.add_node("", None)?;
        let host_root = host_dt.get_mut_node(host_root_id).unwrap();

        // Clone the properties of the root node as-is.
        host_root.set_props(hyp_root.props().cloned())?;

        // Selectively clone the sub-nodes.
        for &c in hyp_root.children() {
            let hyp_child = hyp_dt.get_node(c).unwrap();
            if hyp_child.name().starts_with("chosen") || hyp_child.name() == "soc" {
                // For "soc" and "chosen" just clone the properties. We'll add sub-nodes (e.g. for
                // devices we're passing through) later if necessary.
                let host_child_id = host_dt.add_node(hyp_child.name(), Some(host_root_id))?;
                let host_child = host_dt.get_mut_node(host_child_id).unwrap();
                if hyp_child.name().starts_with("chosen") {
                    if let Some(p) = hyp_child.props().find(|p| p.name().starts_with("bootargs")) {
                        host_child.insert_prop(p.clone())?;
                    }
                } else {
                    host_child.set_props(hyp_child.props().cloned())?;
                }
            }
        }

        Ok(Self { tree: host_dt })
    }

    /// Adds a "memory" node to the device tree with the given base and size.
    pub fn add_memory_node(mut self, mem_base: u64, mem_size: u64) -> DeviceTreeResult<Self> {
        let mut mem_name = ArrayString::<32>::new();
        fmt::write(&mut mem_name, format_args!("memory@{:08x}", mem_base)).unwrap();
        let mem_id = self.tree.add_node(mem_name.as_str(), self.tree.root())?;
        let mem_node = self.tree.get_mut_node(mem_id).unwrap();
        mem_node.add_prop("device_type")?.set_value_str("memory")?;
        // TODO: Assumes #address-cells/#size-cells of 2.
        mem_node
            .add_prop("reg")?
            .set_value_u64(&[mem_base, mem_size])?;

        Ok(self)
    }

    /// Adds CPU nodes to the device tree.
    pub fn add_cpu_nodes(mut self) -> DeviceTreeResult<Self> {
        CpuInfo::get().add_host_cpu_nodes(&mut self.tree)?;
        Ok(self)
    }

    /// Updates the "chosen" node with the location of the initramfs image.
    pub fn set_initramfs_addr(mut self, start_addr: u64, len: u64) -> DeviceTreeResult<Self> {
        let chosen_id = self
            .tree
            .iter()
            .find(|n| n.name().starts_with("chosen"))
            .unwrap()
            .id();
        let chosen_node = self.tree.get_mut_node(chosen_id).unwrap();

        chosen_node
            .add_prop("linux,initrd-start")?
            .set_value_u64(&[start_addr])?;
        let end_addr = start_addr.checked_add(len).unwrap();
        chosen_node
            .add_prop("linux,initrd-end")?
            .set_value_u64(&[end_addr])?;

        Ok(self)
    }

    pub fn tree(self) -> DeviceTree<A> {
        self.tree
    }
}

/// Loader for the host VM.
///
/// Given the hypervisor's device tree and the host kernel & initramfs image, build a device-tree
/// and address space for the host VM.
///
/// In order to allow the host VM to allocate physically-aligned blocks necessary for guest VM
/// creation (specifically, the root of the G-stage page-table), we guarantee that each
/// contiguous T::TOP_LEVEL_ALIGN block of the guest physical address space of the host VM maps to
/// a contiguous T::TOP_LEVEL_ALIGN block of the host physical address space.
pub struct HostVmLoader<T: PlatformPageTable, A: Allocator + Clone> {
    hypervisor_dt: DeviceTree<A>,
    kernel: HwMemRegion,
    initramfs: Option<HwMemRegion>,
    root_builder: HostRootBuilder<T>,
    guest_tracking_pages: SequentialPages,
    fdt_pages: SequentialPages,
    zero_pages: Vec<SequentialPages, A>,
    guest_phys_base: GuestPhysAddr,
}

impl<T: PlatformPageTable, A: Allocator + Clone> HostVmLoader<T, A> {
    /// Creates a new loader with the given device-tree and kernel & initramfs images. Uses
    /// `page_alloc` to allocate any additional pages that are necessary to load the VM.
    pub fn new(
        hypervisor_dt: DeviceTree<A>,
        kernel: HwMemRegion,
        initramfs: Option<HwMemRegion>,
        mut page_alloc: HypPageAlloc<A>,
    ) -> Self {
        // Reserve pages for tracking the host's guests.
        let guest_tracking_pages = page_alloc.take_pages(2);

        // Reserve a contiguous chunk for the host's FDT. We assume it will be no bigger than the
        // size of the hypervisor's FDT and we align it to `T::TOP_LEVEL_ALIGN` to maintain the
        // contiguous mapping guarantee from GPA -> HPA mentioned above.
        let fdt_size = {
            let size = DeviceTreeSerializer::new(&hypervisor_dt).output_size();
            ((size as u64) + T::TOP_LEVEL_ALIGN - 1) & !(T::TOP_LEVEL_ALIGN - 1)
        };
        let num_fdt_pages = fdt_size / PageSize::Size4k as u64;
        let fdt_pages = page_alloc
            .take_pages_with_alignment(num_fdt_pages.try_into().unwrap(), T::TOP_LEVEL_ALIGN);

        // We use the size of our (the hypervisor's) physical address to estimate the size of the
        // host's guest phsyical address space since we build the host's address space to match the
        // actual physical address space, but with the holes (for hypervisor memory, other reserved
        // regions) removed. This results in a bit of an overestimate for determining the number of
        // page-table pages, but we should expect the holes to be pretty small.
        //
        // TODO: Support discontiguous physical memory.
        let (phys_mem_base, phys_mem_size) = {
            let node = hypervisor_dt
                .iter()
                .find(|n| n.name().starts_with("memory"))
                .unwrap();
            let mut reg = node
                .props()
                .find(|p| p.name() == "reg")
                .unwrap()
                .value_u64();
            (reg.next().unwrap(), reg.next().unwrap())
        };

        let (zero_pages, root_builder) =
            HostRootBuilder::<T>::from_hyp_mem(page_alloc, phys_mem_size);

        Self {
            hypervisor_dt,
            kernel,
            initramfs,
            root_builder,
            guest_tracking_pages,
            fdt_pages,
            zero_pages,
            guest_phys_base: RawAddr::guest(phys_mem_base, PageOwnerId::host()),
        }
    }

    /// Builds a device tree for the host VM, flattening it to a range of pages that will be
    /// mapped into the address space in `build_address_space()`.
    pub fn build_device_tree(self) -> Self {
        // Now that the hypervisor is done claiming memory, determine the actual size of the host's
        // address space.
        let ram_size = self
            .zero_pages
            .iter()
            .fold(0, |acc, r| acc + r.length_bytes())
            + self.fdt_pages.length_bytes()
            + self.kernel.size()
            + self.initramfs.map(|r| r.size()).unwrap_or(0);
        assert!(ram_size >= FDT_OFFSET + self.fdt_pages.length_bytes());

        // Construct a stripped-down device-tree for the host VM.
        let mut host_dt_builder = HostDtBuilder::new(&self.hypervisor_dt)
            .unwrap()
            .add_memory_node(self.guest_phys_base.bits(), ram_size)
            .unwrap()
            .add_cpu_nodes()
            .unwrap();
        if let Some(r) = self.initramfs {
            host_dt_builder = host_dt_builder
                .set_initramfs_addr(
                    self.guest_phys_base
                        .checked_increment(INITRAMFS_OFFSET)
                        .unwrap()
                        .bits(),
                    r.size(),
                )
                .unwrap();
        }

        // TODO: Add IMSIC & PCIe nodes.
        let host_dt = host_dt_builder.tree();

        println!("Host DT: {}", host_dt);

        // Serialize the device-tree.
        let dt_writer = DeviceTreeSerializer::new(&host_dt);
        assert!(dt_writer.output_size() <= self.fdt_pages.length_bytes().try_into().unwrap());
        let fdt_slice = unsafe {
            // Safe because we own these pages.
            slice::from_raw_parts_mut(
                self.fdt_pages.base().bits() as *mut u8,
                self.fdt_pages.length_bytes().try_into().unwrap(),
            )
        };
        dt_writer.write_to(fdt_slice);
        self
    }

    /// Constructs the address space for the host VM, returning a `Host` that is ready to run.
    pub fn build_address_space(mut self) -> Host<T> {
        // HostRootBuilder guarantees that the host pages it returns start at
        // T::TOP_LEVEL_ALIGN-aligned block, and because we built the HwMemMap with a minimum
        // region alignment of T::TOP_LEVEL_ALIGN any discontiguous ranges are also guaranteed to
        // be aligned.
        let mut zero_pages_iter = self.zero_pages.into_iter().flatten();

        // Now fill in the address space, inserting zero pages around the kernel/initramfs/FDT.
        let mut current_gpa = PageAddr::new(self.guest_phys_base).unwrap();
        let num_pages = KERNEL_OFFSET / PageSize::Size4k as u64;
        self.root_builder = self.root_builder.add_4k_pages(
            current_gpa,
            zero_pages_iter.by_ref().take(num_pages.try_into().unwrap()),
        );
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let num_kernel_pages = self.kernel.size() / PageSize::Size4k as u64;
        let kernel_pages = unsafe {
            // Safe because HwMemMap reserved this region.
            SequentialPages::from_mem_range(self.kernel.base().get_4k_addr(), num_kernel_pages)
        };
        self.root_builder = self
            .root_builder
            .add_4k_data_pages(current_gpa, kernel_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_kernel_pages).unwrap();

        if let Some(r) = self.initramfs {
            let num_pages =
                (INITRAMFS_OFFSET - (KERNEL_OFFSET + self.kernel.size())) / PageSize::Size4k as u64;
            self.root_builder = self.root_builder.add_4k_pages(
                current_gpa,
                zero_pages_iter.by_ref().take(num_pages.try_into().unwrap()),
            );
            current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

            let num_initramfs_pages = r.size() / PageSize::Size4k as u64;
            let initramfs_pages = unsafe {
                // Safe because HwMemMap reserved this region.
                SequentialPages::from_mem_range(r.base().get_4k_addr(), num_initramfs_pages)
            };
            self.root_builder = self
                .root_builder
                .add_4k_data_pages(current_gpa, initramfs_pages.into_iter());
            current_gpa = current_gpa.checked_add_pages(num_initramfs_pages).unwrap();
        }

        let num_pages = (FDT_OFFSET - (current_gpa.bits() - self.guest_phys_base.bits()))
            / PageSize::Size4k as u64;
        self.root_builder = self.root_builder.add_4k_pages(
            current_gpa,
            zero_pages_iter.by_ref().take(num_pages.try_into().unwrap()),
        );
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let num_fdt_pages = self.fdt_pages.len();
        self.root_builder = self
            .root_builder
            .add_4k_data_pages(current_gpa, self.fdt_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_fdt_pages).unwrap();

        self.root_builder = self.root_builder.add_4k_pages(current_gpa, zero_pages_iter);
        let mut host = Host::new(self.root_builder.create_host(), self.guest_tracking_pages);
        host.add_device_tree(self.guest_phys_base.bits() + FDT_OFFSET);
        host.set_entry_address(self.guest_phys_base.bits() + KERNEL_OFFSET);
        host
    }
}
