// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayString;
use core::{alloc::Allocator, fmt, slice};
use device_tree::{DeviceTree, DeviceTreeResult, DeviceTreeSerializer};
use drivers::{CpuId, CpuInfo, Imsic, ImsicGuestId};
use page_tracking::{HwMemRegion, HypPageAlloc, PageList};
use riscv_page_tables::GuestStagePageTable;
use riscv_pages::*;

use crate::print_util::*;
use crate::println;
use crate::vm::{HostVm, VmStateFinalized, VmStateInitializing};

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

        // Add a 'chosen' node and copy the bootargs if they exist.
        let host_chosen_id = host_dt.add_node("chosen", Some(host_root_id))?;
        let host_chosen = host_dt.get_mut_node(host_chosen_id).unwrap();
        if let Some(hyp_chosen) = hyp_dt.iter().find(|n| n.name() == "chosen") {
            if let Some(p) = hyp_chosen.props().find(|p| p.name() == "bootargs") {
                host_chosen.insert_prop(p.clone())?;
            }
        }

        Ok(Self { tree: host_dt })
    }

    /// Adds a "memory" node to the device tree with the given base and size.
    pub fn add_memory_node(
        mut self,
        mem_base: GuestPhysAddr,
        mem_size: u64,
    ) -> DeviceTreeResult<Self> {
        let mut mem_name = ArrayString::<32>::new();
        fmt::write(
            &mut mem_name,
            format_args!("memory@{:08x}", mem_base.bits()),
        )
        .unwrap();
        let mem_id = self.tree.add_node(mem_name.as_str(), self.tree.root())?;
        let mem_node = self.tree.get_mut_node(mem_id).unwrap();
        mem_node.add_prop("device_type")?.set_value_str("memory")?;
        // TODO: Assumes #address-cells/#size-cells of 2.
        mem_node
            .add_prop("reg")?
            .set_value_u64(&[mem_base.bits(), mem_size])?;

        Ok(self)
    }

    /// Adds CPU nodes to the device tree.
    pub fn add_cpu_nodes(mut self) -> DeviceTreeResult<Self> {
        CpuInfo::get().add_host_cpu_nodes(&mut self.tree)?;
        Ok(self)
    }

    /// Add any MMIO devices to the device tree.
    pub fn add_device_nodes(mut self, imsic_addr: GuestPhysAddr) -> DeviceTreeResult<Self> {
        // First add a 'soc' subnode of the root.
        let soc_node_id = self.tree.add_node("soc", self.tree.root())?;
        let soc_node = self.tree.get_mut_node(soc_node_id).unwrap();
        soc_node.add_prop("#address-cells")?.set_value_u32(&[2])?;
        soc_node.add_prop("#size-cells")?.set_value_u32(&[2])?;
        soc_node.add_prop("ranges")?;

        // All we have is the IMSIC for now.
        Imsic::get().add_host_imsic_node(&mut self.tree, imsic_addr)?;

        // TODO: Add PCIe nodes.

        Ok(self)
    }

    /// Updates the "chosen" node with the location of the initramfs image.
    pub fn set_initramfs_addr(
        mut self,
        start_addr: GuestPhysAddr,
        len: u64,
    ) -> DeviceTreeResult<Self> {
        let chosen_id = self
            .tree
            .iter()
            .find(|n| n.name().starts_with("chosen"))
            .unwrap()
            .id();
        let chosen_node = self.tree.get_mut_node(chosen_id).unwrap();

        chosen_node
            .add_prop("linux,initrd-start")?
            .set_value_u64(&[start_addr.bits()])?;
        let end_addr = start_addr.checked_increment(len).unwrap();
        chosen_node
            .add_prop("linux,initrd-end")?
            .set_value_u64(&[end_addr.bits()])?;

        Ok(self)
    }

    pub fn tree(self) -> DeviceTree<A> {
        self.tree
    }
}

enum FdtPages {
    Clean(SequentialPages<ConvertedClean>),
    Initialized(SequentialPages<ConvertedInitialized>),
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
pub struct HostVmLoader<T: GuestStagePageTable, A: Allocator + Clone> {
    hypervisor_dt: DeviceTree<A>,
    kernel: HwMemRegion,
    initramfs: Option<HwMemRegion>,
    vm: HostVm<T, VmStateInitializing>,
    fdt_pages: FdtPages,
    zero_pages: PageList<Page<ConvertedClean>>,
    guest_ram_base: GuestPhysAddr,
    ram_size: u64,
}

impl<T: GuestStagePageTable, A: Allocator + Clone> HostVmLoader<T, A> {
    /// Creates a new loader with the given device-tree and kernel & initramfs images. Uses
    /// `page_alloc` to allocate any additional pages that are necessary to load the VM.
    pub fn new(
        hypervisor_dt: DeviceTree<A>,
        kernel: HwMemRegion,
        initramfs: Option<HwMemRegion>,
        guest_ram_base: GuestPhysAddr,
        guest_phys_size: u64,
        mut page_alloc: HypPageAlloc,
    ) -> Self {
        // Reserve a contiguous chunk for the host's FDT. We assume it will be no bigger than the
        // size of the hypervisor's FDT and we align it to `T::TOP_LEVEL_ALIGN` to maintain the
        // contiguous mapping guarantee from GPA -> HPA mentioned above.
        let fdt_size = {
            let size = DeviceTreeSerializer::new(&hypervisor_dt).output_size();
            ((size as u64) + T::TOP_LEVEL_ALIGN - 1) & !(T::TOP_LEVEL_ALIGN - 1)
        };
        let num_fdt_pages = fdt_size / PageSize::Size4k as u64;
        let fdt_pages =
            page_alloc.take_pages(num_fdt_pages.try_into().unwrap(), T::TOP_LEVEL_ALIGN);

        let (zero_pages, vm) = HostVm::from_hyp_mem(page_alloc, guest_phys_size);

        // Now that the hypervisor is done claiming memory, determine the actual size of the host's
        // address space.
        let ram_size = zero_pages.len() as u64 * PageSize::Size4k as u64
            + fdt_pages.length_bytes()
            + kernel.size()
            + initramfs.map(|r| r.size()).unwrap_or(0);
        assert!(ram_size >= FDT_OFFSET + fdt_pages.length_bytes());

        Self {
            hypervisor_dt,
            kernel,
            initramfs,
            vm,
            fdt_pages: FdtPages::Clean(fdt_pages),
            zero_pages,
            guest_ram_base,
            ram_size,
        }
    }

    /// Builds a device tree for the host VM, flattening it to a range of pages that will be
    /// mapped into the address space in `build_address_space()`.
    pub fn build_device_tree(mut self) -> Self {
        let fdt_pages = match self.fdt_pages {
            FdtPages::Clean(pages) => pages,
            _ => panic!("Device tree already written"),
        };

        // Construct a stripped-down device-tree for the host VM.
        //
        // We map the IMSIC at the same location in the guest address space as it is in the
        // supervisor address space.
        let imsic_base = RawAddr::guest(Imsic::get().base_addr().bits(), PageOwnerId::host());
        let mut host_dt_builder = HostDtBuilder::new(&self.hypervisor_dt)
            .unwrap()
            .add_memory_node(self.guest_ram_base, self.ram_size)
            .unwrap()
            .add_cpu_nodes()
            .unwrap()
            .add_device_nodes(imsic_base)
            .unwrap();
        if let Some(r) = self.initramfs {
            host_dt_builder = host_dt_builder
                .set_initramfs_addr(
                    self.guest_ram_base
                        .checked_increment(INITRAMFS_OFFSET)
                        .unwrap(),
                    r.size(),
                )
                .unwrap();
        }
        let host_dt = host_dt_builder.tree();

        println!("Host DT: {}", host_dt);

        // Serialize the device-tree.
        let dt_writer = DeviceTreeSerializer::new(&host_dt);
        assert!(dt_writer.output_size() <= fdt_pages.length_bytes().try_into().unwrap());
        let fdt_slice = unsafe {
            // Safe because we own these pages.
            slice::from_raw_parts_mut(
                fdt_pages.base().bits() as *mut u8,
                fdt_pages.length_bytes().try_into().unwrap(),
            )
        };
        dt_writer.write_to(fdt_slice);
        let fdt_pages =
            SequentialPages::from_pages(fdt_pages.into_iter().map(|p| p.to_initialized_page()))
                .unwrap();
        self.fdt_pages = FdtPages::Initialized(fdt_pages);
        self
    }

    /// Constructs the address space for the host VM, returning a `HostVm` that is ready to run.
    pub fn build_address_space(mut self) -> HostVm<T, VmStateFinalized> {
        // Map the IMSIC interrupt files into the guest address space. The host VM's interrupt
        // file gets mapped to the location of the supervisor interrupt file.
        let imsic = Imsic::get();
        let cpu_info = CpuInfo::get();
        for i in 0..cpu_info.num_cpus() {
            let cpu_id = CpuId::new(i);
            let mut imsic_gpa = PageAddr::new(RawAddr::guest(
                imsic.supervisor_file_addr(cpu_id).unwrap().bits(),
                PageOwnerId::host(),
            ))
            .unwrap();

            // Add a confidential memory region covering this IMSIC.
            //
            // TODO: The IMSIC pages will eventually need special treatment and should probably be
            // in their own region.
            self.vm.add_confidential_memory_region(
                imsic_gpa,
                imsic.guests_per_hart() as u64 * PageSize::Size4k as u64,
            );

            let page = imsic.take_guest_file(cpu_id, ImsicGuestId::HostVm).unwrap();
            self.vm.add_pages(imsic_gpa, [page].into_iter());
            imsic_gpa = imsic_gpa.checked_add_pages(1).unwrap();
            for j in 0..(imsic.guests_per_hart() - 1) {
                let page = imsic
                    .take_guest_file(cpu_id, ImsicGuestId::GuestVm(j))
                    .unwrap();
                self.vm.add_pages(imsic_gpa, [page].into_iter());
                imsic_gpa = imsic_gpa.checked_add_pages(1).unwrap();
            }
        }

        // Host guarantees that the host pages it returns start at T::TOP_LEVEL_ALIGN-aligned block,
        // and because we built the HwMemMap with a minimum region alignment of T::TOP_LEVEL_ALIGN
        // any discontiguous ranges are also guaranteed to be aligned.
        //
        // Now fill in the address space, inserting zero pages around the kernel/initramfs/FDT.
        let mut current_gpa = PageAddr::new(self.guest_ram_base).unwrap();
        self.vm
            .add_confidential_memory_region(current_gpa, self.ram_size);

        let num_pages = KERNEL_OFFSET / PageSize::Size4k as u64;
        self.vm.add_pages(
            current_gpa,
            self.zero_pages.by_ref().take(num_pages.try_into().unwrap()),
        );
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let num_kernel_pages = self.kernel.size() / PageSize::Size4k as u64;
        let kernel_pages: SequentialPages<ConvertedInitialized> = unsafe {
            // Safe because HwMemMap reserved this region.
            SequentialPages::from_mem_range(self.kernel.base(), PageSize::Size4k, num_kernel_pages)
                .unwrap()
        };
        self.vm
            .add_measured_pages(current_gpa, kernel_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_kernel_pages).unwrap();

        if let Some(r) = self.initramfs {
            let num_pages =
                (INITRAMFS_OFFSET - (KERNEL_OFFSET + self.kernel.size())) / PageSize::Size4k as u64;
            self.vm.add_pages(
                current_gpa,
                self.zero_pages.by_ref().take(num_pages.try_into().unwrap()),
            );
            current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

            let num_initramfs_pages = r.size() / PageSize::Size4k as u64;
            let initramfs_pages: SequentialPages<ConvertedInitialized> = unsafe {
                // Safe because HwMemMap reserved this region.
                SequentialPages::from_mem_range(r.base(), PageSize::Size4k, num_initramfs_pages)
                    .unwrap()
            };
            self.vm
                .add_measured_pages(current_gpa, initramfs_pages.into_iter());
            current_gpa = current_gpa.checked_add_pages(num_initramfs_pages).unwrap();
        }

        let num_pages = (FDT_OFFSET - (current_gpa.bits() - self.guest_ram_base.bits()))
            / PageSize::Size4k as u64;
        self.vm.add_pages(
            current_gpa,
            self.zero_pages.by_ref().take(num_pages.try_into().unwrap()),
        );
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let fdt_pages = match self.fdt_pages {
            FdtPages::Initialized(pages) => pages,
            _ => panic!("FDT pages not initialized"),
        };
        let num_fdt_pages = fdt_pages.len();
        self.vm
            .add_measured_pages(current_gpa, fdt_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_fdt_pages).unwrap();

        self.vm.add_pages(current_gpa, self.zero_pages);
        self.vm.set_launch_args(
            self.guest_ram_base
                .checked_increment(KERNEL_OFFSET)
                .unwrap(),
            self.guest_ram_base.checked_increment(FDT_OFFSET).unwrap(),
        );
        self.vm.finalize()
    }
}
