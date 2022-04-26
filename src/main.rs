// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{Allocator, GlobalAlloc, Layout};
use core::slice;

extern crate alloc;

mod abort;
mod asm;
mod data_measure;
mod host_dt_builder;
mod print_util;
mod test_measure;
mod vm;
mod vm_pages;

use abort::abort;
use data_measure::DataMeasure;
use device_tree::{DeviceTree, DeviceTreeSerializer, Fdt};
use host_dt_builder::HostDtBuilder;
use hyp_alloc::HypAlloc;
use page_collections::page_vec::PageVec;
use print_util::*;
use riscv_page_tables::*;
use riscv_pages::*;
use test_measure::TestMeasure;
use vm::Host;
use vm_pages::HostRootBuilder;

extern "C" {
    static _start: u8;
    static _stack_end: u8;
}

// Dummy global allocator - panic if anything tries to do an allocation.
struct GeneralGlobalAlloc;

unsafe impl GlobalAlloc for GeneralGlobalAlloc {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        abort()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        abort()
    }
}

#[global_allocator]
static GENERAL_ALLOCATOR: GeneralGlobalAlloc = GeneralGlobalAlloc;

#[alloc_error_handler]
pub fn alloc_error(_layout: Layout) -> ! {
    abort()
}

/// Powers off this machine.
pub fn poweroff() -> ! {
    // Safety: on this platform, a write of 0x5555 to 0x100000 will trigger the platform to
    // poweroff, which is defined behavior.
    unsafe {
        core::ptr::write_volatile(0x10_0000 as *mut u32, 0x5555);
    }
    abort()
}

/// A flattened Page iterator over a vector of PageRanges. Used for filling in host address space.
struct HostPagesIter {
    ranges: PageVec<PageRange>,
    index: usize,
}

impl HostPagesIter {
    fn new(ranges: PageVec<PageRange>) -> Self {
        Self { ranges, index: 0 }
    }
}

impl Iterator for HostPagesIter {
    type Item = Page4k;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.ranges.len() && self.ranges[self.index].remaining_size() == 0 {
            self.index += 1;
        }
        if self.index >= self.ranges.len() {
            return None;
        }
        self.ranges[self.index].next()
    }
}

/// Builds the hardware memory map from the device-tree. The kernel & initramfs image regions are
/// aligned to `T::TOP_LEVEL_ALIGN` so that they can be mapped directly into the host VM's guest
/// physical address space.
fn build_memory_map<T: PlatformPageTable>(fdt: &Fdt) -> MemMapResult<HwMemMap> {
    let mut builder = HwMemMapBuilder::new(T::TOP_LEVEL_ALIGN);

    // First add the memory regions.
    for r in fdt.memory_regions() {
        // Safety: We own all of memory at this point and we trust the FDT is well-formed.
        unsafe {
            builder = builder.add_memory_region(PhysAddr::new(r.base()), r.size())?;
        }
    }

    // Reserve the region used by the hypervisor image itself, including the stack and FDT
    // passed in by firmware.

    // Safe because we trust the linker placed these symbols correctly.
    let start = unsafe { core::ptr::addr_of!(_start) as u64 };
    let stack_end = unsafe { core::ptr::addr_of!(_stack_end) as u64 };
    // FDT must be after the hypervisor image.
    let fdt_start = fdt.base_addr() as u64;
    assert!(stack_end <= fdt_start);
    let fdt_end = fdt_start
        .checked_add(fdt.size().try_into().unwrap())
        .unwrap();

    // Find the region of DRAM that the hypervisor is in.
    let resv_base = fdt
        .memory_regions()
        .find(|r| start >= r.base() && fdt_end <= r.base().checked_add(r.size()).unwrap())
        .map(|r| PhysAddr::new(r.base()))
        .expect("Hypervisor image does not reside in a contiguous range of DRAM");

    // Reserve everything from the start of the region the hypervisor is in up until the end
    // of the FDT.
    builder = builder.reserve_region(
        HwReservedMemType::HypervisorImage,
        resv_base,
        fdt_end - resv_base.bits(),
    )?;

    // Reserve the regions marked reserved by firmware.
    for r in fdt.reserved_memory_regions() {
        builder = builder.reserve_region(
            HwReservedMemType::FirmwareReserved,
            PhysAddr::new(r.base()),
            r.size(),
        )?;
    }

    // Reserve the host VM images loaded by firmware. We assume the start of these images are
    // aligned to make mapping them in easier.
    if let Some(r) = fdt.host_kernel_region() {
        assert_eq!(r.base() & (T::TOP_LEVEL_ALIGN - 1), 0);
        builder = builder.reserve_region(
            HwReservedMemType::HostKernelImage,
            PhysAddr::new(r.base()),
            r.size(),
        )?;
    }
    if let Some(r) = fdt.host_initramfs_region() {
        assert_eq!(r.base() & (T::TOP_LEVEL_ALIGN - 1), 0);
        builder = builder.reserve_region(
            HwReservedMemType::HostInitramfsImage,
            PhysAddr::new(r.base()),
            r.size(),
        )?;
    }
    let mem_map = builder.build();

    println!("HW memory map:");
    for (i, r) in mem_map.regions().enumerate() {
        println!(
            "[{}] region: 0x{:x} -> 0x{:x}, {}",
            i,
            r.base().bits(),
            r.end().bits() - 1,
            r.mem_type()
        );
    }

    Ok(mem_map)
}

/// Creates a heapfrom the given `mem_map`, marking the region occupied by the heap as reserved.
fn create_heap(mem_map: &mut HwMemMap) -> HypAlloc {
    const HEAP_SIZE: u64 = 16 * 1024 * 1024;

    let heap_base = mem_map
        .regions()
        .find(|r| r.mem_type() == HwMemType::Available && r.size() >= HEAP_SIZE)
        .map(|r| r.base())
        .expect("Not enough free memory for hypervisor heap");
    mem_map
        .reserve_region(
            HwReservedMemType::HypervisorHeap,
            PhysAddr::from(heap_base),
            HEAP_SIZE,
        )
        .unwrap();
    let pages = unsafe {
        // Safe since this region of memory was free in the memory map.
        SequentialPages::from_mem_range(heap_base, HEAP_SIZE / PageSize4k::SIZE_BYTES)
    };
    HypAlloc::from_pages(pages)
}

/// Loads a host VM with the given kernel & initramfs images, passing a patched version of the
/// hypervisor device-tree to the host kernel. Uses `hyp_pages` to allocate any other
/// hypervisor-internal structures, consuming the rest to map into the host VM.
///
/// In order to allow the host VM to allocate physically-aligned blocks necessary for guest VM
/// creation (specifically, the root of the G-stage page-table), we guarantee that each
/// contiguous T::TOP_LEVEL_ALIGN block of the guest physical address space of the host VM maps to
/// a contiguous T::TOP_LEVEL_ALIGN block of the host physical address space.
fn load_host_vm<T, D, A>(
    hyp_dt: DeviceTree<A>,
    host_kernel: HwMemRegion,
    host_initramfs: Option<HwMemRegion>,
    mut hyp_pages: HypPageAlloc,
) -> Host<T, D>
where
    T: PlatformPageTable,
    D: DataMeasure,
    A: Allocator + Clone,
{
    // Reserve pages for tracking the host's guests.
    let host_guests_pages = match SequentialPages::from_pages(hyp_pages.take_pages(2)) {
        Ok(sp) => sp,
        _ => unreachable!(),
    };

    // Reserve a contiguous chunk for the host's FDT. We assume it will be no bigger than the size
    // of the hypervisor's FDT and we align it to `T::TOP_LEVEL_ALIGN` to maintain the contiguous
    // mapping guarantee from GPA -> HPA mentioned above.
    let host_fdt_size = {
        let size = DeviceTreeSerializer::new(&hyp_dt).output_size();
        ((size as u64) + T::TOP_LEVEL_ALIGN - 1) & !(T::TOP_LEVEL_ALIGN - 1)
    };
    let num_fdt_pages = host_fdt_size / PageSize4k::SIZE_BYTES;
    let host_fdt_pages = match SequentialPages::from_pages(
        hyp_pages.take_pages_with_alignment(num_fdt_pages.try_into().unwrap(), T::TOP_LEVEL_ALIGN),
    ) {
        Ok(sp) => sp,
        _ => unreachable!(),
    };

    // We use the size of our (the hypervisor's) physical address to estimate the size of the
    // host's guest phsyical address space since we build the host's address space to match the
    // actual physical address space, but with the holes (for hypervisor memory, other reserved
    // regions) removed. This results in a bit of an overestimate for determining the number of
    // page-table pages, but we should expect the holes to be pretty small.
    //
    // TODO: Support discontiguous physical memory.
    let (phys_mem_base, phys_mem_size) = {
        let node = hyp_dt
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

    let (host_pages, mut host_root_builder) =
        HostRootBuilder::<T, D>::from_hyp_mem(hyp_pages, phys_mem_size);

    // Now that the hypervisor is done claiming memory, determine the actual size of the host's
    // address space.
    let host_ram_size = host_pages.iter().fold(0, |acc, r| acc + r.remaining_size())
        + host_fdt_pages.length_bytes()
        + host_kernel.size()
        + host_initramfs.map(|r| r.size()).unwrap_or(0);
    let host_ram_base = phys_mem_base;

    // Where the kernel, initramfs, and FDT will be located in the guest physical address space.
    //
    // TODO: Kernel offset should be pulled from the header in the kernel image.
    const KERNEL_OFFSET: u64 = 0x20_0000;
    const INITRAMFS_OFFSET: u64 = KERNEL_OFFSET + 0x800_0000;
    // Assuming RAM base at 2GB, ends up at 3GB - 16MB which is consistent with QEMU.
    const FDT_OFFSET: u64 = 0x3f00_0000;
    assert!(host_ram_size >= FDT_OFFSET + host_fdt_pages.length_bytes());

    // Construct a stripped-down device-tree for the host VM.
    let mut host_dt_builder = HostDtBuilder::new(&hyp_dt)
        .unwrap()
        .add_memory_node(host_ram_base, host_ram_size)
        .unwrap();
    if let Some(r) = host_initramfs {
        host_dt_builder = host_dt_builder
            .set_initramfs_addr(
                host_ram_base.checked_add(INITRAMFS_OFFSET).unwrap(),
                r.size(),
            )
            .unwrap();
    }

    // TODO: Add IMSIC & PCIe nodes.
    let host_dt = host_dt_builder.tree();

    println!("Host DT: {}", host_dt);

    // Serialize the device-tree.
    let dt_writer = DeviceTreeSerializer::new(&host_dt);
    assert!(dt_writer.output_size() <= host_fdt_pages.length_bytes().try_into().unwrap());
    let host_fdt_slice = unsafe {
        // Safe because we own these pages.
        slice::from_raw_parts_mut(
            host_fdt_pages.base() as *mut u8,
            host_fdt_pages.length_bytes().try_into().unwrap(),
        )
    };
    dt_writer.write_to(host_fdt_slice);

    // HostRootBuilder guarantees that the host pages it returns start at T::TOP_LEVEL_ALIGN-aligned
    // block, and because we built the HwMemMap with a minimum region alignment of T::TOP_LEVEL_ALIGN
    // any discontiguous ranges are also guaranteed to be aligned.
    let mut host_pages_iter = HostPagesIter::new(host_pages);

    // Now fill in the address space, inserting zero pages around the kernel/initramfs/FDT.
    let mut current_gpa = AlignedPageAddr4k::new(PhysAddr::new(host_ram_base)).unwrap();
    let num_pages = KERNEL_OFFSET / PageSize4k::SIZE_BYTES;
    host_root_builder = host_root_builder.add_4k_pages(
        current_gpa,
        host_pages_iter.by_ref().take(num_pages.try_into().unwrap()),
    );
    current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

    let num_kernel_pages = host_kernel.size() / PageSize4k::SIZE_BYTES;
    let kernel_pages = unsafe {
        // Safe because HwMemMap reserved this region.
        SequentialPages::from_mem_range(host_kernel.base(), num_kernel_pages)
    };
    host_root_builder = host_root_builder.add_4k_data_pages(current_gpa, kernel_pages.into_iter());
    current_gpa = current_gpa.checked_add_pages(num_kernel_pages).unwrap();

    if let Some(r) = host_initramfs {
        let num_pages =
            (INITRAMFS_OFFSET - (KERNEL_OFFSET + host_kernel.size())) / PageSize4k::SIZE_BYTES;
        host_root_builder = host_root_builder.add_4k_pages(
            current_gpa,
            host_pages_iter.by_ref().take(num_pages.try_into().unwrap()),
        );
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let num_initramfs_pages = r.size() / PageSize4k::SIZE_BYTES;
        let initramfs_pages = unsafe {
            // Safe because HwMemMap reserved this region.
            SequentialPages::from_mem_range(r.base(), num_initramfs_pages)
        };
        host_root_builder =
            host_root_builder.add_4k_data_pages(current_gpa, initramfs_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_initramfs_pages).unwrap();
    }

    let num_pages = (FDT_OFFSET - (current_gpa.bits() - host_ram_base)) / PageSize4k::SIZE_BYTES;
    host_root_builder = host_root_builder.add_4k_pages(
        current_gpa,
        host_pages_iter.by_ref().take(num_pages.try_into().unwrap()),
    );
    current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

    host_root_builder =
        host_root_builder.add_4k_data_pages(current_gpa, host_fdt_pages.into_iter());
    current_gpa = current_gpa.checked_add_pages(num_fdt_pages).unwrap();

    host_root_builder = host_root_builder.add_4k_pages(current_gpa, host_pages_iter);
    let mut host = Host::new(host_root_builder.create_host(), host_guests_pages);
    host.add_device_tree(host_ram_base + FDT_OFFSET);
    host
}

// Basic configuration of and running the test VM.
fn test_boot_vm<T: PlatformPageTable, D: DataMeasure>(hart_id: u64, fdt_addr: u64) {
    // Safe because we trust that the firmware passed a valid FDT.
    let hyp_fdt =
        unsafe { Fdt::new_from_raw_pointer(fdt_addr as *const u8) }.expect("Failed to read FDT");

    let mut mem_map = build_memory_map::<T>(&hyp_fdt).expect("Failed to build memory map");
    // Find where QEMU loaded the host kernel image.
    let host_kernel = *mem_map
        .regions()
        .find(|r| r.mem_type() == HwMemType::Reserved(HwReservedMemType::HostKernelImage))
        .expect("No host kernel image");
    let host_initramfs = mem_map
        .regions()
        .find(|r| r.mem_type() == HwMemType::Reserved(HwReservedMemType::HostInitramfsImage))
        .cloned();

    let heap = create_heap(&mut mem_map);
    let hyp_dt = DeviceTree::from(&hyp_fdt, &heap).expect("Failed to construct device-tree");

    // Create an allocator for the remaining pages. Anything that's left over will be mapped
    // into the host VM.
    let hyp_mem = HypPageAlloc::new(mem_map);

    // Now build the host VM's address space.
    let mut host = load_host_vm::<T, D, _>(hyp_dt, host_kernel, host_initramfs, hyp_mem);

    // TODO return host and let main run it.
    let _ = host.run(hart_id);
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    if hart_id != 0 {
        // TODO handle more than 1 cpu
        abort();
    }

    // Safety: This is the very beginning of the kernel, there are no other users of the UART or the
    // CONSOLE_DRIVER global.
    unsafe { UartDriver::new(0x1000_0000).use_as_console() }
    println!("Salus: Boot test VM");

    test_boot_vm::<Sv48x4, TestMeasure>(hart_id, fdt_addr);

    println!("Salus: Host exited");

    poweroff();
}
