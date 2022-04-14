// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};
use core::slice;

extern crate alloc;

mod abort;
mod asm;
mod data_measure;
mod print_util;
mod test_measure;
mod vm;
mod vm_pages;

use abort::abort;
use data_measure::DataMeasure;
use device_tree::Fdt;
use print_util::*;
use riscv_page_tables::*;
use riscv_pages::*;
use test_measure::TestMeasure;
use vm::Host;
use vm_pages::{HostRootBuilder, HostRootStateEmpty};

const RAM_BASE: u64 = 0x8000_0000;

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

/// Builds the hardware memory map from the device-tree, with a minimum region alignment of
/// `T::TOP_LEVEL_ALIGN`.
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

/// Adds a device tree to host memory at the host offset given.
/// Returns the size of the host's device tree.
/// # Safety:
///     All memory past the end of hypervisor memory must be unused and available for writing. (this
///     will all be assigned to the host).
///     `host_dt_addr` must poing to memory that is safe for writing `host_ram_size` bytes to.
unsafe fn pass_device_tree(hyp_fdt: &Fdt, host_dt_addr: u64, host_ram_size: u64) -> u64 {
    // Create a slice of the fdt passed from firmware
    let dt_size = hyp_fdt.size();

    // Update memory size - TODO - other modifications
    let host_slice = slice::from_raw_parts_mut(host_dt_addr as *mut u8, dt_size);
    hyp_fdt.write_with_updated_memory_size(host_slice, host_ram_size);

    dt_size as u64
}

// Basic configuration of and running the test VM.
fn test_boot_vm<T: PlatformPageTable, D: DataMeasure>(hart_id: u64, fdt_addr: u64) {
    // put the host DT somewhere the host can read it.
    const HOST_DT_OFFSET: u64 = 0x220_0000;

    // Safe because we trust that the firmware passed a valid FDT.
    let hyp_fdt =
        unsafe { Fdt::new_from_raw_pointer(fdt_addr as *const u8) }.expect("Failed to read FDT");

    let mem_map = build_memory_map::<T>(&hyp_fdt).expect("Failed to build memory map");
    // Find where QEMU loaded the host kernel image.
    let host_kernel = *mem_map
        .regions()
        .find(|r| r.mem_type() == HwMemType::Reserved(HwReservedMemType::HostKernelImage))
        .expect("No host kernel image");
    // Where the host VM's physical address space starts.
    let host_start_page = mem_map.regions().nth(0).unwrap().base();

    let mut hyp_mem = HypPageAlloc::new(mem_map);
    let host_guests_pages =
        match SequentialPages::<PageSize4k>::from_pages(hyp_mem.by_ref().take(2)) {
            Ok(s) => s,
            Err(_) => unreachable!(),
        };

    let (mut host_pages, host_root_builder) =
        HostRootBuilder::<T, D, HostRootStateEmpty>::from_hyp_mem(hyp_mem);

    let host_base = host_pages.next_addr().bits();
    let host_size = host_pages.remaining_size();

    // Not safe! Although we trust the FDT correctly specified where the host VM kernel is,
    // there's no guarantee that it doesn't overlap with hypervisor memory and that we haven't
    // already trampled over it.
    //
    // TODO: Sanity-check the kernel region and reserve it from hypervisor use from the start.
    let dt_len = unsafe {
        core::ptr::copy(
            host_kernel.base().bits() as *const u8,
            (host_base + 0x20_0000) as *mut u8,
            host_kernel.size().try_into().unwrap(),
        );
        // zero out the data from qemu now that it's been copied to the destination pages.
        core::ptr::write_bytes(
            host_kernel.base().bits() as *mut u8,
            0,
            host_kernel.size().try_into().unwrap(),
        );

        pass_device_tree(&hyp_fdt, host_base + HOST_DT_OFFSET, host_size)
    };

    let data_page_count = (HOST_DT_OFFSET + dt_len) / PageSize4k::SIZE_BYTES + 1;
    let first_zero_addr = host_start_page.checked_add_pages(data_page_count).unwrap();

    let host_root_pages = host_root_builder
        .add_4k_data_pages(
            host_start_page,
            host_pages.by_ref().take(data_page_count as usize),
        )
        .add_4k_pages(first_zero_addr, host_pages)
        .create_host();

    let mut host = Host::new(host_root_pages, host_guests_pages);
    host.add_device_tree(RAM_BASE + HOST_DT_OFFSET);

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
