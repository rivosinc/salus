// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(
    panic_info_message,
    allocator_api,
    alloc_error_handler,
    lang_items,
    asm_const,
    const_ptr_offset_from
)]

use core::alloc::{GlobalAlloc, Layout};

extern crate alloc;

mod abort;
mod asm;
mod ecall;
mod host_vm_loader;
mod print_util;
mod sha256_measure;
mod smp;
mod trap;
mod vm;
mod vm_cpu;
mod vm_pages;

use abort::abort;
use device_tree::{DeviceTree, Fdt};
use drivers::{CpuInfo, Imsic};
use host_vm_loader::HostVmLoader;
use hyp_alloc::HypAlloc;
use print_util::*;
use riscv_page_tables::*;
use riscv_pages::*;
use riscv_regs::{hedeleg, henvcfg, hideleg, hie, scounteren};
use riscv_regs::{Exception, Interrupt, LocalRegisterCopy, ReadWriteable, Writeable, CSR};
use smp::PerCpu;
use spin::Once;
use vm::HostVm;

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

/// The host VM that all CPUs enter at boot.
static HOST_VM: Once<HostVm<Sv48x4>> = Once::new();

/// Powers off this machine.
pub fn poweroff() -> ! {
    // Safety: on this platform, a write of 0x5555 to 0x100000 will trigger the platform to
    // poweroff, which is defined behavior.
    unsafe {
        core::ptr::write_volatile(0x10_0000 as *mut u32, 0x5555);
    }
    abort()
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
            builder = builder.add_memory_region(RawAddr::supervisor(r.base()), r.size())?;
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
        .map(|r| RawAddr::supervisor(r.base()))
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
            RawAddr::supervisor(r.base()),
            r.size(),
        )?;
    }

    // Reserve the host VM images loaded by firmware. We assume the start of these images are
    // aligned to make mapping them in easier.
    if let Some(r) = fdt.host_kernel_region() {
        assert_eq!(r.base() & (T::TOP_LEVEL_ALIGN - 1), 0);
        builder = builder.reserve_region(
            HwReservedMemType::HostKernelImage,
            RawAddr::supervisor(r.base()),
            r.size(),
        )?;
    }
    if let Some(r) = fdt.host_initramfs_region() {
        assert_eq!(r.base() & (T::TOP_LEVEL_ALIGN - 1), 0);
        builder = builder.reserve_region(
            HwReservedMemType::HostInitramfsImage,
            RawAddr::supervisor(r.base()),
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
            r.region_type()
        );
    }

    Ok(mem_map)
}

/// Creates a heapfrom the given `mem_map`, marking the region occupied by the heap as reserved.
fn create_heap(mem_map: &mut HwMemMap) -> HypAlloc {
    const HEAP_SIZE: u64 = 16 * 1024 * 1024;

    let heap_base = mem_map
        .regions()
        .find(|r| r.region_type() == HwMemRegionType::Available && r.size() >= HEAP_SIZE)
        .map(|r| r.base())
        .expect("Not enough free memory for hypervisor heap");
    mem_map
        .reserve_region(
            HwReservedMemType::HypervisorHeap,
            RawAddr::from(heap_base),
            HEAP_SIZE,
        )
        .unwrap();
    let pages = unsafe {
        // Safe since this region of memory was free in the memory map.
        SequentialPages::from_mem_range(
            heap_base.get_4k_addr(),
            HEAP_SIZE / PageSize::Size4k as u64,
        )
    };
    HypAlloc::from_pages(pages)
}

/// Initialize (H)S-level CSRs to a reasonable state.
pub fn setup_csrs() {
    // Clear and disable any interupts.
    CSR.sie.set(0);
    CSR.sip.set(0);
    // Turn FP and vector units off.
    CSR.sstatus.set(0);

    // Delegate traps to VS.
    let mut hedeleg = LocalRegisterCopy::<u64, hedeleg::Register>::new(0);
    hedeleg.modify(Exception::InstructionMisaligned.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::Breakpoint.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::UserEnvCall.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::InstructionPageFault.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::LoadPageFault.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::StorePageFault.to_hedeleg_field().unwrap());
    CSR.hedeleg.set(hedeleg.get());

    let mut hideleg = LocalRegisterCopy::<u64, hideleg::Register>::new(0);
    hideleg.modify(Interrupt::VirtualSupervisorSoft.to_hideleg_field().unwrap());
    hideleg.modify(
        Interrupt::VirtualSupervisorTimer
            .to_hideleg_field()
            .unwrap(),
    );
    hideleg.modify(
        Interrupt::VirtualSupervisorExternal
            .to_hideleg_field()
            .unwrap(),
    );
    CSR.hideleg.set(hideleg.get());

    let mut hie = LocalRegisterCopy::<u64, hie::Register>::new(0);
    hie.modify(Interrupt::VirtualSupervisorSoft.to_hie_field().unwrap());
    hie.modify(Interrupt::VirtualSupervisorTimer.to_hie_field().unwrap());
    hie.modify(Interrupt::VirtualSupervisorExternal.to_hie_field().unwrap());
    CSR.hie.set(hie.get());

    // Make counters available to guests.
    CSR.hcounteren.set(0xffff_ffff_ffff_ffff);

    // Make the basic counters available to any of our U-mode tasks.
    let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
    scounteren.modify(scounteren::cycle.val(1));
    scounteren.modify(scounteren::time.val(1));
    scounteren.modify(scounteren::instret.val(1));
    CSR.scounteren.set(scounteren.get());

    trap::install_trap_handler();
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    // Reset CSRs to a sane state.
    setup_csrs();

    // Safety: This is the very beginning of the kernel, there are no other users of the UART and
    // we expect that a UART is at this address.
    unsafe { UartDriver::init(RawAddr::supervisor(0x1000_0000)) };
    println!("Salus: Boot test VM");

    // Safe because we trust that the firmware passed a valid FDT.
    let hyp_fdt =
        unsafe { Fdt::new_from_raw_pointer(fdt_addr as *const u8) }.expect("Failed to read FDT");

    let mut mem_map = build_memory_map::<Sv48x4>(&hyp_fdt).expect("Failed to build memory map");
    // Find where QEMU loaded the host kernel image.
    let host_kernel = *mem_map
        .regions()
        .find(|r| r.region_type() == HwMemRegionType::Reserved(HwReservedMemType::HostKernelImage))
        .expect("No host kernel image");
    let host_initramfs = mem_map
        .regions()
        .find(|r| {
            r.region_type() == HwMemRegionType::Reserved(HwReservedMemType::HostInitramfsImage)
        })
        .cloned();

    let heap = create_heap(&mut mem_map);
    let hyp_dt = DeviceTree::from(&hyp_fdt, &heap).expect("Failed to construct device-tree");

    // Discover supported CPU extensions.
    CpuInfo::parse_from(&hyp_dt);
    let cpu_info = CpuInfo::get();
    if cpu_info.has_sstc() {
        println!("Sstc support present");
        // Only write henvcfg when Sstc is present to avoid blowing up on versions of QEMU which
        // don't support the *envcfg registers.
        CSR.henvcfg.modify(henvcfg::stce.val(1));
    }
    println!(
        "{} CPU(s) present. Booting on CPU{} (hart {})",
        cpu_info.num_cpus(),
        cpu_info
            .hart_id_to_cpu(hart_id.try_into().unwrap())
            .unwrap()
            .raw(),
        hart_id
    );

    // Probe for the IMSIC.
    Imsic::probe_from(&hyp_dt, &mut mem_map);
    let imsic = Imsic::get();
    println!(
        "IMSIC at 0x{:08x}; {} guest interrupt files supported",
        imsic.base_addr().bits(),
        imsic.guests_per_hart()
    );
    Imsic::setup_this_cpu();

    // Set up per-CPU memory and boot the secondary CPUs.
    PerCpu::init(hart_id, &mut mem_map);
    smp::start_secondary_cpus();

    // We start RAM in the host address space at the same location as it is in the supervisor
    // address space.
    let guest_ram_base = mem_map
        .regions()
        .find(|r| !matches!(r.region_type(), HwMemRegionType::Mmio(_)))
        .map(|r| RawAddr::guest(r.base().bits(), PageOwnerId::host()))
        .unwrap();
    let guest_phys_size = mem_map.regions().last().unwrap().end().bits() - guest_ram_base.bits();

    // Create an allocator for the remaining pages. Anything that's left over will be mapped
    // into the host VM.
    let hyp_mem = HypPageAlloc::new(mem_map, &heap);

    // Now load the host VM.
    let host = HostVmLoader::new(
        hyp_dt,
        host_kernel,
        host_initramfs,
        guest_ram_base,
        guest_phys_size,
        hyp_mem,
    )
    .build_device_tree()
    .build_address_space();
    HOST_VM.call_once(|| host);

    let cpu_id = PerCpu::this_cpu().cpu_id();
    HOST_VM.get().unwrap().run(cpu_id.raw() as u64);
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {
    setup_csrs();
    let cpu_info = CpuInfo::get();
    if cpu_info.has_sstc() {
        CSR.henvcfg.modify(henvcfg::stce.val(1));
    }
    Imsic::setup_this_cpu();

    let me = PerCpu::this_cpu();
    me.set_online();

    HOST_VM.wait().run(me.cpu_id().raw() as u64);
}
