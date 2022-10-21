// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! A small Risc-V hypervisor to enable trusted execution environments.

#![no_main]
#![no_std]
#![feature(
    panic_info_message,
    allocator_api,
    alloc_error_handler,
    lang_items,
    if_let_guard,
    asm_const,
    ptr_sub_ptr,
    slice_ptr_get,
    let_chains,
    is_some_and
)]

use core::alloc::{Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;

extern crate alloc;

mod asm;
mod guest_tracking;
mod host_vm_loader;
mod smp;
mod trap;
mod vm;
mod vm_cpu;
mod vm_id;
mod vm_pages;
mod vm_pmu;

use device_tree::{DeviceTree, Fdt};
use drivers::{imsic::Imsic, iommu::Iommu, pci::PcieRoot, pmu::PmuInfo, uart::UartDriver, CpuInfo};
use host_vm_loader::HostVmLoader;
use hyp_alloc::HypAlloc;
use page_tracking::*;
use riscv_page_tables::*;
use riscv_pages::*;
use riscv_regs::{hedeleg, henvcfg, hideleg, hie, satp, scounteren};
use riscv_regs::{sstatus, vlenb, Readable, RiscvCsrInterface, MAX_VECTOR_REGISTER_LEN};
use riscv_regs::{
    Exception, Interrupt, LocalRegisterCopy, ReadWriteable, SatpHelpers, Writeable, CSR, CSR_CYCLE,
    CSR_TIME,
};
use s_mode_utils::abort::abort;
use s_mode_utils::print::*;
use s_mode_utils::sbi_console::SbiConsoleV01;
use smp::PerCpu;
use spin::Once;
use vm::HostVm;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    abort();
}

extern "C" {
    static _start: u8;
    static _stack_end: u8;
}

/// The allocator used for boot-time dynamic memory allocations.
static HYPERVISOR_ALLOCATOR: Once<HypAlloc> = Once::new();

/// The hypervisor page table root address and mode to load in satp on secondary CPUs
static SATP_VAL: Once<u64> = Once::new();

// Implementation of GlobalAlloc that forwards allocations to the boot-time allocator.
struct GeneralGlobalAlloc;

unsafe impl GlobalAlloc for GeneralGlobalAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HYPERVISOR_ALLOCATOR
            .get()
            .and_then(|a| a.allocate(layout).ok())
            .map(|p| p.as_mut_ptr())
            .unwrap_or(core::ptr::null_mut())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Unwrap ok, there must've been an allocator to allocate the pointer in the first place.
        HYPERVISOR_ALLOCATOR
            .get()
            .unwrap()
            .deallocate(NonNull::new(ptr).unwrap(), layout);
    }
}

#[global_allocator]
static GENERAL_ALLOCATOR: GeneralGlobalAlloc = GeneralGlobalAlloc;

/// Aborts if the system hits an allocation error.
#[alloc_error_handler]
pub fn alloc_error(_layout: Layout) -> ! {
    abort()
}

// Powers off this machine.
fn poweroff() -> ! {
    println!("Shutting down");
    // Safety: on this platform, a write of 0x5555 to 0x100000 will trigger the platform to
    // poweroff, which is defined behavior.
    unsafe {
        core::ptr::write_volatile(0x10_0000 as *mut u32, 0x5555);
    }
    abort()
}

/// The host VM that all CPUs enter at boot.
static HOST_VM: Once<HostVm<Sv48x4>> = Once::new();

/// Builds the hardware memory map from the device-tree. The kernel & initramfs image regions are
/// aligned to `T::TOP_LEVEL_ALIGN` so that they can be mapped directly into the host VM's guest
/// physical address space.
fn build_memory_map<T: GuestStagePagingMode>(fdt: &Fdt) -> MemMapResult<HwMemMap> {
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

    // Find the region of DRAM that the hypervisor is in.
    let resv_base = fdt
        .memory_regions()
        .find(|r| start >= r.base() && stack_end <= r.base().checked_add(r.size()).unwrap())
        .map(|r| RawAddr::supervisor(r.base()))
        .expect("Hypervisor image does not reside in a contiguous range of DRAM");

    // Reserve everything from the start of the region the hypervisor is in up until the top of
    // the hypervisor stack.
    builder = builder.reserve_region(
        HwReservedMemType::HypervisorImage,
        resv_base,
        stack_end - resv_base.bits(),
    )?;

    // FDT must be after the hypervisor image.
    let fdt_start = fdt.base_addr() as u64;
    assert!(stack_end <= fdt_start);
    builder = builder.reserve_region(
        HwReservedMemType::HypervisorImage,
        RawAddr::supervisor(fdt_start),
        fdt.size() as u64,
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
    Ok(builder.build())
}

// Returns the number of PTE pages needed to map all regions in the given memory map.
// Slightly overestimates of number of pages needed as some regions will share PTE pages in reality.
fn pte_page_count(mem_map: &HwMemMap) -> u64 {
    mem_map.regions().fold(0, |acc, r| {
        acc + Sv48::max_pte_pages(r.size() / PageSize::Size4k as u64)
    })
}

// Returns the base address of the first available region in the memory map that is at least `size`
// bytes long. Returns None if no region is big enough.
fn find_available_region(mem_map: &HwMemMap, size: u64) -> Option<SupervisorPageAddr> {
    mem_map
        .regions()
        .find(|r| r.region_type() == HwMemRegionType::Available && r.size() >= size)
        .map(|r| r.base())
}

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
        | HwMemRegionType::Reserved(HwReservedMemType::HypervisorPtes)
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
    let region_page_count = PageSize::Size4k.round_up(size) / PageSize::Size4k as u64;
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
            mapper.map_4k_addr(virt, phys, pte_fields).unwrap();
        }
    }
}

// Creates the Sv48 page table based on the accessible regions of memory in the provided memory
// map.
fn setup_hyp_paging(mem_map: &mut HwMemMap) {
    let num_pte_pages = pte_page_count(mem_map);
    let pte_base = find_available_region(mem_map, num_pte_pages * PageSize::Size4k as u64)
        .expect("Not enough free memory for hypervisor Sv48 page table");
    let mut pte_pages = mem_map
        .reserve_and_take_pages(
            HwReservedMemType::HypervisorPtes,
            SupervisorPageAddr::new(RawAddr::from(pte_base)).unwrap(),
            PageSize::Size4k,
            num_pte_pages,
        )
        .unwrap()
        .clean()
        .into_iter();
    // Create empty sv48 page table
    let root_page = pte_pages.next().unwrap();
    let sv48: FirstStagePageTable<Sv48> =
        FirstStagePageTable::new(root_page).expect("creating sv48");

    // Map all the regions in the memory map that the hypervisor could need.
    for (base, size, perms) in mem_map.regions().filter_map(hyp_map_params) {
        hyp_map_region(&sv48, base, size, perms, &mut || pte_pages.next());
    }

    // TODO - reset device is hard coded in vm.rs
    map_fixed_device(0x10_0000, &sv48, &mut || pte_pages.next());

    // Install the page table in satp
    let mut satp = LocalRegisterCopy::<u64, satp::Register>::new(0);
    satp.set_from(&sv48, 0);
    // Store the SATP value for other CPUs. They load from the global in start_secondary.
    SATP_VAL.call_once(|| satp.get());
    CSR.satp.set(satp.get());
    tlb::sfence_vma(None, None);
}

// Adds some hard-coded device location to the given sv48 page table so that the devices can be
// accessed by the hypervisor. Identity maps a single page at base to base.
fn map_fixed_device(
    base: u64,
    sv48: &FirstStagePageTable<Sv48>,
    get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
) {
    let virt_base = PageAddr::new(RawAddr::supervisor_virt(base)).unwrap();
    let phys_base = PageAddr::new(RawAddr::supervisor(base)).unwrap();
    let pte_fields = PteFieldBits::leaf_with_perms(PteLeafPerms::RW);
    let mapper = sv48
        .map_range(virt_base, PageSize::Size4k, 1, get_pte_page)
        .unwrap();
    // Safe to map access to the device because this will be the only mapping it is used through.
    unsafe {
        mapper
            .map_4k_addr(virt_base, phys_base, pte_fields)
            .unwrap();
    }
}

/// Creates a heap from the given `mem_map`, marking the region occupied by the heap as reserved.
fn create_heap(mem_map: &mut HwMemMap) {
    const HEAP_SIZE: u64 = 16 * 1024 * 1024;

    let heap_base = find_available_region(mem_map, HEAP_SIZE)
        .expect("Not enough free memory for hypervisor heap");
    mem_map
        .reserve_region(
            HwReservedMemType::HypervisorHeap,
            RawAddr::from(heap_base),
            HEAP_SIZE,
        )
        .unwrap();
    let pages: SequentialPages<InternalDirty> = unsafe {
        // Safe since this region of memory was free in the memory map.
        SequentialPages::from_mem_range(
            heap_base,
            PageSize::Size4k,
            HEAP_SIZE / PageSize::Size4k as u64,
        )
        .unwrap()
    };
    HYPERVISOR_ALLOCATOR.call_once(|| HypAlloc::from_pages(pages.clean()));
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
    hedeleg.modify(Exception::IllegalInstruction.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::Breakpoint.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::LoadMisaligned.to_hedeleg_field().unwrap());
    hedeleg.modify(Exception::StoreMisaligned.to_hedeleg_field().unwrap());
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

    // TODO: Handle virtualization of timer/htimedelta (see issue #46)
    // Enable access to timer for now.
    CSR.hcounteren.set(1 << (CSR_TIME - CSR_CYCLE));

    // Make the basic counters available to any of our U-mode tasks.
    let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
    scounteren.modify(scounteren::cycle.val(1));
    scounteren.modify(scounteren::time.val(1));
    scounteren.modify(scounteren::instret.val(1));
    CSR.scounteren.set(scounteren.get());

    trap::install_trap_handler();
}

fn check_vector_width() {
    // Because we just ran setup_csrs(), we know vectors are off
    // Turn vectors on
    CSR.sstatus.read_and_set_bits(sstatus::vs::Initial.value);

    // vlenb converted from bytes to bits
    let rwidth = CSR.vlenb.read(vlenb::value);
    println!("vector register width: {} bits", rwidth * 8);
    if rwidth > MAX_VECTOR_REGISTER_LEN as u64 {
        println!(
            "Vector registers too wide: {} bits, maximum is {} bits",
            rwidth * 8,
            MAX_VECTOR_REGISTER_LEN * 8
        );
        panic!("Aborting boot.");
    }
    // Turn vectors off
    CSR.sstatus.read_and_clear_bits(sstatus::vs::Dirty.value);
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    // Reset CSRs to a sane state.
    setup_csrs();

    SbiConsoleV01::set_as_console();
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

    // Create a heap for boot-time memory allocations.
    create_heap(&mut mem_map);

    let hyp_dt = DeviceTree::from(&hyp_fdt).expect("Failed to construct device-tree");

    // Find the UART and switch to it as the system console.
    UartDriver::probe_from(&hyp_dt, &mut mem_map).expect("Failed to probe UART");

    // Discover the CPU topology.
    CpuInfo::parse_from(&hyp_dt);
    let cpu_info = CpuInfo::get();
    if cpu_info.has_sstc() {
        println!("Sstc support present");
        // Only write henvcfg when Sstc is present to avoid blowing up on versions of QEMU which
        // don't support the *envcfg registers.
        CSR.henvcfg.modify(henvcfg::stce.val(1));
    }
    if cpu_info.has_sscofpmf() {
        // Only probe for PMU counters if we have Sscofpmf; we can't expose counters to guests
        // unless we have support for per-mode filtering.
        println!("Sscofpmf support present");
        if let Err(e) = PmuInfo::init() {
            println!("PmuInfo::init() failed with {:?}", e);
        }
    }
    if cpu_info.has_vector() {
        // Will panic if register width too long. (currently 256 bits)
        check_vector_width();
    } else {
        println!("No vector support");
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
    Imsic::probe_from(&hyp_dt, &mut mem_map).expect("Failed to probe IMSIC");
    let imsic_geometry = Imsic::get().phys_geometry();
    println!(
        "IMSIC at 0x{:08x}; {} guest interrupt files supported",
        imsic_geometry.base_addr().bits(),
        imsic_geometry.guests_per_hart()
    );
    Imsic::setup_this_cpu();

    // Probe for a PCI bus.
    PcieRoot::probe_from(&hyp_dt, &mut mem_map).expect("Failed to set up PCIe");
    let pci = PcieRoot::get();
    for dev in pci.devices() {
        let dev = dev.lock();
        println!(
            "Found func {}; type: {}, MSI: {}, MSI-X: {}, PCIe: {}",
            dev.info(),
            dev.info().header_type(),
            dev.has_msi(),
            dev.has_msix(),
            dev.is_pcie(),
        );
        for bar in dev.bar_info().bars() {
            println!(
                "BAR{:}: type {:?}, size 0x{:x}",
                bar.index(),
                bar.bar_type(),
                bar.size()
            );
        }
    }

    setup_hyp_paging(&mut mem_map);

    // Set up per-CPU memory and boot the secondary CPUs.
    PerCpu::init(hart_id, &mut mem_map);

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

    // We start RAM in the host address space at the same location as it is in the supervisor
    // address space.
    //
    // Unwrap ok here and below since we must have at least one RAM region.
    let guest_ram_base = mem_map
        .regions()
        .find(|r| !matches!(r.region_type(), HwMemRegionType::Mmio(_)))
        .map(|r| RawAddr::guest(r.base().bits(), PageOwnerId::host()))
        .unwrap();
    // For the purposes of calculating the total size of the host VM's guest physical address
    // space, use the start and end of the (real) physical memory map. This is a bit of an
    // over-estimate given that not all of the physical memory map ends up getting mapped into
    // the host VM.
    let guest_phys_size = mem_map.regions().last().unwrap().end().bits()
        - mem_map.regions().next().unwrap().base().bits();

    // Create an allocator for the remaining pages. Anything that's left over will be mapped
    // into the host VM.
    let mut hyp_mem = HypPageAlloc::new(mem_map);

    // Find and initialize the IOMMU.
    match Iommu::probe_from(PcieRoot::get(), &mut || {
        hyp_mem.take_pages_for_host_state(1).into_iter().next()
    }) {
        Ok(_) => {
            println!(
                "Found RISC-V IOMMU version 0x{:x}",
                Iommu::get().unwrap().version()
            );
        }
        Err(e) => {
            println!("Failed to probe IOMMU: {:?}", e);
        }
    };

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

    // Lock down the boot time allocator before allowing the host VM to be entered.
    HYPERVISOR_ALLOCATOR.get().unwrap().seal();

    smp::start_secondary_cpus();

    HOST_VM.call_once(|| host);
    let cpu_id = PerCpu::this_cpu().cpu_id();
    HOST_VM.get().unwrap().run(cpu_id.raw() as u64);
    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {
    setup_csrs();

    CSR.satp.set(*SATP_VAL.get().unwrap());
    tlb::sfence_vma(None, None);

    let cpu_info = CpuInfo::get();
    if cpu_info.has_sstc() {
        CSR.henvcfg.modify(henvcfg::stce.val(1));
    }
    Imsic::setup_this_cpu();

    let me = PerCpu::this_cpu();
    me.set_online();

    HOST_VM.wait().run(me.cpu_id().raw() as u64);
    poweroff();
}
