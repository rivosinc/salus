// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! A small Risc-V hypervisor to enable trusted execution environments.

#![no_main]
#![no_std]
#![feature(
    allocator_api,
    alloc_error_handler,
    if_let_guard,
    slice_ptr_get,
    let_chains,
    negative_impls
)]
#![feature(custom_test_frameworks)]
#![test_runner(crate::tests::test_runner)]
#![reexport_test_harness_main = "test_main"]
#![cfg_attr(test, allow(unused))]

use core::alloc::{Allocator, GlobalAlloc, Layout};
use core::fmt::Display;
use core::ptr::NonNull;
use test_system::*;
extern crate alloc;

mod asm;
mod backtrace;
mod guest_tracking;
mod host_vm;
mod hyp_layout;
mod hyp_map;
mod smp;
mod trap;
mod umode;
mod vm;
mod vm_cpu;
mod vm_id;
mod vm_interrupts;
mod vm_pages;
mod vm_pmu;

use backtrace::backtrace;
use device_tree::{DeviceTree, DeviceTreeError, Fdt};
use drivers::{
    imsic::Imsic, iommu::Iommu, pci::PcieRoot, pmu::PmuInfo, reset::ResetDriver, uart::UartDriver,
    CpuInfo,
};
use host_vm::{HostVm, HostVmLoader, HOST_VM_ALIGN};
use hyp_alloc::HypAlloc;
use hyp_map::HypMap;
use page_tracking::*;
use riscv_elf::ElfMap;
use riscv_page_tables::*;
use riscv_pages::*;
use riscv_regs::{hedeleg, henvcfg, hideleg, hie, hstateen0, scounteren};
use riscv_regs::{sstatus, vlenb, Readable, RiscvCsrInterface, MAX_VECTOR_REGISTER_LEN};
use riscv_regs::{
    Exception, Interrupt, LocalRegisterCopy, ReadWriteable, Writeable, CSR, CSR_CYCLE, CSR_TIME,
};
use s_mode_utils::abort::abort;
use s_mode_utils::print::*;
use s_mode_utils::sbi_console::SbiConsoleV01;
use smp::PerCpu;
use sync::Once;
use umode::UmodeTask;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    if let Some(location) = info.location() {
        println!(
            "panic occurred in file '{}' at line {}",
            location.file(),
            location.line()
        );
    }

    println!("panic backtrace:");
    // Skip the first 2 calls on the stack.
    // They are always the same, and are from the
    // panic handling code
    if let Some(bt) = backtrace() {
        bt.skip(2).for_each(|frame| {
            print!("{}", frame);
        });
    }

    abort()
}

extern "C" {
    static _start: u8;
    static _stack_end: u8;
    static _umode_bin: u8;
    static _umode_bin_len: u8;
}

/// The allocator used for boot-time dynamic memory allocations.
static HYPERVISOR_ALLOCATOR: Once<HypAlloc> = Once::new();

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
    ResetDriver::shutdown();
}

/// The host VM that all CPUs enter at boot.
static HOST_VM: Once<HostVm<Sv48x4>> = Once::new();

/// Builds the hardware memory map from the device-tree. The kernel & initramfs image regions are
/// aligned to `HOST_VM_ALIGN` so that they can be mapped directly into the host VM's guest
/// physical address space.
fn build_memory_map(fdt: &Fdt) -> MemMapResult<HwMemMap> {
    let mut builder = HwMemMapBuilder::new(HOST_VM_ALIGN as u64);

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
    let start = core::ptr::addr_of!(_start) as u64;
    let stack_end = core::ptr::addr_of!(_stack_end) as u64;

    // Find the region of DRAM that the hypervisor is in.
    let resv_base = fdt
        .memory_regions()
        .find(|r| start >= r.base() && stack_end <= r.base().checked_add(r.size()).unwrap())
        .map(|r| RawAddr::supervisor(r.base()))
        .ok_or(MemMapError::NonContiguousHypervisorImage)?;

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
        assert_eq!(r.base() & (HOST_VM_ALIGN as u64 - 1), 0);
        builder = builder.reserve_region(
            HwReservedMemType::HostKernelImage,
            RawAddr::supervisor(r.base()),
            HOST_VM_ALIGN.round_up(r.size()),
        )?;
    }
    if let Some(r) = fdt.host_initramfs_region() {
        assert_eq!(r.base() & (HOST_VM_ALIGN as u64 - 1), 0);
        builder = builder.reserve_region(
            HwReservedMemType::HostInitramfsImage,
            RawAddr::supervisor(r.base()),
            HOST_VM_ALIGN.round_up(r.size()),
        )?;
    }
    Ok(builder.build())
}

// Returns the base address of the first available region in the memory map that is at least `size`
// bytes long. Returns None if no region is big enough.
fn find_available_region(mem_map: &HwMemMap, size: u64) -> Option<SupervisorPageAddr> {
    mem_map
        .regions()
        .find(|r| r.region_type() == HwMemRegionType::Available && r.size() >= size)
        .map(|r| r.base())
}

/// Creates a heap from the given `mem_map`, marking the region occupied by the heap as reserved.
fn create_heap(mem_map: &mut HwMemMap) -> Result<(), Error> {
    const HEAP_SIZE: u64 = 16 * 1024 * 1024;

    let heap_base = find_available_region(mem_map, HEAP_SIZE).ok_or(Error::HeapOutOfSpace)?;

    mem_map
        .reserve_region(
            HwReservedMemType::HypervisorHeap,
            RawAddr::from(heap_base),
            HEAP_SIZE,
        )
        .map_err(Error::HeapReserve)?;
    let pages: SequentialPages<InternalDirty> = unsafe {
        // Safe since this region of memory was free in the memory map.
        SequentialPages::from_mem_range(
            heap_base,
            PageSize::Size4k,
            HEAP_SIZE / PageSize::Size4k as u64,
        )
        .map_err(|_| Error::HeapUnaligned)
    }?;
    HYPERVISOR_ALLOCATOR.call_once(|| HypAlloc::from_pages(pages.clean()));
    Ok(())
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

    // Enable access to timer for now.
    CSR.hcounteren.set(1 << (CSR_TIME - CSR_CYCLE));

    // Make the basic counters available to any of our U-mode tasks.
    let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
    scounteren.modify(scounteren::cycle.val(1));
    scounteren.modify(scounteren::time.val(1));
    scounteren.modify(scounteren::instret.val(1));
    CSR.scounteren.set(scounteren.get());

    trap::install_init_trap_handler();
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

#[repr(C)]
struct CpuParams {
    satp: u64,
}

/// The entry point for the test runner
#[cfg(test)]
#[no_mangle]
extern "C" fn _primary_init(_hart_id: u64, _fdt_addr: u64) -> CpuParams {
    setup_csrs();
    SbiConsoleV01::set_as_console();

    println!("\n\nSalus: Booting into test runner");

    // test_main created automatically by test_harness
    test_main();

    poweroff();
}

#[cfg(test)]
#[no_mangle]
extern "C" fn _primary_main() {}

#[derive(Debug)]
enum RequiredCpuFeature {
    Aia,
    Sstc,
    Svadu,
}

#[derive(Debug)]
enum RequiredDeviceProbe {
    Imsic(drivers::imsic::ImsicError),
    Pci(drivers::pci::PciError),
    Reset(drivers::reset::Error),
    Uart(drivers::uart::Error),
}

impl Display for RequiredDeviceProbe {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use RequiredDeviceProbe::*;
        match self {
            Imsic(e) => write!(f, "IMSIC: {:?}", e),
            Pci(e) => write!(f, "PCI: {:?}", e),
            Reset(e) => write!(f, "Reset: {:?}", e),
            Uart(e) => write!(f, "UART: {:?}", e),
        }
    }
}

/// Errors from initialization
#[derive(Debug)]
enum Error {
    /// Problem building memory map
    BuildMemoryMap(MemMapError),
    /// CPU missing feature
    CpuMissingFeature(RequiredCpuFeature),
    /// Problem generating CPU topology
    CpuTopologyGeneration(drivers::cpu::Error),
    // Error creating hypervisor allocator
    CreateHypervisorAllocator(PageTrackingError),
    /// Creating hypervisor map failed
    CreateHypervisorMap(hyp_map::Error),
    /// Creating (per CPU) SMP state
    CreateSmpState(smp::Error),
    /// Problem creating derived device tree
    FdtCreation(DeviceTreeError),
    /// Problem parsing device tree
    FdtParsing(DeviceTreeError),
    /// Not enough free memory for heap
    HeapOutOfSpace,
    /// Heap memory is unaligned
    HeapUnaligned,
    /// Unable to reserve memory for heap
    HeapReserve(MemMapError),
    /// Kernel is missing
    KernelMissing,
    /// Loading user-mode binary failed
    LoadUserMode(riscv_elf::Error),
    /// Probing of required device failed
    RequiredDeviceProbe(RequiredDeviceProbe),
    /// Setup of user-mode failed
    SetupUserMode(umode::Error),
    /// Problem running secondary CPUs
    StartSecondaryCpus(smp::Error),
    /// User-mode NOP failed
    UserModeNop(umode::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use Error::*;
        match self {
            BuildMemoryMap(e) => write!(f, "Failed to build memory map: {:?}", e),
            CpuMissingFeature(feature) => write!(f, "Missing required CPU feature: {:?}", feature),
            CpuTopologyGeneration(e) => write!(f, "Failed to generate CPU topology: {}", e),
            CreateHypervisorAllocator(e) => {
                write!(f, "Failed to create hypervisor page allocator: {:?}", e)
            }
            CreateHypervisorMap(e) => write!(f, "Cannot create Hypervisor map: {:?}", e),
            CreateSmpState(e) => write!(f, "Error during (per CPU) SMP setup: {}", e),
            FdtCreation(e) => write!(f, "Failed to construct device-tree: {}", e),
            FdtParsing(e) => write!(f, "Failed to read FDT: {}", e),
            HeapOutOfSpace => write!(f, "Not enough free memory for hypervisor heap"),
            HeapUnaligned => write!(f, "Heap memory is unaligned"),
            HeapReserve(e) => write!(f, "Error reserving heap memory: {:?}", e),
            KernelMissing => write!(f, "No host kernel image"),
            LoadUserMode(e) => write!(f, "Cannot load user-mode ELF binary: {:?}", e),
            RequiredDeviceProbe(e) => write!(f, "Failed to probe required device: {}", e),
            SetupUserMode(e) => write!(f, "Failed to setup user-mode: {:?}", e),
            StartSecondaryCpus(e) => write!(f, "Error running secondary CPUs: {}", e),
            UserModeNop(e) => write!(f, "Failed to execute a NOP in user-mode: {:?}", e),
        }
    }
}

/// Initialization logic for the bootstrap CPU. Bootstraps salus, starts
/// secondary CPUs and builds the HOST VM.
///
/// Note: this runs with a 1:1 map of physical to virtual mapping.
#[cfg(not(test))]
fn primary_init(hart_id: u64, fdt_addr: u64) -> Result<CpuParams, Error> {
    // Reset CSRs to a sane state.
    setup_csrs();

    SbiConsoleV01::set_as_console();
    println!("Salus: Boot test VM");

    test_declare_pass!("Salus Boot", hart_id);

    // Safe because we trust that the firmware passed a valid FDT.
    let hyp_fdt =
        unsafe { Fdt::new_from_raw_pointer(fdt_addr as *const u8) }.map_err(Error::FdtParsing)?;

    let mut mem_map = build_memory_map(&hyp_fdt).map_err(Error::BuildMemoryMap)?;

    // Find where QEMU loaded the host kernel image.
    let host_kernel = *mem_map
        .regions()
        .find(|r| r.region_type() == HwMemRegionType::Reserved(HwReservedMemType::HostKernelImage))
        .ok_or(Error::KernelMissing)?;
    let host_initramfs = mem_map
        .regions()
        .find(|r| {
            r.region_type() == HwMemRegionType::Reserved(HwReservedMemType::HostInitramfsImage)
        })
        .cloned();

    // Create a heap for boot-time memory allocations.
    create_heap(&mut mem_map)?;

    let hyp_dt = DeviceTree::from(&hyp_fdt).map_err(Error::FdtCreation)?;

    // Find the UART and switch to it as the system console.
    UartDriver::probe_from(&hyp_dt, &mut mem_map)
        .map_err(|e| Error::RequiredDeviceProbe(RequiredDeviceProbe::Uart(e)))?;

    // Discover the CPU topology.
    CpuInfo::parse_from(&hyp_dt).map_err(Error::CpuTopologyGeneration)?;
    let cpu_info = CpuInfo::get();
    if !cpu_info.has_aia() {
        // We require AIA support for interrupts and SMP support; no point continuing without it.
        return Err(Error::CpuMissingFeature(RequiredCpuFeature::Aia));
    }
    if !cpu_info.has_sstc() {
        // We don't implement or use the SBI timer extension and thus require Sstc for timers.
        return Err(Error::CpuMissingFeature(RequiredCpuFeature::Sstc));
    }
    if !cpu_info.has_svadu() {
        // Salus assumes that hardware will update the accessed and dirty bits in the page table.
        // It can't handle faults for updating them.
        return Err(Error::CpuMissingFeature(RequiredCpuFeature::Svadu));
    }

    set_henvcfg(cpu_info);

    if cpu_info.has_sscofpmf() {
        // Only probe for PMU counters if we have Sscofpmf; we can't expose counters to guests
        // unless we have support for per-mode filtering.
        println!("Sscofpmf support present");
        if let Err(e) = PmuInfo::init() {
            test_declare_fail!("PMU counters");
            println!("PmuInfo::init() failed with {:?}", e);
        } else {
            test_declare_pass!("PMU counters");
        }
    }
    if cpu_info.has_vector() {
        // Will panic if register width too long. (currently 256 bits)
        check_vector_width();
    } else {
        println!("No vector support");
    }

    set_hstateen(cpu_info);

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
    Imsic::probe_from(&hyp_dt, &mut mem_map)
        .map_err(|e| Error::RequiredDeviceProbe(RequiredDeviceProbe::Imsic(e)))?;
    let imsic_geometry = Imsic::get().phys_geometry();
    println!(
        "IMSIC at 0x{:08x}; {} guest interrupt files supported",
        imsic_geometry.base_addr().bits(),
        imsic_geometry.guests_per_hart()
    );
    Imsic::setup_this_cpu();

    // Probe for a PCI bus.
    PcieRoot::probe_from(&hyp_dt, &mut mem_map)
        .map_err(|e| Error::RequiredDeviceProbe(RequiredDeviceProbe::Pci(e)))?;
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

    // Probe for hardcoded reset device. Not really a probe.
    ResetDriver::probe_from(&hyp_dt, &mut mem_map)
        .map_err(|e| Error::RequiredDeviceProbe(RequiredDeviceProbe::Reset(e)))?;

    // Create an allocator for the remaining pages. Anything that's left over will be mapped
    // into the host VM.
    let mut hyp_mem = HypPageAlloc::new(&mut mem_map).map_err(Error::CreateHypervisorAllocator)?;
    // NOTE: Do not modify the hardware memory map from here on.
    let mem_map = mem_map; // Remove mutability.

    // We start RAM in the host address space at the same location as it is in the supervisor
    // address space.
    //
    // Unwrap ok here and below since we must have at least one RAM region.
    let guest_ram_base = mem_map
        .regions()
        .find(|r| !matches!(r.region_type(), HwMemRegionType::Mmio(_)))
        .map(|r| r.base().as_guest_phys(PageOwnerId::host()))
        .unwrap();
    // For the purposes of calculating the total size of the host VM's guest physical address
    // space, use the start and end of the (real) physical memory map. This is a bit of an
    // over-estimate given that not all of the physical memory map ends up getting mapped into
    // the host VM.
    let guest_phys_size = mem_map.regions().last().unwrap().end().bits()
        - mem_map.regions().next().unwrap().base().bits();

    // Parse the user-mode ELF containing the user-mode task.
    // Safe, because it comes from the Linker
    let umode_bytes = unsafe {
        let umode_bin = core::ptr::addr_of!(_umode_bin);
        let umode_bin_len = core::ptr::addr_of!(_umode_bin_len) as usize;
        core::slice::from_raw_parts::<u8>(umode_bin, umode_bin_len)
    };
    let umode_elf = ElfMap::new(umode_bytes).map_err(Error::LoadUserMode)?;

    println!("HW memory map:");
    for (i, r) in mem_map.regions().enumerate() {
        println!(
            "[{:02}] region: 0x{:016x} -> 0x{:016x}, {}",
            i,
            r.base().bits(),
            r.end().bits() - 1,
            r.region_type()
        );
    }

    println!("umode memory map:");
    for (i, s) in umode_elf.segments().enumerate() {
        println!(
            "[{:02}] region: 0x{:016x} -> 0x{:016x}, {}",
            i,
            s.vaddr(),
            s.vaddr() + s.size() as u64,
            s.perms()
        );
    }

    // Create the hypervisor mapping from the hardware memory map and the U-mode ELF.
    HypMap::init(mem_map, &umode_elf).map_err(Error::CreateHypervisorMap)?;

    // Set up per-CPU memory and prepare the structures for secondary CPUs boot.
    PerCpu::init(hart_id, &mut hyp_mem).map_err(Error::CreateSmpState)?;

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

    // Initialize global Umode state.
    UmodeTask::init(umode_elf);

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

    smp::start_secondary_cpus().map_err(Error::StartSecondaryCpus)?;

    HOST_VM.call_once(|| host);

    // Return launch parameters for this CPU.
    Ok(CpuParams {
        satp: PerCpu::this_cpu().page_table().satp(),
    })
}

/// Rust Entry point for the bootstrap CPU. Calls into init to allow
/// clean error collection and presentation
///
/// Note: this runs with a 1:1 map of physical to virtual mapping.
#[cfg(not(test))]
#[no_mangle]
extern "C" fn _primary_init(hart_id: u64, fdt_addr: u64) -> CpuParams {
    match primary_init(hart_id, fdt_addr) {
        Err(e) => {
            // Safe as nothing can fail before console is setup
            println!("Error during initialization on bootstrap CPU: {}", e);
            // Safe as if shutdown fails it triggers a wfi loop (via abort())
            poweroff();
        }
        Ok(params) => params,
    }
}

/// Run salus on the bootstrap CPU. This runs with the hypervisor page table
fn primary_main() -> Result<(), Error> {
    // Install trap handler with stack overflow checking.
    trap::install_trap_handler();
    // Setup U-mode task for this CPU.
    UmodeTask::setup_this_cpu().map_err(Error::SetupUserMode)?;
    // Do a NOP request to the U-mode task to check it's functional in this CPU.
    UmodeTask::nop().map_err(Error::UserModeNop)?;

    let cpu_id = PerCpu::this_cpu().cpu_id();
    HOST_VM.get().unwrap().run(cpu_id.raw() as u64);
    poweroff();
}

/// Start salus on the bootstrap CPU. This is called once we switch to the hypervisor page-table.
#[cfg(not(test))]
#[no_mangle]
extern "C" fn _primary_main() {
    if let Err(e) = primary_main() {
        // Safe as nothing can fail before console is setup
        println!("Error during main on bootstrap CPU: {}", e);
        // Safe as if shutdown fails it triggers a wfi loop (via abort())
        poweroff();
    }
}

#[cfg(test)]
#[no_mangle]
extern "C" fn _secondary_init(_hart_id: u64) {}

#[cfg(test)]
#[no_mangle]
extern "C" fn _secondary_main() {}

/// Initialization logic for the secondary CPUs.
///
/// Note: this runs with a 1:1 map of physical to virtual mapping.
#[cfg(not(test))]
fn secondary_init(hart_id: u64) -> Result<CpuParams, Error> {
    setup_csrs();
    PerCpu::setup_this_cpu(hart_id).map_err(Error::CreateSmpState)?;

    test_declare_pass!("secondary init", hart_id);

    let cpu_info = CpuInfo::get();
    set_henvcfg(cpu_info);
    set_hstateen(cpu_info);
    Imsic::setup_this_cpu();

    let this_cpu = PerCpu::this_cpu();
    this_cpu.set_online();

    Ok(CpuParams {
        satp: this_cpu.page_table().satp(),
    })
}

/// Rust entry point for secondary CPUs. Initializes the current CPU and returns.
///
/// Note: this runs with a 1:1 map of physical to virtual mapping.
#[cfg(not(test))]
#[no_mangle]
extern "C" fn _secondary_init(hart_id: u64) -> CpuParams {
    match secondary_init(hart_id) {
        Err(e) => {
            // Safe as nothing can fail before console is setup
            println!(
                "Error during initialization on secondary CPU ({}): {}",
                hart_id, e
            );
            // Safe as if shutdown fails it triggers a wfi loop (via abort())
            poweroff();
        }
        Ok(params) => params,
    }
}

/// Run salus on secondary CPUs. This runs with the hypervisor page-table.
#[cfg(not(test))]
fn secondary_main() -> Result<(), Error> {
    // Install trap handler with stack overflow checking.
    trap::install_trap_handler();
    // Setup U-mode task for this CPU.
    UmodeTask::setup_this_cpu().map_err(Error::SetupUserMode)?;
    // Do a NOP request to the U-mode task to check it's functional in this CPU.
    UmodeTask::nop().map_err(Error::UserModeNop)?;

    HOST_VM.wait().run(PerCpu::this_cpu().cpu_id().raw() as u64);
    poweroff();
}

/// Start salus on secondary CPUs. This is called once we switch to the hypervisor page-table.
#[cfg(not(test))]
#[no_mangle]
extern "C" fn _secondary_main() {
    if let Err(e) = secondary_main() {
        // Safe as nothing can fail before console is setup
        println!("Error during main on secondary CPU: {}", e);
        // Safe as if shutdown fails it triggers a wfi loop (via abort())
        poweroff();
    }
}

// Configures henvcfg to select what features are available to the host VM.
fn set_henvcfg(cpu_info: &CpuInfo) {
    // Sstc is present, it's checked as a required feature early in boot.
    CSR.henvcfg.modify(henvcfg::stce.val(1));

    if cpu_info.has_zicboz() {
        CSR.henvcfg.modify(henvcfg::cbze.val(1));
    }
    if cpu_info.has_zicbom() {
        CSR.henvcfg.modify(henvcfg::cbie.val(1));
        CSR.henvcfg.modify(henvcfg::cbcfe.val(1));
    }
    // Svadu is a required feature for Salus.
    CSR.henvcfg.modify(henvcfg::adue.val(1));
}

// Sets the hstateen CSR.
fn set_hstateen(cpu_info: &CpuInfo) {
    // If sstateen is present, enable supported extensions.
    if cpu_info.has_sstateen() {
        CSR.hstateen0.modify(hstateen0::ctr.val(1));
        CSR.hstateen0.modify(hstateen0::aia_imsic.val(1));
        CSR.hstateen0.modify(hstateen0::aia.val(1));
        CSR.hstateen0.modify(hstateen0::csrind.val(1));
        CSR.hstateen0.modify(hstateen0::envcfg.val(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub trait Testable {
        fn run(&self) -> TestResult;
    }

    impl<T> Testable for T
    where
        T: Fn() -> TestResult,
    {
        fn run(&self) -> TestResult {
            println!("---{}---\t", core::any::type_name::<T>());
            self()
        }
    }

    pub fn test_runner(tests: &[&dyn Testable]) {
        println!("Running {} tests\n", tests.len());
        for test in tests {
            match test.run() {
                Ok(()) => println!("pass"),
                Err(TestFailure::Fail) => println!("FAIL"),
                Err(TestFailure::FailedAt(loc)) => println!("FAILED at {loc}"),
            }
        }
    }
}
