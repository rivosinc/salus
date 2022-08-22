// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(
    asm_const,
    panic_info_message,
    allocator_api,
    alloc_error_handler,
    lang_items
)]

use core::alloc::{GlobalAlloc, Layout};
extern crate alloc;
extern crate test_workloads;

use device_tree::Fdt;
use riscv_regs::{CSR, CSR_CYCLE, CSR_INSTRET};
use s_mode_utils::abort::abort;
use s_mode_utils::ecall::ecall_send;
use s_mode_utils::print_sbi::*;
use sbi::api::{pmu, reset, tsm, tsm_aia};
use sbi::{
    PmuCounterConfigFlags, PmuCounterStartFlags, PmuCounterStopFlags, PmuEventType, PmuFirmware,
    PmuHardware, SbiMessage,
};

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
    // `shutdown` should not return, so unrapping the result is appropriate.
    reset::shutdown().unwrap();

    abort()
}

const PAGE_SIZE_4K: u64 = 4096;

// Safety: addr must point to `num_pages` of memory that isn't currently used by this program. This
// memory will be overwritten and access will be removed.
unsafe fn convert_pages(addr: u64, num_pages: u64) {
    tsm::convert_pages(addr, num_pages).expect("TsmConvertPages failed");

    // Fence the pages we just converted.
    //
    // TODO: Boot secondary CPUs and test the invalidation flow with multiple CPUs.
    tsm::initiate_fence().expect("Tellus - TsmInitiateFence failed");
}

fn reclaim_pages(addr: u64, num_pages: u64) {
    tsm::reclaim_pages(addr, num_pages).expect("TsmReclaimPages failed");

    for i in 0u64..((num_pages * PAGE_SIZE_4K) / 8) {
        let m = (addr + i) as *const u64;
        unsafe {
            if core::ptr::read_volatile(m) != 0 {
                panic!("Tellus - Read back non-zero at qword offset {i:x} after exiting from TVM!");
            }
        }
    }
}

fn get_vcpu_reg(vmid: u64, register: sbi::TvmCpuRegister) -> u64 {
    tsm::get_vcpu_reg(vmid, 0, register).expect("Tellus - TvmCpuGetRegister failed")
}

fn set_vcpu_reg(vmid: u64, register: sbi::TvmCpuRegister, value: u64) {
    tsm::set_vcpu_reg(vmid, 0, register, value).expect("Tellus - TvmCpuGetRegister failed")
}

fn exercise_pmu_functionality() {
    use sbi::api::pmu::{configure_matching_counters, start_counters, stop_counters};
    let num_counters = pmu::get_num_counters().expect("Tellus - GetNumCounters returned error");
    for i in 0u64..num_counters {
        let result = pmu::get_counter_info(i);
        if result.is_err() {
            println!("GetCounterInfo for {i} failed with {result:?}");
        }
    }

    let event_type = PmuEventType::Hardware(PmuHardware::Instructions);
    let config_flags = PmuCounterConfigFlags::default();
    let result =
        configure_matching_counters(0, (1 << num_counters) - 1, config_flags, event_type, 0);
    let counter_index = result.expect("configure_matching_counters failed with {result:?}");

    let start_flags = PmuCounterStartFlags::default();
    let result = start_counters(counter_index, 0x1, start_flags, 0);
    if result.is_err() && !matches!(result.err().unwrap(), sbi::Error::AlreadyStarted) {
        result.expect("start_counters failed with result {result:?}");
    }

    let result = stop_counters(counter_index, 0x1, PmuCounterStopFlags::default());
    result.expect("stop_counters failed with {result:?}");

    if CSR.hpmcounter[(CSR_INSTRET - CSR_CYCLE) as usize].get_value() == 0 {
        panic!("Read CSR of CSR_INSTRET returned 0");
    }

    let event_type = PmuEventType::Firmware(PmuFirmware::AccessLoad);
    pmu::configure_matching_counters(
        0,
        (1 << num_counters) - 1,
        PmuCounterConfigFlags::default(),
        event_type,
        0,
    )
    .expect_err("Successfully configured FW counter");
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const SHARED_PAGES_START_ADDRESS: u64 = 0x1_0000_0000;
    const NUM_VCPUS: u64 = 1;
    const NUM_TEE_PTE_PAGES: u64 = 10;
    const NUM_GUEST_DATA_PAGES: u64 = 160;
    const NUM_GUEST_ZERO_PAGES: u64 = 10;
    const PRE_FAULTED_ZERO_PAGES: u64 = 2;
    const NUM_GUEST_PAD_PAGES: u64 = 32;
    const NUM_GUEST_SHARED_PAGES: u64 = 1;
    const GUEST_MMIO_ADDRESS: u64 = 0x1000_8000;
    // TODO: Consider moving to a common module to ensure that the host and guest are in lockstep
    const GUEST_SHARE_PING: u64 = 0xBAAD_F00D;
    const GUEST_SHARE_PONG: u64 = 0xF00D_BAAD;

    if hart_id != 0 {
        // TODO handle more than 1 cpu
        abort();
    }

    console_write_bytes(b"Tellus: Booting the test VM\n");

    // Safe because we trust the host to boot with a valid fdt_addr pass in register a1.
    let fdt = match unsafe { Fdt::new_from_raw_pointer(fdt_addr as *const u8) } {
        Ok(f) => f,
        Err(e) => panic!("Bad FDT from hypervisor: {}", e),
    };
    let mem_range = fdt.memory_regions().next().unwrap();
    println!(
        "Tellus - Mem base: {:x} size: {:x}",
        mem_range.base(),
        mem_range.size()
    );

    let tsm_info = tsm::get_info().expect("Tellus - TsmGetInfo failed");
    let tvm_create_pages = 4
        + tsm_info.tvm_state_pages
        + ((NUM_VCPUS * tsm_info.tvm_bytes_per_vcpu) + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
    println!("Donating {} pages for TVM creation", tvm_create_pages);

    // Make sure TsmGetInfo fails if we pass it a bogus address.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmGetInfo {
        dest_addr: 0x1000,
        len: core::mem::size_of::<sbi::TsmInfo>() as u64,
    });
    // Safety: The passed info pointer is bogus and nothing should be written to our memory.
    unsafe { ecall_send(&msg).expect_err("TsmGetInfo succeeded with an invalid pointer") };

    // Donate the pages necessary to create the TVM.
    let mut next_page = (mem_range.base() + mem_range.size() / 2) & !0x3fff;
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, tvm_create_pages);
    }

    // Now create the TVM.
    let state_pages_base = next_page;
    let tvm_page_directory_addr = state_pages_base;
    let tvm_state_addr = tvm_page_directory_addr + 4 * PAGE_SIZE_4K;
    let tvm_vcpu_addr = tvm_state_addr + tsm_info.tvm_state_pages * PAGE_SIZE_4K;
    let vmid = tsm::tvm_create(
        tvm_page_directory_addr,
        tvm_state_addr,
        NUM_VCPUS,
        tvm_vcpu_addr,
    )
    .expect("Tellus - TvmCreate returned error");
    println!("Tellus - TvmCreate Success vmid: {vmid:x}");
    next_page += PAGE_SIZE_4K * tvm_create_pages;

    // Add pages for the page table
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_TEE_PTE_PAGES);
    }
    tsm::add_page_table_pages(vmid, next_page, NUM_TEE_PTE_PAGES)
        .expect("Tellus - AddPageTablePages returned error");
    next_page += PAGE_SIZE_4K * NUM_TEE_PTE_PAGES;

    // Add vCPU0.
    tsm::add_vcpu(vmid, 0).expect("Tellus - TvmCpuCreate returned error");

    // Set the IMSIC params for the TVM.
    let aia_params = sbi::TvmAiaParams {
        imsic_base_addr: 0x2800_0000,
        group_index_bits: 0,
        group_index_shift: 24,
        hart_index_bits: 8,
        guest_index_bits: 0,
        guests_per_hart: 0,
    };
    tsm_aia::tvm_aia_init(vmid, aia_params).expect("Tellus - TvmAiaInit failed");
    tsm_aia::set_vcpu_imsic_addr(vmid, 0, 0x2800_0000).expect("Tellus - TvmCpuSetImsicAddr failed");

    /*
        The Tellus composite image includes the guest image
        |========== --------> 0x8020_0000 (Tellus _start)
        | Tellus code and data
        | ....
        | .... (Zero padding)
        | ....
        |======== -------> 0x8020_0000 + 4096*NUM_GUEST_PAD_PAGES
        | Guest code and data (Guest _start is mapped at GPA 0x8020_0000)
        |
        |=========================================
    */

    let guest_image_base = USABLE_RAM_START_ADDRESS + PAGE_SIZE_4K * NUM_GUEST_PAD_PAGES;
    // Safety: Safe to make a slice out of the guest image as it is read-only and not used by this
    // program.
    let guest_image = unsafe {
        core::slice::from_raw_parts(
            guest_image_base as *const u8,
            (PAGE_SIZE_4K * NUM_GUEST_DATA_PAGES) as usize,
        )
    };
    let donated_pages_base = next_page;

    // Declare the confidential region of the guest's physical address space.
    tsm::add_confidential_memory_region(
        vmid,
        USABLE_RAM_START_ADDRESS,
        (NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES) * PAGE_SIZE_4K,
    )
    .expect("Tellus - TvmAddConfidentialMemoryRegion failed");

    // Add data pages
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_GUEST_DATA_PAGES);
    }
    tsm::add_measured_pages(
        vmid,
        guest_image,
        next_page,
        sbi::TsmPageType::Page4k,
        USABLE_RAM_START_ADDRESS,
    )
    .expect("Tellus - TvmAddMeasuredPages returned error");
    next_page += PAGE_SIZE_4K * NUM_GUEST_DATA_PAGES;

    // Convert the zero pages and map a few of them up front. We'll fault the rest in as necessary.
    let zero_pages_start = next_page;
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_GUEST_ZERO_PAGES);
    }
    tsm::add_zero_pages(
        vmid,
        zero_pages_start,
        sbi::TsmPageType::Page4k,
        PRE_FAULTED_ZERO_PAGES,
        USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K,
    )
    .expect("Tellus - TvmAddZeroPages failed");

    next_page += NUM_GUEST_ZERO_PAGES * PAGE_SIZE_4K;
    let shared_page_base = next_page;

    let mut zero_pages_added = PRE_FAULTED_ZERO_PAGES;

    // Add a page of emualted MMIO.
    tsm::add_emulated_mmio_region(vmid, GUEST_MMIO_ADDRESS, PAGE_SIZE_4K)
        .expect("Tellus - TvmAddEmulatedMmioRegion failed");

    // Set the entry point.
    tsm::set_vcpu_reg(vmid, 0, sbi::TvmCpuRegister::EntryPc, 0x8020_0000)
        .expect("Tellus - TvmCpuSetRegister returned error");

    // Set the kernel_init() parameter.
    tsm::set_vcpu_reg(
        vmid,
        0,
        sbi::TvmCpuRegister::EntryArg,
        SHARED_PAGES_START_ADDRESS,
    )
    .expect("Tellus - TvmCpuSetRegister returned error");

    tsm::add_shared_memory_region(
        vmid,
        SHARED_PAGES_START_ADDRESS,
        NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
    )
    .expect("Tellus -- TvmAddSharedMemoryRegion returned error");

    tsm::add_shared_memory_region(
        vmid,
        SHARED_PAGES_START_ADDRESS,
        NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
    )
    .expect_err("Tellus -- TvmAddSharedMemoryRegion succeeded second time");

    // TODO test that access to pages crashes somehow
    tsm::tvm_finalize(vmid).expect("Tellus - Finalize returned error");

    loop {
        // Safety: running a VM can't affect host memory as that memory isn't accessible to the VM.
        match tsm::tvm_run(vmid, 0) {
            Err(e) => {
                println!("Tellus - Run returned error {:?}", e);
                panic!("Could not run guest VM");
            }
            Ok(cause) => {
                match cause {
                    sbi::TvmCpuExitCode::ConfidentialPageFault => {
                        let fault_addr = get_vcpu_reg(vmid, sbi::TvmCpuRegister::ExitCause0);
                        // Fault in the page.
                        if zero_pages_added >= NUM_GUEST_ZERO_PAGES {
                            panic!("Ran out of pages to fault in");
                        }
                        tsm::add_zero_pages(
                            vmid,
                            zero_pages_start + zero_pages_added * PAGE_SIZE_4K,
                            sbi::TsmPageType::Page4k,
                            1,
                            fault_addr & !(PAGE_SIZE_4K - 1),
                        )
                        .expect("Tellus - TvmAddZeroPages failed");
                        zero_pages_added += 1;
                    }
                    sbi::TvmCpuExitCode::SharedPageFault => {
                        // Figure out where the fault occurred.
                        let fault_addr = get_vcpu_reg(vmid, sbi::TvmCpuRegister::ExitCause0);
                        if fault_addr != SHARED_PAGES_START_ADDRESS {
                            panic!("Unexpected shared page fault address at {fault_addr:x}");
                        }
                        // Safety: shared_page_base points to pages that will only be accessed as
                        // volatile from here on.
                        unsafe {
                            tsm::add_shared_pages(
                                vmid,
                                shared_page_base,
                                sbi::TsmPageType::Page4k,
                                NUM_GUEST_SHARED_PAGES,
                                fault_addr,
                            )
                            .expect("Tellus -- TvmAddSharedPages failed");
                        }

                        // Safety: We own the page, and are writing a value expected by the guest
                        // Note that any access to shared pages must use volatile memory semantics
                        // to guard against the compiler's no-aliasing assumptions.
                        unsafe {
                            core::ptr::write_volatile(
                                shared_page_base as *mut u64,
                                GUEST_SHARE_PING,
                            );
                        }
                    }
                    sbi::TvmCpuExitCode::MmioPageFault => {
                        let fault_addr = get_vcpu_reg(vmid, sbi::TvmCpuRegister::ExitCause0);
                        let op = sbi::TvmMmioOpCode::from_reg(get_vcpu_reg(
                            vmid,
                            sbi::TvmCpuRegister::ExitCause1,
                        ))
                        .unwrap();
                        // Handle the load or store.
                        use sbi::TvmMmioOpCode::*;
                        match op {
                            Load8 | Load8U | Load16 | Load16U | Load32 | Load32U | Load64 => {
                                set_vcpu_reg(vmid, sbi::TvmCpuRegister::MmioLoadValue, 0x42);
                            }
                            Store8 | Store16 | Store32 | Store64 => {
                                let val = get_vcpu_reg(vmid, sbi::TvmCpuRegister::MmioStoreValue);
                                println!("Guest says: 0x{:x} at 0x{:x}", val, fault_addr);
                            }
                        }
                    }
                    sbi::TvmCpuExitCode::WaitForInterrupt => {
                        continue;
                    }
                    _ => {
                        println!("Tellus - Guest exited with status {:?}", cause);
                        break;
                    }
                }
            }
        }
    }

    tsm::tvm_destroy(vmid).expect("Tellus - TvmDestroy returned error");

    // Safety: We own the page.
    // Note that any access to shared pages must use volatile memory semantics
    // to guard against the compiler's no-aliasing assumptions.
    let guest_written_value = unsafe { core::ptr::read_volatile(shared_page_base as *mut u64) };
    if guest_written_value != GUEST_SHARE_PONG {
        println!("Tellus - unexpected value from guest shared page: 0x{guest_written_value:x}");
    }
    // Check that we can reclaim previously-converted pages and that they have been cleared.
    reclaim_pages(
        donated_pages_base,
        NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES,
    );
    reclaim_pages(state_pages_base, tvm_create_pages);
    exercise_pmu_functionality();
    println!("Tellus - All OK");
    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
