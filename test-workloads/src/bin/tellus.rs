// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};

extern crate alloc;
extern crate test_workloads;

use device_tree::Fdt;
use s_mode_utils::abort::abort;
use s_mode_utils::ecall::ecall_send;
use s_mode_utils::print_sbi::*;
use sbi::SbiMessage;

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
    let msg = SbiMessage::Reset(sbi::ResetFunction::shutdown());
    // Safety: This ecall doesn't touch memory and will never return.
    unsafe {
        ecall_send(&msg).unwrap();
    }

    abort()
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const PAGE_SIZE_4K: u64 = 4096;
    const NUM_VCPUS: u64 = 1;
    const NUM_TEE_PTE_PAGES: u64 = 10;
    const NUM_GUEST_DATA_PAGES: u64 = 10;
    const NUM_GUEST_ZERO_PAGES: u64 = 10;
    const NUM_GUEST_PAD_PAGES: u64 = 32;

    if hart_id != 0 {
        // TODO handle more than 1 cpu
        abort();
    }

    // Safe because this is the very begining of the program and nothing has set the console driver
    // yet.
    unsafe {
        SbiConsoleDriver::new().use_as_console();
    }
    console_write_str("Tellus: Booting the test VM\n");

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

    let mut tsm_info = sbi::TsmInfo::default();
    let tsm_info_size = core::mem::size_of::<sbi::TsmInfo>() as u64;
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmGetInfo {
        dest_addr: &mut tsm_info as *mut _ as u64,
        len: tsm_info_size,
    });
    // Safety: The passed info pointer is uniquely owned so it's safe to modify in SBI.
    let tsm_info_len = unsafe { ecall_send(&msg).expect("TsmGetInfo failed") };
    assert_eq!(tsm_info_len, tsm_info_size);
    let tvm_create_pages = 4
        + tsm_info.tvm_state_pages
        + ((NUM_VCPUS * tsm_info.tvm_bytes_per_vcpu) + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
    println!("Donating {} pages for TVM creation", tvm_create_pages);

    // Try to create a TEE
    let mut next_page = (mem_range.base() + mem_range.size() / 2) & !0x3fff;
    let tvm_page_directory_addr = next_page;
    let tvm_state_addr = tvm_page_directory_addr + 4 * PAGE_SIZE_4K;
    let tvm_vcpu_addr = tvm_state_addr + tsm_info.tvm_state_pages * PAGE_SIZE_4K;
    let tvm_create_params = sbi::TvmCreateParams {
        tvm_page_directory_addr,
        tvm_state_addr,
        tvm_num_vcpus: NUM_VCPUS,
        tvm_vcpu_addr,
    };
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCreate {
        params_addr: (&tvm_create_params as *const sbi::TvmCreateParams) as u64,
        len: core::mem::size_of::<sbi::TvmCreateParams>() as u64,
    });
    // Safety: `TvmCreate` donates the pages pointed to by `tvm_create_params` to SBI. This is safe
    // because those pages are not used by this program.
    let vmid = unsafe { ecall_send(&msg).expect("Tellus - TvmCreate returned error") };
    println!("Tellus - TvmCreate Success vmid: {vmid:x}");
    next_page += PAGE_SIZE_4K * tvm_create_pages;

    // Add pages for the page table
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPageTablePages {
        guest_id: vmid,
        page_addr: next_page,
        num_pages: NUM_TEE_PTE_PAGES,
    });
    // Safety: `AddPageTablePages` donates the pages pointed to by `tvm_create_params` to SBI. This
    // is safe because those pages are not used by this program.
    unsafe { ecall_send(&msg).expect("Tellus - AddPageTablePages returned error") };
    next_page += PAGE_SIZE_4K * NUM_TEE_PTE_PAGES;

    // Add vCPU0.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuCreate {
        guest_id: vmid,
        vcpu_id: 0,
    });
    // Safety: Creating a vcpu doesn't touch any memory owned here.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmCpuCreate returned error");
    }

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

    let first_guest_page = USABLE_RAM_START_ADDRESS + PAGE_SIZE_4K * NUM_GUEST_PAD_PAGES;
    let measurement_page_addr = next_page;
    // Add data pages
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPages {
        guest_id: vmid,
        page_addr: first_guest_page,
        page_type: 0,
        num_pages: NUM_GUEST_DATA_PAGES,
        gpa: USABLE_RAM_START_ADDRESS,
        measure_preserve: false,
    });
    // Safety: `AddPages` donates the pages pointed to by `tvm_create_params` to SBI. This is
    // safe because those pages are not used by this program.
    unsafe {
        ecall_send(&msg).expect("Tellus - AddPages returned error");
    }

    let msg = SbiMessage::Measurement(sbi::MeasurementFunction::GetSelfMeasurement {
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });

    // Safety: The measurement page is uniquely owned and can be written to safely by SBI
    match unsafe { ecall_send(&msg) } {
        Err(e) => {
            println!("Host measurement error {e:?}");
            panic!("Host measurement call failed");
        }
        Ok(_) => {
            let measurement =
                unsafe { core::ptr::read_volatile(measurement_page_addr as *const u64) };
            println!("Host measurement was {measurement:x}");
        }
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::GetGuestMeasurement {
        guest_id: vmid,
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });

    // Safety: The measurement page is uniquely owned and can be written to safely by SBI
    match unsafe { ecall_send(&msg) } {
        Err(e) => {
            println!("Guest measurement error {e:?}");
            panic!("Guest measurement call failed");
        }
        Ok(_) => {
            let measurement =
                unsafe { core::ptr::read_volatile(measurement_page_addr as *const u64) };
            println!("Guest measurement was {measurement:x}");
        }
    }

    // Add zeroed (non-measured) pages
    // TODO: Make sure that these guest pages are actually zero
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPages {
        guest_id: vmid,
        page_addr: first_guest_page + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K,
        page_type: 0,
        num_pages: NUM_GUEST_ZERO_PAGES,
        gpa: USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K,
        measure_preserve: true,
    });
    // Safety: `AddPages` donates the pages pointed to by `tvm_create_params` to SBI. This is
    // safe because those pages are not used by this program.
    unsafe {
        ecall_send(&msg).expect("Tellus - AddPages Zeroed returned error");
    }

    // Set the entry point.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuSetRegister {
        guest_id: vmid,
        vcpu_id: 0,
        register: sbi::TvmCpuRegister::Pc,
        value: 0x8020_0000,
    });
    // Safety: Setting a guest register doesn't affect host memory safety.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmCpuSetRegister returned error");
    }

    // TODO test that access to pages crashes somehow

    let msg = SbiMessage::Tee(sbi::TeeFunction::Finalize { guest_id: vmid });
    // Safety: `Finalize` doesn't touch memory.
    unsafe {
        ecall_send(&msg).expect("Tellus - Finalize returned error");
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuRun {
        guest_id: vmid,
        vcpu_id: 0,
    });
    // Safety: running a VM can't affect host memory as that memory isn't accessible to the VM.
    match unsafe { ecall_send(&msg) } {
        Err(e) => {
            println!("Tellus - Run returned error {:?}", e);
            panic!("Could not run guest VM");
        }
        Ok(_) => println!("Tellus - Run success"),
    }

    let num_remove_pages = NUM_GUEST_DATA_PAGES;
    let msg = SbiMessage::Tee(sbi::TeeFunction::RemovePages {
        guest_id: vmid,
        gpa: USABLE_RAM_START_ADDRESS + num_remove_pages * PAGE_SIZE_4K,
        remap_addr: first_guest_page,
        num_pages: NUM_GUEST_DATA_PAGES,
    });
    // Safety: destroying a VM doesn't write to memory that's accessible from the host.
    unsafe {
        ecall_send(&msg).expect("Tellus - RemovePages returned error");
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmDestroy { guest_id: vmid });
    // Safety: destroying a VM doesn't write to memory that's accessible from the host.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmDestroy returned error");
    }

    // check that guest pages have been cleared
    for i in 0u64..(NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES) / 8 {
        let m = (first_guest_page + i) as *const u64;
        unsafe {
            if core::ptr::read_volatile(m) != 0 {
                panic!("Tellus - Read back non-zero at qword offset {i:x} after exiting from TVM!");
            }
        }
    }

    println!("Tellus - All OK");

    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
