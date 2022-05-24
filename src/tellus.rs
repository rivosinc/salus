// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};

extern crate alloc;

mod asm;
mod ecall;
mod print_sbi;

use sbi::SbiMessage;

use device_tree::Fdt;
use ecall::ecall_send;
use print_sbi::*;

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
    ecall_send(&msg).unwrap();

    abort()
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const PAGE_SIZE_4K: u64 = 4096;
    const NUM_TEE_CREATE_PAGES: u64 = 6;
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

    // Try to create a TEE
    let mut next_page = (mem_range.base() + mem_range.size() / 2) & !0x3fff;
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCreate(next_page));
    let vmid = ecall_send(&msg).expect("Tellus - TvmCreate returned error");
    println!("Tellus - TvmCreate Success vmid: {vmid:x}");
    next_page += PAGE_SIZE_4K * NUM_TEE_CREATE_PAGES;

    // Add pages for the page table
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPageTablePages {
        guest_id: vmid,
        page_addr: next_page,
        num_pages: NUM_TEE_PTE_PAGES,
    });
    ecall_send(&msg).expect("Tellus - AddPageTablePages returned error");
    next_page += PAGE_SIZE_4K * NUM_TEE_PTE_PAGES;

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
    ecall_send(&msg).expect("Tellus - AddPages returned error");

    let msg = SbiMessage::Measurement(sbi::MeasurementFunction::GetSelfMeasurement {
        measurement_version: 1,
        measurement_type: 1,
        page_addr: measurement_page_addr,
    });

    match ecall_send(&msg) {
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
        page_addr: measurement_page_addr,
    });

    match ecall_send(&msg) {
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
    ecall_send(&msg).expect("Tellus - AddPages Zeroed returned error");

    // TODO test that access to pages crashes somehow

    let msg = SbiMessage::Tee(sbi::TeeFunction::Finalize { guest_id: vmid });
    ecall_send(&msg).expect("Tellus - Finalize returned error");

    let msg = SbiMessage::Tee(sbi::TeeFunction::Run { guest_id: vmid });
    match ecall_send(&msg) {
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
    ecall_send(&msg).expect("Tellus - RemovePages returned error");

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmDestroy { guest_id: vmid });
    ecall_send(&msg).expect("Tellus - TvmDestroy returned error");

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
