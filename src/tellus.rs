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

    // Safe becasue we trust the host to boot with a valid fdt_addr pass in register a1.
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
    next_page += 4096 * 6;

    // Add pages for the page table
    let num_pte_pages = 10;
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPageTablePages {
        guest_id: vmid,
        page_addr: next_page,
        num_pages: num_pte_pages,
    });
    ecall_send(&msg).expect("Tellus - AddPageTablePages returned error");
    next_page += 4096 * num_pte_pages;

    let first_guest_page = next_page;

    // write junk to some memory given to guest, make sure it's zeroed later.
    unsafe {
        // not safe, but it's a test and no one uses this memory.
        let m = first_guest_page as *mut u64;
        *m = 0x5446_5446_5446_5446;
        let m = (first_guest_page + 0x6000) as *mut u64;
        *m = 0x5446_5446_5446_5446;
    };

    // Add data pages
    let num_data_pages = 10;
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPages {
        guest_id: vmid,
        page_addr: first_guest_page,
        page_type: 0,
        num_pages: num_data_pages,
        gpa: 0x8000_0000,
        measure_preserve: true,
    });
    ecall_send(&msg).expect("Tellus - AddPages returned error");

    // Add zeroed (non-measured) pages
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPages {
        guest_id: vmid,
        page_addr: first_guest_page + num_data_pages * 0x1000,
        page_type: 0,
        num_pages: num_data_pages,
        gpa: 0x8000_0000 + num_data_pages * 0x1000,
        measure_preserve: false,
    });
    ecall_send(&msg).expect("Tellus - AddPages Zeroed returned error");

    // TODO test that access to pages crashes somehow

    let msg = SbiMessage::Tee(sbi::TeeFunction::Finalize { guest_id: vmid });
    ecall_send(&msg).expect("Tellus - Finalize returned error");

    let num_remove_pages = num_data_pages / 2;
    let msg = SbiMessage::Tee(sbi::TeeFunction::RemovePages {
        guest_id: vmid,
        gpa: 0x8000_0000 + num_remove_pages * 0x1000,
        remap_addr: first_guest_page + num_remove_pages * 0x1000,
        num_pages: 5,
    });
    ecall_send(&msg).expect("Tellus - RemovePages returned error");

    // check that junk has been cleared on removed memory
    let read_res = unsafe {
        // not safe, but it's a test and no one uses this memory.
        let m = (first_guest_page + (num_remove_pages + 1) * 0x1000) as *mut u64;
        core::ptr::read_volatile(m)
    };
    if read_res != 0 {
        panic!(
            "Tellus - Read back non-zero after unmapping from TVM! : {:x}",
            read_res
        );
    }

    /* TODO - need to put code in guest
    let msg = SbiMessage::Tee(sbi::TeeFunction::Run { guest_id: vmid });
    match ecall_send(&msg) {
        Err(e) => println!("Tellus - Run returned error {:?}", e),
        Ok(_) => println!("Tellus - Run success"),
    }
    */

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmDestroy { guest_id: vmid });
    ecall_send(&msg).expect("Tellus - TvmDestroy returned error");

    // check that junk has been cleared on removed memory
    let read_res = unsafe {
        // not safe, but it's a test and no one uses this memory.
        let m = first_guest_page as *mut u64;
        core::ptr::read_volatile(m)
    };
    if read_res != 0 {
        panic!(
            "Tellus - Read back non-zero after exiting from TVM! : {:x}",
            read_res
        );
    }

    println!("Tellus - All OK");

    poweroff();
}
