// Copyright (c) 2022 by Rivos Inc.
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

use ecall::ecall_send;
use print_sbi::*;
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

#[no_mangle]
#[allow(clippy::zero_ptr)]
extern "C" fn kernel_init() {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const NUM_GUEST_DATA_PAGES: u64 = 10;
    const PAGE_SIZE_4K: u64 = 4096;
    // Safe because this is the very begining of the program and nothing has set the console driver
    // yet.
    unsafe {
        SbiConsoleDriver::new().use_as_console();
    }

    let measurement_page_addr = USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
    let msg = SbiMessage::Measurement(sbi::MeasurementFunction::GetSelfMeasurement {
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });

    println!("*****************************************");
    println!("Hello world from Tellus guest            ");

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

    println!("Exiting guest by causing a fault         ");
    println!("*****************************************");

    // TODO: Implement mechanism to gracefully exit guest
    // Not safe, but deliberately intended to cause a fault
    unsafe {
        core::ptr::read_volatile(0 as *const u64);
    }
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
