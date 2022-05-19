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
    // Safe because this is the very begining of the program and nothing has set the console driver
    // yet.
    unsafe {
        SbiConsoleDriver::new().use_as_console();
    }

    println!("*****************************************");
    println!("Hello world from Tellus guest            ");
    println!("Exiting guest by causing a fault         ");
    println!("*****************************************");

    // TODO: Implement mechanism to gracefully exit guest
    // Not safe, but deliberately intended to cause a fault
    unsafe {
        core::ptr::read_volatile(0 as *const u64);
    }
}
