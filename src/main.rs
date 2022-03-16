// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};

extern crate alloc;

mod abort;
mod asm;
mod print_util;

use abort::abort;
use print_util::*;

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

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, _fdt_addr: u64) {
    if hart_id != 0 {
        // TODO handle more than 1 cpu
        abort();
    }

    // Safety: This is the very beginning of the kernel, there are no other users of the UART or the
    // CONSOLE_DRIVER global.
    unsafe { UartDriver::new(0x1000_0000).use_as_console() }
    println!("Salus: start");

    println!("Salus: exiting");

    poweroff();
}
