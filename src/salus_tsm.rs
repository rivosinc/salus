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

extern crate alloc;

mod asm;
mod tsm_core;

use core::alloc::{GlobalAlloc, Layout};
use s_mode_utils::abort::abort;
use s_mode_utils::print::*;
use s_mode_utils::sbi_console::SbiConsole;
use tsm_core::*;

// Implementation of GlobalAlloc that forwards allocations to the boot-time allocator.
struct GeneralGlobalAlloc;

unsafe impl GlobalAlloc for GeneralGlobalAlloc {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        abort();
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        abort();
    }
}

#[global_allocator]
static GENERAL_ALLOCATOR: GeneralGlobalAlloc = GeneralGlobalAlloc;

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(_hart_id: u64, _fdt_addr: u64) {
    SbiConsole::set_as_console();
    println!("Salus-TSM: Booting");
    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {
    poweroff();
}
