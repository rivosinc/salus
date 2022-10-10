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
mod guest_tracking;
mod host_vm_core;
mod host_vm_loader;
mod smp;
mod trap;
mod tsm_core;
mod vm;
mod vm_cpu;
mod vm_id;
mod vm_pages;
mod vm_pmu;

#[cfg(target_feature = "v")]
use riscv_regs::{sstatus, vlenb, Readable, RiscvCsrInterface, MAX_VECTOR_REGISTER_LEN};
use s_mode_utils::print::*;
use s_mode_utils::sbi_console::SbiConsole;
use tsm_core::*;

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    use host_vm_core::*;
    // Reset CSRs to a sane state.
    host_vm_setup_csrs();

    SbiConsole::set_as_console();
    println!("Salus: Boot test VM");

    // Will panic if register width too long. (currently 256 bits)
    #[cfg(target_feature = "v")]
    check_vector_width();

    host_vm_kernel_init(hart_id, fdt_addr);
    poweroff();
}

#[no_mangle]
#[cfg(not(feature = "salustsm"))]
extern "C" fn secondary_init(hart_id: u64) {
    use host_vm_core::*;
    host_vm_setup_csrs();
    host_vm_secondary_init(hart_id);
    poweroff();
}
