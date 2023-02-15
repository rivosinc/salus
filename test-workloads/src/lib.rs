// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! This crate provides test workloads to exercise the salus hypervisor and the Confidential
//! computing API.

#![no_std]

use s_mode_utils::abort::abort;
use s_mode_utils::print::*;

mod asm;
pub mod consts;

/// Panics the running test workload.
#[panic_handler]
pub fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    abort();
}
