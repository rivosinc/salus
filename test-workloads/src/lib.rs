// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! This crate provides test workloads to exercise the salus hypervisor and the Confidential
//! computing API.

#![no_std]

use s_mode_utils::abort::abort;
use s_mode_utils::print_sbi::*;

mod asm;

/// Panics the running test workload.
#[panic_handler]
pub fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    abort();
}
