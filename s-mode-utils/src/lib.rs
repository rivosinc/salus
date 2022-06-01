// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![feature(
    panic_info_message,
    allocator_api,
    alloc_error_handler,
    lang_items,
    asm_const,
    const_ptr_offset_from
)]

pub use core::alloc::{GlobalAlloc, Layout};

pub mod abort;
pub mod ecall;
pub mod print_sbi;
