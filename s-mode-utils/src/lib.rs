// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Provides basic functions for code running in S-mode.

#![no_std]
#![feature(
    panic_info_message,
    allocator_api,
    alloc_error_handler,
    lang_items,
    asm_const
)]

pub use core::alloc::{GlobalAlloc, Layout};

/// Provides the ability to terminate the running S-mode code.
pub mod abort;
/// Supports invoking ecalls in the HS or M mode code above.
pub mod ecall;
/// Implementation of `print` macros.
pub mod print;
/// Console driver using SBI.
pub mod sbi_console;
