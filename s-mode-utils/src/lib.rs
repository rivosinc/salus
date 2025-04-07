// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Provides basic functions for code running in S-mode.

#![no_std]
#![feature(allocator_api, alloc_error_handler)]

pub use core::alloc::{GlobalAlloc, Layout};

/// Provides the ability to terminate the running S-mode code.
pub mod abort;
/// Implementation of `print` macros.
pub mod print;
/// Console driver using SBI.
pub mod sbi_console;
