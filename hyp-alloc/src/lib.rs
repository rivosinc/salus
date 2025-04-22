// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![feature(allocator_api, slice_ptr_get)]

//! Provides basic allocation ability for the hypervisor.

extern crate alloc;

/// A simple type-safe arena with support for using a custom allocator.
pub mod arena;
/// A simple thread-safe bump-pointer allocator backed by a fixed-length contiguous range of Pages.
pub mod hyp_alloc;

pub use crate::hyp_alloc::HypAlloc;
pub use arena::{Arena, ArenaId};

#[cfg(test)]
#[macro_use]
extern crate std;
