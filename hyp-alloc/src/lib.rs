// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![feature(
    allocator_api,
    nonnull_slice_from_raw_parts,
    slice_ptr_get,
    slice_ptr_len
)]

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
