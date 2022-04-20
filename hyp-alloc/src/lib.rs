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

extern crate alloc;

pub mod hyp_alloc;

pub use crate::hyp_alloc::HypAlloc;

#[cfg(test)]
#[macro_use]
extern crate std;
