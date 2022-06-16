// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![feature(try_reserve_kind)]

//! Type wrappers that are backed by whole pages.
//!
//! When working without an allocator, but with a set of available whole pages, it is useful to to
//! store data in those pages and track them as somewhat-normal rust types.
//!
//! Providing types that closely mirror standard rust containers but are backed by these loaned
//! pages makes the hypervisor code easier to comprehend when coming from a more standard rust
//! codebase.
//!
//! Each page is given to the hypervisor for a specific purpose. For example, a page can be used to
//! hold a guest's state(`PageBox<State>`), or a list of children VMs (`PageVec<Guest>).

extern crate alloc;

/// A Page-backed version of std::sync::Arc.
pub mod page_arc;
/// A Page-backed version of std::collections::Box.
pub mod page_box;
/// A Page-backed version of std::collections::Vec.
pub mod page_vec;
