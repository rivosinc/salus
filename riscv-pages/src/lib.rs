// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # RiscV page types
//!
//! - `Page` is the basic building block, representing pages of host supervisor memory.
#![no_std]

mod page;
mod sequential_pages;

pub use page::{
    Page, Page4k, PageAddr, PageAddr4k, PageSize, PageSize1GB, PageSize2MB, PageSize4k,
    PageSize512GB, Pfn, PhysAddr, PhysPage, UnmappedPage,
};
pub use sequential_pages::SequentialPages;
