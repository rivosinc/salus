// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # RiscV page types
//!
//! - `Page` is the basic building block, representing pages of host supervisor memory.
#![no_std]

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;

mod page;
mod page_owner_id;
mod sequential_pages;

pub use page::{
    CleanPage, GuestPageAddr, GuestPageAddr4k, GuestPfn, GuestPhysAddr, Page, Page4k, PageAddr,
    PageAddr4k, PageSize, PageSize1GB, PageSize2MB, PageSize4k, PageSize512GB, Pfn, PhysPage,
    RawAddr, SupervisorPageAddr, SupervisorPageAddr4k, SupervisorPfn, SupervisorPhysAddr,
    UnmappedPage,
};
pub use page_owner_id::{AddressSpace, GuestPhys, PageOwnerId, SupervisorPhys};
pub use sequential_pages::{
    Error as SequentialPagesError, SeqPageIter, SequentialPages, SequentialPages4k,
};
