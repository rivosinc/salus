// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # Page ownership tracking
//!
//! ## Key types
//!
//! - `Page` is the basic building block, representing pages of host supervisor memory. Provided by
//!   the `riscv-pages` crate.
//! - `HwMemMap` - Map of system memory, used to determine address ranges to create `Page`s from.
//! - `HypPageAlloc` - Initial manager of physical memory. The hypervisor allocates pages from
//! here to store local state. It's turned in to a `PageTracker` and a pool of ram for the host VM.
//! - `PageTracker` - Contains information about active VMs (page owners), manages allocation of
//! unique owner IDs, and per-page state such as current and previous owner. This is system-wide
//! state updated whenever a page owner changes or a VM starts or stops.
//!
//! ## Initialization
//!
//! `HwMemMap` -> `HypPageAlloc` ---> `PageTracker`
//!                                 \
//!                                  -------> `SequentialPages` for hypervisor setup

#![no_std]
#![feature(allocator_api, try_reserve_kind, let_chains)]

extern crate alloc;

/// `Page`-backed collections resembling those in the standard library.
pub mod collections;
mod hw_mem_map;
mod page_info;
/// Implements a linked-list of pages using `PageTracker`.
pub mod page_list;
/// Handles tracking the owner and state of each page.
pub mod page_tracker;
/// Implements a `TlbVersion` type, used for tracking the progress of TLB shootdowns.
pub mod tlb_version;

pub use hw_mem_map::Error as MemMapError;
pub use hw_mem_map::Result as MemMapResult;
pub use hw_mem_map::{HwMemMap, HwMemMapBuilder, HwMemRegion, HwMemRegionType, HwReservedMemType};
pub use page_info::MAX_PAGE_OWNERS;
pub use page_list::{LockedPageList, PageList};
pub use page_tracker::Error as PageTrackingError;
pub use page_tracker::Result as PageTrackingResult;
pub use page_tracker::{HypPageAlloc, PageTracker};
pub use tlb_version::TlbVersion;

#[cfg(test)]
#[macro_use]
extern crate std;
