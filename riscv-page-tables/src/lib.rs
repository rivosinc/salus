// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! # Page table management for HS mode on Risc-V.
//!
//! ## Key types
//!
//! - `Page` is the basic building block, representing pages of host supervisor memory. Provided by
//!   the `riscv-pages` crate.
//! - `PageTracker` tracks per-page ownership and typing information, and is used to verify the
//!   safety of page table operations. Provided by the `page-tracking` crate.
//! - `GuestStagePageTable` is a top-level page table structures used to manipulate address translation
//!   and protection.
//! - `PageTable` provides a generic implementation of a single level of multi-level translation.
//! - `Sv48x4`, `Sv48`, etc. define standard RISC-V translation modes for 1st or 2nd-stage translation
//!   tables.
//!
//! ## Safety
//!
//! Safe interfaces are exposed by giving each `GuestStagePageTable` ownership of the pages used to
//! construct the page tables. In this way the pages can be manipulated as needed, but only by the
//! owning page table. The details of managing the pages are contained in the page table.
//!
//! Note that leaf pages mapped into the table are assumed to never be safe to "own", they are
//! implicitly shared with the user of the page table (the entity on the other end of that stage of
//! address translation). Interacting directly with memory currently mapped to a VM will lead to
//! pain so the interfaces don't support that.
#![no_std]

extern crate alloc;

// Include std when running unit tests.
#[cfg(test)]
#[macro_use]
extern crate std;

mod page_table;
/// Provides access to the fields of a riscv PTE.
mod pte;
/// Interfaces to build and manage sv48 page tables for S and U mode access.
mod sv48;
/// Interfaces to build and manage sv48x4 page tables for VMs.
pub mod sv48x4;
/// Priovides stubs for test harnesses.
#[cfg(test)]
mod test_stubs;
/// Provides low-level TLB management functions such as fencing.
pub mod tlb;

pub use page_table::Error as PageTableError;
pub use page_table::Result as PageTableResult;
pub use page_table::{
    FirstStageMapper, FirstStagePageTable, FirstStagePagingMode, GuestStageMapper,
    GuestStagePageTable, GuestStagePagingMode, PagingMode, ENTRIES_PER_PAGE,
};
pub use pte::{PteFieldBits, PteLeafPerms};
pub use sv48::Sv48;
pub use sv48x4::Sv48x4;
