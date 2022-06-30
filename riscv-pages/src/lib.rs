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

mod memory_type;
mod page;
mod page_owner_id;
mod sequential_pages;
mod state;

pub use memory_type::{DeviceMemType, MemType};
pub use page::*;
pub use page_owner_id::{
    AddressSpace, GuestPhys, GuestVirt, PageOwnerId, SupervisorPhys, SupervisorVirt,
};
pub use sequential_pages::{Error as SequentialPagesError, SeqPageIter, SequentialPages};
pub use state::*;
