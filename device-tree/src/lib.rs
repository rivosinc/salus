// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Library for interacting wtih device-trees.
#![no_std]
#![feature(allocator_api, split_array)]

extern crate alloc;

mod device_tree;
mod error;
mod fdt;
mod serialize;

pub use crate::device_tree::{DeviceTree, DeviceTreeIter, DeviceTreeNode};
pub use error::Error as DeviceTreeError;
pub use error::Result as DeviceTreeResult;
pub use fdt::{Cpu, Fdt, FdtMemoryRegion, ImsicInfo};
pub use serialize::DeviceTreeSerializer;
