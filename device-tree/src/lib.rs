// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Library for interacting wtih device-trees.
#![no_std]
#![feature(allocator_api, split_array, slice_take)]

extern crate alloc;

mod device_tree;
mod error;
mod fdt;
mod serialize;

pub use crate::device_tree::{DeviceTree, DeviceTreeIter, DeviceTreeNode};
pub use error::Error as DeviceTreeError;
pub use error::Result as DeviceTreeResult;
pub use fdt::{Fdt, FdtMemoryRegion};
pub use serialize::DeviceTreeSerializer;
