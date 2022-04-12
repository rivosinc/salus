// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Library for interacting wtih device-trees.
#![no_std]

mod fdt;

pub use fdt::{Fdt, Error, Result};
