// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! # Hardware drivers
//!
//! - `CpuInfo` holds the topology and static properties  of the CPUs the hypervisor is running on.
#![no_std]
#![feature(allocator_api)]

extern crate alloc;

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;

pub mod cpu;

pub use cpu::{CpuId, CpuInfo, MAX_CPUS};
