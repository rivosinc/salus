// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::CpuId;

/// Errors that can be returned by the IMSIC driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The IMSIC node was not present in the device tree.
    MissingImsicNode,
    /// The specified property was missing from the IMSIC device tree node.
    MissingProperty(&'static str),
    /// Unexpected number of parent interrupts specified in the device tree.
    InvalidParentInterruptCount(usize),
    /// Invalid number of guest files per hart specified in the device tree.
    InvalidGuestsPerHart(usize),
    /// Invalid group index shift specified in the device tree.
    InvalidGroupIndexShift(u32),
    /// Unexpected number of MMIO regions specified in the device tree.
    InvalidMmioRegionCount(usize),
    /// Misaligned MMIO region specified in the device tree.
    MisalignedMmioRegion(u64),
    /// Invalid parent interrupt specification in the device tree.
    InvalidParentInterrupt(u32, u32),
    /// There were more interrupt files than CPUs found in the device tree.
    TooManyInterruptFiles,
    /// There were fewer interrupt files than CPUs found in the device tree.
    MissingInterruptFiles,
    /// A group base address did not match the expected pattern.
    InvalidGeometry,
    /// Failed to add an MMIO region to the system memory map.
    AddingMmioRegion(page_tracking::MemMapError),
    /// The requested CPU does not exist.
    InvalidCpu(CpuId),
    /// No guest file for the specified guest.
    InvalidGuestFile,
    /// Guest file for this guest already taken.
    GuestFileTaken,
    /// Attempt to free a guest file that's not taken.
    GuestFileFree,
}

/// Holds the result of IMSIC operations.
pub type Result<T> = core::result::Result<T, Error>;
