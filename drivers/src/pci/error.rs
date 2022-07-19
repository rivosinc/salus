// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use super::address::{Address, Bus};
use super::header::HeaderType;

/// Errors resulting from interacting with PCI devices.
#[derive(Clone, Copy, Debug)]
pub enum Error {
    /// The PCI configuration space provided by device tree isn't aligned to 4k.
    ConfigSpaceMisaligned(u64),
    /// The PCI configuration size provided by device tree isn't divisible by 4k.
    ConfigSpaceNotPageMultiple(u64),
    /// A PCI BAR resource provided by the device tree isn't aligned to 4k.
    BarSpaceMisaligned(u64),
    /// A PCI BAR resource provided by the device tree isn't divisible by 4k.
    BarSpaceNotPageMultiple(u64),
    /// The device tree contained an MMIO region that overlaps with other memory regions.
    InvalidMmioRegion(page_tracking::MemMapError),
    /// The device tree entry for the PCI host didn't provide a base register for Configuration
    /// Space.
    NoConfigBase,
    /// No compatible PCI host controller found in the device tree.
    NoCompatibleHostNode,
    /// The device tree entry for the PCI host didn't provide a size register for Configuration
    /// Space.
    NoConfigSize,
    /// The device tree entry for the PCI host didn't provide a `reg` property.
    NoRegProperty,
    /// The device tree entry for the PCI host didn't provide a `ranges` property.
    NoRangesProperty,
    /// Too many PCI resource ranges were specified in the device tree's `ranges` property.
    TooManyBarSpaces,
    /// The device tree provided an invalid bus number in the `bus-range` property.
    InvalidBusNumber(u32),
    /// Invalid value in a PCI header at `address`.
    UnsupportedHeaderType(Address, HeaderType),
    /// Bus is not within the bounds of a config space.
    OutOfBoundsBusNumber(Bus),
    /// Failed to allocate memory.
    AllocError,
}

/// Holds results for PCI operations.
pub type Result<T> = core::result::Result<T, Error>;
