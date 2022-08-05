// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::SupervisorPageAddr;

use super::address::{Address, Bus};
use super::device::HeaderType;
use super::resource::PciResourceType;

/// Errors resulting from interacting with PCI devices.
#[derive(Clone, Copy, Debug)]
pub enum Error {
    /// The PCI configuration space provided by device tree isn't aligned to 4k.
    ConfigSpaceMisaligned(u64),
    /// The PCI configuration size provided by device tree isn't divisible by 4k.
    ConfigSpaceNotPageMultiple(u64),
    /// A PCI BAR resource provided by the device tree isn't aligned to 4k.
    ResourceMisaligned(u64),
    /// A PCI BAR resource provided by the device tree isn't divisible by 4k.
    ResourceNotPageMultiple(u64),
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
    /// Multiple PCI resources of the given type were specified in the device tree.
    DuplicateResource(PciResourceType),
    /// Attempt to claim a resource that has already been taken.
    ResourceTaken,
    /// The device tree provided an invalid bus number in the `bus-range` property.
    InvalidBusNumber(u32),
    /// No 'msi-parent' device tree property was specified in the device tree.
    MissingMsiParent,
    /// The 'msi-parent' property did not refer to an IMSIC.
    InvalidMsiParent,
    /// Invalid value in a PCI header at `address`.
    UnsupportedHeaderType(Address, HeaderType),
    /// Bus is not within the bounds of a config space.
    OutOfBoundsBusNumber(Bus),
    /// Failed to allocate memory.
    AllocError,
    /// Ran out of bus numbers while assigning buses.
    OutOfBuses,
    /// Unsupported size or alignment in emulated config space access.
    UnsupportedConfigAccess,
    /// Offset in emulated config space is invalid.
    InvalidConfigOffset,
    /// The device targetted by the emulated config space access does not exist.
    DeviceNotFound(Address),
    /// Too many capabilities were found for a PCI device.
    TooManyCapabilities,
    /// The device has MSI support, but is not 64-bit capable.
    MsiNot64BitCapable,
    /// The device has a vendor capability structure with an invalid length field.
    InvalidVendorCapabilityLength(usize),
    /// The device has an unsupported PCI Express capability version.
    UnsupportedExpressCapabilityVersion(u8),
    /// The PCI Express device has an unsupported/unknwon device type.
    UnsupportedExpressDevice(u8),
    /// The device has a 64-bit BAR at an odd-numbered index.
    Invalid64BitBarIndex,
    /// The device has a non-power-of-2 sized BAR.
    InvalidBarSize(u64),
    /// A BAR or bridge window is programmed with an invalid address.
    InvalidBarAddress(u64),
    /// A VM has programmed a BAR or bridge window to cover a page it does not own.
    UnownedBarPage(SupervisorPageAddr),
    /// The PCI device is already owned.
    DeviceOwned,
    /// The PCI device is not owned, or is owned by another VM.
    DeviceNotOwned,
}

/// Holds results for PCI operations.
pub type Result<T> = core::result::Result<T, Error>;
