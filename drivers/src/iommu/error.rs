// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::SupervisorPageAddr;

use super::device_directory::{DeviceId, GscId};
use crate::imsic::ImsicLocation;
use crate::pci::{Address, PciError};

/// Errors resulting from interacting with the IOMMU.
#[derive(Clone, Copy, Debug)]
pub enum Error {
    /// Error encountered while probing and enabling the IOMMU PCI device.
    ProbingIommu(PciError),
    /// Couldn't find the IOMMU registers BAR.
    MissingRegisters,
    /// Unexpected IOMMU register set size.
    InvalidRegisterSize(u64),
    /// IOMMU register set is misaligned.
    MisalignedRegisters,
    /// Missing required G-stage translation support.
    MissingGStageSupport,
    /// Missing required MSI translation support.
    MissingMsiSupport,
    /// Missing A/D update support.
    MissingAmoHwadSupport,
    /// Not enough pages were supplied to create an MSI page table.
    InsufficientMsiTablePages,
    /// The supplied MSI page table pages were not properly aligned.
    MisalignedMsiTablePages,
    /// Ownership mismatch in MSI page table pages.
    UnownedMsiTablePages,
    /// Attempt to map an invalid IMSIC location in an MSI page table.
    InvalidImsicLocation(ImsicLocation),
    /// The destination of an MSI page table mapping is not owned by the VM.
    MsiPageNotOwned(SupervisorPageAddr),
    /// The MSI page table entry is already mapped.
    MsiAlreadyMapped(ImsicLocation),
    /// The MSI page table entry is not mapped.
    MsiNotMapped(ImsicLocation),
    /// Failed to allocate a page.
    OutOfPages,
    /// Got a leaf entry when a non-leaf entry was expected.
    NotIntermediateTable,
    /// Unable to map a PCI BDF address to an IOMMU device ID.
    PciAddressTooLarge(Address),
    /// Mismatch between page table and device ownership.
    OwnerMismatch,
    /// No device context found.
    DeviceNotFound(DeviceId),
    /// The device already has an active device context.
    DeviceAlreadyEnabled(DeviceId),
    /// The device does not have an active device context.
    DeviceNotEnabled(DeviceId),
    /// The head/tail pointer is out of bounds for the queue.
    InvalidQueuePointer(usize),
    /// No more elements can be pushed to the queue.
    QueueFull,
    /// No elements are available to be popped from the queue.
    QueueEmpty,
    /// Ran out of available GSCIDs.
    OutOfGscIds,
    /// The supplied GSCID was invalid.
    InvalidGscId(GscId),
    /// Attempted to free a GSCID that's not currently allocated.
    GscIdAlreadyFree(GscId),
    /// Attempted to free a GSCID that's currently being used for translation.
    GscIdInUse(GscId),
    /// MSI translation not supported.
    MsiTranslationUnsupported,
    /// No feasible device directory mode.
    DeviceDirectoryUnsupported,
}

/// Holds results for IOMMU operations.
pub type Result<T> = core::result::Result<T, Error>;
