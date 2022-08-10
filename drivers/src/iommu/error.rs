// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::SupervisorPageAddr;

use crate::imsic::ImsicLocation;
use crate::pci::PciError;

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
}

/// Holds results for IOMMU operations.
pub type Result<T> = core::result::Result<T, Error>;
