// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

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
}

/// Holds results for IOMMU operations.
pub type Result<T> = core::result::Result<T, Error>;
