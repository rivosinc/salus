// Copyright (c) 2023 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use attestation::MSMT_REGISTERS;
use data_model::DataInit;

/// CDI ID length.
pub const CDI_ID_LEN: usize = 20;
/// Length of a SHA384 hash.
pub const SHA384_LEN: usize = 48;

/// Compound Device Identifier (CDI) ID type.
pub type CdiId = [u8; CDI_ID_LEN];
/// Measurement registers for the Sha384 case.
pub type MeasurementRegisterSha384 = [u8; SHA384_LEN];

/// Structure passed with `GetEvidence` in the Umode Shared Region.
/// Represents the status of the DICE layer needed to generate a
/// certificate.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GetEvidenceShared {
    /// Measurement registers in SHA-384. In `fwid` order.
    pub msmt_regs: [MeasurementRegisterSha384; MSMT_REGISTERS],
    /// CDI Id.
    pub cdi_id: CdiId,
}

// Safety: `GetEvidenceShared` is a POD struct without implicit padding and therefore can be
// initialized from a byte array.
unsafe impl DataInit for GetEvidenceShared {}
