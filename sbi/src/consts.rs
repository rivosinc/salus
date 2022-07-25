// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(missing_docs, dead_code)]

// Extension constants
pub const EXT_PUT_CHAR: u64 = 0x01;
pub const EXT_BASE: u64 = 0x10;
pub const EXT_HART_STATE: u64 = 0x48534D;
pub const EXT_PMU: u64 = 0x504D55;
pub const EXT_RESET: u64 = 0x53525354;
pub const EXT_TEE: u64 = 0x41544545;
pub const EXT_MEASUREMENT: u64 = 0x5464545;
// TODO Replace the measurement extension once the `GetEvidence` implementation is complete
pub const EXT_ATTESTATION: u64 = 0x41545354; // ATST

pub const SBI_SUCCESS: i64 = 0;
