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
pub const EXT_DBCN: u64 = 0x4442434E; // DBCN
pub const EXT_ATTESTATION: u64 = 0x41545354; // ATST
pub const EXT_TEE_HOST: u64 = 0x54454548; // TEEH
pub const EXT_TEE_INTERRUPT: u64 = 0x54454549; // TEEI
pub const EXT_TEE_GUEST: u64 = 0x54454547; // TEEG

pub const SBI_SUCCESS: i64 = 0;
pub const SBI_ERR_INVALID_ADDRESS: i64 = -5;
