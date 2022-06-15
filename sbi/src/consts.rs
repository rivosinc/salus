// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(missing_docs, dead_code)]

// Extension constants
pub const EXT_PUT_CHAR: u64 = 0x01;
pub const EXT_BASE: u64 = 0x10;
pub const EXT_HART_STATE: u64 = 0x48534D;
pub const EXT_RESET: u64 = 0x53525354;
pub const EXT_TEE: u64 = 0x544545;
pub const EXT_MEASUREMENT: u64 = 0x5464545;

// Error constants from the sbi [spec](https://github.com/riscv-non-isa/riscv-sbi-doc/releases)
pub const SBI_SUCCESS: i64 = 0;
pub const SBI_ERR_FAILED: i64 = -1;
pub const SBI_ERR_NOT_SUPPORTED: i64 = -2;
pub const SBI_ERR_INVALID_PARAM: i64 = -3;
pub const SBI_ERR_DENIED: i64 = -4;
pub const SBI_ERR_INVALID_ADDRESS: i64 = -5;
pub const SBI_ERR_ALREADY_AVAILABLE: i64 = -6;
pub const SBI_ERR_ALREADY_STARTED: i64 = -7;
pub const SBI_ERR_ALREADY_STOPPED: i64 = -8;
