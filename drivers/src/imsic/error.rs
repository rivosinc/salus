// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::CpuId;

/// Errors that can be returned when claiming or releasing guest interrupt files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
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
