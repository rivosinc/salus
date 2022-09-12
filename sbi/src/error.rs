// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Errors passed over the SBI protocol.
///
/// Constants from the SBI [spec](https://github.com/riscv-non-isa/riscv-sbi-doc/releases).
#[repr(i64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Generic failure in execution of the SBI call.
    Failed = -1,
    /// Extension or function is not supported.
    NotSupported = -2,
    /// Parameter passed isn't valid.
    InvalidParam = -3,
    /// Permission denied.
    Denied = -4,
    /// Address passed is invalid.
    InvalidAddress = -5,
    /// The given hart has already been started.
    AlreadyAvailable = -6,
    /// Some of the given counters have already been started.
    AlreadyStarted = -7,
    /// Some of the given counters have already been stopped.
    AlreadyStopped = -8,
    /// The buffer passed as a parameter is not large enough.
    InsufficientBufferCapacity = -9,
}

impl Error {
    /// Parse the given error code to an `Error` enum.
    pub fn from_code(e: i64) -> Self {
        use Error::*;
        match e {
            -1 => Failed,
            -2 => NotSupported,
            -3 => InvalidParam,
            -4 => Denied,
            -5 => InvalidAddress,
            -6 => AlreadyAvailable,
            -7 => AlreadyStarted,
            -8 => AlreadyStopped,
            -9 => InsufficientBufferCapacity,
            _ => Failed,
        }
    }
}

/// Holds the result of a TEE operation.
pub type Result<T> = core::result::Result<T, Error>;
