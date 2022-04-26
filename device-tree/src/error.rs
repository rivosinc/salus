// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::collections::TryReserveError;
use core::{fmt, result, str};
use fdt_rs::error::DevTreeError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    StrError(str::Utf8Error),
    InvalidNodeId,
    MalformedFdt,
    PropNotFound,
    FdtError(DevTreeError),
    AllocError(TryReserveError),
}

impl From<str::Utf8Error> for Error {
    fn from(e: str::Utf8Error) -> Self {
        Error::StrError(e)
    }
}

impl From<TryReserveError> for Error {
    fn from(e: TryReserveError) -> Self {
        Error::AllocError(e)
    }
}

impl From<DevTreeError> for Error {
    fn from(e: DevTreeError) -> Self {
        Error::FdtError(e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match self {
            Error::StrError(e) => write!(f, "String error: {}", e),
            Error::InvalidNodeId => write!(f, "Invalid node ID"),
            Error::MalformedFdt => write!(f, "Malformed FDT"),
            Error::PropNotFound => write!(f, "Property not found"),
            Error::FdtError(e) => write!(f, "FDT error: {}", e),
            Error::AllocError(e) => write!(f, "Memory allocation error: {}", e),
        }
    }
}

pub type Result<T> = result::Result<T, Error>;
