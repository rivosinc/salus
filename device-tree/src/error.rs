// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use alloc::collections::TryReserveError;
use core::{fmt, result, str};
use fdt_rs::error::DevTreeError;

/// Errors returned from the `device-tree` crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Error parsing a device tree string.
    StrError(str::Utf8Error),
    /// The given node id wasn't found.
    InvalidNodeId,
    /// The passed FDT isn't a proper tree(more than one node without a parent).
    MalformedFdt,
    /// The given property isn't found in the node.
    PropNotFound,
    /// Propagated error from `fdt_rs` parsing.
    FdtError(DevTreeError),
    /// Couldn't allocate space to store the node/property.
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
            Error::StrError(e) => write!(f, "String error: {e}"),
            Error::InvalidNodeId => write!(f, "Invalid node ID"),
            Error::MalformedFdt => write!(f, "Malformed FDT"),
            Error::PropNotFound => write!(f, "Property not found"),
            Error::FdtError(e) => write!(f, "FDT error: {e}"),
            Error::AllocError(e) => write!(f, "Memory allocation error: {e}"),
        }
    }
}

/// Result from device tree operations.
pub type Result<T> = result::Result<T, Error>;
