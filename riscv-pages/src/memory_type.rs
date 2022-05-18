// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt;

/// Describes the type of memory a page represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemType {
    /// Ordinary, idempotent system RAM.
    Ram,

    /// Memory-mapped IO. Reads and writes may have side-effects.
    Mmio(DeviceMemType),
}

/// Identifies the class of device a page of MMIO represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMemType {
    /// An IMSIC interrupt file page.
    Imsic,
    // TODO: Add more types here.
}

impl fmt::Display for MemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match &self {
            MemType::Ram => write!(f, "RAM"),
            MemType::Mmio(d) => write!(f, "MMIO ({}", d),
        }
    }
}

impl fmt::Display for DeviceMemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match &self {
            DeviceMemType::Imsic => write!(f, "IMSIC"),
        }
    }
}
