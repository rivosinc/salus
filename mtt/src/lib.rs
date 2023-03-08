// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! This crate provides types and API function related to the Memory Translation Table (MTT).
//! The MTT is maintained by the TSM (Salus), and is used by hardware to distinguish between
//! confidential and non-confidential memory ranges.
#![no_std]
#![feature(if_let_guard)]
// For testing purposes
#![feature(rustc_private)]

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
extern crate libc;

/// Types related to MTT functionality.
pub mod mtt;
