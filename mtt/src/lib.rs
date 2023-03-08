// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! This crate provides types and API function related to the Memory Translation Table (MTT).
//! The MTT is maintained by the TSM (Salus), and is used by hardware to distinguish between
//! confidential and non-confidential memory ranges.
#![no_std]

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;
