// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

//! Trait for measurement and implementations for different algorithms.

/// Base trait for measuring pages as they are added to VMs.
pub mod data_measure;
/// A Sha256-based implementation of DataMeasure.
pub mod sha256;
