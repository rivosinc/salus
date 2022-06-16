// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Holds a page measurement and allows updating that measurement when adding pages.
pub trait DataMeasure {
    /// Updates the current measurement to include the contents of `page`.
    fn add_page(&mut self, gpa: u64, page: &[u8]);
    /// Returns the current measurement.
    fn get_measurement(&self) -> &[u8];
}
