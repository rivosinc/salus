// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

pub trait DataMeasure {
    fn add_page(&mut self, gpa: u64, page: &[u8]);
    fn get_measurement(&self) -> &[u8];
}
