// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{Page, PageSize};

use crate::data_measure::DataMeasure;

#[derive(Default)]
pub struct TestMeasure {
    measurement: u64,
}

impl DataMeasure for TestMeasure {
    type MeasurementResult = u64;

    fn add_page<S: PageSize>(&mut self, gpa: u64, page: &Page<S>) {
        self.measurement ^= gpa;
        self.measurement ^= page.u64_iter().fold(0, |a, x| a ^ x);
    }

    fn get_measurement(&self) -> Self::MeasurementResult {
        self.measurement
    }
}
