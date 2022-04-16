// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{Page, PageSize};
use crate::data_measure::DataMeasure;
use sha2::{Digest, Sha256};
#[derive(Default)]
pub struct TestMeasure {
    measurement: [u8; 32],
}

impl DataMeasure for TestMeasure {
    type MeasurementResult = [u8; 32];

    fn add_page<S: PageSize>(&mut self, gpa: u64, page: &Page<S>) {
        self.measurement = Sha256::digest(page.as_bytes()).as_slice().try_into().unwrap();
    }

    fn get_measurement(&self) -> Self::MeasurementResult {
        self.measurement
    }
}
