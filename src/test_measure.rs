// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::data_measure::DataMeasure;
use riscv_pages::{Page, PageSize};
use sha2::{Digest, Sha256};
#[derive(Default)]
pub struct TestMeasure {
    measurement: [u8; 32],
}

impl DataMeasure for TestMeasure {
    fn add_page<S: PageSize>(&mut self, gpa: u64, page: &Page<S>) {
        let mut digest = Sha256::new();
        digest.update(self.measurement);
        digest.update(gpa.to_le_bytes());
        digest.update(page.as_bytes());
        self.measurement = digest.finalize().as_slice().try_into().unwrap();
    }

    fn get_measurement(&self) -> &[u8] {
        &self.measurement
    }
}

impl TestMeasure {
    pub fn new() -> Self {
        TestMeasure {
            measurement: [0u8; 32],
        }
    }
}
