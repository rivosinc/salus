// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::data_measure::DataMeasure;
use sha2::{Digest, Sha256};

/// The number of bytes in a sha256 digest.
pub const SHA256_DIGEST_BYTES: usize = 32;

/// Maintains a Sha256 measurement of the pages added.
pub struct Sha256Measure {
    measurement: [u8; SHA256_DIGEST_BYTES],
}

impl DataMeasure for Sha256Measure {
    fn add_page(&mut self, gpa: u64, bytes: &[u8]) {
        let mut digest = Sha256::new();
        digest.update(self.measurement);
        digest.update(gpa.to_le_bytes());
        digest.update(bytes);
        self.measurement = digest.finalize().as_slice().try_into().unwrap();
    }

    fn get_measurement(&self) -> &[u8] {
        &self.measurement
    }
}

impl Sha256Measure {
    /// Creates a new, zeroed measurement.
    pub fn new() -> Self {
        Sha256Measure {
            measurement: [0u8; SHA256_DIGEST_BYTES],
        }
    }
}

impl Default for Sha256Measure {
    fn default() -> Self {
        Self::new()
    }
}
