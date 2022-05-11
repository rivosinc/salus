// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{Page, PageSize};

pub trait DataMeasure: Default {
    fn add_page<S: PageSize>(&mut self, gpa: u64, page: &Page<S>);

    fn get_measurement(&self) -> &[u8];
}
