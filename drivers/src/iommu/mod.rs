// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

mod core;
mod error;
mod msi_page_table;
mod registers;

pub use self::core::Iommu;
pub use error::Error as IommuError;
pub use error::Result as IommuResult;
pub use msi_page_table::MsiPageTable;
