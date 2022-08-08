// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

mod error;
mod main;
mod registers;

pub use error::Error as IommuError;
pub use error::Result as IommuResult;
pub use main::Iommu;
