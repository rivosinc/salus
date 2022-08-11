// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

mod core;
mod error;
mod registers;

pub use self::core::Iommu;
pub use error::Error as IommuError;
pub use error::Result as IommuResult;
