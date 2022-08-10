// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

mod core;
mod error;
mod geometry;

pub use self::core::{Imsic, ImsicGuestPage, ImsicGuestPageIter, ImsicInterruptId};
pub use error::Error as ImsicError;
pub use error::Result as ImsicResult;
pub use geometry::*;
