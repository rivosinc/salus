// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Host interfaces for confidential computing.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub mod tsm;
