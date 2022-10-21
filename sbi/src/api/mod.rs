// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Debug Console for printing strings through SBI.
pub mod debug_console;

/// Host interfaces for reset extension.
pub mod reset;

/// Host interfaces for hart state management.
pub mod state;

/// Host interfaces for confidential computing.
pub mod tee_host;

/// Host interfaces for confidential computing interrupt virtualization.
pub mod tee_interrupt;

/// Guest interfaces for confidential computing.
pub mod tee_guest;

/// Host interfaces for PMU.
pub mod pmu;

/// Base SBI inferfaces.
pub mod base;

/// Host interfaces for attestation.
pub mod attestation;
