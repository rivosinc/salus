// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// A TLB version number.
///
/// We use TLB versions to track the progress of a coordinated TLB shootdown across multiple CPUs.
/// A shootdown is considered completed if the TLB version at which the shootdown was initiated is
/// older (less than) the current TLB version.
///
/// TODO: Overflow. A simple '!=' check against the current version should be sufficient since a
/// page cannot have been invalidated at a future version. Let's do the more defensive thing for now
/// though given how long it'll take a 64-bit counter to overflow.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TlbVersion(u64);

impl TlbVersion {
    /// Creates a new TLB version number, starting from 0.
    pub fn new() -> Self {
        TlbVersion(0)
    }

    /// Increments this TLB version number.
    pub fn increment(self) -> Self {
        TlbVersion(self.0 + 1)
    }
}
