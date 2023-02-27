// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! # Synchronization primitives.
//!
//! Synchronization primitves like mutexes and read/write-locks that are
//! usable in bare-metal environments. For now, we simply re-export those
//! supported by the `spin` crate.
#![no_std]

pub use spin::{Mutex, MutexGuard, Once, RwLock, RwLockReadGuard, RwLockWriteGuard};
