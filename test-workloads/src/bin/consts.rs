// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Not all constants are used in both binaries.
#![allow(dead_code)]

/// Constants defining the layout of the Tellus & GuestVM address space.
///
/// The composite Tellus + Guest VM image has this layout in physical memory:
///
/// +-------------------------+ 0x8020_0000
/// | Tellus image            |
/// +-------------------------+ +NUM_TELLUS_IMAGE_PAGES
/// | Guest image             |
/// +-------------------------+
///
/// The guest VM's address space is constructed to look like this:
///
/// +-------------------------+ 0x1000_0000
/// | MMIO (4kB)              |
/// |-------------------------|
/// | <empty>                 |
/// |-------------------------| 0x2800_0000
/// | IMSIC (1MB)             |
/// |-------------------------|
/// | <empty>                 |
/// |-------------------------| 0x8020_0000
/// | Guest image             |
/// |-------------------------| +NUM_GUEST_DATA_PAGES
/// | Guest zero pages        |
/// |-------------------------| +NUM_GUEST_ZERO_PAGES
/// | <empty>                 |
/// |-------------------------| 0x1_0000_0000
/// | Shared pages            |
/// +-------------------------+ +NUM_GUEST_SHARED_PAGES

pub const PAGE_SIZE_4K: u64 = 4096;
pub const NUM_TELLUS_IMAGE_PAGES: u64 = 32;
pub const GUEST_MMIO_START_ADDRESS: u64 = 0x1000_8000;
pub const GUEST_MMIO_END_ADDRESS: u64 = GUEST_MMIO_START_ADDRESS + PAGE_SIZE_4K - 1;
pub const IMSIC_START_ADDRESS: u64 = 0x2800_0000;
pub const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
pub const NUM_GUEST_DATA_PAGES: u64 = 160;
pub const GUEST_ZERO_PAGES_START_ADDRESS: u64 =
    USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
pub const NUM_GUEST_ZERO_PAGES: u64 = 10;
pub const PRE_FAULTED_ZERO_PAGES: u64 = 2;
pub const GUEST_ZERO_PAGES_END_ADDRESS: u64 =
    GUEST_ZERO_PAGES_START_ADDRESS + NUM_GUEST_ZERO_PAGES * PAGE_SIZE_4K - 1;
pub const GUEST_SHARED_PAGES_START_ADDRESS: u64 = 0x1_0000_0000;
pub const NUM_GUEST_SHARED_PAGES: u64 = 1;
pub const GUEST_SHARED_PAGES_END_ADDRESS: u64 =
    GUEST_SHARED_PAGES_START_ADDRESS + NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K - 1;
pub const GUEST_SHARE_PING: u64 = 0xBAAD_F00D;
pub const GUEST_SHARE_PONG: u64 = 0xF00D_BAAD;
