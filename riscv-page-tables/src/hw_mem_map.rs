// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{AlignedPageAddr, PageSize4k};

/// Represents the raw system memory map. Owns all system memory, its unsafe constructor
/// should only be called once as this struct must be the unique owner of the memory pointed to by
/// the map. `HwMemMap` is used as the foundation of safely assigning memory ownership of system
/// RAM, configuring it correctly is _critical_ to the safety of the system.
pub struct HwMemMap {
    ram_base: AlignedPageAddr<PageSize4k>,
    ram_size: u64,
    usable_ram_base: AlignedPageAddr<PageSize4k>,
}

impl HwMemMap {
    /// Creates the system memory map from the given parameters"
    ///
    /// - ram_base: The first address accessible, normally 0x8000_0000 on riscv.
    /// - ram_size: The amount of system memory available in bytes.
    /// - usable_ram_base - the first address of usable system memory. This is ram that can be used
    /// by the hypervisor. It must be an address past the code and stack of the running program.
    ///
    /// # Safety
    ///
    /// Memory from the first page through ram_size must be uniquely owned and never referenced
    /// until taken out of HwMemMap by converting it to usable memory.
    /// `HwMemMap` must be unique and have complete ownership of all memory from `usable_ram_base`
    /// to the end of memory(`ram_size`).
    pub unsafe fn new(
        ram_base: AlignedPageAddr<PageSize4k>,
        ram_size: u64,
        usable_ram_base: AlignedPageAddr<PageSize4k>,
    ) -> Self {
        Self {
            ram_base,
            ram_size,
            usable_ram_base,
        }
    }

    /// Returns the base address of system RAM.
    pub fn ram_base(&self) -> AlignedPageAddr<PageSize4k> {
        self.ram_base
    }

    /// Returns the size of system Memory.
    pub fn ram_size(&self) -> u64 {
        self.ram_size
    }

    /// Returns the address of the first usable RAM page. This is the first page beyond those used
    /// for the hypervisor code, device tree, and stack.
    pub fn usable_ram_base(&self) -> AlignedPageAddr<PageSize4k> {
        self.usable_ram_base
    }
}
