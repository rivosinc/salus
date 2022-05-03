// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::alloc::Allocator;
use device_tree::DeviceTree;
use once_cell::unsync::OnceCell;
use spin::Mutex;

/// Holds static global information about the CPU we're running on, such as which ISA extensions
/// are supported.
#[derive(Debug, Default)]
pub struct Cpu {
    has_sstc: bool,
    // TODO: Add any other features that we should enumerate.
}

static CPU: Mutex<OnceCell<Cpu>> = Mutex::new(OnceCell::new());

impl Cpu {
    /// Initializes the global `Cpu` state from the a device-tree. Must be called first before
    /// any of the methods below which read `Cpu` are called. Panics if the device-tree is
    /// malformed (missing CPU nodes or expected properties).
    pub fn parse_features_from<A: Allocator + Clone>(dt: &DeviceTree<A>) {
        // Find the ISA string from the CPU node in the device-tree.
        let cpu_node = dt
            .iter()
            .find(|n| {
                n.props()
                    .any(|p| (p.name() == "device_type") && (p.value_str().unwrap_or("") == "cpu"))
            })
            .expect("No CPU node in device-tree");
        let isa_string = cpu_node
            .props()
            .find(|p| p.name() == "riscv,isa")
            .expect("No 'riscv,isa' property in device-tree")
            .value_str()
            .unwrap();
        let cpu = Cpu {
            has_sstc: isa_string.split('_').any(|f| f == "sstc"),
        };
        CPU.lock().set(cpu).unwrap();
    }

    /// Returns true if the Sstc extension is supported.
    pub fn has_sstc() -> bool {
        CPU.lock()
            .get()
            .expect("Cpu features not initialized")
            .has_sstc
    }
}
