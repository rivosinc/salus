// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Wrapper for basic FDT interaction.

use crate::DeviceTreeResult;
use core::str;
use fdt_rs::base::iters::{DevTreeNodeIter, DevTreeReserveEntryIter};
use fdt_rs::base::parse::ParsedTok;
use fdt_rs::base::{DevTree, DevTreeProp};
use fdt_rs::prelude::*;

/// Represents a flattened device-tree (FDT) as passed to the hypervisor by firmware. Currently
/// this is a lightweight wrapper on top of `fdt_rs::base::DevTree`.
#[derive(Copy, Clone, Debug)]
pub struct Fdt<'a> {
    inner: DevTree<'a>,
}

impl<'a> Fdt<'a> {
    /// Constructs an FDT from the FDT blob located at `fdt_addr`.
    ///
    /// TODO: Most methods assume that the top-level #address-cells/#size-cells == 2. Enforce
    /// this somehow, or handle other values.
    ///
    /// # Safety
    /// Must point to a 32-byte-aligned and valid FDT blob.
    pub unsafe fn new_from_raw_pointer(fdt_addr: *const u8) -> DeviceTreeResult<Self> {
        let inner = DevTree::from_raw_pointer(fdt_addr)?;
        Ok(Self { inner })
    }

    /// Returns the base address of the FDT as a raw pointer.
    pub fn base_addr(&self) -> *const u8 {
        self.inner.buf().as_ptr()
    }

    /// Returns the total size of the FDT.
    pub fn size(&self) -> usize {
        self.inner.totalsize()
    }

    /// Returns an iterator over the memory regions advertised by the FDT in its 'memory' nodes.
    pub fn memory_regions(&self) -> MemoryRegionIter<'_, 'a> {
        MemoryRegionIter::new(self)
    }

    /// Returns an iterator over the memory regions marked as reserved in the FDT.
    pub fn reserved_memory_regions(&self) -> ReservedRegionIter<'_, 'a> {
        ReservedRegionIter::new(self)
    }

    /// Returns the range of memory where the host VM's kernel is loaded, if present.
    pub fn host_kernel_region(&self) -> Option<FdtMemoryRegion> {
        self.get_module_node_region("multiboot,kernel")
    }

    /// Returns the range of memory where the host VM's initramfs is loaded, if present.
    pub fn host_initramfs_region(&self) -> Option<FdtMemoryRegion> {
        self.get_module_node_region("multiboot,ramdisk")
    }

    /// This returns the property buf for the first property with the give name
    pub fn get_property(&self, name: &str) -> Option<&str> {
        let mut iter = self.inner.parse_iter();

        while let Ok(Some(token)) = iter.next() {
            if let ParsedTok::Prop(prop) = token {
                let found_name = match self.inner.string_at_offset(prop.name_offset) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                if found_name.contains(name) {
                    return match str::from_utf8(prop.prop_buf) {
                        Ok(s) => Some(s),
                        Err(_) => continue,
                    };
                };
            }
        }
        None
    }

    pub(crate) fn inner(&self) -> DevTree {
        self.inner
    }

    /// Returns the 'reg' property of a 'multiboot,module' node with the given compatible
    /// string as an `FdtMemoryRegion`.
    ///
    /// TODO: Assumes there's only one (address + length) pair in 'reg' and that the first
    /// compatible string for such nodes is always 'multiboot,module'.
    fn get_module_node_region(&self, compat: &str) -> Option<FdtMemoryRegion> {
        // fdt-rs only matches the first compatible string with compatible_nodes(), which for
        // QEMU-generated FDTs is always 'multiboot,module'. First find those nodes, and then
        // inspect all the strings in 'compatible' property to see if it matches the secondary
        // compatible string.
        let mut modules = self.inner.compatible_nodes("multiboot,module");
        while let Ok(Some(node)) = modules.next() {
            let compat_prop = node
                .props()
                .find(|p| Ok(p.name().unwrap_or("empty") == "compatible"))
                .ok()?;
            let mut compat_strings = compat_prop.as_ref()?.iter_str();
            if !compat_strings.any(|s| Ok(s == compat)).ok()? {
                continue;
            }
            let reg_prop = node
                .props()
                .find(|p| Ok(p.name().unwrap_or("empty") == "reg"))
                .ok()?;
            let base = reg_prop.as_ref()?.u64(0).ok()?;
            let size = reg_prop.as_ref()?.u64(1).ok()?;
            return Some(FdtMemoryRegion { base, size });
        }
        None
    }
}

/// A base address + length pair representing a region of memory.
#[derive(Copy, Clone, Debug, Default)]
pub struct FdtMemoryRegion {
    base: u64,
    size: u64,
}

impl FdtMemoryRegion {
    /// Returns the base address of the memory region.
    pub fn base(&self) -> u64 {
        self.base
    }
    /// Returns the size of the memory region.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// An iterator over the regions in a 'memory' node.
#[derive(Clone)]
pub struct MemoryRegionIter<'a, 'dt> {
    inner: DevTreeNodeIter<'a, 'dt>,
    prop: Option<DevTreeProp<'a, 'dt>>,
    index: usize,
}

impl<'a, 'dt> MemoryRegionIter<'a, 'dt> {
    /// Creates a new iterator over the memory regions in the given `Fdt`.
    fn new(fdt: &'a Fdt<'dt>) -> Self {
        Self {
            inner: fdt.inner.nodes(),
            prop: None,
            index: 0,
        }
    }

    /// Advances the iterator to the next 'reg' property in a memory node.
    fn next_mem_reg(&mut self) -> Option<DevTreeProp<'a, 'dt>> {
        let node = self
            .inner
            .find(|n| Ok(n.name().unwrap_or("").starts_with("memory")))
            .unwrap_or(None)?;
        // We don't care what's next; silence the unused Result<> warning.
        let _ = self.inner.next();
        node.props()
            .find(|p| Ok(p.name().unwrap_or("") == "reg"))
            .unwrap_or(None)
    }
}

impl<'a, 'dt> Iterator for MemoryRegionIter<'a, 'dt> {
    type Item = FdtMemoryRegion;

    // This assumes the 'memory' nodes are well-formed (have correctly-sized, non-empty 'reg'
    // properties).
    fn next(&mut self) -> Option<Self::Item> {
        if self.prop.is_none() {
            self.prop = self.next_mem_reg();
        }
        let p = self.prop.as_ref()?;
        let base = p.u64(self.index).ok()?;
        let size = p.u64(self.index + 1).ok()?;
        self.index += 2;
        if self.index * core::mem::size_of::<u64>() >= p.length() {
            self.prop = None;
        }
        Some(FdtMemoryRegion { base, size })
    }
}

/// An iterator over the reserved memory regions from an FDT header.
#[derive(Clone)]
pub struct ReservedRegionIter<'a, 'dt> {
    inner: DevTreeReserveEntryIter<'a, 'dt>,
}

impl<'a, 'dt> ReservedRegionIter<'a, 'dt> {
    fn new(fdt: &'a Fdt<'dt>) -> Self {
        Self {
            inner: fdt.inner.reserved_entries(),
        }
    }
}

impl<'a, 'dt: 'a> Iterator for ReservedRegionIter<'a, 'dt> {
    type Item = FdtMemoryRegion;

    fn next(&mut self) -> Option<Self::Item> {
        let range = self.inner.next()?;
        Some(FdtMemoryRegion {
            base: u64::from(range.address),
            size: u64::from(range.size),
        })
    }
}
