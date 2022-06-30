// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::alloc::Allocator;
use core::ops::Range;

use data_model::Le32;
use device_tree::DeviceTree;
use page_tracking::HwMemMap;
use riscv_pages::{DeviceMemType, PageAddr, PageSize, RawAddr, SupervisorPageAddr};

use super::address::{Address, Bus, Device, Function, Segment};
use super::header::Header;

// See PCI Express Base Specification
const PCIE_ECAM_FN_SHIFT: u64 = 12;
const PCIE_FUNCTION_HEADER_LEN: u64 = 0x40;

/// Errors resulting parsing PCI from device tree.
#[derive(Debug)]
pub enum Error {
    /// The PCI configuration space provided by device tree isn't aligned to 4k.
    ConfigSpaceMisaligned(u64),
    /// The PCI configuration size provided by device tree isn't divisible by 4k.
    ConfigSpaceNotPageMultiple(u64),
    /// The device tree contained an MMIO region that overlaps with other memory regions.
    InvalidMmioRegion(page_tracking::MemMapError),
    /// The device tree entry for the PCI host didn't provide a base register for Configuration
    /// Space.
    NoConfigBase,
    /// No compatible PCI host controller found in the device tree.
    NoCompatibleHostNode,
    /// The device tree entry for the PCI host didn't provide a size register for Configuration
    /// Space.
    NoConfigSize,
    /// The device tree entry for the PCI host didn't provide `reg` property.
    NoRegProperty,
    /// The device tree provided an invalid start bus in the `bus-range` property.
    StartBusInvalid(u32),
}
/// Holds results for PCI probing from device tree.
pub type Result<T> = core::result::Result<T, Error>;

/// The configuration space for PCI starting at `self.start_bus`.
pub struct PciConfigSpace {
    config_base: SupervisorPageAddr,
    config_size: u64,
    segment: Segment,
    start_bus: Bus,
}

impl PciConfigSpace {
    /// Creates a `PciConfigSpace` by finding a supported configuration in the passed `DeviceTree`.
    pub fn probe_from<A: Allocator + Clone>(
        dt: &DeviceTree<A>,
        mem_map: &mut HwMemMap,
    ) -> Result<Self> {
        let pci_node = dt
            .iter()
            .find(|n| n.compatible(["pci-host-ecam-generic"]) && !n.disabled())
            .ok_or(Error::NoCompatibleHostNode)?;
        let mut regs = pci_node
            .props()
            .find(|p| p.name() == "reg")
            .ok_or(Error::NoRegProperty)?
            .value_u64();

        let config_addr_raw = regs.next().ok_or(Error::NoConfigBase)?;
        let config_base = PageAddr::new(RawAddr::supervisor(config_addr_raw))
            .ok_or(Error::ConfigSpaceMisaligned(config_addr_raw))?;

        let config_size = regs.next().ok_or(Error::NoConfigSize)?;
        if config_size % (PageSize::Size4k as u64) != 0 {
            return Err(Error::ConfigSpaceNotPageMultiple(config_size));
        }

        // Checks for a `bus-range` property specifying a start bus.
        let start_bus_index = pci_node
            .props()
            .find(|p| p.name() == "bus-range")
            .and_then(|props| props.value_u32().next())
            .unwrap_or(0);
        let start_bus = start_bus_index
            .try_into()
            .map_err(|_| Error::StartBusInvalid(start_bus_index))?;

        unsafe {
            // Safety: Have to trust that the device tree points to valid PCI space.
            // Any overlaps will be caught by `add_mmio_region` and the error will be propagated.
            mem_map
                .add_mmio_region(
                    DeviceMemType::PciRoot,
                    RawAddr::from(config_base),
                    config_size,
                )
                .map_err(Error::InvalidMmioRegion)?;
        }

        Ok(Self {
            config_base,
            config_size,
            // TODO create a config space per segment
            segment: Segment::default(),
            start_bus,
        })
    }

    /// Returns an iterator across the top-level buses.
    pub fn busses(&self) -> HostControllersIter {
        let header_addr = Address::new(
            self.segment,
            self.start_bus,
            Device::default(),
            Function::default(),
        );

        HostControllersIter {
            config_space: self,
            segment: self.segment,
            fn_range: self.function_scan_range(header_addr),
        }
    }

    /// Gets a Header object at the given address if it exists.
    pub fn header_for(&self, address: Address) -> Option<Header> {
        let offset = self.header_offset(address)?;
        // Safety: config_base is guaranteed to be uniquely owned PCI memory by construction and the
        // range of memory used below is within the owned range as checked.
        unsafe {
            if offset.checked_add(PCIE_FUNCTION_HEADER_LEN)? > self.config_size {
                return None;
            }
            Header::new(address, (self.config_base.bits() + offset) as *mut Le32)
        }
    }

    // Returns the range of headers to check based on if this is a multi-function device.
    // If no header present: 0..0 don't scan anything.
    // If the header is present and multi-function: 0..8 - Check all the possible headers.
    // If the header is present and not multi-function: 0..1
    fn function_scan_range(&self, header_addr: Address) -> Range<u32> {
        // Only applicable to first function of a device.
        if header_addr.function() != Function::default() {
            return 0..0;
        }
        self.header_for(header_addr)
            .map(|header| if header.multi_function() { 0..8 } else { 0..1 })
            .unwrap_or(0..0)
    }

    // Returns the offset of the given address within this PciConfigSpace.
    fn header_offset(&self, address: Address) -> Option<u64> {
        (address.bits() as u64)
            .checked_sub(Address::bus_address(self.start_bus).bits() as u64)
            .map(|a| a << PCIE_ECAM_FN_SHIFT)
    }
}

/// Iterates host controllers in a given config space.
pub struct HostControllersIter<'a> {
    config_space: &'a PciConfigSpace,
    segment: Segment,
    // The range of functions to check. If signle controller it's 0, or 0-7 if multi function.
    fn_range: Range<u32>,
}

impl<'a> Iterator for HostControllersIter<'a> {
    type Item = BusConfig<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let fn_idx = self.fn_range.next()?;
        // fn_idx is guaranteed to be a valid function number.
        let this_fn: Function = fn_idx.try_into().unwrap();
        Some(BusConfig {
            config_space: self.config_space,
            addr: Address::new(
                self.segment,
                // The bus number is given by the function index.
                this_fn.into(),
                Device::default(),
                Function::default(),
            ),
        })
    }
}

/// The configuration space for a PCI bus.
pub struct BusConfig<'a> {
    config_space: &'a PciConfigSpace,
    addr: Address,
}

impl<'a> BusConfig<'a> {
    /// Returns an iterator over the possible devices in this bus configuration space.
    pub fn devices(&self) -> DevIter {
        DevIter {
            config_space: self.config_space,
            next_dev: Some(self.addr),
        }
    }
}

/// Iterates over device configuration spaces in a bus.
pub struct DevIter<'a> {
    config_space: &'a PciConfigSpace,
    next_dev: Option<Address>,
}

impl<'a> Iterator for DevIter<'a> {
    type Item = PciDev<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let dev_addr = self.next_dev?;
        self.config_space.header_offset(dev_addr)?;
        self.next_dev = dev_addr.next_device();
        Some(PciDev {
            config_space: self.config_space,
            dev_addr,
        })
    }
}

/// Configuration space for a PCI device.
pub struct PciDev<'a> {
    config_space: &'a PciConfigSpace,
    dev_addr: Address,
}

impl<'a> PciDev<'a> {
    /// Returns and iterator for each function present on this device.
    /// This will yield 0 to 8 functions depending on if the primary header is valid and if it
    /// indicates that the devices is multi-function.
    pub fn functions(&self) -> FnIter {
        FnIter {
            config_space: self.config_space,
            dev_addr: self.dev_addr,
            range: self.config_space.function_scan_range(self.dev_addr),
        }
    }
}

/// Iterator across function headers of a device.
pub struct FnIter<'a> {
    config_space: &'a PciConfigSpace,
    dev_addr: Address,
    range: Range<u32>,
}

impl<'a> Iterator for FnIter<'a> {
    type Item = Header;

    fn next(&mut self) -> Option<Self::Item> {
        let fn_idx = self.range.next()?;
        // fn_idx is guaranteed to be a valid function number.
        let this_fn: Function = fn_idx.try_into().unwrap();
        let function_addr = Address::new(
            self.dev_addr.segment(),
            self.dev_addr.bus(),
            self.dev_addr.device(),
            this_fn,
        );
        self.config_space.header_for(function_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_for() {
        let mut config_mem = vec![0xffff_ffffu32; 1024 * 4];
        // align to 4k
        let align_offset = config_mem.as_ptr().align_offset(4096);
        let config_slice = &mut config_mem[align_offset..];
        let config_start = config_slice.as_ptr() as u64;
        // Set the first header as valid and the second as invalid.
        config_slice[0] = 0x5555_6666;
        config_slice[1024] = 0xffff_ffff;

        let config_space = PciConfigSpace {
            config_base: PageAddr::new(RawAddr::supervisor(config_start)).unwrap(),
            config_size: 8192,
            segment: Segment::default(),
            start_bus: Bus::default(),
        };

        let first_addr = Address::default();
        assert!(config_space.header_for(first_addr).is_some());
        let second_address = first_addr.next_function().unwrap();
        assert!(config_space.header_for(second_address).is_none());
    }
}
