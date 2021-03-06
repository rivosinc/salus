// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::ops::Range;

use data_model::Le32;
use riscv_pages::SupervisorPageAddr;

use super::address::*;
use super::header::{Header, HeaderWord};

// See PCI Express Base Specification
const PCIE_ECAM_FN_SHIFT: u64 = 12;
const PCIE_FUNCTION_CONFIG_SIZE: usize = 4096;

/// A PCIe ECAM configuration space for a root complex covering `self.bus_range`.
pub struct PciConfigSpace {
    config_base: SupervisorPageAddr,
    config_size: u64,
    segment: Segment,
    bus_range: BusRange,
}

impl PciConfigSpace {
    /// Creates a new `PciConfigSpace` at `config_base` covering `bus_range` in `segment`.
    pub fn new(
        config_base: SupervisorPageAddr,
        config_size: u64,
        segment: Segment,
        bus_range: BusRange,
    ) -> Self {
        Self {
            config_base,
            config_size,
            segment,
            bus_range,
        }
    }

    /// Returns the `BusConfig` for the bus at `bus_num`.
    pub fn bus(&self, bus_num: Bus) -> Option<BusConfig> {
        if bus_num < self.bus_range.start || bus_num > self.bus_range.end {
            return None;
        }

        let header_addr = Address::new(
            self.segment,
            bus_num,
            Device::default(),
            Function::default(),
        );
        Some(BusConfig {
            config_space: self,
            addr: header_addr,
        })
    }

    /// Returns the configuration space for the function at the given address if the address is
    /// valid for this ECAM space.
    pub fn config_space_for(&self, address: Address) -> Option<PciFuncConfigSpace> {
        let offset = self.config_space_offset(address)?;
        if offset.checked_add(PCIE_FUNCTION_CONFIG_SIZE as u64)? > self.config_size {
            return None;
        }
        // Safety: config_base is guaranteed to be uniquely owned PCI memory by construction and the
        // range of memory used below is within the owned range as checked.
        let func_config = unsafe {
            PciFuncConfigSpace::new(address, (self.config_base.bits() + offset) as *mut Le32)
        };
        Some(func_config)
    }

    /// Gets a Header object at the given address if it exists.
    pub fn header_for(&self, address: Address) -> Option<Header> {
        let func_config = self.config_space_for(address)?;
        Header::new(&func_config)
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
    fn config_space_offset(&self, address: Address) -> Option<u64> {
        (address.bits() as u64)
            .checked_sub(Address::bus_address(self.bus_range.start).bits() as u64)
            .map(|a| a << PCIE_ECAM_FN_SHIFT)
    }
}

/// The configuration space for a single PCI function. Used to access the registers within the
/// configuration space of the function.
pub struct PciFuncConfigSpace {
    address: Address,
    base_ptr: *mut Le32,
}

impl PciFuncConfigSpace {
    /// Creates a new function config space for the function at `address` with a config space located
    /// at `base_ptr`.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that `base_ptr` points to a valid PCIe function config space of length
    /// `PCIE_FUNCTION_CONFIG_SIZE` and that `PciFunctionConfigSpace` can read and write to that area
    /// safely.
    pub unsafe fn new(address: Address, base_ptr: *mut Le32) -> Self {
        Self { address, base_ptr }
    }

    /// Returns the PCI bus address of this function.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Reads a 32-bit value from this function's config space.
    pub fn read_dword(&self, dword: HeaderWord) -> u32 {
        unsafe {
            // Safety: A HeaderWord is guaranteed to be within the bounds of the config region at
            // self.base_ptr.
            let dword_ptr = self.base_ptr.add(dword as usize);
            core::ptr::read_volatile(dword_ptr).to_native()
        }
    }

    /// Writes a 32-bit value to this function's config sapce.
    pub fn write_dword(&mut self, dword: HeaderWord, val: u32) {
        unsafe {
            // Safety: A HeaderWord is guaranteed to be within the bounds of the config region at
            // self.base_ptr.
            let dword_ptr = self.base_ptr.add(dword as usize);
            core::ptr::write_volatile(dword_ptr, Le32::from(val))
        };
    }
}

/// The configuration space for a PCI bus. Used to enumerate the devices on a single bus.
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
    type Item = DevConfig<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let dev_addr = self.next_dev?;
        self.config_space.config_space_offset(dev_addr)?;
        self.next_dev = dev_addr.next_device();
        Some(DevConfig {
            config_space: self.config_space,
            dev_addr,
        })
    }
}

/// Configuration space for a PCI device. Used to enumerate the functions of a single device.
pub struct DevConfig<'a> {
    config_space: &'a PciConfigSpace,
    dev_addr: Address,
}

impl<'a> DevConfig<'a> {
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
    use riscv_pages::{PageAddr, RawAddr};

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

        let config_space = PciConfigSpace::new(
            PageAddr::new(RawAddr::supervisor(config_start)).unwrap(),
            8192,
            Segment::default(),
            BusRange::default(),
        );

        let first_addr = Address::default();
        assert!(config_space.header_for(first_addr).is_some());
        let second_address = first_addr.next_function().unwrap();
        assert!(config_space.header_for(second_address).is_none());
    }
}
