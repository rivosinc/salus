// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::ops::Range;
use core::ptr::NonNull;

use riscv_pages::{PageSize, SupervisorPageAddr, SupervisorPageRange};

use super::address::*;
use super::device::PciDeviceInfo;
use super::registers::CommonRegisters;

// See PCI Express Base Specification
const PCIE_ECAM_FN_SHIFT: u64 = 12;
const PCIE_ECAM_DEV_SHIFT: u64 = 15;
const PCIE_ECAM_BUS_SHIFT: u64 = 20;
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

    /// Returns a pointer to the configuration space registers for the function at the given address if
    /// the address is valid for this ECAM space.
    pub fn registers_for(&self, address: Address) -> Option<NonNull<CommonRegisters>> {
        let offset = self.config_space_offset(address)?;
        if offset.checked_add(PCIE_FUNCTION_CONFIG_SIZE as u64)? > self.config_size {
            return None;
        }
        NonNull::new((self.config_base.bits() + offset) as *mut CommonRegisters)
    }

    /// Gets a `PciDeviceInfo` object for the device at `address`, if present.
    pub fn info_for(&self, address: Address) -> Option<PciDeviceInfo> {
        // Safety: Each register block in this ECAM space points to a valid (hardware-)initialized
        // common config space header.
        let registers = unsafe { self.registers_for(address)?.as_ref() };
        PciDeviceInfo::read_from(address, registers)
    }

    /// Maps an offset within this config space to the PCI address of the device whose config space
    /// is mapped at that location, along with the offset within that device's config space.
    pub fn offset_to_address(&self, offset: usize) -> Option<(Address, usize)> {
        if offset as u64 >= self.config_size {
            return None;
        }
        let bus =
            (((offset >> PCIE_ECAM_BUS_SHIFT) as u32) & Bus::MAX_VAL) + self.bus_range.start.bits();
        let dev = ((offset >> PCIE_ECAM_DEV_SHIFT) as u32) & Device::MAX_VAL;
        let func = ((offset >> PCIE_ECAM_FN_SHIFT) as u32) & Function::MAX_VAL;
        let address = Address::try_from_components(self.segment.bits(), bus, dev, func)?;
        let dev_offset = offset & (PCIE_FUNCTION_CONFIG_SIZE - 1);
        Some((address, dev_offset))
    }

    /// Returns the memory range occupied by this config space.
    pub fn mem_range(&self) -> SupervisorPageRange {
        SupervisorPageRange::new(self.config_base, PageSize::num_4k_pages(self.config_size))
    }

    /// Returns the PCI segment (domain) for this config space.
    pub fn segment(&self) -> Segment {
        self.segment
    }

    /// Returns the PCI bus numbers covered by this config space.
    pub fn bus_range(&self) -> BusRange {
        self.bus_range
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
        self.info_for(header_addr)
            .map(|info| if info.multi_function() { 0..8 } else { 0..1 })
            .unwrap_or(0..0)
    }

    // Returns the offset of the given address within this PciConfigSpace.
    fn config_space_offset(&self, address: Address) -> Option<u64> {
        (address.bits() as u64)
            .checked_sub(Address::bus_address(self.bus_range.start).bits() as u64)
            .map(|a| a << PCIE_ECAM_FN_SHIFT)
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
    type Item = PciDeviceInfo;

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
        self.config_space.info_for(function_addr)
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
        assert!(config_space.info_for(first_addr).is_some());
        let second_address = first_addr.next_function().unwrap();
        assert!(config_space.info_for(second_address).is_none());
    }
}
