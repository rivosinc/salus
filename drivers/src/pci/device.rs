// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt;
use core::ptr::NonNull;

use tock_registers::interfaces::{Readable, Writeable};

use super::address::*;
use super::bus::PciBus;
use super::error::*;
use super::registers::*;

/// The Vendor Id from the PCI header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct VendorId(u16);

impl VendorId {
    pub const fn invalid() -> Self {
        VendorId(0xffff)
    }
}

/// The Device Id from the PCI header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct DeviceId(u16);

/// The Class of the device from the PCI Header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Class(u8);

/// The SubClass of the device from the PCI Header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SubClass(u8);

/// The Header type of a PCI Header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum HeaderType {
    Endpoint,
    PciBridge,
    CardBusBridge,
    Unknown(u8),
}

impl fmt::Display for HeaderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderType::Endpoint => write!(f, "Endpoint"),
            HeaderType::PciBridge => write!(f, "PciBridge"),
            HeaderType::CardBusBridge => write!(f, "CardBusBridge"),
            HeaderType::Unknown(h) => write!(f, "Unknown(0x{:x})", h),
        }
    }
}

/// Attributes of a PCI function read from the common PCI configuration space header.
#[derive(Clone, Debug)]
pub struct PciDeviceInfo {
    address: Address,
    vendor_id: VendorId,
    device_id: DeviceId,
    class: Class,
    subclass: SubClass,
    multi_function: bool,
    header_type: HeaderType,
}

impl PciDeviceInfo {
    /// Reads a PCI configuration header from `regs`.
    pub fn read_from(address: Address, regs: &CommonRegisters) -> Option<Self> {
        let vendor_id = VendorId(regs.vendor_id.get());
        if vendor_id == VendorId::invalid() {
            return None;
        }
        let device_id = DeviceId(regs.dev_id.get());
        let class = Class(regs.class.get());
        let subclass = SubClass(regs.subclass.get());
        let multi_function = regs.header_type.read(Type::MultiFunction) == 1;
        let header_type = match regs.header_type.read(Type::Layout) {
            0 => HeaderType::Endpoint,
            1 => HeaderType::PciBridge,
            2 => HeaderType::CardBusBridge,
            x => HeaderType::Unknown(x),
        };

        let info = Self {
            address,
            vendor_id,
            device_id,
            class,
            subclass,
            multi_function,
            header_type,
        };
        Some(info)
    }

    /// Returns the PCI Adress of this PCI header.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Returns the vendor ID from this PCI header.
    pub fn vendor_id(&self) -> VendorId {
        self.vendor_id
    }

    /// Returns the device ID from this PCI header.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Returns the device class from this PCI header.
    pub fn class(&self) -> Class {
        self.class
    }

    /// Returns the device subclass from this PCI header.
    pub fn subclass(&self) -> SubClass {
        self.subclass
    }

    /// Returns the header type from this PCI header.
    pub fn header_type(&self) -> HeaderType {
        self.header_type
    }

    /// Returns true if the multi-function bit is set in the header type field of this header.
    pub fn multi_function(&self) -> bool {
        self.multi_function
    }
}

impl core::fmt::Display for PciDeviceInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Address: {} Vendor: {:X} Device: {:X} Class: {:X} SubClass: {:X}",
            self.address,
            self.vendor_id().0,
            self.device_id().0,
            self.class().0,
            self.subclass().0
        )
    }
}

/// Represents a PCI endpoint.
pub struct PciEndpoint {
    _registers: &'static mut EndpointRegisters,
    info: PciDeviceInfo,
}

impl PciEndpoint {
    /// Creates a new `PciEndpoint` using the config space at `registers`.
    fn new(registers: &'static mut EndpointRegisters, info: PciDeviceInfo) -> Result<Self> {
        Ok(Self {
            _registers: registers,
            info,
        })
    }
}

/// Represents a PCI bridge.
pub struct PciBridge {
    registers: &'static mut BridgeRegisters,
    info: PciDeviceInfo,
    bus_range: BusRange,
    child_bus: Option<PciBus>,
}

impl PciBridge {
    /// Creates a new `PciBridge` use the config space at `registers`. Downstream buses are initially
    /// unenumerated.
    fn new(registers: &'static mut BridgeRegisters, info: PciDeviceInfo) -> Result<Self> {
        // Prevent config cycles from passing beyond this bridge until we're ready to enumreate.
        registers.sub_bus.set(0);
        registers.sec_bus.set(0);
        registers.pri_bus.set(0);
        Ok(Self {
            registers,
            info,
            bus_range: BusRange::default(),
            child_bus: None,
        })
    }

    /// Configures the secondary and subordinate bus numbers of the bridge such that configuration
    /// cycles from `range.start` to `range.end` (inclusive) will be forwarded downstream.
    pub fn assign_bus_range(&mut self, range: BusRange) {
        self.registers.sub_bus.set(range.end.bits() as u8);
        self.registers.sec_bus.set(range.start.bits() as u8);
        let pri_bus = self.info.address().bus();
        self.registers.pri_bus.set(pri_bus.bits() as u8);
        self.bus_range = range;
    }

    /// Sets the bus that is directly downstream of this bridge.
    pub fn set_child_bus(&mut self, bus: PciBus) {
        self.child_bus = Some(bus)
    }

    /// Returns the secondary bus directly downstream of this bridge.
    pub fn child_bus(&self) -> Option<&PciBus> {
        self.child_bus.as_ref()
    }
}

/// Represents a single PCI device.
pub enum PciDevice {
    /// A function endpoint (type 0) device.
    Endpoint(PciEndpoint),
    /// A bridge (type 1) device.
    Bridge(PciBridge),
}

impl PciDevice {
    /// Creates a `PciDevice` from a function config space.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `registers_ptr` points to a valid and uniquely-owned
    /// configuration space, and that it and `info` reference the same device.
    pub unsafe fn new(
        registers_ptr: NonNull<CommonRegisters>,
        info: PciDeviceInfo,
    ) -> Result<Self> {
        match info.header_type() {
            HeaderType::Endpoint => {
                let registers = registers_ptr.cast().as_mut();
                let ep = PciEndpoint::new(registers, info)?;
                Ok(PciDevice::Endpoint(ep))
            }
            HeaderType::PciBridge => {
                let registers = registers_ptr.cast().as_mut();
                let bridge = PciBridge::new(registers, info)?;
                Ok(PciDevice::Bridge(bridge))
            }
            h => Err(Error::UnsupportedHeaderType(info.address(), h)),
        }
    }

    /// Returns the `PciDeviceInfo` for this device.
    pub fn info(&self) -> &PciDeviceInfo {
        match self {
            PciDevice::Endpoint(ep) => &ep.info,
            PciDevice::Bridge(bridge) => &bridge.info,
        }
    }
}

// PciEndpoint and PciBridge hold raw pointers to their config spaces. Access to that config space is
// done through their respective interfaces which allow them to be shared and sent between threads.
unsafe impl Send for PciDevice {}
unsafe impl Sync for PciDevice {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn dev_info() {
        let mut test_config: [u32; 128] = [0xdead_beef; 128];
        test_config[0] = 0xa9a9_b8b8; // device and vendor id
        test_config[1] = 0x0000_0000; // status and command
        test_config[2] = 0xc7d6_e5f4; // class, subclass, prog IF, revision ID
        test_config[3] = 0xde00_beef; // BIST, header type, latency time, cache line size
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let regs = unsafe { (header_mem.as_mut_ptr() as *mut CommonRegisters).as_ref() }.unwrap();
        let info = PciDeviceInfo::read_from(Address::default(), regs).expect("can't create header");
        assert_eq!(info.vendor_id(), VendorId(0xb8b8));
        assert_eq!(info.device_id(), DeviceId(0xa9a9));
        assert_eq!(info.class(), Class(0xc7));
        assert_eq!(info.subclass(), SubClass(0xd6));
        assert_eq!(info.header_type(), HeaderType::Endpoint);
        assert!(!info.multi_function());

        // Set multi-function bit.
        test_config[3] = 0xde80_beef;
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let regs = unsafe { (header_mem.as_mut_ptr() as *mut CommonRegisters).as_ref() }.unwrap();
        let info = PciDeviceInfo::read_from(Address::default(), regs).expect("can't create header");
        assert_eq!(info.header_type(), HeaderType::Endpoint);
        assert!(info.multi_function());

        // Invalid vendor ID should not produce a valid header
        test_config[0] = 0xa9a9_ffff;
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let regs = unsafe { (header_mem.as_mut_ptr() as *mut CommonRegisters).as_ref() }.unwrap();
        assert!(PciDeviceInfo::read_from(Address::default(), regs).is_none());
    }
}
