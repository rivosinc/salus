// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use super::address::Address;
use super::config_space::PciFuncConfigSpace;

use core::fmt;

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

// The MSB of the header type byte indicates multi-funciton, the lower 7 are the type.
const HEADER_TYPE_MASK: u8 = 0x7f;
const HEADER_MULTI_FUNCTION_MASK: u8 = 0x80;

/// Index of words in the PCI Configuration header for a function.
///
/// TODO: Type-safe definition of the config space register map.
pub enum HeaderWord {
    /// Vendor ID and Device ID.
    Vendor = 0,
    /// Class and subclass codes.
    Class = 2,
    /// Header type (and other legacy bits).
    Type = 3,
}

/// The header for a PCI function in configuration space.
#[derive(Clone, Debug)]
pub struct Header {
    address: Address,
    vendor_id: VendorId,
    device_id: DeviceId,
    class: Class,
    subclass: SubClass,
    multi_function: bool,
    header_type: HeaderType,
}

impl Header {
    /// Reads a PCI configuration header from `func_config`.
    pub fn new(func_config: &PciFuncConfigSpace) -> Option<Self> {
        let id_dword = func_config.read_dword(HeaderWord::Vendor);
        let vendor_id = VendorId((id_dword & 0x0000_ffff) as u16);
        if vendor_id == VendorId::invalid() {
            return None;
        }
        let device_id = DeviceId((id_dword >> 16) as u16);

        let class_dword = func_config.read_dword(HeaderWord::Class);
        let class = Class((class_dword >> 24) as u8);
        let subclass = SubClass((class_dword >> 16) as u8);

        let type_dword = func_config.read_dword(HeaderWord::Type);
        let multi_function = ((type_dword >> 16) as u8) & HEADER_MULTI_FUNCTION_MASK != 0;
        let header_type = match ((type_dword >> 16) as u8) & HEADER_TYPE_MASK {
            0 => HeaderType::Endpoint,
            1 => HeaderType::PciBridge,
            2 => HeaderType::CardBusBridge,
            x => HeaderType::Unknown(x),
        };

        let header = Self {
            address: func_config.address(),
            vendor_id,
            device_id,
            class,
            subclass,
            multi_function,
            header_type,
        };
        Some(header)
    }

    /// Rertuns the PCI Adress of this PCI header.
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

impl core::fmt::Display for Header {
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

#[cfg(test)]
mod tests {
    use super::*;
    use data_model::Le32;
    use std::vec::Vec;

    #[test]
    fn create() {
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
        let func_config = unsafe {
            PciFuncConfigSpace::new(Address::default(), header_mem.as_mut_ptr() as *mut Le32)
        };
        let header = Header::new(&func_config).expect("can't create header");
        assert_eq!(header.vendor_id(), VendorId(0xb8b8));
        assert_eq!(header.device_id(), DeviceId(0xa9a9));
        assert_eq!(header.class(), Class(0xc7));
        assert_eq!(header.subclass(), SubClass(0xd6));
        assert_eq!(header.header_type(), HeaderType::Endpoint);
        assert!(!header.multi_function());

        // Set multi-function bit.
        test_config[HeaderWord::Type as usize] = 0xde80_beef;
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let func_config = unsafe {
            PciFuncConfigSpace::new(Address::default(), header_mem.as_mut_ptr() as *mut Le32)
        };
        let header = Header::new(&func_config).expect("can't create header");
        assert_eq!(header.header_type(), HeaderType::Endpoint);
        assert!(header.multi_function());

        // Invalid vendor ID should not produce a valid header
        test_config[HeaderWord::Vendor as usize] = 0xa9a9_ffff;
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let func_config = unsafe {
            PciFuncConfigSpace::new(Address::default(), header_mem.as_mut_ptr() as *mut Le32)
        };
        assert!(Header::new(&func_config).is_none());
    }
}
