// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use data_model::Le32;

use super::address::Address;

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
    Endpoint = 0,
    PciBridge = 1,
    CardBusBridge = 2,
}

impl fmt::Display for HeaderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeaderType::Endpoint => write!(f, "Endpoint"),
            HeaderType::PciBridge => write!(f, "PciBridge"),
            HeaderType::CardBusBridge => write!(f, "CardBusBridge"),
        }
    }
}

// The MSB of the header type byte indicates multi-funciton, the lower 7 are the type.
const HEADER_TYPE_MASK: u8 = 0x7f;
const HEADER_MULTI_FUNCTION_MASK: u8 = 0x80;

// Index of words in the PCI Configuration header for a function.
enum HeaderWord {
    Vendor = 0,
    Class = 2,
    Type = 3,
}

/// The header for a PCI function in configuration space.
pub struct Header {
    address: Address,
    base_ptr: *mut Le32,
}

impl Header {
    /// Creates a new Header.
    ///
    /// # Safety: Caller must guarantee that base through base + the legnth of the header is valid
    /// and that `Header` can read and write to that area safely.
    pub unsafe fn new(address: Address, base_ptr: *mut Le32) -> Option<Self> {
        let header = Self { address, base_ptr };
        if !header.present() {
            None
        } else {
            Some(header)
        }
    }

    /// Rertuns the PCI Adress of this PCI header.
    pub fn address(&self) -> Address {
        self.address
    }

    fn present(&self) -> bool {
        self.vendor_id() != VendorId::invalid()
    }

    /// Returns the vendor ID from this PCI header.
    pub fn vendor_id(&self) -> VendorId {
        let header_word = self.read_u32(HeaderWord::Vendor);
        let v = (header_word & 0x0000_ffff) as u16;
        VendorId(v)
    }

    /// Returns the device ID from this PCI header.
    pub fn device_id(&self) -> DeviceId {
        let header_word = self.read_u32(HeaderWord::Vendor);
        let d = (header_word >> 16) as u16;
        DeviceId(d)
    }

    /// Returns the device class from this PCI header.
    pub fn class(&self) -> Class {
        let word = self.read_u32(HeaderWord::Class);
        Class((word >> 24) as u8)
    }

    /// Returns the device subclass from this PCI header.
    pub fn subclass(&self) -> SubClass {
        let word = self.read_u32(HeaderWord::Class);
        SubClass((word >> 16) as u8)
    }

    /// Returns the header type from this PCI header.
    pub fn header_type(&self) -> Option<HeaderType> {
        let word = self.read_u32(HeaderWord::Type);
        match ((word >> 16) as u8) & HEADER_TYPE_MASK {
            0 => Some(HeaderType::Endpoint),
            1 => Some(HeaderType::PciBridge),
            2 => Some(HeaderType::CardBusBridge),
            _ => None,
        }
    }

    /// Returns true if the multi-function bit is set in the header type field of this header.
    pub fn multi_function(&self) -> bool {
        let word = self.read_u32(HeaderWord::Type);
        ((word >> 16) as u8) & HEADER_MULTI_FUNCTION_MASK != 0
    }

    fn read_u32(&self, word: HeaderWord) -> u32 {
        unsafe {
            // Safety: The target and self own their memory regions and the read call is bounded by
            // those regions of memory and offset is guaranteed to be valid in the constructor as the
            // backing area must span all `HeaderWord` values.
            let word_ptr = self.base_ptr.add(word as usize);
            core::ptr::read_volatile(word_ptr).to_native()
        }
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
    use std::vec::Vec;

    #[test]
    fn create() {
        let mut test_config: [u32; 16] = [
            0xa9a9_b8b8, // device and vendor id
            0x0000_0000, // status and command
            0xc7d6_e5f4, // class, subclass, prog IF, revision ID
            0xde00_beef, // BIST, header type, latency time, cache line size
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
            0xdead_beef,
        ];
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let header = unsafe {
            Header::new(Address::default(), header_mem.as_mut_ptr() as *mut Le32)
                .expect("can't create header")
        };
        assert!(header.present());
        assert_eq!(header.vendor_id(), VendorId(0xb8b8));
        assert_eq!(header.device_id(), DeviceId(0xa9a9));
        assert_eq!(header.class(), Class(0xc7));
        assert_eq!(header.subclass(), SubClass(0xd6));
        assert_eq!(header.header_type(), Some(HeaderType::Endpoint));
        assert!(!header.multi_function());

        // Set multi-function bit.
        test_config[HeaderWord::Type as usize] = 0xde80_beef;
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        let header = unsafe {
            Header::new(Address::default(), header_mem.as_mut_ptr() as *mut Le32)
                .expect("can't create header")
        };
        assert_eq!(header.header_type(), Some(HeaderType::Endpoint));
        assert!(header.multi_function());

        // Invalid vendor ID should not produce a valid header
        test_config[HeaderWord::Vendor as usize] = 0xa9a9_ffff;
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        unsafe {
            assert!(
                Header::new(Address::default(), header_mem.as_mut_ptr() as *mut Le32,).is_none()
            );
        }
    }
}
