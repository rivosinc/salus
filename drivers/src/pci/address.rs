// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// PCI addresses are composed of a segment, bus, device, and function. Each of those components will
/// implement the `AddressComponent` type.
pub trait AddressComponent {
    const SHIFT: u32;
    const BITS: u32;
    const MAX_VAL: u32 = (1 << Self::BITS) - 1;
}

// Implements conversions from and to the specified types to the given address component.
macro_rules! unsigned_conversions {
    ($T:ident, $($F:ident),+) => {
        $(
        impl TryFrom<$F> for $T {
            type Error = ();
            fn try_from(v: $F) -> core::result::Result<Self, Self::Error> {
                if (Self::MAX_VAL as u64) < (v as u64) {
                    Err(())
                } else {
                    Ok($T(v as u32))
                }
            }
        }

        impl PartialEq<$F> for $T {
            fn eq(&self, other: &$F) -> bool {
                self.0 as u64 == *other as u64
            }
        }
        )+
    }
}

// Implements functionality common to all the `Address` components.
macro_rules! address_type {
    ($T:ident) => {
        impl core::fmt::Display for $T {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{:X}", self.0)
            }
        }

        impl $T {
            /// Retruns the u32 bits that represent this component index.
            pub fn bits(&self) -> u32 {
                self.0
            }

            /// Returns the maximum possible value for this address component.
            pub fn max() -> Self {
                Self(Self::MAX_VAL)
            }

            /// Returns the next value for this address component.
            pub fn next(&self) -> Option<Self> {
                Self::try_from(self.0 + 1).ok()
            }
        }

        unsigned_conversions!($T, u8, u16, u32, u64);
    };
}

/// The function portion of a PCI address, 3 bits.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Function(u32);

impl AddressComponent for Function {
    const SHIFT: u32 = 0;
    const BITS: u32 = 3;
}
address_type!(Function);

/// The device portion of a PCI address, 5 bits.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Device(u32);

impl AddressComponent for Device {
    const BITS: u32 = 5;
    const SHIFT: u32 = 3;
}
address_type!(Device);

/// The bus portion of a PCI address, 8 bits.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Bus(u32);

impl AddressComponent for Bus {
    const BITS: u32 = 8;
    const SHIFT: u32 = 8;
}
address_type!(Bus);

/// A range of PCI bus numbers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BusRange {
    /// Start bus number.
    pub start: Bus,
    /// End bus number (inclusive).
    pub end: Bus,
}

// Because Functions are only 3 bits, they are trivally valid Bus numbers(8 bits).
impl From<Function> for Bus {
    fn from(f: Function) -> Self {
        Bus(f.bits())
    }
}

/// The segment portion of a PCI address, 16 bits.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Segment(u32);

impl AddressComponent for Segment {
    const BITS: u32 = 16;
    const SHIFT: u32 = 16;
}
address_type!(Segment);

/// The address of a PCI function:
/// 16bits segment, 8 bits bus, 5 bits device, 3 bits function.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Address(u32);

impl Address {
    /// Creates a Address from the passed address components. Returns None if any are out of
    /// range.
    pub fn try_from_components(seg: u32, bus: u32, dev: u32, func: u32) -> Option<Address> {
        Some(Self::new(
            Segment::try_from(seg).ok()?,
            Bus::try_from(bus).ok()?,
            Device::try_from(dev).ok()?,
            Function::try_from(func).ok()?,
        ))
    }

    /// Creates a new `Address` based on the provided components.
    pub fn new(seg: Segment, bus: Bus, dev: Device, func: Function) -> Address {
        Address(seg.0 << Segment::SHIFT | bus.0 << Bus::SHIFT | dev.0 << Device::SHIFT | func.0)
    }

    /// Creates an `Address` for the given `Bus` on the first segment.
    pub fn bus_address(bus: Bus) -> Address {
        Address(bus.0 << Bus::SHIFT)
    }

    /// Returns the function portion of the address.
    pub fn function(&self) -> Function {
        Function(self.0 & Function::MAX_VAL)
    }

    /// Returns the device portion of the address.
    pub fn device(&self) -> Device {
        Device((self.0 >> Device::SHIFT) & Device::MAX_VAL)
    }

    /// Returns the bus portion of the address.
    pub fn bus(&self) -> Bus {
        Bus((self.0 >> Bus::SHIFT) & Bus::MAX_VAL)
    }

    /// Returns the segment portion of the address.
    pub fn segment(&self) -> Segment {
        Segment((self.0 >> Segment::SHIFT) & Segment::MAX_VAL)
    }

    /// Returns the u32 used to represent this address to PCI.
    pub fn bits(&self) -> u32 {
        self.0
    }

    /// Returns the address of the next PCI function or `None` if no more functions are available
    /// for the current device.
    pub fn next_function(&self) -> Option<Address> {
        Function::try_from(self.function().0 + 1)
            .ok()
            .map(|func| Address::new(self.segment(), self.bus(), self.device(), func))
    }

    /// Returns the address of the first function in the next PCI device or `None` if the next is
    /// out of range.
    pub fn next_device(&self) -> Option<Address> {
        Device::try_from(self.device().0 + 1)
            .ok()
            .map(|dev| Address::new(self.segment(), self.bus(), dev, Function::default()))
    }

    /// Returns the address of the first function in the next device of the next bus or `None` if
    /// the next is out of range.
    pub fn next_bus(&self) -> Option<Address> {
        Bus::try_from(self.bus().0 + 1)
            .ok()
            .map(|bus| Address::new(self.segment(), bus, Device::default(), Function::default()))
    }

    /// Returns the address of the first function in the next segment or `None` if the next is out
    /// of range.
    pub fn next_segment(&self) -> Option<Address> {
        Segment::try_from(self.segment().0 + 1)
            .ok()
            .map(|seg| Address::new(seg, Bus::default(), Device::default(), Function::default()))
    }
}

impl core::fmt::Display for Address {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}-{}-{}-{}",
            self.segment(),
            self.bus(),
            self.device(),
            self.function()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a new address from teh given components of panic if any component is invalid.
    fn make_address(s: u32, b: u32, d: u32, f: u32) -> Address {
        Address::try_from_components(s, b, d, f).unwrap()
    }

    #[test]
    fn component_try_from() {
        assert!(Function::try_from(0u32).is_ok());
        assert!(Function::try_from(0u64).is_ok());
        assert!(Function::try_from(Function::MAX_VAL).is_ok());
        assert!(Function::try_from(Function::MAX_VAL as u64).is_ok());
        assert!(Function::try_from(Function::MAX_VAL + 1).is_err());
        assert!(Function::try_from(Function::MAX_VAL as u64 + 1).is_err());
    }

    #[test]
    fn checked_new() {
        assert!(Address::try_from_components(0, 0, 0, 0).is_some());
        assert!(Address::try_from_components(0xffff, 0xff, 0x1f, 0x7).is_some());
        assert!(Address::try_from_components(0xffff, 0xff, 0x1f, 0x8).is_none());
        assert!(Address::try_from_components(0xffff, 0xff, 0x20, 0x7).is_none());
        assert!(Address::try_from_components(0xffff, 0x100, 0x1f, 0x7).is_none());
        assert!(Address::try_from_components(0x1_0000, 0xff, 0x1f, 0x7).is_none());
    }

    #[test]
    fn next_addr() {
        let a = Address::try_from_components(0, 0, 0, 0).unwrap();

        assert_eq!(Some(make_address(0, 0, 0, 1)), a.next_function());
        assert_eq!(Some(make_address(0, 0, 1, 0)), a.next_device());
        assert_eq!(Some(make_address(0, 1, 0, 0)), a.next_bus());
        assert_eq!(Some(make_address(1, 0, 0, 0)), a.next_segment());

        // Moving from the last function of one device to the first function on the next.
        assert!(make_address(0, 0, 0, 7).next_function().is_none());

        // Move to next device from function 2 of the current.
        assert_eq!(
            make_address(0, 0, 1, 2).next_device(),
            Some(make_address(0, 0, 2, 0))
        );

        // None is returned when out of address space.
        assert!(make_address(0xffff, 0xff, 0x1f, 0x7)
            .next_function()
            .is_none());
    }
}
