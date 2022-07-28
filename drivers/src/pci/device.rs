// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt;
use core::mem::size_of;
use core::ptr::NonNull;

use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::LocalRegisterCopy;

use super::address::*;
use super::bus::PciBus;
use super::error::*;
use super::mmio_builder::{MmioReadBuilder, MmioWriteBuilder};
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

// Macro that itself defines a `span!()` macro for the given struct field which evaluates to a
// const range pattern which can be used in `match` expressions. Note that the type of the field
// must also be specified, though its cross-checked against the actual field span in a unit test.
//
// This is as hairy as it is because of the limitations of match arm range patterns and constant
// expressions in Rust. Ideally we would be able to reuse `memoffset::span_of()!` to implmenet this,
// however it's currently not possible to use `span_of!()`/`offset_of!()` in const expressions, see
// https://github.com/Gilnaa/memoffset/issues/4#issuecomment-1069658383.
//
// TODO: Replace this with `span_of!()` when it's possible to use it in const expressions.
macro_rules! define_field_span {
    ($st:ident, $field:tt, $field_type:ty) => {
        pub mod $field {
            use super::$st;

            pub const START_OFFSET: usize = $st::FIELD_OFFSETS.$field.get_byte_offset();
            pub const END_OFFSET: usize = START_OFFSET + core::mem::size_of::<$field_type>() - 1;

            macro_rules! span {
                () => {
                    $field::START_OFFSET..=$field::END_OFFSET
                };
            }

            #[cfg(test)]
            mod tests {
                use memoffset::span_of;

                #[test]
                fn check_field_span() {
                    let actual_span = span_of!(super::$st, $field);
                    assert_eq!(super::START_OFFSET, actual_span.start);
                    assert_eq!(super::END_OFFSET, actual_span.end - 1);
                }
            }

            pub(crate) use span;
        }
    };
}

mod common_offsets {
    use super::CommonRegisters;

    define_field_span!(CommonRegisters, vendor_id, u16);
    define_field_span!(CommonRegisters, dev_id, u16);
    define_field_span!(CommonRegisters, command, u16);
    define_field_span!(CommonRegisters, status, u16);
    define_field_span!(CommonRegisters, rev_id, u8);
    define_field_span!(CommonRegisters, prog_if, u8);
    define_field_span!(CommonRegisters, subclass, u8);
    define_field_span!(CommonRegisters, class, u8);
    define_field_span!(CommonRegisters, header_type, u8);
}

mod endpoint_offsets {
    use super::EndpointRegisters;

    define_field_span!(EndpointRegisters, bar, [u32; 6]);
    define_field_span!(EndpointRegisters, subsys_vendor_id, u16);
    define_field_span!(EndpointRegisters, subsys_id, u16);
    define_field_span!(EndpointRegisters, cap_ptr, u8);
}

mod bridge_offsets {
    use super::BridgeRegisters;

    define_field_span!(BridgeRegisters, bar, [u32; 2]);
    define_field_span!(BridgeRegisters, pri_bus, u8);
    define_field_span!(BridgeRegisters, sec_bus, u8);
    define_field_span!(BridgeRegisters, sub_bus, u8);
    define_field_span!(BridgeRegisters, io_base, u8);
    define_field_span!(BridgeRegisters, io_limit, u8);
    define_field_span!(BridgeRegisters, sec_status, u16);
    define_field_span!(BridgeRegisters, mem_base, u16);
    define_field_span!(BridgeRegisters, mem_limit, u16);
    define_field_span!(BridgeRegisters, pref_base, u16);
    define_field_span!(BridgeRegisters, pref_limit, u16);
    define_field_span!(BridgeRegisters, pref_base_upper, u32);
    define_field_span!(BridgeRegisters, pref_limit_upper, u32);
    define_field_span!(BridgeRegisters, io_base_upper, u16);
    define_field_span!(BridgeRegisters, io_limit_upper, u16);
    define_field_span!(BridgeRegisters, cap_ptr, u8);
    define_field_span!(BridgeRegisters, bridge_control, u16);
}

/// Represents a PCI endpoint.
pub struct PciEndpoint {
    registers: &'static mut EndpointRegisters,
    info: PciDeviceInfo,
}

impl PciEndpoint {
    /// Creates a new `PciEndpoint` using the config space at `registers`.
    fn new(registers: &'static mut EndpointRegisters, info: PciDeviceInfo) -> Result<Self> {
        Ok(Self { registers, info })
    }

    // Emulate a read from the endpoint-specific registers of this device's config space.
    fn emulate_config_read(&self, op: &mut MmioReadBuilder) {
        use endpoint_offsets::*;
        match op.offset() {
            bar::span!() => {
                let index = (op.offset() - bar::START_OFFSET) / size_of::<u32>();
                op.push_dword(self.registers.bar[index].get());
            }
            subsys_vendor_id::span!() => {
                op.push_word(self.registers.subsys_vendor_id.get());
            }
            subsys_id::span!() => {
                op.push_word(self.registers.subsys_id.get());
            }
            cap_ptr::span!() => {
                // TODO: Point to virtualized capabilites.
                op.push_byte(0);
            }
            _ => {
                // No INTx, cardbus, etc.
                op.push_byte(0);
            }
        }
    }

    // Emulate a write to the endpoint-specific registers of this device's config space.
    fn emulate_config_write(&mut self, op: &mut MmioWriteBuilder) {
        use endpoint_offsets::*;
        match op.offset() {
            bar::span!() => {
                let index = (op.offset() - bar::START_OFFSET) / size_of::<u32>();
                let reg = op.pop_dword(self.registers.bar[index].get());
                self.registers.bar[index].set(reg);
            }
            _ => {
                op.pop_byte();
            }
        }
    }
}

/// Represents a PCI bridge.
pub struct PciBridge {
    registers: &'static mut BridgeRegisters,
    info: PciDeviceInfo,
    bus_range: BusRange,
    child_bus: Option<PciBus>,
    virtual_primary_bus: Bus,
    virtual_bus_reset: u16,
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
            virtual_primary_bus: Bus::default(),
            virtual_bus_reset: 0,
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

    // Emulate a read from the bridge-specific registers of this device's config space.
    fn emulate_config_read(&self, op: &mut MmioReadBuilder) {
        use bridge_offsets::*;
        match op.offset() {
            bar::span!() => {
                let index = (op.offset() - bar::START_OFFSET) / size_of::<u32>();
                op.push_dword(self.registers.bar[index].get());
            }
            pri_bus::span!() => {
                op.push_byte(self.virtual_primary_bus.bits() as u8);
            }
            sec_bus::span!() => {
                let bus_num = self.child_bus.as_ref().unwrap().virtual_secondary_bus_num();
                op.push_byte(bus_num.bits() as u8);
            }
            sub_bus::span!() => {
                let bus_num = self
                    .child_bus
                    .as_ref()
                    .unwrap()
                    .virtual_subordinate_bus_num();
                op.push_byte(bus_num.bits() as u8);
            }
            io_base::span!() => {
                op.push_byte(self.registers.io_base.get());
            }
            io_limit::span!() => {
                op.push_byte(self.registers.io_limit.get());
            }
            sec_status::span!() => {
                op.push_word(self.registers.sec_status.readable_bits());
            }
            mem_base::span!() => {
                op.push_word(self.registers.mem_base.get());
            }
            mem_limit::span!() => {
                op.push_word(self.registers.mem_limit.get());
            }
            pref_base::span!() => {
                op.push_word(self.registers.pref_base.get());
            }
            pref_limit::span!() => {
                op.push_word(self.registers.pref_limit.get());
            }
            pref_base_upper::span!() => {
                op.push_dword(self.registers.pref_base_upper.get());
            }
            pref_limit_upper::span!() => {
                op.push_dword(self.registers.pref_limit_upper.get());
            }
            io_base_upper::span!() => {
                op.push_word(self.registers.io_base_upper.get());
            }
            io_limit_upper::span!() => {
                op.push_word(self.registers.io_limit_upper.get());
            }
            cap_ptr::span!() => {
                // TODO: Point to virtualized capabilites.
                op.push_byte(0);
            }
            bridge_control::span!() => {
                let mut reg = LocalRegisterCopy::<u16, BridgeControl::Register>::new(
                    self.registers.bridge_control.readable_bits(),
                );
                reg.modify(BridgeControl::SecondaryBusReset.val(self.virtual_bus_reset));
                op.push_word(reg.get());
            }
            _ => {
                // No INTx, cardbus, etc.
                op.push_byte(0);
            }
        }
    }

    // Emulate a write to the bridge-specific registers of this device's config space.
    fn emulate_config_write(&mut self, op: &mut MmioWriteBuilder) {
        use bridge_offsets::*;
        match op.offset() {
            bar::span!() => {
                let index = (op.offset() - bar::START_OFFSET) / size_of::<u32>();
                let reg = op.pop_dword(self.registers.bar[index].get());
                self.registers.bar[index].set(reg);
            }
            pri_bus::span!() => {
                // The primary bus register doesn't do anything on PCIe, but we emulate one here to
                // be spec compliant.
                self.virtual_primary_bus = Bus::try_from(op.pop_byte()).unwrap();
            }
            sec_bus::span!() => {
                let bus_num = Bus::try_from(op.pop_byte()).unwrap();
                self.child_bus
                    .as_mut()
                    .unwrap()
                    .set_virtual_secondary_bus_num(bus_num);
            }
            sub_bus::span!() => {
                let bus_num = Bus::try_from(op.pop_byte()).unwrap();
                self.child_bus
                    .as_mut()
                    .unwrap()
                    .set_virtual_subordinate_bus_num(bus_num);
            }
            io_base::span!() => {
                self.registers.io_base.set(op.pop_byte());
            }
            io_limit::span!() => {
                self.registers.io_limit.set(op.pop_byte());
            }
            sec_status::span!() => {
                let reg = LocalRegisterCopy::<u16, SecondaryStatus::Register>::new(
                    op.pop_word(self.registers.sec_status.non_clearable_bits()),
                );
                self.registers.sec_status.set(reg.writeable_bits());
            }
            mem_base::span!() => {
                self.registers
                    .mem_base
                    .set(op.pop_word(self.registers.mem_base.get()));
            }
            mem_limit::span!() => {
                self.registers
                    .mem_limit
                    .set(op.pop_word(self.registers.mem_limit.get()));
            }
            pref_base::span!() => {
                self.registers
                    .pref_base
                    .set(op.pop_word(self.registers.pref_base.get()));
            }
            pref_limit::span!() => {
                self.registers
                    .pref_limit
                    .set(op.pop_word(self.registers.pref_limit.get()));
            }
            pref_base_upper::span!() => {
                self.registers
                    .pref_base_upper
                    .set(op.pop_dword(self.registers.pref_base_upper.get()));
            }
            pref_limit_upper::span!() => {
                self.registers
                    .pref_limit_upper
                    .set(op.pop_dword(self.registers.pref_limit_upper.get()));
            }
            io_base_upper::span!() => {
                self.registers
                    .io_base_upper
                    .set(op.pop_word(self.registers.io_base_upper.get()));
            }
            io_limit_upper::span!() => {
                self.registers
                    .io_limit_upper
                    .set(op.pop_word(self.registers.io_limit_upper.get()));
            }
            bridge_control::span!() => {
                let reg = LocalRegisterCopy::<u16, BridgeControl::Register>::new(
                    op.pop_word(self.registers.bridge_control.get()),
                );
                // TODO: We virtualize the secondary bus reset bit for now. Implement a VM-triggered
                // reset if necessary and it's safe to do so.
                self.virtual_bus_reset = reg.read(BridgeControl::SecondaryBusReset);
                self.registers.bridge_control.set(reg.writeable_bits());
            }
            _ => {
                op.pop_byte();
            }
        }
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

    /// Emulates a read from the configuration space of this device at `offset`.
    pub fn emulate_config_read(&self, offset: usize, len: usize) -> u32 {
        let mut op = MmioReadBuilder::new(offset, len);
        while !op.done() {
            let regs = self.common_registers();
            let info = self.info();
            use common_offsets::*;
            match op.offset() {
                vendor_id::span!() => {
                    op.push_word(info.vendor_id().0);
                }
                dev_id::span!() => {
                    op.push_word(info.device_id().0);
                }
                command::span!() => {
                    op.push_word(regs.command.readable_bits());
                }
                status::span!() => {
                    op.push_word(regs.status.readable_bits());
                }
                rev_id::span!() => {
                    op.push_byte(regs.rev_id.get());
                }
                prog_if::span!() => {
                    op.push_byte(regs.prog_if.get());
                }
                subclass::span!() => {
                    op.push_byte(info.subclass().0);
                }
                class::span!() => {
                    op.push_byte(info.class().0);
                }
                header_type::span!() => {
                    op.push_byte(regs.header_type.get());
                }
                PCI_TYPE_HEADER_START..=PCI_TYPE_HEADER_END => {
                    self.emulate_type_specific_read(&mut op);
                }
                offset => {
                    if offset <= PCI_COMMON_HEADER_END {
                        // Everything else in the common part of the header is unimplemented and we can
                        // safely return 0.
                        op.push_byte(0);
                    } else {
                        // Make everything beyond the standard header appear unimplemented.
                        //
                        // TODO: Capabilities.
                        op.push_dword(!0x0);
                    }
                }
            };
        }
        op.result()
    }

    /// Emulates a write to the configuration space of this device at `offset`.
    pub fn emulate_config_write(&mut self, offset: usize, value: u32, len: usize) {
        let mut op = MmioWriteBuilder::new(offset, value, len);
        while !op.done() {
            let regs = self.common_registers_mut();
            use common_offsets::*;
            match op.offset() {
                command::span!() => {
                    let mut reg = LocalRegisterCopy::<u16, Command::Register>::new(
                        op.pop_word(regs.command.get()),
                    );
                    // TODO: No DMA until the IOMMU is enabled.
                    reg.modify(Command::BusMasterEnable.val(0));
                    regs.command.set(reg.writeable_bits());
                }
                status::span!() => {
                    // Make sure we only write the RW1C bits if the write operation covers that byte.
                    let reg = LocalRegisterCopy::<u16, Status::Register>::new(
                        op.pop_word(regs.status.non_clearable_bits()),
                    );
                    regs.status.set(reg.writeable_bits());
                }
                PCI_TYPE_HEADER_START..=PCI_TYPE_HEADER_END => {
                    self.emulate_type_specific_write(&mut op);
                }
                _ => {
                    // We don't allow writes to other bits of the common header and everything beyond
                    // it is unimplemented for now.
                    //
                    // TODO: Capabilities.
                    op.pop_byte();
                }
            }
        }
    }

    // Returns a reference to the common portion of this device's PCI header.
    fn common_registers(&self) -> &CommonRegisters {
        match self {
            PciDevice::Endpoint(ep) => &ep.registers.common,
            PciDevice::Bridge(bridge) => &bridge.registers.common,
        }
    }

    // Returns a mutable reference to the common portion of this device's PCI header.
    fn common_registers_mut(&mut self) -> &mut CommonRegisters {
        match self {
            PciDevice::Endpoint(ep) => &mut ep.registers.common,
            PciDevice::Bridge(bridge) => &mut bridge.registers.common,
        }
    }

    // Emulates a read from the type-specific registers of this device.
    fn emulate_type_specific_read(&self, read_op: &mut MmioReadBuilder) {
        match self {
            PciDevice::Endpoint(ep) => ep.emulate_config_read(read_op),
            PciDevice::Bridge(bridge) => bridge.emulate_config_read(read_op),
        }
    }

    // Emulates a write to the type-specific registers of this device.
    fn emulate_type_specific_write(&mut self, write_op: &mut MmioWriteBuilder) {
        match self {
            PciDevice::Endpoint(ep) => ep.emulate_config_write(write_op),
            PciDevice::Bridge(bridge) => bridge.emulate_config_write(write_op),
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
