// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::fmt;
use core::mem::size_of;
use core::ptr::NonNull;
use page_tracking::PageTracker;
use riscv_pages::*;
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::registers::ReadWrite;
use tock_registers::LocalRegisterCopy;

use super::address::*;
use super::bus::PciBus;
use super::capabilities::*;
use super::error::*;
use super::mmio_builder::*;
use super::registers::*;
use super::resource::*;

/// The Vendor Id from the PCI header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct VendorId(u16);

impl VendorId {
    /// Creates a new `VendorId` from the raw `id`.
    pub fn new(id: u16) -> Self {
        VendorId(id)
    }

    /// The invalid (not-present) `VendorId`.
    pub const fn invalid() -> Self {
        VendorId(0xffff)
    }

    /// Returns the raw `VendorId` value.
    pub fn bits(&self) -> u16 {
        self.0
    }
}

/// The Device Id from the PCI header.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct DeviceId(u16);

impl DeviceId {
    /// Creates a new `DeviceId` from the raw `id`.
    pub fn new(id: u16) -> Self {
        DeviceId(id)
    }

    /// Returns the raw `DeviceId` value.
    pub fn bits(&self) -> u16 {
        self.0
    }
}

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

mod common_offsets {
    use super::CommonRegisters;
    use crate::define_field_span;

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
    use crate::define_field_span;

    define_field_span!(EndpointRegisters, bar, [u32; 6]);
    define_field_span!(EndpointRegisters, subsys_vendor_id, u16);
    define_field_span!(EndpointRegisters, subsys_id, u16);
    define_field_span!(EndpointRegisters, cap_ptr, u8);
}

mod bridge_offsets {
    use super::BridgeRegisters;
    use crate::define_field_span;

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

/// Describes a single PCI BAR.
#[derive(Clone, Debug)]
pub struct PciBarInfo {
    index: usize,
    bar_type: PciResourceType,
    size: u64,
}

impl PciBarInfo {
    /// Returns the index of this BAR.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the type of resource this BAR maps.
    pub fn bar_type(&self) -> PciResourceType {
        self.bar_type
    }

    /// Returns the size of this BAR.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Describes the BARs of a PCI device.
pub struct PciDeviceBarInfo {
    bars: ArrayVec<PciBarInfo, PCI_ENDPOINT_BARS>,
}

impl PciDeviceBarInfo {
    // Probes the size and type of each BAR from `registers`.
    fn new(registers: &mut [ReadWrite<u32, BaseAddress::Register>]) -> Result<Self> {
        let mut bars = ArrayVec::new();
        let mut index = 0;
        while index < registers.len() {
            let bar_index = index;
            let bar_type = PciResourceType::from_bar_register(registers[index].extract());

            // Write all 1s to detect the number of bits that are implemented.
            registers[index].set(!0);
            let val = registers[index].get();
            registers[index].set(0);
            index += 1;
            if val == 0 {
                // If we read back all 0s, the BAR is unimplemented.
                continue;
            }
            let bits_lo = val & !((1u32 << BaseAddress::Address.shift) - 1);
            let bits_hi = if bar_type.is_64bit() {
                // For 64-bit BARs the upper bits are in the adjacent register.
                if bar_index % 2 != 0 {
                    return Err(Error::Invalid64BitBarIndex);
                }
                registers[index].set(!0);
                let val = registers[index].get();
                registers[index].set(0);
                index += 1;
                val
            } else {
                !0
            };
            let size = !((bits_lo as u64) | ((bits_hi as u64) << 32)) + 1;
            if !size.is_power_of_two() {
                return Err(Error::InvalidBarSize(size));
            }

            let bar = PciBarInfo {
                index: bar_index,
                bar_type,
                size,
            };
            bars.push(bar);
        }

        Ok(Self { bars })
    }

    /// Returns an iterator over this device's BARs.
    pub fn bars(&self) -> impl ExactSizeIterator<Item = &PciBarInfo> {
        self.bars.iter()
    }

    // Returns the `PciBarInfo` with the given BAR index.
    fn get(&self, index: usize) -> Option<&PciBarInfo> {
        self.bars.iter().find(|b| b.index() == index)
    }

    // Returns the type of the BAR at `index`.
    fn index_to_type(&self, index: usize) -> Option<PciResourceType> {
        // If `index` is the upper half of a 64-bit BAR, return the type of the lower half.
        self.bars
            .iter()
            .find(|b| b.index() == index || (b.index() + 1 == index && b.bar_type().is_64bit()))
            .map(|b| b.bar_type())
    }
}

// Common state between bridges and endpoints.
struct PciDeviceCommon {
    info: PciDeviceInfo,
    capabilities: PciCapabilities,
    bar_info: PciDeviceBarInfo,
    owner: Option<PageOwnerId>,
}

/// Represents a PCI endpoint.
pub struct PciEndpoint {
    registers: &'static mut EndpointRegisters,
    common: PciDeviceCommon,
}

impl PciEndpoint {
    /// Creates a new `PciEndpoint` using the config space at `registers`.
    fn new(registers: &'static mut EndpointRegisters, info: PciDeviceInfo) -> Result<Self> {
        let capabilities =
            PciCapabilities::new(&mut registers.common, registers.cap_ptr.get() as usize)?;
        let bar_info = PciDeviceBarInfo::new(&mut registers.bar)?;
        let common = PciDeviceCommon {
            info,
            capabilities,
            bar_info,
            owner: None,
        };
        Ok(Self { registers, common })
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
                op.push_byte(self.common.capabilities.start_offset() as u8);
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
                // Discard BAR writes if the BAR is enabled.
                let io_enabled = self.registers.common.command.is_set(Command::IoEnable);
                let mem_enabled = self.registers.common.command.is_set(Command::MemoryEnable);
                if let Some(bar_type) = self.common.bar_info.index_to_type(index) &&
                    ((bar_type == PciResourceType::IoPort && io_enabled) ||
                     (bar_type != PciResourceType::IoPort && mem_enabled))
                {
                    return;
                }
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
    common: PciDeviceCommon,
    bus_range: BusRange,
    child_bus: Option<PciBus>,
    virtual_primary_bus: Bus,
    virtual_bus_reset: u16,
    has_io_window: bool,
    has_pref_window: bool,
}

impl PciBridge {
    /// Creates a new `PciBridge` use the config space at `registers`. Downstream buses are initially
    /// unenumerated.
    fn new(registers: &'static mut BridgeRegisters, info: PciDeviceInfo) -> Result<Self> {
        // Prevent config cycles from passing beyond this bridge until we're ready to enumreate.
        registers.sub_bus.set(0);
        registers.sec_bus.set(0);
        registers.pri_bus.set(0);
        // Check if the IO and prefetchable memory windows are implemented. We need to do this by
        // checking if the registers are writeable since 0 is a valid base and limit.
        let has_io_window = {
            registers.io_limit.set(!0);
            let val = registers.io_limit.get();
            registers.io_limit.set(0);
            val != 0
        };
        let has_pref_window = {
            registers.pref_limit.set(!0);
            let val = registers.pref_limit.get();
            registers.pref_limit.set(0);
            val != 0
        };
        let capabilities =
            PciCapabilities::new(&mut registers.common, registers.cap_ptr.get() as usize)?;
        let bar_info = PciDeviceBarInfo::new(&mut registers.bar)?;
        let common = PciDeviceCommon {
            info,
            capabilities,
            bar_info,
            owner: None,
        };
        Ok(Self {
            registers,
            common,
            bus_range: BusRange::default(),
            child_bus: None,
            virtual_primary_bus: Bus::default(),
            virtual_bus_reset: 0,
            has_io_window,
            has_pref_window,
        })
    }

    /// Configures the secondary and subordinate bus numbers of the bridge such that configuration
    /// cycles from `range.start` to `range.end` (inclusive) will be forwarded downstream.
    pub fn assign_bus_range(&mut self, range: BusRange) {
        self.registers.sub_bus.set(range.end.bits() as u8);
        self.registers.sec_bus.set(range.start.bits() as u8);
        let pri_bus = self.common.info.address().bus();
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
                op.push_byte(self.common.capabilities.start_offset() as u8);
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
        let io_enabled = self.registers.common.command.is_set(Command::IoEnable);
        let mem_enabled = self.registers.common.command.is_set(Command::MemoryEnable);
        use bridge_offsets::*;
        match op.offset() {
            bar::span!() => {
                let index = (op.offset() - bar::START_OFFSET) / size_of::<u32>();
                let reg = op.pop_dword(self.registers.bar[index].get());
                // Discard BAR writes if the BAR is enabled.
                if let Some(bar_type) = self.common.bar_info.index_to_type(index) &&
                    ((bar_type == PciResourceType::IoPort && io_enabled) ||
                     (bar_type != PciResourceType::IoPort && mem_enabled))
                {
                    return;
                }
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
                let reg = op.pop_byte();
                if !io_enabled {
                    self.registers.io_base.set(reg);
                }
            }
            io_limit::span!() => {
                let reg = op.pop_byte();
                if !io_enabled {
                    self.registers.io_limit.set(reg);
                }
            }
            sec_status::span!() => {
                let reg = LocalRegisterCopy::<u16, SecondaryStatus::Register>::new(
                    op.pop_word(self.registers.sec_status.non_clearable_bits()),
                );
                self.registers.sec_status.set(reg.writeable_bits());
            }
            mem_base::span!() => {
                let reg = op.pop_word(self.registers.mem_base.get());
                if !mem_enabled {
                    self.registers.mem_base.set(reg);
                }
            }
            mem_limit::span!() => {
                let reg = op.pop_word(self.registers.mem_limit.get());
                if !mem_enabled {
                    self.registers.mem_limit.set(reg);
                }
            }
            pref_base::span!() => {
                let reg = op.pop_word(self.registers.pref_base.get());
                if !mem_enabled {
                    self.registers.pref_base.set(reg);
                }
            }
            pref_limit::span!() => {
                let reg = op.pop_word(self.registers.pref_limit.get());
                if !mem_enabled {
                    self.registers.pref_limit.set(reg);
                }
            }
            pref_base_upper::span!() => {
                let reg = op.pop_dword(self.registers.pref_base_upper.get());
                if !mem_enabled {
                    self.registers.pref_base_upper.set(reg);
                }
            }
            pref_limit_upper::span!() => {
                let reg = op.pop_dword(self.registers.pref_limit_upper.get());
                if !mem_enabled {
                    self.registers.pref_limit_upper.set(reg);
                }
            }
            io_base_upper::span!() => {
                let reg = op.pop_word(self.registers.io_base_upper.get());
                if !io_enabled {
                    self.registers.io_base_upper.set(reg);
                }
            }
            io_limit_upper::span!() => {
                let reg = op.pop_word(self.registers.io_limit_upper.get());
                if !io_enabled {
                    self.registers.io_limit_upper.set(reg);
                }
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

    // Returns the base and limit of the bridge's IO window if it's implemented and enabled.
    fn get_io_window(&self) -> Option<(u64, u64)> {
        const SHIFT: usize = 12;
        if self.has_io_window {
            let base = {
                let lo = self.registers.io_base.read(IoWindow::Address) as u64;
                let hi = self.registers.io_base_upper.get() as u64;
                (hi << 16) | (lo << SHIFT)
            };
            let limit = {
                let lo = self.registers.io_limit.read(IoWindow::Address) as u64;
                let hi = self.registers.io_limit_upper.get() as u64;
                (hi << 16) | (lo << SHIFT) | ((1u64 << SHIFT) - 1)
            };
            if limit < base {
                None
            } else {
                Some((base, limit))
            }
        } else {
            None
        }
    }

    // Returns the base and limit of the bridge's 32-bit memory window if it's enabled.
    fn get_mem_window(&self) -> Option<(u64, u64)> {
        const SHIFT: usize = 20;
        let base = (self.registers.mem_base.read(MemWindow::Address) as u64) << SHIFT;
        let limit = ((self.registers.mem_limit.read(MemWindow::Address) as u64) << SHIFT)
            | ((1u64 << SHIFT) - 1);
        if limit < base {
            None
        } else {
            Some((base, limit))
        }
    }

    // Returns the base and limit of the bridge's prefetchable memory window if it's implemented and
    // enabled.
    fn get_pref_window(&self) -> Option<(u64, u64)> {
        const SHIFT: usize = 20;
        if self.has_pref_window {
            let base = {
                let lo = self.registers.pref_base.read(MemWindow::Address) as u64;
                let hi = self.registers.pref_base_upper.get() as u64;
                (hi << 32) | (lo << SHIFT)
            };
            let limit = {
                let lo = self.registers.pref_limit.read(MemWindow::Address) as u64;
                let hi = self.registers.pref_limit_upper.get() as u64;
                (hi << 32) | (lo << SHIFT) | ((1u64 << SHIFT) - 1)
            };
            if limit < base {
                None
            } else {
                Some((base, limit))
            }
        } else {
            None
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

// Returns `Ok` if `range` is PCI BAR memory owned by `guest_id`.
fn bar_range_is_owned(
    range: SupervisorPageRange,
    page_tracker: &PageTracker,
    guest_id: PageOwnerId,
) -> Result<()> {
    for p in range {
        if !page_tracker.is_mapped_page(p, guest_id, MemType::Mmio(DeviceMemType::PciBar)) {
            return Err(Error::UnownedBarPage(p));
        }
    }
    Ok(())
}

// Returns `Ok` if the specified bridge window is assigned a valid address for the VM in `context`.
fn bridge_window_is_valid(base: u64, limit: u64, context: &MmioEmulationContext) -> Result<()> {
    let phys_addr = context
        .resources
        .pci_to_physical_addr(base)
        .ok_or(Error::InvalidBarAddress(base))?;
    let page_range = SupervisorPageRange::new(
        PageAddr::with_round_down(phys_addr, PageSize::Size4k),
        PageSize::num_4k_pages(limit - base),
    );
    bar_range_is_owned(page_range, &context.page_tracker, context.guest_id)
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
        &self.common().info
    }

    /// Returns the `PciDeviceBarInfo` for this device.
    pub fn bar_info(&self) -> &PciDeviceBarInfo {
        &self.common().bar_info
    }

    /// Returns if the device supports MSI.
    pub fn has_msi(&self) -> bool {
        self.common().capabilities.has_msi()
    }

    /// Returns if the device supports MSI-X.
    pub fn has_msix(&self) -> bool {
        self.common().capabilities.has_msix()
    }

    /// Returns if the device is a PCI-Express device.
    pub fn is_pcie(&self) -> bool {
        self.common().capabilities.is_pcie()
    }

    /// Returns the device's owner.
    pub fn owner(&self) -> Option<PageOwnerId> {
        self.common().owner
    }

    /// Takes ownership over the device if it is not already owned.
    pub fn take(&mut self, owner: PageOwnerId) -> Result<()> {
        if self.owner().is_some() {
            return Err(Error::DeviceOwned);
        }
        self.common_mut().owner = Some(owner);
        Ok(())
    }

    /// Emulates a read from the configuration space of this device at `offset`.
    pub fn emulate_config_read(
        &self,
        offset: usize,
        len: usize,
        _context: MmioEmulationContext,
    ) -> u32 {
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
                    let mut reg = LocalRegisterCopy::<u16, Status::Register>::new(
                        regs.status.readable_bits(),
                    );
                    // We always virtualize a capability list, so make sure CAP_LIST is set.
                    reg.modify(Status::CapabilitiesList.val(1));
                    op.push_word(reg.get());
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
                PCI_CAPS_START..=PCI_CONFIG_SPACE_END => {
                    self.common().capabilities.emulate_read(&mut op);
                }
                offset => {
                    if offset <= PCI_COMMON_HEADER_END {
                        // Everything else in the common part of the header is unimplemented and we can
                        // safely return 0.
                        op.push_byte(0);
                    } else {
                        // Make everything beyond the standard PCI configuration space appear
                        // unimplemented.
                        //
                        // TODO: Extended config space emulation?
                        op.push_dword(!0x0);
                    }
                }
            };
        }
        op.result()
    }

    /// Emulates a write to the configuration space of this device at `offset`.
    pub fn emulate_config_write(
        &mut self,
        offset: usize,
        value: u32,
        len: usize,
        context: MmioEmulationContext,
    ) {
        let mut op = MmioWriteBuilder::new(offset, value, len);
        while !op.done() {
            use common_offsets::*;
            match op.offset() {
                command::span!() => {
                    let mut reg = LocalRegisterCopy::<u16, Command::Register>::new(
                        op.pop_word(self.common_registers().command.get()),
                    );

                    // Check that the VM has assigned valid BARs / bridge windows for this device
                    // before allowing it to enable IO or memory space access.
                    if reg.is_set(Command::IoEnable) && self.can_enable_io_space(&context).is_err()
                    {
                        reg.modify(Command::IoEnable.val(0));
                    }
                    if reg.is_set(Command::MemoryEnable)
                        && self.can_enable_mem_space(&context).is_err()
                    {
                        reg.modify(Command::MemoryEnable.val(0));
                    }

                    // TODO: No DMA until the IOMMU is enabled.
                    reg.modify(Command::BusMasterEnable.val(0));
                    self.common_registers().command.set(reg.writeable_bits());
                }
                status::span!() => {
                    // Make sure we only write the RW1C bits if the write operation covers that byte.
                    let reg = LocalRegisterCopy::<u16, Status::Register>::new(
                        op.pop_word(self.common_registers().status.non_clearable_bits()),
                    );
                    self.common_registers().status.set(reg.writeable_bits());
                }
                PCI_TYPE_HEADER_START..=PCI_TYPE_HEADER_END => {
                    self.emulate_type_specific_write(&mut op);
                }
                PCI_CAPS_START..=PCI_CONFIG_SPACE_END => {
                    self.common_mut().capabilities.emulate_write(&mut op);
                }
                _ => {
                    // We don't allow writes to other bits of the common header and everything beyond
                    // the standard config space is unimplemented.
                    //
                    // TODO: Extended config space emulation?
                    op.pop_byte();
                }
            }
        }
    }

    // Returns the PCI bus address programmed in the BAR at `bar_index`.
    fn get_bar_addr(&self, index: usize) -> Result<u64> {
        let bar = self
            .bar_info()
            .get(index)
            .ok_or(Error::BarNotPresent(index))?;
        let regs = self.bar_registers();
        let addr_lo = regs[index].get() & !((1u32 << BaseAddress::Address.shift) - 1);
        let addr_hi = if bar.bar_type().is_64bit() {
            regs[index + 1].get()
        } else {
            0
        };
        Ok((addr_lo as u64) | ((addr_hi as u64) << 32))
    }

    // Returns `Ok` if the specified BAR is assigned a valid address for the VM in `context`.
    fn bar_assignment_is_valid(
        &self,
        bar: &PciBarInfo,
        context: &MmioEmulationContext,
    ) -> Result<()> {
        // Unwrap ok: BAR index is guaranteed to be valid since it's in `self.bar_info`.
        let pci_addr = self.get_bar_addr(bar.index()).unwrap();
        let phys_addr = context
            .resources
            .pci_to_physical_addr(pci_addr)
            .ok_or(Error::InvalidBarAddress(pci_addr))?;
        let page_range = SupervisorPageRange::new(
            PageAddr::with_round_down(phys_addr, PageSize::Size4k),
            PageSize::num_4k_pages(bar.size()),
        );
        bar_range_is_owned(page_range, &context.page_tracker, context.guest_id)
    }

    // Returns `Ok` if IO space access can safely be enabled for this device.
    fn can_enable_io_space(&self, context: &MmioEmulationContext) -> Result<()> {
        self.bar_info()
            .bars()
            .filter(|b| b.bar_type() == PciResourceType::IoPort)
            .try_for_each(|b| self.bar_assignment_is_valid(b, context))?;

        if let PciDevice::Bridge(bridge) = self {
            if let Some((base, limit)) = bridge.get_io_window() {
                bridge_window_is_valid(base, limit, context)?;
            }
        }

        Ok(())
    }

    // Returns `Ok` if memory space access can safely be enabled for this device.
    fn can_enable_mem_space(&self, context: &MmioEmulationContext) -> Result<()> {
        self.bar_info()
            .bars()
            .filter(|b| b.bar_type() != PciResourceType::IoPort)
            .try_for_each(|b| self.bar_assignment_is_valid(b, context))?;

        if let PciDevice::Bridge(bridge) = self {
            if let Some((base, limit)) = bridge.get_mem_window() {
                bridge_window_is_valid(base, limit, context)?;
            }
            if let Some((base, limit)) = bridge.get_pref_window() {
                bridge_window_is_valid(base, limit, context)?;
            }
        }

        Ok(())
    }

    // Returns a reference to the common portion of this device's PCI header.
    fn common_registers(&self) -> &CommonRegisters {
        match self {
            PciDevice::Endpoint(ep) => &ep.registers.common,
            PciDevice::Bridge(bridge) => &bridge.registers.common,
        }
    }

    // Returns a reference to this device's BAR registers.
    fn bar_registers(&self) -> &[ReadWrite<u32, BaseAddress::Register>] {
        match self {
            PciDevice::Endpoint(ep) => &ep.registers.bar,
            PciDevice::Bridge(bridge) => &bridge.registers.bar,
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

    // Returns a reference to the `PciDeviceCommon` for this device.
    fn common(&self) -> &PciDeviceCommon {
        match self {
            PciDevice::Endpoint(ep) => &ep.common,
            PciDevice::Bridge(bridge) => &bridge.common,
        }
    }

    // Returns a mutable reference to the `PciDeviceCommon` for this device.
    fn common_mut(&mut self) -> &mut PciDeviceCommon {
        match self {
            PciDevice::Endpoint(ep) => &mut ep.common,
            PciDevice::Bridge(bridge) => &mut bridge.common,
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
