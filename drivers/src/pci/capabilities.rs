// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::mem::size_of;
use enum_dispatch::enum_dispatch;
use memoffset::offset_of;
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::registers::ReadOnly;
use tock_registers::LocalRegisterCopy;

use super::error::*;
use super::mmio_builder::{MmioReadBuilder, MmioWriteBuilder};
use super::registers::*;
use super::resource::*;

// Standard PCI capability IDs.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CapabilityId {
    PowerManagement = 1,
    Msi = 5,
    Vendor = 9,
    BridgeSubsystem = 13,
    PciExpress = 16,
    MsiX = 17,
    EnhancedAllocation = 20,
}

impl CapabilityId {
    // Returns the `CapabilityId` from the raw register value.
    fn from_raw(id: u8) -> Option<Self> {
        use CapabilityId::*;
        match id {
            1 => Some(PowerManagement),
            5 => Some(Msi),
            9 => Some(Vendor),
            13 => Some(BridgeSubsystem),
            16 => Some(PciExpress),
            17 => Some(MsiX),
            20 => Some(EnhancedAllocation),
            _ => None,
        }
    }
}

mod header_offsets {
    use super::CapabilityHeader;
    use crate::define_field_span;

    define_field_span!(CapabilityHeader, cap_id, u8);
    define_field_span!(CapabilityHeader, next_cap, u8);
}

mod pmc_offsets {
    use super::PowerManagementRegisters;
    use crate::define_field_span;

    define_field_span!(PowerManagementRegisters, pmc, u16);
    define_field_span!(PowerManagementRegisters, pmcsr, u16);
}

mod msi_offsets {
    use super::MsiRegisters;
    use crate::define_field_span;

    define_field_span!(MsiRegisters, msg_control, u16);
    define_field_span!(MsiRegisters, msg_addr, u32);
    define_field_span!(MsiRegisters, msg_upper_addr, u32);
    define_field_span!(MsiRegisters, msg_data, u16);
    define_field_span!(MsiRegisters, extended_msg_data, u16);
    define_field_span!(MsiRegisters, mask_bits, u32);
    define_field_span!(MsiRegisters, pending_bits, u32);
}

mod msix_offsets {
    use super::MsiXRegisters;
    use crate::define_field_span;

    define_field_span!(MsiXRegisters, msg_control, u16);
    define_field_span!(MsiXRegisters, table_offset, u32);
    define_field_span!(MsiXRegisters, pba_offset, u32);
}

mod vendor_offsets {
    use super::VendorCapabilityHeader;
    use crate::define_field_span;

    define_field_span!(VendorCapabilityHeader, cap_length, u8);
}

mod bridge_subsys_offsets {
    use super::BridgeSubsystemRegisters;
    use crate::define_field_span;

    define_field_span!(BridgeSubsystemRegisters, ssvid, u16);
    define_field_span!(BridgeSubsystemRegisters, ssid, u16);
}

mod express_offsets {
    use super::ExpressRegisters;
    use crate::define_field_span;

    define_field_span!(ExpressRegisters, exp_caps, u16);
    define_field_span!(ExpressRegisters, dev_caps, u32);
    define_field_span!(ExpressRegisters, dev_control, u16);
    define_field_span!(ExpressRegisters, link_caps, u32);
    define_field_span!(ExpressRegisters, link_status, u16);
}

// Type-specific capability structures.
#[enum_dispatch]
enum CapabilityType {
    PowerManagement,
    Msi,
    MsiX,
    Vendor,
    BridgeSubsystem,
    PciExpress,
    EnhancedAllocation,
}

// Common functionality required by all capabilities.
#[enum_dispatch(CapabilityType)]
trait Capability {
    // Returns the length of the capability, including the common header.
    fn length(&self) -> usize;

    // Emulates a read from the type-specific registers of this capability.
    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize);

    // Emulates a write to the type-specific registers of this capability.
    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, cap_offset: usize);
}

struct PowerManagement {
    registers: &'static mut PowerManagementRegisters,
}

impl PowerManagement {
    fn new(header: &mut CapabilityHeader) -> Self {
        // Safety: `header` points to a valid and unqiuely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let registers = unsafe {
            (header as *mut CapabilityHeader as *mut PowerManagementRegisters)
                .as_mut()
                .unwrap()
        };
        Self { registers }
    }
}

impl Capability for PowerManagement {
    fn length(&self) -> usize {
        size_of::<PowerManagementRegisters>()
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize) {
        use pmc_offsets::*;
        match cap_offset {
            pmc::span!() => {
                op.push_word(self.registers.pmc.readable_bits());
            }
            pmcsr::span!() => {
                op.push_word(self.registers.pmcsr.readable_bits());
            }
            _ => {
                op.push_byte(0);
            }
        }
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, _cap_offset: usize) {
        // TODO: Support PME and D-state transitions if necessary.
        op.pop_byte();
    }
}

struct Msi {
    registers: &'static mut MsiRegisters,
    per_vector_masks: bool,
}

impl Msi {
    fn new(header: &mut CapabilityHeader) -> Result<Self> {
        // Safety: `header` points to a valid and unqiuely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let registers = unsafe {
            (header as *mut CapabilityHeader as *mut MsiRegisters)
                .as_mut()
                .unwrap()
        };
        // Keep things simple by only supporting 64-bit MSI. Basically all devices emulated by QEMU
        // use 64-bit MSI anyway.
        if registers
            .msg_control
            .read(MsiMessageControl::Address64BitCapable)
            == 0
        {
            return Err(Error::MsiNot64BitCapable);
        }
        let per_vector_masks = registers
            .msg_control
            .read(MsiMessageControl::VectorMaskingCapable)
            != 0;
        Ok(Self {
            registers,
            per_vector_masks,
        })
    }
}

impl Capability for Msi {
    fn length(&self) -> usize {
        if self.per_vector_masks {
            size_of::<MsiRegisters>()
        } else {
            offset_of!(MsiRegisters, mask_bits)
        }
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize) {
        use msi_offsets::*;
        match cap_offset {
            msg_control::span!() => {
                op.push_word(self.registers.msg_control.readable_bits());
            }
            msg_addr::span!() => {
                op.push_dword(self.registers.msg_addr.get());
            }
            msg_upper_addr::span!() => {
                op.push_dword(self.registers.msg_upper_addr.get());
            }
            msg_data::span!() => {
                op.push_word(self.registers.msg_data.get());
            }
            extended_msg_data::span!() => {
                op.push_word(self.registers.extended_msg_data.get());
            }
            mask_bits::span!() if self.per_vector_masks => {
                op.push_dword(self.registers.mask_bits.get());
            }
            pending_bits::span!() if self.per_vector_masks => {
                op.push_dword(self.registers.pending_bits.get());
            }
            _ => {
                op.push_byte(0);
            }
        }
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, cap_offset: usize) {
        use msi_offsets::*;
        match cap_offset {
            msg_control::span!() => {
                let reg = LocalRegisterCopy::<u16, MsiMessageControl::Register>::new(
                    op.pop_word(self.registers.msg_control.get()),
                );
                self.registers.msg_control.set(reg.writeable_bits());
            }
            msg_addr::span!() => {
                let reg = op.pop_dword(self.registers.msg_addr.get());
                self.registers.msg_addr.set(reg);
            }
            msg_upper_addr::span!() => {
                let reg = op.pop_dword(self.registers.msg_upper_addr.get());
                self.registers.msg_upper_addr.set(reg);
            }
            msg_data::span!() => {
                let reg = op.pop_word(self.registers.msg_data.get());
                self.registers.msg_data.set(reg);
            }
            extended_msg_data::span!() => {
                let reg = op.pop_word(self.registers.extended_msg_data.get());
                self.registers.extended_msg_data.set(reg);
            }
            mask_bits::span!() if self.per_vector_masks => {
                let reg = op.pop_dword(self.registers.mask_bits.get());
                self.registers.mask_bits.set(reg);
            }
            _ => {
                op.pop_byte();
            }
        }
    }
}

struct MsiX {
    registers: &'static mut MsiXRegisters,
}

impl MsiX {
    fn new(header: &mut CapabilityHeader) -> Self {
        // Safety: `header` points to a valid and unqiuely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let registers = unsafe {
            (header as *mut CapabilityHeader as *mut MsiXRegisters)
                .as_mut()
                .unwrap()
        };
        Self { registers }
    }
}

impl Capability for MsiX {
    fn length(&self) -> usize {
        size_of::<MsiXRegisters>()
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize) {
        use msix_offsets::*;
        match cap_offset {
            msg_control::span!() => {
                op.push_word(self.registers.msg_control.readable_bits());
            }
            table_offset::span!() => {
                op.push_dword(self.registers.table_offset.get());
            }
            pba_offset::span!() => {
                op.push_dword(self.registers.pba_offset.get());
            }
            _ => {
                op.push_byte(0);
            }
        }
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, cap_offset: usize) {
        use msix_offsets::*;
        match cap_offset {
            msg_control::span!() => {
                let reg = LocalRegisterCopy::<u16, MsiXMessageControl::Register>::new(
                    op.pop_word(self.registers.msg_control.get()),
                );
                self.registers.msg_control.set(reg.writeable_bits());
            }
            _ => {
                op.pop_byte();
            }
        }
    }
}

struct Vendor {
    header: &'static mut VendorCapabilityHeader,
    length: usize,
}

impl Vendor {
    fn new(header: &mut CapabilityHeader) -> Result<Self> {
        // Safety: `header` points to a valid and unqiuely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let vendor_header = unsafe {
            (header as *mut CapabilityHeader as *mut VendorCapabilityHeader)
                .as_mut()
                .unwrap()
        };
        let length = vendor_header.cap_length.get() as usize;
        // Sanity check the reported length.
        if length < size_of::<VendorCapabilityHeader>() || length > PCI_MAX_CAP_LENGTH {
            return Err(Error::InvalidVendorCapabilityLength(length));
        }
        Ok(Self {
            header: vendor_header,
            length,
        })
    }
}

impl Capability for Vendor {
    fn length(&self) -> usize {
        self.length
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize) {
        use vendor_offsets::*;
        match cap_offset {
            cap_length::span!() => {
                op.push_byte(self.length as u8);
            }
            x if cap_length::END_OFFSET < x && x < self.length => {
                // Safety: We've verified that the read offset is within the bounds of the capability
                // structure. Further, byte-sized reads are always valid in PCI configuration space.
                let reg = unsafe {
                    let ptr = (self.header as *const VendorCapabilityHeader as *const u8).add(x);
                    core::ptr::read_volatile(ptr)
                };
                op.push_byte(reg);
            }
            _ => {
                op.push_byte(0);
            }
        }
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, _cap_offset: usize) {
        // The vendor capability structure is opaque, so treat all the registers as read-only.
        op.pop_byte();
    }
}

struct BridgeSubsystem {
    registers: &'static mut BridgeSubsystemRegisters,
}

impl BridgeSubsystem {
    fn new(header: &mut CapabilityHeader) -> Self {
        // Safety: `header` points to a valid and unqiuely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let registers = unsafe {
            (header as *mut CapabilityHeader as *mut BridgeSubsystemRegisters)
                .as_mut()
                .unwrap()
        };
        Self { registers }
    }
}

impl Capability for BridgeSubsystem {
    fn length(&self) -> usize {
        size_of::<BridgeSubsystemRegisters>()
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize) {
        use bridge_subsys_offsets::*;
        match cap_offset {
            ssvid::span!() => {
                op.push_word(self.registers.ssvid.get());
            }
            ssid::span!() => {
                op.push_word(self.registers.ssid.get());
            }
            _ => {
                op.push_byte(0);
            }
        }
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, _cap_offset: usize) {
        op.pop_byte();
    }
}

// The maximum number of extend allocation entries. The capability's num_entries field allows up to
// 63 entries, but we have no reason to support that many.
const MAX_ENHANCED_ALLOCATION_ENTRIES: usize = 8;

pub struct EnhancedAllocationEntry {
    fixed: &'static mut EnhancedAllocationEntryFixed,
    base_high: Option<&'static mut ReadOnly<u32>>,
    max_offset_high: Option<&'static mut ReadOnly<u32>>,
}

impl EnhancedAllocationEntry {
    pub fn enabled(&self) -> bool {
        self.fixed
            .header
            .read(EnhancedAllocationEntryHeader::Enable)
            != 0
    }

    pub fn bar_equivalent_indicator(&self) -> usize {
        self.fixed
            .header
            .read(EnhancedAllocationEntryHeader::BarEquivalentIndicator) as usize
    }

    fn properties(&self) -> Option<EnhancedAllocationEntryHeader::PrimaryProperties::Value> {
        use EnhancedAllocationEntryHeader::*;
        self.fixed
            .header
            .read_as_enum(PrimaryProperties)
            .or_else(|| self.fixed.header.read_as_enum(SecondaryProperties))
    }

    pub fn resource_type(&self) -> Option<PciResourceType> {
        use EnhancedAllocationEntryHeader::PrimaryProperties::Value::*;
        Some(match self.properties()? {
            Mem => PciResourceType::Mem64,
            PrefetchableMem => PciResourceType::PrefetchableMem64,
            IoPort => PciResourceType::IoPort,
            _ => return None,
        })
    }

    pub fn base(&self) -> u64 {
        use EnhancedAllocationEntryAddress::Address;
        let low = self.fixed.base.read(Address) << Address.shift;
        let high = self.base_high.as_ref().map(|r| r.get()).unwrap_or(0);
        low as u64 | ((high as u64) << 32)
    }

    pub fn max_offset(&self) -> u64 {
        use EnhancedAllocationEntryAddress::Address;
        let low = self.fixed.max_offset.read(Address) << Address.shift;
        // Fill in the low-order bits.
        let low = low | !(Address.mask << Address.shift);
        let high = self.max_offset_high.as_ref().map(|r| r.get()).unwrap_or(0);
        low as u64 | ((high as u64) << 32)
    }
}

pub struct EnhancedAllocation {
    pub entries: ArrayVec<EnhancedAllocationEntry, MAX_ENHANCED_ALLOCATION_ENTRIES>,
    length: usize,
}

impl EnhancedAllocation {
    fn new(config_regs: &mut CommonRegisters, header: &mut CapabilityHeader) -> Self {
        // Safety: `header` points to a valid and uniquely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let header = unsafe {
            (header as *mut CapabilityHeader as *mut EnhancedAllocationHeader)
                .as_mut()
                .unwrap()
        };

        let mut length = size_of::<EnhancedAllocationHeader>();
        if config_regs.header_type.read(Type::Layout) == 1 {
            length += size_of::<u32>();
        }

        let num_entries = header
            .num_entries
            .read(EnhancedAllocationNumEntries::NumEntries) as usize;
        let mut entries = ArrayVec::new();
        for _ in 0..core::cmp::min(num_entries, MAX_ENHANCED_ALLOCATION_ENTRIES) {
            // Safety: `header` points to a valid capability structure and we are trusting that the
            // hardware-supplied num_entries field doesn't move us outside the ECAM memory region.
            let fixed = unsafe {
                ((header as *mut EnhancedAllocationHeader).byte_add(length)
                    as *mut EnhancedAllocationEntryFixed)
                    .as_mut()
                    .unwrap()
            };

            let num_dw = fixed.header.read(EnhancedAllocationEntryHeader::Size) as usize + 1;
            let size = num_dw * size_of::<u32>();
            length += size;

            if size < size_of::<EnhancedAllocationEntryFixed>() {
                // entry too short, ignore.
                continue;
            }

            // Compute references to optional registers following the fixed part.
            let ptr = fixed as *mut EnhancedAllocationEntryFixed;
            let mut regs = (size_of::<EnhancedAllocationEntryFixed>()..size)
                .step_by(size_of::<u32>())
                // Safety: the memory location is within the entry of a valid capability structure,
                // and we trust hardware with ECAM layout.
                .map(|pos| unsafe { (ptr.byte_add(pos) as *mut ReadOnly<u32>).as_mut().unwrap() });

            use EnhancedAllocationEntryAddress::Size;
            let base_high = (fixed.base.read(Size) == 1).then(|| regs.next()).flatten();
            let max_offset_high = (fixed.max_offset.read(Size) == 1)
                .then(|| regs.next())
                .flatten();

            entries.push(EnhancedAllocationEntry {
                fixed,
                base_high,
                max_offset_high,
            });
        }

        Self { entries, length }
    }
}

impl Capability for EnhancedAllocation {
    fn length(&self) -> usize {
        self.length
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, _cap_offset: usize) {
        // TODO: Implement reading if necessary.
        op.push_byte(0);
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, _cap_offset: usize) {
        // TODO: support entry enable bit writes if necessary.
        op.pop_byte();
    }
}

// The possible PCI express device types as reported in the FLAGS register of the PCI express
// capabilities register.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PciExpressDeviceType {
    Endpoint = 0,
    LegacyEndpoint = 1,
    RootPort = 4,
    UpstreamSwitchPort = 5,
    DownstreamSwitchPort = 6,
    PciExpressToPciBridge = 7,
    PciToPciExpressBridge = 8,
    RootComplexIntegratedEndpoint = 9,
    RootComplexEventCollector = 10,
}

impl PciExpressDeviceType {
    // Returns the device type corresponding to the raw code.
    fn from_raw(raw: u8) -> Option<Self> {
        use PciExpressDeviceType::*;
        match raw {
            0 => Some(Endpoint),
            1 => Some(LegacyEndpoint),
            4 => Some(RootPort),
            5 => Some(UpstreamSwitchPort),
            6 => Some(DownstreamSwitchPort),
            7 => Some(PciExpressToPciBridge),
            8 => Some(PciToPciExpressBridge),
            9 => Some(RootComplexIntegratedEndpoint),
            10 => Some(RootComplexEventCollector),
            _ => None,
        }
    }

    // Returns if this device type implements the root control and status registers.
    fn has_root_control(&self) -> bool {
        use PciExpressDeviceType::*;
        matches!(
            self,
            RootComplexIntegratedEndpoint | RootComplexEventCollector
        )
    }

    // Returns if this device type implements the link control and status registers.
    fn has_link_control(&self) -> bool {
        use PciExpressDeviceType::*;
        matches!(
            self,
            Endpoint
                | LegacyEndpoint
                | RootPort
                | UpstreamSwitchPort
                | DownstreamSwitchPort
                | PciExpressToPciBridge
                | PciToPciExpressBridge
        )
    }
}

struct PciExpress {
    registers: &'static mut ExpressRegisters,
    version: u8,
    device_type: PciExpressDeviceType,
}

impl PciExpress {
    fn new(header: &mut CapabilityHeader) -> Result<Self> {
        // Safety: `header` points to a valid and unqiuely-owned capability structure and we are
        // trusting that the hardware reported the type of the capability correctly.
        let registers = unsafe {
            (header as *mut CapabilityHeader as *mut ExpressRegisters)
                .as_mut()
                .unwrap()
        };
        let version = registers.exp_caps.read(ExpressCapabilities::Version) as u8;
        if version != 1 && version != 2 {
            return Err(Error::UnsupportedExpressCapabilityVersion(version));
        }
        let raw_type = registers.exp_caps.read(ExpressCapabilities::DeviceType) as u8;
        let device_type = PciExpressDeviceType::from_raw(raw_type)
            .ok_or(Error::UnsupportedExpressDevice(raw_type))?;
        Ok(Self {
            registers,
            version,
            device_type,
        })
    }
}

impl Capability for PciExpress {
    fn length(&self) -> usize {
        if self.version == 2 {
            size_of::<ExpressRegisters>()
        } else if self.device_type.has_root_control() {
            offset_of!(ExpressRegisters, dev_caps2)
        } else if self.device_type.has_link_control() {
            offset_of!(ExpressRegisters, slot_caps)
        } else {
            offset_of!(ExpressRegisters, link_caps)
        }
    }

    fn emulate_read(&self, op: &mut MmioReadBuilder, cap_offset: usize) {
        use express_offsets::*;
        match cap_offset {
            exp_caps::span!() => {
                op.push_word(self.registers.exp_caps.readable_bits());
            }
            dev_caps::span!() => {
                op.push_dword(self.registers.dev_caps.readable_bits());
            }
            dev_control::span!() => {
                op.push_word(self.registers.dev_control.readable_bits());
            }
            link_caps::span!() if self.device_type.has_link_control() => {
                op.push_dword(self.registers.link_caps.readable_bits());
            }
            link_status::span!() if self.device_type.has_link_control() => {
                op.push_word(self.registers.link_status.readable_bits());
            }
            _ => {
                // Make all other capability and status bits appear unimplemented.
                op.push_byte(0);
            }
        }
    }

    fn emulate_write(&mut self, op: &mut MmioWriteBuilder, cap_offset: usize) {
        use express_offsets::*;
        match cap_offset {
            dev_control::span!() => {
                let reg = LocalRegisterCopy::<u16, DeviceControl::Register>::new(
                    op.pop_word(self.registers.dev_control.get()),
                );
                self.registers.dev_control.set(reg.writeable_bits());
            }
            _ => {
                // We don't support writes to any of the othe control registers for now.
                op.pop_byte();
            }
        }
    }
}

// Represents a single PCI capability.
struct PciCapability {
    id: CapabilityId,
    offset: usize,
    next: usize,
    cap_type: CapabilityType,
}

impl PciCapability {
    // Creates a new capability of type `id` at `header`, which itself is at `offset` within the
    // configuration space.
    fn new(
        config_regs: &mut CommonRegisters,
        header: &mut CapabilityHeader,
        id: CapabilityId,
        offset: usize,
    ) -> Result<Self> {
        let cap_type = match id {
            CapabilityId::PowerManagement => PowerManagement::new(header).into(),
            CapabilityId::Msi => Msi::new(header)?.into(),
            CapabilityId::MsiX => MsiX::new(header).into(),
            CapabilityId::Vendor => Vendor::new(header)?.into(),
            CapabilityId::BridgeSubsystem => BridgeSubsystem::new(header).into(),
            CapabilityId::PciExpress => PciExpress::new(header)?.into(),
            CapabilityId::EnhancedAllocation => EnhancedAllocation::new(config_regs, header).into(),
        };
        Ok(PciCapability {
            id,
            offset,
            next: 0,
            cap_type,
        })
    }

    // Returns the ID of this capability.
    fn id(&self) -> CapabilityId {
        self.id
    }

    // Returns the offset of this capability within the configuration space of this device.
    fn offset(&self) -> usize {
        self.offset
    }

    // Sets the offset of the next capability (from the start of the configuration space) in the
    // linked list.
    fn set_next(&mut self, next: usize) {
        self.next = next;
    }

    // Returns the length of this capability structure.
    fn length(&self) -> usize {
        self.cap_type.length()
    }

    // Emulates a read from this capability structure.
    fn emulate_read(&self, op: &mut MmioReadBuilder) {
        let cap_offset = op.offset() - self.offset;
        use header_offsets::*;
        match cap_offset {
            cap_id::span!() => {
                op.push_byte(self.id as u8);
            }
            next_cap::span!() => {
                op.push_byte(self.next as u8);
            }
            _ => {
                self.cap_type.emulate_read(op, cap_offset);
            }
        }
    }

    // Emulates a write to this capability structure.
    fn emulate_write(&mut self, op: &mut MmioWriteBuilder) {
        let cap_offset = op.offset() - self.offset;
        use header_offsets::*;
        match cap_offset {
            cap_id::span!() | next_cap::span!() => {
                op.pop_byte();
            }
            _ => {
                self.cap_type.emulate_write(op, cap_offset);
            }
        }
    }
}

// The maximum number of capabilities we support for a single device. Enough for the typical PCI
// devices virtualized by QEMU.
const MAX_PCI_CAPS: usize = 8;

/// Maps the location of PCI capabilities in a device's config space and handles emulation of
/// reads and writes to these capabilities.
pub struct PciCapabilities {
    caps: ArrayVec<PciCapability, MAX_PCI_CAPS>,
}

impl PciCapabilities {
    /// Creates a new `PciCapabilities` by parsing the PCI capability linked-list starting at
    /// `start_offset` within the standard PCI configuration space pointed to by `config_regs`.
    pub fn new(config_regs: &mut CommonRegisters, start_offset: usize) -> Result<Self> {
        let mut caps = ArrayVec::new();
        let mut current_offset = start_offset;
        while current_offset > PCI_TYPE_HEADER_END && current_offset < PCI_CONFIG_SPACE_END {
            let cap_ptr = (config_regs as *mut CommonRegisters as usize + current_offset)
                as *mut CapabilityHeader;
            // Safety: `cap_ptr` is within the valid and uniquely-owned PCI configuration space
            // referred to by `config_regs` and we are trusting that the hardware has initialized the
            // capability offset registers such that they refer to valid PCI capability headers.
            let header = unsafe { cap_ptr.as_mut().unwrap() };
            let offset = current_offset;
            // Per the spec, the bottom two bits of the next capability pointer should always be
            // discarded.
            //
            // TODO: We should really check that capabilities don't overlap, but this is difficult
            // since they're dynamically sized and the list can be in any order. So for now we trust
            // that the hardware provides a valid config space.
            current_offset = (header.next_cap.get() as usize) & !0x3;
            if let Some(id) = CapabilityId::from_raw(header.cap_id.get()) {
                let cap = PciCapability::new(config_regs, header, id, offset)?;
                caps.try_push(cap).map_err(|_| Error::TooManyCapabilities)?;
            };
        }

        // Now link all the capabilities together to form a linked-list in the virtual config space.
        for i in 1..caps.len() {
            let next_ptr = caps[i].offset();
            caps[i - 1].set_next(next_ptr);
        }

        Ok(Self { caps })
    }

    /// Returns if an MSI capability is present.
    pub fn has_msi(&self) -> bool {
        self.capability_by_id(CapabilityId::Msi).is_some()
    }

    /// Returns if an MSI-X capability is present.
    pub fn has_msix(&self) -> bool {
        self.capability_by_id(CapabilityId::MsiX).is_some()
    }

    /// Returns if a PCI-Express capability is present.
    pub fn is_pcie(&self) -> bool {
        self.capability_by_id(CapabilityId::PciExpress).is_some()
    }

    /// Emulates a read from this device's capabilities structures.
    pub fn emulate_read(&self, op: &mut MmioReadBuilder) {
        if let Some(cap) = self.capability_by_offset(op.offset()) {
            cap.emulate_read(op);
        } else {
            op.push_byte(0);
        }
    }

    /// Emulates a write to this device's capabilities structures.
    pub fn emulate_write(&mut self, op: &mut MmioWriteBuilder) {
        if let Some(cap) = self.capability_by_offset_mut(op.offset()) {
            cap.emulate_write(op);
        } else {
            op.pop_byte();
        }
    }

    /// Returns the offset (in the PCI configuration space) of the first emulated capability.
    pub fn start_offset(&self) -> usize {
        self.caps.first().map(|cap| cap.offset()).unwrap_or(0)
    }

    /// Returns the enhanced allocation capability, if present.
    pub fn enhanced_allocation(&self) -> Option<&EnhancedAllocation> {
        self.capability_by_id(CapabilityId::EnhancedAllocation)
            .and_then(|c| match c.cap_type {
                CapabilityType::EnhancedAllocation(ref cap) => Some(cap),
                _ => None,
            })
    }

    // Returns a reference to the capability at `offset`.
    fn capability_by_offset(&self, offset: usize) -> Option<&PciCapability> {
        self.caps
            .iter()
            .find(|cap| cap.offset() <= offset && offset < (cap.offset() + cap.length()))
    }

    // Returns a mutable reference to the capability at `offset`.
    fn capability_by_offset_mut(&mut self, offset: usize) -> Option<&mut PciCapability> {
        self.caps
            .iter_mut()
            .find(|cap| cap.offset() <= offset && offset < (cap.offset() + cap.length()))
    }

    // Gets the offset of the capability with the given ID.
    fn capability_by_id(&self, id: CapabilityId) -> Option<&PciCapability> {
        self.caps.iter().find(|cap| cap.id() == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn parse_caps() {
        let mut test_config: [u32; 64] = [0; 64];
        test_config[13] = 0x40; // Start of the capability list.
        test_config[16] = 0x0000_4801; // PMC
        test_config[17] = 0xdead_beef;
        test_config[18] = 0x0003_5411; // MSI-X
        test_config[19] = 0x0000_0002;
        test_config[20] = 0x0000_0004;
        test_config[21] = 0xaaaa_5c03; // VPD (don't care)
        test_config[22] = 0xbbbb_cccc;
        test_config[23] = 0x0004_6009; // Vendor
        test_config[24] = 0x0004_0014; // Enhanced Allocation
        test_config[25] = 0x8000_0002; // - entry 0
        test_config[26] = 0x1000_0000; //   - base
        test_config[27] = 0x0000_0ffc; //   - max_offset
        test_config[28] = 0x8000_0003; // - entry 1
        test_config[29] = 0x2000_0002; //   - base
        test_config[30] = 0x0000_fffc; //   - max_offset
        test_config[31] = 0x0000_0002; //   - base high
        test_config[32] = 0x8000_0003; // - entry 2
        test_config[33] = 0x3000_0000; //   - base
        test_config[34] = 0xffff_fffe; //   - max_offset
        test_config[35] = 0x0000_000f; //   - max_offset high
        test_config[36] = 0x8001_1724; // - entry 3
        test_config[37] = 0x4000_0002; //   - base
        test_config[38] = 0xffff_fffe; //   - max_offset
        test_config[39] = 0x0000_0004; //   - base high
        test_config[40] = 0x0000_00ff; //   - max_offset high
        let mut header_mem: Vec<u8> = test_config
            .iter()
            .map(|v| v.to_le_bytes())
            .flatten()
            .collect();
        // Not safe, just a test.
        let regs = unsafe { (header_mem.as_mut_ptr() as *mut CommonRegisters).as_mut() }.unwrap();
        let caps = PciCapabilities::new(regs, 0x40).unwrap();
        assert!(caps.has_msix());
        assert!(!caps.is_pcie());
        assert!(!caps.has_msi());
        assert_eq!(
            caps.capability_by_id(CapabilityId::PowerManagement)
                .unwrap()
                .offset(),
            0x40
        );
        assert_eq!(
            caps.capability_by_id(CapabilityId::Vendor)
                .unwrap()
                .offset(),
            0x5c
        );

        let ea_cap = caps.enhanced_allocation().unwrap();
        assert_eq!(ea_cap.entries.len(), 4);
        assert!(ea_cap.entries[0].enabled());
        assert_eq!(ea_cap.entries[0].bar_equivalent_indicator(), 0);
        assert_eq!(
            ea_cap.entries[0].resource_type(),
            Some(PciResourceType::Mem64)
        );
        assert_eq!(ea_cap.entries[0].base(), 0x1000_0000);
        assert_eq!(ea_cap.entries[0].max_offset(), 0xfff);
        assert_eq!(ea_cap.entries[1].base(), 0x2_2000_0000);
        assert_eq!(ea_cap.entries[1].max_offset(), 0xffff);
        assert_eq!(ea_cap.entries[2].base(), 0x3000_0000);
        assert_eq!(ea_cap.entries[2].max_offset(), 0xf_ffff_ffff);
        assert_eq!(ea_cap.entries[3].bar_equivalent_indicator(), 2);
        assert_eq!(
            ea_cap.entries[3].resource_type(),
            Some(PciResourceType::PrefetchableMem64)
        );
        assert_eq!(ea_cap.entries[3].base(), 0x4_4000_0000);
        assert_eq!(ea_cap.entries[3].max_offset(), 0xff_ffff_ffff);
    }
}
