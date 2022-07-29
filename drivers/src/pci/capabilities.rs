// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use tock_registers::interfaces::Readable;

use super::error::*;
use super::mmio_builder::{MmioReadBuilder, MmioWriteBuilder};
use super::registers::*;

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

// Represents a single PCI capability.
struct PciCapability {
    id: CapabilityId,
    offset: usize,
    next: usize,
}

impl PciCapability {
    // Creates a new capability of type `id` at `header`, which itself is at `offset` within the
    // configuration space.
    fn new(_header: &mut CapabilityHeader, id: CapabilityId, offset: usize) -> Result<Self> {
        Ok(PciCapability {
            id,
            offset,
            next: 0,
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
        core::mem::size_of::<CapabilityHeader>()
    }

    // Emulates a read from this capability structure.
    fn emulate_read(&self, op: &mut MmioReadBuilder) {
        let cap_offset = op.offset() - self.offset;
        use header_offsets::*;
        match cap_offset {
            cap_id::span!() => {
                // TODO: Hide the capability until we're ready to pass them through.
                op.push_byte(0);
            }
            next_cap::span!() => {
                op.push_byte(self.next as u8);
            }
            _ => {
                // TODO: Emulate type-specific registers.
                op.push_byte(0);
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
                // TODO: Emulate type-specific registers.
                op.pop_byte();
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
                let cap = PciCapability::new(header, id, offset)?;
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
        self.caps.get(0).map(|cap| cap.offset()).unwrap_or(0)
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
        test_config[23] = 0x0004_0009; // Vendor
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
    }
}
