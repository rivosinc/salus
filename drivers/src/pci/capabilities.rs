// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use tock_registers::interfaces::Readable;

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

// Maps a PCI capability ID to its offset in config space.
#[derive(Clone, Copy, Debug)]
struct PciCapability {
    id: CapabilityId,
    offset: usize,
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
    pub fn new(config_regs: &mut CommonRegisters, start_offset: usize) -> Self {
        let mut caps = ArrayVec::new();
        let mut current_offset = start_offset;
        while current_offset > PCI_TYPE_HEADER_END && current_offset < PCI_CONFIG_SPACE_END {
            let cap_ptr = (config_regs as *mut CommonRegisters as usize + current_offset)
                as *mut CapabilityHeader;
            // Safety: `cap_ptr` is within the valid and uniquely-owned PCI configuration space
            // referred to by `config_regs` and we are trusting that the hardware has initialized the
            // capability offset registers such that they refer to valid PCI capability headers.
            let header = unsafe { cap_ptr.as_ref().unwrap() };
            let offset = current_offset;
            // Per the spec, the bottom two bits of the next capability pointer should always be
            // discarded.
            //
            // TODO: We should really check that capabilities don't overlap, but this is difficult
            // since they're dynamically sized and the list can be in any order. So for now we trust
            // that the hardware provides a valid config space.
            current_offset = (header.next_cap.get() as usize) & !0x3;
            if let Some(id) = CapabilityId::from_raw(header.cap_id.get()) {
                let cap = PciCapability { id, offset };
                if caps.try_push(cap).is_err() {
                    break;
                }
            };
        }
        Self { caps }
    }

    /// Returns if an MSI capability is present.
    pub fn has_msi(&self) -> bool {
        self.offset_by_id(CapabilityId::Msi).is_some()
    }

    /// Returns if an MSI-X capability is present.
    pub fn has_msix(&self) -> bool {
        self.offset_by_id(CapabilityId::MsiX).is_some()
    }

    /// Returns if a PCI-Express capability is present.
    pub fn is_pcie(&self) -> bool {
        self.offset_by_id(CapabilityId::PciExpress).is_some()
    }

    // Gets the offset of the capability with the given ID.
    fn offset_by_id(&self, id: CapabilityId) -> Option<usize> {
        self.caps
            .iter()
            .find(|cap| cap.id == id)
            .map(|cap| cap.offset)
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
        let caps = PciCapabilities::new(regs, 0x40);
        assert!(caps.has_msix());
        assert!(!caps.is_pcie());
        assert!(!caps.has_msi());
        assert_eq!(caps.offset_by_id(CapabilityId::PowerManagement), Some(0x40));
        assert_eq!(caps.offset_by_id(CapabilityId::Vendor), Some(0x5c));
    }
}
