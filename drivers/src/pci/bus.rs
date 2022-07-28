// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;

use super::address::*;
use super::config_space::PciConfigSpace;
use super::device::PciDevice;
use super::error::*;
use super::root::{PciDeviceArena, PciDeviceId};

/// A entry for a single device on a `PciBus`.
pub struct BusDevice {
    /// The PCI bus address of the device.
    pub address: Address,
    /// The ID of the device in the `PciDeviceArena`.
    pub id: PciDeviceId,
}

/// Represents a PCI bus.
pub struct PciBus {
    bus_range: BusRange,
    virtual_bus_range: BusRange,
    devices: Vec<BusDevice>,
}

impl PciBus {
    /// Creates a `PciBus` by enumerating `bus_num` in `config_space`. Devices discovered while
    /// enumerating the bus are addeded to `device_arena`.
    pub fn enumerate(
        config_space: &PciConfigSpace,
        bus_num: Bus,
        device_arena: &mut PciDeviceArena,
    ) -> Result<Self> {
        let bus_config = config_space
            .bus(bus_num)
            .ok_or(Error::OutOfBoundsBusNumber(bus_num))?;
        let mut devices = Vec::new();
        for dev in bus_config.devices() {
            for info in dev.functions() {
                // Unwrap ok, if we have a header the config space for the corresponding function
                // must exist.
                let registers_ptr = config_space.registers_for(info.address()).unwrap();
                // Safety: We trust that PciConfigSpace returned a valid config space pointer for the
                // same device as the one referred to by info.address(). We guarantee that the created
                // device has unique ownership of the register space via the bus enumeration process
                // by creating at most one device per PCI address.
                let pci_dev = unsafe { PciDevice::new(registers_ptr, info.clone()) }?;
                let id = device_arena
                    .try_insert(pci_dev)
                    .map_err(|_| Error::AllocError)?;
                let entry = BusDevice {
                    address: info.address(),
                    id,
                };
                devices.try_reserve(1).map_err(|_| Error::AllocError)?;
                devices.push(entry);
            }
        }

        // Recursively enumerate the buses behind any bridges on this bus.
        let mut cur_bus = bus_num;
        for bd in devices.iter() {
            let bridge_id = bd.id;
            let sec_bus = cur_bus.next().ok_or(Error::OutOfBuses)?;
            match device_arena.get_mut(bridge_id) {
                Some(PciDevice::Bridge(bridge)) => {
                    // Set the bridge to cover everything beyond sec_bus until we've enumerated
                    // the buses behind the bridge.
                    bridge.assign_bus_range(BusRange {
                        start: sec_bus,
                        end: Bus::max(),
                    });
                }
                _ => continue,
            };

            let child_bus = PciBus::enumerate(config_space, sec_bus, device_arena)?;
            let sub_bus = child_bus.subordinate_bus_num();

            // Avoid double mutable borrow of device_arena by re-acquiring the reference to the bridge
            // device here. PciBus::enumerate() may have added devices and re-allocated the arena.
            match device_arena.get_mut(bridge_id) {
                Some(PciDevice::Bridge(bridge)) => {
                    // Now constrain the bus assignment to only the buses we enumerated.
                    bridge.assign_bus_range(BusRange {
                        start: sec_bus,
                        end: sub_bus,
                    });
                    bridge.set_child_bus(child_bus);
                }
                // The device must be a bridge.
                _ => unreachable!(),
            }

            cur_bus = sub_bus;
        }

        Ok(Self {
            bus_range: BusRange {
                start: bus_num,
                end: cur_bus,
            },
            virtual_bus_range: BusRange::default(),
            devices,
        })
    }

    /// Returns an iterator over the device IDs on this bus.
    pub fn devices(&self) -> core::slice::Iter<BusDevice> {
        self.devices.iter()
    }

    /// Returns the subordinate bus number (the highest-numbered downstream bus) for this bus.
    pub fn subordinate_bus_num(&self) -> Bus {
        self.bus_range.end
    }

    /// Sets the virtualized secondary bus number for this bus.
    pub fn set_virtual_secondary_bus_num(&mut self, bus: Bus) {
        self.virtual_bus_range.start = bus;
    }

    /// Returns the virtualized secondary bus number for this bus.
    pub fn virtual_secondary_bus_num(&self) -> Bus {
        self.virtual_bus_range.start
    }

    /// Sets the virtualized subordinate bus number for this bus.
    pub fn set_virtual_subordinate_bus_num(&mut self, bus: Bus) {
        self.virtual_bus_range.end = bus;
    }

    /// Returns the virtualized subordinate bus number for this bus.
    pub fn virtual_subordinate_bus_num(&self) -> Bus {
        self.virtual_bus_range.end
    }
}
