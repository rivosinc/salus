// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use device_tree::{DeviceTree, DeviceTreeNode, NodeId};
use sync::Mutex;

use super::address::*;
use super::config_space::PciConfigSpace;
use super::device::PciDevice;
use super::error::*;
use super::root::{PciArenaId, PciDeviceArena};

/// A entry for a single device on a `PciBus`.
pub struct BusDevice {
    /// The PCI bus address of the device.
    pub address: Address,
    /// The ID of the device in the `PciDeviceArena`.
    pub id: PciArenaId,
}

/// Represents a PCI bus.
pub struct PciBus {
    bus_range: BusRange,
    virtual_bus_range: BusRange,
    devices: Vec<BusDevice>,
}

fn pci_address_from_dt_node(
    config_space: &PciConfigSpace,
    node: &DeviceTreeNode,
) -> Option<Address> {
    let regs = node.props().find(|p| p.name() == "reg")?;
    let config_offset = regs.value_u32().next()?;
    let (addr, _) = config_space.offset_to_address((config_offset as usize >> 8) << 12)?;
    Some(addr)
}

impl PciBus {
    /// Creates a `PciBus` by enumerating `bus_num` in `config_space`. Devices discovered while
    /// enumerating the bus are addeded to `device_arena`.
    pub fn enumerate(
        dt: &DeviceTree,
        config_space: &PciConfigSpace,
        bus_num: Bus,
        bus_node: Option<NodeId>,
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

                // Locate the corresponding device tree node.
                let dev_node = bus_node.and_then(|id| {
                    dt.get_node(id)?.children().copied().find(|&id| {
                        dt.get_node(id)
                            .filter(|node| !node.disabled())
                            .and_then(|node| pci_address_from_dt_node(config_space, node))
                            .is_some_and(|addr| addr == info.address())
                    })
                });

                // Safety: We trust that PciConfigSpace returned a valid config space pointer for the
                // same device as the one referred to by info.address(). We guarantee that the created
                // device has unique ownership of the register space via the bus enumeration process
                // by creating at most one device per PCI address.
                let pci_dev = unsafe { PciDevice::new(registers_ptr, info.clone(), dev_node) }?;
                let id = device_arena
                    .try_insert(Mutex::new(pci_dev))
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

            // ID must be valid, we just added it above.
            let mut dev = device_arena.get(bridge_id).unwrap().lock();
            let dt_node = dev.dt_node();
            match *dev {
                PciDevice::Bridge(ref mut bridge) => {
                    // Set the bridge to cover everything beyond sec_bus until we've enumerated
                    // the buses behind the bridge.
                    bridge.assign_bus_range(BusRange {
                        start: sec_bus,
                        end: Bus::max(),
                    });
                }
                _ => continue,
            };

            // Unlock `dev` and drop the mutable borrow of `device_arena` for the
            // `PciBus::enumerate()` call.
            drop(dev);

            let child_bus = PciBus::enumerate(dt, config_space, sec_bus, dt_node, device_arena)?;
            let sub_bus = child_bus.subordinate_bus_num();

            // Avoid double mutable borrow of device_arena by re-acquiring the reference to the bridge
            // device here. PciBus::enumerate() may have added devices and re-allocated the arena.
            match *device_arena.get(bridge_id).unwrap().lock() {
                PciDevice::Bridge(ref mut bridge) => {
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
