// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;

use super::address::*;
use super::config_space::PciConfigSpace;
use super::device::PciDeviceType;
use super::error::*;
use super::root::{PciDeviceArena, PciDeviceId};

struct BusDevice(Address, PciDeviceId);

/// Represents a PCI bus.
pub struct PciBus {
    _bus_range: BusRange,
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
            for header in dev.functions() {
                // Unwrap ok, if we have a header the config space for the corresponding function
                // must exist.
                let func_config = config_space.config_space_for(header.address()).unwrap();
                let pci_dev = PciDeviceType::new(func_config, header.clone())?;
                let id = device_arena
                    .try_insert(pci_dev)
                    .map_err(|_| Error::AllocError)?;
                let entry = BusDevice(header.address(), id);
                devices.try_reserve(1).map_err(|_| Error::AllocError)?;
                devices.push(entry);
            }
        }

        // TODO: Recursively enumerate any bridges on this bus.

        Ok(Self {
            _bus_range: BusRange {
                start: bus_num,
                end: bus_num,
            },
            devices,
        })
    }

    /// Returns an iterator over the device IDs on this bus.
    pub fn devices(&self) -> impl ExactSizeIterator<Item = PciDeviceId> + '_ {
        self.devices.iter().map(|bd| bd.1)
    }
}
