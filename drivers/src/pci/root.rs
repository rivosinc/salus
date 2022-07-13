// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::alloc::Allocator;
use device_tree::DeviceTree;
use page_tracking::HwMemMap;
use riscv_pages::{DeviceMemType, PageAddr, PageSize, RawAddr};
use spin::Once;

use super::address::*;
use super::config_space::PciConfigSpace;
use super::header::*;

/// Errors resulting enumerating the PCIe hierarchy.
#[derive(Debug)]
pub enum Error {
    /// The PCI configuration space provided by device tree isn't aligned to 4k.
    ConfigSpaceMisaligned(u64),
    /// The PCI configuration size provided by device tree isn't divisible by 4k.
    ConfigSpaceNotPageMultiple(u64),
    /// The device tree contained an MMIO region that overlaps with other memory regions.
    InvalidMmioRegion(page_tracking::MemMapError),
    /// The device tree entry for the PCI host didn't provide a base register for Configuration
    /// Space.
    NoConfigBase,
    /// No compatible PCI host controller found in the device tree.
    NoCompatibleHostNode,
    /// The device tree entry for the PCI host didn't provide a size register for Configuration
    /// Space.
    NoConfigSize,
    /// The device tree entry for the PCI host didn't provide `reg` property.
    NoRegProperty,
    /// The device tree provided an invalid bus number in the `bus-range` property.
    InvalidBusNumber(u32),
    /// Invalid value in a PCI header at `address`.
    UnknownHeaderType(Address),
}
/// Holds results for PCI probing from device tree.
pub type Result<T> = core::result::Result<T, Error>;

/// Represents a PCI-Express root complex.
pub struct PcieRoot {
    config_space: PciConfigSpace,
}

static PCIE_ROOT: Once<PcieRoot> = Once::new();

impl PcieRoot {
    /// Creates a `PcieRoot` singleton by finding a supported configuration in the passed `DeviceTree`.
    pub fn probe_from<A: Allocator + Clone>(
        dt: &DeviceTree<A>,
        mem_map: &mut HwMemMap,
    ) -> Result<()> {
        let pci_node = dt
            .iter()
            .find(|n| n.compatible(["pci-host-ecam-generic"]) && !n.disabled())
            .ok_or(Error::NoCompatibleHostNode)?;

        // Find the ECAM MMIO region, which should be the first entry in the `reg` property.
        let mut regs = pci_node
            .props()
            .find(|p| p.name() == "reg")
            .ok_or(Error::NoRegProperty)?
            .value_u64();

        let config_addr_raw = regs.next().ok_or(Error::NoConfigBase)?;
        let config_base = PageAddr::new(RawAddr::supervisor(config_addr_raw))
            .ok_or(Error::ConfigSpaceMisaligned(config_addr_raw))?;

        let config_size = regs.next().ok_or(Error::NoConfigSize)?;
        if config_size % (PageSize::Size4k as u64) != 0 {
            return Err(Error::ConfigSpaceNotPageMultiple(config_size));
        }

        unsafe {
            // Safety: Have to trust that the device tree points to valid PCI space.
            // Any overlaps will be caught by `add_mmio_region` and the error will be propagated.
            mem_map
                .add_mmio_region(
                    DeviceMemType::PciRoot,
                    RawAddr::from(config_base),
                    config_size,
                )
                .map_err(Error::InvalidMmioRegion)?;
        }

        // Find the bus range this root complex covers.
        let bus_range = {
            match pci_node.props().find(|p| p.name() == "bus-range") {
                Some(p) => {
                    let mut iter = p.value_u32();
                    let start_bus_index = iter.next().unwrap_or(0);
                    let start_bus = start_bus_index
                        .try_into()
                        .map_err(|_| Error::InvalidBusNumber(start_bus_index))?;
                    let end_bus_index = iter.next().unwrap_or(255);
                    let end_bus = end_bus_index
                        .try_into()
                        .map_err(|_| Error::InvalidBusNumber(end_bus_index))?;
                    BusRange {
                        start: start_bus,
                        end: end_bus,
                    }
                }
                None => BusRange {
                    start: Bus::try_from(0u8).unwrap(),
                    end: Bus::try_from(255u8).unwrap(),
                },
            }
        };
        // TODO: Segment assignment in the case of multiple PCIe domains.
        let config_space =
            PciConfigSpace::new(config_base, config_size, Segment::default(), bus_range);

        PCIE_ROOT.call_once(|| Self { config_space });
        Ok(())
    }

    /// Gets a reference to the `PcieRoot` singleton. Panics if `PcieRoot::probe_from()` has not yet
    /// been called to initialize it.
    pub fn get() -> &'static Self {
        PCIE_ROOT.get().unwrap()
    }

    /// Walks the PCIe hierarchy, calling `f` on each device function.
    pub fn for_each_device<F: FnMut(&Header)>(&self, mut f: F) -> Result<()> {
        for bus in self.config_space.busses() {
            for dev in bus.devices() {
                for header in dev.functions() {
                    match header.header_type() {
                        Some(HeaderType::Endpoint) => f(&header),
                        // TODO: Assign and enumerate child busses.
                        Some(HeaderType::PciBridge) => f(&header),
                        Some(HeaderType::CardBusBridge) => (),
                        None => return Err(Error::UnknownHeaderType(header.address())),
                    }
                }
            }
        }
        Ok(())
    }
}
