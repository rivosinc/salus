// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use device_tree::DeviceTree;
use page_tracking::HwMemMap;
use riscv_pages::{DeviceMemType, PageAddr, PageSize, RawAddr, SupervisorPageAddr};
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
    /// A PCI BAR resource provided by the device tree isn't aligned to 4k.
    BarSpaceMisaligned(u64),
    /// A PCI BAR resource provided by the device tree isn't divisible by 4k.
    BarSpaceNotPageMultiple(u64),
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
    /// The device tree entry for the PCI host didn't provide a `reg` property.
    NoRegProperty,
    /// The device tree entry for the PCI host didn't provide a `ranges` property.
    NoRangesProperty,
    /// Too many PCI resource ranges were specified in the device tree's `ranges` property.
    TooManyBarSpaces,
    /// The device tree provided an invalid bus number in the `bus-range` property.
    InvalidBusNumber(u32),
    /// Invalid value in a PCI header at `address`.
    UnknownHeaderType(Address),
}
/// Holds results for PCI probing from device tree.
pub type Result<T> = core::result::Result<T, Error>;

// The maximum number of BAR resources we support at the root complex.
const MAX_BAR_SPACES: usize = 4;

/// PCI BAR resource types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciBarType {
    /// IO space.
    Io,
    /// 32-bit non-prefetchable memory space.
    Mem32,
    /// 32-bit prefetchable memory space.
    PrefetchableMem32,
    /// 64-bit non-prefetchable memory space. 64-bit memory spaces are supposed to be prefetchable,
    /// but many device trees (including from QEMU) don't set the prefetch bit.
    Mem64,
    /// 64-bit prefetchable memory space.
    PrefetchableMem64,
}

/// Describes a PCI BAR space of a particular type.
#[derive(Debug)]
pub struct PciBarSpace {
    resource_type: PciBarType,
    addr: SupervisorPageAddr,
    size: u64,
    pci_addr: u64,
}

impl PciBarSpace {
    /// Returns the type of resource this BAR space maps.
    pub fn resource_type(&self) -> PciBarType {
        self.resource_type
    }

    /// Returns the CPU physical address of this BAR space.
    pub fn addr(&self) -> SupervisorPageAddr {
        self.addr
    }

    /// Returns the size of this BAR space.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the PCI bus address of this BAR space.
    pub fn pci_addr(&self) -> u64 {
        self.pci_addr
    }
}

/// Represents a PCI-Express root complex.
pub struct PcieRoot {
    config_space: PciConfigSpace,
    bar_spaces: ArrayVec<PciBarSpace, MAX_BAR_SPACES>,
}

static PCIE_ROOT: Once<PcieRoot> = Once::new();

// A `u64` from two `u32` cells in a device tree.
struct U64Cell(u32, u32);

impl From<U64Cell> for u64 {
    fn from(u: U64Cell) -> u64 {
        ((u.0 as u64) << 32) | (u.1 as u64)
    }
}

impl PcieRoot {
    /// Creates a `PcieRoot` singleton by finding a supported configuration in the passed `DeviceTree`.
    pub fn probe_from(dt: &DeviceTree, mem_map: &mut HwMemMap) -> Result<()> {
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
                    DeviceMemType::PciConfig,
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

        // Parse the 'ranges' property for the various BAR resources. Assuming '#address-cells' is 3
        // and '#size-cells' is 2, each range is 7x u32 cells:
        //
        // cells[0] describes the type of resource.
        // cells[1:2] are the 64-bit PCI address of the resource.
        // cells[3:4] are the 64-bit CPU address of the resource.
        // cells[5:6] are the 64-bit size of the resource.
        //
        // See IEEE 1275-1994 for more details.
        //
        // TODO: Assuming fixed '#address-cells'/'#size-cells'.
        let ranges_prop = pci_node
            .props()
            .find(|p| p.name() == "ranges")
            .ok_or(Error::NoRangesProperty)?;
        let mut ranges = ranges_prop.value_u32();
        let mut bar_spaces = ArrayVec::new();
        const PCI_ADDR_CELLS: usize = 3;
        const CELLS_PER_RANGE: usize = PCI_ADDR_CELLS + 4;
        while ranges.len() >= CELLS_PER_RANGE {
            // We've already guaranteed there are enough cells for this range.
            let phys_hi = ranges.next().unwrap();

            // Determine the resource type.
            const PREFETCH_BIT: u32 = 1 << 30;
            let prefetchable = (phys_hi & PREFETCH_BIT) != 0;
            const SPACE_CODE_SHIFT: u32 = 24;
            const SPACE_CODE_MASK: u32 = 0x3;
            let resource_type = match (phys_hi >> SPACE_CODE_SHIFT) & SPACE_CODE_MASK {
                0x0 => {
                    // Config space. Ignore it since we already got it from 'reg'.
                    continue;
                }
                0x1 => PciBarType::Io,
                0x2 => {
                    if prefetchable {
                        PciBarType::PrefetchableMem32
                    } else {
                        PciBarType::Mem32
                    }
                }
                0x3 => {
                    if prefetchable {
                        PciBarType::PrefetchableMem64
                    } else {
                        PciBarType::Mem64
                    }
                }
                _ => unreachable!(),
            };

            // Next is the PCI address, followed by the CPU address and region size.
            let pci_addr = U64Cell(ranges.next().unwrap(), ranges.next().unwrap()).into();
            let cpu_addr = U64Cell(ranges.next().unwrap(), ranges.next().unwrap()).into();
            let addr = PageAddr::new(RawAddr::supervisor(cpu_addr))
                .ok_or(Error::BarSpaceMisaligned(cpu_addr))?;
            let size = U64Cell(ranges.next().unwrap(), ranges.next().unwrap()).into();
            if size % (PageSize::Size4k as u64) != 0 {
                return Err(Error::BarSpaceNotPageMultiple(size));
            }

            let bar_space = PciBarSpace {
                resource_type,
                addr,
                size,
                pci_addr,
            };
            bar_spaces
                .try_push(bar_space)
                .map_err(|_| Error::TooManyBarSpaces)?;

            unsafe {
                // Safety: Have to trust that the device tree points to valid PCI space.
                // Any overlaps will be caught by `add_mmio_region` and the error will be propagated.
                mem_map
                    .add_mmio_region(DeviceMemType::PciBar, RawAddr::from(addr), size)
                    .map_err(Error::InvalidMmioRegion)?;
            }
        }

        PCIE_ROOT.call_once(|| Self {
            config_space,
            bar_spaces,
        });
        Ok(())
    }

    /// Gets a reference to the `PcieRoot` singleton. Panics if `PcieRoot::probe_from()` has not yet
    /// been called to initialize it.
    pub fn get() -> &'static Self {
        PCIE_ROOT.get().unwrap()
    }

    /// Walks the PCIe hierarchy, calling `f` on each device function.
    pub fn for_each_device<F: FnMut(&Header)>(&self, mut f: F) -> Result<()> {
        let bus = self.config_space.root_bus();
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
        Ok(())
    }

    /// Returns an iterator over the root's PCI BAR spaces.
    pub fn bar_spaces(&self) -> core::slice::Iter<PciBarSpace> {
        self.bar_spaces.iter()
    }
}
