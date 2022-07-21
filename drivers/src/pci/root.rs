// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::alloc::Global;
use arrayvec::ArrayVec;
use core::marker::PhantomData;
use device_tree::DeviceTree;
use hyp_alloc::{Arena, ArenaId};
use page_tracking::HwMemMap;
use riscv_pages::*;
use spin::{Mutex, Once};

use super::address::*;
use super::bus::PciBus;
use super::config_space::PciConfigSpace;
use super::device::*;
use super::error::*;

/// An arena of PCI devices.
pub type PciDeviceArena = Arena<PciDeviceType, Global>;

/// Identifiers in the PCI device arena.
pub type PciDeviceId = ArenaId<PciDeviceType>;

/// PCI BAR resource types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciBarType {
    /// IO port space.
    IoPort = 0,
    /// 32-bit non-prefetchable memory space.
    Mem32 = 1,
    /// 32-bit prefetchable memory space.
    PrefetchableMem32 = 2,
    /// 64-bit non-prefetchable memory space. 64-bit memory spaces are supposed to be prefetchable,
    /// but many device trees (including from QEMU) don't set the prefetch bit.
    Mem64 = 3,
    /// 64-bit prefetchable memory space.
    PrefetchableMem64 = 4,
}

const MAX_BAR_SPACES: usize = PciBarType::PrefetchableMem64 as usize + 1;

impl PciBarType {
    fn from_index(index: usize) -> Option<Self> {
        use PciBarType::*;
        match index {
            0 => Some(IoPort),
            1 => Some(Mem32),
            2 => Some(PrefetchableMem32),
            3 => Some(Mem64),
            4 => Some(PrefetchableMem64),
            _ => None,
        }
    }
}

/// Describes a PCI BAR space of a particular type.
#[derive(Debug)]
struct PciBarSpace {
    addr: SupervisorPageAddr,
    size: u64,
    taken: bool,
    _pci_addr: u64,
}

struct PcieRootInner {
    config_space: PciConfigSpace,
    bar_spaces: ArrayVec<Option<PciBarSpace>, MAX_BAR_SPACES>,
    root_bus: PciBus,
    device_arena: PciDeviceArena,
}

impl PcieRootInner {
    /// Calls `f` for each device on `bus`.
    fn for_each_device_on<F: FnMut(&dyn PciDevice)>(&self, bus: &PciBus, f: &mut F) {
        for id in bus.devices() {
            // If the ID appears on a bus, it must be in the arena.
            let dev = self.device_arena.get(id).unwrap();
            match dev {
                PciDeviceType::Endpoint(ep) => f(ep),
                PciDeviceType::Bridge(bridge) => {
                    f(bridge);
                    // Recrusively walk the buses behind the bridge.
                    bridge
                        .child_bus()
                        .inspect(|b| self.for_each_device_on(b, f));
                }
            };
        }
    }
}

/// Represents a PCI-Express root complex.
pub struct PcieRoot {
    inner: Mutex<PcieRootInner>,
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
        for _ in 0..bar_spaces.capacity() {
            bar_spaces.push(None);
        }
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
                0x1 => PciBarType::IoPort,
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
                addr,
                size,
                taken: false,
                _pci_addr: pci_addr,
            };
            if bar_spaces[resource_type as usize].is_some() {
                return Err(Error::DuplicateBarSpace(resource_type));
            }
            bar_spaces[resource_type as usize] = Some(bar_space);

            unsafe {
                // Safety: Have to trust that the device tree points to valid PCI space.
                // Any overlaps will be caught by `add_mmio_region` and the error will be propagated.
                mem_map
                    .add_mmio_region(DeviceMemType::PciBar, RawAddr::from(addr), size)
                    .map_err(Error::InvalidMmioRegion)?;
            }
        }

        // Enumerate the PCI hierarchy.
        let mut device_arena = PciDeviceArena::new(Global);
        let root_bus = PciBus::enumerate(&config_space, bus_range.start, &mut device_arena)?;

        let inner = PcieRootInner {
            config_space,
            bar_spaces,
            root_bus,
            device_arena,
        };
        PCIE_ROOT.call_once(|| Self {
            inner: Mutex::new(inner),
        });
        Ok(())
    }

    /// Gets a reference to the `PcieRoot` singleton. Panics if `PcieRoot::probe_from()` has not yet
    /// been called to initialize it.
    pub fn get() -> &'static Self {
        PCIE_ROOT.get().unwrap()
    }

    /// Walks the PCIe hierarchy, calling `f` on each device function.
    pub fn for_each_device<F: FnMut(&dyn PciDevice)>(&self, mut f: F) {
        let inner = self.inner.lock();
        inner.for_each_device_on(&inner.root_bus, &mut f)
    }

    /// Returns the memory range occupied by this root complex's config space.
    pub fn config_space(&self) -> SupervisorPageRange {
        self.inner.lock().config_space.mem_range()
    }

    /// Returns an iterator over the root's PCI BAR spaces.
    pub fn bar_spaces(&self) -> PciBarSpaceIter {
        PciBarSpaceIter::new(self)
    }

    /// Takes ownership of the PCI BAR space identified by `resource_type`, returning an iterator over
    /// the pages occupied by that resource.
    pub fn take_bar_space(&self, resource_type: PciBarType) -> Option<PciBarPageIter> {
        let mut inner = self.inner.lock();
        let space = inner.bar_spaces[resource_type as usize].as_mut()?;
        if space.taken {
            return None;
        }
        space.taken = true;
        let mem_range = SupervisorPageRange::new(space.addr, PageSize::num_4k_pages(space.size));
        // Safety: We have unique ownership of the memory since we've just taken it from this PcieRoot.
        let iter = unsafe { PciBarPageIter::new(mem_range) };
        Some(iter)
    }
}

/// An iterator over the types of BAR resources for a PCI root complex.
pub struct PciBarSpaceIter<'a> {
    root: &'a PcieRoot,
    index: usize,
}

impl<'a> PciBarSpaceIter<'a> {
    /// Creates a new iterator over `root`'s BAR resources.
    fn new(root: &'a PcieRoot) -> Self {
        Self { root, index: 0 }
    }
}

impl<'a> Iterator for PciBarSpaceIter<'a> {
    type Item = (PciBarType, SupervisorPageRange);

    fn next(&mut self) -> Option<Self::Item> {
        let inner = self.root.inner.lock();
        while let Some(mem_type) = PciBarType::from_index(self.index) {
            let i = self.index;
            self.index += 1;
            if let Some(ref space) = inner.bar_spaces[i] {
                let mem_range =
                    SupervisorPageRange::new(space.addr, PageSize::num_4k_pages(space.size));
                return Some((mem_type, mem_range));
            }
        }
        None
    }
}

/// A `PhysPage` implementation representing a page of a PCI BAR resource.
pub struct PciBarPage<S: State> {
    addr: SupervisorPageAddr,
    state: PhantomData<S>,
}

impl<S: State> PhysPage for PciBarPage<S> {
    /// Creates a new `PciBarPage` at the given page-aligned address. Pages are always 4kB.
    ///
    /// # Safety
    ///
    /// The caller must ensure `addr` refers to a uniquely-owned page of a PCI BAR resource.
    unsafe fn new_with_size(addr: SupervisorPageAddr, size: PageSize) -> Self {
        assert_eq!(size, PageSize::Size4k);
        Self {
            addr,
            state: PhantomData,
        }
    }

    fn addr(&self) -> SupervisorPageAddr {
        self.addr
    }

    fn size(&self) -> PageSize {
        PageSize::Size4k
    }

    fn mem_type() -> MemType {
        MemType::Mmio(DeviceMemType::PciBar)
    }
}

// We assume PCI BAR pages are always clean, though in reality they may maintain state depending on
// the device they map. How that state is wiped is device-dependent however, and we'll need something
// like the proposed TDISP extension in order to support authenticating and measuring the state of a
// device. For now, devices are only directly assigned to the host VM.
impl MappablePhysPage<MeasureOptional> for PciBarPage<MappableClean> {}
impl AssignablePhysPage<MeasureOptional> for PciBarPage<ConvertedClean> {
    type MappablePage = PciBarPage<MappableClean>;
}
impl ConvertedPhysPage for PciBarPage<ConvertedClean> {
    type DirtyPage = PciBarPage<ConvertedClean>;
}
impl InvalidatedPhysPage for PciBarPage<Invalidated> {}
impl ReclaimablePhysPage for PciBarPage<ConvertedClean> {
    type MappablePage = PciBarPage<MappableClean>;
}

/// An iterator yielding owning references to PCI BAR resources as `PciBarPage`s.
pub struct PciBarPageIter {
    mem_range: SupervisorPageRange,
}

impl PciBarPageIter {
    /// Creates a new `PciBarPageIter` over `mem_range`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `mem_range` is occupied by a valid PCI BAR resource and that
    /// the memory in that range is uniquely owned.
    unsafe fn new(mem_range: SupervisorPageRange) -> Self {
        Self { mem_range }
    }
}

impl Iterator for PciBarPageIter {
    type Item = PciBarPage<ConvertedClean>;

    fn next(&mut self) -> Option<Self::Item> {
        let addr = self.mem_range.next()?;
        // Safety: The address is guaranteed to be valid and the page it refers to is guaranteed to
        // be uniquely owned at construction of the PciBarPageIter.
        let page = unsafe { PciBarPage::new(addr) };
        Some(page)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.mem_range.size_hint()
    }
}

impl ExactSizeIterator for PciBarPageIter {}
