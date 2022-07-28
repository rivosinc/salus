// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::alloc::Global;
use arrayvec::{ArrayString, ArrayVec};
use core::marker::PhantomData;
use device_tree::{DeviceTree, DeviceTreeResult};
use hyp_alloc::{Arena, ArenaId};
use page_tracking::HwMemMap;
use riscv_pages::*;
use spin::{Mutex, Once};

use crate::Imsic;

use super::address::*;
use super::bus::PciBus;
use super::config_space::PciConfigSpace;
use super::device::*;
use super::error::*;

/// An arena of PCI devices.
pub type PciDeviceArena = Arena<PciDevice, Global>;

/// Identifiers in the PCI device arena.
pub type PciDeviceId = ArenaId<PciDevice>;

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

// The number of "PCI address" cells, as specified in the standard PCI binding.
const PCI_ADDR_CELLS: usize = 3;
// Number of u32 cells per 'ranges' property in the device tree.
const CELLS_PER_RANGE: usize = PCI_ADDR_CELLS + 4;

// Format of the first PCI address cell which specifies the type of the resource.
const PCI_ADDR_PREFETCH_BIT: u32 = 1 << 30;
const PCI_ADDR_SPACE_CODE_SHIFT: u32 = 24;
const PCI_ADDR_SPACE_CODE_MASK: u32 = 0x3;

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

    // Reads a PCI BAR resource type from the first cell in a PCI address range.
    fn from_dt_cell(cell: u32) -> Option<Self> {
        let prefetchable = (cell & PCI_ADDR_PREFETCH_BIT) != 0;
        use PciBarType::*;
        match (cell >> PCI_ADDR_SPACE_CODE_SHIFT) & PCI_ADDR_SPACE_CODE_MASK {
            0x0 => {
                // Config space. Ignore it since we already got it from 'reg'.
                None
            }
            0x1 => Some(IoPort),
            0x2 => {
                if prefetchable {
                    Some(PrefetchableMem32)
                } else {
                    Some(Mem32)
                }
            }
            0x3 => {
                if prefetchable {
                    Some(PrefetchableMem64)
                } else {
                    Some(Mem64)
                }
            }
            _ => unreachable!(),
        }
    }

    // Returns the PCI address cell used to encode this resource type.
    fn to_dt_cell(self) -> u32 {
        use PciBarType::*;
        match self {
            IoPort => 0x1 << PCI_ADDR_SPACE_CODE_SHIFT,
            Mem32 => 0x2 << PCI_ADDR_SPACE_CODE_SHIFT,
            PrefetchableMem32 => (0x2 << PCI_ADDR_SPACE_CODE_SHIFT) | PCI_ADDR_PREFETCH_BIT,
            Mem64 => 0x3 << PCI_ADDR_SPACE_CODE_SHIFT,
            PrefetchableMem64 => (0x3 << PCI_ADDR_SPACE_CODE_SHIFT) | PCI_ADDR_PREFETCH_BIT,
        }
    }
}

/// Describes a PCI BAR space of a particular type.
#[derive(Debug)]
struct PciBarSpace {
    addr: SupervisorPageAddr,
    size: u64,
    taken: bool,
    pci_addr: u64,
}

struct PcieRootInner {
    config_space: PciConfigSpace,
    bar_spaces: ArrayVec<Option<PciBarSpace>, MAX_BAR_SPACES>,
    root_bus: PciBus,
    device_arena: PciDeviceArena,
    msi_parent_phandle: u32,
}

impl PcieRootInner {
    // Calls `f` for each device on `bus`.
    fn for_each_device_on<F: FnMut(&PciDevice)>(&self, bus: &PciBus, f: &mut F) {
        for bd in bus.devices() {
            // If the ID appears on a bus, it must be in the arena.
            let dev = self.device_arena.get(bd.id).unwrap();
            f(dev);
            if let PciDevice::Bridge(bridge) = dev {
                // Recrusively walk the buses behind the bridge.
                bridge
                    .child_bus()
                    .inspect(|b| self.for_each_device_on(b, f));
            };
        }
    }

    // Returns the device ID for the device at the virtualized PCI address `address` on `bus`.
    fn device_by_virtual_address_on(&self, bus: &PciBus, address: Address) -> Option<PciDeviceId> {
        if address.bus() == bus.virtual_secondary_bus_num() {
            // The device is on this bus.
            return bus
                .devices()
                .find(|bd| {
                    bd.address.device() == address.device()
                        && bd.address.function() == address.function()
                })
                .map(|bd| bd.id);
        } else {
            for bd in bus.devices() {
                let dev = self.device_arena.get(bd.id).unwrap();
                if let PciDevice::Bridge(bridge) = dev {
                    // Check if the device is behind the virtualized buses assigned to this bridge.
                    // Unwrap is ok here since buses must have been assigned at this point.
                    let child_bus = bridge.child_bus().unwrap();
                    if child_bus.virtual_secondary_bus_num() <= address.bus()
                        && child_bus.virtual_subordinate_bus_num() >= address.bus()
                    {
                        return self.device_by_virtual_address_on(child_bus, address);
                    }
                }
            }
        }

        None
    }

    // Returns the ID of the device whose config space is mapped at `offset` in the virtualized
    // config space, along with the offset within the device's config space.
    fn virtual_config_offset_to_device(&self, offset: usize) -> Result<(PciDeviceId, usize)> {
        let (address, dev_offset) = self
            .config_space
            .offset_to_address(offset)
            .ok_or(Error::InvalidConfigOffset)?;
        let dev_id = self
            .device_by_virtual_address_on(&self.root_bus, address)
            .ok_or(Error::DeviceNotFound(address))?;
        Ok((dev_id, dev_offset))
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

// Returns if an access of `len` bytes at `offset` is supported in the virtualized config space.
fn valid_config_mmio_access(offset: u64, len: usize) -> bool {
    // We only support naturally-aligned byte, word, or dword accesses.
    matches!(len, 1 | 2 | 4) && (offset & (len as u64 - 1)) == 0
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

        // Make sure the IMSIC is our MSI controller.
        let msi_parent_phandle = pci_node
            .props()
            .find(|p| p.name() == "msi-parent")
            .and_then(|p| p.value_u32().next())
            .ok_or(Error::MissingMsiParent)?;
        if msi_parent_phandle != Imsic::get().phandle() {
            return Err(Error::InvalidMsiParent);
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

        // Find the segment number specified in the device tree, if any.
        let segment = pci_node
            .props()
            .find(|p| p.name() == "linux,pci-domain")
            .and_then(|p| p.value_u32().next())
            .and_then(|v| Segment::try_from(v).ok())
            .unwrap_or_default();
        let config_space = PciConfigSpace::new(config_base, config_size, segment, bus_range);

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
        while ranges.len() >= CELLS_PER_RANGE {
            // Get the resource type from the first cell. We've already guaranteed there are enough
            // cells for this range.
            let resource_type = match PciBarType::from_dt_cell(ranges.next().unwrap()) {
                Some(r) => r,
                None => {
                    ranges.advance_by(CELLS_PER_RANGE - 1).unwrap();
                    continue;
                }
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
                pci_addr,
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
            msi_parent_phandle,
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
    pub fn for_each_device<F: FnMut(&PciDevice)>(&self, mut f: F) {
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

    /// Adds a node for this PCIe root complex to the host's device tree in `dt`. It's assumed that
    /// the config space and BAR resources will be identity-mapped into the host VM's guest physical
    /// address space (i.e. GPA == SPA for the various PCI memory regions). It is up to the caller to
    /// set up those mappings, however.
    pub fn add_host_pcie_node(&self, dt: &mut DeviceTree) -> DeviceTreeResult<()> {
        let inner = self.inner.lock();

        let soc_node_id = dt.iter().find(|n| n.name() == "soc").unwrap().id();
        let mut pci_name = ArrayString::<32>::new();
        let config_mem = inner.config_space.mem_range();
        core::fmt::write(
            &mut pci_name,
            format_args!("pci@{:x}", config_mem.base().bits()),
        )
        .unwrap();
        let pci_id = dt.add_node(pci_name.as_str(), Some(soc_node_id))?;
        let pci_node = dt.get_mut_node(pci_id).unwrap();

        // We forward all properties as-is, or use the values we've assumed while parsing the device
        // tree (e.g. '#address-cells`). This means that the config space and BAR resources are assumed
        // to be identity mapped in the host, though it's up to the caller to actually set up the
        // emulation region and install the mapping. Note that we do not forward 'interrupt-map' and
        // related properties since we aren't doing legacy INTx emulation.
        pci_node
            .add_prop("compatible")?
            .set_value_str("pci-host-ecam-generic")?;
        pci_node
            .add_prop("reg")?
            .set_value_u64(&[config_mem.base().bits(), config_mem.length_bytes()])?;
        let mut ranges = ArrayVec::<u32, { CELLS_PER_RANGE * MAX_BAR_SPACES }>::new();
        for (i, space) in inner
            .bar_spaces
            .iter()
            .enumerate()
            .filter_map(|(i, space)| space.as_ref().map(|s| (i, s)))
        {
            let resource_type = PciBarType::from_index(i).unwrap();
            ranges.push(resource_type.to_dt_cell());
            ranges.push((space.pci_addr >> 32) as u32);
            ranges.push(space.pci_addr as u32);
            ranges.push((space.addr.bits() >> 32) as u32);
            ranges.push(space.addr.bits() as u32);
            ranges.push((space.size >> 32) as u32);
            ranges.push(space.size as u32);
        }
        pci_node.add_prop("ranges")?.set_value_u32(&ranges)?;
        pci_node.add_prop("device_type")?.set_value_str("pci")?;
        pci_node.add_prop("dma-coherent")?;
        pci_node
            .add_prop("linux,pci-domain")?
            .set_value_u32(&[inner.config_space.segment().bits()])?;
        let bus_range = inner.config_space.bus_range();
        pci_node
            .add_prop("bus-range")?
            .set_value_u32(&[bus_range.start.bits(), bus_range.end.bits()])?;
        pci_node
            .add_prop("msi-parent")?
            .set_value_u32(&[inner.msi_parent_phandle])?;
        pci_node.add_prop("#size-cells")?.set_value_u32(&[2])?;
        pci_node.add_prop("#address-cells")?.set_value_u32(&[3])?;

        Ok(())
    }

    /// Emulates a read of `len` bytes at `offset` in the config space.
    pub fn emulate_config_read(&self, offset: u64, len: usize) -> u64 {
        self.do_emulate_config_read(offset, len).unwrap_or(!0x0)
    }

    /// Emulates a write of `len` bytes at `offset` in the config space.
    pub fn emulate_config_write(&self, offset: u64, value: u64, len: usize) {
        // If the write failed, just discard it.
        let _ = self.do_emulate_config_write(offset, value, len);
    }

    fn do_emulate_config_read(&self, offset: u64, len: usize) -> Result<u64> {
        if !valid_config_mmio_access(offset, len) {
            return Err(Error::UnsupportedConfigAccess);
        }
        let inner = self.inner.lock();
        let (dev_id, dev_offset) = inner.virtual_config_offset_to_device(offset as usize)?;
        // If the device ID is present in the hierarchy, then it must be in the arena.
        let dev = inner.device_arena.get(dev_id).unwrap();
        Ok(dev.emulate_config_read(dev_offset, len) as u64)
    }

    fn do_emulate_config_write(&self, offset: u64, value: u64, len: usize) -> Result<()> {
        if !valid_config_mmio_access(offset, len) {
            return Err(Error::UnsupportedConfigAccess);
        }
        let mut inner = self.inner.lock();
        let (dev_id, dev_offset) = inner.virtual_config_offset_to_device(offset as usize)?;
        // If the device ID is present in the hierarchy, then it must be in the arena.
        let dev = inner.device_arena.get_mut(dev_id).unwrap();
        dev.emulate_config_write(dev_offset, value as u32, len);
        Ok(())
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
