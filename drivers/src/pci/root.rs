// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::alloc::Global;
use arrayvec::{ArrayString, ArrayVec};
use core::marker::PhantomData;
use device_tree::{DeviceTree, DeviceTreeResult};
use hyp_alloc::{Arena, ArenaId};
use page_tracking::{HwMemMap, PageTracker};
use riscv_pages::*;
use spin::{Mutex, Once};

use crate::Imsic;

use super::address::*;
use super::bus::PciBus;
use super::config_space::PciConfigSpace;
use super::device::*;
use super::error::*;
use super::mmio_builder::MmioEmulationContext;
use super::resource::*;

/// An arena of PCI devices.
pub type PciDeviceArena = Arena<Mutex<PciDevice>, Global>;

/// Identifiers in the PCI device arena.
pub type PciArenaId = ArenaId<Mutex<PciDevice>>;

// The number of "PCI address" cells, as specified in the standard PCI binding.
const PCI_ADDR_CELLS: usize = 3;
// Number of u32 cells per 'ranges' property in the device tree.
const CELLS_PER_RANGE: usize = PCI_ADDR_CELLS + 4;

/// Represents a PCI-Express root complex.
pub struct PcieRoot {
    config_space: PciConfigSpace,
    root_bus: PciBus,
    device_arena: PciDeviceArena,
    resources: Mutex<PciRootResources>,
    msi_parent_phandle: u32,
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
        let mut resources = PciRootResources::new();
        while ranges.len() >= CELLS_PER_RANGE {
            // Get the resource type from the first cell. We've already guaranteed there are enough
            // cells for this range.
            let resource_type = match PciResourceType::from_dt_cell(ranges.next().unwrap()) {
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
                .ok_or(Error::ResourceMisaligned(cpu_addr))?;
            let size = U64Cell(ranges.next().unwrap(), ranges.next().unwrap()).into();
            if size % (PageSize::Size4k as u64) != 0 {
                return Err(Error::ResourceNotPageMultiple(size));
            }

            let resource = PciRootResource::new(addr, size, pci_addr);
            resources.insert(resource_type, resource)?;

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

        PCIE_ROOT.call_once(|| Self {
            config_space,
            root_bus,
            device_arena,
            resources: Mutex::new(resources),
            msi_parent_phandle,
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
        // Silence bogus auto-deref lint, see https://github.com/rust-lang/rust-clippy/issues/9101.
        #[allow(clippy::explicit_auto_deref)]
        self.for_each_device_on(&self.root_bus, &mut |id| {
            let dev = self.device_arena.get(id).unwrap();
            f(&*dev.lock());
        });
    }

    /// Returns the memory range occupied by this root complex's config space.
    pub fn config_space(&self) -> SupervisorPageRange {
        self.config_space.mem_range()
    }

    /// Returns an iterator over the root's PCI resources.
    pub fn resources(&self) -> PciResourceIter {
        PciResourceIter::new(self)
    }

    /// Takes ownership of the PCI BAR space identified by `resource_type`, returning an iterator over
    /// the pages occupied by that resource.
    pub fn take_resource(&self, resource_type: PciResourceType) -> Option<PciBarPageIter> {
        let mut resources = self.resources.lock();
        let res = resources.get_mut(resource_type)?;
        res.take().ok()?;
        let mem_range = SupervisorPageRange::new(res.addr(), PageSize::num_4k_pages(res.size()));
        // Safety: We have unique ownership of the memory since we've just taken it from this PcieRoot.
        let iter = unsafe { PciBarPageIter::new(mem_range) };
        Some(iter)
    }

    /// Takes ownership over all unowned devices in the PCI hierarchy on behalf of the host VM.
    pub fn take_host_devices(&self) {
        self.for_each_device_on(&self.root_bus, &mut |id| {
            let dev = self.device_arena.get(id).unwrap();
            let _ = dev.lock().take(PageOwnerId::host());
        });
    }

    /// Adds a node for this PCIe root complex to the host's device tree in `dt`. It's assumed that
    /// the config space and BAR resources will be identity-mapped into the host VM's guest physical
    /// address space (i.e. GPA == SPA for the various PCI memory regions). It is up to the caller to
    /// set up those mappings, however.
    pub fn add_host_pcie_node(&self, dt: &mut DeviceTree) -> DeviceTreeResult<()> {
        let soc_node_id = dt.iter().find(|n| n.name() == "soc").unwrap().id();
        let mut pci_name = ArrayString::<32>::new();
        let config_mem = self.config_space.mem_range();
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
        let mut ranges = ArrayVec::<u32, { CELLS_PER_RANGE * MAX_RESOURCE_TYPES }>::new();
        let resources = self.resources.lock();
        for i in 0..MAX_RESOURCE_TYPES {
            let res_type = PciResourceType::from_index(i).unwrap();
            if let Some(res) = resources.get(res_type) {
                ranges.push(res_type.to_dt_cell());
                ranges.push((res.pci_addr() >> 32) as u32);
                ranges.push(res.pci_addr() as u32);
                ranges.push((res.addr().bits() >> 32) as u32);
                ranges.push(res.addr().bits() as u32);
                ranges.push((res.size() >> 32) as u32);
                ranges.push(res.size() as u32);
            }
        }
        pci_node.add_prop("ranges")?.set_value_u32(&ranges)?;
        pci_node.add_prop("device_type")?.set_value_str("pci")?;
        pci_node.add_prop("dma-coherent")?;
        pci_node
            .add_prop("linux,pci-domain")?
            .set_value_u32(&[self.config_space.segment().bits()])?;
        let bus_range = self.config_space.bus_range();
        pci_node
            .add_prop("bus-range")?
            .set_value_u32(&[bus_range.start.bits(), bus_range.end.bits()])?;
        pci_node
            .add_prop("msi-parent")?
            .set_value_u32(&[self.msi_parent_phandle])?;
        pci_node.add_prop("#size-cells")?.set_value_u32(&[2])?;
        pci_node.add_prop("#address-cells")?.set_value_u32(&[3])?;

        Ok(())
    }

    /// Emulates a read of `len` bytes at `offset` in the config space by the VM with `guest_id`.
    pub fn emulate_config_read(
        &self,
        offset: u64,
        len: usize,
        page_tracker: PageTracker,
        guest_id: PageOwnerId,
    ) -> u64 {
        self.do_emulate_config_read(offset, len, page_tracker, guest_id)
            .unwrap_or(!0x0)
    }

    /// Emulates a write of `len` bytes at `offset` in the config space by the VM with `guest_id`.
    pub fn emulate_config_write(
        &self,
        offset: u64,
        value: u64,
        len: usize,
        page_tracker: PageTracker,
        guest_id: PageOwnerId,
    ) {
        // If the write failed, just discard it.
        let _ = self.do_emulate_config_write(offset, value, len, page_tracker, guest_id);
    }

    fn do_emulate_config_read(
        &self,
        offset: u64,
        len: usize,
        page_tracker: PageTracker,
        guest_id: PageOwnerId,
    ) -> Result<u64> {
        if !valid_config_mmio_access(offset, len) {
            return Err(Error::UnsupportedConfigAccess);
        }
        let (dev_id, dev_offset) = self.virtual_config_offset_to_device(offset as usize)?;
        // If the device ID is present in the hierarchy, then it must be in the arena.
        let dev = self.device_arena.get(dev_id).unwrap().lock();
        if dev.owner() != Some(guest_id) {
            return Err(Error::DeviceNotOwned);
        }
        let resources = self.resources.lock();
        let context = MmioEmulationContext {
            page_tracker,
            guest_id,
            resources: &resources,
        };
        Ok(dev.emulate_config_read(dev_offset, len, context) as u64)
    }

    fn do_emulate_config_write(
        &self,
        offset: u64,
        value: u64,
        len: usize,
        page_tracker: PageTracker,
        guest_id: PageOwnerId,
    ) -> Result<()> {
        if !valid_config_mmio_access(offset, len) {
            return Err(Error::UnsupportedConfigAccess);
        }
        let (dev_id, dev_offset) = self.virtual_config_offset_to_device(offset as usize)?;
        // If the device ID is present in the hierarchy, then it must be in the arena.
        let mut dev = self.device_arena.get(dev_id).unwrap().lock();
        if dev.owner() != Some(guest_id) {
            return Err(Error::DeviceNotOwned);
        }
        let resources = self.resources.lock();
        let context = MmioEmulationContext {
            page_tracker,
            guest_id,
            resources: &resources,
        };
        dev.emulate_config_write(dev_offset, value as u32, len, context);
        Ok(())
    }

    // Calls `f` for each device on `bus`.
    fn for_each_device_on<F: FnMut(PciArenaId)>(&self, bus: &PciBus, f: &mut F) {
        for bd in bus.devices() {
            f(bd.id);
            // If the ID appears on a bus, it must be in the arena.
            let dev = self.device_arena.get(bd.id).unwrap();
            if let PciDevice::Bridge(ref bridge) = *dev.lock() {
                // Recursively walk the buses behind the bridge.
                bridge
                    .child_bus()
                    .inspect(|b| self.for_each_device_on(b, f));
            };
        }
    }

    // Returns the device ID for the device at the virtualized PCI address `address` on `bus`.
    fn device_by_virtual_address_on(&self, bus: &PciBus, address: Address) -> Option<PciArenaId> {
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
                if let PciDevice::Bridge(ref bridge) = *dev.lock() {
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
    fn virtual_config_offset_to_device(&self, offset: usize) -> Result<(PciArenaId, usize)> {
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

/// An iterator over the types of BAR resources for a PCI root complex.
pub struct PciResourceIter<'a> {
    root: &'a PcieRoot,
    index: usize,
}

impl<'a> PciResourceIter<'a> {
    /// Creates a new iterator over `root`'s BAR resources.
    fn new(root: &'a PcieRoot) -> Self {
        Self { root, index: 0 }
    }
}

impl<'a> Iterator for PciResourceIter<'a> {
    type Item = (PciResourceType, SupervisorPageRange);

    fn next(&mut self) -> Option<Self::Item> {
        let resources = self.root.resources.lock();
        while let Some(mem_type) = PciResourceType::from_index(self.index) {
            self.index += 1;
            if let Some(res) = resources.get(mem_type) {
                let mem_range =
                    SupervisorPageRange::new(res.addr(), PageSize::num_4k_pages(res.size()));
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
