// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use arrayvec::{ArrayString, ArrayVec};
use core::{fmt, marker::PhantomData, ops::Deref};
use device_tree::{DeviceTree, DeviceTreeResult};
use page_tracking::HwMemMap;
use riscv_pages::*;
use riscv_regs::{
    hstatus, sie, stopei, ReadWriteable, Readable, RiscvCsrInterface, Writeable, CSR,
};
use sync::{Mutex, Once};

use super::error::{Error, Result};
use super::geometry::*;
use super::sw_file::SwFile;
use crate::{CpuId, CpuInfo, MAX_CPUS};

/// The maximum number of IMSIC interrupt IDs, as per the AIA specification.
pub const MAX_INTERRUPT_IDS: usize = 2048;

const MAX_GUEST_FILES: usize = 7;
const MAX_MMIO_REGIONS: usize = 8;

/// IMSIC external interrupt IDs.
/// For now, we only expect to handle IPIs at HS-level.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImsicInterruptId {
    /// Interrupt ID for inter-processer notifications.
    Ipi = 1,
}

impl ImsicInterruptId {
    /// Returns the interrupt corresponding to `id`.
    fn from_raw(id: u64) -> Option<Self> {
        match id {
            1 => Some(ImsicInterruptId::Ipi),
            _ => None,
        }
    }

    /// Returns the indirect EIE register number and bit position for this interrupt.
    fn offset_and_bit(&self) -> (usize, usize) {
        (*self as usize / 64, *self as usize % 64)
    }
}

/// A `PhysPage` implementation representing an IMSIC guest interrupt file page.
pub struct ImsicGuestPage<S: State> {
    addr: SupervisorPageAddr,
    location: ImsicLocation,
    state: PhantomData<S>,
}

impl<S: State> ImsicGuestPage<S> {
    /// Returns the location of the IMSIC file that this page corresponds to in the physical IMSIC
    /// geometry.
    pub fn location(&self) -> ImsicLocation {
        self.location
    }
}

impl<S: State> PhysPage for ImsicGuestPage<S> {
    /// Creates a new `ImsicGuestPage` at the given page-aligned address. IMSIC pages are always 4kB.
    ///
    /// # Safety
    ///
    /// The caller must ensure `addr` refers to a uniquely owned IMSIC guest interrupt file.
    unsafe fn new_with_size(addr: SupervisorPageAddr, size: PageSize) -> Self {
        assert_eq!(size, PageSize::Size4k);
        // This page must refer to an IMSIC page, so it must have a valid location in the physical
        // IMSIC geometry.
        let location = Imsic::get().phys_geometry().addr_to_location(addr).unwrap();
        Self {
            addr,
            location,
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
        MemType::Mmio(DeviceMemType::Imsic)
    }
}

// IMSIC interrupt file pages retain no state so they are always considered "clean".
impl MappablePhysPage<MeasureOptional> for ImsicGuestPage<MappableClean> {}
impl AssignablePhysPage<MeasureOptional> for ImsicGuestPage<ConvertedClean> {
    type MappablePage = ImsicGuestPage<MappableClean>;
}
impl ConvertedPhysPage for ImsicGuestPage<ConvertedClean> {
    type DirtyPage = ImsicGuestPage<ConvertedClean>;
}
impl InvalidatedPhysPage for ImsicGuestPage<Invalidated> {}
impl ReclaimablePhysPage for ImsicGuestPage<ConvertedClean> {
    type MappablePage = ImsicGuestPage<MappableClean>;
}

/// An iterator over owning references to IMSIC guest file pages.
pub struct ImsicGuestPageIter {
    mem_range: SupervisorPageRange,
}

impl ImsicGuestPageIter {
    // Creates a new `ImsicGuestPageIter` from `mem_range`.
    //
    // # Safety
    //
    // The caller must guarantee that `mem_range` maps a uniquely-owned range of IMSIC guest
    // interrupt file pages.
    unsafe fn new(mem_range: SupervisorPageRange) -> Self {
        Self { mem_range }
    }
}

impl Iterator for ImsicGuestPageIter {
    type Item = ImsicGuestPage<ConvertedClean>;

    fn next(&mut self) -> Option<Self::Item> {
        let addr = self.mem_range.next()?;
        // Safety: The address is guaranteed to be valid and the page it refers to is guaranteed to
        // be uniquely owned at construction of the `ImsicGuestPageIter`.
        let page = unsafe { ImsicGuestPage::new(addr) };
        Some(page)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.mem_range.size_hint()
    }
}

impl ExactSizeIterator for ImsicGuestPageIter {}

// Holds the IMSIC state for a particular CPU.
struct ImsicPerCpu {
    group: ImsicGroupId,
    hart: ImsicHartId,
    // Index at which this CPU appears in 'interrupts-extended'.
    dt_index: usize,
    // True if the host has claimed the guest file pages for this CPU.
    taken: bool,
}

// Holds the per-CPU IMSIC state for all CPUs.
struct ImsicCpuState {
    // Indexed by `CpuId`.
    cpus: ArrayVec<Option<ImsicPerCpu>, MAX_CPUS>,
}

impl ImsicCpuState {
    fn get_cpu(&self, cpu: CpuId) -> Option<&ImsicPerCpu> {
        self.cpus.get(cpu.raw()).and_then(|c| c.as_ref())
    }

    fn get_cpu_mut(&mut self, cpu: CpuId) -> Option<&mut ImsicPerCpu> {
        self.cpus.get_mut(cpu.raw()).and_then(|c| c.as_mut())
    }

    fn dt_index_to_cpu(&self, index: usize) -> Option<CpuId> {
        let cpu = self
            .cpus
            .iter()
            .position(|c| c.as_ref().filter(|cpu| cpu.dt_index == index).is_some())?;
        Some(CpuId::new(cpu))
    }
}

// Determine how many guests are actually supported when running on an actual RISC-V CPU by probing
// HGEIE.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
fn get_guests_per_hart(_guest_index_bits: u32) -> usize {
    let old = CSR.hgeie.get();
    CSR.hgeie.set(!0);
    let guests_per_hart = CSR.hgeie.get().count_ones() as usize;
    CSR.hgeie.set(old);
    guests_per_hart
}

// Assume we can support the full number of guests indicated by `guest_index_bits` in testing
// environments.
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
fn get_guests_per_hart(guest_index_bits: u32) -> usize {
    (1 << guest_index_bits) - 1
}

// Helper struct to provide scoped access to particular guest interrupt file's CSRs.
struct ImsicGuestCsrAccess {
    old_vgein: u64,
}

// The Deref implementation for ImsicGuestCsrAccess currently just returns a reference to the static
// CSR instance, the correct external interrupt source is selected via `hstatus.vgein`.
impl Deref for ImsicGuestCsrAccess {
    type Target = CSR;

    #[inline(always)]
    fn deref(&self) -> &CSR {
        CSR
    }
}

impl Drop for ImsicGuestCsrAccess {
    fn drop(&mut self) {
        CSR.hstatus.modify(hstatus::vgein.val(self.old_vgein));
    }
}

/// System-wide IMSIC state. Used to discover the IMSIC topology from the device-tree and to
/// manage the allocation of guest interrupt files.
pub struct Imsic {
    per_cpu: Mutex<ImsicCpuState>,
    mmio_regions: ArrayVec<SupervisorPageRange, MAX_MMIO_REGIONS>,
    geometry: ImsicGeometry<SupervisorPhys>,
    interrupt_ids: usize,
    phandle: u32,
}

static IMSIC: Once<Imsic> = Once::new();

impl Imsic {
    /// Discovers the IMSIC topology from a device-tree and updates `mem_map` with the IMSIC's
    /// MMIO regions. All guest interrupt files are initialized to free. Panics if the device-tree
    /// is malformed in any way.
    ///
    /// Per-hart IMSIC addresses must follow the scheme specified in the RISC-V AIA spec:
    ///
    /// XLEN-1           >=24                                 12    0
    /// |                  |                                  |     |
    /// -------------------------------------------------------------
    /// |xxxxxx|Group Index|xxxxxxxxxxx|HART Index|Guest Index|  0  |
    /// -------------------------------------------------------------
    ///
    /// For more details about the system IMSIC layout, see the AIA specification.
    pub fn probe_from(dt: &DeviceTree, mem_map: &mut HwMemMap) -> Result<()> {
        // If both M and S level IMSICs are present in the device-tree the M-level IMSIC should
        // have its status set to "disabled" by firmware.
        let imsic_node = dt
            .iter()
            .find(|n| n.compatible(["riscv,imsics"]) && !n.disabled())
            .ok_or(Error::MissingImsicNode)?;

        // There must be a parent interrupt for each CPU.
        let num_cpus = CpuInfo::get().num_cpus();
        let interrupts_prop = imsic_node
            .props()
            .find(|p| p.name() == "interrupts-extended")
            .ok_or(Error::MissingProperty("interrupts-extended"))?;
        // Assumes CPU's #interrupt-cells is 1.
        let interrupts = interrupts_prop.value_u32();
        if interrupts.len() != num_cpus * 2 {
            return Err(Error::InvalidParentInterruptCount(interrupts.len()));
        }

        // Find the IMSIC's phandle. The PCIe controller will refer to it via 'msi-parent'.
        let phandle = imsic_node
            .props()
            .find(|p| p.name() == "phandle")
            .and_then(|p| p.value_u32().next())
            .ok_or(Error::MissingProperty("phandle"))?;

        let interrupt_ids = imsic_node
            .props()
            .find(|p| p.name() == "riscv,num-ids")
            .and_then(|p| p.value_u32().next().map(|v| v as usize))
            .ok_or(Error::MissingProperty("riscv,num-ids"))?;
        if interrupt_ids == 0 || interrupt_ids > MAX_INTERRUPT_IDS {
            return Err(Error::InvalidInterruptIds(interrupt_ids));
        }

        // We must have at least one guest interrupt file to start the host.
        let guest_index_bits = imsic_node
            .props()
            .find(|p| p.name() == "riscv,guest-index-bits")
            .and_then(|p| p.value_u32().next())
            .ok_or(Error::MissingProperty("riscv,guest-index-bits"))?;
        if guest_index_bits == 0 {
            return Err(Error::InvalidGuestsPerHart(0));
        }

        // The actual number of guest files may be less than 2^(guest_index_bits) - 1; we need to
        // interrogate HGEIE.
        let guests_per_hart = get_guests_per_hart(guest_index_bits);
        if guests_per_hart == 0 || guests_per_hart > MAX_GUEST_FILES {
            return Err(Error::InvalidGuestsPerHart(guests_per_hart));
        }

        // The hart index bits immediately follow the guest index bits.
        let hart_index_bits = imsic_node
            .props()
            .find(|p| p.name() == "riscv,hart-index-bits")
            .and_then(|p| p.value_u32().next())
            .unwrap_or_else(|| num_cpus.next_power_of_two().ilog2());

        // If there are multiple groups, their index must be in the upper 8 bits of the address.
        let group_index_bits = imsic_node
            .props()
            .find(|p| p.name() == "riscv,group-index-bits")
            .and_then(|p| p.value_u32().next())
            .unwrap_or(0);
        let group_index_shift = imsic_node
            .props()
            .find(|p| p.name() == "riscv,group-index-shift")
            .and_then(|p| p.value_u32().next())
            .unwrap_or(MIN_GROUP_INDEX_SHIFT);

        let regs_prop = imsic_node
            .props()
            .find(|p| p.name() == "reg")
            .ok_or(Error::MissingProperty("reg"))?;
        let geometry = {
            let mut regs = regs_prop.value_u64();
            if regs.len() == 0 || (regs.len() % 2) != 0 {
                return Err(Error::InvalidMmioRegionCount(regs.len()));
            }
            // Each MMIO range is expected to map a group (or range of groups). Mask out the
            // group bits to get the base address pattern.
            let base_addr = RawAddr::supervisor(
                regs.next().unwrap() & !(((1 << group_index_bits) - 1) << group_index_shift),
            );
            ImsicGeometry::new(
                PageAddr::new(base_addr)
                    .ok_or_else(|| Error::MisalignedMmioRegion(base_addr.bits()))?,
                group_index_bits,
                group_index_shift,
                hart_index_bits,
                guest_index_bits,
                guests_per_hart,
            )
        }?;

        // Now match up interrupt files to CPUs. The "hart index" for a CPU is the order in which
        // its interrupt appears in 'interrupt-parents'.
        let per_hart_size = (1 << geometry.guest_index_bits()) * PageSize::Size4k as u64;
        let mut dt_index = 0;
        let mut interrupts = interrupts_prop.value_u32();
        // We assume that each region listed in the 'reg' property is densely packed with per-hart
        // IMSIC files.
        let mut regs = regs_prop.value_u64();
        let mut mmio_regions = ArrayVec::new();
        let mut per_cpu_state = ArrayVec::new();
        for _ in 0..num_cpus {
            per_cpu_state.push(None);
        }
        while let Some(region_base_addr) = regs.next() {
            let region_base_addr = PageAddr::new(RawAddr::supervisor(region_base_addr))
                .ok_or(Error::MisalignedMmioRegion(region_base_addr))?;
            // Each MMIO region must map the start of a group.
            let location = geometry
                .addr_to_location(region_base_addr)
                .ok_or_else(|| Error::InvalidMmioRegionLocation(region_base_addr.bits()))?;
            if location.hart().bits() != 0 || location.file().bits() != 0 {
                return Err(Error::InvalidMmioRegion(region_base_addr.bits()));
            }
            // Unwrap ok, we've guaranteed there are an even number of `reg` cells.
            let region_size = regs.next().unwrap();
            if region_size % per_hart_size != 0 {
                return Err(Error::MisalignedMmioRegion(region_base_addr.bits()));
            }
            let harts_in_region = region_size / per_hart_size;

            for i in 0..harts_in_region {
                // Each 'interrupts-extended' property is of the from "<phandle> <interrupt-id>".
                // The phandle refers to the CPU's 'interrupt-controller' node, and the interrupt
                // ID should always be the supervisor external interrupt.
                let cpu_intc_phandle = interrupts.next().ok_or(Error::TooManyInterruptFiles)?;
                let cpu_int = interrupts.next().ok_or(Error::TooManyInterruptFiles)?;
                if cpu_int != (sie::sext.shift as u32) {
                    return Err(Error::InvalidParentInterrupt(cpu_intc_phandle, cpu_int));
                }
                let cpu_id = CpuInfo::get()
                    .intc_phandle_to_cpu(cpu_intc_phandle)
                    .ok_or(Error::InvalidParentInterrupt(cpu_intc_phandle, cpu_int))?;
                // Unwrap ok since we must be page-aligned.
                let cpu_base_addr = region_base_addr
                    .checked_add_pages(i * per_hart_size / PageSize::Size4k as u64)
                    .unwrap();
                // Map the interrupt file location back to group/hart indexes.
                let location = geometry
                    .addr_to_location(cpu_base_addr)
                    .ok_or(Error::InvalidCpuLocation(cpu_id))?;

                let cpu_state = ImsicPerCpu {
                    group: location.group(),
                    hart: location.hart(),
                    dt_index,
                    taken: false,
                };
                if per_cpu_state[cpu_id.raw()].is_some() {
                    return Err(Error::DuplicateCpuEntries(cpu_id));
                }
                per_cpu_state[cpu_id.raw()] = Some(cpu_state);

                dt_index += 1;
            }

            mmio_regions.push(PageAddrRange::new(
                region_base_addr,
                PageSize::num_4k_pages(region_size),
            ));
        }
        if dt_index != num_cpus {
            return Err(Error::MissingInterruptFiles);
        }

        // Add the IMSIC ranges to the system memory map.
        for mmio in mmio_regions.iter() {
            unsafe {
                // We trust that the device-tree described the IMSIC topology correctly.
                mem_map
                    .add_mmio_region(
                        DeviceMemType::Imsic,
                        mmio.base().into(),
                        mmio.length_bytes(),
                    )
                    .map_err(Error::AddingMmioRegion)
            }?;
        }

        let imsic = Imsic {
            per_cpu: Mutex::new(ImsicCpuState {
                cpus: per_cpu_state,
            }),
            mmio_regions,
            geometry,
            interrupt_ids,
            phandle,
        };
        IMSIC.call_once(|| imsic);
        Ok(())
    }

    /// Initializes the IMSIC-related CSRs on this CPU. Upon return, the IMSIC on this CPU is set
    /// up to receive IPIs.
    pub fn setup_this_cpu() {
        // Enable external interrupt delivery.
        CSR.si_eidelivery.set(1);
        // We don't care about prioritization, so just set EITHRESHOLD to 0.
        CSR.si_eithreshold.set(0);

        // The only interrupts we handle right now are IPIs.
        let (offset, bit) = ImsicInterruptId::Ipi.offset_and_bit();
        CSR.si_eie[offset].read_and_set_bits(1 << bit);
    }

    /// Returns a reference to the global IMSIC state.
    pub fn get() -> &'static Self {
        IMSIC.get().unwrap()
    }

    /// Returns a descriptor of the physical IMSIC layout.
    pub fn phys_geometry(&self) -> SupervisorImsicGeometry {
        self.geometry.clone()
    }

    /// Returns a descriptor of the expected virtual IMSIC layout for the host VM. The host VM
    /// is expected to use guest interrupt file 0 for its supervisor-level interrupt file with
    /// the remaining guest interrupt files mapped immediately contiguous to it.
    pub fn host_vm_geometry(&self) -> GuestImsicGeometry {
        let phys = &self.geometry;
        ImsicGeometry::new(
            phys.base_addr().as_guest_phys(PageOwnerId::host()),
            phys.group_index_bits(),
            phys.group_index_shift(),
            phys.hart_index_bits(),
            phys.guest_index_bits(),
            phys.guests_per_hart() - 1,
        )
        .unwrap()
    }

    /// Returns the number of implemented external interrupt IDs.
    pub fn interrupt_ids(&self) -> usize {
        self.interrupt_ids
    }

    /// Takes ownership over the guest interrupt file pages for `cpu`, returning an iterator over
    /// the pages.
    pub fn take_guest_files(&self, cpu: CpuId) -> Result<ImsicGuestPageIter> {
        let mut cpus = self.per_cpu.lock();
        let pcpu = cpus.get_cpu_mut(cpu).ok_or(Error::InvalidCpu(cpu))?;
        if pcpu.taken {
            return Err(Error::GuestFilesTaken(cpu));
        }
        pcpu.taken = true;
        let start_loc = ImsicLocation::new(pcpu.group, pcpu.hart, ImsicFileId::guest(0));
        // If the CPU is valid, then its IMSIC location must be valid.
        let base_addr = self.geometry.location_to_addr(start_loc).unwrap();
        let page_range = PageAddrRange::new(base_addr, self.geometry.guests_per_hart() as u64);
        // Safety: `page_range` is a range of IMSIC guest files that we just took ownership of.
        let iter = unsafe { ImsicGuestPageIter::new(page_range) };
        Ok(iter)
    }

    /// Returns the IMSIC location specifier of the interrupt file on the specified physical CPU.
    pub fn phys_file_location(&self, cpu: CpuId, file: ImsicFileId) -> Result<ImsicLocation> {
        let cpus = self.per_cpu.lock();
        let pcpu = cpus.get_cpu(cpu).ok_or(Error::InvalidCpu(cpu))?;
        Ok(ImsicLocation::new(pcpu.group, pcpu.hart, file))
    }

    // Returns the address of the interrupt file on the specified physical CPU.
    fn phys_file_addr(&self, cpu: CpuId, file: ImsicFileId) -> Result<SupervisorPageAddr> {
        self.phys_file_location(cpu, file).and_then(|loc| {
            self.geometry
                .location_to_addr(loc)
                .ok_or(Error::InvalidGuestFile)
        })
    }

    /// Returns the phandle of this IMSIC's node in the device-tree.
    pub fn phandle(&self) -> u32 {
        self.phandle
    }

    /// Sends an IPI with the raw `id` to the specified CPU and interrupt file by writing the
    /// interrupt file's memory-mapped `seteipnum` register. This can be used to inject arbitrary
    /// interrupts into the destination interrupt file. The caller is responsible for ensuring
    /// that the destination CPU and interrupt file is in the proper state to receive the
    /// interrupt.
    pub fn send_ipi_raw(&self, cpu: CpuId, file: ImsicFileId, id: u32) -> Result<()> {
        let addr = self.phys_file_addr(cpu, file)?;
        if id == 0 || (id as usize) >= self.interrupt_ids() {
            return Err(Error::InvalidInterruptId(id));
        }
        unsafe {
            // Safe since `addr` maps a valid IMSIC interrupt file.
            core::ptr::write_volatile(addr.bits() as *mut u32, id)
        };
        Ok(())
    }

    /// Sends a supervisor-level IPI to the specified CPU.
    pub fn send_ipi(&self, cpu: CpuId) -> Result<()> {
        self.send_ipi_raw(cpu, ImsicFileId::Supervisor, ImsicInterruptId::Ipi as u32)
    }

    /// Claims and returns the ID of the next pending interrupt in this CPU's supervisor-level
    /// interrupt file, or `None` if no interrupt is pending.
    pub fn next_pending_interrupt() -> Option<ImsicInterruptId> {
        let raw_id = CSR.stopei.atomic_replace(0) >> stopei::interrupt_id.shift;
        ImsicInterruptId::from_raw(raw_id)
    }

    // Returns the number EIE/EIP registers used by the IMSIC.
    fn num_ei_regs(&self) -> usize {
        self.interrupt_ids.div_ceil(64)
    }

    // Returns an ImsicGuestCsrAccess struct providing scoped access to guest_file's CSRs.
    fn get_guest_csrs(&self, guest_file: ImsicFileId) -> Result<ImsicGuestCsrAccess> {
        let vgein = match guest_file {
            ImsicFileId::Guest(g) if (g as usize) < self.geometry.guests_per_hart() => {
                guest_file.bits()
            }
            _ => {
                return Err(Error::InvalidGuestFile);
            }
        };
        let old_vgein = CSR.hstatus.read(hstatus::vgein);
        CSR.hstatus.modify(hstatus::vgein.val(vgein as u64));
        Ok(ImsicGuestCsrAccess { old_vgein })
    }

    /// Prepares `sw_file` for saving the state of `guest_file`. Interrupt delivery from `guest_file`
    /// is disabled upon return.
    pub fn save_guest_file_prepare(
        &self,
        guest_file: ImsicFileId,
        sw_file: &mut SwFile,
    ) -> Result<()> {
        let csrs = self.get_guest_csrs(guest_file)?;
        for i in 0..self.num_ei_regs() {
            sw_file.set_eip(i, 0);
            sw_file.set_eie(i, csrs.vsi_eie[i].get_value());
        }
        sw_file.set_eithreshold(csrs.vsi_eithreshold.get());
        sw_file.set_eidelivery(csrs.vsi_eidelivery.atomic_replace(0));
        Ok(())
    }

    /// Completes saving of `guest_file` to `sw_file`. The caller must ensure that any translations
    /// in the CPU or MSI page table referencing `guest_file` have been flushed. Upon return
    /// `guest_file` is available for reuse.
    pub fn save_guest_file_finish(
        &self,
        guest_file: ImsicFileId,
        sw_file: &mut SwFile,
    ) -> Result<()> {
        let csrs = self.get_guest_csrs(guest_file)?;
        for i in 0..self.num_ei_regs() {
            // TODO: This needs to be done using an atomic-OR with an actual MRIF.
            let eip = csrs.vsi_eip[i].get_value() | sw_file.eip(i);
            sw_file.set_eip(i, eip);
        }
        Ok(())
    }

    /// Clears the `guest_file`.
    pub fn clear_guest_file(&self, guest_file: ImsicFileId) -> Result<()> {
        let csrs = self.get_guest_csrs(guest_file)?;
        for i in 0..self.num_ei_regs() {
            csrs.vsi_eip[i].set_value(0);
            csrs.vsi_eie[i].set_value(0);
        }
        csrs.vsi_eithreshold.set(0);
        csrs.vsi_eidelivery.set(0);
        Ok(())
    }

    /// Restores `guest_file` from `sw_file`. If `sw_file` is an MRIF mapped into an MSI
    /// page table then the caller must ensure that any translations for `sw_file` have been
    /// flushed. Interrupt delivery from `guest_file` is enabled upon return.
    pub fn restore_guest_file(&self, guest_file: ImsicFileId, sw_file: &mut SwFile) -> Result<()> {
        let csrs = self.get_guest_csrs(guest_file)?;
        for i in 0..self.num_ei_regs() {
            csrs.vsi_eip[i].read_and_set_bits(sw_file.eip(i));
            csrs.vsi_eie[i].set_value(sw_file.eie(i));
        }
        csrs.vsi_eithreshold.set(sw_file.eithreshold());
        csrs.vsi_eidelivery.set(sw_file.eidelivery());
        Ok(())
    }

    /// Adds an IMSIC node to the host device-tree using the layout specified in
    /// `self.host_vm_geometry()`. It is up to the caller to remap interrupt files appropriately.
    /// In particular, the guest interrupt files dedicated to the host VM should be mapped to the
    /// supervisor interrupt file location in the guest physical address space. The caller is also
    /// responsible for masking guest interrupt files that are inaccessible to the host VM in HGEIE
    /// and other CSRs.
    pub fn add_host_imsic_node(&self, dt: &mut DeviceTree) -> DeviceTreeResult<()> {
        let geometry = self.host_vm_geometry();
        let soc_node_id = dt.iter().find(|n| n.name() == "soc").unwrap().id();
        let mut imsic_name = ArrayString::<32>::new();
        fmt::write(
            &mut imsic_name,
            format_args!("imsics@{:x}", self.mmio_regions[0].base().bits()),
        )
        .unwrap();
        let imsic_id = dt.add_node(imsic_name.as_str(), Some(soc_node_id))?;
        let imsic_node = dt.get_mut_node(imsic_id).unwrap();

        // We report the full capabilities of the IMSIC to the host. It's up to the caller to
        // map pages accordingly, and to hide the guest interrupt file that the host itself is
        // using from HGEIE/HGEIP.
        imsic_node
            .add_prop("compatible")?
            .set_value_str("riscv,imsics")?;
        imsic_node
            .add_prop("phandle")?
            .set_value_u32(&[self.phandle])?;
        imsic_node.add_prop("msi-controller")?;
        imsic_node.add_prop("interrupt-controller")?;
        imsic_node
            .add_prop("#interrupt-cells")?
            .set_value_u32(&[0])?;
        imsic_node
            .add_prop("riscv,guest-index-bits")?
            .set_value_u32(&[geometry.guest_index_bits()])?;
        imsic_node
            .add_prop("riscv,hart-index-bits")?
            .set_value_u32(&[geometry.hart_index_bits()])?;
        imsic_node.add_prop("riscv,ipi-id")?.set_value_u32(&[1])?;
        imsic_node
            .add_prop("riscv,num-ids")?
            .set_value_u32(&[self.interrupt_ids as u32])?;
        if self.geometry.group_index_bits() > 0 {
            // These only matter when we have multiple groups.
            imsic_node
                .add_prop("riscv,group-index-bits")?
                .set_value_u32(&[geometry.group_index_bits()])?;
            imsic_node
                .add_prop("riscv,group-index-shift")?
                .set_value_u32(&[geometry.group_index_shift()])?;
        }

        // Now add a 'reg' entry for each MMIO region, replicating exactly the `reg` property
        // set in the hypervisor's device tree.
        let mut regs = ArrayVec::<u64, { 2 * MAX_MMIO_REGIONS }>::new();
        for mmio in self.mmio_regions.iter() {
            regs.push(mmio.base().bits());
            regs.push(mmio.length_bytes());
        }
        imsic_node.add_prop("reg")?.set_value_u64(&regs)?;

        // Add an 'interrupts-extended' entry for each CPU, which must be in hart-index order.
        let mut interrupts = ArrayVec::<u32, { 2 * MAX_CPUS }>::new();
        let num_cpus = CpuInfo::get().num_cpus();
        let cpus = self.per_cpu.lock();
        for i in 0..num_cpus {
            // Unwrap ok, each index must appear in `per_cpu_state` by construction.
            let cpu_id = cpus.dt_index_to_cpu(i).unwrap();
            let phandle = CpuInfo::get().cpu_to_intc_phandle(cpu_id).unwrap();
            interrupts.push(phandle);
            interrupts.push(sie::sext.shift as u32);
        }
        imsic_node
            .add_prop("interrupts-extended")?
            .set_value_u32(&interrupts)?;

        Ok(())
    }
}
