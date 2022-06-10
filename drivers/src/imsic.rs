// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::{ArrayString, ArrayVec};
use core::{alloc::Allocator, fmt, marker::PhantomData};
use device_tree::{DeviceTree, DeviceTreeResult};
use riscv_page_tables::HwMemMap;
use riscv_pages::*;
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use riscv_regs::Readable;
use riscv_regs::{sie, stopei, Writeable, CSR};
use spin::{Mutex, Once};

use crate::{CpuId, CpuInfo, MAX_CPUS};

const MAX_GUEST_FILES: usize = 7;
const MAX_MMIO_REGIONS: usize = 8;
const MIN_GROUP_SHIFT: u32 = 24; // As mandated by the AIA spec.

/// Errors that can be returned when claiming or releasing guest interrupt files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidCpu(CpuId),
    InvalidGuestFile,
    GuestFileTaken,
    GuestFileFree,
}

pub type Result<T> = core::result::Result<T, Error>;

/// IMSIC indirect CSRs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImsicRegister {
    Eidelivery,
    Eithreshold,
    Eie(u64),
}

impl ImsicRegister {
    /// Returns the ISELECT value used to access this register.
    fn to_raw(self) -> u64 {
        match self {
            ImsicRegister::Eidelivery => 0x70,
            ImsicRegister::Eithreshold => 0x72,
            ImsicRegister::Eie(i) => 0xc0 + i / 64,
        }
    }
}

/// IMSIC external interrupt IDs.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImsicInterruptId {
    // For now, we only expect to handle IPIs at HS-level.
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

    /// Returns the indirect EIE register used to enable this interrupt.
    fn eie_register(&self) -> ImsicRegister {
        ImsicRegister::Eie(*self as u64 / 64)
    }

    /// Returns the bit position of this interrupt in its indirect EIE register.
    fn eie_bit(&self) -> u64 {
        *self as u64 % 64
    }
}

/// A `PhysPage` implementation representing an IMSIC guest interrupt file page.
pub struct ImsicGuestPage<S: State> {
    addr: SupervisorPageAddr,
    state: PhantomData<S>,
}

impl<S: State> PhysPage for ImsicGuestPage<S> {
    /// Creates a new `ImsicGuestPage` at the given page-aligned address. IMSIC pages are always 4kB.
    ///
    /// # Safety
    ///
    /// The caller must ensure `addr` refers to a uniquely owned IMSIC guest interrupt file.
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
        MemType::Mmio(DeviceMemType::Imsic)
    }
}

// IMSIC interrupt file pages retain no state so they are always considered "clean".
impl MappablePhysPage<MeasureOptional> for ImsicGuestPage<MappableClean> {}
impl AssignablePhysPage<MeasureOptional> for ImsicGuestPage<ConvertedClean> {
    type MappablePage = ImsicGuestPage<MappableClean>;
}
impl ConvertedPhysPage for ImsicGuestPage<ConvertedClean> {}
impl InvalidatedPhysPage for ImsicGuestPage<Invalidated> {
    type ConvertedPage = ImsicGuestPage<ConvertedClean>;
}
impl ReclaimablePhysPage for ImsicGuestPage<ConvertedClean> {
    type MappablePage = ImsicGuestPage<MappableClean>;
}

/// Represents an IMSIC guest interrupt file ID for use in reading/writing VGEIN, HGEIE, etc.
/// The host VM always gets interrupt file 1 and the remaining files are for its guests. File 0
/// is always invalid, as per the AIA spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImsicGuestId {
    HostVm,
    GuestVm(usize),
}

impl ImsicGuestId {
    /// Creates an `ImsicGuestId` from the raw interrupt file number.
    pub fn from_raw_index(index: usize) -> Option<Self> {
        match index {
            0 => None,
            1 => Some(ImsicGuestId::HostVm),
            i => Some(ImsicGuestId::GuestVm(i - 2)),
        }
    }

    /// Returns the raw interrupt file number for this interrupt file ID.
    pub fn to_raw_index(&self) -> usize {
        match self {
            ImsicGuestId::HostVm => 1,
            ImsicGuestId::GuestVm(g) => g + 2,
        }
    }
}

/// Indicates whether or not a guest interrupt file is available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImsicGuestState {
    Free,
    Taken,
}

/// Holds per-CPU IMSIC state.
struct ImsicCpuState {
    // The base address of this hart's IMSIC. Corresponds to the supervisor level interrupt file
    // for this hart; the guest interrupt files immediately follow it.
    base_addr: SupervisorPageAddr,
    guest_files: ArrayVec<ImsicGuestState, MAX_GUEST_FILES>,
}

impl ImsicCpuState {
    /// Creates and initializes the IMSIC per-CPU state starting at `base_addr`. All interrupt files
    /// are initially free.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `base_addr` is the address of an IMSIC supervisor-level
    /// interrupt file with `num_guests` VS-level interrupt files. The caller must uniquely own
    /// this set of pages.
    unsafe fn new(base_addr: SupervisorPageAddr, num_guests: usize) -> Self {
        let mut guest_files = ArrayVec::new();
        for _ in 0..num_guests {
            guest_files.push(ImsicGuestState::Free)
        }
        Self {
            base_addr,
            guest_files,
        }
    }

    fn base_addr(&self) -> SupervisorPageAddr {
        self.base_addr
    }

    fn take_guest_file(
        &mut self,
        guest_id: ImsicGuestId,
    ) -> Result<ImsicGuestPage<ConvertedClean>> {
        let index = guest_id.to_raw_index();
        let state = self
            .guest_files
            .get_mut(index - 1)
            .ok_or(Error::InvalidGuestFile)?;
        if *state != ImsicGuestState::Free {
            return Err(Error::GuestFileTaken);
        }
        *state = ImsicGuestState::Taken;
        let addr = self.base_addr.checked_add_pages(index as u64).unwrap();
        let page = unsafe {
            // Safe since we've verified that the guest file is free and we know that this is an
            // IMSIC address.
            ImsicGuestPage::new(addr)
        };
        Ok(page)
    }

    fn put_guest_file(&mut self, page: ImsicGuestPage<ConvertedClean>) -> Result<()> {
        let guest_id = page
            .pfn()
            .bits()
            .checked_sub(self.base_addr.pfn().bits())
            .and_then(|i| ImsicGuestId::from_raw_index(i as usize))
            .ok_or(Error::InvalidGuestFile)?;
        let state = self
            .guest_files
            .get_mut(guest_id.to_raw_index() - 1)
            .ok_or(Error::InvalidGuestFile)?;
        if *state != ImsicGuestState::Taken {
            return Err(Error::GuestFileFree);
        }
        *state = ImsicGuestState::Free;
        Ok(())
    }
}

/// A contiguous region of IMSICs. Usually one per IMSIC group in the system.
#[derive(Clone, Copy, Debug)]
struct ImsicMmioRegion {
    base_addr: SupervisorPageAddr,
    size: u64,
}

impl ImsicMmioRegion {
    /// Creates a new IMSIC MMIO region at [`base_addr`, `base_addr` + `size`).
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the specified address ranges maps to an IMSIC MMIO region.
    unsafe fn new(base_addr: SupervisorPageAddr, size: u64) -> Self {
        Self { base_addr, size }
    }

    fn base_addr(&self) -> SupervisorPageAddr {
        self.base_addr
    }

    fn size(&self) -> u64 {
        self.size
    }
}

/// Holds the global state of the IMSICs across the system.
struct ImsicState {
    per_cpu_state: ArrayVec<ImsicCpuState, MAX_CPUS>,
    hart_index_map: ArrayVec<usize, MAX_CPUS>,
    mmio_regions: ArrayVec<ImsicMmioRegion, MAX_MMIO_REGIONS>,
    group_index_bits: u32,
    group_index_shift: u32,
    hart_index_bits: u32,
    guest_index_bits: u32,
    guests_per_hart: usize,
    interrupt_ids: u32,
}

impl ImsicState {
    fn get_cpu(&self, cpu: CpuId) -> Option<&ImsicCpuState> {
        let &index = self.hart_index_map.get(cpu.raw())?;
        self.per_cpu_state.get(index)
    }

    fn get_cpu_mut(&mut self, cpu: CpuId) -> Option<&mut ImsicCpuState> {
        let &index = self.hart_index_map.get(cpu.raw())?;
        self.per_cpu_state.get_mut(index)
    }
}

/// System-wide IMSIC state. Used to discover the IMSIC topology from the device-tree and to
/// manage the allocation of guest interrupt files.
pub struct Imsic {
    inner: Mutex<ImsicState>,
}

static IMSIC: Once<Imsic> = Once::new();

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
    (1 << guest_index_bits as usize) - 1
}

fn indirect_csr_write(reg: ImsicRegister, val: u64) {
    CSR.siselect.set(reg.to_raw());
    CSR.sireg.set(val);
}

fn indirect_csr_set_bits(reg: ImsicRegister, mask: u64) {
    CSR.siselect.set(reg.to_raw());
    CSR.sireg.read_and_set_bits(mask);
}

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
    pub fn probe_from<A: Allocator + Clone>(dt: &DeviceTree<A>, mem_map: &mut HwMemMap) {
        // If both M and S level IMSICs are present in the device-tree the M-level IMSIC should
        // have its status set to "disabled" by firmware.
        let imsic_node = dt
            .iter()
            .find(|n| {
                n.props().any(|p| {
                    p.name() == "compatible" && p.value_str().unwrap_or("").contains("riscv,imsics")
                }) && !n
                    .props()
                    .any(|p| p.name() == "status" && p.value_str().unwrap_or("") == "disabled")
            })
            .expect("No IMSIC node in deivce-tree");

        // There must be a parent interrupt for each CPU.
        let num_cpus = CpuInfo::get().num_cpus();
        let interrupts_prop = imsic_node
            .props()
            .find(|p| p.name() == "interrupts-extended")
            .expect("No 'interrupts-extended' property in IMSIC node");
        // Assumes CPU's #interrupt-cells is 1.
        assert_eq!(interrupts_prop.value_u32().count(), num_cpus * 2);

        let interrupt_ids = imsic_node
            .props()
            .find(|p| p.name() == "riscv,num-ids")
            .expect("Number of interrupt IDs not set in IMSIC node")
            .value_u32()
            .next()
            .unwrap();

        // We must have at least one guest interrupt file to start the host.
        let guest_index_bits = imsic_node
            .props()
            .find(|p| p.name() == "riscv,guest-index-bits")
            .expect("No 'riscv,guest-index-bits' in IMSIC node")
            .value_u32()
            .next()
            .unwrap();
        assert!(guest_index_bits > 0);

        // The actual number of guest files may be less than 2^(guest_index_bits) - 1; we need to
        // interrogate HGEIE.
        let guests_per_hart = get_guests_per_hart(guest_index_bits);
        assert!(guests_per_hart > 0);
        assert!(guests_per_hart < (1 << guest_index_bits));
        assert!(guests_per_hart <= MAX_GUEST_FILES);

        // The hart index bits immediately follow the guest index bits.
        let hart_index_bits = imsic_node
            .props()
            .find(|p| p.name() == "riscv,hart-index-bits")
            .and_then(|p| p.value_u32().next())
            .unwrap_or_else(|| num_cpus.next_power_of_two().log2());

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
            .unwrap_or(MIN_GROUP_SHIFT);
        assert!(group_index_shift < u64::BITS);
        assert!(group_index_shift >= MIN_GROUP_SHIFT);
        let pfn_shift = (PageSize::Size4k as u64).log2();
        assert!(group_index_shift >= pfn_shift + hart_index_bits + guest_index_bits);

        // Now match up interrupt files to CPUs. The "hart index" for a CPU is the order in which
        // its interrupt appears in 'interrupt-parents'.
        let per_hart_size = (1 << guest_index_bits) * PageSize::Size4k as u64;
        let mut hart_index_map = ArrayVec::new();
        for _ in 0..num_cpus {
            hart_index_map.push(0);
        }
        let mut hart_index = 0;
        let mut interrupts = interrupts_prop.value_u32();
        // We assume that each region listed in the 'reg' property is densely packed with per-hart
        // IMSIC files.
        let mut regs = imsic_node
            .props()
            .find(|p| p.name() == "reg")
            .expect("No 'reg' property in IMSIC node")
            .value_u64();
        let mut mmio_regions = ArrayVec::new();
        let mut per_cpu_state = ArrayVec::new();
        while let Some(region_base_addr) = regs.next() {
            let region_base_addr = PageAddr::new(RawAddr::supervisor(region_base_addr)).unwrap();
            let region_size = regs.next().unwrap();
            assert_eq!(region_size % per_hart_size, 0);
            let harts_in_region = region_size / per_hart_size;

            for i in 0..harts_in_region {
                // Each 'interrupts-extended' property is of the from "<phandle> <interrupt-id>".
                // The phandle refers to the CPU's 'interrupt-controller' node, and the interrupt
                // ID should always be the supervisor external interrupt.
                let cpu_intc_phandle = interrupts.next().unwrap();
                let cpu_int = interrupts.next().unwrap();
                assert_eq!(cpu_int, sie::sext.shift as u32);
                let cpu_id = CpuInfo::get()
                    .intc_phandle_to_cpu(cpu_intc_phandle)
                    .unwrap();
                let cpu_base_addr = region_base_addr
                    .checked_add_pages(i * per_hart_size / PageSize::Size4k as u64)
                    .unwrap();

                // Map hart index -> CPU ID and put this hart's state in the per-CPU array.
                hart_index_map[cpu_id.raw()] = hart_index;
                unsafe {
                    // We trust that the device-tree described the IMSIC topology correctly.
                    per_cpu_state.push(ImsicCpuState::new(cpu_base_addr, guests_per_hart))
                };

                hart_index += 1;
            }

            unsafe {
                // We trust that the device-tree described the IMSIC topology correctly.
                mmio_regions.push(ImsicMmioRegion::new(region_base_addr, region_size))
            };
        }
        assert_eq!(per_cpu_state.len(), num_cpus);

        // Now make sure all MMIO regions match the expected geometry, and add them to the system
        // memory map.
        let base_addr = mmio_regions[0].base_addr();
        for mmio in mmio_regions.iter() {
            let masked_addr = mmio.base_addr().bits()
                & !((1 << (pfn_shift + hart_index_bits + guest_index_bits)) - 1)
                & !(((1 << group_index_bits) - 1) << group_index_shift);
            assert_eq!(masked_addr, base_addr.bits());

            unsafe {
                // We trust that the device-tree described the IMSIC topology correctly.
                mem_map
                    .add_mmio_region(
                        DeviceMemType::Imsic,
                        RawAddr::from(mmio.base_addr()),
                        mmio.size(),
                    )
                    .unwrap();
            }
        }

        let imsic = ImsicState {
            per_cpu_state,
            hart_index_map,
            mmio_regions,
            group_index_bits,
            group_index_shift,
            hart_index_bits,
            guest_index_bits,
            guests_per_hart,
            interrupt_ids,
        };
        IMSIC.call_once(|| Self {
            inner: Mutex::new(imsic),
        });
    }

    /// Initializes the IMSIC-related CSRs on this CPU. Upon return, the IMSIC on this CPU is set
    /// up to receive IPIs.
    pub fn setup_this_cpu() {
        // Enable external interrupt delivery.
        indirect_csr_write(ImsicRegister::Eidelivery, 1);
        // We don't care about prioritization, so just set EITHRESHOLD to 0.
        indirect_csr_write(ImsicRegister::Eithreshold, 0);

        // The only interrupts we handle right now are IPIs.
        let id = ImsicInterruptId::Ipi;
        indirect_csr_set_bits(id.eie_register(), 1 << id.eie_bit());
    }

    /// Returns a reference to the global IMSIC state.
    pub fn get() -> &'static Self {
        IMSIC.get().unwrap()
    }

    /// Allocates a guest interrupt file on the specified CPU, returning an `ImsicGuestPage`
    /// representing ownership of that file upon success.
    pub fn take_guest_file(
        &self,
        cpu: CpuId,
        guest_id: ImsicGuestId,
    ) -> Result<ImsicGuestPage<ConvertedClean>> {
        let mut imsic = self.inner.lock();
        let pcpu = imsic.get_cpu_mut(cpu).ok_or(Error::InvalidCpu(cpu))?;
        pcpu.take_guest_file(guest_id)
    }

    /// Releases the guest interrupt file represented by `page` on the given CPU.
    pub fn put_guest_file(&self, cpu: CpuId, page: ImsicGuestPage<ConvertedClean>) -> Result<()> {
        let mut imsic = self.inner.lock();
        let pcpu = imsic.get_cpu_mut(cpu).ok_or(Error::InvalidCpu(cpu))?;
        pcpu.put_guest_file(page)
    }

    /// Returns the number of guest interrupt files suppoorted on each CPU.
    pub fn guests_per_hart(&self) -> usize {
        self.inner.lock().guests_per_hart
    }

    /// Returns the base address of the system's IMSIC hierarchy.
    pub fn base_addr(&self) -> SupervisorPageAddr {
        self.inner.lock().mmio_regions[0].base_addr()
    }

    /// Returns the base address of the supervisor level interrupt file for the given CPU. Can
    /// be used to direct MSIs to that CPU.
    pub fn supervisor_file_addr(&self, cpu: CpuId) -> Result<SupervisorPageAddr> {
        let imisc = self.inner.lock();
        let pcpu = imisc.get_cpu(cpu).ok_or(Error::InvalidCpu(cpu))?;
        Ok(pcpu.base_addr())
    }

    /// Sends an IPI to the specified CPU.
    pub fn send_ipi(&self, cpu: CpuId) -> Result<()> {
        let addr = self.supervisor_file_addr(cpu)?;
        unsafe {
            // Safe since `addr` maps a valid supervisor-level IMSIC interrupt file.
            core::ptr::write_volatile(addr.bits() as *mut u32, ImsicInterruptId::Ipi as u32)
        };
        Ok(())
    }

    /// Claims and returns the ID of the next pending interrupt in this CPU's supervisor-level
    /// interrupt file, or `None` if no interrupt is pending.
    pub fn next_pending_interrupt() -> Option<ImsicInterruptId> {
        let raw_id = CSR.stopei.atomic_replace(0) >> stopei::interrupt_id.shift;
        ImsicInterruptId::from_raw(raw_id)
    }

    /// Adds an IMSIC node to the host device-tree, with the IMSIC starting at `guest_base_addr`.
    /// The IMSIC hierarchy is otherwise replicated as-is from the supervisor-level IMSIC. It is
    /// up to the caller to remap interrupt files appropriately. In particular, the guest interrupt
    /// files dedicated to the host VM should be mapped to the supervisor interrupt file location in
    /// the guest physical address space. The caller is also responsible for masking guest interrupt
    /// files that are inaccessible to the host VM in HGEIE and other CSRs.
    pub fn add_host_imsic_node<A: Allocator + Clone>(
        &self,
        dt: &mut DeviceTree<A>,
        guest_base_addr: GuestPhysAddr,
    ) -> DeviceTreeResult<()> {
        let imsic = self.inner.lock();

        let soc_node_id = dt.iter().find(|n| n.name() == "soc").unwrap().id();
        let mut imsic_name = ArrayString::<32>::new();
        fmt::write(
            &mut imsic_name,
            format_args!("imsics@{:x}", guest_base_addr.bits()),
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
        imsic_node.add_prop("msi-controller")?;
        imsic_node.add_prop("interrupt-controller")?;
        imsic_node
            .add_prop("#interrupt-cells")?
            .set_value_u32(&[0])?;
        imsic_node
            .add_prop("riscv,guest-index-bits")?
            .set_value_u32(&[imsic.guest_index_bits])?;
        imsic_node
            .add_prop("riscv,hart-index-bits")?
            .set_value_u32(&[imsic.hart_index_bits])?;
        imsic_node.add_prop("riscv,ipi-id")?.set_value_u32(&[1])?;
        imsic_node
            .add_prop("riscv,num-ids")?
            .set_value_u32(&[imsic.interrupt_ids])?;
        if imsic.group_index_bits > 0 {
            // These only matter when we have multiple groups.
            imsic_node
                .add_prop("riscv,group-index-bits")?
                .set_value_u32(&[imsic.group_index_bits])?;
            imsic_node
                .add_prop("riscv,group-index-shift")?
                .set_value_u32(&[imsic.group_index_shift])?;
        }

        // Now add a 'reg' entry for each MMIO region. We replicate the same IMSIC topology, except
        // shifted so that it starts at `guest_base_addr` in the address space.
        let mut regs = ArrayVec::<u64, { 2 * MAX_MMIO_REGIONS }>::new();
        let offset = guest_base_addr.bits() - imsic.mmio_regions[0].base_addr().bits();
        for mmio in imsic.mmio_regions.iter() {
            regs.push(mmio.base_addr().bits() + offset);
            regs.push(mmio.size());
        }
        imsic_node.add_prop("reg")?.set_value_u64(&regs)?;

        // Add an 'interrupts-extended' entry for each CPU.
        let mut interrupts = ArrayVec::<u32, { 2 * MAX_CPUS }>::new();
        let num_cpus = CpuInfo::get().num_cpus();
        for i in 0..num_cpus {
            let cpu_id = CpuId::new(imsic.hart_index_map.iter().position(|&h| h == i).unwrap());
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
