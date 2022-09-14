// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use attestation::measurement::AttestationManager;
use core::arch::global_asm;
use core::marker::PhantomData;
use drivers::{imsic::*, iommu::*, pci::PciBarPage, pci::PciDevice, pci::PcieRoot};
use page_tracking::collections::PageVec;
use page_tracking::{
    LockedPageList, PageList, PageTracker, PageTrackingError, TlbVersion, MAX_PAGE_OWNERS,
};
use riscv_page_tables::{
    tlb, GuestStageMapper, GuestStagePageTable, GuestStagePagingMode, PageTableError,
};
use riscv_pages::*;
use riscv_regs::{
    hgatp, hstatus, DecodedInstruction, Exception, LocalRegisterCopy, PrivilegeLevel, Readable,
    RiscvCsrInterface, Writeable, CSR,
};
use spin::{Mutex, Once};

use crate::vm::{Vm, VmStateFinalized, VmStateInitializing};
use crate::vm_cpu::VmCpus;
use crate::vm_id::VmId;

#[derive(Debug)]
pub enum Error {
    GuestId(PageTrackingError),
    Paging(PageTableError),
    PageFault(PageFaultType),
    NestingTooDeep,
    UnalignedAddress,
    UnsupportedPageSize(PageSize),
    NonContiguousPages,
    AddressOverflow,
    TlbCountUnderflow,
    InvalidTlbVersion,
    TlbFenceInProgress,
    OverlappingVmRegion,
    InsufficientVmRegionSpace,
    InvalidMapRegion,
    SharedPageNotMapped,
    Measurement(attestation::Error),
    VmCreationFailed(crate::vm::Error),
    NoImsicVirtualization,
    ImsicGeometryAlreadySet,
    IommuContextAlreadySet,
    NoIommu,
    AllocatingGscId(IommuError),
    CreatingMsiPageTable(IommuError),
    InvalidImsicLocation,
    MsiTableMapping(IommuError),
    AttachingDevice(IommuError),
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum InstructionFetchError {
    FailedDecode(u32),
    FetchFault,
}

pub type InstructionFetchResult = core::result::Result<DecodedInstruction, InstructionFetchError>;

/// The number of pages for the `VmRegionList` vector.
pub const TVM_REGION_LIST_PAGES: u64 = 1;

/// The base number of state pages required to be donated for creating a new VM. For now, we just need
/// one page to hold the VM state itself and whatever is required to hold the `VmRegionList`.
pub const TVM_STATE_PAGES: u64 = 1 + TVM_REGION_LIST_PAGES;

global_asm!(include_str!("guest_mem.S"));

// The copy to/from guest memory routines defined in guest_mem.S.
extern "C" {
    fn _copy_to_guest(dest_gpa: u64, src: *const u8, len: usize) -> usize;
    fn _copy_from_guest(dest: *mut u8, src_gpa: u64, len: usize) -> usize;
    fn _fetch_guest_instruction(src_gva: u64, raw_inst: *mut u32) -> isize;
}

/// A TLB version + reference count pair, used to track if a given TLB version is currently active.
#[derive(Clone, Default, Debug)]
struct RefCountedTlbVersion {
    version: TlbVersion,
    count: u64,
}

impl RefCountedTlbVersion {
    /// Creates a new reference counter for `version`.
    fn new(version: TlbVersion) -> Self {
        Self { version, count: 0 }
    }

    /// Returns the inner version number.
    fn version(&self) -> TlbVersion {
        self.version
    }

    /// Returns the reference count for this version.
    fn count(&self) -> u64 {
        self.count
    }

    /// Increments the reference count for this version.
    fn get(&mut self) {
        self.count += 1;
    }

    /// Decrements the reference count for this version.
    fn put(&mut self) -> Result<()> {
        self.count = self.count.checked_sub(1).ok_or(Error::TlbCountUnderflow)?;
        Ok(())
    }
}

struct TlbTrackerInner {
    current: RefCountedTlbVersion,
    prev: Option<RefCountedTlbVersion>,
}

/// Tracker for TLB versioning. Used to track which TLB versions are active in an `ActiveVmPages`
/// and coordinate increments of the TLB version when requested via a fence operation.
struct TlbTracker {
    inner: Mutex<TlbTrackerInner>,
}

impl TlbTracker {
    /// Creates a new `TlbTracker` with no references.
    fn new() -> Self {
        let inner = TlbTrackerInner {
            current: RefCountedTlbVersion::default(),
            prev: None,
        };
        Self {
            inner: Mutex::new(inner),
        }
    }

    /// Returns the current TLB version of this tracker.
    fn current(&self) -> TlbVersion {
        self.inner.lock().current.version
    }

    /// Attempts to increment the current TLB version. The TLB version can only be incremented if
    /// there are no outstanding references to versions other than the current version.
    fn increment(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        if inner.prev.as_ref().filter(|v| v.count() != 0).is_none() {
            // We're only ok to proceed with an increment if there's no references to the previous
            // TLB version.
            let next = inner.current.version().increment();
            inner.prev = Some(inner.current.clone());
            inner.current = RefCountedTlbVersion::new(next);
            Ok(())
        } else {
            Err(Error::TlbFenceInProgress)
        }
    }

    /// Acquires a reference to the current TLB version.
    fn get_version(&self) -> TlbVersion {
        let mut inner = self.inner.lock();
        inner.current.get();
        inner.current.version()
    }

    /// Drops a reference to the given TLB version.
    fn put_version(&self, version: TlbVersion) -> Result<()> {
        let mut inner = self.inner.lock();
        if inner.current.version() == version {
            inner.current.put()
        } else if let Some(prev) = inner.prev.as_mut().filter(|v| v.version() == version) {
            prev.put()
        } else {
            Err(Error::InvalidTlbVersion)
        }
    }
}

/// Types of regions in a VM's guest physical address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VmRegionType {
    /// Memory that is private to this VM.
    Confidential,
    /// Memory that is shared with the parent
    Shared,
    /// Emulated MMIO region; accesses always cause a fault that is forwarded to the VM's host.
    Mmio,
    /// IMSIC interrupt file pages.
    Imsic,
    /// PCI BAR pages.
    Pci,
}

/// A contiguous region of guest physical address space.
struct VmRegion {
    start: GuestPageAddr,
    end: GuestPageAddr,
    region_type: VmRegionType,
}

/// The regions of guest physical address space for a VM. Used to track which parts of the address
/// space are designated for a particular purpose. The region list is created during VM initialization
/// and remains static after the VM is finalized. Pages may only be inserted into a VM's address space
/// if the mapping falls within a region of the proper type.
pub struct VmRegionList {
    regions: Mutex<PageVec<VmRegion>>,
}

impl VmRegionList {
    /// Creates a new `VmRegionList` using `pages` as the backing store.
    pub fn new(pages: SequentialPages<InternalClean>, page_tracker: PageTracker) -> Self {
        Self {
            regions: Mutex::new(PageVec::new(pages, page_tracker)),
        }
    }

    /// Inserts a region at [`start`, `end`) of type `region_type`.
    fn add(
        &self,
        mut start: GuestPageAddr,
        end: GuestPageAddr,
        region_type: VmRegionType,
    ) -> Result<()> {
        let mut regions = self.regions.lock();
        // Keep the list sorted, inserting the region in the requested spot as long as it doesn't
        // overlap with anything else.
        let mut index = 0;
        for other in regions.iter() {
            if other.start > start {
                if other.start < end {
                    return Err(Error::OverlappingVmRegion);
                }
                break;
            } else if other.end > start {
                return Err(Error::OverlappingVmRegion);
            }
            index += 1;
        }
        let region = VmRegion {
            start,
            end,
            region_type,
        };
        regions
            .try_reserve(1)
            .map_err(|_| Error::InsufficientVmRegionSpace)?;
        regions.insert(index, region);

        // Coalesce with same-typed regions before and after this one, if possible.
        // Avoid potential underflows and overflows on index.
        if let Some(ref mut before) = index.checked_sub(1).and_then(|i| regions.get_mut(i)) {
            if before.end == start && before.region_type == region_type {
                before.end = end;
                start = before.start;
                regions.remove(index);
                index -= 1;
            }
        }

        if let Some(ref mut after) = index.checked_add(1).and_then(|i| regions.get_mut(i)) {
            if after.start == end && after.region_type == region_type {
                after.start = start;
                regions.remove(index);
            }
        }

        Ok(())
    }

    /// Returns if the range [`start`, `end`) is fully contained within a region of type `region_type`.
    fn contains(
        &self,
        start: GuestPageAddr,
        end: GuestPageAddr,
        region_type: VmRegionType,
    ) -> bool {
        let regions = self.regions.lock();
        regions
            .iter()
            .any(|r| r.start <= start && r.end >= end && r.region_type == region_type)
    }

    /// Returns the type of the region that `addr` resides in, or `None` if it's not in any region.
    fn find(&self, addr: GuestPhysAddr) -> Option<VmRegionType> {
        let regions = self.regions.lock();
        regions
            .iter()
            .find(|r| r.start.bits() <= addr.bits() && r.end.bits() > addr.bits())
            .map(|r| r.region_type)
    }
}

/// Wrapper for a `GuestStageMapper` created from the page table of `VmPages`. Measures pages as
/// they are inserted, if necessary.
pub struct VmPagesMapper<'a, T: GuestStagePagingMode, S, M> {
    vm_pages: &'a VmPages<T, S>,
    mapper: GuestStageMapper<'a, T>,
    _mapper_type: PhantomData<M>,
}

impl<'a, T: GuestStagePagingMode, S, M> VmPagesMapper<'a, T, S, M> {
    // Creates a new `VmPagesMapper` for `num_pages` starting at `page_addr`, which must lie within
    // a region of type `region_type`.
    fn new_in_region(
        vm_pages: &'a VmPages<T, S>,
        page_addr: GuestPageAddr,
        num_pages: u64,
        region_type: VmRegionType,
    ) -> Result<Self> {
        let end = page_addr
            .checked_add_pages(num_pages)
            .ok_or(Error::AddressOverflow)?;
        if !vm_pages.regions.contains(page_addr, end, region_type) {
            return Err(Error::InvalidMapRegion);
        }
        let mapper = vm_pages
            .root
            .map_range(page_addr, PageSize::Size4k, num_pages, &mut || {
                vm_pages.pte_pages.pop()
            })
            .map_err(Error::Paging)?;
        Ok(Self {
            vm_pages,
            mapper,
            _mapper_type: PhantomData,
        })
    }

    // Maps `page` at `to_addr`.
    fn do_map_page<P, MR>(&self, to_addr: GuestPageAddr, page: P) -> Result<()>
    where
        P: MappablePhysPage<MR>,
        MR: MeasureRequirement,
    {
        self.mapper.map_page(to_addr, page).map_err(Error::Paging)
    }
}

pub enum ZeroPages {}
/// A `VmPagesMapper` for confidential zero pages.
pub type ZeroPagesMapper<'a, T, S> = VmPagesMapper<'a, T, S, ZeroPages>;

impl<'a, T: GuestStagePagingMode, S> ZeroPagesMapper<'a, T, S> {
    /// Maps a zero page into the guest's address space.
    pub fn map_page(&self, to_addr: GuestPageAddr, page: Page<MappableClean>) -> Result<()> {
        self.do_map_page(to_addr, page)
    }
}

pub enum MeasuredPages {}
/// A `VmPagesMapper` for confidential measured pages.
pub type MeasuredPagesMapper<'a, T> = VmPagesMapper<'a, T, VmStateInitializing, MeasuredPages>;

impl<'a, T: GuestStagePagingMode> MeasuredPagesMapper<'a, T> {
    /// Maps a page into the guest's address space and measures it.
    pub fn map_page<S, M, D, H>(
        &self,
        to_addr: GuestPageAddr,
        page: Page<S>,
        measurement: &AttestationManager<D, H>,
    ) -> Result<()>
    where
        S: Mappable<M>,
        M: MeasureRequirement,
        D: digest::Digest,
        H: hkdf::HmacImpl<D>,
    {
        measurement
            .extend_tvm_page(page.as_bytes(), to_addr.bits())
            .map_err(Error::Measurement)?;
        self.do_map_page(to_addr, page)
    }
}

pub enum SharedPages {}
/// A `VmPagesMapper` for shared (non-confidential) pages.
pub type SharedPagesMapper<'a, T, S> = VmPagesMapper<'a, T, S, SharedPages>;

impl<'a, T: GuestStagePagingMode, S> SharedPagesMapper<'a, T, S> {
    /// Maps a shared page into the guest's address space.
    pub fn map_page(&self, to_addr: GuestPageAddr, page: Page<MappableShared>) -> Result<()> {
        self.do_map_page(to_addr, page)
    }
}

pub enum ImsicPages {}
/// A `VmPagesMapper` for IMSIC guest file pages.
pub type ImsicPagesMapper<'a, T, S> = VmPagesMapper<'a, T, S, ImsicPages>;

impl<'a, T: GuestStagePagingMode, S> ImsicPagesMapper<'a, T, S> {
    /// Maps an IMSIC page into the guest's address space, also updating the MSI page tables if
    /// necessary.
    pub fn map_page(
        &self,
        to_addr: GuestPageAddr,
        page: ImsicGuestPage<MappableClean>,
    ) -> Result<()> {
        let dest_location = page.location();
        self.do_map_page(to_addr, page)?;
        if let Some(geometry) = self.vm_pages.imsic_geometry() &&
            let Some(iommu_context) = self.vm_pages.iommu_context.get()
        {
            let src_location = geometry
                .addr_to_location(to_addr)
                .ok_or(Error::InvalidImsicLocation)?;
            iommu_context.msi_page_table
                .map(src_location, dest_location)
                .map_err(Error::MsiTableMapping)?;
        }
        Ok(())
    }
}

pub enum PciPages {}
/// A `VmPagesMapper` for PCI BAR memory pages.
pub type PciPagesMapper<'a, T, S> = VmPagesMapper<'a, T, S, PciPages>;

impl<'a, T: GuestStagePagingMode, S> PciPagesMapper<'a, T, S> {
    /// Maps a PCI BAR memory page into the guest's address space.
    pub fn map_page(&self, to_addr: GuestPageAddr, page: PciBarPage<MappableClean>) -> Result<()> {
        self.do_map_page(to_addr, page)
    }
}

/// The possible sources of a guest page fault.
#[derive(Clone, Copy, Debug)]
pub enum PageFaultType {
    /// A page fault taken when accessing a confidential memory region. The host may handle these
    /// faults by inserting a confidential page into the guest's address space.
    Confidential(GuestPageAddr),
    /// A page fault taken when accessing a shared memory region. The host may handle these faults
    /// by inserting a page into the guest's address space.
    Shared(GuestPageAddr),
    /// A page fault taken to an emulated MMIO page.
    Mmio(GuestPhysAddr),
    /// A page fault taken when accessing memory outside of any valid region of guest physical address
    /// space. These faults are not resolvable.
    Unmapped(Exception),
}

/// Represents the active VM address space. Holds a reference to the TLB version of the address space
/// at the time the address space was activated. Used to directly access a guest's memory.
pub struct ActiveVmPages<'a, T: GuestStagePagingMode> {
    tlb_version: TlbVersion,
    vm_pages: &'a VmPages<T>,
}

impl<'a, T: GuestStagePagingMode> Drop for ActiveVmPages<'a, T> {
    fn drop(&mut self) {
        // Unwrap ok since tlb_tracker won't increment the version while there are outstanding
        // references.
        self.vm_pages
            .tlb_tracker
            .put_version(self.tlb_version)
            .unwrap();
    }
}

impl<'a, T: GuestStagePagingMode> ActiveVmPages<'a, T> {
    fn new(vm_pages: &'a VmPages<T>, vmid: VmId, prev_tlb_version: Option<TlbVersion>) -> Self {
        let mut hgatp = LocalRegisterCopy::<u64, hgatp::Register>::new(0);
        hgatp.modify(hgatp::vmid.val(vmid.vmid()));
        hgatp.modify(hgatp::ppn.val(Pfn::from(vm_pages.root_address()).bits()));
        hgatp.modify(hgatp::mode.val(T::HGATP_VALUE));
        CSR.hgatp.set(hgatp.get());

        let tlb_version = vm_pages.tlb_tracker.get_version();
        // Fence if this VMID was previously running on this CPU with an old TLB version.
        if let Some(v) = prev_tlb_version && v < tlb_version {
            // We flush all translations for this VMID since we don't have an efficient way to
            // track which pages need fencing. Even if we let the user provide the range to be
            // fenced it would require reverse-mapping the address so we could update that it was
            // fenced in PageTracker, which is almost certainly more expensive than just flushing
            // everything to begin with.
            //
            // TODO: Keep a list of pages in VmPages that are pending conversion so that we can
            // do more fine-grained invalidation.
            tlb::hfence_gvma(None, Some(vmid.vmid()));
        }

        Self {
            tlb_version,
            vm_pages,
        }
    }

    /// Returns the TLB version at which this ActiveVmPages was entered.
    pub fn tlb_version(&self) -> TlbVersion {
        self.tlb_version
    }

    /// Copies from `src` to the guest physical address in `dest`. Returns an error if a fault was
    /// encountered while copying.
    pub fn copy_to_guest(&self, dest: GuestPhysAddr, src: &[u8]) -> Result<()> {
        // Need to disable any translation in VSATP since we're dealing with guest physical addresses.
        let old_vsatp = CSR.vsatp.atomic_replace(0);
        // Safety: _copy_to_guest internally detects and handles an invalid guest physical
        // address in `dest`.
        let bytes = unsafe { _copy_to_guest(dest.bits(), src.as_ptr(), src.len()) };
        CSR.vsatp.set(old_vsatp);
        if bytes == src.len() {
            Ok(())
        } else {
            let fault_addr = dest
                .checked_increment(bytes as u64)
                .ok_or(Error::AddressOverflow)?;
            Err(Error::PageFault(self.get_page_fault_cause(
                Exception::GuestStorePageFault,
                fault_addr,
            )))
        }
    }

    /// Copies from the guest physical address in `src` to `dest`. Returns an error if a fault was
    /// encountered while copying.
    pub fn copy_from_guest(&self, dest: &mut [u8], src: GuestPhysAddr) -> Result<()> {
        // Need to disable any translation in VSATP since we're dealing with guest physical addresses.
        let old_vsatp = CSR.vsatp.atomic_replace(0);
        // Safety: _copy_from_guest internally detects and handles an invalid guest physical address
        // in `src`.
        let bytes = unsafe { _copy_from_guest(dest.as_mut_ptr(), src.bits(), dest.len()) };
        CSR.vsatp.set(old_vsatp);
        if bytes == dest.len() {
            Ok(())
        } else {
            let fault_addr = src
                .checked_increment(bytes as u64)
                .ok_or(Error::AddressOverflow)?;
            Err(Error::PageFault(self.get_page_fault_cause(
                Exception::GuestLoadPageFault,
                fault_addr,
            )))
        }
    }

    /// Copies `count` pages from `src_addr` in the current guest to the converted pages starting at
    /// `from_addr`. The pages are then mapped into the child's address space at `to_addr`.
    pub fn copy_and_add_data_pages_builder<D: digest::Digest, H: hkdf::HmacImpl<D>>(
        &self,
        src_addr: GuestPageAddr,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, VmStateInitializing>,
        to_addr: GuestPageAddr,
        measurement: &AttestationManager<D, H>,
    ) -> Result<u64> {
        let converted_pages = self.vm_pages.get_converted_pages(from_addr, count)?;
        let mapper = to.map_measured_pages(to_addr, count)?;

        // Make sure we can initialize the full set of pages before mapping them.
        let page_tracker = self.vm_pages.page_tracker.clone();
        let mut initialized_pages = LockedPageList::new(page_tracker.clone());
        for (dirty, src_addr) in converted_pages.zip(src_addr.iter_from()) {
            match dirty.try_initialize(|bytes| self.copy_from_guest(bytes, src_addr.into())) {
                Ok(p) => initialized_pages.push(p).unwrap(),
                Err((e, p)) => {
                    // Unwrap ok since the page must have been locked.
                    page_tracker.unlock_page(p).unwrap();
                    return Err(e);
                }
            };
        }

        // Now map & measure the pages.
        let new_owner = to.page_owner_id();
        for (initialized, to_addr) in initialized_pages.zip(to_addr.iter_from()) {
            // Unwrap ok since we've guaranteed there's space for another owner.
            let mappable = page_tracker
                .assign_page_for_mapping(initialized, new_owner)
                .unwrap();
            // Unwrap ok since the address is in range and we haven't mapped it yet.
            mapper.map_page(to_addr, mappable, measurement).unwrap();
        }

        Ok(count)
    }

    /// Fetches and decodes the instruction at `pc` in the guest's virtual address space.
    pub fn fetch_guest_instruction(
        &self,
        pc: GuestVirtAddr,
        priv_level: PrivilegeLevel,
    ) -> InstructionFetchResult {
        // Set SPVP to reflect the privilege level we took the trap in so that
        let old_hstatus = CSR.hstatus.get();
        let mut hstatus = LocalRegisterCopy::<u64, hstatus::Register>::new(old_hstatus);
        hstatus.modify(hstatus::spvp.val(priv_level as u64));
        CSR.hstatus.set(hstatus.get());

        let mut raw_inst = 0u32;
        // Safety: _fetch_guest_instruction internally detects and handles an invalid guest virtual
        // address in `pc' and will only write up to 4 bytes to `raw_inst`.
        let ret = unsafe { _fetch_guest_instruction(pc.bits(), &mut raw_inst) };
        CSR.hstatus.set(old_hstatus);
        if ret < 0 {
            return Err(InstructionFetchError::FetchFault);
        }

        DecodedInstruction::from_raw(raw_inst)
            .map_err(|_| InstructionFetchError::FailedDecode(raw_inst))
    }

    /// Returns the cause of a guest page fault of type `exception` taken at `fault_addr` from this VM.
    pub fn get_page_fault_cause(
        &self,
        exception: Exception,
        fault_addr: GuestPhysAddr,
    ) -> PageFaultType {
        use PageFaultType::*;
        match self.vm_pages.regions.find(fault_addr) {
            // Mask off the page offset for confidential and shared faults to avoid revealing more
            // information than necessary to the host.
            Some(VmRegionType::Confidential) => {
                Confidential(PageAddr::with_round_down(fault_addr, PageSize::Size4k))
            }
            Some(VmRegionType::Shared) => {
                Shared(PageAddr::with_round_down(fault_addr, PageSize::Size4k))
            }
            Some(VmRegionType::Mmio) => match exception {
                Exception::GuestLoadPageFault | Exception::GuestStorePageFault => Mmio(fault_addr),
                _ => Unmapped(exception),
            },
            // TODO: Faults in an IMSIC region should report a separate fault type so that the host
            // can "swap in" a vCPU currently using an MRIF.
            Some(VmRegionType::Imsic) => Unmapped(exception),
            _ => Unmapped(exception),
        }
    }
}

/// A pool of page-table pages for a VM. Left over pages are released when the pool is dropped.
struct PtePagePool {
    pages: Mutex<PageList<Page<InternalClean>>>,
}

impl PtePagePool {
    /// Creates an empty `PtePagePool`.
    fn new(page_tracker: PageTracker) -> Self {
        Self {
            pages: Mutex::new(PageList::new(page_tracker)),
        }
    }

    /// Adds `page` to the pool.
    fn push(&self, page: Page<InternalClean>) {
        // Unwrap ok, we must uniquely own the page and it isn't on another list.
        self.pages.lock().push(page).unwrap();
    }

    /// Pops a page from the pool, if any are present.
    fn pop(&self) -> Option<Page<InternalClean>> {
        self.pages.lock().pop()
    }
}

impl Drop for PtePagePool {
    fn drop(&mut self) {
        let pages = self.pages.get_mut();
        let page_tracker = pages.page_tracker();
        for p in pages {
            // Unwrap ok, the page was assigned to us so we must be able to release it.
            page_tracker.release_page(p).unwrap();
        }
    }
}

/// The IOMMU context for a VM.
pub struct VmIommuContext {
    msi_page_table: MsiPageTable,
    // Global soft-context ID. Released on `drop()`.
    gscid: GscId,
}

impl VmIommuContext {
    // Creates a new `VmIommuContext` using `msi_page_table`.
    fn new(msi_page_table: MsiPageTable) -> Result<Self> {
        let gscid = Iommu::get()
            .ok_or(Error::NoIommu)?
            .alloc_gscid(msi_page_table.owner())
            .map_err(Error::AllocatingGscId)?;
        Ok(Self {
            msi_page_table,
            gscid,
        })
    }
}

impl Drop for VmIommuContext {
    fn drop(&mut self) {
        // Unwrap ok: presence of an IOMMU is checked at creation time
        let iommu = Iommu::get().unwrap();

        // Detach any devices we own from the IOMMU.
        let owner = self.msi_page_table.owner();
        let pci = PcieRoot::get();
        for dev in pci.devices() {
            let mut dev = dev.lock();
            if dev.owner() == Some(owner) {
                // Unwrap ok: `self.gscid` must be valid and match the ownership of the device
                // to have been attached in the first place.
                //
                // Silence buggy clippy warning.
                #[allow(clippy::explicit_auto_deref)]
                iommu.detach_pci_device(&mut *dev, self.gscid).unwrap();
            }
        }

        // Unwrap ok: `self.gscid` must be valid and freeable since we've detached all devices
        // using it.
        iommu.free_gscid(self.gscid).unwrap();
    }
}

/// VmPages is the single management point for memory used by virtual machines.
///
/// After initial setup all memory not used for Hypervisor purposes is managed by a VmPages
/// instance. Rules around sharing and isolating memory are enforced by this module.
///
/// Machines are allowed to donate pages to child machines and to share donated pages with parent
/// machines.
pub struct VmPages<T: GuestStagePagingMode, S = VmStateFinalized> {
    page_owner_id: PageOwnerId,
    page_tracker: PageTracker,
    tlb_tracker: TlbTracker,
    regions: VmRegionList,
    // How many nested TVMs deep this VM is, with 0 being the host.
    nesting: usize,
    root: GuestStagePageTable<T>,
    pte_pages: PtePagePool,
    imsic_geometry: Once<GuestImsicGeometry>,
    iommu_context: Once<VmIommuContext>,
    phantom: PhantomData<S>,
}

impl<T: GuestStagePagingMode, S> VmPages<T, S> {
    /// Returns the `PageOwnerId` associated with the pages contained in this machine.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.page_owner_id
    }

    /// Returns the address of the root page table for this VM.
    pub fn root_address(&self) -> SupervisorPageAddr {
        // TODO: Cache this to avoid bouncing off the lock?
        self.root.get_root_address()
    }

    /// Returns the global page tracking structure.
    pub fn page_tracker(&self) -> PageTracker {
        self.page_tracker.clone()
    }

    /// Returns this VM's IMSIC geometry if it was set up for IMSIC virtualization.
    pub fn imsic_geometry(&self) -> Option<GuestImsicGeometry> {
        self.imsic_geometry.get().cloned()
    }

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&self, page: Page<InternalClean>) -> Result<()> {
        if page.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(page.size()));
        }
        self.pte_pages.push(page);
        Ok(())
    }

    fn do_map_pages<M>(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
        region_type: VmRegionType,
    ) -> Result<VmPagesMapper<T, S, M>> {
        VmPagesMapper::new_in_region(self, page_addr, count, region_type)
    }

    /// Locks `count` 4kB pages starting at `page_addr` for mapping of zero-filled pages in a
    /// region of confidential memory, returning a `VmPagesMapper` that can be used to insert
    /// the pages.
    pub fn map_zero_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<ZeroPagesMapper<T, S>> {
        self.do_map_pages(page_addr, count, VmRegionType::Confidential)
    }

    /// Same as `map_zero_pages()`, but for pages in shared (non-confidential) regions.
    pub fn map_shared_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<SharedPagesMapper<T, S>> {
        self.do_map_pages(page_addr, count, VmRegionType::Shared)
    }

    /// Same as `map_zero_pages()`, but for IMSIC guest interrupt file pages.
    pub fn map_imsic_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<ImsicPagesMapper<T, S>> {
        self.do_map_pages(page_addr, count, VmRegionType::Imsic)
    }

    /// Same as `map_zero_pages()`, but for PCI BAR memory pages.
    pub fn map_pci_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<PciPagesMapper<T, S>> {
        self.do_map_pages(page_addr, count, VmRegionType::Pci)
    }
}

impl<T: GuestStagePagingMode> VmPages<T, VmStateFinalized> {
    /// Returns a list of converted and locked pages created from `num_pages` starting at `page_addr`.
    fn get_converted_pages(
        &self,
        page_addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<LockedPageList<Page<ConvertedDirty>>> {
        let version = self.tlb_tracker.current();
        self.root
            .get_converted_range::<Page<ConvertedDirty>>(
                page_addr,
                PageSize::Size4k,
                num_pages,
                version,
            )
            .map_err(Error::Paging)
    }

    /// Converts `num_pages` starting at guest physical address `page_addr` to confidential memory.
    pub fn convert_pages(&self, page_addr: GuestPageAddr, num_pages: u64) -> Result<()> {
        if self.nesting >= MAX_PAGE_OWNERS - 1 {
            // We shouldn't bother converting pages if we won't be able to assign them.
            return Err(Error::NestingTooDeep);
        }

        let invalidated_pages = self
            .root
            .invalidate_range::<Page<Invalidated>>(page_addr, PageSize::Size4k, num_pages)
            .map_err(Error::Paging)?;
        let version = self.tlb_tracker.current();
        for page in invalidated_pages {
            // Unwrap ok since the page was just invalidated.
            self.page_tracker.convert_page(page, version).unwrap();
        }
        Ok(())
    }

    /// Reclaims `num_pages` of confidential memory starting at guest physical address `page_addr`.
    pub fn reclaim_pages(&self, page_addr: GuestPageAddr, num_pages: u64) -> Result<()> {
        // TODO: Support reclaim of converted pages that haven't yet been fenced.
        let converted_pages = self.get_converted_pages(page_addr, num_pages)?;
        // Unwrap ok since the PTE for the page must have previously been invalid and all of
        // the intermediate page-tables must already have been populatd.
        let mapper = self.map_zero_pages(page_addr, num_pages).unwrap();
        for (page, addr) in converted_pages.zip(page_addr.iter_from()) {
            // Unwrap ok since we know that it's a converted page.
            let mappable = self.page_tracker.reclaim_page(page.clean()).unwrap();
            mapper.map_page(addr, mappable).unwrap();
        }
        Ok(())
    }

    /// Converts the guest interrupt file at `imsic_addr` to confidential.
    pub fn convert_imsic(&self, imsic_addr: GuestPageAddr) -> Result<()> {
        if self.nesting >= MAX_PAGE_OWNERS - 1 {
            // We shouldn't bother converting pages if we won't be able to assign them.
            return Err(Error::NestingTooDeep);
        }

        // Make sure it's actually an IMSIC address and that it's a guest (not supervisor) file.
        let geometry = self
            .imsic_geometry
            .get()
            .ok_or(Error::NoImsicVirtualization)?;
        let location = geometry
            .addr_to_location(imsic_addr)
            .ok_or(Error::InvalidImsicLocation)?;
        if location.file() == ImsicFileId::Supervisor {
            return Err(Error::InvalidImsicLocation);
        }

        let mut invalidated = self
            .root
            .invalidate_range::<ImsicGuestPage<Invalidated>>(imsic_addr, PageSize::Size4k, 1)
            .map_err(Error::Paging)?;
        // Unwrap ok since the page was just invalidated.
        invalidated
            .next()
            .and_then(|p| {
                self.page_tracker
                    .convert_page(p, self.tlb_tracker.current())
                    .ok()
            })
            .unwrap();

        // Unmap it from our MSI page table as well, if we have one.
        if let Some(iommu_context) = self.iommu_context.get() {
            // Unwrap ok: we've already checked that `location` is valid and it must've been mapped
            // in the MSI page table if it was in the CPU page tables.
            iommu_context.msi_page_table.unmap(location).unwrap();
        }

        Ok(())
    }

    /// Reclaims the confidential guest interrupt file at `imsic_addr`.
    pub fn reclaim_imsic(&self, imsic_addr: GuestPageAddr) -> Result<()> {
        let mut converted = self
            .root
            .get_converted_range::<ImsicGuestPage<ConvertedClean>>(
                imsic_addr,
                PageSize::Size4k,
                1,
                self.tlb_tracker.current(),
            )
            .map_err(Error::Paging)?;
        // Unwrap ok since the PTE for the page must have previously been invalid and all of
        // the intermediate page-tables must already have been populated.
        let mapper = self.map_imsic_pages(imsic_addr, 1).unwrap();
        // Unwrap ok since it must be a converted page.
        let mappable = converted
            .next()
            .and_then(|p| self.page_tracker.reclaim_page(p).ok())
            .unwrap();
        // Unwrap ok since `imsic_addr` is within the range of the mapper.
        mapper.map_page(imsic_addr, mappable).unwrap();
        Ok(())
    }

    /// Initiates a page conversion fence for this `VmPages` by incrementing the TLB version.
    pub fn initiate_fence(&self) -> Result<()> {
        self.tlb_tracker.increment()?;
        // If we have an IOMMU context then we need to issue a fence there as well as our page
        // tables may be used for DMA translation.
        if let Some(iommu_context) = self.iommu_context.get() {
            // Unwrap ok since we must have an IOMMU to have a `VmIommuContext`.
            Iommu::get().unwrap().fence(iommu_context.gscid, None);
        }
        Ok(())
    }

    /// Assigns the converted pages in `pages` to `new_owner` as state pages.
    fn assign_state_pages_for(
        &self,
        pages: LockedPageList<Page<ConvertedDirty>>,
        new_owner: PageOwnerId,
    ) -> PageList<Page<InternalClean>> {
        let mut assigned_pages = PageList::new(self.page_tracker.clone());
        for page in pages {
            // Unwrap ok since we've guaranteed there is space for another owner.
            let assigned = self
                .page_tracker
                .assign_page_for_internal_state(page.clean(), new_owner)
                .unwrap();
            // Unwrap ok since we uniquely own the page and it can't be on another list.
            assigned_pages.push(assigned).unwrap();
        }
        assigned_pages
    }

    /// Creates a new `Vm` using pages donated by `self`. The returned `Vm` is in the initializing
    /// state, ready for its address space to be constructed.
    pub fn create_guest_vm(
        &self,
        page_root_addr: GuestPageAddr,
        state_addr: GuestPageAddr,
        vcpus_addr: GuestPageAddr,
        num_vcpu_pages: u64,
    ) -> Result<(Vm<T, VmStateInitializing>, Page<InternalClean>)> {
        if (page_root_addr.bits() as *const u64).align_offset(T::TOP_LEVEL_ALIGN as usize) != 0 {
            return Err(Error::UnalignedAddress);
        }

        // Make sure we can grab the pages first before we start wiping and assigning them.
        let guest_root_pages = self.get_converted_pages(page_root_addr, 4)?;
        if !guest_root_pages.is_contiguous() {
            return Err(Error::NonContiguousPages);
        }
        let state_pages = self.get_converted_pages(state_addr, TVM_STATE_PAGES)?;
        if !state_pages.is_contiguous() {
            return Err(Error::NonContiguousPages);
        }
        let vcpu_pages = self.get_converted_pages(vcpus_addr, num_vcpu_pages)?;
        if !vcpu_pages.is_contiguous() {
            return Err(Error::NonContiguousPages);
        }
        let id = self
            .page_tracker
            .add_active_guest()
            .map_err(Error::GuestId)?;

        let guest_root_pages =
            SequentialPages::from_pages(self.assign_state_pages_for(guest_root_pages, id)).unwrap();
        let guest_root =
            GuestStagePageTable::new(guest_root_pages, id, self.page_tracker.clone()).unwrap();

        let mut state_pages = self.assign_state_pages_for(state_pages, id);
        let box_page = state_pages.next().unwrap();
        let region_vec_pages = SequentialPages::from_pages(state_pages).unwrap();
        let region_vec = VmRegionList::new(region_vec_pages, self.page_tracker.clone());

        let vcpu_pages =
            SequentialPages::from_pages(self.assign_state_pages_for(vcpu_pages, id)).unwrap();

        Ok((
            Vm::new(
                VmPages::new(guest_root, region_vec, self.nesting + 1),
                VmCpus::new(id, vcpu_pages, self.page_tracker.clone()).unwrap(),
            )
            .map_err(Error::VmCreationFailed)?,
            box_page,
        ))
    }

    /// Adds pages to be used for building page table entries to a guest of this VM.
    pub fn add_pte_pages_to<S>(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, S>,
    ) -> Result<()> {
        let converted_pages = self.get_converted_pages(from_addr, count)?;
        let new_owner = to.page_owner_id();
        for page in converted_pages {
            // Unwrap ok since we've guaranteed the page is assignable.
            let assigned = self
                .page_tracker
                .assign_page_for_internal_state(page.clean(), new_owner)
                .unwrap();
            // Unwrap ok, pages must be 4kB.
            to.add_pte_page(assigned).unwrap();
        }
        Ok(())
    }

    /// Adds zero-filled pages to the given guest.
    pub fn add_zero_pages_to<S>(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, S>,
        to_addr: GuestPageAddr,
    ) -> Result<u64> {
        let converted_pages = self.get_converted_pages(from_addr, count)?;
        let mapper = to.map_zero_pages(to_addr, count)?;
        let new_owner = to.page_owner_id();
        for (page, guest_addr) in converted_pages.zip(to_addr.iter_from()) {
            // Unwrap ok since we've guaranteed there's space for another owner.
            let mappable = self
                .page_tracker
                .assign_page_for_mapping(page.clean(), new_owner)
                .unwrap();
            // Unwrap ok since the address is in range and we haven't mapped it yet.
            mapper.map_page(guest_addr, mappable).unwrap();
        }
        Ok(count)
    }

    /// Maps num_pages of shared 4Kb pages starting at `from_addr` to the specified guest. The
    /// range must fit in a range declared by a call to `add_shared_memory_region`.
    pub fn add_shared_pages_to<S>(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, S>,
        to_addr: GuestPageAddr,
    ) -> Result<()> {
        let shared_list = self
            .root
            .get_shareable_range::<Page<Shareable>>(from_addr, PageSize::Size4k, count)
            .map_err(|_| Error::SharedPageNotMapped)?;
        let mapper = to.map_shared_pages(to_addr, count)?;
        let owner = self.page_owner_id();
        for (page, addr) in shared_list.zip(to_addr.iter_from()) {
            // Unwrap ok: we have exclusive ownership, and get_shareable_range() has ensured success
            let mappable = self.page_tracker.share_page(page, owner).unwrap();
            // Unwrap ok: we have exclusive ownership and have already filled in the PTE.
            mapper.map_page(addr, mappable).unwrap();
        }
        Ok(())
    }

    /// Activates the address space represented by this `VmPages`. The reference to the address space
    /// is dropped when the returned `ActiveVmPages` is dropped. Flushes TLB entries for the given
    /// VMID if it was previously active with a stale TLB version on this CPU.
    ///
    /// The caller must ensure that VMID has been allocated to reference this address space on this
    /// CPU and that there are no stale translations tagged with VMID referencing other VM address
    /// spaces in this CPU's TLB.
    pub fn enter_with_vmid(
        &self,
        vmid: VmId,
        prev_tlb_version: Option<TlbVersion>,
    ) -> ActiveVmPages<T> {
        ActiveVmPages::new(self, vmid, prev_tlb_version)
    }
}

impl<T: GuestStagePagingMode> VmPages<T, VmStateInitializing> {
    /// Creates a new `VmPages` from the given root page table.
    pub fn new(root: GuestStagePageTable<T>, regions: VmRegionList, nesting: usize) -> Self {
        let page_tracker = root.page_tracker();
        Self {
            page_owner_id: root.page_owner_id(),
            page_tracker: page_tracker.clone(),
            tlb_tracker: TlbTracker::new(),
            regions,
            nesting,
            root,
            pte_pages: PtePagePool::new(page_tracker),
            imsic_geometry: Once::new(),
            iommu_context: Once::new(),
            phantom: PhantomData,
        }
    }

    /// Sets the IMSIC geometry for this VM by adding the memory regions that will be occupied by
    /// IMSIC interrupt files.
    pub fn set_imsic_geometry(&self, geometry: GuestImsicGeometry) -> Result<()> {
        let to_set = geometry.clone();
        let actual = self
            .imsic_geometry
            .try_call_once(|| {
                for range in geometry.group_ranges() {
                    let end = range
                        .base()
                        .checked_add_pages(range.num_pages())
                        .ok_or(Error::AddressOverflow)?;
                    self.regions.add(range.base(), end, VmRegionType::Imsic)?;
                }
                Ok(geometry)
            })?
            .clone();
        // Check if the IMSIC geometry we specified was the one that was actually set in
        // `try_call_once()`; we may have raced with another thread.
        if to_set != actual {
            return Err(Error::ImsicGeometryAlreadySet);
        }
        Ok(())
    }

    /// Creates an IOMMU context for this VM using `msi_table_pages` as the backing pages for
    /// the MSI page table.
    pub fn add_iommu_context(&self, msi_table_pages: SequentialPages<InternalClean>) -> Result<()> {
        // No point having an IOMMU context if we aren't doing IMSIC virtualization.
        let imsic_geometry = self
            .imsic_geometry
            .get()
            .ok_or(Error::NoImsicVirtualization)?
            .clone();
        let msi_pt = MsiPageTable::new(
            msi_table_pages,
            imsic_geometry,
            Imsic::get().phys_geometry(),
            self.page_tracker.clone(),
            self.page_owner_id,
        )
        .map_err(Error::CreatingMsiPageTable)?;
        let iommu_context = VmIommuContext::new(msi_pt)?;
        let gscid = iommu_context.gscid;
        let set_gscid = self.iommu_context.call_once(|| iommu_context).gscid;
        // Check if the `VmIommuContext` that was set was actually the one we created.
        if gscid != set_gscid {
            return Err(Error::IommuContextAlreadySet);
        }
        Ok(())
    }

    // Adds a region of type `region_type`.
    fn do_add_region(
        &self,
        page_addr: GuestPageAddr,
        len: u64,
        region_type: VmRegionType,
    ) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;
        self.regions.add(page_addr, end, region_type)
    }

    /// Adds a confidential memory region of `len` bytes starting at `page_addr` to this VM's
    /// address space.
    pub fn add_confidential_memory_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_add_region(page_addr, len, VmRegionType::Confidential)
    }

    /// Adds a shared memory region of `len` bytes starting at `page_addr` to this VM's address
    /// space.
    pub fn add_shared_memory_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_add_region(page_addr, len, VmRegionType::Shared)
    }

    /// Adds an emulated MMIO region of `len` bytes starting at `page_addr` to this VM's address
    /// space.
    pub fn add_mmio_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_add_region(page_addr, len, VmRegionType::Mmio)
    }

    /// Adds a PCI BAR memory region of `len` bytes starting at `page_addr` to this VM's address
    /// space.
    pub fn add_pci_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_add_region(page_addr, len, VmRegionType::Pci)
    }

    /// Like `map_zero_pages()`, but for measured pages mapped into a region of confidential
    /// memory.
    pub fn map_measured_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<MeasuredPagesMapper<T>> {
        self.do_map_pages(page_addr, count, VmRegionType::Confidential)
    }

    /// Attaches the given PCI device to this VM by enabling DMA translation via the IOMMU using
    /// this VM's page tables.
    pub fn attach_pci_device(&self, dev: &mut PciDevice) -> Result<()> {
        let iommu_context = self.iommu_context.get().ok_or(Error::NoIommu)?;
        Iommu::get()
            .unwrap()
            .attach_pci_device(
                dev,
                &self.root,
                &iommu_context.msi_page_table,
                iommu_context.gscid,
            )
            .map_err(Error::AttachingDevice)
    }

    /// Consumes this `VmPages`, returning a finalized one.
    pub fn finalize(self) -> VmPages<T, VmStateFinalized> {
        VmPages {
            page_owner_id: self.page_owner_id,
            page_tracker: self.page_tracker,
            tlb_tracker: self.tlb_tracker,
            regions: self.regions,
            nesting: self.nesting,
            root: self.root,
            pte_pages: self.pte_pages,
            imsic_geometry: self.imsic_geometry,
            iommu_context: self.iommu_context,
            phantom: PhantomData,
        }
    }
}
