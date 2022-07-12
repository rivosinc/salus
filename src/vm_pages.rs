// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use attestation::measurement::{AttestationManager, MeasurementIndex};
use core::arch::global_asm;
use core::{marker::PhantomData, ops::Deref};
use data_measure::data_measure::DataMeasure;
use data_measure::sha256::Sha256Measure;
use page_tracking::collections::PageVec;
use page_tracking::{
    LockedPageList, PageList, PageTracker, PageTrackingError, TlbVersion, MAX_PAGE_OWNERS,
};
use riscv_page_tables::{
    tlb, GuestStagePageTable, PageTableError, PageTableMapper, PlatformPageTable,
};
use riscv_pages::*;
use riscv_regs::{
    hgatp, hstatus, DecodedInstruction, Exception, LocalRegisterCopy, PrivilegeLevel, Readable,
    Writeable, CSR,
};
use spin::Mutex;

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
    MeasurementBufferTooSmall,
    AddressOverflow,
    TlbCountUnderflow,
    InvalidTlbVersion,
    TlbFenceInProgress,
    OverlappingVmRegion,
    InsufficientVmRegionSpace,
    InvalidMapRegion,
    SharedPageNotMapped,
    Measurement(attestation::Error),
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

/// Wrapper for a `PageTableMapper` created from the page table of `VmPages`. Measures pages as
/// they are inserted, if necessary.
pub struct VmPagesMapper<'a, T: GuestStagePageTable, S> {
    mapper: PageTableMapper<'a, T>,
    phantom: PhantomData<S>,
}

impl<'a, T: GuestStagePageTable, S> VmPagesMapper<'a, T, S> {
    /// Creates a new `VmPagesMapper` for `num_pages` starting at `page_addr`, which must lie within
    /// a region of type `region_type`.
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
            mapper,
            phantom: PhantomData,
        })
    }

    /// Maps an unmeasured page into the guest's address space.
    pub fn map_page<P>(&self, to_addr: GuestPageAddr, page: P) -> Result<()>
    where
        P: MappablePhysPage<MeasureOptional>,
    {
        self.mapper.map_page(to_addr, page).map_err(Error::Paging)
    }
}

impl<'a, T: GuestStagePageTable> VmPagesMapper<'a, T, VmStateInitializing> {
    /// Maps a page into the guest's address space and measures it.
    pub fn map_page_with_measurement<S, M, D>(
        &self,
        to_addr: GuestPageAddr,
        page: Page<S>,
        measurement: &AttestationManager<D>,
    ) -> Result<()>
    where
        S: Mappable<M>,
        M: MeasureRequirement,
        D: digest::Digest,
    {
        measurement
            .extend_msmt_register(
                MeasurementIndex::TvmPage,
                page.as_bytes(),
                Some(to_addr.bits()),
            )
            .map_err(Error::Measurement)?;
        self.mapper.map_page(to_addr, page).map_err(Error::Paging)
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

/// Represents a reference to the current VM address space. The previous address space is restored
/// when dropped. Used to directly access a guest's memory.
pub struct ActiveVmPages<'a, T: GuestStagePageTable> {
    prev_hgatp: u64,
    tlb_version: TlbVersion,
    vm_pages: &'a VmPages<T>,
}

impl<'a, T: GuestStagePageTable> Drop for ActiveVmPages<'a, T> {
    fn drop(&mut self) {
        CSR.hgatp.set(self.prev_hgatp);

        // Unwrap ok since tlb_tracker won't increment the version while there are outstanding
        // references.
        self.vm_pages
            .tlb_tracker
            .put_version(self.tlb_version)
            .unwrap();
    }
}

impl<'a, T: GuestStagePageTable> Deref for ActiveVmPages<'a, T> {
    type Target = VmPages<T>;

    fn deref(&self) -> &VmPages<T> {
        self.vm_pages
    }
}

impl<'a, T: GuestStagePageTable> ActiveVmPages<'a, T> {
    fn new(vm_pages: &'a VmPages<T>, vmid: VmId, prev_tlb_version: Option<TlbVersion>) -> Self {
        let mut hgatp = LocalRegisterCopy::<u64, hgatp::Register>::new(0);
        hgatp.modify(hgatp::vmid.val(vmid.vmid()));
        hgatp.modify(hgatp::ppn.val(Pfn::from(vm_pages.root_address()).bits()));
        hgatp.modify(hgatp::mode.val(T::HGATP_VALUE));
        let prev_hgatp = CSR.hgatp.atomic_replace(hgatp.get());

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
            prev_hgatp,
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
    pub fn copy_and_add_data_pages_builder<D: digest::Digest>(
        &self,
        src_addr: GuestPageAddr,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, VmStateInitializing>,
        to_addr: GuestPageAddr,
        measurement: &AttestationManager<D>,
    ) -> Result<u64> {
        let converted_pages = self.get_converted_pages(from_addr, count)?;
        let mapper = to.map_pages(to_addr, count)?;

        // Make sure we can initialize the full set of pages before mapping them.
        let mut initialized_pages = LockedPageList::new(self.page_tracker.clone());
        for (dirty, src_addr) in converted_pages.zip(src_addr.iter_from()) {
            match dirty.try_initialize(|bytes| self.copy_from_guest(bytes, src_addr.into())) {
                Ok(p) => initialized_pages.push(p).unwrap(),
                Err((e, p)) => {
                    // Unwrap ok since the page must have been locked.
                    self.page_tracker.unlock_page(p).unwrap();
                    return Err(e);
                }
            };
        }

        // Now map & measure the pages.
        let new_owner = to.page_owner_id();
        for (initialized, to_addr) in initialized_pages.zip(to_addr.iter_from()) {
            // Unwrap ok since we've guaranteed there's space for another owner.
            let mappable = self
                .page_tracker
                .assign_page_for_mapping(initialized, new_owner)
                .unwrap();
            // Unwrap ok since the address is in range and we haven't mapped it yet.
            mapper
                .map_page_with_measurement(to_addr, mappable, measurement)
                .unwrap();
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
        match self.regions.find(fault_addr) {
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
            None => Unmapped(exception),
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

/// VmPages is the single management point for memory used by virtual machines.
///
/// After initial setup all memory not used for Hypervisor purposes is managed by a VmPages
/// instance. Rules around sharing and isolating memory are enforced by this module.
///
/// Machines are allowed to donate pages to child machines and to share donated pages with parent
/// machines.
pub struct VmPages<T: GuestStagePageTable, S = VmStateFinalized> {
    page_owner_id: PageOwnerId,
    page_tracker: PageTracker,
    tlb_tracker: TlbTracker,
    regions: VmRegionList,
    // How many nested TVMs deep this VM is, with 0 being the host.
    nesting: usize,
    root: PlatformPageTable<T>,
    measurement: Mutex<Sha256Measure>,
    pte_pages: PtePagePool,
    phantom: PhantomData<S>,
}

impl<T: GuestStagePageTable, S> VmPages<T, S> {
    /// Returns the `PageOwnerId` associated with the pages contained in this machine.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.page_owner_id
    }

    /// Copies the measurement for this guest into `dest`.
    pub fn get_measurement(&self, dest: &mut [u8]) -> Result<()> {
        let measurement = self.measurement.lock();
        let src = measurement.get_measurement();
        if src.len() > dest.len() {
            return Err(Error::MeasurementBufferTooSmall);
        }
        let (left, _) = dest.split_at_mut(src.len());
        left.copy_from_slice(src);
        Ok(())
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

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&self, page: Page<InternalClean>) -> Result<()> {
        if page.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(page.size()));
        }
        self.pte_pages.push(page);
        Ok(())
    }

    /// Locks `count` 4kB pages starting at `page_addr` for mapping, returning a `VmPagesMapper` that
    /// can be used to insert (and measure, if necessary) the pages.
    pub fn map_pages(&self, page_addr: GuestPageAddr, count: u64) -> Result<VmPagesMapper<T, S>> {
        VmPagesMapper::new_in_region(self, page_addr, count, VmRegionType::Confidential)
    }

    /// Maps num_pages of shared 4Kb pages starting at page_addr. The range must fit in
    /// a range declared by a call to `add_shared_memory_region`.
    pub fn add_shared_pages(
        &self,
        from_addr: GuestPageAddr,
        num_pages: u64,
        from: &VmPages<T, VmStateFinalized>,
        guest_addr: GuestPageAddr,
    ) -> Result<()> {
        let shared_list = from
            .root
            .get_shareable_range::<Page<Shareable>>(from_addr, PageSize::Size4k, num_pages)
            .map_err(|_| Error::SharedPageNotMapped)?;
        let mapper =
            VmPagesMapper::new_in_region(self, guest_addr, num_pages, VmRegionType::Shared)?;
        let owner = from.root.page_owner_id();
        for (page, addr) in shared_list.zip(guest_addr.iter_from()) {
            // Unwrap ok: we have exclusve ownership, and get_shareable_range() has ensured success
            let mappable = self.page_tracker.share_page(page, owner).unwrap();
            // Unwrap ok: we have exclusive ownership and have already filled in the PTE.
            mapper.map_page(addr, mappable).unwrap();
        }
        Ok(())
    }
}

impl<T: GuestStagePageTable> VmPages<T, VmStateFinalized> {
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
        let mapper =
            VmPagesMapper::new_in_region(self, page_addr, num_pages, VmRegionType::Confidential)
                .unwrap();
        for (page, addr) in converted_pages.zip(page_addr.iter_from()) {
            // Unwrap ok since we know that it's a converted page.
            let mappable = self.page_tracker.reclaim_page(page.clean()).unwrap();
            mapper.map_page(addr, mappable).unwrap();
        }
        Ok(())
    }

    /// Initiates a page conversion fence for this `VmPages` by incrementing the TLB version.
    pub fn initiate_fence(&self) -> Result<()> {
        self.tlb_tracker.increment()
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
            PlatformPageTable::new(guest_root_pages, id, self.page_tracker.clone()).unwrap();

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
            ),
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
        let mapper = to.map_pages(to_addr, count)?;
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

    /// Activates the address space represented by this `VmPages`. The address space is exited (and
    /// the previous one restored) when the returned `ActiveVmPages` is dropped. Flushes TLB entries
    /// for the given VMID if it was previously active with a stale TLB version on this CPU.
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

impl<T: GuestStagePageTable> VmPages<T, VmStateInitializing> {
    /// Creates a new `VmPages` from the given root page table.
    pub fn new(root: PlatformPageTable<T>, regions: VmRegionList, nesting: usize) -> Self {
        let page_tracker = root.page_tracker();
        Self {
            page_owner_id: root.page_owner_id(),
            page_tracker: page_tracker.clone(),
            tlb_tracker: TlbTracker::new(),
            regions,
            nesting,
            root,
            measurement: Mutex::new(Sha256Measure::new()),
            pte_pages: PtePagePool::new(page_tracker),
            phantom: PhantomData,
        }
    }

    /// Adds a confidential memory region of `len` bytes starting at `page_addr` to this VM's address space.
    pub fn add_confidential_memory_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;
        self.regions.add(page_addr, end, VmRegionType::Confidential)
    }

    /// Adds a shared memory region of `len` bytes starting at `page_addr` to this VM's address space.
    pub fn add_shared_memory_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;
        self.regions.add(page_addr, end, VmRegionType::Shared)
    }

    /// Adds an emulated MMIO region of `len` bytes starting at `page_addr` to this VM's address space.
    pub fn add_mmio_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;
        self.regions.add(page_addr, end, VmRegionType::Mmio)
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
            measurement: self.measurement,
            pte_pages: self.pte_pages,
            phantom: PhantomData,
        }
    }
}
