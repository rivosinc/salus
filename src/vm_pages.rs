// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

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
use riscv_regs::{hgatp, LocalRegisterCopy, Writeable, CSR};
use spin::Mutex;

use crate::smp::PerCpu;
use crate::vm::{Vm, VmStateFinalized, VmStateInitializing};
use crate::vm_cpu::VmCpus;
use crate::vm_id::VmId;

#[derive(Debug)]
pub enum Error {
    GuestId(PageTrackingError),
    Paging(PageTableError),
    UnhandledPageFault,
    NestingTooDeep,
    // Page table root must be aligned to 16k to be used for sv48x4 mappings
    UnalignedVmPages(GuestPageAddr),
    UnsupportedPageSize(PageSize),
    NonContiguousPages,
    MeasurementBufferTooSmall,
    AddressOverflow,
    TlbCountUnderflow,
    InvalidTlbVersion,
    TlbFenceInProgress,
    InvalidRegionType,
    OverlappingVmRegion,
    InsufficientVmRegionSpace,
}

pub type Result<T> = core::result::Result<T, Error>;

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
    /// Regions of private memory that have been fully populated during VM creation.
    PreMapped,
    /// Placeholder region type to reserve a range of guest physical address space while the region
    /// is mapped.
    Reserved,
}

/// A contiguous region of guest physical address space.
struct VmRegion {
    start: GuestPageAddr,
    end: GuestPageAddr,
    region_type: VmRegionType,
}

/// Represents a reserved region in `VmRegionList` that has yet to be committed. Removes the reserved
/// region if dropped without being passed to `VmRegionList::commit()`.
struct VmReservedRegion<'a> {
    owner: &'a VmRegionList,
    start: GuestPageAddr,
    region_type: VmRegionType,
}

impl<'a> VmReservedRegion<'a> {
    /// Creates a new reserved region holder in `owner` for the region beginning at `start` with
    /// type `region_type`.
    fn new(owner: &'a VmRegionList, start: GuestPageAddr, region_type: VmRegionType) -> Self {
        Self {
            owner,
            start,
            region_type,
        }
    }
}

impl<'a> Drop for VmReservedRegion<'a> {
    fn drop(&mut self) {
        let mut regions = self.owner.regions.lock();
        regions.retain(|r| r.start != self.start);
    }
}

/// The regions of guest physical address space for a VM. Used to track which parts of the address
/// space are designated for a particular purpose. The region list is created during VM initialization
/// and remains static after the VM is finalized.
///
/// Regions are added to the `VmRegionList` in two steps: first the region is reserved in the address
/// map with `prepare()` and then, after whatever work is necessary to populate the region is done,
/// the region is committed with `commit()`. This is done in order to guarantee that a region can
/// be added to the address map before more expensive operations are carried out (e.g. zeroing pages,
/// filling in pages tables) while allowing for easy error cleanup should a subsequent step fail.
///
/// Example usage:
///
/// ```rust,ignore
/// fn expensive_map_operation(start: GuestPageAddr, end: GuestPageAddr) -> Result<()> { ... }
///
/// fn add_region(regions: &VmRegionList, start: GuestPageAddr, end: GuestPageAddr) -> Result<()> {
///     let region = regions.prepare(start, end, VmRegionType::PreMapped)?;
///     expensive_map_operation(start, end)?;
///     regions.commit(region);
/// }
/// ```
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

    /// Reserves a region at [`start`, `end`) of type `region_type`, returning a `VmReservedRegion`
    /// object if it does not conflict with any existing regions. The `VmReservedRegion` should
    /// be passed to `commit()` to finalize addition of the region to this `VmRegionList`. If the
    /// `VmReservedRegion` is dropped before it is committed, the reservation is removed.
    fn prepare(
        &self,
        start: GuestPageAddr,
        end: GuestPageAddr,
        region_type: VmRegionType,
    ) -> Result<VmReservedRegion> {
        if region_type == VmRegionType::Reserved {
            return Err(Error::InvalidRegionType);
        }
        let mut regions = self.regions.lock();
        // Keep the list sorted, inserting a reserved region in the requested spot as long as it
        // doesn't overlap with anything else.
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
            region_type: VmRegionType::Reserved,
        };
        regions
            .try_reserve(1)
            .map_err(|_| Error::InsufficientVmRegionSpace)?;
        regions.insert(index, region);
        Ok(VmReservedRegion::new(self, start, region_type))
    }

    /// Commits `region` to this `VmRegionList`. Coalesces adjacent regions of the same type.
    fn commit(&self, region: VmReservedRegion) {
        let mut regions = self.regions.lock();
        // Unwrap ok, the reserved region must already be in the list.
        let (mut index, mut r) = regions
            .iter_mut()
            .enumerate()
            .find(|(_, r)| r.start == region.start)
            .unwrap();

        // Coalesce with same-typed regions before and after this one, if possible.
        let mut start = r.start;
        let end = r.end;
        let region_type = region.region_type;
        r.region_type = region_type;
        if let Some(ref mut before) = regions.get_mut(index - 1) {
            if before.end == start && before.region_type == region_type {
                before.end = end;
                start = before.start;
                regions.remove(index);
                index -= 1;
            }
        }
        if let Some(ref mut after) = regions.get_mut(index + 1) {
            if after.start == end && after.region_type == region_type {
                after.start = start;
                regions.remove(index);
            }
        }
        core::mem::forget(region);
    }
}

/// Wrapper for a `PageTableMapper` created from the page table of `VmPages`. Measures pages as
/// they are inserted, if necessary.
pub struct VmPagesMapper<'a, T: GuestStagePageTable, S> {
    mapper: PageTableMapper<'a, T>,
    region: Option<VmReservedRegion<'a>>,
    vm_pages: &'a VmPages<T, S>,
}

impl<'a, T: GuestStagePageTable, S> VmPagesMapper<'a, T, S> {
    /// Creates a new `VmPagesMapper` for `num_pages` starting at `page_addr`.
    fn new(vm_pages: &'a VmPages<T, S>, page_addr: GuestPageAddr, num_pages: u64) -> Result<Self> {
        let mapper = vm_pages
            .root
            .map_range(page_addr, PageSize::Size4k, num_pages, &mut || {
                vm_pages.pte_pages.pop()
            })
            .map_err(Error::Paging)?;
        Ok(Self {
            mapper,
            region: None,
            vm_pages,
        })
    }

    /// Creates a new `VmPagesMapper` for `num_pages` starting at `page_addr` in a new region of type
    /// `region_type`.
    fn with_new_region(
        vm_pages: &'a VmPages<T, S>,
        page_addr: GuestPageAddr,
        num_pages: u64,
        region_type: VmRegionType,
    ) -> Result<Self> {
        let end = page_addr
            .checked_add_pages(num_pages)
            .ok_or(Error::AddressOverflow)?;
        let region = vm_pages.regions.prepare(page_addr, end, region_type)?;
        let mapper = vm_pages
            .root
            .map_range(page_addr, PageSize::Size4k, num_pages, &mut || {
                vm_pages.pte_pages.pop()
            })
            .map_err(Error::Paging)?;
        Ok(Self {
            mapper,
            region: Some(region),
            vm_pages,
        })
    }

    /// Maps an unmeasured page into the guest's address space.
    pub fn map_page<P>(&self, to_addr: GuestPageAddr, page: P) -> Result<()>
    where
        P: MappablePhysPage<MeasureOptional>,
    {
        self.mapper.map_page(to_addr, page).map_err(Error::Paging)
    }

    /// Completes the mapping operation, finishing insertion of the new region if necessary.
    pub fn finish(self) {
        if let Some(region) = self.region {
            self.vm_pages.regions.commit(region);
        }
    }
}

impl<'a, T: GuestStagePageTable> VmPagesMapper<'a, T, VmStateInitializing> {
    /// Maps a page into the guest's address space and measures it.
    pub fn map_page_with_measurement<S, M>(
        &self,
        to_addr: GuestPageAddr,
        page: Page<S>,
    ) -> Result<()>
    where
        S: Mappable<M>,
        M: MeasureRequirement,
    {
        {
            let mut measurement = self.vm_pages.measurement.lock();
            measurement.add_page(to_addr.bits(), page.as_bytes());
        }
        self.mapper.map_page(to_addr, page).map_err(Error::Paging)
    }
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
        // Safety: _copy_to_guest internally detects and handles an invalid guest physical
        // address in `dest`.
        self.do_guest_copy(dest, src.as_ptr(), src.len(), |gpa, ptr, len| unsafe {
            _copy_to_guest(gpa.bits(), ptr, len)
        })
    }

    /// Copies from the guest physical address in `src` to `dest`. Returns an error if a fault was
    /// encountered while copying.
    pub fn copy_from_guest(&self, dest: &mut [u8], src: GuestPhysAddr) -> Result<()> {
        // Safety: _copy_from_guest internally detects and handles an invalid guest physical address
        // in `src`.
        self.do_guest_copy(src, dest.as_ptr(), dest.len(), |gpa, ptr, len| unsafe {
            _copy_from_guest(ptr as *mut u8, gpa.bits(), len)
        })
    }

    /// Uses `copy_fn` to copy `len` bytes between `guest_addr` and `host_ptr`.
    fn do_guest_copy<F>(
        &self,
        guest_addr: GuestPhysAddr,
        host_ptr: *const u8,
        len: usize,
        mut copy_fn: F,
    ) -> Result<()>
    where
        F: FnMut(GuestPhysAddr, *const u8, usize) -> usize,
    {
        let this_cpu = PerCpu::this_cpu();
        this_cpu.enter_guest_memcpy();
        let bytes = copy_fn(guest_addr, host_ptr, len);
        this_cpu.exit_guest_memcpy();
        if bytes == len {
            Ok(())
        } else {
            // TODO: Report faulting address and region so that the caller can turn it into a VM
            // exit code and allow the host to handle the page fault.
            Err(Error::UnhandledPageFault)
        }
    }

    /// Copies `count` pages from `src_addr` in the current guest to the converted pages starting at
    /// `from_addr`. The pages are then mapped into the child's address space at `to_addr`.
    pub fn copy_and_add_data_pages_builder(
        &self,
        src_addr: GuestPageAddr,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, VmStateInitializing>,
        to_addr: GuestPageAddr,
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
                    self.page_tracker.put_converted_page(p).unwrap();
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
            mapper.map_page_with_measurement(to_addr, mappable).unwrap();
        }
        mapper.finish();

        Ok(count)
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
        let mapper = VmPagesMapper::new(self, page_addr, num_pages).unwrap();
        for (page, addr) in converted_pages.zip(page_addr.iter_from()) {
            // Unwrap ok since we know that it's a converted page.
            let mappable = self.page_tracker.reclaim_page(page.clean()).unwrap();
            mapper.map_page(addr, mappable).unwrap();
        }
        mapper.finish();
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
            return Err(Error::UnalignedVmPages(page_root_addr));
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
    pub fn add_zero_pages_builder(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, VmStateInitializing>,
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
        mapper.finish();
        Ok(count)
    }

    /// Handles a page fault for the given address.
    pub fn handle_page_fault(&self, addr: GuestPhysAddr) -> Result<()> {
        if self.root.do_fault(addr) {
            Ok(())
        } else {
            Err(Error::UnhandledPageFault)
        }
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

    /// Locks `count` 4kB pages starting at `page_addr` for mapping, returning a `VmPagesMapper` that
    /// can be used to insert (and measure, if necessary) the pages.
    pub fn map_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<VmPagesMapper<T, VmStateInitializing>> {
        VmPagesMapper::with_new_region(self, page_addr, count, VmRegionType::PreMapped)
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
