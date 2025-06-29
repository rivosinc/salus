// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use attestation::AttestationManager;
use core::marker::PhantomData;
use drivers::{imsic::*, iommu::*, pci::PciBarPage, pci::PciDevice, pci::PcieRoot};
use page_tracking::{
    LockedPageList, PageList, PageTracker, PageTrackingError, TlbVersion, MAX_PAGE_OWNERS,
};
use riscv_page_tables::{
    tlb, GuestStageMapper, GuestStagePageTable, GuestStagePagingMode, PageTableError,
    ENTRIES_PER_PAGE,
};
use riscv_pages::*;
use riscv_regs::{
    hgatp, hstatus, DecodedInstruction, Exception, LocalRegisterCopy, PrivilegeLevel, Readable,
    RiscvCsrInterface, Writeable, CSR,
};
use sync::{Mutex, Once, RwLock, RwLockReadGuard};

use crate::hyp_layout::UmodeSlotId;
use crate::hyp_map::{Error as HypMapError, HypMap, UmodeSlotPerm};
use crate::smp::PerCpu;
use crate::vm::{VmStateAny, VmStateFinalized, VmStateInitializing};
use crate::vm_id::VmId;

#[derive(Debug)]
// annoying false unused "feature": https://github.com/rust-lang/rust/issues/123068
#[allow(dead_code)]
pub enum Error {
    Paging(PageTableError),
    PageFault(PageFaultType, Exception, GuestPhysAddr),
    NestingTooDeep,
    UnalignedAddress,
    UnsupportedPageSize(PageSize),
    AddressOverflow,
    TlbCountUnderflow,
    InvalidTlbVersion,
    TlbFenceInProgress,
    OverlappingVmRegion,
    InsufficientVmRegionSpace,
    VmRegionNotFound,
    VmRegionNotConfidentialOrShared,
    VmRegionNotRemovable,
    InvalidMapRegion,
    EmptyPageRange,
    PageRangeNotEmpty,
    Measurement(attestation::Error),
    NoImsicVirtualization,
    ImsicGeometryAlreadySet,
    IommuContextAlreadySet,
    NoIommu,
    AllocatingGscId(IommuError),
    CreatingMsiPageTable(IommuError),
    InvalidImsicLocation,
    MsiTableMapping(IommuError),
    AttachingDevice(IommuError),
    PageTracker(PageTrackingError),
    HypMap(HypMapError),
    InsufficientPtePages,
    InvalidPageSize,
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum InstructionFetchError {
    FailedDecode(u32),
    FetchFault,
}

pub type InstructionFetchResult = core::result::Result<DecodedInstruction, InstructionFetchError>;

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

    /// Decrements the reference count for this version, returning the new reference count.
    fn put(&mut self) -> Result<u64> {
        self.count = self.count.checked_sub(1).ok_or(Error::TlbCountUnderflow)?;
        Ok(self.count)
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
    fn current_version(&self) -> TlbVersion {
        self.inner.lock().current.version
    }

    /// Returns the minimum TLB version with active references.
    fn min_version(&self) -> TlbVersion {
        let inner = self.inner.lock();
        if let Some(prev) = inner.prev.as_ref()
            && prev.count() != 0
        {
            prev.version()
        } else {
            inner.current.version()
        }
    }

    /// Attempts to increment the current TLB version. The TLB version can only be incremented if
    /// there are no outstanding references to versions other than the current version. Returns
    /// true if there were no outstanding references to the current TLB version.
    fn increment(&self) -> Result<bool> {
        let mut inner = self.inner.lock();
        if inner.prev.as_ref().filter(|v| v.count() != 0).is_none() {
            // We're only ok to proceed with an increment if there's no references to the previous
            // TLB version.
            let next = inner.current.version().increment();
            if inner.current.count() != 0 {
                inner.prev = Some(inner.current.clone());
            } else {
                inner.prev = None;
            }
            inner.current = RefCountedTlbVersion::new(next);
            Ok(inner.prev.is_none())
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

    /// Drops a reference to the given TLB version. Returns true if the last reference has been
    /// dropped to the previous TLB version, indicating that a TLB shootdown has completed.
    fn put_version(&self, version: TlbVersion) -> Result<bool> {
        let mut inner = self.inner.lock();
        if inner.current.version() == version {
            inner.current.put()?;
            Ok(false)
        } else if let Some(prev) = inner.prev.as_mut().filter(|v| v.version() == version) {
            let count = prev.put()?;
            if count == 0 {
                inner.prev = None;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err(Error::InvalidTlbVersion)
        }
    }
}

/// A reference to a range of pages in a VM's address space that have been pinned in the shared
/// state. The pin is released (shared reference count dropped) in drop(). Used for long-term
/// sharing of memory between a VM and the hypervisor.
pub struct PinnedPages {
    range: SupervisorPageRange,
    page_tracker: PageTracker,
    owner: PageOwnerId,
}

impl PinnedPages {
    // Safety: The caller must guarantee that the pages in the specified range are in the "Shared"
    // state and are owned by `owner`.
    unsafe fn new(
        range: SupervisorPageRange,
        page_tracker: PageTracker,
        owner: PageOwnerId,
    ) -> Self {
        Self {
            range,
            page_tracker,
            owner,
        }
    }

    /// Returns the range of pages in the pinned region.
    pub fn range(&self) -> SupervisorPageRange {
        self.range
    }
}

impl Drop for PinnedPages {
    fn drop(&mut self) {
        for addr in self.range {
            // Unwrap ok: the caller guaranteed at construction that the range of pages is shared
            // and owned by `self.owner`.
            self.page_tracker
                .release_page_by_addr(addr, PageSize::Size4k, self.owner)
                .unwrap();
        }
    }
}

/// A reference to a U-mode slot in the current page-table that is mapped with VM pages in the
/// shared state. On drop(), the shared reference count is dropped and the slot unmapped.
pub struct GuestUmodeMapping {
    slot: UmodeSlotId,
    num_pages: u64,
    page_tracker: PageTracker,
    owner: PageOwnerId,
}

impl GuestUmodeMapping {
    // Safety: the caller must guarantee that the slot maps `num_pages` pages owned by `owner` in
    // state "Shared".
    unsafe fn new(
        slot: UmodeSlotId,
        num_pages: u64,
        page_tracker: PageTracker,
        owner: PageOwnerId,
    ) -> Self {
        Self {
            slot,
            num_pages,
            page_tracker,
            owner,
        }
    }

    /// Returns the start of the mapping (i.e., the address of the mapped U-mode slot).
    pub fn vaddr(&self) -> PageAddr<SupervisorVirt> {
        HypMap::umode_slot_va(self.slot)
    }
}

impl Drop for GuestUmodeMapping {
    fn drop(&mut self) {
        // Unwrap okay: the caller guaranteed at construction that slot is mapped with `num_pages` pages.
        let unmapped = PerCpu::this_cpu()
            .page_table()
            .unmap_umode_slot(self.slot, self.num_pages)
            .unwrap();
        for addr in unmapped {
            // Unwrap ok: the caller guaranteed at construction that the mapped pages are shared and
            // owned by `self.owner`.
            self.page_tracker
                .release_page_by_addr(addr, PageSize::Size4k, self.owner)
                .unwrap();
        }
        // Unmapped pages in current CPU. Flush TLBs.
        tlb::sfence_vma(None, None);
    }
}

// Types of regions in a VM's guest physical address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VmRegionType {
    // Memory that is private to this VM.
    Confidential,
    // Memory that is shared with the parent
    Shared,
    // Emulated MMIO region; accesses always cause a fault that is forwarded to the VM's host.
    Mmio,
    // IMSIC interrupt file pages.
    Imsic,
    // PCI BAR pages.
    Pci,
    // Memory that is private to this VM and marked removable.
    ConfidentialRemovable,
    // Memory that is shared with the host and marked removable.
    SharedRemovable,
}

// A contiguous region of guest physical address space.
#[derive(Clone, Debug)]
struct VmRegion {
    start: GuestPageAddr,
    end: GuestPageAddr,
    region_type: VmRegionType,
}

// The maximum number of distinct memory regions we support in `VmRegionList`.
const MAX_MEM_REGIONS: usize = 128;

// The regions of guest physical address space for a VM. Used to track which parts of the address
// space are designated for a particular purpose. Pages may only be inserted into a VM's address
// space if the mapping falls within a region of the proper type.
struct VmRegionList {
    regions: ArrayVec<VmRegion, MAX_MEM_REGIONS>,
}

impl VmRegionList {
    // Creates an empty `VmRegionList`.
    fn new() -> Self {
        Self {
            regions: ArrayVec::new(),
        }
    }

    // Inserts a region at [`start`, `end`) of type `region_type`.
    fn add(
        &mut self,
        start: GuestPageAddr,
        end: GuestPageAddr,
        region_type: VmRegionType,
    ) -> Result<()> {
        // Keep the list sorted, inserting the region in the requested spot as long as it doesn't
        // overlap with anything else.
        let mut index = 0;
        for other in self.regions.iter() {
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
        self.regions
            .try_insert(index, region)
            .map_err(|_| Error::InsufficientVmRegionSpace)?;
        self.try_coalesce_at(index);
        Ok(())
    }

    // Removes the region at [`start`, `end`) of type `region_type`.
    fn remove(
        &mut self,
        start: GuestPageAddr,
        end: GuestPageAddr,
        region_type: VmRegionType,
    ) -> Result<()> {
        let (mut index, region) = self
            .regions
            .iter()
            .enumerate()
            .find(|(_i, r)| r.start <= start && r.end >= end && r.region_type == region_type)
            .ok_or(Error::InvalidMapRegion)?;

        let r_start = region.start;
        let r_end = region.end;

        // Check if we have enough space to accommodate remaining range. We need to check this
        // only when we need two extra regions. We will always get one free region given we are
        // removing the current region. Due to this we only need to check for `is_full` condition
        // to make sure we can accommodate two extra regions.
        if r_start < start && r_end > end && self.regions.is_full() {
            return Err(Error::InsufficientVmRegionSpace);
        }

        // Remove the memory region.
        self.regions.remove(index);

        // Add the remaining range at the start as a new region.
        if r_start < start {
            let new_region = VmRegion {
                start: r_start,
                end: start,
                region_type,
            };

            // Using insert instead of try_insert given we have already checked for free space.
            self.regions.insert(index, new_region);

            index += 1;
        }

        // Add the remaining range at the end as a new region.
        if r_end > end {
            let new_region = VmRegion {
                start: end,
                end: r_end,
                region_type,
            };

            // Using insert instead of try_insert given we have already checked for free space.
            self.regions.insert(index, new_region);
        }

        Ok(())
    }

    // Updates the region type with the new provided type. The region must already exist
    // and be marked with the expected type. If the range covers only a subrange of the
    // region, it is split accordingly.
    fn update(
        &mut self,
        start: GuestPageAddr,
        end: GuestPageAddr,
        region_type: VmRegionType,
        new_region_type: VmRegionType,
        coalesce: bool,
    ) -> Result<()> {
        let mut index = self
            .regions
            .iter()
            .position(|r| r.start <= start && end <= r.end && r.region_type == region_type)
            .ok_or(Error::VmRegionNotFound)?;

        // Make sure we have space to split the region.
        let to_split = &self.regions[index].clone();
        let mut to_reserve = 0;
        if to_split.start != start {
            to_reserve += 1;
        }
        if to_split.end != end {
            to_reserve += 1;
        }
        if self.regions.remaining_capacity() < to_reserve {
            return Err(Error::InsufficientVmRegionSpace);
        }

        // Now do the split, marking the range with the `new_region_type` type.
        if to_split.start != start {
            let prev = VmRegion {
                start: to_split.start,
                end: start,
                region_type,
            };
            self.regions.insert(index, prev);
            index += 1;
        }
        self.regions[index] = VmRegion {
            start,
            end,
            region_type: new_region_type,
        };
        if to_split.end != end {
            let next = VmRegion {
                start: end,
                end: to_split.end,
                region_type,
            };
            self.regions.insert(index + 1, next);
        }

        if coalesce {
            self.try_coalesce_at(index);
        }

        Ok(())
    }

    // Try to coalesce the region list at `index`, returning the index of the coalesced region.
    // Called after modifying the region list to coalesce entries with the same region type.
    fn try_coalesce_at(&mut self, mut index: usize) -> Option<usize> {
        if let Some(region) = self.regions.get(index) {
            let mut start = region.start;
            let end = region.end;
            let region_type = region.region_type;

            // Coalesce with same-typed regions before and after this one, if possible.
            // Avoid potential underflows and overflows on index.
            if let Some(ref mut prev) = index.checked_sub(1).and_then(|i| self.regions.get_mut(i)) {
                if prev.end == start && prev.region_type == region_type {
                    prev.end = end;
                    start = prev.start;
                    self.regions.remove(index);
                    index -= 1;
                }
            }

            if let Some(ref mut next) = index.checked_add(1).and_then(|i| self.regions.get_mut(i)) {
                if next.start == end && next.region_type == region_type {
                    next.start = start;
                    self.regions.remove(index);
                }
            }

            Some(index)
        } else {
            None
        }
    }

    // Returns if the range [`start`, `end`) is fully contained within a region of type `region_type`.
    fn contains<F>(&self, start: GuestPageAddr, end: GuestPageAddr, pred: F) -> bool
    where
        F: Fn(VmRegionType) -> bool,
    {
        self.regions
            .iter()
            .any(|r| r.start <= start && r.end >= end && pred(r.region_type))
    }

    // Returns the type of the region that `addr` resides in, or `None` if it's not in any region.
    fn find(&self, addr: GuestPhysAddr) -> Option<VmRegionType> {
        self.regions
            .iter()
            .find(|r| r.start.bits() <= addr.bits() && r.end.bits() > addr.bits())
            .map(|r| r.region_type)
    }
}

/// Wrapper for a `GuestStageMapper` created from the page table of `VmPages`. Measures pages as
/// they are inserted, if necessary.
pub struct VmPagesMapper<'a, T: GuestStagePagingMode, M> {
    vm_pages: &'a VmPages<T>,
    mapper: GuestStageMapper<'a, T>,
    _region_guard: RwLockReadGuard<'a, VmRegionList>,
    _mapper_type: PhantomData<M>,
}

impl<'a, T: GuestStagePagingMode, M> VmPagesMapper<'a, T, M> {
    // Creates a new `VmPagesMapper` for `num_pages` starting at `page_addr`, which must lie within
    // a region of type `region_type`.
    fn new_in_region<F>(
        vm_pages: &'a VmPages<T>,
        page_addr: GuestPageAddr,
        page_size: PageSize,
        num_pages: u64,
        pred: F,
    ) -> Result<Self>
    where
        F: Fn(VmRegionType) -> bool,
    {
        let end = page_addr
            .checked_add_pages_with_size(num_pages, page_size)
            .ok_or(Error::AddressOverflow)?;
        let regions = vm_pages.regions.read();
        if !regions.contains(page_addr, end, pred) {
            return Err(Error::InvalidMapRegion);
        }
        let mapper = vm_pages
            .root
            .map_range(page_addr, page_size, num_pages, &mut || {
                vm_pages.pte_pages.pop()
            })
            .map_err(Error::Paging)?;
        Ok(Self {
            vm_pages,
            mapper,
            _region_guard: regions,
            _mapper_type: PhantomData,
        })
    }

    // Creates a new `VmPagesMapper` for remapping `num_pages` starting at `page_addr`, which must lie
    // within a region of type `region_type` and should have been already mapped.
    fn new_in_region_mapped(
        vm_pages: &'a VmPages<T>,
        page_addr: GuestPageAddr,
        page_size: PageSize,
        num_pages: u64,
        region_type: VmRegionType,
    ) -> Result<Self> {
        let end = page_addr
            .checked_add_pages_with_size(num_pages, page_size)
            .ok_or(Error::AddressOverflow)?;
        let regions = vm_pages.regions.read();
        if !regions.contains(page_addr, end, |r| r == region_type) {
            return Err(Error::InvalidMapRegion);
        }
        let mapper = vm_pages
            .root
            .remap_range(page_addr, page_size, num_pages)
            .map_err(Error::Paging)?;
        Ok(Self {
            vm_pages,
            mapper,
            _region_guard: regions,
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

    // Remaps `page` at `to_addr` and returns previous SupervisorPageAddr address.
    fn do_remap_page<P, MR>(&self, to_addr: GuestPageAddr, page: P) -> Result<SupervisorPageAddr>
    where
        P: MappablePhysPage<MR>,
        MR: MeasureRequirement,
    {
        self.mapper.remap_page(to_addr, page).map_err(Error::Paging)
    }
}

pub enum ZeroPages {}
/// A `VmPagesMapper` for confidential zero pages.
pub type ZeroPagesMapper<'a, T> = VmPagesMapper<'a, T, ZeroPages>;

impl<'a, T: GuestStagePagingMode> ZeroPagesMapper<'a, T> {
    /// Maps a zero page into the guest's address space.
    pub fn map_page(&self, to_addr: GuestPageAddr, page: Page<MappableClean>) -> Result<()> {
        self.do_map_page(to_addr, page)
    }
}

pub enum MeasuredPages {}
/// A `VmPagesMapper` for confidential measured pages.
pub type MeasuredPagesMapper<'a, T> = VmPagesMapper<'a, T, MeasuredPages>;

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
pub type SharedPagesMapper<'a, T> = VmPagesMapper<'a, T, SharedPages>;

impl<'a, T: GuestStagePagingMode> SharedPagesMapper<'a, T> {
    /// Maps a shared page into the guest's address space.
    pub fn map_page(&self, to_addr: GuestPageAddr, page: Page<MappableShared>) -> Result<()> {
        self.do_map_page(to_addr, page)
    }
}

pub enum ImsicPages {}
/// A `VmPagesMapper` for IMSIC guest file pages.
pub type ImsicPagesMapper<'a, T> = VmPagesMapper<'a, T, ImsicPages>;

impl<'a, T: GuestStagePagingMode> ImsicPagesMapper<'a, T> {
    /// Maps an IMSIC page into the guest's address space, also updating the MSI page tables if
    /// necessary.
    pub fn map_page(
        &self,
        to_addr: GuestPageAddr,
        page: ImsicGuestPage<MappableClean>,
    ) -> Result<()> {
        let dest_location = page.location();
        self.do_map_page(to_addr, page)?;
        if let Some(geometry) = self.vm_pages.imsic_geometry.get()
            && let Some(iommu_context) = self.vm_pages.iommu_context.get()
        {
            let src_location = geometry
                .addr_to_location(to_addr)
                .ok_or(Error::InvalidImsicLocation)?;
            iommu_context
                .msi_page_table
                .map(src_location, dest_location)
                .map_err(Error::MsiTableMapping)?;
        }
        Ok(())
    }

    /// Remaps an IMSIC page into the guest's address space, also updating the MSI page tables if
    /// necessary.
    pub fn remap_page(
        &self,
        to_addr: GuestPageAddr,
        page: ImsicGuestPage<MappableClean>,
    ) -> Result<SupervisorPageAddr> {
        let dest_location = page.location();
        let prev_addr = self.do_remap_page(to_addr, page)?;
        if let Some(geometry) = self.vm_pages.imsic_geometry.get()
            && let Some(iommu_context) = self.vm_pages.iommu_context.get()
        {
            let src_location = geometry
                .addr_to_location(to_addr)
                .ok_or(Error::InvalidImsicLocation)?;
            iommu_context
                .msi_page_table
                .remap(src_location, dest_location)
                .map_err(Error::MsiTableMapping)?;
        }
        Ok(prev_addr)
    }
}

pub enum PciPages {}
/// A `VmPagesMapper` for PCI BAR memory pages.
pub type PciPagesMapper<'a, T> = VmPagesMapper<'a, T, PciPages>;

impl<'a, T: GuestStagePagingMode> PciPagesMapper<'a, T> {
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
    Confidential,
    /// A page fault taken when accessing a shared memory region. The host may handle these faults
    /// by inserting a page into the guest's address space.
    Shared,
    /// A page fault taken to an emulated MMIO page.
    Mmio,
    /// A page fault taken to an IMSIC guest interrupt file page.
    Imsic,
    /// A page fault taken when accessing memory outside of any valid region of guest physical
    /// address space. These faults are not resolvable.
    Unmapped,
}

/// Represents the active VM address space. Holds a reference to the TLB version of the address space
/// at the time the address space was activated. Used to directly access a guest's memory.
pub struct ActiveVmPages<'a, T: GuestStagePagingMode> {
    tlb_version: TlbVersion,
    vm_pages: FinalizedVmPages<'a, T>,
}

impl<'a, T: GuestStagePagingMode> Drop for ActiveVmPages<'a, T> {
    fn drop(&mut self) {
        self.vm_pages
            .inner
            .tlb_tracker
            .put_version(self.tlb_version)
            .unwrap();
    }
}

impl<'a, T: GuestStagePagingMode> ActiveVmPages<'a, T> {
    fn new(
        vm_pages: FinalizedVmPages<'a, T>,
        vmid: VmId,
        prev_tlb_version: Option<TlbVersion>,
    ) -> Self {
        let mut hgatp = LocalRegisterCopy::<u64, hgatp::Register>::new(0);
        hgatp.modify(hgatp::vmid.val(vmid.vmid()));
        hgatp.modify(hgatp::ppn.val(Pfn::from(vm_pages.root_address()).bits()));
        hgatp.modify(hgatp::mode.val(T::HGATP_MODE));
        CSR.hgatp.set(hgatp.get());

        let tlb_version = vm_pages.inner.tlb_tracker.get_version();
        // Fence if this VMID was previously running on this CPU with an old TLB version.
        if let Some(v) = prev_tlb_version
            && v < tlb_version
        {
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
            let fault_type = self.get_page_fault_cause(Exception::GuestStorePageFault, fault_addr);
            Err(Error::PageFault(
                fault_type,
                Exception::GuestStorePageFault,
                fault_addr,
            ))
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
            let fault_type = self.get_page_fault_cause(Exception::GuestLoadPageFault, fault_addr);
            Err(Error::PageFault(
                fault_type,
                Exception::GuestLoadPageFault,
                fault_addr,
            ))
        }
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
        match self.vm_pages.inner.regions.read().find(fault_addr) {
            Some(VmRegionType::Confidential) | Some(VmRegionType::ConfidentialRemovable) => {
                Confidential
            }
            Some(VmRegionType::Shared) | Some(VmRegionType::SharedRemovable) => Shared,
            Some(VmRegionType::Mmio) => match exception {
                Exception::GuestLoadPageFault | Exception::GuestStorePageFault => Mmio,
                _ => Unmapped,
            },
            Some(VmRegionType::Imsic) => match exception {
                // Only stores can be made to an IMSIC page.
                Exception::GuestStorePageFault => Imsic,
                _ => Unmapped,
            },
            _ => Unmapped,
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
            pages: Mutex::new(PageList::new(page_tracker, PageSize::Size4k)),
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
        let gscid = alloc_gscid(msi_page_table.owner()).map_err(Error::AllocatingGscId)?;
        Ok(Self {
            msi_page_table,
            gscid,
        })
    }
}

impl Drop for VmIommuContext {
    fn drop(&mut self) {
        // Detach any devices we own from the IOMMU.
        let owner = self.msi_page_table.owner();
        for pci in PcieRoot::get_roots() {
            for dev in pci.devices() {
                let mut dev = dev.lock();
                if dev.owner() == Some(owner)
                    && let Some(iommu) = Iommu::get_for_device(&dev)
                {
                    // Unwrap ok: `self.gscid` must be valid and match the ownership of the device
                    // to have been attached in the first place.
                    iommu.detach_pci_device(&mut dev, self.gscid).unwrap();
                }
            }
        }

        // Unwrap ok: `self.gscid` must be valid and freeable since we've detached all devices
        // using it.
        free_gscid(self.gscid).unwrap();
    }
}

/// VmPages is the single management point for memory used by virtual machines.
///
/// After initial setup all memory not used for Hypervisor purposes is managed by a VmPages
/// instance. Rules around sharing and isolating memory are enforced by this module.
///
/// Machines are allowed to donate pages to child machines and to share donated pages with parent
/// machines.
pub struct VmPages<T: GuestStagePagingMode> {
    page_owner_id: PageOwnerId,
    page_tracker: PageTracker,
    tlb_tracker: TlbTracker,
    regions: RwLock<VmRegionList>,
    // How many nested TVMs deep this VM is, with 0 being the host.
    nesting: usize,
    root: GuestStagePageTable<T>,
    pte_pages: PtePagePool,
    imsic_geometry: Once<GuestImsicGeometry>,
    iommu_context: Once<VmIommuContext>,
}

impl<T: GuestStagePagingMode> VmPages<T> {
    /// Creates a new `VmPages` from the given root page table.
    pub fn new(root: GuestStagePageTable<T>, nesting: usize) -> Self {
        let page_tracker = root.page_tracker();
        Self {
            page_owner_id: root.page_owner_id(),
            page_tracker,
            tlb_tracker: TlbTracker::new(),
            regions: RwLock::new(VmRegionList::new()),
            nesting,
            root,
            pte_pages: PtePagePool::new(page_tracker),
            imsic_geometry: Once::new(),
            iommu_context: Once::new(),
        }
    }

    /// Returns the `PageOwnerId` associated with the pages contained in this machine.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.page_owner_id
    }

    /// Returns the global page tracking structure.
    pub fn page_tracker(&self) -> PageTracker {
        self.page_tracker
    }

    /// Returns a `VmPagesRef` to `self` in state `S`. The caller must ensure that `S` matches
    /// the current state of the VM to which this `VmPages` belongs.
    pub fn as_ref<S>(&self) -> VmPagesRef<T, S> {
        VmPagesRef::new(self)
    }
}

/// A reference to a `VmPages` in a particular state `S` that exposes the appropriate functionality
/// for a VM in that state.
pub struct VmPagesRef<'a, T: GuestStagePagingMode, S> {
    inner: &'a VmPages<T>,
    _vm_state: PhantomData<S>,
}

impl<'a, T: GuestStagePagingMode, S> VmPagesRef<'a, T, S> {
    // Creates a new `VmPagesRef` in state `S` to `vm_pages`.
    fn new(vm_pages: &'a VmPages<T>) -> Self {
        Self {
            inner: vm_pages,
            _vm_state: PhantomData,
        }
    }

    /// Returns the `PageOwnerId` associated with the pages contained in this machine.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.inner.page_owner_id
    }

    /// Returns the level of nesting of this VM, with 0 being the host.
    pub fn nesting(&self) -> usize {
        self.inner.nesting
    }

    // Returns the address of the root page table for this VM.
    fn root_address(&self) -> SupervisorPageAddr {
        // TODO: Cache this to avoid bouncing off the lock?
        self.inner.root.get_root_address()
    }

    /// Returns this VM's IMSIC geometry if it was set up for IMSIC virtualization.
    pub fn imsic_geometry(&self) -> Option<GuestImsicGeometry> {
        self.inner.imsic_geometry.get().cloned()
    }

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&self, page: Page<InternalClean>) -> Result<()> {
        if page.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(page.size()));
        }
        self.inner.pte_pages.push(page);
        Ok(())
    }

    fn do_map_pages<M, F>(
        &self,
        page_addr: GuestPageAddr,
        page_size: PageSize,
        count: u64,
        pred: F,
    ) -> Result<VmPagesMapper<'a, T, M>>
    where
        F: Fn(VmRegionType) -> bool,
    {
        if count == 0 {
            return Err(Error::EmptyPageRange);
        }
        VmPagesMapper::new_in_region(self.inner, page_addr, page_size, count, pred)
    }

    fn do_remap_pages<M>(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
        region_type: VmRegionType,
    ) -> Result<VmPagesMapper<'a, T, M>> {
        if count == 0 {
            return Err(Error::EmptyPageRange);
        }
        VmPagesMapper::new_in_region_mapped(
            self.inner,
            page_addr,
            PageSize::Size4k,
            count,
            region_type,
        )
    }

    /// Same as `map_zero_pages()`, but for IMSIC guest interrupt file pages.
    pub fn map_imsic_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<ImsicPagesMapper<'a, T>> {
        self.do_map_pages(page_addr, PageSize::Size4k, count, |r| {
            r == VmRegionType::Imsic
        })
    }

    /// Same as `map_imsic_pages()`, but for remapping the virtual address to a different
    /// physical address.
    pub fn remap_imsic_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<ImsicPagesMapper<'a, T>> {
        self.do_remap_pages(page_addr, count, VmRegionType::Imsic)
    }

    /// Same as `map_zero_pages()`, but for PCI BAR memory pages.
    pub fn map_pci_pages(
        &self,
        page_addr: GuestPageAddr,
        count: u64,
    ) -> Result<PciPagesMapper<'a, T>> {
        self.do_map_pages(page_addr, PageSize::Size4k, count, |r| {
            r == VmRegionType::Pci
        })
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

        self.inner.regions.write().add(page_addr, end, region_type)
    }

    // Removes a region of type `region_type`.
    fn do_remove_region(
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

        self.inner
            .regions
            .write()
            .remove(page_addr, end, region_type)
    }
}

impl<'a, T: GuestStagePagingMode, S> Clone for VmPagesRef<'a, T, S> {
    fn clone(&self) -> VmPagesRef<'a, T, S> {
        VmPagesRef::new(self.inner)
    }
}

/// Represents the address space of VM that may be in any state.
pub type AnyVmPages<'a, T> = VmPagesRef<'a, T, VmStateAny>;

/// Represents the address space of a finalized, or runnable, VM. Used to expose the operations
/// that are possible on a runnable VM, including conversion of pages and construction of child
/// VMs.
pub type FinalizedVmPages<'a, T> = VmPagesRef<'a, T, VmStateFinalized>;

impl<'a, T: GuestStagePagingMode> FinalizedVmPages<'a, T> {
    /// Adds an emulated MMIO region of `len` bytes starting at `page_addr` to this VM's address
    /// space.
    pub fn add_mmio_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_add_region(page_addr, len, VmRegionType::Mmio)
    }

    /// Removes an emulated MMIO region of `len` bytes starting at `page_addr` from this VM's address
    /// space.
    pub fn remove_mmio_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_remove_region(page_addr, len, VmRegionType::Mmio)
    }

    /// Converts the specified memory region from confidential to shared.
    pub fn share_mem_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_convert_mem_region(page_addr, len, true)
    }

    /// Converts the specified memory region from shared to confidential.
    pub fn unshare_mem_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_convert_mem_region(page_addr, len, false)
    }

    fn do_convert_mem_region(&self, page_addr: GuestPageAddr, len: u64, share: bool) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;

        let (region_type, new_region_type) = if share {
            // Check the address range lies within a "Confidential" region of guest physical
            // address space and update the region type to "ConfidentialRemovable".
            (
                VmRegionType::Confidential,
                VmRegionType::ConfidentialRemovable,
            )
        } else {
            // Check the address range lies within a "Shared" region of guest physical address
            // space and update the region type to "SharedRemovable".
            (VmRegionType::Shared, VmRegionType::SharedRemovable)
        };
        self.inner
            .regions
            .write()
            .update(page_addr, end, region_type, new_region_type, false)?;

        Ok(())
    }

    /// Completes guest memory sharing by verifying the pages have been previously removed
    /// and by marking the region as "Shared".
    pub fn complete_share_mem_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_complete_convert_mem_region(page_addr, len, true)
    }

    /// Completes guest memory unsharing by verifying the pages have been previously removed
    /// and by marking the region as "Confidential".
    pub fn complete_unshare_mem_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_complete_convert_mem_region(page_addr, len, false)
    }

    fn do_complete_convert_mem_region(
        &self,
        page_addr: GuestPageAddr,
        len: u64,
        share: bool,
    ) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;

        // Check all the pages have been removed.
        if !self.inner.root.range_is_empty(page_addr, len) {
            return Err(Error::PageRangeNotEmpty);
        }

        let (region_type, new_region_type) = if share {
            // Check the address range lies within a "ConfidentialRemovable" region of guest
            // physical address space and update the region type to "Shared".
            (VmRegionType::ConfidentialRemovable, VmRegionType::Shared)
        } else {
            // Check the address range lies within a "SharedRemovable" region of guest physical
            // address space and update the region type to "Confidential".
            (VmRegionType::SharedRemovable, VmRegionType::Confidential)
        };
        self.inner
            .regions
            .write()
            .update(page_addr, end, region_type, new_region_type, true)?;

        Ok(())
    }

    /// The host has rejected the sharing request, meaning we must revert the region back
    /// to its previous state.
    pub fn reject_share_mem_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_reject_convert_mem_region(page_addr, len, true)
    }

    /// The host has rejected the unsharing request, meaning we must revert the region back
    /// to its previous state.
    pub fn reject_unshare_mem_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_reject_convert_mem_region(page_addr, len, false)
    }

    fn do_reject_convert_mem_region(
        &self,
        page_addr: GuestPageAddr,
        len: u64,
        share: bool,
    ) -> Result<()> {
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;

        let (region_type, new_region_type) = if share {
            // Check the address range lies within a "ConfidentialRemovable" region of guest
            // physical address space and update the region type to "Confidential".
            (
                VmRegionType::ConfidentialRemovable,
                VmRegionType::Confidential,
            )
        } else {
            // Check the address range lies within a "SharedRemovable" region of guest physical
            // address space and update the region type to "Shared".
            (VmRegionType::SharedRemovable, VmRegionType::Shared)
        };
        self.inner
            .regions
            .write()
            .update(page_addr, end, region_type, new_region_type, true)?;

        Ok(())
    }

    /// Locks `count` `page_size` pages starting at `page_addr` for mapping of zero-filled pages
    /// in a region of confidential memory, returning a `VmPagesMapper` that can be used to insert
    /// the pages.
    pub fn map_zero_pages(
        &self,
        page_addr: GuestPageAddr,
        page_size: PageSize,
        count: u64,
    ) -> Result<ZeroPagesMapper<'a, T>> {
        self.do_map_pages(page_addr, page_size, count, |r| {
            r == VmRegionType::Confidential || r == VmRegionType::ConfidentialRemovable
        })
    }

    /// Same as `map_zero_pages()`, but for pages in shared (non-confidential) regions.
    pub fn map_shared_pages(
        &self,
        page_addr: GuestPageAddr,
        page_size: PageSize,
        count: u64,
    ) -> Result<SharedPagesMapper<'a, T>> {
        self.do_map_pages(page_addr, page_size, count, |r| {
            r == VmRegionType::Shared || r == VmRegionType::SharedRemovable
        })
    }

    fn do_get_converted_pages<P: ConvertedPhysPage>(
        &self,
        page_addr: GuestPageAddr,
        target_page_size: PageSize,
        num_pages: u64,
    ) -> Result<LockedPageList<P::DirtyPage>> {
        if num_pages == 0 {
            return Err(Error::EmptyPageRange);
        }

        let mut prev_pa: Option<SupervisorPageAddr> = None;
        let version = self.inner.tlb_tracker.min_version();
        let converted = self
            .inner
            .root
            .get_invalidated_pages(page_addr, num_pages * target_page_size as u64, |pa, ps| {
                // Converted pages should always be 4k given the host is
                // exclusively mapped with 4k pages.
                if ps.is_huge() {
                    return false;
                }

                // We need to check the first subpage is suitably aligned and that every following
                // subpage covering the same huge page is contiguous to the previous one. That's
                // because we have some logic below that allows to create huge pages out of 4k pages,
                // by skipping the intermediate pages. That's why we must ensure the whole range is
                // made of physically contiguous subpages for each huge page and they are suitably
                // aligned.
                if let Some(addr) = prev_pa {
                    let next_pa = addr.checked_add_pages_with_size(1, ps).unwrap();
                    // Check the current page is contiguous with the previous page
                    if next_pa.is_aligned(target_page_size) {
                        // This is our way of resetting the check for contiguous subpages.
                        prev_pa = None;
                    } else {
                        // Check the current page is contiguous with the previous page
                        if next_pa != pa {
                            return false;
                        }
                        prev_pa = Some(pa);
                    }
                } else {
                    if !pa.is_aligned(target_page_size) {
                        // This is the first page, we must ensure it's suitably aligned with the requested
                        // page size.
                        return false;
                    }
                    prev_pa = Some(pa);
                }

                self.inner.page_tracker.is_converted_page(
                    pa,
                    ps,
                    self.inner.page_owner_id,
                    P::mem_type(),
                    version,
                )
            })
            .map_err(Error::Paging)?;

        // Lock the pages for assignment.
        let mut locked_pages = LockedPageList::new(self.inner.page_tracker(), target_page_size);
        for (paddr, _) in converted.filter(|(pa, _)| pa.is_aligned(target_page_size)) {
            // Unwrap ok: The pages are guaranteed to be converted and no one else can get a
            // reference to them until the iterator is destroyed.
            let page = self
                .inner
                .page_tracker
                .get_converted_page::<P>(paddr, target_page_size, self.inner.page_owner_id, version)
                .unwrap();
            locked_pages.push(page).unwrap();
        }

        Ok(locked_pages)
    }

    /// Acquries an exclusive reference to the `num_pages` converted pages starting at `page_addr`.
    pub fn get_converted_pages(
        &self,
        page_addr: GuestPageAddr,
        page_size: PageSize,
        num_pages: u64,
    ) -> Result<LockedPageList<Page<ConvertedDirty>>> {
        self.do_get_converted_pages::<Page<ConvertedDirty>>(page_addr, page_size, num_pages)
    }

    /// Acquries an exclusive reference to the `num_pages` shared host pages starting at `page_addr`.
    pub fn get_shareable_pages(
        &self,
        page_addr: GuestPageAddr,
        target_page_size: PageSize,
        num_pages: u64,
    ) -> Result<impl 'a + Iterator<Item = Page<Shareable>>> {
        let page_tracker = self.inner.page_tracker();
        let page_owner_id = self.inner.page_owner_id();
        let mut prev_pa: Option<SupervisorPageAddr> = None;
        Ok(self
            .inner
            .root
            .get_mapped_pages(
                page_addr,
                num_pages * target_page_size as u64,
                move |pa, ps| {
                    // Shareable pages should always be 4k given the host is
                    // exclusively mapped with 4k pages.
                    if ps.is_huge() {
                        return false;
                    }

                    // We need to check the first subpage is suitably aligned and that every following
                    // subpage covering the same huge page is contiguous to the previous one. That's
                    // because we have some logic below that allows to create huge pages out of 4k pages,
                    // by skipping the intermediate pages. That's why we must ensure the whole range is
                    // made of physically contiguous subpages for each huge page and they are suitably
                    // aligned.
                    if let Some(addr) = prev_pa {
                        let next_pa = addr.checked_add_pages_with_size(1, ps).unwrap();
                        // Check the current page is contiguous with the previous page
                        if next_pa.is_aligned(target_page_size) {
                            // This is our way of resetting the check for contiguous subpages.
                            prev_pa = None;
                        } else {
                            // Check the current page is contiguous with the previous page
                            if next_pa != pa {
                                return false;
                            }
                            prev_pa = Some(pa);
                        }
                    } else {
                        if !pa.is_aligned(target_page_size) {
                            // This is the first page, we must ensure it's suitably aligned with the requested
                            // page size.
                            return false;
                        }
                        prev_pa = Some(pa);
                    }

                    page_tracker.is_shareable_page(pa, ps, page_owner_id, MemType::Ram)
                },
            )
            .map_err(Error::Paging)?
            .filter_map(move |(addr, _)| {
                if !addr.is_aligned(target_page_size) {
                    return None;
                }
                Some(
                    page_tracker
                        .get_shareable_page(addr, target_page_size, page_owner_id)
                        .unwrap(),
                )
            }))
    }

    /// Acquries an exclusive reference to the `num_pages` shared pages starting at `page_addr`.
    pub fn share_pages_with_umode(
        &self,
        page_addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<impl 'a + Iterator<Item = Page<Shareable>>> {
        Ok(self
            .inner
            .root
            .get_mapped_pages(
                page_addr,
                num_pages * PageSize::Size4k as u64,
                |addr, ps| {
                    // TODO: Support sharing of huge pages with U-mode.
                    if ps.is_huge() {
                        return false;
                    }
                    self.inner.page_tracker.is_shareable_page(
                        addr,
                        ps,
                        self.inner.page_owner_id,
                        MemType::Ram,
                    )
                },
            )
            .map_err(Error::Paging)?
            .map(|(addr, _)| {
                self.inner
                    .page_tracker
                    .get_shareable_page(addr, PageSize::Size4k, self.inner.page_owner_id)
                    .unwrap()
            }))
    }

    fn do_convert_pages<P: InvalidatedPhysPage>(
        &self,
        page_addr: GuestPageAddr,
        num_pages: u64,
    ) -> Result<()> {
        if self.inner.nesting >= MAX_PAGE_OWNERS - 1 {
            // We shouldn't bother converting pages if we won't be able to assign them.
            return Err(Error::NestingTooDeep);
        }
        if num_pages == 0 {
            return Err(Error::EmptyPageRange);
        }

        let version = self.inner.tlb_tracker.current_version();
        let invalidated = self
            .inner
            .root
            .invalidate_range(
                page_addr,
                num_pages * PageSize::Size4k as u64,
                |addr, ps| {
                    // Convertable pages should always be 4k given the host is
                    // exclusively mapped with 4k pages.
                    if ps.is_huge() {
                        return false;
                    }
                    self.inner.page_tracker.is_mapped_page(
                        addr,
                        ps,
                        self.inner.page_owner_id,
                        P::mem_type(),
                    )
                },
            )
            .map_err(Error::Paging)?;
        for (paddr, _) in invalidated {
            // Safety: We've verified the typing of the page and we must have unique
            // ownership since the page was mapped before it was invalidated.
            let page = unsafe { P::new(paddr) };
            // Unwrap ok: Page was mapped and has just been invalidated.
            self.inner.page_tracker.convert_page(page, version).unwrap();
        }

        Ok(())
    }

    /// Converts `num_pages` starting at guest physical address `page_addr` to confidential memory.
    pub fn convert_pages(&self, page_addr: GuestPageAddr, num_pages: u64) -> Result<()> {
        self.do_convert_pages::<Page<Invalidated>>(page_addr, num_pages)
    }

    /// Reclaims `num_pages` of confidential memory starting at guest physical address `page_addr`.
    pub fn reclaim_pages(&self, page_addr: GuestPageAddr, num_pages: u64) -> Result<()> {
        // TODO: Support reclaim of converted pages that haven't yet been fenced.
        let converted_pages = self.get_converted_pages(page_addr, PageSize::Size4k, num_pages)?;
        // Unwrap ok since the PTE for the page must have previously been invalid and all of
        // the intermediate page-tables must already have been populatd.
        let mapper = self
            .map_zero_pages(page_addr, PageSize::Size4k, num_pages)
            .unwrap();
        for (page, addr) in converted_pages.zip(page_addr.iter_from()) {
            // Unwrap ok since we know that it's a converted page.
            let mappable = self.inner.page_tracker.reclaim_page(page.clean()).unwrap();
            mapper.map_page(addr, mappable).unwrap();
        }
        Ok(())
    }

    /// Acquries an exclusive reference to the converted IMSIC page at `imsic_addr`.
    pub fn get_converted_imsic(
        &self,
        imsic_addr: GuestPageAddr,
    ) -> Result<LockedPageList<ImsicGuestPage<ConvertedClean>>> {
        self.do_get_converted_pages::<ImsicGuestPage<ConvertedClean>>(
            imsic_addr,
            PageSize::Size4k,
            1,
        )
    }

    /// Converts the guest interrupt file at `imsic_addr` to confidential.
    pub fn convert_imsic(&self, imsic_addr: GuestPageAddr) -> Result<()> {
        // Make sure it's actually an IMSIC address and that it's a guest (not supervisor) file.
        let geometry = self
            .inner
            .imsic_geometry
            .get()
            .ok_or(Error::NoImsicVirtualization)?;
        let location = geometry
            .addr_to_location(imsic_addr)
            .ok_or(Error::InvalidImsicLocation)?;
        if location.file() == ImsicFileId::Supervisor {
            return Err(Error::InvalidImsicLocation);
        }

        self.do_convert_pages::<ImsicGuestPage<Invalidated>>(imsic_addr, 1)?;

        // Unmap it from our MSI page table as well, if we have one.
        if let Some(iommu_context) = self.inner.iommu_context.get() {
            // Unwrap ok: we've already checked that `location` is valid and it must've been mapped
            // in the MSI page table if it was in the CPU page tables.
            iommu_context.msi_page_table.unmap(location).unwrap();
        }

        Ok(())
    }

    /// Reclaims the confidential guest interrupt file at `imsic_addr`.
    pub fn reclaim_imsic(&self, imsic_addr: GuestPageAddr) -> Result<()> {
        let mut converted = self.get_converted_imsic(imsic_addr)?;
        // Unwrap ok since the PTE for the page must have previously been invalid and all of
        // the intermediate page-tables must already have been populated.
        let mapper = self.map_imsic_pages(imsic_addr, 1).unwrap();
        // Unwrap ok since it must be a converted page.
        let mappable = converted
            .next()
            .and_then(|p| self.inner.page_tracker.reclaim_page(p).ok())
            .unwrap();
        // Unwrap ok since `imsic_addr` is within the range of the mapper.
        mapper.map_page(imsic_addr, mappable).unwrap();
        Ok(())
    }

    /// Invalidates the IMSIC interrupt file mapped at `imsic_addr` and begins the unassignment
    /// process.
    pub fn unassign_imsic_begin(&self, imsic_addr: GuestPageAddr) -> Result<()> {
        // Make sure it's actually an IMSIC address.
        let geometry = self
            .inner
            .imsic_geometry
            .get()
            .ok_or(Error::NoImsicVirtualization)?;
        let location = geometry
            .addr_to_location(imsic_addr)
            .ok_or(Error::InvalidImsicLocation)?;

        let invalidated = self
            .inner
            .root
            .invalidate_range(imsic_addr, PageSize::Size4k as u64, |addr, ps| {
                // Page covering the IMSIC interrupt file is always 4k.
                if ps.is_huge() {
                    return false;
                }
                self.inner.page_tracker.is_mapped_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Mmio(DeviceMemType::Imsic),
                )
            })
            .map_err(Error::Paging)?;
        for (paddr, _) in invalidated {
            // Safety: We've verified the typing of the page and we must have unique
            // ownership since the page was mapped before it was invalidated.
            let page: ImsicGuestPage<Invalidated> = unsafe { ImsicGuestPage::new(paddr) };
            // Unwrap ok: Page was mapped and has just been invalidated.
            self.inner
                .page_tracker
                .block_page(page, self.inner.tlb_tracker.current_version())
                .unwrap();
        }

        // Unmap it from our MSI page table as well, if we have one.
        if let Some(iommu_context) = self.inner.iommu_context.get() {
            // Unwrap ok: we've already checked that `location` is valid and it must've been mapped
            // in the MSI page table if it was in the CPU page tables.
            iommu_context.msi_page_table.unmap(location).unwrap();
        }

        Ok(())
    }

    /// Verifies that the TLB flush for the IMSIC interrupt file that was mapped at `imsic_addr`
    /// has been completed and completes unassignment of the page.
    pub fn unassign_imsic_end(&self, imsic_addr: GuestPageAddr) -> Result<()> {
        let version = self.inner.tlb_tracker.min_version();
        let unmapped = self
            .inner
            .root
            .unmap_range(imsic_addr, PageSize::Size4k as u64, |addr, ps| {
                // Page covering the IMSIC interrupt file is always 4k.
                if ps.is_huge() {
                    return false;
                }
                self.inner.page_tracker.is_blocked_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Mmio(DeviceMemType::Imsic),
                    Some(version),
                )
            })
            .map_err(Error::Paging)?;
        for (paddr, ps) in unmapped {
            if ps != PageSize::Size4k {
                return Err(Error::InvalidPageSize);
            }
            // Unwrap ok: Page was blocked and has just been unmapped.
            self.inner.page_tracker.remove_page(paddr, ps).unwrap();
        }
        Ok(())
    }

    // Begins removal of the `page`. Unlike `unassign_imsic_begin()`, this method does not
    // invalidate the address. It's only responsible for blocking the page in page_tracker.
    // For now we only use it for remapping purpose where the leaf entry is still valid but
    // the underlying address has been replaced.
    pub fn block_imsic_page(&self, page: ImsicGuestPage<Invalidated>) -> Result<()> {
        self.inner
            .page_tracker
            .block_page(page, self.inner.tlb_tracker.current_version())
            .map_err(Error::PageTracker)?;
        Ok(())
    }

    /// Verifies that the TLB flush for the IMSIC interrupt file that was mapped at `imsic_addr`
    /// has been completed and completes removal of the page.
    pub fn remove_imsic_page(&self, imsic_addr: SupervisorPageAddr) -> Result<()> {
        self.inner
            .page_tracker
            .remove_page(imsic_addr, PageSize::Size4k)
            .map_err(Error::PageTracker)?;
        Ok(())
    }

    /// Initiates a page conversion fence for this `VmPages` by incrementing the TLB version.
    pub fn initiate_fence(&self) -> Result<()> {
        self.inner.tlb_tracker.increment()?;
        // If we have an IOMMU context then we need to issue a fence there as well as our page
        // tables may be used for DMA translation.
        if let Some(iommu_context) = self.inner.iommu_context.get() {
            for iommu in Iommu::get_iommus() {
                iommu.fence(iommu_context.gscid, None);
            }
        }
        Ok(())
    }

    /// Invalidates a page range.
    pub fn block_pages(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        let invalidated = self
            .inner
            .root
            .invalidate_range(page_addr, len, |addr, ps| {
                self.inner.page_tracker.is_blockable_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Ram,
                )
            })
            .map_err(Error::Paging)?;
        let version = self.inner.tlb_tracker.current_version();
        for (paddr, page_size) in invalidated {
            // Safety: We've verified the typing of the page and its ownership
            // before it was invalidated.
            let page: Page<Invalidated> = unsafe { Page::new_with_size(paddr, page_size) };
            // Unwrap ok: Page was mapped or shared and has just been invalidated.
            self.inner.page_tracker.block_page(page, version).unwrap();
        }

        Ok(())
    }

    /// Unblocks previously invalidated page range.
    pub fn unblock_pages(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        let valid_pages = self
            .inner
            .root
            .validate_range(page_addr, len, |addr, ps| {
                self.inner.page_tracker.is_blocked_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Ram,
                    None,
                )
            })
            .map_err(Error::Paging)?;
        for (paddr, page_size) in valid_pages {
            // Unwrap ok: Page was blocked and has just been marked valid.
            self.inner
                .page_tracker
                .unblock_page(paddr, page_size)
                .unwrap();
        }

        Ok(())
    }

    // Promotes a page range, creating one page of the next larger page size.
    pub fn promote_page(&self, page_addr: GuestPageAddr, page_size: PageSize) -> Result<()> {
        let end = page_addr
            .checked_add_pages_with_size(1, page_size)
            .ok_or(Error::AddressOverflow)?;
        let regions = self.inner.regions.read();
        if !(regions.contains(page_addr, end, |r| {
            r == VmRegionType::Confidential
                || r == VmRegionType::ConfidentialRemovable
                || r == VmRegionType::Shared
                || r == VmRegionType::SharedRemovable
        })) {
            return Err(Error::VmRegionNotConfidentialOrShared);
        }
        let (pa, free_page) = self
            .inner
            .root
            .promote_page(page_addr, page_size, |addr, ps| {
                self.inner.page_tracker.is_blocked_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Ram,
                    Some(self.inner.tlb_tracker.min_version()),
                )
            })
            .map_err(Error::Paging)?;
        // Returns to the PTE pages pool the now-unused page table page
        self.inner.pte_pages.push(free_page);
        // Unwrap ok: Page was blocked and has just been promoted.
        self.inner.page_tracker.unblock_page(pa, page_size).unwrap();
        Ok(())
    }

    // Demotes a page, creating a page range of the next smaller page size.
    pub fn demote_page(&self, page_addr: GuestPageAddr, page_size: PageSize) -> Result<()> {
        let end = page_addr
            .checked_add_pages_with_size(ENTRIES_PER_PAGE, page_size)
            .ok_or(Error::AddressOverflow)?;
        let one_size_larger_page_size = page_size.size_up().ok_or(Error::InvalidPageSize)?;
        let regions = self.inner.regions.read();
        if !(regions.contains(page_addr, end, |r| {
            r == VmRegionType::Confidential
                || r == VmRegionType::ConfidentialRemovable
                || r == VmRegionType::Shared
                || r == VmRegionType::SharedRemovable
        })) {
            return Err(Error::VmRegionNotConfidentialOrShared);
        }
        // Allocate a free page to be able to insert a new page table page.
        let free_page = self
            .inner
            .pte_pages
            .pop()
            .ok_or(Error::InsufficientPtePages)?;
        let pa = self
            .inner
            .root
            .demote_page(page_addr, page_size, free_page, |addr, ps| {
                self.inner.page_tracker.is_blocked_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Ram,
                    Some(self.inner.tlb_tracker.min_version()),
                )
            })
            .map_err(Error::Paging)?;
        // Unwrap ok: Page was blocked and has just been demoted.
        self.inner
            .page_tracker
            .unblock_page(pa, one_size_larger_page_size)
            .unwrap();
        Ok(())
    }

    /// Removes previously invalidated page range.
    pub fn remove_pages(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        // Check the address range lies within a removable region of guest physical address space.
        let end = PageAddr::new(
            RawAddr::from(page_addr)
                .checked_increment(len)
                .ok_or(Error::AddressOverflow)?,
        )
        .ok_or(Error::UnalignedAddress)?;
        let regions = self.inner.regions.read();
        if !regions.contains(page_addr, end, |r| {
            r == VmRegionType::ConfidentialRemovable || r == VmRegionType::SharedRemovable
        }) {
            return Err(Error::VmRegionNotRemovable);
        }

        // Does this need to be TLB current_version instead of TLB min_version?
        let tlb_version = self.inner.tlb_tracker.min_version();
        // Now check each page is removable before it is unmapped.
        let unmapped = self
            .inner
            .root
            .unmap_range(page_addr, len, |addr, ps| {
                self.inner.page_tracker.is_blocked_page(
                    addr,
                    ps,
                    self.inner.page_owner_id,
                    MemType::Ram,
                    Some(tlb_version),
                )
            })
            .map_err(Error::Paging)?;
        for (paddr, page_size) in unmapped {
            // Unwrap ok: Page was blocked and has just been unmapped.
            self.inner
                .page_tracker
                .remove_page(paddr, page_size)
                .unwrap();
        }
        Ok(())
    }

    /// Pins `count` physically-contiguous pages starting at `page_addr` as shared pages, returning
    /// a `PinnedPages` structure that will release the pin when dropped. Used to share memory
    /// between a VM and the hypervisor.
    pub fn pin_shared_pages(&self, page_addr: GuestPageAddr, count: u64) -> Result<PinnedPages> {
        if count == 0 {
            return Err(Error::EmptyPageRange);
        }

        // Get the mapped pages, making sure they're contiguous.
        let mut prev_addr: Option<SupervisorPageAddr> = None;
        let pages = self
            .inner
            .root
            .get_mapped_pages(page_addr, count * PageSize::Size4k as u64, |addr, ps| {
                // Pinned shared pages are always 4k.
                if ps.is_huge() {
                    return false;
                }
                if let Some(p) = prev_addr
                    && p.checked_add_pages_with_size(1, ps) != Some(addr)
                {
                    false
                } else {
                    prev_addr = Some(addr);
                    self.inner.page_tracker.is_shareable_page(
                        addr,
                        ps,
                        self.inner.page_owner_id,
                        MemType::Ram,
                    )
                }
            })
            .map_err(Error::Paging)?
            .map(|(addr, _)| {
                self.inner
                    .page_tracker
                    .get_shareable_page::<Page<Shareable>>(
                        addr,
                        PageSize::Size4k,
                        self.inner.page_owner_id,
                    )
                    .unwrap()
            });

        let mut base_addr: Option<SupervisorPageAddr> = None;
        for (i, page) in pages.enumerate() {
            if i == 0 {
                base_addr = Some(page.addr());
            }

            // Unwrap ok: The page is guaranteed to be in a shareable state until the iterator is
            // destroyed.
            self.inner
                .page_tracker
                .share_page(page, self.inner.page_owner_id)
                .unwrap();
        }
        // Safety: the range of pages is shared and owned by the current VM.
        let pin = unsafe {
            // Unwrap ok, we know there must be at least one page.
            PinnedPages::new(
                SupervisorPageRange::new(base_addr.unwrap(), count),
                self.inner.page_tracker,
                self.inner.page_owner_id,
            )
        };
        Ok(pin)
    }

    ///  Pins `count` pages starting at `page_addr` as shared pages, and maps them in a U-mode
    /// slot. Returns a `GuestUmodeMapping` structure that will unmap and release the pin when
    /// dropped. Used to share memory between a VM and U-mode.
    pub fn map_in_umode_slot(
        &self,
        slot: UmodeSlotId,
        page_addr: GuestPageAddr,
        count: u64,
        slot_perm: UmodeSlotPerm,
    ) -> Result<GuestUmodeMapping> {
        let pages = self.share_pages_with_umode(page_addr, count)?;
        // TODO: Check that guest request for mapping it writable is consistent with the guest mappings.
        let mapper = PerCpu::this_cpu()
            .page_table()
            .umode_slot_mapper(slot, count, slot_perm)
            .map_err(Error::HypMap)?;
        let to_page_addr = mapper.vaddr();

        for (page, virt) in pages.zip(to_page_addr.iter_from()) {
            let phys = page.addr();
            // Unwrap ok: The page is guaranteed to be in a shareable state until the iterator is
            // destroyed.
            self.inner
                .page_tracker
                .share_page(page, self.inner.page_owner_id)
                .unwrap();

            // Safety: pages are guest-owned and marked as shared
            unsafe {
                // Unwrap okay: Address is in range and it is not mapped.
                mapper.map_addr(virt, phys).unwrap();
            }
        }

        // Safety: slot is valid and mapped with pages owned by current VM.
        let umode_mapping = unsafe {
            GuestUmodeMapping::new(
                slot,
                count,
                self.inner.page_tracker,
                self.inner.page_owner_id,
            )
        };
        Ok(umode_mapping)
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
    ) -> ActiveVmPages<'a, T> {
        ActiveVmPages::new(self.clone(), vmid, prev_tlb_version)
    }
}

impl<'a, T: GuestStagePagingMode> From<FinalizedVmPages<'a, T>> for AnyVmPages<'a, T> {
    fn from(src: FinalizedVmPages<'a, T>) -> AnyVmPages<'a, T> {
        VmPagesRef::new(src.inner)
    }
}

/// Represents the address space of a initializing VM. Used to expose the operations that are
/// possible on a VM in the process of construction, including mapping and measuring pages.
pub type InitializingVmPages<'a, T> = VmPagesRef<'a, T, VmStateInitializing>;

impl<'a, T: GuestStagePagingMode> InitializingVmPages<'a, T> {
    /// Sets the IMSIC geometry for this VM by adding the memory regions that will be occupied by
    /// IMSIC interrupt files.
    pub fn set_imsic_geometry(&self, geometry: GuestImsicGeometry) -> Result<()> {
        let to_set = geometry.clone();
        let actual = self
            .inner
            .imsic_geometry
            .try_call_once(|| {
                let mut regions = self.inner.regions.write();
                for range in geometry.group_ranges() {
                    let end = range
                        .base()
                        .checked_add_pages(range.num_pages())
                        .ok_or(Error::AddressOverflow)?;
                    regions.add(range.base(), end, VmRegionType::Imsic)?;
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
            .inner
            .imsic_geometry
            .get()
            .ok_or(Error::NoImsicVirtualization)?
            .clone();
        let msi_pt = MsiPageTable::new(
            msi_table_pages,
            imsic_geometry,
            Imsic::get().phys_geometry(),
            self.inner.page_tracker,
            self.inner.page_owner_id,
        )
        .map_err(Error::CreatingMsiPageTable)?;
        let iommu_context = VmIommuContext::new(msi_pt)?;
        let gscid = iommu_context.gscid;
        let set_gscid = self.inner.iommu_context.call_once(|| iommu_context).gscid;
        // Check if the `VmIommuContext` that was set was actually the one we created.
        if gscid != set_gscid {
            return Err(Error::IommuContextAlreadySet);
        }
        Ok(())
    }

    /// Adds a confidential memory region of `len` bytes starting at `page_addr` to this VM's
    /// address space.
    pub fn add_confidential_memory_region(&self, page_addr: GuestPageAddr, len: u64) -> Result<()> {
        self.do_add_region(page_addr, len, VmRegionType::Confidential)
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
        page_size: PageSize,
        count: u64,
    ) -> Result<MeasuredPagesMapper<'a, T>> {
        self.do_map_pages(page_addr, page_size, count, |r| {
            r == VmRegionType::Confidential
        })
    }

    /// Attaches the given PCI device to this VM by enabling DMA translation via the IOMMU using
    /// this VM's page tables.
    pub fn attach_pci_device(&self, dev: &mut PciDevice) -> Result<()> {
        let iommu_context = self.inner.iommu_context.get().ok_or(Error::NoIommu)?;
        let iommu = Iommu::get_for_device(dev).ok_or(Error::NoIommu)?;
        let msi_pt = iommu
            .supports_msi_page_tables()
            .then_some(&iommu_context.msi_page_table);
        iommu
            .attach_pci_device(dev, &self.inner.root, msi_pt, iommu_context.gscid)
            .map_err(Error::AttachingDevice)
    }
}

impl<'a, T: GuestStagePagingMode> From<InitializingVmPages<'a, T>> for AnyVmPages<'a, T> {
    fn from(src: InitializingVmPages<'a, T>) -> AnyVmPages<'a, T> {
        VmPagesRef::new(src.inner)
    }
}
