// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::{marker::PhantomData, ops::Deref};
use data_measure::data_measure::DataMeasure;
use data_measure::sha256::Sha256Measure;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{
    tlb, GuestStagePageTable, LockedPageList, PageList, PageTracker, TlbVersion, MAX_PAGE_OWNERS,
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
    GuestId(riscv_page_tables::PageTrackingError),
    InsufficientPtePageStorage,
    Paging(riscv_page_tables::PageTableError),
    PageFaultHandling, // TODO - individual errors from sv48x4
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
}

pub type Result<T> = core::result::Result<T, Error>;

/// The minimum number of pages required to track free page-table pages.
pub const MIN_PTE_VEC_PAGES: u64 = 1;

/// The base number of state pages required to be donated for creating a new VM: pages for the
/// page-table page vector, and one page to hold the VM state itself.
pub const TVM_STATE_PAGES: u64 = MIN_PTE_VEC_PAGES + 1;

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

    /// Uses `copy_fn` to copy `len` bytes between `guest_addr` and `host_ptr`. Attempts to handle
    /// any page faults that occur during the copy.
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
        let mut copied = 0;
        let mut cur_gpa = guest_addr;
        let mut cur_ptr = host_ptr;
        while copied < len {
            this_cpu.enter_guest_memcpy();
            let bytes = copy_fn(cur_gpa, cur_ptr, len - copied);
            this_cpu.exit_guest_memcpy();
            copied += bytes;
            if copied < len {
                // Partial copy: we encountered a page fault. See if we can handle it and retry.
                cur_gpa = cur_gpa
                    .checked_increment(bytes as u64)
                    .ok_or(Error::AddressOverflow)?;
                self.vm_pages.handle_page_fault(cur_gpa)?;

                // Safety: cur_ptr + bytes must be less than the original host_ptr + len.
                cur_ptr = unsafe { cur_ptr.add(bytes) };
            }
        }
        Ok(())
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
        let new_owner = to.page_owner_id();
        for (dirty, (src_addr, to_addr)) in
            converted_pages.zip(src_addr.iter_from().zip(to_addr.iter_from()))
        {
            let initialized = dirty
                .try_initialize(|bytes| self.copy_from_guest(bytes, src_addr.into()))
                .map_err(|(e, _)| e)?;
            // Unwrap ok since we've guaranteed there's space for another owner.
            let mappable = self
                .page_tracker
                .assign_page_for_mapping(initialized, new_owner)
                .unwrap();
            to.add_measured_4k_page(to_addr, mappable)?;
        }
        Ok(count)
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
    // How many nested TVMs deep this VM is, with 0 being the host.
    nesting: usize,
    // Locking order: `root` -> `measurement` -> `pte_pages`
    root: Mutex<T>,
    measurement: Mutex<Sha256Measure>,
    pte_pages: Mutex<PageVec<Page<InternalClean>>>,
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
        self.root.lock().get_root_address()
    }

    /// Returns the global page tracking structure.
    pub fn page_tracker(&self) -> PageTracker {
        self.page_tracker.clone()
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
        let mut root = self.root.lock();
        root.get_converted_range::<Page<ConvertedDirty>>(
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

        let invalidated_pages = {
            let mut root = self.root.lock();
            root.invalidate_range::<Page<Invalidated>>(page_addr, PageSize::Size4k, num_pages)
        }
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
        let mut pte_pages = self.pte_pages.lock();
        for (page, addr) in converted_pages.zip(page_addr.iter_from()) {
            // Unwrap ok since we know that it's a converted page.
            let mappable = self.page_tracker.reclaim_page(page.clean()).unwrap();
            {
                let mut root = self.root.lock();
                // Unwrap ok since the PTE for the page must have previously been invalid and all of
                // the intermediate page-tables must already have been populatd.
                root.map_page(addr, mappable, &mut || pte_pages.pop())
                    .unwrap();
            }
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
        let guest_root = T::new(guest_root_pages, id, self.page_tracker.clone()).unwrap();

        let mut state_pages = self.assign_state_pages_for(state_pages, id);
        let state_page = state_pages.next().unwrap();
        let pte_vec_pages = SequentialPages::from_pages(state_pages).unwrap();

        let vcpu_pages =
            SequentialPages::from_pages(self.assign_state_pages_for(vcpu_pages, id)).unwrap();

        Ok((
            Vm::new(
                VmPages::new(guest_root, pte_vec_pages, self.nesting + 1),
                VmCpus::new(id, vcpu_pages).unwrap(),
            ),
            state_page,
        ))
    }

    /// Adds pages to be used for building page table entries
    pub fn add_pte_pages_builder(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &VmPages<T, VmStateInitializing>,
    ) -> Result<()> {
        let converted_pages = self.get_converted_pages(from_addr, count)?;
        let new_owner = to.page_owner_id();
        for page in converted_pages {
            // Unwrap ok since we've guaranteed the page is assignable.
            let assigned = self
                .page_tracker
                .assign_page_for_internal_state(page.clean(), new_owner)
                .unwrap();
            to.add_pte_page(assigned)?;
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
        let new_owner = to.page_owner_id();
        for (page, guest_addr) in converted_pages.zip(to_addr.iter_from()) {
            // Unwrap ok since we've guaranteed there's space for another owner.
            let mappable = self
                .page_tracker
                .assign_page_for_mapping(page.clean(), new_owner)
                .unwrap();
            to.add_4k_page(guest_addr, mappable)?;
        }
        Ok(count)
    }

    /// Handles a page fault for the given address.
    pub fn handle_page_fault(&self, addr: GuestPhysAddr) -> Result<()> {
        let mut root = self.root.lock();
        if root.do_fault(addr) {
            Ok(())
        } else {
            Err(Error::PageFaultHandling)
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
    /// Creates a new `VmPages` from the given root page table, using `pte_vec_page` for a vector
    /// of page-table pages.
    pub fn new(root: T, pte_vec_pages: SequentialPages<InternalClean>, nesting: usize) -> Self {
        Self {
            page_owner_id: root.page_owner_id(),
            page_tracker: root.page_tracker(),
            tlb_tracker: TlbTracker::new(),
            nesting,
            root: Mutex::new(root),
            measurement: Mutex::new(Sha256Measure::new()),
            pte_pages: Mutex::new(PageVec::from(pte_vec_pages)),
            phantom: PhantomData,
        }
    }

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&self, page: Page<InternalClean>) -> Result<()> {
        if page.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(page.size()));
        }
        let mut pte_pages = self.pte_pages.lock();
        pte_pages
            .try_reserve(1)
            .map_err(|_| Error::InsufficientPtePageStorage)?;
        pte_pages.push(page);
        Ok(())
    }

    /// Maps a page into the guest's address space and measures it.
    pub fn add_measured_4k_page<S, M>(&self, to_addr: GuestPageAddr, page: Page<S>) -> Result<()>
    where
        S: Mappable<M>,
        M: MeasureRequirement,
    {
        if page.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(page.size()));
        }
        let mut root = self.root.lock();
        let mut measurement = self.measurement.lock();
        let mut pte_pages = self.pte_pages.lock();
        root.map_page_with_measurement(to_addr, page, &mut || pte_pages.pop(), &mut *measurement)
            .map_err(Error::Paging)
    }

    /// Maps an unmeasured page into the guest's address space.
    pub fn add_4k_page<P>(&self, to_addr: GuestPageAddr, page: P) -> Result<()>
    where
        P: MappablePhysPage<MeasureOptional>,
    {
        if page.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(page.size()));
        }
        let mut root = self.root.lock();
        let mut pte_pages = self.pte_pages.lock();
        root.map_page(to_addr, page, &mut || pte_pages.pop())
            .map_err(Error::Paging)
    }

    /// Consumes this `VmPages`, returning a finalized one.
    pub fn finalize(self) -> VmPages<T, VmStateFinalized> {
        VmPages {
            page_owner_id: self.page_owner_id,
            page_tracker: self.page_tracker,
            tlb_tracker: self.tlb_tracker,
            nesting: self.nesting,
            root: self.root,
            measurement: self.measurement,
            pte_pages: self.pte_pages,
            phantom: PhantomData,
        }
    }
}
