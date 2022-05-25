// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use core::alloc::Allocator;
use core::marker::PhantomData;
use data_measure::data_measure::DataMeasure;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{GuestStagePageTable, HypPageAlloc, PageState};
use riscv_pages::{
    CleanPage, GuestPageAddr, GuestPhysAddr, MemType, Page, PageOwnerId, PageSize, PhysPage,
    SeqPageIter, SequentialPages, SupervisorPageAddr,
};
use spin::Mutex;

use crate::sha256_measure::Sha256Measure;

#[derive(Debug)]
pub enum Error {
    GuestId(riscv_page_tables::PageTrackingError),
    InsufficientPtePageStorage,
    Paging(riscv_page_tables::PageTableError),
    PageFaultHandling, // TODO - individual errors from sv48x4
    SettingOwner(riscv_page_tables::PageTrackingError),
    // Vm pages must be aligned to 16k to be used for sv48x4 mappings
    UnalignedVmPages(GuestPageAddr),
    UnownedPage(GuestPageAddr),
    MeasurementBufferTooSmall,
}

pub type Result<T> = core::result::Result<T, Error>;

pub enum VmPagesBuilding {}
pub enum VmPagesConstructed {}

/// VmPages is the single management point for memory used by virtual machines.
///
/// After initial setup all memory not used for Hypervisor purposes is managed by a VmPages
/// instance. Rules around sharing and isolating memory are enforced by this module.
///
/// Machines are allowed to donate pages to child machines and to share donated pages with parent
/// machines.
pub struct VmPages<T: GuestStagePageTable, S = VmPagesConstructed> {
    page_owner_id: PageOwnerId,
    phys_pages: PageState,
    // Locking order: `root` must be locked before `measurement`
    root: Mutex<T>,
    measurement: Mutex<Sha256Measure>,
    phantom: PhantomData<S>,
}

impl<T: GuestStagePageTable, S> VmPages<T, S> {
    /// Creates a new `VmPages` from the given root page table.
    fn new(root: T) -> Self {
        Self {
            page_owner_id: root.page_owner_id(),
            phys_pages: root.phys_pages(),
            root: Mutex::new(root),
            measurement: Mutex::new(Sha256Measure::new()),
            phantom: PhantomData,
        }
    }

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
    pub fn phys_pages(&self) -> PageState {
        self.phys_pages.clone()
    }
}

impl<T: GuestStagePageTable> VmPages<T, VmPagesConstructed> {
    /// Creates a `GuestRootBuilder` from pages owned by `self`.
    /// The `GuestRootBuilder` is used to build a guest VM owned by `self`'s root.page_owner_id().
    pub fn create_guest_root_builder(
        &self,
        from_addr: GuestPageAddr,
    ) -> Result<(GuestRootBuilder<T>, Page)> {
        if (from_addr.bits() as *const u64).align_offset(T::TOP_LEVEL_ALIGN as usize) != 0 {
            return Err(Error::UnalignedVmPages(from_addr));
        }
        let id = self.phys_pages.add_active_guest().map_err(Error::GuestId)?;
        let mut root = self.root.lock();
        let mut clean_pages = root
            .invalidate_range(from_addr, 6)
            .map_err(Error::Paging)?
            .map(CleanPage::from)
            .map(Page::from)
            .map(|p| {
                self.phys_pages.set_page_owner(p.addr(), id).unwrap();
                p
            });

        // Can't fail if enough aligned pages are provided(checked above).
        let guest_root_pages = SequentialPages::from_pages(clean_pages.by_ref().take(4)).unwrap();
        let guest_root = T::new(guest_root_pages, id, self.phys_pages.clone()).unwrap();
        let pte_page = clean_pages.next().unwrap();
        let state_page = clean_pages.next().unwrap();

        Ok((GuestRootBuilder::new(guest_root, pte_page), state_page))
    }

    /// Adds pages to be used for building page table entries
    pub fn add_pte_pages_builder(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &GuestRootBuilder<T>,
    ) -> Result<()> {
        let mut root = self.root.lock();
        let clean_pages = root
            .invalidate_range(from_addr, count)
            .map_err(Error::Paging)?
            .map(CleanPage::from)
            .map(Page::from);
        for page in clean_pages {
            self.phys_pages
                .set_page_owner(page.addr(), to.page_owner_id())
                .map_err(Error::SettingOwner)?;
            to.add_pte_page(page)?;
        }
        Ok(())
    }

    /// Add data pages to the given builder
    // TODO add other page sizes
    pub fn add_4k_pages_builder(
        &self,
        from_addr: GuestPageAddr,
        count: u64,
        to: &GuestRootBuilder<T>,
        to_addr: GuestPageAddr,
        measure_preserve: bool,
    ) -> Result<u64> {
        let mut root = self.root.lock();
        let unmapped_pages = root
            .invalidate_range::<Page>(from_addr, count)
            .map_err(Error::Paging)?;
        for (unmapped_page, guest_addr) in unmapped_pages.zip(to_addr.iter_from()) {
            let page = unmapped_page.to_page();
            self.phys_pages
                .set_page_owner(page.addr(), to.page_owner_id())
                .map_err(Error::SettingOwner)?;
            if measure_preserve {
                to.add_data_page(guest_addr, page)?;
            } else {
                to.add_zero_page(guest_addr, page)?;
            }
        }
        Ok(count)
    }

    /// Remove pages owned and return them to the previous owner.
    pub fn remove_4k_pages(&self, from_addr: GuestPageAddr, count: u64) -> Result<u64> {
        let mut root = self.root.lock();
        let clean_pages = root
            .unmap_range(from_addr, count)
            .map_err(Error::Paging)?
            .map(CleanPage::from)
            .map(Page::from);
        for (page, guest_addr) in clean_pages.zip(from_addr.iter_from()) {
            let owner = self
                .phys_pages
                .pop_owner(page.addr())
                .map_err(|_| Error::UnownedPage(guest_addr))?;
            if owner != self.page_owner_id {
                return Err(Error::UnownedPage(guest_addr));
            }
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

    pub fn write_measurements_to_guest_owned_page(&self, gpa: GuestPhysAddr) -> Result<usize> {
        let mut root = self.root.lock();
        let measurement = self.measurement.lock();
        let bytes = measurement.get_measurement();
        root.write_mapped_page(gpa, 0, bytes)
            .map(|_| bytes.len())
            .map_err(Error::Paging)
    }

    /// Writes `bytes` to the specified guest address.
    pub fn write_to_guest_owned_page(&self, gpa: GuestPhysAddr, bytes: &[u8]) -> Result<usize> {
        let mut root = self.root.lock();
        root.write_mapped_page(gpa, 0, bytes)
            .map(|_| bytes.len())
            .map_err(Error::Paging)
    }
}

impl<T: GuestStagePageTable> VmPages<T, VmPagesBuilding> {
    /// Maps a page into the guest's address space and measures it.
    fn add_measured_4k_page(
        &self,
        to_addr: GuestPageAddr,
        page: Page,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        let mut root = self.root.lock();
        let mut measurement = self.measurement.lock();
        root.map_page_with_measurement(to_addr, page, get_pte_page, &mut *measurement)
            .map_err(Error::Paging)
    }

    /// Maps an unmeasured page into the guest's address space.
    fn add_4k_page<P: PhysPage>(
        &self,
        to_addr: GuestPageAddr,
        page: P,
        get_pte_page: &mut dyn FnMut() -> Option<Page>,
    ) -> Result<()> {
        let mut root = self.root.lock();
        root.map_page(to_addr, page, get_pte_page)
            .map_err(Error::Paging)
    }
}

/// Keeps the state of the host's pages.
pub struct HostRootPages<T: GuestStagePageTable> {
    inner: VmPages<T, VmPagesConstructed>,
}

impl<T: GuestStagePageTable> HostRootPages<T> {
    pub fn into_inner(self) -> VmPages<T> {
        self.inner
    }
}

/// Builder used to construct the page management structure for the host.
///
/// Note that HostRootBuilder enforces that the GPA -> HPA mappings that are created always map
/// a T::TOP_LEVEL_ALIGN-aligned chunk.
pub struct HostRootBuilder<T: GuestStagePageTable> {
    inner: VmPages<T, VmPagesBuilding>,
    pte_pages: SeqPageIter,
}

impl<T: GuestStagePageTable> HostRootBuilder<T> {
    /// To be used to create the initial `HostRootPages` for the host VM.
    pub fn from_hyp_mem<A: Allocator>(
        mut hyp_mem: HypPageAlloc<A>,
        host_gpa_size: u64,
    ) -> (Vec<SequentialPages, A>, Self) {
        let root_table_pages = hyp_mem.take_pages_with_alignment(4, T::TOP_LEVEL_ALIGN);
        let num_pte_pages = T::max_pte_pages(host_gpa_size / PageSize::Size4k as u64);
        let pte_pages = hyp_mem.take_pages(num_pte_pages as usize).into_iter();

        let (phys_pages, host_pages) = PageState::from(hyp_mem, T::TOP_LEVEL_ALIGN);
        let root = T::new(root_table_pages, PageOwnerId::host(), phys_pages).unwrap();

        (
            host_pages,
            Self {
                inner: VmPages::new(root),
                pte_pages,
            },
        )
    }

    /// Adds data pages that are measured and mapped to the page tables for the host.
    pub fn add_measured_pages<I>(mut self, to_addr: GuestPageAddr, pages: I) -> Self
    where
        I: Iterator<Item = Page>,
    {
        let pte_pages = &mut self.pte_pages;
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(vm_addr.size(), page.addr().size());
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            self.inner
                .phys_pages
                .set_page_owner(page.addr(), self.inner.page_owner_id())
                .unwrap();
            self.inner
                .add_measured_4k_page(vm_addr, page, &mut || pte_pages.next())
                .unwrap();
        }
        self
    }

    /// Add pages which need not be measured to the host page tables.
    pub fn add_pages<I, P>(mut self, to_addr: GuestPageAddr, pages: I) -> Self
    where
        I: Iterator<Item = P>,
        P: PhysPage,
    {
        let pte_pages = &mut self.pte_pages;
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(vm_addr.size(), page.addr().size());
            if P::mem_type() == MemType::Ram {
                // GPA -> SPA mappings need to match T::TOP_LEVEL_ALIGN alignment for RAM pages.
                assert_eq!(
                    vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                    page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
                );
            }
            self.inner
                .phys_pages
                .set_page_owner(page.addr(), self.inner.page_owner_id())
                .unwrap();
            self.inner
                .add_4k_page(vm_addr, page, &mut || pte_pages.next())
                .unwrap();
        }

        self
    }

    /// Returns the host root pages as configured with data and zero pages.
    pub fn create_host(self) -> HostRootPages<T> {
        let root = self.inner.root.into_inner();
        HostRootPages {
            inner: VmPages::new(root),
        }
    }
}

/// Builder used to configure `VmPages` for a new guest VM.
pub struct GuestRootBuilder<T: GuestStagePageTable> {
    inner: VmPages<T, VmPagesBuilding>,
    pte_pages: Mutex<PageVec<Page>>,
}

impl<T: GuestStagePageTable> GuestRootBuilder<T> {
    /// Create a new `GuestRootBuilder` with `root` as the backing page table and `pte_page` used to
    /// hose a Vec of pte pages.
    pub fn new(root: T, pte_page: Page) -> Self {
        Self {
            inner: VmPages::new(root),
            pte_pages: Mutex::new(PageVec::from(SequentialPages::from(pte_page))),
        }
    }

    /// Return the page owner ID these pages will be assigned to.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.inner.page_owner_id()
    }

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&self, page: Page) -> Result<()> {
        let mut pte_pages = self.pte_pages.lock();
        pte_pages
            .try_reserve(1)
            .map_err(|_| Error::InsufficientPtePageStorage)?;
        pte_pages.push(page);
        Ok(())
    }

    /// Add a measured data page for the guest to use.
    /// Currently only supports 4k pages.
    pub fn add_data_page(&self, gpa: GuestPageAddr, page: Page) -> Result<()> {
        let mut pte_pages = self.pte_pages.lock();
        self.inner
            .add_measured_4k_page(gpa, page, &mut || pte_pages.pop())
    }

    /// Add a zeroed data page for the guest to use.
    /// Currently only supports 4k pages.
    pub fn add_zero_page(&self, gpa: GuestPageAddr, page: Page) -> Result<()> {
        let mut pte_pages = self.pte_pages.lock();
        self.inner.add_4k_page(gpa, page, &mut || pte_pages.pop())
    }

    /// Consumes the builder and returns the guest's VmPages struct.
    pub fn create_pages(self) -> VmPages<T, VmPagesConstructed> {
        let root = self.inner.root.into_inner();
        VmPages::new(root)
    }

    /// Copies the current measurement for the builder into `dest`.
    pub fn get_measurement(&self, dest: &mut [u8]) -> Result<()> {
        self.inner.get_measurement(dest)
    }
}
