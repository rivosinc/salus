// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use data_measure::data_measure::DataMeasure;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{GuestStagePageTable, PageState};
use riscv_pages::{
    CleanPage, GuestPageAddr, GuestPhysAddr, Page, PageOwnerId, PageSize, PhysPage,
    SequentialPages, SupervisorPageAddr,
};
use spin::Mutex;

use crate::sha256_measure::Sha256Measure;
use crate::vm::{Vm, VmStateFinalized, VmStateInitializing};
use crate::vm_cpu::{VmCpus, VM_CPUS_PAGES};

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
    UnsupportedPageSize(PageSize),
    MeasurementBufferTooSmall,
}

pub type Result<T> = core::result::Result<T, Error>;

/// The minimum number of pages required to track free page-table pages.
pub const MIN_PTE_VEC_PAGES: u64 = 1;

/// The number of pages required to be donated for creating a new VM.
///
/// TODO: Expose this to the host via a TEECALL.
pub const NEW_GUEST_PAGES: u64 = 4 + VM_CPUS_PAGES + MIN_PTE_VEC_PAGES + 1;

/// VmPages is the single management point for memory used by virtual machines.
///
/// After initial setup all memory not used for Hypervisor purposes is managed by a VmPages
/// instance. Rules around sharing and isolating memory are enforced by this module.
///
/// Machines are allowed to donate pages to child machines and to share donated pages with parent
/// machines.
pub struct VmPages<T: GuestStagePageTable, S = VmStateFinalized> {
    page_owner_id: PageOwnerId,
    phys_pages: PageState,
    // Locking order: `root` -> `measurement` -> `pte_pages`
    root: Mutex<T>,
    measurement: Mutex<Sha256Measure>,
    pte_pages: Mutex<PageVec<Page>>,
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
    pub fn phys_pages(&self) -> PageState {
        self.phys_pages.clone()
    }
}

impl<T: GuestStagePageTable> VmPages<T, VmStateFinalized> {
    /// Creates a `GuestRootBuilder` from pages owned by `self`.
    /// The `GuestRootBuilder` is used to build a guest VM owned by `self`'s root.page_owner_id().
    pub fn create_guest_vm(
        &self,
        from_addr: GuestPageAddr,
    ) -> Result<(Vm<T, VmStateInitializing>, Page)> {
        if from_addr.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(from_addr.size()));
        }
        if (from_addr.bits() as *const u64).align_offset(T::TOP_LEVEL_ALIGN as usize) != 0 {
            return Err(Error::UnalignedVmPages(from_addr));
        }
        let id = self.phys_pages.add_active_guest().map_err(Error::GuestId)?;
        let mut root = self.root.lock();
        let mut clean_pages = root
            .invalidate_range(from_addr, NEW_GUEST_PAGES)
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
        let pte_vec_pages =
            SequentialPages::from_pages(clean_pages.by_ref().take(MIN_PTE_VEC_PAGES as usize))
                .unwrap();
        let state_page = clean_pages.next().unwrap();
        // TODO: Make the max number of vCPUs configurable at TVM creation time so the host doesn't
        // have to unconditionally donate enough pages to support MAX_CPUS.
        let vcpu_pages =
            SequentialPages::from_pages(clean_pages.by_ref().take(VM_CPUS_PAGES as usize)).unwrap();

        Ok((
            Vm::new(
                VmPages::new(guest_root, pte_vec_pages),
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
        if from_addr.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(from_addr.size()));
        }
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
        to: &VmPages<T, VmStateInitializing>,
        to_addr: GuestPageAddr,
        measure_preserve: bool,
    ) -> Result<u64> {
        if from_addr.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(from_addr.size()));
        }
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
                to.add_measured_4k_page(guest_addr, page)?;
            } else {
                to.add_4k_page(guest_addr, page)?;
            }
        }
        Ok(count)
    }

    /// Remove pages owned and return them to the previous owner.
    pub fn remove_4k_pages(&self, from_addr: GuestPageAddr, count: u64) -> Result<u64> {
        if from_addr.size() != PageSize::Size4k {
            return Err(Error::UnsupportedPageSize(from_addr.size()));
        }
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

impl<T: GuestStagePageTable> VmPages<T, VmStateInitializing> {
    /// Creates a new `VmPages` from the given root page table, using `pte_vec_page` for a vector
    /// of page-table pages.
    pub fn new(root: T, pte_vec_pages: SequentialPages) -> Self {
        Self {
            page_owner_id: root.page_owner_id(),
            phys_pages: root.phys_pages(),
            root: Mutex::new(root),
            measurement: Mutex::new(Sha256Measure::new()),
            pte_pages: Mutex::new(PageVec::from(pte_vec_pages)),
            phantom: PhantomData,
        }
    }

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&self, page: Page) -> Result<()> {
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
    pub fn add_measured_4k_page(&self, to_addr: GuestPageAddr, page: Page) -> Result<()> {
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
    pub fn add_4k_page<P: PhysPage>(&self, to_addr: GuestPageAddr, page: P) -> Result<()> {
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
            phys_pages: self.phys_pages,
            root: self.root,
            measurement: self.measurement,
            pte_pages: self.pte_pages,
            phantom: PhantomData,
        }
    }
}
