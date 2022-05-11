// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use core::alloc::Allocator;
use riscv_page_tables::{HypPageAlloc, PageState, PlatformPageTable};
use riscv_pages::{
    AlignedPageAddr4k, CleanPage, Page4k, PageOwnerId, PageSize, PageSize4k, PhysAddr, SeqPageIter,
    SequentialPages, SequentialPages4k, UnmappedPage,
};

use page_collections::page_vec::PageVec;

use crate::data_measure::DataMeasure;
use crate::sha256_measure::Sha256Measure;

#[derive(Debug)]
pub enum Error {
    GuestId(riscv_page_tables::PageTrackingError),
    InvalidRange,
    InsufficientPtePageStorage,
    Mapping4kPage(riscv_page_tables::PageTableError),
    Non4kPteEntry,
    PageFaultHandling, // TODO - individual errors from sv48x4
    SettingOwner(riscv_page_tables::PageTrackingError),
    // Vm pages must be aligned to 16k to be used for sv48x4 mappings
    UnalignedVmPages(AlignedPageAddr4k),
    UnownedPage(AlignedPageAddr4k),
}

pub type Result<T> = core::result::Result<T, Error>;

/// VmPages is the single management point for memory used by virtual machines.
///
/// After initial setup all memory not used for Hypervisor purposes is managed by a VmPages
/// instance. Rules around sharing and isolating memory are enforced by this module.
///
/// Machines are allowed to donate pages to child machines and to share donated pages with parent
/// machines.
pub struct VmPages<T: PlatformPageTable> {
    root: T,
    measurement: Sha256Measure,
}

impl<T: PlatformPageTable> VmPages<T> {
    /// Returns the `PageOwnerId` associated with the pages contained in this machine.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.root.page_owner_id()
    }

    pub fn get_measurement(&self) -> &[u8] {
        self.measurement.get_measurement()
    }

    /// Creates a `GuestRootBuilder` from pages owned by `self`.
    /// The `GuestRootBuilder` is used to build a guest VM owned by `self`'s root.page_owner_id().
    pub fn create_guest_root_builder(
        &mut self,
        from_addr: AlignedPageAddr4k,
    ) -> Result<(GuestRootBuilder<T>, Page4k)> {
        if (from_addr.bits() as *const u64).align_offset(16 * 1024) != 0 {
            return Err(Error::UnalignedVmPages(from_addr));
        }
        let mut phys_pages = self.root.phys_pages();
        let pp_clone = phys_pages.clone(); // Because iterator borrows `phys_pages`
        let id = phys_pages.add_active_guest().map_err(Error::GuestId)?;
        let mut clean_pages = self
            .root
            .invalidate_range(from_addr, 6)
            .ok_or(Error::InvalidRange)?
            .map(UnmappedPage::from)
            .map(|up| up.unwrap_4k())
            .map(|p| {
                phys_pages.set_page_owner(p.addr(), id).unwrap();
                p
            });

        // Can't fail if enough aligned pages are provided(checked above).
        let root_pages = SequentialPages::from_pages(clean_pages.by_ref().take(4)).unwrap();
        let root = T::new(root_pages, id, pp_clone).unwrap();
        let pte_page = clean_pages.next().unwrap();
        let state_page = clean_pages.next().unwrap();

        Ok((GuestRootBuilder::new(root, pte_page), state_page))
    }

    /// Adds pages to be used for building page table entries
    pub fn add_pte_pages_builder(
        &mut self,
        from_addr: AlignedPageAddr4k,
        count: u64,
        to: &mut GuestRootBuilder<T>,
    ) -> Result<()> {
        let mut phys_pages = self.root.phys_pages();
        let clean_pages = self
            .root
            .invalidate_range(from_addr, count)
            .ok_or(Error::InvalidRange)?
            .map(CleanPage::from);
        for clean_page in clean_pages {
            let unmapped_page: UnmappedPage = clean_page.into();
            let page = unmapped_page.ok4k_or(Error::Non4kPteEntry)?;
            phys_pages
                .set_page_owner(page.addr(), to.page_owner_id())
                .map_err(Error::SettingOwner)?;
            to.add_pte_page(page)?;
        }
        Ok(())
    }

    /// Add data pages to the given builder
    // TODO add other page sizes
    pub fn add_4k_pages_builder(
        &mut self,
        from_addr: AlignedPageAddr4k,
        count: u64,
        to: &mut GuestRootBuilder<T>,
        to_addr: AlignedPageAddr4k,
        measure_preserve: bool,
    ) -> Result<u64> {
        let mut phys_pages = self.root.phys_pages();
        let unmapped_pages = self
            .root
            .invalidate_range(from_addr, count)
            .ok_or(Error::InvalidRange)?;
        for (unmapped_page, guest_addr) in unmapped_pages.zip(to_addr.iter_from()) {
            let page = unmapped_page.ok4k_or(Error::Non4kPteEntry)?;
            phys_pages
                .set_page_owner(page.addr(), to.page_owner_id())
                .map_err(Error::SettingOwner)?;
            if measure_preserve {
                to.add_data_page(guest_addr.bits(), page)?;
            } else {
                to.add_zero_page(guest_addr.bits(), page)?;
            }
        }
        Ok(count)
    }

    /// Remove pages owned and return them to the previous owner.
    pub fn remove_4k_pages(&mut self, from_addr: AlignedPageAddr4k, count: u64) -> Result<u64> {
        let owner_id = self.root.page_owner_id();
        let mut pp_clone = self.root.phys_pages();
        let clean_pages = self
            .root
            .unmap_range(from_addr, count)
            .ok_or(Error::InvalidRange)?
            .map(CleanPage::from);
        for clean_page in clean_pages {
            let unmapped_page: UnmappedPage = clean_page.into();
            let page = unmapped_page.ok4k_or(Error::Non4kPteEntry)?;
            let owner = pp_clone
                .pop_owner(page.addr())
                .map_err(|_| Error::UnownedPage(page.addr()))?;
            if owner != owner_id {
                return Err(Error::UnownedPage(page.addr()));
            }
        }
        Ok(count)
    }

    /// Returns the root 2nd-stage page table for this VM.
    pub fn root(&self) -> &T {
        &self.root
    }

    /// Handles a page fault for the given address.
    pub fn handle_page_fault(&mut self, addr: u64) -> Result<()> {
        if self.root.do_guest_fault(addr) {
            Ok(())
        } else {
            Err(Error::PageFaultHandling)
        }
    }

    // Writes self measurements to the specified GPA
    pub fn write_measurements_to_guest_owned_page(&mut self, gpa: u64) -> Result<usize> {
        self.root
            .write_guest_owned_page(gpa, 0, self.measurement.get_measurement())
            .map(|_| self.measurement.get_measurement().len())
            .map_err(|_| Error::UnownedPage(AlignedPageAddr4k::with_round_down(PhysAddr::new(gpa))))
    }

    // Writes to the specified GPA
    pub fn write_to_guest_owned_page(&mut self, gpa: u64, bytes: &[u8]) -> Result<usize> {
        self.root
            .write_guest_owned_page(gpa, 0, bytes)
            .map(|_| bytes.len())
            .map_err(|_| Error::UnownedPage(AlignedPageAddr4k::with_round_down(PhysAddr::new(gpa))))
    }
}

impl<T: PlatformPageTable> Drop for VmPages<T> {
    fn drop(&mut self) {
        self.root
            .phys_pages()
            .rm_active_guest(self.root.page_owner_id());
    }
}

/// Keeps the state of the host's pages.
pub struct HostRootPages<T: PlatformPageTable> {
    inner: VmPages<T>,
}

impl<T: PlatformPageTable> HostRootPages<T> {
    pub fn into_inner(self) -> VmPages<T> {
        self.inner
    }
}

/// Builder used to construct the page management structure for the host.
///
/// Note that HostRootBuilder enforces that the GPA -> HPA mappings that are created always map
/// a T::TOP_LEVEL_ALIGN-aligned chunk.
pub struct HostRootBuilder<T: PlatformPageTable> {
    root: T,
    pte_pages: SeqPageIter<PageSize4k>,
    measurement: Sha256Measure,
}

impl<T: PlatformPageTable> HostRootBuilder<T> {
    /// To be used to create the initial `HostRootPages` for the host VM.
    pub fn from_hyp_mem<A: Allocator>(
        mut hyp_mem: HypPageAlloc<A>,
        host_gpa_size: u64,
    ) -> (Vec<SequentialPages4k, A>, Self) {
        let root_table_pages = hyp_mem.take_pages_with_alignment(4, T::TOP_LEVEL_ALIGN);
        let num_pte_pages = T::max_pte_pages(host_gpa_size / PageSize4k::SIZE_BYTES);
        let pte_pages = hyp_mem.take_pages(num_pte_pages as usize).into_iter();

        let (phys_pages, host_pages) = PageState::from(hyp_mem, T::TOP_LEVEL_ALIGN);
        let root = T::new(root_table_pages, PageOwnerId::host(), phys_pages).unwrap();

        (
            host_pages,
            Self {
                root,
                pte_pages,
                measurement: Sha256Measure::new(),
            },
        )
    }

    /// Adds data pages that are measured and mapped to the page tables for the host.
    pub fn add_4k_data_pages<I>(mut self, to_addr: AlignedPageAddr4k, pages: I) -> Self
    where
        I: Iterator<Item = Page4k>,
    {
        let root = &mut self.root;
        let pte_pages = &mut self.pte_pages;
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            self.measurement.add_page(vm_addr.bits(), &page);
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            root.phys_pages()
                .set_page_owner(page.addr(), root.page_owner_id())
                .unwrap();
            root.map_page_4k(vm_addr.bits(), page, &mut || pte_pages.next())
                .unwrap();
        }
        self
    }

    /// Add zeroed pages to the host page tables
    pub fn add_4k_pages<I>(mut self, to_addr: AlignedPageAddr4k, pages: I) -> Self
    where
        I: Iterator<Item = Page4k>,
    {
        let root = &mut self.root;
        let pte_pages = &mut self.pte_pages;

        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            root.phys_pages()
                .set_page_owner(page.addr(), root.page_owner_id())
                .unwrap();
            root.map_page_4k(vm_addr.bits(), page, &mut || pte_pages.next())
                .unwrap();
        }

        self
    }

    /// Returns the host root pages as configured with data and zero pages.
    pub fn create_host(self) -> HostRootPages<T> {
        HostRootPages {
            inner: VmPages {
                root: self.root,
                measurement: self.measurement,
            },
        }
    }
}

/// Builder used to configure `VmPages` for a new guest VM.
pub struct GuestRootBuilder<T: PlatformPageTable> {
    root: T,
    measurement: Sha256Measure,
    pte_pages: PageVec<Page4k>,
}

impl<T: PlatformPageTable> GuestRootBuilder<T> {
    /// Return the page owner ID these pages will be assigned to.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.root.page_owner_id()
    }
}

impl<T: PlatformPageTable> GuestRootBuilder<T> {
    /// Create a new `GuestRootBuilder` with `root` as the backing page table and `pte_page` used to
    /// hose a Vec of pte pages.
    pub fn new(root: T, pte_page: Page4k) -> Self {
        Self {
            root,
            measurement: Sha256Measure::new(),
            pte_pages: PageVec::from(SequentialPages::<PageSize4k>::from(pte_page)),
        }
    }

    /// Add a page to be used for building the guest's page tables.
    /// Currently only supports 4k pages.
    pub fn add_pte_page(&mut self, page: Page4k) -> Result<()> {
        self.pte_pages
            .try_reserve(1)
            .map_err(|_| Error::InsufficientPtePageStorage)?;
        self.pte_pages.push(page);
        Ok(())
    }

    /// Add a measured data page for the guest to use.
    /// Currently only supports 4k pages.
    pub fn add_data_page(&mut self, gpa: u64, page: Page4k) -> Result<()> {
        self.measurement.add_page(gpa, &page);
        self.root
            .map_page_4k(gpa, page, &mut || self.pte_pages.pop())
            .map_err(Error::Mapping4kPage)
    }

    /// Add a zeroed data page for the guest to use.
    /// Currently only supports 4k pages.
    pub fn add_zero_page(&mut self, gpa: u64, page: Page4k) -> Result<()> {
        self.root
            .map_page_4k(gpa, page, &mut || self.pte_pages.pop())
            .map_err(Error::Mapping4kPage)
    }

    /// Consumes the builder and returns the guest's VmPages struct.
    pub fn create_pages(self) -> VmPages<T> {
        VmPages {
            root: self.root,
            measurement: self.measurement,
        }
    }

    pub fn get_measurement(&self) -> &[u8] {
        self.measurement.get_measurement()
    }
}
