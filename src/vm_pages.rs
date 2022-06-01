// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::{marker::PhantomData, ops::Deref};
use data_measure::data_measure::DataMeasure;
use data_measure::sha256::Sha256Measure;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{GuestStagePageTable, PageState};
use riscv_pages::{
    CleanPage, GuestPageAddr, GuestPhysAddr, Page, PageOwnerId, PageSize, Pfn, PhysPage,
    SequentialPages, SupervisorPageAddr,
};
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
    SettingOwner(riscv_page_tables::PageTrackingError),
    // Page table root must be aligned to 16k to be used for sv48x4 mappings
    UnalignedVmPages(GuestPageAddr),
    UnownedPage(GuestPageAddr),
    UnsupportedPageSize(PageSize),
    MeasurementBufferTooSmall,
    AddressOverflow,
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

/// Represents a reference to the current VM address space. The previous address space is restored
/// when dropped. Used to directly access a guest's memory.
pub struct ActiveVmPages<'a, T: GuestStagePageTable> {
    prev_hgatp: u64,
    vm_pages: &'a VmPages<T>,
}

impl<'a, T: GuestStagePageTable> Drop for ActiveVmPages<'a, T> {
    fn drop(&mut self) {
        CSR.hgatp.set(self.prev_hgatp);
    }
}

impl<'a, T: GuestStagePageTable> Deref for ActiveVmPages<'a, T> {
    type Target = VmPages<T>;

    fn deref(&self) -> &VmPages<T> {
        self.vm_pages
    }
}

impl<'a, T: GuestStagePageTable> ActiveVmPages<'a, T> {
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

// Convenience wrapper for invalidating, cleaning, and taking ownership over a contiguous range of
// pages.
fn take_and_clean_pages_for<T: GuestStagePageTable>(
    root: &mut T,
    addr: GuestPageAddr,
    num_pages: u64,
    new_owner: PageOwnerId,
) -> Result<impl Iterator<Item = Page> + '_> {
    let phys_pages = root.phys_pages();
    let taken_pages = root
        .invalidate_range(addr, num_pages)
        .map_err(Error::Paging)?
        .map(CleanPage::from)
        .map(Page::from)
        .map(move |p| {
            phys_pages.set_page_owner(p.addr(), new_owner).unwrap();
            p
        });
    Ok(taken_pages)
}

impl<T: GuestStagePageTable> VmPages<T, VmStateFinalized> {
    /// Creates a new `Vm` using pages donated by `self`. The returned `Vm` is in the initializing
    /// state, ready for its address space to be constructed.
    pub fn create_guest_vm(
        &self,
        page_root_addr: GuestPageAddr,
        state_addr: GuestPageAddr,
        vcpus_addr: GuestPageAddr,
        num_vcpu_pages: u64,
    ) -> Result<(Vm<T, VmStateInitializing>, Page)> {
        if (page_root_addr.bits() as *const u64).align_offset(T::TOP_LEVEL_ALIGN as usize) != 0 {
            return Err(Error::UnalignedVmPages(page_root_addr));
        }
        let id = self.phys_pages.add_active_guest().map_err(Error::GuestId)?;
        let mut root = self.root.lock();

        let guest_root_pages = SequentialPages::from_pages(take_and_clean_pages_for(
            &mut *root,
            page_root_addr,
            4,
            id,
        )?)
        .unwrap();
        let guest_root = T::new(guest_root_pages, id, self.phys_pages.clone()).unwrap();

        let mut state_pages =
            take_and_clean_pages_for(&mut *root, state_addr, TVM_STATE_PAGES, id)?;
        let state_page = state_pages.next().unwrap();
        let pte_vec_pages = SequentialPages::from_pages(state_pages).unwrap();

        let vcpu_pages = SequentialPages::from_pages(take_and_clean_pages_for(
            &mut *root,
            vcpus_addr,
            num_vcpu_pages,
            id,
        )?)
        .unwrap();

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
        let pt_pages = take_and_clean_pages_for(&mut *root, from_addr, count, to.page_owner_id())?;
        for page in pt_pages {
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

    /// Activates the address space represented by this `VmPages`. The address space is exited (and
    /// the previous one restored) when the returned `ActiveVmPages` is dropped.
    ///
    /// The caller must ensure that VMID has been allocated to reference this address space on this
    /// CPU and that there are no stale translations tagged with VMID referencing other VM address
    /// spaces in this CPU's TLB.
    pub fn enter_with_vmid(&self, vmid: VmId) -> ActiveVmPages<T> {
        let mut hgatp = LocalRegisterCopy::<u64, hgatp::Register>::new(0);
        hgatp.modify(hgatp::vmid.val(vmid.vmid()));
        hgatp.modify(hgatp::ppn.val(Pfn::from(self.root_address()).bits()));
        hgatp.modify(hgatp::mode.val(T::HGATP_VALUE));
        let prev_hgatp = CSR.hgatp.atomic_replace(hgatp.get());

        ActiveVmPages {
            prev_hgatp,
            vm_pages: self,
        }
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
