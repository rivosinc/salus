// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use page_tracking::PageTracker;
use riscv_pages::*;
use spin::Mutex;

use super::error::*;
use crate::imsic::{GuestImsicGeometry, ImsicLocation, SupervisorImsicGeometry};

// An MSI page-table entry. Only the first u64 is used in "write-through" mode.
//
// TODO: Support memory-resident interrupt file (MRIF) format PTEs.
#[repr(C)]
struct MsiPte {
    pte: u64,
    _reserved: u64,
}

// Write-through PTEs have just the V and W bits set.
const MSI_PTE_PFN_SHIFT: usize = 10;
const MSI_PTE_VALID: u64 = 1u64 << 0;
const MSI_PTE_WRITE: u64 = 1u64 << 2;

impl MsiPte {
    // Marks the PTE as valid and mapping `pfn`.
    fn set(&mut self, pfn: SupervisorPfn) {
        self.pte = (pfn.bits() << MSI_PTE_PFN_SHIFT) | MSI_PTE_VALID | MSI_PTE_WRITE;
    }

    // Invalidates the PTE.
    fn clear(&mut self) {
        self.pte = 0;
    }

    // Returns if this is a valid write-through PTE.
    fn valid(&self) -> bool {
        (self.pte & (MSI_PTE_VALID | MSI_PTE_WRITE)) == (MSI_PTE_VALID | MSI_PTE_WRITE)
    }
}

// An index within an MSI page table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MsiPageTableIndex(usize);

impl MsiPageTableIndex {
    // Creates an index from the IMSIC identified by `location` in `geometry`.
    fn from(geometry: &GuestImsicGeometry, location: ImsicLocation) -> Option<Self> {
        if !geometry.location_is_valid(location) {
            return None;
        }
        // An MSI page table index is created by densely packing the guest, hart, and group index
        // bits. See the IOMMU specification for details.
        let index = (location.file().bits() as u64)
            | (location.hart().bits() << geometry.guest_index_bits())
            | (location.group().bits()
                << (geometry.guest_index_bits() + geometry.hart_index_bits()));
        Some(MsiPageTableIndex(index as usize))
    }

    fn bits(&self) -> usize {
        self.0
    }
}

struct MsiPageTableInner {
    pages: SequentialPages<InternalClean>,
    src_geometry: GuestImsicGeometry,
    dest_geometry: SupervisorImsicGeometry,
    page_tracker: PageTracker,
    owner: PageOwnerId,
}

impl MsiPageTableInner {
    // Returns a mutable reference to the PTE at `index`.
    fn entry_for_index(&mut self, index: MsiPageTableIndex) -> Option<&mut MsiPte> {
        if index.bits() >= (self.pages.length_bytes() as usize) / core::mem::size_of::<MsiPte>() {
            return None;
        }
        // Safety: We've verified that `index` is within the memory owned by `self.pages`.
        unsafe {
            (self.pages.base().bits() as *mut MsiPte)
                .add(index.bits())
                .as_mut()
        }
    }
}

impl Drop for MsiPageTableInner {
    fn drop(&mut self) {
        // Safe since we uniquely own the pages in `self.pages`.
        let pages: SequentialPages<InternalDirty> = unsafe {
            SequentialPages::from_mem_range(self.pages.base(), PageSize::Size4k, self.pages.len())
        }
        .unwrap();
        for p in pages {
            // Unwrap ok, the page must've been assigned to us to begin with.
            self.page_tracker.release_page(p).unwrap();
        }
    }
}

/// An MSI page table. Used by the IOMMU to provide translation for incoming MSI writes from a PCI
/// device. MSI page tables have a flat structure with each entry providing translation for one
/// 4kB page: from the guest physical address of an IMSIC interrupt file to the supervisor
/// physical address of that file or a memory-resident interrupt file (MRIF).
pub struct MsiPageTable {
    inner: Mutex<MsiPageTableInner>,
}

impl MsiPageTable {
    /// Creates a new `MsiPageTable` to translate from `src_geometry` to `dest_geometry`. Uses
    /// `pages` for storage, which must be at least as large as `required_table_size()` in length.
    pub fn new(
        pages: SequentialPages<InternalClean>,
        src_geometry: GuestImsicGeometry,
        dest_geometry: SupervisorImsicGeometry,
        page_tracker: PageTracker,
        owner: PageOwnerId,
    ) -> Result<Self> {
        let table_size = Self::required_table_size(&src_geometry);
        if pages.length_bytes() < table_size {
            return Err(Error::InsufficientMsiTablePages);
        }
        // The MSI page table base must be aligned to the table size.
        if pages.base().bits() % table_size != 0 {
            return Err(Error::MisalignedMsiTablePages);
        }
        for addr in pages.base().iter_from().take(pages.len() as usize) {
            if !page_tracker.is_internal_state_page(addr, owner) {
                return Err(Error::UnownedMsiTablePages);
            }
        }
        let inner = MsiPageTableInner {
            pages,
            src_geometry,
            dest_geometry,
            page_tracker,
            owner,
        };
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Maps the IMSIC location `src` in guest physical address space to the physical IMSIC file
    /// identified by `dest`. `src` must not currently be mapped and `dest` must be owned by the
    /// owner of this `MsiPageTable`.
    ///
    /// TODO: Enforce that `dest` isn't aliased in the MSI page table. While aliasing could cause
    /// incorrect behavior of a guest VM, it does not affect host memory safety or violate the
    /// isolation between VMs since we still verify the ownership of `dest`.
    pub fn map(&self, src: ImsicLocation, dest: ImsicLocation) -> Result<()> {
        let mut inner = self.inner.lock();
        // Make sure we own the IMSIC page referenced by `dest`.
        let dest_addr = inner
            .dest_geometry
            .location_to_addr(dest)
            .ok_or(Error::InvalidImsicLocation(dest))?;
        if !inner.page_tracker.is_mapped_page(
            dest_addr,
            inner.owner,
            MemType::Mmio(DeviceMemType::Imsic),
        ) {
            return Err(Error::MsiPageNotOwned(dest_addr));
        }

        let index = MsiPageTableIndex::from(&inner.src_geometry, src)
            .ok_or(Error::InvalidImsicLocation(src))?;
        // Unwrap ok: We've validated `src` so `index` must be valid for this page table.
        let entry = inner.entry_for_index(index).unwrap();
        if entry.valid() {
            return Err(Error::MsiAlreadyMapped(src));
        }
        entry.set(dest_addr.pfn());

        Ok(())
    }

    /// Remaps the IMSIC location `src` in guest physical address space to the new physical IMSIC
    /// file identified by `dest`. `src` must be currently mapped and `dest` must be owned by
    /// the owner of this `MsiPageTable`.
    pub fn remap(&self, src: ImsicLocation, dest: ImsicLocation) -> Result<()> {
        let mut inner = self.inner.lock();
        // Make sure we own the IMSIC page referenced by `dest`.
        let dest_addr = inner
            .dest_geometry
            .location_to_addr(dest)
            .ok_or(Error::InvalidImsicLocation(dest))?;
        if !inner.page_tracker.is_mapped_page(
            dest_addr,
            inner.owner,
            MemType::Mmio(DeviceMemType::Imsic),
        ) {
            return Err(Error::MsiPageNotOwned(dest_addr));
        }

        let entry = MsiPageTableIndex::from(&inner.src_geometry, src)
            .and_then(|index| inner.entry_for_index(index))
            .ok_or(Error::InvalidImsicLocation(src))?;
        if !entry.valid() {
            return Err(Error::MsiNotMapped(src));
        }
        entry.set(dest_addr.pfn());

        Ok(())
    }

    /// Removes the mapping for the specified IMSIC location in guest physical address space.
    pub fn unmap(&self, location: ImsicLocation) -> Result<()> {
        let mut inner = self.inner.lock();
        let index = MsiPageTableIndex::from(&inner.src_geometry, location)
            .ok_or(Error::InvalidImsicLocation(location))?;
        // Unwrap ok: We've validated `location` so `index` must be valid for this page table.
        let entry = inner.entry_for_index(index).unwrap();
        if !entry.valid() {
            return Err(Error::MsiNotMapped(location));
        }
        entry.clear();

        Ok(())
    }

    /// Returns the base physical address of this page table.
    pub fn base_address(&self) -> SupervisorPageAddr {
        self.inner.lock().pages.base()
    }

    /// Returns the owner of this page table
    pub fn owner(&self) -> PageOwnerId {
        self.inner.lock().owner
    }

    /// Returns the IMSIC geometry used to define the input to this page table.
    pub fn src_geometry(&self) -> GuestImsicGeometry {
        self.inner.lock().src_geometry.clone()
    }

    /// Returns the address and mask used to identify guest physical addresses which should be
    /// translated using this page table.
    pub fn msi_address_pattern(&self) -> (GuestPageAddr, u64) {
        let inner = self.inner.lock();
        (
            inner.src_geometry.base_addr(),
            inner.src_geometry.index_mask(),
        )
    }

    /// Returns the required size for an MSI page table with the specified input geometry. The
    /// page table must also be aligned to this size.
    pub fn required_table_size(geometry: &GuestImsicGeometry) -> u64 {
        core::cmp::max(
            PageSize::Size4k as usize,
            (1 << geometry.index_bits()) * core::mem::size_of::<MsiPte>(),
        ) as u64
    }
}
