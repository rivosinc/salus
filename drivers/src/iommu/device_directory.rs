// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// TODO: Remove once hooked up to the public IOMMU interface.
#![allow(dead_code)]

use assertions::const_assert;
use core::marker::PhantomData;
use riscv_page_tables::{GuestStagePageTable, PlatformPageTable};
use riscv_pages::*;
use riscv_regs::dma_wmb;
use spin::Mutex;

use super::error::*;
use super::msi_page_table::MsiPageTable;
use crate::pci::Address;

// Maximum number of device ID bits used by the IOMMU.
const DEVICE_ID_BITS: usize = 24;
// Number of bits used to index into the leaf table.
const LEAF_INDEX_BITS: usize = 6;
// Number of bits used to index into intermediate tables.
const NON_LEAF_INDEX_BITS: usize = 9;

/// The device ID. Used to index into the device directory table. For PCI devices behind an IOMMU
/// this is equivalent to the requester ID of the PCI device (i.e. the bits of the B/D/F).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceId(u32);

impl DeviceId {
    /// Creates a new `DeviceId` from the raw `val`.
    pub fn new(val: u32) -> Option<DeviceId> {
        if (val & !((1 << DEVICE_ID_BITS) - 1)) == 0 {
            Some(Self(val))
        } else {
            None
        }
    }

    /// Returns the raw bits of this `DeviceId`.
    pub fn bits(&self) -> u32 {
        self.0
    }

    // Returns the bits from this `DeviceId` used to index at `level`.
    fn level_index_bits(&self, level: usize) -> usize {
        if level == 0 {
            (self.0 as usize) & ((1 << LEAF_INDEX_BITS) - 1)
        } else {
            let shift = LEAF_INDEX_BITS + NON_LEAF_INDEX_BITS * (level - 1);
            ((self.0 as usize) >> shift) & ((1 << NON_LEAF_INDEX_BITS) - 1)
        }
    }
}

impl TryFrom<Address> for DeviceId {
    type Error = Error;

    fn try_from(address: Address) -> Result<Self> {
        Self::new(address.bits()).ok_or(Error::PciAddressTooLarge(address))
    }
}

/// Global Soft-Context ID. The equivalent of hgatp.VMID, but always 16 bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GscId(u16);

impl GscId {
    /// Creates a `GscId` from the raw `id`.
    pub(super) fn new(id: u16) -> Self {
        GscId(id)
    }

    /// Returns the raw bits of this `GscId`.
    pub fn bits(&self) -> u16 {
        self.0
    }
}

// Defines the translation context for a device. A valid device context enables translation for
// DMAs from the corresponding device according to the tables programmed into the device context.
#[repr(C)]
struct DeviceContext {
    tc: u64,
    iohgatp: u64,
    fsc: u64,
    ta: u64,
    msiptp: u64,
    msi_addr_mask: u64,
    msi_addr_pattern: u64,
    _reserved: u64,
}

// There are a bunch of other bits in `tc` for ATS, etc. but we only care about V for now.
const DC_VALID: u64 = 1 << 0;

// Set in invalidated device contexts to indicate that the device context corresponds to a real
// device. Prevents enabling of device contexts that weren't explicitly added with `add_device()`.
const DC_SW_INVALIDATED: u64 = 1 << 31;

impl DeviceContext {
    // Clears the device context structure.
    fn init(&mut self) {
        self.tc = DC_SW_INVALIDATED;
        self.iohgatp = 0;
        self.fsc = 0;
        self.ta = 0;
        self.msiptp = 0;
        self.msi_addr_mask = 0;
        self.msi_addr_pattern = 0;
    }

    // Returns if the device context corresponds to a present device.
    fn present(&self) -> bool {
        (self.tc & (DC_VALID | DC_SW_INVALIDATED)) != 0
    }

    // Returns if the device context is valid.
    fn valid(&self) -> bool {
        (self.tc & DC_VALID) != 0
    }

    // Marks the device context as valid, using `pt` and `msi_pt` for translation.
    fn set<T: GuestStagePageTable>(
        &mut self,
        pt: &PlatformPageTable<T>,
        msi_pt: &MsiPageTable,
        gscid: GscId,
    ) {
        const MSI_MODE_FLAT: u64 = 0x1;
        const MSI_MODE_SHIFT: u64 = 60;
        self.msiptp = msi_pt.base_address().pfn().bits() | (MSI_MODE_FLAT << MSI_MODE_SHIFT);

        let (addr, mask) = msi_pt.msi_address_pattern();
        self.msi_addr_mask = mask >> PFN_SHIFT;
        self.msi_addr_pattern = addr.pfn().bits();

        const GSCID_SHIFT: u64 = 44;
        const HGATP_MODE_SHIFT: u64 = 60;
        self.iohgatp = pt.get_root_address().pfn().bits()
            | ((gscid.bits() as u64) << GSCID_SHIFT)
            | (T::HGATP_VALUE << HGATP_MODE_SHIFT);

        // Ensure the writes to the other context fields are visible before we mark the context
        // as valid.
        dma_wmb();

        self.tc = DC_VALID;
    }

    // Marks the device context as invalid.
    fn invalidate(&mut self) {
        self.tc = DC_SW_INVALIDATED;
    }
}

// A non-leaf device directory table entry. If valid, a non-leaf entry must point to the next
// level directory table in the hierarchy.
#[repr(C)]
struct NonLeafEntry(u64);

const NL_PFN_BITS: u64 = 44;
const NL_PFN_MASK: u64 = (1 << NL_PFN_BITS) - 1;
const NL_PFN_SHIFT: u64 = 12;
const NL_VALID: u64 = 1;

impl NonLeafEntry {
    // Returns if this entry is marked valid.
    fn valid(&self) -> bool {
        (self.0 & NL_VALID) != 0
    }

    // Returns the PFN referred to by this entry.
    fn pfn(&self) -> SupervisorPfn {
        Pfn::supervisor((self.0 >> NL_PFN_SHIFT) & NL_PFN_MASK)
    }

    // Sets `self` to valid with the given `pfn`.
    fn set(&mut self, pfn: SupervisorPfn) {
        self.0 = (pfn.bits() << NL_PFN_SHIFT) | NL_VALID;
    }
}

// Represents a single entry in the device directory hierarchy.
enum DeviceDirectoryEntry<'a> {
    PresentLeaf(&'a mut DeviceContext),
    NotPresentLeaf(&'a mut DeviceContext),
    NextLevel(DeviceDirectoryTable<'a>),
    Invalid(&'a mut NonLeafEntry, usize),
}

// Represents a single device directory table. Intermediate DDTs (level > 0) are made up entirely
// of non-leaf entries, while Leaf DDTs (level == 0) are made up entirely of `DeviceContext`s.
struct DeviceDirectoryTable<'a> {
    table_addr: SupervisorPageAddr,
    level: usize,
    phantom: PhantomData<&'a mut DeviceDirectoryInner>,
}

impl<'a> DeviceDirectoryTable<'a> {
    // Creates the root `DeviceDirectoryTable` from `owner`.
    fn from_root(owner: &'a mut DeviceDirectoryInner) -> Self {
        Self {
            table_addr: owner.root.addr(),
            level: owner.num_levels - 1,
            phantom: PhantomData,
        }
    }

    // Creates a `DeviceDirectoryTable` from the non-leaf directory entry `nle` at `level`.
    // Returns `None` if `nle` is not valid.
    //
    // # Safety
    //
    // If `nle` is valid, the PFN it contains must point to a next-level directory table page
    // that is uniquely owned by this device directory.
    unsafe fn from_non_leaf_entry(nle: &'a mut NonLeafEntry, level: usize) -> Option<Self> {
        if nle.valid() {
            Some(Self {
                // Unwrap ok: PFNs always map to 4kB-aligned addresses.
                table_addr: SupervisorPageAddr::from_pfn(nle.pfn(), PageSize::Size4k).unwrap(),
                level,
                phantom: PhantomData,
            })
        } else {
            None
        }
    }

    // Returns the `DeviceDirectoryEntry` for `id` in this table.
    fn entry_for_id(&mut self, id: DeviceId) -> DeviceDirectoryEntry<'a> {
        let index = id.level_index_bits(self.level);
        use DeviceDirectoryEntry::*;
        if self.is_leaf() {
            // Safety: self.table_addr is properly aligned and must point to an array of
            // `DeviceContext`s if this is a leaf table. Further, `index` is guaranteed
            // to be within in the bounds of the table.
            let dc = unsafe {
                let ptr = (self.table_addr.bits() as *mut DeviceContext).add(index);
                // Pointer must be non-NULL.
                ptr.as_mut().unwrap()
            };
            if dc.present() {
                PresentLeaf(dc)
            } else {
                NotPresentLeaf(dc)
            }
        } else {
            // Safety: self.table_addr is properly aligned and must point to an array of
            // `NonLeafEntry`s if this is an intermediate table. Further, `index` is guaranteed
            // to be within in the bounds of the table.
            let nle = unsafe {
                let ptr = (self.table_addr.bits() as *mut NonLeafEntry).add(index);
                // Pointer must be non-NULL.
                ptr.as_mut().unwrap()
            };
            if nle.valid() {
                // Safety: If `nle` is valid, the PFN is guaranteed to point to a next-level
                // directory table owned by this device directory.
                let table = unsafe { Self::from_non_leaf_entry(nle, self.level - 1).unwrap() };
                NextLevel(table)
            } else {
                Invalid(nle, self.level - 1)
            }
        }
    }

    // Returns the next-level table mapping `id`, using `get_page` to allocate a directory table
    // page if necessary.
    fn next_level_or_fill(
        &mut self,
        id: DeviceId,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<DeviceDirectoryTable<'a>> {
        use DeviceDirectoryEntry::*;
        let table = match self.entry_for_id(id) {
            NextLevel(t) => t,
            Invalid(nle, level) => {
                let page = get_page().ok_or(Error::OutOfPages)?;
                nle.set(page.pfn());
                // Safety: We just allocated the page this entry points to and thus have unique
                // ownership over the memory it refers to.
                unsafe {
                    // Unwrap ok, we just marked the entry as valid.
                    Self::from_non_leaf_entry(nle, level).unwrap()
                }
            }
            _ => {
                return Err(Error::NotIntermediateTable);
            }
        };
        Ok(table)
    }

    // Returns if this is a leaf directory table.
    fn is_leaf(&self) -> bool {
        self.level == 0
    }
}

struct DeviceDirectoryInner {
    root: Page<InternalClean>,
    num_levels: usize,
}

impl DeviceDirectoryInner {
    fn get_context_for_id(&mut self, id: DeviceId) -> Option<&mut DeviceContext> {
        let mut entry = DeviceDirectoryTable::from_root(self).entry_for_id(id);
        use DeviceDirectoryEntry::*;
        while let NextLevel(mut t) = entry {
            entry = t.entry_for_id(id);
        }
        match entry {
            PresentLeaf(dc) => Some(dc),
            _ => None,
        }
    }
}

/// Defines the layout of the device directory table. Intermediate and leaf tables have the same
/// format regardless of the number of levels.
pub trait DirectoryMode {
    /// The number of levels in the device directory hierarchy.
    const LEVELS: usize;
    /// The value that should be programmed into ddtp.iommu_mode for this translation mode.
    const IOMMU_MODE: u64;
}

/// A 3-level device directory table supporting up to 24-bit requester IDs.
pub enum Ddt3Level {}

impl DirectoryMode for Ddt3Level {
    const LEVELS: usize = 3;
    const IOMMU_MODE: u64 = 4;
}

/// Represents the device directory table for the IOMMU. The IOMMU hardware uses the DDT to map
/// a requester ID to the translation context for the device.
pub struct DeviceDirectory<D: DirectoryMode> {
    inner: Mutex<DeviceDirectoryInner>,
    phantom: PhantomData<D>,
}

impl<D: DirectoryMode> DeviceDirectory<D> {
    /// Creates a new `DeviceDirectory` using `root` as the root table page.
    pub fn new(root: Page<InternalClean>) -> Self {
        let inner = DeviceDirectoryInner {
            root,
            num_levels: D::LEVELS,
        };
        Self {
            inner: Mutex::new(inner),
            phantom: PhantomData,
        }
    }

    /// Returns the base address of this `DeviceDirectory`.
    pub fn base_address(&self) -> SupervisorPageAddr {
        self.inner.lock().root.addr()
    }

    /// Adds and initializes a device context for `id` in this `DeviceDirectory`. The device
    /// context is initially invalid, i.e. translation is off for the device. Uses `get_page`
    /// to allocate intermediate directory table pages, if necessary.
    pub fn add_device(
        &self,
        id: DeviceId,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        // Silence bogus auto-deref lint, see https://github.com/rust-lang/rust-clippy/issues/9101.
        #[allow(clippy::explicit_auto_deref)]
        let mut table = DeviceDirectoryTable::from_root(&mut *inner);
        while !table.is_leaf() {
            table = table.next_level_or_fill(id, get_page)?;
        }
        if let DeviceDirectoryEntry::NotPresentLeaf(dc) = table.entry_for_id(id) {
            dc.init();
        }
        Ok(())
    }

    /// Enables IOMMU translation for the specified device, using `pt` for 2nd-stage translation
    /// and `msi_pt` for MSI translation. The device must have been previously added with
    /// `add_device()`.
    pub fn enable_device<T: GuestStagePageTable>(
        &self,
        id: DeviceId,
        pt: &PlatformPageTable<T>,
        msi_pt: &MsiPageTable,
        gscid: GscId,
    ) -> Result<()> {
        if pt.page_owner_id() != msi_pt.owner() {
            return Err(Error::PageTableOwnerMismatch);
        }
        let mut inner = self.inner.lock();
        let entry = inner
            .get_context_for_id(id)
            .ok_or(Error::DeviceNotFound(id))?;
        if entry.valid() {
            return Err(Error::DeviceAlreadyEnabled(id));
        }
        entry.set(pt, msi_pt, gscid);
        Ok(())
    }

    /// Disables IOMMU translation for the specified device.
    pub fn disable_device(&self, id: DeviceId) -> Result<()> {
        let mut inner = self.inner.lock();
        let entry = inner
            .get_context_for_id(id)
            .ok_or(Error::DeviceNotFound(id))?;
        if !entry.valid() {
            return Err(Error::DeviceNotEnabled(id));
        }
        entry.invalidate();
        Ok(())
    }
}

fn _assert_ddt_layout() {
    const_assert!(core::mem::size_of::<DeviceContext>() << LEAF_INDEX_BITS == 4096);
    const_assert!(core::mem::size_of::<NonLeafEntry>() << NON_LEAF_INDEX_BITS == 4096);
}
