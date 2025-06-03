// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use enum_dispatch::enum_dispatch;
use riscv_page_tables::{GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::*;
use riscv_regs::dma_wmb;
use static_assertions::const_assert;
use sync::Mutex;

use super::error::*;
use super::msi_page_table::MsiPageTable;
use crate::pci::Address;

// Maximum number of device ID bits used by the IOMMU.
const DEVICE_ID_BITS: usize = 24;

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
// This is the base format, used when capabilities.MSI_FLAT isn't set.
#[repr(C)]
struct DeviceContextBase {
    tc: u64,
    iohgatp: u64,
    fsc: u64,
    ta: u64,
}

// Extended format of the device context, used when capabilities.MSI_FLAT is set. The additional
// fields control MSI address matching and translation.
#[repr(C)]
struct DeviceContextExtended {
    base: DeviceContextBase,
    msiptp: u64,
    msi_addr_mask: u64,
    msi_addr_pattern: u64,
    _reserved: u64,
}

// There are a bunch of other bits in `tc` for ATS, etc. but we only care about V for now.
const DC_VALID: u64 = 1 << 0;

// Indicates that hardware should update the AD bits in G stage translation PTEs.
const DC_GADE: u64 = 1 << 7;

// Indicates that hardware should update the AD bits in first translation PTEs.
const DC_SADE: u64 = 1 << 8;

// Set in invalidated device contexts to indicate that the device context corresponds to a real
// device. Prevents enabling of device contexts that weren't explicitly added with `add_device()`.
const DC_SW_INVALIDATED: u64 = 1 << 31;

// Trait abstracting over base / extended device context format.
trait DeviceContext {
    const INDEX_BITS: [u8; 3];

    fn base(&self) -> &DeviceContextBase;
    fn base_mut(&mut self) -> &mut DeviceContextBase;

    // Returns the bits from `device_id` used to index at `level`.
    fn level_index_bits(device_id: DeviceId, level: usize) -> usize {
        let mask = (1 << Self::INDEX_BITS[level]) - 1;
        let shift = Self::INDEX_BITS.iter().take(level).sum::<u8>() as usize;
        (device_id.0 as usize >> shift) & mask
    }

    // Clears the device context structure.
    fn init(&mut self) {
        self.base_mut().tc = DC_SW_INVALIDATED;
        self.base_mut().iohgatp = 0;
        self.base_mut().fsc = 0;
        self.base_mut().ta = 0;
    }

    // Returns if the device context corresponds to a present device.
    fn present(&self) -> bool {
        (self.base().tc & (DC_VALID | DC_SW_INVALIDATED)) != 0
    }

    // Returns if the device context is valid.
    fn valid(&self) -> bool {
        (self.base().tc & DC_VALID) != 0
    }

    // Marks the device context as valid, using `pt` and `msi_pt` for translation.
    fn set<T: GuestStagePagingMode>(
        &mut self,
        pt: &GuestStagePageTable<T>,
        msi_pt: Option<&MsiPageTable>,
        gscid: GscId,
    ) {
        // This default implementation (appropriate for the base format) doesn't support MSI
        // translation, and hence should never receive a valid `msi_pt` parameter.
        assert!(msi_pt.is_none());

        const GSCID_SHIFT: u64 = 44;
        const HGATP_MODE_SHIFT: u64 = 60;
        self.base_mut().iohgatp = pt.get_root_address().pfn().bits()
            | ((gscid.bits() as u64) << GSCID_SHIFT)
            | (T::HGATP_MODE << HGATP_MODE_SHIFT);

        // Ensure the writes to the other context fields are visible before we mark the context
        // as valid.
        dma_wmb();

        let ade = if cfg!(feature = "hardware_ad_updates") {
            DC_SADE | DC_GADE
        } else {
            0
        };
        self.base_mut().tc = DC_VALID | ade;
    }

    // Marks the device context as invalid.
    fn invalidate(&mut self) {
        self.base_mut().tc = DC_SW_INVALIDATED;
    }
}

impl DeviceContext for DeviceContextBase {
    const INDEX_BITS: [u8; 3] = [7, 9, 8];

    fn base(&self) -> &DeviceContextBase {
        self
    }
    fn base_mut(&mut self) -> &mut DeviceContextBase {
        self
    }
}

impl DeviceContext for DeviceContextExtended {
    const INDEX_BITS: [u8; 3] = [6, 9, 9];

    fn base(&self) -> &DeviceContextBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut DeviceContextBase {
        &mut self.base
    }

    fn set<T: GuestStagePagingMode>(
        &mut self,
        pt: &GuestStagePageTable<T>,
        msi_pt: Option<&MsiPageTable>,
        gscid: GscId,
    ) {
        if let Some(msi_pt) = msi_pt {
            const MSI_MODE_FLAT: u64 = 0x1;
            const MSI_MODE_SHIFT: u64 = 60;
            self.msiptp = msi_pt.base_address().pfn().bits() | (MSI_MODE_FLAT << MSI_MODE_SHIFT);

            let (addr, mask) = msi_pt.msi_address_pattern();
            self.msi_addr_mask = mask >> PFN_SHIFT;
            self.msi_addr_pattern = addr.pfn().bits();
        } else {
            self.msiptp = 0;
        }

        // NB: Pass a None `msi_pt` parameter to the base implementation.
        self.base.set(pt, None, gscid);
    }
}

// A non-leaf device directory table entry. If valid, a non-leaf entry must point to the next
// level directory table in the hierarchy.
#[repr(C)]
struct NonLeafEntry(u64);

const NL_PFN_BITS: u64 = 44;
const NL_PFN_MASK: u64 = (1 << NL_PFN_BITS) - 1;
const NL_PFN_SHIFT: u64 = 10;
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

// Checks whether the table sizes at the different levels for a given DeviceContext look ok.
const fn _check_ddt_layout<DC: DeviceContext>() -> bool {
    DC::INDEX_BITS.len() == 3
        && (size_of::<DC>() << DC::INDEX_BITS[0] == 4096)
        && (size_of::<NonLeafEntry>() << DC::INDEX_BITS[1] == 4096)
        && (size_of::<NonLeafEntry>() << DC::INDEX_BITS[2] <= 4096)
}

const_assert!(_check_ddt_layout::<DeviceContextBase>());
const_assert!(_check_ddt_layout::<DeviceContextExtended>());

// Represents a single entry in the device directory hierarchy.
enum DeviceDirectoryEntry<'a, DC: DeviceContext> {
    Leaf(&'a mut DC),
    NextLevel(DeviceDirectoryTable<'a, DC>),
    Invalid(&'a mut NonLeafEntry, usize),
}

// Represents a single device directory table. Intermediate DDTs (level > 0) are made up entirely
// of non-leaf entries, while Leaf DDTs (level == 0) are made up entirely of `DeviceContext`s.
#[derive(Clone)]
struct DeviceDirectoryTable<'a, DC: DeviceContext> {
    table_addr: SupervisorPageAddr,
    level: usize,
    phantom: PhantomData<&'a DC>,
}

impl<'a, DC: DeviceContext + 'a> DeviceDirectoryTable<'a, DC> {
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
                table_addr: nle.pfn().into(),
                level,
                phantom: PhantomData,
            })
        } else {
            None
        }
    }

    // Returns the `DeviceDirectoryEntry` for `id` in this table.
    fn entry_for_id(&mut self, id: DeviceId) -> DeviceDirectoryEntry<'a, DC> {
        let index = DC::level_index_bits(id, self.level);
        use DeviceDirectoryEntry::*;
        if self.is_leaf() {
            // Safety: self.table_addr is properly aligned and must point to an array of
            // `DeviceContext`s if this is a leaf table. Further, `index` is guaranteed
            // to be within in the bounds of the table.
            let dc = unsafe {
                let ptr = (self.table_addr.bits() as *mut DC).add(index);
                // Pointer must be non-NULL.
                ptr.as_mut().unwrap()
            };
            Leaf(dc)
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

    // Returns if this is a leaf directory table.
    fn is_leaf(&self) -> bool {
        self.level == 0
    }

    // Get the device context for the device identified by `id`.
    fn get_context_for_id(&mut self, id: DeviceId) -> Option<&mut DC> {
        let mut entry = self.entry_for_id(id);
        use DeviceDirectoryEntry::*;
        while let NextLevel(mut t) = entry {
            entry = t.entry_for_id(id);
        }
        match entry {
            Leaf(dc) if dc.present() => Some(dc),
            _ => None,
        }
    }

    // Returns the device context for the device identified by `id`, creating it if necessary.
    fn create_context_for_id(
        &mut self,
        id: DeviceId,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<&mut DC> {
        let mut entry = self.entry_for_id(id);
        loop {
            use DeviceDirectoryEntry::*;
            let mut table = match entry {
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
                Leaf(dc) => return Ok(dc),
            };
            entry = table.entry_for_id(id);
        }
    }
}

// A trait providing directory mutation operations. Note that it does not have the specific device
// context as type parameter, but is implemented (generically) for `DeviceDirectoryTable`
// parameterized with specific device context types. Thus, this trait connects the API layer to the
// differently typed table implementations.
#[enum_dispatch(DeviceDirectoryOpsDispatch)]
trait DeviceDirectoryOps {
    // Adds the device with `id`, creating tables as necessary.
    fn add_device(
        &mut self,
        id: DeviceId,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()>;

    // Updates the device context for `id` with the given parameters and marks it valid.
    fn enable_device<T: GuestStagePagingMode>(
        &mut self,
        id: DeviceId,
        pt: &GuestStagePageTable<T>,
        msi_pt: Option<&MsiPageTable>,
        gscid: GscId,
    ) -> Result<()>;

    // Invalidates the device context for `id`.
    fn disable_device(&mut self, id: DeviceId) -> Result<()>;
}

impl<'a, DC: DeviceContext + 'a> DeviceDirectoryOps for DeviceDirectoryTable<'a, DC> {
    fn add_device(
        &mut self,
        id: DeviceId,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        let dc = self.create_context_for_id(id, get_page)?;
        if !dc.present() {
            dc.init();
        }
        Ok(())
    }

    fn enable_device<T: GuestStagePagingMode>(
        &mut self,
        id: DeviceId,
        pt: &GuestStagePageTable<T>,
        msi_pt: Option<&MsiPageTable>,
        gscid: GscId,
    ) -> Result<()> {
        let entry = self
            .get_context_for_id(id)
            .ok_or(Error::DeviceNotFound(id))?;
        if entry.valid() {
            return Err(Error::DeviceAlreadyEnabled(id));
        }
        entry.set(pt, msi_pt, gscid);
        Ok(())
    }

    fn disable_device(&mut self, id: DeviceId) -> Result<()> {
        let entry = self
            .get_context_for_id(id)
            .ok_or(Error::DeviceNotFound(id))?;
        if !entry.valid() {
            return Err(Error::DeviceNotEnabled(id));
        }
        entry.invalidate();
        Ok(())
    }
}

// A helper enum for dispatching DeviceDirectoryOps calls from non-type-parameterized context to
// the type-parameterized DeviceDirectoryOps implementations for DeviceDirectoryTable.
#[enum_dispatch]
enum DeviceDirectoryOpsDispatch<'a> {
    Base(DeviceDirectoryTable<'a, DeviceContextBase>),
    Extended(DeviceDirectoryTable<'a, DeviceContextExtended>),
}

// Represents a device directory instance. The instance is protected by a Mutex and thus separate
// from the API layer offered by `DeviceDirectory`.
struct DeviceDirectoryInner {
    root: Page<InternalClean>,
    num_levels: usize,
    format: DeviceContextFormat,
}

impl DeviceDirectoryInner {
    fn ops(&mut self) -> DeviceDirectoryOpsDispatch {
        use DeviceDirectoryOpsDispatch::*;
        match self.format {
            DeviceContextFormat::Base => Base(DeviceDirectoryTable::from_root(self)),
            DeviceContextFormat::Extended => Extended(DeviceDirectoryTable::from_root(self)),
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

/// Indicates which device context format to use.
pub enum DeviceContextFormat {
    Base,
    Extended,
}

/// Represents the device directory table for the IOMMU. The IOMMU hardware uses the DDT to map
/// a requester ID to the translation context for the device.
pub struct DeviceDirectory<D: DirectoryMode> {
    inner: Mutex<DeviceDirectoryInner>,
    phantom: PhantomData<D>,
}

impl<D: DirectoryMode> DeviceDirectory<D> {
    /// Creates a new `DeviceDirectory` using `root` as the root table page.
    pub fn new(root: Page<InternalClean>, format: DeviceContextFormat) -> Self {
        let inner = DeviceDirectoryInner {
            root,
            num_levels: D::LEVELS,
            format,
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

    /// Returns whether this IOMMU instance supports MSI page tables.
    pub fn supports_msi_page_tables(&self) -> bool {
        matches!(self.inner.lock().format, DeviceContextFormat::Extended)
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
        inner.ops().add_device(id, get_page)
    }

    /// Enables IOMMU translation for the specified device, using `pt` for 2nd-stage translation
    /// and `msi_pt` for MSI translation. The device must have been previously added with
    /// `add_device()`.
    pub fn enable_device<T: GuestStagePagingMode>(
        &self,
        id: DeviceId,
        pt: &GuestStagePageTable<T>,
        msi_pt: Option<&MsiPageTable>,
        gscid: GscId,
    ) -> Result<()> {
        if msi_pt.is_some_and(|msi_pt| msi_pt.owner() != pt.page_owner_id()) {
            return Err(Error::OwnerMismatch);
        }

        let mut inner = self.inner.lock();
        if msi_pt.is_some() && !matches!(inner.format, DeviceContextFormat::Extended) {
            return Err(Error::MsiTranslationUnsupported);
        }
        inner.ops().enable_device(id, pt, msi_pt, gscid)
    }

    /// Disables IOMMU translation for the specified device.
    pub fn disable_device(&self, id: DeviceId) -> Result<()> {
        let mut inner = self.inner.lock();
        inner.ops().disable_device(id)
    }
}
