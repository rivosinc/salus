// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::cell::{RefCell, RefMut};
use page_tracking::{HwMemMap, HwMemRegion, HwMemRegionType, HwReservedMemType, HypPageAlloc};
use riscv_elf::{ElfMap, ElfSegment, ElfSegmentPerms};
use riscv_page_tables::{
    FirstStageMapper, FirstStagePageTable, PagingMode, PteFieldBits, PteLeafPerms, Sv48,
};
use riscv_pages::{
    InternalClean, Page, PageAddr, PageSize, RawAddr, SeqPageIter, SupervisorPageAddr,
    SupervisorPhys, SupervisorPhysAddr, SupervisorVirt,
};
use riscv_regs::{satp, sstatus, LocalRegisterCopy, ReadWriteable, SatpHelpers, CSR};
use spin::Once;
use static_assertions::const_assert;

// Maximum number of regions unique to every pagetable (private).
const MAX_PRIVATE_REGIONS: usize = 32;
// Maximum number of regions shared across all pagetables.
const MAX_SHARED_REGIONS: usize = 32;

// Private regions vector.
type PrivateRegionsVec = ArrayVec<PrivateRegion, MAX_PRIVATE_REGIONS>;
// Shared regions vector.
type SharedRegionsVec = ArrayVec<SharedRegion, MAX_SHARED_REGIONS>;

/// Errors returned by creating or modifying hypervisor mappings.
#[derive(Debug)]
pub enum Error {
    /// U-mode ELF segment is not page aligned.
    ElfUnalignedSegment,
    /// U-mode ELF segment is not in U-mode VA area.
    ElfInvalidAddress,
    /// Not enough space on the U-mode map area.
    OutOfMap,
    /// Could not create a mapper for the U-mode area.
    MapperCreationFailed,
    /// Could not map the U-mode area.
    MapFailed,
    /// Could not unmap the U-mode area.
    UnmapFailed,
}

// Represents a virtual address region of the hypervisor that will be the same in all pagetables.
struct SharedRegion {
    vaddr: PageAddr<SupervisorVirt>,
    paddr: PageAddr<SupervisorPhys>,
    page_count: usize,
    pte_fields: PteFieldBits,
}

impl SharedRegion {
    // Creates a shared region from a Hw Memory Map entry.
    fn from_hw_mem_region(r: &HwMemRegion) -> Option<Self> {
        let perms = match r.region_type() {
            HwMemRegionType::Available => {
                // map available memory as rw - unsure what it'll be used for.
                Some(PteLeafPerms::RW)
            }
            HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved) => {
                // No need to map regions reserved for firmware use
                None
            }
            HwMemRegionType::Reserved(HwReservedMemType::HypervisorImage) => {
                Some(PteLeafPerms::RWX)
            }
            HwMemRegionType::Reserved(HwReservedMemType::HostKernelImage)
            | HwMemRegionType::Reserved(HwReservedMemType::HostInitramfsImage) => {
                Some(PteLeafPerms::R)
            }
            HwMemRegionType::Reserved(HwReservedMemType::HypervisorHeap)
            | HwMemRegionType::Reserved(HwReservedMemType::HypervisorPerCpu)
            | HwMemRegionType::Reserved(HwReservedMemType::PageMap)
            | HwMemRegionType::Mmio(_) => Some(PteLeafPerms::RW),
        };

        if let Some(pte_perms) = perms {
            let paddr = r.base();
            // vaddr == paddr in mapping HW memory map.
            let vaddr = r.base().as_supervisor_virt();
            let page_count = PageSize::num_4k_pages(r.size()) as usize;
            let pte_fields = PteFieldBits::leaf_with_perms(pte_perms);
            Some(Self {
                vaddr,
                paddr,
                page_count,
                pte_fields,
            })
        } else {
            None
        }
    }

    // Map this region into a page table.
    fn map(
        &self,
        sv48: &FirstStagePageTable<Sv48>,
        get_pte_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) {
        let mapper = sv48
            .map_range(
                self.vaddr,
                PageSize::Size4k,
                self.page_count as u64,
                get_pte_page,
            )
            .unwrap();
        for (virt, phys) in self
            .vaddr
            .iter_from()
            .zip(self.paddr.iter_from())
            .take(self.page_count)
        {
            // Safety: all shared regions come from the HW memory map. we will create exactly one
            // mapping for each page and will switch to using that mapping exclusively.
            unsafe {
                mapper.map_addr(virt, phys, self.pte_fields).unwrap();
            }
        }
    }
}

// U-mode binary mappings start here.
const UMODE_VA_START: u64 = 0xffffffff00000000;
// Size in bytes of the U-mode binary VA area.
const UMODE_VA_SIZE: u64 = 128 * 1024 * 1024;
// U-mode binary mappings end here.
const UMODE_VA_END: u64 = UMODE_VA_START + UMODE_VA_SIZE;

// The addresses between `UMODE_MAPPINGS_START` and `UMODE_MAPPINGS_START` + `UMODE_MAPPINGS_SIZE`
// is an area of the private page table where the hypervisor can map pages shared from guest
// VMs. The area is divided in slots, of equal size `UMODE_MAPPING_SLOT_SIZE`.
// Must be multiple of 4k.
const UMODE_MAPPING_SLOT_SIZE: u64 = 4 * 1024 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_MAPPING_SLOT_SIZE));

// Start of the private U-mode mappings area.  Must be 4k-aligned.
const UMODE_MAPPINGS_START: u64 = UMODE_VA_END + 4 * 1024 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_MAPPINGS_START));
//The number of slots available for mapping.
const UMODE_MAPPING_SLOTS: u64 = 2;
// Maximum size of the private mappings area. Must be 4k-aligned.
const UMODE_MAPPINGS_SIZE: u64 = UMODE_MAPPING_SLOTS * UMODE_MAPPING_SLOT_SIZE;
const UMODE_MAPPINGS_END: u64 = UMODE_MAPPINGS_START + UMODE_MAPPINGS_SIZE;
const_assert!(PageSize::Size4k.is_aligned(UMODE_MAPPINGS_END));

/// Start of the U-mode buffer.
const UMODE_BUFFER_START: u64 = UMODE_MAPPINGS_END + 4 * 1024 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_BUFFER_START));
/// Size of the U-mode buffer.
pub const UMODE_BUFFER_SIZE: u64 = 16 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_BUFFER_SIZE));

/// Generic Id names for each of the U-mode mapping slots.
/// There is no mandated use for each of the slots, and caller can decide to map each of them
/// readable or writable based on the requirement.
#[derive(Copy, Clone)]
pub enum UmodeSlotId {
    A,
    B,
}

// Returns true if `addr` is contained in the U-mode VA area.
fn is_umode_addr(addr: u64) -> bool {
    (UMODE_VA_START..UMODE_VA_END).contains(&addr)
}

// Returns true if (`addr`, `addr` + `len`) is a valid non-empty range in the VA area.
fn is_valid_umode_range(addr: u64, len: usize) -> bool {
    len != 0 && is_umode_addr(addr) && is_umode_addr(addr + len as u64 - 1)
}

// Represents a virtual address region that will point to different physical page on each pagetable.
struct PrivateRegion {
    // The address space where this region starts.
    vaddr: PageAddr<SupervisorVirt>,
    // Number of bytes of the VA area
    size: usize,
    // PTE bits for the mappings.
    pte_fields: PteFieldBits,
    // Data to be populated at the beginning of the VA area
    data: Option<&'static [u8]>,
}

impl PrivateRegion {
    // Creates a per-pagetable region from an U-mode ELF segment.
    fn from_umode_elf_segment(seg: &ElfSegment<'static>) -> Result<Self, Error> {
        // Sanity check for segment alignments.
        //
        // In general ELF might have segments overlapping in the same page, possibly with different
        // permissions. In order to maintain separation and expected permissions on every page, the
        // linker script for umode ELF creates different segments at different pages. Failure to do so
        // would make `map_range()` in `map()` fail.
        //
        // The following check enforces that the segment starts at a 4k page aligned address. Unless
        // the linking is completely corrupt, this also means that it starts at a different page.
        let vaddr = PageAddr::new(RawAddr::supervisor_virt(seg.vaddr()))
            .ok_or(Error::ElfUnalignedSegment)?;
        // Sanity check for VA area of the segment.
        if !is_valid_umode_range(seg.vaddr(), seg.size()) {
            return Err(Error::ElfInvalidAddress);
        }
        let pte_perms = match seg.perms() {
            ElfSegmentPerms::ReadOnly => PteLeafPerms::UR,
            ElfSegmentPerms::ReadWrite => PteLeafPerms::URW,
            ElfSegmentPerms::ReadOnlyExecute => PteLeafPerms::URX,
        };
        let pte_fields = PteFieldBits::leaf_with_perms(pte_perms);
        Ok(Self {
            vaddr,
            size: seg.size(),
            pte_fields,
            data: seg.data(),
        })
    }

    // Maps this region into a page table.
    fn map(&self, sv48: &FirstStagePageTable<Sv48>, hyp_mem: &mut HypPageAlloc) {
        // Allocate and populate first.
        let page_count = PageSize::num_4k_pages(self.size as u64);
        let pages = hyp_mem.take_pages_for_hyp_state(page_count as usize);
        // Copy data if present.
        if let Some(data) = self.data {
            let dest = pages.base().bits() as *mut u8;
            let len = core::cmp::min(data.len(), self.size);
            // Safe because we copy the minimum between the data size and the VA size.
            unsafe {
                core::ptr::copy(data.as_ptr(), dest, len);
            }
        }
        // Map the populated pages in the page table.
        let mapper = sv48
            .map_range(self.vaddr, PageSize::Size4k, page_count, &mut || {
                hyp_mem.take_pages_for_hyp_state(1).into_iter().next()
            })
            .unwrap();
        for (virt, phys) in self
            .vaddr
            .iter_from()
            .zip(pages.base().iter_from())
            .take(page_count as usize)
        {
            // Safety: all per-pagetable regions are user mappings. User mappings are not considered
            // aliases because they cannot be accessed by supervisor mode directly (sstatus.SUM needs
            // to be 1).
            unsafe {
                mapper.map_addr(virt, phys, self.pte_fields).unwrap();
            }
        }
    }

    // Restores private region to initial-state.
    fn restore(&self) {
        let mut copied = 0;
        // We have to reset the full pages mapped for this segment.
        let mapped_size = PageSize::Size4k.round_up(self.size as u64) as usize;
        // Copy data at the beginning if it's present.
        if let Some(data) = self.data {
            // In case data is bigger than region size, write up to region end only.
            let len = core::cmp::min(self.size, data.len());
            let data = &data[0..len];
            // Copy original data to umode area.
            // Write to user mapping setting SUM in SSTATUS.
            CSR.sstatus.modify(sstatus::sum.val(1));
            // Safety:
            // - this write is in a umode region guaranteed to be mapped by HypMap in every page table.
            // - the region starts at self.vaddr and is self.size byte long. `len` is <= `self.size`.
            unsafe {
                core::ptr::copy(data.as_ptr(), self.vaddr.bits() as *mut u8, len);
            }
            // Restore SUM.
            CSR.sstatus.modify(sstatus::sum.val(0));
            copied = len;
        }
        // Clear data from the end of copy to the end of mapped_area.
        let len = mapped_size - copied;
        let dest = self.vaddr.bits() + copied as u64;
        // Write to user mapping setting SUM in SSTATUS.
        CSR.sstatus.modify(sstatus::sum.val(1));
        // Safety:
        // - this write is in a umode region guaranteed to be mapped by HypMap in every page table.
        // - writing to this region start at offset `copied` and goes until the mapped size of the region.
        unsafe {
            core::ptr::write_bytes(dest as *mut u8, 0, len);
        }
        // Restore SUM.
        CSR.sstatus.modify(sstatus::sum.val(0));
    }
}

/// A page table that contains hypervisor mappings.
pub struct HypPageTable {
    /// The pagetable containing hypervisor mappings.
    sv48: FirstStagePageTable<Sv48>,
    /// U-mode buffer for this page-table.
    umode_buffer: RefCell<UmodeBuffer>,
    /// A pte page pool for U-mode mappings.
    pte_pages: RefCell<SeqPageIter<InternalClean>>,
}

impl HypPageTable {
    /// Returns the value of the SATP register for this page table.
    pub fn satp(&self) -> u64 {
        let mut satp = LocalRegisterCopy::<u64, satp::Register>::new(0);
        satp.set_from(&self.sv48, 0);
        satp.get()
    }

    /// Restores U-mode ELF mappings to initial state.
    pub fn restore_umode(&self) {
        for r in HypMap::get()
            .private_regions()
            .filter(|r| r.pte_fields == PteFieldBits::leaf_with_perms(PteLeafPerms::URW))
        {
            r.restore();
        }
    }

    /// Returns a mapper for U-mode slot `slot` for `num_pages` pages. If `writable` is true, the
    /// mapper will map pages User-writable, otherwhise will be mapped User-readable.
    pub fn umode_slot_mapper(
        &self,
        slot: UmodeSlotId,
        num_pages: u64,
        writable: bool,
    ) -> Result<UmodeSlotMapper, Error> {
        if num_pages > PageSize::num_4k_pages(UMODE_MAPPING_SLOT_SIZE) {
            return Err(Error::OutOfMap);
        }
        let vaddr = HypMap::umode_slot_va(slot);
        let mapper = self
            .sv48
            .map_range(vaddr, PageSize::Size4k, num_pages, &mut || {
                self.pte_pages.borrow_mut().next()
            })
            .map_err(|_| Error::MapperCreationFailed)?;
        let perms = if writable {
            PteFieldBits::leaf_with_perms(PteLeafPerms::URW)
        } else {
            PteFieldBits::leaf_with_perms(PteLeafPerms::UR)
        };

        Ok(UmodeSlotMapper {
            vaddr,
            mapper,
            perms,
        })
    }

    /// Unmaps `num_pages` from umode slot `slot` and returns the iterator of page addresses unmapped.
    pub fn unmap_umode_slot(
        &self,
        slot: UmodeSlotId,
        num_pages: u64,
    ) -> Result<impl Iterator<Item = SupervisorPageAddr> + '_, Error> {
        let vaddr = HypMap::umode_slot_va(slot);
        if num_pages > PageSize::num_4k_pages(UMODE_MAPPING_SLOT_SIZE) {
            return Err(Error::OutOfMap);
        }
        self.sv48
            .unmap_range(vaddr, PageSize::Size4k, num_pages)
            .map_err(|_| Error::UnmapFailed)
    }

    /// Get the page-table U-mode buffer as a mutable reference.
    pub fn umode_buffer_mut(&self) -> RefMut<UmodeBuffer> {
        self.umode_buffer.borrow_mut()
    }
}

// Global reference to the Hypervisor Map.
static HYPMAP: Once<HypMap> = Once::new();

/// A set of global mappings of the hypervisor that can be used to create page tables.
pub struct HypMap {
    shared_regions: SharedRegionsVec,
    private_regions: PrivateRegionsVec,
}

impl HypMap {
    /// Creates a new hypervisor map from a hardware memory mem map and a umode ELF.
    pub fn init(mem_map: HwMemMap, umode_elf: &ElfMap<'static>) -> Result<(), Error> {
        // All shared mappings come from the HW Memory Map.
        let shared_regions = mem_map
            .regions()
            .filter_map(SharedRegion::from_hw_mem_region)
            .collect();
        // All private mappings come from the U-mode ELF.
        let private_regions = umode_elf
            .segments()
            .map(PrivateRegion::from_umode_elf_segment)
            .collect::<Result<_, _>>()?;
        let hypmap = HypMap {
            shared_regions,
            private_regions,
        };
        HYPMAP.call_once(|| hypmap);
        Ok(())
    }

    /// Gets the global reference to the Hypervisor Map.
    pub fn get() -> &'static HypMap {
        // Unwrap okay. This must be called after `init`.
        HYPMAP.get().unwrap()
    }

    // Returns an iterator for this Hypervisor private regions.
    fn private_regions(&self) -> impl Iterator<Item = &PrivateRegion> {
        self.private_regions.iter()
    }

    /// Returns the virtual address of U-mode mapping slot `slot`.
    pub fn umode_slot_va(slot: UmodeSlotId) -> PageAddr<SupervisorVirt> {
        let slot_num = match slot {
            UmodeSlotId::A => 0,
            UmodeSlotId::B => 1,
        };
        // UMODE_MAPPING_START and UMODE_MAPPING_SLOT_SIZE are required by static assertion to be
        // page aligned, `round_down` will be a nop.
        PageAddr::with_round_down(
            RawAddr::supervisor_virt(UMODE_MAPPINGS_START + slot_num * UMODE_MAPPING_SLOT_SIZE),
            PageSize::Size4k,
        )
    }

    // Returns the virtual address of the U-mode buffer.
    fn umode_buffer_va() -> PageAddr<SupervisorVirt> {
        // UMODE_BUFFER_START is 4k aligned (enforced by static assertion), round_down wil be a no-op.
        PageAddr::with_round_down(
            RawAddr::supervisor_virt(UMODE_BUFFER_START),
            PageSize::Size4k,
        )
    }

    /// Creates a new page table based on this memory map.
    pub fn new_page_table(&self, hyp_mem: &mut HypPageAlloc) -> HypPageTable {
        // Create empty sv48 page table
        // Unwrap okay: we expect to have at least one page free.
        let root_page = hyp_mem
            .take_pages_for_hyp_state(1)
            .into_iter()
            .next()
            .unwrap();
        let sv48: FirstStagePageTable<Sv48> =
            FirstStagePageTable::new(root_page).expect("creating first sv48");
        // Map regions shared across all pagetables.
        for r in &self.shared_regions {
            r.map(&sv48, &mut || {
                hyp_mem.take_pages_for_hyp_state(1).into_iter().next()
            });
        }
        // Map regions unique to a pagetable.
        for r in &self.private_regions {
            r.map(&sv48, hyp_mem);
        }
        // Alloc and map the U-mode buffer for this page-table.
        let umode_buffer = UmodeBuffer::map(&sv48, hyp_mem);
        // Alloc pte_pages for U-mode mappings.
        let pte_pages = hyp_mem
            .take_pages_for_hyp_state(Sv48::max_pte_pages(
                UMODE_MAPPINGS_SIZE / PageSize::Size4k as u64,
            ) as usize)
            .into_iter();
        HypPageTable {
            sv48,
            umode_buffer: RefCell::new(umode_buffer),
            pte_pages: RefCell::new(pte_pages),
        }
    }
}

/// Represents a hypervisor page table mapper for a U-mode slot. Only
/// guest pages can be mapped into it.
pub struct UmodeSlotMapper<'a> {
    vaddr: PageAddr<SupervisorVirt>,
    mapper: FirstStageMapper<'a, Sv48>,
    perms: PteFieldBits,
}

impl UmodeSlotMapper<'_> {
    /// Returns the first virtual page address mappable by this mapper.
    pub fn vaddr(&self) -> PageAddr<SupervisorVirt> {
        self.vaddr
    }

    /// Maps a a guest page into an address in the range of this U-mode slot.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that the page at address `paddr` is owned by a guest and has been
    /// shared with the hypervisor.
    pub unsafe fn map_addr(
        &self,
        vaddr: PageAddr<SupervisorVirt>,
        paddr: PageAddr<SupervisorPhys>,
    ) -> Result<(), Error> {
        // Safety: pages are mapped in user mode, so no aliases of salus mappings have been
        // created. Pages are owned by guest, so no mapping of hypervisor pages are created.
        self.mapper
            .map_addr(vaddr, paddr, self.perms)
            .map_err(|_| Error::MapFailed)
    }
}

/// Umode Buffer errors.
#[derive(Debug)]
pub enum UmodeBufferError {
    /// Address overflow on increment.
    AddressOverflow,
    /// Buffer is full.
    NoSpace,
}

/// The U-mode buffer is a private area of the page table used by the hypervisor to pass data to the
/// current U-mode operation. This avoids mapping hypervisor data directly in U-mode. This area is
/// written by the hypervisor and mapped read-only in U-mode.
pub struct UmodeBuffer {
    /// The start address of the U-mode buffer memory.
    start: SupervisorPageAddr,
    /// Pointer to the unused part of the buffer.
    addr: SupervisorPhysAddr,
    /// End of buffer memory.
    max_addr: SupervisorPageAddr,
}

impl UmodeBuffer {
    /// Creates a buffer of hypervisor memory that can be mapped into a U-mode slot.
    pub fn map(sv48: &FirstStagePageTable<Sv48>, hyp_mem: &mut HypPageAlloc) -> UmodeBuffer {
        // Allocate buffer pages.
        let num_pages = PageSize::num_4k_pages(UMODE_BUFFER_SIZE);
        let pages = hyp_mem.take_pages_for_hyp_state(num_pages as usize);
        let start = pages.base();
        // Unwrap okay: we allocated num_pages already.
        let max_addr = start.checked_add_pages(num_pages).unwrap();

        // Map the allocated pages as read-only in U-mode at UMODE_BUFFER_START.
        let vaddr = HypMap::umode_buffer_va();
        let pte_fields = PteFieldBits::leaf_with_perms(PteLeafPerms::UR);
        // Unwrap okay: this is called once when we are creating the page-table. Guaranteed to be
        // unmapped.
        let mapper = sv48
            .map_range(vaddr, PageSize::Size4k, num_pages, &mut || {
                hyp_mem.take_pages_for_hyp_state(1).into_iter().next()
            })
            .unwrap();
        for (virt, phys) in vaddr
            .iter_from()
            .zip(pages.base().iter_from())
            .take(num_pages as usize)
        {
            // Safety: These pages are mapped read-only in the VA area reserved for the U-mode
            // buffer mappings. Hypervisor will write to these pages using the physical mappings and
            // U-mode will read them through this mapping. Safe because these are per-CPU mappings,
            // and when the hypervisor will be writing to these pages via the physical mappings no
            // CPU will be able to access these pages through the U-mode mappings.
            unsafe {
                mapper.map_addr(virt, phys, pte_fields).unwrap();
            }
        }
        UmodeBuffer {
            start,
            addr: start.raw(),
            max_addr,
        }
    }

    /// Appends `data` in the current CPU's buffer. Returns the U-mode virtual address where `data`
    /// has been written.
    pub fn append(&mut self, data: &[u8]) -> Result<RawAddr<SupervisorVirt>, UmodeBufferError> {
        let end = self
            .addr
            .checked_increment(data.len() as u64)
            .ok_or(UmodeBufferError::AddressOverflow)?;
        if end >= self.max_addr.raw() {
            return Err(UmodeBufferError::NoSpace);
        }
        // Safety: the range `(self.addr..self.max_addr)` is valid and owned by this buffer and we
        // verified above that `self.addr` + `data.len()` is included in the range. `buffer` is a
        // mutable reference to buffer that owns this range.
        unsafe {
            core::ptr::copy(data.as_ptr(), self.addr.bits() as *mut u8, data.len());
        }
        let offset = self.addr.bits() - self.start.bits();
        self.addr = end;
        // Unwrap okay: `offset` cannot be bigger than `self.max_addr - self.start`.
        Ok(HypMap::umode_buffer_va()
            .raw()
            .checked_increment(offset)
            .unwrap())
    }

    /// Cleans up used memory and reset the current CPU's buffer to initial state.
    pub fn clear(&mut self) {
        let size = self.addr.bits() - self.start.bits();
        // Safety: the range `(self.addr..self.max_addr)` is valid and owned by this buffer.
        // `self.addr` cannot exceed `self.max_addr`. `self` is a mutable reference to the buffer
        // that owns this range.
        unsafe {
            core::ptr::write_bytes(self.start.bits() as *mut u8, 0, size as usize);
        }
        self.addr = self.start.into();
    }
}
