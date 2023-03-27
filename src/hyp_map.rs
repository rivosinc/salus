// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use crate::hyp_layout::*;

use arrayvec::ArrayVec;
use core::cell::RefCell;
use data_model::{DataInit, VolatileMemory, VolatileMemoryError, VolatileSlice};
use page_tracking::{HwMemMap, HwMemRegion, HwMemRegionType, HwReservedMemType, HypPageAlloc};
use riscv_elf::{ElfMap, ElfSegment, ElfSegmentPerms};
use riscv_page_tables::{
    FirstStageMapper, FirstStagePageTable, PagingMode, PteFieldBits, PteLeafPerms, Sv48,
};
use riscv_pages::{
    InternalClean, InternalDirty, Page, PageAddr, PageSize, RawAddr, SeqPageIter, SequentialPages,
    SupervisorPageAddr, SupervisorPhys, SupervisorVirt,
};
use riscv_regs::{satp, sstatus, LocalRegisterCopy, ReadWriteable, SatpHelpers, CSR};
use sync::Once;

// The copy to/from guest memory routines defined in extable.S.
extern "C" {
    fn _copy_to_user(dest_addr: u64, src: *const u8, len: usize) -> usize;
    fn _copy_from_user(dest: *mut u8, src_addr: u64, len: usize) -> usize;
}

// Maximum number of U-mode ELF regions.
const MAX_UMODE_ELF_REGIONS: usize = 32;
// Maximum number of hardware map regions.
const MAX_HW_MAP_REGIONS: usize = 32;

// Umode ELF regions vector.
type UmodeElfRegionsVec = ArrayVec<UmodeElfRegion, MAX_UMODE_ELF_REGIONS>;
// Hw map regions vector.
type HwMapRegionsVec = ArrayVec<HwMapRegion, MAX_HW_MAP_REGIONS>;

// Global reference to the Hypervisor Map.
static HYPMAP: Once<HypMap> = Once::new();

/// Errors returned by creating or modifying hypervisor mappings.
#[derive(Debug)]
pub enum Error {
    /// U-mode ELF segment is not page aligned.
    ElfUnalignedSegment,
    /// U-mode ELF segment is not in U-mode VA area.
    ElfInvalidAddress,
    /// Not enough space in the U-mode map area.
    OutOfMap,
    /// Could not create a mapper for the U-mode area.
    MapperCreationFailed,
    /// Could not map the U-mode area.
    MapFailed,
    /// Could not unmap the U-mode area.
    UnmapFailed,
    /// U-mode Input Memory Error.
    UmodeInput(VolatileMemoryError),
    /// Invalid U-mode address.
    InvalidAddress,
    /// Page Fault while accessing U-mode memory.
    UmodePageFault(RawAddr<SupervisorVirt>, usize),
    /// Stack overlaps with other regions.
    InvalidStackSize(u64),
}

// Represents a virtual address region of the hypervisor created from the Hardware Memory Map.
struct HwMapRegion {
    vaddr: PageAddr<SupervisorVirt>,
    paddr: PageAddr<SupervisorPhys>,
    page_count: usize,
    pte_fields: PteFieldBits,
}

impl HwMapRegion {
    // Creates a hypervisor region from a Hw Memory Map entry.
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
            // Safety: all regions come from the HW memory map. we will create exactly one mapping for
            // each page and will switch to using that mapping exclusively.
            unsafe {
                mapper.map_addr(virt, phys, self.pte_fields).unwrap();
            }
        }
    }
}

// Represents a virtual address region that will point to different physical page on each pagetable.
struct UmodeElfRegion {
    // The address space where this region starts.
    vaddr: PageAddr<SupervisorVirt>,
    // Number of bytes of the VA area
    size: usize,
    // PTE bits for the mappings.
    pte_fields: PteFieldBits,
    // Data to be populated at the beginning of the VA area
    data: Option<&'static [u8]>,
}

impl UmodeElfRegion {
    // Creates a region from an U-mode ELF segment.
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
        if !is_valid_umode_binary_range(seg.vaddr(), seg.size()) {
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
            // Safety: all regions are user mappings. User mappings are not considered aliases because
            // they cannot be accessed by supervisor mode directly (sstatus.SUM needs to be 1).
            unsafe {
                mapper.map_addr(virt, phys, self.pte_fields).unwrap();
            }
        }
    }

    // Restores region to initial-state.
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

/// The U-mode Input Region is a private region of the page table used by the hypervisor to pass
/// data to the current U-mode operation. This avoids mapping hypervisor data directly in
/// U-mode. This area is written by the hypervisor and mapped read-only in U-mode.
struct UmodeInputRegion {
    /// Volatile Slice used to write to this area from the hypervisor.
    vslice: VolatileSlice<'static>,
}

impl UmodeInputRegion {
    /// Allocate hypervisor pages and map them read-only in U-mode VA space.
    fn map(sv48: &FirstStagePageTable<Sv48>, hyp_mem: &mut HypPageAlloc) -> Self {
        // Allocate pages.
        let num_pages = PageSize::num_4k_pages(UMODE_INPUT_SIZE);
        let pages = hyp_mem.take_pages_for_hyp_state(num_pages as usize);
        let start = pages.base();

        // Map the allocated pages as read-only in U-mode at UMODE_INPUT_START.
        // UMODE_INPUT_START is 4k aligned (enforced by static assertion), round_down wil be a no-op.
        let vaddr = PageAddr::with_round_down(
            RawAddr::supervisor_virt(UMODE_INPUT_START),
            PageSize::Size4k,
        );
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
            // input region mappings. Hypervisor will write to these pages using the physical
            // mappings and U-mode will read them through this mapping. Safe because these are
            // per-CPU mappings, and when the hypervisor will be writing to these pages via the
            // physical mappings no CPU will be able to access these pages through the U-mode
            // mappings.
            unsafe {
                mapper.map_addr(virt, phys, pte_fields).unwrap();
            }
        }
        // Safety: the range `(start..max_addr)` is mapped in U-mode as read-only and was uniquely
        // claimed for this area from the hypervisor map above.
        let vslice = unsafe {
            VolatileSlice::from_raw_parts(start.raw().bits() as *mut u8, UMODE_INPUT_SIZE as usize)
        };
        Self { vslice }
    }

    // Writes `data` in the current CPU's U-mode Input Region.
    fn store<T: DataInit>(&mut self, data: T) -> Result<(), Error> {
        let vref = self.vslice.get_ref(0).map_err(Error::UmodeInput)?;
        vref.store(data);
        Ok(())
    }

    // Clears the U-mode Input Region.
    fn clear(&mut self) {
        self.vslice.write_bytes(0);
    }
}

/// Represents a hypervisor stack region.
pub struct HypStackRegion {
    paddr: PageAddr<SupervisorPhys>,
}

#[allow(dead_code)]
impl HypStackRegion {
    fn new(stack_pages: SequentialPages<InternalDirty>) -> Result<Self, Error> {
        let page_count = stack_pages.len();
        if page_count != HYP_STACK_PAGES {
            return Err(Error::InvalidStackSize(page_count));
        }
        let paddr = stack_pages.base();
        Ok(HypStackRegion { paddr })
    }

    // Map hypervisor stack in current page-table.
    fn map(&self, sv48: &FirstStagePageTable<Sv48>, hyp_mem: &mut HypPageAlloc) {
        let page_count = HYP_STACK_PAGES;
        // Unmap stack pages from the 1:1 map.
        sv48.unmap_range(
            self.paddr.as_supervisor_virt(),
            PageSize::Size4k,
            page_count,
        )
        .expect("unmapping stack physical mappings failed")
        .count();

        // Map stack pages at stack virtual address.
        let vaddr = HYP_STACK_BOTTOM_PAGE_ADDR;
        let pte_fields = PteFieldBits::leaf_with_perms(PteLeafPerms::RW);
        let mapper = sv48
            .map_range(vaddr, PageSize::Size4k, page_count, &mut || {
                hyp_mem.take_pages_for_hyp_state(1).into_iter().next()
            })
            .expect("Stack mapping failed");
        for (virt, phys) in vaddr
            .iter_from()
            .zip(self.paddr.iter_from())
            .take(page_count as usize)
        {
            // Safety: we unmapped the stack pages from the 1:1 map, so no aliases have been created
            // in this page table.
            unsafe {
                mapper.map_addr(virt, phys, pte_fields).unwrap();
            }
        }
    }
}

/// Mapping permission for a U-mode mapping slot.
pub enum UmodeSlotPerm {
    Readonly,
    Writable,
}

/// A page table that contains hypervisor mappings.
pub struct HypPageTable {
    /// The pagetable containing hypervisor mappings.
    sv48: FirstStagePageTable<Sv48>,
    /// U-mode input region for this page-table.
    umode_input: RefCell<UmodeInputRegion>,
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
            .umode_elf_regions()
            .filter(|r| r.pte_fields == PteFieldBits::leaf_with_perms(PteLeafPerms::URW))
        {
            r.restore();
        }
    }

    /// Returns a mapper for U-mode slot `slot` for `num_pages` pages.
    pub fn umode_slot_mapper(
        &self,
        slot: UmodeSlotId,
        num_pages: u64,
        slot_perm: UmodeSlotPerm,
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
        let perms = match slot_perm {
            UmodeSlotPerm::Readonly => PteFieldBits::leaf_with_perms(PteLeafPerms::UR),
            UmodeSlotPerm::Writable => PteFieldBits::leaf_with_perms(PteLeafPerms::URW),
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

    /// Copies `data` into the U-mode Input Region. The structure will be accessible read-only in
    /// U-mode at address UMODE_INPUT_START.
    pub fn copy_to_umode_input<T: DataInit>(&self, data: T) -> Result<(), Error> {
        self.umode_input.borrow_mut().store(data)
    }

    /// Clear the U-mode Input Region.
    pub fn clear_umode_input(&self) {
        self.umode_input.borrow_mut().clear()
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

/// A set of global mappings of the hypervisor that can be used to create page tables.
pub struct HypMap {
    hw_map_regions: HwMapRegionsVec,
    umode_elf_regions: UmodeElfRegionsVec,
}

impl HypMap {
    /// Creates a new hypervisor map from a hardware memory mem map and a umode ELF.
    pub fn init(mem_map: HwMemMap, umode_elf: &ElfMap<'static>) -> Result<(), Error> {
        let hw_map_regions = mem_map
            .regions()
            .filter_map(HwMapRegion::from_hw_mem_region)
            .collect();
        let umode_elf_regions = umode_elf
            .segments()
            .map(UmodeElfRegion::from_umode_elf_segment)
            .collect::<Result<_, _>>()?;
        let hypmap = HypMap {
            hw_map_regions,
            umode_elf_regions,
        };
        HYPMAP.call_once(|| hypmap);
        Ok(())
    }

    /// Gets the global reference to the Hypervisor Map.
    pub fn get() -> &'static HypMap {
        // Unwrap okay. This must be called after `init`.
        HYPMAP.get().unwrap()
    }

    // Returns an iterator for the U-mode ELF regions.
    fn umode_elf_regions(&self) -> impl Iterator<Item = &UmodeElfRegion> {
        self.umode_elf_regions.iter()
    }

    /// Returns the virtual address of U-mode mapping slot `slot`.
    pub fn umode_slot_va(slot: UmodeSlotId) -> PageAddr<SupervisorVirt> {
        match slot {
            UmodeSlotId::A => UMODE_MAPPINGS_A_PAGE_ADDR,
            UmodeSlotId::B => UMODE_MAPPINGS_B_PAGE_ADDR,
        }
    }

    pub fn copy_from_umode(dest: &mut [u8], src: RawAddr<SupervisorVirt>) -> Result<(), Error> {
        if !is_valid_umode_binary_range(src.bits(), dest.len()) {
            return Err(Error::InvalidAddress);
        }
        // Reading from user mapping, set SUM in SSTATUS.
        CSR.sstatus.modify(sstatus::sum.val(1));
        // Safety: _copy_from_user internally detects and handles an invalid u-mode address
        // in `src`, and copies at most `dest.len()` bytes.
        let bytes = unsafe { _copy_from_user(dest.as_mut_ptr(), src.bits(), dest.len()) };
        // Restore SUM.
        CSR.sstatus.modify(sstatus::sum.val(0));
        if bytes == dest.len() {
            Ok(())
        } else {
            Err(Error::UmodePageFault(src, bytes))
        }
    }

    pub fn copy_to_umode(dest: RawAddr<SupervisorVirt>, src: &[u8]) -> Result<(), Error> {
        if !is_valid_umode_binary_range(dest.bits(), src.len()) {
            return Err(Error::InvalidAddress);
        }
        // Writing to user mapping, set SUM in SSTATUS.
        CSR.sstatus.modify(sstatus::sum.val(1));
        // Safety: _copy_to_user internally detects and handles an invalid u-mode address
        // in `dest`, and copies at most `src.len()` bytes.
        let bytes = unsafe { _copy_to_user(dest.bits(), src.as_ptr(), src.len()) };
        // Restore SUM.
        CSR.sstatus.modify(sstatus::sum.val(0));
        if bytes == src.len() {
            Ok(())
        } else {
            Err(Error::UmodePageFault(dest, bytes))
        }
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
        // Map hardware map regions
        for r in &self.hw_map_regions {
            r.map(&sv48, &mut || {
                hyp_mem.take_pages_for_hyp_state(1).into_iter().next()
            });
        }
        // Map U-mode ELF region.
        for r in &self.umode_elf_regions {
            r.map(&sv48, hyp_mem);
        }
        // Alloc and map the U-mode Input Region for this page-table.
        let umode_input = UmodeInputRegion::map(&sv48, hyp_mem);
        // Alloc pte_pages for U-mode mappings.
        let pte_pages = hyp_mem
            .take_pages_for_hyp_state(Sv48::max_pte_pages(
                UMODE_MAPPINGS_SIZE / PageSize::Size4k as u64,
            ) as usize)
            .into_iter();
        HypPageTable {
            sv48,
            umode_input: RefCell::new(umode_input),
            pte_pages: RefCell::new(pte_pages),
        }
    }
}
