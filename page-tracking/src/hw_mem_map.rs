// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::{fmt, result};
use riscv_pages::{
    DeviceMemType, MemType, PageAddr, PageSize, RawAddr, SequentialPages, SupervisorPageAddr,
    SupervisorPhysAddr,
};

/// The maximum number of regions in a `HwMemMap`. Statically sized since we don't have
/// dynamic memory allocation at the point at which the memory map is constructed.
const MAX_HW_MEM_REGIONS: usize = 64;

type RegionVec = ArrayVec<HwMemRegion, MAX_HW_MEM_REGIONS>;

/// Represents the raw system memory map. Owns all system memory. `HwMemMap` is used as the
/// foundation of safely assigning memory ownership of system RAM, configuring it correctly is
/// _critical_ to the safety of the system.
///
/// Use `HwMemMapBuilder` to build a `HwMemMap` populated with the physical memory and MMIO regions,
/// as well as the initial reserved physical memory regions. Reserved regions and MMIO regions can
/// still be added after construction of the `HwMemMap` as additional reserved regions are created
/// or devices are discovered.
///
/// TODO: NUMA awareness.
#[derive(Default, Debug)]
pub struct HwMemMap {
    // Maintained in sorted order.
    regions: RegionVec,
    // Alignment required for physical memory regions (and reserved sub-regions) in the map. Must be
    // a multiple of the system page size.
    min_ram_alignment: u64,
}

/// A builder for a `HwMemMap`. Call `add_memory_region()` once for each range of physical memory
/// in the system and `reserve_range()` for each range of memory that should be reserved from the
/// start. The constructed `HwMemMap` must be the unique owner of the memory pointed to by the map.
pub struct HwMemMapBuilder {
    inner: HwMemMap,
}

/// Describes a contiguous region in the hardware memory map.
#[derive(Debug, Clone, Copy)]
pub struct HwMemRegion {
    region_type: HwMemRegionType,
    base: SupervisorPageAddr,
    size: u64,
    page_size: Option<PageSize>,
}

/// Describes the usage of a region in the hardware memory map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwMemRegionType {
    /// Physical memory that can be used by the hypervisor for any purpose.
    Available,

    /// Physical memory that is reserved (may be inaccessible or should not be overwritten).
    Reserved(HwReservedMemType),

    /// Memory-mapped IO.
    Mmio(DeviceMemType),
}

impl From<HwMemRegionType> for MemType {
    fn from(region_type: HwMemRegionType) -> MemType {
        match region_type {
            HwMemRegionType::Mmio(dev) => MemType::Mmio(dev),
            _ => MemType::Ram,
        }
    }
}

/// Describes the purpose of a reserved region in the hardware memory mpa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwReservedMemType {
    /// Firmware says this region is reserved. Can't be used for any purpose and should not
    /// be accessed.
    FirmwareReserved,

    /// The hypervisor image itself (code, data, stack, FDT from firmware). Can't be re-used
    /// for anything else.
    HypervisorImage,

    /// The hypervisor heap. For boot-time dynamic memory allocation.
    HypervisorHeap,

    /// The hypervisor per-CPU memory area.
    HypervisorPerCpu,

    /// Sv48 page tables for the hyeprvisor running in HS mode.
    HypervisorPtes,

    /// The system page map.
    PageMap,

    /// The host VM's kernel image as loaded by firmware into (otherwise usable) memory. The
    /// hypervisor should take care not to overwrite these.
    HostKernelImage,

    /// The host VM's initramfs image as loaded by firmware into (otherwise usable) memory. The
    /// hypervisor should take care not to overwrite these.
    HostInitramfsImage,
}

/// Errors that can be raised while building the memory map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Memory region size is unaligned.
    UnalignedRegion,

    /// Memory region overlaps with another one.
    OverlappingRegion,

    /// Reserved region isn't a subset of an existing memory region.
    InvalidReservedRegion,

    /// No more entries available in the memory map.
    OutOfSpace,

    /// Creation of SequentialPages failed
    SequentialPages,
}
/// Holds the result of memory map operations.
pub type Result<T> = result::Result<T, Error>;

impl HwMemRegion {
    /// Returns the type of the memory region.
    pub fn region_type(&self) -> HwMemRegionType {
        self.region_type
    }

    /// Returns the 4kB page-aligned base adddress of the region.
    pub fn base(&self) -> SupervisorPageAddr {
        self.base
    }

    /// Returns the total size of the region.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the 4kB page-aligned base adddress of the region.
    pub fn end(&self) -> SupervisorPageAddr {
        // Unwrap ok because `size` must be a mutliple of the page size.
        let pages = self.size / PageSize::Size4k as u64;
        self.base.checked_add_pages(pages).unwrap()
    }

    /// Returns the page size if this has been partitioned, or None if it hasn't
    pub fn page_size(&self) -> Option<PageSize> {
        self.page_size
    }

    /// Break a `HwMemRegion` into multiple `HwMemRegion`s that can all fit into vm pages of the same size
    fn partition(&self, page_size: PageSize) -> RegionVec {
        let mut array_vec = RegionVec::default();

        // Degenerate case
        if self.size == 0 {
            return array_vec;
        }

        // Current size too small to hold a page, go to a smaller page
        // unwrap() won't fail because memory regions should be 4k aligned
        if self.size < page_size as u64 {
            return self.partition(page_size.down_size().unwrap());
        }

        // We have at least one page. If this is aligned we are done.
        if page_size.is_aligned(self.base.bits()) && page_size.is_aligned(self.size) {
            let mut new_reg = *self;
            new_reg.page_size = Some(page_size);
            array_vec.push(new_reg);
            return array_vec;
        }

        // If this isn't aligned and couldn't contain a page size (i.e it is less than 2 pages long)
        // down_size the page and try again
        // unwrap() won't fail because memory regions should be 4k aligned
        if page_size.num_pages_contained_in(self.size) < 2
            && !page_size.is_aligned(self.base.bits())
        {
            return self.partition(page_size.down_size().unwrap());
        }

        // Break into 3 parts;
        // Bottom: base <-> 1st pagesize aligned address
        // Middle: 1st pagesize aligned address <-> last pagesize aligned address
        // Top: last pagesize aligned address <-> end
        // First or last may be 0, handled recursively
        //
        //memory, from low to high addresses
        //-----------  <== bottom_addr (start of bottom)
        //-                 (bottom)
        //-            <== middle_start_addr (end of bottom, start of middle)
        //-                 (middle)
        //-            <== middle_end_addr   (end of middle, start of top)
        //-                 (top)
        //-----------  <== top_end_addr (end of top)
        //

        let bottom_start_addr = self.base.bits();
        let middle_start_addr = page_size.round_up(self.base().bits());
        let middle_end_addr = page_size.round_down(self.end().bits());
        let top_end_addr = self.base.bits() + self.size();

        // Bottom
        // unwrap() won't fail because memory regions should be 4k aligned
        let bottom_len = middle_start_addr - bottom_start_addr;
        HwMemRegion {
            size: bottom_len,
            page_size: page_size.down_size(),
            ..*self
        }
        .partition(page_size.down_size().unwrap())
        .into_iter()
        .for_each(|m| array_vec.push(m));

        // Middle (find offset to middle in 4k pages to work with checked_add_pages() )
        // unwrap() won't fail because memory regions should be 4k aligned
        let middle_start_pages = PageSize::num_4k_pages(bottom_len);
        let middle_start = self.base.checked_add_pages(middle_start_pages).unwrap();

        let middle_len = middle_end_addr - middle_start_addr;
        HwMemRegion {
            base: middle_start,
            size: middle_len,
            page_size: Some(page_size),
            ..*self
        }
        .partition(page_size)
        .into_iter()
        .for_each(|m| array_vec.push(m));

        // Top
        // unwrap() won't fail because memory regions should be 4k aligned
        let top_start_pages = PageSize::num_4k_pages(bottom_len + middle_len);
        let top_start = self.base.checked_add_pages(top_start_pages).unwrap();

        HwMemRegion {
            base: top_start,
            size: top_end_addr - middle_end_addr,
            page_size: page_size.down_size(),
            ..*self
        }
        .partition(page_size.down_size().unwrap())
        .into_iter()
        .for_each(|m| array_vec.push(m));

        array_vec
    }
}

impl HwMemMapBuilder {
    /// Creates an empty system memory map with a minimum physical memory region alignment of
    /// `min_ram_alignment`. Use `add_memory_region()` and `add_mmio_region()` to populate it.
    pub fn new(min_ram_alignment: u64) -> Self {
        assert!(PageSize::Size4k.is_aligned(min_ram_alignment));
        let inner = HwMemMap {
            regions: RegionVec::default(),
            min_ram_alignment,
        };
        Self { inner }
    }

    /// Adds a range of initially-available RAM to the system map. The base address must be aligned
    /// to `min_ram_alignment`; `size` will be rounded down if un-aligned. Must not overlap with any
    /// previously-added regions.
    ///
    /// Subsets of the region may later be marked as reserved by calling `reserve_region()`.
    ///
    /// # Safety
    ///
    /// The region must be a valid range of memory and uniquely owned by `HwMemMapBuilder`.
    pub unsafe fn add_memory_region(mut self, base: SupervisorPhysAddr, size: u64) -> Result<Self> {
        if !self.inner.is_aligned(base.bits()) {
            return Err(Error::UnalignedRegion);
        }
        let base = PageAddr::new(base).unwrap();
        let size = self.inner.align_down(size);
        let region = HwMemRegion {
            region_type: HwMemRegionType::Available,
            base,
            size,
            page_size: None,
        };
        let mut index = 0;
        for other in &self.inner.regions {
            if other.base() > region.base() {
                if region.end() > other.base() {
                    return Err(Error::OverlappingRegion);
                }
                break;
            } else if region.base() < other.end() {
                return Err(Error::OverlappingRegion);
            }
            index += 1;
        }
        self.inner
            .regions
            .try_insert(index, region)
            .map_err(|_| Error::OutOfSpace)?;
        Ok(self)
    }

    /// Reserves a range of RAM for the specified purpose. The range must be a subset of
    /// a previously-added available memory region and must not overlap any other reserved
    /// regions. `base` and `size` will be rounded to the nearest `min_ram_alignment` boundary.
    pub fn reserve_region(
        mut self,
        resv_type: HwReservedMemType,
        base: SupervisorPhysAddr,
        size: u64,
    ) -> Result<Self> {
        self.inner.reserve_region(resv_type, base, size)?;
        Ok(self)
    }

    /// Adds a range of MMIO to the system map. The base address will be rounded down and the size
    /// will be rounded up to a multiple of the system page size. Must not overlap with any physical
    /// memory regions, or other MMIO regions.
    ///
    /// # Safety
    ///
    /// The region must be a valid range of MMIO and uniquely owned by `HwMemMapBuilder`.
    pub unsafe fn add_mmio_region(
        mut self,
        dev_type: DeviceMemType,
        base: SupervisorPhysAddr,
        size: u64,
    ) -> Result<Self> {
        self.inner.add_mmio_region(dev_type, base, size)?;
        Ok(self)
    }

    /// Go through all of the `HwMemRegion`s, and partition them into exactly pageable sizes
    /// After running this function all `Available` memory will be split into regions
    /// such that each region can be split exactly into pages and the `page_size` field
    /// of the `HwMemRegion` will be set.
    pub fn page_size_partition(mut self) -> Self {
        let rv: RegionVec = self
            .inner
            .regions()
            .flat_map(|f| {
                if f.region_type() == HwMemRegionType::Available {
                    f.partition(PageSize::Size512G)
                } else {
                    RegionVec::from_iter(Some(*f))
                }
            })
            .collect();

        self.inner.regions = rv;
        self
    }

    /// Returns the constructed HwMemMap.
    pub fn build(self) -> HwMemMap {
        self.inner
    }
}

impl HwMemMap {
    /// Reserves a range of memory. See `HwMemMapBuilder::reserve_region()`.
    pub fn reserve_region(
        &mut self,
        resv_type: HwReservedMemType,
        base: SupervisorPhysAddr,
        size: u64,
    ) -> Result<()> {
        // Unwrap ok since we align `base` to `min_ram_alignemnt` first, which is itself guaranteed
        // to be 4kB-aligned.
        let base = PageAddr::new(RawAddr::supervisor(self.align_down(base.bits()))).unwrap();
        let size = self.align_up(size);
        let region = HwMemRegion {
            region_type: HwMemRegionType::Reserved(resv_type),
            base,
            size,
            page_size: None,
        };
        let mut index = self
            .regions
            .iter()
            .position(|other| {
                other.region_type() == HwMemRegionType::Available
                    && region.base() >= other.base()
                    && region.end() <= other.end()
            })
            .ok_or(Error::InvalidReservedRegion)?;

        // Now insert, splitting if necessary.
        if region.base() > self.regions[index].base() {
            let other = self.regions[index];
            let before = HwMemRegion {
                region_type: HwMemRegionType::Available,
                base: other.base(),
                size: region.base().bits() - other.base().bits(),
                page_size: None,
            };
            self.regions
                .try_insert(index, before)
                .map_err(|_| Error::OutOfSpace)?;
            index += 1;
        }
        let end = self.regions[index].end();
        self.regions[index] = region;
        index += 1;
        if region.end() < end {
            let after = HwMemRegion {
                region_type: HwMemRegionType::Available,
                base: region.end(),
                size: end.bits() - region.end().bits(),
                page_size: None,
            };
            self.regions
                .try_insert(index, after)
                .map_err(|_| Error::OutOfSpace)?;
        }
        Ok(())
    }

    /// Reserves a range of memory. See `HwMemMapBuilder::reserve_region()`. Then returns the
    /// reserved pages as a `SequentialPages` type.
    pub fn reserve_and_take_pages(
        &mut self,
        resv_type: HwReservedMemType,
        base: SupervisorPageAddr,
        page_size: PageSize,
        count: u64,
    ) -> Result<SequentialPages<riscv_pages::InternalDirty>> {
        self.reserve_region(resv_type, base.into(), count * page_size as u64)?;

        // Safe to create pages from these addresses because they are guaranteed to be uniquely
        // owned by the above reservation.
        unsafe {
            SequentialPages::<riscv_pages::InternalDirty>::from_mem_range(base, page_size, count)
                .map_err(|_| Error::SequentialPages)
        }
    }

    /// Adds an MMIO region. See `HwMemMapBuilder::add_mmio_region()`.
    ///
    /// # Safety
    ///
    /// The region must be a valid range of MMIO and uniquely owned by `HwMemMapBuilder`.
    pub unsafe fn add_mmio_region(
        &mut self,
        dev_type: DeviceMemType,
        base: SupervisorPhysAddr,
        size: u64,
    ) -> Result<()> {
        let base = PageAddr::with_round_down(base, PageSize::Size4k);
        let size = PageSize::Size4k.round_up(size);
        let region = HwMemRegion {
            region_type: HwMemRegionType::Mmio(dev_type),
            base,
            size,
            page_size: None,
        };
        let mut index = 0;
        for other in &self.regions {
            if other.base() > region.base() {
                if region.end() > other.base() {
                    return Err(Error::OverlappingRegion);
                }
                break;
            } else if region.base() < other.end() {
                return Err(Error::OverlappingRegion);
            }
            index += 1;
        }
        self.regions
            .try_insert(index, region)
            .map_err(|_| Error::OutOfSpace)?;
        Ok(())
    }

    /// Returns an iterator over all regions in the memory map.
    pub fn regions(&self) -> core::slice::Iter<HwMemRegion> {
        self.regions.iter()
    }

    /// Returns true if the value is aligned to the minimum alignment.
    fn is_aligned(&self, val: u64) -> bool {
        val & (self.min_ram_alignment - 1) == 0
    }

    /// Rounds up the given value to the minimum alignment.
    fn align_up(&self, val: u64) -> u64 {
        (val + self.min_ram_alignment - 1) & !(self.min_ram_alignment - 1)
    }

    /// Rounds down the given value to the minimum alignment.
    fn align_down(&self, val: u64) -> u64 {
        val & !(self.min_ram_alignment - 1)
    }
}

impl fmt::Display for HwMemRegionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match &self {
            HwMemRegionType::Available => write!(f, "available"),
            HwMemRegionType::Reserved(r) => write!(f, "reserved ({})", r),
            HwMemRegionType::Mmio(d) => write!(f, "mmio ({})", d),
        }
    }
}

impl fmt::Display for HwReservedMemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match &self {
            HwReservedMemType::FirmwareReserved => write!(f, "firmware"),
            HwReservedMemType::HypervisorImage => write!(f, "hypervisor image"),
            HwReservedMemType::HypervisorHeap => write!(f, "hypervisor heap"),
            HwReservedMemType::HypervisorPerCpu => write!(f, "hypervisor pcpu"),
            HwReservedMemType::HypervisorPtes => write!(f, "hypervisor page tables"),
            HwReservedMemType::PageMap => write!(f, "page map"),
            HwReservedMemType::HostKernelImage => write!(f, "host kernel"),
            HwReservedMemType::HostInitramfsImage => write!(f, "host initramfs"),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn mem_map_ordering() {
        const REGION_SIZE: u64 = 0x4000_0000;
        let mem_map = unsafe {
            // Not safe -- it's a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(RawAddr::supervisor(0x1_0000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(RawAddr::supervisor(0x8000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(RawAddr::supervisor(0x1_8000_0000), REGION_SIZE)
                .unwrap()
                .build()
        };

        let mut last = PageAddr::new(RawAddr::supervisor(0)).unwrap();
        for r in mem_map.regions() {
            assert!(last < r.base());
            assert_eq!(r.size(), REGION_SIZE);
            last = r.base();
        }
    }

    #[test]
    fn mmio_and_reserved_mem() {
        const REGION_SIZE: u64 = 0x4000_0000;
        let builder = unsafe {
            // Not safe -- it's a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(RawAddr::supervisor(0x8000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(RawAddr::supervisor(0x1_0000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(RawAddr::supervisor(0x1_8000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(RawAddr::supervisor(0x2_0000_0000), REGION_SIZE)
                .unwrap()
                .add_mmio_region(
                    DeviceMemType::Imsic,
                    RawAddr::supervisor(0x4000_0000),
                    0x10_0000,
                )
                .unwrap()
        };

        let mem_map = builder
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x1_1000_0000),
                0x1000_0000,
            )
            .unwrap()
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x8000_0000),
                REGION_SIZE,
            )
            .unwrap()
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x1_8000_0000),
                0x1000_0000,
            )
            .unwrap()
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x2_3000_0000),
                0x1000_0000,
            )
            .unwrap()
            .build();

        let expected = vec![
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x4000_0000)).unwrap(),
                size: 0x10_0000,
                region_type: HwMemRegionType::Mmio(DeviceMemType::Imsic),
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x8000_0000)).unwrap(),
                size: REGION_SIZE,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_0000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Available,
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_1000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_2000_0000)).unwrap(),
                size: 0x2000_0000,
                region_type: HwMemRegionType::Available,
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_8000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_9000_0000)).unwrap(),
                size: 0x3000_0000,
                region_type: HwMemRegionType::Available,
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x2_0000_0000)).unwrap(),
                size: 0x3000_0000,
                region_type: HwMemRegionType::Available,
                page_size: None,
            },
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x2_3000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
        ];

        let zipped = expected.iter().zip(mem_map.regions());
        for (i, j) in zipped {
            assert_eq!(i.base(), j.base());
            assert_eq!(i.size(), j.size());
            assert_eq!(i.region_type(), j.region_type());
        }
    }

    #[test]
    fn test_builder_partitioning() {
        const REGION_SIZE: u64 = 0x4000_0000;
        const S4K: u64 = PageSize::Size4k as u64;
        const S2M: u64 = PageSize::Size2M as u64;

        let mut builder = unsafe {
            // Not safe -- it's a test.
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                // region 1) 0x8000_0000 <-> 0xC000_0000
                .add_memory_region(RawAddr::supervisor(0x8000_0000), REGION_SIZE)
                .unwrap()
                // region 2) 0x1_0000_0000 <-> 0x1_4000_0000
                .add_memory_region(RawAddr::supervisor(0x1_0000_0000), REGION_SIZE)
                .unwrap()
                // region 3) 0x4_0000_0000 <-> 0x4_4000_4000
                .add_memory_region(RawAddr::supervisor(0x4_0000_0000), REGION_SIZE)
                .unwrap()
                // region 4) 0x5_0000_1000 <-> 0x 5_4000_1000
                .add_memory_region(RawAddr::supervisor(0x5_0000_0000 + S4K), REGION_SIZE)
                .unwrap()
                // region 5) 0x5_FFE0_0000 <-> 0x6_3FE0_0000
                .add_memory_region(RawAddr::supervisor(0x6_0000_0000 - S2M), REGION_SIZE)
                .unwrap()
                // region 6) 0x6_FFE0_0000 <-> 0x7_3FE0_0000
                .add_memory_region(RawAddr::supervisor(0x7_0000_0000 - S2M), REGION_SIZE)
                .unwrap()
                // region 7) 0x7_FFC0_1000 <-> 0x8_7FC0_0FFC
                .add_memory_region(
                    RawAddr::supervisor(0x8_0000_0000 - S2M + S4K),
                    REGION_SIZE * 2,
                )
                // region 8) 0x10_0000 <-> 0x20_0000
                .unwrap()
                .add_mmio_region(
                    DeviceMemType::Imsic,
                    RawAddr::supervisor(0x10_0000),
                    0x10_0000,
                )
                .unwrap()
        };

        builder = builder
            // reserve 1) 0x1_1000_0000 <-> 0x1_2000_0000
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x1_1000_0000),
                0x1000_0000,
            )
            .unwrap()
            // reserve 2) 0x8000_0000 <-> 0xC000_0000
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x8000_0000),
                REGION_SIZE,
            )
            .unwrap()
            // reserve 3) 0x1_0000_0000 <-> 0x1_1000_0000
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x1_0000_0000),
                0x1000_0000,
            )
            .unwrap()
            // reserve 4 0x7_0000_0000 <-> 0x7_1000_0000
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                RawAddr::supervisor(0x7_0000_0000),
                0x1000_0000,
            )
            .unwrap();

        builder = builder.page_size_partition();
        let mem_map = builder.build();

        let expected = vec![
            // region 8
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x10_0000)).unwrap(),
                size: 0x10_0000,
                region_type: HwMemRegionType::Mmio(DeviceMemType::Imsic),
                page_size: None,
            },
            // region 1, reserve 2
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x8000_0000)).unwrap(),
                size: REGION_SIZE,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            // region 2, reserve 3
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_0000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            // region 2, reserve 1
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_1000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            // region 2
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x1_2000_0000)).unwrap(),
                size: 0x2000_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size2M),
            },
            // region 3
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x4_0000_0000)).unwrap(),
                size: 0x4000_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size1G),
            },
            // region 4a
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x5_0000_1000)).unwrap(),
                size: 0x1f_f000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size4k),
            },
            // region 4b
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x5_0020_0000)).unwrap(),
                size: 0x3fe0_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size2M),
            },
            // region 4c
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x5_4000_0000)).unwrap(),
                size: 0x1000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size4k),
            },
            // region5
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x5_ffe0_0000)).unwrap(),
                size: 0x4000_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size2M),
            },
            // region 6a
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x6_ffe0_0000)).unwrap(),
                size: 0x20_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size2M),
            },
            // region 6, reserve 4
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x7_0000_0000)).unwrap(),
                size: 0x1000_0000,
                region_type: HwMemRegionType::Reserved(HwReservedMemType::FirmwareReserved),
                page_size: None,
            },
            // region 6b
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x7_1000_0000)).unwrap(),
                size: 0x2fe00000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size2M),
            },
            // region 7a
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x7_ffe0_1000)).unwrap(),
                size: 0x1f_f000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size4k),
            },
            // region 7b
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x8_0000_0000)).unwrap(),
                size: 0x4000_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size1G),
            },
            // region 7c
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x8_4000_0000)).unwrap(),
                size: 0x3fe0_0000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size2M),
            },
            // region 7d
            HwMemRegion {
                base: PageAddr::new(RawAddr::supervisor(0x87fe00000)).unwrap(),
                size: 0x1000,
                region_type: HwMemRegionType::Available,
                page_size: Some(PageSize::Size4k),
            },
        ];

        let zipped = expected.iter().zip(mem_map.regions());
        for (i, j) in zipped {
            assert_eq!(i.base(), j.base());
            assert_eq!(i.size(), j.size());
            assert_eq!(i.region_type(), j.region_type());
            assert_eq!(i.page_size(), j.page_size())
        }
    }
}
