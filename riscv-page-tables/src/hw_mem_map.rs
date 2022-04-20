// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::{fmt, result};
use riscv_pages::{AlignedPageAddr4k, PageSize, PageSize4k, PhysAddr};

/// The maximum number of regions in a `HwMemMap`. Statically sized since we don't have
/// dynamic memory allocation at the point at which the memory map is constructed.
const MAX_HW_MEM_REGIONS: usize = 32;

type RegionVec = ArrayVec<HwMemRegion, MAX_HW_MEM_REGIONS>;

/// Represents the raw system memory map. Owns all system memory. `HwMemMap` is used as the
/// foundation of safely assigning memory ownership of system RAM, configuring it correctly is
/// _critical_ to the safety of the system.
///
/// Use `HwMemMapBuilder` to build a `HwMemMap` populated with the physical memory regions and
/// initial reserved regions with `add_memory_region()` and `reserve_region()` respectively.
/// `reserve_region()` can still be called after construction of the `HwMemMap` to reserve
/// additional ranges if necessary.
///
/// TODO: Should IO memory be included in this map? Or only regions that will end up be backed by
/// Page structs?
///
/// TODO: NUMA awareness.
#[derive(Default)]
pub struct HwMemMap {
    // Maintained in sorted order.
    regions: RegionVec,
    // Alignment required for each region in the map. Must be a multiple of the system page size.
    min_alignment: u64,
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
    mem_type: HwMemType,
    base: AlignedPageAddr4k,
    size: u64,
}

/// Describes the usage of a region in the hardware memory map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwMemType {
    /// Can be used by the hypervisor for any purpose.
    Available,

    /// The region is reserved (may be inaccessible or should not be overwritten).
    Reserved(HwReservedMemType),
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

    /// The system page map.
    PageMap,

    /// The host VM's kernel and initramfs images as loaded by firmware into (otherwise usable)
    /// memory. The hypervisor should take care not to overwrite these.
    HostKernelImage,
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
}
pub type Result<T> = result::Result<T, Error>;

impl HwMemRegion {
    pub fn mem_type(&self) -> HwMemType {
        self.mem_type
    }
    pub fn base(&self) -> AlignedPageAddr4k {
        self.base
    }
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn end(&self) -> AlignedPageAddr4k {
        // Unwrap ok because `size` must be a mutliple of the page size.
        let pages = self.size / PageSize4k::SIZE_BYTES;
        self.base.checked_add_pages(pages).unwrap()
    }
}

impl HwMemMapBuilder {
    /// Creates an empty system memory map with a minimum region alignment of `min_alignment`.
    /// Use `add_memory_region()` to populate it.
    pub fn new(min_alignment: u64) -> Self {
        assert!(PageSize4k::is_aligned(min_alignment));
        let inner = HwMemMap {
            regions: RegionVec::default(),
            min_alignment,
        };
        Self { inner }
    }

    /// Adds a range of initially-available RAM to the system map. The base address must be aligned
    /// to `min_alignment`; `size` will be rounded down if un-aligned. Must not overlap with any
    /// previously-added regions.
    ///
    /// Subsets of the region may later be marked as reserved by calling `reserve_region()`.
    ///
    /// # Safety
    ///
    /// The region must be a valid range of memory and uniquely owned by `HwMemMapBuilder`.
    pub unsafe fn add_memory_region(mut self, base: PhysAddr, size: u64) -> Result<Self> {
        if !self.inner.is_aligned(base.bits()) {
            return Err(Error::UnalignedRegion);
        }
        let base = AlignedPageAddr4k::new(base).unwrap();
        let size = self.inner.align_down(size);
        let region = HwMemRegion {
            mem_type: HwMemType::Available,
            base,
            size,
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

    /// Reserves a range of memory for the specified purpose. The range must be a subset of
    /// a previously-added available memory region and must not overlap any other reserved
    /// regions. `base` and `size` will be rounded to the nearest `min_alignment` boundary.
    pub fn reserve_region(
        mut self,
        resv_type: HwReservedMemType,
        base: PhysAddr,
        size: u64,
    ) -> Result<Self> {
        self.inner.reserve_region(resv_type, base, size)?;
        Ok(self)
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
        base: PhysAddr,
        size: u64,
    ) -> Result<()> {
        // Unwrap ok since we align `base` to `min_alignemnt` first, which is itself guaranteed
        // to be 4kB-aligned.
        let base = AlignedPageAddr4k::new(PhysAddr::new(self.align_down(base.bits()))).unwrap();
        let size = self.align_up(size);
        let region = HwMemRegion {
            mem_type: HwMemType::Reserved(resv_type),
            base,
            size,
        };
        let mut index = self
            .regions
            .iter()
            .position(|other| {
                other.mem_type() == HwMemType::Available
                    && region.base() >= other.base()
                    && region.end() <= other.end()
            })
            .ok_or(Error::InvalidReservedRegion)?;

        // Now insert, splitting if necessary.
        if region.base() > self.regions[index].base() {
            let other = self.regions[index];
            let before = HwMemRegion {
                mem_type: HwMemType::Available,
                base: other.base(),
                size: region.base().bits() - other.base().bits(),
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
                mem_type: HwMemType::Available,
                base: region.end(),
                size: end.bits() - region.end().bits(),
            };
            self.regions
                .try_insert(index, after)
                .map_err(|_| Error::OutOfSpace)?;
        }
        Ok(())
    }

    /// Returns an iterator over all regions in the memory map.
    pub fn regions(&self) -> core::slice::Iter<HwMemRegion> {
        self.regions.iter()
    }

    /// Returns true if the value is aligned to the minimum alignment.
    fn is_aligned(&self, val: u64) -> bool {
        val & (self.min_alignment - 1) == 0
    }

    /// Rounds up the given value to the minimum alignment.
    fn align_up(&self, val: u64) -> u64 {
        (val + self.min_alignment - 1) & !(self.min_alignment - 1)
    }

    /// Rounds down the given value to the minimum alignment.
    fn align_down(&self, val: u64) -> u64 {
        val & !(self.min_alignment - 1)
    }
}

impl fmt::Display for HwMemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match &self {
            HwMemType::Available => write!(f, "available"),
            HwMemType::Reserved(r) => write!(f, "reserved ({})", r),
        }
    }
}

impl fmt::Display for HwReservedMemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match &self {
            HwReservedMemType::FirmwareReserved => write!(f, "firmware"),
            HwReservedMemType::HypervisorImage => write!(f, "hypervisor image"),
            HwReservedMemType::HypervisorHeap => write!(f, "hypervisor heap"),
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
            HwMemMapBuilder::new(PageSize4k::SIZE_BYTES)
                .add_memory_region(PhysAddr::new(0x1_0000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(PhysAddr::new(0x8000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(PhysAddr::new(0x1_8000_0000), REGION_SIZE)
                .unwrap()
                .build()
        };

        let mut last = AlignedPageAddr4k::new(PhysAddr::new(0)).unwrap();
        for r in mem_map.regions() {
            assert!(last < r.base());
            assert_eq!(r.size(), REGION_SIZE);
            last = r.base();
        }
    }

    #[test]
    fn reserved_mem() {
        const REGION_SIZE: u64 = 0x4000_0000;
        let builder = unsafe {
            // Not safe -- it's a test.
            HwMemMapBuilder::new(PageSize4k::SIZE_BYTES)
                .add_memory_region(PhysAddr::new(0x8000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(PhysAddr::new(0x1_0000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(PhysAddr::new(0x1_8000_0000), REGION_SIZE)
                .unwrap()
                .add_memory_region(PhysAddr::new(0x2_0000_0000), REGION_SIZE)
                .unwrap()
        };

        let mem_map = builder
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                PhysAddr::new(0x1_1000_0000),
                0x1000_0000,
            )
            .unwrap()
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                PhysAddr::new(0x8000_0000),
                REGION_SIZE,
            )
            .unwrap()
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                PhysAddr::new(0x1_8000_0000),
                0x1000_0000,
            )
            .unwrap()
            .reserve_region(
                HwReservedMemType::FirmwareReserved,
                PhysAddr::new(0x2_3000_0000),
                0x1000_0000,
            )
            .unwrap()
            .build();

        let expected = vec![
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x8000_0000)).unwrap(),
                size: REGION_SIZE,
                mem_type: HwMemType::Reserved(HwReservedMemType::FirmwareReserved),
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x1_0000_0000)).unwrap(),
                size: 0x1000_0000,
                mem_type: HwMemType::Available,
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x1_1000_0000)).unwrap(),
                size: 0x1000_0000,
                mem_type: HwMemType::Reserved(HwReservedMemType::FirmwareReserved),
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x1_2000_0000)).unwrap(),
                size: 0x2000_0000,
                mem_type: HwMemType::Available,
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x1_8000_0000)).unwrap(),
                size: 0x1000_0000,
                mem_type: HwMemType::Reserved(HwReservedMemType::FirmwareReserved),
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x1_9000_0000)).unwrap(),
                size: 0x3000_0000,
                mem_type: HwMemType::Available,
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x2_0000_0000)).unwrap(),
                size: 0x3000_0000,
                mem_type: HwMemType::Available,
            },
            HwMemRegion {
                base: AlignedPageAddr4k::new(PhysAddr::new(0x2_3000_0000)).unwrap(),
                size: 0x1000_0000,
                mem_type: HwMemType::Reserved(HwReservedMemType::FirmwareReserved),
            },
        ];

        let zipped = expected.iter().zip(mem_map.regions());
        for (i, j) in zipped {
            assert_eq!(i.base(), j.base());
            assert_eq!(i.size(), j.size());
            assert_eq!(i.mem_type(), j.mem_type());
        }
    }
}
