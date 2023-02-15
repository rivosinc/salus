// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;

use super::error::*;

/// Identifies a group or node in the IMSIC topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImsicGroupId(u64);

impl ImsicGroupId {
    /// Creates a new `ImsicGroupId` from `id`.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw value of this `ImsicGroupId`.
    pub fn bits(&self) -> u64 {
        self.0
    }
}

/// Identifies a hart within a group in the IMSIC topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImsicHartId(u64);

impl ImsicHartId {
    /// Creates a new `ImsicHartId` from `id`.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw value of this `ImsicHartId`.
    pub fn bits(&self) -> u64 {
        self.0
    }
}

/// Identifies the interrupt file page within a hart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImsicFileId {
    /// The supervisor-level interrupt file. Always at index 0.
    Supervisor,
    /// Guest (virtual supervisor) interrupt files. Immediately follow the supervisor file.
    Guest(u32),
}

impl ImsicFileId {
    /// The `ImsicFileId` for the supervisor-level interrupt file.
    pub fn supervisor() -> Self {
        ImsicFileId::Supervisor
    }

    /// The `ImsicFileId` for the specified guest interrupt file.
    pub fn guest(guest: u32) -> Self {
        ImsicFileId::Guest(guest)
    }

    /// Returns the `ImsicFileId` corresponding to the raw `index`.
    pub fn from_index(index: u32) -> Self {
        if index == 0 {
            ImsicFileId::Supervisor
        } else {
            ImsicFileId::Guest(index - 1)
        }
    }

    /// Returns the raw value of this `ImsicFileId`.
    pub fn bits(&self) -> u32 {
        match self {
            ImsicFileId::Supervisor => 0,
            ImsicFileId::Guest(g) => g + 1,
        }
    }
}

/// Specifies the location of an IMSIC interrupt file within an IMSIC topology. Can be translated
/// to and from the page address of interrupt file using `ImsicGeometry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImsicLocation {
    group: ImsicGroupId,
    hart: ImsicHartId,
    file: ImsicFileId,
}

impl ImsicLocation {
    /// Creates a new `ImsicLocation` from its individual components.
    pub fn new(group: ImsicGroupId, hart: ImsicHartId, file: ImsicFileId) -> Self {
        Self { group, hart, file }
    }

    /// Returns the group ID of this IMSIC.
    pub fn group(&self) -> ImsicGroupId {
        self.group
    }

    /// Returns the hart ID of this IMSIC.
    pub fn hart(&self) -> ImsicHartId {
        self.hart
    }

    /// Returns the file ID of this IMSIC.
    pub fn file(&self) -> ImsicFileId {
        self.file
    }
}

/// Describes the layout of the IMSICs in a system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImsicGeometry<AS: AddressSpace> {
    base_addr: PageAddr<AS>,
    group_index_bits: u32,
    group_index_shift: u32,
    hart_index_bits: u32,
    guest_index_bits: u32,
    guests_per_hart: usize,
}

/// An `ImsicGeometry` specifying the supervisor's IMSIC layout.
pub type SupervisorImsicGeometry = ImsicGeometry<SupervisorPhys>;
/// An `ImsicGeometry` specifying a guest's IMSIC layout (in guest physical address space).
pub type GuestImsicGeometry = ImsicGeometry<GuestPhys>;

/// The guest index bits are always the least-significant bits of the PFN.
pub const GUEST_INDEX_SHIFT: u32 = 12;
/// The minimum shift for the group index bits, as mandated by the AIA specification.
pub const MIN_GROUP_INDEX_SHIFT: u32 = 24;

fn gen_mask(num_bits: u32, shift: u32) -> u64 {
    ((1 << num_bits) - 1) << shift
}

fn extract(val: u64, num_bits: u32, shift: u32) -> u64 {
    (val >> shift) & ((1 << num_bits) - 1)
}

impl<AS: AddressSpace> ImsicGeometry<AS> {
    /// Creates a new `ImsicGeometry` starting at `base_addr` (address of group 0, hart 0) with
    /// the specified index widths.
    pub fn new(
        base_addr: PageAddr<AS>,
        group_index_bits: u32,
        group_index_shift: u32,
        hart_index_bits: u32,
        guest_index_bits: u32,
        guests_per_hart: usize,
    ) -> Result<Self> {
        if guests_per_hart >= (1 << guest_index_bits) {
            return Err(Error::InvalidGuestsPerHart(guests_per_hart));
        }
        if (group_index_shift + group_index_bits) >= u64::BITS
            || group_index_shift < MIN_GROUP_INDEX_SHIFT
            || group_index_shift < (GUEST_INDEX_SHIFT + hart_index_bits + guest_index_bits)
        {
            return Err(Error::InvalidGroupIndexShift(group_index_shift));
        }
        let geo = Self {
            base_addr,
            group_index_bits,
            group_index_shift,
            hart_index_bits,
            guest_index_bits,
            guests_per_hart,
        };
        if (base_addr.bits() & geo.index_mask()) != 0 {
            return Err(Error::InvalidAddressPattern(base_addr.bits()));
        }
        Ok(geo)
    }

    /// Returns the base address of the IMSIC geometry.
    pub fn base_addr(&self) -> PageAddr<AS> {
        self.base_addr
    }

    /// Returns the number of guest index bits.
    pub fn guest_index_bits(&self) -> u32 {
        self.guest_index_bits
    }

    /// Returns the number of hart index bits.
    pub fn hart_index_bits(&self) -> u32 {
        self.hart_index_bits
    }

    /// Returns the shift of the group index bits in an IMSIC address.
    pub fn group_index_shift(&self) -> u32 {
        self.group_index_shift
    }

    /// Returns the number of group index bits.
    pub fn group_index_bits(&self) -> u32 {
        self.group_index_bits
    }

    /// Returns the number of guest files per hart.
    pub fn guests_per_hart(&self) -> usize {
        self.guests_per_hart
    }

    /// Returns the total number of index bits.
    pub fn index_bits(&self) -> u32 {
        self.group_index_bits + self.hart_index_bits + self.guest_index_bits
    }

    /// Returns a bit mask with all the index bits (guest, hart, group) set.
    pub fn index_mask(&self) -> u64 {
        let hart_mask = gen_mask(
            self.hart_index_bits + self.guest_index_bits,
            GUEST_INDEX_SHIFT,
        );
        let group_mask = gen_mask(self.group_index_bits, self.group_index_shift);
        hart_mask | group_mask
    }

    /// Returns if the specified location is valid in this geometry.
    pub fn location_is_valid(&self, loc: ImsicLocation) -> bool {
        (loc.group().bits() < (1 << self.group_index_bits))
            && (loc.hart().bits() < (1 << self.hart_index_bits))
            && ((loc.file().bits() as usize) <= self.guests_per_hart)
    }

    /// Translates the given IMSIC location to the address of the interrupt file it refers to.
    pub fn location_to_addr(&self, loc: ImsicLocation) -> Option<PageAddr<AS>> {
        if !self.location_is_valid(loc) {
            return None;
        }
        let num_pages = (loc.file().bits() as u64)
            | (loc.hart().bits() << self.guest_index_bits)
            | (loc.group().bits() << (self.group_index_shift - GUEST_INDEX_SHIFT));
        self.base_addr.checked_add_pages(num_pages)
    }

    /// Translates the given address to the corresponding IMSIC location specifier.
    pub fn addr_to_location(&self, addr: PageAddr<AS>) -> Option<ImsicLocation> {
        if (addr.bits() & !self.index_mask()) != self.base_addr.bits() {
            return None;
        }
        let group = extract(addr.bits(), self.group_index_bits, self.group_index_shift);
        let hart = extract(
            addr.bits(),
            self.hart_index_bits,
            self.guest_index_bits + GUEST_INDEX_SHIFT,
        );
        let file = extract(addr.bits(), self.guest_index_bits, GUEST_INDEX_SHIFT) as u32;
        if file as usize > self.guests_per_hart {
            return None;
        }
        let loc = ImsicLocation::new(
            ImsicGroupId::new(group),
            ImsicHartId::new(hart),
            ImsicFileId::from_index(file),
        );
        Some(loc)
    }

    /// Returns an iterator over the address ranges for each IMSIC group.
    pub fn group_ranges(&self) -> impl ExactSizeIterator<Item = PageAddrRange<AS>> {
        GroupRangeIter::new(self)
    }
}

// An iterator over the address ranges for each IMSIC group.
struct GroupRangeIter<AS: AddressSpace> {
    base_addr: PageAddr<AS>,
    group_index_shift: u32,
    index: usize,
    total: usize,
    group_pages: u64,
}

impl<AS: AddressSpace> GroupRangeIter<AS> {
    fn new(geometry: &ImsicGeometry<AS>) -> Self {
        let group_pages = 1 << (geometry.hart_index_bits() + geometry.guest_index_bits());
        Self {
            base_addr: geometry.base_addr(),
            group_index_shift: geometry.group_index_shift(),
            index: 0,
            total: 1 << geometry.group_index_bits(),
            group_pages,
        }
    }
}

impl<AS: AddressSpace> Iterator for GroupRangeIter<AS> {
    type Item = PageAddrRange<AS>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let pages = (self.index as u64) << (self.group_index_shift - GUEST_INDEX_SHIFT);
        let base = self.base_addr.checked_add_pages(pages)?;
        self.index += 1;
        Some(PageAddrRange::new(base, self.group_pages))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.total - self.index;
        (count, Some(count))
    }
}

impl<AS: AddressSpace> ExactSizeIterator for GroupRangeIter<AS> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        const NUM_GUESTS: usize = 10;
        let geometry = ImsicGeometry::new(
            PageAddr::new(RawAddr::supervisor(0x2800_0000)).unwrap(),
            0,
            MIN_GROUP_INDEX_SHIFT,
            4,
            4,
            NUM_GUESTS,
        )
        .unwrap();
        assert_eq!(geometry.index_bits(), 8);
        assert_eq!(geometry.index_mask(), 0x000f_f000);
        let h0s = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(0),
            ImsicFileId::supervisor(),
        );
        assert_eq!(geometry.location_to_addr(h0s).unwrap().bits(), 0x2800_0000);
        let h3g6 = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(3),
            ImsicFileId::guest(6),
        );
        assert_eq!(geometry.location_to_addr(h3g6).unwrap().bits(), 0x2803_7000);
        let h20s = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(20),
            ImsicFileId::supervisor(),
        );
        assert!(!geometry.location_is_valid(h20s));
        let h2g2 = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(2),
            ImsicFileId::guest(2),
        );
        assert_eq!(
            geometry
                .addr_to_location(PageAddr::new(RawAddr::supervisor(0x2802_3000)).unwrap())
                .unwrap(),
            h2g2
        );
    }

    #[test]
    fn with_groups() {
        const NUM_GUESTS: usize = 3;
        let geometry = ImsicGeometry::new(
            PageAddr::new(RawAddr::supervisor(0x2800_0000)).unwrap(),
            2,
            MIN_GROUP_INDEX_SHIFT,
            4,
            4,
            NUM_GUESTS,
        )
        .unwrap();
        assert_eq!(geometry.index_bits(), 10);
        assert_eq!(geometry.index_mask(), 0x030f_f000);
        let g0h0s = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(0),
            ImsicFileId::supervisor(),
        );
        assert_eq!(
            geometry.location_to_addr(g0h0s).unwrap().bits(),
            0x2800_0000
        );
        let g2h3g1 = ImsicLocation::new(
            ImsicGroupId::new(2),
            ImsicHartId::new(3),
            ImsicFileId::guest(1),
        );
        assert_eq!(
            geometry.location_to_addr(g2h3g1).unwrap().bits(),
            0x2a03_2000
        );
        let g1h1s = ImsicLocation::new(
            ImsicGroupId::new(1),
            ImsicHartId::new(1),
            ImsicFileId::supervisor(),
        );
        assert_eq!(
            geometry
                .addr_to_location(PageAddr::new(RawAddr::supervisor(0x2901_0000)).unwrap())
                .unwrap(),
            g1h1s
        );

        let mut range_iter = geometry.group_ranges();
        assert_eq!(range_iter.len(), 4);
        assert_eq!(range_iter.next().unwrap().base().bits(), 0x2800_0000);
        assert_eq!(range_iter.next().unwrap().base().bits(), 0x2900_0000);
        assert_eq!(range_iter.next().unwrap().length_bytes(), 0x10_0000);
    }
}
