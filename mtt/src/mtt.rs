// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::{
    InternalClean, Page, PageAddr, PageSize, Pfn, PhysPage, RawAddr, SequentialPages,
    SupervisorPageAddr, SupervisorPfn,
};
use sync::Mutex;

/// MTT related errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The MTT L2 entry corresponding to the physical address was invalid.
    /// This should be treated as an operational error.
    InvalidL2Entry,
    /// The MTT L1 entry corresponding to the physical address was invalid (encoding of 11b).
    /// This should be treated as an operational error.
    InvalidL1Entry,
    /// The MTT L2 entries for the specified 1GB range failed to pass the invariant that all
    /// 16 consecutive entries (each mapping 64MB) must have the same type.
    ///  This should be treated as an operational error.
    Invalid1GBRange,
    /// The caller specified an operation on a 1GB range, but the L2 entry for the physical
    /// address does not map a 1GB range.
    Non1GBL2Entry,
    /// The caller specified a operation on a 64MB range, but the L2 entry for the physical
    /// address doesn't map a 64MB range.
    Non64MBL2Entry,
    /// The caller specified a operation on a 4K page, but the L2 entry for the physical
    /// address doesn't map a 4K page.
    Non4KL2Entry,
    /// The caller specified an invalid physical address (too large for example)
    InvalidAddress,
    /// The address isn't aligned on a 1G boundary
    InvalidAligment1G,
    /// The address isn't aligned on a 2MB boundary
    InvalidAligment2M,
    /// The range type and length are mismatched
    RangeTypeMismatch,
    /// The specificied page size is yet supported
    UnsupportedPageSize,
    /// The caller specified an invalid number of pages for the range (example: 0,
    /// or would overflow address space)
    InvalidAddressRange,
    /// The platform initialization hasn't been completed. The implementation is expected
    /// to initialize pointers to the MTT L2 and L1 page pool using platform-FW tables.
    PlatformInitNotCompleted,
    /// The platform configuration for the MTT L2 and L1 is invalid
    InvalidMttConfiguration,
    /// The platform didn't allocate sufficient L1 page pool pages to map the specified address
    InsufficientL1Pages,
    /// L1 pages must be confidential
    NonConfidentialL1Page,
}

/// Holds the result of a MTT operation.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
/// API-related enums for non-confidential and confidential types
pub enum MttMemoryType {
    /// A confidential memory range
    Confidential,
    /// A non-confidential memory range
    NonConfidential,
}

mod mtt_const {
    pub const L2_TYPE_SHIFT: u64 = 34;
    // Bits 63:36 must be zero
    pub const L2_ZERO_MASK: u64 = 0xffff_fff0_0000_0000;
    // Bits 35:34
    pub const L2_TYPE_MASK: u64 = 0x000c_0000_0000;
    // Bits 33:0
    pub const L2_INFO_MASK: u64 = 0x0003_ffff_ffff;
    // Bits 35:34 for a 1G non-confidential region
    // By convention, this is a total of 16 identical consecutive entries, each mapping 64MB
    pub const L2_1G_NC_TYPE: u64 = 0;
    // Bits 35:34 for a 1G confidential region
    // By convention, this is a total of 16 identical consecutive entries, each mapping 64MB
    pub const L2_1G_C_TYPE: u64 = 1;
    // Bits 35:34 for a 64-MB region mapped using a bit-mask in a 4K L1-page
    pub const L2_L1_TYPE: u64 = 2;
    // Bits 35:34 for a 64-MB region mapped using a 2MB sub-region bitmask
    pub const L2_64M_TYPE: u64 = 3;
    // L2 has 1MB entries
    pub const MTT_L2_ENTRY_COUNT: usize = 1024 * 1024;
    // L2 is 8MB long
    pub const L2_BYTE_SIZE: usize = 1024 * 1024 * 8;
    // A L1 page consists of 512 8-byte entries
    pub const L1_ENTRY_COUNT: usize = 512;
    // Bits 25:17 for the 9-bit index
    pub const L1_INDEX_MASK: u64 = 0x03fe_0000;
    pub const L1_INDEX_SHIFT: u64 = 17;
    // Bits 16:12 for the 5-bit sub-index
    pub const L1_SUBINDEX_MASK: u64 = 0x1_f000;
    // 1-bit left shift for the 2-bits per entry for bits (16:12)
    pub const L1_SUBINDEX_SHIFT: u64 = 11;
    // 2-bits per entry
    pub const L1_ENTRY_MASK: u64 = 0x3;
    // Max physical address
    pub const MAX_PAGE_ADDR: u64 = 0x3fff_ffff_f000;
    // Bits 45:26
    pub const MTT_ADDR_MASK: u64 = 0x3fff_fc00_0000;
    // Shift for MTT index
    pub const MTT_INDEX_SHIFT: u64 = 26;
    // A 1G region is mapped by 16 consecutive entries of the same type
    pub const NUM_L2_1G_RANGE_ENTRIES: usize = 16;
}

use mtt_const::*;

struct Mtt1GEntry<'a> {
    // 16-entries of the same type each spanning 64MB by convention.
    entries: &'a mut [u64],
}

impl<'a> Mtt1GEntry<'a> {
    // Converts the 16 entries corresponding to 1GB address region into individual entries
    // each spanning 64MB. Each individual entry can have its own type since since it's no
    // no longer subject to the conventional requirement for all entries to have to same
    // type.
    fn split(self) {
        let val = if self.is_conf() {
            Mtt64MEntry::raw_conf_64m()
        } else {
            Mtt64MEntry::raw_non_conf_64m()
        };
        self.entries.iter_mut().for_each(|entry| *entry = val);
    }

    fn is_conf(&self) -> bool {
        self.entries[0] == Mtt1GEntry::raw_conf_1g()
    }

    /// # Safety
    /// The caller must ensure that is is safe to destroy 1G range represented by `self.entries`.
    unsafe fn set_memory_type(&mut self, memory_type: MttMemoryType) {
        let val = if memory_type == MttMemoryType::Confidential {
            Mtt1GEntry::raw_conf_1g()
        } else {
            Mtt1GEntry::raw_non_conf_1g()
        };
        self.entries.iter_mut().for_each(|entry| *entry = val);
    }

    fn new(entries: &'a mut [u64]) -> Result<Self> {
        use Error::*;
        let value = *entries.first().ok_or(InvalidL2Entry)?;
        if entries.len() == NUM_L2_1G_RANGE_ENTRIES && entries.iter().all(|entry| *entry == value) {
            Ok(Self { entries })
        } else {
            Err(InvalidL2Entry)
        }
    }

    fn raw_conf_1g() -> u64 {
        L2_1G_C_TYPE << L2_TYPE_SHIFT
    }

    fn raw_non_conf_1g() -> u64 {
        L2_1G_NC_TYPE << L2_TYPE_SHIFT
    }
}

struct Mtt64MEntry<'a> {
    entry: &'a mut u64,
}

impl<'a> Mtt64MEntry<'a> {
    fn new(entry: &'a mut u64) -> Mtt64MEntry {
        Self { entry }
    }

    fn raw_conf_64m() -> u64 {
        L2_64M_TYPE << L2_TYPE_SHIFT | 0xffff_ffff
    }

    fn raw_non_conf_64m() -> u64 {
        L2_64M_TYPE << L2_TYPE_SHIFT
    }

    fn get_bit_index(addr: SupervisorPageAddr) -> u64 {
        (addr.bits() & (64 * 1024 * 1024 - 1)) / (PageSize::Size2M as u64)
    }

    /// # Safety
    /// The caller must ensure that is is safe to destroy the contents of the entire 2MB range
    /// spanned by `addr`.
    unsafe fn set_memory_type(&mut self, addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        let bit_index = Self::get_bit_index(addr);
        if memory_type == MttMemoryType::Confidential {
            *self.entry |= 1 << bit_index;
        } else {
            *self.entry &= !(1 << bit_index);
        }
    }

    fn is_conf(&self, addr: SupervisorPageAddr) -> bool {
        let bit_index = Self::get_bit_index(addr);
        *self.entry & (1 << bit_index) != 0
    }

    // Converts an existing MTT L2 entry representing a 64MB range into a MttL1Entry.
    // The new entry will span the same 64MB range, but it will be broken up into
    // 32-regions of 2MB each. Each of the 2MB regions is mapped to 16 8-bytes entries
    // with 2-bits per 4K page in the region.
    // Note that it's OK to update the L1 page without any concern about a potential
    // race condition with an hardware initiated walk of the MTT. This is because we
    // bare converting from MTT2BPages to MTTL1Dir, and the type of MTT L2 entry remains
    // unchanged until it's atomically updated below following the L1 page write.
    fn split_to_l1_entry(self, l1_page: Page<InternalClean>) {
        let l1_page_addr = l1_page.as_bytes().as_ptr() as *mut u64;
        // Safety: We are writing in a page that's uniquely owned
        // and will write no more than 4K bytes.
        let l1_slice =
            unsafe { &mut *core::slice::from_raw_parts_mut(l1_page_addr, L1_ENTRY_COUNT) };
        let value = *self.entry;
        for i in 0usize..=31 {
            // Mark the 2MB sub-range as confidential if the bit in the 32-bit mask is set
            let write_value = if (value & (1 << i)) != 0 {
                // This represents 32 4K-pages of confidential memory in a L1-page
                0x5555_5555_5555_5555u64
            } else {
                0
            };
            l1_slice.iter_mut().skip(i * 16).take(16).for_each(|entry| {
                *entry = write_value;
            });
        }
        *self.entry = MttL1Entry::raw_l1_entry(l1_page.pfn());
    }
}

struct MttL1Entry<'a> {
    l1_page: &'a mut [u64],
}

impl<'a> MttL1Entry<'a> {
    // # Safety
    // The caller must guarantee that pfn is an uniquely owned page that isn't
    // aliased elsewhere.
    unsafe fn new(pfn: SupervisorPfn) -> Self {
        let l1_page_addr = PageAddr::from(pfn).bits() as *mut u64;
        let l1_page = core::slice::from_raw_parts_mut(l1_page_addr, L1_ENTRY_COUNT);
        Self { l1_page }
    }

    fn raw_l1_entry(pfn: SupervisorPfn) -> u64 {
        L2_L1_TYPE << L2_TYPE_SHIFT | pfn.bits()
    }

    // The L1 is a 4KB page that's used to manage the memory type (non-confidential
    // or confidential) at a 4K-page granularity. This is done by dividing the 64MB
    // region containing the physical address into 16KB regions of 4K each, and the
    // 2-bit encoding in the L1 page pointed to by the L2 entry determines the type.
    // The 12-bit index into the 4K page is computed using the 14-bits [25:12] of
    // the physical address, and then shifting right by 2.
    fn get_l1_index_and_shift(phys_addr: SupervisorPageAddr) -> (usize, u64) {
        let addr_bits = phys_addr.bits();
        let l1_index = (addr_bits & L1_INDEX_MASK) >> L1_INDEX_SHIFT;
        let sub_index = (addr_bits & L1_SUBINDEX_MASK) >> L1_SUBINDEX_SHIFT;
        (l1_index as usize, sub_index)
    }

    // Enumeration of the defined MTT L1 entry types.
    // The bit encoding is as follows:
    // 00b: The 4K region is non-confidential
    // 01b: The 4K region is confidential
    // 11b: Invalid encoding
    fn memory_type_from_l1_value(value: u64) -> Option<MttMemoryType> {
        match value {
            0 => Some(MttMemoryType::NonConfidential),
            1 => Some(MttMemoryType::Confidential),
            _ => None,
        }
    }

    fn is_conf(&self, addr: SupervisorPageAddr) -> bool {
        use MttMemoryType::*;
        let (l1_index, shift) = Self::get_l1_index_and_shift(addr);
        let value = self.l1_page[l1_index];
        Self::memory_type_from_l1_value((value >> shift) & L1_ENTRY_MASK).unwrap_or(NonConfidential)
            == Confidential
    }

    /// # Safety
    /// The caller must ensure that is is safe to destroy the contents of the page for `addr`.
    unsafe fn set_memory_type(&mut self, addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        let (l1_index, shift) = MttL1Entry::get_l1_index_and_shift(addr);
        let value = self.l1_page[l1_index];
        let bit_value = if memory_type == MttMemoryType::NonConfidential {
            0u64
        } else {
            1u64
        };
        // Clear the original 2-bits and OR in the new value
        let value = (value & !(L1_ENTRY_MASK << shift)) | (bit_value << shift);
        self.l1_page[l1_index] = value;
    }
}

// Enumeration of defined L2 entry types.
// The MTT (Memory Tracking Table) L2 is a 8MB physically contiguous range of memory allocated
// by trusted platform-FW. When enabled in hardware, it partitions memory into non-confidential
// and confidential regions. The enumerations below correspond to the values that can be encoded
// in the 64-bit entries in the table. The entries are encoded as follows:
// |--------------------------------------------------
// |  63:36: Zero | 35:34: Type | 33:0: Info
// |---------------------------------------------------
// The 2-bit encoding for Type is as follows:
// 00b: Non-confidential 1GB range
// 01b: Confidential 1GB range
// 10b: A 64-MB range further described by 16K entries in a 4K page (pointed to by Info[33:0])
// 11b: A 64-MB range composed of 32x2MB sub-ranges, and further described by Info[31:0]
enum MttL2Entry<'a> {
    // The entire 1GB range containing the address is non-confidential.
    // 1GB ranges are mapped using 64MB subranges, and by convention, the invariant is
    // that each of the 16 consecutive entries in the L2 table have the same type.
    NonConfidential1G(Mtt1GEntry<'a>),
    // The entire 1GB range containing the address is confidential.
    // 1GB ranges are mapped using 64MB subranges, and by convention, the invariant is
    // that each of the 16 consecutive entries in the L2 table have the same type.
    Confidential1G(Mtt1GEntry<'a>),
    // The 64-MB range has been partitioned into 2MB regions, and each sub-region can be
    // confidential or non-confidential
    Mixed2M(Mtt64MEntry<'a>),
    // The 64-MB range has been partitioned into 16K regions of 4KB size.
    // Each sub-region can be confidential or non-confidential
    L1Entry(MttL1Entry<'a>),
}

impl<'a> MttL2Entry<'a> {
    fn get_memory_type(&self, addr: SupervisorPageAddr) -> MttMemoryType {
        use MttL2Entry::*;
        match self {
            Confidential1G(_) => MttMemoryType::Confidential,
            NonConfidential1G(_) => MttMemoryType::NonConfidential,
            Mixed2M(mixed_2m_entry) => {
                if mixed_2m_entry.is_conf(addr) {
                    MttMemoryType::Confidential
                } else {
                    MttMemoryType::NonConfidential
                }
            }
            L1Entry(l1_entry) => {
                if l1_entry.is_conf(addr) {
                    MttMemoryType::Confidential
                } else {
                    MttMemoryType::NonConfidential
                }
            }
        }
    }

    /// # Safety
    /// The caller must ensure that is is safe to destroy the contents of the entire range
    /// spanned by `addr`.
    unsafe fn set_memory_type(&mut self, addr: SupervisorPageAddr, memory_type: MttMemoryType) {
        use MttL2Entry::*;
        match self {
            Confidential1G(ref mut conf_1g) => conf_1g.set_memory_type(memory_type),
            NonConfidential1G(ref mut nc_1g) => nc_1g.set_memory_type(memory_type),
            Mixed2M(ref mut mixed_2m_entry) => mixed_2m_entry.set_memory_type(addr, memory_type),
            L1Entry(ref mut l1_entry) => l1_entry.set_memory_type(addr, memory_type),
        }
    }

    fn page_size_from_entry(&self) -> PageSize {
        use MttL2Entry::*;
        match self {
            NonConfidential1G(_) | Confidential1G(_) => PageSize::Size1G,
            Mixed2M(_) => PageSize::Size2M,
            L1Entry(_) => PageSize::Size4k,
        }
    }
}

struct MttL2Directory {
    mtt_l2_base: &'static mut [u64],
}

impl MttL2Directory {
    // Returns the index of the MTT L2 entry for the physical address
    fn get_mtt_index(phys_addr: SupervisorPageAddr) -> usize {
        let addr = phys_addr.bits();
        ((addr & MTT_ADDR_MASK) >> MTT_INDEX_SHIFT) as usize
    }

    fn aligned_1g_addr(phys_addr: SupervisorPageAddr) -> SupervisorPageAddr {
        PageAddr::with_round_down(phys_addr.into(), PageSize::Size1G)
    }

    fn entry_for_addr(
        mtt_l2_base: &mut [u64],
        phys_addr: SupervisorPageAddr,
    ) -> Result<MttL2Entry> {
        use Error::*;
        use MttL2Entry::*;
        let mtt_index = Self::get_mtt_index(phys_addr);
        let value = mtt_l2_base[mtt_index];
        if (value & L2_ZERO_MASK) != 0 {
            return Err(InvalidL2Entry);
        }
        let mtt_l2_type = (value & L2_TYPE_MASK) >> L2_TYPE_SHIFT;
        match mtt_l2_type {
            // INFO must be 0 for 1GB entries and all 16-entries in the range
            // must be of the same type by convention.
            L2_1G_NC_TYPE if (value & L2_INFO_MASK) == 0 => {
                let mtt_index = Self::get_mtt_index(Self::aligned_1g_addr(phys_addr));
                let entry = Mtt1GEntry::new(&mut mtt_l2_base[mtt_index..=mtt_index + 15])?;
                Ok(NonConfidential1G(entry))
            }
            L2_1G_C_TYPE if (value & L2_INFO_MASK) == 0 => {
                let mtt_index = Self::get_mtt_index(Self::aligned_1g_addr(phys_addr));
                let entry = Mtt1GEntry::new(&mut mtt_l2_base[mtt_index..=mtt_index + 15])?;
                Ok(Confidential1G(entry))
            }
            L2_L1_TYPE => {
                let pfn = Pfn::supervisor(value & L2_INFO_MASK);
                // Safety: We are deferencing an existing L1 page entry in the MTT L2 table.
                // This entry was created by Salus, or was created by trusted platform-FW,
                // and the pfn corresponding to the page is guaranteed not be aliased and
                // will remain valid for the lifetime of the L2 entry.
                let l1_entry = unsafe { MttL1Entry::new(pfn) };
                Ok(L1Entry(l1_entry))
            }
            L2_64M_TYPE => Ok(Mixed2M(Mtt64MEntry::new(&mut mtt_l2_base[mtt_index]))),
            _ => Err(InvalidL2Entry),
        }
    }
}

// Helper for iterating over a MTT range.
struct MttRange<'a> {
    addr: SupervisorPageAddr,
    len: usize,
    mtt_l2_base: &'a mut [u64],
}

// Enum for the value returned by range_iterator() callback.
// IncrementAddress: Advance to the next address in the range.
// RetryAddress: Retry with the same address.
// Break: Terminate the loop and return the encapsulated result.
enum CallbackResult {
    IncrementAddress,
    RetryAddress,
    Break(Result<()>),
}

impl<'a> MttRange<'a> {
    fn new(addr: SupervisorPageAddr, len: usize, mtt_l2_base: &'a mut [u64]) -> Result<Self> {
        use Error::InvalidAddressRange;
        if !PageSize::Size4k.is_aligned(len as u64) {
            return Err(InvalidAddressRange);
        }

        let range_end = addr
            .checked_add_pages(PageSize::num_4k_pages(len as u64))
            .ok_or(InvalidAddressRange)?;
        if range_end.bits() & !MAX_PAGE_ADDR != 0 {
            return Err(InvalidAddressRange);
        }

        Ok(Self {
            addr,
            len,
            mtt_l2_base,
        })
    }

    fn range_iterator<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(SupervisorPageAddr, PageSize, MttL2Entry) -> CallbackResult,
    {
        use CallbackResult::*;
        use MttL2Entry::*;
        loop {
            if self.len == 0 {
                break Ok(());
            }
            // Unwrap OK: Range has been validated already
            let l2_entry = MttL2Directory::entry_for_addr(self.mtt_l2_base, self.addr).unwrap();
            let page_size = match l2_entry {
                Confidential1G(_) | NonConfidential1G(_)
                    if PageSize::Size1G.is_aligned(self.addr.bits())
                        && self.len >= PageSize::Size1G as usize =>
                {
                    PageSize::Size1G
                }
                Confidential1G(_) | NonConfidential1G(_) | Mixed2M(_)
                    if PageSize::Size2M.is_aligned(self.addr.bits())
                        && self.len >= PageSize::Size2M as usize =>
                {
                    PageSize::Size2M
                }
                _ => PageSize::Size4k,
            };
            match callback(self.addr, page_size, l2_entry) {
                IncrementAddress => {
                    // Unwrap OK: Range has already been validated.
                    self.addr = self.addr.checked_add_pages_with_size(1, page_size).unwrap();
                    self.len -= page_size as usize;
                }
                RetryAddress => {}
                Break(result) => break result,
            }
        }
    }
}

/// Holds references to the platform allocated L2 table in a Mutex.
/// This should be constructed by calling init() with the platform allocated backing
/// memory for the L2.
pub struct Mtt {
    inner: Mutex<MttL2Directory>,
}

/// Exposes functions to change and query the type of memory ranges in the MTT.
impl Mtt {
    // Performs sanity checks on the platform L2 table
    // TODO: Add additional sanity checks
    fn validate_mtt(&self, l2_base_addr: u64) -> Result<()> {
        use Error::*;
        let addr =
            SupervisorPageAddr::new(RawAddr::supervisor(l2_base_addr)).ok_or(InvalidAddress)?;
        if self.is_confidential(addr, L2_BYTE_SIZE)? {
            Err(InvalidMttConfiguration)
        } else {
            Ok(())
        }
    }

    /// Creates a Mtt instance.
    /// The function performs some basic sanity checks on the passed-in parameters.
    pub fn init(l2_base_addr: SequentialPages<InternalClean>) -> Result<Self> {
        use Error::*;
        if l2_base_addr.length_bytes() as usize != L2_BYTE_SIZE {
            return Err(InvalidMttConfiguration);
        }
        let l2_addr_bits = l2_base_addr.base().bits();
        // L2 must be 8MB aligned
        if l2_addr_bits & (L2_BYTE_SIZE as u64 - 1) != 0 {
            return Err(InvalidMttConfiguration);
        }
        // Safety: We are referencing an owned page that's guaranteed not be aliased
        // and will not exceed the already verified bounds.
        let mtt_l2_base: &'static mut [u64] = unsafe {
            core::slice::from_raw_parts_mut(l2_addr_bits as *mut u64, MTT_L2_ENTRY_COUNT)
        };
        let instance = Self {
            inner: Mutex::new(MttL2Directory { mtt_l2_base }),
        };
        instance.validate_mtt(l2_addr_bits)?;
        Ok(instance)
    }

    fn is_range_of_type(
        &self,
        addr: SupervisorPageAddr,
        len: usize,
        memory_type: MttMemoryType,
    ) -> Result<bool> {
        use CallbackResult::*;
        let mut l2_dir = self.inner.lock();
        let mut is_range_type_same = true;
        {
            let mut callback =
                |addr: SupervisorPageAddr, _page_size: PageSize, l2_entry: MttL2Entry| {
                    let region_memory_type = l2_entry.get_memory_type(addr);
                    if region_memory_type != memory_type {
                        is_range_type_same = false;
                        Break(Ok(()))
                    } else {
                        IncrementAddress
                    }
                };

            let mut mtt_range = MttRange::new(addr, len, l2_dir.mtt_l2_base)?;
            mtt_range.range_iterator(&mut callback)?;
        }
        Ok(is_range_type_same)
    }

    /// # Safety
    /// The caller must ensure that is is safe to destroy the contents of the entire range
    /// spanned by `addr`
    unsafe fn set_range_type(
        &mut self,
        addr: SupervisorPageAddr,
        len: usize,
        memory_type: MttMemoryType,
        get_l1_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<bool> {
        use CallbackResult::*;
        let mut mtt_needs_invalidation = false;
        let mut l2_dir = self.inner.lock();
        // The first pass simply splits the ranges if necessary. This is to ensure that
        // we can allocate a sufficient number of L1 pages. The subsequent pass actually
        // changes the type, which ensures that the entire operation is "all" or "none".
        {
            let mut mtt_range = MttRange::new(addr, len, l2_dir.mtt_l2_base)?;
            let mut callback =
                |addr: SupervisorPageAddr, page_size: PageSize, l2_entry: MttL2Entry| {
                    let region_memory_type = l2_entry.get_memory_type(addr);
                    if region_memory_type != memory_type {
                        mtt_needs_invalidation = true;
                        let target_page_size = l2_entry.page_size_from_entry();
                        if target_page_size != page_size {
                            // We are going from 1GB -> 64MB or 64MB -> L1, so
                            // just split to the level below, and retry
                            match l2_entry {
                                MttL2Entry::NonConfidential1G(nc_1g) => nc_1g.split(),
                                MttL2Entry::Confidential1G(conf_1g) => conf_1g.split(),
                                MttL2Entry::Mixed2M(mixed_2m) => {
                                    if let Some(l1_page) = get_l1_page() {
                                        mixed_2m.split_to_l1_entry(l1_page)
                                    } else {
                                        return Break(Err(Error::InsufficientL1Pages));
                                    }
                                }
                                MttL2Entry::L1Entry(_) => unreachable!(),
                            }
                            // Retry same address with the new type
                            return RetryAddress;
                        }
                    }
                    // Advance to the next address in the region
                    IncrementAddress
                };

            mtt_range.range_iterator(&mut callback)?;
        }
        {
            if mtt_needs_invalidation {
                let mut mtt_range = MttRange::new(addr, len, l2_dir.mtt_l2_base)?;
                let mut callback =
                    |addr: SupervisorPageAddr, _: PageSize, mut l2_entry: MttL2Entry| {
                        l2_entry.set_memory_type(addr, memory_type);
                        // Advance to the next address in the region
                        IncrementAddress
                    };
                mtt_range.range_iterator(&mut callback)?;
            }
        }
        Ok(mtt_needs_invalidation)
    }

    /// Marks the region of length `len` starting with `addr` as confidential in
    /// the Memory Tracking Table (MTT). On success, the function returns whether
    /// the caller should make an ECALL to invalidate the MTT.
    /// `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    /// `get_l1_page` will be called to allocate a 4K-page if the function determines
    /// that the region mapping requires a MTT mapping at a 4K granularity.
    /// # Safety
    /// The caller must ensure that is is safe to destroy the contents of the entire range
    /// spanned by `addr`, since any subsequent reads following a successful conversion
    /// will be decrypted using the key for confidential memory.
    /// The callback must ensure that the allocated page isn't aliased and remains allocated
    /// until it's explictly released (TBD), and it must not perform any MTT related
    /// operations as it will result in a deadlock.
    pub unsafe fn set_confidential(
        &mut self,
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<bool> {
        self.set_range_type(addr, len, MttMemoryType::Confidential, get_l1_page)
    }

    /// Marks the region of length `len` starting with `addr` as non-confidential in
    /// the Memory Tracking Table (MTT). On success, the function returns whether
    /// the caller should make an ECALL to invalidate the MTT.
    /// `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    /// `get_l1_page` will be called to allocate a 4K-page if the function determines
    /// that the region mapping requires a MTT mapping at a 4K granularity.
    /// # Safety
    /// The caller must ensure that is is safe to destroy the contents of the entire range
    /// spanned by `addr`, since any subsequent reads following a successful conversion
    /// will be decrypted using the key for non-confidential memory.
    /// The callback must ensure that the allocated page isn't aliased and remains allocated
    /// until it's explictly released (TBD), and it must not perform any MTT related
    /// operations as it will result in a deadlock.
    pub unsafe fn set_non_confidential(
        &mut self,
        addr: SupervisorPageAddr,
        len: usize,
        get_l1_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<bool> {
        self.set_range_type(addr, len, MttMemoryType::NonConfidential, get_l1_page)
    }

    /// Returns where region of length `len` starting with `addr` is confidential in
    /// the Memory Tracking Table (MTT).
    /// 1: `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    pub fn is_confidential(&self, addr: SupervisorPageAddr, len: usize) -> Result<bool> {
        self.is_range_of_type(addr, len, MttMemoryType::Confidential)
    }

    /// Returns where region of length `len` starting with `addr` is non-confidential in
    /// the Memory Tracking Table (MTT).
    /// 1: `len` must be a multiple of a 4K-page size. The region cannot exceed the
    /// maximum physical page address (currently defined as 46-bits of PA).
    pub fn is_non_confidential(&self, addr: SupervisorPageAddr, len: usize) -> Result<bool> {
        self.is_range_of_type(addr, len, MttMemoryType::NonConfidential)
    }
}
