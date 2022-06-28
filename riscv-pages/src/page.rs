// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
/// Represents pages of memory.
use core::slice;

use crate::state::*;
use crate::{AddressSpace, GuestPhys, MemType, PageOwnerId, SupervisorPhys, SupervisorVirt};

// PFN constants, currently sv48x4 hard-coded
// TODO parameterize based on address mode
const PFN_SHIFT: u64 = 12;
const PFN_BITS: u64 = 44;
const PFN_MASK: u64 = (1 << PFN_BITS) - 1;

/// The page sizes supported by Risc-V.
#[repr(u64)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PageSize {
    /// Page
    Size4k = 4 * 1024,
    /// Mega
    Size2M = 2 * 1024 * 1024,
    /// Giga
    Size1G = 1024 * 1024 * 1024,
    /// Tera
    Size512G = 512 * 1024 * 1024 * 1024,
}

impl PageSize {
    /// Returns `val` divided by 4kB, rounded up.
    pub const fn num_4k_pages(val: u64) -> u64 {
        (val + PageSize::Size4k as u64 - 1) / (PageSize::Size4k as u64)
    }

    /// Checks if the given quantity is aligned to this page size.
    pub fn is_aligned(&self, val: u64) -> bool {
        (val & (*self as u64 - 1)) == 0
    }

    /// Rounds up the quantity to the nearest multiple of this page size.
    pub fn round_up(&self, val: u64) -> u64 {
        (val + *self as u64 - 1) & !(*self as u64 - 1)
    }

    /// Rounds down the quantity to the nearest multiple of this page size.
    pub fn round_down(&self, val: u64) -> u64 {
        val & !(*self as u64 - 1)
    }

    /// Returns if the size is a huge page (> 4kB) size.
    pub fn is_huge(&self) -> bool {
        !matches!(*self, PageSize::Size4k)
    }
}

/// A raw address in an address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RawAddr<AS: AddressSpace>(u64, AS);

impl<AS: AddressSpace> RawAddr<AS> {
    /// Creates a `RawAddr` at `addr` in the given `address_space`.
    pub fn new(addr: u64, address_space: AS) -> Self {
        Self(addr, address_space)
    }

    /// Returns the inner 64 address.
    pub fn bits(&self) -> u64 {
        self.0
    }

    /// Returns the address space for the address.
    pub fn address_space(&self) -> AS {
        self.1
    }

    /// Returns the address incremented by the given number of bytes.
    /// Returns None if the result would overlfow.
    pub fn checked_increment(&self, increment: u64) -> Option<Self> {
        let addr = self.0.checked_add(increment)?;
        Some(Self(addr, self.1))
    }
}

impl RawAddr<SupervisorPhys> {
    /// Creates a `RawAddr` in the `SupervisorPhys` address space.
    /// Short for `RawAddr::new(addr, SupervisorPhys)`.
    pub fn supervisor(addr: u64) -> Self {
        Self(addr, SupervisorPhys)
    }
}

impl RawAddr<SupervisorVirt> {
    /// Creates a `RawAddr` in the `SupervisorVirt` address space.
    /// Short for `RawAddr::new(addr, SupervisorVirt)`.
    pub fn supervisor_virt(addr: u64) -> Self {
        Self(addr, SupervisorVirt)
    }
}

impl RawAddr<GuestPhys> {
    /// Creates a `RawAddr` in the `GuestPhys` address space of the VM provided by `PageOwnerId`.
    /// Short for `RawAddr::new(addr, GuestPhys::new(id))`.
    pub fn guest(addr: u64, id: PageOwnerId) -> Self {
        Self(addr, GuestPhys::new(id))
    }
}

/// Convenience type alias for supervisor-physical addresses.
pub type SupervisorPhysAddr = RawAddr<SupervisorPhys>;
/// Convenience type alias for guest-physical addresses.
pub type GuestPhysAddr = RawAddr<GuestPhys>;

impl<AS: AddressSpace> From<PageAddr<AS>> for RawAddr<AS> {
    fn from(p: PageAddr<AS>) -> RawAddr<AS> {
        p.addr
    }
}

impl<AS: AddressSpace> PartialOrd for RawAddr<AS> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

/// An address of a Page in an address space. It is guaranteed to be aligned to a page boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PageAddr<AS: AddressSpace> {
    addr: RawAddr<AS>,
}

/// A page-aligned address in the `SupervisorPhys` address space.
pub type SupervisorPageAddr = PageAddr<SupervisorPhys>;
/// A page-aligned address in a `GuestPhys` address space.
pub type GuestPageAddr = PageAddr<GuestPhys>;

impl<AS: AddressSpace> PageAddr<AS> {
    /// Creates a 4kB-aligned `PageAddr` from a `RawAddr`, returning `None` if the address isn't
    /// aligned.
    pub fn new(addr: RawAddr<AS>) -> Option<Self> {
        Self::with_alignment(addr, PageSize::Size4k)
    }

    /// Creates a `PageAddr` from a `RawAddr`, returns `None` if the address isn't aligned to the
    /// requested page size.
    pub fn with_alignment(addr: RawAddr<AS>, alignment: PageSize) -> Option<Self> {
        if alignment.is_aligned(addr.bits()) {
            Some(PageAddr { addr })
        } else {
            None
        }
    }

    /// Creates a `PageAddr` from a `Pfn`.
    pub fn from_pfn(pfn: Pfn<AS>, alignment: PageSize) -> Option<Self> {
        let phys_addr = RawAddr(pfn.0 << PFN_SHIFT, pfn.1);
        Self::with_alignment(phys_addr, alignment)
    }

    /// Creates a `PageAddr` from a `RawAddr`, rounding up to the nearest multiple of the page
    /// size.
    pub fn with_round_up(addr: RawAddr<AS>, alignment: PageSize) -> Self {
        Self {
            addr: RawAddr::new(alignment.round_up(addr.bits()), addr.address_space()),
        }
    }

    /// Same as above, but rounding down to the nearest multiple of the page size.
    pub fn with_round_down(addr: RawAddr<AS>, alignment: PageSize) -> Self {
        Self {
            addr: RawAddr::new(alignment.round_down(addr.bits()), addr.address_space()),
        }
    }

    /// Gets the raw bits of the page address.
    pub fn bits(&self) -> u64 {
        self.addr.0
    }

    /// Returns if this address is aligned to the given page size.
    pub fn is_aligned(&self, alignment: PageSize) -> bool {
        alignment.is_aligned(self.addr.0)
    }

    /// Iterates from this address in 4kB chunks.
    pub fn iter_from(&self) -> PageAddrIter<AS> {
        PageAddrIter::new(*self, PageSize::Size4k).unwrap()
    }

    /// Iterates from this address in `page_size` chunks, if this address is properly aligned.
    pub fn iter_from_with_size(&self, page_size: PageSize) -> Option<PageAddrIter<AS>> {
        PageAddrIter::new(*self, page_size)
    }

    /// Gets the pfn of the page address.
    pub fn pfn(&self) -> Pfn<AS> {
        Pfn::new((self.addr.0 >> PFN_SHIFT) & PFN_MASK, self.addr.1)
    }

    /// Adds `n` 4kB pages to the current address.
    pub fn checked_add_pages(&self, n: u64) -> Option<Self> {
        self.checked_add_pages_with_size(n, PageSize::Size4k)
    }

    /// Adds `n` `page_size`-sized pages to the current address if the address is properly aligned.
    pub fn checked_add_pages_with_size(&self, n: u64, page_size: PageSize) -> Option<Self> {
        n.checked_mul(page_size as u64)
            .and_then(|inc| self.addr.checked_increment(inc))
            .and_then(|addr| Self::with_alignment(addr, page_size))
    }

    /// Gets the index of the page in the system (the linear page count from address 0).
    pub fn index(&self) -> usize {
        self.pfn().bits() as usize
    }
}

impl<AS: AddressSpace> PartialOrd for PageAddr<AS> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.addr.partial_cmp(&other.addr)
    }
}

/// Generate page addresses for the given size. 4096, 8192, 12288, ... until the end of u64's range
pub struct PageAddrIter<AS: AddressSpace> {
    next: Option<PageAddr<AS>>,
    increment: PageSize,
}

impl<AS: AddressSpace> PageAddrIter<AS> {
    /// Creates a new `PageAddrIter` starting at the page `start` and incrementing by the size given
    /// in `increment`.
    ///
    /// # Example
    /// ```rust
    /// use riscv_pages::{PageAddr, PageAddrIter, PageSize, RawAddr};
    /// let start = PageAddr::new(RawAddr::supervisor(0x8000_0000)).ok_or(())?;
    /// let mut addr_iter = PageAddrIter::new(start, PageSize::Size4k).ok_or(())?;
    /// assert_eq!(Some(start), addr_iter.next());
    /// assert_eq!(
    ///     Some(start.checked_add_pages(1).ok_or(())?),
    ///     addr_iter.next()
    /// );
    /// # Ok::<(), ()>(())
    /// ```
    pub fn new(start: PageAddr<AS>, increment: PageSize) -> Option<Self> {
        let next = PageAddr::with_alignment(RawAddr::from(start), increment)?;
        Some(Self {
            next: Some(next),
            increment,
        })
    }
}

impl<AS: AddressSpace> Iterator for PageAddrIter<AS> {
    type Item = PageAddr<AS>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next;
        if let Some(n) = self.next {
            self.next = n.checked_add_pages_with_size(1, self.increment);
        }
        next
    }
}

/// The page number of a page.
#[derive(Copy, Clone)]
pub struct Pfn<AS: AddressSpace>(u64, AS);

impl<AS: AddressSpace> Pfn<AS> {
    /// Creates a new `Pfn` from the given raw u64.
    /// `bits` should be the frame number of a page in the given `address_space`.
    pub fn new(bits: u64, address_space: AS) -> Self {
        Pfn(bits, address_space)
    }

    /// Returns the raw bits of the page number.
    pub fn bits(&self) -> u64 {
        self.0
    }

    /// Returns the address space that this `Pfn` refers to.
    pub fn address_space(&self) -> AS {
        self.1
    }
}

impl Pfn<SupervisorPhys> {
    /// Creates a PFN from raw bits.
    pub fn supervisor(bits: u64) -> Self {
        Pfn(bits, SupervisorPhys)
    }
}

/// A page frame number in the supervisor's physical address space.
pub type SupervisorPfn = Pfn<SupervisorPhys>;
/// A page frame number in a guest's physical address space.
pub type GuestPfn = Pfn<GuestPhys>;

impl<AS: AddressSpace> From<PageAddr<AS>> for Pfn<AS> {
    fn from(page: PageAddr<AS>) -> Pfn<AS> {
        Pfn(page.addr.0 >> PFN_SHIFT, page.addr.1)
    }
}

/// Trait representing a page in the physical address space. The page may be backed by RAM, or
/// a device (for MMIO).
pub trait PhysPage: Sized {
    /// Creates a new 4kB `PhysPage` at the specified address.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the page referenced by `addr` is uniquely owned and not
    /// mapped into the address space of any VMs. Furthermore, the backing memory must be of the
    /// same type as this `PhysPage` is intended to represent (e.g. ordinary RAM for `Page`).
    unsafe fn new(addr: SupervisorPageAddr) -> Self {
        Self::new_with_size(addr, PageSize::Size4k)
    }

    /// Creates a new `PhysPage` with `size` page size at the specified address. Panics if `addr`
    /// is not aligned to `size`.
    ///
    /// # Safety
    ///
    /// See new().
    unsafe fn new_with_size(addr: SupervisorPageAddr, size: PageSize) -> Self;

    /// Returns the base address of this page.
    fn addr(&self) -> SupervisorPageAddr;

    /// Returns the type of memory this represents.
    fn mem_type() -> MemType;

    /// Returns the page size of this page.
    fn size(&self) -> PageSize;

    /// Returns the page frame number (PFN) of this page.
    fn pfn(&self) -> SupervisorPfn {
        self.addr().pfn()
    }
}

/// Trait representing a page that can be cleaned to transform from a dirty to clean state.
pub trait CleanablePhysPage: PhysPage {
    /// The destination page type of the clean() operation.
    type CleanPage: PhysPage;

    /// Consumes the page, cleaning it and returning it in a cleaned state.
    fn clean(self) -> Self::CleanPage;
}

/// Trait representing a page that can be initialized.
pub trait InitializablePhysPage: PhysPage {
    /// The type of page created by initializing this `InitializablePhysPage`.
    type InitializedPage: PhysPage;
    /// The type of page created if initializing this `InitializablePhysPage` fails and it is left
    /// in a dirty state.
    type DirtyPage: PhysPage;

    /// Consumes the page and attempts to initialize it by calling `func` with a mutable slice of
    /// this page's bytes. Returns the page in an intialized state upon success, or as a dirty
    /// page upon failure.
    fn try_initialize<F, E>(self, func: F) -> Result<Self::InitializedPage, (E, Self::DirtyPage)>
    where
        F: Fn(&mut [u8]) -> Result<(), E>;

    /// Converts the page to an initialized page. The caller is responsible for initializing the
    /// contents of the page to a known state.
    fn to_initialized_page(self) -> Self::InitializedPage;
}

/// Trait representing a page that can be mapped into a VM's address space.
pub trait MappablePhysPage<M: MeasureRequirement>: PhysPage {}

/// Trait representing a converted, but unassigned, page.
pub trait ConvertedPhysPage: PhysPage {
    /// The page type representing the initial state of a page just after conversion.
    type DirtyPage: ConvertedPhysPage;
}

/// Trait representing a converted page that can be assigned to a child VM. Pages transition from
/// converted to assignable by cleaning or initializing the converted page.
pub trait AssignablePhysPage<M: MeasureRequirement>: ConvertedPhysPage {
    /// The page type representing an assignable page that has been assigned as a mapped page
    /// in a child VM.
    type MappablePage: MappablePhysPage<M>;
}

/// Trait representing a page that has been invalidated in a VM. Invalidated pages are created
/// by unmapping a previously-mapped page in a VM's address space.
pub trait InvalidatedPhysPage: PhysPage {}

/// Trait representing a mapped page that's eligible for conversion to a Shared state
pub trait ShareablePhysPage: PhysPage {}

/// Trait representing a converted page that is eligible to be reclaimed by the owner. Pages
/// transition from converted to reclaimable once they have been cleaned.
pub trait ReclaimablePhysPage: ConvertedPhysPage {
    /// The page type representing a page that has been reclaimed as a mappable page in the
    /// owning VM. Pages must be clean to be mapped back into the owner.
    type MappablePage: MappablePhysPage<MeasureOptional>;
}

/// Base type representing a page of RAM.
///
/// `Page` is a key abstraction; it owns all memory of the backing page.
/// This guarantee allows the memory within pages to be assigned to virtual machines, and the taken
/// back to be uniquely owned by a `Page` here in the hypervisor.
pub struct Page<S: State> {
    addr: SupervisorPageAddr,
    size: PageSize,
    state: PhantomData<S>,
}

impl<S: State> Page<S> {
    /// Returns the u64 at the given index in the page.
    pub fn get_u64(&self, index: usize) -> Option<u64> {
        let offset = index * core::mem::size_of::<u64>();
        if offset >= self.size as usize {
            None
        } else {
            let address = self.addr.bits() + offset as u64;
            unsafe {
                // Safe because Page guarantees all contained memory is uniquely owned and
                // valid and the address must be in the owned page because of the above index range
                // check.
                Some(core::ptr::read_volatile(address as *const u64))
            }
        }
    }

    /// Returns an iterator across all u64 values contained in the page.
    pub fn u64_iter(&self) -> U64Iter<S> {
        U64Iter {
            page: self,
            index: 0,
        }
    }

    /// Returns the contents of the page as a slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        let base_ptr = self.addr.bits() as *const u8;
        // Safe because the borrow cannot outlive the lifetime
        // of the underlying page, and the upper bound is
        // guaranteed to be within the size of the page
        unsafe { slice::from_raw_parts(base_ptr, self.size as usize) }
    }
}

impl Page<InternalClean> {
    /// Test-only function that creates a page by allocating extra memory for alignement, then
    /// leaking all the memory. While technically "safe" it does leak all that memory so use with
    /// caution.
    #[cfg(test)]
    pub fn new_in_test() -> Self {
        let mem = vec![0u8; 8192];
        let ptr = mem.as_ptr();
        let aligned_ptr = unsafe {
            // Safe because it's only a test and the above allocation guarantees that the result is
            // still a valid pointer.
            ptr.add(ptr.align_offset(4096))
        };
        let page = Self {
            addr: PageAddr::new(RawAddr::supervisor(aligned_ptr as u64)).unwrap(),
            size: PageSize::Size4k,
            state: PhantomData,
        };
        std::mem::forget(mem);
        page
    }
}

impl<S: State> PhysPage for Page<S> {
    /// Creates a new `Page` representing a page of ordinary RAM at `addr`.
    ///
    /// # Safety
    ///
    /// See PhysPage::new(). `addr` must refer to a page of ordinary, idempotent system RAM.
    unsafe fn new_with_size(addr: SupervisorPageAddr, size: PageSize) -> Self {
        assert!(addr.is_aligned(size));
        Self {
            addr,
            size,
            state: PhantomData,
        }
    }

    fn addr(&self) -> SupervisorPageAddr {
        self.addr
    }

    fn size(&self) -> PageSize {
        self.size
    }

    fn mem_type() -> MemType {
        MemType::Ram
    }
}

impl<S: Cleanable> CleanablePhysPage for Page<S> {
    type CleanPage = Page<S::Cleaned>;

    fn clean(self) -> Self::CleanPage {
        unsafe {
            // Safe because page owns all the memory at its address.
            core::ptr::write_bytes(self.addr.bits() as *mut u8, 0, self.size as usize);
        }
        Page {
            addr: self.addr,
            size: self.size,
            state: PhantomData,
        }
    }
}

impl<S: Initializable> InitializablePhysPage for Page<S> {
    type InitializedPage = Page<ConvertedInitialized>;
    type DirtyPage = Page<ConvertedDirty>;

    fn try_initialize<F, E>(self, func: F) -> Result<Self::InitializedPage, (E, Self::DirtyPage)>
    where
        F: Fn(&mut [u8]) -> Result<(), E>,
    {
        let base_ptr = self.addr.bits() as *mut u8;
        // Safety: same as Page::as_bytes();
        let bytes = unsafe { slice::from_raw_parts_mut(base_ptr, self.size as usize) };
        func(bytes)
            .map_err(|e| {
                (
                    e,
                    Page {
                        addr: self.addr,
                        size: self.size,
                        state: PhantomData,
                    },
                )
            })
            .map(|_| Page {
                addr: self.addr,
                size: self.size,
                state: PhantomData,
            })
    }

    fn to_initialized_page(self) -> Self::InitializedPage {
        Page {
            addr: self.addr,
            size: self.size,
            state: PhantomData,
        }
    }
}

impl<S: Mappable<M>, M: MeasureRequirement> MappablePhysPage<M> for Page<S> {}

impl<S: Converted> ConvertedPhysPage for Page<S> {
    // Converted pages are considered dirty until they're cleaned or initialized.
    type DirtyPage = Page<ConvertedDirty>;
}

impl<S: Assignable<M>, M: MeasureRequirement> AssignablePhysPage<M> for Page<S> {
    type MappablePage = Page<S::Mappable>;
}

impl InvalidatedPhysPage for Page<Invalidated> {}

// Pages are only reclaimable if they're clean.
impl ReclaimablePhysPage for Page<ConvertedClean> {
    type MappablePage = Page<MappableClean>;
}

impl ShareablePhysPage for Page<Shareable> {}

/// An iterator of the 64-bit words contained in a page.
pub struct U64Iter<'a, S: State> {
    page: &'a Page<S>,
    index: usize,
}

impl<'a, S: State> Iterator for U64Iter<'a, S> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.page.get_u64(self.index);
        self.index += 1;
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unaligned_phys() {
        // check that an unaligned address fails to create a PageAddr.
        assert!(SupervisorPageAddr::new(RawAddr::supervisor(0x01)).is_none());
        assert!(SupervisorPageAddr::new(RawAddr::supervisor(0x1000)).is_some());
        assert!(
            SupervisorPageAddr::with_alignment(RawAddr::supervisor(0x1000), PageSize::Size2M)
                .is_none()
        );
        assert!(SupervisorPageAddr::with_alignment(
            RawAddr::supervisor(0x20_0000),
            PageSize::Size2M
        )
        .is_some());
        assert!(SupervisorPageAddr::with_alignment(
            RawAddr::supervisor(0x20_0000),
            PageSize::Size1G
        )
        .is_none());
        assert!(SupervisorPageAddr::with_alignment(
            RawAddr::supervisor(0x4000_0000),
            PageSize::Size1G
        )
        .is_some());
        assert!(SupervisorPageAddr::with_alignment(
            RawAddr::supervisor(0x4000_0000),
            PageSize::Size512G
        )
        .is_none());
        assert!(SupervisorPageAddr::with_alignment(
            RawAddr::supervisor(0x80_0000_0000),
            PageSize::Size512G
        )
        .is_some());
    }

    #[test]
    fn round_phys() {
        assert!(
            PageAddr::with_round_up(RawAddr::supervisor(0x12_2345), PageSize::Size4k).bits()
                == 0x12_3000
        );
        assert!(
            PageAddr::with_round_down(RawAddr::supervisor(0x4567_9521), PageSize::Size4k).bits()
                == 0x4567_9000
        );
    }

    #[test]
    fn page_iter_start() {
        let addr4k = PageAddr::new(RawAddr::supervisor(0)).unwrap();
        let mut addrs = addr4k.iter_from();
        assert_eq!(addrs.next(), Some(addr4k));
        assert_eq!(
            addrs.next(),
            Some(PageAddr::new(RawAddr::supervisor(4096)).unwrap())
        );
        assert_eq!(
            addrs.next(),
            Some(PageAddr::new(RawAddr::supervisor(8192)).unwrap())
        );
        assert_eq!(
            addrs.next(),
            Some(PageAddr::new(RawAddr::supervisor(12288)).unwrap())
        );

        let addr_m = PageAddr::with_alignment(RawAddr::supervisor(0), PageSize::Size2M).unwrap();
        let mut addrs = addr_m.iter_from_with_size(PageSize::Size2M).unwrap();
        assert_eq!(addrs.next(), Some(addr_m));
        assert_eq!(
            addrs.next(),
            Some(
                PageAddr::with_alignment(RawAddr::supervisor(1024 * 1024 * 2), PageSize::Size2M)
                    .unwrap()
            )
        );
    }

    #[test]
    fn page_iter_end_addr_space() {
        let addr4k = PageAddr::new(RawAddr::supervisor(0_u64.wrapping_sub(4096))).unwrap();
        let mut addrs = addr4k.iter_from();
        assert_eq!(addrs.next(), Some(addr4k));
        assert_eq!(addrs.next(), None);

        let addr_m = PageAddr::with_alignment(
            RawAddr::supervisor(0_u64.wrapping_sub(1024 * 1024 * 2)),
            PageSize::Size2M,
        )
        .unwrap();
        let mut addrs = addr_m.iter_from_with_size(PageSize::Size2M).unwrap();
        assert_eq!(addrs.next(), Some(addr_m));
        assert_eq!(addrs.next(), None);
    }

    #[test]
    fn u64_index_range_4k() {
        let p = Page::new_in_test();
        assert!(p.get_u64(0).is_some());
        assert!(p.get_u64(1).is_some());
        assert!(p.get_u64(511).is_some());
        assert!(p.get_u64(512).is_none());
    }

    #[test]
    fn u64_iter() {
        let p = Page::new_in_test();
        assert_eq!(p.u64_iter().count(), 512);

        unsafe {
            // Just a test!
            let ptr = p.addr().bits() as *mut u64;
            for i in 0..512 {
                *ptr.add(i) = i as u64;
            }
        }

        assert!(p.u64_iter().enumerate().all(|(i, v)| i as u64 == v));
    }

    #[test]
    pub fn test_as_bytes() {
        let mut mem = [0u8; 8192];
        let ptr = mem.as_mut_ptr();

        let aligned_ptr = unsafe {
            // Safe because the above allocation guarantees that the result is
            // still a valid pointer.
            ptr.add(ptr.align_offset(4096))
        };

        // Safe because the above allocation guarantees that the result is
        // still a valid pointer.
        unsafe {
            *aligned_ptr.add(4095) = 0xAA;
        };

        let addr = PageAddr::new(RawAddr::supervisor(aligned_ptr as u64)).unwrap();

        // Safe the above allocation guarantees that the result is
        // still a valid pointer.
        let page: Page<ConvertedDirty> = unsafe { Page::new(addr) };

        assert!(page.as_bytes().last().unwrap() == &0xAA);
    }
}
