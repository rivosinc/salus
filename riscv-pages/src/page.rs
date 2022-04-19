// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Represents pages of memory.
use core::marker::PhantomData;
use core::slice;

// PFN constants, currently sv48x4 hard-coded
// TODO parameterize based on address mode
const PFN_SHIFT: u64 = 12;
const PFN_BITS: u64 = 44;
const PFN_MASK: u64 = (1 << PFN_BITS) - 1;

/// Implementors of `PageSize` are valid sizes for leaf pages.
pub trait PageSize: Copy + Clone + PartialEq + core::fmt::Debug {
    const SIZE_BYTES: u64;

    /// Checks if the given physical address is aligned to this page size.
    fn is_aligned(addr: &PhysAddr) -> bool {
        (addr.bits() & (Self::SIZE_BYTES - 1)) == 0
    }
}

/// Standard 4k pages - leaf nodes of L1 tables.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PageSize4k {}
impl PageSize for PageSize4k {
    const SIZE_BYTES: u64 = 4096;
}

/// 2MB pages - leaf nodes of L2 tables.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PageSize2MB {}
impl PageSize for PageSize2MB {
    const SIZE_BYTES: u64 = PageSize4k::SIZE_BYTES * 512;
}

/// 1GB pages - leaf nodes of L3 tables.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PageSize1GB {}
impl PageSize for PageSize1GB {
    const SIZE_BYTES: u64 = PageSize2MB::SIZE_BYTES * 512;
}

/// 512GB pages - leaf nodes of L4 tables.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PageSize512GB {}
impl PageSize for PageSize512GB {
    const SIZE_BYTES: u64 = PageSize1GB::SIZE_BYTES * 512;
}

/// A valid address to physical memory.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PhysAddr(u64);

impl PhysAddr {
    /// Creates a `PhysAddr` from a given u64 raw address.
    pub fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Returns the inner 64 address.
    pub fn bits(&self) -> u64 {
        self.0
    }

    /// Returns the address incremented by the given number of bytes.
    /// Returns None if the result would overlfow.
    pub fn checked_increment(&self, increment: u64) -> Option<Self> {
        self.0.checked_add(increment).map(PhysAddr)
    }
}

impl<S: PageSize> From<AlignedPageAddr<S>> for PhysAddr {
    fn from(p: AlignedPageAddr<S>) -> PhysAddr {
        p.addr
    }
}

/// An address of a Page of physical memory. It is guaranteed to be aligned to a page boundary.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AlignedPageAddr<S: PageSize> {
    addr: PhysAddr,
    phantom: PhantomData<S>,
}

/// Exports a more convenient way to write `AlignedPageAddr<PageSize4k>`
pub type AlignedPageAddr4k = AlignedPageAddr<PageSize4k>;

impl<S: PageSize> AlignedPageAddr<S> {
    /// Creates a `AlignedPageAddr` from a `PhysAddr`, returns `None` if the address isn't aligned to the
    /// required page size.
    pub fn new(addr: PhysAddr) -> Option<Self> {
        if S::is_aligned(&addr) {
            Some(AlignedPageAddr {
                addr,
                phantom: PhantomData,
            })
        } else {
            None
        }
    }

    /// Creates a `AlignedPageAddr` from a `PhysAddr`, rounding up to the nearest multiple of the page
    /// size.
    pub fn with_round_up(addr: PhysAddr) -> Self {
        Self {
            addr: PhysAddr::new((addr.bits() + S::SIZE_BYTES - 1) & !(S::SIZE_BYTES - 1)),
            phantom: PhantomData,
        }
    }

    /// Same as above, but rounding down to the nearest multiple of the page size.
    pub fn with_round_down(addr: PhysAddr) -> Self {
        Self {
            addr: PhysAddr::new(addr.bits() & !(S::SIZE_BYTES - 1)),
            phantom: PhantomData,
        }
    }

    /// Returns the first 4k page address. This is OK because all page sizes must be a multiple of
    /// 4k.
    pub fn get_4k_addr(&self) -> AlignedPageAddr<PageSize4k> {
        // Unwrap is OK as all pages are 4k aligned.
        AlignedPageAddr::new(self.addr).unwrap()
    }

    /// Gets the raw bits of the page address.
    pub fn bits(&self) -> u64 {
        self.addr.0
    }

    /// Moves to the next page address.
    pub fn iter_from(&self) -> AlignedPageAddrIter<S> {
        AlignedPageAddrIter::new(*self)
    }

    /// Gets the pfn of the page address.
    pub fn pfn(&self) -> Pfn {
        Pfn::from_bits((self.addr.0 >> PFN_SHIFT) & PFN_MASK)
    }

    /// Adds n pages to the current address.
    pub fn checked_add_pages(&self, n: u64) -> Option<Self> {
        n.checked_mul(S::SIZE_BYTES)
            .and_then(|inc| self.addr.checked_increment(inc))
            .and_then(Self::new)
    }

    /// Gets the index of the page in the system (the linear page count from address 0).
    pub fn index(&self) -> usize {
        self.pfn().bits() as usize
    }
}

impl<S: PageSize> TryFrom<Pfn> for AlignedPageAddr<S> {
    type Error = (); // TODO error type
    fn try_from(pfn: Pfn) -> core::result::Result<Self, Self::Error> {
        let phys_addr = PhysAddr(pfn.0 << PFN_SHIFT);
        Self::new(phys_addr).ok_or(())
    }
}

/// Generate page addresses for the given size. 4096, 8192, 12288, ... until the end of u64's range
pub struct AlignedPageAddrIter<S: PageSize> {
    next: Option<u64>,
    phantom: PhantomData<S>,
}

impl<S: PageSize> AlignedPageAddrIter<S> {
    pub fn new(start: AlignedPageAddr<S>) -> Self {
        Self {
            next: Some(start.addr.0),
            phantom: PhantomData,
        }
    }
}

impl<S: PageSize> Iterator for AlignedPageAddrIter<S> {
    type Item = AlignedPageAddr<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next;
        if let Some(n) = self.next {
            self.next = n.checked_add(S::SIZE_BYTES);
        }
        next.and_then(|a| AlignedPageAddr::new(PhysAddr::new(a)))
    }
}

/// A page that was unmapped from the guest.
pub enum UnmappedPage {
    Page(Page<PageSize4k>),
    Mega(Page<PageSize2MB>),
    Giga(Page<PageSize1GB>),
    Tera(Page<PageSize512GB>),
}

impl UnmappedPage {
    /// Returns the wrapped 4k Page if that's the type, otherwise, panic.
    pub fn unwrap_4k(self) -> Page<PageSize4k> {
        if let UnmappedPage::Page(p) = self {
            p
        } else {
            panic!("Tried to unwrap as 4k addr");
        }
    }

    /// Returns either Ok(4kpage) or the provided error
    pub fn ok4k_or<E>(self, err: E) -> core::result::Result<Page<PageSize4k>, E> {
        if let UnmappedPage::Page(p) = self {
            Ok(p)
        } else {
            Err(err)
        }
    }

    /// Returns the wrapped 2MB Page if that's the type, otherwise, panic.
    pub fn unwrap_2mb(self) -> Page<PageSize2MB> {
        if let UnmappedPage::Mega(p) = self {
            p
        } else {
            panic!("Tried to unwrap as 2MB addr");
        }
    }
}

/// The page number of a page.
#[derive(Copy, Clone)]
pub struct Pfn(u64);

impl Pfn {
    /// Creates a PFN from raw bits.
    pub fn from_bits(bits: u64) -> Pfn {
        Pfn(bits)
    }

    /// Returns the raw bits of the page number.
    pub fn bits(&self) -> u64 {
        self.0
    }
}

impl<S: PageSize> From<AlignedPageAddr<S>> for Pfn {
    fn from(page: AlignedPageAddr<S>) -> Pfn {
        Pfn(page.addr.0 >> PFN_SHIFT)
    }
}

/// Trait implemented by physical pages of any size.
pub trait PhysPage {
    fn pfn(&self) -> Pfn;
}

/// Base type representing a page, generic for different sizes.
/// `Page` is a key abstraction; it owns all memory of the backing page.
/// This guarantee allows the memory within pages to be assigned to virtual machines, and the taken
/// back to be uniquely owned by a `Page` here in the hypervisor.
pub struct Page<S: PageSize> {
    addr: AlignedPageAddr<S>,
}

/// A 4k page. Shorthand for `Page<PageSize4k>`
pub type Page4k = Page<PageSize4k>;

impl<S: PageSize> Page<S> {
    /// # Safety
    /// The caller must guarantee that memory from `addr` to `addr`+PageSize if uniquely owned.
    /// `new` is intended to be used _only_ when the backing memory region is unmapped from all
    /// virutal machines and can be uniquely owned by the resulting `Page`.
    pub unsafe fn new(addr: AlignedPageAddr<S>) -> Self {
        Self { addr }
    }

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
            addr: AlignedPageAddr::new(PhysAddr::new(aligned_ptr as u64)).unwrap(),
        };
        std::mem::forget(mem);
        page
    }

    /// Returns the starting address of this page.
    pub fn addr(&self) -> AlignedPageAddr<S> {
        AlignedPageAddr::<S>::new(self.addr.addr).unwrap()
    }

    /// Returns the u64 at the given index in the page.
    pub fn get_u64(&self, index: usize) -> Option<u64> {
        let offset = index * core::mem::size_of::<u64>();
        if offset >= S::SIZE_BYTES as usize {
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

    pub fn as_bytes(&self) -> &[u8] {
        let base_ptr = self.addr.bits() as *const u8;
        // Safe because the borrow cannot outlive the lifetime
        // of the underlying page, and the upper bound is
        // guaranteed to be within the size of the page
        unsafe { slice::from_raw_parts(base_ptr, S::SIZE_BYTES as usize) }
    }
}

impl<S: PageSize> PhysPage for Page<S> {
    fn pfn(&self) -> Pfn {
        self.addr.pfn()
    }
}

/// A page that was unmapped from the guest and cleared of all guest data.
pub struct CleanPage(UnmappedPage);

impl From<CleanPage> for UnmappedPage {
    fn from(cp: CleanPage) -> UnmappedPage {
        cp.0
    }
}

impl From<UnmappedPage> for CleanPage {
    fn from(p: UnmappedPage) -> CleanPage {
        let (addr, size) = match p {
            UnmappedPage::Page(ref p) => (p.addr.bits(), PageSize4k::SIZE_BYTES),
            UnmappedPage::Mega(ref p) => (p.addr.bits(), PageSize2MB::SIZE_BYTES),
            UnmappedPage::Giga(ref p) => (p.addr.bits(), PageSize1GB::SIZE_BYTES),
            UnmappedPage::Tera(ref p) => (p.addr.bits(), PageSize512GB::SIZE_BYTES),
        };
        unsafe {
            // Safe because page owns all the memory at its address.
            core::ptr::write_bytes(addr as *mut u8, 0, size as usize);
        }
        CleanPage(p)
    }
}

pub struct U64Iter<'a, S: PageSize> {
    page: &'a Page<S>,
    index: usize,
}

impl<'a, S: PageSize> Iterator for U64Iter<'a, S> {
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
        // check that an unaligned address fails to create a AlignedPageAddr.
        assert!(AlignedPageAddr::<PageSize4k>::new(PhysAddr::new(0x01)).is_none());
        assert!(AlignedPageAddr::<PageSize4k>::new(PhysAddr::new(0x1000)).is_some());
        assert!(AlignedPageAddr::<PageSize2MB>::new(PhysAddr::new(0x1000)).is_none());
        assert!(AlignedPageAddr::<PageSize2MB>::new(PhysAddr::new(0x20_0000)).is_some());
        assert!(AlignedPageAddr::<PageSize1GB>::new(PhysAddr::new(0x20_0000)).is_none());
        assert!(AlignedPageAddr::<PageSize1GB>::new(PhysAddr::new(0x4000_0000)).is_some());
        assert!(AlignedPageAddr::<PageSize512GB>::new(PhysAddr::new(0x4000_0000)).is_none());
        assert!(AlignedPageAddr::<PageSize512GB>::new(PhysAddr::new(0x80_0000_0000)).is_some());
    }

    #[test]
    fn round_phys() {
        assert!(
            AlignedPageAddr::<PageSize4k>::with_round_up(PhysAddr::new(0x12_2345)).bits()
                == 0x12_3000
        );
        assert!(
            AlignedPageAddr::<PageSize4k>::with_round_down(PhysAddr::new(0x4567_9521)).bits()
                == 0x4567_9000
        );
    }

    #[test]
    fn page_iter_start() {
        let addr4k: AlignedPageAddr<PageSize4k> = AlignedPageAddr::new(PhysAddr::new(0)).unwrap();
        let mut addrs = addr4k.iter_from();
        assert_eq!(addrs.next(), Some(addr4k));
        assert_eq!(
            addrs.next(),
            Some(AlignedPageAddr::new(PhysAddr::new(4096)).unwrap())
        );
        assert_eq!(
            addrs.next(),
            Some(AlignedPageAddr::new(PhysAddr::new(8192)).unwrap())
        );
        assert_eq!(
            addrs.next(),
            Some(AlignedPageAddr::new(PhysAddr::new(12288)).unwrap())
        );

        let addr_m: AlignedPageAddr<PageSize2MB> = AlignedPageAddr::new(PhysAddr::new(0)).unwrap();
        let mut addrs = addr_m.iter_from();
        assert_eq!(addrs.next(), Some(addr_m));
        assert_eq!(
            addrs.next(),
            Some(AlignedPageAddr::new(PhysAddr::new(1024 * 1024 * 2)).unwrap())
        );
    }

    #[test]
    fn page_iter_end_addr_space() {
        let addr4k: AlignedPageAddr<PageSize4k> =
            AlignedPageAddr::new(PhysAddr::new(0_u64.wrapping_sub(4096))).unwrap();
        let mut addrs = addr4k.iter_from();
        assert_eq!(addrs.next(), Some(addr4k));
        assert_eq!(addrs.next(), None);

        let addr_m: AlignedPageAddr<PageSize2MB> =
            AlignedPageAddr::new(PhysAddr::new(0_u64.wrapping_sub(1024 * 1024 * 2))).unwrap();
        let mut addrs = addr_m.iter_from();
        assert_eq!(addrs.next(), Some(addr_m));
        assert_eq!(addrs.next(), None);
    }

    #[test]
    fn u64_index_range_4k() {
        let p: Page<PageSize4k> = Page::new_in_test();
        assert!(p.get_u64(0).is_some());
        assert!(p.get_u64(1).is_some());
        assert!(p.get_u64(511).is_some());
        assert!(p.get_u64(512).is_none());
    }

    #[test]
    fn u64_iter() {
        let p: Page<PageSize4k> = Page::new_in_test();
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

        let addr: AlignedPageAddr<PageSize4k> =
            AlignedPageAddr::new(PhysAddr::new(aligned_ptr as u64)).unwrap();

        // Safe the above allocation guarantees that the result is
        // still a valid pointer.
        let page = unsafe { Page::new(addr) };

        assert!(page.as_bytes().last().unwrap() == &0xAA);
    }
}
