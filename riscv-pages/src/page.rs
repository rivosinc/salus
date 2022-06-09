// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Represents pages of memory.
use core::slice;

use crate::{AddressSpace, GuestPhys, MemType, PageOwnerId, SupervisorPhys, SupervisorVirt};

// PFN constants, currently sv48x4 hard-coded
// TODO parameterize based on address mode
const PFN_SHIFT: u64 = 12;
const PFN_BITS: u64 = 44;
const PFN_MASK: u64 = (1 << PFN_BITS) - 1;

#[repr(u64)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PageSize {
    Size4k = 4 * 1024,
    Size2M = 2 * 1024 * 1024,
    Size1G = 1024 * 1024 * 1024,
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
#[derive(Copy, Clone, Debug)]
pub struct RawAddr<AS: AddressSpace>(u64, AS);

impl<AS: AddressSpace> RawAddr<AS> {
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
    pub fn supervisor(addr: u64) -> Self {
        Self(addr, SupervisorPhys)
    }
}

impl RawAddr<SupervisorVirt> {
    pub fn supervisor_virt(addr: u64) -> Self {
        Self(addr, SupervisorVirt)
    }
}

impl RawAddr<GuestPhys> {
    pub fn guest(addr: u64, id: PageOwnerId) -> Self {
        Self(addr, GuestPhys::new(id))
    }
}

/// Convenience type aliases for supervisor-physical and guest-physical addresses.
pub type SupervisorPhysAddr = RawAddr<SupervisorPhys>;
pub type GuestPhysAddr = RawAddr<GuestPhys>;

impl<AS: AddressSpace> From<PageAddr<AS>> for RawAddr<AS> {
    fn from(p: PageAddr<AS>) -> RawAddr<AS> {
        p.addr
    }
}

impl<AS: AddressSpace> PartialEq for RawAddr<AS> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<AS: AddressSpace> PartialOrd for RawAddr<AS> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

/// An address of a Page in an address space. It is guaranteed to be aligned to a page boundary.
#[derive(Copy, Clone, Debug)]
pub struct PageAddr<AS: AddressSpace> {
    addr: RawAddr<AS>,
}

pub type SupervisorPageAddr = PageAddr<SupervisorPhys>;
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

impl<AS: AddressSpace> PartialEq for PageAddr<AS> {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
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

/// A page that was unmapped from the guest.
pub struct UnmappedPhysPage<P: PhysPage>(P);

impl<P: PhysPage> UnmappedPhysPage<P> {
    /// Creates a new `UnmappedPhysPage` wrapping the given page.
    pub fn new(page: P) -> Self {
        Self(page)
    }

    /// Returns the wrapped `PhysPage`. Note that the page retains any contents it had when it was
    /// mapped.
    pub fn to_page(self) -> P {
        self.0
    }
}

pub type UnmappedPage = UnmappedPhysPage<Page>;

/// The page number of a page.
#[derive(Copy, Clone)]
pub struct Pfn<AS: AddressSpace>(u64, AS);

impl<AS: AddressSpace> Pfn<AS> {
    pub fn new(bits: u64, address_space: AS) -> Self {
        Pfn(bits, address_space)
    }

    /// Returns the raw bits of the page number.
    pub fn bits(&self) -> u64 {
        self.0
    }

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

pub type SupervisorPfn = Pfn<SupervisorPhys>;
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

/// Base type representing a page of RAM.
///
/// `Page` is a key abstraction; it owns all memory of the backing page.
/// This guarantee allows the memory within pages to be assigned to virtual machines, and the taken
/// back to be uniquely owned by a `Page` here in the hypervisor.
pub struct Page {
    addr: SupervisorPageAddr,
    size: PageSize,
}

impl Page {
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
        };
        std::mem::forget(mem);
        page
    }

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
    pub fn u64_iter(&self) -> U64Iter {
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

impl PhysPage for Page {
    /// Creates a new `Page` representing a page of ordinary RAM at `addr`.
    ///
    /// # Safety
    ///
    /// See PhysPage::new(). `addr` must refer to a page of ordinary, idempotent system RAM.
    unsafe fn new_with_size(addr: SupervisorPageAddr, size: PageSize) -> Self {
        assert!(addr.is_aligned(size));
        Self { addr, size }
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

/// A page that was unmapped from the guest and cleared of all guest data.
pub struct CleanPage(Page);

impl From<UnmappedPage> for CleanPage {
    fn from(p: UnmappedPage) -> CleanPage {
        unsafe {
            // Safe because page owns all the memory at its address.
            core::ptr::write_bytes(p.0.addr.bits() as *mut u8, 0, p.0.size() as usize);
        }
        CleanPage(p.0)
    }
}

impl From<CleanPage> for Page {
    fn from(p: CleanPage) -> Page {
        p.0
    }
}

pub struct U64Iter<'a> {
    page: &'a Page,
    index: usize,
}

impl<'a> Iterator for U64Iter<'a> {
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
        let page = unsafe { Page::new(addr) };

        assert!(page.as_bytes().last().unwrap() == &0xAA);
    }
}
