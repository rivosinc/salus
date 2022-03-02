// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Represents pages of memory.
use core::marker::PhantomData;

// PFN constants, currently sv48x4 hard-coded
// TODO parameterize based on address mode
const PFN_SHIFT: u64 = 12;
const PFN_BITS: u64 = 44;
const PFN_MASK: u64 = (1 << PFN_BITS) - 1;

/// Implementors of `PageSize` are valid sizes for leaf pages.
pub trait PageSize {
    const SIZE_BYTES: u64;

    /// Checks if the given physical address is aligned to this page size.
    fn is_aligned(addr: &PhysAddr) -> bool {
        (addr.bits() & (Self::SIZE_BYTES - 1)) == 0
    }
}

/// Standard 4k pages - leaf nodes of L1 tables.
pub enum PageSize4k {}
impl PageSize for PageSize4k {
    const SIZE_BYTES: u64 = 4096;
}

/// 2MB pages - leaf nodes of L2 tables.
pub enum PageSize2MB {}
impl PageSize for PageSize2MB {
    const SIZE_BYTES: u64 = PageSize4k::SIZE_BYTES * 512;
}

/// 1GB pages - leaf nodes of L3 tables.
pub enum PageSize1GB {}
impl PageSize for PageSize1GB {
    const SIZE_BYTES: u64 = PageSize2MB::SIZE_BYTES * 512;
}

/// 512GB pages - leaf nodes of L4 tables.
pub enum PageSize512GB {}
impl PageSize for PageSize512GB {
    const SIZE_BYTES: u64 = PageSize1GB::SIZE_BYTES * 512;
}

/// A valid address to physical memory.
#[derive(Copy, Clone)]
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
}

impl<S: PageSize> From<PageAddr<S>> for PhysAddr {
    fn from(p: PageAddr<S>) -> PhysAddr {
        p.addr
    }
}

/// An address of a Page of physical memory. It is guaranteed to be aligned to a page boundary.
#[derive(Copy, Clone)]
pub struct PageAddr<S: PageSize> {
    addr: PhysAddr,
    phantom: PhantomData<S>,
}

/// Exports a more convenient way to write `PageAddr<PageSize4k>`
pub type PageAddr4k = PageAddr<PageSize4k>;

impl<S: PageSize> PageAddr<S> {
    /// Creates a `PageAddr` from a `PhysAddr`, returns `None` if the address isn't aligned to the
    /// required page size.
    pub fn new(addr: PhysAddr) -> Option<Self> {
        if S::is_aligned(&addr) {
            Some(PageAddr {
                addr,
                phantom: PhantomData,
            })
        } else {
            None
        }
    }

    /// Gets the raw bits of the page address.
    pub fn bits(&self) -> u64 {
        self.addr.0
    }

    /// Gets the pfn of the page address.
    pub fn pfn(&self) -> Pfn {
        Pfn::from_bits((self.addr.0 >> PFN_SHIFT) & PFN_MASK)
    }
}

impl<S: PageSize> TryFrom<Pfn> for PageAddr<S> {
    type Error = (); // TODO error type
    fn try_from(pfn: Pfn) -> core::result::Result<Self, Self::Error> {
        let phys_addr = PhysAddr(pfn.0 << PFN_SHIFT);
        Self::new(phys_addr).ok_or(())
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
        if let UnmappedPage::Page(addr) = self {
            addr
        } else {
            panic!("Tried to unwrap as 4k addr");
        }
    }

    /// Returns the wrapped 2MB Page if that's the type, otherwise, panic.
    pub fn unwrap_2mb(self) -> Page<PageSize2MB> {
        if let UnmappedPage::Mega(addr) = self {
            addr
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

impl<S: PageSize> From<PageAddr<S>> for Pfn {
    fn from(page: PageAddr<S>) -> Pfn {
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
    addr: PageAddr<S>,
}

/// A 4k page. Shorthand for `Page<PageSize4k>`
pub type Page4k = Page<PageSize4k>;

impl<S: PageSize> Page<S> {
    /// # Safety
    /// The caller must guarantee that memory from `addr` to `addr`+PageSize if uniquely owned.
    /// `new` is intended to be used _only_ when the backing memory region is unmapped from all
    /// virutal machines and can be uniquely owned by the resulting `Page`.
    pub unsafe fn new(addr: PageAddr<S>) -> Self {
        Self { addr }
    }

    /// Returns the address of this page.
    pub fn addr(&self) -> PageAddr<S> {
        PageAddr::<S>::new(self.addr.addr).unwrap()
    }
}

impl<S: PageSize> PhysPage for Page<S> {
    fn pfn(&self) -> Pfn {
        self.addr.pfn()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unaligned_phys() {
        // check that an unaligned address fails to create a PageAddr.
        assert!(PageAddr::<PageSize4k>::new(PhysAddr::new(0x01)).is_none());
        assert!(PageAddr::<PageSize4k>::new(PhysAddr::new(0x1000)).is_some());
        assert!(PageAddr::<PageSize2MB>::new(PhysAddr::new(0x1000)).is_none());
        assert!(PageAddr::<PageSize2MB>::new(PhysAddr::new(0x20_0000)).is_some());
        assert!(PageAddr::<PageSize1GB>::new(PhysAddr::new(0x20_0000)).is_none());
        assert!(PageAddr::<PageSize1GB>::new(PhysAddr::new(0x4000_0000)).is_some());
        assert!(PageAddr::<PageSize512GB>::new(PhysAddr::new(0x4000_0000)).is_none());
        assert!(PageAddr::<PageSize512GB>::new(PhysAddr::new(0x80_0000_0000)).is_some());
    }
}
