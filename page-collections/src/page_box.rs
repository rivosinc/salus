// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::borrow;
use core::fmt::{self, Display};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use riscv_pages::{InternalClean, InternalDirty, Page, PageAddr, PageSize, PhysPage, RawAddr};

/// A PageBox is a Box-like container that holds a backing page filled with the given type.
/// Because Salus borrows pages from the host for data it is necessary to track the backing pages
/// with the data they are used to store.
///
/// Note that `drop` leaks the page unless the page has been taken with `take_backing_page`. It is
/// the user's responsibility to take the page out and return it to the donor.
///
/// # Creation
///
/// 1. Page given to Salus by the host for tracking a data type `T` (for example TEE state).
/// 2. Remove page from the host: unmap_page -> `Page`
/// 3. Pass the page to the PageBox: `let data = PageBox::new(GuestInfo::new(), page);`
/// 4. Use type
/// 5. When done call `take_backing_page`, which will call the destructor of `T` and return the
///    `Page`
/// 6. Return the page to the host that donated it.
///
/// # Example
///
/// ```rust
/// use page_collections::page_box::PageBox;
/// use riscv_pages::{Page, InternalClean, InternalDirty};
///
/// struct TestData {
///     a: u64,
///     b: u64,
/// }
///
/// fn add_in_box(a:u64, b:u64, backing_page: Page<InternalClean>) -> (u64, Page<InternalDirty>) {
///     let boxxed_data = PageBox::new_with(
///         TestData {a, b},
///         backing_page,
///     );
///     let sum = boxxed_data.a + boxxed_data.b;
///     (sum, boxxed_data.to_page())
/// }
/// ```
#[allow(clippy::derive_partial_eq_without_eq)] // Silence buggy clippy warning.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageBox<T>(NonNull<T>, PageSize);

impl<T> PageBox<T> {
    /// Creates a `PageBox` that wraps the given data using `page` to store it.
    /// To avoid leaking the page, the returned `PageBox` must have its page reclaimed with
    /// `to_page` instead of being dropped.
    pub fn new_with(data: T, page: Page<InternalClean>) -> Self {
        let ptr = page.addr().bits() as *mut T;
        assert!(!ptr.is_null()); // Explicitly ban pages at zero address.
        unsafe {
            // Safe as the memory is totally owned and T fits in this page.
            assert!(core::mem::size_of::<T>() <= page.size() as usize);
            core::ptr::write(ptr, data);
        }
        Self(NonNull::new(ptr).unwrap(), page.size())
    }

    /// Consumes the `PageBox` and returns the page that was used to hold the data.
    pub fn to_page(self) -> Page<InternalDirty> {
        // Safe because self must have ownership of the page it was built with and the contained
        // data is aligned and owned so can be dropped.
        unsafe {
            core::ptr::drop_in_place(self.0.as_ptr());
            Page::new_with_size(
                PageAddr::new(RawAddr::supervisor(self.0.as_ptr() as u64)).unwrap(),
                self.1,
            )
        }
    }

    /// Returns the contained data and the page that was used to hold it.
    pub fn into_inner(self) -> (T, Page<InternalDirty>) {
        unsafe {
            // Safe because self must have ownership of the page it was built with and the contained
            // data is aligned and owned so can be read with ptr::read.
            let page = Page::new_with_size(
                PageAddr::new(RawAddr::supervisor(self.0.as_ptr() as u64)).unwrap(),
                self.1,
            );
            let data = core::ptr::read(self.0.as_ptr());
            core::mem::forget(self);
            (data, page)
        }
    }

    /// Leaks the backing page for the box and returns a static reference to the contained data.
    /// Useful for objects that live for the rest of the program.
    pub fn leak(b: Self) -> &'static mut T {
        // Safe because self owns all this memory and by using 'ManuallyDrop', it will never be
        // freed for reuse.
        unsafe { &mut *core::mem::ManuallyDrop::new(b).0.as_ptr() }
    }

    /// Creates a `PageBox` from a raw, page-aligned pointer to `T`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr`:
    ///  - points to a properly initialized `T`
    ///  - points to a uniquely-owned page of size `page_size`
    ///  - is aligned to `page_size`
    pub unsafe fn from_raw(ptr: *mut T, page_size: PageSize) -> PageBox<T> {
        assert!(page_size.is_aligned(ptr as u64));
        Self(NonNull::new(ptr).unwrap(), page_size)
    }
}

impl<T> Drop for PageBox<T> {
    /// Drop can't return the backing memory so it is leaked, however it will run `T`'s destructor.
    fn drop(&mut self) {
        // Safe because self must have ownership of the page it was built with and the contained
        // data is aligned and owned so can be dropped.
        unsafe {
            core::ptr::drop_in_place(self.0.as_ptr());
        }
    }
}

impl<T> borrow::Borrow<T> for PageBox<T> {
    fn borrow(&self) -> &T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.0.as_ref() }
    }
}

impl<T> borrow::BorrowMut<T> for PageBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.0.as_mut() }
    }
}

impl<T> AsRef<T> for PageBox<T> {
    fn as_ref(&self) -> &T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.0.as_ref() }
    }
}

impl<T> AsMut<T> for PageBox<T> {
    fn as_mut(&mut self) -> &mut T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.0.as_mut() }
    }
}

impl<T> Deref for PageBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for PageBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.0.as_mut() }
    }
}

impl<T: Display> Display for PageBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.0.as_ref().fmt(f) }
    }
}

// Safety: Like Box<T>, PageBox<T> is Send/Sync iff T is Send/Sync.
unsafe impl<T> Send for PageBox<T> where T: Send {}
unsafe impl<T> Sync for PageBox<T> where T: Sync {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn to_inner() {
        let mem = vec![0u8; PageSize::Size4k as usize * 2];
        let aligned_addr = PageSize::Size4k.round_up(mem.as_ptr() as u64);
        // This is not safe, but it's only for a test and mem isn't touched until backing_page is
        // dropped.
        let backing_page =
            unsafe { Page::new(PageAddr::new(RawAddr::supervisor(aligned_addr)).unwrap()) };

        let pb = PageBox::new_with([5u8; 128], backing_page);

        let (vals, _p) = pb.into_inner();
        assert_eq!(vals, [5u8; 128]);
    }
}
