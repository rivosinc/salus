// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt::{self, Display};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::{borrow, cmp};
use riscv_pages::{
    InternalClean, InternalDirty, Page, PageAddr, PageSize, PhysPage, RawAddr, SequentialPages,
};

use crate::PageTracker;

/// A PageBox is a Box-like container that holds a backing page filled with the given type.
/// Because Salus borrows pages from the host for data it is necessary to track the backing pages
/// with the data they are used to store. The backing page is returned to the previous owner
/// using `PageTracker` when the `PageBox` is dropped.
///
/// # Creation
///
/// 1. Page given to Salus by the host for tracking a data type `T` (for example TEE state).
/// 2. Remove page from the host: unmap_page -> `Page`
/// 3. Pass the page to the PageBox: `let data = PageBox::new(GuestInfo::new(), page, page_tracker);`
/// 4. Use type
/// 5. When the `PageBox` is dropped, ownership of the `Page` is returned to the donor.
///
/// # Example
///
/// ```rust
/// use page_tracking::collections::PageBox;
/// use page_tracking::PageTracker;
/// use riscv_pages::{InternalClean, InternalDirty, Page};
///
/// struct TestData {
///     a: u64,
///     b: u64,
/// }
///
/// fn add_in_box(
///     a: u64,
///     b: u64,
///     backing_page: Page<InternalClean>,
///     page_tracker: PageTracker,
/// ) -> u64 {
///     let boxxed_data = PageBox::new_with(TestData { a, b }, backing_page.into(), page_tracker);
///     let sum = boxxed_data.a + boxxed_data.b;
///     sum
/// }
/// ```
pub struct PageBox<T> {
    ptr: NonNull<T>,
    page_size: PageSize,
    page_count: u64,
    page_tracker: PageTracker,
}

impl<T> PageBox<T> {
    /// Try to create a `PageBox` that wraps the given data using `pages` to store it, returning it to its
    /// previous owner on `drop()` using `page_tracker`. Returns `None` if not enough pages are supplied.
    pub fn try_new(
        data: T,
        pages: SequentialPages<InternalClean>,
        page_tracker: PageTracker,
    ) -> Option<Self> {
        if core::mem::size_of::<T>() <= pages.length_bytes() as usize {
            Some(Self::new_with(data, pages, page_tracker))
        } else {
            None
        }
    }

    /// Create a `PageBox` that wraps the given data using `pages` to store it, returning it to its
    /// previous owner on `drop()` using `page_tracker`. Caller must ensure enough pages are supplied.
    pub fn new_with(
        data: T,
        pages: SequentialPages<InternalClean>,
        page_tracker: PageTracker,
    ) -> Self {
        let ptr = pages.base().bits() as *mut T;
        assert!(!ptr.is_null()); // Explicitly ban pages at zero address.
        unsafe {
            // Safe as the memory is totally owned and T fits in this page.
            assert!(core::mem::size_of::<T>() <= pages.length_bytes() as usize);
            core::ptr::write(ptr, data);
        }
        Self {
            ptr: NonNull::new(ptr).unwrap(),
            page_size: pages.page_size(),
            page_count: pages.len(),
            page_tracker,
        }
    }

    /// Consumes the `PageBox` and returns the pages that were used to hold the data.
    pub fn to_pages(mut self) -> SequentialPages<InternalDirty> {
        let pages = unsafe {
            // Safe because self must have ownership of the page it was built with and the contained
            // data is aligned and owned so can be dropped.
            core::ptr::drop_in_place(self.ptr.as_ptr());
            // Safe because we're consuming this PageBox.
            self.take_pages()
        };
        core::mem::forget(self);
        pages
    }

    /// Returns the contained data and the page that was used to hold it.
    pub fn into_inner(mut self) -> (T, SequentialPages<InternalDirty>) {
        let (data, pages) = unsafe {
            // Safe because self must have ownership of the pages it was built with and the contained
            // data is aligned and owned so can be read with ptr::read.
            let data = core::ptr::read(self.ptr.as_ptr());
            // Safe because we've extracted the contained data and we're consuming this PageBox.
            let pages = self.take_pages();
            (data, pages)
        };
        core::mem::forget(self);
        (data, pages)
    }

    /// Leaks the backing page for the box and returns a static reference to the contained data.
    /// Useful for objects that live for the rest of the program.
    pub fn leak(b: Self) -> &'static mut T {
        // Safe because self owns all this memory and by using 'ManuallyDrop', it will never be
        // freed for reuse.
        unsafe { &mut *core::mem::ManuallyDrop::new(b).ptr.as_ptr() }
    }

    /// Creates a `PageBox` from a raw, page-aligned pointer to `T`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr`:
    ///  - points to a properly initialized `T`
    ///  - points to a uniquely-owned contiguous `page_count` pages of size `page_size`
    ///  - is aligned to `page_size`
    ///  - the backing pages can contain `T`
    pub unsafe fn from_raw(
        ptr: *mut T,
        page_size: PageSize,
        page_count: u64,
        page_tracker: PageTracker,
    ) -> PageBox<T> {
        assert!(page_size.is_aligned(ptr as u64));
        assert!(core::mem::size_of::<T>() as u64 <= page_size as u64 * page_count);
        Self {
            ptr: NonNull::new(ptr).unwrap(),
            page_size,
            page_count,
            page_tracker,
        }
    }

    /// Returns the `SequentialPages` backing this `PageBox`.
    ///
    /// # Safety
    ///
    /// This method semantically takes ownership of the backing storage of this container without
    /// preventing further usage. It is the caller's responsibility to ensure that this `PageBox`
    /// is never used again.
    unsafe fn take_pages(&mut self) -> SequentialPages<InternalDirty> {
        // Unwrap ok, the backing pages must've been contiguous.
        SequentialPages::from_mem_range(
            PageAddr::new(RawAddr::supervisor(self.ptr.as_ptr() as u64)).unwrap(),
            self.page_size,
            self.page_count,
        )
        .unwrap()
    }
}

impl<T> PartialEq for PageBox<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T> PartialOrd for PageBox<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.ptr.partial_cmp(&other.ptr)
    }
}

impl<T> Drop for PageBox<T> {
    fn drop(&mut self) {
        // Safe because self must have ownership of the page it was built with and the contained
        // data is aligned and owned so can be dropped.
        unsafe {
            core::ptr::drop_in_place(self.ptr.as_ptr());
        }

        // Safe because we're in drop() and this PageBox can no longer be used.
        let pages = unsafe { self.take_pages() };
        for p in pages.into_iter() {
            // Unwrap ok: we have unique ownership of the page so we must be able to release it.
            self.page_tracker.release_page(p).unwrap();
        }
    }
}

impl<T> borrow::Borrow<T> for PageBox<T> {
    fn borrow(&self) -> &T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> borrow::BorrowMut<T> for PageBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> AsRef<T> for PageBox<T> {
    fn as_ref(&self) -> &T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> AsMut<T> for PageBox<T> {
    fn as_mut(&mut self) -> &mut T {
        // Safe because the pointer is page-aligned, totally owned, and only created with a valid
        // `T` instance.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Deref for PageBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> DerefMut for PageBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: Display> Display for PageBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.ptr.as_ref().fmt(f) }
    }
}

// Safety: Like Box<T>, PageBox<T> is Send/Sync iff T is Send/Sync.
unsafe impl<T> Send for PageBox<T> where T: Send {}
unsafe impl<T> Sync for PageBox<T> where T: Sync {}

/// An object with static lifetime using a `Page` as storage. Equivalent to creating a `PageBox<T>`
/// and then immediately `leak()`ing it. Can be used for creating static data-structures at startup.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StaticPageRef<T>(NonNull<T>);

impl<T> StaticPageRef<T> {
    /// Creates a new `StaticPageRef` holding `data` and using `page` as the backing storage.
    pub fn new_with(data: T, page: Page<InternalClean>) -> Self {
        let ptr = page.addr().bits() as *mut T;
        assert!(!ptr.is_null()); // Explicitly ban pages at zero address.
        unsafe {
            // Safe as the memory is totally owned and T fits in this page.
            assert!(core::mem::size_of::<T>() <= page.size() as usize);
            core::ptr::write(ptr, data);
        }
        Self(NonNull::new(ptr).unwrap())
    }
}

impl<T> Clone for StaticPageRef<T> {
    fn clone(&self) -> Self {
        StaticPageRef(self.0)
    }
}

impl<T> Deref for StaticPageRef<T> {
    type Target = T;

    fn deref(&self) -> &'static T {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.0.as_ref() }
    }
}

// Safety: Like Box<T>, StaticPageRef<T> is Send/Sync iff T is Send/Sync.
unsafe impl<T> Send for StaticPageRef<T> where T: Sync + Send {}
unsafe impl<T> Sync for StaticPageRef<T> where T: Sync + Send {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TlbVersion;
    use riscv_pages::*;

    #[test]
    fn to_inner() {
        const TEST_PAGE_COUNT: usize = 3;
        const TEST_BYTES: usize = TEST_PAGE_COUNT * PageSize::Size4k as usize;
        let (page_tracker, mut pages) = PageTracker::new_in_test();
        let assigned_pages =
            SequentialPages::from_pages(pages.by_ref().take(TEST_PAGE_COUNT).map(|p| {
                page_tracker
                    .assign_page_for_internal_state(p, PageOwnerId::host())
                    .unwrap()
            }))
            .unwrap();

        let pb = PageBox::new_with([5u8; TEST_BYTES], assigned_pages, page_tracker.clone());
        let (vals, dirty_pages) = pb.into_inner();
        assert_eq!(vals, [5u8; TEST_BYTES]);

        assert_eq!(dirty_pages.len(), TEST_PAGE_COUNT as u64);
        for p in dirty_pages.into_iter() {
            assert!(page_tracker.release_page(p).is_ok());
        }
    }

    #[test]
    fn page_drop() {
        const TEST_PAGE_COUNT: usize = 3;
        const TEST_BYTES: usize = TEST_PAGE_COUNT * PageSize::Size4k as usize;
        let (page_tracker, mut pages) = PageTracker::new_in_test();
        let assigned_pages =
            SequentialPages::from_pages(pages.by_ref().take(TEST_PAGE_COUNT).map(|p| {
                page_tracker
                    .assign_page_for_internal_state(p, PageOwnerId::host())
                    .unwrap()
            }))
            .unwrap();

        let mut addr = assigned_pages.base();
        let page_size = assigned_pages.page_size();
        let page_count = assigned_pages.len();
        assert_eq!(page_count, TEST_PAGE_COUNT as u64);

        {
            let _pb = PageBox::new_with([5u8; TEST_BYTES], assigned_pages, page_tracker.clone());
        }

        for _pi in 0..TEST_PAGE_COUNT {
            // Should go back to hypervisor-owned after the PageBox was dropped.
            assert!(page_tracker
                .get_converted_page::<Page<ConvertedDirty>>(
                    addr,
                    PageOwnerId::hypervisor(),
                    TlbVersion::new()
                )
                .is_ok());
            addr = addr.checked_add_pages_with_size(1, page_size).unwrap();
        }
    }

    #[test]
    #[should_panic]
    fn new_with_not_enough_backing_pages_should_panic() {
        const TEST_PAGE_COUNT: usize = 3;
        const TEST_BYTES: usize = 1 + TEST_PAGE_COUNT * PageSize::Size4k as usize /* 1 byte bigger than backing pages */;
        let (page_tracker, mut pages) = PageTracker::new_in_test();
        let assigned_pages =
            SequentialPages::from_pages(pages.by_ref().take(TEST_PAGE_COUNT).map(|p| {
                page_tracker
                    .assign_page_for_internal_state(p, PageOwnerId::host())
                    .unwrap()
            }))
            .unwrap();

        let _pb = PageBox::new_with([5u8; TEST_BYTES], assigned_pages, page_tracker.clone());
    }

    #[test]
    fn try_new_not_enough_backing_pages_should_fail() {
        const TEST_PAGE_COUNT: usize = 3;
        const TEST_BYTES: usize = 1 + TEST_PAGE_COUNT * PageSize::Size4k as usize /* 1 byte bigger than backing pages */;
        let (page_tracker, mut pages) = PageTracker::new_in_test();
        let assigned_pages =
            SequentialPages::from_pages(pages.by_ref().take(TEST_PAGE_COUNT).map(|p| {
                page_tracker
                    .assign_page_for_internal_state(p, PageOwnerId::host())
                    .unwrap()
            }))
            .unwrap();

        let pb = PageBox::try_new([5u8; TEST_BYTES], assigned_pages, page_tracker.clone());
        assert!(pb.is_none());
    }
}
