// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::borrow::Borrow;
use core::fmt::{self, Display};
use core::ops::Deref;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use riscv_pages::{InternalClean, Page, PageSize, PhysPage};

use crate::page_box::PageBox;

#[repr(C)]
struct PageArcInner<T> {
    data: T,
    rc: AtomicUsize,
    page_size: PageSize,
}

/// A `PageArc` is a simplified `Arc`-like container, using a `Page` as the backing store for
/// a reference-counted piece of data. Unlike `Arc`, `PageArc` does not support weak references
/// and the backing `Page` is *not* automatically freed when the last reference to the `PageArc`
/// is dropped. Instead, the user must first convert the `PageArc` into a uniquely-owned `PageBox`
/// and then reclaim the page with `PageBox::to_page()` or `PageBox::into_inner()`.
#[allow(clippy::derive_partial_eq_without_eq)] // Silence buggy clippy warning.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageArc<T>(NonNull<PageArcInner<T>>);

impl<T> PageArc<T> {
    /// Creates a `PageArc` that wraps the given data using `page` as the backing store.
    /// To avoid leaking the page, the returned `PageArc` must have its page reclaimed by first
    /// turning it into a uniquely-owned `PageBox` with `try_unwrap()` and then consuming the
    /// `PageBox` with `PageBox::to_page()`.
    pub fn new_with(data: T, page: Page<InternalClean>) -> Self {
        let ptr = page.addr().bits() as *mut PageArcInner<T>;
        assert!(!ptr.is_null()); // Explicitly ban pages at zero address.
        let inner = PageArcInner {
            data,
            rc: AtomicUsize::new(1),
            page_size: page.size(),
        };
        assert!(core::mem::size_of::<PageArcInner<T>>() <= page.size() as usize);
        unsafe {
            // Safe as the memory is totally owned and PageArcInner<T> fits in the page.
            core::ptr::write(ptr, inner);
        }
        Self(NonNull::new(ptr).unwrap())
    }

    /// Attempts to turn a `PageArc` into a uniquely-owned `PageBox` if `this` is the sole owner of
    /// the data pointed to by the `PageArc`.
    pub fn try_unwrap(this: PageArc<T>) -> Result<PageBox<T>, PageArc<T>> {
        if this
            .inner()
            .rc
            .compare_exchange(1, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return Err(this);
        }

        // Acquire here to synchronize with the release in drop().
        this.inner().rc.load(Ordering::Acquire);

        let page_size = this.inner().page_size;
        // Safety: PageArcInner is repr(C) with data as the first field, therefore a pointer to data
        // is a page-aligned and properly initialized pointer to T.
        let boxed = unsafe { PageBox::from_raw(Self::as_ptr(&this) as *mut T, page_size) };
        core::mem::forget(this);
        Ok(boxed)
    }

    /// Returns the reference count of this `PageArc`. While this function itself is safe, care
    /// must be taken to avoid TOCTTOU races when using the result of this function.
    pub fn ref_count(this: &PageArc<T>) -> usize {
        this.inner().rc.load(Ordering::SeqCst)
    }

    /// Returns a pointer to the data this `PageArc` contains.
    pub fn as_ptr(this: &PageArc<T>) -> *const T {
        &this.inner().data as *const T
    }

    fn inner(&self) -> &PageArcInner<T> {
        // Safe since we were initialized to point to a valid PageArcInner in the constructor and
        // if we still hold a reference the structure we point to must still be alive.
        unsafe { self.0.as_ref() }
    }
}

impl<T> Clone for PageArc<T> {
    fn clone(&self) -> PageArc<T> {
        // Relaxed ordering is okay here since we must've held a reference to begin with, preventing
        // another thread for observing that this PageArc has only a refcount of 1 and being able to
        // delete it.
        let old_rc = self.inner().rc.fetch_add(1, Ordering::Relaxed);
        assert!(old_rc < usize::MAX);
        PageArc(self.0)
    }
}

impl<T> Drop for PageArc<T> {
    fn drop(&mut self) {
        if self.inner().rc.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }

        // While the above release guarantees that no loads or stores from *this thread* can be
        // re-ordered after the release, we must acquire here in order to synchronize with the
        // release from other threads so that their loads and stores are visible to us.
        self.inner().rc.load(Ordering::Acquire);

        // Safe because we must have unique ownership of self at this point, and therefore the page
        // it was built with and the (aligned & properly initialized) T it contains.
        unsafe {
            let ptr = Self::as_ptr(self) as *mut T;
            core::ptr::drop_in_place(ptr);
        }
    }
}

impl<T> Borrow<T> for PageArc<T> {
    fn borrow(&self) -> &T {
        // Safe because the pointer is page-aligned, we have a live reference, and only created with
        // a valid `T` instance.
        unsafe { &self.0.as_ref().data }
    }
}

impl<T> AsRef<T> for PageArc<T> {
    fn as_ref(&self) -> &T {
        // Safe because the pointer is page-aligned, we have a live reference, and only created with
        // a valid `T` instance.
        unsafe { &self.0.as_ref().data }
    }
}

impl<T> Deref for PageArc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Safe because the pointer is page-aligned, we have a live reference, and only created with
        // a valid `T` instance.
        unsafe { &self.0.as_ref().data }
    }
}

impl<T: Display> Display for PageArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safe because this pointer is guaranteed to be valid in the constructor.
        unsafe { self.0.as_ref().data.fmt(f) }
    }
}

// Safety: Like Arc<T>, PageArc<T> is Send/Sync iff T is Send/Sync.
unsafe impl<T> Send for PageArc<T> where T: Sync + Send {}
unsafe impl<T> Sync for PageArc<T> where T: Sync + Send {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use riscv_pages::{PageAddr, RawAddr};

    fn stub_page() -> Page<InternalClean> {
        let mem = vec![0u8; PageSize::Size4k as usize * 2];
        let aligned_addr = PageSize::Size4k.round_up(mem.as_ptr() as u64);
        // This is not safe, but it's only for a test and mem isn't touched until backing_page is
        // dropped.
        let backing_page =
            unsafe { Page::new(PageAddr::new(RawAddr::supervisor(aligned_addr)).unwrap()) };
        core::mem::forget(mem);
        backing_page
    }

    #[test]
    fn lifecycle() {
        let arc = PageArc::new_with([5u8; 128], stub_page());
        assert_eq!(PageArc::ref_count(&arc), 1);
        {
            let copy = arc.clone();
            assert_eq!(PageArc::ref_count(&copy), 2);
            assert_eq!(PageArc::ref_count(&arc), 2);
        }
        assert_eq!(PageArc::ref_count(&arc), 1);

        let pb = PageArc::try_unwrap(arc).unwrap();
        let (vals, _) = pb.into_inner();
        assert_eq!(vals, [5u8; 128]);
    }

    #[derive(Debug)]
    struct ArcTest<'a> {
        flag: &'a mut bool,
    }

    impl<'a> Drop for ArcTest<'a> {
        fn drop(&mut self) {
            *self.flag = true;
        }
    }

    #[test]
    fn destructor() {
        let mut destroyed = false;
        {
            let arc = PageArc::new_with(
                ArcTest {
                    flag: &mut destroyed,
                },
                stub_page(),
            );
            assert_eq!(PageArc::ref_count(&arc), 1);
            let pb = PageArc::try_unwrap(arc).unwrap();
            let (t, _) = pb.into_inner();

            // The transitions from PageArc<T> -> PageBox<T> -> T shouldn't cause T to get dropped.
            assert!(!*t.flag);
        }
        assert!(destroyed);
    }
}
