// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::borrow::Borrow;
use core::fmt::{self, Display};
use core::ops::Deref;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use riscv_pages::{InternalClean, InternalDirty, Page, PageAddr, PageSize, PhysPage, RawAddr};

use crate::collections::PageBox;
use crate::PageTracker;

#[repr(C)]
struct PageArcInner<T> {
    data: T,
    rc: AtomicUsize,
    page_size: PageSize,
    page_tracker: PageTracker,
}

/// A `PageArc` is a simplified `Arc`-like container, using a `Page` as the backing store for
/// a reference-counted piece of data. Unlike `Arc`, `PageArc` does not support weak references.
///
/// The backing `Page` is automatically released back to the previous owner using `PageTracker` when
/// the last reference to the `PageArc` is dropped. The user may also convert the `PageArc` into a
/// uniquely-owned `PageBox`.
#[allow(clippy::derive_partial_eq_without_eq)] // Silence buggy clippy warning.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageArc<T>(NonNull<PageArcInner<T>>);

impl<T> PageArc<T> {
    /// Creates a `PageArc` that wraps the given data using `page` as the backing store. The page is
    /// released back to the previous owner when the last reference is dropped using `page_tracker`.
    pub fn new_with(data: T, page: Page<InternalClean>, page_tracker: PageTracker) -> Self {
        let ptr = page.addr().bits() as *mut PageArcInner<T>;
        assert!(!ptr.is_null()); // Explicitly ban pages at zero address.
        let inner = PageArcInner {
            data,
            rc: AtomicUsize::new(1),
            page_size: page.size(),
            page_tracker,
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
        let page_tracker = this.inner().page_tracker.clone();
        // Safety: PageArcInner is repr(C) with data as the first field, therefore a pointer to data
        // is a page-aligned and properly initialized pointer to T.
        let boxed =
            unsafe { PageBox::from_raw(Self::as_ptr(&this) as *mut T, page_size, page_tracker) };
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

        let page_tracker = self.inner().page_tracker.clone();
        let page_size = self.inner().page_size;
        // Safe because we now have unique ownership of the page this PageArc was constructed with.
        let page: Page<InternalDirty> = unsafe {
            Page::new_with_size(
                PageAddr::new(RawAddr::supervisor(self.0.as_ptr() as u64)).unwrap(),
                page_size,
            )
        };

        // Unwrap ok: we have unique ownership of the page so we must be able to release it.
        page_tracker.release_page(page).unwrap();
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
    use crate::TlbVersion;
    use riscv_pages::*;

    fn stub_page() -> (PageTracker, Page<InternalClean>) {
        let (page_tracker, mut pages) = PageTracker::new_in_test();
        let assigned_page = pages
            .by_ref()
            .take(1)
            .map(|p| {
                page_tracker
                    .assign_page_for_internal_state(p, PageOwnerId::host())
                    .unwrap()
            })
            .next()
            .unwrap();
        (page_tracker, assigned_page)
    }

    #[test]
    fn lifecycle() {
        let (page_tracker, backing_page) = stub_page();
        let addr = backing_page.addr();
        {
            let arc = PageArc::new_with([5u8; 128], backing_page, page_tracker.clone());
            assert_eq!(PageArc::ref_count(&arc), 1);
            {
                let copy = arc.clone();
                assert_eq!(PageArc::ref_count(&copy), 2);
                assert_eq!(PageArc::ref_count(&arc), 2);
            }
            assert_eq!(PageArc::ref_count(&arc), 1);
        }

        // Should go back to hypervisor-owned after the PageArc was dropped.
        assert!(page_tracker
            .get_converted_page::<Page<ConvertedDirty>>(
                addr,
                PageOwnerId::hypervisor(),
                TlbVersion::new()
            )
            .is_ok());
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
        let (page_tracker, backing_page) = stub_page();
        let mut destroyed = false;
        {
            let arc = PageArc::new_with(
                ArcTest {
                    flag: &mut destroyed,
                },
                backing_page,
                page_tracker.clone(),
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
