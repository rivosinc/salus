// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::collections::{TryReserveError, TryReserveErrorKind};
use alloc::vec::Vec;
use core::mem::{self, ManuallyDrop};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::SliceIndex;
use riscv_pages::{InternalClean, InternalDirty, PageAddr, PageSize, RawAddr, SequentialPages};

use crate::PageTracker;

/// Similar to `Vec` but backed by an integer number of pre-allocated pages. Used to avoid having an
/// allocator but allow using a Vec for simple storage.
///
/// `RawPageVec` will leak its pages on drop if they aren't reclaimed with `to_pages`. `PageVec` will
/// release the pages back to their previous owner in `PageTracker` upon being dropped.
///
/// To avoid panics, `RawPageVec` and `PageVec` require the use of `try_reserve` before `push`.
/// Pushing is fallible as there is no allocator from which to request more memory. Pushing more
/// elements than the `Vec` has capacity for will result in a panic because there is no allocator to
/// handle it.
///
/// ## Example
///
/// ```rust
/// use core::result::Result;
/// use page_tracking::collections::RawPageVec;
/// use riscv_pages::{InternalClean, InternalDirty, Page, PageSize, SequentialPages};
///
/// fn sum_in_page<I>(
///     vals: I,
///     pages: SequentialPages<InternalClean>,
/// ) -> Result<(u64, SequentialPages<InternalDirty>), ()>
/// where
///     I: IntoIterator<Item = u64>,
/// {
///     let mut v = RawPageVec::from(pages);
///     assert_eq!(v.len(), 0);
///     let capacity = v.capacity();
///     assert!(v.try_reserve(capacity).is_ok());
///     for item in vals.into_iter().take(capacity) {
///         v.push(item);
///     }
///     let res = v.iter().sum();
///
///     v.clear();
///
///     Ok((res, v.to_pages()))
/// }
/// ```
#[derive(Debug)]
pub struct RawPageVec<T>(ManuallyDrop<Vec<T>>, PageSize);

impl<T> RawPageVec<T> {
    /// Destroys the elements held by this `RawPageVec` and returns its backing pages.    /// emptied.
    pub fn to_pages(mut self) -> SequentialPages<InternalDirty> {
        // Ensures destructors of any T's still owned are called.
        self.clear();
        // Safe since we're consuming this `RawPageVec`.
        unsafe { self.take_pages() }
    }

    /// See `std::vec::try_reserve`
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        if self.len() + additional > self.capacity() {
            Err(TryReserveErrorKind::CapacityOverflow.into())
        } else {
            Ok(())
        }
    }

    /// See `std::vec::push`
    pub fn push(&mut self, item: T) {
        self.0.push(item)
    }

    /// See `std::vec::retain`
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f)
    }

    /// See `std::vec::insert`
    pub fn insert(&mut self, index: usize, element: T) {
        self.0.insert(index, element);
    }

    /// See `std::vec::remove`
    pub fn remove(&mut self, index: usize) -> T {
        self.0.remove(index)
    }

    /// See `std::vec::pop`
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    /// See `std::vec::clear`
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// See `std::vec::capacity`
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// See `std::vec::get_mut`
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut <I as SliceIndex<[T]>>::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.0.get_mut(index)
    }

    /// Returns the `SequentialPages` backing this `RawPageVec`.
    ///
    /// # Safety
    ///
    /// This method semantically takes ownership of the backing storage of this container without
    /// preventing further usage. It is the caller's responsibility to ensure that this `RawPageVec`
    /// is never used again.
    unsafe fn take_pages(&mut self) -> SequentialPages<InternalDirty> {
        let page_size = self.1 as usize;
        let vec_page_count = (self.capacity() * mem::size_of::<T>() + page_size - 1) / page_size;
        // Unwrap ok, the backing pages must've been contiguous.
        SequentialPages::from_mem_range(
            PageAddr::new(RawAddr::supervisor(self.0.as_ptr() as u64)).unwrap(),
            self.1,
            vec_page_count as u64,
        )
        .unwrap()
    }
}

impl<T> From<SequentialPages<InternalClean>> for RawPageVec<T> {
    fn from(pages: SequentialPages<InternalClean>) -> Self {
        let capacity_bytes = pages.length_bytes() as usize;
        let capacity = capacity_bytes / mem::size_of::<T>();
        // Safe because the memory is page-aligned, fully owned, and contiguous(guaranteed by
        // `SequentialPages`).
        unsafe {
            RawPageVec(
                ManuallyDrop::new(Vec::from_raw_parts(
                    pages.base().bits() as *mut T,
                    0,
                    capacity,
                )),
                pages.page_size(),
            )
        }
    }
}

impl<T> Deref for RawPageVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.0.as_ptr(), self.0.len()) }
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for RawPageVec<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for RawPageVec<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

/// A `Page`-backed `Vec` whose backing pages are automatically released back to the pervious owner
/// when dropped.
pub struct PageVec<T>(RawPageVec<T>, PageTracker);

impl<T> PageVec<T> {
    /// Creates a new `PageVec` backed by `pages`.
    pub fn new(pages: SequentialPages<InternalClean>, page_tracker: PageTracker) -> Self {
        Self(RawPageVec::from(pages), page_tracker)
    }
}

impl<T> Deref for PageVec<T> {
    type Target = RawPageVec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for PageVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Drop for PageVec<T> {
    fn drop(&mut self) {
        self.0.clear();
        // Safe since we're in drop() and this `PageVec` can't be used again.
        let pages = unsafe { self.0.take_pages() };
        for p in pages.into_iter() {
            // Unwrap ok: we have unique ownership of the page so we must be able to release it.
            self.1.release_page(p).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TlbVersion;
    use alloc::vec;
    use riscv_pages::{ConvertedDirty, Page, PageOwnerId, PhysPage};

    #[test]
    fn raw_vec() {
        let mem = vec![0u8; PageSize::Size4k as usize * 2];
        let aligned_addr = PageSize::Size4k.round_up(mem.as_ptr() as u64);
        // This is not safe, but it's only for a test and mem isn't touched until backing_page is
        // dropped.
        let backing_page =
            unsafe { Page::new(PageAddr::new(RawAddr::supervisor(aligned_addr)).unwrap()) };

        let seq_pages = SequentialPages::from_pages([backing_page]).unwrap();
        let mut v = RawPageVec::from(seq_pages);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        let capacity = v.capacity();
        assert_eq!(
            capacity,
            PageSize::Size4k as usize / core::mem::size_of::<u64>()
        );
        assert!(v.try_reserve(10).is_ok());
        assert!(v.try_reserve(capacity).is_ok());
        for item in 0..10 {
            v.push(item as u64);
        }
        assert_eq!(v.len(), 10);
        assert!(!v.is_empty());
        assert_eq!(45u64, v.iter().sum());

        v.clear();
        assert_eq!(v.to_pages().into_iter().count(), 1);
    }

    #[test]
    fn with_page_drop() {
        let (page_tracker, mut pages) = PageTracker::new_in_test();
        let assigned_pages = SequentialPages::from_pages(pages.by_ref().take(2).map(|p| {
            page_tracker
                .assign_page_for_internal_state(p, PageOwnerId::host())
                .unwrap()
        }))
        .unwrap();
        let first_page_addr = assigned_pages.base();
        {
            let mut v = PageVec::new(assigned_pages, page_tracker.clone());
            for item in 0..10 {
                v.push(item as u64);
            }
        }
        // Should go back to hypervisor-owned after the PageVec was dropped.
        assert!(page_tracker
            .get_converted_page::<Page<ConvertedDirty>>(
                first_page_addr,
                PageOwnerId::hypervisor(),
                TlbVersion::new()
            )
            .is_ok());
    }
}
