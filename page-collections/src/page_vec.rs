// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::collections::{TryReserveError, TryReserveErrorKind};
use alloc::vec::Vec;
use core::mem::{self, ManuallyDrop};
use core::ops::{Index, IndexMut};
use core::slice::SliceIndex;

use riscv_pages::{PageAddr, PageSize, RawAddr, SequentialPages};

/// Similar to Vec but backed by an integer number of pre-allocated pages.
/// Used to avoid having an allocator but allow using a Vec for simple storage.
/// `PageVec` will leak its pages on drop if they aren't reclaimed with `to_pages`.
/// To avoid panics, `PageVec` requires the use of `try_reserve` before `push`.  Pushing is fallible
/// as there is no allocator from which to request more memory. Pushing more elements than the Vec
/// has capacity for will result in a panic because there is no allocator to handle it.
///
///
/// ## Example
///
/// ```rust
/// use page_collections::page_vec::PageVec;
/// use riscv_pages::{SequentialPages, Page, PageSize};
/// use core::result::Result;
///
/// fn sum_in_page<I>(vals: I, pages: SequentialPages)
///     -> Result<(u64, SequentialPages), ()>
/// where
///     I: IntoIterator<Item = u64>,
/// {
///     let mut v = PageVec::from(pages);
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
pub struct PageVec<T>(ManuallyDrop<Vec<T>>, PageSize);

impl<T> PageVec<T> {
    /// Destroys the given Vec and returns its backing pages.
    /// `vec` must be empty.
    /// If the Vec is backed by one page, `SinglePage` holds that page.
    /// If multiple pages back the Vec, they are returned in a new Vec contained in `MultiPage`.
    /// In the MultiPage case, the resultant vec should itself be passed to `to_pages` after being
    /// emptied.
    pub fn to_pages(mut self) -> SequentialPages {
        let page_size = self.1 as usize;
        self.clear(); // Ensures destructors of any T's still owned are called.
        let vec_page_count = (self.capacity() * mem::size_of::<T>() + page_size - 1) / page_size;
        unsafe {
            // Safe to create `SequentialPages` from the contained vec as the constructor of PageVec
            // guarantees the owned pages originated from `SequentialPages` so must be valid.
            SequentialPages::from_mem_range(
                PageAddr::new(RawAddr::supervisor(self.0.as_ptr() as u64)).unwrap(),
                self.1,
                vec_page_count as u64,
            )
            .unwrap()
        }
    }

    // Wrapper implementations from `Vec`
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        if self.len() + additional > self.capacity() {
            Err(TryReserveErrorKind::CapacityOverflow.into())
        } else {
            Ok(())
        }
    }

    pub fn push(&mut self, item: T) {
        self.0.push(item)
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f)
    }

    pub fn remove(&mut self, index: usize) -> T {
        self.0.remove(index)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    pub fn clear(&mut self) {
        self.0.clear()
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut <I as SliceIndex<[T]>>::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.0.get_mut(index)
    }
}

impl<T> From<SequentialPages> for PageVec<T> {
    fn from(pages: SequentialPages) -> Self {
        let capacity_bytes = pages.length_bytes() as usize;
        let capacity = capacity_bytes / mem::size_of::<T>();
        // Safe because the memory is page-aligned, fully owned, and contiguous(guaranteed by
        // `SequentialPages`).
        unsafe {
            PageVec(
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

impl<T> core::ops::Deref for PageVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.0.as_ptr(), self.0.len()) }
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for PageVec<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for PageVec<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use riscv_pages::{Page, PhysPage};

    #[test]
    fn basic() {
        let mem = vec![0u8; PageSize::Size4k as usize * 2];
        let aligned_addr = PageSize::Size4k.round_up(mem.as_ptr() as u64);
        // This is not safe, but it's only for a test and mem isn't touched until backing_page is
        // dropped.
        let backing_page =
            unsafe { Page::new(PageAddr::new(RawAddr::supervisor(aligned_addr)).unwrap()) };

        let seq_pages = SequentialPages::from_pages([backing_page]).unwrap();
        let mut v = PageVec::from(seq_pages);
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
}
