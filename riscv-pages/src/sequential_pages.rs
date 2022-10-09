// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::cmp::min;
use core::fmt;
use core::marker::PhantomData;
use core::num::NonZeroU64;

use crate::page::{CleanablePhysPage, Page, PageSize, PhysPage, SupervisorPageAddr};
use crate::state::*;

/// An error resulting from trying to convert an iterator of pages to `SequentialPages`.
pub enum Error<S: State, I: Iterator<Item = Page<S>>> {
    /// There were zero pages in the list of pages given to create `Self`.
    Empty,
    /// Pages in the initialization page list were not contiguous.
    NonContiguous(I),
    /// Pages in the initialization page list were of different page sizes (4k and 2M for example).
    NonUniformSize(I),
    /// The sequence of pages would overflow the address space.
    Overflow(I),
}

/// An error resulting from trying to create a `SequentialPages` from a range of pages that are not
/// aligned to the requested size.
#[derive(Clone, Copy, Debug)]
pub struct UnalignedPages;

/// `SequentialPages` holds a range of consecutive pages of the same size and state. Each page's
/// address is one page after the previous. This forms a contiguous area of memory suitable for
/// holding an array or other linear data.
#[derive(Debug)]
pub struct SequentialPages<S: State> {
    addr: SupervisorPageAddr,
    page_size: PageSize,
    count: u64,
    state: PhantomData<S>,
}

impl<S: State> SequentialPages<S> {
    /// Creates a `SequentialPages` form the passed iterator.
    ///
    /// If the passed pages are not consecutive, an Error will be returned holding an iterator to
    /// the passed in pages so they don't leak.
    pub fn from_pages<T>(pages: T) -> Result<Self, Error<S, impl Iterator<Item = Page<S>>>>
    where
        T: IntoIterator<Item = Page<S>>,
    {
        let mut page_iter = pages.into_iter();

        let first_page = page_iter.next().ok_or(Error::Empty)?;

        let addr = first_page.addr();
        let page_size = first_page.size();

        let mut last_addr = addr;
        let mut seq = Self {
            addr,
            page_size,
            count: 1,
            state: PhantomData,
        };
        while let Some(page) = page_iter.next() {
            if page.size() != page_size {
                return Err(Error::NonUniformSize(
                    seq.into_iter()
                        .chain(core::iter::once(page))
                        .chain(page_iter),
                ));
            }
            let next_addr = match last_addr.checked_add_pages_with_size(1, page_size) {
                Some(a) => a,
                None => {
                    return Err(Error::Overflow(
                        seq.into_iter()
                            .chain(core::iter::once(page))
                            .chain(page_iter),
                    ))
                }
            };
            if page.addr() != next_addr {
                return Err(Error::NonContiguous(
                    seq.into_iter()
                        .chain(core::iter::once(page))
                        .chain(page_iter),
                ));
            }
            last_addr = page.addr();
            seq.count += 1;
        }

        Ok(seq)
    }

    /// Returns the address of the first page in the sequence(the start of the contiguous memory
    /// region).
    pub fn base(&self) -> SupervisorPageAddr {
        self.addr
    }

    /// Returns the number of sequential pages contained in this structure.
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Returns true if there are no pages in self.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the length of the contiguous memory region formed by the owned pages.
    pub fn length_bytes(&self) -> u64 {
        // Guaranteed not to overflow by the constructor.
        self.page_size as u64 * self.count
    }

    /// Returns the size of the backing pages.
    pub fn page_size(&self) -> PageSize {
        self.page_size
    }

    /// Returns `SequentialPages` for the memory range provided.
    /// # Safety
    /// The range's ownership is given to `SequentialPages`, the caller must uniquely own that
    /// memory.
    pub unsafe fn from_mem_range(
        addr: SupervisorPageAddr,
        page_size: PageSize,
        count: u64,
    ) -> Result<Self, UnalignedPages> {
        if !addr.is_aligned(page_size) {
            return Err(UnalignedPages);
        }
        Ok(Self {
            addr,
            page_size,
            count,
            state: PhantomData,
        })
    }

    /// Returns `SequentialPages` for the page range [start, end) with size `page_size`. Both
    /// start and end must be aligned to the requested size.
    ///
    /// # Safety
    /// The range's ownership is given to `SequentialPages`, the caller must uniquely own that
    /// memory.
    pub unsafe fn from_page_range(
        start: SupervisorPageAddr,
        end: SupervisorPageAddr,
        page_size: PageSize,
    ) -> Result<Self, UnalignedPages> {
        if !start.is_aligned(page_size) || !end.is_aligned(page_size) {
            return Err(UnalignedPages);
        }
        Ok(Self {
            addr: start,
            page_size,
            count: end.bits().checked_sub(start.bits()).unwrap() / page_size as u64,
            state: PhantomData,
        })
    }

    /// Returns an iterator across the addresses of all pages in `self`.
    pub fn page_addrs(&self) -> impl Iterator<Item = SupervisorPageAddr> {
        self.addr.iter_from().take(self.count as usize)
    }

    /// Return an iterator whose elements are `SequentialPages` of `chunk_size` pages. The last
    /// entry might contain fewer pages. This consumes `self`.
    pub fn into_chunks_iter(self, chunk_size: NonZeroU64) -> SeqPageChunkIter<S> {
        let count = self.count;
        let addr = if count > 0 { Some(self.addr) } else { None };
        let page_size = self.page_size;

        SeqPageChunkIter {
            chunk_size,
            addr,
            page_size,
            count,
            state: PhantomData,
        }
    }
}

impl<S: Cleanable> SequentialPages<S> {
    /// Consumes this range of pages, returning them in a cleaned state.
    pub fn clean(self) -> SequentialPages<S::Cleaned> {
        SequentialPages::from_pages(self.into_iter().map(|p| p.clean())).unwrap()
    }
}

impl<S: State> From<Page<S>> for SequentialPages<S> {
    fn from(p: Page<S>) -> Self {
        Self {
            addr: p.addr(),
            page_size: p.size(),
            count: 1,
            state: PhantomData,
        }
    }
}

/// An iterator of the individual pages previously used to build a `SequentialPages` struct.
/// Used to reclaim the pages from `SequentialPages`, returned from `SequentialPages::into_iter`.
pub struct SeqPageIter<S: State> {
    pages: SequentialPages<S>,
}

impl<S: State> Iterator for SeqPageIter<S> {
    type Item = Page<S>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pages.count == 0 {
            return None;
        }
        let addr = self.pages.addr;
        self.pages.addr = self
            .pages
            .addr
            .checked_add_pages_with_size(1, self.pages.page_size)
            .unwrap();
        self.pages.count -= 1;
        // Safe because `pages` owns the memory, which can be converted to pages because it is owned
        // and aligned.
        unsafe { Some(Page::new_with_size(addr, self.pages.page_size)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.pages.count as usize;
        (count, Some(count))
    }
}

impl<S: State> ExactSizeIterator for SeqPageIter<S> {}

impl<S: State> IntoIterator for SequentialPages<S> {
    type Item = Page<S>;
    type IntoIter = SeqPageIter<S>;
    fn into_iter(self) -> Self::IntoIter {
        SeqPageIter { pages: self }
    }
}

impl<S: State, I: Iterator<Item = Page<S>>> fmt::Debug for Error<S, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Error::Empty => f.debug_struct("Empty").finish(),
            Error::NonContiguous(_) => f.debug_struct("NonContiguous").finish_non_exhaustive(),
            Error::NonUniformSize(_) => f.debug_struct("NonUniform").finish_non_exhaustive(),
            Error::Overflow(_) => f.debug_struct("Overflow").finish_non_exhaustive(),
        }
    }
}

pub struct SeqPageChunkIter<S: State> {
    chunk_size: NonZeroU64,
    addr: Option<SupervisorPageAddr>,
    page_size: PageSize,
    count: u64,
    state: PhantomData<S>,
}

impl<S: State> Iterator for SeqPageChunkIter<S> {
    type Item = SequentialPages<S>;

    fn next(&mut self) -> Option<SequentialPages<S>> {
        if self.count == 0 {
            return None;
        }

        // Unwrap safe: address is none only on `count == 0`.
        let addr = self.addr.unwrap();
        let to_remove = min(self.count, self.chunk_size.get());
        self.count -= to_remove;
        self.addr = if self.count > 0 {
            Some(
                addr.checked_add_pages_with_size(to_remove, self.page_size)
                    .unwrap(),
            )
        } else {
            None
        };

        // Safe because `self` owns the memory, and we're creating a sequence from pages that
        // have been removed.
        unsafe { Some(SequentialPages::from_mem_range(addr, self.page_size, to_remove).unwrap()) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let chunk_size = self.chunk_size.get();
        let count = ((self.count + chunk_size - 1) / chunk_size) as usize;
        (count, Some(count))
    }
}

impl<S: State> ExactSizeIterator for SeqPageChunkIter<S> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PageAddr, RawAddr};

    #[test]
    fn create_success() {
        let pages: [Page<ConvertedDirty>; 4] = unsafe {
            // Not safe, but memory won't be touched in the test...
            [
                Page::new(PageAddr::new(RawAddr::supervisor(0x1000)).unwrap()),
                Page::new(PageAddr::new(RawAddr::supervisor(0x2000)).unwrap()),
                Page::new(PageAddr::new(RawAddr::supervisor(0x3000)).unwrap()),
                Page::new(PageAddr::new(RawAddr::supervisor(0x4000)).unwrap()),
            ]
        };

        assert!(SequentialPages::from_pages(pages).is_ok());
    }

    #[test]
    fn create_failure() {
        let pages: [Page<ConvertedDirty>; 4] = unsafe {
            // Not safe, but memory won't be touched in the test...
            [
                Page::new(PageAddr::new(RawAddr::supervisor(0x1000)).unwrap()),
                Page::new(PageAddr::new(RawAddr::supervisor(0x2000)).unwrap()),
                Page::new(PageAddr::new(RawAddr::supervisor(0x4000)).unwrap()),
                Page::new(PageAddr::new(RawAddr::supervisor(0x5000)).unwrap()),
            ]
        };
        let result = SequentialPages::from_pages(pages);
        match result {
            Ok(_) => panic!("didn't fail with non-sequential pages"),
            Err(Error::NonContiguous(returned_pages)) => {
                assert_eq!(returned_pages.count(), 4);
            }
            Err(_) => {
                panic!("failed with unexpected error");
            }
        }
    }

    #[test]
    fn create_fail_empty() {
        let pages: [Page<ConvertedDirty>; 0] = [];
        let result = SequentialPages::from_pages(pages);
        match result {
            Ok(_) => panic!("didn't fail with empty pages"),
            Err(Error::Empty) => (),
            Err(_) => panic!("failed with unexpected error"),
        }
    }

    #[test]
    fn from_single() {
        let p: Page<ConvertedDirty> = unsafe {
            // Not safe, Just a test.
            Page::new(PageAddr::new(RawAddr::supervisor(0x1000)).unwrap())
        };
        let seq = SequentialPages::from(p);
        let mut pages = seq.into_iter();
        assert_eq!(0x1000, pages.next().unwrap().addr().bits());
        assert!(pages.next().is_none());
    }

    #[test]
    fn unsafe_range() {
        // Not safe, but this is a test
        let seq: SequentialPages<ConvertedDirty> = unsafe {
            SequentialPages::from_mem_range(
                PageAddr::new(RawAddr::supervisor(0x1000)).unwrap(),
                PageSize::Size4k,
                4,
            )
            .unwrap()
        };
        let mut pages = seq.into_iter();
        assert_eq!(0x1000, pages.next().unwrap().addr().bits());
        assert_eq!(0x2000, pages.next().unwrap().addr().bits());
        assert_eq!(0x3000, pages.next().unwrap().addr().bits());
        assert_eq!(0x4000, pages.next().unwrap().addr().bits());
        assert!(pages.next().is_none());
    }

    fn create_test_sequential_pages(
        base: SupervisorPageAddr,
        len: u64,
    ) -> SequentialPages<ConvertedDirty> {
        // NOT SAFE, but this is a test
        unsafe { SequentialPages::from_mem_range(base, PageSize::Size4k, len).unwrap() }
    }

    #[test]
    fn chunks_iterator_exact_multiple() {
        // Test chunk iterator when the chunk size is a multiple of the length.
        let base_addr = PageAddr::new(RawAddr::supervisor(0x1000)).unwrap();
        let seq_len: u64 = 16;
        let chunk_size = NonZeroU64::new(4).unwrap();

        // Create a SequentialPages of length 16
        let seq = create_test_sequential_pages(base_addr, seq_len);
        let page_size = seq.page_size();

        // ... and a chunk iterator of 4 pages.
        let mut iter = seq.into_chunks_iter(chunk_size);
        assert_eq!(iter.size_hint(), (4, Some(4)));
        let mut addr = base_addr;
        // First four call will contain exactly `chunk_size` sequential pages.
        for _i in 0..4 {
            let subseq = iter.next().unwrap();
            assert_eq!(subseq.base(), addr);
            assert_eq!(subseq.len(), chunk_size.get());
            assert_eq!(subseq.page_size(), page_size);
            addr = addr
                .checked_add_pages_with_size(chunk_size.get(), page_size)
                .unwrap();
        }
        // Fifth call should be None.
        assert!(iter.next().is_none());
    }

    #[test]
    fn chunks_iterator_with_remainder() {
        // Test chunk iterator when the chunk size is not a multiple of the length.
        let base_addr = PageAddr::new(RawAddr::supervisor(0x1000)).unwrap();
        let seq_len: u64 = 16;
        let chunk_size = NonZeroU64::new(5).unwrap();

        // Create a SequentialPages of length 16
        let seq = create_test_sequential_pages(base_addr, seq_len);
        let page_size = seq.page_size();

        // ... and a chunk iterator of 5 pages.
        let mut iter = seq.into_chunks_iter(chunk_size);
        assert_eq!(iter.size_hint(), (4, Some(4)));
        let mut addr = base_addr;
        // First three call will contain exactly `chunk_size` sequential pages.
        for _i in 0..3 {
            let subseq = iter.next().unwrap();
            assert_eq!(subseq.page_size(), page_size);
            assert_eq!(subseq.base(), addr);
            assert_eq!(subseq.len(), chunk_size.get());
            addr = addr
                .checked_add_pages_with_size(chunk_size.get(), page_size)
                .unwrap();
        }
        // Fifth call should be contain a sequential page of length one.
        let subseq = iter.next().unwrap();
        assert_eq!(subseq.page_size(), page_size);
        assert_eq!(subseq.base(), addr);
        assert_eq!(subseq.len(), 1);
        // Sixth call should be none.
        assert!(iter.next().is_none());
    }

    #[test]
    fn chunks_iterator_at_last_page() {
        // Test chunk iterator page limit: the last page of the sequence is at the last possible
        // page address of a u64 address space. This is to check we can safely iterate over
        // sequences of any u64 address.
        let seq_len: u64 = 16;
        let base_addr = PageAddr::new(RawAddr::supervisor(
            u64::MAX - seq_len * PageSize::Size4k as u64 + 1,
        ))
        .unwrap();
        let chunk_size = NonZeroU64::new(1).unwrap();

        // Create a SequentialPages of length 16 at `(u64::MAX - 16 * PageSize::Size4k)`
        let seq = create_test_sequential_pages(base_addr, seq_len);
        let page_size = seq.page_size();

        // ... and a chunk iterator of 1 pages.
        let mut iter = seq.into_chunks_iter(chunk_size);
        assert_eq!(iter.size_hint(), (16, Some(16)));
        let mut addr = base_addr;

        for _i in 0..15 {
            let subseq = iter.next().unwrap();
            assert_eq!(subseq.page_size(), page_size);
            assert_eq!(subseq.base(), addr);
            assert_eq!(subseq.len(), chunk_size.get());
            addr = addr
                .checked_add_pages_with_size(chunk_size.get(), page_size)
                .unwrap();
        }
        // Check last page of the iterator.
        let subseq = iter.next().unwrap();
        assert_eq!(subseq.page_size(), page_size);
        assert_eq!(subseq.base(), addr);
        assert_eq!(subseq.len(), chunk_size.get());
        // Check that we're done.
        assert!(iter.next().is_none());
    }

    #[test]
    fn chunks_iterator_with_empty_sequence() {
        // Test chunk iterator with an empty sequence.
        let seq_len: u64 = 0;
        let base_addr = PageAddr::new(RawAddr::supervisor(0x1000)).unwrap();
        let chunk_size = NonZeroU64::new(1).unwrap();
        let seq = create_test_sequential_pages(base_addr, seq_len);
        let mut iter = seq.into_chunks_iter(chunk_size);

        // Check that we can't iterate on an empty Sequence.
        assert!(iter.next().is_none());
    }
}
