// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt;
use core::marker::PhantomData;

use crate::page::{CleanablePhysPage, Page, PageSize, PhysPage, SupervisorPageAddr};
use crate::state::*;

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

/// An error resulting from trying to convert an iterator of pages to `SequentialPages`.
pub enum Error<S: State, I: Iterator<Item = Page<S>>> {
    Empty,
    NonContiguous(I),
    NonUniformSize(I),
    Overflow(I),
}

/// An error resulting from trying to create a `SequentialPages` from a range of pages that are not
/// aligned to the requested size.
#[derive(Clone, Copy, Debug)]
pub struct UnalignedPages;

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
}

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
}
