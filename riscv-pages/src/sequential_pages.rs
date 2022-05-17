// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::fmt;

use crate::page::{Page, PageSize, SupervisorPageAddr};

/// `SequentialPages` holds a range of consecutive pages of the same size. Each page's address is one
/// page after the previous. This forms a contiguous area of memory suitable for holding an array or
/// other linear data.
#[derive(Debug)]
pub struct SequentialPages {
    addr: SupervisorPageAddr,
    count: u64,
}

/// An error resulting from trying to convert an iterator of pages to `SequentialPages`.
pub enum Error<I: Iterator<Item = Page>> {
    Empty,
    NonContiguous(I),
    NonUniformSize(I),
    Overflow(I),
}

/// An error resulting from trying to create a `SequentialPages` from a range of pages with
/// mismatched sizes.
#[derive(Clone, Copy, Debug)]
pub struct PageSizeMismatch;

impl SequentialPages {
    /// Creates a `SequentialPages` form the passed iterator.
    ///
    /// If the passed pages are not consecutive, an Error will be returned holding an iterator to
    /// the passed in pages so they don't leak.
    /// If passed an empty iterator Err(None) will be returned.
    pub fn from_pages<T>(pages: T) -> Result<Self, Error<impl Iterator<Item = Page>>>
    where
        T: IntoIterator<Item = Page>,
    {
        let mut page_iter = pages.into_iter();

        let first_page = page_iter.next().ok_or(Error::Empty)?;

        let addr = first_page.addr();

        let mut last_addr = addr;
        let mut seq = Self { addr, count: 1 };
        while let Some(page) = page_iter.next() {
            let this_addr = page.addr();
            if this_addr.size() != last_addr.size() {
                return Err(Error::NonUniformSize(
                    seq.into_iter()
                        .chain(core::iter::once(page))
                        .chain(page_iter),
                ));
            }
            let next_addr = match last_addr.checked_add_pages(1) {
                Some(a) => a,
                None => {
                    return Err(Error::Overflow(
                        seq.into_iter()
                            .chain(core::iter::once(page))
                            .chain(page_iter),
                    ))
                }
            };
            if this_addr != next_addr {
                return Err(Error::NonContiguous(
                    seq.into_iter()
                        .chain(core::iter::once(page))
                        .chain(page_iter),
                ));
            }
            last_addr = this_addr;
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
        self.addr.size() as u64 * self.count
    }

    /// Returns the size of the backing pages.
    pub fn page_size(&self) -> PageSize {
        self.addr.size()
    }

    /// Returns `SequentialPages` for the memory range provided.
    /// # Safety
    /// The range's ownership is given to `SequentialPages`, the caller must uniquely own that
    /// memory.
    pub unsafe fn from_mem_range(addr: SupervisorPageAddr, count: u64) -> Self {
        Self { addr, count }
    }

    /// Returns `SequentialPages` for the page range [start, end). The pages must be of the same
    /// size.
    ///
    /// # Safety
    /// The range's ownership is given to `SequentialPages`, the caller must uniquely own that
    /// memory.
    pub unsafe fn from_page_range(
        start: SupervisorPageAddr,
        end: SupervisorPageAddr,
    ) -> Result<Self, PageSizeMismatch> {
        if start.size() != end.size() {
            return Err(PageSizeMismatch);
        }
        Ok(Self {
            addr: start,
            count: end.bits().checked_sub(start.bits()).unwrap() / start.size() as u64,
        })
    }
}

impl From<Page> for SequentialPages {
    fn from(p: Page) -> Self {
        Self {
            addr: p.addr(),
            count: 1,
        }
    }
}

/// An iterator of the individual pages previously used to build a `SequentialPages` struct.
/// Used to reclaim the pages from `SequentialPages`, returned from `SequentialPages::into_iter`.
pub struct SeqPageIter {
    pages: SequentialPages,
}

impl Iterator for SeqPageIter {
    type Item = Page;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pages.count == 0 {
            return None;
        }
        let addr = self.pages.addr;
        self.pages.addr = self.pages.addr.checked_add_pages(1).unwrap();
        self.pages.count -= 1;
        // Safe because `pages` owns the memory, which can be converted to pages because it is owned
        // and aligned.
        unsafe { Some(Page::new(addr)) }
    }
}

impl IntoIterator for SequentialPages {
    type Item = Page;
    type IntoIter = SeqPageIter;
    fn into_iter(self) -> Self::IntoIter {
        SeqPageIter { pages: self }
    }
}

impl<I: Iterator<Item = Page>> fmt::Debug for Error<I> {
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
        let pages = unsafe {
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
        let pages = unsafe {
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
        let pages: [Page; 0] = [];
        let result = SequentialPages::from_pages(pages);
        match result {
            Ok(_) => panic!("didn't fail with empty pages"),
            Err(Error::Empty) => (),
            Err(_) => panic!("failed with unexpected error"),
        }
    }

    #[test]
    fn from_single() {
        let p = unsafe {
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
        let seq = unsafe {
            SequentialPages::from_mem_range(PageAddr::new(RawAddr::supervisor(0x1000)).unwrap(), 4)
        };
        let mut pages = seq.into_iter();
        assert_eq!(0x1000, pages.next().unwrap().addr().bits());
        assert_eq!(0x2000, pages.next().unwrap().addr().bits());
        assert_eq!(0x3000, pages.next().unwrap().addr().bits());
        assert_eq!(0x4000, pages.next().unwrap().addr().bits());
        assert!(pages.next().is_none());
    }
}
