// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use riscv_pages::{ConvertedPhysPage, PhysPage, SupervisorPageAddr};

use crate::{PageTracker, PageTrackingResult};

/// A linked list of exclusively-owned `PhysPages` created using links in the array of `PageInfo`
/// structs. This list can be used to pass around a list of non-contiguous pages without having
/// to allocate storage (e.g. in a `Vec<>`). Pages are unlinked from the list by calling `pop()`.
/// Any pages remaining on the list when the list is dropped are unlinked.
pub struct PageList<P: PhysPage> {
    page_tracker: PageTracker,
    head: Option<SupervisorPageAddr>,
    tail: Option<SupervisorPageAddr>,
    len: usize,
    page_state: PhantomData<P>,
}

impl<P: PhysPage> PageList<P> {
    /// Creates an empty `PageList`.
    pub fn new(page_tracker: PageTracker) -> Self {
        Self {
            page_tracker,
            head: None,
            tail: None,
            len: 0,
            page_state: PhantomData,
        }
    }

    /// Creates a `PageList` from the head page of an already-constructed list.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all pages in the list (`head` and the pages linked from it)
    /// are uniquely owned and are of type `P`.
    pub(crate) unsafe fn from_raw_parts(
        page_tracker: PageTracker,
        head: SupervisorPageAddr,
    ) -> Self {
        let mut len = 1;
        let mut tail = head;
        while let Some(addr) = page_tracker.linked_page(tail) {
            len += 1;
            tail = addr;
        }

        Self {
            page_tracker,
            head: Some(head),
            tail: Some(tail),
            len,
            page_state: PhantomData,
        }
    }

    /// Appends `page` to the end of the list. Returns an error if `page` is already linked.
    pub fn push(&mut self, page: P) -> PageTrackingResult<()> {
        if let Some(tail_addr) = self.tail {
            self.page_tracker.link_pages(tail_addr, page.addr())?;
            self.tail = Some(page.addr());
        } else {
            self.head = Some(page.addr());
            self.tail = Some(page.addr());
        }
        self.len += 1;
        Ok(())
    }

    /// Removes the head of the list.
    pub fn pop(&mut self) -> Option<P> {
        let addr = self.head?;
        self.head = self.page_tracker.unlink_page(addr);
        if self.head.is_none() {
            // List is now empty.
            self.tail = None;
        }
        self.len -= 1;
        // Safety: This list has unique ownership of the page ever sicne it was pushed.
        Some(unsafe { P::new(addr) })
    }

    /// Returns if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Returns the number of pages in the list.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the address of the page at the head of the list.
    pub fn peek(&self) -> Option<SupervisorPageAddr> {
        self.head
    }

    /// Returns if the list of pages is contiguous.
    pub fn is_contiguous(&self) -> bool {
        if self.head.is_none() {
            return true;
        }
        let mut prev = self.head.unwrap();
        while let Some(addr) = self.page_tracker.linked_page(prev) {
            if let Some(next) = prev.checked_add_pages(1) && addr == next {
                prev = next;
            } else {
                return false;
            }
        }
        true
    }

    /// Returns the `PageTracker` this list is using.
    pub fn page_tracker(&self) -> PageTracker {
        self.page_tracker.clone()
    }
}

impl<P: PhysPage> Iterator for PageList<P> {
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<P: PhysPage> ExactSizeIterator for PageList<P> {}

impl<P: PhysPage> Drop for PageList<P> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Like `PageList`, but for pages that are locked for assignment or reclaim. Pages are
/// unlocked when the list is dropped, in addition to unlinking them.
pub struct LockedPageList<P: ConvertedPhysPage> {
    inner: PageList<P>,
}

impl<P: ConvertedPhysPage> LockedPageList<P> {
    /// Creates an empty `LockedPageList`.
    pub fn new(page_tracker: PageTracker) -> Self {
        Self {
            inner: PageList::new(page_tracker),
        }
    }
}

impl<P: ConvertedPhysPage> Iterator for LockedPageList<P> {
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<P: ConvertedPhysPage> ExactSizeIterator for LockedPageList<P> {}

impl<P: ConvertedPhysPage> Drop for LockedPageList<P> {
    fn drop(&mut self) {
        while let Some(p) = self.inner.pop() {
            // Unwrap ok since pages on the list must be uniquely-owned and locked PhysPage to be
            // on the list.
            self.inner.page_tracker.unlock_page(p).unwrap();
        }
    }
}

impl<P: ConvertedPhysPage> Deref for LockedPageList<P> {
    type Target = PageList<P>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P: ConvertedPhysPage> DerefMut for LockedPageList<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_pages::*;

    #[test]
    fn page_list() {
        let (page_tracker, mut pages) = PageTracker::new_in_test();

        let first_page = pages.next().unwrap();
        let first_page_addr = first_page.addr();
        {
            let mut list = PageList::new(page_tracker.clone());
            list.push(first_page).unwrap();
            for _ in 0..5 {
                list.push(pages.next().unwrap()).unwrap();
            }
            // Not safe -- just a test.
            let already_linked: Page<ConvertedClean> = unsafe { Page::new(first_page_addr) };
            assert!(list.push(already_linked).is_err());
        }

        let mut new_list = PageList::new(page_tracker.clone());
        for _ in 0..5 {
            new_list.push(pages.next().unwrap()).unwrap();
        }
        // Not safe -- just a test.
        let was_linked: Page<ConvertedClean> = unsafe { Page::new(first_page_addr) };
        new_list.push(was_linked).unwrap();
    }
}
