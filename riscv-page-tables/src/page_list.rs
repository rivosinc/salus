// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use riscv_pages::{ConvertedPhysPage, PhysPage, SupervisorPageAddr};

use crate::page_tracking::PageTracker;
use crate::PageTrackingResult;

/// A linked list of exclusively-owned `PhysPages` created using links in the array of `PageInfo`
/// structs. This list can be used to pass around a list of non-contiguous pages without having
/// to allocate storage (e.g. in a `Vec<>`). Pages are unlinked from the list by calling `pop()`.
/// Any pages remaining on the list when the list is dropped are unlinked.
pub struct PageList<P: PhysPage> {
    page_tracker: PageTracker,
    head: Option<SupervisorPageAddr>,
    tail: Option<SupervisorPageAddr>,
    page_state: PhantomData<P>,
}

impl<P: PhysPage> PageList<P> {
    /// Creates an empty `PageList`.
    pub fn new(page_tracker: PageTracker) -> Self {
        Self {
            page_tracker,
            head: None,
            tail: None,
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
        // Safety: This list has unique ownership of the page ever sicne it was pushed.
        Some(unsafe { P::new(addr) })
    }

    /// Returns if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
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
}

impl<P: PhysPage> Iterator for PageList<P> {
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop()
    }
}

impl<P: PhysPage> Drop for PageList<P> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Like `PageList`, but for converted pages that are locked for assignemnt or reclaim. Pages are
/// released back to the "Converted" state when the list is dropped, in addition to unlinking them.
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
}

impl<P: ConvertedPhysPage> Drop for LockedPageList<P> {
    fn drop(&mut self) {
        while let Some(p) = self.inner.pop() {
            // Unwrap ok since pages on the list must be uniquely-owned ConvertedPhysPages to be
            // on the list and all unqiuely-owned ConvertedPhysPages must by definition be in the
            // "ConvertedLocked" state.
            self.page_tracker.put_converted_page(p).unwrap();
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
