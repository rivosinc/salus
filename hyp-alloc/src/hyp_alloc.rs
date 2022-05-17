// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use core::slice;
use riscv_pages::{PageAddr, PageSize, RawAddr, SequentialPages};
use spin::Mutex;

struct HypAllocInner {
    mem: NonNull<[u8]>,
    end: usize,
    page_size: PageSize,
}

impl HypAllocInner {
    /// Returns the length of the underlying heap storage.
    fn capacity(&self) -> usize {
        // SAFETY: We construct a valid reference only for the unsafe block, returning a copy of
        // the slice length.
        unsafe { self.mem.as_ref().len() }
    }

    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align();
        let size = layout.size();
        let capacity = self.capacity();

        let mem: *mut u8 = self.mem.as_ptr().cast();
        // Won't wrap since self.end is known not to overflow by construction.
        let end_ptr = mem.wrapping_add(self.end);

        let align_offset = end_ptr.align_offset(align);
        let block_start_offset = self.end.checked_add(align_offset).ok_or(AllocError)?;
        if block_start_offset > capacity {
            return Err(AllocError);
        }

        let block_end_offset = block_start_offset.checked_add(size).ok_or(AllocError)?;
        if block_end_offset > capacity {
            return Err(AllocError);
        }
        self.end = block_end_offset;

        let block_start_ptr = mem.wrapping_add(block_start_offset);
        // SAFETY: No other mutable reference to this slice exists.
        let block_slice = unsafe { slice::from_raw_parts_mut(block_start_ptr, size) };

        // SAFETY: block_slice is a non-null pointer to a valid slice.
        Ok(unsafe { NonNull::new_unchecked(block_slice) })
    }
}

/// A simple thread-safe bump-pointer allocator backed by a fixed-length contiguous range of Pages.
/// Implements the `Allocator` trait so that it may be used with standard containers supporting the
/// allocator API.
pub struct HypAlloc {
    inner: Mutex<HypAllocInner>,
}

impl HypAlloc {
    /// Creates an allocator from a range of pages which will be used as storage for the allocator.
    pub fn from_pages(pages: SequentialPages) -> Self {
        let inner = HypAllocInner {
            mem: NonNull::slice_from_raw_parts(
                NonNull::new(pages.base().bits() as *mut u8).unwrap(),
                pages.length_bytes().try_into().unwrap(),
            ),
            end: 0,
            page_size: pages.page_size(),
        };
        HypAlloc {
            inner: Mutex::new(inner),
        }
    }

    /// Destroys the allocator and returns the pages which were used as storage.
    pub fn to_pages(self) -> SequentialPages {
        let inner = self.inner.lock();
        let base = inner.mem.as_mut_ptr() as u64;
        let num_pages = (inner.mem.len() as u64) / inner.page_size as u64;
        unsafe {
            // Safe since this allocator must own this range of pages.
            SequentialPages::from_mem_range(
                PageAddr::with_size(RawAddr::supervisor(base), inner.page_size).unwrap(),
                num_pages,
            )
        }
    }
}

unsafe impl<'a> Allocator for &'a HypAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align();
        let size = layout.size();
        if size == 0 {
            // SAFETY: align is always nonzero.
            let aligned_ptr = unsafe { NonNull::new_unchecked(align as *mut u8) };

            // SAFETY: a zero-length slice is safe to construct with a dangling pointer.
            let empty_slice = unsafe { slice::from_raw_parts_mut(aligned_ptr.as_ptr(), 0) };

            // SAFETY: empty_slice is a dangling, aligned non-null pointer.
            return Ok(unsafe { NonNull::new_unchecked(empty_slice) });
        }

        let mut inner = self.inner.lock();
        inner.allocate(layout)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    fn stub_heap() -> HypAlloc {
        const ONE_MEG: usize = 1024 * 1024;
        const MEM_ALIGN: usize = 2 * ONE_MEG;
        const MEM_SIZE: usize = 256 * ONE_MEG;
        let backing_mem = vec![0u8; MEM_SIZE + MEM_ALIGN];
        let aligned_pointer = unsafe {
            // Not safe - just a test
            backing_mem
                .as_ptr()
                .add(backing_mem.as_ptr().align_offset(MEM_ALIGN))
        };
        let start_page = PageAddr::new(RawAddr::supervisor(aligned_pointer as u64)).unwrap();
        let num_pages = (MEM_SIZE as u64) / PageSize::Size4k as u64;
        let pages = unsafe {
            // Not safe - just a test
            SequentialPages::from_mem_range(start_page, num_pages)
        };
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        HypAlloc::from_pages(pages)
    }

    #[test]
    fn basic_alloc() {
        let alloc = stub_heap();
        {
            let mut vec = Vec::new_in(&alloc);
            vec.push(1);
            vec.push(5);
            vec.push(10);
            assert_eq!(vec.len(), 3);

            let five = Box::new_in(5, &alloc);
            assert_eq!(*five, 5);
        }
        let _ = alloc.to_pages();
    }
}
