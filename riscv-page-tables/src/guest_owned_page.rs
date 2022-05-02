// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::result;
use riscv_pages::{AlignedPageAddr4k, PageSize, PageSize4k};
// Supports read and write operations on an encapsulated SPA.
// Note that we currently use single-byte volatile reads and writes,
// and should consider adding support for u64 chunks if performance
// becomes a concern. This is unlikely to be an issue for reading and
// writing tens of bytes for measurements
// TODO: Consider adding support for non-4K pages
pub struct GuestOwnedPage {
    spa: AlignedPageAddr4k,
}

/// Errors that can be raised while building the memory map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The specified offset was invalid
    InvalidOffset,
    /// The read / write operation would have exceeded the page boundary
    OutOfBounds,
}
pub type Result<T> = result::Result<T, Error>;

impl GuestOwnedPage {
    /// # Safety
    /// The caller must ensure that that spa points to valid SPA, and corresponds to a valid GPA
    /// The caller must also ensure the following until  this GuestOwnedPage instance is dropped:
    /// 1) The GPA -> SPA mapping will remain invariant in the guest
    /// 2) The SPA will not be remapped to another guest
    /// 3) No other GuestOwnedPage instance exists with the same SPA (i.e., no aliasing)
    pub unsafe fn new(spa: AlignedPageAddr4k) -> Self {
        GuestOwnedPage { spa }
    }

    // Reads the specified number of bytes from the specified SPA offset
    // Returns an error if:
    // 1) Bytes to be read + offset > sizeof(SPA page)
    // 2) Destination buffer is big enough
    pub fn read(&self, spa_offset: usize, buffer: &mut [u8]) -> Result<()> {
        let last_offset = spa_offset
            .checked_add(buffer.len())
            .ok_or(Error::InvalidOffset)?;
        if last_offset <= PageSize4k::SIZE_BYTES as usize {
            let spa = self.spa.bits() as *const u8;
            for (i, c) in buffer.iter_mut().enumerate() {
                // Bounds checks were successfully completed above
                unsafe {
                    *c = core::ptr::read_volatile(spa.offset((spa_offset + i).try_into().unwrap()));
                }
            }
            return Ok(());
        }

        Err(Error::OutOfBounds)
    }

    // Writes the specified number of bytes from the input buffer at the specified SPA offset
    // Returns an error if:
    // 1) Bytes to be written + offset > sizeof(SPA page)
    // 2) Input buffer is too small
    pub fn write(&mut self, spa_offset: usize, buffer: &[u8]) -> Result<()> {
        let last_offset = spa_offset
            .checked_add(buffer.len())
            .ok_or(Error::InvalidOffset)?;
        if last_offset <= PageSize4k::SIZE_BYTES as usize {
            let spa = self.spa.bits() as *mut u8;
            for (i, c) in buffer.iter().enumerate() {
                // Bounds checks were successfully completed above
                unsafe {
                    core::ptr::write_volatile(spa.offset((spa_offset + i).try_into().unwrap()), *c);
                }
            }
            return Ok(());
        }

        Err(Error::OutOfBounds)
    }
}

#[test]
fn test_spa() {
    use riscv_pages::PhysAddr;
    let source_buffer = [0xAAu8; 4096];
    // The compiler thinks that the buffers don't need to be mutable
    // but it has no visibility into what GuestOwnedPage does with
    // the physical address
    let mut dest_buffer = [0u8; 8192];
    let mut aligned_ptr = unsafe {
        // Safe because the above allocation guarantees that the result is
        // still a valid pointer.
        dest_buffer
            .as_ptr()
            .add(dest_buffer.as_ptr().align_offset(4096))
    };

    let mut spa = unsafe {
        GuestOwnedPage::new(AlignedPageAddr4k::new(PhysAddr::new(aligned_ptr as u64)).unwrap())
    };

    assert!(spa.write(0, &source_buffer).is_ok());
    assert!(spa.write(1, &source_buffer).is_err());
    assert!(spa.write(1, &source_buffer).is_err());
    assert!(spa.write(usize::MAX, &source_buffer).is_err());
    let align_offset = unsafe { aligned_ptr.sub(dest_buffer.as_ptr() as usize) as usize };

    assert!(dest_buffer[align_offset..align_offset + 4096] == source_buffer);

    let delta = [0xBBu8; 256];
    // Bytes 0..255 are now 0xBB
    assert!(spa.write(0, &delta).is_ok());
    assert!(dest_buffer[align_offset..align_offset + delta.len()] == delta[..]);
    assert!(dest_buffer[align_offset + 1..=align_offset + delta.len() + 1] != delta[..]);

    let mut delta_compare = [0u8; 256];
    assert!(spa.read(0, &mut delta_compare).is_ok());
    assert!(delta_compare == delta);

    // Bytes at align_offset 256 must still be 0xAA
    assert!(spa.read(256, &mut delta_compare).is_ok());
    assert!(delta_compare == source_buffer[0..=255]);

    // Read align_offset exceeds bounds
    assert!(spa.read(4096, &mut delta_compare).is_err());
    assert!(spa.read(usize::MAX, &mut delta_compare).is_err());

    // Reset everything to 0xAAs
    assert!(spa.write(0, &source_buffer[0..=255]).is_ok());
    assert!(dest_buffer[align_offset..align_offset + 4096] == source_buffer);
}
