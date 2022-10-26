// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Types for volatile access to memory.
//!
//! Two of the core rules for safe rust is no data races and no aliased mutable references.
//! `VolatileRef` and `VolatileSlice`, along with types that produce those which implement
//! `VolatileMemory`, allow us to sidestep that rule by wrapping pointers that absolutely have to be
//! accessed volatile. Some systems really do need to operate on shared memory and can't have the
//! compiler reordering or eliding access because it has no visibility into what other systems are
//! doing with that hunk of memory.
//!
//! For the purposes of maintaining safety, volatile memory has some rules of its own:
//! 1. No references or slices to volatile memory (`&` or `&mut`).
//! 2. Access should always been done with a volatile read or write.
//! The First rule is because having references of any kind to memory considered volatile would
//! violate pointer aliasing. The second is because unvolatile accesses are inherently undefined if
//! done concurrently without synchronization. With volatile access we know that the compiler has
//! not reordered or elided the access.

use core::cmp::min;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr::{copy, read_volatile, write_bytes, write_volatile};
use core::result;
use core::usize;

use crate::DataInit;

/// Errors from incorect indexing of volatile memory.
#[derive(Debug, PartialEq, Eq)]
pub enum VolatileMemoryError {
    /// Access out of bounds of the volatile memory slice.
    OutOfBounds {
        /// The address that is out of bounds.
        addr: usize,
    },
    /// Overflows `usize` in creating slice.
    Overflow {
        /// The base that overflows when combined with offset.
        base: usize,
        /// The offset that overflows when added to base.
        offset: usize,
    },
}

/// Result for volatile memory operations.
pub type VolatileMemoryResult<T> = result::Result<T, VolatileMemoryError>;

/// Convenience function for computing `base + offset` which returns
/// `Err(VolatileMemoryError::Overflow)` instead of panicking in the case `base + offset` exceeds
/// `u64::MAX`.
///
/// # Examples
///
/// ```
/// # use data_model::*;
/// # fn get_slice(offset: usize, count: usize) -> VolatileMemoryResult<()> {
/// let mem_end = calc_offset(offset, count)?;
/// if mem_end > 100 {
///     return Err(VolatileMemoryError::OutOfBounds { addr: mem_end });
/// }
/// # Ok(())
/// # }
/// ```
pub fn calc_offset(base: usize, offset: usize) -> VolatileMemoryResult<usize> {
    match base.checked_add(offset) {
        None => Err(VolatileMemoryError::Overflow { base, offset }),
        Some(m) => Ok(m),
    }
}

/// Trait for types that support raw volatile access to their data.
pub trait VolatileMemory {
    /// Gets a slice of memory at `offset` that is `count` bytes in length and supports volatile
    /// access.
    fn get_slice(&self, offset: usize, count: usize) -> VolatileMemoryResult<VolatileSlice>;

    /// Gets a `VolatileRef` at `offset`.
    fn get_ref<T: DataInit>(&self, offset: usize) -> VolatileMemoryResult<VolatileRef<T>> {
        let slice = self.get_slice(offset, size_of::<T>())?;
        Ok(VolatileRef {
            addr: slice.as_mut_ptr() as *mut T,
            phantom: PhantomData,
        })
    }
}

/// A slice of raw memory that supports volatile access. Like `core::slice`, but unlike
/// `core::slice`, it doesn't automatically deref to `&mut [u8]`.
#[derive(Copy, Clone, Debug)]
pub struct VolatileSlice<'a> {
    ptr: *mut u8,
    len: usize,
    phantom: PhantomData<&'a [u8]>,
}

impl<'a> VolatileSlice<'a> {
    /// Creates a slice of raw memory that must support volatile access.
    pub fn new(buf: &'a mut [u8]) -> VolatileSlice {
        Self {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
            phantom: PhantomData,
        }
    }

    /// Creates a `VolatileSlice` from a pointer and a length.
    ///
    /// # Safety
    ///
    /// In order to use this method safely, `addr` must be valid for reads and writes of `len` bytes
    /// and should live for the entire duration of lifetime `'a`.
    pub unsafe fn from_raw_parts(addr: *mut u8, len: usize) -> VolatileSlice<'a> {
        Self {
            ptr: addr,
            len,
            phantom: PhantomData,
        }
    }

    /// Gets a const pointer to this slice's memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Gets a mutable pointer to this slice's memory.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Gets the length of this slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Creates a copy of this slice with the address increased by `count` bytes, and the size
    /// reduced by `count` bytes.
    pub fn offset(self, count: usize) -> VolatileMemoryResult<VolatileSlice<'a>> {
        let new_addr =
            (self.ptr as usize)
                .checked_add(count)
                .ok_or(VolatileMemoryError::Overflow {
                    base: self.ptr as usize,
                    offset: count,
                })?;
        let new_size = self
            .len
            .checked_sub(count)
            .ok_or(VolatileMemoryError::OutOfBounds { addr: new_addr })?;
        Ok(VolatileSlice {
            ptr: new_addr as *mut u8,
            len: new_size,
            phantom: PhantomData,
        })
    }

    /// Similar to `get_slice` but the returned slice outlives this slice.
    ///
    /// The returned slice's lifetime is still limited by the underlying data's lifetime.
    pub fn sub_slice(self, offset: usize, count: usize) -> VolatileMemoryResult<VolatileSlice<'a>> {
        let mem_end = calc_offset(offset, count)?;
        if mem_end > self.len() {
            return Err(VolatileMemoryError::OutOfBounds { addr: mem_end });
        }
        let new_addr = (self.as_mut_ptr() as usize).checked_add(offset).ok_or(
            VolatileMemoryError::Overflow {
                base: self.as_mut_ptr() as usize,
                offset,
            },
        )?;

        // Safe because we have verified that the new memory is a subset of the original slice.
        Ok(unsafe { VolatileSlice::from_raw_parts(new_addr as *mut u8, count) })
    }

    /// Sets each byte of this slice with the given byte, similar to `memset`.
    ///
    /// The bytes of this slice are accessed in an arbitray order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use data_model::{VolatileSlice, VolatileMemoryResult};
    /// # fn test_write_45() -> VolatileMemoryResult<()> {
    /// let mut mem = [0u8; 32];
    /// let vslice = VolatileSlice::new(&mut mem[..]);
    /// vslice.write_bytes(45);
    /// for &v in &mem[..] {
    ///     assert_eq!(v, 45);
    /// }
    /// # Ok(())
    /// # }
    pub fn write_bytes(&self, value: u8) {
        // Safe because the memory is valid and needs only byte alignment.
        unsafe {
            write_bytes(self.as_mut_ptr(), value, self.len());
        }
    }

    /// Copies `self.size()` or `buf.len()` times the size of `T` bytes, whichever is smaller, to
    /// `buf`.
    ///
    /// The copy happens from smallest to largest address in `T` sized chunks using volatile reads.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # use data_model::{VolatileSlice, VolatileMemoryResult};
    /// # fn test_write_null() -> VolatileMemoryResult<()> {
    /// let mut mem = [0u8; 32];
    /// let vslice = VolatileSlice::new(&mut mem[..]);
    /// let mut buf = [5u8; 16];
    /// vslice.copy_to(&mut buf[..]);
    /// for v in &buf[..] {
    ///     assert_eq!(buf[0], 0);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy_to<T>(&self, buf: &mut [T])
    where
        T: DataInit,
    {
        let mut addr = self.as_mut_ptr() as *const u8;
        for v in buf.iter_mut().take(self.len() / size_of::<T>()) {
            // Safe because buf and self own their memory regions and the read call is bounded by
            // those regions of memory. `v` must be valid as it was returned from the but iterator
            // and addr must be valid because the length of self was used in the call to `take`.
            unsafe {
                *v = read_volatile(addr as *const T);
                addr = addr.add(size_of::<T>());
            }
        }
    }

    /// Copies `self.size()` or `slice.size()` bytes, whichever is smaller, to `slice`.
    ///
    /// The copies happen in an undefined order.
    /// # Examples
    ///
    /// ```
    /// # use data_model::{VolatileMemory, VolatileSlice, VolatileMemoryResult};
    /// # fn test_write_null() -> VolatileMemoryResult<()> {
    /// let mut mem = [0u8; 32];
    /// let vslice = VolatileSlice::new(&mut mem[..]);
    /// vslice.copy_to_volatile_slice(vslice.get_slice(16, 16)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy_to_volatile_slice(&self, slice: VolatileSlice) {
        // Safe because the target slices(`self` and `slice`) own the range they point to and the
        // copy is limited to the smaller of those ranges.
        unsafe {
            copy(
                self.as_mut_ptr() as *const u8,
                slice.as_mut_ptr(),
                min(self.len(), slice.len()),
            );
        }
    }

    /// Copies `self.size()` or `buf.len()` times the size of `T` bytes, whichever is smaller, to
    /// this slice's memory.
    ///
    /// The copy happens from smallest to largest address in `T` sized chunks using volatile writes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::path::Path;
    /// # use data_model::{VolatileMemory, VolatileSlice, VolatileMemoryResult};
    /// # fn test_write_null() -> VolatileMemoryResult<()> {
    /// let mut mem = [0u8; 32];
    /// let vslice = VolatileSlice::new(&mut mem[..]);
    /// let buf = [5u8; 64];
    /// vslice.copy_from(&buf[..]);
    /// for i in 0..4 {
    ///     assert_eq!(vslice.get_ref::<u32>(i * 4)?.load(), 0x05050505);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy_from<T>(&self, buf: &[T])
    where
        T: DataInit,
    {
        let mut addr = self.as_mut_ptr();
        for &v in buf.iter().take(self.len() / size_of::<T>()) {
            // Save becuse the two volatile slices own their memory and git copy is limited to the
            // minimum length of the two.
            unsafe {
                write_volatile(addr as *mut T, v);
                addr = addr.add(size_of::<T>());
            }
        }
    }
}

impl<'a> VolatileMemory for VolatileSlice<'a> {
    fn get_slice(&self, offset: usize, count: usize) -> VolatileMemoryResult<VolatileSlice> {
        self.sub_slice(offset, count)
    }
}

/// A memory location that supports volatile access of a `T`.
///
/// # Examples
///
/// ```
/// # use data_model::VolatileRef;
///   let mut v = 5u32;
///   assert_eq!(v, 5);
///   let v_ref = unsafe { VolatileRef::new(&mut v as *mut u32) };
///   assert_eq!(v_ref.load(), 5);
///   v_ref.store(500);
///   assert_eq!(v, 500);
#[derive(Debug)]
pub struct VolatileRef<'a, T: DataInit>
where
    T: 'a,
{
    addr: *mut T,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: DataInit> VolatileRef<'a, T> {
    /// Creates a reference to raw memory that must support volatile access of `T` sized chunks.
    ///
    /// # Safety
    /// To use this safely, the caller must guarantee that the memory at `addr` is big enough for a
    /// `T` and is available for the duration of the lifetime of the new `VolatileRef`. The caller
    /// must also guarantee that all other users of the given chunk of memory are using volatile
    /// accesses.
    pub unsafe fn new(addr: *mut T) -> VolatileRef<'a, T> {
        VolatileRef {
            addr,
            phantom: PhantomData,
        }
    }

    /// Gets the address of this slice's memory.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.addr
    }

    /// Gets the size of this slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::mem::size_of;
    /// # use data_model::VolatileRef;
    /// let v_ref = unsafe { VolatileRef::new(0 as *mut u32) };
    /// assert_eq!(v_ref.size(), size_of::<u32>());
    /// ```
    pub fn size(&self) -> usize {
        size_of::<T>()
    }

    /// Does a volatile write of the value `v` to the address of this ref.
    #[inline(always)]
    pub fn store(&self, v: T) {
        unsafe { write_volatile(self.addr, v) };
    }

    /// Does a volatile read of the value at the address of this ref.
    #[inline(always)]
    pub fn load(&self) -> T {
        // For the purposes of demonstrating why read_volatile is necessary, try replacing the code
        // in this function with the commented code below and running `cargo test --release`.
        // unsafe { *(self.addr as *const T) }
        unsafe { read_volatile(self.addr) }
    }

    /// Converts this `T` reference to a raw slice with the same size and address.
    pub fn to_slice(&self) -> VolatileSlice<'a> {
        unsafe { VolatileSlice::from_raw_parts(self.as_mut_ptr() as *mut u8, self.size()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Barrier};
    use std::thread::spawn;
    use std::vec::Vec;

    #[derive(Clone)]
    struct VecMem {
        mem: Arc<Vec<u8>>,
    }

    impl VecMem {
        fn new(size: usize) -> VecMem {
            let mut mem = Vec::new();
            mem.resize(size, 0);
            VecMem { mem: Arc::new(mem) }
        }
    }

    impl VolatileMemory for VecMem {
        fn get_slice(&self, offset: usize, count: usize) -> VolatileMemoryResult<VolatileSlice> {
            let mem_end = calc_offset(offset, count)?;
            if mem_end > self.mem.len() {
                return Err(VolatileMemoryError::OutOfBounds { addr: mem_end });
            }

            let new_addr = (self.mem.as_ptr() as usize).checked_add(offset).ok_or(
                VolatileMemoryError::Overflow {
                    base: self.mem.as_ptr() as usize,
                    offset,
                },
            )?;

            Ok(unsafe { VolatileSlice::from_raw_parts(new_addr as *mut u8, count) })
        }
    }

    #[test]
    fn ref_store() {
        let mut a = [0u8; 1];
        let a_ref = VolatileSlice::new(&mut a[..]);
        let v_ref = a_ref.get_ref(0).unwrap();
        v_ref.store(2u8);
        assert_eq!(a[0], 2);
    }

    #[test]
    fn ref_load() {
        let mut a = [5u8; 1];
        {
            let a_ref = VolatileSlice::new(&mut a[..]);
            let c = {
                let v_ref = a_ref.get_ref::<u8>(0).unwrap();
                assert_eq!(v_ref.load(), 5u8);
                v_ref
            };
            // To make sure we can take a v_ref out of the scope we made it in:
            c.load();
            // but not too far:
            // c
        } //.load()
        ;
    }

    #[test]
    fn ref_to_slice() {
        let mut a = [1u8; 5];
        let a_ref = VolatileSlice::new(&mut a[..]);
        let v_ref = a_ref.get_ref(1).unwrap();
        v_ref.store(0x12345678u32);
        let ref_slice = v_ref.to_slice();
        assert_eq!(v_ref.as_mut_ptr() as usize, ref_slice.as_mut_ptr() as usize);
        assert_eq!(v_ref.size(), ref_slice.len());
    }

    #[test]
    fn observe_mutate() {
        let a = VecMem::new(1);
        let a_clone = a.clone();
        let v_ref = a.get_ref::<u8>(0).unwrap();
        v_ref.store(99);

        let start_barrier = Arc::new(Barrier::new(2));
        let thread_start_barrier = start_barrier.clone();
        let end_barrier = Arc::new(Barrier::new(2));
        let thread_end_barrier = end_barrier.clone();
        spawn(move || {
            thread_start_barrier.wait();
            let clone_v_ref = a_clone.get_ref::<u8>(0).unwrap();
            clone_v_ref.store(0);
            thread_end_barrier.wait();
        });

        assert_eq!(v_ref.load(), 99);

        start_barrier.wait();
        end_barrier.wait();

        assert_eq!(v_ref.load(), 0);
    }

    #[test]
    fn slice_len() {
        let a = VecMem::new(100);
        let s = a.get_slice(0, 27).unwrap();
        assert_eq!(s.len(), 27);

        let s = a.get_slice(34, 27).unwrap();
        assert_eq!(s.len(), 27);

        let s = s.get_slice(20, 5).unwrap();
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn slice_overflow_error() {
        use core::usize::MAX;
        let a = VecMem::new(1);
        let res = a.get_slice(MAX, 1).unwrap_err();
        assert_eq!(
            res,
            VolatileMemoryError::Overflow {
                base: MAX,
                offset: 1,
            }
        );
    }

    #[test]
    fn slice_oob_error() {
        let a = VecMem::new(100);
        a.get_slice(50, 50).unwrap();
        let res = a.get_slice(55, 50).unwrap_err();
        assert_eq!(res, VolatileMemoryError::OutOfBounds { addr: 105 });
    }

    #[test]
    fn ref_overflow_error() {
        use core::usize::MAX;
        let a = VecMem::new(1);
        let res = a.get_ref::<u8>(MAX).unwrap_err();
        assert_eq!(
            res,
            VolatileMemoryError::Overflow {
                base: MAX,
                offset: 1,
            }
        );
    }

    #[test]
    fn ref_oob_error() {
        let a = VecMem::new(100);
        a.get_ref::<u8>(99).unwrap();
        let res = a.get_ref::<u16>(99).unwrap_err();
        assert_eq!(res, VolatileMemoryError::OutOfBounds { addr: 101 });
    }

    #[test]
    fn ref_oob_too_large() {
        let a = VecMem::new(3);
        let res = a.get_ref::<u32>(0).unwrap_err();
        assert_eq!(res, VolatileMemoryError::OutOfBounds { addr: 4 });
    }
}
