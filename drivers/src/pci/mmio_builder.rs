// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Helpers for PCI configuration space emulation.

#![allow(dead_code)]

use core::cmp::min;
use core::mem::size_of;

// Returns a bit mask covering the specified number of bytes.
fn byte_mask(bytes: usize) -> u32 {
    ((1u64 << (bytes * 8)) - 1) as u32
}

// Returns the bytes in the specified range in `val`.
fn select_bytes(val: u32, offset: usize, len: usize) -> u32 {
    (val >> (offset * 8)) & byte_mask(len)
}

// Updates the bytes in `dest` in the specified range with `src`.
fn update_bytes(dest: u32, offset: usize, len: usize, src: u32) -> u32 {
    let mask = byte_mask(len) << (offset * 8);
    dest & !mask | ((src << (offset * 8)) & mask)
}

/// A builder for emulated MMIO config space read operations.
///
/// In order to support partial accesses to word or dword registers, pushing a word or dword when
/// we're unaligned results in pushing only the bytes between the offset within the word/dword and
/// the end of the word/dword. For example, pushing a word register when `self.offset` is 1 results
/// in only the upper byte of the word being pushed into the result.
pub struct MmioReadBuilder {
    offset: usize,
    len: usize,
    written: usize,
    result: u32,
}

impl MmioReadBuilder {
    /// Creates a new `MmioReadBuilder` for a read of `len` bytes at `offset`.
    pub fn new(offset: usize, len: usize) -> Self {
        Self {
            offset,
            len,
            written: 0,
            result: 0,
        }
    }

    /// Returns the current offset of the read.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns if the read is completed.
    pub fn done(&self) -> bool {
        self.written == self.len
    }

    /// Push a byte register to this builder.
    pub fn push_byte(&mut self, value: u8) {
        self.result |= (value as u32) << (self.written * 8);
        self.written += 1;
        self.offset += 1;
    }

    /// Push a word register to this builder. We only push a subset of the word if we're unaligned
    /// or the remaining number of bytes is less than the size of a word.
    pub fn push_word(&mut self, value: u16) {
        let offset_in_word = self.offset & 0x1;
        let to_write = min(size_of::<u16>() - offset_in_word, self.len - self.written);
        let value = select_bytes(value as u32, offset_in_word, to_write);
        self.result |= value << (self.written * 8);
        self.written += to_write;
        self.offset += to_write;
    }

    /// Push a dword register to this builder. We only push a subset of the dword if we're unaligned
    /// or the remaining number of bytes is less than the size of a dword.
    pub fn push_dword(&mut self, value: u32) {
        let offset_in_dword = self.offset & 0x3;
        let to_write = min(size_of::<u32>() - offset_in_dword, self.len - self.written);
        let value = select_bytes(value, offset_in_dword, to_write);
        self.result |= value << (self.written * 8);
        self.written += to_write;
        self.offset += to_write;
    }

    /// Consumes this `MmioReadBuilder`, returning the result of the read as a dword value.
    pub fn result(self) -> u32 {
        self.result
    }
}

/// A builder for emulated MMIO config space write operations.
///
/// Like `MmioReadBuilder`, we need to support partial accesses to word or dword registers. This is
/// done by performing a read-modify-write operation on a source word/dword with only the bytes that
/// were popped. For example, popping a dword register when `self.offset` is 2 only updates upper word
/// in the dword.
pub struct MmioWriteBuilder {
    offset: usize,
    len: usize,
    read: usize,
    value: u32,
}

impl MmioWriteBuilder {
    /// Creates a new `MmioWriteBuilder` for a `len` bytes write of `value` at `offset`.
    pub fn new(offset: usize, value: u32, len: usize) -> Self {
        Self {
            offset,
            len,
            read: 0,
            value,
        }
    }

    /// Returns the current offset of the write.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns if the write is completed.
    pub fn done(&self) -> bool {
        self.read == self.len
    }

    /// Pops a byte register from this builder.
    pub fn pop_byte(&mut self) -> u8 {
        let result = (self.value >> (self.read * 8)) as u8;
        self.read += 1;
        self.offset += 1;
        result
    }

    /// Pops a word register from this builder, returning `dest` updated with the bytes that were
    /// popped.
    pub fn pop_word(&mut self, dest: u16) -> u16 {
        let offset_in_word = self.offset & 0x1;
        let to_read = min(size_of::<u16>() - offset_in_word, self.len - self.read);
        let result = update_bytes(
            dest as u32,
            offset_in_word,
            to_read,
            self.value >> (self.read * 8),
        );
        self.read += to_read;
        self.offset += to_read;
        result as u16
    }

    /// Pops a dword register from this builder, returning `dest` updated with the bytes that were
    /// popped.
    pub fn pop_dword(&mut self, dest: u32) -> u32 {
        let offset_in_dword = self.offset & 0x3;
        let to_read = min(size_of::<u32>() - offset_in_dword, self.len - self.read);
        let result = update_bytes(
            dest,
            offset_in_dword,
            to_read,
            self.value >> (self.read * 8),
        );
        self.read += to_read;
        self.offset += to_read;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmio_builder() {
        let mut op = MmioReadBuilder::new(0, 4);
        op.push_byte(0xaa);
        op.push_byte(0xbb);
        op.push_word(0x1122);
        assert!(op.done());
        assert_eq!(op.result(), 0x1122bbaa);

        let mut op = MmioReadBuilder::new(1, 1);
        op.push_word(0xbeef);
        assert!(op.done());
        assert_eq!(op.result(), 0xbe);

        let mut op = MmioReadBuilder::new(2, 2);
        op.push_dword(0xdeadbeef);
        assert!(op.done());
        assert_eq!(op.result(), 0xdead);

        let mut op = MmioReadBuilder::new(3, 1);
        op.push_dword(0xfeedface);
        assert!(op.done());
        assert_eq!(op.result(), 0xfe);

        let mut op = MmioWriteBuilder::new(0, 0xdeadbeef, 4);
        assert_eq!(op.pop_byte(), 0xef);
        assert_eq!(op.pop_byte(), 0xbe);
        assert_eq!(op.pop_word(0), 0xdead);
        assert!(op.done());

        let mut op = MmioWriteBuilder::new(1, 0x99, 1);
        assert_eq!(op.pop_word(0xabcd), 0x99cd);
        assert!(op.done());

        let mut op = MmioWriteBuilder::new(2, 0x42, 1);
        assert_eq!(op.pop_dword(0xf00ff00f), 0xf042f00f);
        assert!(op.done());

        let mut op = MmioWriteBuilder::new(2, 0xfeed, 2);
        assert_eq!(op.pop_dword(0xdeadbeef), 0xfeedbeef);
        assert!(op.done());
    }
}
