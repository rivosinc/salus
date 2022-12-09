// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

//! RiscV ELF loader library for salus

// For testing use the std crate.
#[cfg(test)]
#[macro_use]
extern crate std;

use arrayvec::ArrayVec;
use core::{fmt, result};

// Maximum size of Program Headers supported by the loader.
const ELF_SEGMENTS_MAX: usize = 8;

/// Elf Offset Helper
///
/// An Elf Offset. A separate type to be sure to never used it
/// directly, but only through `slice_*` functions.
#[repr(packed, C)]
#[derive(Copy, Clone)]
pub struct ElfOffset64 {
    inner: u64,
}

impl ElfOffset64 {
    fn as_usize(&self) -> usize {
        // We're 64-bit. u64 fits in a usize.
        self.inner as usize
    }

    fn usize_add(self, other: usize) -> Option<ElfOffset64> {
        let inner = self.inner.checked_add(other as u64)?;
        Some(Self { inner })
    }
}

impl From<usize> for ElfOffset64 {
    fn from(val: usize) -> Self {
        Self { inner: val as u64 }
    }
}

fn slice_check_offset(bytes: &[u8], offset: ElfOffset64) -> bool {
    bytes.len() > offset.as_usize()
}

fn slice_check_range(bytes: &[u8], offset: ElfOffset64, size: usize) -> bool {
    if size < 1 {
        return false;
    }

    if let Some(last) = offset.usize_add(size - 1) {
        slice_check_offset(bytes, last)
    } else {
        false
    }
}

fn slice_get_range(bytes: &[u8], offset: ElfOffset64, len: usize) -> Option<&[u8]> {
    if slice_check_range(bytes, offset, len) {
        let start = offset.as_usize();
        Some(&bytes[start..start + len])
    } else {
        None
    }
}

/// ELF64 Program Header Table Entry
#[repr(packed, C)]
#[derive(Copy, Clone)]
pub struct ElfProgramHeader64 {
    p_type: u32,
    p_flags: u32,
    p_offset: ElfOffset64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

// ELF Segment Types
// The array element specifies a loadable segment
const PT_LOAD: u32 = 1;

// Elf Segment Permission
// Execute
const PF_X: u32 = 0x1;
// Write
const PF_W: u32 = 0x2;
// Read
const PF_R: u32 = 0x4;

/// ELF64 Header
#[repr(packed, C)]
#[derive(Copy, Clone)]
pub struct ElfHeader64 {
    ei_magic: [u8; 4],
    ei_class: u8,
    ei_data: u8,
    ei_version: u8,
    ei_osabi: u8,
    ei_abiversion: u8,
    ei_pad: [u8; 7],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: ElfOffset64,
    e_shoff: ElfOffset64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

const EI_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const EI_CLASS_64: u8 = 2;
const EI_DATA_LE: u8 = 1;
const EI_VERSION_1: u8 = 1;
const E_TYPE_EXEC: u16 = 2;
const E_MACHINE_RISCV: u16 = 0xf3;
const E_VERSION_1: u32 = 1;
const E_EHSIZE: u16 = 0x40;

/// ELF Loader Errors.
#[derive(Debug)]
pub enum Error {
    /// Requested to read after EOF.
    BadOffset,
    /// The ELF magic number is wrong.
    InvalidMagicNumber,
    /// Unexpected ELF Class
    InvalidClass,
    /// Unsupported Endiannes
    InvalidEndianness,
    /// ELF is not RISC V.
    NotRiscV,
    /// Unexpected ELF version.
    BadElfVersion,
    /// Unexpected ELF object type.
    BadElfType,
    /// Unexpected ELF header size.
    BadElfHeaderSize,
    /// Unexpected ELF PH Entry size.
    BadEntrySize,
    /// No Program Header table.
    NoProgramHeader,
    /// Malformed Program Header.
    ProgramHeaderMalformed,
    /// Segment Permissions Unsupported
    UnsupportedProgramHeaderFlags(u32),
}

#[derive(Debug)]
/// Mapping Permissions of an ELF Segment
pub enum ElfSegmentPerms {
    /// Read-Only
    ReadOnly,
    /// Read-Write
    ReadWrite,
    /// Executable Page (Read Only)
    ReadOnlyExecute,
}

impl fmt::Display for ElfSegmentPerms {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match &self {
            Self::ReadOnly => write!(f, "RO"),
            Self::ReadWrite => write!(f, "RW"),
            Self::ReadOnlyExecute => write!(f, "RX"),
        }
    }
}

/// A structure representing a segment.
#[derive(Debug)]
pub struct ElfSegment<'elf> {
    data: Option<&'elf [u8]>,
    vaddr: u64,
    size: usize,
    perms: ElfSegmentPerms,
}

impl<'elf> ElfSegment<'elf> {
    fn new(
        data: Option<&'elf [u8]>,
        vaddr: u64,
        size: usize,
        flags: u32,
    ) -> Result<ElfSegment<'elf>, Error> {
        let perms = if flags == PF_R {
            Ok(ElfSegmentPerms::ReadOnly)
        } else if flags == PF_R | PF_W {
            Ok(ElfSegmentPerms::ReadWrite)
        } else if flags == PF_R | PF_X {
            Ok(ElfSegmentPerms::ReadOnlyExecute)
        } else {
            Err(Error::UnsupportedProgramHeaderFlags(flags))
        }?;
        // Check size is valid
        vaddr
            .checked_add(size as u64)
            .ok_or(Error::ProgramHeaderMalformed)?;
        Ok(ElfSegment {
            data,
            vaddr,
            size,
            perms,
        })
    }

    /// Returns a reference to the data that must be populated at the beginning of the segment.
    pub fn data(&self) -> Option<&'elf [u8]> {
        self.data
    }

    /// Returns the Virtual Address of the start of the segment.
    pub fn vaddr(&self) -> u64 {
        self.vaddr
    }

    /// Return the size of the Virtual Address area.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return the mapping permissions of the segment.
    pub fn perms(&self) -> &ElfSegmentPerms {
        &self.perms
    }
}

/// A structure that checks and prepares and ELF for loading into memory.
pub struct ElfMap<'elf> {
    segments: ArrayVec<ElfSegment<'elf>, ELF_SEGMENTS_MAX>,
}

impl<'elf> ElfMap<'elf> {
    /// Create a new ElfMap from a slice containing an ELF file.
    pub fn new(bytes: &'elf [u8]) -> Result<ElfMap<'elf>, Error> {
        // Chek ELF Header
        let hbytes = slice_get_range(bytes, 0.into(), core::mem::size_of::<ElfHeader64>())
            .ok_or(Error::BadOffset)?;
        // Safe because we are sure that the size of the slice is the same size as ElfHeader64.
        let header: &'elf ElfHeader64 = unsafe { &*(hbytes.as_ptr() as *const ElfHeader64) };
        // Check magic number
        if header.ei_magic != EI_MAGIC {
            return Err(Error::InvalidMagicNumber);
        }
        // Check is 64bit ELF.
        if header.ei_class != EI_CLASS_64 {
            return Err(Error::InvalidClass);
        }
        // Check is Little-Endian
        if header.ei_data != EI_DATA_LE {
            return Err(Error::InvalidEndianness);
        }
        // Check ELF versions.
        if header.ei_version != EI_VERSION_1 || header.e_version != E_VERSION_1 {
            return Err(Error::BadElfVersion);
        }
        if header.e_type != E_TYPE_EXEC {
            return Err(Error::BadElfType);
        }
        // Check is RISC-V.
        if header.e_machine != E_MACHINE_RISCV {
            return Err(Error::NotRiscV);
        }
        // Check EH size.
        if header.e_ehsize != E_EHSIZE {
            return Err(Error::BadElfHeaderSize);
        }

        // Check Program Header Table
        let phnum = header.e_phnum as usize;
        // e_phoff is invalid if zero
        if phnum == 0 || header.e_phoff.as_usize() == 0 {
            return Err(Error::NoProgramHeader);
        }
        let phentsize = header.e_phentsize as usize;
        // Check that e_phentsize is >= of size of ElfProgramHeader64
        if core::mem::size_of::<ElfProgramHeader64>() > phentsize {
            return Err(Error::BadEntrySize);
        }
        // Check that we can read the program header table.
        let program_headers =
            slice_get_range(bytes, header.e_phoff, phnum * phentsize).ok_or(Error::BadOffset)?;

        // Load segments
        let mut segments = ArrayVec::<ElfSegment, ELF_SEGMENTS_MAX>::new();
        let num_segs = core::cmp::min(phnum, ELF_SEGMENTS_MAX);
        for i in 0..num_segs {
            // Find the i-th ELF Program Header.
            let phbytes = slice_get_range(program_headers, (i * phentsize).into(), phentsize)
                .ok_or(Error::BadOffset)?;
            // Safe because we are sure that the size of the slice is at least as big as ElfProgramHeader64
            let ph: &'elf ElfProgramHeader64 =
                unsafe { &*(phbytes.as_ptr() as *const ElfProgramHeader64) };

            // Ignore if not a load segment.
            if ph.p_type != PT_LOAD {
                continue;
            }
            // Create a segment from the PH.
            let data_size = ph.p_filesz as usize;
            let data = if data_size > 0 {
                Some(slice_get_range(bytes, ph.p_offset, data_size).ok_or(Error::BadOffset)?)
            } else {
                None
            };
            let vaddr = ph.p_vaddr;
            let size = ph.p_memsz as usize;
            let flags = ph.p_flags;
            let segment = ElfSegment::new(data, vaddr, size, flags)?;
            segments.push(segment);
        }
        Ok(Self { segments })
    }

    /// Return an iterator containings loadable segments of this ELF file.
    pub fn segments(&self) -> impl Iterator<Item = &ElfSegment<'elf>> {
        self.segments.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_header() -> ElfHeader64 {
        ElfHeader64 {
            ei_magic: EI_MAGIC,
            ei_class: EI_CLASS_64,
            ei_data: EI_DATA_LE,
            ei_version: EI_VERSION_1,
            ei_osabi: 0,      // Not used.
            ei_abiversion: 0, // Not used.
            ei_pad: [0u8; 7],
            e_type: E_TYPE_EXEC,
            e_machine: E_MACHINE_RISCV,
            e_version: E_VERSION_1,
            e_entry: 0,                    // Not used at this time.
            e_phoff: ElfOffset64::from(0), // Do not add a program header table.
            e_shoff: ElfOffset64::from(0), // Do not add a section header table.
            e_flags: 0,                    // Not used.
            e_ehsize: E_EHSIZE,
            e_phentsize: core::mem::size_of::<ElfProgramHeader64>() as u16,
            e_phnum: 0,     // No PHs
            e_shentsize: 0, // Not used.
            e_shnum: 0,     // Not used.
            e_shstrndx: 0,  // Not used.
        }
    }

    fn build_ph(p_type: u32, p_flags: u32, p_offset: usize, p_filesz: u64) -> ElfProgramHeader64 {
        ElfProgramHeader64 {
            p_type,
            p_flags,
            p_offset: ElfOffset64::from(p_offset),
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz,
            p_memsz: 0,
            p_align: 0,
        }
    }

    fn set_header(bytes: &mut [u8], header: &ElfHeader64) {
        let ptr = bytes.as_ptr() as *mut ElfHeader64;
        // Safe because bytes >= Elf header size.
        unsafe { *ptr = *header };
    }

    fn set_ph(bytes: &mut [u8], off: usize, ph: &ElfProgramHeader64) {
        assert!(off + core::mem::size_of::<ElfProgramHeader64>() <= bytes.len());
        let off: isize = off.try_into().unwrap();
        // Safe because we can fit `ElfProgramHeader64` at offset `off` `in bytes`.
        unsafe {
            let ptr = bytes.as_ptr().offset(off) as *mut ElfProgramHeader64;
            *ptr = *ph
        };
    }

    #[test]
    fn header_test() {
        const HEADER_SIZE: usize = core::mem::size_of::<ElfHeader64>();
        const PH_SIZE: usize = core::mem::size_of::<ElfProgramHeader64>();
        let mut bytes = [0u8; HEADER_SIZE + PH_SIZE * 2 + 20];
        {
            // Simple ELF program header. No PHs.
            let header = build_header();
            set_header(&mut bytes, &header);
            let rc = ElfMap::new(&bytes);
            // Should fail as we require PHs.
            assert!(rc.is_err());
        }
        {
            // Lie about the number of PHs.
            let mut header = build_header();
            header.e_phnum = 42;
            set_header(&mut bytes, &header);
            let rc = ElfMap::new(&bytes);
            // Should fail as we require PHs.
            assert!(rc.is_err());
        }
        {
            // Create an elf with a non-loadable PH.
            let ph = build_ph(0 /* PT_NULL */, 0, HEADER_SIZE + PH_SIZE, 10);
            let mut header = build_header();
            // add a PH, right after header.
            header.e_phnum = 1;
            header.e_phoff = ElfOffset64::from(HEADER_SIZE);
            set_header(&mut bytes, &header);
            set_ph(&mut bytes, HEADER_SIZE, &ph);
            let rc = ElfMap::new(&bytes);
            // This should succeed and the segments should be empty.
            assert!(rc.is_ok());
            let map = rc.unwrap();
            assert!(map.segments().next().is_none());
        }
        {
            // Create an elf with a PH with no data.
            let ph = build_ph(PT_LOAD, PF_R, HEADER_SIZE + PH_SIZE, 0);
            let mut header = build_header();
            // add a PH, right after header.
            header.e_phnum = 1;
            header.e_phoff = ElfOffset64::from(HEADER_SIZE);
            set_header(&mut bytes, &header);
            set_ph(&mut bytes, HEADER_SIZE, &ph);
            let rc = ElfMap::new(&bytes);
            // This should succeed and there should be a segment with no data.
            assert!(rc.is_ok());
            let map = rc.unwrap();
            let mut segs = map.segments();
            let s = segs.next();
            assert!(s.is_some());
            assert!(s.unwrap().data().is_none());
        }
        {
            // Create an elf with one non-loadable PH and one loadable.
            let ph1 = build_ph(0 /* PT_NULL */, 0, HEADER_SIZE + PH_SIZE * 2, 9);
            let ph2 = build_ph(PT_LOAD, PF_R, HEADER_SIZE + PH_SIZE * 2 + 9, 11);
            let mut header = build_header();
            // add PHs, right after header.
            header.e_phnum = 2;
            header.e_phoff = ElfOffset64::from(HEADER_SIZE);
            set_header(&mut bytes, &header);
            set_ph(&mut bytes, HEADER_SIZE, &ph1);
            set_ph(&mut bytes, HEADER_SIZE + PH_SIZE, &ph2);
            let rc = ElfMap::new(&bytes);
            // This should succeed and there should be one segment only with 11 bytes.
            assert!(rc.is_ok());
            let map = rc.unwrap();
            let mut segs = map.segments();
            let s = segs.next();
            assert!(s.is_some());
            assert_eq!(s.unwrap().data().unwrap().len(), 11);
            assert!(segs.next().is_none());
        }
        {
            // Create an elf with two loadable segments.
            let ph1 = build_ph(PT_LOAD, PF_R, HEADER_SIZE + PH_SIZE * 2, 9);
            let ph2 = build_ph(PT_LOAD, PF_R, HEADER_SIZE + PH_SIZE * 2 + 9, 11);
            let mut header = build_header();
            // add PHs, right after header.
            header.e_phnum = 2;
            header.e_phoff = ElfOffset64::from(HEADER_SIZE);
            set_header(&mut bytes, &header);
            set_ph(&mut bytes, HEADER_SIZE, &ph1);
            set_ph(&mut bytes, HEADER_SIZE + PH_SIZE, &ph2);
            let rc = ElfMap::new(&bytes);
            // This should succeed and there should be one segment only with 11 bytes.
            assert!(rc.is_ok());
            let map = rc.unwrap();
            let mut segs = map.segments();
            let s = segs.next();
            assert!(s.is_some());
            assert_eq!(s.unwrap().data().unwrap().len(), 9);
            let s = segs.next();
            assert!(s.is_some());
            assert_eq!(s.unwrap().data().unwrap().len(), 11);
            assert!(segs.next().is_none());
        }
        {
            // Create an elf with a single PH with an offset outside the file.
            let ph = build_ph(PT_LOAD, PF_R, bytes.len(), 10);
            let mut header = build_header();
            // add a PH, right after header.
            header.e_phnum = 1;
            header.e_phoff = ElfOffset64::from(HEADER_SIZE);
            set_header(&mut bytes, &header);
            set_ph(&mut bytes, HEADER_SIZE, &ph);
            let rc = ElfMap::new(&bytes);
            // This should fail.
            assert!(rc.is_err());
        }
        {
            // Create an elf with a single PH with a size that goes over the end of file.
            let ph = build_ph(PT_LOAD, PF_R, HEADER_SIZE + PH_SIZE, bytes.len() as u64);
            let mut header = build_header();
            // add a PH, right after header.
            header.e_phnum = 1;
            header.e_phoff = ElfOffset64::from(HEADER_SIZE);
            set_header(&mut bytes, &header);
            set_ph(&mut bytes, HEADER_SIZE, &ph);
            let rc = ElfMap::new(&bytes);
            // This should fail.
            assert!(rc.is_err());
        }
    }

    #[test]
    fn offset_test() {
        let bytes1 = [0u8; 5];
        let bytes2 = [0u8; 6];
        let off1 = ElfOffset64 { inner: 3 };
        let off2 = ElfOffset64 { inner: 5 };
        let off3 = ElfOffset64 { inner: 6 };

        let r = slice_check_offset(&bytes1, off1);
        assert_eq!(r, true);
        let r = slice_check_offset(&bytes1, off2);
        assert_eq!(r, false);
        let r = slice_check_offset(&bytes1, off3);
        assert_eq!(r, false);

        let r = slice_check_offset(&bytes2, off1);
        assert_eq!(r, true);
        let r = slice_check_offset(&bytes2, off2);
        assert_eq!(r, true);
        let r = slice_check_offset(&bytes2, off3);
        assert_eq!(r, false);

        let r = slice_check_range(&bytes1, 0.into(), 0);
        assert_eq!(r, false);
        let r = slice_get_range(&bytes1, 0.into(), 0);
        assert!(r.is_none());

        let r = slice_check_range(&bytes1, 0.into(), 5);
        assert_eq!(r, true);
        let r = slice_get_range(&bytes1, 0.into(), 5);
        assert!(r.is_some());
        assert_eq!(r.unwrap().len(), 5);

        let r = slice_check_range(&bytes1, 0.into(), 6);
        assert_eq!(r, false);
        let r = slice_get_range(&bytes1, 0.into(), 6);
        assert!(r.is_none());

        let r = slice_check_range(&bytes1, 4.into(), 1);
        assert_eq!(r, true);
        let r = slice_get_range(&bytes1, 4.into(), 1);
        assert!(r.is_some());
        assert_eq!(r.unwrap().len(), 1);

        let r = slice_check_range(&bytes1, 5.into(), 1);
        assert_eq!(r, false);
        let r = slice_get_range(&bytes1, 5.into(), 1);
        assert!(r.is_none());
    }
}
