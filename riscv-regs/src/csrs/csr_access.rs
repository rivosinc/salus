// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! `ReadWriteRiscvCsr` type for RISC-V CSRs.

use core::arch::asm;
use core::marker::PhantomData;

use tock_registers::fields::Field;
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::RegisterLongName;

/// Read/Write registers.
#[derive(Copy, Clone)]
pub struct ReadWriteRiscvCsr<R: RegisterLongName, const V: u16> {
    associated_register: PhantomData<R>,
}

// TODO: Read-only and write-only register implementations.

impl<R: RegisterLongName, const V: u16> ReadWriteRiscvCsr<R, V> {
    pub const fn new() -> Self {
        ReadWriteRiscvCsr {
            associated_register: PhantomData,
        }
    }

    // Special methods only available on RISC-V CSRs, not found in the
    // usual tock-registers interface, others implemented through the
    // respective [`Readable`] and [`Writeable`] trait implementations

    /// Atomically swap the contents of a CSR
    ///
    /// Reads the current value of a CSR and replaces it with the
    /// specified value in a single instruction, returning the
    /// previous value.
    ///
    /// This method corresponds to the RISC-V `CSRRW rd, csr, rs1`
    /// instruction where `rs1 = in(reg) value_to_set` and `rd =
    /// out(reg) <return value>`.
    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    pub fn atomic_replace(&self, val_to_set: u64) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrrw {rd}, {csr}, {rs1}",
                 rd = out(reg) r,
                 csr = const V,
                 rs1 = in(reg) val_to_set);
        }
        r
    }

    /// Atomically swap the contents of a CSR
    ///
    /// Reads the current value of a CSR and replaces it with the
    /// specified value in a single instruction, returning the
    /// previous value.
    ///
    /// This method corresponds to the RISC-V `CSRRW rd, csr, rs1`
    /// instruction where `rs1 = in(reg) value_to_set` and `rd =
    /// out(reg) <return value>`.
    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    pub fn atomic_replace(&self, _value_to_set: u64) -> u64 {
        unimplemented!("RISC-V CSR {} Atomic Read/Write", V)
    }

    /// Atomically read a CSR and set bits specified in a bitmask
    ///
    /// This method corresponds to the RISC-V `CSRRS rd, csr, rs1`
    /// instruction where `rs1 = in(reg) bitmask` and `rd = out(reg)
    /// <return value>`.
    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    pub fn read_and_set_bits(&self, bitmask: u64) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrrs {rd}, {csr}, {rs1}",
                 rd = out(reg) r,
                 csr = const V,
                 rs1 = in(reg) bitmask);
        }
        r
    }

    /// Atomically read a CSR and set bits specified in a bitmask
    ///
    /// This method corresponds to the RISC-V `CSRRS rd, csr, rs1`
    /// instruction where `rs1 = in(reg) bitmask` and `rd = out(reg)
    /// <return value>`.
    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    pub fn read_and_set_bits(&self, bitmask: u64) -> u64 {
        unimplemented!(
            "RISC-V CSR {} Atomic Read and Set Bits, bitmask {:04x}",
            V,
            bitmask
        )
    }

    /// Atomically read a CSR and clear bits specified in a bitmask
    ///
    /// This method corresponds to the RISC-V `CSRRC rd, csr, rs1`
    /// instruction where `rs1 = in(reg) bitmask` and `rd = out(reg)
    /// <return value>`.
    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    pub fn read_and_clear_bits(&self, bitmask: u64) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrrc {rd}, {csr}, {rs1}",
                 rd = out(reg) r,
                 csr = const V,
                 rs1 = in(reg) bitmask);
        }
        r
    }

    /// Atomically read a CSR and clear bits specified in a bitmask
    ///
    /// This method corresponds to the RISC-V `CSRRC rd, csr, rs1`
    /// instruction where `rs1 = in(reg) bitmask` and `rd = out(reg)
    /// <return value>`.
    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    pub fn read_and_clear_bits(&self, bitmask: u64) -> u64 {
        unimplemented!(
            "RISC-V CSR {} Atomic Read and Clear Bits, bitmask {:04x}",
            V,
            bitmask
        )
    }

    /// Atomically read field and set all bits to 1
    ///
    /// This method corresponds to the RISC-V `CSRRS rd, csr, rs1`
    /// instruction, where `rs1` is the bitmask described by the
    /// [`Field`].
    ///
    /// The previous value of the field is returned.
    #[inline]
    pub fn read_and_set_field(&self, field: Field<u64, R>) -> u64 {
        field.read(self.read_and_set_bits(field.mask << field.shift))
    }

    /// Atomically read field and set all bits to 0
    ///
    /// This method corresponds to the RISC-V `CSRRC rd, csr, rs1`
    /// instruction, where `rs1` is the bitmask described by the
    /// [`Field`].
    ///
    /// The previous value of the field is returned.
    #[inline]
    pub fn read_and_clear_field(&self, field: Field<u64, R>) -> u64 {
        field.read(self.read_and_clear_bits(field.mask << field.shift))
    }
}

impl<R: RegisterLongName, const V: u16> Readable for ReadWriteRiscvCsr<R, V> {
    type T = u64;
    type R = R;

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn get(&self) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrr {rd}, {csr}", rd = out(reg) r, csr = const V);
        }
        r
    }

    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn get(&self) -> u64 {
        unimplemented!("reading RISC-V CSR {}", V)
    }
}

impl<R: RegisterLongName, const V: u16> Writeable for ReadWriteRiscvCsr<R, V> {
    type T = u64;
    type R = R;

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn set(&self, val_to_set: u64) {
        unsafe {
            asm!("csrw {csr}, {rs}", rs = in(reg) val_to_set, csr = const V);
        }
    }

    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn set(&self, _val_to_set: u64) {
        unimplemented!("writing RISC-V CSR {}", V)
    }
}
