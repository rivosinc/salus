// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! `ReadWriteRiscvCsr` type for RISC-V CSRs.

#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;
use core::marker::PhantomData;

use tock_registers::fields::Field;
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::RegisterLongName;

/// Trait defining the possible operations on a RISC-V CSR.
pub trait RiscvCsrInterface {
    type R: RegisterLongName;

    /// Reads the contents of a CSR.
    ///
    /// This method corresponds to the RISC-V `CSRR rd, csr`
    /// instruction where `rd = out(reg) <return value>`.
    fn get_value(&self) -> u64;

    /// Writes the value of a CSR.
    ///
    /// This method corresponds to the RISC-V `CSRW csr, rs`
    /// instruction where `rs = in(reg) val_to_set`.
    fn set_value(&self, val_to_set: u64);

    /// Atomically swap the contents of a CSR
    ///
    /// Reads the current value of a CSR and replaces it with the
    /// specified value in a single instruction, returning the
    /// previous value.
    ///
    /// This method corresponds to the RISC-V `CSRRW rd, csr, rs1`
    /// instruction where `rs1 = in(reg) value_to_set` and `rd =
    /// out(reg) <return value>`.
    fn atomic_replace(&self, val_to_set: u64) -> u64;

    /// Atomically read a CSR and set bits specified in a bitmask
    ///
    /// This method corresponds to the RISC-V `CSRRS rd, csr, rs1`
    /// instruction where `rs1 = in(reg) bitmask` and `rd = out(reg)
    /// <return value>`.
    fn read_and_set_bits(&self, bitmask: u64) -> u64;

    /// Atomically read a CSR and set bits specified in a bitmask
    ///
    /// This method corresponds to the RISC-V `CSRRS rd, csr, rs1`
    /// instruction where `rs1 = in(reg) bitmask` and `rd = out(reg)
    /// <return value>`.
    fn read_and_clear_bits(&self, bitmask: u64) -> u64;

    /// Atomically read field and set all bits to 1
    ///
    /// This method corresponds to the RISC-V `CSRRS rd, csr, rs1`
    /// instruction, where `rs1` is the bitmask described by the
    /// [`Field`].
    ///
    /// The previous value of the field is returned.
    fn read_and_set_field(&self, field: Field<u64, Self::R>) -> u64 {
        field.read(self.read_and_set_bits(field.mask << field.shift))
    }

    /// Atomically read field and set all bits to 0
    ///
    /// This method corresponds to the RISC-V `CSRRC rd, csr, rs1`
    /// instruction, where `rs1` is the bitmask described by the
    /// [`Field`].
    ///
    /// The previous value of the field is returned.
    fn read_and_clear_field(&self, field: Field<u64, Self::R>) -> u64 {
        field.read(self.read_and_clear_bits(field.mask << field.shift))
    }
}

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
}

impl<R: RegisterLongName, const V: u16> RiscvCsrInterface for ReadWriteRiscvCsr<R, V> {
    type R = R;

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn get_value(&self) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrr {rd}, {csr}", rd = out(reg) r, csr = const V);
        }
        r
    }

    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn get_value(&self) -> u64 {
        unimplemented!("reading RISC-V CSR {}", V)
    }

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn set_value(&self, val_to_set: u64) {
        unsafe {
            asm!("csrw {csr}, {rs}", rs = in(reg) val_to_set, csr = const V);
        }
    }

    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn set_value(&self, _val_to_set: u64) {
        unimplemented!("writing RISC-V CSR {}", V)
    }

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn atomic_replace(&self, val_to_set: u64) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrrw {rd}, {csr}, {rs1}",
                 rd = out(reg) r,
                 csr = const V,
                 rs1 = in(reg) val_to_set);
        }
        r
    }
    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn atomic_replace(&self, _value_to_set: u64) -> u64 {
        unimplemented!("RISC-V CSR {} Atomic Read/Write", V)
    }

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn read_and_set_bits(&self, bitmask: u64) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrrs {rd}, {csr}, {rs1}",
                 rd = out(reg) r,
                 csr = const V,
                 rs1 = in(reg) bitmask);
        }
        r
    }

    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn read_and_set_bits(&self, bitmask: u64) -> u64 {
        unimplemented!(
            "RISC-V CSR {} Atomic Read and Set Bits, bitmask {:04x}",
            V,
            bitmask
        )
    }

    #[cfg(all(target_arch = "riscv64", target_os = "none"))]
    #[inline]
    fn read_and_clear_bits(&self, bitmask: u64) -> u64 {
        let r: u64;
        unsafe {
            asm!("csrrc {rd}, {csr}, {rs1}",
                 rd = out(reg) r,
                 csr = const V,
                 rs1 = in(reg) bitmask);
        }
        r
    }

    // Mock implementations for tests on Travis-CI.
    #[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
    fn read_and_clear_bits(&self, bitmask: u64) -> u64 {
        unimplemented!(
            "RISC-V CSR {} Atomic Read and Clear Bits, bitmask {:04x}",
            V,
            bitmask
        )
    }
}

// The Readable and Writeable traits aren't object-safe so unfortunately we can't implement them
// for RiscvCsrInterface.
impl<R: RegisterLongName, const V: u16> Readable for ReadWriteRiscvCsr<R, V> {
    type T = u64;
    type R = R;

    fn get(&self) -> u64 {
        self.get_value()
    }
}

impl<R: RegisterLongName, const V: u16> Writeable for ReadWriteRiscvCsr<R, V> {
    type T = u64;
    type R = R;

    fn set(&self, val_to_set: u64) {
        self.set_value(val_to_set);
    }
}
