// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// General purpose registers for RISC-V 64.

/// Array of rv64 general purpose registers with accessors/setters.
/// Used to save state of guest VMs when they aren't running.
/// `repr(C)` because it is referenced from assembly.
#[derive(Default)]
#[repr(C)]
pub struct GeneralPurposeRegisters([u64; 32]);

/// Index of risc-v general purpose registers in `GeneralPurposeRegisters`.
pub enum GprIndex {
    RA = 0,
    GP,
    TP,
    S0,
    S1,
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    SP,
}

impl GeneralPurposeRegisters {
    /// Returns the value of the given register.
    pub fn reg(&self, reg_index: GprIndex) -> u64 {
        self.0[reg_index as usize]
    }

    /// Sets the value of the given register.
    pub fn set_reg(&mut self, reg_index: GprIndex, val: u64) {
        self.0[reg_index as usize] = val;
    }

    /// Returns the argument registers.
    /// This is avoids many calls when an SBI handler needs all of the argmuent regs.
    pub fn a_regs(&self) -> &[u64] {
        &self.0[GprIndex::A0 as usize..=GprIndex::A7 as usize]
    }
}
