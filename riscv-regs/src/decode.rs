// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

/// Instruction decoding for RISC-V 64.
use riscv_decode::{decode, instruction_length};

// Use the types from the riscv_decode crate.
pub use riscv_decode::{DecodingError, Instruction};

/// A RISC-V instruction that has been decoded. Only supports 2 or 4 bytes instructions for now.
#[derive(Clone, Copy, Debug)]
pub struct DecodedInstruction {
    instruction: Instruction,
    len: usize,
    raw: u32,
}

impl DecodedInstruction {
    /// Creates a new `DecodedInstruction` from raw instruction bytes.
    pub fn from_raw(raw: u32) -> Result<Self, DecodingError> {
        let len = instruction_length(raw as u16);
        let instruction = decode(raw)?;
        Ok(Self {
            instruction,
            len,
            raw,
        })
    }

    /// Returns the raw instruction bytes.
    pub fn raw(&self) -> u32 {
        self.raw
    }

    /// Returns the decoded instruction.
    pub fn instruction(&self) -> Instruction {
        self.instruction
    }

    /// Returns the length of the raw instruction in bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }
}
