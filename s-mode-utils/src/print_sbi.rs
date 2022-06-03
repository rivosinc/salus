// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sbi::SbiMessage;

use crate::ecall::ecall_send;
pub use crate::println;

#[macro_export]
macro_rules! print {
    ($($args:tt)*) => {
        {
            use core::fmt::Write;
            write!(SbiConsoleDriver{}, $($args)*).unwrap();
        }
    };
}

#[macro_export]
macro_rules! println {
    ($($args:tt)*) => {
        {
            use core::fmt::Write;
            writeln!(SbiConsoleDriver{}, $($args)*).unwrap();
        }
    };
}

/// Write an entire byte sequence to the SBI console.
pub fn console_write_bytes(bytes: &[u8]) {
    for &b in bytes {
        let message = SbiMessage::PutChar(b as u64);
        // Safety: message doesn't contain pointers and the ecall doesn't touch memory so this is
        // trivially safe.
        unsafe {
            ecall_send(&message).unwrap();
        }
    }
}

/// Driver for an SBI based console.
pub struct SbiConsoleDriver {}

impl core::fmt::Write for SbiConsoleDriver {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        console_write_bytes(s.as_bytes());
        core::fmt::Result::Ok(())
    }
}
