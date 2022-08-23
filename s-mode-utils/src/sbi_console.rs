// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sbi::SbiMessage;

use crate::ecall::ecall_send;
use crate::print::{Console, ConsoleWriter};

/// Driver for an SBI based console.
pub struct SbiConsole {}

static SBI_CONSOLE: SbiConsole = SbiConsole {};

impl SbiConsole {
    /// Sets the SBI console as the system console.
    pub fn set_as_console() {
        Console::set_writer(&SBI_CONSOLE);
    }
}

impl ConsoleWriter for SbiConsole {
    /// Write an entire byte sequence to the SBI console.
    fn write_bytes(&self, bytes: &[u8]) {
        for &b in bytes {
            let message = SbiMessage::PutChar(b as u64);
            // Safety: message doesn't contain pointers and the ecall doesn't touch memory so this
            // is trivially safe.
            unsafe {
                ecall_send(&message).unwrap();
            }
        }
    }
}
