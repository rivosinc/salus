// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sbi::api::debug_console::console_puts;
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
        // Ignore errors as there isn't currently a way to report them if the console doesn't work.
        let _ = console_puts(bytes);
    }
}

/// Driver for the legacy SBI console from v0.1 of the SBI spec.
pub struct SbiConsoleV01 {}

static SBI_CONSOLE_V01: SbiConsoleV01 = SbiConsoleV01 {};

impl SbiConsoleV01 {
    /// Sets the SBI legacy console as the system console.
    pub fn set_as_console() {
        Console::set_writer(&SBI_CONSOLE_V01);
    }
}

impl ConsoleWriter for SbiConsoleV01 {
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
