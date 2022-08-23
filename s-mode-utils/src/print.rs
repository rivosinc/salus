// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use spin::Mutex;

pub use crate::{print, println};

/// Interface for a console driver.
pub trait ConsoleWriter: Sync {
    /// Writes `bytes` to the console.
    fn write_bytes(&self, bytes: &[u8]);
}

/// Represents the system console, used by the `print!` and `println!` macros.
pub struct Console {
    writer: Option<&'static dyn ConsoleWriter>,
}

impl Console {
    const fn new() -> Self {
        Self { writer: None }
    }

    /// Sets the writer for the system console.
    pub fn set_writer(writer: &'static dyn ConsoleWriter) {
        CONSOLE.lock().writer = Some(writer);
    }
}

/// The `Console` singleton.
pub static CONSOLE: Mutex<Console> = Mutex::new(Console::new());

/// `print` macro based on writing to `CONSOLE`.
#[macro_export]
macro_rules! print {
    ($($args:tt)*) => {
        {
            use core::fmt::Write;
            write!(CONSOLE.lock(), $($args)*).unwrap();
        }
    };
}

/// `println` macro based on writing to `CONSOLE`.
#[macro_export]
macro_rules! println {
    ($($args:tt)*) => {
        {
            use core::fmt::Write;
            writeln!(CONSOLE.lock(), $($args)*).unwrap();
        }
    };
}

impl core::fmt::Write for Console {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        if let Some(w) = self.writer {
            w.write_bytes(s.as_bytes());
        }
        Ok(())
    }
}
