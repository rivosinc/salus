// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use sync::Mutex;

pub use crate::{print, println};

/// Interface for a console driver.
pub trait ConsoleDriver: Sync {
    /// Writes `bytes` to the console.
    fn write_bytes(&self, bytes: &[u8]);

    /// Read from the console into `bytes`. Returns the number of bytes read.
    /// This default implementation doesn't produce any input.
    fn read_bytes(&self, _bytes: &mut [u8]) -> usize {
        0
    }
}

/// Represents the system console, used by the `print!` and `println!` macros.
pub struct Console {
    driver: Option<&'static dyn ConsoleDriver>,
}

impl Console {
    const fn new() -> Self {
        Self { driver: None }
    }

    /// Reads characters from the console into `buf`. Returns the number of bytes read, which may
    /// be less than the buffer size if there is not enough input.
    pub fn read(buf: &mut [u8]) -> usize {
        CONSOLE
            .lock()
            .driver
            .map(|d| d.read_bytes(buf))
            .unwrap_or(0)
    }

    /// Sets the driver for the system console.
    pub fn set_driver(driver: &'static dyn ConsoleDriver) {
        CONSOLE.lock().driver = Some(driver);
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
        if let Some(w) = self.driver {
            w.write_bytes(s.as_bytes());
        }
        Ok(())
    }
}
