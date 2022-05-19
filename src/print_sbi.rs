// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sbi::SbiMessage;

use crate::ecall_send;
use core::arch::asm;

#[macro_export]
macro_rules! print {
	($($args:tt)*) => {
		unsafe {
			use core::fmt::Write;
			CONSOLE_DRIVER.as_mut().map(|c| write!(c, $($args)*));
		}
	};
}

#[macro_export]
macro_rules! println {
	($($args:tt)*) => {
		unsafe {
			use core::fmt::Write;
			CONSOLE_DRIVER.as_mut().map(|c| writeln!(c, $($args)*));
		}
	};
}

macro_rules! _unreachable {
    () => {{
        println!("reached unreachable statement @ {}:{}", file!(), line!());
        abort();
    }};
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    abort();
}

/// Silently ends execution of this thread forever.
pub fn abort() -> ! {
    loop {
        // Safety: the WFI op has defined behavior and no side effects other then stoping execution
        // for some time.
        unsafe {
            asm!("wfi", options(nomem, nostack));
        }
    }
}

pub static mut CONSOLE_DRIVER: Option<SbiConsoleDriver> = None;

/// Driver for a standard UART.
pub struct SbiConsoleDriver(u8);

impl SbiConsoleDriver {
    /// Creates a new UART driver at the given base address.
    ///
    /// # Safety
    ///
    /// Only safe if the given base address points to a MMIO UART device and is not null.
    pub unsafe fn new() -> Self {
        SbiConsoleDriver(0)
    }

    /// Sets this instance as the global console singleton.
    ///
    /// # Safety
    ///
    /// Only safe to use during kernel initialization before any code that might use the console.
    pub unsafe fn use_as_console(self) {
        CONSOLE_DRIVER = Some(self);
        // In case any interrupts affect us, insert a compiler fence.
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }

    #[inline(always)]
    pub fn write_byte(&self, b: u8) {
        // Uses an SBI message sent via ecall to write byte
        let message = SbiMessage::PutChar(b as u64);
        ecall_send(&message).unwrap();
    }

    /// Write an entire byte sequence to this UART.
    pub fn write_bytes(&self, bytes: &[u8]) {
        for &b in bytes {
            self.write_byte(b)
        }
    }
}

impl core::fmt::Write for SbiConsoleDriver {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.write_bytes(s.as_bytes());
        core::fmt::Result::Ok(())
    }
}

/// Writes a string without formatting to the console.
#[allow(dead_code)] // Fixes spurious clippy warning
pub fn console_write_str(s: &str) {
    // Safety: Depending on the safe usage of `use_as_console`, CONSOLE_DRIVER will only be mutated
    // once before any users of the console, making the global safe to access.
    unsafe {
        if let Some(c) = CONSOLE_DRIVER.as_ref() {
            c.write_bytes(s.as_bytes())
        };
    }
}
