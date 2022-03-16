// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::ptr::NonNull;

use crate::abort::abort;

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

pub static mut CONSOLE_DRIVER: Option<UartDriver> = None;

/// Driver for a standard UART.
pub struct UartDriver {
    base_address: NonNull<u8>,
}

impl UartDriver {
    /// Creates a new UART driver at the given base address.
    ///
    /// # Safety
    ///
    /// Only safe if the given base address points to a MMIO UART device and is not null.
    pub unsafe fn new(base_address: usize) -> Self {
        UartDriver {
            base_address: NonNull::new_unchecked(base_address as _),
        }
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
        // Safety: the caller of ::new() had to guarantee that the given address belongs to an
        // actual UART and that nobody else is using it, thereby making this defined behavior.
        unsafe { core::ptr::write_volatile(self.base_address.as_ptr(), b) }
    }

    /// Write an entire byte sequence to this UART.
    pub fn write_bytes(&self, bytes: &[u8]) {
        for &b in bytes {
            self.write_byte(b)
        }
    }
}

impl core::fmt::Write for UartDriver {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.write_bytes(s.as_bytes());
        core::fmt::Result::Ok(())
    }
}
