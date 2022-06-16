// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::ptr::NonNull;
use riscv_pages::SupervisorPhysAddr;
use spin::{Mutex, Once};

/// Provides basic print support in the bare-metal hypervisor environment.
#[macro_export]
macro_rules! print {
    ($($args:tt)*) => {
	unsafe {
	    use core::fmt::Write;
	    CONSOLE_DRIVER.get_mut().map(|c| write!(c, $($args)*));
	}
    };
}

/// Provides basic println support in the bare-metal hypervisor environment.
#[macro_export]
macro_rules! println {
    ($($args:tt)*) => {
	unsafe {
	    use core::fmt::Write;
	    CONSOLE_DRIVER.get_mut().map(|c| writeln!(c, $($args)*));
	}
    };
}

pub static mut CONSOLE_DRIVER: Once<UartDriver> = Once::new();

/// Driver for a standard UART.
pub struct UartDriver {
    base_address: Mutex<NonNull<u8>>,
}

impl UartDriver {
    /// Creates a new UART driver at the given base address.
    ///
    /// # Safety
    ///
    /// Only safe if the given base address points to a MMIO UART device and is not null.
    pub unsafe fn init(base_address: SupervisorPhysAddr) {
        let uart = UartDriver {
            base_address: Mutex::new(NonNull::new_unchecked(base_address.bits() as _)),
        };
        CONSOLE_DRIVER.call_once(|| uart);
    }

    /// Write an entire byte sequence to this UART.
    pub fn write_bytes(&self, bytes: &[u8]) {
        let base_address = self.base_address.lock();
        for &b in bytes {
            // Safety: the caller of ::new() had to guarantee that the given address belongs to an
            // actual UART and that nobody else is using it, thereby making this defined behavior.
            unsafe { core::ptr::write_volatile(base_address.as_ptr(), b) };
        }
    }
}

impl core::fmt::Write for UartDriver {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.write_bytes(s.as_bytes());
        core::fmt::Result::Ok(())
    }
}
