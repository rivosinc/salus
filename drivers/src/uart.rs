// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::ptr::NonNull;
use riscv_pages::SupervisorPhysAddr;
use s_mode_utils::print::*;
use spin::{Mutex, Once};

static UART_DRIVER: Once<UartDriver> = Once::new();

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
        UART_DRIVER.call_once(|| uart);
        Console::set_writer(UART_DRIVER.get().unwrap());
    }
}

impl ConsoleWriter for UartDriver {
    /// Write an entire byte sequence to this UART.
    fn write_bytes(&self, bytes: &[u8]) {
        let base_address = self.base_address.lock();
        for &b in bytes {
            // Safety: the caller of ::new() had to guarantee that the given address belongs to an
            // actual UART and that nobody else is using it, thereby making this defined behavior.
            unsafe { core::ptr::write_volatile(base_address.as_ptr(), b) };
        }
    }
}

// Safety: Access to the pointer to the UART's registers is guarded by a Mutex and the UartDriver
// API guarantees that it is used safely.
unsafe impl Send for UartDriver {}
unsafe impl Sync for UartDriver {}
