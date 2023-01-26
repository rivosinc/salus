// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

mod hypcalls;

pub use crate::hypcalls::*;
use core::arch::global_asm;

global_asm!(include_str!("start.S"));

// Panic handler for U-mode programs.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic : {:?}", info);
    hyp_panic()
}

/// Writer for printing characters through the hypervisor via the putchar hypcall.
pub struct UserWriter {}

impl core::fmt::Write for UserWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for c in s.chars() {
            // Ignore errors for putchar.
            let _ = hyp_putchar(c);
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! print {
    ($($args:tt)*) => {
        {
            use core::fmt::Write;
            let mut writer = UserWriter {};
            write!(&mut writer, $($args)*).unwrap();
        }
    };
}

#[macro_export]
macro_rules! println {
    ($($args:tt)*) => {
        {
            use core::fmt::Write;
            let mut writer = UserWriter {};
            writeln!(&mut writer, $($args)*).unwrap();
        }
    };
}
