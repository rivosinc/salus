// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]

use core::arch::{asm, global_asm};

global_asm!(include_str!("start.S"));

// Loop making ecalls as the kernel will kill the task on an ecall (the only syscall supported is
// `exit`).
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // Safe to make an ecall that won't return.
    unsafe {
        loop {
            asm!("ecall");
        }
    }
}
