// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;

#[cfg(all(target_arch = "riscv64", target_os = "none"))]
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

#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
/// Silently ends execution of this thread forever.
pub fn abort() -> ! {
    panic!("");
}
