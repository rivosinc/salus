// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::asm;

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
