// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sbi::Result as SbiResult;
use sbi::SbiMessage;

use core::arch::asm;

pub fn ecall_send(msg: &SbiMessage) -> SbiResult<u64> {
    let mut a0; // normally error code
    let mut a1; // normally return value
    unsafe {
        // Questionable safety, must trust hypervisor
        asm!("ecall", inlateout("a0") msg.a0()=>a0, inlateout("a1")msg.a1()=>a1,
                in("a2")msg.a2(), in("a3") msg.a3(),
                in("a4")msg.a4(), in("a5") msg.a5(),
                in("a6")msg.a6(), in("a7") msg.a7(), options(nomem, nostack));
    }

    msg.result(a0, a1)
}
