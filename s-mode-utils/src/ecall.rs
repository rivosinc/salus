// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sbi_rs::Result as SbiResult;
use sbi_rs::SbiMessage;

#[cfg(all(target_arch = "riscv64", target_os = "none"))]
use core::arch::asm;

/// Send an ecall to the firmware or hypervisor.
///
/// # Safety
///
/// The caller must verify that any memory references contained in `msg` obey Rust's memory
/// safety rules. For example, any pointers to memory that will be modified in the handling of
/// the ecall must be uniquely owned. Similarly any pointers read by the ecall must not be
/// mutably borrowed.
///
/// In addition the caller is placing trust in the firmware or hypervisor to maintain the promises
/// of the interface w.r.t. reading and writing only within the provided bounds.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub unsafe fn ecall_send(msg: &SbiMessage) -> SbiResult<u64> {
    // normally error code
    let mut a0;
    // normally return value
    let mut a1;
    asm!("ecall", inlateout("a0") msg.a0()=>a0, inlateout("a1")msg.a1()=>a1,
                in("a2")msg.a2(), in("a3") msg.a3(),
                in("a4")msg.a4(), in("a5") msg.a5(),
                in("a6")msg.a6(), in("a7") msg.a7(), options(nostack));

    msg.result(a0, a1)
}

// Make ecalls panic in tests as there isn't an SBI interface beneath the tests.
#[cfg(not(any(target_arch = "riscv64", target_os = "none")))]
pub unsafe fn ecall_send(_msg: &SbiMessage) -> SbiResult<u64> {
    panic!("Test attempted ecall");
}
