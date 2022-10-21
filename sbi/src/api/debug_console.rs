// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::{ecall_send, DebugConsoleFunction, Result, SbiMessage};

/// Prints the given string in a platfrom-dependent way.
pub fn console_puts(chars: &[u8]) -> Result<()> {
    let msg = SbiMessage::DebugConsole(DebugConsoleFunction::PutString {
        len: chars.len() as u64,
        addr: chars.as_ptr() as u64,
    });

    // Safety: The sbi implementation is trusted not to write memory when printing to the console.
    unsafe { ecall_send(&msg) }?;

    Ok(())
}
