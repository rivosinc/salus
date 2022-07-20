// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::StateFunction::*;
use crate::{ecall_send, Result, SbiMessage};

/// Starts the given cpu executing at `start_addr` with `opaque` in register a1.
///
/// # Safety
///
/// start_addr must point to code that can be safely executed.
/// opaque, if a pointer, must point to data that is safe to access from the newly running context.
pub unsafe fn hart_start(hart_id: u64, start_addr: u64, opaque: u64) -> Result<()> {
    let msg = SbiMessage::HartState(HartStart {
        hart_id,
        start_addr,
        opaque,
    });
    // Safety: Passes one pointer to SBI, that pointer is guaranteed by the linker to be the
    // code to start secondary CPUs.
    ecall_send(&msg)?;
    Ok(())
}
