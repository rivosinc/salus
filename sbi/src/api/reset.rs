// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::ResetFunction;
use crate::{ecall_send, Result, SbiMessage};
use crate::{ResetReason, ResetType};

/// Resets the system.
pub fn reset(reset_type: ResetType, reason: ResetReason) -> Result<()> {
    let msg = SbiMessage::Reset(ResetFunction::Reset { reset_type, reason });
    // Safety: Reset terminates this VM.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Shuts down the system.
pub fn shutdown() -> Result<()> {
    let msg = SbiMessage::Reset(ResetFunction::shutdown());
    // Safety: This ecall doesn't touch memory and will never return.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}
