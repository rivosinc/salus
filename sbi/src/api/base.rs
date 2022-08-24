// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
use crate::ecall_send;
use crate::{BaseFunction::*, Error as SbiError, Result, SbiMessage};

/// Returns the implemented version of the SBI standard.
pub fn get_specification_version() -> Result<u64> {
    let msg = SbiMessage::Base(GetSpecificationVersion);
    // Safety: This ecall doesn't touch memory
    unsafe { ecall_send(&msg) }
}

/// Returns the ID of the SBI implementation.
pub fn get_implementation_id() -> Result<u64> {
    let msg = SbiMessage::Base(GetImplementationID);
    // Safety: This ecall doesn't touch memory
    unsafe { ecall_send(&msg) }
}

/// Returns the version of this SBI implementation.
pub fn get_implementation_version() -> Result<u64> {
    let msg = SbiMessage::Base(GetImplementationVersion);
    // Safety: This ecall doesn't touch memory
    unsafe { ecall_send(&msg) }
}

/// Checks if the given SBI extension is supported.
/// A return value of 0 (which denotes not implemented per the documentation)
/// is automatically converted into NotSupported error.
pub fn probe_sbi_extension(sbi_extension_id: u64) -> Result<u64> {
    let msg = SbiMessage::Base(ProbeSbiExtension(sbi_extension_id));
    // Safety: This ecall doesn't touch memory
    let result = unsafe { ecall_send(&msg) }?;
    if result == 0 {
        Err(SbiError::NotSupported)
    } else {
        Ok(result)
    }
}

/// Returns the vendor that produced this machine(`mvendorid`).
pub fn get_machine_vendor_id() -> Result<u64> {
    let msg = SbiMessage::Base(GetMachineVendorID);
    // Safety: This ecall doesn't touch memory
    unsafe { ecall_send(&msg) }
}

/// Returns the architecture implementation ID of this machine(`marchid`).
pub fn get_machine_architecture_id() -> Result<u64> {
    let msg = SbiMessage::Base(GetMachineArchitectureID);
    // Safety: This ecall doesn't touch memory
    unsafe { ecall_send(&msg) }
}

/// Returns the implementation ID of this machine(`mimpid`).
pub fn get_machine_implementation_id() -> Result<u64> {
    let msg = SbiMessage::Base(GetMachineImplementationID);
    // Safety: This ecall doesn't touch memory
    unsafe { ecall_send(&msg) }
}
