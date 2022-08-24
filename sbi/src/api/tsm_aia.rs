// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::TeeAiaFunction::*;
use crate::TvmAiaParams;
use crate::{ecall_send, Result, SbiMessage};

/// Configures AIA virtualization for `tvm_id` with the settings in `tvm_aia_params`.
pub fn tvm_aia_init(tvm_id: u64, tvm_aia_params: TvmAiaParams) -> Result<()> {
    let msg = SbiMessage::TeeAia(TvmAiaInit {
        tvm_id,
        params_addr: (&tvm_aia_params as *const TvmAiaParams) as u64,
        len: core::mem::size_of::<TvmAiaParams>() as u64,
    });
    // Safety: `TvmConfigureAia` will only read up to `len` bytes of the `TvmAiaParams` structure
    // we passed in.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Sets the guest physical address of the specified vCPU's virtualized IMSIC to `imsic_addr`.
pub fn set_vcpu_imsic_addr(tvm_id: u64, vcpu_id: u64, imsic_addr: u64) -> Result<()> {
    let msg = SbiMessage::TeeAia(TvmCpuSetImsicAddr {
        tvm_id,
        vcpu_id,
        imsic_addr,
    });
    // Safety: `TvmCpuSetImsicAddr` doesn't touch host memory in any way.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Converts the guest interrupt file at `imsic_addr` for use with a TVM.
///
/// # Safety
///
/// The caller must not access the guest interrupt file again until it has been reclaimed.
pub unsafe fn convert_imsic(imsic_addr: u64) -> Result<()> {
    let msg = SbiMessage::TeeAia(TsmConvertImsic { imsic_addr });
    // The caller must guarantee that they won't access the page at `imsic_addr`.
    ecall_send(&msg)?;
    Ok(())
}

/// Reclaims the guest interrupt file at `imsic_addr` that was previously converted with
/// `convert_imsic()`.
pub fn reclaim_imsic(imsic_addr: u64) -> Result<()> {
    let msg = SbiMessage::TeeAia(TsmReclaimImsic { imsic_addr });
    // Safety: The referenced page is made available again, which is safe since it hasn't been
    // accessible since conversion.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}
