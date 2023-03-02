// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::arch::asm;
use u_mode_api::{CdiOp, CdiSel, Error as UmodeApiError, HypCall, TryIntoRegisters, UmodeRequest};

/// Send an ecall to the hypervisor.
///
/// # Safety
///
/// The caller must verify that any memory references contained in `regs` obey Rust's memory
/// safety rules. For example, any pointers to memory that will be modified in the handling of
/// the ecall must be uniquely owned. Similarly any pointers read by the ecall must not be
/// mutably borrowed.
unsafe fn ecall(regs: &mut [u64; 8]) {
    // TODO: at the moment we save and restore all registers. Once the interface is more stable this
    // can be modified to avoid save/restore a few registers.
    asm!("ecall",
         inlateout("a0") regs[0],
         inlateout("a1") regs[1],
         inlateout("a2") regs[2],
         inlateout("a3") regs[3],
         inlateout("a4") regs[4],
         inlateout("a5") regs[5],
         inlateout("a6") regs[6],
         inlateout("a7") regs[7], options(nostack));
}

/// Print a character.
pub fn hyp_putchar(c: char) {
    let mut regs = [0u64; 8];
    let hypc = HypCall::PutChar(c as u8);
    hypc.to_registers(&mut regs);
    // Safety: This ecall does not contain any memory reference.
    unsafe {
        ecall(&mut regs);
    }
    // No return.
}

/// Panic and exit immediately.
pub fn hyp_panic() -> ! {
    let mut regs = [0u64; 8];
    let hypc = HypCall::Panic;
    // Loop indefinitely to expose the `no_return` behaviour. In the extremely corrupt scenario
    // where this would actually return, at least we would keep looping into the hypervisor...
    loop {
        hypc.to_registers(&mut regs);
        // Safety: This ecall does not contain any memory reference.
        unsafe {
            ecall(&mut regs);
        }
    }
}

/// Complete current operation (sending a Result to the hypervisor)
/// and request the next operation to execute.
pub fn hyp_nextop(result: Result<u64, UmodeApiError>) -> Result<UmodeRequest, UmodeApiError> {
    let mut regs = [0u64; 8];
    let hypc = HypCall::NextOp(result);
    hypc.to_registers(&mut regs);
    // Safety: This ecall does not contain any memory reference.
    unsafe {
        ecall(&mut regs);
    }
    // In case there's an error on decoding the request, return it to caller.
    UmodeRequest::try_from_registers(&regs)
}

/// Get the ID of the CDI selected by `cdi_sel`.
pub fn hyp_cdi_id(cdi_sel: CdiSel, id: &mut [u8]) {
    let mut regs = [0u64; 8];
    let hypc = HypCall::Cdi {
        cdi_sel,
        cdi_op: CdiOp::Id {
            idout_addr: id.as_ptr() as u64,
            idout_len: id.len() as u64,
        },
    };
    hypc.to_registers(&mut regs);
    // Safety: we trust the hypervisor to write at `idout_addr` for `idout_len` bytes. This range is
    // entirely contained in `id`, of which we have a mutable reference.
    unsafe {
        ecall(&mut regs);
    }
}

/// Sign a message with the secret key of the CDI selected by `cdi_sel`.
pub fn hyp_cdi_sign(cdi_sel: CdiSel, msg: &[u8], signature: &mut [u8]) {
    let mut regs = [0u64; 8];
    let hypc = HypCall::Cdi {
        cdi_sel,
        cdi_op: CdiOp::Sign {
            msg_addr: msg.as_ptr() as u64,
            msg_len: msg.len() as u64,
            signout_addr: signature.as_ptr() as u64,
            signout_len: signature.len() as u64,
        },
    };
    hypc.to_registers(&mut regs);
    // Safety: we trust the hypervisor to write at `sign_addr` for `sign_len` bytes. This range is
    // entirely contained in `signature`, of which we have a mutable reference. We also trust the
    // hypervisor to read from `msg_addr` for `msg_len` bytes, which is entirely contained in `msg`.
    unsafe {
        ecall(&mut regs);
    }
}
