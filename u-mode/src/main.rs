// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![no_main]

//! # Salus U-mode binary.
//!
//! This is Salus U-mode code. It is used to offload functionalities
//! of the hypervisor in user mode. There's a copy of this task in
//! each CPU.
//!
//! The task it's based on a request loop. This task can be reset at
//! any time by the hypervisor, so it shouldn't hold non-recoverable
//! state.

extern crate libuser;

use data_model::VolatileSlice;
use libuser::*;
use u_mode_api::{Error as UmodeApiError, UmodeRequest};

struct UmodeTask {
    vslice: VolatileSlice<'static>,
}

impl UmodeTask {
    // Copy memory from input to output.
    //
    // Arguments:
    //    out_addr: starting address of output
    //    in_addr: starting address of input
    //    len: length of input and output
    //
    // U-mode Shared Region: Not used.
    fn op_memcopy(&self, out_addr: u64, in_addr: u64, len: usize) -> Result<(), UmodeApiError> {
        // Safety: we trust the hypervisor to have mapped at `in_addr` `len` bytes for reading.
        let input = unsafe { &*core::ptr::slice_from_raw_parts(in_addr as *const u8, len) };
        // Safety: we trust the hypervisor to have mapped at `out_addr` `len` bytes valid
        // for reading and writing.
        let output = unsafe { &mut *core::ptr::slice_from_raw_parts_mut(out_addr as *mut u8, len) };
        output[0..len].copy_from_slice(&input[0..len]);
        Ok(())
    }

    // Run the main loop, receiving requests from the hypervisor and executing them.
    fn run_loop(&self) -> ! {
        let mut res = Ok(());
        loop {
            // Return result and wait for next operation.
            let req = hyp_nextop(res);
            res = match req {
                Ok(req) => match req {
                    UmodeRequest::Nop => Ok(()),
                    UmodeRequest::MemCopy {
                        out_addr,
                        in_addr,
                        len,
                    } => self.op_memcopy(out_addr, in_addr, len as usize),
                },
                Err(err) => Err(err),
            };
        }
    }
}

#[no_mangle]
extern "C" fn task_main(cpuid: u64, shared_addr: u64, shared_size: u64) -> ! {
    // Safety: we trust the hypervisor to have mapped an area of memory starting at `shared_addr`
    // valid for at least `shared_size` bytes.
    let vslice =
        unsafe { VolatileSlice::from_raw_parts(shared_addr as *mut u8, shared_size as usize) };
    let task = UmodeTask { vslice };
    println!(
        "umode/#{}: U-mode Shared Region: {:016x} - {} bytes",
        cpuid, shared_addr, shared_size
    );
    task.run_loop()
}
