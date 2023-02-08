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

use data_model::{VolatileMemory, VolatileSlice};
use libuser::*;
use u_mode_api::{Error as UmodeApiError, UmodeRequest};

struct UmodeTask {
    vslice: VolatileSlice<'static>,
}

impl UmodeTask {
    // Run the main loop, receiving requests from the hypervisor and executing them.
    fn run_loop(&self) -> ! {
        let mut res = Ok(());
        loop {
            // Return result and wait for next operation.
            let req = hyp_nextop(res);
            res = match req {
                Ok(req) => match req {
                    UmodeRequest::Nop => Ok(()),
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
