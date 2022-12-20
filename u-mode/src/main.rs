// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![no_main]

extern crate libuser;

use libuser::*;
use u_mode_api::UmodeOp;

#[no_mangle]
extern "C" fn task_main(cpuid: u64) -> ! {
    println!("umode/#{} initialized.", cpuid);
    // Initialization done.
    let mut res = Ok(());
    loop {
        // Return result and wait for next operation.
        let req = hyp_nextop(res);
        res = match req {
            Ok(req) => match req.op() {
                UmodeOp::Nop => Ok(()),
                UmodeOp::Hello => {
                    println!("----------------------------");
                    println!(" ___________________");
                    println!("< Hello from UMODE! >");
                    println!(" -------------------");
                    println!("        \\   ^__^");
                    println!("         \\  (oo)\\_______");
                    println!("            (__)\\       )\\/\\");
                    println!("                ||----w |");
                    println!("                ||     ||");
                    println!("----------------------------");
                    Ok(())
                }
            },
            Err(err) => Err(err),
        };
    }
}
