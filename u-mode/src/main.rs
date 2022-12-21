// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![no_main]

extern crate libuser;

#[no_mangle]
extern "C" fn task_main(_data: u64) {
    panic!("");
}
