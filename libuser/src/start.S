// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

.option norvc

.section .text.start

// The entry point for a task.
.global _start
_start:

.option push
.option norelax
    la gp, __global_pointer$
.option pop
    la sp, _stack_end

    // a0 contains cpu id
    call task_main

    // ecall to panic
    li a7, 0
    ecall

