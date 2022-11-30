// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

.option norvc

.section .text.start

// The entry point for the boot CPU.
.global _start
_start:

.option push
.option norelax
    la gp, __global_pointer$
.option pop
    la sp, _stack_end
    csrw sstatus, zero
    csrw sie, zero

    la t1, kernel_init
    la ra, 1f
    jr t1

    j 1f
1:
    wfi
    j 1b

// The entry point for secondary CPUs.
.global _secondary_start
_secondary_start:

.option push
.option norelax
    la gp, __global_pointer$
.option pop
    csrw sstatus, zero
    csrw sie, zero
    // TP holds the address of our PerCpu struct, which is also the top of our stack.
    mv sp, a1
    mv tp, a1

    la t1, secondary_init
    la ra, 1b
    jr t1
