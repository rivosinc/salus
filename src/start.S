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
    csrw sstatus, zero
    csrw sie, zero

    // Clear the BSS
    la   a3, _bss_start
    la   a4, _bss_end
1:
    sd   zero, (a3)
    addi a3, a3, 8
    blt  a3, a4, 1b

    la   sp, _stack_end
    call kernel_init
    j    wfi_loop

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
    mv   sp, a1
    mv   tp, a1
    call secondary_init
wfi_loop:
    wfi
    j    wfi_loop
