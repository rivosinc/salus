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
    call primary_init
    // a0 contains SATP value
    csrw satp, a0
    // Flush TLBs.
    sfence.vma
    // Switch to VA-mapped stack.
    li  sp, {HYP_STACK_TOP}
    call primary_main
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
    // At start, A1 holds the top of the stack.
    mv   sp, a1
    call secondary_init
    // a0 contains SATP value
    csrw satp, a0
    // Flush TLBs.
    sfence.vma
    // Switch to VA-mapped stack.
    li   sp, {HYP_STACK_TOP}
    call secondary_main
wfi_loop:
    wfi
    j    wfi_loop
