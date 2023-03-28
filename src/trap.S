// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

.attribute arch, "rv64gcv"
.section .text.init

/*
 * Store the trap frame directly on the current stack.
 * `sscratch` contains the SP to be saved in the frame.
 * `sscratch` is not preserved.
 */
.macro save_frame
    addi sp, sp, -({tf_size})
    sd   ra, ({tf_ra})(sp)
    sd   gp, ({tf_gp})(sp)
    sd   tp, ({tf_tp})(sp)
    sd   s0, ({tf_s0})(sp)
    sd   s1, ({tf_s1})(sp)
    sd   a0, ({tf_a0})(sp)
    sd   a1, ({tf_a1})(sp)
    sd   a2, ({tf_a2})(sp)
    sd   a3, ({tf_a3})(sp)
    sd   a4, ({tf_a4})(sp)
    sd   a5, ({tf_a5})(sp)
    sd   a6, ({tf_a6})(sp)
    sd   a7, ({tf_a7})(sp)
    sd   s2, ({tf_s2})(sp)
    sd   s3, ({tf_s3})(sp)
    sd   s4, ({tf_s4})(sp)
    sd   s5, ({tf_s5})(sp)
    sd   s6, ({tf_s6})(sp)
    sd   s7, ({tf_s7})(sp)
    sd   s8, ({tf_s8})(sp)
    sd   s9, ({tf_s9})(sp)
    sd   s10, ({tf_s10})(sp)
    sd   s11, ({tf_s11})(sp)
    sd   t0, ({tf_t0})(sp)
    sd   t1, ({tf_t1})(sp)
    sd   t2, ({tf_t2})(sp)
    sd   t3, ({tf_t3})(sp)
    sd   t4, ({tf_t4})(sp)
    sd   t5, ({tf_t5})(sp)
    sd   t6, ({tf_t6})(sp)
    csrr t0, sscratch
    sd   t0, ({tf_sp})(sp)
    csrw sscratch, zero
    csrr t1, sstatus
    sd   t1, ({tf_sstatus})(sp)
    csrr t2, sepc
    sd   t2, ({tf_sepc})(sp)
.endm

// Early trap handler. Doesn't check for stack overflow as hypervisor mappings are not enabled yet.
.global _trap_init_entry
.balign 4
_trap_init_entry:
    /* Skip the buffer overflow check at init time. */
    j .L_same_stack_trap

/// Trap handler entry point.
.global _trap_entry
.balign 4
_trap_entry:
    /* Save T0 in sscratch */
    csrw sscratch, t0
    /* Check for stack overflow. If it is, switch to overflow stack and panic _quickly_. */
    li t0, {hyp_stack_bottom}
    blt sp, t0, _stack_overflow
    /* Restores T0 from sscratch */
    csrr t0, sscratch

.L_same_stack_trap:
    /* Save frame on current stack. */
    csrw sscratch, sp
    save_frame

    /* Now enter the rust trap handler. */
    move a0, sp
    call handle_trap

    /* Restore state and sret. */
    ld   t0, ({tf_sstatus})(sp)
    csrw sstatus, t0
    ld   t1, ({tf_sepc})(sp)
    csrw sepc, t1
    ld   ra, ({tf_ra})(sp)
    ld   gp, ({tf_gp})(sp)
    ld   tp, ({tf_tp})(sp)
    ld   s0, ({tf_s0})(sp)
    ld   s1, ({tf_s1})(sp)
    ld   a0, ({tf_a0})(sp)
    ld   a1, ({tf_a1})(sp)
    ld   a2, ({tf_a2})(sp)
    ld   a3, ({tf_a3})(sp)
    ld   a4, ({tf_a4})(sp)
    ld   a5, ({tf_a5})(sp)
    ld   a6, ({tf_a6})(sp)
    ld   a7, ({tf_a7})(sp)
    ld   s2, ({tf_s2})(sp)
    ld   s3, ({tf_s3})(sp)
    ld   s4, ({tf_s4})(sp)
    ld   s5, ({tf_s5})(sp)
    ld   s6, ({tf_s6})(sp)
    ld   s7, ({tf_s7})(sp)
    ld   s8, ({tf_s8})(sp)
    ld   s9, ({tf_s9})(sp)
    ld   s10, ({tf_s10})(sp)
    ld   s11, ({tf_s11})(sp)
    ld   t0, ({tf_t0})(sp)
    ld   t1, ({tf_t1})(sp)
    ld   t2, ({tf_t2})(sp)
    ld   t3, ({tf_t3})(sp)
    ld   t4, ({tf_t4})(sp)
    ld   t5, ({tf_t5})(sp)
    ld   t6, ({tf_t6})(sp)
    ld   sp, ({tf_sp})(sp)
    sret

_stack_overflow:
    /* Acquire the emergency stack on a spinlock. See "A spinlock with fences" in [1].
     *
     * [1]: RISC-V Memory Consistency Model Specification (DRAFT), Dec 1st 2017, 2.3.5.
     */
    la t1, overflow_stack_lock
    li t0, 1
1:
    amoswap.w t0, t0, (t1)
    fence r, rw
    bnez t0, 1b

    /* Restore T0 from sscratch. Saved at the beginning of the trap handler. */
    csrr t0, sscratch
    /* NOTE: Register T1 will be clobbered in this stack frame.*/

    csrw sscratch, sp
    la sp, _overflow_stack_end
    save_frame

    /* Now enter the rust trap handler. */
    move a0, sp
    call handle_stack_overflow
.Lwfi_loop:
    wfi
    j .Lwfi_loop
