// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


/// Enter the guest given in `GuestInfo` from `a0`.
/// Currently trashes sscratch.
.section .text.init
.global _run_guest
_run_guest:
    /* Save hypervisor state */

    /* store the GuestInfo pointer in sscratch while running the guest. */
    csrrw x0, sscratch, a0

    /* Save Host GPRs (except T0-T6 and a0, which is GuestInfo and stashed in sscratch) */
    addi sp, sp, -176
    sd   ra, (0)(sp)
    sd   gp, (8)(sp)
    sd   tp, (16)(sp)
    sd   s0, (24)(sp)
    sd   s1, (32)(sp)
    sd   a1, (40)(sp)
    sd   a2, (48)(sp)
    sd   a3, (56)(sp)
    sd   a4, (64)(sp)
    sd   a5, (72)(sp)
    sd   a6, (80)(sp)
    sd   a7, (88)(sp)
    sd   s2, (96)(sp)
    sd   s3, (104)(sp)
    sd   s4, (112)(sp)
    sd   s5, (120)(sp)
    sd   s6, (128)(sp)
    sd   s7, (136)(sp)
    sd   s8, (144)(sp)
    sd   s9, (152)(sp)
    sd   s10, (160)(sp)
    sd   s11, (168)(sp)

    // stack pointer gets stashed at the beginning of GuestInfo
    sd   sp, (0)(a0)

    // restore guest CSRs
    ld   t1, (8)(a0)
    csrrw t1, sepc, t1
    sd   t1, (8)(a0)

    ld   t1, (16)(a0)
    csrrw t1, sie, t1
    sd   t1, (16)(a0)

    ld   t1, (24)(a0)
    csrrw t1, scause, t1
    sd   t1, (24)(a0)

    // Set stvec so that hypervisor resumes after the sret when the guest exits.
    la      t1, _guest_exit
    csrrw   t1, stvec, t1
    sd      t1, (32)(a0)

    ld      t1, (40)(a0)
    csrrw   t1, 0x680, t1 // hgatp
    sd      t1, (40)(a0)
    ld      t1, (48)(a0)

    csrrw   t1, 0x602, t1 // hedeleg
    sd      t1, (48)(a0)

    ld      t1, (56)(a0)
    csrrw   t1, 0x603, t1 // hideleg
    sd      t1, (56)(a0)

    ld      t1, (64)(a0)
    csrrw   t1, 0x600, t1 // hstatus
    sd      t1, (64)(a0)

    ld      t1, (72)(a0)
    csrrw   t1, 0x606, t1 // hcounteren
    sd      t1, (72)(a0)

    ld      t1, (80)(a0)
    csrrw   t1, sstatus, t1
    sd      t1, (80)(a0)

    ld      t1, (88)(a0)
    csrrw   t1, stval, t1
    sd      t1, (88)(a0)

    ld      t1, (96)(a0)
    csrrw   t1, 0x643, t1 // htval
    sd      t1, (96)(a0)

    /* restore the gprs from this GuestInfo */
    ld   ra, (104+0)(a0)
    ld   gp, (104+8)(a0)
    ld   tp, (104+16)(a0)
    ld   s0, (104+24)(a0)
    ld   s1, (104+32)(a0)
    ld   a1, (104+48)(a0)
    ld   a2, (104+56)(a0)
    ld   a3, (104+64)(a0)
    ld   a4, (104+72)(a0)
    ld   a5, (104+80)(a0)
    ld   a6, (104+88)(a0)
    ld   a7, (104+96)(a0)
    ld   s2, (104+104)(a0)
    ld   s3, (104+112)(a0)
    ld   s4, (104+120)(a0)
    ld   s5, (104+128)(a0)
    ld   s6, (104+136)(a0)
    ld   s7, (104+144)(a0)
    ld   s8, (104+152)(a0)
    ld   s9, (104+160)(a0)
    ld   s10, (104+168)(a0)
    ld   s11, (104+176)(a0)
    ld   t0, (104+184)(a0)
    ld   t1, (104+192)(a0)
    ld   t2, (104+200)(a0)
    ld   t3, (104+208)(a0)
    ld   t4, (104+216)(a0)
    ld   t5, (104+224)(a0)
    ld   t6, (104+232)(a0)
    ld   sp, (104+240)(a0)
    ld   a0, (104+40)(a0)

    sret

.align 2
_guest_exit:
    /* pull the stashed Guest Info */
    csrrw a0, sscratch, a0
    /* push the registers to this GuestInfo */
    sd   ra, (104+0)(a0)
    sd   gp, (104+8)(a0)
    sd   tp, (104+16)(a0)
    sd   s0, (104+24)(a0)
    sd   s1, (104+32)(a0)
    // Guest a0 is currently in sscratch and will be saved to offset 40 below.
    sd   a1, (104+48)(a0)
    sd   a2, (104+56)(a0)
    sd   a3, (104+64)(a0)
    sd   a4, (104+72)(a0)
    sd   a5, (104+80)(a0)
    sd   a6, (104+88)(a0)
    sd   a7, (104+96)(a0)
    sd   s2, (104+104)(a0)
    sd   s3, (104+112)(a0)
    sd   s4, (104+120)(a0)
    sd   s5, (104+128)(a0)
    sd   s6, (104+136)(a0)
    sd   s7, (104+144)(a0)
    sd   s8, (104+152)(a0)
    sd   s9, (104+160)(a0)
    sd   s10, (104+168)(a0)
    sd   s11, (104+176)(a0)
    sd   t0, (104+184)(a0)
    sd   t1, (104+192)(a0)
    sd   t2, (104+200)(a0)
    sd   t3, (104+208)(a0)
    sd   t4, (104+216)(a0)
    sd   t5, (104+224)(a0)
    sd   t6, (104+232)(a0)
    sd   sp, (104+240)(a0)

    // Save Guest a0 after recovering from sscratch.
    csrr t0, sscratch
    sd   t0, (104+40)(a0)

    // Save guest CSRs - restore Host copies
    ld   t1, (8)(a0)
    csrrw t1, sepc, t1
    sd   t1, (8)(a0)

    ld   t1, (16)(a0)
    csrrw t1, sie, t1
    sd   t1, (16)(a0)

    ld   t1, (24)(a0)
    csrrw t1, scause, t1
    sd   t1, (24)(a0)

    ld      t1, (32)(a0)
    csrrw   x0, stvec, t1

    ld      t1, (40)(a0)
    csrrw   t1, 0x680, t1 // hgatp
    sd      t1, (40)(a0)

    ld      t1, (48)(a0)
    csrrw   t1, 0x602, t1 // hedeleg
    sd      t1, (48)(a0)

    ld      t1, (56)(a0)
    csrrw   t1, 0x603, t1 // hideleg
    sd      t1, (56)(a0)

    ld      t1, (64)(a0)
    csrrw   t1, 0x600, t1 // hstatus
    sd      t1, (64)(a0)

    ld      t1, (72)(a0)
    csrrw   t1, 0x606, t1 // hcounteren
    sd      t1, (72)(a0)

    ld      t1, (80)(a0)
    csrrw   t1, sstatus, t1
    sd      t1, (80)(a0)

    ld      t1, (88)(a0)
    csrrw   t1, stval, t1
    sd      t1, (88)(a0)

    ld      t1, (96)(a0)
    csrrw   t1, 0x643, t1 // htval
    sd      t1, (96)(a0)

    // Get the host stack pointer back.
    ld   sp, (0)(a0)

    /* restore hypervisor regs */
    ld   ra, (0)(sp)
    ld   gp, (8)(sp)
    ld   tp, (16)(sp)
    ld   s0, (24)(sp)
    ld   s1, (32)(sp)
    ld   a1, (40)(sp)
    ld   a2, (48)(sp)
    ld   a3, (56)(sp)
    ld   a4, (64)(sp)
    ld   a5, (72)(sp)
    ld   a6, (80)(sp)
    ld   a7, (88)(sp)
    ld   s2, (96)(sp)
    ld   s3, (104)(sp)
    ld   s4, (112)(sp)
    ld   s5, (120)(sp)
    ld   s6, (128)(sp)
    ld   s7, (136)(sp)
    ld   s8, (144)(sp)
    ld   s9, (152)(sp)
    ld   s10, (160)(sp)
    ld   s11, (168)(sp)
    addi sp, sp, 176

    /* Now a0 has the GuestInfo Pointer */

    /* return */
    ret
