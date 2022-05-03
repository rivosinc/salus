// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


/// Enter the guest given in `GuestInfo` from `a0`.
///
/// TODO: Can we get asm offsets for rust structs?
.section .text.init
.global _run_guest
_run_guest:
    /* Save hypervisor state */

    /* Save Host GPRs (except T0-T6 and a0, which is GuestInfo and stashed in sscratch) */
    sd   ra, (0)(a0)
    sd   gp, (8)(a0)
    sd   tp, (16)(a0)
    sd   s0, (24)(a0)
    sd   s1, (32)(a0)
    sd   a1, (48)(a0)
    sd   a2, (56)(a0)
    sd   a3, (64)(a0)
    sd   a4, (72)(a0)
    sd   a5, (80)(a0)
    sd   a6, (88)(a0)
    sd   a7, (96)(a0)
    sd   s2, (104)(a0)
    sd   s3, (112)(a0)
    sd   s4, (120)(a0)
    sd   s5, (128)(a0)
    sd   s6, (136)(a0)
    sd   s7, (144)(a0)
    sd   s8, (152)(a0)
    sd   s9, (160)(a0)
    sd   s10, (168)(a0)
    sd   s11, (176)(a0)
    sd   sp, (240)(a0)

    /* Swap in guest CSRs. */
    ld    t1, (296+256)(a0)
    csrrw t1, sstatus, t1
    sd    t1, (256)(a0)

    ld    t1, (296+264)(a0)
    csrrw t1, hstatus, t1
    sd    t1, (264)(a0)

    ld    t1, (296+272)(a0)
    csrrw t1, scounteren, t1
    sd    t1, (272)(a0)

    ld    t1, (296+280)(a0)
    csrw  sepc, t1

    /* Set stvec so that hypervisor resumes after the sret when the guest exits. */
    la    t1, _guest_exit
    csrrw t1, stvec, t1
    sd    t1, (280)(a0)

    /* Save sscratch and replace with pointer to GuestInfo. */
    csrrw t1, sscratch, a0
    sd    t1, (288)(a0)

    /* Restore the gprs from this GuestInfo */
    ld   ra, (296+0)(a0)
    ld   gp, (296+8)(a0)
    ld   tp, (296+16)(a0)
    ld   s0, (296+24)(a0)
    ld   s1, (296+32)(a0)
    ld   a1, (296+48)(a0)
    ld   a2, (296+56)(a0)
    ld   a3, (296+64)(a0)
    ld   a4, (296+72)(a0)
    ld   a5, (296+80)(a0)
    ld   a6, (296+88)(a0)
    ld   a7, (296+96)(a0)
    ld   s2, (296+104)(a0)
    ld   s3, (296+112)(a0)
    ld   s4, (296+120)(a0)
    ld   s5, (296+128)(a0)
    ld   s6, (296+136)(a0)
    ld   s7, (296+144)(a0)
    ld   s8, (296+152)(a0)
    ld   s9, (296+160)(a0)
    ld   s10, (296+168)(a0)
    ld   s11, (296+176)(a0)
    ld   t0, (296+184)(a0)
    ld   t1, (296+192)(a0)
    ld   t2, (296+200)(a0)
    ld   t3, (296+208)(a0)
    ld   t4, (296+216)(a0)
    ld   t5, (296+224)(a0)
    ld   t6, (296+232)(a0)
    ld   sp, (296+240)(a0)
    ld   a0, (296+40)(a0)

    sret

.align 2
_guest_exit:
    /* Pull GuestInfo out of sscratch, swapping with guest's a0 */
    csrrw a0, sscratch, a0

    /* Save guest GPRs. */
    sd   ra, (296+0)(a0)
    sd   gp, (296+8)(a0)
    sd   tp, (296+16)(a0)
    sd   s0, (296+24)(a0)
    sd   s1, (296+32)(a0)
    sd   a1, (296+48)(a0)
    sd   a2, (296+56)(a0)
    sd   a3, (296+64)(a0)
    sd   a4, (296+72)(a0)
    sd   a5, (296+80)(a0)
    sd   a6, (296+88)(a0)
    sd   a7, (296+96)(a0)
    sd   s2, (296+104)(a0)
    sd   s3, (296+112)(a0)
    sd   s4, (296+120)(a0)
    sd   s5, (296+128)(a0)
    sd   s6, (296+136)(a0)
    sd   s7, (296+144)(a0)
    sd   s8, (296+152)(a0)
    sd   s9, (296+160)(a0)
    sd   s10, (296+168)(a0)
    sd   s11, (296+176)(a0)
    sd   t0, (296+184)(a0)
    sd   t1, (296+192)(a0)
    sd   t2, (296+200)(a0)
    sd   t3, (296+208)(a0)
    sd   t4, (296+216)(a0)
    sd   t5, (296+224)(a0)
    sd   t6, (296+232)(a0)
    sd   sp, (296+240)(a0)

    /* Save Guest a0 after recovering from sscratch. */
    csrr  t0, sscratch
    sd    t0, (296+40)(a0)

    /* Swap in host CSRs. */
    ld    t1, (256)(a0)
    csrrw t1, sstatus, t1
    sd    t1, (296+256)(a0)

    ld    t1, (264)(a0)
    csrrw t1, hstatus, t1
    sd    t1, (296+264)(a0)

    ld    t1, (272)(a0)
    csrrw t1, scounteren, t1
    sd    t1, (296+272)(a0)

    ld    t1, (280)(a0)
    csrw  stvec, t1

    ld    t1, (288)(a0)
    csrw  sscratch, t1

    /* Save guest EPC. */
    csrr  t1, sepc
    sd    t1, (296+280)(a0)

    /* Restore hypervisor GPRs. */
    ld   ra, (0)(a0)
    ld   gp, (8)(a0)
    ld   tp, (16)(a0)
    ld   s0, (24)(a0)
    ld   s1, (32)(a0)
    ld   a1, (48)(a0)
    ld   a2, (56)(a0)
    ld   a3, (64)(a0)
    ld   a4, (72)(a0)
    ld   a5, (80)(a0)
    ld   a6, (88)(a0)
    ld   a7, (96)(a0)
    ld   s2, (104)(a0)
    ld   s3, (112)(a0)
    ld   s4, (120)(a0)
    ld   s5, (128)(a0)
    ld   s6, (136)(a0)
    ld   s7, (144)(a0)
    ld   s8, (152)(a0)
    ld   s9, (160)(a0)
    ld   s10, (168)(a0)
    ld   s11, (176)(a0)
    ld   sp, (240)(a0)

    ret
