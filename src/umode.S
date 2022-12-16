// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Enter U-mode using `UmodeCpuArchState` in `a0`.
.section .text.init
.global _run_umode
_run_umode:
    /* Save hypervisor state */

    /* Save Host GPRs (except T0-T6 and a0, which is the U-mode state and stashed in sscratch) */
    sd   ra, ({hyp_ra})(a0)
    sd   gp, ({hyp_gp})(a0)
    sd   tp, ({hyp_tp})(a0)
    sd   s0, ({hyp_s0})(a0)
    sd   s1, ({hyp_s1})(a0)
    sd   a1, ({hyp_a1})(a0)
    sd   a2, ({hyp_a2})(a0)
    sd   a3, ({hyp_a3})(a0)
    sd   a4, ({hyp_a4})(a0)
    sd   a5, ({hyp_a5})(a0)
    sd   a6, ({hyp_a6})(a0)
    sd   a7, ({hyp_a7})(a0)
    sd   s2, ({hyp_s2})(a0)
    sd   s3, ({hyp_s3})(a0)
    sd   s4, ({hyp_s4})(a0)
    sd   s5, ({hyp_s5})(a0)
    sd   s6, ({hyp_s6})(a0)
    sd   s7, ({hyp_s7})(a0)
    sd   s8, ({hyp_s8})(a0)
    sd   s9, ({hyp_s9})(a0)
    sd   s10, ({hyp_s10})(a0)
    sd   s11, ({hyp_s11})(a0)
    sd   sp, ({hyp_sp})(a0)

    /* Swap in U-mode CSRs. */
    ld    t1, ({umode_sepc})(a0)
    csrw  sepc, t1

    /* Swap in sstatus. */
    ld    t1, ({umode_sstatus})(a0)
    csrrw t1, sstatus, t1
    sd    t1, ({hyp_sstatus})(a0)

    /* Set stvec so that hypervisor resumes after the sret when U-mode exits. */
    la    t1, _umode_exit
    csrrw t1, stvec, t1
    sd    t1, ({hyp_stvec})(a0)

    /* Save sscratch and replace with pointer to the U-mode state. */
    csrrw t1, sscratch, a0
    sd    t1, ({hyp_sscratch})(a0)

    /* Restore the gprs from the U-mode state */
    ld   ra, ({umode_ra})(a0)
    ld   gp, ({umode_gp})(a0)
    ld   tp, ({umode_tp})(a0)
    ld   s0, ({umode_s0})(a0)
    ld   s1, ({umode_s1})(a0)
    ld   a1, ({umode_a1})(a0)
    ld   a2, ({umode_a2})(a0)
    ld   a3, ({umode_a3})(a0)
    ld   a4, ({umode_a4})(a0)
    ld   a5, ({umode_a5})(a0)
    ld   a6, ({umode_a6})(a0)
    ld   a7, ({umode_a7})(a0)
    ld   s2, ({umode_s2})(a0)
    ld   s3, ({umode_s3})(a0)
    ld   s4, ({umode_s4})(a0)
    ld   s5, ({umode_s5})(a0)
    ld   s6, ({umode_s6})(a0)
    ld   s7, ({umode_s7})(a0)
    ld   s8, ({umode_s8})(a0)
    ld   s9, ({umode_s9})(a0)
    ld   s10, ({umode_s10})(a0)
    ld   s11, ({umode_s11})(a0)
    ld   t0, ({umode_t0})(a0)
    ld   t1, ({umode_t1})(a0)
    ld   t2, ({umode_t2})(a0)
    ld   t3, ({umode_t3})(a0)
    ld   t4, ({umode_t4})(a0)
    ld   t5, ({umode_t5})(a0)
    ld   t6, ({umode_t6})(a0)
    ld   sp, ({umode_sp})(a0)
    ld   a0, ({umode_a0})(a0)

    sret

.align 2
_umode_exit:
    /* Pull U-mode state out of sscratch, swapping with a0 */
    csrrw a0, sscratch, a0

    /* Save U-mode GPRs. */
    sd   ra, ({umode_ra})(a0)
    sd   gp, ({umode_gp})(a0)
    sd   tp, ({umode_tp})(a0)
    sd   s0, ({umode_s0})(a0)
    sd   s1, ({umode_s1})(a0)
    sd   a1, ({umode_a1})(a0)
    sd   a2, ({umode_a2})(a0)
    sd   a3, ({umode_a3})(a0)
    sd   a4, ({umode_a4})(a0)
    sd   a5, ({umode_a5})(a0)
    sd   a6, ({umode_a6})(a0)
    sd   a7, ({umode_a7})(a0)
    sd   s2, ({umode_s2})(a0)
    sd   s3, ({umode_s3})(a0)
    sd   s4, ({umode_s4})(a0)
    sd   s5, ({umode_s5})(a0)
    sd   s6, ({umode_s6})(a0)
    sd   s7, ({umode_s7})(a0)
    sd   s8, ({umode_s8})(a0)
    sd   s9, ({umode_s9})(a0)
    sd   s10, ({umode_s10})(a0)
    sd   s11, ({umode_s11})(a0)
    sd   t0, ({umode_t0})(a0)
    sd   t1, ({umode_t1})(a0)
    sd   t2, ({umode_t2})(a0)
    sd   t3, ({umode_t3})(a0)
    sd   t4, ({umode_t4})(a0)
    sd   t5, ({umode_t5})(a0)
    sd   t6, ({umode_t6})(a0)
    sd   sp, ({umode_sp})(a0)

    /* Save U-mode a0 after recovering from sscratch. */
    csrr  t0, sscratch
    sd    t0, ({umode_a0})(a0)

    /* Swap in host CSRs. */
    ld    t1, ({hyp_stvec})(a0)
    csrw  stvec, t1

    ld    t1, ({hyp_sstatus})(a0)
    csrrw t1, sstatus, t1
    sd    t1, ({umode_sstatus})(a0)

    ld    t1, ({hyp_sscratch})(a0)
    csrw  sscratch, t1

    /* Save U-mode EPC. */
    csrr  t1, sepc
    sd    t1, ({umode_sepc})(a0)

    /* Restore hypervisor GPRs. */
    ld   ra, ({hyp_ra})(a0)
    ld   gp, ({hyp_gp})(a0)
    ld   tp, ({hyp_tp})(a0)
    ld   s0, ({hyp_s0})(a0)
    ld   s1, ({hyp_s1})(a0)
    ld   a1, ({hyp_a1})(a0)
    ld   a2, ({hyp_a2})(a0)
    ld   a3, ({hyp_a3})(a0)
    ld   a4, ({hyp_a4})(a0)
    ld   a5, ({hyp_a5})(a0)
    ld   a6, ({hyp_a6})(a0)
    ld   a7, ({hyp_a7})(a0)
    ld   s2, ({hyp_s2})(a0)
    ld   s3, ({hyp_s3})(a0)
    ld   s4, ({hyp_s4})(a0)
    ld   s5, ({hyp_s5})(a0)
    ld   s6, ({hyp_s6})(a0)
    ld   s7, ({hyp_s7})(a0)
    ld   s8, ({hyp_s8})(a0)
    ld   s9, ({hyp_s9})(a0)
    ld   s10, ({hyp_s10})(a0)
    ld   s11, ({hyp_s11})(a0)
    ld   sp, ({hyp_sp})(a0)

    ret
