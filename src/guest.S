// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


/// Enter the guest given in `GuestInfo` from `a0`.
.section .text.init
.global _run_guest
_run_guest:
    /* Save hypervisor state */

    /* Save Host GPRs (except T0-T6 and a0, which is GuestInfo and stashed in sscratch) */
    sd   ra, ({host_ra})(a0)
    sd   gp, ({host_gp})(a0)
    sd   tp, ({host_tp})(a0)
    sd   s0, ({host_s0})(a0)
    sd   s1, ({host_s1})(a0)
    sd   a1, ({host_a1})(a0)
    sd   a2, ({host_a2})(a0)
    sd   a3, ({host_a3})(a0)
    sd   a4, ({host_a4})(a0)
    sd   a5, ({host_a5})(a0)
    sd   a6, ({host_a6})(a0)
    sd   a7, ({host_a7})(a0)
    sd   s2, ({host_s2})(a0)
    sd   s3, ({host_s3})(a0)
    sd   s4, ({host_s4})(a0)
    sd   s5, ({host_s5})(a0)
    sd   s6, ({host_s6})(a0)
    sd   s7, ({host_s7})(a0)
    sd   s8, ({host_s8})(a0)
    sd   s9, ({host_s9})(a0)
    sd   s10, ({host_s10})(a0)
    sd   s11, ({host_s11})(a0)
    sd   sp, ({host_sp})(a0)

    /* Swap in guest CSRs. */
    ld    t1, ({guest_sstatus})(a0)
    csrrw t1, sstatus, t1
    sd    t1, ({host_sstatus})(a0)

    ld    t1, ({guest_hstatus})(a0)
    csrrw t1, hstatus, t1
    sd    t1, ({host_hstatus})(a0)

    ld    t1, ({guest_scounteren})(a0)
    csrrw t1, scounteren, t1
    sd    t1, ({host_scounteren})(a0)

    ld    t1, ({guest_sepc})(a0)
    csrw  sepc, t1

    /* Set stvec so that hypervisor resumes after the sret when the guest exits. */
    la    t1, _guest_exit
    csrrw t1, stvec, t1
    sd    t1, ({host_stvec})(a0)

    /* Save sscratch and replace with pointer to GuestInfo. */
    csrrw t1, sscratch, a0
    sd    t1, ({host_sscratch})(a0)

    /* Restore the gprs from this GuestInfo */
    ld   ra, ({guest_ra})(a0)
    ld   gp, ({guest_gp})(a0)
    ld   tp, ({guest_tp})(a0)
    ld   s0, ({guest_s0})(a0)
    ld   s1, ({guest_s1})(a0)
    ld   a1, ({guest_a1})(a0)
    ld   a2, ({guest_a2})(a0)
    ld   a3, ({guest_a3})(a0)
    ld   a4, ({guest_a4})(a0)
    ld   a5, ({guest_a5})(a0)
    ld   a6, ({guest_a6})(a0)
    ld   a7, ({guest_a7})(a0)
    ld   s2, ({guest_s2})(a0)
    ld   s3, ({guest_s3})(a0)
    ld   s4, ({guest_s4})(a0)
    ld   s5, ({guest_s5})(a0)
    ld   s6, ({guest_s6})(a0)
    ld   s7, ({guest_s7})(a0)
    ld   s8, ({guest_s8})(a0)
    ld   s9, ({guest_s9})(a0)
    ld   s10, ({guest_s10})(a0)
    ld   s11, ({guest_s11})(a0)
    ld   t0, ({guest_t0})(a0)
    ld   t1, ({guest_t1})(a0)
    ld   t2, ({guest_t2})(a0)
    ld   t3, ({guest_t3})(a0)
    ld   t4, ({guest_t4})(a0)
    ld   t5, ({guest_t5})(a0)
    ld   t6, ({guest_t6})(a0)
    ld   sp, ({guest_sp})(a0)
    ld   a0, ({guest_a0})(a0)

    sret

.align 2
_guest_exit:
    /* Pull GuestInfo out of sscratch, swapping with guest's a0 */
    csrrw a0, sscratch, a0

    /* Save guest GPRs. */
    sd   ra, ({guest_ra})(a0)
    sd   gp, ({guest_gp})(a0)
    sd   tp, ({guest_tp})(a0)
    sd   s0, ({guest_s0})(a0)
    sd   s1, ({guest_s1})(a0)
    sd   a1, ({guest_a1})(a0)
    sd   a2, ({guest_a2})(a0)
    sd   a3, ({guest_a3})(a0)
    sd   a4, ({guest_a4})(a0)
    sd   a5, ({guest_a5})(a0)
    sd   a6, ({guest_a6})(a0)
    sd   a7, ({guest_a7})(a0)
    sd   s2, ({guest_s2})(a0)
    sd   s3, ({guest_s3})(a0)
    sd   s4, ({guest_s4})(a0)
    sd   s5, ({guest_s5})(a0)
    sd   s6, ({guest_s6})(a0)
    sd   s7, ({guest_s7})(a0)
    sd   s8, ({guest_s8})(a0)
    sd   s9, ({guest_s9})(a0)
    sd   s10, ({guest_s10})(a0)
    sd   s11, ({guest_s11})(a0)
    sd   t0, ({guest_t0})(a0)
    sd   t1, ({guest_t1})(a0)
    sd   t2, ({guest_t2})(a0)
    sd   t3, ({guest_t3})(a0)
    sd   t4, ({guest_t4})(a0)
    sd   t5, ({guest_t5})(a0)
    sd   t6, ({guest_t6})(a0)
    sd   sp, ({guest_sp})(a0)

    /* Save Guest a0 after recovering from sscratch. */
    csrr  t0, sscratch
    sd    t0, ({guest_a0})(a0)

    /* Swap in host CSRs. */
    ld    t1, ({host_sstatus})(a0)
    csrrw t1, sstatus, t1
    sd    t1, ({guest_sstatus})(a0)

    ld    t1, ({host_hstatus})(a0)
    csrrw t1, hstatus, t1
    sd    t1, ({guest_hstatus})(a0)

    ld    t1, ({host_scounteren})(a0)
    csrrw t1, scounteren, t1
    sd    t1, ({guest_scounteren})(a0)

    ld    t1, ({host_stvec})(a0)
    csrw  stvec, t1

    ld    t1, ({host_sscratch})(a0)
    csrw  sscratch, t1

    /* Save guest EPC. */
    csrr  t1, sepc
    sd    t1, ({guest_sepc})(a0)

    /* Restore hypervisor GPRs. */
    ld   ra, ({host_ra})(a0)
    ld   gp, ({host_gp})(a0)
    ld   tp, ({host_tp})(a0)
    ld   s0, ({host_s0})(a0)
    ld   s1, ({host_s1})(a0)
    ld   a1, ({host_a1})(a0)
    ld   a2, ({host_a2})(a0)
    ld   a3, ({host_a3})(a0)
    ld   a4, ({host_a4})(a0)
    ld   a5, ({host_a5})(a0)
    ld   a6, ({host_a6})(a0)
    ld   a7, ({host_a7})(a0)
    ld   s2, ({host_s2})(a0)
    ld   s3, ({host_s3})(a0)
    ld   s4, ({host_s4})(a0)
    ld   s5, ({host_s5})(a0)
    ld   s6, ({host_s6})(a0)
    ld   s7, ({host_s7})(a0)
    ld   s8, ({host_s8})(a0)
    ld   s9, ({host_s9})(a0)
    ld   s10, ({host_s10})(a0)
    ld   s11, ({host_s11})(a0)
    ld   sp, ({host_sp})(a0)

    ret
