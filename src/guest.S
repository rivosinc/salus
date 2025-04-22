// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


/// Enter the guest given in `GuestInfo` from `a0`.
.attribute arch, "rv64gcv"
.section .text.init
.global _run_guest
_run_guest:
    /* Save hypervisor state */

    /* Save hypervisor GPRs (except T0-T6 and a0, which is GuestInfo and stashed in sscratch) */
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

    /* Swap in guest CSRs. */
    ld    t1, ({guest_sstatus})(a0)
    csrrw t1, sstatus, t1
    sd    t1, ({hyp_sstatus})(a0)

    ld    t1, ({guest_hstatus})(a0)
    csrrw t1, hstatus, t1
    sd    t1, ({hyp_hstatus})(a0)

    ld    t1, ({guest_scounteren})(a0)
    csrrw t1, scounteren, t1
    sd    t1, ({hyp_scounteren})(a0)

    ld    t1, ({guest_sepc})(a0)
    csrw  sepc, t1

    /* Set stvec so that hypervisor resumes after the sret when the guest exits. */
    la    t1, _guest_exit
    csrrw t1, stvec, t1
    sd    t1, ({hyp_stvec})(a0)

    /* Save sscratch and replace with pointer to GuestInfo. */
    csrrw t1, sscratch, a0
    sd    t1, ({hyp_sscratch})(a0)

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

_restore_csrs:
    /* Swap in hypervisor CSRs. */
    ld    t1, ({hyp_sstatus})(a0)
    csrrw t1, sstatus, t1
    sd    t1, ({guest_sstatus})(a0)

    ld    t1, ({hyp_hstatus})(a0)
    csrrw t1, hstatus, t1
    sd    t1, ({guest_hstatus})(a0)

    ld    t1, ({hyp_scounteren})(a0)
    csrrw t1, scounteren, t1
    sd    t1, ({guest_scounteren})(a0)

    ld    t1, ({hyp_stvec})(a0)
    csrw  stvec, t1

    ld    t1, ({hyp_sscratch})(a0)
    csrw  sscratch, t1

    /* Save guest EPC. */
    csrr  t1, sepc
    sd    t1, ({guest_sepc})(a0)

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

/// Restore guest FPU state. A0 = pointer to VmCpuState.
.global _restore_fp
_restore_fp:
    /* Temporarily enable FPU access, saving/restoring sstatus. */
    csrr  t1, sstatus
    li    t0, {sstatus_fs_dirty}
    or    t0, t0, t1
    csrw  sstatus, t0
    fld   f0, ({guest_f0})(a0)
    fld   f1, ({guest_f1})(a0)
    fld   f2, ({guest_f2})(a0)
    fld   f3, ({guest_f3})(a0)
    fld   f4, ({guest_f4})(a0)
    fld   f5, ({guest_f5})(a0)
    fld   f6, ({guest_f6})(a0)
    fld   f7, ({guest_f7})(a0)
    fld   f8, ({guest_f8})(a0)
    fld   f9, ({guest_f9})(a0)
    fld   f10, ({guest_f10})(a0)
    fld   f11, ({guest_f11})(a0)
    fld   f12, ({guest_f12})(a0)
    fld   f13, ({guest_f13})(a0)
    fld   f14, ({guest_f14})(a0)
    fld   f15, ({guest_f15})(a0)
    fld   f16, ({guest_f16})(a0)
    fld   f17, ({guest_f17})(a0)
    fld   f18, ({guest_f18})(a0)
    fld   f19, ({guest_f19})(a0)
    fld   f20, ({guest_f20})(a0)
    fld   f21, ({guest_f21})(a0)
    fld   f22, ({guest_f22})(a0)
    fld   f23, ({guest_f23})(a0)
    fld   f24, ({guest_f24})(a0)
    fld   f25, ({guest_f25})(a0)
    fld   f26, ({guest_f26})(a0)
    fld   f27, ({guest_f27})(a0)
    fld   f28, ({guest_f28})(a0)
    fld   f29, ({guest_f29})(a0)
    fld   f30, ({guest_f30})(a0)
    fld   f31, ({guest_f31})(a0)
    ld    t0, ({guest_fcsr})(a0)
    fscsr t0
    csrw  sstatus, t1
    ret

/// Save guest FPU state. A0 = pointer to VmCpuState.
.global _save_fp
_save_fp:
    /* Temporarily enable FPU access, saving/restoring sstatus. */
    csrr  t1, sstatus
    li    t0, {sstatus_fs_dirty}
    or    t0, t0, t1
    csrw  sstatus, t0
    fsd   f0, ({guest_f0})(a0)
    fsd   f1, ({guest_f1})(a0)
    fsd   f2, ({guest_f2})(a0)
    fsd   f3, ({guest_f3})(a0)
    fsd   f4, ({guest_f4})(a0)
    fsd   f5, ({guest_f5})(a0)
    fsd   f6, ({guest_f6})(a0)
    fsd   f7, ({guest_f7})(a0)
    fsd   f8, ({guest_f8})(a0)
    fsd   f9, ({guest_f9})(a0)
    fsd   f10, ({guest_f10})(a0)
    fsd   f11, ({guest_f11})(a0)
    fsd   f12, ({guest_f12})(a0)
    fsd   f13, ({guest_f13})(a0)
    fsd   f14, ({guest_f14})(a0)
    fsd   f15, ({guest_f15})(a0)
    fsd   f16, ({guest_f16})(a0)
    fsd   f17, ({guest_f17})(a0)
    fsd   f18, ({guest_f18})(a0)
    fsd   f19, ({guest_f19})(a0)
    fsd   f20, ({guest_f20})(a0)
    fsd   f21, ({guest_f21})(a0)
    fsd   f22, ({guest_f22})(a0)
    fsd   f23, ({guest_f23})(a0)
    fsd   f24, ({guest_f24})(a0)
    fsd   f25, ({guest_f25})(a0)
    fsd   f26, ({guest_f26})(a0)
    fsd   f27, ({guest_f27})(a0)
    fsd   f28, ({guest_f28})(a0)
    fsd   f29, ({guest_f29})(a0)
    fsd   f30, ({guest_f30})(a0)
    fsd   f31, ({guest_f31})(a0)
    frcsr t0
    sd    t0, ({guest_fcsr})(a0)
    csrw  sstatus, t1
    ret

// Save guest vector state. A0 = pointer to VmCpuState.
.global _save_vector
_save_vector:
    .option push
    .option arch,+v
    // Temporarily enable vectors
    csrr  t5, sstatus
    li    t0, {sstatus_vs_enable}
    csrrs zero, sstatus, t0

    // Store vector CSRs
    csrr  t4, vcsr
    sd    t4, {guest_vcsr}(a0)
    csrr  t4, vstart
    sd    t4, {guest_vstart}(a0)
    csrr  t4, vtype
    sd    t4, {guest_vtype}(a0)
    csrr  t4, vl
    sd    t4, {guest_vl}(a0)

    // Store register file
    addi   t3, a0, {guest_v0}
    vs8r.v v0, (t3)
    addi   t3, a0, {guest_v8}
    vs8r.v v8, (t3)
    addi   t3, a0, {guest_v16}
    vs8r.v v16, (t3)
    addi   t3, a0, {guest_v24}
    vs8r.v v24, (t3)

    // Restore sstatus
    csrw  sstatus, t5
    ret
    .option pop

// Restore guest vector state. A0 = pointer to VmCpuState.
.global _restore_vector
_restore_vector:
    .option push
    .option arch,+v
    // Temporarily enable vectors
    csrr  t5, sstatus
    li    t4, {sstatus_vs_enable}
    csrrs zero, sstatus, t4

    // Restore type and length
    ld     t1, ({guest_vl})(a0)
    ld     t2, ({guest_vtype})(a0)
    vsetvl t0,t1,t2

    // Restore vstart
    ld     t0, ({guest_vstart})(a0)
    csrw   vstart, t0

    // Restore vcsr
    ld     t0, ({guest_vcsr})(a0)
    csrw   vcsr, t0

    // Restore register file
    addi   t3, a0, {guest_v0}
    vl8r.v v0, (t3)
    addi   t3, a0, {guest_v8}
    vl8r.v v8, (t3)
    addi   t3, a0, {guest_v16}
    vl8r.v v16, (t3)
    addi   t3, a0, {guest_v24}
    vl8r.v v24, (t3)

    // Restore sstatus
    csrw  sstatus, t5
    ret
    .option pop
