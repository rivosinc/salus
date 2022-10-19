.section .text.init
.global _restore_vector,_save_vector

// t5 -> cached sstatus
// a0 -> pointer to VmCpuState
// _restore_vectors restores vector state for the
// guest before leaving salus and running the guest.
.align 2
_restore_vector:

    // Save sstatus
    csrr t5, sstatus

    // Enable vector commands
    li t4, {sstatus_vs_enable}
    csrrs zero, sstatus, t4

    // Restore type and length
    ld t1, ({guest_vl})(a0)
    ld t2, ({guest_vtype})(a0)
    vsetvl t0,t1,t2

    // Restore vstart
    ld t0, ({guest_vstart})(a0)
    csrw vstart, t0

    // Restore vcsr
    ld t0, ({guest_vcsr})(a0)
    csrw vcsr, t0

    // Restore register file
    addi t3, a0, {guest_v0}
    vl8r.v  v0, (t3)
    addi t3, a0, {guest_v8}
    vl8r.v  v8, (t3)
    addi t3, a0, {guest_v16}
    vl8r.v  v16, (t3)
    addi t3, a0, {guest_v24}
    vl8r.v  v24, (t3)

    csrw sstatus, t5
    ret

// t5 -> cached sstatus
// a0 -> pointer to VmCpuState
// _save_vector saves vector state from a guest to
// salus memory before returning to salus
.align 2
_save_vector:
    // Save sstatus so we can enable vectors
    csrr  t5, sstatus

    // Enable vectors
    li    t0, {sstatus_vs_enable}
    csrrs zero, sstatus, t0

    // Store csr's
    csrr  t4, vcsr
    sd t4, {guest_vcsr}(a0)
    csrr t4, vstart
    sd t4, {guest_vstart}(a0)
    csrr t4, vtype
    sd t4, {guest_vtype}(a0)
    csrr t4, vl
    sd t4, {guest_vl}(a0)

    // Store register file
    addi t3, a0, {guest_v0}
    vs8r.v  v0, (t3)
    addi t3, a0, {guest_v8}
    vs8r.v  v8, (t3)
    addi t3, a0, {guest_v16}
    vs8r.v  v16, (t3)
    addi t3, a0, {guest_v24}
    vs8r.v  v24, (t3)

    // Restore sstatus
    csrw  sstatus, t5

    ret
