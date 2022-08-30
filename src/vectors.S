.section .text.init
.global _run_guest_vector,_guest_exit_vector

// t5 -> cached sstatus
// Restore vectors for guest
// i.e. memory to main state
.align 2
_run_guest_vector:
    
    // Save sstatus
    csrr t5, sstatus

    // Enable vector commands
    addi t4, zero, {sstatus_vs_enable}
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

// t1 -> cached sstatus
// Save vectors for guest
// i.e. from guest to mem
.align 2
_guest_exit_vector:
    // Save sstatus
    csrr  t5, sstatus

    // Get guest_sstatus (sstatus at exit)
    ld    t1, {guest_sstatus}(a0)

    
    // if !(sstatus & sstatus) return
    li    t0, {sstatus_vs_dirty}
    and   t2, t1, t0
    bne   t2, t0, _guest_exit_vector_return
    
    // Set sstatus value to clean
    not   t0, t0
    and   t1, t1, t0
    li    t0, {sstatus_vs_clean}
    or    t1, t1, t0

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

    // For debugging only: clear out vector registers 
    // to be sure they are restored correctly.
    // TODO: Remove this before going into production.
    vmv.v.i v0, 0
    vmv.v.i v4, 0
    vmv.v.i v8, 0
    vmv.v.i v12, 0
    vmv.v.i v16, 0
    vmv.v.i v20, 0
    vmv.v.i v24, 0
    vmv.v.i v28, 0

    // Save corrected guest_sstatus
    sd t1, {guest_sstatus}(a0)

    // Restore sstatus
    csrw  sstatus, t5

_guest_exit_vector_return:
    ret
