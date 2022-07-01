// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Very unoptimized memcpy() to/from guest memory functions, using the HLV/HSV instructions.

// Adds the instruction at 'lbl' to the exception table.
.macro add_extable lbl
.pushsection .extable, "a"
.balign      8
.quad        \lbl
.popsection
.endm

.section .text

// memcpy() to a guest physical address using HSV.
.global _copy_to_guest
_copy_to_guest:
    // handle_trap assumes t0 holds the address of where we want to jump to when we encounter
    // a fault and will stick SCAUSE in t1.
    la    t0, _ret_from_copy
    // _ret_from_copy assumes the return value is in t2.
    mv    t2, zero
1:
    beq   t2, a2, _ret_from_copy
    lb    t3, (a1)
    // HSV.B encoding:
    //   0110001 rs2[4:0] rs1[4:0] 100 00000 1110011
2:
    .word 0x63c54073 // hsv.b t3, (a0)
    add_extable 2b
    addi  a0, a0, 1
    addi  a1, a1, 1
    addi  t2, t2, 1
    j     1b

// memcpy() from a guest physical address using HLV.
.global _copy_from_guest
_copy_from_guest:
    // handle_trap assumes t0 holds the address of where we want to jump to when we encounter
    // a fault and will stick SCAUSE in t1.
    la    t0, _ret_from_copy
    // _ret_from_copy assumes the return value is in t2.
    mv    t2, zero
1:
    beq   t2, a2, _ret_from_copy
    // HLV.B encoding:
    //   0110000 00000 rs1[4:0] 100 rd[4:0] 1110011
2:
    .word 0x6005ce73 // hlv.b t3, (a1)
    add_extable 2b
    sb    t3, (a0)
    addi  a0, a0, 1
    addi  a1, a1, 1
    addi  t2, t2, 1
    j     1b

.align 2
_ret_from_copy:
    mv    a0, t2
    ret
