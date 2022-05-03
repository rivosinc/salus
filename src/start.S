// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

.option norvc

.section .text.start

.global _start
_start:

.option push
.option norelax
	la		gp, _global_pointer
.option pop
    la		sp, _stack_end
    li		t0, 0b11 << 11 | (1 << 13)
    csrw	sstatus, t0
    csrw	sie, zero

    la t1, kernel_init
    la ra, 1f
    jr t1

    j 1f
1:
    wfi
    j 1b
