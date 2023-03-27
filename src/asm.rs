// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;

use crate::hyp_layout::HYP_STACK_TOP;

global_asm!(include_str!("start.S"), HYP_STACK_TOP=const HYP_STACK_TOP);
global_asm!(include_str!("mem_extable.S"));
