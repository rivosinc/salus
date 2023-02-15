// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;

global_asm!(include_str!("start.S"));
global_asm!(include_str!("mem_extable.S"));
