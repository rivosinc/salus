// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::mem::size_of;
use memoffset::offset_of;
use riscv_regs::{GeneralPurposeRegisters, GprIndex, Readable, Trap, Writeable, CSR};

use crate::print_util::*;
use crate::{print, println};

/// Stores the trap context as pushed onto the stack by the trap handler.
#[repr(C)]
struct TrapFrame {
    gprs: GeneralPurposeRegisters,
    sstatus: u64,
    sepc: u64,
}

// The assembly entry point for handling traps.
extern "C" {
    fn _trap_entry();
}

const fn gpr_offset(index: GprIndex) -> usize {
    offset_of!(TrapFrame, gprs) + (index as usize) * size_of::<u64>()
}

global_asm!(
    include_str!("trap.S"),
    tf_size = const size_of::<TrapFrame>(),
    tf_ra = const gpr_offset(GprIndex::RA),
    tf_gp = const gpr_offset(GprIndex::GP),
    tf_tp = const gpr_offset(GprIndex::TP),
    tf_s0 = const gpr_offset(GprIndex::S0),
    tf_s1 = const gpr_offset(GprIndex::S1),
    tf_a0 = const gpr_offset(GprIndex::A0),
    tf_a1 = const gpr_offset(GprIndex::A1),
    tf_a2 = const gpr_offset(GprIndex::A2),
    tf_a3 = const gpr_offset(GprIndex::A3),
    tf_a4 = const gpr_offset(GprIndex::A4),
    tf_a5 = const gpr_offset(GprIndex::A5),
    tf_a6 = const gpr_offset(GprIndex::A6),
    tf_a7 = const gpr_offset(GprIndex::A7),
    tf_s2 = const gpr_offset(GprIndex::S2),
    tf_s3 = const gpr_offset(GprIndex::S3),
    tf_s4 = const gpr_offset(GprIndex::S4),
    tf_s5 = const gpr_offset(GprIndex::S5),
    tf_s6 = const gpr_offset(GprIndex::S6),
    tf_s7 = const gpr_offset(GprIndex::S7),
    tf_s8 = const gpr_offset(GprIndex::S8),
    tf_s9 = const gpr_offset(GprIndex::S9),
    tf_s10 = const gpr_offset(GprIndex::S10),
    tf_s11 = const gpr_offset(GprIndex::S11),
    tf_t0 = const gpr_offset(GprIndex::T0),
    tf_t1 = const gpr_offset(GprIndex::T1),
    tf_t2 = const gpr_offset(GprIndex::T2),
    tf_t3 = const gpr_offset(GprIndex::T3),
    tf_t4 = const gpr_offset(GprIndex::T4),
    tf_t5 = const gpr_offset(GprIndex::T5),
    tf_t6 = const gpr_offset(GprIndex::T6),
    tf_sp = const gpr_offset(GprIndex::SP),
    tf_sstatus = const offset_of!(TrapFrame, sstatus),
    tf_sepc = const offset_of!(TrapFrame, sepc),
);

/// The rust entry point for handling traps. For now we don't expect to take any traps in HS mode
/// outside of VM exits, so this handler just dumps state and panics.
///
/// TODO: If/when the serial driver takes locks we will need to bust them here in order to avoid
/// deadlock.
#[no_mangle]
extern "C" fn handle_trap(tf_ptr: *const TrapFrame) {
    // Safe since we trust that TrapFrame was properly intialized by _trap_entry.
    let tf = unsafe { tf_ptr.as_ref().unwrap() };
    let scause = CSR.scause.get();

    if let Ok(t) = Trap::from_scause(scause) {
        print!("Unexpected trap: {}, ", t);
    } else {
        print!("Unexpected trap: <not decoded>, ");
    }
    println!("SCAUSE: 0x{:08x}", scause);
    println!(
        "SEPC: 0x{:08x}, SSTATUS: 0x{:08x}, STVAL: 0x{:08x}",
        tf.sepc,
        tf.sstatus,
        CSR.stval.get()
    );
    use GprIndex::*;
    println!(
        "RA: 0x{:08x}, GP: 0x{:08x}, TP: 0x{:08x}, S0: 0x{:08x}",
        tf.gprs.reg(RA),
        tf.gprs.reg(GP),
        tf.gprs.reg(TP),
        tf.gprs.reg(S0)
    );
    println!(
        "S1: 0x{:08x}, A0: 0x{:08x}, A1: 0x{:08x}, A2: 0x{:08x}",
        tf.gprs.reg(S1),
        tf.gprs.reg(A0),
        tf.gprs.reg(A1),
        tf.gprs.reg(A2)
    );
    println!(
        "A3: 0x{:08x}, A4: 0x{:08x}, A5: 0x{:08x}, A6: 0x{:08x}",
        tf.gprs.reg(A3),
        tf.gprs.reg(A4),
        tf.gprs.reg(A5),
        tf.gprs.reg(A6)
    );
    println!(
        "A7: 0x{:08x}, S2: 0x{:08x}, S3: 0x{:08x}, S4: 0x{:08x}",
        tf.gprs.reg(A7),
        tf.gprs.reg(S2),
        tf.gprs.reg(S3),
        tf.gprs.reg(S4)
    );
    println!(
        "S5: 0x{:08x}, S6: 0x{:08x}, S7: 0x{:08x}, S8: 0x{:08x}",
        tf.gprs.reg(S5),
        tf.gprs.reg(S6),
        tf.gprs.reg(S7),
        tf.gprs.reg(S8)
    );
    println!(
        "S9: 0x{:08x}, S10: 0x{:08x}, S11: 0x{:08x}, T0: 0x{:08x}",
        tf.gprs.reg(S9),
        tf.gprs.reg(S10),
        tf.gprs.reg(S11),
        tf.gprs.reg(T0)
    );
    println!(
        "S9: 0x{:08x}, S10: 0x{:08x}, S11: 0x{:08x}, T0: 0x{:08x}",
        tf.gprs.reg(S9),
        tf.gprs.reg(S10),
        tf.gprs.reg(S11),
        tf.gprs.reg(T0)
    );
    println!(
        "T1: 0x{:08x}, T2: 0x{:08x}, T3: 0x{:08x}, T4: 0x{:08x}",
        tf.gprs.reg(T1),
        tf.gprs.reg(T2),
        tf.gprs.reg(T3),
        tf.gprs.reg(T4)
    );
    println!(
        "T5: 0x{:08x}, T6: 0x{:08x}, SP: 0x{:08x}",
        tf.gprs.reg(T5),
        tf.gprs.reg(T6),
        tf.gprs.reg(SP)
    );

    panic!("Unexpected trap");
}

/// Installs a handler for HS-level traps.
pub fn install_trap_handler() {
    CSR.stvec.set((_trap_entry as usize).try_into().unwrap());
}
