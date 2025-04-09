// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::fmt;
use core::mem::size_of;
use drivers::imsic::{Imsic, ImsicInterruptId};
use memoffset::offset_of;
use riscv_regs::{
    sie, GeneralPurposeRegisters, GprIndex, Interrupt, Readable, RiscvCsrInterface, Trap,
    Writeable, CSR,
};
use s_mode_utils::print::*;

use crate::hyp_layout::HYP_STACK_BOTTOM;

#[no_mangle]
static overflow_stack_lock: u64 = 0;

/// Stores the trap context as pushed onto the stack by the trap handler.
#[repr(C)]
struct TrapFrame {
    gprs: GeneralPurposeRegisters,
    sstatus: u64,
    sepc: u64,
}

impl fmt::Display for TrapFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SEPC: 0x{:016x}, SSTATUS: 0x{:016x}",
            self.sepc, self.sstatus,
        )?;
        use GprIndex::*;
        writeln!(
            f,
            "RA:  0x{:016x}, GP:  0x{:016x}, TP:  0x{:016x}, S0:  0x{:016x}",
            self.gprs.reg(RA),
            self.gprs.reg(GP),
            self.gprs.reg(TP),
            self.gprs.reg(S0)
        )?;
        writeln!(
            f,
            "S1:  0x{:016x}, A0:  0x{:016x}, A1:  0x{:016x}, A2:  0x{:016x}",
            self.gprs.reg(S1),
            self.gprs.reg(A0),
            self.gprs.reg(A1),
            self.gprs.reg(A2)
        )?;
        writeln!(
            f,
            "A3:  0x{:016x}, A4:  0x{:016x}, A5:  0x{:016x}, A6:  0x{:016x}",
            self.gprs.reg(A3),
            self.gprs.reg(A4),
            self.gprs.reg(A5),
            self.gprs.reg(A6)
        )?;
        writeln!(
            f,
            "A7:  0x{:016x}, S2:  0x{:016x}, S3:  0x{:016x}, S4:  0x{:016x}",
            self.gprs.reg(A7),
            self.gprs.reg(S2),
            self.gprs.reg(S3),
            self.gprs.reg(S4)
        )?;
        writeln!(
            f,
            "S5:  0x{:016x}, S6:  0x{:016x}, S7:  0x{:016x}, S8:  0x{:016x}",
            self.gprs.reg(S5),
            self.gprs.reg(S6),
            self.gprs.reg(S7),
            self.gprs.reg(S8)
        )?;
        writeln!(
            f,
            "S9:  0x{:016x}, S10: 0x{:016x}, S11: 0x{:016x}, T0:  0x{:016x}",
            self.gprs.reg(S9),
            self.gprs.reg(S10),
            self.gprs.reg(S11),
            self.gprs.reg(T0)
        )?;
        writeln!(
            f,
            "T1:  0x{:016x}, T2:  0x{:016x}, T3:  0x{:016x}, T4:  0x{:016x}",
            self.gprs.reg(T1),
            self.gprs.reg(T2),
            self.gprs.reg(T3),
            self.gprs.reg(T4)
        )?;
        writeln!(
            f,
            "T5:  0x{:016x}, T6:  0x{:016x}, SP:  0x{:016x}",
            self.gprs.reg(T5),
            self.gprs.reg(T6),
            self.gprs.reg(SP)
        )
    }
}

extern "C" {
    // The assembly entry point for handling traps.
    fn _trap_entry();
    // The assembly entry point for handling traps at init time.
    fn _trap_init_entry();

    // The location of the exception table.
    static _extable_start: u8;
    static _extable_end: u8;
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
    hyp_stack_bottom = const HYP_STACK_BOTTOM,
);

/// Attempts to handle an interrupt, returning true if the interrupt was successfully handled.
fn handle_interrupt(irq: Interrupt) -> bool {
    match irq {
        Interrupt::SupervisorExternal => {
            let mut handled = false;
            while let Some(id) = Imsic::next_pending_interrupt() {
                match id {
                    // For now IPIs just wake up the CPU.
                    ImsicInterruptId::Ipi => {
                        handled = true;
                    }
                }
            }
            handled
        }
        // TODO: Handle supervisor guest external interrupts.
        _ => false,
    }
}

/// An entry in the `.extable` section of the binary. Each entry corresponds to an instruction that
/// could generate an exception in supervisor mode (e.g. when copying to/from guest memory). Exceptions
/// that are taken on these instructions are recovered from by:
///   - setting SEPC to the value stored in T0 when the trap was taken
///   - setting T1 to the value of SCAUSE
///   - returning from the trap
///
/// TODO: Put recovery address in exception table too.
#[repr(C)]
struct ExceptionTableEntry {
    // The PC of the potentially-faulting instruction
    pc: u64,
}

/// Returns the exception table as a slice of `ExceptionTableEntry` structs.
fn extable() -> &'static [ExceptionTableEntry] {
    // Safety: we trust that the linker placed the .extable section correctly, along with the
    // _extable_{start,end} symbols.
    unsafe {
        let start = core::ptr::addr_of!(_extable_start) as *const ExceptionTableEntry;
        let end = core::ptr::addr_of!(_extable_end) as *const ExceptionTableEntry;
        core::slice::from_raw_parts(start, end.offset_from(start) as usize)
    }
}

/// Returns if `pc` is in the exception table.
fn pc_in_extable(pc: u64) -> bool {
    extable().iter().any(|e| e.pc == pc)
}

#[no_mangle]
extern "C" fn handle_stack_overflow(tf_ptr: *mut TrapFrame) {
    let tf = unsafe { tf_ptr.as_mut().unwrap() };
    println!("Stack overflow (please note: T1 register is clobbered below)");
    println!("{}", tf);
    panic!("Stack overflow!");
}

/// The rust entry point for handling traps. The only traps we expect to take in HS mode are IPIs
/// (to wake the receiving CPU from WFI) and guest page faults while copying to/from guest memory.
/// For everything else we just dump state and panic.
///
/// TODO: If/when the serial driver takes locks we will need to bust them here in order to avoid
/// deadlock.
#[no_mangle]
extern "C" fn handle_trap(tf_ptr: *mut TrapFrame) {
    // Safe since we trust that TrapFrame was properly intialized by _trap_entry.
    let mut tf = unsafe { tf_ptr.as_mut().unwrap() };
    let scause = CSR.scause.get();
    let stval = CSR.stval.get();

    if let Ok(t) = Trap::from_scause(scause) {
        match t {
            Trap::Interrupt(i) => {
                if handle_interrupt(i) {
                    return;
                }
            }
            Trap::Exception(_) => {
                if pc_in_extable(tf.sepc) {
                    // We took an exception on an instruction in the exception table. Follow the
                    // defined recovery procedure; see ExceptionTableEntry above.
                    tf.sepc = tf.gprs.reg(GprIndex::T0);
                    tf.gprs.set_reg(GprIndex::T1, scause);
                    return;
                }
            }
        };
        print!("Unexpected trap: {}, ", t);
    } else {
        print!("Unexpected trap: <not decoded>, ");
    }

    println!("SCAUSE: 0x{:08x}, STVAL: 0x{:08x}", scause, stval);
    println!("{}", tf);

    panic!("Unexpected trap");
}

/// Installs init-time handler for HS-level traps.
///
/// This is the same as the trap handler below but doesn't check for
/// stack overflow, as hypervisor mappings are not enabled yet.
pub fn install_init_trap_handler() {
    CSR.stvec
        .set((_trap_init_entry as usize).try_into().unwrap());

    // We only expect supervisor-level external interrupts.
    CSR.sie.read_and_set_bits(1 << sie::sext.shift);
}

/// Installs a handler for HS-level traps.
pub fn install_trap_handler() {
    CSR.stvec.set((_trap_entry as usize).try_into().unwrap());

    // We only expect supervisor-level external interrupts for now.
    CSR.sie.read_and_set_bits(1 << sie::sext.shift);
}
