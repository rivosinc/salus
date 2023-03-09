// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use crate::hyp_layout::{UmodeSlotId, UMODE_INPUT_SIZE, UMODE_INPUT_START};
use crate::hyp_map::{Error as HypMapError, HypMap, UmodeSlotPerm};
use crate::smp::PerCpu;
use crate::vm::FinalizedVm;
use crate::vm_pages::{Error as VmPagesError, FinalizedVmPages, GuestUmodeMapping};

use attestation::{AttestationManager, Error as AttestationError};
use core::arch::global_asm;
use core::fmt;
use core::mem::size_of;
use core::ops::ControlFlow;
use data_model::DataInit;
use memoffset::offset_of;
use rice::cdi::CompoundDeviceIdentifier;
use riscv_elf::ElfMap;
use riscv_page_tables::GuestStagePagingMode;
use riscv_pages::{GuestPhysAddr, PageAddr, PageSize, RawAddr, SupervisorVirt};
use riscv_regs::{Exception::UserEnvCall, GeneralPurposeRegisters, GprIndex, Readable, Trap, CSR};
use s_mode_utils::print::*;
use signature::Signer;
use sync::Once;
use u_mode_api::{
    CdiOp, CdiSel, Error as UmodeApiError, HypCall, OpResult, TryIntoRegisters, UmodeRequest,
    CDIOP_SIGN_MAXMSG,
};

/// Host GPR and which must be saved/restored when entering/exiting U-mode.
#[derive(Default)]
#[repr(C)]
struct HostCpuRegs {
    gprs: GeneralPurposeRegisters,
    sstatus: u64,
    stvec: u64,
    sscratch: u64,
}

/// Umode GPR and CSR state which must be saved/restored when exiting/entering U-mode.
#[derive(Default)]
#[repr(C)]
struct UmodeCpuRegs {
    gprs: GeneralPurposeRegisters,
    sepc: u64,
    sstatus: u64,
}

/// CSRs written on an exit from virtualization that are used by the host to determine the cause of
/// the trap.
#[derive(Default)]
#[repr(C)]
struct TrapRegs {
    scause: u64,
    stval: u64,
}

/// CPU register state that must be saved or restored when entering/exiting U-mode.
#[derive(Default)]
#[repr(C)]
struct UmodeCpuArchState {
    hyp_regs: HostCpuRegs,
    umode_regs: UmodeCpuRegs,
    trap_csrs: TrapRegs,
}

impl fmt::Display for UmodeCpuArchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let uregs = &self.umode_regs;
        writeln!(
            f,
            "SEPC: 0x{:016x}, SSTATUS: 0x{:016x}",
            uregs.sepc, uregs.sstatus,
        )?;
        writeln!(
            f,
            "SCAUSE: 0x{:016x}, STVAL: 0x{:016x}",
            self.trap_csrs.scause, self.trap_csrs.stval,
        )?;
        use GprIndex::*;
        writeln!(
            f,
            "RA:  0x{:016x}, GP:  0x{:016x}, TP:  0x{:016x}, S0:  0x{:016x}",
            uregs.gprs.reg(RA),
            uregs.gprs.reg(GP),
            uregs.gprs.reg(TP),
            uregs.gprs.reg(S0)
        )?;
        writeln!(
            f,
            "S1:  0x{:016x}, A0:  0x{:016x}, A1:  0x{:016x}, A2:  0x{:016x}",
            uregs.gprs.reg(S1),
            uregs.gprs.reg(A0),
            uregs.gprs.reg(A1),
            uregs.gprs.reg(A2)
        )?;
        writeln!(
            f,
            "A3:  0x{:016x}, A4:  0x{:016x}, A5:  0x{:016x}, A6:  0x{:016x}",
            uregs.gprs.reg(A3),
            uregs.gprs.reg(A4),
            uregs.gprs.reg(A5),
            uregs.gprs.reg(A6)
        )?;
        writeln!(
            f,
            "A7:  0x{:016x}, S2:  0x{:016x}, S3:  0x{:016x}, S4:  0x{:016x}",
            uregs.gprs.reg(A7),
            uregs.gprs.reg(S2),
            uregs.gprs.reg(S3),
            uregs.gprs.reg(S4)
        )?;
        writeln!(
            f,
            "S5:  0x{:016x}, S6:  0x{:016x}, S7:  0x{:016x}, S8:  0x{:016x}",
            uregs.gprs.reg(S5),
            uregs.gprs.reg(S6),
            uregs.gprs.reg(S7),
            uregs.gprs.reg(S8)
        )?;
        writeln!(
            f,
            "S9:  0x{:016x}, S10: 0x{:016x}, S11: 0x{:016x}, T0:  0x{:016x}",
            uregs.gprs.reg(S9),
            uregs.gprs.reg(S10),
            uregs.gprs.reg(S11),
            uregs.gprs.reg(T0)
        )?;
        writeln!(
            f,
            "T1:  0x{:016x}, T2:  0x{:016x}, T3:  0x{:016x}, T4:  0x{:016x}",
            uregs.gprs.reg(T1),
            uregs.gprs.reg(T2),
            uregs.gprs.reg(T3),
            uregs.gprs.reg(T4)
        )?;
        writeln!(
            f,
            "T5:  0x{:016x}, T6:  0x{:016x}, SP:  0x{:016x}",
            uregs.gprs.reg(T5),
            uregs.gprs.reg(T6),
            uregs.gprs.reg(SP)
        )
    }
}

extern "C" {
    // umode context switch. Defined in umode.S
    fn _run_umode(g: *mut UmodeCpuArchState);
}

#[allow(dead_code)]
const fn hyp_gpr_offset(index: GprIndex) -> usize {
    offset_of!(UmodeCpuArchState, hyp_regs)
        + offset_of!(HostCpuRegs, gprs)
        + (index as usize) * size_of::<u64>()
}

#[allow(dead_code)]
const fn umode_gpr_offset(index: GprIndex) -> usize {
    offset_of!(UmodeCpuArchState, umode_regs)
        + offset_of!(UmodeCpuRegs, gprs)
        + (index as usize) * size_of::<u64>()
}

macro_rules! hyp_csr_offset {
    ($reg:tt) => {
        offset_of!(UmodeCpuArchState, hyp_regs) + offset_of!(HostCpuRegs, $reg)
    };
}

macro_rules! umode_csr_offset {
    ($reg:tt) => {
        offset_of!(UmodeCpuArchState, umode_regs) + offset_of!(UmodeCpuRegs, $reg)
    };
}

global_asm!(
    include_str!("umode.S"),
    hyp_ra = const hyp_gpr_offset(GprIndex::RA),
    hyp_gp = const hyp_gpr_offset(GprIndex::GP),
    hyp_tp = const hyp_gpr_offset(GprIndex::TP),
    hyp_s0 = const hyp_gpr_offset(GprIndex::S0),
    hyp_s1 = const hyp_gpr_offset(GprIndex::S1),
    hyp_a1 = const hyp_gpr_offset(GprIndex::A1),
    hyp_a2 = const hyp_gpr_offset(GprIndex::A2),
    hyp_a3 = const hyp_gpr_offset(GprIndex::A3),
    hyp_a4 = const hyp_gpr_offset(GprIndex::A4),
    hyp_a5 = const hyp_gpr_offset(GprIndex::A5),
    hyp_a6 = const hyp_gpr_offset(GprIndex::A6),
    hyp_a7 = const hyp_gpr_offset(GprIndex::A7),
    hyp_s2 = const hyp_gpr_offset(GprIndex::S2),
    hyp_s3 = const hyp_gpr_offset(GprIndex::S3),
    hyp_s4 = const hyp_gpr_offset(GprIndex::S4),
    hyp_s5 = const hyp_gpr_offset(GprIndex::S5),
    hyp_s6 = const hyp_gpr_offset(GprIndex::S6),
    hyp_s7 = const hyp_gpr_offset(GprIndex::S7),
    hyp_s8 = const hyp_gpr_offset(GprIndex::S8),
    hyp_s9 = const hyp_gpr_offset(GprIndex::S9),
    hyp_s10 = const hyp_gpr_offset(GprIndex::S10),
    hyp_s11 = const hyp_gpr_offset(GprIndex::S11),
    hyp_sp = const hyp_gpr_offset(GprIndex::SP),
    hyp_sstatus = const hyp_csr_offset!(sstatus),
    hyp_stvec = const hyp_csr_offset!(stvec),
    hyp_sscratch = const hyp_csr_offset!(sscratch),
    umode_ra = const umode_gpr_offset(GprIndex::RA),
    umode_gp = const umode_gpr_offset(GprIndex::GP),
    umode_tp = const umode_gpr_offset(GprIndex::TP),
    umode_s0 = const umode_gpr_offset(GprIndex::S0),
    umode_s1 = const umode_gpr_offset(GprIndex::S1),
    umode_a0 = const umode_gpr_offset(GprIndex::A0),
    umode_a1 = const umode_gpr_offset(GprIndex::A1),
    umode_a2 = const umode_gpr_offset(GprIndex::A2),
    umode_a3 = const umode_gpr_offset(GprIndex::A3),
    umode_a4 = const umode_gpr_offset(GprIndex::A4),
    umode_a5 = const umode_gpr_offset(GprIndex::A5),
    umode_a6 = const umode_gpr_offset(GprIndex::A6),
    umode_a7 = const umode_gpr_offset(GprIndex::A7),
    umode_s2 = const umode_gpr_offset(GprIndex::S2),
    umode_s3 = const umode_gpr_offset(GprIndex::S3),
    umode_s4 = const umode_gpr_offset(GprIndex::S4),
    umode_s5 = const umode_gpr_offset(GprIndex::S5),
    umode_s6 = const umode_gpr_offset(GprIndex::S6),
    umode_s7 = const umode_gpr_offset(GprIndex::S7),
    umode_s8 = const umode_gpr_offset(GprIndex::S8),
    umode_s9 = const umode_gpr_offset(GprIndex::S9),
    umode_s10 = const umode_gpr_offset(GprIndex::S10),
    umode_s11 = const umode_gpr_offset(GprIndex::S11),
    umode_t0 = const umode_gpr_offset(GprIndex::T0),
    umode_t1 = const umode_gpr_offset(GprIndex::T1),
    umode_t2 = const umode_gpr_offset(GprIndex::T2),
    umode_t3 = const umode_gpr_offset(GprIndex::T3),
    umode_t4 = const umode_gpr_offset(GprIndex::T4),
    umode_t5 = const umode_gpr_offset(GprIndex::T5),
    umode_t6 = const umode_gpr_offset(GprIndex::T6),
    umode_sp = const umode_gpr_offset(GprIndex::SP),
    umode_sepc = const umode_csr_offset!(sepc),
    umode_sstatus = const umode_csr_offset!(sstatus),
);

#[derive(Debug)]
/// Errors returned by U-mode functions.
pub enum Error {
    /// U-mode task returned an error while running.
    Exec(ExecError),
    /// U-mode task completed execution but returned an error.
    Request(UmodeApiError),
    /// Error while sharing data.
    HypMap(HypMapError),
    /// Error while mapping VM pages.
    VmPages(VmPagesError),
    /// Attestation Manager Errors.
    Attestation(AttestationError),
    /// Address Overflow
    AddressOverflow,
}

/// Errors returned by U-mode task execution.
#[derive(Debug)]
pub enum ExecError {
    /// Received an unexpected trap while running Umode.
    UnexpectedTrap,
    /// Umode called panic.
    Panic,
    /// U-mode API error.
    Api(UmodeApiError),
    /// User address access error
    UmodeAccess(HypMapError),
    /// Unexpected CDI Operation
    UnexpectedCdiOp(CdiSel, CdiOp),
    /// CDI error
    Cdi(rice::Error),
    /// Incorrect size of buffer provided.
    BufferSize(u64, u64),
}

struct UmodeExecutionContext<'a, T: DataInit> {
    input_data: Option<T>,
    req: UmodeRequest,
    attestation: Option<&'a AttestationManager<sha2::Sha384>>,
}

// Entry for umode task.
static UMODE_ENTRY: Once<u64> = Once::new();

/// Represents a per-CPU U-mode task.
pub struct UmodeTask {
    arch: UmodeCpuArchState,
}

impl UmodeTask {
    /// Initialize U-mode tasks. Must be called once before `setup_this_cpu()`.
    pub fn init(umode_elf: ElfMap) {
        UMODE_ENTRY.call_once(|| umode_elf.entry());
        // Consumes the ElfMap.
    }

    /// Initialize this CPU's U-mode task. Must be called once on each physical CPU.
    pub fn setup_this_cpu() -> Result<(), Error> {
        let arch = UmodeCpuArchState::default();
        let mut task = UmodeTask { arch };
        task.reset()?;
        // Install umode cpu state in the current cpu.
        PerCpu::this_cpu().set_umode_task(task);
        Ok(())
    }

    fn map_guest_range_in_umode_slot<T: GuestStagePagingMode>(
        vm_pages: FinalizedVmPages<T>,
        addr: GuestPhysAddr,
        len: usize,
        slot: UmodeSlotId,
        perm: UmodeSlotPerm,
    ) -> Result<(RawAddr<SupervisorVirt>, GuestUmodeMapping), Error> {
        let base = PageAddr::with_round_down(addr, PageSize::Size4k);
        let end = addr
            .checked_increment(len as u64)
            .ok_or(Error::AddressOverflow)?;
        let umode_mapping = vm_pages
            .map_in_umode_slot(
                slot,
                base,
                PageSize::num_4k_pages(end.bits() - base.bits()),
                perm,
            )
            .map_err(Error::VmPages)?;
        let vaddr = umode_mapping
            .vaddr()
            .raw()
            .checked_increment(addr.bits() - base.bits())
            .ok_or(Error::AddressOverflow)?;
        Ok((vaddr, umode_mapping))
    }

    pub fn nop() -> Result<u64, Error> {
        let ctx = UmodeExecutionContext::<u8> {
            input_data: None,
            req: UmodeRequest::Nop,
            attestation: None,
        };
        Self::execute_request(ctx)
    }

    pub fn attestation_evidence<T: GuestStagePagingMode>(
        vm: &FinalizedVm<T>,
        csr_gpa: GuestPhysAddr,
        csr_len: usize,
        certout_gpa: GuestPhysAddr,
        certout_len: usize,
    ) -> Result<u64, Error> {
        // Map input CSR in Slot A as read-only.
        let (csr_vaddr, _csr_mapping) = Self::map_guest_range_in_umode_slot(
            vm.vm_pages(),
            csr_gpa,
            csr_len,
            UmodeSlotId::A,
            UmodeSlotPerm::Readonly,
        )?;
        // Map output certificate in Slot B as writable.
        let (certout_vaddr, _certout_mapping) = Self::map_guest_range_in_umode_slot(
            vm.vm_pages(),
            certout_gpa,
            certout_len,
            UmodeSlotId::B,
            UmodeSlotPerm::Writable,
        )?;
        let attestation_mgr = vm.attestation_mgr();
        // Gather measurement registers from the attestation manager and transform it in a array.
        let msmt_genarray = attestation_mgr
            .measurement_registers()
            .map_err(Error::Attestation)?;
        let zero = [0u8; u_mode_api::cert::SHA384_LEN];
        let mut msmt_regs = [zero; attestation::MSMT_REGISTERS];
        for (i, r) in msmt_genarray.iter().enumerate() {
            msmt_regs[i].copy_from_slice(r.as_slice());
        }
        let input_data = u_mode_api::cert::MeasurementRegisters { msmt_regs };
        let ctx = UmodeExecutionContext {
            input_data: Some(input_data),
            req: UmodeRequest::GetEvidence {
                csr_addr: csr_vaddr.bits(),
                csr_len,
                certout_addr: certout_vaddr.bits(),
                certout_len,
            },
            attestation: Some(attestation_mgr),
        };
        Self::execute_request(ctx)
    }

    fn reset(&mut self) -> Result<(), Error> {
        // Initialize umode CPU state to run at ELF entry.
        let mut arch = UmodeCpuArchState::default();
        // Unwrap okay: this is called after `Self::init()`.
        arch.umode_regs.sepc = *UMODE_ENTRY.get().unwrap();
        // Set cpu id as a0.
        arch.umode_regs
            .gprs
            .set_reg(GprIndex::A0, PerCpu::this_cpu().cpu_id().raw() as u64);
        // Set U-mode Input Region address as a1.
        arch.umode_regs
            .gprs
            .set_reg(GprIndex::A1, UMODE_INPUT_START);
        // Set U-mode Input Region size as a2.
        arch.umode_regs.gprs.set_reg(GprIndex::A2, UMODE_INPUT_SIZE);
        // sstatus set to 0 (by default) is actually okay.
        self.arch = arch;
        // Run task until it initializes itself and calls HypCall::NextOp().
        self.run(None)
            .map_err(Error::Exec)?
            .map_err(Error::Request)?;
        Ok(())
    }

    fn execute_request<T: DataInit>(exec_ctx: UmodeExecutionContext<T>) -> Result<u64, Error> {
        let this_cpu = PerCpu::this_cpu();
        let mut task = this_cpu.umode_task_mut();
        // Save request in registers.
        exec_ctx
            .req
            .to_registers(task.arch.umode_regs.gprs.a_regs_mut());
        if let Some(data) = exec_ctx.input_data {
            this_cpu
                .page_table()
                .copy_to_umode_input(data)
                .map_err(Error::HypMap)?;
        }
        let ret = task.run(exec_ctx.attestation);
        // Errors encountered while executing the operation are unrecoverable: reset U-mode.
        if ret.is_err() {
            println!("Umode error: {:?}", ret);
            // 1. restore memory to original state
            this_cpu.page_table().restore_umode();
            // 2. setup umode again.
            // Panic if we can't restore umode: the hypervisor is in an unrecoverable state.
            task.reset().expect("Failed to recover U-mode.");
        }
        if exec_ctx.input_data.is_some() {
            this_cpu.page_table().clear_umode_input();
        }
        let res = ret.map_err(Error::Exec)?;
        res.map_err(Error::Request)
    }

    fn handle_cdi_op(
        attestation: Option<&AttestationManager<sha2::Sha384>>,
        cdi_sel: CdiSel,
        cdi_op: CdiOp,
    ) -> Result<(), ExecError> {
        if let Some(attmgr) = attestation {
            let cdi = match cdi_sel {
                CdiSel::AttestationCurrent => Ok(attmgr.attestation_current_cdi()),
                _ => Err(ExecError::UnexpectedCdiOp(cdi_sel, cdi_op)),
            }?;
            match cdi_op {
                CdiOp::Id {
                    idout_addr,
                    idout_len,
                } => {
                    let id = cdi.id().map_err(ExecError::Cdi)?;
                    if idout_len as usize != id.len() {
                        return Err(ExecError::BufferSize(idout_len, id.len() as u64));
                    }
                    HypMap::copy_to_umode(RawAddr::supervisor_virt(idout_addr), &id)
                        .map_err(ExecError::UmodeAccess)
                }
                CdiOp::Sign {
                    msg_addr,
                    msg_len,
                    signout_addr,
                    signout_len,
                } => {
                    let mut msg_buf = [0u8; CDIOP_SIGN_MAXMSG];
                    if msg_len > CDIOP_SIGN_MAXMSG as u64 {
                        return Err(ExecError::BufferSize(msg_len, CDIOP_SIGN_MAXMSG as u64));
                    }
                    let msg = &mut msg_buf[0..msg_len as usize];
                    HypMap::copy_from_umode(msg, RawAddr::supervisor_virt(msg_addr))
                        .map_err(ExecError::UmodeAccess)?;
                    let signature = cdi.sign(msg).to_bytes();
                    if signout_len as usize != signature.len() {
                        return Err(ExecError::BufferSize(signout_len, signature.len() as u64));
                    }
                    HypMap::copy_to_umode(RawAddr::supervisor_virt(signout_addr), &signature)
                        .map_err(ExecError::UmodeAccess)
                }
            }
        } else {
            Err(ExecError::UnexpectedCdiOp(cdi_sel, cdi_op))
        }
    }

    fn handle_ecall(
        &mut self,
        attestation: Option<&AttestationManager<sha2::Sha384>>,
    ) -> ControlFlow<Result<OpResult, ExecError>> {
        let regs = self.arch.umode_regs.gprs.a_regs();
        let cflow = match HypCall::try_from_registers(regs) {
            Ok(hypercall) => match hypercall {
                HypCall::Panic => {
                    println!("U-mode panic!");
                    println!("{}", self.arch);
                    ControlFlow::Break(Err(ExecError::Panic))
                }
                HypCall::PutChar(byte) => {
                    if let Some(c) = char::from_u32(byte as u32) {
                        print!("{}", c);
                    }
                    // No return. Leave umode registers untouched and return.
                    ControlFlow::Continue(())
                }
                HypCall::Cdi { cdi_sel, cdi_op } => {
                    match Self::handle_cdi_op(attestation, cdi_sel, cdi_op) {
                        Ok(_) => ControlFlow::Continue(()),
                        Err(err) => ControlFlow::Break(Err(err)),
                    }
                }
                HypCall::NextOp(result) => ControlFlow::Break(Ok(result)),
            },
            Err(err) => {
                // Hypervisor could not parse the hypercall. Stop running umode and return error.
                ControlFlow::Break(Err(ExecError::Api(err)))
            }
        };
        // Increase SEPC to skip ecall on entry.
        self.arch.umode_regs.sepc += 4;
        cflow
    }

    /// Run until it exits
    fn run_to_exit(&mut self) {
        unsafe {
            // Safe to run umode code as it only touches memory assigned to it through umode mappings.
            _run_umode(&mut self.arch as *mut UmodeCpuArchState);
        }
        // Save off the trap information.
        self.arch.trap_csrs.scause = CSR.scause.get();
        self.arch.trap_csrs.stval = CSR.stval.get();
    }

    // Run `umode` until result is returned.
    fn run(
        &mut self,
        attestation: Option<&AttestationManager<sha2::Sha384>>,
    ) -> Result<OpResult, ExecError> {
        loop {
            self.run_to_exit();
            match Trap::from_scause(self.arch.trap_csrs.scause).unwrap() {
                Trap::Exception(UserEnvCall) => match self.handle_ecall(attestation) {
                    ControlFlow::Continue(_) => continue,
                    ControlFlow::Break(res) => break res,
                },
                _ => {
                    println!("Unexpected U-mode Trap:");
                    println!("{}", self.arch);
                    break Err(ExecError::UnexpectedTrap);
                }
            }
        }
    }
}
