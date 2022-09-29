// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use attestation::{
    certificate::Certificate, measurement::AttestationManager, request::CertReq,
    Error as AttestationError, MAX_CSR_LEN,
};
use core::{mem, ops::ControlFlow, slice};
use der::Decode;
use drivers::{
    imsic::*, iommu::*, pci::PciBarPage, pci::PciDevice, pci::PcieRoot, pmu::PmuInfo, CpuId,
    CpuInfo, MAX_CPUS,
};
use page_tracking::{HypPageAlloc, PageList, PageTracker};
use riscv_page_tables::{GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::*;
use riscv_regs::{DecodedInstruction, Exception, GprIndex, Instruction, Trap};
use s_mode_utils::print::*;
use sbi::{Error as SbiError, *};

use crate::guest_tracking::{GuestStateGuard, GuestVm, Guests, Result as GuestTrackingResult};
use crate::smp;
use crate::vm_cpu::{
    ActiveVmCpu, VmCpuSharedArea, VmCpuSharedState, VmCpuSharedStateRef, VmCpuStatus, VmCpuTrap,
    VmCpus, VM_CPU_BYTES, VM_CPU_SHARED_LAYOUT, VM_CPU_SHARED_PAGES,
};
use crate::vm_pages::Error as VmPagesError;
use crate::vm_pages::{
    ActiveVmPages, AnyVmPages, InstructionFetchError, PageFaultType, VmPages, VmPagesRef,
    VmRegionList, TVM_REGION_LIST_PAGES, TVM_STATE_PAGES,
};

#[derive(Debug)]
pub enum Error {
    AttestationManagerCreationFailed(attestation::Error),
    AttestationManagerFinalizeFailed(attestation::Error),
    MissingImsicAddress,
    AliasedImsicAddresses,
    MissingBootCpu,
}

pub type Result<T> = core::result::Result<T, Error>;

// What we report ourselves as in sbi_get_sbi_impl_id(). Just pick something unclaimed so no one
// confuses us with BBL/OpenSBI.
const SBI_IMPL_ID_SALUS: u64 = 7;

/// Possible MMIO instructions.
#[derive(Clone, Copy, Debug)]
pub enum MmioOpcode {
    Load64,
    Load32,
    Load32U,
    Load16,
    Load16U,
    Load8,
    Load8U,
    Store64,
    Store32,
    Store16,
    Store8,
}

impl MmioOpcode {
    /// Returns if the MMIO operation is a load.
    pub fn is_load(&self) -> bool {
        use MmioOpcode::*;
        matches!(
            self,
            Load8 | Load8U | Load16 | Load16U | Load32 | Load32U | Load64
        )
    }
}

/// A decoded MMIO operation.
#[derive(Clone, Copy, Debug)]
pub struct MmioOperation {
    opcode: MmioOpcode,
    register: GprIndex,
    len: usize,
}

impl MmioOperation {
    /// Creates an `MmioOperation` from `instruction` if the MMIO is supported using that instruction.
    fn from_instruction(instruction: DecodedInstruction) -> Option<Self> {
        use Instruction::*;
        let (opcode, reg_index) = match instruction.instruction() {
            Lb(i) => (MmioOpcode::Load8, i.rd()),
            Lh(i) => (MmioOpcode::Load16, i.rd()),
            Lw(i) => (MmioOpcode::Load32, i.rd()),
            Lbu(i) => (MmioOpcode::Load8U, i.rd()),
            Lhu(i) => (MmioOpcode::Load16U, i.rd()),
            Lwu(i) => (MmioOpcode::Load32U, i.rd()),
            Ld(i) => (MmioOpcode::Load64, i.rd()),
            Sb(s) => (MmioOpcode::Store8, s.rs2()),
            Sh(s) => (MmioOpcode::Store16, s.rs2()),
            Sw(s) => (MmioOpcode::Store32, s.rs2()),
            Sd(s) => (MmioOpcode::Store64, s.rs2()),
            _ => {
                return None;
            }
        };
        Some(Self {
            opcode,
            register: GprIndex::from_raw(reg_index).unwrap(),
            len: instruction.len(),
        })
    }

    /// Returns the operation as a `MmioOpcode`.
    pub fn opcode(&self) -> MmioOpcode {
        self.opcode
    }

    /// Returns the target register for the operation. Either 'rd' for load instructions, or 'rs2' for
    /// store instructions.
    pub fn register(&self) -> GprIndex {
        self.register
    }

    /// Returns the length of the raw instruction.
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Exit cause for a TVM from the TvmCpuRun ECALL.
#[derive(Clone, Copy, Debug)]
pub enum VmExitCause {
    FatalEcall(SbiMessage),
    ResumableEcall(SbiMessage),
    PageFault(Exception, GuestPageAddr),
    MmioFault(MmioOperation, GuestPhysAddr),
    Wfi(DecodedInstruction),
    UnhandledTrap(u64),
}

impl VmExitCause {
    /// Returns if the exit cause is fatal.
    pub fn is_fatal(&self) -> bool {
        use VmExitCause::*;
        matches!(self, FatalEcall(_) | UnhandledTrap(_))
    }
}

#[derive(Clone, Copy, Debug)]
enum EcallError {
    Sbi(SbiError),
    PageFault(PageFaultType, Exception, GuestPhysAddr),
}

type EcallResult<T> = core::result::Result<T, EcallError>;

impl From<VmPagesError> for EcallError {
    fn from(error: VmPagesError) -> EcallError {
        match error {
            VmPagesError::PageFault(pf, e, addr) => EcallError::PageFault(pf, e, addr),
            // TODO: Map individual error types. InvalidAddress is likely not the right value for
            // each error.
            _ => EcallError::Sbi(SbiError::InvalidAddress),
        }
    }
}

impl From<AttestationError> for EcallError {
    fn from(error: AttestationError) -> EcallError {
        match error {
            AttestationError::InvalidMeasurementRegisterDescIndex(_) => {
                EcallError::Sbi(SbiError::Failed)
            }
            // TODO: Map individual error types.
            // InvalidParam may not be the right value for each error.
            _ => EcallError::Sbi(SbiError::InvalidParam),
        }
    }
}

impl From<SbiError> for EcallError {
    fn from(error: SbiError) -> EcallError {
        EcallError::Sbi(error)
    }
}

#[derive(Clone, Copy, Debug)]
enum EcallAction {
    LegacyOk,
    Unhandled,
    Continue(SbiReturn),
    Break(VmExitCause, SbiReturn),
    Retry(VmExitCause),
}

impl From<EcallResult<u64>> for EcallAction {
    fn from(result: EcallResult<u64>) -> EcallAction {
        use EcallAction::*;
        match result {
            Ok(val) => Continue(SbiReturn::success(val)),
            Err(EcallError::Sbi(e)) => Continue(e.into()),
            Err(EcallError::PageFault(pf, e, addr)) => {
                use PageFaultType::*;
                match pf {
                    // Unhandleable page faults or page faults in MMIO space just result in an
                    // error to the caller.
                    Unmapped | Mmio => Continue(SbiReturn::from(SbiError::InvalidAddress)),
                    Confidential | Shared => {
                        let addr = PageAddr::with_round_down(addr, PageSize::Size4k);
                        Retry(VmExitCause::PageFault(e, addr))
                    }
                }
            }
        }
    }
}

type AttestationSha384 = AttestationManager<sha2::Sha384>;

/// A VM that is being run.
pub struct Vm<T: GuestStagePagingMode> {
    vcpus: VmCpus,
    vm_pages: VmPages<T>,
    guests: Option<Guests<T>>,
    attestation_mgr: AttestationSha384,
}

impl<T: GuestStagePagingMode> Vm<T> {
    /// Creates a new `Vm` using the given initial page table and vCPU tracking table.
    pub fn new(vm_pages: VmPages<T>, vcpus: VmCpus) -> Result<Self> {
        let vm_id = vm_pages.page_owner_id().raw();
        Ok(Self {
            vcpus,
            vm_pages,
            guests: None,
            attestation_mgr: AttestationSha384::new(
                // Fake compound device identifiers (DICE CDI)
                // TODO Get the CDI from e.g. the TSM driver.
                b"RANDOMATTESTATIONCDI",
                b"RANDOMSEALINGCDI",
                vm_id,
                const_oid::db::rfc5912::ID_SHA_384,
            )
            .map_err(Error::AttestationManagerCreationFailed)?,
        })
    }

    /// Same as `new()`, but with a `Guests` for tracking nested guest VMs.
    pub fn with_guest_tracking(
        vm_pages: VmPages<T>,
        vcpus: VmCpus,
        guests: Guests<T>,
    ) -> Result<Self> {
        let mut this = Self::new(vm_pages, vcpus)?;
        this.guests = Some(guests);
        Ok(this)
    }

    /// Returns this VM's ID.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.vm_pages.page_owner_id()
    }

    /// Returns the `PageTracker` singleton.
    pub fn page_tracker(&self) -> PageTracker {
        self.vm_pages.page_tracker()
    }

    // Check that all vCPUs have been assigned an IMSIC address and that there are no conflicts
    // prior to guest finalization.
    fn validate_imsic_addrs(&self) -> Result<()> {
        let vm_pages: AnyVmPages<T> = self.vm_pages.as_ref();
        if vm_pages.imsic_geometry().is_some() {
            for i in 0..self.vcpus.num_vcpus() {
                // Check that this vCPU was assigned an IMSIC address.
                let location = if let Ok(vcpu) = self.vcpus.get_vcpu(i as u64) {
                    vcpu.get_imsic_location().ok_or(Error::MissingImsicAddress)
                } else {
                    continue;
                }?;

                // And make sure it doesn't conflict with any other vCPUs.
                for j in i + 1..self.vcpus.num_vcpus() {
                    let other = self
                        .vcpus
                        .get_vcpu(j as u64)
                        .ok()
                        .and_then(|v| v.get_imsic_location());
                    if other == Some(location) {
                        return Err(Error::AliasedImsicAddresses);
                    }
                }
            }
        }

        Ok(())
    }

    /// Completes intialization of the `Vm`. The caller must ensure that it is currently in the
    /// initializing state.
    pub fn finalize(&mut self) -> Result<()> {
        // Enable the boot vCPU; we assume this is always vCPU 0.
        //
        // TODO: Should we allow a non-0 boot vCPU to be specified when creating the TVM?
        let (sepc, arg) = {
            let mut vcpu = self
                .vcpus
                .power_on_vcpu(0)
                .map_err(|_| Error::MissingBootCpu)?;
            vcpu.latch_entry_args()
        };
        // Latch and measure the entry point of the boot vCPU.
        self.attestation_mgr.set_epc(sepc);
        self.attestation_mgr.set_arg(arg);
        self.validate_imsic_addrs()?;
        self.attestation_mgr
            .finalize()
            .map_err(Error::AttestationManagerFinalizeFailed)
    }
}

impl<T: GuestStagePagingMode> Drop for Vm<T> {
    fn drop(&mut self) {
        // Recursively destroy this VM's children before we drop() this VM so that any donated pages
        // are guaranteed to have been returned before we destroy this VM's page table. This could
        // also be done by implementing Drop for Guests, but doing explicitly avoids relying on
        // struct field ordering for proper drop() ordering.
        self.guests = None;

        let page_tracker = self.page_tracker();
        page_tracker.rm_active_guest(self.page_owner_id());
    }
}

/// A reference to a `Vm` in a particular state `S` that exposes the appropriate functionality
/// for a VM in that state.
pub struct VmRef<'a, T: GuestStagePagingMode, S> {
    inner: GuestStateGuard<'a, T, S>,
}

impl<'a, T: GuestStagePagingMode, S> VmRef<'a, T, S> {
    /// Creates a new `VmRef` from a guarded reference to a `Vm` in state `S`.
    pub fn new(inner: GuestStateGuard<'a, T, S>) -> Self {
        VmRef { inner }
    }

    // Returns a reference to the wrapped `Vm`.
    fn vm(&self) -> &Vm<T> {
        self.inner.vm()
    }

    // Returns a `VmPagesRef` to this VM's `VmPages` in the same state as this VM.
    fn vm_pages(&self) -> VmPagesRef<T, S> {
        self.vm().vm_pages.as_ref()
    }

    // Returns this VM's ID.
    fn page_owner_id(&self) -> PageOwnerId {
        self.vm().page_owner_id()
    }

    // Returns the `PageTracker` singleton.
    fn page_tracker(&self) -> PageTracker {
        self.vm().page_tracker()
    }

    // Returns a reference to this VM's `AttestationManager`.
    fn attestation_mgr(&self) -> &AttestationSha384 {
        &self.vm().attestation_mgr
    }

    /// Convenience function to turn a raw u64 from an SBI call to a `GuestPageAddr`.
    fn guest_addr_from_raw(&self, guest_addr: u64) -> EcallResult<GuestPageAddr> {
        PageAddr::new(RawAddr::guest(guest_addr, self.page_owner_id()))
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))
    }

    /// Gets the location of the specified vCPU's virtualized IMSIC.
    fn get_vcpu_imsic_location(&self, vcpu_id: u64) -> EcallResult<ImsicLocation> {
        let vcpu = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        vcpu.get_imsic_location()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))
    }
}

pub enum VmStateAny {}
/// Represents a VM that may be in any state.
pub type AnyVm<'a, T> = VmRef<'a, T, VmStateAny>;

pub enum VmStateInitializing {}
/// Represents a VM in the process of construction.
pub type InitializingVm<'a, T> = VmRef<'a, T, VmStateInitializing>;

impl<'a, T: GuestStagePagingMode> InitializingVm<'a, T> {
    /// Adds a vCPU to this VM.
    fn add_vcpu(&self, vcpu_id: u64, shared_area: VmCpuSharedArea) -> EcallResult<()> {
        self.vm()
            .vcpus
            .add_vcpu(vcpu_id, shared_area)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    /// Sets the location of the specified vCPU's virtualized IMSIC.
    fn set_vcpu_imsic_location(&self, vcpu_id: u64, location: ImsicLocation) -> EcallResult<()> {
        let geometry = self
            .vm_pages()
            .imsic_geometry()
            .ok_or(EcallError::Sbi(SbiError::NotSupported))?;
        if !geometry.location_is_valid(location) {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let mut vcpu = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        if vcpu.get_imsic_location().is_some() {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        vcpu.set_imsic_location(location);
        Ok(())
    }

    /// Binds the specified vCPU to an IMSIC interrupt file.
    fn bind_vcpu(&self, vcpu_id: u64, interrupt_file: ImsicFileId) -> EcallResult<()> {
        let mut vcpu = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        // TODO: Bind to this (physical) CPU as well.
        vcpu.set_interrupt_file(interrupt_file);
        Ok(())
    }
}

pub enum VmStateFinalized {}
/// Represents a finalized, or runnable, VM.
pub type FinalizedVm<'a, T> = VmRef<'a, T, VmStateFinalized>;

impl<'a, T: GuestStagePagingMode> FinalizedVm<'a, T> {
    /// Sets the entry point of the specified vCPU and makes it runnable.
    fn start_vcpu(&self, vcpu_id: u64, start_addr: u64, opaque: u64) -> EcallResult<()> {
        let mut vcpu = self
            .vm()
            .vcpus
            .power_on_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        vcpu.set_entry_args(start_addr, opaque);
        Ok(())
    }

    /// Gets the state of the specified vCPU.
    fn get_vcpu_status(&self, vcpu_id: u64) -> EcallResult<u64> {
        let vcpu_status = self
            .vm()
            .vcpus
            .get_vcpu_status(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        let status = match vcpu_status {
            VmCpuStatus::Runnable | VmCpuStatus::Running => HartState::Started,
            VmCpuStatus::PoweredOff => HartState::Stopped,
            VmCpuStatus::NotPresent => {
                return Err(EcallError::Sbi(SbiError::InvalidParam));
            }
        };
        Ok(status as u64)
    }

    fn process_decoded_instruction(
        active_vcpu: &mut ActiveVmCpu<T>,
        inst: DecodedInstruction,
    ) -> ControlFlow<VmExitCause> {
        // We only emulate WFI and Crrss for PMU registers for now.
        // Everything else gets redirected as an illegal instruction exception.
        match inst.instruction() {
            Instruction::Csrrs(csr_type)
            if let Ok(value) = active_vcpu.pmu().get_cached_csr_value(csr_type.csr().into())  => {
                // Unwrap ok: rd is 12-bits and instruction is already decoded.
                let gpr_index = GprIndex::from_raw(csr_type.rd()).unwrap();
                active_vcpu.set_gpr(gpr_index, value);
                active_vcpu.inc_sepc(inst.len() as u64);
                ControlFlow::Continue(())
            }
            Instruction::Wfi => {
                // Just advance SEPC and exit. We place no constraints on when a vCPU
                // may be resumed from WFI since, per the privileged spec, it's only
                // a hint and it's perfectly valid for WFI to be a no-op.
                active_vcpu.inc_sepc(inst.len() as u64);
                ControlFlow::Break(VmExitCause::Wfi(inst))
            }
            _ => {
                active_vcpu.inject_exception(
                    Exception::IllegalInstruction,
                    inst.raw() as u64,
                );
                ControlFlow::Continue(())
            }
        }
    }

    /// Run this guest until an unhandled exit is encountered.
    fn run_vcpu(&self, vcpu_id: u64, parent_vcpu: Option<&mut ActiveVmCpu<T>>) -> EcallResult<u64> {
        // Take the vCPU out of self.vcpus, giving us exclusive ownership.
        let mut active_vcpu = self
            .vm()
            .vcpus
            .activate_vcpu(vcpu_id, self.vm_pages(), parent_vcpu)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        // Run until there's an exit we can't handle.
        let cause = loop {
            let exit = active_vcpu.run();
            use SbiReturnType::*;
            match exit {
                VmCpuTrap::Ecall(Some(sbi_msg)) => {
                    match self.handle_ecall(sbi_msg, &mut active_vcpu) {
                        EcallAction::LegacyOk => {
                            active_vcpu.set_ecall_result(Legacy(0));
                        }
                        EcallAction::Unhandled => {
                            active_vcpu.set_ecall_result(Standard(SbiReturn::from(
                                SbiError::NotSupported,
                            )));
                        }
                        EcallAction::Continue(sbi_ret) => {
                            active_vcpu.set_ecall_result(Standard(sbi_ret));
                        }
                        EcallAction::Break(reason, sbi_ret) => {
                            active_vcpu.set_ecall_result(Standard(sbi_ret));
                            break reason;
                        }
                        EcallAction::Retry(reason) => {
                            break reason;
                        }
                    }
                }
                VmCpuTrap::Ecall(None) => {
                    // Unrecognized ECALL, return an error.
                    active_vcpu.set_ecall_result(Standard(SbiReturn::from(SbiError::NotSupported)));
                }
                VmCpuTrap::PageFault {
                    exception,
                    fault_addr,
                    fault_pc,
                    priv_level,
                } => {
                    let pf = active_vcpu
                        .active_pages()
                        .get_page_fault_cause(exception, fault_addr);
                    use PageFaultType::*;
                    match pf {
                        Confidential | Shared => {
                            break VmExitCause::PageFault(
                                exception,
                                PageAddr::with_round_down(fault_addr, PageSize::Size4k),
                            );
                        }
                        Mmio => {
                            // We need the faulting instruction for MMIO faults.
                            use InstructionFetchError::*;
                            let inst = match active_vcpu
                                .active_pages()
                                .fetch_guest_instruction(fault_pc, priv_level)
                            {
                                Ok(inst) => inst,
                                Err(FetchFault) => {
                                    // If we took a fault while trying to fetch the instruction,
                                    // then something must have happened in between the load/store
                                    // page fault and now which caused the PC to become invalid.
                                    // Let the VM retry the instruction so that we can take and
                                    // handle the instruction fetch fault instead.
                                    continue;
                                }
                                Err(FailedDecode(raw_inst)) => {
                                    active_vcpu.inject_exception(
                                        Exception::IllegalInstruction,
                                        raw_inst as u64,
                                    );
                                    continue;
                                }
                            };

                            // Make sure that the instruction is actually valid for MMIO.
                            let mmio_op = match MmioOperation::from_instruction(inst) {
                                Some(mmio_op) => mmio_op,
                                None => {
                                    active_vcpu.inject_exception(
                                        Exception::IllegalInstruction,
                                        inst.raw() as u64,
                                    );
                                    continue;
                                }
                            };

                            break VmExitCause::MmioFault(mmio_op, fault_addr);
                        }
                        Unmapped => {
                            break VmExitCause::UnhandledTrap(
                                Trap::Exception(exception).to_scause(),
                            );
                        }
                    };
                }
                VmCpuTrap::VirtualInstruction {
                    fault_pc,
                    priv_level,
                } => {
                    use InstructionFetchError::*;
                    let inst = match active_vcpu
                        .active_pages()
                        .fetch_guest_instruction(fault_pc, priv_level)
                    {
                        Ok(inst) => inst,
                        Err(FetchFault) => {
                            continue;
                        }
                        Err(FailedDecode(raw_inst)) => {
                            active_vcpu
                                .inject_exception(Exception::IllegalInstruction, raw_inst as u64);
                            continue;
                        }
                    };

                    match Self::process_decoded_instruction(&mut active_vcpu, inst) {
                        ControlFlow::Continue(_) => continue,
                        ControlFlow::Break(reason) => break reason,
                    };
                }
                VmCpuTrap::DelegatedException { exception, stval } => {
                    active_vcpu.inject_exception(exception, stval);
                }
                VmCpuTrap::Other(ref trap_csrs) => {
                    println!("Unhandled guest exit, SCAUSE = 0x{:08x}", trap_csrs.scause);
                    break VmExitCause::UnhandledTrap(trap_csrs.scause);
                }
            }
        };

        active_vcpu.exit(cause);

        Ok(cause.is_fatal().into())
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&self, msg: SbiMessage, active_vcpu: &mut ActiveVmCpu<T>) -> EcallAction {
        match msg {
            SbiMessage::PutChar(c) => {
                // put char - legacy command
                print!("{}", c as u8 as char);
                EcallAction::LegacyOk
            }
            SbiMessage::Reset(ResetFunction::Reset { .. }) => {
                EcallAction::Break(VmExitCause::FatalEcall(msg), SbiReturn::success(0))
            }
            SbiMessage::Base(base_func) => EcallAction::Continue(self.handle_base_msg(base_func)),
            SbiMessage::HartState(hsm_func) => self.handle_hart_state_msg(hsm_func),
            SbiMessage::TeeHost(host_func) => self.handle_tee_host_msg(host_func, active_vcpu),
            SbiMessage::TeeInterrupt(interrupt_func) => {
                self.handle_tee_interrupt_msg(interrupt_func, active_vcpu.active_pages())
            }
            SbiMessage::TeeGuest(guest_func) => self.handle_tee_guest_msg(guest_func),
            SbiMessage::Attestation(attestation_func) => {
                self.handle_attestation_msg(attestation_func, active_vcpu.active_pages())
            }
            SbiMessage::Pmu(pmu_func) => self.handle_pmu_msg(pmu_func, active_vcpu).into(),
        }
    }

    fn handle_pmu_msg(
        &self,
        pmu_func: PmuFunction,
        active_vcpu: &mut ActiveVmCpu<T>,
    ) -> EcallResult<u64> {
        use PmuFunction::*;
        fn get_num_counters() -> EcallResult<u64> {
            let pmu_info = PmuInfo::get()?;
            Ok(pmu_info.get_num_counters())
        }

        fn get_counter_info(counter_index: u64) -> EcallResult<u64> {
            let pmu_info = PmuInfo::get()?;
            let info = pmu_info.get_counter_info(counter_index)?;
            Ok(info.raw())
        }

        fn start_counters<T: GuestStagePagingMode>(
            counter_index: u64,
            counter_mask: u64,
            start_flags: PmuCounterStartFlags,
            initial_value: u64,
            active_vcpu: &mut ActiveVmCpu<T>,
        ) -> EcallResult<u64> {
            let counter_mask = active_vcpu
                .pmu()
                .get_startable_counter_range(counter_index, counter_mask)?;
            let result = sbi::api::pmu::start_counters(
                counter_index,
                counter_mask,
                start_flags,
                initial_value,
            );
            // Special case "already started" to handle counters that are autostarted following configuration.
            // Examples of such counters include the legacy timer and insret.
            if result.is_ok() || matches!(result, Err(SbiError::AlreadyStarted)) {
                active_vcpu
                    .pmu()
                    .update_started_counters(counter_index, counter_mask);
            }
            result.map(|_| 0).map_err(EcallError::from)
        }

        fn stop_counters<T: GuestStagePagingMode>(
            counter_index: u64,
            counter_mask: u64,
            stop_flags: PmuCounterStopFlags,
            active_vcpu: &mut ActiveVmCpu<T>,
        ) -> EcallResult<u64> {
            let counter_mask = active_vcpu
                .pmu()
                .get_stoppable_counter_range(counter_index, counter_mask)?;
            let result = sbi::api::pmu::stop_counters(counter_index, counter_mask, stop_flags);
            // Special case "already stopped" to handle counters that can be reset following a stop
            if result.is_ok()
                || (matches!(result, Err(SbiError::AlreadyStopped)) && stop_flags.is_reset_flag())
            {
                active_vcpu
                    .pmu()
                    .update_stopped_counters(counter_index, counter_mask, stop_flags);
            }
            result.map(|_| 0).map_err(EcallError::from)
        }

        fn configure_counters<T: GuestStagePagingMode>(
            counter_index: u64,
            counter_mask: u64,
            config_flags: PmuCounterConfigFlags,
            event_type: PmuEventType,
            event_data: u64,
            active_vcpu: &mut ActiveVmCpu<T>,
        ) -> EcallResult<u64> {
            let config_flags = config_flags.set_sinh().set_minh();
            let counter_mask = active_vcpu.pmu().get_configurable_counter_range(
                counter_index,
                counter_mask,
                config_flags,
            )?;
            let platform_counter_index = sbi::api::pmu::configure_matching_counters(
                counter_index,
                counter_mask,
                config_flags,
                event_type,
                event_data,
            )?;
            active_vcpu.pmu().update_configured_counter(
                platform_counter_index,
                config_flags,
                event_type,
                event_data,
            )?;
            Ok(platform_counter_index)
        }

        match pmu_func {
            GetNumCounters => get_num_counters(),
            GetCounterInfo(counter_index) => get_counter_info(counter_index),
            StartCounters {
                counter_index,
                counter_mask,
                start_flags,
                initial_value,
            } => start_counters(
                counter_index,
                counter_mask,
                start_flags,
                initial_value,
                active_vcpu,
            ),
            StopCounters {
                counter_index,
                counter_mask,
                stop_flags,
            } => stop_counters(counter_index, counter_mask, stop_flags, active_vcpu),
            ConfigureMatchingCounters {
                counter_index,
                counter_mask,
                config_flags,
                event_type,
                event_data,
            } => configure_counters(
                counter_index,
                counter_mask,
                config_flags,
                event_type,
                event_data,
                active_vcpu,
            ),
            ReadFirmwareCounter(_) => Err(EcallError::Sbi(SbiError::NotSupported)),
        }
    }

    fn handle_base_msg(&self, base_func: BaseFunction) -> SbiReturn {
        use BaseFunction::*;
        let ret = match base_func {
            GetSpecificationVersion => 3,
            GetImplementationID => SBI_IMPL_ID_SALUS,
            GetImplementationVersion => 0,
            ProbeSbiExtension(ext) => match ext {
                sbi::EXT_PUT_CHAR
                | sbi::EXT_BASE
                | sbi::EXT_HART_STATE
                | sbi::EXT_RESET
                | sbi::EXT_TEE_HOST
                | sbi::EXT_TEE_INTERRUPT
                | sbi::EXT_TEE_GUEST
                | sbi::EXT_ATTESTATION => 1,
                sbi::EXT_PMU if PmuInfo::get().is_ok() => 1,
                _ => 0,
            },
            // TODO: 0 is valid result for the GetMachine* SBI calls but we should probably
            // report real values here.
            _ => 0,
        };
        SbiReturn::success(ret)
    }

    fn handle_hart_state_msg(&self, hsm_func: StateFunction) -> EcallAction {
        use StateFunction::*;
        match hsm_func {
            HartStart {
                hart_id,
                start_addr,
                opaque,
            } => match self.start_vcpu(hart_id, start_addr, opaque) {
                Ok(()) => {
                    // Forward the ECALL along, but mask the initial PC/A1 values.
                    let msg = SbiMessage::HartState(StateFunction::HartStart {
                        hart_id,
                        start_addr: 0,
                        opaque: 0,
                    });
                    EcallAction::Break(VmExitCause::ResumableEcall(msg), SbiReturn::success(0))
                }
                result @ Err(_) => result.map(|_| 0).into(),
            },
            HartStop => EcallAction::Break(
                VmExitCause::FatalEcall(SbiMessage::HartState(hsm_func)),
                SbiReturn::success(0),
            ),
            HartStatus { hart_id } => self.get_vcpu_status(hart_id).into(),
            _ => EcallAction::Unhandled,
        }
    }

    fn handle_tee_host_msg(
        &self,
        host_func: TeeHostFunction,
        active_vcpu: &mut ActiveVmCpu<T>,
    ) -> EcallAction {
        use TeeHostFunction::*;
        match host_func {
            TsmGetInfo { dest_addr, len } => self
                .get_tsm_info(dest_addr, len, active_vcpu.active_pages())
                .into(),
            TvmCreate { params_addr, len } => self
                .add_guest(params_addr, len, active_vcpu.active_pages())
                .into(),
            TvmDestroy { guest_id } => self.destroy_guest(guest_id).into(),
            TsmConvertPages {
                page_addr,
                page_type,
                num_pages,
            } => self.convert_pages(page_addr, page_type, num_pages).into(),
            TsmReclaimPages {
                page_addr,
                page_type,
                num_pages,
            } => self.reclaim_pages(page_addr, page_type, num_pages).into(),
            TsmInitiateFence => self.initiate_fence(active_vcpu).into(),
            TsmLocalFence => self.local_fence(active_vcpu).into(),
            AddPageTablePages {
                guest_id,
                page_addr,
                num_pages,
            } => self
                .guest_add_page_table_pages(guest_id, page_addr, num_pages)
                .into(),
            TvmAddConfidentialMemoryRegion {
                guest_id,
                guest_addr,
                len,
            } => self
                .guest_add_confidential_memory_region(guest_id, guest_addr, len)
                .into(),
            TvmAddEmulatedMmioRegion {
                guest_id,
                guest_addr,
                len,
            } => self.guest_add_mmio_region(guest_id, guest_addr, len).into(),
            TvmAddZeroPages {
                guest_id,
                page_addr,
                page_type,
                num_pages,
                guest_addr,
            } => self
                .guest_add_zero_pages(guest_id, page_addr, page_type, num_pages, guest_addr)
                .into(),
            TvmAddMeasuredPages {
                guest_id,
                src_addr,
                dest_addr,
                page_type,
                num_pages,
                guest_addr,
            } => self
                .guest_add_measured_pages(
                    guest_id,
                    src_addr,
                    dest_addr,
                    page_type,
                    num_pages,
                    guest_addr,
                    active_vcpu.active_pages(),
                )
                .into(),
            Finalize { guest_id } => self.guest_finalize(guest_id).into(),
            TvmCpuRun { guest_id, vcpu_id } => {
                self.guest_run_vcpu(guest_id, vcpu_id, active_vcpu).into()
            }
            TvmCpuNumRegisterSets { guest_id } => {
                self.guest_num_vcpu_register_sets(guest_id).into()
            }
            TvmCpuGetRegisterSet { guest_id, index } => {
                self.guest_get_vcpu_register_set(guest_id, index).into()
            }
            TvmCpuCreate {
                guest_id,
                vcpu_id,
                shared_page_addr,
            } => self
                .guest_add_vcpu(guest_id, vcpu_id, shared_page_addr)
                .into(),
            TvmAddSharedMemoryRegion {
                guest_id,
                guest_addr,
                len,
            } => self
                .guest_add_shared_memory_region(guest_id, guest_addr, len)
                .into(),
            TvmAddSharedPages {
                guest_id,
                page_addr,
                page_type,
                num_pages,
                guest_addr,
            } => self
                .guest_add_shared_pages(guest_id, page_addr, page_type, num_pages, guest_addr)
                .into(),
        }
    }

    fn handle_attestation_msg(
        &self,
        attestation_func: AttestationFunction,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallAction {
        use AttestationFunction::*;
        match attestation_func {
            GetCapabilities {
                caps_addr_out,
                caps_size,
            } => self
                .get_attestation_capabilities(caps_addr_out, caps_size as usize, active_pages)
                .into(),
            GetEvidence {
                cert_request_addr,
                cert_request_size,
                request_data_addr,
                evidence_format,
                cert_addr_out,
                cert_size,
            } => self
                .guest_get_evidence(
                    cert_request_addr,
                    cert_request_size as usize,
                    request_data_addr,
                    evidence_format,
                    cert_addr_out,
                    cert_size as usize,
                    active_pages,
                )
                .into(),

            ExtendMeasurement {
                measurement_data_addr,
                measurement_data_size,
                measurement_index,
            } => self
                .guest_extend_measurement(
                    measurement_data_addr,
                    measurement_data_size as usize,
                    measurement_index as usize,
                    active_pages,
                )
                .into(),

            ReadMeasurement {
                measurement_data_addr_out,
                measurement_data_size,
                measurement_index,
            } => self
                .guest_read_measurement(
                    measurement_data_addr_out,
                    measurement_data_size as usize,
                    measurement_index as usize,
                    active_pages,
                )
                .into(),
        }
    }

    fn handle_tee_interrupt_msg(
        &self,
        interrupt_func: TeeInterruptFunction,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallAction {
        use TeeInterruptFunction::*;
        match interrupt_func {
            TvmAiaInit {
                tvm_id,
                params_addr,
                len,
            } => self
                .guest_aia_init(tvm_id, params_addr, len as usize, active_pages)
                .into(),
            TvmCpuSetImsicAddr {
                tvm_id,
                vcpu_id,
                imsic_addr,
            } => self
                .guest_set_vcpu_imsic_addr(tvm_id, vcpu_id, imsic_addr)
                .into(),
            TsmConvertImsic { imsic_addr } => self.convert_imsic(imsic_addr).into(),
            TsmReclaimImsic { imsic_addr } => self.reclaim_imsic(imsic_addr).into(),
        }
    }

    fn get_tsm_info(
        &self,
        dest_addr: u64,
        len: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        let dest_addr = RawAddr::guest(dest_addr, self.page_owner_id());
        if len < mem::size_of::<sbi::TsmInfo>() as u64 {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let len = mem::size_of::<sbi::TsmInfo>();
        // Since we're the hypervisor we're ready from boot.
        let tsm_info = sbi::TsmInfo {
            tsm_state: sbi::TsmState::TsmReady,
            tsm_version: 0,
            tvm_state_pages: TVM_STATE_PAGES,
            tvm_max_vcpus: MAX_CPUS as u64,
            tvm_bytes_per_vcpu: VM_CPU_BYTES,
        };
        // Safety: &tsm_info points to len bytes of initialized memory.
        let tsm_info_bytes: &[u8] =
            unsafe { slice::from_raw_parts((&tsm_info as *const sbi::TsmInfo).cast(), len) };
        active_pages
            .copy_to_guest(dest_addr, tsm_info_bytes)
            .map_err(EcallError::from)?;
        Ok(len as u64)
    }

    /// Converts `num_pages` starting at guest physical address `page_addr` to confidential memory.
    fn convert_pages(
        &self,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
    ) -> EcallResult<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO: Support converting hugepages.
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let page_addr = self.guest_addr_from_raw(page_addr)?;
        self.vm_pages()
            .convert_pages(page_addr, num_pages)
            .map_err(EcallError::from)?;
        Ok(num_pages)
    }

    /// Reclaims `num_pages` of confidential memory starting at guest physical address `page_addr`.
    fn reclaim_pages(
        &self,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
    ) -> EcallResult<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO: Support converting hugepages.
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let page_addr = self.guest_addr_from_raw(page_addr)?;
        self.vm_pages()
            .reclaim_pages(page_addr, num_pages)
            .map_err(EcallError::from)?;
        Ok(num_pages)
    }

    fn initiate_fence(&self, active_vcpu: &mut ActiveVmCpu<T>) -> EcallResult<u64> {
        self.vm_pages().initiate_fence().map_err(EcallError::from)?;
        active_vcpu.sync_tlb();
        Ok(0)
    }

    fn local_fence(&self, active_vcpu: &mut ActiveVmCpu<T>) -> EcallResult<u64> {
        // Nothing to do here other than to check if there's TLB maintenance to be done.
        active_vcpu.sync_tlb();
        Ok(0)
    }

    fn guests(&self) -> Option<&Guests<T>> {
        self.vm().guests.as_ref()
    }

    fn add_guest(
        &self,
        params_addr: u64,
        len: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        if self.guests().is_none() {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        // Read the params from the VM's address space.
        let params_addr = RawAddr::guest(params_addr, self.page_owner_id());
        if len < mem::size_of::<sbi::TvmCreateParams>() as u64 {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let mut param_bytes = [0u8; mem::size_of::<sbi::TvmCreateParams>()];
        active_pages
            .copy_from_guest(param_bytes.as_mut_slice(), params_addr)
            .map_err(EcallError::from)?;

        // Safety: `param_bytes` points to `size_of::<TvmCreateParams>()` contiguous, initialized
        // bytes.
        let params: sbi::TvmCreateParams =
            unsafe { core::ptr::read_unaligned(param_bytes.as_slice().as_ptr().cast()) };

        // Now create the VM, claiming the pages that the host donated to us.
        let page_root_addr = self.guest_addr_from_raw(params.tvm_page_directory_addr)?;
        let state_addr = self.guest_addr_from_raw(params.tvm_state_addr)?;
        let vcpu_addr = self.guest_addr_from_raw(params.tvm_vcpu_addr)?;
        let num_vcpu_pages = PageSize::num_4k_pages(params.tvm_num_vcpus * VM_CPU_BYTES);
        let (guest_vm, state_page) = self
            .vm_pages()
            .create_guest_vm(page_root_addr, state_addr, vcpu_addr, num_vcpu_pages)
            .map_err(EcallError::from)?;
        let id = guest_vm.page_owner_id();

        let guest = GuestVm::new(guest_vm, state_page);
        self.guests()
            .and_then(|g| g.add(guest).ok())
            .ok_or(EcallError::Sbi(SbiError::Failed))?;

        Ok(id.raw())
    }

    fn destroy_guest(&self, guest_id: u64) -> EcallResult<u64> {
        let guest_id = PageOwnerId::new(guest_id).ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        self.guests()
            .and_then(|g| g.remove(guest_id).ok())
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(0)
    }

    /// Retrieves the guest VM with the ID `guest_id`.
    fn guest_by_id(&self, guest_id: u64) -> EcallResult<GuestVm<T>> {
        let guest_id = PageOwnerId::new(guest_id).ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let guest = self
            .guests()
            .and_then(|g| g.get(guest_id))
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(guest)
    }

    // converts the given guest from init to running
    fn guest_finalize(&self, guest_id: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        guest
            .finalize()
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(0)
    }

    // Returns the number of register sets in the vCPU shared-memory state area for `guest_id`.
    fn guest_num_vcpu_register_sets(&self, guest_id: u64) -> EcallResult<u64> {
        // All guests have the same layout since we don't support customization of virtualized
        // features currently, but make sure that the specified guest_id is at least valid.
        self.guest_by_id(guest_id)?;
        Ok(VM_CPU_SHARED_LAYOUT.len() as u64)
    }

    // Get the location of the register set at `index` in the vCPU shared-memory state area for
    // `guest_id`.
    fn guest_get_vcpu_register_set(&self, guest_id: u64, index: u64) -> EcallResult<u64> {
        // As above, make sure the `guest_id` is valid even though the layout is uniform (for now).
        self.guest_by_id(guest_id)?;
        let regset = VM_CPU_SHARED_LAYOUT
            .get(index as usize)
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(u32::from(*regset) as u64)
    }

    // Adds a vCPU with `vcpu_id` to a guest VM with a shared-memory state area at
    // `shared_page_addr`.
    fn guest_add_vcpu(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        shared_page_addr: u64,
    ) -> EcallResult<u64> {
        // Pin the pages that the VM wants to use for the shared state buffer.
        let shared_page_addr = self.guest_addr_from_raw(shared_page_addr)?;
        let pin = self
            .vm_pages()
            .pin_shared_pages(shared_page_addr, VM_CPU_SHARED_PAGES)
            .map_err(EcallError::from)?;
        let shared_area = VmCpuSharedArea::from_pinned_pages(pin)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidAddress))?;

        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        guest_vm.add_vcpu(vcpu_id, shared_area)?;
        Ok(0)
    }

    /// Runs a guest VM's vCPU.
    fn guest_run_vcpu(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        active_vcpu: &mut ActiveVmCpu<T>,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        guest_vm.run_vcpu(vcpu_id, Some(active_vcpu))
    }

    fn guest_add_page_table_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        num_pages: u64,
    ) -> EcallResult<u64> {
        let from_page_addr = self.guest_addr_from_raw(from_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        self.vm_pages()
            .add_pte_pages_to(from_page_addr, num_pages, guest.as_any_vm().vm_pages())
            .map_err(EcallError::from)?;

        Ok(0)
    }

    /// Adds a region of confidential memory to the specified guest.
    fn guest_add_confidential_memory_region(
        &self,
        guest_id: u64,
        guest_addr: u64,
        len: u64,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        guest_vm
            .vm_pages()
            .add_confidential_memory_region(page_addr, len)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    /// Adds an emulated MMIO region to the specified guest.
    fn guest_add_mmio_region(&self, guest_id: u64, guest_addr: u64, len: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        guest_vm
            .vm_pages()
            .add_mmio_region(page_addr, len)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn guest_add_zero_pages(
        &self,
        guest_id: u64,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
    ) -> EcallResult<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        let from_page_addr = self.guest_addr_from_raw(page_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        let to_page_addr = PageAddr::new(RawAddr::guest(guest_addr, guest.page_owner_id()))
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Zero pages may be added to either running or initialized VMs.
        self.vm_pages()
            .add_zero_pages_to(
                from_page_addr,
                num_pages,
                guest.as_any_vm().vm_pages(),
                to_page_addr,
            )
            .map_err(EcallError::from)?;

        Ok(num_pages)
    }

    #[allow(clippy::too_many_arguments)]
    fn guest_add_measured_pages(
        &self,
        guest_id: u64,
        src_addr: u64,
        dest_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        let src_page_addr = self.guest_addr_from_raw(src_addr)?;
        let from_page_addr = self.guest_addr_from_raw(dest_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let to_page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        active_pages
            .copy_and_add_data_pages_builder(
                src_page_addr,
                from_page_addr,
                num_pages,
                guest_vm.vm_pages(),
                to_page_addr,
                guest_vm.attestation_mgr(),
            )
            .map_err(EcallError::from)?;

        Ok(num_pages)
    }

    fn get_attestation_capabilities(
        &self,
        caps_addr_out: u64,
        caps_size: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        if caps_size < core::mem::size_of::<AttestationCapabilities>() {
            return Err(EcallError::Sbi(SbiError::InsufficientBufferCapacity));
        }
        let caps = self
            .attestation_mgr()
            .capabilities()
            .map_err(EcallError::from)?;

        let caps_gpa = RawAddr::guest(caps_addr_out, self.page_owner_id());
        // Safety: &caps points to an AttestationCapabilities structure, as
        // specified by the attestation manager `capabilities()` method.
        let caps_bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                (&caps as *const AttestationCapabilities).cast(),
                core::mem::size_of::<AttestationCapabilities>(),
            )
        };
        active_pages
            .copy_to_guest(caps_gpa, caps_bytes)
            .map_err(EcallError::from)?;

        Ok(0)
    }

    #[allow(clippy::too_many_arguments)]
    fn guest_get_evidence(
        &self,
        cert_request_addr: u64,
        cert_request_size: usize,
        _request_data_addr: u64,
        _evidence_format: u64,
        cert_addr_out: u64,
        cert_size: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        if cert_request_size > MAX_CSR_LEN {
            return Err(EcallError::Sbi(SbiError::InsufficientBufferCapacity));
        }

        let mut csr_bytes = [0u8; MAX_CSR_LEN];
        let csr_gpa = RawAddr::guest(cert_request_addr, self.page_owner_id());
        active_pages
            .copy_from_guest(&mut csr_bytes.as_mut_slice()[..cert_request_size], csr_gpa)
            .map_err(EcallError::from)?;

        let csr = CertReq::from_der(&csr_bytes[..cert_request_size])
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        println!(
            "CSR version {:?} Signature algorithm {:?}",
            csr.info.version, csr.algorithm.oid
        );

        csr.verify()
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;

        let cert_gpa = RawAddr::guest(cert_addr_out, self.page_owner_id());
        let mut cert_bytes_buffer = [0u8; sbi::api::attestation::MAX_CERT_SIZE];
        let cert_bytes =
            Certificate::from_csr(&csr, self.attestation_mgr(), &mut cert_bytes_buffer)
                .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        let cert_bytes_len = cert_bytes.len();

        // Check that the guest gave us enough space
        if cert_size < cert_bytes_len {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        active_pages
            .copy_to_guest(cert_gpa, cert_bytes)
            .map_err(EcallError::from)?;

        Ok(cert_bytes_len as u64)
    }

    fn guest_extend_measurement(
        &self,
        msmt_addr: u64,
        msmt_size: usize,
        index: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        let caps = self
            .attestation_mgr()
            .capabilities()
            .map_err(EcallError::from)?;

        // Check that the measurement buffer size matches exactly the hash
        // algorithm one.
        if msmt_size != caps.hash_algorithm.size() {
            return Err(EcallError::Sbi(SbiError::InsufficientBufferCapacity));
        }

        let mut measurement_data = [0u8; sbi::MAX_HASH_SIZE];
        let measurement_data_gpa = RawAddr::guest(msmt_addr, self.page_owner_id());
        active_pages
            .copy_from_guest(
                &mut measurement_data.as_mut_slice()[..msmt_size],
                measurement_data_gpa,
            )
            .map_err(EcallError::from)?;

        // Ask the attestation manager extend the measurement register.
        // If the passed index is invalid, `extend_msmt_register` will return
        // an error.
        self.attestation_mgr()
            .extend_msmt_register(
                (index as u8).try_into().map_err(EcallError::from)?,
                &measurement_data,
                None,
            )
            .map_err(EcallError::from)?;

        Ok(0)
    }

    fn guest_read_measurement(
        &self,
        msmt_addr: u64,
        msmt_size: usize,
        index: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        let caps = self
            .attestation_mgr()
            .capabilities()
            .map_err(EcallError::from)?;

        let measurement_data = self
            .attestation_mgr()
            .read_msmt_register((index as u8).try_into().map_err(EcallError::from)?)
            .map_err(EcallError::from)?;

        // Check if the passed buffer size is large enough.
        if msmt_size < caps.hash_algorithm.size() || msmt_size < measurement_data.len() {
            return Err(EcallError::Sbi(SbiError::InsufficientBufferCapacity));
        }

        // Copy the measurement register data into the guest.
        let measurement_data_gpa = RawAddr::guest(msmt_addr, self.page_owner_id());
        active_pages
            .copy_to_guest(measurement_data_gpa, measurement_data.as_slice())
            .map_err(EcallError::from)?;

        Ok(measurement_data.len() as u64)
    }

    fn guest_aia_init(
        &self,
        guest_id: u64,
        params_addr: u64,
        params_len: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        // Read the params from the VM's address space.
        let params_addr = RawAddr::guest(params_addr, self.page_owner_id());
        let mut param_bytes = [0u8; mem::size_of::<sbi::TvmAiaParams>()];
        if params_len < param_bytes.len() {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        active_pages
            .copy_from_guest(param_bytes.as_mut_slice(), params_addr)
            .map_err(EcallError::from)?;
        // Safety: `param_bytes` points to `size_of::<TvmAiaParams>()` contiguous, initialized
        // bytes.
        let params: sbi::TvmAiaParams =
            unsafe { core::ptr::read_unaligned(param_bytes.as_slice().as_ptr().cast()) };

        // Validate the supplied IMSIC geometry. We don't support nested IMSIC virtualization, so
        // reject attempts to configure a guest with guest interrupt files.
        if params.guests_per_hart != 0 {
            return Err(EcallError::Sbi(SbiError::NotSupported));
        }
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let base_addr = guest_vm.guest_addr_from_raw(params.imsic_base_addr)?;
        let geometry = ImsicGeometry::new(
            base_addr,
            params.group_index_bits,
            params.group_index_shift,
            params.hart_index_bits,
            params.guest_index_bits,
            0,
        )
        .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        guest_vm
            .vm_pages()
            .set_imsic_geometry(geometry)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn guest_set_vcpu_imsic_addr(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        imsic_addr: u64,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let imsic_addr = guest_vm.guest_addr_from_raw(imsic_addr)?;
        let geometry = guest_vm
            .vm_pages()
            .imsic_geometry()
            .ok_or(EcallError::Sbi(SbiError::NotSupported))?;
        let location = geometry
            .addr_to_location(imsic_addr)
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        // We'll verify that there's no aliasing between locations during finalize().
        guest_vm.set_vcpu_imsic_location(vcpu_id, location)?;
        Ok(0)
    }

    fn convert_imsic(&self, imsic_addr: u64) -> EcallResult<u64> {
        let imsic_addr = self.guest_addr_from_raw(imsic_addr)?;
        self.vm_pages()
            .convert_imsic(imsic_addr)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn reclaim_imsic(&self, imsic_addr: u64) -> EcallResult<u64> {
        let imsic_addr = self.guest_addr_from_raw(imsic_addr)?;
        self.vm_pages()
            .reclaim_imsic(imsic_addr)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn guest_add_shared_memory_region(
        &self,
        guest_id: u64,
        guest_addr: u64,
        len: u64,
    ) -> EcallResult<u64> {
        let page_addr = self.guest_addr_from_raw(guest_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        guest_vm
            .vm_pages()
            .add_shared_memory_region(page_addr, len)
            .map_err(EcallError::from)?;
        Ok(len)
    }

    fn guest_add_shared_pages(
        &self,
        guest_id: u64,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
    ) -> EcallResult<u64> {
        if page_type != TsmPageType::Page4k || num_pages == 0 {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let page_addr = self.guest_addr_from_raw(page_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest.as_any_vm();
        let guest_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        self.vm_pages()
            .add_shared_pages_to(page_addr, num_pages, guest_vm.vm_pages(), guest_addr)
            .map_err(EcallError::from)?;

        Ok(num_pages)
    }

    fn handle_tee_guest_msg(&self, guest_func: TeeGuestFunction) -> EcallAction {
        use TeeGuestFunction::*;
        match guest_func {
            AddMemoryRegion {
                region_type,
                addr,
                len,
            } => {
                let result = self.add_memory_region(region_type, addr, len);
                // Notify the host if the call succeeded.
                match result {
                    Ok(r) => EcallAction::Break(
                        VmExitCause::ResumableEcall(SbiMessage::TeeGuest(guest_func)),
                        SbiReturn::success(r),
                    ),
                    Err(_) => result.into(),
                }
            }
        }
    }

    fn add_memory_region(
        &self,
        region_type: TeeMemoryRegion,
        addr: u64,
        len: u64,
    ) -> EcallResult<u64> {
        let addr = self.guest_addr_from_raw(addr)?;
        use TeeMemoryRegion::*;
        match region_type {
            Shared => self
                .vm_pages()
                .add_shared_memory_region(addr, len)
                .map_err(EcallError::from),
            EmulatedMmio => self
                .vm_pages()
                .add_mmio_region(addr, len)
                .map_err(EcallError::from),
            _ => Err(EcallError::Sbi(SbiError::InvalidParam)),
        }?;
        Ok(0)
    }
}

/// Errors encountered during MMIO emulation.
#[derive(Clone, Copy, Debug)]
enum MmioEmulationError {
    FailedDecode(u32),
    InvalidInstruction(Instruction),
    InvalidAddress(u64),
}

/// Represents the special VM that serves as the host for the system.
pub struct HostVm<T: GuestStagePagingMode> {
    inner: GuestVm<T>,
    vcpu_shared: Vec<VmCpuSharedState>,
}

impl<T: GuestStagePagingMode> HostVm<T> {
    /// Creates an initializing host VM with an expected guest physical address space size of
    /// `host_gpa_size` from the hypervisor page allocator. Returns the remaining free pages
    /// from the allocator, along with the newly constructed `HostVm`.
    pub fn from_hyp_mem(
        mut hyp_mem: HypPageAlloc,
        host_gpa_size: u64,
    ) -> (PageList<Page<ConvertedClean>>, Self) {
        let root_table_pages =
            hyp_mem.take_pages_for_host_state_with_alignment(4, T::TOP_LEVEL_ALIGN);
        let num_pte_pages = T::max_pte_pages(host_gpa_size / PageSize::Size4k as u64);
        let pte_pages = hyp_mem
            .take_pages_for_host_state(num_pte_pages as usize)
            .into_iter();
        let vm_state_page = hyp_mem.take_pages_for_host_state(1);
        let guest_tracking_pages = hyp_mem.take_pages_for_host_state(2);
        let region_vec_pages = hyp_mem.take_pages_for_host_state(TVM_REGION_LIST_PAGES as usize);

        // Pages for the array of vCPUs.
        let num_cpus = CpuInfo::get().num_cpus();
        let num_vcpu_pages = PageSize::num_4k_pages(VM_CPU_BYTES * num_cpus as u64);
        let vcpus_pages = hyp_mem.take_pages_for_host_state(num_vcpu_pages as usize);

        let imsic_geometry = Imsic::get().host_vm_geometry();
        // Reserve MSI page table pages if we have an IOMMU.
        let msi_table_pages = Iommu::get().map(|_| {
            let msi_table_size = MsiPageTable::required_table_size(&imsic_geometry);
            hyp_mem.take_pages_for_host_state_with_alignment(
                PageSize::num_4k_pages(msi_table_size) as usize,
                msi_table_size,
            )
        });

        let (page_tracker, host_pages) = PageTracker::from(hyp_mem, T::TOP_LEVEL_ALIGN);
        let root =
            GuestStagePageTable::new(root_table_pages, PageOwnerId::host(), page_tracker.clone())
                .unwrap();
        let region_vec = VmRegionList::new(region_vec_pages, page_tracker.clone());
        let vm_pages = VmPages::new(root, region_vec, 0);
        let init_pages = vm_pages.as_ref();
        init_pages.set_imsic_geometry(imsic_geometry).unwrap();
        for p in pte_pages {
            init_pages.add_pte_page(p).unwrap();
        }
        if let Some(pages) = msi_table_pages {
            init_pages.add_iommu_context(pages).unwrap();
        }

        let vm = Vm::with_guest_tracking(
            vm_pages,
            VmCpus::new(PageOwnerId::host(), vcpus_pages, page_tracker.clone()).unwrap(),
            Guests::new(guest_tracking_pages, page_tracker),
        )
        .unwrap();
        let mut this = Self {
            inner: GuestVm::new(vm, vm_state_page.into_iter().next().unwrap()),
            vcpu_shared: Vec::with_capacity(num_cpus),
        };

        {
            let init_vm = this.inner.as_initializing_vm().unwrap();
            let imsic = Imsic::get();
            for i in 0..num_cpus {
                this.vcpu_shared.push(VmCpuSharedState::default());
                // Safety: This slot in vcpu_shared points to a valid VmCpuSharedState struct
                // that is guaranteed to live as long as the containing HostVm structure.
                let shared_area =
                    unsafe { VmCpuSharedArea::new(&mut this.vcpu_shared[i]) }.unwrap();
                init_vm.add_vcpu(i as u64, shared_area).unwrap();
                let imsic_loc = imsic.supervisor_file_location(CpuId::new(i)).unwrap();
                init_vm
                    .set_vcpu_imsic_location(i as u64, imsic_loc)
                    .unwrap();
            }
        }

        (host_pages, this)
    }

    // Returns a reference to the shared-state buffer for the given vCPU.
    fn vcpu_state(&self, vcpu_id: u64) -> Option<VmCpuSharedStateRef> {
        let ptr = self.vcpu_shared.get(vcpu_id as usize)? as *const _;
        // Safety: ptr refers to a valid VmCpuSharedState struct with the same lifetime as `self`.
        Some(unsafe { VmCpuSharedStateRef::new(ptr as *mut _) })
    }

    /// Sets the launch arguments (entry point and FDT) for the host vCPU.
    pub fn set_launch_args(&self, entry_addr: GuestPhysAddr, fdt_addr: GuestPhysAddr) {
        // Unwrap ok: there must be a CPU 0.
        let vcpu = self.vcpu_state(0).unwrap();
        vcpu.set_sepc(entry_addr.bits());
        vcpu.set_gpr(GprIndex::A1, fdt_addr.bits());
    }

    /// Adds a region of confidential memory to the host VM.
    pub fn add_confidential_memory_region(&mut self, addr: GuestPageAddr, len: u64) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages()
            .add_confidential_memory_region(addr, len)
            .unwrap();
    }

    /// Adds an emulated MMIO region to the host VM.
    pub fn add_mmio_region(&mut self, addr: GuestPageAddr, len: u64) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages().add_mmio_region(addr, len).unwrap();
    }

    /// Adds a PCI BAR memory region to the host VM.
    pub fn add_pci_region(&mut self, addr: GuestPageAddr, len: u64) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages().add_pci_region(addr, len).unwrap();
    }

    /// Adds data pages that are measured and mapped to the page tables for the host. Requires
    /// that the GPA map the SPA in T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    pub fn add_measured_pages<I, S, M>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = Page<S>>,
        S: Assignable<M>,
        M: MeasureRequirement,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        let page_tracker = vm.page_tracker();
        // Unwrap ok since we've donate sufficient PT pages to map the entire address space up front.
        let mapper = vm
            .vm_pages()
            .map_measured_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper
                .map_page(vm_addr, mappable, vm.attestation_mgr())
                .unwrap();
        }
    }

    /// Add zero pages to the host page tables. Requires that the GPA map the SPA in
    /// T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    pub fn add_zero_pages<I>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = Page<ConvertedClean>>,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        let page_tracker = vm.page_tracker();
        // Unwrap ok since we've donate sufficient PT pages to map the entire address space up front.
        let mapper = vm
            .vm_pages()
            .map_zero_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    /// Adds the IMSIC pages for `cpu` to the host. The first page in `pages` is set as the
    /// host's interrupt file for `cpu` while the remaining pages are added as guest interrupt
    /// files for the host to assign.
    pub fn add_imsic_pages<I>(&mut self, cpu: CpuId, pages: I)
    where
        I: ExactSizeIterator<Item = ImsicGuestPage<ConvertedClean>>,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        // We assigned an IMSIC geometry and vCPU IMSIC locations in `from_hyp_mem()`.
        let location = vm.get_vcpu_imsic_location(cpu.raw() as u64).unwrap();
        let to_addr = vm
            .vm_pages()
            .imsic_geometry()
            .and_then(|g| g.location_to_addr(location))
            .unwrap();
        // Unwrap ok since we've donated sufficient PT pages to map the entire address space up
        // front.
        let mapper = vm
            .vm_pages()
            .map_imsic_pages(to_addr, pages.len() as u64)
            .unwrap();
        let page_tracker = vm.page_tracker();
        for (i, (page, vm_addr)) in pages.zip(to_addr.iter_from()).enumerate() {
            if i == 0 {
                // Set the first page as the vCPU's supervisor-level interrupt file.
                vm.bind_vcpu(cpu.raw() as u64, page.location().file())
                    .unwrap();
            }

            // Map in the remaining guest interrupt file pages as guest files for the host VM.
            //
            // TODO: This is sufficient for the host VM since vCPUs are never migrated, but in the
            // event we need to support nested IMSIC virtualization for guest VMs we'll need to be
            // able to bind a vCPU to multiple interrupt files.
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    /// Add PCI BAR pages to the host page tables.
    pub fn add_pci_pages<I>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = PciBarPage<ConvertedClean>>,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        let page_tracker = vm.page_tracker();
        // Unwrap ok since we've donated sufficient PT pages to map the entire address space up
        // front.
        let mapper = vm
            .vm_pages()
            .map_pci_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    /// Attaches the given PCI device to the host VM.
    pub fn attach_pci_device(&self, dev: &mut PciDevice) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages().attach_pci_device(dev).unwrap();
    }

    /// Completes intialization of the host VM, making it runnable.
    pub fn finalize(&self) -> GuestTrackingResult<()> {
        self.inner.finalize()
    }

    /// Run the host VM's vCPU with ID `vcpu_id`. Does not return.
    pub fn run(&self, vcpu_id: u64) {
        let vm = self.inner.as_finalized_vm().unwrap();
        loop {
            // Wait until this vCPU is ready to run.
            while !self.vcpu_is_runnable(vcpu_id) {
                smp::wfi();
            }

            // Unwrap ok: vcpu_id must exist if it's runnable.
            let vcpu = self.vcpu_state(vcpu_id).unwrap();
            // Run until we shut down, or this vCPU stops.
            loop {
                vm.run_vcpu(vcpu_id, None).unwrap();
                let scause = vcpu.scause();
                if let Ok(Trap::Exception(e)) = Trap::from_scause(scause) {
                    use Exception::*;
                    match e {
                        VirtualSupervisorEnvCall => {
                            // Read the ECALL arguments written to the A* regs in shared memory.
                            let mut a_regs = [0u64; 8];
                            for (i, reg) in a_regs.iter_mut().enumerate() {
                                // Unwrap ok: A0-A7 are valid GPR indices.
                                let index =
                                    GprIndex::from_raw(GprIndex::A0 as u32 + i as u32).unwrap();
                                *reg = vcpu.gpr(index);
                            }
                            use SbiMessage::*;
                            match SbiMessage::from_regs(&a_regs) {
                                Ok(Reset(_)) => {
                                    println!("Host VM requested shutdown");
                                    return;
                                }
                                Ok(HartState(StateFunction::HartStart { hart_id, .. })) => {
                                    smp::send_ipi(CpuId::new(hart_id as usize));
                                }
                                Ok(HartState(StateFunction::HartStop)) => {
                                    break;
                                }
                                _ => {
                                    println!("Unhandled ECALL from host");
                                    return;
                                }
                            }
                        }
                        GuestLoadPageFault | GuestStorePageFault => {
                            if let Err(err) = self.handle_page_fault(vcpu_id) {
                                println!("Unhandled page fault: {:?}", err);
                                return;
                            }
                        }
                        _ => {
                            println!("Unhandled host VM exception {:?}", e);
                            return;
                        }
                    }
                } else {
                    println!("Unexpected host VM trap (SCAUSE = 0x{:x})", scause);
                    return;
                }
            }
        }
    }

    /// Returns if the vCPU with `vcpu_id` is runnable.
    fn vcpu_is_runnable(&self, vcpu_id: u64) -> bool {
        let vm = self.inner.as_finalized_vm().unwrap();
        vm.get_vcpu_status(vcpu_id)
            .is_ok_and(|s| *s == HartState::Started as u64)
    }

    // Handle a page fault for `vcpu_id`.
    fn handle_page_fault(&self, vcpu_id: u64) -> core::result::Result<(), MmioEmulationError> {
        // For now, the only thing we're expecting is MMIO emulation faults in PCI config space.
        let vcpu = self.vcpu_state(vcpu_id).unwrap();
        let addr = (vcpu.htval() << 2) | (vcpu.stval() & 0x3);
        let pci = PcieRoot::get();
        if addr < pci.config_space().base().bits() {
            return Err(MmioEmulationError::InvalidAddress(addr));
        }
        let offset = addr - pci.config_space().base().bits();
        if offset > pci.config_space().length_bytes() {
            return Err(MmioEmulationError::InvalidAddress(addr));
        }

        // Figure out from HTINST what the MMIO operation was. We know the source/destination is
        // always A0.
        let raw_inst = vcpu.htinst() as u32;
        let inst = DecodedInstruction::from_raw(raw_inst)
            .map_err(|_| MmioEmulationError::FailedDecode(raw_inst))?;
        use Instruction::*;
        let (write, width) = match inst.instruction() {
            Lb(_) | Lbu(_) => (false, 1),
            Lh(_) | Lhu(_) => (false, 2),
            Lw(_) | Lwu(_) => (false, 4),
            Ld(_) => (false, 8),
            Sb(_) => (true, 1),
            Sh(_) => (true, 2),
            Sw(_) => (true, 4),
            Sd(_) => (true, 8),
            i => {
                return Err(MmioEmulationError::InvalidInstruction(i));
            }
        };

        let vm = self.inner.as_finalized_vm().unwrap();
        let page_tracker = vm.page_tracker();
        let guest_id = vm.page_owner_id();
        if write {
            let val = vcpu.gpr(GprIndex::A0);
            pci.emulate_config_write(offset, val, width, page_tracker, guest_id);
        } else {
            let val = pci.emulate_config_read(offset, width, page_tracker, guest_id);
            vcpu.set_gpr(GprIndex::A0, val);
        }

        Ok(())
    }
}
