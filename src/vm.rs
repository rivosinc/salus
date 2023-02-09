// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use attestation::{AttestationManager, Error as AttestationError, TcgPcrIndex};
use core::{mem, num::Wrapping, ops::ControlFlow, ops::Neg, slice};
use drivers::{imsic::*, pmu::PmuInfo};
use page_tracking::collections::PageBox;
use page_tracking::{LockedPageList, PageList, PageTracker, TlbVersion};
use riscv_page_tables::{GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::*;
use riscv_regs::{DecodedInstruction, Exception, GprIndex, Instruction, Interrupt, Trap, CSR};
use s_mode_utils::print::*;
use sbi_rs::{salus::*, Error as SbiError, *};
use spin::Once;
use u_mode_api::Error as UmodeApiError;

use crate::guest_tracking::{GuestStateGuard, GuestVm, Guests};
use crate::hyp_map::{UmodeSlotId, UmodeSlotPerm};
use crate::umode::{Error as UmodeError, ExecError, UmodeTask};
use crate::vm_cpu::{ActiveVmCpu, VmCpu, VmCpuParent, VmCpuStatus, VmCpuTrap, VmCpus, VM_CPUS_MAX};
use crate::vm_pages::Error as VmPagesError;
use crate::vm_pages::{
    ActiveVmPages, AnyVmPages, GuestUmodeMapping, InstructionFetchError, PageFaultType, VmPages,
    VmPagesRef,
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

// Report ourselves as being SBI v1.0 compliant.
const SBI_SPEC_MAJOR_VERSION_SHIFT: u64 = 24;
const SBI_SPEC_VERSION: u64 = 1 << SBI_SPEC_MAJOR_VERSION_SHIFT;

// The number of pages required for `NaclShmem`.
const NACL_SHMEM_PAGES: u64 =
    PageSize::num_4k_pages(core::mem::size_of::<sbi_rs::NaclShmem>() as u64);

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
    BlockingEcall(SbiMessage, TlbVersion),
    ForwardedEcall(SbiMessage),
    PageFault(Exception, GuestPageAddr),
    MmioFault(MmioOperation, GuestPhysAddr),
    Wfi(DecodedInstruction),
    HostInterrupt(Interrupt),
    UnhandledTrap(u64),
}

impl VmExitCause {
    /// Returns if the exit cause is fatal.
    pub fn is_fatal(&self) -> bool {
        use VmExitCause::*;
        matches!(self, FatalEcall(_) | UnhandledTrap(_))
    }

    // Returns if the vCPU can immediately resume execution from the exit.
    fn is_resumable(&self) -> bool {
        use VmExitCause::*;
        !self.is_fatal() && !matches!(self, BlockingEcall(..))
    }
}

/// Possible error conditions from handling an ECALL from a VM.
#[derive(Clone, Copy, Debug)]
pub enum EcallError {
    /// A standard SBI error.
    Sbi(SbiError),
    /// The requested action would cause a page fault.
    PageFault(PageFaultType, Exception, GuestPhysAddr),
}

pub type EcallResult<T> = core::result::Result<T, EcallError>;

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

impl From<UmodeApiError> for EcallError {
    fn from(error: UmodeApiError) -> EcallError {
        match error {
            UmodeApiError::InvalidArgument => EcallError::Sbi(SbiError::InvalidParam),
            _ => EcallError::Sbi(SbiError::Failed),
        }
    }
}

impl From<UmodeError> for EcallError {
    fn from(error: UmodeError) -> EcallError {
        match error {
            UmodeError::Exec(ExecError::Umode(api_error)) => api_error.into(),
            _ => EcallError::Sbi(SbiError::Failed),
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
    Unhandled,
    Continue(SbiReturn),
    Break(VmExitCause, SbiReturn),
    Retry(VmExitCause),
    Forward(SbiMessage),
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
                    Unmapped | Mmio | Imsic => Continue(SbiReturn::from(SbiError::InvalidAddress)),
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
    // Only used by Host VM to track guest VMs.
    guests: Option<Guests<T>>,
    attestation_mgr: AttestationSha384,
    // Latched htimedelta (-CSR_TIME) at the time of first VCPU run.
    htimedelta: Once<u64>,
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
            htimedelta: Once::new(),
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
            for i in 0..VM_CPUS_MAX {
                // Check that this vCPU was assigned an IMSIC address.
                let location = if let Ok(vcpu) = self.vcpus.get_vcpu(i as u64) {
                    vcpu.get_imsic_location().ok_or(Error::MissingImsicAddress)
                } else {
                    continue;
                }?;

                // And make sure it doesn't conflict with any other vCPUs.
                for j in i + 1..VM_CPUS_MAX {
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

    /// Completes intialization of the `Vm`, setting the entry point of the VM to `entry_sepc` and
    /// and `entry_arg`. The caller must ensure that it is currently in the initializing state.
    pub fn finalize(&mut self, entry_sepc: u64, entry_arg: u64) -> Result<()> {
        // Enable the boot vCPU; we assume this is always vCPU 0.
        //
        // TODO: Should we allow a non-0 boot vCPU to be specified when creating the TVM?
        self.vcpus
            .get_vcpu(0)
            .and_then(|v| v.power_on(entry_sepc, entry_arg))
            .map_err(|_| Error::MissingBootCpu)?;
        // Measure the entry point of the boot vCPU.
        self.attestation_mgr.set_epc(entry_sepc);
        self.attestation_mgr.set_arg(entry_arg);
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

    /// Returns a `VmPagesRef` to this VM's `VmPages` in the same state as this VM.
    pub fn vm_pages(&self) -> VmPagesRef<T, S> {
        self.vm().vm_pages.as_ref()
    }

    /// Returns this VM's ID.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.vm().page_owner_id()
    }

    /// Returns the `PageTracker` singleton.
    pub fn page_tracker(&self) -> PageTracker {
        self.vm().page_tracker()
    }

    /// Returns a reference to this VM's `AttestationManager`.
    pub fn attestation_mgr(&self) -> &AttestationSha384 {
        &self.vm().attestation_mgr
    }

    // Convenience function to turn a raw u64 from an SBI call to a `GuestPageAddr`.
    fn guest_addr_from_raw(&self, guest_addr: u64) -> EcallResult<GuestPageAddr> {
        PageAddr::new(RawAddr::guest(guest_addr, self.page_owner_id()))
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))
    }

    /// Gets the location of the specified vCPU's virtualized IMSIC.
    pub fn get_vcpu_imsic_location(&self, vcpu_id: u64) -> EcallResult<ImsicLocation> {
        let vcpu = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        vcpu.get_imsic_location()
            .ok_or(EcallError::Sbi(SbiError::NotSupported))
    }

    // Gets the address of `vcpu_id`'s IMSIC in guest physical address space. Shortcut for
    // `get_vcpu_imisc_location()` + translating the location.
    fn get_vcpu_imsic_addr(&self, vcpu_id: u64) -> EcallResult<GuestPageAddr> {
        let location = self.get_vcpu_imsic_location(vcpu_id)?;
        self.vm_pages()
            .imsic_geometry()
            .and_then(|g| g.location_to_addr(location))
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
    pub fn add_vcpu(&self, vcpu_box: PageBox<VmCpu>) -> EcallResult<()> {
        self.vm()
            .vcpus
            .add_vcpu(vcpu_box)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    /// Sets the location of the specified vCPU's virtualized IMSIC.
    pub fn set_vcpu_imsic_location(
        &self,
        vcpu_id: u64,
        location: ImsicLocation,
    ) -> EcallResult<()> {
        let geometry = self
            .vm_pages()
            .imsic_geometry()
            .ok_or(EcallError::Sbi(SbiError::NotSupported))?;
        if !geometry.location_is_valid(location) {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|v| v.enable_imsic_virtualization(location, geometry.guests_per_hart()))
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }
}

pub enum VmStateFinalized {}
/// Represents a finalized, or runnable, VM.
pub type FinalizedVm<'a, T> = VmRef<'a, T, VmStateFinalized>;

impl<'a, T: GuestStagePagingMode> FinalizedVm<'a, T> {
    // Sets the entry point of the specified vCPU and makes it runnable.
    fn start_vcpu(&self, vcpu_id: u64, start_addr: u64, opaque: u64) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|v| v.power_on(start_addr, opaque))
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    /// Gets the state of the specified vCPU.
    pub fn get_vcpu_status(&self, vcpu_id: u64) -> EcallResult<u64> {
        let vcpu_status = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?
            .status();
        use VmCpuStatus::*;
        let status = match vcpu_status {
            Runnable | Running | Blocked(_) => HartState::Started,
            PoweredOff => HartState::Stopped,
        };
        Ok(status as u64)
    }

    /// Set htimedelta for the current VM and all of it's VCPU's.
    fn set_vcpu_htimedelta(&self) -> u64 {
        let htimedelta: u64 = Wrapping(CSR.hpmcounter[1].get_value()).neg().0;

        for i in 0..VM_CPUS_MAX {
            // Set htimedelta for all VCPUs.
            if let Ok(vcpu) = self.vm().vcpus.get_vcpu(i as u64) {
                vcpu.set_htimedelta(htimedelta);
            }
        }

        htimedelta
    }

    /// Sets htimedelta for the VM.
    fn set_htimedelta(&self) {
        self.vm()
            .htimedelta
            .call_once(|| self.set_vcpu_htimedelta());
    }

    /// Run `vcpu_id` until an unhandled exit is encountered. Save/restore `host_context` on entry/exit
    /// from the vCPU being run.
    pub fn run_vcpu(&self, vcpu_id: u64, host_context: VmCpuParent) -> EcallResult<u64> {
        let vcpu = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;

        // Set htimedelta for ALL VCPU's of the VM.
        self.set_htimedelta();

        // Activate the vCPU, giving us exclusive ownership over the ability to run it.
        let mut active_vcpu = vcpu
            .activate(self.vm_pages(), host_context)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        // Run until there's an exit we can't handle.
        let cause = loop {
            let exit = active_vcpu.run();
            use SbiReturnType::*;
            match exit {
                VmCpuTrap::Ecall(Some(sbi_msg)) => {
                    match self.handle_ecall(sbi_msg, &mut active_vcpu) {
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
                        EcallAction::Forward(sbi_msg) => {
                            break VmExitCause::ForwardedEcall(sbi_msg);
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
                        Confidential | Shared | Imsic => {
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

                    match self.handle_virtual_instruction(inst, &mut active_vcpu) {
                        ControlFlow::Continue(_) => continue,
                        ControlFlow::Break(reason) => break reason,
                    };
                }
                VmCpuTrap::DelegatedException { exception, stval } => {
                    active_vcpu.inject_exception(exception, stval);
                }
                VmCpuTrap::OtherException(ref trap_csrs) => {
                    println!("Unhandled guest exit, SCAUSE = 0x{:08x}", trap_csrs.scause);
                    break VmExitCause::UnhandledTrap(trap_csrs.scause);
                }
                VmCpuTrap::HostInterrupt(i) => {
                    break VmExitCause::HostInterrupt(i);
                }
                VmCpuTrap::InterruptEmulation => {
                    // Need to re-run the vCPU to inject the interrupt.
                    continue;
                }
                VmCpuTrap::OtherInterrupt(i) => {
                    // We don't expect Salus to take external interrupt itself, so everything else
                    // is considered unexpected.
                    println!("Unexpected guest interrupt {:?}", i);
                    break VmExitCause::UnhandledTrap(Trap::Interrupt(i).to_scause());
                }
            }
        };

        active_vcpu.exit(cause);

        Ok(u64::from(!cause.is_resumable()))
    }

    // Handles a virtual instruction trap taken due to `inst`.
    fn handle_virtual_instruction(
        &self,
        inst: DecodedInstruction,
        active_vcpu: &mut ActiveVmCpu<T>,
    ) -> ControlFlow<VmExitCause> {
        // We only emulate WFI and CSR instructions (for select CSRs) for now.
        // Everything else gets redirected as an illegal instruction exception.
        if let Some((csr, value, mask, rd)) = Self::decode_csr_instruction(inst, active_vcpu) {
            if let Ok(rd_value) = active_vcpu.virtual_csr_rmw(csr, value, mask) {
                active_vcpu.set_gpr(rd, rd_value);
                active_vcpu.inc_sepc(inst.len() as u64);
            } else {
                active_vcpu.inject_exception(Exception::IllegalInstruction, inst.raw() as u64);
            }
            ControlFlow::Continue(())
        } else if matches!(inst.instruction(), Instruction::Wfi) {
            // Just advance SEPC and exit. We place no constraints on when a vCPU
            // may be resumed from WFI since, per the privileged spec, it's only
            // a hint and it's perfectly valid for WFI to be a no-op.
            active_vcpu.inc_sepc(inst.len() as u64);
            ControlFlow::Break(VmExitCause::Wfi(inst))
        } else {
            active_vcpu.inject_exception(Exception::IllegalInstruction, inst.raw() as u64);
            ControlFlow::Continue(())
        }
    }

    // Decodes a CSR read-modify-write instruction into a (csr num, value, mask, dest) tuple.
    fn decode_csr_instruction(
        inst: DecodedInstruction,
        active_vcpu: &ActiveVmCpu<T>,
    ) -> Option<(u16, u64, u64, GprIndex)> {
        // Unwrap ok for all of the `GprIndex::from_raw()` below since the rs1/rd fields of
        // instructions must obviously map to GPRs.
        match inst.instruction() {
            Instruction::Csrrw(csr_type) => {
                let rs1 = GprIndex::from_raw(csr_type.rs1()).unwrap();
                let value = active_vcpu.get_gpr(rs1);
                let rd = GprIndex::from_raw(csr_type.rd()).unwrap();
                Some((csr_type.csr() as u16, value, !0u64, rd))
            }
            Instruction::Csrrs(csr_type) => {
                let rs1 = GprIndex::from_raw(csr_type.rs1()).unwrap();
                let mask = active_vcpu.get_gpr(rs1);
                let rd = GprIndex::from_raw(csr_type.rd()).unwrap();
                Some((csr_type.csr() as u16, !0u64, mask, rd))
            }
            Instruction::Csrrc(csr_type) => {
                let rs1 = GprIndex::from_raw(csr_type.rs1()).unwrap();
                let mask = active_vcpu.get_gpr(rs1);
                let rd = GprIndex::from_raw(csr_type.rd()).unwrap();
                Some((csr_type.csr() as u16, 0u64, mask, rd))
            }
            Instruction::Csrrwi(csri_type) => {
                let value = csri_type.zimm() as u64;
                let rd = GprIndex::from_raw(csri_type.rd()).unwrap();
                Some((csri_type.csr() as u16, value, !0u64, rd))
            }
            Instruction::Csrrsi(csri_type) => {
                let mask = csri_type.zimm() as u64;
                let rd = GprIndex::from_raw(csri_type.rd()).unwrap();
                Some((csri_type.csr() as u16, !0u64, mask, rd))
            }
            Instruction::Csrrci(csri_type) => {
                let mask = csri_type.zimm() as u64;
                let rd = GprIndex::from_raw(csri_type.rd()).unwrap();
                Some((csri_type.csr() as u16, 0u64, mask, rd))
            }
            _ => None,
        }
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&self, msg: SbiMessage, active_vcpu: &mut ActiveVmCpu<T>) -> EcallAction {
        match msg {
            SbiMessage::PutChar(_) => EcallAction::Forward(msg),
            SbiMessage::Reset(ResetFunction::Reset { .. }) => {
                EcallAction::Break(VmExitCause::FatalEcall(msg), SbiReturn::success(0))
            }
            SbiMessage::Base(base_func) => {
                EcallAction::Continue(self.handle_base_msg(base_func, active_vcpu))
            }
            SbiMessage::DebugConsole(debug_con_func) => self.handle_debug_console(debug_con_func),
            SbiMessage::HartState(hsm_func) => self.handle_hart_state_msg(hsm_func),
            SbiMessage::Nacl(nacl_func) => self.handle_nacl_msg(nacl_func, active_vcpu),
            SbiMessage::TeeHost(host_func) => self.handle_tee_host_msg(host_func, active_vcpu),
            SbiMessage::TeeInterrupt(interrupt_func) => {
                self.handle_tee_interrupt_msg(interrupt_func, active_vcpu)
            }
            SbiMessage::TeeGuest(guest_func) => self.handle_tee_guest_msg(guest_func, active_vcpu),
            SbiMessage::Attestation(attestation_func) => {
                self.handle_attestation_msg(attestation_func, active_vcpu.active_pages())
            }
            SbiMessage::Pmu(pmu_func) => self.handle_pmu_msg(pmu_func, active_vcpu).into(),
            SbiMessage::Vendor(regs) => self.handle_vendor_msg(&regs, active_vcpu).into(),
        }
    }

    fn handle_vendor_msg(
        &self,
        regs: &[u64],
        active_vcpu: &mut ActiveVmCpu<T>,
    ) -> EcallResult<u64> {
        let vendor_msg = SalusSbiMessage::from_regs(regs)
            .map_err(|_| EcallError::Sbi(SbiError::NotSupported))?;
        match vendor_msg {
            SalusSbiMessage::SalusTest(test_function) => {
                self.handle_salus_test(test_function, active_vcpu.active_pages())
            }
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
            let result = active_vcpu.pmu().start_counters(
                counter_index,
                counter_mask,
                start_flags,
                initial_value,
            );
            result.map(|_| 0).map_err(EcallError::from)
        }

        fn stop_counters<T: GuestStagePagingMode>(
            counter_index: u64,
            counter_mask: u64,
            stop_flags: PmuCounterStopFlags,
            active_vcpu: &mut ActiveVmCpu<T>,
        ) -> EcallResult<u64> {
            let result = active_vcpu
                .pmu()
                .stop_counters(counter_index, counter_mask, stop_flags);
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
            let result = active_vcpu.pmu().configure_matching_counters(
                counter_index,
                counter_mask,
                config_flags,
                event_type,
                event_data,
            );
            result.map_err(EcallError::from)
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

    fn handle_base_msg(&self, base_func: BaseFunction, active_vcpu: &ActiveVmCpu<T>) -> SbiReturn {
        use BaseFunction::*;
        let ret = match base_func {
            GetSpecificationVersion => SBI_SPEC_VERSION,
            GetImplementationID => SBI_IMPL_ID_SALUS,
            GetImplementationVersion => 0,
            ProbeSbiExtension(ext) => match ext {
                sbi_rs::EXT_PUT_CHAR
                | sbi_rs::EXT_BASE
                | sbi_rs::EXT_HART_STATE
                | sbi_rs::EXT_RESET
                | sbi_rs::EXT_DBCN
                | sbi_rs::EXT_NACL
                | sbi_rs::EXT_TEE_HOST
                | sbi_rs::EXT_TEE_INTERRUPT
                | sbi_rs::EXT_ATTESTATION => 1,
                sbi_rs::EXT_PMU if PmuInfo::get().is_ok() => 1,
                sbi_rs::EXT_TEE_GUEST => (!active_vcpu.is_host_vcpu()) as u64,
                _ => 0,
            },
            // TODO: 0 is valid result for the GetMachine* SBI calls but we should probably
            // report real values here.
            _ => 0,
        };
        SbiReturn::success(ret)
    }

    fn handle_debug_console(&self, debug_con_func: DebugConsoleFunction) -> EcallAction {
        match debug_con_func {
            DebugConsoleFunction::PutString { .. } => {
                EcallAction::Forward(SbiMessage::DebugConsole(debug_con_func))
            }
        }
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

    fn handle_nacl_msg(
        &self,
        nacl_func: NaclFunction,
        active_vcpu: &mut ActiveVmCpu<T>,
    ) -> EcallAction {
        use NaclFunction::*;
        match nacl_func {
            SetShmem { shmem_pfn } => self.set_shmem_area(shmem_pfn, active_vcpu).into(),
        }
    }

    fn set_shmem_area(&self, shmem_pfn: u64, active_vcpu: &mut ActiveVmCpu<T>) -> EcallResult<u64> {
        if shmem_pfn != u64::MAX {
            // Pin the pages that the VM wants to use for the shared state buffer.
            let shared_page_addr = self.guest_addr_from_raw(shmem_pfn << PFN_SHIFT)?;
            let pin = self
                .vm_pages()
                .pin_shared_pages(shared_page_addr, NACL_SHMEM_PAGES)
                .map_err(EcallError::from)?;
            active_vcpu
                .register_shmem_area(pin)
                .map_err(|_| EcallError::Sbi(SbiError::InvalidAddress))?;
        } else {
            active_vcpu.unregister_shmem_area();
        }
        Ok(0)
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
            TvmCreate { params_addr, len } => self.add_guest(params_addr, len, active_vcpu).into(),
            TvmDestroy { guest_id } => self.destroy_guest(guest_id).into(),
            TsmConvertPages {
                page_addr,
                num_pages,
            } => self.convert_pages(page_addr, num_pages).into(),
            TsmReclaimPages {
                page_addr,
                num_pages,
            } => self.reclaim_pages(page_addr, num_pages).into(),
            TsmInitiateFence => self.initiate_fence(active_vcpu).into(),
            TsmLocalFence => self.local_fence(active_vcpu).into(),
            AddPageTablePages {
                guest_id,
                page_addr,
                num_pages,
            } => self
                .guest_add_page_table_pages(guest_id, page_addr, num_pages)
                .into(),
            TvmAddMemoryRegion {
                guest_id,
                guest_addr,
                len,
            } => self
                .guest_add_memory_region(guest_id, guest_addr, len)
                .into(),
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
            Finalize {
                guest_id,
                entry_sepc,
                entry_arg,
            } => self.guest_finalize(guest_id, entry_sepc, entry_arg).into(),
            TvmCpuRun { guest_id, vcpu_id } => {
                self.guest_run_vcpu(guest_id, vcpu_id, active_vcpu).into()
            }
            TvmCpuCreate {
                guest_id,
                vcpu_id,
                state_page_addr,
            } => self
                .guest_add_vcpu(guest_id, vcpu_id, state_page_addr)
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
            TvmInitiateFence { guest_id } => self.guest_initiate_fence(guest_id).into(),
        }
    }

    fn get_tsm_info(
        &self,
        dest_addr: u64,
        len: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        let dest_addr = RawAddr::guest(dest_addr, self.page_owner_id());
        if len < mem::size_of::<sbi_rs::TsmInfo>() as u64 {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let len = mem::size_of::<sbi_rs::TsmInfo>();

        // Since we're the hypervisor we're ready from boot.
        let tsm_info = sbi_rs::TsmInfo {
            tsm_state: sbi_rs::TsmState::TsmReady,
            tsm_version: 0,
            tvm_state_pages: GuestVm::<T>::required_pages(),
            tvm_max_vcpus: VM_CPUS_MAX as u64,
            tvm_vcpu_state_pages: VmCpus::required_state_pages_per_vcpu(),
        };
        // Safety: &tsm_info points to len bytes of initialized memory.
        let tsm_info_bytes: &[u8] =
            unsafe { slice::from_raw_parts((&tsm_info as *const sbi_rs::TsmInfo).cast(), len) };
        active_pages
            .copy_to_guest(dest_addr, tsm_info_bytes)
            .map_err(EcallError::from)?;
        Ok(len as u64)
    }

    /// Converts `num_pages` of 4kB page-size starting at guest physical address `page_addr` to confidential memory.
    fn convert_pages(&self, page_addr: u64, num_pages: u64) -> EcallResult<u64> {
        let page_addr = self.guest_addr_from_raw(page_addr)?;
        self.vm_pages()
            .convert_pages(page_addr, num_pages)
            .map_err(EcallError::from)?;
        Ok(num_pages)
    }

    /// Reclaims `num_pages` of 4kB page-size confidential memory starting at guest physical address `page_addr`.
    fn reclaim_pages(&self, page_addr: u64, num_pages: u64) -> EcallResult<u64> {
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

    // Cleans and assigns the pages in `pages` as internal state pages for `owner`.
    fn assign_pages(
        pages: LockedPageList<Page<ConvertedDirty>>,
        owner: PageOwnerId,
    ) -> PageList<Page<InternalClean>> {
        let page_tracker = pages.page_tracker();
        let mut assigned = PageList::new(page_tracker.clone());
        for page in pages {
            // Unwrap ok: we have an exclusive reference to the converted page, so it must be
            // assignable.
            let page = page_tracker
                .assign_page_for_internal_state(page.clean(), owner)
                .unwrap();
            // Unwrap ok: since we had an exclusive reference to the page prior to assignment, it
            // can't be on any other list.
            assigned.push(page).unwrap();
        }
        assigned
    }

    fn add_guest(
        &self,
        params_addr: u64,
        len: u64,
        host_active_vcpu: &mut ActiveVmCpu<T>,
    ) -> EcallResult<u64> {
        if self.guests().is_none() {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        // Read the params from the VM's address space.
        let params_addr = RawAddr::guest(params_addr, self.page_owner_id());
        if len < mem::size_of::<sbi_rs::TvmCreateParams>() as u64 {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        let mut param_bytes = [0u8; mem::size_of::<sbi_rs::TvmCreateParams>()];
        host_active_vcpu
            .active_pages()
            .copy_from_guest(param_bytes.as_mut_slice(), params_addr)
            .map_err(EcallError::from)?;

        // Safety: `param_bytes` points to `size_of::<TvmCreateParams>()` contiguous, initialized
        // bytes.
        let params: sbi_rs::TvmCreateParams =
            unsafe { core::ptr::read_unaligned(param_bytes.as_slice().as_ptr().cast()) };

        // Now claim the pages that the host donated to us.
        let page_root_addr = self.guest_addr_from_raw(params.tvm_page_directory_addr)?;
        let guest_root_pages = self
            .vm_pages()
            .get_converted_pages(page_root_addr, 4)
            .map_err(EcallError::from)?;
        if !guest_root_pages.is_contiguous() {
            return Err(EcallError::Sbi(SbiError::InvalidAddress));
        }
        // Unwrap ok: guest_root_pages must be non-empty.
        let guest_root_base = guest_root_pages.peek().unwrap().bits();
        if (guest_root_base as *const u64).align_offset(T::TOP_LEVEL_ALIGN as usize) != 0 {
            return Err(EcallError::Sbi(SbiError::InvalidAddress));
        }
        let state_page_addr = self.guest_addr_from_raw(params.tvm_state_addr)?;
        let guest_box_pages = self
            .vm_pages()
            .get_converted_pages(state_page_addr, GuestVm::<T>::required_pages())
            .map_err(EcallError::from)?;
        if !guest_box_pages.is_contiguous() {
            return Err(EcallError::Sbi(SbiError::InvalidAddress));
        }

        // Allocate an ID for the new guest so that we can assign pages to it.
        let id = self
            .page_tracker()
            .add_active_guest()
            .map_err(|_| EcallError::Sbi(SbiError::Failed))?;

        // Assert safe here. We checked above that `guest_root_pages` is contiguous.
        let guest_root_pages =
            SequentialPages::from_pages(Self::assign_pages(guest_root_pages, id)).unwrap();
        // Unwrap ok: `guest_root_pages` is suitably aligned and owned by the new VM.
        let guest_root =
            GuestStagePageTable::new(guest_root_pages, id, self.page_tracker()).unwrap();

        let vm = Vm::new(
            VmPages::new(guest_root, self.vm_pages().nesting() + 1),
            VmCpus::new(),
        )
        .map_err(|_| EcallError::Sbi(SbiError::Failed))?;

        // Assert safe here. We checked above that `guest_box_pages` is contiguous.
        let guest_box_pages =
            SequentialPages::from_pages(Self::assign_pages(guest_box_pages, id)).unwrap();
        // Unwrap safe. We allocated `GustVm::<T>::required_pages()` above.
        let guest_vm = GuestVm::new(vm, guest_box_pages).unwrap();

        self.guests()
            .and_then(|g| g.add(guest_vm).ok())
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

    // Converts the guest TVM from initializing to runnable, and sets the initial entry point for
    // the TVM.
    fn guest_finalize(&self, guest_id: u64, entry_sepc: u64, entry_arg: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        guest
            .finalize(entry_sepc, entry_arg)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(0)
    }

    // Adds a vCPU with `vcpu_id` to a guest VM.
    fn guest_add_vcpu(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        state_page_addr: u64,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Get the converted pages that will be used to hold the private vCPU state. These pages
        // must be physically contiguous.
        let state_page_addr = self.guest_addr_from_raw(state_page_addr)?;
        let pages = self
            .vm_pages()
            .get_converted_pages(state_page_addr, VmCpus::required_state_pages_per_vcpu())
            .map_err(EcallError::from)?;
        if !pages.is_contiguous() {
            return Err(EcallError::Sbi(SbiError::InvalidAddress));
        }

        // Assert safe here. We checked above that `pages` is contiguous.
        let vcpu_pages =
            SequentialPages::from_pages(Self::assign_pages(pages, guest_vm.page_owner_id()))
                .unwrap();
        let vcpu_box = PageBox::new_with(
            VmCpu::new(vcpu_id, guest_vm.page_owner_id()),
            vcpu_pages,
            self.page_tracker(),
        );
        guest_vm.add_vcpu(vcpu_box)?;

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
        guest_vm.run_vcpu(vcpu_id, VmCpuParent::HostVm(active_vcpu))
    }

    fn guest_add_page_table_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        num_pages: u64,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest.as_any_vm();
        let from_page_addr = self.guest_addr_from_raw(from_addr)?;
        let pages = self
            .vm_pages()
            .get_converted_pages(from_page_addr, num_pages)
            .map_err(EcallError::from)?;
        for page in pages {
            // Unwrap ok: we have an exclusive reference to the converted page, so it must be
            // assignable.
            let page = self
                .page_tracker()
                .assign_page_for_internal_state(page.clean(), guest_vm.page_owner_id())
                .unwrap();
            // Unwrap ok: converted pages are always 4kB.
            guest_vm.vm_pages().add_pte_page(page).unwrap();
        }

        Ok(0)
    }

    fn guest_add_memory_region(
        &self,
        guest_id: u64,
        guest_addr: u64,
        len: u64,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let guest_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        guest_vm
            .vm_pages()
            .add_confidential_memory_region(guest_addr, len)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn guest_add_zero_pages(
        &self,
        guest_id: u64,
        page_addr: u64,
        page_type: sbi_rs::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
    ) -> EcallResult<u64> {
        if page_type != sbi_rs::TsmPageType::Page4k {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Get the pages we're trying to insert.
        let from_page_addr = self.guest_addr_from_raw(page_addr)?;
        let pages = self
            .vm_pages()
            .get_converted_pages(from_page_addr, num_pages)
            .map_err(EcallError::from)?;

        // Reserve the PTEs in the destination page table.
        let to_page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        let mapper = guest_vm
            .vm_pages()
            .map_zero_pages(to_page_addr, num_pages)
            .map_err(EcallError::from)?;

        for (page, addr) in pages.zip(to_page_addr.iter_from()) {
            // Unwrap ok: we have an exclusive reference to the converted page, so it must be
            // assignable.
            let page = self
                .page_tracker()
                .assign_page_for_mapping(page.clean(), guest_vm.page_owner_id())
                .unwrap();
            // Unwrap ok: the address is in range and we haven't mapped it yet.
            mapper.map_page(addr, page).unwrap();
        }

        Ok(num_pages)
    }

    #[allow(clippy::too_many_arguments)]
    fn guest_add_measured_pages(
        &self,
        guest_id: u64,
        src_addr: u64,
        dest_addr: u64,
        page_type: sbi_rs::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        if page_type != sbi_rs::TsmPageType::Page4k {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_initializing_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Get the pages we're going to be copying to and inserting.
        let from_page_addr = self.guest_addr_from_raw(dest_addr)?;
        let pages = self
            .vm_pages()
            .get_converted_pages(from_page_addr, num_pages)
            .map_err(EcallError::from)?;

        // Reserve the PTEs in the destination page table.
        let to_page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        let mapper = guest_vm
            .vm_pages()
            .map_measured_pages(to_page_addr, num_pages)
            .map_err(EcallError::from)?;

        // Make sure we can initialize the full set of pages before we start actually inserting
        // them into the destination page table.
        let src_page_addr = self.guest_addr_from_raw(src_addr)?;
        let mut initialized_pages = LockedPageList::new(self.page_tracker());
        for (page, addr) in pages.zip(src_page_addr.iter_from()) {
            match page.try_initialize(|bytes| active_pages.copy_from_guest(bytes, addr.into())) {
                Ok(p) => {
                    // Unwrap ok since the page cannot have been on any other list.
                    initialized_pages.push(p).unwrap();
                }
                Err((e, p)) => {
                    // Unwrap ok since the page must have been locked.
                    self.page_tracker().unlock_page(p).unwrap();
                    return Err(EcallError::from(e));
                }
            };
        }

        // Now insert the pages.
        for (page, addr) in initialized_pages.zip(to_page_addr.iter_from()) {
            // Unwrap ok: we have an exclusive reference to the converted page, so it must be
            // assignable.
            let page = self
                .page_tracker()
                .assign_page_for_mapping(page, guest_vm.page_owner_id())
                .unwrap();
            // Unwrap ok: the address is in range and we haven't mapped it yet.
            mapper
                .map_page(addr, page, guest_vm.attestation_mgr())
                .unwrap();
        }

        Ok(num_pages)
    }

    fn guest_add_shared_pages(
        &self,
        guest_id: u64,
        page_addr: u64,
        page_type: sbi_rs::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
    ) -> EcallResult<u64> {
        if page_type != TsmPageType::Page4k || num_pages == 0 {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }

        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Get the pages we're trying to insert.
        let from_page_addr = self.guest_addr_from_raw(page_addr)?;
        let pages = self
            .vm_pages()
            .get_shareable_pages(from_page_addr, num_pages)
            .map_err(EcallError::from)?;

        // Reserve the PTEs in the destination page table.
        let to_page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        let mapper = guest_vm
            .vm_pages()
            .map_shared_pages(to_page_addr, num_pages)
            .map_err(EcallError::from)?;

        for (page, addr) in pages.zip(to_page_addr.iter_from()) {
            // Unwrap ok: The page is guaranteed to be in a shareable state until the iterator is
            // destroyed.
            let page = self
                .page_tracker()
                .share_page(page, self.page_owner_id())
                .unwrap();
            // Unwrap ok: the address is in range and we haven't mapped it yet.
            mapper.map_page(addr, page).unwrap();
        }

        Ok(num_pages)
    }

    fn guest_initiate_fence(&self, guest_id: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        // TODO: This uses the same TLB version as the guest itself would use if it needed to
        // do a TLB shootdown, for example if it were to convert pages for a nested TVM. We
        // would need a separate "self" TLB version and "parent" TLB version if we wanted to
        // support concurrent invalidations by the TVM and the TVM's parent. Since we don't
        // support nesting at the moment, just use the same TLB version.
        guest_vm
            .vm_pages()
            .initiate_fence()
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn handle_salus_test(
        &self,
        _test_func: SalusTestFunction,
        _active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        Err(EcallError::Sbi(SbiError::NotSupported))
    }

    fn map_guest_range_in_umode_slot(
        &self,
        slot: UmodeSlotId,
        addr: u64,
        len: u64,
        slot_perm: UmodeSlotPerm,
    ) -> EcallResult<(u64, GuestUmodeMapping)> {
        let base = PageSize::Size4k.round_down(addr);
        let end = addr
            .checked_add(len)
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let umode_mapping = self
            .vm_pages()
            .map_in_umode_slot(
                slot,
                self.guest_addr_from_raw(base)?,
                PageSize::num_4k_pages(end - base),
                slot_perm,
            )
            .map_err(EcallError::from)?;
        let vaddr = umode_mapping.vaddr().bits() + (addr - base);
        Ok((vaddr, umode_mapping))
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

    fn guest_get_evidence(
        &self,
        csr_guest_addr: u64,
        csr_len: usize,
        _request_data_addr: u64,
        _evidence_format: u64,
        certout_guest_addr: u64,
        certout_len: usize,
    ) -> EcallResult<u64> {
        // Map CSR read-only.
        let (csr_addr, _csr_mapping) = self.map_guest_range_in_umode_slot(
            UmodeSlotId::A,
            csr_guest_addr,
            csr_len as u64,
            UmodeSlotPerm::Readonly,
        )?;
        // Map Output Certificate writable.
        let (certout_addr, _certout_mapping) = self.map_guest_range_in_umode_slot(
            UmodeSlotId::B,
            certout_guest_addr,
            certout_len as u64,
            UmodeSlotPerm::Writable,
        )?;
        // Gather measurement registers from the attestation manager and transform it in a array.
        let msmt_genarray = self.attestation_mgr().measurement_registers()?;
        let zero = [0u8; u_mode_api::cert::SHA384_LEN];
        let mut msmt_regs = [zero; attestation::MSMT_REGISTERS];
        for (i, r) in msmt_genarray.iter().enumerate() {
            msmt_regs[i].copy_from_slice(r.as_slice());
        }
        // Get the CDI ID for the attestation layer.
        let cdi_id = self.attestation_mgr().attestation_cdi_id()?;
        let shared_data = u_mode_api::cert::GetEvidenceShared { msmt_regs, cdi_id };
        let request = u_mode_api::UmodeRequest::GetEvidence {
            csr_addr,
            csr_len,
            certout_addr,
            certout_len,
        };
        // Send request to U-mode.
        Ok(UmodeTask::send_req_with_shared_data(request, shared_data)?)
    }

    fn guest_extend_measurement(
        &self,
        msmt_addr: u64,
        msmt_size: usize,
        index: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> EcallResult<u64> {
        // Check that the index is valid.
        let msmt_idx: TcgPcrIndex = index.try_into().map_err(EcallError::from)?;

        let caps = self
            .attestation_mgr()
            .capabilities()
            .map_err(EcallError::from)?;

        // Check that the measurement buffer size matches exactly the hash
        // algorithm one.
        if msmt_size != caps.hash_algorithm.size() {
            return Err(EcallError::Sbi(SbiError::InsufficientBufferCapacity));
        }

        let mut measurement_data = [0u8; sbi_rs::MAX_HASH_SIZE];
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
            .extend_msmt_register(msmt_idx, &measurement_data, None)
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
        // Check that the index is valid.
        let msmt_idx: TcgPcrIndex = index.try_into().map_err(EcallError::from)?;

        let caps = self
            .attestation_mgr()
            .capabilities()
            .map_err(EcallError::from)?;

        let measurement_data = self
            .attestation_mgr()
            .read_msmt_register(msmt_idx)
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

    fn handle_tee_interrupt_msg(
        &self,
        interrupt_func: TeeInterruptFunction,
        active_vcpu: &ActiveVmCpu<T>,
    ) -> EcallAction {
        use TeeInterruptFunction::*;
        match interrupt_func {
            TvmAiaInit {
                tvm_id,
                params_addr,
                len,
            } => self
                .guest_aia_init(
                    tvm_id,
                    params_addr,
                    len as usize,
                    active_vcpu.active_pages(),
                )
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
            TvmCpuBindImsic {
                tvm_id,
                vcpu_id,
                imsic_mask,
            } => self
                .guest_bind_vcpu(tvm_id, vcpu_id, imsic_mask, active_vcpu)
                .into(),
            TvmCpuUnbindImsicBegin { tvm_id, vcpu_id } => {
                self.guest_unbind_vcpu_begin(tvm_id, vcpu_id).into()
            }
            TvmCpuUnbindImsicEnd { tvm_id, vcpu_id } => {
                self.guest_unbind_vcpu_end(tvm_id, vcpu_id).into()
            }
            TvmCpuInjectExternalInterrupt {
                tvm_id,
                vcpu_id,
                interrupt_id,
            } => self
                .guest_inject_ext_interrupt(tvm_id, vcpu_id, interrupt_id)
                .into(),
            TvmCpuRebindImsicBegin {
                tvm_id,
                vcpu_id,
                imsic_mask,
            } => self
                .guest_rebind_vcpu_begin(tvm_id, vcpu_id, imsic_mask, active_vcpu)
                .into(),
            TvmCpuRebindImsicClone { tvm_id, vcpu_id } => {
                self.guest_rebind_vcpu_clone(tvm_id, vcpu_id).into()
            }
            TvmCpuRebindImsicEnd { tvm_id, vcpu_id } => {
                self.guest_rebind_vcpu_end(tvm_id, vcpu_id).into()
            }
        }
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
        let mut param_bytes = [0u8; mem::size_of::<sbi_rs::TvmAiaParams>()];
        if params_len < param_bytes.len() {
            return Err(EcallError::Sbi(SbiError::InvalidParam));
        }
        active_pages
            .copy_from_guest(param_bytes.as_mut_slice(), params_addr)
            .map_err(EcallError::from)?;
        // Safety: `param_bytes` points to `size_of::<TvmAiaParams>()` contiguous, initialized
        // bytes.
        let params: sbi_rs::TvmAiaParams =
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

    /// Begins the process of binding `vcpu_id` to the given guest interrupt file on the current
    /// CPU by initializing the interrupt file.
    pub fn bind_vcpu_begin(&self, vcpu_id: u64, interrupt_file: ImsicFileId) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.bind_imsic_prepare(interrupt_file))
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    /// Completes the process of binding `vcpu_id` to a guest interrupt file by restoring interrupt
    /// state from the vCPU's software interrupt file.
    pub fn bind_vcpu_end(&self, vcpu_id: u64) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.bind_imsic_finish())
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    fn guest_bind_vcpu(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        imsic_mask: u64,
        active_vcpu: &ActiveVmCpu<T>,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // imsic_mask is in the same format as HGEIE, where bits [1:N] specify the guest interrupt
        // files. We only support binding a single interrupt file for now.
        if imsic_mask.count_ones() != 1 {
            return Err(EcallError::Sbi(SbiError::InvalidParam))?;
        }
        let imsic_index = imsic_mask
            .trailing_zeros()
            .checked_sub(1)
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Get the IMSIC page that we're going to assign.
        let base_location = active_vcpu
            .get_imsic_location()
            .ok_or(EcallError::Sbi(SbiError::NotSupported))?;
        let src_location = ImsicLocation::new(
            base_location.group(),
            base_location.hart(),
            ImsicFileId::guest(imsic_index),
        );
        let from_page_addr = self
            .vm_pages()
            .imsic_geometry()
            .and_then(|g| g.location_to_addr(src_location))
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let imsic_pages = self
            .vm_pages()
            .get_converted_imsic(from_page_addr)
            .map_err(EcallError::from)?;

        // Make sure we can map the page before starting the bind process.
        let to_page_addr = guest_vm.get_vcpu_imsic_addr(vcpu_id)?;
        let mapper = guest_vm
            .vm_pages()
            .map_imsic_pages(to_page_addr, 1)
            .map_err(EcallError::from)?;

        // Prepare the destination interrupt file.
        //
        // Unwrap ok: imsic_pages is exactly one page long and its location must be valid.
        let interrupt_file = Imsic::get()
            .phys_geometry()
            .addr_to_location(imsic_pages.peek().unwrap())
            .unwrap()
            .file();
        guest_vm.bind_vcpu_begin(vcpu_id, interrupt_file)?;

        for (page, addr) in imsic_pages.zip(to_page_addr.iter_from()) {
            // Unwrap ok: we have an exclusive reference to the converted page, so it must be
            // assignable.
            let page = self
                .page_tracker()
                .assign_page_for_mapping(page, guest_vm.page_owner_id())
                .unwrap();
            // Unwrap ok: the address is in range and we haven't mapped it yet.
            mapper.map_page(addr, page).unwrap();
        }

        // Unwrap ok: we know the vCPU is already in the "binding" state.
        guest_vm.bind_vcpu_end(vcpu_id).unwrap();

        Ok(0)
    }

    fn rebind_vcpu_begin(&self, vcpu_id: u64, interrupt_file: ImsicFileId) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.rebind_imsic_prepare(interrupt_file))
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    fn guest_rebind_vcpu_begin(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        imsic_mask: u64,
        active_vcpu: &ActiveVmCpu<T>,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // imsic_mask is in the same format as HGEIE, where bits [1:N] specify the guest interrupt
        // files. We only support binding a single interrupt file for now.
        if imsic_mask.count_ones() != 1 {
            return Err(EcallError::Sbi(SbiError::InvalidParam))?;
        }
        let imsic_index = imsic_mask
            .trailing_zeros()
            .checked_sub(1)
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Get the IMSIC page that we're going to assign.
        let base_location = active_vcpu
            .get_imsic_location()
            .ok_or(EcallError::Sbi(SbiError::NotSupported))?;
        let src_location = ImsicLocation::new(
            base_location.group(),
            base_location.hart(),
            ImsicFileId::guest(imsic_index),
        );
        let from_page_addr = self
            .vm_pages()
            .imsic_geometry()
            .and_then(|g| g.location_to_addr(src_location))
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        let imsic_pages = self
            .vm_pages()
            .get_converted_imsic(from_page_addr)
            .map_err(EcallError::from)?;

        // to_page_addr contains the virtual address set by host. it's where guest vcpu views
        // its interrupt file.
        let to_page_addr = guest_vm.get_vcpu_imsic_addr(vcpu_id)?;
        let mapper = guest_vm
            .vm_pages()
            .remap_imsic_pages(to_page_addr, 1)
            .map_err(EcallError::from)?;

        // Get the destination interrupt file.
        //
        // Unwrap ok: imsic_pages is exactly one page long and its location must be valid.
        let interrupt_file = Imsic::get()
            .phys_geometry()
            .addr_to_location(imsic_pages.peek().unwrap())
            .unwrap()
            .file();

        // Clears the new guest interrupt file and sets the state to Rebinding.
        guest_vm.rebind_vcpu_begin(vcpu_id, interrupt_file)?;

        for (page, addr) in imsic_pages.zip(to_page_addr.iter_from()) {
            // Unwrap ok: we have an exclusive reference to the converted page, so it must be
            // assignable.
            let page = self
                .page_tracker()
                .assign_page_for_mapping(page, guest_vm.page_owner_id())
                .unwrap();
            // Unwrap ok: the address is in the range and mapper validated that address is remappable.
            let prev_addr = mapper.remap_page(addr, page).unwrap();
            // Safety: We've verified the typing of the page and we must have unique
            // ownership since the page was mapped before it was replaced.
            let prev_page: ImsicGuestPage<Invalidated> = unsafe { ImsicGuestPage::new(prev_addr) };
            // Unwrap ok: Page was mapped and has just been invalidated.
            guest_vm
                .vm_pages()
                .unassign_imsic_page_begin(prev_page)
                .unwrap();
        }

        Ok(0)
    }

    fn rebind_vcpu_clone(&self, vcpu_id: u64) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.rebind_imsic_clone())
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    fn guest_rebind_vcpu_clone(&self, guest_id: u64, vcpu_id: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        let prev_imisc_loc = guest_vm
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.prev_imsic_location())
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        // Unwrap ok: prev_imisc_loc must've been a valid location given it was bound to the vCPU.
        let prev_imsic_addr = Imsic::get()
            .phys_geometry()
            .location_to_addr(prev_imisc_loc)
            .unwrap();

        // Makes sure the TLB flush has been completed and unassigns the previous imsic page from
        // page_tracker.
        guest_vm
            .vm_pages()
            .unassign_imsic_page_end(prev_imsic_addr)
            .map_err(EcallError::from)?;

        // Saves the previous guest interrupt file's state.
        guest_vm.rebind_vcpu_clone(vcpu_id)?;
        Ok(0)
    }

    fn rebind_vcpu_end(&self, vcpu_id: u64) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.rebind_imsic_finish())
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    fn guest_rebind_vcpu_end(&self, guest_id: u64, vcpu_id: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        guest_vm.rebind_vcpu_end(vcpu_id)?;
        Ok(0)
    }

    fn unbind_vcpu_begin(&self, vcpu_id: u64) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.unbind_imsic_prepare())
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    fn guest_unbind_vcpu_begin(&self, guest_id: u64, vcpu_id: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Make sure we're in the proper state to unbind the vCPU before we go unmapping
        // the page.
        let guest_addr = guest_vm.get_vcpu_imsic_addr(vcpu_id)?;
        guest_vm.unbind_vcpu_begin(vcpu_id)?;

        // Unwrap ok: guest_addr must've been mapped if it was bound to a vCPU in guest_vm.
        guest_vm
            .vm_pages()
            .unassign_imsic_begin(guest_addr)
            .unwrap();

        Ok(0)
    }

    fn unbind_vcpu_end(&self, vcpu_id: u64) -> EcallResult<()> {
        self.vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .and_then(|vcpu| vcpu.unbind_imsic_finish())
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))
    }

    fn guest_unbind_vcpu_end(&self, guest_id: u64, vcpu_id: u64) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;

        // Make sure the TLB flush has been completed.
        let guest_addr = guest_vm.get_vcpu_imsic_addr(vcpu_id)?;
        guest_vm
            .vm_pages()
            .unassign_imsic_end(guest_addr)
            .map_err(EcallError::from)?;

        // Finish saving the IMSIC state to the SW file.
        guest_vm.unbind_vcpu_end(vcpu_id)?;

        Ok(0)
    }

    fn inject_ext_interrupt(&self, vcpu_id: u64, interrupt_id: u64) -> EcallResult<()> {
        let vcpu = self
            .vm()
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        vcpu.inject_ext_interrupt(interrupt_id as usize)
            .map_err(|_| EcallError::Sbi(SbiError::Denied))
    }

    fn guest_inject_ext_interrupt(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        interrupt_id: u64,
    ) -> EcallResult<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest
            .as_finalized_vm()
            .ok_or(EcallError::Sbi(SbiError::InvalidParam))?;
        guest_vm.inject_ext_interrupt(vcpu_id, interrupt_id)?;
        Ok(0)
    }

    fn handle_tee_guest_msg(
        &self,
        guest_func: TeeGuestFunction,
        active_vcpu: &ActiveVmCpu<T>,
    ) -> EcallAction {
        // Guest ABI is not supported for Host.
        if active_vcpu.is_host_vcpu() {
            return Err(EcallError::Sbi(SbiError::NotSupported)).into();
        }

        use TeeGuestFunction::*;
        let result = match guest_func {
            AddMmioRegion { addr, len } => self.add_mmio_region(addr, len),
            RemoveMmioRegion { addr, len } => self.remove_mmio_region(addr, len),
            ShareMemory { addr, len } | UnshareMemory { addr, len } => {
                let result = if matches!(guest_func, ShareMemory { .. }) {
                    self.share_mem_region(addr, len)
                } else {
                    self.unshare_mem_region(addr, len)
                };

                // Block if we need a TLB invalidation.
                let action = match result {
                    Ok(tlbv) if tlbv > self.vm_pages().min_tlb_version() => EcallAction::Break(
                        VmExitCause::BlockingEcall(SbiMessage::TeeGuest(guest_func), tlbv),
                        SbiReturn::success(0),
                    ),
                    Ok(_) => EcallAction::Break(
                        VmExitCause::ResumableEcall(SbiMessage::TeeGuest(guest_func)),
                        SbiReturn::success(0),
                    ),
                    Err(_) => result.map(|_| 0).into(),
                };
                return action;
            }
            AllowExternalInterrupt { id } => self.allow_ext_interrupt(id, active_vcpu),
            DenyExternalInterrupt { id } => self.deny_ext_interrupt(id, active_vcpu),
        };

        // Notify the host if a TEE-Guest call succeeds.
        match result {
            Ok(r) => EcallAction::Break(
                VmExitCause::ResumableEcall(SbiMessage::TeeGuest(guest_func)),
                SbiReturn::success(r),
            ),
            Err(_) => result.into(),
        }
    }

    fn add_mmio_region(&self, addr: u64, len: u64) -> EcallResult<u64> {
        let addr = self.guest_addr_from_raw(addr)?;
        self.vm_pages()
            .add_mmio_region(addr, len)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn remove_mmio_region(&self, addr: u64, len: u64) -> EcallResult<u64> {
        let addr = self.guest_addr_from_raw(addr)?;
        self.vm_pages()
            .remove_mmio_region(addr, len)
            .map_err(EcallError::from)?;
        Ok(0)
    }

    fn share_mem_region(&self, addr: u64, len: u64) -> EcallResult<TlbVersion> {
        let addr = self.guest_addr_from_raw(addr)?;
        self.vm_pages()
            .share_mem_region_begin(addr, len)
            .map_err(EcallError::from)
    }

    fn unshare_mem_region(&self, addr: u64, len: u64) -> EcallResult<TlbVersion> {
        let addr = self.guest_addr_from_raw(addr)?;
        self.vm_pages()
            .unshare_mem_region_begin(addr, len)
            .map_err(EcallError::from)
    }

    fn allow_ext_interrupt(&self, id: i64, active_vcpu: &ActiveVmCpu<T>) -> EcallResult<u64> {
        if id == -1 {
            active_vcpu.allow_all_ext_interrupts()
        } else {
            active_vcpu.allow_ext_interrupt(id as usize)
        }
        .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(0)
    }

    fn deny_ext_interrupt(&self, id: i64, active_vcpu: &ActiveVmCpu<T>) -> EcallResult<u64> {
        if id == -1 {
            active_vcpu.deny_all_ext_interrupts()
        } else {
            active_vcpu.deny_ext_interrupt(id as usize)
        }
        .map_err(|_| EcallError::Sbi(SbiError::InvalidParam))?;
        Ok(0)
    }
}
