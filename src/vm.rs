// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use attestation::{request::CertReq, MAX_CSR_LEN};
use core::{mem, slice};
use data_measure::sha256::SHA256_DIGEST_BYTES;
use der::Decode;
use drivers::{CpuId, CpuInfo, ImsicGuestId, MAX_CPUS};
use page_tracking::{HypPageAlloc, PageList, PageTracker};
use riscv_page_tables::{GuestStagePageTable, PlatformPageTable};
use riscv_pages::*;
use riscv_regs::{GprIndex, Trap};
use s_mode_utils::abort::abort;
use sbi::Error as SbiError;
use sbi::*;

use crate::guest_tracking::{GuestState, Guests};
use crate::print_util::*;
use crate::smp;
use crate::vm_cpu::{VirtualRegister, VmCpuExit, VmCpuStatus, VmCpus, VM_CPU_BYTES};
use crate::vm_pages::{
    ActiveVmPages, PageFaultType, VmPages, VmRegionList, TVM_REGION_LIST_PAGES, TVM_STATE_PAGES,
};
use crate::{print, println};

const GUEST_ID_SELF_MEASUREMENT: u64 = 0;

// What we report ourselves as in sbi_get_sbi_impl_id(). Just pick something unclaimed so no one
// confuses us with BBL/OpenSBI.
const SBI_IMPL_ID_SALUS: u64 = 7;

/// Powers off this machine.
pub fn poweroff() -> ! {
    // Safety: on this platform, a write of 0x5555 to 0x100000 will trigger the platform to
    // poweroff, which is defined behavior.
    unsafe {
        core::ptr::write_volatile(0x10_0000 as *mut u32, 0x5555);
    }
    abort()
}

pub enum VmStateInitializing {}
pub enum VmStateFinalized {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VmExitCause {
    PowerOff(ResetType, ResetReason),
    CpuStart(u64),
    CpuStop,
    ConfidentialPageFault(GuestPageAddr),
    UnhandledTrap(u64),
}

impl VmExitCause {
    fn code(&self) -> TvmCpuExitCode {
        use VmExitCause::*;
        match self {
            PowerOff(_, _) => TvmCpuExitCode::SystemReset,
            CpuStart(_) => TvmCpuExitCode::HartStart,
            CpuStop => TvmCpuExitCode::HartStop,
            ConfidentialPageFault(_) => TvmCpuExitCode::ConfidentialPageFault,
            UnhandledTrap(_) => TvmCpuExitCode::UnhandledException,
        }
    }

    fn cause0(&self) -> Option<u64> {
        use VmExitCause::*;
        match self {
            PowerOff(reset_type, _) => Some(*reset_type as u64),
            CpuStart(hart_id) => Some(*hart_id),
            ConfidentialPageFault(addr) => Some(addr.bits()),
            UnhandledTrap(scause) => Some(*scause),
            _ => None,
        }
    }

    fn cause1(&self) -> Option<u64> {
        use VmExitCause::*;
        match self {
            PowerOff(_, reset_reason) => Some(*reset_reason as u64),
            _ => None,
        }
    }
}

impl From<PageFaultType> for VmExitCause {
    fn from(fault: PageFaultType) -> VmExitCause {
        use VmExitCause::*;
        match fault {
            PageFaultType::Confidential(addr) => {
                // Mask off the page offset to avoid revealing more information than necessary to
                // the host.
                ConfidentialPageFault(PageAddr::with_round_down(addr, PageSize::Size4k))
            }
            PageFaultType::Unmapped(e, _) => UnhandledTrap(Trap::Exception(e).to_scause()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EcallAction {
    LegacyOk,
    Unhandled,
    Continue(SbiReturn),
    Break(VmExitCause, SbiReturn),
}

/// A VM that is being run.
pub struct Vm<T: GuestStagePageTable, S = VmStateFinalized> {
    vcpus: VmCpus,
    vm_pages: VmPages<T, S>,
    guests: Option<Guests<T>>,
}

impl<T: GuestStagePageTable, S> Vm<T, S> {
    /// Returns this VM's ID.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.vm_pages.page_owner_id()
    }

    /// Returns the `PageTracker` singleton.
    pub fn page_tracker(&self) -> PageTracker {
        self.vm_pages.page_tracker()
    }

    /// Convenience function to turn a raw u64 from an SBI call to a `GuestPageAddr`.
    fn guest_addr_from_raw(&self, guest_addr: u64) -> sbi::Result<GuestPageAddr> {
        PageAddr::new(RawAddr::guest(guest_addr, self.page_owner_id()))
            .ok_or(SbiError::InvalidParam)
    }
}

impl<T: GuestStagePageTable> Vm<T, VmStateInitializing> {
    /// Create a new guest using the given initial page table and vCPU tracking table.
    pub fn new(vm_pages: VmPages<T, VmStateInitializing>, vcpus: VmCpus) -> Self {
        Self {
            vcpus,
            vm_pages,
            guests: None,
        }
    }

    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    fn add_guest_tracking_pages(&mut self, pages: SequentialPages<InternalClean>) {
        self.guests = Some(Guests::new(pages, self.vm_pages.page_tracker()));
    }

    /// Sets a vCPU register.
    fn set_vcpu_reg(&self, vcpu_id: u64, register: TvmCpuRegister, value: u64) -> sbi::Result<()> {
        let vcpu = self
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        use TvmCpuRegister::*;
        match register {
            EntryPc => vcpu.set_sepc(value),
            EntryArg => vcpu.set_gpr(GprIndex::A1, value),
            _ => {
                return Err(SbiError::InvalidParam);
            }
        };
        Ok(())
    }

    /// Gets a vCPU register.
    fn get_vcpu_reg(&self, vcpu_id: u64, register: TvmCpuRegister) -> sbi::Result<u64> {
        let vcpu = self
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        use TvmCpuRegister::*;
        match register {
            EntryPc => Ok(vcpu.get_sepc()),
            EntryArg => Ok(vcpu.get_gpr(GprIndex::A1)),
            _ => Err(SbiError::InvalidParam),
        }
    }

    /// Adds a vCPU to this VM.
    fn add_vcpu(&self, vcpu_id: u64) -> sbi::Result<()> {
        let vcpu = self
            .vcpus
            .add_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        vcpu.set_gpr(GprIndex::A0, vcpu_id);
        Ok(())
    }

    /// Completes intialization of the `Vm`, returning it in a finalized state.
    pub fn finalize(self) -> Vm<T, VmStateFinalized> {
        Vm {
            vcpus: self.vcpus,
            vm_pages: self.vm_pages.finalize(),
            guests: self.guests,
        }
    }

    /// Destroys this `Vm`.
    pub fn destroy(&mut self) {
        let page_tracker = self.vm_pages.page_tracker();
        page_tracker.rm_active_guest(self.vm_pages.page_owner_id());
    }
}

impl<T: GuestStagePageTable> Vm<T, VmStateFinalized> {
    /// Binds the specified vCPU to an IMSIC interrupt file.
    fn bind_vcpu(&self, vcpu_id: u64, interrupt_file: ImsicGuestId) -> sbi::Result<()> {
        let vcpu = self
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        // TODO: Bind to this (physical) CPU as well.
        vcpu.set_interrupt_file(interrupt_file);
        Ok(())
    }

    /// Makes the specified vCPU runnable.
    fn power_on_vcpu(&self, vcpu_id: u64) -> sbi::Result<()> {
        self.vcpus
            .power_on_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        Ok(())
    }

    /// Sets the entry point of the specified vCPU and makes it runnable.
    fn start_vcpu(&self, vcpu_id: u64, start_addr: u64, opaque: u64) -> sbi::Result<()> {
        let vcpu = self
            .vcpus
            .power_on_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        vcpu.set_sepc(start_addr);
        vcpu.set_gpr(GprIndex::A1, opaque);
        Ok(())
    }

    /// Gets the state of the specified vCPU.
    fn get_vcpu_status(&self, vcpu_id: u64) -> sbi::Result<u64> {
        let vcpu_status = self
            .vcpus
            .get_vcpu_status(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let status = match vcpu_status {
            VmCpuStatus::Runnable | VmCpuStatus::Running => HartState::Started,
            VmCpuStatus::PoweredOff => HartState::Stopped,
            VmCpuStatus::NotPresent => {
                return Err(SbiError::InvalidParam);
            }
        };
        Ok(status as u64)
    }

    /// Gets a vCPU register.
    fn get_vcpu_reg(&self, vcpu_id: u64, register: TvmCpuRegister) -> sbi::Result<u64> {
        let vcpu = self
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        use TvmCpuRegister::*;
        match register {
            ExitCause0 => Ok(vcpu.get_virt_reg(VirtualRegister::Cause0)),
            ExitCause1 => Ok(vcpu.get_virt_reg(VirtualRegister::Cause1)),
            _ => Err(SbiError::InvalidParam),
        }
    }

    /// Run this guest until an unhandled exit is encountered.
    fn run_vcpu(&self, vcpu_id: u64) -> sbi::Result<TvmCpuExitCode> {
        // Take the vCPU out of self.vcpus, giving us exclusive ownership.
        let mut vcpu = self
            .vcpus
            .take_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let exit_code = {
            let mut vcpu = vcpu.lock();

            // Run until there's an exit we can't handle.
            let cause = loop {
                // Activate this vCPU and its address space. We re-activate after every exit (even
                // if it was handled) so that any pending TLB maintenance can be completed.
                let mut active_vcpu = vcpu.activate(&self.vm_pages).unwrap();

                let exit = active_vcpu.run_to_exit();
                use SbiReturnType::*;
                match exit {
                    VmCpuExit::Ecall(Some(sbi_msg)) => {
                        match self.handle_ecall(sbi_msg, active_vcpu.active_pages()) {
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
                        }
                    }
                    VmCpuExit::Ecall(None) => {
                        // Unrecognized ECALL, return an error.
                        active_vcpu
                            .set_ecall_result(Standard(SbiReturn::from(SbiError::NotSupported)));
                    }
                    VmCpuExit::PageFault(e, addr) => {
                        let pf = active_vcpu.active_pages().get_page_fault_cause(e, addr);
                        break pf.into();
                    }
                    VmCpuExit::DelegatedException(e, stval) => {
                        active_vcpu.inject_exception(e, stval);
                    }
                    VmCpuExit::Other(ref trap_csrs) => {
                        println!("Unhandled guest exit, SCAUSE = 0x{:08x}", trap_csrs.scause);
                        break VmExitCause::UnhandledTrap(trap_csrs.scause);
                    }
                }
            };

            // Populate the virtual trap cause registers so that the host can retrieve the detailed
            // exit cause.
            if let Some(cause0) = cause.cause0() {
                vcpu.set_virt_reg(VirtualRegister::Cause0, cause0);
            }
            if let Some(cause1) = cause.cause1() {
                vcpu.set_virt_reg(VirtualRegister::Cause1, cause1);
            }
            cause.code()
        };

        // Disable the vCPU if the exit cause indicates it is no longer runnable.
        use TvmCpuExitCode::*;
        if matches!(exit_code, SystemReset | HartStop | UnhandledException) {
            vcpu.power_off();
        }

        Ok(exit_code)
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&self, msg: SbiMessage, active_pages: &ActiveVmPages<T>) -> EcallAction {
        match msg {
            SbiMessage::PutChar(c) => {
                // put char - legacy command
                print!("{}", c as u8 as char);
                EcallAction::LegacyOk
            }
            SbiMessage::Reset(ResetFunction::Reset { reset_type, reason }) => EcallAction::Break(
                VmExitCause::PowerOff(reset_type, reason),
                SbiReturn::success(0),
            ),
            SbiMessage::Base(base_func) => EcallAction::Continue(self.handle_base_msg(base_func)),
            SbiMessage::HartState(hsm_func) => self.handle_hart_state_msg(hsm_func),
            SbiMessage::Tee(tee_func) => {
                EcallAction::Continue(self.handle_tee_msg(tee_func, active_pages))
            }
            SbiMessage::Measurement(measurement_func) => {
                EcallAction::Continue(self.handle_measurement_msg(measurement_func, active_pages))
            }
            SbiMessage::Attestation(attestation_func) => {
                EcallAction::Continue(self.handle_attestation_msg(attestation_func, active_pages))
            }
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
                | sbi::EXT_TEE
                | sbi::EXT_MEASUREMENT => 1,
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
                Ok(()) => EcallAction::Break(VmExitCause::CpuStart(hart_id), SbiReturn::success(0)),
                Err(e) => EcallAction::Continue(SbiReturn::from(e)),
            },
            HartStop => EcallAction::Break(VmExitCause::CpuStop, SbiReturn::success(0)),
            HartStatus { hart_id } => EcallAction::Continue(self.get_vcpu_status(hart_id).into()),
            _ => EcallAction::Unhandled,
        }
    }

    fn handle_tee_msg(&self, tee_func: TeeFunction, active_pages: &ActiveVmPages<T>) -> SbiReturn {
        use TeeFunction::*;
        match tee_func {
            TsmGetInfo { dest_addr, len } => self.get_tsm_info(dest_addr, len, active_pages).into(),
            TvmCreate { params_addr, len } => self.add_guest(params_addr, len, active_pages).into(),
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
            TsmInitiateFence => self
                .vm_pages
                .initiate_fence()
                .map_err(|_| SbiError::Failed)
                .map(|_| 0)
                .into(),
            TsmLocalFence => {
                // Nothing to do here as the fence itself will occur once we re-enter `VmPages` the
                // next time we're run.
                SbiReturn::success(0)
            }
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
                    active_pages,
                )
                .into(),
            Finalize { guest_id } => self.guest_finalize(guest_id).into(),
            TvmCpuRun { guest_id, vcpu_id } => self.guest_run_vcpu(guest_id, vcpu_id).into(),
            TvmCpuCreate { guest_id, vcpu_id } => self.guest_add_vcpu(guest_id, vcpu_id).into(),
            TvmCpuSetRegister {
                guest_id,
                vcpu_id,
                register,
                value,
            } => self
                .guest_set_vcpu_reg(guest_id, vcpu_id, register, value)
                .into(),
            TvmCpuGetRegister {
                guest_id,
                vcpu_id,
                register,
            } => self.guest_get_vcpu_reg(guest_id, vcpu_id, register).into(),
            GetGuestMeasurement {
                measurement_version,
                measurement_type,
                dest_addr,
                guest_id,
            } => self
                .guest_get_measurement(
                    measurement_version,
                    measurement_type,
                    dest_addr,
                    guest_id,
                    active_pages,
                )
                .into(),
        }
    }

    fn handle_measurement_msg(
        &self,
        measurement_func: MeasurementFunction,
        active_pages: &ActiveVmPages<T>,
    ) -> SbiReturn {
        use MeasurementFunction::*;
        match measurement_func {
            GetSelfMeasurement {
                measurement_version,
                measurement_type,
                dest_addr,
            } => self
                .guest_get_measurement(
                    measurement_version,
                    measurement_type,
                    dest_addr,
                    GUEST_ID_SELF_MEASUREMENT,
                    active_pages,
                )
                .into(),
        }
    }

    fn handle_attestation_msg(
        &self,
        attestation_func: AttestationFunction,
        active_pages: &ActiveVmPages<T>,
    ) -> SbiReturn {
        use AttestationFunction::*;
        match attestation_func {
            GetEvidence {
                csr_addr,
                csr_len,
                cert_addr,
                cert_len,
            } => self
                .guest_get_evidence(
                    csr_addr,
                    csr_len as usize,
                    cert_addr,
                    cert_len as usize,
                    active_pages,
                )
                .into(),

            ExtendMeasurement {
                measurement_addr,
                len,
            } => self
                .guest_extend_measurement(measurement_addr, len as usize, active_pages)
                .into(),
        }
    }

    fn get_tsm_info(
        &self,
        dest_addr: u64,
        len: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> sbi::Result<u64> {
        let dest_addr = RawAddr::guest(dest_addr, self.vm_pages.page_owner_id());
        if len < mem::size_of::<sbi::TsmInfo>() as u64 {
            return Err(SbiError::InvalidParam);
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
            .map_err(|_| SbiError::InvalidAddress)?;
        Ok(len as u64)
    }

    /// Converts `num_pages` starting at guest physical address `page_addr` to confidential memory.
    fn convert_pages(
        &self,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
    ) -> sbi::Result<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO: Support converting hugepages.
            return Err(SbiError::InvalidParam);
        }
        let page_addr = self.guest_addr_from_raw(page_addr)?;
        self.vm_pages
            .convert_pages(page_addr, num_pages)
            .map_err(|_| SbiError::InvalidAddress)?;
        Ok(num_pages)
    }

    /// Reclaims `num_pages` of confidential memory starting at guest physical address `page_addr`.
    fn reclaim_pages(
        &self,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
    ) -> sbi::Result<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO: Support converting hugepages.
            return Err(SbiError::InvalidParam);
        }
        let page_addr = self.guest_addr_from_raw(page_addr)?;
        self.vm_pages
            .reclaim_pages(page_addr, num_pages)
            .map_err(|_| SbiError::InvalidAddress)?;
        Ok(num_pages)
    }

    fn add_guest(
        &self,
        params_addr: u64,
        len: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> sbi::Result<u64> {
        if self.guests.is_none() {
            return Err(SbiError::InvalidParam); // TODO different error
        }

        // Read the params from the VM's address space.
        let params_addr = RawAddr::guest(params_addr, self.vm_pages.page_owner_id());
        if len < mem::size_of::<sbi::TvmCreateParams>() as u64 {
            return Err(SbiError::InvalidParam);
        }
        let mut param_bytes = [0u8; mem::size_of::<sbi::TvmCreateParams>()];
        active_pages
            .copy_from_guest(param_bytes.as_mut_slice(), params_addr)
            .map_err(|_| SbiError::InvalidAddress)?;
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
            .vm_pages
            .create_guest_vm(page_root_addr, state_addr, vcpu_addr, num_vcpu_pages)
            .map_err(|_| SbiError::InvalidParam)?;
        let id = guest_vm.page_owner_id();

        let guest_state = GuestState::new(guest_vm, state_page);
        self.guests
            .as_ref()
            .unwrap()
            .add(guest_state)
            .map_err(|_| SbiError::Failed)?;

        Ok(id.raw())
    }

    fn destroy_guest(&self, guest_id: u64) -> sbi::Result<u64> {
        let guest_id = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        self.guests
            .as_ref()
            .and_then(|g| g.remove(guest_id).ok())
            .ok_or(SbiError::InvalidParam)?;
        Ok(0)
    }

    // converts the given guest from init to running
    fn guest_finalize(&self, guest_id: u64) -> sbi::Result<u64> {
        let guest_id = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        let guest = self
            .guests
            .as_ref()
            .and_then(|g| g.finalize(guest_id).ok())
            .ok_or(SbiError::InvalidParam)?;
        let guest_vm = guest.as_finalized_vm().unwrap();

        // Power on vCPU0 initially. Remaining vCPUs will get powered on by the VM itself via
        // HSM SBI calls.
        //
        // TODO: Should the boot vCPU be specified explicilty?
        guest_vm.power_on_vcpu(0)?;
        Ok(0)
    }

    /// Retrieves the guest VM with the ID `guest_id`.
    fn guest_by_id(&self, guest_id: u64) -> sbi::Result<GuestState<T>> {
        let guest_id = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        let guest = self
            .guests
            .as_ref()
            .and_then(|g| g.get(guest_id))
            .ok_or(SbiError::InvalidParam)?;
        Ok(guest)
    }

    /// Adds a vCPU with `vcpu_id` to a guest VM.
    fn guest_add_vcpu(&self, guest_id: u64, vcpu_id: u64) -> sbi::Result<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest.as_initializing_vm().ok_or(SbiError::InvalidParam)?;
        guest_vm.add_vcpu(vcpu_id)?;
        Ok(0)
    }

    /// Sets a register in a guest VM's vCPU.
    fn guest_set_vcpu_reg(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        register: TvmCpuRegister,
        value: u64,
    ) -> sbi::Result<u64> {
        let guest = self.guest_by_id(guest_id)?;
        // The only writeable registers are writeable only during initialization.
        let guest_vm = guest.as_initializing_vm().ok_or(SbiError::InvalidParam)?;
        guest_vm.set_vcpu_reg(vcpu_id, register, value)?;
        Ok(0)
    }

    /// Gets a register in a guest VM's vCPU.
    fn guest_get_vcpu_reg(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        register: TvmCpuRegister,
    ) -> sbi::Result<u64> {
        let guest = self.guest_by_id(guest_id)?;
        use TvmCpuRegister::*;
        let value = match register {
            EntryArg | EntryPc => {
                let guest_vm = guest.as_initializing_vm().ok_or(SbiError::InvalidParam)?;
                guest_vm.get_vcpu_reg(vcpu_id, register)?
            }
            ExitCause0 | ExitCause1 => {
                let guest_vm = guest.as_finalized_vm().ok_or(SbiError::InvalidParam)?;
                guest_vm.get_vcpu_reg(vcpu_id, register)?
            }
        };
        Ok(value)
    }

    /// Runs a guest VM's vCPU.
    fn guest_run_vcpu(&self, guest_id: u64, vcpu_id: u64) -> sbi::Result<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest.as_finalized_vm().ok_or(SbiError::InvalidParam)?;
        let exit_code = guest_vm.run_vcpu(vcpu_id)?;
        Ok(exit_code as u64)
    }

    fn guest_add_page_table_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        num_pages: u64,
    ) -> sbi::Result<u64> {
        let from_page_addr = self.guest_addr_from_raw(from_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        if let Some(vm) = guest.as_initializing_vm() {
            self.vm_pages
                .add_pte_pages_to(from_page_addr, num_pages, &vm.vm_pages)
        } else if let Some(vm) = guest.as_finalized_vm() {
            self.vm_pages
                .add_pte_pages_to(from_page_addr, num_pages, &vm.vm_pages)
        } else {
            return Err(SbiError::InvalidParam);
        }
        .map_err(|_| SbiError::InvalidAddress)?;

        Ok(0)
    }

    /// Adds a region of confidential memory to the specified guest.
    fn guest_add_confidential_memory_region(
        &self,
        guest_id: u64,
        guest_addr: u64,
        len: u64,
    ) -> sbi::Result<u64> {
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest.as_initializing_vm().ok_or(SbiError::InvalidParam)?;
        let page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        guest_vm
            .vm_pages
            .add_confidential_memory_region(page_addr, len)
            .map_err(|_| SbiError::InvalidAddress)?;
        Ok(0)
    }

    fn guest_add_zero_pages(
        &self,
        guest_id: u64,
        page_addr: u64,
        page_type: sbi::TsmPageType,
        num_pages: u64,
        guest_addr: u64,
    ) -> sbi::Result<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(SbiError::InvalidParam);
        }

        let from_page_addr = self.guest_addr_from_raw(page_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        let to_page_addr = PageAddr::new(RawAddr::guest(guest_addr, guest.page_owner_id()))
            .ok_or(SbiError::InvalidParam)?;

        // Zero pages may be added to either running or initialized VMs.
        if let Some(vm) = guest.as_initializing_vm() {
            self.vm_pages
                .add_zero_pages_to(from_page_addr, num_pages, &vm.vm_pages, to_page_addr)
        } else if let Some(vm) = guest.as_finalized_vm() {
            self.vm_pages
                .add_zero_pages_to(from_page_addr, num_pages, &vm.vm_pages, to_page_addr)
        } else {
            return Err(SbiError::InvalidParam);
        }
        .map_err(|_| SbiError::InvalidAddress)?;

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
    ) -> sbi::Result<u64> {
        if page_type != sbi::TsmPageType::Page4k {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(SbiError::InvalidParam);
        }

        let src_page_addr = self.guest_addr_from_raw(src_addr)?;
        let from_page_addr = self.guest_addr_from_raw(dest_addr)?;
        let guest = self.guest_by_id(guest_id)?;
        let guest_vm = guest.as_initializing_vm().ok_or(SbiError::InvalidParam)?;
        let to_page_addr = guest_vm.guest_addr_from_raw(guest_addr)?;
        active_pages
            .copy_and_add_data_pages_builder(
                src_page_addr,
                from_page_addr,
                num_pages,
                &guest_vm.vm_pages,
                to_page_addr,
            )
            .map_err(|_| SbiError::InvalidParam)?;

        Ok(num_pages)
    }

    // TODO: Add code to return actual measurements
    fn guest_get_measurement(
        &self,
        measurement_version: u64,
        measurement_type: u64,
        dest_addr: u64,
        guest_id: u64,
        active_pages: &ActiveVmPages<T>,
    ) -> sbi::Result<u64> {
        if (measurement_version != 1) || (measurement_type != 1) {
            return Err(SbiError::InvalidParam);
        }

        // TODO: Define a compile-time constant for the maximum length of any measurement we
        // would conceivably use.
        let mut bytes = [0u8; SHA256_DIGEST_BYTES];
        if guest_id == GUEST_ID_SELF_MEASUREMENT {
            // The guest_id of 0 is a special identifier used to retrieve
            // measurements for self.
            self.vm_pages.get_measurement(&mut bytes)
        } else {
            let guest_id = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
            let guest = self
                .guests
                .as_ref()
                .and_then(|g| g.get(guest_id))
                .ok_or(SbiError::InvalidParam)?;
            let result = if let Some(vm) = guest.as_finalized_vm() {
                vm.vm_pages.get_measurement(&mut bytes)
            } else {
                guest
                    .as_initializing_vm()
                    .unwrap()
                    .vm_pages
                    .get_measurement(&mut bytes)
            };
            result
        }
        .map_err(|_| SbiError::Failed)?;

        let gpa = RawAddr::guest(dest_addr, self.vm_pages.page_owner_id());
        active_pages
            .copy_to_guest(gpa, &bytes)
            .map_err(|_| SbiError::InvalidAddress)?;
        Ok(bytes.len() as u64)
    }

    fn guest_get_evidence(
        &self,
        csr_addr: u64,
        csr_len: usize,
        _cert_addr: u64,
        _cert_len: usize,
        active_pages: &ActiveVmPages<T>,
    ) -> sbi::Result<u64> {
        if csr_len > MAX_CSR_LEN {
            return Err(SbiError::InvalidParam);
        }

        let mut csr_bytes = [0u8; MAX_CSR_LEN];
        let csr_gpa = RawAddr::guest(csr_addr, self.vm_pages.page_owner_id());
        active_pages
            .copy_from_guest(&mut csr_bytes.as_mut_slice()[..csr_len], csr_gpa)
            .map_err(|_| SbiError::InvalidAddress)?;
        println!("CSR len {}", csr_len);

        let csr = CertReq::from_der(&csr_bytes[..csr_len]).map_err(|_| SbiError::InvalidParam)?;
        println!(
            "CSR version {:?} Signature algorithm {:?}",
            csr.info.version, csr.algorithm.oid
        );

        csr.verify().map_err(|_| SbiError::InvalidParam)?;

        Ok(0)
    }

    fn guest_extend_measurement(
        &self,
        _msmt_addr: u64,
        _len: usize,
        _active_pages: &ActiveVmPages<T>,
    ) -> sbi::Result<u64> {
        Err(SbiError::NotSupported)
    }

    /// Destroys this `Vm`.
    pub fn destroy(&mut self) {
        // Recursively destroy this VM's children before we drop() this VM so that any donated pages
        // are guaranteed to have been returned before we destroy this VM's page table. This could
        // also be done by implementing Drop for Guests, but doing explicitly avoids relying on struct
        // field ordering for proper drop() ordering.
        self.guests = None;

        let page_tracker = self.vm_pages.page_tracker();
        page_tracker.rm_active_guest(self.vm_pages.page_owner_id());
    }
}

/// Represents the special VM that serves as the host for the system.
pub struct HostVm<T: GuestStagePageTable, S = VmStateFinalized> {
    inner: Vm<T, S>,
}

impl<T: GuestStagePageTable> HostVm<T, VmStateInitializing> {
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
        let guest_tracking_pages = hyp_mem.take_pages_for_host_state(2);
        let region_vec_pages = hyp_mem.take_pages_for_host_state(TVM_REGION_LIST_PAGES as usize);

        // Pages for the array of vCPUs.
        let num_vcpu_pages = PageSize::num_4k_pages(VM_CPU_BYTES * MAX_CPUS as u64);
        let vcpus_pages = hyp_mem.take_pages_for_host_state(num_vcpu_pages as usize);

        let (page_tracker, host_pages) = PageTracker::from(hyp_mem, T::TOP_LEVEL_ALIGN);
        let root =
            PlatformPageTable::new(root_table_pages, PageOwnerId::host(), page_tracker.clone())
                .unwrap();
        let region_vec = VmRegionList::new(region_vec_pages, page_tracker.clone());
        let vm_pages = VmPages::new(root, region_vec, 0);
        for p in pte_pages {
            vm_pages.add_pte_page(p).unwrap();
        }
        let mut vm = Vm::new(
            vm_pages,
            VmCpus::new(PageOwnerId::host(), vcpus_pages, page_tracker).unwrap(),
        );
        vm.add_guest_tracking_pages(guest_tracking_pages);

        let cpu_info = CpuInfo::get();
        for i in 0..cpu_info.num_cpus() {
            vm.add_vcpu(i as u64).unwrap();
        }

        (host_pages, Self { inner: vm })
    }

    /// Sets the launch arguments (entry point and FDT) for the host vCPU.
    pub fn set_launch_args(&self, entry_addr: GuestPhysAddr, fdt_addr: GuestPhysAddr) {
        self.inner
            .set_vcpu_reg(0, TvmCpuRegister::EntryPc, entry_addr.bits())
            .unwrap();
        self.inner
            .set_vcpu_reg(0, TvmCpuRegister::EntryArg, fdt_addr.bits())
            .unwrap();
    }

    /// Adds a region of confidential memory to the host VM.
    pub fn add_confidential_memory_region(&mut self, addr: GuestPageAddr, len: u64) {
        self.inner
            .vm_pages
            .add_confidential_memory_region(addr, len)
            .unwrap();
    }

    /// Adds data pages that are measured and mapped to the page tables for the host. Requires
    /// that the GPA map the SPA in T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    pub fn add_measured_pages<I, S, M>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = Page<S>>,
        S: Assignable<M>,
        M: MeasureRequirement,
    {
        let page_tracker = self.inner.vm_pages.page_tracker();
        // Unwrap ok since we've donate sufficient PT pages to map the entire address space up front.
        let mapper = self
            .inner
            .vm_pages
            .map_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            let mappable = page_tracker
                .assign_page_for_mapping(page, self.inner.page_owner_id())
                .unwrap();
            mapper.map_page_with_measurement(vm_addr, mappable).unwrap();
        }
    }

    /// Add pages which need not be measured to the host page tables. For RAM pages, requires that
    /// the GPA map the SPA in T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    pub fn add_pages<I, P>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = P>,
        P: AssignablePhysPage<MeasureOptional>,
    {
        let page_tracker = self.inner.vm_pages.page_tracker();
        // Unwrap ok since we've donate sufficient PT pages to map the entire address space up front.
        let mapper = self
            .inner
            .vm_pages
            .map_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            if P::mem_type() == MemType::Ram {
                // GPA -> SPA mappings need to match T::TOP_LEVEL_ALIGN alignment for RAM pages.
                assert_eq!(
                    vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                    page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
                );
            }
            let mappable = page_tracker
                .assign_page_for_mapping(page, self.inner.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    /// Completes intialization of the host, returning it in a finalized state.
    pub fn finalize(self) -> HostVm<T, VmStateFinalized> {
        HostVm {
            inner: self.inner.finalize(),
        }
    }
}

impl<T: GuestStagePageTable> HostVm<T, VmStateFinalized> {
    /// Run the host VM's vCPU with ID `vcpu_id`. Does not return.
    pub fn run(&self, vcpu_id: u64) {
        self.inner.bind_vcpu(vcpu_id, ImsicGuestId::HostVm).unwrap();

        // Always make vCPU0 the boot CPU.
        if vcpu_id == 0 {
            self.inner.power_on_vcpu(0).unwrap();
        }

        loop {
            // Wait until this vCPU is ready to run.
            while !self.vcpu_is_runnable(vcpu_id) {
                smp::wfi();
            }

            // Run until we shut down, or this vCPU stops.
            loop {
                use TvmCpuExitCode::*;
                match self.inner.run_vcpu(vcpu_id).unwrap() {
                    SystemReset => {
                        println!("Host VM requested shutdown");
                        poweroff();
                    }
                    HartStart => {
                        let id = self
                            .inner
                            .get_vcpu_reg(vcpu_id, TvmCpuRegister::ExitCause0)
                            .unwrap();
                        smp::send_ipi(CpuId::new(id as usize));
                    }
                    HartStop => {
                        break;
                    }
                    _ => {
                        println!("Unhandled host VM exit; shutting down");
                        poweroff();
                    }
                };
            }
        }
    }

    /// Returns if the vCPU with `vcpu_id` is runnable.
    fn vcpu_is_runnable(&self, vcpu_id: u64) -> bool {
        matches!(
            self.inner.vcpus.get_vcpu_status(vcpu_id),
            Ok(VmCpuStatus::Runnable)
        )
    }
}
