// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use core::{alloc::Allocator, mem, slice};
use drivers::{CpuId, CpuInfo, ImsicGuestId, MAX_CPUS};
use page_collections::page_box::PageBox;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{GuestStagePageTable, HypPageAlloc, PageState};
use riscv_pages::{
    GuestPageAddr, GuestPhysAddr, MemType, Page, PageAddr, PageOwnerId, PageSize, PhysPage,
    RawAddr, SequentialPages,
};
use riscv_regs::GprIndex;
use sbi::Error as SbiError;
use sbi::{
    self, BaseFunction, HartState, MeasurementFunction, SbiMessage, SbiReturn, StateFunction,
    TeeFunction,
};
use spin::RwLock;

use crate::print_util::*;
use crate::sha256_measure::SHA256_DIGEST_BYTES;
use crate::smp;
use crate::vm_cpu::{VmCpuExit, VmCpuStatus, VmCpus, VM_CPU_BYTES};
use crate::vm_pages::{self, ActiveVmPages, VmPages, TVM_CREATE_PAGES};
use crate::{print, println};

const GUEST_ID_SELF_MEASUREMENT: u64 = 0;

// What we report ourselves as in sbi_get_sbi_impl_id(). Just pick something unclaimed so no one
// confuses us with BBL/OpenSBI.
const SBI_IMPL_ID_SALUS: u64 = 7;

pub enum VmStateInitializing {}
pub enum VmStateFinalized {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmExitCause {
    PowerOff,
    CpuStart(u64),
    CpuStop,
    UnhandledTrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EcallAction {
    LegacyOk,
    Unhandled,
    Continue(SbiReturn),
    Break(VmExitCause, SbiReturn),
}

struct Guests<T: GuestStagePageTable> {
    inner: PageVec<PageBox<GuestState<T>>>,
    phys_pages: PageState,
}

impl<T: GuestStagePageTable> Guests<T> {
    fn add(&mut self, guest_state: PageBox<GuestState<T>>) -> sbi::Result<()> {
        self.inner
            .try_reserve(1)
            .map_err(|_| SbiError::InvalidParam)?;
        self.inner.push(guest_state);
        Ok(())
    }

    // Returns the index in to the guest array of the given guest id. If the guest ID isn't valid or
    // isn't owned by this VM, then return an error.
    fn get_guest_index(&self, guest_id: u64) -> sbi::Result<usize> {
        let page_owner_id = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        self.inner
            .iter()
            .enumerate()
            .find(|(_i, g)| page_owner_id == g.page_owner_id())
            .map(|(guest_index, _guest)| guest_index)
            .ok_or(SbiError::InvalidParam)
    }

    fn remove(&mut self, guest_id: u64) -> sbi::Result<()> {
        let to_remove = PageOwnerId::new(guest_id).ok_or(SbiError::InvalidParam)?;
        self.inner.retain(|g| {
            if g.page_owner_id() == to_remove {
                self.phys_pages.rm_active_guest(g.page_owner_id());
                true
            } else {
                false
            }
        });
        Ok(())
    }

    // Returns a mutable reference to the guest for the given ID if it exists, otherwise None.
    fn guest_mut(&mut self, guest_id: u64) -> sbi::Result<&mut PageBox<GuestState<T>>> {
        let guest_index = self.get_guest_index(guest_id)?;
        self.inner
            .get_mut(guest_index)
            .ok_or(SbiError::InvalidParam)
    }

    // Returns an immutable reference to the guest for the given ID if it exists, otherwise None.
    fn guest(&self, guest_id: u64) -> sbi::Result<&PageBox<GuestState<T>>> {
        let guest_index = self.get_guest_index(guest_id)?;
        self.inner.get(guest_index).ok_or(SbiError::InvalidParam)
    }

    // Returns the initializing guest if it's present, otherwise None.
    fn initializing_guest(&self, guest_id: u64) -> sbi::Result<&Vm<T, VmStateInitializing>> {
        self.guest(guest_id)
            .and_then(|g| g.as_initializing_vm().ok_or(SbiError::InvalidParam))
    }

    // Returns the runnable guest if it's present, otherwise None
    fn running_guest(&self, guest_id: u64) -> sbi::Result<&Vm<T, VmStateFinalized>> {
        self.guest(guest_id)
            .and_then(|g| g.as_finalized_vm().ok_or(SbiError::InvalidParam))
    }
}

enum GuestState<T: GuestStagePageTable> {
    Init(Vm<T, VmStateInitializing>),
    Running(Vm<T, VmStateFinalized>),
    Temp,
}

impl<T: GuestStagePageTable> GuestState<T> {
    fn page_owner_id(&self) -> PageOwnerId {
        match self {
            Self::Init(v) => v.vm_pages.page_owner_id(),
            Self::Running(v) => v.vm_pages.page_owner_id(),
            Self::Temp => unreachable!(),
        }
    }

    fn as_initializing_vm(&self) -> Option<&Vm<T, VmStateInitializing>> {
        match self {
            Self::Init(ref v) => Some(v),
            Self::Running(_) => None,
            Self::Temp => unreachable!(),
        }
    }

    fn as_finalized_vm(&self) -> Option<&Vm<T, VmStateFinalized>> {
        match self {
            Self::Init(_) => None,
            Self::Running(ref v) => Some(v),
            Self::Temp => unreachable!(),
        }
    }
}

/// A VM that is being run.
pub struct Vm<T: GuestStagePageTable, S = VmStateFinalized> {
    vcpus: VmCpus,
    vm_pages: VmPages<T, S>,
    guests: Option<RwLock<Guests<T>>>,
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

    /// Sets a vCPU register.
    fn set_vcpu_reg(
        &self,
        vcpu_id: u64,
        register: sbi::TvmCpuRegister,
        value: u64,
    ) -> sbi::Result<()> {
        let vcpu = self
            .vcpus
            .get_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let mut vcpu = vcpu.lock();
        match register {
            sbi::TvmCpuRegister::Pc => vcpu.set_sepc(value),
            sbi::TvmCpuRegister::A1 => vcpu.set_gpr(GprIndex::A1, value),
        };
        Ok(())
    }

    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    fn add_guest_tracking_pages(&mut self, pages: SequentialPages) {
        let guests = PageVec::from(pages);
        self.guests = Some(RwLock::new(Guests {
            inner: guests,
            phys_pages: self.vm_pages.phys_pages(),
        }));
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
    fn finalize(self) -> Vm<T, VmStateFinalized> {
        Vm {
            vcpus: self.vcpus,
            vm_pages: self.vm_pages.finalize(),
            guests: self.guests,
        }
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

    /// Run this guest until an unhandled exit is encountered.
    fn run_vcpu(&self, vcpu_id: u64) -> sbi::Result<VmExitCause> {
        // Take the vCPU out of self.vcpus, giving us exclusive ownership.
        let mut vcpu = self
            .vcpus
            .take_vcpu(vcpu_id)
            .map_err(|_| SbiError::InvalidParam)?;
        let exit = {
            let mut vcpu = vcpu.lock();

            // Activate this vCPU and its address space.
            let mut active_vcpu = vcpu.activate(&self.vm_pages).unwrap();

            // Run until there's an exit we can't handle.
            loop {
                let exit = active_vcpu.run_to_exit();
                match exit {
                    VmCpuExit::Ecall(Some(sbi_msg)) => {
                        match self.handle_ecall(sbi_msg, active_vcpu.active_pages()) {
                            EcallAction::LegacyOk => {
                                // for legacy, leave the a0 and a1 registers as-is.
                            }
                            EcallAction::Unhandled => {
                                active_vcpu
                                    .set_ecall_result(SbiReturn::from(sbi::Error::NotSupported));
                            }
                            EcallAction::Continue(sbi_ret) => {
                                active_vcpu.set_ecall_result(sbi_ret);
                            }
                            EcallAction::Break(reason, sbi_ret) => {
                                active_vcpu.set_ecall_result(sbi_ret);
                                break reason;
                            }
                        }
                    }
                    VmCpuExit::Ecall(None) => {
                        // Unrecognized ECALL, return an error.
                        active_vcpu.set_ecall_result(SbiReturn::from(sbi::Error::NotSupported));
                    }
                    VmCpuExit::PageFault(addr) => {
                        if self.handle_guest_fault(addr).is_err() {
                            break VmExitCause::UnhandledTrap;
                        }
                    }
                    VmCpuExit::DelegatedException(e, stval) => {
                        active_vcpu.inject_exception(e, stval);
                    }
                    VmCpuExit::Other(ref trap_csrs) => {
                        println!("Unhandled guest exit, SCAUSE = 0x{:08x}", trap_csrs.scause);
                        break VmExitCause::UnhandledTrap;
                    }
                }
            }
        };
        if matches!(exit, VmExitCause::PowerOff | VmExitCause::CpuStop) {
            vcpu.power_off();
        }
        Ok(exit)
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&self, msg: SbiMessage, active_pages: &ActiveVmPages<T>) -> EcallAction {
        match msg {
            SbiMessage::PutChar(c) => {
                // put char - legacy command
                print!("{}", c as u8 as char);
                EcallAction::LegacyOk
            }
            SbiMessage::Reset(_) => {
                EcallAction::Break(VmExitCause::PowerOff, SbiReturn::success(0))
            }
            SbiMessage::Base(base_func) => EcallAction::Continue(self.handle_base_msg(base_func)),
            SbiMessage::HartState(hsm_func) => self.handle_hart_state_msg(hsm_func),
            SbiMessage::Tee(tee_func) => {
                EcallAction::Continue(self.handle_tee_msg(tee_func, active_pages))
            }
            SbiMessage::Measurement(measurement_func) => {
                EcallAction::Continue(self.handle_measurement_msg(measurement_func, active_pages))
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
            TvmCreate(state_page) => self.add_guest(state_page).into(),
            TvmDestroy { guest_id } => self.destroy_guest(guest_id).into(),
            AddPageTablePages {
                guest_id,
                page_addr,
                num_pages,
            } => self
                .guest_add_page_table_pages(guest_id, page_addr, num_pages)
                .into(),
            AddPages {
                guest_id,
                page_addr,
                page_type,
                num_pages,
                gpa,
                measure_preserve,
            } => self
                .guest_add_pages(
                    guest_id,
                    page_addr,
                    page_type,
                    num_pages,
                    gpa,
                    measure_preserve,
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
            RemovePages {
                guest_id,
                gpa,
                remap_addr: _, // TODO - remove
                num_pages,
            } => self.guest_rm_pages(guest_id, gpa, num_pages).into(),
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

    // Handle access faults. For example, when a returned page needs to be demand-faulted back to
    // the page table.
    fn handle_guest_fault(&self, fault_addr: GuestPhysAddr) -> vm_pages::Result<()> {
        self.vm_pages.handle_page_fault(fault_addr)?;

        // Get instruction that caused the fault
        //   - disable ints
        //   - load hstatus with value from guest
        //   - set stvec to catch traps during access
        //   - read instruction using HLV.HU (or tow for 32 bit).
        //   - reset stvec
        //   - reset hstatus
        //   - re-enable ints

        // Determine width of faulting access
        // determine destination/source register
        // Check how to service access (device emulation for example) and run.
        // if load, set destination register

        Ok(())
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
            tvm_create_pages: TVM_CREATE_PAGES,
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

    fn add_guest(&self, donor_pages_addr: u64) -> sbi::Result<u64> {
        println!("Add guest {:x}", donor_pages_addr);
        if self.guests.is_none() {
            return Err(SbiError::InvalidParam); // TODO different error
        }

        let from_page_addr = PageAddr::new(RawAddr::guest(
            donor_pages_addr,
            self.vm_pages.page_owner_id(),
        ))
        .ok_or(SbiError::InvalidAddress)?;

        let (guest_vm, state_page) = self
            .vm_pages
            .create_guest_vm(from_page_addr)
            .map_err(|_| SbiError::InvalidParam)?;
        let id = guest_vm.vm_pages.page_owner_id();

        // create a boxpage for builder state and add it to the list of vms.
        let guest_state: PageBox<GuestState<T>> =
            PageBox::new_with(GuestState::Init(guest_vm), state_page);

        let mut guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.write();
        guests.add(guest_state)?;

        Ok(id.raw())
    }

    fn destroy_guest(&self, guest_id: u64) -> sbi::Result<u64> {
        let mut guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.write();
        guests.remove(guest_id)?;
        Ok(0)
    }

    // converts the given guest from init to running
    fn guest_finalize(&self, guest_id: u64) -> sbi::Result<u64> {
        let mut guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.write();
        guests.guest_mut(guest_id).map(|g| {
            let mut temp = GuestState::Temp;
            core::mem::swap(&mut **g, &mut temp);
            let mut running = match temp {
                GuestState::Init(v) => GuestState::Running(v.finalize()),
                t => t,
            };
            core::mem::swap(&mut **g, &mut running);
        })?;
        let guest_vm = guests.running_guest(guest_id)?;
        // Power on vCPU0 initially. Remaining vCPUs will get powered on by the VM itself via
        // HSM SBI calls.
        //
        // TODO: Should the boot vCPU be specified explicilty?
        guest_vm.power_on_vcpu(0)?;
        Ok(0)
    }

    /// Adds a vCPU with `vcpu_id` to a guest VM.
    fn guest_add_vcpu(&self, guest_id: u64, vcpu_id: u64) -> sbi::Result<u64> {
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.initializing_guest(guest_id)?;
        guest_vm.add_vcpu(vcpu_id)?;
        Ok(0)
    }

    /// Sets a register in a guest VM's vCPU.
    fn guest_set_vcpu_reg(
        &self,
        guest_id: u64,
        vcpu_id: u64,
        register: sbi::TvmCpuRegister,
        value: u64,
    ) -> sbi::Result<u64> {
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.initializing_guest(guest_id)?;
        guest_vm.set_vcpu_reg(vcpu_id, register, value)?;
        Ok(0)
    }

    /// Runs a guest VM's vCPU.
    fn guest_run_vcpu(&self, guest_id: u64, vcpu_id: u64) -> sbi::Result<u64> {
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.running_guest(guest_id)?;
        guest_vm.run_vcpu(vcpu_id)?;
        Ok(0) // TODO: Return the exit reason to the host.
    }

    fn guest_add_page_table_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        num_pages: u64,
    ) -> sbi::Result<u64> {
        let from_page_addr =
            PageAddr::new(RawAddr::guest(from_addr, self.vm_pages.page_owner_id()))
                .ok_or(SbiError::InvalidAddress)?;

        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.initializing_guest(guest_id)?;
        self.vm_pages
            .add_pte_pages_builder(from_page_addr, num_pages, &guest_vm.vm_pages)
            .map_err(|e| {
                println!("Salus - pte_pages_builder error {e:?}");
                SbiError::InvalidAddress
            })?;

        Ok(0)
    }

    fn guest_rm_pages(&self, guest_id: u64, gpa: u64, num_pages: u64) -> sbi::Result<u64> {
        println!("Salus - Rm pages {guest_id:x} gpa:{gpa:x} num_pages:{num_pages}",);
        let from_page_addr = PageAddr::new(RawAddr::guest(gpa, self.vm_pages.page_owner_id()))
            .ok_or(SbiError::InvalidAddress)?;

        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        // TODO: Enforce that pages can't be removed while the guest is alive.
        let guest_vm = guests.running_guest(guest_id)?;
        guest_vm
            .vm_pages
            .remove_4k_pages(from_page_addr, num_pages)
            .map_err(|e| {
                println!("Salus - remove_4k_pages error {e:?}");
                SbiError::InvalidAddress
            })
    }

    /// page_type: 0 => 4K, 1=> 2M, 2=> 1G, 3=512G
    fn guest_add_pages(
        &self,
        guest_id: u64,
        from_addr: u64,
        page_type: u64,
        num_pages: u64,
        to_addr: u64,
        measure_preserve: bool,
    ) -> sbi::Result<u64> {
        println!(
            "Add pages {from_addr:x} page_type:{page_type} num_pages:{num_pages} to_addr:{to_addr:x}",
        );
        if page_type != 0 {
            // TODO - support huge pages.
            // TODO - need to break up mappings if given address that's part of a huge page.
            return Err(SbiError::InvalidParam);
        }

        let from_page_addr =
            PageAddr::new(RawAddr::guest(from_addr, self.vm_pages.page_owner_id()))
                .ok_or(SbiError::InvalidAddress)?;
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.initializing_guest(guest_id)?;
        let to_page_addr =
            PageAddr::new(RawAddr::guest(to_addr, guest_vm.vm_pages.page_owner_id()))
                .ok_or(SbiError::InvalidAddress)?;
        self.vm_pages
            .add_4k_pages_builder(
                from_page_addr,
                num_pages,
                &guest_vm.vm_pages,
                to_page_addr,
                measure_preserve,
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
            let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
            let _ = guests.get_guest_index(guest_id)?;
            if let Ok(running_guest) = guests.running_guest(guest_id) {
                running_guest.vm_pages.get_measurement(&mut bytes)
            } else {
                guests
                    .initializing_guest(guest_id)
                    .unwrap()
                    .vm_pages
                    .get_measurement(&mut bytes)
            }
        }
        .map_err(|_| SbiError::Failed)?;

        let gpa = RawAddr::guest(dest_addr, self.vm_pages.page_owner_id());
        active_pages
            .copy_to_guest(gpa, &bytes)
            .map_err(|_| SbiError::InvalidAddress)?;
        Ok(bytes.len() as u64)
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
    pub fn from_hyp_mem<A: Allocator>(
        mut hyp_mem: HypPageAlloc<A>,
        host_gpa_size: u64,
    ) -> (Vec<SequentialPages, A>, Self) {
        let root_table_pages = hyp_mem.take_pages_with_alignment(4, T::TOP_LEVEL_ALIGN);
        let num_pte_pages = T::max_pte_pages(host_gpa_size / PageSize::Size4k as u64);
        let pte_pages = hyp_mem.take_pages(num_pte_pages as usize).into_iter();
        let guest_tracking_pages = hyp_mem.take_pages(2);

        // Pages for the array of vCPUs.
        let num_vcpu_pages = PageSize::num_4k_pages(VM_CPU_BYTES * MAX_CPUS as u64);
        let vcpus_pages = hyp_mem.take_pages(num_vcpu_pages as usize);

        // Pages for the array of free page-table pages.
        let num_pte_vec_pages = PageSize::Size4k
            .round_up(num_pte_pages * mem::size_of::<Page>() as u64)
            / (PageSize::Size4k as u64);
        let pte_vec_pages = hyp_mem.take_pages(num_pte_vec_pages as usize);

        let (phys_pages, host_pages) = PageState::from(hyp_mem, T::TOP_LEVEL_ALIGN);
        let root = T::new(root_table_pages, PageOwnerId::host(), phys_pages).unwrap();
        let vm_pages = VmPages::new(root, pte_vec_pages);
        for p in pte_pages {
            vm_pages.add_pte_page(p).unwrap();
        }
        let mut vm = Vm::new(
            vm_pages,
            VmCpus::new(PageOwnerId::host(), vcpus_pages).unwrap(),
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
            .set_vcpu_reg(0, sbi::TvmCpuRegister::Pc, entry_addr.bits())
            .unwrap();
        self.inner
            .set_vcpu_reg(0, sbi::TvmCpuRegister::A1, fdt_addr.bits())
            .unwrap();
    }

    /// Adds data pages that are measured and mapped to the page tables for the host. Requires
    /// that the GPA map the SPA in T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    pub fn add_measured_pages<I>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: Iterator<Item = Page>,
    {
        let phys_pages = self.inner.vm_pages.phys_pages();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(vm_addr.size(), page.addr().size());
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            phys_pages
                .set_page_owner(page.addr(), self.inner.vm_pages.page_owner_id())
                .unwrap();
            self.inner
                .vm_pages
                .add_measured_4k_page(vm_addr, page)
                .unwrap();
        }
    }

    /// Add pages which need not be measured to the host page tables. For RAM pages, requires that
    /// the GPA map the SPA in T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    pub fn add_pages<I, P>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: Iterator<Item = P>,
        P: PhysPage,
    {
        let phys_pages = self.inner.vm_pages.phys_pages();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(vm_addr.size(), page.addr().size());
            if P::mem_type() == MemType::Ram {
                // GPA -> SPA mappings need to match T::TOP_LEVEL_ALIGN alignment for RAM pages.
                assert_eq!(
                    vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                    page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
                );
            }
            phys_pages
                .set_page_owner(page.addr(), self.inner.vm_pages.page_owner_id())
                .unwrap();
            self.inner.vm_pages.add_4k_page(vm_addr, page).unwrap();
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
                match self.inner.run_vcpu(vcpu_id).unwrap() {
                    VmExitCause::PowerOff => {
                        println!("Host VM requested shutdown");
                        crate::poweroff();
                    }
                    VmExitCause::CpuStart(id) => {
                        smp::send_ipi(CpuId::new(id as usize));
                    }
                    VmExitCause::CpuStop => {
                        break;
                    }
                    VmExitCause::UnhandledTrap => {
                        println!("Unhandled host VM exception; shutting down");
                        crate::poweroff();
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
