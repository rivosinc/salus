// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use drivers::ImsicGuestId;
use page_collections::page_box::PageBox;
use page_collections::page_vec::PageVec;
use riscv_page_tables::{GuestStagePageTable, PageState};
use riscv_pages::{GuestPhysAddr, PageAddr, PageOwnerId, RawAddr, SequentialPages};
use sbi::Error as SbiError;
use sbi::{self, MeasurementFunction, ResetFunction, SbiMessage, SbiReturn, TeeFunction};
use spin::{Mutex, RwLock};

use crate::print_util::*;
use crate::sha256_measure::SHA256_DIGEST_BYTES;
use crate::vm_cpu::{VmCpu, VmCpuExit};
use crate::vm_pages::{self, GuestRootBuilder, HostRootPages, VmPages};
use crate::{print, println};

const GUEST_ID_SELF_MEASUREMENT: u64 = 0;

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

    // returns the initializing guest if it's present and runnable, otherwise none
    fn initializing_guest(&self, guest_id: u64) -> sbi::Result<&GuestRootBuilder<T>> {
        self.guest(guest_id)
            .and_then(|g| g.as_guest_root_builder().ok_or(SbiError::InvalidParam))
    }

    // Returns the runnable guest if it's present and runnable, otherwise None
    fn running_guest(&self, guest_id: u64) -> sbi::Result<&Vm<T>> {
        self.guest(guest_id)
            .and_then(|g| g.as_vm().ok_or(SbiError::InvalidParam))
    }
}

enum GuestState<T: GuestStagePageTable> {
    Init(GuestRootBuilder<T>),
    Running(Vm<T>),
    Temp,
}

impl<T: GuestStagePageTable> GuestState<T> {
    fn page_owner_id(&self) -> PageOwnerId {
        match self {
            Self::Init(grb) => grb.page_owner_id(),
            Self::Running(v) => v.vm_pages.page_owner_id(),
            Self::Temp => unreachable!(),
        }
    }

    fn as_guest_root_builder(&self) -> Option<&GuestRootBuilder<T>> {
        match self {
            Self::Init(ref grb) => Some(grb),
            Self::Running(_) => None,
            Self::Temp => unreachable!(),
        }
    }

    fn as_vm(&self) -> Option<&Vm<T>> {
        match self {
            Self::Init(_) => None,
            Self::Running(ref v) => Some(v),
            Self::Temp => unreachable!(),
        }
    }
}

/// A VM that is being run.
pub struct Vm<T: GuestStagePageTable> {
    // TODO: Support multiple vCPUs.
    vcpu: Mutex<VmCpu>,
    vm_pages: VmPages<T>,
    guests: Option<RwLock<Guests<T>>>,
}

impl<T: GuestStagePageTable> Vm<T> {
    /// Create a new guest using the given initial page table and pool of initial pages.
    fn new(vm_pages: VmPages<T>) -> Self {
        Self {
            vcpu: Mutex::new(VmCpu::new(&vm_pages)),
            vm_pages,
            guests: None,
        }
    }

    /// Sets the launch arguments (entry point and A1) for vCPU0.
    fn set_launch_args(&self, entry_addr: GuestPhysAddr, a1: u64) {
        let mut vcpu = self.vcpu.lock();
        vcpu.set_launch_args(entry_addr, a1);
    }

    /// Sets the interrupt file for vCPU0.
    fn set_interrupt_file(&self, interrupt_file: ImsicGuestId) {
        let mut vcpu = self.vcpu.lock();
        vcpu.set_interrupt_file(interrupt_file);
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

    /// Run this guest until an unhandled exit is encountered.
    fn run(&self, _vcpu_id: u64) -> VmCpuExit {
        loop {
            let mut vcpu = self.vcpu.lock();
            let exit = vcpu.run_to_exit();
            match exit {
                VmCpuExit::Ecall(Some(sbi_msg)) => {
                    match self.handle_ecall(sbi_msg) {
                        Ok(Some(sbi_ret)) => {
                            vcpu.set_ecall_result(sbi_ret);
                        }
                        Ok(None) => {
                            // for legacy, leave the a0 and a1 registers as-is.
                        }
                        Err(error_code) => {
                            vcpu.set_ecall_result(SbiReturn::from(error_code));
                        }
                    }
                }
                VmCpuExit::Ecall(None) => {
                    // Unrecognized ECALL, return an error.
                    vcpu.set_ecall_result(SbiReturn::from(sbi::Error::NotSupported));
                }
                VmCpuExit::PageFault(addr) => {
                    if self.handle_guest_fault(addr).is_err() {
                        return exit;
                    }
                }
                VmCpuExit::Other(ref trap_csrs) => {
                    println!("Unhandled guest exit, SCAUSE = 0x{:08x}", trap_csrs.scause);
                    return exit;
                }
            }
        }
    }

    /// Handles ecalls from the guest.
    fn handle_ecall(&self, msg: SbiMessage) -> sbi::Result<Option<SbiReturn>> {
        match msg {
            SbiMessage::PutChar(c) => {
                // put char - legacy command
                print!("{}", c as u8 as char);
                Ok(None)
            }
            SbiMessage::Reset(r) => {
                match r {
                    ResetFunction::Reset {
                        reset_type: _,
                        reason: _,
                    } => {
                        // TODO do shutdown of VM or system if from primary host VM
                        println!("Vm shutdown/reboot request");
                        crate::poweroff();
                    }
                }
            }
            SbiMessage::Base(_) => Err(SbiError::NotSupported), // TODO
            SbiMessage::HartState(_) => Err(SbiError::NotSupported), // TODO
            SbiMessage::Tee(tee_func) => Ok(Some(self.handle_tee_msg(tee_func))),
            SbiMessage::Measurement(measurement_func) => {
                Ok(Some(self.handle_measurement_msg(measurement_func)))
            }
        }
    }

    fn handle_tee_msg(&self, tee_func: TeeFunction) -> SbiReturn {
        use TeeFunction::*;
        match tee_func {
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
            Run { guest_id } => self.guest_run(guest_id).into(),
            RemovePages {
                guest_id,
                gpa,
                remap_addr: _, // TODO - remove
                num_pages,
            } => self.guest_rm_pages(guest_id, gpa, num_pages).into(),
            GetGuestMeasurement {
                measurement_version,
                measurement_type,
                page_addr,
                guest_id,
            } => self
                .guest_get_measurement(measurement_version, measurement_type, page_addr, guest_id)
                .into(),
        }
    }

    fn handle_measurement_msg(&self, measurement_func: MeasurementFunction) -> SbiReturn {
        use MeasurementFunction::*;
        match measurement_func {
            GetSelfMeasurement {
                measurement_version,
                measurement_type,
                page_addr,
            } => self
                .guest_get_measurement(
                    measurement_version,
                    measurement_type,
                    page_addr,
                    GUEST_ID_SELF_MEASUREMENT,
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

        let (guest_builder, state_page) = self
            .vm_pages
            .create_guest_root_builder(from_page_addr)
            .map_err(|_| SbiError::InvalidParam)?;
        // unwrap can't fail because a valid guest must have a valid guest id.
        let id = guest_builder.page_owner_id();

        // create a boxpage for builder state and add it to the list of vms.
        let guest_state: PageBox<GuestState<T>> =
            PageBox::new_with(GuestState::Init(guest_builder), state_page);

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
                GuestState::Init(gbr) => GuestState::Running(Vm::new(gbr.create_pages())),
                t => t,
            };
            core::mem::swap(&mut **g, &mut running);
        })?;
        Ok(0)
    }

    fn guest_run(&self, guest_id: u64) -> sbi::Result<u64> {
        let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
        let guest_vm = guests.running_guest(guest_id)?;
        guest_vm.run(0); // TODO: Take vCPU ID.
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
        let grb = guests.initializing_guest(guest_id)?;
        self.vm_pages
            .add_pte_pages_builder(from_page_addr, num_pages, grb)
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
        let grb = guests.initializing_guest(guest_id)?;
        let to_page_addr = PageAddr::new(RawAddr::guest(to_addr, grb.page_owner_id()))
            .ok_or(SbiError::InvalidAddress)?;
        self.vm_pages
            .add_4k_pages_builder(
                from_page_addr,
                num_pages,
                grb,
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
        page_addr: u64,
        guest_id: u64,
    ) -> sbi::Result<u64> {
        let gpa = RawAddr::guest(page_addr, self.vm_pages.page_owner_id());
        if (measurement_version != 1) || (measurement_type != 1) || PageAddr::new(gpa).is_none() {
            return Err(SbiError::InvalidParam);
        }

        // The guest_id of 0 is a special identifier used to retrieve
        // measurements for self. Note that since we are borrowing
        // measurements from vm_pages, we can't take another mutable
        // reference to vm_pages to write to the GPA, so we have to
        // call a helper method to retrieve the measurements and write
        // them using the same mutable reference
        let result = if guest_id == GUEST_ID_SELF_MEASUREMENT {
            self.vm_pages.write_measurements_to_guest_owned_page(gpa)
        } else {
            // TODO: Define a compile-time constant for the maximum length of any measurement we
            // would conceivably use.
            let mut bytes = [0u8; SHA256_DIGEST_BYTES];
            let guests = self.guests.as_ref().ok_or(SbiError::InvalidParam)?.read();
            let _ = guests.get_guest_index(guest_id)?;
            if let Ok(running_guest) = guests.running_guest(guest_id) {
                running_guest
                    .vm_pages
                    .get_measurement(&mut bytes)
                    .map_err(|_| SbiError::Failed)?;
            } else {
                guests
                    .initializing_guest(guest_id)
                    .unwrap()
                    .get_measurement(&mut bytes)
                    .map_err(|_| SbiError::Failed)?;
            }

            self.vm_pages.write_to_guest_owned_page(gpa, &bytes)
        };

        result
            .map(|bytes| bytes as u64)
            .map_err(|_| SbiError::InvalidAddress)
    }
}

/// Represents the special VM that serves as the host for the system.
pub struct Host<T: GuestStagePageTable> {
    inner: Vm<T>,
}

impl<T: GuestStagePageTable> Host<T> {
    /// Creates a new `Host` using the given initial page table root.
    /// `guests`: A vec for storing guest info if "nested" guests will be created. Must have
    /// length zero and capacity limits the number of nested guests.
    pub fn new(page_root: HostRootPages<T>, pages: SequentialPages) -> Self {
        let mut inner = Vm::new(page_root.into_inner());
        inner.add_guest_tracking_pages(pages);
        inner.set_interrupt_file(ImsicGuestId::HostVm);
        Self { inner }
    }

    /// Sets the launch arguments (entry point and FDT) for the host vCPU.
    pub fn set_launch_args(&self, entry_addr: GuestPhysAddr, fdt_addr: GuestPhysAddr) {
        self.inner.set_launch_args(entry_addr, fdt_addr.bits());
    }

    /// Run the host. Only returns for system shutdown
    pub fn run(&mut self, vcpu_id: u64) -> VmCpuExit {
        // TODO - return value need to be host specific
        self.inner.run(vcpu_id)
    }
}
