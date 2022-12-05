// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use drivers::{imsic::*, CpuId};

use crate::smp::PerCpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BindingImsic(ImsicError),
    RebindingImsic(ImsicError),
    UnbindingImsic(ImsicError),
    WrongBindStatus,
    WrongPhysicalCpu,
    InvalidInterruptId(usize),
    DeniedInterruptId(usize),
}

pub type Result<T> = core::result::Result<T, Error>;

// State of binding a vCPU to a physical CPU.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BindStatus {
    Binding(CpuId, ImsicFileId),
    Bound(CpuId, ImsicFileId),
    Rebinding(CpuId, ImsicFileId, CpuId, ImsicFileId),
    Cloned(CpuId, ImsicFileId),
    Unbinding(CpuId, ImsicFileId),
    Unbound,
}

const ALLOW_LIST_ENTRIES: usize = MAX_INTERRUPT_IDS / 64;

// Bitmap tracking the per-vCPU allowed external interrupts.
struct AllowList {
    bits: ArrayVec<u64, ALLOW_LIST_ENTRIES>,
    num_ids: usize,
}

impl AllowList {
    fn new(num_ids: usize) -> Self {
        let mut bits = ArrayVec::new();
        let entries = (num_ids + 63) / 64;
        for _ in 0..entries {
            bits.push(0);
        }

        Self { bits, num_ids }
    }

    fn allow_id(&mut self, id: usize) -> Result<()> {
        if id == 0 || id >= self.num_ids {
            return Err(Error::InvalidInterruptId(id));
        }
        self.bits[id / 64] |= 1 << (id % 64);
        Ok(())
    }

    fn allow_all(&mut self) {
        self.bits.iter_mut().for_each(|i| *i = !0u64);
    }

    fn deny_id(&mut self, id: usize) -> Result<()> {
        if id == 0 || id >= self.num_ids {
            return Err(Error::InvalidInterruptId(id));
        }
        self.bits[id / 64] &= !(1 << (id % 64));
        Ok(())
    }

    fn deny_all(&mut self) {
        self.bits.iter_mut().for_each(|i| *i = 0u64);
    }

    fn is_allowed(&self, id: usize) -> bool {
        id < self.num_ids && id != 0 && (self.bits[id / 64] & (1 << (id % 64))) != 0
    }
}

/// Virtual external interrupt state for a vCPU.
pub struct VmCpuExtInterrupts {
    bind_status: BindStatus,
    // This is the virtual location where we want the guest VCPU to see the interrupt file.
    // Physical location information is part of the bind_status in form of CpuId and ImsicFileId.
    imsic_location: ImsicLocation,
    sw_file: SwFile,
    allowed_ids: AllowList,
}

impl VmCpuExtInterrupts {
    /// Creates a new `VmCpuExtInterrupts` to track the external interrupt state of a vCPU with
    /// `imsic_location` as the location of the vCPU's virtualized IMSIC.
    pub fn new(imsic_location: ImsicLocation) -> Self {
        Self {
            bind_status: BindStatus::Unbound,
            imsic_location,
            sw_file: SwFile::new(),
            allowed_ids: AllowList::new(Imsic::get().interrupt_ids()),
        }
    }

    /// Returns the location of this vCPU's virtualized IMSIC in guest physical address space.
    pub fn imsic_location(&self) -> ImsicLocation {
        self.imsic_location
    }

    /// Prepares to bind this vCPU to `interrupt_file` on the current physical CPU.
    pub fn bind_imsic_prepare(&mut self, interrupt_file: ImsicFileId) -> Result<()> {
        // The vCPU must be completely unbound to start a bind operation.
        if self.bind_status != BindStatus::Unbound {
            return Err(Error::WrongBindStatus);
        }

        // Initialize the guest IMSIC file we're getting bound to.
        Imsic::get()
            .clear_guest_file(interrupt_file)
            .map_err(Error::BindingImsic)?;
        self.bind_status = BindStatus::Binding(PerCpu::this_cpu().cpu_id(), interrupt_file);
        Ok(())
    }

    /// Completes the IMSIC bind operation started in `bind_imsic_prepare()`.
    pub fn bind_imsic_finish(&mut self) -> Result<ImsicFileId> {
        // Make sure there's a bind operation was started on this CPU.
        let (cpu, interrupt_file) = match self.bind_status {
            BindStatus::Binding(cpu, interrupt_file) => (cpu, interrupt_file),
            _ => {
                return Err(Error::WrongBindStatus);
            }
        };
        if cpu != PerCpu::this_cpu().cpu_id() {
            return Err(Error::WrongPhysicalCpu);
        }

        // Finish restoring state from the SW interrupt file.
        let imsic = Imsic::get();
        imsic
            .restore_guest_file(interrupt_file, &mut self.sw_file)
            .map_err(Error::BindingImsic)?;
        self.bind_status = BindStatus::Bound(cpu, interrupt_file);
        Ok(interrupt_file)
    }

    /// Prepares to rebind this vCPU to `interrupt_file` on the current physical CPU.
    pub fn rebind_imsic_prepare(&mut self, interrupt_file: ImsicFileId) -> Result<()> {
        // The vCPU must be bound to an other IMSIC file.
        let (cpu, old_interrupt_file) = match self.bind_status {
            BindStatus::Bound(cpu, old_interrupt_file) => (cpu, old_interrupt_file),
            _ => return Err(Error::WrongBindStatus),
        };

        // Initialize the guest IMSIC file we're getting bound to.
        Imsic::get()
            .clear_guest_file(interrupt_file)
            .map_err(Error::BindingImsic)?;
        self.bind_status = BindStatus::Rebinding(
            cpu,
            old_interrupt_file,
            PerCpu::this_cpu().cpu_id(),
            interrupt_file,
        );
        Ok(())
    }

    /// Copies the guest interrupt file over to sw file.
    pub fn rebind_imsic_clone(&mut self) -> Result<()> {
        // Make sure there's a rebind operation in progress and we are on old CPU right now.
        let (cpu, interrupt_file, new_cpu, new_file) = match self.bind_status {
            BindStatus::Rebinding(cpu, interrupt_file, new_cpu, new_file) => {
                (cpu, interrupt_file, new_cpu, new_file)
            }
            _ => return Err(Error::WrongBindStatus),
        };

        if cpu != PerCpu::this_cpu().cpu_id() {
            return Err(Error::WrongPhysicalCpu);
        }

        // Copy and then clear the guest interrupt file.
        let imsic = Imsic::get();
        imsic
            .save_guest_file_prepare(interrupt_file, &mut self.sw_file)
            .map_err(Error::BindingImsic)?;
        imsic
            .save_guest_file_finish(interrupt_file, &mut self.sw_file)
            .map_err(Error::RebindingImsic)?;
        self.bind_status = BindStatus::Cloned(new_cpu, new_file);
        Ok(())
    }

    /// Get the previous guest interrupt file's location.
    pub fn prev_imsic_location(&self) -> Result<ImsicLocation> {
        // Make sure we are in the rebinding state.
        let (cpu, interrupt_file) = match self.bind_status {
            BindStatus::Rebinding(cpu, interrupt_file, _, _) => (cpu, interrupt_file),
            _ => {
                return Err(Error::WrongBindStatus);
            }
        };

        let prev_loc = Imsic::get()
            .phys_file_location(cpu, interrupt_file)
            .map_err(Error::RebindingImsic)?;
        Ok(prev_loc)
    }

    /// Completes the IMSIC rebind operation started in `rebind_imsic_prepare()`.
    pub fn rebind_imsic_finish(&mut self) -> Result<ImsicFileId> {
        // Make sure there's a rebind operation was started on this CPU.
        let (cpu, interrupt_file) = match self.bind_status {
            BindStatus::Cloned(cpu, interrupt_file) => (cpu, interrupt_file),
            _ => {
                return Err(Error::WrongBindStatus);
            }
        };

        if cpu != PerCpu::this_cpu().cpu_id() {
            return Err(Error::WrongPhysicalCpu);
        }

        // Finish restoring state from the SW interrupt file.
        Imsic::get()
            .restore_guest_file(interrupt_file, &mut self.sw_file)
            .map_err(Error::RebindingImsic)?;
        self.bind_status = BindStatus::Bound(cpu, interrupt_file);

        Ok(interrupt_file)
    }

    /// Prepares to unbind this vCPU from its current interrupt file.
    pub fn unbind_imsic_prepare(&mut self) -> Result<()> {
        // We must be bound (to the current CPU) to start an unbind.
        let (cpu, interrupt_file) = match self.bind_status {
            BindStatus::Bound(cpu, interrupt_file) => (cpu, interrupt_file),
            _ => {
                return Err(Error::WrongBindStatus);
            }
        };
        if cpu != PerCpu::this_cpu().cpu_id() {
            return Err(Error::WrongPhysicalCpu);
        }

        let imsic = Imsic::get();
        imsic
            .save_guest_file_prepare(interrupt_file, &mut self.sw_file)
            .map_err(Error::UnbindingImsic)?;
        self.bind_status = BindStatus::Unbinding(cpu, interrupt_file);
        Ok(())
    }

    /// Completes the IMSIC unbind operation started in `unbind_imsic_prepare()`.
    pub fn unbind_imsic_finish(&mut self) -> Result<()> {
        // We must be bound (to the current CPU) to start an unbind.
        let (cpu, interrupt_file) = match self.bind_status {
            BindStatus::Unbinding(cpu, interrupt_file) => (cpu, interrupt_file),
            _ => {
                return Err(Error::WrongBindStatus);
            }
        };
        if cpu != PerCpu::this_cpu().cpu_id() {
            return Err(Error::WrongPhysicalCpu);
        }

        let imsic = Imsic::get();
        imsic
            .save_guest_file_finish(interrupt_file, &mut self.sw_file)
            .map_err(Error::UnbindingImsic)?;
        self.bind_status = BindStatus::Unbound;
        Ok(())
    }

    /// Returns true if this vCPU is bound to the current physical CPU.
    pub fn is_bound_on_this_cpu(&self) -> bool {
        match self.bind_status {
            BindStatus::Bound(cpu_id, _) => cpu_id == PerCpu::this_cpu().cpu_id(),
            _ => false,
        }
    }

    /// Adds `id` to the list of injectable external interrupts.
    pub fn allow_interrupt(&mut self, id: usize) -> Result<()> {
        self.allowed_ids.allow_id(id)
    }

    /// Allows injection of all external interrupts.
    pub fn allow_all_interrupts(&mut self) {
        self.allowed_ids.allow_all();
    }

    /// Removes `id` from the list of injectable external interrupts.
    pub fn deny_interrupt(&mut self, id: usize) -> Result<()> {
        self.allowed_ids.deny_id(id)
    }

    /// Disables injection of all external interrupts.
    pub fn deny_all_interrupts(&mut self) {
        self.allowed_ids.deny_all();
    }

    /// Injects the specified external interrupt ID into this vCPU, if allowed.
    pub fn inject_interrupt(&mut self, id: usize) -> Result<()> {
        if !self.allowed_ids.is_allowed(id) {
            return Err(Error::DeniedInterruptId(id));
        }
        match self.bind_status {
            BindStatus::Bound(cpu_id, file) => {
                // Unwrap ok: CPU ID and file must be valid if a vCPU is bound to it.
                Imsic::get().send_ipi_raw(cpu_id, file, id as u32).unwrap();
            }
            _ => {
                // For everything else, just update the SW file.
                self.sw_file.set_eip_bit(id);
            }
        }
        Ok(())
    }
}
