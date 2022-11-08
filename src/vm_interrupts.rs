// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use drivers::{imsic::*, CpuId};

use crate::smp::PerCpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BindingImsic(ImsicError),
    UnbindingImsic(ImsicError),
    WrongBindStatus,
    WrongPhysicalCpu,
}

pub type Result<T> = core::result::Result<T, Error>;

// State of binding a vCPU to a physical CPU.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BindStatus {
    Binding(CpuId, ImsicFileId),
    Bound(CpuId, ImsicFileId),
    Unbinding(CpuId, ImsicFileId),
    Unbound,
}

/// Virtual external interrupt state for a vCPU.
pub struct VmCpuExtInterrupts {
    bind_status: BindStatus,
    imsic_location: ImsicLocation,
    sw_file: SwFile,
}

impl VmCpuExtInterrupts {
    /// Creates a new `VmCpuExtInterrupts` to track the external interrupt state of a vCPU with
    /// `imsic_location` as the location of the vCPU's virtualized IMSIC.
    pub fn new(imsic_location: ImsicLocation) -> Self {
        Self {
            bind_status: BindStatus::Unbound,
            imsic_location,
            sw_file: SwFile::new(),
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
        let imsic = Imsic::get();
        imsic
            .restore_guest_file_prepare(interrupt_file, &mut self.sw_file)
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
            .restore_guest_file_finish(interrupt_file, &mut self.sw_file)
            .map_err(Error::BindingImsic)?;
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
}
