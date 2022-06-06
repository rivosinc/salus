// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Rust SBI message parsing.
//! `SbiMessage` is an enum of all the SBI extensions.
//! For each extension, a function enum is defined to contain the SBI function data.
#![no_std]

use riscv_regs::{GeneralPurposeRegisters, GprIndex};

pub const EXT_PUT_CHAR: u64 = 0x01;
pub const EXT_BASE: u64 = 0x10;
pub const EXT_HART_STATE: u64 = 0x48534D;
pub const EXT_RESET: u64 = 0x53525354;
pub const EXT_TEE: u64 = 0x544545;
pub const EXT_MEASUREMENT: u64 = 0x5464545;

/// Error constants from the sbi [spec](https://github.com/riscv-non-isa/riscv-sbi-doc/releases)
pub const SBI_SUCCESS: i64 = 0;
pub const SBI_ERR_FAILED: i64 = -1;
pub const SBI_ERR_NOT_SUPPORTED: i64 = -2;
pub const SBI_ERR_INVALID_PARAM: i64 = -3;
pub const SBI_ERR_DENIED: i64 = -4;
pub const SBI_ERR_INVALID_ADDRESS: i64 = -5;
pub const SBI_ERR_ALREADY_AVAILABLE: i64 = -6;
pub const SBI_ERR_ALREADY_STARTED: i64 = -7;
pub const SBI_ERR_ALREADY_STOPPED: i64 = -8;

/// Errors passed over the SBI protocol
#[derive(Debug)]
pub enum Error {
    InvalidAddress,
    InvalidParam,
    Failed,
    NotSupported,
    UnknownSbiExtension,
}

impl Error {
    /// Parse the given error code to an `Error` enum.
    pub fn from_code(e: i64) -> Self {
        use Error::*;
        match e {
            SBI_ERR_INVALID_ADDRESS => InvalidAddress,
            SBI_ERR_INVALID_PARAM => InvalidParam,
            SBI_ERR_NOT_SUPPORTED => NotSupported,
            _ => Failed,
        }
    }

    /// Convert `Self` to a 64bit error code to be returned over SBI.
    pub fn to_code(&self) -> i64 {
        use Error::*;
        match self {
            InvalidAddress => SBI_ERR_INVALID_ADDRESS,
            InvalidParam => SBI_ERR_INVALID_PARAM,
            Failed => SBI_ERR_FAILED,
            NotSupported => SBI_ERR_NOT_SUPPORTED,
            UnknownSbiExtension => SBI_ERR_INVALID_PARAM,
        }
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// Functions defined for the Base extension
#[derive(Clone, Copy)]
pub enum BaseFunction {
    GetSpecificationVersion,
    GetImplementationID,
    GetImplementationVersion,
    ProbeSbiExtension(u64),
    GetMachineVendorID,
    GetMachineArchitectureID,
    GetMachineImplementationID,
}

impl BaseFunction {
    pub fn from_regs(args: &[u64]) -> Result<Self> {
        use BaseFunction::*;

        match args[6] {
            0 => Ok(GetSpecificationVersion),
            1 => Ok(GetImplementationID),
            2 => Ok(GetImplementationVersion),
            3 => Ok(ProbeSbiExtension(args[0])),
            4 => Ok(GetMachineVendorID),
            5 => Ok(GetMachineArchitectureID),
            6 => Ok(GetMachineImplementationID),
            _ => Err(Error::InvalidParam),
        }
    }

    pub fn a6(&self) -> u64 {
        use BaseFunction::*;
        match self {
            GetSpecificationVersion => 0,
            GetImplementationID => 1,
            GetImplementationVersion => 2,
            ProbeSbiExtension(_) => 3,
            GetMachineVendorID => 4,
            GetMachineArchitectureID => 5,
            GetMachineImplementationID => 6,
        }
    }

    pub fn a0(&self) -> u64 {
        use BaseFunction::*;
        match self {
            ProbeSbiExtension(ext) => *ext,
            _ => 0,
        }
    }
}

/// Functions defined for the State extension
#[derive(Clone, Copy)]
pub enum StateFunction {
    HartStart {
        hart_id: u64,
        start_addr: u64,
        opaque: u64,
    },
    HartStop,
    HartStatus {
        hart_id: u64,
    },
    HartSuspend {
        suspend_type: u32,
        resume_addr: u64,
        opaque: u64,
    },
}

/// Return value for the HartStatus SBI call.
#[repr(u64)]
pub enum HartState {
    Started = 0,
    Stopped = 1,
    StartPending = 2,
    StopPending = 3,
    Suspended = 4,
    SuspendPending = 5,
}

impl StateFunction {
    pub fn from_regs(args: &[u64]) -> Result<Self> {
        use StateFunction::*;
        match args[6] {
            0 => Ok(HartStart {
                hart_id: args[0],
                start_addr: args[1],
                opaque: args[2],
            }),
            1 => Ok(HartStop),
            2 => Ok(HartStatus { hart_id: args[0] }),
            3 => Ok(HartSuspend {
                suspend_type: args[0] as u32,
                resume_addr: args[1],
                opaque: args[2],
            }),
            _ => Err(Error::InvalidParam),
        }
    }

    pub fn a6(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart { .. } => 0,
            HartStop => 1,
            HartStatus { .. } => 2,
            HartSuspend { .. } => 3,
        }
    }

    pub fn a0(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart {
                hart_id,
                start_addr: _,
                opaque: _,
            } => *hart_id,
            HartStatus { hart_id } => *hart_id,
            HartSuspend {
                suspend_type,
                resume_addr: _,
                opaque: _,
            } => *suspend_type as u64,
            _ => 0,
        }
    }

    pub fn a1(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart {
                hart_id: _,
                start_addr,
                opaque: _,
            } => *start_addr,
            HartSuspend {
                suspend_type: _,
                resume_addr,
                opaque: _,
            } => *resume_addr,
            _ => 0,
        }
    }

    pub fn a2(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart {
                hart_id: _,
                start_addr: _,
                opaque,
            } => *opaque,
            HartSuspend {
                suspend_type: _,
                resume_addr: _,
                opaque,
            } => *opaque,
            _ => 0,
        }
    }
}

/// Funcions for the Reset extension
#[derive(Copy, Clone)]
pub enum ResetFunction {
    Reset {
        reset_type: ResetType,
        reason: ResetReason,
    },
}

#[derive(Copy, Clone)]
pub enum ResetType {
    Shutdown,
    ColdReset,
    WarmReset,
}

impl ResetType {
    fn from_reg(a0: u64) -> Result<Self> {
        use ResetType::*;
        Ok(match a0 {
            0 => Shutdown,
            1 => ColdReset,
            2 => WarmReset,
            _ => return Err(Error::InvalidParam),
        })
    }
}

#[derive(Copy, Clone)]
pub enum ResetReason {
    NoReason,
    SystemFailure,
}

impl ResetReason {
    fn from_reg(a1: u64) -> Result<Self> {
        use ResetReason::*;
        Ok(match a1 {
            0 => NoReason,
            2 => SystemFailure,
            _ => return Err(Error::InvalidParam),
        })
    }
}

impl ResetFunction {
    pub fn shutdown() -> Self {
        ResetFunction::Reset {
            reset_type: ResetType::Shutdown,
            reason: ResetReason::NoReason,
        }
    }

    fn from_regs(a6: u64, a0: u64, a1: u64) -> Result<Self> {
        use ResetFunction::*;

        Ok(match a6 {
            0 => Reset {
                reset_type: ResetType::from_reg(a0)?,
                reason: ResetReason::from_reg(a1)?,
            },
            _ => return Err(Error::InvalidParam),
        })
    }

    fn get_a0(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type: _,
                reason,
            } => *reason as u64,
        }
    }

    fn get_a1(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type,
                reason: _,
            } => *reset_type as u64,
        }
    }
}

#[repr(u64)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TvmCpuRegister {
    Pc = 0,
    A1 = 1,
}

impl TvmCpuRegister {
    pub fn from_reg(a2: u64) -> Result<Self> {
        match a2 {
            0 => Ok(TvmCpuRegister::Pc),
            1 => Ok(TvmCpuRegister::A1),
            _ => Err(Error::InvalidParam),
        }
    }
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub enum TsmState {
    /// TSM has not been loaded on this platform.
    #[default]
    TsmNotLoaded = 0,
    /// TSM has been loaded, but has not yet been initialized.
    TsmLoaded = 1,
    /// TSM has been loaded & initialized, and is ready to accept TEECALLs.
    TsmReady = 2,
}

#[repr(C)]
#[derive(Default)]
pub struct TsmInfo {
    /// The current state of the TSM. If the state is not `TsmReady`, the remaining fields are
    /// invalid and will be initialized to 0.
    pub tsm_state: TsmState,
    /// Version number of the running TSM.
    pub tsm_version: u32,
    /// The number of 4kB pages which must be donated to the TSM for storing TVM state in the
    /// `TvmCreate` TEECALL.
    pub tvm_state_pages: u64,
    /// The maximum number of vCPUs a TVM can support.
    pub tvm_max_vcpus: u64,
    /// The number of bytes per vCPU which must be donated to the TSM when creating a new TVM.
    pub tvm_bytes_per_vcpu: u64,
}

#[repr(C)]
pub struct TvmCreateParams {
    /// The base physical address of the 16kB region that should be used for the TVM's page
    /// directory. Must be 16kB-aligned.
    pub tvm_page_directory_addr: u64,
    /// The base physical address of the region to be used to hold the TVM's global state. Must
    /// be page-aligned and `TsmInfo::tvm_state_pages` pages in length.
    pub tvm_state_addr: u64,
    /// The maximum number of vCPUs that will be created for this TVM. Must be less than or equal
    /// to `TsmInfo::tvm_max_vcpus`.
    pub tvm_num_vcpus: u64,
    /// The base physical address of the region to be used to hold the TVM's vCPU state. Must be
    /// page-aligned and `TsmInfo::tvm_bytes_per_vcpu` * `tvm_num_vcpus` bytes in length, rounded
    /// up to the nearest multiple of 4kB.
    pub tvm_vcpu_addr: u64,
}

#[derive(Copy, Clone)]
pub enum TeeFunction {
    /// Creates a TVM from the parameters in the `TvmCreateParams` structure at physical address
    /// `params_addr`. Returns a guest ID that can be used to refer to the TVM in TVM management
    /// TEECALLs.
    ///
    /// a6 = 0
    /// a0 = base physical address of the `TvmCreateParams` structure
    /// a1 = length of the `TvmCreateParams` structure in bytes
    TvmCreate { params_addr: u64, len: u64 },
    /// Message to destroy a TVM created with `TvmCreate`.
    /// a6 = 1, a0 = guest id returned from `TvmCreate`.
    TvmDestroy { guest_id: u64 },
    /// Message from the host to add page tables pages to a TVM it created with `TvmCreate`. Pages
    /// must be added to the page table before mappings for more memory can be made. These must be
    /// 4k Pages.
    /// a6 = 2, a0 = guest_id, a1 = address of the first page, and a2 = number of pages
    AddPageTablePages {
        guest_id: u64,
        page_addr: u64,
        num_pages: u64,
    },
    /// Message from the host to add page(s) to a TVM it created with `TvmCreate`.
    /// a6 = 3,
    /// a0 = guest_id,
    /// a1 = address of the first page,
    /// a2 = page_type: 0 => 4k, 1=> 2M, 2=> 1G, 3=512G, Others: reserved
    /// a3 = number of pages
    /// a4 = Guest Address
    /// a4 = if non-zero don't zero pages before passing to the guest(only allowed before starting
    /// the guest, pages will be added to measurement of the guest.)
    AddPages {
        guest_id: u64,
        page_addr: u64,
        page_type: u64,
        num_pages: u64,
        gpa: u64,
        measure_preserve: bool,
    },
    /// Moves a VM from the initializing state to the Runnable state
    /// a6 = 4
    /// a0 = guest id
    Finalize { guest_id: u64 },
    /// Runs the given vCPU in the TVM
    /// a6 = 5
    /// a0 = guest id
    /// a1 = vCPU id
    TvmCpuRun { guest_id: u64, vcpu_id: u64 },
    /// Removes pages that were previously added with `AddPages`.
    /// a6 = 6
    /// a0 = guest id,
    /// a1 = guest address to unmap
    /// a2 = address to remap the pages to in the requestor
    /// a3 = number of pages
    RemovePages {
        guest_id: u64,
        gpa: u64,
        remap_addr: u64, // TODO should we track this locally?
        num_pages: u64,
    },
    /// Copies the measurements for the specified guest to the physical address `dest_addr`.
    /// The measurement version and type must be set to 1 for now.
    /// a6 = 7
    /// a0 = measurement version
    /// a1 = measurement type
    /// a2 = dest_addr
    /// a3 = guest id
    GetGuestMeasurement {
        measurement_version: u64,
        measurement_type: u64,
        dest_addr: u64,
        guest_id: u64,
    },
    /// Adds a vCPU with ID `vcpu_id` to the guest `guest_id`. vCPUs may not be added after the TVM
    /// is finalized.
    ///
    /// a6 = 8
    /// a0 = guest id
    /// a1 = vCPU id
    TvmCpuCreate { guest_id: u64, vcpu_id: u64 },
    /// Sets the register identified by `register` to `value` in the vCPU with ID `vcpu_id`. vCPU
    /// register state may not be modified after the TVM is finalized.
    ///
    /// a6 = 9
    /// a0 = guest id
    /// a1 = vCPU id
    /// a2 = register id
    /// a3 = register value
    TvmCpuSetRegister {
        guest_id: u64,
        vcpu_id: u64,
        register: TvmCpuRegister,
        value: u64,
    },
    /// Writes up to `len` bytes of the `TsmInfo` structure to the physical address `dest_addr`.
    /// Returns the number of bytes written.
    ///
    /// a6 = 10
    /// a0 = destination address of the `TsmInfo` structure
    /// a1 = maximum number of bytes to be written
    TsmGetInfo { dest_addr: u64, len: u64 },
}

impl TeeFunction {
    // Takes registers a0-6 as the input.
    pub fn from_regs(args: &[u64]) -> Result<Self> {
        use TeeFunction::*;
        match args[6] {
            0 => Ok(TvmCreate {
                params_addr: args[0],
                len: args[1],
            }),
            1 => Ok(TvmDestroy { guest_id: args[0] }),
            2 => Ok(AddPageTablePages {
                guest_id: args[0],
                page_addr: args[1],
                num_pages: args[2],
            }),
            3 => Ok(AddPages {
                guest_id: args[0],
                page_addr: args[1],
                page_type: args[2],
                num_pages: args[3],
                gpa: args[4],
                measure_preserve: args[5] == 0,
            }),
            4 => Ok(Finalize { guest_id: args[0] }),
            5 => Ok(TvmCpuRun {
                guest_id: args[0],
                vcpu_id: args[1],
            }),
            6 => Ok(RemovePages {
                guest_id: args[0],
                gpa: args[1],
                remap_addr: args[2],
                num_pages: args[3],
            }),
            7 => Ok(GetGuestMeasurement {
                measurement_version: args[0],
                measurement_type: args[1],
                dest_addr: args[2],
                guest_id: args[3],
            }),
            8 => Ok(TvmCpuCreate {
                guest_id: args[0],
                vcpu_id: args[1],
            }),
            9 => Ok(TvmCpuSetRegister {
                guest_id: args[0],
                vcpu_id: args[1],
                register: TvmCpuRegister::from_reg(args[2])?,
                value: args[3],
            }),
            10 => Ok(TsmGetInfo {
                dest_addr: args[0],
                len: args[1],
            }),
            _ => Err(Error::InvalidParam),
        }
    }

    pub fn a6(&self) -> u64 {
        use TeeFunction::*;
        match self {
            TvmCreate {
                params_addr: _,
                len: _,
            } => 0,
            TvmDestroy { guest_id: _ } => 1,
            AddPageTablePages {
                guest_id: _,
                page_addr: _,
                num_pages: _,
            } => 2,
            AddPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                gpa: _,
                measure_preserve: _,
            } => 3,
            Finalize { guest_id: _ } => 4,
            TvmCpuRun {
                guest_id: _,
                vcpu_id: _,
            } => 5,
            RemovePages {
                guest_id: _,
                gpa: _,
                remap_addr: _,
                num_pages: _,
            } => 6,
            GetGuestMeasurement {
                measurement_type: _,
                measurement_version: _,
                dest_addr: _,
                guest_id: _,
            } => 7,
            TvmCpuCreate {
                guest_id: _,
                vcpu_id: _,
            } => 8,
            TvmCpuSetRegister {
                guest_id: _,
                vcpu_id: _,
                register: _,
                value: _,
            } => 9,
            TsmGetInfo {
                dest_addr: _,
                len: _,
            } => 10,
        }
    }

    pub fn a0(&self) -> u64 {
        use TeeFunction::*;
        match self {
            TvmCreate {
                params_addr,
                len: _,
            } => *params_addr,
            TvmDestroy { guest_id } => *guest_id,
            AddPageTablePages {
                guest_id,
                page_addr: _,
                num_pages: _,
            } => *guest_id,
            AddPages {
                guest_id,
                page_addr: _,
                page_type: _,
                num_pages: _,
                gpa: _,
                measure_preserve: _,
            } => *guest_id,
            Finalize { guest_id } => *guest_id,
            TvmCpuRun {
                guest_id,
                vcpu_id: _,
            } => *guest_id,
            RemovePages {
                guest_id,
                gpa: _,
                remap_addr: _,
                num_pages: _,
            } => *guest_id,
            GetGuestMeasurement {
                measurement_version,
                measurement_type: _,
                dest_addr: _,
                guest_id: _,
            } => *measurement_version,
            TvmCpuCreate {
                guest_id,
                vcpu_id: _,
            } => *guest_id,
            TvmCpuSetRegister {
                guest_id,
                vcpu_id: _,
                register: _,
                value: _,
            } => *guest_id,
            TsmGetInfo { dest_addr, len: _ } => *dest_addr,
        }
    }

    pub fn a1(&self) -> u64 {
        use TeeFunction::*;
        match self {
            TvmCreate {
                params_addr: _,
                len,
            } => *len,
            AddPageTablePages {
                guest_id: _,
                page_addr,
                num_pages: _,
            } => *page_addr,
            AddPages {
                guest_id: _,
                page_addr,
                page_type: _,
                num_pages: _,
                gpa: _,
                measure_preserve: _,
            } => *page_addr,
            TvmCpuRun {
                guest_id: _,
                vcpu_id,
            } => *vcpu_id,
            RemovePages {
                guest_id: _,
                gpa,
                remap_addr: _,
                num_pages: _,
            } => *gpa,
            GetGuestMeasurement {
                measurement_version: _,
                measurement_type,
                dest_addr: _,
                guest_id: _,
            } => *measurement_type,
            TvmCpuCreate {
                guest_id: _,
                vcpu_id,
            } => *vcpu_id,
            TvmCpuSetRegister {
                guest_id: _,
                vcpu_id,
                register: _,
                value: _,
            } => *vcpu_id,
            TsmGetInfo { dest_addr: _, len } => *len,
            _ => 0,
        }
    }

    pub fn a2(&self) -> u64 {
        use TeeFunction::*;
        match self {
            AddPageTablePages {
                guest_id: _,
                page_addr: _,
                num_pages,
            } => *num_pages,
            AddPages {
                guest_id: _,
                page_addr: _,
                page_type,
                num_pages: _,
                gpa: _,
                measure_preserve: _,
            } => *page_type,
            RemovePages {
                guest_id: _,
                gpa: _,
                remap_addr,
                num_pages: _,
            } => *remap_addr,
            GetGuestMeasurement {
                measurement_version: _,
                measurement_type: _,
                dest_addr,
                guest_id: _,
            } => *dest_addr,
            TvmCpuSetRegister {
                guest_id: _,
                vcpu_id: _,
                register,
                value: _,
            } => *register as u64,
            _ => 0,
        }
    }

    pub fn a3(&self) -> u64 {
        use TeeFunction::*;
        match self {
            AddPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages,
                gpa: _,
                measure_preserve: _,
            } => *num_pages,
            RemovePages {
                guest_id: _,
                gpa: _,
                remap_addr: _,
                num_pages,
            } => *num_pages,
            GetGuestMeasurement {
                measurement_version: _,
                measurement_type: _,
                dest_addr: _,
                guest_id,
            } => *guest_id,
            TvmCpuSetRegister {
                guest_id: _,
                vcpu_id: _,
                register: _,
                value,
            } => *value,
            _ => 0,
        }
    }

    pub fn a4(&self) -> u64 {
        use TeeFunction::*;
        match self {
            AddPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                gpa,
                measure_preserve: _,
            } => *gpa,
            _ => 0,
        }
    }

    pub fn a5(&self) -> u64 {
        use TeeFunction::*;
        match self {
            AddPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                gpa: _,
                measure_preserve,
            } => {
                if *measure_preserve {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    pub fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        // TODO - Does it need function-specific returns?
        match a0 {
            0 => Ok(a1),
            e => Err(Error::from_code(e as i64)),
        }
    }
}

#[derive(Copy, Clone)]
pub enum MeasurementFunction {
    /// Copies the measurements for the current VM to the (guest) physical address in `dest_addr`.
    /// The measurement version and type must be set to 1 for now.
    /// a6 = 0
    /// a0 = measurement version
    /// a1 = measurement type
    /// a2 = dest_addr
    GetSelfMeasurement {
        measurement_version: u64,
        measurement_type: u64,
        dest_addr: u64,
    },
}

impl MeasurementFunction {
    // Takes registers a0-6 as the input.
    pub fn from_regs(args: &[u64]) -> Result<Self> {
        use MeasurementFunction::*;
        match args[6] {
            0 => Ok(GetSelfMeasurement {
                measurement_version: args[0],
                measurement_type: args[1],
                dest_addr: args[2],
            }),
            _ => Err(Error::InvalidParam),
        }
    }

    pub fn a6(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version: _,
                measurement_type: _,
                dest_addr: _,
            } => 0,
        }
    }

    pub fn a0(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version,
                measurement_type: _,
                dest_addr: _,
            } => *measurement_version,
        }
    }

    pub fn a1(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version: _,
                measurement_type,
                dest_addr: _,
            } => *measurement_type,
        }
    }

    pub fn a2(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version: _,
                measurement_type: _,
                dest_addr,
            } => *dest_addr,
        }
    }

    pub fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        match a0 {
            0 => Ok(a1),
            e => Err(Error::from_code(e as i64)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SbiReturn {
    pub error_code: i64,
    pub return_value: u64,
}

impl SbiReturn {
    pub fn success(return_value: u64) -> Self {
        Self {
            error_code: SBI_SUCCESS,
            return_value,
        }
    }
}

impl From<Result<u64>> for SbiReturn {
    fn from(result: Result<u64>) -> SbiReturn {
        match result {
            Ok(rv) => Self::success(rv),
            Err(e) => Self::from(e),
        }
    }
}

impl From<Error> for SbiReturn {
    fn from(error: Error) -> SbiReturn {
        SbiReturn {
            error_code: error.to_code(),
            return_value: 0,
        }
    }
}

/// SBI Message used to invoke the specified SBI extension in the firmware.
#[derive(Clone, Copy)]
pub enum SbiMessage {
    Base(BaseFunction),
    PutChar(u64),
    HartState(StateFunction),
    Reset(ResetFunction),
    Tee(TeeFunction),
    Measurement(MeasurementFunction),
}

impl SbiMessage {
    /// Creates an SbiMessage struct from the given GPRs. Intended for use from the ECALL handler
    /// and passed the saved register state from the calling OS. A7 must contain a valid SBI
    /// extension and the other A* registers will be interpreted based on the extension A7 selects.
    pub fn from_regs(gprs: &GeneralPurposeRegisters) -> Result<Self> {
        use GprIndex::*;
        match gprs.reg(A7) {
            EXT_PUT_CHAR => Ok(SbiMessage::PutChar(gprs.reg(A0))),
            EXT_BASE => BaseFunction::from_regs(gprs.a_regs()).map(SbiMessage::Base),
            EXT_HART_STATE => StateFunction::from_regs(gprs.a_regs()).map(SbiMessage::HartState),
            EXT_RESET => ResetFunction::from_regs(gprs.reg(A6), gprs.reg(A0), gprs.reg(A1))
                .map(SbiMessage::Reset),
            EXT_TEE => TeeFunction::from_regs(gprs.a_regs()).map(SbiMessage::Tee),
            EXT_MEASUREMENT => {
                MeasurementFunction::from_regs(gprs.a_regs()).map(SbiMessage::Measurement)
            }
            _ => Err(Error::UnknownSbiExtension),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a7(&self) -> u64 {
        match self {
            SbiMessage::Base(_) => EXT_BASE,
            SbiMessage::PutChar(_) => EXT_PUT_CHAR,
            SbiMessage::HartState(_) => EXT_HART_STATE,
            SbiMessage::Reset(_) => EXT_RESET,
            SbiMessage::Tee(_) => EXT_TEE,
            SbiMessage::Measurement(_) => EXT_MEASUREMENT,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a6(&self) -> u64 {
        match self {
            SbiMessage::Base(_) => 0, //TODO
            SbiMessage::HartState(f) => f.a6(),
            SbiMessage::PutChar(_) => 0,
            SbiMessage::Reset(_) => 0,
            SbiMessage::Tee(f) => f.a6(),
            SbiMessage::Measurement(f) => f.a6(),
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a5(&self) -> u64 {
        match self {
            SbiMessage::Tee(f) => f.a5(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a4(&self) -> u64 {
        match self {
            SbiMessage::Tee(f) => f.a4(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a3(&self) -> u64 {
        match self {
            SbiMessage::Tee(f) => f.a3(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a2(&self) -> u64 {
        match self {
            SbiMessage::HartState(f) => f.a2(),
            SbiMessage::Tee(f) => f.a2(),
            SbiMessage::Measurement(f) => f.a2(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a1(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.get_a1(),
            SbiMessage::HartState(f) => f.a1(),
            SbiMessage::Tee(f) => f.a1(),
            SbiMessage::Measurement(f) => f.a1(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a0(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.get_a0(),
            SbiMessage::PutChar(c) => *c,
            SbiMessage::HartState(f) => f.a0(),
            SbiMessage::Tee(f) => f.a0(),
            SbiMessage::Measurement(f) => f.a0(),
            _ => 0,
        }
    }

    /// Returns the result returned in the SbiMessage. Intended for use after an SbiMessage has been
    /// handled by the firmware. Interprets the given registers based on the extension and function
    /// and returns the approprate result.
    ///
    /// # Example
    ///
    /// ```rust
    /// pub fn ecall_send(msg: &SbiMessage) -> Result<u64> {
    ///     let mut a0 = msg.a0(); // error code
    ///     let mut a1 = msg.a1(); // return value
    ///     unsafe {
    ///         // Safe, but relies on trusting the hypervisor or firmware.
    ///         asm!("ecall", inout("a0") a0, inout("a1")a1,
    ///                 in("a2")msg.a2(), in("a3") msg.a3(),
    ///                 in("a4")msg.a4(), in("a5") msg.a5(),
    ///                 in("a6")msg.a6(), in("a7") msg.a7());
    ///     }
    ///
    ///     msg.result(a0, a1)
    /// }
    /// ```
    pub fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        match self {
            SbiMessage::Base(_) => {
                if a0 == 0 {
                    Ok(a1)
                } else {
                    Err(Error::InvalidParam) // TODO - set error
                }
            } //TODO
            SbiMessage::HartState(_) => Ok(a1), //TODO
            SbiMessage::PutChar(_) => Ok(0),
            SbiMessage::Reset(_) => Err(Error::InvalidParam),
            SbiMessage::Tee(f) => f.result(a0, a1),
            SbiMessage::Measurement(f) => f.result(a0, a1),
        }
    }
}
