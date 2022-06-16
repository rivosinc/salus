// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Rust SBI message parsing.
//! `SbiMessage` is an enum of all the SBI extensions.
//! For each extension, a function enum is defined to contain the SBI function data.
#![no_std]

mod consts;
pub use consts::*;

use riscv_regs::{GeneralPurposeRegisters, GprIndex};

/// Errors passed over the SBI protocol.
///
/// Constants from the SBI [spec](https://github.com/riscv-non-isa/riscv-sbi-doc/releases).
#[repr(i64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Generic failure in execution of the SBI call.
    Failed = -1,
    /// Extension or function is not supported.
    NotSupported = -2,
    /// Parameter passed isn't valid.
    InvalidParam = -3,
    /// Permission denied.
    Denied = -4,
    /// Address passed is invalid.
    InvalidAddress = -5,
    /// The given hart has already been started.
    AlreadyAvailable = -6,
    /// Some of the given counters have already been started.
    AlreadyStarted = -7,
    /// Some of the given counters have already been stopped.
    AlreadyStopped = -8,
}

impl Error {
    /// Parse the given error code to an `Error` enum.
    pub fn from_code(e: i64) -> Self {
        use Error::*;
        match e {
            -1 => Failed,
            -2 => NotSupported,
            -3 => InvalidParam,
            -4 => Denied,
            -5 => InvalidAddress,
            -6 => AlreadyAvailable,
            -7 => AlreadyStarted,
            -8 => AlreadyStopped,
            _ => Failed,
        }
    }
}

/// Holds the result of a TEE operation.
pub type Result<T> = core::result::Result<T, Error>;

/// A Trait for an SbiFunction. Implementers use this trait to specify how to parse from and
/// serialize into the a0-a7 registers used to make SBI calls.
pub trait SbiFunction {
    /// Returns the `u64` value that should be stored in register a6 before making the ecall for
    /// this function.
    fn a6(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a5 before making the ecall for
    /// this function.
    fn a5(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a4 before making the ecall for
    /// this function.
    fn a4(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a3 before making the ecall for
    /// this function.
    fn a3(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a2 before making the ecall for
    /// this function.
    fn a2(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a1 before making the ecall for
    /// this function.
    fn a1(&self) -> u64 {
        0
    }
    /// Returns the `u64` value that should be stored in register a0 before making the ecall for
    /// this function.
    fn a0(&self) -> u64 {
        0
    }
    /// Returns a result parsed from the a0 and a1 return value registers.
    fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        match a0 {
            0 => Ok(a1),
            e => Err(Error::from_code(e as i64)),
        }
    }
}

/// Functions defined for the Base extension
#[derive(Clone, Copy)]
pub enum BaseFunction {
    /// Returns the implemented version of the SBI standard.
    GetSpecificationVersion,
    /// Returns the ID of the SBI implementation.
    GetImplementationID,
    /// Returns the version of this SBI implementation.
    GetImplementationVersion,
    /// Checks if the given SBI extension is supported.
    ProbeSbiExtension(u64),
    /// Returns the vendor that produced this machine(`mvendorid`).
    GetMachineVendorID,
    /// Returns the architecture implementation ID this machine(`marchid`).
    GetMachineArchitectureID,
    /// Returns the implementation ID of this machine(`mimpid`).
    GetMachineImplementationID,
}

impl BaseFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    fn from_regs(args: &[u64]) -> Result<Self> {
        use BaseFunction::*;

        match args[6] {
            0 => Ok(GetSpecificationVersion),
            1 => Ok(GetImplementationID),
            2 => Ok(GetImplementationVersion),
            3 => Ok(ProbeSbiExtension(args[0])),
            4 => Ok(GetMachineVendorID),
            5 => Ok(GetMachineArchitectureID),
            6 => Ok(GetMachineImplementationID),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for BaseFunction {
    fn a6(&self) -> u64 {
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

    fn a0(&self) -> u64 {
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
    /// Starts the given hart.
    HartStart {
        /// a0 - hart id to start.
        hart_id: u64,
        /// a1 - address to start the hart.
        start_addr: u64,
        /// a2 - value to be set in a1 when starting the hart.
        opaque: u64,
    },
    /// Stops the current hart.
    HartStop,
    /// Returns the status of the given hart.
    HartStatus {
        /// a0 - ID of the hart to check.
        hart_id: u64,
    },
    /// Requests that the calling hart be suspended.
    HartSuspend {
        /// a0 - Specifies the type of suspend to initiate.
        suspend_type: u32,
        /// a1 - The address to jump to on resume.
        resume_addr: u64,
        /// a2 - An opaque value to load in a1 when resuming the hart.
        opaque: u64,
    },
}

/// Return value for the HartStatus SBI call.
#[repr(u64)]
pub enum HartState {
    /// The hart is physically powered-up and executing normally.
    Started = 0,
    /// The hart is not executing in supervisor-mode or any lower privilege mode.
    Stopped = 1,
    /// Some other hart has requested to start, operation still in progress.
    StartPending = 2,
    /// Some other hart has requested to stop, operation still in progress.
    StopPending = 3,
    /// This hart is in a platform specific suspend (or low power) state.
    Suspended = 4,
    /// The hart has requested to put itself in a platform specific low power state, in progress.
    SuspendPending = 5,
    /// An interrupt or platform specific hardware event has caused the hart to resume normal
    /// execution. Resuming is ongoing.
    ResumePending = 6,
}

impl StateFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    fn from_regs(args: &[u64]) -> Result<Self> {
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
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for StateFunction {
    fn a6(&self) -> u64 {
        use StateFunction::*;
        match self {
            HartStart { .. } => 0,
            HartStop => 1,
            HartStatus { .. } => 2,
            HartSuspend { .. } => 3,
        }
    }

    fn a0(&self) -> u64 {
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

    fn a1(&self) -> u64 {
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

    fn a2(&self) -> u64 {
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

/// Functions for the Reset extension
#[derive(Copy, Clone)]
pub enum ResetFunction {
    /// Performs a system reset.
    Reset {
        /// Determines the type of reset to perform.
        reset_type: ResetType,
        /// Represents the reason for system reset.
        reason: ResetReason,
    },
}

/// The types of reset a supervisor can request.
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResetType {
    /// Powers down the system.
    Shutdown = 0,
    /// Powers down, then reboots.
    ColdReset = 1,
    /// Reboots, doesn't power down.
    WarmReset = 2,
}

impl ResetType {
    // Creates a reset type from the a0 register value or returns an error if no mapping is
    // known for the given value.
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

/// Reasons why a supervisor requests a reset.
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResetReason {
    /// Used for normal resets.
    NoReason = 0,
    /// Used when the system has failed.
    SystemFailure = 1,
}

impl ResetReason {
    // Creates a reset reason from the a1 register value or returns an error if no mapping is
    // known for the given value.
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
    /// Creates an operation to shutdown the machine.
    pub fn shutdown() -> Self {
        ResetFunction::Reset {
            reset_type: ResetType::Shutdown,
            reason: ResetReason::NoReason,
        }
    }
}

impl ResetFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    fn from_regs(args: &[u64]) -> Result<Self> {
        use ResetFunction::*;

        Ok(match args[6] {
            0 => Reset {
                reset_type: ResetType::from_reg(args[0])?,
                reason: ResetReason::from_reg(args[1])?,
            },
            _ => return Err(Error::NotSupported),
        })
    }
}

impl SbiFunction for ResetFunction {
    fn a0(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type: _,
                reason,
            } => *reason as u64,
        }
    }

    fn a1(&self) -> u64 {
        match self {
            ResetFunction::Reset {
                reset_type,
                reason: _,
            } => *reset_type as u64,
        }
    }
}

/// List of registers that can be set prior to starting a confidential VM.
#[repr(u64)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TvmCpuRegister {
    /// Program Counter.
    Pc = 0,
    /// A1 - commonly used for a device tree pointer.
    A1 = 1,
}

impl TvmCpuRegister {
    // Returns the cpu register specified by the index or an error if the index is out of range.
    fn from_reg(a2: u64) -> Result<Self> {
        match a2 {
            0 => Ok(TvmCpuRegister::Pc),
            1 => Ok(TvmCpuRegister::A1),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Provides the state of the confidential VM supervisor.
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

/// Information returned from the system about the entity that manages confidential VMs and
/// confidential memory isolation.
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

/// Parameters used for creating a new confidential VM.
#[repr(C)]
pub struct TvmCreateParams {
    /// The base physical address of the 16kB confidential memory region that should be used for the
    /// TVM's page directory. Must be 16kB-aligned.
    pub tvm_page_directory_addr: u64,
    /// The base physical address of the confidential memory region to be used to hold the TVM's
    /// global state. Must be page-aligned and `TsmInfo::tvm_state_pages` pages in length.
    pub tvm_state_addr: u64,
    /// The maximum number of vCPUs that will be created for this TVM. Must be less than or equal
    /// to `TsmInfo::tvm_max_vcpus`.
    pub tvm_num_vcpus: u64,
    /// The base physical address of the confidential memory region to be used to hold the TVM's
    /// vCPU state. Must be page-aligned and `TsmInfo::tvm_bytes_per_vcpu` * `tvm_num_vcpus` bytes
    /// in length, rounded up to the nearest multiple of 4kB.
    pub tvm_vcpu_addr: u64,
}

/// Types of pages allowed to used for creating or managing confidential VMs.
#[repr(u64)]
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub enum TsmPageType {
    #[default]
    /// Standard 4k pages.
    Page4k = 0,
    /// 2 Megabyte pages.
    Page2M = 1,
    /// 1 Gigabyte pages.
    Page1G = 2,
    /// 512 Gigabyte pages.
    Page512G = 3,
}

impl TsmPageType {
    /// Attempts to create a page type from the given u64 register value. Returns an error if the
    /// value is greater than 3(512GB).
    pub fn from_reg(reg: u64) -> Result<Self> {
        use TsmPageType::*;
        match reg {
            0 => Ok(Page4k),
            1 => Ok(Page2M),
            2 => Ok(Page1G),
            3 => Ok(Page512G),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Functions provided by the TEE extension.
#[derive(Copy, Clone)]
pub enum TeeFunction {
    /// Creates a TVM from the parameters in the `TvmCreateParams` structure at the non-confidential
    /// physical address `params_addr`. Returns a guest ID that can be used to refer to the TVM in
    /// TVM management TEECALLs.
    ///
    /// a6 = 0
    TvmCreate {
        /// a0 = base physical address of the `TvmCreateParams` structure
        params_addr: u64,
        /// a1 = length of the `TvmCreateParams` structure in bytes
        len: u64,
    },
    /// Message to destroy a TVM created with `TvmCreate`.
    /// a6 = 1
    TvmDestroy {
        /// a0 = guest id returned from `TvmCreate`.
        guest_id: u64,
    },
    /// Adds `num_pages` 4kB pages of confidential memory starting at `page_addr` to the page-table
    /// page pool for the specified guest.
    ///
    /// 4k Pages.
    /// a6 = 2
    AddPageTablePages {
        /// a0 = guest_id
        guest_id: u64,
        /// a1 = address of the first page
        page_addr: u64,
        /// a2 = number of pages
        num_pages: u64,
    },
    /// Maps `num_pages` zero-filled pages of confidential memory starting at `page_addr` into the
    /// specified guest's address space at `guest_addr`.
    ///
    /// a6 = 3
    TvmAddZeroPages {
        /// a0 = guest_id
        guest_id: u64,
        /// a1 = physical address of the pages to insert
        page_addr: u64,
        /// a2 = page size
        page_type: TsmPageType,
        /// a3 = number of pages
        num_pages: u64,
        /// a4 = guest physical address
        guest_addr: u64,
    },
    /// Copies `num_pages` pages from non-confidential memory at `src_addr` to confidential
    /// memory at `dest_addr`, then measures and maps the pages at `dest_addr` into the specified
    /// guest's address space at `guest_addr`.
    ///
    /// a6 = 11
    TvmAddMeasuredPages {
        /// a0 = guest_id
        guest_id: u64,
        /// a1 = physical address of the pages to copy from
        src_addr: u64,
        /// a2 = physical address of the pages to insert
        dest_addr: u64,
        /// a3 = page size
        page_type: TsmPageType,
        /// a4 = number of pages
        num_pages: u64,
        /// a5 = guest physical address
        guest_addr: u64,
    },
    /// Moves a VM from the initializing state to the Runnable state
    /// a6 = 4
    Finalize {
        /// a0 = guest id
        guest_id: u64,
    },
    /// Runs the given vCPU in the TVM
    /// a6 = 5
    TvmCpuRun {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
    },
    /// Copies the measurements for the specified guest to the non-confidential physical address
    /// `dest_addr`. The measurement version and type must be set to 1 for now.
    ///
    /// a6 = 7
    GetGuestMeasurement {
        /// a0 = measurement version
        measurement_version: u64,
        /// a1 = measurement type
        measurement_type: u64,
        /// a2 = dest_addr
        dest_addr: u64,
        /// a3 = guest id
        guest_id: u64,
    },
    /// Adds a vCPU with ID `vcpu_id` to the guest `guest_id`. vCPUs may not be added after the TVM
    /// is finalized.
    ///
    /// a6 = 8
    TvmCpuCreate {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
    },
    /// Sets the register identified by `register` to `value` in the vCPU with ID `vcpu_id`. vCPU
    /// register state may not be modified after the TVM is finalized.
    ///
    /// a6 = 9
    TvmCpuSetRegister {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
        /// a2 = register id
        register: TvmCpuRegister,
        /// a3 = register value
        value: u64,
    },
    /// Writes up to `len` bytes of the `TsmInfo` structure to the non-confidential physical address
    /// `dest_addr`. Returns the number of bytes written.
    ///
    /// a6 = 10
    TsmGetInfo {
        /// a0 = destination address of the `TsmInfo` structure
        dest_addr: u64,
        /// a1 = maximum number of bytes to be written
        len: u64,
    },
    /// Converts `num_pages` of non-confidential memory starting at `page_addr`. The converted pages
    /// remain non-confidential, and thus may not be assinged for use by a child TVM, until the
    /// fence procedure, described below, has been completed.
    ///
    /// a6 = 12
    TsmConvertPages {
        /// a0 = base address of pages to convert
        page_addr: u64,
        /// a1 = page size
        page_type: TsmPageType,
        /// a2 = number of pages
        num_pages: u64,
    },
    /// Reclaims `num_pages` of confidential memory starting at `page_addr`. The pages must not
    /// be currently assigned to an active TVM.
    ///
    /// a6 = 13
    TsmReclaimPages {
        /// a0 = base address of pages to reclaim
        page_addr: u64,
        /// a1 = page size
        page_type: TsmPageType,
        /// a2 = number of pages
        num_pages: u64,
    },
    /// Initiates a TLB invalidation sequence for all pages marked for conversion via calls to
    /// `TsmConvertPages` between the previous `TsmInitiateFence` and now. The TLB invalidation
    /// sequence is completed when `TsmLocalFence` has been invoked on all other CPUs, after which
    /// the pages covered by the invalidation sequence are considered to be fully converted &
    /// confidential, and may be assigned for use by child TVMs. An error is returned if a TLB
    /// invalidation sequence is already in progress.
    ///
    /// a6 = 14
    TsmInitiateFence,
    /// Invalidates TLB entries for all pages pending conversion by an in-progress TLB invalidation
    /// operation on the local CPU.
    ///
    /// a6 = 15
    TsmLocalFence,
}

impl TeeFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    fn from_regs(args: &[u64]) -> Result<Self> {
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
            3 => Ok(TvmAddZeroPages {
                guest_id: args[0],
                page_addr: args[1],
                page_type: TsmPageType::from_reg(args[2])?,
                num_pages: args[3],
                guest_addr: args[4],
            }),
            4 => Ok(Finalize { guest_id: args[0] }),
            5 => Ok(TvmCpuRun {
                guest_id: args[0],
                vcpu_id: args[1],
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
            11 => Ok(TvmAddMeasuredPages {
                guest_id: args[0],
                src_addr: args[1],
                dest_addr: args[2],
                page_type: TsmPageType::from_reg(args[3])?,
                num_pages: args[4],
                guest_addr: args[5],
            }),
            12 => Ok(TsmConvertPages {
                page_addr: args[0],
                page_type: TsmPageType::from_reg(args[1])?,
                num_pages: args[2],
            }),
            13 => Ok(TsmReclaimPages {
                page_addr: args[0],
                page_type: TsmPageType::from_reg(args[1])?,
                num_pages: args[2],
            }),
            14 => Ok(TsmInitiateFence),
            15 => Ok(TsmLocalFence),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for TeeFunction {
    fn a6(&self) -> u64 {
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
            TvmAddZeroPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => 3,
            Finalize { guest_id: _ } => 4,
            TvmCpuRun {
                guest_id: _,
                vcpu_id: _,
            } => 5,
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
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => 11,
            TsmConvertPages {
                page_addr: _,
                page_type: _,
                num_pages: _,
            } => 12,
            TsmReclaimPages {
                page_addr: _,
                page_type: _,
                num_pages: _,
            } => 13,
            TsmInitiateFence => 14,
            TsmLocalFence => 15,
        }
    }

    fn a0(&self) -> u64 {
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
            TvmAddZeroPages {
                guest_id,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *guest_id,
            Finalize { guest_id } => *guest_id,
            TvmCpuRun {
                guest_id,
                vcpu_id: _,
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
            TvmAddMeasuredPages {
                guest_id,
                src_addr: _,
                dest_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *guest_id,
            TsmConvertPages {
                page_addr,
                page_type: _,
                num_pages: _,
            } => *page_addr,
            TsmReclaimPages {
                page_addr,
                page_type: _,
                num_pages: _,
            } => *page_addr,
            _ => 0,
        }
    }

    fn a1(&self) -> u64 {
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
            TvmAddZeroPages {
                guest_id: _,
                page_addr,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *page_addr,
            TvmCpuRun {
                guest_id: _,
                vcpu_id,
            } => *vcpu_id,
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
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr,
                dest_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *src_addr,
            TsmConvertPages {
                page_addr: _,
                page_type,
                num_pages: _,
            } => *page_type as u64,
            TsmReclaimPages {
                page_addr: _,
                page_type,
                num_pages: _,
            } => *page_type as u64,
            _ => 0,
        }
    }

    fn a2(&self) -> u64 {
        use TeeFunction::*;
        match self {
            AddPageTablePages {
                guest_id: _,
                page_addr: _,
                num_pages,
            } => *num_pages,
            TvmAddZeroPages {
                guest_id: _,
                page_addr: _,
                page_type,
                num_pages: _,
                guest_addr: _,
            } => *page_type as u64,
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
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *dest_addr,
            TsmConvertPages {
                page_addr: _,
                page_type: _,
                num_pages,
            } => *num_pages,
            TsmReclaimPages {
                page_addr: _,
                page_type: _,
                num_pages,
            } => *num_pages,
            _ => 0,
        }
    }

    fn a3(&self) -> u64 {
        use TeeFunction::*;
        match self {
            TvmAddZeroPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages,
                guest_addr: _,
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
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr: _,
                page_type,
                num_pages: _,
                guest_addr: _,
            } => *page_type as u64,
            _ => 0,
        }
    }

    fn a4(&self) -> u64 {
        use TeeFunction::*;
        match self {
            TvmAddZeroPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr,
            } => *guest_addr,
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr: _,
                page_type: _,
                num_pages,
                guest_addr: _,
            } => *num_pages,
            _ => 0,
        }
    }

    fn a5(&self) -> u64 {
        use TeeFunction::*;
        match self {
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr,
            } => *guest_addr,
            _ => 0,
        }
    }
}

/// Functions provided by the measurement extension.
#[derive(Copy, Clone)]
pub enum MeasurementFunction {
    /// Copies the measurements for the current VM to the (guest) physical address in `dest_addr`.
    /// The measurement version and type must be set to 1 for now.
    /// a6 = 0
    GetSelfMeasurement {
        /// a0 = measurement version
        measurement_version: u64,
        /// a1 = measurement type
        measurement_type: u64,
        /// a2 = dest_addr
        dest_addr: u64,
    },
}

impl MeasurementFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    fn from_regs(args: &[u64]) -> Result<Self> {
        use MeasurementFunction::*;
        match args[6] {
            0 => Ok(GetSelfMeasurement {
                measurement_version: args[0],
                measurement_type: args[1],
                dest_addr: args[2],
            }),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for MeasurementFunction {
    fn a6(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version: _,
                measurement_type: _,
                dest_addr: _,
            } => 0,
        }
    }

    fn a0(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version,
                measurement_type: _,
                dest_addr: _,
            } => *measurement_version,
        }
    }

    fn a1(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version: _,
                measurement_type,
                dest_addr: _,
            } => *measurement_type,
        }
    }

    fn a2(&self) -> u64 {
        use MeasurementFunction::*;
        match self {
            GetSelfMeasurement {
                measurement_version: _,
                measurement_type: _,
                dest_addr,
            } => *dest_addr,
        }
    }
}

/// The values returned from an SBI function call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SbiReturn {
    /// The error code (0 for success).
    pub error_code: i64,
    /// The return value if the operation is successful.
    pub return_value: u64,
}

impl SbiReturn {
    /// Returns an `SbiReturn` that indicates success.
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
            error_code: error as i64,
            return_value: 0,
        }
    }
}

impl From<SbiReturn> for Result<u64> {
    fn from(ret: SbiReturn) -> Result<u64> {
        match ret.error_code {
            SBI_SUCCESS => Ok(ret.return_value),
            e => Err(Error::from_code(e)),
        }
    }
}

/// SBI return value conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SbiReturnType {
    /// Legacy (v0.1) extensions return a single value in A0, usually with the convention that 0
    /// is success and < 0 is an implementation defined error code.
    Legacy(u64),
    /// Modern extensions use the standard error code values enumerated above.
    Standard(SbiReturn),
}

/// SBI Message used to invoke the specified SBI extension in the firmware.
#[derive(Clone, Copy)]
pub enum SbiMessage {
    /// The base SBI extension functions.
    Base(BaseFunction),
    /// The legacy PutChar extension.
    PutChar(u64),
    /// The extension for getting/setting the state of CPUs.
    HartState(StateFunction),
    /// Handles system reset.
    Reset(ResetFunction),
    /// Provides capabilities for starting confidential virtual machines.
    Tee(TeeFunction),
    /// Allows acquiring measurement of the system that booted the running code.
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
            EXT_RESET => ResetFunction::from_regs(gprs.a_regs()).map(SbiMessage::Reset),
            EXT_TEE => TeeFunction::from_regs(gprs.a_regs()).map(SbiMessage::Tee),
            EXT_MEASUREMENT => {
                MeasurementFunction::from_regs(gprs.a_regs()).map(SbiMessage::Measurement)
            }
            _ => Err(Error::NotSupported),
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
            SbiMessage::Reset(r) => r.a1(),
            SbiMessage::HartState(f) => f.a1(),
            SbiMessage::Tee(f) => f.a1(),
            SbiMessage::Measurement(f) => f.a1(),
            _ => 0,
        }
    }

    /// Returns the register value for this `SbiMessage`.
    pub fn a0(&self) -> u64 {
        match self {
            SbiMessage::Reset(r) => r.a0(),
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
        let ret = SbiReturn {
            error_code: a0 as i64,
            return_value: a1,
        };
        match self {
            // For legacy messages, a0 is 0 on success and an implementation-defined error value on
            // failure. Nothing is returned in a1.
            SbiMessage::PutChar(_) => match a0 as i64 {
                SBI_SUCCESS => Ok(0),
                _ => Err(Error::Failed),
            },
            _ => ret.into(),
        }
    }
}
