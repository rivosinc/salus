// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// The exit cause for a TVM vCPU returned from TvmCpuRun. Certain exit causes may be accompanied
/// by more detailed cause information (e.g. faulting address) in which case that information can
/// be retrieved by accessing the virtual `ExitCause` registers of the vCPU.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TvmCpuExitCode {
    /// The vCPU exited due to an interrupt directed at the host.
    HostInterrupt = 0,

    /// The vCPU made a sbi_system_reset() call. The reset type and cuase are stored in the
    /// `ExitCause0` and `ExitCause1` registers, respectively. All vCPUs of the TVM are no longer
    /// runnable.
    SystemReset = 1,

    /// The vCPU made a sbi_hart_start() call. The target hart ID is stored in `ExitCause0` and the
    /// vCPU matching the hart ID is made runnable.
    HartStart = 2,

    /// The vCPU made a sbi_hart_stop() call. The vCPU is no longer runnable.
    HartStop = 3,

    /// The vCPU encountered a guest page fault in a confidential memory region. The faulting guest
    /// physical address is stored in `ExitCause0`. The vCPU will resume at the faulting instruction
    /// the next time it is run.
    ///
    /// TODO: Do we need to differentiate between the type (fetch/load/store) of page fault?
    ConfidentialPageFault = 4,

    /// The vCPU encountered a guest page fault in an emulated MMIO memory region. The faulting
    /// guest physical address is stored in `ExitCause0` and the type of memory operation the vCPU
    /// was attempting to execute is stored in `ExitCause1`. The `MmioLoadValue` and `MmioStoreValue`
    /// registers are used to complete an emulated MMIO access; see `TvmMmioOpCode` for more details.
    /// The vCPU resumes at the following instruction the next time it is run.
    MmioPageFault = 5,

    /// The vCPU executed a WFI instruction.
    WaitForInterrupt = 6,

    /// The vCPU encountered an exception that the TSM cannot handle internally and that cannot
    /// be safely delegated to the host. The value of the SCAUSE register is stored in `ExitCause0`.
    /// The vCPU is no longer runnable.
    UnhandledException = 7,

    /// The vCPU encountered a guest page fault in a shared memory region. The faulting guest
    /// physical address is stored in `ExitCause0`. The vCPU will resume at the faulting instruction
    /// the next time it is run.
    ///
    /// TODO: Do we need to differentiate between the type (fetch/load/store) of page fault?
    SharedPageFault = 8,
}

impl TvmCpuExitCode {
    /// Creates a `TvmCpuExitCode` from the raw value as returned in A1.
    pub fn from_reg(a1: u64) -> Result<Self> {
        use TvmCpuExitCode::*;
        match a1 {
            0 => Ok(HostInterrupt),
            1 => Ok(SystemReset),
            2 => Ok(HartStart),
            3 => Ok(HartStop),
            4 => Ok(ConfidentialPageFault),
            5 => Ok(MmioPageFault),
            6 => Ok(WaitForInterrupt),
            7 => Ok(UnhandledException),
            8 => Ok(SharedPageFault),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// List of possible operations a TVM's vCPU can make when accessing an emulated MMIO region.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TvmMmioOpCode {
    /// Loads a 64-bit value. The result of the emulated MMIO load can be written to `MmioLoadValue`.
    Load64 = 0,
    /// Loads and sign-extends a 32-bit value.
    Load32 = 1,
    /// Loads and zero-extends a 32-bit value.
    Load32U = 2,
    /// Loads and sign-extends a 16-bit value.
    Load16 = 3,
    /// Loads and zero-extends a 16-bit value.
    Load16U = 4,
    /// Loads and sign-extends an 8-bit value.
    Load8 = 5,
    /// Loads and zero-extends an 8-bit value.
    Load8U = 6,

    /// Stores a 64-bit value. The value to be stored by the emulated MMIO store can be read from
    /// `MmioStoreValue`.
    Store64 = 7,
    /// Stores a 32-bit value.
    Store32 = 8,
    /// Stores a 16-bit value.
    Store16 = 9,
    /// Stores an 8-bit value.
    Store8 = 10,
    // TODO: AMO instructions?
}

impl TvmMmioOpCode {
    /// Returns the `TvmMmioOpCode` specified by the raw value.
    pub fn from_reg(cause1: u64) -> Result<Self> {
        use TvmMmioOpCode::*;
        match cause1 {
            0 => Ok(Load64),
            1 => Ok(Load32),
            2 => Ok(Load32U),
            3 => Ok(Load16),
            4 => Ok(Load16U),
            5 => Ok(Load8),
            6 => Ok(Load8U),
            7 => Ok(Store64),
            8 => Ok(Store32),
            9 => Ok(Store16),
            10 => Ok(Store8),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// List of registers that can be read or written for a TVM's vCPU.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TvmCpuRegister {
    /// Entry point (initial SEPC) of the boot CPU of a TVM. Read-write prior to TVM finalization;
    /// inaccessible after the TVM has started.
    EntryPc = 0,

    /// Boot argument (stored in A1, usually a pointer to a device-tree) of the boot CPU of a TVM.
    /// Read-write prior to TVM finalization; inaccessible after the TVM has started.
    EntryArg = 1,

    /// Detailed TVM CPU exit cause register. Read-only, and only accessible after the TVM has
    /// started.
    ExitCause0 = 2,

    /// An additional exit cause register with the same access properties as `ExitCause0`.
    ExitCause1 = 3,

    /// Value used to complete an emulated MMIO load by a TVM CPU. Read-write, and only accessible
    /// after the TVM has started.
    MmioLoadValue = 4,

    /// Value stored by a TVM CPU to an emulated MMIO region. Read-only, and only accessible after
    /// the TVM has started.
    MmioStoreValue = 5,
}

impl TvmCpuRegister {
    /// Returns the cpu register specified by the index or an error if the index is out of range.
    pub fn from_reg(a2: u64) -> Result<Self> {
        use TvmCpuRegister::*;
        match a2 {
            0 => Ok(EntryPc),
            1 => Ok(EntryArg),
            2 => Ok(ExitCause0),
            3 => Ok(ExitCause1),
            4 => Ok(MmioLoadValue),
            5 => Ok(MmioStoreValue),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Identifies a register set in the vCPU shared-memory state area layout.
///
/// Register sets are roughly grouped by the specification or extension that defines them.
/// While it is likely that registers from multiple register sets will be used in the course
/// of handling a vCPU exit, the register sets are grouped in this way in order to allow for
/// extensibility by future extensions, and to avoid being overly prescriptive with regards
/// to TSM or hypervisor implementation.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterSetId {
    /// General purpose registers.
    Gprs = 0,
    /// Supervisor CSRs.
    SupervisorCsrs = 1,
    /// HS-level CSRs.
    HypervisorCsrs = 2,
}

impl RegisterSetId {
    /// Returns the `RegisterSetId` corresponding to `val`.
    pub fn from_raw(val: u16) -> Result<Self> {
        match val {
            0 => Ok(RegisterSetId::Gprs),
            1 => Ok(RegisterSetId::SupervisorCsrs),
            2 => Ok(RegisterSetId::HypervisorCsrs),
            _ => Err(Error::InvalidParam),
        }
    }

    /// Returns the size in bytes of the register set structure corresponding to this ID.
    pub fn struct_size(&self) -> usize {
        match self {
            RegisterSetId::Gprs => core::mem::size_of::<Gprs>(),
            RegisterSetId::SupervisorCsrs => core::mem::size_of::<SupervisorCsrs>(),
            RegisterSetId::HypervisorCsrs => core::mem::size_of::<HypervisorCsrs>(),
        }
    }
}

/// Specifies the location of a particular register set in the vCPU shared-memory state area.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RegisterSetLocation {
    /// The type of the register set, identified by the `RegisterSetId` enum.
    pub id: u16,
    /// The version of the register set structure.
    ///
    /// As the definition of a particular register set may change across revisions of this API,
    /// this version number can be used to identify the specific layout that is being used.
    pub version: u16,
    /// The offset of the register set from the start of the vCPU's shared-memory state area.
    pub offset: u32,
}

/// General purpose registers. Structure for register sets of type `RegisterSetId::Gprs`.
///
/// The TSM will always read or write the minimum number of registers in this set to complete
/// the requested action, in order to avoid leaking information from the TVM.
///
/// The TSM will write to these registers upon return from `TvmCpuRun` when:
///  - The vCPU takes a store guest page fault in an emulated MMIO region.
///  - The vCPU makes an ECALL that is to be forwarded to the host.
///
/// The TSM will read from these registers when:
///  - The vCPU takes a load guest page fault in an emulated MMIO region.
///  - The host calls `TvmFinalize`, latching the entry point argument (stored in 'A1') for the
///    TVM's boot vCPU.
#[repr(C)]
#[derive(Default)]
pub struct Gprs(pub [u64; 32]);

/// Supervisor CSRs. Structure for register sets of type `RegisterSetId::SupervisorCsrs`.
#[repr(C)]
#[derive(Default)]
pub struct SupervisorCsrs {
    /// Initial SEPC value (entry point) of a TVM vCPU. Latched for the TVM's boot vCPU at
    /// `TvmFinalzie`; ignored for all other vCPUs.
    pub sepc: u64,
    /// SCAUSE value for the trap taken by the TVM vCPU. Written by the TSM upon return from
    /// `TvmCpuRun`.
    pub scause: u64,
    /// STVAL value for guest page faults or virtual instruction exceptions taken by the TVM vCPU.
    /// Written by the TSM upon return from `TvmCpuRun`.
    ///
    /// Note that guest virtual addresses are not exposed by the TSM, so only the 2 LSBs will
    /// ever be non-zero for guest page fault exceptions.
    pub stval: u64,
}

/// HS-level CSRs. Structure for register sets of type `RegisterSetId::HypervisorCsrs`.
#[repr(C)]
#[derive(Default)]
pub struct HypervisorCsrs {
    /// HTVAL value for guest page faults taken by the TVM vCPU. Written by the TSM upon return
    /// from `TvmCpuRun`.
    pub htval: u64,
    /// HTINST value for guest page faults or virtual instruction exceptions taken by the TVM vCPU.
    /// Written (in certain circumstances) by the TSM upon return from `TvmCpuRun`.
    ///
    /// The TSM will only write `htinst` in the following cases:
    ///  - MMIO load page faults. The value written to the register in `gprs` corresponding to the
    ///    'rd' register in the instruction will be used to complete the load upon the next call to
    ///    `TvmCpuRun` for this vCPU.
    ///  - MMIO store page faults. The TSM will write the value to be stored by the vCPU to the
    ///    register in `gprs` corresponding to the 'rs2' register in the instruction upon return
    ///    from `TvmCpuRun`.
    pub htinst: u64,
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

    /// Returns the size of this page type in bytes.
    pub fn size_bytes(&self) -> u64 {
        match self {
            TsmPageType::Page4k => 4096,
            TsmPageType::Page2M => 2 * 1024 * 1024,
            TsmPageType::Page1G => 1024 * 1024 * 1024,
            TsmPageType::Page512G => 512 * 1024 * 1024 * 1024,
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
    ///
    /// a6 = 1
    TvmDestroy {
        /// a0 = guest id returned from `TvmCreate`.
        guest_id: u64,
    },
    /// Adds `num_pages` 4kB pages of confidential memory starting at `page_addr` to the page-table
    /// page pool for the specified guest.
    ///
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
    /// specified guest's address space at `guest_addr`. The mapping must lie within a region of
    /// confidential memory created with `TvmAddConfidentialMemoryRegion`. Zero pages may be added
    /// at any time.
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
    /// Moves a VM from the initializing state to the Runnable state
    ///
    /// a6 = 4
    Finalize {
        /// a0 = guest id
        guest_id: u64,
    },
    /// Runs the given vCPU in the TVM
    ///
    /// a6 = 5
    TvmCpuRun {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
    },
    /// Writes the layout that the TSM will use for the vCPU shared-memory state area for vCPUs
    /// of the TVM identified by `guest_id` to the non-confidential physical address `layout_addr`.
    /// The caller uses this to discover the size and layout of the structure that will be used
    /// to communicate vCPU state in shared-memory.
    ///
    /// Returns the number of bytes written to `layout_addr` upon success, or an error if
    /// `layout_addr` is invalid or `layout_len` is insufficiently large to describe the entire
    /// layout.
    ///
    /// a6 = 21
    TvmCpuGetMemLayout {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = base physical address of the array of `RegisterSetLayout` structs
        layout_addr: u64,
        /// a2 = length of the `RegisterSetLayout` array in bytes
        layout_len: u64,
    },
    /// Adds a vCPU with ID `vcpu_id` to the guest `guest_id`, registering `shared_page_addr` as
    /// the location of the shared-memory state area for this vCPU.
    ///
    /// `shared_page_addr` must be page-aligned and point to a sufficient number of non-confidential
    /// pages to hold a structure with the layout specified by `TvmCpuGetMemLayout`. These pages
    /// are "pinned" in the non-confidential state (i.e. cannot be converted to confidential) until
    /// the TVM is destroyed.
    ///
    /// vCPUs may not be added after the TVM is finalized.
    ///
    /// a6 = 8
    TvmCpuCreate {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
        /// a2 = page address of shared state structure
        shared_page_addr: u64,
    },
    /// Sets the register identified by `register` to `value` in the vCPU with ID `vcpu_id`. See
    /// the defintion of `TvmCpuRegister` for details on which registers are writeable and when.
    ///
    /// TODO: Remove once communication of vCPU state over shared memory is implemented.
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
    /// Copies `num_pages` pages from non-confidential memory at `src_addr` to confidential
    /// memory at `dest_addr`, then measures and maps the pages at `dest_addr` into the specified
    /// guest's address space at `guest_addr`. The mapping must lie within a region of confidential
    /// memory created with `TvmAddConfidentialMemoryRegion`. Measured pages may only be added prior
    /// to TVM finalization.
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
    /// Converts `num_pages` of non-confidential memory starting at `page_addr`. The converted pages
    /// remain non-confidential, and thus may not be assigned for use by a child TVM, until the
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
    /// Gets the register identified by `register` in the vCPU with ID `vcpu_id`. See the definition
    /// of `TvmCpuRegister` for details on which registers are readable and when. The contents of
    /// the specified register are returned upon success.
    ///
    /// TODO: Remove once communication of vCPU state over shared memory is implemented.
    ///
    /// a6 = 16
    TvmCpuGetRegister {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
        /// a2 = register id
        register: TvmCpuRegister,
    },
    /// Marks the specified range of guest physical address space as reserved for the mapping of
    /// confidential memory. The region is initially unpopulated. Pages of confidential memory may
    /// be inserted with `TvmAddZeroPages` and `TvmAddMeasuredPages`. Both `guest_addr` and `len`
    /// must be 4kB-aligned. Confidential memory regions may only be added to TVMs prior to
    /// finalization.
    ///
    /// a6 = 17
    TvmAddConfidentialMemoryRegion {
        /// a0 = guest_id
        guest_id: u64,
        /// a1 = start of the confidential memory region
        guest_addr: u64,
        /// a2 = length of the confidential memory region
        len: u64,
    },
    /// Marks the specified range of guest physical address space as used for emulated MMIO.
    /// The region is unpopulated; attempts by a TVM vCPU to access this region will cause a
    /// `MmioPageFault` exit from `TvmCpuRun`. Both `guest_addr` and `len` must be 4kB-aligned.
    /// Emulated MMIO regions may only be added to TVMs prior to finalization.
    ///
    /// a6 = 18
    TvmAddEmulatedMmioRegion {
        /// a0 = guest_id
        guest_id: u64,
        /// a1 = start of the emulated MMIO region
        guest_addr: u64,
        /// a2 = length of the emulated MMIO region
        len: u64,
    },
    /// Marks the specified range of guest physical address space as reserved for the mapping of
    /// shared memory. The region is initially unpopulated. Pages of shared memory may be inserted with
    /// `TvmAddSharedPages`. Attempts by a TVM vCPU to access an unpopulated region will cause a `SharedPageFault`
    /// exit from `TvmCpuRun` Both `guest_addr` and `len` must be 4kB-aligned.
    /// Shared memory regions may only be added to TVMs prior to finalization.
    ///
    /// a6 = 19
    TvmAddSharedMemoryRegion {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = start of the shared memory region
        guest_addr: u64,
        /// a2 = length of the shared memory region
        len: u64,
    },
    /// Maps non-confidential shared pages in a range previously defined by `TvmAddSharedMemoryRegion`.
    /// The call may be made before or after TVM finalization, and shared pages can be mapped-in on demand.
    ///
    /// a6 = 20
    TvmAddSharedPages {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = start of the shared memory region
        page_addr: u64,
        /// a2 = page size (must be Page4k for now)
        page_type: TsmPageType,
        /// a3 = number of pages
        num_pages: u64,
        /// a4 = guest physical address
        guest_addr: u64,
    },
}

impl TeeFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
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
            8 => Ok(TvmCpuCreate {
                guest_id: args[0],
                vcpu_id: args[1],
                shared_page_addr: args[2],
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
            16 => Ok(TvmCpuGetRegister {
                guest_id: args[0],
                vcpu_id: args[1],
                register: TvmCpuRegister::from_reg(args[2])?,
            }),
            17 => Ok(TvmAddConfidentialMemoryRegion {
                guest_id: args[0],
                guest_addr: args[1],
                len: args[2],
            }),
            18 => Ok(TvmAddEmulatedMmioRegion {
                guest_id: args[0],
                guest_addr: args[1],
                len: args[2],
            }),
            19 => Ok(TvmAddSharedMemoryRegion {
                guest_id: args[0],
                guest_addr: args[1],
                len: args[2],
            }),
            20 => Ok(TvmAddSharedPages {
                guest_id: args[0],
                page_addr: args[1],
                page_type: TsmPageType::from_reg(args[2])?,
                num_pages: args[3],
                guest_addr: args[4],
            }),
            21 => Ok(TvmCpuGetMemLayout {
                guest_id: args[0],
                layout_addr: args[1],
                layout_len: args[2],
            }),
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
            TvmCpuCreate {
                guest_id: _,
                vcpu_id: _,
                shared_page_addr: _,
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
            TvmCpuGetRegister {
                guest_id: _,
                vcpu_id: _,
                register: _,
            } => 16,
            TvmAddConfidentialMemoryRegion {
                guest_id: _,
                guest_addr: _,
                len: _,
            } => 17,
            TvmAddEmulatedMmioRegion {
                guest_id: _,
                guest_addr: _,
                len: _,
            } => 18,
            TvmAddSharedMemoryRegion {
                guest_id: _,
                guest_addr: _,
                len: _,
            } => 19,
            TvmAddSharedPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => 20,
            TvmCpuGetMemLayout {
                guest_id: _,
                layout_addr: _,
                layout_len: _,
            } => 21,
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
            TvmCpuCreate {
                guest_id,
                vcpu_id: _,
                shared_page_addr: _,
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
            TvmCpuGetRegister {
                guest_id,
                vcpu_id: _,
                register: _,
            } => *guest_id,
            TvmAddConfidentialMemoryRegion {
                guest_id,
                guest_addr: _,
                len: _,
            } => *guest_id,
            TvmAddEmulatedMmioRegion {
                guest_id,
                guest_addr: _,
                len: _,
            } => *guest_id,
            TvmAddSharedMemoryRegion {
                guest_id,
                guest_addr: _,
                len: _,
            } => *guest_id,
            TvmAddSharedPages {
                guest_id,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *guest_id,
            TvmCpuGetMemLayout {
                guest_id,
                layout_addr: _,
                layout_len: _,
            } => *guest_id,
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
            TvmCpuCreate {
                guest_id: _,
                vcpu_id,
                shared_page_addr: _,
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
            TvmCpuGetRegister {
                guest_id: _,
                vcpu_id,
                register: _,
            } => *vcpu_id,
            TvmAddConfidentialMemoryRegion {
                guest_id: _,
                guest_addr,
                len: _,
            } => *guest_addr,
            TvmAddEmulatedMmioRegion {
                guest_id: _,
                guest_addr,
                len: _,
            } => *guest_addr,
            TvmAddSharedMemoryRegion {
                guest_id: _,
                guest_addr,
                len: _,
            } => *guest_addr,
            TvmAddSharedPages {
                guest_id: _,
                page_addr,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *page_addr,
            TvmCpuGetMemLayout {
                guest_id: _,
                layout_addr,
                layout_len: _,
            } => *layout_addr,
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
            TvmCpuCreate {
                guest_id: _,
                vcpu_id: _,
                shared_page_addr,
            } => *shared_page_addr,
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
            TvmCpuGetRegister {
                guest_id: _,
                vcpu_id: _,
                register,
            } => *register as u64,
            TvmAddConfidentialMemoryRegion {
                guest_id: _,
                guest_addr: _,
                len,
            } => *len,
            TvmAddEmulatedMmioRegion {
                guest_id: _,
                guest_addr: _,
                len,
            } => *len,
            TvmAddSharedMemoryRegion {
                guest_id: _,
                guest_addr: _,
                len,
            } => *len,
            TvmAddSharedPages {
                guest_id: _,
                page_addr: _,
                page_type,
                num_pages: _,
                guest_addr: _,
            } => *page_type as u64,
            TvmCpuGetMemLayout {
                guest_id: _,
                layout_addr: _,
                layout_len,
            } => *layout_len,
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
            TvmAddSharedPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages,
                guest_addr: _,
            } => *num_pages,
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
            TvmAddSharedPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr,
            } => *guest_addr,
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
