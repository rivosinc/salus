// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Identifies a register set in the vCPU shared-memory state area layout. Each identifier
/// maps to a structure defining the in-memory layout of the registers in the register set.
/// This mapping is guaranteed to remain stable across versions of the specification; if a
/// future revision modifies the definition of a register set structure then it must use a
/// new `RegisterSetId`.
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
///
/// TODO: Provide a way to discover the length of a register set in order to allow hosts to
/// compute the size of shared-memory state area even if they don't recognize all the IDs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RegisterSetLocation {
    /// The type of the register set, identified by the `RegisterSetId` enum.
    pub id: u16,
    /// The offset of the register set from the start of the vCPU's shared-memory state area.
    pub offset: u16,
}

impl From<RegisterSetLocation> for u32 {
    fn from(regset: RegisterSetLocation) -> u32 {
        (regset.id as u32) | ((regset.offset as u32) << 16)
    }
}

impl From<u32> for RegisterSetLocation {
    fn from(val: u32) -> Self {
        Self {
            id: (val & 0xffff) as u16,
            offset: (val >> 16) as u16,
        }
    }
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
    /// The number of 4kB pages which must be donated to the TSM when creating a new VCPU.
    pub tvm_vcpu_state_pages: u64,
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
}

/// Types of pages allowed to used for creating or managing confidential VMs.
#[repr(u64)]
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
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

/// Types of memory regions that can be created in a TVM.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeeMemoryRegion {
    /// A region of memory where a TVM's host may insert pages of confidential memory. The region
    /// is initially unpopulated. Accesses by the TVM to unmapped pages in the regino will cause
    /// a guest page fault exit with the faulting page address reported to the TVM's host.
    Confidential = 0,
    /// A region of memory where a TVM's host may insert pages of shared, non-confidential memory.
    /// The region is initially unpopulated. Accesses by the TVM to unmapped pages in the region
    /// will cause a guest page fault exit with the faulting page address reported to the TVM's
    /// host.
    Shared = 1,
    /// An unpopulated region of memory where accesses by the TVM will cause guest page fault
    /// exit with the faulting address and a transformed version of the faulting instruction
    /// reported to the TVM's host, allowing the host to emulate the faulting memory access.
    EmulatedMmio = 2,
}

impl TeeMemoryRegion {
    /// Returns the `TeeMemoryRegion` corresponding to the value.
    pub fn from_reg(reg: u64) -> Result<Self> {
        use TeeMemoryRegion::*;
        match reg {
            0 => Ok(Confidential),
            1 => Ok(Shared),
            2 => Ok(EmulatedMmio),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Functions provided by the TEE Host extension.
#[derive(Copy, Clone, Debug)]
pub enum TeeHostFunction {
    /// Writes up to `len` bytes of the `TsmInfo` structure to the non-confidential physical address
    /// `dest_addr`. Returns the number of bytes written.
    ///
    /// a6 = 0
    TsmGetInfo {
        /// a0 = destination address of the `TsmInfo` structure
        dest_addr: u64,
        /// a1 = maximum number of bytes to be written
        len: u64,
    },
    /// Converts `num_pages` of 4kB page-size non-confidential memory starting at `page_addr`. The converted pages
    /// remain non-confidential, and thus may not be assigned for use by a child TVM, until the
    /// fence procedure, described below, has been completed.
    ///
    /// a6 = 1
    TsmConvertPages {
        /// a0 = base address of pages to convert
        page_addr: u64,
        /// a1 = number of pages
        num_pages: u64,
    },
    /// Reclaims `num_pages` of 4kB page-size confidential memory starting at `page_addr`. The pages must not
    /// be currently assigned to an active TVM.
    ///
    /// a6 = 2
    TsmReclaimPages {
        /// a0 = base address of pages to reclaim
        page_addr: u64,
        /// a1 = number of pages
        num_pages: u64,
    },
    /// Initiates a TLB invalidation sequence for all pages marked for conversion via calls to
    /// `TsmConvertPages` between the previous `TsmInitiateFence` and now. The TLB invalidation
    /// sequence is completed when `TsmLocalFence` has been invoked on all other CPUs, after which
    /// the pages covered by the invalidation sequence are considered to be fully converted &
    /// confidential, and may be assigned for use by child TVMs. An error is returned if a TLB
    /// invalidation sequence is already in progress.
    ///
    /// a6 = 3
    TsmInitiateFence,
    /// Invalidates TLB entries for all pages pending conversion by an in-progress TLB invalidation
    /// operation on the local CPU.
    ///
    /// a6 = 4
    TsmLocalFence,
    /// Creates a TVM from the parameters in the `TvmCreateParams` structure at the non-confidential
    /// physical address `params_addr`. Returns a guest ID that can be used to refer to the TVM in
    /// TVM management TEECALLs.
    ///
    /// a6 = 5
    TvmCreate {
        /// a0 = base physical address of the `TvmCreateParams` structure
        params_addr: u64,
        /// a1 = length of the `TvmCreateParams` structure in bytes
        len: u64,
    },
    /// Moves a VM from the initializing state to the Runnable state
    ///
    /// a6 = 6
    Finalize {
        /// a0 = guest id
        guest_id: u64,
    },
    /// Message to destroy a TVM created with `TvmCreate`.
    ///
    /// a6 = 7
    TvmDestroy {
        /// a0 = guest id returned from `TvmCreate`.
        guest_id: u64,
    },
    /// Adds a memory region to the TVM identified by `guest_id` at the specified range of guest
    /// physical address space. Both `addr` and `len` must be 4kB-aligned and must not overlap with
    /// any previously-added regions.
    ///
    /// Only `Confidential` regions may be added by the host, and they may only be added prior to
    /// TVM finalization.
    ///
    /// a6 = 8
    TvmAddMemoryRegion {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = type of memory region
        region_type: TeeMemoryRegion,
        /// a2 = start of the region
        guest_addr: u64,
        /// a3 = length of the region
        len: u64,
    },
    /// Adds `num_pages` 4kB pages of confidential memory starting at `page_addr` to the page-table
    /// page pool for the specified guest.
    ///
    /// a6 = 9
    AddPageTablePages {
        /// a0 = guest_id
        guest_id: u64,
        /// a1 = address of the first page
        page_addr: u64,
        /// a2 = number of pages
        num_pages: u64,
    },
    /// Copies `num_pages` pages from non-confidential memory at `src_addr` to confidential
    /// memory at `dest_addr`, then measures and maps the pages at `dest_addr` into the specified
    /// guest's address space at `guest_addr`. The mapping must lie within a region of confidential
    /// memory created with `TvmAddMemoryRegion`. Measured pages may only be added prior to TVM
    /// finalization.
    ///
    /// a6 = 10
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
    /// Maps `num_pages` zero-filled pages of confidential memory starting at `page_addr` into the
    /// specified guest's address space at `guest_addr`. The mapping must lie within a region of
    /// confidential memory created with `TvmAddMemoryRegion`. Zero pages may only be added after
    /// the TVM has been finalized.
    ///
    /// a6 = 11
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
    /// Maps non-confidential shared pages in a region of shared memory previously registered by
    /// the guest via `AddMemoryRegion` in the TEE-Guest API.
    ///
    /// a6 = 12
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
    /// Returns the number of register sets in the vCPU shared-memory state area for vCPUs of
    /// `guest_id`.
    ///
    /// a6 = 13
    TvmCpuNumRegisterSets {
        /// a0 = guest id
        guest_id: u64,
    },
    /// Returns the `RegisterSetLocation` of the register set at `index` in the vCPU shared-memory
    /// state area for vCPUs of `guest_id`.
    ///
    /// The host calls this function for each `index` from 0 to the value returned by
    /// `TvmCpuNumRegisterSets` to enumerate the register sets in the vCPU shared-memory state
    /// area. From this enumeration process the caller discovers the size and layout of the
    /// structure that will be used to communicate vCPU state in shared-memory.
    ///
    /// a6 = 14
    TvmCpuGetRegisterSet {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = index of the register set
        index: u64,
    },
    /// Adds a vCPU with ID `vcpu_id` to the guest `guest_id`, registering `shared_page_addr` as
    /// the location of the shared-memory state area for this vCPU.
    ///
    /// `state_page_addr` must be page-aligned and point to a confidential memory region used to hold
    /// the TVM's vCPU state. Must be `TsmInfo::tvm_vcpu_state_pages` pages in length.
    ///
    /// `shared_page_addr` must be page-aligned and point to a sufficient number of non-confidential
    /// pages to hold a structure with the layout specified by the register set enumeration process
    /// described above. These pages are "pinned" in the non-confidential state (i.e. cannot be
    /// converted to confidential) until the TVM is destroyed.
    ///
    /// vCPUs may not be added after the TVM is finalized.
    ///
    /// a6 = 15
    TvmCpuCreate {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
        /// a2 = address of the first page donated for the vCPU state
        state_page_addr: u64,
        /// a3 = page address of shared state structure
        shared_page_addr: u64,
    },
    /// Runs the given vCPU in the TVM
    ///
    /// Returns 0 if the vCPU can be resumed via a subsequent call to `TvmCpuRun`, or a value other
    /// than 0 if the vCPU was terminated and is no longer runnable.
    ///
    /// Returns an error if:
    ///   - the specified TVM or vCPU does not exist
    ///   - the vCPU exists but is not currently runnable
    ///   - if hardware AIA virtualization is enabled for the TVM and the vCPU is not bound to
    ///     this physical CPU
    ///
    /// a6 = 16
    TvmCpuRun {
        /// a0 = guest id
        guest_id: u64,
        /// a1 = vCPU id
        vcpu_id: u64,
    },
    /// Initiates a TLB invalidation sequence for all pages that have been invalidated in the
    /// given TVM's address space since the previous call to `TvmInitiateFence`. The TLB
    /// invalidation sequence is completed when all vCPUs in the TVM that were running prior to
    /// to the call to `TvmInitiateFence` have taken a trap into the TSM, which the host can
    /// cause by IPI'ing the physical CPUs on which the TVM's vCPUs are running. An error is
    /// returned if a TLB invalidation sequence is already in progress for the TVM.
    ///
    /// a6 = 17
    TvmInitiateFence {
        /// a0 = guest id
        guest_id: u64,
    },
}

impl TeeHostFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use TeeHostFunction::*;
        match args[6] {
            0 => Ok(TsmGetInfo {
                dest_addr: args[0],
                len: args[1],
            }),
            1 => Ok(TsmConvertPages {
                page_addr: args[0],
                num_pages: args[1],
            }),
            2 => Ok(TsmReclaimPages {
                page_addr: args[0],
                num_pages: args[1],
            }),
            3 => Ok(TsmInitiateFence),
            4 => Ok(TsmLocalFence),
            5 => Ok(TvmCreate {
                params_addr: args[0],
                len: args[1],
            }),
            6 => Ok(Finalize { guest_id: args[0] }),
            7 => Ok(TvmDestroy { guest_id: args[0] }),
            8 => Ok(TvmAddMemoryRegion {
                guest_id: args[0],
                region_type: TeeMemoryRegion::from_reg(args[1])?,
                guest_addr: args[2],
                len: args[3],
            }),
            9 => Ok(AddPageTablePages {
                guest_id: args[0],
                page_addr: args[1],
                num_pages: args[2],
            }),
            10 => Ok(TvmAddMeasuredPages {
                guest_id: args[0],
                src_addr: args[1],
                dest_addr: args[2],
                page_type: TsmPageType::from_reg(args[3])?,
                num_pages: args[4],
                guest_addr: args[5],
            }),
            11 => Ok(TvmAddZeroPages {
                guest_id: args[0],
                page_addr: args[1],
                page_type: TsmPageType::from_reg(args[2])?,
                num_pages: args[3],
                guest_addr: args[4],
            }),
            12 => Ok(TvmAddSharedPages {
                guest_id: args[0],
                page_addr: args[1],
                page_type: TsmPageType::from_reg(args[2])?,
                num_pages: args[3],
                guest_addr: args[4],
            }),
            13 => Ok(TvmCpuNumRegisterSets { guest_id: args[0] }),
            14 => Ok(TvmCpuGetRegisterSet {
                guest_id: args[0],
                index: args[1],
            }),
            15 => Ok(TvmCpuCreate {
                guest_id: args[0],
                vcpu_id: args[1],
                state_page_addr: args[2],
                shared_page_addr: args[3],
            }),
            16 => Ok(TvmCpuRun {
                guest_id: args[0],
                vcpu_id: args[1],
            }),
            17 => Ok(TvmInitiateFence { guest_id: args[0] }),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for TeeHostFunction {
    fn a6(&self) -> u64 {
        use TeeHostFunction::*;
        match self {
            TsmGetInfo {
                dest_addr: _,
                len: _,
            } => 0,
            TsmConvertPages {
                page_addr: _,
                num_pages: _,
            } => 1,
            TsmReclaimPages {
                page_addr: _,
                num_pages: _,
            } => 2,
            TsmInitiateFence => 3,
            TsmLocalFence => 4,
            TvmCreate {
                params_addr: _,
                len: _,
            } => 5,
            Finalize { guest_id: _ } => 6,
            TvmDestroy { guest_id: _ } => 7,
            TvmAddMemoryRegion {
                guest_id: _,
                region_type: _,
                guest_addr: _,
                len: _,
            } => 8,
            AddPageTablePages {
                guest_id: _,
                page_addr: _,
                num_pages: _,
            } => 9,
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => 10,
            TvmAddZeroPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => 11,
            TvmAddSharedPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => 12,
            TvmCpuNumRegisterSets { guest_id: _ } => 13,
            TvmCpuGetRegisterSet {
                guest_id: _,
                index: _,
            } => 14,
            TvmCpuCreate {
                guest_id: _,
                vcpu_id: _,
                state_page_addr: _,
                shared_page_addr: _,
            } => 15,
            TvmCpuRun {
                guest_id: _,
                vcpu_id: _,
            } => 16,
            TvmInitiateFence { guest_id: _ } => 17,
        }
    }

    fn a0(&self) -> u64 {
        use TeeHostFunction::*;
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
                state_page_addr: _,
                shared_page_addr: _,
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
                num_pages: _,
            } => *page_addr,
            TsmReclaimPages {
                page_addr,
                num_pages: _,
            } => *page_addr,
            TvmAddMemoryRegion {
                guest_id,
                region_type: _,
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
            TvmCpuNumRegisterSets { guest_id } => *guest_id,
            TvmCpuGetRegisterSet { guest_id, index: _ } => *guest_id,
            TvmInitiateFence { guest_id } => *guest_id,
            _ => 0,
        }
    }

    fn a1(&self) -> u64 {
        use TeeHostFunction::*;
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
                state_page_addr: _,
                shared_page_addr: _,
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
                num_pages,
            } => *num_pages,
            TsmReclaimPages {
                page_addr: _,
                num_pages,
            } => *num_pages,
            TvmAddMemoryRegion {
                guest_id: _,
                region_type,
                guest_addr: _,
                len: _,
            } => *region_type as u64,
            TvmAddSharedPages {
                guest_id: _,
                page_addr,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *page_addr,
            TvmCpuGetRegisterSet { guest_id: _, index } => *index,
            _ => 0,
        }
    }

    fn a2(&self) -> u64 {
        use TeeHostFunction::*;
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
                state_page_addr,
                shared_page_addr: _,
            } => *state_page_addr,
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr,
                page_type: _,
                num_pages: _,
                guest_addr: _,
            } => *dest_addr,
            TvmAddMemoryRegion {
                guest_id: _,
                region_type: _,
                guest_addr,
                len: _,
            } => *guest_addr,
            TvmAddSharedPages {
                guest_id: _,
                page_addr: _,
                page_type,
                num_pages: _,
                guest_addr: _,
            } => *page_type as u64,
            _ => 0,
        }
    }

    fn a3(&self) -> u64 {
        use TeeHostFunction::*;
        match self {
            TvmAddZeroPages {
                guest_id: _,
                page_addr: _,
                page_type: _,
                num_pages,
                guest_addr: _,
            } => *num_pages,
            TvmCpuCreate {
                guest_id: _,
                vcpu_id: _,
                state_page_addr: _,
                shared_page_addr,
            } => *shared_page_addr,
            TvmAddMeasuredPages {
                guest_id: _,
                src_addr: _,
                dest_addr: _,
                page_type,
                num_pages: _,
                guest_addr: _,
            } => *page_type as u64,
            TvmAddMemoryRegion {
                guest_id: _,
                region_type: _,
                guest_addr: _,
                len,
            } => *len,
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
        use TeeHostFunction::*;
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
        use TeeHostFunction::*;
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
