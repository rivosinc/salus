// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::global_asm;
use core::{mem::size_of, ops::Deref, ops::DerefMut};
use drivers::{imsic::ImsicFileId, imsic::ImsicLocation, pmu, CpuId, CpuInfo};
use memoffset::offset_of;
use page_tracking::collections::PageVec;
use page_tracking::{PageTracker, TlbVersion};
use riscv_page_tables::GuestStagePagingMode;
use riscv_pages::{
    GuestPhysAddr, GuestVirtAddr, InternalClean, PageOwnerId, RawAddr, SequentialPages,
};
use riscv_regs::{hstatus, scounteren, sstatus, RiscvCsrInterface};
use riscv_regs::{
    Exception, FloatingPointRegisters, GeneralPurposeRegisters, GprIndex, LocalRegisterCopy,
    PrivilegeLevel, Readable, Trap, Writeable, CSR, CSR_CYCLE,
};
use sbi::{
    api::pmu::*, Error as SbiError, PmuCounterConfigFlags, PmuCounterStartFlags,
    PmuCounterStopFlags, PmuEventType, Result as SbiResult, SbiMessage, SbiReturnType,
    TvmMmioOpCode,
};
use spin::{Mutex, RwLock, RwLockReadGuard};

use crate::smp::PerCpu;
use crate::vm::MmioOperation;
use crate::vm_id::VmId;
use crate::vm_pages::{ActiveVmPages, VmPages};
use crate::{println, CONSOLE_DRIVER};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Error {
    BadCpuId,
    VmCpuExists,
    VmCpuNotFound,
    VmCpuRunning,
    VmCpuOff,
    VmCpuAlreadyPowered,
    InsufficientVmCpuStorage,
    WrongAddressSpace,
}

pub type Result<T> = core::result::Result<T, Error>;

/// The number of bytes required to hold the state of a single vCPU. We include the overhead for the
/// `PageVec<>` itself to ensure enough bytes are donated for it as well.
pub const VM_CPU_BYTES: u64 = (size_of::<VmCpusInner>() + size_of::<PageVec<VmCpusInner>>()) as u64;

/// Host GPR and CSR state which must be saved/restored when entering/exiting virtualization.
#[derive(Default)]
#[repr(C)]
struct HostCpuState {
    gprs: GeneralPurposeRegisters,
    sstatus: u64,
    hstatus: u64,
    scounteren: u64,
    stvec: u64,
    sscratch: u64,
}

/// Guest GPR and CSR state which must be saved/restored when exiting/entering virtualization.
#[derive(Default)]
#[repr(C)]
struct GuestCpuState {
    gprs: GeneralPurposeRegisters,
    fprs: FloatingPointRegisters,
    fcsr: u64,
    sstatus: u64,
    hstatus: u64,
    scounteren: u64,
    sepc: u64,
}

/// The CSRs that are only in effect when virtualization is enabled (V=1) and must be saved and
/// restored whenever we switch between VMs.
#[derive(Default)]
#[repr(C)]
struct GuestVCpuState {
    htimedelta: u64,
    vsstatus: u64,
    vsie: u64,
    vstvec: u64,
    vsscratch: u64,
    vsepc: u64,
    vscause: u64,
    vstval: u64,
    vsatp: u64,
    vstimecmp: u64,
}

/// CSRs written on an exit from virtualization that are used by the host to determine the cause of
/// the trap.
#[derive(Default, Clone)]
#[repr(C)]
pub struct VmCpuTrapState {
    pub scause: u64,
    pub stval: u64,
    pub htval: u64,
    pub htinst: u64,
}

#[derive(Default, Copy, Clone)]
#[allow(dead_code)]
struct CounterState {
    value: u64,
    config_flags: PmuCounterConfigFlags,
    event_type: PmuEventType,
    event_data: u64,
}

#[derive(Copy, Clone)]
enum PmuCounterState {
    NotConfigured,
    Configured(CounterState),
    Started(CounterState),
    Poisoned(CounterState),
}

impl Default for PmuCounterState {
    fn default() -> Self {
        Self::NotConfigured
    }
}

pub struct VmPmuState {
    // Stores information about the current state of PMU counters.
    counter_state: [PmuCounterState; drivers::pmu::MAX_HARDWARE_COUNTERS],
}

impl Default for VmPmuState {
    fn default() -> Self {
        Self {
            counter_state: [PmuCounterState::default(); drivers::pmu::MAX_HARDWARE_COUNTERS],
        }
    }
}

#[derive(Default, Copy, Clone)]
struct CounterMaskIter {
    counter_index: u64,
    counter_mask: u64,
    mask_index: u64,
}

impl CounterMaskIter {
    fn new(counter_index: u64, counter_mask: u64) -> Self {
        Self {
            counter_index,
            counter_mask,
            mask_index: 0,
        }
    }
}

// Convenience iterator to return counter_state indexes relative to counter_index
// if the counter_mask bit is set (assumes sanitized input).
impl Iterator for CounterMaskIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.counter_mask == 0 {
                break None;
            }
            let result = self.counter_index + self.mask_index;
            let mask_bit_set = self.counter_mask & (1 << self.mask_index);
            self.counter_mask &= !(1 << self.mask_index);
            self.mask_index += 1;
            if mask_bit_set != 0 {
                break Some(result as usize);
            }
        }
    }
}

impl VmPmuState {
    // Sets the bit to enable access to the CSR for counter_index
    fn set_hcounteren_bit(counter_index: u64) {
        // Unwrap ok: Guaranteed to succeed since we have already tested the condition in the call chain
        let pmu_info = pmu::PmuInfo::get().unwrap();
        let csr = pmu_info.counter_index_to_csr(counter_index).unwrap();
        CSR.hcounteren
            .read_and_set_bits(1 << (csr - CSR_CYCLE as u64));
    }

    // Clears the bit to enable access to the CSR for counter_index
    fn clear_hcounteren_bit(counter_index: u64) {
        // Unwrap ok: Guaranteed to succeed since we have already tested the condition in the call chain
        let pmu_info = pmu::PmuInfo::get().unwrap();
        let csr = pmu_info.counter_index_to_csr(counter_index).unwrap();
        CSR.hcounteren
            .read_and_clear_bits(1 << (csr - CSR_CYCLE as u64));
    }

    /// Updates internal state for PMU counters.
    /// This should be called following a successful SBI call to configure counters.
    pub fn update_configured_counter(
        &mut self,
        counter_index: u64,
        config_flags: PmuCounterConfigFlags,
        event_type: PmuEventType,
        event_data: u64,
    ) -> SbiResult<()> {
        use PmuCounterState::*;
        let new_state = CounterState {
            config_flags,
            event_type,
            event_data,
            value: 0,
        };
        let pmu_info = pmu::PmuInfo::get()?;
        // Ensure that platform configuration returned a valid counter index.
        pmu_info
            .get_counter_info(counter_index)
            .map_err(|_| SbiError::NotSupported)?;
        let state = &mut self.counter_state[counter_index as usize];
        match state {
            // If skip_match is set, counter must already be configured
            Configured(c) | Started(c) if config_flags.is_skip_match() => {
                if config_flags.is_auto_start() {
                    Self::set_hcounteren_bit(counter_index);
                    *state = Started(*c);
                }
                Ok(())
            }
            NotConfigured if !config_flags.is_skip_match() => {
                *state = if config_flags.is_auto_start() {
                    Self::set_hcounteren_bit(counter_index);
                    Started(new_state)
                } else {
                    Configured(new_state)
                };
                Ok(())
            }
            _ => Err(SbiError::InvalidParam),
        }
    }

    /// Updates internal state for PMU counters. Assumes a sanitized counter_index and counter_mask.
    /// This should be called following a successful SBI call to start counters.
    pub fn update_started_counters(&mut self, counter_index: u64, counter_mask: u64) {
        use PmuCounterState::*;
        let bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        for i in bitmask_iter {
            if let Configured(c) = self.counter_state[i] {
                self.counter_state[i] = Started(c);
                Self::set_hcounteren_bit(i as u64);
            }
        }
    }

    /// Updates internal state for PMU counters. Assumes a sanitized counter_index and counter_mask.
    /// This should be called following a successful SBI call to stop counters.
    pub fn update_stopped_counters(
        &mut self,
        counter_index: u64,
        counter_mask: u64,
        stop_flags: PmuCounterStopFlags,
    ) {
        use PmuCounterState::*;
        let bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        for i in bitmask_iter {
            let state = &mut self.counter_state[i];
            let is_started_counter = matches!(state, Started(_));
            match state {
                // Deliberately more permissive since the implementation permits
                // operations even on stopped counters (example: stop_flag_reset).
                Configured(c) | Started(c) => {
                    if stop_flags.is_reset_flag() {
                        Self::clear_hcounteren_bit(i as u64);
                        *state = NotConfigured;
                    } else {
                        if is_started_counter {
                            c.value = VmPmuState::read_counter_csr(i as u64);
                        }
                        *state = Configured(*c);
                    }
                }
                _ => {}
            }
        }
    }

    /// Returns a filtered counter_mask if the PMU counter range can be started.
    pub fn get_startable_counter_range(
        &self,
        counter_index: u64,
        counter_mask: u64,
    ) -> SbiResult<u64> {
        use PmuCounterState::*;
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        let mut bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
        bitmask_iter
            .find(|i| matches!(self.counter_state[*i], Configured(_)))
            .map_or_else(|| Err(SbiError::InvalidParam), |_| Ok(counter_mask))
    }

    /// Returns a filtered counter mask if the PMU counter range can be stopped.
    pub fn get_stoppable_counter_range(
        &self,
        counter_index: u64,
        counter_mask: u64,
    ) -> SbiResult<u64> {
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        // The current PMU driver attempts to stop counters before configuration since some platform
        // counters are automatically started. If we check for configured counters at this point,
        // the subsequent call to start counters will fail with an AlreadyStarted error.
        Ok(counter_mask)
    }

    /// Returns a filtered counter_mask if the PMU counter range can be configured.
    pub fn get_configurable_counter_range(
        &self,
        counter_index: u64,
        counter_mask: u64,
        config_flags: PmuCounterConfigFlags,
    ) -> SbiResult<u64> {
        use PmuCounterState::*;
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_mask = pmu_info.filter_counter_mask(counter_index, counter_mask)?;
        if !config_flags.is_skip_match() {
            let mut bitmask_iter = CounterMaskIter::new(counter_index, counter_mask);
            bitmask_iter
                .find(|i| matches!(self.counter_state[*i], NotConfigured))
                .map_or_else(|| Err(SbiError::InvalidParam), |_| Ok(counter_mask))
        } else {
            // If skip_match is set, the counter must already be configured
            let state = self
                .counter_state
                .get(counter_index as usize)
                .ok_or(SbiError::InvalidParam)?;
            if matches!(state, Started(_) | Configured(_)) {
                Ok(counter_mask)
            } else {
                Err(SbiError::InvalidParam)
            }
        }
    }

    fn read_counter_csr(counter_index: u64) -> u64 {
        // Unwrap ok: Guaranteed to succeed since we have already tested the condition in the call chain
        let pmu_info = pmu::PmuInfo::get().unwrap();
        // Unwrap ok: The CSR for a configured counter must be valid
        let csr = pmu_info.counter_index_to_csr(counter_index).unwrap();
        CSR.hpmcounter[(csr - CSR_CYCLE as u64) as usize].get_value()
    }

    /// Returns the cached value for a PMU CSR. We return 0 for counters that couldn't be
    /// configured or started on the resume path.
    pub fn get_cached_csr_value(&self, csr: u64) -> SbiResult<u64> {
        use PmuCounterState::*;
        let pmu_info = pmu::PmuInfo::get()?;
        let counter_index = pmu_info.csr_to_counter_index(csr)?;
        match self.counter_state[counter_index as usize] {
            Configured(c) => Ok(c.value),
            Poisoned(c) => Ok(c.value),
            _ => Err(SbiError::Failed),
        }
    }

    /// Saves the internal state for PMU counters. Stops started counters, and resets all configured counters.
    /// This should be called in anticipation of an outbound context switch.
    fn save_counters(&mut self) {
        use PmuCounterState::*;
        fn reset_all_counters(counter_mask: u64) -> SbiResult<()> {
            if counter_mask != 0 {
                stop_counters(
                    0,
                    counter_mask,
                    PmuCounterStopFlags::default().set_reset_flag(),
                )
                .or_else(|e| {
                    // Treat already stopped error as success
                    if matches!(e, SbiError::AlreadyStopped) {
                        Ok(())
                    } else {
                        Err(e)
                    }
                })
            } else {
                Ok(())
            }
        }

        if let Ok(pmu_info) = pmu::PmuInfo::get() {
            let num_counters = pmu_info.get_num_counters() as usize;
            let mut counter_mask = 0;
            for (i, state) in self.counter_state.iter_mut().take(num_counters).enumerate() {
                let include_counter = matches!(state, Configured(_) | Started(_));
                if include_counter {
                    counter_mask |= 1 << i;
                    if let Started(c) = state {
                        c.value = VmPmuState::read_counter_csr(i as u64);
                    }
                }
            }
            let result = reset_all_counters(counter_mask);
            if result.is_err() {
                println!(
                    "Warning: PMU failed to reset counters with mask {counter_mask:x}, {result:?}"
                );
            }
        }
    }

    /// Restores configured PMU counters, restarts started counters and enables CSR access as necessary.
    /// This should be called in anticipation of an inbound context switch.
    fn restore_counters(&mut self) {
        use PmuCounterState::*;
        fn resume_counter(counter_index: u64, c: &CounterState) -> SbiResult<()> {
            let start_flags = PmuCounterStartFlags::default().set_init_value();
            start_counters(counter_index, 0x1, start_flags, c.value)
                .map(|_| VmPmuState::set_hcounteren_bit(counter_index))
        }

        fn configure_counter(counter_index: u64, c: &CounterState) -> SbiResult<u64> {
            let config_flags = c
                .config_flags
                .unset_auto_start()
                .unset_skip_match()
                .unset_clear_value();
            configure_matching_counters(
                counter_index,
                0x1,
                config_flags,
                c.event_type,
                c.event_data,
            )
        }

        if let Ok(pmu_info) = pmu::PmuInfo::get() {
            let num_counters = pmu_info.get_num_counters() as usize;
            for (i, state) in self.counter_state.iter_mut().take(num_counters).enumerate() {
                let counter_index = i as u64;
                let is_started_counter = matches!(state, Started(_));
                match state {
                    Configured(c) | Started(c) => {
                        let result = configure_counter(counter_index, c).and_then(|_| {
                            if is_started_counter {
                                resume_counter(counter_index, c)
                            } else {
                                Ok(())
                            }
                        });
                        if result.is_err() {
                            *state = Poisoned(*c);
                            println!(
                                "Warning: Failed to restore counter {counter_index}, {result:?}"
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// (v)CPU register state that must be saved or restored when entering/exiting a VM or switching
/// between VMs.
#[derive(Default)]
#[repr(C)]
struct VmCpuState {
    // CPU state that's shared between our's and the guest's execution environment. Saved/restored
    // when entering/exiting a VM.
    host_regs: HostCpuState,
    guest_regs: GuestCpuState,

    // CPU state that only applies when V=1, e.g. the VS-level CSRs. Saved/restored on activation of
    // the vCPU.
    guest_vcpu_csrs: GuestVCpuState,

    // Read on VM exit.
    trap_csrs: VmCpuTrapState,
}

// The vCPU context switch, defined in guest.S
extern "C" {
    fn _run_guest(g: *mut VmCpuState);
}

#[allow(dead_code)]
const fn host_gpr_offset(index: GprIndex) -> usize {
    offset_of!(VmCpuState, host_regs)
        + offset_of!(HostCpuState, gprs)
        + (index as usize) * size_of::<u64>()
}

#[allow(dead_code)]
const fn guest_gpr_offset(index: GprIndex) -> usize {
    offset_of!(VmCpuState, guest_regs)
        + offset_of!(GuestCpuState, gprs)
        + (index as usize) * size_of::<u64>()
}

const fn guest_fpr_offset(index: usize) -> usize {
    offset_of!(VmCpuState, guest_regs) + offset_of!(GuestCpuState, fprs) + index * size_of::<u64>()
}

macro_rules! host_csr_offset {
    ($reg:tt) => {
        offset_of!(VmCpuState, host_regs) + offset_of!(HostCpuState, $reg)
    };
}

macro_rules! guest_csr_offset {
    ($reg:tt) => {
        offset_of!(VmCpuState, guest_regs) + offset_of!(GuestCpuState, $reg)
    };
}

global_asm!(
    include_str!("guest.S"),
    host_ra = const host_gpr_offset(GprIndex::RA),
    host_gp = const host_gpr_offset(GprIndex::GP),
    host_tp = const host_gpr_offset(GprIndex::TP),
    host_s0 = const host_gpr_offset(GprIndex::S0),
    host_s1 = const host_gpr_offset(GprIndex::S1),
    host_a1 = const host_gpr_offset(GprIndex::A1),
    host_a2 = const host_gpr_offset(GprIndex::A2),
    host_a3 = const host_gpr_offset(GprIndex::A3),
    host_a4 = const host_gpr_offset(GprIndex::A4),
    host_a5 = const host_gpr_offset(GprIndex::A5),
    host_a6 = const host_gpr_offset(GprIndex::A6),
    host_a7 = const host_gpr_offset(GprIndex::A7),
    host_s2 = const host_gpr_offset(GprIndex::S2),
    host_s3 = const host_gpr_offset(GprIndex::S3),
    host_s4 = const host_gpr_offset(GprIndex::S4),
    host_s5 = const host_gpr_offset(GprIndex::S5),
    host_s6 = const host_gpr_offset(GprIndex::S6),
    host_s7 = const host_gpr_offset(GprIndex::S7),
    host_s8 = const host_gpr_offset(GprIndex::S8),
    host_s9 = const host_gpr_offset(GprIndex::S9),
    host_s10 = const host_gpr_offset(GprIndex::S10),
    host_s11 = const host_gpr_offset(GprIndex::S11),
    host_sp = const host_gpr_offset(GprIndex::SP),
    host_sstatus = const host_csr_offset!(sstatus),
    host_hstatus = const host_csr_offset!(hstatus),
    host_scounteren = const host_csr_offset!(scounteren),
    host_stvec = const host_csr_offset!(stvec),
    host_sscratch = const host_csr_offset!(sscratch),
    guest_ra = const guest_gpr_offset(GprIndex::RA),
    guest_gp = const guest_gpr_offset(GprIndex::GP),
    guest_tp = const guest_gpr_offset(GprIndex::TP),
    guest_s0 = const guest_gpr_offset(GprIndex::S0),
    guest_s1 = const guest_gpr_offset(GprIndex::S1),
    guest_a0 = const guest_gpr_offset(GprIndex::A0),
    guest_a1 = const guest_gpr_offset(GprIndex::A1),
    guest_a2 = const guest_gpr_offset(GprIndex::A2),
    guest_a3 = const guest_gpr_offset(GprIndex::A3),
    guest_a4 = const guest_gpr_offset(GprIndex::A4),
    guest_a5 = const guest_gpr_offset(GprIndex::A5),
    guest_a6 = const guest_gpr_offset(GprIndex::A6),
    guest_a7 = const guest_gpr_offset(GprIndex::A7),
    guest_s2 = const guest_gpr_offset(GprIndex::S2),
    guest_s3 = const guest_gpr_offset(GprIndex::S3),
    guest_s4 = const guest_gpr_offset(GprIndex::S4),
    guest_s5 = const guest_gpr_offset(GprIndex::S5),
    guest_s6 = const guest_gpr_offset(GprIndex::S6),
    guest_s7 = const guest_gpr_offset(GprIndex::S7),
    guest_s8 = const guest_gpr_offset(GprIndex::S8),
    guest_s9 = const guest_gpr_offset(GprIndex::S9),
    guest_s10 = const guest_gpr_offset(GprIndex::S10),
    guest_s11 = const guest_gpr_offset(GprIndex::S11),
    guest_t0 = const guest_gpr_offset(GprIndex::T0),
    guest_t1 = const guest_gpr_offset(GprIndex::T1),
    guest_t2 = const guest_gpr_offset(GprIndex::T2),
    guest_t3 = const guest_gpr_offset(GprIndex::T3),
    guest_t4 = const guest_gpr_offset(GprIndex::T4),
    guest_t5 = const guest_gpr_offset(GprIndex::T5),
    guest_t6 = const guest_gpr_offset(GprIndex::T6),
    guest_sp = const guest_gpr_offset(GprIndex::SP),
    guest_f0 = const guest_fpr_offset(0),
    guest_f1 = const guest_fpr_offset(1),
    guest_f2 = const guest_fpr_offset(2),
    guest_f3 = const guest_fpr_offset(3),
    guest_f4 = const guest_fpr_offset(4),
    guest_f5 = const guest_fpr_offset(5),
    guest_f6 = const guest_fpr_offset(6),
    guest_f7 = const guest_fpr_offset(7),
    guest_f8 = const guest_fpr_offset(8),
    guest_f9 = const guest_fpr_offset(9),
    guest_f10 = const guest_fpr_offset(10),
    guest_f11 = const guest_fpr_offset(11),
    guest_f12 = const guest_fpr_offset(12),
    guest_f13 = const guest_fpr_offset(13),
    guest_f14 = const guest_fpr_offset(14),
    guest_f15 = const guest_fpr_offset(15),
    guest_f16 = const guest_fpr_offset(16),
    guest_f17 = const guest_fpr_offset(17),
    guest_f18 = const guest_fpr_offset(18),
    guest_f19 = const guest_fpr_offset(19),
    guest_f20 = const guest_fpr_offset(20),
    guest_f21 = const guest_fpr_offset(21),
    guest_f22 = const guest_fpr_offset(22),
    guest_f23 = const guest_fpr_offset(23),
    guest_f24 = const guest_fpr_offset(24),
    guest_f25 = const guest_fpr_offset(25),
    guest_f26 = const guest_fpr_offset(26),
    guest_f27 = const guest_fpr_offset(27),
    guest_f28 = const guest_fpr_offset(28),
    guest_f29 = const guest_fpr_offset(29),
    guest_f30 = const guest_fpr_offset(30),
    guest_f31 = const guest_fpr_offset(31),
    guest_fcsr = const guest_csr_offset!(fcsr),
    sstatus_fs_dirty = const sstatus::fs::Dirty.value,
    sstatus_fs_clean = const sstatus::fs::Clean.value,
    guest_sstatus = const guest_csr_offset!(sstatus),
    guest_hstatus = const guest_csr_offset!(hstatus),
    guest_scounteren = const guest_csr_offset!(scounteren),
    guest_sepc = const guest_csr_offset!(sepc),
);

/// Identifies the exit cause for a vCPU.
pub enum VmCpuExit {
    /// ECALLs from VS mode.
    Ecall(Option<SbiMessage>),
    /// G-stage page faults.
    PageFault {
        exception: Exception,
        fault_addr: GuestPhysAddr,
        fault_pc: GuestVirtAddr,
        priv_level: PrivilegeLevel,
    },
    /// Instruction emulation trap.
    VirtualInstruction {
        fault_pc: GuestVirtAddr,
        priv_level: PrivilegeLevel,
    },
    /// An exception which we expected to handle directly at VS, but trapped to HS instead.
    DelegatedException { exception: Exception, stval: u64 },
    /// Everything else that we currently don't or can't handle.
    Other(VmCpuTrapState),
    // TODO: Add other exit causes as needed.
}

// Interface for vCPU state save/restore on context switch.
trait VmCpuSaveState {
    // Saves this `ActiveVmCpu`'s state, effectively de-activating it until a corresponding call to
    // `restore()`.
    fn save(&mut self);

    // Re-activates this `ActiveVmCpu` by restoring the state that was saved with `save()`.
    fn restore(&mut self);
}

/// An activated vCPU. A vCPU in this state has entered the VM's address space and is ready to run.
pub struct ActiveVmCpu<'vcpu, 'pages, 'prev, T: GuestStagePagingMode> {
    vcpu: &'vcpu mut VmCpu,
    vm_pages: &'pages VmPages<T>,
    // `None` if this vCPU is itself running a child vCPU. Restored when the child vCPU exits.
    active_pages: Option<ActiveVmPages<'pages, T>>,
    // The parent vCPU which activated us.
    parent_vcpu: Option<&'prev mut dyn VmCpuSaveState>,
}

impl<'vcpu, 'pages, 'prev, T: GuestStagePagingMode> ActiveVmCpu<'vcpu, 'pages, 'prev, T> {
    // Restores and activates the vCPU state from `vcpu`, with the VM address space represented by
    // `vm_pages`.
    fn restore_from(
        vcpu: &'vcpu mut VmCpu,
        vm_pages: &'pages VmPages<T>,
        parent_vcpu: Option<&'prev mut ActiveVmCpu<T>>,
    ) -> Self {
        let this_cpu = PerCpu::this_cpu();
        if let Some(ref c) = vcpu.current_cpu && c.cpu != this_cpu.cpu_id() {
            // If we've changed CPUs, then any per-CPU state is invalid.
            //
            // TODO: Migration between physical CPUs needs to be done explicitly via TEECALL.
            vcpu.current_cpu = None;
        }

        // We store the parent vCPU as a trait object as a means of type-erasure for ActiveVmCpu's
        // inner lifetimes.
        let parent_vcpu = parent_vcpu.map(|p| p as &mut dyn VmCpuSaveState);

        let mut active_vcpu = Self {
            vcpu,
            vm_pages,
            active_pages: None,
            parent_vcpu,
        };
        active_vcpu.restore();
        active_vcpu
    }

    /// Runs this vCPU until it exits.
    pub fn run_to_exit(&mut self) -> VmCpuExit {
        self.complete_pending_mmio_op();

        // TODO, HGEIE programinng:
        //  - Track which guests the host wants interrupts from (by trapping HGEIE accesses from
        //    VS level) and update HGEIE[2:] appropriately.
        //  - If this is the host: clear HGEIE[1] on entry; inject SGEI into host VM if we receive
        //    any SGEI at HS level.
        //  - If this is a guest: set HGEIE[1] on entry; switch to the host VM for any SGEI that
        //    occur, injecting an SEI for the host interrupts and SGEI for guest VM interrupts.

        // TODO: Enforce that the vCPU has an assigned interrupt file before running.

        unsafe {
            // Safe to run the guest as it only touches memory assigned to it by being owned
            // by its page table.
            _run_guest(&mut self.state as *mut VmCpuState);
        }

        // Save off the trap information.
        self.state.trap_csrs.scause = CSR.scause.get();
        self.state.trap_csrs.stval = CSR.stval.get();
        self.state.trap_csrs.htval = CSR.htval.get();
        self.state.trap_csrs.htinst = CSR.htinst.get();

        // Determine the exit cause from the trap CSRs.
        use Exception::*;
        match Trap::from_scause(self.state.trap_csrs.scause).unwrap() {
            Trap::Exception(VirtualSupervisorEnvCall) => {
                let sbi_msg = SbiMessage::from_regs(self.state.guest_regs.gprs.a_regs()).ok();
                VmCpuExit::Ecall(sbi_msg)
            }
            Trap::Exception(GuestInstructionPageFault)
            | Trap::Exception(GuestLoadPageFault)
            | Trap::Exception(GuestStorePageFault) => {
                let fault_addr = RawAddr::guest(
                    self.state.trap_csrs.htval << 2 | self.state.trap_csrs.stval & 0x03,
                    self.guest_id,
                );
                VmCpuExit::PageFault {
                    exception: Exception::from_scause_reason(self.state.trap_csrs.scause).unwrap(),
                    fault_addr,
                    // Note that this address is not necessarily guest virtual as the guest may or
                    // may not have 1st-stage translation enabled in VSATP. We still use GuestVirtAddr
                    // here though to distinguish it from addresses (e.g. in HTVAL, or passed via a
                    // TEECALL) which are exclusively guest-physical. Furthermore we only access guest
                    // instructions via the HLVX instruction, which will take the VSATP translation
                    // mode into account.
                    fault_pc: RawAddr::guest_virt(self.state.guest_regs.sepc, self.guest_id),
                    priv_level: PrivilegeLevel::from_hstatus(self.state.guest_regs.hstatus),
                }
            }
            Trap::Exception(VirtualInstruction) => {
                VmCpuExit::VirtualInstruction {
                    // See above re: this address being guest virtual.
                    fault_pc: RawAddr::guest_virt(self.state.guest_regs.sepc, self.guest_id),
                    priv_level: PrivilegeLevel::from_hstatus(self.state.guest_regs.hstatus),
                }
            }
            Trap::Exception(e) => {
                if e.to_hedeleg_field()
                    .map_or(false, |f| CSR.hedeleg.matches_any(f))
                {
                    // Even if we intended to delegate this exception it might not be set in
                    // medeleg, in which case firmware may send it our way instead.
                    VmCpuExit::DelegatedException {
                        exception: e,
                        stval: self.state.trap_csrs.stval,
                    }
                } else {
                    VmCpuExit::Other(self.state.trap_csrs.clone())
                }
            }
            _ => VmCpuExit::Other(self.state.trap_csrs.clone()),
        }
    }

    /// Sets up access to the virtual MMIO registers for `mmio_op`.
    ///
    /// If `mmio_op` is a load instruction, writes to `MmioLoadValue` will be forwarded to the actual
    /// destination register the next time this `VmCpu` is run. If `mmio_op` is a store instruction,
    /// reads from `MmioStoreValue` will return the value from the source register of the store
    /// instruction until the next time this `VmCpu` is run.
    pub fn set_pending_mmio_op(&mut self, mmio_op: MmioOperation) {
        self.pending_mmio_op = Some(mmio_op);

        // Populate MmioStoreValue with whatever the VM was trying to store.
        use TvmMmioOpCode::*;
        let val = match mmio_op.opcode() {
            Store8 => self.get_gpr(mmio_op.register()) as u8 as u64,
            Store16 => self.get_gpr(mmio_op.register()) as u16 as u64,
            Store32 => self.get_gpr(mmio_op.register()) as u32 as u64,
            Store64 => self.get_gpr(mmio_op.register()),
            _ => 0,
        };
        self.set_virt_reg(VirtualRegister::MmioStore, val);
    }

    /// Delivers the given exception to the vCPU, setting up its register state to handle the trap
    /// the next time it is run.
    pub fn inject_exception(&mut self, exception: Exception, stval: u64) {
        // Update previous privelege level and interrupt state in VSSTATUS.
        let mut vsstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(CSR.vsstatus.get());
        if vsstatus.is_set(sstatus::sie) {
            vsstatus.modify(sstatus::spie.val(1));
        }
        vsstatus.modify(sstatus::sie.val(0));
        let sstatus =
            LocalRegisterCopy::<u64, sstatus::Register>::new(self.state.guest_regs.sstatus);
        vsstatus.modify(sstatus::spp.val(sstatus.read(sstatus::spp)));
        CSR.vsstatus.set(vsstatus.get());

        CSR.vscause.set(Trap::Exception(exception).to_scause());
        CSR.vstval.set(stval);
        CSR.vsepc.set(self.state.guest_regs.sepc);

        // Redirect the vCPU to its STVEC on entry.
        self.state.guest_regs.sepc = CSR.vstvec.get();
    }

    /// Increments the current `sepc` CSR value by `value`.
    pub fn inc_sepc(&mut self, value: u64) {
        self.state.guest_regs.sepc += value;
    }

    /// Increments SEPC and Updates A0/A1 with the result of an SBI call.
    pub fn set_ecall_result(&mut self, result: SbiReturnType) {
        self.inc_sepc(4); // ECALL is always a 4-byte instruction.
        match result {
            SbiReturnType::Legacy(a0) => {
                self.set_gpr(GprIndex::A0, a0);
            }
            SbiReturnType::Standard(ret) => {
                self.set_gpr(GprIndex::A0, ret.error_code as u64);
                self.set_gpr(GprIndex::A1, ret.return_value as u64);
            }
        }
    }

    /// Returns this active vCPU's `ActiveVmPages`.
    pub fn active_pages(&self) -> &ActiveVmPages<'pages, T> {
        // Unwrap ok since it's not possible to externally hold a reference to an `ActiveVmCpu` with
        // `self.active_pages` set to `None`. The only instance where `self.active_pages` is `None` is
        // when we're swapped out to run a child vCPU, in which case that child `ActiveVmCpu` holds
        // the exclusive reference to the parent `ActiveVmCpu`.
        self.active_pages.as_ref().unwrap()
    }

    /// Performs any pending TLB maintenance for this VM's address space.
    pub fn sync_tlb(&mut self) {
        // Exit and re-enter so that we pick up the new TLB version.
        self.active_pages = None;
        self.restore_vm_pages();
    }

    /// Completes any pending MMIO operation for this CPU.
    fn complete_pending_mmio_op(&mut self) {
        if let Some(mmio_op) = self.pending_mmio_op {
            let val = self.get_virt_reg(VirtualRegister::MmioLoad);
            use TvmMmioOpCode::*;
            // Complete any pending load operations.
            match mmio_op.opcode() {
                Load8 => {
                    self.set_gpr(mmio_op.register(), val as i8 as u64);
                }
                Load8U => {
                    self.set_gpr(mmio_op.register(), val as u8 as u64);
                }
                Load16 => {
                    self.set_gpr(mmio_op.register(), val as i16 as u64);
                }
                Load16U => {
                    self.set_gpr(mmio_op.register(), val as u16 as u64);
                }
                Load32 => {
                    self.set_gpr(mmio_op.register(), val as i32 as u64);
                }
                Load32U => {
                    self.set_gpr(mmio_op.register(), val as u32 as u64);
                }
                Load64 => {
                    self.set_gpr(mmio_op.register(), val);
                }
                _ => (),
            };

            self.pending_mmio_op = None;
            self.set_virt_reg(VirtualRegister::MmioLoad, 0);
            self.set_virt_reg(VirtualRegister::MmioStore, 0);

            // Advance SEPC past the faulting instruction.
            self.inc_sepc(mmio_op.len() as u64);
        }
    }

    // Saves the VS-level CSRs.
    fn save_vcpu_csrs(&mut self) {
        self.state.guest_vcpu_csrs.htimedelta = CSR.htimedelta.get();
        self.state.guest_vcpu_csrs.vsstatus = CSR.vsstatus.get();
        self.state.guest_vcpu_csrs.vsie = CSR.vsie.get();
        self.state.guest_vcpu_csrs.vstvec = CSR.vstvec.get();
        self.state.guest_vcpu_csrs.vsscratch = CSR.vsscratch.get();
        self.state.guest_vcpu_csrs.vsepc = CSR.vsepc.get();
        self.state.guest_vcpu_csrs.vscause = CSR.vscause.get();
        self.state.guest_vcpu_csrs.vstval = CSR.vstval.get();
        self.state.guest_vcpu_csrs.vsatp = CSR.vsatp.get();
        if CpuInfo::get().has_sstc() {
            self.state.guest_vcpu_csrs.vstimecmp = CSR.vstimecmp.get();
        }
    }

    // Restores the VS-level CSRs.
    fn restore_vcpu_csrs(&mut self) {
        // Safe as these don't take effect until V=1.
        CSR.htimedelta.set(self.state.guest_vcpu_csrs.htimedelta);
        CSR.vsstatus.set(self.state.guest_vcpu_csrs.vsstatus);
        CSR.vsie.set(self.state.guest_vcpu_csrs.vsie);
        CSR.vstvec.set(self.state.guest_vcpu_csrs.vstvec);
        CSR.vsscratch.set(self.state.guest_vcpu_csrs.vsscratch);
        CSR.vsepc.set(self.state.guest_vcpu_csrs.vsepc);
        CSR.vscause.set(self.state.guest_vcpu_csrs.vscause);
        CSR.vstval.set(self.state.guest_vcpu_csrs.vstval);
        CSR.vsatp.set(self.state.guest_vcpu_csrs.vsatp);
        if CpuInfo::get().has_sstc() {
            CSR.vstimecmp.set(self.state.guest_vcpu_csrs.vstimecmp);
        }
    }

    // Restores the VM's address space.
    fn restore_vm_pages(&mut self) {
        // Get the VMID to use for this VM's address space on this physical CPU.
        let this_cpu = PerCpu::this_cpu();
        let mut vmid_tracker = this_cpu.vmid_tracker_mut();
        // If VMIDs rolled over next_vmid() will do the necessary flush and the previous TLB version
        // doesn't matter.
        let (vmid, prev_tlb_version) = match self.current_cpu {
            Some(ref c) if c.vmid.version() == vmid_tracker.current_version() => {
                (c.vmid, Some(c.tlb_version))
            }
            _ => (vmid_tracker.next_vmid(), None),
        };

        // enter_with_vmid() will fence if prev_tlb_version is stale.
        let active_pages = self.vm_pages.enter_with_vmid(vmid, prev_tlb_version);

        // Update our per-CPU state in case VMID or TLB version changed.
        if let Some(ref mut c) = self.current_cpu {
            c.vmid = vmid;
            c.tlb_version = active_pages.tlb_version();
        } else {
            self.current_cpu = Some(CurrentCpu {
                cpu: this_cpu.cpu_id(),
                vmid,
                tlb_version: active_pages.tlb_version(),
            });
        }

        self.active_pages = Some(active_pages);
    }
}

impl<T: GuestStagePagingMode> VmCpuSaveState for ActiveVmCpu<'_, '_, '_, T> {
    fn save(&mut self) {
        self.active_pages = None;
        self.save_vcpu_csrs();
        self.pmu_state.save_counters();
    }

    fn restore(&mut self) {
        self.restore_vcpu_csrs();
        self.restore_vm_pages();
        self.pmu_state.restore_counters();
    }
}

impl<T: GuestStagePagingMode> Deref for ActiveVmCpu<'_, '_, '_, T> {
    type Target = VmCpu;

    fn deref(&self) -> &VmCpu {
        self.vcpu
    }
}

impl<T: GuestStagePagingMode> DerefMut for ActiveVmCpu<'_, '_, '_, T> {
    fn deref_mut(&mut self) -> &mut VmCpu {
        self.vcpu
    }
}

impl<T: GuestStagePagingMode> Drop for ActiveVmCpu<'_, '_, '_, T> {
    fn drop(&mut self) {
        self.save();
        if let Some(ref mut p) = self.parent_vcpu {
            p.restore();
        }
    }
}

/// Used to store any per-physical-CPU state for a virtual CPU of a VM.
struct CurrentCpu {
    cpu: CpuId,
    vmid: VmId,
    tlb_version: TlbVersion,
}

/// Virtual CPU registers that are used to store vCPU state accessible to the VM's host without
/// giving the host access to internal register state.
pub enum VirtualRegister {
    /// 1st detailed exit cause register. Usage depends on the exit code.
    Cause0,
    /// 2nd detailed exit cause register. Usage depends on the exit code.
    Cause1,
    /// xxx
    MmioLoad,
    /// xxx
    MmioStore,
}

/// Virtual register state of a vCPU.
#[derive(Default)]
struct VirtualRegisters {
    cause0: u64,
    cause1: u64,
    mmio_load: u64,
    mmio_store: u64,
}

/// Represents a single virtual CPU of a VM.
pub struct VmCpu {
    state: VmCpuState,
    virt_regs: VirtualRegisters,
    imsic_location: Option<ImsicLocation>,
    pmu_state: VmPmuState,
    current_cpu: Option<CurrentCpu>,
    // TODO: interrupt_file should really be part of CurrentCpu, but we have no way to migrate it
    // at present.
    interrupt_file: Option<ImsicFileId>,
    pending_mmio_op: Option<MmioOperation>,
    guest_id: PageOwnerId,
}

impl VmCpu {
    /// Creates a new vCPU.
    pub fn new(guest_id: PageOwnerId) -> Self {
        let mut state = VmCpuState::default();

        let mut hstatus = LocalRegisterCopy::<u64, hstatus::Register>::new(0);
        hstatus.modify(hstatus::spv.val(1));
        hstatus.modify(hstatus::spvp::Supervisor);
        if !guest_id.is_host() {
            // Trap on WFI for non-host VMs. Trapping WFI for the host is pointless since all we'd do
            // in the hypervisor is WFI ourselves.
            hstatus.modify(hstatus::vtw.val(1));
        }
        state.guest_regs.hstatus = hstatus.get();

        let mut sstatus = LocalRegisterCopy::<u64, sstatus::Register>::new(0);
        sstatus.modify(sstatus::spp::Supervisor);
        sstatus.modify(sstatus::fs::Initial);
        state.guest_regs.sstatus = sstatus.get();

        let mut scounteren = LocalRegisterCopy::<u64, scounteren::Register>::new(0);
        scounteren.modify(scounteren::cycle.val(1));
        scounteren.modify(scounteren::time.val(1));
        scounteren.modify(scounteren::instret.val(1));
        state.guest_regs.scounteren = scounteren.get();

        Self {
            state,
            virt_regs: VirtualRegisters::default(),
            imsic_location: None,
            current_cpu: None,
            pending_mmio_op: None,
            interrupt_file: None,
            guest_id,
            pmu_state: VmPmuState::default(),
        }
    }

    /// Sets the `sepc` CSR, or the PC value the vCPU will jump to when it is run.
    pub fn set_sepc(&mut self, sepc: u64) {
        self.state.guest_regs.sepc = sepc;
    }

    /// Gets the current `sepc` CSR value of the vCPU.
    pub fn get_sepc(&mut self) -> u64 {
        self.state.guest_regs.sepc
    }

    /// Sets one of the vCPU's general-purpose registers.
    pub fn set_gpr(&mut self, gpr: GprIndex, value: u64) {
        self.state.guest_regs.gprs.set_reg(gpr, value);
    }

    /// Gets one of the vCPU's general purpose registers.
    pub fn get_gpr(&mut self, gpr: GprIndex) -> u64 {
        self.state.guest_regs.gprs.reg(gpr)
    }

    /// Set one of the vCPU's virtual registers.
    pub fn set_virt_reg(&mut self, reg: VirtualRegister, value: u64) {
        use VirtualRegister::*;
        match reg {
            Cause0 => {
                self.virt_regs.cause0 = value;
            }
            Cause1 => {
                self.virt_regs.cause1 = value;
            }
            MmioLoad => {
                self.virt_regs.mmio_load = value;
            }
            MmioStore => {
                self.virt_regs.mmio_store = value;
            }
        }
    }

    /// Gets one of the vCPU's virtual registers.
    pub fn get_virt_reg(&mut self, reg: VirtualRegister) -> u64 {
        use VirtualRegister::*;
        match reg {
            Cause0 => self.virt_regs.cause0,
            Cause1 => self.virt_regs.cause1,
            MmioLoad => self.virt_regs.mmio_load,
            MmioStore => self.virt_regs.mmio_store,
        }
    }

    /// Sets the location of this vCPU's virtualized IMSIC.
    pub fn set_imsic_location(&mut self, imsic_location: ImsicLocation) {
        self.imsic_location = Some(imsic_location);
    }

    /// Returns the location of this vCPU's virtualized IMSIC.
    pub fn get_imsic_location(&self) -> Option<ImsicLocation> {
        self.imsic_location
    }

    /// Sets the interrupt file for this vCPU.
    pub fn set_interrupt_file(&mut self, interrupt_file: ImsicFileId) {
        self.interrupt_file = Some(interrupt_file);

        // Update VGEIN so that the selected interrupt file gets used next time the vCPU is run.
        let mut hstatus =
            LocalRegisterCopy::<u64, hstatus::Register>::new(self.state.guest_regs.hstatus);
        hstatus.modify(hstatus::vgein.val(interrupt_file.bits() as u64));
        self.state.guest_regs.hstatus = hstatus.get();
    }

    /// Activates `vcpu` with the VM address space in `vm_pages`, returning a reference to it as an
    /// `ActiveVmCpu`. If `parent_vcpu` is not `None`, its state is saved before this vCPU is
    /// activated and is restored when the returned `ActiveVmCpu` is dropped.
    pub fn activate<'vcpu, 'pages, 'prev: 'vcpu + 'pages, T: GuestStagePagingMode>(
        &'vcpu mut self,
        vm_pages: &'pages VmPages<T>,
        mut parent_vcpu: Option<&'prev mut ActiveVmCpu<T>>,
    ) -> Result<ActiveVmCpu<'vcpu, 'pages, 'prev, T>> {
        if self.guest_id != vm_pages.page_owner_id() {
            return Err(Error::WrongAddressSpace);
        }

        if let Some(ref mut p) = parent_vcpu {
            p.save();
        }

        Ok(ActiveVmCpu::restore_from(self, vm_pages, parent_vcpu))
    }

    pub fn pmu(&mut self) -> &mut VmPmuState {
        &mut self.pmu_state
    }
}

/// Represents the state of a vCPU in a VM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmCpuStatus {
    /// There is no vCPU with this ID in the VM.
    NotPresent,
    /// The vCPU is present, but not powered on.
    PoweredOff,
    /// The vCPU is available to be run.
    Runnable,
    /// The vCPU has been claimed exclusively for running on a (physical) CPU.
    Running,
}

struct VmCpusInner {
    // Locking: status must be locked before vcpu.
    status: RwLock<VmCpuStatus>,
    vcpu: Mutex<VmCpu>,
}

/// A reference to an "Available" (idle) `VmCpu`. The `VmCpu` is guaranteed not to change states
/// while this reference is held.
pub struct IdleVmCpu<'a> {
    _status: RwLockReadGuard<'a, VmCpuStatus>,
    vcpu: &'a Mutex<VmCpu>,
}

impl<'a> Deref for IdleVmCpu<'a> {
    type Target = Mutex<VmCpu>;

    fn deref(&self) -> &Mutex<VmCpu> {
        self.vcpu
    }
}

/// A reference to an exclusively-owned `VmCpu` in the "Running" state. The `VmCpu` transitions
/// back to idle when this reference is dropped.
pub struct RunningVmCpu<'a> {
    parent: &'a VmCpus,
    vcpu: &'a Mutex<VmCpu>,
    id: u64,
    power_off: bool,
}

impl<'a> RunningVmCpu<'a> {
    /// Mark this vCPU as powered off when it is returned.
    pub fn power_off(&mut self) {
        self.power_off = true;
    }
}

impl<'a> Deref for RunningVmCpu<'a> {
    type Target = Mutex<VmCpu>;

    fn deref(&self) -> &Mutex<VmCpu> {
        self.vcpu
    }
}

impl<'a> Drop for RunningVmCpu<'a> {
    fn drop(&mut self) {
        let entry = self.parent.inner.get(self.id as usize).unwrap();
        let mut status = entry.status.write();
        assert_eq!(*status, VmCpuStatus::Running);
        *status = if self.power_off {
            VmCpuStatus::PoweredOff
        } else {
            VmCpuStatus::Runnable
        };
    }
}

/// The set of vCPUs in a VM.
pub struct VmCpus {
    inner: PageVec<VmCpusInner>,
}

impl VmCpus {
    /// Creates a new vCPU tracking structure backed by `pages`.
    pub fn new(
        guest_id: PageOwnerId,
        pages: SequentialPages<InternalClean>,
        page_tracker: PageTracker,
    ) -> Result<Self> {
        let num_vcpus = pages.length_bytes() / VM_CPU_BYTES;
        if num_vcpus == 0 {
            return Err(Error::InsufficientVmCpuStorage);
        }
        let mut inner = PageVec::new(pages, page_tracker);
        for _ in 0..num_vcpus {
            let entry = VmCpusInner {
                status: RwLock::new(VmCpuStatus::NotPresent),
                vcpu: Mutex::new(VmCpu::new(guest_id)),
            };
            inner.push(entry);
        }
        Ok(Self { inner })
    }

    /// Adds the vCPU at `vcpu_id` as an available vCPU, returning a reference to it.
    pub fn add_vcpu(&self, vcpu_id: u64) -> Result<IdleVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let mut status = entry.status.write();
        if *status != VmCpuStatus::NotPresent {
            return Err(Error::VmCpuExists);
        }
        *status = VmCpuStatus::PoweredOff;
        Ok(IdleVmCpu {
            _status: status.downgrade(),
            vcpu: &entry.vcpu,
        })
    }

    /// Returns a reference to the vCPU with `vcpu_id` if it exists and is not currently running.
    /// The returned `IdleVmCpu` is guaranteed not to change state until it is dropped.
    pub fn get_vcpu(&self, vcpu_id: u64) -> Result<IdleVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let status = entry.status.read();
        match *status {
            VmCpuStatus::PoweredOff | VmCpuStatus::Runnable => Ok(IdleVmCpu {
                _status: status,
                vcpu: &entry.vcpu,
            }),
            VmCpuStatus::Running => Err(Error::VmCpuRunning),
            VmCpuStatus::NotPresent => Err(Error::VmCpuNotFound),
        }
    }

    /// Marks the vCPU with `vcpu_id` as powered on and runnable and returns a reference to it.
    /// The returned `IdleVmCpu` is guaranteed not to change state until it is dropped.
    pub fn power_on_vcpu(&self, vcpu_id: u64) -> Result<IdleVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let mut status = entry.status.write();
        match *status {
            VmCpuStatus::PoweredOff => {
                *status = VmCpuStatus::Runnable;
                Ok(IdleVmCpu {
                    _status: status.downgrade(),
                    vcpu: &entry.vcpu,
                })
            }
            VmCpuStatus::Running | VmCpuStatus::Runnable => Err(Error::VmCpuAlreadyPowered),
            VmCpuStatus::NotPresent => Err(Error::VmCpuNotFound),
        }
    }

    /// Takes exclusive ownership of the vCPU with `vcpu_id`, marking it as running. The vCPU is
    /// returned to the "Available" state when the returned `RunningVmCpu` is dropped.
    pub fn take_vcpu(&self, vcpu_id: u64) -> Result<RunningVmCpu> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        let mut status = entry.status.write();
        match *status {
            VmCpuStatus::Runnable => {
                *status = VmCpuStatus::Running;
                Ok(RunningVmCpu {
                    parent: self,
                    vcpu: &entry.vcpu,
                    id: vcpu_id,
                    power_off: false,
                })
            }
            VmCpuStatus::Running => Err(Error::VmCpuRunning),
            VmCpuStatus::PoweredOff => Err(Error::VmCpuOff),
            VmCpuStatus::NotPresent => Err(Error::VmCpuNotFound),
        }
    }

    /// Returns the status of the specified vCPU.
    pub fn get_vcpu_status(&self, vcpu_id: u64) -> Result<VmCpuStatus> {
        let entry = self.inner.get(vcpu_id as usize).ok_or(Error::BadCpuId)?;
        Ok(*entry.status.read())
    }
}
