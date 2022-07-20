// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Functions for the Performance Monitoring Unit (PMU) extension
/// Specific details can be found in the SBI documentation for the PMU extension.
#[derive(Copy, Clone)]
pub enum PmuFunction {
    /// Returns the total number of performance counters (hardware and firmware).
    GetNumCounters,
    /// Returns information about hardware counter specified by the inner value.
    GetCounterInfo(u64),
    /// Configures the counters selected by counter_index_base and counter_index_mask.
    /// See the sbi_pmu_counter_config_matching documentation for details.
    ConfigureMatchingCounters {
        /// Counter index base.
        counter_index_base: u64,
        /// Counter index mask.
        counter_index_mask: u64,
        /// Counter configuration flags.
        config_flags: PmuCounterConfigFlags,
        /// Counter event type.
        event_type: PmuEventType,
        /// Counter event data.
        event_data: u64,
    },
    /// Starts the counters selected by counter_index_base and counter_index_mask.
    /// See the sbi_pmu_counter_start documentation for details.
    StartCounters {
        /// Counter index base.
        counter_index_base: u64,
        /// Counter index mask.
        counter_index_mask: u64,
        /// Counter start flags.
        start_flags: PmuCounterStartFlags,
        /// Counter initial value (used in conjunction with start_flags).
        initial_value: u64,
    },
    /// Stops the counters selected by counter_index and counter_index_mask.
    /// See the sbi_pmu_counter_stop documentation for details.
    StopCounters {
        /// Counter index base.
        counter_index_base: u64,
        /// Counter index mask.
        counter_index_mask: u64,
        /// Counter stop flags.
        stop_flags: PmuCounterStopFlags,
    },
    /// Returns the current value firmware counter specified by the inner value.
    ReadFirmwareCounter(u64),
}

/// This encapsulates the bit-fields for PMU config_flags parameter as described in the SBI documentation
/// for sbi_pmu_counter_config_matching
#[derive(Copy, Clone, Default)]
pub struct PmuCounterConfigFlags(u64);

impl PmuCounterConfigFlags {
    /// Constructs a new PmuCounterConfigFlags from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        const CONFIG_FLAG_INVERSE_MASK: u64 = !0xEF;
        if value & CONFIG_FLAG_INVERSE_MASK == 0 {
            Ok(PmuCounterConfigFlags(value))
        } else {
            Err(Error::InvalidParam)
        }
    }

    /// Returns the raw inner value.
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// Sets the skip_match bit-flag (skips counter matching).
    pub fn set_skip_match(self) -> Self {
        PmuCounterConfigFlags(self.0 | 1)
    }

    /// Returns if the skip_match bit-flag is set.
    pub fn is_skip_match_set(&self) -> bool {
        self.0 & 1 != 0
    }

    /// Sets the clear_value bit-flag (clears the counter value).
    pub fn set_clear_value(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 1))
    }

    /// Returns if the clear_value bit-flag is set.
    pub fn is_clear_value_set(&self) -> bool {
        self.0 & (1 << 1) != 0
    }

    /// Sets the auto_start bit-flag (automatically starts the counter).
    pub fn set_auto_start(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 2))
    }

    /// Returns if the auto_start bit-flag is set.
    pub fn is_auto_start_set(&self) -> bool {
        self.0 & (1 << 2) != 0
    }

    /// Sets the vuinh bit-flag (inhibit counter in VU-mode).
    pub fn set_vuinh(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 3))
    }

    /// Returns if the vuinh bit-flag is set.
    pub fn get_vuinh(&self) -> bool {
        self.0 & (1 << 3) != 0
    }

    /// Sets the vsinh bit-flag (inhibit counter in VS-mode).
    pub fn set_vsinh(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 4))
    }

    /// Returns if the vsinh bit-flag is set.
    pub fn is_get_vsinh_set(&self) -> bool {
        self.0 & (1 << 4) != 0
    }

    /// Sets the uinh bit-flag (inhibit counter in U-mode).
    pub fn set_uinh(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 5))
    }

    /// Returns if the uinh bit-flag is set.
    pub fn is_get_uinh_set(&self) -> bool {
        self.0 & (1 << 5) != 0
    }

    /// Sets the sinh bit-flag (inhibit counter in S-mode).
    pub fn set_sinh(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 6))
    }

    /// Returns if the sinh bit-flag is set.
    pub fn is_get_sinh_set(&self) -> bool {
        self.0 & (1 << 6) != 0
    }

    /// Sets the minh bit-flag (inhibit counter in M-mode).
    pub fn set_minh(self) -> Self {
        PmuCounterConfigFlags(self.0 | (1 << 7))
    }

    /// Returns if the minh bit-flag is set.
    pub fn is_get_minh_set(&self) -> bool {
        self.0 & (1 << 7) != 0
    }
}

/// This encapsulates the bit-fields for PMU start_flags parameter as described in the SBI documentation
/// for sbi_pmu_counter_start
#[derive(Copy, Clone, Debug, Default)]
pub struct PmuCounterStartFlags(u64);

impl PmuCounterStartFlags {
    /// Constructs a new PmuCounterStartFlags from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        match value {
            0 | 1 => Ok(PmuCounterStartFlags(value)),
            _ => Err(Error::InvalidParam),
        }
    }

    /// Returns the raw value of the inner bit-flag field.
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// Sets the set_init_value bit-flag (set initial counter value).
    pub fn set_start_set_init_value(self) -> Self {
        PmuCounterStartFlags(self.0 | 1)
    }

    /// Returns if set_init_value bit-flag is set.
    pub fn is_start_set_init_value_set(&self) -> bool {
        self.0 & 1 != 0
    }
}

/// This encapsulates the bit-fields for PMU stop_flags parameter as described in the SBI documentation
/// for sbi_pmu_counter_stop
#[derive(Copy, Clone, Debug, Default)]
pub struct PmuCounterStopFlags(u64);

impl PmuCounterStopFlags {
    /// Constructs a new PmuCounterStopFlags from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        match value {
            0 | 1 => Ok(PmuCounterStopFlags(value)),
            _ => Err(Error::InvalidParam),
        }
    }

    /// Returns the raw value of the inner bit-flag field.
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// Sets the stop_reset bit-flag (resets the counter after stopping).
    pub fn set_stop_flag_reset(self) -> Self {
        PmuCounterStopFlags(self.0 | 1)
    }

    /// Returns if the stop_reset bit flag is set.
    pub fn is_stop_flag_reset_set(&self) -> bool {
        self.0 & 1 != 0
    }
}

/// This encapsulates the counter information returned by the call to sbi_pmu_counter_get_info.
#[derive(Copy, Clone, Debug)]
pub struct PmuCounterInfo(u64);

impl PmuCounterInfo {
    /// Constructs a PmuCounterInfo from the passed in value.
    pub fn new(value: u64) -> Self {
        PmuCounterInfo(value)
    }

    /// Returns the inner value.
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// Returns if the counter is a hardware counter.
    pub fn is_hardware_counter(&self) -> bool {
        self.0 & (1 << 63) == 0
    }

    /// Returns if the counter is a firmware counter.
    pub fn is_firmware_counter(&self) -> bool {
        !self.is_hardware_counter()
    }

    /// Returns the 12-bit CSR number associated with the counter.
    pub fn get_csr(&self) -> u64 {
        self.0 & 0b1111_1111_1111_u64
    }

    /// Returns the counter width (one less than number of CSR-bits).
    pub fn get_counter_width(&self) -> u64 {
        (self.0 >> 12) & 0b1111_11
    }
}

#[derive(Copy, Clone, Debug)]
/// Enumeration of the event types.
pub enum PmuEventType {
    /// Represents the hardware general events (type #0) in the SBI documentation.
    Hardware(PmuHardware),
    /// Represents the hardware cache events (type #1) in the SBI documentation.
    Cache(PmuHwCacheParams),
    /// Represents the firmware events (type #15) in the SBI documentation.
    Firmware(PmuFirmware),
}

impl PmuEventType {
    /// Returns the encoded representation of the type.
    pub fn raw(&self) -> u64 {
        const EVENT_TYPE_SHIFT: u64 = 16;
        const HARDWARE_CACHE_EVENT_TYPE: u64 = 1;
        const FIRMWARE_EVENT_TYPE: u64 = 0xF;
        use PmuEventType::*;
        match self {
            Hardware(p) => p.raw(),
            Cache(p) => p.raw() | (HARDWARE_CACHE_EVENT_TYPE << EVENT_TYPE_SHIFT),
            Firmware(p) => p.raw() | (FIRMWARE_EVENT_TYPE << EVENT_TYPE_SHIFT),
        }
    }

    /// Constructs PmuEventType from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        use PmuEventType::*;
        const EVENT_TYPE_SHIFT: u64 = 16;
        const EVENT_TYPE_INVERSE_MASK: u64 = !(0xF << EVENT_TYPE_SHIFT);
        match value >> EVENT_TYPE_SHIFT {
            0 => Ok(Hardware(PmuHardware::from_regs(
                value & EVENT_TYPE_INVERSE_MASK,
            )?)),
            1 => Ok(Cache(PmuHwCacheParams::from_regs(
                value & EVENT_TYPE_INVERSE_MASK,
            )?)),
            0xF => Ok(Firmware(PmuFirmware::from_regs(
                value & EVENT_TYPE_INVERSE_MASK,
            )?)),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Enumeration of the hardware event types.
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
pub enum PmuHardware {
    /// Identifier for CPU cycle events.
    CpuCycles = 1,
    /// Identifier for retired instruction events.
    Instructions = 2,
    /// Identifier for cache hit events.
    CacheReferences = 3,
    /// Identifier for cache miss events.
    CacheMisses = 4,
    /// Identifier for branch instruction events.
    BranchInstructions = 5,
    /// Identifier for branch misprediction events.
    BranchMisses = 6,
    /// Identifier for bus cycle events.
    BusCycles = 7,
    /// Identifier for stalled front-end cycle events.
    StalledCyclesFrontEnd = 8,
    /// Identifier for stalled back-end cycle events.
    StalledCyclesBackEnd = 9,
    /// Identifier for reference CPU cycle  events.
    ReferenceCpuCycles = 10,
}

impl PmuHardware {
    /// Returns the raw value for the type.
    pub fn raw(&self) -> u64 {
        match self {
            PmuHardware::CpuCycles => 1,
            PmuHardware::Instructions => 2,
            PmuHardware::CacheReferences => 3,
            PmuHardware::CacheMisses => 4,
            PmuHardware::BranchInstructions => 5,
            PmuHardware::BranchMisses => 6,
            PmuHardware::BusCycles => 7,
            PmuHardware::StalledCyclesFrontEnd => 8,
            PmuHardware::StalledCyclesBackEnd => 9,
            PmuHardware::ReferenceCpuCycles => 10,
        }
    }

    /// Constructs PmuHardware from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        use PmuHardware::*;
        match value {
            0 => Ok(CpuCycles),
            1 => Ok(Instructions),
            2 => Ok(CacheReferences),
            4 => Ok(CacheMisses),
            5 => Ok(BranchInstructions),
            6 => Ok(BranchMisses),
            7 => Ok(BusCycles),
            8 => Ok(StalledCyclesFrontEnd),
            9 => Ok(StalledCyclesBackEnd),
            10 => Ok(ReferenceCpuCycles),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Enumeration of cache event types (for use with PmuHardware of type CacheReferences/CacheMisses).
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
pub enum PmuHwCache {
    /// Identifier for first level data cache.
    L1DataCache = 0,
    /// Identifier for first level instruction cache.
    L1InstructionCache = 1,
    /// Identifier for last level cache.
    LlCache = 2,
    /// Identifier for data TLB cache.
    DataTlbCache = 3,
    /// Identifier for instruction TLB cache.
    InstructionTlbCache = 4,
    /// Identifier for branch predictor cache.
    BranchPredictorCache = 5,
    /// Identifier for NUMA node cache.
    NumaNodeCache = 6,
}

impl PmuHwCache {
    /// Returns the raw value for the type.
    pub fn raw(&self) -> u64 {
        use PmuHwCache::*;
        match self {
            L1DataCache => 0,
            L1InstructionCache => 1,
            LlCache => 2,
            DataTlbCache => 3,
            InstructionTlbCache => 4,
            BranchPredictorCache => 5,
            NumaNodeCache => 6,
        }
    }

    /// Constructs PmuHwCache from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        use PmuHwCache::*;
        match value {
            0 => Ok(L1DataCache),
            1 => Ok(L1InstructionCache),
            2 => Ok(LlCache),
            3 => Ok(DataTlbCache),
            4 => Ok(InstructionTlbCache),
            5 => Ok(BranchPredictorCache),
            6 => Ok(NumaNodeCache),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Enumeration of cache op_ids (for use with PmuHardware of type CacheReferences/CacheMisses).
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
pub enum PmuHwCacheOpId {
    /// Identifier for a cache read op_id.
    Read = 0,
    /// Identifier for a cache write op_id.
    Write = 1,
    /// Identifier for a cache prefetch op_id.
    Prefetch = 2,
}

impl PmuHwCacheOpId {
    /// Returns the raw value.
    pub fn raw(&self) -> u64 {
        use PmuHwCacheOpId::*;
        match self {
            Read => 0,
            Write => 1,
            Prefetch => 2,
        }
    }

    /// Constructs PmuHwCacheOpId from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        use PmuHwCacheOpId::*;
        match value {
            0 => Ok(Read),
            1 => Ok(Write),
            2 => Ok(Prefetch),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Enumeration of results returned by cache counter reads.
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
pub enum PmuHwCacheResultId {
    /// Cache miss.
    CacheMiss = 0,
    /// Cache hit.
    CacheHit = 1,
}

impl PmuHwCacheResultId {
    /// Returns the raw value.
    pub fn raw(&self) -> u64 {
        use PmuHwCacheResultId::*;
        match self {
            CacheMiss => 0,
            CacheHit => 1,
        }
    }

    /// Constructs PmuHwCacheResultId from a valid passed-in value.
    pub fn from_regs(value: u64) -> Result<Self> {
        use PmuHwCacheResultId::*;
        match value {
            0 => Ok(CacheMiss),
            1 => Ok(CacheHit),
            _ => Err(Error::InvalidParam),
        }
    }
}

/// Structure to encapsulate parameters for PmuHardware of type CacheReferences/CacheMisses).
#[derive(Copy, Clone, Debug)]
pub struct PmuHwCacheParams {
    cache_id: PmuHwCache,
    op_id: PmuHwCacheOpId,
    result_id: PmuHwCacheResultId,
}

impl PmuHwCacheParams {
    /// Constructs PmuHwCacheParams with the passed-in parameters.
    pub fn new(cache_id: PmuHwCache, op_id: PmuHwCacheOpId, result_id: PmuHwCacheResultId) -> Self {
        Self {
            cache_id,
            op_id,
            result_id,
        }
    }

    /// Constructs PmuHwCacheParams from a valid encoded value.
    pub fn from_regs(value: u64) -> Result<Self> {
        Ok(Self {
            cache_id: PmuHwCache::from_regs(value >> 3)?,
            op_id: PmuHwCacheOpId::from_regs((value >> 1) & 3)?,
            result_id: PmuHwCacheResultId::from_regs(value & 1)?,
        })
    }

    /// Returns the encoded value corresponding to the inner values.
    pub fn raw(&self) -> u64 {
        self.result_id.raw() | (self.op_id.raw() << 1) | (self.cache_id.raw() << 3)
    }
}

/// Enumeration for raw hardware event types.
pub enum PmuHwRawEvents {}

#[derive(Copy, Clone, Debug)]
#[repr(u64)]
/// Enumeration of the firmware event types.
pub enum PmuFirmware {
    /// Misaligned load trap event.
    MisalignedLoad = 0,
    /// Misaligned store trap event.
    MisalignedStore = 1,
    /// Access load trap event.
    AccessLoad = 2,
    /// Access store trap event.
    AccessStore = 3,
    /// Illegal instruction trap event.
    IllegalInstruction = 4,
    /// Set timer event.
    SetTimer = 5,
    /// IPI sent event.
    IpiSent = 6,
    /// IPI received event.
    IpiReceived = 7,
    /// FENCE.I request sent event.
    FenceISent = 8,
    /// FENCE.I request received event.
    FenceIReceived = 9,
    /// SFENCE.VMA request sent event.
    SfenceVmaSent = 10,
    /// SFENCE.VMA request received event.
    SfenceVmaReceived = 11,
    /// SFENCE.ASID request sent event.
    SfenceAsidSent = 12,
    /// SFENCE.ASID request received event.
    SfenceAsidReceived = 13,
    /// HFENCE.GVMA request sent event.
    HfenceGvmaSent = 14,
    /// H
    HfenceGvmaReceived = 15,
    /// HFENCE.GVMA request sent event.
    HfenceVmidSent = 16,
    /// HFENCE.GVMA request received event.
    HfenceVmidReceived = 17,
    /// HFENCE.VVMA request sent event.
    HfenceVvmaSent = 18,
    /// HFEMA request received event.
    HfenceVvmaReceived = 19,
    /// HFENCE.ASID request sent event.
    HfenceVvmaAsidSent = 20,
    /// HFENCE.ASID request received event.
    HfenceVvmaAsidReceived = 21,
}

impl PmuFirmware {
    fn raw(&self) -> u64 {
        use PmuFirmware::*;
        match self {
            MisalignedLoad => 0,
            MisalignedStore => 1,
            AccessLoad => 2,
            AccessStore => 3,
            IllegalInstruction => 4,
            SetTimer => 5,
            IpiSent => 6,
            IpiReceived => 7,
            FenceISent => 8,
            FenceIReceived => 9,
            SfenceVmaSent => 10,
            SfenceVmaReceived => 11,
            SfenceAsidSent => 12,
            SfenceAsidReceived => 13,
            HfenceGvmaSent => 14,
            HfenceGvmaReceived => 15,
            HfenceVmidSent => 16,
            HfenceVmidReceived => 17,
            HfenceVvmaSent => 18,
            HfenceVvmaReceived => 19,
            HfenceVvmaAsidSent => 20,
            HfenceVvmaAsidReceived => 21,
        }
    }

    fn from_regs(value: u64) -> Result<Self> {
        use PmuFirmware::*;
        match value {
            0 => Ok(MisalignedLoad),
            1 => Ok(MisalignedStore),
            2 => Ok(AccessLoad),
            3 => Ok(AccessStore),
            4 => Ok(IllegalInstruction),
            5 => Ok(SetTimer),
            6 => Ok(IpiSent),
            7 => Ok(IpiReceived),
            8 => Ok(FenceISent),
            9 => Ok(FenceIReceived),
            10 => Ok(SfenceVmaSent),
            11 => Ok(SfenceVmaReceived),
            12 => Ok(SfenceAsidSent),
            13 => Ok(SfenceAsidReceived),
            14 => Ok(HfenceGvmaSent),
            15 => Ok(HfenceGvmaReceived),
            16 => Ok(HfenceVmidSent),
            17 => Ok(HfenceVmidReceived),
            18 => Ok(HfenceVvmaSent),
            19 => Ok(HfenceVvmaReceived),
            20 => Ok(HfenceVvmaAsidSent),
            21 => Ok(HfenceVvmaAsidReceived),
            _ => Err(Error::InvalidParam),
        }
    }
}

impl PmuFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use PmuFunction::*;
        match args[6] {
            0 => Ok(GetNumCounters),
            1 => Ok(GetCounterInfo(args[0])),
            2 => Ok(ConfigureMatchingCounters {
                counter_index_base: args[0],
                counter_index_mask: args[1],
                config_flags: PmuCounterConfigFlags::from_regs(args[2])?,
                event_type: PmuEventType::from_regs(args[3])?,
                event_data: args[4],
            }),
            3 => Ok(StartCounters {
                counter_index_base: args[0],
                counter_index_mask: args[1],
                start_flags: PmuCounterStartFlags::from_regs(args[2])?,
                initial_value: args[3],
            }),
            4 => Ok(StopCounters {
                counter_index_base: args[0],
                counter_index_mask: args[1],
                stop_flags: PmuCounterStopFlags::from_regs(args[2])?,
            }),
            5 => Ok(ReadFirmwareCounter(args[0])),
            _ => Err(Error::NotSupported),
        }
    }
}

impl SbiFunction for PmuFunction {
    fn a6(&self) -> u64 {
        use PmuFunction::*;
        match self {
            GetNumCounters => 0,
            GetCounterInfo(_) => 1,
            ConfigureMatchingCounters {
                counter_index_base: _,
                counter_index_mask: _,
                config_flags: _,
                event_type: _,
                event_data: _,
            } => 2,
            StartCounters {
                counter_index_base: _,
                counter_index_mask: _,
                start_flags: _,
                initial_value: _,
            } => 3,
            StopCounters {
                counter_index_base: _,
                counter_index_mask: _,
                stop_flags: _,
            } => 4,
            ReadFirmwareCounter(_) => 5,
        }
    }

    fn a5(&self) -> u64 {
        0
    }

    fn a4(&self) -> u64 {
        use PmuFunction::*;
        match self {
            ConfigureMatchingCounters {
                counter_index_base: _,
                counter_index_mask: _,
                config_flags: _,
                event_type: _,
                event_data,
            } => *event_data,
            _ => 0,
        }
    }

    fn a3(&self) -> u64 {
        use PmuFunction::*;
        match self {
            ConfigureMatchingCounters {
                counter_index_base: _,
                counter_index_mask: _,
                config_flags: _,
                event_type,
                event_data: _,
            } => event_type.raw(),
            StartCounters {
                counter_index_base: _,
                counter_index_mask: _,
                start_flags: _,
                initial_value,
            } => *initial_value,
            _ => 0,
        }
    }

    fn a2(&self) -> u64 {
        use PmuFunction::*;
        match self {
            ConfigureMatchingCounters {
                counter_index_base: _,
                counter_index_mask: _,
                config_flags,
                event_type: _,
                event_data: _,
            } => config_flags.raw(),
            StartCounters {
                counter_index_base: _,
                counter_index_mask: _,
                start_flags,
                initial_value: _,
            } => start_flags.raw(),
            StopCounters {
                counter_index_base: _,
                counter_index_mask: _,
                stop_flags,
            } => stop_flags.raw(),
            _ => 0,
        }
    }

    fn a1(&self) -> u64 {
        use PmuFunction::*;
        match self {
            ConfigureMatchingCounters {
                counter_index_base: _,
                counter_index_mask,
                config_flags: _,
                event_type: _,
                event_data: _,
            } => *counter_index_mask,
            StartCounters {
                counter_index_base: _,
                counter_index_mask,
                start_flags: _,
                initial_value: _,
            } => *counter_index_mask,
            StopCounters {
                counter_index_base: _,
                counter_index_mask,
                stop_flags: _,
            } => *counter_index_mask,
            _ => 0,
        }
    }

    fn a0(&self) -> u64 {
        use PmuFunction::*;
        match self {
            GetCounterInfo(counter_index) => *counter_index,
            ConfigureMatchingCounters {
                counter_index_base,
                counter_index_mask: _,
                config_flags: _,
                event_type: _,
                event_data: _,
            } => *counter_index_base,
            StartCounters {
                counter_index_base,
                counter_index_mask: _,
                start_flags: _,
                initial_value: _,
            } => *counter_index_base,
            StopCounters {
                counter_index_base,
                counter_index_mask: _,
                stop_flags: _,
            } => *counter_index_base,
            ReadFirmwareCounter(counter_index) => *counter_index,
            _ => 0,
        }
    }

    fn result(&self, a0: u64, a1: u64) -> Result<u64> {
        match a0 {
            0 => Ok(a1),
            e => Err(Error::from_code(e as i64)),
        }
    }
}
