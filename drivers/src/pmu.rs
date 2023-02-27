// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::ops::ControlFlow;

use arrayvec::ArrayVec;
use sbi_rs::{
    api::pmu, Error as SbiError, PmuCounterInfo, PmuCounterStopFlags, Result as SbiResult,
};
use sync::Once;

/// Maximum number of supported platform PMU counters
pub const MAX_HARDWARE_COUNTERS: usize = 32;

/// Caches information about platform hardware counters.
pub struct PmuInfo {
    hw_counters_info: ArrayVec<Option<PmuCounterInfo>, MAX_HARDWARE_COUNTERS>,
    valid_hw_counters_mask: u64,
}

impl PmuInfo {
    fn new() -> Self {
        Self {
            hw_counters_info: ArrayVec::new(),
            valid_hw_counters_mask: 0,
        }
    }

    fn is_valid_hardware_counter(info: PmuCounterInfo) -> bool {
        use riscv_regs::{CSR_CYCLE, CSR_HPMCOUNTER31};
        info.get_csr() >= (CSR_CYCLE as u64) && info.get_csr() <= (CSR_HPMCOUNTER31 as u64)
    }

    fn init_counter(&mut self, counter_index: u64) -> SbiResult<ControlFlow<()>> {
        let info = pmu::get_counter_info(counter_index);
        let counter_index = counter_index as usize;
        if let Ok(info) = info {
            // Assume that all HW counters are enumerated upfront. The counters may be sparse,
            // but the presence of a firmware counter indicates HW counters are exhausted.
            if info.is_firmware_counter() {
                return Ok(ControlFlow::Break(()));
            } else if !Self::is_valid_hardware_counter(info) {
                return Err(SbiError::NotSupported);
            }
            self.hw_counters_info.push(Some(info));
            self.valid_hw_counters_mask |= 1 << counter_index;
        } else {
            // Hardware counters can be sparse, and may not be implemented.
            self.hw_counters_info.push(None);
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Initializes the global PmuInfo structure with the information obtained from the platform SBI.
    pub fn init() -> SbiResult<()> {
        let mut pmu_info = PmuInfo::new();
        let num_counters = pmu::get_num_counters()?;
        for i in 0..num_counters.min(MAX_HARDWARE_COUNTERS as u64) {
            if matches!(pmu_info.init_counter(i)?, ControlFlow::Break(())) {
                break;
            }
        }
        if !pmu_info.hw_counters_info.is_empty() {
            // Stop and reset all HW counters at the outset
            let stop_flags = PmuCounterStopFlags::default().set_reset_flag();
            let _ = pmu::stop_counters(0, pmu_info.valid_hw_counters_mask, stop_flags);
            PMU_INFO.call_once(|| pmu_info);
            Ok(())
        } else {
            Err(SbiError::NotSupported)
        }
    }

    /// Returns a reference to the global PmuInfo structure.
    pub fn get() -> SbiResult<&'static PmuInfo> {
        PMU_INFO.get().ok_or(SbiError::NotSupported)
    }

    /// Returns the counter index for the specified CSR
    pub fn csr_to_counter_index(&self, csr: u64) -> SbiResult<u64> {
        let counter_index = self
            .hw_counters_info
            .iter()
            .position(|info| info.filter(|i| i.get_csr() == csr).is_some())
            .ok_or(SbiError::InvalidParam)? as u64;
        Ok(counter_index)
    }

    /// Returns the number of hardware counters supported by the platform
    pub fn get_num_counters(&self) -> u64 {
        self.hw_counters_info.len() as u64
    }

    /// Returns cached information for the hardware counter specified by counter_index.
    pub fn get_counter_info(&self, counter_index: u64) -> SbiResult<PmuCounterInfo> {
        let info = self
            .hw_counters_info
            .get(counter_index as usize)
            .and_then(|info| info.as_ref())
            .ok_or(SbiError::InvalidParam)?;
        Ok(*info)
    }

    /// Returns the CSR for the specified counter index
    pub fn counter_index_to_csr(&self, counter_index: u64) -> SbiResult<u64> {
        Ok(self.get_counter_info(counter_index)?.get_csr())
    }

    /// Validates the counter_index and counter_mask bit-mask for counter start and stop operations.
    /// The counter_mask is relative to counter_index. The return value is a sanitized version
    /// of the original counter_mask, with set bits representing implemented platform counters.
    pub fn filter_counter_mask(&self, counter_index: u64, counter_mask: u64) -> SbiResult<u64> {
        let counter_index = counter_index as usize;
        // Validate entire counter mask to ensure that no spurious bits corresponding to
        // non-hardware counter indexes were set. Since hardware counters can be sparse, it's OK to set bits
        // for unimplemented hardware counters, as long as they don't exceed the last known valid index. The
        // platform will automatically select available matching counters based on the mask.
        if (u64::BITS as usize - counter_mask.leading_zeros() as usize) + counter_index
            > self.hw_counters_info.len()
        {
            return Err(SbiError::InvalidParam);
        }
        Ok(counter_mask & (self.valid_hw_counters_mask >> counter_index))
    }
}

static PMU_INFO: Once<PmuInfo> = Once::new();
