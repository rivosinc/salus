// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::ecall_send;
use crate::{
    PmuCounterConfigFlags, PmuCounterStartFlags, PmuCounterStopFlags, PmuEventType, PmuFunction,
};
use crate::{PmuCounterInfo, Result, SbiMessage};

/// Returns the number of PMU counters supported by the platform
pub fn get_num_counters() -> Result<u64> {
    let msg = SbiMessage::Pmu(PmuFunction::GetNumCounters);
    // Safety: PmuFunction doesn't touch memory
    unsafe { ecall_send(&msg) }
}

/// Returns information about the PMU counter specified by counter_index
pub fn get_counter_info(counter_index: u64) -> Result<PmuCounterInfo> {
    let msg = SbiMessage::Pmu(PmuFunction::GetCounterInfo(counter_index));
    // Safety: PmuFunction doesn't touch memory
    let info = unsafe { ecall_send(&msg) }?;
    Ok(PmuCounterInfo::new(info))
}

/// Configures PMU counters specified by counter_index and counter_mask
pub fn configure_matching_counters(
    counter_index: u64,
    counter_mask: u64,
    config_flags: PmuCounterConfigFlags,
    event_type: PmuEventType,
    event_data: u64,
) -> Result<u64> {
    let msg = SbiMessage::Pmu(PmuFunction::ConfigureMatchingCounters {
        counter_index,
        counter_mask,
        config_flags,
        event_type,
        event_data,
    });
    // Safety: PmuFunction does not touch memory.
    unsafe { ecall_send(&msg) }
}

/// Starts the counters specified by counter_index and counter_mask
pub fn start_counters(
    counter_index: u64,
    counter_mask: u64,
    start_flags: PmuCounterStartFlags,
    initial_value: u64,
) -> Result<()> {
    let msg = SbiMessage::Pmu(PmuFunction::StartCounters {
        counter_index,
        counter_mask,
        start_flags,
        initial_value,
    });
    // Safety: PmuFunction does not touch memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Stops the counters specified by counter_index and counter_mask
pub fn stop_counters(
    counter_index: u64,
    counter_mask: u64,
    stop_flags: PmuCounterStopFlags,
) -> Result<()> {
    let msg = SbiMessage::Pmu(PmuFunction::StopCounters {
        counter_index,
        counter_mask,
        stop_flags,
    });
    // Safety: PmuFunction does not touch memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Reads the firmware counter specified by counter_index
pub fn read_firmware_counter(counter_index: u64) -> Result<u64> {
    let msg = SbiMessage::Pmu(PmuFunction::ReadFirmwareCounter(counter_index));
    // Safety: PmuFunction does not touch memory.
    unsafe { ecall_send(&msg) }
}
