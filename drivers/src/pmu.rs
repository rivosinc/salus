use arrayvec::ArrayVec;
use s_mode_utils::ecall::ecall_send;
use sbi::PmuFunction::GetNumCounters;
use sbi::*;
use spin::Once;

const MAX_COUNTERS: usize = 32;

/// Caches information about platform hardware and fimrware counters.
#[derive(Debug)]
pub struct PmuInfo {
    /// Platform counter information and the original counter index.
    pub counters_info: ArrayVec<(PmuCounterInfo, bool), MAX_COUNTERS>,
}

static PMU_INFO: Once<PmuInfo> = Once::new();

impl PmuInfo {
    /// Initializes the global PmuInfo structure with the information obtained from the platform SBI.
    pub fn init() {
        let msg = SbiMessage::Pmu(GetNumCounters);
        // Safety: PmuFunction does not touch memory.
        let result = unsafe { ecall_send(&msg) };
        let num_counters = result.unwrap_or(0);
        let mut counters_info = ArrayVec::new();
        for i in 0u64..num_counters {
            let msg = SbiMessage::Pmu(PmuFunction::GetCounterInfo(i));
            // Safety: PmuFunction does not touch memory.
            let (info, is_valid) =
                unsafe { ecall_send(&msg) }.map_or_else(|_| (0, false), |info| (info, true));
            if counters_info
                .try_push((PmuCounterInfo::new(info), is_valid))
                .is_err()
            {
                break;
            }
        }

        let pmu_info = PmuInfo { counters_info };
        PMU_INFO.call_once(|| pmu_info);
    }

    /// Returns a reference to the global PmuInfo structure. Panics if
    /// init() has been called
    pub fn get() -> &'static PmuInfo {
        PMU_INFO.get().unwrap()
    }
}
