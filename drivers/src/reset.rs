// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use device_tree::DeviceTree;
use page_tracking::HwMemMap;
use riscv_pages::{DeviceMemType, PageSize, RawAddr};
use s_mode_utils::abort::abort;

/// The hardcoded base address and len.
const RESET_BASE: u64 = 0x10_0000;
const RESET_LEN: u64 = PageSize::Size4k as u64;

/// Errors that can be returned by the RESET driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Failed to add an MMIO region to the system memory map.
    AddingMmioRegion(page_tracking::MemMapError),
}

/// Holds the result of a RESET driver operation.
pub type Result<T> = core::result::Result<T, Error>;

/// Driver for platform RESET.
pub struct ResetDriver {}

impl ResetDriver {
    /// Probe for a RESET device. At this moment this device always exists and is hardcoded at 0x10_0000.
    pub fn probe_from(_dt: &DeviceTree, mem_map: &mut HwMemMap) -> Result<()> {
        // Safety: being hardcoded we _know_ the device is there.
        unsafe {
            mem_map
                .add_mmio_region(
                    DeviceMemType::Reset,
                    RawAddr::supervisor(RESET_BASE),
                    RESET_LEN,
                )
                .map_err(Error::AddingMmioRegion)
        }?;
        Ok(())
    }

    /// Powers off the machine.
    pub fn shutdown() -> ! {
        // Safety: on this platform, a write of 0x5555 to 0x100000 will trigger the platform to
        // poweroff, which is defined behavior.
        unsafe {
            core::ptr::write_volatile(RESET_BASE as *mut u32, 0x5555);
        }
        abort()
    }
}
