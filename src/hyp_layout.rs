// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

// VM-Layout of Salus.
//
//
// +-------------------------+ 0x0000_0000_0000_0000
// | 1:1 HwMemoryMap         |
// +-------------------------+ (Highest HwMemoryMap address)
// | (unused)                |
// +-------------------------+ 0xffff_ffff_0000_0000 (UMODE_START, UMODE_BINARY_START)
// | U-mode ELF mappings     |
// +-------------------------+ +UMODE_BINARY_SIZE (UMODE_BINARY_END)
// | (unused 4Mb)            |
// +-------------------------+ UMODE_MAPPINGS_START
// | U-mode Slot A           |
// +-------------------------+ +UMODE_MAPPINGS_SLOT_SIZE
// | U-mode Slot B           |
// +-------------------------+ +UMODE_MAPPINGS_SLOT_SIZE (UMODE_MAPPINGS_END)
// | (unused 4Mb)            |
// +-------------------------+ UMODE_INPUT_START
// | Umode Input Area        |
// +-------------------------+ +UMODE_INPUT_SIZE

use riscv_pages::{PageAddr, PageSize, SupervisorVirt};
use static_assertions::const_assert;

/// U-mode mappings start here.
pub const UMODE_START: u64 = 0xffff_ffff_0000_0000;
/// U-mode binary mappings start here.
pub const UMODE_BINARY_START: u64 = UMODE_START;
/// Size in bytes of the U-mode binary VA area.
pub const UMODE_BINARY_SIZE: u64 = 128 * 1024 * 1024;
/// U-mode binary mappings end here.
pub const UMODE_BINARY_END: u64 = UMODE_BINARY_START + UMODE_BINARY_SIZE;

/// The addresses between `UMODE_MAPPINGS_START` and `UMODE_MAPPINGS_START` + `UMODE_MAPPINGS_SIZE`
/// is an area of the private page table where the hypervisor can map pages shared from guest
/// VMs. The area is divided in slots, of equal size `UMODE_MAPPING_SLOT_SIZE`.
/// Must be multiple of 4k.
pub const UMODE_MAPPING_SLOT_SIZE: u64 = 4 * 1024 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_MAPPING_SLOT_SIZE));

/// Start of the private U-mode mappings area.  Must be 4k-aligned.
pub const UMODE_MAPPINGS_START: u64 = UMODE_BINARY_END + 4 * 1024 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_MAPPINGS_START));
/// The number of slots available for mapping.
pub const UMODE_MAPPING_SLOTS: u64 = 2;

/// Generic Id names for each of the U-mode mapping slots.
/// There is no mandated use for each of the slots, and caller can decide to map each of them
/// readable or writable based on the requirement.
#[derive(Copy, Clone)]
pub enum UmodeSlotId {
    A,
    B,
}

/// Maximum size of the private mappings area. Must be 4k-aligned.
pub const UMODE_MAPPINGS_SIZE: u64 = UMODE_MAPPING_SLOTS * UMODE_MAPPING_SLOT_SIZE;
/// Starting page address of slot A.
pub const UMODE_MAPPINGS_A_PAGE_ADDR: PageAddr<SupervisorVirt> =
    PageAddr::<SupervisorVirt>::new_const::<UMODE_MAPPINGS_START>();
/// Starting page address of slot B.
pub const UMODE_MAPPINGS_B_PAGE_ADDR: PageAddr<SupervisorVirt> =
    PageAddr::<SupervisorVirt>::new_const::<{ UMODE_MAPPINGS_START + UMODE_MAPPING_SLOT_SIZE }>();
/// End of the private U-mode mappings area. Must be 4k-aligned.
pub const UMODE_MAPPINGS_END: u64 = UMODE_MAPPINGS_START + UMODE_MAPPINGS_SIZE;
const_assert!(PageSize::Size4k.is_aligned(UMODE_MAPPINGS_END));

/// Start of the U-mode Input Region.
pub const UMODE_INPUT_START: u64 = UMODE_MAPPINGS_END + 4 * 1024 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_INPUT_START));
/// Size of the U-mode Input Region.
pub const UMODE_INPUT_SIZE: u64 = 4 * 1024;
const_assert!(PageSize::Size4k.is_aligned(UMODE_INPUT_SIZE));

// Returns true if `addr` is contained in the U-mode binary area.
fn is_umode_binary_addr(addr: u64) -> bool {
    (UMODE_BINARY_START..UMODE_BINARY_END).contains(&addr)
}

/// Returns true if (`addr`, `addr` + `len`) is a valid non-empty range in the VA area.
pub fn is_valid_umode_binary_range(addr: u64, len: usize) -> bool {
    len != 0 && is_umode_binary_addr(addr) && is_umode_binary_addr(addr + len as u64 - 1)
}
