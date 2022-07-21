// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use assertions::const_assert;
use tock_registers::register_bitfields;
use tock_registers::registers::{ReadOnly, ReadWrite};

// PCI register definitions. See the PCI Express Base Specification for more details.
//
// Note that the definitions of legacy, PCI-only regsiters and bits are largely omitted.

register_bitfields![u16,
    pub Command [
        IoEnable OFFSET(0) NUMBITS(1),
        MemoryEnable OFFSET(1) NUMBITS(1),
        BusMasterEnable OFFSET(2) NUMBITS(1),
        ParityErrorResponse OFFSET(6) NUMBITS(1),
        SerrEnable OFFSET(8) NUMBITS(1),
        InterruptDisable OFFSET(10) NUMBITS(1),
    ],

    pub Status [
        ImmediateReadiness OFFSET(0) NUMBITS(1),
        InterruptStatus OFFSET(3) NUMBITS(1),
        CapabilitiesList OFFSET(4) NUMBITS(1),
        MasterDataParityError OFFSET(8) NUMBITS(1),
        SignaledTargetAbort OFFSET(11) NUMBITS(1),
        ReceivedTargetAbort OFFSET(12) NUMBITS(1),
        ReceivedMasterAbort OFFSET(13) NUMBITS(1),
        SignaledSystemError OFFSET(14) NUMBITS(1),
        DetectedParityError OFFSET(15) NUMBITS(1),
    ],

    pub SecondaryStatus [
        MasterDataParityError OFFSET(8) NUMBITS(1),
        SignaledTargetAbort OFFSET(11) NUMBITS(1),
        ReceivedTargetAbort OFFSET(12) NUMBITS(1),
        ReceivedMasterAbort OFFSET(13) NUMBITS(1),
        ReceivedSystemError OFFSET(14) NUMBITS(1),
        DetectedParityError OFFSET(15) NUMBITS(1),
    ],

    pub BridgeControl [
        ParityErrorResponse OFFSET(0) NUMBITS(1),
        SerrEnable OFFSET(1) NUMBITS(1),
        IsaEnable OFFSET(2) NUMBITS(1),
        VgaEnable OFFSET(3) NUMBITS(1),
        Vga16Bit OFFSET(4) NUMBITS(1),
        SecondaryBusReset OFFSET(6) NUMBITS(1),
    ],
];

register_bitfields![u8,
    pub Type [
        Layout OFFSET(0) NUMBITS(7) [
            Type0 = 0,
            Type1 = 1,
        ],
        MultiFunction OFFSET(7) NUMBITS(1) [],
    ],

    pub Bist [
        CompletionCode OFFSET(0) NUMBITS(3) [],
        Start OFFSET(6) NUMBITS(1) [],
        Capable OFFSET(7) NUMBITS(1) [],
    ],
];

register_bitfields![u32,
    pub BaseAddress [
        SpaceType OFFSET(0) NUMBITS(1) [
            Memory = 0,
            Io = 1,
        ],
        MemoryType OFFSET(1) NUMBITS(2) [
            Bits32 = 0,
            Bits64 = 2,
        ],
        Prefetchable OFFSET(3) NUMBITS(1) [],
        Address OFFSET(4) NUMBITS(28) [],
    ],
];

/// Common portion of the PCI configuration header.
#[repr(C)]
pub struct CommonRegisters {
    pub vendor_id: ReadOnly<u16>,
    pub dev_id: ReadOnly<u16>,
    pub command: ReadWrite<u16, Command::Register>,
    pub status: ReadWrite<u16, Status::Register>,
    pub rev_id: ReadOnly<u8>,
    pub prog_if: ReadOnly<u8>,
    pub subclass: ReadOnly<u8>,
    pub class: ReadOnly<u8>,
    pub cl_size: ReadWrite<u8>,
    pub lat_timer: ReadOnly<u8>,
    pub header_type: ReadOnly<u8, Type::Register>,
    pub bist: ReadWrite<u8, Bist::Register>,
}

/// Endpoint (type 0) PCI configuration registers.
#[repr(C)]
pub struct EndpointRegisters {
    pub common: CommonRegisters,
    pub bar: [ReadWrite<u32, BaseAddress::Register>; 6],
    pub cardbus: ReadOnly<u32>,
    pub subsys_vendor_id: ReadOnly<u16>,
    pub subsys_id: ReadOnly<u16>,
    pub expansion_rom: ReadOnly<u32>,
    pub cap_ptr: ReadOnly<u8>,
    _reserved: [u8; 7],
    pub int_line: ReadOnly<u8>,
    pub int_pin: ReadWrite<u8>,
    pub min_gnt: ReadOnly<u8>,
    pub max_lat: ReadOnly<u8>,
}

/// Bridge (type 1) PCI configuration registers.
#[repr(C)]
pub struct BridgeRegisters {
    pub common: CommonRegisters,
    pub bar: [ReadWrite<u32, BaseAddress::Register>; 2],
    pub pri_bus: ReadWrite<u8>,
    pub sec_bus: ReadWrite<u8>,
    pub sub_bus: ReadWrite<u8>,
    pub sec_lat: ReadWrite<u8>,
    pub io_base: ReadWrite<u8>,
    pub io_limit: ReadWrite<u8>,
    pub sec_status: ReadWrite<u16, SecondaryStatus::Register>,
    pub mem_base: ReadWrite<u16>,
    pub mem_limit: ReadWrite<u16>,
    pub pref_base: ReadWrite<u16>,
    pub pref_limit: ReadWrite<u16>,
    pub pref_base_upper: ReadWrite<u32>,
    pub pref_limit_upper: ReadWrite<u32>,
    pub io_base_upper: ReadWrite<u16>,
    pub io_limit_upper: ReadWrite<u16>,
    pub cap_ptr: ReadOnly<u8>,
    _reserved: [u8; 3],
    pub expansion_rom: ReadOnly<u32>,
    pub int_line: ReadOnly<u8>,
    pub int_pin: ReadWrite<u8>,
    pub bridge_control: ReadWrite<u16, BridgeControl::Register>,
}

fn _assert_register_layout() {
    const_assert!(core::mem::size_of::<CommonRegisters>() == 0x10);
    const_assert!(core::mem::size_of::<EndpointRegisters>() == 0x40);
    const_assert!(core::mem::size_of::<BridgeRegisters>() == 0x40);
}
