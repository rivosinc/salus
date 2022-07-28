// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use assertions::const_assert;
use const_field_offset::FieldOffsets;
use tock_registers::interfaces::Readable;
use tock_registers::registers::{ReadOnly, ReadWrite};
use tock_registers::{register_bitfields, LocalRegisterCopy, RegisterLongName, UIntLike};

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
#[derive(FieldOffsets)]
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

/// End byte offset of the common part of a PCI header.
pub const PCI_COMMON_HEADER_END: usize = 0xf;

/// Endpoint (type 0) PCI configuration registers.
#[repr(C)]
#[derive(FieldOffsets)]
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
#[derive(FieldOffsets)]
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

/// Start byte offset of the type-specific part of a PCI header.
pub const PCI_TYPE_HEADER_START: usize = 0x10;
/// End byte offset of the type-specific part of a PCI header.
pub const PCI_TYPE_HEADER_END: usize = 0x3f;

/// Trait for specifying various mask values for a register.
///
/// TODO: Make the `*_mask()` functions const values.
pub trait RegisterMasks: RegisterLongName {
    type RegType: UIntLike;

    /// Returns a bit mask of the bits that are writeable by a VM for this register.
    fn writeable_mask() -> Self::RegType;

    /// Returns a bit mask of the bits that are readable by a VM for this register.
    fn readable_mask() -> Self::RegType;

    /// Returns a bit mask of the clearable (RW1C) bits for this register.
    fn clearable_mask() -> Self::RegType;
}

/// Helpers for masking off virtualized bits in emulated config register accesses.
pub trait RegisterHelpers {
    type RegType: UIntLike;

    /// Returns the value in `self` with the bits that should not be writeable by a VM masked off.
    fn writeable_bits(&self) -> Self::RegType;

    /// Returns the value in `self` with the bits that should not be readable by a VM masked off.
    fn readable_bits(&self) -> Self::RegType;

    /// Returns the value in `self` with the clearable (RW1C) bits masked off.
    fn non_clearable_bits(&self) -> Self::RegType;
}

// PCI config register readable/writeable masks.
//
// In general, we pass-through any non-legacy control and status bits for reads and writes. Anything
// related to INTx is hidden since we don't support virtualizing it. A few other bits are virtualized
// and are not passed through, as noted below.

impl RegisterMasks for Command::Register {
    type RegType = u16;

    fn writeable_mask() -> u16 {
        let mut mask = LocalRegisterCopy::<u16, Command::Register>::new(0);
        mask.modify(Command::IoEnable.val(1));
        mask.modify(Command::MemoryEnable.val(1));
        mask.modify(Command::BusMasterEnable.val(1));
        mask.modify(Command::ParityErrorResponse.val(1));
        mask.modify(Command::SerrEnable.val(1));
        mask.get()
    }

    fn readable_mask() -> u16 {
        Self::writeable_mask()
    }

    fn clearable_mask() -> u16 {
        0
    }
}

// STATUS.CAP_LIST is virtualized.
impl RegisterMasks for Status::Register {
    type RegType = u16;

    fn writeable_mask() -> u16 {
        let mut mask = LocalRegisterCopy::<u16, Status::Register>::new(0);
        mask.modify(Status::MasterDataParityError.val(1));
        mask.modify(Status::SignaledTargetAbort.val(1));
        mask.modify(Status::ReceivedTargetAbort.val(1));
        mask.modify(Status::ReceivedMasterAbort.val(1));
        mask.modify(Status::SignaledSystemError.val(1));
        mask.modify(Status::DetectedParityError.val(1));
        mask.get()
    }

    fn readable_mask() -> u16 {
        let mut mask = LocalRegisterCopy::<u16, Status::Register>::new(Self::writeable_mask());
        mask.modify(Status::ImmediateReadiness.val(1));
        mask.get()
    }

    fn clearable_mask() -> u16 {
        Self::writeable_mask()
    }
}

impl RegisterMasks for SecondaryStatus::Register {
    type RegType = u16;

    fn writeable_mask() -> u16 {
        let mut mask = LocalRegisterCopy::<u16, SecondaryStatus::Register>::new(0);
        mask.modify(SecondaryStatus::MasterDataParityError.val(1));
        mask.modify(SecondaryStatus::SignaledTargetAbort.val(1));
        mask.modify(SecondaryStatus::ReceivedTargetAbort.val(1));
        mask.modify(SecondaryStatus::ReceivedMasterAbort.val(1));
        mask.modify(SecondaryStatus::ReceivedSystemError.val(1));
        mask.modify(SecondaryStatus::DetectedParityError.val(1));
        mask.get()
    }

    fn readable_mask() -> u16 {
        Self::writeable_mask()
    }

    fn clearable_mask() -> u16 {
        Self::writeable_mask()
    }
}

// BRIDGE_CTL.BUS_RESET is virtualized.
impl RegisterMasks for BridgeControl::Register {
    type RegType = u16;

    fn writeable_mask() -> u16 {
        let mut mask = LocalRegisterCopy::<u16, BridgeControl::Register>::new(0);
        mask.modify(BridgeControl::ParityErrorResponse.val(1));
        mask.modify(BridgeControl::SerrEnable.val(1));
        mask.get()
    }

    fn readable_mask() -> u16 {
        Self::writeable_mask()
    }

    fn clearable_mask() -> u16 {
        0
    }
}

// Macro to implement RegisterHelpers for the given type.
macro_rules! reg_helpers_impl {
    ($reg_type:tt) => {
        impl<T: UIntLike, R: RegisterMasks<RegType = T>> RegisterHelpers for $reg_type<T, R> {
            type RegType = T;

            fn writeable_bits(&self) -> Self::RegType {
                self.get() & R::writeable_mask()
            }

            fn readable_bits(&self) -> Self::RegType {
                self.get() & R::readable_mask()
            }

            fn non_clearable_bits(&self) -> Self::RegType {
                self.get() & !R::clearable_mask()
            }
        }
    };
}

reg_helpers_impl!(LocalRegisterCopy);
reg_helpers_impl!(ReadWrite);

fn _assert_register_layout() {
    const_assert!(core::mem::size_of::<CommonRegisters>() == 0x10);
    const_assert!(core::mem::size_of::<EndpointRegisters>() == 0x40);
    const_assert!(core::mem::size_of::<BridgeRegisters>() == 0x40);
}
