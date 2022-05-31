// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Tock Register interface for using CSR registers.

pub mod csr_access;
pub mod defs;
pub mod traps;

pub use tock_registers::interfaces::ReadWriteable;
pub use tock_registers::interfaces::Readable;
pub use tock_registers::interfaces::Writeable;
pub use tock_registers::LocalRegisterCopy;

pub use defs::*;
pub use traps::*;

use super::inst::*;
use csr_access::ReadWriteRiscvCsr;

pub struct CSR {
    pub sstatus: ReadWriteRiscvCsr<sstatus::Register, CSR_SSTATUS>,
    pub sie: ReadWriteRiscvCsr<sie::Register, CSR_SIE>,
    pub stvec: ReadWriteRiscvCsr<stvec::Register, CSR_STVEC>,
    pub scounteren: ReadWriteRiscvCsr<scounteren::Register, CSR_SCOUNTEREN>,
    pub sscratch: ReadWriteRiscvCsr<sscratch::Register, CSR_SSCRATCH>,
    pub sepc: ReadWriteRiscvCsr<sepc::Register, CSR_SEPC>,
    pub scause: ReadWriteRiscvCsr<scause::Register, CSR_SCAUSE>,
    pub stval: ReadWriteRiscvCsr<stval::Register, CSR_STVAL>,
    pub sip: ReadWriteRiscvCsr<sie::Register, CSR_SIP>,
    pub stimecmp: ReadWriteRiscvCsr<stimecmp::Register, 0x14d>,
    pub siselect: ReadWriteRiscvCsr<siselect::Register, 0x150>,
    pub sireg: ReadWriteRiscvCsr<sireg::Register, 0x151>,
    pub stopei: ReadWriteRiscvCsr<stopei::Register, 0x15c>,
    pub satp: ReadWriteRiscvCsr<satp::Register, CSR_SATP>,
    pub stopi: ReadWriteRiscvCsr<stopi::Register, 0xdb0>,

    pub hstatus: ReadWriteRiscvCsr<hstatus::Register, CSR_HSTATUS>,
    pub hedeleg: ReadWriteRiscvCsr<hedeleg::Register, CSR_HEDELEG>,
    pub hideleg: ReadWriteRiscvCsr<hideleg::Register, CSR_HIDELEG>,
    pub hie: ReadWriteRiscvCsr<hie::Register, CSR_HIE>,
    pub hcounteren: ReadWriteRiscvCsr<hcounteren::Register, CSR_HCOUNTEREN>,
    pub hgeie: ReadWriteRiscvCsr<hgeie::Register, CSR_HGEIE>,
    pub htval: ReadWriteRiscvCsr<htval::Register, CSR_HTVAL>,
    pub hip: ReadWriteRiscvCsr<hip::Register, CSR_HIP>,
    pub hvip: ReadWriteRiscvCsr<hvip::Register, CSR_HVIP>,
    pub htinst: ReadWriteRiscvCsr<htval::Register, CSR_HTINST>,
    pub hgeip: ReadWriteRiscvCsr<hgeie::Register, CSR_HGEIP>,
    pub henvcfg: ReadWriteRiscvCsr<henvcfg::Register, CSR_HENVCFG>,
    pub hgatp: ReadWriteRiscvCsr<hgatp::Register, CSR_HGATP>,
    pub htimedelta: ReadWriteRiscvCsr<htimedelta::Register, CSR_HTIMEDELTA>,

    pub vsstatus: ReadWriteRiscvCsr<sstatus::Register, CSR_VSSTATUS>,
    pub vsie: ReadWriteRiscvCsr<sie::Register, CSR_VSIE>,
    pub vstvec: ReadWriteRiscvCsr<stvec::Register, CSR_VSTVEC>,
    pub vsscratch: ReadWriteRiscvCsr<sscratch::Register, CSR_VSSCRATCH>,
    pub vsepc: ReadWriteRiscvCsr<sepc::Register, CSR_VSEPC>,
    pub vscause: ReadWriteRiscvCsr<scause::Register, CSR_VSCAUSE>,
    pub vstval: ReadWriteRiscvCsr<stval::Register, CSR_VSTVAL>,
    pub vsip: ReadWriteRiscvCsr<sie::Register, CSR_VSIP>,
    pub vstimecmp: ReadWriteRiscvCsr<stimecmp::Register, 0x24d>,
    pub vsiselect: ReadWriteRiscvCsr<siselect::Register, 0x250>,
    pub vsireg: ReadWriteRiscvCsr<sireg::Register, 0x251>,
    pub vstopei: ReadWriteRiscvCsr<stopei::Register, 0x25c>,
    pub vsatp: ReadWriteRiscvCsr<satp::Register, CSR_VSATP>,
    pub vstopi: ReadWriteRiscvCsr<stopi::Register, 0xeb0>,
}

// Define the "addresses" of each CSR register.
pub const CSR: &CSR = &CSR {
    sstatus: ReadWriteRiscvCsr::new(),
    sie: ReadWriteRiscvCsr::new(),
    stvec: ReadWriteRiscvCsr::new(),
    scounteren: ReadWriteRiscvCsr::new(),
    sscratch: ReadWriteRiscvCsr::new(),
    sepc: ReadWriteRiscvCsr::new(),
    scause: ReadWriteRiscvCsr::new(),
    stval: ReadWriteRiscvCsr::new(),
    sip: ReadWriteRiscvCsr::new(),
    stimecmp: ReadWriteRiscvCsr::new(),
    siselect: ReadWriteRiscvCsr::new(),
    sireg: ReadWriteRiscvCsr::new(),
    stopei: ReadWriteRiscvCsr::new(),
    satp: ReadWriteRiscvCsr::new(),
    stopi: ReadWriteRiscvCsr::new(),

    hstatus: ReadWriteRiscvCsr::new(),
    hedeleg: ReadWriteRiscvCsr::new(),
    hideleg: ReadWriteRiscvCsr::new(),
    hie: ReadWriteRiscvCsr::new(),
    hcounteren: ReadWriteRiscvCsr::new(),
    hgeie: ReadWriteRiscvCsr::new(),
    htval: ReadWriteRiscvCsr::new(),
    hip: ReadWriteRiscvCsr::new(),
    hvip: ReadWriteRiscvCsr::new(),
    htinst: ReadWriteRiscvCsr::new(),
    hgeip: ReadWriteRiscvCsr::new(),
    henvcfg: ReadWriteRiscvCsr::new(),
    hgatp: ReadWriteRiscvCsr::new(),
    htimedelta: ReadWriteRiscvCsr::new(),

    vsstatus: ReadWriteRiscvCsr::new(),
    vsie: ReadWriteRiscvCsr::new(),
    vstvec: ReadWriteRiscvCsr::new(),
    vsscratch: ReadWriteRiscvCsr::new(),
    vsepc: ReadWriteRiscvCsr::new(),
    vscause: ReadWriteRiscvCsr::new(),
    vstval: ReadWriteRiscvCsr::new(),
    vsip: ReadWriteRiscvCsr::new(),
    vstimecmp: ReadWriteRiscvCsr::new(),
    vsiselect: ReadWriteRiscvCsr::new(),
    vsireg: ReadWriteRiscvCsr::new(),
    vstopei: ReadWriteRiscvCsr::new(),
    vsatp: ReadWriteRiscvCsr::new(),
    vstopi: ReadWriteRiscvCsr::new(),
};
