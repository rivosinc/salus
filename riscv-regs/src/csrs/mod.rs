// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Tock Register interface for using CSR registers.

pub mod csr_access;
pub mod defs;
pub mod traps;

pub use csr_access::RiscvCsrInterface;
pub use tock_registers::interfaces::ReadWriteable;
pub use tock_registers::interfaces::Readable;
pub use tock_registers::interfaces::Writeable;
pub use tock_registers::LocalRegisterCopy;

pub use defs::*;
pub use traps::*;

use super::inst::*;
use csr_access::ReadWriteRiscvCsr;
use seq_macro::seq;

pub struct CSR {
    pub sstatus: ReadWriteRiscvCsr<sstatus::Register, CSR_SSTATUS>,
    pub sie: ReadWriteRiscvCsr<sie::Register, CSR_SIE>,
    pub stvec: ReadWriteRiscvCsr<stvec::Register, CSR_STVEC>,
    pub scounteren: ReadWriteRiscvCsr<scounteren::Register, CSR_SCOUNTEREN>,
    pub sscratch: ReadWriteRiscvCsr<sscratch::Register, CSR_SSCRATCH>,
    pub sepc: ReadWriteRiscvCsr<sepc::Register, CSR_SEPC>,
    pub scause: ReadWriteRiscvCsr<scause::Register, CSR_SCAUSE>,
    pub stval: ReadWriteRiscvCsr<stval::Register, CSR_STVAL>,
    pub sip: ReadWriteRiscvCsr<sip::Register, CSR_SIP>,
    pub stimecmp: ReadWriteRiscvCsr<stimecmp::Register, CSR_STIMECMP>,
    pub siselect: ReadWriteRiscvCsr<siselect::Register, CSR_SISELECT>,
    pub sireg: ReadWriteRiscvCsr<sireg::Register, CSR_SIREG>,
    pub stopei: ReadWriteRiscvCsr<stopei::Register, CSR_STOPEI>,
    pub satp: ReadWriteRiscvCsr<satp::Register, CSR_SATP>,
    pub stopi: ReadWriteRiscvCsr<stopi::Register, CSR_STOPI>,

    pub hstatus: ReadWriteRiscvCsr<hstatus::Register, CSR_HSTATUS>,
    pub hedeleg: ReadWriteRiscvCsr<hedeleg::Register, CSR_HEDELEG>,
    pub hideleg: ReadWriteRiscvCsr<hideleg::Register, CSR_HIDELEG>,
    pub hie: ReadWriteRiscvCsr<hie::Register, CSR_HIE>,
    pub hcounteren: ReadWriteRiscvCsr<hcounteren::Register, CSR_HCOUNTEREN>,
    pub hgeie: ReadWriteRiscvCsr<hgeie::Register, CSR_HGEIE>,
    pub hvictl: ReadWriteRiscvCsr<hvictl::Register, CSR_HVICTL>,
    pub htval: ReadWriteRiscvCsr<htval::Register, CSR_HTVAL>,
    pub hip: ReadWriteRiscvCsr<hip::Register, CSR_HIP>,
    pub hvip: ReadWriteRiscvCsr<hvip::Register, CSR_HVIP>,
    pub htinst: ReadWriteRiscvCsr<htinst::Register, CSR_HTINST>,
    pub hgeip: ReadWriteRiscvCsr<hgeip::Register, CSR_HGEIP>,
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
    pub vsip: ReadWriteRiscvCsr<sip::Register, CSR_VSIP>,
    pub vstimecmp: ReadWriteRiscvCsr<stimecmp::Register, CSR_VSTIMECMP>,
    pub vsiselect: ReadWriteRiscvCsr<siselect::Register, CSR_VSISELECT>,
    pub vsireg: ReadWriteRiscvCsr<sireg::Register, CSR_VSIREG>,
    pub vstopei: ReadWriteRiscvCsr<stopei::Register, CSR_VSTOPEI>,
    pub vsatp: ReadWriteRiscvCsr<satp::Register, CSR_VSATP>,
    pub vstopi: ReadWriteRiscvCsr<stopi::Register, CSR_VSTOPI>,

    pub vstart: ReadWriteRiscvCsr<vstart::Register, CSR_VSTART>,
    pub vcsr: ReadWriteRiscvCsr<vcsr::Register, CSR_VCSR>,
    pub vl: ReadWriteRiscvCsr<vl::Register, CSR_VL>,
    pub vtype: ReadWriteRiscvCsr<vtype::Register, CSR_VTYPE>,
    pub vlenb: ReadWriteRiscvCsr<vlenb::Register, CSR_VLENB>,

    pub hpmcounter: [&'static dyn RiscvCsrInterface<R = hpmcounter::Register>; 32],
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
    hvictl: ReadWriteRiscvCsr::new(),
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

    vstart: ReadWriteRiscvCsr::new(),
    vcsr: ReadWriteRiscvCsr::new(),
    vl: ReadWriteRiscvCsr::new(),
    vtype: ReadWriteRiscvCsr::new(),
    vlenb: ReadWriteRiscvCsr::new(),

    hpmcounter: seq!(N in 0xc00..=0xc1f {[
        #( &ReadWriteRiscvCsr::<hpmcounter::Register, N>::new(), )*
    ]}),
};
