// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use const_oid::ObjectIdentifier;
use digest::{Digest, OutputSizeUser};
use generic_array::GenericArray;
use sbi_rs::MeasurementRegisterDescriptor;

use crate::{Error, Result, TcgPcrIndex};

/// Number of static measurement registers
pub const STATIC_MSMT_REGISTERS: usize = 4;

/// Number of dynamically extensible measurement registers
pub const DYNAMIC_MSMT_REGISTERS: usize = 4;

pub(crate) const MSMT_REGISTERS: usize = STATIC_MSMT_REGISTERS + DYNAMIC_MSMT_REGISTERS;

pub struct MeasurementRegisterBuilder {
    // The DiceTcbInfo FWIDs list index.
    fwid_index: u8,

    // TCG PCR index.
    pcr_index: u8,

    // A static register cannot be extended after the platform is booted, i.e.
    // after the static TCB is finalized.
    // A call to `finalize` will lock static registers, making subsequent calls
    // to `extend` fail.
    static_measurement: bool,

    // For a given workload, a stable measurement register must not change
    // between boots. If it does change, this means the workload TCB actually
    // changed. Examples of stable measurements include include the OS one, but
    // a platform configuration measurement is not stable.
    // A sealing CDI must only be derived from stable measurements, as we don't
    // want sealed data to depend on layers that could change between platforms.
    // An attestation CDI must include non-stable measurements.
    stable: bool,
}

impl MeasurementRegisterBuilder {
    pub fn build<D: Digest>(&self, hash_algorithm: ObjectIdentifier) -> MeasurementRegister<D> {
        MeasurementRegister {
            tcb_layer: None,
            fwid_index: self.fwid_index,
            pcr_index: self.pcr_index,
            extensible: true,
            static_measurement: self.static_measurement,
            stable: self.stable,
            hash_algorithm,
            digest: GenericArray::default(),
        }
    }
}

macro_rules! msmt_reg {
    ($fw_index:expr, $pcr_index:expr, $static:expr, $stable:expr) => {{
        MeasurementRegisterBuilder {
            fwid_index: $fw_index,
            pcr_index: $pcr_index as u8,
            static_measurement: $static,
            stable: $stable,
        }
    }};
}

macro_rules! msmt_static_reg {
    ($fw_index:expr, $pcr_index:expr, $stable:expr) => {{
        msmt_reg!($fw_index, $pcr_index, true, $stable)
    }};
}

macro_rules! msmt_dynamic_reg {
    ($fw_index:expr, $pcr_index:expr, $stable:expr) => {{
        msmt_reg!($fw_index, $pcr_index, false, $stable)
    }};
}

pub const TVM_MSMT_REGISTERS: [MeasurementRegisterBuilder; MSMT_REGISTERS] = [
    // TCG PCR 0 - Platform trusted firmware code.
    // This comes from the hardware RoT.
    msmt_static_reg!(0, TcgPcrIndex::PlatformCode, true),
    // TCG PCR 1 - Platform trusted firmware configuration and data
    // This comes from the hardware RoT.
    // This is not a stable measurement, i.e. the TVM TCB could still be trusted
    // if this measurement changes.
    msmt_static_reg!(1, TcgPcrIndex::PlatformConfiguration, false),
    // TCG PCR 2 - TVM pages
    // The TVM pages, as measured by the TSM.
    msmt_static_reg!(2, TcgPcrIndex::TvmPage, true),
    // TCG PCR 3 - TVM configuration and data
    // The TVM configuration, including its EPC and ARG.
    msmt_static_reg!(3, TcgPcrIndex::TvmConfiguration, true),
    // TCG PCR 17 - Dynamic measurements
    msmt_dynamic_reg!(4, TcgPcrIndex::RuntimePcr0, true),
    // TCG PCR 18 - Dynamic measurements
    msmt_dynamic_reg!(5, TcgPcrIndex::RuntimePcr1, true),
    // TCG PCR 19 - Dynamic measurements
    msmt_dynamic_reg!(6, TcgPcrIndex::RuntimePcr2, true),
    // TCG PCR 20 - Dynamic measurements
    msmt_dynamic_reg!(7, TcgPcrIndex::RuntimePcr3, true),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementRegister<D: Digest> {
    // The DiceTcbInfo sequence index.
    // Only relevant when using the DiceMultitcbinfo format.
    pub tcb_layer: Option<u8>,

    // The DiceTcbInfo FWIDs list index.
    pub fwid_index: u8,

    // TCG PCR index.
    pub pcr_index: u8,

    // Can this register be extended?
    pub extensible: bool,

    // A static register cannot be extended after the platform is booted, i.e.
    // after the static TCB is finalized.
    // A call to `finalize` will lock static registers, making subsequent calls
    // to `extend` fail.
    pub static_measurement: bool,

    // For a given workload, a stable measurement register must not change
    // between boots. If it does change, this means the workload TCB actually
    // changed. For example, a platform config measurement is not stable, while
    // an OS one is.
    // A sealing CDI must only be derived from stable measurements, as we don't
    // want sealed data to depend on layers that could change between platforms.
    // An attestation CDI must include non-stable measurements.
    pub stable: bool,

    // The measured data hash algorithm.
    pub hash_algorithm: ObjectIdentifier,

    // The measured data hash.
    pub digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
}

impl<D: Digest> MeasurementRegister<D> {
    pub fn extend(&mut self, bytes: &[u8], address: Option<u64>) -> Result<()> {
        let mut hasher = self
            .extensible
            .then_some(D::new_with_prefix(self.digest.clone()))
            .ok_or(Error::LockedMeasurementRegister(self.fwid_index))?;

        if let Some(address) = address {
            hasher.update(address.to_le_bytes());
        }
        hasher.update(bytes);
        self.digest = hasher.finalize();

        Ok(())
    }

    pub fn finalize(&mut self) {
        self.static_measurement.then(|| self.extensible = false);
    }

    pub fn to_sbi_descriptor(&self) -> MeasurementRegisterDescriptor {
        MeasurementRegisterDescriptor::new(
            self.tcb_layer.unwrap_or(0),
            self.fwid_index,
            self.pcr_index,
            !self.static_measurement,
        )
    }
}
