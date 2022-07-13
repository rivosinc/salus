// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use const_oid::{db::rfc5912::ID_SHA_384, ObjectIdentifier};
use der::Encode;
use digest::{Digest, OutputSizeUser};
use generic_array::GenericArray;
use spin::RwLock;

use crate::{
    extensions::{
        dice::tcbinfo::{DiceTcbInfo, MAX_FWID_LEN, MAX_TCB_INFO_HEADER_LEN, TCG_DICE_TCB_INFO},
        Extension,
    },
    Error, Result,
};

/// Number of static measurement registers
pub const STATIC_MSMT_REGISTERS: usize = 4;

/// Number of dynamically extensible measurement registers
pub const DYNAMIC_MSMT_REGISTERS: usize = 4;

const MSMT_REGISTERS: usize = STATIC_MSMT_REGISTERS + DYNAMIC_MSMT_REGISTERS;

/// Largest possible DiceTcbInfo extension length
pub const MAX_TCB_INFO_EXTN_LEN: usize = (MSMT_REGISTERS * MAX_FWID_LEN) + MAX_TCB_INFO_HEADER_LEN;

macro_rules! is_static_measurement {
    ($idx:ident) => {{
        if $idx < STATIC_MSMT_REGISTERS {
            true
        } else {
            false
        }
    }};
}

/// Measurement register indexes aliases.
#[repr(u8)]
pub enum MeasurementIndex {
    /// TVM pages extend the first OS measurement register (PCR 8)
    TvmPage = 2,

    /// Dynamic TVM pages use the first DRTM register (PCR17)
    TvmPageExtension = 4,
}

const TCG_PCR_MAPPING: [u8; MSMT_REGISTERS] = [
    0,  // TCG PCR 0  - SRTM
    1,  // TCG PCR 1  - Platform config
    8,  // TCG PCR 8  - OS
    9,  // TCG PCR 9  - OS
    17, // TCG PCR 17 - DRTM
    18, // TCG PCR 18 - DRTM
    19, // TCG PCR 19 - DRTM
    20, // TCG PCR 20 - DRTM
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct MeasurementRegister<D: Digest> {
    // The DiceTcbInfo sequence index.
    // Only relevant when using the DiceMultitcbinfo format.
    tcb_layer: Option<u8>,

    // The DiceTcbInfo FWIDs list index.
    fwid_index: u8,

    // TCG PCR index.
    pcr_index: u8,

    // Can this register be extended?
    extensible: bool,

    // A static register can not be extended after the platform is booted, i.e.
    // after the static TCB is finalized.
    // A call to `finalize` will lock static registers, making subsequent calls
    // to `extend` fail.
    static_measurement: bool,

    // The measured data hash algorithm.
    hash_algorithm: ObjectIdentifier,

    // The measured data hash.
    digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
}

impl<D: Digest> MeasurementRegister<D> {
    fn new(
        fwid_index: u8,
        pcr_index: u8,
        static_measurement: bool,
        hash_algorithm: ObjectIdentifier,
    ) -> Self {
        MeasurementRegister {
            tcb_layer: None,
            fwid_index,
            pcr_index,
            extensible: true,
            static_measurement,
            hash_algorithm,
            digest: GenericArray::default(),
        }
    }

    fn extend(&mut self, bytes: &[u8], address: Option<u64>) -> Result<()> {
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

    fn finalize(&mut self) {
        self.static_measurement.then(|| self.extensible = false);
    }
}

/// The attestation manager.
#[derive(Debug)]
pub struct AttestationManager<D: Digest> {
    // Measurement registers
    measurements: RwLock<ArrayVec<MeasurementRegister<D>, MSMT_REGISTERS>>,
}

impl<'a, D: Digest> AttestationManager<D> {
    /// Create a new attestation manager.
    pub fn new(hash_algorithm: ObjectIdentifier) -> Self {
        let mut measurements = ArrayVec::<MeasurementRegister<D>, MSMT_REGISTERS>::new();

        for (idx, tcg_pcr) in TCG_PCR_MAPPING.iter().enumerate().take(MSMT_REGISTERS) {
            measurements.insert(
                idx,
                MeasurementRegister::new(
                    idx as u8,
                    *tcg_pcr,
                    is_static_measurement!(idx),
                    hash_algorithm,
                ),
            );
        }

        AttestationManager {
            measurements: RwLock::new(measurements),
        }
    }

    /// Extend one of the measurement registers.
    /// Optionally, this function takes the measured data physical address as
    /// an argument. The register will then be extended with both the data and
    /// its address.
    pub fn extend_msmt_register(
        &self,
        msmt_idx: MeasurementIndex,
        bytes: &[u8],
        address: Option<u64>,
    ) -> Result<()> {
        let idx = msmt_idx as usize;
        if idx > MSMT_REGISTERS - 1 {
            return Err(Error::InvalidMeasurementRegisterIndex(idx));
        }

        self.measurements.write()[idx].extend(bytes, address)
    }

    /// Finalize locks all measurement registers that must no longer be
    /// extended. This should be called after the platform boot process is
    /// finished in order to only allow for dynamic measurements.
    pub fn finalize(&self) {
        for m in self.measurements.write().iter_mut() {
            m.finalize()
        }
    }

    /// Encode the measured TCB into a DiceTcbInfo extension blob.
    pub fn encode_to_tcb_info_extension(&self, extn_buf: &'a mut [u8]) -> Result<&'a [u8]> {
        let mut tcb_info_bytes = [0u8; MAX_TCB_INFO_EXTN_LEN];
        let mut tcb_info = DiceTcbInfo::new();
        let measurements = self.measurements.read();

        for m in measurements.iter() {
            tcb_info.add_fwid::<D>(m.hash_algorithm, &m.digest)?;
        }

        let extn_value = tcb_info
            .encode_to_slice(&mut tcb_info_bytes)
            .map_err(Error::InvalidTcbInfoExtensionDer)?;

        let extn = Extension {
            extn_id: TCG_DICE_TCB_INFO,
            critical: true,
            extn_value,
        };

        extn.encode_to_slice(extn_buf)
            .map_err(Error::InvalidExtensionDer)
    }
}

impl<D: Digest> Default for AttestationManager<D> {
    fn default() -> Self {
        Self::new(ID_SHA_384)
    }
}
