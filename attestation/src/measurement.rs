// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use const_oid::ObjectIdentifier;
use core::marker::PhantomData;
use der::Encode;
use digest::{Digest, OutputSizeUser};
use ed25519_dalek::{Keypair, SecretKey, SECRET_KEY_LENGTH};
use generic_array::GenericArray;
use hkdf::{Hkdf, HmacImpl};
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

const CDI_LEN: usize = 32;

/// The TVM CDI ID length
/// The TVM CDI ID is derived from the TSM CDI.
pub const CDI_ID_LEN: usize = 20;

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
pub struct AttestationManager<D: Digest, H: HmacImpl<D> = hmac::Hmac<D>> {
    // Measurement registers
    measurements: RwLock<ArrayVec<MeasurementRegister<D>, MSMT_REGISTERS>>,

    // Compound Device Identifier (CDI) for the managed TVM
    cdi_attest: [u8; CDI_LEN],
    // CDI ID, a 20 bytes identifier derived from the CDI itself
    cdi_id: [u8; CDI_ID_LEN],

    _pd: PhantomData<H>,
}

impl<'a, D: Digest, H: HmacImpl<D>> AttestationManager<D, H> {
    /// Create a new attestation manager.
    pub fn new(cdi: &'a [u8], vm_id: u64, hash_algorithm: ObjectIdentifier) -> Result<Self> {
        // Check that we're using a valid hash function for ed25519.
        // We need the hash length to be at least as long as an ed25519
        // secret key (32 bytes).
        if <D as OutputSizeUser>::output_size() < SECRET_KEY_LENGTH {
            return Err(Error::DerivedKeyTooShort);
        }

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

        // Derive a TVM specific CDI from the TSM CDI.
        let kdf = Hkdf::<D, H>::new(Some(&vm_id.to_le_bytes()), cdi);
        let mut cdi_attest = [0u8; CDI_LEN];
        kdf.expand(b"CDI_Attest", &mut cdi_attest)
            .map_err(Error::InvalidCdiExpansion)?;

        // Derive a reduced and salted CDI ID from the TVM specific CDI
        let id_salt: [u8; 64] = [
            // From the OpenDice implementation.
            0xDB, 0xDB, 0xAE, 0xBC, 0x80, 0x20, 0xDA, 0x9F, 0xF0, 0xDD, 0x5A, 0x24, 0xC8, 0x3A,
            0xA5, 0xA5, 0x42, 0x86, 0xDF, 0xC2, 0x63, 0x03, 0x1E, 0x32, 0x9B, 0x4D, 0xA1, 0x48,
            0x43, 0x06, 0x59, 0xFE, 0x62, 0xCD, 0xB5, 0xB7, 0xE1, 0xE0, 0x0F, 0xC6, 0x80, 0x30,
            0x67, 0x11, 0xEB, 0x44, 0x4A, 0xF7, 0x72, 0x09, 0x35, 0x94, 0x96, 0xFC, 0xFF, 0x1D,
            0xB9, 0x52, 0x0B, 0xA5, 0x1C, 0x7B, 0x29, 0xEA,
        ];
        let kdf_id = Hkdf::<D, H>::new(Some(&id_salt), &cdi_attest);
        let mut cdi_id = [0u8; CDI_ID_LEN];
        kdf_id
            .expand(b"ID", &mut cdi_id)
            .map_err(Error::InvalidCdiExpansion)?;

        Ok(AttestationManager {
            measurements: RwLock::new(measurements),
            cdi_attest,
            cdi_id,
            _pd: PhantomData,
        })
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

    /// Generate an asymetric key pair from the attestation CDI.
    /// This key pair will be used for signing certificates requests coming from TVMs.
    #[allow(clippy::needless_borrow)]
    pub fn key_pair(&self) -> Result<Keypair> {
        // From the OpenDice implementation
        let asym_salt: [u8; 64] = [
            0x63, 0xB6, 0xA0, 0x4D, 0x2C, 0x07, 0x7F, 0xC1, 0x0F, 0x63, 0x9F, 0x21, 0xDA, 0x79,
            0x38, 0x44, 0x35, 0x6C, 0xC2, 0xB0, 0xB4, 0x41, 0xB3, 0xA7, 0x71, 0x24, 0x03, 0x5C,
            0x03, 0xF8, 0xE1, 0xBE, 0x60, 0x35, 0xD3, 0x1F, 0x28, 0x28, 0x21, 0xA7, 0x45, 0x0A,
            0x02, 0x22, 0x2A, 0xB1, 0xB3, 0xCF, 0xF1, 0x67, 0x9B, 0x05, 0xAB, 0x1C, 0xA5, 0xD1,
            0xAF, 0xFB, 0x78, 0x9C, 0xCD, 0x2B, 0x0B, 0x3B,
        ];

        // First we extract a Pseudo Random Key (PRK) for our private key.
        let key_pair_bytes = Hkdf::<D, H>::extract(Some(&asym_salt), &self.cdi_attest).0;
        // We checked that the PRK is long enough when creating the attestation
        // manager instance.
        let sk = SecretKey::from_bytes(&key_pair_bytes.as_slice()[..SECRET_KEY_LENGTH])
            .map_err(Error::InvalidKey)?;

        // The public key is then derived from the secret one.
        let pk = (&sk).into();

        Ok(Keypair {
            public: pk,
            secret: sk,
        })
    }

    /// Encode the CDI ID into a UTF-8 slice.
    pub fn encode_cdi_id(&self, output: &mut [u8]) -> Result<()> {
        hex::encode_to_slice(self.cdi_id, output).map_err(Error::InvalidCdiId)
    }
}
