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
use hkdf::HmacImpl;
use spin::RwLock;

use crate::{
    extensions::{
        dice::tcbinfo::{DiceTcbInfo, MAX_FWID_LEN, MAX_TCB_INFO_HEADER_LEN, TCG_DICE_TCB_INFO},
        Extension,
    },
    kdf::{derive_attestation_cdi, derive_secret_key, extract_cdi},
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

#[derive(Clone, Debug)]
enum AttestationCdi {}
#[derive(Clone, Debug)]
enum CsrCdi {}
#[derive(Clone, Debug)]
enum SealingCdi {}

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

// A CDI builder
trait CdiBuilder<'a> {
    // Build the next layer CDI from the current one.
    fn build_next_cdi<D: Digest, H: HmacImpl<D>>(
        &mut self,
        tci_digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
        tci_info: &[u8],
    ) -> Result<()>;
}

#[derive(Debug)]
struct Cdi<T> {
    // Current layer Compound Device Identifier (CDI)
    current_cdi: [u8; CDI_LEN],

    // Next layer Compound Device Identifier (CDI)
    next_cdi: Option<[u8; CDI_LEN]>,

    _pd_t: PhantomData<T>,
}

impl<'a, T> Cdi<T> {
    fn new<D: Digest, H: HmacImpl<D>>(cdi: &'a [u8]) -> Result<Self> {
        // Extract the current CDI from the passed CDI slice.
        let mut current_cdi = [0u8; CDI_LEN];
        extract_cdi::<D, H>(cdi, &mut current_cdi)?;
        Ok(Cdi {
            current_cdi,
            next_cdi: None,
            _pd_t: PhantomData,
        })
    }

    fn next_cdi(&'a self) -> Result<&'a [u8]> {
        if let Some(next_cdi) = &self.next_cdi {
            return Ok(next_cdi);
        }

        Err(Error::MissingCdi)
    }
}

impl<'a> CdiBuilder<'a> for Cdi<AttestationCdi> {
    fn build_next_cdi<D: Digest, H: HmacImpl<D>>(
        &mut self,
        tci_digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
        tci_info: &[u8],
    ) -> Result<()> {
        // We must not build the CDI multiple times.
        if self.next_cdi.is_some() {
            return Err(Error::NextCDIAlreadyExists);
        }

        let mut cdi_attest = [0u8; CDI_LEN];
        derive_attestation_cdi::<D, H>(
            &self.current_cdi,
            Some(tci_info),
            Some(tci_digest.as_slice()),
            &mut cdi_attest,
        )?;
        self.next_cdi = Some(cdi_attest);

        Ok(())
    }
}

impl<'a> CdiBuilder<'a> for Cdi<CsrCdi> {
    fn build_next_cdi<D: Digest, H: HmacImpl<D>>(
        &mut self,
        _tci_digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
        tci_info: &[u8],
    ) -> Result<()> {
        // We must not build the CDI multiple times.
        if self.next_cdi.is_some() {
            return Err(Error::NextCDIAlreadyExists);
        }

        let mut cdi_csr = [0u8; CDI_LEN];
        derive_attestation_cdi::<D, H>(
            &self.current_cdi,
            Some(tci_info),
            None,
            &mut cdi_csr,
        )?;
        self.next_cdi = Some(cdi_csr);

        Ok(())
    }
}

/// The attestation manager.
#[derive(Debug)]
pub struct AttestationManager<D: Digest, H: HmacImpl<D> = hmac::Hmac<D>> {
    // Measurement registers
    measurements: RwLock<ArrayVec<MeasurementRegister<D>, MSMT_REGISTERS>>,

    // Attestation Compound Device Identifier (CDI)
    attestation_cdi: RwLock<Cdi<AttestationCdi>>,

    // Compound Device Identifier (CDI) used for servicing the certificate
    // signing requests (CSR).
    // This is a CDI derived from the current CDI and the VM identifier.
    csr_cdi: RwLock<Cdi<CsrCdi>>,

    // TVM identifier
    vm_id: u64,

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

        Ok(AttestationManager {
            measurements: RwLock::new(measurements),
            attestation_cdi: RwLock::new(Cdi::new::<D, H>(cdi)?),
            csr_cdi: RwLock::new(Cdi::new::<D, H>(cdi)?),
            vm_id,
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

    fn attestation_tci(&self) -> GenericArray<u8, <D as OutputSizeUser>::OutputSize> {
        let mut hasher = D::new();
        self.measurements
            .read()
            .iter()
            .filter(|m| m.static_measurement)
            .for_each(|m| hasher.update(m.digest.clone()));

        hasher.finalize()
    }

    /// Finalize locks all measurement registers that must no longer be
    /// extended. This should be called after the platform boot process is
    /// finished in order to only allow for dynamic measurements.
    pub fn finalize(&self) -> Result<()> {
        for m in self.measurements.write().iter_mut() {
            m.finalize()
        }

        // Build the attestation CDI.
        self.attestation_cdi
            .write()
            .build_next_cdi::<D, H>(self.attestation_tci(), &self.vm_id.to_le_bytes())?;

        // Build the CSR CDI.
        // The TCI is empty, we only rely on the TSM one.
        self.csr_cdi
            .write()
            .build_next_cdi::<D, H>(D::new().finalize(), &self.vm_id.to_le_bytes())
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

    /// Generate an asymetric key pair from the CSR CDI.
    /// This key pair will be used for signing certificates requests coming from TVMs.
    #[allow(clippy::needless_borrow)]
    pub fn csr_key_pair(&self) -> Result<Keypair> {
        let mut secret_bytes = [0u8; SECRET_KEY_LENGTH];
        derive_secret_key::<D, H>(&self.attestation_cdi.read().next_cdi()?, &mut secret_bytes)?;
        let secret = SecretKey::from_bytes(&secret_bytes).map_err(Error::InvalidKey)?;

        // The public key is then derived from the secret one.
        Ok(Keypair {
            public: (&secret).into(),
            secret,
        })
    }
}
