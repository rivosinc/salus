// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use const_oid::ObjectIdentifier;
use core::marker::PhantomData;
use digest::{Digest, OutputSizeUser};
use ed25519_dalek::SECRET_KEY_LENGTH;
use generic_array::GenericArray;
use hkdf::HmacImpl;
use rice::{
    cdi::{CdiType, CDI_ID_LEN},
    layer::Layer,
    x509::{certificate::MAX_CERT_SIZE, extensions::dice::tcbinfo::DiceTcbInfo, request::CertReq},
};
use sbi_rs::{AttestationCapabilities, EvidenceFormat, HashAlgorithm};
use spin::RwLock;

use crate::{
    measurement::{MeasurementRegister, MeasurementRegisterDigest, TVM_MSMT_REGISTERS},
    Error, Result, TcgPcrIndex, DYNAMIC_MSMT_REGISTERS, MSMT_REGISTERS, STATIC_MSMT_REGISTERS,
};

// TODO Get the SVN from the RoT
const TCB_SVN: u64 = 0xdeadbeef;

const CDI_LEN: usize = 32;

/// The TVM configuration data.
/// This structure extends PCR3 when the TVM finalizes.
#[derive(Clone, Default, Debug)]
pub struct TvmConfiguration {
    // Initial program counter for the TVM.
    entry_pc: u64,

    // Initial TVM argument (ARG1).
    entry_arg: u64,
}

impl TvmConfiguration {
    fn set_epc(&mut self, epc: u64) {
        self.entry_pc = epc;
    }

    fn set_arg(&mut self, a1: u64) {
        self.entry_arg = a1;
    }
}

/// The attestation manager.
pub struct AttestationManager<D: Digest, H: HmacImpl<D> = hmac::Hmac<D>> {
    // Measurement registers
    measurements: RwLock<ArrayVec<MeasurementRegister<D>, MSMT_REGISTERS>>,

    // The attestation DICE layer (Built from the attestation TCI)
    attestation_layer: Layer<CDI_LEN, D, H>,

    // The sealing DICE layer (Built from the sealing TCI)
    sealing_layer: Layer<CDI_LEN, D, H>,

    // TVM identifier
    vm_id: u64,

    // TVM configuration.
    // The data here goes into PCR3 when the TVM finalizes.
    tvm_config: RwLock<TvmConfiguration>,

    _pd: PhantomData<H>,
}

impl<'a, D: Digest, H: HmacImpl<D>> AttestationManager<D, H> {
    /// Create a new attestation manager.
    pub fn new(
        attestation_cdi: &'a [u8],
        sealing_cdi: &'a [u8],
        vm_id: u64,
        hash_algorithm: ObjectIdentifier,
    ) -> Result<Self> {
        // Check that we're using a valid hash function for ed25519.
        // We need the hash length to be at least as long as an ed25519
        // secret key (32 bytes).
        if <D as OutputSizeUser>::output_size() < SECRET_KEY_LENGTH {
            return Err(Error::DerivedKeyTooShort);
        }

        let mut measurements = ArrayVec::<MeasurementRegister<D>, MSMT_REGISTERS>::new();

        for (idx, msmt) in TVM_MSMT_REGISTERS.iter().enumerate().take(MSMT_REGISTERS) {
            measurements.insert(idx, msmt.build(hash_algorithm));
        }

        // The CDIs must have the same length as CDI_LEN, so we extract them if
        // that's not the case.
        let mut tmp_a_cdi = [0u8; CDI_LEN];
        let extracted_attestation_cdi = if attestation_cdi.len() != CDI_LEN {
            rice::kdf::extract_cdi::<D, H>(attestation_cdi, &mut tmp_a_cdi)
                .map_err(Error::DiceCdiExtraction)?;
            &tmp_a_cdi
        } else {
            attestation_cdi
        };

        let mut tmp_s_cdi = [0u8; CDI_LEN];
        let extracted_sealing_cdi = if sealing_cdi.len() != CDI_LEN {
            rice::kdf::extract_cdi::<D, H>(sealing_cdi, &mut tmp_s_cdi)
                .map_err(Error::DiceCdiExtraction)?;
            &tmp_s_cdi
        } else {
            sealing_cdi
        };

        Ok(AttestationManager {
            measurements: RwLock::new(measurements),
            attestation_layer: Layer::new(extracted_attestation_cdi, CdiType::Attestation)
                .map_err(Error::DiceLayerBuild)?,
            sealing_layer: Layer::new(extracted_sealing_cdi, CdiType::Sealing)
                .map_err(Error::DiceLayerBuild)?,
            vm_id,
            tvm_config: RwLock::new(Default::default()),
            _pd: PhantomData,
        })
    }

    /// Extend one of the measurement registers.
    /// Optionally, this function takes the measured data physical address as
    /// an argument. The register will then be extended with both the data and
    /// its address.
    pub fn extend_msmt_register(
        &self,
        msmt_idx: TcgPcrIndex,
        bytes: &[u8],
        address: Option<u64>,
    ) -> Result<()> {
        self.measurements
            .write()
            .iter_mut()
            .find(|m| m.pcr_index == msmt_idx as u8)
            .ok_or(Error::InvalidMeasurementRegisterIndex(msmt_idx as usize))?
            .extend(bytes, address)
    }

    /// Read a measurement register data.
    pub fn read_msmt_register(
        &self,
        msmt_idx: TcgPcrIndex,
    ) -> Result<GenericArray<u8, <D as OutputSizeUser>::OutputSize>> {
        Ok(self
            .measurements
            .read()
            .iter()
            .find(|m| m.pcr_index == msmt_idx as u8)
            .ok_or(Error::InvalidMeasurementRegisterIndex(msmt_idx as usize))?
            .digest
            .clone())
    }

    /// Extend the TVM pages measurement.
    /// This is a extend_msmt_register wrapper, where the address is not
    /// optional, and the measurement register is fixed to TvmPage.
    pub fn extend_tvm_page(&self, bytes: &[u8], address: u64) -> Result<()> {
        self.extend_msmt_register(TcgPcrIndex::TvmPage, bytes, Some(address))
    }

    /// Extend the TVM configuration measurement.
    /// This is a extend_msmt_register wrapper, where the address is not
    /// optional, and the measurement register is fixed to TvmPage.
    pub fn extend_tvm_configuration(&self) -> Result<()> {
        self.extend_msmt_register(
            TcgPcrIndex::TvmConfiguration,
            &self.tvm_config.read().entry_pc.to_le_bytes(),
            None,
        )?;
        self.extend_msmt_register(
            TcgPcrIndex::TvmConfiguration,
            &self.tvm_config.read().entry_arg.to_le_bytes(),
            None,
        )
    }

    fn attestation_tci(&self) -> GenericArray<u8, <D as OutputSizeUser>::OutputSize> {
        // The attestation TCI only includes the static measurements.
        let mut hasher = D::new();
        self.measurements
            .read()
            .iter()
            .filter(|m| m.static_measurement)
            .for_each(|m| hasher.update(m.digest.clone()));

        hasher.finalize()
    }

    fn sealing_tci(&self) -> GenericArray<u8, <D as OutputSizeUser>::OutputSize> {
        // The sealing TCI includes all stable measurements.
        let mut hasher = D::new();
        self.measurements
            .read()
            .iter()
            .filter(|m| m.stable)
            .for_each(|m| hasher.update(m.digest.clone()));

        hasher.finalize()
    }

    /// Finalize locks all measurement registers that must no longer be
    /// extended. This should be called after the platform boot process is
    /// finished in order to only allow for dynamic measurements.
    pub fn finalize(&self) -> Result<()> {
        // Extend the TVM configuration PCR.
        self.extend_tvm_configuration()?;
        for m in self.measurements.write().iter_mut() {
            m.finalize()
        }

        // Build the next attestation DICE layer.
        self.attestation_layer
            .roll(
                Some(&self.vm_id.to_le_bytes()),
                Some(&self.attestation_tci()),
            )
            .map_err(Error::DiceRoll)?;

        // Build the next sealing DICE layer.
        self.sealing_layer
            .roll(Some(&self.vm_id.to_le_bytes()), Some(&self.sealing_tci()))
            .map_err(Error::DiceRoll)?;

        Ok(())
    }

    /// Extract data from attestation layer for U-mode operation.
    pub fn measurement_registers(
        &self,
    ) -> Result<ArrayVec<MeasurementRegisterDigest<D>, MSMT_REGISTERS>> {
        Ok(self
            .measurements
            .read()
            .iter()
            .map(|m| m.digest.clone())
            .collect())
    }

    /// Return the CDI ID of the attestation layer.
    pub fn attestation_cdi_id(&self) -> Result<[u8; CDI_ID_LEN]> {
        self.attestation_layer.cdi_id().map_err(Error::DiceCdiId)
    }

    /// Build a DER-formatted x.509 certificate from a CSR.
    /// The built certificate is signed by the TSM, and contains the provided
    /// subject and subject PKI.
    pub fn csr_certificate(&self, csr: &CertReq) -> Result<ArrayVec<u8, MAX_CERT_SIZE>> {
        let mut tcb_info_bytes = [0u8; 4096];
        let mut tcb_info = DiceTcbInfo::new();
        let measurements = self.measurements.read();

        for m in measurements.iter() {
            tcb_info
                .add_fwid::<D>(m.hash_algorithm, &m.digest)
                .map_err(Error::DiceTcbInfo)?;
        }

        let tcb_info_extn = tcb_info.to_extension(&mut tcb_info_bytes).unwrap();
        let extensions: [&[u8]; 1] = [tcb_info_extn];
        self.attestation_layer
            .csr_certificate(csr, Some(&extensions))
            .map_err(Error::DiceCsrCertificate)
    }

    /// Set the TVM initial PC.
    pub fn set_epc(&self, epc: u64) {
        self.tvm_config.write().set_epc(epc);
    }

    /// Set the TVM initial argument (A1).
    pub fn set_arg(&self, a1: u64) {
        self.tvm_config.write().set_arg(a1);
    }

    /// Build the attestation capabilities.
    pub fn capabilities(&self) -> Result<AttestationCapabilities> {
        let mut caps = AttestationCapabilities::new(
            TCB_SVN,
            HashAlgorithm::Sha384,
            EvidenceFormat::DiceTcbInfo,
            STATIC_MSMT_REGISTERS as u8,
            DYNAMIC_MSMT_REGISTERS as u8,
        );

        for (idx, m) in self.measurements.read().iter().enumerate() {
            caps.add_measurement_register(m.to_sbi_descriptor(), idx)
                .map_err(|_| Error::InvalidMeasurementRegisterDescIndex(idx))?;
        }

        Ok(caps)
    }
}
