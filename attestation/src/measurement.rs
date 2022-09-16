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
use sbi::{AttestationCapabilities, EvidenceFormat, HashAlgorithm, MeasurementRegisterDescriptor};
use spin::RwLock;

use crate::{
    extensions::{
        dice::tcbinfo::{DiceTcbInfo, MAX_FWID_LEN, MAX_TCB_INFO_HEADER_LEN, TCG_DICE_TCB_INFO},
        Extension,
    },
    kdf::{derive_attestation_cdi, derive_sealing_cdi, derive_secret_key, extract_cdi},
    Error, Result,
};

/// Number of static measurement registers
pub const STATIC_MSMT_REGISTERS: usize = 4;

/// Number of dynamically extensible measurement registers
pub const DYNAMIC_MSMT_REGISTERS: usize = 4;

const MSMT_REGISTERS: usize = STATIC_MSMT_REGISTERS + DYNAMIC_MSMT_REGISTERS;

// TODO Get the SVN from the RoT
const TCB_SVN: u64 = 0xdeadbeef;

/// Largest possible DiceTcbInfo extension length
pub const MAX_TCB_INFO_EXTN_LEN: usize = (MSMT_REGISTERS * MAX_FWID_LEN) + MAX_TCB_INFO_HEADER_LEN;

const CDI_LEN: usize = 32;

/// The TVM CDI ID length
/// The TVM CDI ID is derived from the TSM CDI.
pub const CDI_ID_LEN: usize = 20;

/// Our TCG PCR indexes mapping.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TcgPcrIndex {
    /// Platform code Measurement (PCR0)
    PlatformCode = 0,

    /// Platform configuration measurement (PCR1)
    PlatformConfiguration = 1,

    /// TVM pages (PCR2)
    TvmPage = 2,

    /// TVM configuration and data (PCR3)
    TvmConfiguration = 3,

    /// Runtime measurement (PCR17)
    RuntimePcr0 = 17,

    /// Runtime measurement (PCR18)
    RuntimePcr1 = 18,

    /// Runtime measurement (PCR19)
    RuntimePcr2 = 19,

    /// Runtime measurement (PCR20)
    RuntimePcr3 = 20,
}

impl TryFrom<u8> for TcgPcrIndex {
    type Error = crate::Error;
    fn try_from(item: u8) -> Result<Self> {
        match item {
            0 => Ok(TcgPcrIndex::PlatformCode),
            1 => Ok(TcgPcrIndex::PlatformConfiguration),
            2 => Ok(TcgPcrIndex::TvmPage),
            3 => Ok(TcgPcrIndex::TvmConfiguration),
            17 => Ok(TcgPcrIndex::RuntimePcr0),
            18 => Ok(TcgPcrIndex::RuntimePcr1),
            19 => Ok(TcgPcrIndex::RuntimePcr2),
            20 => Ok(TcgPcrIndex::RuntimePcr3),
            _ => Err(Error::InvalidMeasurementRegisterIndex(item as usize)),
        }
    }
}

#[derive(Clone, Debug)]
enum AttestationCdi {}
#[derive(Clone, Debug)]
enum CsrCdi {}
#[derive(Clone, Debug)]
enum SealingCdi {}

struct MeasurementRegisterBuilder {
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
    fn build<D: Digest>(&self, hash_algorithm: ObjectIdentifier) -> MeasurementRegister<D> {
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

const TVM_MSMT_REGISTERS: [MeasurementRegisterBuilder; MSMT_REGISTERS] = [
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

    // A static register cannot be extended after the platform is booted, i.e.
    // after the static TCB is finalized.
    // A call to `finalize` will lock static registers, making subsequent calls
    // to `extend` fail.
    static_measurement: bool,

    // For a given workload, a stable measurement register must not change
    // between boots. If it does change, this means the workload TCB actually
    // changed. For example, a platform config measurement is not stable, while
    // an OS one is.
    // A sealing CDI must only be derived from stable measurements, as we don't
    // want sealed data to depend on layers that could change between platforms.
    // An attestation CDI must include non-stable measurements.
    stable: bool,

    // The measured data hash algorithm.
    hash_algorithm: ObjectIdentifier,

    // The measured data hash.
    digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
}

impl<D: Digest> MeasurementRegister<D> {
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

    fn to_sbi_descriptor(&self) -> MeasurementRegisterDescriptor {
        MeasurementRegisterDescriptor::new(
            self.tcb_layer.unwrap_or(0),
            self.fwid_index,
            self.pcr_index,
            !self.static_measurement,
        )
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
        derive_attestation_cdi::<D, H>(&self.current_cdi, Some(tci_info), None, &mut cdi_csr)?;
        self.next_cdi = Some(cdi_csr);

        Ok(())
    }
}

impl<'a> CdiBuilder<'a> for Cdi<SealingCdi> {
    fn build_next_cdi<D: Digest, H: HmacImpl<D>>(
        &mut self,
        tci_digest: GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
        tci_info: &[u8],
    ) -> Result<()> {
        // The sealing CDI can be generated even after the next layer
        // is finalized.
        let mut cdi_sealing = [0u8; CDI_LEN];
        derive_sealing_cdi::<D, H>(
            &self.current_cdi,
            Some(tci_info),
            Some(tci_digest.as_slice()),
            &mut cdi_sealing,
        )?;
        self.next_cdi = Some(cdi_sealing);

        Ok(())
    }
}

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

    // Sealing Compound Device Identifier (CDI)
    sealing_cdi: RwLock<Cdi<SealingCdi>>,

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

        Ok(AttestationManager {
            measurements: RwLock::new(measurements),
            attestation_cdi: RwLock::new(Cdi::new::<D, H>(attestation_cdi)?),
            csr_cdi: RwLock::new(Cdi::new::<D, H>(attestation_cdi)?),
            sealing_cdi: RwLock::new(Cdi::new::<D, H>(sealing_cdi)?),
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

        // Build the attestation CDI.
        self.attestation_cdi
            .write()
            .build_next_cdi::<D, H>(self.attestation_tci(), &self.vm_id.to_le_bytes())?;

        // Build the CSR CDI.
        // The TCI is empty, we only rely on the TSM one.
        self.csr_cdi
            .write()
            .build_next_cdi::<D, H>(D::new().finalize(), &self.vm_id.to_le_bytes())?;

        // Build the sealing CDI.
        self.sealing_cdi
            .write()
            .build_next_cdi::<D, H>(self.sealing_tci(), &self.vm_id.to_le_bytes())
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
