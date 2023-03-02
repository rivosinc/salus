// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

extern crate libuser;
use libuser::*;

use der::Decode;
use ed25519_dalek::{Signature, PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH};
use generic_array::GenericArray;
use rice::cdi::CompoundDeviceIdentifier;
use rice::cdi::CDI_ID_LEN;
use rice::layer::Layer;
use rice::x509::extensions::dice::tcbinfo::DiceTcbInfo;
use rice::x509::request::CertReq;
use rice::x509::MAX_CSR_LEN;
use signature::{Error as SignatureError, Signer};
use u_mode_api::cert::*;
use u_mode_api::CdiSel;
use zeroize::Zeroize;

#[derive(Debug)]
pub enum Error {
    /// Input CSR buffer size too small.
    CsrBufferTooSmall(usize, usize),
    /// Cannot parse CSR.
    CsrParseFailed(der::Error),
    /// Cannot verify CSR.
    CsrVerificationFailed(rice::Error),
    /// Cannot add FWID extension.
    FwidAddFailed(rice::Error),
    /// Could not create TcbInfo Extensions.
    TcbInfoFailed(rice::Error),
    /// Cannot create Certificate.
    CertificateCreationFailed(rice::Error),
    /// Output Certificate buffer too small.
    CertificateBufferTooSmall(usize, usize),
}

struct UmodeCdi {
    cdi: CdiSel,
}

impl Zeroize for UmodeCdi {
    fn zeroize(&mut self) {
        // No secret information.
    }
}

impl CompoundDeviceIdentifier<PUBLIC_KEY_LENGTH, Signature> for UmodeCdi {
    fn id(&self) -> Result<[u8; CDI_ID_LEN], rice::Error> {
        let mut id = [0u8; CDI_ID_LEN];
        hyp_cdi_id(self.cdi, &mut id);
        Ok(id)
    }

    fn next(&self, _info: Option<&[u8]>, _next_tci: Option<&[u8]>) -> Result<Self, rice::Error> {
        todo!();
    }

    fn public_key(&self) -> [u8; PUBLIC_KEY_LENGTH] {
        todo!();
    }
}

impl Signer<Signature> for UmodeCdi {
    fn try_sign(&self, msg: &[u8]) -> Result<Signature, SignatureError> {
        let mut signature = [0u8; SIGNATURE_LENGTH];
        hyp_cdi_sign(self.cdi, msg, &mut signature);
        Signature::from_bytes(&signature)
    }
}

const ATTESTATION_CURRENT_CDI: UmodeCdi = UmodeCdi {
    cdi: CdiSel::AttestationCurrent,
};

const ATTESTATION_NEXT_CDI: UmodeCdi = UmodeCdi {
    cdi: CdiSel::AttestationNext,
};

pub fn get_certificate_sha384(
    csr_input: &[u8],
    evidence: MeasurementRegisters,
    cert_output: &mut [u8],
) -> Result<u64, Error> {
    // Copy CSR from input.
    let csr_len = csr_input.len();
    if csr_len > MAX_CSR_LEN {
        return Err(Error::CsrBufferTooSmall(csr_len, MAX_CSR_LEN));
    }
    let mut csr_bytes = [0u8; MAX_CSR_LEN];
    csr_bytes[0..csr_len].copy_from_slice(csr_input);

    let mut tcb_info_bytes = [0u8; 4096];
    let mut tcb_info = DiceTcbInfo::new();
    let hash_algorithm = const_oid::db::rfc5912::ID_SHA_384;

    let csr = CertReq::from_der(&csr_bytes[0..csr_len]).map_err(Error::CsrParseFailed)?;

    println!(
        "U-mode CSR version {:?} Signature algorithm {:?}",
        csr.info.version, csr.algorithm.oid
    );

    csr.verify().map_err(Error::CsrVerificationFailed)?;

    for m in evidence.msmt_regs.iter() {
        tcb_info
            .add_fwid::<sha2::Sha384>(hash_algorithm, GenericArray::from_slice(m.as_slice()))
            .map_err(Error::FwidAddFailed)?;
    }

    let tcb_info_extn = tcb_info
        .to_extension(&mut tcb_info_bytes)
        .map_err(Error::TcbInfoFailed)?;
    let extensions: [&[u8]; 1] = [tcb_info_extn];

    let layer: Layer<PUBLIC_KEY_LENGTH, Signature, UmodeCdi, sha2::Sha384> =
        Layer::new(ATTESTATION_CURRENT_CDI, Some(ATTESTATION_NEXT_CDI));
    let cert_der = layer
        .csr_certificate(&csr, Some(&extensions))
        .map_err(Error::CertificateCreationFailed)?;
    let cert_der_len = cert_der.len();
    if cert_output.len() < cert_der_len {
        return Err(Error::CertificateBufferTooSmall(
            cert_output.len(),
            cert_der_len,
        ));
    }
    // Copy cert to output.
    cert_output[0..cert_der_len].copy_from_slice(&cert_der);
    Ok(cert_der_len as u64)
}
