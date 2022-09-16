// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;

use crate::{
    ecall_send, AttestationCapabilities, AttestationFunction, Error, EvidenceFormat, Result,
    SbiMessage, EVIDENCE_DATA_BLOB_SIZE, MAX_HASH_SIZE,
};

/// Maximum supported size for the attestation evidence certificate.
pub const MAX_CERT_SIZE: usize = 4096;

/// Get the attestation capabilities.
pub fn get_capabilities() -> Result<AttestationCapabilities> {
    let caps = AttestationCapabilities::default();
    let msg = SbiMessage::Attestation(AttestationFunction::GetCapabilities {
        caps_addr_out: (&caps as *const AttestationCapabilities) as u64,
        caps_size: core::mem::size_of::<AttestationCapabilities>() as u64,
    });

    // Safety: &caps is the single reference to a variable defined in this scope.
    unsafe { ecall_send(&msg) }?;

    Ok(caps)
}

/// Get an attestation evidence.
/// This function returns a serialized, DER formatted X.509 certificate.
/// The attestation evidence is included as a certificate extension.
///
/// # Arguments
///
/// * `cert_request` - The [Certificate Signing Request]((https://datatracker.ietf.org/doc/html/rfc2986) buffer.
/// * `request_data` - A data blob that will be included in the generated
///                    certificate, as [UserNotice]((https://datatracker.ietf.org/doc/html/rfc2986)
///                    X.509 certificate extension. This is typically used to
///                    pass a cryptographic nonce.
/// * `evidence_format` - The format of the attestation evidence as defined by [`EvidenceFormat`](crate::api::attestation::EvidenceFormat).
pub fn get_evidence(
    cert_request: &[u8],
    request_data: &[u8],
    evidence_format: EvidenceFormat,
) -> Result<ArrayVec<u8, MAX_CERT_SIZE>> {
    if request_data.len() != EVIDENCE_DATA_BLOB_SIZE {
        return Err(Error::InvalidParam);
    }

    let mut cert_bytes = ArrayVec::<u8, MAX_CERT_SIZE>::new();
    let msg = SbiMessage::Attestation(AttestationFunction::GetEvidence {
        cert_request_addr: cert_request.as_ptr() as u64,
        cert_request_size: cert_request.len() as u64,
        request_data_addr: request_data.as_ptr() as u64,
        evidence_format: evidence_format as u64,
        cert_addr_out: (cert_bytes.as_ptr()) as u64,
        cert_size: MAX_CERT_SIZE as u64,
    });

    // Safety: GetEvidence only reads the pages pointed to by `cert_request` and
    // `request_data`. This is safe because they're owned by the borrowed slices
    // passed as arguments.
    // GetEvidence writes to a single reference to `cert_bytes``, which is
    // defined in this scope.
    let len = unsafe { ecall_send(&msg) }?;

    // Safety: cert_bytes is backed by a MAX_CERT_SIZE array.
    unsafe {
        if len as usize > MAX_CERT_SIZE {
            return Err(Error::Failed);
        }
        cert_bytes.set_len(len as usize);
    };

    Ok(cert_bytes)
}

/// Extend a measurement register.
/// # Arguments
///
/// * `digest` - The digest to extend the measurement register with.
/// * `index` - The TCG PCR index of the extended measurement register.
pub fn extend_measurement(digest: &[u8], index: usize) -> Result<()> {
    let msg = SbiMessage::Attestation(AttestationFunction::ExtendMeasurement {
        measurement_data_addr: digest.as_ptr() as u64,
        measurement_data_size: digest.len() as u64,
        measurement_index: index as u64,
    });

    // Safety: ExtendMeasurement only reads the pages pointed to by `digest`.
    // This is safe because they're owned by the borrowed slice passed as an
    // argument to this function.
    unsafe { ecall_send(&msg) }?;

    Ok(())
}

/// Read a measurement register data.
/// # Arguments
///
/// * `index` - The measurement register TCG PCR index.
pub fn read_measurement(index: usize) -> Result<ArrayVec<u8, MAX_HASH_SIZE>> {
    let mut msmt_bytes = ArrayVec::<u8, MAX_HASH_SIZE>::new();

    let msg = SbiMessage::Attestation(AttestationFunction::ReadMeasurement {
        measurement_data_addr_out: msmt_bytes.as_ptr() as u64,
        measurement_data_size: MAX_HASH_SIZE as u64,
        measurement_index: index as u64,
    });

    // Safety: ReadMeasurement writes into a single reference to `msmt_bytes`,
    // which is defined in this scope.
    let len = unsafe { ecall_send(&msg) }?;

    // Safety: msmt_bytes is backed by a MAX_HASH_SIZE array.
    unsafe {
        if len as usize > MAX_HASH_SIZE {
            return Err(Error::Failed);
        }
        msmt_bytes.set_len(len as usize);
    };

    Ok(msmt_bytes)
}
