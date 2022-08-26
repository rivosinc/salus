// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;

use crate::{
    ecall_send, AttestationFunction, Error, EvidenceFormat, Result, SbiMessage,
    EVIDENCE_DATA_BLOB_SIZE,
};

/// Maximum supported size for the attestation evidence certificate.
pub const MAX_CERT_SIZE: usize = 4096;

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
