// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use der::Encode;
use ed25519::pkcs8::{DecodePublicKey, PublicKeyBytes};
use ed25519_dalek::{PublicKey, Signature, Verifier};
use lazy_static::lazy_static;
use spki::AlgorithmIdentifier;

use crate::{request::CertReq, Error, Result, MAX_CERT_LEN};

pub trait CertVerifier {
    /// Verifies a CSR signature
    fn verify_csr(&self, csr: &CertReq) -> Result<()>;
}

pub struct Ed25519Verifier {}

impl Ed25519Verifier {
    const PUB_KEY_DER_SIZE: usize = 64;
}

impl CertVerifier for Ed25519Verifier {
    fn verify_csr(&self, csr: &CertReq) -> Result<()> {
        let mut pub_key_der_bytes = [0u8; Self::PUB_KEY_DER_SIZE];
        let pub_key_der = csr
            .info
            .public_key
            .encode_to_slice(&mut pub_key_der_bytes)
            .map_err(Error::InvalidDer)?;
        let pub_key_bytes =
            PublicKeyBytes::from_public_key_der(pub_key_der).map_err(Error::InvalidPublicKeyDer)?;
        let pub_key = PublicKey::from_bytes(&pub_key_bytes.to_bytes())
            .map_err(|_| Error::InvalidPublicKey)?;

        let mut csr_info_bytes = [0u8; MAX_CERT_LEN];
        let csr_info = csr
            .info
            .encode_to_slice(&mut csr_info_bytes)
            .map_err(Error::InvalidCertReq)?;

        let sig =
            Signature::try_from(csr.signature.raw_bytes()).map_err(|_| Error::InvalidSignature)?;

        pub_key
            .verify(csr_info, &sig)
            .map_err(|_| Error::InvalidSignature)
    }
}

pub fn verifier_from_algorithm(alg: AlgorithmIdentifier) -> Result<&'static dyn CertVerifier> {
    match alg.oid {
        ed25519::pkcs8::ALGORITHM_OID => {
            lazy_static! {
                static ref ED25519_V: Ed25519Verifier = Ed25519Verifier {};
            }
            Ok(&*ED25519_V)
        }

        _ => return Err(Error::UnsupportedAlgorithm(alg)),
    }
}
