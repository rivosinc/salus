// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use der::asn1::BitStringRef;
use der::{Decode, Enumerated, Sequence};
use spki::{AlgorithmIdentifier, SubjectPublicKeyInfo};

use crate::{attr::Attributes, name::Name, verify::verifier_from_algorithm, Result};

/// Version identifier for certification request information.
///
/// (RFC 2986 designates `0` as the only valid version)
#[derive(Clone, Debug, Copy, PartialEq, Eq, Enumerated)]
#[asn1(type = "INTEGER")]
#[repr(u8)]
pub enum Version {
    /// Denotes PKCS#8 v1
    V1 = 0,
}

/// PKCS#10 `CertificationRequestInfo` as defined in [RFC 2986 Section 4].
///
/// ```text
/// CertificationRequestInfo ::= SEQUENCE {
///     version       INTEGER { v1(0) } (v1,...),
///     subject       Name,
///     subjectPKInfo SubjectPublicKeyInfo{{ PKInfoAlgorithms }},
///     attributes    [0] Attributes{{ CRIAttributes }}
/// }
/// ```
///
/// [RFC 2986 Section 4]: https://datatracker.ietf.org/doc/html/rfc2986#section-4
#[derive(Clone, Debug, PartialEq, Eq, Sequence)]
pub struct CertReqInfo<'a> {
    /// Certification request version.
    pub version: Version,

    /// Subject name.
    pub subject: Name<'a>,

    /// Subject public key info.
    pub public_key: SubjectPublicKeyInfo<'a>,

    /// Request attributes.
    #[asn1(context_specific = "0", tag_mode = "IMPLICIT")]
    pub attributes: Attributes<'a>,
}

impl<'a> TryFrom<&'a [u8]> for CertReqInfo<'a> {
    type Error = der::Error;

    fn try_from(bytes: &'a [u8]) -> core::result::Result<Self, Self::Error> {
        Self::from_der(bytes)
    }
}

/// PKCS#10 `CertificationRequest` as defined in [RFC 2986 Section 4].
///
/// ```text
/// CertificationRequest ::= SEQUENCE {
///     certificationRequestInfo CertificationRequestInfo,
///     signatureAlgorithm AlgorithmIdentifier{{ SignatureAlgorithms }},
///     signature          BIT STRING
/// }
/// ```
///
/// [RFC 2986 Section 4]: https://datatracker.ietf.org/doc/html/rfc2986#section-4
#[derive(Clone, Debug, PartialEq, Eq, Sequence)]
pub struct CertReq<'a> {
    /// Certification request information.
    pub info: CertReqInfo<'a>,

    /// Signature algorithm identifier.
    pub algorithm: AlgorithmIdentifier<'a>,

    /// Signature.
    pub signature: BitStringRef<'a>,
}

impl<'a> TryFrom<&'a [u8]> for CertReq<'a> {
    type Error = der::Error;

    fn try_from(bytes: &'a [u8]) -> core::result::Result<Self, Self::Error> {
        Self::from_der(bytes)
    }
}

impl<'a> CertReq<'a> {
    /// Verifies a CSR signature
    pub fn verify(&self) -> Result<()> {
        verifier_from_algorithm(self.algorithm)?.verify_csr(self)
    }
}
