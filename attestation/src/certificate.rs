// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use const_oid::AssociatedOid;
use der::asn1::{BitStringRef, SequenceOf, SetOf, UIntRef, Utf8StringRef};
use der::{AnyRef, Decode, Encode};
use der::{Enumerated, Sequence};
use ed25519_dalek::{Keypair, Signer};
use spki::{AlgorithmIdentifier, SubjectPublicKeyInfo};

use crate::{
    attr::AttributeTypeAndValue,
    extensions::{
        pkix::keyusage::{KeyUsage, KeyUsageFlags, KEY_VALUE_EXTENSION_LEN},
        pkix::basicconstraints::{BasicConstraints, BASIC_CONSTRAINTS_EXTENSION_LEN},
        Extension, Extensions,
    },
    measurement::{AttestationManager, MAX_TCB_INFO_EXTN_LEN},
    name::{Name, RdnSequence, RelativeDistinguishedName},
    request::CertReq,
    time::{Time, Validity},
    Error, Result, MAX_CERT_ATV, MAX_CERT_EXTENSIONS, MAX_CERT_LEN, MAX_CERT_RDN,
};

/// Certificate `Version` as defined in [RFC 5280 Section 4.1].
///
/// ```text
/// Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
/// ```
///
/// [RFC 5280 Section 4.1]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1
#[derive(Clone, Debug, Copy, PartialEq, Eq, Enumerated)]
#[asn1(type = "INTEGER")]
#[repr(u8)]
pub enum Version {
    /// Version 1 (default)
    V1 = 0,

    /// Version 2
    V2 = 1,

    /// Version 3
    V3 = 2,
}

impl Default for Version {
    fn default() -> Self {
        Self::V1
    }
}

/// X.509 `TbsCertificate` as defined in [RFC 5280 Section 4.1]
///
/// ASN.1 structure containing the names of the subject and issuer, a public
/// key associated with the subject, a validity period, and other associated
/// information.
///
/// ```text
/// TBSCertificate  ::=  SEQUENCE  {
///     version         [0]  EXPLICIT Version DEFAULT v1,
///     serialNumber         CertificateSerialNumber,
///     signature            AlgorithmIdentifier,
///     issuer               Name,
///     validity             Validity,
///     subject              Name,
///     subjectPublicKeyInfo SubjectPublicKeyInfo,
///     issuerUniqueID  [1]  IMPLICIT UniqueIdentifier OPTIONAL,
///                          -- If present, version MUST be v2 or v3
///     subjectUniqueID [2]  IMPLICIT UniqueIdentifier OPTIONAL,
///                          -- If present, version MUST be v2 or v3
///     extensions      [3]  Extensions OPTIONAL
///                          -- If present, version MUST be v3 --
/// }
/// ```
///
/// [RFC 5280 Section 4.1]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1
#[derive(Clone, Debug, Eq, PartialEq, Sequence)]
#[allow(missing_docs)]
pub struct TbsCertificate<'a> {
    /// The certificate version
    ///
    /// Note that this value defaults to Version 1 per the RFC. However,
    /// fields such as `issuer_unique_id`, `subject_unique_id` and `extensions`
    /// require later versions. Care should be taken in order to ensure
    /// standards compliance.
    #[asn1(context_specific = "0", default = "Default::default")]
    pub version: Version,

    pub serial_number: UIntRef<'a>,
    pub signature: AlgorithmIdentifier<'a>,
    pub issuer: Name<'a>,
    pub validity: Validity,
    pub subject: Name<'a>,
    pub subject_public_key_info: SubjectPublicKeyInfo<'a>,

    #[asn1(context_specific = "1", tag_mode = "IMPLICIT", optional = "true")]
    pub issuer_unique_id: Option<BitStringRef<'a>>,

    #[asn1(context_specific = "2", tag_mode = "IMPLICIT", optional = "true")]
    pub subject_unique_id: Option<BitStringRef<'a>>,

    #[asn1(context_specific = "3", tag_mode = "EXPLICIT", optional = "true")]
    pub extensions: Option<Extensions<'a>>,
}

/// X.509 certificates are defined in [RFC 5280 Section 4.1].
///
/// ```text
/// Certificate  ::=  SEQUENCE  {
///     tbsCertificate       TBSCertificate,
///     signatureAlgorithm   AlgorithmIdentifier,
///     signature            BIT STRING
/// }
/// ```
///
/// [RFC 5280 Section 4.1]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1
#[derive(Clone, Debug, Eq, PartialEq, Sequence)]
#[allow(missing_docs)]
pub struct Certificate<'a> {
    pub tbs_certificate: TbsCertificate<'a>,
    pub signature_algorithm: AlgorithmIdentifier<'a>,
    pub signature: BitStringRef<'a>,
}

impl<'a> Certificate<'a> {
    /// Build a certificate from a CSR.
    ///
    /// @csr: The CSR input
    /// @cdi_id: The CDI identifier of the certificate issuer (e.g. the TSM).
    /// @key_pair: The ED25519 key pair generated from the CDI.
    /// @attestation_mgr: The attestation manager to provide the DiceTcbInfo extension
    /// @certificate_buffer: A buffer to hold the DER encoded certificate data.
    pub fn from_csr<D: digest::Digest>(
        csr: &CertReq<'a>,
        cdi_id: &'a [u8],
        key_pair: &'a [u8],
        attestation_mgr: &AttestationManager<D>,
        certificate_buf: &'a mut [u8],
    ) -> Result<&'a [u8]> {
        // The serial number is the CDI ID
        let serial_number = UIntRef::new(cdi_id).map_err(Error::InvalidDer)?;

        // Issuer contains one ATV for one RDN: `SN=<CDI_ID>`
        let issuer_atv = AttributeTypeAndValue {
            oid: const_oid::db::rfc4519::SN,
            value: AnyRef::from(Utf8StringRef::new(cdi_id).map_err(Error::InvalidDer)?),
        };
        let mut atv_set = SetOf::<AttributeTypeAndValue, MAX_CERT_ATV>::new();
        atv_set.add(issuer_atv).map_err(Error::InvalidDer)?;
        let rdn = RelativeDistinguishedName(atv_set);
        let mut rdn_sequence = SequenceOf::<RelativeDistinguishedName, MAX_CERT_RDN>::new();
        rdn_sequence.add(rdn).map_err(Error::InvalidDer)?;
        let issuer = RdnSequence(rdn_sequence);

        // We only support ED25519 signature
        let signature_algorithm = AlgorithmIdentifier {
            oid: ed25519::pkcs8::ALGORITHM_OID,
            parameters: None,
        };

        let validity = Validity {
            not_before: Time::past().map_err(Error::InvalidDer)?,
            not_after: Time::never().map_err(Error::InvalidDer)?,
        };

        // Certficate extensions
        let mut extensions = SequenceOf::<_, MAX_CERT_EXTENSIONS>::new();

        // Add the DiceTcbInfo extension
        let mut extn_buffer = [0u8; MAX_TCB_INFO_EXTN_LEN];
        let extn_bytes = attestation_mgr.encode_to_tcb_info_extension(&mut extn_buffer)?;
        extensions
            .add(Extension::from_der(extn_bytes).map_err(Error::InvalidDer)?)
            .map_err(Error::InvalidDer)?;

        // Add the keyUsage extension.
        // The SubjecPublicKeyInfo passed through the CSR should be used for key
        // agreement or wrapping.
        let key_usage = KeyUsage::new(KeyUsageFlags::KeyEncipherment | KeyUsageFlags::KeyAgreement);
        let mut key_usage_buffer = [0u8; KEY_VALUE_EXTENSION_LEN];
        let key_usage_bytes = key_usage
            .encode_to_slice(&mut key_usage_buffer)
            .map_err(Error::InvalidDer)?;

        let key_usage_extension = Extension {
            extn_id: KeyUsage::OID,
            critical: true,
            extn_value: key_usage_bytes,
        };

        extensions
            .add(key_usage_extension)
            .map_err(Error::InvalidDer)?;

        // Add the basicConstraints extension.
        // We are not a CA.
        let basic_constraints = BasicConstraints {
            ca: false,
            path_len_constraint: None,
        };
        let mut basic_constraints_buffer = [0u8; BASIC_CONSTRAINTS_EXTENSION_LEN];
        let basic_constraints_bytes = basic_constraints
            .encode_to_slice(&mut basic_constraints_buffer)
            .map_err(Error::InvalidDer)?;
        let basic_constraints_extension = Extension {
            extn_id: BasicConstraints::OID,
            critical: true,
            extn_value: basic_constraints_bytes,
        };

        extensions
            .add(basic_constraints_extension)
            .map_err(Error::InvalidDer)?;

        // We copy the public key information and subject from the CSR
        let tbs_certificate = TbsCertificate {
            version: Version::V3,
            serial_number,
            issuer,
            validity,
            signature: signature_algorithm,
            subject: csr.info.subject.clone(),
            subject_public_key_info: csr.info.public_key,
            issuer_unique_id: None,
            subject_unique_id: None,
            extensions: Some(extensions),
        };

        // We can now sign the TBS and generate the actual certificate
        let key = Keypair::from_bytes(key_pair).map_err(|_| Error::InvalidKey)?;
        let mut tbs_bytes_buffer = [0u8; MAX_CERT_LEN];
        let tbs_bytes = tbs_certificate
            .encode_to_slice(&mut tbs_bytes_buffer)
            .map_err(Error::InvalidDer)?;
        let signature = key.sign(tbs_bytes).to_bytes();

        let certificate = Certificate {
            tbs_certificate,
            signature: BitStringRef::from_bytes(&signature).map_err(Error::InvalidDer)?,
            signature_algorithm,
        };

        certificate
            .encode_to_slice(certificate_buf)
            .map_err(Error::InvalidDer)
    }
}
