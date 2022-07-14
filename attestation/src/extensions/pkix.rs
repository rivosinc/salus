// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! PKIX X.509 Certificate Extensions (RFC 5280)

pub mod name;

mod authkeyid;
/// keyUsage X.509 extension module.
/// This extension describes the intended usage for the subject public key
/// information (a.k.a. the public key) a certificate is bound to: Key
/// agreement, key wrapping, certificate signing authority, etc.
pub mod keyusage;

use crate::attr::AttributeTypeAndValue;

use const_oid::{AssociatedOid, ObjectIdentifier};

pub use const_oid::db::rfc5280::{
    ID_CE_INHIBIT_ANY_POLICY, ID_CE_ISSUER_ALT_NAME, ID_CE_SUBJECT_ALT_NAME,
    ID_CE_SUBJECT_DIRECTORY_ATTRIBUTES, ID_CE_SUBJECT_KEY_IDENTIFIER,
};

use der::asn1::{OctetStringRef, SequenceOf};

/// SubjectKeyIdentifier as defined in [RFC 5280 Section 4.2.1.2].
///
/// ```text
/// SubjectKeyIdentifier ::= KeyIdentifier
/// ```
///
/// [RFC 5280 Section 4.2.1.2]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.2
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SubjectKeyIdentifier<'a>(pub OctetStringRef<'a>);

impl<'a> AssociatedOid for SubjectKeyIdentifier<'a> {
    const OID: ObjectIdentifier = ID_CE_SUBJECT_KEY_IDENTIFIER;
}

impl_newtype!(SubjectKeyIdentifier<'a>, OctetStringRef<'a>);

/// SubjectAltName as defined in [RFC 5280 Section 4.2.1.6].
///
/// ```text
/// SubjectAltName ::= GeneralNames
/// ```
///
/// [RFC 5280 Section 4.2.1.6]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.6
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubjectAltName<'a>(pub name::GeneralNames<'a>);

impl<'a> AssociatedOid for SubjectAltName<'a> {
    const OID: ObjectIdentifier = ID_CE_SUBJECT_ALT_NAME;
}

impl_newtype!(SubjectAltName<'a>, name::GeneralNames<'a>);

/// IssuerAltName as defined in [RFC 5280 Section 4.2.1.7].
///
/// ```text
/// IssuerAltName ::= GeneralNames
/// ```
///
/// [RFC 5280 Section 4.2.1.7]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.7
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IssuerAltName<'a>(pub name::GeneralNames<'a>);

impl<'a> AssociatedOid for IssuerAltName<'a> {
    const OID: ObjectIdentifier = ID_CE_ISSUER_ALT_NAME;
}

impl_newtype!(IssuerAltName<'a>, name::GeneralNames<'a>);

/// SubjectDirectoryAttributes as defined in [RFC 5280 Section 4.2.1.8].
///
/// ```text
/// SubjectDirectoryAttributes ::= SEQUENCE SIZE (1..MAX) OF AttributeSet
/// ```
///
/// [RFC 5280 Section 4.2.1.8]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.8
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubjectDirectoryAttributes<'a>(pub SequenceOf<AttributeTypeAndValue<'a>, 8>);

impl<'a> AssociatedOid for SubjectDirectoryAttributes<'a> {
    const OID: ObjectIdentifier = ID_CE_SUBJECT_DIRECTORY_ATTRIBUTES;
}

impl_newtype!(
    SubjectDirectoryAttributes<'a>,
    SequenceOf<AttributeTypeAndValue<'a>, 8>
);
