// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Standardized X.509 Certificate Extensions

use der::asn1::SequenceOf;
use der::Sequence;
use spki::ObjectIdentifier;

use crate::MAX_CERT_EXTENSIONS;

pub mod pkix;

/// Extension as defined in [RFC 5280 Section 4.1.2.9].
///
/// The ASN.1 definition for Extension objects is below. The extnValue type
/// may be further parsed using a decoder corresponding to the extnID value.
///
/// ```text
/// Extension  ::=  SEQUENCE  {
///     extnID      OBJECT IDENTIFIER,
///     critical    BOOLEAN DEFAULT FALSE,
///     extnValue   OCTET STRING
///                 -- contains the DER encoding of an ASN.1 value
///                 -- corresponding to the extension type identified
///                 -- by extnID
/// }
/// ```
///
/// [RFC 5280 Section 4.1.2.9]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.9
#[derive(Clone, Debug, Eq, PartialEq, Sequence)]
#[allow(missing_docs)]
pub struct Extension<'a> {
    pub extn_id: ObjectIdentifier,

    #[asn1(default = "Default::default")]
    pub critical: bool,

    #[asn1(type = "OCTET STRING")]
    pub extn_value: &'a [u8],
}

/// Extensions as defined in [RFC 5280 Section 4.1.2.9].
///
/// ```text
/// Extensions  ::=  SEQUENCE SIZE (1..MAX) OF Extension
/// ```
///
/// [RFC 5280 Section 4.1.2.9]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.9
pub type Extensions<'a> = SequenceOf<Extension<'a>, MAX_CERT_EXTENSIONS>;
