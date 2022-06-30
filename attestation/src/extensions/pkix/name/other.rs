// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use der::{asn1::ObjectIdentifier, AnyRef, Sequence};

/// OtherName as defined in [RFC 5280 Section 4.2.1.6].
///
/// ```text
/// OtherName ::= SEQUENCE {
///     type-id    OBJECT IDENTIFIER,
///     value      [0] EXPLICIT ANY DEFINED BY type-id
/// }
/// ```
///
/// [RFC 5280 Section 4.2.1.6]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.6
#[derive(Clone, Debug, Eq, PartialEq, Sequence)]
#[allow(missing_docs)]
pub struct OtherName<'a> {
    pub type_id: ObjectIdentifier,

    #[asn1(context_specific = "0", tag_mode = "EXPLICIT")]
    pub value: AnyRef<'a>,
}

#[test]
#[cfg(test)]
#[cfg(feature = "alloc")]
fn test() {
    use alloc::string::ToString;
    use der::{Decode, Encode};
    use hex_literal::hex;

    let input = hex!("3021060A2B060104018237140203A0130C1155706E5F323134393530313330406D696C");
    let decoded = OtherName::from_der(&input).unwrap();

    let onval = decoded.value.utf8_string().unwrap();
    assert_eq!(onval.to_string(), "Upn_214950130@mil");

    let encoded = decoded.to_vec().unwrap();
    assert_eq!(&input[..], &encoded);
}
