// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::fmt;
use der::asn1::{SequenceOf, SetOf};
use der::{Decode, Error, ErrorKind, Length};

use crate::attr::AttributeTypeAndValue;
use crate::{MAX_CSR_ATV, MAX_CSR_RDN, MAX_CSR_RDN_LEN, MAX_CSR_RDN_SEQUENCE_LEN};

/// X.501 Name as defined in [RFC 5280 Section 4.1.2.4]. X.501 Name is used to represent distinguished names.
///
/// ```text
/// Name ::= CHOICE { rdnSequence  RDNSequence }
/// ```
///
/// [RFC 5280 Section 4.1.2.4]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.4
pub type Name<'a> = RdnSequence<'a>;

/// X.501 RDNSequence as defined in [RFC 5280 Section 4.1.2.4].
///
/// ```text
/// RDNSequence ::= SEQUENCE OF RelativeDistinguishedName
/// ```
///
/// [RFC 5280 Section 4.1.2.4]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.4
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RdnSequence<'a>(pub SequenceOf<RelativeDistinguishedName<'a>, MAX_CSR_RDN>);

impl RdnSequence<'_> {
    /// Converts an RDNSequence string into an encoded RDNSequence
    ///
    /// This function follows the rules in [RFC 4514].
    ///
    /// [RFC 4514]: https://datatracker.ietf.org/doc/html/rfc4514
    pub fn encode_from_string(s: &str) -> Result<ArrayVec<u8, MAX_CSR_RDN_SEQUENCE_LEN>, Error> {
        let ders = split(s, b',')
            .map(RelativeDistinguishedName::encode_from_string)
            .collect::<Result<ArrayVec<_, MAX_CSR_RDN>, der::Error>>()?;

        let mut rdn_bytes = ArrayVec::<u8, MAX_CSR_RDN_SEQUENCE_LEN>::new();
        for der in ders.iter() {
            for b in der.iter() {
                rdn_bytes.try_push(*b).map_err(|_| {
                    Error::new(
                        ErrorKind::Overlength,
                        Length::new(MAX_CSR_RDN_SEQUENCE_LEN as u16),
                    )
                })?;
            }
        }

        Ok(rdn_bytes)
    }
}

/// Serializes the structure according to the rules in [RFC 4514].
///
/// [RFC 4514]: https://datatracker.ietf.org/doc/html/rfc4514
impl fmt::Display for RdnSequence<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, rdn) in self.0.iter().enumerate() {
            match i {
                0 => write!(f, "{}", rdn)?,
                _ => write!(f, ",{}", rdn)?,
            }
        }

        Ok(())
    }
}

impl_newtype!(
    RdnSequence<'a>,
    SequenceOf<RelativeDistinguishedName<'a>, MAX_CSR_RDN>
);

/// Find the indices of all non-escaped separators.
fn find(s: &str, b: u8) -> impl '_ + Iterator<Item = usize> {
    (0..s.len())
        .filter(move |i| s.as_bytes()[*i] == b)
        .filter(|i| {
            let x = i
                .checked_sub(2)
                .map(|i| s.as_bytes()[i])
                .unwrap_or_default();

            let y = i
                .checked_sub(1)
                .map(|i| s.as_bytes()[i])
                .unwrap_or_default();

            y != b'\\' || x == b'\\'
        })
}

/// Split a string at all non-escaped separators.
fn split(s: &str, b: u8) -> impl '_ + Iterator<Item = &'_ str> {
    let mut prev = 0;
    find(s, b).chain([s.len()].into_iter()).map(move |i| {
        let x = &s[prev..i];
        prev = i + 1;
        x
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelativeDistinguishedName<'a>(pub SetOf<AttributeTypeAndValue<'a>, MAX_CSR_ATV>);

impl RelativeDistinguishedName<'_> {
    /// Converts an RelativeDistinguishedName string into an encoded RelativeDistinguishedName
    ///
    /// This function follows the rules in [RFC 4514].
    ///
    /// [RFC 4514]: https://datatracker.ietf.org/doc/html/rfc4514
    pub fn encode_from_string(s: &str) -> Result<ArrayVec<u8, MAX_CSR_RDN_LEN>, der::Error> {
        let ders = split(s, b'+')
            .map(AttributeTypeAndValue::encode_from_string)
            .collect::<Result<ArrayVec<_, MAX_CSR_ATV>, der::Error>>()?;

        let atvs = ders
            .iter()
            .map(|der| AttributeTypeAndValue::from_der(der))
            .collect::<Result<ArrayVec<_, MAX_CSR_ATV>, der::Error>>()?;

        let mut rdn_bytes = ArrayVec::<u8, MAX_CSR_RDN_LEN>::new();
        for atv in atvs.iter() {
            for b in atv.to_array().unwrap().iter() {
                rdn_bytes.try_push(*b).map_err(|_| {
                    Error::new(ErrorKind::Overlength, Length::new(MAX_CSR_RDN_LEN as u16))
                })?;
            }
        }

        Ok(rdn_bytes)
    }
}

/// Serializes the structure according to the rules in [RFC 4514].
///
/// [RFC 4514]: https://datatracker.ietf.org/doc/html/rfc4514
impl fmt::Display for RelativeDistinguishedName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, atv) in self.0.iter().enumerate() {
            match i {
                0 => write!(f, "{}", atv)?,
                _ => write!(f, "+{}", atv)?,
            }
        }

        Ok(())
    }
}

impl_newtype!(
    RelativeDistinguishedName<'a>,
    SetOf<AttributeTypeAndValue<'a>, MAX_CSR_ATV>
);
