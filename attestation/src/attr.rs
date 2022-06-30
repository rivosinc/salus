// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use core::fmt::{self, Write};

use const_oid::db::DB;
use der::asn1::{AnyRef, ObjectIdentifier, SetOf};
use der::{Decode, Encode, Error, ErrorKind, Length, Sequence, Tag, Tagged, ValueOrd};

use crate::{MAX_CSR_ATV, MAX_CSR_ATV_LEN, MAX_CSR_ATV_VALUE, MAX_CSR_ATV_VALUE_LEN};

pub type AttributeType = ObjectIdentifier;

/// X.501 `AttributeValue` as defined in [RFC 5280 Appendix A.1].
///
/// ```text
/// AttributeValue          ::= ANY
/// ```
///
/// [RFC 5280 Appendix A.1]: https://datatracker.ietf.org/doc/html/rfc5280#appendix-A.1
pub type AttributeValue<'a> = AnyRef<'a>;

/// X.501 `Attribute` as defined in [RFC 5280 Appendix A.1].
///
/// ```text
/// Attribute               ::= SEQUENCE {
///     type             AttributeType,
///     values    SET OF AttributeValue -- at least one value is required
/// }
/// ```
///
/// Note that [RFC 2986 Section 4] defines a constrained version of this type:
///
/// ```text
/// Attribute { ATTRIBUTE:IOSet } ::= SEQUENCE {
///     type   ATTRIBUTE.&id({IOSet}),
///     values SET SIZE(1..MAX) OF ATTRIBUTE.&Type({IOSet}{@type})
/// }
/// ```
///
/// The unconstrained version should be preferred.
///
/// [RFC 2986 Section 4]: https://datatracker.ietf.org/doc/html/rfc2986#section-4
/// [RFC 5280 Appendix A.1]: https://datatracker.ietf.org/doc/html/rfc5280#appendix-A.1
#[derive(Clone, Debug, PartialEq, Eq, Sequence, ValueOrd)]
#[allow(missing_docs)]
pub struct Attribute<'a> {
    pub oid: AttributeType,
    pub values: SetOf<AttributeValue<'a>, MAX_CSR_ATV_VALUE>,
}

impl<'a> TryFrom<&'a [u8]> for Attribute<'a> {
    type Error = Error;

    fn try_from(bytes: &'a [u8]) -> Result<Self, Self::Error> {
        Self::from_der(bytes)
    }
}

/// X.501 `Attributes` as defined in [RFC 2986 Section 4].
///
/// ```text
/// Attributes { ATTRIBUTE:IOSet } ::= SET OF Attribute{{ IOSet }}
/// ```
///
/// [RFC 2986 Section 4]: https://datatracker.ietf.org/doc/html/rfc2986#section-4
pub type Attributes<'a> = SetOf<Attribute<'a>, MAX_CSR_ATV>;

/// X.501 `AttributeTypeAndValue` as defined in [RFC 5280 Appendix A.1].
///
/// ```text
/// AttributeTypeAndValue ::= SEQUENCE {
///   type     AttributeType,
///   value    AttributeValue
/// }
/// ```
///
/// [RFC 5280 Appendix A.1]: https://datatracker.ietf.org/doc/html/rfc5280#appendix-A.1
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Sequence, ValueOrd)]
#[allow(missing_docs)]
pub struct AttributeTypeAndValue<'a> {
    pub oid: AttributeType,
    pub value: AnyRef<'a>,
}

#[derive(Copy, Clone)]
enum Escape {
    None,
    Some,
    Hex(u8),
}

struct Parser {
    state: Escape,
    bytes: ArrayVec<u8, MAX_CSR_ATV_VALUE_LEN>,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            state: Escape::None,
            bytes: ArrayVec::<u8, MAX_CSR_ATV_VALUE_LEN>::new(),
        }
    }

    fn try_push(&mut self, c: u8) -> Result<(), Error> {
        self.state = Escape::None;
        self.bytes.try_push(c).map_err(|_| {
            Error::new(
                ErrorKind::Overlength,
                Length::new(MAX_CSR_ATV_VALUE_LEN as u16),
            )
        })
    }

    pub fn add(&mut self, c: u8) -> Result<(), Error> {
        match (self.state, c) {
            (Escape::Hex(p), b'0'..=b'9') => self.try_push(p | (c - b'0'))?,
            (Escape::Hex(p), b'a'..=b'f') => self.try_push(p | (c - b'a' + 10))?,
            (Escape::Hex(p), b'A'..=b'F') => self.try_push(p | (c - b'A' + 10))?,

            (Escape::Some, b'0'..=b'9') => self.state = Escape::Hex((c - b'0') << 4),
            (Escape::Some, b'a'..=b'f') => self.state = Escape::Hex((c - b'a' + 10) << 4),
            (Escape::Some, b'A'..=b'F') => self.state = Escape::Hex((c - b'A' + 10) << 4),

            (Escape::Some, b' ' | b'"' | b'#' | b'=' | b'\\') => self.try_push(c)?,
            (Escape::Some, b'+' | b',' | b';' | b'<' | b'>') => self.try_push(c)?,

            (Escape::None, b'\\') => self.state = Escape::Some,
            (Escape::None, ..) => self.try_push(c)?,

            _ => return Err(ErrorKind::Failed.into()),
        }

        Ok(())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl AttributeTypeAndValue<'_> {
    pub(crate) fn to_array(self) -> Result<ArrayVec<u8, MAX_CSR_ATV_LEN>, Error> {
        let mut atv_bytes = ArrayVec::<u8, MAX_CSR_ATV_LEN>::new();

        let mut _atv_slice = [0u8; MAX_CSR_ATV_LEN];
        let atv_slice = self.encode_to_slice(&mut _atv_slice)?;

        for b in atv_slice {
            atv_bytes.try_push(*b).map_err(|_| {
                Error::new(ErrorKind::Overlength, Length::new(MAX_CSR_ATV_LEN as u16))
            })?;
        }

        Ok(atv_bytes)
    }

    /// Parses the hex value in the `OID=#HEX` format.
    fn encode_hex(
        oid: ObjectIdentifier,
        val: &str,
    ) -> Result<ArrayVec<u8, MAX_CSR_ATV_LEN>, Error> {
        // Ensure an even number of hex bytes.
        let mut iter = match val.len() % 2 {
            0 => [].iter().cloned().chain(val.bytes()),
            1 => [0u8].iter().cloned().chain(val.bytes()),
            _ => unreachable!(),
        };

        // Decode der bytes from hex.
        let mut bytes = ArrayVec::<u8, MAX_CSR_ATV_VALUE_LEN>::new();
        while let (Some(h), Some(l)) = (iter.next(), iter.next()) {
            let mut byte = 0u8;

            for (half, shift) in [(h, 4), (l, 0)] {
                match half {
                    b'0'..=b'9' => byte |= (half - b'0') << shift,
                    b'a'..=b'f' => byte |= (half - b'a' + 10) << shift,
                    b'A'..=b'F' => byte |= (half - b'A' + 10) << shift,
                    _ => return Err(ErrorKind::Failed.into()),
                }
            }

            bytes.push(byte);
        }

        // Serialize.
        let value = AnyRef::from_der(&bytes)?;
        let atv = AttributeTypeAndValue { oid, value };
        atv.to_array()
    }

    /// Parses the string value in the `NAME=STRING` format.
    fn encode_str(
        oid: ObjectIdentifier,
        val: &str,
    ) -> Result<ArrayVec<u8, MAX_CSR_ATV_LEN>, Error> {
        // Undo escaping.
        let mut parser = Parser::new();
        for c in val.bytes() {
            parser.add(c)?;
        }

        // Serialize.
        let value = AnyRef::new(Tag::Utf8String, parser.as_bytes())?;
        let atv = AttributeTypeAndValue { oid, value };
        atv.to_array()
    }

    /// Converts an AttributeTypeAndValue string into an encoded AttributeTypeAndValue
    ///
    /// This function follows the rules in [RFC 4514].
    ///
    /// [RFC 4514]: https://datatracker.ietf.org/doc/html/rfc4514
    pub fn encode_from_string(s: &str) -> Result<ArrayVec<u8, MAX_CSR_ATV_LEN>, Error> {
        let idx = s.find('=').ok_or_else(|| Error::from(ErrorKind::Failed))?;
        let (key, val) = s.split_at(idx);
        let val = &val[1..];

        // Either decode or lookup the OID for the given key.
        let oid = match DB.by_name(key) {
            Some(oid) => *oid,
            None => ObjectIdentifier::new(key)?,
        };

        // If the value is hex-encoded DER...
        match val.strip_prefix('#') {
            Some(val) => Self::encode_hex(oid, val),
            None => Self::encode_str(oid, val),
        }
    }
}

/// Serializes the structure according to the rules in [RFC 4514].
///
/// [RFC 4514]: https://datatracker.ietf.org/doc/html/rfc4514
impl fmt::Display for AttributeTypeAndValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self.value.tag() {
            Tag::PrintableString => self.value.printable_string().ok().map(|s| s.as_str()),
            Tag::Utf8String => self.value.utf8_string().ok().map(|s| s.as_str()),
            Tag::Ia5String => self.value.ia5_string().ok().map(|s| s.as_str()),
            _ => None,
        };

        if let (Some(key), Some(val)) = (DB.by_oid(&self.oid), val) {
            write!(f, "{}=", key)?;

            let mut iter = val.char_indices().peekable();
            while let Some((i, c)) = iter.next() {
                match c {
                    '#' if i == 0 => write!(f, "\\#")?,
                    ' ' if i == 0 || iter.peek().is_none() => write!(f, "\\ ")?,
                    '"' | '+' | ',' | ';' | '<' | '>' | '\\' => write!(f, "\\{}", c)?,
                    '\x00'..='\x1f' | '\x7f' => write!(f, "\\{:02x}", c as u8)?,
                    _ => f.write_char(c)?,
                }
            }
        } else {
            //     let value = self.value.to_vec().or(Err(fmt::Error))?;

            //     write!(f, "{}=#", self.oid)?;
            //     for c in value {
            //         write!(f, "{:02x}", c)?;
            //     }
        }

        Ok(())
    }
}
