// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use const_oid::db::rfc5280::ID_CE_KEY_USAGE;
use const_oid::AssociatedOid;
use der::asn1::ObjectIdentifier;
use flagset::{flags, FlagSet};

pub(crate) const KEY_VALUE_EXTENSION_LEN: usize = 8;

flags! {
    /// Key usage flags as defined in [RFC 5280 Section 4.2.1.3].
    ///
    /// ```text
    /// KeyUsage ::= BIT STRING {
    ///      digitalSignature        (0),
    ///      nonRepudiation          (1),  -- recent editions of X.509 have
    ///                                    -- renamed this bit to contentCommitment
    ///      keyEncipherment         (2),
    ///      dataEncipherment        (3),
    ///      keyAgreement            (4),
    ///      keyCertSign             (5),
    ///      cRLSign                 (6),
    ///      encipherOnly            (7),
    ///      decipherOnly            (8)
    /// }
    /// ```
    ///
    /// [RFC 5280 Section 4.2.1.3]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.3
    #[allow(missing_docs)]
    pub enum KeyUsageFlags: u16 {
        DigitalSignature = 1 << 0,
        NonRepudiation = 1 << 1,
        KeyEncipherment = 1 << 2,
        DataEncipherment = 1 << 3,
        KeyAgreement = 1 << 4,
        KeyCertSign = 1 << 5,
        CRLSign = 1 << 6,
        EncipherOnly = 1 << 7,
        DecipherOnly = 1 << 8,
    }
}

/// KeyUsage as defined in [RFC 5280 Section 4.2.1.3].
///
/// [RFC 5280 Section 4.2.1.3]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.2.1.3
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KeyUsage(pub FlagSet<KeyUsageFlags>);

impl AssociatedOid for KeyUsage {
    const OID: ObjectIdentifier = ID_CE_KEY_USAGE;
}

impl_newtype!(KeyUsage, FlagSet<KeyUsageFlags>);

impl KeyUsage {
    /// Build a KeyUsage from a set of KeyUsageFlags
    pub fn new(flags: impl Into<FlagSet<KeyUsageFlags>>) -> KeyUsage {
        KeyUsage(flags.into())
    }
}
