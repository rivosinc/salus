// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use const_oid::{AssociatedOid, ObjectIdentifier};
use der::asn1::{BitStringRef, OctetStringRef, SequenceOf, UIntRef, Utf8StringRef};
use der::Sequence;
use digest::{Digest, OutputSizeUser};
use generic_array::GenericArray;

use crate::{Error, Result, MAX_TCBINFO_FWID};

/// The DiceTcbInfo OID.
/// tcg OBJECT IDENTIFIER ::= {2 23 133}
/// tcg-dice OBJECT IDENTIFIER ::= { tcg platformClass(5) 4 }
pub const TCG_DICE_TCB_INFO: ObjectIdentifier = ObjectIdentifier::new_unwrap("2.23.133.5.4");

pub(crate) const MAX_HASH_OUTPUT_LEN: usize = 64;
pub(crate) const MAX_FWID_LEN: usize = ObjectIdentifier::MAX_SIZE + MAX_HASH_OUTPUT_LEN;
pub(crate) const MAX_TCB_INFO_HEADER_LEN: usize = 128;

/// A list of flags that enumerate potentially simultaneous operational
/// states of the target TCB.
pub type OperationalFlags<'a> = BitStringRef<'a>;

/// DiceTcbInfo as defined by the [DICE Attestation Architecture].
///
/// ```text
/// DiceTcbInfo ::== SEQUENCE {
///     vendor     [0] IMPLICIT UTF8String OPTIONAL,
///     model      [1] IMPLICIT UTF8String OPTIONAL,
///     version    [2] IMPLICIT UTF8String OPTIONAL,
///     svn        [3] IMPLICIT INTEGER OPTIONAL,
///     layer      [4] IMPLICIT INTEGER OPTIONAL,
///     index      [5] IMPLICIT INTEGER OPTIONAL,
///     fwids      [6] IMPLICIT FWIDLIST OPTIONAL,
///     flags      [7] IMPLICIT OperationalFlags OPTIONAL,
///     vendorInfo [8] IMPLICIT OCTET STRING OPTIONAL,
///     type       [9] IMPLICIT OCTET STRING OPTIONAL
/// }
///
/// FWIDLIST ::== SEQUENCE SIZE (1..MAX) OF FWID
/// FWID ::== SEQUENCE {
///     hashAlg OBJECT IDENTIFIER,
///     digest OCTET STRING
/// }
///
/// OperationalFlags ::= BIT STRING {
///     notConfigured (0),
///     notSecure (1),
///     recovery (2),
///     debug (3)
/// }
/// ````
/// [DICE Attestation Architecture]:https://trustedcomputinggroup.org/wp-content/uploads/DICE-Attestation-Architecture-r23-final.pdf
#[derive(Clone, Debug, Default, Eq, PartialEq, Sequence)]
pub struct DiceTcbInfo<'a> {
    #[asn1(context_specific = "0", tag_mode = "IMPLICIT", optional = "true")]
    /// The entity that created the target TCB (e.g., a TCI value).
    pub vendor: Option<Utf8StringRef<'a>>,

    #[asn1(context_specific = "1", tag_mode = "IMPLICIT", optional = "true")]
    /// The product name associated with the target TCB.
    pub model: Option<Utf8StringRef<'a>>,

    #[asn1(context_specific = "2", tag_mode = "IMPLICIT", optional = "true")]
    /// The revision string associated with the target TCB.
    pub version: Option<Utf8StringRef<'a>>,

    #[asn1(context_specific = "3", tag_mode = "IMPLICIT", optional = "true")]
    /// The security version number associated with the target TCB.
    pub svn: Option<UIntRef<'a>>,

    #[asn1(context_specific = "4", tag_mode = "IMPLICIT", optional = "true")]
    /// The DICE layer associated with this measurement of the target TCB.
    pub layer: Option<UIntRef<'a>>,

    #[asn1(context_specific = "5", tag_mode = "IMPLICIT", optional = "true")]
    /// A value that enumerates measurement of assets within the target TCB and DICE layer.
    pub index: Option<UIntRef<'a>>,

    #[asn1(context_specific = "6", tag_mode = "IMPLICIT", optional = "true")]
    /// A list of FWID values resulting from applying the hashAlg function over
    /// the object being measured (e.g., target TCB elements used to compute TCI
    /// and CDI values). FWIDs are computed by the DICE layer that is the
    /// Attesting Environment and certificate Issuer.
    pub fwids: Option<FwIdList<'a>>,

    #[asn1(context_specific = "7", tag_mode = "IMPLICIT", optional = "true")]
    /// A list of flags that enumerate potentially simultaneous operational
    /// states of the target TCB:
    /// * notConfigured (Bit 0)
    /// * notSecure     (Bit 1)
    /// * recovery      (Bit 2)
    /// * debug         (Bit 3)
    pub flags: Option<OperationalFlags<'a>>,

    #[asn1(context_specific = "8", tag_mode = "IMPLICIT", optional = "true")]
    /// Vendor supplied values that encode vendor, model, or device specific state.
    pub vendor_info: Option<OctetStringRef<'a>>,

    #[asn1(context_specific = "9", tag_mode = "IMPLICIT", optional = "true")]
    /// A machine readable description of the measurement.
    pub r#type: Option<OctetStringRef<'a>>,
}

impl<'a> DiceTcbInfo<'a> {
    /// DiceTcbInfo constructor.
    pub fn new() -> Self {
        DiceTcbInfo::default()
    }

    /// Add a `FwId`` entry to the `fwids` list.
    /// This adds an entry at the end of the sequence of `FwId`.
    pub fn add_fwid<D: Digest>(
        &mut self,
        hash_alg: ObjectIdentifier,
        digest: &'a GenericArray<u8, <D as OutputSizeUser>::OutputSize>,
    ) -> Result<&mut Self> {
        let fwid = FwId {
            hash_alg,
            digest: OctetStringRef::new(digest.as_slice()).map_err(Error::InvalidDigest)?,
        };

        if self.fwids.is_none() {
            self.fwids = Some(SequenceOf::<_, MAX_TCBINFO_FWID>::new());
        }

        if let Some(fwids) = self.fwids.as_mut() {
            fwids.add(fwid).map_err(Error::InvalidDigest)?;
        }

        Ok(self)
    }
}

impl<'a> AssociatedOid for DiceTcbInfo<'a> {
    const OID: ObjectIdentifier = TCG_DICE_TCB_INFO;
}

/// A TCB layer hash, together with its hash algorithm.
#[derive(Clone, Debug, Eq, PartialEq, Sequence)]
pub struct FwId<'a> {
    /// Hash algorithm
    pub hash_alg: ObjectIdentifier,
    /// Digest
    pub digest: OctetStringRef<'a>,
}

/// A list of TCB layer hashes.
pub type FwIdList<'a> = SequenceOf<FwId<'a>, MAX_TCBINFO_FWID>;
