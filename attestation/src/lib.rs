// Copyright (c) 2021 The RustCrypto Project Developers
// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Pure Rust, heapless attestation crate.
#![no_std]

/// Maximum supported length for a CSR
pub const MAX_CSR_LEN: usize = 4096;

/// Maximum number of FW ID entries in the `TcbInfo` FWID array
pub const MAX_TCBINFO_FWID: usize = 32;

pub(crate) const MAX_CSR_ATV: usize = 8;
pub(crate) const MAX_CSR_ATV_VALUE: usize = 8;
pub(crate) const MAX_CSR_ATV_VALUE_LEN: usize = 64;
pub(crate) const MAX_CSR_ATV_TYPE_LEN: usize = 64;
pub(crate) const MAX_CSR_ATV_LEN: usize =
    MAX_CSR_ATV_TYPE_LEN + (MAX_CSR_ATV_VALUE * MAX_CSR_ATV_VALUE_LEN);

pub(crate) const MAX_CSR_RDN: usize = 8;
pub(crate) const MAX_CSR_RDN_LEN: usize = MAX_CSR_ATV * MAX_CSR_ATV_LEN;
pub(crate) const MAX_CSR_RDN_SEQUENCE_LEN: usize = MAX_CSR_RDN * MAX_CSR_RDN_LEN;

pub(crate) const MAX_CERT_EXTENSIONS: usize = 8;
pub(crate) const MAX_CERT_ATV: usize = MAX_CSR_ATV;
pub(crate) const MAX_CERT_RDN: usize = MAX_CSR_RDN;

/// Attestation errors
#[derive(Debug)]
pub enum Error {
    /// Invalid CSR
    InvalidCertReq(der::Error),

    /// Invalid DER payload
    InvalidDer(der::Error),

    /// Invalid X.509 extension DER
    InvalidExtensionDer(der::Error),

    /// Invalid public key bytes
    InvalidPublicKey,

    /// Invalid public key DER
    InvalidPublicKeyDer(spki::Error),

    /// Invalid digital signature
    InvalidSignature,

    /// Invalid X.509 TCB info extension DER
    InvalidTcbInfoExtensionDer(der::Error),

    /// Invalid Key bytes
    InvalidKey(ed25519_dalek::SignatureError),

    /// Unsupported signing algorithm
    UnsupportedAlgorithm(const_oid::ObjectIdentifier),

    /// Measurement register is locked
    LockedMeasurementRegister(u8),

    /// Invalid measurement register index
    InvalidMeasurementRegisterIndex(usize),

    /// Invalid measurement register descriptor index
    InvalidMeasurementRegisterDescIndex(usize),

    /// Invalid data digest
    InvalidDigest(der::Error),

    /// Failed to expand the CDI
    InvalidCdiExpansion(hkdf::InvalidLength),

    /// Invalid CDI ID
    InvalidCdiId(hex::FromHexError),

    /// Derived Key is too short
    DerivedKeyTooShort,

    /// Missing CDI
    MissingCdi,

    /// Next layer CDI is already built
    NextCDIAlreadyExists,
}

/// Custom attestation result.
pub type Result<T> = core::result::Result<T, Error>;

/// Our TCG PCR indexes mapping.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TcgPcrIndex {
    /// Platform code Measurement (PCR0)
    PlatformCode = 0,

    /// Platform configuration measurement (PCR1)
    PlatformConfiguration = 1,

    /// TVM pages (PCR2)
    TvmPage = 2,

    /// TVM configuration and data (PCR3)
    TvmConfiguration = 3,

    /// Runtime measurement (PCR17)
    RuntimePcr0 = 17,

    /// Runtime measurement (PCR18)
    RuntimePcr1 = 18,

    /// Runtime measurement (PCR19)
    RuntimePcr2 = 19,

    /// Runtime measurement (PCR20)
    RuntimePcr3 = 20,
}

impl TryFrom<u8> for TcgPcrIndex {
    type Error = crate::Error;
    fn try_from(item: u8) -> Result<Self> {
        match item {
            0 => Ok(TcgPcrIndex::PlatformCode),
            1 => Ok(TcgPcrIndex::PlatformConfiguration),
            2 => Ok(TcgPcrIndex::TvmPage),
            3 => Ok(TcgPcrIndex::TvmConfiguration),
            17 => Ok(TcgPcrIndex::RuntimePcr0),
            18 => Ok(TcgPcrIndex::RuntimePcr1),
            19 => Ok(TcgPcrIndex::RuntimePcr2),
            20 => Ok(TcgPcrIndex::RuntimePcr3),
            _ => Err(Error::InvalidMeasurementRegisterIndex(item as usize)),
        }
    }
}

/// Implements the following traits for a newtype of a `der` decodable/encodable type:
///
/// - `From` conversions to/from the inner type
/// - `AsRef` and `AsMut`
/// - `DecodeValue` and `EncodeValue`
/// - `FixedTag` mapping to the inner value's `FixedTag::TAG`
///
/// The main case is simplifying newtypes which need an `AssociatedOid`
#[macro_export]
macro_rules! impl_newtype {
    ($newtype:ty, $inner:ty) => {
        #[allow(unused_lifetimes)]
        impl<'a> From<$inner> for $newtype {
            #[inline]
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> From<$newtype> for $inner {
            #[inline]
            fn from(value: $newtype) -> Self {
                value.0
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> AsRef<$inner> for $newtype {
            #[inline]
            fn as_ref(&self) -> &$inner {
                &self.0
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> AsMut<$inner> for $newtype {
            #[inline]
            fn as_mut(&mut self) -> &mut $inner {
                &mut self.0
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> ::der::FixedTag for $newtype {
            const TAG: ::der::Tag = <$inner as ::der::FixedTag>::TAG;
        }

        impl<'a> ::der::DecodeValue<'a> for $newtype {
            fn decode_value<R: ::der::Reader<'a>>(
                decoder: &mut R,
                header: ::der::Header,
            ) -> ::der::Result<Self> {
                Ok(Self(<$inner as ::der::DecodeValue>::decode_value(
                    decoder, header,
                )?))
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> ::der::EncodeValue for $newtype {
            fn encode_value(&self, encoder: &mut dyn ::der::Writer) -> ::der::Result<()> {
                self.0.encode_value(encoder)
            }

            fn value_len(&self) -> ::der::Result<::der::Length> {
                self.0.value_len()
            }
        }
    };
}

mod attr;
/// x.509 certificate module
pub mod certificate;
/// x.509 certificate extensions
pub mod extensions;
/// Key Derivation Function module
mod kdf;
/// The attesation manager
pub mod manager;
/// TCB layer measurement module
mod measurement;
mod name;
/// Certificate Signing Resquest module
pub mod request;
mod time;
mod verify;

// Alias and be less mouthful.
pub use manager::AttestationManager;
