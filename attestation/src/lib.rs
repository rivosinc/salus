// Copyright (c) 2021 The RustCrypto Project Developers
// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Pure Rust, heapless attestation crate.
#![no_std]

/// Attestation errors
#[derive(Debug)]
pub enum Error {
    /// The DICE engine failed to extract a CDI.
    DiceCdiExtraction(rice::Error),

    /// The DICE engine failed to build a new layer.
    DiceLayerBuild(rice::Error),

    /// The DICE engine failed to build the next layer CDI and certificates.
    DiceRoll(rice::Error),

    /// The DICE engine failed to generate a TCB DICE extension.
    DiceTcbInfo(rice::Error),

    /// Invalid measurement register descriptor index
    InvalidMeasurementRegisterDescIndex(usize),

    /// Invalid measurement register index
    InvalidMeasurementRegisterIndex(usize),

    /// Measurement register is locked
    LockedMeasurementRegister(u8),

    /// Derived Key is too short
    DerivedKeyTooShort,

    /// The DICE engined failed to retrieve the CDI ID.
    DiceCdiId(rice::Error),
}

/// Custom attestation result.
pub type Result<T> = core::result::Result<T, Error>;

/// Number of static measurement registers
pub const STATIC_MSMT_REGISTERS: usize = 4;

/// Number of dynamically extensible measurement registers
pub const DYNAMIC_MSMT_REGISTERS: usize = 4;

/// Total number of measurement registers
pub const MSMT_REGISTERS: usize = STATIC_MSMT_REGISTERS + DYNAMIC_MSMT_REGISTERS;

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

impl TryFrom<usize> for TcgPcrIndex {
    type Error = crate::Error;
    fn try_from(item: usize) -> Result<Self> {
        // Transform into `u8` first and then into TcgPcrIndex
        TryInto::<u8>::try_into(item)
            .map_err(|_| Error::InvalidMeasurementRegisterIndex(item))?
            .try_into()
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

/// The attesation manager
pub mod manager;
// TCB layer measurement module
mod measurement;

// Alias and be less mouthful.
pub use manager::AttestationManager;
