// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use digest::Digest;
use hkdf::{Hkdf, HmacImpl};

use crate::{Error, Result};

// From the OpenDice implementation.
pub(crate) const ID_SALT: [u8; 64] = [
    0xDB, 0xDB, 0xAE, 0xBC, 0x80, 0x20, 0xDA, 0x9F, 0xF0, 0xDD, 0x5A, 0x24, 0xC8, 0x3A, 0xA5, 0xA5,
    0x42, 0x86, 0xDF, 0xC2, 0x63, 0x03, 0x1E, 0x32, 0x9B, 0x4D, 0xA1, 0x48, 0x43, 0x06, 0x59, 0xFE,
    0x62, 0xCD, 0xB5, 0xB7, 0xE1, 0xE0, 0x0F, 0xC6, 0x80, 0x30, 0x67, 0x11, 0xEB, 0x44, 0x4A, 0xF7,
    0x72, 0x09, 0x35, 0x94, 0x96, 0xFC, 0xFF, 0x1D, 0xB9, 0x52, 0x0B, 0xA5, 0x1C, 0x7B, 0x29, 0xEA,
];

// From the OpenDice implementation
pub(crate) const ASYM_SALT: [u8; 64] = [
    0x63, 0xB6, 0xA0, 0x4D, 0x2C, 0x07, 0x7F, 0xC1, 0x0F, 0x63, 0x9F, 0x21, 0xDA, 0x79, 0x38, 0x44,
    0x35, 0x6C, 0xC2, 0xB0, 0xB4, 0x41, 0xB3, 0xA7, 0x71, 0x24, 0x03, 0x5C, 0x03, 0xF8, 0xE1, 0xBE,
    0x60, 0x35, 0xD3, 0x1F, 0x28, 0x28, 0x21, 0xA7, 0x45, 0x0A, 0x02, 0x22, 0x2A, 0xB1, 0xB3, 0xCF,
    0xF1, 0x67, 0x9B, 0x05, 0xAB, 0x1C, 0xA5, 0xD1, 0xAF, 0xFB, 0x78, 0x9C, 0xCD, 0x2B, 0x0B, 0x3B,
];

// Generic HKDF-based derivation function
fn kdf<D: Digest, H: HmacImpl<D>>(
    input_key_material: &[u8],
    salt: &[u8],
    info: &[&[u8]],
    output_key_material: &mut [u8],
) -> Result<()> {
    let kdf = Hkdf::<D, H>::new(Some(salt), input_key_material);
    kdf.expand_multi_info(info, output_key_material)
        .map_err(Error::InvalidCdiExpansion)
}

pub(crate) fn extract_cdi<D: Digest, H: HmacImpl<D>>(cdi: &[u8], new_cdi: &mut [u8]) -> Result<()> {
    let kdf = Hkdf::<D, H>::new(None, cdi);
    kdf.expand(&[0u8; 0], new_cdi)
        .map_err(Error::InvalidCdiExpansion)
}

// Derive the next layer CDI from the current CDI.
// @id is the ID the next layer CDI will be bound to (e.g. "CDI_Sealing").
// @info is the HKDF expansion additional context information.
// @salt is either a fixed salt or the current layer TCI. If None is passed,
// ID_SALT will be used.
pub(crate) fn derive_next_cdi<D: Digest, H: HmacImpl<D>>(
    current_cdi: &[u8],
    id: &str,
    info: Option<&[u8]>,
    salt: Option<&[u8]>,
    next_cdi: &mut [u8],
) -> Result<()> {
    kdf::<D, H>(
        current_cdi,
        salt.unwrap_or(&ID_SALT),
        &[id.as_bytes(), info.unwrap_or(&[0u8; 0])],
        next_cdi,
    )
}

// Derive the attestation CDI from the current CDI.
pub(crate) fn derive_attestation_cdi<D: Digest, H: HmacImpl<D>>(
    cdi: &[u8],
    info: Option<&[u8]>,
    salt: Option<&[u8]>,
    cdi_attest: &mut [u8],
) -> Result<()> {
    derive_next_cdi::<D, H>(cdi, "CDI_Attestation", info, salt, cdi_attest)
}

// Derive the sealing CDI from the current CDI.
pub(crate) fn derive_sealing_cdi<D: Digest, H: HmacImpl<D>>(
    cdi: &[u8],
    info: Option<&[u8]>,
    salt: Option<&[u8]>,
    cdi_sealing: &mut [u8],
) -> Result<()> {
    derive_next_cdi::<D, H>(cdi, "CDI_Sealing", info, salt, cdi_sealing)
}

// Extract and expand a private key from a CDI.
pub(crate) fn derive_secret_key<D: Digest, H: HmacImpl<D>>(
    cdi: &[u8],
    secret_key: &mut [u8],
) -> Result<()> {
    kdf::<D, H>(cdi, &ASYM_SALT, &[b"Key_Pair"], secret_key)
}

// Extract and expand an authority ID from a public key.
// The public key should be created from a private key bound to the CDI this ID
// relates to, e.g. a private key created with `derive_secret_key()`.
pub(crate) fn derive_cdi_id<D: Digest, H: HmacImpl<D>>(
    public_key: &[u8],
    cdi_id: &mut [u8],
) -> Result<()> {
    kdf::<D, H>(public_key, &ID_SALT, &[b"CDI_ID"], cdi_id)
}
