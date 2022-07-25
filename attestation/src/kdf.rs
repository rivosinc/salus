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

pub(crate) fn tsm_cdi<D: Digest, H: HmacImpl<D>>(cdi: &[u8], tsm_cdi: &mut [u8]) -> Result<()> {
    let kdf = Hkdf::<D, H>::new(None, cdi);
    kdf.expand(&[0u8; 0], tsm_cdi)
        .map_err(Error::InvalidCdiExpansion)
}

// Derive a TVM specific Attestation CDI from the TSM CDI.
pub(crate) fn tvm_attest_cdi<D: Digest, H: HmacImpl<D>>(
    cdi: &[u8],
    vm_id: u64,
    cdi_attest: &mut [u8],
) -> Result<()> {
    kdf::<D, H>(
        cdi,
        &ID_SALT,
        &[b"CDI_Attest", &vm_id.to_le_bytes()],
        cdi_attest,
    )
}

// Extract and expand a private key from the Attestation CDI
pub(crate) fn attest_secret_key<D: Digest, H: HmacImpl<D>>(
    cdi_attest: &[u8],
    secret_key: &mut [u8],
) -> Result<()> {
    kdf::<D, H>(cdi_attest, &ASYM_SALT, &[b"Key_Pair"], secret_key)
}

// Extract and expand an authority ID from the a ED25519 public key
pub(crate) fn tvm_attest_cdi_id<D: Digest, H: HmacImpl<D>>(
    public_key: &[u8],
    cdi_id: &mut [u8],
) -> Result<()> {
    kdf::<D, H>(public_key, &ID_SALT, &[b"CDI_ID"], cdi_id)
}
