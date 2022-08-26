// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// The data blob passed to the GetEvidence call must be 64 bytes long.
pub const EVIDENCE_DATA_BLOB_SIZE: usize = 64;

/// Attestation evidence formats.
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
pub enum EvidenceFormat {
    /// Single layer DICE TCB
    /// https://trustedcomputinggroup.org/resource/dice-attestation-architecture/
    DiceTcbInfo = 0,

    /// X.509 extension for multiple layers DICE TCB.
    /// https://trustedcomputinggroup.org/resource/dice-attestation-architecture/
    DiceMultiTcbInfo = 1,

    /// Open DICE profile
    /// https://pigweed.googlesource.com/open-dice/+/HEAD/docs/specification.md
    OpenDice = 2,
}

/// Functions provided by the attestation extension.
#[derive(Copy, Clone)]
pub enum AttestationFunction {
    /// Get an attestion evidence from a Certificate Signing Request (CSR)
    /// (https://datatracker.ietf.org/doc/html/rfc2986).
    /// The caller passes the CSR and its length through the first 2 arguments.
    /// The third argument is the address where the caller places a data blob
    /// that will be included in the generated certificate. Typically, this is a
    /// cryptographic nonce.
    /// The fourth argument is the evidence format: DiceTcbInfo (0),
    /// DiceMultiTcbInfo (1) or OpenDice (2).
    /// The fifthh argument is the address where the generated certificate will be placed.
    /// The evidence is formatted an x.509 DiceTcbInfo certificate extension
    ///
    /// a6 = 0
    /// a0 = CSR address
    /// a1 = CSR length
    /// a2 = Data blob address
    /// a3 = Attestation evidence format
    /// a4 = Generated certificate address
    /// a5 = Reserved length for the generated certificate address
    GetEvidence {
        /// a0 = CSR address
        cert_request_addr: u64,
        /// a1 = CSR length
        cert_request_size: u64,
        /// a2 = User data blob
        request_data_addr: u64,
        /// a3 = Attestation evidence format
        evidence_format: u64,
        /// a4 = Generated Certificate address
        cert_addr_out: u64,
        /// a5 = Reserved length for the generated certificate address
        cert_size: u64,
    },

    /// Extend the TCB measurement with an additional measurement.
    /// TBD: Do we allow for a specific PCR index to be passed, or do we extend
    /// one dedicated PCR with all runtime extended measurements?
    ///
    /// a6 = 0
    /// a0 = Measurement entry address
    /// a1 = Measurement entry length
    ExtendMeasurement {
        /// a0 = measurement address
        measurement_addr: u64,
        /// a1 = measurement length
        len: u64,
    },
}

impl AttestationFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use AttestationFunction::*;
        match args[6] {
            0 => Ok(GetEvidence {
                cert_request_addr: args[0],
                cert_request_size: args[1],
                request_data_addr: args[2],
                evidence_format: args[3],
                cert_addr_out: args[4],
                cert_size: args[5],
            }),

            1 => Ok(ExtendMeasurement {
                measurement_addr: args[0],
                len: args[1],
            }),

            _ => Err(Error::InvalidParam),
        }
    }
}

impl SbiFunction for AttestationFunction {
    fn a6(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr: _,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => 0,

            ExtendMeasurement {
                measurement_addr: _,
                len: _,
            } => 1,
        }
    }

    fn a5(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr: _,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size,
            } => *cert_size,
            _ => 0,
        }
    }

    fn a4(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr: _,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out,
                cert_size: _,
            } => *cert_addr_out,
            _ => 0,
        }
    }

    fn a3(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr: _,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format,
                cert_addr_out: _,
                cert_size: _,
            } => *evidence_format,
            _ => 0,
        }
    }

    fn a2(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr: _,
                cert_request_size: _,
                request_data_addr,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => *request_data_addr,

            ExtendMeasurement {
                measurement_addr: _,
                len,
            } => *len,
        }
    }

    fn a1(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr: _,
                cert_request_size,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => *cert_request_size,

            ExtendMeasurement {
                measurement_addr: _,
                len,
            } => *len,
        }
    }

    fn a0(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                cert_request_addr,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => *cert_request_addr,

            ExtendMeasurement {
                measurement_addr,
                len: _,
            } => *measurement_addr,
        }
    }
}
