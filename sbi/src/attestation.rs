// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

/// Functions provided by the attestation extension.
#[derive(Copy, Clone)]
pub enum AttestationFunction {
    /// Get an attestion evidence from a CSR (https://datatracker.ietf.org/doc/html/rfc2986).
    /// The caller passes the CSR and its length through the first 2 arguments.
    /// The third argument is the address where the generated certificate will be placed.
    /// The evidence is formatted an x.509 DiceTcbInfo certificate extension
    ///
    /// a6 = 0
    /// a0 = CSR address
    /// a1 = CSR length
    /// a2 = Generated certificate address
    /// a3 = Reserved length for the generated certificate address
    GetEvidence {
        /// a0 = CSR address
        csr_addr: u64,
        /// a1 = CSR length
        csr_len: u64,
        /// a2 = Generated Certificate address
        cert_addr: u64,
        /// a3 = Reserved length for the generated certificate address
        cert_len: u64,
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
                csr_addr: args[0],
                csr_len: args[1],
                cert_addr: args[2],
                cert_len: args[3],
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
                csr_addr: _,
                csr_len: _,
                cert_addr: _,
                cert_len: _,
            } => 0,

            ExtendMeasurement {
                measurement_addr: _,
                len: _,
            } => 1,
        }
    }

    fn a3(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                csr_addr: _,
                csr_len: _,
                cert_addr: _,
                cert_len,
            } => *cert_len,

            ExtendMeasurement {
                measurement_addr: _,
                len,
            } => *len,
        }
    }

    fn a2(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetEvidence {
                csr_addr: _,
                csr_len: _,
                cert_addr,
                cert_len: _,
            } => *cert_addr,

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
                csr_addr: _,
                csr_len,
                cert_addr: _,
                cert_len: _,
            } => *csr_len,

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
                csr_addr,
                csr_len: _,
                cert_addr: _,
                cert_len: _,
            } => *csr_addr,

            ExtendMeasurement {
                measurement_addr,
                len: _,
            } => *measurement_addr,
        }
    }
}
