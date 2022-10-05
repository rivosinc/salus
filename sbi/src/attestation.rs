// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use crate::error::*;
use crate::function::*;

use flagset::{flags, FlagSet};

/// The data blob passed to the GetEvidence call must be 64 bytes long.
pub const EVIDENCE_DATA_BLOB_SIZE: usize = 64;

/// The maximum number of measurement registers in the attestation capabilities.
pub const MAX_MEASUREMENT_REGISTERS: usize = 32;

flags! {
    /// Attestation evidence formats.
    #[derive(Default)]
    #[repr(u8)]
    pub enum EvidenceFormat: u8 {
        /// Single layer DICE TCB
        /// https://trustedcomputinggroup.org/resource/dice-attestation-architecture/
        #[default]
        DiceTcbInfo = 1,
        /// X.509 extension for multiple layers DICE TCB.
        /// https://trustedcomputinggroup.org/resource/dice-attestation-architecture/
        DiceMultiTcbInfo = 2,
        /// Open DICE profile
        /// https://pigweed.googlesource.com/open-dice/+/HEAD/docs/specification.md
        OpenDice = 4,
    }
}

/// A list of supported hash algorithms.
#[derive(Copy, Clone, Default, Debug)]
#[repr(u8)]
pub enum HashAlgorithm {
    /// SHA-384
    #[default]
    Sha384 = 1,
    /// SHA-512
    Sha512 = 2,
}

impl HashAlgorithm {
    /// The hash algorithm output size, in bytes.
    pub fn size(&self) -> usize {
        match self {
            HashAlgorithm::Sha384 => 48,
            HashAlgorithm::Sha512 => 64,
        }
    }
}

/// The largest supported hash algorithm output size.
pub const MAX_HASH_SIZE: usize = 64;

/// Attestation Capabilities
///
/// This structure exposes the supported attestation capabilities to the SBI
/// GetCapabilities caller. It lets the caller know which hash algorithms,
/// evidence formats, and measurements mappings the SBI implementation supports.
#[repr(C)]
#[derive(Default)]
pub struct AttestationCapabilities {
    /// The TCB Secure Version Number.
    pub tcb_svn: u64,
    /// The supported hash algorithm.
    pub hash_algorithm: HashAlgorithm,
    /// The supported evidence formats. This is a bitmap.
    pub evidence_formats: FlagSet<EvidenceFormat>,
    /// Number of static measurement registers.
    pub static_measurements: u8,
    /// Number of runtime measurement registers.
    pub runtime_measurements: u8,
    /// Array of all measurement register descriptors.
    pub measurement_registers: [MeasurementRegisterDescriptor; MAX_MEASUREMENT_REGISTERS],
}

impl AttestationCapabilities {
    /// Create new attestation capabilities structure.
    pub fn new(
        tcb_svn: u64,
        hash_algorithm: HashAlgorithm,
        evidence_formats: impl Into<FlagSet<EvidenceFormat>>,
        static_measurements: u8,
        runtime_measurements: u8,
    ) -> Self {
        AttestationCapabilities {
            tcb_svn,
            hash_algorithm,
            evidence_formats: evidence_formats.into(),
            static_measurements,
            runtime_measurements,
            ..Default::default()
        }
    }

    /// Add a measurement register to the attestation capabilities.
    pub fn add_measurement_register(
        &mut self,
        register: MeasurementRegisterDescriptor,
        index: usize,
    ) -> Result<&mut AttestationCapabilities> {
        // We do not allow for a sparse measurement register array.
        if index + 1 > (self.static_measurements + self.runtime_measurements) as usize {
            return Err(Error::Failed);
        }

        *self
            .measurement_registers
            .get_mut(index)
            .ok_or(Error::Failed)? = register;

        Ok(self)
    }
}

/// Measurement register descriptor.
///
/// This structure describes an attestation measurement register.
/// The AttestationCapabilities structure includes an array of those descriptors
/// for all the supported measurement registers.
#[repr(C)]
#[derive(Default)]
pub struct MeasurementRegisterDescriptor {
    tcb_layer_index: u8,
    fwid_index: u8,
    tcg_pcr_index: u8,
    runtime: bool,
}

impl MeasurementRegisterDescriptor {
    /// Create a new measurement register descriptor.
    pub fn new(tcb_layer_index: u8, fwid_index: u8, tcg_pcr_index: u8, runtime: bool) -> Self {
        MeasurementRegisterDescriptor {
            tcb_layer_index,
            fwid_index,
            tcg_pcr_index,
            runtime,
        }
    }
}

/// Functions provided by the attestation extension.
#[derive(Copy, Clone, Debug)]
pub enum AttestationFunction {
    /// Get the SBI implementation attestation capabilities.
    /// The attestation capabilities let the SBI implementations expose which
    /// hash algorithm is being used for measurements, which evidence formats
    /// are supported. The attestation capabilities structure also contains a
    /// map of all  measurement registers.
    ///
    /// a6 = 0
    /// a0 = Attestation capabilities buffer
    /// a1 = Attestation capabilities buffer size
    GetCapabilities {
        /// a0 = Capabilities structure address
        caps_addr_out: u64,
        /// a1 = Capabilities structure length
        caps_size: u64,
    },

    /// Get an attestion evidence from a Certificate Signing Request (CSR)
    /// <https://datatracker.ietf.org/doc/html/rfc2986>.
    /// The caller passes the CSR and its length through the first 2 arguments.
    /// The third argument is the address where the caller places a data blob
    /// that will be included in the generated certificate. Typically, this is a
    /// cryptographic nonce.
    /// The fourth argument is the evidence format: DiceTcbInfo (0),
    /// DiceMultiTcbInfo (1) or OpenDice (2).
    /// The fifthh argument is the address where the generated certificate will be placed.
    /// The evidence is formatted an x.509 DiceTcbInfo certificate extension
    ///
    /// a6 = 1
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

    /// Extend a measurement register with an additional measurement.
    /// The first parameter is the address of the measurement buffer.
    /// The second argument is the length of the measurement buffer, which must
    /// be the same as the hash algorithm size reported by `GetCapabilities`.
    /// The third parameter is the measurement register index, and it must be
    /// one of the reported 'TCG_PCR_INDEX` from the runtime measurement
    /// registers array.
    ///
    /// This function is not supported if the SBI implementation does not
    /// support runtime measurements (i.e. `NUM_RMSMT_REGS` as reported by the
    /// `GetCapabilities` function is set to 0).
    ///
    /// a6 = 2
    /// a0 = Measurement data buffer address
    /// a1 = Measurement data buffer length
    /// a2 = Measurement register index.
    ExtendMeasurement {
        /// a0 = measurement data buffer address
        measurement_data_addr: u64,
        /// a1 = measurement data buffer length
        measurement_data_size: u64,
        /// a2 = measurement register index
        measurement_index: u64,
    },

    /// Read a measurement register data back.
    /// The first parameter is the address of the measurement buffer allocated
    /// by the caller, for the SBI implementation to write the measurement data
    /// into.
    /// The second argument is the length of the measurement buffer, which must
    /// be at least as large as the hash algorithm size reported by
    /// `GetCapabilities`.
    /// The third parameter is the measurement register index, and it must be
    /// one of the reported 'TCG_PCR_INDEX` from the runtime measurement
    /// registers array.
    /// The returned value is the length of the measurement data.
    ///
    /// a6 = 3
    /// a0 = Measurement data buffer address
    /// a1 = Measurement data buffer length
    /// a2 = Measurement register index.
    ReadMeasurement {
        /// a0 = measurement data buffer address
        measurement_data_addr_out: u64,
        /// a1 = measurement data buffer length
        measurement_data_size: u64,
        /// a2 = measurement register index
        measurement_index: u64,
    },
}

impl AttestationFunction {
    /// Attempts to parse `Self` from the passed in `a0-a7`.
    pub(crate) fn from_regs(args: &[u64]) -> Result<Self> {
        use AttestationFunction::*;
        match args[6] {
            0 => Ok(GetCapabilities {
                caps_addr_out: args[0],
                caps_size: args[1],
            }),

            1 => Ok(GetEvidence {
                cert_request_addr: args[0],
                cert_request_size: args[1],
                request_data_addr: args[2],
                evidence_format: args[3],
                cert_addr_out: args[4],
                cert_size: args[5],
            }),

            2 => Ok(ExtendMeasurement {
                measurement_data_addr: args[0],
                measurement_data_size: args[1],
                measurement_index: args[2],
            }),

            3 => Ok(ReadMeasurement {
                measurement_data_addr_out: args[0],
                measurement_data_size: args[1],
                measurement_index: args[2],
            }),

            _ => Err(Error::InvalidParam),
        }
    }
}

impl SbiFunction for AttestationFunction {
    fn a6(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetCapabilities {
                caps_addr_out: _,
                caps_size: _,
            } => 0,

            GetEvidence {
                cert_request_addr: _,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => 1,

            ExtendMeasurement {
                measurement_data_addr: _,
                measurement_data_size: _,
                measurement_index: _,
            } => 2,

            ReadMeasurement {
                measurement_data_addr_out: _,
                measurement_data_size: _,
                measurement_index: _,
            } => 3,
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
                measurement_data_addr: _,
                measurement_data_size: _,
                measurement_index,
            } => *measurement_index,

            ReadMeasurement {
                measurement_data_addr_out: _,
                measurement_data_size: _,
                measurement_index,
            } => *measurement_index,

            _ => 0,
        }
    }

    fn a1(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetCapabilities {
                caps_addr_out: _,
                caps_size,
            } => *caps_size,

            GetEvidence {
                cert_request_addr: _,
                cert_request_size,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => *cert_request_size,

            ExtendMeasurement {
                measurement_data_addr: _,
                measurement_data_size,
                measurement_index: _,
            } => *measurement_data_size,

            ReadMeasurement {
                measurement_data_addr_out: _,
                measurement_data_size,
                measurement_index: _,
            } => *measurement_data_size,
        }
    }

    fn a0(&self) -> u64 {
        use AttestationFunction::*;
        match self {
            GetCapabilities {
                caps_addr_out,
                caps_size: _,
            } => *caps_addr_out,

            GetEvidence {
                cert_request_addr,
                cert_request_size: _,
                request_data_addr: _,
                evidence_format: _,
                cert_addr_out: _,
                cert_size: _,
            } => *cert_request_addr,

            ExtendMeasurement {
                measurement_data_addr,
                measurement_data_size: _,
                measurement_index: _,
            } => *measurement_data_addr,

            ReadMeasurement {
                measurement_data_addr_out,
                measurement_data_size: _,
                measurement_index: _,
            } => *measurement_data_addr_out,
        }
    }
}
