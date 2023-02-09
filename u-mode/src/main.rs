// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

#![no_std]
#![no_main]

//! # Salus U-mode binary.
//!
//! This is Salus U-mode code. It is used to offload functionalities
//! of the hypervisor in user mode. There's a copy of this task in
//! each CPU.
//!
//! The task it's based on a request loop. This task can be reset at
//! any time by the hypervisor, so it shouldn't hold non-recoverable
//! state.

extern crate libuser;

use data_model::{VolatileMemory, VolatileSlice};
use libuser::*;
use test_system::*;
use u_mode_api::{Error as UmodeApiError, UmodeRequest};

mod cert;

// Dummy global allocator - panic if anything tries to do an allocation.
struct GeneralGlobalAlloc;

unsafe impl core::alloc::GlobalAlloc for GeneralGlobalAlloc {
    unsafe fn alloc(&self, _layout: core::alloc::Layout) -> *mut u8 {
        panic!("alloc called!");
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {
        panic!("dealloc called!");
    }
}

// Global allocator linking (not usage) required by rice dependencies.
#[global_allocator]
static GENERAL_ALLOCATOR: GeneralGlobalAlloc = GeneralGlobalAlloc;

struct UmodeTask {
    vslice: VolatileSlice<'static>,
}

impl UmodeTask {
    // Get an attestation evidence.
    // This function returns a serialized, DER formatted X.509 certificate.
    // The attestation evidence is included as a certificate extension.
    //
    // Arguments:
    //   csr_addr: starting address of the input Certificate Signing Request.
    //   csr_len: size of the input Certificate Signing Request.
    //   certout_addr: starting address of the output Certificate.
    //   certout_len: size for the output Certificate.
    //
    // U-mode Shared Region: contains an instance of `GetEvidenceShared`.
    fn op_get_evidence(
        &self,
        csr_addr: u64,
        csr_len: usize,
        certout_addr: u64,
        certout_len: usize,
    ) -> Result<u64, UmodeApiError> {
        // Safety: we trust the hypervisor to have mapped at `csr_addr` `csr_len` bytes for reading.
        let csr = unsafe { &*core::ptr::slice_from_raw_parts(csr_addr as *const u8, csr_len) };
        // Safety: we trust the hypervisor to have mapped at `certout_addr` `certout_len` bytes valid
        // for reading and writing.
        let certout = unsafe {
            &mut *core::ptr::slice_from_raw_parts_mut(certout_addr as *mut u8, certout_len)
        };
        let shared_data = self
            .vslice
            .get_ref(0)
            .map_err(|_| UmodeApiError::Failed)?
            .load();
        cert::get_certificate_sha384(csr, shared_data, certout).map_err(|e| {
            println!("get_certificate failed: {:?}", e);
            use cert::Error::*;
            match e {
                CsrBufferTooSmall(_, _)
                | CsrParseFailed(_)
                | CsrVerificationFailed(_)
                | CertificateBufferTooSmall(_, _) => UmodeApiError::InvalidArgument,
                _ => UmodeApiError::Failed,
            }
        })
    }

    // Run the main loop, receiving requests from the hypervisor and executing them.
    fn run_loop(&self) -> ! {
        let mut res = Ok(0);
        loop {
            // Return result and wait for next operation.
            let req = hyp_nextop(res);
            res = match req {
                Ok(req) => match req {
                    UmodeRequest::Nop => Ok(0),
                    UmodeRequest::GetEvidence {
                        csr_addr,
                        csr_len,
                        certout_addr,
                        certout_len,
                    } => self.op_get_evidence(csr_addr, csr_len, certout_addr, certout_len),
                },
                Err(err) => Err(err),
            };
        }
    }
}

#[no_mangle]
extern "C" fn task_main(cpuid: u64, shared_addr: u64, shared_size: u64) -> ! {
    // Safety: we trust the hypervisor to have mapped an area of memory starting at `shared_addr`
    // valid for at least `shared_size` bytes.
    let vslice =
        unsafe { VolatileSlice::from_raw_parts(shared_addr as *mut u8, shared_size as usize) };
    let task = UmodeTask { vslice };
    println!(
        "umode/#{}: U-mode Shared Region: {:016x} - {} bytes",
        cpuid, shared_addr, shared_size
    );
    test_declare_pass!("umode start", cpuid);
    task.run_loop()
}
