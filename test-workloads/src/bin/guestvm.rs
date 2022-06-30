// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};
use der::Decode;

extern crate alloc;
extern crate test_workloads;

use attestation::{certificate::Certificate, MAX_CERT_LEN, MAX_CSR_LEN};
use s_mode_utils::abort::abort;
use s_mode_utils::ecall::ecall_send;
use s_mode_utils::print_sbi::*;
use sbi::SbiMessage;

// Dummy global allocator - panic if anything tries to do an allocation.
struct GeneralGlobalAlloc;

unsafe impl GlobalAlloc for GeneralGlobalAlloc {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        abort()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        abort()
    }
}

#[global_allocator]
static GENERAL_ALLOCATOR: GeneralGlobalAlloc = GeneralGlobalAlloc;

#[alloc_error_handler]
pub fn alloc_error(_layout: Layout) -> ! {
    abort()
}

/// Test `CertReq` encoded as ASN.1 DER
const TEST_CSR: &[u8] = include_bytes!("test-ed25519.der");

#[no_mangle]
#[allow(clippy::zero_ptr)]
extern "C" fn kernel_init(_hart_id: u64, shared_page_addr: u64) {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const NUM_GUEST_DATA_PAGES: u64 = 160;
    const NUM_GUEST_ZERO_PAGES: u64 = 10;
    const PAGE_SIZE_4K: u64 = 4096;

    let mut next_page = USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
    let measurement_page_addr = next_page;
    let msg = SbiMessage::Measurement(sbi::MeasurementFunction::GetSelfMeasurement {
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });
    next_page += PAGE_SIZE_4K;

    if TEST_CSR.len() > MAX_CSR_LEN as usize {
        panic!("Test CSR is too large")
    }

    let csr_addr = next_page;
    let cert_bytes = [0u8; MAX_CERT_LEN];

    // Safety: csr_addr is the unique reference to the CSR page.
    unsafe {
        core::ptr::copy(TEST_CSR.as_ptr(), csr_addr as *mut u8, TEST_CSR.len());
    }
    let attestation_msg = SbiMessage::Attestation(sbi::AttestationFunction::GetEvidence {
        csr_addr,
        csr_len: TEST_CSR.len() as u64,
        cert_addr: cert_bytes.as_ptr() as u64,
        cert_len: cert_bytes.len() as u64,
    });
    next_page += PAGE_SIZE_4K;

    println!("*****************************************");
    println!("Hello world from Tellus guest            ");

    // Safety: msg contains a unique reference to the measurement page and SBI is safe to write to
    // that page.
    match unsafe { ecall_send(&msg) } {
        Err(e) => {
            println!("Guest measurement error {e:?}");
            panic!("Guest measurement call failed");
        }
        Ok(_) => {
            let measurement =
                unsafe { core::ptr::read_volatile(measurement_page_addr as *const u64) };
            println!("Guest measurement was {measurement:x}");
        }
    }

    // Safety: msg contains a unique reference to the CSR and certificate pages
    // and SBI is safe to write to that page.
    let cert_len = match unsafe { ecall_send(&attestation_msg) } {
        Err(e) => {
            println!("Attestation error {e:?}");
            panic!("Guest evidence call failed");
        }
        Ok(cert_len) => cert_len,
    };

    println!(
        "Evidence certificate is at 0x{:x} - len {}",
        cert_bytes.as_ptr() as u64,
        cert_len
    );

    // This is mostly as a good practice, because the TSM implementation
    // should return an error if the provided certificate buffer is too small.
    if cert_len > cert_bytes.len() as u64 {
        panic!("Generated certificate is too large")
    }

    match Certificate::from_der(&cert_bytes[..cert_len as usize]) {
        Err(e) => {
            println!("Attestation error {e:?}");
            panic!("Cert parsing error");
        }
        Ok(cert) => {
            println!(
                "Certificate version:{:?} Issuer:{} Signature algorithm:{}",
                cert.tbs_certificate.version,
                cert.tbs_certificate.issuer,
                cert.signature_algorithm.oid
            );
        }
    }

    // Touch the rest of the data pages to force Tellus to fault them in.
    let end =
        USABLE_RAM_START_ADDRESS + (NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES) * PAGE_SIZE_4K;
    while next_page < end {
        let ptr = (next_page + 2 * core::mem::size_of::<u64>() as u64) as *mut u64;
        // Safety: next_page is properly aligned and should be a writable part of our address space.
        unsafe {
            core::ptr::write(ptr, 0xdeadbeef);
            let val = core::ptr::read(ptr);
            assert_eq!(val, 0xdeadbeef);
        }
        next_page += PAGE_SIZE_4K;
    }

    println!("Accessing shared page at 0x{shared_page_addr:x}     ");
    // Safety: We are assuming that the shared_page_addr is valid, and will be mapped in on a fault
    unsafe {
        if core::ptr::read_volatile(shared_page_addr as *const u64) == shared_page_addr + 1 {
            // Write a known value for verification purposes
            core::ptr::write_volatile(shared_page_addr as *mut u64, shared_page_addr);
        }
    }

    println!("Exiting guest by causing a fault         ");
    println!("*****************************************");

    // TODO: Implement mechanism to gracefully exit guest
    // Not safe, but deliberately intended to cause a fault
    unsafe {
        core::ptr::read_volatile(0 as *const u64);
    }
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
