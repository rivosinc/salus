// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};
use core::mem::size_of;

extern crate alloc;
extern crate test_workloads;

use attestation::{MAX_CERT_LEN, MAX_CSR_LEN};
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
extern "C" fn kernel_init() {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const NUM_GUEST_DATA_PAGES: u64 = 10;
    const PAGE_SIZE_4K: u64 = 4096;

    let measurement_page_addr = USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
    let msg = SbiMessage::Measurement(sbi::MeasurementFunction::GetSelfMeasurement {
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });

    if TEST_CSR.len() > MAX_CSR_LEN as usize {
        panic!("Test CSR is too large")
    }
    let csr_addr = measurement_page_addr + PAGE_SIZE_4K;
    // Align the cert address
    let csr_size_aligned = (TEST_CSR.len() + size_of::<u64>() - 1) & !(size_of::<u64>() - 1);
    let cert_addr = csr_addr + csr_size_aligned as u64;

    // Safety: csr_addr is the unique reference to the CSR page.
    unsafe {
        core::ptr::copy(TEST_CSR.as_ptr(), csr_addr as *mut u8, TEST_CSR.len());
    }
    let attestation_msg = SbiMessage::Attestation(sbi::AttestationFunction::GetEvidence {
        csr_addr,
        csr_len: TEST_CSR.len() as u64,
        cert_addr,
        cert_len: MAX_CERT_LEN as u64,
    });

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
    match unsafe { ecall_send(&attestation_msg) } {
        Err(e) => {
            println!("Attestation error {e:?}");
            panic!("Guest evidence call failed");
        }
        Ok(cert_len) => {
            println!("Evidence cert is at 0x{:x} - len {}", cert_addr, cert_len);
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
