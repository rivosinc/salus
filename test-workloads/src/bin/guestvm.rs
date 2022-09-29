// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;
use der::Decode;
use hex_literal::hex;

extern crate alloc;
extern crate test_workloads;

mod consts;

use ::attestation::{
    certificate::Certificate,
    extensions::dice::tcbinfo::{DiceTcbInfo, TCG_DICE_TCB_INFO},
    measurement::TcgPcrIndex::{RuntimePcr1, TvmPage},
    MAX_CSR_LEN,
};
use consts::*;
use s_mode_utils::abort::abort;
use s_mode_utils::{print::*, sbi_console::SbiConsole};
use sbi::api::{attestation, base, reset, tee_guest};

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

#[cfg(target_feature = "v")]
pub fn print_vector_csrs() {
    let mut vl: u64;
    let mut vcsr: u64;
    let mut vtype: u64;
    let mut vlenb: u64;

    unsafe {
        // safe because we are only reading csr's
        asm!(
            "csrrs {vl}, vl, zero",
            "csrrs {vcsr}, vcsr, zero",
            "csrrs {vtype}, vtype, zero",
            "csrrs {vlenb}, vlenb, zero",
            vlenb = out(reg) vlenb,
            vl= out(reg) vl,
            vcsr = out(reg) vcsr,
            vtype = out(reg) vtype,
            options(nostack)
        );
    }
    println!("vl    {}", vl);
    println!("vlenb {}", vlenb);
    println!("vcsr  {:#x}", vcsr);
    println!("vtype {:#x}", vtype);
}

#[cfg(target_feature = "v")]
pub fn test_vector() {
    let vec_len: u64 = 8;
    let vtype: u64 = 0xda;
    let enable: u64 = 0x200;

    unsafe {
        // safe because we are only setting a bit in a csr
        asm!(
            "csrrs zero, sstatus, {enable}",
            enable = in(reg) enable,
        );
    }
    println!("Vectors Enabled");
    print_vector_csrs();

    unsafe {
        // safe because we are only setting the vector csr's
        asm!(
            "vsetvl x0, {vec_len}, {vtype}",
            vec_len = in(reg) vec_len,
            vtype = in(reg) vtype,
            options(nostack),
        )
    }

    print_vector_csrs();

    const REG_WIDTH_IN_U64S: usize = 4;

    let mut inbuf = [0_u64; (32 * REG_WIDTH_IN_U64S)];
    let mut refbuf = [0_u64; (32 * REG_WIDTH_IN_U64S)];
    for i in 0..inbuf.len() {
        inbuf[i] = 1 << (i % 53) as u64;
        refbuf[i] = inbuf[i];
    }

    let bufp1 = inbuf.as_ptr();
    let bufp2: *const u64;
    let bufp3: *const u64;
    let bufp4: *const u64;
    unsafe {
        // safe because we don't go past the length of inbuf
        bufp2 = bufp1.add(8 * REG_WIDTH_IN_U64S);
        bufp3 = bufp1.add(16 * REG_WIDTH_IN_U64S);
        bufp4 = bufp1.add(24 * REG_WIDTH_IN_U64S);
    }
    println!("Loading vector registers");
    unsafe {
        // safe because the assembly reads into the vector register file
        asm!(
            "vl8r.v  v0, ({bufp1})",
            "vl8r.v  v8, ({bufp2})",
            "vl8r.v  v16, ({bufp3})",
            "vl8r.v  v24, ({bufp4})",
            bufp1 = in(reg) bufp1,
            bufp2 = in(reg) bufp2,
            bufp3 = in(reg) bufp3,
            bufp4 = in(reg) bufp4,
            options(nostack)
        );
    }

    print_vector_csrs();

    // Overwrite memory to verify that it changes
    for i in 0..inbuf.len() {
        inbuf[i] = 99_u64;
    }

    println!("Reading vector registers");
    unsafe {
        // safe because enough memory provided to store entire register file
        asm!(
            "vs8r.v  v0, ({bufp1})",
            "vs8r.v  v8, ({bufp2})",
            "vs8r.v  v16, ({bufp3})",
            "vs8r.v  v24, ({bufp4})",
            bufp1 = in(reg) bufp1,
            bufp2 = in(reg) bufp2,
            bufp3 = in(reg) bufp3,
            bufp4 = in(reg) bufp4,
            options(nostack)
        )
    }
    print_vector_csrs();

    println!("Verify registers");
    let mut should_panic = false;
    for i in 0..inbuf.len() {
        if inbuf[i] != refbuf[i] {
            println!("error:  {} {} {}", i, refbuf[i], inbuf[i]);
            should_panic = true;
        }
    }

    if should_panic {
        panic!("Vector registers did not restore correctly");
    }
}

/// Test `CertReq` encoded as ASN.1 DER
const TEST_CSR: &[u8] = include_bytes!("test-ed25519.der");

fn test_attestation() {
    if base::probe_sbi_extension(sbi::EXT_ATTESTATION).is_err() {
        println!("Platform doesn't support attestation extension");
        return;
    }

    let caps = attestation::get_capabilities().expect("Failed to get attestation capabilities");
    println!(
        "caps: SVN {:#x} Evidence formats {:?}",
        caps.tcb_svn, caps.evidence_formats
    );
    if !caps
        .evidence_formats
        .contains(sbi::EvidenceFormat::DiceTcbInfo)
    {
        panic!("DICE TcbInfo format is not supported")
    }

    // SHA384 for "helloworld"
    let digest = hex!("97982a5b1414b9078103a1c008c4e3526c27b41cdbcf80790560a40f2a9bf2ed4427ab1428789915ed4b3dc07c454bd9");
    attestation::extend_measurement(&digest, RuntimePcr1 as usize)
        .expect("Failed to extend runtime PCR #1");

    let pcr_read =
        attestation::read_measurement(RuntimePcr1 as usize).expect("Failed to read runtime PCR #1");
    println!("Runtime PCR #1 measurement: {:x?}", pcr_read.as_slice());

    // SHA384 for (RuntimePCR1 || SHA384(b"helloworld"))
    let expected_pcr = hex!("4d783cdfa8d6bc1293d0f1bfb58f9f0b05a8ef723cb8745d142d60ac3b9d213c51f06aa1e9b92ff09f64e4a2d0c8fe87");
    if pcr_read.as_slice() != expected_pcr {
        panic!("Wrong runtime PCR #1 measurement")
    }

    if TEST_CSR.len() > MAX_CSR_LEN as usize {
        panic!("Test CSR is too large")
    }

    let request_data = [0u8; sbi::EVIDENCE_DATA_BLOB_SIZE];
    let cert_bytes = match attestation::get_evidence(
        TEST_CSR,
        &request_data,
        sbi::EvidenceFormat::DiceTcbInfo,
    ) {
        Err(e) => {
            println!("Attestation error {e:?}");
            panic!("Guest evidence call failed");
        }
        Ok(cert_bytes) => cert_bytes,
    };

    println!(
        "Evidence certificate is at 0x{:x} - len {}",
        cert_bytes.as_ptr() as u64,
        cert_bytes.len()
    );

    let mut tcb_info_extn = DiceTcbInfo::default();
    let cert = Certificate::from_der(&cert_bytes.as_slice()).expect("Cert parsing error");

    // Look for a DiceTcbInfo extension
    if let Some(extensions) = cert.tbs_certificate.extensions.as_ref() {
        for extn in extensions.iter() {
            if extn.extn_id != TCG_DICE_TCB_INFO {
                continue;
            }

            tcb_info_extn = DiceTcbInfo::from_der(extn.extn_value).expect("Invalid TCB DER");
        }
    };

    // Extract the TVM pages measurement register from the list of FwIds.
    let tvm_fwid = tcb_info_extn
        .fwids
        .as_ref()
        .map(|fwids| fwids.get(TvmPage as usize).expect("Missing TVM page fwid"))
        .expect("Missing TVM fwids");

    println!(
        "Certificate version:{:?} Issuer:{} Signature algorithm:{}",
        cert.tbs_certificate.version, cert.tbs_certificate.issuer, cert.signature_algorithm.oid
    );

    println!(
        "Guest measurement:{:x?} (Hash algorithm {} - {} bytes)",
        tvm_fwid.digest.as_bytes(),
        tvm_fwid.hash_alg,
        tvm_fwid.digest.as_bytes().len()
    );
}

#[no_mangle]
#[allow(clippy::zero_ptr)]
extern "C" fn kernel_init(_hart_id: u64, shared_page_addr: u64) {
    SbiConsole::set_as_console();

    println!("*****************************************");
    println!("Hello world from Tellus guest            ");

    base::probe_sbi_extension(sbi::EXT_TEE_GUEST).expect("TEE-Guest extension not present");

    let mut next_page = USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
    test_attestation();
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

    tee_guest::add_shared_memory_region(
        GUEST_SHARED_PAGES_START_ADDRESS,
        NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
    )
    .expect("GuestVm -- AddSharedMemoryRegion failed");
    println!("Accessing shared page at 0x{shared_page_addr:x}     ");
    // Safety: We are assuming that the shared_page_addr is valid, and will be mapped in on a fault
    unsafe {
        if core::ptr::read_volatile(shared_page_addr as *const u64) == GUEST_SHARE_PING {
            // Write a known value for verification purposes
            core::ptr::write_volatile(shared_page_addr as *mut u64, GUEST_SHARE_PONG);
        }
    }

    #[cfg(target_feature = "v")]
    test_vector();

    tee_guest::add_emulated_mmio_region(GUEST_MMIO_START_ADDRESS, PAGE_SIZE_4K)
        .expect("GuestVm - AddEmulatedMmioRegion failed");
    // Try reading and writing MMIO.
    let write_ptr = GUEST_MMIO_START_ADDRESS as *mut u32;
    // Safety: write_ptr is properly aligned and a writable part of our address space.
    unsafe {
        core::ptr::write_volatile(write_ptr, 0xaabbccdd);
    }
    let read_ptr = (GUEST_MMIO_START_ADDRESS + 0x20) as *const u8;
    // Safety: read_ptr is properly aligned and a readable part of our address space.
    let val = unsafe { core::ptr::read_volatile(read_ptr) };
    println!("Host says: 0x{:x}", val);

    // Make sure we return from WFI.
    //
    // Safety: WFI behavior is well-defined.
    unsafe { asm!("wfi", options(nomem, nostack)) };

    println!("Exiting guest");
    println!("*****************************************");

    reset::reset(sbi::ResetType::Shutdown, sbi::ResetReason::NoReason)
        .expect("Guest shutdown failed");
    unreachable!();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
