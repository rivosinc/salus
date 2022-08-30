// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;
use der::Decode;

extern crate alloc;
extern crate test_workloads;

use attestation::{
    certificate::Certificate,
    extensions::dice::tcbinfo::{DiceTcbInfo, TCG_DICE_TCB_INFO},
    measurement::MeasurementIndex::TvmPage,
    MAX_CERT_LEN, MAX_CSR_LEN,
};
use s_mode_utils::abort::abort;
use s_mode_utils::ecall::ecall_send;
use s_mode_utils::{print::*, sbi_console::SbiConsole};
use sbi::api::{base, reset};
use sbi::{SbiMessage, EXT_ATTESTATION};

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
        options(nostack));
    }

    print_vector_csrs();

    // Overwrite memory to verify that it changes
    for i in 0..inbuf.len() {
        inbuf[i] = 99_u64;
    }

    println!("reading vector registers");
    unsafe {
        // safe because enough memory provided to store entire register file
        asm! {
        "vs8r.v  v0, ({bufp1})",
        "vs8r.v  v8, ({bufp2})",
        "vs8r.v  v16, ({bufp3})",
        "vs8r.v  v24, ({bufp4})",
        bufp1 = in(reg) bufp1,
        bufp2 = in(reg) bufp2,
        bufp3 = in(reg) bufp3,
        bufp4 = in(reg) bufp4,
        options(nostack)}
    }
    print_vector_csrs();

    println!("Verify registers");
    for i in 0..inbuf.len() {
        if inbuf[i] != refbuf[i] {
            println!("error:  {} {} {}", i, refbuf[i], inbuf[i]);
        }
    }
}

/// Test `CertReq` encoded as ASN.1 DER
const TEST_CSR: &[u8] = include_bytes!("test-ed25519.der");

fn test_attestation(csr_addr: u64) {
    let cert_bytes = [0u8; MAX_CERT_LEN];
    if base::probe_sbi_extension(EXT_ATTESTATION).is_err() {
        println!("Platform doesn't support attestation extension");
        return;
    }

    if TEST_CSR.len() > MAX_CSR_LEN as usize {
        panic!("Test CSR is too large")
    }

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

    let mut tcb_info_extn = DiceTcbInfo::default();
    let cert = Certificate::from_der(&cert_bytes[..cert_len as usize]).expect("Cert parsing error");

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

    print!(
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
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const NUM_GUEST_DATA_PAGES: u64 = 160;
    const NUM_GUEST_ZERO_PAGES: u64 = 10;
    const PAGE_SIZE_4K: u64 = 4096;
    const GUEST_MMIO_ADDRESS: u64 = 0x1000_8000;
    // TODO: Consider moving to a common module to ensure that the host and guest are in lockstep
    const GUEST_SHARE_PING: u64 = 0xBAAD_F00D;
    const GUEST_SHARE_PONG: u64 = 0xF00D_BAAD;

    SbiConsole::set_as_console();

    println!("*****************************************");
    println!("Hello world from Tellus guest            ");

    let mut next_page = USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
    test_attestation(next_page);
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
        if core::ptr::read_volatile(shared_page_addr as *const u64) == GUEST_SHARE_PING {
            // Write a known value for verification purposes
            core::ptr::write_volatile(shared_page_addr as *mut u64, GUEST_SHARE_PONG);
        }
    }

    #[cfg(target_feature = "v")]
    test_vector();

    // Try reading and writing MMIO.
    let write_ptr = GUEST_MMIO_ADDRESS as *mut u32;
    // Safety: write_ptr is properly aligned and a writable part of our address space.
    unsafe {
        core::ptr::write_volatile(write_ptr, 0xaabbccdd);
    }
    let read_ptr = (GUEST_MMIO_ADDRESS + 0x20) as *const u8;
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
