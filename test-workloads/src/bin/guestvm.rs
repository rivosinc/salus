// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]
#![allow(missing_docs)]

use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;
use der::Decode;
use hex_literal::hex;

extern crate alloc;
extern crate test_workloads;

use ::attestation::TcgPcrIndex::{RuntimePcr1, TvmPage};
use rice::x509::{
    certificate::Certificate,
    extensions::dice::tcbinfo::{DiceTcbInfo, TCG_DICE_TCB_INFO},
    MAX_CSR_LEN,
};
use riscv_regs::{sie, stopi, Interrupt, Readable, RiscvCsrInterface, Writeable, CSR};
use s_mode_utils::abort::abort;
use s_mode_utils::{print::*, sbi_console::SbiConsole};
use sbi_rs::api::{attestation, base, cove_guest, reset};
use test_system::*;
use test_workloads::consts::*;

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

pub fn print_vector_csrs() {
    let mut vl: u64;
    let mut vcsr: u64;
    let mut vtype: u64;
    let mut vlenb: u64;

    unsafe {
        // safe because we are only reading csr's
        asm!(
            ".option push",
            ".option arch, +v",
            "csrrs {vl}, vl, zero",
            "csrrs {vcsr}, vcsr, zero",
            "csrrs {vtype}, vtype, zero",
            "csrrs {vlenb}, vlenb, zero",
            ".option pop",
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

pub fn test_vector() -> TestResult {
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
            ".option push",
            ".option arch, +v",
            "vsetvl x0, {vec_len}, {vtype}",
            ".option pop",
            vec_len = in(reg) vec_len,
            vtype = in(reg) vtype,
            options(nostack),
        )
    }

    print_vector_csrs();

    const REG_WIDTH_IN_U64S: usize = 4;

    // write arbitrary bytes to  to vector registers
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
            ".option push",
            ".option arch, +v",
            "vl8r.v  v0, ({bufp1})",
            "vl8r.v  v8, ({bufp2})",
            "vl8r.v  v16, ({bufp3})",
            "vl8r.v  v24, ({bufp4})",
            ".option pop",
            bufp1 = in(reg) bufp1,
            bufp2 = in(reg) bufp2,
            bufp3 = in(reg) bufp3,
            bufp4 = in(reg) bufp4,
            options(nostack)
        );
    }

    print_vector_csrs();

    // Overwrite memory to verify that it changes
    for elem in &mut inbuf {
        *elem = 99_u64;
    }

    println!("Reading vector registers");
    unsafe {
        // safe because enough memory provided to store entire register file
        asm!(
            ".option push",
            ".option arch, +v",
            "vs8r.v  v0, ({bufp1})",
            "vs8r.v  v8, ({bufp2})",
            "vs8r.v  v16, ({bufp3})",
            "vs8r.v  v24, ({bufp4})",
            ".option pop",
            bufp1 = in(reg) bufp1,
            bufp2 = in(reg) bufp2,
            bufp3 = in(reg) bufp3,
            bufp4 = in(reg) bufp4,
            options(nostack)
        )
    }
    print_vector_csrs();

    println!("Verify registers");
    let mut failed = false;
    for i in 0..inbuf.len() {
        if inbuf[i] != refbuf[i] {
            println!("error:  {} {} {}", i, refbuf[i], inbuf[i]);
            failed = true;
        }
    }

    if failed {
        println!("Vector registers did not restore correctly");
        Err(TestFailure::Fail)
    } else {
        Ok(())
    }
}

/// Test `CertReq` encoded as ASN.1 DER
const TEST_CSR: &[u8] = include_bytes!("test-ed25519.der");

fn test_attestation() -> TestResult {
    if base::probe_sbi_extension(sbi_rs::EXT_ATTESTATION).is_err() {
        println!("Platform doesn't support attestation extension");
        return Err(TestFailure::Fail);
    }

    let caps = attestation::get_capabilities().expect("Failed to get attestation capabilities");
    println!(
        "caps: SVN {:#x} Evidence formats {:?}",
        caps.tcb_svn, caps.evidence_formats
    );
    if !caps
        .evidence_formats
        .contains(sbi_rs::EvidenceFormat::DiceTcbInfo)
    {
        println!("DICE TcbInfo format is not supported");
        return Err(TestFailure::Fail);
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
    let result = pcr_read.as_slice() != expected_pcr;
    test_assert!(!result, "pcr attestation measurement");
    if pcr_read.as_slice() != expected_pcr {
        println!("Wrong runtime PCR #1 measurement");
        return Err(TestFailure::Fail);
    }

    if TEST_CSR.len() > MAX_CSR_LEN {
        println!("Test CSR is too large");
        return Err(TestFailure::Fail);
    }

    let request_data = [0u8; sbi_rs::EVIDENCE_DATA_BLOB_SIZE];
    let cert_bytes = match attestation::get_evidence(
        TEST_CSR,
        &request_data,
        sbi_rs::EvidenceFormat::DiceTcbInfo,
    ) {
        Err(e) => {
            println!("Attestation error {e:?}");
            println!("Guest evidence call failed");
            return Err(TestFailure::Fail);
        }
        Ok(cert_bytes) => cert_bytes,
    };

    println!(
        "Evidence certificate is at 0x{:x} - len {}",
        cert_bytes.as_ptr() as u64,
        cert_bytes.len()
    );

    let mut tcb_info_extn = DiceTcbInfo::default();
    let cert = Certificate::from_der(cert_bytes.as_slice()).expect("Cert parsing error");

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

    Ok(())
}

fn test_memory_sharing() -> TestResult {
    // Convert some of our memory to shared.
    //
    // Safety: We haven't touched this memory and we won't touch it until the call returns.
    unsafe {
        cove_guest::share_memory(
            GUEST_SHARED_PAGES_START_ADDRESS,
            NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
        )
        .expect("GuestVm -- ShareMemory failed");
    }
    println!("Accessing shared page at 0x{GUEST_SHARED_PAGES_START_ADDRESS:x}     ");
    // Safety: We are assuming that GUEST_SHARED_PAGES_START_ADDRESS is valid, and will be mapped
    // in on a fault.
    unsafe {
        if core::ptr::read_volatile(GUEST_SHARED_PAGES_START_ADDRESS as *const u64)
            == GUEST_SHARE_PING
        {
            // Write a known value for verification purposes
            core::ptr::write_volatile(
                GUEST_SHARED_PAGES_START_ADDRESS as *mut u64,
                GUEST_SHARE_PONG,
            );
        }
    }

    // Now convert the page back to confidential and make sure we can fault it back in.
    //
    // Safety: We don't care about the contents of this memory and we won't touch it until the
    // call returns.
    unsafe {
        cove_guest::unshare_memory(
            GUEST_SHARED_PAGES_START_ADDRESS,
            NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
        )
        .expect("GuestVm -- UnshareMemory failed");
    }
    // Safety: We are assuming that the host will fault in GUEST_SHARED_PAGES_START_ADDRESS.
    unsafe { core::ptr::write_volatile(GUEST_SHARED_PAGES_START_ADDRESS as *mut u64, 0xdeadbeef) };

    // And share it again.
    //
    // Safety: We don't care about the contents of this memory and we won't touch it again.
    unsafe {
        cove_guest::share_memory(
            GUEST_SHARED_PAGES_START_ADDRESS,
            NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
        )
        .expect("GuestVm -- ShareMemory failed");
    }

    Ok(())
}

fn test_emulated_mmio() -> TestResult {
    for _ in 0..2 {
        cove_guest::add_emulated_mmio_region(GUEST_MMIO_START_ADDRESS, PAGE_SIZE_4K)
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
        cove_guest::remove_emulated_mmio_region(GUEST_MMIO_START_ADDRESS, PAGE_SIZE_4K)
            .expect("GuestVm - RemoveEmulatedMmioRegion failed");
    }

    Ok(())
}

fn test_interrupts() -> TestResult {
    const INTERRUPT_ID: usize = 3;

    // Make sure we return from WFI.
    //
    // Safety: WFI behavior is well-defined.
    unsafe { asm!("wfi", options(nomem, nostack)) };

    // Enable external interrupt signalling so that we can check for them in SIP (and that the
    // host sees them in HGEIP), but leave SSTATUS.SIE unset so that we don't trap.
    //
    // TODO: Set up and interrupt handler and enable interrupts so that we can receive injected
    // interrupts.
    CSR.sie.read_and_set_field(sie::sext);

    // Enable interrupt delivery in the IMSIC.
    CSR.si_eidelivery.set(1);
    CSR.si_eithreshold.set(0);
    CSR.si_eie[INTERRUPT_ID / 64].read_and_set_bits(1 << (INTERRUPT_ID % 64));

    cove_guest::allow_external_interrupt(3).expect("GuestVm - AllowExternalInterrupt failed");

    // VSIP implementation is buggy in QEMU; use VSTOPI to check that we got the interrupt instead.
    if CSR.stopi.read(stopi::interrupt_id) == Interrupt::SupervisorExternal as u64 {
        println!("External interrupt pending in GuestVm");
    } else {
        println!("External interrupt NOT pending in GuestVm");
    }

    Ok(())
}

fn test_huge_pages() -> TestResult {
    // Safety: We are assuming that GUEST_PROMOTE_HUGE_PAGE_START_ADDRESS is valid, and the
    // entire 2M range starting from it will be mapped in on a fault.
    unsafe {
        let huge_page_base_addr = GUEST_PROMOTE_HUGE_PAGE_START_ADDRESS as *mut u64;
        // First we write to the special address (start address of the huge page) to trigger
        // a page fault that will be handled in a special way by tellus. That means instead of
        // adding one 4k zero page, tellus will add 512 4k zero pages to cover the entire 2M
        // page. We expect tellus to promote the 512 4k pages to a 2M page right after the zero
        // pages have been added.
        core::ptr::write(huge_page_base_addr, 0xdeadbeef);
        // At this point we read to validate the range was properly faulted in.
        let val = core::ptr::read(huge_page_base_addr);
        assert_eq!(val, 0xdeadbeef);
        // Here we want to validate the last 4k page of the 2M page is properly covered by the
        // page table.
        let last_4k_page_addr = (GUEST_PROMOTE_HUGE_PAGE_START_ADDRESS
            + ((NUM_GUEST_ZERO_PAGES_PROMOTE_HUGE_PAGE - 1) * PAGE_SIZE_4K))
            as *mut u64;
        // We write to this address, which would generate a panic if the page wasn't faulted
        // already. This verifies the 2M range is correctly faulted in, but this also checks
        // that page promotion didn't break the page table.
        core::ptr::write(last_4k_page_addr, 0xdeadbeef);
        let val = core::ptr::read(last_4k_page_addr);
        assert_eq!(val, 0xdeadbeef);
    }

    // Safety: We are assuming that GUEST_DEMOTE_HUGE_PAGE_START_ADDRESS is valid, and the
    // entire 2M range starting from it will be mapped in on a fault.
    unsafe {
        let huge_page_base_addr = GUEST_DEMOTE_HUGE_PAGE_START_ADDRESS as *mut u64;
        // First we write to the special address (start address of the huge page) to trigger
        // a page fault that will be handled in a special way by tellus. That means instead of
        // adding one 4k zero page, tellus will add one 2M zero page to cover the entire 2M
        // page. We expect tellus to demote the 2M page into 512 4M pages right after the zero
        // page has been added.
        core::ptr::write(huge_page_base_addr, 0xdeadbeef);
        // At this point we read to validate the range was properly faulted in.
        let val = core::ptr::read(huge_page_base_addr);
        assert_eq!(val, 0xdeadbeef);
        // Here we want to validate the last 4k page of the 2M page is properly covered by the
        // page table.
        let last_4k_page_addr = (GUEST_DEMOTE_HUGE_PAGE_START_ADDRESS
            + ((NUM_GUEST_ZERO_PAGES_DEMOTE_HUGE_PAGE - 1) * PAGE_SIZE_4K))
            as *mut u64;
        // We write to this address, which would generate a panic if the page wasn't faulted
        // already. This verifies the 2M range is correctly faulted in, but this also checks
        // that page demotion didn't break the page table.
        core::ptr::write(last_4k_page_addr, 0xdeadbeef);
        let val = core::ptr::read(last_4k_page_addr);
        assert_eq!(val, 0xdeadbeef);
    }

    Ok(())
}

#[no_mangle]
#[allow(clippy::zero_ptr)]
extern "C" fn kernel_init(hart_id: u64, boot_args: u64) {
    base::probe_sbi_extension(sbi_rs::EXT_COVE_GUEST).expect("COVE-Guest extension not present");

    // Convert a page to shared memory for use with the debug console.
    //
    // Safety: We haven't touched this memory and we won't touch it until the call returns.
    unsafe {
        cove_guest::share_memory(GUEST_DBCN_ADDRESS, PAGE_SIZE_4K)
            .expect("GuestVm -- ShareMemory failed");
    }
    let console_mem = unsafe {
        core::slice::from_raw_parts_mut(GUEST_DBCN_ADDRESS as *mut u8, PAGE_SIZE_4K as usize)
    };
    SbiConsole::set_as_console(console_mem);

    println!("*****************************************");
    println!("Hello world from Tellus guest            ");
    test_declare_pass!("guestvm boot", hart_id);

    let vectors_enabled = boot_args & BOOT_ARG_VECTORS_ENABLED != 0;
    if vectors_enabled {
        println!("guestvm vector extension enabled (on)");
    } else {
        println!("guestvm vector extension disabled (off)");
    }

    let mut next_page = USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K;
    test_runtest!("test attestation", { test_attestation() });
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

    test_runtest!("Test memory sharing", { test_memory_sharing() });

    if vectors_enabled {
        test_runtest!("test vectors", { test_vector() });
    }

    test_runtest!("test emulated mmio", { test_emulated_mmio() });

    test_runtest!("test interrupts", { test_interrupts() });

    test_runtest!("test huge pages", { test_huge_pages() });

    println!("Exiting guest");
    println!("*****************************************");

    reset::reset(sbi_rs::ResetType::Shutdown, sbi_rs::ResetReason::NoReason)
        .expect("Guest shutdown failed");
    unreachable!();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
