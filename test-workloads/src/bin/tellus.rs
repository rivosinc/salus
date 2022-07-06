// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(panic_info_message, allocator_api, alloc_error_handler, lang_items)]

use core::alloc::{GlobalAlloc, Layout};

extern crate alloc;
extern crate test_workloads;

use device_tree::Fdt;
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

/// Powers off this machine.
pub fn poweroff() -> ! {
    let msg = SbiMessage::Reset(sbi::ResetFunction::shutdown());
    // Safety: This ecall doesn't touch memory and will never return.
    unsafe {
        ecall_send(&msg).unwrap();
    }

    abort()
}

const PAGE_SIZE_4K: u64 = 4096;

fn convert_pages(addr: u64, num_pages: u64) {
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmConvertPages {
        page_addr: addr,
        page_type: sbi::TsmPageType::Page4k,
        num_pages: num_pages,
    });
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe { ecall_send(&msg).expect("TsmConvertPages failed") };

    // Fence the pages we just converted.
    //
    // TODO: Boot secondary CPUs and test the invalidation flow with multiple CPUs.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmInitiateFence);
    // Safety: TsmInitiateFence doesn't read or write any memory we have access to.
    unsafe { ecall_send(&msg).expect("TsmInitiateFence failed") };
}

fn reclaim_pages(addr: u64, num_pages: u64) {
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmReclaimPages {
        page_addr: addr,
        page_type: sbi::TsmPageType::Page4k,
        num_pages: num_pages,
    });
    // Safety: The referenced pages are made accessible again, which is safe since we haven't
    // done anything with them since they were converted.
    unsafe { ecall_send(&msg).expect("TsmReclaimPages failed") };

    for i in 0u64..((num_pages * PAGE_SIZE_4K) / 8) {
        let m = (addr + i) as *const u64;
        unsafe {
            if core::ptr::read_volatile(m) != 0 {
                panic!("Tellus - Read back non-zero at qword offset {i:x} after exiting from TVM!");
            }
        }
    }
}

fn get_vcpu_reg(vmid: u64, register: sbi::TvmCpuRegister) -> u64 {
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuGetRegister {
        guest_id: vmid,
        vcpu_id: 0,
        register,
    });
    // Safety: `TvmCpuGetRegister` doesn't access our memory.
    unsafe { ecall_send(&msg).expect("Tellus - TvmCpuGetRegister failed") }
}

fn set_vcpu_reg(vmid: u64, register: sbi::TvmCpuRegister, value: u64) {
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuSetRegister {
        guest_id: vmid,
        vcpu_id: 0,
        register,
        value,
    });
    // Safety: `TvmCpuSetRegister` doesn't access our memory.
    unsafe { ecall_send(&msg).expect("Tellus - TvmCpuSetRegister failed") };
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    const USABLE_RAM_START_ADDRESS: u64 = 0x8020_0000;
    const SHARED_PAGES_START_ADDRESS: u64 = 0x1_0000_0000;
    const NUM_VCPUS: u64 = 1;
    const NUM_TEE_PTE_PAGES: u64 = 10;
    const NUM_GUEST_DATA_PAGES: u64 = 160;
    const NUM_GUEST_ZERO_PAGES: u64 = 10;
    const PRE_FAULTED_ZERO_PAGES: u64 = 2;
    const NUM_GUEST_PAD_PAGES: u64 = 32;
    const NUM_GUEST_SHARED_PAGES: u64 = 1;
    const GUEST_MMIO_ADDRESS: u64 = 0x1000_8000;
    // TODO: Consider moving to a common module to ensure that the host and guest are in lockstep
    const GUEST_SHARE_PING: u64 = 0xBAAD_F00D;
    const GUEST_SHARE_PONG: u64 = 0xF00D_BAAD;

    if hart_id != 0 {
        // TODO handle more than 1 cpu
        abort();
    }

    console_write_bytes(b"Tellus: Booting the test VM\n");

    // Safe because we trust the host to boot with a valid fdt_addr pass in register a1.
    let fdt = match unsafe { Fdt::new_from_raw_pointer(fdt_addr as *const u8) } {
        Ok(f) => f,
        Err(e) => panic!("Bad FDT from hypervisor: {}", e),
    };
    let mem_range = fdt.memory_regions().next().unwrap();
    println!(
        "Tellus - Mem base: {:x} size: {:x}",
        mem_range.base(),
        mem_range.size()
    );

    let mut tsm_info = sbi::TsmInfo::default();
    let tsm_info_size = core::mem::size_of::<sbi::TsmInfo>() as u64;
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmGetInfo {
        dest_addr: &mut tsm_info as *mut _ as u64,
        len: tsm_info_size,
    });
    // Safety: The passed info pointer is uniquely owned so it's safe to modify in SBI.
    let tsm_info_len = unsafe { ecall_send(&msg).expect("TsmGetInfo failed") };
    assert_eq!(tsm_info_len, tsm_info_size);
    let tvm_create_pages = 4
        + tsm_info.tvm_state_pages
        + ((NUM_VCPUS * tsm_info.tvm_bytes_per_vcpu) + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
    println!("Donating {} pages for TVM creation", tvm_create_pages);

    // Make sure TsmGetInfo fails if we pass it a bogus address.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TsmGetInfo {
        dest_addr: 0x1000,
        len: tsm_info_size,
    });
    // Safety: The passed info pointer is bogus and nothing should be written to our memory.
    unsafe { ecall_send(&msg).expect_err("TsmGetInfo succeeded with an invalid pointer") };

    // Donate the pages necessary to create the TVM.
    let mut next_page = (mem_range.base() + mem_range.size() / 2) & !0x3fff;
    convert_pages(next_page, tvm_create_pages);

    // Now create the TVM.
    let state_pages_base = next_page;
    let tvm_page_directory_addr = state_pages_base;
    let tvm_state_addr = tvm_page_directory_addr + 4 * PAGE_SIZE_4K;
    let tvm_vcpu_addr = tvm_state_addr + tsm_info.tvm_state_pages * PAGE_SIZE_4K;
    let tvm_create_params = sbi::TvmCreateParams {
        tvm_page_directory_addr,
        tvm_state_addr,
        tvm_num_vcpus: NUM_VCPUS,
        tvm_vcpu_addr,
    };
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCreate {
        params_addr: (&tvm_create_params as *const sbi::TvmCreateParams) as u64,
        len: core::mem::size_of::<sbi::TvmCreateParams>() as u64,
    });
    // Safety: We trust the TSM to only read up to `len` bytes of the `TvmCreateParams` structure
    // pointed to by `params_addr.
    let vmid = unsafe { ecall_send(&msg).expect("Tellus - TvmCreate returned error") };
    println!("Tellus - TvmCreate Success vmid: {vmid:x}");
    next_page += PAGE_SIZE_4K * tvm_create_pages;

    // Add pages for the page table
    convert_pages(next_page, NUM_TEE_PTE_PAGES);
    let msg = SbiMessage::Tee(sbi::TeeFunction::AddPageTablePages {
        guest_id: vmid,
        page_addr: next_page,
        num_pages: NUM_TEE_PTE_PAGES,
    });
    // Safety: `AddPageTablePages` only accesses pages that have been previously converted.
    unsafe { ecall_send(&msg).expect("Tellus - AddPageTablePages returned error") };
    next_page += PAGE_SIZE_4K * NUM_TEE_PTE_PAGES;

    // Add vCPU0.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuCreate {
        guest_id: vmid,
        vcpu_id: 0,
    });
    // Safety: Creating a vcpu doesn't touch any memory owned here.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmCpuCreate returned error");
    }

    /*
        The Tellus composite image includes the guest image
        |========== --------> 0x8020_0000 (Tellus _start)
        | Tellus code and data
        | ....
        | .... (Zero padding)
        | ....
        |======== -------> 0x8020_0000 + 4096*NUM_GUEST_PAD_PAGES
        | Guest code and data (Guest _start is mapped at GPA 0x8020_0000)
        |
        |=========================================
    */

    let measurement_page_addr = next_page;
    next_page += PAGE_SIZE_4K;

    let guest_image_base = USABLE_RAM_START_ADDRESS + PAGE_SIZE_4K * NUM_GUEST_PAD_PAGES;
    let donated_pages_base = next_page;

    // Declare the confidential region of the guest's physical address space.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddConfidentialMemoryRegion {
        guest_id: vmid,
        guest_addr: USABLE_RAM_START_ADDRESS,
        len: (NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES) * PAGE_SIZE_4K,
    });
    // Safety: `TvmAddConfidentialMemoryRegion` doesn't access our memory at all.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmAddConfidentialMemoryRegion failed");
    }

    // Add data pages
    convert_pages(next_page, NUM_GUEST_DATA_PAGES);
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddMeasuredPages {
        guest_id: vmid,
        src_addr: guest_image_base,
        dest_addr: next_page,
        page_type: sbi::TsmPageType::Page4k,
        num_pages: NUM_GUEST_DATA_PAGES,
        guest_addr: USABLE_RAM_START_ADDRESS,
    });
    // Safety: `TvmAddMeasuredPages` only writes pages that have already been converted, and only
    // reads the pages pointed to by `src_addr`. This is safe because those pages are not used by
    // this program.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmAddMeasuredPages returned error");
    }
    next_page += PAGE_SIZE_4K * NUM_GUEST_DATA_PAGES;

    let msg = SbiMessage::Measurement(sbi::MeasurementFunction::GetSelfMeasurement {
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });

    // Safety: The measurement page is uniquely owned and can be written to safely by SBI
    match unsafe { ecall_send(&msg) } {
        Err(e) => {
            println!("Host measurement error {e:?}");
            panic!("Host measurement call failed");
        }
        Ok(_) => {
            let measurement =
                unsafe { core::ptr::read_volatile(measurement_page_addr as *const u64) };
            println!("Host measurement was {measurement:x}");
        }
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::GetGuestMeasurement {
        guest_id: vmid,
        measurement_version: 1,
        measurement_type: 1,
        dest_addr: measurement_page_addr,
    });

    // Safety: The measurement page is uniquely owned and can be written to safely by SBI
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

    // Convert the zero pages and map a few of them up front. We'll fault the rest in as necessary.
    let zero_pages_start = next_page;
    convert_pages(zero_pages_start, NUM_GUEST_ZERO_PAGES);
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddZeroPages {
        guest_id: vmid,
        page_addr: zero_pages_start,
        page_type: sbi::TsmPageType::Page4k,
        num_pages: PRE_FAULTED_ZERO_PAGES,
        guest_addr: USABLE_RAM_START_ADDRESS + NUM_GUEST_DATA_PAGES * PAGE_SIZE_4K,
    });

    next_page += NUM_GUEST_ZERO_PAGES * PAGE_SIZE_4K;
    let shared_page_base = next_page;

    // Safety: `TvmAddZeroPages` only touches pages that we've already converted.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmAddZeroPages failed");
    }
    let mut zero_pages_added = PRE_FAULTED_ZERO_PAGES;

    // Add a page of emualted MMIO.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddEmulatedMmioRegion {
        guest_id: vmid,
        guest_addr: GUEST_MMIO_ADDRESS,
        len: PAGE_SIZE_4K,
    });
    // Safety: Doesn't affect host memory safety.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmAddEmulatedMmioRegion failed");
    }

    // Set the entry point.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuSetRegister {
        guest_id: vmid,
        vcpu_id: 0,
        register: sbi::TvmCpuRegister::EntryPc,
        value: 0x8020_0000,
    });
    // Safety: Setting a guest register doesn't affect host memory safety.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmCpuSetRegister returned error");
    }

    // Set the kernel_init() parameter.
    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuSetRegister {
        guest_id: vmid,
        vcpu_id: 0,
        register: sbi::TvmCpuRegister::EntryArg,
        value: SHARED_PAGES_START_ADDRESS,
    });
    // Safety: Setting a guest register doesn't affect host memory safety.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmCpuSetRegister returned error");
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddSharedMemoryRegion {
        guest_id: vmid,
        guest_addr: SHARED_PAGES_START_ADDRESS,
        len: NUM_GUEST_SHARED_PAGES * PAGE_SIZE_4K,
    });
    // Safety: `TvmAddSharedMemoryRegion` doesn't affect host memory
    unsafe {
        ecall_send(&msg).expect("Tellus -- TvmAddSharedMemoryRegion returned error");
    }

    // Safety: `TvmAddSharedMemoryRegion` doesn't affect host memory
    unsafe {
        ecall_send(&msg).expect_err("Tellus -- TvmAddSharedMemoryRegion succeeded second time");
    }

    // TODO test that access to pages crashes somehow
    let msg = SbiMessage::Tee(sbi::TeeFunction::Finalize { guest_id: vmid });
    // Safety: `Finalize` doesn't touch memory.
    unsafe {
        ecall_send(&msg).expect("Tellus - Finalize returned error");
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmCpuRun {
        guest_id: vmid,
        vcpu_id: 0,
    });
    loop {
        // Safety: running a VM can't affect host memory as that memory isn't accessible to the VM.
        match unsafe { ecall_send(&msg) } {
            Err(e) => {
                println!("Tellus - Run returned error {:?}", e);
                panic!("Could not run guest VM");
            }
            Ok(exit_code) => {
                let cause = sbi::TvmCpuExitCode::from_reg(exit_code).unwrap();
                match cause {
                    sbi::TvmCpuExitCode::ConfidentialPageFault => {
                        let fault_addr = get_vcpu_reg(vmid, sbi::TvmCpuRegister::ExitCause0);
                        // Fault in the page.
                        if zero_pages_added >= NUM_GUEST_ZERO_PAGES {
                            panic!("Ran out of pages to fault in");
                        }
                        let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddZeroPages {
                            guest_id: vmid,
                            page_addr: zero_pages_start + zero_pages_added * PAGE_SIZE_4K,
                            page_type: sbi::TsmPageType::Page4k,
                            num_pages: 1,
                            guest_addr: fault_addr & !(PAGE_SIZE_4K - 1),
                        });
                        // Safety: `TvmAddZeroPages` only touches pages that we've already converted.
                        unsafe {
                            ecall_send(&msg).expect("Tellus - TvmAddZeroPages failed");
                        }
                        zero_pages_added += 1;
                    }
                    sbi::TvmCpuExitCode::SharedPageFault => {
                        // Figure out where the fault occurred.
                        let fault_addr = get_vcpu_reg(vmid, sbi::TvmCpuRegister::ExitCause0);
                        if fault_addr != SHARED_PAGES_START_ADDRESS {
                            panic!("Unexpected shared page fault address at {fault_addr:x}");
                        }
                        let msg = SbiMessage::Tee(sbi::TeeFunction::TvmAddSharedPages {
                            guest_id: vmid,
                            page_addr: shared_page_base,
                            page_type: sbi::TsmPageType::Page4k,
                            num_pages: NUM_GUEST_SHARED_PAGES,
                            guest_addr: fault_addr,
                        });
                        // Safety: `TvmAddSharedPages` only touches pages owned by us
                        unsafe {
                            ecall_send(&msg).expect("Tellus -- TvmAddSharedPages failed");
                        }

                        // Safety: We own the page, and are writing a value expected by the guest
                        // Note that any access to shared pages must use volatile memory semantics
                        // to guard against the compiler's no-aliasing assumptions.
                        unsafe {
                            core::ptr::write_volatile(
                                shared_page_base as *mut u64,
                                GUEST_SHARE_PING,
                            );
                        }
                    }
                    sbi::TvmCpuExitCode::MmioPageFault => {
                        let fault_addr = get_vcpu_reg(vmid, sbi::TvmCpuRegister::ExitCause0);
                        let op = sbi::TvmMmioOpCode::from_reg(get_vcpu_reg(
                            vmid,
                            sbi::TvmCpuRegister::ExitCause1,
                        ))
                        .unwrap();
                        // Handle the load or store.
                        use sbi::TvmMmioOpCode::*;
                        match op {
                            Load8 | Load8U | Load16 | Load16U | Load32 | Load32U | Load64 => {
                                set_vcpu_reg(vmid, sbi::TvmCpuRegister::MmioLoadValue, 0x42);
                            }
                            Store8 | Store16 | Store32 | Store64 => {
                                let val = get_vcpu_reg(vmid, sbi::TvmCpuRegister::MmioStoreValue);
                                println!("Guest says: 0x{:x} at 0x{:x}", val, fault_addr);
                            }
                        }
                    }
                    _ => {
                        println!("Tellus - Guest exited with status {:?}", cause);
                        break;
                    }
                }
            }
        }
    }

    let msg = SbiMessage::Tee(sbi::TeeFunction::TvmDestroy { guest_id: vmid });
    // Safety: destroying a VM doesn't write to memory that's accessible from the host.
    unsafe {
        ecall_send(&msg).expect("Tellus - TvmDestroy returned error");
    }

    // Safety: We own the page.
    // Note that any access to shared pages must use volatile memory semantics
    // to guard against the compiler's no-aliasing assumptions.
    let guest_written_value = unsafe { core::ptr::read_volatile(shared_page_base as *mut u64) };
    if guest_written_value != GUEST_SHARE_PONG {
        println!("Tellus - unexpected value from guest shared page: 0x{guest_written_value:x}");
    }
    // Check that we can reclaim previously-converted pages and that they have been cleared.
    reclaim_pages(
        donated_pages_base,
        NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES,
    );
    reclaim_pages(state_pages_base, tvm_create_pages);

    println!("Tellus - All OK");

    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
