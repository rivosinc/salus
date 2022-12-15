// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![no_main]
#![no_std]
#![feature(
    asm_const,
    panic_info_message,
    allocator_api,
    alloc_error_handler,
    lang_items
)]

use core::alloc::{GlobalAlloc, Layout};
extern crate alloc;
extern crate test_workloads;

mod consts;

use consts::*;
use core::arch::asm;
use core::ops::Range;
use device_tree::Fdt;
use riscv_regs::{
    hie, hip, sie, DecodedInstruction, Exception, GprIndex, Instruction, Interrupt,
    LocalRegisterCopy, Readable, RiscvCsrInterface, Trap, Writeable, CSR, CSR_CYCLE, CSR_HTINST,
    CSR_HTVAL,
};
use s_mode_utils::abort::abort;
use s_mode_utils::ecall::ecall_send;
use s_mode_utils::{print::*, sbi_console::SbiConsole};
use sbi::api::{base, pmu, reset, tee_host, tee_interrupt};
use sbi::{
    PmuCounterConfigFlags, PmuCounterStartFlags, PmuCounterStopFlags, PmuEventType, PmuFirmware,
    PmuHardware, SbiMessage, EXT_PMU, EXT_TEE_HOST, EXT_TEE_INTERRUPT,
};

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
    // `shutdown` should not return, so unrapping the result is appropriate.
    reset::shutdown().unwrap();

    abort()
}

// Safety: addr must point to `num_pages` of memory that isn't currently used by this program. This
// memory will be overwritten and access will be removed.
unsafe fn convert_pages(addr: u64, num_pages: u64) {
    tee_host::convert_pages(addr, num_pages).expect("TsmConvertPages failed");

    // Fence the pages we just converted.
    //
    // TODO: Boot secondary CPUs and test the invalidation flow with multiple CPUs.
    tee_host::tsm_initiate_fence().expect("Tellus - TsmInitiateFence failed");
}

fn reclaim_pages(addr: u64, num_pages: u64) {
    tee_host::reclaim_pages(addr, num_pages).expect("TsmReclaimPages failed");

    for i in 0u64..((num_pages * PAGE_SIZE_4K) / 8) {
        let m = (addr + i) as *const u64;
        unsafe {
            if core::ptr::read_volatile(m) != 0 {
                panic!("Tellus - Read back non-zero at qword offset {i:x} after exiting from TVM!");
            }
        }
    }
}

fn exercise_pmu_functionality() {
    use sbi::api::pmu::{configure_matching_counters, start_counters, stop_counters};
    if base::probe_sbi_extension(EXT_PMU).is_err() {
        println!("Platform doesn't support PMU extensions");
        return;
    }
    let num_counters = pmu::get_num_counters().expect("Tellus - GetNumCounters returned error");
    for i in 0u64..num_counters {
        let result = pmu::get_counter_info(i);
        if result.is_err() {
            println!("GetCounterInfo for {i} failed with {result:?}");
        }
    }

    let event_type = PmuEventType::Hardware(PmuHardware::Instructions);
    let config_flags = PmuCounterConfigFlags::default();
    let result =
        configure_matching_counters(0, (1 << num_counters) - 1, config_flags, event_type, 0);
    let counter_index = result.expect("configure_matching_counters failed with {result:?}");
    println!("Assigned counter {} for instruction count", counter_index);

    let start_flags = PmuCounterStartFlags::default();
    let result = start_counters(counter_index, 0x1, start_flags, 0);
    if !matches!(result, Ok(_) | Err(sbi::Error::AlreadyStarted)) {
        panic!("start_counters failed with result {result:?}");
    }

    let result = stop_counters(counter_index, 0x1, PmuCounterStopFlags::default());
    result.expect("stop_counters failed with {result:?}");

    let info = pmu::get_counter_info(counter_index).expect("Tellus - GetCounterInfo failed");
    let csr_index = (info.get_csr() as u16) - CSR_CYCLE;
    let inst_count = CSR.hpmcounter[csr_index as usize].get_value();
    if inst_count == 0 {
        panic!("Read of counter {} returned 0", csr_index);
    }
    println!("Instruction counter: {}", inst_count);

    let event_type = PmuEventType::Firmware(PmuFirmware::AccessLoad);
    pmu::configure_matching_counters(
        0,
        (1 << num_counters) - 1,
        PmuCounterConfigFlags::default(),
        event_type,
        0,
    )
    .expect_err("Successfully configured FW counter");
}

fn store_into_vectors() {
    let vec_len: u64 = 8;
    let vtype: u64 = 0xda;
    let enable: u64 = 0x200;

    println!("Writing tellus vector registers");
    unsafe {
        // safe because we are only setting the vector csr's
        asm!(
            "csrrs zero, sstatus, {enable}",
            "vsetvl x0, {vec_len}, {vtype}",
            vec_len = in(reg) vec_len,
            vtype = in(reg) vtype,
            enable = in(reg) enable,
            options(nostack),
        )
    }

    const REG_WIDTH_IN_U64S: usize = 4;

    let mut inbuf = [0_u64; (32 * REG_WIDTH_IN_U64S)];
    for i in 0..inbuf.len() {
        inbuf[i] = 0xDEADBEEFCAFEBABE;
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
}

fn check_vectors() {
    println!("Reading vector registers");
    const REG_WIDTH_IN_U64S: usize = 4;

    let inbuf = [0_u64; (32 * REG_WIDTH_IN_U64S)];
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

    println!("Verify registers");
    let mut should_panic = false;
    for i in 0..inbuf.len() {
        if inbuf[i] != 0xDEADBEEFCAFEBABE {
            println!("error:  {} {} {}", i, 0xDEADBEEFCAFEBABE_u64, inbuf[i]);
            should_panic = true;
        }
    }

    if should_panic {
        panic!("Vector registers did not restore correctly");
    }
}

/// The entry point of the Rust part of the kernel.
#[no_mangle]
extern "C" fn kernel_init(hart_id: u64, fdt_addr: u64) {
    const NUM_TEE_PTE_PAGES: u64 = 10;
    const NUM_CONVERTED_ZERO_PAGES: u64 = NUM_GUEST_ZERO_PAGES + NUM_GUEST_SHARED_PAGES;

    if hart_id != 0 {
        // TODO handle more than 1 cpu
        abort();
    }

    SbiConsole::set_as_console();

    println!("Tellus: Booting the test VM");

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

    // the 4 is to skip the rv64 or rv32 at the begging of the string
    let vector_enabled = match fdt.get_property("riscv,isa") {
        Some(rv) if rv.len() < 5 => false,
        Some(rv) => rv.split('_').next().unwrap_or("")[4..].contains('v'),
        None => false,
    };

    if vector_enabled {
        println!("Tellus - Vector enabled");
    } else {
        println!("Tellus - Vector disabled");
    };

    base::probe_sbi_extension(EXT_TEE_HOST).expect("Platform doesn't support TEE extension");
    let tsm_info = tee_host::get_info().expect("Tellus - TsmGetInfo failed");
    let tvm_create_pages = 4 + tsm_info.tvm_state_pages;
    println!("Donating {} pages for TVM creation", tvm_create_pages);

    // Make sure TsmGetInfo fails if we pass it a bogus address.
    let msg = SbiMessage::TeeHost(sbi::TeeHostFunction::TsmGetInfo {
        dest_addr: 0x1000,
        len: core::mem::size_of::<sbi::TsmInfo>() as u64,
    });
    // Safety: The passed info pointer is bogus and nothing should be written to our memory.
    unsafe { ecall_send(&msg).expect_err("TsmGetInfo succeeded with an invalid pointer") };

    // Donate the pages necessary to create the TVM.
    let mut next_page = (mem_range.base() + mem_range.size() / 2) & !0x3fff;
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, tvm_create_pages);
    }

    // Now create the TVM.
    let state_pages_base = next_page;
    let tvm_page_directory_addr = state_pages_base;
    let tvm_state_addr = tvm_page_directory_addr + 4 * PAGE_SIZE_4K;

    let vmid = tee_host::tvm_create(tvm_page_directory_addr, tvm_state_addr)
        .expect("Tellus - TvmCreate returned error");
    println!("Tellus - TvmCreate Success vmid: {vmid:x}");
    next_page += PAGE_SIZE_4K * tvm_create_pages;

    // Set aside pages for the shared mem area.
    let num_shmem_pages =
        (core::mem::size_of::<sbi::TsmShmemArea>() as u64 + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
    let shmem_addr = next_page;
    next_page += num_shmem_pages * PAGE_SIZE_4K;
    // Safety: We own `shmem_addr` and will only access it through volatile reads/writes.
    unsafe { tee_host::register_shmem(shmem_addr).expect("TsmSetShmem failed") };
    // Safety: `shmem_addr` points to a sufficient number of pages to hold a TsmShmemArea struct
    // and will not be used for any other purpose for the duration of `kernel_init()`.
    let shmem = unsafe { tee_host::TsmShmemAreaRef::new(shmem_addr as *mut _) };

    // Add pages for the page table
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_TEE_PTE_PAGES);
    }
    tee_host::add_page_table_pages(vmid, next_page, NUM_TEE_PTE_PAGES)
        .expect("Tellus - AddPageTablePages returned error");
    next_page += PAGE_SIZE_4K * NUM_TEE_PTE_PAGES;

    // Add vCPU0.

    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    let vcpu_pages_base = next_page;
    unsafe {
        convert_pages(vcpu_pages_base, tsm_info.tvm_vcpu_state_pages);
    }
    next_page += PAGE_SIZE_4K * tsm_info.tvm_vcpu_state_pages;
    tee_host::add_vcpu(vmid, 0, vcpu_pages_base).expect("Tellus - TvmCpuCreate returned error");

    let has_aia = base::probe_sbi_extension(EXT_TEE_INTERRUPT).is_ok();
    // CPU0, guest interrupt file 0.
    let imsic_file_addr = IMSIC_START_ADDRESS + PAGE_SIZE_4K;
    if has_aia {
        // Check HGEIE to see how many guests we have.
        CSR.hgeie.set(!0u64);
        let hgeie = CSR.hgeie.atomic_replace(0);
        println!("Found {:} guest interrupt files", hgeie.count_ones());

        // Set the IMSIC params for the TVM.
        let aia_params = sbi::TvmAiaParams {
            imsic_base_addr: 0x2800_0000,
            group_index_bits: 0,
            group_index_shift: 24,
            hart_index_bits: 8,
            guest_index_bits: 0,
            guests_per_hart: 0,
        };
        tee_interrupt::tvm_aia_init(vmid, aia_params).expect("Tellus - TvmAiaInit failed");
        tee_interrupt::set_vcpu_imsic_addr(vmid, 0, 0x2800_0000)
            .expect("Tellus - TvmCpuSetImsicAddr failed");

        // Try to convert a guest interrupt file.
        //
        // Safety: We trust that the IMSIC is actually at IMSIC_START_ADDRESS, and we aren't
        // touching this page at all in this program.
        unsafe { tee_interrupt::convert_imsic(imsic_file_addr) }
            .expect("Tellus - TsmConvertImsic failed");
        tee_host::tsm_initiate_fence().expect("Tellus - TsmInitiateFence failed");
    } else {
        println!("Platform doesn't support TEE AIA extension");
    }

    let guest_image_base = USABLE_RAM_START_ADDRESS + PAGE_SIZE_4K * NUM_TELLUS_IMAGE_PAGES;
    // Safety: Safe to make a slice out of the guest image as it is read-only and not used by this
    // program.
    let guest_image = unsafe {
        core::slice::from_raw_parts(
            guest_image_base as *const u8,
            (PAGE_SIZE_4K * NUM_GUEST_DATA_PAGES) as usize,
        )
    };
    let donated_pages_base = next_page;

    // Declare the confidential region of the guest's physical address space.
    tee_host::add_memory_region(
        vmid,
        USABLE_RAM_START_ADDRESS,
        GUEST_RAM_END_ADDRESS - USABLE_RAM_START_ADDRESS,
    )
    .expect("Tellus - TvmAddMemoryRegion failed");

    // Add data pages
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_GUEST_DATA_PAGES);
    }
    tee_host::add_measured_pages(
        vmid,
        guest_image,
        next_page,
        sbi::TsmPageType::Page4k,
        USABLE_RAM_START_ADDRESS,
    )
    .expect("Tellus - TvmAddMeasuredPages returned error");
    next_page += PAGE_SIZE_4K * NUM_GUEST_DATA_PAGES;

    // Convert pages to handle confidential page faults.
    let zero_pages_base = next_page;
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_CONVERTED_ZERO_PAGES);
    }

    next_page += NUM_CONVERTED_ZERO_PAGES * PAGE_SIZE_4K;
    let shared_page_base = next_page;

    // Tell the guest if we have vector support via its boot argument.
    let boot_arg = if vector_enabled {
        BOOT_ARG_VECTORS_ENABLED
    } else {
        0
    };
    // TODO test that access to pages crashes somehow
    tee_host::tvm_finalize(vmid, 0x8020_0000, boot_arg).expect("Tellus - Finalize returned error");

    // Map a few zero pages up front. We'll fault the rest in as necessary.
    tee_host::add_zero_pages(
        vmid,
        zero_pages_base,
        sbi::TsmPageType::Page4k,
        PRE_FAULTED_ZERO_PAGES,
        GUEST_ZERO_PAGES_START_ADDRESS,
    )
    .expect("Tellus - TvmAddZeroPages failed");
    let mut zero_pages_added = PRE_FAULTED_ZERO_PAGES;

    if vector_enabled {
        store_into_vectors();
    }

    // Bind to a guest interrupt file if AIA is enabled.
    if has_aia {
        tee_interrupt::bind_vcpu_imsic(vmid, 0, 1 << 1).expect("Tellus - TvmCpuBindImsic failed");
    }

    // For now we run on a single CPU so try to exercise the rebinding interface
    // on the same PCPU.
    // CPU0, guest interrupt file 2.
    let imsic_file_num = 2;
    let imsic_file_addr = IMSIC_START_ADDRESS + PAGE_SIZE_4K * imsic_file_num;
    if has_aia {
        // Try to convert a guest interrupt file.
        //
        // Safety: We trust that the IMSIC is actually at IMSIC_START_ADDRESS, and we aren't
        // touching this page at all in this program.
        unsafe { tee_interrupt::convert_imsic(imsic_file_addr) }
            .expect("Tellus - TsmConvertImsic failed");
        tee_host::tsm_initiate_fence().expect("Tellus - TsmInitiateFence failed");

        tee_interrupt::rebind_vcpu_imsic_begin(vmid, 0, 1 << imsic_file_num)
            .expect("Tellus - TvmCpuRebindImsicBegin failed");
        tee_host::tvm_initiate_fence(vmid).expect("Tellus - TvmInitiateFence failed");
        tee_interrupt::rebind_vcpu_imsic_clone(vmid, 0)
            .expect("Tellus - TvmCpuRebindImsicClone failed");
        tee_interrupt::rebind_vcpu_imsic_end(vmid, 0)
            .expect("Tellus - TvmCpuRebindImsicEnd failed");

        // Reclaim previous imsic file address.
        tee_interrupt::reclaim_imsic(IMSIC_START_ADDRESS + PAGE_SIZE_4K)
            .expect("Tellus - TsmReclaimImsic failed");

        CSR.hgeie.set(1 << imsic_file_num);
    }

    // Test that a pending timer or external interruptcauses us to exit the guest.
    CSR.stimecmp.set(0);
    // EIDELIVERY = 1
    CSR.siselect.set(0x70);
    CSR.sireg.set(1);
    // EITHRESHOLD = 0
    CSR.siselect.set(0x72);
    CSR.sireg.set(0);
    // EIE0[1] = 1
    CSR.siselect.set(0xc0);
    CSR.sireg.read_and_set_bits(1 << 1);
    // EIP0[1] = 1
    CSR.siselect.set(0x80);
    CSR.sireg.read_and_set_bits(1 << 1);
    let mut sie = LocalRegisterCopy::new(0);
    sie.modify(sie::stimer.val(1));
    sie.modify(sie::sext.val(1));
    CSR.sie.set(sie.get());

    let mut shared_mem_region: Option<Range<u64>> = None;
    let mut mmio_region: Option<Range<u64>> = None;
    loop {
        // Safety: running a VM will only write the `TsmShmemArea` struct that was registered
        // with `register_shmem()`.
        let blocked = tee_host::tvm_run(vmid, 0).expect("Could not run guest VM") != 0;
        let scause = CSR.scause.get();
        if let Ok(t) = Trap::from_scause(scause) {
            use Exception::*;
            use Interrupt::*;
            match t {
                Trap::Exception(VirtualSupervisorEnvCall) => {
                    // Read the ECALL arguments written to the A* regs in shared memory.
                    let mut a_regs = [0u64; 8];
                    for (i, reg) in a_regs.iter_mut().enumerate() {
                        *reg = shmem.gpr(GprIndex::A0 as usize + i);
                    }
                    use SbiMessage::*;
                    match SbiMessage::from_regs(&a_regs) {
                        Ok(Reset(_)) => {
                            println!("Guest VM requested shutdown");
                            break;
                        }
                        Ok(TeeGuest(guest_func)) => {
                            use sbi::TeeGuestFunction::*;
                            match guest_func {
                                AddMmioRegion { addr, len } => {
                                    mmio_region = Some(Range {
                                        start: addr,
                                        end: addr + len,
                                    });
                                }
                                ShareMemory { addr, len } => {
                                    if shared_mem_region.is_some() {
                                        panic!("GuestVm already set a shared memory region");
                                    }
                                    shared_mem_region = Some(Range {
                                        start: addr,
                                        end: addr + len,
                                    });

                                    if blocked {
                                        tee_host::tvm_run(vmid, 0).unwrap_err();
                                        println!("Tellus - TVM fence on page sharing");
                                        tee_host::tvm_initiate_fence(vmid).unwrap();
                                    }
                                }
                                UnshareMemory { addr, len } => {
                                    assert_eq!(
                                        shared_mem_region,
                                        Some(Range {
                                            start: addr,
                                            end: addr + len,
                                        })
                                    );
                                    shared_mem_region = None;

                                    if blocked {
                                        tee_host::tvm_run(vmid, 0).unwrap_err();
                                        println!("Tellus - TVM fence on page unsharing");
                                        tee_host::tvm_initiate_fence(vmid).unwrap();
                                    }
                                }
                                AllowExternalInterrupt { id } => {
                                    // Try to inject the allow-listed interrupt into the guest.
                                    // If guest allowed all interrupts (-1), just pick a random
                                    // one.
                                    let id = u64::try_from(id).unwrap_or(7);
                                    println!("Injecting interrupt {id} into guest");
                                    tee_interrupt::inject_external_interrupt(vmid, 0, id)
                                        .expect("Tellus - InjectExternalInterrupt failed");

                                    // Check that we see the external interrupt pending in HGEIP/HIP.
                                    if (CSR.hgeip.get() & (1 << imsic_file_num)) == 0 {
                                        panic!("Injected interrupt, but no HGEI pending.");
                                    }
                                    if CSR.hip.read(hip::sgext) == 0 {
                                        panic!("Injected interrupt, but no SG_EXT pending.");
                                    }

                                    // Now check if we can get an SG_EXT.
                                    CSR.hie.read_and_set_field(hie::sgext);
                                }
                                _ => {
                                    continue;
                                }
                            }
                        }
                        _ => {
                            println!("Unexpected ECALL from guest");
                            break;
                        }
                    }
                }
                Trap::Exception(GuestLoadPageFault) | Trap::Exception(GuestStorePageFault) => {
                    let fault_addr = (shmem.csr(CSR_HTVAL) << 2) | (CSR.stval.get() & 0x3);
                    match fault_addr {
                        addr if shared_mem_region
                            .as_ref()
                            .filter(|r| r.contains(&addr))
                            .is_some() =>
                        {
                            // Safety: shared_page_base points to pages that will only be accessed
                            // as volatile from here on.
                            unsafe {
                                tee_host::add_shared_pages(
                                    vmid,
                                    shared_page_base,
                                    sbi::TsmPageType::Page4k,
                                    NUM_GUEST_SHARED_PAGES,
                                    addr & !(PAGE_SIZE_4K - 1),
                                )
                                .expect("Tellus -- TvmAddSharedPages failed");
                            }

                            // Safety: We own the page, and are writing a value expected by the
                            // guest. Note that any access to shared pages must use volatile memory
                            // semantics to guard against the compiler's no-aliasing assumptions.
                            unsafe {
                                core::ptr::write_volatile(
                                    shared_page_base as *mut u64,
                                    GUEST_SHARE_PING,
                                );
                            }
                        }
                        addr if mmio_region.as_ref().filter(|r| r.contains(&addr)).is_some() => {
                            let inst = DecodedInstruction::from_raw(shmem.csr(CSR_HTINST) as u32)
                                .expect("Failed to decode faulting MMIO instruction")
                                .instruction();
                            // Handle the load or store; the source/dest register is always A0.
                            use Instruction::*;
                            match inst {
                                Lb(_) | Lbu(_) | Lh(_) | Lhu(_) | Lw(_) | Lwu(_) | Ld(_) => {
                                    shmem.set_gpr(GprIndex::A0 as usize, 0x42);
                                }
                                Sb(_) | Sh(_) | Sw(_) | Sd(_) => {
                                    let val = shmem.gpr(GprIndex::A0 as usize);
                                    println!("Guest says: 0x{:x} at 0x{:x}", val, fault_addr);
                                }
                                _ => {
                                    println!("Unexpected guest MMIO instruction: {:?}", inst);
                                    return;
                                }
                            }
                        }
                        addr if !blocked => {
                            // Fault in the page.
                            if zero_pages_added >= NUM_CONVERTED_ZERO_PAGES {
                                panic!("Ran out of pages to fault in");
                            }
                            tee_host::add_zero_pages(
                                vmid,
                                zero_pages_base + zero_pages_added * PAGE_SIZE_4K,
                                sbi::TsmPageType::Page4k,
                                1,
                                addr & !(PAGE_SIZE_4K - 1),
                            )
                            .expect("Tellus - TvmAddZeroPages failed");
                            zero_pages_added += 1;
                        }
                        _ => {
                            println!("Unhandled guest page fault at 0x{:x}", fault_addr);
                            break;
                        }
                    }
                }
                Trap::Exception(VirtualInstruction) => {
                    let inst = DecodedInstruction::from_raw(CSR.stval.get() as u32)
                        .expect("Failed to decode faulting virtual instruction")
                        .instruction();
                    use Instruction::*;
                    match inst {
                        Wfi => {
                            continue;
                        }
                        _ => {
                            println!("Unexpected guest virtual instruction: {:?}", inst);
                        }
                    }
                }
                Trap::Exception(e) => {
                    println!("Guest VM terminated with exception {:?}", e);
                    break;
                }
                Trap::Interrupt(SupervisorTimer) => {
                    println!("Got our timer; turning it off now");
                    CSR.sie.read_and_clear_field(sie::stimer);
                }
                Trap::Interrupt(SupervisorExternal) => {
                    println!("Got our external interrupt; turning it off now");
                    CSR.sie.read_and_clear_field(sie::sext);
                }
                Trap::Interrupt(SupervisorGuestExternal) => {
                    println!("Got a supervisor guest external interrupt; turning it off now");
                    CSR.hie.read_and_clear_field(hie::sgext);
                }
                Trap::Interrupt(i) => {
                    println!("Unexpected interrupt {:?}", i);
                    break;
                }
            }
        } else {
            println!("Guest VM terminated with unexpected cause 0x{:x}", scause,);
        }
    }

    if vector_enabled {
        check_vectors();
    }

    if has_aia {
        // A fence is needed in between begin & end.
        tee_interrupt::unbind_vcpu_imsic_begin(vmid, 0).expect("Tellus - TvmCpuUnbindImsic failed");
        tee_host::tvm_initiate_fence(vmid).expect("Tellus - TvmInitiateFence failed");
        tee_interrupt::unbind_vcpu_imsic_end(vmid, 0).expect("Tellus - TvmCpuUnbindImsic failed");
    }

    tee_host::tvm_destroy(vmid).expect("Tellus - TvmDestroy returned error");

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
        NUM_GUEST_DATA_PAGES + NUM_CONVERTED_ZERO_PAGES,
    );
    reclaim_pages(state_pages_base, tvm_create_pages);
    reclaim_pages(vcpu_pages_base, tsm_info.tvm_vcpu_state_pages);
    if has_aia {
        tee_interrupt::reclaim_imsic(imsic_file_addr).expect("Tellus - TsmReclaimImsic failed");
    }
    exercise_pmu_functionality();
    tee_host::unregister_shmem().expect("TsmSetShmem failed");
    println!("Tellus - All OK");
    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
