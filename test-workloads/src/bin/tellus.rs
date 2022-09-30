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

use arrayvec::ArrayVec;
use consts::*;
#[cfg(target_feature = "v")]
use core::arch::asm;
use core::{ops::Range, ptr};
use device_tree::Fdt;
use riscv_regs::{DecodedInstruction, Exception, GprIndex, Instruction, Trap, CSR, CSR_CYCLE};
use s_mode_utils::abort::abort;
use s_mode_utils::ecall::ecall_send;
use s_mode_utils::{print::*, sbi_console::SbiConsole};
use sbi::api::{base, pmu, reset, tee_host, tee_interrupt};
use sbi::{
    PmuCounterConfigFlags, PmuCounterStartFlags, PmuCounterStopFlags, PmuEventType, PmuFirmware,
    PmuHardware, SbiMessage, TeeMemoryRegion, EXT_PMU, EXT_TEE_HOST, EXT_TEE_INTERRUPT,
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

// Maximum number of register sets we support in the shared-memory area.
const MAX_REGISTER_SETS: usize = 8;

// Wrapper for a vCPU shared-memory state area with a layout provided by the TSM.
//
// TODO: Is there a way to unify this and VmCpuSharedStateRef?
struct TvmCpuSharedMem {
    addr: u64,
    layout: ArrayVec<sbi::RegisterSetLocation, MAX_REGISTER_SETS>,
}

macro_rules! define_accessors {
    ($regset:ident, $field:ident, $get:ident, $set:ident) => {
        #[allow(dead_code)]
        pub fn $get(&self) -> u64 {
            // Safety: The caller guaranteed at construction that `self.adddr` points to a valid
            // shared-memory state area for the layout provided by the TSM.
            unsafe { ptr::addr_of!((*self.$regset()).$field).read_volatile() }
        }

        #[allow(dead_code)]
        pub fn $set(&self, val: u64) {
            // Safety: The caller guaranteed at construction that `self.adddr` points to a valid
            // shared-memory state area for the layout provided by the TSM.
            unsafe { ptr::addr_of_mut!((*self.$regset()).$field).write_volatile(val) };
        }
    };
}

impl TvmCpuSharedMem {
    // Creates a new `TvmCpuSharedMem` starting at `addr` using the specified `layout`.
    //
    // Safety: `addr` must be aligned and point to a sufficiently large contiguous range of pages
    // to hold the structure described by `layout`. This memory must remain valid and not be
    // accessed for any other purpose for the lifetime of this structure.
    unsafe fn new(
        addr: u64,
        layout: ArrayVec<sbi::RegisterSetLocation, MAX_REGISTER_SETS>,
    ) -> Self {
        Self { addr, layout }
    }

    // Returns the address of the given register set.
    fn register_set_addr(&self, id: sbi::RegisterSetId) -> u64 {
        let offset = self
            .layout
            .iter()
            .find(|e| e.id == id as u16)
            .map(|e| e.offset)
            .expect("Failed to find register set");
        self.addr + offset as u64
    }

    fn s_csrs(&self) -> *mut sbi::SupervisorCsrs {
        self.register_set_addr(sbi::RegisterSetId::SupervisorCsrs) as *mut _
    }

    define_accessors! {s_csrs, sepc, sepc, set_sepc}
    define_accessors! {s_csrs, scause, scause, set_scause}
    define_accessors! {s_csrs, stval, stval, set_stval}

    fn hs_csrs(&self) -> *mut sbi::HypervisorCsrs {
        self.register_set_addr(sbi::RegisterSetId::HypervisorCsrs) as *mut _
    }

    define_accessors! {hs_csrs, htval, htval, set_htval}
    define_accessors! {hs_csrs, htinst, htinst, set_htinst}

    fn gprs(&self) -> *mut sbi::Gprs {
        self.register_set_addr(sbi::RegisterSetId::Gprs) as *mut _
    }

    // Gets the general purpose register at `index`.
    fn gpr(&self, index: GprIndex) -> u64 {
        // Safety: `index` is guaranteed to be a valid GPR index and the caller guaranteed at
        // construction that `self.addr` points to a valid shared-memory state area for the
        // layout provided by the TSM.
        unsafe { ptr::addr_of!((*self.gprs()).0[index as usize]).read_volatile() }
    }

    // Sets the general purpose register at `index`.
    fn set_gpr(&self, index: GprIndex, val: u64) {
        // Safety: `index` is guaranteed to be a valid GPR index and the caller guaranteed at
        // construction that `self.addr` points to a valid shared-memory state area for the
        // layout provided by the TSM.
        unsafe { ptr::addr_of_mut!((*self.gprs()).0[index as usize]).write_volatile(val) };
    }

    // Returns the number of pages required for a shared-memory state area with the given layout.
    fn required_pages(layout: &[sbi::RegisterSetLocation]) -> u64 {
        let bytes = layout.iter().fold(0, |acc, e| {
            let id = sbi::RegisterSetId::from_raw(e.id).expect("Invalid RegisterSetId");
            acc + id.struct_size() as u64
        });
        (bytes + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K
    }
}

// Safety: addr must point to `num_pages` of memory that isn't currently used by this program. This
// memory will be overwritten and access will be removed.
unsafe fn convert_pages(addr: u64, num_pages: u64) {
    tee_host::convert_pages(addr, num_pages).expect("TsmConvertPages failed");

    // Fence the pages we just converted.
    //
    // TODO: Boot secondary CPUs and test the invalidation flow with multiple CPUs.
    tee_host::initiate_fence().expect("Tellus - TsmInitiateFence failed");
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

#[cfg(target_feature = "v")]
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

#[cfg(target_feature = "v")]
fn check_vectors() {
    println!("Reading vector registers");
    const REG_WIDTH_IN_U64S: usize = 4;

    let mut inbuf = [0_u64; (32 * REG_WIDTH_IN_U64S)];
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
    const NUM_VCPUS: u64 = 1;
    const NUM_TEE_PTE_PAGES: u64 = 10;

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

    base::probe_sbi_extension(EXT_TEE_HOST).expect("Platform doesn't support TEE extension");
    let tsm_info = tee_host::get_info().expect("Tellus - TsmGetInfo failed");
    let tvm_create_pages = 4
        + tsm_info.tvm_state_pages
        + ((NUM_VCPUS * tsm_info.tvm_bytes_per_vcpu) + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
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
    let tvm_vcpu_addr = tvm_state_addr + tsm_info.tvm_state_pages * PAGE_SIZE_4K;
    let vmid = tee_host::tvm_create(
        tvm_page_directory_addr,
        tvm_state_addr,
        NUM_VCPUS,
        tvm_vcpu_addr,
    )
    .expect("Tellus - TvmCreate returned error");
    println!("Tellus - TvmCreate Success vmid: {vmid:x}");
    next_page += PAGE_SIZE_4K * tvm_create_pages;

    // Add pages for the page table
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_TEE_PTE_PAGES);
    }
    tee_host::add_page_table_pages(vmid, next_page, NUM_TEE_PTE_PAGES)
        .expect("Tellus - AddPageTablePages returned error");
    next_page += PAGE_SIZE_4K * NUM_TEE_PTE_PAGES;

    // Get the layout of the shared-memory state area.
    let mut vcpu_mem_layout = ArrayVec::new();
    let num_regsets =
        tee_host::num_vcpu_register_sets(vmid).expect("Tellus - TvmCpuNumRegisterSets failed");
    assert!(num_regsets <= MAX_REGISTER_SETS as u64);
    for i in 0..num_regsets {
        let regset =
            tee_host::get_vcpu_register_set(vmid, i).expect("Tellus - TvmCpuGetRegisterSet");
        vcpu_mem_layout.push(regset);
    }
    let num_vcpu_shared_pages = TvmCpuSharedMem::required_pages(&vcpu_mem_layout);

    // Add vCPU0.
    let vcpu_state_addr = next_page;
    next_page += num_vcpu_shared_pages * PAGE_SIZE_4K;
    // Safety: We own `vcpu_state_addr` and will only access it through volatile reads/writes.
    unsafe { tee_host::add_vcpu(vmid, 0, vcpu_state_addr) }
        .expect("Tellus - TvmCpuCreate returned error");
    // Safety: `vcpu_state_addr` points to a sufficient number of pages to hold the requested layout
    // and will not be used for any other purpose for the duration of `kernel_init()`.
    let vcpu = unsafe { TvmCpuSharedMem::new(vcpu_state_addr, vcpu_mem_layout) };

    let has_aia = base::probe_sbi_extension(EXT_TEE_INTERRUPT).is_ok();
    // CPU0, guest interrupt file 0.
    let imsic_file_addr = IMSIC_START_ADDRESS + PAGE_SIZE_4K;
    if has_aia {
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

        // Try to convert a guest interfupt file.
        //
        // Safety: We trust that the IMSIC is actually at IMSIC_START_ADDRESS, and we aren't
        // touching this page at all in this program.
        unsafe { tee_interrupt::convert_imsic(imsic_file_addr) }
            .expect("Tellus - TsmConvertImsic failed");
        tee_host::initiate_fence().expect("Tellus - TsmInitiateFence failed");
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
    tee_host::add_confidential_memory_region(
        vmid,
        USABLE_RAM_START_ADDRESS,
        (NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES) * PAGE_SIZE_4K,
    )
    .expect("Tellus - TvmAddConfidentialMemoryRegion failed");

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

    // Convert the zero pages.
    let zero_pages_base = next_page;
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    unsafe {
        convert_pages(next_page, NUM_GUEST_ZERO_PAGES);
    }

    next_page += NUM_GUEST_ZERO_PAGES * PAGE_SIZE_4K;
    let shared_page_base = next_page;

    // Set the entry point.
    vcpu.set_sepc(0x8020_0000);
    // Set the kernel_init() parameter.
    vcpu.set_gpr(GprIndex::A1, GUEST_SHARED_PAGES_START_ADDRESS);

    // TODO test that access to pages crashes somehow
    tee_host::tvm_finalize(vmid).expect("Tellus - Finalize returned error");

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

    #[cfg(target_feature = "v")]
    store_into_vectors();

    let mut shared_mem_region: Option<Range<u64>> = None;
    let mut mmio_region: Option<Range<u64>> = None;
    loop {
        // Safety: running a VM will only write the `TvmCpuSharedState` struct that was registered
        // with `add_vcpu()`.
        tee_host::tvm_run(vmid, 0).expect("Could not run guest VM");
        let scause = vcpu.scause();
        if let Ok(Trap::Exception(e)) = Trap::from_scause(scause) {
            use Exception::*;
            match e {
                VirtualSupervisorEnvCall => {
                    // Read the ECALL arguments written to the A* regs in shared memory.
                    let mut a_regs = [0u64; 8];
                    for (i, reg) in a_regs.iter_mut().enumerate() {
                        // Unwrap ok: A[0-7] are valid GPR indices.
                        let index = GprIndex::from_raw(GprIndex::A0 as u32 + i as u32).unwrap();
                        *reg = vcpu.gpr(index);
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
                                AddMemoryRegion {
                                    region_type,
                                    addr,
                                    len,
                                } => {
                                    use TeeMemoryRegion::*;
                                    match region_type {
                                        Shared => {
                                            shared_mem_region = Some(Range {
                                                start: addr,
                                                end: addr + len,
                                            });
                                        }
                                        EmulatedMmio => {
                                            mmio_region = Some(Range {
                                                start: addr,
                                                end: addr + len,
                                            });
                                        }
                                        _ => {
                                            println!(
                                                "Unexpected memory region {:?} from guest",
                                                region_type
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            println!("Unexpected ECALL from guest");
                            break;
                        }
                    }
                }
                GuestLoadPageFault | GuestStorePageFault => {
                    let fault_addr = (vcpu.htval() << 2) | (vcpu.stval() & 0x3);
                    match fault_addr {
                        GUEST_ZERO_PAGES_START_ADDRESS..=GUEST_ZERO_PAGES_END_ADDRESS => {
                            // Fault in the page.
                            if zero_pages_added >= NUM_GUEST_ZERO_PAGES {
                                panic!("Ran out of pages to fault in");
                            }
                            tee_host::add_zero_pages(
                                vmid,
                                zero_pages_base + zero_pages_added * PAGE_SIZE_4K,
                                sbi::TsmPageType::Page4k,
                                1,
                                fault_addr & !(PAGE_SIZE_4K - 1),
                            )
                            .expect("Tellus - TvmAddZeroPages failed");
                            zero_pages_added += 1;
                        }
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
                            let inst = DecodedInstruction::from_raw(vcpu.htinst() as u32)
                                .expect("Failed to decode faulting MMIO instruction")
                                .instruction();
                            // Handle the load or store; the source/dest register is always A0.
                            use Instruction::*;
                            match inst {
                                Lb(_) | Lbu(_) | Lh(_) | Lhu(_) | Lw(_) | Lwu(_) | Ld(_) => {
                                    vcpu.set_gpr(GprIndex::A0, 0x42);
                                }
                                Sb(_) | Sh(_) | Sw(_) | Sd(_) => {
                                    let val = vcpu.gpr(GprIndex::A0);
                                    println!("Guest says: 0x{:x} at 0x{:x}", val, fault_addr);
                                }
                                _ => {
                                    println!("Unexpected guest MMIO instruction: {:?}", inst);
                                    return;
                                }
                            }
                        }
                        _ => {
                            println!("Unhandled guest page fault at 0x{:x}", fault_addr);
                            break;
                        }
                    }
                }
                VirtualInstruction => {
                    let inst = DecodedInstruction::from_raw(vcpu.stval() as u32)
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
                _ => {
                    println!("Guest VM terminated with exception {:?}", e);
                    break;
                }
            }
        } else {
            println!("Guest VM terminated with unexpected cause 0x{:x}", scause);
        }
    }

    #[cfg(target_feature = "v")]
    check_vectors();

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
        NUM_GUEST_DATA_PAGES + NUM_GUEST_ZERO_PAGES,
    );
    reclaim_pages(state_pages_base, tvm_create_pages);
    if has_aia {
        tee_interrupt::reclaim_imsic(imsic_file_addr).expect("Tellus - TsmReclaimImsic failed");
    }
    exercise_pmu_functionality();
    println!("Tellus - All OK");
    poweroff();
}

#[no_mangle]
extern "C" fn secondary_init(_hart_id: u64) {}
