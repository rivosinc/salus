// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::arch::asm;
use core::cell::{RefCell, RefMut};
use drivers::{imsic::Imsic, CpuId, CpuInfo};
use page_tracking::{HwMemMap, HwMemRegionType, HwReservedMemType};
use riscv_pages::{PageSize, RawAddr, SupervisorPageAddr};
use riscv_regs::{sstatus, ReadWriteable, CSR};
use s_mode_utils::print::*;
use sbi::api::state;
use spin::Once;

use crate::hyp_map::HypPageTable;
use crate::umode::UmodeTask;
use crate::vm_id::VmIdTracker;

// The secondary CPU entry point, defined in start.S.
extern "C" {
    fn _secondary_start();
}

/// Per-CPU data. A pointer to this struct is loaded into TP when a CPU starts. This structure
/// sits at the top of a secondary CPU's stack.
#[repr(C)]
pub struct PerCpu {
    cpu_id: CpuId,
    vmid_tracker: RefCell<VmIdTracker>,
    page_table: Once<HypPageTable>,
    umode_task: Once<UmodeTask>,
    online: Once<bool>,
}

/// The number of pages we allocate per CPU: the CPU's stack + it's `PerCpu` structure.
const PER_CPU_PAGES: u64 = 8;

/// The base address of the per-CPU memory region.
static PER_CPU_BASE: Once<SupervisorPageAddr> = Once::new();

impl PerCpu {
    /// Initializes the `PerCpu` structures for each CPU, taking memory from `mem_map`. This (the
    /// boot CPU's) per-CPU area is initialized and loaded into TP as well.
    pub fn init(boot_hart_id: u64, mem_map: &mut HwMemMap) {
        let cpu_info = CpuInfo::get();

        // Find somewhere to put the per-CPU memory.
        let total_size = PER_CPU_PAGES * cpu_info.num_cpus() as u64 * PageSize::Size4k as u64;
        let pcpu_base = mem_map
            .regions()
            .find(|r| r.region_type() == HwMemRegionType::Available && r.size() >= total_size)
            .map(|r| r.base())
            .expect("Not enough free memory for per-CPU area");
        mem_map
            .reserve_region(
                HwReservedMemType::HypervisorPerCpu,
                RawAddr::from(pcpu_base),
                total_size,
            )
            .unwrap();
        PER_CPU_BASE.call_once(|| pcpu_base);

        // Now initialize each PerCpu structure.
        for i in 0..cpu_info.num_cpus() {
            let cpu_id = CpuId::new(i);
            let ptr = Self::ptr_for_cpu(cpu_id);
            let pcpu = PerCpu {
                cpu_id,
                vmid_tracker: RefCell::new(VmIdTracker::new()),
                page_table: Once::new(),
                umode_task: Once::new(),
                online: Once::new(),
            };
            // Safety: ptr is guaranteed to be properly aligned and point to valid memory owned by
            // PerCpu. No other CPUs are alive at this point, so it cannot be concurrently modified
            // either.
            unsafe { core::ptr::write(ptr as *mut PerCpu, pcpu) };
        }

        // Load TP with the address of our PerCpu struct so that we're consistent with secondary
        // CPUs once they're brought up.
        let my_tp = Self::ptr_for_cpu(cpu_info.hart_id_to_cpu(boot_hart_id as u32).unwrap()) as u64;
        unsafe {
            // Safe since we're the only users of TP.
            asm!("mv tp, {rs}", rs = in(reg) my_tp)
        };

        let me = Self::this_cpu();
        me.set_online();
    }

    /// Returns a pointer to the `PerCpu` for the given CPU.
    fn ptr_for_cpu(cpu_id: CpuId) -> *const PerCpu {
        let cpu_end = PER_CPU_BASE
            .get()
            .unwrap()
            .checked_add_pages((1 + cpu_id.raw() as u64) * PER_CPU_PAGES)
            .unwrap();
        let pcpu_addr = cpu_end.bits() - core::mem::size_of::<PerCpu>() as u64;
        pcpu_addr as *const PerCpu
    }

    /// Returns this CPU's `PerCpu` structure.
    pub fn this_cpu() -> &'static PerCpu {
        assert!(PER_CPU_BASE.get().is_some()); // Make sure PerCpu has been set up.
        let tp: u64;
        unsafe {
            // Safe since we're the only users of TP.
            asm!("mv {rd}, tp", rd = out(reg) tp)
        };
        let pcpu_ptr = tp as *const PerCpu;
        let pcpu = unsafe {
            // Safe since TP is set up to point to a valid PerCpu struct in init().
            pcpu_ptr.as_ref().unwrap()
        };
        pcpu
    }

    /// Set the CPU pagetable (once). Must be called after `PerCpu::init()`.
    pub fn set_cpu_page_table(cpu: CpuId, page_table: HypPageTable) {
        let pcpu = Self::ptr_for_cpu(cpu);
        // Safe since pcpu is set up to point to a valid PerCpu struct in init().
        unsafe {
            (*pcpu).page_table.call_once(|| page_table);
        }
    }

    /// Returns this CPU's ID.
    pub fn cpu_id(&self) -> CpuId {
        self.cpu_id
    }

    /// Marks this CPU as online.
    pub fn set_online(&self) {
        self.online.call_once(|| true);
    }

    /// Set the CPU umode task (once). Must be called after `PerCpu::init()`.
    pub fn set_umode_task(&self, umode_task: UmodeTask) {
        self.umode_task.call_once(|| umode_task);
    }

    /// Get the CPU page table. Must be called after `set_cpu_page_table` has been called for this
    /// cpu.
    pub fn page_table(&self) -> &HypPageTable {
        // Unwrap okay: this is called after `set_cpu_page_table`
        self.page_table.get().unwrap()
    }

    /// Get the  CPU umode structure. Must be  called after `set_umode_task` has been  called for this
    /// cpu.
    pub fn umode_task(&self) -> &UmodeTask {
        // Unwrap okay: this is called after `set_umode_task`
        self.umode_task.get().unwrap()
    }

    /// Returns a mutable reference to this CPU's VMID tracker.
    pub fn vmid_tracker_mut(&self) -> RefMut<VmIdTracker> {
        self.vmid_tracker.borrow_mut()
    }
}

// PerCpu state obviously cannot be shared between threads.
impl !Sync for PerCpu {}

/// Halts this CPU until an interrupt (for example, delivered via `kick_cpu()`) is received.
pub fn wfi() {
    CSR.sstatus.modify(sstatus::sie.val(1));
    // Safety: WFI behavior is well-defined.
    unsafe { asm!("wfi", options(nomem, nostack)) };
    CSR.sstatus.modify(sstatus::sie.val(0));
}

/// Sends an IPI to `cpu`.
pub fn send_ipi(cpu: CpuId) {
    Imsic::get().send_ipi(cpu).unwrap();
}

/// Boots secondary CPUs, using the HSM SBI call. Upon return, all secondary CPUs will have
/// entered secondary_init().
pub fn start_secondary_cpus() {
    let cpu_info = CpuInfo::get();
    let boot_cpu = PerCpu::this_cpu().cpu_id();
    for i in 0..cpu_info.num_cpus() {
        let cpu_id = CpuId::new(i);
        if cpu_id == boot_cpu {
            continue;
        }

        // Start the hart with it's PerCpu struct in A1; _secondary_start will stash it in TP.
        let pcpu = PerCpu::ptr_for_cpu(cpu_id);
        // Safety: _secondary_start is guaranteed by the linker to be the code to start secondary
        // CPUs. pcpu will only be shared with one cpu.
        unsafe {
            state::hart_start(
                cpu_info.cpu_to_hart_id(cpu_id).unwrap() as u64,
                (_secondary_start as *const fn()) as u64,
                pcpu as u64,
            )
            .expect("Failed to start CPU {i}");
        }

        // Synchronize with the CPU coming online. TODO: Timeout?
        let pcpu = unsafe {
            // Safe since TP is set up to point to a valid PerCpu struct in init().
            pcpu.as_ref().unwrap()
        };
        pcpu.online.wait();
    }

    println!("Brought online {} CPU(s)", cpu_info.num_cpus());
}
