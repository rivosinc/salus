// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::{ArrayString, ArrayVec};
use core::{fmt, num, ops::ControlFlow, slice};
use device_tree::{DeviceTree, DeviceTreeResult, DeviceTreeSerializer};
use drivers::{imsic::*, iommu::*, pci::*, CpuId, CpuInfo};
use page_tracking::collections::PageBox;
use page_tracking::{HwMemRegion, HypPageAlloc, PageList, PageTracker};
use riscv_page_tables::{GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::*;
use riscv_regs::{
    DecodedInstruction, Exception, GeneralPurposeRegisters, GprIndex, Instruction, Trap,
    CSR_HTINST, CSR_HTVAL, CSR_SCAUSE, CSR_STVAL,
};
use s_mode_utils::print::*;
use sbi::{self, DebugConsoleFunction, SbiMessage, StateFunction};

use crate::guest_tracking::{GuestVm, Guests, Result as GuestTrackingResult};
use crate::smp;
use crate::vm::{FinalizedVm, Vm};
use crate::vm_cpu::{VmCpu, VmCpuExitReporting, VmCpuParent, VmCpus};
use crate::vm_pages::VmPages;

// Where the kernel, initramfs, and FDT will be located in the guest physical address space.
//
// TODO: Kernel offset should be pulled from the header in the kernel image.
const KERNEL_OFFSET: u64 = 0x20_0000;
const INITRAMFS_OFFSET: u64 = KERNEL_OFFSET + 0x800_0000;
// Assuming RAM base at 2GB, ends up at 3GB - 16MB which is consistent with QEMU.
const FDT_OFFSET: u64 = 0x3f00_0000;

// A builder for the host VM's device-tree. Starting with the hypervisor's device-tree, makes the
// necessary modifications to create a device-tree that reflects the hardware available to the
// host VM.
struct HostDtBuilder {
    tree: DeviceTree,
}

impl HostDtBuilder {
    // Creates a new builder from the hypervisor device-tree.
    fn new(hyp_dt: &DeviceTree) -> DeviceTreeResult<Self> {
        let mut host_dt = DeviceTree::new();
        let hyp_root = hyp_dt.get_node(hyp_dt.root().unwrap()).unwrap();
        let host_root_id = host_dt.add_node("", None)?;
        let host_root = host_dt.get_mut_node(host_root_id).unwrap();

        // Clone the properties of the root node as-is.
        host_root.set_props(hyp_root.props().cloned())?;

        // Add a 'chosen' node and copy the bootargs if they exist.
        let host_chosen_id = host_dt.add_node("chosen", Some(host_root_id))?;
        let host_chosen = host_dt.get_mut_node(host_chosen_id).unwrap();
        if let Some(hyp_chosen) = hyp_dt.iter().find(|n| n.name() == "chosen") {
            if let Some(p) = hyp_chosen.props().find(|p| p.name() == "bootargs") {
                host_chosen.insert_prop(p.clone())?;
            }
        }

        Ok(Self { tree: host_dt })
    }

    // Adds a "memory" node to the device tree with the given base and size.
    fn add_memory_node(mut self, mem_base: GuestPhysAddr, mem_size: u64) -> DeviceTreeResult<Self> {
        let mut mem_name = ArrayString::<32>::new();
        fmt::write(
            &mut mem_name,
            format_args!("memory@{:08x}", mem_base.bits()),
        )
        .unwrap();
        let mem_id = self.tree.add_node(mem_name.as_str(), self.tree.root())?;
        let mem_node = self.tree.get_mut_node(mem_id).unwrap();
        mem_node.add_prop("device_type")?.set_value_str("memory")?;
        // TODO: Assumes #address-cells/#size-cells of 2.
        mem_node
            .add_prop("reg")?
            .set_value_u64(&[mem_base.bits(), mem_size])?;

        Ok(self)
    }

    // Adds CPU nodes to the device tree.
    fn add_cpu_nodes(mut self) -> DeviceTreeResult<Self> {
        CpuInfo::get().add_host_cpu_nodes(&mut self.tree)?;
        Ok(self)
    }

    // Add any MMIO devices to the device tree.
    fn add_device_nodes(mut self) -> DeviceTreeResult<Self> {
        // First add a 'soc' subnode of the root.
        let soc_node_id = self.tree.add_node("soc", self.tree.root())?;
        let soc_node = self.tree.get_mut_node(soc_node_id).unwrap();
        soc_node
            .add_prop("compatible")?
            .set_value_str("simple-bus")?;
        soc_node.add_prop("#address-cells")?.set_value_u32(&[2])?;
        soc_node.add_prop("#size-cells")?.set_value_u32(&[2])?;
        soc_node.add_prop("ranges")?;

        Imsic::get().add_host_imsic_node(&mut self.tree)?;
        PcieRoot::get().add_host_pcie_node(&mut self.tree)?;

        Ok(self)
    }

    // Updates the "chosen" node with the location of the initramfs image.
    fn set_initramfs_addr(mut self, start_addr: GuestPhysAddr, len: u64) -> DeviceTreeResult<Self> {
        let chosen_id = self
            .tree
            .iter()
            .find(|n| n.name().starts_with("chosen"))
            .unwrap()
            .id();
        let chosen_node = self.tree.get_mut_node(chosen_id).unwrap();

        chosen_node
            .add_prop("linux,initrd-start")?
            .set_value_u64(&[start_addr.bits()])?;
        let end_addr = start_addr.checked_increment(len).unwrap();
        chosen_node
            .add_prop("linux,initrd-end")?
            .set_value_u64(&[end_addr.bits()])?;

        Ok(self)
    }

    fn tree(self) -> DeviceTree {
        self.tree
    }
}

enum FdtPages {
    Clean(SequentialPages<ConvertedClean>),
    Initialized(SequentialPages<ConvertedInitialized>),
}

/// Loader for the host VM.
///
/// Given the hypervisor's device tree and the host kernel & initramfs image, build a device-tree
/// and address space for the host VM.
///
/// In order to allow the host VM to allocate physically-aligned blocks necessary for guest VM
/// creation (specifically, the root of the G-stage page-table), we guarantee that each
/// contiguous T::TOP_LEVEL_ALIGN block of the guest physical address space of the host VM maps to
/// a contiguous T::TOP_LEVEL_ALIGN block of the host physical address space.
pub struct HostVmLoader<T: GuestStagePagingMode> {
    hypervisor_dt: DeviceTree,
    kernel: HwMemRegion,
    initramfs: Option<HwMemRegion>,
    vm: HostVm<T>,
    fdt_pages: FdtPages,
    zero_pages: PageList<Page<ConvertedClean>>,
    guest_ram_base: GuestPhysAddr,
    ram_size: u64,
}

impl<T: GuestStagePagingMode> HostVmLoader<T> {
    /// Creates a new loader with the given device-tree and kernel & initramfs images. Uses
    /// `page_alloc` to allocate any additional pages that are necessary to load the VM.
    pub fn new(
        hypervisor_dt: DeviceTree,
        kernel: HwMemRegion,
        initramfs: Option<HwMemRegion>,
        guest_ram_base: GuestPhysAddr,
        guest_phys_size: u64,
        mut page_alloc: HypPageAlloc,
    ) -> Self {
        // Reserve a contiguous chunk for the host's FDT. We assume it will be no bigger than the
        // size of the hypervisor's FDT and we align it to `T::TOP_LEVEL_ALIGN` to maintain the
        // contiguous mapping guarantee from GPA -> HPA mentioned above.
        let fdt_size = {
            let size = DeviceTreeSerializer::new(&hypervisor_dt).output_size();
            ((size as u64) + T::TOP_LEVEL_ALIGN - 1) & !(T::TOP_LEVEL_ALIGN - 1)
        };
        let num_fdt_pages = fdt_size / PageSize::Size4k as u64;
        let fdt_pages =
            page_alloc.take_pages(num_fdt_pages.try_into().unwrap(), T::TOP_LEVEL_ALIGN);

        let (zero_pages, vm) = HostVm::from_hyp_mem(page_alloc, guest_phys_size);

        // Now that the hypervisor is done claiming memory, determine the actual size of the host's
        // address space.
        let ram_size = zero_pages.len() as u64 * PageSize::Size4k as u64
            + fdt_pages.length_bytes()
            + kernel.size()
            + initramfs.map(|r| r.size()).unwrap_or(0);
        assert!(ram_size >= FDT_OFFSET + fdt_pages.length_bytes());

        Self {
            hypervisor_dt,
            kernel,
            initramfs,
            vm,
            fdt_pages: FdtPages::Clean(fdt_pages),
            zero_pages,
            guest_ram_base,
            ram_size,
        }
    }

    /// Builds a device tree for the host VM, flattening it to a range of pages that will be
    /// mapped into the address space in `build_address_space()`.
    pub fn build_device_tree(mut self) -> Self {
        let fdt_pages = match self.fdt_pages {
            FdtPages::Clean(pages) => pages,
            _ => panic!("Device tree already written"),
        };

        // Construct a stripped-down device-tree for the host VM.
        let mut host_dt_builder = HostDtBuilder::new(&self.hypervisor_dt)
            .unwrap()
            .add_memory_node(self.guest_ram_base, self.ram_size)
            .unwrap()
            .add_cpu_nodes()
            .unwrap()
            .add_device_nodes()
            .unwrap();
        if let Some(r) = self.initramfs {
            host_dt_builder = host_dt_builder
                .set_initramfs_addr(
                    self.guest_ram_base
                        .checked_increment(INITRAMFS_OFFSET)
                        .unwrap(),
                    r.size(),
                )
                .unwrap();
        }
        let host_dt = host_dt_builder.tree();

        println!("Host DT: {}", host_dt);

        // Serialize the device-tree.
        let dt_writer = DeviceTreeSerializer::new(&host_dt);
        assert!(dt_writer.output_size() <= fdt_pages.length_bytes().try_into().unwrap());
        let fdt_slice = unsafe {
            // Safe because we own these pages.
            slice::from_raw_parts_mut(
                fdt_pages.base().bits() as *mut u8,
                fdt_pages.length_bytes().try_into().unwrap(),
            )
        };
        dt_writer.write_to(fdt_slice);
        let fdt_pages =
            SequentialPages::from_pages(fdt_pages.into_iter().map(|p| p.to_initialized_page()))
                .unwrap();
        self.fdt_pages = FdtPages::Initialized(fdt_pages);
        self
    }

    /// Constructs the address space for the host VM, returning a `HostVm` that is ready to run.
    pub fn build_address_space(mut self) -> HostVm<T> {
        let imsic = Imsic::get();
        let cpu_info = CpuInfo::get();
        // Map the IMSIC interrupt files into the guest address space. The host VM's interrupt
        // file gets mapped to the location of the supervisor interrupt file.
        for i in 0..cpu_info.num_cpus() {
            let cpu_id = CpuId::new(i);
            let imsic_pages = imsic.take_guest_files(cpu_id).unwrap();
            self.vm.add_imsic_pages(cpu_id, imsic_pages);
        }

        let pci = PcieRoot::get();
        pci.take_host_devices();
        // Identity-map the PCIe BAR resources.
        for (res_type, range) in pci.resources() {
            let gpa =
                PageAddr::new(RawAddr::guest(range.base().bits(), PageOwnerId::host())).unwrap();
            self.vm.add_pci_region(gpa, range.length_bytes());
            let pages = pci.take_host_resource(res_type).unwrap();
            self.vm.add_pci_pages(gpa, pages);
        }
        // Attach our PCI devices to the IOMMU.
        if Iommu::get().is_some() {
            for dev in pci.devices() {
                let mut dev = dev.lock();
                if dev.owner() == Some(PageOwnerId::host()) {
                    // Silence buggy clippy warning.
                    #[allow(clippy::explicit_auto_deref)]
                    self.vm.attach_pci_device(&mut *dev);
                }
            }
        }

        // Host guarantees that the host pages it returns start at T::TOP_LEVEL_ALIGN-aligned block,
        // and because we built the HwMemMap with a minimum region alignment of T::TOP_LEVEL_ALIGN
        // any discontiguous ranges are also guaranteed to be aligned.
        //
        // Now fill in the address space, inserting zero pages around the kernel/initramfs/FDT.
        let mut current_gpa = PageAddr::new(self.guest_ram_base).unwrap();
        self.vm
            .add_confidential_memory_region(current_gpa, self.ram_size);

        let mut zero_ranges = ArrayVec::<_, 3>::new();
        let num_pages = KERNEL_OFFSET / PageSize::Size4k as u64;
        zero_ranges.push(PageAddrRange::new(current_gpa, num_pages));
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let num_kernel_pages = self.kernel.size() / PageSize::Size4k as u64;
        let kernel_pages: SequentialPages<ConvertedInitialized> = unsafe {
            // Safe because HwMemMap reserved this region.
            SequentialPages::from_mem_range(self.kernel.base(), PageSize::Size4k, num_kernel_pages)
                .unwrap()
        };
        self.vm
            .add_measured_pages(current_gpa, kernel_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_kernel_pages).unwrap();

        if let Some(r) = self.initramfs {
            let num_pages =
                (INITRAMFS_OFFSET - (KERNEL_OFFSET + self.kernel.size())) / PageSize::Size4k as u64;
            zero_ranges.push(PageAddrRange::new(current_gpa, num_pages));
            current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

            let num_initramfs_pages = r.size() / PageSize::Size4k as u64;
            let initramfs_pages: SequentialPages<ConvertedInitialized> = unsafe {
                // Safe because HwMemMap reserved this region.
                SequentialPages::from_mem_range(r.base(), PageSize::Size4k, num_initramfs_pages)
                    .unwrap()
            };
            self.vm
                .add_measured_pages(current_gpa, initramfs_pages.into_iter());
            current_gpa = current_gpa.checked_add_pages(num_initramfs_pages).unwrap();
        }

        let num_pages = (FDT_OFFSET - (current_gpa.bits() - self.guest_ram_base.bits()))
            / PageSize::Size4k as u64;
        zero_ranges.push(PageAddrRange::new(current_gpa, num_pages));
        current_gpa = current_gpa.checked_add_pages(num_pages).unwrap();

        let fdt_pages = match self.fdt_pages {
            FdtPages::Initialized(pages) => pages,
            _ => panic!("FDT pages not initialized"),
        };
        let num_fdt_pages = fdt_pages.len();
        self.vm
            .add_measured_pages(current_gpa, fdt_pages.into_iter());
        current_gpa = current_gpa.checked_add_pages(num_fdt_pages).unwrap();

        self.vm
            .finalize(
                self.guest_ram_base
                    .checked_increment(KERNEL_OFFSET)
                    .unwrap(),
                self.guest_ram_base.checked_increment(FDT_OFFSET).unwrap(),
            )
            .unwrap();

        // Fill in the zero pages.
        for r in zero_ranges.iter() {
            self.vm.add_zero_pages(
                r.base(),
                self.zero_pages.by_ref().take(r.num_pages() as usize),
            );
        }
        self.vm.add_zero_pages(current_gpa, self.zero_pages);

        // Set up MMIO emulation for the PCIe config space.
        let config_mem = pci.config_space();
        let config_gpa = PageAddr::new(RawAddr::guest(
            config_mem.base().bits(),
            PageOwnerId::host(),
        ))
        .unwrap();
        self.vm
            .add_mmio_region(config_gpa, config_mem.length_bytes());

        self.vm
    }
}

/// Errors encountered during MMIO emulation.
#[derive(Clone, Copy, Debug)]
enum MmioEmulationError {
    FailedDecode(u32),
    InvalidInstruction(Instruction),
    InvalidAddress(u64),
}

impl core::fmt::Display for MmioEmulationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MmioEmulationError::FailedDecode(i) => write!(f, "FailedDecode {i:x}"),
            MmioEmulationError::InvalidInstruction(i) => write!(f, "InvalidInstruction {i:?}"),
            MmioEmulationError::InvalidAddress(i) => write!(f, "InvalidAddress {i:x}"),
        }
    }
}

#[derive(Default)]
struct HostVmRunner {
    scause: u64,
    stval: u64,
    htval: u64,
    htinst: u64,
    gprs: GeneralPurposeRegisters,
}

impl HostVmRunner {
    fn new() -> Self {
        HostVmRunner::default()
    }

    // Runs `vcpu_id` in `vm`.
    fn run<T: GuestStagePagingMode>(
        &mut self,
        vm: FinalizedVm<T>,
        vcpu_id: u64,
    ) -> ControlFlow<()> {
        // Run until we shut down, or this vCPU stops.
        loop {
            vm.run_vcpu(vcpu_id, VmCpuParent::Tsm(self)).unwrap();
            if let Ok(Trap::Exception(e)) = Trap::from_scause(self.scause) {
                use Exception::*;
                match e {
                    VirtualSupervisorEnvCall => {
                        // Read the ECALL arguments written to the A* regs in shared memory.
                        use SbiMessage::*;
                        match SbiMessage::from_regs(self.gprs.a_regs()) {
                            Ok(Reset(_)) => {
                                println!("Host VM requested shutdown");
                                return ControlFlow::Break(());
                            }
                            Ok(HartState(StateFunction::HartStart { hart_id, .. })) => {
                                smp::send_ipi(CpuId::new(hart_id as usize));
                            }
                            Ok(HartState(StateFunction::HartStop)) => {
                                return ControlFlow::Continue(())
                            }
                            Ok(DebugConsole(DebugConsoleFunction::PutString { len, addr })) => {
                                // Can't do anything about errors right now.
                                let _ = self.handle_put_string(&vm, addr, len);
                            }
                            _ => {
                                println!("Unhandled ECALL from host");
                                return ControlFlow::Break(());
                            }
                        }
                    }
                    GuestLoadPageFault | GuestStorePageFault => {
                        if let Err(err) = self.handle_page_fault(vm.page_tracker()) {
                            println!("Unhandled page fault: {}", err);
                            return ControlFlow::Break(());
                        }
                    }
                    _ => {
                        println!("Unhandled host VM exception {:?}", e);
                        return ControlFlow::Break(());
                    }
                }
            } else {
                println!("Unexpected host VM trap (SCAUSE = 0x{:x})", self.scause);
                return ControlFlow::Break(());
            }
        }
    }

    fn handle_page_fault(
        &mut self,
        page_tracker: PageTracker,
    ) -> core::result::Result<(), MmioEmulationError> {
        // For now, the only thing we're expecting is MMIO emulation faults in PCI config space.
        let addr = (self.htval << 2) | (self.stval & 0x3);
        let pci = PcieRoot::get();
        if addr < pci.config_space().base().bits() {
            return Err(MmioEmulationError::InvalidAddress(addr));
        }
        let offset = addr - pci.config_space().base().bits();
        if offset > pci.config_space().length_bytes() {
            return Err(MmioEmulationError::InvalidAddress(addr));
        }

        // Figure out from HTINST what the MMIO operation was. We know the source/destination is
        // always A0.
        let raw_inst = self.htinst as u32;
        let inst = DecodedInstruction::from_raw(raw_inst)
            .map_err(|_| MmioEmulationError::FailedDecode(raw_inst))?;
        use Instruction::*;
        let (write, width) = match inst.instruction() {
            Lb(_) | Lbu(_) => (false, 1),
            Lh(_) | Lhu(_) => (false, 2),
            Lw(_) | Lwu(_) => (false, 4),
            Ld(_) => (false, 8),
            Sb(_) => (true, 1),
            Sh(_) => (true, 2),
            Sw(_) => (true, 4),
            Sd(_) => (true, 8),
            i => {
                return Err(MmioEmulationError::InvalidInstruction(i));
            }
        };

        if write {
            let val = self.gprs.reg(GprIndex::A0);
            pci.emulate_config_write(offset, val, width, page_tracker, PageOwnerId::host());
        } else {
            let val = pci.emulate_config_read(offset, width, page_tracker, PageOwnerId::host());
            self.gprs.set_reg(GprIndex::A0, val);
        }

        Ok(())
    }

    fn handle_put_string<T: GuestStagePagingMode>(
        &mut self,
        vm: &FinalizedVm<T>,
        addr: u64,
        len: u64,
    ) -> core::result::Result<u64, u64> {
        // Pin the pages that we'll be printing from. We assume that the buffer is physically
        // contiguous, which should be the case since that's how we set up the host VM's address
        // space.
        let page_addr = GuestPageAddr::with_round_down(
            GuestPhysAddr::guest(addr, vm.page_owner_id()),
            PageSize::Size4k,
        );
        let offset = addr - page_addr.bits();
        let num_pages = PageSize::num_4k_pages(offset + len);
        let pinned = vm
            .vm_pages()
            .pin_shared_pages(page_addr, num_pages)
            .map_err(|_| 0u64)?;

        // Print the bytes in chunks. We copy to a temporary buffer as the bytes could be modified
        // concurrently by the VM on another CPU.
        let mut copied = 0;
        let mut hyp_addr = pinned.range().base().bits() + offset;
        while copied != len {
            let mut buf = [0u8; 256];
            let to_copy = core::cmp::min(buf.len(), (len - copied) as usize);
            for c in buf.iter_mut() {
                // Safety: We've confirmed that the address is within a region of accessible memory
                // and cannot be remapped as long as we hold the pin. `u8`s are always aligned and
                // properly initialized.
                *c = unsafe { core::ptr::read_volatile(hyp_addr as *const u8) };
                hyp_addr += 1;
            }
            let s = core::str::from_utf8(&buf[..to_copy]).map_err(|_| copied)?;
            print!("{s}");
            copied += to_copy as u64;
        }

        Ok(len)
    }
}

impl VmCpuExitReporting for HostVmRunner {
    fn set_csr(&mut self, csr_num: u16, val: u64) {
        // We could use the actual CSRs here, but if we have the values already sticking them in
        // memory is probably faster.
        match csr_num {
            CSR_SCAUSE => {
                self.scause = val;
            }
            CSR_STVAL => {
                self.stval = val;
            }
            CSR_HTVAL => {
                self.htval = val;
            }
            CSR_HTINST => {
                self.htinst = val;
            }
            _ => (),
        }
    }

    fn set_guest_gpr(&mut self, index: GprIndex, val: u64) {
        self.gprs.set_reg(index, val);
    }

    fn guest_gpr(&self, index: GprIndex) -> u64 {
        self.gprs.reg(index)
    }
}

// Pages used by the `PageVec` for the Host VM guest tracking.
const HOSTVM_GUEST_TRACKING_PAGES: usize = 2;

/// Represents the special VM that serves as the host for the system.
pub struct HostVm<T: GuestStagePagingMode> {
    inner: GuestVm<T>,
}

impl<T: GuestStagePagingMode> HostVm<T> {
    // Creates an initializing host VM with an expected guest physical address space size of
    // `host_gpa_size` from the hypervisor page allocator. Returns the remaining free pages
    // from the allocator, along with the newly constructed `HostVm`.
    fn from_hyp_mem(
        mut hyp_mem: HypPageAlloc,
        host_gpa_size: u64,
    ) -> (PageList<Page<ConvertedClean>>, Self) {
        let root_table_pages =
            hyp_mem.take_pages_for_host_state_with_alignment(4, T::TOP_LEVEL_ALIGN);
        let num_pte_pages = T::max_pte_pages(host_gpa_size / PageSize::Size4k as u64);
        let pte_pages = hyp_mem
            .take_pages_for_host_state(num_pte_pages as usize)
            .into_iter();
        let vm_state_pages =
            hyp_mem.take_pages_for_host_state(GuestVm::<T>::required_pages() as usize);
        let guest_tracking_pages = hyp_mem.take_pages_for_host_state(HOSTVM_GUEST_TRACKING_PAGES);

        // Pages for the array of vCPUs.
        let num_cpus = CpuInfo::get().num_cpus();
        let vcpu_required_state_pages = VmCpus::required_state_pages_per_vcpu();
        let num_vcpu_pages = vcpu_required_state_pages * num_cpus as u64;
        let vcpu_state_pages = hyp_mem.take_pages_for_host_state(num_vcpu_pages as usize);

        let imsic_geometry = Imsic::get().host_vm_geometry();
        // Reserve MSI page table pages if we have an IOMMU.
        let msi_table_pages = Iommu::get().map(|_| {
            let msi_table_size = MsiPageTable::required_table_size(&imsic_geometry);
            hyp_mem.take_pages_for_host_state_with_alignment(
                PageSize::num_4k_pages(msi_table_size) as usize,
                msi_table_size,
            )
        });

        let (page_tracker, host_pages) = PageTracker::from(hyp_mem, T::TOP_LEVEL_ALIGN);
        let root =
            GuestStagePageTable::new(root_table_pages, PageOwnerId::host(), page_tracker.clone())
                .unwrap();
        let vm_pages = VmPages::new(root, 0);
        let init_pages = vm_pages.as_ref();
        init_pages.set_imsic_geometry(imsic_geometry).unwrap();
        for p in pte_pages {
            init_pages.add_pte_page(p).unwrap();
        }
        if let Some(pages) = msi_table_pages {
            init_pages.add_iommu_context(pages).unwrap();
        }

        let vm = Vm::with_guest_tracking(
            vm_pages,
            VmCpus::new(),
            Guests::new(guest_tracking_pages, page_tracker.clone()),
        )
        .unwrap();

        // Unwrap okay, we allocated 'GuestVm::<T>::required_pages()` pages.
        let inner = GuestVm::new(vm, vm_state_pages).unwrap();
        let this = Self { inner };

        {
            let init_vm = this.inner.as_initializing_vm().unwrap();
            let imsic = Imsic::get();
            // Unwrap safe, `vcpu_required_state_pages` must not be zero.
            let mut state_pages_iter = vcpu_state_pages
                .into_chunks_iter(num::NonZeroU64::new(vcpu_required_state_pages).unwrap());
            for i in 0..num_cpus {
                // Allocate vCPU.
                let vcpu = VmCpu::new(i as u64, PageOwnerId::host());
                let vcpu_pages = state_pages_iter.next().unwrap();
                let vcpu_box = PageBox::new_with(vcpu, vcpu_pages, page_tracker.clone());
                init_vm.add_vcpu(vcpu_box).unwrap();

                let imsic_loc = imsic
                    .phys_file_location(CpuId::new(i), ImsicFileId::Supervisor)
                    .unwrap();
                init_vm
                    .set_vcpu_imsic_location(i as u64, imsic_loc)
                    .unwrap();
            }
        }
        (host_pages, this)
    }

    // Adds a region of confidential memory to the host VM.
    fn add_confidential_memory_region(&mut self, addr: GuestPageAddr, len: u64) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages()
            .add_confidential_memory_region(addr, len)
            .unwrap();
    }

    // Adds a PCI BAR memory region to the host VM.
    fn add_pci_region(&mut self, addr: GuestPageAddr, len: u64) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages().add_pci_region(addr, len).unwrap();
    }

    // Adds data pages that are measured and mapped to the page tables for the host. Requires
    // that the GPA map the SPA in T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    fn add_measured_pages<I, S, M>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = Page<S>>,
        S: Assignable<M>,
        M: MeasureRequirement,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        let page_tracker = vm.page_tracker();
        // Unwrap ok since we've donate sufficient PT pages to map the entire address space up front.
        let mapper = vm
            .vm_pages()
            .map_measured_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper
                .map_page(vm_addr, mappable, vm.attestation_mgr())
                .unwrap();
        }
    }

    // Adds the IMSIC pages for `cpu` to the host. The first page in `pages` is set as the
    // host's interrupt file for `cpu` while the remaining pages are added as guest interrupt
    // files for the host to assign.
    fn add_imsic_pages<I>(&mut self, cpu: CpuId, pages: I)
    where
        I: ExactSizeIterator<Item = ImsicGuestPage<ConvertedClean>>,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        // We assigned an IMSIC geometry and vCPU IMSIC locations in `from_hyp_mem()`.
        let location = vm.get_vcpu_imsic_location(cpu.raw() as u64).unwrap();
        let to_addr = vm
            .vm_pages()
            .imsic_geometry()
            .and_then(|g| g.location_to_addr(location))
            .unwrap();
        // Unwrap ok since we've donated sufficient PT pages to map the entire address space up
        // front.
        let mapper = vm
            .vm_pages()
            .map_imsic_pages(to_addr, pages.len() as u64)
            .unwrap();
        let page_tracker = vm.page_tracker();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            // The first guest interrupt file will be the host's virtual supervisor interrupt file,
            // with the remaining files serving as guest interrupt files for the host.
            //
            // TODO: This is sufficient for the host VM since vCPUs are never migrated, but in the
            // event we need to support nested IMSIC virtualization for guest VMs we'll need to be
            // able to bind a vCPU to multiple interrupt files.
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    // Add PCI BAR pages to the host page tables.
    fn add_pci_pages<I>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = PciBarPage<ConvertedClean>>,
    {
        let vm = self.inner.as_initializing_vm().unwrap();
        let page_tracker = vm.page_tracker();
        // Unwrap ok since we've donated sufficient PT pages to map the entire address space up
        // front.
        let mapper = vm
            .vm_pages()
            .map_pci_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    // Attaches the given PCI device to the host VM.
    fn attach_pci_device(&self, dev: &mut PciDevice) {
        let vm = self.inner.as_initializing_vm().unwrap();
        vm.vm_pages().attach_pci_device(dev).unwrap();
    }

    // Completes intialization of the host VM, making it runnable.
    fn finalize(
        &self,
        entry_addr: GuestPhysAddr,
        fdt_addr: GuestPhysAddr,
    ) -> GuestTrackingResult<()> {
        self.inner.finalize(entry_addr.bits(), fdt_addr.bits())
    }

    // Add zero pages to the host page tables. Requires that the GPA map the SPA in
    // T::TOP_LEVEL_ALIGN-aligned contiguous chunks.
    fn add_zero_pages<I>(&mut self, to_addr: GuestPageAddr, pages: I)
    where
        I: ExactSizeIterator<Item = Page<ConvertedClean>>,
    {
        let vm = self.inner.as_finalized_vm().unwrap();
        let page_tracker = vm.page_tracker();
        // Unwrap ok since we've donate sufficient PT pages to map the entire address space up front.
        let mapper = vm
            .vm_pages()
            .map_zero_pages(to_addr, pages.len() as u64)
            .unwrap();
        for (page, vm_addr) in pages.zip(to_addr.iter_from()) {
            assert_eq!(page.size(), PageSize::Size4k);
            assert_eq!(
                vm_addr.bits() & (T::TOP_LEVEL_ALIGN - 1),
                page.addr().bits() & (T::TOP_LEVEL_ALIGN - 1)
            );
            let mappable = page_tracker
                .assign_page_for_mapping(page, vm.page_owner_id())
                .unwrap();
            mapper.map_page(vm_addr, mappable).unwrap();
        }
    }

    // Adds an emulated MMIO region to the host VM.
    fn add_mmio_region(&mut self, addr: GuestPageAddr, len: u64) {
        let vm = self.inner.as_finalized_vm().unwrap();
        vm.vm_pages().add_mmio_region(addr, len).unwrap();
    }

    // Bind `vcpu_id` to its virtual supervisor interrupt file.
    fn bind_vcpu(&self, vcpu_id: u64) {
        // vCPU ID == physical CPU ID for the host VM.
        assert_eq!(smp::PerCpu::this_cpu().cpu_id().raw() as u64, vcpu_id);
        let vm = self.inner.as_finalized_vm().unwrap();
        // We always use the first (physical) guest interrupt file as the host's virtual supervisor
        // interrupt file. The page for the interrupt file has already been mapped.
        vm.bind_vcpu_begin(vcpu_id, ImsicFileId::guest(0)).unwrap();
        vm.bind_vcpu_end(vcpu_id).unwrap();
    }

    /// Run the host VM's vCPU with ID `vcpu_id`. Does not return.
    pub fn run(&self, vcpu_id: u64) {
        self.bind_vcpu(vcpu_id);
        loop {
            // Wait until this vCPU is ready to run.
            while !self.vcpu_is_runnable(vcpu_id) {
                smp::wfi();
            }

            let vm = self.inner.as_finalized_vm().unwrap();
            let mut runner = HostVmRunner::new();
            if let ControlFlow::Break(_) = runner.run(vm, vcpu_id) {
                return;
            }
        }
    }

    // Returns if the vCPU with `vcpu_id` is runnable.
    fn vcpu_is_runnable(&self, vcpu_id: u64) -> bool {
        let vm = self.inner.as_finalized_vm().unwrap();
        vm.get_vcpu_status(vcpu_id)
            .is_ok_and(|s| s == sbi::HartState::Started as u64)
    }
}
