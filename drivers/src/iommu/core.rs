// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;
use riscv_regs::{mmio_wmb, pause};
use spin::{Mutex, Once};
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::LocalRegisterCopy;

use super::device_directory::*;
use super::error::{Error, Result};
use super::queue::*;
use super::registers::*;
use crate::pci::{self, PciArenaId, PcieRoot};

/// IOMMU device. Responsible for managing address translation for PCI devices.
pub struct Iommu {
    _arena_id: PciArenaId,
    registers: &'static mut IommuRegisters,
    _command_queue: Mutex<CommandQueue>,
    _ddt: DeviceDirectory<Ddt3Level>,
}

// The global IOMMU singleton.
static IOMMU: Once<Iommu> = Once::new();

// Identifiers from the QEMU RFC implementation.
const IOMMU_VENDOR_ID: u16 = 0x1efd;
const IOMMU_DEVICE_ID: u16 = 0x8001;

impl Iommu {
    /// Probes for and initializes the IOMMU device on the given PCI root. Uses `get_page` to
    /// allocate pages for IOMMU-internal structures.
    pub fn probe_from(
        pci: &PcieRoot,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<()> {
        let arena_id = pci
            .take_and_enable_hypervisor_device(
                pci::VendorId::new(IOMMU_VENDOR_ID),
                pci::DeviceId::new(IOMMU_DEVICE_ID),
            )
            .map_err(Error::ProbingIommu)?;
        let (iommu_addr, regs_base, regs_size) = {
            let dev = pci.get_device(arena_id).unwrap().lock();
            // IOMMU registers are in BAR0.
            let bar = dev.bar_info().get(0).ok_or(Error::MissingRegisters)?;
            // Unwrap ok: we've already determined BAR0 is valid.
            let pci_addr = dev.get_bar_addr(0).unwrap();
            let regs_base = pci.pci_to_physical_addr(pci_addr).unwrap();
            let regs_size = bar.size();
            (dev.info().address(), regs_base, regs_size)
        };
        if regs_size < core::mem::size_of::<IommuRegisters>() as u64 {
            return Err(Error::InvalidRegisterSize(regs_size));
        }
        if regs_base.bits() % core::mem::size_of::<IommuRegisters>() as u64 != 0 {
            return Err(Error::MisalignedRegisters);
        }
        // Safety: We've taken unique ownership of the IOMMU PCI device and have verified that
        // BAR0 points to a suitably sized and aligned register set.
        let registers = unsafe { (regs_base.bits() as *mut IommuRegisters).as_mut().unwrap() };

        // We need support for Sv48x4 G-stage translation and MSI page-tables at minimum.
        if !registers.capabilities.is_set(Capabilities::Sv48x4) {
            return Err(Error::MissingGStageSupport);
        }
        if !registers.capabilities.is_set(Capabilities::MsiFlat) {
            return Err(Error::MissingMsiSupport);
        }

        // Initialize the command queue.
        let command_queue = CommandQueue::new(get_page().ok_or(Error::OutOfPages)?);
        let mut cqb = LocalRegisterCopy::<u64, QueueBase::Register>::new(0);
        cqb.modify(QueueBase::Log2SzMinus1.val(command_queue.capacity().ilog2() as u64 - 1));
        cqb.modify(QueueBase::Ppn.val(command_queue.base_address().pfn().bits()));
        registers.cqb.set(cqb.get());
        registers.cqcsr.write(CqControl::Enable.val(1));
        while !registers.cqcsr.is_set(CqControl::On) {
            pause();
        }

        // TODO: Set up fault queue.

        // Set up an initial device directory table.
        let ddt = DeviceDirectory::new(get_page().ok_or(Error::OutOfPages)?);
        for dev in pci.devices() {
            let addr = dev.lock().info().address();
            if addr == iommu_addr {
                // Skip the IOMMU itself.
                continue;
            }
            ddt.add_device(addr.try_into()?, get_page)?;
        }
        let mut ddtp = LocalRegisterCopy::<u64, DirectoryPointer::Register>::new(0);
        ddtp.modify(DirectoryPointer::Ppn.val(ddt.base_address().pfn().bits()));
        ddtp.modify(DirectoryPointer::Mode.val(Ddt3Level::IOMMU_MODE));
        // Ensure writes to the DDT have completed before we point the IOMMU at it.
        mmio_wmb();
        registers.ddtp.set(ddtp.get());

        let iommu = Iommu {
            _arena_id: arena_id,
            registers,
            _command_queue: Mutex::new(command_queue),
            _ddt: ddt,
        };
        IOMMU.call_once(|| iommu);
        Ok(())
    }

    /// Gets a reference to the `Iommu` singleton.
    pub fn get() -> Option<&'static Self> {
        IOMMU.get()
    }

    /// Returns the version of this IOMMU device.
    pub fn version(&self) -> u64 {
        self.registers.capabilities.read(Capabilities::Version)
    }
}

// `Iommu` holds `UnsafeCell`s for register access. Access to these registers is guarded by the
// `Iommu` interface which allow them to be shared and sent between threads.
unsafe impl Send for Iommu {}
unsafe impl Sync for Iommu {}
