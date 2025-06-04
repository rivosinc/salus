// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use device_tree::DeviceTree;
use riscv_page_tables::{GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::*;
use riscv_regs::{mmio_wmb, pause};
use sync::{Mutex, Once};
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::LocalRegisterCopy;

use super::device_directory::*;
use super::error::{Error, Result};
use super::gscid::{GscId, GSCIDS};
use super::msi_page_table::MsiPageTable;
use super::queue::*;
use super::registers::*;
use crate::pci::{PciDevice, PcieRoot};

/// IOMMU device. Responsible for managing address translation for PCI devices.
pub struct Iommu {
    registers: &'static mut IommuRegisters,
    command_queue: Mutex<CommandQueue>,
    ddt: DeviceDirectory,
    phandle: Option<u32>,
}

// The global list of IOMMUs.
static IOMMUS: [Once<Iommu>; 8] = [Once::INIT; 8];

// Identifiers from the QEMU RFC implementation.
const IOMMU_PCI_ID_TABLE: [(u16, u16); 3] = [
    (0x1b36, 0x0014), // vanilla qemu IOMMU model
    (0x1efd, 0xedf1), // Rivos qemu IOMMU model
    (0x1efd, 0x0008), // Rivos hardware IOMMU
];

// Suppress clippy warning about common suffix in favor or matching mode names as per IOMMU spec.
#[allow(clippy::enum_variant_names)]
enum DirectoryMode {
    OneLevel,
    TwoLevel,
    ThreeLevel,
}

impl DirectoryMode {
    fn id(&self) -> u64 {
        use DirectoryMode::*;
        match self {
            OneLevel => 2,
            TwoLevel => 3,
            ThreeLevel => 4,
        }
    }

    fn num_levels(&self) -> usize {
        use DirectoryMode::*;
        match self {
            OneLevel => 1,
            TwoLevel => 2,
            ThreeLevel => 3,
        }
    }
}

impl Iommu {
    /// Probes for and initializes the given IOMMU device. Uses `get_page` to allocate pages for
    /// IOMMU-internal structures.
    pub fn probe(
        dt: &DeviceTree,
        pci: &PcieRoot,
        dev: &Mutex<PciDevice>,
        get_page: &mut dyn FnMut() -> Option<Page<InternalClean>>,
    ) -> Result<&'static Iommu> {
        let mut dev = dev.lock();

        let pci_ids = (dev.info().vendor_id().bits(), dev.info().device_id().bits());
        if !IOMMU_PCI_ID_TABLE.contains(&pci_ids) {
            return Err(Error::NotAnIommu);
        }

        pci.take_and_enable_hypervisor_device(&mut dev)
            .map_err(Error::ProbingIommu)?;

        // IOMMU registers are in BAR0.
        let bar = dev.bar_info().get(0).ok_or(Error::MissingRegisters)?;
        // Unwrap ok: we've already determined BAR0 is valid.
        let pci_addr = dev.get_bar_addr(0).unwrap();
        let regs_base = pci.pci_to_physical_addr(pci_addr).unwrap();
        let regs_size = bar.size();
        let dt_node_id = dev.dt_node();

        // We're done with inspecting `dev`, so unlock the mutex. It's not only good practice to
        // keep the locked section small, but we'll be iterating all devices when checking which to
        // add to this IOMMU, which would attempt to acquire the same lock again.
        drop(dev);

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

        if cfg!(feature = "hardware_ad_updates")
            && !registers.capabilities.is_set(Capabilities::AmoHwad)
        {
            return Err(Error::MissingAmoHwadSupport);
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
        let format = if registers.capabilities.is_set(Capabilities::MsiFlat) {
            DeviceContextFormat::Extended
        } else {
            DeviceContextFormat::Base
        };

        let ddt_root = get_page().ok_or(Error::OutOfPages)?;
        let mut ddtp = LocalRegisterCopy::<u64, DirectoryPointer::Register>::new(0);
        ddtp.modify(DirectoryPointer::Ppn.val(ddt_root.pfn().bits()));

        // Probe the directory mode to use.
        let mode = [
            DirectoryMode::ThreeLevel,
            DirectoryMode::TwoLevel,
            DirectoryMode::OneLevel,
        ]
        .iter()
        .find(|mode| {
            ddtp.modify(DirectoryPointer::Mode.val(mode.id()));
            registers.ddtp.set(ddtp.get());
            while registers.ddtp.is_set(DirectoryPointer::Busy) {
                pause();
            }
            registers.ddtp.read(DirectoryPointer::Mode) == mode.id()
        })
        .ok_or(Error::DeviceDirectoryUnsupported)?;

        let ddt = DeviceDirectory::new(ddt_root, format, mode.num_levels());

        let phandle = dt_node_id.and_then(|id| dt.get_node(id)).and_then(|node| {
            node.props()
                .find(|p| p.name() == "phandle")
                .and_then(|p| p.value_u32().next())
        });

        // Add devices assigned to this IOMMU to the ddt.
        if let Some(phandle) = phandle {
            for pci in PcieRoot::get_roots() {
                for dev in pci.devices() {
                    let dev = dev.lock();
                    if let Some(spec) = dev.iommu_specifier()
                        && spec.iommu_phandle() == phandle
                    {
                        ddt.add_device(spec.iommu_dev_id(), get_page).unwrap();
                    }
                }
            }
        }

        let iommu = Iommu {
            registers,
            command_queue: Mutex::new(command_queue),
            ddt,
            phandle,
        };

        // Send a DDT invalidation command to make sure the IOMMU notices any added devices.
        let commands = [Command::iodir_inval_ddt(None), Command::iofence()];
        // Unwrap ok: These are the first commands to the IOMMU, so 2 CQ entries will be available.
        iommu.submit_commands_sync(&commands).unwrap();

        // Store the iommu object in a slot in `IOMMUS`. We try slots in order until we find one
        // that initializes successfully.
        let mut iommu = Some(iommu);
        for slot in IOMMUS.iter() {
            assert!(iommu.is_some());

            // Note that `Once::call_once()` guarantees to only invoke the closure when it is the
            // first call. The closure holds a mutable reference to `iommu`, so it will only move
            // the object out of the option when the slot gets initialized successfully. We break
            // the loop once that happens, and this maintains the loop invariant `iommu.is_some()`
            // which is why the `unwrap` call in the closure is OK.
            slot.call_once(|| iommu.take().unwrap());
            if iommu.is_none() {
                // Unwrap OK: We just wrote the slot.
                return Ok(slot.get().unwrap());
            }
        }

        Err(Error::TooManyIommus)
    }

    /// Iterates all probed `Iommu`s in the system.
    pub fn get_iommus() -> impl Iterator<Item = &'static Self> {
        IOMMUS.iter().map_while(|slot| slot.get())
    }

    /// Gets the IOMMU matching the given phandle.
    pub fn get_by_phandle(phandle: u32) -> Option<&'static Iommu> {
        Iommu::get_iommus().find(|iommu| iommu.phandle == Some(phandle))
    }

    /// Gets the IOMMU for the given PciDevice.
    pub fn get_for_device(dev: &PciDevice) -> Option<&'static Iommu> {
        Self::get_by_phandle(dev.iommu_specifier()?.iommu_phandle())
    }

    /// Returns the version of this IOMMU device.
    pub fn version(&self) -> u64 {
        self.registers.capabilities.read(Capabilities::Version)
    }

    /// Returns whether this IOMMU instance supports MSI page tables.
    pub fn supports_msi_page_tables(&self) -> bool {
        self.ddt.supports_msi_page_tables()
    }

    /// Enables DMA for the given PCI device, using `pt` for 2nd-stage and `msi_pt` for MSI
    /// translation.
    pub fn attach_pci_device<T: GuestStagePagingMode>(
        &self,
        dev: &mut PciDevice,
        pt: &GuestStagePageTable<T>,
        msi_pt: Option<&MsiPageTable>,
        gscid: GscId,
    ) -> Result<()> {
        let dev_id = dev
            .iommu_specifier()
            .filter(|spec| Some(spec.iommu_phandle()) == self.phandle)
            .map(|spec| spec.iommu_dev_id())
            .ok_or(Error::IommuMismatch)?;

        // Make sure the GSCID is valid and that it matches up with the device and page table
        // owner.
        let mut gscids = GSCIDS.lock();
        let state = gscids
            .get_mut(gscid.bits() as usize)
            .and_then(|g| g.as_mut())
            .ok_or(Error::InvalidGscId(gscid))?;
        if pt.page_owner_id() != state.owner
            || !msi_pt.is_none_or(|pt| pt.owner() == state.owner)
            || dev.owner() != Some(state.owner)
        {
            return Err(Error::OwnerMismatch);
        }
        self.ddt.enable_device(dev_id, pt, msi_pt, gscid)?;
        dev.set_iommu_attached();
        state.ref_count += 1;
        Ok(())
    }

    /// Disables DMA translation for the given PCI device.
    pub fn detach_pci_device(&self, dev: &mut PciDevice, gscid: GscId) -> Result<()> {
        let dev_id = dev
            .iommu_specifier()
            .filter(|spec| Some(spec.iommu_phandle()) == self.phandle)
            .map(|spec| spec.iommu_dev_id())
            .ok_or(Error::IommuMismatch)?;

        {
            // Verify that the GSCID is valid and that it matches up with the device owner.
            let mut gscids = GSCIDS.lock();
            let state = gscids
                .get_mut(gscid.bits() as usize)
                .and_then(|g| g.as_mut())
                .ok_or(Error::InvalidGscId(gscid))?;
            if dev.owner() != Some(state.owner) {
                return Err(Error::OwnerMismatch);
            }
            dev.clear_iommu_attached();
            self.ddt.disable_device(dev_id)?;
            // Drop our reference to the GSCID used for the device.
            state.ref_count -= 1;
        }
        // Flush translation caches for the device we just disabled.
        let commands = [Command::iodir_inval_ddt(Some(dev_id)), Command::iofence()];
        // Unwrap ok: we must have room for 2 commands in the CQ since we synchronously wait on
        // commands to finish.
        self.submit_commands_sync(&commands).unwrap();
        Ok(())
    }

    /// Synchronizes the IOMMU's translation caches with updates made to the 2nd-stage and MSI
    /// page tables identified by `gscid`. If `addr` is not `None`, only flushes translations
    /// for `addr`.
    pub fn fence(&self, gscid: GscId, addr: Option<GuestPageAddr>) {
        let commands = [
            Command::iotinval_gvma(Some(gscid), addr),
            Command::iofence(),
        ];
        // Unwrap ok: we must have room for 2 commands in the CQ since we synchronously wait on
        // commands to finish.
        self.submit_commands_sync(&commands).unwrap();
    }

    // Posts the commands in `commands` to the CQ, synchronously waiting for their completion.
    fn submit_commands_sync(&self, commands: &[Command]) -> Result<()> {
        let mut cq = self.command_queue.lock();
        for &cmd in commands.iter() {
            cq.push(cmd)?;
        }
        // Make sure writes to the CQ have completed before we make them visible to HW.
        mmio_wmb();
        let tail = cq.tail() as u32;
        self.registers.cqt.set(tail);
        while self.registers.cqh.get() != tail {
            // TODO: timeout?
            pause();
        }
        // Unwrap ok since we're setting head == tail.
        cq.update_head(tail as usize).unwrap();
        Ok(())
    }
}

// `Iommu` holds `UnsafeCell`s for register access. Access to these registers is guarded by the
// `Iommu` interface which allow them to be shared and sent between threads.
unsafe impl Send for Iommu {}
unsafe impl Sync for Iommu {}
