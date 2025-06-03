// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use riscv_page_tables::{GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::*;
use riscv_regs::{mmio_wmb, pause};
use sync::{Mutex, Once};
use tock_registers::interfaces::{Readable, Writeable};
use tock_registers::LocalRegisterCopy;

use super::device_directory::*;
use super::error::{Error, Result};
use super::msi_page_table::MsiPageTable;
use super::queue::*;
use super::registers::*;
use crate::pci::{self, PciArenaId, PciDevice, PcieRoot};

// Tracks the state of an allocated global soft-context ID (GSCID).
#[derive(Clone, Copy, Debug)]
struct GscIdState {
    owner: PageOwnerId,
    ref_count: usize,
}

// We use a fixed-sized array to track available GSCIDs. We can't use a versioning scheme like we
// would for CPU VMIDs since reassigning GSCIDs on overflow would require us to temporarily disable
// DMA from all devices, which is extremely disruptive. Set a max of 64 allocated GSCIDs for now
// since it's unlikely we'll have more than that number of active VMs with assigned devices for
// the time being.
const MAX_GSCIDS: usize = 64;

/// IOMMU device. Responsible for managing address translation for PCI devices.
pub struct Iommu {
    _arena_id: PciArenaId,
    registers: &'static mut IommuRegisters,
    command_queue: Mutex<CommandQueue>,
    ddt: DeviceDirectory,
    gscids: Mutex<[Option<GscIdState>; MAX_GSCIDS]>,
}

// The global IOMMU singleton.
static IOMMU: Once<Iommu> = Once::new();

// Identifiers from the QEMU RFC implementation.
const IOMMU_VENDOR_ID: u16 = 0x1efd;
const IOMMU_DEVICE_ID: u16 = 0xedf1;

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

        for pci in PcieRoot::get_roots() {
            for dev in pci.devices() {
                let addr = dev.lock().info().address();
                if addr == iommu_addr {
                    // Skip the IOMMU itself.
                    continue;
                }
                ddt.add_device(addr.try_into()?, get_page)?;
            }
        }

        let iommu = Iommu {
            _arena_id: arena_id,
            registers,
            command_queue: Mutex::new(command_queue),
            ddt,
            gscids: Mutex::new([None; MAX_GSCIDS]),
        };

        // Send a DDT invalidation command to make sure the IOMMU notices the added devices.
        let commands = [Command::iodir_inval_ddt(None), Command::iofence()];
        // Unwrap ok: These are the first commands to the IOMMU, so 2 CQ entries will be
        // available.
        iommu.submit_commands_sync(&commands).unwrap();

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

    /// Returns whether this IOMMU instance supports MSI page tables.
    pub fn supports_msi_page_tables(&self) -> bool {
        self.ddt.supports_msi_page_tables()
    }

    /// Allocates a new GSCID for `owner`.
    pub fn alloc_gscid(&self, owner: PageOwnerId) -> Result<GscId> {
        let mut gscids = self.gscids.lock();
        let next = gscids
            .iter()
            .position(|g| g.is_none())
            .ok_or(Error::OutOfGscIds)?;
        let state = GscIdState {
            owner,
            ref_count: 0,
        };
        gscids[next] = Some(state);
        Ok(GscId::new(next as u16))
    }

    /// Releases `gscid`, which must not be in use in any active device contexts.
    pub fn free_gscid(&self, gscid: GscId) -> Result<()> {
        let mut gscids = self.gscids.lock();
        let state = gscids
            .get_mut(gscid.bits() as usize)
            .ok_or(Error::InvalidGscId(gscid))?;
        match state {
            Some(s) if s.ref_count > 0 => {
                return Err(Error::GscIdInUse(gscid));
            }
            None => {
                return Err(Error::GscIdAlreadyFree(gscid));
            }
            _ => {
                *state = None;
            }
        }
        Ok(())
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
        let dev_id = DeviceId::try_from(dev.info().address())?;
        // Make sure the GSCID is valid and that it matches up with the device and page table
        // owner.
        let mut gscids = self.gscids.lock();
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
        let dev_id = DeviceId::try_from(dev.info().address())?;
        {
            // Verify that the GSCID is valid and that it matches up with the device owner.
            let mut gscids = self.gscids.lock();
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
