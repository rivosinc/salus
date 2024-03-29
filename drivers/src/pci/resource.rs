// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayVec;
use riscv_pages::{PageSize, SupervisorPageAddr, SupervisorPhysAddr};
use tock_registers::LocalRegisterCopy;

use super::error::*;
use super::registers::*;

/// PCI BAR resource types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciResourceType {
    /// IO port space.
    IoPort = 0,
    /// 32-bit non-prefetchable memory space.
    Mem32 = 1,
    /// 32-bit prefetchable memory space.
    PrefetchableMem32 = 2,
    /// 64-bit non-prefetchable memory space. 64-bit memory spaces are supposed to be prefetchable,
    /// but many device trees (including from QEMU) don't set the prefetch bit.
    Mem64 = 3,
    /// 64-bit prefetchable memory space.
    PrefetchableMem64 = 4,
}

/// The number of different PCI resource types.
pub const MAX_RESOURCE_TYPES: usize = PciResourceType::PrefetchableMem64 as usize + 1;

// Format of the first PCI address cell which specifies the type of the resource.
const PCI_ADDR_PREFETCH_BIT: u32 = 1 << 30;
const PCI_ADDR_SPACE_CODE_SHIFT: u32 = 24;
const PCI_ADDR_SPACE_CODE_MASK: u32 = 0x3;

impl PciResourceType {
    /// Return the resource type corresponding to the raw `index`.
    pub fn from_index(index: usize) -> Option<Self> {
        use PciResourceType::*;
        match index {
            0 => Some(IoPort),
            1 => Some(Mem32),
            2 => Some(PrefetchableMem32),
            3 => Some(Mem64),
            4 => Some(PrefetchableMem64),
            _ => None,
        }
    }

    /// Reads a PCI BAR resource type from the first cell in a PCI address range.
    pub fn from_dt_cell(cell: u32) -> Option<Self> {
        let prefetchable = (cell & PCI_ADDR_PREFETCH_BIT) != 0;
        use PciResourceType::*;
        match (cell >> PCI_ADDR_SPACE_CODE_SHIFT) & PCI_ADDR_SPACE_CODE_MASK {
            0x0 => {
                // Config space. Ignore it since we already got it from 'reg'.
                None
            }
            0x1 => Some(IoPort),
            0x2 => {
                if prefetchable {
                    Some(PrefetchableMem32)
                } else {
                    Some(Mem32)
                }
            }
            0x3 => {
                if prefetchable {
                    Some(PrefetchableMem64)
                } else {
                    Some(Mem64)
                }
            }
            _ => unreachable!(),
        }
    }

    /// Returns the PCI address cell used to encode this resource type.
    pub fn to_dt_cell(self) -> u32 {
        use PciResourceType::*;
        match self {
            IoPort => 0x1 << PCI_ADDR_SPACE_CODE_SHIFT,
            Mem32 => 0x2 << PCI_ADDR_SPACE_CODE_SHIFT,
            PrefetchableMem32 => (0x2 << PCI_ADDR_SPACE_CODE_SHIFT) | PCI_ADDR_PREFETCH_BIT,
            Mem64 => 0x3 << PCI_ADDR_SPACE_CODE_SHIFT,
            PrefetchableMem64 => (0x3 << PCI_ADDR_SPACE_CODE_SHIFT) | PCI_ADDR_PREFETCH_BIT,
        }
    }

    /// Decodes the resource type from a BAR register.
    pub fn from_bar_register(reg: LocalRegisterCopy<u32, BaseAddress::Register>) -> Self {
        let is_mem = matches!(
            reg.read_as_enum(BaseAddress::SpaceType),
            Some(BaseAddress::SpaceType::Value::Memory)
        );
        let is_64bit = is_mem
            && matches!(
                reg.read_as_enum(BaseAddress::MemoryType),
                Some(BaseAddress::MemoryType::Value::Bits64)
            );
        let prefetch = is_mem && reg.is_set(BaseAddress::Prefetchable);
        if is_mem {
            match (is_64bit, prefetch) {
                (true, true) => PciResourceType::PrefetchableMem64,
                (true, false) => PciResourceType::Mem64,
                (false, true) => PciResourceType::PrefetchableMem32,
                (false, false) => PciResourceType::Mem32,
            }
        } else {
            PciResourceType::IoPort
        }
    }

    /// Returns if this resource uses 64-bit addresses.
    pub fn is_64bit(&self) -> bool {
        matches!(
            self,
            PciResourceType::Mem64 | PciResourceType::PrefetchableMem64
        )
    }
}

/// Describes a single PCI root resource.
#[derive(Debug)]
pub struct PciRootResource {
    addr: SupervisorPageAddr,
    // Size of the resource being passed through to the host.
    host_size: u64,
    // Total size of the resource including space allocated by the hypervisor.
    total_size: u64,
    // Whether or not the resource has been claimed by the host.
    taken: bool,
    pci_addr: u64,
}

impl PciRootResource {
    /// Creates a new root resource with the given address and size.
    pub fn new(addr: SupervisorPageAddr, size: u64, pci_addr: u64) -> Self {
        Self {
            addr,
            host_size: size,
            total_size: size,
            taken: false,
            pci_addr,
        }
    }

    /// Returns the CPU physical address of the resource.
    pub fn addr(&self) -> SupervisorPageAddr {
        self.addr
    }

    /// Returns the total size of the resource.
    pub fn size(&self) -> u64 {
        self.total_size
    }

    /// Returns the size of the resource exposed to the host.
    pub fn host_size(&self) -> u64 {
        self.host_size
    }

    /// Returns the PCI bus address of the resource.
    pub fn pci_addr(&self) -> u64 {
        self.pci_addr
    }

    /// Allocates part of the resource for hypervisor use. Hypervisor resources are allocated from
    /// the end of the resource block so that the alignment of the host-exposed portion of the
    /// resource remains unaffected.
    pub fn alloc_for_hypervisor(&mut self, size: u64) -> Result<SupervisorPageAddr> {
        // Make sure we always allocate at least a page so that the host/hypervisor boundary is
        // not in the middle of page.
        let size = core::cmp::max(size, PageSize::Size4k as u64).next_power_of_two();
        // The start address must be `size`-aligned.
        self.host_size = self
            .host_size
            .checked_sub(size)
            .ok_or(Error::OutOfResources)?
            & !(size - 1);
        // Unwrap ok since `host_size` is less than what it was before and it must've been valid
        // to begin with.
        Ok(self
            .addr
            .checked_add_pages(PageSize::num_4k_pages(self.host_size))
            .unwrap())
    }

    /// Marks the host-exposed portion of the resource as exclusively allocated.
    pub fn take_for_host(&mut self) -> Result<()> {
        if self.taken {
            return Err(Error::ResourceTaken);
        }
        self.taken = true;
        Ok(())
    }
}

// We only allow at most one resource of each type at the root complex.
const MAX_ROOT_RESOURCES: usize = MAX_RESOURCE_TYPES;

/// The PCI resources for a root complex.
pub struct PciRootResources {
    resources: ArrayVec<Option<PciRootResource>, MAX_ROOT_RESOURCES>,
}

impl PciRootResources {
    /// Creates an initially-empty `PciRootResources`.
    pub fn new() -> Self {
        let mut resources = ArrayVec::new();
        for _ in 0..resources.capacity() {
            resources.push(None);
        }
        Self { resources }
    }

    /// Inserts the specified resource if a resource of the same type does not exist already.
    pub fn insert(
        &mut self,
        resource_type: PciResourceType,
        resource: PciRootResource,
    ) -> Result<()> {
        if self.resources[resource_type as usize].is_some() {
            return Err(Error::DuplicateResource(resource_type));
        }
        self.resources[resource_type as usize] = Some(resource);
        Ok(())
    }

    /// Returns a reference to the resource with the given type.
    pub fn get(&self, resource_type: PciResourceType) -> Option<&PciRootResource> {
        self.resources[resource_type as usize].as_ref()
    }

    /// Returns a mutable reference to the resource with the given type.
    pub fn get_mut(&mut self, resource_type: PciResourceType) -> Option<&mut PciRootResource> {
        self.resources[resource_type as usize].as_mut()
    }

    /// Returns a present resource compatible with `resource_type`.
    pub fn find_matching(&self, resource_type: PciResourceType) -> Option<PciResourceType> {
        use PciResourceType::*;
        match resource_type {
            IoPort | Mem32 | Mem64 => self.get(resource_type).map(|_| resource_type),
            // Fall back to the non-prefetchable equivalent if the prefetchable resource doesn't
            // exist.
            PrefetchableMem32 => self
                .get(resource_type)
                .map(|_| resource_type)
                .or_else(|| self.find_matching(Mem32)),
            PrefetchableMem64 => self
                .get(resource_type)
                .map(|_| resource_type)
                .or_else(|| self.find_matching(Mem64)),
        }
    }

    /// Translates the given PCI bus address to a CPU physical address.
    pub fn pci_to_physical_addr(&self, pci_addr: u64) -> Option<SupervisorPhysAddr> {
        let res = self
            .resources
            .iter()
            .find(|res| {
                res.as_ref()
                    .filter(|r| r.pci_addr() <= pci_addr && pci_addr <= r.pci_addr() + r.size())
                    .is_some()
            })
            .and_then(|res| res.as_ref())?;
        SupervisorPhysAddr::from(res.addr())
            .checked_increment(pci_addr.checked_sub(res.pci_addr())?)
    }

    /// Translates the given CPU physical address to a PCI bus address.
    pub fn physical_to_pci_addr(&self, addr: SupervisorPhysAddr) -> Option<u64> {
        let res = self
            .resources
            .iter()
            .find(|res| {
                res.as_ref()
                    .filter(|r| {
                        r.addr().bits() <= addr.bits() && addr.bits() <= r.addr().bits() + r.size()
                    })
                    .is_some()
            })
            .and_then(|res| res.as_ref())?;
        res.pci_addr()
            .checked_add(addr.bits().checked_sub(res.addr().bits())?)
    }
}

impl Default for PciRootResources {
    fn default() -> Self {
        Self::new()
    }
}
