// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// `PageOwnerId` represents the entity that owns a given page.
/// The hypervisor (Salus = HS mode) is special cased as is the primary host running in VS mode.
/// 0 = host
/// 1 = hypervisor
/// 2..u64::max = guest id
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PageOwnerId {
    id: u64,
}

impl PageOwnerId {
    const HOST: u64 = 0;
    const HYPERVISOR: u64 = 1;

    /// Creates a new PageOwnerId with the give raw value
    pub fn new(id: u64) -> Option<Self> {
        if id == Self::HOST || id == Self::HYPERVISOR {
            None
        } else {
            Some(Self { id })
        }
    }

    /// Returns the ID of the host (the primary VM running in VS mode).
    pub fn host() -> Self {
        Self { id: Self::HOST }
    }

    /// Returns the ID of the hypervisor (running in HS mode).
    pub fn hypervisor() -> Self {
        Self {
            id: Self::HYPERVISOR,
        }
    }

    /// Returns true if this is owner by the host (the primary VM running in VS mode).
    pub fn is_host(&self) -> bool {
        self.id == Self::HOST
    }

    /// Returns the raw value of the PageOwnerId.
    pub fn raw(&self) -> u64 {
        self.id
    }
}

/// `AddressSpace` identifies the address space that a raw address is in.
pub trait AddressSpace: Clone + Copy {
    /// Returns the `PageOwnerId` for the address space.
    fn id(&self) -> PageOwnerId;
}

/// Represents the supervisor (i.e. "actual") physical address space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SupervisorPhys;

impl AddressSpace for SupervisorPhys {
    fn id(&self) -> PageOwnerId {
        PageOwnerId::hypervisor()
    }
}

/// Represents a guest physical address space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GuestPhys(PageOwnerId);

impl GuestPhys {
    /// Creates a new tag representing a guest physical address space with the given ID.
    pub fn new(id: PageOwnerId) -> Self {
        // Hypervisor should never own pages that are mapped into a GPA.
        assert!(id != PageOwnerId::hypervisor());
        Self(id)
    }
}

impl AddressSpace for GuestPhys {
    fn id(&self) -> PageOwnerId {
        self.0
    }
}
