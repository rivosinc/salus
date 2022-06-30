// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;

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

/// Identifies if a raw address is virtual (subject to at least one stage of translation) or physical
/// (the output of first-stage translation).
pub trait AddressType: Clone + Copy + PartialEq + Eq {}

/// Physical addresses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Physical {}
impl AddressType for Physical {}

/// Virtual addresses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Virtual {}
impl AddressType for Virtual {}

/// `AddressSpace` identifies the address space that a raw address is in.
pub trait AddressSpace: Clone + Copy + PartialEq + Eq {
    /// Returns the `PageOwnerId` for the address space.
    fn id(&self) -> PageOwnerId;
}

/// Represents a supervisor address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Supervisor<T: AddressType>(PhantomData<T>);

impl<T: AddressType> Default for Supervisor<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: AddressType> AddressSpace for Supervisor<T> {
    fn id(&self) -> PageOwnerId {
        PageOwnerId::hypervisor()
    }
}

/// Represents the supervisor (i.e. "actual") physical address space.
pub type SupervisorPhys = Supervisor<Physical>;

/// Represents the supervisor's virtual address space.
pub type SupervisorVirt = Supervisor<Virtual>;

/// Represents a guest address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Guest<T: AddressType>(PageOwnerId, PhantomData<T>);

impl<T: AddressType> Guest<T> {
    /// Creates a new tag representing a guest address space with the given ID.
    pub fn new(id: PageOwnerId) -> Self {
        // Hypervisor should never own pages that are mapped into a GPA.
        assert!(id != PageOwnerId::hypervisor());
        Self(id, PhantomData)
    }
}

impl<T: AddressType> AddressSpace for Guest<T> {
    fn id(&self) -> PageOwnerId {
        self.0
    }
}

/// Represents a guest physical address space.
pub type GuestPhys = Guest<Physical>;

/// Represents a guest virtual address space.
pub type GuestVirt = Guest<Virtual>;
