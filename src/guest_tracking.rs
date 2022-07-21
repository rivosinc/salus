// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use core::ops::Deref;
use page_tracking::collections::{PageArc, PageVec};
use page_tracking::PageTracker;
use riscv_page_tables::GuestStagePageTable;
use riscv_pages::{InternalClean, Page, PageOwnerId, SequentialPages};
use spin::{Mutex, RwLock, RwLockReadGuard};

use crate::vm::{Vm, VmStateFinalized, VmStateInitializing};

/// Guest tracking-related errors.
#[derive(Debug)]
pub enum Error {
    InsufficientGuestStorage,
    InvalidGuestId,
    GuestInUse,
    GuestNotInitializing,
    VmFinalizeFailed(crate::vm::Error),
}

pub type Result<T> = core::result::Result<T, Error>;

/// Wrapper enum for a `Vm<T, S>` for each possible state S so that we can store `Vm`s of any
/// state in a `PageArc`.
enum GuestStateInner<T: GuestStagePageTable> {
    Init(Vm<T, VmStateInitializing>),
    Running(Vm<T, VmStateFinalized>),
    Temp,
}

impl<T: GuestStagePageTable> GuestStateInner<T> {
    /// Returns a reference to `self` as an initializing VM.
    fn as_initializing_vm(&self) -> Option<&Vm<T, VmStateInitializing>> {
        match self {
            Self::Init(ref v) => Some(v),
            Self::Running(_) => None,
            Self::Temp => unreachable!(),
        }
    }

    /// Returns a reference to `self` as a finalized VM.
    fn as_finalized_vm(&self) -> Option<&Vm<T, VmStateFinalized>> {
        match self {
            Self::Init(_) => None,
            Self::Running(ref v) => Some(v),
            Self::Temp => unreachable!(),
        }
    }

    /// Returns the `PageOwnerId` for the wrapped VM.
    fn page_owner_id(&self) -> PageOwnerId {
        match self {
            Self::Init(vm) => vm.page_owner_id(),
            Self::Running(vm) => vm.page_owner_id(),
            Self::Temp => unreachable!(),
        }
    }

    /// Converts `self` from an initializing VM to a finalized VM.
    fn finalize(&mut self) -> Result<()> {
        if !matches!(self, Self::Init(_)) {
            return Err(Error::GuestNotInitializing);
        }
        let mut temp = Self::Temp;
        core::mem::swap(self, &mut temp);
        let mut running = match temp {
            Self::Init(v) => Self::Running(v.finalize().map_err(Error::VmFinalizeFailed)?),
            _ => unreachable!(),
        };
        core::mem::swap(self, &mut running);
        Ok(())
    }

    /// Destroys `self`.
    fn destroy(&mut self) {
        // TODO: Use Drop instead, which is currently not possible due to a move from self in finalize.
        match self {
            Self::Init(ref mut v) => v.destroy(),
            Self::Running(ref mut v) => v.destroy(),
            Self::Temp => unreachable!(),
        };
    }
}

/// A refernce to a VM in a paritcular state. While this refernce is held the wrapped `Vm` is
/// guaranteed not to change state.
pub struct VmRef<'a, T: GuestStagePageTable, S> {
    vm: RwLockReadGuard<'a, GuestStateInner<T>>,
    phantom: PhantomData<S>,
}

impl<'a, T: GuestStagePageTable> Deref for VmRef<'a, T, VmStateInitializing> {
    type Target = Vm<T, VmStateInitializing>;

    fn deref(&self) -> &Self::Target {
        self.vm.as_initializing_vm().unwrap()
    }
}

impl<'a, T: GuestStagePageTable> Deref for VmRef<'a, T, VmStateFinalized> {
    type Target = Vm<T, VmStateFinalized>;

    fn deref(&self) -> &Self::Target {
        self.vm.as_finalized_vm().unwrap()
    }
}

/// A (reference-counted) reference to a guest VM.
pub struct GuestState<T: GuestStagePageTable> {
    inner: PageArc<RwLock<GuestStateInner<T>>>,
}

impl<T: GuestStagePageTable> Clone for GuestState<T> {
    fn clone(&self) -> GuestState<T> {
        GuestState {
            inner: self.inner.clone(),
        }
    }
}

impl<T: GuestStagePageTable> GuestState<T> {
    /// Creates a new initializing `GuestState` from `vm`, using `page` as storage.
    pub fn new(vm: Vm<T, VmStateInitializing>, page: Page<InternalClean>) -> Self {
        let page_tracker = vm.page_tracker();
        Self {
            inner: PageArc::new_with(RwLock::new(GuestStateInner::Init(vm)), page, page_tracker),
        }
    }

    /// Returns a reference to `self` as an initializing VM.
    pub fn as_initializing_vm(&self) -> Option<VmRef<T, VmStateInitializing>> {
        let guest = self.inner.read();
        guest.as_initializing_vm()?;
        Some(VmRef {
            vm: guest,
            phantom: PhantomData,
        })
    }

    /// Returns a reference to `self` as a finalized VM.
    pub fn as_finalized_vm(&self) -> Option<VmRef<T, VmStateFinalized>> {
        let guest = self.inner.read();
        guest.as_finalized_vm()?;
        Some(VmRef {
            vm: guest,
            phantom: PhantomData,
        })
    }

    /// Returns the `PageOwnerId` for the wrapped VM.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.inner.read().page_owner_id()
    }
}

/// Tracks the guest VMs for a host VM.
pub struct Guests<T: GuestStagePageTable> {
    guests: Mutex<PageVec<GuestState<T>>>,
}

impl<T: GuestStagePageTable> Guests<T> {
    /// Creates a new `Guests` using `vec_pages` as storage.
    pub fn new(vec_pages: SequentialPages<InternalClean>, page_tracker: PageTracker) -> Self {
        Self {
            guests: Mutex::new(PageVec::new(vec_pages, page_tracker)),
        }
    }

    /// Adds `guest` to this guest tracking table.
    pub fn add(&self, guest: GuestState<T>) -> Result<()> {
        let mut guests = self.guests.lock();
        guests
            .try_reserve(1)
            .map_err(|_| Error::InsufficientGuestStorage)?;
        guests.push(guest);
        Ok(())
    }

    /// Returns the guest with the given ID.
    pub fn get(&self, id: PageOwnerId) -> Option<GuestState<T>> {
        let guests = self.guests.lock();
        guests.iter().find(|g| g.page_owner_id() == id).cloned()
    }

    /// Removes the guest with the given ID if there are no outstanding references to it.
    pub fn remove(&self, id: PageOwnerId) -> Result<()> {
        // Pull the last reference to this guest out of the vector first so we don't do the final
        // drop under the lock.
        let guest = {
            let mut guests = self.guests.lock();
            let (index, guest) = guests
                .iter()
                .enumerate()
                .find(|(_, g)| g.page_owner_id() == id)
                .ok_or(Error::InvalidGuestId)?;
            // This use of ref_count() is sound since we hold the lock on self.guests and no new
            // references can be created if we hold the only reference.
            if PageArc::ref_count(&guest.inner) != 1 {
                return Err(Error::GuestInUse);
            }
            let last = guest.clone();
            guests.remove(index);
            last
        };
        guest.inner.write().destroy();
        Ok(())
    }

    /// Finalizes the guest with the given ID, converting the guest from the initializing to the
    /// finalized state. The finalized guest is returned upon success.
    pub fn finalize(&self, id: PageOwnerId) -> Result<GuestState<T>> {
        let guests = self.guests.lock();
        let guest = guests
            .iter()
            .find(|g| g.page_owner_id() == id)
            .ok_or(Error::InvalidGuestId)?;
        {
            // Use try_write() here since there shouldn't be any outstanding references to a VM that
            // we're attempting to finalize. This prevents us from blocking a potentially
            // long-running operation on a VM that isn't even in the proper state (e.g. a finalized
            // VM that's running a vCPU).
            let mut state = guest.inner.try_write().ok_or(Error::GuestInUse)?;
            state.finalize()?;
        }
        Ok(guest.clone())
    }
}
