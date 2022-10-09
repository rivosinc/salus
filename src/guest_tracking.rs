// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use page_tracking::collections::{PageArc, PageVec};
use page_tracking::PageTracker;
use riscv_page_tables::GuestStagePagingMode;
use riscv_pages::{InternalClean, Page, PageOwnerId, SequentialPages};
use spin::{Mutex, RwLock, RwLockReadGuard};

use crate::vm::{AnyVm, FinalizedVm, InitializingVm, Vm, VmRef};

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

// The possible states of a guest `Vm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuestState {
    Init,
    Running,
}

// Wrapper enum for a `Vm<T>` plus its state.
struct GuestVmInner<T: GuestStagePagingMode> {
    vm: Vm<T>,
    state: GuestState,
}

impl<T: GuestStagePagingMode> GuestVmInner<T> {
    // Creates a new initializing `GuestVmInner` from `vm`.
    fn new(vm: Vm<T>) -> Self {
        Self {
            vm,
            state: GuestState::Init,
        }
    }

    // Converts `self` from an initializing VM to a finalized VM.
    fn finalize(&mut self) -> Result<()> {
        if self.state != GuestState::Init {
            return Err(Error::GuestNotInitializing);
        }
        self.vm.finalize().map_err(Error::VmFinalizeFailed)?;
        self.state = GuestState::Running;
        Ok(())
    }
}

/// A shared reference to a `Vm` in a particular state. While this reference is held the wrapped
/// `Vm` is guaranteed not to change state.
pub struct GuestStateGuard<'a, T: GuestStagePagingMode, S> {
    inner: RwLockReadGuard<'a, GuestVmInner<T>>,
    _vm_state: PhantomData<S>,
}

impl<'a, T: GuestStagePagingMode, S> GuestStateGuard<'a, T, S> {
    fn new(inner: RwLockReadGuard<'a, GuestVmInner<T>>) -> Self {
        Self {
            inner,
            _vm_state: PhantomData,
        }
    }

    /// Returns a reference to the wrapped `Vm`.
    pub fn vm(&self) -> &Vm<T> {
        &self.inner.vm
    }
}

/// A (reference-counted) reference to a guest VM.
pub struct GuestVm<T: GuestStagePagingMode> {
    inner: PageArc<RwLock<GuestVmInner<T>>>,
}

impl<T: GuestStagePagingMode> Clone for GuestVm<T> {
    fn clone(&self) -> GuestVm<T> {
        GuestVm {
            inner: self.inner.clone(),
        }
    }
}

impl<T: GuestStagePagingMode> GuestVm<T> {
    /// Return required pages necessary to create a GuestVM.
    pub const fn required_pages() -> u64 {
        PageArc::<RwLock<GuestVmInner<T>>>::required_pages()
    }

    /// Creates a new initializing `GuestVm` from `vm`, using `page` as storage.
    pub fn new(vm: Vm<T>, page: Page<InternalClean>) -> Self {
        // assert it fits in a single page for now.
        assert!(Self::required_pages() == 1);
        let page_tracker = vm.page_tracker();
        Self {
            inner: PageArc::new_with(
                RwLock::new(GuestVmInner::new(vm)),
                page.into(),
                page_tracker,
            ),
        }
    }

    /// Returns a reference to `self` as an initializing VM.
    pub fn as_initializing_vm(&self) -> Option<InitializingVm<T>> {
        let guest = self.inner.read();
        match guest.state {
            GuestState::Init => Some(VmRef::new(GuestStateGuard::new(guest))),
            _ => None,
        }
    }

    /// Returns a reference to `self` as a finalized VM.
    pub fn as_finalized_vm(&self) -> Option<FinalizedVm<T>> {
        let guest = self.inner.read();
        match guest.state {
            GuestState::Running => Some(VmRef::new(GuestStateGuard::new(guest))),
            _ => None,
        }
    }

    /// Returns a reference to `self` as a VM in any state.
    pub fn as_any_vm(&self) -> AnyVm<T> {
        VmRef::new(GuestStateGuard::new(self.inner.read()))
    }

    /// Returns the `PageOwnerId` for the wrapped VM.
    pub fn page_owner_id(&self) -> PageOwnerId {
        self.inner.read().vm.page_owner_id()
    }

    /// Converts the guest from the initializing to the finalized state.
    pub fn finalize(&self) -> Result<()> {
        // Use try_write() here since there shouldn't be any outstanding references to a VM that
        // we're attempting to finalize. This prevents us from blocking on a potentially
        // long-running operation on a VM that isn't even in the proper state (e.g. a finalized
        // VM that's running a vCPU).
        let mut inner = self.inner.try_write().ok_or(Error::GuestInUse)?;
        inner.finalize()
    }
}

/// Tracks the guest VMs for a host VM.
pub struct Guests<T: GuestStagePagingMode> {
    guests: Mutex<PageVec<GuestVm<T>>>,
}

impl<T: GuestStagePagingMode> Guests<T> {
    /// Creates a new `Guests` using `vec_pages` as storage.
    pub fn new(vec_pages: SequentialPages<InternalClean>, page_tracker: PageTracker) -> Self {
        Self {
            guests: Mutex::new(PageVec::new(vec_pages, page_tracker)),
        }
    }

    /// Adds `guest` to this guest tracking table.
    pub fn add(&self, guest: GuestVm<T>) -> Result<()> {
        let mut guests = self.guests.lock();
        guests
            .try_reserve(1)
            .map_err(|_| Error::InsufficientGuestStorage)?;
        guests.push(guest);
        Ok(())
    }

    /// Returns the guest with the given ID.
    pub fn get(&self, id: PageOwnerId) -> Option<GuestVm<T>> {
        let guests = self.guests.lock();
        guests.iter().find(|g| g.page_owner_id() == id).cloned()
    }

    /// Removes the guest with the given ID if there are no outstanding references to it.
    pub fn remove(&self, id: PageOwnerId) -> Result<()> {
        // Pull the last reference to this guest out of the vector first so we don't do the final
        // drop under the lock.
        let _guest = {
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
        Ok(())
    }
}
