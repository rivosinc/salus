// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::marker::PhantomData;
use core::mem::size_of;
use data_model::{DataInit, VolatileMemory, VolatileSlice};
use riscv_pages::*;

use super::device_directory::{DeviceId, GscId};
use super::error::*;

/// Type marker for a queue where software is the producer.
pub enum Producer {}
/// Type marker for a queue where software is the consumer.
pub enum Consumer {}

/// An IOMMU queue, used to communicate between software and the IOMMU hardware. The queue is
/// a page-aligned ring buffer of uniformly-sized structs (though different types of queues can
/// have different struct sizes/layouts). For simplicity, we use a single 4kB page for queue
/// storage.
pub struct Queue<T: DataInit, Q> {
    mem: VolatileSlice<'static>,
    head: usize,
    tail: usize,
    capacity: usize,
    _elem_type: PhantomData<T>,
    _queue_type: PhantomData<Q>,
}

impl<T: DataInit, Q> Queue<T, Q> {
    /// Creates a new `Queue` with elements of type `T`, using `page` as the backing store.
    pub fn new(page: Page<InternalClean>) -> Self {
        // Queue structs are always powers of 2.
        //
        // TODO: Use `const_assert!` once `generic_const_exprs` stabilizes.
        assert!(size_of::<T>().is_power_of_two());
        assert!(size_of::<T>() < PageSize::Size4k as usize);
        let capacity = (page.size() as usize) / size_of::<T>();
        // Safety: We uniquely own the memory in `page` and are able to read and write it as bytes.
        let mem = unsafe {
            core::slice::from_raw_parts_mut(page.addr().bits() as *mut u8, page.size() as usize)
        };
        Self {
            mem: VolatileSlice::new(mem),
            head: 0,
            tail: 0,
            capacity,
            _elem_type: PhantomData,
            _queue_type: PhantomData,
        }
    }

    /// Returns the base physical address of this queue.
    pub fn base_address(&self) -> SupervisorPageAddr {
        // Unwrap ok since the queue must've been created from a page-aligned slice of memory.
        PageAddr::new(RawAddr::supervisor(self.mem.as_ptr() as u64)).unwrap()
    }

    /// Returns the total number of elements that can be held in the queue.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    /// Returns if the queue is full.
    pub fn is_full(&self) -> bool {
        (self.tail + 1) & (self.capacity - 1) == self.head
    }

    /// Returns the head index of the queue.
    #[allow(dead_code)]
    pub fn head(&self) -> usize {
        self.head
    }

    /// Returns the tail index of the queue.
    pub fn tail(&self) -> usize {
        self.tail
    }
}

impl<T: DataInit> Queue<T, Producer> {
    /// Updates the head pointer of the queue to `haed`. Expected to be used to update the queue's
    /// software head pointer with a head pointer read from an IOMMU register.
    pub fn update_head(&mut self, head: usize) -> Result<()> {
        // Make sure `head` isn't being advanced past `self.tail`. Silence clippy as its suggestion
        // is less readable and is just as many comparisons.
        #[allow(clippy::nonminimal_bool)]
        if head >= self.capacity
            || (self.head <= self.tail && (head > self.tail || self.head > head))
            || (self.head > self.tail && self.tail < head && head < self.head)
        {
            return Err(Error::InvalidQueuePointer(head));
        }
        self.head = head;
        Ok(())
    }

    /// Pushes an element to the tail of the queue.
    pub fn push(&mut self, elem: T) -> Result<()> {
        if self.is_full() {
            return Err(Error::QueueFull);
        }
        // Unwrap ok since `self.tail` must be in bounds.
        let tail_ref = self.mem.get_ref(self.tail * size_of::<T>()).unwrap();
        tail_ref.store(elem);
        self.tail = (self.tail + 1) & (self.capacity - 1);
        Ok(())
    }
}

// TODO: Remove once we have support for the fault queue in place.
#[allow(dead_code)]
impl<T: DataInit> Queue<T, Consumer> {
    /// Updates the tail pointer of the queue to `tail`. Expected to be used to update the queue's
    /// software tail pointer with a tail pointer read from an IOMMU register.
    pub fn update_tail(&mut self, tail: usize) -> Result<()> {
        // Make sure `tail` isn't being advanced to or past `self.head`.
        if tail >= self.capacity
            || (self.head < self.tail && self.head <= tail && tail < self.tail)
            || (self.head > self.tail && (tail >= self.head || self.tail > tail))
        {
            return Err(Error::InvalidQueuePointer(tail));
        }
        self.tail = tail;
        Ok(())
    }

    /// Pops an element from the head of the queue.
    pub fn pop(&mut self) -> Result<T> {
        if self.is_empty() {
            return Err(Error::QueueEmpty);
        }
        // Unwrap ok since `self.head` must be in bounds.
        let head_ref = self.mem.get_ref(self.head * size_of::<T>()).unwrap();
        self.head += 1;
        Ok(head_ref.load())
    }
}

/// An entry in the IOMMU command queuee. Used to perform translation cache invalidations.
///
/// Note that completion of a particular invalidation command does not guarantee that the
/// invalidation itself has completed; an `IOFENCE.C` command is needed to flush all in-flight
/// commands.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Command {
    op: u64,
    addr: u64,
}

const FUNC3_SHIFT: u64 = 7;

impl Command {
    /// Creates a new `IOTINVAL.GVMA` comamnd for flushing 2nd-stage translation caches.
    ///
    /// If `gscid` is not `None`, only translations matching the specified GSCID are flushed.
    ///
    /// If `gpa` is not `None`, only translations matching the specified guest physical address
    /// are flushed.
    pub fn iotinval_gvma(gscid: Option<GscId>, gpa: Option<GuestPageAddr>) -> Self {
        const IOTINVAL_OP: u64 = 0x1;
        const GVMA_FUNC: u64 = 0x1;
        let mut op = IOTINVAL_OP | (GVMA_FUNC << FUNC3_SHIFT);

        const GV: u64 = 1 << 12;
        const GSCID_SHIFT: u64 = 40;
        if let Some(g) = gscid {
            op |= GV | ((g.bits() as u64) << GSCID_SHIFT);
        }

        const AV: u64 = 1 << 11;
        let addr = if let Some(g) = gpa {
            op |= AV;
            g.bits()
        } else {
            0
        };

        Self { op, addr }
    }

    /// Creates a new `IODIR.INVAL_DDT` command for flushing device directory table caches.
    ///
    /// If `dev` is not `None`, only translations for the specified device ID are flushed.
    pub fn iodir_inval_ddt(dev: Option<DeviceId>) -> Self {
        const IODIR_OP: u64 = 0x3;
        const INVAL_DDT_FUNC: u64 = 0x0;
        let mut op = IODIR_OP | (INVAL_DDT_FUNC << FUNC3_SHIFT);

        const DV: u64 = 1 << 10;
        const DID_SHIFT: u64 = 40;
        if let Some(d) = dev {
            op |= DV | ((d.bits() as u64) << DID_SHIFT);
        }

        Self { op, addr: 0 }
    }

    /// Creates a new `IOFENCE.C` command for synchronizing the command queue. Upon completion of
    /// this command, all prior commands submitted to the command queue are guaranteed to have
    /// completed.
    pub fn iofence() -> Self {
        const IOFENCE_OP: u64 = 0x2;
        const PR: u64 = 1 << 10;
        const PW: u64 = 1 << 11;

        // TODO: Make PR/PW optional. Probably not needed on every fence.
        Self {
            op: IOFENCE_OP | PR | PW,
            addr: 0,
        }
    }
}

// Safety: `Command` is a POD struct without implicit padding and therefore can be initialized
// from a byte array.
unsafe impl DataInit for Command {}

/// The IOMMU command queue.
pub type CommandQueue = Queue<Command, Producer>;

// TODO: Fault queue.
