// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::{alloc::Allocator, cmp, fmt, marker::PhantomData};

/// A simple type-safe arena with support for using a custom allocator. Can be used to implement
/// index-based data-structures like trees or graphs. Backed by a `Vec<Option<T>>`.
///
/// TODO: Support ID re-use, e.g. via a generation counter in `ArenaId`.
pub struct Arena<T, A: Allocator> {
    vals: Vec<Option<T>, A>,
}

/// An index used to retrieve an object from the arena. Type-safe in order to prevent potential
/// mis-use of the index.
pub struct ArenaId<T> {
    index: usize,
    phantom: PhantomData<*const T>,
}

// ArenaId<T> is trivially Send/Sync since it's just an integer. Access to the T it refers to must
// be done through the Arena<T> interface, which itself is only Send/Sync if T is Send/Sync.
unsafe impl<T> Send for ArenaId<T> {}
unsafe impl<T> Sync for ArenaId<T> {}

impl<T> Clone for ArenaId<T> {
    fn clone(&self) -> Self {
        ArenaId {
            index: self.index,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for ArenaId<T> {}

impl<T> PartialEq for ArenaId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> PartialOrd for ArenaId<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.index.partial_cmp(&other.index)
    }
}

impl<T> fmt::Display for ArenaId<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.index)
    }
}

impl<T> fmt::Debug for ArenaId<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_struct("ArenaId")
            .field("index", &self.index)
            .finish()
    }
}

impl<T, A: Allocator> Arena<T, A> {
    /// Creates a new arena with the given allocator.
    pub fn new(alloc: A) -> Self {
        Self {
            vals: Vec::new_in(alloc),
        }
    }

    /// Creates a new arena with the given allocator and initial capacity.
    pub fn new_with_capacity(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let mut vec = Vec::new_in(alloc);
        vec.try_reserve(capacity)?;
        let arena = Self { vals: vec };
        Ok(arena)
    }

    /// Inserts the given value in the arena returning its ID, or an error if the allocation failed.
    pub fn try_insert(&mut self, val: T) -> Result<ArenaId<T>, TryReserveError> {
        self.vals.try_reserve(1)?;
        self.vals.push(Some(val));
        let id = ArenaId {
            index: self.vals.len() - 1,
            phantom: PhantomData,
        };
        Ok(id)
    }

    /// Inserts the given value in the arena, returning its ID. Panics if allocation fails.
    pub fn insert(&mut self, val: T) -> ArenaId<T> {
        self.try_insert(val).unwrap()
    }

    /// Removes the object with the given ID from the arena. Subsequent calls to `get()` or
    /// `get_mut()` with the same ID will return `None`.
    pub fn remove(&mut self, id: ArenaId<T>) {
        if let Some(v) = self.vals.get_mut(id.index) {
            *v = None;
        }
    }

    /// Returns a reference to the object in the arean with the given ID, if it exists.
    pub fn get(&self, id: ArenaId<T>) -> Option<&T> {
        self.vals.get(id.index)?.as_ref()
    }

    /// Returns a mutable reference to the object in the arean with the given ID, if it exists.
    pub fn get_mut(&mut self, id: ArenaId<T>) -> Option<&mut T> {
        self.vals.get_mut(id.index)?.as_mut()
    }
}

impl<T: Default, A: Allocator> Arena<T, A> {
    /// Creates a new object in the arena, returning its ID.
    pub fn alloc(&mut self) -> ArenaId<T> {
        self.insert(T::default())
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Arena<T, A> {
    fn clone(&self) -> Self {
        Self {
            vals: self.vals.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn arena_basics() {
        let mut arena = Arena::<u32, _>::new(Global);
        let id0 = arena.insert(1);
        let id1 = arena.alloc();
        {
            let val1 = arena.get_mut(id1).unwrap();
            *val1 = 2;
        }
        {
            let val0 = arena.get(id0).unwrap();
            assert_eq!(*val0, 1);
        }
        {
            let val1 = arena.get(id1).unwrap();
            assert_eq!(*val1, 2);
        }
        arena.remove(id0);
        let _ = arena.insert(3);
        assert!(arena.get(id0).is_none());
    }
}
