// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Trait representing the possible requirements for a page's measurement.
pub trait MeasureRequirement {}

/// The page must be measured before it is mapped.
#[derive(Debug)]
pub enum MeasureRequired {}
impl MeasureRequirement for MeasureRequired {}

/// The page does not need to be measured.
#[derive(Debug)]
pub enum MeasureOptional {}
impl MeasureRequirement for MeasureOptional {}

/// Trait implemented by possible runtime states for a `PhysPage`. We use these states to enforce
/// safe state transitions while a page is "in motion" within the TSM.
pub trait State {}

/// Trait for states in which a `PhysPage` can be mapped into a page table.
pub trait Mappable<M: MeasureRequirement>: State {}

/// Trait for converted, but unassigned, `PhysPage` states.
pub trait Converted: State {}

/// Trait for states in which a `PhysPage` can be assigned for use by a child VM, either as a
/// mappable page or as an internal state page.
pub trait Assignable<M: MeasureRequirement>: Converted {
    /// The type of page that can be mapped to the guest, either cleaned or measured.
    type Mappable: Mappable<M>;
}

/// Trait for states in which a `PhysPage` is (or was) used to store internal data. These states
/// are mutually exclusive with Mappable and Assignable states.
pub trait InternalData: State {}

/// Trait for states in which a `PhysPage` can be transformed into a clean state.
pub trait Cleanable: State {
    /// The page state that results from cleaning this `Cleanable` page.
    type Cleaned: State;
}

/// Trait for states in which a `PhysPage` can be transformed into an initialized page.
pub trait Initializable: State {}

/// A page that has been acquired by invalidating a page table entry. It must be converted before
/// it can be further assigned to a child VM.
#[derive(Debug)]
pub enum Invalidated {}
impl State for Invalidated {}

/// A page that has been converted and has unknown or uninitialized contents. Dirty pages must be
/// cleaned (zeroed out) or initialized before they can be assigned.
#[derive(Debug)]
pub enum ConvertedDirty {}
impl State for ConvertedDirty {}
impl Converted for ConvertedDirty {}
impl Cleanable for ConvertedDirty {
    type Cleaned = ConvertedClean;
}
impl Initializable for ConvertedDirty {}

/// A page that has been zero-filled. Zero-filled pages can be assigned as mapped pages, possibly
/// without measurement, or used to store internal state.
#[derive(Debug)]
pub enum ConvertedClean {}
impl State for ConvertedClean {}
impl Converted for ConvertedClean {}
impl Assignable<MeasureOptional> for ConvertedClean {
    type Mappable = MappableClean;
}
impl Initializable for ConvertedClean {}

/// A page that has been initialized from a user-provided source. Initialized pages can be assigned
/// as mapped pages.
#[derive(Debug)]
pub enum ConvertedInitialized {}
impl State for ConvertedInitialized {}
impl Converted for ConvertedInitialized {}
impl Assignable<MeasureRequired> for ConvertedInitialized {
    type Mappable = MappableInitialized;
}

/// A clean (zero-filled) page that has been assigned as a mapped page.
#[derive(Debug)]
pub enum MappableClean {}
impl State for MappableClean {}
impl Mappable<MeasureOptional> for MappableClean {}

/// An initialized page that is ready to be mapped. Initialized pages must be measured.
#[derive(Debug)]
pub enum MappableInitialized {}
impl State for MappableInitialized {}
impl Mappable<MeasureRequired> for MappableInitialized {}

/// A page that may be used to back an internal data-structure. Pages can only enter this state
/// if they were previously clean.
#[derive(Debug)]
pub enum InternalClean {}
impl State for InternalClean {}
impl InternalData for InternalClean {}

/// A page that was used to back an internal data-structure, but has been released by thata
/// data-structure.
#[derive(Debug)]
pub enum InternalDirty {}
impl State for InternalDirty {}
impl InternalData for InternalDirty {}
impl Cleanable for InternalDirty {
    type Cleaned = InternalClean;
}

/// A shared (and implicitly dirty) page that has been assigned as a mapped page.
#[derive(Debug)]
pub enum MappableShared {}
impl State for MappableShared {}

/// A mapped page that is eligible for conversion to a Shared page.
#[derive(Debug)]
pub enum Shareable {}
impl State for Shareable {}
impl Mappable<MeasureOptional> for MappableShared {}
