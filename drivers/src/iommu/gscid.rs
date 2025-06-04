// SPDX-FileCopyrightText: 2025 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use riscv_pages::*;
use sync::Mutex;

use super::error::*;

/// Global Soft-Context ID. The equivalent of hgatp.VMID, but always 16 bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GscId(u16);

impl GscId {
    /// Creates a `GscId` from the raw `id`.
    pub(super) fn new(id: u16) -> Self {
        GscId(id)
    }

    /// Returns the raw bits of this `GscId`.
    pub fn bits(&self) -> u16 {
        self.0
    }
}

// Tracks the state of an allocated global soft-context ID (GSCID).
#[derive(Clone, Copy, Debug)]
pub(super) struct GscIdState {
    pub(super) owner: PageOwnerId,
    pub(super) ref_count: usize,
}

// We use a fixed-sized array to track available GSCIDs. We can't use a versioning scheme like we
// would for CPU VMIDs since reassigning GSCIDs on overflow would require us to temporarily disable
// DMA from all devices, which is extremely disruptive. Set a max of 64 allocated GSCIDs for now
// since it's unlikely we'll have more than that number of active VMs with assigned devices for
// the time being.
const MAX_GSCIDS: usize = 64;

// The global GSCID allocation table.
pub(super) static GSCIDS: Mutex<[Option<GscIdState>; MAX_GSCIDS]> = Mutex::new([None; MAX_GSCIDS]);

/// Allocates a new GSCID for `owner`.
pub fn alloc_gscid(owner: PageOwnerId) -> Result<GscId> {
    let mut gscids = GSCIDS.lock();
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
pub fn free_gscid(gscid: GscId) -> Result<()> {
    let mut gscids = GSCIDS.lock();
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
