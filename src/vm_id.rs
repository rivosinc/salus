// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use riscv_page_tables::tlb;
use riscv_regs::{hgatp, ReadWriteable, Readable, Writeable, CSR};
use spin::Once;

/// Number of supported bits for VMID in HGATP csr.
static VMIDLEN: Once<u64> = Once::new();

/// Represents the VMID tag for a VM, as programmed into the HGATP CSR. VMIDs are versioned in
/// order to detect rollover of the VMID counter.
#[derive(Clone, Copy, Debug, Default)]
pub struct VmId {
    vmid: u64,
    version: u64,
}

impl VmId {
    /// Returns the raw VMID value.
    pub fn vmid(&self) -> u64 {
        self.vmid
    }

    /// Returns the version number of this VMID.
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Tracks the assignment of VMIDs for a particular physcial CPU.
pub struct VmIdTracker {
    next_vmid: u64,
    current_version: u64,
}

// Probes HGATP for the number of supported VMID bits.
fn get_vmid_bits() -> u64 {
    let old = CSR.hgatp.get();
    CSR.hgatp.modify(hgatp::vmid.val(hgatp::vmid.mask));
    let vmid_bits = CSR.hgatp.read(hgatp::vmid).trailing_ones() as u64;
    CSR.hgatp.set(old);
    vmid_bits
}

/// Returns the VMID length.
pub fn get_vmid_len() -> u64 {
    // Unwrap okay: this is called after `VmIdTracker::init()`
    *VMIDLEN.get().unwrap()
}

impl VmIdTracker {
    pub fn init() {
        VMIDLEN.call_once(get_vmid_bits);
    }

    /// Returns an initialized `VmIdTracker`.
    pub const fn new() -> Self {
        Self {
            next_vmid: 0,
            current_version: 0,
        }
    }

    /// Returns the current VMID version number.
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Assigns a VMID, rolling over and flushing the TLB if necessary.
    pub fn next_vmid(&mut self) -> VmId {
        let vmid_bits = get_vmid_len();
        if self.next_vmid == 0 {
            // We rolled over. Bump the version and flush everything since we're now potentially
            // reusing old VMIDs.
            self.current_version += 1;
            tlb::hfence_gvma(None, None);
        }
        let vmid = self.next_vmid;
        self.next_vmid = (self.next_vmid + 1) & ((1 << vmid_bits) - 1);
        VmId {
            vmid,
            version: self.current_version,
        }
    }
}
