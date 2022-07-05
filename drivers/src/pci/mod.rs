// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

mod address;
mod config_space;
mod header;

use address::*;
use config_space::*;
use header::*;

use device_tree::DeviceTree;
use page_tracking::HwMemMap;

use core::alloc::Allocator;

/// Top level PCI errors
#[derive(Debug)]
pub enum Error {
    /// Creating a PCI config space from the device tree.
    CreatingConfigSpace(config_space::Error),
    /// Invalid value in a PCI header at `address`.
    UnknownHeaderType(Address),
}
/// Top level results from PCI operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Finds the PCI bus in the device tree and scans the  buses, calling the provided callback with
/// each endpoint.
pub fn probe_pci<A: Allocator + Clone, F>(
    dt: &DeviceTree<A>,
    mem_map: &mut HwMemMap,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&Header),
{
    let pci = PciConfigSpace::probe_from(dt, mem_map).map_err(Error::CreatingConfigSpace)?;
    for bus in pci.busses() {
        for dev in bus.devices() {
            for header in dev.functions() {
                match header.header_type() {
                    Some(HeaderType::Endpoint) => f(&header),
                    // TODO - scan the bridges.
                    Some(HeaderType::PciBridge) => (),
                    Some(HeaderType::CardBusBridge) => (),
                    None => return Err(Error::UnknownHeaderType(header.address())),
                }
            }
        }
    }
    Ok(())
}
