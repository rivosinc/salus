// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use super::address::*;
use super::bus::PciBus;
use super::config_space::PciFuncConfigSpace;
use super::error::*;
use super::header::*;

/// Common functionality implemented by PCI devices, regardless of type.
pub trait PciDevice {
    /// Returns the config space header for this device.
    fn header(&self) -> &Header;

    /// Returns the PCI bus address of this device.
    fn address(&self) -> Address {
        self.header().address()
    }
}

/// Represents a PCI endpoint.
pub struct PciEndpoint {
    _func_config: PciFuncConfigSpace,
    header: Header,
}

impl PciEndpoint {
    /// Creates a new `PciEndpoint` using `func_config`.
    fn new(func_config: PciFuncConfigSpace, header: Header) -> Result<Self> {
        Ok(Self {
            _func_config: func_config,
            header,
        })
    }
}

impl PciDevice for PciEndpoint {
    fn header(&self) -> &Header {
        &self.header
    }
}

/// Represents a PCI bridge.
pub struct PciBridge {
    func_config: PciFuncConfigSpace,
    header: Header,
    bus_range: BusRange,
    child_bus: Option<PciBus>,
}

impl PciBridge {
    /// Creates a new `PciBridge` using `func_config`. Downstream buses are initially unenumerated.
    fn new(mut func_config: PciFuncConfigSpace, header: Header) -> Result<Self> {
        // Prevent config cycles from passing beyond this bridge until we're ready to enumreate.
        func_config.write_dword(HeaderWord::BusNumber, 0);
        Ok(Self {
            func_config,
            header,
            bus_range: BusRange::default(),
            child_bus: None,
        })
    }

    /// Configures the secondary and subordinate bus numbers of the bridge such that configuration
    /// cycles from `range.start` to `range.end` (inclusive) will be forwarded downstream.
    pub fn assign_bus_range(&mut self, range: BusRange) {
        let bus_dword =
            self.address().bus().bits() | (range.start.bits() << 8) | (range.end.bits() << 16);
        self.func_config
            .write_dword(HeaderWord::BusNumber, bus_dword);
        self.bus_range = range;
    }

    /// Sets the bus that is directly downstream of this bridge.
    pub fn set_child_bus(&mut self, bus: PciBus) {
        self.child_bus = Some(bus)
    }

    /// Returns the secondary bus directly downstream of this bridge.
    pub fn child_bus(&self) -> Option<&PciBus> {
        self.child_bus.as_ref()
    }
}

impl PciDevice for PciBridge {
    fn header(&self) -> &Header {
        &self.header
    }
}

/// The top-level PCI device types.
pub enum PciDeviceType {
    /// A function endpoint (type 0) device.
    Endpoint(PciEndpoint),
    /// A bridge (type 1) device.
    Bridge(PciBridge),
}

impl PciDeviceType {
    /// Creates a `PciDeviceType` from a function config space and config space header.
    pub fn new(func_config: PciFuncConfigSpace, header: Header) -> Result<Self> {
        match header.header_type() {
            HeaderType::Endpoint => Ok(PciDeviceType::Endpoint(PciEndpoint::new(
                func_config,
                header,
            )?)),
            HeaderType::PciBridge => {
                Ok(PciDeviceType::Bridge(PciBridge::new(func_config, header)?))
            }
            h => Err(Error::UnsupportedHeaderType(header.address(), h)),
        }
    }
}

// PciEndpoint and PciBridge hold raw pointers to their config spaces. Access to that config space is
// done through their respective interfaces which allow them to be shared and sent between threads.
unsafe impl Send for PciDeviceType {}
unsafe impl Sync for PciDeviceType {}
