// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use core::ptr::NonNull;
use device_tree::DeviceTree;
use page_tracking::HwMemMap;
use riscv_pages::{DeviceMemType, RawAddr};
use s_mode_utils::print::*;
use sync::{Mutex, Once};

/// Errors that can be returned by the UART driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// No compatible UART device found in the device tree.
    UartNotFound,
    /// Missing `reg` property or cells in the device tree node.
    MissingRegisters,
    /// Misaligned or otherwise invalid register set specified in the device tree.
    InvalidRegisterLocation,
    /// Failed to add an MMIO region to the system memory map.
    AddingMmioRegion(page_tracking::MemMapError),
}

/// Holds the result of a UART driver operation.
pub type Result<T> = core::result::Result<T, Error>;

// Standard 16500a register set length.
const UART_REGISTERS_LEN: u64 = 8;

#[allow(dead_code)]
enum UartRegister {
    Thr,
    Rbr,
    Dll,
    Ier,
    Dlh,
    Iir,
    Fcr,
    Lcr,
    Mcr,
    Lsr,
    Msr,
    Sr,
}

impl UartRegister {
    fn offset(&self) -> usize {
        use UartRegister::*;
        match self {
            Thr => 0,
            Rbr => 0,
            Dll => 0,
            Ier => 1,
            Dlh => 1,
            Iir => 2,
            Fcr => 2,
            Lcr => 3,
            Mcr => 4,
            Lsr => 5,
            Msr => 6,
            Sr => 7,
        }
    }
}

static UART_DRIVER: Once<UartDriver> = Once::new();

/// Driver for a standard UART.
pub struct UartDriver {
    base_address: Mutex<NonNull<u8>>,
}

impl UartDriver {
    /// Probes for a UART device from `dt` adding its MMIO registers to `mem_map`. Upon success
    /// the UART device is set as the system console.
    pub fn probe_from(dt: &DeviceTree, mem_map: &mut HwMemMap) -> Result<()> {
        let node = dt
            .iter()
            .find(|n| n.compatible(["ns16550a"]))
            .ok_or(Error::UartNotFound)?;
        let mut regs = node
            .props()
            .find(|p| p.name() == "reg")
            .ok_or(Error::MissingRegisters)?
            .value_u64();
        let base_address = regs.next().ok_or(Error::MissingRegisters)?;
        let len = regs.next().ok_or(Error::MissingRegisters)?;
        if (base_address % UART_REGISTERS_LEN != 0) || base_address == 0 || len < UART_REGISTERS_LEN
        {
            return Err(Error::InvalidRegisterLocation);
        }
        // Safety: We trust that the device tree accurately described the location of the UART.
        unsafe {
            mem_map
                .add_mmio_region(DeviceMemType::Uart, RawAddr::supervisor(base_address), len)
                .map_err(Error::AddingMmioRegion)
        }?;
        // Unwrap ok, we've already verified that base_address is non-NULL.
        let uart = UartDriver {
            base_address: Mutex::new(NonNull::new(base_address as _).unwrap()),
        };
        UART_DRIVER.call_once(|| uart);
        Console::set_driver(UART_DRIVER.get().unwrap());
        Ok(())
    }

    fn read_reg(&self, reg: UartRegister) -> u8 {
        let base_address = self.base_address.lock();
        // Safety: We trust firmware to provide a valid base address in the device tree.
        unsafe { core::ptr::read_volatile(base_address.as_ptr().byte_add(reg.offset())) }
    }
}

impl ConsoleDriver for UartDriver {
    /// Write an entire byte sequence to this UART.
    fn write_bytes(&self, bytes: &[u8]) {
        let base_address = self.base_address.lock();
        for &b in bytes {
            // Safety: the caller of ::new() had to guarantee that the given address belongs to an
            // actual UART and that nobody else is using it, thereby making this defined behavior.
            unsafe { core::ptr::write_volatile(base_address.as_ptr(), b) };
        }
    }

    /// Read bytes from this UART.
    fn read_bytes(&self, bytes: &mut [u8]) -> usize {
        let mut n = 0;
        while n < bytes.len() && self.read_reg(UartRegister::Lsr) & 1 != 0 {
            bytes[n] = self.read_reg(UartRegister::Rbr);
            n += 1;
        }
        n
    }
}

// Safety: Access to the pointer to the UART's registers is guarded by a Mutex and the UartDriver
// API guarantees that it is used safely.
unsafe impl Send for UartDriver {}
unsafe impl Sync for UartDriver {}
