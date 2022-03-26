A micro hypervisor for RISC-V systems.

# Quick Start

## Building

```
rustup target add riscv64gc-unknown-none-elf
cargo build
```

## running

There is a `Makefile` provided with targets for running various hosts in qemu.

A recent qemu-system-riscv64 is required as is a copy of openSBI firmware and a
linux kernel if that is the desired payload. The paths to the various
components can be seen in the Makefile.

## Test VM

`tellus` is a target build with `cargo build --bin=tellus` that runs in VS mode
and provides the ability to send test API calls to `salus` running in HS mode.

A makefile shortcut is provided:

`make run_tellus`

This will build salus and tellus, then boot them with the system-install qemu.
A built copy of openSBI is required to be in the expected path.

# Overview - Initial prototype

```
  +---U-mode--+ +-----VS-mode-----+ +-VS-mode-+
  |           | |                 | |         |
  |           | | +---VU-mode---+ | |         |
  |   Salus   | | | VMM(crosvm) | | |  Guest  |
  | Delegated | | +-------------+ | |         |
  |   Tasks   | |                 | |         |
  |           | |    Host(linux)  | |         |
  +-----------+ +-----------------+ +---------+
        |                |               |
   TBD syscall          SBI          SBI(TG-API)
        |                |               |
  +-------------HS-mode-----------------------+
  |       Salus                               |
  +-------------------------------------------+
                         |
                        SBI
                         |
  +----------M-mode---------------------------+
  |       Firmware(OpenSBI)                   |
  +-------------------------------------------+
```

### Host

Normally Linux, this is the primary operating system for the device running in
VS mode.

Responsibilities:
- Scheduling
- Memory allocation (except memory kept by firmware and salus at boot)
- Guest VM start/stop/scheduling via TEE API provided by salus
- Device drivers and delegation

#### VMM

The virtual machine manager that runs in userspace of the host.

- qemu/kvm or crosvm
- configures memory and devices for guests
- runs any virtualized or para virtualized devices
- runs guests with `vcpu_run`.

### Guests

VS-mode operating systems started by the host.

- Can run confidential or shared workloads.
- Uses memory shared from or donated by the host
- scheduled by the host
- can start sub-guests

### Salus

The code in this repository. An HS-mode hypervisor.

- starts the host and guests
- manages stage-2 translations and IOMMU configuration for guest isolation
- delegates some tasks such as attestation to u-mode helpers

### Firmware

M-mode code.

OpenSBI currently - boots the system, loads salus to memory and passes it the
device tree.
