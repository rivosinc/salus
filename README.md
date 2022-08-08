A micro hypervisor for RISC-V systems.

# Quick Start

## Building

```
rustup target add riscv64gc-unknown-none-elf
cargo build
```

## running

There is a `Makefile` provided with targets for running various hosts in qemu.

### Prerequisites

- rust toolchain from [rustup](rustup.rs)
- rustup target add riscv64gc-unknown-none-elf
- install `gcc-riscv64-unknown-elf` for `riscv64-unknown-elf-objcopy`
- qemu-system-riscv64 - build using  qemu
  [instructions](https://wiki.qemu.org/Hosts/Linux) with
  `--target-list=riscv64-softmmu`
  - Note qemu 7.0.0 has a bug emulating riscv and a newer version from git must be used.

## Test VM

A pair of test VMs are located in `test-workloads`.

`tellus` is a target build with `cargo build --bin=tellus` that runs in VS mode
and provides the ability to send test API calls to `salus` running in HS mode.

`guestvm` is a test confidential guest. It is started by `tellus` and used for
testing the guest side of the TSM API.

A makefile shortcut is provided:

`make run_tellus`

This will build salus, tellus, and the guestvm then boot them with the
system-installed qemu.

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
   TBD syscall      SBI (TH-API)    SBI(TG-API)
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
- Guest VM start/stop/scheduling via TEE TH-API provided by salus
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
- Confidential guests use TG-API for salus/host services

### Salus

The code in this repository. An HS-mode hypervisor.

- starts the host and guests
- manages stage-2 translations and IOMMU configuration for guest isolation
- delegates some tasks such as attestation to u-mode helpers
- measured by the trusted firmware/RoT

### Firmware

M-mode code.

OpenSBI currently - boots the system, loads salus to memory and passes it the
device tree.
