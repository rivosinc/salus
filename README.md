A micro hypervisor for RISC-V systems.

# Quick Start

## Building

```
rustup target add riscv64gc-unknown-none-elf
make salus
```

Note that Salus relies on unstable features from the nightly toolchain so there
may be build breakages from time-to-time. Try running `rustup upgrade` first
should you run into build failures.

## Running

The `Makefile` provides targets for running various hosts in QEMU.

### Prerequisites

Toolchains:
- Rust: Install [rustup](https://www.rust-lang.org/tools/install), then: `rustup target add riscv64gc-unknown-none-elf`
- GCC: Install `gcc-riscv64-unknown-elf` built from
  [RISC-V collab](https://github.com/riscv-collab/riscv-gnu-toolchain#installation-linux)

QEMU:
- Out-of-tree patches are required; see table below.
- Install `libslirp-dev` for QEMU to build SLIRP network stack
- Build using QEMU [instructions](https://wiki.qemu.org/Hosts/Linux) with
  `--target-list=riscv64-softmmu`
- Set the `QEMU=` variable to point to the compiled QEMU tree when using the
  `make run_*` targets described below.

Linux kernel:
- Out-of-tree patches are required; see table below.
- Build: `ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- make defconfig Image`
- Set the `LINUX=` variable to point to the compiled Linux kernel tree when
  using linux related targets described below.

Buildroot:
- Out-of-tree patches are required; see table below.
- Build: `make qemu_riscv64_virt_defconfig && make`
- Set the `BUILDROOT=` variable to point to the buildroot source directory while
  running `make run_buildroot` targets described below.

Debian:
- Download and extract a pre-baked `riscv64-virt` image from https://people.debian.org/~gio/dqib/.
- Set the `DEBIAN=` variable to point to the extracted archive when using the
  `make run_debian` target described below.

Latest known-working branches:

| Project | Branch |
| ------- | ------ |
| QEMU    | https://github.com/rivosinc/qemu/tree/salus-integration-10312022 |
| Linux   | https://github.com/rivosinc/linux/tree/salus-integration-10312022 |
| Buildroot| https://github.com/rivosinc/buildroot/tree/salus-integration-2022.08.2 |

### Linux VM

The `make run_linux` target will boot a bare Linux kernel as the host VM
that will panic upon reaching `init` due to the lack of a root filesystem.

To boot a more functional Linux VM, use the `make run_debian` target which
will boot a Debian VM with emulated storage and network devices using pre-baked
Debian initrd and rootfs images.

Example:

```
make run_debian \
    QEMU=<path-to-qemu-tree> \
    LINUX=<path-to-linux-tree> \
    DEBIAN=<path-to-pre-baked-image>
```

To boot a quick functional Linux VM with busybox based rootfs built from
buildroot, use the `make run_buildroot` target. The above buildroot tree
must be compiled to generate the rootfs with networking enabled.

Example:

```
make run_buildroot \
    QEMU=<path-to-qemu-tree> \
    LINUX=<path-to-linux-tree> \
    BUILDROOT=<path-to-buildroot repo>
```

Once booted, the VM can be SSH'ed into with `root:root` at `localhost:7722`.

Additional emulated devices may be added with the `EXTRA_QEMU_ARGS` Makefile
variable. Note that only PCI devices using MSI/MSI-X will be usable by the VM.
`virtio-pci` devices may also be used with `iommu_platform=on,disable-legacy=on`
flags.

Example:

```
make run_debian ... \
     EXTRA_QEMU_ARGS="-device virtio-net-pci,iommu_platform=on,disable-legacy=on"
```

### Test VM

A pair of test VMs are located in `test-workloads`.

`tellus` is a target build with `make tellus` that runs in VS mode
and provides the ability to send test API calls to `salus` running in HS mode.

`guestvm` is a test confidential guest. It is started by `tellus` and used for
testing the guest side of the TSM API.

A makefile shortcut is provided:

```
make run_tellus \
    QEMU=<path-to-qemu-tree>
```

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

OpenSBI currently boots salus from the memory (0x80200000) where qemu loader
loaded it and passes the device tree to Salus.

The above instructions use OpenSBI inbuilt in Qemu. If OpenSBI needs to be
built from scratch, fw_dynamic should be used for `-bios` argument in the qemu
commandline.

### Vectors

Salus is able to detect if the CPU supports the vector extension. The same
binary will run on processors with or without the exension, and will enable
vector code if it is present.
