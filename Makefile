# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Path variables:
#
#  RV64_PREFIX: Path prefix for riscv64 toolchain. Default: use toolchain in $PATH.
#  QEMU: Path to compiled QEMU tree. Default: use QEMU in $PATH.
#  LINUX: Path to compiled Linux kernel tree.
#  DEBIAN: Path to a pre-baked Debian image.

RV64_PREFIX ?= riscv64-unknown-elf-
OBJCOPY ?= $(RV64_PREFIX)objcopy

QEMU ?=
ifneq ($(QEMU),)
QEMU_PATH := $(QEMU)/build/riscv64-softmmu/
else
QEMU_PATH :=
endif
QEMU_BIN := $(QEMU_PATH)qemu-system-riscv64

LINUX ?=
LINUX_BIN := $(LINUX)/arch/riscv/boot/Image

DEBIAN ?=
ROOTFS_IMAGE := $(DEBIAN)/image.qcow2
INITRD_IMAGE := $(DEBIAN)/initrd
BUILDROOT_IMAGE := $(BUILDROOT)/output/images/rootfs.ext2

RELEASE_BINS := target/riscv64gc-unknown-none-elf/release/
DEBUG_BINS := target/riscv64gc-unknown-none-elf/debug/
TEST_BINS := target/riscv64gc-unknown-none-elf/debug/deps/

KERNEL_ADDR := 0xc0200000
INITRD_ADDR := 0xc2200000
BOOTARGS := console=hvc0 earlycon=sbi
IOMMU_ARGS := -device x-riscv-iommu-pci
NETWORK_ARGS := -netdev user,id=usernet,hostfwd=tcp:127.0.0.1:7722-0.0.0.0:22 -device e1000e,netdev=usernet
NVME_DEVICE_ARGS := -device nvme,serial=deadbeef,drive=hd

# QEMU options:
#
#  NCPU: Number of CPUs to boot with. Default: 2
#  MEM_SIZE: RAM size for the emulated system. Default: 2GB
#  EXTRA_QEMU_ARGS: Any extra flags to pass to QEMU (e.g. other devices).

NCPU ?= 2
MEM_SIZE ?= 2048
EXTRA_QEMU_ARGS ?=

ifneq ($(VECTORS),)
CPU_TYPE := rv64,v=on,vlen=256,elen=64
else
CPU_TYPE := rv64
endif

CPU_ARGS := $(CPU_TYPE),x-smaia=true,x-ssaia=true,sscofpmf=true

MACH_ARGS := -M virt,aia=aplic-imsic,aia-guests=4 -cpu $(CPU_ARGS)
MACH_ARGS += -smp $(NCPU) -m $(MEM_SIZE) -nographic

HOST_TRIPLET := $(shell cargo -Vv | grep '^host:' | awk ' { print $$2; } ')

all: salus tellus guestvm

.PHONY: check
check:
	cargo test \
		--target $(HOST_TRIPLET) \
		--workspace \
		--exclude test_workloads \
		--exclude libuser \
		--lib
	cargo test \
		--target $(HOST_TRIPLET) \
		--workspace \
		--exclude test_workloads \
		--exclude libuser \
		--doc

CARGO_FLAGS :=

.PHONY: salus
salus: umode sbirs
	cargo build $(CARGO_FLAGS) --release --bin salus

.PHONY: salus_debug
salus_debug: umode sbirs
	cargo build $(CARGO_FLAGS) --bin salus

salus_test:
	cargo test --no-run $(CARGO_FLAGS) --bin salus

tellus_bin: tellus
	${OBJCOPY} -O binary $(RELEASE_BINS)tellus tellus_raw
	${OBJCOPY} -O binary $(RELEASE_BINS)guestvm guestvm_raw
	./create_guest_image.sh tellus_raw guestvm_raw tellus_guestvm

guestvm: sbirs
	RUSTFLAGS='-Ctarget-feature=+v -Clink-arg=-Tlds/guest.lds' cargo build $(CARGO_FLAGS) --package test_workloads --release --bin guestvm

.PHONY: tellus
tellus: guestvm sbirs
	cargo build $(CARGO_FLAGS) --package test_workloads --bin tellus --release

.PHONY: umode
umode: sbirs
	RUSTFLAGS='-Clink-arg=-Tlds/umode.lds' cargo build  --release --package umode

# Ensures the sbi submodule is present and up-to-date.
.PHONY: sbirs
sbirs:
	# Check if the submodule needs to be initialized
	if [ "$$(git submodule status | grep -c '^-')" -gt 0 ]; then \
		git submodule update --init; \
	fi

# Runnable targets:
#
#  run_salus_test: runs salus unit tests under qemu
#  run_tellus_gdb: Run Tellus as the host VM with GDB debugging enabled.
#  run_tellus: Run Tellus as the host VM.
#  run_linux: Run a bare Linux kernel as the host VM.
#  run_debian: Run a Linux kernel as the host VM with a Debian rootfs.
#  run_buildroot: Run a Linux kernel as the host VM with a buildroot rootfs

run_salus_test: salus_test
	$(QEMU_BIN) \
		-s $(MACH_ARGS) \
		-kernel $(TEST_BINS)salus*[0-9] \
		-device guest-loader,kernel=tellus_guestvm,addr=$(KERNEL_ADDR) \
		$(EXTRA_QEMU_ARGS)

run_tellus_gdb: tellus_bin salus_debug
	$(QEMU_BIN) \
		-s -S $(MACH_ARGS) \
		-kernel $(DEBUG_BINS)salus \
		-device guest-loader,kernel=tellus_guestvm,addr=$(KERNEL_ADDR) \
		$(EXTRA_QEMU_ARGS)

run_tellus: tellus_bin salus
	$(QEMU_BIN) \
		$(MACH_ARGS) \
		-kernel $(RELEASE_BINS)salus \
		-device guest-loader,kernel=tellus_guestvm,addr=$(KERNEL_ADDR) \
		$(EXTRA_QEMU_ARGS)

run_linux: salus
	$(QEMU_BIN) \
		$(MACH_ARGS) \
		-kernel $(RELEASE_BINS)salus \
		-device guest-loader,kernel=$(LINUX_BIN),addr=$(KERNEL_ADDR) \
		-append "$(BOOTARGS)" \
		$(EXTRA_QEMU_ARGS)

run_debian: salus
	$(QEMU_BIN) \
		$(MACH_ARGS) \
		-kernel $(RELEASE_BINS)salus \
		-device guest-loader,kernel=$(LINUX_BIN),addr=$(KERNEL_ADDR) \
		-append "$(BOOTARGS) root=LABEL=rootfs" \
		-device guest-loader,initrd=$(INITRD_IMAGE),addr=$(INITRD_ADDR) \
		-drive file=$(ROOTFS_IMAGE),if=none,id=hd \
		$(NVME_DEVICE_ARGS) \
		$(IOMMU_ARGS) \
		$(NETWORK_ARGS) \
		$(EXTRA_QEMU_ARGS)
run_buildroot: salus
	$(QEMU_BIN) \
		$(MACH_ARGS) \
		-kernel $(RELEASE_BINS)salus \
		-device guest-loader,kernel=$(LINUX_BIN),addr=$(KERNEL_ADDR) \
		-append "$(BOOTARGS) root=/dev/nvme0n1" \
		-drive file="$(BUILDROOT_IMAGE),format=raw,id=hd" \
		$(NVME_DEVICE_ARGS) \
		$(IOMMU_ARGS) \
		$(NETWORK_ARGS) \
		$(EXTRA_QEMU_ARGS)

run_buildroot_debug: salus_debug
	$(QEMU_BIN) \
		$(MACH_ARGS) \
		-kernel $(DEBUG_BINS)salus \
		-device guest-loader,kernel=$(LINUX_BIN),addr=$(KERNEL_ADDR) \
		-append "$(BOOTARGS) root=/dev/nvme0n1" \
		-drive file="$(BUILDROOT_IMAGE),format=raw,id=hd" \
		$(NVME_DEVICE_ARGS) \
		$(IOMMU_ARGS) \
		$(NETWORK_ARGS) \
		-s -S \
		$(EXTRA_QEMU_ARGS)

.PHONY: lint
lint:
	cargo clippy --all -- -D warnings -Wmissing-docs

.PHONY: format
format:
	cargo fmt -- --check --config format_code_in_doc_comments=true

# Currently (Nightly 1.65) the test_workloads crate causes rustdoc to panic, so exclude it.
.PHONY: doc
doc:
	cargo doc --workspace --exclude test_workloads

.PHONY: ci
ci: salus guestvm tellus lint format check doc
