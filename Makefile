OBJCOPY ?= riscv64-unknown-elf-objcopy
MACH_ARGS ?= -M virt,aia=aplic-imsic,aia-guests=4 -cpu rv64,x-aia=true
NCPU ?= 1
MEM_SIZE ?= 4096

# Sanitize LOCAL_PATH
ifdef LOCAL_PATH
LOCAL_PATH:=${LOCAL_PATH}/
endif

all: salus tellus guestvm

.PHONY: salus
salus:
	cargo build --release --bin salus

.PHONY: salus_debug
salus_debug:
	cargo build --bin salus

tellus_bin: tellus
	${OBJCOPY} -O binary target/riscv64gc-unknown-none-elf/release/tellus tellus_raw
	${OBJCOPY} -O binary target/riscv64gc-unknown-none-elf/release/guestvm guestvm_raw
	./create_guest_image.sh tellus_raw guestvm_raw tellus_guestvm

guestvm:
	RUSTFLAGS='-Clink-arg=-Tlds/guest.lds' cargo build --package test_workloads --release --bin guestvm

.PHONY: tellus
tellus: guestvm
	cargo build --package test_workloads --bin tellus --release

run_tellus_gdb: tellus_bin salus_debug
	     ${LOCAL_PATH}qemu-system-riscv64 \
		     -s -S \
		     ${MACH_ARGS} -smp ${NCPU} -m ${MEM_SIZE} \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/debug/salus \
		     -device guest-loader,kernel=tellus_guestvm,addr=0xc0200000

run_tellus: tellus_bin salus
	     ${LOCAL_PATH}qemu-system-riscv64 \
		     ${GDB_ARGS} \
		     ${MACH_ARGS} -smp ${NCPU} -m ${MEM_SIZE} \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/release/salus \
		     -device guest-loader,kernel=tellus_guestvm,addr=0xc0200000

run_linux: salus
	     ${LOCAL_PATH}qemu-system-riscv64 \
		     ${GDB_ARGS} \
		     ${MACH_ARGS} -smp ${NCPU} -m ${MEM_SIZE} \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/release/salus \
		     -device guest-loader,kernel=../linux/arch/riscv/boot/Image,addr=0xc0200000 \
		     -append "console=ttyS0 earlycon=sbi keep_bootcon bootmem_debug"
