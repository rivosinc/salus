all: salus tellus

.PHONY: salus
salus:
	cargo build --release --bin salus

.PHONY: salus_debug
salus_debug:
	cargo build --bin salus

tellus_bin: tellus
	riscv64-unknown-elf-objcopy -O binary target/riscv64gc-unknown-none-elf/release/tellus tellus_raw

.PHONY: tellus
tellus:
	cargo build --bin tellus --release

run_tellus_gdb: tellus_bin salus_debug
	     qemu-system-riscv64 \
		     -s -S \
		     -machine virt -cpu rv64 -m 4096 -smp 1 \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/debug/salus \
		     -device guest-loader,kernel=tellus_raw,addr=0xc0200000

run_tellus: tellus_bin salus
	     qemu-system-riscv64 \
		     ${GDB_ARGS} \
		     -machine virt -m 4096 -smp 1 \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/release/salus \
		     -device guest-loader,kernel=tellus_raw,addr=0xc0200000

run_linux: salus
	     qemu-system-riscv64 \
		     ${GDB_ARGS} \
		     -machine virt -cpu rv64 -m 4096 -smp 1 \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/release/salus \
		     -device guest-loader,kernel=../linux/arch/riscv/boot/Image,addr=0xc0200000 \
		     -append "console=ttyS0 earlycon=sbi keep_bootcon bootmem_debug"
