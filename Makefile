# To run in gdb: $ make GDB_ARGS='-s -S' run_linux

all: salus

.PHONY: salus
salus:
	cargo build --release --bin salus

run_linux: salus
	     qemu-system-riscv64 \
		     ${GDB_ARGS} \
		     -machine virt -m 4096 -smp 1 \
		     -nographic \
		     -bios ../opensbi/build/platform/generic/firmware/fw_jump.bin \
		     -kernel target/riscv64gc-unknown-none-elf/release/salus \
		     -device loader,file=../linux/arch/riscv/boot/Image,addr=0xc0200000 \
		     -append "console=ttyS0 earlycon=sbi keep_bootcon bootmem_debug"
