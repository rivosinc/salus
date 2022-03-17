A micro hypervisor for RISC-V systems.

## Building

```
rustup target add riscv64gc-unknown-none-elf
cargo build
```

## running

Requires a version of qemu with the H-extension. Currently that requires
pulling a few patches and building locally.

## Test VM

`tellus` is a target build with `cargo build --bin=tellus` that runs in VS mode
and provides the ability to send test API calls to `salus` running in HS mode.
