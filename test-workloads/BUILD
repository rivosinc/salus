# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_clippy", "rust_library", "rustfmt_test")
load("//:objcopy.bzl", "objcopy_to_bin")

objcopy_to_bin(
    name = "tellus_raw",
    src = ":tellus",
)

objcopy_to_bin(
    name = "guestvm_raw",
    src = ":guestvm",
)

rust_binary(
    name = "create_guest_image",
    srcs = glob(["create_guest_image/*.rs"]),
)

# from salus/test-workloads/src/consts.rs
NUM_TELLUS_IMAGE_PAGES = 512

PAGE_SIZE_4K = 4096

max_tellus_size = NUM_TELLUS_IMAGE_PAGES * PAGE_SIZE_4K

genrule(
    name = "tellus_guestvm_rule",
    srcs = [
        ":tellus_raw",
        ":guestvm_raw",
    ],
    outs = ["tellus_guestvm"],
    cmd = "$(location :create_guest_image) $(SRCS) $@ " + str(max_tellus_size),
    tools = [":create_guest_image"],
)

rust_binary(
    name = "tellus",
    srcs = [
        "src/bin/tellus.rs",
        "src/consts.rs",
    ],
    crate_root = "src/bin/tellus.rs",
    data = glob([
        "src/**/*.S",
    ]),
    linker_script = "tellus.lds",
    proc_macro_deps = ["@salus-index//:hex-literal"],
    rustc_flags = [
        "-Ctarget-feature=+v",
        "--codegen=link-arg=-nostartfiles",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//attestation",
        "//device-tree",
        "//rice",
        "//riscv-regs",
        "//s-mode-utils",
        "//sbi-rs",
        "//sync",
        "//test-system",
        "//test-workloads",
        "@rice-index//:der",
        "@salus-index//:arrayvec",
    ],
)

rust_binary(
    name = "guestvm",
    srcs = [
        "src/bin/guestvm.rs",
        "src/consts.rs",
    ],
    crate_root = "src/bin/guestvm.rs",
    data = glob([
        "src/**/*.S",
        "src/**/*.der",
    ]),
    linker_script = "guest.lds",
    proc_macro_deps = ["@salus-index//:hex-literal"],
    rustc_flags = [
        "-Ctarget-feature=+v",
        "--codegen=link-arg=-nostartfiles",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//attestation",
        "//device-tree",
        "//rice",
        "//riscv-regs",
        "//s-mode-utils",
        "//sbi-rs",
        "//test-system",
        "//test-workloads",
        "@rice-index//:der",
        "@salus-index//:arrayvec",
    ],
)

rust_library(
    name = "test-workloads",
    srcs = glob(["src/*.rs"]),
    data = glob([
        "src/**/*.S",
        "src/**/*.der",
    ]),
    proc_macro_deps = ["@salus-index//:hex-literal"],
    deps = [
        "//attestation",
        "//device-tree",
        "//rice",
        "//riscv-regs",
        "//s-mode-utils",
        "//sbi-rs",
        "@rice-index//:der",
        "@salus-index//:arrayvec",
    ],
)

rust_clippy(
    name = "clippy",
    deps = [
        "guestvm",
        "tellus",
        "test-workloads",
    ],
)

rustfmt_test(
    name = "rustfmt",
    targets = [
        "test-workloads",
        "tellus",
        "guestvm",
        "create_guest_image",
    ],
)
