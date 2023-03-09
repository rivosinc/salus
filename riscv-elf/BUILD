# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "riscv-elf",
    srcs = glob(["src/**/*.rs"]),
    deps = ["@salus-index//:arrayvec"],
)

rust_clippy(
    name = "clippy",
    deps = ["riscv-elf"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["riscv-elf"],
)

rust_test(
    name = "riscv-elf-test",
    crate = ":riscv-elf",
)

rust_doc(
    name = "riscv-elf-doc",
    crate = ":riscv-elf",
)
