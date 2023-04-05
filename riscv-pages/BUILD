# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "riscv-pages",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//sbi-rs",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["riscv-pages"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["riscv-pages"],
)

rust_test(
    name = "riscv-pages-test",
    crate = ":riscv-pages",
    rustc_flags = [
        "-Dwarnings",
    ],
)

rust_doc(
    name = "riscv-pages-doc",
    crate = ":riscv-pages",
)
