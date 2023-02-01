# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "hyp-alloc",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//riscv-pages",
        "//sync",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["hyp-alloc"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["hyp-alloc"],
)

rust_test(
    name = "hyp-alloc-test",
    crate = ":hyp-alloc",
)
