# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "mtt",
    srcs = glob(["src/**/*.rs"]),
    crate_root = "src/lib.rs",
    deps = [
        "//riscv-pages",
        "//page-tracking",
        "//sync",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["mtt"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["mtt"],
)

rust_test(
    name = "mtt-test",
    crate = ":mtt",
)
