# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "device-tree",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//hyp-alloc",
        "@salus-index//:fdt-rs",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["device-tree"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["device-tree"],
)

rust_test(
    name = "device-tree-test",
    crate = ":device-tree",
)
