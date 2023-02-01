# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_library", "rustfmt_test")

rust_library(
    name = "sync",
    srcs = glob(["src/**/*.rs"]),
    deps = ["@salus-index//:spin"],
)

rust_clippy(
    name = "clippy",
    deps = ["sync"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["sync"],
)
