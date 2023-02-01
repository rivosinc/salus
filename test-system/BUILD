# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_library", "rustfmt_test")

rust_library(
    name = "test-system",
    srcs = glob(["src/**/*.rs"]),
)

rust_clippy(
    name = "clippy",
    deps = ["test-system"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["test-system"],
)
