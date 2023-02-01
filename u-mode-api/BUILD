# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_library", "rustfmt_test")

rust_library(
    name = "u-mode-api",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//attestation",
        "//data-model",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["u-mode-api"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["u-mode-api"],
)
