# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rustfmt_test")

rust_library(
    name = "s-mode-utils",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//sbi-rs",
        "//sync",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["s-mode-utils"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["s-mode-utils"],
)

rust_doc(
    name = "s-mode-utils-doc",
    crate = ":s-mode-utils",
)
