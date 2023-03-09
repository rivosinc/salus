# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rustfmt_test")

rust_library(
    name = "libuser",
    srcs = glob(["src/**/*.rs"]),
    compile_data = glob(["src/**/*.S"]),
    deps = ["//u-mode-api"],
)

rust_clippy(
    name = "clippy",
    deps = ["libuser"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["libuser"],
)

rust_doc(
    name = "libuser-doc",
    crate = ":libuser",
)
