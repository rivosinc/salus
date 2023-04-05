# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "data-model",
    srcs = glob(["src/**/*.rs"]),
    deps = ["@salus-index//:static_assertions"],
)

rust_clippy(
    name = "clippy",
    deps = ["data-model"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["data-model"],
)

rust_test(
    name = "data-model-test",
    testonly = True,
    crate = ":data-model",
    rustc_flags = [
        "-Dwarnings",
    ],
)

rust_doc(
    name = "data-model-doc",
    crate = ":data-model",
)
