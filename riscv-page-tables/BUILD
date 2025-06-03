# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "riscv-page-tables",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//page-tracking",
        "//riscv-pages",
        "//sync",
    ],
    rustc_flags = [
        "-Ctarget-feature=+h",
    ],
    crate_features = select({
      "//:hardware_ad_updates": ["hardware_ad_updates"],
      "//conditions:default": [],
    }),
)

rust_clippy(
    name = "clippy",
    deps = ["riscv-page-tables"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["riscv-page-tables"],
)

rust_test(
    name = "riscv-page-tables-test",
    crate = ":riscv-page-tables",
    rustc_flags = [
        "-Dwarnings",
    ],
)

rust_doc(
    name = "riscv-page-tables-doc",
    crate = ":riscv-page-tables",
)
