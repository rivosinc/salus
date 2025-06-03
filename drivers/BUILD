# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rust_test", "rustfmt_test")

rust_library(
    name = "drivers",
    srcs = glob(["src/**/*.rs"]),
    proc_macro_deps = ["@salus-index//:enum_dispatch"],
    deps = [
        "//data-model",
        "//device-tree",
        "//hyp-alloc",
        "//page-tracking",
        "//riscv-page-tables",
        "//riscv-pages",
        "//riscv-regs",
        "//s-mode-utils",
        "//sbi-rs",
        "//sync",
        "@salus-index//:arrayvec",
        "@salus-index//:const-field-offset",
        "@salus-index//:memoffset",
        "@salus-index//:static_assertions",
        "@salus-index//:tock-registers",
    ],
    crate_features = select({
      "//:unsafe_enhanced_allocation": ["unsafe_enhanced_allocation"],
      "//conditions:default": [],
    }) + select({
      "//:hardware_ad_updates": ["hardware_ad_updates"],
      "//conditions:default": [],
    }),
)

rust_clippy(
    name = "clippy",
    deps = ["drivers"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["drivers"],
)

rust_test(
    name = "drivers-test",
    crate = ":drivers",
    rustc_flags = [
        "-Dwarnings",
    ],
)

rust_doc(
    name = "drivers-doc",
    crate = ":drivers",
)
