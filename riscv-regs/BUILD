# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rustfmt_test")

rust_library(
    name = "riscv-regs",
    srcs = glob(["src/**/*.rs"]),
    proc_macro_deps = [
        "@salus-index//:seq-macro",
    ],
    deps = [
        "//riscv-page-tables",
        "//riscv-pages",
        "@salus-index//:riscv-decode",
        "@salus-index//:tock-registers",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["riscv-regs"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["riscv-regs"],
)

rust_doc(
    name = "riscv-regs-doc",
    crate = ":riscv-regs",
)
