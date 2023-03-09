# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_clippy", "rust_doc", "rustfmt_test")
load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")

rust_binary(
    name = "umode",
    srcs = glob(["src/*.rs"]),
    linker_script = "umode.lds",
    rustc_flags = ["--codegen=link-arg=-nostartfiles"],
    deps = [
        "//data-model",
        "//libuser",
        "//rice",
        "//test-system",
        "//u-mode-api",
        "@rice-index//:const-oid",
        "@rice-index//:der",
        "@rice-index//:ed25519",
        "@rice-index//:ed25519-dalek",
        "@rice-index//:generic-array",
        "@rice-index//:sha2",
        "@rice-index//:signature",
        "@rice-index//:zeroize",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["umode"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["umode"],
)

rust_doc(
    name = "u-mode-doc",
    crate = ":umode",
)
