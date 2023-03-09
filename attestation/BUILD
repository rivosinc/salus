# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

load("@rules_rust//rust:defs.bzl", "rust_clippy", "rust_doc", "rust_library", "rustfmt_test")

rust_library(
    name = "attestation",
    srcs = glob(["src/**/*.rs"]),
    deps = [
        "//rice",
        "//sbi-rs",
        "//sync",
        "@rice-index//:const-oid",
        "@rice-index//:der",
        "@rice-index//:digest",
        "@rice-index//:ed25519",
        "@rice-index//:ed25519-dalek",
        "@rice-index//:generic-array",
        "@rice-index//:hkdf",
        "@rice-index//:hmac",
        "@rice-index//:spki",
        "@salus-index//:arrayvec",
        "@salus-index//:flagset",
        "@salus-index//:hex",
        "@salus-index//:typenum",
    ],
)

rust_clippy(
    name = "clippy",
    deps = ["attestation"],
)

rustfmt_test(
    name = "rustfmt",
    targets = ["attestation"],
)

rust_doc(
    name = "attestation-doc",
    crate = ":attestation",
)
