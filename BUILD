# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

# to build salus:
# bazel build //:salus-all

# before pull request
# bazel build //:clippy-all
# bazel test //:rustfmt-all
# bazel test //:test-all

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_clippy", "rust_doc", "rust_test", "rustfmt_test")
load("//:objcopy.bzl", "objcopy_to_object")
load("//:lds.bzl", "lds_rule")

filegroup(
    name = "salus-all",
    srcs = [
        "salus",
        "//test-workloads:tellus_guestvm",
    ],
)

filegroup(
    name = "clippy-all",
    srcs = [
        "salus-clippy",
        "//attestation:clippy",
        "//data-model:clippy",
        "//device-tree:clippy",
        "//drivers:clippy",
        "//hyp-alloc:clippy",
        "//libuser:clippy",
        "//mtt:clippy",
        "//page-tracking:clippy",
        "//rice:clippy",
        "//riscv-elf:clippy",
        "//riscv-page-tables:clippy",
        "//riscv-pages:clippy",
        "//riscv-regs:clippy",
        "//s-mode-utils:clippy",
        "//sbi-rs:clippy",
        "//test-system:clippy",
        "//test-workloads:clippy",
        "//u-mode:clippy",
        "//u-mode-api:clippy",
    ],
)

filegroup(
    name = "doc-all",
    srcs = [
        "salus-doc",
        "//attestation:attestation-doc",
        "//data-model:data-model-doc",
        "//device-tree:device-tree-doc",
        "//drivers:drivers-doc",
        "//hyp-alloc:hyp-alloc-doc",
        "//libuser:libuser-doc",
        "//page-tracking:page-tracking-doc",
        "//riscv-elf:riscv-elf-doc",
        "//riscv-page-tables:riscv-page-tables-doc",
        "//riscv-pages:riscv-pages-doc",
        "//riscv-regs:riscv-regs-doc",
        "//s-mode-utils:s-mode-utils-doc",
        "//sync:sync-doc",
        "//test-system:test-system-doc",
        "//u-mode:u-mode-doc",
        "//u-mode-api:u-mode-api-doc",
    ],
)

test_suite(
    name = "rustfmt-all",
    tests = [
        "salus-rustfmt",
        "//attestation:rustfmt",
        "//data-model:rustfmt",
        "//device-tree:rustfmt",
        "//drivers:rustfmt",
        "//hyp-alloc:rustfmt",
        "//libuser:rustfmt",
        "//mtt:rustfmt",
        "//page-tracking:rustfmt",
        "//rice:rustfmt",
        "//riscv-elf:rustfmt",
        "//riscv-page-tables:rustfmt",
        "//riscv-pages:rustfmt",
        "//riscv-regs:rustfmt",
        "//s-mode-utils:rustfmt",
        "//sbi-rs:rustfmt",
        "//test-system:rustfmt",
        "//test-workloads:rustfmt",
        "//u-mode:rustfmt",
        "//u-mode-api:rustfmt",
    ],
)

test_suite(
    name = "test-all",
    tests = [
        "//data-model:data-model-test",
        "//device-tree:device-tree-test",
        "//drivers:drivers-test",
        "//hyp-alloc:hyp-alloc-test",
        "//mtt:mtt-test",
        "//page-tracking:page-tracking-test",
        "//rice:rice-test",
        "//riscv-elf:riscv-elf-test",
        "//riscv-page-tables:riscv-page-tables-test",
        "//riscv-pages:riscv-pages-test",
    ],
)

###############

config_setting(
    name = "debug_mode",
    values = {
        "compilation_mode": "dbg",
    },
)

config_setting(
    name = "optimized_mode",
    values = {
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "fastbuild",
    values = {
        "compilation_mode": "fastbuild",
    },
)

lds_rule(
    name = "l_rule",
    template = "src/salus_lds.tmpl",
)

objcopy_to_object(
    name = "umode_to_object",
    src = "//u-mode:umode",
    out = "umode.o",
)

salus_deps = [
        "//attestation",
        "//data-model",
        "//device-tree",
        "//drivers",
        "//hyp-alloc",
        "//mtt",
        "//page-tracking",
        "//rice",
        "//riscv-elf",
        "//riscv-page-tables",
        "//riscv-pages",
        "//riscv-regs",
        "//s-mode-utils",
        "//sbi-rs",
        "//sync",
        "//test-system",
        "//u-mode-api",
        "@rice-index//:const-oid",
        "@rice-index//:der",
        "@rice-index//:digest",
        "@rice-index//:ed25519-dalek",
        "@rice-index//:generic-array",
        "@rice-index//:hkdf",
        "@rice-index//:sha2",
        "@rice-index//:signature",
        "@salus-index//:arrayvec",
        "@salus-index//:memoffset",
        "@salus-index//:static_assertions",
]

rust_binary(
    name = "salus",
    srcs = glob(["src/*.rs"]),
    compile_data = glob(["src/*.S"]) + [
        ":umode_to_object",
        ":l_rule",
    ],
    rustc_flags = [
        "-Ctarget-feature=+h",
        "--codegen=link-arg=-nostartfiles",
        "-Clink-arg=-T$(location //:l_rule)",
    ],
    deps = salus_deps,
)

rust_clippy(
    name = "salus-clippy",
    deps = ["salus"],
)

rustfmt_test(
    name = "salus-rustfmt",
    targets = ["salus"],
)

rust_doc(
    name = "salus-doc",
    crate = ":salus",
)

rust_test(
    name = "salus-unit-tests",
    srcs = glob(["src/*.rs"]),
    crate_root = "src/main.rs",
    data = glob(["src/*.S"]) + ["src/salus-test.lds"],
    rustc_flags = [
        "-Ctarget-feature=+h",
        "-Clink-arg=-Tsrc/salus-test.lds",
        "--codegen=link-arg=-nostartfiles",
        "-Dwarnings",
    ],
    deps = salus_deps,
)
