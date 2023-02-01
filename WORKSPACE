# SPDX-FileCopyrightText: Copyright (c) 2023 by Rivos Inc.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

workspace(name = "salus")

#
# Rivos rules, repositories, and toolchains
#
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_rivos",
    sha256 = "4339236fa2eeed863a664c42c6ffc004e66da349e83891da71dde9cdf9317a09",
    strip_prefix = "rules_rivos-0.1.0",
    urls = ["https://github.com/rivosinc/rules_rivos/archive/refs/tags/v0.1.0.tar.gz"],
)

load("@rules_rivos//lib:repositories.bzl", "rivos_repositories")

rivos_repositories()

register_toolchains(
    "@rules_rivos//toolchains:all",
)

load("@rules_rivos//lib:deps.bzl", "rivos_dependencies")

rivos_dependencies()

#
# Rust rules and toolchains
#

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

# Register Rust toolchains. rules_rust gives us the latest stable Rust
# release from the rules_rust release, and the nightly build from that
# date. For Salus, we use the nightly build to take advantage of some
# nightly-only language features, but we keep it stable and only update
# it once per month.
rust_register_toolchains(
    edition = "2021",
    extra_target_triples = ["riscv64gc-unknown-none-elf"],
    iso_date = "2023-2-26",
)

load("//:deps.bzl", "salus_dependencies")

salus_dependencies()

load("//sbi-rs:deps.bzl", "sbi_dependencies")

sbi_dependencies()

load("//rice:deps.bzl", "rice_dependencies")

rice_dependencies()

load("//:repos.bzl", "salus_repositories")

salus_repositories()
