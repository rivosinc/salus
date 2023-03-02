# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")

# on changes to crate depdencies, run the following command:
# scripts/repin.sh

def salus_dependencies():
    crates_repository(
        name = "salus-index",
        isolated = False,
        cargo_lockfile = "@salus//bazel-locks:Cargo.Bazel.lock",
        lockfile = "@salus//bazel-locks:cargo-bazel-lock.json",
        packages = {
            "arrayvec": crate.spec(
                version = "0.7.2",
                default_features = False,
            ),
            "const-field-offset": crate.spec(
                version = "0.1.2",
            ),
            "const-oid": crate.spec(
                version = "0.9.0",
                features = ["db"],
            ),
            "der": crate.spec(
                version = "0.6.0",
                features = ["derive", "flagset", "oid"],
            ),
            "digest": crate.spec(
                version = "0.10.3",
                default_features = False,
            ),
            "ed25519": crate.spec(
                version = "1.5.2",
                default_features = False,
                features = ["pkcs8"],
            ),
            "ed25519-dalek": crate.spec(
                version = "1.0.1",
                default_features = False,
                features = ["u64_backend"],
            ),
            "generic-array": crate.spec(version = "0.14.5"),
            "hex": crate.spec(
                version = "0.4.3",
                default_features = False,
            ),
            "hex-literal": crate.spec(version = "0.3.4"),
            "hkdf": crate.spec(version = "0.12.3"),
            "hmac": crate.spec(version = "0.12.1"),
            "spki": crate.spec(version = "0.6.0"),
            "typenum": crate.spec(version = "1.15.0"),
            "enum_dispatch": crate.spec(
                version = "0.3.8",
            ),
            "flagset": crate.spec(
                version = "0.4.3",
            ),
            "memoffset": crate.spec(
                version = ">=0.6.5",
                features = ["unstable_const"],
            ),
            "riscv-decode": crate.spec(
                version = "0.2",
            ),
            "signature": crate.spec(version = "1.6.4", default_features = False),
            "spin": crate.spec(
                version = "0.9",
                default_features = False,
                features = ["once", "rwlock", "spin_mutex"],
            ),
            "static_assertions": crate.spec(
                version = "1.1",
            ),
            "tock-registers": crate.spec(
                version = "0.7",
            ),
            "fdt-rs": crate.spec(
                git = "https://github.com/rivosinc/fdt-rs",
                default_features = False,
                rev = "4aa8151",
            ),
            "sha2": crate.spec(version = "0.10", default_features = False),
            "seq-macro": crate.spec(version = "0.3.2"),
            "zeroize": crate.spec(version = "1.5.7"),
        },
    )
