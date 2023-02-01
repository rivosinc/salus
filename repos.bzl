# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

load("@salus-index//:defs.bzl", cr_index = "crate_repositories")
load("@rice-index//:defs.bzl", cr_rice = "crate_repositories")
load("@sbi-index//:defs.bzl", cr_sbi = "crate_repositories")

def salus_repositories():
    cr_index()
    cr_rice()
    cr_sbi()
