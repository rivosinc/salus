#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# This script can be used to generate valid output (JSON) for Rust Analyzer

bazel build "//:clippy-all" --@rules_rust//:error_format=json 2>&1| grep '{"message"' || echo '{}'
