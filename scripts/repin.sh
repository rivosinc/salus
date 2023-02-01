#!/bin/sh
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run twice if needed. Bazel sync sometimes runs into network issues

CARGO_BAZEL_REPIN=1 bazel sync --only=salus-index --only=sbi-index --only=rice-index

if [ $? -ne 0 ]
then
    CARGO_BAZEL_REPIN=1 bazel sync --only=salus-index --only=sbi-index --only=rice-index
fi
