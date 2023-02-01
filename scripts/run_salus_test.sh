#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

. scripts/common_variables

${QEMU_BIN} \
    ${MACH_ARGS} \
    -kernel bazel-bin/test-*/salus-unit-tests
    ${EXTRA_QEMU_ARGS}
