#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

. scripts/common_variables

set -x

${GDB} ${SALUS_BINS}salus ${GDB_ARGS} --ex "target remote localhost:1234"
