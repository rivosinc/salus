#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

riscv64-unknown-elf-gdb bazel-bin/src/salus --ex "target remote localhost:1234"
