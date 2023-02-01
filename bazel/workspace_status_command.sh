#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022 by Rivos Inc.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Bazel executes this script when it detects that the workspace has
# changed, and provides the current git commit id to the build. This is
# a "stable" flag because it will only change if the workspace has
# changed.
#
# https://bazel.build/docs/user-manual#workspace-status

echo "STABLE_GIT_COMMIT $(git describe --always --dirty=-modified || echo "unknown")"
