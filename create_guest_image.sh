#!/bin/bash

# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -e
trap EXIT
# This is 32 4K pages
MAX_TELLUS_SIZE=131072
TELLUS_PATH=${1}
GUESTVM_PATH=${2}
OUTPUT_PATH=${3}

if [[ "$OSTYPE" == "darwin"* ]]; then
    TELLUS_SIZE=$( (stat -f "%z" "${TELLUS_PATH}" | xargs) )
else
    TELLUS_SIZE=$( (stat -c "%s" "${TELLUS_PATH}" | xargs) )
fi

if [ "${TELLUS_SIZE}" -gt "${MAX_TELLUS_SIZE}" ]; then
	echo "The binary $TELLUS_PATH exceeds the max size (${MAX_TELLUS_SIZE})"
	exit 1
fi

ZERO_PAD_SIZE=$((${MAX_TELLUS_SIZE} - ${TELLUS_SIZE}))
cp "${TELLUS_PATH}" "${OUTPUT_PATH}"
dd status=none if=/dev/zero seek="${TELLUS_SIZE}" of="${OUTPUT_PATH}" bs=1 count=${ZERO_PAD_SIZE}
cat ${GUESTVM_PATH} >> ${OUTPUT_PATH}
