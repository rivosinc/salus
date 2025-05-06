#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

. scripts/common_variables

${QEMU_BIN} \
    ${MACH_ARGS} \
    -kernel ${SALUS_BINS}salus \
    -device guest-loader,kernel=${TELLUS_BINS}tellus_guestvm,addr=${KERNEL_ADDR} \
    ${IOMMU_ARGS} \
    ${EXTRA_QEMU_ARGS}
