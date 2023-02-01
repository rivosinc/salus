#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

. scripts/common_variables

${QEMU_BIN} \
    ${MACH_ARGS} \
    -kernel ${SALUS_BINS}salus \
    -device guest-loader,kernel=${LINUX_BIN},addr=${KERNEL_ADDR} \
    -append "${BOOTARGS} root=LABEL=rootfs" \
    -device guest-loader,initrd=${INITRD_IMAGE},addr=${INITRD_ADDR} \
    -drive file=${ROOTFS_IMAGE},if=none,id=hd \
    ${NVME_DEVICE_ARGS} \
    ${IOMMU_ARGS} \
    ${NETWORK_ARGS} \
    ${EXTRA_QEMU_ARGS}
