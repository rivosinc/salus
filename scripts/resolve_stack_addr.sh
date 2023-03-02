#!/bin/bash
# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

# This script is used to resolve addresses from a stack dump to
# symbolic names. It can be used two ways:
#
# 1. With a single argument it will resolve the address represented by the argument
#
# > ./scripts/resolve_stack_addr.sh 0x00000000802d7a2a
#   const_oid::arcs::Arcs::try_next at external/rice-index__const-oid-0.9.1/src/arcs.rs:51
#
# 2. with no arguments it will read from stdin, which allows you to cut and paste a full stack
#
# > ./scripts/resolve_stack_addr.sh
#   0x000000008034757a
#   0x00000000802fc6be
#   0x00000000802f3b72
# ^D <-- type a literal control-D when done to specify EOF
#   <core::array::iter::IntoIter<T,_> at /rustc/1db9c061df216e2da/library/core/src/array/iter.rs:249
#   <core::array::iter::IntoIter<T,_> as core::next::{{closure}} at /rustc/116e2da/library/core/src/array/iter.rs:249
#   hyp_alloc::arena::Arena<T,A>::ids::{{closure}} at hyp-alloc/src/arena.rs:118

. scripts/common_variables

if [ x$1 == x ]; then

    stack=$(</dev/stdin)

    for s in $stack
    do
        addr2line -fiCpe ${SALUS_BINS}salus $s
    done

else
    addr2line -fiCpe ${SALUS_BINS}salus $1
fi
