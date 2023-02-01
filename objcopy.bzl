# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

load(
    "@bazel_tools//tools/cpp:toolchain_utils.bzl",
    "find_cpp_toolchain",
    "use_cpp_toolchain",
)

def _objcopy_to_object_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    src = ctx.files.src[0]
    out = ctx.outputs.out

    command_line = ["-I", "binary", "-O", "elf64-littleriscv", src.path, out.path]

    ctx.actions.run(
        mnemonic = "ObjCopyElfToBinary",
        executable = cc_toolchain.objcopy_executable,
        arguments = command_line,
        inputs = depset(
            [src],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [out],
    )

    return [DefaultInfo(files = depset([out]))]

objcopy_to_object = rule(
    implementation = _objcopy_to_object_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "out": attr.output(),
    },
    toolchains = use_cpp_toolchain(),
)

def _objcopy_to_bin_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    src = ctx.files.src[0]
    out = ctx.actions.declare_file(ctx.label.name + ".out")

    command_line = ["-O", "binary", src.path, out.path]

    ctx.actions.run(
        mnemonic = "ObjCopyToBinary",
        executable = cc_toolchain.objcopy_executable,
        arguments = command_line,
        inputs = depset(
            [src],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [out],
    )

    return [DefaultInfo(files = depset([out]))]

objcopy_to_bin = rule(
    implementation = _objcopy_to_bin_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    toolchains = use_cpp_toolchain(),
)
