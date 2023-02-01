# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0

def _lds_rule_impl(ctx):
    out = ctx.actions.declare_file("salus.lds")
    ctx.actions.expand_template(
        output = out,
        template = ctx.file.template,
        substitutions = {"{LVL}": ctx.var["COMPILATION_MODE"]},
    )
    return [DefaultInfo(files = depset([out]))]

lds_rule = rule(
    implementation = _lds_rule_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = [".tmpl"],
            mandatory = True,
        ),
    },
)
