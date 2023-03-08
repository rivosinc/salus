<!--
SPDX-FileCopyrightText: 2023 Rivos Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Using Visual Studio Code with Bazel

These instructions assume familarity with the use of Rust with Visual Studio
Code in general and so include Salus specific configuration.

## Configure Rust Analyzer

This provides for definition lookups, navigation, etc:

Add the following JSON to `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Generate rust-project.json",
            "command": "bazel",
            "args": [
                "run",
                "@rules_rust//tools/rust_analyzer:gen_rust_project"
            ],
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "group": "build",
            "problemMatcher": [],
            "presentation": {
                "reveal": "never",
                "panel": "dedicated",
            },
            "runOptions": {
                "runOn": "folderOpen"
            }
        },
    ]
}
```

## Override check tool

Since the default `cargo check` behaviour is not available with Bazel a custom
script provides substitute functionality

Add the following to your `.vscode/settings.json`

```json
{
    "rust-analyzer.checkOnSave.overrideCommand": [
        "scripts/rust-analyzer-check.sh"
    ]
}
```
