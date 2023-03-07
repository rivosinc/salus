# SPDX-FileCopyrightText: 2023 Rivos Inc.
#
# SPDX-License-Identifier: Apache-2.0


# Do not add dependencies to the bazel targets here.
# The dependencies are handled by bazel.
all: update_submodules
	bazel build //:salus-all

check: update_submodules
	bazel test //:test-all

salus: update_submodules
	bazel build //:salus-all

salus_debug: update_submodules
	bazel build -c dbg //:salus-all

salus_test: update_submodules
	bazel build //:salus-unit-tests
	scripts/run_salus_test.sh

tellus_bin: update_submodules
	bazel build //test-workloads:tellus_raw

guestvm: update_submodules
	bazel build //test-workloads:guestvm_raw

tellus: update_submodules
	bazel build //test-workloads:tellus

umode: update_submodules
	bazel build //umode:

update_submodules:
	# Check if the submodule needs to be initialized
	if [ "$$(git submodule status | grep -c '^-')" -gt 0 ]; then \
		git submodule update --init; \
	fi

run_salus_test: salus_test
	./scripts/run_salus_test.sh

run_tellus_gdb: tellus_bin salus_debug
	./scripts/run/tellus_gdb.sh

run_tellus: all
	./scripts/run_tellus.sh


run_linux: salus
	./scripts/run_linux.sh

run_debian: salus
	./scripts/run_debian.sh

run_buildroot: salus
	./scripts/run_buildroot.sh

run_buildroot_debug: salus_debug
	./scripts/run_buildroot_dbg.sh

lint:
	bazel build //:clippy-all

format:
	bazel test //:rustfmt-all

reformat:
	bazel run @rules_rust//:rustfmt

#doc: currently not handled correctly by rules_rust

ci: salus guestvm tellus lint format check # doc
