# HAL Project Makefile
#
# This Makefile implements workflows from the Au-Zone Software Process Specification:
#   - Code formatting with nightly Rust and ruff for Python
#   - Linting with clippy and ruff
#   - Testing with cargo-nextest and llvm-cov (using profiling profile)
#   - SBOM generation and license policy validation
#   - Pre-release quality checks and version verification
#
# IMPORTANT: Since HAL is performance-focused, tests use the 'profiling' profile
# which produces optimized binaries with full debug symbols.
#
# Prerequisites: See README.md or CONTRIBUTING.md for tool installation.
#
# ===========================================================================

# Use bash for shell commands (required for source <(...) syntax)
SHELL := /bin/bash

# Project configuration
PROJECT_NAME := hal
# Five extension modules (tensor/codec/image/decoder/tracker) since 0.29 (the single crates/python
# and the edgefirst-hal meta-crate are both gone).
PYTHON_PACKAGES := tensor codec image decoder tracker
PYTHON_CRATES := $(addprefix crates/python-,$(PYTHON_PACKAGES))
# cargo package names, for --exclude: these are cdylibs with no meaningful
# line coverage of their own, and llvm-cov cannot instrument them usefully.
PYTHON_CRATE_NAMES := edgefirst-python-common \
	$(addprefix edgefirst-python-,$(PYTHON_PACKAGES))
TEST_DIR := tests

# The five modular C-API leaves: standalone packages excluded from the
# workspace (static/dynamic feature conflict, plans R3-R5). Every invocation
# MUST carry --target-dir target: their artifact-reading tests hardcode the
# shared <root>/target dir, and EF_REQUIRE_FRESH_ARTIFACTS=1 makes a wrong
# target dir a hard failure instead of a silent skip.
CAPI_MODULAR_CRATES := tensor-capi image-capi codec-capi decoder-capi tracker-capi

# Use profiling profile for testing (optimized with debug symbols)
CARGO_PROFILE := --profile profiling
# For cargo-llvm-cov nextest, the cargo profile flag is different
LLVM_COV_PROFILE := --cargo-profile profiling

# Rust features (all except opencv, which requires libclang at build time)
RUST_FEATURES := --features opengl,ndarray

# Python interpreter name for maturin -i (cross-compile requires a versioned
# name like 'python3.10' because maturin parses major.minor from the filename
# — it cannot execute a target-arch Python to introspect its ABI).
PYTHON_INTERPRETER := $(shell \
	if [ -x venv/bin/python ]; then \
		venv/bin/python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"; \
	elif command -v python3 >/dev/null 2>&1; then \
		python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"; \
	fi)

# Target triple for cross-compilation. Empty value means a native build.
#   make wheel                                    # native wheel
#   make TARGET=aarch64-unknown-linux-gnu wheel   # cross-compile via zig
TARGET ?=

# Optional abi3 stable-ABI version (py38, py311, ...). When set, selects the
# matching 'abi3-<PYABI>' feature in crates/python/Cargo.toml and produces a
# version-agnostic wheel (tagged cp<abi>-abi3) usable by any CPython at or
# above that minimum. Empty value produces a wheel tagged for the interpreter
# chosen by -i (e.g. cp310-cp310).
#   make PYABI=py38 wheel                         # abi3 wheel, Python 3.8+
#   make TARGET=aarch64-unknown-linux-gnu PYABI=py311 wheel
PYABI ?=

# ===========================================================================
# STANDARD TARGETS
# ===========================================================================

.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "    make format         - Format all source code (Rust nightly + Python ruff)"
	@echo "    make lint           - Run all linters (clippy + ruff)"
	@echo "    make check          - Run cargo check (fast compilation check)"
	@echo ""
	@echo "  Building & Testing:"
	@echo "    make build          - Build with coverage instrumentation (profiling profile)"
	@echo "    make wheel          - Build Python wheel (release, zig + manylinux2014)"
	@echo "                          Options:"
	@echo "                            TARGET=<triple>  cross-compile (e.g. aarch64-unknown-linux-gnu)"
	@echo "                            PYABI=py38|py311 produce an abi3 wheel (stable ABI)"
	@echo "    make test           - Run all tests with coverage"
	@echo "    make test-rust      - Run Rust tests only"
	@echo "    make test-python    - Run Python tests only"
	@echo "    make test-ontarget  - Run the suite on SSH hosts you supply"
	@echo "                          TARGETS='host1 host2' (required)"
	@echo "    make bench          - Run benchmarks"
	@echo ""
	@echo "  Quality & Release:"
	@echo "    make sbom           - Generate SBOM and check license policy"
	@echo "    make verify-version - Verify version consistency across files"
	@echo "    make pre-release    - Run all pre-release checks"
	@echo "    make clean          - Remove build artifacts"
	@echo ""
	@echo "Prerequisites: See README.md or CONTRIBUTING.md for tool installation."

# ===========================================================================
# FORMATTING
# ===========================================================================

.PHONY: format
format: format-rust format-python format-capi-modular
	@echo "✓ All formatting complete"

.PHONY: format-rust
format-rust:
	@echo "Formatting Rust code..."
	@cargo fmt --all
	@echo "✓ Rust formatting complete"

.PHONY: format-python
format-python:
	@echo "Formatting Python code with ruff..."
	@if [ -f "venv/bin/ruff" ]; then \
		. venv/bin/activate && ruff format $(PYTHON_CRATES) $(TEST_DIR); \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff format $(PYTHON_CRATES) $(TEST_DIR); \
	else \
		echo "Warning: ruff not found (see README.md for installation)"; \
	fi
	@echo "✓ Python formatting complete"

.PHONY: format-capi-modular
format-capi-modular:
	@echo "Formatting the five modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		cargo fmt --manifest-path crates/$$c/Cargo.toml || exit 1; \
	done
	@echo "✓ Modular C-API formatting complete"

# ===========================================================================
# LINTING
# ===========================================================================

.PHONY: lint
lint: lint-rust lint-tensor-dynamic lint-python lint-capi-modular
	@echo "✓ All linting complete"

.PHONY: lint-rust
lint-rust:
	@echo "Running clippy (strict mode)..."
	@cargo clippy --workspace --all-targets $(RUST_FEATURES) \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c)) \
		-- -D warnings
	@echo "✓ Clippy passed"

.PHONY: lint-python
lint-python:
	@echo "Running ruff linter..."
	@if [ -f "venv/bin/ruff" ]; then \
		. venv/bin/activate && ruff check $(PYTHON_CRATES) $(TEST_DIR); \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check $(PYTHON_CRATES) $(TEST_DIR); \
	else \
		echo "Warning: ruff not found (see README.md for installation)"; \
	fi
	@echo "✓ Python linting complete"

.PHONY: lint-capi-modular
lint-capi-modular:
	@echo "Running clippy on the five modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		cargo clippy --manifest-path crates/$$c/Cargo.toml --all-targets --target-dir target -- -D warnings || exit 1; \
	done
	@echo "✓ Modular C-API clippy passed"

# ===========================================================================
# BUILDING
# ===========================================================================

.PHONY: check
check: check-capi-modular
	@echo "Running cargo check..."
	@cargo check $(RUST_FEATURES) --workspace \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c))
	@echo "✓ Check passed"

.PHONY: check-capi-modular
check-capi-modular:
	@echo "Running cargo check on the five modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		cargo check --manifest-path crates/$$c/Cargo.toml --target-dir target || exit 1; \
	done
	@echo "✓ Modular C-API check passed"

.PHONY: build
build: capi-libs
	@echo "Building with coverage instrumentation (profiling profile)..."
	@if ! cargo llvm-cov --version >/dev/null 2>&1; then \
		echo "ERROR: cargo-llvm-cov not installed (see README.md for installation)"; \
		exit 1; \
	fi
	@source <(cargo llvm-cov show-env --export-prefix) && \
		cargo build $(CARGO_PROFILE) $(RUST_FEATURES) --workspace \
			$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c))
	@echo "✓ Build complete"

.PHONY: capi-libs
capi-libs:
	@echo "Building the five modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		cargo build --manifest-path crates/$$c/Cargo.toml --target-dir target || exit 1; \
	done
	@echo "✓ Modular C-API build complete"
	@$(MAKE) capi-symlinks

# G5 (scripts/check-single-home.sh) measures footprint against
# target/release -- BASELINE_BYTES is a release measurement, and a debug
# comparison would be meaningless -- while capi-libs above only ever
# produces target/debug. This is that release build, kept separate so a
# plain `make capi-libs` (used by test-capi-modular and friends) stays fast.
.PHONY: capi-libs-release
capi-libs-release:
	@echo "Building the five modular C-API crates (release, for G5 footprint)..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		cargo build --release --manifest-path crates/$$c/Cargo.toml --target-dir target || exit 1; \
	done
	@echo "✓ Modular C-API release build complete"
	@$(MAKE) capi-symlinks

# build.rs sets `-soname libedgefirst_X.so.0` / install_name
# `libedgefirst_X.0.dylib`, but cargo only ever writes the unversioned
# name, so no C binary (or Rust test harness) can load one via DT_NEEDED /
# @rpath. Cargo never writes the versioned symlink, so this target creates
# it next to each leaf.
.PHONY: capi-symlinks
capi-symlinks:
	@for l in tensor image codec decoder tracker; do \
		for d in target/debug target/release target/profiling; do \
			[ -f $$d/libedgefirst_$$l.so ] && \
				ln -sf libedgefirst_$$l.so $$d/libedgefirst_$$l.so.0 || true; \
			[ -f $$d/libedgefirst_$$l.dylib ] && \
				ln -sf libedgefirst_$$l.dylib $$d/libedgefirst_$$l.0.dylib || true; \
		done; \
	done

# One relocatable archive: headers, soversioned libraries, pkg-config,
# LICENSE, INSTALL.txt. tar.gz on Linux, zip on Windows and macOS. The same
# script release.yml's build-capi job runs, so local `make package` and CI
# cannot drift.
.PHONY: package
package: capi-libs-release
	@mkdir -p dist
	@./scripts/package-capi.sh --outdir dist
	@echo "✓ C archive in dist/"

.PHONY: build-python
build-python:
	@echo "Building Python bindings..."
	@if command -v maturin >/dev/null 2>&1; then \
		for c in $(PYTHON_CRATES); do \
			maturin build --auditwheel skip -m "$$c/Cargo.toml" $(CARGO_PROFILE) || exit 1; \
		done; \
	else \
		echo "ERROR: maturin not found (see README.md for installation)"; \
		exit 1; \
	fi
	@echo "✓ Python bindings built"

.PHONY: wheel
wheel:
	@if [ ! -x "venv/bin/maturin" ] && ! command -v maturin >/dev/null 2>&1; then \
		echo "ERROR: maturin not found (see README.md for installation)"; \
		exit 1; \
	fi
	@if ! command -v zig >/dev/null 2>&1; then \
		echo "ERROR: zig not found (required for --zig manylinux2014 compliance)"; \
		exit 1; \
	fi
	@if [ -z "$(PYTHON_INTERPRETER)" ]; then \
		echo "ERROR: could not detect Python interpreter (expected venv/bin/python or python3 on PATH)"; \
		exit 1; \
	fi
	@echo "Building Python wheels (release, manylinux2014)..."
	@echo "  packages: $(PYTHON_PACKAGES)"
	@echo "  target:   $(if $(TARGET),$(TARGET),native)"
	@echo "  abi:      $(if $(PYABI),abi3-$(PYABI),$(PYTHON_INTERPRETER))"
	@rm -rf target/wheels
	@for c in $(PYTHON_CRATES); do \
		if [ -f "venv/bin/activate" ]; then \
			. venv/bin/activate && \
				maturin build --release --zig --compatibility manylinux2014 --auditwheel skip \
					$(if $(TARGET),--target $(TARGET)) \
					-m "$$c/Cargo.toml" -i $(PYTHON_INTERPRETER) \
					$(if $(PYABI),--features abi3-$(PYABI)) || exit 1; \
		else \
			maturin build --release --zig --compatibility manylinux2014 --auditwheel skip \
				$(if $(TARGET),--target $(TARGET)) \
				-m "$$c/Cargo.toml" -i $(PYTHON_INTERPRETER) \
				$(if $(PYABI),--features abi3-$(PYABI)) || exit 1; \
		fi; \
	done
	@python3 scripts/check_wheel_layout.py target/wheels
	@echo "✓ Wheel built in target/wheels/"

# ===========================================================================
# TESTING
# ===========================================================================

.PHONY: test
test: test-rust test-tensor-dynamic test-python test-capi-modular test-capi-link test-two-library-user
	@echo ""
	@echo "=================================================="
	@echo "✓ All tests passed"
	@echo "=================================================="

.PHONY: test-rust
test-rust:
	@echo "Running Rust tests with coverage (profiling profile)..."
	@if ! cargo nextest --version >/dev/null 2>&1; then \
		echo "ERROR: cargo-nextest not installed (see README.md for installation)"; \
		exit 1; \
	fi
	@if ! cargo llvm-cov --version >/dev/null 2>&1; then \
		echo "ERROR: cargo-llvm-cov not installed (see README.md for installation)"; \
		exit 1; \
	fi
	@cargo llvm-cov nextest $(LLVM_COV_PROFILE) $(RUST_FEATURES) --workspace \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c)) \
		--lcov --output-path target/rust-coverage.lcov -j 1
	@echo "✓ Rust tests passed"
	@echo "Coverage report: target/rust-coverage.lcov"
	@$(MAKE) --no-print-directory test-doc

# cargo-nextest does not run doctests, so `make test` never compiled a single
# `///` example. A private `use` that made a public type unnameable shipped
# green through 1474 nextest cases and was caught only by running these by
# hand. The PyO3 binding crates are excluded on purpose: their doc comments
# are Python `__doc__` in reStructuredText, not Rust.
.PHONY: test-doc
test-doc:
	@echo "Running Rust doctests (nextest does not)..."
	@cargo test --workspace --doc $(RUST_FEATURES) \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c)) \
		--exclude edgefirst-python-common
	@echo "✓ Doctests passed"

.PHONY: test-python
test-python:
	@echo "Running Python tests..."
	@echo "  Installing Python bindings..."
	@# --no-deps is required, not an optimisation. Every edgefirst-* sibling is
	@# being built from THIS tree in this same loop, so the `~=` pins in each
	@# pyproject.toml -- which describe what a *published* wheel needs from PyPI
	@# -- must not be resolved. Resolving them either fails outright (the version
	@# under development is unpublished, as any unreleased version is) or, worse,
	@# silently installs a published sibling over the locally built one, so the
	@# suite tests released code instead of the branch. numpy is the only
	@# third-party runtime dependency, so it is installed explicitly.
	@if [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && pip install -q numpy || exit 1; \
	else \
		pip install -q numpy || exit 1; \
	fi
	@for c in $(PYTHON_CRATES); do \
		if [ -f "venv/bin/activate" ]; then \
			. venv/bin/activate && pip install -q --no-deps --force-reinstall "$$c/" || exit 1; \
		else \
			pip install -q --no-deps --force-reinstall "$$c/" || exit 1; \
		fi; \
	done
	@echo "  Running tests..."
	@if [ -f "venv/bin/slipcover" ]; then \
		. venv/bin/activate && \
			python -m slipcover --xml --out target/python-coverage.xml \
				-m pytest $(TEST_DIR); \
	elif command -v slipcover >/dev/null 2>&1; then \
		python -m slipcover --xml --out target/python-coverage.xml \
			-m pytest $(TEST_DIR); \
	elif [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && \
			python -m pytest $(TEST_DIR); \
	else \
		python -m pytest $(TEST_DIR); \
	fi
	@echo "✓ Python tests passed"

# The five modular C-API leaves (tensor/image/codec/decoder/tracker-capi).
# Each leaf is a standalone package excluded from the workspace, so it is
# built and tested via --manifest-path/--target-dir rather than picked up
# by the workspace-wide test-rust run.
#
# Coverage note, made consciously: these five leave `cargo llvm-cov nextest
# --workspace`'s lcov (test-rust above). Their tests still RUN here (plain
# `cargo test`); doctests do not apply -- each leaf's crate-type is
# staticlib/cdylib only (no rlib), so cargo never compiles or runs doctests
# for it, on any invocation. Nothing was lost: the old workspace `test-doc`
# run skipped these crates for the identical reason. Only their lines leave
# the coverage denominator.
.PHONY: test-capi-modular
test-capi-modular: capi-libs
	@echo "Running tests for the five modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		EF_REQUIRE_FRESH_ARTIFACTS=1 cargo test --manifest-path crates/$$c/Cargo.toml --target-dir target -- --test-threads=1 || exit 1; \
	done
	@echo "✓ Modular C-API tests passed"

# `test-rust` above runs the STATIC backend only, and cannot also run the
# dynamic one: `static` is edgefirst-tensor's default feature and the two
# backends are mutually exclusive (the compile_error! guard at
# crates/tensor/src/lib.rs), so `--workspace` can only ever select one. The
# dynamic backend therefore needs a build of its own, and until this lane
# existed it had none -- every gate in `dynamic_primitives.rs`, and the
# whole `protocol_roundtrip.rs` suite under `dynamic`, ran only when someone
# typed the command by hand. On a branch whose standard is "demonstrated red
# before it is trusted green", a gate nothing re-runs decays to no gate.
#
# `capi-libs` first, and not incidentally: `edgefirst-tensor-ffi` declares
# its symbols with no `#[link]` attribute, so `dynamic-test-link` (opt-in,
# never enabled by a production consumer) points build.rs at target/debug
# for the real `libedgefirst_tensor.so`. A stale or missing one is a link
# error, not a silent pass.
#
# Targets are named one by one rather than via `--all-targets`, and the
# reason is worth stating so nobody "simplifies" this later: the crate's own
# unit tests (`--lib`) are static-only and do not compile under `dynamic` --
# a pre-existing gap this lane does not close.
#
# `vocabulary` was excluded by name here when this lane was written, because
# `ef_tensor_builder_alloc` collapsed every refusal to ENOMEM and its
# `a_defined_but_unbacked_code_errors_instead_of_panicking` could not pass.
# That was left as a marker rather than a silent omission, and task P2d
# removed the defect -- so the exclusion is gone with it, which was that
# task's acceptance test.
#
# `logical_shape` (task P2e) is deliberately NOT cfg-gated to either
# backend: it pins a contract both must honour, and the defect it caught was
# identical on both -- which is exactly the kind G13 cannot see, since G13
# compares the two against each other.
DYNAMIC_TEST_TARGETS := dynamic_primitives protocol_roundtrip map_access \
	identity scenarios no_global_state vocabulary logical_shape

.PHONY: test-tensor-dynamic
test-tensor-dynamic: capi-libs
	@echo "Running edgefirst-tensor's dynamic-backend tests..."
	@for t in $(DYNAMIC_TEST_TARGETS); do \
		LD_LIBRARY_PATH=$(CURDIR)/target/debug cargo test -p edgefirst-tensor \
			--no-default-features --features dynamic,dynamic-test-link,ndarray \
			--test $$t || exit 1; \
	done
	@echo "✓ Dynamic-backend tests passed"

# The dynamic backend's own clippy run. `lint-rust` above is `--workspace
# --all-targets` on default features, i.e. static, so nothing lints this
# code either. `--all-targets` is not usable here for the same reason the
# test lane names its targets: the static-only `--lib` unit tests do not
# compile under `dynamic`.
.PHONY: lint-tensor-dynamic
lint-tensor-dynamic:
	@echo "Running clippy on the dynamic backend (strict mode)..."
	@cargo clippy -p edgefirst-tensor --no-default-features \
		--features dynamic,ndarray --lib -- -D warnings
	@for t in $(DYNAMIC_TEST_TARGETS); do \
		cargo clippy -p edgefirst-tensor --no-default-features \
			--features dynamic,dynamic-test-link,ndarray --test $$t -- -D warnings \
			|| exit 1; \
	done
	@echo "✓ Dynamic-backend clippy passed"

# Compiles every shipped header as C11 and C++17 under -Werror, then links
# AND RUNS one consumer binary per modular C-API library. `-fsyntax-only`
# (used elsewhere) proves a header parses; this is the only lane that proves
# a library actually loads and its symbols resolve at runtime -- see
# scripts/check-headers.sh for why that gap mattered.
.PHONY: test-capi-link
test-capi-link: capi-libs
	@./scripts/check-headers.sh

# The Definition of Done for the single-tensor-home plan, as a command. See
# scripts/check-single-home.sh for what each gate measures and why.
.PHONY: check-single-home
check-single-home: capi-libs capi-libs-release
	@./scripts/check-single-home.sh

# ===========================================================================
# DIFFERENTIAL TESTING (G13)
# ===========================================================================
#
# G13 (scripts/check-single-home.sh --differential-only) proves the two
# tensor backends agree test-for-test, not merely that both compile -- see
# that gate's own comment for why a compile-only or method-parity check
# would have caught none of the four real behavioural divergences task P2
# found (a missing reshape primitive, `map` refusing PBO-backed tensors,
# `clone_fd` refusing SHM-backed ones, `create_image(dtype="int8")`
# reporting `uint8`). It needs one venv with each backend installed,
# supplied by PY_STATIC_VENV/PY_DYNAMIC_VENV -- building those two venvs is
# packaging work, done here rather than by hand, so G13 can run in CI.
#
# Both venvs build from the SAME commit (whatever HEAD is when this runs),
# not a historical checkout: the four Python extensions carry `static`/
# `dynamic` forwarding features (this task, mirroring edgefirst-tracker/
# -image/-codec/-decoder and python-common), so `--no-default-features
# --features static` reaches a genuine static-linked build without a
# second checkout. A first design pinned the static side to the last
# all-static commit in a scratch worktree; reviewed and rejected before
# landing, because a fixed historical pin decays as the branch grows --
# every legitimate change to shared code after the pin becomes a false
# divergence, making G13 noisier over time instead of staying meaningful.
# Building both sides from identical source removes that decay entirely.
DIFFERENTIAL_STATIC_VENV := $(CURDIR)/target/differential-static-venv
DIFFERENTIAL_DYNAMIC_VENV := $(CURDIR)/target/differential-dynamic-venv
# Matches venv's own pinned versions (`pip freeze`), not "latest" -- G13's
# own guard flags divergent optional-package availability between the two
# venvs as `cannot_measure` rather than let it read as a false backend
# divergence, but mismatched versions of the SAME package could still shift
# behaviour underneath the comparison without tripping that guard. The full
# list the suite actually reaches for (confirmed against tests/, not
# guessed): pyyaml/safetensors/psutil were missing from an earlier version
# of this list, caught by p2a-dynamic-surface (G13's own author) before
# this lane ran for real -- would have produced import-skip cannot_measure
# on both venvs identically, silently narrowing what G13 actually compares.
DIFFERENTIAL_TEST_DEPS := numpy==2.4.6 opencv-python-headless==5.0.0.93 pillow==12.2.0 \
	pytest==9.0.3 pyyaml==6.0.3 safetensors==0.8.0 psutil==7.2.2

# NOT a dedicated --target-dir per venv, unlike the wheel output directory
# below. Flipping edgefirst-tensor's static/dynamic feature rebuilds the
# whole dependency graph, so `venv-static` then `venv-dynamic` in the same
# `test-differential` run will thrash the shared target/ between the two
# builds either way -- a separate target-dir per side would trade that
# rebuild for a second, persistent full build cache instead, and this host
# is at 97% disk. Considered and rejected for now, not overlooked; revisit
# if this lane's own build time becomes the bottleneck rather than disk.
# A DEDICATED wheel output directory per venv (`-o`), never the shared
# target/wheels/ `make wheel`/`make build-python` use -- reusing that one
# is exactly how a stale wheel from unrelated manual testing (an aarch64
# cross-build left over from verifying task P2's own work) got installed
# into an x86_64 venv instead of the freshly-built native one. Always
# cleared before building into it, so an install glob over its contents can
# never see anything but this run's own output. Each crate's own build.rs
# carries the matching half of this discipline for its OWN generated
# artifacts: a static build actively removes any stale libedgefirst_tensor.
# so* a previous dynamic build in the same tree left behind, rather than
# just skipping regeneration -- proven necessary, not assumed, the same
# failure shape recurring one layer down (see crates/python-tensor/build.rs).
#
# Isolation (a dedicated, always-cleared directory) closes THIS instance of
# that bug; it does not close the class. Isolation fails silently if
# anything ever writes the wrong architecture into the right directory --
# a `--target` added later, a cached artifact, a stray copy step. So this
# also asserts the actual architecture of every .so this run is about to
# install, not just where it came from: a wheel built for the wrong host
# cannot import, and "cannot import" looks exactly like "imports and
# misbehaves" until someone tries it -- the whole reason G13 exists. A
# build-time assertion names the mismatched file directly; G13 itself,
# even hardened, can only ever read the resulting failure as
# `cannot_measure`, never say which artifact was wrong.
define DIFFERENTIAL_ASSERT_ARCH
	host_arch=$$(uname -m); \
	case "$$host_arch" in \
		x86_64) want='x86-64' ;; \
		aarch64) want='ARM aarch64' ;; \
		*) echo "ERROR: $(1) -- unrecognized host arch $$host_arch, add it to DIFFERENTIAL_ASSERT_ARCH" >&2; exit 1 ;; \
	esac; \
	for w in $(1)/*.whl; do \
		for so in $$(unzip -Z1 "$$w" | grep -E '\.so(\.[0-9]+)?$$'); do \
			got=$$(unzip -p "$$w" "$$so" | file - | grep -oE 'x86-64|ARM aarch64'); \
			if [ "$$got" != "$$want" ]; then \
				echo "ERROR: $$w:$$so is $$got, host is $$host_arch (wants $$want) -- installing this would silently fail to import" >&2; \
				exit 1; \
			fi; \
		done; \
	done
endef

# Architecture is not the only claim a wheel can silently fail to keep --
# three stale-artifact bugs surfaced in this same task (the ARM library,
# a stale libedgefirst_tensor.so* bundled into a subsequent static wheel by
# build.rs, and a stale venv compared as if it were current), and the
# common thread is stale output surviving into a fresh-looking artifact.
# So this also asserts the actual BACKEND matches what each side claims,
# not just that the architecture is right: a "static" wheel could still
# link dynamically (a feature flag not threaded through correctly) and
# pass the architecture check while comparing dynamic against itself.
#
# Dynamic: reuses scripts/check_wheel_layout.py's own A3 check (already
# verified against both directions when it was written) rather than a
# second, divergent copy of the same rule -- the tensor wheel must carry
# exactly one libedgefirst_tensor.so, the other three must carry none.
#
# Static: the inverse of dynamic's claim, and simpler -- NOTHING in this
# set should carry libedgefirst_tensor.so at all, since a genuinely
# static build embeds the implementation and needs nothing external.
define DIFFERENTIAL_ASSERT_STATIC_SELF_CONTAINED
	for w in $(1)/*.whl; do \
		found=$$(unzip -Z1 "$$w" | grep -c 'libedgefirst_tensor' || true); \
		if [ "$$found" -ne 0 ]; then \
			echo "ERROR: $$w carries libedgefirst_tensor -- this is the STATIC set, it should embed the implementation and link nothing external" >&2; \
			exit 1; \
		fi; \
	done
endef

# `mkdir -p` after each `rm -rf`, not left to maturin's own `-o` handling:
# observed once, on a clean tree, that `venv-dynamic`'s first wheel build
# failed with "failed to create file .../edgefirst_tensor-...whl: No such
# file or directory" while the otherwise-identical venv-static run in the
# same session did not. Not reproduced deterministically after several
# further runs, including immediately re-running venv-dynamic alone right
# after the failure -- so the exact trigger (a maturin timing assumption
# about who creates -o's directory, some interaction with the nested
# tensor-capi build only the dynamic path runs, or something else) is not
# confirmed. Explicit creation removes the dependency on that assumption
# regardless of which it turns out to be, and costs nothing to keep.
.PHONY: venv-static
venv-static:
	@echo "Building the static-backend comparison venv..."
	@rm -rf $(DIFFERENTIAL_STATIC_VENV) target/differential-wheels-static
	@mkdir -p target/differential-wheels-static
	@venv/bin/python -m venv $(DIFFERENTIAL_STATIC_VENV)
	@$(DIFFERENTIAL_STATIC_VENV)/bin/pip install -q --upgrade pip
	@$(DIFFERENTIAL_STATIC_VENV)/bin/pip install -q maturin $(DIFFERENTIAL_TEST_DEPS)
	@for c in $(PYTHON_CRATES); do \
		$(DIFFERENTIAL_STATIC_VENV)/bin/maturin build --release --auditwheel skip \
			--no-default-features --features static,tracing \
			-o target/differential-wheels-static \
			-m "$$c/Cargo.toml" -i $(DIFFERENTIAL_STATIC_VENV)/bin/python || exit 1; \
	done
	@$(call DIFFERENTIAL_ASSERT_ARCH,target/differential-wheels-static)
	@$(call DIFFERENTIAL_ASSERT_STATIC_SELF_CONTAINED,target/differential-wheels-static)
	@$(DIFFERENTIAL_STATIC_VENV)/bin/pip install -q --no-deps \
		target/differential-wheels-static/*.whl
	@echo "✓ Static-backend comparison venv ready: $(DIFFERENTIAL_STATIC_VENV)"

.PHONY: venv-dynamic
venv-dynamic:
	@echo "Building the dynamic-backend comparison venv..."
	@rm -rf $(DIFFERENTIAL_DYNAMIC_VENV) target/differential-wheels-dynamic
	@mkdir -p target/differential-wheels-dynamic
	@venv/bin/python -m venv $(DIFFERENTIAL_DYNAMIC_VENV)
	@$(DIFFERENTIAL_DYNAMIC_VENV)/bin/pip install -q --upgrade pip
	@$(DIFFERENTIAL_DYNAMIC_VENV)/bin/pip install -q maturin $(DIFFERENTIAL_TEST_DEPS)
	@for c in $(PYTHON_CRATES); do \
		$(DIFFERENTIAL_DYNAMIC_VENV)/bin/maturin build --release --auditwheel skip \
			-o target/differential-wheels-dynamic \
			-m "$$c/Cargo.toml" -i $(DIFFERENTIAL_DYNAMIC_VENV)/bin/python || exit 1; \
	done
	@$(call DIFFERENTIAL_ASSERT_ARCH,target/differential-wheels-dynamic)
	@python3 scripts/check_wheel_layout.py target/differential-wheels-dynamic
	@$(DIFFERENTIAL_DYNAMIC_VENV)/bin/pip install -q --no-deps \
		target/differential-wheels-dynamic/*.whl
	@echo "✓ Dynamic-backend comparison venv ready: $(DIFFERENTIAL_DYNAMIC_VENV)"

.PHONY: test-differential
test-differential: venv-static venv-dynamic
	@echo "Running G13 (static vs. dynamic, test for test)..."
	@PY_STATIC_VENV=$(DIFFERENTIAL_STATIC_VENV) PY_DYNAMIC_VENV=$(DIFFERENTIAL_DYNAMIC_VENV) \
		./scripts/check-single-home.sh --differential-only

# G3: the minimal two-library user. Links only edgefirst_codec and
# edgefirst_tensor (not the other three leaves) and decodes a real JPEG into
# a host-memory tensor. Proves the two-library split is actually usable, not
# just that the libraries happen to build.
.PHONY: test-two-library-user
test-two-library-user: capi-libs
	@cc -std=c11 -Wall -Wextra -Werror -o target/test_two_library_user \
		crates/codec-capi/tests/c/test_two_library_user.c \
		-Icrates/codec-capi/include -Icrates/tensor-capi/include \
		-Ltarget/debug -ledgefirst_codec -ledgefirst_tensor \
		-Wl,-rpath,$(PWD)/target/debug
	@./target/test_two_library_user

# Optional CUDA device-pointer tests (convert -> cuda_map -> cudaMemcpy). Not
# part of `make test`: they need a CUDA-capable GPU + libcudart at runtime.
# On a dev PC with only the driver, install the CUDA runtime into the local
# venv (`pip install nvidia-cuda-runtime-cu12`) and this target points the HAL's
# dlopen at it via LD_LIBRARY_PATH. On Jetson/Orin libcudart is already on the
# system path so the venv lookup is simply skipped. Runtime-skips cleanly when
# no GPU/libcudart is present.
# Deliberately NOT part of `make test`: it needs hardware reachable over SSH.
# CI runs the hardware suite on its own runners; this covers the boards CI does
# not have. Hosts are supplied by the caller -- there is no built-in list.
.PHONY: test-ontarget
test-ontarget:
	@echo "Running the test suite on target hardware..."
	@./scripts/on-target-test.sh $(TARGETS)
	@echo "✓ on-target run complete (see target/on-target-results/)"

.PHONY: test-cuda
test-cuda:
	@echo "Running CUDA device-pointer tests..."
	@CUDART_LIB=$$( \
		if [ -d "$(CURDIR)/venv" ]; then \
			find "$(CURDIR)/venv" -name 'libcudart.so*' -printf '%h\n' 2>/dev/null | head -1; \
		fi); \
	if [ -z "$$CUDART_LIB" ]; then \
		for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/targets/x86_64-linux/lib; do \
			if [ -e "$$d/libcudart.so" ] || [ -e "$$d/libcudart.so.12" ]; then CUDART_LIB="$$d"; break; fi; \
		done; \
	fi; \
	if [ -n "$$CUDART_LIB" ]; then \
		echo "  using libcudart at $$CUDART_LIB"; \
	fi; \
	LD_LIBRARY_PATH="$$CUDART_LIB:$$LD_LIBRARY_PATH" \
		cargo test --features opengl -p edgefirst-image --lib cuda -- --nocapture --test-threads=1
	@echo "✓ CUDA device-pointer tests complete"

.PHONY: bench
bench:
	@echo "Running benchmarks..."
	@cargo bench $(RUST_FEATURES) --workspace \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c))
	@echo "✓ Benchmarks complete"

# On-target nvJPEG decode benchmark (Jetson). Points the loader at the CUDA
# library path (libnvjpeg.so.12 is not on the default loader path) and runs
# only the codec bench; the nvjpeg cells self-skip without CUDA/libnvjpeg.
.PHONY: bench-nvjpeg
bench-nvjpeg:
	@echo "Running nvJPEG benchmark (on-target)..."
	@EDGEFIRST_ENABLE_NVJPEG=1 \
		LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:$$LD_LIBRARY_PATH" \
		cargo bench -p edgefirst-codec --bench codec_benchmark -- --json nvjpeg-bench.json
	@echo "✓ nvJPEG benchmark complete (results in nvjpeg-bench.json)"

# ===========================================================================
# SBOM & LICENSE COMPLIANCE
# ===========================================================================

.PHONY: sbom
sbom:
	@echo "Generating SBOM..."
	@if [ ! -f ".github/scripts/generate_sbom.sh" ]; then \
		echo "ERROR: .github/scripts/generate_sbom.sh not found"; \
		exit 1; \
	fi
	@.github/scripts/generate_sbom.sh
	@echo "Validating SBOM format..."
	@if command -v cyclonedx >/dev/null 2>&1; then \
		cyclonedx validate --input-file sbom.json; \
	else \
		echo "Warning: cyclonedx CLI not found, skipping validation"; \
	fi
	@echo "Checking license policy compliance..."
	@python3 .github/scripts/check_license_policy.py sbom.json
	@if [ -f "NOTICE" ]; then \
		echo "Validating NOTICE file..."; \
		if [ -f ".github/scripts/validate_notice.py" ]; then \
			python3 .github/scripts/validate_notice.py NOTICE sbom.json || \
				echo "⚠️  NOTICE validation failed - may need manual update"; \
		fi; \
	fi
	@echo "✓ SBOM generated and validated"

# ===========================================================================
# VERSION VERIFICATION
# ===========================================================================

.PHONY: verify-version
verify-version:
	@echo "Verifying version consistency..."
	@CARGO_VERSION=$$(grep -A10 '^\[workspace.package\]' Cargo.toml | grep 'version = ' | sed 's/.*version = "\(.*\)"/\1/'); \
	echo "Workspace version: $$CARGO_VERSION"; \
	MM=$$(echo "$$CARGO_VERSION" | cut -d. -f1,2); \
	fail=0; \
	for c in $(PYTHON_CRATES); do \
		[ -f "$$c/pyproject.toml" ] || continue; \
		if grep -q 'dynamic = \["version"\]' "$$c/pyproject.toml"; then \
			echo "  $$c/pyproject.toml: ✓ (dynamic, from Cargo)"; \
		else \
			echo "  $$c/pyproject.toml: ✗ expected dynamic version"; fail=1; \
		fi; \
		pin=$$(grep -o 'edgefirst-tensor ~= [0-9.]*' "$$c/pyproject.toml" || true); \
		if [ -n "$$pin" ]; then \
			case "$$pin" in \
				*"$$MM"*) echo "  $$c sibling pin: ✓ ($$pin)";; \
				*) echo "  $$c sibling pin: ✗ ($$pin vs $$CARGO_VERSION)"; fail=1;; \
			esac; \
		fi; \
	done; \
	[ "$$fail" -eq 0 ] || exit 1

# ===========================================================================
# PRE-RELEASE CHECKS
# ===========================================================================

.PHONY: pre-release
pre-release: format lint verify-version test sbom
	@echo ""
	@echo "=================================================="
	@echo "✓ All pre-release checks passed"
	@echo "=================================================="
	@echo ""
	@CARGO_VERSION=$$(grep -A10 '^\[workspace.package\]' Cargo.toml | grep 'version = ' | sed 's/.*version = "\(.*\)"/\1/'); \
	echo "Next steps:"; \
	echo "  1. Review changes: git status && git diff"; \
	echo "  2. Commit on the release branch: git commit -s -m 'Release v$$CARGO_VERSION'"; \
	echo "  3. Push: git push origin release/$$CARGO_VERSION"; \
	echo "  4. Open a PR to main and wait for CI/CD and reviews"; \
	echo "  5. Merge the PR — tag-release.yml creates v$$CARGO_VERSION automatically"; \
	echo "     (never create release tags manually)"

# ===========================================================================
# CLEANUP
# ===========================================================================

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -rf target/rust-coverage.lcov target/python-rust-coverage.lcov target/python-coverage.xml
	@rm -rf target/python/ test-results.xml
	@rm -rf dist/ *.egg-info/ .pytest_cache/ __pycache__/
	@rm -f sbom.json *-sbom.json *.cdx.json
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete"
