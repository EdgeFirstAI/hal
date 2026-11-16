# HAL Project Makefile
#
# Workflows:
#   - Code formatting with nightly Rust and ruff for Python
#   - Linting with clippy and ruff
#   - Testing with cargo-nextest and llvm-cov (using profiling profile)
#   - SBOM generation and license policy validation
#   - Pre-release quality checks and version verification
#
# Tests use the 'profiling' profile, which produces optimized binaries with
# full debug symbols.
#
# Prerequisites: See README.md or CONTRIBUTING.md for tool installation.
#
# ===========================================================================

# Use bash for shell commands
SHELL := /bin/bash

# Project configuration
PROJECT_NAME := hal
# One extension module per package.
PYTHON_PACKAGES := tensor codec image decoder tracker kernels
PYTHON_CRATES := $(addprefix crates/python-,$(PYTHON_PACKAGES))
# cargo package names of the Python extension crates, for --exclude.
PYTHON_CRATE_NAMES := edgefirst-python-common \
	$(addprefix edgefirst-python-,$(PYTHON_PACKAGES))
TEST_DIR := tests
# Paths ruff checks, including the Python in python-common's .md files.
RUFF_PATHS := $(PYTHON_CRATES) crates/python-common $(TEST_DIR)

# The modular C-API leaves: standalone packages excluded from the workspace,
# in build order.
CAPI_MODULAR_CRATES := tensor-capi image-capi codec-capi decoder-capi tracker-capi kernels-capi

# Selects edgefirst-tensor's dynamic backend for checks of the libraries (see
# ARCHITECTURE.md, Tensor Backend Selection).
TENSOR_DYNAMIC := --config $(CURDIR)/.cargo/tensor-dynamic.toml
# Fails when RUSTFLAGS is set in the environment.
define require_fragment_rustflags
	@if [ -n "$$RUSTFLAGS" ]; then \
		echo "ERROR: RUSTFLAGS is set, which hides the tensor backend cfg; use CARGO_TARGET_<TRIPLE>_RUSTFLAGS instead" >&2; \
		exit 1; \
	fi
endef
# The C-API leaves that call into libedgefirst_tensor instead of embedding it.
CAPI_DYNAMIC_CRATES := image-capi codec-capi decoder-capi
# The leaves whose .cargo/config.toml selects the dynamic backend.
DYNAMIC_LEAVES := $(addprefix crates/,$(CAPI_DYNAMIC_CRATES)) $(addprefix crates/python-,tensor codec image decoder)
# Every library a shared-library leaf compiles in, checked under both backends.
BACKEND_CHECK_CRATES := edgefirst-tensor edgefirst-codec edgefirst-image \
	edgefirst-decoder edgefirst-tracker edgefirst-python-common
BACKEND_CHECK_FEATURES := edgefirst-python-common/codec,edgefirst-python-common/image,edgefirst-python-common/decoder,edgefirst-python-common/tracker,edgefirst-python-common/kernels

# The crates.io packages, each checked on its own as a dependent builds it, without workspace feature unification.
PUBLISHED_CRATES := edgefirst-tensor-abi edgefirst-tensor-ffi edgefirst-tensor edgefirst-egl edgefirst-gl \
	edgefirst-codec edgefirst-kernels edgefirst-decoder-abi edgefirst-decoder edgefirst-tracker edgefirst-image

# Use profiling profile for testing (optimized with debug symbols)
CARGO_PROFILE := --profile profiling
# The same profile, as cargo-llvm-cov nextest spells it
LLVM_COV_PROFILE := --cargo-profile profiling

# Rust features (all except opencv)
RUST_FEATURES := --features opengl,ndarray,edgefirst-decoder/model-files

# The interpreter that runs this repository's helper scripts: the first of
# python3 and python that executes.
PYTHON := $(shell \
	for c in python3 python; do \
		if command -v "$$c" >/dev/null 2>&1 && "$$c" -c 'pass' >/dev/null 2>&1; then \
			echo "$$c"; break; \
		fi; \
	done)
# Falls back to the literal python3 when neither executes.
PYTHON := $(if $(PYTHON),$(PYTHON),python3)

# Versioned Python interpreter name for maturin -i, such as 'python3.10'.
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
# matching 'abi3-<PYABI>' feature and produces a version-agnostic wheel (tagged
# cp<abi>-abi3) usable by any CPython at or above that minimum. Empty value
# produces a wheel tagged for the interpreter chosen by -i (e.g. cp310-cp310).
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
	@echo "    make check-benchmarks - Type-check the excluded benchmarks workspace"
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
	@echo "    make check-macho    - Assert shipped Mach-O string pools are aligned"
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
		. venv/bin/activate && ruff format $(RUFF_PATHS); \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff format $(RUFF_PATHS); \
	else \
		echo "Warning: ruff not found (see README.md for installation)"; \
	fi
	@echo "✓ Python formatting complete"

.PHONY: format-capi-modular
format-capi-modular:
	@echo "Formatting the modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		cargo fmt --manifest-path crates/$$c/Cargo.toml || exit 1; \
	done
	@echo "✓ Modular C-API formatting complete"

# ===========================================================================
# LINTING
# ===========================================================================

.PHONY: lint
lint: lint-rust lint-tensor-dynamic lint-python lint-python-abi3-py38 lint-capi-modular lint-backend-scope
	@echo "✓ All linting complete"

.PHONY: lint-rust
lint-rust:
	@echo "Running clippy (strict mode)..."
	@cargo clippy --workspace --all-targets $(RUST_FEATURES) \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c)) \
		-- -D warnings
	@echo "✓ Clippy passed"

# Lints the Python bindings under the abi3-py38 configuration, one crate at a
# time.
.PHONY: lint-python-abi3-py38
lint-python-abi3-py38:
	@echo "Running clippy on the Python bindings under abi3-py38..."
	@for pkg in $(PYTHON_PACKAGES); do \
		(cd crates/python-$$pkg && cargo clippy --features abi3-py38 --all-targets -- -D warnings) || exit 1; \
	done
	@echo "✓ abi3-py38 Python-bindings clippy passed"

.PHONY: lint-python
lint-python:
	@echo "Running ruff linter..."
	@if [ -f "venv/bin/ruff" ]; then \
		. venv/bin/activate && ruff check $(RUFF_PATHS); \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check $(RUFF_PATHS); \
	else \
		echo "Warning: ruff not found (see README.md for installation)"; \
	fi
	@echo "✓ Python linting complete"

.PHONY: lint-capi-modular
lint-capi-modular:
	@echo "Running clippy on the modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		(cd crates/$$c && cargo clippy --all-targets --target-dir $(CURDIR)/target -- -D warnings) || exit 1; \
	done
	@echo "✓ Modular C-API clippy passed"

# ===========================================================================
# BUILDING
# ===========================================================================

.PHONY: check
check: check-capi-modular check-standalone check-backend-guards check-benchmarks
	@echo "Running cargo check..."
	@cargo check $(RUST_FEATURES) --workspace \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c))
	@echo "✓ Check passed"

.PHONY: check-standalone
check-standalone:
	@echo "Running cargo check on each published crate on its own..."
	@for c in $(PUBLISHED_CRATES); do \
		cargo check -p $$c || exit 1; \
	done
	@cargo check -p edgefirst-decoder --features model-files
	@echo "✓ Standalone check passed"
	@$(MAKE) --no-print-directory check-backends

# Checks every library a shared-library leaf compiles in against both tensor
# backends.
.PHONY: check-backends
check-backends:
	$(require_fragment_rustflags)
	@echo "Checking the libraries against the static tensor backend..."
	@cargo check $(foreach c,$(BACKEND_CHECK_CRATES),-p $(c)) --features $(BACKEND_CHECK_FEATURES) --lib
	@echo "Checking the libraries against the dynamic tensor backend..."
	@cargo check $(TENSOR_DYNAMIC) $(foreach c,$(BACKEND_CHECK_CRATES),-p $(c)) --features $(BACKEND_CHECK_FEATURES) --lib
	@echo "✓ Both backends check"

# Fails when a path outside BACKEND_SCOPE_ALLOWED names the tensor backend, when
# RUSTFLAGS selects it, or when a leaf's .cargo/config.toml differs from
# .cargo/tensor-dynamic.toml.
BACKEND_SCOPE_ALLOWED := ^(crates/tensor/|crates/build-support/|crates/(image|codec|decoder|tensor)-capi/|crates/python-(tensor|codec|image|decoder)/|\.cargo/tensor-dynamic\.toml|Cargo\.toml|Makefile|\.github/|scripts/|tests/|[^:]*\.md:)
.PHONY: lint-backend-scope
lint-backend-scope:
	@echo "Checking that only the tensor crate and the leaves name the tensor backend..."
	@hits=$$(git grep --untracked -nIiE 'edgefirst_tensor_dynamic|edgefirst-tensor/(static|dynamic)|CARGO_FEATURE_(STATIC|DYNAMIC)\b' -- . ':!.claude' \
		| grep -vE '$(BACKEND_SCOPE_ALLOWED)' || true); \
	if [ -n "$$hits" ]; then echo "$$hits"; echo "ERROR: the tensor backend is named outside edgefirst-tensor and the leaves"; exit 1; fi
	@hits=$$(git grep --untracked -nIiE 'RUSTFLAGS[^#]*edgefirst_tensor_dynamic' -- Makefile .github scripts || true); \
	if [ -n "$$hits" ]; then echo "$$hits"; echo "ERROR: select the dynamic backend with a .cargo/config.toml, never RUSTFLAGS"; exit 1; fi
	@for d in $(DYNAMIC_LEAVES); do \
		cmp -s .cargo/tensor-dynamic.toml $$d/.cargo/config.toml \
			|| { echo "ERROR: $$d/.cargo/config.toml differs from .cargo/tensor-dynamic.toml"; exit 1; }; \
	done
	@echo "✓ Backend scope clean"

# Checks that each leaf rejects the wrong tensor backend with its own message.
.PHONY: check-backend-guards
check-backend-guards:
	$(require_fragment_rustflags)
	@echo "Checking that each leaf rejects the wrong tensor backend..."
	@out=$$(cargo check $(TENSOR_DYNAMIC) --manifest-path crates/tensor-capi/Cargo.toml --target-dir target 2>&1); \
	echo "$$out" | grep -q 'edgefirst-tensor-capi is the tensor implementation' \
		|| { echo "$$out" | tail -20; echo "ERROR: tensor-capi accepted the dynamic backend"; exit 1; }
	@for c in $(CAPI_DYNAMIC_CRATES); do \
		out=$$(cargo check --manifest-path crates/$$c/Cargo.toml --target-dir target 2>&1); \
		echo "$$out" | grep -q "edgefirst-$$c links libedgefirst_tensor and must be built with the dynamic" \
			|| { echo "$$out" | tail -20; echo "ERROR: $$c accepted the static backend"; exit 1; }; \
	done
	@for p in tensor codec image decoder; do \
		out=$$(env -u EDGEFIRST_TENSOR_STATIC_COMPARISON cargo check -p edgefirst-python-$$p 2>&1); \
		echo "$$out" | grep -q "edgefirst-python-$$p links libedgefirst_tensor and must be built with the dynamic" \
			|| { echo "$$out" | tail -20; echo "ERROR: python-$$p accepted the static backend"; exit 1; }; \
	done
	@EDGEFIRST_TENSOR_STATIC_COMPARISON=1 cargo check -q -p edgefirst-python-image \
		|| { echo "ERROR: the static comparison opt-in does not build"; exit 1; }
	@echo "✓ Every leaf guard fires"

.PHONY: check-capi-modular
check-capi-modular:
	@echo "Running cargo check on the modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		(cd crates/$$c && cargo check --target-dir $(CURDIR)/target) || exit 1; \
	done
	@echo "✓ Modular C-API check passed"

# Type-checks benchmarks/, which the root workspace excludes.
.PHONY: check-benchmarks
check-benchmarks:
	@echo "Running cargo check on the benchmarks workspace..."
	@cargo check --manifest-path benchmarks/Cargo.toml --workspace --all-targets --locked \
		--target-dir $(CURDIR)/target
	@echo "✓ Benchmarks check passed"

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
	@echo "Building the modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		(cd crates/$$c && cargo build --target-dir $(CURDIR)/target) || exit 1; \
	done
	@echo "✓ Modular C-API build complete"
	@$(MAKE) capi-symlinks

# Builds the C libraries under the release or profiling profile.
.PHONY: capi-libs-release capi-libs-profiling
capi-libs-release capi-libs-profiling: capi-libs-%:
	@echo "Building the modular C-API crates ($*)..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		(cd crates/$$c && cargo build --profile $* $(if $(TARGET),--target $(TARGET)) --target-dir $(CURDIR)/target) || exit 1; \
	done
	@echo "✓ Modular C-API $* build complete"
	@$(MAKE) capi-symlinks

# Creates the versioned symlink (libedgefirst_X.so.0, libedgefirst_X.0.dylib)
# next to each leaf library.
.PHONY: capi-symlinks
capi-symlinks:
	@for l in tensor image codec decoder tracker kernels; do \
		for d in target/debug target/release target/profiling; do \
			[ -f $$d/libedgefirst_$$l.so ] && \
				ln -sf libedgefirst_$$l.so $$d/libedgefirst_$$l.so.0 || true; \
			[ -f $$d/libedgefirst_$$l.dylib ] && \
				ln -sf libedgefirst_$$l.dylib $$d/libedgefirst_$$l.0.dylib || true; \
		done; \
	done

# One relocatable archive: headers, soversioned libraries, pkg-config,
# LICENSE, INSTALL.txt. tar.gz on Linux, zip on Windows and macOS.
# PACKAGE_PROFILE=profiling packages a test build.
PACKAGE_PROFILE ?= release
.PHONY: package
package: capi-libs-$(PACKAGE_PROFILE)
	@mkdir -p dist
	@./scripts/package-capi.sh --libdir target/$(PACKAGE_PROFILE) --outdir dist
	@$(PYTHON) scripts/check_macho_alignment.py dist
	@echo "✓ C archive in dist/"

# Runs the Mach-O alignment check over the MACHO_DIRS paths that exist.
# Override MACHO_DIRS to scan elsewhere, e.g.
#   make check-macho MACHO_DIRS="target/wheels dist"
MACHO_DIRS ?= target/release target/wheels

.PHONY: check-macho
check-macho:
	@built=""; \
	for d in $(MACHO_DIRS); do [ -e "$$d" ] && built="$$built $$d"; done; \
	if [ -z "$$built" ]; then \
		echo "check-macho: nothing to scan (looked in: $(MACHO_DIRS))"; \
	else \
		$(PYTHON) scripts/check_macho_alignment.py $$built; \
	fi

.PHONY: build-python
build-python:
	@echo "Building Python bindings..."
	@if command -v maturin >/dev/null 2>&1; then \
		for c in $(PYTHON_CRATES); do \
			(cd "$$c" && maturin build --auditwheel skip $(CARGO_PROFILE)) || exit 1; \
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
			. venv/bin/activate && (cd "$$c" && \
				maturin build --release --zig --compatibility manylinux2014 --auditwheel skip \
					$(if $(TARGET),--target $(TARGET)) \
					-i $(PYTHON_INTERPRETER) \
					$(if $(PYABI),--features abi3-$(PYABI))) || exit 1; \
		else \
			(cd "$$c" && maturin build --release --zig --compatibility manylinux2014 --auditwheel skip \
				$(if $(TARGET),--target $(TARGET)) \
				-i $(PYTHON_INTERPRETER) \
				$(if $(PYABI),--features abi3-$(PYABI))) || exit 1; \
		fi; \
	done
	@$(PYTHON) scripts/check_wheel_layout.py target/wheels
	@$(PYTHON) scripts/check_macho_alignment.py target/wheels
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

# Runs the doctests of the Rust crates, which cargo-nextest does not run.
# The PyO3 binding crates are excluded.
.PHONY: test-doc
test-doc:
	@echo "Running Rust doctests (nextest does not)..."
	@cargo test --workspace --doc $(RUST_FEATURES) \
		$(foreach c,$(PYTHON_CRATE_NAMES),--exclude $(c)) \
		--exclude edgefirst-python-common
	@echo "✓ Doctests passed"

# The PEP 517 source-directory install (`pip install <dir>/`) of each Python
# package. Narrow it with PYTHON_PACKAGES:
#
#   make install-python PYTHON_PACKAGES=tensor
.PHONY: install-python
install-python:
	@echo "  Installing Python bindings ($(PYTHON_PACKAGES))..."
	@if [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && pip install -q numpy || exit 1; \
	else \
		pip install -q numpy || exit 1; \
	fi
	@for c in $(PYTHON_CRATES); do \
		if [ -f "venv/bin/activate" ]; then \
			. venv/bin/activate && MATURIN_PEP517_ARGS="--auditwheel skip --profile profiling" \
				pip install -q --no-deps --force-reinstall "$$c/" || exit 1; \
		else \
			MATURIN_PEP517_ARGS="--auditwheel skip --profile profiling" \
				pip install -q --no-deps --force-reinstall "$$c/" || exit 1; \
		fi; \
	done

.PHONY: test-python
test-python: install-python
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

# Builds and tests the modular C-API leaves (CAPI_MODULAR_CRATES) via
# --manifest-path/--target-dir.
.PHONY: test-capi-modular
test-capi-modular: capi-libs
	@echo "Running tests for the modular C-API crates..."
	@for c in $(CAPI_MODULAR_CRATES); do \
		(cd crates/$$c && EF_REQUIRE_FRESH_ARTIFACTS=1 cargo test --target-dir $(CURDIR)/target -- --test-threads=1) || exit 1; \
	done
	@echo "✓ Modular C-API tests passed"

# Runs the named edgefirst-tensor test targets against the dynamic backend.
DYNAMIC_TEST_TARGETS := dynamic_primitives protocol_roundtrip map_access \
	identity scenarios no_global_state vocabulary logical_shape

.PHONY: test-tensor-dynamic
test-tensor-dynamic: capi-libs
	$(require_fragment_rustflags)
	@echo "Running edgefirst-tensor's dynamic-backend tests..."
	@for t in $(DYNAMIC_TEST_TARGETS); do \
		LD_LIBRARY_PATH=$(CURDIR)/target/debug cargo test $(TENSOR_DYNAMIC) -p edgefirst-tensor \
			--test $$t || exit 1; \
	done
	@echo "✓ Dynamic-backend tests passed"

# Clippy for the dynamic backend.
.PHONY: lint-tensor-dynamic
lint-tensor-dynamic:
	$(require_fragment_rustflags)
	@echo "Running clippy on the dynamic backend (strict mode)..."
	@cargo clippy $(TENSOR_DYNAMIC) -p edgefirst-tensor --lib -- -D warnings
	@for t in $(DYNAMIC_TEST_TARGETS); do \
		cargo clippy $(TENSOR_DYNAMIC) -p edgefirst-tensor \
			--test $$t -- -D warnings \
			|| exit 1; \
	done
	@echo "✓ Dynamic-backend clippy passed"

# Compiles every shipped header as C11 and C++17 under -Werror, then links
# and runs one consumer binary per modular C-API library.
.PHONY: test-capi-link
test-capi-link: capi-libs
	@./scripts/check-headers.sh

# Runs scripts/check-single-home.sh over the debug and release C libraries.
.PHONY: check-single-home
check-single-home: capi-libs capi-libs-release
	@./scripts/check-single-home.sh

# ===========================================================================
# DIFFERENTIAL TESTING
# ===========================================================================
#
# Builds one venv per tensor backend from the same commit and runs
# scripts/check-single-home.sh --differential-only over them. The static
# side builds from the repository root with
# EDGEFIRST_TENSOR_STATIC_COMPARISON=1.
DIFFERENTIAL_STATIC_VENV := $(CURDIR)/target/differential-static-venv
DIFFERENTIAL_DYNAMIC_VENV := $(CURDIR)/target/differential-dynamic-venv
# The packages the Python suite imports, pinned to the versions in venv.
DIFFERENTIAL_TEST_DEPS := numpy==2.4.6 pillow==12.2.0 \
	pytest==9.0.3 pyyaml==6.0.3 safetensors==0.8.0 psutil==7.2.2

# Both venvs share one --target-dir, and each has its own wheel output
# directory, cleared before the build. DIFFERENTIAL_ASSERT_ARCH checks the
# architecture of every .so about to be installed.
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

# Asserts that no wheel of the static set carries libedgefirst_tensor. The
# dynamic set is checked by scripts/check_wheel_layout.py.
define DIFFERENTIAL_ASSERT_STATIC_SELF_CONTAINED
	for w in $(1)/*.whl; do \
		found=$$(unzip -Z1 "$$w" | grep -c 'libedgefirst_tensor' || true); \
		if [ "$$found" -ne 0 ]; then \
			echo "ERROR: $$w carries libedgefirst_tensor -- this is the STATIC set, it should embed the implementation and link nothing external" >&2; \
			exit 1; \
		fi; \
	done
endef

# Build one comparison venv per backend from freshly built wheels.
.PHONY: venv-static
venv-static:
	@echo "Building the static-backend comparison venv..."
	@rm -rf $(DIFFERENTIAL_STATIC_VENV) target/differential-wheels-static
	@mkdir -p target/differential-wheels-static
	@venv/bin/python -m venv $(DIFFERENTIAL_STATIC_VENV)
	@$(DIFFERENTIAL_STATIC_VENV)/bin/pip install -q --upgrade pip
	@$(DIFFERENTIAL_STATIC_VENV)/bin/pip install -q maturin $(DIFFERENTIAL_TEST_DEPS)
	@for c in $(PYTHON_CRATES); do \
		EDGEFIRST_TENSOR_STATIC_COMPARISON=1 $(DIFFERENTIAL_STATIC_VENV)/bin/maturin build --profile profiling --auditwheel skip \
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
		(cd "$$c" && $(DIFFERENTIAL_DYNAMIC_VENV)/bin/maturin build --profile profiling --auditwheel skip \
			-o $(CURDIR)/target/differential-wheels-dynamic \
			-i $(DIFFERENTIAL_DYNAMIC_VENV)/bin/python) || exit 1; \
	done
	@$(call DIFFERENTIAL_ASSERT_ARCH,target/differential-wheels-dynamic)
	@$(PYTHON) scripts/check_wheel_layout.py target/differential-wheels-dynamic
	@$(DIFFERENTIAL_DYNAMIC_VENV)/bin/pip install -q --no-deps \
		target/differential-wheels-dynamic/*.whl
	@echo "✓ Dynamic-backend comparison venv ready: $(DIFFERENTIAL_DYNAMIC_VENV)"

.PHONY: test-differential
test-differential: venv-static venv-dynamic
	@echo "Running the backend comparison (static vs. dynamic, test for test)..."
	@PY_STATIC_VENV=$(DIFFERENTIAL_STATIC_VENV) PY_DYNAMIC_VENV=$(DIFFERENTIAL_DYNAMIC_VENV) \
		./scripts/check-single-home.sh --differential-only

# Builds and runs the two-library C program, which links edgefirst_codec and
# edgefirst_tensor and decodes a JPEG into a host-memory tensor.
.PHONY: test-two-library-user
test-two-library-user: capi-libs
	@cc -std=c11 -Wall -Wextra -Werror -o target/test_two_library_user \
		crates/codec-capi/tests/c/test_two_library_user.c \
		-Icrates/codec-capi/include -Icrates/tensor-capi/include \
		-Ltarget/debug -ledgefirst_codec -ledgefirst_tensor \
		-Wl,-rpath,$(PWD)/target/debug
	@./target/test_two_library_user

# Runs the test suite on the SSH hosts given in TARGETS.
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
			if [ -e "$$d/libcudart.so" ] || [ -e "$$d/libcudart.so.13" ] || [ -e "$$d/libcudart.so.12" ] || [ -e "$$d/libcudart.so.11.0" ]; then CUDART_LIB="$$d"; break; fi; \
		done; \
	fi; \
	if [ -n "$$CUDART_LIB" ]; then \
		echo "  using libcudart at $$CUDART_LIB"; \
		export HAL_TEST_REQUIRE_CUDA=1; \
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
# library path and runs only the codec bench.
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

# Targets whose dependency graph reaches a distributed artifact.
SBOM_TARGETS := \
	x86_64-unknown-linux-gnu \
	aarch64-unknown-linux-gnu \
	x86_64-pc-windows-msvc \
	aarch64-apple-darwin \
	x86_64-apple-darwin \
	aarch64-apple-ios \
	aarch64-apple-ios-sim \
	aarch64-linux-android \
	x86_64-linux-android

# The C-API leaf manifests, each scanned with its own lockfile.
SBOM_CAPI_MANIFESTS := $(wildcard crates/*-capi/Cargo.toml)

# Generates the SBOM and checks the licence policy with the scripts from
# EdgeFirstAI/.github, then checks NOTICE against it.
.PHONY: sbom
sbom:
	@echo "Fetching the org license policy..."
	@mkdir -p target
	@CI_SCRIPTS=$$(.github/scripts/fetch-ci-scripts.sh); \
		echo "Generating SBOM (policy from $$CI_SCRIPTS)..."; \
		PROJECT_NAME=hal PROJECT_TYPE=library VERSION_FILE=Cargo.toml \
		SOURCE_DIRS="crates tests" SBOM_MODE=$${SBOM_MODE:-dependency} \
		SBOM_TARGETS="$(SBOM_TARGETS)" \
		SBOM_EXTRA_MANIFESTS="$(SBOM_CAPI_MANIFESTS)" \
		OUTPUT_DIR=sbom SCRIPT_DIR="$$CI_SCRIPTS" \
		"$$CI_SCRIPTS/generate_sbom.sh" 2>&1 | tee target/sbom-policy.log; \
		exit $${PIPESTATUS[0]}
	@$(PYTHON) .github/scripts/generate_notice.py --check sbom/sbom.json NOTICE
	@echo "Validating SBOM format..."
	@if command -v cyclonedx >/dev/null 2>&1; then \
		cyclonedx validate --input-file sbom/sbom.json; \
	else \
		echo "Warning: cyclonedx CLI not found, skipping validation"; \
	fi
	@echo "✓ SBOM generated and validated (sbom/sbom.json)"

# Regenerates NOTICE from sbom/sbom.json.
.PHONY: notice
notice:
	@CI_SCRIPTS=$$(.github/scripts/fetch-ci-scripts.sh); \
		$(PYTHON) .github/scripts/generate_notice.py sbom/sbom.json > NOTICE.new; \
		mv NOTICE.new NOTICE; \
		$(PYTHON) "$$CI_SCRIPTS/validate_notice.py" sbom/sbom.json --notice NOTICE
	@echo "✓ NOTICE regenerated from sbom/sbom.json"

# ===========================================================================
# VERSION VERIFICATION
# ===========================================================================

# Checks version consistency with .github/scripts/verify_version.py.
.PHONY: verify-version
verify-version:
	@$(PYTHON) .github/scripts/verify_version.py

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
