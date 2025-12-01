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
PYTHON_CRATE := crates/python
TEST_DIR := tests

# Use profiling profile for testing (optimized with debug symbols)
CARGO_PROFILE := --profile profiling
# For cargo-llvm-cov nextest, the cargo profile flag is different
LLVM_COV_PROFILE := --cargo-profile profiling

# Rust features for testing
RUST_FEATURES := --all-features

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
	@echo "    make test           - Run all tests with coverage"
	@echo "    make test-rust      - Run Rust tests only"
	@echo "    make test-python    - Run Python tests only"
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
format: format-rust format-python
	@echo "✓ All formatting complete"

.PHONY: format-rust
format-rust:
	@echo "Formatting Rust code with nightly..."
	@cargo +nightly fmt --all 2>/dev/null || cargo fmt --all
	@echo "✓ Rust formatting complete"

.PHONY: format-python
format-python:
	@echo "Formatting Python code with ruff..."
	@if [ -f "venv/bin/ruff" ]; then \
		. venv/bin/activate && ruff format $(PYTHON_CRATE) $(TEST_DIR); \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff format $(PYTHON_CRATE) $(TEST_DIR); \
	else \
		echo "Warning: ruff not found (see README.md for installation)"; \
	fi
	@echo "✓ Python formatting complete"

# ===========================================================================
# LINTING
# ===========================================================================

.PHONY: lint
lint: lint-rust lint-python
	@echo "✓ All linting complete"

.PHONY: lint-rust
lint-rust:
	@echo "Running clippy (strict mode)..."
	@cargo clippy --all-targets $(RUST_FEATURES) -- -D warnings
	@echo "✓ Clippy passed"

.PHONY: lint-python
lint-python:
	@echo "Running ruff linter..."
	@if [ -f "venv/bin/ruff" ]; then \
		. venv/bin/activate && ruff check $(PYTHON_CRATE) $(TEST_DIR); \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check $(PYTHON_CRATE) $(TEST_DIR); \
	else \
		echo "Warning: ruff not found (see README.md for installation)"; \
	fi
	@echo "✓ Python linting complete"

# ===========================================================================
# BUILDING
# ===========================================================================

.PHONY: check
check:
	@echo "Running cargo check..."
	@cargo check $(RUST_FEATURES) --workspace
	@echo "✓ Check passed"

.PHONY: build
build:
	@echo "Building with coverage instrumentation (profiling profile)..."
	@if ! cargo llvm-cov --version >/dev/null 2>&1; then \
		echo "ERROR: cargo-llvm-cov not installed (see README.md for installation)"; \
		exit 1; \
	fi
	@source <(cargo llvm-cov show-env --export-prefix) && \
		cargo build $(CARGO_PROFILE) $(RUST_FEATURES) --workspace
	@echo "✓ Build complete"

.PHONY: build-python
build-python:
	@echo "Building Python bindings..."
	@if command -v maturin >/dev/null 2>&1; then \
		maturin build -m $(PYTHON_CRATE)/Cargo.toml $(CARGO_PROFILE); \
	else \
		echo "ERROR: maturin not found (see README.md for installation)"; \
		exit 1; \
	fi
	@echo "✓ Python bindings built"

# ===========================================================================
# TESTING
# ===========================================================================

.PHONY: test
test: test-rust test-python
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
		--exclude edgefirst_hal --lcov --output-path target/rust-coverage.lcov
	@echo "✓ Rust tests passed"
	@echo "Coverage report: target/rust-coverage.lcov"

.PHONY: test-python
test-python:
	@echo "Running Python tests..."
	@echo "  Installing Python bindings..."
	@if [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && pip install -q $(PYTHON_CRATE)/[test]; \
	else \
		pip install -q $(PYTHON_CRATE)/[test]; \
	fi
	@echo "  Running tests..."
	@if [ -f "venv/bin/slipcover" ]; then \
		. venv/bin/activate && \
			python -m slipcover --xml --out target/python-coverage.xml \
				-m unittest discover -s $(TEST_DIR) -p "test*.py"; \
	elif command -v slipcover >/dev/null 2>&1; then \
		python -m slipcover --xml --out target/python-coverage.xml \
			-m unittest discover -s $(TEST_DIR) -p "test*.py"; \
	elif [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && \
			python -m unittest discover -s $(TEST_DIR) -p "test*.py"; \
	else \
		python -m unittest discover -s $(TEST_DIR) -p "test*.py"; \
	fi
	@echo "✓ Python tests passed"

.PHONY: bench
bench:
	@echo "Running benchmarks..."
	@cargo bench $(RUST_FEATURES) --workspace
	@echo "✓ Benchmarks complete"

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
	if [ -f "$(PYTHON_CRATE)/pyproject.toml" ]; then \
		PY_VERSION=$$(grep -m1 '^version = ' $(PYTHON_CRATE)/pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
		echo -n "  $(PYTHON_CRATE)/pyproject.toml: "; \
		if [ "$$PY_VERSION" = "$$CARGO_VERSION" ]; then \
			echo "✓"; \
		else \
			echo "✗ ($$PY_VERSION != $$CARGO_VERSION)"; \
			exit 1; \
		fi; \
	fi; \
	if [ -f "CHANGELOG.md" ]; then \
		echo -n "  CHANGELOG.md: "; \
		if grep -q "\[$$CARGO_VERSION\]" CHANGELOG.md || grep -q "## $$CARGO_VERSION" CHANGELOG.md; then \
			echo "✓"; \
		else \
			echo "✗ (version $$CARGO_VERSION not found)"; \
			exit 1; \
		fi; \
	fi; \
	echo "✓ Version verification complete"

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
	@echo "Next steps:"
	@echo "  1. Review changes: git status && git diff"
	@echo "  2. Commit: git add -A && git commit -m 'Prepare release'"
	@echo "  3. Push: git push origin main"
	@echo "  4. Wait for CI/CD to pass"
	@CARGO_VERSION=$$(grep -A10 '^\[workspace.package\]' Cargo.toml | grep 'version = ' | sed 's/.*version = "\(.*\)"/\1/'); \
	echo "  5. Tag: git tag -a -m 'Version $$CARGO_VERSION' v$$CARGO_VERSION"; \
	echo "  6. Push tag: git push origin v$$CARGO_VERSION"

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
