# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest hooks. GPU marker registration so ``-m gpu`` is valid
even when a leaf test file is collected in isolation.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: GPU-backed tests (skipped on Linux/Windows CI; required on macOS)",
    )
