# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest hooks: registers the ``gpu`` marker."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: GPU-backed tests (skipped on Linux CI; required on macOS, and on Windows when ANGLE is configured)",
    )
