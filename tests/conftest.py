# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest hooks: registers the ``gpu`` and ``codec_sessions`` markers,
and holds a test's codec session slots for as long as the test runs."""

import pytest

from tests.codec_sessions import sessions


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: GPU-backed tests (skipped on Linux CI; required on macOS, and on Windows when ANGLE is configured)",
    )
    config.addinivalue_line(
        "markers",
        "codec_sessions(n): the test holds up to n hardware codec sessions at once; "
        "it takes n slots (tests/codec_sessions.py) before it runs",
    )


@pytest.fixture(autouse=True)
def _codec_session_slots(request):
    marker = request.node.get_closest_marker("codec_sessions")
    if marker is None:
        yield
        return
    with sessions(marker.args[0]):
        yield
