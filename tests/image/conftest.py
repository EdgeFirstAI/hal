# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Decodes JPEGs on the CPU engine, except in a test marked ``codec_sessions``."""

import pytest


@pytest.fixture(autouse=True)
def _cpu_jpeg_decoder(request, monkeypatch):
    if request.node.get_closest_marker("codec_sessions") is None:
        monkeypatch.setenv("EDGEFIRST_CODEC_ENGINE", "cpu")
