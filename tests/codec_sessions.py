# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Slots that bound the hardware codec sessions tests hold at once, across
the threads and processes of every test run on the machine.

A slot is an exclusive lock on one of ``SLOTS`` files in the temp directory,
the same files the Rust tests lock (``crates/codec/tests/common/sessions.rs``).
A test takes as many slots as the sessions it keeps open at once, before it
opens them, and frees them when the ``with`` block ends.
"""

from __future__ import annotations

import contextlib
import errno
import os
import random
import sys
import tempfile
import time

SLOTS = 4

_HELD = (
    {errno.EACCES, errno.EDEADLK}
    if sys.platform == "win32"
    else {errno.EWOULDBLOCK, errno.EAGAIN}
)


def _temp_dir() -> str:
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return os.environ.get("TMPDIR") or "/tmp"


def _slot_path(index: int) -> str:
    return os.path.join(_temp_dir(), f"edgefirst-codec-session-{index}.lock")


def _try_lock(path: str):
    """The open slot file at ``path``, locked, or ``None`` when another holder
    has it. Any other locking failure raises."""
    f = open(path, "a+b")  # noqa: SIM115
    try:
        if sys.platform == "win32":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        f.close()
        if e.errno in _HELD:
            return None
        raise
    return f


@contextlib.contextmanager
def sessions(n: int):
    """Hold ``n`` slots for the ``with`` block, waiting until ``n`` are free
    at the same time. The slots are taken all at once or not at all."""
    assert n <= SLOTS, f"{n} sessions requested; at most {SLOTS} are held at once"
    while True:
        held = []
        for index in range(SLOTS):
            if len(held) == n:
                break
            f = _try_lock(_slot_path(index))
            if f is not None:
                held.append(f)
        if len(held) == n:
            break
        for f in held:
            f.close()
        time.sleep(random.uniform(0.005, 0.036))
    try:
        yield
    finally:
        for f in held:
            f.close()
