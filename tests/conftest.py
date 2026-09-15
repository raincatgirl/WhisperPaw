"""Pytest configuration for whisperpaw tests.

Forces the ``WPAW_READ_STDIN_OVERRIDE`` and ``WPAW_ZOOM_STDIN_OVERRIDE``
env vars to an empty string so that the stdin readers in
:mod:`whisperpaw.read` and :mod:`whisperpaw.zoom` short-circuit and
don't trip pytest's stdin capture (which raises OSError when anything
reads stdin while stdout is being captured).
"""
from __future__ import annotations

import os

#: Set once for the whole test session. The override is read every time
#: ``_read_stdin()`` / ``zoom._read_stdin()`` is called, so a test can
#: still monkeypatch it via ``monkeypatch.setenv`` if it needs a
#: specific value.
os.environ.setdefault("WPAW_READ_STDIN_OVERRIDE", "")
os.environ.setdefault("WPAW_ZOOM_STDIN_OVERRIDE", "")
