"""Test-wide settings."""

from __future__ import annotations

import os

# the commands trust the system's certificates by default (process-wide, via truststore): tests that run them keep
# Python's own list, so the process stays as it was (tests/test_certs.py checks the default on its own)
os.environ.setdefault('WISSI_VIZ_SYSTEM_CERTS', '0')
