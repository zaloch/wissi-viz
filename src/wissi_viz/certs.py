"""Downloads trust the operating system's certificates (the default for the ``wissi-viz`` commands and the app).

Python checks HTTPS against its own certificate list, not the macOS keychain or the Windows certificate store. On a
network that inspects HTTPS (most corporate networks), or with a Python whose certificate list was never installed
(python.org's macOS installer), every download then fails with ``CERTIFICATE_VERIFY_FAILED``. So wissi-viz verifies
certificates with the operating system instead (via ``truststore``), as browsers do: whatever the computer trusts,
including what an IT department installed, is trusted for the sample pack and demo slides, benchmark images, model
downloads (SAM, Grounding DINO, SAM 3 from Hugging Face) and tile APIs. ``--no-system-certs`` (or
``WISSI_VIZ_SYSTEM_CERTS=0``) keeps Python's own list.

From Python (NiceGUI apps, notebooks) nothing changes by itself: ``wissi_viz.certs.use_system_certs()`` before the first
download.
"""

from __future__ import annotations

import argparse
import os

import truststore

FLAG = '--system-certs'  # the default now; still accepted
OFF = '--no-system-certs'
ENV = 'WISSI_VIZ_SYSTEM_CERTS'
HELP = (
    f"keep Python's own certificate list for downloads instead of this computer's (macOS keychain, Windows store), "
    f'which is the default (or set {ENV}=0)'
)
_on = False


def use_system_certs() -> None:
    """Verify HTTPS with the operating system's trust store from now on (this process, every library)."""
    global _on
    if not _on:
        truststore.inject_into_ssl()
        _on = True


def enabled() -> bool:
    return _on


def take_flag(argv: list[str]) -> list[str]:
    """Remove ``--system-certs`` / ``--no-system-certs`` from ``argv`` (anywhere) and use the system's certificates
    unless ``--no-system-certs`` is given or ``WISSI_VIZ_SYSTEM_CERTS`` is 0 / false / no / off."""
    rest = [a for a in argv if a not in (FLAG, OFF)]
    off = OFF in argv or os.environ.get(ENV, '').strip().lower() in ('0', 'false', 'no', 'off')
    if not off:
        use_system_certs()
    return rest


def add_arg(ap: argparse.ArgumentParser) -> None:
    """List the opt-out in a command's ``--help`` (it is taken out of the arguments before parsing)."""
    ap.add_argument(OFF, action='store_true', help=HELP)


def hint(error: object) -> str:
    """A next step for a certificate failure, '' for any other error."""
    if 'CERTIFICATE_VERIFY_FAILED' not in str(error):
        return ''
    if _on:
        return (
            "\n\nThis computer's certificates do not trust that server either: the network may block it, or its "
            'certificate is not installed here (ask IT for the certificate of the HTTPS-inspecting proxy).'
        )
    return (
        f"\n\nPython's own certificate list does not trust that server (common on networks that inspect HTTPS, and with "
        f"python.org's macOS Python). Run without {OFF} / {ENV}=0, or call wissi_viz.certs.use_system_certs(), to "
        "trust this computer's certificates."
    )
