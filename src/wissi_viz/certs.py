"""Trust the operating system's certificates for downloads (``--system-certs`` / ``WISSI_VIZ_SYSTEM_CERTS=1``).

Python checks HTTPS against its own certificate list, not the macOS keychain or the Windows certificate store. On a
network that inspects HTTPS (most corporate networks) every download then fails with ``CERTIFICATE_VERIFY_FAILED``.
With the flag, wissi-viz verifies certificates with the operating system instead (via ``truststore``), as browsers
do, so whatever your IT department installed is trusted: sample pack and demo slides, benchmark images, SAM weights
and tile APIs.

From Python (NiceGUI apps, notebooks): ``wissi_viz.certs.use_system_certs()`` before the first download.
"""

from __future__ import annotations

import argparse
import os

import truststore

FLAG = '--system-certs'
ENV = 'WISSI_VIZ_SYSTEM_CERTS'
HELP = (
    "trust this computer's certificates (macOS keychain, Windows store) for downloads: for networks that "
    f'inspect HTTPS, e.g. corporate ones (or set {ENV}=1)'
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
    """Remove ``--system-certs`` from ``argv`` (anywhere) and switch it on if given or if ``WISSI_VIZ_SYSTEM_CERTS`` is set."""
    rest = [a for a in argv if a != FLAG]
    if len(rest) != len(argv) or os.environ.get(ENV, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        use_system_certs()
    return rest


def add_arg(ap: argparse.ArgumentParser) -> None:
    """List the flag in a command's ``--help`` (it is taken out of the arguments before parsing)."""
    ap.add_argument(FLAG, action='store_true', help=HELP)


def hint(error: object) -> str:
    """A next step for a certificate failure, '' for any other error."""
    if 'CERTIFICATE_VERIFY_FAILED' not in str(error) or _on:
        return ''
    return (
        f'\n\nThis network inspects HTTPS (common on corporate networks). Run again with {FLAG} '
        f'(or set {ENV}=1) to trust the certificates installed on this computer.'
    )
