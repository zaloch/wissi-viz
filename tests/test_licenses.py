"""The core install (and the permissive extras) must stay permissively licensed."""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, metadata, requires

import pytest

PERMISSIVE = re.compile(r'\b(MIT|BSD|Apache|ISC|HPND|MIT-CMU|PSF|Python Software Foundation|Zlib|0BSD|'
                        r'Unlicense|CC0|Historical Permission)\b', re.I)
COPYLEFT = re.compile(r'\b(L?GPL|AGPL|MPL|EPL|CDDL|SSPL|EUPL|Lesser General Public|General Public License)\b', re.I)
CORE_EXTRAS = ('analysis', 'io', 'slideio', 'nicegui', 'server', 'remote', 'bench', 'all')


def _name(req: str) -> str:
    return re.split(r'[\s;<>=!~\[]', req, maxsplit=1)[0].lower()


def _licence(dist: str) -> str:
    m = metadata(dist)
    parts = [m.get('License-Expression') or '', (m.get('License') or '')[:300]]
    parts += [c for c in (m.get_all('Classifier') or []) if c.startswith('License')]
    return ' | '.join(p for p in parts if p)


def _direct(extra: str | None):
    out = []
    for r in requires('wissi-viz') or []:
        marker = r.split(';', 1)[1] if ';' in r else ''
        if extra is None and 'extra' not in marker:
            out.append(_name(r))
        elif extra is not None and re.search(rf'extra\s*==\s*["\']{extra}["\']', marker):
            out.append(_name(r))
    return out


@pytest.mark.parametrize('extra', [None, *CORE_EXTRAS])
def test_direct_dependencies_are_permissive(extra):
    for dist in _direct(extra):
        if dist == 'wissi-viz':
            continue
        try:
            lic = _licence(dist)
        except PackageNotFoundError:
            pytest.skip(f'{dist} not installed')
        assert PERMISSIVE.search(lic), f'{dist}: no permissive licence found in {lic!r}'
        # dual/multi-licensed packages: a copyleft mention is only OK alongside a permissive option
        if COPYLEFT.search(lic) and not re.search(r'\bOR\b', lic):
            pytest.fail(f'{dist}: copyleft licence in core/permissive extra: {lic!r}')


def test_lgpl_backends_are_opt_in():
    assert set(_direct('lgpl')) >= {'pylibczirw', 'openslide-python'}
    core = set(_direct(None))
    for extra in CORE_EXTRAS:
        core |= set(_direct(extra))
    assert not core & {'pylibczirw', 'openslide-python', 'openslide-bin', 'orjson'}


def test_core_import_does_not_load_lgpl():
    import subprocess
    import sys
    code = ('import sys, wissi_viz, wissi_viz.server, wissi_viz.sources.czi, wissi_viz.sources.tiff;'
            'bad=[m for m in ("pylibCZIrw","openslide") if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
