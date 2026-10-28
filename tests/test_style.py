"""Import policy for the library: imports go at the top of the module.

A function may import something only when one of these holds:

* it is an optional extra (not the standard library and not a core dependency), so ``import wissi_viz``
  keeps working with the core install;
* it sits inside ``try: ... except ImportError`` (a guarded probe);
* it is an internal module that needs an extra itself (``metrics``, ``spatial``, ``segment``, ``sam``,
  ``nicegui``, ``plots``), imported from a core module only where that feature is used;
* it breaks a known import cycle;
* it is a platform-only standard-library module (``resource``, ``winreg``, ...).

Everything else (standard library, numpy, fastapi, tifffile, sibling modules) belongs at the top.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'src' / 'wissi_viz'

STDLIB = set(sys.stdlib_module_names) | {'__future__'}
# standard-library modules that exist only on some platforms may be imported where they are used
PLATFORM_ONLY = {'resource', 'fcntl', 'termios', 'pwd', 'grp', 'winreg', 'msvcrt', '_winapi'}
# import names of the core dependencies (pyproject `dependencies`) and what fastapi always brings along
CORE = {'numpy', 'fastapi', 'starlette', 'anyio', 'pydantic', 'tifffile', 'zarr', 'imagecodecs', 'czifile', 'PIL'}
CORE |= {'mapbox_earcut', 'truststore'}
DIST_TO_IMPORT = {'pillow': 'PIL', 'mapbox-earcut': 'mapbox_earcut'}
# modules that require an extra anyway may import that extra at the top
EXTRA_AT_TOP = {
    'metrics.py': {'scipy', 'shapely', 'skimage', 'pandas'},
    'spatial.py': {'scipy', 'shapely', 'skimage', 'pandas'},
    'segment.py': {'scipy', 'skimage'},
    'sam.py': {'scipy', 'skimage'},
    'nicegui.py': {'nicegui'},
}
FEATURE_MODULES = {'metrics', 'spatial', 'segment', 'sam', 'nicegui', 'plots', 'analyses'}
CYCLES = {('annotations', 'annio'), ('annotations', 'objects')}


def _guarded(stack: list[ast.AST]) -> bool:
    for n in stack:
        if isinstance(n, ast.Try):
            for h in n.handlers:
                names = []
                if isinstance(h.type, ast.Name):
                    names = [h.type.id]
                elif isinstance(h.type, ast.Tuple):
                    names = [e.id for e in h.type.elts if isinstance(e, ast.Name)]
                if {'ImportError', 'ModuleNotFoundError'} & set(names):
                    return True
    return False


def _allowed(node: ast.Import | ast.ImportFrom, path: Path) -> bool:
    here = path.stem
    if isinstance(node, ast.ImportFrom) and node.level > 0:
        if path.parent != PKG and node.level == 1:  # a sibling inside sources/ or connectors/: at the top
            return False
        targets = {node.module.split('.')[0]} if node.module else {a.name for a in node.names}
        if here not in FEATURE_MODULES and targets & FEATURE_MODULES:
            return True
        return any((here, t) in CYCLES for t in targets)
    top = (node.names[0].name if isinstance(node, ast.Import) else node.module).split('.')[0]
    if top in PLATFORM_ONLY:
        return True
    if top in STDLIB or top in CORE or top == 'wissi_viz':
        return False
    return top not in EXTRA_AT_TOP.get(path.name, set())


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    out: list[str] = []

    def visit(node: ast.AST, stack: list[ast.AST], in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, [*stack, child], True)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                if in_func and not _guarded(stack) and not _allowed(child, path):
                    out.append(f'{path.relative_to(ROOT)}:{child.lineno}: {ast.unparse(child)}')
            else:
                visit(child, [*stack, child], in_func)

    visit(tree, [], False)
    return out


def test_core_set_matches_pyproject():
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    block = re.search(r'^dependencies = \[(.*?)^\]', text, re.S | re.M).group(1)
    dists = re.findall(r'^\s*"([A-Za-z0-9_.-]+)', block, re.M)
    assert dists, 'no core dependencies parsed'
    imports = {DIST_TO_IMPORT.get(d.lower(), d.lower().replace('-', '_')) for d in dists}
    assert imports <= CORE, f'update CORE in tests/test_style.py: {sorted(imports - CORE)}'


@pytest.mark.parametrize('path', sorted(PKG.rglob('*.py')), ids=lambda p: str(p.relative_to(PKG)))
def test_imports_are_at_module_level(path):
    bad = _violations(path)
    assert not bad, 'move these imports to the top of the module:\n' + '\n'.join(bad)


def test_policy_catches_a_stdlib_import_in_a_function():
    probe = PKG / 'tiles.py'
    lazy = [
        n for n in ast.walk(ast.parse('def f():\n    import json\n    import torch\n')) if isinstance(n, ast.Import)
    ]
    assert not _allowed(lazy[0], probe)  # standard library: must be at the top
    assert _allowed(lazy[1], probe)  # optional extra: may stay lazy
