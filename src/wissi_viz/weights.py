"""Model files downloaded once from this project's GitHub releases, checked against the release's SHA-256 sums.

    d = fetch('grounding-dino-tiny-onnx-v1', {'grounding-dino-tiny-int8.onnx': 'model.onnx', 'vocab.txt': 'vocab.txt'},
              'grounding-dino-tiny-int8')

Files land in the user cache (``<cache>/models/<name>/``) and are reused from there. GitHub is reachable from most
networks, including corporate ones (for HTTPS inspection, see :mod:`wissi_viz.certs`).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import certs
from ._compat import user_dir

RELEASES = 'https://github.com/zaloch/wissi-viz/releases/download'


def cache_dir(name: str) -> Path:
    return user_dir('cache') / 'models' / name


def have(name: str, files: dict[str, str]) -> Path | None:
    """The folder if a download of ``files`` finished there, else None."""
    d = cache_dir(name)
    return d if (d / '.complete').exists() and all((d / local).exists() for local in files.values()) else None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def fetch(
    release: str,
    files: dict[str, str],
    name: str,
    progress: Callable[[float, str], None] | None = None,
    also: Callable[[str], bool] | None = None,
) -> Path:
    """Download ``files`` (release asset name → local file name) of ``release`` into the cache folder ``name``, each
    checked against the release's ``SHA256SUMS``; files already there are kept. ``progress(fraction, message)`` is
    called while downloading (a job's ``update``: raising from it cancels the download). ``also(asset)`` picks
    further assets listed in ``SHA256SUMS`` (kept under their own names), e.g. a graph's external weights."""
    d = cache_dir(name)
    if have(name, files):
        return d
    d.mkdir(parents=True, exist_ok=True)
    base = f'{RELEASES}/{release}'
    try:
        sums = urllib.request.urlopen(f'{base}/SHA256SUMS', timeout=60).read().decode()
    except Exception as e:
        if getattr(e, 'code', None) == 404:
            raise FileNotFoundError(f'{release} is not published yet ({base})') from e
        raise RuntimeError(f'could not reach {base}: {e}{certs.hint(e)}') from e
    want = {ln.split()[1]: ln.split()[0] for ln in sums.splitlines() if len(ln.split()) == 2}
    if also is not None:
        files = {**files, **{n: n for n in want if also(n) and n not in files}}
    todo = [(r, loc) for r, loc in files.items() if not (d / loc).exists()]
    for i, (remote, local) in enumerate(todo):
        if remote not in want:
            raise FileNotFoundError(f'{remote} is not in {release}')
        tmp = d / f'{local}.part'
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(f'{base}/{remote}', timeout=600) as r, open(tmp, 'wb') as f:
                total = int(r.headers.get('Content-Length') or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(
                            (i + done / total) / len(todo),
                            f'downloading {remote} ({done / 2**20:,.0f} of {total / 2**20:,.0f} MB, once)',
                        )
        except BaseException as e:
            tmp.unlink(missing_ok=True)
            if isinstance(e, OSError):  # network (URLError, timeouts, TLS) or disk
                raise RuntimeError(f'download of {remote} failed: {e}{certs.hint(e)}') from e
            raise
        if h.hexdigest() != want[remote]:
            tmp.unlink(missing_ok=True)
            raise ValueError(f'{remote}: checksum mismatch (download again)')
        os.replace(tmp, d / local)
    (d / '.complete').write_text(release, encoding='utf-8')  # every file is here (the set, not just some)
    return d


def ort_session(path: Path, device: str | None = None):
    """An ONNX Runtime session on the best available device (``device``: auto | cuda | dml | cpu). Not CoreML: it
    handles the dynamic shapes of these graphs poorly, and ONNX Runtime's CPU kernels are fast on Apple silicon."""
    import onnxruntime as ort

    avail = ort.get_available_providers()
    want = {'cuda': ['CUDAExecutionProvider'], 'dml': ['DmlExecutionProvider'], 'cpu': []}.get(device or 'auto')
    if want is None:  # auto: best available
        want = [p for p in ('CUDAExecutionProvider', 'DmlExecutionProvider') if p in avail]
    so = ort.SessionOptions()
    so.log_severity_level = 3  # errors only (int8 graphs log harmless shape-merge warnings)
    return ort.InferenceSession(str(path), so, providers=[p for p in want if p in avail] + ['CPUExecutionProvider'])
