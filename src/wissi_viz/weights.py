"""Model files downloaded once from this project's GitHub releases, checked against the release's SHA-256 sums.

    d = fetch('grounding-dino-tiny-onnx-v1', {'grounding-dino-tiny-int8.onnx': 'model.onnx', 'vocab.txt': 'vocab.txt'},
              'grounding-dino-tiny-int8')

Files land in the user cache (``<cache>/models/<name>/``) and are reused from there. GitHub is reachable from most
networks, including corporate ones (for HTTPS inspection, see :mod:`wissi_viz.certs`).
"""

from __future__ import annotations

import hashlib
import os
import time
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


def download(
    url: str, dest: Path, sha256_hex: str, progress: Callable[[float, str], None] | None = None, label: str = ''
) -> Path:
    """Download one large file to ``dest``, resuming a ``.part`` left by an interrupted run, and check its SHA-256.
    ``progress(fraction, message)`` as in :func:`fetch` (raising from it stops the download; the part is kept)."""
    dest = Path(dest)
    if dest.exists() and sha256(dest) == sha256_hex:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    label = label or dest.name
    for attempt in range(4):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # the server sent the whole file: start again
                    have = 0
                total = have + int(r.headers.get('Content-Length') or 0)
                with open(tmp, 'ab' if have else 'wb') as f:
                    done = have
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if progress and total:
                            progress(
                                done / total,
                                f'downloading {label} ({done / 2**20:,.0f} of {total / 2**20:,.0f} MB, once)',
                            )
            break
        except OSError as e:
            if getattr(e, 'code', None) == 416:  # the part is already complete
                break
            if attempt == 3 or getattr(e, 'code', None) in (401, 403, 404):
                raise RuntimeError(f'download of {label} failed: {e}{certs.hint(e)}') from e
            time.sleep(2 * (attempt + 1))
    got = sha256(tmp)
    if got != sha256_hex:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'{label}: checksum mismatch ({got[:12]}…, expected {sha256_hex[:12]}…): the file at {url} '
                         'changed or the download was corrupted; try again')  # fmt: skip
    os.replace(tmp, dest)
    return dest


_PUBLISHED: dict[str, tuple[float, bool | None]] = {}


def published(release: str, timeout: float = 4.0) -> bool | None:
    """Whether ``release`` is on GitHub (its ``SHA256SUMS`` exists): True, False, or None when GitHub cannot be
    reached. A yes is remembered; a no or an unknown is asked again after ten minutes."""
    t, v = _PUBLISHED.get(release, (0.0, None))
    if v is True or (t and time.monotonic() - t < 600):
        return v
    try:  # GET, not HEAD: the asset store's signed links answer HEAD with 401 (SHA256SUMS is a few hundred bytes)
        with urllib.request.urlopen(f'{RELEASES}/{release}/SHA256SUMS', timeout=timeout) as r:
            r.read(1 << 16)
            v = True
    except OSError as e:  # HTTPError (404: not published), URLError, timeouts, TLS
        v = False if getattr(e, 'code', None) == 404 else None
    _PUBLISHED[release] = (time.monotonic(), v)
    return v


def ort_session(path: Path | bytes, device: str | None = None):
    """An ONNX Runtime session on the best available device (``device``: auto | cuda | dml | cpu). Not CoreML: it
    handles the dynamic shapes of these graphs poorly, and ONNX Runtime's CPU kernels are fast on Apple silicon.
    ``path`` may also be a serialized model (bytes)."""
    import onnxruntime as ort

    avail = ort.get_available_providers()
    want = {'cuda': ['CUDAExecutionProvider'], 'dml': ['DmlExecutionProvider'], 'cpu': []}.get(device or 'auto')
    if want is None:  # auto: best available
        want = [p for p in ('CUDAExecutionProvider', 'DmlExecutionProvider') if p in avail]
    so = ort.SessionOptions()
    so.log_severity_level = 3  # errors only (int8 graphs log harmless shape-merge warnings)
    model = path if isinstance(path, bytes) else str(path)
    return ort.InferenceSession(model, so, providers=[p for p in want if p in avail] + ['CPUExecutionProvider'])
