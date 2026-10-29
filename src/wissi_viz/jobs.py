"""Background jobs for long runs started from the app (find all, segment everything, area classification): progress,
a message, cancellation and the result, polled over HTTP (``GET /jobs/{id}``)."""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised inside a job when the user cancelled it."""


class Job:
    def __init__(self, jid: str, kind: str, slide: str | None, title: str) -> None:
        self.id = jid
        self.kind = kind
        self.slide = slide
        self.title = title
        self.state = 'queued'  # queued | running | done | error | cancelled
        self.progress = 0.0
        self.message = ''
        self.result: Any = None
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self._cancel = threading.Event()

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        """Report progress (0–1) and/or a message; raises :class:`Cancelled` once the user cancelled."""
        if progress is not None:
            self.progress = float(min(1.0, max(0.0, progress)))
        if message is not None:
            self.message = message
        if self._cancel.is_set():
            raise Cancelled()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'kind': self.kind,
            'slide': self.slide,
            'title': self.title,
            'state': self.state,
            'progress': round(self.progress, 4),
            'message': self.message,
            'error': self.error,
            'result': self.result,
            'seconds': round((self.finished or time.time()) - self.started, 1),
        }


class JobRunner:
    """Runs jobs on a small thread pool and keeps the most recent ones for polling."""

    def __init__(self, workers: int = 2, keep: int = 64) -> None:
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix='wissi-job')
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._keep = keep
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Job], Any], slide: str | None = None, title: str = '') -> Job:
        with self._lock:
            job = Job(f'j{next(self._ids)}-{int(time.time() * 1000) % 100000}', kind, slide, title or kind)
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                old = next(iter(self._jobs.values()))
                if old.state in ('queued', 'running'):
                    break
                self._jobs.popitem(last=False)

        def run() -> None:
            job.state = 'running'
            try:
                job.result = fn(job)
                job.state = 'cancelled' if job.cancelled else 'done'
                job.progress = 1.0 if job.state == 'done' else job.progress
            except Cancelled:
                job.state = 'cancelled'
            except (ValueError, PermissionError, FileNotFoundError) as e:  # for the user (no examples, nothing
                # found, a licence to accept, a model not published), not a bug
                job.state = 'error'
                job.error = str(e)
            except Exception as e:  # reported to the page through the job, and logged
                log.exception('job %s failed', job.id)
                job.state = 'error'
                job.error = f'{type(e).__name__}: {e}'
            job.finished = time.time()

        self._pool.submit(run)
        return job

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def list(self, slide: str | None = None) -> list[Job]:
        return [j for j in self._jobs.values() if slide is None or j.slide == slide]
