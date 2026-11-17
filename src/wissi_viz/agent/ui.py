"""What tools do to the app window: commands to the open pages (Server-Sent Events, ``/app/events``) and what the
pages report back (the panes and what they show, ``/app/agent/state``)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator


class UiBus:
    """Commands from tools (any thread) to every open app page; the pages' last reported state."""

    def __init__(self) -> None:
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()
        self.state: dict = {}
        self.state_time = 0.0

    @property
    def listening(self) -> bool:
        with self._lock:
            return bool(self._subs)

    def send(self, cmd: dict) -> bool:
        """Send a command to every open page; False when no page is open (the tool still did its part)."""
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, cmd)
            except RuntimeError:  # that page's loop is gone
                pass
        return bool(subs)

    def report(self, state: dict) -> None:
        self.state = state if isinstance(state, dict) else {}
        self.state_time = time.time()

    async def stream(self, keepalive: float = 20.0) -> AsyncIterator[str]:
        """One page's events, as SSE lines; a comment every ``keepalive`` s keeps proxies from closing it."""
        q: asyncio.Queue = asyncio.Queue()
        sub = (asyncio.get_running_loop(), q)
        with self._lock:
            self._subs.append(sub)
        try:
            yield ': hello\n\n'
            while True:
                try:
                    cmd = await asyncio.wait_for(q.get(), keepalive)
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
                    continue
                yield f'data: {json.dumps(cmd, ensure_ascii=False)}\n\n'
        finally:
            with self._lock:
                if sub in self._subs:
                    self._subs.remove(sub)
