"""wissi-viz's tools over MCP (the Model Context Protocol; the MCP Python SDK, MIT).

* **In the app** (``wissi-viz serve --mcp``, or the desktop app's setting): streamable HTTP at ``/mcp/<secret>/``
  on the app's own server, for remote clients — ChatGPT (developer mode / connectors), Claude (custom connectors),
  any MCP client. The secret in the path is the credential (ChatGPT connectors take OAuth or no auth, not headers);
  put the server behind HTTPS (a tunnel) to reach it from those services.
* **Bridge** (``wissi-viz mcp``): stdio, for Claude Desktop and Claude Code. Calls the running app
  (``/app/agent/call``, with the token the app writes to its config folder) so what a tool does shows in the window;
  without a running app it opens its own session (no window)."""

from __future__ import annotations

import argparse
import base64
import json
from collections.abc import Callable
from typing import Any

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .._version import __version__
from ..session import Session
from .running import app_caller, find_app
from .tools import TOOLS, Host, Picture
from .tools import call as run

INSTRUCTIONS = (
    'wissi-viz is a whole-slide image viewer (histology, brightfield and fluorescence). Slides are giant images in '
    'level-0 pixels; regions are [x0, y0, x1, y1] rectangles. Start with list_slides, then list_layers / '
    'list_views. Tools that show things act on the app window the person is looking at. view_image returns a '
    'picture of a region.'
)


def _content(result: Any) -> list:
    if isinstance(result, Picture):
        return [types.ImageContent(type='image', data=base64.b64encode(result.png).decode(), mime_type='image/png'),
                types.TextContent(type='text', text=result.text)]  # fmt: skip
    if isinstance(result, dict) and result.get('image_png_b64'):
        return [types.ImageContent(type='image', data=result['image_png_b64'], mime_type='image/png'),
                types.TextContent(type='text', text=result.get('text') or '')]  # fmt: skip
    return [types.TextContent(type='text', text=json.dumps(result, ensure_ascii=False, default=str))]


def build(call: Callable[[str, dict], Any], name: str = 'wissi-viz') -> Server:
    """An MCP server whose tools are :data:`.tools.TOOLS`, run by ``call(name, args)`` (in a worker thread)."""

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, description=t.description, input_schema=t.schema(),
                       annotations=types.ToolAnnotations(read_only_hint=not (t.writes or t.ui),
                                                         destructive_hint=False, open_world_hint=False))
            for t in TOOLS.values()])  # fmt: skip

    async def call_tool(ctx, params) -> types.CallToolResult:
        args = dict(params.arguments or {})
        result = await anyio.to_thread.run_sync(lambda: call(params.name, args))
        err = isinstance(result, dict) and set(result) == {'error'}
        return types.CallToolResult(content=_content(result), is_error=err)

    return Server(name, version=__version__, instructions=INSTRUCTIONS, on_list_tools=list_tools,
                  on_call_tool=call_tool)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    """``wissi-viz mcp``: an MCP server on stdin / stdout for Claude Desktop and Claude Code."""
    ap = argparse.ArgumentParser(prog='wissi-viz mcp', description=main.__doc__)
    ap.add_argument('--app', help='URL of a running app (default: the one this user runs)')
    ap.add_argument('--token', help='its token (default: from the config folder)')
    ap.add_argument('paths', nargs='*', help='slides to open when no app is running')
    a = ap.parse_args(argv)
    found = (a.app.rstrip('/'), a.token) if a.app and a.token else find_app()
    if found:
        call = app_caller(*found)
    else:  # no window: a session of its own
        s = Session(remember=True)
        if a.paths:
            s.open(a.paths, trusted=True)
        host = Host(s)

        def call(name: str, args: dict) -> Any:
            return run(host, name, args)

    server = build(call)

    async def go():
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())

    anyio.run(go)
