"""The agent: wissi-viz's tools for a language model — the bundled local one (an OpenAI-compatible server such as
Ollama), or Claude / ChatGPT through MCP. The tools are defined once (:mod:`.tools`); :mod:`.mcp_server` serves them
over MCP (stdio for Claude Desktop / Claude Code, streamable HTTP for remote clients); :mod:`.llm` runs them with a
local model; :mod:`.ui` carries what a tool does to the app window (show a slide, go to a region, …)."""

from .tools import TOOLS, Host, call, schemas

__all__ = ['TOOLS', 'Host', 'call', 'schemas']
