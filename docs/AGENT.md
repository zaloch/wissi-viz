# The agent (trial)

wissi-viz's tools for a language model, defined once (`wissi_viz/agent/tools.py`), used three ways:

| Who drives | How | Needs |
|---|---|---|
| **The bundled local model** | the **Agent** panel in the app (header) | an OpenAI-compatible model server: Ollama by default |
| **Claude Desktop / Claude Code** | MCP over stdio: `wissi-viz mcp` | `[agent]` (the MCP Python SDK, MIT) |
| **ChatGPT, a remote Claude, any MCP client** | MCP over streamable HTTP: `wissi-viz serve --mcp` | `[agent]`, and HTTPS (a tunnel) to reach it |

What a tool does that changes the view (show slides, go to a region, show or hide a layer, split channels, open a
figure) happens in the app window you are looking at: the page follows the tools' commands
(`/app/events`, Server-Sent Events) and reports what its panes show (`/app/agent/state`).

## Tools

| Tool | Does |
|---|---|
| `list_slides` | the open slides: size, µm per pixel, brightfield or channels, metadata, which panes show them |
| `open_slides` | open files or folders |
| `show_slides` | one slide per pane |
| `go_to` | a pane to a rectangle, or a centre and a width in µm |
| `current_view` | what each pane shows |
| `view_image` | a PNG of a region (layers outlined, chosen channels): for models that see |
| `list_layers`, `layer_summary` | layers with counts and classes; per-class counts, shares, means, density |
| `show_layer` | show / hide a layer or one class |
| `set_channels` | fluorescence: channels shown, split into a pane per channel plus the merge, or merge |
| `list_analyses`, `run_analysis` | the spatial analyses (result layers are added to the slide) |
| `list_views`, `save_view` | saved views (points of interest), filtered by rules on their fields |
| `make_figure`, `export_figure` | a figure of saved views; a PNG / TIFF / PDF / SVG / PPTX file |

Errors come back to the model as `{"error": "…"}` with what exists (slide names, channels), so it can correct itself.

## The bundled model

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve      # the tools need more than the default context
ollama pull qwen3:4b
wissi-viz-app                                 # Agent (header) → ask
```

or the container: `docker compose -f deploy/agent/compose.yaml up -d` (Ollama, MIT; pulls `qwen3:4b`).
*Model…* in the panel points at any OpenAI-compatible server (llama.cpp's `llama-server`, LM Studio, vLLM) and
model. Candidates, all with permissive licences: Qwen3 4B (Apache-2.0, default), Phi-4-mini 3.8B (MIT), Granite 4
Micro 3B (Apache-2.0); below about 4 B parameters tool calling gets unreliable (see the trial).

## Claude

Claude Desktop (*Settings › Developer › Edit config*):

```json
{"mcpServers": {"wissi-viz": {"command": "wissi-viz", "args": ["mcp"]}}}
```

Claude Code: `claude mcp add wissi-viz -- wissi-viz mcp`.

`wissi-viz mcp` finds the running app (it writes its address and token to the config folder, readable by you only,
and removes them on exit) and calls its tools there, so they act in your window; with no app running it opens its
own session (`wissi-viz mcp slide.svs …`).

## ChatGPT and other remote clients

```bash
wissi-viz serve /data/slides --mcp            # prints http://127.0.0.1:8010/mcp/<secret>/
```

The secret path is the password (ChatGPT's connectors take OAuth or no authentication, not a header). Reach it over
HTTPS — a tunnel such as `cloudflared tunnel --url http://127.0.0.1:8010` — and add
`https://<tunnel>/mcp/<secret>/` as a connector (ChatGPT: developer mode). Anyone with the address can use the tools:
keep it private, and use `--key` for opening files.

## The trial

`scripts/agent_trial.py --model qwen3:4b` builds a test session (two brightfield slides with cells, a fluorescence
slide, saved views) and asks ten things a person might ask — count slides, list layers, count a class, show a slide,
go to a region, hide a layer, filter saved views, make a figure, split channels, save a view — each checked by what
the model did and answered. `.github/workflows/agent-trial.yml` runs it for several models on Ollama (CPU).
