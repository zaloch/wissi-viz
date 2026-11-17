"""The bundled agent: a small open model on an OpenAI-compatible server (Ollama by default; llama.cpp's server, LM
Studio, vLLM work too) runs wissi-viz's tools. Plain HTTP (no client library): ``/v1/models`` to check, and
``/v1/chat/completions`` with ``tools`` for each step.

Small models (≈ 4 B parameters) call tools reliably only with short, precise tool descriptions and a short
loop; :func:`chat` stops after ``max_steps`` tool rounds, reads tool calls written as text by models that do not
use the tool-call field, and drops ``<think>`` blocks."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .._compat import user_dir
from .tools import Host, Picture, call, schemas

DEFAULT = {'base_url': 'http://127.0.0.1:11434/v1', 'model': 'qwen3:4b', 'api_key': '', 'max_steps': 8,
           'temperature': 0.2}  # fmt: skip
#: models worth trying (all permissive licences); Ollama tags
MODELS = [
    {'tag': 'qwen3:4b', 'name': 'Qwen3 4B', 'licence': 'Apache-2.0', 'size_gb': 2.5},
    {'tag': 'phi4-mini', 'name': 'Phi-4-mini 3.8B', 'licence': 'MIT', 'size_gb': 2.5},
    {'tag': 'granite4:micro', 'name': 'Granite 4 Micro 3B', 'licence': 'Apache-2.0', 'size_gb': 2.1},
    {'tag': 'qwen3:1.7b', 'name': 'Qwen3 1.7B', 'licence': 'Apache-2.0', 'size_gb': 1.4},
]
SYSTEM = (
    'You operate wissi-viz, a whole-slide image viewer, for the person you talk to. Use the tools to answer and to '
    'act; do not invent slide names, layers or numbers: look them up (list_slides, list_layers, list_views). '
    'Regions are [x0, y0, x1, y1] in level-0 pixels. Call one tool at a time when the next depends on the result. '
    'Answer briefly, with the numbers the tools gave.'
)


def settings_file():
    return user_dir('config') / 'agent-llm.json'


def load_settings() -> dict:
    try:
        d = json.loads(settings_file().read_text(encoding='utf-8'))
        return {**DEFAULT, **{k: v for k, v in d.items() if k in DEFAULT}}
    except (OSError, ValueError):
        return dict(DEFAULT)


def save_settings(d: dict) -> dict:
    cur = {**load_settings(), **{k: v for k, v in (d or {}).items() if k in DEFAULT}}
    p = settings_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur), encoding='utf-8')
    return cur


def _post(cfg: dict, path: str, body: dict | None = None, timeout: float = 300) -> dict:
    url = cfg['base_url'].rstrip('/') + path
    headers = {'Content-Type': 'application/json'}
    if cfg.get('api_key'):
        headers['Authorization'] = f'Bearer {cfg["api_key"]}'
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method='POST' if body is not None else 'GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise ConnectionError(f'{url}: {e.code} {e.read().decode("utf-8", "replace")[:300]}') from e
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(f'{url}: {getattr(e, "reason", e)}') from e


def status(cfg: dict | None = None) -> dict:
    """Whether the model server answers, and which models it has."""
    cfg = cfg or load_settings()
    try:
        r = _post(cfg, '/models', None, timeout=5)
    except ConnectionError as e:
        return {'ok': False, 'error': str(e), 'settings': _public(cfg), 'suggested': MODELS}
    have = [m.get('id') for m in r.get('data') or []]
    return {'ok': True, 'models': have, 'has_model': cfg['model'] in have or any(h.startswith(cfg['model']) for h in have),
            'settings': _public(cfg), 'suggested': MODELS}  # fmt: skip


def _public(cfg: dict) -> dict:
    return {**cfg, 'api_key': bool(cfg.get('api_key'))}


_THINK = re.compile(r'<think>.*?</think>\s*', re.S)
_CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def text_calls(content: str) -> list[dict]:
    """Tool calls a model wrote as text: ``<tool_call>{…}</tool_call>`` (Qwen), or a bare JSON object."""
    found = [m.group(1) for m in _CALL.finditer(content or '')]
    s = (content or '').strip()
    if not found and s.startswith('{') and s.endswith('}'):
        found = [s]
    out = []
    for i, raw in enumerate(found):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        name = d.get('name') or d.get('function') or d.get('tool')
        args = d.get('arguments') or d.get('parameters') or d.get('args') or {}
        if isinstance(name, str):
            out.append(
                {'id': f'text{i}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
            )
    return out


def _brief(result: Any) -> Any:
    if isinstance(result, Picture):
        return {'picture': result.text, 'note': 'a picture: shown to the person, not to you'}
    return result


def chat(
    host: Host, messages: list[dict], cfg: dict | None = None, on_step: Callable[[dict], None] | None = None
) -> dict:
    """One turn: the model answers, calling tools as it needs. ``messages``: the conversation so far
    ([{role: user | assistant, content}]). Returns {'reply', 'steps': [{'tool', 'args', 'result'}], 'model'}."""
    cfg = cfg or load_settings()
    msgs: list[dict] = [{'role': 'system', 'content': SYSTEM}]
    msgs += [
        {'role': m['role'], 'content': str(m.get('content') or '')}
        for m in messages
        if m.get('role') in ('user', 'assistant')
    ]
    steps: list[dict] = []
    tools = schemas()
    for _ in range(int(cfg.get('max_steps') or 8)):
        r = _post(cfg, '/chat/completions', {'model': cfg['model'], 'messages': msgs, 'tools': tools,
                                             'tool_choice': 'auto', 'temperature': cfg.get('temperature', 0.2)})  # fmt: skip
        m = (r.get('choices') or [{}])[0].get('message') or {}
        content = _THINK.sub('', m.get('content') or '')
        calls = m.get('tool_calls') or text_calls(content)
        if not calls:
            return {'reply': content.strip(), 'steps': steps, 'model': r.get('model') or cfg['model']}
        msgs.append({'role': 'assistant', 'content': content if m.get('tool_calls') else '', 'tool_calls': calls})
        for c in calls:
            fn = c.get('function') or {}
            name = fn.get('name') or ''
            try:
                args = fn.get('arguments') or {}
                args = json.loads(args) if isinstance(args, str) else dict(args)
            except (ValueError, TypeError):
                args = {}
            res = call(host, name, args)
            step = {'tool': name, 'args': args, 'result': _brief(res)}
            if isinstance(res, Picture):
                step['image'] = res
            steps.append(step)
            if on_step:
                on_step(step)
            msgs.append({'role': 'tool', 'tool_call_id': c.get('id') or name, 'name': name,
                         'content': json.dumps(_brief(res), ensure_ascii=False, default=str)[:8000]})  # fmt: skip
    return {'reply': f'Stopped after {len(steps)} tool calls (the step limit): ask again to go on.', 'steps': steps,
            'model': cfg['model']}  # fmt: skip
