"""The agent: wissi-viz's tools (wissi_viz.agent.tools), served over MCP (mcp_server) and run by a local model (llm),
and what they do to the app window (ui)."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import tifffile

from wissi_viz.agent import llm, tools
from wissi_viz.agent.ui import UiBus


@pytest.fixture()
def session(tmp_path):
    from wissi_viz.annotations import AnnotationLayer
    from wissi_viz.session import Session

    W, H = 1200, 900
    img = np.full((H, W, 3), 230, np.uint8)
    img[200:400, 300:600] = (120, 40, 120)
    desc = f'Aperio Image Library v12 \r\n{W}x{H} [0,0 {W}x{H}] (256x256) JPEG/RGB Q=90|AppMag = 20|MPP = 0.5'
    with tifffile.TiffWriter(tmp_path / 'A1.svs', bigtiff=True) as tw:
        tw.write(img, description=desc, tile=(256, 256), compression='zlib', photometric='rgb', metadata=None)
    s = Session(roots=[tmp_path], remember=False, cache_dir=tmp_path / 'cache')
    sid = s.open([tmp_path / 'A1.svs'])['opened'][0]
    L = AnnotationLayer(name='cells', classes={'Tumour': '#ff0000', 'Immune': '#00ff00'})
    for i in range(30):
        L.add_point(310 + 9 * i, 210 + 5 * i, cls='Tumour' if i % 3 else 'Immune')
    s.registry.add_annotations(sid, L, name='cells')
    s.add_view(sid, [300, 200, 600, 400], 'Core', {'Region': 'Core', 'Score': 3})
    s.add_view(sid, [0, 0, 300, 200], 'Edge', {'Region': 'Edge', 'Score': 1})
    yield s
    s.shutdown()


class _Spy(UiBus):
    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, cmd):
        self.sent.append(cmd)
        return True


def test_tools_read_act_and_explain_their_errors(session, tmp_path):
    ui = _Spy()
    h = tools.Host(session, ui, exports=tmp_path / 'out')
    sl = tools.call(h, 'list_slides', {})['slides']
    assert sl[0]['name'] == 'A1.svs' and sl[0]['mpp'] == 0.5 and sl[0]['mode'] == 'rgb'
    assert tools.call(h, 'list_layers', {'slide': 'A1'})['layers'][0] == {
        'name': 'cells', 'objects': 30, 'classes': {'Tumour': 20, 'Immune': 10}, 'origin': None}  # fmt: skip
    summ = tools.call(h, 'layer_summary', {'slide': 'A1.svs', 'layer': 'cells'})
    assert summ['n'] == 30 if 'n' in summ else json.dumps(summ).count('Tumour') >= 1
    r = tools.call(h, 'go_to', {'slide': 'A1', 'x': 600, 'y': 450, 'width_um': 100})
    assert r['bounds'] == [500.0, 375.0, 700.0, 525.0] and ui.sent[-1]['type'] == 'goto'
    tools.call(h, 'show_layer', {'slide': 'A1', 'layer': 'cells', 'visible': False})
    assert ui.sent[-1] == {'type': 'layer', 'slide': sl[0]['id'], 'layer': 'cells', 'visible': False, 'class': None}
    pic = tools.call(h, 'view_image', {'slide': 'A1', 'bounds': [300, 200, 600, 400], 'size': 120, 'layers': ['cells']})
    assert isinstance(pic, tools.Picture) and pic.png[:4] == b'\x89PNG' and '150 µm wide' in pic.text
    assert 'brightfield' in tools.call(h, 'set_channels', {'slide': 'A1', 'split': True})['error']
    assert 'no open slide' in tools.call(h, 'list_layers', {'slide': 'nope'})['error']
    assert (
        'no tool' in tools.call(h, 'fly', {})['error']
        and 'unexpected' in tools.call(h, 'list_slides', {'x': 1})['error']
    )
    # saved views → a figure → a file
    vs = tools.call(h, 'list_views', {'rules': [{'field': 'Region', 'op': 'in', 'values': ['Core']}]})
    assert [v['name'] for v in vs['views']] == ['Core'] and 'Score' in vs['fields']
    vs = tools.call(h, 'list_views', {'rules': [{'field': 'region', 'op': 'equals', 'values': 'core'}]})
    assert [v['name'] for v in vs['views']] == ['Core']  # field-name case, op synonyms, a scalar value
    none = tools.call(h, 'list_views', {'rules': [{'field': 'Region', 'values': ['Middle']}]})
    assert none['views'] == [] and 'Region: Core, Edge' in none['note']
    v = tools.call(
        h, 'save_view', {'slide': 'A1', 'bounds': [600, 400, 900, 600], 'name': 'Third', 'fields': {'Region': 'Core'}}
    )
    assert v['view']['name'] == 'Third' and ui.sent[-1] == {'type': 'views'}
    f = tools.call(h, 'make_figure', {'name': 'Fig A', 'rules': [{'field': 'Region', 'values': ['Core', 'Edge']}],
                                      'rows_by': 'Region', 'chart_field': 'Score'})  # fmt: skip
    assert len(f['views']) == 3 and ui.sent[-1]['type'] == 'figure'
    pytest.importorskip('matplotlib')
    out = tools.call(h, 'export_figure', {'figure': 'Fig A', 'format': 'pdf', 'dpi': 72})
    assert out['path'].endswith('Fig A.pdf') and open(out['path'], 'rb').read(4) == b'%PDF'
    assert len(tools.schemas()) == len(tools.TOOLS) and all(
        s['function']['parameters']['type'] == 'object' for s in tools.schemas()
    )


def test_local_model_loop_runs_tools(session, monkeypatch):
    """A scripted model: a tool call in the tool-call field, one written as text (Qwen style), then the answer."""
    seen = []
    replies = iter([
        {'choices': [{'message': {'content': '<think>which slides?</think>', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'list_slides', 'arguments': '{}'}}]}}]},
        {'choices': [{'message': {'content': '<tool_call>{"name": "list_layers", "arguments": {"slide": "A1"}}</tool_call>'}}]},
        {'choices': [{'message': {'content': '<think>done</think>A1.svs has one layer, cells (30 objects).'}}], 'model': 'qwen3:4b'},
    ])  # fmt: skip

    def fake(cfg, path, body=None, timeout=900):
        seen.append(body)
        return next(replies)

    monkeypatch.setattr(llm, '_post', fake)
    out = llm.chat(
        tools.Host(session, _Spy()), [{'role': 'user', 'content': 'What layers are there?'}], dict(llm.DEFAULT)
    )
    assert out['reply'] == 'A1.svs has one layer, cells (30 objects).' and out['model'] == 'qwen3:4b'
    assert [s['tool'] for s in out['steps']] == ['list_slides', 'list_layers']
    assert seen[0]['tools'][0]['function']['name'] == 'list_slides' and seen[0]['messages'][0]['role'] == 'system'
    assert seen[2]['messages'][-1]['role'] == 'tool' and 'cells' in seen[2]['messages'][-1]['content']
    assert llm.text_calls('{"name": "go_to", "arguments": {"slide": "A1"}}')[0]['function']['name'] == 'go_to'
    names = {'go_to', 'list_layers'}
    fn = [c['function'] for t in ('list_layers {"slide": "A1"}',
                                  '[{"type": "function", "function": {"name": "go_to", "arguments": {"slide": "A1"}}}]')
          for c in llm.text_calls(t, names)]  # fmt: skip
    assert [f['name'] for f in fn] == ['list_layers', 'go_to'] and json.loads(fn[1]['arguments']) == {'slide': 'A1'}
    echoed = '{"type": "function", "function": {"name": "go_to", "description": "x", "parameters": {"type": "object", "properties": {}}}}'
    assert llm.text_calls(echoed, names) == [] and llm.text_calls('{"name": "fly", "arguments": {}}', names) == []
    assert llm.text_calls('There are 3 slides.', names) == []
    monkeypatch.setattr(llm, '_post', lambda *a, **k: (_ for _ in ()).throw(ConnectionError('refused')))
    assert llm.status(dict(llm.DEFAULT))['ok'] is False


def test_mcp_server_in_process(session):
    pytest.importorskip('mcp')
    from mcp import Client

    from wissi_viz.agent import mcp_server

    h = tools.Host(session, _Spy())
    srv = mcp_server.build(lambda name, args: tools.call(h, name, args))

    async def go():
        async with Client(srv) as c:
            ts = (await c.list_tools()).tools
            assert {t.name for t in ts} == set(tools.TOOLS)
            r = await c.call_tool('list_layers', {'slide': 'A1'})
            assert json.loads(r.content[0].text)['layers'][0]['objects'] == 30 and not r.is_error
            r = await c.call_tool('view_image', {'slide': 'A1', 'size': 64})
            assert r.content[0].type == 'image' and r.content[0].mime_type == 'image/png'
            r = await c.call_tool('list_layers', {'slide': 'nope'})
            assert r.is_error

    asyncio.run(go())


def test_app_routes_events_and_mcp_over_http(session, monkeypatch, tmp_path):
    pytest.importorskip('mcp')
    from starlette.testclient import TestClient

    from wissi_viz.session import create_session_app

    monkeypatch.setattr(llm, 'settings_file', lambda: tmp_path / 'agent-llm.json')
    app = create_session_app(session, mode='server', key='k', mcp='s3cret')
    h = {'x-wissi-token': session.token, 'x-wissi-key': 'k'}
    with TestClient(app) as c:
        assert len(c.get('/app/agent/tools', headers=h).json()['tools']) == len(tools.TOOLS)
        r = c.post(
            '/app/agent/call', json={'tool': 'view_image', 'args': {'slide': 'A1', 'size': 64}}, headers=h
        ).json()
        assert r['image_png_b64'] and 'A1.svs' in r['text']
        assert c.post('/app/agent/call', json={'tool': 'list_slides'}).status_code == 403
        c.post('/app/agent/state', json={'panes': [{'slide': 'x', 'bounds': [0, 0, 5, 5]}], 'active': 0}, headers=h)
        assert c.post('/app/agent/call', json={'tool': 'current_view'}, headers=h).json()['active'] == 0
        assert (
            c.post('/app/agent/settings', json={'model': 'phi4-mini'}, headers=h).json()['settings']['model']
            == 'phi4-mini'
        )
        monkeypatch.setattr(
            llm, '_post', lambda cfg, path, body=None, timeout=300: {'choices': [{'message': {'content': 'Hi.'}}]}
        )
        assert (
            c.post('/app/agent/chat', json={'messages': [{'role': 'user', 'content': 'hi'}]}, headers=h).json()['reply']
            == 'Hi.'
        )
        assert c.get('/app/events', params={'token': 'bad'}).status_code == 403
        # MCP over streamable HTTP at /mcp/<secret>/
        mh = {'Accept': 'application/json, text/event-stream', 'Content-Type': 'application/json'}
        init = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18',
                'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '1'}}}  # fmt: skip
        r = c.post('/mcp/s3cret/', json=init, headers=mh)
        assert r.status_code == 200, r.text
        assert r.json()['result']['serverInfo']['name'] == 'wissi-viz'
        r = c.post('/mcp/s3cret/', json={'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                   headers={**mh, 'mcp-protocol-version': '2025-06-18'})  # fmt: skip
        assert len(r.json()['result']['tools']) == len(tools.TOOLS)
        assert c.post('/mcp/wrong/', json=init, headers=mh).status_code == 404


def test_page_commands_stream():
    bus = UiBus()

    async def go():
        gen = bus.stream(keepalive=5)
        assert await gen.__anext__() == ': hello\n\n'
        assert bus.listening and bus.send({'type': 'show', 'slides': ['a']})
        line = await gen.__anext__()
        await gen.aclose()
        return line

    assert json.loads(asyncio.run(go())[6:]) == {'type': 'show', 'slides': ['a']}
    assert not bus.listening and not bus.send({'type': 'x'})


def test_running_app_is_found(monkeypatch, tmp_path):
    from wissi_viz.agent import running

    monkeypatch.setattr(running, 'running_file', lambda: tmp_path / 'agent.json')
    running.announce('http://127.0.0.1:9', 'tok')
    assert json.loads((tmp_path / 'agent.json').read_text())['token'] == 'tok'
    assert running.find_app() is None  # nothing answers there
    running.withdraw()
    assert not (tmp_path / 'agent.json').exists()
