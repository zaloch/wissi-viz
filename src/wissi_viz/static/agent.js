// The agent panel (app page): chat with the bundled local model (an OpenAI-compatible server such as Ollama), which
// runs wissi-viz's tools (wissi_viz/agent); how to connect Claude or ChatGPT over MCP instead. The page also follows
// the tools' commands (Server-Sent Events from /app/events) and reports what its panes show (/app/agent/state).

let X = null;
const A = { open: false, msgs: [], busy: false, status: null, es: null, stateTimer: 0 };
const $ = (id) => document.getElementById(id);

export function initAgent(ctx) {
  X = ctx;
  $('b-agent')?.addEventListener('click', () => (A.open ? close() : open()));
  listen();
  X.grid.addEventListener('views', reportSoon);
  X.grid.addEventListener('activepane', reportSoon);
  return { open, close, isOpen: () => A.open };
}

// ------------------------------------------------------------------------------------- commands from the tools
function listen() {
  if (!X.TOKEN || typeof EventSource === 'undefined') return;
  const es = new EventSource(`${X.BASE}/app/events?token=${encodeURIComponent(X.TOKEN)}`);
  A.es = es;
  es.onmessage = (e) => { try { run(JSON.parse(e.data)); } catch (err) { console.warn('agent command', err); } };
}
function run(c) {
  if (c.type === 'show') X.showSlides(c.slides);
  else if (c.type === 'goto') X.goTo(c.slide, c.bounds, c.pane);
  else if (c.type === 'layer') X.setLayer(c.slide, c.layer, c.visible, c.class);
  else if (c.type === 'channels') X.setChannels(c.slide, c.show, c.split);
  else if (c.type === 'slides') X.onResult({ slides: c.slides });
  else if (c.type === 'layers') X.refreshLayers(c.slide);
  else if (c.type === 'views') X.loadViews();
  else if (c.type === 'figure') X.openFigure(c.id);
  if (c.type !== 'slides' && c.type !== 'views') X.toast({ title: `Agent: ${label(c)}`, ms: 2500 });
}
function label(c) {
  return { show: 'showing slides', goto: 'going to a region', layer: `${c.visible ? 'showing' : 'hiding'} ${c.layer}`,
    channels: c.split === true ? 'splitting channels' : c.split === false ? 'merging channels' : 'channels', layers: 'new layers',
    figure: 'a figure' }[c.type] || c.type;
}
function reportSoon() {
  clearTimeout(A.stateTimer);
  A.stateTimer = setTimeout(() => {
    const views = X.grid.getViews?.() || [];
    const panes = (X.grid.panes || []).map((p, i) => (p.slide?.info ? { slide: p.slide.info.id, name: p.slide.info.name, bounds: views[i]?.bounds?.map((v) => Math.round(v)), mpp: views[i]?.mpp ?? null } : null));
    X.http(`${X.BASE}/app/agent/state`, { panes, active: X.active() }).catch(() => {});
  }, 400);
}

// --------------------------------------------------------------------------------------------------- the panel
async function open() {
  A.open = true;
  $('agent').hidden = false;
  $('b-agent')?.classList.add('on');
  render();
  refreshStatus();
  setTimeout(() => $('ag-input')?.focus(), 50);
}
function close() {
  A.open = false;
  $('agent').hidden = true;
  $('b-agent')?.classList.remove('on');
}
async function refreshStatus() {
  try { A.status = await X.http(`${X.BASE}/app/agent/status`); } catch (e) { A.status = { ok: false, error: e.message }; }
  render();
}
function render() {
  const h = X.h;
  const box = $('agent');
  if (!A.open) return;
  const st = A.status;
  const ok = st?.ok && st.has_model;
  const chip = h('span', { id: 'ag-status', class: 'chip' + (ok ? '' : ' warn'), title: st?.error || '' },
    !st ? 'Checking…' : !st.ok ? 'No model server' : !st.has_model ? `Model missing: ${st.settings.model}` : `${st.settings.model}`);
  const set = h('button', { id: 'ag-settings', class: 'btn sm', title: 'Model server and model' }, 'Model…');
  set.onclick = () => { A.showSettings = !A.showSettings; render(); };
  const conn = h('button', { id: 'ag-connect', class: 'btn sm', title: 'Use Claude or ChatGPT instead: connect them over MCP' }, 'Claude / ChatGPT…');
  conn.onclick = () => { A.showConnect = !A.showConnect; render(); };
  const x = h('button', { id: 'ag-close', class: 'btn icon ghost', title: 'Close the agent', 'aria-label': 'Close the agent' }, '✕');
  x.onclick = close;
  const kids = [h('div', { class: 'row' }, h('b', {}, 'Agent'), chip, h('span', { class: 'grow' }), set, conn, x)];
  if (A.showSettings) kids.push(settingsBox());
  if (A.showConnect) kids.push(connectBox());
  if (st && !ok && !A.showSettings) {
    kids.push(h('div', { class: 'panel-note warn' }, !st.ok
      ? `No model server at ${st.settings?.base_url}. Start Ollama (ollama serve) and pull a model: ollama pull ${st.settings?.model} · Or Model… for another server`
      : `Pull the model: ollama pull ${st.settings.model} · Or Model… to choose one of: ${(st.models || []).slice(0, 4).join(', ')}`));
  }
  const log = h('div', { id: 'ag-log', role: 'log', 'aria-live': 'polite' });
  if (!A.msgs.length) log.append(h('div', { class: 'panel-note' }, 'Ask about the open slides, or what to do: “Which layers does this slide have?” · “Show the tumour region of slide 2” · “Make a figure of the Core views by Treatment”'));
  for (const m of A.msgs) log.append(msgEl(m));
  if (A.busy) log.append(h('div', { class: 'ag-msg ag-wait' }, 'Working…'));
  const inp = h('textarea', { id: 'ag-input', rows: 2, placeholder: 'Ask or tell the agent · Enter: send · Shift+Enter: new line', 'aria-label': 'Message to the agent' });
  inp.value = A.draft || '';
  inp.oninput = () => { A.draft = inp.value; };
  inp.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } };
  const go = h('button', { id: 'ag-send', class: 'btn primary', disabled: A.busy }, 'Send');
  go.onclick = send;
  const clear = h('button', { id: 'ag-clear', class: 'btn sm ghost', title: 'Start over', disabled: !A.msgs.length || A.busy }, 'Clear');
  clear.onclick = () => { A.msgs = []; render(); };
  kids.push(log, h('div', { class: 'row', style: 'align-items: flex-end' }, inp, h('div', { style: 'display: flex; flex-direction: column; gap: 4px' }, go, clear)));
  box.replaceChildren(...kids);
  log.scrollTop = log.scrollHeight;
}
function msgEl(m) {
  const h = X.h;
  if (m.role === 'user') return h('div', { class: 'ag-msg ag-user' }, m.content);
  const el = h('div', { class: 'ag-msg ag-bot' + (m.error ? ' ag-err' : '') });
  for (const st of m.steps || []) {
    const res = JSON.stringify(st.result);
    const d = h('details', { class: 'ag-step' }, h('summary', {}, `${st.result?.error ? '⚠ ' : ''}${st.tool}`, h('span', { class: 'dim mono' }, ` ${JSON.stringify(st.args).slice(0, 80)}`)),
      h('pre', {}, res.length > 1500 ? `${res.slice(0, 1500)}…` : res));
    if (st.image) d.append(h('img', { src: st.image, alt: st.result?.picture || 'picture' }));
    el.append(d);
  }
  el.append(h('div', { class: 'ag-text' }, m.content || ''));
  return el;
}
async function send() {
  const text = (A.draft || '').trim();
  if (!text || A.busy) return;
  A.msgs.push({ role: 'user', content: text });
  A.draft = '';
  A.busy = true;
  render();
  try {
    const r = await X.http(`${X.BASE}/app/agent/chat`, { messages: A.msgs.map((m) => ({ role: m.role, content: m.content })) });
    A.msgs.push({ role: 'assistant', content: r.reply, steps: r.steps });
  } catch (e) {
    A.msgs.push({ role: 'assistant', content: e.message, error: true });
  } finally {
    A.busy = false;
    render();
  }
}
function settingsBox() {
  const h = X.h;
  const s = A.status?.settings || {};
  const base = h('input', { id: 'ag-base', value: s.base_url || '', 'aria-label': 'Model server URL (OpenAI-compatible)' });
  const have = A.status?.models || [];
  const opts = [...new Set([...have, ...(A.status?.suggested || []).map((m) => m.tag), s.model].filter(Boolean))];
  const model = h('select', { id: 'ag-model', 'aria-label': 'Model' }, opts.map((m) => {
    const sug = (A.status?.suggested || []).find((x) => x.tag === m);
    return h('option', { value: m, selected: m === s.model }, `${m}${sug ? ` · ${sug.licence}` : ''}${have.includes(m) ? '' : ' (not pulled)'}`);
  }));
  const save = h('button', { id: 'ag-save', class: 'btn sm primary' }, 'Save');
  save.onclick = async () => {
    try {
      await X.http(`${X.BASE}/app/agent/settings`, { base_url: base.value.trim(), model: model.value });
      A.showSettings = false;
      refreshStatus();
    } catch (e) { X.toast({ kind: 'bad', title: 'Not saved', sub: e.message }); }
  };
  return h('div', { class: 'ag-box' }, h('div', { class: 'f2' }, 'Server', base, 'Model', model),
    h('div', { class: 'panel-note' }, 'Any OpenAI-compatible server: Ollama (default; start it with OLLAMA_CONTEXT_LENGTH=16384: the tools need room), llama.cpp, LM Studio, vLLM. Models of about 4 B parameters call tools reliably; smaller ones often do not.'),
    h('div', { class: 'row' }, h('span', { class: 'grow' }), save));
}
function connectBox() {
  const h = X.h;
  const cfg = JSON.stringify({ mcpServers: { 'wissi-viz': { command: 'wissi-viz', args: ['mcp'] } } }, null, 2);
  const copy = (t) => () => navigator.clipboard?.writeText(t).then(() => X.toast({ title: 'Copied', ms: 1500 }));
  const b1 = h('button', { id: 'ag-copy-claude', class: 'btn sm' }, 'Copy');
  b1.onclick = copy(cfg);
  const b2 = h('button', { id: 'ag-copy-code', class: 'btn sm' }, 'Copy');
  b2.onclick = copy('claude mcp add wissi-viz -- wissi-viz mcp');
  return h('div', { class: 'ag-box' },
    h('b', {}, 'Claude Desktop'), h('div', { class: 'panel-note' }, 'Settings › Developer › Edit config: add this, restart Claude. Its tools then act in this window.'),
    h('pre', { class: 'mono' }, cfg), h('div', { class: 'row' }, h('span', { class: 'grow' }), b1),
    h('b', {}, 'Claude Code'), h('pre', { class: 'mono' }, 'claude mcp add wissi-viz -- wissi-viz mcp'), h('div', { class: 'row' }, h('span', { class: 'grow' }), b2),
    h('b', {}, 'ChatGPT, or a remote Claude'), h('div', { class: 'panel-note' }, 'Run wissi-viz serve --mcp (it prints the address: the path is the password), reach it over HTTPS (a tunnel), add it as a connector (developer mode). Needs the [agent] extra.'));
}
