// wissi-viz app page (assets/desktop.html): the slide rail, the inspector (Layers / Analyze / Measure / Draw)
// and the overlays on the slide (object card, hover tip, radial menu, toasts, key hints).
// Talks to the viewer element (<wissi-viz>, static/index.js), the tile API under `api`, and the app's own
// token-protected endpoints (/app/*, see wissi_viz/session.py) or the desktop window's native bridge.

const cfg = window.WISSI_APP || {};
const API = cfg.api, BASE = cfg.base || '', TOKEN = cfg.token, MODE = cfg.mode;   // BASE: where the app is mounted
const SERVER = MODE === 'server';
const CAPS = Object.assign({ open: !SERVER, browse: false, connect: false, close: !SERVER, key: false }, cfg.caps || {});
const MAX_PANES = 32;
const $ = (id) => document.getElementById(id);
const grid = $('grid');
const isMac = navigator.platform.toUpperCase().includes('MAC');
let slides = [];   // [{id, name, path}]
let panes = [];    // slide id per pane
let active = 0;

// ------------------------------------------------------------------------------------------------ small helpers
/** Element builder: text always goes in as text (names come from files and other people). */
function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') e[k] = v;
    else if (k === 'value' || k === 'checked' || k === 'disabled' || k === 'hidden' || k === 'selected') e[k] = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : String(c));
  return e;
}
const SVGNS = 'http://www.w3.org/2000/svg';
function s(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : String(c));
  return e;
}
/** An icon from our own path data (never from data). */
function icon(name, size = 16) {
  const d = ICONS[name] || '';
  const e = s('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8,
    'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' });
  for (const part of d.split('|')) {
    const [kind, ...rest] = part.split(' ');
    const a = rest.join(' ');
    if (kind === 'p') e.append(s('path', { d: a }));
    else if (kind === 'c') { const [cx, cy, r, f] = a.split(','); e.append(s('circle', { cx, cy, r, fill: f ? 'currentColor' : 'none' })); }
    else if (kind === 'r') { const [x, y, w, hh] = a.split(','); e.append(s('rect', { x, y, width: w, height: hh, rx: 1.5 })); }
    else if (kind === 'e') { const [cx, cy, rx, ry] = a.split(','); e.append(s('ellipse', { cx, cy, rx, ry })); }
  }
  return e;
}
const ICONS = {
  pan: 'p M7 11V6a1.5 1.5 0 013 0v5M10 10V4.5a1.5 1.5 0 013 0V10M13 10V5.5a1.5 1.5 0 013 0V12M16 9.5a1.5 1.5 0 013 0V14a7 7 0 01-7 7h-1a6 6 0 01-5-2.7L3.6 14.6a1.5 1.5 0 012.4-1.8L7 14',
  point: 'c 12,12,3.5,1|c 12,12,8',
  rect: 'r 4,6,16,12',
  ellipse: 'e 12,12,8.5,6',
  polygon: 'p M5 8l7-4 7 5-2 9H7z',
  freehand: 'p M4 16c3-6 5 2 8-3s4-6 8-3',
  line: 'p M5 19L19 5|c 5,19,1.6,1|c 19,5,1.6,1',
  sam: 'p M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z',
  eye: 'p M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z|c 12,12,3',
  eyeOff: 'p M3 3l18 18M10.6 5.1A10 10 0 0112 5c6.4 0 10 7 10 7a17 17 0 01-3.2 4M6.6 6.6A17 17 0 002 12s3.6 7 10 7a10 10 0 005.4-1.6',
  more: 'c 5,12,1.4,1|c 12,12,1.4,1|c 19,12,1.4,1',
  check: 'p M5 12l5 5 9-10',
  close: 'p M6 6l12 12M18 6L6 18',
  density: 'p M4 20V10M9 20V4M14 20v-7M19 20V8',
  distance: 'c 6,17,2.5|c 18,7,2.5|p M8 15l8-6',
  region: 'p M4 7l6-3 8 4 2 9-7 4-8-4z',
  pattern: 'c 6,6,2|c 18,6,2|c 6,18,2|c 18,18,2|c 12,12,2',
  zoom: 'c 11,11,7|p M20 20l-4-4M8 11h6M11 8v6',
  copy: 'r 8,8,12,12|p M16 8V5a1 1 0 00-1-1H5a1 1 0 00-1 1v10a1 1 0 001 1h3',
};
const fmt = (v, d = 3) => (v == null || !Number.isFinite(v) ? '–' : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US')
  : Number.isInteger(v) ? String(v) : (+v.toPrecision(d)).toString());
const log = (m, err = false) => { const l = $('log'); l.textContent = m; l.className = err ? 'err' : ''; };
const enc = encodeURIComponent;
const typing = (t) => t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));

// ------------------------------------------------------------------------------------------------ bridge
// pywebview's js_api in the desktop window; token-protected HTTP otherwise
const native = () => window.pywebview && window.pywebview.api;
const NATIVE = new URLSearchParams(location.search).has('native');
const nativeReady = new Promise((res) => {
  if (!NATIVE || native()) return res(!!native());
  window.addEventListener('pywebviewready', () => res(true), { once: true });
  setTimeout(() => res(!!native()), 10000);
});
let KEY = '';   // wissi-viz serve --key: typed once per tab
try { KEY = sessionStorage.getItem('wissi-viz-key') || ''; } catch (e) { /* storage blocked */ }
async function http(path, body, askKey = true, method = null) {
  const headers = { 'X-Wissi-Token': TOKEN, 'Content-Type': 'application/json' };
  if (KEY) headers['X-Wissi-Key'] = KEY;
  const r = await fetch(path, { method: method || (body ? 'POST' : 'GET'), headers, body: body ? JSON.stringify(body) : undefined });
  if (r.status === 401 && CAPS.key && askKey) {
    const k = window.prompt('This server needs its key to open files or connect:');
    if (k) {
      KEY = k;
      try { sessionStorage.setItem('wissi-viz-key', k); } catch (e) { /* storage blocked */ }
      return http(path, body, false, method);
    }
  }
  if (!r.ok) {
    let m = await r.text();
    try { m = JSON.parse(m).detail || m; } catch (e) { /* plain text */ }
    throw new Error(m);
  }
  return r.json();
}
/** The tile API (no token): JSON, with the server's error message on failure. */
async function api(path, body = null, method = null) {
  const r = await fetch(API + path, { method: method || (body ? 'POST' : 'GET'), cache: 'no-cache',
    headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `${r.status} ${r.statusText}`);
  return j;
}
const quiet = (p) => p.catch((e) => ({ errors: [e.message] }));
const call = async (name, ...args) => {
  if (name === 'connect') return quiet(http(`${BASE}/app/connect`, { spec: args[0], images: args[1] }));
  if (name === 'browse') return quiet(http(`${BASE}/app/browse` + (args[0] ? `?dir=${enc(args[0])}` : '')));
  if (await nativeReady && native()) return native()[name](...args);
  if (name === 'open_paths') return quiet(http(`${BASE}/app/open`, { paths: args[0], slide: args[1] }));
  if (name === 'close_slide') return quiet(http(`${BASE}/app/close`, { slide: args[0] }));
  if (name === 'state') {
    if (TOKEN) return http(`${BASE}/app/state`);
    const list = await (await fetch(`${API}/slides`, { cache: 'no-cache' })).json();   // a page without a session
    return { slides: list.map((x) => ({ ...x, path: x.name })), recent: [] };
  }
  if (!CAPS.open) return null;
  showPathBox(name); return null;               // no native dialogs: the in-page picker / path box
};

// ------------------------------------------------------------------------------------------------ slides & panes
function activeSlide() { return panes[active] || null; }
function curPane() { return grid.panes ? grid.panes[active] : null; }

function onResult(res) {
  if (!res) return;
  if (res.slides) slides = res.slides;
  const opened = res.opened || [];
  if (opened.length === 1 && panes.length) panes[active] = opened[0];
  else if (opened.length) { panes = opened.slice(0, MAX_PANES); active = 0; }
  if (opened.length) apply();
  for (const L of res.layers || []) grid.refreshAnnotations(L.slide);
  renderList();
  const msg = [];
  if (opened.length) msg.push(`opened ${opened.length} slide${opened.length > 1 ? 's' : ''}`);
  if ((res.layers || []).length) msg.push(`added ${res.layers.map((l) => l.layer).join(', ')}`);
  if (res.connected) msg.push(`${res.service}: ${res.connected} image${res.connected > 1 ? 's' : ''} in the slide list`);
  if (res.installed) msg.push(`shortcut: ${res.installed.join(', ')}`);
  if ((res.errors || []).length) { log(res.errors.join(' · '), true); toast({ kind: 'bad', title: 'Could not open everything', sub: res.errors[0] }); }
  else if (msg.length) log(msg.join(' · '));
  if ((res.layers || []).length) toast({ kind: 'good', title: `Added ${res.layers.length} layer${res.layers.length > 1 ? 's' : ''}`, sub: res.layers.map((l) => l.layer).join(', ') });
}

function apply() {
  panes = panes.filter((id) => slides.some((x) => x.id === id));
  if (!panes.length && slides.length) panes = [slides[0].id];
  $('empty').style.display = panes.length ? 'none' : '';
  $('b-layer').disabled = !panes.length;
  $('n').value = Math.max(1, panes.length);
  active = Math.min(active, Math.max(0, panes.length - 1));
  if (panes.length) grid.setSlides(panes);
  if (panes.length > 1) grid.setActive(active);
  renderList();
  renderCrumb();
  scheduleInspector();
}

function setPaneCount(n) {
  if (!slides.length) return;
  const next = panes.slice(0, n);
  for (let i = next.length; i < n; i++) {
    const unused = slides.find((x) => !next.includes(x.id));
    next.push((unused || slides[i % slides.length]).id);
  }
  panes = next;
  apply();
}

function renderCrumb() {
  const sl = slides.find((x) => x.id === activeSlide());
  const c = $('crumb');
  c.replaceChildren();
  if (!sl) return;
  c.append(h('b', {}, sl.name));
  const info = curPane()?.slide?.info;
  if (info?.mpp) c.append(` · ${info.mpp[0].toFixed(3)} µm/px`);
  if (panes.length > 1) c.append(` · pane ${active + 1} of ${panes.length}`);
}

function renderList() {
  const ul = $('slides');
  ul.replaceChildren();
  for (const sl of slides) {
    const shown = panes.map((id, i) => (id === sl.id ? i + 1 : 0)).filter(Boolean);
    const remote = /^[a-z0-9_-]+:/i.test(sl.path || '') && !/^[a-z]:[\\/]/i.test(sl.path || '');
    const img = h('img', { class: 'thumb', alt: '', loading: 'lazy', src: `${API}/slides/${enc(sl.id)}/thumbnail?size=160` });
    img.onerror = () => { img.removeAttribute('src'); img.classList.add('blank'); };
    const x = h('button', { title: 'Close', 'aria-label': `Close ${sl.name}`, hidden: !CAPS.close }, '✕');
    x.onclick = async (e) => {
      e.stopPropagation();
      const r = await call('close_slide', sl.id);
      if (r?.slides) slides = r.slides;
      panes = panes.map((id) => (id === sl.id ? null : id)).filter(Boolean);
      apply();
    };
    const li = h('li', { class: (shown.length ? 'shown' : '') + (sl.id === activeSlide() ? ' active' : ''), title: sl.path || sl.name },
      img, h('span', { class: 'txt' }, h('span', { class: 'name' }, sl.name),
        h('span', { class: 'sub' }, remote ? (sl.path || '').split(':')[0] + ' · tile API' : shown.length ? `pane ${shown.join(', ')}` : '')),
      h('span', { class: 'pane' }, shown.join(' ')), x);
    li.onclick = () => { if (!panes.length) panes = [sl.id]; else panes[active] = sl.id; apply(); };
    li.ondblclick = () => { if (panes.length < MAX_PANES && !panes.includes(sl.id)) { panes.push(sl.id); active = panes.length - 1; apply(); } };
    ul.append(li);
  }
}

function renderRecent(items) {
  const box = $('recent');
  box.replaceChildren();
  if (!items || !items.length) return;
  box.append(h('h2', { class: 'lbl' }, 'Recent'));
  for (const p of items.slice(0, 8)) {
    const parts = p.split(/[\\/]/);
    const a = h('a', { href: '#', title: p }, parts.pop(), h('span', {}, parts.slice(-2).join('/')));
    a.onclick = async (e) => { e.preventDefault(); onResult(await call('open_paths', [p], null)); };
    box.append(a);
  }
}

// ---- the in-page file picker and path box (browser window / server)
function showPathBox(why) {
  $('empty').style.display = '';
  $('pathbox').style.display = 'block';
  $('paths').dataset.slide = why === 'pick_layer' ? activeSlide() || '' : '';
  $('paths').placeholder = SERVER ? 'Or paste paths on the server, one per line' : 'Or paste file or folder paths, one per line';
  if (CAPS.browse) browse($('browser').dataset.dir || '');
  else $('paths').focus();
  log(SERVER ? 'Pick files on the server' : 'Browser window: pick files below or paste paths (native dialogs need the desktop window)');
}
const openPaths = async (list) => onResult(await call('open_paths', list, $('paths').dataset.slide || null));
async function browse(dir) {
  const box = $('browser');
  const res = await call('browse', dir);
  if (!res || res.errors) { box.style.display = 'none'; if (res) log(res.errors.join(' · '), true); return; }
  box.dataset.dir = res.dir;
  box.style.display = 'block';
  box.replaceChildren(h('div', { class: 'cwd', title: res.dir }, res.dir || 'Folders you can open from'));
  const item = (kind, name, act, title) => {
    const a = h('a', { href: '#', title: title || name }, h('span', { class: 'k' }, kind), h('span', { class: 'n' }, name));
    a.onclick = (e) => { e.preventDefault(); act(); };
    box.append(a);
  };
  if (res.parent != null) item('↑', 'up', () => browse(res.parent), res.parent || 'Folders you can open from');
  if (res.entries.some((e) => e.kind === 'slide')) item('all', 'Open every slide in this folder', () => openPaths([res.dir]), res.dir);
  const label = { folder: 'folder', slide: 'slide', table: 'cells', annotations: 'annotations' };
  for (const e of res.entries) item(label[e.kind] || e.kind, e.name, e.kind === 'folder' ? () => browse(e.path) : () => openPaths([e.path]), e.path);
  if (!res.entries.length) box.append(h('div', { class: 'none' }, 'Nothing wissi-viz opens here'));
}

// ------------------------------------------------------------------------------------------------ toolbar
const TOOLS = [['pan', 'Pan'], ['point', 'Point'], ['rect', 'Rectangle'], ['ellipse', 'Ellipse'], ['polygon', 'Polygon'],
  ['freehand', 'Freehand'], ['line', 'Ruler'], ['sam', 'SAM: click an object']];
function buildTools() {
  const nav = $('tools');
  TOOLS.forEach(([t, label], i) => {
    const b = h('button', { class: 'slot' + (t === 'pan' ? ' on' : ''), title: `${label} (${i + 1})`, 'aria-label': label, dataset: { tool: t } },
      icon(t, 18), h('i', {}, String(i + 1)));
    b.onclick = () => setTool(t);
    nav.append(b);
  });
}
function setTool(t) {
  grid.setTool(t);
  for (const b of $('tools').children) b.classList.toggle('on', b.dataset.tool === t);
  $('a-sam').classList.toggle('on', t === 'sam');
  if (t === 'sam') samStatus();
}

// ------------------------------------------------------------------------------------------------ HUD: toasts, key hints
function toast({ kind = '', title, sub = '', action = null, ms = 5000 }) {
  const box = $('toasts');
  const mark = kind === 'bad' ? h('span', { style: 'color: var(--danger); font-weight: 700' }, '!')
    : h('span', { style: `color: ${kind === 'analysis' ? 'var(--violet)' : 'var(--good)'}` }, icon('check', 20));
  const t = h('div', { class: `toast rise ${kind}`, role: 'status' }, mark, h('span', { class: 'grow' }, h('b', {}, title), sub ? h('span', { class: 's' }, sub) : null));
  if (action) {
    const b = h('button', { class: 'btn' }, action.label);
    b.onclick = () => { action.fn(); t.remove(); };
    t.append(b);
  }
  const x = h('button', { class: 'btn icon ghost', 'aria-label': 'Dismiss' }, icon('close', 14));
  x.onclick = () => t.remove();
  t.append(x);
  box.prepend(t);
  while (box.children.length > 3) box.lastChild.remove();
  if (ms) setTimeout(() => t.remove(), ms);
}
let flashT = 0;
function keyFlash(key, label) {
  const k = $('keyflash');
  k.replaceChildren(h('span', { class: 'kc' }, key), label);
  k.style.display = 'flex';
  clearTimeout(flashT);
  flashT = setTimeout(() => { k.style.display = 'none'; }, 700);
}

// ------------------------------------------------------------------------------------------------ inspector tabs
let tab = 'layers';
function setTab(t) {
  tab = t;
  for (const b of $('tabs').children) b.setAttribute('aria-selected', String(b.dataset.tab === t));
  for (const p of document.querySelectorAll('#insp .pane-body')) p.classList.toggle('on', p.id === `p-${t}`);
  if ($('insp').classList.contains('hidden')) toggleInspector(true);
  renderInspector();
}
function toggleInspector(show) {
  const on = show ?? $('insp').classList.contains('hidden');
  $('insp').classList.toggle('hidden', !on);
  $('b-insp').classList.toggle('on', on);
}
let insT = 0;
function scheduleInspector() { clearTimeout(insT); insT = setTimeout(renderInspector, 60); }
function renderInspector() {
  if ($('insp').classList.contains('hidden')) return;
  if (tab === 'layers') renderLayers();
  else if (tab === 'analyze') renderAnalyze();
  else if (tab === 'measure') renderMeasure();
  else if (tab === 'draw') { refreshDrawLists(); samStatus(); }
}
const layersOf = (p) => (p?.slide ? [...p.slide.layers.values()] : []);
const paneSel = () => ($('l-scope').querySelector('[aria-pressed=true]')?.dataset.scope === 'all' ? 'all' : active);

// ------------------------------------------------------------------------------------------------ Layers tab
const expanded = new Set();
const colorBy = new Map();     // `${slide}|${layer}` → column ('' = class)
const summaries = new Map();   // `${slide}|${layer}|${version}|${tissue}` → Promise<summary>
let hoveredLayer = null;
let solo = null;               // {layer, prev: [[pane, name, visible], …]}

function getSummary(sid, L, tissue = false) {
  const key = `${sid}|${L.name}|${L.meta.version}|${tissue}`;
  if (!summaries.has(key)) {
    const p = api(`/slides/${enc(sid)}/annotations/summary?layer=${enc(L.name)}&tissue=${tissue}`);
    p.catch(() => summaries.delete(key));
    summaries.set(key, p);
  }
  return summaries.get(key);
}

function renderDisplay(p) {
  const body = $('disp-body'), sum = $('disp-sum');
  body.replaceChildren();
  const info = p?.slide?.info;
  if (!info) { sum.textContent = ''; return; }
  if (info.mode === 'channels' && p.channels) {
    sum.textContent = `${p.channels.filter((c) => c.visible).length} of ${p.channels.length} channels`;
    const max = info.dtype === 'uint16' ? 65535 : 255;
    p.channels.forEach((c, i) => {
      const vis = h('input', { type: 'checkbox', checked: c.visible, 'aria-label': `Show ${c.name}` });
      vis.onchange = () => { grid.setChannel(active, i, { visible: vis.checked }); renderDisplay(curPane()); };
      const col = h('input', { type: 'color', value: toHex(c.color), 'aria-label': `${c.name} colour` });
      col.oninput = () => grid.setChannel(active, i, { color: col.value });
      const lo = h('input', { type: 'range', min: 0, max, value: c.lo, 'aria-label': `${c.name} minimum` });
      const hi = h('input', { type: 'range', min: 0, max, value: c.hi, 'aria-label': `${c.name} maximum` });
      lo.oninput = () => grid.setChannel(active, i, { min: +lo.value });
      hi.oninput = () => grid.setChannel(active, i, { max: +hi.value });
      body.append(h('div', { class: 'row' }, vis, col, h('span', { class: 'grow' }, c.name)),
        h('div', { class: 'row', style: 'padding-left: 26px' }, lo, hi));
    });
    return;
  }
  const a = p.adjust || { brightness: 0, contrast: 1, gamma: 1 };
  sum.textContent = p.stain ? `stain: ${p.stain.name}` : 'brightfield';
  const stain = h('select', { 'aria-label': 'Stain view' }, ...['off', 'hematoxylin', 'eosin', 'dab', 'fluorescence']
    .map((m) => h('option', { selected: (p.stain?.name || 'off') === m }, m)));
  stain.onchange = () => { grid.setStain(active, stain.value); renderDisplay(curPane()); };
  body.append(h('div', { class: 'field', style: 'grid-template-columns: 88px 1fr' }, 'Stain view', stain));
  for (const [label, key, min, max, step] of [['Brightness', 'brightness', -0.5, 0.5, 0.01], ['Contrast', 'contrast', 0.3, 3, 0.01], ['Gamma', 'gamma', 0.3, 3, 0.01]]) {
    const out = h('output', {}, fmt(a[key] ?? 0, 2));
    const r = h('input', { type: 'range', min, max, step, value: a[key] ?? 0, 'aria-label': label });
    r.oninput = () => { grid.setAdjust(active, { [key]: +r.value }); out.textContent = fmt(+r.value, 2); };
    body.append(h('label', { class: 'field' }, label, r, out));
  }
}
function toHex(c) {
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return '#' + c.slice(0, 3).map((v) => Math.round(Math.max(0, Math.min(1, v > 1 ? v / 255 : v)) * 255).toString(16).padStart(2, '0')).join('');
  return '#ffffff';
}

function renderLayers() {
  const p = curPane();
  renderDisplay(p);
  const list = $('l-list');
  const sid = activeSlide();
  const sl = slides.find((x) => x.id === sid);
  $('l-title').textContent = sl ? `Layers on ${sl.name}` : 'Layers';
  const q = $('l-filter').value.trim().toLowerCase();
  const all = layersOf(p);
  const shown = all.filter((L) => !q || L.name.toLowerCase().includes(q) || (L.meta.classes || []).some((c) => c.name.toLowerCase().includes(q)));
  list.replaceChildren(...shown.map((L) => layerCard(p, sid, L, q)));
  $('l-empty').hidden = all.length > 0 || !sid;
  $('l-add').disabled = !sid || !CAPS.open;
  $('l-add').hidden = !CAPS.open;
}

function layerCard(p, sid, L, q) {
  const vis = p.layerVisible(L);
  const analysis = L.meta.style?.origin === 'analysis';
  const isSolo = solo?.layer === L.name;
  const card = h('div', { class: `lcard${analysis ? ' analysis' : ''}${vis ? '' : ' off'}${isSolo ? ' solo' : ''}`, dataset: { layer: L.name } });
  const eye = h('button', { class: `eye${vis ? ' on' : ''}`, title: `${vis ? 'Hide' : 'Show'} (H)`, 'aria-label': `${vis ? 'Hide' : 'Show'} ${L.name}`, 'aria-pressed': String(vis) },
    icon(vis ? 'eye' : 'eyeOff'));
  eye.onclick = () => toggleLayer(L.name);
  const cm = L.meta.colormap;
  const classes = L.meta.classes || [];
  const sw = cm ? h('span', { class: 'swg', style: `background: linear-gradient(90deg, ${(cm.lut || []).filter((_, i) => i % 64 === 0).concat([cm.lut?.[255]]).filter(Boolean).join(',') || '#444,#ccc'})` })
    : h('span', { style: 'display: inline-flex; gap: 2px' }, ...classes.slice(0, 3).map((c) => h('span', { class: 'sw', style: `background: ${c.color}` })));
  const nm = h('span', { class: 'nm', title: `${L.name} — click for style and classes`, role: 'button', tabindex: 0 }, L.name);
  const key = `${sid}|${L.name}`;
  nm.onclick = () => { expanded.has(key) ? expanded.delete(key) : expanded.add(key); renderLayers(); };
  nm.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); nm.onclick(); } };
  const count = h('span', { class: 'chip', title: 'objects' }, (L.meta.count ?? 0).toLocaleString('en-US'));
  const soloB = h('button', { class: `solo-b${isSolo ? ' on' : ''}`, title: 'Solo: show only this layer (S)', 'aria-pressed': String(isSolo) }, 'S');
  soloB.onclick = () => toggleSolo(L.name);
  const more = h('button', { class: 'btn icon ghost', title: 'More', 'aria-label': `More for ${L.name}` }, icon('more'));
  more.onclick = (e) => layerMenu(e, sid, L);
  card.append(h('div', { class: 'lhead' }, eye, sw, nm, analysis ? h('span', { class: 'badge' }, 'analysis') : null, count, soloB, more));
  card.onmouseenter = () => {
    hoveredLayer = L.name;
    if (vis) { grid.flashLayer(active, L.name, 900); card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 900); }
  };
  if (expanded.has(key) || (q && classes.some((c) => c.name.toLowerCase().includes(q)))) card.append(layerBody(p, sid, L));
  return card;
}

function layerBody(p, sid, L) {
  const st = L.meta.style || (L.meta.style = {});
  const body = h('div', { class: 'lbody' });
  const op = h('input', { type: 'range', min: 0.05, max: 1, step: 0.05, value: st.opacity ?? 1, 'aria-label': `${L.name} opacity` });
  const opOut = h('output', {}, `${Math.round((st.opacity ?? 1) * 100)}%`);
  op.oninput = () => { grid.setLayerStyle(paneSel(), L.name, { opacity: +op.value }); opOut.textContent = `${Math.round(op.value * 100)}%`; };
  const lw = h('input', { type: 'range', min: 0, max: 4, step: 0.25, value: st.line_width ?? 1.5, 'aria-label': `${L.name} outline width` });
  const lwOut = h('output', {}, fmt(st.line_width ?? 1.5, 2));
  lw.oninput = () => { grid.setLayerStyle(paneSel(), L.name, { line_width: +lw.value }); lwOut.textContent = fmt(+lw.value, 2); };
  const fill = h('input', { type: 'range', min: 0, max: 0.8, step: 0.05, value: st.fill_opacity ?? 0, 'aria-label': `${L.name} fill` });
  const fillOut = h('output', {}, `${Math.round((st.fill_opacity ?? 0) * 100)}%`);
  fill.oninput = () => { grid.setLayerStyle(paneSel(), L.name, { fill_opacity: +fill.value }); fillOut.textContent = `${Math.round(fill.value * 100)}%`; };
  body.append(h('label', { class: 'field' }, 'Opacity', op, opOut), h('label', { class: 'field' }, 'Outline', lw, lwOut), h('label', { class: 'field' }, 'Fill', fill, fillOut));
  const sel = h('select', { 'aria-label': `Colour ${L.name} by` }, h('option', { value: '' }, 'class'));
  sel.value = colorBy.get(`${sid}|${L.name}`) || '';
  const cby = h('label', { class: 'field', style: 'grid-template-columns: 88px 1fr' }, 'Colour by', sel);
  body.append(cby);
  sel.onchange = async () => {
    const col = sel.value || null;
    try {
      await api(`/slides/${enc(sid)}/annotations/style`, { layer: L.name, color_by: col, cmap: 'viridis' });
      colorBy.set(`${sid}|${L.name}`, sel.value);
      grid.refreshAnnotations(sid);
      log(col ? `${L.name}: coloured by ${col}` : `${L.name}: coloured by class`);
    } catch (e) { log(`colour by: ${e.message}`, true); }
  };
  const clsBox = h('div', { style: 'display: flex; flex-direction: column; gap: 2px' });
  body.append(clsBox);
  const classes = L.meta.classes || [];
  const drawClasses = (counts = {}) => {
    const max = Math.max(1, ...Object.values(counts));
    clsBox.replaceChildren(...classes.slice(0, 60).map((c) => {
      const on = !(L.hidden && L.hidden.has(c.name));
      const box = h('button', { class: 'box', style: on ? `background: ${c.color}; border-color: ${c.color}` : '', 'aria-label': `${on ? 'Hide' : 'Show'} ${c.name}`, 'aria-pressed': String(on) },
        on ? s('svg', { width: 12, height: 12, viewBox: '0 0 24 24', fill: 'none', stroke: '#0b0d10', 'stroke-width': 3.5, 'stroke-linecap': 'round', 'aria-hidden': 'true' }, s('path', { d: 'M5 12l5 5 9-10' })) : null);
      box.onclick = () => { grid.setClassVisible(paneSel(), L.name, c.name, !on); drawClasses(counts); };
      const col = h('input', { type: 'color', value: toHex(c.color), 'aria-label': `${c.name} colour` });
      col.oninput = () => { L.setClassColor?.(c.name, col.value); c.color = col.value; grid.requestRender(); };
      const n = counts[c.name];
      return h('div', { class: `cls${on ? '' : ' off'}` }, box, h('span', { class: 'nm2', title: c.name }, c.name),
        n != null ? h('span', { class: 'bar', style: 'width: 64px' }, h('span', { style: `width: ${(100 * n / max).toFixed(1)}%; background: ${c.color}` })) : null,
        n != null ? h('span', { class: 'chip' }, n.toLocaleString('en-US')) : null, col);
    }));
  };
  drawClasses();
  getSummary(sid, L).then((sm) => {
    const counts = Object.fromEntries((sm.classes || []).map((c) => [c.name, c.n]));
    drawClasses(counts);
    for (const col of sm.numeric || []) sel.append(h('option', { value: col, selected: col === sel.value }, col));
    if (!(sm.numeric || []).length) cby.hidden = true;
  }).catch(() => {});
  return body;
}

function toggleLayer(name) {
  const p = curPane();
  const L = p?.slide?.layers.get(name);
  if (!L) return;
  const on = !p.layerVisible(L);
  grid.setLayerVisible(paneSel(), name, on);
  if (on) grid.flashLayer(active, name, 700);
  renderLayers();
}
function toggleSolo(name) {
  const p = curPane();
  if (!p?.slide) return;
  if (solo) {
    for (const [pi, n, v] of solo.prev) grid.setLayerVisible(pi, n, v);
    const was = solo.layer;
    solo = null;
    if (was === name) { renderLayers(); return; }
  }
  const targets = paneSel() === 'all' ? grid.panes : [p];
  const prev = [];
  for (const q of targets) for (const L of layersOf(q)) prev.push([q.index, L.name, q.layerVisible(L)]);
  solo = { layer: name, prev };
  for (const q of targets) for (const L of layersOf(q)) grid.setLayerVisible(q.index, L.name, L.name === name);
  grid.flashLayer(active, name, 700);
  renderLayers();
}

let menuEl = null;
function closeMenu() { menuEl?.remove(); menuEl = null; }
function layerMenu(ev, sid, L) {
  ev.stopPropagation();
  closeMenu();
  const item = (label, k, fn, cls = '') => {
    const b = h('button', { class: cls }, label, k ? h('span', { class: 'kc' }, k) : null);
    b.onclick = (e) => { e.stopPropagation(); if (fn(b) !== false) closeMenu(); };
    return b;
  };
  const bbox = L.meta.bbox;
  menuEl = h('div', { class: 'menu', role: 'menu' },
    bbox ? item('Zoom to layer', 'Z', () => grid.goToBounds(active, bbox[0], bbox[1], bbox[2], bbox[3])) : null,
    item(`Export as ${$('a-fmt').selectedOptions[0].text}`, 'E', () => {
      location.href = `${API}/slides/${enc(sid)}/annotations/export?layer=${enc(L.name)}&fmt=${$('a-fmt').value}`;
    }),
    item('Summary', 'M', () => { setTab('measure'); $('ms-layer').value = L.name; renderMeasure(); }),
    item('Measure its regions', '', () => { setTab('measure'); $('m-regions').value = L.name; }),
    item('Analyse these cells', 'A', () => { setTab('analyze'); $('an-objects').value = L.name; renderAnalyze(); }),
    h('hr'),
    item('Remove from slide', 'Del', (b) => {
      if (!b.dataset.sure) { b.dataset.sure = '1'; b.firstChild.textContent = 'Click again to remove'; return false; }
      api(`/slides/${enc(sid)}/annotations?layer=${enc(L.name)}`, null, 'DELETE')
        .then(() => { grid.refreshAnnotations(sid); toast({ title: `Removed ${L.name}` }); })
        .catch((e) => log(`remove: ${e.message}`, true));
      return true;
    }, 'danger'));
  document.body.append(menuEl);
  const r = ev.currentTarget.getBoundingClientRect();
  menuEl.style.left = `${Math.max(8, Math.min(window.innerWidth - menuEl.offsetWidth - 8, r.right - menuEl.offsetWidth))}px`;
  menuEl.style.top = `${Math.min(window.innerHeight - menuEl.offsetHeight - 8, r.bottom + 4)}px`;
}

// ------------------------------------------------------------------------------------------------ Analyze tab
let catalog = null, catalogError = '';
let picked = null;
const results = [];

async function ensureCatalog() {
  if (catalog || catalogError) return;
  try { catalog = await api('/analyses'); } catch (e) { catalogError = e.message; }
}
function cellLayers(p) { return layersOf(p).filter((L) => L.meta.style?.origin !== 'analysis' && (L.meta.count ?? 0) >= 3); }
function regionLayers(p, except) { return layersOf(p).filter((L) => L.name !== except && L.meta.style?.origin !== 'analysis'); }

async function renderAnalyze() {
  await ensureCatalog();
  const p = curPane();
  const objSel = $('an-objects');
  const cells = cellLayers(p);
  const keep = objSel.value;
  objSel.replaceChildren(...cells.map((L) => h('option', { value: L.name }, `${L.name} · ${(L.meta.count ?? 0).toLocaleString('en-US')}`)));
  if (cells.some((L) => L.name === keep)) objSel.value = keep;
  else if (cells.length) objSel.value = [...cells].sort((a, b) => (b.meta.count ?? 0) - (a.meta.count ?? 0))[0].name;
  const box = $('an-catalog');
  box.replaceChildren();
  if (catalogError) { box.append(h('div', { class: 'panel-note' }, catalogError)); return; }
  if (!cells.length) { box.append(h('div', { class: 'panel-note' }, 'Add a cell table (or segment the slide) to run spatial analyses.')); $('an-form').hidden = true; return; }
  const hasRegions = regionLayers(p, objSel.value).length > 0;
  const groups = {};
  for (const a of catalog) (groups[a.group] ||= []).push(a);
  const ic = { Density: 'density', Distances: 'distance', Regions: 'region', Patterns: 'pattern' };
  for (const [g, items] of Object.entries(groups)) {
    const col = h('div', { class: 'an-group' }, h('span', { class: 'lbl' }, g === 'Regions' && !hasRegions ? 'Regions · draw or add a region layer first' : g));
    for (const a of items) {
      const dis = a.needs_regions && !hasRegions;
      const b = h('button', { class: 'an-card', 'aria-pressed': String(picked === a.id), disabled: dis, title: dis ? 'Needs a region layer: draw one (Draw tab) or add an annotation file' : a.desc },
        h('span', { class: 'ic' }, icon(ic[a.group] || 'pattern')), h('span', { class: 'grow' }, h('b', {}, a.name), h('span', { class: 'd' }, a.desc)), h('span', { class: 'sp' }, a.speed));
      b.onclick = () => { picked = picked === a.id ? null : a.id; renderAnalyze(); };
      col.append(b);
      if (picked === a.id) col.append(analysisForm(p, a));
    }
    box.append(col);
  }
}

function analysisForm(p, a) {
  const form = h('div', { id: 'an-form-live', class: 'rise', style: 'border: 1px solid var(--violet); border-radius: 12px; background: #15112a; padding: 10px; display: flex; flex-direction: column; gap: 9px' });
  const objName = $('an-objects').value;
  const L = p.slide.layers.get(objName);
  const classes = (L?.meta.classes || []).map((c) => c.name);
  const values = {};
  let classIdx = 0;
  for (const prm of a.params) {
    let input;
    if (prm.kind === 'class' || prm.kind === 'class?') {
      const opts = (prm.kind === 'class?' ? ['all'] : []).concat(classes);
      input = h('select', { 'aria-label': prm.label }, ...opts.map((c) => h('option', {}, c)));
      input.value = prm.kind === 'class?' ? 'all' : classes[Math.min(classIdx++, classes.length - 1)] || '';
    } else if (prm.kind === 'number') {
      const out = h('output', {}, `${fmt(prm.default)} ${prm.unit || ''}`.trim());
      const r = h('input', { type: 'range', min: prm.min, max: prm.max, step: prm.step || 1, value: prm.default, 'aria-label': prm.label });
      r.oninput = () => { out.textContent = `${fmt(+r.value)} ${prm.unit || ''}`.trim(); };
      values[prm.key] = () => +r.value;
      form.append(h('label', { class: 'field' }, prm.label, r, out));
      continue;
    } else if (prm.kind === 'column') {
      input = h('select', { 'aria-label': prm.label }, h('option', { value: '' }, 'loading…'));
      getSummary(activeSlide(), L).then((sm) => input.replaceChildren(...(sm.numeric || []).map((c) => h('option', {}, c)))).catch(() => {});
    } else if (prm.kind === 'regions') {
      input = h('select', { 'aria-label': prm.label }, ...regionLayers(p, objName).map((R) => h('option', {}, R.name)));
    }
    values[prm.key] = () => input.value;
    form.append(h('label', { class: 'field', style: 'grid-template-columns: 84px 1fr' }, prm.label, input));
  }
  const run = h('button', { class: 'run', id: 'an-run' }, `Run ${a.name}`);
  run.onclick = async () => {
    if (run.classList.contains('busy')) return;
    const params = Object.fromEntries(Object.entries(values).map(([k, f]) => [k, f()]));
    const sid = activeSlide();
    run.classList.add('busy');
    const t0 = performance.now();
    const tick = setInterval(() => { run.textContent = `Running… ${((performance.now() - t0) / 1000).toFixed(1)} s`; }, 200);
    log(`${a.name}: running on ${objName}…`);
    try {
      const res = await api(`/slides/${enc(sid)}/analyses`, { analysis: a.id, objects: objName, params });
      res.slide = sid;
      // one result on the slide at a time: earlier analysis layers are hidden (still in Layers / Results)
      const earlier = res.layers.length ? results.filter((x) => x.slide === sid).flatMap((x) => x.layers).filter((n) => !res.layers.includes(n)) : [];
      results.unshift(res);
      grid.refreshAnnotations(sid);
      setTimeout(() => { for (const n of earlier) grid.setLayerVisible(active, n, false); scheduleInspector(); }, 400);
      renderResults(true);
      $('an-results').firstChild?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      const n = res.layers.length;
      const k0 = res.kpis[0];
      toast({ kind: 'analysis', title: `${res.title}`, sub: `${k0 ? `${k0.label} ${k0.value}` : 'done'}${n ? ` · ${n} layer${n > 1 ? 's' : ''} on the slide` : ''}`,
        action: n ? { label: 'Show', fn: () => showResultLayers(res) } : null, ms: 7000 });
      log(`${res.title}: done in ${res.seconds} s`);
    } catch (e) {
      toast({ kind: 'bad', title: `${a.name} failed`, sub: e.message, ms: 9000 });
      log(`${a.name}: ${e.message}`, true);
    }
    clearInterval(tick);
    run.classList.remove('busy');
    run.textContent = `Run ${a.name}`;
  };
  form.append(run);
  return form;
}

function showResultLayers(res) {
  if (activeSlide() !== res.slide) { const i = panes.indexOf(res.slide); if (i >= 0) { active = i; grid.setActive(i); } }
  setTimeout(() => {
    for (const n of res.layers) { grid.setLayerVisible(active, n, true); grid.flashLayer(active, n, 1200); }
    setTab('layers');
  }, 250);
}

function renderResults(fresh = false) {
  const box = $('an-results');
  box.replaceChildren();
  $('an-count').textContent = results.length ? `${results.length}` : '';
  if (!results.length) { box.append(h('div', { class: 'panel-note', id: 'an-none' }, 'Nothing yet. Each run adds layers to the slide and a result here.')); return; }
  results.forEach((r, i) => box.append(resultCard(r, fresh && i === 0)));
}

function resultCard(r, fresh) {
  const card = h('div', { class: `result${fresh ? ' new' : ''}` });
  card.append(h('div', { class: 'row' }, h('b', { class: 'grow', style: 'font-weight: 600' }, r.title), h('span', { class: 'chip' }, `${r.seconds} s`)));
  if (r.kpis.length) card.append(h('div', { class: 'kpis' }, ...r.kpis.map((k) => h('span', { class: 'kpi' }, h('span', { class: 'k' }, k.label), h('span', { class: 'v' }, k.value)))));
  if (r.chart) card.append(h('div', { class: 'chart' }, chart(r.chart)));
  if (r.note) card.append(h('div', { class: 'panel-note' }, r.note));
  const tbl = h('div', { class: 'tablebox', hidden: true });
  const btns = h('div', { class: 'row' });
  if (r.layers.length) { const b = h('button', { class: 'btn grow' }, `Show layer${r.layers.length > 1 ? 's' : ''}`); b.onclick = () => showResultLayers(r); btns.append(b); }
  if (r.table && r.table.rows.length) {
    const t = h('button', { class: 'btn' }, 'Table');
    t.onclick = () => { if (!tbl.childNodes.length) tbl.append(dataTable(r.table)); tbl.hidden = !tbl.hidden; };
    const c = h('button', { class: 'btn', title: r.table.n > r.table.rows.length ? `The first ${r.table.rows.length} of ${r.table.n} rows` : 'All rows' }, 'CSV');
    c.onclick = () => downloadCsv(r.table, `${r.title}.csv`);
    btns.append(t, c);
  }
  card.append(btns, tbl);
  return card;
}

function dataTable(t) {
  return h('table', { class: 'ct' }, h('tr', {}, ...t.columns.map((c) => h('th', {}, c))),
    ...t.rows.slice(0, 200).map((row) => h('tr', {}, ...row.map((v) => h('td', {}, typeof v === 'number' ? fmt(v, 4) : v ?? '–')))));
}
function downloadCsv(t, name) {
  const q = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const text = [t.columns.map(q).join(','), ...t.rows.map((r) => r.map(q).join(','))].join('\n');
  const a = h('a', { href: URL.createObjectURL(new Blob([text], { type: 'text/csv' })), download: name.replace(/[\\/:*?"<>|]+/g, '_') });
  document.body.append(a); a.click(); a.remove();
}

// ---- charts (small, dependency-free SVG)
const SERIES = ['#b49cff', '#5ed0ff', '#ffb454', '#2ecca0', '#ff7b9b', '#9aa3ad', '#e6e98a', '#7aa7ff'];
function chart(c) {
  const W = 340, H = c.kind === 'heat' ? 30 + 18 * Math.min(12, (c.rows || []).length) : 150;
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': c.ylabel || c.label || 'chart' });
  const txt = (x, y, t, a = {}) => s('text', { x, y, fill: '#9aa3ad', 'font-size': 9, 'font-family': 'ui-monospace, monospace', ...a }, t);
  const wrap = h('div', {}, svg);
  if (c.kind === 'heat') {
    const rows = c.rows || [], cols = c.cols || [], v = c.values || [];
    const flat = v.flat().filter((x) => x != null);
    const lo = Math.min(0, ...flat), hi = Math.max(0, ...flat), m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const L = 74, T = 24, cw = (W - L - 4) / Math.max(1, cols.length), rh = 18;
    cols.forEach((name, j) => svg.append(txt(L + cw * j + cw / 2, T - 6, String(name).slice(0, 8), { 'text-anchor': 'middle' })));
    rows.slice(0, 12).forEach((name, i) => {
      svg.append(txt(L - 4, T + rh * i + 12, String(name).slice(0, 11), { 'text-anchor': 'end' }));
      cols.forEach((_, j) => {
        const x = v[i]?.[j];
        const col = x == null ? '#1a1d22' : lo < 0 ? (x < 0 ? mix('#1a1d22', '#5ed0ff', -x / m) : mix('#1a1d22', '#ffb454', x / m)) : mix('#1a1d22', '#b49cff', x / (hi || 1));
        svg.append(s('rect', { x: L + cw * j + 1, y: T + rh * i + 1, width: cw - 2, height: rh - 2, rx: 2, fill: col }, s('title', {}, `${name} · ${cols[j]}: ${fmt(x)}`)));
        if (cols.length <= 8 && x != null) svg.append(txt(L + cw * j + cw / 2, T + rh * i + 12, fmt(x, 2), { 'text-anchor': 'middle', fill: '#e8eaed' }));
      });
    });
    wrap.append(h('div', { class: 'panel-note' }, lo < 0 ? `${c.label || ''}: blue below 0, amber above` : c.label || ''));
    return wrap;
  }
  const L = 36, R = 6, T = 8, B = 22, pw = W - L - R, ph = H - T - B;
  const series = (c.series || []).filter((x) => x && x.values);
  const all = series.flatMap((x) => x.values).filter((x) => x != null && Number.isFinite(x));
  let lo = Math.min(0, ...all), hi = Math.max(...all, 1e-9);
  if (c.kind === 'lines') { lo = Math.min(...all); hi = Math.max(...all); if (lo === hi) { lo -= 1; hi += 1; } }
  const y = (v) => T + ph - ((v - lo) / (hi - lo)) * ph;
  for (const f of [0, 0.5, 1]) {
    const v = lo + (hi - lo) * f;
    svg.append(s('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: '#262b33' }), txt(L - 4, y(v) + 3, fmt(v, 2), { 'text-anchor': 'end' }));
  }
  const xs = c.x || [];
  const numeric = xs.every((v) => typeof v === 'number');
  const x0 = numeric ? xs[0] : 0, x1 = numeric ? (c.x_end ?? xs[xs.length - 1]) : xs.length;
  const xp = (v) => L + ((v - x0) / ((x1 - x0) || 1)) * pw;
  if (c.kind === 'bars') {
    const n = xs.length || 1, gw = pw / n, k = series.length || 1, bw = Math.max(1, (gw - 2) / k);
    series.forEach((sr, si) => sr.values.forEach((v, i) => {
      if (v == null) return;
      svg.append(s('rect', { x: L + gw * i + 1 + bw * si, y: y(Math.max(0, v)), width: Math.max(1, bw - 1), height: Math.abs(y(v) - y(0)), rx: 1,
        fill: SERIES[si % SERIES.length], opacity: 0.9 }, s('title', {}, `${xs[i]}: ${fmt(v)}`)));
    }));
    const labels = numeric ? [[0, xs[0]], [n / 2, xs[Math.floor(n / 2)]], [n, c.x_end ?? xs[n - 1]]] : (n <= 8 ? xs.map((v, i) => [i + 0.5, v]) : [[0.5, xs[0]], [n - 0.5, xs[n - 1]]]);
    labels.forEach(([i, v], k) => svg.append(txt(L + gw * i, H - 8, typeof v === 'number' ? fmt(v, 3) : String(v).slice(0, 10),
      { 'text-anchor': k === 0 && numeric ? 'start' : k === labels.length - 1 && numeric ? 'end' : 'middle' })));
  } else {
    const band = series.filter((x) => x.band);
    if (band.length === 2) {
      const pts = xs.map((v, i) => `${xp(v)},${y(band[1].values[i])}`).concat(xs.map((v, i) => `${xp(v)},${y(band[0].values[i])}`).reverse());
      svg.append(s('polygon', { points: pts.join(' '), fill: 'rgba(154,163,173,.18)' }));
    }
    series.filter((x) => !x.band).forEach((sr, si) => {
      const d = sr.values.map((v, i) => (v == null ? null : `${xp(xs[i])},${y(v)}`)).filter(Boolean);
      svg.append(s('polyline', { points: d.join(' '), fill: 'none', stroke: SERIES[si % SERIES.length], 'stroke-width': 1.8 }));
    });
    [x0, (x0 + x1) / 2, x1].forEach((v, k) => svg.append(txt(xp(v), H - 8, fmt(v, 3), { 'text-anchor': ['start', 'middle', 'end'][k] })));
    if (lo < 0 && hi > 0) svg.append(s('line', { x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: '#4a5562', 'stroke-dasharray': '3 3' }));
  }
  if (c.marker != null && numeric) {
    const mx = xp(c.marker);
    svg.append(s('line', { x1: mx, x2: mx, y1: T, y2: T + ph, stroke: '#ffffff', 'stroke-width': 1.5 }), txt(Math.min(mx + 3, W - 60), T + 9, `${c.marker_label || ''} ${fmt(c.marker, 3)}`, { fill: '#e8eaed' }));
  }
  const named = series.filter((x) => !x.band);
  if (named.length > 1 || c.xlabel) {
    wrap.append(h('div', { class: 'row', style: 'flex-wrap: wrap; gap: 4px 10px; font-size: 11px; color: var(--dim)' },
      ...(named.length > 1 ? named.map((sr, si) => h('span', {}, h('span', { class: 'sw', style: `background: ${SERIES[si % SERIES.length]}; height: 8px; width: 8px; margin-right: 4px` }), sr.name)) : []),
      c.xlabel ? h('span', { class: 'grow', style: 'text-align: right' }, c.xlabel) : null));
  }
  return wrap;
}
function mix(a, b, t) {
  t = Math.max(0, Math.min(1, t || 0));
  const p = (x) => [1, 3, 5].map((i) => parseInt(x.slice(i, i + 2), 16));
  const [r1, g1, b1] = p(a), [r2, g2, b2] = p(b);
  return `rgb(${Math.round(r1 + (r2 - r1) * t)},${Math.round(g1 + (g2 - g1) * t)},${Math.round(b1 + (b2 - b1) * t)})`;
}

// ------------------------------------------------------------------------------------------------ Measure tab
async function renderMeasure() {
  const p = curPane();
  const sid = activeSlide();
  refreshDrawLists();
  const sel = $('ms-layer');
  const ls = layersOf(p);
  const keep = sel.value;
  sel.replaceChildren(...ls.map((L) => h('option', { value: L.name }, L.name)));
  if (ls.some((L) => L.name === keep)) sel.value = keep;
  else if (ls.length) sel.value = [...ls].sort((a, b) => (b.meta.count ?? 0) - (a.meta.count ?? 0))[0].name;
  const body = $('ms-body');
  const L = p?.slide?.layers.get(sel.value);
  if (!L) { body.replaceChildren(h('div', { class: 'panel-note' }, 'No layers on this slide yet.')); return; }
  body.replaceChildren(h('div', { class: 'panel-note' }, 'Summarising…'));
  let sm;
  try { sm = await getSummary(sid, L, true); } catch (e) { body.replaceChildren(h('div', { class: 'panel-note' }, e.message)); return; }
  if (sel.value !== L.name) return;
  const tile = (k, v, title) => h('div', { class: 'tile', title }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v));
  const tiles = h('div', { class: 'tiles' }, tile('Objects', sm.n.toLocaleString('en-US')));
  if (sm.per_mm2 != null) tiles.append(tile('per mm² tissue', fmt(sm.per_mm2)), tile('Tissue mm²', fmt(sm.tissue_mm2, 3)));
  else if (sm.area_mm2 != null) tiles.append(tile('Area mm²', fmt(sm.area_mm2, 3)), tile('Classes', String(sm.classes.length)));
  else tiles.append(tile('Classes', String(sm.classes.length)), tile('per mm²', '–', 'needs µm per pixel and scikit-image for the tissue area'));
  const stack = h('div', { class: 'stack' }, ...sm.classes.map((c) => h('span', { style: `width: ${c.pct}%; background: ${c.color}`, title: `${c.name} ${c.pct}%` })));
  const legend = h('div', { class: 'row', style: 'flex-wrap: wrap; gap: 4px 12px; font-size: 11px; color: var(--dim)' },
    ...sm.classes.slice(0, 10).map((c) => h('span', {}, h('span', { class: 'sw', style: `background: ${c.color}; width: 8px; height: 8px; margin-right: 4px` }), `${c.name} ${c.pct}%`)));
  const cols = sm.columns.slice(0, 2);
  const head = h('tr', {}, h('th', {}, 'Class'), h('th', {}, 'n'), h('th', {}, '%'), ...cols.map((c) => h('th', { title: c }, c.length > 12 ? c.slice(0, 11) + '…' : c)),
    sm.area_mm2 != null ? h('th', {}, 'mm²') : null);
  const rows = sm.classes.map((c) => h('tr', {}, h('td', {}, h('span', { class: 'sw', style: `background: ${c.color}; width: 8px; height: 8px; margin-right: 6px` }), c.name),
    h('td', {}, c.n.toLocaleString('en-US')), h('td', {}, fmt(c.pct)), ...cols.map((k) => h('td', {}, fmt(c.means[k]))), sm.area_mm2 != null ? h('td', {}, fmt(c.area_mm2, 3)) : null));
  body.replaceChildren(tiles, stack, legend, h('div', { class: 'tablebox' }, h('table', { class: 'ct' }, head, ...rows)));
}

function refreshDrawLists() {
  const p = curPane();
  const ls = layersOf(p);
  const names = ls.map((L) => L.name);
  $('a-layers').replaceChildren(...names.map((n) => h('option', { value: n })));
  const classes = new Set(ls.flatMap((L) => (L.meta.classes || []).map((c) => c.name)));
  $('a-classes').replaceChildren(...[...classes].filter(Boolean).map((n) => h('option', { value: n })));
  const keep = (sel, extra) => { const v = sel.value; sel.replaceChildren(...extra, ...names.map((n) => h('option', { value: n }, n))); if ([...sel.options].some((o) => o.value === v)) sel.value = v; };
  const regionsWas = $('m-regions').value;
  keep($('m-regions'), []);
  if (!regionsWas) {   // regions: the smallest layer that is not an analysis result (drawn ROIs, annotation files)
    const small = ls.filter((L) => L.meta.style?.origin !== 'analysis').sort((a, b) => (a.meta.count ?? 0) - (b.meta.count ?? 0))[0];
    if (small) $('m-regions').value = small.name;
  }
  keep($('m-objects'), [h('option', { value: '' }, '(none)')]);
  if (!$('m-objects').value) {
    const big = ls.filter((L) => (L.meta.count ?? 0) > 500).map((L) => L.name)[0];
    if (big) $('m-objects').value = big;
  }
}

$('m-run').onclick = async () => {
  const sid = activeSlide(), layer = $('m-regions').value;
  if (!sid || !layer) return;
  const q = new URLSearchParams({ layer, intensity: $('m-int').checked, texture: $('m-tex').checked });
  if ($('m-objects').value) q.set('objects', $('m-objects').value);
  const url = `${API}/slides/${enc(sid)}/annotations/metrics?${q}`;
  $('m-run').disabled = true;
  log(`measuring ${layer}…`);
  try {
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    renderMetricsTable(j);
    $('m-csv').hidden = false;
    $('m-csv').href = `${url}&fmt=csv`;
    log(`${j.rows.length} region${j.rows.length === 1 ? '' : 's'} × ${j.columns.length} measurements`);
  } catch (err) { log(`measure: ${err.message || err}`, true); }
  $('m-run').disabled = false;
};
function renderMetricsTable(j) {
  // regions as columns, measurements as rows: hundreds of variables stay readable in a narrow panel
  const ri = j.columns.indexOf('region');
  const t = h('table', {}, h('tr', {}, h('th', {}, 'measurement'), ...j.rows.map((r) => h('th', { style: 'position: static' }, r[ri]))));
  j.columns.forEach((c, i) => {
    if (c === 'region' || c === 'index') return;
    t.append(h('tr', {}, h('th', {}, c), ...j.rows.map((r) => h('td', {}, typeof r[i] === 'number' ? fmt(r[i], 4) : r[i] ?? '–'))));
  });
  $('mtable').replaceChildren(t);
  $('mtable').style.display = 'block';
}

// ------------------------------------------------------------------------------------------------ Draw tab
const applyDraw = () => { grid.setDrawLayer($('a-layer').value.trim() || 'Annotations'); grid.setDrawClass($('a-class').value.trim() || null, $('a-color').value); };
['a-layer', 'a-class', 'a-color'].forEach((id) => $(id).addEventListener('change', applyDraw));
$('a-sam').onclick = () => setTool(grid.tool === 'sam' ? 'pan' : 'sam');
$('a-accept').onclick = () => grid.samAccept();
$('a-discard').onclick = () => grid.samClear();
let samChecked = false;
async function samStatus() {
  if (samChecked) return;
  samChecked = true;
  try {
    const st = await (await fetch(`${API}/sam/status`)).json();
    $('a-sam-status').textContent = st.loaded ? `SAM: ${st.backend} on ${st.device}` : `SAM backends installed: ${st.available.join(', ')} (loads on first click)`;
  } catch (_) { $('a-sam-status').textContent = 'SAM: not available on this server'; }
}
grid.addEventListener('samstart', () => log('SAM: segmenting…'));
grid.addEventListener('sam', (e) => {
  const d = e.detail;
  if (d.error) { log(`SAM: ${d.error}`, true); return; }
  log(`SAM (${d.backend}): ${d.polygons.length} region${d.polygons.length === 1 ? '' : 's'}, score ${(d.score ?? 0).toFixed(2)}`
      + (d.encode_ms != null ? `, encoded in ${d.encode_ms} ms` : '') + `, ${d.decode_ms} ms — Enter to accept, Esc to discard`);
  $('a-sam-status').textContent = `SAM: ${d.backend}`;
});
grid.addEventListener('draw', (e) => {
  toast({ kind: 'good', title: `Saved to ${e.detail?.layer || $('a-layer').value || 'Annotations'}`, sub: $('a-class').value || '', ms: 3000 });
  setTimeout(scheduleInspector, 300);
});

// ------------------------------------------------------------------------------------------------ HUD on the slide
function placeIn(elm, cx, cy, dx = 16, dy = 12) {
  const r = $('main').getBoundingClientRect();
  const w = elm.offsetWidth, hh = elm.offsetHeight;
  let x = cx - r.left + dx, y = cy - r.top + dy;
  if (x + w > r.width - 8) x = cx - r.left - w - dx;
  if (y + hh > r.height - 8) y = Math.max(8, r.height - hh - 8);
  elm.style.left = `${Math.max(8, x)}px`;
  elm.style.top = `${Math.max(8, y)}px`;
}

let lastClick = { x: 0, y: 0 };
grid.addEventListener('pointerdown', (e) => { lastClick = { x: e.clientX, y: e.clientY }; closeRadial(); closeMenu(); }, true);
grid.addEventListener('paneclick', () => hideCard());
grid.addEventListener('objectclick', async (e) => {
  const d = e.detail;
  const sid = slides.find((x) => x.id === d.slide) ? d.slide : panes[d.pane];
  let f = null;
  try { f = await api(`/slides/${enc(sid)}/annotations/feature/${d.index}?layer=${enc(d.layer)}&pct=1`); } catch (_) { f = { properties: d.properties || {}, percentiles: {}, order: [] }; }
  showCard(d, f);
});
function hideCard() { $('card').style.display = 'none'; }
function showCard(d, f) {
  const card = $('card');
  const p = grid.panes[d.pane];
  const L = p?.slide?.layers.get(d.layer);
  const color = (L?.meta.classes || []).find((c) => c.name === d.class)?.color || '#9aa3ad';
  const props = f.properties || {};
  const pct = f.percentiles || {};
  const order = (f.order || []).filter((k) => k in props).concat(Object.keys(props).filter((k) => typeof props[k] === 'number' && !(f.order || []).includes(k)));
  const nums = order.filter((k) => typeof props[k] === 'number').slice(0, 6);
  const text = Object.keys(props).filter((k) => typeof props[k] === 'string' && k !== 'classification').slice(0, 3);
  const close = h('button', { class: 'btn icon ghost', 'aria-label': 'Close' }, icon('close', 14));
  close.onclick = hideCard;
  const zoom = h('button', { class: 'btn grow' }, 'Zoom in');
  zoom.onclick = () => grid.goTo(d.pane, d.centroid[0], d.centroid[1], { magnification: 40 });
  const copy = h('button', { class: 'btn grow' }, 'Copy');
  copy.onclick = () => { navigator.clipboard?.writeText(JSON.stringify({ layer: d.layer, index: d.index, class: d.class, x: d.centroid[0], y: d.centroid[1], ...props })); toast({ title: 'Copied', sub: `${d.layer} #${d.index}`, ms: 2000 }); };
  card.replaceChildren(
    h('div', { class: 'row' }, h('span', { style: `width: 10px; height: 10px; border-radius: 3px; background: ${color}` }), h('b', { style: 'font-weight: 700' }, d.class || 'object'),
      h('span', { class: 'grow' }), h('span', { class: 'mono dim', style: 'font-size: 11px' }, `${d.layer} #${d.index.toLocaleString('en-US')}`), close),
    nums.length ? h('div', { class: 'bars' }, ...nums.flatMap((k) => [
      h('span', { class: 'dim', title: k, style: 'overflow: hidden; text-overflow: ellipsis; white-space: nowrap' }, k),
      pct[k] != null ? h('span', { class: 'bar', title: `${pct[k]} percentile in ${d.layer}` }, h('span', { style: `width: ${pct[k]}%; background: ${color}` })) : h('span'),
      h('span', { class: 'mono', style: 'text-align: right' }, fmt(props[k])),
    ])) : null,
    nums.some((k) => pct[k] != null) ? h('div', { class: 'dim', style: 'font-size: 11px; margin-top: 6px' }, 'bars: percentile within this layer') : null,
    ...text.map((k) => h('div', { class: 'row', style: 'font-size: 12px; margin-top: 4px' }, h('span', { class: 'dim' }, k), h('span', { class: 'grow', style: 'text-align: right' }, props[k]))),
    h('div', { class: 'row', style: 'margin-top: 10px' }, zoom, copy));
  $('tip').style.display = 'none';
  card.style.display = 'block';
  card.classList.remove('rise'); void card.offsetWidth; card.classList.add('rise');
  placeIn(card, lastClick.x, lastClick.y, 18, -40);
}

grid.addEventListener('objecthover', (e) => {
  const tip = $('tip'), d = e.detail;
  if (!d || $('card').style.display === 'block' || $('radial').style.display === 'block') { tip.style.display = 'none'; return; }
  tip.replaceChildren(h('span', { class: 'sw', style: `background: ${d.color || '#9aa3ad'}; width: 8px; height: 8px; margin-right: 6px` }), h('b', { style: 'font-weight: 600' }, d.class || 'object'),
    h('span', { class: 'dim' }, ` · ${d.layer}`));
  tip.style.display = 'block';
  placeIn(tip, d.clientX, d.clientY, 14, 14);
  hoveredLayer = d.layer;
});

let radialCtx = null;
grid.addEventListener('panecontext', (e) => {
  const d = e.detail;
  radialCtx = d;
  hideCard();
  const r = $('radial');
  const m = $('main').getBoundingClientRect();
  r.style.left = `${Math.max(0, Math.min(m.width - 250, d.clientX - m.left - 125))}px`;
  r.style.top = `${Math.max(0, Math.min(m.height - 250, d.clientY - m.top - 125))}px`;
  for (const b of r.querySelectorAll('button')) b.disabled = (b.dataset.r === 'hide' || b.dataset.r === 'solo') && !d.object;
  r.querySelector('[data-r=hide]').firstChild.textContent = d.object ? `Hide ${d.object.layer.length > 8 ? 'layer' : d.object.layer} ` : 'Hide layer ';
  r.style.display = 'block';
  $('tip').style.display = 'none';
});
function closeRadial() { $('radial').style.display = 'none'; radialCtx = null; }
for (const b of $('radial').querySelectorAll('button')) {
  b.onclick = () => {
    const d = radialCtx;
    closeRadial();
    if (!d) return;
    const act = b.dataset.r;
    if (act === 'hide' && d.object) toggleLayer(d.object.layer);
    else if (act === 'solo' && d.object) toggleSolo(d.object.layer);
    else if (act === 'sam') setTool('sam');
    else if (act === 'measure') setTab('measure');
    else if (act === 'fit') grid.fit('all');
    else if (act === 'copy') {
      const mpp = grid.panes[d.pane]?.slide?.info?.mpp;
      const t = mpp ? `${(d.x * mpp[0]).toFixed(1)}, ${(d.y * mpp[1]).toFixed(1)} µm` : `${d.x.toFixed(0)}, ${d.y.toFixed(0)} px`;
      navigator.clipboard?.writeText(t);
      toast({ title: 'Copied', sub: t, ms: 2000 });
    }
  };
}
document.addEventListener('pointerdown', (e) => {
  if (!e.target.closest('#radial')) closeRadial();
  if (menuEl && !e.target.closest('.menu')) closeMenu();
});

// ------------------------------------------------------------------------------------------------ keyboard
document.addEventListener('keydown', (e) => {
  const mod = isMac ? e.metaKey : e.ctrlKey;
  if (mod && e.key.toLowerCase() === 'o') { e.preventDefault(); e.shiftKey ? openFolder() : openSlides(); return; }
  if (mod && e.key.toLowerCase() === 'l') { e.preventDefault(); addLayer(); return; }
  if (e.key === 'Escape') { hideCard(); closeRadial(); closeMenu(); return; }
  if (mod || e.altKey || typing(e.target) || document.querySelector('dialog[open]')) return;
  const k = e.key;
  if (k >= '1' && k <= '8' && !e.repeat) { const [t, label] = TOOLS[+k - 1]; setTool(t); keyFlash(k, label); return; }
  const tabs = { l: ['layers', 'Layers'], a: ['analyze', 'Analyze'], m: ['measure', 'Measure'], d: ['draw', 'Draw'] };
  const lk = k.toLowerCase();
  if (tabs[lk] && !e.shiftKey) { setTab(tabs[lk][0]); keyFlash(k.toUpperCase(), tabs[lk][1]); return; }
  if (lk === 's' && hoveredLayer) { toggleSolo(hoveredLayer); keyFlash('S', `solo ${hoveredLayer}`); return; }
  if (lk === 'h' && hoveredLayer) { toggleLayer(hoveredLayer); keyFlash('H', `${hoveredLayer}`); return; }
  if (lk === 'f') { grid.fit('all'); keyFlash('F', 'fit'); return; }
  if (k === '?') { $('help').showModal(); return; }
  if (k === '/') { e.preventDefault(); setTab('layers'); $('l-filter').focus(); }
});

// ------------------------------------------------------------------------------------------------ wiring
const openSlides = async () => onResult(await call('pick_slides'));
const openFolder = async () => onResult(await call('pick_folder'));
const addLayer = async () => { if (activeSlide()) onResult(await call('pick_layer', activeSlide())); };
const openConnect = () => { $('c-msg').textContent = ''; $('connect').showModal(); $('c-spec').focus(); };
function wire() {
  buildTools();
  const sel = $('n');
  for (let i = 1; i <= MAX_PANES; i++) sel.add(new Option(i, i));
  sel.onchange = () => setPaneCount(+sel.value);
  $('sync').onchange = (e) => grid.setSync(e.target.value);
  $('b-open').onclick = openSlides;
  $('b-folder').onclick = openFolder;
  $('b-layer').onclick = addLayer;
  $('l-add').onclick = addLayer;
  $('fit').onclick = () => grid.fit('all');
  $('b-side').onclick = () => $('side').classList.toggle('hidden');
  $('b-insp').onclick = () => { toggleInspector(); renderInspector(); };
  $('b-help').onclick = () => $('help').showModal();
  $('b-connect').onclick = openConnect;
  for (const b of $('tabs').children) b.onclick = () => setTab(b.dataset.tab);
  for (const b of $('l-scope').children) b.onclick = () => { for (const x of $('l-scope').children) x.setAttribute('aria-pressed', String(x === b)); };
  $('l-filter').oninput = () => renderLayers();
  $('ms-layer').onchange = () => renderMeasure();
  $('an-objects').onchange = () => renderAnalyze();
  $('disp').ontoggle = () => renderDisplay(curPane());
  const acts = { open: openSlides, folder: openFolder, connect: openConnect };
  document.querySelectorAll('[data-act]').forEach((b) => (b.onclick = acts[b.dataset.act]));
  $('c-form').onsubmit = async (e) => {
    if (e.submitter && e.submitter.value === 'cancel') return;   // method=dialog closes it
    e.preventDefault();
    const msg = $('c-msg'), spec = $('c-spec').value.trim();
    msg.className = '';
    if (!spec) { msg.textContent = 'Paste a service description or a .json path'; msg.className = 'err'; return; }
    $('c-go').disabled = true;
    msg.textContent = 'Connecting…';
    const res = await call('connect', spec, $('c-images').value.trim() || null);
    $('c-go').disabled = false;
    if (!res.connected) { msg.textContent = (res.errors || ['failed']).join(' · '); msg.className = 'err'; return; }
    $('connect').close();
    onResult(res);
  };
  $('b-paths').onclick = async () => {
    const list = $('paths').value.split(/\r?\n/).map((x) => x.trim().replace(/^"(.*)"$/, '$1')).filter(Boolean);
    if (list.length) await openPaths(list);
    $('paths').value = '';
  };
  grid.addEventListener('activepane', (e) => { active = e.detail.pane; renderList(); renderCrumb(); hideCard(); scheduleInspector(); });
  grid.addEventListener('slideload', () => { renderCrumb(); scheduleInspector(); });
  grid.addEventListener('layers', () => scheduleInspector());
  // drag-over highlight (the drop itself is read natively, with full paths)
  let dragDepth = 0;
  document.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('drag'); });
  document.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('drag'); } });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault(); dragDepth = 0; document.body.classList.remove('drag');
    if (!native() && CAPS.open) showPathBox('drop');         // a browser can't see a dropped file's path
  });
  applyDraw();
  grid.hoverEvents = true;
  grid.contextEvents = true;
}

/** Show these slide ids in panes (1–32), as if picked in the slide list. */
function showSlides(ids) { panes = ids.filter((id) => slides.some((x) => x.id === id)).slice(0, MAX_PANES); active = 0; apply(); }

window.wissiApp = { onResult, activeSlide, showSlides, setTool, setTab, toast };

async function start() {
  wire();
  await customElements.whenDefined('wissi-viz');
  const st = await call('state');
  slides = st.slides || [];
  // ?show=a,b (an embedding page picks the panes), else the session's choice, else the first four slides
  const asked = (new URLSearchParams(location.search).get('show') || '').split(',').filter((id) => slides.some((x) => x.id === id));
  panes = (asked.length ? asked : st.show && st.show.length ? st.show : slides.slice(0, Math.min(4, slides.length)).map((x) => x.id)).slice(0, MAX_PANES);
  const tabAsked = new URLSearchParams(location.search).get('tab');
  if (tabAsked) setTab(tabAsked);
  apply();
  renderRecent(st.recent);
  if (!CAPS.open) {
    for (const id of ['b-open', 'b-folder', 'b-layer']) $(id).hidden = true;
    document.querySelectorAll('#empty [data-act=open], #empty [data-act=folder]').forEach((b) => (b.hidden = true));
  }
  $('b-connect').hidden = !CAPS.connect;
  document.querySelector('#empty [data-act=connect]').hidden = !CAPS.connect;
  if (SERVER) {
    document.querySelector('#empty h1').textContent = slides.length ? 'Pick a slide'
      : CAPS.open ? 'Open a slide from this server' : CAPS.connect ? 'Connect to a tile API' : 'No slides on this server';
    document.querySelector('#side .hint').textContent = 'Click a slide to show it in the active pane; double-click to add a pane.'
      + (CAPS.open || CAPS.connect ? ' The list is shared by everyone using this server.' : '');
    log(`${slides.length} slide${slides.length === 1 ? '' : 's'} on this server · press ? for shortcuts`);
  } else if (!(await nativeReady)) log('Browser window: paste paths to open files · press ? for shortcuts');
  else log('Press ? for keyboard shortcuts');
}
start();
