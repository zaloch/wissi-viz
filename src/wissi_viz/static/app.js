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
  if (name === 'projects') return TOKEN ? quiet(http(`${BASE}/app/projects`, { projects: args[0] })) : { projects: args[0] };
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
  $('vtools').hidden = !panes.length;
  if (!panes.length) $('minimap').hidden = true;
  else if (mapOn) $('minimap').hidden = false;
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

// ---- the slide list: your projects first, then the folders the other slides are in
let projects = [];   // [{name, slides: [path, …]}] (paths, so they survive restarts)
const lsGet = (k, d = null) => { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* storage blocked: per-session only */ } };
const closedGroups = new Set(JSON.parse(lsGet('wissi-closed-groups', '[]')));
const slidePath = (sl) => sl.path || sl.name;
const isRemote = (sl) => /^[a-z0-9_-]+:/i.test(sl.path || '') && !/^[a-z]:[\\/]/i.test(sl.path || '');
function folderOf(sl) {
  if (isRemote(sl)) { const svc = (sl.path || '').split(':')[0]; return { key: `api:${svc}`, label: svc, kind: 'tile API', title: `Images of ${svc}` }; }
  const parts = (sl.path || '').split(/[\\/]/);
  parts.pop();
  const dir = parts.join('/') || '/';
  return { key: `dir:${dir}`, label: parts.filter(Boolean).slice(-2).join('/') || dir, kind: 'folder', title: dir };
}
async function saveProjects(render = true) {
  const r = await call('projects', projects);
  if (r?.projects && !renaming) projects = r.projects;
  else if (r?.errors) toast({ kind: 'bad', title: 'Projects not saved', sub: r.errors[0] });
  if (render && !renaming) renderList();   // never while a name is being typed: that would wipe it
}
function uniqueName(base) { let n = 1, name = base; while (projects.some((p) => p.name === name)) name = `${base} ${++n}`; return name; }
function newProject(seed = []) {
  const name = uniqueName('New project');
  projects.unshift({ name, slides: seed.map(slidePath) });
  renaming = name;
  renderList();   // the name field at once (saved when it is named)
}
function addToProject(name, sl) {
  const pr = projects.find((x) => x.name === name);
  if (!pr || pr.slides.includes(slidePath(sl))) return;
  pr.slides.push(slidePath(sl));
  saveProjects();
  toast({ title: `Added to ${name}`, sub: sl.name, ms: 2000 });
}
function removeFromProject(name, sl) {
  const pr = projects.find((x) => x.name === name);
  if (!pr) return;
  pr.slides = pr.slides.filter((x) => x !== slidePath(sl));
  saveProjects();
}
let renaming = null;
function toggleGroup(key) {
  closedGroups.has(key) ? closedGroups.delete(key) : closedGroups.add(key);
  lsSet('wissi-closed-groups', JSON.stringify([...closedGroups]));
  renderList();
}

function renderList() {
  const box = $('slides');
  const q = ($('s-filter').value || '').trim().toLowerCase();
  const match = (sl) => !q || sl.name.toLowerCase().includes(q) || (sl.path || '').toLowerCase().includes(q);
  const byPath = new Map(slides.map((sl) => [slidePath(sl), sl]));
  const inProject = new Set();
  const groups = projects.map((pr) => {
    const items = pr.slides.map((x) => byPath.get(x)).filter(Boolean);
    items.forEach((sl) => inProject.add(sl.id));
    return { key: `proj:${pr.name}`, label: pr.name, kind: 'project', project: pr, items, title: `${items.length} slide${items.length === 1 ? '' : 's'}` };
  });
  const folders = new Map();
  for (const sl of slides) {
    if (inProject.has(sl.id)) continue;
    const f = folderOf(sl);
    if (!folders.has(f.key)) folders.set(f.key, { ...f, items: [] });
    folders.get(f.key).items.push(sl);
  }
  groups.push(...folders.values());
  box.replaceChildren();
  for (const g of groups) {
    const items = g.items.filter(match);
    if (q && !items.length) continue;
    const closed = closedGroups.has(g.key) && !q;
    const sec = h('section', { class: `sgroup${closed ? ' closed' : ''}`, dataset: { key: g.key, kind: g.kind }, role: 'group', 'aria-label': g.label });
    const tw = h('button', { class: 'tw', 'aria-label': `${closed ? 'Expand' : 'Collapse'} ${g.label}`, 'aria-expanded': String(!closed) },
      s('svg', { width: 12, height: 12, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2.5, 'stroke-linecap': 'round', 'aria-hidden': 'true' }, s('path', { d: 'M6 9l6 6 6-6' })));
    tw.onclick = () => toggleGroup(g.key);
    let gn;
    if (renaming === g.label && g.kind === 'project') {
      gn = h('input', { class: 'gn', value: g.label, 'aria-label': 'Project name', style: 'font-size: 12px; padding: 2px 6px' });
      let finished = false;
      const done = (ok) => {
        if (finished) return;   // Enter, then the blur of the removed input
        finished = true;
        const v = gn.value.trim();
        renaming = null;
        if (ok && v && v !== g.label && !projects.some((x) => x.name === v)) {
          if (closedGroups.delete(g.key)) lsSet('wissi-closed-groups', JSON.stringify([...closedGroups]));
          g.project.name = v;
          saveProjects();
        } else {   // unchanged: put the name back in place (re-rendering the list here would swallow the click that blurred it)
          saveProjects(false);   // a new project keeps its default name (no re-render: see above)
          const span = h('span', { class: 'gn', title: g.title || g.label }, g.label);
          span.onclick = () => toggleGroup(g.key);
          span.ondblclick = () => { renaming = g.label; renderList(); };
          gn.replaceWith(span);
        }
      };
      gn.onkeydown = (e) => { if (e.key === 'Enter') done(true); else if (e.key === 'Escape') done(false); e.stopPropagation(); };
      gn.onblur = () => done(true);

    } else {
      gn = h('span', { class: 'gn', title: g.title || g.label }, g.label);
      gn.onclick = () => toggleGroup(g.key);
      if (g.kind === 'project') gn.ondblclick = () => { renaming = g.label; renderList(); };
    }
    const more = h('button', { class: 'btn icon ghost more', title: 'More', 'aria-label': `More for ${g.label}` }, icon('more', 14));
    more.onclick = (e) => openMenu(e, g.kind === 'project' ? [
      ['Show in panes', '', () => showSlides(items.map((x) => x.id))],
      ['Rename', '', () => { renaming = g.label; renderList(); }],
      'hr',
      ['Delete project', '', () => { projects = projects.filter((x) => x !== g.project); saveProjects(); }, 'danger', 'The slides stay open'],
    ] : [
      ['Show in panes', '', () => showSlides(items.map((x) => x.id))],
      ['Make a project of these', '', () => newProject(items)],
    ]);
    sec.append(h('div', { class: 'ghead' }, tw, gn, h('span', { class: 'gk' }, g.kind === 'project' ? 'project' : g.kind), h('span', { class: 'chip' }, String(items.length)), more));
    const ul = h('ul', {});
    for (const sl of items) ul.append(slideItem(sl, g));
    if (g.kind === 'project' && !items.length) ul.append(h('li', { class: 'dim', style: 'font-size: 12px; cursor: default; padding: 6px 8px' }, 'Drag slides here, or use a slide’s ⋯ menu'));
    sec.append(ul);
    if (g.kind === 'project') {   // drop slides on a project
      sec.ondragover = (e) => { if (e.dataTransfer.types.includes('text/wissi-slide')) { e.preventDefault(); e.stopPropagation(); sec.classList.add('drop'); } };
      sec.ondragleave = () => sec.classList.remove('drop');
      sec.ondrop = (e) => {
        const id = e.dataTransfer.getData('text/wissi-slide');
        e.preventDefault(); e.stopPropagation(); sec.classList.remove('drop');
        const sl = slides.find((x) => x.id === id);
        if (sl) addToProject(g.label, sl);
      };
    }
    box.append(sec);
  }
  if (q && !box.children.length) box.append(h('div', { class: 'panel-note', style: 'padding: 8px' }, `No slide matches “${q}”.`));
  const nameField = box.querySelector('input.gn');
  if (nameField && document.activeElement !== nameField) { nameField.focus(); nameField.select(); }
  $('s-count').textContent = slides.length ? String(slides.length) : '';
}

function slideItem(sl, g) {
  const shown = panes.map((id, i) => (id === sl.id ? i + 1 : 0)).filter(Boolean);
  const img = h('img', { class: 'thumb', alt: '', loading: 'lazy', src: `${API}/slides/${enc(sl.id)}/thumbnail?size=160` });
  img.onerror = () => { img.removeAttribute('src'); img.classList.add('blank'); };
  const x = h('button', { title: 'Close', 'aria-label': `Close ${sl.name}`, hidden: !CAPS.close, style: 'order: 3' }, '✕');
  x.onclick = async (e) => {
    e.stopPropagation();
    const r = await call('close_slide', sl.id);
    if (r?.slides) slides = r.slides;
    panes = panes.map((id) => (id === sl.id ? null : id)).filter(Boolean);
    apply();
  };
  const more = h('button', { class: 'more', title: 'More', 'aria-label': `More for ${sl.name}`, style: 'order: 2' }, '⋯');
  more.onclick = (e) => {
    e.stopPropagation();
    const items = [
      ['Show in the active pane', '', () => { if (!panes.length) panes = [sl.id]; else panes[active] = sl.id; apply(); }],
      ['Add as a new pane', '', () => addPane(sl.id)],
      'hr',
      ...projects.filter((pr) => !pr.slides.includes(slidePath(sl))).slice(0, 8).map((pr) => [`Add to ${pr.name}`, '', () => addToProject(pr.name, sl)]),
      ['New project with this slide', '', () => newProject([sl])],
    ];
    if (g.kind === 'project') items.push(['Remove from this project', '', () => removeFromProject(g.label, sl)]);
    openMenu(e, items);
  };
  const li = h('li', { class: (shown.length ? 'shown' : '') + (sl.id === activeSlide() ? ' active' : ''), title: sl.path || sl.name, draggable: 'true', role: 'listitem' },
    img, h('span', { class: 'txt' }, h('span', { class: 'name' }, sl.name),
      h('span', { class: 'sub' }, isRemote(sl) ? `${(sl.path || '').split(':')[0]} · tile API` : shown.length ? `pane ${shown.join(', ')}` : '')),
    h('span', { class: 'pane' }, shown.join(' ')), x, more);
  li.onclick = () => { if (!panes.length) panes = [sl.id]; else panes[active] = sl.id; apply(); };
  li.ondblclick = () => addPane(sl.id);
  li.ondragstart = (e) => { e.dataTransfer.setData('text/wissi-slide', sl.id); e.dataTransfer.effectAllowed = 'copy'; };
  return li;
}
function addPane(id) { if (panes.length < MAX_PANES && !panes.includes(id)) { panes.push(id); active = panes.length - 1; apply(); } }

/** A small menu at the clicked element: entries [label, key, fn, class?, title?] or 'hr'. */
function openMenu(ev, entries) {
  ev.stopPropagation();
  closeMenu();
  const el = h('div', { class: 'menu', role: 'menu' });
  for (const e of entries) {
    if (e === 'hr') { if (el.lastChild && el.lastChild.tagName !== 'HR') el.append(h('hr')); continue; }
    const [label, k, fn, cls, title] = e;
    const b = h('button', { class: cls || '', role: 'menuitem', title: title || null }, label, k ? h('span', { class: 'kc' }, k) : null);
    b.onclick = (x) => { x.stopPropagation(); if (fn(b) !== false) closeMenu(); };
    el.append(b);
  }
  if (el.lastChild?.tagName === 'HR') el.lastChild.remove();
  menuEl = el;
  document.body.append(el);
  const r = ev.currentTarget.getBoundingClientRect();
  el.style.left = `${Math.max(8, Math.min(window.innerWidth - el.offsetWidth - 8, r.right - el.offsetWidth))}px`;
  el.style.top = `${Math.min(window.innerHeight - el.offsetHeight - 8, r.bottom + 4)}px`;
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

// ---- the in-page file picker (browser window / server): places, a clickable path, filter, multi-select
const pk = { dir: '', parent: null, entries: [], sel: new Set(), mode: 'files', root: '' };
function showPathBox(why) {
  pk.mode = why === 'pick_folder' ? 'folder' : why === 'pick_layer' ? 'layer' : 'files';
  $('empty').style.display = '';
  $('empty').classList.add('picking');
  $('pathbox').style.display = 'block';
  $('paths').dataset.slide = pk.mode === 'layer' ? activeSlide() || '' : '';
  $('paths').placeholder = SERVER ? 'Or paste paths on the server, one per line' : 'Or paste file or folder paths, one per line';
  $('pk-folder').hidden = pk.mode === 'layer';
  $('pk-open').textContent = pk.mode === 'layer' ? 'Add to the slide' : 'Open selected';
  if (CAPS.browse) { $('picker').classList.add('on'); browse(pk.dir || ''); setTimeout(() => $('pk-filter').focus(), 50); }
  else $('paths').focus();
  log(pk.mode === 'folder' ? 'Pick a folder: go into it, then “Open this folder”'
    : pk.mode === 'layer' ? 'Pick cell tables or annotation files for the active slide'
      : SERVER ? 'Pick slides on the server' : 'Pick slides: click to select, double-click to open (native dialogs need the desktop window)');
}
function closePicker() {
  $('picker').classList.remove('on');
  $('pathbox').style.display = 'none';
  $('empty').classList.remove('picking');
  $('empty').style.display = panes.length ? 'none' : '';
}
const openPaths = async (list) => {
  const res = await call('open_paths', list, $('paths').dataset.slide || null);
  if (res && !(res.errors || []).length) closePicker();
  onResult(res);
};
let browseSeq = 0;
async function browse(dir) {
  const seq = ++browseSeq;
  $('pk-filter').value = '';   // now, not when the listing arrives: that would wipe what is typed meanwhile
  const res = await call('browse', dir);
  if (seq !== browseSeq) return;   // a later folder was asked for meanwhile (a slow listing must not overwrite it)
  if (!res || res.errors) { if (res) { log(res.errors.join(' · '), true); $('pk-info').textContent = res.errors[0]; } return; }
  pk.dir = res.dir; pk.parent = res.parent; pk.entries = res.entries; pk.sel.clear();
  $('browser').dataset.dir = res.dir;
  renderPlaces(res.places || []);
  renderCrumbs(res.dir);
  renderEntries();
}
function renderPlaces(places) {
  const nav = $('pk-places');
  nav.replaceChildren();
  let group = '';
  for (const pl of places) {
    if (pl.group !== group) { group = pl.group; nav.append(h('span', { class: 'lbl' }, group)); }
    const b = h('button', { class: pk.dir === pl.path ? 'on' : '', title: pl.path }, pl.name);
    b.onclick = () => browse(pl.path);
    nav.append(b);
  }
}
function renderCrumbs(dir) {
  const box = $('pk-crumbs');
  box.replaceChildren();
  $('pk-up').disabled = pk.parent == null;
  if (!dir) { box.append(h('button', {}, 'Folders you can open from')); return; }
  const win = /^[a-z]:[\\/]/i.test(dir) || dir.includes('\\');
  const sep = win ? '\\' : '/';
  const parts = dir.split(/[\\/]/);
  let acc = '';
  parts.forEach((part, i) => {
    acc = i === 0 ? (part || sep) : (acc.endsWith(sep) ? acc + part : acc + sep + part);
    if (!part && i > 0) return;
    const target = acc;
    const b = h('button', { title: target }, i === 0 && !part ? sep : part);
    b.onclick = () => browse(target);
    box.append(b);
    if (i < parts.length - 1) box.append(h('span', { class: 'dim' }, '›'));
  });
  box.scrollLeft = box.scrollWidth;
}
const human = (n) => (n == null ? '' : n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`);
function renderEntries() {
  const box = $('browser');
  const q = $('pk-filter').value.trim().toLowerCase();
  const wanted = pk.mode === 'layer' ? ['folder', 'table', 'annotations'] : pk.mode === 'folder' ? ['folder', 'slide'] : ['folder', 'slide', 'table', 'annotations'];
  const list = pk.entries.filter((e) => wanted.includes(e.kind) && (!q || e.name.toLowerCase().includes(q)));
  box.replaceChildren();
  const label = { folder: 'folder', slide: 'slide', table: 'cells', annotations: 'ann.' };
  for (const e of list) {
    const isDir = e.kind === 'folder';
    const sel = pk.sel.has(e.path);
    const cb = isDir || pk.mode === 'folder' ? h('span') : h('input', { type: 'checkbox', checked: sel, 'aria-label': `Select ${e.name}`, tabindex: -1 });
    const ic = isDir ? s('svg', { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.8, 'stroke-linejoin': 'round', 'aria-hidden': 'true' }, s('path', { d: 'M3 6h6l2 2h10v11H3z' }))
      : h('span', { class: 'k' }, label[e.kind] || e.kind);
    const a = h('a', { href: '#', title: e.path, class: sel ? 'sel' : '', role: 'listitem', dataset: { kind: e.kind } }, cb, ic, h('span', { class: 'n' }, e.name), h('span', { class: 'z' }, isDir ? '' : human(e.size)));
    a.onclick = (ev) => {
      ev.preventDefault();
      if (isDir) { browse(e.path); return; }
      if (pk.mode === 'folder') return;
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.target === cb) pk.sel.has(e.path) ? pk.sel.delete(e.path) : pk.sel.add(e.path);
      else { const was = pk.sel.has(e.path) && pk.sel.size === 1; pk.sel.clear(); if (!was) pk.sel.add(e.path); }
      renderEntries();
    };
    a.ondblclick = (ev) => { ev.preventDefault(); if (!isDir && pk.mode !== 'folder') openPaths([e.path]); };
    box.append(a);
  }
  if (!list.length) box.append(h('div', { class: 'none' }, q ? `Nothing matches “${q}” here` : 'Nothing wissi-viz opens here'));
  const nSlides = pk.entries.filter((e) => e.kind === 'slide').length;
  $('pk-folder').disabled = !pk.dir || (!nSlides && !pk.entries.some((e) => e.kind === 'folder'));
  $('pk-folder').textContent = nSlides ? `Open this folder (${nSlides} slide${nSlides === 1 ? '' : 's'})` : 'Open this folder';
  $('pk-open').disabled = !pk.sel.size;
  if (pk.sel.size) $('pk-open').textContent = `${pk.mode === 'layer' ? 'Add' : 'Open'} ${pk.sel.size} selected`;
  else $('pk-open').textContent = pk.mode === 'layer' ? 'Add to the slide' : 'Open selected';
  $('pk-info').textContent = pk.mode === 'folder' ? 'Go into the folder, then open it' : pk.sel.size ? 'Shift / ⌘-click to select more' : 'Click to select · double-click to open';
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
  else if (tab === 'image') renderImage();
  else if (tab === 'analyze') renderAnalyze();
  else if (tab === 'measure') renderMeasure();
  else if (tab === 'draw') renderDraw();
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

// ------------------------------------------------------------------------------------------------ Image tab
// per-channel histograms (fluorescence) or R, G, B (brightfield) of the whole image, the view or a CZI scene, with
// draggable minimum / maximum, gamma, colour, show / solo, auto contrast, and linked channels
const histCache = new Map();    // `${sid}|${rect}` → Promise<histogram>
let linkCh = false;
const panesOf = (sid) => (grid.panes || []).filter((q) => q.slide?.info?.id === sid).map((q) => q.index);
function histRect(p) {
  const info = p?.slide?.info;
  const v = $('i-scope').value;
  if (v === 'view' && p?.view) {
    const hw = p.rect.w / 2 / p.view.scale, hh = p.rect.h / 2 / p.view.scale;
    return [p.view.cx - hw, p.view.cy - hh, p.view.cx + hw, p.view.cy + hh].map((x) => Math.round(x));
  }
  const sc = (info?.extra?.scenes || []).find((x) => `scene:${x.index}` === v);
  return sc ? sc.rect : null;
}
function getHist(sid, rect) {
  const key = `${sid}|${rect ? rect.join(',') : 'all'}`;
  if (!histCache.has(key)) {
    const p = api(`/slides/${enc(sid)}/histogram${rect ? `?rect=${rect.join(',')}` : ''}`);
    p.catch(() => histCache.delete(key));
    histCache.set(key, p);
    if (histCache.size > 24) histCache.delete(histCache.keys().next().value);
  }
  return histCache.get(key);
}
async function renderImage() {
  const p = curPane();
  const info = p?.slide?.info;
  const body = $('disp-body'), sum = $('disp-sum');
  const sid = info?.id;
  const sl = slides.find((x) => x.id === activeSlide());
  $('i-title').textContent = sl ? `Image · ${sl.name}` : 'Image';
  // the histogram's scope: whole image, this view, or a scene (CZI files with several scenes)
  const scopeSel = $('i-scope'), keep = scopeSel.value;
  const scenes = info?.extra?.scenes || [];
  scopeSel.replaceChildren(h('option', { value: 'slide' }, 'the whole image'), h('option', { value: 'view' }, 'this view'),
    ...scenes.map((x) => h('option', { value: `scene:${x.index}` }, x.name)));
  scopeSel.value = [...scopeSel.options].some((o) => o.value === keep) ? keep : 'slide';
  const sc = $('i-scenes');
  sc.replaceChildren();
  if (scenes.length) {
    sc.append(`${scenes.length} scenes: `);
    scenes.forEach((x, k) => {
      const b = h('a', { href: '#', style: 'color: var(--accent)' }, x.name);
      b.onclick = (e) => { e.preventDefault(); grid.goToBounds(active, ...x.rect); scopeSel.value = `scene:${x.index}`; renderImage(); };
      sc.append(k ? ', ' : '', b);
    });
  }
  if (!info) { body.replaceChildren(h('div', { class: 'panel-note' }, 'No slide in this pane.')); sum.textContent = ''; return; }
  $('i-link').hidden = info.mode !== 'channels';
  $('i-auto').textContent = info.mode === 'channels' ? 'Auto all' : 'Auto levels';
  let hist = null;
  const rect = histRect(p);
  body.replaceChildren(h('div', { class: 'panel-note' }, 'Reading the histogram…'));
  try { hist = await getHist(sid, rect); } catch (e) { hist = null; log(`histogram: ${e.message}`, true); }
  if (curPane()?.slide?.info?.id !== sid) return;
  body.replaceChildren();
  if (info.mode === 'channels' && p.channels) {
    sum.textContent = `${p.channels.filter((c) => c.visible).length} of ${p.channels.length} channels`;
    p.channels.forEach((c, i) => body.append(channelCard(p, sid, i, c, hist?.channels?.[i])));
    return;
  }
  // brightfield
  const a = p.adjust || { brightness: 0, contrast: 1, gamma: 1 };
  sum.textContent = p.stain ? `stain: ${p.stain.name}` : 'brightfield';
  const stain = h('select', { 'aria-label': 'Stain view' }, ...['off', 'hematoxylin', 'eosin', 'dab', 'fluorescence']
    .map((m) => h('option', { selected: (p.stain?.name || 'off') === m }, m)));
  stain.onchange = () => { grid.setStain(panesOf(sid), stain.value); renderImage(); };
  const card = h('div', { class: 'hist' }, h('div', { class: 'hh' }, h('b', {}, 'R · G · B'), h('span', { class: 'dim', style: 'font-size: 11px' }, hist ? `level ${hist.level}` : '')));
  if (hist) card.append(histCanvas(hist.channels, ['#ff6b6b', '#5fe08f', '#5ea8ff'], null));
  body.append(card, h('div', { class: 'field', style: 'grid-template-columns: 88px 1fr' }, 'Stain view', stain));
  for (const [label, key, min, max, step] of [['Brightness', 'brightness', -0.5, 0.5, 0.01], ['Contrast', 'contrast', 0.3, 3, 0.01], ['Gamma', 'gamma', 0.3, 3, 0.01]]) {
    const out = h('output', {}, fmt(a[key] ?? 0, 2));
    const r = h('input', { type: 'range', min, max, step, value: a[key] ?? 0, 'aria-label': label, id: `i-${key}` });
    r.oninput = () => { grid.setAdjust(panesOf(sid), { [key]: +r.value }); out.textContent = fmt(+r.value, 2); };
    body.append(h('label', { class: 'field' }, label, r, out));
  }
  const inv = h('input', { type: 'checkbox', checked: !!a.invert, id: 'i-invert' });
  inv.onchange = () => grid.setAdjust(panesOf(sid), { invert: inv.checked });
  body.append(h('label', { class: 'dim', style: 'font-size: 12px; display: flex; gap: 6px; align-items: center' }, inv, 'invert'));
}
function autoLevels(sid, hist) {   // brightfield: stretch the darkest 0.1 % … brightest 0.1 % to the full range
  if (!hist) return;
  const lo = Math.min(...hist.channels.map((c) => c.p['0.1'])), hi = Math.max(...hist.channels.map((c) => c.p['99.9']));
  const contrast = Math.max(0.3, Math.min(3, 255 / Math.max(8, hi - lo)));
  grid.setAdjust(panesOf(sid), { contrast, brightness: Math.max(-0.5, Math.min(0.5, (127.5 - (lo + hi) / 2) / 255 * contrast)) });
}

function channelCard(p, sid, i, c, hh) {
  const card = h('div', { class: `hist${c.visible ? '' : ' off'}`, dataset: { ch: String(i) } });
  const vis = h('input', { type: 'checkbox', checked: c.visible, 'aria-label': `Show ${c.name}` });
  vis.onchange = () => { grid.setChannel(panesOf(sid), i, { visible: vis.checked }); renderImage(); };
  const col = h('input', { type: 'color', value: toHex(c.color), 'aria-label': `${c.name} colour` });
  col.oninput = () => { grid.setChannel(panesOf(sid), i, { color: col.value }); draw(); };
  const soloB = h('button', { class: 'solo-b', title: 'Show only this channel (again: show all)' }, 'S');
  soloB.onclick = () => {
    const only = p.channels.every((x, k) => x.visible === (k === i));
    p.channels.forEach((_, k) => grid.setChannel(panesOf(sid), k, { visible: only || k === i }));
    renderImage();
  };
  const autoB = h('button', { class: 'solo-b', title: 'Auto contrast from the histogram (0.1–99.9 %)' }, 'A');
  autoB.onclick = () => { if (hh) setWin(hh.p['0.1'], hh.p['99.9']); };
  card.append(h('div', { class: 'hh' }, vis, col, h('b', { title: c.name }, c.name), h('span', { class: 'dim mono', style: 'font-size: 10px' }, hh ? `max ${fmt(hh.max)}` : ''), autoB, soloB));
  const lo = h('input', { type: 'number', value: Math.round(c.lo), 'aria-label': `${c.name} minimum` });
  const hi = h('input', { type: 'number', value: Math.round(c.hi), 'aria-label': `${c.name} maximum` });
  const ga = h('input', { type: 'number', value: c.gamma ?? 1, step: 0.05, min: 0.1, max: 5, 'aria-label': `${c.name} gamma` });
  const setWin = (a, b) => {
    a = Math.max(0, Math.round(a)); b = Math.max(a + 1, Math.round(b));
    const targets = linkCh ? p.channels.map((_, k) => k) : [i];
    for (const k of targets) grid.setChannel(panesOf(sid), k, { min: a, max: b });
    lo.value = a; hi.value = b;
    if (linkCh) document.querySelectorAll('#disp-body .hist').forEach((el) => el !== card && el._redraw?.());
    draw();
  };
  lo.onchange = () => setWin(+lo.value, +hi.value);
  hi.onchange = () => setWin(+lo.value, +hi.value);
  ga.onchange = () => { grid.setChannel(panesOf(sid), i, { gamma: Math.max(0.1, +ga.value || 1) }); draw(); };
  let draw = () => {};
  if (hh) {
    const cv = histCanvas([hh], [toHex(c.color)], { ch: c, onWin: setWin });
    draw = cv._redraw;
    card._redraw = () => { lo.value = Math.round(c.lo); hi.value = Math.round(c.hi); draw(); };
    card.append(cv);
  }
  card.append(h('div', { class: 'nums' }, h('label', {}, 'min', lo), h('label', {}, 'max', hi), h('label', {}, 'gamma', ga)));
  return card;
}

/** A histogram canvas (log or linear counts) with the display window drawn on it; with `win`, drag the minimum /
 * maximum (or shift-drag the whole window). */
function histCanvas(chans, colors, win) {
  const cv = h('canvas', { width: 340, height: 64, 'aria-label': 'Histogram' + (win ? ': drag the minimum and maximum' : '') });
  const domain = () => Math.max(...chans.map((c) => c.hi), win ? win.ch.hi * 1.02 : 0);
  const redraw = () => {
    const g = cv.getContext('2d'), W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    const D = domain(), logy = $('i-log').checked;
    chans.forEach((c, k) => {
      const n = c.counts.length, max = Math.max(1, ...c.counts);
      g.beginPath();
      g.moveTo(0, H);
      c.counts.forEach((v, j) => {
        const x = ((c.lo + (j + 0.5) * (c.hi - c.lo) / n) / D) * W;
        const y = H - (logy ? Math.log1p(v) / Math.log1p(max) : v / max) * (H - 4);
        g.lineTo(x, y);
      });
      g.lineTo(W, H);
      g.closePath();
      g.fillStyle = colors[k] + (chans.length > 1 ? '55' : '88');
      g.strokeStyle = colors[k];
      g.fill();
      g.lineWidth = 1; g.stroke();
    });
    if (win) {
      const D2 = domain(), x0 = (win.ch.lo / D2) * W, x1 = (win.ch.hi / D2) * W, gm = win.ch.gamma ?? 1;
      g.fillStyle = 'rgba(0,0,0,.45)';
      g.fillRect(0, 0, x0, H); g.fillRect(x1, 0, W - x1, H);
      g.strokeStyle = '#fff'; g.lineWidth = 1.5;
      g.beginPath();   // the transfer curve
      for (let t = 0; t <= 1.0001; t += 0.05) { const x = x0 + t * (x1 - x0), y = H - Math.pow(t, gm) * (H - 2) - 1; t ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      for (const x of [x0, x1]) { g.fillStyle = '#fff'; g.fillRect(x - 1.5, 0, 3, H); }
    }
  };
  if (win) {
    let drag = null;
    cv.onpointerdown = (e) => {
      const r = cv.getBoundingClientRect(), D = domain(), v = ((e.clientX - r.left) / r.width) * D;
      drag = e.shiftKey ? { k: 'both', v, lo: win.ch.lo, hi: win.ch.hi } : { k: Math.abs(v - win.ch.lo) < Math.abs(v - win.ch.hi) ? 'lo' : 'hi' };
      cv.setPointerCapture(e.pointerId);
      cv.onpointermove(e);
    };
    cv.onpointermove = (e) => {
      if (!drag) return;
      const r = cv.getBoundingClientRect(), D = domain(), v = Math.max(0, Math.min(D, ((e.clientX - r.left) / r.width) * D));
      if (drag.k === 'lo') win.onWin(Math.min(v, win.ch.hi - 1), win.ch.hi);
      else if (drag.k === 'hi') win.onWin(win.ch.lo, Math.max(v, win.ch.lo + 1));
      else win.onWin(drag.lo + v - drag.v, drag.hi + v - drag.v);
    };
    cv.onpointerup = () => { drag = null; };
  }
  cv._redraw = redraw;
  requestAnimationFrame(redraw);
  return cv;
}
function toHex(c) {
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return '#' + c.slice(0, 3).map((v) => Math.round(Math.max(0, Math.min(1, v > 1 ? v / 255 : v)) * 255).toString(16).padStart(2, '0')).join('');
  return '#ffffff';
}

// where a layer came from → its badge and its group in the "By source" view
const ORIGINS = {
  drawn: ['Drawn', 'manual'], sam: ['SAM', 'manual'], 'drawn+sam': ['Drawn + SAM', 'manual'],
  found: ['Found', 'found'], detected: ['Detected', 'detected'], analysis: ['Analysis', 'analysis'], imported: ['Imported', 'imported'],
};
const SOURCE_GROUPS = [['manual', 'Annotations (drawn & SAM)'], ['found', 'Found from examples'], ['detected', 'Detected by algorithms'],
  ['analysis', 'Analysis results'], ['imported', 'Imported files & tables']];
const originOf = (L) => (ORIGINS[L.meta.style?.origin] ? L.meta.style.origin : 'imported');
const groupOf = (L) => ORIGINS[originOf(L)][1];
const srcBadge = (L) => h('span', { class: `src ${originOf(L)}`, title: `Source: ${ORIGINS[originOf(L)][0]}` }, ORIGINS[originOf(L)][0]);
let layerMode = lsGet('wissi-layer-mode', 'nested');
let layerSrc = lsGet('wissi-layer-src', 'all');

function renderLayers() {
  const p = curPane();
  const list = $('l-list');
  const sid = activeSlide();
  const sl = slides.find((x) => x.id === sid);
  $('l-title').textContent = sl ? `Layers on ${sl.name}` : 'Layers';
  for (const b of $('l-mode').children) b.setAttribute('aria-pressed', String(b.dataset.mode === layerMode));
  const q = $('l-filter').value.trim().toLowerCase();
  const all = layersOf(p);
  // source chips: All + each source present, with counts
  const counts = {};
  for (const L of all) counts[groupOf(L)] = (counts[groupOf(L)] || 0) + 1;
  if (layerSrc !== 'all' && !counts[layerSrc]) layerSrc = 'all';
  const chips = $('l-chips');
  chips.replaceChildren();
  if (all.length) {
    for (const [k, label] of [['all', 'All'], ...SOURCE_GROUPS.filter(([g]) => counts[g]).map(([g, t]) => [g, t.split(' ')[0]])]) {
      const b = h('button', { class: 'fchip', 'aria-pressed': String(layerSrc === k), dataset: { src: k }, title: k === 'all' ? 'Every layer' : SOURCE_GROUPS.find(([g]) => g === k)[1] },
        label, h('span', { class: 'n' }, String(k === 'all' ? all.length : counts[k])));
      b.onclick = () => { layerSrc = k; lsSet('wissi-layer-src', k); renderLayers(); };
      chips.append(b);
    }
  }
  const shown = all.filter((L) => (layerSrc === 'all' || groupOf(L) === layerSrc)
    && (!q || L.name.toLowerCase().includes(q) || (L.meta.classes || []).some((c) => c.name.toLowerCase().includes(q))));
  if (layerMode === 'flat') list.replaceChildren(...shown.map((L) => layerRow(p, sid, L)));
  else if (layerMode === 'source') {
    list.replaceChildren();
    for (const [g, title] of SOURCE_GROUPS) {
      const ls = shown.filter((L) => groupOf(L) === g);
      if (!ls.length) continue;
      const allOn = ls.every((L) => p.layerVisible(L));
      const eye = h('button', { class: 'btn ghost', style: 'font-size: 11px; padding: 1px 7px', title: `${allOn ? 'Hide' : 'Show'} every layer in this group` }, allOn ? 'hide all' : 'show all');
      eye.onclick = () => { for (const L of ls) grid.setLayerVisible(paneSel(), L.name, !allOn); renderLayers(); };
      list.append(h('div', { class: 'lgroup-h', dataset: { group: g } }, h('span', { class: 'lbl' }, title), h('span', { class: 'chip' }, String(ls.length)), h('span', { class: 'grow' }), eye),
        ...ls.map((L) => layerCard(p, sid, L, q)));
    }
  } else list.replaceChildren(...shown.map((L) => layerCard(p, sid, L, q)));
  $('l-empty').hidden = all.length > 0 || !sid;
  $('l-add').disabled = !sid || !CAPS.open;
  $('l-add').hidden = !CAPS.open;
}

/** The flat view: one line per layer. */
function layerRow(p, sid, L) {
  const vis = p.layerVisible(L);
  const eye = h('button', { class: `eye${vis ? ' on' : ''}`, 'aria-label': `${vis ? 'Hide' : 'Show'} ${L.name}`, 'aria-pressed': String(vis), title: `${vis ? 'Hide' : 'Show'} (H)` }, icon(vis ? 'eye' : 'eyeOff', 14));
  eye.onclick = () => toggleLayer(L.name);
  const sw = h('span', { class: 'sw', style: `background: ${(L.meta.classes || [])[0]?.color || '#9aa3ad'}` });
  const more = h('button', { class: 'btn icon ghost', title: 'More', 'aria-label': `More for ${L.name}` }, icon('more', 14));
  more.onclick = (e) => layerMenu(e, sid, L);
  const row = h('div', { class: `lflat lcard${vis ? '' : ' off'}`, dataset: { layer: L.name }, style: 'border-radius: 8px' }, eye, sw, h('span', { class: 'nm', title: L.name }, L.name), srcBadge(L),
    h('span', { class: 'chip' }, (L.meta.count ?? 0).toLocaleString('en-US')), more);
  row.onmouseenter = () => { hoveredLayer = L.name; if (vis) grid.flashLayer(active, L.name, 900); };
  return row;
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
  card.append(h('div', { class: 'lhead' }, eye, sw, nm, srcBadge(L), count, soloB, more));
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

// ------------------------------------------------------------------------------------------------ Annotate tab
// ① examples: outlines drawn or made with SAM (saved to the layer + class above) and objects picked on the slide;
// ② find all like them: similar objects of a layer, segment everything (SAM / classic), or areas (pixel classifier)
const applyDraw = () => { grid.setDrawLayer($('a-layer').value.trim() || 'Annotations'); grid.setDrawClass($('a-class').value.trim() || null, $('a-color').value); };
['a-layer', 'a-class', 'a-color'].forEach((id) => $(id).addEventListener('change', () => { applyDraw(); renderDraw(); }));
$('a-sam').onclick = () => { setPicking(false); setTool(grid.tool === 'sam' ? 'pan' : 'sam'); };
$('a-accept').onclick = () => grid.samAccept();
$('a-discard').onclick = () => grid.samClear();
let samChecked = false;
async function samStatus() {
  if (samChecked) return;
  samChecked = true;
  try {
    const st = await (await fetch(`${API}/sam/status`)).json();
    $('a-sam-status').textContent = st.loaded ? `SAM: ${st.backend} on ${st.device}`
      : `SAM backends: ${st.available.join(', ')}${st.available.length === 1 ? ' (no model installed: the classic colour fill; pip install "wissi-viz[sam]" for SAM 2.1)' : ' (loads on first click)'}`;
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
  setTimeout(() => { scheduleInspector(); renderDraw(); }, 400);
});

// ---- picking objects as examples
let picking = false;
const picks = [];   // {slide, layer, index, neg, cls, color}
let lastMods = { shift: false };
function setPicking(on) {
  picking = on;
  $('a-pick').classList.toggle('on', on);
  $('a-pick').setAttribute('aria-pressed', String(on));
  document.body.classList.toggle('picking', on);
  if (on) { setTool('pan'); log('Pick objects: click = like this, shift-click = not like this · P or Esc to stop'); }
}
$('a-pick').onclick = () => setPicking(!picking);
function addPick(d) {
  const sid = slides.find((x) => x.id === d.slide) ? d.slide : panes[d.pane];
  const i = picks.findIndex((x) => x.slide === sid && x.layer === d.layer && x.index === d.index);
  if (i >= 0) picks.splice(i, 1);
  else {
    const L = grid.panes[d.pane]?.slide?.layers.get(d.layer);
    const color = d.color || (L?.meta.classes || []).find((c) => c.name === d.class)?.color;
    picks.push({ slide: sid, layer: d.layer, index: d.index, neg: lastMods.shift, cls: d.class, color });
    grid.flashLayer(d.pane, d.layer, 300);
  }
  renderPicks();
  renderDraw();
}
function renderPicks() {
  const box = $('ex-picks');
  const mine = picks.filter((x) => x.slide === activeSlide());
  box.replaceChildren(...mine.map((x) => {
    const rm = h('button', { 'aria-label': 'Remove this example', title: 'Remove' }, '✕');
    rm.onclick = () => { picks.splice(picks.indexOf(x), 1); renderPicks(); renderDraw(); };
    return h('span', { class: `exc${x.neg ? ' neg' : ''}`, title: `${x.layer} #${x.index}` },
      h('span', { class: 'sw', style: `background: ${x.color || '#9aa3ad'}; width: 8px; height: 8px` }), `${x.neg ? 'not ' : ''}${x.cls || 'object'} #${x.index}`, rm);
  }));
  if (mine.length) {
    const clr = h('button', { class: 'btn ghost', style: 'font-size: 11px; padding: 1px 7px' }, 'clear');
    clr.onclick = () => { for (const x of mine) picks.splice(picks.indexOf(x), 1); renderPicks(); renderDraw(); };
    box.append(clr);
  }
}

// ---- find all
let findMode = 'objects', findCaps = null, findJob = null;
for (const b of $('f-modes').children) b.onclick = () => { findMode = b.dataset.mode; renderDraw(); };
$('f-strict').oninput = () => { $('f-strict-o').textContent = (+$('f-strict').value).toFixed(2); };
$('f-size').oninput = () => { $('f-size-o').textContent = `${$('f-size').value} µm`; };
async function ensureFindCaps() {
  if (findCaps) return findCaps;
  try { findCaps = await api('/find/capabilities'); } catch (e) { findCaps = { error: e.message }; }
  return findCaps;
}
function exampleCount(sid, cb) {   // shapes of the draw layer in the draw class (from the layer's summary)
  const p = curPane();
  const L = p?.slide?.layers.get($('a-layer').value.trim() || 'Annotations');
  if (!L) return cb({ n: 0, classes: [] });
  getSummary(sid, L).then((sm) => cb({ n: (sm.classes || []).find((c) => c.name === $('a-class').value.trim())?.n || 0, classes: sm.classes || [] }))
    .catch(() => cb({ n: 0, classes: [] }));
}
async function renderDraw() {
  refreshDrawLists();
  samStatus();
  renderPicks();
  const p = curPane(), sid = activeSlide();
  const caps = await ensureFindCaps();
  for (const b of $('f-modes').children) {
    b.setAttribute('aria-pressed', String(b.dataset.mode === findMode));
    b.disabled = !!caps.error || (b.dataset.mode === 'areas' && caps.areas === false);
    if (b.dataset.mode === 'areas' && caps.areas === false) b.title = 'Needs scikit-learn: pip install "wissi-viz[analysis]"';
  }
  // what to search: object layers (not the examples' own layer), biggest detected first
  const drawLayer = $('a-layer').value.trim() || 'Annotations';
  const cands = layersOf(p).filter((L) => L.name !== drawLayer && (L.meta.count ?? 0) >= 3 && groupOf(L) !== 'analysis')
    .sort((a, b) => (groupOf(b) === 'detected') - (groupOf(a) === 'detected') || (b.meta.count ?? 0) - (a.meta.count ?? 0));
  const cs = $('f-cand'), keepC = cs.value;
  cs.replaceChildren(...cands.map((L) => h('option', { value: L.name }, `${L.name} · ${(L.meta.count ?? 0).toLocaleString('en-US')}`)));
  if (cands.some((L) => L.name === keepC)) cs.value = keepC;
  const ws = $('f-where'), keepW = ws.value;
  const regions = layersOf(p).filter((L) => ['manual', 'imported', 'detected'].includes(groupOf(L)) && (L.meta.count ?? 0) < 500 && L.name !== drawLayer);
  ws.replaceChildren(h('option', { value: 'view' }, 'this view'), h('option', { value: 'slide' }, 'the whole slide'),
    ...regions.map((L) => h('option', { value: `region:${L.name}` }, `inside ${L.name}`)));
  ws.value = [...ws.options].some((o) => o.value === keepW) ? keepW : 'view';
  $('f-cand-row').hidden = findMode !== 'objects';
  $('f-size-row').hidden = findMode !== 'segment';
  $('f-detail-row').hidden = findMode !== 'areas';
  $('f-strict-row').hidden = findMode === 'areas';
  const mine = picks.filter((x) => x.slide === sid);
  const hint = $('f-hint'), run = $('f-run');
  exampleCount(sid, ({ n, classes }) => {
    const pos = n + mine.filter((x) => !x.neg).length;
    $('ex-count').textContent = `${n} outline${n === 1 ? '' : 's'} · ${mine.length} picked`;
    let msg = '', ok = !!sid && !caps.error && !findJob;
    if (caps.error) msg = caps.error;
    else if (findMode === 'objects') {
      if (!cands.length) { msg = 'No object layer to search: add a cell table, or use “Segment everything”.'; ok = false; }
      else if (!pos) { msg = `Mark at least one example of “${$('a-class').value || 'a class'}” (SAM, draw, or pick objects).`; ok = false; }
      else msg = `${pos} example${pos === 1 ? '' : 's'} → every similar object of ${cs.value}.`;
    } else if (findMode === 'segment') {
      msg = pos ? `${caps.sam || 'SAM'} outlines every object, keeping those like your ${pos} example${pos === 1 ? '' : 's'}.`
        : `${caps.sam || 'SAM'} outlines every object about ${$('f-size').value} µm across (no examples: keeps all). The whole slide can take long without a GPU.`;
    } else {
      const k = classes.filter((c) => c.n > 0);
      msg = k.length >= 2 ? `Train on ${k.map((c) => `${c.name} (${c.n})`).join(', ')} in ${drawLayer} → map those areas.`
        : `Draw shapes of at least two classes in ${drawLayer} (e.g. Tumor and Stroma: change the class, draw, repeat).`;
      ok = ok && k.length >= 2;
    }
    hint.textContent = msg;
    run.disabled = !ok;
    run.textContent = findMode === 'areas' ? 'Find areas' : findMode === 'segment' ? 'Segment everything' : 'Find all like these';
  });
}

function findScope() {
  const v = $('f-where').value, p = curPane();
  if (v === 'view' && p?.view) {
    const hw = p.rect.w / 2 / p.view.scale, hh = p.rect.h / 2 / p.view.scale;
    return { kind: 'view', rect: [p.view.cx - hw, p.view.cy - hh, p.view.cx + hw, p.view.cy + hh] };
  }
  if (v.startsWith('region:')) return { kind: 'region', layer: v.slice(7) };
  return { kind: 'slide' };
}
$('f-run').onclick = async () => {
  const sid = activeSlide();
  if (!sid || findJob) return;
  const drawLayer = $('a-layer').value.trim() || 'Annotations', cls = $('a-class').value.trim();
  const mine = picks.filter((x) => x.slide === sid);
  const body = {
    mode: findMode, scope: findScope(), strictness: +$('f-strict').value,
    examples: { layer: drawLayer, classes: cls ? [cls] : null, picks: mine.map(({ layer, index, neg }) => ({ layer, index, neg })), name: cls || null, color: $('a-color').value },
    candidates: $('f-cand').value || null, object_um: +$('f-size').value, training: drawLayer, detail: $('f-detail').value,
  };
  if (!(await layerExists(drawLayer))) body.examples.layer = null;
  try {
    const job = await api(`/slides/${enc(sid)}/find`, body);
    trackJob(job, sid);
  } catch (e) { toast({ kind: 'bad', title: 'Could not start', sub: e.message, ms: 8000 }); log(`find: ${e.message}`, true); }
};
async function layerExists(name) { return !!curPane()?.slide?.layers.get(name); }
$('f-cancel').onclick = () => { if (findJob) api(`/jobs/${enc(findJob.id)}/cancel`, {}).catch(() => {}); };
function trackJob(job, sid) {
  findJob = job;
  $('f-job').hidden = false;
  $('f-run').disabled = true;
  $('f-run').classList.add('busy');
  let lastRefresh = 0;
  const tick = async () => {
    let j;
    try { j = await api(`/jobs/${enc(job.id)}`); } catch (e) { j = { state: 'error', error: e.message }; }
    $('f-bar').style.width = `${Math.round((j.progress || 0) * 100)}%`;
    $('f-msg').textContent = `${j.message || j.state}${j.seconds != null ? ` · ${j.seconds} s` : ''}`;
    if (j.state === 'running' || j.state === 'queued') {
      if (performance.now() - lastRefresh > 3000) { lastRefresh = performance.now(); grid.refreshAnnotations(sid); }   // partial results
      setTimeout(tick, 400);
      return;
    }
    findJob = null;
    $('f-job').hidden = true;
    $('f-run').classList.remove('busy');
    grid.refreshAnnotations(sid);
    if (j.state === 'done') {
      const r = j.result;
      findResults.unshift({ ...r, slide: sid, title: j.title });
      renderFindResults(true);
      toast({ kind: 'analysis', title: `${j.title}: ${r.count.toLocaleString('en-US')} ${r.mode === 'areas' ? 'areas' : 'found'}`, sub: `${r.layer} · ${r.where}`,
        action: { label: 'Show', fn: () => showFound(r.layer, sid) }, ms: 8000 });
      setTimeout(() => showFound(r.layer, sid, false), 400);
      log(`${j.title}: ${r.count} in ${r.where} (${r.seconds} s)${r.note ? ` · ${r.note}` : ''}`);
    } else if (j.state === 'cancelled') { toast({ title: 'Cancelled' }); log('find: cancelled'); }
    else { toast({ kind: 'bad', title: `${j.title || 'Find'} failed`, sub: j.error, ms: 10000 }); log(`find: ${j.error}`, true); }
    renderDraw();
  };
  tick();
}
function showFound(layer, sid, flash = true) {
  if (activeSlide() !== sid) return;
  grid.setLayerVisible(active, layer, true);
  if (flash) grid.flashLayer(active, layer, 1200);
  scheduleInspector();
}
const findResults = [];
function renderFindResults(fresh = false) {
  const box = $('f-results');
  box.replaceChildren(...findResults.slice(0, 8).map((r, i) => {
    const card = h('div', { class: `result${fresh && i === 0 ? ' new' : ''}`, dataset: { layer: r.layer } });
    const count = h('span', { class: 'chip' }, `${r.count.toLocaleString('en-US')}${r.of && r.mode !== 'areas' ? ` of ${r.of.toLocaleString('en-US')}` : ''}`);
    card.append(h('div', { class: 'row' }, h('b', { class: 'grow', style: 'font-weight: 600' }, r.layer), count));
    if (r.kpis?.length) card.append(h('div', { class: 'kpis' }, ...r.kpis.map(([k, v]) => h('span', { class: 'kpi' }, h('span', { class: 'k' }, k), h('span', { class: 'v' }, v)))));
    if (r.result) {   // live strictness: the server keeps the scores, so this re-filters without measuring again
      const out = h('output', { class: 'mono' }, (+r.strictness).toFixed(2));
      const sl = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: r.strictness, 'aria-label': `Strictness of ${r.layer}` });
      let t = 0;
      sl.oninput = () => {
        out.textContent = (+sl.value).toFixed(2);
        clearTimeout(t);
        t = setTimeout(async () => {
          try {
            const res = await api(`/slides/${enc(r.slide)}/find/threshold`, { result: r.result, strictness: +sl.value });
            r.count = res.count; r.strictness = res.strictness;
            count.textContent = `${res.count.toLocaleString('en-US')} of ${res.of.toLocaleString('en-US')}`;
            grid.refreshAnnotations(r.slide);
          } catch (e) { log(`strictness: ${e.message}`, true); }
        }, 220);
      };
      card.append(h('label', { class: 'field' }, 'Strictness', sl, out));
    }
    if (r.note) card.append(h('div', { class: 'panel-note' }, r.note));
    const show = h('button', { class: 'btn grow' }, 'Show');
    show.onclick = () => showFound(r.layer, r.slide);
    const rm = h('button', { class: 'btn', title: 'Remove this layer from the slide' }, 'Remove');
    rm.onclick = () => api(`/slides/${enc(r.slide)}/annotations?layer=${enc(r.layer)}`, null, 'DELETE')
      .then(() => { findResults.splice(findResults.indexOf(r), 1); grid.refreshAnnotations(r.slide); renderFindResults(); toast({ title: `Removed ${r.layer}` }); })
      .catch((e) => log(`remove: ${e.message}`, true));
    card.append(h('div', { class: 'row' }, show, rm));
    return card;
  }));
}

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
grid.addEventListener('pointerdown', (e) => { lastClick = { x: e.clientX, y: e.clientY }; lastMods = { shift: e.shiftKey }; closeRadial(); closeMenu(); }, true);
grid.addEventListener('paneclick', () => hideCard());
grid.addEventListener('objectclick', async (e) => {
  const d = e.detail;
  if (picking) { addPick(d); return; }
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
  if (e.key === 'Escape') { hideCard(); closeRadial(); closeMenu(); if (picking) setPicking(false); if ($('picker').classList.contains('on') && !typing(e.target)) closePicker(); return; }
  if (mod || e.altKey || typing(e.target) || document.querySelector('dialog[open]')) return;
  const k = e.key;
  if (k >= '1' && k <= '8' && !e.repeat) { const [t, label] = TOOLS[+k - 1]; setTool(t); keyFlash(k, label); return; }
  const tabs = { l: ['layers', 'Layers'], i: ['image', 'Image'], a: ['analyze', 'Analyze'], m: ['measure', 'Measure'], d: ['draw', 'Annotate'] };
  const lk = k.toLowerCase();
  if (tabs[lk] && !e.shiftKey) { setTab(tabs[lk][0]); keyFlash(k.toUpperCase(), tabs[lk][1]); return; }
  if (lk === 's' && hoveredLayer) { toggleSolo(hoveredLayer); keyFlash('S', `solo ${hoveredLayer}`); return; }
  if (lk === 'h' && hoveredLayer) { toggleLayer(hoveredLayer); keyFlash('H', `${hoveredLayer}`); return; }
  if (lk === 'f') { grid.fit('all'); keyFlash('F', 'fit'); return; }
  if (lk === 'p') { setPicking(!picking); if (picking) setTab('draw'); keyFlash('P', picking ? 'pick objects' : 'picking off'); return; }
  if (lk === 'o') { toggleMap(); keyFlash('O', 'overview'); return; }
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
  for (const b of $('l-mode').children) b.onclick = () => { layerMode = b.dataset.mode; lsSet('wissi-layer-mode', layerMode); renderLayers(); };
  $('s-filter').oninput = () => renderList();
  $('s-newproj').onclick = () => newProject([]);
  // file picker
  $('pk-up').onclick = () => { if (pk.parent != null) browse(pk.parent); };
  $('pk-filter').oninput = () => renderEntries();
  $('pk-filter').onkeydown = (e) => {
    if (e.key !== 'Enter') return;
    const v = $('pk-filter').value.trim();
    if (/^(\/|~|[a-z]:[\\/])/i.test(v)) { browse(v); return; }   // a typed path: go there
    const first = $('browser').querySelector('a');
    if (first) first.dataset.kind === 'folder' ? first.click() : first.dispatchEvent(new MouseEvent('dblclick'));
  };
  $('pk-close').onclick = closePicker;
  $('pk-folder').onclick = () => openPaths([pk.dir]);
  $('pk-open').onclick = () => openPaths([...pk.sel]);
  // image tab
  $('i-scope').onchange = () => renderImage();
  $('i-log').onchange = () => document.querySelectorAll('#disp-body canvas').forEach((c) => c._redraw?.());
  $('i-link').onclick = () => { linkCh = !linkCh; $('i-link').setAttribute('aria-pressed', String(linkCh)); $('i-link').classList.toggle('on', linkCh); };
  $('i-auto').onclick = async () => {
    const p = curPane(), sid = p?.slide?.info?.id;
    if (!sid) return;
    const hist = await getHist(sid, histRect(p)).catch(() => null);
    if (!hist) return;
    if (p.slide.info.mode === 'channels') hist.channels.forEach((c, i) => grid.setChannel(panesOf(sid), i, { min: Math.round(c.p['0.1']), max: Math.max(Math.round(c.p['0.1']) + 1, Math.round(c.p['99.9'])) }));
    else autoLevels(sid, hist);
    renderImage();
    toast({ title: 'Auto contrast', sub: $('i-scope').selectedOptions[0].text, ms: 2000 });
  };
  $('i-reset').onclick = () => {
    const p = curPane(), info = p?.slide?.info;
    if (!info) return;
    const sid = info.id;
    if (info.mode === 'channels') (info.channels || []).forEach((c, i) => grid.setChannel(panesOf(sid), i, { visible: true, gamma: 1,
      min: c.window ? c.window[0] : 0, max: c.window ? c.window[1] : (info.dtype === 'uint16' ? 65535 : 255) }));
    else { grid.setAdjust(panesOf(sid), { brightness: 0, contrast: 1, gamma: 1, invert: false }); grid.setStain(panesOf(sid), 'off'); }
    renderImage();
  };
  $('i-scalebar').onchange = () => setScaleBar($('i-scalebar').checked);
  $('i-fps').onchange = () => setPerf($('i-fps').checked);
  $('fps').onclick = () => setPerf(!perfOn);
  // view tools on the slide
  for (const b of $('vtools').querySelectorAll('button')) b.onclick = () => viewTool(b.dataset.v, b);
  $('minimap').onpointerdown = (e) => { mapNav(e); $('minimap').setPointerCapture(e.pointerId); $('minimap').onpointermove = mapNav; };
  $('minimap').onpointerup = () => { $('minimap').onpointermove = null; };
  grid.addEventListener('views', () => { drawMap(); showMag(); });
  $('ms-layer').onchange = () => renderMeasure();
  $('an-objects').onchange = () => renderAnalyze();
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
  grid.addEventListener('activepane', (e) => { active = e.detail.pane; renderList(); renderCrumb(); hideCard(); scheduleInspector(); loadMap(); });
  grid.addEventListener('slideload', () => { renderCrumb(); scheduleInspector(); loadMap(); });
  grid.addEventListener('layers', () => scheduleInspector());
  // drag-over highlight (the drop itself is read natively, with full paths)
  let dragDepth = 0;
  const fromOutside = (e) => !e.dataTransfer?.types?.includes('text/wissi-slide');   // not a slide dragged in the list
  document.addEventListener('dragenter', (e) => { if (!fromOutside(e)) return; e.preventDefault(); dragDepth++; document.body.classList.add('drag'); });
  document.addEventListener('dragend', () => { dragDepth = 0; document.body.classList.remove('drag'); });
  document.addEventListener('dragleave', (e) => { if (!fromOutside(e)) return; if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('drag'); } });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault(); dragDepth = 0; document.body.classList.remove('drag');
    if (!fromOutside(e)) return;
    if (!native() && CAPS.open) showPathBox('drop');         // a browser can't see a dropped file's path
  });
  applyDraw();
  setScaleBar(lsGet('wissi-scalebar', '1') !== '0');
  setPerf(perfOn);
  toggleMap(mapOn);
  grid.hoverEvents = true;
  grid.contextEvents = true;
}

/** Show these slide ids in panes (1–32), as if picked in the slide list. */
function showSlides(ids) { panes = ids.filter((id) => slides.some((x) => x.id === id)).slice(0, MAX_PANES); active = 0; apply(); }

// ------------------------------------------------------------------------------------------------ view tools on the slide
function viewTool(v, b) {
  const p = curPane();
  const view = p?.view;
  if (v === 'in' || v === 'out') { if (view) grid.goTo(active, view.cx, view.cy, { zoom: view.scale * (v === 'in' ? 2 : 0.5) }); }
  else if (v === 'fit') grid.fit(panes.length > 1 && $('sync').value !== 'none' ? 'all' : active);
  else if (v === 'one') { if (view) grid.goTo(active, view.cx, view.cy, { zoom: 1 / (window.devicePixelRatio || 1) }); }
  else if (v === 'mag') {
    if (!view) return;
    if (!p.slide?.info?.mpp) { toast({ kind: 'bad', title: 'No µm per pixel', sub: 'this slide has no pixel size, so magnifications are unknown' }); return; }
    grid.goTo(active, view.cx, view.cy, { magnification: +b.dataset.m });
  } else if (v === 'map') toggleMap();
  else if (v === 'scale') setScaleBar(b.getAttribute('aria-pressed') !== 'true');
  else if (v === 'snap') snapshot();
  else if (v === 'full') {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else ($('main').requestFullscreen || $('main').webkitRequestFullscreen)?.call($('main'));
  }
}
function showMag() {
  const p = curPane(), mpp = p?.slide?.info?.mpp;
  const el = $('v-mag');
  if (!p?.view || !mpp) { el.textContent = '–'; return; }
  const mag = 10 / (mpp[0] / p.view.scale);   // 10 µm per screen px ≈ 1×
  el.textContent = mag >= 10 ? `${Math.round(mag)}×` : `${mag.toFixed(1)}×`;
  for (const b of $('vtools').querySelectorAll('[data-v=mag]')) b.classList.toggle('on', Math.abs(Math.log(mag / +b.dataset.m)) < 0.06);
}
function setScaleBar(on) {
  grid.setAttribute('scalebar', on ? 'true' : 'false');
  $('vtools').querySelector('[data-v=scale]').setAttribute('aria-pressed', String(on));
  $('i-scalebar').checked = on;
  lsSet('wissi-scalebar', on ? '1' : '0');
}
async function snapshot() {
  const p = curPane();
  if (!p?.view) return;
  const blob = await grid.snapshot(active, { scaleBar: grid.getAttribute('scalebar') !== 'false' });
  if (!blob) return;
  const sl = slides.find((x) => x.id === activeSlide());
  const name = `${(sl?.name || 'slide').replace(/\.[^.]+$/, '')} ${Math.round(p.view.cx)},${Math.round(p.view.cy)}.png`.replace(/[\\/:*?"<>|]+/g, '_');
  const a = h('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  toast({ title: 'Snapshot saved', sub: name, ms: 3000 });
}
// overview map: the slide's thumbnail with the active pane's view on it; click or drag to go there
let mapOn = lsGet('wissi-map', '0') === '1', mapFor = null;
function toggleMap(on = !mapOn) {
  mapOn = on;
  lsSet('wissi-map', on ? '1' : '0');
  $('vtools').querySelector('[data-v=map]').setAttribute('aria-pressed', String(on));
  $('minimap').hidden = !on;
  if (on) loadMap();
}
function loadMap() {
  if (!mapOn) return;
  const sid = curPane()?.slide?.info?.id;
  if (!sid || mapFor === sid) { drawMap(); return; }
  mapFor = sid;
  const img = $('minimap').querySelector('img');
  img.onload = () => drawMap();
  img.src = `${API}/slides/${enc(sid)}/thumbnail?size=440`;
}
function drawMap() {
  if (!mapOn) return;
  const p = curPane(), info = p?.slide?.info;
  const img = $('minimap').querySelector('img'), cv = $('minimap').querySelector('canvas');
  if (!info || !p.view || !img.naturalWidth) return;
  if (mapFor !== info.id) { loadMap(); return; }
  const w = img.clientWidth, hh = img.clientHeight;
  cv.width = w; cv.height = hh; cv.style.width = `${w}px`; cv.style.height = `${hh}px`;
  const g = cv.getContext('2d'), k = w / info.width;
  const vw = p.rect.w / p.view.scale, vh = p.rect.h / p.view.scale;
  const x = (p.view.cx - vw / 2) * k, y = (p.view.cy - vh / 2) * k;
  g.clearRect(0, 0, w, hh);
  g.fillStyle = 'rgba(0,0,0,.35)'; g.fillRect(0, 0, w, hh);
  g.clearRect(x, y, vw * k, vh * k);
  g.strokeStyle = '#5ed0ff'; g.lineWidth = 2; g.strokeRect(x, y, Math.max(3, vw * k), Math.max(3, vh * k));
}
function mapNav(e) {
  const p = curPane(), info = p?.slide?.info, img = $('minimap').querySelector('img');
  if (!info || !p.view) return;
  const r = img.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  grid.goTo(active, Math.max(0, Math.min(1, fx)) * info.width, Math.max(0, Math.min(1, fy)) * info.height, { animate: false });
}
// FPS: frames the viewer drew in the last half second (idle when nothing moves: it only draws on change)
let perfOn = lsGet('wissi-perf', '0') === '1', lastFrames = 0, lastFpsAt = performance.now();
function setPerf(on) { perfOn = on; lsSet('wissi-perf', on ? '1' : '0'); $('perf').style.display = on ? 'block' : 'none'; $('i-fps').checked = on; }
setInterval(() => {
  const t = performance.now(), n = grid.framesDrawn || 0;
  const fps = Math.round(((n - lastFrames) * 1000) / Math.max(1, t - lastFpsAt));
  lastFrames = n; lastFpsAt = t;
  const el = $('fps');
  el.textContent = fps > 1 ? `${fps} fps` : 'idle';
  el.classList.toggle('hot', fps > 1);
  if (perfOn) {
    let st = {};
    try { st = grid.stats?.() || {}; } catch (_) { /* viewer not ready */ }
    $('perf').textContent = `${fps > 1 ? `${fps} fps` : 'idle'} · ${(grid._frameMs || 0).toFixed(1)} ms/frame · ${st.textures ?? 0} tiles · ${st.textureMB ?? 0} MB GPU`;
  }
}, 500);

window.wissiApp = { onResult, activeSlide, showSlides, setTool, setTab, toast };

async function start() {
  wire();
  await customElements.whenDefined('wissi-viz');
  const st = await call('state');
  slides = st.slides || [];
  projects = st.projects || [];
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
