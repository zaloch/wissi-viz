// Figure composer (the Figures screen of the app page): a figure is a recipe — which saved views (a fixed list or
// rules over their fields), laid out how (rows by a field, or a grid; a journal page, slides, or the layouts of a
// PowerPoint template), labelled and charted how. The server lays the recipe out into pages of windows in mm
// (wissi_viz/figures.py, POST /app/figures/layout); this module draws them, lets each window be moved, resized or
// changed, keeps the recipe saved (POST /app/figures) and exports it (POST /app/figures/export).

let X = null;   // what app.js lends: { h, s, http, quiet, toast, saveBlob, API, BASE, TOKEN, key, enc, ... }
const F = {
  open: false, list: [], cur: null, data: null, sel: null, seq: 0, layoutTimer: 0, saveTimer: 0, dirty: false,
  templates: [], exportOpen: false,
};
const $ = (id) => document.getElementById(id);
const mmPt = 25.4 / 72;

export function initFigures(ctx) {
  X = ctx;
  $('b-figs')?.addEventListener('click', () => (F.open ? closeFigures() : openFigures()));
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('resize', () => { if (F.open && F.data) renderPages(); });
  return { open: openFigures, close: closeFigures, newFrom: newFigureFrom, isOpen: () => F.open };
}

// ------------------------------------------------------------------------------------------------- open / close
async function openFigures() {
  F.open = true;
  $('figs').hidden = false;
  $('b-figs')?.classList.add('on');
  $('b-figs')?.setAttribute('aria-pressed', 'true');
  skeleton();
  await loadList();
  if (!F.cur) {
    if (F.list.length) await pick(F.list[0].id);
    else await newFigure();
  } else relayout(0);
}
function closeFigures() {
  F.open = false;
  flushSave();
  $('figs').hidden = true;
  $('b-figs')?.classList.remove('on');
  $('b-figs')?.setAttribute('aria-pressed', 'false');
  F.exportOpen = false;
}
/** A new figure of these saved views (ids), opened in the composer. */
async function newFigureFrom(ids) {
  if (!F.open) { F.open = true; $('figs').hidden = false; $('b-figs')?.classList.add('on'); skeleton(); await loadList(); }
  await newFigure(ids);
}

async function loadList() {
  const r = await X.quiet(X.http(`${X.BASE}/app/figures`));
  F.list = r.figures || [];
  const t = await X.quiet(X.http(`${X.BASE}/app/figures/templates`));
  F.templates = t.templates || [];
}
async function pick(id) {
  flushSave();
  const f = F.list.find((x) => x.id === id);
  if (!f) return;
  F.cur = { id: f.id, name: f.name, recipe: f.recipe || {} };
  F.sel = null;
  F.data = null;
  render();
  relayout(0);
}
async function newFigure(ids = null) {
  flushSave();
  const n = F.list.length + 1;
  let name = `Figure ${n}`;
  while (F.list.some((f) => f.name === name)) name = `${name}′`;
  const recipe = { template: 'pub', source: { kind: 'views', ids: ids || X.checkedViewIds() || [] }, labels: '{Name}' };
  const r = await X.quiet(X.http(`${X.BASE}/app/figures`, { name, recipe }));
  if (!r.figure) { X.toast({ kind: 'bad', title: 'Figure not made', sub: (r.errors || [])[0] || '' }); return; }
  F.list = r.figures;
  F.cur = { id: r.figure.id, name, recipe };
  F.sel = null;
  F.data = null;
  render();
  relayout(0);
}
async function openSnapshot(file) {
  const headers = { 'X-Wissi-Token': X.TOKEN };
  if (X.key()) headers['X-Wissi-Key'] = X.key();
  const r = await fetch(`${X.BASE}/app/figures/snapshot`, { method: 'POST', headers, body: file });
  let j = {};
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) { X.toast({ kind: 'bad', title: 'Snapshot not opened', sub: j.detail || r.statusText }); return; }
  F.list = j.figures;
  X.toast({ title: `Opened: ${j.figure.name}`, sub: 'Views as exported · Use current views: Source (left)' });
  await pick(j.figure.id);
}
async function relink(old) {
  const nw = window.prompt(`Slide not found:\n${old}\n\nIts new path (its views follow):`, old);
  if (!nw || nw === old) return;
  try {
    const r = await X.http(`${X.BASE}/app/views-relink`, { old, new: nw });
    X.toast({ title: `${r.relinked} view${r.relinked === 1 ? '' : 's'} relinked`, sub: nw });
    X.loadViews();
    relayout(0);
  } catch (e) { X.toast({ kind: 'bad', title: 'Not relinked', sub: e.message }); }
}
async function deleteFigure() {
  if (!F.cur) return;
  if (!window.confirm(`Delete “${F.cur.name}”? Its views stay.`)) return;
  clearTimeout(F.saveTimer); F.dirty = false;
  const r = await X.quiet(X.http(`${X.BASE}/app/figures`, { delete: F.cur.id }));
  F.list = r.figures || [];
  F.cur = null;
  if (F.list.length) await pick(F.list[0].id); else await newFigure();
}

// ---------------------------------------------------------------------------------------------- recipe changes
const R = () => F.cur.recipe;
function change(fn, { layout = true } = {}) {
  if (!F.cur) return;
  fn(F.cur.recipe);
  F.dirty = true;
  clearTimeout(F.saveTimer);
  F.saveTimer = setTimeout(flushSave, 700);
  if (layout) relayout(); else render();
}
function flushSave() {
  clearTimeout(F.saveTimer);
  if (!F.cur || !F.dirty) return;
  F.dirty = false;
  const { id, name, recipe } = F.cur;
  X.http(`${X.BASE}/app/figures`, { id, name, recipe }).then((r) => { F.list = r.figures || F.list; renderBar(); }).catch((e) => {
    X.toast({ kind: 'bad', title: 'Figure not saved', sub: e.message });
  });
}
function relayout(delay = 120) {
  clearTimeout(F.layoutTimer);
  F.layoutTimer = setTimeout(doLayout, delay);
}
async function doLayout() {
  if (!F.cur) return;
  const seq = ++F.seq;
  const cur = F.cur;
  let d;
  try {
    d = await X.http(`${X.BASE}/app/figures/layout`, { recipe: cur.recipe, name: cur.name });
  } catch (e) {
    if (seq === F.seq) { $('fg-pages').replaceChildren(X.h('div', { class: 'panel-note warn' }, `Layout failed: ${e.message}`)); }
    return;
  }
  if (seq !== F.seq || cur !== F.cur) return;   // a newer layout was asked for meanwhile
  F.data = d;
  if (d.opened && d.slides) X.onResult({ slides: d.slides });
  const r = cur.recipe;
  if (r.source?.kind === 'query' && r.members == null) { r.members = d.members; F.dirty = true; flushSave(); }
  if (F.sel && !allWindows().some((w) => w.key === F.sel)) F.sel = null;
  render();
}
function update() {   // a query's members become every view that matches now
  change((r) => { r.members = null; });
}

// ------------------------------------------------------------------------------------------------------ render
function skeleton() {
  const h = X.h;
  if ($('fg-bar')) return;
  $('figs').replaceChildren(
    h('div', { id: 'fg-bar', class: 'row' }),
    h('div', { id: 'fg-body' },
      h('aside', { id: 'fg-src', 'aria-label': 'Source' }),
      h('div', { id: 'fg-mid' }, h('div', { id: 'fg-tools', class: 'row', role: 'toolbar', 'aria-label': 'Layout' }),
        h('div', { id: 'fg-layouts', role: 'group', 'aria-label': 'Layouts of the template', hidden: true }),
        h('div', { id: 'fg-canvas' }, h('div', { id: 'fg-pages' }))),
      h('aside', { id: 'fg-insp', 'aria-label': 'Figure settings' })),
    h('div', { id: 'fg-export', class: 'hud', role: 'dialog', 'aria-label': 'Export', hidden: true }));
  $('fg-canvas').addEventListener('pointerdown', (e) => {
    if (e.target === $('fg-canvas') || e.target === $('fg-pages')) { F.sel = null; renderPages(); renderInsp(); }
  });
}
function render() {
  if (!F.open) return;
  skeleton();
  renderBar(); renderSrc(); renderTools(); renderLayouts(); renderPages(); renderInsp(); renderExport();
}
const seg = (label, items, cur, onpick, extra = {}) => {
  const h = X.h;
  return h('span', { class: 'seg', role: 'group', 'aria-label': label, ...extra },
    items.map(([v, t, title]) => {
      const b = h('button', { 'aria-pressed': String(v === cur), title: title || t, dataset: { v: String(v) } }, t);
      b.onclick = () => onpick(v);
      return b;
    }));
};
function select(label, options, cur, onchange, extra = {}) {
  const h = X.h;
  const sel = h('select', { 'aria-label': label, ...extra },
    options.map(([v, t]) => h('option', { value: String(v), selected: String(v) === String(cur) }, t)));
  sel.onchange = () => onchange(sel.value);
  return sel;
}

function renderBar() {
  const h = X.h;
  const bar = $('fg-bar');
  if (!F.cur) { bar.replaceChildren(); return; }
  const figSel = select('Figure', F.list.map((f) => [f.id, f.name]), F.cur.id, (v) => pick(Number(v)), { id: 'fg-figure' });
  const name = h('input', { id: 'fg-name', value: F.cur.name, 'aria-label': 'Figure name', style: 'width: 190px' });
  name.onchange = () => {
    const v = name.value.trim();
    if (v && v !== F.cur.name) { F.cur.name = v; F.dirty = true; flushSave(); relayout(); }
  };
  const nb = h('button', { id: 'fg-new', class: 'btn', title: 'New figure: views checked in the Views list, else none' }, '+ New');
  nb.onclick = () => newFigure();
  const del = h('button', { id: 'fg-delete', class: 'btn icon ghost', title: 'Delete figure (its views stay)', 'aria-label': 'Delete figure' });
  del.append(X.s('svg', { width: 15, height: 15, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8, 'aria-hidden': 'true' },
    X.s('path', { d: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13' })));
  del.onclick = deleteFigure;
  const tpl = seg('Template', [['pub', 'Publication', 'Journal page: 85 or 180 mm wide'], ['ppt', 'Presentation', 'Slides: 16:9 or 4:3'],
    ['pptx', 'From PPTX', 'Slides on a PowerPoint template: its master, layouts and arrangements']], R().template || 'pub',
  (v) => change((r) => { r.template = v; r.page = v === 'pub' ? 'double' : '16:9'; if (v === 'pptx' && !r.pptx && F.templates[0]) r.pptx = { id: F.templates[0].id, layout: null }; }),
  { id: 'fg-template' });
  const d = F.data;
  const pend = d ? d.pending.add.length + d.pending.remove.length : 0;
  const status = h('span', { id: 'fg-status', class: 'chip' + (pend ? ' warn' : '') });
  if (!d) status.textContent = 'Laying out…';
  else if (R().source?.kind !== 'query') status.textContent = `Fixed list · ${d.members.length} view${d.members.length === 1 ? '' : 's'}`;
  else if (pend) {
    status.append(`${d.pending.add.length ? `${d.pending.add.length} new view${d.pending.add.length === 1 ? '' : 's'} match` : ''}${d.pending.add.length && d.pending.remove.length ? ' · ' : ''}${d.pending.remove.length ? `${d.pending.remove.length} no longer match` : ''}`);
    const up = h('button', { id: 'fg-update', class: 'btn sm primary', title: 'Make the figure every view that matches now' }, 'Update');
    up.onclick = update;
    status.append(up);
  } else status.textContent = `Up to date · ${d.members.length} view${d.members.length === 1 ? '' : 's'}`;
  const ex = h('button', { id: 'fg-export-open', class: 'btn primary', title: 'Export: PNG, TIFF, PDF, SVG or PowerPoint' }, 'Export…');
  ex.onclick = () => { F.exportOpen = !F.exportOpen; renderExport(); };
  const snapIn = h('input', { type: 'file', accept: '.json', id: 'fg-snap-file', hidden: true });
  snapIn.onchange = () => { if (snapIn.files[0]) openSnapshot(snapIn.files[0]); snapIn.value = ''; };
  const snapB = h('button', { id: 'fg-snap-open', class: 'btn', title: 'Open a snapshot (.recipe.json, saved beside an export): the figure as it was exported' }, 'Open snapshot…');
  snapB.onclick = () => snapIn.click();
  const close = h('button', { id: 'fg-close', class: 'btn icon ghost', title: 'Close figures · Back to the slides', 'aria-label': 'Close figures' }, '✕');
  close.onclick = closeFigures;
  bar.replaceChildren(h('b', {}, 'Figure'), figSel, name, nb, del, snapB, snapIn, h('span', { class: 'sep' }), tpl, h('span', { class: 'grow' }), status, ex, close);
}

// ---------------------------------------------------------------------------------------------------- source
function renderSrc() {
  const h = X.h;
  const box = $('fg-src');
  const r = F.cur ? R() : null;
  if (!r) { box.replaceChildren(); return; }
  const src = r.source || { kind: 'views', ids: [] };
  const d = F.data;
  const views = d?.views || [];
  const kids = [h('div', { class: 'lbl' }, 'Source'),
    seg('Source', [['views', 'Views', 'A fixed list: tick views'], ['query', 'Query', 'Rules: every view that matches']], src.kind,
      (v) => change((rr) => {
        rr.source = { ...(rr.source || {}), kind: v };
        if (v === 'views' && !(rr.source.ids || []).length && d) rr.source.ids = [...d.members];
        rr.members = null;
      }), { id: 'fg-srckind', style: 'display: flex' })];
  const thumb = (v) => {
    const img = h('img', { class: 'thumb', alt: '', loading: 'lazy' });
    if (v.sid) img.src = `${X.API}/slides/${X.enc(v.sid)}/thumbnail?size=96&bounds=${v.bounds.map((x) => Math.round(x)).join(',')}`;
    else img.classList.add('blank');
    img.onerror = () => { img.removeAttribute('src'); img.classList.add('blank'); };
    return img;
  };
  if (src.kind === 'frozen') {
    kids.splice(1, 1);   // no Views / Query switch: a snapshot's views are fixed
    const cur = h('button', { id: 'fg-unfreeze', class: 'btn sm', title: 'Draw with the saved views as they are now (frames, fields) instead' }, 'Use current views');
    cur.onclick = () => change((rr) => { rr.source = { kind: 'views', ids: (rr.source.views || []).map((v) => v.id) }; });
    kids.push(h('div', { class: 'panel-note' }, `Snapshot: ${views.length} view${views.length === 1 ? '' : 's'} as exported · Frames and fields fixed`), h('div', { class: 'row' }, cur));
    for (const v of views) kids.push(h('div', { class: 'fv', title: v.slide }, thumb(v), h('span', { class: 'txt' }, h('span', { class: 'name' }, v.name || `View ${v.id}`), h('span', { class: 'sub' }, v.meta?.Slide || ''))));
  } else if (src.kind !== 'query') {
    const ids = new Set(src.ids || []);
    const fromList = h('button', { id: 'fg-fromlist', class: 'btn sm', title: 'Use the views checked in the Views list' }, 'Checked in list');
    fromList.onclick = () => {
      const c = X.checkedViewIds();
      if (!c.length) { X.toast({ title: 'No views checked', sub: 'Views list · Tick views · Then Checked in list' }); return; }
      change((rr) => { rr.source = { kind: 'views', ids: c }; });
    };
    const all = h('button', { id: 'fg-all', class: 'btn sm ghost', title: 'Tick every view' }, 'All');
    all.onclick = () => change((rr) => { rr.source = { kind: 'views', ids: views.map((v) => v.id) }; });
    const none = h('button', { id: 'fg-none', class: 'btn sm ghost', title: 'Untick every view' }, 'None');
    none.onclick = () => change((rr) => { rr.source = { kind: 'views', ids: [] }; });
    kids.push(h('div', { class: 'panel-note' }, 'Tick views · Order: Order in the toolbar'), h('div', { class: 'row' }, fromList, all, none));
    const ul = h('div', { id: 'fg-views', role: 'list' });
    for (const v of views) {
      const cb = h('input', { type: 'checkbox', checked: ids.has(v.id), 'aria-label': `Use ${v.name || 'view ' + v.id}` });
      cb.onchange = () => change((rr) => {
        const cur = new Set(rr.source.ids || []);
        if (cb.checked) cur.add(v.id); else cur.delete(v.id);
        rr.source.ids = [...(rr.source.ids || []).filter((i) => cur.has(i)), ...[...cur].filter((i) => !(rr.source.ids || []).includes(i))];
      });
      ul.append(h('label', { class: 'fv', role: 'listitem', title: v.slide }, cb, thumb(v),
        h('span', { class: 'txt' }, h('span', { class: 'name' }, v.name || `View ${v.id}`), h('span', { class: 'sub' }, v.meta.Slide || ''))));
    }
    if (!views.length) ul.append(h('div', { class: 'panel-note' }, 'No saved views · Viewer: frame a region · V saves it'));
    kids.push(ul);
  } else {
    const rules = src.rules || [];
    kids.push(h('div', { class: 'panel-note' }, 'Set rules · Every view that matches joins on Update'));
    rules.forEach((q, i) => kids.push(ruleRow(q, i)));
    const add = h('button', { id: 'fg-addrule', class: 'btn sm', title: 'Add a rule on a field' }, '+ Rule');
    add.onclick = () => change((rr) => {
      const f = (d?.fields || []).find((x) => !['Name', 'Slide'].includes(x)) || 'Slide';
      rr.source.rules = [...(rr.source.rules || []), { field: f, op: 'in', values: [] }];
    });
    kids.push(add);
    if (d) {
      const inFig = new Set(d.members), adds = new Set(d.pending.add), rem = new Set(d.pending.remove);
      kids.push(h('div', { class: 'row', style: 'margin-top: 6px' }, h('span', { class: 'lbl' }, 'Matches'), h('span', { class: 'grow' }),
        h('span', { class: 'dim mono', style: 'font-size: 11px' }, `${d.members.length - d.pending.remove.length + d.pending.add.length} views`)));
      const ul = h('div', { id: 'fg-matches', role: 'list' });
      for (const v of views.filter((x) => inFig.has(x.id) || adds.has(x.id))) {
        const tag = adds.has(v.id) ? h('span', { class: 'chip warn' }, 'joins on Update') : rem.has(v.id) ? h('span', { class: 'chip warn' }, 'leaves on Update') : null;
        ul.append(h('div', { class: 'fv' + (adds.has(v.id) ? ' dimmed' : ''), role: 'listitem', title: v.slide }, thumb(v),
          h('span', { class: 'txt' }, h('span', { class: 'name' }, v.name || `View ${v.id}`), h('span', { class: 'sub' }, v.meta.Slide || '')), tag));
      }
      kids.push(ul);
    }
  }
  if (d?.closed?.length) kids.push(h('div', { class: 'panel-note warn' }, `${d.closed.length} slide${d.closed.length === 1 ? '' : 's'} not open here: blank in the figure`));
  for (const p of d?.lost || []) {
    const b = h('button', { class: 'btn sm fg-relink', title: `Not found: ${p}` }, `Relink ${p.split(/[\\/]/).pop()}…`);
    b.onclick = () => relink(p);
    kids.push(b);
  }
  box.replaceChildren(...kids);
}
function ruleRow(q, i) {
  const h = X.h;
  const d = F.data;
  const fields = d?.fields || [q.field];
  const set = (fn) => change((rr) => { fn(rr.source.rules[i]); });
  const fsel = select('Field', fields.map((f) => [f, f]), q.field, (v) => set((x) => { x.field = v; x.values = []; }));
  const op = select('Test', [['in', 'is'], ['not in', 'is not'], ['>=', '≥'], ['<=', '≤'], ['has', 'is set'], ['missing', 'is empty']], q.op || 'in',
    (v) => set((x) => { x.op = v; }));
  const rm = h('button', { class: 'btn icon ghost rm', title: 'Remove rule', 'aria-label': 'Remove rule' }, '✕');
  rm.onclick = () => change((rr) => { rr.source.rules.splice(i, 1); });
  const row = h('div', { class: 'rule' }, h('div', { class: 'row' }, fsel, op, rm));
  if ((q.op || 'in') === 'in' || q.op === 'not in') {
    const chosen = new Set((q.values || []).map(String));
    const vals = (d?.values?.[q.field] || []).slice(0, 60);
    const chips = h('div', { class: 'vals' });
    for (const v of vals) {
      const b = h('button', { class: 'vchip', 'aria-pressed': String(chosen.has(v)) }, v);
      b.onclick = () => set((x) => { const s = new Set((x.values || []).map(String)); if (s.has(v)) s.delete(v); else s.add(v); x.values = [...s]; });
      chips.append(b);
    }
    if (!vals.length) chips.append(h('span', { class: 'dim', style: 'font-size: 12px' }, 'No values'));
    row.append(chips);
  } else if (q.op === '>=' || q.op === '<=') {
    const inp = h('input', { type: 'number', step: 'any', value: (q.values || [])[0] ?? '', 'aria-label': 'Value', style: 'width: 100%' });
    inp.onchange = () => set((x) => { x.values = [inp.value]; });
    row.append(inp);
  }
  return row;
}

// --------------------------------------------------------------------------------------------------- toolbar
function renderTools() {
  const h = X.h;
  const box = $('fg-tools');
  if (!F.cur) { box.replaceChildren(); return; }
  const r = R();
  const L = r.layout || {};
  const d = F.data;
  const fields = d?.fields || ['Name', 'Slide'];
  const mode = L.mode === 'field' && L.rows ? 'field' : 'grid';
  const kids = [
    seg('Layout', [['field', 'Rows by field', 'A row per value of a field'], ['grid', 'Grid', 'Rows and columns']], mode,
      (v) => change((rr) => { rr.layout = { ...(rr.layout || {}), mode: v, rows: v === 'field' ? (rr.layout?.rows || fields.find((f) => !['Name'].includes(f)) || 'Slide') : rr.layout?.rows }; }),
      { id: 'fg-mode' }),
  ];
  if (mode === 'field') kids.push(h('label', {}, 'Rows', select('Rows field', fields.map((f) => [f, f]), L.rows, (v) => change((rr) => { rr.layout.rows = v; }), { id: 'fg-rows' })));
  const ord = (L.order || [])[0];
  kids.push(h('label', {}, 'Order', select('Order', [['', 'As listed'], ...fields.map((f) => [f, f])], ord?.field || '', (v) => change((rr) => {
    rr.layout = { ...(rr.layout || {}), order: v ? [{ field: v, desc: !!ord?.desc }] : [] };
  }), { id: 'fg-order' })));
  if (ord) {
    const dir = h('button', { id: 'fg-desc', class: 'btn icon', title: ord.desc ? 'Descending · Select: ascending' : 'Ascending · Select: descending', 'aria-label': 'Order direction' }, ord.desc ? '↓' : '↑');
    dir.onclick = () => change((rr) => { rr.layout.order[0].desc = !rr.layout.order[0].desc; });
    kids.push(dir);
  }
  const slots = r.template === 'pptx' && currentLayout()?.slots?.length;
  if (!slots) {
    kids.push(h('label', {}, 'Columns', select('Columns', [['auto', 'Auto'], ...[1, 2, 3, 4, 5, 6, 8].map((n) => [n, String(n)])], L.cols ?? 'auto',
      (v) => change((rr) => { rr.layout = { ...(rr.layout || {}), cols: v === 'auto' ? 'auto' : Number(v) }; }), { id: 'fg-cols' })));
    if (r.template !== 'pub') {
      kids.push(h('label', {}, 'Per slide', select('Windows per slide', [1, 2, 3, 4, 6, 8, 9, 12].map((n) => [n, String(n)]), L.per_page ?? 6,
        (v) => change((rr) => { rr.layout = { ...(rr.layout || {}), per_page: Number(v) }; }), { id: 'fg-perpage' })));
    }
  }
  const ch = h('button', { id: 'fg-chart', class: 'btn' + (r.chart ? ' on' : ''), 'aria-pressed': String(!!r.chart), title: 'Chart window: one value per view, by group' }, 'Chart');
  ch.onclick = () => change((rr) => { rr.chart = rr.chart ? null : { kind: 'box', measure: defaultMeasure(), group: rr.layout?.mode === 'field' ? rr.layout.rows : null }; });
  kids.push(ch);
  const fl = flChannels();
  if (fl.length >= 2) {
    const C = r.channels || {};
    const split = C.mode === 'split';
    kids.push(h('label', { title: 'Fluorescence: one window per view (merge), or a window per channel plus the merge' }, 'Channels',
      seg('Channels', [['merge', 'Merge', 'One window per view: channels blended'], ['split', 'Split', 'A window per channel, plus the merge: a row per view']], split ? 'split' : 'merge',
        (v) => change((rr) => { rr.channels = { ...(rr.channels || {}), mode: v }; }), { id: 'fg-chmode' })));
    const shown = C.show ? new Set(C.show) : null;
    const chips = h('span', { id: 'fg-chans', class: 'chchips', role: 'group', 'aria-label': 'Channels shown' }, fl.map((c) => {
      const on = !shown || shown.has(c.name);
      const b = h('button', { class: 'chchip', 'aria-pressed': String(on), title: `${c.name}: ${on ? 'hide' : 'show'} in every window · Alt-click: only this one` },
        h('i', { style: `background: ${c.color}` }), c.name);
      b.onclick = (e) => change((rr) => {
        const cur = new Set(rr.channels?.show || fl.map((x) => x.name));
        if (e.altKey || e.shiftKey) { const alone = cur.size === 1 && cur.has(c.name); cur.clear(); if (alone) fl.forEach((x) => cur.add(x.name)); else cur.add(c.name); }
        else if (cur.has(c.name)) { if (cur.size > 1) cur.delete(c.name); } else cur.add(c.name);
        rr.channels = { ...(rr.channels || {}), show: cur.size === fl.length ? null : [...cur] };
      });
      return b;
    }));
    kids.push(chips);
    if (split) {
      const mg = h('button', { id: 'fg-chmerge', class: 'btn sm' + (C.merge !== false ? ' on' : ''), 'aria-pressed': String(C.merge !== false), title: 'A merge window after the channels' }, '+ Merge');
      mg.onclick = () => change((rr) => { rr.channels = { ...(rr.channels || {}), merge: rr.channels?.merge === false }; });
      const gr = h('button', { id: 'fg-chgray', class: 'btn sm' + (C.gray ? ' on' : ''), 'aria-pressed': String(!!C.gray), title: 'Single channels in grey' }, 'Grey');
      gr.onclick = () => change((rr) => { rr.channels = { ...(rr.channels || {}), gray: !rr.channels?.gray }; });
      kids.push(mg, gr);
    }
  }
  kids.push(h('span', { class: 'grow' }));
  kids.push(h('label', { title: 'Auto: black when every view is fluorescence' }, 'Background',
    seg('Background', [['auto', 'Auto', 'Black when every view is fluorescence, else white'], ['white', 'White'], ['black', 'Black']], r.background || 'auto',
      (v) => change((rr) => { rr.background = v; }), { id: 'fg-bg' })));
  if (r.template === 'pub') {
    kids.push(h('label', {}, 'Page', select('Page', [['double', 'Double column · 180 mm'], ['single', 'Single column · 85 mm'], ['full', 'Full page · 180 × 240 mm']], r.page || 'double',
      (v) => change((rr) => { rr.page = v; }), { id: 'fg-page' })));
  } else if (r.template === 'ppt') {
    kids.push(h('label', {}, 'Slide', select('Slide size', [['16:9', '16:9'], ['4:3', '4:3']], r.page || '16:9', (v) => change((rr) => { rr.page = v; }), { id: 'fg-page' })));
  } else {
    const tsel = select('Template', [['', F.templates.length ? 'Choose template' : 'No template yet'], ...F.templates.map((t) => [t.id, t.name])], r.pptx?.id || '',
      (v) => change((rr) => { rr.pptx = v ? { id: v, layout: null } : null; }), { id: 'fg-tpl' });
    const up = h('input', { type: 'file', accept: '.pptx,.potx', id: 'fg-tpl-file', hidden: true });
    up.onchange = () => { if (up.files[0]) uploadTemplate(up.files[0]); up.value = ''; };
    const add = h('button', { id: 'fg-tpl-add', class: 'btn', title: 'Add a PowerPoint template (.pptx, .potx) or a past talk: its master, layouts and arrangements' }, 'Add template…');
    add.onclick = () => up.click();
    kids.push(h('label', {}, 'Template', tsel), add, up);
  }
  box.replaceChildren(...kids);
}
/** The channels of the figure's fluorescence views (by name, first seen colour). */
function flChannels() {
  const d = F.data;
  if (!d) return [];
  const inFig = new Set(d.members);
  const out = new Map();
  for (const v of d.views || []) if (inFig.has(v.id) && v.fluorescence && v.channels) for (const c of v.channels) if (!out.has(c.name)) out.set(c.name, c);
  return [...out.values()];
}
function defaultMeasure() {
  const L = F.data?.layers?.[0];
  if (L) return { kind: 'count', layer: L.name, class: null };
  const num = (F.data?.fields || []).find((f) => !['Size (MB)', 'Modified', 'Name', 'Slide'].includes(f) && (F.data.values[f] || []).length && F.data.values[f].every((v) => !isNaN(Number(v))));
  return num ? { kind: 'field', field: num } : { kind: 'count', layer: null };
}
async function uploadTemplate(file) {
  const headers = { 'X-Wissi-Token': X.TOKEN };
  if (X.key()) headers['X-Wissi-Key'] = X.key();
  X.toast({ title: `Reading ${file.name}…`, ms: 2500 });
  const r = await fetch(`${X.BASE}/app/figures/templates?name=${encodeURIComponent(file.name)}`, { method: 'POST', headers, body: file });
  let j = {};
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) { X.toast({ kind: 'bad', title: 'Template not added', sub: j.detail || r.statusText }); return; }
  F.templates = j.templates || F.templates;
  const t = j.template;
  const learned = t.layouts.filter((x) => x.source === 'slides').length;
  X.toast({ title: `${t.name}: ${t.layouts.length - learned} layout${t.layouts.length - learned === 1 ? '' : 's'}${learned ? ` + ${learned} from its slides` : ''}`,
    sub: `${t.fonts?.heading || ''}${t.fonts?.body ? ' · ' + t.fonts.body : ''} · Choose a layout above the page` });
  change((rr) => { rr.template = 'pptx'; rr.pptx = { id: t.id, layout: (t.layouts.find((x) => x.slots.length >= 2) || t.layouts.find((x) => x.slots.length) || t.layouts[0])?.name || null }; });
}
function currentLayout() {
  const t = F.data?.template;
  if (!t) return null;
  const name = R().pptx?.layout;
  return t.layouts.find((x) => x.name === name) || t.layouts.find((x) => x.slots.length) || t.layouts[0] || null;
}
function renderLayouts() {
  const h = X.h;
  const box = $('fg-layouts');
  const t = F.data?.template;
  if (!F.cur || R().template !== 'pptx' || !t) { box.hidden = true; box.replaceChildren(); return; }
  box.hidden = false;
  const cur = currentLayout();
  const [W, H] = t.size;
  const k = 64 / W;
  const kids = [h('span', { class: 'dim', style: 'font-size: 12px' }, 'Layouts')];
  for (const L of t.layouts) {
    const mini = h('span', { class: 'mini', style: `width: ${W * k}px; height: ${H * k}px; background: ${t.background || '#fff'}` });
    for (const m of [...(t.master || []), ...(L.master || [])]) {
      if (m.kind === 'rect') mini.append(h('span', { style: `left: ${m.rect[0] * k}px; top: ${m.rect[1] * k}px; width: ${m.rect[2] * k}px; height: ${m.rect[3] * k}px; background: ${m.color}` }));
    }
    if (L.title) mini.append(h('span', { class: 't', style: `left: ${L.title[0] * k}px; top: ${L.title[1] * k}px; width: ${L.title[2] * k}px; height: ${Math.max(2, L.title[3] * k)}px` }));
    for (const s of L.slots) mini.append(h('span', { class: 's', style: `left: ${s[0] * k}px; top: ${s[1] * k}px; width: ${s[2] * k}px; height: ${s[3] * k}px` }));
    const b = h('button', { class: 'lay', 'aria-pressed': String(L === cur), title: `${L.name} · ${L.slots.length ? `${L.slots.length} window${L.slots.length === 1 ? '' : 's'}` : 'windows in a grid'}${L.source === 'slides' ? ' · learned from the slides' : ''}` },
      mini, h('span', { class: 'n' }, L.name), L.source === 'slides' ? h('span', { class: 'from' }, 'from slides') : null);
    b.onclick = () => change((rr) => { rr.pptx = { ...(rr.pptx || {}), layout: L.name }; rr.windows = {}; });
    kids.push(b);
  }
  box.replaceChildren(...kids);
}

// ------------------------------------------------------------------------------------------------------- pages
function allWindows() { return (F.data?.pages || []).flatMap((p) => p.windows); }
function windowByKey(key) { return allWindows().find((w) => w.key === key) || null; }
let SC = 3;   // px per mm on screen
function renderPages() {
  const h = X.h;
  const box = $('fg-pages');
  const d = F.data;
  if (!d) { box.replaceChildren(h('div', { class: 'panel-note' }, 'Laying out…')); return; }
  const avail = Math.max(240, $('fg-canvas').clientWidth - 48);
  const W = d.pages[0]?.w || 180;
  SC = Math.min(5, avail / W);
  const tpl = d.template;
  const lay = R().template === 'pptx' ? currentLayout() : null;
  const kids = [];
  if (d.warnings?.length) kids.push(h('div', { class: 'panel-note warn' }, d.warnings.join(' · ')));
  if (!d.members.length) kids.push(h('div', { class: 'panel-note' }, R().source?.kind === 'query' ? 'No view matches · Set rules (left)' : 'No views · Tick views (left)'));
  d.pages.forEach((p, pi) => {
    const bg = p.bg === 'black' ? '#000' : p.bg === 'white' ? '#fff' : (tpl?.background || '#fff');
    const page = h('div', { class: 'fpage', role: 'group', 'aria-label': `${d.pages.length > 1 ? `Page ${pi + 1}` : 'Page'}`,
      style: `width: ${p.w * SC}px; height: ${p.h * SC}px; background: ${bg}; color: ${p.ink}` });
    page.dataset.page = String(pi);
    if (R().template === 'pptx' && tpl) {
      for (const m of [...(tpl.master || []), ...(lay?.master || [])]) {
        const st = `left: ${m.rect[0] * SC}px; top: ${m.rect[1] * SC}px; width: ${m.rect[2] * SC}px; height: ${m.rect[3] * SC}px`;
        page.append(m.kind === 'rect' ? h('span', { class: 'fmaster', style: `${st}; background: ${m.color}` }) : h('img', { class: 'fmaster', alt: '', src: m.src, style: st }));
      }
    }
    if (p.title?.rect) {
      const [x, y, w, hh] = p.title.rect;
      const fs = Math.min(28 * mmPt * SC, hh * SC * 0.6);
      page.append(h('div', { class: 'ftitle', style: `left: ${x * SC}px; top: ${y * SC}px; width: ${w * SC}px; height: ${hh * SC}px; font-size: ${fs}px;${tpl?.fonts?.heading ? ` font-family: '${tpl.fonts.heading}', Georgia, serif;` : ''}` }, p.title.text || ''));
    }
    for (const c of p.cols || []) {
      const [x, y, w, hh] = c.rect;
      page.append(h('div', { class: 'fcol', style: `left: ${x * SC}px; top: ${y * SC}px; width: ${w * SC}px; height: ${hh * SC}px; font-size: ${d.font * mmPt * SC}px; color: ${c.color || p.ink}` }, c.text));
    }
    for (const hd of p.heads) {
      const [x, y, w, hh] = hd.rect;
      page.append(h('div', { class: 'fhead', style: `left: ${x * SC}px; top: ${y * SC}px; width: ${w * SC}px; height: ${hh * SC}px; font-size: ${d.font * mmPt * SC}px` },
        h('span', {}, hd.text)));
    }
    for (const w of p.windows) page.append(windowEl(w, p, d));
    kids.push(page);
    if (d.pages.length > 1) kids.push(h('div', { class: 'fpno dim' }, `${R().template === 'pub' ? 'Page' : 'Slide'} ${pi + 1} of ${d.pages.length}`));
  });
  box.replaceChildren(...kids);
}
function windowEl(w, p, d) {
  const h = X.h;
  const [x, y, ww, wh] = w.rect;
  const fs = d.font * mmPt * SC;
  const el = h('div', { class: `fwin k-${w.kind}${F.sel === w.key ? ' sel' : ''}${w.edited ? ' edited' : ''}`, tabindex: 0, role: 'button',
    'aria-label': `Window ${w.letter || ''} ${w.kind === 'view' ? w.label || '' : w.kind}`.trim(),
    style: `left: ${x * SC}px; top: ${y * SC}px; width: ${ww * SC}px; height: ${wh * SC}px`, dataset: { key: w.key } });
  if (w.kind === 'view') {
    if (w.sid) {
      const px = Math.min(1600, Math.round(Math.max(ww, wh) * SC * (window.devicePixelRatio || 1)));
      el.append(h('img', { alt: '', draggable: 'false', src: thumbUrl(w.sid, w.bounds, px, w) }));
      if (w.inset) {
        const [ix, iy, iw, ih] = w.inset.rect, [fx0, fy0, fw0, fh0] = w.inset.frame;
        const edge = w.fluorescence ? '#fff' : '#111';
        const ipx = Math.min(1200, Math.round(Math.max(iw, ih) * SC * (window.devicePixelRatio || 1)));
        el.append(h('img', { class: 'finset', alt: '', draggable: 'false', src: thumbUrl(w.sid, w.inset.bounds, ipx, w),
          style: `left: ${(ix - x) * SC}px; top: ${(iy - y) * SC}px; width: ${iw * SC}px; height: ${ih * SC}px; border-color: ${edge}` }));
        const fr = h('span', { class: 'fframe', title: 'The part the inset shows · Drag: move it', style: `left: ${(fx0 - x) * SC}px; top: ${(fy0 - y) * SC}px; width: ${fw0 * SC}px; height: ${fh0 * SC}px; border-color: ${edge}` });
        fr.addEventListener('pointerdown', (e) => dragFrame(e, w, fr));
        el.append(fr);
      }
      if (w.legend?.length) {
        el.append(h('span', { class: 'fkey', style: `font-size: ${fs * 0.8}px; color: ${w.fluorescence ? '#fff' : '#111'}` },
          w.legend.map((k) => h('span', {}, h('i', { style: `background: ${k.color}` }), k.name))));
      }
    } else el.append(h('span', { class: 'fmissing' }, 'Slide not open'));
    if (w.scalebar) {
      const sb = w.scalebar;
      el.append(h('span', { class: 'fsb', style: `color: ${sb.color}; font-size: ${fs * 0.8}px` }, sb.text,
        h('i', { style: `width: ${sb.mm * SC}px; background: ${sb.color}; height: ${Math.max(2, wh * 0.015 * SC)}px` })));
    }
  } else if (w.kind === 'chart') {
    el.append(chartSVG(d.series, ww * SC, wh * SC, p.ink, fs));
  } else if (w.kind === 'text') {
    el.append(h('span', { class: 'ftext', style: `font-size: ${fs}px` }, w.text || 'Text · Edit in the panel (right)'));
  } else {
    el.append(h('span', { class: 'fmissing' }, w.missing ? 'View deleted' : 'Empty'));
  }
  if (w.letter) {
    const col = w.kind === 'view' ? (w.fluorescence ? '#fff' : '#111') : p.ink;
    el.append(h('span', { class: 'flet', style: `color: ${col}; font-size: ${fs * 1.3}px` }, w.letter));
  }
  const out = [el];
  if (w.kind === 'view' && w.label && w.label_rect) {
    const [lx, ly, lw] = w.label_rect;
    out.push(h('div', { class: 'flabel', style: `left: ${lx * SC}px; top: ${ly * SC}px; width: ${lw * SC}px; font-size: ${fs}px` }, w.label));
  }
  if (F.sel === w.key) for (const c of ['nw', 'ne', 'sw', 'se']) el.append(h('span', { class: `hdl ${c}`, dataset: { c } }));
  if (F.sel === w.key && w.kind === 'view' && w.fluorescence) { const bar = windowChips(w, 'fchips'); if (bar) el.append(bar); }
  el.addEventListener('pointerdown', (e) => startDrag(e, w, el));
  el.addEventListener('focus', () => { if (F.sel !== w.key) { F.sel = w.key; renderInsp(); markSel(); } });
  const frag = document.createDocumentFragment();
  frag.append(...out);
  return frag;
}
/** Channel chips for one window: click shows / hides a channel in this window only (Alt-click: only it). */
function windowChips(w, cls) {
  const h = X.h;
  const v = (F.data?.views || []).find((x) => x.id === w.view);
  const chans = v?.channels || [];
  if (chans.length < 2) return null;
  const cur = new Set(w.channels || chans.map((_, i) => i));
  const bar = h('span', { class: cls, role: 'group', 'aria-label': 'Channels in this window' });
  chans.forEach((c, i) => {
    const b = h('button', { class: 'chchip', 'aria-pressed': String(cur.has(i)), title: `${c.name}: ${cur.has(i) ? 'hide' : 'show'} in this window · Alt-click: only this one` },
      h('i', { style: `background: ${c.color}` }), c.name);
    b.onpointerdown = (e) => e.stopPropagation();
    b.onclick = (e) => {
      e.stopPropagation();
      const next = new Set(cur);
      if (e.altKey || e.shiftKey) { next.clear(); if (cur.size === 1 && cur.has(i)) chans.forEach((_, k) => next.add(k)); else next.add(i); }
      else if (next.has(i)) { if (next.size > 1) next.delete(i); } else next.add(i);
      setShows(w, { kind: 'view', channels: [...next].sort((a, b2) => a - b2) });
    };
    bar.append(b);
  });
  return bar;
}
function thumbUrl(sid, bounds, px, w) {
  const q = [`size=${px}`, `bounds=${bounds.map((v) => Math.round(v)).join(',')}`];
  if (w?.channels) q.push(`channels=${w.channels.join(',')}`);
  if (w?.gray) q.push('gray=true');
  if (w?.layers?.length) q.push(`layers=${encodeURIComponent(w.layers.join('|'))}`);
  return `${X.API}/slides/${X.enc(sid)}/thumbnail?${q.join('&')}`;
}
function markSel() {
  if (F.data && allWindows().some((w) => w.key === F.sel && w.kind === 'view' && w.fluorescence)) { renderPages(); return; }
  for (const el of document.querySelectorAll('#fg-pages .fwin')) {
    const on = el.dataset.key === F.sel;
    el.classList.toggle('sel', on);
    el.querySelectorAll('.hdl').forEach((x) => x.remove());
    if (on) for (const c of ['nw', 'ne', 'sw', 'se']) el.append(X.h('span', { class: `hdl ${c}`, dataset: { c } }));
  }
}

// ---- move / resize a window (snaps to the page's edges and centre, and to other windows' edges)
function startDrag(e, w, el) {
  if (e.button !== 0) return;
  e.preventDefault();
  el.focus({ preventScroll: true });
  const corner = e.target.dataset?.c || null;
  if (F.sel !== w.key) { F.sel = w.key; markSel(); renderInsp(); }
  const page = F.data.pages[w.page];
  const r0 = [...w.rect];
  const sx = e.clientX, sy = e.clientY;
  let moved = false, rect = r0;
  const others = page.windows.filter((o) => o.key !== w.key).map((o) => o.rect);
  const xs = [0, page.w / 2, page.w, ...others.flatMap((o) => [o[0], o[0] + o[2]])];
  const ys = [0, page.h / 2, page.h, ...others.flatMap((o) => [o[1], o[1] + o[3]])];
  const tol = 6 / SC;
  const snap = (v, list) => { let best = v, d = tol; for (const c of list) if (Math.abs(c - v) < d) { d = Math.abs(c - v); best = c; } return best; };
  const onMove = (ev) => {
    const dx = (ev.clientX - sx) / SC, dy = (ev.clientY - sy) / SC;
    if (!moved && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 3) return;
    moved = true;
    let [x, y, ww, hh] = r0;
    if (!corner) {
      x += dx; y += dy;
      const sxL = snap(x, xs), sxR = snap(x + ww, xs) - ww;
      x = Math.abs(sxL - x) <= Math.abs(sxR - x) ? sxL : sxR;
      const syT = snap(y, ys), syB = snap(y + hh, ys) - hh;
      y = Math.abs(syT - y) <= Math.abs(syB - y) ? syT : syB;
    } else {
      let x1 = x + ww, y1 = y + hh;
      if (corner.includes('w')) x = snap(x + dx, xs); else x1 = snap(x1 + dx, xs);
      if (corner.includes('n')) y = snap(y + dy, ys); else y1 = snap(y1 + dy, ys);
      if (ev.shiftKey) {   // keep the shape
        const a = r0[2] / r0[3];
        if (corner.includes('w')) x = x1 - (y1 - y) * a; else x1 = x + (y1 - y) * a;
      }
      ww = Math.max(5, x1 - x); hh = Math.max(5, y1 - y);
    }
    rect = [x, y, ww, hh];
    Object.assign(el.style, { left: `${x * SC}px`, top: `${y * SC}px`, width: `${ww * SC}px`, height: `${hh * SC}px` });
  };
  const onUp = () => {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    if (moved) setRect(w, rect.map((v) => Math.round(v * 10) / 10));
  };
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
}
function dragFrame(e, w, fr) {
  if (e.button !== 0 || F.sel !== w.key) return;   // a click on an unselected window selects it
  e.preventDefault(); e.stopPropagation();
  const sx = e.clientX, sy = e.clientY, [x, y, ww, hh] = w.rect;
  const src = [...w.inset.src];
  let next = src;
  const onMove = (ev) => {
    const dx = (ev.clientX - sx) / SC / ww, dy = (ev.clientY - sy) / SC / hh;
    next = [Math.min(1 - src[2], Math.max(0, src[0] + dx)), Math.min(1 - src[3], Math.max(0, src[1] + dy)), src[2], src[3]];
    fr.style.left = `${next[0] * ww * SC}px`; fr.style.top = `${next[1] * hh * SC}px`;
  };
  const onUp = () => {
    window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp);
    if (next !== src) setInset(w, { src: next.map((v) => Math.round(v * 1000) / 1000) });
  };
  window.addEventListener('pointermove', onMove); window.addEventListener('pointerup', onUp);
  void x; void y;
}
function setInset(w, cfg) {
  override(w, (o) => { o.inset = cfg === null ? undefined : { ...(o.inset || (w.inset ? { src: w.inset.src, corner: w.inset.corner, size: w.inset.size } : {})), ...cfg }; if (cfg === null) delete o.inset; });
}
function override(w, fn) {
  change((r) => {
    if (w.extra) {
      const i = Number(w.key.slice(1));
      if (r.extra?.[i]) fn(r.extra[i]);
    } else {
      r.windows = r.windows || {};
      r.windows[w.key] = r.windows[w.key] || {};
      fn(r.windows[w.key]);
    }
  });
}
function setRect(w, rect) { override(w, (o) => { o.rect = rect; }); }
function setShows(w, shows) { override(w, (o) => { o.shows = shows; }); }
function resetWindow(w) {
  change((r) => {
    if (w.extra) r.extra.splice(Number(w.key.slice(1)), 1);
    else if (r.windows) delete r.windows[w.key];
  });
  if (w.extra) F.sel = null;   // a window placed in the layout stays selected
}
function addWindow(kind) {
  const d = F.data;
  const p = d?.pages[0];
  if (!p) return;
  const pi = Math.max(0, d.pages.findIndex((x) => x.windows.some((w) => w.key === F.sel)));
  change((r) => {
    r.extra = [...(r.extra || []), { page: pi, rect: [p.w * 0.35, p.h * 0.35, p.w * 0.3, p.w * 0.3 * 0.75], shows: kind === 'text' ? { kind: 'text', text: '' } : { kind: 'empty' } }];
  });
  F.sel = `x${(R().extra || []).length - 1}`;
}
function onKey(e) {
  if (!F.open) return;
  const t = e.target;
  const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
  if (e.key === 'Escape') {
    if (typing) { t.blur(); e.stopPropagation(); return; }
    if (F.exportOpen) { F.exportOpen = false; renderExport(); } else if (F.sel) { F.sel = null; renderPages(); renderInsp(); } else closeFigures();
    e.preventDefault(); e.stopPropagation(); return;
  }
  if (typing) { e.stopPropagation(); return; }
  const w = F.sel && windowByKey(F.sel);
  if (w && e.key.startsWith('Arrow')) {
    const k = e.shiftKey ? 5 : 0.5;
    const [x, y, ww, hh] = w.rect;
    const dx = e.key === 'ArrowLeft' ? -k : e.key === 'ArrowRight' ? k : 0, dy = e.key === 'ArrowUp' ? -k : e.key === 'ArrowDown' ? k : 0;
    setRect(w, [x + dx, y + dy, ww, hh]);
    e.preventDefault();
  } else if (w && (e.key === 'Delete' || e.key === 'Backspace')) {
    if (w.extra) resetWindow(w); else setShows(w, { kind: 'empty' });
    e.preventDefault();
  }
  e.stopPropagation();   // the viewer's shortcuts stay off while composing
}

// ------------------------------------------------------------------------------------------------------- chart
function niceStep(x) { if (!(x > 0)) return 1; const e = 10 ** Math.floor(Math.log10(x)); return [1, 2, 5, 10].map((m) => m * e).find((v) => v >= x) || 10 * e; }
function stats(vals) {
  const v = vals.filter((x) => x != null && isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const q = (p) => { const i = (v.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return v[lo] + (v[hi] - v[lo]) * (i - lo); };
  const q1 = q(0.25), q3 = q(0.75), iqr = q3 - q1;
  return { q1, q3, median: q(0.5), lo: Math.min(...v.filter((x) => x >= q1 - 1.5 * iqr)), hi: Math.max(...v.filter((x) => x <= q3 + 1.5 * iqr)), mean: v.reduce((a, b) => a + b, 0) / v.length };
}
function chartSVG(series, W, H, ink, fs) {
  const S = X.s;
  const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'fchart', role: 'img', 'aria-label': series ? `${series.kind} chart of ${series.ylabel}` : 'Chart' });
  if (!series || !series.groups.length) { svg.append(S('text', { x: W / 2, y: H / 2, fill: ink, 'font-size': fs, 'text-anchor': 'middle' }, 'No values · Set the measure (right)')); return svg; }
  const ml = fs * 5.2 * 0.9, mb = fs * 3.2 * 0.9, mt = fs * 2.2 * 0.9, mr = 4;
  const pw = Math.max(10, W - ml - mr), ph = Math.max(10, H - mb - mt);
  const vals = series.groups.flatMap((g) => g.values.map((x) => x.value)).filter((v) => v != null);
  const lo = Math.min(0, ...vals), hiRaw = Math.max(...vals, lo + 1e-9);
  const step = niceStep((hiRaw - lo) / 4), hi = Math.ceil(hiRaw / step) * step;
  const Y = (v) => mt + ph - ((v - lo) / (hi - lo || 1)) * ph;
  const n = series.groups.length;
  const Xc = (i) => ml + ((i + 0.5) / n) * pw;
  svg.append(S('line', { x1: ml, y1: mt, x2: ml, y2: mt + ph, stroke: ink, 'stroke-width': 1 }), S('line', { x1: ml, y1: mt + ph, x2: ml + pw, y2: mt + ph, stroke: ink, 'stroke-width': 1 }));
  for (let v = lo; v <= hi + 1e-9; v += step) {
    svg.append(S('line', { x1: ml - 3, x2: ml, y1: Y(v), y2: Y(v), stroke: ink }), S('text', { x: ml - 5, y: Y(v) + fs * 0.32, fill: ink, 'font-size': fs * 0.9, 'text-anchor': 'end' }, String(+v.toFixed(6))));
  }
  svg.append(S('text', { x: fs * 0.9, y: mt + ph / 2, fill: ink, 'font-size': fs, 'text-anchor': 'middle', transform: `rotate(-90 ${fs * 0.9} ${mt + ph / 2})` }, series.ylabel || ''));
  const bw = Math.min(40, (pw / n) * 0.5);
  series.groups.forEach((g, i) => {
    const cx = Xc(i), st = stats(g.values.map((x) => x.value)), c = g.color;
    svg.append(S('text', { x: cx, y: mt + ph + fs * 1.3, fill: ink, 'font-size': fs * 0.9, 'text-anchor': 'middle' }, g.name));
    if (st && series.kind === 'box') {
      svg.append(S('rect', { x: cx - bw / 2, y: Y(st.q3), width: bw, height: Math.max(1, Y(st.q1) - Y(st.q3)), fill: c + '33', stroke: c, 'stroke-width': 1.2 }),
        S('line', { x1: cx - bw / 2, x2: cx + bw / 2, y1: Y(st.median), y2: Y(st.median), stroke: c, 'stroke-width': 2 }),
        S('line', { x1: cx, x2: cx, y1: Y(st.lo), y2: Y(st.q1), stroke: c }), S('line', { x1: cx, x2: cx, y1: Y(st.q3), y2: Y(st.hi), stroke: c }));
    } else if (st && series.kind === 'bar') {
      svg.append(S('rect', { x: cx - bw * 0.6, y: Y(st.mean), width: bw * 1.2, height: Math.max(0, Y(lo) - Y(st.mean)), fill: c + '99', stroke: c }));
    } else if (st) {
      svg.append(S('line', { x1: cx - bw / 2, x2: cx + bw / 2, y1: Y(st.median), y2: Y(st.median), stroke: c, 'stroke-width': 2 }));
    }
    const pts = g.values.filter((x) => x.value != null), k = Math.min(pts.length, 5);
    pts.forEach((x, j) => svg.append(S('circle', { cx: cx + ((j % k) - (k - 1) / 2) * (bw * 0.14), cy: Y(x.value), r: Math.max(1.5, fs * 0.22), fill: c },
      S('title', {}, `${x.label}: ${x.value}`))));
  });
  return svg;
}

// --------------------------------------------------------------------------------------------------- inspector
function renderInsp() {
  const h = X.h;
  const box = $('fg-insp');
  if (!F.cur) { box.replaceChildren(); return; }
  const r = R();
  const d = F.data;
  const w = F.sel && windowByKey(F.sel);
  const kids = [];
  if (w) kids.push(...windowPanel(w));
  else {
    const slide = r.template !== 'pub';
    kids.push(h('div', { class: 'lbl' }, slide ? 'Slides' : 'Figure'));
    const lab = h('input', { id: 'fg-labels', value: r.labels ?? '{Name}', 'aria-label': 'Labels: fields in braces', class: 'mono', style: 'width: 100%' });
    lab.onchange = () => change((rr) => { rr.labels = lab.value; });
    const chips = h('div', { class: 'vals' }, (d?.fields || []).slice(0, 24).map((f) => {
      const b = h('button', { class: 'vchip', title: `Insert {${f}}` }, f);
      b.onclick = () => { lab.value = `${lab.value}${lab.value.trim() ? ' · ' : ''}{${f}}`; lab.onchange(); };
      return b;
    }));
    const grid = h('div', { class: 'f2' },
      'Labels', lab, h('span'), chips,
      'Letters', seg('Letters', [[true, 'A B C'], [false, 'None']], r.letters !== false, (v) => change((rr) => { rr.letters = v; }), { id: 'fg-letters' }),
      'Scale bar', select('Scale bar', [['auto', 'Auto'], ['', 'None'], ...[10, 20, 50, 100, 200, 500, 1000, 2000].map((v) => [v, v >= 1000 ? `${v / 1000} mm` : `${v} µm`])],
        r.scalebar === null ? '' : r.scalebar ?? 'auto', (v) => change((rr) => { rr.scalebar = v === 'auto' ? 'auto' : v === '' ? null : Number(v); }), { id: 'fg-sb' }),
      'Window shape', select('Window shape', [['auto', 'As the views'], [1, '1 : 1'], [4 / 3, '4 : 3'], [3 / 2, '3 : 2'], [16 / 9, '16 : 9'], [3 / 4, '3 : 4']], r.layout?.aspect ?? 'auto',
        (v) => change((rr) => { rr.layout = { ...(rr.layout || {}), aspect: v === 'auto' ? 'auto' : Number(v) }; }), { id: 'fg-aspect' }),
      'Text size', select('Text size', [['', 'Auto'], ...(slide ? [10, 12, 14, 16, 18, 20, 24] : [6, 7, 8, 9, 10, 12]).map((v) => [v, `${v} pt`])], r.font ?? '',
        (v) => change((rr) => { rr.font = v ? Number(v) : null; }), { id: 'fg-font' }),
      'Gap', select('Gap', [['', 'Auto'], ...[0, 0.5, 1, 1.5, 2, 3, 4, 6, 8].map((v) => [v, `${v} mm`])], r.layout?.gap ?? '',
        (v) => change((rr) => { rr.layout = { ...(rr.layout || {}), gap: v === '' ? null : Number(v) }; }), { id: 'fg-gap' }));
    if (d?.layers?.length) {
      const cur = new Set(r.layers || []);
      grid.append('Layers', h('span', { class: 'vals', id: 'fg-layers' }, d.layers.map((L) => {
        const b = h('button', { class: 'vchip', 'aria-pressed': String(cur.has(L.name)), title: `${L.name}: outlines ${cur.has(L.name) ? 'off' : 'on'} in every window, with a key` }, L.name);
        b.onclick = () => change((rr) => { const n = new Set(rr.layers || []); if (n.has(L.name)) n.delete(L.name); else n.add(L.name); rr.layers = [...n]; });
        return b;
      })));
    }
    if (slide) {
      const ti = h('input', { id: 'fg-title', value: r.title || '', placeholder: F.cur.name, 'aria-label': 'Slide title' });
      ti.onchange = () => change((rr) => { rr.title = ti.value; });
      grid.append('Title', ti);
    }
    kids.push(grid);
    const bgNote = d ? (d.background === 'black' ? 'Black page: every view is fluorescence' : d.background === 'white' ? (r.background === 'auto' ? 'White page: a brightfield view is in it' : 'White page') : 'Template’s background') : '';
    if (bgNote) kids.push(h('div', { class: 'panel-note' }, `${bgNote} · Background: toolbar`));
    const addT = h('button', { id: 'fg-add-text', class: 'btn sm', title: 'Add a text window' }, '+ Text');
    addT.onclick = () => addWindow('text');
    const addE = h('button', { id: 'fg-add-empty', class: 'btn sm', title: 'Add a window: then choose what it shows' }, '+ Window');
    addE.onclick = () => addWindow('empty');
    const nEdit = Object.keys(r.windows || {}).length + (r.extra || []).length;
    const reset = h('button', { id: 'fg-reset-all', class: 'btn sm ghost', title: 'Windows back to the layout: moves, sizes and changes undone', disabled: !nEdit }, 'Reset windows');
    reset.onclick = () => change((rr) => { rr.windows = {}; rr.extra = []; });
    kids.push(h('div', { class: 'row', style: 'flex-wrap: wrap' }, addT, addE, h('span', { class: 'grow' }), reset));
    kids.push(h('div', { class: 'panel-note' }, 'Select a window: move it, drag a corner to resize (Shift: keep shape), arrows nudge, Delete empties it.'));
    if (r.chart) kids.push(...chartPanel());
  }
  box.replaceChildren(...kids);
}
function windowPanel(w) {
  const h = X.h;
  const d = F.data;
  const kids = [];
  const back = h('button', { id: 'fg-win-close', class: 'btn icon ghost', title: 'Back to the figure’s settings', 'aria-label': 'Back to the figure settings' }, '✕');
  back.onclick = () => { F.sel = null; markSel(); renderInsp(); };
  kids.push(h('div', { class: 'row' }, h('span', { class: 'lbl' }, `Window ${w.letter || ''}`), h('span', { class: 'grow' }), back));
  const views = d.views || [];
  const inFig = new Set(d.members);
  const opts = [['chart', 'Chart'], ['text', 'Text'], ['empty', 'Empty'],
    ...views.filter((v) => inFig.has(v.id)).map((v) => [`v${v.id}`, `View · ${v.name || v.id}`]),
    ...views.filter((v) => !inFig.has(v.id)).map((v) => [`v${v.id}`, `Other view · ${v.name || v.id}`])];
  const cur = w.kind === 'view' ? `v${w.view}` : w.kind;
  kids.push(h('label', { class: 'f2' }, 'Shows', select('Window shows', opts, cur, (v) => {
    if (v.startsWith('v')) setShows(w, { kind: 'view', view: Number(v.slice(1)) });
    else if (v === 'text') setShows(w, { kind: 'text', text: w.text || '' });
    else setShows(w, { kind: v });
  }, { id: 'fg-shows' })));
  if (w.kind === 'view') {
    const v = views.find((x) => x.id === w.view);
    if (v) {
      const img = h('img', { class: 'thumb big', alt: '' });
      if (w.sid) img.src = `${X.API}/slides/${X.enc(w.sid)}/thumbnail?size=240&bounds=${w.bounds.map((x) => Math.round(x)).join(',')}`;
      kids.push(h('div', { class: 'row' }, img, h('div', { class: 'txt' }, h('b', {}, v.name || `View ${v.id}`), h('div', { class: 'dim', style: 'font-size: 12px' }, v.meta.Slide || ''))));
      kids.push(h('div', { class: 'f2 meta' }, Object.entries(v.meta).filter(([k]) => k !== 'Slide').slice(0, 12).flatMap(([k, val]) => [k, h('span', {}, String(val ?? ''))])));
      if (w.fluorescence) { const bar = windowChips(w, 'chchips'); if (bar) kids.push(h('div', { class: 'f2' }, 'Channels', bar)); }
      if (d.layers?.length && w.sid) {
        const cur = new Set(w.layers || []);
        kids.push(h('div', { class: 'f2' }, 'Layers', h('span', { class: 'vals', id: 'fg-wlayers' }, d.layers.map((L) => {
          const b = h('button', { class: 'vchip', 'aria-pressed': String(cur.has(L.name)), title: `${L.name}: outlines ${cur.has(L.name) ? 'off' : 'on'} in this window` }, L.name);
          b.onclick = () => { const n = new Set(cur); if (n.has(L.name)) n.delete(L.name); else n.add(L.name); override(w, (o) => { o.layers = [...n]; }); };
          return b;
        }))));
      }
      if (!w.inset) {
        const ib = h('button', { id: 'fg-inset-add', class: 'btn sm', title: 'A zoom of part of this window, in a corner: drag its dashed frame to choose the part' }, '+ Inset');
        ib.onclick = () => setInset(w, { src: [0.4, 0.4, 0.2, 0.2], corner: 'tr', size: 0.4 });
        kids.push(h('div', { class: 'row' }, ib));
      } else {
        const zoom = Math.round((w.inset.size / w.inset.src[2]) * 10) / 10;
        const rm = h('button', { id: 'fg-inset-remove', class: 'btn sm ghost', title: 'Remove the inset' }, 'Remove');
        rm.onclick = () => setInset(w, null);
        kids.push(h('div', { class: 'f2' }, 'Inset', h('span', { class: 'row', style: 'flex-wrap: wrap; gap: 4px' },
          seg('Inset corner', [['tl', '↖'], ['tr', '↗'], ['bl', '↙'], ['br', '↘']], w.inset.corner, (v) => setInset(w, { corner: v }), { id: 'fg-inset-corner' }),
          select('Inset zoom', [2, 3, 4, 6, 8].map((z) => [z, `${z}×`]), [2, 3, 4, 6, 8].reduce((a, b) => (Math.abs(b - zoom) < Math.abs(a - zoom) ? b : a)),
            (v) => { const z = Number(v), f = w.inset.size / z, c = [w.inset.src[0] + w.inset.src[2] / 2, w.inset.src[1] + w.inset.src[3] / 2];
              setInset(w, { src: [Math.max(0, Math.min(1 - f, c[0] - f / 2)), Math.max(0, Math.min(1 - f, c[1] - f / 2)), f, f] }); }, { id: 'fg-inset-zoom' }),
          rm)));
      }
      const adj = h('button', { id: 'fg-adjust', class: 'btn', title: 'Show the view in the viewer: frame it, then Update the view (toast)' }, 'Adjust in viewer');
      adj.onclick = () => adjustInViewer(v);
      kids.push(h('div', { class: 'row' }, h('span', { class: 'chip' }, 'View locked'), h('span', { class: 'grow' }), adj));
    }
  } else if (w.kind === 'text') {
    const ta = h('textarea', { id: 'fg-text', rows: 4, 'aria-label': 'Text of the window', style: 'width: 100%' });
    ta.value = w.text || '';
    ta.onchange = () => setShows(w, { kind: 'text', text: ta.value });
    kids.push(ta);
  } else if (w.kind === 'chart') {
    kids.push(...chartPanel());
  }
  const [x, y, ww, hh] = w.rect;
  kids.push(h('div', { class: 'panel-note mono' }, `${x.toFixed(1)}, ${y.toFixed(1)} mm · ${ww.toFixed(1)} × ${hh.toFixed(1)} mm${w.edited ? ' · placed by hand' : ''}`));
  const reset = h('button', { id: 'fg-win-reset', class: 'btn sm', title: w.extra ? 'Remove this window' : 'Back to the layout: place, size and what it shows', disabled: !w.edited }, w.extra ? 'Remove window' : 'Reset window');
  reset.onclick = () => resetWindow(w);
  kids.push(h('div', { class: 'row' }, reset));
  return kids;
}
function chartPanel() {
  const h = X.h;
  const r = R();
  const d = F.data;
  const c = r.chart;
  if (!c) return [];
  const m = c.measure || {};
  const setC = (fn) => change((rr) => { fn(rr.chart); });
  const kids = [h('div', { class: 'lbl', style: 'margin-top: 8px' }, 'Chart'),
    seg('Chart kind', [['box', 'Box'], ['bar', 'Bar'], ['dots', 'Dots']], c.kind || 'box', (v) => setC((cc) => { cc.kind = v; }), { id: 'fg-ckind' })];
  const grid = h('div', { class: 'f2' });
  grid.append('Measure', select('Measure', [['count', 'Count of objects'], ['density', 'Objects per mm²'], ['field', 'A field’s value']], m.kind || 'count',
    (v) => setC((cc) => { cc.measure = v === 'field' ? { kind: 'field', field: (d?.fields || []).find((f) => (d.values[f] || []).every((x) => !isNaN(Number(x)))) || null } : { kind: v, layer: m.layer || d?.layers?.[0]?.name || null, class: null }; }), { id: 'fg-cmeasure' }));
  if ((m.kind || 'count') === 'field') {
    const nums = (d?.fields || []).filter((f) => (d.values[f] || []).length && d.values[f].every((x) => !isNaN(Number(x))));
    grid.append('Field', select('Field', nums.map((f) => [f, f]), m.field || '', (v) => setC((cc) => { cc.measure.field = v; }), { id: 'fg-cfield' }));
  } else {
    const layers = d?.layers || [];
    grid.append('Layer', select('Layer', layers.length ? layers.map((L) => [L.name, L.name]) : [['', 'No layers on these slides']], m.layer || '', (v) => setC((cc) => { cc.measure.layer = v; cc.measure.class = null; }), { id: 'fg-clayer' }));
    const L = layers.find((x) => x.name === m.layer);
    if (L?.classes?.length) grid.append('Class', select('Class', [['', 'All'], ...L.classes.map((k) => [k, k])], m.class || '', (v) => setC((cc) => { cc.measure.class = v || null; }), { id: 'fg-cclass' }));
  }
  grid.append('Group by', select('Group by', [['', 'None'], ...(d?.fields || []).map((f) => [f, f])], c.group || '', (v) => setC((cc) => { cc.group = v || null; }), { id: 'fg-cgroup' }));
  kids.push(grid);
  const rm = h('button', { id: 'fg-chart-remove', class: 'btn sm ghost', title: 'Remove the chart window' }, 'Remove chart');
  rm.onclick = () => change((rr) => { rr.chart = null; });
  kids.push(h('div', { class: 'row' }, h('span', { class: 'dim', style: 'font-size: 12px' }, 'One value per view · Data: CSV on export'), h('span', { class: 'grow' }), rm));
  return kids;
}
function adjustInViewer(v) {
  closeFigures();
  X.showView({ sid: v.sid, view: v, bounds: v.bounds, id: `v${v.id}`, name: v.name, meta: v.meta });
  X.toast({ title: `Frame “${v.name || 'the view'}”`, sub: 'Pan and zoom · Then Update view: the figure follows', ms: 60000,
    action: { label: 'Update view', fn: async () => {
      const b = X.activeBounds();
      if (!b) return;
      await X.quiet(X.http(`${X.BASE}/app/views/${v.id}`, { bounds: b.map((x) => Math.round(x * 10) / 10) }));
      X.loadViews();
      openFigures();
    } } });
}

// ------------------------------------------------------------------------------------------------------ export
function renderExport() {
  const h = X.h;
  const box = $('fg-export');
  if (!F.exportOpen || !F.cur) { box.hidden = true; return; }
  box.hidden = false;
  const r = R();
  const slide = r.template !== 'pub';
  const fmt = F.fmt || (slide ? 'pptx' : 'tiff');
  F.fmt = fmt;
  const d = F.data;
  const pend = d ? d.pending.add.length + d.pending.remove.length : 0;
  const dpi = select('Resolution', [[150, '150 dpi'], [300, '300 dpi'], [600, '600 dpi'], [1200, '1200 dpi']], F.dpi || 300, (v) => { F.dpi = Number(v); }, { id: 'fg-dpi' });
  const snap = h('input', { type: 'checkbox', id: 'fg-snap', checked: F.snap !== false });
  snap.onchange = () => { F.snap = snap.checked; };
  const data = h('input', { type: 'checkbox', id: 'fg-data', checked: F.dataCsv !== false, disabled: !r.chart });
  data.onchange = () => { F.dataCsv = data.checked; };
  const go = h('button', { id: 'fg-export-go', class: 'btn primary' }, 'Export');
  go.onclick = () => doExport(fmt, go);
  const cancel = h('button', { id: 'fg-export-cancel', class: 'btn' }, 'Cancel');
  cancel.onclick = () => { F.exportOpen = false; renderExport(); };
  const n = d?.pages.length || 1;
  const notes = { png: 'Pictures: one per page', tiff: 'Pictures (LZW): one per page · For journals', pdf: 'Text and chart as vectors · One file', svg: 'Text and chart as vectors · One per page', pptx: slide ? `One slide per page${r.template === 'pptx' ? ' · On the template’s master' : ''} · Text stays editable` : 'One slide the size of the page' };
  box.replaceChildren(
    h('div', { class: 'row' }, h('b', {}, `Export ${F.cur.name}`)),
    seg('Format', [['png', 'PNG'], ['tiff', 'TIFF'], ['pdf', 'PDF'], ['svg', 'SVG'], ['pptx', 'PPTX']], fmt, (v) => { F.fmt = v; renderExport(); }, { id: 'fg-fmt' }),
    h('div', { class: 'panel-note' }, `${notes[fmt]}${n > 1 && ['png', 'tiff', 'svg'].includes(fmt) ? ` · ${n} pages: a zip` : ''}`),
    h('div', { class: 'f2' }, 'Resolution', dpi),
    h('label', { class: 'row' }, snap, 'Recipe snapshot (.recipe.json): redraw exactly later'),
    h('label', { class: 'row' }, data, 'Chart data (.csv): one row per view'),
    pend ? h('div', { class: 'panel-note warn' }, `${pend} view${pend === 1 ? '' : 's'} would change on Update · Export as shown, or Update first`) : null,
    h('div', { class: 'row' }, h('span', { class: 'grow' }), cancel, go));
}
async function doExport(fmt, btn) {
  flushSave();
  const headers = { 'X-Wissi-Token': X.TOKEN, 'Content-Type': 'application/json' };
  if (X.key()) headers['X-Wissi-Key'] = X.key();
  btn.disabled = true; btn.textContent = 'Exporting…';
  try {
    const r = await fetch(`${X.BASE}/app/figures/export`, { method: 'POST', headers,
      body: JSON.stringify({ recipe: R(), name: F.cur.name, fmt, dpi: F.dpi || 300, snapshot: F.snap !== false, data: !!R().chart && F.dataCsv !== false }) });
    if (!r.ok) {
      let m = r.statusText;
      try { m = (await r.json()).detail || m; } catch (e) { /* plain */ }
      throw new Error(m);
    }
    const name = decodeURIComponent(r.headers.get('X-Filename') || `${F.cur.name}.${fmt}`);
    const where = await X.saveBlob(await r.blob(), name);
    if (where) X.toast({ title: 'Figure exported', sub: where });
    F.exportOpen = false;
    renderExport();
  } catch (e) {
    X.toast({ kind: 'bad', title: 'Export failed', sub: e.message, ms: 10000 });
  } finally {
    btn.disabled = false; btn.textContent = 'Export';
  }
}
