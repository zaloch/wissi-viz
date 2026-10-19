// wsiviz — GPU whole-slide image grid viewer (WebGL2, zero dependencies).
//
//   <script type="module" src="/wsiviz/static/index.js"></script>
//   <wsi-grid api="/wsiviz" slides="slideA,slideB,slideC,slideD"></wsi-grid>
//
// One WebGL2 context renders 1–16 panes (browsers cap live contexts at ~16, so one
// context per slide would not scale). All panes share a GPU texture LRU, a worker pool
// that fetches + decodes off the main thread, and a priority scheduler that loads
// coarse levels first, then the view centre-out, then a predictive margin.

const MAX_CH = 8;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const now = () => performance.now();

function hexToRgb(h) {
  h = String(h || '#ffffff').replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h.slice(0, 6), 16) || 0;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
const rgbToHex = (c) => '#' + c.map((v) => Math.round(clamp(v, 0, 1) * 255).toString(16).padStart(2, '0')).join('');

// ------------------------------------------------------------------ fetch + decode (worker)
async function wsivizLoad(url, kind, signal) {
  const r = await fetch(url, { signal, credentials: 'same-origin' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  if (kind === 'image') {
    const blob = await r.blob();
    const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    return { bmp, transfer: [bmp] };
  }
  if (kind === 'raw') {
    const ct = r.headers.get('content-type') || '';
    let buf;
    if (ct.includes('x-wsiviz-raw')) {
      buf = await new Response(r.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
    } else buf = await r.arrayBuffer();
    return { buf, transfer: [buf] };
  }
  const buf = await r.arrayBuffer();
  return { buf, transfer: [buf] };
}

const WORKER_SRC = `const wsivizLoad = ${wsivizLoad.toString()};
const ctrls = new Map();
self.onmessage = async (e) => {
  const m = e.data;
  if (m.abort) { const c = ctrls.get(m.id); if (c) c.abort(); return; }
  const ac = new AbortController(); ctrls.set(m.id, ac);
  try { const r = await wsivizLoad(m.url, m.kind, ac.signal); const t = r.transfer; delete r.transfer; r.id = m.id; self.postMessage(r, t); }
  catch (err) { self.postMessage({ id: m.id, error: err && err.name === 'AbortError' ? 'abort' : String(err && err.message || err) }); }
  finally { ctrls.delete(m.id); }
};`;

class Fetcher {
  constructor(n) {
    this.seq = 0;
    this.cb = new Map();
    this.workers = [];
    this.rr = 0;
    try {
      const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'text/javascript' }));
      for (let i = 0; i < n; i++) {
        const w = new Worker(url);
        w.onmessage = (e) => this._msg(e.data);
        this.workers.push(w);
      }
    } catch (e) {
      this.workers = []; // CSP forbids blob workers → main-thread fallback
    }
  }
  _msg(m) {
    const cb = this.cb.get(m.id);
    if (!cb) return;
    this.cb.delete(m.id);
    if (m.error) cb.reject(new Error(m.error));
    else cb.resolve(m);
  }
  load(url, kind) {
    const id = ++this.seq;
    const promise = new Promise((resolve, reject) => this.cb.set(id, { resolve, reject }));
    if (this.workers.length) {
      const w = this.workers[this.rr++ % this.workers.length];
      w.postMessage({ id, url, kind });
      return { promise, abort: () => { w.postMessage({ abort: true, id }); } };
    }
    const ac = new AbortController();
    wsivizLoad(url, kind, ac.signal).then(
      (r) => { delete r.transfer; r.id = id; this._msg(r); },
      (err) => this._msg({ id, error: err.name === 'AbortError' ? 'abort' : String(err.message || err) }),
    );
    return { promise, abort: () => ac.abort() };
  }
  destroy() { this.workers.forEach((w) => w.terminate()); }
}

// ------------------------------------------------------------------ shaders
const VS_TILE = `#version 300 es
uniform vec4 u_rect; uniform vec3 u_view; uniform vec2 u_vp;
out vec2 v_uv;
void main(){
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
  v_uv = c;
  vec2 s = (u_rect.xy + c * u_rect.zw - u_view.xy) * u_view.z;
  gl_Position = vec4(s.x / (u_vp.x * 0.5), -s.y / (u_vp.y * 0.5), 0.0, 1.0);
}`;

const FS_RGB = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec4 u_adj; // brightness, contrast, gamma, invert
in vec2 v_uv; out vec4 o;
void main(){
  vec3 c = texture(u_tex, v_uv).rgb;
  c = (c - 0.5) * u_adj.y + 0.5 + u_adj.x;
  c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / u_adj.z));
  if (u_adj.w > 0.5) c = 1.0 - c;
  o = vec4(c, 1.0);
}`;

function fsChannels(n) {
  let decl = '', body = '';
  for (let i = 0; i < n; i++) {
    decl += `uniform highp usampler2D u_c${i};\n`;
    body += `  if (u_n > ${i}) acc += u_col[${i}] * chan(u_c${i}, u_win[${i}]);\n`;
  }
  return `#version 300 es
precision highp float; precision highp int;
${decl}
uniform int u_n; uniform vec3 u_col[${n}]; uniform vec4 u_win[${n}]; // lo, hi, gamma, -
uniform int u_nearest; uniform vec3 u_bg;
in vec2 v_uv; out vec4 o;
float px(highp usampler2D t, ivec2 p, ivec2 sz){ return float(texelFetch(t, clamp(p, ivec2(0), sz - 1), 0).r); }
float chan(highp usampler2D t, vec4 w){
  ivec2 sz = textureSize(t, 0);
  vec2 f = v_uv * vec2(sz) - 0.5;
  float v;
  if (u_nearest == 1) v = px(t, ivec2(floor(f + 0.5)), sz);
  else {
    ivec2 i = ivec2(floor(f)); vec2 a = fract(f);
    v = mix(mix(px(t, i, sz), px(t, i + ivec2(1, 0), sz), a.x),
            mix(px(t, i + ivec2(0, 1), sz), px(t, i + ivec2(1, 1), sz), a.x), a.y);
  }
  v = clamp((v - w.x) / max(w.y - w.x, 1e-6), 0.0, 1.0);
  return pow(v, 1.0 / w.z);
}
void main(){
  vec3 acc = u_bg;
${body}  o = vec4(min(acc, vec3(1.0)), 1.0);
}`;
}

const VS_PTS = `#version 300 es
layout(location=0) in vec2 a_pos; layout(location=1) in float a_size;
layout(location=2) in uint a_cls; layout(location=3) in uint a_kind;
uniform vec3 u_view; uniform vec2 u_vp; uniform vec2 u_ps; uniform int u_onlyPoints;
uniform sampler2D u_ctex; uniform float u_alpha; uniform float u_dpr;
out vec4 v_col;
void main(){
  vec2 s = (a_pos - u_view.xy) * u_view.z;
  gl_Position = vec4(s.x / (u_vp.x * 0.5), -s.y / (u_vp.y * 0.5), 0.0, 1.0);
  float sz = a_kind == 0u && a_size <= 0.0 ? u_ps.y : clamp(a_size * u_view.z, u_ps.x, u_ps.y);
  if (u_onlyPoints == 1 && a_kind != 0u) { sz = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }
  gl_PointSize = sz;
  v_col = texelFetch(u_ctex, ivec2(int(a_cls), 0), 0);
  v_col.a *= u_alpha;
}`;
const FS_PTS = `#version 300 es
precision mediump float;
in vec4 v_col; out vec4 o;
void main(){ vec2 c = gl_PointCoord - 0.5; float d = dot(c, c); if (d > 0.25 || v_col.a <= 0.0) discard;
  o = vec4(v_col.rgb, v_col.a * smoothstep(0.25, 0.16, d)); }`;

const VS_SEG = `#version 300 es
layout(location=0) in vec4 a_seg; layout(location=1) in uint a_cls;
uniform vec3 u_view; uniform vec2 u_vp; uniform float u_w; uniform sampler2D u_ctex; uniform float u_alpha;
out vec4 v_col;
void main(){
  vec2 p0 = (a_seg.xy - u_view.xy) * u_view.z, p1 = (a_seg.zw - u_view.xy) * u_view.z;
  vec2 d = p1 - p0; float L = length(d); vec2 t = L > 1e-6 ? d / L : vec2(1.0, 0.0); vec2 n = vec2(-t.y, t.x);
  float along = float(gl_VertexID & 1), side = (gl_VertexID >> 1) == 0 ? -1.0 : 1.0;
  vec2 p = mix(p0, p1, along) + t * (along * 2.0 - 1.0) * u_w * 0.5 + n * side * u_w * 0.5;
  gl_Position = vec4(p.x / (u_vp.x * 0.5), -p.y / (u_vp.y * 0.5), 0.0, 1.0);
  v_col = texelFetch(u_ctex, ivec2(int(a_cls), 0), 0); v_col.a *= u_alpha;
}`;
const VS_TRI = `#version 300 es
layout(location=0) in vec2 a_pos; layout(location=1) in uint a_cls;
uniform vec3 u_view; uniform vec2 u_vp; uniform sampler2D u_ctex; uniform float u_alpha;
out vec4 v_col;
void main(){
  vec2 s = (a_pos - u_view.xy) * u_view.z;
  gl_Position = vec4(s.x / (u_vp.x * 0.5), -s.y / (u_vp.y * 0.5), 0.0, 1.0);
  v_col = texelFetch(u_ctex, ivec2(int(a_cls), 0), 0); v_col.a *= u_alpha;
}`;
const FS_FLAT = `#version 300 es
precision mediump float; in vec4 v_col; out vec4 o;
void main(){ if (v_col.a <= 0.0) discard; o = v_col; }`;

function compile(gl, vs, fs) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src);
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    const name = info.name.replace(/\[0\]$/, '');
    u[name] = gl.getUniformLocation(p, info.name);
  }
  return { p, u };
}

// ------------------------------------------------------------------ binary annotation packs
function unpack(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x31415357) throw new Error('bad annotation pack'); // 'WSA1'
  const jl = dv.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jl)));
  let off = 8 + jl;
  const take = (Ctor, n) => {
    const a = new Ctor(buf, off, n);
    off += Math.ceil((n * Ctor.BYTES_PER_ELEMENT) / 4) * 4;
    return a;
  };
  return { meta, take };
}

// ------------------------------------------------------------------ data model
class Slide {
  constructor(grid, url) {
    this.grid = grid;
    this.url = url.replace(/\/+$/, '');
    this.info = null;
    this.error = null;
    this.layers = new Map();
    this.ready = this._load();
  }
  async _load() {
    try {
      const r = await fetch(this.url + '/info.json', { credentials: 'same-origin' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const info = await r.json();
      info.levels.forEach((lv) => { lv.sx = info.width / lv.width; lv.sy = info.height / lv.height; });
      info.bg = hexToRgb(info.background);
      // HTTP/2+ multiplexes → allow many parallel tile requests; HTTP/1.1 browsers cap at 6 per host,
      // so keep our own (prioritised, abortable) queue instead of the browser's FIFO.
      const e = performance.getEntriesByName(this.url + '/info.json')[0];
      const proto = e && e.nextHopProtocol;
      if (proto) this.grid.maxInflight = /^h[23]/.test(proto) ? 24 : 5; // leave 1 of 6 HTTP/1.1 sockets for info/annotation calls
      this.info = info;
      this._syncLayers(info.annotations || []);
      if (info.annotations_pending) this._annTimer = setTimeout(() => this.refreshAnnotations(), 500);
    } catch (e) {
      this.error = String(e.message || e);
    }
    this.grid._slideReady(this);
  }
  _syncLayers(metas) {
    const names = new Set();
    for (const m of metas) {
      names.add(m.name);
      const L = this.layers.get(m.name);
      if (L && L.meta.version === m.version) continue;
      if (L) L.destroy();
      this.layers.set(m.name, new AnnLayer(this.grid, this, m));
    }
    for (const [n, L] of this.layers) if (!names.has(n)) { L.destroy(); this.layers.delete(n); }
  }
  async refreshAnnotations() {
    // never hold one of the browser's ~6 HTTP/1.1 connections open: poll while the server parses sidecars
    clearTimeout(this._annTimer);
    try {
      const r = await fetch(this.url + '/annotations?wait=false', { credentials: 'same-origin', cache: 'no-cache' });
      if (r.ok) {
        this._syncLayers(await r.json());
        if (r.headers.get('X-Pending') === '1') this._annTimer = setTimeout(() => this.refreshAnnotations(), 800);
      }
    } catch (_) { this._annTimer = setTimeout(() => this.refreshAnnotations(), 3000); }
    this.grid.requestRender();
    this.grid._panelRefresh();
  }
  tileUrl(level, x, y, c) {
    const i = this.info;
    if (i.tile_url) return new URL(i.tile_url.replace('{level}', level).replace('{x}', x).replace('{y}', y).replace('{c}', c ?? 0), document.baseURI).href;
    return `${this.url}/tiles/${level}/${x}/${y}?v=${encodeURIComponent(i.version)}${c == null ? '' : '&c=' + c}`;
  }
  annUrl(path, layer, version) {
    return `${this.url}/annotations/${path}?layer=${encodeURIComponent(layer)}&v=${version}`;
  }
}

class AnnLayer {
  constructor(grid, slide, meta) {
    this.grid = grid;
    this.slide = slide;
    this.meta = meta;
    this.name = meta.name;
    this.state = 'idle';
    this.detail = new Map(); // bucket → {vaoSeg, nSeg, vaoTri, nTri, bytes, last, bufs}
    this.hidden = new Set();
  }
  get style() { return this.meta.style || {}; }
  _ingest(buf) {
    const gl = this.grid.gl;
    const { meta, take } = unpack(buf);
    const n = meta.count, B = meta.buckets;
    this.cen = take(Float32Array, n * 2);
    this.size = take(Float32Array, n);
    this.orig = take(Uint32Array, n);
    this.cls = take(Uint16Array, n);
    this.kind = take(Uint8Array, n);
    this.buckets = take(Float32Array, B * 8);
    this.n = n;
    this.nb = B;
    this.classes = meta.classes || [];
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.bufs = [
      this._attr(0, this.cen, 2, gl.FLOAT, false),
      this._attr(1, this.size, 1, gl.FLOAT, false),
      this._attr(2, this.cls, 1, gl.UNSIGNED_SHORT, true),
      this._attr(3, this.kind, 1, gl.UNSIGNED_BYTE, true),
    ];
    gl.bindVertexArray(null);
    this.ctex = gl.createTexture();
    this._classTex();
    this.grid.gpuBytes += buf.byteLength;
    this.state = 'ready';
  }
  _attr(loc, data, size, type, int, divisor = 0) {
    const gl = this.grid.gl;
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    if (int) gl.vertexAttribIPointer(loc, size, type, 0, 0);
    else gl.vertexAttribPointer(loc, size, type, false, 0, 0);
    if (divisor) gl.vertexAttribDivisor(loc, divisor);
    return b;
  }
  _classTex() {
    const gl = this.grid.gl;
    const n = Math.max(1, this.classes.length);
    const px = new Uint8Array(n * 4);
    this.classes.forEach((c, i) => {
      const [r, g, b] = hexToRgb(c.color);
      px.set([r * 255, g * 255, b * 255, this.hidden.has(c.name) ? 0 : 255], i * 4);
    });
    gl.bindTexture(gl.TEXTURE_2D, this.ctex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }
  setClassVisible(name, vis) {
    if (vis) this.hidden.delete(name); else this.hidden.add(name);
    if (this.state === 'ready') this._classTex();
  }
  setClassColor(name, color) {
    const c = this.classes.find((k) => k.name === name);
    if (c) { c.color = color; if (this.state === 'ready') this._classTex(); }
  }
  ingestBucket(i, buf) {
    const gl = this.grid.gl;
    const { meta, take } = unpack(buf);
    const S = meta.segments, T = meta.tri_verts;
    const seg = take(Float32Array, S * 4), tv = take(Float32Array, T * 2);
    const sc = take(Uint16Array, S), tc = take(Uint16Array, T);
    const d = { nSeg: S, nTri: T, bytes: buf.byteLength, last: this.grid.frame, bufs: [] };
    if (S) {
      d.vaoSeg = gl.createVertexArray();
      gl.bindVertexArray(d.vaoSeg);
      d.bufs.push(this._attr(0, seg, 4, gl.FLOAT, false, 1), this._attr(1, sc, 1, gl.UNSIGNED_SHORT, true, 1));
    }
    if (T) {
      d.vaoTri = gl.createVertexArray();
      gl.bindVertexArray(d.vaoTri);
      d.bufs.push(this._attr(0, tv, 2, gl.FLOAT, false), this._attr(1, tc, 1, gl.UNSIGNED_SHORT, true));
    }
    gl.bindVertexArray(null);
    this.detail.set(i, d);
    this.grid.gpuBytes += d.bytes;
  }
  dropBucket(i) {
    const d = this.detail.get(i);
    if (!d) return;
    const gl = this.grid.gl;
    d.bufs.forEach((b) => gl.deleteBuffer(b));
    if (d.vaoSeg) gl.deleteVertexArray(d.vaoSeg);
    if (d.vaoTri) gl.deleteVertexArray(d.vaoTri);
    this.grid.gpuBytes -= d.bytes;
    this.detail.delete(i);
  }
  destroy() {
    const gl = this.grid.gl;
    for (const i of [...this.detail.keys()]) this.dropBucket(i);
    if (this.state === 'ready') {
      this.bufs.forEach((b) => gl.deleteBuffer(b));
      gl.deleteVertexArray(this.vao);
      gl.deleteTexture(this.ctex);
    }
    this.state = 'dead';
  }
  // CPU hit test on the overview (only buckets around the point are scanned)
  hit(x, y, tol) {
    if (this.state !== 'ready') return -1;
    let best = -1, bestSize = Infinity;
    const B = this.buckets;
    for (let b = 0; b < this.nb; b++) {
      const o = b * 8;
      if (x < B[o] - tol || x > B[o + 2] + tol || y < B[o + 1] - tol || y > B[o + 3] + tol) continue;
      const s0 = B[o + 4], n = B[o + 5];
      for (let f = s0; f < s0 + n; f++) {
        const r = this.size[f] / 2 + tol;
        const dx = x - this.cen[2 * f], dy = y - this.cen[2 * f + 1];
        if (dx * dx + dy * dy <= r * r && this.size[f] < bestSize && !this.hidden.has(this.classes[this.cls[f]]?.name)) {
          best = f;
          bestSize = this.size[f];
        }
      }
    }
    return best;
  }
}

class Pane {
  constructor(grid, index) {
    this.grid = grid;
    this.index = index;
    this.slide = null;
    this.view = null; // {cx, cy, scale}  scale = CSS px per level-0 px
    this.target = null; // animation target
    this.vel = null; // inertia (level-0 px / ms)
    this.channels = null;
    this.adjust = { brightness: 0, contrast: 1, gamma: 1, invert: false };
    this.layerVis = new Map();
    this.rect = { x: 0, y: 0, w: 0, h: 0 };
    this.selection = null;
    this.nearest = false;
    this.el = null;
    this.lastEmit = '';
  }
  layerVisible(L) {
    const v = this.layerVis.get(L.name);
    return v === undefined ? L.style.visible !== false : v;
  }
}

// ------------------------------------------------------------------ the element
const CSS = `
:host { display:block; position:relative; width:100%; height:100%; min-height:240px; contain: strict; }
.root { position:absolute; inset:0; overflow:hidden; background:var(--wsi-gap, #0b0d10); outline:none;
  font: 12px/1.3 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#e8e8e8; touch-action:none; user-select:none; }
canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
canvas.ov { pointer-events:none; }
.panes { position:absolute; inset:0; display:grid; gap:var(--wsi-gap-size, 2px); }
.pane { position:relative; min-width:0; min-height:0; outline:1px solid rgba(255,255,255,.08); cursor:grab; }
.pane.active { outline:2px solid var(--wsi-accent, #3ea6ff); outline-offset:-2px; z-index:1; }
.pane.drag { cursor:grabbing; }
.pane.tool { cursor:crosshair; }
.label { position:absolute; left:6px; top:6px; padding:2px 6px; border-radius:4px; background:rgba(0,0,0,.55);
  max-width:calc(100% - 60px); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; pointer-events:none; }
.scale { position:absolute; left:6px; bottom:6px; pointer-events:none; padding:3px 6px 4px; border-radius:4px; background:rgba(0,0,0,.5); }
.scale i { display:block; height:3px; background:#fff; box-shadow:0 0 2px #000; margin-top:2px; }
.coords { position:absolute; right:6px; bottom:6px; pointer-events:none; padding:2px 6px; border-radius:4px; background:rgba(0,0,0,.5); font-variant-numeric:tabular-nums; }
.coords:empty { display:none; }
.msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; pointer-events:none; opacity:.7; }
.btn { position:absolute; right:6px; top:6px; width:26px; height:26px; border-radius:6px; border:0; background:rgba(0,0,0,.55);
  color:#fff; cursor:pointer; font-size:14px; display:flex; align-items:center; justify-content:center; }
.panel { position:absolute; right:6px; top:36px; width:min(270px, calc(100% - 12px)); max-height:calc(100% - 44px); overflow:auto;
  background:rgba(18,20,24,.94); border:1px solid rgba(255,255,255,.12); border-radius:8px; padding:8px; z-index:3; cursor:default; }
.panel h4 { margin:6px 0 4px; font-size:11px; text-transform:uppercase; letter-spacing:.06em; opacity:.6; font-weight:600; }
.row { display:flex; align-items:center; gap:6px; margin:3px 0; }
.row label { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.row input[type=color] { width:22px; height:18px; padding:0; border:0; background:none; }
.row input[type=range] { width:100%; }
.sub { padding-left:18px; font-size:11px; opacity:.9; }
`;

const SVG_GEAR = '⚙';

export class WsiGrid extends HTMLElement {
  static get observedAttributes() { return ['api', 'slides', 'layout', 'sync', 'controls', 'quality', 'tool', 'draw-layer']; }

  constructor() {
    super();
    const sr = this.attachShadow({ mode: 'open' });
    sr.innerHTML = `<style>${CSS}</style><div class="root" tabindex="0"><canvas class="gl"></canvas><canvas class="ov"></canvas><div class="panes"></div></div>`;
    this.$root = sr.querySelector('.root');
    this.$canvas = sr.querySelector('canvas.gl');
    this.$ov = sr.querySelector('canvas.ov');
    this.$panes = sr.querySelector('.panes');
    this.panes = [];
    this.slides = new Map();
    this.active = 0;
    this.frame = 0;
    this.texBytes = 0;
    this.gpuBytes = 0;
    this.tex = new Map(); // key → {tex, bytes, last}
    this.inflight = new Map(); // key → {abort, prio}
    this.failed = new Map(); // key → retry-at
    this.uploads = [];
    this.layoutSpec = 'auto';
    this.syncMode = 'none';
    this.tool = 'pan';
    this.drawLayer = null;
    this.quality = 1.0;
    this.showControls = true;
    this.maxInflight = 5;
    this.budget = Math.min(1024, ((navigator.deviceMemory || 8) * 96)) * 2 ** 20; // GPU texture budget (bytes)
    this._raf = 0;
    this._drawing = null;
  }

  // ---------------------------------------------------------------- lifecycle
  connectedCallback() {
    if (this.gl) return;
    const gl = this.$canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false,
      preserveDrawingBuffer: false, powerPreference: 'high-performance' });
    if (!gl) {
      this.$root.innerHTML = '<div class="msg">WebGL2 is not available in this browser.</div>';
      return;
    }
    this.gl = gl;
    this.$canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this._lost = true; });
    this.$canvas.addEventListener('webglcontextrestored', () => this._restore());
    this.ctx2d = this.$ov.getContext('2d');
    this._initGL();
    this.fetcher = new Fetcher(Math.min(6, Math.max(2, (navigator.hardwareConcurrency || 4) >> 1)));
    this._ro = new ResizeObserver(() => this._measure());
    this._ro.observe(this);
    this._bindInput();
    if (!this.api) this.api = this.getAttribute('api') || new URL('..', import.meta.url).href.replace(/\/$/, '');
    if (!this.panes.length) {
      const s = this.getAttribute('slides');
      this.setSlides(s ? s.split(',').map((x) => x.trim()).filter(Boolean) : []);
    }
    this.requestRender();
  }

  disconnectedCallback() {
    this._ro?.disconnect();
    cancelAnimationFrame(this._raf);
    this._raf = 0;
  }

  attributeChangedCallback(name, _old, val) {
    if (name === 'api') this.api = (val || '').replace(/\/$/, '');
    else if (name === 'slides' && this.gl) this.setSlides((val || '').split(',').map((x) => x.trim()).filter(Boolean));
    else if (name === 'layout') this.setLayout(val || 'auto');
    else if (name === 'sync') this.setSync(val || 'none');
    else if (name === 'controls') { this.showControls = val !== 'false'; this._buildPanes(); }
    else if (name === 'quality') this.quality = clamp(parseFloat(val) || 1, 0.25, 4);
    else if (name === 'tool') this.setTool(val || 'pan');
    else if (name === 'draw-layer') this.drawLayer = val || null;
  }

  _initGL() {
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.prog = {
      rgb: compile(gl, VS_TILE, FS_RGB),
      ch: compile(gl, VS_TILE, fsChannels(MAX_CH)),
      pts: compile(gl, VS_PTS, FS_PTS),
      seg: compile(gl, VS_SEG, FS_FLAT),
      tri: compile(gl, VS_TRI, FS_FLAT),
    };
    this.emptyVao = gl.createVertexArray();
  }

  _restore() {
    this._lost = false;
    this.tex.clear();
    this.texBytes = 0;
    this.gpuBytes = 0;
    this._initGL();
    for (const s of this.slides.values()) for (const L of s.layers.values()) { L.state = 'idle'; L.detail.clear(); }
    this.requestRender();
  }

  // ---------------------------------------------------------------- public API
  /** Panes from slide ids / URLs / {slide|src, view, channels, layers} objects. Up to 16. */
  setSlides(list) {
    list = (list || []).slice(0, 16);
    const n = Math.max(1, list.length);
    while (this.panes.length > n) this.panes.pop();
    while (this.panes.length < n) this.panes.push(new Pane(this, this.panes.length));
    list.forEach((ref, i) => this.setSlide(i, ref));
    if (!list.length) this.panes[0].slide = null;
    this.active = Math.min(this.active, this.panes.length - 1);
    this._buildPanes();
    this._gcSlides();
  }

  setSlide(pane, ref) {
    const p = this._pane(pane);
    if (!p) return;
    const opts = typeof ref === 'object' && ref ? ref : { slide: ref };
    const url = opts.src || opts.url || (opts.slide != null ? `${this.api}/slides/${encodeURIComponent(opts.slide)}` : null);
    if (!url) { p.slide = null; this.requestRender(); return; }
    const abs = new URL(url, document.baseURI).href.replace(/\/+$/, '');
    let s = this.slides.get(abs);
    if (!s) { s = new Slide(this, abs); this.slides.set(abs, s); }
    const changed = p.slide !== s;
    p.slide = s;
    if (changed) { p.view = null; p.target = null; p.channels = null; p.selection = null; p.layerVis = new Map(); }
    p.pending = opts;
    if (s.info) this._initPane(p);
    this._updateLabel(p);
    this._gcSlides();
    this.requestRender();
  }

  setLayout(spec) {
    this.layoutSpec = spec || 'auto';
    this._applyLayout();
  }

  /** 'none' | 'pan' | 'zoom' | 'all' — linked navigation across panes (in µm when calibrated). */
  setSync(mode) { this.syncMode = mode || 'none'; }

  setActive(pane) {
    const p = this._pane(pane);
    if (!p) return;
    this.active = p.index;
    this.panes.forEach((q) => q.el?.classList.toggle('active', q === p && this.panes.length > 1));
    this._emit('activepane', { pane: p.index });
    this._panelRefresh();
  }

  /**
   * Centre a pane on (x, y). Coordinates are level-0 pixels (opts.units = 'px', default) or microns ('um').
   * Zoom: opts.zoom (screen px per image px) | opts.mpp (µm per screen px) | opts.magnification (e.g. 40)
   *       | opts.fov (image px visible across the pane) | opts.fovUm. Omit to keep the current zoom.
   */
  goTo(pane, x, y, opts = {}) {
    for (const p of this._panes(pane)) {
      const info = p.slide?.info;
      if (!info) { if (p.slide) p.pendingGoto = [x, y, opts]; continue; }
      const mpp = info.mpp ? info.mpp[0] : null;
      let cx = x, cy = y;
      if (opts.units === 'um') {
        if (!mpp) continue;
        cx = x / info.mpp[0];
        cy = y / info.mpp[1];
      }
      let scale = (p.target || p.view)?.scale ?? this._fitScale(p);
      if (opts.zoom) scale = opts.zoom;
      else if (opts.mpp && mpp) scale = mpp / opts.mpp;
      else if (opts.magnification && mpp) scale = mpp / (10 / opts.magnification);
      else if (opts.fov) scale = p.rect.w / opts.fov;
      else if (opts.fovUm && mpp) scale = p.rect.w / (opts.fovUm / mpp);
      const t = { cx, cy, scale: clamp(scale, this._minScale(p), this._maxScale(p)) };
      if (opts.animate === false || !p.view) { p.view = { ...t }; p.target = null; }
      else p.target = t;
      p.vel = null;
    }
    this.requestRender();
  }

  /** Centre on a bounding box (x0, y0, x1, y1) in level-0 px, with padding. */
  goToBounds(pane, x0, y0, x1, y1, opts = {}) {
    for (const p of this._panes(pane)) {
      const pad = opts.padding ?? 0.1;
      const w = Math.max(1, x1 - x0) * (1 + 2 * pad), h = Math.max(1, y1 - y0) * (1 + 2 * pad);
      const zoom = Math.min(p.rect.w / w, p.rect.h / h);
      this.goTo(p.index, (x0 + x1) / 2, (y0 + y1) / 2, { ...opts, zoom });
    }
  }

  fit(pane = 'all', animate = true) {
    for (const p of this._panes(pane)) {
      const info = p.slide?.info;
      if (!info) continue;
      this.goTo(p.index, info.width / 2, info.height / 2, { zoom: this._fitScale(p), animate });
    }
  }

  getView(pane) {
    const p = this._pane(pane);
    if (!p?.view) return null;
    return this._viewDetail(p);
  }

  getViews() { return this.panes.map((p) => (p.view ? this._viewDetail(p) : null)); }

  /** Channel display: {visible, color, min, max, gamma}. */
  setChannel(pane, index, opts) {
    for (const p of this._panes(pane)) {
      const c = p.channels?.[index];
      if (!c) continue;
      if (opts.visible != null) c.visible = !!opts.visible;
      if (opts.color) c.color = hexToRgb(opts.color);
      if (opts.min != null) c.lo = +opts.min;
      if (opts.max != null) c.hi = +opts.max;
      if (opts.gamma != null) c.gamma = +opts.gamma;
    }
    this.requestRender();
  }

  /** Brightfield display: {brightness, contrast, gamma, invert}. */
  setAdjust(pane, opts) {
    for (const p of this._panes(pane)) Object.assign(p.adjust, opts);
    this.requestRender();
  }

  setLayerVisible(pane, layer, visible) {
    for (const p of this._panes(pane)) p.layerVis.set(layer, !!visible);
    this.requestRender();
    this._panelRefresh();
  }

  setLayerStyle(pane, layer, style) {
    for (const p of this._panes(pane)) {
      const L = p.slide?.layers.get(layer);
      if (L) Object.assign(L.meta.style, style);
    }
    this.requestRender();
  }

  setClassVisible(pane, layer, cls, visible) {
    for (const p of this._panes(pane)) p.slide?.layers.get(layer)?.setClassVisible(cls, visible);
    this.requestRender();
  }

  /** Re-fetch annotation layers (call after changing them on the server). slide = id, URL, or omit for all. */
  refreshAnnotations(slide) {
    for (const s of this.slides.values()) {
      if (!slide || s.url.endsWith('/slides/' + encodeURIComponent(slide)) || s.url === slide) s.refreshAnnotations();
    }
  }

  /** 'pan' | 'point' | 'rect' | 'ellipse' | 'polygon' | 'line' | 'freehand' */
  setTool(tool) {
    this.tool = tool || 'pan';
    this._drawing = null;
    this.panes.forEach((p) => p.el?.classList.toggle('tool', this.tool !== 'pan'));
    this._drawOverlay();
  }

  select(pane, layer, index) {
    const p = this._pane(pane);
    if (!p) return;
    p.selection = index == null ? null : { layer, f: index };
    this.requestRender();
  }

  stats() {
    return { textures: this.tex.size, textureMB: +(this.texBytes / 2 ** 20).toFixed(1), annotationMB: +(this.gpuBytes / 2 ** 20).toFixed(1),
      inflight: this.inflight.size, queuedUploads: this.uploads.length, frame: this.frame, budgetMB: this.budget / 2 ** 20 };
  }

  setTextureBudget(mb) { this.budget = mb * 2 ** 20; this._evict(); }

  // ---------------------------------------------------------------- internals: panes & layout
  _pane(i) {
    if (i == null || i === 'active') return this.panes[this.active];
    return typeof i === 'number' ? this.panes[i] : null;
  }
  _panes(i) {
    if (i === 'all' || i === '*') return this.panes;
    if (Array.isArray(i)) return i.map((k) => this.panes[k]).filter(Boolean);
    const p = this._pane(i);
    return p ? [p] : [];
  }

  _buildPanes() {
    if (!this.$panes) return;
    this.$panes.innerHTML = '';
    for (const p of this.panes) {
      const el = document.createElement('div');
      el.className = 'pane' + (p.index === this.active && this.panes.length > 1 ? ' active' : '') + (this.tool !== 'pan' ? ' tool' : '');
      el.innerHTML = `<div class="label"></div><div class="scale" style="display:none"><span></span><i></i></div><div class="coords"></div><div class="msg"></div>`
        + (this.showControls ? `<button class="btn" title="Display settings">${SVG_GEAR}</button>` : '');
      p.el = el;
      p.$label = el.querySelector('.label');
      p.$scale = el.querySelector('.scale');
      p.$coords = el.querySelector('.coords');
      p.$msg = el.querySelector('.msg');
      const btn = el.querySelector('.btn');
      if (btn) {
        btn.addEventListener('pointerdown', (e) => e.stopPropagation());
        btn.addEventListener('click', (e) => { e.stopPropagation(); this.setActive(p.index); this._togglePanel(p); });
      }
      this.$panes.appendChild(el);
      this._updateLabel(p);
    }
    this._applyLayout();
  }

  _applyLayout() {
    const n = this.panes.length;
    let rows, cols;
    const m = /^(\d+)\s*x\s*(\d+)$/i.exec(this.layoutSpec || '');
    if (m) { rows = +m[1]; cols = +m[2]; }
    else {
      const w = this.clientWidth || 4, h = this.clientHeight || 3;
      cols = Math.ceil(Math.sqrt(n * (w / h) / (4 / 3)));
      cols = clamp(cols, 1, n);
      rows = Math.ceil(n / cols);
      while (cols > 1 && Math.ceil(n / (cols - 1)) === rows) cols--;
    }
    this.$panes.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    this.$panes.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    this._measure();
  }

  _measure() {
    if (!this.gl) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.dpr = dpr;
    const W = this.$root.clientWidth, H = this.$root.clientHeight;
    const cw = Math.max(1, Math.round(W * dpr)), ch = Math.max(1, Math.round(H * dpr));
    if (this.$canvas.width !== cw || this.$canvas.height !== ch) {
      this.$canvas.width = this.$ov.width = cw;
      this.$canvas.height = this.$ov.height = ch;
    }
    const root = this.$root.getBoundingClientRect();
    for (const p of this.panes) {
      if (!p.el) continue;
      const r = p.el.getBoundingClientRect();
      const old = p.rect;
      p.rect = { x: r.left - root.left, y: r.top - root.top, w: r.width, h: r.height };
      if (p.view && old.w > 0 && (old.w !== r.width || old.h !== r.height) && p.fitted) {
        p.view.scale = this._fitScale(p);
        p.target = null;
      }
      if (p.slide?.info && !p.view && p.rect.w > 0) this._initPane(p);
    }
    this.requestRender();
  }

  _updateLabel(p) {
    if (!p.$label) return;
    const s = p.slide;
    p.$label.textContent = s ? (s.info?.name || s.info?.id || '') : '';
    p.$label.style.display = s && p.$label.textContent ? '' : 'none';
    p.$msg.textContent = !s ? '' : s.error ? `⚠ ${s.error}` : s.info ? '' : 'loading…';
  }

  _gcSlides() {
    const used = new Set(this.panes.map((p) => p.slide).filter(Boolean));
    for (const [url, s] of this.slides) {
      if (used.has(s)) continue;
      for (const L of s.layers.values()) L.destroy();
      for (const [k, t] of this.tex) if (k.startsWith(url + '|')) { this.gl.deleteTexture(t.tex); this.texBytes -= t.bytes; this.tex.delete(k); }
      this.slides.delete(url);
    }
  }

  _slideReady(s) {
    for (const p of this.panes) if (p.slide === s) { this._updateLabel(p); if (s.info && p.rect.w > 0) this._initPane(p); }
    this._panelRefresh();
    this.requestRender();
    if (s.info) this._emit('slideload', { url: s.url, info: s.info, panes: this.panes.filter((p) => p.slide === s).map((p) => p.index) });
  }

  _initPane(p) {
    const info = p.slide.info;
    if (!p.channels && info.mode === 'channels') {
      p.channels = info.channels.map((c, i) => ({
        name: c.name, color: hexToRgb(c.color), visible: c.visible !== false && i < MAX_CH,
        lo: c.window ? c.window[0] : 0, hi: c.window ? c.window[1] : (info.dtype === 'uint16' ? 65535 : 255), gamma: 1,
      }));
    }
    const o = p.pending || {};
    if (o.channels) o.channels.forEach((c, i) => c && this.setChannel(p.index, i, c));
    if (o.layers) for (const [k, v] of Object.entries(o.layers)) p.layerVis.set(k, !!v);
    if (o.adjust) Object.assign(p.adjust, o.adjust);
    p.pending = null;
    if (!p.view && p.rect.w > 0) {
      const z = this._fitScale(p);
      p.view = { cx: info.width / 2, cy: info.height / 2, scale: z };
      p.fitted = true;
      if (o.view) this.goTo(p.index, o.view.x, o.view.y, { ...o.view, animate: false });
      if (p.pendingGoto) { this.goTo(p.index, ...p.pendingGoto.slice(0, 2), { ...p.pendingGoto[2], animate: false }); p.pendingGoto = null; }
    }
  }

  _fitScale(p) {
    const i = p.slide?.info;
    if (!i || !p.rect.w) return 1;
    return Math.min(p.rect.w / i.width, p.rect.h / i.height) * 0.96;
  }
  _minScale(p) { return this._fitScale(p) * 0.25; }
  _maxScale(p) {
    const i = p.slide?.info;
    return i?.mpp ? Math.max(4, i.mpp[0] / 0.03) : 64; // allow ~0.03 µm per screen px
  }

  _viewDetail(p) {
    const v = p.view, info = p.slide?.info;
    const hw = p.rect.w / 2 / v.scale, hh = p.rect.h / 2 / v.scale;
    const d = { pane: p.index, slide: info?.id ?? null, url: p.slide?.url ?? null, x: v.cx, y: v.cy, zoom: v.scale,
      bounds: [v.cx - hw, v.cy - hh, v.cx + hw, v.cy + hh] };
    if (info?.mpp) {
      d.mpp = info.mpp[0] / v.scale;
      d.magnification = 10 / d.mpp;
      d.x_um = v.cx * info.mpp[0];
      d.y_um = v.cy * info.mpp[1];
    }
    return d;
  }

  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true })); }

  // ---------------------------------------------------------------- input
  _paneAt(clientX, clientY) {
    const root = this.$root.getBoundingClientRect();
    const x = clientX - root.left, y = clientY - root.top;
    for (const p of this.panes) {
      const r = p.rect;
      if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return { p, sx: x - r.x, sy: y - r.y };
    }
    return null;
  }
  _toImage(p, sx, sy) {
    const v = p.view;
    return [v.cx + (sx - p.rect.w / 2) / v.scale, v.cy + (sy - p.rect.h / 2) / v.scale];
  }

  _bindInput() {
    const root = this.$root;
    const ptrs = new Map();
    let drag = null;

    const down = (e) => {
      if (e.target.closest && (e.target.closest('.panel') || e.target.closest('.btn'))) return;
      const hit = this._paneAt(e.clientX, e.clientY);
      if (!hit || !hit.p.view) return;
      root.focus({ preventScroll: true });
      if (hit.p.index !== this.active) this.setActive(hit.p.index);
      root.setPointerCapture(e.pointerId);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.tool !== 'pan' && ptrs.size === 1 && e.button === 0) { this._drawDown(hit, e); return; }
      const p = hit.p;
      p.target = null;
      p.vel = null;
      drag = { p, moved: 0, t: now(), hist: [], x0: e.clientX, y0: e.clientY };
      p.el.classList.add('drag');
      if (ptrs.size === 2) drag.pinch = this._pinchState(ptrs);
    };
    const move = (e) => {
      const hit = this._paneAt(e.clientX, e.clientY);
      if (hit?.p.view) {
        const [ix, iy] = this._toImage(hit.p, hit.sx, hit.sy);
        const info = hit.p.slide.info;
        hit.p.$coords.textContent = info?.mpp ? `${(ix * info.mpp[0]).toFixed(1)}, ${(iy * info.mpp[1]).toFixed(1)} µm`
          : `${ix.toFixed(0)}, ${iy.toFixed(0)} px`;
        if (this._drawing && this._drawing.p === hit.p) this._drawMove(hit, e);
      }
      if (!ptrs.has(e.pointerId)) return;
      const last = ptrs.get(e.pointerId);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (!drag) return;
      const p = drag.p;
      if (ptrs.size >= 2 && drag.pinch) {
        const st = this._pinchState(ptrs);
        const f = st.d / drag.pinch.d;
        this._zoomAt(p, st.cx, st.cy, f, false);
        p.view.cx -= (st.cx - drag.pinch.cx) / p.view.scale;
        p.view.cy -= (st.cy - drag.pinch.cy) / p.view.scale;
        drag.pinch = st;
        drag.moved += 10;
      } else {
        const dx = e.clientX - last.x, dy = e.clientY - last.y;
        p.view.cx -= dx / p.view.scale;
        p.view.cy -= dy / p.view.scale;
        drag.moved += Math.abs(dx) + Math.abs(dy);
        const t = now();
        drag.hist.push({ t, dx: -dx / p.view.scale, dy: -dy / p.view.scale });
        while (drag.hist.length && t - drag.hist[0].t > 80) drag.hist.shift();
        this._syncFrom(p, -dx / p.view.scale, -dy / p.view.scale, 1);
      }
      this.requestRender();
    };
    const up = (e) => {
      ptrs.delete(e.pointerId);
      if (this._drawing && this.tool !== 'pan') { this._drawUp(e); return; }
      if (!drag) return;
      const p = drag.p;
      if (ptrs.size === 1) { drag.pinch = null; return; }
      if (ptrs.size) return;
      p.el.classList.remove('drag');
      if (drag.moved < 4 && now() - drag.t < 400) this._click(p, e);
      else if (drag.hist.length > 1) {
        const dt = Math.max(16, drag.hist[drag.hist.length - 1].t - drag.hist[0].t);
        const vx = drag.hist.reduce((a, h) => a + h.dx, 0) / dt, vy = drag.hist.reduce((a, h) => a + h.dy, 0) / dt;
        if (Math.hypot(vx, vy) * p.view.scale > 0.15) p.vel = { vx, vy };
      }
      drag = null;
      this.requestRender();
    };
    root.addEventListener('pointerdown', down);
    root.addEventListener('pointermove', move);
    root.addEventListener('pointerup', up);
    root.addEventListener('pointercancel', up);
    root.addEventListener('pointerleave', () => this.panes.forEach((p) => p.$coords && (p.$coords.textContent = '')));
    root.addEventListener('wheel', (e) => {
      const hit = this._paneAt(e.clientX, e.clientY);
      if (!hit || !hit.p.view) return;
      e.preventDefault();
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      const f = Math.exp(-clamp(dy, -300, 300) * (e.ctrlKey ? 0.01 : 0.0022));
      this._zoomAt(hit.p, hit.sx, hit.sy, f, true);
    }, { passive: false });
    root.addEventListener('dblclick', (e) => {
      if (this.tool !== 'pan') { this._drawFinish(); return; }
      const hit = this._paneAt(e.clientX, e.clientY);
      if (hit?.p.view) this._zoomAt(hit.p, hit.sx, hit.sy, e.shiftKey ? 0.5 : 2, true);
    });
    root.addEventListener('keydown', (e) => {
      const p = this.panes[this.active];
      if (!p?.view) return;
      const step = 120 / p.view.scale;
      const k = e.key;
      if (k === 'Escape') { this._drawing = null; this._drawOverlay(); if (this.tool !== 'pan') return; }
      if (k === 'Enter') { this._drawFinish(); return; }
      if (k === 'ArrowLeft') this._panBy(p, -step, 0);
      else if (k === 'ArrowRight') this._panBy(p, step, 0);
      else if (k === 'ArrowUp') this._panBy(p, 0, -step);
      else if (k === 'ArrowDown') this._panBy(p, 0, step);
      else if (k === '+' || k === '=') this._zoomAt(p, p.rect.w / 2, p.rect.h / 2, 1.6, true);
      else if (k === '-' || k === '_') this._zoomAt(p, p.rect.w / 2, p.rect.h / 2, 1 / 1.6, true);
      else if (k === '0') this.fit(this.active);
      else if (k === 'Tab' && this.panes.length > 1) this.setActive((this.active + (e.shiftKey ? this.panes.length - 1 : 1)) % this.panes.length);
      else return;
      e.preventDefault();
    });
  }

  _pinchState(ptrs) {
    const [a, b] = [...ptrs.values()];
    const root = this.$root.getBoundingClientRect();
    const p = this.panes[this.active];
    return { d: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), cx: (a.x + b.x) / 2 - root.left - p.rect.x, cy: (a.y + b.y) / 2 - root.top - p.rect.y };
  }

  _panBy(p, dx, dy) {
    const base = p.target || p.view;
    p.target = { ...base, cx: base.cx + dx, cy: base.cy + dy };
    this._syncFrom(p, dx, dy, 1, true);
    this.requestRender();
  }

  _zoomAt(p, sx, sy, f, animate) {
    const base = animate ? (p.target || p.view) : p.view;
    const s2 = clamp(base.scale * f, this._minScale(p), this._maxScale(p));
    const fx = sx - p.rect.w / 2, fy = sy - p.rect.h / 2;
    const ix = base.cx + fx / base.scale, iy = base.cy + fy / base.scale;
    const t = { cx: ix - fx / s2, cy: iy - fy / s2, scale: s2 };
    const realF = s2 / base.scale;
    if (animate) p.target = t;
    else p.view = t;
    p.fitted = false;
    p.vel = null;
    this._syncFrom(p, t.cx - base.cx, t.cy - base.cy, realF, animate);
    this.requestRender();
  }

  // Linked navigation: apply the same physical displacement / zoom factor to the other panes.
  _syncFrom(src, dx, dy, f, animate = false) {
    if (this.syncMode === 'none' || this.panes.length < 2) return;
    const sm = src.slide?.info?.mpp?.[0];
    for (const p of this.panes) {
      if (p === src || !p.view) continue;
      const m = p.slide?.info?.mpp?.[0];
      const k = sm && m ? sm / m : src.view.scale / p.view.scale;
      const base = animate ? (p.target || p.view) : p.view;
      const t = { ...base };
      if (this.syncMode !== 'zoom') { t.cx += dx * k; t.cy += dy * k; }
      if (this.syncMode !== 'pan') t.scale = clamp(base.scale * f, this._minScale(p), this._maxScale(p));
      if (animate) p.target = t; else { p.view = t; p.target = null; }
    }
  }

  async _click(p, e) {
    const hit = this._paneAt(e.clientX, e.clientY);
    if (!hit || hit.p !== p) return;
    const [x, y] = this._toImage(p, hit.sx, hit.sy);
    const detail = { pane: p.index, slide: p.slide?.info?.id, x, y, button: e.button };
    if (p.slide?.info?.mpp) { detail.x_um = x * p.slide.info.mpp[0]; detail.y_um = y * p.slide.info.mpp[1]; }
    this._emit('paneclick', detail);
    const tol = 5 / p.view.scale;
    let found = null;
    for (const L of p.slide.layers.values()) {
      if (!p.layerVisible(L)) continue;
      const f = L.hit(x, y, tol);
      if (f >= 0 && (!found || L.size[f] < found.L.size[found.f])) found = { L, f };
    }
    p.selection = found ? { layer: found.L.name, f: found.f } : null;
    this.requestRender();
    if (!found) return;
    const { L, f } = found;
    const idx = L.orig[f];
    const d = { ...detail, layer: L.name, index: idx, class: L.classes[L.cls[f]]?.name,
      centroid: [L.cen[2 * f], L.cen[2 * f + 1]], size: L.size[f] };
    try {
      const r = await fetch(`${p.slide.url}/annotations/feature/${idx}?layer=${encodeURIComponent(L.name)}`, { credentials: 'same-origin' });
      if (r.ok) d.properties = (await r.json()).properties;
    } catch (_) { /* properties are optional */ }
    this._emit('objectclick', d);
  }

  // ---------------------------------------------------------------- drawing tools
  _drawDown(hit, e) {
    const p = hit.p;
    const pt = this._toImage(p, hit.sx, hit.sy);
    const t = this.tool;
    if (t === 'point') { this._drawEmit(p, { type: 'Point', coordinates: pt }); return; }
    if (t === 'polygon' || t === 'line') {
      if (!this._drawing || this._drawing.p !== p) this._drawing = { p, pts: [pt], cur: pt, kind: t };
      else this._drawing.pts.push(pt);
    } else this._drawing = { p, pts: [pt], cur: pt, kind: t, dragging: true };
    this._drawOverlay();
  }
  _drawMove(hit) {
    const d = this._drawing;
    d.cur = this._toImage(hit.p, hit.sx, hit.sy);
    if (d.kind === 'freehand' && d.dragging) d.pts.push(d.cur);
    this._drawOverlay();
  }
  _drawUp() {
    const d = this._drawing;
    if (!d || !d.dragging) return;
    const [a, b] = [d.pts[0], d.cur];
    let g = null;
    if (d.kind === 'rect' && Math.abs(a[0] - b[0]) > 0 && Math.abs(a[1] - b[1]) > 0) {
      g = { type: 'Polygon', coordinates: [[a, [b[0], a[1]], b, [a[0], b[1]], a]] };
    } else if (d.kind === 'ellipse') {
      const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2, rx = Math.abs(b[0] - a[0]) / 2, ry = Math.abs(b[1] - a[1]) / 2;
      if (rx > 0 && ry > 0) {
        const ring = [];
        for (let i = 0; i <= 64; i++) { const t = (i / 64) * 2 * Math.PI; ring.push([cx + rx * Math.cos(t), cy + ry * Math.sin(t)]); }
        g = { type: 'Polygon', coordinates: [ring] };
      }
    } else if (d.kind === 'freehand' && d.pts.length > 2) {
      g = { type: 'Polygon', coordinates: [[...d.pts, d.pts[0]]] };
    }
    this._drawing = null;
    if (g) this._drawEmit(d.p, g);
    this._drawOverlay();
  }
  _drawFinish() {
    const d = this._drawing;
    if (!d || (d.kind !== 'polygon' && d.kind !== 'line')) return;
    const pts = d.pts.filter((q, i) => i === 0 || q[0] !== d.pts[i - 1][0] || q[1] !== d.pts[i - 1][1]);
    this._drawing = null;
    if (d.kind === 'polygon' && pts.length >= 3) this._drawEmit(d.p, { type: 'Polygon', coordinates: [[...pts, pts[0]]] });
    if (d.kind === 'line' && pts.length >= 2) this._drawEmit(d.p, { type: 'LineString', coordinates: pts });
    this._drawOverlay();
  }
  async _drawEmit(p, geometry) {
    this._emit('draw', { pane: p.index, slide: p.slide?.info?.id, url: p.slide?.url, tool: this.tool, geometry });
    if (this.drawLayer && p.slide) {
      try {
        await fetch(`${p.slide.url}/annotations/add?layer=${encodeURIComponent(this.drawLayer)}`, {
          method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: 'Feature', geometry, properties: { classification: { name: this.drawLayer } } }),
        });
        await p.slide.refreshAnnotations();
      } catch (_) { /* surfaced through the draw event */ }
    }
  }

  // ---------------------------------------------------------------- frame loop
  requestRender() {
    if (!this._raf && this.gl && this.isConnected) this._raf = requestAnimationFrame((t) => this._frame(t));
  }

  _frame() {
    this._raf = 0;
    if (this._lost) return;
    const t = now();
    const dt = this._lastT ? Math.min(64, t - this._lastT) : 16;
    this._lastT = t;
    let again = false;
    for (const p of this.panes) if (this._animate(p, dt)) again = true;
    if (this._processUploads()) again = true;
    this.frame++;
    this._draw();
    this._schedule();
    this._evict();
    this._emitViews(t);
    if (again) this.requestRender();
    else this._lastT = 0;
  }

  _animate(p, dt) {
    if (!p.view) return false;
    let moving = false;
    if (p.target) {
      const k = 1 - Math.pow(1 - 0.22, dt / 16);
      const v = p.view, g = p.target;
      const ls = Math.log(v.scale), lg = Math.log(g.scale);
      v.scale = Math.exp(ls + (lg - ls) * k);
      v.cx += (g.cx - v.cx) * k;
      v.cy += (g.cy - v.cy) * k;
      if (Math.abs(lg - ls) < 0.002 && Math.hypot(g.cx - v.cx, g.cy - v.cy) * v.scale < 0.3) { p.view = { ...g }; p.target = null; }
      moving = true;
    }
    if (p.vel) {
      const decay = Math.pow(0.94, dt / 16);
      const dx = p.vel.vx * dt, dy = p.vel.vy * dt;
      p.view.cx += dx;
      p.view.cy += dy;
      this._syncFrom(p, dx, dy, 1);
      p.vel.vx *= decay;
      p.vel.vy *= decay;
      if (Math.hypot(p.vel.vx, p.vel.vy) * p.view.scale < 0.02) p.vel = null;
      moving = true;
    }
    // keep the slide reachable
    const i = p.slide?.info;
    if (i) {
      p.view.cx = clamp(p.view.cx, -i.width * 0.25, i.width * 1.25);
      p.view.cy = clamp(p.view.cy, -i.height * 0.25, i.height * 1.25);
    }
    return moving;
  }

  _emitViews(t) {
    if (t - (this._lastEmit || 0) < 120) { if (!this._emitPending) { this._emitPending = true; setTimeout(() => { this._emitPending = false; this._emitViews(now()); }, 130); } return; }
    this._lastEmit = t;
    for (const p of this.panes) {
      if (!p.view) continue;
      const key = `${p.view.cx.toFixed(1)}|${p.view.cy.toFixed(1)}|${p.view.scale.toPrecision(5)}|${p.rect.w}|${p.rect.h}`;
      if (key === p.lastEmit) continue;
      p.lastEmit = key;
      this._emit('viewchange', this._viewDetail(p));
      this._updateScaleBar(p);
    }
  }

  _updateScaleBar(p) {
    const i = p.slide?.info;
    if (!p.$scale) return;
    if (!i?.mpp || !p.view) { p.$scale.style.display = 'none'; return; }
    const umPerPx = i.mpp[0] / p.view.scale;
    const target = 110 * umPerPx;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const nice = [1, 2, 5, 10].map((m) => m * pow).filter((v) => v <= target).pop() || pow;
    p.$scale.style.display = '';
    p.$scale.querySelector('i').style.width = `${nice / umPerPx}px`;
    p.$scale.querySelector('span').textContent = nice >= 1000 ? `${nice / 1000} mm` : nice >= 1 ? `${nice} µm` : `${nice * 1000} nm`;
  }

  // ---------------------------------------------------------------- tiles: geometry
  _level(info, scaleDev) {
    // finest level whose pixels are not smaller than ~1 device pixel (× quality bias)
    const want = (1 / scaleDev) * this.quality;
    let L = 0;
    for (let i = 0; i < info.levels.length; i++) if (info.levels[i].downsample <= want * 1.0001) L = i;
    return L;
  }

  _visRect(p, margin = 0) {
    const v = p.view;
    const hw = (p.rect.w / 2) / v.scale + margin, hh = (p.rect.h / 2) / v.scale + margin;
    return [v.cx - hw, v.cy - hh, v.cx + hw, v.cy + hh];
  }

  _tileRange(info, L, r) {
    const lv = info.levels[L], ts = info.tile_size;
    const nx = Math.ceil(lv.width / ts), ny = Math.ceil(lv.height / ts);
    const x0 = clamp(Math.floor(r[0] / lv.sx / ts), 0, nx - 1), x1 = clamp(Math.floor(r[2] / lv.sx / ts), 0, nx - 1);
    const y0 = clamp(Math.floor(r[1] / lv.sy / ts), 0, ny - 1), y1 = clamp(Math.floor(r[3] / lv.sy / ts), 0, ny - 1);
    if (r[2] < 0 || r[3] < 0 || r[0] > info.width || r[1] > info.height) return null;
    return { x0, x1, y0, y1 };
  }

  _key(s, L, x, y, c) { return `${s.url}|${L}|${x}|${y}|${c ?? 'rgb'}`; }

  _visibleChannels(p) {
    if (p.slide.info.mode !== 'channels') return [null];
    const out = [];
    p.channels.forEach((c, i) => { if (c.visible && out.length < MAX_CH) out.push(i); });
    return out;
  }

  _ready(p, L, x, y, chans) {
    for (const c of chans) if (!this.tex.has(this._key(p.slide, L, x, y, c))) return false;
    return true;
  }

  // ---------------------------------------------------------------- drawing
  _draw() {
    const gl = this.gl;
    const H = this.$canvas.height, dpr = this.dpr || 1;
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, this.$canvas.width, this.$canvas.height);
    gl.clearColor(0.043, 0.051, 0.063, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.SCISSOR_TEST);
    for (const p of this.panes) {
      const r = p.rect;
      if (r.w <= 0 || r.h <= 0) continue;
      const x = Math.round(r.x * dpr), y = Math.round(r.y * dpr), w = Math.round(r.w * dpr), h = Math.round(r.h * dpr);
      p.dev = { x, y, w, h };
      gl.viewport(x, H - y - h, w, h);
      gl.scissor(x, H - y - h, w, h);
      const info = p.slide?.info;
      const bg = info ? (info.mode === 'channels' ? [0, 0, 0] : info.bg) : [0.06, 0.07, 0.08];
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (!info || !p.view) continue;
      this._drawTiles(p);
      this._drawAnnotations(p);
    }
    gl.disable(gl.SCISSOR_TEST);
    this._drawOverlay();
  }

  _drawTiles(p) {
    const gl = this.gl, s = p.slide, info = s.info;
    const scaleDev = p.view.scale * this.dpr;
    const target = this._level(info, scaleDev);
    p.level = target;
    const vis = this._visRect(p);
    const chans = this._visibleChannels(p);
    const isCh = info.mode === 'channels';
    if (isCh && !chans.length) return;
    // find the finest level ≥ target that fully covers the view; draw from there down to target (painter's)
    const levels = [];
    for (let L = target; L < info.levels.length; L++) {
      const rg = this._tileRange(info, L, vis);
      if (!rg) break;
      levels.push([L, rg]);
      let full = true;
      for (let ty = rg.y0; ty <= rg.y1 && full; ty++) for (let tx = rg.x0; tx <= rg.x1; tx++) if (!this._ready(p, L, tx, ty, chans)) { full = false; break; }
      if (full) break;
    }
    const prog = isCh ? this.prog.ch : this.prog.rgb;
    gl.useProgram(prog.p);
    gl.bindVertexArray(this.emptyVao);
    gl.disable(gl.BLEND);
    gl.uniform3f(prog.u.u_view, p.view.cx, p.view.cy, scaleDev);
    gl.uniform2f(prog.u.u_vp, p.dev.w, p.dev.h);
    if (isCh) {
      gl.uniform1i(prog.u.u_n, chans.length);
      const col = new Float32Array(MAX_CH * 3), win = new Float32Array(MAX_CH * 4);
      chans.forEach((c, i) => { const ch = p.channels[c]; col.set(ch.color, i * 3); win.set([ch.lo, ch.hi, ch.gamma, 0], i * 4); });
      gl.uniform3fv(prog.u.u_col, col);
      gl.uniform4fv(prog.u.u_win, win);
      gl.uniform1i(prog.u.u_nearest, p.nearest || scaleDev > 6 ? 1 : 0);
      gl.uniform3f(prog.u.u_bg, 0, 0, 0);
      for (let i = 0; i < MAX_CH; i++) gl.uniform1i(prog.u['u_c' + i], i);
    } else {
      const a = p.adjust;
      gl.uniform4f(prog.u.u_adj, a.brightness, a.contrast, a.gamma, a.invert ? 1 : 0);
      gl.uniform1i(prog.u.u_tex, 0);
    }
    for (let k = levels.length - 1; k >= 0; k--) {
      const [L, rg] = levels[k];
      const lv = info.levels[L], ts = info.tile_size;
      for (let ty = rg.y0; ty <= rg.y1; ty++) {
        for (let tx = rg.x0; tx <= rg.x1; tx++) {
          if (!this._ready(p, L, tx, ty, chans)) continue;
          const tw = Math.min(ts, lv.width - tx * ts), th = Math.min(ts, lv.height - ty * ts);
          chans.forEach((c, i) => {
            const t = this.tex.get(this._key(s, L, tx, ty, c));
            t.last = this.frame;
            gl.activeTexture(gl.TEXTURE0 + i);
            gl.bindTexture(gl.TEXTURE_2D, t.tex);
          });
          gl.uniform4f(prog.u.u_rect, tx * ts * lv.sx, ty * ts * lv.sy, tw * lv.sx, th * lv.sy);
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
      }
    }
    gl.activeTexture(gl.TEXTURE0);
  }

  _drawAnnotations(p) {
    const gl = this.gl, s = p.slide;
    if (!s.layers.size) return;
    const scaleDev = p.view.scale * this.dpr;
    const vis = this._visRect(p, 2 / p.view.scale);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    for (const L of s.layers.values()) {
      if (!p.layerVisible(L)) continue;
      if (L.state !== 'ready') continue; // overview requested by the scheduler
      const st = L.style;
      const B = L.buckets;
      const ptsRanges = [], kindRanges = [], detail = [];
      const minPx = st.detail_min_px ?? 6;
      for (let b = 0; b < L.nb; b++) {
        const o = b * 8;
        if (B[o] > vis[2] || B[o + 2] < vis[0] || B[o + 1] > vis[3] || B[o + 3] < vis[1]) continue; // frustum cull
        const wantDetail = B[o + 6] * p.view.scale >= minPx;
        const d = wantDetail ? L.detail.get(b) : null;
        if (d) { d.last = this.frame; detail.push(d); kindRanges.push([B[o + 4], B[o + 5]]); }
        else ptsRanges.push([B[o + 4], B[o + 5]]);
      }
      const alpha = st.opacity ?? 1;
      // dots (zoomed-out objects, and point features everywhere)
      if (ptsRanges.length || kindRanges.length) {
        const pr = this.prog.pts;
        gl.useProgram(pr.p);
        gl.bindVertexArray(L.vao);
        gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
        gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
        gl.uniform2f(pr.u.u_ps, 1.5 * this.dpr, (st.point_size ?? 4) * this.dpr);
        gl.uniform1f(pr.u.u_alpha, alpha * 0.9);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
        const run = (ranges, only) => {
          gl.uniform1i(pr.u.u_onlyPoints, only);
          ranges.sort((a, b) => a[0] - b[0]);
          let s0 = -1, e0 = -1;
          for (const [a, n] of ranges) {
            if (a === e0) { e0 = a + n; continue; }
            if (s0 >= 0) gl.drawArrays(gl.POINTS, s0, e0 - s0);
            s0 = a; e0 = a + n;
          }
          if (s0 >= 0) gl.drawArrays(gl.POINTS, s0, e0 - s0);
        };
        run(ptsRanges, 0);
        if (!detail.length) run(kindRanges, 1);
      }
      // fills
      const fo = (st.fill_opacity ?? 0) * alpha;
      if (fo > 0 && detail.length) {
        const pr = this.prog.tri;
        gl.useProgram(pr.p);
        gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
        gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
        gl.uniform1f(pr.u.u_alpha, fo);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
        for (const d of detail) if (d.nTri) { gl.bindVertexArray(d.vaoTri); gl.drawArrays(gl.TRIANGLES, 0, d.nTri); }
      }
      // outlines
      if (detail.length) {
        const pr = this.prog.seg;
        gl.useProgram(pr.p);
        gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
        gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
        gl.uniform1f(pr.u.u_w, Math.max(1, (st.line_width ?? 1.5) * this.dpr));
        gl.uniform1f(pr.u.u_alpha, alpha);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
        for (const d of detail) if (d.nSeg) { gl.bindVertexArray(d.vaoSeg); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, d.nSeg); }
      }
      // point features on top of detailed buckets
      if (kindRanges.length && detail.length) {
        const pr = this.prog.pts;
        gl.useProgram(pr.p);
        gl.bindVertexArray(L.vao);
        gl.uniform1i(pr.u.u_onlyPoints, 1);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex);
        for (const [a, n] of kindRanges) gl.drawArrays(gl.POINTS, a, n);
      }
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  _drawOverlay() {
    const c = this.ctx2d;
    if (!c) return;
    const dpr = this.dpr || 1;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, this.$ov.width, this.$ov.height);
    for (const p of this.panes) {
      if (!p.view) continue;
      const toS = (x, y) => [(p.rect.x + p.rect.w / 2 + (x - p.view.cx) * p.view.scale) * dpr, (p.rect.y + p.rect.h / 2 + (y - p.view.cy) * p.view.scale) * dpr];
      c.save();
      c.beginPath();
      c.rect(p.rect.x * dpr, p.rect.y * dpr, p.rect.w * dpr, p.rect.h * dpr);
      c.clip();
      if (p.selection) {
        const L = p.slide?.layers.get(p.selection.layer);
        if (L?.state === 'ready') {
          const f = p.selection.f;
          const [sx, sy] = toS(L.cen[2 * f], L.cen[2 * f + 1]);
          const r = Math.max(6 * dpr, (L.size[f] / 2) * p.view.scale * dpr * 1.15);
          c.lineWidth = 2 * dpr;
          c.strokeStyle = '#fff';
          c.setLineDash([5 * dpr, 4 * dpr]);
          c.beginPath(); c.arc(sx, sy, r, 0, Math.PI * 2); c.stroke();
          c.setLineDash([]);
        }
      }
      const d = this._drawing;
      if (d && d.p === p) {
        c.lineWidth = 2 * dpr;
        c.strokeStyle = '#ffeb3b';
        c.fillStyle = 'rgba(255,235,59,.12)';
        c.beginPath();
        const a = d.pts[0], b = d.cur;
        if (d.kind === 'rect') { const [x0, y0] = toS(...a), [x1, y1] = toS(...b); c.rect(x0, y0, x1 - x0, y1 - y0); }
        else if (d.kind === 'ellipse') {
          const [x0, y0] = toS(...a), [x1, y1] = toS(...b);
          c.ellipse((x0 + x1) / 2, (y0 + y1) / 2, Math.abs(x1 - x0) / 2, Math.abs(y1 - y0) / 2, 0, 0, Math.PI * 2);
        } else {
          d.pts.forEach((q, i) => { const [x, y] = toS(...q); i ? c.lineTo(x, y) : c.moveTo(x, y); });
          if (d.kind !== 'freehand') { const [x, y] = toS(...b); c.lineTo(x, y); }
        }
        if (d.kind !== 'line') c.fill();
        c.stroke();
      }
      c.restore();
    }
  }

  // ---------------------------------------------------------------- scheduling (what to load next)
  _schedule() {
    const want = new Map(); // key → {prio, url, kind, meta}
    const add = (key, prio, url, kind, meta) => {
      const w = want.get(key);
      if (!w || w.prio > prio) want.set(key, { prio, url, kind, meta });
    };
    for (const p of this.panes) {
      const s = p.slide, info = s?.info;
      if (!info || !p.view || p.rect.w <= 0) continue;
      const chans = this._visibleChannels(p);
      const scaleDev = p.view.scale * this.dpr;
      const target = this._level(info, scaleDev);
      const top = info.levels.length - 1;
      const vis = this._visRect(p);
      const cx = p.view.cx, cy = p.view.cy, diag = Math.hypot(vis[2] - vis[0], vis[3] - vis[1]) || 1;
      const queueLevel = (L, rect, base) => {
        const rg = this._tileRange(info, L, rect);
        if (!rg) return;
        const lv = info.levels[L], ts = info.tile_size;
        for (let ty = rg.y0; ty <= rg.y1; ty++) for (let tx = rg.x0; tx <= rg.x1; tx++) {
          const mx = (tx + 0.5) * ts * lv.sx, my = (ty + 0.5) * ts * lv.sy;
          const dist = Math.hypot(mx - cx, my - cy) / diag;
          for (const c of chans) {
            const key = this._key(s, L, tx, ty, c);
            if (this.tex.has(key)) { this.tex.get(key).last = this.frame; continue; }
            add(key, base + dist, s.tileUrl(L, tx, ty, c), c == null ? 'image' : 'raw', { s, L, tx, ty, c });
          }
        }
      };
      // 1. whole coarsest level (instant overview, fallback for everything)
      queueLevel(top, [0, 0, info.width, info.height], 0);
      // 2. parent of target (fast fallback), 3. target centre-out
      if (target + 1 < top) queueLevel(target + 1, vis, 0.5);
      queueLevel(target, vis, 1);
      // 4. margin ring + motion prediction
      const ts0 = info.tile_size * info.levels[target].downsample;
      const v = p.vel || (p.target ? { vx: (p.target.cx - cx) / 300, vy: (p.target.cy - cy) / 300 } : null);
      const m = [vis[0] - ts0, vis[1] - ts0, vis[2] + ts0, vis[3] + ts0];
      if (v) {
        const ax = v.vx * 600, ay = v.vy * 600;
        if (ax < 0) m[0] += ax; else m[2] += ax;
        if (ay < 0) m[1] += ay; else m[3] += ay;
      }
      queueLevel(target, m, 3);
      // zooming in: also the next finer level for the target view
      if (p.target && p.target.scale > p.view.scale * 1.2) {
        const tl = this._level(info, p.target.scale * this.dpr);
        if (tl < target) queueLevel(tl, this._visRect({ ...p, view: p.target }), 2);
      }
      // annotations: overview per visible layer, detail buckets in view
      for (const L of s.layers.values()) {
        if (!p.layerVisible(L)) continue;
        if (L.state === 'idle' || L.state === 'error-retry') {
          add(`ann|${s.url}|${L.name}|${L.meta.version}`, 0.2, s.annUrl('overview', L.name, L.meta.version), 'bin', { ann: L });
          continue;
        }
        if (L.state !== 'ready') continue;
        const B = L.buckets, minPx = L.style.detail_min_px ?? 6;
        const vr = this._visRect(p, ts0 * 0.5);
        for (let b = 0; b < L.nb; b++) {
          const o = b * 8;
          if (B[o] > vr[2] || B[o + 2] < vr[0] || B[o + 1] > vr[3] || B[o + 3] < vr[1]) continue;
          if (B[o + 6] * p.view.scale < minPx) continue;
          const d = L.detail.get(b);
          if (d) { d.last = this.frame; continue; }
          const dist = Math.hypot((B[o] + B[o + 2]) / 2 - cx, (B[o + 1] + B[o + 3]) / 2 - cy) / diag;
          add(`bkt|${s.url}|${L.name}|${L.meta.version}|${b}`, 1.2 + dist, s.annUrl('bucket/' + b, L.name, L.meta.version), 'bin', { ann: L, bucket: b });
        }
      }
    }
    // cancel what is no longer wanted
    for (const [k, r] of this.inflight) if (!want.has(k)) { r.abort(); this.inflight.delete(k); }
    const t = now();
    const todo = [];
    for (const [k, w] of want) {
      if (this.inflight.has(k)) continue;
      const f = this.failed.get(k);
      if (f && f > t) continue;
      todo.push([k, w]);
    }
    todo.sort((a, b) => a[1].prio - b[1].prio);
    for (const [k, w] of todo) {
      if (this.inflight.size >= this.maxInflight) break;
      this._start(k, w);
    }
    this._pendingWork = todo.length > 0;
  }

  _start(key, w) {
    const req = this.fetcher.load(w.url, w.kind);
    this.inflight.set(key, req);
    if (w.meta.ann && w.meta.bucket == null) w.meta.ann.state = 'loading';
    req.promise.then((res) => {
      if (this.inflight.get(key) !== req) { res.bmp?.close?.(); return; }
      this.inflight.delete(key);
      this.failed.delete(key);
      this.uploads.push({ key, res, meta: w.meta });
      this.requestRender();
    }, (err) => {
      if (this.inflight.get(key) === req) this.inflight.delete(key);
      if (w.meta.ann && w.meta.bucket == null && w.meta.ann.state === 'loading') w.meta.ann.state = 'error-retry';
      if (err.message !== 'abort') {
        const n = (this._failCount = (this._failCount || 0) + 1);
        this.failed.set(key, now() + Math.min(30000, 1000 * 2 ** Math.min(5, n % 6)));
        setTimeout(() => this.requestRender(), 1100);
      }
      this.requestRender();
    });
  }

  _processUploads() {
    const gl = this.gl;
    const deadline = now() + 6; // ms of GPU uploads per frame → no jank while panning
    let n = 0;
    while (this.uploads.length && (n < 2 || now() < deadline)) {
      const { key, res, meta } = this.uploads.shift();
      n++;
      if (meta.ann) {
        const L = meta.ann;
        if (L.state === 'dead') continue;
        try {
          if (meta.bucket == null) L._ingest(res.buf);
          else L.ingestBucket(meta.bucket, res.buf);
        } catch (e) { console.warn('wsiviz annotation', e); L.state = 'error'; }
        continue;
      }
      const { s, L, tx, ty, c } = meta;
      if (!this.slides.has(s.url) || !s.info) { res.bmp?.close?.(); continue; }
      const lv = s.info.levels[L], ts = s.info.tile_size;
      const w = Math.min(ts, lv.width - tx * ts), h = Math.min(ts, lv.height - ty * ts);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      let bytes;
      if (c == null) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, res.bmp);
        bytes = res.bmp.width * res.bmp.height * 4;
        res.bmp.close();
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      } else {
        const u16 = s.info.dtype === 'uint16';
        const data = u16 ? new Uint16Array(res.buf, 0, w * h) : new Uint8Array(res.buf, 0, w * h);
        gl.texImage2D(gl.TEXTURE_2D, 0, u16 ? gl.R16UI : gl.R8UI, w, h, 0, gl.RED_INTEGER, u16 ? gl.UNSIGNED_SHORT : gl.UNSIGNED_BYTE, data);
        bytes = w * h * (u16 ? 2 : 1);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const old = this.tex.get(key);
      if (old) { gl.deleteTexture(old.tex); this.texBytes -= old.bytes; }
      this.tex.set(key, { tex, bytes, last: this.frame });
      this.texBytes += bytes;
    }
    return this.uploads.length > 0;
  }

  _evict() {
    const gl = this.gl;
    if (this.texBytes > this.budget) {
      const cand = [...this.tex.entries()].filter(([, t]) => t.last < this.frame - 1);
      cand.sort((a, b) => a[1].last - b[1].last);
      // never evict the coarsest level of a slide on screen
      for (const [k, t] of cand) {
        if (this.texBytes <= this.budget * 0.85) break;
        gl.deleteTexture(t.tex);
        this.texBytes -= t.bytes;
        this.tex.delete(k);
      }
    }
    const annBudget = this.budget * 0.5;
    if (this.gpuBytes > annBudget) {
      const cand = [];
      for (const s of this.slides.values()) for (const L of s.layers.values()) for (const [b, d] of L.detail) if (d.last < this.frame - 1) cand.push([L, b, d.last]);
      cand.sort((a, b) => a[2] - b[2]);
      for (const [L, b] of cand) { if (this.gpuBytes <= annBudget * 0.85) break; L.dropBucket(b); }
    }
  }

  // ---------------------------------------------------------------- built-in display panel
  _togglePanel(p) {
    const open = p.el.querySelector('.panel');
    this.panes.forEach((q) => q.el?.querySelector('.panel')?.remove());
    if (open) return;
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.addEventListener('pointerdown', (e) => e.stopPropagation());
    panel.addEventListener('wheel', (e) => e.stopPropagation(), { passive: true });
    p.el.appendChild(panel);
    this._fillPanel(p, panel);
  }

  _panelRefresh() {
    for (const p of this.panes) { const el = p.el?.querySelector('.panel'); if (el) this._fillPanel(p, el); }
  }

  _fillPanel(p, panel) {
    const info = p.slide?.info;
    panel.innerHTML = '';
    if (!info) { panel.textContent = 'No slide'; return; }
    const h = (t) => { const e = document.createElement('h4'); e.textContent = t; panel.appendChild(e); };
    const row = (html) => { const e = document.createElement('div'); e.className = 'row'; e.innerHTML = html; panel.appendChild(e); return e; };
    const meta = row(`<label>${info.width}×${info.height}${info.mpp ? ` · ${info.mpp[0].toFixed(3)} µm/px` : ''}</label>`);
    meta.style.opacity = '.7';
    if (info.mode === 'channels') {
      h('Channels');
      const max = info.dtype === 'uint16' ? 65535 : 255;
      p.channels.forEach((c, i) => {
        const r = row(`<input type="checkbox" ${c.visible ? 'checked' : ''}><input type="color" value="${rgbToHex(c.color)}"><label title="${c.name}">${c.name}</label>`);
        r.children[0].onchange = (e) => this.setChannel(p.index, i, { visible: e.target.checked });
        r.children[1].oninput = (e) => this.setChannel(p.index, i, { color: e.target.value });
        const sliders = row(`<span class="sub">min</span><input type="range" min="0" max="${max}" value="${c.lo}"><span class="sub">max</span><input type="range" min="0" max="${max}" value="${c.hi}">`);
        sliders.children[1].oninput = (e) => this.setChannel(p.index, i, { min: +e.target.value });
        sliders.children[3].oninput = (e) => this.setChannel(p.index, i, { max: +e.target.value });
      });
    } else {
      h('Display');
      const a = p.adjust;
      const mk = (label, key, min, max, step) => {
        const r = row(`<label>${label}</label><input type="range" min="${min}" max="${max}" step="${step}" value="${a[key]}">`);
        r.children[1].oninput = (e) => this.setAdjust(p.index, { [key]: +e.target.value });
      };
      mk('Brightness', 'brightness', -0.5, 0.5, 0.01);
      mk('Contrast', 'contrast', 0.3, 3, 0.01);
      mk('Gamma', 'gamma', 0.3, 3, 0.01);
    }
    if (p.slide.layers.size) {
      h('Annotations');
      for (const L of p.slide.layers.values()) {
        const r = row(`<input type="checkbox" ${p.layerVisible(L) ? 'checked' : ''}><label title="${L.name}">${L.name} <span style="opacity:.6">(${L.meta.count.toLocaleString()})</span></label>`);
        r.children[0].onchange = (e) => this.setLayerVisible(p.index, L.name, e.target.checked);
        for (const c of (L.meta.classes || []).slice(0, 40)) {
          const cr = row(`<span class="sub"></span><input type="checkbox" ${L.hidden.has(c.name) ? '' : 'checked'}><input type="color" value="${c.color}"><label>${c.name}</label>`);
          cr.children[1].onchange = (e) => { L.setClassVisible(c.name, e.target.checked); this.requestRender(); };
          cr.children[2].oninput = (e) => { L.setClassColor(c.name, e.target.value); this.requestRender(); };
        }
      }
    }
  }
}

/** Single-slide convenience element: <wsi-viewer api="/wsiviz" slide="id"></wsi-viewer> */
export class WsiViewer extends WsiGrid {
  static get observedAttributes() { return [...super.observedAttributes, 'slide', 'src']; }
  attributeChangedCallback(name, old, val) {
    if (name === 'slide' || name === 'src') { if (this.gl) this.setSlides([name === 'src' ? { src: val } : val]); return; }
    super.attributeChangedCallback(name, old, val);
  }
  connectedCallback() {
    if (!this.getAttribute('slides')) {
      const s = this.getAttribute('slide'), src = this.getAttribute('src');
      if (s || src) this.setAttribute('slides', '');
      this._initial = src ? [{ src }] : s ? [s] : [];
    }
    super.connectedCallback();
    if (this._initial?.length) { this.setSlides(this._initial); this._initial = null; }
  }
}

if (!customElements.get('wsi-grid')) customElements.define('wsi-grid', WsiGrid);
if (!customElements.get('wsi-viewer')) customElements.define('wsi-viewer', WsiViewer);
