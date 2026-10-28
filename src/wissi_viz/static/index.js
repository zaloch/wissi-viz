// wissi-viz — GPU whole-slide image grid viewer (WebGL2, zero dependencies).
//
//   <script type="module" src="/wissi-viz/static/index.js"></script>
//   <wissi-viz api="/wissi-viz" slides="slideA,slideB,slideC,slideD"></wissi-viz>
//
// One WebGL2 context renders 1–32 panes (browsers cap live contexts at ~16, so one
// context per slide would not scale). All panes share a GPU texture LRU, a worker pool
// that fetches + decodes off the main thread, and a priority scheduler that loads
// coarse levels first, then the view centre-out, then a predictive margin.

const MAX_CH = 8;
export const MAX_PANES = 32;   // one WebGL2 context, scissored per pane
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const now = () => performance.now();

// Level of detail.
//  motionBias  while a pane moves, fetch the level whose texels are up to this many device px (2 = one 2× level
//              coarser); the full-resolution level is fetched in the background and swapped in when it is ready
//  minify      draw an already-loaded FINER level down to this many device px per texel (0.5 = keep sharp tiles
//              while zooming out instead of dropping to the coarser level)
//  quadtree    draw only the finest loaded tile for each region (no painter's-algorithm overdraw)
//  background  while moving, full-res tiles use spare request slots only (never block urgent tiles)
//  prefetch    at rest, quietly prefetch the next finer level around the centre (next zoom step is instant)
//  destination while a pane animates toward a known view (goTo, double-click, wheel), fetch for where it will
//              land first; the in-between views only get spare capacity
//  adaptive    while moving and frames are slow (> ~20 ms, measured including the browser's commit), shrink the
//              WebGL backing store to ¾ or ½ (CSS stretches it); full resolution on the first frame at rest. At rest
//              with expensive frames, arriving tiles are batched into fewer redraws
//  decoupled   start the next request as soon as one finishes, and size GPU uploads to the real frame time,
//              so slow frames (weak/software GPUs, many panes) don't throttle the network
const LOD_DEFAULT = { motionBias: 2, minify: 0.5, quadtree: true, background: true, prefetch: false, decoupled: true, destination: true,
  overlays: 'server', adaptive: true };  // overlays: 'server' (pyramid tiles on demand) | 'client' (download columns once, build in a worker) | 'auto'
const BG_PRIO = 4;
const CLIENT_OVERLAY_MAX = 300_000;    // overlays: 'auto' → client mode for layers up to this many objects …
const CLIENT_OVERLAY_TOTAL = 1_000_000; // … while the viewer's client-side total stays below this
const LOD_LEGACY = { motionBias: 1, minify: 1, quadtree: false, background: false, prefetch: false, decoupled: false, destination: false,
  adaptive: false };

function hexToRgb(h) {
  h = String(h || '#ffffff').replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h.slice(0, 6), 16) || 0;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
const STAIN_PRESETS = { // Ruifrok & Johnston OD vectors
  HE: { hematoxylin: [0.650, 0.704, 0.286], second: [0.072, 0.990, 0.105] },
  HDAB: { hematoxylin: [0.650, 0.704, 0.286], second: [0.268, 0.570, 0.776] },
};
const norm3 = (v) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function inv3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C || 1e-9;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
const rgbToHex = (c) => '#' + c.map((v) => Math.round(clamp(v, 0, 1) * 255).toString(16).padStart(2, '0')).join('');

// ------------------------------------------------------------------ fetch + decode (worker)
async function wissiLoad(url, kind, signal) {
  const r = await fetch(url, { signal, credentials: 'same-origin' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  if (kind === 'image') {
    const blob = await r.blob();
    const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    return { bmp, bytes: blob.size, transfer: [bmp] };
  }
  if (kind === 'raw') {
    const ct = r.headers.get('content-type') || '';
    let buf;
    const len = +(r.headers.get('content-length') || 0);
    if (ct.includes('x-wissi-raw')) {
      buf = await new Response(r.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
    } else buf = await r.arrayBuffer();
    return { buf, bytes: len || buf.byteLength, transfer: [buf] };
  }
  const buf = await r.arrayBuffer();
  return { buf, bytes: buf.byteLength, transfer: [buf] };
}

const WORKER_SRC = `const wissiLoad = ${wissiLoad.toString()};
const ctrls = new Map();
self.onmessage = async (e) => {
  const m = e.data;
  if (m.abort) { const c = ctrls.get(m.id); if (c) c.abort(); return; }
  const ac = new AbortController(); ctrls.set(m.id, ac);
  try { const r = await wissiLoad(m.url, m.kind, ac.signal); const t = r.transfer; delete r.transfer; r.id = m.id; self.postMessage(r, t); }
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
    wissiLoad(url, kind, ac.signal).then(
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
uniform vec4 u_uv; // texture sub-rect: crops remote tile overlap (DeepZoom) and edge padding
out vec2 v_uv;
void main(){
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
  v_uv = u_uv.xy + c * u_uv.zw;
  vec2 s = (u_rect.xy + c * u_rect.zw - u_view.xy) * u_view.z;
  gl_Position = vec4(s.x / (u_vp.x * 0.5), -s.y / (u_vp.y * 0.5), 0.0, 1.0);
}`;

const FS_RGB = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec4 u_adj; // brightness, contrast, gamma, invert
// colour deconvolution (Ruifrok & Johnston): od = -ln(I); conc = od * M^-1 (rows of M = stain OD vectors)
uniform int u_stain;          // 0 off, 1 single stain as brightfield, 2 pseudo-fluorescence composite
uniform int u_stainSel;       // 0 hematoxylin, 1 second stain (eosin / DAB), 2 residual
uniform mat3 u_minv; uniform mat3 u_m; uniform float u_stainScale;
uniform vec3 u_col0; uniform vec3 u_col1;
in vec2 v_uv; out vec4 o;
void main(){
  vec3 c = texture(u_tex, v_uv).rgb;
  if (u_stain > 0) {
    vec3 od = -log(max(c, vec3(1.0 / 255.0)));
    vec3 conc = max(od * u_minv, vec3(0.0));
    if (u_stain == 1) {
      float k = conc[u_stainSel];
      c = exp(-k * u_m[u_stainSel]);         // that stain alone, as it would look in brightfield
    } else {
      vec2 t = clamp(conc.xy / u_stainScale, 0.0, 1.0);
      c = min(t.x * u_col0 + t.y * u_col1, vec3(1.0));   // dark background, stains as "channels"
    }
  }
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

// object-pyramid dots: positions are u16 within a tile (normalised), drawn per tile.
// mode 0 = raw features (a_size = size px, a_kind), 1 = aggregated with per-class histogram (pick the most
// common *visible* class), 2 = aggregated colour-by-value (a_col = mean colour bin)
const VS_PPTS = `#version 300 es
layout(location=0) in vec2 a_pos; layout(location=1) in float a_size;
layout(location=2) in uint a_col; layout(location=3) in uint a_kind;
layout(location=4) in uvec4 a_h0; layout(location=5) in uvec4 a_h1;
uniform vec3 u_view; uniform vec2 u_vp; uniform vec2 u_ps; uniform int u_onlyPoints;
uniform sampler2D u_ctex; uniform float u_alpha; uniform vec3 u_tile; uniform int u_mode; uniform float u_bin;
uniform int u_slots[8]; uniform int u_vis;
out vec4 v_col; flat out int v_sq;
void main(){
  vec2 w = u_tile.xy + a_pos * u_tile.z;
  vec2 s = (w - u_view.xy) * u_view.z;
  gl_Position = vec4(s.x / (u_vp.x * 0.5), -s.y / (u_vp.y * 0.5), 0.0, 1.0);
  int col = int(a_col);
  float sz;
  v_sq = 0;
  if (u_mode == 0) {
    sz = a_kind == 0u && a_size <= 0.0 ? u_ps.y : clamp(a_size * u_view.z, u_ps.x, u_ps.y);
    if (u_onlyPoints == 1 && a_kind != 0u) sz = 0.0;
  } else {
    sz = clamp(u_bin * u_view.z * 0.9, 1.0, 48.0);
    v_sq = 1;
    if (u_mode == 1) {
      uint h[8] = uint[8](a_h0.x, a_h0.y, a_h0.z, a_h0.w, a_h1.x, a_h1.y, a_h1.z, a_h1.w);
      uint best = 0u; int bi = -1;
      for (int j = 0; j < 8; j++) if (((u_vis >> j) & 1) == 1 && h[j] > best) { best = h[j]; bi = j; }
      if (bi < 0) sz = 0.0; else col = u_slots[bi];
    }
  }
  if (sz <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = sz;
  v_col = texelFetch(u_ctex, ivec2(col, 0), 0);
  v_col.a *= u_alpha;
}`;
const FS_PPTS = `#version 300 es
precision mediump float;
in vec4 v_col; flat in int v_sq; out vec4 o;
void main(){ if (v_col.a <= 0.0) discard;
  if (v_sq == 1) { o = v_col; return; }
  vec2 c = gl_PointCoord - 0.5; float d = dot(c, c); if (d > 0.25) discard;
  o = vec4(v_col.rgb, v_col.a * smoothstep(0.25, 0.16, d)); }`;

// parametric cells (ellipse / rect from the bounding box): one instanced quad each, outline + fill by SDF
// Instanced cell shapes (cx, cy, rx, ry) binned on a coarse grid by centre — counting sort, row-major — so a draw
// can submit only the grid rows that intersect the view. Real GPUs shrug at thousands of off-screen instances,
// but without one (VDI / thin clients: Chrome's SwiftShader) every instance costs ~10 µs even when clipped:
// 12.7k instances per pane at 1:1 took ~140 ms a frame.
function gridShapes(shp, col, kind, n) {
  const G = Math.max(1, Math.min(32, Math.round(Math.sqrt(n / 32))));
  if (G === 1) return { shp, col, kind, grid: null };
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, pad = 0;
  for (let i = 0; i < n; i++) {
    const x = shp[4 * i], y = shp[4 * i + 1];
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    const r = Math.max(shp[4 * i + 2], shp[4 * i + 3]); if (r > pad) pad = r;
  }
  const cw = (x1 - x0) / G || 1, ch = (y1 - y0) / G || 1;
  const cell = new Uint16Array(n), off = new Uint32Array(G * G + 1);
  for (let i = 0; i < n; i++) {
    const c = Math.min(G - 1, ((shp[4 * i] - x0) / cw) | 0) + G * Math.min(G - 1, ((shp[4 * i + 1] - y0) / ch) | 0);
    cell[i] = c; off[c + 1]++;
  }
  for (let c = 0; c < G * G; c++) off[c + 1] += off[c];
  const pos = off.slice(0, G * G), s2 = new Float32Array(n * 4), c2 = new col.constructor(n), k2 = new kind.constructor(n);
  for (let i = 0; i < n; i++) {
    const j = pos[cell[i]]++;
    s2[4 * j] = shp[4 * i]; s2[4 * j + 1] = shp[4 * i + 1]; s2[4 * j + 2] = shp[4 * i + 2]; s2[4 * j + 3] = shp[4 * i + 3];
    c2[j] = col[i]; k2[j] = kind[i];
  }
  return { shp: s2, col: c2, kind: k2, grid: { G, x0, y0, cw, ch, off, pad } };
}

const VS_SHP = `#version 300 es
layout(location=0) in vec4 a_shp; layout(location=1) in uint a_col; layout(location=2) in uint a_kind;
uniform vec3 u_view; uniform vec2 u_vp; uniform float u_w;
out vec2 v_p; flat out vec2 v_r; flat out int v_kind; flat out uint v_ci;
void main(){
  vec2 c = (a_shp.xy - u_view.xy) * u_view.z;
  vec2 r = max(a_shp.zw * u_view.z, vec2(0.5));
  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  vec2 ext = r + u_w + 1.0;
  vec2 p = c + corner * ext;
  gl_Position = vec4(p.x / (u_vp.x * 0.5), -p.y / (u_vp.y * 0.5), 0.0, 1.0);
  v_p = corner * ext; v_r = r; v_kind = int(a_kind); v_ci = a_col;
}`;
const FS_SHP = `#version 300 es
precision highp float;
in vec2 v_p; flat in vec2 v_r; flat in int v_kind; flat in uint v_ci; out vec4 o;
uniform sampler2D u_ctex; uniform float u_w; uniform float u_fill; uniform float u_alpha;
void main(){
  float d;
  if (v_kind == 3) { vec2 q = v_p / v_r; float k = length(q); d = (k - 1.0) * k / max(length(q / v_r), 1e-6); }
  else { vec2 q = abs(v_p) - v_r; d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0); }
  vec4 col = texelFetch(u_ctex, ivec2(int(v_ci), 0), 0);
  float line = 1.0 - smoothstep(u_w * 0.5 - 0.5, u_w * 0.5 + 0.5, abs(d));
  float fill = d < 0.0 ? u_fill : 0.0;
  float a = max(line, fill) * col.a * u_alpha;
  if (a <= 0.003) discard;
  o = vec4(col.rgb, a);
}`;

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
    this.grid._emit('layers', { slide: this.info?.id, url: this.url, names: [...this.layers.keys()] });
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
    const i = this.info, lv = i.levels[level];
    // browser-direct tiles (only for native RGB levels; virtual levels and raw channels come from wissi_viz)
    if (i.tile_url && i.mode === 'rgb' && !lv.virtual) {
      const id = lv.remote_level ?? level;
      const u = i.tile_url.replaceAll('{level}', id).replaceAll('{x}', x).replaceAll('{col}', x)
        .replaceAll('{y}', y).replaceAll('{row}', y).replaceAll('{c}', c ?? 0);
      return new URL(u, document.baseURI).href;
    }
    return `${this.url}/tiles/${level}/${x}/${y}?v=${encodeURIComponent(i.version)}${c == null ? '' : '&c=' + c}`;
  }
  annUrl(path, layer, version) {
    return `${this.url}/annotations/${path}?layer=${encodeURIComponent(layer)}&v=${version}`;
  }
}


// ------------------------------------------------------------------ client-side object pyramid (worker)
// The same pyramid the server builds (annotations.py _build_pyramid), built in the browser from the layer's columns.
const PYR_WORKER_SRC = `
const SLOTS = 8;
function csort(keys, K, idx) {             // stable counting sort of idx by integer keys[idx] in [0, K)
  const cnt = new Uint32Array(K + 1);
  for (let i = 0; i < idx.length; i++) cnt[keys[idx[i]] + 1]++;
  for (let k = 0; k < K; k++) cnt[k + 1] += cnt[k];
  const out = new Uint32Array(idx.length);
  for (let i = 0; i < idx.length; i++) out[cnt[keys[idx[i]]]++] = idx[i];
  return out;
}
function tiles(x, y, span, make, transfer) {   // group points by tile; make(sel, x0, y0) → tile arrays
  const m = x.length;
  let ntx = 1, nty = 1;
  const tx = new Uint32Array(m), ty = new Uint32Array(m);
  for (let i = 0; i < m; i++) { tx[i] = Math.max(0, Math.floor(x[i] / span)); ty[i] = Math.max(0, Math.floor(y[i] / span));
    if (tx[i] + 1 > ntx) ntx = tx[i] + 1; if (ty[i] + 1 > nty) nty = ty[i] + 1; }
  let idx = new Uint32Array(m); for (let i = 0; i < m; i++) idx[i] = i;
  idx = csort(tx, ntx, idx); idx = csort(ty, nty, idx);
  const out = [];
  let a = 0;
  while (a < m) {
    let b = a + 1;
    const X = tx[idx[a]], Y = ty[idx[a]];
    while (b < m && tx[idx[b]] === X && ty[idx[b]] === Y) b++;
    const sel = idx.subarray(a, b), x0 = X * span, y0 = Y * span;
    const pos = new Uint16Array(sel.length * 2);
    for (let i = 0; i < sel.length; i++) {
      pos[2 * i] = Math.min(65535, Math.max(0, Math.round((x[sel[i]] - x0) / span * 65535)));
      pos[2 * i + 1] = Math.min(65535, Math.max(0, Math.round((y[sel[i]] - y0) / span * 65535)));
    }
    const t = { tx: X, ty: Y, n: sel.length, x0, y0, span, pos, ...make(sel) };
    for (const v of Object.values(t)) if (v && v.buffer) transfer.push(v.buffer);
    out.push(t);
    a = b;
  }
  return out;
}
self.onmessage = (e) => {
  const { id, buf } = e.data;
  try {
    const dv = new DataView(buf), jl = dv.getUint32(4, true);
    const M = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jl)));
    let off = 8 + jl;
    const take = (C, n) => { const a = new C(buf.slice(off, off + n * C.BYTES_PER_ELEMENT)); off += Math.ceil(n * C.BYTES_PER_ELEMENT / 4) * 4; return a; };
    const n = M.n, cx = take(Float32Array, n), cy = take(Float32Array, n), rx = take(Uint16Array, n), ry = take(Uint16Array, n);
    const col = take(Uint16Array, n), cls = take(Uint16Array, n), kind = take(Uint8Array, n);
    const transfer = [];
    const k0 = Math.min(24, Math.max(0, Math.ceil(Math.log2(Math.max(M.median_size, 1)))));
    const extent = Math.max(M.bbox[2], M.bbox[3], 1), kmax = Math.max(k0 + 1, Math.ceil(Math.log2(extent / 64)));
    const TB = M.tile_bins, value = M.mode === 'value';
    // top classes → histogram slots ('other' folded into the last slot when there are more than 8 classes)
    const cc = new Uint32Array(Math.max(M.nclasses, 1)); for (let i = 0; i < n; i++) cc[cls[i]]++;
    let top = [...cc.keys()].filter((c) => cc[c] > 0).sort((a, b) => cc[b] - cc[a]).slice(0, SLOTS);
    const other = M.nclasses > SLOTS;
    const slotOf = new Uint8Array(Math.max(M.nclasses, 1)).fill(other ? SLOTS - 1 : 0);
    if (other) top = top.slice(0, SLOTS - 1);
    top.forEach((c, j) => { slotOf[c] = j; });
    let otherClass = null;
    if (other) { let best = -1; for (let c = 0; c < M.nclasses; c++) if (!top.includes(c) && cc[c] > best) { best = cc[c]; otherClass = c; } }
    const slots = top.concat(other ? [-1] : []);
    const tc = top.concat(new Array(SLOTS - top.length).fill(otherClass ?? top[top.length - 1] ?? 0));
    const levels = [];
    // level 0: every object (raw), with its radii for GPU outlines
    const size = new Uint16Array(n);
    for (let i = 0; i < n; i++) size[i] = Math.min(65535, Math.max(rx[i], ry[i]) / 2);
    levels.push({ k: k0, raw: true, tiles: tiles(cx, cy, 2 ** k0 * TB, (sel) => {
      const m = sel.length, o = { size: new Uint16Array(m), col: new Uint16Array(m), cls: new Uint16Array(m), orig: new Uint32Array(m),
        kind: new Uint8Array(m), shp: new Float32Array(m * 4) };
      for (let i = 0; i < m; i++) { const f = sel[i]; o.size[i] = size[f]; o.col[i] = col[f]; o.cls[i] = cls[f]; o.orig[i] = f; o.kind[i] = kind[f];
        o.shp[4 * i] = cx[f]; o.shp[4 * i + 1] = cy[f]; o.shp[4 * i + 2] = rx[f] / 4; o.shp[4 * i + 3] = ry[f] / 4; }
      return o; }, transfer) });
    // aggregated levels, each from the one below
    let X = Float64Array.from(cx), Y = Float64Array.from(cy), W = new Float64Array(n).fill(1);
    let REP = new Uint32Array(n); for (let i = 0; i < n; i++) REP[i] = i;
    let H = null, VS = null, VC = null;
    if (value) { VS = new Float64Array(n); VC = new Float64Array(n); for (let i = 0; i < n; i++) if (col[i] !== M.nan) { VS[i] = col[i]; VC[i] = 1; } }
    else { H = new Float64Array(n * SLOTS); for (let i = 0; i < n; i++) H[i * SLOTS + slotOf[cls[i]]] = 1; }
    for (let k = k0 + 1; k <= kmax; k++) {
      const s = 2 ** k, m = X.length;
      const bx = new Uint32Array(m), by = new Uint32Array(m);
      let nbx = 1, nby = 1;
      for (let i = 0; i < m; i++) { bx[i] = Math.max(0, Math.floor(X[i] / s)); by[i] = Math.max(0, Math.floor(Y[i] / s));
        if (bx[i] + 1 > nbx) nbx = bx[i] + 1; if (by[i] + 1 > nby) nby = by[i] + 1; }
      let idx = new Uint32Array(m); for (let i = 0; i < m; i++) idx[i] = i;
      idx = csort(bx, nbx, idx); idx = csort(by, nby, idx);
      // count groups
      let g = 0; for (let i = 0; i < m; i++) if (i === 0 || bx[idx[i]] !== bx[idx[i - 1]] || by[idx[i]] !== by[idx[i - 1]]) g++;
      const nX = new Float64Array(g), nY = new Float64Array(g), nW = new Float64Array(g), nR = new Uint32Array(g);
      const nH = value ? null : new Float64Array(g * SLOTS), nVS = value ? new Float64Array(g) : null, nVC = value ? new Float64Array(g) : null;
      let j = -1;
      for (let ii = 0; ii < m; ii++) {
        const i = idx[ii];
        if (ii === 0 || bx[i] !== bx[idx[ii - 1]] || by[i] !== by[idx[ii - 1]]) { j++; nR[j] = REP[i]; }
        nW[j] += W[i]; nX[j] += X[i] * W[i]; nY[j] += Y[i] * W[i];
        if (value) { nVS[j] += VS[i]; nVC[j] += VC[i]; } else for (let q = 0; q < SLOTS; q++) nH[j * SLOTS + q] += H[i * SLOTS + q];
      }
      for (let q = 0; q < g; q++) { nX[q] /= nW[q]; nY[q] /= nW[q]; }
      X = nX; Y = nY; W = nW; REP = nR; H = nH; VS = nVS; VC = nVC;
      levels.push({ k, raw: false, hist: !value, tiles: tiles(X, Y, 2 ** k * TB, (sel) => {
        const m2 = sel.length, o = { count: new Uint16Array(m2), col: new Uint16Array(m2), orig: new Uint32Array(m2) };
        if (!value) o.hist = new Uint8Array(m2 * SLOTS);
        for (let i = 0; i < m2; i++) {
          const f = sel[i]; o.count[i] = Math.min(65535, W[f]); o.orig[i] = REP[f];
          if (value) o.col[i] = VC[f] > 0 ? Math.round(VS[f] / VC[f]) : M.nan;   // (rounding matches the server)
          else { let mx = 0, am = 0; for (let q = 0; q < SLOTS; q++) if (H[f * SLOTS + q] > mx) { mx = H[f * SLOTS + q]; am = q; }
            o.col[i] = tc[am]; for (let q = 0; q < SLOTS; q++) { const h = H[f * SLOTS + q]; o.hist[i * SLOTS + q] = h > 0 ? Math.max(1, Math.round(255 * h / mx)) : 0; } }
        }
        return o; }, transfer) });
    }
    const pyr = { k0, kmax, levels: levels.length, tile_bins: TB, mode: value ? 'value' : 'class', slots, other_class: otherClass,
      tiles: levels.map((L) => L.tiles.length), points: levels.map((L) => L.tiles.reduce((a, t) => a + t.n, 0)) };
    self.postMessage({ id, pyr, levels }, transfer);
  } catch (err) { self.postMessage({ id, error: String(err && err.stack || err) }); }
};`;
let _pyrWorker = null, _pyrSeq = 0;
const _pyrCbs = new Map();
function buildPyramidInWorker(buf) {
  if (!_pyrWorker) {
    _pyrWorker = new Worker(URL.createObjectURL(new Blob([PYR_WORKER_SRC], { type: 'text/javascript' })));
    _pyrWorker.onmessage = (e) => { const cb = _pyrCbs.get(e.data.id); _pyrCbs.delete(e.data.id); if (cb) (e.data.error ? cb.reject(new Error(e.data.error)) : cb.resolve(e.data)); };
  }
  const id = ++_pyrSeq;
  return new Promise((resolve, reject) => { _pyrCbs.set(id, { resolve, reject }); _pyrWorker.postMessage({ id, buf }, [buf]); });
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
    if (meta.index_only) {
      // object pyramid: only the bucket table + which pyramid tiles exist; dots arrive tile by tile
      this.pyr = meta.pyramid;
      this.buckets = take(Float32Array, B * 8);
      this.tileSets = [];
      for (let l = 0; l < this.pyr.levels; l++) {
        const t = take(Int32Array, this.pyr.tiles[l] * 2), set = new Set();
        for (let i = 0; i < t.length; i += 2) set.add(t[i] * 65536 + t[i + 1]);
        this.tileSets.push(set);
      }
      this.ptiles = new Map();
      this.n = n;
      this.nb = B;
      this.medSize = meta.median_size || 0;
      this.clientOk = !!meta.client_ok;
      this.client = null;
      this.classes = meta.classes || [];
      this.colormap = meta.colormap || null;
      this.ctex = gl.createTexture();
      this._classTex();
      this.grid.gpuBytes += buf.byteLength;
      this.state = 'ready';
      this.considerClient();
      return;
    }
    this.cen = take(Float32Array, n * 2);
    this.size = take(Float32Array, n);
    this.orig = take(Uint32Array, n);
    this.cls = take(Uint16Array, n);
    this.col = take(Uint16Array, n); // colour-texture index (class, or colormap bin)
    this.kind = take(Uint8Array, n);
    this.buckets = take(Float32Array, B * 8);
    this.n = n;
    this.nb = B;
    // typical object size → draw order (cells under ROIs) unless style.z is given
    const step = Math.max(1, Math.floor(n / 2000)), sample = [];
    for (let i = 0; i < n; i += step) sample.push(this.size[i]);
    sample.sort((a, b) => a - b);
    this.medSize = sample.length ? sample[sample.length >> 1] : 0;
    this.classes = meta.classes || [];
    this.colormap = meta.colormap || null;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.bufs = [
      this._attr(0, this.cen, 2, gl.FLOAT, false),
      this._attr(1, this.size, 1, gl.FLOAT, false),
      this._attr(2, this.col, 1, gl.UNSIGNED_SHORT, true),
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
    // texels: [class colours …][256-entry colormap LUT][NaN colour]
    const gl = this.grid.gl;
    const nc = this.classes.length;
    const cm = this.colormap;
    const n = Math.max(1, nc + (cm ? 257 : 0));
    const px = new Uint8Array(n * 4);
    this.classes.forEach((c, i) => {
      const [r, g, b] = hexToRgb(c.color);
      px.set([r * 255, g * 255, b * 255, this.hidden.has(c.name) ? 0 : 255], i * 4);
    });
    if (cm) {
      const lo = this.valueRange?.[0] ?? 0, hi = this.valueRange?.[1] ?? 255; // optional on-the-fly range filter (bins)
      cm.lut.forEach((h, i) => { const [r, g, b] = hexToRgb(h); px.set([r * 255, g * 255, b * 255, i >= lo && i <= hi ? 255 : 0], (nc + i) * 4); });
      const [r, g, b] = hexToRgb(cm.nan);
      px.set([r * 255, g * 255, b * 255, this.hideNaN ? 0 : 160], (nc + 256) * 4);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.ctex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }
  // pyramid level for a zoom: aggregated bins ≥ ~4 device px, or raw features once bins would be finer than a cell
  pyrLevel(scaleDev) {
    const P = this.pyr, kNeed = Math.ceil(Math.log2(4 / scaleDev));
    return kNeed <= P.k0 ? 0 : Math.min(P.levels - 1, kNeed - P.k0);
  }
  pyrSpan(level) { return 2 ** (this.pyr.k0 + level) * this.pyr.tile_bins; }
  pyrTiles(level, r) {
    const span = this.pyrSpan(level), set = this.tileSets[level], out = [];
    const x0 = Math.max(0, Math.floor(r[0] / span)), x1 = Math.floor(r[2] / span);
    const y0 = Math.max(0, Math.floor(r[1] / span)), y1 = Math.floor(r[3] / span);
    for (let ty = y0; ty <= y1; ty++) for (let tx = x0; tx <= x1; tx++) if (set.has(tx * 65536 + ty)) out.push([tx, ty]);
    return out;
  }
  pkey(level, tx, ty) { return `${level}|${tx}|${ty}`; }
  // client mode: fetch the layer's columns once, build the pyramid in a worker, then never ask again.
  // 'auto': small layers only, and only while this viewer holds < CLIENT_OVERLAY_TOTAL objects client-side
  // (measured: 1M objects ≈ 70 MB of browser memory + a ~2 s worker build; 16 × 1M = 1.1 GB, 32 s).
  // Decided when the layer loads and again whenever setLod changes the overlay mode, so switching to
  // 'client' after a layer has loaded still takes effect.
  considerClient() {
    if (this.client != null || !this.clientOk || this.state !== 'ready') return;
    const g = this.grid, om = g.lod.overlays, n = this.n;
    const auto = om === 'auto' && n <= CLIENT_OVERLAY_MAX && (g.clientObjects || 0) + n <= CLIENT_OVERLAY_TOTAL;
    if (om !== 'client' && !auto) return;
    this.client = 'pending';
    g.clientObjects = (g.clientObjects || 0) + n;
    this.clientCounted = n;
    g.requestRender();
  }

  async ingestColumns(buf) {
    this.client = 'building';
    try {
      const t0 = now();
      const r = await buildPyramidInWorker(buf);
      if (this.state === 'dead') return;
      this.pyr = r.pyr;
      this.tileSets = r.levels.map((L) => new Set(L.tiles.map((t) => t.tx * 65536 + t.ty)));
      this.local = r.levels.map((L) => new Map(L.tiles.map((t) => [this.pkey(r.levels.indexOf(L), t.tx, t.ty), { ...t, k: L.k, raw: L.raw, hasHist: !!L.hist }])));
      for (const k of [...this.ptiles.keys()]) this.dropPTile(k);   // (server tiles of another mode)
      this.buildMs = now() - t0;
      this.client = 'ready';
    } catch (e) { console.warn('wissi-viz client pyramid', e); this.client = null; }
    this.grid.requestRender();
  }
  // a locally built tile → GPU (same layout as server tiles; raw tiles also get an outline VAO)
  localPTile(key) {
    const lvl = +key.split('|')[0], t = this.local[lvl].get(key);
    if (!t) return null;
    const gl = this.grid.gl;
    const d = { n: t.n, raw: t.raw, hist: t.hasHist, x0: t.x0, y0: t.y0, span: t.span, k: t.k, bytes: 0, last: this.grid.frame, bufs: [] };
    const attr = (vao, loc, data, size, type, int, norm = false, div = 0) => {
      const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      if (int) gl.vertexAttribIPointer(loc, size, type, 0, 0); else gl.vertexAttribPointer(loc, size, type, norm, 0, 0);
      if (div) gl.vertexAttribDivisor(loc, div);
      d.bufs.push(b); d.bytes += data.byteLength;
    };
    d.vao = gl.createVertexArray(); gl.bindVertexArray(d.vao);
    attr(d.vao, 0, t.pos, 2, gl.UNSIGNED_SHORT, false, true);
    if (t.raw) {
      attr(d.vao, 1, t.size, 1, gl.UNSIGNED_SHORT, false); attr(d.vao, 2, t.col, 1, gl.UNSIGNED_SHORT, true); attr(d.vao, 3, t.kind, 1, gl.UNSIGNED_BYTE, true);
      d.cpu = { pos: t.pos, size: t.size, cls: t.cls, orig: t.orig, kind: t.kind };
      const g = gridShapes(t.shp, t.col, t.kind, t.n);
      d.vaoShp = gl.createVertexArray(); gl.bindVertexArray(d.vaoShp);
      attr(d.vaoShp, 0, g.shp, 4, gl.FLOAT, false, false, 1); attr(d.vaoShp, 1, g.col, 1, gl.UNSIGNED_SHORT, true, false, 1);
      attr(d.vaoShp, 2, g.kind, 1, gl.UNSIGNED_BYTE, true, false, 1);
      d.shpBufs = d.bufs.slice(-3);
      d.shpGrid = g.grid;
    } else {
      attr(d.vao, 1, t.count, 1, gl.UNSIGNED_SHORT, false); attr(d.vao, 2, t.col, 1, gl.UNSIGNED_SHORT, true);
      if (t.hasHist) {
        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, t.hist, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(4); gl.vertexAttribIPointer(4, 4, gl.UNSIGNED_BYTE, 8, 0);
        gl.enableVertexAttribArray(5); gl.vertexAttribIPointer(5, 4, gl.UNSIGNED_BYTE, 8, 4);
        d.bufs.push(b); d.bytes += t.hist.byteLength;
      }
      d.cpu = { orig: t.orig };
    }
    gl.bindVertexArray(null);
    this.ptiles.set(key, d);
    this.grid.gpuBytes += d.bytes;
    return d;
  }
  ingestPTile(key, buf) {
    const gl = this.grid.gl;
    const { meta, take } = unpack(buf);
    const n = meta.n;
    const pos = take(Uint16Array, n * 2);
    const t = { n, raw: meta.raw, hist: !!meta.hist, x0: meta.x0, y0: meta.y0, span: meta.span, k: meta.k,
      bytes: buf.byteLength, last: this.grid.frame, bufs: [] };
    t.vao = gl.createVertexArray();
    gl.bindVertexArray(t.vao);
    const attr = (loc, data, size, type, int, norm = false) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      if (int) gl.vertexAttribIPointer(loc, size, type, 0, 0); else gl.vertexAttribPointer(loc, size, type, norm, 0, 0);
      t.bufs.push(b);
      return b;
    };
    attr(0, pos, 2, gl.UNSIGNED_SHORT, false, true);
    if (meta.raw) {
      const size = take(Uint16Array, n), col = take(Uint16Array, n), cls = take(Uint16Array, n);
      const orig = take(Uint32Array, n), kind = take(Uint8Array, n);
      attr(1, size, 1, gl.UNSIGNED_SHORT, false);
      attr(2, col, 1, gl.UNSIGNED_SHORT, true);
      attr(3, kind, 1, gl.UNSIGNED_BYTE, true);
      t.cpu = { pos, size, cls, orig, kind };        // for click hit-testing in the browser
    } else {
      const cnt = take(Uint16Array, n), col = take(Uint16Array, n), orig = take(Uint32Array, n);
      attr(1, cnt, 1, gl.UNSIGNED_SHORT, false);
      attr(2, col, 1, gl.UNSIGNED_SHORT, true);
      if (meta.hist) {
        const h = take(Uint8Array, n * 8);
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, h, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(4);
        gl.vertexAttribIPointer(4, 4, gl.UNSIGNED_BYTE, 8, 0);
        gl.enableVertexAttribArray(5);
        gl.vertexAttribIPointer(5, 4, gl.UNSIGNED_BYTE, 8, 4);
        t.bufs.push(b);
      }
      t.cpu = { orig };
    }
    gl.bindVertexArray(null);
    this.ptiles.set(key, t);
    this.grid.gpuBytes += t.bytes;
  }
  dropPTile(key) {
    const t = this.ptiles?.get(key);
    if (!t) return;
    const gl = this.grid.gl;
    t.bufs.forEach((b) => gl.deleteBuffer(b));
    gl.deleteVertexArray(t.vao);
    if (t.vaoShp) gl.deleteVertexArray(t.vaoShp);
    this.grid.gpuBytes -= t.bytes;
    this.ptiles.delete(key);
  }
  // slot visibility bitmask for aggregated dots (a slot is visible when its class is; 'other' when any rarer class is)
  slotMask() {
    const S = this.pyr.slots;
    let m = 0;
    S.forEach((c, j) => {
      if (c >= 0) { if (!this.hidden.has(this.classes[c]?.name)) m |= 1 << j; }
      else if (this.classes.some((k, i) => !S.includes(i) && !this.hidden.has(k.name))) m |= 1 << j;
    });
    return m;
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
    const S = meta.segments, T = meta.tri_verts, P = meta.shapes || 0;
    const seg = take(Float32Array, S * 4), tv = take(Float32Array, T * 2);
    const sc = take(Uint16Array, S), tc = take(Uint16Array, T);
    const d = { nSeg: S, nTri: T, nShp: P, bytes: buf.byteLength, last: this.grid.frame, bufs: [] };
    if (P) {
      const g = gridShapes(take(Float32Array, P * 4), take(Uint16Array, P), take(Uint8Array, P), P);
      d.vaoShp = gl.createVertexArray();
      gl.bindVertexArray(d.vaoShp);
      d.shpBufs = [this._attr(0, g.shp, 4, gl.FLOAT, false, 1), this._attr(1, g.col, 1, gl.UNSIGNED_SHORT, true, 1),
        this._attr(2, g.kind, 1, gl.UNSIGNED_BYTE, true, 1)];
      d.shpGrid = g.grid;
      d.bufs.push(...d.shpBufs);
    }
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
    if (d.vaoShp) gl.deleteVertexArray(d.vaoShp);
    this.grid.gpuBytes -= d.bytes;
    this.detail.delete(i);
  }
  destroy() {
    const gl = this.grid.gl;
    for (const i of [...this.detail.keys()]) this.dropBucket(i);
    for (const k of [...(this.ptiles?.keys() || [])]) this.dropPTile(k);
    if (this.clientCounted) { this.grid.clientObjects -= this.clientCounted; this.clientCounted = 0; }
    this.local = null;
    if (this.state === 'ready') {
      if (this.bufs) this.bufs.forEach((b) => gl.deleteBuffer(b));
      if (this.vao) gl.deleteVertexArray(this.vao);
      gl.deleteTexture(this.ctex);
    }
    this.state = 'dead';
  }
  // Click → {f (feature index), cx, cy, size, cls} | null; undefined = "ask the server" (pyramid layer whose
  // raw tiles around the point are not loaded, i.e. the pane shows aggregated dots there)
  hitInfo(x, y, tol, level) {
    if (this.state !== 'ready') return null;
    if (!this.pyr) {
      const f = this.hit(x, y, tol);
      return f < 0 ? null : { f: this.orig[f], cx: this.cen[2 * f], cy: this.cen[2 * f + 1], size: this.size[f], cls: this.cls[f] };
    }
    if (level !== 0 && this.client !== 'ready') return undefined;
    let best = null;
    const tiles = this.pyrTiles(0, [x - tol, y - tol, x + tol, y + tol]);
    for (const [tx, ty] of tiles) {
      const t = this.ptiles.get(this.pkey(0, tx, ty)) || (this.client === 'ready' ? this.localPTile(this.pkey(0, tx, ty)) : null);
      if (!t) return undefined;
      const c = t.cpu;
      for (let i = 0; i < t.n; i++) {
        const px = t.x0 + c.pos[2 * i] / 65535 * t.span, py = t.y0 + c.pos[2 * i + 1] / 65535 * t.span;
        const r = c.size[i] / 2 + tol, dx = x - px, dy = y - py;
        if (dx * dx + dy * dy <= r * r && (!best || c.size[i] < best.size) && !this.hidden.has(this.classes[c.cls[i]]?.name))
          best = { f: c.orig[i], cx: px, cy: py, size: c.size[i], cls: c.cls[i] };
      }
    }
    return best;
  }
  async serverHit(x, y, tol) {
    const hidden = this.classes.map((c, i) => (this.hidden.has(c.name) ? i : -1)).filter((i) => i >= 0).join(',');
    try {
      const r = await fetch(`${this.slide.url}/annotations/hit?layer=${encodeURIComponent(this.name)}&x=${x}&y=${y}&tol=${tol}` +
        (hidden ? `&hidden=${hidden}` : ''), { credentials: 'same-origin' });
      const j = r.ok ? await r.json() : null;
      return j && j.index != null ? { f: j.index, cx: j.cx, cy: j.cy, size: j.size, cls: j.cls } : null;
    } catch (_) { return null; }
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
.legend { position:absolute; right:6px; bottom:34px; max-width:min(240px, 45%); max-height:45%; overflow:hidden; pointer-events:none;
  padding:5px 7px; border-radius:6px; background:rgba(0,0,0,.6); font-size:11px; line-height:1.35; }
.legend:empty { display:none; }
.legend b { display:block; font-weight:600; opacity:.85; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.legend .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.legend .grad { height:8px; border-radius:2px; margin:3px 0 1px; }
.legend .ends { display:flex; justify-content:space-between; opacity:.85; font-variant-numeric:tabular-nums; gap:8px; }
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
  static get observedAttributes() { return ['api', 'slides', 'layout', 'sync', 'controls', 'quality', 'tool', 'draw-layer', 'legend', 'lod']; }

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
    this.failCounts = new Map(); // key → consecutive failures (back-off)
    this.queued = new Set(); // keys fetched + decoded, waiting for a GPU upload slot (not to be re-requested)
    this.layoutSpec = 'auto';
    this.syncMode = 'none';
    this.tool = 'pan';
    this.drawLayer = null;
    this.drawClass = null;   // class of new drawings / SAM masks (default: the draw layer's name)
    this.drawColor = null;
    this.quality = 1.0;
    this.showControls = true;
    this.showLegend = true;
    this.counters = { requests: 0, completed: 0, bytes: 0, aborted: 0, cancelled: 0, failed: 0, uploaded: 0, wasted: 0 };
    // level-of-detail policy (see setLod). LOD_LEGACY reproduces the pre-0.4 behaviour for A/B benchmarks.
    this.lod = { ...LOD_DEFAULT };
    this.fx = { frames: 0, draws: 0, fill: 0, area: 0 }; // draw statistics: tile quads, device px filled vs pane px
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
    else if (name === 'legend') { this.showLegend = val !== 'false'; this._legends(); }
    else if (name === 'quality') this.quality = clamp(parseFloat(val) || 1, 0.25, 4);
    else if (name === 'lod') { let v = val; try { if (v && v.trim().startsWith('{')) v = JSON.parse(v); } catch (e) { /* preset name */ } this.setLod(v || 'default'); }
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
      ppts: compile(gl, VS_PPTS, FS_PPTS),
      shp: compile(gl, VS_SHP, FS_SHP),
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
  /** Panes from slide ids / URLs / {slide|src, view, channels, layers} objects. Up to MAX_PANES (32). */
  setSlides(list) {
    list = (list || []).slice(0, MAX_PANES);
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

  /**
   * Colour deconvolution view for brightfield panes.
   * mode: 'off' | 'hematoxylin' | 'eosin' | 'dab' | 'second' | 'residual' | 'fluorescence'
   * opts: { preset: 'HE' | 'HDAB', vectors: {hematoxylin: [r,g,b], second: [r,g,b]}, scale, colors: [hex, hex] }
   */
  setStain(pane, mode, opts = {}) {
    for (const p of this._panes(pane)) {
      if (!mode || mode === 'off') { p.stain = null; continue; }
      const preset = (opts.preset || (mode === 'dab' ? 'HDAB' : 'HE')).toUpperCase();
      const v = opts.vectors || STAIN_PRESETS[preset];
      const M = [norm3(v.hematoxylin), norm3(v.second)];
      M.push(norm3(cross3(M[0], M[1])));
      const Minv = inv3(M);
      const sel = { hematoxylin: 0, eosin: 1, dab: 1, second: 1, residual: 2 }[mode] ?? 0;
      const cols = (opts.colors || (preset === 'HDAB' ? ['#3d7bff', '#ffb000'] : ['#3d7bff', '#ff4fa3'])).map(hexToRgb);
      p.stain = {
        mode: mode === 'fluorescence' ? 2 : 1, sel, scale: opts.scale ?? 1.6, col0: cols[0], col1: cols[1],
        // GLSL: conc = od * u_minv  → upload Minv column-major; u_m[i] = row i of M → upload rows as columns
        minvCols: new Float32Array([0, 1, 2].flatMap((j) => [0, 1, 2].map((i) => Minv[i][j]))),
        mRowsAsCols: new Float32Array(M.flat()),
        name: mode, preset,
      };
    }
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
    this._legends();
    this.requestRender();
  }

  /** Re-fetch annotation layers (call after changing them on the server). slide = id, URL, or omit for all. */
  refreshAnnotations(slide) {
    for (const s of this.slides.values()) {
      if (!slide || s.url.endsWith('/slides/' + encodeURIComponent(slide)) || s.url === slide) s.refreshAnnotations();
    }
  }

  /** Layer that drawings and accepted SAM masks are saved to (created on first use), and their class / colour. */
  setDrawLayer(name) { this.drawLayer = name || null; }
  setDrawClass(name, color = null) { this.drawClass = name || null; this.drawColor = color || null; }

  /** 'pan' | 'point' | 'rect' | 'ellipse' | 'polygon' | 'line' | 'freehand' | 'sam'
   *  sam: click = include, shift/alt-click = exclude, drag = box; Enter or double-click accepts, Esc discards. */
  setTool(tool) {
    if (this.tool === 'sam' && tool !== 'sam') this.panes.forEach((p) => { p.sam = null; });
    this.tool = tool || 'pan';
    this._drawing = null;
    this.panes.forEach((p) => p.el?.classList.toggle('tool', this.tool !== 'pan'));
    this._drawOverlay();
  }

  async select(pane, layer, index) {
    const p = this._pane(pane);
    if (!p) return;
    if (index == null) { p.selection = null; this.requestRender(); return; }
    const L = p.slide?.layers.get(layer);
    let sel = { layer, f: index };
    if (L?.state === 'ready' && !L.pyr) {
      const f = Array.prototype.indexOf.call(L.orig, index);
      if (f >= 0) sel = { layer, f: index, cx: L.cen[2 * f], cy: L.cen[2 * f + 1], size: L.size[f] };
    } else if (p.slide) {
      try {
        const r = await fetch(`${p.slide.url}/annotations/feature/${index}?layer=${encodeURIComponent(layer)}`, { credentials: 'same-origin' });
        const j = r.ok ? await r.json() : null;
        if (j?.centroid) sel = { layer, f: index, cx: j.centroid[0], cy: j.centroid[1], size: j.size };
      } catch (_) { /* no outline then */ }
    }
    p.selection = sel;
    this.requestRender();
  }

  /**
   * How much of each pane is drawn at full resolution right now.
   * sharp: fraction of the visible slide area covered by target-level tiles (all visible channels);
   * blank: fraction covered by no level at all (only background) — what a user sees as "holes".
   */
  coverage() {
    const per = [];
    for (const p of this.panes) {
      const info = p.slide?.info;
      if (!info || !p.view || p.rect.w <= 0) { per.push(null); continue; }
      const chans = this._visibleChannels(p);
      const target = this._level(info, p.view.scale * this.dpr);
      const vis = this._visRect(p);
      const clip = [Math.max(0, vis[0]), Math.max(0, vis[1]), Math.min(info.width, vis[2]), Math.min(info.height, vis[3])];
      if (clip[2] <= clip[0] || clip[3] <= clip[1]) { per.push({ pane: p.index, offSlide: true }); continue; } // nothing to draw
      const area = Math.max(1, (clip[2] - clip[0]) * (clip[3] - clip[1]));
      const tileArea = (L, rg, fn) => {
        const lv = info.levels[L], ts = info.tile_size;
        let a = 0;
        for (let ty = rg.y0; ty <= rg.y1; ty++) for (let tx = rg.x0; tx <= rg.x1; tx++) {
          if (!fn(L, tx, ty)) continue;
          const x0 = Math.max(clip[0], tx * ts * lv.sx), x1 = Math.min(clip[2], (tx + 1) * ts * lv.sx);
          const y0 = Math.max(clip[1], ty * ts * lv.sy), y1 = Math.min(clip[3], (ty + 1) * ts * lv.sy);
          if (x1 > x0 && y1 > y0) a += (x1 - x0) * (y1 - y0);
        }
        return a;
      };
      const scaleDev = p.view.scale * this.dpr;
      const floor = this._drawFloor(info, scaleDev, target);
      let sharp;
      if (floor < target) { // full resolution = drawn from the target level or finer
        sharp = 0;
        this._cover(p, info, floor, clip, chans, (L, x, y, r) => { if (L <= target) sharp += (r[2] - r[0]) * (r[3] - r[1]); });
        sharp /= area;
      } else {
        const rg = this._tileRange(info, target, clip);
        sharp = rg ? tileArea(target, rg, (L, x, y) => this._ready(p, L, x, y, chans)) / area : 1;
      }
      // sample a grid of points: finest drawable level covering each → blank fraction and mean detail
      // (detail 1 = texel ≤ 1 device px (× quality); 0.5 = one 2× level coarser; 0 = nothing drawn)
      let covered = 0, n = 0, detail = 0;
      for (let gy = 0; gy < 12; gy++) for (let gx = 0; gx < 12; gx++, n++) {
        const X = clip[0] + (gx + 0.5) / 12 * (clip[2] - clip[0]), Y = clip[1] + (gy + 0.5) / 12 * (clip[3] - clip[1]);
        for (let L = floor; L < info.levels.length; L++) {
          const lv = info.levels[L], ts = info.tile_size;
          if (this._ready(p, L, Math.floor(X / lv.sx / ts), Math.floor(Y / lv.sy / ts), chans)) {
            covered++;
            detail += Math.min(1, this.quality / (lv.downsample * scaleDev));
            break;
          }
        }
      }
      per.push({ pane: p.index, level: target, sharp: Math.min(1, sharp), blank: 1 - covered / n, detail: detail / n });
    }
    const ok = per.filter((c) => c && !c.offSlide);
    const mean = (k) => (ok.length ? ok.reduce((a, c) => a + c[k], 0) / ok.length : 0);
    return { sharp: mean('sharp'), blank: mean('blank'), detail: mean('detail'), panes: per };
  }

  stats() {
    return { textures: this.tex.size, textureMB: +(this.texBytes / 2 ** 20).toFixed(1), annotationMB: +(this.gpuBytes / 2 ** 20).toFixed(1),
      inflight: this.inflight.size, queuedUploads: this.uploads.length, frame: this.frame, budgetMB: this.budget / 2 ** 20,
      ...this.counters, maxInflight: this.maxInflight,
      overdraw: this.fx.area ? +(this.fx.fill / this.fx.area).toFixed(3) : null,
      drawsPerFrame: this.fx.frames ? +(this.fx.draws / this.fx.frames).toFixed(1) : null, lowresFrames: this.fx.lowres || 0,
      backingResizes: this.fx.resizes || 0, renderScale: this._rs || 1, lod: { ...this.lod } };
  }

  setTextureBudget(mb) { this.budget = mb * 2 ** 20; this._evict(); }

  /** Level-of-detail policy: setLod('default'|'legacy') or any of {motionBias, minify, quadtree, background, prefetch,
   *  decoupled, destination, adaptive, overlays: 'server'|'client'|'auto'} (merged into the current policy). */
  setLod(opts) {
    const was = this.lod.overlays;
    this.lod = opts === 'legacy' ? { ...LOD_LEGACY } : opts === 'default' || opts == null ? { ...LOD_DEFAULT } : { ...this.lod, ...opts };
    if (this.lod.overlays !== was) for (const s of this.slides.values()) for (const L of s.layers.values()) L.considerClient?.();
    this.requestRender();
    return { ...this.lod };
  }

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
      el.innerHTML = `<div class="label"></div><div class="scale" style="display:none"><span></span><i></i></div><div class="coords"></div><div class="legend"></div><div class="msg"></div>`
        + (this.showControls ? `<button class="btn" title="Display settings">${SVG_GEAR}</button>` : '');
      p.el = el;
      p.$label = el.querySelector('.label');
      p.$scale = el.querySelector('.scale');
      p.$coords = el.querySelector('.coords');
      p.$legend = el.querySelector('.legend');
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
    this._cw = cw;
    this._ch = ch;
    if (this.$canvas.width !== cw || this.$canvas.height !== ch) { this.$canvas.width = cw; this.$canvas.height = ch; }
    if (this.$ov.width !== cw || this.$ov.height !== ch) { this.$ov.width = cw; this.$ov.height = ch; }
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
        // the coordinate readout is written at most once per frame (pointer events can arrive at 120–1000 Hz)
        this._coordsHit = hit;
        if (!this._coordsRaf) this._coordsRaf = requestAnimationFrame(() => this._writeCoords());
        if (this._drawing && this._drawing.p === hit.p) this._drawMove(hit, e);
        if (this.hoverEvents && !ptrs.size && this.tool === 'pan') this._hoverAt(hit, e);
      } else if (this._hovered) this._hoverAt(null, e);
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
      if (drag.moved < 4 && now() - drag.t < 400 && e.button !== 2) this._click(p, e);
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
    root.addEventListener('pointerleave', (e) => { if (this._hovered) this._hoverAt(null, e); });
    root.addEventListener('contextmenu', async (e) => {
      if (!this.contextEvents) return;
      const hit = this._paneAt(e.clientX, e.clientY);
      if (!hit?.p.view) return;
      e.preventDefault();
      const [x, y] = this._toImage(hit.p, hit.sx, hit.sy);
      const h = await this.hitAt(hit.p, hit.sx, hit.sy, { server: false });
      this._emit('panecontext', { pane: hit.p.index, slide: hit.p.slide?.info?.id, x, y, clientX: e.clientX, clientY: e.clientY,
        object: h && { layer: h.layer, index: h.index, class: h.class } });
    });
    root.addEventListener('pointerup', up);
    root.addEventListener('pointercancel', up);
    root.addEventListener('pointerleave', () => this.panes.forEach((p) => { if (p.$coords && p._coordsTxt) { p._coordsTxt = ''; p.$coords.textContent = ''; } }));
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
      if (k === 'Escape') { this._drawing = null; if (this.tool === 'sam') this.samClear(); this._drawOverlay(); if (this.tool !== 'pan') return; }
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

  _writeCoords() {
    this._coordsRaf = 0;
    const hit = this._coordsHit;
    if (!hit?.p.view || !hit.p.$coords) return;
    const [ix, iy] = this._toImage(hit.p, hit.sx, hit.sy);
    const info = hit.p.slide?.info;
    const txt = info?.mpp ? `${(ix * info.mpp[0]).toFixed(1)}, ${(iy * info.mpp[1]).toFixed(1)} µm` : `${ix.toFixed(0)}, ${iy.toFixed(0)} px`;
    if (hit.p._coordsTxt !== txt) { hit.p._coordsTxt = txt; hit.p.$coords.textContent = txt; }
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

  /** The smallest visible object under a pane point (sx, sy: px from the pane's top-left), or null.
   *  server: false → only objects whose data is already in the browser (no request; for hover). */
  async hitAt(p, sx, sy, { server = true } = {}) {
    if (!p?.view || !p.slide) return null;
    const [x, y] = this._toImage(p, sx, sy);
    const tol = 5 / p.view.scale;
    const scaleDev = p.view.scale * this.dpr;
    let found = null;
    for (const L of p.slide.layers.values()) {
      if (!p.layerVisible(L) || L.state !== 'ready') continue;
      let h = L.hitInfo(x, y, tol, L.pyr ? L.pyrLevel(scaleDev) : 0);
      if (h === undefined) { if (!server) continue; h = await L.serverHit(x, y, tol); }
      if (h && (!found || h.size < found.size)) found = { ...h, L };
    }
    if (!found) return null;
    const c = found.L.classes[found.cls];
    return { layer: found.L.name, index: found.f, class: c?.name, color: c?.color, cx: found.cx, cy: found.cy, size: found.size, L: found.L };
  }

  _hoverAt(hit, e) {
    if (!hit) { if (this._hovered) { this._hovered = null; this._emit('objecthover', null); } return; }
    this._hoverPt = { hit, x: e.clientX, y: e.clientY };
    if (this._hoverT) return;
    this._hoverT = setTimeout(async () => {
      this._hoverT = 0;
      const { hit: h0, x, y } = this._hoverPt;
      const h = await this.hitAt(h0.p, h0.sx, h0.sy, { server: false });
      const key = h ? `${h.layer}|${h.index}` : null;
      if (key === (this._hovered?.key ?? null)) { if (h) this._emit('objecthover', { ...this._hovered.d, clientX: x, clientY: y }); return; }
      const d = h && { pane: h0.p.index, layer: h.layer, index: h.index, class: h.class, color: h.color, size: h.size, clientX: x, clientY: y };
      this._hovered = h ? { key, d } : null;
      this._emit('objecthover', d);
    }, 50);
  }

  /** Make a layer's objects pulse for a moment (the app does this when its row is hovered). */
  flashLayer(pane, layer, ms = 900) {
    for (const p of this._panes(pane)) {
      const L = p.slide?.layers.get(layer);
      if (L) { L._flashUntil = now() + ms; L._flashMs = ms; }
    }
    this.requestRender();
  }

  _layerAlpha(L, st) {
    const a = st.opacity ?? 1;
    if (!L._flashUntil) return a;
    const t = now();
    if (t >= L._flashUntil) { L._flashUntil = 0; return a; }
    this.requestRender();
    const k = 1 - (L._flashUntil - t) / L._flashMs;   // 0 → 1: two pulses
    return a * (0.25 + 0.75 * Math.abs(Math.cos(k * Math.PI * 2)));
  }

  async _click(p, e) {
    const hit = this._paneAt(e.clientX, e.clientY);
    if (!hit || hit.p !== p) return;
    const [x, y] = this._toImage(p, hit.sx, hit.sy);
    const detail = { pane: p.index, slide: p.slide?.info?.id, x, y, button: e.button };
    if (p.slide?.info?.mpp) { detail.x_um = x * p.slide.info.mpp[0]; detail.y_um = y * p.slide.info.mpp[1]; }
    this._emit('paneclick', detail);
    const found = await this.hitAt(p, hit.sx, hit.sy);
    p.selection = found ? { layer: found.layer, f: found.index, cx: found.cx, cy: found.cy, size: found.size } : null;
    this.requestRender();
    if (!found) return;
    const L = found.L, idx = found.index;
    const d = { ...detail, layer: L.name, index: idx, class: found.class,
      centroid: [found.cx, found.cy], size: found.size };
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
    if (t === 'sam') {
      this._drawing = { p, pts: [pt], cur: pt, kind: 'sam', dragging: true, neg: e.shiftKey || e.altKey, sx: hit.sx, sy: hit.sy };
      this._drawOverlay();
      return;
    }
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
    if (d.kind === 'sam') {
      this._drawing = null;
      const px = Math.hypot(b[0] - a[0], b[1] - a[1]) * d.p.view.scale;
      if (px < 5) this._samPrompt(d.p, { point: a, label: d.neg ? 0 : 1 });
      else this._samPrompt(d.p, { box: [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])] });
      this._drawOverlay();
      return;
    }
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
    if (this.tool === 'sam') { this.samAccept(); return; }
    const d = this._drawing;
    if (!d || (d.kind !== 'polygon' && d.kind !== 'line')) return;
    const pts = d.pts.filter((q, i) => i === 0 || q[0] !== d.pts[i - 1][0] || q[1] !== d.pts[i - 1][1]);
    this._drawing = null;
    if (d.kind === 'polygon' && pts.length >= 3) this._drawEmit(d.p, { type: 'Polygon', coordinates: [[...pts, pts[0]]] });
    if (d.kind === 'line' && pts.length >= 2) this._drawEmit(d.p, { type: 'LineString', coordinates: pts });
    this._drawOverlay();
  }
  async _drawEmit(p, geometry, extra = null) {
    this._emit('draw', { pane: p.index, slide: p.slide?.info?.id, url: p.slide?.url, tool: this.tool, geometry, class: this.drawClass || this.drawLayer, properties: extra });
    if (this.drawLayer && p.slide) {
      try {
        await fetch(`${p.slide.url}/annotations/add?layer=${encodeURIComponent(this.drawLayer)}`, {
          method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: 'Feature', geometry, properties: { classification: { name: this.drawClass || this.drawLayer, ...(this.drawColor ? { color: this.drawColor } : {}) }, ...(extra || {}) } }),
        });
        await p.slide.refreshAnnotations();
      } catch (_) { /* surfaced through the draw event */ }
    }
  }

  // ---------------------------------------------------------------- SAM (segment from clicks / boxes)
  async _samPrompt(p, prompt) {
    if (!p.slide?.info || !p.view) return;
    // the session freezes the pane's view rectangle: the server encodes it once, every further prompt is cheap
    if (!p.sam || p.sam.slide !== p.slide) {
      const hw = p.rect.w / 2 / p.view.scale, hh = p.rect.h / 2 / p.view.scale;
      p.sam = { slide: p.slide, rect: [p.view.cx - hw, p.view.cy - hh, p.view.cx + hw, p.view.cy + hh],
        points: [], labels: [], box: null, polys: [], seq: 0 };
    }
    const s = p.sam;
    if (prompt.box) s.box = prompt.box;
    else { s.points.push(prompt.point); s.labels.push(prompt.label); }
    const seq = ++s.seq;
    s.busy = true;
    this._emit('samstart', { pane: p.index, slide: p.slide.info.id, points: s.points.length, box: !!s.box });
    this._drawOverlay();
    try {
      const r = await fetch(`${p.slide.url}/sam`, {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rect: s.rect, points: s.points, labels: s.labels, box: s.box }),
      });
      const j = await r.json();
      if (p.sam !== s || seq !== s.seq) return;           // superseded by a newer prompt
      if (!r.ok) throw new Error(j.detail || r.statusText);
      s.polys = j.polygons || [];
      s.info = j;
      this._emit('sam', { pane: p.index, slide: p.slide.info.id, polygons: s.polys, score: j.score, backend: j.backend,
        encode_ms: j.encode_ms, decode_ms: j.decode_ms });
    } catch (err) {
      this._emit('sam', { pane: p.index, slide: p.slide.info.id, error: String(err.message || err) });
    } finally {
      if (p.sam === s && seq === s.seq) s.busy = false;
      this._ovDirty = true;
      this._drawOverlay();
    }
  }

  /** Save the current SAM mask(s) of a pane (default: every pane with one) to the draw layer, with the draw class. */
  async samAccept(pane = null) {
    const ps = pane == null ? this.panes.filter((p) => p.sam?.polys?.length) : this._panes(pane);
    for (const p of ps) {
      const s = p.sam;
      if (!s?.polys?.length) continue;
      p.sam = null;
      for (const ring of s.polys) {
        await this._drawEmit(p, { type: 'Polygon', coordinates: [[...ring, ring[0]]] },
          { source: 'sam', score: s.info?.score, backend: s.info?.backend });
      }
    }
    this._ovDirty = true;
    this._drawOverlay();
  }

  samClear(pane = 'all') { for (const p of this._panes(pane)) p.sam = null; this._ovDirty = true; this._drawOverlay(); }

  // ---------------------------------------------------------------- frame loop
  requestRender() {
    if (!this._raf && this.gl && this.isConnected) { this._reqAt = now(); this._raf = requestAnimationFrame((t) => this._frame(t)); }
  }

  _frame() {
    this._raf = 0;
    if (this._lost) return;
    const t = now();
    try { this._frameBody(t); } finally { if (this.jsFrameMs) { this.jsFrameMs.push(now() - t); if (this.jsFrameMs.length > 20000) this.jsFrameMs.length = 0; } }
    this._probeCost(t);
  }

  // What a frame really costs the main thread: our JS *plus* the browser's style/paint/commit that follow the rAF
  // callback (without a GPU, Chrome reads the whole canvas back during commit — often far more than our JS). A
  // message posted now runs right after that work.
  _probeCost(t) {
    if (!this._mc) {
      this._mc = new MessageChannel();
      this._mc.port1.onmessage = (e) => { const c = now() - e.data; this._frameCost = this._frameCost == null ? c : 0.7 * this._frameCost + 0.3 * c; };
    }
    this._mc.port2.postMessage(t);
  }

  // A tile (or overlay tile) arrived. Normally: draw it on the next frame. At rest on a machine where frames are
  // expensive (≥ 40 ms, e.g. no GPU), batch arrivals into one redraw every ~2 frame costs (≤ 250 ms) so a stream of
  // tiles doesn't keep the main thread saturated — the user's next drag or wheel isn't stuck behind redraws.
  _arrived() {
    if (this._raf) return;
    const fc = this._frameCost || 0;
    if (fc < 40 || !this.lod.adaptive || this.panes.some((p) => p.view && this._isMoving(p))) { this.requestRender(); return; }
    const wait = Math.min(250, 2 * fc) - (now() - (this._prevFrameAt || 0));
    if (wait <= 0) { this.requestRender(); return; }
    if (!this._arriveTimer) this._arriveTimer = setTimeout(() => { this._arriveTimer = 0; this.counters.coalesced = (this.counters.coalesced || 0) + 1; this.requestRender(); }, wait);
  }

  _frameBody(t) {
    // frame cost for adaptive resolution: the interval between back-to-back frames, or — after an idle gap, e.g.
    // input-driven frames during a drag — how long the browser took to give us the frame we asked for
    const gap = this._prevFrameAt ? t - this._prevFrameAt : 1e9;
    const sample = gap < 250 ? gap : this._reqAt ? t - this._reqAt : 16;
    this._prevFrameAt = t;
    this._frameMs = 0.75 * (this._frameMs || 16) + 0.25 * Math.min(500, Math.max(sample, 0));
    const dt = this._lastT ? Math.min(64, t - this._lastT) : 16;
    this._lastT = t;
    let again = false;
    for (const p of this.panes) if (this._animate(p, dt)) again = true;
    for (const p of this.panes) this._trackMotion(p, t);
    const moreUploads = this._processUploads();
    this.frame++;
    if (this._legendDirty) { this._legendDirty = false; this._legends(); }
    this._draw();
    this._schedule();
    this._evict();
    this._emitViews(t);
    if (again) this.requestRender();
    else {
      this._lastT = 0;
      if (moreUploads) this._arrived();   // back-to-back frames normally; batched at rest when frames are expensive
    }
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

  // A pane "moves" while animating/flinging or while its view changed in the last 150 ms (pans driven from
  // outside). When it comes to rest one more frame is needed to switch to full-resolution fetching.
  _trackMotion(p, t) {
    if (!p.view) return;
    const v = p.view, k = `${v.cx}|${v.cy}|${v.scale}|${p.rect.w}|${p.rect.h}`;
    if (k !== p._viewKey) {
      p._viewKey = k;
      p.movedAt = t;
      if (this.lod.motionBias > 1 || this.lod.prefetch || this.lod.adaptive) this._wakeAt(t + (this.lod.prefetch ? 420 : 170));
    }
  }

  // One timer for the whole grid: render once more when the latest motion has settled (switch to full resolution)
  _wakeAt(t) {
    this._restAt = Math.max(this._restAt || 0, t);
    if (this._restTimer) return;
    const fire = () => {
      const left = this._restAt - now();
      if (left > 2) { this._restTimer = setTimeout(fire, left); return; }
      this._restTimer = 0;
      this.requestRender();
    };
    this._restTimer = setTimeout(fire, Math.max(0, this._restAt - now()));
  }

  _isMoving(p, t = now()) { return !!(p.target || p.vel) || t - (p.movedAt || 0) < 150; }

  _emitViews(t) {
    if (t - (this._lastEmit || 0) < 120) { if (!this._emitPending) { this._emitPending = true; setTimeout(() => { this._emitPending = false; this._emitViews(now()); }, 130); } return; }
    this._lastEmit = t;
    let changed = false;
    for (const p of this.panes) {
      if (!p.view) continue;
      const key = `${p.view.cx.toFixed(1)}|${p.view.cy.toFixed(1)}|${p.view.scale.toPrecision(5)}|${p.rect.w}|${p.rect.h}`;
      if (key === p.lastEmit) continue;
      p.lastEmit = key;
      changed = true;
      p._detail = this._viewDetail(p);
      if (this._hasViewListeners !== false) this._emit('viewchange', p._detail);
      this._updateScaleBar(p);
    }
    // one snapshot of every pane per emit: apps (NiceGUI) listen to this instead of 16 per-pane events, and a
    // throttled snapshot can never lose a pane's latest view
    if (changed) this._emit('views', { views: this.panes.map((p) => (p.view ? p._detail || this._viewDetail(p) : null)) });
  }

  _updateScaleBar(p) {
    const i = p.slide?.info;
    if (!p.$scale) return;
    if (!i?.mpp || !p.view) { if (p._scaleKey !== 'none') { p._scaleKey = 'none'; p.$scale.style.display = 'none'; } return; }
    const umPerPx = i.mpp[0] / p.view.scale;
    const target = 110 * umPerPx;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const nice = [1, 2, 5, 10].map((m) => m * pow).filter((v) => v <= target).pop() || pow;
    const w = `${Math.round(nice / umPerPx)}px`, txt = nice >= 1000 ? `${nice / 1000} mm` : nice >= 1 ? `${nice} µm` : `${nice * 1000} nm`;
    if (p._scaleKey === w + txt) return;           // unchanged (panning): no DOM writes, no style/layout
    p._scaleKey = w + txt;
    p.$scale.style.display = '';
    (p._scaleI ||= p.$scale.querySelector('i')).style.width = w;
    (p._scaleS ||= p.$scale.querySelector('span')).textContent = txt;
  }

  // ---------------------------------------------------------------- tiles: geometry
  _level(info, scaleDev) {
    // full resolution: the coarsest level whose texels are ≤ ~1 device pixel (× quality bias)
    return this._levelFor(info, scaleDev, this.quality);
  }

  // coarsest level whose texels are at most `texel` device px
  _levelFor(info, scaleDev, texel) {
    const want = texel / scaleDev;
    let L = 0;
    for (let i = 0; i < info.levels.length; i++) if (info.levels[i].downsample <= want * 1.0001) L = i;
    return L;
  }

  // finest level that may be drawn (if loaded): texels down to `lod.minify` device px, never coarser than `full`
  _drawFloor(info, scaleDev, full) {
    if (!this.lod.quadtree || this.lod.minify >= 1) return full;
    const want = this.quality * this.lod.minify / scaleDev;
    let L = full;
    while (L > 0 && info.levels[L - 1].downsample >= want * 0.9999) L--;
    return L;
  }

  // tile footprint in level-0 px
  _foot(info, L, tx, ty) {
    const lv = info.levels[L], ts = info.tile_size;
    const tw = Math.min(ts, lv.width - tx * ts), th = Math.min(ts, lv.height - ty * ts);
    return [tx * ts * lv.sx, ty * ts * lv.sy, (tx * ts + tw) * lv.sx, (ty * ts + th) * lv.sy];
  }

  // Quadtree over the pyramid restricted to rect `r` (the 3-D view frustum: x, y and level). Calls
  // emit(L, tx, ty, subRect) for the finest loaded tile covering each part of r, levels top..floor; parts no
  // loaded tile covers are left out. Returns nothing; coarse tiles are only drawn where finer ones are missing.
  _cover(p, info, floor, r, chans, emit) {
    const top = info.levels.length - 1;
    const visit = (L, tx, ty, rect) => { // → list of uncovered sub-rects, or true if all of rect is uncovered
      const ready = this._ready(p, L, tx, ty, chans);
      if (L <= floor) {
        if (ready) { emit(L, tx, ty, rect); return null; }
        return true;
      }
      const rg = this._tileRange(info, L - 1, rect);
      let holes = null, all = true, n = 0;
      if (rg) {
        for (let y = rg.y0; y <= rg.y1; y++) for (let x = rg.x0; x <= rg.x1; x++) {
          const f = this._foot(info, L - 1, x, y);
          const c = [Math.max(f[0], rect[0]), Math.max(f[1], rect[1]), Math.min(f[2], rect[2]), Math.min(f[3], rect[3])];
          if (c[2] <= c[0] || c[3] <= c[1]) continue;
          n++;
          const h = visit(L - 1, x, y, c);
          if (h === true) (holes ||= []).push(c);
          else { all = false; if (h) (holes ||= []).push(...h); }
        }
      }
      if (!n) all = true; // (degenerate: no children overlap)
      if (all) { if (ready) { emit(L, tx, ty, rect); return null; } return true; }
      if (!holes) return null;
      if (ready) { for (const h of holes) emit(L, tx, ty, h); return null; }
      return holes;
    };
    const rg = this._tileRange(info, top, r);
    if (!rg) return;
    for (let y = rg.y0; y <= rg.y1; y++) for (let x = rg.x0; x <= rg.x1; x++) {
      const f = this._foot(info, top, x, y);
      const c = [Math.max(f[0], r[0]), Math.max(f[1], r[1]), Math.min(f[2], r[2]), Math.min(f[3], r[3])];
      if (c[2] > c[0] && c[3] > c[1]) visit(top, x, y, c);
    }
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

  // Tile key strings are made once per tile and cached by a numeric id: the scheduler and the draw loop look up
  // thousands of tiles per frame, and building strings each time fed the garbage collector (pauses up to 127 ms)
  _key(s, L, x, y, c) {
    // exact in a double up to 2^53: level < 64, x / y < 65536, channel < 4095 → < 2^50
    const n = ((L * 65536 + x) * 65536 + y) * 4096 + (c == null ? 0 : c + 1);
    let m = s._keys;
    if (!m) m = s._keys = new Map();
    let k = m.get(n);
    if (k === undefined) {
      if (m.size > 200000) m.clear();
      k = `${s.url}|${L}|${x}|${y}|${c ?? 'rgb'}`;
      m.set(n, k);
    }
    return k;
  }

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
  // Adaptive resolution: the fraction of the canvas resolution to render at this frame (1 = full). Only while
  // something moves and frames are slower than ~20 ms; aims at ~16 ms frames (cost ∝ scale²). Snaps to 1, ¾ or ½
  // so the backing store is only reallocated when the step changes; full resolution as soon as motion stops.
  _renderScale() {
    if (!this.lod.adaptive) return (this._rs = 1);
    const t = now();
    if (!this.panes.some((p) => p.view && this._isMoving(p, t))) { this._rsc = 1; return (this._rs = 1); }
    // frame time from the frame clock, or the measured main-thread cost when that's worse (input-driven frames)
    const fm = Math.max(this._frameMs || 16, this._frameCost || 0), cur = this._rsc || 1;
    if (fm < 20 && cur >= 1) return (this._rs = 1);
    const want = clamp(cur * Math.sqrt(16.7 / Math.max(fm, 1)), 0.5, 1);
    const next = (this._rsc = clamp(0.6 * cur + 0.4 * want + (fm < 12 ? 0.05 : 0), 0.5, 1));   // smoothed; recovers when fast
    return (this._rs = next >= 0.9 ? 1 : next >= 0.65 ? 0.75 : 0.5);
  }

  _draw() {
    const CW = this._cw || this.$canvas.width, CH = this._ch || this.$canvas.height, dpr0 = this.dpr || 1;
    const rs = this._renderScale();
    // Lower resolution = a smaller WebGL backing store, stretched by CSS. (Rendering into a small offscreen buffer
    // and blitting up would cut our fill cost but not the browser's: without a GPU, Chrome's software compositor
    // reads the whole drawing buffer back every frame, and that readback ∝ backing-store pixels dominated frames.)
    const W = rs < 1 ? Math.max(1, Math.round(CW * rs)) : CW, H = rs < 1 ? Math.max(1, Math.round(CH * rs)) : CH;
    if (this.$canvas.width !== W || this.$canvas.height !== H) {
      this.$canvas.width = W;
      this.$canvas.height = H;
      this.fx.resizes = (this.fx.resizes || 0) + 1;
    }
    if (rs < 1) {
      this.dpr = dpr0 * W / CW;
      this.fx.lowres = (this.fx.lowres || 0) + 1;
    }
    try {
      this._drawScene(W, H);
    } finally {
      this.dpr = dpr0;
    }
    this._drawOverlay();
  }

  _drawScene(W, H) {
    const gl = this.gl, dpr = this.dpr || 1;
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, W, H);
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
      this.fx.area += w * h;
      this._drawTiles(p);
      this._drawAnnotations(p);
    }
    gl.disable(gl.SCISSOR_TEST);
    this.fx.frames++;
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
    const quad = this.lod.quadtree;
    // legacy: find the finest level ≥ target that fully covers the view; draw from there down to target (painter's)
    const levels = [];
    for (let L = target; !quad && L < info.levels.length; L++) {
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
      const st = p.stain;
      gl.uniform1i(prog.u.u_stain, st ? st.mode : 0);
      if (st) {
        gl.uniform1i(prog.u.u_stainSel, st.sel);
        gl.uniformMatrix3fv(prog.u.u_minv, false, st.minvCols);
        gl.uniformMatrix3fv(prog.u.u_m, false, st.mRowsAsCols);
        gl.uniform1f(prog.u.u_stainScale, st.scale);
        gl.uniform3fv(prog.u.u_col0, st.col0);
        gl.uniform3fv(prog.u.u_col1, st.col1);
      }
    }
    const fx = this.fx, area2 = scaleDev * scaleDev;
    if (quad) {
      // quadtree: the finest loaded tile per region, coarse tiles only through the holes (sub-rect draws)
      const ov = info.overlap || 0, ts = info.tile_size;
      const floor = this._drawFloor(info, scaleDev, target);
      this._cover(p, info, floor, vis, chans, (L, tx, ty, r) => {
        const lv = info.levels[L];
        const tw = Math.min(ts, lv.width - tx * ts), th = Math.min(ts, lv.height - ty * ts);
        let t0 = null;
        chans.forEach((c, i) => {
          const t = this.tex.get(this._key(s, L, tx, ty, c));
          t.last = this.frame;
          t.drawn = true;
          if (!t0) t0 = t;
          gl.activeTexture(gl.TEXTURE0 + i);
          gl.bindTexture(gl.TEXTURE_2D, t.tex);
        });
        const ox = t0.w > tw && tx > 0 ? Math.min(ov, t0.w - tw) : 0;
        const oy = t0.h > th && ty > 0 ? Math.min(ov, t0.h - th) : 0;
        const fx0 = tx * ts * lv.sx, fy0 = ty * ts * lv.sy, fw = tw * lv.sx, fh = th * lv.sy;
        const u0 = (r[0] - fx0) / fw, v0 = (r[1] - fy0) / fh, u1 = (r[2] - fx0) / fw, v1 = (r[3] - fy0) / fh;
        gl.uniform4f(prog.u.u_uv, (ox + u0 * tw) / t0.w, (oy + v0 * th) / t0.h, (u1 - u0) * tw / t0.w, (v1 - v0) * th / t0.h);
        gl.uniform4f(prog.u.u_rect, r[0], r[1], r[2] - r[0], r[3] - r[1]);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        fx.draws++;
        fx.fill += (r[2] - r[0]) * (r[3] - r[1]) * area2;
      });
      gl.activeTexture(gl.TEXTURE0);
      return;
    }
    for (let k = levels.length - 1; k >= 0; k--) {
      const [L, rg] = levels[k];
      const lv = info.levels[L], ts = info.tile_size;
      for (let ty = rg.y0; ty <= rg.y1; ty++) {
        for (let tx = rg.x0; tx <= rg.x1; tx++) {
          if (!this._ready(p, L, tx, ty, chans)) continue;
          const tw = Math.min(ts, lv.width - tx * ts), th = Math.min(ts, lv.height - ty * ts);
          let t0 = null;
          chans.forEach((c, i) => {
            const t = this.tex.get(this._key(s, L, tx, ty, c));
            t.last = this.frame;
            t.drawn = true;
            if (!t0) t0 = t;
            gl.activeTexture(gl.TEXTURE0 + i);
            gl.bindTexture(gl.TEXTURE_2D, t.tex);
          });
          // A remote tile larger than its footprint carries overlap on interior edges and/or padding
          // past the image edge: sample only the footprint.
          const ov = info.overlap || 0;
          const ox = t0.w > tw && tx > 0 ? Math.min(ov, t0.w - tw) : 0;
          const oy = t0.h > th && ty > 0 ? Math.min(ov, t0.h - th) : 0;
          gl.uniform4f(prog.u.u_uv, ox / t0.w, oy / t0.h, tw / t0.w, th / t0.h);
          gl.uniform4f(prog.u.u_rect, tx * ts * lv.sx, ty * ts * lv.sy, tw * lv.sx, th * lv.sy);
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
          fx.draws++;
          { const x0 = Math.max(vis[0], tx * ts * lv.sx), x1 = Math.min(vis[2], (tx * ts + tw) * lv.sx);
            const y0 = Math.max(vis[1], ty * ts * lv.sy), y1 = Math.min(vis[3], (ty * ts + th) * lv.sy);
            if (x1 > x0 && y1 > y0) fx.fill += (x1 - x0) * (y1 - y0) * area2; }
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
    const order = [...s.layers.values()].filter((L) => L.state === 'ready' && p.layerVisible(L))
      .sort((a, b) => (a.style.z ?? 0) - (b.style.z ?? 0) || a.medSize - b.medSize);
    for (const L of order) {
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
      const alpha = this._layerAlpha(L, st);
      if (L.client === 'ready') {
        const outlines = L.pyrLevel(scaleDev) === 0 && L.medSize * p.view.scale >= minPx;
        this._drawPyramid(p, L, vis, scaleDev, alpha, outlines);
        if (outlines) this._drawLocalShapes(p, L, vis, scaleDev, alpha);
        continue;
      }
      if (L.client) continue;   // client mode still downloading / building
      if (L.pyr) this._drawPyramid(p, L, vis, scaleDev, alpha, ptsRanges.length === 0 && detail.length > 0);
      // dots (zoomed-out objects, and point features everywhere)
      if (!L.pyr && (ptsRanges.length || kindRanges.length)) {
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
        if (st.overview !== 'none') run(ptsRanges, 0);
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
      if (detail.length && (st.line_width ?? 1.5) > 0) {
        const pr = this.prog.seg;
        gl.useProgram(pr.p);
        gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
        gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
        gl.uniform1f(pr.u.u_w, Math.max(1, (st.line_width ?? 1.5) * this.dpr));
        gl.uniform1f(pr.u.u_alpha, alpha);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
        for (const d of detail) if (d.nSeg) { gl.bindVertexArray(d.vaoSeg); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, d.nSeg); }
      }
      // parametric cells (bbox ellipses / rects): outline + fill in one instanced pass
      if (detail.some((d) => d.nShp)) {
        const pr = this.prog.shp;
        gl.useProgram(pr.p);
        gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
        gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
        gl.uniform1f(pr.u.u_w, Math.max(1, (st.line_width ?? 1.5) * this.dpr));
        gl.uniform1f(pr.u.u_fill, fo);
        gl.uniform1f(pr.u.u_alpha, alpha);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
        for (const d of detail) if (d.nShp) this._drawShapes(d, d.nShp, vis, p.view.scale);
      }
      // point features on top of detailed buckets
      if (!L.pyr && kindRanges.length && detail.length) {
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

  // Object pyramid: the level for this zoom where loaded, the coarsest level underneath where not.
  // allDetailed → every visible bucket shows outlines, so raw dots are only drawn for point features.
  _drawPyramid(p, L, vis, scaleDev, alpha, allDetailed) {
    const gl = this.gl, st = L.style, lvl = L.pyrLevel(scaleDev), top = L.pyr.levels - 1;
    const tiles = (lv) => L.pyrTiles(lv, vis).map(([tx, ty]) => L.ptiles.get(L.pkey(lv, tx, ty)));
    const target = tiles(lvl);
    const list = target.every(Boolean) ? target : [...(lvl !== top ? tiles(top) : []), ...target];
    if (st.overview === 'none' && lvl !== 0) return;
    const pr = this.prog.ppts;
    gl.useProgram(pr.p);
    // integer inputs a tile leaves unbound (raw: no histogram; aggregated: no kind) need integer defaults
    gl.vertexAttribI4ui(3, 0, 0, 0, 0); gl.vertexAttribI4ui(4, 0, 0, 0, 0); gl.vertexAttribI4ui(5, 0, 0, 0, 0);
    gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
    gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
    gl.uniform2f(pr.u.u_ps, 1.5 * this.dpr, (st.point_size ?? 4) * this.dpr);
    gl.uniform1f(pr.u.u_alpha, alpha * 0.9);
    gl.uniform1i(pr.u.u_onlyPoints, allDetailed ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
    const slots = new Int32Array(8);
    L.pyr.slots.forEach((c, j) => { slots[j] = c >= 0 ? c : (L.pyr.other_class ?? 0); });
    gl.uniform1iv(pr.u.u_slots, slots);
    gl.uniform1i(pr.u.u_vis, L.slotMask());
    for (const t of list) {
      if (!t || !t.n) continue;
      t.last = this.frame;
      gl.uniform3f(pr.u.u_tile, t.x0, t.y0, t.span);
      gl.uniform1i(pr.u.u_mode, t.raw ? 0 : (t.hist ? 1 : 2));
      gl.uniform1f(pr.u.u_bin, 2 ** t.k);
      gl.bindVertexArray(t.vao);
      gl.drawArrays(gl.POINTS, 0, t.n);
    }
  }

  // Instanced shapes of one bucket / tile: only the grid rows (column spans) that intersect the view.
  // WebGL2 has no base-instance draw, so each row re-points the instance attributes at its first instance.
  _drawShapes(d, n, vis, scale) {
    const gl = this.gl, g = d.shpGrid;
    gl.bindVertexArray(d.vaoShp);
    const at = (a) => {    // point the instance attributes at instance a (the VAO remembers it)
      if ((d._at || 0) === a) return;
      const [bS, bC, bK] = d.shpBufs;
      gl.bindBuffer(gl.ARRAY_BUFFER, bS); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, a * 16);
      gl.bindBuffer(gl.ARRAY_BUFFER, bC); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_SHORT, 0, a * 2);
      gl.bindBuffer(gl.ARRAY_BUFFER, bK); gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_BYTE, 0, a);
      d._at = a;
    };
    if (!g) { if (d.shpBufs) at(0); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n); this.fx.shapes = (this.fx.shapes || 0) + n; return; }
    const pad = g.pad + 4 / scale, G = g.G;
    const c0 = Math.floor((vis[0] - pad - g.x0) / g.cw), c1 = Math.floor((vis[2] + pad - g.x0) / g.cw);
    const r0 = Math.floor((vis[1] - pad - g.y0) / g.ch), r1 = Math.floor((vis[3] + pad - g.y0) / g.ch);
    if (c1 < 0 || r1 < 0 || c0 >= G || r0 >= G) return;
    const a0 = Math.max(0, c0), a1 = Math.min(G - 1, c1);
    for (let r = Math.max(0, r0); r <= Math.min(G - 1, r1); r++) {
      const a = g.off[r * G + a0], b = g.off[r * G + a1 + 1];
      if (b <= a) continue;
      at(a);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b - a);
      this.fx.shapes = (this.fx.shapes || 0) + (b - a);
    }
  }

  _drawLocalShapes(p, L, vis, scaleDev, alpha) {
    const gl = this.gl, st = L.style, pr = this.prog.shp;
    gl.useProgram(pr.p);
    gl.uniform3f(pr.u.u_view, p.view.cx, p.view.cy, scaleDev);
    gl.uniform2f(pr.u.u_vp, p.dev.w, p.dev.h);
    gl.uniform1f(pr.u.u_w, Math.max(1, (st.line_width ?? 1.5) * this.dpr));
    gl.uniform1f(pr.u.u_fill, (st.fill_opacity ?? 0) * alpha);
    gl.uniform1f(pr.u.u_alpha, alpha);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, L.ctex); gl.uniform1i(pr.u.u_ctex, 0);
    for (const [tx, ty] of L.pyrTiles(0, vis)) {
      const t = L.ptiles.get(L.pkey(0, tx, ty));
      if (!t || !t.vaoShp || !t.n) continue;
      this._drawShapes(t, t.n, vis, p.view.scale);
    }
  }

  _drawOverlay() {
    const c = this.ctx2d;
    if (!c) return;
    // nothing selected and nothing being drawn: leave the (already clear) 2-D layer alone — clearing it every frame
    // made the compositor re-upload a full-screen canvas per frame
    const need = !!this._drawing || this.panes.some((p) => (p.selection && p.selection.cx != null) || p.sam);
    if (!need && !this._ovDirty) return;
    this._ovDirty = need;
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
      if (p.selection && p.selection.cx != null) {
        const L = p.slide?.layers.get(p.selection.layer);
        if (L?.state === 'ready') {
          const [sx, sy] = toS(p.selection.cx, p.selection.cy);
          const r = Math.max(6 * dpr, (p.selection.size / 2) * p.view.scale * dpr * 1.15);
          c.lineWidth = 2 * dpr;
          c.strokeStyle = '#fff';
          c.setLineDash([5 * dpr, 4 * dpr]);
          c.beginPath(); c.arc(sx, sy, r, 0, Math.PI * 2); c.stroke();
          c.setLineDash([]);
        }
      }
      if (p.sam) {
        const s = p.sam;
        c.lineWidth = 2 * dpr;
        c.strokeStyle = s.busy ? 'rgba(94,208,255,.55)' : '#5ed0ff';
        c.fillStyle = 'rgba(94,208,255,.22)';
        for (const ring of s.polys || []) {
          c.beginPath();
          ring.forEach((q, i) => { const [x, y] = toS(q[0], q[1]); i ? c.lineTo(x, y) : c.moveTo(x, y); });
          c.closePath(); c.fill(); c.stroke();
        }
        if (s.box) {
          const [x0, y0] = toS(s.box[0], s.box[1]), [x1, y1] = toS(s.box[2], s.box[3]);
          c.setLineDash([6 * dpr, 4 * dpr]); c.strokeStyle = '#ffeb3b'; c.strokeRect(x0, y0, x1 - x0, y1 - y0); c.setLineDash([]);
        }
        s.points.forEach((q, i) => {
          const [x, y] = toS(q[0], q[1]);
          c.fillStyle = s.labels[i] ? '#3ddc84' : '#ff5252';
          c.strokeStyle = '#000';
          c.beginPath(); c.arc(x, y, 5 * dpr, 0, Math.PI * 2); c.fill(); c.stroke();
        });
      }
      const d = this._drawing;
      if (d && d.kind === 'sam' && d.p === p) {
        const [x0, y0] = toS(...d.pts[0]), [x1, y1] = toS(...d.cur);
        c.lineWidth = 1.5 * dpr; c.setLineDash([6 * dpr, 4 * dpr]); c.strokeStyle = '#ffeb3b';
        c.strokeRect(x0, y0, x1 - x0, y1 - y0); c.setLineDash([]);
      } else if (d && d.p === p) {
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
    const add = (key, prio, url, kind, meta, bg = false) => {
      const w = want.get(key);
      if (!w || w.prio > prio) want.set(key, { prio, url, kind, meta, bg });
    };
    for (const p of this.panes) {
      const s = p.slide, info = s?.info;
      if (!info || !p.view || p.rect.w <= 0) continue;
      const chans = this._visibleChannels(p);
      const scaleDev = p.view.scale * this.dpr;
      const full = this._level(info, scaleDev);
      const lod = this.lod, tNow = now();
      const moving = this._isMoving(p, tNow);
      // while moving: fetch a coarser level (texels ≤ motionBias device px) first; full resolution in the background
      const target = moving && lod.motionBias > 1 ? this._levelFor(info, scaleDev, this.quality * lod.motionBias) : full;
      const top = info.levels.length - 1;
      const vis = this._visRect(p);
      const cx = p.view.cx, cy = p.view.cy, diag = Math.hypot(vis[2] - vis[0], vis[3] - vis[1]) || 1;
      const queueLevel = (L, rect, base, bg = false) => {
        const rg = this._tileRange(info, L, rect);
        if (!rg) return;
        const lv = info.levels[L], ts = info.tile_size;
        for (let ty = rg.y0; ty <= rg.y1; ty++) for (let tx = rg.x0; tx <= rg.x1; tx++) {
          const mx = (tx + 0.5) * ts * lv.sx, my = (ty + 0.5) * ts * lv.sy;
          const ddx = mx - cx, ddy = my - cy, dist = Math.sqrt(ddx * ddx + ddy * ddy) / diag;
          for (const c of chans) {
            const key = this._key(s, L, tx, ty, c);
            if (this.tex.has(key)) { this.tex.get(key).last = this.frame; continue; }
            add(key, base + dist, null, c == null ? 'image' : 'raw', { s, L, tx, ty, c }, bg);
          }
        }
      };
      // 1. whole coarsest level (instant overview, fallback for everything)
      queueLevel(top, [0, 0, info.width, info.height], 0);
      const ts0 = info.tile_size * info.levels[target].downsample;
      if (lod.destination && p.target) {
        // animating toward a known view: fetch for the destination; in-between views get spare capacity only
        const dv = p.target, dScale = dv.scale * this.dpr, dRect = this._visRect({ ...p, view: dv });
        const dFull = this._level(info, dScale);
        const dMove = lod.motionBias > 1 ? this._levelFor(info, dScale, this.quality * lod.motionBias) : dFull;
        if (dMove + 1 < top) queueLevel(dMove + 1, dRect, 0.5);
        queueLevel(dMove, dRect, 1);
        if (dFull !== dMove) queueLevel(dFull, dRect, BG_PRIO, lod.background);
        queueLevel(target, vis, 2.5);
      } else {
      // 2. parent of target (fast fallback), 3. target centre-out
      if (target + 1 < top) queueLevel(target + 1, vis, 0.5);
      queueLevel(target, vis, 1);
      // 4. margin ring + motion prediction
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
        const tl = lod.motionBias > 1 ? this._levelFor(info, p.target.scale * this.dpr, this.quality * lod.motionBias)
          : this._level(info, p.target.scale * this.dpr);
        if (tl < target) queueLevel(tl, this._visRect({ ...p, view: p.target }), 2);
      }
      // full resolution while moving: background priority (spare request slots only, see BG_PRIO)
      if (target !== full) queueLevel(full, vis, lod.background ? BG_PRIO : 1.5, lod.background);
      // at rest: prefetch the next finer level around the centre so the next zoom step is instant
      if (lod.prefetch && !moving && full > 0 && tNow - (p.movedAt || 0) > 400) {
        const hw = (vis[2] - vis[0]) / 4, hh = (vis[3] - vis[1]) / 4;
        queueLevel(full - 1, [cx - hw, cy - hh, cx + hw, cy + hh], BG_PRIO + 1, true);
      }
      }
      // annotations: overview per visible layer, detail buckets in view
      for (const L of s.layers.values()) {
        if (!p.layerVisible(L)) continue;
        // keep the overview wanted while it loads: cancelling it (it is not "in view" yet) could drop a response
        // already on its way and leave the layer stuck in 'loading'
        if (L.client === 'pending' && this.inflight.has(`cols|${s.url}|${L.name}|${L.meta.version}`)) {
          add(`cols|${s.url}|${L.name}|${L.meta.version}`, 0.3, s.annUrl('columns', L.name, L.meta.version), 'bin', { ann: L, columns: true });
        }
        if (L.state === 'idle' || L.state === 'error-retry' || L.state === 'loading') {
          add(`ann|${s.url}|${L.name}|${L.meta.version}`, 0.2, s.annUrl('overview', L.name, L.meta.version), 'bin', { ann: L });
          continue;
        }
        if (L.state !== 'ready') continue;
        if (L.client === 'pending') {
          add(`cols|${s.url}|${L.name}|${L.meta.version}`, 0.3, s.annUrl('columns', L.name, L.meta.version), 'bin', { ann: L, columns: true });
          continue;
        }
        if (L.client === 'building') continue;
        if (L.client === 'ready') {   // tiles come from the local pyramid: GPU upload only, no requests
          const lvl = L.pyrLevel(scaleDev), top = L.pyr.levels - 1;
          let made = 0;
          for (const lv of new Set([top, lvl])) for (const [tx, ty] of L.pyrTiles(lv, vis)) {
            const pk = L.pkey(lv, tx, ty), t = L.ptiles.get(pk);
            if (t) { t.last = this.frame; continue; }
            if (made++ < 24) L.localPTile(pk); else this._localPending = true;
          }
          continue;
        }
        if (L.pyr) {  // object pyramid: coarsest tile (instant overview) + the level for this zoom, centre-out
          const lvl = L.pyrLevel(scaleDev), top = L.pyr.levels - 1;
          const want = (lv, rect, base) => {
            for (const [tx, ty] of L.pyrTiles(lv, rect)) {
              const pk = L.pkey(lv, tx, ty);
              const t = L.ptiles.get(pk);
              if (t) { t.last = this.frame; continue; }
              const span = L.pyrSpan(lv);
              const dist = Math.hypot((tx + 0.5) * span - cx, (ty + 0.5) * span - cy) / diag;
              add(`pt|${s.url}|${L.name}|${L.meta.version}|${pk}`, base + Math.min(dist, 2), s.annUrl(`ptile/${lv}/${tx}/${ty}`, L.name, L.meta.version),
                'bin', { ann: L, ptile: pk });
            }
          };
          want(top, vis, 0.25);
          if (lvl !== top) want(lvl, vis, 1.1);
        }
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
    for (const [k, r] of this.inflight) if (!want.has(k)) { r.abort(); this.inflight.delete(k); this.counters.cancelled++; }
    const t = now();
    const todo = [];
    for (const [k, w] of want) {
      if (this.inflight.has(k) || this.queued.has(k)) continue;
      const f = this.failed.get(k);
      if (f && f > t) continue;
      todo.push([k, w]);
    }
    todo.sort((a, b) => a[1].prio - b[1].prio);
    const bgSlots = this.maxInflight - Math.max(1, Math.ceil(this.maxInflight / 3)); // keep ⅓ free for urgent tiles
    for (const [k, w] of todo) {
      if (this.inflight.size >= this.maxInflight) break;
      if (w.bg && this.inflight.size >= bgSlots) continue;
      // back-pressure: don't fetch faster than tiles can be uploaded to the GPU (slow/software GPUs)
      if (this.uploads.length >= Math.max(8, this.maxInflight)) break;
      this._start(k, w);
    }
    this._pendingWork = todo.length > 0;
    this._todo = this.lod.decoupled ? todo : null;
    this._bgSlots = bgSlots;
    if (this._localPending) { this._localPending = false; this.requestRender(); }   // more local overlay tiles to upload
  }

  // Between frames: when a request finishes, start the next wanted one (from the last schedule) right away.
  // Entries are at most one frame stale; the next _schedule cancels anything no longer in view.
  _pump() {
    const todo = this._todo;
    if (!todo) return;
    while (todo.length && this.inflight.size < this.maxInflight && this.uploads.length < Math.max(8, this.maxInflight)) {
      const [k, w] = todo.shift();
      if (this.inflight.has(k) || this.queued.has(k) || this.tex.has(k)) continue;
      const f = this.failed.get(k);
      if (f && f > now()) continue;
      if (w.bg && this.inflight.size >= this._bgSlots) continue;
      this._start(k, w);
    }
  }

  _start(key, w) {
    const m = w.meta;
    const url = w.url || m.s.tileUrl(m.L, m.tx, m.ty, m.c);
    const req = this.fetcher.load(url, w.kind);
    this.inflight.set(key, req);
    const C = this.counters;
    C.requests++;
    if (w.meta.ann && w.meta.bucket == null && !w.meta.ptile && !w.meta.columns) w.meta.ann.state = 'loading';
    req.promise.then((res) => {
      if (this.inflight.get(key) !== req) { res.bmp?.close?.(); return; }
      this.inflight.delete(key);
      this.failed.delete(key);
      this.failCounts.delete(key);
      C.completed++;
      C.bytes += res.bytes || 0;
      this.uploads.push({ key, res, meta: w.meta });
      this.queued.add(key);
      this._pump();
      this._arrived();
    }, (err) => {
      if (this.inflight.get(key) === req) this.inflight.delete(key);
      if (w.meta.ann && w.meta.bucket == null && !w.meta.ptile && !w.meta.columns && w.meta.ann.state === 'loading') w.meta.ann.state = 'error-retry';
      if (err.message === 'abort') C.aborted++; else C.failed++;
      if (err.message !== 'abort') {
        // per-tile exponential back-off (0.5 s, 1 s, 2 s … ≤ 30 s), and wake up when it expires:
        // an idle viewer only renders on demand, so without this the retry would never happen
        const n = (this.failCounts.get(key) || 0) + 1;
        this.failCounts.set(key, n);
        const delay = Math.min(30000, 500 * 2 ** Math.min(6, n - 1));
        this.failed.set(key, now() + delay);
        setTimeout(() => this.requestRender(), delay + 30);
      }
      if (err.message !== 'abort') this._pump();
      this.requestRender();
    });
  }

  _processUploads() {
    const gl = this.gl;
    // ms of GPU uploads per frame → no jank while panning; when frames are slow anyway (GPU-bound), spend up to
    // 20 % of the frame on uploads so the texture pipeline keeps up with the network
    // (at rest with batched redraws — expensive frames, see _arrived — take everything queued, up to 40 ms)
    const batched = this.lod.adaptive && (this._frameCost || 0) >= 40 && !this.panes.some((p) => p.view && this._isMoving(p));
    const budget = !this.lod.decoupled ? 6 : batched ? 40 : Math.min(40, Math.max(6, 0.2 * (this._frameMs || 16)));
    const deadline = now() + budget;
    let n = 0;
    while (this.uploads.length && (n < 2 || now() < deadline)) {
      const { key, res, meta } = this.uploads.shift();
      this.queued.delete(key);
      n++;
      if (meta.ann) {
        const L = meta.ann;
        if (L.state === 'dead') continue;
        try {
          if (meta.columns) { if (L.state === 'ready' && L.client === 'pending') L.ingestColumns(res.buf); }
          else if (meta.ptile) { if (L.state === 'ready') L.ingestPTile(meta.ptile, res.buf); }
          else if (meta.bucket == null) { L._ingest(res.buf); this._legendDirty = true; }
          else L.ingestBucket(meta.bucket, res.buf);
        } catch (e) { console.warn('wissi-viz annotation', e); L.state = 'error'; }
        continue;
      }
      const { s, L, tx, ty, c } = meta;
      if (!this.slides.has(s.url) || !s.info) { res.bmp?.close?.(); continue; }
      const lv = s.info.levels[L], ts = s.info.tile_size;
      const w = Math.min(ts, lv.width - tx * ts), h = Math.min(ts, lv.height - ty * ts);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      let bytes, texW = w, texH = h;
      if (c == null) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, res.bmp);
        texW = res.bmp.width;
        texH = res.bmp.height;
        bytes = texW * texH * 4;
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
      this.tex.set(key, { tex, bytes, last: this.frame, w: c == null ? texW : w, h: c == null ? texH : h, drawn: false });
      this.texBytes += bytes;
      this.counters.uploaded++;
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
        if (!t.drawn) this.counters.wasted++;
        gl.deleteTexture(t.tex);
        this.texBytes -= t.bytes;
        this.tex.delete(k);
      }
    }
    const annBudget = this.budget * 0.5;
    if (this.gpuBytes > annBudget) {
      const cand = [];
      for (const s of this.slides.values()) for (const L of s.layers.values()) {
        for (const [b, d] of L.detail) if (d.last < this.frame - 1) cand.push([L, b, d.last, false]);
        if (L.ptiles) for (const [k, t] of L.ptiles) if (t.last < this.frame - 1) cand.push([L, k, t.last, true]);
      }
      cand.sort((a, b) => a[2] - b[2]);
      for (const [L, b, , pt] of cand) { if (this.gpuBytes <= annBudget * 0.85) break; if (pt) L.dropPTile(b); else L.dropBucket(b); }
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

  _legends() {
    const fmt = (v) => { const a = Math.abs(v); return a >= 1e4 || (a > 0 && a < 1e-2) ? v.toExponential(1) : a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2); };
    for (const p of this.panes) {
      if (!p.$legend) continue;
      if (!this.showLegend || !p.slide) { p.$legend.innerHTML = ''; continue; }
      let html = '';
      const shownLayers = [...p.slide.layers.values()].reverse()
        .filter((L) => L.state === 'ready' && p.layerVisible(L) && L.style.legend !== false
          && (L.colormap || L.classes.length > 1 || L.style.legend === true));
      for (const L of shownLayers.slice(0, 4)) {
        const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
        if (L.colormap) {
          const cm = L.colormap, stops = cm.lut.filter((_, i) => i % 32 === 0 || i === 255).join(',');
          const lo = cm.log ? 10 ** cm.vmin : cm.vmin, hi = cm.log ? 10 ** cm.vmax : cm.vmax;
          html += `<b>${esc(cm.label || L.name)}</b><div class="grad" style="background:linear-gradient(90deg,${stops})"></div>`
            + `<div class="ends"><span>${fmt(lo)}</span><span>${cm.log ? 'log' : ''}</span><span>${fmt(hi)}</span></div>`;
        } else if (L.classes.length > 1 || L.style.legend === true) {
          html += `<b>${esc(L.name)}</b>`;
          const shown = L.classes.filter((c) => !L.hidden.has(c.name));
          for (const c of shown.slice(0, 8)) html += `<div><span class="sw" style="background:${c.color}"></span>${esc(c.name)}</div>`;
          if (shown.length > 8) html += `<div style="opacity:.7">+${shown.length - 8} more</div>`;
        }
      }
      if (shownLayers.length > 4) html += `<div style="opacity:.7;margin-top:2px">+${shownLayers.length - 4} more layers</div>`;
      p.$legend.innerHTML = html;
    }
  }

  _panelRefresh() {
    this._legends();
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
      const sr = row(`<label>Stain view</label><select>${['off', 'hematoxylin', 'eosin', 'dab', 'fluorescence'].map((m) => `<option ${p.stain?.name === m ? 'selected' : ''}>${m}</option>`).join('')}</select>`);
      sr.children[1].onchange = (e) => this.setStain(p.index, e.target.value);
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

/** Single-slide convenience element: <wsi-viewer api="/wissi-viz" slide="id"></wsi-viewer> */
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

/** Main element: <wissi-viz api="/wissi-viz" slides="a,b,c"></wissi-viz> (aliases: <wsi-grid>, <wsi-viewer> for one slide). */
export class WissiViz extends WsiGrid {}
if (!customElements.get('wissi-viz')) customElements.define('wissi-viz', WissiViz);
if (!customElements.get('wsi-grid')) customElements.define('wsi-grid', class extends WsiGrid {});
if (!customElements.get('wsi-viewer')) customElements.define('wsi-viewer', WsiViewer);
