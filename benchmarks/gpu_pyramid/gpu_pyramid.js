// EXPERIMENTAL (not wired into the viewer yet) — benchmarks/gpu_pyramid_ab.py compares it with the worker.
// WebGPU compute build of the client-side object pyramid: same input (the /annotations/columns payload) and the same
// output ({pyr, levels: [{k, raw, hist, tiles: [{tx, ty, n, x0, y0, span, pos, …}]}]}) as PYR_WORKER_SRC, class mode.
//
// Every aggregated level is built straight from the objects (power-of-two bins nest, so a level-k dot is the mean of
// all objects in its bin — the same as the worker's level-by-level weighted means):
//   insert   per object: hash-insert its bin, count, 8-slot class histogram, min "quadrant path" (→ representative)
//   sum      per object: fixed-point position sums relative to the bin origin; representative = min index on min path
//   tcount   per bin: occupied bins per 128×128-bin tile
//   scan     one thread: exclusive prefix sum over tiles
//   scatter  per bin: mean position → tile-relative u16, dominant class, normalised histogram, written in tile order
// The raw level (every object, for outlines) is tcount → scan → scatter over objects.
// Readback in two steps: tile offsets (tiny), then exactly the bytes used. The CPU only makes per-tile views.

const SLOTS = 8, TB_BINS = 128, WG = 256;

const WGSL = /* wgsl */`
struct P {
  n: u32, k: u32, k0: u32, cap: u32,
  s: f32, span: f32, nbx: u32, ntx: u32,
  ntiles: u32, tbBase: u32, outBase: u32, bound: u32,
  offX: u32, offY: u32, offRX: u32, offRY: u32,
  offCol: u32, offCls: u32, offKind: u32, tcBase: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> obj: array<u32>;                 // the columns payload, as words
@group(0) @binding(2) var<storage, read> tabs: array<u32>;                // [slotOf[nclasses] | tc[8]]
@group(0) @binding(3) var<storage, read_write> ht: array<atomic<u32>>;    // keys|cnt|pathmax|repmax|sx|sy|hist×8 (× cap)
@group(0) @binding(4) var<storage, read_write> tb: array<atomic<u32>>;    // per level: count/cursor[ntiles] | off[ntiles+1]
@group(0) @binding(5) var<storage, read_write> entryOf: array<u32>;
@group(0) @binding(6) var<storage, read_write> outw: array<u32>;

fn fat(off: u32, i: u32) -> f32 { return bitcast<f32>(obj[off + i]); }
fn u16at(off: u32, i: u32) -> u32 { return (obj[off + (i >> 1u)] >> ((i & 1u) * 16u)) & 0xFFFFu; }
fn u8at(off: u32, i: u32) -> u32 { return (obj[off + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu; }
fn hsh(k: u32) -> u32 { var x = k; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
fn binOf(v: f32, s: f32) -> u32 { return u32(max(0.0, floor(v / s))); }
// quadrant path from level k-1 down to k0+1, y bit before x bit: the worker's stable (by, bx) sorts pick, at each
// level, the child bin with the smallest (y, x) — so the representative is the min index on the min path
fn pathCode(x: f32, y: f32) -> u32 {
  var code = 0u;
  var j = i32(p.k) - 1;
  loop {
    if (j <= i32(p.k0)) { break; }
    let s = exp2(f32(j));
    code = code * 4u + ((binOf(y, s) & 1u) << 1u) + (binOf(x, s) & 1u);
    j = j - 1;
  }
  return code;
}
// fixed-point scale for this bin: count × s × 2^q < 2^31
fn shiftFor(c: u32) -> f32 { return floor(31.0 - log2(f32(c)) - log2(p.s)) - 1.0; }
fn q16(v: f32) -> u32 { return u32(clamp(floor(v * 65535.0 + 0.5), 0.0, 65535.0)); }

@compute @workgroup_size(${WG}) fn agg_insert(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let x = fat(p.offX, i); let y = fat(p.offY, i);
  let key = binOf(y, p.s) * p.nbx + binOf(x, p.s) + 1u;
  let m = p.cap - 1u;
  var h = hsh(key) & m;
  loop {
    let r = atomicCompareExchangeWeak(&ht[h], 0u, key);
    if (r.exchanged || r.old_value == key) { break; }
    if (r.old_value != 0u) { h = (h + 1u) & m; }      // (old 0 = spurious failure: retry the same slot)
  }
  entryOf[i] = h;
  atomicAdd(&ht[p.cap + h], 1u);
  let slot = tabs[u16at(p.offCls, i)];
  atomicAdd(&ht[6u * p.cap + h * 8u + slot], 1u);
  atomicMax(&ht[2u * p.cap + h], 0xFFFFFFFFu - pathCode(x, y));
}

@compute @workgroup_size(${WG}) fn agg_sum(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let x = fat(p.offX, i); let y = fat(p.offY, i);
  let h = entryOf[i];
  let c = atomicLoad(&ht[p.cap + h]);
  let sc = exp2(shiftFor(c));
  let fx = u32(floor((x - f32(binOf(x, p.s)) * p.s) * sc + 0.5));
  let fy = u32(floor((y - f32(binOf(y, p.s)) * p.s) * sc + 0.5));
  atomicAdd(&ht[4u * p.cap + h], fx);
  atomicAdd(&ht[5u * p.cap + h], fy);
  if (0xFFFFFFFFu - pathCode(x, y) == atomicLoad(&ht[2u * p.cap + h])) { atomicMax(&ht[3u * p.cap + h], 0xFFFFFFFFu - i); }
}

@compute @workgroup_size(${WG}) fn agg_tcount(@builtin(global_invocation_id) g: vec3<u32>) {
  let h = g.x; if (h >= p.cap) { return; }
  let key = atomicLoad(&ht[h]); if (key == 0u) { return; }
  let b = key - 1u; let bx = b % p.nbx; let by = b / p.nbx;
  atomicAdd(&tb[p.tbBase + (by / ${TB_BINS}u) * p.ntx + bx / ${TB_BINS}u], 1u);
}

@compute @workgroup_size(1) fn scan() {
  var run = 0u;
  for (var t = 0u; t < p.ntiles; t++) {
    let c = atomicLoad(&tb[p.tbBase + t]);
    atomicStore(&tb[p.tbBase + p.ntiles + t], run);
    atomicStore(&tb[p.tbBase + t], 0u);          // becomes the scatter cursor
    run += c;
  }
  atomicStore(&tb[p.tbBase + 2u * p.ntiles], run);
}

@compute @workgroup_size(${WG}) fn agg_scatter(@builtin(global_invocation_id) g: vec3<u32>) {
  let h = g.x; if (h >= p.cap) { return; }
  let key = atomicLoad(&ht[h]); if (key == 0u) { return; }
  let b = key - 1u; let bx = b % p.nbx; let by = b / p.nbx;
  let tx = bx / ${TB_BINS}u; let ty = by / ${TB_BINS}u; let tile = ty * p.ntx + tx;
  let dst = atomicLoad(&tb[p.tbBase + p.ntiles + tile]) + atomicAdd(&tb[p.tbBase + tile], 1u);
  let c = atomicLoad(&ht[p.cap + h]);
  let inv = exp2(-shiftFor(c)) / f32(c);
  let mx_ = f32(atomicLoad(&ht[4u * p.cap + h])) * inv;
  let my_ = f32(atomicLoad(&ht[5u * p.cap + h])) * inv;
  let lx = f32(bx - tx * ${TB_BINS}u) * p.s + mx_;
  let ly = f32(by - ty * ${TB_BINS}u) * p.s + my_;
  var H: array<u32, 8>;
  var mx = 0u; var am = 0u;
  for (var q = 0u; q < 8u; q++) { H[q] = atomicLoad(&ht[6u * p.cap + h * 8u + q]); if (H[q] > mx) { mx = H[q]; am = q; } }
  var w0 = 0u; var w1 = 0u;
  for (var q = 0u; q < 8u; q++) {
    var v = 0u;
    if (H[q] > 0u) { v = max(1u, (510u * H[q] + mx) / (2u * mx)); }   // = Math.round(255 h / mx), exactly
    if (q < 4u) { w0 |= v << (8u * q); } else { w1 |= v << (8u * (q - 4u)); }
  }
  let o = p.outBase; let B = p.bound;
  outw[o + dst] = q16(lx / p.span) | (q16(ly / p.span) << 16u);
  outw[o + B + dst] = min(c, 65535u);
  outw[o + 2u * B + dst] = tabs[p.tcBase + am];
  outw[o + 3u * B + dst] = 0xFFFFFFFFu - atomicLoad(&ht[3u * p.cap + h]);
  outw[o + 4u * B + 2u * dst] = w0;
  outw[o + 4u * B + 2u * dst + 1u] = w1;
}

@compute @workgroup_size(${WG}) fn raw_tcount(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let tile = binOf(fat(p.offY, i), p.span) * p.ntx + binOf(fat(p.offX, i), p.span);
  atomicAdd(&tb[p.tbBase + tile], 1u);
}

@compute @workgroup_size(${WG}) fn raw_scatter(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let x = fat(p.offX, i); let y = fat(p.offY, i);
  let tx = binOf(x, p.span); let ty = binOf(y, p.span); let tile = ty * p.ntx + tx;
  let dst = atomicLoad(&tb[p.tbBase + p.ntiles + tile]) + atomicAdd(&tb[p.tbBase + tile], 1u);
  let rx = u16at(p.offRX, i); let ry = u16at(p.offRY, i);
  let o = p.outBase; let B = p.bound;
  outw[o + dst] = q16((x - f32(tx) * p.span) / p.span) | (q16((y - f32(ty) * p.span) / p.span) << 16u);
  outw[o + B + dst] = min(65535u, max(rx, ry) / 2u);
  outw[o + 2u * B + dst] = u16at(p.offCol, i);
  outw[o + 3u * B + dst] = u16at(p.offCls, i);
  outw[o + 4u * B + dst] = i;
  outw[o + 5u * B + dst] = u8at(p.offKind, i);
  let sb = o + 6u * B + 4u * dst;
  outw[sb] = bitcast<u32>(x); outw[sb + 1u] = bitcast<u32>(y);
  outw[sb + 2u] = bitcast<u32>(f32(rx) / 4.0); outw[sb + 3u] = bitcast<u32>(f32(ry) / 4.0);
}
`;

let _dev = null, _pipes = null, _bgl = null, _info = null;
export const gpuInfo = () => _info;
export async function gpuDevice() {
  if (_dev) return _dev;
  if (!navigator.gpu) return null;
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!a) return null;
  _info = { vendor: a.info?.vendor, architecture: a.info?.architecture, description: a.info?.description, fallback: !!a.isFallbackAdapter };
  _dev = await a.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxBufferSize: a.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: Math.min(8, a.limits.maxStorageBuffersPerShaderStage) } });
  const mod = _dev.createShaderModule({ code: WGSL });
  const ci = await mod.getCompilationInfo();
  const errs = ci.messages.filter((m) => m.type === 'error');
  if (errs.length) { _dev = null; throw new Error('WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')); }
  const S = GPUShaderStage.COMPUTE;
  _bgl = _dev.createBindGroupLayout({ entries: [
    { binding: 0, visibility: S, buffer: { type: 'uniform' } },
    { binding: 1, visibility: S, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: S, buffer: { type: 'read-only-storage' } },
    ...[3, 4, 5, 6].map((b) => ({ binding: b, visibility: S, buffer: { type: 'storage' } })) ] });
  const layout = _dev.createPipelineLayout({ bindGroupLayouts: [_bgl] });
  _pipes = {};
  for (const e of ['agg_insert', 'agg_sum', 'agg_tcount', 'scan', 'agg_scatter', 'raw_tcount', 'raw_scatter'])
    _pipes[e] = _dev.createComputePipeline({ layout, compute: { module: mod, entryPoint: e } });
  return _dev;
}

const nextPow2 = (v) => 2 ** Math.ceil(Math.log2(Math.max(2, v)));

/** Build the pyramid on the GPU. Returns null when this payload isn't supported (value mode, no WebGPU, too big). */
export async function buildPyramidGPU(buf, timing = {}) {
  const dev = await gpuDevice();
  if (!dev) return null;
  const T0 = performance.now();
  const dv = new DataView(buf), jl = dv.getUint32(4, true);
  const M = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jl)));
  if (M.mode === 'value') return null;
  const n = M.n;
  let off = 8 + jl;
  const offs = {};
  const take = (name, bytesPer) => { offs[name] = off / 4; const a = off; off += Math.ceil(n * bytesPer / 4) * 4; return a; };
  const aX = take('X', 4), aY = take('Y', 4); take('RX', 2); take('RY', 2); take('Col', 2); const aCls = take('Cls', 2); take('Kind', 1);
  // class slots exactly as the worker
  const cls = new Uint16Array(buf, aCls, n);
  const cc = new Uint32Array(Math.max(M.nclasses, 1)); for (let i = 0; i < n; i++) cc[cls[i]]++;
  let top = [...cc.keys()].filter((c) => cc[c] > 0).sort((a, b) => cc[b] - cc[a]).slice(0, SLOTS);
  const other = M.nclasses > SLOTS;
  const slotOf = new Uint32Array(Math.max(M.nclasses, 1)).fill(other ? SLOTS - 1 : 0);
  if (other) top = top.slice(0, SLOTS - 1);
  top.forEach((c, j) => { slotOf[c] = j; });
  let otherClass = null;
  if (other) { let best = -1; for (let c = 0; c < M.nclasses; c++) if (!top.includes(c) && cc[c] > best) { best = cc[c]; otherClass = c; } }
  const slots = top.concat(other ? [-1] : []);
  const tc = top.concat(new Array(SLOTS - top.length).fill(otherClass ?? top[top.length - 1] ?? 0));
  const k0 = Math.min(24, Math.max(0, Math.ceil(Math.log2(Math.max(M.median_size, 1)))));
  const extent = Math.max(M.bbox[2], M.bbox[3], 1), kmax = Math.max(k0 + 1, Math.ceil(Math.log2(extent / 64)));
  if (kmax - 1 - k0 > 15) return null;                       // path code would overflow 32 bits
  const TB = M.tile_bins;
  if (TB !== TB_BINS) return null;
  // level plans (CPU knows every size up front)
  const xmax = M.bbox[2], ymax = M.bbox[3];
  const plans = [];
  let tbWords = 0, outWords = 0, maxCap = 0;
  { const span = 2 ** k0 * TB, ntx = Math.floor(xmax / span) + 1, nty = Math.floor(ymax / span) + 1;
    plans.push({ k: k0, raw: true, s: 2 ** k0, span, nbx: 0, ntx, ntiles: ntx * nty, bound: n, cap: 0, words: 10 }); }
  for (let k = k0 + 1; k <= kmax; k++) {
    const s = 2 ** k, nbx = Math.floor(xmax / s) + 1, nby = Math.floor(ymax / s) + 1;
    const bound = Math.min(n, nbx * nby), ntx = Math.ceil(nbx / TB), nty = Math.ceil(nby / TB);
    const cap = nextPow2(Math.ceil(bound * 1.5) + 1);
    plans.push({ k, raw: false, s, span: s * TB, nbx, ntx, ntiles: ntx * nty, bound, cap, words: 6 });
    maxCap = Math.max(maxCap, cap);
  }
  for (const L of plans) { L.tbBase = tbWords; tbWords += 2 * L.ntiles + 1; L.outBase = outWords; outWords += L.words * L.bound; }
  const U = GPUBufferUsage;
  const mk = (bytes, usage) => dev.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
  const objB = mk(buf.byteLength, U.STORAGE | U.COPY_DST);
  dev.queue.writeBuffer(objB, 0, buf, 0, Math.floor(buf.byteLength / 4) * 4);
  const tabsArr = new Uint32Array(slotOf.length + SLOTS); tabsArr.set(slotOf); tabsArr.set(tc, slotOf.length);
  const tabsB = mk(tabsArr.byteLength, U.STORAGE | U.COPY_DST); dev.queue.writeBuffer(tabsB, 0, tabsArr);
  const htB = mk(14 * maxCap * 4, U.STORAGE | U.COPY_DST);
  const tbB = mk(tbWords * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
  const entB = mk(n * 4, U.STORAGE);
  const outB = mk(outWords * 4, U.STORAGE | U.COPY_SRC);
  const enc = dev.createCommandEncoder();
  const uniBufs = [];
  for (const L of plans) {
    const u = new ArrayBuffer(80), ui = new Uint32Array(u), uf = new Float32Array(u);
    ui[0] = n; ui[1] = L.k; ui[2] = k0; ui[3] = L.cap; uf[4] = L.s; uf[5] = L.span; ui[6] = L.nbx; ui[7] = L.ntx;
    ui[8] = L.ntiles; ui[9] = L.tbBase; ui[10] = L.outBase; ui[11] = L.bound;
    ui[12] = offs.X; ui[13] = offs.Y; ui[14] = offs.RX; ui[15] = offs.RY; ui[16] = offs.Col; ui[17] = offs.Cls; ui[18] = offs.Kind; ui[19] = slotOf.length;
    const ub = mk(80, U.UNIFORM | U.COPY_DST); dev.queue.writeBuffer(ub, 0, u); uniBufs.push(ub);
    const bg = dev.createBindGroup({ layout: _bgl, entries: [ub, objB, tabsB, htB, tbB, entB, outB].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    if (!L.raw) enc.clearBuffer(htB, 0, 14 * L.cap * 4);
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, bg);
    const run = (name, count) => { pass.setPipeline(_pipes[name]); pass.dispatchWorkgroups(Math.ceil(count / WG)); };
    if (L.raw) { run('raw_tcount', n); pass.setPipeline(_pipes.scan); pass.dispatchWorkgroups(1); run('raw_scatter', n); }
    else {
      run('agg_insert', n); run('agg_sum', n); run('agg_tcount', L.cap);
      pass.setPipeline(_pipes.scan); pass.dispatchWorkgroups(1); run('agg_scatter', L.cap);
    }
    pass.end();
  }
  const tbRead = mk(tbWords * 4, U.MAP_READ | U.COPY_DST);
  enc.copyBufferToBuffer(tbB, 0, tbRead, 0, tbWords * 4);
  dev.queue.submit([enc.finish()]);
  const T1 = performance.now();
  await tbRead.mapAsync(GPUMapMode.READ);
  const tbv = new Uint32Array(tbRead.getMappedRange().slice(0));
  tbRead.unmap();
  const T2 = performance.now();
  // exactly the bytes used: per level, each field's first `total` records
  const enc2 = dev.createCommandEncoder();
  let rw = 0;
  for (const L of plans) {
    L.total = tbv[L.tbBase + 2 * L.ntiles];
    L.fields = [];
    const fieldWords = L.raw ? [1, 1, 1, 1, 1, 1, 4] : [1, 1, 1, 1, 2];
    let fo = L.outBase;
    for (const w of fieldWords) { L.fields.push({ src: fo, dst: rw, words: w * L.total }); rw += w * L.total; fo += w * L.bound; }
  }
  const outRead = mk(rw * 4, U.MAP_READ | U.COPY_DST);
  for (const L of plans) for (const f of L.fields) if (f.words) enc2.copyBufferToBuffer(outB, f.src * 4, outRead, f.dst * 4, f.words * 4);
  dev.queue.submit([enc2.finish()]);
  await outRead.mapAsync(GPUMapMode.READ);
  const all = outRead.getMappedRange().slice(0);
  outRead.unmap();
  const T3 = performance.now();
  for (const b of [objB, tabsB, htB, tbB, entB, outB, tbRead, outRead, ...uniBufs]) b.destroy();
  // per-tile views (no per-tile copies)
  const u32 = (f) => new Uint32Array(all, f.dst * 4, f.words);
  const levels = [];
  for (const L of plans) {
    const F = L.fields, off0 = L.tbBase + L.ntiles;
    const pos = new Uint16Array(all, F[0].dst * 4, 2 * L.total);
    let cols;
    if (L.raw) cols = { size: new Uint16Array(u32(F[1])), col: new Uint16Array(u32(F[2])), cls: new Uint16Array(u32(F[3])), orig: u32(F[4]),
      kind: new Uint8Array(u32(F[5])), shp: new Float32Array(all, F[6].dst * 4, 4 * L.total) };
    else cols = { count: new Uint16Array(u32(F[1])), col: new Uint16Array(u32(F[2])), orig: u32(F[3]), hist: new Uint8Array(all, F[4].dst * 4, 8 * L.total) };
    const tiles = [];
    for (let t = 0; t < L.ntiles; t++) {
      const a = tbv[off0 + t], b = t + 1 < L.ntiles ? tbv[off0 + t + 1] : L.total;
      if (b <= a) continue;
      const tx = t % L.ntx, ty = Math.floor(t / L.ntx);
      const tile = { tx, ty, n: b - a, x0: tx * L.span, y0: ty * L.span, span: L.span, pos: pos.subarray(2 * a, 2 * b) };
      for (const [name, arr] of Object.entries(cols)) {
        const per = name === 'hist' ? 8 : name === 'shp' ? 4 : 1;
        tile[name] = arr.subarray(per * a, per * b);
      }
      tiles.push(tile);
    }
    levels.push(L.raw ? { k: L.k, raw: true, tiles } : { k: L.k, raw: false, hist: true, tiles });
  }
  const pyr = { k0, kmax, levels: levels.length, tile_bins: TB, mode: 'class', slots, other_class: otherClass,
    tiles: levels.map((L) => L.tiles.length), points: levels.map((L) => L.tiles.reduce((a, t) => a + t.n, 0)) };
  const T4 = performance.now();
  Object.assign(timing, { setup_upload_encode: T1 - T0, gpu_compute: T2 - T1, readback: T3 - T2, tile_views: T4 - T3, total: T4 - T0,
    gpuMB: (buf.byteLength + 14 * maxCap * 4 + tbWords * 4 + n * 4 + outWords * 4) / 2 ** 20, readMB: rw * 4 / 2 ** 20 });
  return { pyr, levels };
}
