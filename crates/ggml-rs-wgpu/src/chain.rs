//! A decode step's ops chained on the adapter ([`ggml_rs::chain`]): vectors kept in buffers here, the ops recorded and
//! run in one compute pass of one submit, only what the host asks for read back. The quantized matmuls are the GGUF
//! kernels (`shaders`, one row of `x`); RMSNorm, the residual add, the SwiGLU of a fused gate-up, RoPE, a store into a
//! cache and a decode step's attention have kernels of their own here, on the same bind group layout (weights, a
//! table or a cache at 0, the input at 1, the output at 2, the parameters at 3).

use std::sync::{Arc, Mutex};

use ggml_rs::chain::{ChainRecorder, DeltaNet, DeviceChain, DeviceVec};
use ggml_rs::{QuantizedTensor, Tensor};

use crate::{Gpu, WgpuBackend, WgpuQuant};

const HEAD: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
"#;

/// `y = x / sqrt(mean(x^2) + eps) * w` over `p[0].x` elements (`p[0].y` the bits of `eps`), one workgroup.
const RMSNORM: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = x[i]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    for (var i = li; i < n; i += 256u) { y[i] = x[i] * inv * bitcast<f32>(w[i]); }
}
"#;

/// [`RMSNORM`] of row `wg.x + 65535 wg.y` of `x` (rows of `p[0].x`, `p[0].w` of them), every row with the same
/// weights `w`, or with `p[0].z` rows of them the row's `row % p[0].z` (a hyper-connection's streams).
const RMSNORM_ROWS: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].w) { return; }
    let at = row * n;
    var wrows = p[0].z;
    if (wrows == 0u) { wrows = 1u; }
    let wat = (row % wrows) * n;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = x[at + i]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    for (var i = li; i < n; i += 256u) { y[at + i] = x[at + i] * inv * bitcast<f32>(w[wat + i]); }
}
"#;

/// [`RMSNORM_ROWS`] of rows a multiple of 4 long: vec4 loads, a thread's squares in four running sums (its loads in
/// flight together, where one sum waited on each load in turn). `p[0]`: n, the bits of eps, weight rows, rows.
const RMSNORM_ROWS4: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y4: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n4 = p[0].x / 4u;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].w) { return; }
    let at = row * n4;
    var wrows = p[0].z;
    if (wrows == 0u) { wrows = 1u; }
    let wat = (row % wrows) * n4;
    var s = vec4<f32>(0.0);
    for (var i = li; i < n4; i += 256u) { let v = x4[at + i]; s += v * v; }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(p[0].x) + bitcast<f32>(p[0].y));
    for (var i = li; i < n4; i += 256u) { y4[at + i] = x4[at + i] * inv * w4[wat + i]; }
}
"#;

/// `x += y`, then each row of `x` RMS-normed into `out` with `w` (`[n]`, one for every row), rows a multiple of 4
/// long: a residual's add and the next norm in one dispatch. `p[0]`: n, the bits of eps.
const ADD_RMSNORM_ROWS4: &str = r#"
@group(0) @binding(0) var<storage, read> yv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> x4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n4 = p[0].x / 4u;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].z) { return; }
    let at = row * n4;
    var s = vec4<f32>(0.0);
    for (var i = li; i < n4; i += 256u) {
        let v = x4[at + i] + yv[at + i];
        x4[at + i] = v;
        s += v * v;
    }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(p[0].x) + bitcast<f32>(p[0].y));
    for (var i = li; i < n4; i += 256u) { out[at + i] = x4[at + i] * inv * w4[i]; }
}
"#;

/// A hyper-connection's gates, `p[0]`: rank, writes, streams, rows. `t` (binding 6) `[rows, rank + writes]`: its first
/// `rank` become `silu(t / streams)`, the rest 0; `post` (binding 7) `[rows, writes]` gets their `2 sigmoid(t /
/// streams)`. As the host's `hc_gates`.
const HC_GATES: &str = r#"
@group(0) @binding(6) var<storage, read_write> t: array<f32>;
@group(0) @binding(7) var<storage, read_write> post: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rank = p[0].x;
    let writes = p[0].y;
    let streams = f32(p[0].z);
    let width = rank + writes;
    let i = id.x;
    if (i >= width * p[0].w) { return; }
    let r = i / width;
    let c = i % width;
    let v = t[i] / streams;
    if (c < rank) {
        t[i] = v / (1.0 + exp(-v));
    } else {
        post[r * writes + c - rank] = 2.0 / (1.0 + exp(-v));
        t[i] = 0.0;
    }
}
"#;

/// A hyper-connection's branch input: `y[r, j] = sum over s of x[r, s, j] / (1 + exp(-w[r, s, j])) / streams` (`w` the
/// logits, `x` the normed streams), `p[0]`: d, streams, rows. As the host's `hc_mix`.
const HC_MIX: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let streams = p[0].y;
    let i = id.x;
    if (i >= d * p[0].z) { return; }
    let r = i / d;
    let j = i % d;
    var acc = 0.0;
    for (var s = 0u; s < streams; s++) {
        let at = (r * streams + s) * d + j;
        acc += x[at] / (1.0 + exp(-bitcast<f32>(w[at]))) / f32(streams);
    }
    y[i] = acc;
}
"#;

/// A hyper-connection site's write-back: `y[r, s, j] += w[r, s] * x[r, j]` (`w` the write weights, `x` the branch
/// output, `y` the streams), `p[0]`: d, streams, rows. As the host's `stream_apply`.
const STREAM_APPLY: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let streams = p[0].y;
    let i = id.x;
    if (i >= d * streams * p[0].z) { return; }
    let r = i / (d * streams);
    let s = (i / d) % streams;
    let j = i % d;
    y[i] = y[i] + bitcast<f32>(w[r * streams + s]) * x[r * d + j];
}
"#;

/// `y[i] += w[p[0].y] * x[i]` for `i < p[0].x`: a weighted sum's term, its weight read from the device.
const AXPY_AT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[i] = y[i] + bitcast<f32>(w[p[0].y]) * x[i]; }
}
"#;

/// `y += x` over `p[0].x` elements.
const ADD: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[i] = y[i] + x[i]; }
}
"#;

/// `y[r] = silu(x[r][..ff]) * x[r][ff..]` for each of `p[0].y` rows, `ff = p[0].x`.
const SILU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].x;
    if (i < ff * p[0].y) {
        let r = i / ff;
        let j = i % ff;
        let g = x[r * 2u * ff + j];
        y[i] = (g / (1.0 + exp(-g))) * x[r * 2u * ff + ff + j];
    }
}
"#;

/// `y[r] = gelu_approx(x[r][..ff]) * x[r][ff..]` (the tanh approximation, as the CPU's) for each of `p[0].y` rows.
const GELU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].x;
    if (i < ff * p[0].y) {
        let r = i / ff;
        let j = i % ff;
        let g = x[r * 2u * ff + j];
        let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
        y[i] = 0.5 * g * (1.0 + tanh(inner)) * x[r * 2u * ff + ff + j];
    }
}
"#;

/// RoPE in place on `y` (`[p[0].w rows, p[0].x heads, p[0].y head_dim]`), on the first `rot` (`p[1].x`, 0: all) of
/// each head: row `r`'s pair `k` by the sine `w[r * rot + 2k]` and cosine `w[r * rot + 2k + 1]` (made on the host, as
/// the CPU's rope makes them), the pairs `(2k, 2k + 1)` or with `p[0].z` `(k, k + rot / 2)`.
const ROPE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let heads = p[0].x;
    let hd = p[0].y;
    var rot = p[1].x;
    if (rot == 0u) { rot = hd; }
    let half = rot / 2u;
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].w * heads * half) { return; }
    let r = i / (heads * half);
    let h = (i / half) % heads;
    let k = i % half;
    // (each head its own table where `p[1].y` is set: LTX's split rotary)
    var tr = r * rot;
    if (p[1].y != 0u) { tr = (r * heads + h) * rot; }
    let s = bitcast<f32>(w[tr + 2u * k]);
    let c = bitcast<f32>(w[tr + 2u * k + 1u]);
    let base = (r * heads + h) * hd;
    var ia = base + 2u * k;
    var ib = ia + 1u;
    if (p[0].z != 0u) {
        ia = base + k;
        ib = ia + half;
    }
    let a = y[ia];
    let b = y[ib];
    y[ia] = a * c - b * s;
    y[ib] = a * s + b * c;
}
"#;

/// `p[1].x` rows of `p[0].x` from `x` into `y`'s rows from `p[0].y`, each `p[0].z` long, at `p[0].w` in it.
const STORE_ROWS: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let len = p[0].x;
    if (i < len * p[1].x) {
        let r = i / len;
        y[(p[0].y + r) * p[0].z + p[0].w + i % len] = x[i];
    }
}
"#;

/// A prompt's attention over positions split in runs of 256, a workgroup a (query head `h`, run, query `s`): as
/// [`ATTENTION_PART`] for query `s` at position `past + s`, over positions up to its own and (with a window) its last
/// `window`; a run past them writes nothing to add. `y`: the output `[rows, n_h, hd]`, then each (query, head, run)'s
/// `hd` weighted values, then its `m` and `l`. `p[0]`: `n_h`, `n_kv`, `hd`, `past`; `p[1]`: the window (0: none), the
/// runs, the bits of the scale, the rows.
const ATTENTION_ROWS_PART: &str = r#"
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let past = p[0].w;
    let window = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let rows = p[1].w;
    let h = wg.x;
    let run = wg.y;
    let s = wg.z;
    let kh = h / (n_h / n_kv);
    let kvd = n_kv * hd;
    let row = 2u * kvd;
    let hi = past + s + 1u;
    var lo = 0u;
    if (window != 0u && hi > window) { lo = hi - window; }
    let start = run * 256u;
    let end = min(start + 256u, hi);
    let qb = (s * n_h + h) * hd;
    let t = start + li;
    let live = t >= lo && t < end;
    var sv = -3.4e38;
    if (live) {
        let kb = t * row + kh * hd;
        var d0 = 0.0;
        for (var d = 0u; d < hd; d++) { d0 += x[qb + d] * bitcast<f32>(w[kb + d]); }
        sv = d0 * scale;
    }
    red[li] = sv;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] = max(red[li], red[li + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = 0.0;
    if (live) { e = exp(sv - m); }
    sc[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let l = red[0];
    let unit = (s * n_h + h) * runs + run;
    let part = rows * n_h * hd + unit * hd;
    var n = 0u;
    if (end > start) { n = end - start; }
    for (var d = li; d < hd; d += 256u) {
        var acc = 0.0;
        for (var i = 0u; i < n; i++) { acc += sc[i] * bitcast<f32>(w[(start + i) * row + kvd + kh * hd + d]); }
        y[part + d] = acc;
    }
    if (li == 0u) {
        let ml = rows * n_h * hd + rows * n_h * runs * hd + unit * 2u;
        y[ml] = m;
        y[ml + 1u] = l;
    }
}
"#;

/// [`ATTENTION_ROWS_PART`] for a head size a multiple of 4 (at most 512), as [`ATTENTION_PART4`] is a step's: the
/// query in the workgroup's memory, each key a vec4 at a time, each value column's products in four running sums.
/// The same parts, largest and sum for the join. (A check of four drafted rows over 15,888 positions of 24 heads of
/// 256: its sixteen layers' parts 39 ms with a sum a load.)
const ATTENTION_ROWS_PART4: &str = r#"
@group(0) @binding(0) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> qs: array<vec4<f32>, 128>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let past = p[0].w;
    let window = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let rows = p[1].w;
    let h = wg.x;
    let run = wg.y;
    let s = wg.z;
    let kh = h / (n_h / n_kv);
    let hd4 = hd / 4u;
    let kvd4 = n_kv * hd4;
    let row4 = 2u * kvd4;
    if (li < hd4) {
        qs[li] = q4[(s * n_h + h) * hd4 + li];
    }
    workgroupBarrier();
    let hi = past + s + 1u;
    var lo = 0u;
    if (window != 0u && hi > window) { lo = hi - window; }
    let start = run * 256u;
    let end = min(start + 256u, hi);
    let t = start + li;
    let live = t >= lo && t < end;
    var sv = -3.4e38;
    if (live) {
        let kb = t * row4 + kh * hd4;
        var a = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d++) { a += qs[d] * kv4[kb + d]; }
        sv = (a.x + a.y + a.z + a.w) * scale;
    }
    red[li] = sv;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] = max(red[li], red[li + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = 0.0;
    if (live) { e = exp(sv - m); }
    sc[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let l = red[0];
    let unit = (s * n_h + h) * runs + run;
    let part = rows * n_h * hd + unit * hd;
    var n = 0u;
    if (end > start) { n = end - start; }
    for (var d4 = li; d4 < hd4; d4 += 256u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        let vb = start * row4 + kvd4 + kh * hd4 + d4;
        var i = 0u;
        for (; i + 4u <= n; i += 4u) {
            a0 += sc[i] * kv4[vb + i * row4];
            a1 += sc[i + 1u] * kv4[vb + (i + 1u) * row4];
            a2 += sc[i + 2u] * kv4[vb + (i + 2u) * row4];
            a3 += sc[i + 3u] * kv4[vb + (i + 3u) * row4];
        }
        for (; i < n; i++) { a0 += sc[i] * kv4[vb + i * row4]; }
        let o = (a0 + a1) + (a2 + a3);
        y[part + d4 * 4u] = o.x;
        y[part + d4 * 4u + 1u] = o.y;
        y[part + d4 * 4u + 2u] = o.z;
        y[part + d4 * 4u + 3u] = o.w;
    }
    if (li == 0u) {
        let ml = rows * n_h * hd + rows * n_h * runs * hd + unit * 2u;
        y[ml] = m;
        y[ml + 1u] = l;
    }
}
"#;

/// A prompt's attention in one pass, a workgroup a (head, tile of queries) ([`attention_tiled`]'s kernel, its sizes
/// put in): the keys and values a block at a time through workgroup memory (the block's keys, then its values, rows
/// padded a vec4 against bank conflicts), each query's softmax online (a thread a query keeps its running largest
/// score and sum; the scores, then weights, a block's `[key][query]`), each thread a few queries by four of the head's
/// dims of the output. No parts and no join: the output `[rows, n_h, hd]` alone. Within WebGPU's portable limits
/// (16 KB of workgroup memory, 256 invocations). `p[0]`: `n_h`, `n_kv`, `past`, the dispatch's rows; `p[1]`: `kv_len`,
/// the scale's bits, the dispatch's first row among the queries, its mask (bit 0: none, every query over every
/// position; above it a window, 0 none). A query `r` of the dispatch is at position `past + first + r`.
const ATTENTION_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y4: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

const HD4: u32 = HD4_u;
const TQ: u32 = TQ_u;
const TQ4: u32 = TQ_u / 4u;
const TK: u32 = TK_u;
const KS: u32 = HD4_u + 1u;

var<workgroup> kt: array<vec4<f32>, KT_LEN>;
var<workgroup> pt: array<vec4<f32>, PT_LEN>;
var<workgroup> al: array<f32, TQ_u>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let past = p[0].z;
    let rows = p[0].w;
    let kv_len = p[1].x;
    let scale = bitcast<f32>(p[1].y);
    let first = p[1].z;
    let full = (p[1].w & 1u) == 1u;
    let window = p[1].w >> 1u;
    let h = wg.x;
    let q0 = wg.y * TQ;
    let kh = h / (n_h / n_kv);
    let row4 = 2u * n_kv * HD4;
    // the tile's first and last queries' positions, and the keys they see
    let lo_pos = past + first + q0;
    let hi_pos = past + first + min(q0 + TQ, rows) - 1u;
    var hi = kv_len;
    var lo = 0u;
    if (!full) {
        hi = min(kv_len, hi_pos + 1u);
        if (window != 0u && lo_pos + 1u > window) { lo = lo_pos + 1u - window; }
    }
    let kb0 = lo / TK;
    let kb1 = (hi + TK - 1u) / TK;
    // the scores': key `sj` of a block, queries `sg * QPT ..` of the tile (a row past the dispatch's reads its last)
    let sj = li % TK;
    let sg = li / TK;
QB
    // the values': dims `4 vl ..`, queries `vg * QO ..`
    let vl = li % HD4;
    let vg = li / HD4;
    let vq4 = vg * (QO_u / 4u);
OACC
    var m = -3.0e38;
    var l = 0.0;
    for (var kb = kb0; kb < kb1; kb++) {
STAGE_K
        workgroupBarrier();
SCORES
        workgroupBarrier();
        // the weights, a thread a query; the block's values in its keys' place meanwhile
        if (li < TQ) {
            var mb = -3.0e38;
            for (var j = 0u; j < TK; j++) { mb = max(mb, pt[j * TQ4 + li / 4u][li % 4u]); }
            let mn = max(m, mb);
            let a = exp(m - mn);
            var sum = 0.0;
            for (var j = 0u; j < TK; j++) {
                let s = pt[j * TQ4 + li / 4u][li % 4u];
                var e = 0.0;
                if (s > -1.0e38) { e = exp(s - mn); }
                pt[j * TQ4 + li / 4u][li % 4u] = e;
                sum += e;
            }
            l = l * a + sum;
            m = mn;
            al[li] = a;
        }
STAGE_V
        workgroupBarrier();
VALUES
        workgroupBarrier();
    }
    if (li < TQ) { al[li] = select(0.0, 1.0 / l, l > 0.0); }
    workgroupBarrier();
STORE
}
"#;

/// [`ATTENTION_TILED`] for a head `hd` wide (64, 128 or 256): its tiles (64 queries by 16 keys, 32 by 8 at 256), the
/// per-thread parts unrolled (named, so registers: a thread's array indexed in a loop goes to memory).
fn attention_tiled(hd: usize) -> String {
    assert!(matches!(hd, 64 | 128 | 256), "a tiled attention's head {hd} wide");
    let hd4 = hd / 4;
    let (tq, tk) = if hd == 256 { (32usize, 8usize) } else { (64, 16) };
    let qpt = tq * tk / 256;
    let qo = tq * hd4 / 256;
    let staged = tk * hd4 / 256;
    let qb: String = (0..qpt).map(|i| format!("    let qb{i} = ((first + min(q0 + sg * {qpt}u + {i}u, rows - 1u)) * n_h + h) * HD4;\n")).collect();
    let oacc: String = (0..qo).map(|i| format!("    var o{i} = vec4<f32>();\n")).collect();
    let stage = |ofs: &str| -> String {
        (0..staged)
            .map(|e| {
                format!(
                    "        {{\n            let i = li + {e}u * 256u;\n            let t = kb * TK + i / HD4;\n            var v = vec4<f32>();\n            if (t < kv_len) {{ v = kv4[t * row4 + {ofs} + i % HD4]; }}\n            kt[(i / HD4) * KS + i % HD4] = v;\n        }}\n"
                )
            })
            .collect()
    };
    let mut scores = String::new();
    for i in 0..qpt {
        scores += &format!("        var s{i} = 0.0;\n");
    }
    scores += "        for (var d = 0u; d < HD4; d++) {\n            let k = kt[sj * KS + d];\n";
    for i in 0..qpt {
        scores += &format!("            s{i} += dot(q4[qb{i} + d], k);\n");
    }
    scores += "        }\n        let kp = kb * TK + sj;\n";
    for i in 0..qpt {
        scores += &format!(
            "        var w{i} = -3.0e38;\n        {{\n            let pos = past + first + q0 + sg * {qpt}u + {i}u;\n            var ok = kp < kv_len;\n            if (!full) {{ ok = ok && kp <= pos && (window == 0u || kp + window > pos); }}\n            if (ok) {{ w{i} = s{i} * scale; }}\n        }}\n"
        );
    }
    if qpt == 4 {
        scores += "        pt[sj * TQ4 + sg] = vec4<f32>(w0, w1, w2, w3);\n";
    } else {
        assert_eq!(qpt, 1);
        scores += "        pt[sj * TQ4 + sg / 4u][sg % 4u] = w0;\n";
    }
    let mut values = String::new();
    for i in 0..qo {
        values += &format!("        o{i} *= al[vg * {qo}u + {i}u];\n");
    }
    values += "        for (var j = 0u; j < TK; j++) {\n            let v = kt[j * KS + vl];\n";
    for c in 0..qo / 4 {
        values += &format!("            let p{c} = pt[j * TQ4 + vq4 + {c}u];\n");
        for (e, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            values += &format!("            o{} += p{c}.{comp} * v;\n", 4 * c + e);
        }
    }
    values += "        }\n";
    let store: String = (0..qo)
        .map(|i| format!("    {{\n        let r = q0 + vg * {qo}u + {i}u;\n        if (r < rows) {{ y4[((first + r) * n_h + h) * HD4 + vl] = o{i} * al[vg * {qo}u + {i}u]; }}\n    }}\n"))
        .collect();
    ATTENTION_TILED
        .replace("HD4_u", &format!("{hd4}u"))
        .replace("TQ_u", &format!("{tq}u"))
        .replace("TK_u", &format!("{tk}u"))
        .replace("QO_u", &format!("{qo}u"))
        .replace("KT_LEN", &(tk * (hd4 + 1)).to_string())
        .replace("PT_LEN", &(tk * tq / 4).to_string())
        .replace("QB\n", &qb)
        .replace("OACC\n", &oacc)
        .replace("STAGE_K\n", &stage("kh * HD4"))
        .replace("STAGE_V\n", &stage("(n_kv + kh) * HD4"))
        .replace("SCORES\n", &scores)
        .replace("VALUES\n", &values)
        .replace("STORE\n", &store)
}

/// Whether a prompt's attention of `rows` queries without tensor cores is tiled ([`ATTENTION_TILED`]): heads 64, 128 or
/// 256 wide, two tiles of queries or more (fewer, a check of drafts' few rows over a long cache, would be a workgroup a
/// head going through all of it alone: the runs' kernel spreads the positions over the GPU).
pub(crate) fn attention_tiled_for(rows: usize, head_dim: usize) -> bool {
    matches!(head_dim, 64 | 128 | 256) && rows >= 2 * if head_dim == 256 { 32 } else { 64 }
}

/// The out of [`Recorder::attention_rows_runs`]: the output `[rows, n_h, head_dim]`, then each (query, head, run)'s
/// weighted values, its largest score and its sum.
pub(crate) fn attention_runs_out_len(rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
    rows * n_h * head_dim + rows * n_h * kv_len.div_ceil(SPLIT).max(1) * (head_dim + 2)
}

/// The most work (FLOPs) one dispatch of a prompt's attention without tensor cores takes on: its queries in chunks
/// past it (a dispatch's own time under the OS's limit on a submission, Windows' 2 s, however slow the GPU).
const ATTENTION_DISPATCH_FLOPS: f64 = (1u64 << 37) as f64;

/// A prompt's causal attention on the tensor cores ([`attention_coop`]'s kernel before its head's width is put in), a
/// workgroup of 4 subgroups a (head `h`, 32 queries), the keys 32 at a time: each block's scores `q . k` (f16 into
/// f32), a subgroup's 16 queries by 16 keys, stored for the softmax; a first pass over the blocks finds each query's
/// largest score and its sum (a thread a query's 8 keys of a block, its 4 threads' joined), the second its weights
/// (as f16) and the values they weigh, a subgroup's 16 queries by half the head. `q16`: the queries `[rows, n_h, hd]`
/// as f16 (padded to 32 rows), `kv16` the cache's rows as f16 (padded to 32 positions); `y` the output `[rows, n_h,
/// hd]` (the padding's rows past it, in what is scratch).
const ATTENTION_COOP: &str = r#"enable f16;
enable wgpu_cooperative_matrix;
struct Params { n_h: u32, n_kv: u32, past: u32, rows: u32, kv_len: u32, scale: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(0) var<storage, read> kv16: array<f16>;
@group(0) @binding(1) var<storage, read> q16: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const HD: u32 = HEAD_DIMu;
// a block's scores [query][key] and its weights the same way as f16 (8 vec4s a query); each thread's largest score
// and sum, for its query's 4 threads to join
var<workgroup> s_sh: array<f32, 1024>;
var<workgroup> p_sh: array<vec4<f16>, 256>;
var<workgroup> m_sh: array<f32, 128>;
var<workgroup> l_sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let q0 = wg.y * 32u;
    let sg = li / 32u;
    // the subgroup's 16 queries, and its 16 keys of a block's scores (its half of the head's values)
    let rg = sg / 2u;
    let half = sg % 2u;
    let kh = h / (p.n_h / p.n_kv);
    let kvd = p.n_kv * HD;
    let row = 2u * kvd;
    let qs = p.n_h * HD;
    let scale = bitcast<f32>(p.scale);
    // the thread's query of the 32 and its 8 keys of a block
    let tr = li / 4u;
    let tc = (li % 4u) * 8u;
    let qpos = p.past + q0 + tr;
    // the blocks the last query sees
    let hi = min(p.kv_len, p.past + q0 + 32u);
    let blocks = (hi + 31u) / 32u;
    var m = -3.4e38;
    var l = 0.0;
    for (var kb = 0u; kb < blocks; kb++) {
SCORES
        for (var j = 0u; j < 8u; j++) {
            let kp = kb * 32u + tc + j;
            if (kp <= qpos && kp < p.kv_len) {
                let sv = s_sh[tr * 32u + tc + j] * scale;
                if (sv > m) {
                    l = l * exp(m - sv) + 1.0;
                    m = sv;
                } else {
                    l += exp(sv - m);
                }
            }
        }
        workgroupBarrier();
    }
    m_sh[li] = m;
    l_sh[li] = l;
    workgroupBarrier();
    let r4 = tr * 4u;
    let mq = max(max(m_sh[r4], m_sh[r4 + 1u]), max(m_sh[r4 + 2u], m_sh[r4 + 3u]));
    var lq = 0.0;
    for (var i = 0u; i < 4u; i++) {
        if (l_sh[r4 + i] > 0.0) { lq += l_sh[r4 + i] * exp(m_sh[r4 + i] - mq); }
    }
    let inv = 1.0 / lq;
DECLARE_O
    for (var kb = 0u; kb < blocks; kb++) {
SCORES
        var e = array<f32, 8>();
        for (var j = 0u; j < 8u; j++) {
            let kp = kb * 32u + tc + j;
            if (kp <= qpos && kp < p.kv_len) { e[j] = exp(s_sh[tr * 32u + tc + j] * scale - mq) * inv; }
        }
        let pi = (tr * 32u + tc) / 4u;
        p_sh[pi] = vec4<f16>(vec4<f32>(e[0], e[1], e[2], e[3]));
        p_sh[pi + 1u] = vec4<f16>(vec4<f32>(e[4], e[5], e[6], e[7]));
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk += 16u) {
            let ia = (rg * 512u + kk) / 4u;
            let s8 = 8u;
            let a = coopLoadT<coop_mat16x16<f16, A>>(&p_sh[ia], s8);
VALUES
        }
        workgroupBarrier();
    }
STORES
}
"#;

/// [`ATTENTION_COOP`]'s block of scores: the subgroup's 16 queries by 16 keys over the head (every index and stride a
/// `let` of its own, for naga's SPIR-V), stored for the softmax.
const ATTENTION_COOP_SCORES: &str = r#"        {
            var acc = coop_mat16x16<f32, C>();
            for (var dd = 0u; dd < HD; dd += 16u) {
                let ia = (q0 + rg * 16u) * qs + h * HD + dd;
                let ib = (kb * 32u + half * 16u) * row + kh * HD + dd;
                let a = coopLoadT<coop_mat16x16<f16, A>>(&q16[ia], qs);
                let b = coopLoad<coop_mat16x16<f16, B>>(&kv16[ib], row);
                acc = coopMultiplyAdd(a, b, acc);
            }
            let io = rg * 512u + half * 16u;
            let s32 = 32u;
            coopStoreT(acc, &s_sh[io], s32);
        }
        workgroupBarrier();
"#;

/// [`ATTENTION_COOP`] for a head `hd` wide (a multiple of 32, to 256): a subgroup's half of it `hd / 32` fragments.
fn attention_coop(hd: usize) -> String {
    let frags = hd / 32;
    let declare: String = (0..frags).map(|f| format!("    var o{f} = coop_mat16x16<f32, C>();\n")).collect();
    let values: String = (0..frags)
        .map(|f| format!("            {{\n                let ib = (kb * 32u + kk) * row + kvd + kh * HD + half * (HD / 2u) + {f}u * 16u;\n                let b = coopLoadT<coop_mat16x16<f16, B>>(&kv16[ib], row);\n                o{f} = coopMultiplyAdd(a, b, o{f});\n            }}\n"))
        .collect();
    let stores: String = (0..frags)
        .map(|f| format!("    {{\n        let io = (q0 + rg * 16u) * qs + h * HD + half * (HD / 2u) + {f}u * 16u;\n        coopStoreT(o{f}, &y[io], qs);\n    }}\n"))
        .collect();
    ATTENTION_COOP
        .replace("HEAD_DIM", &hd.to_string())
        .replace("SCORES\n", ATTENTION_COOP_SCORES)
        .replace("DECLARE_O\n", &declare)
        .replace("VALUES\n", &values)
        .replace("STORES\n", &stores)
}

/// [`attention_coop`]'s attention with the keys 128 a block (`attention_coop_wide`'s kernel before its head's width is
/// put in): a workgroup of 4 subgroups a (head `h`, 32 queries) still. A block's scores a subgroup's 32 queries by 32
/// keys, four sums at once (a step of the head two loads of queries and two of keys for four multiplies, where the
/// 32-key kernel's one sum took two loads a multiply and went a multiply after another); its softmax a thread a
/// query's 32 keys, four at a time; its values a subgroup's 32 queries by a quarter of the head, a step of 16 keys two
/// loads of weights and the quarter's of values. A fifth of the barriers a key, half the loads. `kv16` padded to 128
/// positions (zeros: a weight of none times a value).
const ATTENTION_COOP_WIDE: &str = r#"enable f16;
enable wgpu_cooperative_matrix;
struct Params { n_h: u32, n_kv: u32, past: u32, rows: u32, kv_len: u32, scale: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(0) var<storage, read> kv16: array<f16>;
@group(0) @binding(1) var<storage, read> q16: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const HD: u32 = HEAD_DIMu;
// a block's scores [query][key] (32 by 128) and its weights the same way as f16 (32 vec4s a query); each thread's
// largest score and sum, for its query's 4 threads to join
var<workgroup> s_sh: array<f32, 4096>;
var<workgroup> p_sh: array<vec4<f16>, 1024>;
var<workgroup> m_sh: array<f32, 128>;
var<workgroup> l_sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let q0 = wg.y * 32u;
    let sg = li / 32u;
    let kh = h / (p.n_h / p.n_kv);
    let kvd = p.n_kv * HD;
    let row = 2u * kvd;
    let qs = p.n_h * HD;
    let scale = bitcast<f32>(p.scale);
    // the thread's query of the 32 and its 32 keys of a block
    let tr = li / 4u;
    let tc = (li % 4u) * 32u;
    let sb = tr * 128u + tc;
LIMIT
    let blocks = (hi + 127u) / 128u;
    let none = vec4<f32>(-3.4e38);
    var m = -3.4e38;
    var l = 0.0;
    for (var kb = 0u; kb < blocks; kb++) {
SCORES
        // the block's largest of the thread's 32, then its sum against the largest so far
        var bm = none;
        for (var j = 0u; j < 32u; j += 4u) {
            let kp = vec4<u32>(kb * 128u + tc + j) + vec4<u32>(0u, 1u, 2u, 3u);
            let sv = vec4<f32>(s_sh[sb + j], s_sh[sb + j + 1u], s_sh[sb + j + 2u], s_sh[sb + j + 3u]) * scale;
            bm = max(bm, select(none, sv, (kp <= vec4<u32>(qpos)) & (kp < vec4<u32>(p.kv_len))));
        }
        let top = max(m, max(max(bm.x, bm.y), max(bm.z, bm.w)));
        var add = vec4<f32>(0.0);
        for (var j = 0u; j < 32u; j += 4u) {
            let kp = vec4<u32>(kb * 128u + tc + j) + vec4<u32>(0u, 1u, 2u, 3u);
            let sv = vec4<f32>(s_sh[sb + j], s_sh[sb + j + 1u], s_sh[sb + j + 2u], s_sh[sb + j + 3u]) * scale;
            add += select(vec4<f32>(0.0), exp(sv - vec4<f32>(top)), (kp <= vec4<u32>(qpos)) & (kp < vec4<u32>(p.kv_len)));
        }
        l = l * exp(m - top) + add.x + add.y + add.z + add.w;
        m = top;
        workgroupBarrier();
    }
    m_sh[li] = m;
    l_sh[li] = l;
    workgroupBarrier();
    let r4 = tr * 4u;
    let mq = max(max(m_sh[r4], m_sh[r4 + 1u]), max(m_sh[r4 + 2u], m_sh[r4 + 3u]));
    var lq = 0.0;
    for (var i = 0u; i < 4u; i++) {
        if (l_sh[r4 + i] > 0.0) { lq += l_sh[r4 + i] * exp(m_sh[r4 + i] - mq); }
    }
    let inv = 1.0 / lq;
DECLARE_O
    for (var kb = 0u; kb < blocks; kb++) {
SCORES
        for (var j = 0u; j < 32u; j += 4u) {
            let kp = vec4<u32>(kb * 128u + tc + j) + vec4<u32>(0u, 1u, 2u, 3u);
            let sv = vec4<f32>(s_sh[sb + j], s_sh[sb + j + 1u], s_sh[sb + j + 2u], s_sh[sb + j + 3u]) * scale;
            let e = select(vec4<f32>(0.0), exp(sv - vec4<f32>(mq)) * inv, (kp <= vec4<u32>(qpos)) & (kp < vec4<u32>(p.kv_len)));
            p_sh[(sb + j) / 4u] = vec4<f16>(e);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 128u; kk += 16u) {
            let ia0 = kk / 4u;
            let ia1 = (2048u + kk) / 4u;
            let s32 = 32u;
            let pa = coopLoadT<coop_mat16x16<f16, A>>(&p_sh[ia0], s32);
            let pb = coopLoadT<coop_mat16x16<f16, A>>(&p_sh[ia1], s32);
VALUES
        }
        workgroupBarrier();
    }
STORES
}
"#;

/// [`ATTENTION_COOP_WIDE`]'s block of scores: the subgroup's 32 queries by its 32 keys of the block over the head, four
/// sums (every index and stride a `let` of its own, for naga's SPIR-V), stored for the softmax.
const ATTENTION_COOP_WIDE_SCORES: &str = r#"        {
            var a00 = coop_mat16x16<f32, C>();
            var a01 = coop_mat16x16<f32, C>();
            var a10 = coop_mat16x16<f32, C>();
            var a11 = coop_mat16x16<f32, C>();
            for (var dd = 0u; dd < HD; dd += 16u) {
                let iq0 = q0 * qs + h * HD + dd;
                let iq1 = (q0 + 16u) * qs + h * HD + dd;
                let ik0 = (kb * 128u + sg * 32u) * row + kh * HD + dd;
                let ik1 = (kb * 128u + sg * 32u + 16u) * row + kh * HD + dd;
                let qa = coopLoadT<coop_mat16x16<f16, A>>(&q16[iq0], qs);
                let qb = coopLoadT<coop_mat16x16<f16, A>>(&q16[iq1], qs);
                let ka = coopLoad<coop_mat16x16<f16, B>>(&kv16[ik0], row);
                let kc = coopLoad<coop_mat16x16<f16, B>>(&kv16[ik1], row);
                a00 = coopMultiplyAdd(qa, ka, a00);
                a01 = coopMultiplyAdd(qa, kc, a01);
                a10 = coopMultiplyAdd(qb, ka, a10);
                a11 = coopMultiplyAdd(qb, kc, a11);
            }
            let io00 = sg * 32u;
            let io01 = sg * 32u + 16u;
            let io10 = 2048u + sg * 32u;
            let io11 = 2048u + sg * 32u + 16u;
            let s128 = 128u;
            coopStoreT(a00, &s_sh[io00], s128);
            coopStoreT(a01, &s_sh[io01], s128);
            coopStoreT(a10, &s_sh[io10], s128);
            coopStoreT(a11, &s_sh[io11], s128);
        }
        workgroupBarrier();
"#;

/// [`ATTENTION_COOP_WIDE`] for a head `hd` wide (64, 128 or 256: a subgroup's quarter of it `hd / 64` fragments), every
/// query over every position with `full` (no causal mask).
fn attention_coop_wide(hd: usize, full: bool) -> String {
    let frags = hd / 64;
    let declare: String = (0..2).flat_map(|q| (0..frags).map(move |f| format!("    var o{q}_{f} = coop_mat16x16<f32, C>();\n"))).collect();
    let values: String = (0..frags)
        .map(|f| {
            format!(
                "            {{\n                let ib = (kb * 128u + kk) * row + kvd + kh * HD + sg * (HD / 4u) + {f}u * 16u;\n                let vb = coopLoadT<coop_mat16x16<f16, B>>(&kv16[ib], row);\n                o0_{f} = coopMultiplyAdd(pa, vb, o0_{f});\n                o1_{f} = coopMultiplyAdd(pb, vb, o1_{f});\n            }}\n"
            )
        })
        .collect();
    let stores: String = (0..2)
        .flat_map(|q| (0..frags).map(move |f| format!("    {{\n        let io = (q0 + {q}u * 16u) * qs + h * HD + sg * (HD / 4u) + {f}u * 16u;\n        coopStoreT(o{q}_{f}, &y[io], qs);\n    }}\n")))
        .collect();
    let limit = if full {
        "    // every position, the last query's and the first's alike\n    let qpos = p.kv_len;\n    let hi = p.kv_len;"
    } else {
        "    let qpos = p.past + q0 + tr;\n    // the blocks the last query sees\n    let hi = min(p.kv_len, p.past + q0 + 32u);"
    };
    ATTENTION_COOP_WIDE
        .replace("HEAD_DIM", &hd.to_string())
        .replace("LIMIT", limit)
        .replace("SCORES\n", ATTENTION_COOP_WIDE_SCORES)
        .replace("DECLARE_O\n", &declare)
        .replace("VALUES\n", &values)
        .replace("STORES\n", &stores)
}

/// A prompt's (or a diffusion's) attention on the tensor cores in one pass over the keys (`attention_coop_one`'s
/// kernel before its head's width is put in): [`ATTENTION_COOP_WIDE`]'s blocks of 128 keys, each block's scores made
/// once. A query's weights are `exp(score - c)` against a reference `c` of its own, its first block's largest score:
/// a later block's largest more than 8 past it makes that the reference, and the query's sums so far (its row of the
/// values' sums, and its weights' sum) are scaled down to it. The tensor cores' sums cannot be scaled a row each
/// where they are, so a workgroup with such a query stores its sums to the output, scales the rows there and loads
/// them back (rare: a block where some query's largest score grew by more than 8); at the end the sums are stored and
/// each row divided by its weights' sum the same way. The scores twice was 8.1 of a layer's 18.5 ms (2,048 queries
/// over 15,360 positions), the exponentials none of it.
const ATTENTION_COOP_ONE: &str = r#"enable f16;
enable wgpu_cooperative_matrix;
struct Params { n_h: u32, n_kv: u32, past: u32, rows: u32, kv_len: u32, scale: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(0) var<storage, read> kv16: array<f16>;
@group(0) @binding(1) var<storage, read> q16: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const HD: u32 = HEAD_DIMu;
// a block's scores [query][key] (32 by 128, 32 vec4s a query: a thread's 32 of them 8 loads) and its weights the same
// way as f16; each thread's largest score of the block (then, at the end, its sum), and whether some query's
// reference moves at this block (set by a thread that sees it, cleared by the first once the sums are scaled)
var<workgroup> s_sh: array<vec4<f32>, 1024>;
var<workgroup> p_sh: array<vec4<f16>, 1024>;
var<workgroup> m_sh: array<f32, 128>;
var<workgroup> moved: u32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let q0 = wg.y * 32u;
    let sg = li / 32u;
    let kh = h / (p.n_h / p.n_kv);
    let qs = p.n_h * HD;
    let scale = bitcast<f32>(p.scale);
    // the cache's f16 copy a fragment at a time ([`crate::shaders::KV_F16_TILED`]): the 16-position blocks a KV head
    // has, and where the values' fragments begin
    let nb = ((p.kv_len + 127u) / 128u) * 8u;
    let voff = p.n_kv * nb * HD * 16u;
    // the thread's query of the 32, its 32 keys of a block, and its quarter of the query's row of the output
    let tr = li / 4u;
    let tc = (li % 4u) * 32u;
    let sb = tr * 128u + tc;
    let r4 = tr * 4u;
    let yb = (q0 + tr) * qs + h * HD + (li % 4u) * (HD / 4u);
LIMIT
    let blocks = (hi + 127u) / 128u;
    let none = vec4<f32>(-3.4e38);
    // the query's reference (none yet) and the thread's keys' weights' sum against it
    var c = -3.4e38;
    var l = 0.0;
DECLARE_O
    for (var kb = 0u; kb < blocks; kb++) {
SCORES
        // the thread's 32 scores of the block (those the query does not see none), and their largest
LOAD_S
        let tm = max(max(bm.x, bm.y), max(bm.z, bm.w));
        m_sh[li] = tm;
        // (a thread whose largest is more than 8 past its query's reference: that reference moves, as below)
        if (c > -1.0e38 && tm > c + 8.0) { moved = 1u; }
        let go = workgroupUniformLoad(&moved);
        // the block's largest for the query: its first is the reference, one more than 8 past the reference the new
        // one, what is summed so far scaled down to it
        let bq = max(max(m_sh[r4], m_sh[r4 + 1u]), max(m_sh[r4 + 2u], m_sh[r4 + 3u]));
        var factor = 1.0;
        if (bq > -1.0e38) {
            if (c < -1.0e38) {
                c = bq;
            } else if (bq > c + 8.0) {
                factor = exp(c - bq);
                c = bq;
            }
        }
        l *= factor;
        if (go != 0u) {
STORES
            storageBarrier();
            workgroupBarrier();
            if (factor != 1.0) {
                for (var d = 0u; d < HD / 4u; d++) { y[yb + d] = y[yb + d] * factor; }
            }
            if (li == 0u) { moved = 0u; }
            storageBarrier();
            workgroupBarrier();
RELOADS
        }
        // (a query that has seen no key yet: no weights)
        let on = select(0.0, 1.0, c > -1.0e38);
        var add = vec4<f32>(0.0);
WEIGHTS
        l += add.x + add.y + add.z + add.w;
        workgroupBarrier();
        for (var kk = 0u; kk < 128u; kk += 16u) {
            let ia0 = kk / 4u;
            let ia1 = (2048u + kk) / 4u;
            let s32 = 32u;
            let pa = coopLoadT<coop_mat16x16<f16, A>>(&p_sh[ia0], s32);
            let pb = coopLoadT<coop_mat16x16<f16, A>>(&p_sh[ia1], s32);
VALUES
        }
        workgroupBarrier();
    }
    // the query's weights' sum (its 4 threads': one reference), and its row over it
    m_sh[li] = l;
    workgroupBarrier();
    let lq = m_sh[r4] + m_sh[r4 + 1u] + m_sh[r4 + 2u] + m_sh[r4 + 3u];
    let inv = 1.0 / max(lq, 1.0e-30);
STORES
    storageBarrier();
    workgroupBarrier();
    for (var d = 0u; d < HD / 4u; d++) { y[yb + d] = y[yb + d] * inv; }
}
"#;

/// [`ATTENTION_COOP_ONE`] for a head `hd` wide (64, 128 or 256), every query over every position with `full`.
fn attention_coop_one(hd: usize, full: bool) -> String {
    let frags = hd / 64;
    let declare: String = (0..2).flat_map(|q| (0..frags).map(move |f| format!("    var o{q}_{f} = coop_mat16x16<f32, C>();\n"))).collect();
    let values: String = (0..frags)
        .map(|f| {
            format!(
                "            {{\n                let ib = voff + ((kh * nb + kb * 8u + kk / 16u) * (HD / 16u) + sg * (HD / 64u) + {f}u) * 256u;\n                let sv = 16u;\n                let vb = coopLoadT<coop_mat16x16<f16, B>>(&kv16[ib], sv);\n                o0_{f} = coopMultiplyAdd(pa, vb, o0_{f});\n                o1_{f} = coopMultiplyAdd(pb, vb, o1_{f});\n            }}\n"
            )
        })
        .collect();
    // (a subgroup's sums' places in the output: its 32 queries' rows, its quarter of the head)
    let place = |q: usize, f: usize| format!("(q0 + {q}u * 16u) * qs + h * HD + sg * (HD / 4u) + {f}u * 16u");
    let stores: String = (0..2).flat_map(|q| (0..frags).map(move |f| (q, f))).map(|(q, f)| format!("    {{\n        let io = {};\n        coopStoreT(o{q}_{f}, &y[io], qs);\n    }}\n", place(q, f))).collect();
    let reloads: String = (0..2).flat_map(|q| (0..frags).map(move |f| (q, f))).map(|(q, f)| format!("    {{\n        let io = {};\n        o{q}_{f} = coopLoadT<coop_mat16x16<f32, C>>(&y[io], qs);\n    }}\n", place(q, f))).collect();
    // the thread's scores four at a time, each its own name (an array of them would be memory, not registers)
    let load: String = (0..8)
        .map(|j| {
            format!(
                "        let k{j} = vec4<u32>(kb * 128u + tc + {o}u) + vec4<u32>(0u, 1u, 2u, 3u);\n        let v{j} = select(none, s_sh[(sb + {o}u) / 4u] * scale, (k{j} <= vec4<u32>(qpos)) & (k{j} < vec4<u32>(p.kv_len)));\n",
                o = 4 * j
            )
        })
        .chain(std::iter::once(format!("        let bm = {};\n", (1..8).fold("v0".to_string(), |m, j| format!("max({m}, v{j})")))))
        .collect();
    // (a weight as the tensor cores read it, f16, and the sum of those)
    let weights: String = (0..8).map(|j| format!("        let w{j} = vec4<f16>(exp(v{j} - vec4<f32>(c)) * on);\n        p_sh[(sb + {o}u) / 4u] = w{j};\n        add += vec4<f32>(w{j});\n", o = 4 * j)).collect();
    let limit = if full {
        "    // every position, the last query's and the first's alike\n    let qpos = p.kv_len;\n    let hi = p.kv_len;"
    } else {
        "    let qpos = p.past + q0 + tr;\n    // the blocks the last query sees\n    let hi = min(p.kv_len, p.past + q0 + 32u);"
    };
    // (the scores' fragments stored into vec4s: their places and stride in those; the keys' fragments each a run)
    let mut scores = ATTENTION_COOP_WIDE_SCORES.to_string();
    for (scalars, fours) in [
        ("let ik0 = (kb * 128u + sg * 32u) * row + kh * HD + dd;", "let ik0 = ((kh * nb + kb * 8u + sg * 2u) * (HD / 16u) + dd / 16u) * 256u;"),
        ("let ik1 = (kb * 128u + sg * 32u + 16u) * row + kh * HD + dd;", "let ik1 = ((kh * nb + kb * 8u + sg * 2u + 1u) * (HD / 16u) + dd / 16u) * 256u;"),
        ("let ka = coopLoad<coop_mat16x16<f16, B>>(&kv16[ik0], row);", "let sk = 16u;\n                let ka = coopLoad<coop_mat16x16<f16, B>>(&kv16[ik0], sk);"),
        ("let kc = coopLoad<coop_mat16x16<f16, B>>(&kv16[ik1], row);", "let kc = coopLoad<coop_mat16x16<f16, B>>(&kv16[ik1], sk);"),
    ] {
        assert_eq!(scores.matches(scalars).count(), 1, "the scores' keys");
        scores = scores.replace(scalars, fours);
    }
    for (scalars, fours) in [("let io00 = sg * 32u;", "let io00 = sg * 8u;"), ("let io01 = sg * 32u + 16u;", "let io01 = sg * 8u + 4u;"), ("let io10 = 2048u + sg * 32u;", "let io10 = 512u + sg * 8u;"), ("let io11 = 2048u + sg * 32u + 16u;", "let io11 = 512u + sg * 8u + 4u;"), ("let s128 = 128u;", "let s128 = 32u;")] {
        assert_eq!(scores.matches(scalars).count(), 1, "the scores' places");
        scores = scores.replace(scalars, fours);
    }
    ATTENTION_COOP_ONE
        .replace("HEAD_DIM", &hd.to_string())
        .replace("LIMIT", limit)
        .replace("SCORES\n", &scores)
        .replace("LOAD_S\n", &load)
        .replace("WEIGHTS\n", &weights)
        .replace("DECLARE_O\n", &declare)
        .replace("VALUES\n", &values)
        .replace("RELOADS\n", &reloads)
        .replace("STORES\n", &stores)
}

/// [`attention_coop`] with no causal mask: every query over all `kv_len` positions.
fn attention_coop_full(hd: usize) -> String {
    let causal = "    let qpos = p.past + q0 + tr;\n    // the blocks the last query sees\n    let hi = min(p.kv_len, p.past + q0 + 32u);\n";
    let src = attention_coop(hd);
    assert_eq!(src.matches(causal).count(), 1, "the causal limit");
    src.replace(causal, "    // every position, the last query's and the first's alike\n    let qpos = p.kv_len;\n    let hi = p.kv_len;\n")
}

/// A split tensor-core matmul's sums put together: `y[i]` the sum of `x`'s `p[0].y` parts of `p[0].x` each.
const COOP_SUM: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p[0].x;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= n) { return; }
    var acc = x[i];
    for (var z = 1u; z < p[0].y; z++) { acc += x[z * n + i]; }
    y[i] = acc;
}
"#;

/// [`ATTENTION_COOP`] for QSA's queries past its dense span ([`ChainRecorder::qsa_attention`] of a prompt's rows): every
/// position up to the query's own on the tensor cores, those of a block the query did not keep left out (its blocks
/// of `p._pad0` positions, a bit each in `mask`, `p._pad1` words a query: its incomplete tail block's always in).
fn attention_coop_masked(hd: usize) -> String {
    attention_coop(hd)
        .replace("@group(0) @binding(2) var<storage, read_write> y: array<f32>;", "@group(0) @binding(2) var<storage, read> mask: array<u32>;\n@group(0) @binding(6) var<storage, read_write> y: array<f32>;")
        .replace("@group(0) @binding(3) var<uniform> p: Params;", "@group(0) @binding(8) var<uniform> p: Params;\n\n// whether query `row` (at `qpos`) attends to position `kp`: its tail block's, or a block it kept\nfn kept(row: u32, qpos: u32, kp: u32) -> bool {\n    let b = kp / p._pad0;\n    if (b >= (qpos + 1u) / p._pad0) { return true; }\n    return ((mask[row * p._pad1 + b / 32u] >> (b % 32u)) & 1u) == 1u;\n}")
        .replace("if (kp <= qpos && kp < p.kv_len) {", "if (kp <= qpos && kp < p.kv_len && kept(q0 + tr, qpos, kp)) {")
}

/// [`attention_coop_masked`] in one pass over the keys, 128 of them a block ([`attention_coop_one`] with the mask: a
/// query's keys of a block it did not keep count for nothing, as those past it; its reference is its first block
/// with a key it attends to). A prompt's chunk of 512 at 14,336 positions (24 heads of 256 over 2 KV heads): its
/// twelve layers' 81 ms with the scores twice and the keys 32 a block.
fn attention_coop_one_masked(hd: usize) -> String {
    let mut src = attention_coop_one(hd, false)
        .replace("@group(0) @binding(2) var<storage, read_write> y: array<f32>;", "@group(0) @binding(2) var<storage, read> mask: array<u32>;\n@group(0) @binding(6) var<storage, read_write> y: array<f32>;")
        .replace("@group(0) @binding(3) var<uniform> p: Params;", "@group(0) @binding(8) var<uniform> p: Params;\n\n// whether query `row` (at `qpos`) attends to position `kp`: its tail block's, or a block it kept\nfn kept(row: u32, qpos: u32, kp: u32) -> bool {\n    let b = kp / p._pad0;\n    if (b >= (qpos + 1u) / p._pad0) { return true; }\n    return ((mask[row * p._pad1 + b / 32u] >> (b % 32u)) & 1u) == 1u;\n}");
    assert!(src.contains("fn kept(") && src.contains("binding(6) var<storage, read_write> y"), "the mask's bindings");
    for j in 0..8 {
        let seen = format!("(k{j} <= vec4<u32>(qpos)) & (k{j} < vec4<u32>(p.kv_len))");
        assert_eq!(src.matches(&seen).count(), 1, "the keys a query sees");
        src = src.replace(&seen, &format!("{seen} & vec4<bool>(kept(q0 + tr, qpos, k{j}.x), kept(q0 + tr, qpos, k{j}.y), kept(q0 + tr, qpos, k{j}.z), kept(q0 + tr, qpos, k{j}.w))"));
    }
    src
}

/// [`QSA_SCORES`] for `heads` index heads (1 to 8) of a width a multiple of 4: each block's pooled key read once, a
/// vec4 at a time, for every head's sum (a sum a load a head: a prompt's chunk of 512 over 3,584 blocks, 4 heads of
/// 128, 8 ms a layer).
fn qsa_scores4(heads: usize) -> String {
    let each = |f: &dyn Fn(usize) -> String| (0..heads).map(f).collect::<String>();
    let total = (0..heads).map(|h| format!("max(a{h}.x + a{h}.y + a{h}.z + a{h}.w, 0.0)")).collect::<Vec<_>>().join(" + ");
    QSA_SCORES4
        .replace("HEADS", &heads.to_string())
        .replace("SUMS\n", &each(&|h| format!("        var a{h} = vec4<f32>(0.0);\n")))
        .replace("STEPS\n", &each(&|h| format!("            a{h} += qs[{h}u * d4 + i] * v;\n")))
        .replace("TOTAL", &total)
}

const QSA_SCORES4: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pooled: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> qs: array<vec4<f32>, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let d4 = p[0].z / 4u;
    let nb = p[0].w;
    let first = p[1].x;
    let ratio = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let r = wg.y;
    let hd4 = HEADSu * d4;
    for (var i = li; i < hd4; i += 256u) { qs[i] = q[r * hd4 + i]; }
    workgroupBarrier();
    let j = wg.x * 256u + li;
    if (j >= nb) { return; }
    var total = bitcast<f32>(0xff800000u);
    if (j < (first + r + 1u) / ratio) {
SUMS
        for (var i = 0u; i < d4; i++) {
            let v = pooled[j * d4 + i];
STEPS
        }
        total = (TOTAL) * scale;
    }
    scores[r * nb + j] = total;
}
"#;

/// QSA's kept blocks of each query (`list`: `keep` a query, ascending, its first `min(visible, keep)` its own) as a
/// bitmask (`mask`: `p[0].z` words a query), a thread a word: its blocks found in the list by bisection. `p[0]`: the
/// queries, keep, words a query, the first query's position; `p[1].x`: the positions a block.
const QSA_MASK: &str = r#"
@group(0) @binding(0) var<storage, read> list: array<u32>;
@group(0) @binding(6) var<storage, read_write> mask: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rows = p[0].x;
    let keep = p[0].y;
    let mw = p[0].z;
    let first = p[0].w;
    let ratio = p[1].x;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= rows * mw) { return; }
    let r = i / mw;
    let w = i % mw;
    let count = min((first + r + 1u) / ratio, keep);
    let base = r * keep;
    // the first kept block at or past 32 w
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (list[base + mid] < 32u * w) { lo = mid + 1u; } else { hi = mid; }
    }
    var bits = 0u;
    for (var j = lo; j < count; j++) {
        let b = list[base + j];
        if (b >= 32u * w + 32u) { break; }
        bits |= 1u << (b - 32u * w);
    }
    mask[i] = bits;
}
"#;

/// The runs of [`ATTENTION_ROWS_PART`] put together, a workgroup a (head, query); a run with nothing (`l` 0) adds
/// nothing. `p` as for the parts.
const ATTENTION_ROWS_JOIN: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let hd = p[0].z;
    let runs = p[1].y;
    let rows = p[1].w;
    let h = wg.x;
    let s = wg.y;
    let first = (s * n_h + h) * runs;
    let ml = rows * n_h * hd + rows * n_h * runs * hd + first * 2u;
    var m = -3.4e38;
    for (var r = 0u; r < runs; r++) {
        if (y[ml + r * 2u + 1u] > 0.0) { m = max(m, y[ml + r * 2u]); }
    }
    var l = 0.0;
    for (var r = 0u; r < runs; r++) {
        let lr = y[ml + r * 2u + 1u];
        if (lr > 0.0) { l += exp(y[ml + r * 2u] - m) * lr; }
    }
    for (var d = li; d < hd; d += 256u) {
        var acc = 0.0;
        for (var r = 0u; r < runs; r++) {
            if (y[ml + r * 2u + 1u] > 0.0) {
                acc += exp(y[ml + r * 2u] - m) * y[rows * n_h * hd + (first + r) * hd + d];
            }
        }
        y[(s * n_h + h) * hd + d] = acc / l;
    }
}
"#;

/// [`ChainRecorder::argmax_softmax`] of `p[0].x` logits, one workgroup: each thread's largest (the first of equals in
/// its stride) and the sum of exponentials against it as it goes, then the threads' combined, the lower index of
/// equals taken.
const ARGMAX_SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> mv: array<f32, 256>;
var<workgroup> mi: array<u32, 256>;
var<workgroup> ms: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    var m = -3.4e38;
    var idx = 0xffffffffu;
    var s = 0.0;
    for (var i = t; i < n; i += 256u) {
        let v = x[i];
        if (v > m) {
            s = s * exp(m - v) + 1.0;
            m = v;
            idx = i;
        } else {
            s += exp(v - m);
        }
    }
    mv[t] = m;
    mi[t] = idx;
    ms[t] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (t < st) {
            let m1 = mv[t];
            let m2 = mv[t + st];
            let i1 = mi[t];
            let i2 = mi[t + st];
            let take = m2 > m1 || (m2 == m1 && i2 < i1);
            let mm = select(m1, m2, take);
            ms[t] = ms[t] * exp(m1 - mm) + ms[t + st] * exp(m2 - mm);
            mv[t] = mm;
            mi[t] = select(i1, i2, take);
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        out[0] = bitcast<f32>(mi[0]);
        out[1] = mv[0];
        out[2] = ms[0];
    }
}
"#;

/// `y[p[0].y + i] = x[p[0].z + i]` for `i < p[0].x`.
const COPY: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[p[0].y + i] = x[p[0].z + i]; }
}
"#;

/// `y[r * p[0].x + i] = x[r * p[0].z + p[0].w + i]` for `r < p[0].y` rows of `p[0].x`: columns of each row.
const COPY_COLS: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = p[0].x;
    let i = id.x + id.y * 16776960u;
    if (i < width * p[0].y) { y[i] = x[(i / width) * p[0].z + p[0].w + i % width]; }
}
"#;

/// Each row of `x` (rows of `p[0].x`, a workgroup a row: `wg.x + wg.y * 65535`) normed as `p[1].y` says (0 none, 1
/// over its RMS, 2 a layer norm: the mean, then the mean square of the deviations; no weights, `eps` the bits of
/// `p[0].y`) into `y`, times `1 + mods[p[0].z + i]` and plus `mods[p[0].w + i]` unless `p[0].w` is all ones.
/// `p[1].x`: the rows.
const LAYERNORM_MOD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let r = wg.x + wg.y * 65535u;
    if (r >= p[1].x) { return; }
    let at = r * n;
    let mode = p[1].y & 3u;
    // a clean row (before p[1].z, or from p[1].w on) modulated by the second set
    let second = select(0u, p[1].y >> 2u, r < p[1].z || r >= p[1].w);
    var mean = 0.0;
    if (mode == 2u) {
        var s = 0.0;
        for (var i = li; i < n; i += 256u) { s += x[at + i]; }
        part[li] = s;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
            if (li < stride) { part[li] += part[li + stride]; }
            workgroupBarrier();
        }
        mean = part[0] / f32(n);
        workgroupBarrier();
    }
    var inv = 1.0;
    if (mode != 0u) {
        var q = 0.0;
        for (var i = li; i < n; i += 256u) { let d = x[at + i] - mean; q += d * d; }
        part[li] = q;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
            if (li < stride) { part[li] += part[li + stride]; }
            workgroupBarrier();
        }
        inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    }
    let shifted = p[0].w != 0xffffffffu;
    for (var i = li; i < n; i += 256u) {
        var v = (x[at + i] - mean) * inv * (1.0 + mods[second + p[0].z + i]);
        if (shifted) { v += mods[second + p[0].w + i]; }
        y[at + i] = v;
    }
}
"#;

/// `y[i] *= 2 sigmoid(logits[i / p[0].x])` for `i < p[0].y` (`p[0].x` a head's width): a gated attention's heads.
/// [`ChainRecorder::group_norm_rows`]'s sums: a workgroup a (group, 256 pixels), each thread a pixel's values of the
/// group (their sum and their squares' less the group's first value: sums of what is near zero keep their digits),
/// the workgroup's sums into `stats[(g * chunks + j) * 2..]`. `p[0]`: pixels, channels, groups, chunks.
const GROUP_NORM_SUMS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<vec2<f32>, 256>;
var<workgroup> some: array<vec2<f32>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let pixels = p[0].x;
    let c = p[0].y;
    let cpg = c / p[0].z;
    let chunks = p[0].w;
    let g = wg.x;
    let j = wg.y;
    let px = j * 256u + li;
    // (the group's first value: what every value is taken from, for the sums' digits)
    let pilot = x[g * cpg];
    var sums = vec2<f32>(0.0);
    if (px < pixels) {
        let at = px * c + g * cpg;
        for (var i = 0u; i < cpg; i++) {
            let v = x[at + i] - pilot;
            sums += vec2<f32>(v, v * v);
        }
    }
    part[li] = sums;
    workgroupBarrier();
    if (li < 16u) {
        var t = vec2<f32>(0.0);
        for (var i = 0u; i < 16u; i++) { t += part[li * 16u + i]; }
        some[li] = t;
    }
    workgroupBarrier();
    if (li == 0u) {
        var t = vec2<f32>(0.0);
        for (var i = 0u; i < 16u; i++) { t += some[i]; }
        let o = (g * chunks + j) * 2u;
        stats[o] = t.x;
        stats[o + 1u] = t.y;
    }
}
"#;

/// [`ChainRecorder::group_norm_rows`]'s mean and scale: a workgroup a group, its chunks' sums added up, the group's
/// mean and `1 / sqrt(variance + eps)` after the chunks' sums (`stats[(groups * chunks + g) * 2..]`). `p[0]`: pixels,
/// channels, groups, chunks; `p[1].x`: eps's bits.
const GROUP_NORM_STATS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<vec2<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let cpg = p[0].y / p[0].z;
    let chunks = p[0].w;
    let g = wg.x;
    var t = vec2<f32>(0.0);
    for (var j = li; j < chunks; j += 64u) {
        let o = (g * chunks + j) * 2u;
        t += vec2<f32>(stats[o], stats[o + 1u]);
    }
    part[li] = t;
    workgroupBarrier();
    if (li == 0u) {
        var all = vec2<f32>(0.0);
        for (var i = 0u; i < 64u; i++) { all += part[i]; }
        let n = f32(p[0].x) * f32(cpg);
        let m = all.x / n;
        let variance = max(all.y / n - m * m, 0.0);
        let o = (p[0].z * chunks + g) * 2u;
        stats[o] = x[g * cpg] + m;
        stats[o + 1u] = inverseSqrt(variance + bitcast<f32>(p[1].x));
    }
}
"#;

/// [`ChainRecorder::group_norm_rows`]'s rows: each value less its group's mean, times its scale and its channel's
/// weight, plus the channel's bias, through SiLU where `p[1].y` is 1. `p[0]`: pixels, channels, groups, chunks.
const GROUP_NORM_APPLY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].y;
    if (i >= p[0].x * c) { return; }
    let ch = i % c;
    let o = (p[0].z * p[0].w + ch / (c / p[0].z)) * 2u;
    let v = (x[i] - stats[o]) * stats[o + 1u] * weight[ch] + bias[ch];
    y[i] = select(v, v / (1.0 + exp(-v)), p[1].y == 1u);
}
"#;

/// [`ChainRecorder::geglu_rows`]: `p[0]` rows and `ff`.
const GEGLU_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].y;
    if (i >= p[0].x * ff) { return; }
    let at = (i / ff) * 2u * ff + i % ff;
    let g = x[at + ff];
    let z = abs(g) * 0.7071067811865476;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign(g) * (1.0 - poly * exp(-z * z));
    y[i] = x[at] * 0.5 * g * (1.0 + erf);
}
"#;

/// `y[(oy, ox), c] = x[(2 oy, 2 ox), c]`: `p[0]` the input's rows, columns and channels.
const SUBSAMPLE2X_EVEN_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let ow = w / 2u;
    if (i >= (h / 2u) * ow * c) { return; }
    let px = i / c;
    y[i] = x[(2u * (px / ow) * w + 2u * (px % ow)) * c + i % c];
}
"#;

/// [`ChainRecorder::nag_mix`]: a workgroup a row, each thread its share of the row's values four at a time; the row's
/// L1 norms (the guided output's and the plain one's) summed by sixteens through the workgroup's memory. `p[0]`: the
/// row's width (a multiple of 4), rows, and the bits of scale and tau; `p[1].x`: alpha's.
const NAG_MIX: &str = r#"
@group(0) @binding(0) var<storage, read> neg: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<vec2<f32>, 256>;
var<workgroup> some: array<vec2<f32>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let w4 = p[0].x / 4u;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].y) { return; }
    let scale = bitcast<f32>(p[0].z);
    let tau = bitcast<f32>(p[0].w);
    let alpha = bitcast<f32>(p[1].x);
    let at = row * w4;
    var sums = vec2<f32>(0.0);
    for (var c = li; c < w4; c += 256u) {
        let a = pos[at + c];
        let g = abs(a * scale - neg[at + c] * (scale - 1.0));
        let q = abs(a);
        sums += vec2<f32>(g.x + g.y + g.z + g.w, q.x + q.y + q.z + q.w);
    }
    part[li] = sums;
    workgroupBarrier();
    if (li < 16u) {
        var s = vec2<f32>(0.0);
        for (var i = 0u; i < 16u; i++) { s += part[li * 16u + i]; }
        some[li] = s;
    }
    workgroupBarrier();
    var total = vec2<f32>(0.0);
    for (var i = 0u; i < 16u; i++) { total += some[i]; }
    // the guided row within tau times the plain one's size
    let factor = clamp(tau * (total.y + 1.0e-6) / max(total.x, 1.0e-30), 0.0, 1.0);
    for (var c = li; c < w4; c += 256u) {
        let a = pos[at + c];
        let g = a * scale - neg[at + c] * (scale - 1.0);
        pos[at + c] = g * (factor * alpha) + a * (1.0 - alpha);
    }
}
"#;

const HEAD_GATE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].y) { y[i] *= 2.0 / (1.0 + exp(-logits[i / p[0].x])); }
}
"#;

/// `x[i] += y[i] * g` for `i < p[0].x * p[0].y` (rows of `p[0].x`), `g` `mods[p[0].z + i % p[0].x]`, its tanh where
/// `p[0].w` is 1.
/// `y[(oy, ox), c] = x[(2 oy + 1, 2 ox + 1), c]`: `p[0]` the input's rows, columns and channels.
const SUBSAMPLE2X_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let ow = w / 2u;
    if (i >= (h / 2u) * ow * c) { return; }
    let ch = i % c;
    let px = i / c;
    let oy = px / ow;
    let ox = px % ow;
    y[i] = x[((2u * oy + 1u) * w + 2u * ox + 1u) * c + ch];
}
"#;

/// `y[(oy, ox), co] += mean(group co)` of `x`'s space-to-depth: `p[0]` the input's rows, columns, channels in and out,
/// `p[1]` the time slots and the spatial factor (each output pixel's `cin ft fs fs` values in turn by channel, slot, row
/// and column, the slots before the last zero).
const SHUFFLE_DOWN_MEAN_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let cin = p[0].z;
    let cout = p[0].w;
    let ft = p[1].x;
    let fs = p[1].y;
    let ow = w / fs;
    if (i >= (h / fs) * ow * cout) { return; }
    let co = i % cout;
    let px = i / cout;
    let oy = px / ow;
    let ox = px % ow;
    let per = fs * fs;
    let g = cin * ft * per / cout;
    var s = 0.0;
    for (var e = co * g; e < (co + 1u) * g; e++) {
        let t = (e / per) % ft;
        if (t == ft - 1u) {
            let c = e / (ft * per);
            let fy = (e / fs) % fs;
            let fx = e % fs;
            s += x[((oy * fs + fy) * w + ox * fs + fx) * cin + c];
        }
    }
    y[i] += s / f32(g);
}
"#;

/// Space to depth, the time slots one frame's: `p[0]` the input's rows, columns and channels, `p[1]` the time slots and
/// the rows' and columns' factors.
const SPACE_TO_DEPTH_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let w = p[0].y;
    let c = p[0].z;
    let st = p[1].x;
    let sh = p[1].y;
    let sw = p[1].z;
    let co = c * st * sh * sw;
    let ow = w / sw;
    if (i >= (p[0].x / sh) * ow * co) { return; }
    let e = i % co;
    let px = i / co;
    let oy = px / ow;
    let ox = px % ow;
    let ch = e / (st * sh * sw);
    let fy = (e / sw) % sh;
    let fx = e % sw;
    y[i] = x[((oy * sh + fy) * w + ox * sw + fx) * c + ch];
}
"#;

/// `y[r, co] += mean(x[r, co g .. (co + 1) g])`: `p[0]` the rows, the channels in and out.
const GROUP_MEAN_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let cin = p[0].y;
    let cout = p[0].z;
    if (i >= p[0].x * cout) { return; }
    let g = cin / cout;
    let at = (i / cout) * cin + (i % cout) * g;
    var s = 0.0;
    for (var e = 0u; e < g; e++) { s += x[at + e]; }
    y[i] += s / f32(g);
}
"#;

/// `w` (f16 pairs, `p[0].x` words) plus `d` (two f32 a word), rounded to f16.
const ADD_F16: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(6) var<storage, read_write> w: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    w[i] = pack2x16float(unpack2x16float(w[i]) + vec2<f32>(d[2u * i], d[2u * i + 1u]));
}
"#;

const ADD_GATED_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> yv: array<f32>;
@group(0) @binding(1) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    // a clean row (before p[1].x, or from p[1].y on) gated by the second set (p[1].z on)
    let row = i / p[0].x;
    let second = select(0u, p[1].z, row < p[1].x || row >= p[1].y);
    var g = mods[second + p[0].z + i % p[0].x];
    if (p[0].w == 1u) { g = tanh(g); }
    x[i] += yv[i] * g;
}
"#;

/// An image's pixels' rows of `p[0].x` channels f32 (`x`) as f16 padded to `p[0].y` (`q`, two to a word), `p[0].z`
/// pixels, each times `range[0]` ([`F16_RANGE_SET`]'s scale).
const X_F16_PADDED: &str = r#"
@group(0) @binding(0) var<storage, read> range: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = p[0].x;
    let cp2 = p[0].y / 2u;
    // (a pixel's values `p[0].w` apart: its first `c` taken)
    let xs = p[0].w;
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].z * cp2) { return; }
    let px = i / cp2;
    let j = 2u * (i % cp2);
    var v = vec2<f32>(0.0);
    if (j < c) { v.x = x[px * xs + j]; }
    if (j + 1u < c) { v.y = x[px * xs + j + 1u]; }
    q[i] = pack2x16float(v * range[0]);
}
"#;

/// The largest magnitude of `x` (`p[0].x` values) into `range[2]` (its bits: a non-negative f32's order is its bits'),
/// zeroed first ([`F16_RANGE_CLEAR`]).
const F16_RANGE_MAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> range: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { atomicMax(&range[2], bitcast<u32>(abs(x[i]))); }
}
"#;

/// `range[2]` zeroed for [`F16_RANGE_MAX`].
const F16_RANGE_CLEAR: &str = r#"
@group(0) @binding(6) var<storage, read_write> range: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(1)
fn main() {
    range[2] = 0u;
}
"#;

/// A power of two that keeps `x`'s largest (`range[2]`, [`F16_RANGE_MAX`]'s) within 16,384 as f16: `range[0]` it,
/// `range[1]` its inverse (exact both).
const F16_RANGE_SET: &str = r#"
@group(0) @binding(6) var<storage, read_write> range: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(1)
fn main() {
    let m = bitcast<f32>(range[2]);
    var k = 0.0;
    if (m > 16384.0) { k = ceil(log2(m / 16384.0)); }
    k = min(k, 100.0);
    range[0] = bitcast<u32>(exp2(-k));
    range[1] = bitcast<u32>(exp2(k));
}
"#;

/// `x[i] = x[i] * range[1] + b[i % p[0].x]` for `i < p[0].x * p[0].y`: a scaled convolution's sums back to their
/// range, and its bias.
const UNSCALE_BIAS_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> b: array<f32>;
@group(0) @binding(1) var<storage, read> range: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x * p[0].y) { x[i] = x[i] * range[1] + b[i % p[0].x]; }
}
"#;

/// [`ChainRecorder::depth_to_space_rows`]: `y`'s voxel (frames of `p[1].y` rows of `p[1].z`) of `p[0].x` channels
/// from `x`'s. `p[0]`: c, st, sh, sw; `p[1]`: the output's voxels, its rows a frame, its row's columns, the frames
/// dropped first.
const DEPTH_TO_SPACE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = p[0].x;
    let st = p[0].y;
    let sh = p[0].z;
    let sw = p[0].w;
    let oh = p[1].y;
    let ow = p[1].z;
    let i = id.x + id.y * 16776960u;
    if (i >= p[1].x * c) { return; }
    let ch = i % c;
    let v = i / c;
    let ox = v % ow;
    let oy = (v / ow) % oh;
    let ot = v / (ow * oh) + p[1].w;
    let h = oh / sh;
    let w = ow / sw;
    let src = ((ot / st) * h + oy / sh) * w + ox / sw;
    let sc = ((ch * st + ot % st) * sh + oy % sh) * sw + ox % sw;
    y[i] = x[src * (c * st * sh * sw) + sc];
}
"#;

/// `x[i] += b[i % p[0].x]` for `i < p[0].x * p[0].y`.
const ADD_BIAS_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x * p[0].y) { x[i] += b[i % p[0].x]; }
}
"#;

/// Nearest 2x of an image of `p[0].y` by `p[0].z` pixels of `p[0].x` channels: `y`'s pixel `(oy, ox)` is `x`'s
/// `(oy / 2, ox / 2)`.
const UPSAMPLE2X_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = p[0].x;
    let h = p[0].y;
    let w = p[0].z;
    let i = id.x + id.y * 16776960u;
    if (i >= 4u * h * w * c) { return; }
    let ch = i % c;
    let px = i / c;
    let ox = px % (2u * w);
    let oy = px / (2u * w);
    y[i] = x[((oy / 2u) * w + ox / 2u) * c + ch];
}
"#;

/// [`ChainRecorder::shuffle_up_add_rows`]: `y` (`[2h * 2w, cout]`) gets `x` (`[h * w, cin]`) shuffled up. `p[0]`:
/// cin, cout, ft, repeats; `p[1]`: h, w.
const SHUFFLE_UP_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cin = p[0].x;
    let cout = p[0].y;
    let ft = p[0].z;
    let repeats = p[0].w;
    let h = p[1].x;
    let w = p[1].y;
    let i = id.x + id.y * 16776960u;
    if (i >= 4u * h * w * cout) { return; }
    let co = i % cout;
    let px = i / cout;
    let ox = px % (2u * w);
    let oy = px / (2u * w);
    let a = oy % 2u;
    let b = ox % 2u;
    let ci = (4u * co * ft + 4u * (ft - 1u) + 2u * a + b) / repeats;
    y[i] += x[((oy / 2u) * w + ox / 2u) * cin + ci];
}
"#;

/// `y[i] = gelu(x[i])` for `i < p[0].x`, the tanh approximation (as [`GELU_MUL_SPLIT`]'s).
/// [`GELU`] exactly: `x (1 + erf(x / sqrt 2)) / 2`, erf as Abramowitz and Stegun's 7.1.26 (within 1.5e-7).
const GELU_ERF: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) {
        let g = x[i];
        let z = abs(g) * 0.7071067811865476;
        let t = 1.0 / (1.0 + 0.3275911 * z);
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erf = sign(g) * (1.0 - poly * exp(-z * z));
        y[i] = 0.5 * g * (1.0 + erf);
    }
}
"#;

const GELU: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) {
        let g = x[i];
        let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
        y[i] = 0.5 * g * (1.0 + tanh(inner));
    }
}
"#;

/// `y[i] = w[i] * sigmoid(x[i])` for `i < p[0].x` (`w` the values, `x` the gate), as the CPU's mul_sigmoid.
const MUL_SIGMOID: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[i] = bitcast<f32>(w[i]) * (1.0 / (1.0 + exp(-x[i]))); }
}
"#;

/// `y[i] = silu(w[i]) * x[i]` for `i < p[0].x` (`w` the gate, `x` the up projection).
const SILU_MUL: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) {
        let g = bitcast<f32>(w[i]);
        y[i] = (g / (1.0 + exp(-g))) * x[i];
    }
}
"#;

/// `y[r * n + o] = dot(w[o], x[r])` for f32 weights `w` (`[n, k]`, `p[0]`: `n`, `k`), a workgroup an (output, row).
const MATMUL_F32: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let o = wg.x;
    let r = wg.y;
    var s = 0.0;
    for (var i = li; i < k; i += 256u) { s += bitcast<f32>(w[o * k + i]) * x[r * k + i]; }
    part[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { y[r * n + o] = part[0]; }
}
"#;

/// An n-gram layer's gate (`Backend::ple_gate`), a workgroup a (row, stream): the stream's key and query RMS-normed and
/// scaled by their norms, their dot over `sqrt(d)`, its signed square root's sigmoid the gate; `gated` the gate times
/// the row's value, `conv_in` that RMS-normed, scaled by the conv's norm and rounded to f16. `p[0]`: d, streams, the
/// bits of eps.
const PLE_GATE: &str = r#"
@group(0) @binding(0) var<storage, read> key: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> nk: array<f32>;
@group(0) @binding(4) var<storage, read> nq: array<f32>;
@group(0) @binding(5) var<storage, read> nc: array<f32>;
@group(0) @binding(6) var<storage, read_write> gated: array<f32>;
@group(0) @binding(7) var<storage, read_write> conv_in: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> red: array<vec2<f32>, 256>;

fn total(v: vec2<f32>, t: u32) -> vec2<f32> {
    red[t] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (t < st) { red[t] += red[t + st]; }
        workgroupBarrier();
    }
    let out = red[0];
    workgroupBarrier();
    return out;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let d = p[0].x;
    let streams = p[0].y;
    let eps = bitcast<f32>(p[0].z);
    let r = wg.x / streams;
    let s = wg.x % streams;
    let o = r * streams * d + s * d;
    var sq = vec2<f32>(0.0);
    for (var j = t; j < d; j += 256u) {
        let kk = key[o + j];
        let qq = x[o + j];
        sq += vec2<f32>(kk * kk, qq * qq);
    }
    let ss = total(sq, t);
    let ik = 1.0 / sqrt(ss.x / f32(d) + eps);
    let iq = 1.0 / sqrt(ss.y / f32(d) + eps);
    var dot = 0.0;
    for (var j = t; j < d; j += 256u) {
        dot += (key[o + j] * ik * nk[s * d + j]) * (x[o + j] * iq * nq[s * d + j]);
    }
    let g = total(vec2<f32>(dot, 0.0), t).x / sqrt(f32(d));
    var sg = 0.0;
    if (g != 0.0) {
        sg = sign(g) * sqrt(max(abs(g), 1e-6));
    }
    let gate = 1.0 / (1.0 + exp(-sg));
    var gs = 0.0;
    for (var j = t; j < d; j += 256u) {
        let gv = gate * value[r * d + j];
        gated[o + j] = gv;
        gs += gv * gv;
    }
    let inv = 1.0 / sqrt(total(vec2<f32>(gs, 0.0), t).x / f32(d) + eps);
    for (var j = t; j < d; j += 256u) {
        conv_in[o + j] = half(gate * value[r * d + j] * inv * nc[s * d + j]);
    }
}
"#;

/// An n-gram layer's dilated causal conv (`Backend::ple_conv`), a thread a channel through the rows: over the stream
/// of the window's rows then `conv_in`'s, `x[r, c] += gated[r, c] + silu(sum over j of w[c, j] stream[r + j dilation,
/// c])`; the window then the stream's last rows. `p[0]`: width, rows, kernel, dilation.
const PLE_CONV: &str = r#"
@group(0) @binding(0) var<storage, read> gated: array<f32>;
@group(0) @binding(1) var<storage, read> conv_in: array<f32>;
@group(0) @binding(2) var<storage, read> wc: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(7) var<storage, read_write> win: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn stream_at(q: u32, c: u32, width: u32, state: u32) -> f32 {
    if (q < state) {
        return win[q * width + c];
    }
    return conv_in[(q - state) * width + c];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = p[0].x;
    let rows = p[0].y;
    let kernel = p[0].z;
    let dil = p[0].w;
    let c = id.x;
    if (c >= width) {
        return;
    }
    let state = (kernel - 1u) * dil;
    for (var r = 0u; r < rows; r++) {
        var acc = 0.0;
        for (var j = 0u; j < kernel; j++) {
            acc += wc[c * kernel + j] * stream_at(r + j * dil, c, width, state);
        }
        x[r * width + c] = x[r * width + c] + (gated[r * width + c] + acc / (1.0 + exp(-acc)));
    }
    // each place read before it is written: the new row i is the stream's row rows + i, later than i
    for (var i = 0u; i < state; i++) {
        win[i * width + c] = stream_at(rows + i, c, width, state);
    }
}
"#;

/// [`MATVEC_F32_4`] with f16 weights two to a word (`DeviceChain::vec_f16`'s), `k` a multiple of 4: each output's
/// products summed as it sums them. `p[0]`: n, k.
const MATVEC_F16: &str = r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k4 = p[0].y / 4u;
    let o = wg.x;
    var s = vec4<f32>(0.0);
    for (var i = li; i < k4; i += 256u) {
        let pr = w2[o * k4 + i];
        let a = unpack2x16float(pr.x);
        let b = unpack2x16float(pr.y);
        s += vec4<f32>(a.x, a.y, b.x, b.y) * x4[i];
    }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { y[o] = part[0]; }
}
"#;

/// [`MATVEC_F16`] (`narrow`: [`MATVEC_F16_NARROW`]) for `rows` rows (2 to 8, a check of drafts): each weight read once
/// for all of them, each row's products summed in the one-row kernel's order (so a row's outputs are its bit for bit).
/// `p[0]`: n, k.
fn matvec_f16_rows(rows: usize, narrow: bool) -> String {
    assert!((2..=8).contains(&rows), "a few rows (2 to 8)");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    if narrow {
        let regs = each(&|r| format!("    var s{r} = 0.0;"));
        let sums = each(&|r| format!("            s{r} = s{r} + pr.x * x[{r}u * k + 2u * i];\n            s{r} = s{r} + pr.y * x[{r}u * k + 2u * i + 1u];"));
        let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r};"));
        let outs = each(&|r| format!("        var t{r} = 0.0;\n        for (var j = 0u; j < 8u; j++) {{ t{r} += part[{r}u * 256u + li + j]; }}\n        y[{r}u * n + o] = t{r};"));
        return format!(
            r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let o = wg.x * 32u + li / 8u;
    let q = li % 8u;
    let pairs = k / 2u;
{regs}
    if (o < n) {{
        for (var i = q; i < pairs; i += 8u) {{
            let pr = unpack2x16float(w[o * pairs + i]);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    if (q == 0u && o < n) {{
{outs}
    }}
}}
"#,
            len = rows * 256
        );
    }
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("        s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    let adds = each(&|r| format!("            part[{r}u * 256u + li] += part[{r}u * 256u + li + st];"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let o = wg.x;
{regs}
    for (var i = li; i < k4; i += 256u) {{
        let pr = w2[o * k4 + i];
        let a = unpack2x16float(pr.x);
        let b = unpack2x16float(pr.y);
        let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
    }}
{parts}
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {{
        if (li < st) {{
{adds}
        }}
        workgroupBarrier();
    }}
    if (li < {rows}u) {{ y[li * n + o] = part[li * 256u]; }}
}}
"#,
        len = rows * 256
    )
}

/// [`MATVEC_F16`] for a short row (`k` under 2048): eight threads an output (a word, two weights, at a time), 32 outputs
/// a workgroup. `p[0]`: n, k.
const MATVEC_F16_NARROW: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let o = wg.x * 32u + li / 8u;
    let q = li % 8u;
    let pairs = k / 2u;
    var s = 0.0;
    if (o < n) {
        for (var i = q; i < pairs; i += 8u) {
            let pr = unpack2x16float(w[o * pairs + i]);
            s = s + pr.x * x[2u * i];
            s = s + pr.y * x[2u * i + 1u];
        }
    }
    part[li] = s;
    workgroupBarrier();
    if (q == 0u && o < n) {
        var t = 0.0;
        for (var j = 0u; j < 8u; j++) { t += part[li + j]; }
        y[o] = t;
    }
}
"#;

/// [`MATMUL_F32`] of one row with `k` a multiple of 4: vec4 loads, a thread's products in four running sums (its
/// loads in flight together). `p[0]`: n, k.
const MATVEC_F32_4: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k4 = p[0].y / 4u;
    let o = wg.x;
    var s = vec4<f32>(0.0);
    for (var i = li; i < k4; i += 256u) { s += w4[o * k4 + i] * x4[i]; }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { y[o] = part[0]; }
}
"#;

/// [`MATVEC_F32_4`] for 2 to 8 rows (a check of drafts): a workgroup an output, each weight read once for every row,
/// a row's sums in the workgroup's memory a slot each. `p[0]`: n, k, rows.
const MATVEC_F32_4_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let rows = p[0].z;
    let o = wg.x;
    var s: array<vec4<f32>, 8>;
    for (var r = 0u; r < 8u; r++) { s[r] = vec4<f32>(0.0); }
    for (var i = li; i < k4; i += 256u) {
        let wv = w4[o * k4 + i];
        for (var r = 0u; r < rows; r++) { s[r] += wv * x4[r * k4 + i]; }
    }
    for (var r = 0u; r < rows; r++) { part[r * 256u + li] = s[r].x + s[r].y + s[r].z + s[r].w; }
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) {
            for (var r = 0u; r < rows; r++) { part[r * 256u + li] += part[r * 256u + li + st]; }
        }
        workgroupBarrier();
    }
    if (li < rows) { y[li * n + o] = part[li * 256u]; }
}
"#;

/// `y[r, o] = sum over i of x[r, i] w[o, i]` for a prompt's rows ([`MATMUL_F32`]'s, of several): a workgroup a 64x64
/// tile of `y` (64 rows by 64 outputs), a thread 4x4 of it, `k` 16 at a time through the workgroup's memory; split
/// over `k` (the grid's third axis, `p[0].w` of it each), split `s`'s sums to part `s` of `y` (`[splits, rows, n]`)
/// for [`SUM_SPLITS`] to add up. `p[0]`: n, k, rows, k a split.
const MATMUL_F32_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// [16 of k][64 rows (or outputs)], a row of 17 vec4s (the 17th padding, against bank conflicts)
var<workgroup> xs: array<vec4<f32>, 272>;
var<workgroup> ws: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let rows = p[0].z;
    let kc = p[0].w;
    let o0 = wg.x * 64u;
    let r0 = wg.y * 64u;
    let k0 = wg.z * kc;
    let k1 = min(k, k0 + kc);
    let tr = t / 16u;
    let to = t % 16u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var kb = k0; kb < k1; kb += 16u) {
        for (var q = 0u; q < 4u; q++) {
            let idx = t + q * 256u;
            let rr = idx / 16u;
            let kk = idx % 16u;
            let gk = kb + kk;
            var v = 0.0;
            if (r0 + rr < rows && gk < k1) {
                v = x[(r0 + rr) * k + gk];
            }
            xs[kk * 17u + rr / 4u][rr % 4u] = v;
            var u = 0.0;
            if (o0 + rr < n && gk < k1) {
                u = w[(o0 + rr) * k + gk];
            }
            ws[kk * 17u + rr / 4u][rr % 4u] = u;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let xv = xs[kk * 17u + tr];
            let wv = ws[kk * 17u + to];
            a0 += xv.x * wv;
            a1 += xv.y * wv;
            a2 += xv.z * wv;
            a3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let base = wg.z * rows * n;
    let o = o0 + to * 4u;
    let r = r0 + tr * 4u;
    let acc = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i = 0u; i < 4u; i++) {
        if (r + i < rows) {
            for (var j = 0u; j < 4u; j++) {
                if (o + j < n) {
                    y[base + (r + i) * n + o + j] = acc[i][j];
                }
            }
        }
    }
}
"#;

/// A convolution without tensor cores ([`ChainRecorder::conv_rows`], [`ChainRecorder::conv3d_rows`]): the f32 tiled
/// matmul's 64 voxels by 64 outputs a workgroup ([`MATMUL_F32_TILED`]), its tokens' tile gathered through the taps
/// as the tensor cores' kernel takes them (a 3x3's pixel `(y + dy - 1, x + dx - 1)`, zeros past the frame's edge; a
/// 3x3x3's from frame `t + dt - 1` clamped to the clip), the weights the same packed f16 (`[cout][taps][cin padded
/// to 32]`, two to a word), the bias added. `p[0]`: `cout`, `cin`, the voxels, the values a voxel of `x` apart (its
/// first `cin`); `p[1]`: the taps, a row's pixels, a frame's rows, the dispatch's first tile of voxels.
const CONV_F32_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// [16 of k][64 voxels (or outputs)], a row of 17 vec4s (the 17th padding, against bank conflicts)
var<workgroup> xs: array<vec4<f32>, 272>;
var<workgroup> ws: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let cin = p[0].y;
    let m = p[0].z;
    let stride = p[0].w;
    let cp = (cin + 31u) / 32u * 32u;
    let taps = p[1].x;
    let wd = p[1].y;
    let h = p[1].z;
    let plane = h * wd;
    let frames = m / plane;
    let kt = taps * cp;
    let o0 = wg.x * 64u;
    let r0 = (p[1].w + wg.y + wg.z * 65535u) * 64u;
    let tr = t / 16u;
    let to = t % 16u;
    let kk = t % 16u;
    // the voxels this thread loads (rows `t / 16 + 16 q` of the tile): their frame, row and column
    let pa = min(r0 + tr, m - 1u);
    let pb = min(r0 + tr + 16u, m - 1u);
    let pc = min(r0 + tr + 32u, m - 1u);
    let pd = min(r0 + tr + 48u, m - 1u);
    let fa = i32(pa / plane); let ya = i32((pa % plane) / wd); let xa = i32(pa % wd);
    let fb = i32(pb / plane); let yb = i32((pb % plane) / wd); let xb = i32(pb % wd);
    let fc = i32(pc / plane); let yc = i32((pc % plane) / wd); let xc = i32(pc % wd);
    let fd = i32(pd / plane); let yd = i32((pd % plane) / wd); let xd = i32(pd % wd);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var kb = 0u; kb < kt; kb += 16u) {
        let gk = kb + kk;
        let tap = gk / cp;
        let c = gk % cp;
        let live = gk < kt && c < cin;
        // the tap's offsets in time, rows and columns
        var dt = 0;
        var dy = 0;
        var dx = 0;
        if (taps == 27u) {
            dt = i32(tap / 9u) - 1;
            dy = i32((tap / 3u) % 3u) - 1;
            dx = i32(tap % 3u) - 1;
        } else if (taps == 49u) {
            dy = i32(tap / 7u) - 3;
            dx = i32(tap % 7u) - 3;
        } else if (taps == 9u) {
            dy = i32(tap / 3u) - 1;
            dx = i32(tap % 3u) - 1;
        }
        for (var q = 0u; q < 4u; q++) {
            var f = fa;
            var yy = ya;
            var xx = xa;
            var r = r0 + tr;
            if (q == 1u) { f = fb; yy = yb; xx = xb; r = r0 + tr + 16u; }
            if (q == 2u) { f = fc; yy = yc; xx = xc; r = r0 + tr + 32u; }
            if (q == 3u) { f = fd; yy = yd; xx = xd; r = r0 + tr + 48u; }
            let sy = yy + dy;
            let sx = xx + dx;
            let sf = clamp(f + dt, 0, i32(frames) - 1);
            var v = 0.0;
            if (live && r < m && sy >= 0 && sy < i32(h) && sx >= 0 && sx < i32(wd)) {
                v = x[((u32(sf) * h + u32(sy)) * wd + u32(sx)) * stride + c];
            }
            let rr = tr + 16u * q;
            xs[kk * 17u + rr / 4u][rr % 4u] = v;
            var u = 0.0;
            if (o0 + rr < n && gk < kt) {
                let e = (o0 + rr) * kt + gk;
                u = unpack2x16float(w[e / 2u])[e % 2u];
            }
            ws[kk * 17u + rr / 4u][rr % 4u] = u;
        }
        workgroupBarrier();
        for (var j = 0u; j < 16u; j++) {
            let xv = xs[j * 17u + tr];
            let wv = ws[j * 17u + to];
            a0 += xv.x * wv;
            a1 += xv.y * wv;
            a2 += xv.z * wv;
            a3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let o = o0 + to * 4u;
    let r = r0 + tr * 4u;
    let acc = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i = 0u; i < 4u; i++) {
        if (r + i < m) {
            for (var j = 0u; j < 4u; j++) {
                if (o + j < n) {
                    y[(r + i) * n + o + j] = acc[i][j] + bias[o + j];
                }
            }
        }
    }
}
"#;

/// ComfyUI's W4A8 decoded to f16 ([`ChainRecorder::w4a8_f16`]), a thread a code byte (two values, an f16 pair's
/// word). `p[0]`: the rows, the columns.
const W4A8_F16: &str = r#"
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> rel: array<u32>;
@group(0) @binding(2) var<storage, read> channel: array<f32>;
@group(0) @binding(3) var<storage, read> book: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let v = select(bitcast<f32>(((e + 120u) << 23u) | (m << 20u)), f32(m) * 0.001953125, e == 0u);
    return select(v, -v, (b & 128u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let cols = p[0].y;
    let bytes = cols / 2u;
    if (i >= p[0].x * bytes) { return; }
    let row = i / bytes;
    let col = 2u * (i % bytes);
    let b = (codes[i / 4u] >> (8u * (i % 4u))) & 255u;
    let gi = row * (cols / 16u) + col / 16u;
    let s = e4m3((rel[gi / 4u] >> (8u * (gi % 4u))) & 255u);
    let ch = channel[row];
    let lo = clamp(round(book[b & 15u] * s), -127.0, 127.0) * ch;
    let hi = clamp(round(book[b >> 4u] * s), -127.0, 127.0) * ch;
    out[i] = pack2x16float(vec2<f32>(lo, hi));
}
"#;

/// [`W4A8_F16`] with ConvRot's rotation undone (`GS`, a power of four, put in): a workgroup a row's group, decoded
/// into its memory, then the Hadamard matrix's passes of four (a digit of the group's base-4 index each), scaled by
/// the group's square root. `p[0]`: the rows, the columns; a row `wg.y + 65,535 wg.z`, its group `wg.x`.
const W4A8_F16_ROTATED: &str = r#"
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> rel: array<u32>;
@group(0) @binding(2) var<storage, read> channel: array<f32>;
@group(0) @binding(3) var<storage, read> book: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const GS: u32 = GS_u;
var<workgroup> g: array<f32, GS_u>;

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let v = select(bitcast<f32>(((e + 120u) << 23u) | (m << 20u)), f32(m) * 0.001953125, e == 0u);
    return select(v, -v, (b & 128u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cols = p[0].y;
    let row = min(wg.y + wg.z * 65535u, p[0].x - 1u);
    let g0 = wg.x * GS;
    let ch = channel[row];
    for (var cb = t; cb < GS / 2u; cb += 64u) {
        let i = row * (cols / 2u) + g0 / 2u + cb;
        let b = (codes[i / 4u] >> (8u * (i % 4u))) & 255u;
        let gi = row * (cols / 16u) + (g0 + 2u * cb) / 16u;
        let s = e4m3((rel[gi / 4u] >> (8u * (gi % 4u))) & 255u);
        g[2u * cb] = clamp(round(book[b & 15u] * s), -127.0, 127.0) * ch;
        g[2u * cb + 1u] = clamp(round(book[b >> 4u] * s), -127.0, 127.0) * ch;
    }
    workgroupBarrier();
    for (var h = 1u; h < GS; h *= 4u) {
        for (var u = t; u < GS / 4u; u += 64u) {
            let j = (u / h) * 4u * h + u % h;
            let a = g[j];
            let b = g[j + h];
            let c = g[j + 2u * h];
            let d = g[j + 3u * h];
            g[j] = a + b + c - d;
            g[j + h] = a + b - c + d;
            g[j + 2u * h] = a - b + c + d;
            g[j + 3u * h] = -a + b + c + d;
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(f32(GS));
    for (var cb = t; cb < GS / 2u; cb += 64u) {
        out[row * (cols / 2u) + g0 / 2u + cb] = pack2x16float(vec2<f32>(g[2u * cb], g[2u * cb + 1u]) * scale);
    }
}
"#;

/// Swin's windows ([`ChainRecorder::window_rows`]): a thread an output value. `p[0]`: `h`, `w`, `c`, `win`; `p[1]`: the
/// shift, the padded grid's `hp` and `wp`.
const WINDOW_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let w = p[0].y;
    let c = p[0].z;
    let win = p[0].w;
    let shift = p[1].x;
    let hp = p[1].y;
    let wp = p[1].z;
    if (i >= hp * wp * c) { return; }
    let ch = i % c;
    let tok = i / c;
    let n = win * win;
    let wdx = tok / n;
    let t = tok % n;
    let gw = wp / win;
    let py = ((wdx / gw) * win + t / win + shift) % hp;
    let px = ((wdx % gw) * win + t % win + shift) % wp;
    var v = 0.0;
    if (py < p[0].x && px < w) { v = x[(py * w + px) * c + ch]; }
    out[i] = v;
}
"#;

/// [`WINDOW_ROWS`]' way back, added ([`ChainRecorder::unwindow_add_rows`]): a thread a token's value. `p` as its.
const UNWINDOW_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> xw: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let w = p[0].y;
    let c = p[0].z;
    let win = p[0].w;
    let shift = p[1].x;
    let hp = p[1].y;
    let wp = p[1].z;
    if (i >= p[0].x * w * c) { return; }
    let ch = i % c;
    let tok = i / c;
    let ry = (tok / w + hp - shift) % hp;
    let rx = (tok % w + wp - shift) % wp;
    let wdx = (ry / win) * (wp / win) + rx / win;
    let t = (ry % win) * win + rx % win;
    out[i] += xw[(wdx * win * win + t) * c + ch];
}
"#;

/// Swin's window attention, heads 32 wide ([`ChainRecorder::window_attention`]): a workgroup a (window, head), a thread
/// a query (its 32 dims and sums in registers), the keys in turn (each a broadcast: every thread reads the same), the
/// softmax online. `p[0]`: the padded grid's `hp` and `wp`, the heads, `win`; `p[1]`: the shift, the scale's bits.
const WINDOW_ATTENTION: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// a coordinate's region of the rolled grid `n` long (Swin's slices: up to the last window, to the shift, the rest)
fn region(i: u32, n: u32, win: u32, shift: u32) -> u32 {
    if (i < n - win) { return 0u; }
    if (i < n - shift) { return 1u; }
    return 2u;
}

@compute @workgroup_size(N_u)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let hp = p[0].x;
    let wp = p[0].y;
    let heads = p[0].z;
    let win = p[0].w;
    let shift = p[1].x;
    let scale = bitcast<f32>(p[1].y);
    let wdx = wg.x;
    let head = wg.y;
    let n = win * win;
    let c4 = heads * 8u;
    let row4 = 3u * c4;
    let base = wdx * n;
    let qo = (base + t) * row4 + head * 8u;
    let q0 = qkv[qo] * scale;
    let q1 = qkv[qo + 1u] * scale;
    let q2 = qkv[qo + 2u] * scale;
    let q3 = qkv[qo + 3u] * scale;
    let q4 = qkv[qo + 4u] * scale;
    let q5 = qkv[qo + 5u] * scale;
    let q6 = qkv[qo + 6u] * scale;
    let q7 = qkv[qo + 7u] * scale;
    let gw = wp / win;
    let qy = t / win;
    let qx = t % win;
    let gy = (wdx / gw) * win;
    let gx = (wdx % gw) * win;
    var qr = 0u;
    if (shift > 0u) { qr = region(gy + qy, hp, win, shift) * 3u + region(gx + qx, wp, win, shift); }
    var m = -3.0e38;
    var l = 0.0;
    var a0 = vec4<f32>();
    var a1 = vec4<f32>();
    var a2 = vec4<f32>();
    var a3 = vec4<f32>();
    var a4 = vec4<f32>();
    var a5 = vec4<f32>();
    var a6 = vec4<f32>();
    var a7 = vec4<f32>();
    for (var j = 0u; j < n; j++) {
        let ko = (base + j) * row4 + c4 + head * 8u;
        var s = dot(q0, qkv[ko]) + dot(q1, qkv[ko + 1u]) + dot(q2, qkv[ko + 2u]) + dot(q3, qkv[ko + 3u]) + dot(q4, qkv[ko + 4u]) + dot(q5, qkv[ko + 5u]) + dot(q6, qkv[ko + 6u]) + dot(q7, qkv[ko + 7u]);
        let ky = j / win;
        let kx = j % win;
        s += table[((qy + win - 1u - ky) * (2u * win - 1u) + qx + win - 1u - kx) * heads + head];
        if (shift > 0u && region(gy + ky, hp, win, shift) * 3u + region(gx + kx, wp, win, shift) != qr) { s -= 100.0; }
        let mn = max(m, s);
        let a = exp(m - mn);
        let e = exp(s - mn);
        l = l * a + e;
        m = mn;
        let vo = ko + c4;
        a0 = a0 * a + e * qkv[vo];
        a1 = a1 * a + e * qkv[vo + 1u];
        a2 = a2 * a + e * qkv[vo + 2u];
        a3 = a3 * a + e * qkv[vo + 3u];
        a4 = a4 * a + e * qkv[vo + 4u];
        a5 = a5 * a + e * qkv[vo + 5u];
        a6 = a6 * a + e * qkv[vo + 6u];
        a7 = a7 * a + e * qkv[vo + 7u];
    }
    let inv = 1.0 / l;
    let oo = (base + t) * c4 + head * 8u;
    out[oo] = a0 * inv;
    out[oo + 1u] = a1 * inv;
    out[oo + 2u] = a2 * inv;
    out[oo + 3u] = a3 * inv;
    out[oo + 4u] = a4 * inv;
    out[oo + 5u] = a5 * inv;
    out[oo + 6u] = a6 * inv;
    out[oo + 7u] = a7 * inv;
}
"#;

/// A bilinear resize with the corners aligned ([`ChainRecorder::resize_bilinear_rows`]): a thread an output value.
/// `p[0]`: `h`, `w`, `c`, `oh`; `p[1]`: `ow`.
const RESIZE_BILINEAR_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let oh = p[0].w;
    let ow = p[1].x;
    if (i >= oh * ow * c) { return; }
    let ch = i % c;
    let ox = (i / c) % ow;
    let oy = i / (c * ow);
    var sy = 0.0;
    if (oh > 1u) { sy = f32(h - 1u) / f32(oh - 1u) * f32(oy); }
    var sx = 0.0;
    if (ow > 1u) { sx = f32(w - 1u) / f32(ow - 1u) * f32(ox); }
    let y0 = min(u32(sy), h - 1u);
    let x0 = min(u32(sx), w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let x1 = min(x0 + 1u, w - 1u);
    let fy = sy - f32(y0);
    let fx = sx - f32(x0);
    let top = (1.0 - fx) * x[(y0 * w + x0) * c + ch] + fx * x[(y0 * w + x1) * c + ch];
    let bottom = (1.0 - fx) * x[(y1 * w + x0) * c + ch] + fx * x[(y1 * w + x1) * c + ch];
    out[i] = (1.0 - fy) * top + fy * bottom;
}
"#;

/// A picture as patches ([`ChainRecorder::blocks_to_channels_rows`]): a thread an output value. `p[0]`: the picture's
/// side, `c`, the patches' side.
const BLOCKS_TO_CHANNELS_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let s = p[0].x;
    let c = p[0].y;
    let size = p[0].z;
    let g = s / size;
    let cc = c * g * g;
    if (i >= size * size * cc) { return; }
    let oc = i % cc;
    let pix = i / cc;
    let ch = oc / (g * g);
    let hg = (oc / g) % g;
    let wg = oc % g;
    out[i] = x[((hg * size + pix / size) * s + wg * size + pix % size) * c + ch];
}
"#;

/// A modulated deformable convolution's taps ([`ChainRecorder::deform_im2col_rows`]): a thread an output value.
/// `p[0]`: `h`, `w`, `c`, `k`; `p[1]`: the first pixel, the pixels.
const DEFORM_IM2COL_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<f32>;
@group(0) @binding(2) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let k = p[0].w;
    let kk = k * k;
    if (i >= p[1].y * kk * c) { return; }
    let ch = i % c;
    let t = (i / c) % kk;
    let pix = p[1].x + i / (c * kk);
    let pad = f32(k / 2u);
    let y = f32(pix / w) - pad + f32(t / k) + offsets[pix * 2u * kk + 2u * t];
    let xx = f32(pix % w) - pad + f32(t % k) + offsets[pix * 2u * kk + 2u * t + 1u];
    var v = 0.0;
    if (y > -1.0 && y < f32(h) && xx > -1.0 && xx < f32(w)) {
        let y0 = floor(y);
        let x0 = floor(xx);
        let ly = y - y0;
        let lx = xx - x0;
        let iy = i32(y0);
        let ix = i32(x0);
        let hi = i32(h);
        let wi = i32(w);
        if (iy >= 0 && ix >= 0) { v += (1.0 - ly) * (1.0 - lx) * x[(u32(iy) * w + u32(ix)) * c + ch]; }
        if (iy >= 0 && ix + 1 < wi) { v += (1.0 - ly) * lx * x[(u32(iy) * w + u32(ix + 1)) * c + ch]; }
        if (iy + 1 < hi && ix >= 0) { v += ly * (1.0 - lx) * x[(u32(iy + 1) * w + u32(ix)) * c + ch]; }
        if (iy + 1 < hi && ix + 1 < wi) { v += ly * lx * x[(u32(iy + 1) * w + u32(ix + 1)) * c + ch]; }
        v *= mods[pix * kk + t];
    }
    out[i] = v;
}
"#;

/// A 1-D convolution ([`ChainRecorder::conv1d_padded_rows`]): [`CONV_F32_TILED`]'s tiles over steps (64 steps by 64
/// outputs a workgroup), tap `t` of step `s` from step `s + t dilation - pad`. `p[0]`: `cout`, `cin`, the steps, the
/// taps; `p[1]`: the dilation, the dispatch's first tile of steps, `pad`.
const CONV1D_F32_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> xs: array<vec4<f32>, 272>;
var<workgroup> ws: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let cin = p[0].y;
    let m = p[0].z;
    let taps = p[0].w;
    let dil = i32(p[1].x);
    let cp = (cin + 31u) / 32u * 32u;
    let kt = taps * cp;
    let pad = i32(p[1].z);
    let o0 = wg.x * 64u;
    let r0 = (p[1].y + wg.y) * 64u;
    let tr = t / 16u;
    let to = t % 16u;
    let kk = t % 16u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var kb = 0u; kb < kt; kb += 16u) {
        let gk = kb + kk;
        let tap = i32(gk / cp);
        let c = gk % cp;
        let live = gk < kt && c < cin;
        for (var q = 0u; q < 4u; q++) {
            let rr = tr + 16u * q;
            let s = i32(r0 + rr) + tap * dil - pad;
            var v = 0.0;
            if (live && r0 + rr < m && s >= 0 && s < i32(m)) { v = x[u32(s) * cin + c]; }
            xs[kk * 17u + rr / 4u][rr % 4u] = v;
            var u = 0.0;
            if (o0 + rr < n && gk < kt) {
                let e = (o0 + rr) * kt + gk;
                u = unpack2x16float(w[e / 2u])[e % 2u];
            }
            ws[kk * 17u + rr / 4u][rr % 4u] = u;
        }
        workgroupBarrier();
        for (var j = 0u; j < 16u; j++) {
            let xv = xs[j * 17u + tr];
            let wv = ws[j * 17u + to];
            a0 += xv.x * wv;
            a1 += xv.y * wv;
            a2 += xv.z * wv;
            a3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let o = o0 + to * 4u;
    let r = r0 + tr * 4u;
    let acc = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i = 0u; i < 4u; i++) {
        if (r + i < m) {
            for (var j = 0u; j < 4u; j++) {
                if (o + j < n) {
                    y[(r + i) * n + o + j] = acc[i][j] + bias[o + j];
                }
            }
        }
    }
}
"#;

/// A transposed 1-D convolution ([`ChainRecorder::conv_transpose1d_rows`]): a thread an output value, over the taps
/// that reach it (`(o + pad - j)` a multiple of the stride: `k / stride` of them) and every input channel (vec4s).
/// `p[0]`: `cout`, `cin`, the input's steps, `k`; `p[1]`: the stride, `pad`, the output's steps.
const CONV_TRANSPOSE1D_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let cout = p[0].x;
    let cin4 = p[0].y / 4u;
    let len = p[0].z;
    let k = p[0].w;
    let stride = p[1].x;
    let pad = p[1].y;
    if (i >= p[1].z * cout) { return; }
    let co = i % cout;
    let o = i / cout;
    var acc = 0.0;
    // the first tap that reaches `o`: `j = (o + pad) mod stride`, then every stride
    for (var j = (o + pad) % stride; j < k; j += stride) {
        if (o + pad < j) { break; }
        let src = (o + pad - j) / stride;
        if (src >= len) { continue; }
        let wb = (co * k + j) * cin4;
        let xb = src * cin4;
        var s = vec4<f32>(0.0);
        for (var c = 0u; c < cin4; c++) { s += w4[wb + c] * x4[xb + c]; }
        acc += s.x + s.y + s.z + s.w;
    }
    y[i] = acc + bias[co];
}
"#;

/// Snake in place ([`ChainRecorder::snake_rows`]): a thread a value. `p[0]`: the rows, `c`.
const SNAKE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> alpha: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    let a = alpha[i % p[0].y];
    let v = x[i];
    let s = sin(a * v);
    x[i] = v + s * s / (a + 1e-9);
}
"#;

/// Rows gathered ([`ChainRecorder::gather_rows`]): a thread a value, zeros for a missing row. `p[0]`: the rows, `c`,
/// the source's rows, the index's first.
const GATHER_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].y;
    if (i >= p[0].x * c) { return; }
    let src = index[p[0].w + i / c];
    var v = 0.0;
    if (src < p[0].z) { v = x[src * c + i % c]; }
    out[i] = v;
}
"#;

/// Each channel repeated, added ([`ChainRecorder::repeat_cols_add_rows`]): a thread an output value. `p[0]`: the rows,
/// `c`, the repeats.
const REPEAT_COLS_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let width = p[0].y * p[0].z;
    if (i >= p[0].x * width) { return; }
    let r = i / width;
    out[i] += x[r * p[0].y + (i % width) / p[0].z];
}
"#;

/// A depthwise causal 1-D convolution ([`ChainRecorder::depthwise_causal_conv1d_rows`]): a thread an output value.
/// `p[0]`: `c`, `k`, the steps.
const DEPTHWISE_CAUSAL_CONV1D_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].x;
    let k = p[0].y;
    if (i >= p[0].z * c) { return; }
    let ch = i % c;
    let s = i32(i / c);
    var acc = bias[ch];
    for (var j = 0u; j < k; j++) {
        let src = s + i32(j) - i32(k) + 1;
        if (src >= 0) { acc += w[ch * k + j] * x[u32(src) * c + ch]; }
    }
    y[i] = acc;
}
"#;

/// SnakeBeta in place ([`ChainRecorder::snake_beta_rows`]): a thread a value. `p[0]`: the rows, `c`.
const SNAKE_BETA_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> freq: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    let ch = i % p[0].y;
    let v = x[i];
    let s = sin(freq[ch] * v);
    x[i] = v + scale[ch] * s * s;
}
"#;

/// A clamp in place ([`ChainRecorder::clamp_in_place`]): a thread a value. `p[0]`: the values, the bounds' bits.
const CLAMP_IN_PLACE: &str = r#"
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    x[i] = clamp(x[i], bitcast<f32>(p[0].y), bitcast<f32>(p[0].z));
}
"#;

/// `tanh` in place ([`ChainRecorder::tanh_in_place`]): a thread a value. `p[0]`: the values.
const TANH_IN_PLACE: &str = r#"
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    x[i] = tanh(x[i]);
}
"#;

/// Rows times their gates' sigmoids ([`ChainRecorder::mul_sigmoid_rows`]), in place: a thread a value. `p[0]`: the
/// rows, `c`.
const MUL_SIGMOID_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    x[i] = x[i] / (1.0 + exp(-gate[i / p[0].y]));
}
"#;

/// A column's mean ([`ChainRecorder::mean_rows`]): a thread a channel. `p[0]`: the rows, `c`.
const MEAN_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = id.x;
    let rows = p[0].x;
    let c = p[0].y;
    if (ch >= c) { return; }
    var s = 0.0;
    for (var r = 0u; r < rows; r++) { s += x[r * c + ch]; }
    out[ch] = s / f32(rows);
}
"#;

/// One row into many ([`ChainRecorder::broadcast_rows`]): a thread a value. `p[0]`: the rows, `c`, the stride, `at`.
const BROADCAST_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].y;
    if (i >= p[0].x * c) { return; }
    dst[(i / c) * p[0].z + p[0].w + i % c] = src[i % c];
}
"#;

/// `out = x` where positive, else `slope x`: `p[0]` the values, the slope's bits.
const LEAKY_RELU: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    let v = x[i];
    out[i] = select(v * bitcast<f32>(p[0].y), v, v > 0.0);
}
"#;

/// The most work (FLOPs) one dispatch of a convolution without tensor cores takes on: its voxels' tiles in chunks past
/// it, as a prompt's attention's ([`ATTENTION_DISPATCH_FLOPS`]).
const CONV_DISPATCH_FLOPS: f64 = (1u64 << 37) as f64;

/// `y[i] = sum over s of part[s * len + i]`, the splits in order. `p[0]`: len, splits.
const SUM_SPLITS: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let len = p[0].x;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= len) {
        return;
    }
    var acc = 0.0;
    for (var s = 0u; s < p[0].y; s++) {
        acc += part[s * len + i];
    }
    y[i] = acc;
}
"#;

/// A prompt's conv ([`SSM_CONV`]'s sums), a thread a (channel, token): its `kernel` inputs those up to it, the run's
/// or (before its first) the state's; no output another's input. The state is left to [`SSM_CONV_STATE`].
const SSM_CONV_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ch = p[0].x;
    let kern = p[0].z;
    let c = wg.x * 256u + li;
    let t = wg.y;
    if (c >= ch) { return; }
    var acc = 0.0;
    for (var k = 0u; k + 1u < kern; k++) {
        // input t + k - (kern - 1)
        let j = t + k;
        var xv = 0.0;
        if (j + 1u >= kern) { xv = qkv[(j + 1u - kern) * ch + c]; } else { xv = cs[j * ch + c]; }
        acc += xv * cw[c * kern + k];
    }
    acc += qkv[t * ch + c] * cw[c * kern + kern - 1u];
    out[t * ch + c] = acc / (1.0 + exp(-acc));
}
"#;

/// The state after [`SSM_CONV_ROWS`]: the run's last `kernel - 1` inputs (a run as long at least), a thread a channel.
const SSM_CONV_STATE: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(6) var<storage, read_write> cs: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = p[0].x;
    let rows = p[0].y;
    let kern = p[0].z;
    let c = id.x;
    if (c >= ch) { return; }
    for (var k = 0u; k + 1u < kern; k++) { cs[k * ch + c] = qkv[(rows + 1u + k - kern) * ch + c]; }
}
"#;

/// A gated delta net's causal depthwise conv, a thread a channel through the run's tokens (as the host's): its
/// `kernel - 1` inputs before them from the state, each output through SiLU, the state left with the last inputs.
/// `p[0]`: the channels, the tokens, the kernel.
const SSM_CONV: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(6) var<storage, read_write> cs: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = p[0].x;
    let rows = p[0].y;
    let kern = p[0].z;
    let c = id.x;
    if (c >= ch) { return; }
    var win: array<f32, 8>;
    for (var k = 0u; k + 1u < kern; k++) { win[k] = cs[k * ch + c]; }
    for (var t = 0u; t < rows; t++) {
        let xin = qkv[t * ch + c];
        var acc = 0.0;
        for (var k = 0u; k + 1u < kern; k++) { acc += win[k] * cw[c * kern + k]; }
        acc += xin * cw[c * kern + kern - 1u];
        out[t * ch + c] = acc / (1.0 + exp(-acc));
        for (var k = 0u; k + 2u < kern; k++) { win[k] = win[k + 1u]; }
        win[kern - 2u] = xin;
    }
    for (var k = 0u; k + 1u < kern; k++) { cs[k * ch + c] = win[k]; }
}
"#;

/// A gated delta net's recurrence through the run's tokens, a workgroup a value head and a thread a row of its state
/// (as the host's `delta_net_step`), the row held in registers from the run's start to its end: each token's q and k
/// (its key head's) L2-normed, the state decayed by `exp(softplus(alpha + dt_bias) * a)`, the delta rule's update by
/// `sigmoid(beta)`, the state read by q, and the result RMS-normed over the head and gated by `silu(z)` (or
/// `sigmoid(z)`). `DK` (the heads' size, `k_dim` = `v_dim`) is made a constant, and the row `DK / 4` vectors of its
/// own ([`delta_net_one`] puts their code in: an array indexed in a loop is the thread's memory, not its registers).
/// `p[0]`: the value heads, the key heads; `p[1]`: the tokens, the bits of the q scale and of eps, the sigmoid gate. (A
/// row read from memory at each token: 43 us a layer for a decode step of Qwen3.8 27B, 30 us a token of a prompt.)
const DELTA_NET: &str = r#"
@group(0) @binding(0) var<storage, read> cv: array<f32>;
@group(0) @binding(1) var<storage, read> zz: array<f32>;
@group(0) @binding(2) var<storage, read> ba: array<f32>;
@group(0) @binding(3) var<storage, read> sa: array<f32>;
@group(0) @binding(4) var<storage, read> dt: array<f32>;
@group(0) @binding(5) var<storage, read> nm: array<f32>;
@group(0) @binding(6) var<storage, read_write> st: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;
const DK4: u32 = DK_VALUEu / 4u;

var<workgroup> qn: array<vec4<f32>, DK4>;
var<workgroup> kn: array<vec4<f32>, DK4>;
var<workgroup> red: array<f32, 2 * DK>;

@compute @workgroup_size(DK)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let nv = p[0].x;
    let nk = p[0].y;
    let rows = p[1].x;
    let scale_q = bitcast<f32>(p[1].y);
    let eps = bitcast<f32>(p[1].z);
    let sig = p[1].w;
    let h = wg.x;
    let hk = h % nk;
    let ch = 2u * nk * DK + nv * DK;
    let base = (h * DK + i) * DK4;
ROW_LOAD
    for (var t = 0u; t < rows; t++) {
        let r0 = t * ch;
        let q = cv[r0 + hk * DK + i];
        let k = cv[r0 + nk * DK + hk * DK + i];
        red[i] = q * q;
        red[DK + i] = k * k;
        workgroupBarrier();
        for (var s = DK / 2u; s > 0u; s /= 2u) {
            if (i < s) {
                red[i] += red[i + s];
                red[DK + i] += red[DK + i + s];
            }
            workgroupBarrier();
        }
        let inv_q = 1.0 / sqrt(red[0] + eps);
        let inv_k = 1.0 / sqrt(red[DK] + eps);
        qn[i / 4u][i % 4u] = q * inv_q * scale_q;
        kn[i / 4u][i % 4u] = k * inv_k;
        workgroupBarrier();
        let bt = 1.0 / (1.0 + exp(-ba[t * 2u * nv + h]));
        let ab = ba[t * 2u * nv + nv + h] + dt[h];
        var sp = ab;
        if (ab <= 20.0) {
            let e = exp(ab);
            // ln(1 + e) as ln_1p gives it, where 1 + e would round e away
            sp = select(log(1.0 + e), e * (1.0 - 0.5 * e), e < 1e-4);
        }
        let g = exp(sp * sa[h]);
        // the decayed row read by k, its sums in the host's order
        var kv = 0.0;
ONE_KV
        let delta = (cv[r0 + 2u * nk * DK + h * DK + i] - kv) * bt;
        var core = 0.0;
ONE_UPDATE
        red[i] = core * core;
        workgroupBarrier();
        for (var s = DK / 2u; s > 0u; s /= 2u) {
            if (i < s) { red[i] += red[i + s]; }
            workgroupBarrier();
        }
        let inv = 1.0 / sqrt(red[0] / f32(DK) + eps);
        let zv = zz[t * nv * DK + h * DK + i];
        var gate = zv / (1.0 + exp(-zv));
        if (sig != 0u) { gate = 1.0 / (1.0 + exp(-zv)); }
        out[t * nv * DK + h * DK + i] = core * inv * nm[i] * gate;
        workgroupBarrier();
    }
ROW_STORE
}
"#;

/// [`DELTA_NET`] for heads `dk` wide: the row's `dk / 4` vectors and their code, each sum in the host's order.
fn delta_net_one(dk: usize) -> String {
    let n = dk / 4;
    let load: String = (0..n).map(|c| format!("    var sr{c} = st[base + {c}u];\n")).collect();
    let kv: String = (0..n)
        .map(|c| format!("        {{\n            let s = sr{c} * g;\n            let kk = kn[{c}u];\n            kv += s.x * kk.x;\n            kv += s.y * kk.y;\n            kv += s.z * kk.z;\n            kv += s.w * kk.w;\n        }}\n"))
        .collect();
    let update: String = (0..n)
        .map(|c| format!("        {{\n            let s = sr{c} * g + delta * kn[{c}u];\n            sr{c} = s;\n            let qq = qn[{c}u];\n            core += s.x * qq.x;\n            core += s.y * qq.y;\n            core += s.z * qq.z;\n            core += s.w * qq.w;\n        }}\n"))
        .collect();
    let store: String = (0..n).map(|c| format!("    st[base + {c}u] = sr{c};\n")).collect();
    DELTA_NET
        .replace("DK_VALUE", &dk.to_string())
        .replace("ROW_LOAD\n", &load)
        .replace("ONE_KV\n", &kv)
        .replace("ONE_UPDATE\n", &update)
        .replace("ROW_STORE\n", &store)
}

/// A prompt's delta net in three passes, the first: each token's q and k (its key heads') L2-normed as
/// [`DELTA_NET`] norms them, a workgroup a (key head, token), into the scratch `qk` (`[token, key head]`: q then k),
/// and the token's `sigmoid(beta)` and decay of each value head the key head's (after them, `[token, value head]`).
const DELTA_NET_PREP: &str = r#"
@group(0) @binding(0) var<storage, read> cv: array<f32>;
@group(0) @binding(2) var<storage, read> ba: array<f32>;
@group(0) @binding(3) var<storage, read> sa: array<f32>;
@group(0) @binding(4) var<storage, read> dt: array<f32>;
@group(0) @binding(6) var<storage, read_write> qk: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;

var<workgroup> red: array<f32, 2 * DK>;

@compute @workgroup_size(DK)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let nv = p[0].x;
    let nk = p[0].y;
    let rows = p[1].x;
    let scale_q = bitcast<f32>(p[1].y);
    let eps = bitcast<f32>(p[1].z);
    let hk = wg.x;
    let t = wg.y;
    let r0 = t * (2u * nk * DK + nv * DK);
    let q = cv[r0 + hk * DK + i];
    let k = cv[r0 + nk * DK + hk * DK + i];
    red[i] = q * q;
    red[DK + i] = k * k;
    workgroupBarrier();
    for (var s = DK / 2u; s > 0u; s /= 2u) {
        if (i < s) {
            red[i] += red[i + s];
            red[DK + i] += red[DK + i + s];
        }
        workgroupBarrier();
    }
    let inv_q = 1.0 / sqrt(red[0] + eps);
    let inv_k = 1.0 / sqrt(red[DK] + eps);
    let o = (t * nk + hk) * 2u * DK;
    qk[o + i] = q * inv_q * scale_q;
    qk[o + DK + i] = k * inv_k;
    // the value heads that read this key head (h % nk)
    if (i < nv / nk) {
        let h = hk + i * nk;
        let bt = 1.0 / (1.0 + exp(-ba[t * 2u * nv + h]));
        let ab = ba[t * 2u * nv + nv + h] + dt[h];
        var sp = ab;
        if (ab <= 20.0) {
            let e = exp(ab);
            sp = select(log(1.0 + e), e * (1.0 - 0.5 * e), e < 1e-4);
        }
        let bg = rows * nk * 2u * DK + (t * nv + h) * 2u;
        qk[bg] = bt;
        qk[bg + 1u] = exp(sp * sa[h]);
    }
}
"#;

/// The second: the recurrence through the tokens, a workgroup `R` rows of a value head's state and a thread a row
/// held in registers from the first token to the last (as `DK / 4` vectors of its own, the code unrolled: an array
/// indexed in a loop is the thread's memory, not its registers); each token's q and k (the first pass's) in the
/// workgroup's memory, the next token's loaded as this one's are used (one barrier a token); each token's state read
/// by q into `out`, as yet unnormed. ([`delta_net_scan`] puts the rows' code in.)
const DELTA_NET_SCAN: &str = r#"
@group(0) @binding(0) var<storage, read> qk4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cv: array<f32>;
@group(0) @binding(2) var<storage, read> qk: array<f32>;
@group(0) @binding(6) var<storage, read_write> st: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;
const DK4: u32 = DK_VALUEu / 4u;
const R: u32 = R_VALUEu;

// two tokens' q then k
var<workgroup> qks: array<vec4<f32>, 4 * DK4>;

@compute @workgroup_size(R)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nv = p[0].x;
    let nk = p[0].y;
    let rows = p[1].x;
    let h = wg.x / (DK / R);
    let i = (wg.x % (DK / R)) * R + li;
    let hk = h % nk;
    let ch = 2u * nk * DK + nv * DK;
    let base = (h * DK + i) * DK4;
    let bg0 = rows * nk * 2u * DK;
ROW_LOAD
    for (var c = li; c < 2u * DK4; c += R) { qks[c] = qk4[hk * 2u * DK4 + c]; }
    workgroupBarrier();
    for (var t = 0u; t < rows; t++) {
        let cur = (t % 2u) * 2u * DK4;
        let ks = cur + DK4;
        // the next token's q and k loaded now, stored once this one's are used
        let more = t + 1u < rows;
        let o4 = ((t + 1u) * nk + hk) * 2u * DK4;
        var pre0 = vec4<f32>(0.0);
        var pre1 = vec4<f32>(0.0);
        if (more && li < 2u * DK4) { pre0 = qk4[o4 + li]; }
        if (more && li + R < 2u * DK4) { pre1 = qk4[o4 + li + R]; }
        let bt = qk[bg0 + (t * nv + h) * 2u];
        let g = qk[bg0 + (t * nv + h) * 2u + 1u];
        var ka = vec4<f32>(0.0);
        var kb = vec4<f32>(0.0);
ROW_KV
        let kd = ka + kb;
        let kv = (kd.x + kd.y + kd.z + kd.w) * g;
        let delta = (cv[t * ch + 2u * nk * DK + h * DK + i] - kv) * bt;
        var qa = vec4<f32>(0.0);
        var qb = vec4<f32>(0.0);
ROW_UPDATE
        let qd = qa + qb;
        out[t * nv * DK + h * DK + i] = qd.x + qd.y + qd.z + qd.w;
        let nxt = 2u * DK4 - cur;
        if (more && li < 2u * DK4) { qks[nxt + li] = pre0; }
        if (more && li + R < 2u * DK4) { qks[nxt + li + R] = pre1; }
        workgroupBarrier();
    }
ROW_STORE
}
"#;

/// [`DELTA_NET_SCAN`] for heads `dk` wide, `r` rows a workgroup: the row's `dk / 4` vectors and their code.
fn delta_net_scan(dk: usize, r: usize) -> String {
    let n = dk / 4;
    let load: String = (0..n).map(|c| format!("    var sr{c} = st[base + {c}u];\n")).collect();
    let kv: String = (0..n).map(|c| format!("        {} += sr{c} * qks[ks + {c}u];\n", if c % 2 == 0 { "ka" } else { "kb" })).collect();
    let update: String = (0..n).map(|c| format!("        sr{c} = sr{c} * g + delta * qks[ks + {c}u];\n        {} += sr{c} * qks[cur + {c}u];\n", if c % 2 == 0 { "qa" } else { "qb" })).collect();
    let store: String = (0..n).map(|c| format!("    st[base + {c}u] = sr{c};\n")).collect();
    DELTA_NET_SCAN
        .replace("DK_VALUE", &dk.to_string())
        .replace("R_VALUE", &r.to_string())
        .replace("ROW_LOAD\n", &load)
        .replace("ROW_KV\n", &kv)
        .replace("ROW_UPDATE\n", &update)
        .replace("ROW_STORE\n", &store)
}

/// The third: each token's state read by q RMS-normed over its head and gated as [`DELTA_NET`]'s, a workgroup a
/// (value head, token), in place.
const DELTA_NET_NORM: &str = r#"
@group(0) @binding(1) var<storage, read> zz: array<f32>;
@group(0) @binding(5) var<storage, read> nm: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;

var<workgroup> red: array<f32, DK>;

@compute @workgroup_size(DK)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let nv = p[0].x;
    let eps = bitcast<f32>(p[1].z);
    let sig = p[1].w;
    let o = (wg.y * nv + wg.x) * DK + i;
    let core = out[o];
    red[i] = core * core;
    workgroupBarrier();
    for (var s = DK / 2u; s > 0u; s /= 2u) {
        if (i < s) { red[i] += red[i + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(DK) + eps);
    let zv = zz[o];
    var gate = zv / (1.0 + exp(-zv));
    if (sig != 0u) { gate = 1.0 / (1.0 + exp(-zv)); }
    out[o] = core * inv * nm[i] * gate;
}
"#;

/// Positions a workgroup of [`ATTENTION_PART`] takes.
const SPLIT: usize = 256;

/// One query's attention over positions split in runs of 256, a workgroup a (query head `h`, run): its scores, their
/// largest `m`, the sum `l` of their exponentials after `m`, and the exponentials' weighted values, into the scratch
/// after the output. `w` the layer's cache (row `t`: K `[n_kv, hd]` then V), `x` the query `[n_h, hd]`, `y` the output
/// `[n_h, hd]` then each (head, run)'s `hd` weighted values and then its `m` and `l`. `p[0]`: `n_h`, `n_kv`, `hd`, the
/// positions' end; `p[1]`: their start, the runs, the bits of the scale. (A workgroup a head walked every position on
/// one thread a dimension: 37 ms of a 3B Llama's decode step at 2,000 positions.)
const ATTENTION_PART: &str = r#"
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let hi = p[0].w;
    let lo = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let h = wg.x;
    let run = wg.y;
    let kh = h / (n_h / n_kv);
    let kvd = n_kv * hd;
    let row = 2u * kvd;
    let start = lo + run * 256u;
    let end = min(start + 256u, hi);
    let t = start + li;
    var s = -3.4e38;
    if (t < end) {
        let kb = t * row + kh * hd;
        var d0 = 0.0;
        for (var d = 0u; d < hd; d++) { d0 += x[h * hd + d] * bitcast<f32>(w[kb + d]); }
        s = d0 * scale;
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] = max(red[li], red[li + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = 0.0;
    if (t < end) { e = exp(s - m); }
    sc[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let l = red[0];
    let part = n_h * hd + (h * runs + run) * hd;
    let n = end - start;
    for (var d = li; d < hd; d += 256u) {
        var acc = 0.0;
        for (var i = 0u; i < n; i++) { acc += sc[i] * bitcast<f32>(w[(start + i) * row + kvd + kh * hd + d]); }
        y[part + d] = acc;
    }
    if (li == 0u) {
        let ml = n_h * hd + n_h * runs * hd + (h * runs + run) * 2u;
        y[ml] = m;
        y[ml + 1u] = l;
    }
}
"#;

/// [`ATTENTION_PART`] for a head size a multiple of 4 (at most 512): the query in the workgroup's memory, each key read
/// a vec4 at a time with four running sums, and each value column's products in four running sums (a run's positions
/// in fours), where every load waited on the sum before it. The same parts and largest and sum as it leaves for the
/// join.
const ATTENTION_PART4: &str = r#"
@group(0) @binding(0) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> qs: array<vec4<f32>, 128>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let hi = p[0].w;
    let lo = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let h = wg.x;
    let run = wg.y;
    let kh = h / (n_h / n_kv);
    let hd4 = hd / 4u;
    let kvd4 = n_kv * hd4;
    let row4 = 2u * kvd4;
    if (li < hd4) {
        qs[li] = q4[h * hd4 + li];
    }
    workgroupBarrier();
    let start = lo + run * 256u;
    let end = min(start + 256u, hi);
    let t = start + li;
    var s = -3.4e38;
    if (t < end) {
        let kb = t * row4 + kh * hd4;
        var a = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d++) { a += qs[d] * kv4[kb + d]; }
        s = (a.x + a.y + a.z + a.w) * scale;
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] = max(red[li], red[li + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = 0.0;
    if (t < end) { e = exp(s - m); }
    sc[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let l = red[0];
    let part = n_h * hd + (h * runs + run) * hd;
    let n = end - start;
    // a value column a thread (its 4 columns where the head is wider than 256 columns' threads... 4 at a time)
    for (var d4 = li; d4 < hd4; d4 += 256u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        let vb = start * row4 + kvd4 + kh * hd4 + d4;
        var i = 0u;
        for (; i + 4u <= n; i += 4u) {
            a0 += sc[i] * kv4[vb + i * row4];
            a1 += sc[i + 1u] * kv4[vb + (i + 1u) * row4];
            a2 += sc[i + 2u] * kv4[vb + (i + 2u) * row4];
            a3 += sc[i + 3u] * kv4[vb + (i + 3u) * row4];
        }
        for (; i < n; i++) { a0 += sc[i] * kv4[vb + i * row4]; }
        let o = (a0 + a1) + (a2 + a3);
        y[part + d4 * 4u] = o.x;
        y[part + d4 * 4u + 1u] = o.y;
        y[part + d4 * 4u + 2u] = o.z;
        y[part + d4 * 4u + 3u] = o.w;
    }
    if (li == 0u) {
        let ml = n_h * hd + n_h * runs * hd + (h * runs + run) * 2u;
        y[ml] = m;
        y[ml + 1u] = l;
    }
}
"#;

/// The runs of [`ATTENTION_PART`] put together, a workgroup a head: each run's values rescaled from its largest to the
/// head's, over the sum of the rescaled sums. `p` as for the parts.
const ATTENTION_JOIN: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let hd = p[0].z;
    let runs = p[1].y;
    let h = wg.x;
    let ml = n_h * hd + n_h * runs * hd + h * runs * 2u;
    var m = -3.4e38;
    for (var r = 0u; r < runs; r++) { m = max(m, y[ml + r * 2u]); }
    var l = 0.0;
    for (var r = 0u; r < runs; r++) { l += exp(y[ml + r * 2u] - m) * y[ml + r * 2u + 1u]; }
    for (var d = li; d < hd; d += 256u) {
        var acc = 0.0;
        for (var r = 0u; r < runs; r++) {
            acc += exp(y[ml + r * 2u] - m) * y[n_h * hd + (h * runs + r) * hd + d];
        }
        y[h * hd + d] = acc / l;
    }
}
"#;

/// QSA's pooled block keys: `pooled[b, i]` the mean of `raw` rows `b ratio ..`, each `/ ratio` added in turn (as
/// `Backend::qsa_pool`), a workgroup a block. `p[0]`: blocks, ratio, d (at most 256).
const QSA_POOL: &str = r#"
@group(0) @binding(0) var<storage, read> raw: array<f32>;
@group(0) @binding(6) var<storage, read_write> pooled: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let blocks = p[0].x;
    let ratio = p[0].y;
    let d = p[0].z;
    let b = wg.x + wg.y * 65535u;
    if (b >= blocks || i >= d) { return; }
    var acc = 0.0;
    for (var c = 0u; c < ratio; c++) { acc += raw[(b * ratio + c) * d + i] / f32(ratio); }
    pooled[b * d + i] = acc;
}
"#;

/// QSA's block scores, a thread a (block, query): `scale * sum over the heads of relu(q[r, h] . pooled[j])` for the
/// blocks the query sees whole, `-inf` past them. `p[0]`: rows, heads, d, nb; `p[1]`: first, ratio, scale.
const QSA_SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> pooled: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> qs: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let heads = p[0].y;
    let d = p[0].z;
    let nb = p[0].w;
    let first = p[1].x;
    let ratio = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let r = wg.y;
    let hd = heads * d;
    for (var i = li; i < hd; i += 256u) { qs[i] = q[r * hd + i]; }
    workgroupBarrier();
    let j = wg.x * 256u + li;
    if (j >= nb) { return; }
    var total = bitcast<f32>(0xff800000u);
    if (j < (first + r + 1u) / ratio) {
        var sum = 0.0;
        for (var h = 0u; h < heads; h++) {
            var dot = 0.0;
            for (var i = 0u; i < d; i++) { dot += qs[h * d + i] * pooled[j * d + i]; }
            sum += max(dot, 0.0);
        }
        total = sum * scale;
    }
    scores[r * nb + j] = total;
}
"#;

/// QSA's selection, a workgroup a query: the blocks it sees whole sorted (bitonic, in the workgroup's memory) by score,
/// the larger first and the lower block of equals, the first `keep` marked and written in ascending order (all of them
/// where it sees no more). `p[0]`: rows, nb (at most 4096), first, ratio; `p[1]`: keep.
const QSA_SELECT: &str = r#"
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(6) var<storage, read_write> list: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> keys: array<u32, 4096>;
var<workgroup> ids: array<u32, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let nb = p[0].y;
    let first = p[0].z;
    let ratio = p[0].w;
    let keep = p[1].x;
    let r = wg.x;
    let visible = min((first + r + 1u) / ratio, nb);
    let count = min(visible, keep);
    var n = 1u;
    while (n < visible) { n *= 2u; }
    // a larger score a larger key (positive scores' sign bit set, negative ones' bits flipped)
    for (var i = t; i < n; i += 256u) {
        if (i < visible) {
            let u = bitcast<u32>(scores[r * nb + i]);
            keys[i] = select(u | 0x80000000u, ~u, (u >> 31u) == 1u);
            ids[i] = i;
        } else {
            keys[i] = 0u;
            ids[i] = 0xffffffffu;
        }
    }
    workgroupBarrier();
    if (visible > keep) {
        for (var size = 2u; size <= n; size *= 2u) {
            for (var stride = size / 2u; stride > 0u; stride /= 2u) {
                for (var i = t; i < n / 2u; i += 256u) {
                    let a = 2u * stride * (i / stride) + i % stride;
                    let b = a + stride;
                    let ka = keys[a];
                    let kb = keys[b];
                    let ia = ids[a];
                    let ib = ids[b];
                    let before = ka > kb || (ka == kb && ia < ib);
                    if (before != ((a & size) == 0u)) {
                        keys[a] = kb;
                        keys[b] = ka;
                        ids[a] = ib;
                        ids[b] = ia;
                    }
                }
                workgroupBarrier();
            }
        }
    }
    // the kept blocks flagged (keys reused), then their places in ascending order (each thread's run of blocks
    // counted, the runs' counts summed in ids, reused)
    for (var i = t; i < visible; i += 256u) { keys[i] = 0u; }
    workgroupBarrier();
    for (var i = t; i < count; i += 256u) {
        if (visible > keep) { keys[ids[i]] = 1u; } else { keys[i] = 1u; }
    }
    workgroupBarrier();
    let per = (visible + 255u) / 256u;
    let lo = t * per;
    let hi = min(lo + per, visible);
    var mine = 0u;
    for (var i = lo; i < hi; i++) { mine += keys[i]; }
    ids[t] = mine;
    workgroupBarrier();
    if (t == 0u) {
        var at = 0u;
        for (var w = 0u; w < 256u; w++) {
            let c = ids[w];
            ids[w] = at;
            at += c;
        }
    }
    workgroupBarrier();
    var slot = ids[t];
    for (var i = lo; i < hi; i++) {
        if (keys[i] == 1u) {
            list[r * keep + slot] = i;
            slot += 1u;
        }
    }
}
"#;

/// [`ATTENTION_PART4`] for a KV head's whole group of query heads at once (`attention_part_group`'s kernel before the
/// group's size is put in), a workgroup a (KV head, run): each key and each value read once for the group's heads,
/// where a workgroup a query head read them a head each (the 27B's six heads a KV head: at 15,888 positions a step's
/// sixteen layers' parts read the cache's 2 GB six times over, 4.3 ms of its 16.8). The scores a thread a key, a sum a
/// head; the softmaxes together, a head a vec4's component; the values a thread a column of four over a share of the
/// run's keys (256 threads: as many shares as the head's quarter goes into them), the shares' sums put together
/// through the workgroup's memory. The same parts, largest and sum a head for the join.
const ATTENTION_PART_GROUP: &str = r#"
@group(0) @binding(0) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
// the group's queries [head][hd / 4]; then each key's scores, then weights, as they are reduced (two vec4s a key);
// then the shares' sums but the first's [(share - 1) * hd / 4 + column][head]
var<workgroup> buf: array<vec4<f32>, 1152>;
// each key's weights, a head a component (two vec4s a key)
var<workgroup> sc: array<vec4<f32>, 512>;
const G: u32 = GROUPu;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let hi = p[0].w;
    let lo = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let kh = wg.x;
    let run = wg.y;
    let hd4 = hd / 4u;
    let kvd4 = n_kv * hd4;
    let row4 = 2u * kvd4;
    for (var i = li; i < G * hd4; i += 256u) { buf[i] = q4[kh * G * hd4 + i]; }
    workgroupBarrier();
    let start = lo + run * 256u;
    let end = min(start + 256u, hi);
    let t = start + li;
    let live = t < end;
    var s0 = vec4<f32>(-3.4e38);
    var s1 = vec4<f32>(-3.4e38);
    if (live) {
        let kb = t * row4 + kh * hd4;
SCORE_SUMS
        for (var d = 0u; d < hd4; d++) {
            let k = kv4[kb + d];
SCORE_STEPS
        }
SCORES
    }
    workgroupBarrier();
    buf[2u * li] = s0;
    buf[2u * li + 1u] = s1;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) {
            buf[2u * li] = max(buf[2u * li], buf[2u * (li + st)]);
            buf[2u * li + 1u] = max(buf[2u * li + 1u], buf[2u * (li + st) + 1u]);
        }
        workgroupBarrier();
    }
    let m0 = buf[0];
    let m1 = buf[1];
    workgroupBarrier();
    var e0 = vec4<f32>(0.0);
    var e1 = vec4<f32>(0.0);
    if (live) {
        e0 = exp(s0 - m0);
        e1 = exp(s1 - m1);
    }
    sc[2u * li] = e0;
    sc[2u * li + 1u] = e1;
    buf[2u * li] = e0;
    buf[2u * li + 1u] = e1;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) {
            buf[2u * li] += buf[2u * (li + st)];
            buf[2u * li + 1u] += buf[2u * (li + st) + 1u];
        }
        workgroupBarrier();
    }
    let l0 = buf[0];
    let l1 = buf[1];
    workgroupBarrier();
    // the thread's column of four and its share of the run's keys
    let d4 = li % hd4;
    let share = li / hd4;
    var n = 0u;
    if (end > start) { n = end - start; }
    let first = share * hd4;
    let last = min(first + hd4, n);
    let vb = start * row4 + kvd4 + kh * hd4 + d4;
VALUE_SUMS
    for (var i = first; i < last; i++) {
        let v = kv4[vb + i * row4];
        let w0 = sc[2u * i];
        let w1 = sc[2u * i + 1u];
VALUE_STEPS
    }
    if (share > 0u) {
        let at = ((share - 1u) * hd4 + d4) * G;
SHARE_STORES
    }
    workgroupBarrier();
    if (share == 0u) {
        for (var o = 1u; o < 256u / hd4; o++) {
            let at = ((o - 1u) * hd4 + d4) * G;
SHARE_ADDS
        }
STORES
    }
    if (li == 0u) {
LARGEST_AND_SUMS
    }
}
"#;

/// [`ATTENTION_PART_GROUP`] for groups of `g` query heads a KV head (2 to 8).
fn attention_part_group(g: usize) -> String {
    let c = |i: usize| ["x", "y", "z", "w"][i % 4];
    let each = |f: &dyn Fn(usize) -> String| (0..g).map(f).collect::<String>();
    let scores = (0..2)
        .map(|v| {
            let parts: Vec<String> = (0..4).map(|i| if 4 * v + i < g { format!("(a{0}.x + a{0}.y + a{0}.z + a{0}.w) * scale", 4 * v + i) } else { "-3.4e38".into() }).collect();
            format!("        s{v} = vec4<f32>({});\n", parts.join(", "))
        })
        .collect::<String>();
    ATTENTION_PART_GROUP
        .replace("GROUP", &g.to_string())
        .replace("SCORE_SUMS\n", &each(&|i| format!("        var a{i} = vec4<f32>(0.0);\n")))
        .replace("SCORE_STEPS\n", &each(&|i| format!("            a{i} += buf[{i}u * hd4 + d] * k;\n")))
        .replace("SCORES\n", &scores)
        .replace("VALUE_SUMS\n", &each(&|i| format!("    var v{i} = vec4<f32>(0.0);\n")))
        .replace("VALUE_STEPS\n", &each(&|i| format!("        v{i} += w{}.{} * v;\n", i / 4, c(i))))
        .replace("SHARE_STORES\n", &each(&|i| format!("        buf[at + {i}u] = v{i};\n")))
        .replace("SHARE_ADDS\n", &each(&|i| format!("            v{i} += buf[at + {i}u];\n")))
        .replace(
            "STORES\n",
            &each(&|i| format!("        {{\n            let at = n_h * hd + ((kh * G + {i}u) * runs + run) * hd + d4 * 4u;\n            y[at] = v{i}.x;\n            y[at + 1u] = v{i}.y;\n            y[at + 2u] = v{i}.z;\n            y[at + 3u] = v{i}.w;\n        }}\n")),
        )
        .replace(
            "LARGEST_AND_SUMS\n",
            &each(&|i| format!("        {{\n            let ml = n_h * hd + n_h * runs * hd + ((kh * G + {i}u) * runs + run) * 2u;\n            y[ml] = m{}.{};\n            y[ml + 1u] = l{}.{};\n        }}\n", i / 4, c(i), i / 4, c(i))),
        )
}

/// [`attention_part_group`] over a cache held as f16 halves ([`HALVE`]'s): each key and value a vec4 of f16, half the
/// bytes (a step's parts over 15,888 positions read 130 MB a layer at half the card's bandwidth).
fn attention_part_group_halved(g: usize) -> String {
    let f32s = attention_part_group(g);
    let halved = f32s
        .replace("var<storage, read> kv4: array<vec4<f32>>;", "var<storage, read> kv4: array<vec4<f16>>;")
        .replace("let k = kv4[kb + d];", "let k = vec4<f32>(kv4[kb + d]);")
        .replace("let v = kv4[vb + i * row4];", "let v = vec4<f32>(kv4[vb + i * row4]);");
    assert_eq!(halved.matches("f16").count(), 1, "the halves' kernel's reads");
    assert_eq!(halved.matches("vec4<f32>(kv4[").count(), 2, "the halves' kernel's reads");
    format!("enable f16;\n{halved}")
}

/// `p[0].y` pairs of `x2`'s values from pair `p[0].x` as f16, a pair a word of `q` at the same place: a cache's rows
/// as the halves [`attention_part_group_halved`] reads.
const HALVE: &str = r#"
@group(0) @binding(0) var<storage, read> unused: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 65535u * 256u;
    if (i >= p[0].y) { return; }
    let j = p[0].x + i;
    q[j] = pack2x16float(x2[j]);
}
"#;

/// Whether a step's attention parts take a KV head's group of `g` query heads of `head_dim` at once
/// ([`ATTENTION_PART_GROUP`]): 2 to 8 of them, the head 64, 128 or 256 wide, the shares' sums within the kernel's
/// memory (26.6 KB of the workgroup's, where a device allows it).
fn attention_group_for(g: usize, head_dim: usize, workgroup_bytes: u32) -> bool {
    matches!(head_dim, 64 | 128 | 256) && (2..=8).contains(&g) && g * (256 - head_dim / 4) <= 1152 && workgroup_bytes >= 26_624
}

/// QSA's attention part, [`ATTENTION_PART4`]'s sums over a query's entries in turn: its kept blocks' positions (from
/// `list`, ascending) then its tail block's, a workgroup a (head, run of 256 entries, query), the runs' parts as
/// [`ATTENTION_ROWS_PART`] lays them out (for [`ATTENTION_ROWS_JOIN`]), the grid's third axis the queries. `p[0]`: n_h,
/// n_kv, hd, first; `p[1]`: ratio, runs, scale, keep.
const QSA_ATTENTION_PART: &str = r#"
@group(0) @binding(0) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> list: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> qs: array<vec4<f32>, 128>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> at: array<u32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let first = p[0].w;
    let ratio = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let keep = p[1].w;
    let rows = nwg.z;
    let h = wg.x;
    let run = wg.y;
    let r = wg.z;
    let pos = first + r;
    let visible = (pos + 1u) / ratio;
    let count = min(visible, keep);
    let entries = count * ratio + (pos + 1u - visible * ratio);
    let kh = h / (n_h / n_kv);
    let hd4 = hd / 4u;
    let kvd4 = n_kv * hd4;
    let row4 = 2u * kvd4;
    if (li < hd4) {
        qs[li] = q4[(r * n_h + h) * hd4 + li];
    }
    let start = run * 256u;
    let end = min(start + 256u, entries);
    let e = start + li;
    // the entry's position: a kept block's, or the tail's
    var t = 0u;
    if (e < count * ratio) {
        t = list[r * keep + e / ratio] * ratio + e % ratio;
    } else {
        t = visible * ratio + (e - count * ratio);
    }
    at[li] = t;
    workgroupBarrier();
    var s = -3.4e38;
    if (e < end) {
        let kb = t * row4 + kh * hd4;
        var a = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d++) { a += qs[d] * kv4[kb + d]; }
        s = (a.x + a.y + a.z + a.w) * scale;
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] = max(red[li], red[li + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var ex = 0.0;
    if (e < end) { ex = exp(s - m); }
    sc[li] = ex;
    red[li] = ex;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let l = red[0];
    let unit = (r * n_h + h) * runs + run;
    let part = rows * n_h * hd + unit * hd;
    var n = 0u;
    if (end > start) { n = end - start; }
    for (var d4 = li; d4 < hd4; d4 += 256u) {
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        let vb = kvd4 + kh * hd4 + d4;
        var i = 0u;
        for (; i + 4u <= n; i += 4u) {
            a0 += sc[i] * kv4[vb + at[i] * row4];
            a1 += sc[i + 1u] * kv4[vb + at[i + 1u] * row4];
            a2 += sc[i + 2u] * kv4[vb + at[i + 2u] * row4];
            a3 += sc[i + 3u] * kv4[vb + at[i + 3u] * row4];
        }
        for (; i < n; i++) { a0 += sc[i] * kv4[vb + at[i] * row4]; }
        let o = (a0 + a1) + (a2 + a3);
        y[part + d4 * 4u] = o.x;
        y[part + d4 * 4u + 1u] = o.y;
        y[part + d4 * 4u + 2u] = o.z;
        y[part + d4 * 4u + 3u] = o.w;
    }
    if (li == 0u) {
        let ml = rows * n_h * hd + rows * n_h * runs * hd + unit * 2u;
        y[ml] = m;
        y[ml + 1u] = select(0.0, l, n > 0u);
    }
}
"#;

/// A uniform's eight words, the rest of `words` zero.
fn words8(words: &[u32]) -> [u32; 8] {
    let mut all = [0u32; 8];
    all[..words.len()].copy_from_slice(words);
    all
}

fn buffer(v: &DeviceVec) -> &wgpu::Buffer {
    v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector")
}

/// A tensor that is a chain's vector itself ([`DeviceChain::alias`]), read back when the host asks for it.
struct Aliased {
    v: DeviceVec,
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
}

impl std::fmt::Debug for Aliased {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Aliased({})", self.v.len)
    }
}

/// A buffer for a chain's vector of `len`.
/// A grid of `groups` workgroups of 256 for a kernel that indexes over two of its dimensions (`id.x + id.y * 65535 *
/// 256`): a dimension takes 65,535 at most (a chunk of 1,024 rows' FFN is 69,632).
fn grid(groups: u32) -> (u32, u32, u32) {
    (groups.min(65535), groups.div_ceil(65535).max(1), 1)
}

fn vec_buffer(gpu: &Gpu, len: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-chain-vec"),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl ggml_rs::DeviceStorage for Aliased {
    fn len(&self) -> usize {
        self.v.len
    }

    fn device_name(&self) -> &str {
        "webgpu"
    }

    fn copy_to_host(&self) -> Vec<f32> {
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let len = self.v.len;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-alias-read"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if len > 0 {
            enc.copy_buffer_to_buffer(buffer(&self.v), 0, &staging, 0, (len * 4) as u64);
        }
        self.gpu.queue().submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.gpu.wait(None);
        let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping a finished buffer");
        // (as bytes: every target wgpu runs on is little-endian)
        let mut host = vec![0f32; len];
        bytemuck::cast_slice_mut::<f32, u8>(&mut host).copy_from_slice(&view[..len * 4]);
        drop(view);
        staging.unmap();
        host
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_to_device(&self) -> Box<dyn ggml_rs::DeviceStorage> {
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let copy = vec_buffer(&self.gpu, self.v.len);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if self.v.len > 0 {
            enc.copy_buffer_to_buffer(buffer(&self.v), 0, &copy, 0, (self.v.len * 4) as u64);
        }
        self.gpu.queue().submit([enc.finish()]);
        Box::new(Aliased { v: DeviceVec { len: self.v.len, inner: Arc::new(copy) }, gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial) })
    }
}

impl DeviceChain for WgpuBackend {
    fn pieces_in_flight_at_most(&self, pieces: usize) {
        WgpuBackend::pieces_in_flight_at_most(self, pieces);
    }

    fn has_room(&self, bytes: u64) -> bool {
        // (the adapter's own count of its memory in use against its budget; a gigabyte left after)
        self.memory_budget().is_some_and(|(budget, used)| used.saturating_add(bytes).saturating_add(1 << 30) <= budget)
    }

    fn attention_halves(&self, n_h: usize, n_kv: usize, head_dim: usize) -> bool {
        // (OAIY_ATTENTION_F32: a step's attention over the cache as it is)
        static F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let g = if n_kv > 0 && n_h % n_kv == 0 { n_h / n_kv } else { 0 };
        !*F32.get_or_init(|| std::env::var_os("OAIY_ATTENTION_F32").is_some() || std::env::var_os("OAIY_ATTENTION_PART4").is_some())
            && self.gpu.device.features().contains(wgpu::Features::SHADER_F16)
            && attention_group_for(g, head_dim, self.gpu.limits.max_compute_workgroup_storage_size)
    }

    fn vec(&self, len: usize) -> DeviceVec {
        DeviceVec { len, inner: Arc::new(vec_buffer(&self.gpu, len)) }
    }

    fn vec_f16(&self, values: &[f32]) -> Option<DeviceVec> {
        use rayon::prelude::*;
        // checked and packed on every core (a model's hyper-connections are some 700 million values)
        let exact = values.len() % 2 == 0 && values.par_chunks(1 << 16).all(|c| c.iter().all(|&v| half::f16::from_f32(v).to_f32().to_bits() == v.to_bits()));
        if !exact {
            return None;
        }
        let words: Vec<f32> = values.par_chunks_exact(2).map(|p| f32::from_bits(half::f16::from_f32(p[0]).to_bits() as u32 | (half::f16::from_f32(p[1]).to_bits() as u32) << 16)).collect();
        let v = self.vec(words.len());
        DeviceChain::upload(self, &v, &words);
        Some(v)
    }

    fn conv3d_weights(&self, w: &[f32], cout: usize, cin: usize) -> Option<DeviceVec> {
        // (the tensor cores' kernel's layout, the f32 one's too: CONV_F32_TILED)
        if w.len() != cout * cin * 27 {
            return None;
        }
        let cp = cin.div_ceil(32) * 32;
        let mut packed = vec![0f32; cout * 27 * cp];
        for co in 0..cout {
            for c in 0..cin {
                for tap in 0..27 {
                    packed[(co * 27 + tap) * cp + c] = w[(co * cin + c) * 27 + tap];
                }
            }
        }
        self.vec_f16_rounded(&packed)
    }

    fn conv1d_weights(&self, w: &[f32], cout: usize, cin: usize, k: usize) -> Option<DeviceVec> {
        if w.len() != cout * cin * k {
            return None;
        }
        let cp = cin.div_ceil(32) * 32;
        let mut packed = vec![0f32; cout * k * cp];
        for co in 0..cout {
            for c in 0..cin {
                for tap in 0..k {
                    packed[(co * k + tap) * cp + c] = w[(co * cin + c) * k + tap];
                }
            }
        }
        self.vec_f16_rounded(&packed)
    }

    fn conv_weights(&self, w: &[f32], cout: usize, cin: usize, k: usize) -> Option<DeviceVec> {
        let taps = k * k;
        if !matches!(k, 1 | 3 | 7) || w.len() != cout * cin * taps {
            return None;
        }
        let cp = cin.div_ceil(32) * 32;
        let mut packed = vec![0f32; cout * taps * cp];
        for co in 0..cout {
            for c in 0..cin {
                for tap in 0..taps {
                    packed[(co * taps + tap) * cp + c] = w[(co * cin + c) * taps + tap];
                }
            }
        }
        self.vec_f16_rounded(&packed)
    }

    fn attention_rows_full_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
        // the tensor cores' kernel writes its rows padded to 32 and keeps nothing else there
        if rows >= 16 && matches!(head_dim, 64 | 128 | 256) && self.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            rows.div_ceil(32) * 32 * n_h * head_dim
        } else {
            self.attention_rows_out_len(rows, n_h, head_dim, kv_len)
        }
    }

    fn nvfp4_weights(&self, packed: &[u8], scales: &[u8], global: f32, rows: usize, cols: usize) -> Option<(DeviceVec, DeviceVec)> {
        if cols % 64 != 0 || packed.len() != rows * cols / 2 || scales.len() != rows * cols / 16 || !self.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return None;
        }
        // a row's nibbles' words, then its scales' (four a word)
        let (nb, sb) = (cols / 2, cols / 16);
        let mut words: Vec<f32> = Vec::with_capacity(rows * (nb + sb) / 4);
        for r in 0..rows {
            let row = packed[r * nb..(r + 1) * nb].iter().chain(&scales[r * sb..(r + 1) * sb]);
            let bytes: Vec<u8> = row.copied().collect();
            words.extend(bytes.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))));
        }
        let w = self.vec(words.len());
        DeviceChain::upload(self, &w, &words);
        let s = self.vec(4);
        DeviceChain::upload(self, &s, &[1.0, global, 0.0, 0.0]);
        Some((w, s))
    }

    fn vec_f16_rounded(&self, values: &[f32]) -> Option<DeviceVec> {
        use rayon::prelude::*;
        if values.len() % 2 != 0 || values.par_chunks(1 << 16).any(|c| c.iter().any(|v| !v.is_finite() || v.abs() > 65504.0)) {
            return None;
        }
        let words: Vec<f32> = values.par_chunks_exact(2).map(|p| f32::from_bits(half::f16::from_f32(p[0]).to_bits() as u32 | (half::f16::from_f32(p[1]).to_bits() as u32) << 16)).collect();
        let v = self.vec(words.len());
        DeviceChain::upload(self, &v, &words);
        Some(v)
    }

    fn zero(&self, v: &DeviceVec) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(buffer(v), 0, None);
        self.gpu.queue().submit([enc.finish()]);
    }

    fn alias(&self, v: &DeviceVec, shape: Vec<usize>) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), v.len, "chain: an alias of {} values as {shape:?}", v.len);
        let storage = Aliased { v: v.clone(), gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial) };
        Tensor::from_device(Box::new(storage), shape)
    }

    fn aliased(&self, t: &Tensor) -> Option<DeviceVec> {
        let a = t.device_storage()?.as_any().downcast_ref::<Aliased>()?;
        Arc::ptr_eq(&a.gpu, &self.gpu).then(|| a.v.clone())
    }

    fn upload_at(&self, v: &DeviceVec, offset: usize, data: &[f32]) {
        assert!(offset + data.len() <= v.len, "chain: {} values at {offset} into a vector of {}", data.len(), v.len);
        if !data.is_empty() {
            // (the values' own bytes: every target wgpu runs on is little-endian)
            self.gpu.write(buffer(v), (offset * 4) as u64, bytemuck::cast_slice(data));
            // a large write's staging (device memory, with Resizable BAR) let go now: a model's weights uploaded in
            // turn held it all until the next submit (Qwen Image's 14 GB took 31 of a 32 GB card)
            if data.len() >= 16 << 20 {
                self.gpu.queue().submit([]);
                self.gpu.wait(None);
            }
        }
    }

    fn resize(&self, v: &DeviceVec, len: usize) -> DeviceVec {
        let grown = self.vec(len);
        let keep = v.len.min(len);
        if keep > 0 {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buffer(v), 0, buffer(&grown), 0, (keep * 4) as u64);
            self.gpu.queue().submit([enc.finish()]);
        }
        grown
    }

    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize {
        n_h * head_dim + n_h * cap.div_ceil(SPLIT).max(1) * (head_dim + 2)
    }

    fn attention_rows_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
        // (one pass, tiled, or on the tensor cores, their rows padded to 32: the output alone; else its runs' parts)
        let padded = rows.div_ceil(32) * 32 * n_h * head_dim;
        if attention_tiled_for(rows, head_dim) {
            padded
        } else {
            padded.max(attention_runs_out_len(rows, n_h, head_dim, kv_len))
        }
    }

    fn qsa_attention_out_len(&self, rows: usize, n_h: usize, head_dim: usize, keep: usize, ratio: usize) -> usize {
        // the selection sorts 4096 blocks' keys and indices in a workgroup's memory (32 KB, past WebGPU's default 16)
        if self.gpu.limits.max_compute_workgroup_storage_size < 32768 {
            return 0;
        }
        rows * n_h * head_dim + rows * n_h * (keep * ratio + ratio).div_ceil(256) * (head_dim + 2)
    }

    fn holds_exl3(&self, w: &dyn ggml_rs::exl3::PackedLinear) -> bool {
        // (a GGUF's matrix where a packed projection is asked for: held as a quantized weight is)
        if let Some(q) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::QuantLinear>()) {
            return self.holds(&q.w);
        }
        w.as_any()
            .and_then(|a| a.downcast_ref::<crate::exl3::Exl3Gpu>())
            .is_some_and(|g| g.is_on(&self.gpu) && g.single_chunk().is_some())
    }

    fn holds_experts(&self, e: &dyn ggml_rs::exl3::Experts) -> bool {
        e.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()).is_some_and(|g| g.is_on(&self.gpu))
    }

    fn holds(&self, w: &QuantizedTensor) -> bool {
        w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).is_some_and(|q| Arc::ptr_eq(&q.gpu, &self.gpu))
    }

    fn copy_weight(&self, w: &QuantizedTensor) -> Option<QuantizedTensor> {
        WgpuBackend::copy_weight(self, w)
    }

    fn begin(&self) -> Box<dyn ChainRecorder + '_> {
        Box::new(Recorder { backend: self, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 })
    }
}

impl Recorder<'_> {
    /// [`ChainRecorder::exl3_rows`], the input `x`, or (`up` given) the SwiGLU `silu(x) * up` computed as the input
    /// transform reads it (a shared expert's down projection: a dispatch fewer).
    pub(crate) fn exl3_rows_of(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, x: &DeviceVec, up: Option<&DeviceVec>, y: &DeviceVec, rows: usize) {
        // a GGUF's matrix: the quantized matmul of its rows as they are (no transform either side, no channel map),
        // a SwiGLU's product made first where one is asked for
        if let Some(q) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::QuantLinear>()) {
            let (k, n) = q.kn();
            assert!(rows > 0 && x.len >= rows * k && y.len >= rows * n, "chain: a quantized projection [{n}, {k}] of {rows} rows");
            match up {
                Some(u) => {
                    let t = self.scratch(rows * k);
                    self.silu_mul(x, u, &t, rows * k);
                    self.matmul_rows(&q.w, &t, y, rows);
                }
                None => self.matmul_rows(&q.w, x, y, rows),
            }
            return;
        }
        let g = w.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3Gpu>()).expect("an EXL3 projection this adapter holds");
        assert!(g.is_on(&self.backend.gpu), "chain: an EXL3 projection of another adapter");
        let (words, splits) = g.single_chunk().expect("an EXL3 projection in one buffer");
        let (k, n) = g.kn();
        assert!(rows > 0 && x.len >= rows * k && y.len >= rows * n && rows <= 65535, "chain: an EXL3 [{n}, {k}] of {rows} rows");
        let c = g.chain(self.backend);
        // a step's one row: the projection's own scratch (its bind groups kept); a check's few rows the device's shared
        // few-rows scratch (kept too); else this call's
        let few = (2..=crate::exl3::FEW_MAX).contains(&rows) && self.keep && crate::exl3::FewScratch::fits(k, n, splits as usize);
        let (xh, part, yt, jobs) = if rows == 1 && self.keep {
            (c.xh.clone(), c.part.clone(), c.yt.clone(), c.jobs1.clone())
        } else if few {
            let f = self.gpu().few(self.backend);
            (f.xh.clone(), f.part.clone(), f.yt.clone(), f.jobs.clone())
        } else {
            let list: Vec<u32> = (0..rows as u32).flat_map(|r| [0, r]).collect();
            let jobs = self.scratch(list.len());
            crate::exl3::upload_u32(self.backend, &jobs, &list);
            let lens = [rows * k, rows * (splits as usize).max(COOP_SPLITS_MAX) * n, rows * n];
            let [xh, part, yt] = match self.exl3_tmp.take() {
                Some(t) if t.iter().zip(lens).all(|(v, len)| v.len >= len) => t,
                Some([a, b, c]) => {
                    let grow = |v: DeviceVec, len: usize, r: &mut Self| if v.len >= len { v } else { r.scratch(len.max(v.len)) };
                    [grow(a, lens[0], self), grow(b, lens[1], self), grow(c, lens[2], self)]
                }
                None => [self.scratch(lens[0]), self.scratch(lens[1]), self.scratch(lens[2])],
            };
            self.exl3_tmp = Some([xh.clone(), part.clone(), yt.clone()]);
            (xh, part, yt, jobs)
        };
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let imap = c.imap.as_ref().map_or(&d, buffer).clone();
        match up {
            Some(u) => {
                assert!(x.len >= rows * k && u.len >= rows * k, "chain: a SwiGLU's {rows} rows of {k}");
                self.dispatch_wide("exl3-pre-swiglu", crate::exl3::chain_shader("pre-swiglu"), [buffer(x), buffer(&c.suh), &imap, buffer(&jobs), buffer(u), &d, buffer(&xh), &drw], &[k as u32, c.imap.is_none() as u32, 1, 0, 1, 0], ((k / 128) as u32, rows as u32, 1));
            }
            None => self.dispatch_wide("exl3-pre", crate::exl3::chain_shader("pre"), [buffer(x), buffer(&c.suh), &imap, buffer(&jobs), &d, &d, buffer(&xh), &drw], &[k as u32, c.imap.is_none() as u32], ((k / 128) as u32, rows as u32, 1)),
        }
        let ntiles = (n / 16) as u32;
        let grid = |z: usize| (ntiles.min(65535), ntiles.div_ceil(65535), z as u32 * splits);
        let mm = crate::exl3::chain_shader("mm");
        // a prompt's rows on the tensor cores, in blocks of 128, split along k where its workgroups are too few to fill
        // the GPU twice over (a block of 128 rows of a projection 2,048 wide is 16 of them)
        let coop = rows > crate::exl3::FEW_MAX && crate::exl3::coop_on(self.gpu());
        let mut coop_splits = 1;
        if coop {
            const BLOCK: usize = 128;
            let many: Vec<u32> = (0..rows as u32).collect::<Vec<_>>().chunks(BLOCK).flat_map(|b| b.iter().copied().chain(std::iter::repeat(crate::exl3::NONE)).take(BLOCK)).collect();
            let order = self.scratch(many.len());
            crate::exl3::upload_u32(self.backend, &order, &many);
            let blocks = many.len() / BLOCK;
            let groups = ntiles.div_ceil(8) as usize * blocks;
            let kts = k / 16;
            let want = (2 * self.gpu().coop_units() as usize).div_ceil(groups).clamp(1, (kts / 8).clamp(1, COOP_SPLITS_MAX));
            coop_splits = kts.div_ceil(kts.div_ceil(want));
            let src = crate::exl3::g_coop(BLOCK);
            let per = 65535 / coop_splits;
            for first in (0..blocks).step_by(per) {
                let these = (per.min(blocks - first) * coop_splits) as u32;
                self.dispatch_wide(crate::exl3::coop_name(BLOCK), &src, [words, buffer(&xh), buffer(&jobs), buffer(&order), &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, coop_splits as u32, 0, first as u32], (ntiles.div_ceil(8), 1, these));
            }
        } else if rows == 1 {
            self.dispatch_wide("exl3-mm", mm, [words, buffer(&xh), buffer(&jobs), &d, &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0], grid(1));
        } else if rows <= crate::exl3::FEW_MAX {
            // a few rows (a check of drafts): each tile decoded once for all of them, each row summed as one row is
            let order = if few {
                self.gpu().few(self.backend).order.clone()
            } else {
                let order = self.scratch(rows);
                crate::exl3::upload_u32(self.backend, &order, &(0..rows as u32).collect::<Vec<_>>());
                order
            };
            self.dispatch_wide(crate::exl3::few_name(rows), crate::exl3::g_few(rows), [words, buffer(&xh), buffer(&jobs), buffer(&order), &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0, 0], grid(1));
        } else {
            // a prompt's rows summed as the projection's own passes sum them (each tile decoded once for 64 of them here),
            // and a last lone row of a pass as one row is
            const BLOCK: usize = 64;
            let lone = rows % 32 == 1;
            let many: Vec<u32> = (0..(rows - lone as usize) as u32).collect::<Vec<_>>().chunks(BLOCK).flat_map(|b| b.iter().copied().chain(std::iter::repeat(crate::exl3::NONE)).take(BLOCK)).collect();
            let order = self.scratch(many.len());
            crate::exl3::upload_u32(self.backend, &order, &many);
            let per = 65535 / splits as usize;
            let blocks = many.len() / BLOCK;
            let kernel = crate::exl3::g_many(BLOCK);
            for first in (0..blocks).step_by(per) {
                self.dispatch_wide(crate::exl3::many_name(BLOCK), &kernel, [words, buffer(&xh), buffer(&jobs), buffer(&order), &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0, first as u32], grid(per.min(blocks - first)));
            }
            if lone {
                self.dispatch_wide("exl3-mm", mm, [words, buffer(&xh), buffer(&jobs), &d, &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0, rows as u32 - 1], grid(1));
            }
        }
        let post = crate::exl3::chain_shader("post");
        let post_out = if c.omap.is_some() { buffer(&yt) } else { buffer(y) };
        let parts = if coop { coop_splits as u32 } else { splits };
        self.dispatch_wide("exl3-post", post, [buffer(&part), buffer(&c.svh), buffer(&jobs), &d, &d, &d, post_out, &drw], &[n as u32, parts], ((n / 128) as u32, rows as u32, 1));
        if let Some(omap) = &c.omap {
            let gather = crate::exl3::chain_shader("gather");
            self.dispatch_wide("exl3-gather", gather, [buffer(&yt), buffer(omap), buffer(&jobs), &d, &d, &d, buffer(y), &drw], &[n as u32], ((n as u32).div_ceil(256), rows as u32, 1));
        }
    }

    /// [`Self::exl3_rows_of`] of a SwiGLU: `silu(gate) * up`'s rows.
    pub(crate) fn exl3_rows_swiglu(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, gate: &DeviceVec, up: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.exl3_rows_of(w, gate, Some(up), y, rows);
    }
}

impl<'a> Recorder<'a> {
    /// A recording on `backend`, its bind groups kept (the crate's own measurements record kernels directly).
    #[cfg(test)]
    pub(crate) fn new(backend: &'a WgpuBackend) -> Self {
        Recorder { backend, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 }
    }
}

/// One dispatch: its pipeline, bind group and grid.
use crate::Dispatch;

/// A bind group a chain makes again step after step: its pipeline, its three buffers and its parameters.
pub(crate) type GroupKey = (usize, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, [u32; 8]);

/// A bind group of the eight-buffer layout a chain makes again step after step (`Gpu::wide_layout`).
pub(crate) type WideKey = (usize, [wgpu::Buffer; 8], [u32; 8]);

/// Bind groups kept before the cache starts over (a cache grown from buffers that were replaced).
const KEEP_GROUPS: usize = 16384;

/// The work (FLOPs) a piece is submitted past ([`Recorder::weigh`]): 2^38 (some 0.3 s at a TFLOP a second), eight times
/// that on tensor cores; or `OAIY_PIECE_FLOPS`'s.
fn piece_flops(gpu: &crate::Gpu) -> f64 {
    static FLOPS: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let set = *FLOPS.get_or_init(|| std::env::var("OAIY_PIECE_FLOPS").ok().and_then(|v| v.parse().ok()).filter(|&f: &f64| f > 0.0));
    // (a device that feeds its pieces a few at a time: the smaller ones, the one slow spell a card under a power limit
    // has through a long run of them 0.45 s shorter so, the 27B's 15,360 tokens 7.9 s the first time where 8.35, and
    // the run no slower after it)
    set.unwrap_or(if gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) && !gpu.feeds() { (1u64 << 41) as f64 } else { (1u64 << 38) as f64 })
}

/// What a dispatch's workgroup counts for in a piece's work ([`piece_flops`]), whatever its kernel: its 256 values'
/// reads and writes, about what a thousand FLOPs a value take on the tensor cores (an elementwise op over 300 million
/// values a seventh of a piece: some 3 ms of 20, and under a throttled memory still well short of a second).
const WORKGROUP_FLOPS: f64 = 262_144.0;

/// Dispatches a piece of a run submits ([`Recorder::finish`]'s): 128, or `OAIY_PIECE`'s.
fn piece() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("OAIY_PIECE").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(128))
}

/// The most splits along k of a prompt's EXL3 projection on the tensor cores.
const COOP_SPLITS_MAX: usize = 8;

pub(crate) struct Recorder<'a> {
    backend: &'a WgpuBackend,
    /// Every op's dispatch, in order, run in one compute pass (a pass an op cost more than the ops).
    dispatches: Vec<Dispatch>,
    /// What to read back once they have run: the vector, the element it starts at, a staging buffer, the length.
    reads: Vec<(wgpu::Buffer, usize, wgpu::Buffer, usize)>,
    /// Whether bind groups are kept for the steps after ([`ChainRecorder::keep_groups`]).
    keep: bool,
    /// The scratch it took from the GPU's pool ([`Recorder::scratch`]), given back when it has run.
    pooled: Vec<(u64, wgpu::Buffer)>,
    /// Of that, what nothing recorded after reads (an input's f16 or int8 rows once the input is written): taken
    /// again before the pool's (a prompt's 27B chunk made 256 f16 copies, 3.6 GB held to its end).
    spare: Vec<(u64, wgpu::Buffer)>,
    /// Inputs of several rows quantized to int8 for the int8 kernels so far (the vector, its rows and width, the int8
    /// rows): each quantized once for the matmuls that read it, until something writes it.
    q8: Vec<(wgpu::Buffer, usize, usize, DeviceVec)>,
    /// Inputs of a prompt's rows as f16 for the tensor cores so far, as `q8`.
    x16: Vec<(wgpu::Buffer, usize, usize, DeviceVec)>,
    /// The f16 queries and cache rows of the tensor cores' attention: each attention's converted into them as it runs.
    att16: Option<(DeviceVec, DeviceVec)>,
    /// The parts of a tensor-core matmul split along k, each split matmul's in turn.
    parts: Option<DeviceVec>,
    /// A prompt's EXL3 projections' scratch (transformed inputs, partial sums, outputs before the map), each
    /// projection's in turn (a device's layers one recording: each its own, they would all be held to its end).
    exl3_tmp: Option<[DeviceVec; 3]>,
    /// A prompt's routed experts' scratch, each layer's in turn (its shape: rows, top k, hidden, ff).
    pub(crate) moe_tmp: Option<([usize; 4], std::sync::Arc<crate::exl3::Step>)>,
    /// Whether what is recorded waits for [`ChainRecorder::finish`] ([`ChainRecorder::hold`]), and the pieces encoded
    /// as it is (each its command buffer, submitted in turn when it is let go).
    hold: bool,
    held: Vec<wgpu::CommandBuffer>,
    /// The held pieces that go by the device's feed ([`Recorder::fed`]): each its dispatches, encoded at its turn.
    lists: Vec<Vec<Dispatch>>,
    /// The reads copied out by a flush (the first so many of `reads`), and that flush's submission: a recording
    /// waits for its own work, not what went after it (the next chunk's).
    copied: usize,
    flushed: Option<crate::Piece>,
    /// `OAIY_PIECE_STAMPS`: the pieces' timestamps, and how many pieces so far; once a flush has copied them out
    /// (with its reads: a finish waits for its own work, not for what went after), where to
    stamps: Option<(wgpu::QuerySet, u32)>,
    stamped: Option<wgpu::Buffer>,
    /// `OAIY_CHAIN_PROFILE`: each piece's kernels and where their timestamps are copied (read at the finish).
    timed: Vec<(Vec<Arc<wgpu::ComputePipeline>>, wgpu::Buffer)>,
    /// The work (FLOPs) the heavy ops have said of the piece so far ([`Recorder::weigh`]).
    weight: f64,
}

impl Recorder<'_> {
    /// A dispatch recorded; a piece of [`piece`]'s submitted as soon as it is recorded, so the GPU runs
    /// a run's first ops while the CPU records the rest. A later upload (`Queue::write_buffer`) lands after the pieces
    /// already submitted and before the ones after, as the recording's order has it.
    fn push(&mut self, d: Dispatch) {
        // (each workgroup some memory's worth of work whatever its kernel: an op over a large level's values is all
        // traffic, weighed by none, and a piece of 128 of them over a 3D decoder's 4.5 million voxels ran past the
        // OS's 2 s once the card's power limiter had its memory throttled: a lost device)
        let groups = d.2 .0 as f64 * d.2 .1 as f64 * d.2 .2 as f64;
        self.dispatches.push(d);
        self.weight += groups * WORKGROUP_FLOPS;
        if self.dispatches.len() >= piece() || self.weight >= piece_flops(self.gpu()) {
            self.submit_piece();
        }
    }

    /// The dispatches recorded so far encoded as a piece (one compute pass) and submitted (held: kept to be, see
    /// [`ChainRecorder::hold`]); profiled (`OAIY_CHAIN_PROFILE`), each its own pass between two timestamps (a piece
    /// at a time: the whole recording one submission would run past the OS's limit on one, Windows' 2 s).
    /// Whether this recording's pieces go by the device's feed as their dispatches (its pieces in flight are limited):
    /// not one that times its pieces or its kernels (their passes are its own to encode), nor one that has held
    /// command buffers already.
    fn fed(&self) -> bool {
        self.gpu().feeds() && self.held.is_empty() && !crate::profile::chain_on() && !crate::profile::pieces_on()
    }

    fn submit_piece(&mut self) {
        if !self.dispatches.is_empty() && self.fed() {
            let list = std::mem::take(&mut self.dispatches);
            self.weight = 0.0;
            if self.hold {
                self.lists.push(list);
            } else {
                self.gpu().feed(list);
            }
            return;
        }
        if !self.dispatches.is_empty() {
            let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
            let start = std::time::Instant::now();
            let mut piece = self.gpu().device.create_command_encoder(&Default::default());
            if crate::profile::chain_on() && self.gpu().device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                let n = self.dispatches.len();
                let device = &self.gpu().device;
                let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("oaiy-chain-profile"), ty: wgpu::QueryType::Timestamp, count: 2 * n as u32 });
                for (i, (pipeline, group, (x, y, z))) in self.dispatches.iter().enumerate() {
                    let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &set, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) });
                    let mut pass = piece.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(*x, *y, *z);
                }
                let bytes = 16 * n as u64;
                let resolved = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-profile"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-profile-read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                piece.resolve_query_set(&set, 0..2 * n as u32, &resolved, 0);
                piece.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
                self.timed.push((self.dispatches.iter().map(|(p, _, _)| Arc::clone(p)).collect(), staging));
            } else {
                let i = self.next_stamp();
                let timestamp_writes = self.stamp_writes(i);
                let mut pass = piece.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
                for (pipeline, group, (x, y, z)) in &self.dispatches {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(*x, *y, *z);
                }
            }
            self.held.push(piece.finish());
            self.dispatches.clear();
            self.weight = 0.0;
            if !self.hold {
                let held = std::mem::take(&mut self.held);
                self.gpu().submit_piece(held);
            }
            crate::profile::add(&crate::profile::CHAIN_ENCODE, start);
        }
    }

    /// A heavy op's work (its FLOPs) added to the piece's: a piece past [`piece_flops`]'s is submitted then, so none
    /// runs long enough for the OS to reset the GPU (Windows: a submission past 2 s), however slow the GPU and big the
    /// work (a step of 4,096 tokens without tensor cores: 128 dispatches some 2.3 s).
    pub(crate) fn weigh(&mut self, flops: f64) {
        self.weight += flops;
        if self.weight >= piece_flops(self.gpu()) {
            self.submit_piece();
        }
    }

    /// `OAIY_PIECE_STAMPS`: the next piece's place among the recording's timestamps (up to 512 pieces).
    fn next_stamp(&mut self) -> Option<u32> {
        if !crate::profile::pieces_on() || !self.gpu().device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        if self.stamps.is_none() {
            let set = self.gpu().device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("oaiy-chain-pieces"), ty: wgpu::QueryType::Timestamp, count: 1024 });
            self.stamps = Some((set, 0));
        }
        let (_, n) = self.stamps.as_mut().expect("made");
        if *n >= 512 {
            return None;
        }
        *n += 1;
        Some(*n - 1)
    }

    /// The pieces' timestamps resolved and copied out by `enc` (where there are any): the buffer they are read from.
    fn resolve_pieces(&mut self, enc: &mut wgpu::CommandEncoder) -> Option<wgpu::Buffer> {
        let (set, n) = self.stamps.take().filter(|(_, n)| *n > 0)?;
        let device = &self.gpu().device;
        let bytes = 16 * n as u64;
        let resolved = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-pieces"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-pieces-read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        enc.resolve_query_set(&set, 0..2 * n, &resolved, 0);
        enc.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
        Some(staging)
    }

    /// Piece `i`'s pass's timestamps.
    fn stamp_writes(&self, i: Option<u32>) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (set, _) = self.stamps.as_ref()?;
        let i = i?;
        Some(wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(2 * i), end_of_pass_write_index: Some(2 * i + 1) })
    }

    /// A vector of `len` for this recording alone (a prompt's scratch): from the GPU's pool, given back when the
    /// recording has run, so nothing may keep it. Its values are whatever it last held.
    pub(crate) fn scratch(&mut self, len: usize) -> DeviceVec {
        let bytes = ((len.max(1) * 4) as u64).next_power_of_two().max(256);
        if let Some(i) = self.spare.iter().position(|(b, _)| *b == bytes) {
            let (_, b) = self.spare.swap_remove(i);
            return DeviceVec { len, inner: Arc::new(b) };
        }
        let b = self.gpu().pooled(bytes);
        self.pooled.push((bytes, b.clone()));
        DeviceVec { len, inner: Arc::new(b) }
    }

    /// The backend recorded on.
    pub(crate) fn backend(&self) -> &WgpuBackend {
        self.backend
    }

    /// Whether this recording keeps its bind groups.
    pub(crate) fn keeps(&self) -> bool {
        self.keep
    }

    pub(crate) fn gpu(&self) -> &Gpu {
        &self.backend.gpu
    }

    fn uniform(&self, words: &[u32]) -> wgpu::Buffer {
        let all = words8(words);
        let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // (a new buffer: its write is in no waiting piece's way, so not after them as `queue`'s are)
        self.gpu().queue_raw.write_buffer(&buf, 0, &bytes);
        buf
    }

    /// A dispatch whose bind group is the same every step (its buffers and parameters): made once and kept.
    /// `b` written by what was just recorded: its int8 rows (if quantized) are stale.
    fn wrote(&mut self, b: &wgpu::Buffer) {
        let spare = &mut self.spare;
        let mut stale = |(x, _, _, v): &(wgpu::Buffer, usize, usize, DeviceVec)| {
            if x != b {
                return true;
            }
            // what read its rows is recorded: the scratch may be written again (wgpu orders the two)
            spare.push((buffer(v).size(), buffer(v).clone()));
            false
        };
        if !self.q8.is_empty() {
            self.q8.retain(&mut stale);
        }
        if !self.x16.is_empty() {
            self.x16.retain(&mut stale);
        }
    }

    pub(crate) fn dispatch_kept(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, words: &[u32], groups: (u32, u32, u32)) {
        self.wrote(at2);
        if !self.keep {
            let params = self.uniform(words);
            self.dispatch(pipeline, at0, at1, at2, &params, groups);
            return;
        }
        let key: GroupKey = (Arc::as_ptr(pipeline) as usize, at0.clone(), at1.clone(), at2.clone(), words8(words));
        let kept = self.gpu().chain_groups.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned();
        let group = match kept {
            Some(group) => group,
            None => {
                let params = self.uniform(words);
                let group = self.group(at0, at1, at2, &params);
                let mut groups = self.gpu().chain_groups.lock().unwrap_or_else(|p| p.into_inner());
                if groups.len() >= KEEP_GROUPS {
                    groups.clear();
                }
                groups.insert(key, group.clone());
                group
            }
        };
        self.push((Arc::clone(pipeline), group, groups));
    }

    fn group(&self, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, params: &wgpu::Buffer) -> wgpu::BindGroup {
        self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oaiy-chain"),
            layout: &self.gpu().layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: at0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: at1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        })
    }

    fn dispatch(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, params: &wgpu::Buffer, groups: (u32, u32, u32)) {
        self.wrote(at2);
        let group = self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oaiy-chain"),
            layout: &self.gpu().layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: at0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: at1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });
        self.push((Arc::clone(pipeline), group, groups));
    }

    /// [`ChainRecorder::attention_rows`] in f32: the positions in runs of 256 a workgroup a (head, run, query), then the
    /// runs joined.
    pub(crate) fn attention_rows_f32(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) {
        self.attention_rows_f32_masked(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, false)
    }

    /// [`Self::attention_rows_f32`], with `full` every query over every position (no causal mask): in one pass
    /// ([`ATTENTION_TILED`]) where [`attention_tiled_for`], its queries in chunks of bounded work
    /// ([`ATTENTION_DISPATCH_FLOPS`]); else in runs joined ([`Self::attention_rows_runs`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_f32_masked(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) {
        if attention_tiled_for(rows, head_dim) {
            let per_row = 4.0 * (if full { past } else { past + rows }).max(1) as f64 * (n_h * head_dim) as f64;
            let tq = if head_dim == 256 { 32 } else { 64 };
            let chunk = (((ATTENTION_DISPATCH_FLOPS / per_row) as usize) / tq * tq).max(tq);
            self.attention_rows_tiled(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, full, chunk);
        } else {
            self.attention_rows_runs(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, full);
        }
    }

    /// [`ATTENTION_TILED`]'s attention of `rows` queries, `chunk` of them a dispatch, each dispatch's work weighed
    /// ([`Recorder::weigh`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_tiled(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool, chunk: usize) {
        // (a full attention's `past` its positions, whatever its queries)
        let kv_len = if full { past } else { past + rows };
        assert!(
            rows > 0 && n_kv > 0 && n_h % n_kv == 0 && chunk > 0 && kv.len >= kv_len * 2 * n_kv * head_dim && q.len >= rows * n_h * head_dim && out.len >= rows * n_h * head_dim && window.unwrap_or(0) < 1 << 31,
            "chain: a prompt's attention's buffers"
        );
        let name = match head_dim {
            64 => "chain-attention-tiled-64",
            128 => "chain-attention-tiled-128",
            _ => "chain-attention-tiled-256",
        };
        let tq = if head_dim == 256 { 32 } else { 64 };
        let pipeline = self.gpu().named_pipeline(name, || attention_tiled(head_dim));
        let mask = full as u32 | (window.unwrap_or(0) as u32) << 1;
        let mut first = 0;
        while first < rows {
            let n = chunk.min(rows - first);
            let params = self.uniform(&[n_h as u32, n_kv as u32, past as u32, n as u32, kv_len as u32, scale.to_bits(), first as u32, mask]);
            self.dispatch(&pipeline, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, n.div_ceil(tq) as u32, 1));
            self.weigh(4.0 * n as f64 * kv_len as f64 * (n_h * head_dim) as f64);
            first += n;
        }
    }

    /// [`Self::attention_rows_f32_masked`] in runs of 256 positions a workgroup a (head, run, query), the runs then
    /// joined: any head's width, its out [`attention_runs_out_len`] long.
    #[allow(clippy::too_many_arguments)]
    /// [`ChainRecorder::attention`], its parts a KV head's query heads together with `group` where they can be
    /// ([`attention_group_for`]), else a workgroup a query head.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_by(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32, group: bool) {
        let runs = kv_len.saturating_sub(lo).div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= cap * 2 * n_kv * head_dim && kv_len <= cap && out.len >= n_h * head_dim + n_h * runs * (head_dim + 2),
            "chain: attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, kv_len as u32, lo as u32, runs as u32, scale.to_bits()]);
        let g = if n_kv > 0 && n_h % n_kv == 0 { n_h / n_kv } else { 0 };
        if group && attention_group_for(g, head_dim, self.gpu().limits.max_compute_workgroup_storage_size) {
            // a KV head's query heads together: its keys and values read once
            const NAMES: [&str; 9] = ["", "", "chain-attention-group-2", "chain-attention-group-3", "chain-attention-group-4", "chain-attention-group-5", "chain-attention-group-6", "chain-attention-group-7", "chain-attention-group-8"];
            let part = self.gpu().named_pipeline(NAMES[g], || attention_part_group(g));
            self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_kv as u32, runs as u32, 1));
        } else {
            // a head a multiple of 4 wide (at most 512): the vec4 kernel
            let part = if head_dim % 4 == 0 && head_dim <= 512 { self.gpu().named_pipeline("chain-attention-part4", || ATTENTION_PART4.to_string()) } else { self.named("chain-attention-part", ATTENTION_PART) };
            self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, 1));
        }
        let join = self.named("chain-attention-join", ATTENTION_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, 1, 1));
    }

    pub(crate) fn attention_rows_runs(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) {
        self.attention_rows_runs_by(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, full, head_dim % 4 == 0 && head_dim <= 512);
    }

    /// [`Self::attention_rows_runs`], its parts by [`ATTENTION_ROWS_PART4`] with `fours` (a head a multiple of 4 wide,
    /// at most 512), else a sum a load ([`ATTENTION_ROWS_PART`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_runs_by(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool, fours: bool) {
        // (a full attention's `past` its positions, whatever its queries)
        let kv_len = if full { past } else { past + rows };
        let runs = kv_len.div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= kv_len * 2 * n_kv * head_dim && q.len >= rows * n_h * head_dim && out.len >= rows * n_h * head_dim + rows * n_h * runs * (head_dim + 2),
            "chain: a prompt's attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, past as u32, window.unwrap_or(0) as u32, runs as u32, scale.to_bits(), rows as u32]);
        // (a full attention: every query's positions all `past` of them)
        let every = |body: &str| {
            let full = body.replace("    let hi = past + s + 1u;\n", "    let hi = past;\n");
            assert_ne!(full, body, "the parts' causal limit");
            full
        };
        let part = match (fours, full) {
            (true, true) => self.gpu().named_pipeline("chain-attention-rows-part4-full", || every(ATTENTION_ROWS_PART4)),
            (true, false) => self.gpu().named_pipeline("chain-attention-rows-part4", || ATTENTION_ROWS_PART4.to_string()),
            (false, true) => self.gpu().named_pipeline("chain-attention-rows-part-full", || format!("{HEAD}{}", every(ATTENTION_ROWS_PART))),
            (false, false) => self.named("chain-attention-rows-part", ATTENTION_ROWS_PART),
        };
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, rows as u32));
        let join = self.named("chain-attention-rows-join", ATTENTION_ROWS_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, rows as u32, 1));
        self.weigh(4.0 * rows as f64 * kv_len as f64 * (n_h * head_dim) as f64);
    }

    /// [`ChainRecorder::attention_rows`] on the tensor cores ([`ATTENTION_COOP`]): the queries and the cache's rows
    /// as f16 (the cache's each time, to the last query: a few microseconds a layer), the scores f16 into f32 and the
    /// softmax in f32, its weights f16. False (nothing recorded) where the device has no tensor cores, a window is
    /// kept, the head is not 64, 128 or 256 wide, or there are fewer than 16 rows.
    pub(crate) fn attention_rows_coop(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) -> bool {
        self.attention_rows_coop_masked(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale, false)
    }

    /// [`Self::attention_rows_coop`], with `full` every query over every position (no causal mask).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_rows_coop_masked(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32, full: bool) -> bool {
        let name = match (head_dim, full) {
            (64, false) => "chain-attention-coop-64",
            (128, false) => "chain-attention-coop-128",
            (256, false) => "chain-attention-coop-256",
            (64, true) => "chain-attention-coop-full-64",
            (128, true) => "chain-attention-coop-full-128",
            (256, true) => "chain-attention-coop-full-256",
            _ => return false,
        };
        if window.is_some() || rows < 16 || n_kv == 0 || n_h % n_kv != 0 || !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return false;
        }
        // (a full attention's `past` its positions, whatever its queries)
        let kv_len = if full { past } else { past + rows };
        let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
        let rp = rows.div_ceil(32) * 32;
        // the padding's rows are stored past the output (in a causal attention's scratch, at least as long: 16 rows or
        // more; a full one's out is as long as they need, `attention_rows_full_out_len`)
        assert!(
            kv.len >= kv_len * row && q.len >= rows * qs && out.len >= rp * qs && (full || out.len >= self.backend.attention_rows_out_len(rows, n_h, head_dim, kv_len)),
            "chain: a prompt's attention's buffers"
        );
        // (the keys 128 a block, their scores once; OAIY_ATTENTION_PASSES=2: twice, the first pass each query's largest
        // score and sum; OAIY_ATTENTION_NARROW: twice, the keys 32 a block)
        static KERNEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let kernel = *KERNEL.get_or_init(|| if std::env::var_os("OAIY_ATTENTION_NARROW").is_some() { 0 } else if std::env::var("OAIY_ATTENTION_PASSES").is_ok_and(|v| v == "2") { 2 } else { 1 });
        // (the one pass reads the cache's copy a fragment at a time)
        let (q16, kv16) = self.attention_f16(q, kv, rows, kv_len, qs, row, (kernel == 1).then_some((n_kv, head_dim)));
        let words = [n_h as u32, n_kv as u32, past as u32, rows as u32, kv_len as u32, scale.to_bits(), 0, 0];
        let pipeline = self.gpu().named_pipeline(name, || match (kernel, full) {
            (1, _) => attention_coop_one(head_dim, full),
            (2, _) => attention_coop_wide(head_dim, full),
            (_, true) => attention_coop_full(head_dim),
            (_, false) => attention_coop(head_dim),
        });
        self.dispatch_kept(&pipeline, buffer(&kv16), buffer(&q16), buffer(out), &words, (n_h as u32, (rows.div_ceil(32)) as u32, 1));
        self.att16 = Some((q16, kv16));
        self.weigh(4.0 * rows as f64 * kv_len as f64 * (n_h * head_dim) as f64);
        true
    }

    /// A prompt's queries (`rows` of `qs`) and its cache's rows (`kv_len` of `row`) as f16 for the tensor cores'
    /// attention, each padded to 32 (the copies one pair a recording, grown as it needs: each attention's converted
    /// as it runs; put back in `att16` once used).
    fn attention_f16(&mut self, q: &DeviceVec, kv: &DeviceVec, rows: usize, kv_len: usize, qs: usize, row: usize, tiled: Option<(usize, usize)>) -> (DeviceVec, DeviceVec) {
        // (the keys to a block of the wide kernel's: 128)
        let (rp, kp) = (rows.div_ceil(32) * 32, kv_len.div_ceil(128) * 128);
        let (q16, kv16) = match self.att16.take() {
            Some((a, b)) if a.len >= rp * qs / 2 && b.len >= kp * row / 2 => (a, b),
            _ => (self.scratch(rp * qs / 2), self.scratch(kp * row / 2)),
        };
        let conv = self.gpu().named_pipeline("chain-x-f16", || crate::shaders::X_F16.to_string());
        let d = self.gpu().dummy().clone();
        for (src, dst, width, n, padded) in [(q, &q16, qs, rows, rp), (kv, &kv16, row, kv_len, kp)] {
            let groups = ((padded * width / 2) as u32).div_ceil(256);
            match tiled.filter(|_| std::ptr::eq(dst, &kv16)) {
                // the cache's a fragment at a time (`tiled`: its KV heads and their width), the one-pass kernels'
                Some((n_kv, head_dim)) => {
                    let tile = self.gpu().named_pipeline("chain-kv-f16-tiled", || crate::shaders::KV_F16_TILED.to_string());
                    self.dispatch_kept(&tile, &d, buffer(src), buffer(dst), &[n_kv as u32, head_dim as u32, n as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
                }
                None => self.dispatch_kept(&conv, &d, buffer(src), buffer(dst), &[width as u32, n as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1)),
            }
        }
        (q16, kv16)
    }

    /// [`ChainRecorder::qsa_attention`] in f32 ([`QSA_ATTENTION_PART`]): a query's entries in runs of 256, a workgroup
    /// a (head, run, query), the runs joined.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_attention_f32(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) {
        let runs = (keep * ratio + ratio).div_ceil(256);
        assert!(
            head_dim % 4 == 0 && head_dim <= 512 && q.len >= rows * n_h * head_dim && kv.len >= (first + rows) * 2 * n_kv * head_dim && list.len >= rows * keep && out.len >= self.backend.qsa_attention_out_len(rows, n_h, head_dim, keep, ratio),
            "chain: QSA's attention's buffers"
        );
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let words = [n_h as u32, n_kv as u32, head_dim as u32, first as u32, ratio as u32, runs as u32, scale.to_bits(), keep as u32];
        self.dispatch_wide("chain-qsa-attention-part", QSA_ATTENTION_PART, [buffer(kv), buffer(q), buffer(list), &dd, &dd, &dd, buffer(out), &drw], &words, (n_h as u32, runs as u32, rows as u32));
        // the runs joined as a prompt's are (an empty run's sum 0 adds nothing)
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, first as u32, 0, runs as u32, scale.to_bits(), rows as u32]);
        let join = self.named("chain-attention-rows-join", ATTENTION_ROWS_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, rows as u32, 1));
    }

    /// [`ChainRecorder::qsa_attention`] of a prompt's rows on the tensor cores ([`attention_coop_masked`]): the kept
    /// blocks as each query's bitmask ([`QSA_MASK`]), then every position up to the query's on the tensor cores but
    /// those of a block it did not keep (the dense span's work past it, which the tensor cores still do the sooner:
    /// 512 of some 640 blocks kept at 2,560 positions). False (nothing recorded) as [`Self::attention_rows_coop`] is.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qsa_attention_coop(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) -> bool {
        let name = match head_dim {
            64 => "chain-qsa-attention-coop-64",
            128 => "chain-qsa-attention-coop-128",
            256 => "chain-qsa-attention-coop-256",
            _ => return false,
        };
        if rows < 16 || ratio == 0 || n_kv == 0 || n_h % n_kv != 0 || !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return false;
        }
        let kv_len = first + rows;
        let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
        // the padding's rows go past the output, in its scratch (at least as long as they: 16 rows or more)
        assert!(out.len >= rows.div_ceil(32) * 32 * qs, "chain: QSA's attention's buffers");
        let mw = (kv_len / ratio).div_ceil(32).max(1);
        let mask = self.scratch(rows * mw);
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let words = (rows * mw) as u32;
        let groups = words.div_ceil(256);
        self.dispatch_wide("chain-qsa-mask", QSA_MASK, [buffer(list), &d, &d, &d, &d, &d, buffer(&mask), &drw], &[rows as u32, keep as u32, mw as u32, first as u32, ratio as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        // (the keys 128 a block, their scores once; OAIY_ATTENTION_PASSES=2 or OAIY_ATTENTION_NARROW: twice, 32 a block)
        static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let old = *OLD.get_or_init(|| std::env::var_os("OAIY_ATTENTION_NARROW").is_some() || std::env::var("OAIY_ATTENTION_PASSES").is_ok_and(|v| v == "2"));
        let (q16, kv16) = self.attention_f16(q, kv, rows, kv_len, qs, row, (!old).then_some((n_kv, head_dim)));
        let src = if old { attention_coop_masked(head_dim) } else { attention_coop_one_masked(head_dim) };
        let words = [n_h as u32, n_kv as u32, first as u32, rows as u32, kv_len as u32, scale.to_bits(), ratio as u32, mw as u32];
        self.dispatch_wide(name, &src, [buffer(&kv16), buffer(&q16), buffer(&mask), &d, &d, &d, buffer(out), &drw], &words, (n_h as u32, rows.div_ceil(32) as u32, 1));
        self.att16 = Some((q16, kv16));
        true
    }

    /// [`ChainRecorder::delta_net`] for a prompt's rows ([`DELTA_NET_PREP`], [`DELTA_NET_SCAN`], [`DELTA_NET_NORM`]):
    /// every token's q, k and gates at once, then the recurrence a thread a state's row, then every token's norm.
    #[allow(clippy::too_many_arguments)]
    fn delta_net_rows(&mut self, conv: &DeviceVec, z: &DeviceVec, beta_alpha: &DeviceVec, ssm_a: &DeviceVec, dt_bias: &DeviceVec, norm: &DeviceVec, state: &DeviceVec, out: &DeviceVec, d: &DeltaNet, words: &[u32]) {
        let names: [&'static str; 3] = match d.k_dim {
            32 => ["chain-delta-net-prep-32", "chain-delta-net-scan-32", "chain-delta-net-norm-32"],
            64 => ["chain-delta-net-prep-64", "chain-delta-net-scan-64", "chain-delta-net-norm-64"],
            128 => ["chain-delta-net-prep-128", "chain-delta-net-scan-128", "chain-delta-net-norm-128"],
            other => panic!("chain: a delta net of heads of {other}"),
        };
        let dk = d.k_dim;
        let qk = self.scratch(d.rows * d.k_heads * 2 * dk + d.rows * d.v_heads * 2);
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        // a workgroup a warp's 32 rows (more of them than of heads: 0.32 ms a layer of Qwen3.8 27B's for 512 tokens,
        // as a head's 128; 64 0.44)
        let r = 32;
        let size = |s: &str| s.replace("DK_VALUE", &dk.to_string());
        let rows = d.rows as u32;
        self.dispatch_wide(names[0], &size(DELTA_NET_PREP), [buffer(conv), &dd, buffer(beta_alpha), buffer(ssm_a), buffer(dt_bias), &dd, buffer(&qk), &drw], words, (d.k_heads as u32, rows, 1));
        let groups = (d.v_heads * dk / r) as u32;
        self.dispatch_wide(names[1], &delta_net_scan(dk, r), [buffer(&qk), buffer(conv), buffer(&qk), &dd, &dd, &dd, buffer(state), buffer(out)], words, (groups, 1, 1));
        self.dispatch_wide(names[2], &size(DELTA_NET_NORM), [&dd, buffer(z), &dd, &dd, &dd, buffer(norm), &drw, buffer(out)], words, (d.v_heads as u32, rows, 1));
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] through the f32 kernels alone: the decode kernel for one row, the
    /// one-row kernel for a few, the tiled one for a prompt.
    pub(crate) fn matmul_rows_f32(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        assert!(m > 0 && x.len >= m * k && y.len >= m * n, "chain: matmul [{n}, {k}] of {m} rows from {} into {}", x.len, y.len);
        let pipeline = self.gpu().pipeline(q.dtype, m).expect("uploaded weights have a pipeline");
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0];
            let groups = crate::shaders::grid(q.dtype, m, *rows);
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, groups);
        }
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for a prompt's rows on the tensor cores
    /// ([`crate::shaders::coop_tiled`]: f16 weights and tokens into f32 sums). False where the device has no cooperative
    /// matrices or the type no such kernel.
    pub(crate) fn matmul_rows_coop(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        self.matmul_rows_coop_split(w, x, y, m, None)
    }

    /// [`Self::matmul_rows_coop`], split along k as given (else as [`crate::shaders::coop_splits`] chooses).
    pub(crate) fn matmul_rows_coop_split(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        use ggml_quants::GgmlType as T;
        let name = match q.dtype {
            T::Q3_K => "chain-coop-Q3_K",
            T::Q4_K => "chain-coop-Q4_K",
            T::Q5_K => "chain-coop-Q5_K",
            T::Q6_K => "chain-coop-Q6_K",
            T::Q8_0 => "chain-coop-Q8_0",
            _ => return false,
        };
        if !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) || k % 256 != 0 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: a tensor-core matmul [{n}, {k}] of {m} rows");
        let x16 = self.x16_tiled(x, m, k);
        let tile = crate::shaders::COOP_TILE;
        let dtype = q.dtype;
        let pipeline = self.gpu().named_pipeline(name, || crate::shaders::coop_tiled(dtype).expect("a K-quant's tensor-core kernel"));
        let tiles = q.chunks.iter().map(|(_, _, rows)| rows.div_ceil(tile)).max().unwrap_or(1) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, m, n, y, split);
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, splits, 0];
            self.dispatch_kept(&pipeline, chunk, buffer(&x16), &out, &words, (rows.div_ceil(tile), (m as u32).div_ceil(tile), splits));
        }
        self.coop_sum(parts, m, n, y, splits);
        true
    }

    /// The tokens' rows `x` (`m` of `k`) as [`crate::shaders::X_F16_TILED`] gives them (f16, padded to the tile and to
    /// a step of 32): once for every tensor-core matmul that reads them until something writes `x`.
    fn x16_tiled(&mut self, x: &DeviceVec, m: usize, k: usize) -> DeviceVec {
        let tile = crate::shaders::COOP_TILE;
        let padded = (m as u32).div_ceil(tile) as usize * tile as usize;
        let xb = buffer(x).clone();
        if let Some((.., v)) = self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            return v.clone();
        }
        let words = crate::shaders::x_f16_tiled_words(m, k);
        let v = self.scratch(words);
        let conv = self.gpu().named_pipeline("chain-x-f16-tiled", || crate::shaders::X_F16_TILED.to_string());
        let groups = (words as u32).div_ceil(256);
        let d = self.gpu().dummy().clone();
        self.dispatch_kept(&conv, &d, buffer(x), buffer(&v), &[k as u32, m as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.x16.push((xb, m, k, v.clone()));
        v
    }

    /// A tensor-core matmul's splits along k (as given, else as [`crate::shaders::coop_splits`] chooses for `tiles`
    /// workgroups), where its sums go (`y`, or a part of scratch a split, added into `y` after), and the parts.
    fn coop_parts(&mut self, tiles: u32, k: usize, m: usize, n: usize, y: &DeviceVec, split: Option<u32>) -> (u32, wgpu::Buffer, Option<DeviceVec>) {
        // a matmul of too few tiles to fill the GPU's last wave split along k: each split's sums into a part of
        // scratch, then the parts added into y
        let units = self.gpu().coop_units();
        let steps = k.div_ceil(32) as u32;
        let splits = split.unwrap_or_else(|| crate::shaders::coop_splits(tiles, units, steps));
        // (one buffer of parts a recording, grown as it needs: its matmuls run in turn)
        let parts = if splits > 1 {
            let len = splits as usize * m * n;
            let v = match self.parts.take() {
                Some(v) if v.len >= len => v,
                _ => self.scratch(len),
            };
            self.parts = Some(v.clone());
            Some(v)
        } else {
            None
        };
        let out = parts.as_ref().map_or(buffer(y), buffer).clone();
        (splits, out, parts)
    }

    /// A split tensor-core matmul's parts added into `y`.
    fn coop_sum(&mut self, parts: Option<DeviceVec>, m: usize, n: usize, y: &DeviceVec, splits: u32) {
        if let Some(parts) = parts {
            let sum = self.named("chain-coop-sum", COOP_SUM);
            let groups = ((m * n) as u32).div_ceil(256);
            let d = self.gpu().dummy().clone();
            self.dispatch_kept(&sum, &d, buffer(&parts), buffer(y), &[(m * n) as u32, splits], (groups.min(65535), groups.div_ceil(65535), 1));
        }
    }

    /// [`ChainRecorder::conv_rows`] and [`ChainRecorder::conv3d_rows`]: `taps` 1, 9 (3x3) or 27 (3x3x3), `frames` of
    /// `h` rows of `wd` (one frame for a picture), `x`'s voxels `xs` values apart (its first `cin` each).
    #[allow(clippy::too_many_arguments)]
    fn conv_taps(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, taps: usize, x: &DeviceVec, xs: usize, frames: usize, h: usize, wd: usize, y: &DeviceVec) {
        let cp = cin.div_ceil(32) * 32;
        let m = frames * h * wd;
        assert!(m > 0 && xs >= cin && w.len * 2 >= cout * taps * cp && b.len >= cout && x.len >= (m - 1) * xs + cin && y.len >= m * cout, "chain: a convolution of {taps} taps of {frames}x{h}x{wd} voxels, {cin} channels ({xs} apart) to {cout}");
        assert!(cp < 1 << 16 && xs < 1 << 14, "chain: a convolution's {cin} channels {xs} apart");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        if !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            // no tensor cores: the f32 tiled kernel, the voxels' tiles in chunks of bounded work (the inputs as they
            // are, no f16 copy or range to keep)
            let tiles = m.div_ceil(64);
            let per_tile = 2.0 * 64.0 * (cout * cin * taps) as f64;
            let chunk = ((CONV_DISPATCH_FLOPS / per_tile) as usize).clamp(1, 65535);
            let mut first = 0;
            while first < tiles {
                let n = chunk.min(tiles - first);
                let words = [cout as u32, cin as u32, m as u32, xs as u32, taps as u32, wd as u32, h as u32, first as u32];
                self.dispatch_wide("chain-conv-f32-tiled", CONV_F32_TILED, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &words, ((cout as u32).div_ceil(64), n as u32, 1));
                self.weigh(per_tile * n as f64);
                first += n;
            }
            return;
        }
        // the input as f16, each pixel's channels padded to 32's: once for every convolution that reads it until
        // something writes `x` (kept with the matmuls' tiled copies, a width of its own)
        let key = cp | xs << 16 | 1 << 31;
        let xb = buffer(x).clone();
        let x16 = match self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == key) {
            Some((.., v)) => v.clone(),
            None => {
                // `x`'s scale for f16 (its largest within 16,384) set on the device, then the copy scaled; the scale kept
                // beside the copy (a width of its own) for the sums' way back
                let range = self.scratch(4);
                // (over every value of the pixels: a strided input's others too, its range no smaller)
                let len = ((m - 1) * xs + cin) as u32;
                self.dispatch_wide("chain-f16-range-clear", F16_RANGE_CLEAR, [&d, &d, &d, &d, &d, &d, buffer(&range), &drw], &[0], (1, 1, 1));
                self.dispatch_wide("chain-f16-range-max", F16_RANGE_MAX, [buffer(x), &d, &d, &d, &d, &d, buffer(&range), &drw], &[len], grid(len.div_ceil(256)));
                self.dispatch_wide("chain-f16-range-set", F16_RANGE_SET, [&d, &d, &d, &d, &d, &d, buffer(&range), &drw], &[0], (1, 1, 1));
                let v = self.scratch(m * cp / 2);
                let conv = self.gpu().named_pipeline("chain-x-f16-padded", || X_F16_PADDED.to_string());
                let words = (m * cp / 2) as u32;
                self.dispatch_kept(&conv, buffer(&range), buffer(x), buffer(&v), &[cin as u32, cp as u32, m as u32, xs as u32], grid(words.div_ceil(256)));
                self.x16.push((xb.clone(), m, key, v.clone()));
                self.x16.push((xb, m, key ^ (3 << 30), range));
                v
            }
        };
        let xb = buffer(x).clone();
        let range = self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == key ^ (3 << 30)).map(|(.., v)| v.clone()).expect("a convolution's input's scale beside its copy");
        let tile = crate::shaders::COOP_TILE;
        let pipeline = match taps {
            27 => self.gpu().named_pipeline("chain-coop-conv3d", || crate::shaders::coop_conv(27)),
            49 => self.gpu().named_pipeline("chain-coop-conv7x7", || crate::shaders::coop_conv(49)),
            9 => self.gpu().named_pipeline("chain-coop-conv3x3", || crate::shaders::coop_conv(9)),
            _ => self.gpu().named_pipeline("chain-coop-conv1x1", || crate::shaders::coop_conv(1)),
        };
        let kk = taps * cp;
        let tiles = (cout as u32).div_ceil(tile) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, kk, m, cout, y, None);
        let words = [kk as u32, cout as u32, m as u32, 0, cout as u32, wd as u32, splits, h as u32];
        self.dispatch_kept(&pipeline, buffer(w), buffer(&x16), &out, &words, ((cout as u32).div_ceil(tile), (m as u32).div_ceil(tile), splits));
        self.coop_sum(parts, m, cout, y, splits);
        // the sums back to x's range, and the bias
        self.dispatch_wide("chain-unscale-bias-rows", UNSCALE_BIAS_ROWS, [buffer(b), buffer(&range), &d, &d, &d, &d, buffer(y), &drw], &[cout as u32, m as u32], grid(((m * cout) as u32).div_ceil(256)));
        self.weigh(2.0 * (m * cout) as f64 * (cin * taps) as f64);
    }

    /// `y[r] = W x[r]` for a prompt's rows of f16 weights (`[n, k]` two to a word) through the f32 tiled kernel (the
    /// weights read as f32), split along k where its tiles are few.
    pub(crate) fn matmul_f16_tiled(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        let tiles = n.div_ceil(64) * rows.div_ceil(64);
        let want = 1024usize.div_ceil(tiles).min(k / 256).max(1);
        let kc = k.div_ceil(want).div_ceil(16) * 16;
        let splits = k.div_ceil(kc);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let grid = (n.div_ceil(64) as u32, rows.div_ceil(64) as u32, splits as u32);
        let words = [n as u32, k as u32, rows as u32, kc as u32];
        let tiled = MATMUL_F32_TILED.replace("var<storage, read> w: array<f32>;", "var<storage, read> w: array<u32>;\nfn wv(e: u32) -> f32 {\n    let pr = unpack2x16float(w[e / 2u]);\n    return select(pr.x, pr.y, (e & 1u) == 1u);\n}").replace("u = w[(o0 + rr) * k + gk];", "u = wv((o0 + rr) * k + gk);");
        if splits == 1 {
            self.dispatch_wide("chain-matmul-f16-tiled", &tiled, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(y), &drw], &words, grid);
        } else {
            let part = self.scratch(splits * rows * n);
            self.dispatch_wide("chain-matmul-f16-tiled", &tiled, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(&part), &drw], &words, grid);
            let len = (rows * n) as u32;
            self.dispatch_wide("chain-sum-splits", SUM_SPLITS, [buffer(&part), &d, &d, &d, &d, &d, buffer(y), &drw], &[len, splits as u32], (len.div_ceil(256).min(65535), len.div_ceil(256 * 65535), 1));
            // (its parts read: spare for the next split's, where each had its own to the recording's end, a sound
            // step's 360 of 18 MB without tensor cores)
            self.spare.push((buffer(&part).size(), buffer(&part).clone()));
        }
    }

    /// `y[r] = W x[r]` for a prompt's rows of f16 weights (`[n, k]` two to a word, `k` of 4) on the tensor cores
    /// ([`crate::shaders::coop_tiled_f16`]), split along k as given (else as chosen). False where the device has no
    /// cooperative matrices.
    pub(crate) fn matmul_f16_coop(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) -> bool {
        if !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) || k % 4 != 0 || m.div_ceil(crate::shaders::COOP_TILE as usize) > 65535 {
            return false;
        }
        let x16 = self.x16_tiled(x, m, k);
        let tile = crate::shaders::COOP_TILE;
        let pipeline = self.gpu().named_pipeline("chain-coop-f16", crate::shaders::coop_tiled_f16);
        let tiles = (n as u32).div_ceil(tile) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, m, n, y, split);
        let words = [k as u32, n as u32, m as u32, 0, n as u32, 0, splits, 0];
        self.dispatch_kept(&pipeline, buffer(w), buffer(&x16), &out, &words, ((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), splits));
        self.coop_sum(parts, m, n, y, splits);
        true
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for a prompt's rows, from `x`'s rows as int8 (quantized once,
    /// as for [`Self::matmul_rows_q8`]) through [`crate::shaders::tiled_q8`]. False for a type without that kernel.
    pub(crate) fn matmul_rows_tq8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::Q3_K || k % 256 != 0 {
            return false;
        }
        let (len, xs_at) = crate::shaders::q8_len(m, k);
        assert!(x.len >= m * k && y.len >= m * n, "chain: an int8 tiled matmul [{n}, {k}] of {m} rows");
        let xb = buffer(x).clone();
        let xq = match self.q8.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            Some((.., xq)) => xq.clone(),
            None => {
                let xq = self.scratch(len);
                let quant = self.gpu().named_pipeline("chain-q8-quantize", || crate::shaders::QUANT_Q8.to_string());
                let blocks = (m * k / 32) as u32;
                let groups = blocks.div_ceil(256);
                let d = self.gpu().dummy().clone();
                self.dispatch_kept(&quant, &d, buffer(x), buffer(&xq), &[k as u32, m as u32, xs_at as u32], (groups.min(65535), groups.div_ceil(65535), 1));
                self.q8.push((xb, m, k, xq.clone()));
                xq
            }
        };
        let pipeline = self.gpu().named_pipeline("chain-tq8-Q3_K", || crate::shaders::tiled_q8(ggml_quants::GgmlType::Q3_K).expect("Q3_K's int8 tiled kernel"));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = (rows.div_ceil(crate::shaders::TQ8_ROWS), (m as u32).div_ceil(crate::shaders::TQ8_TOKENS), 1);
            self.dispatch_kept(&pipeline, chunk, buffer(&xq), buffer(y), &words, groups);
        }
        true
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for several rows, from `x`'s rows as int8 (quantized once,
    /// [`crate::shaders::QUANT_Q8`], for every matmul that reads them until something writes `x`): the K-quants' int8
    /// kernels, a check of drafts' rows in about 1.3 of a step's time where the f32 kernels take 1.7. False for a type
    /// without one (nothing recorded).
    fn matmul_rows_q8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        let block = if q.dtype == ggml_quants::GgmlType::Q4_0 { 32 } else { 256 };
        if !matches!(q.dtype, ggml_quants::GgmlType::Q3_K | ggml_quants::GgmlType::Q4_K | ggml_quants::GgmlType::Q5_K | ggml_quants::GgmlType::Q6_K | ggml_quants::GgmlType::Q4_0) || k % block != 0 {
            return false;
        }
        let (len, xs_at) = crate::shaders::q8_len(m, k);
        assert!(x.len >= m * k && y.len >= m * n, "chain: an int8 matmul [{n}, {k}] of {m} rows");
        let xb = buffer(x).clone();
        let xq = match self.q8.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            Some((.., xq)) => xq.clone(),
            None => {
                let xq = self.scratch(len);
                let quant = self.gpu().named_pipeline("chain-q8-quantize", || crate::shaders::QUANT_Q8.to_string());
                let blocks = (m * k / 32) as u32;
                let groups = blocks.div_ceil(256);
                let d = self.gpu().dummy().clone();
                self.dispatch_kept(&quant, &d, buffer(x), buffer(&xq), &[k as u32, m as u32, xs_at as u32], (groups.min(65535), groups.div_ceil(65535), 1));
                self.q8.push((xb, m, k, xq.clone()));
                xq
            }
        };
        let mr = if m == 1 { 1 } else { crate::shaders::MULTI_ROWS };
        let name = match (q.dtype, mr == 1) {
            (ggml_quants::GgmlType::Q3_K, true) => "chain-q8-Q3_K-decode",
            (ggml_quants::GgmlType::Q3_K, false) => "chain-q8-Q3_K-multi",
            (ggml_quants::GgmlType::Q4_K, true) => "chain-q8-Q4_K-decode",
            (ggml_quants::GgmlType::Q4_K, false) => "chain-q8-Q4_K-multi",
            (ggml_quants::GgmlType::Q5_K, true) => "chain-q8-Q5_K-decode",
            (ggml_quants::GgmlType::Q5_K, false) => "chain-q8-Q5_K-multi",
            (ggml_quants::GgmlType::Q4_0, true) => "chain-q8-Q4_0-decode",
            (ggml_quants::GgmlType::Q4_0, false) => "chain-q8-Q4_0-multi",
            (_, true) => "chain-q8-Q6_K-decode",
            (_, false) => "chain-q8-Q6_K-multi",
        };
        let pipeline = self.gpu().named_pipeline(name, || crate::shaders::rb_kernel_q8(q.dtype, crate::shaders::rb_rows(q.dtype, mr), mr).expect("a K-quant's int8 kernel"));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = crate::shaders::grid(q.dtype, m, *rows);
            self.dispatch_kept(&pipeline, chunk, buffer(&xq), buffer(y), &words, groups);
        }
        true
    }

    pub(crate) fn named(&self, name: &'static str, body: &'static str) -> Arc<wgpu::ComputePipeline> {
        self.gpu().named_pipeline(name, || format!("{HEAD}{body}"))
    }

    /// A dispatch of an eight-buffer kernel (`Gpu::wide_layout`), its bind group kept as `dispatch_kept`'s.
    pub(crate) fn dispatch_wide(&mut self, name: &'static str, body: &str, bufs: [&wgpu::Buffer; 8], words: &[u32], groups: (u32, u32, u32)) {
        self.wrote(bufs[6]);
        self.wrote(bufs[7]);
        let pipeline = self.gpu().named_pipeline_wide(name, || body.to_string());
        let key: WideKey = (Arc::as_ptr(&pipeline) as usize, bufs.map(|b| b.clone()), words8(words));
        let kept = if self.keep { self.gpu().chain_groups_wide.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() } else { None };
        let group = match kept {
            Some(group) => group,
            None => {
                let params = self.uniform(words);
                let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
                entries.push(wgpu::BindGroupEntry { binding: 8, resource: params.as_entire_binding() });
                let group = self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("oaiy-chain-wide"), layout: &self.gpu().wide_layout().0, entries: &entries });
                if self.keep {
                    let mut groups = self.gpu().chain_groups_wide.lock().unwrap_or_else(|p| p.into_inner());
                    if groups.len() >= KEEP_GROUPS {
                        groups.clear();
                    }
                    groups.insert(key, group.clone());
                }
                group
            }
        };
        self.push((pipeline, group, groups));
    }
}

impl ChainRecorder for Recorder<'_> {
    fn matmul_rows(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) {
        // several rows (a check of drafts, a short chunk) from int8 activations, where the type has kernels for them
        // (OAIY_NO_Q8: the f32 ones)
        static Q8: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let q8 = *Q8.get_or_init(|| std::env::var_os("OAIY_NO_Q8").is_none());
        // (a prompt's rows: the tensor cores where the device has them (f16 into f32), else the int8 tiled kernel,
        // llama.cpp's MMQ's arithmetic, where the type has one)
        let done = ((2..=crate::shaders::MULTI_MAX).contains(&m) && q8 && self.matmul_rows_q8(w, x, y, m))
            || (m > crate::shaders::MULTI_MAX && self.matmul_rows_coop(w, x, y, m))
            || (m > crate::shaders::MULTI_MAX && q8 && self.matmul_rows_tq8(w, x, y, m));
        if !done {
            self.matmul_rows_f32(w, x, y, m);
        }
        self.weigh(2.0 * m as f64 * w.shape().iter().product::<usize>() as f64);
    }

    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32) {
        let pipeline = self.named("chain-rmsnorm", RMSNORM);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[x.len as u32, eps.to_bits()], (1, 1, 1));
    }

    fn rmsnorm_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        assert!(rows > 0 && x.len % rows == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: rmsnorm of {rows} rows of {}", x.len);
        let n = x.len / rows;
        // rows a multiple of 4 long (a model's width, a head's): vec4 loads
        let pipeline = if n % 4 == 0 { self.gpu().named_pipeline("chain-rmsnorm-rows4", || RMSNORM_ROWS4.to_string()) } else { self.named("chain-rmsnorm-rows", RMSNORM_ROWS) };
        let r = rows as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), 0, r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn rmsnorm_heads_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, heads: usize, eps: f32) {
        let r = rows * heads;
        assert!(r > 0 && x.len % r == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: a multi-head rmsnorm of {rows} rows of {heads} heads ({})", x.len);
        let n = x.len / r;
        // (the norms' kernels take the weight's rows in turn: row r of x by w's row r % heads)
        let pipeline = if n % 4 == 0 { self.gpu().named_pipeline("chain-rmsnorm-rows4", || RMSNORM_ROWS4.to_string()) } else { self.named("chain-rmsnorm-rows", RMSNORM_ROWS) };
        let rr = r as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), heads as u32, rr], (rr.min(65535), rr.div_ceil(65535), 1));
    }

    fn rmsnorm_silu_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        assert!(rows > 0 && x.len % rows == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: rmsnorm and SiLU of {rows} rows of {}", x.len);
        let n = x.len / rows;
        let silu = |body: &str| {
            let at = "y4[at + i] = x4[at + i] * inv * w4[wat + i];";
            assert_eq!(body.matches(at).count(), 1, "the norm's store");
            body.replace(at, "let v = x4[at + i] * inv * w4[wat + i];\n        y4[at + i] = v / (vec4<f32>(1.0) + exp(-v));")
        };
        assert!(n % 4 == 0, "chain: rmsnorm and SiLU of rows {n} long (a multiple of 4)");
        let pipeline = self.gpu().named_pipeline("chain-rmsnorm-silu-rows4", || silu(RMSNORM_ROWS4));
        let r = rows as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), 0, r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn add_rmsnorm_rows(&mut self, x: &DeviceVec, y: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        let n = x.len / rows.max(1);
        if rows == 0 || n % 4 != 0 || n * rows != x.len || y.len < x.len || w.len < n || out.len < x.len || Arc::ptr_eq(&x.inner, &out.inner) {
            self.add(x, y);
            self.rmsnorm_rows(x, w, out, rows, eps);
            return;
        }
        let d = self.gpu().dummy().clone();
        let r = rows as u32;
        self.dispatch_wide("chain-add-rmsnorm-rows4", ADD_RMSNORM_ROWS4, [buffer(y), buffer(w), &d, &d, &d, &d, buffer(x), buffer(out)], &[n as u32, eps.to_bits(), r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec) {
        let pipeline = self.named("chain-add", ADD);
        self.dispatch_kept(&pipeline, buffer(y), buffer(y), buffer(acc), &[acc.len as u32], grid((acc.len as u32).div_ceil(256)));
    }

    fn silu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        let ff = out.len / rows;
        assert!(rows > 0 && out.len == rows * ff && fused.len >= 2 * out.len, "chain: SwiGLU of {rows} rows");
        let pipeline = self.named("chain-silu-mul-split", SILU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[ff as u32, rows as u32], grid((out.len as u32).div_ceil(256)));
    }

    fn gelu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        let ff = out.len / rows;
        assert!(rows > 0 && out.len == rows * ff && fused.len >= 2 * out.len, "chain: GeGLU of {rows} rows");
        let pipeline = self.named("chain-gelu-mul-split", GELU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[ff as u32, rows as u32], grid((out.len as u32).div_ceil(256)));
    }

    fn keep_groups(&mut self, keep: bool) {
        self.keep = keep;
    }

    fn hold(&mut self) {
        self.hold = true;
    }

    fn flush(&mut self) {
        // let go: what is held, then what is left, then the reads so far copied out after them
        self.hold = false;
        let held = std::mem::take(&mut self.held);
        if !held.is_empty() {
            self.gpu().submit_piece(held);
        }
        for list in std::mem::take(&mut self.lists) {
            self.gpu().feed(list);
        }
        self.submit_piece();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        for (from, offset, staging, len) in &self.reads[self.copied..] {
            if *len > 0 {
                enc.copy_buffer_to_buffer(from, (*offset * 4) as u64, staging, 0, (*len * 4) as u64);
            }
        }
        self.copied = self.reads.len();
        if let Some(staging) = self.resolve_pieces(&mut enc) {
            self.stamped = Some(staging);
        }
        self.flushed = Some(self.gpu().submit_after(vec![enc.finish()]));
    }

    fn exl3_rows(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.exl3_rows_of(w, x, None, y, rows);
    }

    fn rmsnorm_streams(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, eps: f32) {
        let n = x.len / (rows * streams).max(1);
        assert!(rows > 0 && streams > 0 && n * rows * streams == x.len && w.len >= streams * n && out.len >= x.len, "chain: a norm of {rows} rows of {streams} streams");
        let pipeline = self.named("chain-rmsnorm-rows", RMSNORM_ROWS);
        let r = (rows * streams) as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), streams as u32, r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn hc_gates(&mut self, t: &DeviceVec, post: &DeviceVec, rows: usize, rank: usize, writes: usize, streams: usize) {
        assert!(t.len >= rows * (rank + writes) && post.len >= rows * writes.max(1), "chain: a hyper-connection's gates");
        let len = (rows * (rank + writes)) as u32;
        // nothing read: a dummy at the read bindings (a buffer written may not be bound as read too)
        let d = self.gpu().dummy().clone();
        self.dispatch_wide("chain-hc-gates", HC_GATES, [&d, &d, &d, &d, &d, &d, buffer(t), buffer(post)], &[rank as u32, writes as u32, streams as u32, rows as u32], (len.div_ceil(256), 1, 1));
    }

    fn hc_mix(&mut self, logits: &DeviceVec, normed: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, d: usize) {
        assert!(logits.len >= rows * streams * d && normed.len >= rows * streams * d && out.len >= rows * d, "chain: a hyper-connection's mix");
        let pipeline = self.named("chain-hc-mix", HC_MIX);
        self.dispatch_kept(&pipeline, buffer(logits), buffer(normed), buffer(out), &[d as u32, streams as u32, rows as u32], (((rows * d) as u32).div_ceil(256), 1, 1));
    }

    fn stream_apply(&mut self, x: &DeviceVec, y: &DeviceVec, post: &DeviceVec, rows: usize, streams: usize, d: usize) {
        assert!(x.len >= rows * streams * d && y.len >= rows * d && post.len >= rows * streams, "chain: a hyper-connection's write-back");
        let pipeline = self.named("chain-stream-apply", STREAM_APPLY);
        self.dispatch_kept(&pipeline, buffer(post), buffer(y), buffer(x), &[d as u32, streams as u32, rows as u32], (((rows * streams * d) as u32).div_ceil(256), 1, 1));
    }

    fn moe_rows(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]) {
        let g = experts.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()).expect("experts this adapter holds as groups");
        g.record(self, x, out, assign);
    }

    fn moe_routed(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize) -> bool {
        let Some(g) = experts.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()) else { return false };
        g.record_routed(self, x, out, logits, top_k, rows, None)
    }

    fn moe_routed_into(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, streams_x: &DeviceVec, post: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, streams: usize) -> bool {
        let Some(g) = experts.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()) else { return false };
        // no vector for the sums: they go into the streams
        let none = self.gpu().dummy_rw().clone();
        let out = DeviceVec { len: 0, inner: Arc::new(none) };
        g.record_routed(self, x, &out, logits, top_k, rows, Some((streams_x, post, streams)))
    }

    fn axpy_at(&mut self, acc: &DeviceVec, y: &DeviceVec, weights: &DeviceVec, at: usize, len: usize) {
        assert!(acc.len >= len && y.len >= len && weights.len > at, "chain: a weighted term of {len}");
        let pipeline = self.named("chain-axpy-at", AXPY_AT);
        self.dispatch_kept(&pipeline, buffer(weights), buffer(y), buffer(acc), &[len as u32, at as u32], grid((len as u32).div_ceil(256)));
    }

    fn copy_cols(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, width: usize, stride: usize, at: usize) {
        assert!(rows > 0 && at + width <= stride && src.len >= rows * stride && dst.len >= rows * width, "chain: {rows} rows' columns {at}..{} of {stride}", at + width);
        let pipeline = self.named("chain-copy-cols", COPY_COLS);
        self.dispatch_kept(&pipeline, buffer(src), buffer(src), buffer(dst), &[width as u32, rows as u32, stride as u32, at as u32], grid(((rows * width) as u32).div_ceil(256)));
    }

    fn rope_partial_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, rot: usize, table: &DeviceVec) {
        assert!(rot > 0 && rot % 2 == 0 && rot <= head_dim && x.len >= rows * heads * head_dim && table.len >= rows * rot, "chain: RoPE of {rot} of {head_dim}");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * rot / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, 1, rows as u32, rot as u32], grid(pairs.div_ceil(256)));
    }

    fn matmul_f16_rows_f32(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        if rows <= 8 {
            self.matmul_f16_rows(w, n, k, x, y, rows);
        } else {
            assert!(k % 2 == 0 && w.len * 2 >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows.div_ceil(64) <= 65535, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
            self.matmul_f16_tiled(w, n, k, x, y, rows);
            self.weigh(2.0 * (rows * n) as f64 * k as f64);
        }
    }

    fn norm_mod_rows_clean(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, scale_at: usize, shift_at: Option<usize>, norm: ggml_rs::RowNorm, eps: f32, clean: ggml_rs::CleanRows) {
        let set = if clean == ggml_rs::CleanRows::NONE { 0 } else { clean.offset };
        assert!(rows > 0 && x.len >= rows * n && out.len >= rows * n && mods.len >= set + scale_at + n && shift_at.is_none_or(|s| mods.len >= set + s + n) && set < 1 << 30, "chain: a modulated norm of {rows} rows of {n}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let mode = match norm {
            ggml_rs::RowNorm::None => 0u32,
            ggml_rs::RowNorm::Rms => 1,
            ggml_rs::RowNorm::Layer => 2,
        };
        let bound = |v: usize| v.min(u32::MAX as usize) as u32;
        let words = [n as u32, eps.to_bits(), scale_at as u32, shift_at.map_or(u32::MAX, |s| s as u32), rows as u32, mode | (set as u32) << 2, bound(clean.before), bound(clean.from)];
        let r = rows as u32;
        self.dispatch_wide("chain-layernorm-mod-rows", LAYERNORM_MOD_ROWS, [buffer(x), buffer(mods), &d, &d, &d, &d, buffer(out), &drw], &words, (r.min(65535), r.div_ceil(65535), 1));
    }

    fn add_gated_rows_clean(&mut self, x: &DeviceVec, y: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, gate_at: usize, tanh: bool, clean: ggml_rs::CleanRows) {
        let set = if clean == ggml_rs::CleanRows::NONE { 0 } else { clean.offset };
        assert!(x.len >= rows * n && y.len >= rows * n && mods.len >= set + gate_at + n, "chain: a gated residual of {rows} rows of {n}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let bound = |v: usize| v.min(u32::MAX as usize) as u32;
        let words = [n as u32, rows as u32, gate_at as u32, tanh as u32, bound(clean.before), bound(clean.from), set as u32, 0];
        self.dispatch_wide("chain-add-gated-rows", ADD_GATED_ROWS, [buffer(y), buffer(mods), &d, &d, &d, &d, buffer(x), &drw], &words, grid(((rows * n) as u32).div_ceil(256)));
    }

    fn subsample2x_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        assert!(h % 2 == 0 && w % 2 == 0 && x.len >= h * w * c && out.len >= h * w * c / 4, "chain: subsampling {h}x{w} pixels of {c}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w * c / 4) as u32;
        self.dispatch_wide("chain-subsample2x-rows", SUBSAMPLE2X_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn shuffle_down_mean_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, cin: usize, cout: usize, ft: usize, fs: usize) {
        assert!(fs >= 1 && ft >= 1 && h % fs == 0 && w % fs == 0 && (cin * ft * fs * fs) % cout == 0 && x.len >= h * w * cin && out.len >= h * w / (fs * fs) * cout, "chain: a shuffled mean of {h}x{w} pixels of {cin} into {cout}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w / (fs * fs) * cout) as u32;
        self.dispatch_wide("chain-shuffle-down-mean-add-rows", SHUFFLE_DOWN_MEAN_ADD_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, cin as u32, cout as u32, ft as u32, fs as u32], grid(n.div_ceil(256)));
    }

    fn space_to_depth_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, st: usize, sh: usize, sw: usize) {
        let vol = st * sh * sw;
        assert!(vol > 0 && h % sh == 0 && w % sw == 0 && x.len >= h * w * c && out.len >= h * w / (sh * sw) * c * vol, "chain: space to depth of {h}x{w} pixels of {c} by ({st}, {sh}, {sw})");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w / (sh * sw) * c * vol) as u32;
        self.dispatch_wide("chain-space-to-depth-rows", SPACE_TO_DEPTH_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, c as u32, 0, st as u32, sh as u32, sw as u32], grid(n.div_ceil(256)));
    }

    fn group_mean_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, cin: usize, cout: usize) {
        assert!(cout > 0 && cin % cout == 0 && x.len >= rows * cin && out.len >= rows * cout, "chain: a group mean of {rows} rows of {cin} into {cout}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (rows * cout) as u32;
        self.dispatch_wide("chain-group-mean-add-rows", GROUP_MEAN_ADD_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[rows as u32, cin as u32, cout as u32], grid(n.div_ceil(256)));
    }

    fn add_f16(&mut self, w: &DeviceVec, d: &DeviceVec, len: usize) {
        assert!(len % 2 == 0 && w.len * 2 >= len && d.len >= len, "chain: {len} f16 values plus f32");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let words = (len / 2) as u32;
        self.dispatch_wide("chain-add-f16", ADD_F16, [buffer(d), &dm, &dm, &dm, &dm, &dm, buffer(w), &drw], &[words], grid(words.div_ceil(256)));
    }

    fn conv3d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, x: &DeviceVec, frames: usize, h: usize, wd: usize, y: &DeviceVec) {
        self.conv_taps(w, b, cout, cin, 27, x, cin, frames, h, wd, y);
    }

    fn depth_to_space_rows(&mut self, x: &DeviceVec, out: &DeviceVec, frames: usize, h: usize, w: usize, c: usize, st: usize, sh: usize, sw: usize, drop: usize) {
        let ot = frames * st - drop;
        let voxels = ot * h * sh * w * sw;
        assert!(drop < frames * st && x.len >= frames * h * w * c * st * sh * sw && out.len >= voxels * c, "chain: depth to space of {frames}x{h}x{w} voxels of {c} channels by ({st}, {sh}, {sw})");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-depth-to-space-rows", DEPTH_TO_SPACE_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[c as u32, st as u32, sh as u32, sw as u32, voxels as u32, (h * sh) as u32, (w * sw) as u32, drop as u32], grid(((voxels * c) as u32).div_ceil(256)));
    }

    fn conv_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, x: &DeviceVec, h: usize, wd: usize, y: &DeviceVec) {
        assert!(matches!(k, 1 | 3 | 7), "chain: a {k}x{k} convolution");
        self.conv_taps(w, b, cout, cin, k * k, x, cin, 1, h, wd, y);
    }

    fn conv_rows_strided(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, x: &DeviceVec, xs: usize, h: usize, wd: usize, y: &DeviceVec) {
        assert!(matches!(k, 1 | 3 | 7), "chain: a {k}x{k} convolution");
        self.conv_taps(w, b, cout, cin, k * k, x, xs, 1, h, wd, y);
    }

    fn w4a8_f16(&mut self, codes: &DeviceVec, rel: &DeviceVec, channel: &DeviceVec, book: &DeviceVec, rows: usize, cols: usize, rotation: usize, out: &DeviceVec) {
        assert!(
            rows > 0 && cols % 16 == 0 && codes.len * 8 >= rows * cols && rel.len * 4 >= rows * cols / 16 && channel.len >= rows && book.len >= 16 && out.len * 2 >= rows * cols,
            "chain: a W4A8 matrix of {rows} by {cols}"
        );
        assert!(rotation == 0 || (rotation.is_power_of_two() && rotation.trailing_zeros() % 2 == 0 && (4..=4096).contains(&rotation) && cols % rotation == 0), "chain: W4A8's rotation of {rotation} over {cols} columns");
        let drw = self.gpu().dummy_rw().clone();
        let d = self.gpu().dummy().clone();
        let bufs = [buffer(codes), buffer(rel), buffer(channel), buffer(book), &d, &d, buffer(out), &drw];
        let words = [rows as u32, cols as u32];
        if rotation == 0 {
            let n = (rows * cols / 2) as u32;
            self.dispatch_wide("chain-w4a8-f16", W4A8_F16, bufs, &words, grid(n.div_ceil(256)));
        } else {
            let name: &'static str = match rotation {
                4 => "chain-w4a8-f16-rotated-4",
                16 => "chain-w4a8-f16-rotated-16",
                64 => "chain-w4a8-f16-rotated-64",
                256 => "chain-w4a8-f16-rotated-256",
                1024 => "chain-w4a8-f16-rotated-1024",
                _ => "chain-w4a8-f16-rotated-4096",
            };
            let body = W4A8_F16_ROTATED.replace("GS_u", &format!("{rotation}u"));
            let r = rows as u32;
            self.dispatch_wide(name, &body, bufs, &words, ((cols / rotation) as u32, r.min(65535), r.div_ceil(65535)));
        }
    }

    fn window_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, win: usize, shift: usize) {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        assert!(shift < win && x.len >= h * w * c && out.len >= hp * wp * c, "chain: windows of {h}x{w} tokens of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (hp * wp * c) as u32;
        self.dispatch_wide("chain-window-rows", WINDOW_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[h as u32, w as u32, c as u32, win as u32, shift as u32, hp as u32, wp as u32], grid(n.div_ceil(256)));
    }

    fn unwindow_add_rows(&mut self, windows: &DeviceVec, acc: &DeviceVec, h: usize, w: usize, c: usize, win: usize, shift: usize) {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        assert!(shift < win && windows.len >= hp * wp * c && acc.len >= h * w * c, "chain: windows of {h}x{w} tokens of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (h * w * c) as u32;
        self.dispatch_wide("chain-unwindow-add-rows", UNWINDOW_ADD_ROWS, [buffer(windows), &d, &d, &d, &d, &d, buffer(acc), &drw], &[h as u32, w as u32, c as u32, win as u32, shift as u32, hp as u32, wp as u32], grid(n.div_ceil(256)));
    }

    fn window_attention(&mut self, qkv: &DeviceVec, table: &DeviceVec, out: &DeviceVec, h: usize, w: usize, heads: usize, win: usize, shift: usize, scale: f32) {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        let (n, c) = (win * win, heads * 32);
        assert!(n <= 256 && shift < win && qkv.len >= hp * wp * 3 * c && table.len >= (2 * win - 1) * (2 * win - 1) * heads && out.len >= hp * wp * c, "chain: window attention of {h}x{w} tokens, {heads} heads");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        // (a pipeline a window's size: its threads, a query each)
        let name: &'static str = match n {
            144 => "chain-window-attention-144",
            49 => "chain-window-attention-49",
            16 => "chain-window-attention-16",
            _ => panic!("chain: window attention of {win}x{win} windows"),
        };
        let body = WINDOW_ATTENTION.replace("N_u", &format!("{n}u"));
        self.dispatch_wide(name, &body, [buffer(qkv), buffer(table), &d, &d, &d, &d, buffer(out), &drw], &[hp as u32, wp as u32, heads as u32, win as u32, shift as u32, scale.to_bits()], (((hp / win) * (wp / win)) as u32, heads as u32, 1));
        self.weigh(4.0 * (hp * wp * n) as f64 * c as f64);
    }

    fn resize_bilinear_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, oh: usize, ow: usize) {
        assert!(h > 0 && w > 0 && x.len >= h * w * c && out.len >= oh * ow * c, "chain: a resize of {h}x{w} to {oh}x{ow}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (oh * ow * c) as u32;
        self.dispatch_wide("chain-resize-bilinear-rows", RESIZE_BILINEAR_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[h as u32, w as u32, c as u32, oh as u32, ow as u32], grid(n.div_ceil(256)));
    }

    fn blocks_to_channels_rows(&mut self, x: &DeviceVec, out: &DeviceVec, s: usize, c: usize, size: usize) {
        assert!(size > 0 && s % size == 0 && x.len >= s * s * c && out.len >= s * s * c, "chain: patches of {size} of a {s}-pixel picture");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (s * s * c) as u32;
        self.dispatch_wide("chain-blocks-to-channels-rows", BLOCKS_TO_CHANNELS_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[s as u32, c as u32, size as u32], grid(n.div_ceil(256)));
    }

    fn deform_im2col_rows(&mut self, x: &DeviceVec, offsets: &DeviceVec, modulators: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, k: usize, first: usize, pixels: usize) {
        let kk = k * k;
        assert!(k % 2 == 1 && first + pixels <= h * w && x.len >= h * w * c && offsets.len >= h * w * 2 * kk && modulators.len >= h * w * kk && out.len >= pixels * kk * c, "chain: a deformable convolution's taps of {pixels} pixels");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (pixels * kk * c) as u32;
        self.dispatch_wide("chain-deform-im2col-rows", DEFORM_IM2COL_ROWS, [buffer(x), buffer(offsets), buffer(modulators), &d, &d, &d, buffer(out), &drw], &[h as u32, w as u32, c as u32, k as u32, first as u32, pixels as u32], grid(n.div_ceil(256)));
    }

    fn conv1d_padded_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, dilation: usize, pad: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        let cp = cin.div_ceil(32) * 32;
        assert!(dilation > 0 && len > 0 && pad <= (k - 1) * dilation && w.len * 2 >= cout * k * cp && b.len >= cout && x.len >= len * cin && y.len >= len * cout, "chain: a 1-D convolution of {len} steps, {cin} channels to {cout}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let tiles = len.div_ceil(64);
        let per_tile = 2.0 * 64.0 * (cout * cin * k) as f64;
        let chunk = ((CONV_DISPATCH_FLOPS / per_tile) as usize).clamp(1, 65535);
        let mut first = 0;
        while first < tiles {
            let n = chunk.min(tiles - first);
            let words = [cout as u32, cin as u32, len as u32, k as u32, dilation as u32, first as u32, pad as u32];
            self.dispatch_wide("chain-conv1d-f32-tiled", CONV1D_F32_TILED, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &words, ((cout as u32).div_ceil(64), n as u32, 1));
            self.weigh(per_tile * n as f64);
            first += n;
        }
    }

    fn conv_transpose1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, stride: usize, pad: usize, x: &DeviceVec, len: usize, out: usize, y: &DeviceVec) {
        assert!(out + pad <= (len - 1) * stride + k + stride && cin % 4 == 0 && stride > 0 && w.len >= cout * k * cin && b.len >= cout && x.len >= len * cin && y.len >= out * cout, "chain: a transposed 1-D convolution of {len} steps, {cin} channels to {cout}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (out * cout) as u32;
        self.dispatch_wide("chain-conv-transpose1d-rows", CONV_TRANSPOSE1D_ROWS, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &[cout as u32, cin as u32, len as u32, k as u32, stride as u32, pad as u32, out as u32], grid(n.div_ceil(256)));
        self.weigh(2.0 * (out * cout) as f64 * (cin * k.div_ceil(stride)) as f64);
    }

    fn gather_rows(&mut self, x: &DeviceVec, index: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, first: usize, src_rows: usize) {
        assert!(index.len >= first + rows && out.len >= rows * c && x.len >= src_rows * c && rows * c < 1 << 32, "chain: a gather of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-gather-rows", GATHER_ROWS, [buffer(x), buffer(index), &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, c as u32, src_rows as u32, first as u32], grid(n.div_ceil(256)));
    }

    fn repeat_cols_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, repeat: usize) {
        assert!(x.len >= rows * c && out.len >= rows * c * repeat && repeat > 0, "chain: {rows} rows of {c} repeated {repeat} times");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c * repeat) as u32;
        self.dispatch_wide("chain-repeat-cols-add-rows", REPEAT_COLS_ADD_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, c as u32, repeat as u32], grid(n.div_ceil(256)));
    }

    fn depthwise_causal_conv1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, c: usize, k: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        assert!(w.len >= c * k && b.len >= c && x.len >= len * c && y.len >= len * c, "chain: a depthwise convolution of {len} steps of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (len * c) as u32;
        self.dispatch_wide("chain-depthwise-causal-conv1d-rows", DEPTHWISE_CAUSAL_CONV1D_ROWS, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &[c as u32, k as u32, len as u32], grid(n.div_ceil(256)));
    }

    fn snake_beta_rows(&mut self, x: &DeviceVec, freq: &DeviceVec, scale: &DeviceVec, rows: usize, c: usize) {
        assert!(x.len >= rows * c && freq.len >= c && scale.len >= c, "chain: SnakeBeta of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-snake-beta-rows", SNAKE_BETA_ROWS, [buffer(freq), buffer(scale), &d, &d, &d, &d, buffer(x), &drw], &[rows as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn clamp_in_place(&mut self, x: &DeviceVec, len: usize, lo: f32, hi: f32) {
        assert!(x.len >= len, "chain: a clamp of {len}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = len as u32;
        self.dispatch_wide("chain-clamp-in-place", CLAMP_IN_PLACE, [&d, &d, &d, &d, &d, &d, buffer(x), &drw], &[n, lo.to_bits(), hi.to_bits()], grid(n.div_ceil(256)));
    }

    fn snake_rows(&mut self, x: &DeviceVec, alpha: &DeviceVec, rows: usize, c: usize) {
        assert!(x.len >= rows * c && alpha.len >= c, "chain: Snake of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-snake-rows", SNAKE_ROWS, [buffer(alpha), &d, &d, &d, &d, &d, buffer(x), &drw], &[rows as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn tanh_in_place(&mut self, x: &DeviceVec, len: usize) {
        assert!(x.len >= len, "chain: tanh of {len}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = len as u32;
        self.dispatch_wide("chain-tanh-in-place", TANH_IN_PLACE, [&d, &d, &d, &d, &d, &d, buffer(x), &drw], &[n], grid(n.div_ceil(256)));
    }

    fn mul_sigmoid_rows(&mut self, x: &DeviceVec, gate: &DeviceVec, rows: usize, c: usize) {
        assert!(x.len >= rows * c && gate.len >= rows, "chain: a gate of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-mul-sigmoid-rows", MUL_SIGMOID_ROWS, [buffer(gate), &d, &d, &d, &d, &d, buffer(x), &drw], &[rows as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn mean_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, c: usize) {
        assert!(rows > 0 && x.len >= rows * c && out.len >= c, "chain: a mean of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        self.dispatch_wide("chain-mean-rows", MEAN_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, c as u32], ((c as u32).div_ceil(64), 1, 1));
    }

    fn broadcast_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, c: usize, stride: usize, at: usize) {
        assert!(src.len >= c && at + c <= stride && dst.len >= rows * stride, "chain: a broadcast of {c} into {rows} rows");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-broadcast-rows", BROADCAST_ROWS, [buffer(src), &d, &d, &d, &d, &d, buffer(dst), &drw], &[rows as u32, c as u32, stride as u32, at as u32], grid(n.div_ceil(256)));
    }

    fn leaky_relu(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize, slope: f32) {
        assert!(x.len >= len && out.len >= len, "chain: a leaky ReLU of {len}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = len as u32;
        if Arc::ptr_eq(&x.inner, &out.inner) {
            // (in place: the one buffer bound once, written where it is read)
            let body = LEAKY_RELU.replace("@group(0) @binding(0) var<storage, read> x: array<f32>;
", "").replace("let v = x[i];", "let v = out[i];");
            self.dispatch_wide("chain-leaky-relu-in-place", &body, [&d, &d, &d, &d, &d, &d, buffer(out), &drw], &[n, slope.to_bits()], grid(n.div_ceil(256)));
        } else {
            self.dispatch_wide("chain-leaky-relu", LEAKY_RELU, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[n, slope.to_bits()], grid(n.div_ceil(256)));
        }
    }

    fn matmul_nvfp4_rows(&mut self, w: &DeviceVec, scale: &DeviceVec, b: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(k % 64 == 0 && w.len >= n * (k / 8 + k / 64) && scale.len >= 2 && b.len >= n && x.len >= rows * k && y.len >= rows * n && rows.div_ceil(crate::shaders::COOP_TILE as usize) <= 65535, "chain: an NVFP4 matmul [{n}, {k}] of {rows} rows");
        let x16 = self.x16_tiled(x, rows, k);
        let tile = crate::shaders::COOP_TILE;
        let pipeline = self.gpu().named_pipeline("chain-coop-nvfp4", crate::shaders::coop_tiled_nvfp4);
        let tiles = (n as u32).div_ceil(tile) * (rows as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, rows, n, y, None);
        let words = [k as u32, n as u32, rows as u32, 0, n as u32, (k / 8 + k / 64) as u32, splits, 0];
        self.dispatch_kept(&pipeline, buffer(w), buffer(&x16), &out, &words, ((n as u32).div_ceil(tile), (rows as u32).div_ceil(tile), splits));
        self.coop_sum(parts, rows, n, y, splits);
        // the tensor's own scale, and the bias
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-unscale-bias-rows", UNSCALE_BIAS_ROWS, [buffer(b), buffer(scale), &d, &d, &d, &d, buffer(y), &drw], &[n as u32, rows as u32], grid(((rows * n) as u32).div_ceil(256)));
        self.weigh(2.0 * (rows * n) as f64 * k as f64);
    }

    fn add_bias_rows(&mut self, y: &DeviceVec, b: &DeviceVec, rows: usize, n: usize) {
        assert!(y.len >= rows * n && b.len >= n, "chain: a bias on {rows} rows of {n}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-add-bias-rows", ADD_BIAS_ROWS, [buffer(b), &d, &d, &d, &d, &d, buffer(y), &drw], &[n as u32, rows as u32], grid(((rows * n) as u32).div_ceil(256)));
    }

    fn upsample2x_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        assert!(x.len >= h * w * c && out.len >= 4 * h * w * c, "chain: upsampling {h}x{w} pixels of {c}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-upsample2x-rows", UPSAMPLE2X_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[c as u32, h as u32, w as u32], grid(((4 * h * w * c) as u32).div_ceil(256)));
    }

    fn shuffle_up_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, cin: usize, cout: usize, ft: usize) {
        let repeats = cout * ft * 4 / cin.max(1);
        assert!(ft > 0 && repeats > 0 && repeats * cin == cout * ft * 4 && x.len >= h * w * cin && out.len >= 4 * h * w * cout, "chain: an upsampling shortcut of {cin} channels to {cout}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-shuffle-up-add-rows", SHUFFLE_UP_ADD_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[cin as u32, cout as u32, ft as u32, repeats as u32, h as u32, w as u32], grid(((4 * h * w * cout) as u32).div_ceil(256)));
    }

    fn gelu_erf(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && out.len >= len, "chain: an exact GELU of {len}");
        let pipeline = self.named("chain-gelu-erf", GELU_ERF);
        self.dispatch_kept(&pipeline, buffer(x), buffer(x), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn gelu(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && out.len >= len, "chain: a GELU of {len}");
        let pipeline = self.named("chain-gelu", GELU);
        self.dispatch_kept(&pipeline, buffer(x), buffer(x), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn attention_rows_full(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, kv_len: usize, scale: f32) {
        assert!(rows > 0 && kv_len > 0, "chain: a full attention of {rows} queries over {kv_len} positions");
        // (a full attention's `past` is its positions: no query's own place among them)
        if !self.attention_rows_coop_masked(q, kv, out, rows, n_h, n_kv, head_dim, kv_len, None, scale, true) {
            self.attention_rows_f32_masked(q, kv, out, rows, n_h, n_kv, head_dim, kv_len, None, scale, true);
        }
    }

    fn rope_split_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec) {
        assert!(x.len >= rows * heads * head_dim && table.len >= rows * heads * head_dim, "chain: a split RoPE of {rows} rows");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * head_dim / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, 1, rows as u32, 0, 1], grid(pairs.div_ceil(256)));
    }

    fn group_norm_rows(&mut self, x: &DeviceVec, weight: &DeviceVec, bias: &DeviceVec, out: &DeviceVec, stats: &DeviceVec, pixels: usize, c: usize, groups: usize, eps: f32, silu: bool) {
        let chunks = pixels.div_ceil(256);
        assert!(
            groups > 0 && c % groups == 0 && pixels > 0 && chunks <= 65535 && x.len >= pixels * c && out.len >= pixels * c && weight.len >= c && bias.len >= c && stats.len >= groups * (chunks + 1) * 2,
            "chain: a group norm of {pixels} pixels of {c} in {groups} groups"
        );
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let words = [pixels as u32, c as u32, groups as u32, chunks as u32, eps.to_bits(), silu as u32];
        self.dispatch_wide("chain-group-norm-sums", GROUP_NORM_SUMS, [buffer(x), &d, &d, &d, &d, &d, buffer(stats), &drw], &words, (groups as u32, chunks as u32, 1));
        self.dispatch_wide("chain-group-norm-stats", GROUP_NORM_STATS, [buffer(x), &d, &d, &d, &d, &d, buffer(stats), &drw], &words, (groups as u32, 1, 1));
        self.dispatch_wide("chain-group-norm-apply", GROUP_NORM_APPLY, [buffer(x), buffer(weight), buffer(bias), buffer(stats), &d, &d, buffer(out), &drw], &words, grid(((pixels * c) as u32).div_ceil(256)));
    }

    fn geglu_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize, ff: usize) {
        assert!(fused.len >= rows * 2 * ff && out.len >= rows * ff, "chain: GEGLU of {rows} rows of {ff}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-geglu-rows", GEGLU_ROWS, [buffer(fused), &d, &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, ff as u32], grid(((rows * ff) as u32).div_ceil(256)));
    }

    fn subsample2x_even_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        assert!(h % 2 == 0 && w % 2 == 0 && x.len >= h * w * c && out.len >= h * w * c / 4, "chain: subsampling {h}x{w} pixels of {c}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w * c / 4) as u32;
        self.dispatch_wide("chain-subsample2x-even-rows", SUBSAMPLE2X_EVEN_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn nag_mix(&mut self, pos: &DeviceVec, neg: &DeviceVec, rows: usize, width: usize, scale: f32, tau: f32, alpha: f32) {
        assert!(width % 4 == 0 && pos.len >= rows * width && neg.len >= rows * width, "chain: a guidance mix of {rows} rows of {width}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let rows32 = rows as u32;
        self.dispatch_wide("chain-nag-mix", NAG_MIX, [buffer(neg), &d, &d, &d, &d, &d, buffer(pos), &drw], &[width as u32, rows32, scale.to_bits(), tau.to_bits(), alpha.to_bits()], (rows32.min(65535), rows32.div_ceil(65535), 1));
    }

    fn head_gate_rows(&mut self, y: &DeviceVec, logits: &DeviceVec, rows: usize, heads: usize, head_dim: usize) {
        assert!(y.len >= rows * heads * head_dim && logits.len >= rows * heads, "chain: a head gate of {rows} rows");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let len = (rows * heads * head_dim) as u32;
        self.dispatch_wide("chain-head-gate-rows", HEAD_GATE_ROWS, [buffer(logits), &d, &d, &d, &d, &d, buffer(y), &drw], &[head_dim as u32, len], grid(len.div_ceil(256)));
    }

    fn mul_sigmoid(&mut self, x: &DeviceVec, gate: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && gate.len >= len && out.len >= len && !Arc::ptr_eq(&x.inner, &out.inner), "chain: a gate of {len}");
        let pipeline = self.named("chain-mul-sigmoid", MUL_SIGMOID);
        self.dispatch_kept(&pipeline, buffer(x), buffer(gate), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn silu_mul(&mut self, gate: &DeviceVec, up: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(gate.len >= len && up.len >= len && out.len >= len, "chain: a SwiGLU of {len}");
        let pipeline = self.named("chain-silu-mul", SILU_MUL);
        self.dispatch_kept(&pipeline, buffer(gate), buffer(up), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn ple_gate(&mut self, key: &DeviceVec, x: &DeviceVec, value: &DeviceVec, norm_key: &DeviceVec, norm_query: &DeviceVec, norm_conv: &DeviceVec, gated: &DeviceVec, conv_in: &DeviceVec, rows: usize, streams: usize, d: usize, eps: f32) {
        let width = streams * d;
        assert!(key.len >= rows * width && x.len >= rows * width && value.len >= rows * d && norm_key.len >= width && norm_query.len >= width && norm_conv.len >= width && gated.len >= rows * width && conv_in.len >= rows * width, "chain: an n-gram gate of {rows} rows of {streams} streams of {d}");
        let body = format!("{}{PLE_GATE}", crate::exl3::HALF);
        self.dispatch_wide("chain-ple-gate", &body, [buffer(key), buffer(x), buffer(value), buffer(norm_key), buffer(norm_query), buffer(norm_conv), buffer(gated), buffer(conv_in)], &[d as u32, streams as u32, eps.to_bits()], ((rows * streams) as u32, 1, 1));
    }

    fn ple_conv(&mut self, x: &DeviceVec, gated: &DeviceVec, conv_in: &DeviceVec, window: &DeviceVec, weight: &DeviceVec, rows: usize, width: usize, kernel: usize, dilation: usize) {
        assert!(kernel >= 1 && x.len >= rows * width && gated.len >= rows * width && conv_in.len >= rows * width && window.len >= (kernel - 1) * dilation * width && weight.len >= width * kernel, "chain: an n-gram conv of {rows} rows of {width}");
        let d = self.gpu().dummy().clone();
        self.dispatch_wide("chain-ple-conv", PLE_CONV, [buffer(gated), buffer(conv_in), buffer(weight), &d, &d, &d, buffer(x), buffer(window)], &[width as u32, rows as u32, kernel as u32, dilation as u32], ((width as u32).div_ceil(256), 1, 1));
    }

    fn matmul_f16_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(k % 2 == 0 && w.len * 2 >= n * k && x.len >= rows * k && y.len >= rows * n, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
        // (the tiled kernels take a grid's 65,535 tiles of rows: 64 rows each, the tensor cores' 32)
        assert!(n <= 65535 && rows.div_ceil(64) <= 65535, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
        if rows == 1 {
            // a long row a workgroup (as the f32 one sums it); short ones eight threads each, 32 a workgroup
            if k >= 2048 && k % 4 == 0 {
                let pipeline = self.gpu().named_pipeline("chain-matvec-f16-4", || MATVEC_F16.to_string());
                self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (n as u32, 1, 1));
            } else {
                let pipeline = self.named("chain-matvec-f16-narrow", MATVEC_F16_NARROW);
                self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], ((n as u32).div_ceil(32), 1, 1));
            }
            return;
        }
        // a few rows (a check of drafts): each weight read once for all of them, each row summed as one row is
        if rows <= 8 {
            const NAMES: [[&str; 7]; 2] = [
                ["chain-matvec-f16-rows-2", "chain-matvec-f16-rows-3", "chain-matvec-f16-rows-4", "chain-matvec-f16-rows-5", "chain-matvec-f16-rows-6", "chain-matvec-f16-rows-7", "chain-matvec-f16-rows-8"],
                ["chain-matvec-f16-narrow-2", "chain-matvec-f16-narrow-3", "chain-matvec-f16-narrow-4", "chain-matvec-f16-narrow-5", "chain-matvec-f16-narrow-6", "chain-matvec-f16-narrow-7", "chain-matvec-f16-narrow-8"],
            ];
            let wide = k >= 2048 && k % 4 == 0;
            let pipeline = self.gpu().named_pipeline(NAMES[!wide as usize][rows - 2], || matvec_f16_rows(rows, !wide));
            let groups = if wide { n as u32 } else { (n as u32).div_ceil(32) };
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (groups, 1, 1));
            return;
        }
        // a prompt's rows on the tensor cores (f16 tokens, f32 sums: Qwen3.8-Flash-Next's hyper-connections' 324 by
        // 10,240 and back, its routers' and its delta nets' `ba` some 14 ms of a chunk of 512 where 33; OAIY_NO_COOP_F16
        // the f32 tiled kernel)
        if std::env::var_os("OAIY_NO_COOP_F16").is_some() || !self.matmul_f16_coop(w, n, k, x, y, rows, None) {
            self.matmul_f16_tiled(w, n, k, x, y, rows);
        }
        self.weigh(2.0 * (rows * n) as f64 * k as f64);
    }

    fn matmul_f32_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(w.len >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows.div_ceil(64) <= 65535, "chain: an f32 matmul [{n}, {k}] of {rows} rows");
        if rows == 1 {
            let pipeline = if k % 4 == 0 { self.gpu().named_pipeline("chain-matvec-f32-4", || MATVEC_F32_4.to_string()) } else { self.named("chain-matmul-f32", MATMUL_F32) };
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (n as u32, 1, 1));
            return;
        }
        // a few rows (a check of drafts): each weight read once for all of them, where the tiled kernel's 64-row tiles
        // and its split sums cost a prompt's
        if rows <= 8 && k % 4 == 0 {
            let pipeline = self.gpu().named_pipeline("chain-matvec-f32-4-rows", || MATVEC_F32_4_ROWS.to_string());
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32, rows as u32], (n as u32, 1, 1));
            return;
        }
        // a prompt's rows in 64x64 tiles; few tiles (few outputs) split k for a second pass to add up, enough
        // workgroups to fill the GPU, 256 of k a split at least
        let tiles = n.div_ceil(64) * rows.div_ceil(64);
        let want = 1024usize.div_ceil(tiles).min(k / 256).max(1);
        let kc = k.div_ceil(want).div_ceil(16) * 16;
        let splits = k.div_ceil(kc);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let grid = (n.div_ceil(64) as u32, rows.div_ceil(64) as u32, splits as u32);
        let words = [n as u32, k as u32, rows as u32, kc as u32];
        if splits == 1 {
            self.dispatch_wide("chain-matmul-f32-tiled", MATMUL_F32_TILED, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(y), &drw], &words, grid);
        } else {
            let part = self.scratch(splits * rows * n);
            self.dispatch_wide("chain-matmul-f32-tiled", MATMUL_F32_TILED, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(&part), &drw], &words, grid);
            let len = (rows * n) as u32;
            self.dispatch_wide("chain-sum-splits", SUM_SPLITS, [buffer(&part), &d, &d, &d, &d, &d, buffer(y), &drw], &[len, splits as u32], (len.div_ceil(256).min(65535), len.div_ceil(256 * 65535), 1));
            // (its parts read: spare for the next split's, where each had its own to the recording's end, a sound
            // step's 360 of 18 MB without tensor cores)
            self.spare.push((buffer(&part).size(), buffer(&part).clone()));
        }
        self.weigh(2.0 * (rows * n) as f64 * k as f64);
    }

    fn ssm_conv(&mut self, qkv: &DeviceVec, weight: &DeviceVec, state: &DeviceVec, out: &DeviceVec, rows: usize, channels: usize, kernel: usize) {
        assert!((2..=8).contains(&kernel) && qkv.len >= rows * channels && weight.len >= channels * kernel && state.len >= (kernel - 1) * channels && out.len >= rows * channels, "chain: a conv of {kernel} over {rows} rows of {channels}");
        let q = buffer(qkv);
        let words = [channels as u32, rows as u32, kernel as u32];
        let groups = (channels as u32).div_ceil(256);
        // a prompt's a thread a (channel, token) where the thread a channel walked its tokens (0.26 ms of a 27B's
        // layer for 512), then the state; a step's and a check's as they were
        if rows >= 16 && rows <= 65535 {
            let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
            self.dispatch_wide("chain-ssm-conv-rows", SSM_CONV_ROWS, [q, buffer(weight), buffer(state), &d, &d, &d, &drw, buffer(out)], &words, (groups, rows as u32, 1));
            self.dispatch_wide("chain-ssm-conv-state", SSM_CONV_STATE, [q, &d, &d, &d, &d, &d, buffer(state), &drw], &words, (groups, 1, 1));
            return;
        }
        self.dispatch_wide("chain-ssm-conv", SSM_CONV, [q, buffer(weight), q, q, q, q, buffer(state), buffer(out)], &words, (groups, 1, 1));
    }

    fn delta_net(&mut self, conv: &DeviceVec, z: &DeviceVec, beta_alpha: &DeviceVec, ssm_a: &DeviceVec, dt_bias: &DeviceVec, norm: &DeviceVec, state: &DeviceVec, out: &DeviceVec, d: DeltaNet) {
        let ch = 2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim;
        // a pipeline a head size, the size a constant of it
        let name: &'static str = match d.k_dim {
            16 => "chain-delta-net-16",
            32 => "chain-delta-net-32",
            64 => "chain-delta-net-64",
            128 => "chain-delta-net-128",
            other => panic!("chain: a delta net of heads of {other} (16, 32, 64 or 128)"),
        };
        assert!(d.k_dim == d.v_dim && d.k_heads > 0, "chain: a delta net of heads of {} and {} (one size)", d.k_dim, d.v_dim);
        assert!(
            conv.len >= d.rows * ch && z.len >= d.rows * d.v_heads * d.v_dim && beta_alpha.len >= d.rows * 2 * d.v_heads && ssm_a.len >= d.v_heads && dt_bias.len >= d.v_heads
                && norm.len >= d.v_dim && state.len >= d.v_heads * d.k_dim * d.v_dim && out.len >= d.rows * d.v_heads * d.v_dim,
            "chain: a delta net's buffers"
        );
        let words = [d.v_heads as u32, d.k_heads as u32, d.k_dim as u32, d.v_dim as u32, d.rows as u32, d.scale_q.to_bits(), d.eps.to_bits(), d.sigmoid_gate as u32];
        // a prompt's in three passes (the recurrence's alone in turn); a step's and a check's few rows the one kernel
        // (a check's rows a step's bit for bit)
        if d.rows >= 16 && d.k_dim >= 32 && d.v_heads % d.k_heads == 0 {
            self.delta_net_rows(conv, z, beta_alpha, ssm_a, dt_bias, norm, state, out, &d, &words);
            return;
        }
        self.dispatch_wide(name, &delta_net_one(d.k_dim), [buffer(conv), buffer(z), buffer(beta_alpha), buffer(ssm_a), buffer(dt_bias), buffer(norm), buffer(state), buffer(out)], &words, (d.v_heads as u32, 1, 1));
    }

    fn rope_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool) {
        assert!(x.len >= rows * heads * head_dim && table.len >= rows * head_dim, "chain: RoPE of {rows} rows");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * head_dim / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, neox as u32, rows as u32], grid(pairs.div_ceil(256)));
    }

    fn store_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, len: usize, start: usize, stride: usize, at: usize) {
        assert!(src.len >= rows * len && at + len <= stride && dst.len >= (start + rows) * stride, "chain: storing {rows} rows");
        let pipeline = self.named("chain-store-rows", STORE_ROWS);
        let params = self.uniform(&[len as u32, start as u32, stride as u32, at as u32, rows as u32]);
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, grid(((rows * len) as u32).div_ceil(256)));
    }

    fn attention_rows(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) {
        if !self.attention_rows_coop(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale) {
            self.attention_rows_f32(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale);
        }
    }

    fn copy(&mut self, src: &DeviceVec, src_at: usize, dst: &DeviceVec, dst_at: usize, len: usize) {
        assert!(src_at + len <= src.len && dst_at + len <= dst.len, "chain: copying {len} from {src_at} of {} to {dst_at} of {}", src.len, dst.len);
        let pipeline = self.named("chain-copy", COPY);
        let params = self.uniform(&[len as u32, dst_at as u32, src_at as u32]);
        // (past 16.8 million values the grid's second dimension: a 1024x1024 reference image's tokens in a prefix)
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, grid((len as u32).div_ceil(256)));
    }

    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        // (OAIY_ATTENTION_PART4: a workgroup a query head, as before the groups' kernel)
        static HEADS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let group = !*HEADS.get_or_init(|| std::env::var_os("OAIY_ATTENTION_PART4").is_some());
        self.attention_by(q, kv, out, n_h, n_kv, head_dim, lo, kv_len, cap, scale, group);
    }

    fn halve(&mut self, src: &DeviceVec, dst: &DeviceVec, at: usize, len: usize) {
        assert!(at % 2 == 0 && len % 2 == 0 && at + len <= src.len && (at + len) / 2 <= dst.len, "chain: halves of {len} values at {at} of {} into {}", src.len, dst.len);
        if len == 0 {
            return;
        }
        let pipeline = self.gpu().named_pipeline("chain-halve", || HALVE.to_string());
        let params = self.uniform(&[(at / 2) as u32, (len / 2) as u32]);
        let d = self.gpu().dummy().clone();
        let groups = ((len / 2) as u32).div_ceil(256);
        self.dispatch(&pipeline, &d, buffer(src), buffer(dst), &params, (groups.min(65535), groups.div_ceil(65535), 1));
    }

    fn attention_halved(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        let runs = kv_len.saturating_sub(lo).div_ceil(SPLIT).max(1);
        assert!(
            self.backend.attention_halves(n_h, n_kv, head_dim) && kv.len >= cap * n_kv * head_dim && kv_len <= cap && out.len >= n_h * head_dim + n_h * runs * (head_dim + 2),
            "chain: attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, kv_len as u32, lo as u32, runs as u32, scale.to_bits()]);
        const NAMES: [&str; 9] = ["", "", "chain-attention-halves-2", "chain-attention-halves-3", "chain-attention-halves-4", "chain-attention-halves-5", "chain-attention-halves-6", "chain-attention-halves-7", "chain-attention-halves-8"];
        let g = n_h / n_kv;
        let part = self.gpu().named_pipeline(NAMES[g], || attention_part_group_halved(g));
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_kv as u32, runs as u32, 1));
        let join = self.named("chain-attention-join", ATTENTION_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, 1, 1));
    }

    fn qsa_pool(&mut self, raw: &DeviceVec, pooled: &DeviceVec, blocks: usize, ratio: usize, d: usize) {
        assert!(d <= 256 && raw.len >= blocks * ratio * d && pooled.len >= blocks * d, "chain: QSA's pool of {blocks} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let b = blocks as u32;
        self.dispatch_wide("chain-qsa-pool", QSA_POOL, [buffer(raw), &dd, &dd, &dd, &dd, &dd, buffer(pooled), &drw], &[b, ratio as u32, d as u32], (b.min(65535), b.div_ceil(65535), 1));
    }

    fn qsa_scores(&mut self, q: &DeviceVec, pooled: &DeviceVec, scores: &DeviceVec, rows: usize, heads: usize, d: usize, nb: usize, first: usize, ratio: usize, scale: f32) {
        assert!(heads * d <= 2048 && q.len >= rows * heads * d && pooled.len >= nb * d && scores.len >= rows * nb, "chain: QSA's scores of {rows} rows over {nb} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        // a head a multiple of 4 wide, 8 heads at most: each pooled key read once for them all, a vec4 at a time
        if d % 4 == 0 && (1..=8).contains(&heads) && std::env::var_os("OAIY_QSA_SCORES_F1").is_none() {
            const NAMES: [&str; 9] = ["", "chain-qsa-scores4-1", "chain-qsa-scores4-2", "chain-qsa-scores4-3", "chain-qsa-scores4-4", "chain-qsa-scores4-5", "chain-qsa-scores4-6", "chain-qsa-scores4-7", "chain-qsa-scores4-8"];
            let src = qsa_scores4(heads);
            self.dispatch_wide(NAMES[heads], &src, [buffer(q), buffer(pooled), &dd, &dd, &dd, &dd, buffer(scores), &drw], &[rows as u32, heads as u32, d as u32, nb as u32, first as u32, ratio as u32, scale.to_bits()], ((nb as u32).div_ceil(256), rows as u32, 1));
            return;
        }
        self.dispatch_wide("chain-qsa-scores", QSA_SCORES, [buffer(q), buffer(pooled), &dd, &dd, &dd, &dd, buffer(scores), &drw], &[rows as u32, heads as u32, d as u32, nb as u32, first as u32, ratio as u32, scale.to_bits()], ((nb as u32).div_ceil(256), rows as u32, 1));
    }

    fn qsa_select(&mut self, scores: &DeviceVec, list: &DeviceVec, rows: usize, nb: usize, first: usize, ratio: usize, keep: usize) {
        assert!(nb <= 4096 && keep > 0 && scores.len >= rows * nb && list.len >= rows * keep, "chain: QSA's selection of {keep} of {nb} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-qsa-select", QSA_SELECT, [buffer(scores), &dd, &dd, &dd, &dd, &dd, buffer(list), &drw], &[rows as u32, nb as u32, first as u32, ratio as u32, keep as u32], (rows as u32, 1, 1));
    }

    fn qsa_attention(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) {
        // a prompt's rows on the tensor cores where the device has them (a check's few as a step's)
        if rows > 8 && self.qsa_attention_coop(q, kv, list, out, rows, n_h, n_kv, head_dim, first, ratio, keep, scale) {
            return;
        }
        self.qsa_attention_f32(q, kv, list, out, rows, n_h, n_kv, head_dim, first, ratio, keep, scale);
    }

    fn argmax_softmax(&mut self, x: &DeviceVec, out: &DeviceVec) {
        assert!(x.len > 0 && out.len >= 3, "chain: a draft's token of {} logits", x.len);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-argmax-softmax", ARGMAX_SOFTMAX, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[x.len as u32], (1, 1, 1));
    }

    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize) {
        assert!(offset + len <= v.len, "chain: reading {len} at {offset} of {}", v.len);
        let staging = self.gpu().staging(((len.max(1) * 4) as u64).next_power_of_two().max(256));
        self.reads.push((buffer(v).clone(), offset, staging, len));
    }

    fn finish(mut self: Box<Self>) -> Vec<Vec<f32>> {
        // (profiled: what is left a piece of its own, timed as the others)
        let profiled = crate::profile::chain_on();
        // (by the feed: what is left a piece of its own too, encoded at its turn as the others)
        if profiled || self.fed() {
            self.submit_piece();
        }
        let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
        let start = std::time::Instant::now();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        if !self.dispatches.is_empty() {
            // what the pieces submitted while recording left (`push`), with the reads
            let i = self.next_stamp();
            let timestamp_writes = self.stamp_writes(i);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
            for (pipeline, group, (x, y, z)) in &self.dispatches {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, *z);
            }
        }
        // (the pieces since a flush, if any, resolved with this; else the flush's)
        let late = self.resolve_pieces(&mut enc);
        let pieces = late.clone().or_else(|| self.stamped.take());
        for (from, offset, staging, len) in &self.reads[self.copied..] {
            if *len > 0 {
                enc.copy_buffer_to_buffer(from, (*offset * 4) as u64, staging, 0, (*len * 4) as u64);
            }
        }
        // nothing recorded since a flush: its submission the one waited for (profiled: this one, after every timed piece)
        let since = !self.dispatches.is_empty() || !self.held.is_empty() || !self.lists.is_empty() || self.copied < self.reads.len() || profiled || late.is_some();
        let command = enc.finish();
        crate::profile::add(&crate::profile::CHAIN_ENCODE, start);
        let submitted = std::time::Instant::now();
        let last = match self.flushed.take() {
            Some(piece) if !since => piece,
            _ => {
                // (a held recording's pieces first, in turn)
                let held = std::mem::take(&mut self.held);
                if !held.is_empty() {
                    self.gpu().submit_piece(held);
                }
                for list in std::mem::take(&mut self.lists) {
                    self.gpu().feed(list);
                }
                self.gpu().submit_after(vec![command])
            }
        };
        // (gone to the queue before its staging buffers are mapped: a submission may not copy into a mapped one)
        let index = self.gpu().gone(last);
        for (_, _, staging, len) in &self.reads {
            staging.slice(..(*len as u64 * 4).max(4)).map_async(wgpu::MapMode::Read, |_| {});
        }
        for (_, staging) in &self.timed {
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        if let Some(staging) = &pieces {
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        // this recording's work waited for, not what was submitted after it (the next chunk's, as this one's rows
        // are read)
        self.gpu().wait(Some(index));
        crate::profile::add(&crate::profile::CHAIN_WAIT, submitted);
        let pooled = std::mem::take(&mut self.pooled);
        self.gpu().unpool(pooled);
        if let Some(staging) = pieces {
            let period = self.gpu().queue().get_timestamp_period() as f64;
            let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping the pieces' times");
            let ticks: Vec<u64> = view.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
            drop(view);
            staging.unmap();
            let busy: u64 = ticks.chunks_exact(2).map(|p| p[1].saturating_sub(p[0])).sum();
            let span = ticks.last().copied().unwrap_or(0).saturating_sub(ticks.first().copied().unwrap_or(0));
            use std::sync::atomic::Ordering as O;
            crate::profile::PIECES[0].fetch_add((busy as f64 * period) as u64, O::Relaxed);
            crate::profile::PIECES[1].fetch_add((span as f64 * period) as u64, O::Relaxed);
            crate::profile::PIECES[2].fetch_add(ticks.len() as u64 / 2, O::Relaxed);
            if std::env::var("OAIY_PIECE_STAMPS").is_ok_and(|v| v == "2") {
                // each piece: its start after the first's, and how long (ms)
                static BASE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let base = *BASE.get_or_init(|| ticks[0]);
                let at = |t: u64| t.saturating_sub(base) as f64 * period / 1e6;
                eprintln!("    a recording's {} pieces, {:.1} ms busy over {:.1}, on the GPU's clock {:.1}..{:.1}", ticks.len() / 2, busy as f64 * period / 1e6, span as f64 * period / 1e6, at(ticks[0]), at(*ticks.last().expect("a piece")));
            }
        }
        for (pipelines, staging) in std::mem::take(&mut self.timed) {
            let period = self.gpu().queue().get_timestamp_period() as f64;
            let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping the profile");
            let ticks: Vec<u64> = view.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
            drop(view);
            staging.unmap();
            let mut k = crate::profile::KERNELS.lock().unwrap_or_else(|p| p.into_inner());
            for (i, pipeline) in pipelines.iter().enumerate() {
                let ns = (ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period) as u64;
                let e = k.entry(self.gpu().name_of(pipeline)).or_default();
                e.0 += ns;
                e.1 += 1;
            }
        }
        // copied as bytes (a 512-row chunk's cache rows, 64 MB: a value at a time some 15 ms), 8 MB or more of them on
        // every core, the host's pages first touched there (a checkpoint's 96 recurrent states, 149 MB: 34 ms on one)
        let copied = |staging: &wgpu::Buffer, len: usize| {
            use rayon::prelude::*;
            let view = staging.slice(..(len as u64 * 4).max(4)).get_mapped_range().expect("webgpu: mapping a finished buffer");
            let mut v = vec![0f32; len];
            let (to, from) = (bytemuck::cast_slice_mut::<f32, u8>(&mut v), &view[..len * 4]);
            if from.len() >= 8 << 20 {
                to.par_chunks_mut(1 << 20).zip(from.par_chunks(1 << 20)).for_each(|(t, f)| t.copy_from_slice(f));
            } else {
                to.copy_from_slice(from);
            }
            drop(view);
            staging.unmap();
            v
        };
        let out = if self.reads.iter().map(|r| r.3 * 4).sum::<usize>() >= 8 << 20 && self.reads.len() > 1 {
            use rayon::prelude::*;
            self.reads.par_iter().map(|(_, _, staging, len)| copied(staging, *len)).collect()
        } else {
            self.reads.iter().map(|(_, _, staging, len)| copied(staging, *len)).collect()
        };
        let staged: Vec<(u64, wgpu::Buffer)> = std::mem::take(&mut self.reads).into_iter().map(|(_, _, staging, _)| (staging.size(), staging)).collect();
        self.gpu().unstage(staged);
        crate::profile::add(&crate::profile::LINEAR_WAIT, start);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_quants::GgmlType;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2001) as f32 / 1000.0 - 1.0
        }
    }

    fn close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= 1e-4 * scale, "{what} [{i}]: {x} against {y}");
        }
    }

    /// A chain gives the CPU backend's answer: RMSNorm, a quantized matmul, the fused SwiGLU and an add, read back.
    #[test]
    fn a_chain_matches_the_cpus_ops() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (k, ff) = (256usize, 64usize);
        let mut next = rng(0x1234_5679);
        // a Q8_0 weight [2ff, k]: blocks of a scale and 32 int8s
        let mut bytes = vec![0u8; 2 * ff * (k / 32) * 34];
        for blk in bytes.chunks_mut(34) {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
            for v in blk[2..].iter_mut() {
                *v = ((next() + 1.0) * 127.0) as u8;
            }
        }
        let wq = ggml_rs::QuantizedTensor::from_bytes_cpu(bytes, vec![2 * ff, k], GgmlType::Q8_0);
        let wq = ggml_rs::Backend::to_device_quant(&b, wq);
        assert!(DeviceChain::holds(&b, &wq));
        let xv: Vec<f32> = (0..k).map(|_| next()).collect();
        let nw: Vec<f32> = (0..k).map(|_| next() * 0.5 + 1.0).collect();
        let res: Vec<f32> = (0..ff).map(|_| next()).collect();
        // CPU: rmsnorm, matmul, silu-mul split, add
        let xn = ggml_rs::Backend::rmsnorm(&cpu, &ggml_rs::Tensor::from_vec(xv.clone(), vec![1, k]), &ggml_rs::Tensor::from_vec(nw.clone(), vec![k]), 1e-5);
        let gu = ggml_rs::Backend::linear_q(&cpu, &xn, &ggml_rs::QuantizedTensor::from_bytes_cpu(wq.to_host().bytes().to_vec(), vec![2 * ff, k], GgmlType::Q8_0));
        let act = ggml_rs::Backend::silu_mul_split(&cpu, &gu, ff);
        let want: Vec<f32> = act.data().iter().zip(&res).map(|(a, r)| a + r).collect();
        // the chain
        let (x, n, xnd, gud, actd, acc) = (b.vec(k), b.vec(k), b.vec(k), b.vec(2 * ff), b.vec(ff), b.vec(ff));
        DeviceChain::upload(&b, &x, &xv);
        DeviceChain::upload(&b, &n, &nw);
        DeviceChain::upload(&b, &acc, &res);
        let mut rec = b.begin();
        rec.rmsnorm(&x, &n, &xnd, 1e-5);
        rec.matmul(&wq, &xnd, &gud);
        rec.silu_mul_split(&gud, &actd);
        rec.add(&acc, &actd);
        rec.read(&acc);
        rec.read(&xnd);
        let got = rec.finish();
        assert_eq!(got.len(), 2);
        close(&got[1], xn.data(), "rmsnorm");
        close(&got[0], &want, "the chain");
    }

    /// What a chained one-row matmul costs on the GPU: 28 of one weight in a chain (a layer's worth of dispatches a
    /// model's), at a 3B Llama's shapes, against the weight's bytes; and a chain of small ops alone. (The one-row kernel reads its weights at 170-280 GB/s,
    /// whatever its lanes a row, its loads or its value array: a kernel of wide loads is what would change it.)
    #[test]
    #[ignore = "a timing; run with --nocapture"]
    fn measure_chained_matmuls() {
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        // a 3B Llama's attention and FFN, and Qwen3.8 27B's types: its FFN gate (Q3_K), qkv (Q5_K) and down (Q4_K)
        for (dtype, n, k, block, bytes) in [(GgmlType::Q4_K, 3072usize, 3072usize, 256usize, 144usize), (GgmlType::Q4_K, 16384, 3072, 256, 144), (GgmlType::Q6_K, 3072, 8192, 256, 210),
            (GgmlType::Q3_K, 17408, 5120, 256, 110), (GgmlType::Q5_K, 10240, 5120, 256, 176), (GgmlType::Q4_K, 5120, 17408, 256, 144)] {
            let nbytes = n * (k / block) * bytes;
            let mut next = rng(n as u32);
            let mut raw = vec![0u8; nbytes];
            for v in raw.iter_mut() {
                *v = ((next() + 1.0) * 100.0) as u8;
            }
            let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
            for m in [1usize, 2, 3, 4] {
                let (x, y) = (b.vec(m * k), b.vec(m * n));
                DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
                let reps = 28;
                let run = || {
                    let mut rec = b.begin();
                    for _ in 0..reps {
                        rec.matmul_rows(&w, &x, &y, m);
                    }
                    rec.read_range(&y, 0, 1);
                    rec.finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                let secs = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
                eprintln!("{dtype:?} [{n}, {k}] ({:.1} MB) x {m} rows: {:.1} us a matmul in a chain, {:.0} GB/s", nbytes as f64 / 1e6, secs * 1e6, nbytes as f64 / secs / 1e9);
            }
        }
        let (a, c) = (b.vec(3072), b.vec(3072));
        let run = || {
            let mut rec = b.begin();
            for _ in 0..28 * 8 {
                rec.add(&a, &c);
            }
            rec.read_range(&a, 0, 1);
            rec.finish();
        };
        run();
        let t = std::time::Instant::now();
        for _ in 0..5 {
            run();
        }
        eprintln!("an add of 3072 in a chain: {:.1} us", t.elapsed().as_secs_f64() / 5.0 / (28.0 * 8.0) * 1e6);
        // Qwen3.8 27B's gated delta net, a decode step's and a prompt chunk's, 48 layers' worth in a chain
        let (nv, nk, dim, kern) = (48usize, 16usize, 128usize, 4usize);
        let ch = 2 * nk * dim + nv * dim;
        for rows in [1usize, 6, 64] {
            let v = |n: usize| {
                let v = b.vec(n);
                DeviceChain::upload(&b, &v, &(0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>());
                v
            };
            let (qkv, z, ba, cw, a, dt, nm, conv, state) = (v(rows * ch), v(rows * nv * dim), v(rows * 2 * nv), v(ch * kern), v(nv), v(nv), v(dim), v((kern - 1) * ch), v(nv * dim * dim));
            let (co, out) = (b.vec(rows * ch), b.vec(rows * nv * dim));
            let d = DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: 0.088, eps: 1e-6, sigmoid_gate: false };
            for (what, conv_too) in [("conv", true), ("delta net", false)] {
                let run = || {
                    let mut rec = b.begin();
                    for _ in 0..48 {
                        if conv_too {
                            rec.ssm_conv(&qkv, &cw, &conv, &co, rows, ch, kern);
                        } else {
                            rec.delta_net(&co, &z, &ba, &a, &dt, &nm, &state, &out, d);
                        }
                    }
                    rec.read_range(&out, 0, 1);
                    rec.finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                eprintln!("Qwen3.8 27B's {what} of {rows} rows: {:.1} us a layer", t.elapsed().as_secs_f64() / 5.0 / 48.0 * 1e6);
            }
        }
    }

    /// A prompt's RoPE, its rows stored into a cache, and its causal attention over the cache (with and without a
    /// window) give the CPU backend's answer: 37 queries after 300 positions (two runs of 256).
    #[test]
    fn a_prompts_rope_store_and_attention_match_the_cpus() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (n_h, n_kv, hd, past, rows) = (8usize, 2usize, 64usize, 300usize, 37usize);
        let (qd, kvd, cap) = (n_h * hd, n_kv * hd, 512usize);
        let mut next = rng(91);
        let ks: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
        let vs: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
        let q: Vec<f32> = (0..rows * qd).map(|_| next()).collect();
        let k: Vec<f32> = (0..rows * kvd).map(|_| next()).collect();
        let v: Vec<f32> = (0..rows * kvd).map(|_| next()).collect();
        let theta = 10000.0f32;
        let positions: Vec<u32> = (past..past + rows).map(|p| p as u32).collect();
        let table: Vec<f32> = positions
            .iter()
            .flat_map(|&pos| (0..hd / 2).flat_map(move |j| {
                let (s, c) = (pos as f32 * theta.powf(-2.0 * j as f32 / hd as f32)).sin_cos();
                [s, c]
            }))
            .collect();
        for window in [None, Some(100)] {
            let mut qc = ggml_rs::Tensor::from_vec(q.clone(), vec![rows, n_h, hd]);
            let mut kc = ggml_rs::Tensor::from_vec(k.clone(), vec![rows, n_kv, hd]);
            ggml_rs::Backend::rope(&cpu, &mut qc, &positions, hd, ggml_rs::RopeType::NeoX, theta, None);
            ggml_rs::Backend::rope(&cpu, &mut kc, &positions, hd, ggml_rs::RopeType::NeoX, theta, None);
            let (mut kcache, mut vcache) = (ks.clone(), vs.clone());
            kcache[past * kvd..(past + rows) * kvd].copy_from_slice(kc.data());
            vcache[past * kvd..(past + rows) * kvd].copy_from_slice(&v);
            let want = ggml_rs::Backend::attention(
                &cpu,
                &qc,
                &ggml_rs::Tensor::from_vec(kcache, vec![cap, n_kv, hd]),
                &ggml_rs::Tensor::from_vec(vcache, vec![cap, n_kv, hd]),
                past + rows,
                0.125,
                past,
                window,
            );
            let mut interleaved = Vec::with_capacity(cap * 2 * kvd);
            for t in 0..cap {
                interleaved.extend_from_slice(&ks[t * kvd..(t + 1) * kvd]);
                interleaved.extend_from_slice(&vs[t * kvd..(t + 1) * kvd]);
            }
            let (qv, kv_, vv, tab, cache) = (b.vec(rows * qd), b.vec(rows * kvd), b.vec(rows * kvd), b.vec(rows * hd), b.vec(cap * 2 * kvd));
            let out = b.vec(b.attention_rows_out_len(rows, n_h, hd, past + rows));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv_, &k);
            DeviceChain::upload(&b, &vv, &v);
            DeviceChain::upload(&b, &tab, &table);
            DeviceChain::upload(&b, &cache, &interleaved);
            // the f32 kernels' (the tensor cores' are checked against them)
            let mut rec = Recorder::new(&b);
            rec.rope_rows(&qv, rows, n_h, hd, &tab, true);
            rec.rope_rows(&kv_, rows, n_kv, hd, &tab, true);
            rec.store_rows(&kv_, &cache, rows, kvd, past, 2 * kvd, 0);
            rec.store_rows(&vv, &cache, rows, kvd, past, 2 * kvd, kvd);
            rec.attention_rows_f32(&qv, &cache, &out, rows, n_h, n_kv, hd, past, window, 0.125);
            rec.read_range(&out, 0, rows * qd);
            rec.read_range(&cache, past * 2 * kvd, kvd);
            let got = Box::new(rec).finish();
            close(&got[1], &kc.data()[..kvd], "the first stored key");
            close(&got[0], want.data(), &format!("a prompt's attention, window {window:?}"));
        }
    }

    /// QSA's attention of a prompt's rows on the tensor cores gives the f32 kernel's within f16's rounding: each
    /// query's kept blocks (of 4 positions) a spread of its visible ones, every visible one where they are no more
    /// than it keeps, its tail block's positions too; GQA, heads 128 and 256 wide, rows a tile's multiple and not.
    #[test]
    fn qsa_attention_on_the_tensor_cores_is_the_f32_kernels() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        let ratio = 4usize;
        for (n_h, n_kv, hd, first, rows, keep) in [(8usize, 2usize, 128usize, 300usize, 37usize, 24usize), (16, 2, 256, 2600, 64, 512), (16, 2, 256, 1000, 100, 2048)] {
            let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, first + rows);
            let mut next = rng((qd + first + keep) as u32);
            let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
            let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
            // a query's kept blocks: every visible one where it keeps as many, else a spread of them (ascending)
            let mut list = vec![0f32; rows * keep];
            for r in 0..rows {
                let visible = (first + r + 1) / ratio;
                let count = visible.min(keep);
                // (a spread, strictly ascending and below the visible: from the first on even rows, the last on odd)
                for i in 0..count {
                    let blk = if visible <= keep { i } else if r % 2 == 0 { i * visible / count } else { visible - 1 - (count - 1 - i) * visible / count };
                    list[r * keep + i] = f32::from_bits(blk as u32);
                }
            }
            let (qv, kv, lv) = (b.vec(rows * qd), b.vec(kv_len * row), b.vec(rows * keep));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv, &cache);
            DeviceChain::upload(&b, &lv, &list);
            let scale = 1.0 / (hd as f32).sqrt();
            let len = b.qsa_attention_out_len(rows, n_h, hd, keep, ratio).max(rows.div_ceil(32) * 32 * qd);
            let (want, got) = (b.vec(len), b.vec(len));
            let mut rec = Recorder::new(&b);
            rec.qsa_attention_f32(&qv, &kv, &lv, &want, rows, n_h, n_kv, hd, first, ratio, keep, scale);
            assert!(rec.qsa_attention_coop(&qv, &kv, &lv, &got, rows, n_h, n_kv, hd, first, ratio, keep, scale), "on the tensor cores");
            rec.read_range(&want, 0, rows * qd);
            rec.read_range(&got, 0, rows * qd);
            let r = Box::new(rec).finish();
            let (want, got) = (&r[0], &r[1]);
            let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(got) * norm(want));
            let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {first}, keeping {keep}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}");
            assert!(cos > 0.99999 && worst <= 4e-3 * top, "{rows} rows after {first} keeping {keep}: cosine {cos}, worst {worst} of {top}");
        }
    }

    /// A step's attention with a KV head's query heads together gives what a workgroup a head gives: groups of 2, 3, 4
    /// and 6, heads 64, 128 and 256 wide, a window's start, a run short of its 256, a single position.
    #[test]
    fn a_steps_attention_by_groups_is_the_heads() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        for (n_h, n_kv, hd, lo, kv_len) in [(24usize, 4usize, 256usize, 0usize, 3000usize), (8, 4, 64, 0, 700), (24, 8, 128, 100, 1111), (4, 1, 256, 0, 513), (16, 4, 128, 40, 41), (12, 2, 256, 0, 1)] {
            let (qd, row, cap) = (n_h * hd, 2 * n_kv * hd, kv_len + 3);
            let mut next = rng((qd + kv_len) as u32);
            let q: Vec<f32> = (0..qd).map(|_| next() * 2.0).collect();
            let cache: Vec<f32> = (0..cap * row).map(|_| next() * 2.0).collect();
            let (qv, kv) = (b.vec(qd), b.vec(cap * row));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv, &cache);
            let scale = 1.0 / (hd as f32).sqrt();
            let len = b.attention_out_len(n_h, hd, cap);
            let (want, got) = (b.vec(len), b.vec(len));
            assert!(attention_group_for(n_h / n_kv, hd, b.gpu.limits.max_compute_workgroup_storage_size), "{n_h} heads of {hd} over {n_kv}: by groups");
            let mut rec = Recorder::new(&b);
            rec.attention_by(&qv, &kv, &want, n_h, n_kv, hd, lo, kv_len, cap, scale, false);
            rec.attention_by(&qv, &kv, &got, n_h, n_kv, hd, lo, kv_len, cap, scale, true);
            rec.read_range(&want, 0, qd);
            rec.read_range(&got, 0, qd);
            let r = Box::new(rec).finish();
            let (want, got) = (&r[0], &r[1]);
            let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            assert!(top > 0.0 && worst <= 2e-5 * top, "{n_h} heads ({n_kv} kv) {hd} wide over {lo}..{kv_len}: worst {worst} of {top}");
        }
    }

    /// A step's attention over its cache's f16 halves gives what the f32 cache gives, to f16's rounding: the halves
    /// made in two goes (rows added since the first), groups of 2, 3 and 6, heads 64, 128 and 256 wide.
    #[test]
    fn a_steps_attention_over_halves_is_the_f32_caches() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        for (n_h, n_kv, hd, lo, kv_len) in [(24usize, 4usize, 256usize, 0usize, 3000usize), (8, 4, 64, 0, 700), (24, 8, 128, 100, 1111)] {
            if !b.attention_halves(n_h, n_kv, hd) {
                return;
            }
            let (qd, row, cap) = (n_h * hd, 2 * n_kv * hd, kv_len + 3);
            let mut next = rng((qd + kv_len) as u32);
            let q: Vec<f32> = (0..qd).map(|_| next() * 2.0).collect();
            let cache: Vec<f32> = (0..cap * row).map(|_| next() * 2.0).collect();
            let (qv, kv, half) = (b.vec(qd), b.vec(cap * row), b.vec(cap * row / 2));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv, &cache);
            let scale = 1.0 / (hd as f32).sqrt();
            let len = b.attention_out_len(n_h, hd, cap);
            let (want, got) = (b.vec(len), b.vec(len));
            let mut rec = Recorder::new(&b);
            rec.attention_by(&qv, &kv, &want, n_h, n_kv, hd, lo, kv_len, cap, scale, false);
            let first = kv_len / 3;
            rec.halve(&kv, &half, 0, first * row);
            rec.halve(&kv, &half, first * row, (kv_len - first) * row);
            rec.attention_halved(&qv, &half, &got, n_h, n_kv, hd, lo, kv_len, cap, scale);
            rec.read_range(&want, 0, qd);
            rec.read_range(&got, 0, qd);
            let r = Box::new(rec).finish();
            let (want, got) = (&r[0], &r[1]);
            let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(got) * norm(want));
            let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            eprintln!("{n_h} heads ({n_kv} kv) {hd} wide over {lo}..{kv_len}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}");
            assert!(cos > 0.99999 && worst <= 4e-3 * top, "{n_h} heads ({n_kv} kv) {hd} wide over {lo}..{kv_len}: cosine {cos}, worst {worst} of {top}");
        }
    }

    /// A few rows' attention with its parts' sums four at a time gives what a sum a load gives: a check's few rows
    /// over a long cache, causal from its start and past it, windowed, full, GQA, heads 64, 128 and 256 wide.
    #[test]
    fn a_few_rows_parts_in_fours_are_the_parts() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        for (n_h, n_kv, hd, past, rows, window, full) in [
            (24usize, 4usize, 256usize, 3000usize, 4usize, None, false),
            (8, 2, 64, 300, 1, None, false),
            (16, 2, 128, 0, 5, None, false),
            (8, 8, 128, 500, 3, Some(200usize), false),
            (4, 1, 256, 513, 7, None, true),
            (6, 3, 128, 40, 9, Some(17), false),
        ] {
            let kv_len = if full { past } else { past + rows };
            let (qd, row) = (n_h * hd, 2 * n_kv * hd);
            let mut next = rng((qd + past + rows) as u32);
            let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
            let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
            let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv, &cache);
            let scale = 1.0 / (hd as f32).sqrt();
            let len = attention_runs_out_len(rows, n_h, hd, kv_len);
            let (want, got) = (b.vec(len), b.vec(len));
            let mut rec = Recorder::new(&b);
            rec.attention_rows_runs_by(&qv, &kv, &want, rows, n_h, n_kv, hd, past, window, scale, full, false);
            rec.attention_rows_runs_by(&qv, &kv, &got, rows, n_h, n_kv, hd, past, window, scale, full, true);
            rec.read_range(&want, 0, rows * qd);
            rec.read_range(&got, 0, rows * qd);
            let r = Box::new(rec).finish();
            let (want, got) = (&r[0], &r[1]);
            let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            assert!(top > 0.0 && worst <= 2e-5 * top, "{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past} (window {window:?}, full {full}): worst {worst} of {top}");
        }
    }

    /// A prompt's attention in one pass (tiled) gives the runs' kernel's: causal from the cache's start and past it,
    /// windowed, full (every query over every position: more positions than queries, and as many), GQA, heads 64,
    /// 128 and 256 wide, rows a tile's multiple and not, chunks of a tile or two and one of them all.
    #[test]
    fn a_tiled_attention_is_the_runs_kernels() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        for (n_h, n_kv, hd, past, rows, window, full, chunk) in [
            (8usize, 2usize, 64usize, 300usize, 37usize, None, false, 64usize),
            (16, 2, 128, 0, 100, None, false, 64),
            (24, 4, 256, 214, 64, None, false, 32),
            (8, 8, 128, 500, 130, Some(200usize), false, 64),
            (8, 8, 128, 333, 333, None, true, 128),
            (4, 4, 64, 777, 100, None, true, 64),
            (4, 1, 256, 513, 70, None, true, 32),
            (6, 3, 128, 40, 300, Some(17), false, 1 << 20),
        ] {
            let kv_len = if full { past } else { past + rows };
            let (qd, row) = (n_h * hd, 2 * n_kv * hd);
            let mut next = rng((qd + past + rows) as u32);
            let q: Vec<f32> = (0..rows * qd).map(|_| next() * 2.0).collect();
            let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * 2.0).collect();
            let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv, &cache);
            let scale = 1.0 / (hd as f32).sqrt();
            let (want, got) = (b.vec(attention_runs_out_len(rows, n_h, hd, kv_len)), b.vec(rows * qd));
            let mut rec = Recorder::new(&b);
            rec.attention_rows_runs(&qv, &kv, &want, rows, n_h, n_kv, hd, past, window, scale, full);
            rec.attention_rows_tiled(&qv, &kv, &got, rows, n_h, n_kv, hd, past, window, scale, full, chunk);
            rec.read_range(&want, 0, rows * qd);
            rec.read(&got);
            let r = Box::new(rec).finish();
            let (want, got) = (&r[0], &r[1]);
            let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(got) * norm(want));
            let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows, past {past}, window {window:?}, full {full}, chunk {chunk}: cosine {cos:.8}, worst {worst:.2e} of {top:.2}");
            assert!(cos > 0.999999 && worst <= 2e-5 * top, "{n_h} heads {hd} wide, {rows} rows, past {past}: cosine {cos}, worst {worst} of {top}");
        }
    }

    /// A prompt chunk's causal attention without tensor cores (`--ignored --nocapture`): the tiled kernel's against the
    /// runs' for Qwen3.8 27B's chunk of 512 (24 heads of 256, 4 kv) and a 128-wide model's, the cache before it short
    /// and long.
    #[test]
    #[ignore = "a measurement"]
    fn measure_chunk_attention_f32() {
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        for (n_h, n_kv, hd, rows) in [(24usize, 4usize, 256usize, 512usize), (32, 8, 128, 512), (24, 4, 256, 128), (24, 4, 256, 64)] {
            for past in [0usize, 2048, 8192, 30000] {
                let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
                let mut next = rng((past + rows) as u32);
                let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
                DeviceChain::upload(&b, &qv, &(0..rows * qd).map(|_| next()).collect::<Vec<_>>());
                DeviceChain::upload(&b, &kv, &(0..kv_len * row).map(|_| next()).collect::<Vec<_>>());
                let scale = 1.0 / (hd as f32).sqrt();
                let time = |f: &dyn Fn(&mut Recorder)| {
                    let mut rec = Recorder::new(&b);
                    f(&mut rec);
                    Box::new(rec).finish();
                    let start = std::time::Instant::now();
                    let mut rec = Recorder::new(&b);
                    for _ in 0..4 {
                        f(&mut rec);
                    }
                    Box::new(rec).finish();
                    start.elapsed().as_secs_f64() / 4.0
                };
                let out = b.vec(attention_runs_out_len(rows, n_h, hd, kv_len));
                let tiled = time(&|rec| rec.attention_rows_tiled(&qv, &kv, &out, rows, n_h, n_kv, hd, past, None, scale, false, 1 << 20));
                let runs = time(&|rec| rec.attention_rows_runs(&qv, &kv, &out, rows, n_h, n_kv, hd, past, None, scale, false));
                eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past}: tiled {:.2} ms, runs {:.2} ms", tiled * 1e3, runs * 1e3);
            }
        }
    }

    /// A prompt's attention without tensor cores (`--ignored --nocapture`): the tiled kernel's against the runs' at Qwen
    /// Image's 1024x1024 (32 heads of 128, 4,096 queries over 4,200 positions), the tiled alone at LTX's 17,408.
    #[test]
    #[ignore = "a measurement"]
    fn measure_tiled_attention() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        let (n_h, hd) = (32usize, 128usize);
        for (rows, kv_len, runs_too) in [(4096usize, 4200usize, true), (17408, 17408, false)] {
            let (qd, row) = (n_h * hd, 2 * n_h * hd);
            let mut next = rng(rows as u32);
            let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
            DeviceChain::upload(&b, &qv, &(0..rows * qd).map(|_| next()).collect::<Vec<_>>());
            DeviceChain::upload(&b, &kv, &(0..kv_len * row).map(|_| next()).collect::<Vec<_>>());
            let scale = 1.0 / (hd as f32).sqrt();
            let flops = 4.0 * rows as f64 * kv_len as f64 * qd as f64;
            let time = |f: &dyn Fn(&mut Recorder)| {
                let mut rec = Recorder::new(&b);
                f(&mut rec);
                Box::new(rec).finish();
                let start = std::time::Instant::now();
                let mut rec = Recorder::new(&b);
                for _ in 0..3 {
                    f(&mut rec);
                }
                Box::new(rec).finish();
                start.elapsed().as_secs_f64() / 3.0
            };
            let out = b.vec(rows * qd);
            let tiled = time(&|rec| rec.attention_rows_f32_masked(&qv, &kv, &out, rows, n_h, n_h, hd, kv_len, None, scale, true));
            eprintln!("{rows} queries over {kv_len}: tiled {:.1} ms ({:.1} TFLOP/s)", tiled * 1e3, flops / tiled / 1e12);
            if runs_too {
                let out = b.vec(attention_runs_out_len(rows, n_h, hd, kv_len));
                let runs = time(&|rec| rec.attention_rows_runs(&qv, &kv, &out, rows, n_h, n_h, hd, kv_len, None, scale, true));
                eprintln!("{rows} queries over {kv_len}: runs {:.1} ms ({:.1} TFLOP/s)", runs * 1e3, flops / runs / 1e12);
            }
        }
    }

    /// A prompt's attention on the tensor cores gives the f32 kernels' within f16's rounding: GQA, heads 64, 128 and
    /// 256 wide, from the cache's start and past it, the rows a tile's multiple and not, its scores spread and peaked.
    #[test]
    fn a_prompts_attention_on_the_tensor_cores_is_the_f32_kernels() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        // (each as it is, then with keys far past the rest planted)
        let shapes = [(8usize, 2usize, 64usize, 300usize, 37usize, 1.0f32), (16, 2, 128, 0, 100, 2.0), (24, 4, 256, 214, 64, 1.0), (24, 4, 256, 1000, 150, 3.0)];
        for (planted, (n_h, n_kv, hd, past, rows, spread)) in [false, true].into_iter().flat_map(|p| shapes.into_iter().map(move |s| (p, s))) {
            let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
            let mut next = rng((qd + past) as u32);
            let q: Vec<f32> = (0..rows * qd).map(|_| next() * spread).collect();
            let mut cache: Vec<f32> = (0..kv_len * row).map(|_| next() * spread).collect();
            // (three keys late in the cache made long: a score a dozen or more past a query's largest before it, or as far
            // under: the one-pass kernel's reference moved and its sums scaled down; f16 keeps so large a score to a
            // hundredth, so the answers agree less closely)
            if planted {
                for at in [kv_len / 3, kv_len / 2 + 7, kv_len - 9] {
                    for v in &mut cache[at * row..at * row + n_kv * hd] {
                        *v *= 36.0 / (spread * spread);
                    }
                }
            }
            let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv, &cache);
            let scale = 1.0 / (hd as f32).sqrt();
            let len = b.attention_rows_out_len(rows, n_h, hd, kv_len);
            let (want, got) = (b.vec(len), b.vec(len));
            let mut rec = Recorder::new(&b);
            rec.attention_rows_f32(&qv, &kv, &want, rows, n_h, n_kv, hd, past, None, scale);
            assert!(rec.attention_rows_coop(&qv, &kv, &got, rows, n_h, n_kv, hd, past, None, scale), "on the tensor cores");
            rec.read_range(&want, 0, rows * qd);
            rec.read_range(&got, 0, rows * qd);
            let r = Box::new(rec).finish();
            let (want, got) = (&r[0], &r[1]);
            let dot: f64 = got.iter().zip(want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(got) * norm(want));
            let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(want).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past}{}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}", if planted { ", keys planted" } else { "" });
            let (least, most) = if planted { (0.9999, 3e-2) } else { (0.99999, 4e-3) };
            assert!(cos > least && worst <= most * top, "{n_h} heads {hd} wide, {rows} rows after {past}: cosine {cos}, worst {worst} of {top}");
        }
    }

    /// A prompt's attention on the tensor cores (`--ignored --nocapture`): Qwen3.8 27B's (24 heads of 256, 4 kv) for a
    /// chunk of 512 at several places, and what it does a second (its scores twice and its values once).
    #[test]
    #[ignore = "a measurement"]
    fn measure_prompt_attention() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        // (OAIY_ATT_HD: heads that wide, as many more of them)
        let hd: usize = std::env::var("OAIY_ATT_HD").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        let (n_h, n_kv) = (24 * 256 / hd, 4 * 256 / hd);
        // (OAIY_ATT_ROWS: the chunk that long)
        let rows: usize = std::env::var("OAIY_ATT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        for past in [0usize, 1024, 4096, 15360] {
            let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
            let mut next = rng(past as u32 + 5);
            let (qv, kv) = (b.vec(rows * qd), b.vec(kv_len * row));
            DeviceChain::upload(&b, &qv, &(0..rows * qd).map(|_| next()).collect::<Vec<_>>());
            DeviceChain::upload(&b, &kv, &(0..kv_len * row).map(|_| next()).collect::<Vec<_>>());
            let out = b.vec(b.attention_rows_out_len(rows, n_h, hd, kv_len));
            let scale = 1.0 / (hd as f32).sqrt();
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    assert!(rec.attention_rows_coop(&qv, &kv, &out, rows, n_h, n_kv, hd, past, None, scale));
                }
                rec.read_range(&out, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
            // each query's keys up to its own: scores (twice) and values, 2 FLOPs a multiply-add
            let pairs: f64 = (0..rows).map(|r| (past + r + 1) as f64).sum();
            let flops = 3.0 * 2.0 * pairs * (n_h * hd) as f64;
            eprintln!("{rows} rows after {past}: {ms:.3} ms a layer ({:.0} TFLOPS)", flops / ms / 1e9);
        }
    }

    /// RoPE, a store into a cache and attention over it give the CPU backend's answer (GQA): a few positions in, and
    /// past 512 (three runs of the split attention put together).
    #[test]
    fn rope_store_and_attention_match_the_cpus() {
        for (cap, past) in [(16usize, 9usize), (640, 530)] {
            rope_store_and_attention_case(cap, past);
        }
    }

    fn rope_store_and_attention_case(cap: usize, past: usize) {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (n_h, n_kv, hd) = (8usize, 2usize, 64usize);
        let (qd, kvd) = (n_h * hd, n_kv * hd);
        let mut next = rng(77);
        // the cache's earlier rows, the new token's q, k and v
        let ks: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
        let vs: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
        let q: Vec<f32> = (0..qd).map(|_| next()).collect();
        let k: Vec<f32> = (0..kvd).map(|_| next()).collect();
        let v: Vec<f32> = (0..kvd).map(|_| next()).collect();
        let theta = 10000.0f32;
        let table: Vec<f32> = (0..hd / 2)
            .flat_map(|j| {
                let (s, c) = (past as f32 * theta.powf(-2.0 * j as f32 / hd as f32)).sin_cos();
                [s, c]
            })
            .collect();
        for neox in [false, true] {
            let rope_type = if neox { ggml_rs::RopeType::NeoX } else { ggml_rs::RopeType::Normal };
            // CPU
            let mut qc = ggml_rs::Tensor::from_vec(q.clone(), vec![1, n_h, hd]);
            let mut kc = ggml_rs::Tensor::from_vec(k.clone(), vec![1, n_kv, hd]);
            ggml_rs::Backend::rope(&cpu, &mut qc, &[past as u32], hd, rope_type, theta, None);
            ggml_rs::Backend::rope(&cpu, &mut kc, &[past as u32], hd, rope_type, theta, None);
            let mut kcache = ks.clone();
            let mut vcache = vs.clone();
            kcache[past * kvd..(past + 1) * kvd].copy_from_slice(kc.data());
            vcache[past * kvd..(past + 1) * kvd].copy_from_slice(&v);
            let want = ggml_rs::Backend::attention(
                &cpu,
                &qc,
                &ggml_rs::Tensor::from_vec(kcache, vec![cap, n_kv, hd]),
                &ggml_rs::Tensor::from_vec(vcache, vec![cap, n_kv, hd]),
                past + 1,
                0.125,
                past,
                None,
            );
            // the chain: the cache interleaved a row at a time (K then V)
            let mut interleaved = Vec::with_capacity(cap * 2 * kvd);
            for t in 0..cap {
                interleaved.extend_from_slice(&ks[t * kvd..(t + 1) * kvd]);
                interleaved.extend_from_slice(&vs[t * kvd..(t + 1) * kvd]);
            }
            let (qv, kv_, vv, tab, cache, out) = (b.vec(qd), b.vec(kvd), b.vec(kvd), b.vec(hd), b.vec(cap * 2 * kvd), b.vec(b.attention_out_len(n_h, hd, cap)));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv_, &k);
            DeviceChain::upload(&b, &vv, &v);
            DeviceChain::upload(&b, &tab, &table);
            DeviceChain::upload(&b, &cache, &interleaved);
            let mut rec = b.begin();
            rec.rope(&qv, n_h, hd, &tab, neox);
            rec.rope(&kv_, n_kv, hd, &tab, neox);
            rec.store(&kv_, &cache, past * 2 * kvd);
            rec.store(&vv, &cache, past * 2 * kvd + kvd);
            rec.attention(&qv, &cache, &out, n_h, n_kv, hd, 0, past + 1, cap, 0.125);
            rec.read_range(&out, 0, qd);
            rec.read_range(&cache, past * 2 * kvd, kvd);
            rec.read(&qv);
            let got = rec.finish();
            close(&got[2], qc.data(), "the rotated query");
            close(&got[1], kc.data(), "the stored key");
            close(&got[0], want.data(), "the attention");
        }
    }

    /// A gated delta net's conv and recurrence give the host's answer (`Backend::delta_net_step`, the CPU's): the
    /// output, the conv state and the recurrent state after a decode step and after a run of tokens, from states the
    /// host made, at a small shape and at Qwen3.8 27B's (48 value heads on 16 key heads of 128); runs of a few tokens
    /// (one kernel) and of a prompt's (three passes).
    #[test]
    fn a_delta_net_matches_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        for (nv, nk, dim, kern, rows, sigmoid) in [(8usize, 4usize, 32usize, 4usize, 1usize, false), (8, 4, 32, 4, 7, true), (48, 16, 128, 4, 1, false), (48, 16, 128, 4, 5, true), (8, 4, 32, 4, 37, false), (48, 16, 128, 4, 100, true), (32, 16, 128, 4, 64, false)] {
            let ch = 2 * nk * dim + nv * dim;
            let mut next = rng((nv * 31 + rows) as u32);
            let mut vals = |n: usize, s: f32| (0..n).map(|_| next() * s).collect::<Vec<f32>>();
            let qkv = vals(rows * ch, 1.0);
            let z = vals(rows * nv * dim, 1.0);
            let ba = vals(rows * 2 * nv, 2.0);
            let cw = vals(ch * kern, 0.5);
            let a = vals(nv, 1.0).iter().map(|v| -v.abs() - 0.1).collect::<Vec<f32>>();
            let dt = vals(nv, 1.0);
            let nm = vals(dim, 1.0).iter().map(|v| v + 1.0).collect::<Vec<f32>>();
            let conv0 = vals((kern - 1) * ch, 1.0);
            let state0 = vals(nv * dim * dim, 0.1);
            let (scale, eps) = (1.0 / (dim as f32).sqrt(), 1e-6f32);
            let t = |d: &[f32], shape: Vec<usize>| ggml_rs::Tensor::from_vec(d.to_vec(), shape);
            let mut conv_c = t(&conv0, vec![kern - 1, ch]);
            let mut state_c = t(&state0, vec![nv, dim, dim]);
            // the host's step, or the backend's (its state kept on the GPU between calls)
            let step = |be: &dyn ggml_rs::Backend, conv: &mut ggml_rs::Tensor, state: &mut ggml_rs::Tensor| {
                let (q, zz, bb, ww) = (t(&qkv, vec![rows, ch]), t(&z, vec![rows, nv * dim]), t(&ba, vec![rows, 2 * nv]), t(&cw, vec![ch, kern]));
                let (aa, dd, nn) = (t(&a, vec![nv]), t(&dt, vec![nv]), t(&nm, vec![dim]));
                if sigmoid {
                    be.delta_net_step_sigmoid(&q, &zz, &bb, &ww, &aa, &dd, &nn, conv, state, rows, nv, nk, dim, dim, nv / nk, scale, eps)
                } else {
                    be.delta_net_step(&q, &zz, &bb, &ww, &aa, &dd, &nn, conv, state, rows, nv, nk, dim, dim, nv / nk, scale, eps)
                }
            };
            let want = step(&cpu, &mut conv_c, &mut state_c);
            // twice each way: the second from the first's state
            let (mut conv_g, mut state_g) = (t(&conv0, vec![kern - 1, ch]), t(&state0, vec![nv, dim, dim]));
            let (mut conv_h, mut state_h) = (t(&conv0, vec![kern - 1, ch]), t(&state0, vec![nv, dim, dim]));
            for _ in 0..2 {
                let got = step(&b, &mut conv_g, &mut state_g);
                let host = step(&cpu, &mut conv_h, &mut state_h);
                close(got.data(), host.data(), &format!("{nv} heads, {rows} rows: the backend's step"));
                assert!(state_g.is_device() && conv_g.is_device(), "the state stays on the GPU");
            }
            close(state_g.to_host().data(), state_h.data(), &format!("{nv} heads, {rows} rows: the backend's state"));
            close(conv_g.to_host().data(), conv_h.data(), &format!("{nv} heads, {rows} rows: the backend's conv"));
            let up = |d: &[f32]| {
                let v = b.vec(d.len());
                DeviceChain::upload(&b, &v, d);
                v
            };
            let (qkv_d, z_d, ba_d, cw_d, a_d, dt_d, nm_d, conv_d) = (up(&qkv), up(&z), up(&ba), up(&cw), up(&a), up(&dt), up(&nm), up(&conv0));
            let state_d = up(&state0);
            let (conv_out, out) = (b.vec(rows * ch), b.vec(rows * nv * dim));
            let mut rec = b.begin();
            rec.ssm_conv(&qkv_d, &cw_d, &conv_d, &conv_out, rows, ch, kern);
            rec.delta_net(&conv_out, &z_d, &ba_d, &a_d, &dt_d, &nm_d, &state_d, &out, DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: scale, eps, sigmoid_gate: sigmoid });
            rec.read(&out);
            rec.read(&conv_d);
            let got = rec.finish();
            let what = format!("{nv} heads of {dim}, {rows} rows");
            close(&got[0], want.data(), &format!("{what}: the output"));
            close(&got[1], conv_c.data(), &format!("{what}: the conv state"));
            // the recurrent state, through the alias the host would read it by
            let state = b.alias(&state_d, vec![nv, dim, dim]);
            close(state.to_host().data(), state_c.data(), &format!("{what}: the state"));
            assert!(b.aliased(&state).is_some_and(|v| Arc::ptr_eq(&v.inner, &state_d.inner)));
        }
    }

    /// What a prompt's delta net takes (`--ignored --nocapture`): Qwen3.8 27B's (48 value heads on 16 key heads of 128)
    /// for 512 tokens, as one kernel and in three passes.
    #[test]
    #[ignore = "a measurement"]
    fn measure_delta_net_rows() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        let (nv, nk, dim, rows) = (48usize, 16usize, 128usize, 512usize);
        let ch = 2 * nk * dim + nv * dim;
        let mut next = rng(3);
        let mut up = |n: usize, s: f32| {
            let v = b.vec(n);
            DeviceChain::upload(&b, &v, &(0..n).map(|_| next() * s).collect::<Vec<f32>>());
            v
        };
        let (cv, z, ba, a, dt, nm, state, out) = (up(rows * ch, 1.0), up(rows * nv * dim, 1.0), up(rows * 2 * nv, 2.0), up(nv, 1.0), up(nv, 1.0), up(dim, 1.0), up(nv * dim * dim, 0.1), up(rows * nv * dim, 0.0));
        let d = DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: 1.0 / (dim as f32).sqrt(), eps: 1e-6, sigmoid_gate: false };
        let words = [nv as u32, nk as u32, dim as u32, dim as u32, rows as u32, d.scale_q.to_bits(), d.eps.to_bits(), 0];
        for three in [false, true] {
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..4 {
                    if three {
                        rec.delta_net_rows(&cv, &z, &ba, &a, &dt, &nm, &state, &out, &d, &words);
                    } else {
                        rec.dispatch_wide("chain-delta-net-128", &delta_net_one(dim), [buffer(&cv), buffer(&z), buffer(&ba), buffer(&a), buffer(&dt), buffer(&nm), buffer(&state), buffer(&out)], &words, (nv as u32, 1, 1));
                    }
                }
                rec.read_range(&out, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            eprintln!("{}: {:.3} ms a layer", if three { "three passes" } else { "one kernel" }, t.elapsed().as_secs_f64() / 12.0 * 1e3);
        }
    }

    /// An alias reads its vector as it is when read, and a clone of it is a copy that does not follow the vector.
    #[test]
    fn an_alias_reads_its_vector_and_a_clone_is_a_copy() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let v = b.vec(8);
        DeviceChain::upload(&b, &v, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let t = b.alias(&v, vec![2, 2, 2]);
        assert_eq!(t.to_host().data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let copy = t.clone();
        DeviceChain::upload(&b, &v, &[0.0; 8]);
        assert_eq!(t.to_host().data(), &[0.0; 8]);
        assert_eq!(copy.to_host().data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(b.aliased(&copy).is_some_and(|c| !Arc::ptr_eq(&c.inner, &v.inner)));
        assert!(b.aliased(&ggml_rs::Tensor::zeros(vec![8])).is_none());
        b.zero(&b.aliased(&copy).unwrap());
        assert_eq!(copy.to_host().data(), &[0.0; 8]);
    }

    /// Qwen3.5's partial RoPE, its q/gate split, gate and SwiGLU of two weights, and an f32 matmul give the CPU's
    /// answers.
    #[test]
    fn qwen35s_small_ops_match_the_cpus() {
        use ggml_rs::Backend;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (rows, heads, hd, rot, theta) = (3usize, 4usize, 32usize, 8usize, 1e7f32);
        let mut next = rng(77);
        let x: Vec<f32> = (0..rows * heads * 2 * hd).map(|_| next()).collect();
        let up = |d: &[f32]| {
            let v = b.vec(d.len());
            DeviceChain::upload(&b, &v, d);
            v
        };
        let xd = up(&x);
        let (q, g) = (b.vec(rows * heads * hd), b.vec(rows * heads * hd));
        let past = 17usize;
        let table: Vec<f32> = (past..past + rows)
            .flat_map(|pos| (0..rot / 2).flat_map(move |k| {
                let (s, c) = (pos as f32 * theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                [s, c]
            }))
            .collect();
        let td = up(&table);
        let gated = b.vec(rows * heads * hd);
        let (gate_v, up_v, act) = (up(&x[..64]), up(&x[64..128]), b.vec(64));
        let (n, k) = (5usize, 48usize);
        let wf: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let (wd, xf, yf) = (up(&wf), up(&x[..2 * k]), b.vec(2 * n));
        let mut rec = b.begin();
        rec.copy_cols(&xd, &q, rows * heads, hd, 2 * hd, 0);
        rec.copy_cols(&xd, &g, rows * heads, hd, 2 * hd, hd);
        rec.rope_partial_rows(&q, rows, heads, hd, rot, &td);
        rec.mul_sigmoid(&q, &g, &gated, rows * heads * hd);
        rec.silu_mul(&gate_v, &up_v, &act, 64);
        rec.matmul_f32_rows(&wd, n, k, &xf, &yf, 2);
        for v in [&q, &g, &gated, &act, &yf] {
            rec.read(v);
        }
        let got = rec.finish();
        let (mut qh, mut gh) = (Vec::new(), Vec::new());
        for r in x.chunks_exact(2 * hd) {
            qh.extend_from_slice(&r[..hd]);
            gh.extend_from_slice(&r[hd..]);
        }
        let mut qt = ggml_rs::Tensor::from_vec(qh, vec![rows, heads, hd]);
        let positions: Vec<u32> = (past..past + rows).map(|p| p as u32).collect();
        cpu.rope_partial_neox(&mut qt, &positions, hd, rot, theta);
        close(&got[0], qt.data(), "the partially rotated query");
        close(&got[1], &gh, "the gate half");
        let mut gt = qt.clone();
        cpu.mul_sigmoid_inplace(&mut gt, &ggml_rs::Tensor::from_vec(gh, vec![rows, heads, hd]));
        close(&got[2], gt.data(), "the gated output");
        let sw = cpu.silu_mul(&ggml_rs::Tensor::from_vec(x[..64].to_vec(), vec![64]), &ggml_rs::Tensor::from_vec(x[64..128].to_vec(), vec![64]));
        close(&got[3], sw.data(), "the SwiGLU");
        let mut want = Vec::new();
        for r in 0..2 {
            for o in 0..n {
                want.push((0..k).map(|i| wf[o * k + i] * x[r * k + i]).sum::<f32>());
            }
        }
        close(&got[4], &want, "the f32 matmul");
    }

    /// Flash-Next's hyper-connection ops give the host's: the per-stream norm, the gates, the mix, the write-back, and
    /// a weighted term read from the device.
    #[test]
    fn hyper_connection_ops_match_the_hosts() {
        use ggml_rs::Backend;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (rows, streams, d, rank, writes) = (3usize, 4usize, 64usize, 8usize, 4usize);
        let mut next = rng(5);
        let mut vals = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();
        let x = vals(rows * streams * d);
        let w = vals(streams * d);
        let t0 = vals(rows * (rank + writes));
        let logits = vals(rows * streams * d);
        let y = vals(rows * d);
        let post0 = vals(rows * streams);
        let up = |v: &[f32]| {
            let dv = b.vec(v.len());
            DeviceChain::upload(&b, &dv, v);
            dv
        };
        let t = |v: &[f32], shape: Vec<usize>| ggml_rs::Tensor::from_vec(v.to_vec(), shape);
        let (xd, wd, td, ld, yd, pd) = (up(&x), up(&w), up(&t0), up(&logits), up(&y), up(&post0));
        let (normed, post, mixed) = (b.vec(rows * streams * d), b.vec(rows * writes), b.vec(rows * d));
        let acc = up(&y);
        let weights = up(&[0.5, -1.25, 2.0]);
        let mut rec = b.begin();
        rec.rmsnorm_streams(&xd, &wd, &normed, rows, streams, 1e-6);
        rec.hc_gates(&td, &post, rows, rank, writes, streams);
        rec.hc_mix(&ld, &normed, &mixed, rows, streams, d);
        rec.stream_apply(&xd, &yd, &pd, rows, streams, d);
        rec.axpy_at(&acc, &yd, &weights, 1, rows * d);
        for v in [&normed, &td, &post, &mixed, &xd, &acc] {
            rec.read(v);
        }
        let got = rec.finish();
        let want_normed = cpu.hc_norm(&t(&x, vec![rows, streams * d]), &t(&w, vec![streams * d]), streams, 1e-6);
        close(&got[0], want_normed.data(), "the per-stream norm");
        let mut tt = t(&t0, vec![rows, rank + writes]);
        let want_post = cpu.hc_gates(&mut tt, rank, writes, streams);
        close(&got[1], tt.data(), "the gates' input");
        close(&got[2], want_post.data(), "the write weights");
        let want_mix = cpu.hc_mix(&t(&logits, vec![rows, streams * d]), &want_normed, streams);
        close(&got[3], want_mix.data(), "the mix");
        let mut xs = t(&x, vec![rows, streams * d]);
        cpu.stream_apply(&mut xs, &t(&y, vec![rows, d]), &t(&post0, vec![rows, streams]), streams);
        close(&got[4], xs.data(), "the write-back");
        let want_acc: Vec<f32> = y.iter().map(|v| v + -1.25 * v).collect();
        close(&got[5], &want_acc, "the weighted term");
    }
    /// A prompt's f32 matmul (64x64 tiles, `k` split where the tiles are few) gives the CPU's sums: shapes off the tiles'
    /// edges, a long `k` over few outputs (Qwen3.8-Flash-Next's hyper-connections' and router's), a short one over many;
    /// twice, the second from the pool's scratch as the first left it.
    #[test]
    fn a_prompts_f32_matmul_tiles_and_splits_as_the_cpu() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(91);
        for (n, k, rows) in [(5usize, 37usize, 3usize), (70, 100, 65), (324, 10240, 70), (513, 2560, 129), (1030, 324, 64), (96, 5120, 4), (96, 5120, 2), (10, 64, 8)] {
            let w: Vec<f32> = (0..n * k).map(|_| r()).collect();
            let x: Vec<f32> = (0..rows * k).map(|_| r()).collect();
            let want: Vec<f32> = (0..rows).flat_map(|i| (0..n).map(|o| (0..k).map(|j| w[o * k + j] as f64 * x[i * k + j] as f64).sum::<f64>() as f32).collect::<Vec<_>>()).collect();
            let (wd, xd, yd) = (b.vec(n * k), b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &wd, &w);
            DeviceChain::upload(&b, &xd, &x);
            for _ in 0..2 {
                let mut rec = b.begin();
                rec.keep_groups(false);
                rec.matmul_f32_rows(&wd, n, k, &xd, &yd, rows);
                rec.read(&yd);
                let got = rec.finish().pop().unwrap();
                let scale = (k as f32).sqrt();
                for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                    assert!((g - e).abs() <= 1e-4 * scale, "[{n}, {k}] of {rows} rows [{i}]: {g} against {e}");
                }
            }
        }
    }

    /// A diffusion transformer's ops as the host computes them: the modulated layer norm (with and without a shift,
    /// rows of 4,096 and of 100), the gated residual (its gate's tanh or not), GELU, and a BF16 weight's f16 rounding
    /// (its tiny values to the nearest f16, a value past f16's range refused).
    #[test]
    fn a_diffusion_transformers_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(77);
        for (rows, n) in [(37usize, 4096usize), (5, 100)] {
            let x: Vec<f32> = (0..rows * n).map(|_| r() * 3.0 + 0.5).collect();
            let mods: Vec<f32> = (0..4 * n).map(|_| r()).collect();
            let (xd, md, out) = (b.vec(rows * n), b.vec(4 * n), b.vec(rows * n));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &md, &mods);
            for shift in [None, Some(3 * n)] {
                let mut rec = b.begin();
                rec.layernorm_mod_rows(&xd, &out, rows, n, &md, n, shift, 1e-6);
                rec.read(&out);
                let got = rec.finish().pop().unwrap();
                for row in 0..rows {
                    let v = &x[row * n..(row + 1) * n];
                    let mean = v.iter().map(|&a| a as f64).sum::<f64>() / n as f64;
                    let var = v.iter().map(|&a| (a as f64 - mean).powi(2)).sum::<f64>() / n as f64;
                    for i in 0..n {
                        let want = (v[i] as f64 - mean) / (var + 1e-6).sqrt() * (1.0 + mods[n + i] as f64) + shift.map_or(0.0, |s| mods[s + i] as f64);
                        let g = got[row * n + i] as f64;
                        assert!((g - want).abs() <= 1e-4 * (1.0 + want.abs()), "layer norm row {row} [{i}] shift {shift:?}: {g} against {want}");
                    }
                }
            }
            for tanh in [true, false] {
                let y: Vec<f32> = (0..rows * n).map(|_| r()).collect();
                let yd = b.vec(rows * n);
                DeviceChain::upload(&b, &yd, &y);
                DeviceChain::upload(&b, &xd, &x);
                let mut rec = b.begin();
                rec.add_gated_rows(&xd, &yd, rows, n, &md, 2 * n, tanh);
                rec.read(&xd);
                let got = rec.finish().pop().unwrap();
                for (i, g) in got.iter().enumerate() {
                    let gate = mods[2 * n + i % n];
                    let want = x[i] + y[i] * if tanh { gate.tanh() } else { gate };
                    assert!((g - want).abs() <= 1e-5 * (1.0 + want.abs()), "gated residual [{i}] tanh {tanh}: {g} against {want}");
                }
            }
        }
        let x: Vec<f32> = (0..1000).map(|_| r() * 6.0).collect();
        let (xd, yd) = (b.vec(1000), b.vec(1000));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.gelu(&xd, &yd, 1000);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for (g, &v) in got.iter().zip(&x) {
            let want = 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v * v * v)).tanh());
            assert!((g - want).abs() <= 1e-5 * (1.0 + want.abs()), "gelu({v}): {g} against {want}");
        }
        // a BF16 weight's tiny values to the nearest f16; a value past f16's range refused
        let w = [1.5f32, -3.0e-7, 7.0e-9, 0.333_333_34, 65504.0, -2.0];
        let wd = b.vec_f16_rounded(&w).expect("values within f16's range");
        let mut rec = b.begin();
        rec.read(&wd);
        let words = rec.finish().pop().unwrap();
        let back: Vec<f32> = words.iter().flat_map(|v| [half::f16::from_bits(v.to_bits() as u16).to_f32(), half::f16::from_bits((v.to_bits() >> 16) as u16).to_f32()]).collect();
        assert_eq!(back, w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect::<Vec<_>>());
        assert!(b.vec_f16_rounded(&[1.0, 70000.0]).is_none(), "past f16's range");
        assert!(b.vec_f16_rounded(&[1.0, f32::NAN]).is_none(), "not a number");
    }

    /// ComfyUI's W4A8 decoded on the device as the host decodes it: codes of a random codebook, FP8 group scales (a
    /// subnormal among them), rows' scales; not rotated, and rotated in groups of 16 and 256.
    #[test]
    fn w4a8s_decode_is_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(113);
        let e4m3 = |v: u8| -> f32 {
            let (e, m) = (((v >> 3) & 15) as i32, (v & 7) as f32);
            let x = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
            if v & 128 != 0 { -x } else { x }
        };
        for (rows, cols, rotation) in [(5usize, 48usize, 0usize), (3, 64, 16), (70, 512, 256)] {
            let codes: Vec<u8> = (0..rows * cols / 2).map(|_| ((r() + 1.0) * 127.9) as u8).collect();
            // (scales up to e4m3's 2^3 or so, a subnormal at the first)
            let rel: Vec<u8> = (0..rows * cols / 16).map(|i| if i == 0 { 3 } else { 0x30 + ((r() + 1.0) * 15.9) as u8 }).collect();
            let channel: Vec<f32> = (0..rows).map(|_| r() * 0.01).collect();
            let book: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * (1.0 + 0.1 * r())).collect();
            // the host's: the values, then each group rotated
            let mut want: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let (row, col) = (i / cols, i % cols);
                    let byte = codes[row * cols / 2 + col / 2];
                    let code = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                    (book[code as usize] * e4m3(rel[row * cols / 16 + col / 16])).round_ties_even().clamp(-127.0, 127.0) * channel[row]
                })
                .collect();
            if rotation > 0 {
                const H: [[f32; 4]; 4] = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
                let entry = |mut a: usize, mut c: usize| {
                    let mut v = 1.0 / (rotation as f32).sqrt();
                    while a != 0 || c != 0 {
                        v *= H[a % 4][c % 4];
                        a /= 4;
                        c /= 4;
                    }
                    v
                };
                for group in want.chunks_mut(rotation) {
                    let x = group.to_vec();
                    for (c, y) in group.iter_mut().enumerate() {
                        *y = (0..rotation).map(|a| x[a] * entry(a, c)).sum();
                    }
                }
            }
            let words = |bytes: &[u8]| -> Vec<f32> { bytes.chunks(4).map(|c| { let mut w = [0u8; 4]; w[..c.len()].copy_from_slice(c); f32::from_bits(u32::from_le_bytes(w)) }).collect() };
            let (cw, rw) = (words(&codes), words(&rel));
            let (cd, rd, chd, bd, out) = (b.vec(cw.len()), b.vec(rw.len()), b.vec(rows), b.vec(16), b.vec(rows * cols / 2));
            DeviceChain::upload(&b, &cd, &cw);
            DeviceChain::upload(&b, &rd, &rw);
            DeviceChain::upload(&b, &chd, &channel);
            DeviceChain::upload(&b, &bd, &book);
            let mut rec = b.begin();
            rec.w4a8_f16(&cd, &rd, &chd, &bd, rows, cols, rotation, &out);
            rec.read(&out);
            let got: Vec<f32> = rec.finish().pop().unwrap().iter().flat_map(|w| [half::f16::from_bits(w.to_bits() as u16).to_f32(), half::f16::from_bits((w.to_bits() >> 16) as u16).to_f32()]).collect();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-3 * w.abs() + 1e-6, "{rows}x{cols} rotated {rotation}: [{i}] {g} against {w}");
            }
        }
    }

    /// A sparse decoder's gathers as the host makes them: rows picked by an index (from an offset into it, a missing one
    /// zeros), and each channel repeated and added.
    #[test]
    fn gathered_and_repeated_rows_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(149);
        let (src, c, rows, first, repeat) = (37usize, 12usize, 50usize, 7usize, 3usize);
        let x: Vec<f32> = (0..src * c).map(|_| r()).collect();
        // (every source row, and the missing one, some twice)
        let index: Vec<u32> = (0..first + rows).map(|i| ((i * 13) % (src + 1)) as u32).collect();
        let (xd, id, out) = (b.vec(x.len()), b.vec(index.len()), b.vec(rows * c));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &id, &index.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>());
        let acc: Vec<f32> = (0..rows * c * repeat).map(|_| r()).collect();
        let ad = b.vec(acc.len());
        DeviceChain::upload(&b, &ad, &acc);
        let mut rec = b.begin();
        rec.gather_rows(&xd, &id, &out, rows, c, first, src);
        rec.repeat_cols_add_rows(&out, &ad, rows, c, repeat);
        rec.read(&out);
        rec.read(&ad);
        let got = rec.finish();
        for i in 0..rows * c {
            let s = index[first + i / c] as usize;
            let want = if s < src { x[s * c + i % c] } else { 0. };
            assert_eq!(got[0][i], want, "gathered [{i}]");
        }
        for i in 0..rows * c * repeat {
            let w = c * repeat;
            let want = acc[i] + got[0][(i / w) * c + (i % w) / repeat];
            assert_eq!(got[1][i], want, "repeated [{i}]");
        }
    }

    /// The speech codec's ops as the host computes them: a causal 1-D convolution (its padding all before: 7 taps 3
    /// apart), a depthwise causal one, SnakeBeta, and a clamp.
    #[test]
    fn a_speech_codecs_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(137);
        let (cin, cout, k, dil, len) = (24usize, 20usize, 7usize, 3usize, 40usize);
        let wt: Vec<f32> = (0..cout * cin * k).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
        let x: Vec<f32> = (0..len * cin).map(|_| r()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let wd = b.conv1d_weights(&wt, cout, cin, k).expect("the weights");
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(len * cout), b.vec(cout));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv1d_padded_rows(&wd, &bd, cout, cin, k, dil, (k - 1) * dil, &xd, len, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for t in 0..len {
            for co in 0..cout {
                let mut want = bias[co] as f64;
                for c in 0..cin {
                    for j in 0..k {
                        let s = t as isize + (j * dil) as isize - ((k - 1) * dil) as isize;
                        if s >= 0 {
                            want += wt[(co * cin + c) * k + j] as f64 * x[s as usize * cin + c] as f64;
                        }
                    }
                }
                let g = got[t * cout + co] as f64;
                assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "causal conv at {t}, {co}: {g} against {want}");
            }
        }
        let (c, k, len) = (12usize, 7usize, 30usize);
        let w: Vec<f32> = (0..c * k).map(|_| r()).collect();
        let bias: Vec<f32> = (0..c).map(|_| r()).collect();
        let x: Vec<f32> = (0..len * c).map(|_| r()).collect();
        let (wd, bd, xd, yd) = (b.vec(w.len()), b.vec(c), b.vec(x.len()), b.vec(x.len()));
        DeviceChain::upload(&b, &wd, &w);
        DeviceChain::upload(&b, &bd, &bias);
        DeviceChain::upload(&b, &xd, &x);
        let freq: Vec<f32> = (0..c).map(|_| (r() * 0.5).exp()).collect();
        let scale: Vec<f32> = (0..c).map(|_| 1.0 / ((r() * 0.5).exp() + 1e-9)).collect();
        let (fd, sd, td) = (b.vec(c), b.vec(c), b.vec(x.len()));
        DeviceChain::upload(&b, &fd, &freq);
        DeviceChain::upload(&b, &sd, &scale);
        DeviceChain::upload(&b, &td, &x.iter().map(|v| v * 3.0).collect::<Vec<_>>());
        let mut rec = b.begin();
        rec.depthwise_causal_conv1d_rows(&wd, &bd, c, k, &xd, len, &yd);
        rec.read(&yd);
        rec.snake_beta_rows(&xd, &fd, &sd, len, c);
        rec.read(&xd);
        rec.clamp_in_place(&td, len * c, -1.0, 1.0);
        rec.read(&td);
        let got = rec.finish();
        for t in 0..len {
            for ch in 0..c {
                let mut want = bias[ch];
                for j in 0..k {
                    let s = t as isize + j as isize - k as isize + 1;
                    if s >= 0 {
                        want += w[ch * k + j] * x[s as usize * c + ch];
                    }
                }
                let i = t * c + ch;
                assert!((got[0][i] - want).abs() <= 1e-5 * (1.0 + want.abs()), "depthwise conv at {t}, {ch}: {} against {want}", got[0][i]);
                let v = x[i];
                let snake = v + scale[ch] * (freq[ch] * v).sin().powi(2);
                assert!((got[1][i] - snake).abs() <= 1e-5 * (1.0 + snake.abs()), "SnakeBeta at {t}, {ch}: {} against {snake}", got[1][i]);
                assert_eq!(got[2][i], (v * 3.0).clamp(-1.0, 1.0), "a clamp at {t}, {ch}");
            }
        }
    }

    /// A DAC decoder's ops as the host computes them: 1-D convolutions (7 taps 3 apart, 1 tap; channels not of 32),
    /// transposed ones (stride 4, and an odd stride with output padding: DAC's `ceil(s / 2)` padding, `s % 2` extra),
    /// Snake, and tanh.
    #[test]
    fn a_dacs_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(131);
        for (cin, cout, k, dil, len) in [(20usize, 36usize, 7usize, 3usize, 50usize), (64, 40, 7, 1, 70), (48, 8, 1, 1, 33)] {
            let wt: Vec<f32> = (0..cout * cin * k).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
            let x: Vec<f32> = (0..len * cin).map(|_| r()).collect();
            let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
            let wd = b.conv1d_weights(&wt, cout, cin, k).expect("the weights");
            let (xd, yd, bd) = (b.vec(x.len()), b.vec(len * cout), b.vec(cout));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &bd, &bias);
            let mut rec = b.begin();
            rec.conv1d_rows(&wd, &bd, cout, cin, k, dil, &xd, len, &yd);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let pad = (k / 2 * dil) as isize;
            for t in 0..len {
                for co in 0..cout {
                    let mut want = bias[co] as f64;
                    for c in 0..cin {
                        for j in 0..k {
                            let s = t as isize + (j * dil) as isize - pad;
                            if s >= 0 && (s as usize) < len {
                                want += wt[(co * cin + c) * k + j] as f64 * x[s as usize * cin + c] as f64;
                            }
                        }
                    }
                    let g = got[t * cout + co] as f64;
                    assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "1-D conv of {k} taps {dil} apart at {t}, {co}: {g} against {want}");
                }
            }
        }
        for (cin, cout, stride, len) in [(16usize, 12usize, 4usize, 9usize), (8, 4, 3, 7)] {
            let (k, pad, out_pad) = (2 * stride, stride.div_ceil(2), stride % 2);
            // PyTorch's [cin, cout, k], and the kernel's [cout, k, cin]
            let wt: Vec<f32> = (0..cin * cout * k).map(|_| r() * 0.3).collect();
            let packed: Vec<f32> = (0..cout * k * cin).map(|i| { let (co, j, c) = (i / (k * cin), (i / cin) % k, i % cin); wt[(c * cout + co) * k + j] }).collect();
            let x: Vec<f32> = (0..len * cin).map(|_| r()).collect();
            let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
            let out = (len - 1) * stride - 2 * pad + k + out_pad;
            let mut want = vec![0f64; out * cout];
            for (o, row) in want.chunks_mut(cout).enumerate() {
                for (co, v) in row.iter_mut().enumerate() {
                    *v = bias[co] as f64;
                    for i in 0..len {
                        for j in 0..k {
                            if i * stride + j == o + pad {
                                for c in 0..cin {
                                    *v += wt[(c * cout + co) * k + j] as f64 * x[i * cin + c] as f64;
                                }
                            }
                        }
                    }
                }
            }
            let (wd, xd, bd, yd) = (b.vec(packed.len()), b.vec(x.len()), b.vec(cout), b.vec(out * cout));
            DeviceChain::upload(&b, &wd, &packed);
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &bd, &bias);
            let mut rec = b.begin();
            rec.conv_transpose1d_rows(&wd, &bd, cout, cin, k, stride, pad, &xd, len, out, &yd);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() <= 1e-4 * (1.0 + w.abs()), "transposed 1-D conv, stride {stride}, [{i}]: {g} against {w}");
            }
        }
        let (rows, c) = (11usize, 6usize);
        let x: Vec<f32> = (0..rows * c).map(|_| r() * 3.0).collect();
        let alpha: Vec<f32> = (0..c).map(|_| r() + 1.5).collect();
        let (xd, ad, td) = (b.vec(x.len()), b.vec(c), b.vec(x.len()));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &ad, &alpha);
        DeviceChain::upload(&b, &td, &x);
        let mut rec = b.begin();
        rec.snake_rows(&xd, &ad, rows, c);
        rec.tanh_in_place(&td, rows * c);
        rec.read(&xd);
        rec.read(&td);
        let got = rec.finish();
        for (i, &v) in x.iter().enumerate() {
            let a = alpha[i % c];
            let snake = v + (a * v).sin().powi(2) / (a + 1e-9);
            assert!((got[0][i] - snake).abs() <= 1e-5 * (1.0 + snake.abs()), "Snake [{i}]: {} against {snake}", got[0][i]);
            assert!((got[1][i] - v.tanh()).abs() <= 1e-6, "tanh [{i}]: {} against {}", got[1][i], v.tanh());
        }
    }

    /// BiRefNet's ops as the host computes them: Swin's windows there and back (shifted and not, padded), its window
    /// attention (relative bias, the shifted windows' mask), a bilinear resize with the corners aligned, a picture as
    /// patches, a modulated deformable convolution's taps, a column's mean, a broadcast, and a 7x7 convolution.
    #[test]
    fn birefnets_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(127);
        let close = |got: &[f32], want: &[f32], what: &str| {
            assert_eq!(got.len(), want.len(), "{what}: lengths");
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{what} [{i}]: {g} against {w}");
            }
        };
        // windows of 4 over 6x7 tokens of 5 (padded to 8x8), shifted by 2 and not
        let (h, w, c, win) = (6usize, 7usize, 5usize, 4usize);
        let (hp, wp) = (8usize, 8usize);
        let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
        let xd = b.vec(x.len());
        DeviceChain::upload(&b, &xd, &x);
        for shift in [0usize, 2] {
            let mut want = vec![0f32; hp * wp * c];
            for (i, v) in want.iter_mut().enumerate() {
                let (ch, tok) = (i % c, i / c);
                let (wdx, t) = (tok / (win * win), tok % (win * win));
                let py = ((wdx / (wp / win)) * win + t / win + shift) % hp;
                let px = ((wdx % (wp / win)) * win + t % win + shift) % wp;
                if py < h && px < w {
                    *v = x[(py * w + px) * c + ch];
                }
            }
            let (wd, acc) = (b.vec(hp * wp * c), b.vec(h * w * c));
            DeviceChain::upload(&b, &acc, &vec![1.0; h * w * c]);
            let mut rec = b.begin();
            rec.window_rows(&xd, &wd, h, w, c, win, shift);
            rec.read(&wd);
            rec.unwindow_add_rows(&wd, &acc, h, w, c, win, shift);
            rec.read(&acc);
            let got = rec.finish();
            close(&got[0], &want, &format!("windows shifted {shift}"));
            close(&got[1], &x.iter().map(|v| v + 1.0).collect::<Vec<_>>(), &format!("windows back shifted {shift}"));
        }
        // window attention: 2 heads of 32 over those windows' tokens, shifted and not
        let heads = 2usize;
        let cc = heads * 32;
        let n = win * win;
        let qkv: Vec<f32> = (0..hp * wp * 3 * cc).map(|_| r()).collect();
        let table: Vec<f32> = (0..(2 * win - 1) * (2 * win - 1) * heads).map(|_| r()).collect();
        let (qd, td, od) = (b.vec(qkv.len()), b.vec(table.len()), b.vec(hp * wp * cc));
        DeviceChain::upload(&b, &qd, &qkv);
        DeviceChain::upload(&b, &td, &table);
        let scale = 1.0 / 32f32.sqrt();
        for shift in [0usize, 2] {
            let region = |i: usize, len: usize| if i < len - win { 0 } else if i < len - shift { 1 } else { 2 };
            let mut want = vec![0f32; hp * wp * cc];
            for wdx in 0..(hp / win) * (wp / win) {
                let (gy, gx) = ((wdx / (wp / win)) * win, (wdx % (wp / win)) * win);
                for head in 0..heads {
                    for qi in 0..n {
                        let q = &qkv[(wdx * n + qi) * 3 * cc + head * 32..][..32];
                        let scores: Vec<f32> = (0..n)
                            .map(|kj| {
                                let k = &qkv[(wdx * n + kj) * 3 * cc + cc + head * 32..][..32];
                                let mut s: f32 = q.iter().zip(k).map(|(a, b)| a * scale * b).sum();
                                s += table[((qi / win + win - 1 - kj / win) * (2 * win - 1) + qi % win + win - 1 - kj % win) * heads + head];
                                if shift > 0 && region(gy + qi / win, hp) * 3 + region(gx + qi % win, wp) != region(gy + kj / win, hp) * 3 + region(gx + kj % win, wp) {
                                    s -= 100.0;
                                }
                                s
                            })
                            .collect();
                        let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
                        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                        let l: f32 = e.iter().sum();
                        for d in 0..32 {
                            want[(wdx * n + qi) * cc + head * 32 + d] = (0..n).map(|kj| e[kj] * qkv[(wdx * n + kj) * 3 * cc + 2 * cc + head * 32 + d]).sum::<f32>() / l;
                        }
                    }
                }
            }
            let mut rec = b.begin();
            rec.window_attention(&qd, &td, &od, h, w, heads, win, shift, scale);
            rec.read(&od);
            close(&rec.finish().pop().unwrap(), &want, &format!("window attention shifted {shift}"));
        }
        // a bilinear resize, corners aligned
        let (h, w, c, oh, ow) = (5usize, 7usize, 3usize, 9usize, 4usize);
        let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
        let at = |n_in: usize, n_out: usize, o: usize| {
            let src = if n_out > 1 { (n_in - 1) as f32 / (n_out - 1) as f32 * o as f32 } else { 0.0 };
            let i0 = (src as usize).min(n_in - 1);
            (i0, (i0 + 1).min(n_in - 1), src - i0 as f32)
        };
        let want: Vec<f32> = (0..oh * ow * c)
            .map(|i| {
                let (ch, ox, oy) = (i % c, (i / c) % ow, i / (c * ow));
                let ((y0, y1, fy), (x0, x1, fx)) = (at(h, oh, oy), at(w, ow, ox));
                let v = |y: usize, xx: usize| x[(y * w + xx) * c + ch];
                (1.0 - fy) * ((1.0 - fx) * v(y0, x0) + fx * v(y0, x1)) + fy * ((1.0 - fx) * v(y1, x0) + fx * v(y1, x1))
            })
            .collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(want.len()));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.resize_bilinear_rows(&xd, &yd, h, w, c, oh, ow);
        rec.read(&yd);
        close(&rec.finish().pop().unwrap(), &want, "a bilinear resize");
        // a picture of 8 as patches of 4
        let (s, c, size) = (8usize, 3usize, 4usize);
        let g = s / size;
        let x: Vec<f32> = (0..s * s * c).map(|_| r()).collect();
        let want: Vec<f32> = (0..s * s * c)
            .map(|i| {
                let (oc, pix) = (i % (c * g * g), i / (c * g * g));
                let (ch, hg, wg) = (oc / (g * g), (oc / g) % g, oc % g);
                x[((hg * size + pix / size) * s + wg * size + pix % size) * c + ch]
            })
            .collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(want.len()));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.blocks_to_channels_rows(&xd, &yd, s, c, size);
        rec.read(&yd);
        close(&rec.finish().pop().unwrap(), &want, "patches");
        // a deformable 3x3's taps for pixels 5..30 of a 6x7 image of 4 channels
        let (h, w, c, k, first, pixels) = (6usize, 7usize, 4usize, 3usize, 5usize, 25usize);
        let kk = k * k;
        let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
        let offs: Vec<f32> = (0..h * w * 2 * kk).map(|_| r() * 2.5).collect();
        let mods: Vec<f32> = (0..h * w * kk).map(|_| r() + 1.0).collect();
        let mut want = vec![0f32; pixels * kk * c];
        for pi in 0..pixels {
            let pix = first + pi;
            for t in 0..kk {
                let y = (pix / w) as f32 - 1.0 + (t / k) as f32 + offs[pix * 2 * kk + 2 * t];
                let xx = (pix % w) as f32 - 1.0 + (t % k) as f32 + offs[pix * 2 * kk + 2 * t + 1];
                if y <= -1.0 || y >= h as f32 || xx <= -1.0 || xx >= w as f32 {
                    continue;
                }
                let (y0, x0) = (y.floor(), xx.floor());
                let (ly, lx) = (y - y0, xx - x0);
                for ch in 0..c {
                    let mut v = 0.0;
                    for (dy, dx, wgt) in [(0i64, 0i64, (1.0 - ly) * (1.0 - lx)), (0, 1, (1.0 - ly) * lx), (1, 0, ly * (1.0 - lx)), (1, 1, ly * lx)] {
                        let (yy, xc) = (y0 as i64 + dy, x0 as i64 + dx);
                        if yy >= 0 && xc >= 0 && (yy as usize) < h && (xc as usize) < w {
                            v += wgt * x[(yy as usize * w + xc as usize) * c + ch];
                        }
                    }
                    want[(pi * kk + t) * c + ch] = v * mods[pix * kk + t];
                }
            }
        }
        let (xd, od, md, yd) = (b.vec(x.len()), b.vec(offs.len()), b.vec(mods.len()), b.vec(want.len()));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &od, &offs);
        DeviceChain::upload(&b, &md, &mods);
        let mut rec = b.begin();
        rec.deform_im2col_rows(&xd, &od, &md, &yd, h, w, c, k, first, pixels);
        rec.read(&yd);
        close(&rec.finish().pop().unwrap(), &want, "a deformable convolution's taps");
        // a column's mean, and broadcast into rows 6 apart at 2
        let (rows, c) = (37usize, 3usize);
        let x: Vec<f32> = (0..rows * c).map(|_| r()).collect();
        let mean: Vec<f32> = (0..c).map(|ch| (0..rows).map(|row| x[row * c + ch]).sum::<f32>() / rows as f32).collect();
        let (xd, md, bd) = (b.vec(x.len()), b.vec(c), b.vec(rows * 6));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.mean_rows(&xd, &md, rows, c);
        rec.broadcast_rows(&md, &bd, rows, c, 6, 2);
        rec.read(&md);
        rec.read(&bd);
        let got = rec.finish();
        close(&got[0], &mean, "a mean");
        for row in 0..rows {
            close(&got[1][row * 6 + 2..row * 6 + 5], &mean, "a broadcast");
        }
        // a 7x7 convolution (16 channels to 8, 9x10 pixels)
        let (cin, cout, h, w) = (16usize, 8usize, 9usize, 10usize);
        let wt: Vec<f32> = (0..cout * cin * 49).map(|_| half::f16::from_f32(r() * 0.1).to_f32()).collect();
        let x: Vec<f32> = (0..h * w * cin).map(|_| half::f16::from_f32(r()).to_f32()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
        let wd = b.conv_weights(&wt, cout, cin, 7).expect("the weights");
        let (xd, yd, bd) = (b.vec(x.len()), b.vec(h * w * cout), b.vec(cout));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &bd, &bias);
        let mut rec = b.begin();
        rec.conv_rows(&wd, &bd, cout, cin, 7, &xd, h, w, &yd);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for py in 0..h {
            for px in 0..w {
                for co in 0..cout {
                    let mut want = bias[co] as f64;
                    for ch in 0..cin {
                        for ky in 0..7 {
                            for kx in 0..7 {
                                let (iy, ix) = (py as isize + ky as isize - 3, px as isize + kx as isize - 3);
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                    want += wt[((co * cin + ch) * 7 + ky) * 7 + kx] as f64 * x[(iy as usize * w + ix as usize) * cin + ch] as f64;
                                }
                            }
                        }
                    }
                    let g = got[(py * w + px) * cout + co] as f64;
                    assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "7x7 conv at ({py}, {px}) channel {co}: {g} against {want}");
                }
            }
        }
    }

    /// Real-ESRGAN's ops as the host computes them: a 3x3 convolution of a concatenation's leading channels (its
    /// pixels' values further apart than it reads), and a leaky ReLU, apart and in place.
    #[test]
    fn real_esrgans_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(97);
        for (cin, xs, cout, h, w) in [(64usize, 192usize, 32usize, 9usize, 13usize), (160, 192, 64, 7, 5), (3, 3, 64, 6, 11)] {
            let wt: Vec<f32> = (0..cout * cin * 9).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
            let x: Vec<f32> = (0..h * w * xs).map(|_| half::f16::from_f32(r()).to_f32()).collect();
            let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
            let wd = b.conv_weights(&wt, cout, cin, 3).expect("the weights");
            let (xd, yd, bd) = (b.vec(x.len()), b.vec(h * w * cout), b.vec(cout));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &bd, &bias);
            let mut rec = b.begin();
            rec.conv_rows_strided(&wd, &bd, cout, cin, 3, &xd, xs, h, w, &yd);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            for py in 0..h {
                for px in 0..w {
                    for co in 0..cout {
                        let mut want = bias[co] as f64;
                        for c in 0..cin {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let (iy, ix) = (py as isize + ky as isize - 1, px as isize + kx as isize - 1);
                                    if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                        want += wt[((co * cin + c) * 3 + ky) * 3 + kx] as f64 * x[(iy as usize * w + ix as usize) * xs + c] as f64;
                                    }
                                }
                            }
                        }
                        let g = got[(py * w + px) * cout + co] as f64;
                        assert!((g - want).abs() <= 1e-3 * (1.0 + want.abs()), "3x3 conv of {cin} of {xs} channels to {cout} at ({py}, {px}) channel {co}: {g} against {want}");
                    }
                }
            }
        }
        let x: Vec<f32> = (0..1000).map(|_| r() * 4.0).collect();
        let want: Vec<f32> = x.iter().map(|&v| if v > 0.0 { v } else { 0.2 * v }).collect();
        let (xd, yd) = (b.vec(1000), b.vec(1000));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.leaky_relu(&xd, &yd, 1000, 0.2);
        rec.read(&yd);
        rec.leaky_relu(&xd, &xd, 1000, 0.2);
        rec.read(&xd);
        let got = rec.finish();
        for (i, w) in want.iter().enumerate() {
            assert!((got[0][i] - w).abs() <= 1e-6 * (1.0 + w.abs()), "a leaky ReLU [{i}]: {} against {w}", got[0][i]);
            assert!((got[1][i] - w).abs() <= 1e-6 * (1.0 + w.abs()), "a leaky ReLU in place [{i}]: {} against {w}", got[1][i]);
        }
    }

    /// A VAE's ops as the host computes them: 3x3 and 1x1 convolutions on the tensor cores with their bias (channels
    /// not of 32 and of 32, an image not of the tile, inputs past f16's range: a VAE's reach 230,000), the nearest
    /// upsampling, and Wan's shuffled shortcut (one frame and two).
    #[test]
    fn a_vaes_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(31);
        for (cin, cout, h, w, k, big) in [(20usize, 36usize, 9usize, 13usize, 3usize, 1f32), (64, 160, 16, 16, 3, 1.0), (144, 4, 5, 7, 3, 1.0), (48, 40, 11, 6, 1, 1.0), (32, 24, 7, 9, 3, 300_000.0), (40, 8, 6, 5, 1, 300_000.0)] {
            let taps = k * k;
            let wt: Vec<f32> = (0..cout * cin * taps).map(|_| half::f16::from_f32(r() * 0.2).to_f32()).collect();
            let x: Vec<f32> = (0..h * w * cin).map(|_| half::f16::from_f32(r()).to_f32() * big).collect();
            let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
            let Some(wd) = b.conv_weights(&wt, cout, cin, k) else { return };
            let (xd, yd, bd) = (b.vec(x.len()), b.vec(h * w * cout), b.vec(cout));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &bd, &bias);
            let mut rec = b.begin();
            rec.conv_rows(&wd, &bd, cout, cin, k, &xd, h, w, &yd);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let off = (k / 2) as isize;
            for py in 0..h {
                for px in 0..w {
                    for co in 0..cout {
                        let mut want = bias[co] as f64;
                        for c in 0..cin {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let (iy, ix) = (py as isize + ky as isize - off, px as isize + kx as isize - off);
                                    if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                        want += wt[((co * cin + c) * k + ky) * k + kx] as f64 * x[(iy as usize * w + ix as usize) * cin + c] as f64;
                                    }
                                }
                            }
                        }
                        let g = got[(py * w + px) * cout + co] as f64;
                        assert!((g - want).abs() <= 1e-3 * (big as f64 + want.abs()), "{k}x{k} conv {cin}->{cout} (inputs to {big}) at ({py}, {px}) channel {co}: {g} against {want}");
                    }
                }
            }
        }
        let (h, w, c) = (3usize, 5usize, 6usize);
        let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(4 * x.len()));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.upsample2x_rows(&xd, &yd, h, w, c);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for oy in 0..2 * h {
            for ox in 0..2 * w {
                for ch in 0..c {
                    assert_eq!(got[(oy * 2 * w + ox) * c + ch], x[((oy / 2) * w + ox / 2) * c + ch], "upsampled ({oy}, {ox}) {ch}");
                }
            }
        }
        for (cin, cout, ft) in [(8usize, 8usize, 2usize), (8, 4, 1), (12, 6, 2)] {
            let repeats = cout * ft * 4 / cin;
            let x: Vec<f32> = (0..h * w * cin).map(|_| r()).collect();
            let base: Vec<f32> = (0..4 * h * w * cout).map(|_| r()).collect();
            let (xd, yd) = (b.vec(x.len()), b.vec(base.len()));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &yd, &base);
            let mut rec = b.begin();
            rec.shuffle_up_add_rows(&xd, &yd, h, w, cin, cout, ft);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            // as the host's: repeat each channel, view as (cout, ft, 2, 2), keep the last frame, shuffle into pixels
            for co in 0..cout {
                for a in 0..2 {
                    for bb in 0..2 {
                        for y in 0..h {
                            for xx in 0..w {
                                let e = ((co * ft + ft - 1) * 2 + a) * 2 + bb;
                                let ci = e / repeats;
                                let o = ((2 * y + a) * 2 * w + 2 * xx + bb) * cout + co;
                                assert_eq!(got[o], base[o] + x[(y * w + xx) * cin + ci], "shuffled {cin}->{cout} ft {ft} at {o}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// An NVFP4 matmul on the tensor cores as the host computes it: E2M1 pairs high nibble first, an E4M3 scale a block
    /// of 16 (every byte but the NaNs), the tensor's own scale and a bias after the sums; rows of a tile and not.
    #[test]
    fn an_nvfp4_matmul_is_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let e2m1 = [0f64, 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.];
        let e4m3 = |v: u8| -> f64 {
            let (s, e, m) = (if v & 0x80 != 0 { -1.0 } else { 1.0 }, ((v >> 3) & 15) as i32, (v & 7) as f64);
            s * if e == 0 { m / 8.0 * 2f64.powi(-6) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) }
        };
        let mut seed = 0x1234_5678u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (n, k, rows, global) in [(160usize, 256usize, 130usize, 0.0123f32), (64, 2048, 9, 3.5), (300, 128, 128, 1.0)] {
            let packed: Vec<u8> = (0..n * k / 2).map(|_| next() as u8).collect();
            let scales: Vec<u8> = (0..n * k / 16).map(|_| loop { let v = (next() % 0x78) as u8 | ((next() & 1) as u8) << 7; if v & 0x7f != 0x7f { break v; } }).collect();
            let x: Vec<f32> = (0..rows * k).map(|_| half::f16::from_f32((next() % 2001) as f32 / 1000.0 - 1.0).to_f32()).collect();
            let bias: Vec<f32> = (0..n).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
            let Some((wd, sd)) = b.nvfp4_weights(&packed, &scales, global, n, k) else { return };
            let (xd, yd, bd) = (b.vec(x.len()), b.vec(rows * n), b.vec(n));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &bd, &bias);
            let mut rec = b.begin();
            rec.matmul_nvfp4_rows(&wd, &sd, &bd, n, k, &xd, &yd, rows);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let weight = |o: usize, j: usize| -> f64 {
                let byte = packed[o * k / 2 + j / 2];
                let code = if j % 2 == 0 { byte >> 4 } else { byte & 15 };
                e2m1[code as usize] * e4m3(scales[o * k / 16 + j / 16]) * global as f64
            };
            for r in 0..rows {
                for o in 0..n {
                    let want = bias[o] as f64 + (0..k).map(|j| weight(o, j) * x[r * k + j] as f64).sum::<f64>();
                    let mag: f64 = (0..k).map(|j| (weight(o, j) * x[r * k + j] as f64).abs()).sum();
                    let g = got[r * n + o] as f64;
                    assert!((g - want).abs() <= 1e-5 * mag + 1e-5, "[{n}, {k}] of {rows} rows, row {r} output {o}: {g} against {want}");
                }
            }
        }
    }

    /// Wan's encoder's downsampling ops as the host computes them: the odd rows' odd columns, and the shortcut's
    /// shuffled means (time slots before the last zero; a spatial factor of 2 and of 1).
    #[test]
    fn a_vae_encoders_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(83);
        let (h, w, c) = (6usize, 8usize, 5usize);
        let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(x.len() / 4));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.subsample2x_rows(&xd, &yd, h, w, c);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for oy in 0..h / 2 {
            for ox in 0..w / 2 {
                for ch in 0..c {
                    assert_eq!(got[(oy * w / 2 + ox) * c + ch], x[((2 * oy + 1) * w + 2 * ox + 1) * c + ch], "subsampled ({oy}, {ox}) {ch}");
                }
            }
        }
        for (cin, cout, ft, fs) in [(4usize, 8usize, 2usize, 2usize), (4, 16, 1, 2), (6, 6, 1, 1), (8, 4, 2, 2)] {
            let x: Vec<f32> = (0..h * w * cin).map(|_| r()).collect();
            let (oh, ow) = (h / fs, w / fs);
            let base: Vec<f32> = (0..oh * ow * cout).map(|_| r()).collect();
            let (xd, yd) = (b.vec(x.len()), b.vec(base.len()));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &yd, &base);
            let mut rec = b.begin();
            rec.shuffle_down_mean_add_rows(&xd, &yd, h, w, cin, cout, ft, fs);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let g = cin * ft * fs * fs / cout;
            for oy in 0..oh {
                for ox in 0..ow {
                    for co in 0..cout {
                        let mut s = 0.0f64;
                        for e in co * g..(co + 1) * g {
                            let (c, t, fy, fx) = (e / (ft * fs * fs), (e / (fs * fs)) % ft, (e / fs) % fs, e % fs);
                            if t == ft - 1 {
                                s += x[((oy * fs + fy) * w + ox * fs + fx) * cin + c] as f64;
                            }
                        }
                        let want = base[(oy * ow + ox) * cout + co] as f64 + s / g as f64;
                        let g = got[(oy * ow + ox) * cout + co] as f64;
                        assert!((g - want).abs() < 1e-5, "shuffled mean {cin}->{cout} ({ft}, {fs}) at ({oy}, {ox}) {co}: {g} against {want}");
                    }
                }
            }
        }
    }

    /// LTX's encoder's single-image packing as the host computes it: space to depth (time slots one frame's; space,
    /// time and both) and the shortcut's group means.
    #[test]
    fn ltxs_encoder_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(89);
        let (h, w, c) = (4usize, 6usize, 3usize);
        let x: Vec<f32> = (0..h * w * c).map(|_| r()).collect();
        let xd = b.vec(x.len());
        DeviceChain::upload(&b, &xd, &x);
        for (st, sh, sw) in [(1usize, 2usize, 2usize), (2, 1, 1), (2, 2, 2)] {
            let vol = st * sh * sw;
            let (oh, ow) = (h / sh, w / sw);
            let yd = b.vec(oh * ow * c * vol);
            let mut rec = b.begin();
            rec.space_to_depth_rows(&xd, &yd, h, w, c, st, sh, sw);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            for oy in 0..oh {
                for ox in 0..ow {
                    for ch in 0..c {
                        for t in 0..st {
                            for fy in 0..sh {
                                for fx in 0..sw {
                                    let e = ((ch * st + t) * sh + fy) * sw + fx;
                                    assert_eq!(got[(oy * ow + ox) * c * vol + e], x[((oy * sh + fy) * w + ox * sw + fx) * c + ch], "space to depth ({st}, {sh}, {sw}) at ({oy}, {ox}) {e}");
                                }
                            }
                        }
                    }
                }
            }
        }
        let (rows, cin, cout) = (7usize, 12usize, 4usize);
        let x: Vec<f32> = (0..rows * cin).map(|_| r()).collect();
        let base: Vec<f32> = (0..rows * cout).map(|_| r()).collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(base.len()));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &yd, &base);
        let mut rec = b.begin();
        rec.group_mean_add_rows(&xd, &yd, rows, cin, cout);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for row in 0..rows {
            for co in 0..cout {
                let g = cin / cout;
                let want = base[row * cout + co] as f64 + (0..g).map(|e| x[row * cin + co * g + e] as f64).sum::<f64>() / g as f64;
                assert!((got[row * cout + co] as f64 - want).abs() < 1e-5, "group mean row {row} {co}");
            }
        }
    }

    /// The exact GELU as the host's (erf by its series in f64), across a range of inputs.
    #[test]
    fn an_exact_gelu_is_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        // erf by its Maclaurin series (in f64 near enough to |x| 4.25; past it within 2e-9 of 1)
        let erf = |x: f64| -> f64 {
            if x.abs() > 4.25 {
                return x.signum();
            }
            let (mut term, mut sum, mut n) = (x, x, 0.0);
            while term.abs() > 1e-17 * sum.abs().max(1e-300) && n < 400.0 {
                n += 1.0;
                term *= -x * x / n;
                sum += term / (2.0 * n + 1.0);
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        };
        let x: Vec<f32> = (0..2001).map(|i| (i as f32 - 1000.0) / 125.0).collect();
        let (xd, yd) = (b.vec(x.len()), b.vec(x.len()));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.gelu_erf(&xd, &yd, x.len());
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        for (v, g) in x.iter().zip(&got) {
            let want = 0.5 * *v as f64 * (1.0 + erf(*v as f64 / std::f64::consts::SQRT_2));
            assert!((*g as f64 - want).abs() <= 1e-6 * (1.0 + want.abs()), "gelu({v}): {g} against {want}");
        }
    }

    /// A copy past a grid dimension's 65,535 workgroups (17 million values, from one offset to another), every value
    /// where it belongs.
    #[test]
    fn a_long_copy_lands_every_value() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let len = 17_000_000usize;
        let src: Vec<f32> = (0..len + 3).map(|i| i as f32).collect();
        let (sd, dd) = (b.vec(src.len()), b.vec(len + 5));
        DeviceChain::upload(&b, &sd, &src);
        let mut rec = b.begin();
        rec.copy(&sd, 3, &dd, 5, len);
        rec.read(&dd);
        let got = rec.finish().pop().unwrap();
        for i in [0usize, 1, 16_776_959, 16_776_960, 16_776_961, len - 1] {
            assert_eq!(got[5 + i], (3 + i) as f32, "value {i}");
        }
        assert_eq!(&got[..5], &[0.; 5]);
    }

    /// A LoRA's merge on the device: `B A` by the f16 matmul (`A` transposed its weight, `B`'s rows its tokens) added
    /// into an f16 matrix, each sum rounded to f16, as the host's merge rounds it.
    #[test]
    fn a_lora_merges_into_an_f16_matrix_as_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(71);
        let (n, k, rank) = (40usize, 96usize, 32usize);
        let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r() * 0.1).to_f32()).collect();
        let a: Vec<f32> = (0..rank * k).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
        let bm: Vec<f32> = (0..n * rank).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
        let at: Vec<f32> = (0..k * rank).map(|i| a[(i % rank) * k + i / rank]).collect();
        let (Some(wd), Some(ad)) = (b.vec_f16(&w), b.vec_f16(&at)) else { return };
        let (bd, dd) = (b.vec(bm.len()), b.vec(n * k));
        DeviceChain::upload(&b, &bd, &bm);
        let mut rec = b.begin();
        rec.matmul_f16_rows(&ad, k, rank, &bd, &dd, n);
        rec.add_f16(&wd, &dd, n * k);
        rec.read(&wd);
        let words = rec.finish().pop().unwrap();
        for i in 0..n {
            for j in 0..k {
                let delta: f64 = (0..rank).map(|q| bm[i * rank + q] as f64 * a[q * k + j] as f64).sum();
                let want = half::f16::from_f64(w[i * k + j] as f64 + delta).to_f64();
                let word = words[(i * k + j) / 2].to_bits();
                let got = half::f16::from_bits(if (i * k + j) % 2 == 0 { word as u16 } else { (word >> 16) as u16 }).to_f64();
                assert!((got - want).abs() <= 2e-3 * want.abs() + 1e-4, "({i}, {j}): {got} against {want}");
            }
        }
    }

    /// LTX's block ops as the host computes them: the modulated norm over the RMS and with no norm (an affine), the
    /// split rotary (each head its own table), and the heads' gate.
    #[test]
    fn ltxs_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(53);
        let (rows, n) = (7usize, 96usize);
        let x: Vec<f32> = (0..rows * n).map(|_| r() * 2.0 + 0.3).collect();
        let mods: Vec<f32> = (0..3 * n).map(|_| r()).collect();
        let (xd, md, out) = (b.vec(x.len()), b.vec(mods.len()), b.vec(x.len()));
        DeviceChain::upload(&b, &xd, &x);
        DeviceChain::upload(&b, &md, &mods);
        for norm in [ggml_rs::RowNorm::Rms, ggml_rs::RowNorm::None] {
            let mut rec = b.begin();
            rec.norm_mod_rows(&xd, &out, rows, n, &md, n, Some(2 * n), norm, 1e-6);
            rec.read(&out);
            let got = rec.finish().pop().unwrap();
            for row in 0..rows {
                let v = &x[row * n..(row + 1) * n];
                let inv = match norm {
                    ggml_rs::RowNorm::Rms => 1.0 / (v.iter().map(|&a| (a as f64).powi(2)).sum::<f64>() / n as f64 + 1e-6).sqrt(),
                    _ => 1.0,
                };
                for i in 0..n {
                    let want = v[i] as f64 * inv * (1.0 + mods[n + i] as f64) + mods[2 * n + i] as f64;
                    let g = got[row * n + i] as f64;
                    assert!((g - want).abs() <= 1e-5 * (1.0 + want.abs()), "{norm:?} row {row} [{i}]: {g} against {want}");
                }
            }
        }
        // clean rows (the first two and the last) by a second modulation, three sets on
        let clean = ggml_rs::CleanRows { before: 2, from: rows - 1, offset: 3 * n };
        let both: Vec<f32> = (0..6 * n).map(|_| r()).collect();
        let bd = b.vec(both.len());
        DeviceChain::upload(&b, &bd, &both);
        let acc = b.vec(x.len());
        DeviceChain::upload(&b, &acc, &x);
        let mut rec = b.begin();
        rec.norm_mod_rows_clean(&xd, &out, rows, n, &bd, n, Some(2 * n), ggml_rs::RowNorm::Rms, 1e-6, clean);
        rec.add_gated_rows_clean(&acc, &xd, rows, n, &bd, 0, false, clean);
        rec.read(&out);
        rec.read(&acc);
        let mut reads = rec.finish();
        let (gated, normed) = (reads.pop().unwrap(), reads.pop().unwrap());
        for row in 0..rows {
            let set = if row < 2 || row >= rows - 1 { 3 * n } else { 0 };
            let v = &x[row * n..(row + 1) * n];
            let inv = 1.0 / (v.iter().map(|&a| (a as f64).powi(2)).sum::<f64>() / n as f64 + 1e-6).sqrt();
            for i in 0..n {
                let want = v[i] as f64 * inv * (1.0 + both[set + n + i] as f64) + both[set + 2 * n + i] as f64;
                assert!((normed[row * n + i] as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()), "clean rows' norm, row {row} [{i}]");
                let want = v[i] as f64 * (1.0 + both[set + i] as f64);
                assert!((gated[row * n + i] as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()), "clean rows' gate, row {row} [{i}]");
            }
        }
        // the split rotary: (row, head) its own table, each head's halves paired
        let (heads, hd) = (3usize, 8usize);
        let q: Vec<f32> = (0..rows * heads * hd).map(|_| r()).collect();
        let table: Vec<f32> = (0..rows * heads * hd / 2).flat_map(|i| { let a = i as f32 * 0.37; [a.sin(), a.cos()] }).collect();
        let (qd, td) = (b.vec(q.len()), b.vec(table.len()));
        DeviceChain::upload(&b, &qd, &q);
        DeviceChain::upload(&b, &td, &table);
        let mut rec = b.begin();
        rec.rope_split_rows(&qd, rows, heads, hd, &td);
        rec.read(&qd);
        let got = rec.finish().pop().unwrap();
        for row in 0..rows {
            for h in 0..heads {
                let base = (row * heads + h) * hd;
                for k in 0..hd / 2 {
                    let (s, c) = (table[(row * heads + h) * hd + 2 * k], table[(row * heads + h) * hd + 2 * k + 1]);
                    let (a, bb) = (q[base + k], q[base + k + hd / 2]);
                    assert!((got[base + k] - (a * c - bb * s)).abs() < 1e-5 && (got[base + k + hd / 2] - (a * s + bb * c)).abs() < 1e-5, "rotary row {row} head {h} pair {k}");
                }
            }
        }
        // the heads' gate
        let logits: Vec<f32> = (0..rows * heads).map(|_| r() * 3.0).collect();
        let ld = b.vec(logits.len());
        DeviceChain::upload(&b, &ld, &logits);
        DeviceChain::upload(&b, &qd, &q);
        let mut rec = b.begin();
        rec.head_gate_rows(&qd, &ld, rows, heads, hd);
        rec.read(&qd);
        let got = rec.finish().pop().unwrap();
        for (i, g) in got.iter().enumerate() {
            let want = q[i] * 2.0 / (1.0 + (-logits[i / hd]).exp());
            assert!((g - want).abs() < 1e-5, "gate [{i}]: {g} against {want}");
        }
    }

    /// LTX's VAE ops as the host computes them: a 3x3x3 convolution on the tensor cores (the first and last frames
    /// repeated past the clip's ends, zeros past each frame's edge; one frame and several, channels not of 32, inputs
    /// past f16's range) with its bias, and depth to space (time, space and both; the first frame dropped).
    #[test]
    fn ltxs_vae_ops_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(61);
        for (cin, cout, frames, h, w, big) in [(20usize, 36usize, 3usize, 5usize, 7usize, 1f32), (64, 40, 1, 4, 6, 1.0), (32, 8, 4, 3, 5, 300_000.0)] {
            let wt: Vec<f32> = (0..cout * cin * 27).map(|_| half::f16::from_f32(r() * 0.1).to_f32()).collect();
            let x: Vec<f32> = (0..frames * h * w * cin).map(|_| half::f16::from_f32(r()).to_f32() * big).collect();
            let bias: Vec<f32> = (0..cout).map(|_| r()).collect();
            let Some(wd) = b.conv3d_weights(&wt, cout, cin) else { return };
            let (xd, yd, bd) = (b.vec(x.len()), b.vec(frames * h * w * cout), b.vec(cout));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &bd, &bias);
            let mut rec = b.begin();
            rec.conv3d_rows(&wd, &bd, cout, cin, &xd, frames, h, w, &yd);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            for t in 0..frames {
                for py in 0..h {
                    for px in 0..w {
                        for co in 0..cout {
                            let mut want = bias[co] as f64;
                            for c in 0..cin {
                                for kt in 0..3 {
                                    let it = (t as isize + kt as isize - 1).clamp(0, frames as isize - 1) as usize;
                                    for ky in 0..3 {
                                        for kx in 0..3 {
                                            let (iy, ix) = (py as isize + ky as isize - 1, px as isize + kx as isize - 1);
                                            if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                                want += wt[(((co * cin + c) * 3 + kt) * 3 + ky) * 3 + kx] as f64 * x[((it * h + iy as usize) * w + ix as usize) * cin + c] as f64;
                                            }
                                        }
                                    }
                                }
                            }
                            let g = got[((t * h + py) * w + px) * cout + co] as f64;
                            assert!((g - want).abs() <= 1e-3 * (big as f64 + want.abs()), "3D conv {cin}->{cout} at ({t}, {py}, {px}) channel {co}: {g} against {want}");
                        }
                    }
                }
            }
        }
        for (st, sh, sw, drop) in [(2usize, 2usize, 2usize, 1usize), (2, 1, 1, 1), (1, 2, 2, 0)] {
            let (frames, h, w, c) = (3usize, 2usize, 3usize, 5usize);
            let vol = st * sh * sw;
            let x: Vec<f32> = (0..frames * h * w * c * vol).map(|_| r()).collect();
            let ot = frames * st - drop;
            let out_len = ot * h * sh * w * sw * c;
            let (xd, yd) = (b.vec(x.len()), b.vec(out_len));
            DeviceChain::upload(&b, &xd, &x);
            let mut rec = b.begin();
            rec.depth_to_space_rows(&xd, &yd, frames, h, w, c, st, sh, sw, drop);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            for o in 0..ot {
                for oy in 0..h * sh {
                    for ox in 0..w * sw {
                        for ch in 0..c {
                            let t = o + drop;
                            let (d, i, yy, j, xx, k) = (t / st, t % st, oy / sh, oy % sh, ox / sw, ox % sw);
                            let want = x[((d * h + yy) * w + xx) * c * vol + ch * vol + i * sh * sw + j * sw + k];
                            assert_eq!(got[((o * h * sh + oy) * w * sw + ox) * c + ch], want, "depth to space ({st}, {sh}, {sw}) at ({o}, {oy}, {ox}) {ch}");
                        }
                    }
                }
            }
        }
    }

    /// Full (unmasked) attention as the host computes it: every query over all of a text prefix's positions and the
    /// queries' own (Qwen-Image's image tokens: 32 heads of 128, here 4), on the tensor cores (a ragged 70 queries
    /// over 23 + 70) and in f32 (the same, and 5 queries), and a causal one beside it unchanged.
    #[test]
    fn full_attention_is_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(91);
        let (n_h, hd, scale) = (4usize, 128usize, 1.0 / (128f32).sqrt());
        // (queries over a prefix and themselves, and over fewer positions than they are: a video's over a text's)
        for (rows, kv_len) in [(70usize, 93usize), (5, 14), (70, 23)] {
            let q: Vec<f32> = (0..rows * n_h * hd).map(|_| r()).collect();
            let kv: Vec<f32> = (0..kv_len * 2 * n_h * hd).map(|_| r()).collect();
            let row = 2 * n_h * hd;
            let mut want = vec![0f64; rows * n_h * hd];
            for s in 0..rows {
                for h in 0..n_h {
                    let qv = &q[(s * n_h + h) * hd..(s * n_h + h + 1) * hd];
                    let scores: Vec<f64> = (0..kv_len).map(|t| (0..hd).map(|d| qv[d] as f64 * kv[t * row + h * hd + d] as f64).sum::<f64>() * scale as f64).collect();
                    let m = scores.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = scores.iter().map(|v| (v - m).exp()).collect();
                    let l: f64 = e.iter().sum();
                    for d in 0..hd {
                        want[(s * n_h + h) * hd + d] = (0..kv_len).map(|t| e[t] * kv[t * row + n_h * hd + h * hd + d] as f64).sum::<f64>() / l;
                    }
                }
            }
            let (qd, kvd) = (b.vec(q.len()), b.vec(kv.len()));
            DeviceChain::upload(&b, &qd, &q);
            DeviceChain::upload(&b, &kvd, &kv);
            for coop in [true, false] {
                if coop && (rows < 16 || !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)) {
                    continue;
                }
                let out = b.vec(b.attention_rows_out_len(rows.div_ceil(32) * 32, n_h, hd, kv_len));
                let mut rec = Recorder::new(&b);
                if coop {
                    assert!(rec.attention_rows_coop_masked(&qd, &kvd, &out, rows, n_h, n_h, hd, kv_len, None, scale, true));
                } else {
                    rec.attention_rows_f32_masked(&qd, &kvd, &out, rows, n_h, n_h, hd, kv_len, None, scale, true);
                }
                rec.read_range(&out, 0, rows * n_h * hd);
                let got = Box::new(rec).finish().pop().unwrap();
                let tol = if coop { 2e-3 } else { 1e-5 };
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!((*g as f64 - w).abs() <= tol, "{} {rows} queries over {kv_len} [{i}]: {g} against {w}", if coop { "tensor cores" } else { "f32" });
                }
            }
        }
    }

    /// A matrix of f16 values held as f16 (two to a word) multiplies as it does held as f32: a long row's step the same
    /// bits (summed the same way), a short row's and a prompt's within rounding (a prompt's on the tensor cores within
    /// f16's: its tokens f16, its sums f16 a window); a matrix not all f16 values is not made.
    #[test]
    fn an_f16_matrix_multiplies_as_its_f32_one() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(57);
        assert!(b.vec_f16(&[0.1, 0.5]).is_none(), "0.1 is no f16");
        for (n, k, rows) in [(324usize, 10240usize, 1usize), (513, 2560, 1), (1030, 324, 1), (70, 100, 1), (324, 10240, 70), (1030, 324, 65), (513, 2560, 300), (96, 2560, 512), (10240, 324, 130)] {
            let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r()).to_f32()).collect();
            let x: Vec<f32> = (0..rows * k).map(|_| r()).collect();
            let (w32, xd, y32, y16) = (b.vec(n * k), b.vec(rows * k), b.vec(rows * n), b.vec(rows * n));
            DeviceChain::upload(&b, &w32, &w);
            DeviceChain::upload(&b, &xd, &x);
            let w16 = b.vec_f16(&w).expect("f16 values");
            let mut rec = b.begin();
            rec.matmul_f32_rows(&w32, n, k, &xd, &y32, rows);
            rec.matmul_f16_rows(&w16, n, k, &xd, &y16, rows);
            rec.read(&y32);
            rec.read(&y16);
            let mut got = rec.finish();
            let (h, f) = (got.pop().unwrap(), got.pop().unwrap());
            if rows == 1 && k >= 2048 {
                assert_eq!(h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "[{n}, {k}]: the same sums");
            } else if rows > 8 && b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
                // on the tensor cores: the tokens rounded to f16, the sums f16 a window
                let rms = (f.iter().map(|e| (*e as f64).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
                let err = (h.iter().zip(&f).map(|(a, e)| ((*a - *e) as f64).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
                let worst = h.iter().zip(&f).map(|(a, e)| ((*a - *e) as f64).abs()).fold(0.0, f64::max);
                assert!(err < 2e-3 * rms && worst < 1.5e-2 * rms, "[{n}, {k}] of {rows} rows: RMS error {err:.3e}, the worst {worst:.3e}, of an RMS {rms:.3}");
            } else {
                let scale = (k as f32).sqrt();
                for (i, (a, e)) in h.iter().zip(&f).enumerate() {
                    assert!((a - e).abs() <= 1e-4 * scale, "[{n}, {k}] of {rows} rows [{i}]: {a} against {e}");
                }
            }
        }
    }

    /// Qwen3.8-Flash-Next's f16 matmuls of a chunk of 512 tokens (`--ignored --nocapture`): its hyper-connections' down
    /// and up, its router and its delta nets' `ba`, through the f32 tiled kernel and on the tensor cores (split as
    /// chosen and in 1 to 8).
    #[test]
    #[ignore = "a measurement"]
    fn measure_f16_matmuls() {
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        let m = 512usize;
        let mut r = rng(3);
        for (what, n, k) in [("hyper-connection down", 324usize, 10240usize), ("hyper-connection up", 10240, 324), ("router", 513, 2560), ("delta net ba", 96, 2560)] {
            let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r() * 0.05).to_f32()).collect();
            let w16 = b.vec_f16(&w).expect("f16 values");
            let (x, y) = (b.vec(m * k), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| r()).collect::<Vec<_>>());
            let time = |f: &dyn Fn(&mut Recorder)| {
                let run = || {
                    let mut rec = Recorder::new(&b);
                    for _ in 0..8 {
                        f(&mut rec);
                    }
                    rec.read_range(&y, 0, 1);
                    Box::new(rec).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                t.elapsed().as_secs_f64() / 24.0 * 1e3
            };
            let ms = time(&|rec| rec.matmul_f16_tiled(&w16, n, k, &x, &y, m));
            let mut line = format!("{what} [{n}, {k}]: f32 tiled {ms:.3} ms ({:.1} TFLOPS); tensor cores", 2.0 * (m * n * k) as f64 / ms / 1e9);
            for split in [None, Some(1), Some(2), Some(4), Some(8)] {
                let ms = time(&|rec| {
                    rec.wrote(buffer(&x));
                    assert!(rec.matmul_f16_coop(&w16, n, k, &x, &y, m, split));
                });
                line += &format!(" {}: {ms:.3} ms ({:.0})", split.map_or("chosen".to_string(), |s| s.to_string()), 2.0 * (m * n * k) as f64 / ms / 1e9);
            }
            eprintln!("{line}");
        }
    }

    /// QSA chained gives the host's: the pooled block keys bit for bit, the block scores within rounding, each query's
    /// chosen blocks the same, and its attention over them and its tail within rounding (Qwen3.8-Flash-Next's 4 index
    /// heads of 128, 24 heads of 256 over 2 kv heads, blocks of 4, 64 kept of 750); the selection of 512 of 4096 blocks
    /// and of 3001; and where a query keeps every block, the dense decode attention's bits.
    #[test]
    fn qsa_chained_is_the_hosts() {
        use ggml_rs::Backend;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        // (its selection takes 32 KB of a workgroup's memory: none where a workgroup has less)
        if DeviceChain::qsa_attention_out_len(&b, 1, 1, 64, 1, 4) == 0 {
            return;
        }
        let cpu = ggml_rs::CpuBackend::new();
        let mut r = rng(29);
        let (ratio, d, heads, keep, first, rows) = (4usize, 128usize, 4usize, 64usize, 2998usize, 3usize);
        let (nh, nkv, hd) = (24usize, 2usize, 256usize);
        let total = first + rows;
        let nb = total / ratio;
        let raw: Vec<f32> = (0..total * d).map(|_| r()).collect();
        let t = |v: &[f32], shape: Vec<usize>| ggml_rs::Tensor::from_vec(v.to_vec(), shape);
        let up = |v: &[f32]| {
            let x = b.vec(v.len());
            DeviceChain::upload(&b, &x, v);
            x
        };
        // the pool
        let (rawd, pooled) = (up(&raw), b.vec(nb * d));
        let mut rec = b.begin();
        rec.qsa_pool(&rawd, &pooled, nb, ratio, d);
        rec.read(&pooled);
        let got_pool = rec.finish().pop().unwrap();
        let want_pool = cpu.qsa_pool(&t(&raw, vec![total, d]), nb, ratio);
        assert_eq!(got_pool.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want_pool.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "the pooled keys");
        // the scores, the selection, the attention
        let q: Vec<f32> = (0..rows * heads * d).map(|_| r()).collect();
        let qa: Vec<f32> = (0..rows * nh * hd).map(|_| r()).collect();
        let (k, v): (Vec<f32>, Vec<f32>) = ((0..total * nkv * hd).map(|_| r()).collect(), (0..total * nkv * hd).map(|_| r()).collect());
        let mut kv = Vec::with_capacity(total * 2 * nkv * hd);
        for p in 0..total {
            kv.extend_from_slice(&k[p * nkv * hd..(p + 1) * nkv * hd]);
            kv.extend_from_slice(&v[p * nkv * hd..(p + 1) * nkv * hd]);
        }
        let scale = 1.0 / (d as f32).sqrt();
        let ascale = 1.0 / (hd as f32).sqrt();
        let (qd, scores, list, qad, kvd) = (up(&q), b.vec(rows * nb), b.vec(rows * keep), up(&qa), up(&kv));
        let out = b.vec(DeviceChain::qsa_attention_out_len(&b, rows, nh, hd, keep, ratio));
        let mut rec = b.begin();
        rec.qsa_scores(&qd, &pooled, &scores, rows, heads, d, nb, first, ratio, scale);
        rec.qsa_select(&scores, &list, rows, nb, first, ratio, keep);
        rec.qsa_attention(&qad, &kvd, &list, &out, rows, nh, nkv, hd, first, ratio, keep, ascale);
        rec.read(&scores);
        rec.read(&list);
        rec.read_range(&out, 0, rows * nh * hd);
        let mut got = rec.finish();
        let (got_out, got_list, got_scores) = (got.pop().unwrap(), got.pop().unwrap(), got.pop().unwrap());
        let want_scores = cpu.qsa_block_scores(&t(&q, vec![rows, heads, d]), &want_pool.reshape(vec![nb, d]).unwrap(), first, ratio, scale);
        for (i, (g, w)) in got_scores.iter().zip(want_scores.data()).enumerate() {
            assert!(g == w || (g - w).abs() <= 1e-5 * w.abs().max(1.0), "score {i}: {g} against {w}");
        }
        let want_sel = cpu.qsa_select(&want_scores, first, ratio, keep);
        let width = keep * ratio + ratio;
        for row in 0..rows {
            let mut want: Vec<u32> = want_sel.data()[row * width..row * width + keep * ratio].iter().step_by(ratio).map(|&v| v as u32 / ratio as u32).collect();
            want.sort();
            let got_row: Vec<u32> = got_list[row * keep..(row + 1) * keep].iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_row, want, "row {row}'s blocks");
        }
        let want_out = cpu.sparse_attention(&t(&qa, vec![rows, nh, hd]), &t(&k, vec![total, nkv, hd]), &t(&v, vec![total, nkv, hd]), &want_sel, ascale);
        let scale_out = want_out.data().iter().fold(1e-6f32, |m, x| m.max(x.abs()));
        for (i, (g, w)) in got_out.iter().zip(want_out.data()).enumerate() {
            assert!((g - w).abs() <= 1e-5 * scale_out, "attention {i}: {g} against {w}");
        }
        // the selection at its limits: 4096 blocks (the sort filling the workgroup's memory) and an odd count
        for nb in [4096usize, 3001] {
            let first = nb * ratio - 2;
            let sc: Vec<f32> = (0..2 * nb).map(|_| r().abs() * 4.0).collect();
            let (sd, ld) = (up(&sc), b.vec(2 * keep));
            let mut rec = b.begin();
            rec.qsa_select(&sd, &ld, 2, nb, first, ratio, keep);
            rec.read(&ld);
            let got = rec.finish().pop().unwrap();
            // what a selection must be (the host's picks among equal scores are its own): `keep` of the blocks the
            // row sees, ascending, none dropped above one kept, and of equals the lower kept first
            for row in 0..2 {
                let visible = ((first + row + 1) / ratio).min(nb);
                let got_row: Vec<usize> = got[row * keep..(row + 1) * keep].iter().map(|v| v.to_bits() as usize).collect();
                assert!(got_row.windows(2).all(|w| w[0] < w[1]) && got_row.iter().all(|&j| j < visible), "{nb} blocks: row {row}'s ascending, seen");
                let kept: std::collections::BTreeSet<usize> = got_row.iter().copied().collect();
                let s_row = &sc[row * nb..row * nb + visible];
                let worst_kept = kept.iter().map(|&j| s_row[j]).fold(f32::INFINITY, f32::min);
                for j in (0..visible).filter(|j| !kept.contains(j)) {
                    assert!(s_row[j] <= worst_kept, "{nb} blocks: row {row} dropped {j} ({}) above a kept {worst_kept}", s_row[j]);
                    if s_row[j] == worst_kept {
                        assert!(kept.iter().filter(|&&k| s_row[k] == worst_kept).all(|&k| k < j), "{nb} blocks: row {row}: of equals the lower first");
                    }
                }
            }
        }
        // every block kept: the dense decode attention, bit for bit
        let (few, at) = (40usize, 37usize);
        let mut rec = b.begin();
        let q1 = up(&qa[..nh * hd]);
        let (dense, sparse, l2, s2) = (b.vec(DeviceChain::attention_out_len(&b, nh, hd, total)), b.vec(DeviceChain::qsa_attention_out_len(&b, 1, nh, hd, few, ratio)), b.vec(few), b.vec(few));
        rec.qsa_scores(&q1, &pooled, &s2, 1, 1, d, at / ratio, at, ratio, scale);
        let _ = (heads, &l2);
        rec.qsa_select(&s2, &l2, 1, at / ratio, at, ratio, few);
        rec.qsa_attention(&q1, &kvd, &l2, &sparse, 1, nh, nkv, hd, at, ratio, few, ascale);
        rec.attention(&q1, &kvd, &dense, nh, nkv, hd, 0, at + 1, total, ascale);
        rec.read_range(&sparse, 0, nh * hd);
        rec.read_range(&dense, 0, nh * hd);
        let mut got = rec.finish();
        let (dn, sp) = (got.pop().unwrap(), got.pop().unwrap());
        assert_eq!(sp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), dn.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "every block kept: the dense attention");
    }

    /// The tensor cores through WGSL's cooperative matrices (where the adapter has them): one subgroup's 16x16x16 f16
    /// multiply into an f32 accumulator, the host's sums; the configurations the adapter reports, printed.
    #[test]
    fn a_cooperative_matrix_multiplies_as_the_host() {
        use ggml_rs::ChainRecorder;
        let b = match WgpuBackend::new(Some(1 << 30)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("no adapter: {e}");
                return;
            }
        };
        let features = b.gpu.device.features();
        eprintln!("cooperative matrices {}, f16 {}", features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX), features.contains(wgpu::Features::SHADER_F16));
        if !features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16) {
            return;
        }
        const PROBE: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> bm: array<f16>;
@group(0) @binding(6) var<storage, read_write> c: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(32)
fn main() {
    let ma = coopLoadT<coop_mat16x16<f16, A>>(&a[0], 16u);
    let mb = coopLoadT<coop_mat16x16<f16, B>>(&bm[0], 16u);
    var mc = coop_mat16x16<f32, C>();
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(mc, &c[0], 16u);
}
"#;
        let a: Vec<f32> = (0..256).map(|i| ((i * 7 % 23) as f32 - 11.0) / 8.0).collect();
        let bv: Vec<f32> = (0..256).map(|i| ((i * 5 % 19) as f32 - 9.0) / 4.0).collect();
        let pack = |v: &[f32]| -> Vec<f32> { v.chunks_exact(2).map(|p| f32::from_bits(half::f16::from_f32(p[0]).to_bits() as u32 | (half::f16::from_f32(p[1]).to_bits() as u32) << 16)).collect() };
        let up = |v: &[f32]| {
            let x = b.vec(v.len());
            DeviceChain::upload(&b, &x, v);
            x
        };
        let (ad, bd, cd) = (up(&pack(&a)), up(&pack(&bv)), b.vec(256));
        let mut rec = Recorder::new(&b);
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        rec.dispatch_wide("test-coop", PROBE, [buffer(&ad), buffer(&bd), &d, &d, &d, &d, buffer(&cd), &drw], &[0], (1, 1, 1));
        rec.read(&cd);
        let got = Box::new(rec).finish().pop().unwrap();
        for i in 0..16 {
            for j in 0..16 {
                let want: f32 = (0..16).map(|k| a[i * 16 + k] * bv[k * 16 + j]).sum();
                assert!((got[i * 16 + j] - want).abs() < 1e-3, "c[{i}, {j}]: {} against {want}", got[i * 16 + j]);
            }
        }
    }

    /// The tensor-core matmuls (where the device has them) give their f32 tiled kernels' sums within f16's rounding:
    /// Q3_K, Q4_K, Q5_K, Q6_K and Q8_0, a tile's worth of tokens and a tile and a bit (the edge), rows off the tile.
    #[test]
    fn the_tensor_core_matmuls_are_the_f32_ones() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        // the f16 scales' places in each type's block (d, and dmin where it has one)
        for (dtype, bytes, scales) in [(GgmlType::Q3_K, 110usize, &[108usize][..]), (GgmlType::Q4_K, 144, &[0, 2][..]), (GgmlType::Q5_K, 176, &[0, 2][..]), (GgmlType::Q6_K, 210, &[208][..]), (GgmlType::Q8_0, 34, &[0][..])] {
            for k in [512usize, 2048] {
            let n = 200usize;
            let mut next = rng(n as u32 + bytes as u32 + k as u32);
            let mut raw = vec![0u8; n * (k / if dtype == GgmlType::Q8_0 { 32 } else { 256 }) * bytes];
            for v in raw.iter_mut() {
                *v = ((next() + 1.0) * 100.0) as u8;
            }
            for blk in raw.chunks_exact_mut(bytes) {
                for &at in scales {
                    let d = half::f16::from_f32(0.01 + (blk[(at + 4) % bytes] as f32) * 1e-4).to_bits().to_le_bytes();
                    blk[at] = d[0];
                    blk[at + 1] = d[1];
                }
            }
            let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
            for m in [128usize, 150] {
                let (x, y, yc) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
                DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
                // the f32 tiled kernel's sums (the int8 and tensor-core paths bypassed)
                let mut rec = Recorder::new(&b);
                rec.matmul_rows_f32(&w, &x, &y, m);
                rec.read(&y);
                let want = Box::new(rec).finish().pop().unwrap();
                let mut rec = Recorder::new(&b);
                assert!(rec.matmul_rows_coop(&w, &x, &yc, m), "{dtype:?} on the tensor cores");
                rec.read(&yc);
                let got = Box::new(rec).finish().pop().unwrap();
                let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
                let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (norm(&got) * norm(&want));
                assert!(cos > 0.99999, "{dtype:?} [{n}, {k}] of {m}: cosine {cos}");
            }
            }
        }
    }

    /// How far the tensor cores' sums (f16 a window of steps, then f32) are from the f32 kernel's (`--ignored
    /// --nocapture`; OAIY_COOP_FOLD the window): the relative RMS error and the worst element's, Q3_K and Q6_K at the
    /// 27B's widths, the tokens' rows as a model's (unit RMS, a few channels a hundred times the rest).
    #[test]
    #[ignore = "a measurement"]
    fn measure_coop_fold_error() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        for (dtype, bytes, scales) in [(GgmlType::Q3_K, 110usize, &[108usize][..]), (GgmlType::Q6_K, 210, &[208][..])] {
            for k in [5120usize, 17408] {
                let (n, m) = (256usize, 256usize);
                let mut next = rng(k as u32 + bytes as u32);
                let mut raw = vec![0u8; n * (k / 256) * bytes];
                for v in raw.iter_mut() {
                    *v = ((next() + 1.0) * 100.0) as u8;
                }
                for blk in raw.chunks_exact_mut(bytes) {
                    for &at in scales {
                        let d = half::f16::from_f32(0.001 + (blk[(at + 4) % bytes] as f32) * 1e-5).to_bits().to_le_bytes();
                        blk[at] = d[0];
                        blk[at + 1] = d[1];
                    }
                }
                let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
                let x: Vec<f32> = (0..m * k).map(|i| next() * 1.7 * if i % k % 997 == 3 { 100.0 } else { 1.0 }).collect();
                let (xv, y, yc) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
                DeviceChain::upload(&b, &xv, &x);
                let mut rec = Recorder::new(&b);
                rec.matmul_rows_f32(&w, &xv, &y, m);
                rec.read(&y);
                let want = Box::new(rec).finish().pop().unwrap();
                let mut rec = Recorder::new(&b);
                assert!(rec.matmul_rows_coop(&w, &xv, &yc, m));
                rec.read(&yc);
                let got = Box::new(rec).finish().pop().unwrap();
                let err: f64 = got.iter().zip(&want).map(|(a, e)| ((*a - *e) as f64).powi(2)).sum::<f64>().sqrt();
                let norm: f64 = want.iter().map(|e| (*e as f64).powi(2)).sum::<f64>().sqrt();
                let rms = (norm * norm / want.len() as f64).sqrt();
                let worst = got.iter().zip(&want).map(|(a, e)| ((*a - *e) as f64).abs() / rms).fold(0.0, f64::max);
                eprintln!("{dtype:?} k {k}: relative error {:.2e}, the worst element's {:.2e} of the RMS ({rms:.3})", err / norm, worst);
            }
        }
    }

    /// What splitting a tensor-core matmul along k gains (`--ignored --nocapture`): Qwen3.8 27B's Q3_K matmuls of 512
    /// tokens, each split as chosen and in 1 to 4.
    #[test]
    #[ignore = "a measurement"]
    fn measure_coop_splits() {
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        let m = 512usize;
        for (what, n, k) in [("FFN gate and up", 34816usize, 5120usize), ("FFN gate", 17408, 5120), ("FFN down", 5120, 17408), ("delta net qkv", 10240, 5120), ("delta net gate", 6144, 5120), ("attention q", 12288, 5120)] {
            let mut next = rng(7);
            // (OAIY_BENCH_DTYPE: the weights that type, Q3_K unless asked)
            let (dtype, block) = match std::env::var("OAIY_BENCH_DTYPE").as_deref() {
                Ok("Q4_K") => (GgmlType::Q4_K, 144),
                Ok("Q5_K") => (GgmlType::Q5_K, 176),
                Ok("Q6_K") => (GgmlType::Q6_K, 210),
                Ok("Q8_0") => (GgmlType::Q8_0, 272),
                _ => (GgmlType::Q3_K, 110),
            };
            let raw: Vec<u8> = (0..n * (k / 256) * block).map(|_| ((next() + 1.0) * 100.0) as u8).collect();
            let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
            let (x, y) = (b.vec(m * k), b.vec(m * n));
            let units = b.gpu.coop_units();
            let chosen = crate::shaders::coop_splits((n as u32).div_ceil(128) * (m as u32).div_ceil(128), units, (k / 32) as u32);
            // (OAIY_BENCH_SECONDS: the chosen split's matmul run that long, what it does each second of it: a card
            // under a power limit's rate at the limit, where the rest is a burst's)
            if let Some(secs) = std::env::var("OAIY_BENCH_SECONDS").ok().and_then(|v| v.parse::<f64>().ok()) {
                let run = || {
                    let mut rec = Recorder::new(&b);
                    for _ in 0..8 {
                        rec.matmul_rows_coop_split(&w, &x, &y, m, None);
                    }
                    rec.read_range(&y, 0, 1);
                    Box::new(rec).finish();
                };
                run();
                let (t, mut done, mut said) = (std::time::Instant::now(), Vec::new(), String::new());
                while t.elapsed().as_secs_f64() < secs {
                    run();
                    done.push(t.elapsed().as_secs_f64());
                }
                for sec in 0..secs as usize {
                    let runs = done.iter().filter(|&&at| at >= sec as f64 && at < sec as f64 + 1.0).count();
                    said += &format!(" {:.0}", runs as f64 * 8.0 * 2.0 * (m * n * k) as f64 / 1e12);
                }
                eprintln!("{what} [{n}, {k}] for {secs} s, TFLOPS each second:{said}");
                continue;
            }
            let mut line = format!("{what} [{n}, {k}] (chosen {chosen}):");
            for split in [None, Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)] {
                let run = || {
                    let mut rec = Recorder::new(&b);
                    for _ in 0..8 {
                        rec.matmul_rows_coop_split(&w, &x, &y, m, split);
                    }
                    rec.read_range(&y, 0, 1);
                    Box::new(rec).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
                line += &format!(" {}: {ms:.3} ms ({:.0} TFLOPS)", split.map_or("chosen".to_string(), |s| s.to_string()), 2.0 * (m * n * k) as f64 / ms / 1e9);
            }
            eprintln!("{line}");
        }
    }

    /// The GPU's count of its units (SMs) is some (`--nocapture` shows it), and a matmul's split along k is the one its
    /// waves of a workgroup a unit are fullest at for what the splits' sums cost: Qwen3.8 27B's matmuls of 512 tokens
    /// on an RTX 5090's 170 SMs (as [`measure_coop_splits`] finds them), and none empty.
    #[test]
    fn a_tensor_core_matmul_is_split_to_fill_the_gpu() {
        use crate::shaders::coop_splits;
        // its FFN's gate and up (1088 tiles, k 5120 in 160 steps), down (160 tiles, k 17408), a delta net's gate (192),
        // attention's q (384), k and v (32 each), output (160, k 6144)
        assert_eq!(coop_splits(1088, 170, 160), 1);
        assert_eq!(coop_splits(160, 170, 544), 1);
        assert_eq!(coop_splits(192, 170, 160), 3);
        assert_eq!(coop_splits(384, 170, 160), 2);
        assert_eq!(coop_splits(32, 170, 160), 5);
        assert_eq!(coop_splits(160, 170, 192), 1);
        // splits of 8 steps or more
        assert_eq!(coop_splits(1, 170, 32), 4);
        assert_eq!(coop_splits(1, 170, 18), 2);
        assert_eq!(coop_splits(1, 170, 15), 1);
        assert_eq!(coop_splits(1, 1, 160), 1);
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        if b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            let units = b.gpu.coop_units();
            eprintln!("{units} units (SMs)");
            assert!(units >= 1);
        }
    }

    /// Where [`crate::shaders::coop_tiled`]'s time goes (`--ignored --nocapture`): Qwen3.8 27B's FFN gate for 512 tokens,
    /// the kernel as it is, its decode replaced by constant stores, by no stores, and its multiply-adds taken out.
    #[test]
    #[ignore = "a measurement"]
    fn measure_coop_parts() {
        use ggml_rs::ChainRecorder;
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        let (n, k, m) = (17408usize, 5120usize, 512usize);
        let mut next = rng(5);
        let mut raw = vec![0u8; n * (k / 256) * 110];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).unwrap();
        let (x16, y) = (b.vec(m * k / 2), b.vec(m * n));
        DeviceChain::upload(&b, &x16, &vec![f32::from_bits(0x3c003c00); m * k / 2]);
        let full = crate::shaders::coop_tiled(GgmlType::Q3_K).unwrap();
        let tail = "        } else {\n            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }\n        }";
        let step = &full[full.find("let at4 = buf").unwrap()..full.find(tail).unwrap() + tail.len()];
        let constants = full.replace(step, "let at4 = buf + lr * S4 + lh * 4u;\n        for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.5h); }");
        let nothing = full.replace(step, "");
        let mut no_mma = full.clone();
        for (c, a, bf) in [("c00", "a0", "b0f"), ("c01", "a0", "b1f"), ("c02", "a0", "b2f"), ("c03", "a0", "b3f"), ("c10", "a1", "b0f"), ("c11", "a1", "b1f"), ("c12", "a1", "b2f"), ("c13", "a1", "b3f")] {
            let mma = format!("{c} = coopMultiplyAdd({a}, {bf}, {c});");
            assert!(no_mma.contains(&mma), "{mma}");
            no_mma = no_mma.replace(&mma, "");
        }
        for (name, src) in [("bench-coop-full", full.clone()), ("bench-coop-constants", constants), ("bench-coop-nothing", nothing), ("bench-coop-no-mma", no_mma)] {
            let pipeline = b.gpu.named_pipeline(name, || src.clone());
            let (chunk, row0, rows) = &q.chunks[0];
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 1, 0];
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..4 {
                    rec.dispatch_kept(&pipeline, chunk, buffer(&x16), buffer(&y), &words, (rows.div_ceil(128), (m as u32).div_ceil(128), 1));
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 12.0 * 1e3;
            eprintln!("{name}: {ms:.2} ms ({:.1} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
        }
    }

    /// The Q3_K tensor-core matmul and variants of it, each timed on FFN gate and up's shape (`--ignored --nocapture`):
    /// as it is, and with a part of it taken out (results wrong, what that part costs): the step's barrier, its tokens'
    /// loads, its decode, the f16 sums' folds.
    #[test]
    #[ignore = "a measurement"]
    fn measure_coop_variants() {
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        let (m, n, k) = (512usize, 34816usize, 5120usize);
        let mut next = rng(7);
        let raw: Vec<u8> = (0..n * (k / 256) * 110).map(|_| ((next() + 1.0) * 100.0) as u8).collect();
        let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("on the GPU");
        let x16 = b.vec(m * k / 2);
        let y = b.vec(m * n);
        let base = crate::shaders::coop_tiled(GgmlType::Q3_K).expect("Q3_K's kernel");
        let marked = crate::shaders::coop_tiled_marked(GgmlType::Q3_K).expect("Q3_K's kernel");
        // the loop's barrier (the prologue's kept)
        let step_end = "        xt[xa + 3u] = xr3;\n        workgroupBarrier();\n";
        let no_barrier = {
            let at = base.rfind(step_end).expect("the loop's barrier");
            format!("{}        xt[xa + 3u] = xr3;\n{}", &base[..at], &base[at + step_end.len()..])
        };
        // the loop's token loads (each step's from x16) as the first's
        let loads = "        let xb = xo + b * xs;\n        xr0 = x16[xb];\n        xr1 = x16[xb + 1u];\n        xr2 = x16[xb + 2u];\n        xr3 = x16[xb + 3u];\n";
        let no_x = {
            let at = base.rfind(loads).expect("the loop's loads");
            format!("{}{}", &base[..at], &base[at + loads.len()..])
        };
        let no_decode = {
            let mut out = String::new();
            let mut skipping = false;
            for line in marked.lines() {
                if line.contains("// DECODE BEGIN") {
                    skipping = true;
                }
                if !skipping {
                    out.push_str(line);
                    out.push('\n');
                }
                if line.contains("// DECODE END") {
                    skipping = false;
                }
            }
            out
        };
        let no_fold = base.replace("w0 += 32u", "w0 += 100000u").replace("w0 + 32u", "w0 + 100000u");
        for (name, src) in [("bench-coop-q3k", base.clone()), ("bench-coop-q3k-no-barrier", no_barrier), ("bench-coop-q3k-no-x-loads", no_x), ("bench-coop-q3k-no-decode", no_decode), ("bench-coop-q3k-no-fold", no_fold)] {
            let pipeline = b.gpu.named_pipeline(Box::leak(name.to_string().into_boxed_str()), || src.clone());
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    for (chunk, row0, rows) in &q.chunks {
                        let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 1, 0];
                        rec.dispatch_kept(&pipeline, chunk, buffer(&x16), buffer(&y), &words, (rows.div_ceil(128), (m as u32).div_ceil(128), 1));
                    }
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
            eprintln!("{name}: {ms:.3} ms ({:.0} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
        }
    }

    /// The tensor-core matmul's skeleton (no decode: the weights the first step's, the tokens loaded every step) in
    /// shapes of a workgroup and its subgroups (`--ignored --nocapture`): what the loop alone runs at, f16 sums.
    #[test]
    #[ignore = "a measurement"]
    fn measure_coop_skeletons() {
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        let (m, n, k) = (512usize, 34816usize, 5120usize);
        let x16 = b.vec(m * k / 2);
        let w16 = b.vec(n * k / 2);
        let y = b.vec(m * n);
        // (warps down the rows, across the tokens; fragments each down, across): the tile 128 by 128 either way
        for (wr, wc, fr, fc) in [(4u32, 2u32, 2u32, 4u32), (2, 2, 4, 4), (2, 4, 4, 2), (4, 4, 2, 2)] {
            let threads = 32 * wr * wc;
            let mut src = String::new();
            src += "enable f16;\nenable wgpu_cooperative_matrix;\n";
            src += "struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, splits: u32, _pad1: u32, }\n";
            src += "@group(0) @binding(0) var<storage, read> w16: array<vec4<f16>>;\n@group(0) @binding(1) var<storage, read> x16: array<vec4<f16>>;\n@group(0) @binding(2) var<storage, read_write> y: array<f16>;\n@group(0) @binding(3) var<uniform> p: Params;\n";
            src += "const S4: u32 = 10u;\nconst BUF4: u32 = 1280u;\nvar<workgroup> wt: array<vec4<f16>, 2560>;\nvar<workgroup> xt: array<vec4<f16>, 2560>;\n";
            src += &format!("@compute @workgroup_size({threads})\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n");
            src += "    let r0 = wg.x * 128u;\n    let t0 = wg.y * 128u;\n    let sg = li / 32u;\n";
            src += &format!("    let sr = (sg % {wr}u) * {}u;\n    let st = (sg / {wr}u) * {}u;\n", 16 * fr, 16 * fc);
            // a thread's share of a step's 128 x 32 tokens (and, once, weights): `per` vec4s
            let per = 128 * 8 / threads;
            src += &format!("    let padded = ((p.m + 127u) / 128u) * 128u;\n    let xs = padded * 8u;\n");
            for r in 0..fr {
                for c in 0..fc {
                    src += &format!("    var h{r}{c} = coop_mat16x16<f16, C>();\n");
                }
            }
            // the weights once (whatever is there), the first step's tokens
            src += &format!("    for (var e = li; e < 1024u; e += {threads}u) {{ let row = e / 8u; let q = e % 8u; wt[row * S4 + q] = w16[(r0 + row) * (p.k / 4u) + q]; wt[BUF4 + row * S4 + q] = w16[(r0 + row) * (p.k / 4u) + q]; xt[row * S4 + q] = x16[(t0 + row) * 8u + q]; }}\n");
            src += "    workgroupBarrier();\n    let all = p.k / 32u;\n";
            for v in 0..per {
                src += &format!("    var xr{v} = vec4<f16>();\n");
            }
            src += "    for (var b0 = 0u; b0 < all; b0++) {\n        let b = min(b0 + 1u, all - 1u);\n        let buf = ((b0 + 1u) % 2u) * BUF4;\n        let cur = (b0 % 2u) * BUF4;\n";
            for v in 0..per {
                src += &format!("        let e{v} = li + {}u;\n        xr{v} = x16[b * xs + (t0 + e{v} / 8u) * 8u + e{v} % 8u];\n", v * threads);
            }
            for kk in [0u32, 16] {
                src += "        {\n            let s10 = S4;\n";
                for r in 0..fr {
                    src += &format!("            let ia{r} = cur + (sr + {}u) * S4 + {}u;\n            let a{r} = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia{r}], s10);\n", 16 * r, kk / 4);
                }
                for c in 0..fc {
                    src += &format!("            let ib{c} = cur + (st + {}u) * S4 + {}u;\n            let b{c} = coopLoad<coop_mat16x16<f16, B>>(&xt[ib{c}], s10);\n", 16 * c, kk / 4);
                }
                for r in 0..fr {
                    for c in 0..fc {
                        src += &format!("            h{r}{c} = coopMultiplyAdd(a{r}, b{c}, h{r}{c});\n");
                    }
                }
                src += "        }\n";
            }
            for v in 0..per {
                src += &format!("        xt[buf + (e{v} / 8u) * S4 + e{v} % 8u] = xr{v};\n");
            }
            src += "        workgroupBarrier();\n    }\n";
            for r in 0..fr {
                for c in 0..fc {
                    src += &format!("    {{\n        let o = (t0 + st + {}u) * p.n + r0 + sr + {}u;\n        let ns = p.n;\n        coopStore(h{r}{c}, &y[o], ns);\n    }}\n", 16 * c, 16 * r);
                }
            }
            src += "}\n";
            let name: &'static str = Box::leak(format!("bench-coop-skeleton-{wr}x{wc}-{fr}x{fc}").into_boxed_str());
            let pipeline = b.gpu.named_pipeline(name, || src.clone());
            let run = || {
                let mut rec = Recorder::new(&b);
                for _ in 0..8 {
                    let words = [k as u32, n as u32, m as u32, 0, n as u32, 0, 1, 0];
                    rec.dispatch_kept(&pipeline, buffer(&w16), buffer(&x16), buffer(&y), &words, ((n as u32).div_ceil(128), (m as u32).div_ceil(128), 1));
                }
                rec.read_range(&y, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
            eprintln!("{name} ({threads} threads): {ms:.3} ms ({:.0} TFLOPS)", 2.0 * (m * n * k) as f64 / ms / 1e9);
        }
    }

    /// How fast the host's bytes go up to each card and come back (`--ignored --nocapture`).
    #[test]
    #[ignore = "a measurement"]
    fn measure_transfer_rates() {
        let Ok(b) = WgpuBackend::new(None) else { return };
        let others = b.others(None);
        for g in std::iter::once(&b).chain(&others) {
            let (up, down) = g.transfer_rates();
            let (again_up, again_down) = g.transfer_rates();
            eprintln!("{} at {}: up {up:.1} and {again_up:.1} GB/s, down {down:.1} and {again_down:.1} GB/s", g.adapter().name, g.adapter().pci_bus_id);
        }
    }

    /// What a dispatch costs of itself (`--ignored --nocapture`): 1,000 copies of 256 values, each reading what the
    /// last wrote (a barrier between each two), and each into a vector of its own.
    #[test]
    #[ignore = "a measurement"]
    fn measure_dispatch_overhead() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let (x, y) = (b.vec(256), b.vec(256));
        let many: Vec<DeviceVec> = (0..1000).map(|_| b.vec(256)).collect();
        for dependent in [true, false] {
            let run = || {
                let mut rec = b.begin();
                rec.keep_groups(false);
                for i in 0..1000 {
                    if dependent {
                        let (s, d) = if i % 2 == 0 { (&x, &y) } else { (&y, &x) };
                        rec.copy(s, 0, d, 0, 256);
                    } else {
                        rec.copy(&x, 0, &many[i], 0, 256);
                    }
                }
                rec.read_range(&x, 0, 1);
                rec.finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            eprintln!("1,000 copies, {}: {:.1} us a dispatch", if dependent { "each after the last" } else { "none after another" }, t.elapsed().as_secs_f64() / 3.0 / 1000.0 * 1e6);
        }
    }

    /// What a matmul's loop reaches on the tensor cores with both its tiles in the workgroup's memory (`--ignored
    /// --nocapture`): a workgroup's tile of rows by tokens, its subgroups' shares of it, the k step, each step's
    /// fragments loaded from the tiles (filled once) and multiplied, a barrier a step; 2 workgroups an SM of 170.
    #[test]
    #[ignore = "a measurement"]
    fn measure_coop_tiles() {
        use ggml_rs::ChainRecorder;
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        // (rows, tokens of the workgroup's tile; subgroups down the rows, across the tokens; k a step)
        for (rows, tokens, wr, wt, ks) in [(128u32, 128u32, 4u32, 2u32, 64u32), (128, 128, 2, 2, 64), (128, 128, 2, 4, 64), (128, 128, 2, 2, 32), (256, 128, 4, 2, 32), (128, 256, 2, 4, 32), (128, 128, 4, 2, 32)] {
            let (fr, ft) = (rows / wr / 16, tokens / wt / 16);
            let threads = 32 * wr * wt;
            let stride4 = (ks + 8) / 4;
            let (a4, b4) = (rows * stride4, tokens * stride4);
            let mut body = String::new();
            for r in 0..fr {
                for t in 0..ft {
                    body += &format!("    var c{r}_{t} = coop_mat16x16<f32, C>();\n");
                }
            }
            body += "    for (var it = 0u; it < p[0].x; it++) {\n        for (var kk = 0u; kk < KSu; kk += 16u) {\n            let s4 = STRIDE4u;\n";
            for r in 0..fr {
                body += &format!("            let ia{r} = (wr0 + {}u) * STRIDE4u + kk / 4u;\n            let a{r} = coopLoadT<coop_mat16x16<f16, A>>(&at[ia{r}], s4);\n", r * 16);
            }
            for t in 0..ft {
                body += &format!("            let ib{t} = (wt0 + {}u) * STRIDE4u + kk / 4u;\n            let b{t} = coopLoad<coop_mat16x16<f16, B>>(&bt[ib{t}], s4);\n", t * 16);
            }
            for r in 0..fr {
                for t in 0..ft {
                    body += &format!("            c{r}_{t} = coopMultiplyAdd(a{r}, b{t}, c{r}_{t});\n");
                }
            }
            body += "        }\n        workgroupBarrier();\n    }\n";
            for r in 0..fr {
                for t in 0..ft {
                    body += &format!("    {{\n        let o = ((wg.x * {threads}u / 32u + sg) * {} + {}u) * 256u;\n        coopStoreT(c{r}_{t}, &c[o], 16u);\n    }}\n", fr * ft, r * ft + t);
                }
            }
            let src = format!(
                "enable f16;\nenable wgpu_cooperative_matrix;\n@group(0) @binding(0) var<storage, read> a: array<f16>;\n@group(0) @binding(6) var<storage, read_write> c: array<f32>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\nvar<workgroup> at: array<vec4<f16>, {a4}>;\nvar<workgroup> bt: array<vec4<f16>, {b4}>;\n@compute @workgroup_size({threads})\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n    for (var i = li; i < {a4}u; i += {threads}u) {{ at[i] = vec4<f16>(f16(i % 7u) * 0.01h); }}\n    for (var i = li; i < {b4}u; i += {threads}u) {{ bt[i] = vec4<f16>(f16(i % 5u) * 0.01h); }}\n    workgroupBarrier();\n    let sg = li / 32u;\n    let wr0 = (sg % {wr}u) * {}u;\n    let wt0 = (sg / {wr}u) * {}u;\n{}}}\n",
                fr * 16,
                ft * 16,
                body.replace("KSu", &format!("{ks}u")).replace("STRIDE4u", &format!("{stride4}u"))
            );
            let groups = 340u32;
            let out = b.vec((groups * threads / 32 * fr * ft * 256) as usize);
            let a = b.vec(16);
            let iters = 4096u32 * 64 / ks;
            let name: &'static str = Box::leak(format!("bench-coop-tile-{rows}x{tokens}-{wr}x{wt}-{ks}").into_boxed_str());
            let run = || {
                let mut rec = Recorder::new(&b);
                let d = rec.gpu().dummy().clone();
                let drw = rec.gpu().dummy_rw().clone();
                rec.dispatch_wide(name, &src, [buffer(&a), &d, &d, &d, &d, &d, buffer(&out), &drw], &[iters], (groups, 1, 1));
                rec.read_range(&out, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let secs = t.elapsed().as_secs_f64() / 3.0;
            let flops = groups as f64 * (rows * tokens) as f64 * (iters * ks) as f64 * 2.0;
            eprintln!("a tile of {rows}x{tokens}, subgroups {wr}x{wt} of {}x{} ({} fragments), k {ks} a step ({:.1} KB): {:.1} TFLOPS", fr * 16, ft * 16, fr * ft, (a4 + b4) as f64 * 8.0 / 1024.0, flops / secs / 1e12);
        }
    }

    /// A UNet's ops are their formulas: a group norm of an image's rows (its groups' means and variances over every
    /// pixel, a large offset on a group kept), with and without its SiLU; GEGLU's gate times the exact GELU; and the
    /// even pixels of an image.
    #[test]
    fn a_unets_ops_are_their_formulas() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut next = rng(77);
        // the group norm: 300 pixels (two chunks of 256, the second short) of 64 channels in 32 groups
        for (pixels, c, groups) in [(300usize, 64usize, 32usize), (16usize, 320, 32), (1024, 96, 8)] {
            let cpg = c / groups;
            // (a group's values about an offset of its own, one of them large)
            let x: Vec<f32> = (0..pixels * c).map(|i| next() * (1.0 + (i % c / cpg) as f32 * 0.1) + if (i % c) / cpg == 3 { 250.0 } else { (i % c / cpg) as f32 * 0.5 }).collect();
            let (weight, bias): (Vec<f32>, Vec<f32>) = ((0..c).map(|_| 1.0 + 0.3 * next()).collect(), (0..c).map(|_| 0.2 * next()).collect());
            for silu in [false, true] {
                let eps = 1e-5f32;
                let (xv, wv, bv, ov, sv) = (b.vec(pixels * c), b.vec(c), b.vec(c), b.vec(pixels * c), b.vec(groups * (pixels.div_ceil(256) + 1) * 2));
                DeviceChain::upload(&b, &xv, &x);
                DeviceChain::upload(&b, &wv, &weight);
                DeviceChain::upload(&b, &bv, &bias);
                let mut rec = Recorder::new(&b);
                rec.group_norm_rows(&xv, &wv, &bv, &ov, &sv, pixels, c, groups, eps, silu);
                rec.read(&ov);
                let got = Box::new(rec).finish().pop().unwrap();
                let mut worst = 0f64;
                for g in 0..groups {
                    let values: Vec<f64> = (0..pixels).flat_map(|px| (0..cpg).map(move |i| (px, i))).map(|(px, i)| x[px * c + g * cpg + i] as f64).collect();
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                    for px in 0..pixels {
                        for i in 0..cpg {
                            let ch = g * cpg + i;
                            let v = (x[px * c + ch] as f64 - mean) / (var + eps as f64).sqrt() * weight[ch] as f64 + bias[ch] as f64;
                            let want = if silu { v / (1.0 + (-v).exp()) } else { v };
                            worst = worst.max((got[px * c + ch] as f64 - want).abs());
                        }
                    }
                }
                eprintln!("a group norm of {pixels} pixels of {c} in {groups} groups{}: the worst error {worst:.2e}", if silu { ", through SiLU" } else { "" });
                assert!(worst < 2e-4, "{pixels} pixels of {c}: {worst}");
            }
        }
        // GEGLU
        let (rows, ff) = (7usize, 300usize);
        let fused: Vec<f32> = (0..rows * 2 * ff).map(|_| 3.0 * next()).collect();
        let (fv, ov) = (b.vec(rows * 2 * ff), b.vec(rows * ff));
        DeviceChain::upload(&b, &fv, &fused);
        let mut rec = Recorder::new(&b);
        rec.geglu_rows(&fv, &ov, rows, ff);
        rec.read(&ov);
        let got = Box::new(rec).finish().pop().unwrap();
        let erf = |v: f64| {
            // (Abramowitz and Stegun 7.1.26 is the kernel's; the series here to 1e-12)
            let (mut sum, mut term) = (v, v);
            for n in 1..60 {
                term *= -v * v / n as f64;
                sum += term / (2 * n + 1) as f64;
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        };
        let worst = (0..rows * ff).map(|i| {
            let (r, j) = (i / ff, i % ff);
            let (gate, value) = (fused[r * 2 * ff + j] as f64, fused[r * 2 * ff + ff + j] as f64);
            (got[i] as f64 - gate * 0.5 * value * (1.0 + erf(value / std::f64::consts::SQRT_2))).abs()
        }).fold(0f64, f64::max);
        eprintln!("GEGLU of {rows} rows of {ff}: the worst error {worst:.2e}");
        assert!(worst < 1e-5, "GEGLU: {worst}");
        // the even pixels
        let (h, w, c) = (6usize, 10usize, 5usize);
        let x: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let (xv, ov) = (b.vec(h * w * c), b.vec(h * w * c / 4));
        DeviceChain::upload(&b, &xv, &x);
        let mut rec = Recorder::new(&b);
        rec.subsample2x_even_rows(&xv, &ov, h, w, c);
        rec.read(&ov);
        let got = Box::new(rec).finish().pop().unwrap();
        for oy in 0..h / 2 {
            for ox in 0..w / 2 {
                for ch in 0..c {
                    assert_eq!(got[(oy * (w / 2) + ox) * c + ch], x[(2 * oy * w + 2 * ox) * c + ch], "pixel ({oy}, {ox}), channel {ch}");
                }
            }
        }
    }

    /// Normalised attention guidance's mix is the reference's formula (`ltx::transformer::nag_mix`): rows whose
    /// guided output is within tau of the plain one, rows scaled back to it, and alpha's blend.
    #[test]
    fn nag_mix_is_the_formula() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let (rows, width) = (37usize, 1028usize);
        let mut next = rng(41);
        let pos: Vec<f32> = (0..rows * width).map(|_| next()).collect();
        // (every third row's negative far from it: its guided output past tau, scaled back)
        let neg: Vec<f32> = (0..rows * width).map(|i| if (i / width) % 3 == 0 { -3.0 * next() } else { pos[i] + 0.05 * next() }).collect();
        for (scale, tau, alpha) in [(11.0f32, 2.5f32, 0.25f32), (3.0, 2.5, 1.0), (1.0, 2.5, 1.0)] {
            let (pv, nv) = (b.vec(rows * width), b.vec(rows * width));
            DeviceChain::upload(&b, &pv, &pos);
            DeviceChain::upload(&b, &nv, &neg);
            let mut rec = Recorder::new(&b);
            rec.nag_mix(&pv, &nv, rows, width, scale, tau, alpha);
            rec.read(&pv);
            let got = Box::new(rec).finish().pop().unwrap();
            let (mut worst, mut scaled) = (0f32, 0);
            for r in 0..rows {
                let (p, n) = (&pos[r * width..(r + 1) * width], &neg[r * width..(r + 1) * width]);
                let guided: Vec<f32> = p.iter().zip(n).map(|(a, b)| a * scale - b * (scale - 1.0)).collect();
                let l1 = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
                let factor = (tau as f64 * (l1(p) + 1e-6) / l1(&guided)).clamp(0.0, 1.0) as f32;
                scaled += (factor < 1.0) as usize;
                for c in 0..width {
                    let want = guided[c] * factor * alpha + p[c] * (1.0 - alpha);
                    worst = worst.max((got[r * width + c] - want).abs() / want.abs().max(1.0));
                }
            }
            eprintln!("scale {scale}, tau {tau}, alpha {alpha}: {scaled} of {rows} rows scaled back, the worst error {worst:.2e}");
            assert!(worst < 2e-5, "scale {scale}: {worst}");
            assert!(scale == 1.0 || scaled > 0, "some rows are scaled back");
        }
    }

    /// The cooperative matrices each adapter offers through wgpu (`--ignored --nocapture`): their shapes and their
    /// inputs' and sums' types. wgpu 30 names f32, f16, i32 and u32 only: a driver's 8-bit integer matrices (what
    /// llama.cpp's CUDA backend multiplies K-quants with on these cards) are not among what it passes on.
    #[test]
    #[ignore = "a listing"]
    fn list_cooperative_matrices() {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        for adapter in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
            let info = adapter.get_info();
            let shapes = adapter.cooperative_matrix_properties();
            eprintln!("{} ({:?}): {} shapes", info.name, info.backend, shapes.len());
            for p in shapes.iter() {
                eprintln!("    {} x {} x {}: {:?} inputs, {:?} sums{}", p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, if p.saturating_accumulation { ", saturating" } else { "" });
            }
        }
    }

    /// What the tensor cores reach through cooperative matrices (`--ignored --nocapture`): each subgroup multiplying
    /// 16x16 f16 fragments it holds into 8 accumulators, over and over (the arithmetic alone), its sums f32 and f16
    /// (an RTX 5090: 244 and 485 TFLOPS).
    #[test]
    #[ignore = "a measurement"]
    fn measure_cooperative_matrices() {
        use ggml_rs::ChainRecorder;
        let Ok(b) = WgpuBackend::new(Some(4 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        // the arithmetic alone, its sums f32 and f16
        for (acc, name) in [("f32", "bench-coop-alone"), ("f16", "bench-coop-alone-f16")] {
            let alone = format!(
                "enable f16;\nenable wgpu_cooperative_matrix;\n@group(0) @binding(0) var<storage, read> a: array<f16>;\n@group(0) @binding(6) var<storage, read_write> c: array<{acc}>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\n@compute @workgroup_size(128)\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n    let ma = coopLoadT<coop_mat16x16<f16, A>>(&a[0], 16u);\n    let mb = coopLoadT<coop_mat16x16<f16, B>>(&a[256], 16u);\n{}    for (var it = 0u; it < p[0].x; it++) {{\n{}    }}\n    let o = (wg.x * 4u + li / 32u) * 8u;\n{}}}\n",
                (0..8).map(|i| format!("    var c{i} = coop_mat16x16<{acc}, C>();\n")).collect::<String>(),
                (0..8).map(|i| format!("        c{i} = coopMultiplyAdd(ma, mb, c{i});\n")).collect::<String>(),
                (0..8).map(|i| format!("    coopStoreT(c{i}, &c[(o + {i}u) * 256u], 16u);\n")).collect::<String>()
            );
            let a = b.vec(256);
            DeviceChain::upload(&b, &a, &vec![f32::from_bits(0x3c003c00); 256]);
            let groups = 170 * 16;
            let out = b.vec(groups as usize * 4 * 8 * 256);
            let iters = 2048u32;
            let run = || {
                let mut rec = Recorder::new(&b);
                let d = rec.gpu().dummy().clone();
                let drw = rec.gpu().dummy_rw().clone();
                rec.dispatch_wide(name, &alone, [buffer(&a), &d, &d, &d, &d, &d, buffer(&out), &drw], &[iters], (groups, 1, 1));
                rec.read_range(&out, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let secs = t.elapsed().as_secs_f64() / 5.0;
            let flops = groups as f64 * 4.0 * 8.0 * iters as f64 * 2.0 * 4096.0;
            eprintln!("the arithmetic alone: {:.1} TFLOPS (f16 into {acc})", flops / secs / 1e12);
        }
    }

    /// What the GPU's arithmetic reaches in a kernel's registers (`--ignored --nocapture`): int8 dot products four at a
    /// time (`dot4I8Packed`) against f32 multiply-adds, each thread 16 independent sums, a run's counts per second.
    #[test]
    #[ignore = "a measurement"]
    fn measure_int8_and_f32_rates() {
        use ggml_rs::ChainRecorder;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let body = |dp4a: bool| -> String {
            let (ty, zero, op) = if dp4a { ("i32", "0", "a{i} = a{i} + dot4I8Packed(u{i}, v);") } else { ("f32", "0.0", "a{i} = fma(f{i}, g, a{i});") };
            let decl: String = (0..16).map(|i| format!("    var a{i}: {ty} = {zero};\n    let u{i} = 0x01020304u + {i}u * 0x01010101u + t;\n    let f{i} = f32({i}) * 0.001 + f32(t) * 1e-7;\n")).collect();
            let ops: String = (0..16).map(|i| format!("        {}\n", op.replace("{i}", &i.to_string()))).collect();
            let sum: String = (0..16).map(|i| format!(" + f32(a{i})")).collect();
            format!(
                "@group(0) @binding(0) var<storage, read> unused: array<u32>;\n@group(0) @binding(6) var<storage, read_write> out: array<f32>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let t = id.x;\n{decl}    var v = 0x05060708u + t;\n    var g = 1.0001;\n    for (var it = 0u; it < p[0].x; it++) {{\n{ops}        v = v + 1u;\n        g = g * 0.99999;\n    }}\n    out[t] = 0.0{sum};\n}}\n"
            )
        };
        let out = b.vec(256 * 170 * 64);
        for dp4a in [true, false] {
            let src = body(dp4a);
            let name: &'static str = if dp4a { "bench-dp4a" } else { "bench-ffma" };
            let iters = 4096u32;
            let groups = 170 * 64;
            let run = || {
                let mut rec = crate::chain::Recorder::new(&b);
                let d = rec.gpu().dummy().clone();
                let drw = rec.gpu().dummy_rw().clone();
                rec.dispatch_wide(name, &src, [&d, &d, &d, &d, &d, &d, buffer(&out), &drw], &[iters], (groups, 1, 1));
                rec.read_range(&out, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let secs = t.elapsed().as_secs_f64() / 5.0;
            let ops = (groups as f64) * 256.0 * iters as f64 * 16.0;
            eprintln!("{}: {:.1} T a second ({:.1} T multiply-adds)", if dp4a { "dot4I8Packed" } else { "f32 fma" }, ops / secs / 1e12, ops * if dp4a { 4.0 } else { 1.0 } / secs / 1e12);
        }
    }

    /// A draft's token from logits on the device is the host's: the first of equal largest, and the sum of the
    /// exponentials against it within rounding, over a vocabulary's 248,320 and a few.
    #[test]
    fn a_drafts_token_is_the_first_largest_logit_and_its_share() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(17);
        for (n, peak, tie) in [(248_320usize, 151_000usize, Some(200_000usize)), (5, 3, None), (300, 7, Some(6))] {
            let mut x: Vec<f32> = (0..n).map(|_| r() * 8.0).collect();
            x[peak] = 30.0;
            if let Some(t) = tie {
                x[t] = 30.0;
            }
            let (xd, out) = (b.vec(n), b.vec(3));
            DeviceChain::upload(&b, &xd, &x);
            let mut rec = b.begin();
            rec.argmax_softmax(&xd, &out);
            rec.read(&out);
            let got = rec.finish().pop().unwrap();
            let first = tie.map_or(peak, |t| t.min(peak));
            let total: f64 = x.iter().map(|&v| ((v - 30.0) as f64).exp()).sum();
            assert_eq!(got[0].to_bits() as usize, first, "{n}: the first largest");
            assert_eq!(got[1], 30.0);
            assert!(((got[2] as f64) - total).abs() <= 1e-5 * total, "{n}: {} against {total}", got[2]);
        }
    }

    /// A few rows of an f16 matrix (a check of drafts) give each row's one-row sums bit for bit: long rows (the
    /// hyper-connections' down matrices, the router) and short ones (their up matrices), 2 to 8 rows.
    #[test]
    fn a_few_rows_of_an_f16_matrix_are_each_row_alone() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(61);
        for (n, k) in [(324usize, 10240usize), (513, 2560), (10240, 324), (70, 100)] {
            let w: Vec<f32> = (0..n * k).map(|_| half::f16::from_f32(r()).to_f32()).collect();
            let w16 = b.vec_f16(&w).expect("f16 values");
            for rows in 2..=8usize {
                let x: Vec<f32> = (0..rows * k).map(|_| r()).collect();
                let (xd, yd) = (b.vec(rows * k), b.vec(rows * n));
                DeviceChain::upload(&b, &xd, &x);
                let mut rec = b.begin();
                rec.matmul_f16_rows(&w16, n, k, &xd, &yd, rows);
                rec.read(&yd);
                let got = rec.finish().pop().unwrap();
                let mut want = Vec::new();
                for row in x.chunks_exact(k) {
                    let (x1, y1) = (b.vec(k), b.vec(n));
                    DeviceChain::upload(&b, &x1, row);
                    let mut rec = b.begin();
                    rec.matmul_f16_rows(&w16, n, k, &x1, &y1, 1);
                    rec.read(&y1);
                    want.extend(rec.finish().pop().unwrap());
                }
                assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "[{n}, {k}] of {rows} rows");
            }
        }
    }

    /// An n-gram layer's gate and conv chained give the CPU's: two rows (a prompt's) then one (a step's), its window
    /// carried from the first to the second.
    #[test]
    fn an_ngram_layers_gate_and_conv_match_the_cpus() {
        use ggml_rs::Backend;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let mut r = rng(23);
        let (streams, d, kernel, dil) = (4usize, 320usize, 4usize, 3usize);
        let width = streams * d;
        let state = (kernel - 1) * dil;
        let mut v = |n: usize| (0..n).map(|_| r()).collect::<Vec<f32>>();
        let (nk, nq, nc, wc) = (v(width), v(width), v(width), v(width * kernel));
        let (dnk, dnq, dnc, dwc) = (b.vec(width), b.vec(width), b.vec(width), b.vec(width * kernel));
        for (dv, h) in [(&dnk, &nk), (&dnq, &nq), (&dnc, &nc), (&dwc, &wc)] {
            DeviceChain::upload(&b, dv, h);
        }
        let mut window = Tensor::from_vec(v(state * width), vec![state, width]);
        let dwin = b.vec(state * width);
        DeviceChain::upload(&b, &dwin, window.data());
        for rows in [2usize, 1] {
            let (key, x, value) = (v(rows * width), v(rows * width), v(rows * d));
            let t = |h: &[f32], shape: Vec<usize>| Tensor::from_vec(h.to_vec(), shape);
            let (gated, conv_in) = cpu.ple_gate(&t(&key, vec![rows, width]), &t(&x, vec![rows, width]), &t(&value, vec![rows, d]), &t(&nk, vec![width]), &t(&nq, vec![width]), &t(&nc, vec![width]), streams, 1e-6);
            let mut want = t(&x, vec![rows, width]);
            cpu.ple_conv(&mut want, &gated, &conv_in, &mut window, &t(&wc, vec![width, kernel]), kernel, dil);
            let (dk, dx, dval, dg, dci) = (b.vec(rows * width), b.vec(rows * width), b.vec(rows * d), b.vec(rows * width), b.vec(rows * width));
            DeviceChain::upload(&b, &dk, &key);
            DeviceChain::upload(&b, &dx, &x);
            DeviceChain::upload(&b, &dval, &value);
            let mut rec = b.begin();
            rec.ple_gate(&dk, &dx, &dval, &dnk, &dnq, &dnc, &dg, &dci, rows, streams, d, 1e-6);
            rec.read(&dg);
            rec.read(&dci);
            rec.ple_conv(&dx, &dg, &dci, &dwin, &dwc, rows, width, kernel, dil);
            rec.read(&dx);
            rec.read(&dwin);
            let mut got = rec.finish();
            let (win_got, x_got, ci_got, g_got) = (got.pop().unwrap(), got.pop().unwrap(), got.pop().unwrap(), got.pop().unwrap());
            close(&g_got, gated.data(), "gated");
            // conv_in is rounded to f16: an f16 step apart where the sums round differently
            for (i, (a, e)) in ci_got.iter().zip(conv_in.data()).enumerate() {
                assert!((a - e).abs() <= 2e-3 * e.abs().max(1.0), "conv_in [{i}]: {a} against {e}");
            }
            for (i, (a, e)) in x_got.iter().zip(want.data()).enumerate() {
                assert!((a - e).abs() <= 1e-2 * e.abs().max(1.0), "x [{i}] of {rows} rows: {a} against {e}");
            }
            close(&win_got, window.data(), "the window");
            // the next run's window is the CPU's (as a step after a prompt starts from the prompt's)
            DeviceChain::upload(&b, &dwin, window.data());
        }
    }

    /// Q3_K's int8 kernel against its f32 one (Qwen3.8 27B's FFN gate, [17408, 5120]): the same sums within int8's
    /// rounding, and their times, 1 to 4 rows (`--ignored --nocapture`).
    #[test]
    #[ignore = "a measurement"]
    fn measure_q8_matmuls() {
        let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
        // the f16 scales' places in each type's block (d, and dmin where it has one)
        for (dtype, n, k, block, bytes, scales) in [(GgmlType::Q3_K, 17408usize, 5120usize, 256usize, 110usize, &[108usize][..]), (GgmlType::Q4_K, 5120, 17408, 256, 144, &[0, 2][..]), (GgmlType::Q5_K, 10240, 5120, 256, 176, &[0, 2][..]), (GgmlType::Q6_K, 5120, 6144, 256, 210, &[208][..]), (GgmlType::Q4_0, 17408, 5120, 32, 18, &[0][..])] {
        let nbytes = n * (k / block) * bytes;
        let mut next = rng(n as u32);
        let mut raw = vec![0u8; nbytes];
        for v in raw.iter_mut() {
            *v = ((next() + 1.0) * 100.0) as u8;
        }
        // sane block scales: small and finite
        for blk in raw.chunks_exact_mut(bytes) {
            for &at in scales {
                let d = half::f16::from_f32(0.01 + (blk[(at + 4) % bytes] as f32) * 1e-4).to_bits().to_le_bytes();
                blk[at] = d[0];
                blk[at + 1] = d[1];
            }
        }
        // eight matrices in turn (392 MB, past the L2's 96 MB), as a model's layers stream from memory
        let ws: Vec<_> = (0..8u8)
            .map(|i| {
                let mut r = raw.clone();
                for v in r.iter_mut().step_by(7) {
                    *v = v.wrapping_add(i);
                }
                ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(r, vec![n, k], dtype))
            })
            .collect();
        let w = &ws[0];
        for m in [1usize, 2, 3, 4] {
            let (x, y, y8) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
            DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
            let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
            assert!(rq.matmul_rows_q8(w, &x, &y8, m));
            rq.read(&y8);
            let got = Box::new(rq).finish().pop().unwrap();
            let mut rec = b.begin();
            rec.matmul_rows(w, &x, &y, m);
            rec.read(&y);
            let want = rec.finish().pop().unwrap();
            let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
            let worst = got.iter().zip(&want).map(|(a, e)| (a - e).abs() / scale).fold(0f32, f32::max);
            let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got) * norm(&want));
            let reps = 28;
            let time = |q8: bool| {
                let run = || {
                    let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
                    for i in 0..reps {
                        let w = &ws[i % ws.len()];
                        if q8 {
                            rq.q8.clear();
                            rq.matmul_rows_q8(w, &x, &y8, m);
                        } else {
                            rq.matmul_rows(w, &x, &y, m);
                        }
                    }
                    rq.read_range(&y, 0, 1);
                    Box::new(rq).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                t.elapsed().as_secs_f64() / 5.0 / reps as f64
            };
            let (f, q) = (time(false), time(true));
            eprintln!("{dtype:?} [{n}, {k}] x {m} rows: f32 {:.1} us ({:.0} GB/s), int8 {:.1} us ({:.0} GB/s; quantizing included); worst {worst:.2e} of the largest, cosine {cos:.6}", f * 1e6, nbytes as f64 / f / 1e9, q * 1e6, nbytes as f64 / q / 1e9);
            assert!(cos > 0.9999, "{dtype:?} x {m}: cosine {cos}");
        }
        }
    }

    /// A prompt's Q3_K matmul through the int8 tiled kernel against the f32 one (`--ignored --nocapture`): Qwen3.8 27B's
    /// FFN gate [17408, 5120] and down [5120, 17408] for chunks of 512 and of 100 tokens, the results within int8's
    /// rounding.
    #[test]
    #[ignore = "a measurement"]
    fn measure_tiled_q8() {
        let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
        for (n, k) in [(17408usize, 5120usize), (5120, 17408)] {
            let (block, bytes) = (256usize, 110usize);
            let mut next = rng(n as u32);
            let mut raw = vec![0u8; n * (k / block) * bytes];
            for v in raw.iter_mut() {
                *v = ((next() + 1.0) * 100.0) as u8;
            }
            for blk in raw.chunks_exact_mut(bytes) {
                let d = half::f16::from_f32(0.01 + (blk[(108 + 4) % bytes] as f32) * 1e-4).to_bits().to_le_bytes();
                blk[108] = d[0];
                blk[109] = d[1];
            }
            let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
            for m in [512usize, 100] {
                let (x, y, y8) = (b.vec(m * k), b.vec(m * n), b.vec(m * n));
                DeviceChain::upload(&b, &x, &(0..m * k).map(|_| next()).collect::<Vec<_>>());
                let mut rec = b.begin();
                rec.matmul_rows(&w, &x, &y, m);
                rec.read(&y);
                let want = rec.finish().pop().unwrap();
                let mut rq = Recorder::new(&b);
                assert!(rq.matmul_rows_tq8(&w, &x, &y8, m));
                rq.read(&y8);
                let got = Box::new(rq).finish().pop().unwrap();
                let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
                let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (norm(&got) * norm(&want));
                let time = |q8: bool| {
                    let run = || {
                        let mut rq = Recorder::new(&b);
                        for _ in 0..4 {
                            if q8 {
                                rq.q8.clear();
                                rq.matmul_rows_tq8(&w, &x, &y8, m);
                            } else {
                                rq.matmul_rows(&w, &x, &y, m);
                            }
                        }
                        rq.read_range(&y, 0, 1);
                        Box::new(rq).finish();
                    };
                    run();
                    let t = std::time::Instant::now();
                    for _ in 0..3 {
                        run();
                    }
                    t.elapsed().as_secs_f64() / 12.0
                };
                let (f, q) = (time(false), time(true));
                let flops = 2.0 * (m * n * k) as f64;
                eprintln!("Q3_K [{n}, {k}] x {m}: f32 tiled {:.2} ms ({:.1} TFLOPS), int8 tiled {:.2} ms ({:.1}; quantizing included); cosine {cos:.6}", f * 1e3, flops / f / 1e12, q * 1e3, flops / q / 1e12);
                assert!(cos > 0.9999, "{n}x{k} of {m}: cosine {cos}");
                // the tensor cores
                let yc = b.vec(m * n);
                let mut rc = Recorder::new(&b);
                if rc.matmul_rows_coop(&w, &x, &yc, m) {
                    rc.read(&yc);
                    let got = Box::new(rc).finish().pop().unwrap();
                    let dot: f64 = got.iter().zip(&want).map(|(a, e)| *a as f64 * *e as f64).sum();
                    let cc = dot / (norm(&got) * norm(&want));
                    let run = || {
                        let mut rc = Recorder::new(&b);
                        for _ in 0..4 {
                            rc.x16.clear();
                            rc.matmul_rows_coop(&w, &x, &yc, m);
                        }
                        rc.read_range(&yc, 0, 1);
                        Box::new(rc).finish();
                    };
                    run();
                    let t = std::time::Instant::now();
                    for _ in 0..3 {
                        run();
                    }
                    let c = t.elapsed().as_secs_f64() / 12.0;
                    eprintln!("    tensor cores {:.2} ms ({:.1} TFLOPS); cosine {cc:.6}", c * 1e3, flops / c / 1e12);
                    assert!(cc > 0.9999, "{n}x{k} of {m} on the tensor cores: cosine {cc}");
                }
            }
        }
    }

    /// The one-row K-quant kernel's weight rows a lane, from memory (eight matrices in turn, past the L2): Qwen3.8 27B's
    /// Q3_K FFN gate and Q4_K down, Q5_K qkv, Q6_K (`--ignored --nocapture`).
    #[test]
    #[ignore = "a measurement"]
    fn measure_decode_rows_a_lane() {
        let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
        for (dtype, n, k, bytes) in [(GgmlType::Q3_K, 17408usize, 5120usize, 112usize), (GgmlType::Q4_K, 5120, 17408, 144), (GgmlType::Q5_K, 10240, 5120, 176), (GgmlType::Q6_K, 5120, 6144, 210)] {
            let mut next = rng(n as u32 ^ k as u32);
            let raw_bytes = if dtype == GgmlType::Q3_K { 110 } else { bytes };
            let nbytes = n * (k / 256) * raw_bytes;
            let mut raw = vec![0u8; nbytes];
            for v in raw.iter_mut() {
                *v = ((next() + 1.0) * 100.0) as u8;
            }
            let ws: Vec<_> = (0..8u8)
                .map(|i| {
                    let mut r = raw.clone();
                    for v in r.iter_mut().step_by(7) {
                        *v = v.wrapping_add(i);
                    }
                    ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(r, vec![n, k], dtype))
                })
                .collect();
            let (x, y) = (b.vec(k), b.vec(n));
            DeviceChain::upload(&b, &x, &(0..k).map(|_| next()).collect::<Vec<_>>());
            for (r, ks) in [(1u32, 1u32), (2, 1), (4, 1), (1, 2), (2, 2), (4, 2), (1, 4), (2, 4), (4, 4)] {
                let Some(src) = crate::shaders::rb_kernel_for_test(dtype, r, 1, ks) else { continue };
                let pipeline = b.gpu.named_pipeline(Box::leak(format!("test-rb-{dtype:?}-{r}-{ks}").into_boxed_str()), || src);
                let reps = 32;
                let run = || {
                    let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
                    for i in 0..reps {
                        let q = ws[i % ws.len()].device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).unwrap();
                        for (chunk, row0, rows) in &q.chunks {
                            let words = [k as u32, n as u32, 1, *row0, *rows, q.row_bytes as u32, 0, 0];
                            let groups = rows.div_ceil(4 * r);
                            rq.dispatch_kept(&pipeline, chunk, buffer(&x), buffer(&y), &words, (1, groups.min(65535), groups.div_ceil(65535)));
                        }
                    }
                    rq.read_range(&y, 0, 1);
                    Box::new(rq).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                let secs = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
                eprintln!("{dtype:?} [{n}, {k}] one row, {r} weight rows a lane, {ks} along k: {:.1} us, {:.0} GB/s", secs * 1e6, nbytes as f64 / secs / 1e9);
            }
        }
    }

    /// What reading memory reaches on this adapter through WebGPU: a kernel that sums 400 MB in vec4s, with 1, 2 and
    /// 4 loads in flight a thread, at 4 and 8 warps a workgroup (`--ignored --nocapture`).
    #[test]
    #[ignore = "a measurement"]
    fn measure_read_bandwidth() {
        let Ok(b) = WgpuBackend::new(Some(8 << 30)) else { return };
        let len = 100usize << 20; // 400 MB of f32
        let src = b.vec(len);
        let out = b.vec(1 << 20);
        for (unroll, wg) in [(1u32, 128u32), (2, 128), (4, 128), (4, 256), (8, 256)] {
            let body = format!(
                r#"
@group(0) @binding(0) var<storage, read> s4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> unused: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let n4 = p[0].x;
    let threads = nw.x * {wg}u;
    var acc = vec4<f32>(0.0);
    var i = id.x;
    loop {{
        if (i + {last}u * threads >= n4) {{ break; }}
{loads}
        i += {unroll}u * threads;
    }}
    o[id.x % 1048576u] = acc.x + acc.y + acc.z + acc.w;
}}
"#,
                last = unroll - 1,
                loads = (0..unroll).map(|u| format!("        acc += s4[i + {u}u * threads];")).collect::<Vec<_>>().join("\n"),
            );
            let name: &'static str = Box::leak(format!("test-read-{unroll}-{wg}").into_boxed_str());
            let pipeline = b.gpu.named_pipeline(name, || body);
            let groups = 170 * 16;
            let run = || {
                let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 };
                for _ in 0..8 {
                    rq.dispatch_kept(&pipeline, buffer(&src), buffer(&src), buffer(&out), &[(len / 4) as u32], (groups, 1, 1));
                }
                rq.read_range(&out, 0, 1);
                Box::new(rq).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let secs = t.elapsed().as_secs_f64() / 3.0 / 8.0;
            eprintln!("reading 400 MB, {unroll} loads in flight a thread, {wg} threads a workgroup: {:.0} GB/s", (len * 4) as f64 / secs / 1e9);
        }
    }

    /// A residual's add and the next norm in one dispatch give the two ops' answer; the vec4 norm gives the scalar
    /// one's (within an f32 step: its sums run in another order).
    #[test]
    fn an_add_and_norm_in_one_are_the_two() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(31);
        for (rows, n) in [(1usize, 5120usize), (3, 256), (2, 20)] {
            let (x0, y0, w0): (Vec<f32>, Vec<f32>, Vec<f32>) = ((0..rows * n).map(|_| r()).collect(), (0..rows * n).map(|_| r()).collect(), (0..n).map(|_| r()).collect());
            let up = |v: &[f32]| {
                let d = b.vec(v.len());
                DeviceChain::upload(&b, &d, v);
                d
            };
            let (xa, xb, y, w, oa, ob) = (up(&x0), up(&x0), up(&y0), up(&w0), b.vec(rows * n), b.vec(rows * n));
            let mut rec = b.begin();
            rec.add(&xa, &y);
            rec.rmsnorm_rows(&xa, &w, &oa, rows, 1e-6);
            rec.add_rmsnorm_rows(&xb, &y, &w, &ob, rows, 1e-6);
            for v in [&xa, &xb, &oa, &ob] {
                rec.read(v);
            }
            let got = rec.finish();
            assert_eq!(got[0], got[1], "the sums alike, {rows} rows of {n}");
            // the expected norm on the host
            let want: Vec<f32> = got[0].chunks_exact(n).flat_map(|row| {
                let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-6).sqrt();
                row.iter().zip(&w0).map(move |(v, g)| v * inv * g).collect::<Vec<_>>()
            }).collect();
            close(&got[2], &want, "the two ops");
            close(&got[3], &want, "in one");
        }
    }

}
