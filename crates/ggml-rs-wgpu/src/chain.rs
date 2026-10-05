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

/// [`RMSNORM`] of row `wg.x` of `x` (rows of `p[0].x`), every row with the same weights `w`, or with `p[0].z` rows of
/// them the row's `wg.x % p[0].z` (a hyper-connection's streams).
const RMSNORM_ROWS: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let at = wg.x * n;
    var wrows = p[0].z;
    if (wrows == 0u) { wrows = 1u; }
    let wat = (wg.x % wrows) * n;
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
/// flight together, where one sum waited on each load in turn). `p[0]`: n, the bits of eps, weight rows.
const RMSNORM_ROWS4: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y4: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n4 = p[0].x / 4u;
    let at = wg.x * n4;
    var wrows = p[0].z;
    if (wrows == 0u) { wrows = 1u; }
    let wat = (wg.x % wrows) * n4;
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
    let at = wg.x * n4;
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
    let i = id.x;
    if (i < p[0].x) { y[i] = y[i] + bitcast<f32>(w[p[0].y]) * x[i]; }
}
"#;

/// `y += x` over `p[0].x` elements.
const ADD: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < p[0].x) { y[i] = y[i] + x[i]; }
}
"#;

/// `y[r] = silu(x[r][..ff]) * x[r][ff..]` for each of `p[0].y` rows, `ff = p[0].x`.
const SILU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
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
    let i = id.x;
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
    let i = id.x;
    if (i >= p[0].w * heads * half) { return; }
    let r = i / (heads * half);
    let h = (i / half) % heads;
    let k = i % half;
    let s = bitcast<f32>(w[r * rot + 2u * k]);
    let c = bitcast<f32>(w[r * rot + 2u * k + 1u]);
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
    let i = id.x;
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
    let i = id.x;
    if (i < p[0].x) { y[p[0].y + i] = x[p[0].z + i]; }
}
"#;

/// `y[r * p[0].x + i] = x[r * p[0].z + p[0].w + i]` for `r < p[0].y` rows of `p[0].x`: columns of each row.
const COPY_COLS: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = p[0].x;
    let i = id.x;
    if (i < width * p[0].y) { y[i] = x[(i / width) * p[0].z + p[0].w + i % width]; }
}
"#;

/// `y[i] = w[i] * sigmoid(x[i])` for `i < p[0].x` (`w` the values, `x` the gate), as the CPU's mul_sigmoid.
const MUL_SIGMOID: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < p[0].x) { y[i] = bitcast<f32>(w[i]) * (1.0 / (1.0 + exp(-x[i]))); }
}
"#;

/// `y[i] = silu(w[i]) * x[i]` for `i < p[0].x` (`w` the gate, `x` the up projection).
const SILU_MUL: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
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
        self.gpu.queue.submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("webgpu: device lost while reading a chain's vector");
        let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping a finished buffer");
        let host: Vec<f32> = view.chunks_exact(4).take(len).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
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
        self.gpu.queue.submit([enc.finish()]);
        Box::new(Aliased { v: DeviceVec { len: self.v.len, inner: Arc::new(copy) }, gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial) })
    }
}

fn le_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for f in data {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

impl DeviceChain for WgpuBackend {
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

    fn zero(&self, v: &DeviceVec) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(buffer(v), 0, None);
        self.gpu.queue.submit([enc.finish()]);
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
            self.gpu.queue.write_buffer(buffer(v), (offset * 4) as u64, &le_bytes(data));
        }
    }

    fn resize(&self, v: &DeviceVec, len: usize) -> DeviceVec {
        let grown = self.vec(len);
        let keep = v.len.min(len);
        if keep > 0 {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buffer(v), 0, buffer(&grown), 0, (keep * 4) as u64);
            self.gpu.queue.submit([enc.finish()]);
        }
        grown
    }

    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize {
        n_h * head_dim + n_h * cap.div_ceil(SPLIT).max(1) * (head_dim + 2)
    }

    fn attention_rows_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
        rows * n_h * head_dim + rows * n_h * kv_len.div_ceil(SPLIT).max(1) * (head_dim + 2)
    }

    fn qsa_attention_out_len(&self, rows: usize, n_h: usize, head_dim: usize, keep: usize, ratio: usize) -> usize {
        // the selection sorts 4096 blocks' keys and indices in a workgroup's memory (32 KB, past WebGPU's default 16)
        if self.gpu.limits.max_compute_workgroup_storage_size < 32768 {
            return 0;
        }
        rows * n_h * head_dim + rows * n_h * (keep * ratio + ratio).div_ceil(256) * (head_dim + 2)
    }

    fn holds_exl3(&self, w: &dyn ggml_rs::exl3::PackedLinear) -> bool {
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

    fn begin(&self) -> Box<dyn ChainRecorder + '_> {
        Box::new(Recorder { backend: self, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None })
    }
}

impl Recorder<'_> {
    /// [`ChainRecorder::exl3_rows`], the input `x`, or (`up` given) the SwiGLU `silu(x) * up` computed as the input
    /// transform reads it (a shared expert's down projection: a dispatch fewer).
    pub(crate) fn exl3_rows_of(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, x: &DeviceVec, up: Option<&DeviceVec>, y: &DeviceVec, rows: usize) {
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
            (self.scratch(rows * k), self.scratch(rows * splits as usize * n), self.scratch(rows * n), jobs)
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
        if rows == 1 {
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
        self.dispatch_wide("exl3-post", post, [buffer(&part), buffer(&c.svh), buffer(&jobs), &d, &d, &d, post_out, &drw], &[n as u32, splits], ((n / 128) as u32, rows as u32, 1));
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
        Recorder { backend, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None }
    }
}

/// One dispatch: its pipeline, bind group and grid.
type Dispatch = (Arc<wgpu::ComputePipeline>, wgpu::BindGroup, (u32, u32, u32));

/// A bind group a chain makes again step after step: its pipeline, its three buffers and its parameters.
pub(crate) type GroupKey = (usize, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, [u32; 8]);

/// A bind group of the eight-buffer layout a chain makes again step after step (`Gpu::wide_layout`).
pub(crate) type WideKey = (usize, [wgpu::Buffer; 8], [u32; 8]);

/// Bind groups kept before the cache starts over (a cache grown from buffers that were replaced).
const KEEP_GROUPS: usize = 16384;

/// Dispatches a piece of a run submits ([`Recorder::finish`]'s).
const PIECE: usize = 128;

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
    /// Inputs of several rows quantized to int8 for the int8 kernels so far (the vector, its rows and width, the int8
    /// rows): each quantized once for the matmuls that read it, until something writes it.
    q8: Vec<(wgpu::Buffer, usize, usize, DeviceVec)>,
    /// Inputs of a prompt's rows as f16 for the tensor cores so far, as `q8`.
    x16: Vec<(wgpu::Buffer, usize, usize, DeviceVec)>,
    /// The f16 queries and cache rows of the tensor cores' attention: each attention's converted into them as it runs.
    att16: Option<(DeviceVec, DeviceVec)>,
    /// The parts of a tensor-core matmul split along k, each split matmul's in turn.
    parts: Option<DeviceVec>,
}

impl Recorder<'_> {
    /// A dispatch recorded; a piece of [`PIECE`] submitted as soon as it is recorded (unless profiled), so the GPU runs
    /// a run's first ops while the CPU records the rest. A later upload (`Queue::write_buffer`) lands after the pieces
    /// already submitted and before the ones after, as the recording's order has it.
    fn push(&mut self, d: Dispatch) {
        self.dispatches.push(d);
        if self.dispatches.len() >= PIECE && !crate::profile::chain_on() {
            let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
            let start = std::time::Instant::now();
            let mut piece = self.gpu().device.create_command_encoder(&Default::default());
            {
                let mut pass = piece.begin_compute_pass(&Default::default());
                for (pipeline, group, (x, y, z)) in &self.dispatches {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(*x, *y, *z);
                }
            }
            self.gpu().queue.submit([piece.finish()]);
            self.dispatches.clear();
            crate::profile::add(&crate::profile::CHAIN_ENCODE, start);
        }
    }

    /// A vector of `len` for this recording alone (a prompt's scratch): from the GPU's pool, given back when the
    /// recording has run, so nothing may keep it. Its values are whatever it last held.
    pub(crate) fn scratch(&mut self, len: usize) -> DeviceVec {
        let bytes = ((len.max(1) * 4) as u64).next_power_of_two().max(256);
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
        self.gpu().queue.write_buffer(&buf, 0, &bytes);
        buf
    }

    /// A dispatch whose bind group is the same every step (its buffers and parameters): made once and kept.
    /// `b` written by what was just recorded: its int8 rows (if quantized) are stale.
    fn wrote(&mut self, b: &wgpu::Buffer) {
        if !self.q8.is_empty() {
            self.q8.retain(|(x, ..)| x != b);
        }
        if !self.x16.is_empty() {
            self.x16.retain(|(x, ..)| x != b);
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
        let kv_len = past + rows;
        let runs = kv_len.div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= kv_len * 2 * n_kv * head_dim && q.len >= rows * n_h * head_dim && out.len >= rows * n_h * head_dim + rows * n_h * runs * (head_dim + 2),
            "chain: a prompt's attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, past as u32, window.unwrap_or(0) as u32, runs as u32, scale.to_bits(), rows as u32]);
        let part = self.named("chain-attention-rows-part", ATTENTION_ROWS_PART);
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, rows as u32));
        let join = self.named("chain-attention-rows-join", ATTENTION_ROWS_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, rows as u32, 1));
    }

    /// [`ChainRecorder::attention_rows`] on the tensor cores ([`ATTENTION_COOP`]): the queries and the cache's rows
    /// as f16 (the cache's each time, to the last query: a few microseconds a layer), the scores f16 into f32 and the
    /// softmax in f32, its weights f16. False (nothing recorded) where the device has no tensor cores, a window is
    /// kept, the head is not 64, 128 or 256 wide, or there are fewer than 16 rows.
    pub(crate) fn attention_rows_coop(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) -> bool {
        let name = match head_dim {
            64 => "chain-attention-coop-64",
            128 => "chain-attention-coop-128",
            256 => "chain-attention-coop-256",
            _ => return false,
        };
        if window.is_some() || rows < 16 || n_kv == 0 || n_h % n_kv != 0 || !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return false;
        }
        let kv_len = past + rows;
        let (qs, row) = (n_h * head_dim, 2 * n_kv * head_dim);
        let (rp, kp) = (rows.div_ceil(32) * 32, kv_len.div_ceil(32) * 32);
        // the padding's rows are stored past the output, in its scratch (at least as long: 16 rows or more)
        assert!(
            kv.len >= kv_len * row && q.len >= rows * qs && out.len >= rp * qs && out.len >= self.backend.attention_rows_out_len(rows, n_h, head_dim, kv_len),
            "chain: a prompt's attention's buffers"
        );
        // the f16 copies: one pair a recording, grown as it needs (each attention's converted as it runs)
        let (q16, kv16) = match self.att16.take() {
            Some((a, b)) if a.len >= rp * qs / 2 && b.len >= kp * row / 2 => (a, b),
            _ => (self.scratch(rp * qs / 2), self.scratch(kp * row / 2)),
        };
        let conv = self.gpu().named_pipeline("chain-x-f16", || crate::shaders::X_F16.to_string());
        let d = self.gpu().dummy().clone();
        for (src, dst, width, n, padded) in [(q, &q16, qs, rows, rp), (kv, &kv16, row, kv_len, kp)] {
            let groups = ((padded * width / 2) as u32).div_ceil(256);
            self.dispatch_kept(&conv, &d, buffer(src), buffer(dst), &[width as u32, n as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        }
        let pipeline = self.gpu().named_pipeline(name, || attention_coop(head_dim));
        let words = [n_h as u32, n_kv as u32, past as u32, rows as u32, kv_len as u32, scale.to_bits(), 0, 0];
        self.dispatch_kept(&pipeline, buffer(&kv16), buffer(&q16), buffer(out), &words, (n_h as u32, (rp / 32) as u32, 1));
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
        // the tokens' rows as f16, padded to the tile, once for every matmul that reads them until something writes x
        let tile = crate::shaders::COOP_TILE;
        let padded = (m as u32).div_ceil(tile) as usize * tile as usize;
        let xb = buffer(x).clone();
        let x16 = match self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            Some((.., v)) => v.clone(),
            None => {
                let v = self.scratch(padded * k / 2);
                let conv = self.gpu().named_pipeline("chain-x-f16", || crate::shaders::X_F16.to_string());
                let pairs = (padded * k / 2) as u32;
                let groups = pairs.div_ceil(256);
                let d = self.gpu().dummy().clone();
                self.dispatch_kept(&conv, &d, buffer(x), buffer(&v), &[k as u32, m as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
                self.x16.push((xb, m, k, v.clone()));
                v
            }
        };
        let dtype = q.dtype;
        let pipeline = self.gpu().named_pipeline(name, || crate::shaders::coop_tiled(dtype).expect("a K-quant's tensor-core kernel"));
        // a matmul of too few tiles to fill the GPU's last wave split along k: each split's sums into a part of
        // scratch, then the parts added into y
        let units = self.gpu().coop_units();
        let steps = (k / 32) as u32;
        let tiles = q.chunks.iter().map(|(_, _, rows)| rows.div_ceil(tile)).max().unwrap_or(1) * (m as u32).div_ceil(tile);
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
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, splits, 0];
            self.dispatch_kept(&pipeline, chunk, buffer(&x16), &out, &words, (rows.div_ceil(tile), (m as u32).div_ceil(tile), splits));
        }
        if let Some(parts) = parts {
            let sum = self.named("chain-coop-sum", COOP_SUM);
            let groups = ((m * n) as u32).div_ceil(256);
            let d = self.gpu().dummy().clone();
            self.dispatch_kept(&sum, &d, buffer(&parts), buffer(y), &[(m * n) as u32, splits], (groups.min(65535), groups.div_ceil(65535), 1));
        }
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
        if (2..=crate::shaders::MULTI_MAX).contains(&m) && q8 && self.matmul_rows_q8(w, x, y, m) {
            return;
        }
        // a prompt's rows: the tensor cores where the device has them (f16 into f32), else the int8 tiled kernel
        // (llama.cpp's MMQ's arithmetic), where the type has one
        if m > crate::shaders::MULTI_MAX && self.matmul_rows_coop(w, x, y, m) {
            return;
        }
        if m > crate::shaders::MULTI_MAX && q8 && self.matmul_rows_tq8(w, x, y, m) {
            return;
        }
        self.matmul_rows_f32(w, x, y, m);
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
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits()], (rows as u32, 1, 1));
    }

    fn add_rmsnorm_rows(&mut self, x: &DeviceVec, y: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        let n = x.len / rows.max(1);
        if rows == 0 || n % 4 != 0 || n * rows != x.len || y.len < x.len || w.len < n || out.len < x.len || Arc::ptr_eq(&x.inner, &out.inner) {
            self.add(x, y);
            self.rmsnorm_rows(x, w, out, rows, eps);
            return;
        }
        let d = self.gpu().dummy().clone();
        self.dispatch_wide("chain-add-rmsnorm-rows4", ADD_RMSNORM_ROWS4, [buffer(y), buffer(w), &d, &d, &d, &d, buffer(x), buffer(out)], &[n as u32, eps.to_bits()], (rows as u32, 1, 1));
    }

    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec) {
        let pipeline = self.named("chain-add", ADD);
        self.dispatch_kept(&pipeline, buffer(y), buffer(y), buffer(acc), &[acc.len as u32], ((acc.len as u32).div_ceil(256), 1, 1));
    }

    fn silu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        let ff = out.len / rows;
        assert!(rows > 0 && out.len == rows * ff && fused.len >= 2 * out.len, "chain: SwiGLU of {rows} rows");
        let pipeline = self.named("chain-silu-mul-split", SILU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[ff as u32, rows as u32], ((out.len as u32).div_ceil(256), 1, 1));
    }

    fn gelu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        let ff = out.len / rows;
        assert!(rows > 0 && out.len == rows * ff && fused.len >= 2 * out.len, "chain: GeGLU of {rows} rows");
        let pipeline = self.named("chain-gelu-mul-split", GELU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[ff as u32, rows as u32], ((out.len as u32).div_ceil(256), 1, 1));
    }

    fn keep_groups(&mut self, keep: bool) {
        self.keep = keep;
    }

    fn exl3_rows(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.exl3_rows_of(w, x, None, y, rows);
    }

    fn rmsnorm_streams(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, eps: f32) {
        let n = x.len / (rows * streams).max(1);
        assert!(rows > 0 && streams > 0 && n * rows * streams == x.len && w.len >= streams * n && out.len >= x.len, "chain: a norm of {rows} rows of {streams} streams");
        let pipeline = self.named("chain-rmsnorm-rows", RMSNORM_ROWS);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), streams as u32], ((rows * streams) as u32, 1, 1));
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
        self.dispatch_kept(&pipeline, buffer(weights), buffer(y), buffer(acc), &[len as u32, at as u32], ((len as u32).div_ceil(256), 1, 1));
    }

    fn copy_cols(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, width: usize, stride: usize, at: usize) {
        assert!(rows > 0 && at + width <= stride && src.len >= rows * stride && dst.len >= rows * width, "chain: {rows} rows' columns {at}..{} of {stride}", at + width);
        let pipeline = self.named("chain-copy-cols", COPY_COLS);
        self.dispatch_kept(&pipeline, buffer(src), buffer(src), buffer(dst), &[width as u32, rows as u32, stride as u32, at as u32], (((rows * width) as u32).div_ceil(256), 1, 1));
    }

    fn rope_partial_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, rot: usize, table: &DeviceVec) {
        assert!(rot > 0 && rot % 2 == 0 && rot <= head_dim && x.len >= rows * heads * head_dim && table.len >= rows * rot, "chain: RoPE of {rot} of {head_dim}");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * rot / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, 1, rows as u32, rot as u32], (pairs.div_ceil(256), 1, 1));
    }

    fn mul_sigmoid(&mut self, x: &DeviceVec, gate: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && gate.len >= len && out.len >= len && !Arc::ptr_eq(&x.inner, &out.inner), "chain: a gate of {len}");
        let pipeline = self.named("chain-mul-sigmoid", MUL_SIGMOID);
        self.dispatch_kept(&pipeline, buffer(x), buffer(gate), buffer(out), &[len as u32], ((len as u32).div_ceil(256), 1, 1));
    }

    fn silu_mul(&mut self, gate: &DeviceVec, up: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(gate.len >= len && up.len >= len && out.len >= len, "chain: a SwiGLU of {len}");
        let pipeline = self.named("chain-silu-mul", SILU_MUL);
        self.dispatch_kept(&pipeline, buffer(gate), buffer(up), buffer(out), &[len as u32], ((len as u32).div_ceil(256), 1, 1));
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
        assert!(k % 2 == 0 && w.len * 2 >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows <= 65535, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
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
        }
    }

    fn matmul_f32_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(w.len >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows <= 65535, "chain: an f32 matmul [{n}, {k}] of {rows} rows");
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
        }
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
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, neox as u32, rows as u32], (pairs.div_ceil(256), 1, 1));
    }

    fn store_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, len: usize, start: usize, stride: usize, at: usize) {
        assert!(src.len >= rows * len && at + len <= stride && dst.len >= (start + rows) * stride, "chain: storing {rows} rows");
        let pipeline = self.named("chain-store-rows", STORE_ROWS);
        let params = self.uniform(&[len as u32, start as u32, stride as u32, at as u32, rows as u32]);
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, (((rows * len) as u32).div_ceil(256), 1, 1));
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
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, ((len as u32).div_ceil(256), 1, 1));
    }

    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        let runs = kv_len.saturating_sub(lo).div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= cap * 2 * n_kv * head_dim && kv_len <= cap && out.len >= n_h * head_dim + n_h * runs * (head_dim + 2),
            "chain: attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, kv_len as u32, lo as u32, runs as u32, scale.to_bits()]);
        // a head a multiple of 4 wide (at most 512): the vec4 kernel
        let part = if head_dim % 4 == 0 && head_dim <= 512 { self.gpu().named_pipeline("chain-attention-part4", || ATTENTION_PART4.to_string()) } else { self.named("chain-attention-part", ATTENTION_PART) };
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, 1));
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
        self.dispatch_wide("chain-qsa-scores", QSA_SCORES, [buffer(q), buffer(pooled), &dd, &dd, &dd, &dd, buffer(scores), &drw], &[rows as u32, heads as u32, d as u32, nb as u32, first as u32, ratio as u32, scale.to_bits()], ((nb as u32).div_ceil(256), rows as u32, 1));
    }

    fn qsa_select(&mut self, scores: &DeviceVec, list: &DeviceVec, rows: usize, nb: usize, first: usize, ratio: usize, keep: usize) {
        assert!(nb <= 4096 && keep > 0 && scores.len >= rows * nb && list.len >= rows * keep, "chain: QSA's selection of {keep} of {nb} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-qsa-select", QSA_SELECT, [buffer(scores), &dd, &dd, &dd, &dd, &dd, buffer(list), &drw], &[rows as u32, nb as u32, first as u32, ratio as u32, keep as u32], (rows as u32, 1, 1));
    }

    fn qsa_attention(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) {
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

    fn argmax_softmax(&mut self, x: &DeviceVec, out: &DeviceVec) {
        assert!(x.len > 0 && out.len >= 3, "chain: a draft's token of {} logits", x.len);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-argmax-softmax", ARGMAX_SOFTMAX, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[x.len as u32], (1, 1, 1));
    }

    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize) {
        assert!(offset + len <= v.len, "chain: reading {len} at {offset} of {}", v.len);
        let staging = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-read"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.reads.push((buffer(v).clone(), offset, staging, len));
    }

    fn finish(mut self: Box<Self>) -> Vec<Vec<f32>> {
        let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
        let start = std::time::Instant::now();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        // profiled: each dispatch a pass between two timestamps (up to the query set's 4096)
        let timed = crate::profile::chain_on() && self.gpu().device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut stamps = None;
        if timed {
            let n = self.dispatches.len().min(wgpu::QUERY_SET_MAX_QUERIES as usize / 2);
            let device = &self.gpu().device;
            let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("oaiy-chain-profile"), ty: wgpu::QueryType::Timestamp, count: (2 * n).max(2) as u32 });
            for (i, (pipeline, group, (x, y, z))) in self.dispatches.iter().enumerate() {
                let timestamp_writes = (i < n).then(|| wgpu::ComputePassTimestampWrites { query_set: &set, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, *z);
            }
            let bytes = (16 * n).max(16) as u64;
            let resolved = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-profile"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-profile-read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            if n > 0 {
                enc.resolve_query_set(&set, 0..2 * n as u32, &resolved, 0);
                enc.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
            }
            stamps = Some((staging, n));
        } else {
            // what the pieces submitted while recording left (`push`), with the reads
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (pipeline, group, (x, y, z)) in &self.dispatches {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, *z);
            }
        }
        for (from, offset, staging, len) in &self.reads {
            if *len > 0 {
                enc.copy_buffer_to_buffer(from, (*offset * 4) as u64, staging, 0, (*len * 4) as u64);
            }
        }
        let command = enc.finish();
        crate::profile::add(&crate::profile::CHAIN_ENCODE, start);
        let submitted = std::time::Instant::now();
        self.gpu().queue.submit([command]);
        for (_, _, staging, len) in &self.reads {
            staging.slice(..(*len as u64 * 4).max(4)).map_async(wgpu::MapMode::Read, |_| {});
        }
        if let Some((staging, _)) = &stamps {
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.gpu().device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("webgpu: device lost while waiting for a chain");
        crate::profile::add(&crate::profile::CHAIN_WAIT, submitted);
        let pooled = std::mem::take(&mut self.pooled);
        self.gpu().unpool(pooled);
        if let Some((staging, n)) = stamps {
            let period = self.gpu().queue.get_timestamp_period() as f64;
            let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping the profile");
            let ticks: Vec<u64> = view.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
            drop(view);
            staging.unmap();
            let mut k = crate::profile::KERNELS.lock().unwrap_or_else(|p| p.into_inner());
            for (i, (pipeline, _, _)) in self.dispatches.iter().take(n).enumerate() {
                let ns = (ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period) as u64;
                let e = k.entry(self.gpu().name_of(pipeline)).or_default();
                e.0 += ns;
                e.1 += 1;
            }
        }
        let out = self
            .reads
            .iter()
            .map(|(_, _, staging, len)| {
                let view = staging.slice(..(*len as u64 * 4).max(4)).get_mapped_range().expect("webgpu: mapping a finished buffer");
                // copied as bytes (a 512-row chunk's cache rows, 64 MB: a value at a time some 15 ms)
                let mut v = vec![0f32; *len];
                bytemuck::cast_slice_mut::<f32, u8>(&mut v).copy_from_slice(&view[..*len * 4]);
                drop(view);
                staging.unmap();
                v
            })
            .collect();
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

    /// A prompt's attention on the tensor cores gives the f32 kernels' within f16's rounding: GQA, heads 64, 128 and
    /// 256 wide, from the cache's start and past it, the rows a tile's multiple and not, its scores spread and peaked.
    #[test]
    fn a_prompts_attention_on_the_tensor_cores_is_the_f32_kernels() {
        let Ok(b) = WgpuBackend::new(Some(2 << 30)) else { return };
        if !b.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return;
        }
        for (n_h, n_kv, hd, past, rows, spread) in [(8usize, 2usize, 64usize, 300usize, 37usize, 1.0f32), (16, 2, 128, 0, 100, 2.0), (24, 4, 256, 214, 64, 1.0), (24, 4, 256, 1000, 150, 3.0)] {
            let (qd, row, kv_len) = (n_h * hd, 2 * n_kv * hd, past + rows);
            let mut next = rng((qd + past) as u32);
            let q: Vec<f32> = (0..rows * qd).map(|_| next() * spread).collect();
            let cache: Vec<f32> = (0..kv_len * row).map(|_| next() * spread).collect();
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
            eprintln!("{n_h} heads ({n_kv} kv) {hd} wide, {rows} rows after {past}: cosine {cos:.7}, worst {worst:.2e} of {top:.2}");
            assert!(cos > 0.99999 && worst <= 4e-3 * top, "{n_h} heads {hd} wide, {rows} rows after {past}: cosine {cos}, worst {worst} of {top}");
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

    /// A matrix of f16 values held as f16 (two to a word) multiplies as it does held as f32: a long row's step the same
    /// bits (summed the same way), a short row's and a prompt's within rounding; a matrix not all f16 values is not
    /// made.
    #[test]
    fn an_f16_matrix_multiplies_as_its_f32_one() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let mut r = rng(57);
        assert!(b.vec_f16(&[0.1, 0.5]).is_none(), "0.1 is no f16");
        for (n, k, rows) in [(324usize, 10240usize, 1usize), (513, 2560, 1), (1030, 324, 1), (70, 100, 1), (324, 10240, 70), (1030, 324, 65)] {
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
            } else {
                let scale = (k as f32).sqrt();
                for (i, (a, e)) in h.iter().zip(&f).enumerate() {
                    assert!((a - e).abs() <= 1e-4 * scale, "[{n}, {k}] of {rows} rows [{i}]: {a} against {e}");
                }
            }
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
            let raw: Vec<u8> = (0..n * (k / 256) * 110).map(|_| ((next() + 1.0) * 100.0) as u8).collect();
            let w = ggml_rs::Backend::to_device_quant(&b, ggml_rs::QuantizedTensor::from_bytes_cpu(raw, vec![n, k], GgmlType::Q3_K));
            let (x, y) = (b.vec(m * k), b.vec(m * n));
            let units = b.gpu.coop_units();
            let chosen = crate::shaders::coop_splits((n as u32).div_ceil(128) * (m as u32).div_ceil(128), units, (k / 32) as u32);
            let mut line = format!("{what} [{n}, {k}] (chosen {chosen}):");
            for split in [None, Some(1), Some(2), Some(3), Some(4)] {
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
        assert_eq!(coop_splits(192, 170, 160), 5);
        assert_eq!(coop_splits(384, 170, 160), 3);
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
            let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None };
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
                    let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None };
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
                    let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None };
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
                let mut rq = Recorder { backend: &b, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None };
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
