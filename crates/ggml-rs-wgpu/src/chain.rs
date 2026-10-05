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

/// [`RMSNORM`] of row `wg.x` of `x` (rows of `p[0].x`), every row with the same weights `w`.
const RMSNORM_ROWS: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let at = wg.x * n;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = x[at + i]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    for (var i = li; i < n; i += 256u) { y[at + i] = x[at + i] * inv * bitcast<f32>(w[i]); }
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
/// `sigmoid(z)`). `DK` (the heads' size, `k_dim` = `v_dim`) is made a constant. `p[0]`: the value heads, the key heads;
/// `p[1]`: the tokens, the bits of the q scale and of eps, the sigmoid gate. (A row read from memory at each token:
/// 43 us a layer for a decode step of Qwen3.8 27B, 30 us a token of a prompt.)
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
    var row: array<vec4<f32>, DK4>;
    for (var c = 0u; c < DK4; c++) { row[c] = st[base + c]; }
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
        for (var c = 0u; c < DK4; c++) {
            let s = row[c] * g;
            let kk = kn[c];
            kv += s.x * kk.x;
            kv += s.y * kk.y;
            kv += s.z * kk.z;
            kv += s.w * kk.w;
        }
        let delta = (cv[r0 + 2u * nk * DK + h * DK + i] - kv) * bt;
        var core = 0.0;
        for (var c = 0u; c < DK4; c++) {
            let s = row[c] * g + delta * kn[c];
            row[c] = s;
            let qq = qn[c];
            core += s.x * qq.x;
            core += s.y * qq.y;
            core += s.z * qq.z;
            core += s.w * qq.w;
        }
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
    for (var c = 0u; c < DK4; c++) { st[base + c] = row[c]; }
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

    fn holds(&self, w: &QuantizedTensor) -> bool {
        w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).is_some_and(|q| Arc::ptr_eq(&q.gpu, &self.gpu))
    }

    fn begin(&self) -> Box<dyn ChainRecorder + '_> {
        Box::new(Recorder { backend: self, dispatches: Vec::new(), reads: Vec::new(), keep: true })
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

struct Recorder<'a> {
    backend: &'a WgpuBackend,
    /// Every op's dispatch, in order, run in one compute pass (a pass an op cost more than the ops).
    dispatches: Vec<Dispatch>,
    /// What to read back once they have run: the vector, the element it starts at, a staging buffer, the length.
    reads: Vec<(wgpu::Buffer, usize, wgpu::Buffer, usize)>,
    /// Whether bind groups are kept for the steps after ([`ChainRecorder::keep_groups`]).
    keep: bool,
}

impl Recorder<'_> {
    fn gpu(&self) -> &Gpu {
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
    fn dispatch_kept(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, words: &[u32], groups: (u32, u32, u32)) {
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
        self.dispatches.push((Arc::clone(pipeline), group, groups));
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
        self.dispatches.push((Arc::clone(pipeline), group, groups));
    }

    fn named(&self, name: &'static str, body: &'static str) -> Arc<wgpu::ComputePipeline> {
        self.gpu().named_pipeline(name, || format!("{HEAD}{body}"))
    }

    /// A dispatch of an eight-buffer kernel (`Gpu::wide_layout`), its bind group kept as `dispatch_kept`'s.
    fn dispatch_wide(&mut self, name: &'static str, body: &str, bufs: [&wgpu::Buffer; 8], words: &[u32], groups: (u32, u32, u32)) {
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
        self.dispatches.push((pipeline, group, groups));
    }
}

impl ChainRecorder for Recorder<'_> {
    fn matmul_rows(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        assert!(m > 0 && x.len >= m * k && y.len >= m * n, "chain: matmul [{n}, {k}] of {m} rows from {} into {}", x.len, y.len);
        // the kernel for these rows: the decode kernel for one, the one-row kernel for a few, the tiled one for a prompt
        let pipeline = self.gpu().pipeline(q.dtype, m).expect("uploaded weights have a pipeline");
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0];
            let groups = crate::shaders::grid(q.dtype, m, *rows);
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, groups);
        }
    }

    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32) {
        let pipeline = self.named("chain-rmsnorm", RMSNORM);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[x.len as u32, eps.to_bits()], (1, 1, 1));
    }

    fn rmsnorm_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        assert!(rows > 0 && x.len % rows == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: rmsnorm of {rows} rows of {}", x.len);
        let pipeline = self.named("chain-rmsnorm-rows", RMSNORM_ROWS);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[(x.len / rows) as u32, eps.to_bits()], (rows as u32, 1, 1));
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

    fn matmul_f32_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(w.len >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows <= 65535, "chain: an f32 matmul [{n}, {k}] of {rows} rows");
        let pipeline = self.named("chain-matmul-f32", MATMUL_F32);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (n as u32, rows as u32, 1));
    }

    fn ssm_conv(&mut self, qkv: &DeviceVec, weight: &DeviceVec, state: &DeviceVec, out: &DeviceVec, rows: usize, channels: usize, kernel: usize) {
        assert!((2..=8).contains(&kernel) && qkv.len >= rows * channels && weight.len >= channels * kernel && state.len >= (kernel - 1) * channels && out.len >= rows * channels, "chain: a conv of {kernel} over {rows} rows of {channels}");
        let q = buffer(qkv);
        self.dispatch_wide("chain-ssm-conv", SSM_CONV, [q, buffer(weight), q, q, q, q, buffer(state), buffer(out)], &[channels as u32, rows as u32, kernel as u32], ((channels as u32).div_ceil(256), 1, 1));
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
        let source = DELTA_NET.replace("DK_VALUE", &d.k_dim.to_string());
        self.dispatch_wide(name, &source, [buffer(conv), buffer(z), buffer(beta_alpha), buffer(ssm_a), buffer(dt_bias), buffer(norm), buffer(state), buffer(out)], &words, (d.v_heads as u32, 1, 1));
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
        let part = self.named("chain-attention-part", ATTENTION_PART);
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, 1));
        let join = self.named("chain-attention-join", ATTENTION_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, 1, 1));
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

    fn finish(self: Box<Self>) -> Vec<Vec<f32>> {
        let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
        let start = std::time::Instant::now();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        {
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
        self.gpu().queue.submit([enc.finish()]);
        for (_, _, staging, len) in &self.reads {
            staging.slice(..(*len as u64 * 4).max(4)).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.gpu().device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("webgpu: device lost while waiting for a chain");
        let out = self
            .reads
            .iter()
            .map(|(_, _, staging, len)| {
                let view = staging.slice(..(*len as u64 * 4).max(4)).get_mapped_range().expect("webgpu: mapping a finished buffer");
                let v: Vec<f32> = view.chunks_exact(4).take(*len).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
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
            let mut rec = b.begin();
            rec.rope_rows(&qv, rows, n_h, hd, &tab, true);
            rec.rope_rows(&kv_, rows, n_kv, hd, &tab, true);
            rec.store_rows(&kv_, &cache, rows, kvd, past, 2 * kvd, 0);
            rec.store_rows(&vv, &cache, rows, kvd, past, 2 * kvd, kvd);
            rec.attention_rows(&qv, &cache, &out, rows, n_h, n_kv, hd, past, window, 0.125);
            rec.read_range(&out, 0, rows * qd);
            rec.read_range(&cache, past * 2 * kvd, kvd);
            let got = rec.finish();
            close(&got[1], &kc.data()[..kvd], "the first stored key");
            close(&got[0], want.data(), &format!("a prompt's attention, window {window:?}"));
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
    /// host made, at a small shape and at Qwen3.8 27B's (48 value heads on 16 key heads of 128).
    #[test]
    fn a_delta_net_matches_the_hosts() {
        use ggml_rs::Backend;
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        for (nv, nk, dim, kern, rows) in [(8usize, 4usize, 32usize, 4usize, 1usize), (8, 4, 32, 4, 7), (48, 16, 128, 4, 1), (48, 16, 128, 4, 5)] {
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
            let want = cpu.delta_net_step(&t(&qkv, vec![rows, ch]), &t(&z, vec![rows, nv * dim]), &t(&ba, vec![rows, 2 * nv]), &t(&cw, vec![ch, kern]),
                &t(&a, vec![nv]), &t(&dt, vec![nv]), &t(&nm, vec![dim]), &mut conv_c, &mut state_c, rows, nv, nk, dim, dim, nv / nk, scale, eps);
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
            rec.delta_net(&conv_out, &z_d, &ba_d, &a_d, &dt_d, &nm_d, &state_d, &out, DeltaNet { rows, v_heads: nv, k_heads: nk, k_dim: dim, v_dim: dim, scale_q: scale, eps, sigmoid_gate: false });
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
}
