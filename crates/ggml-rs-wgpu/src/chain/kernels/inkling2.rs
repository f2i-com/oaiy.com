//! Inkling-2 / Inkling-2-Loop single-submit GPU decode kernels:
//! Block Attention Residuals (`mix_depth`), cross-pass controller cat, static/dynamic causal
//! short convolutions, per-head L2 clipping, V unweighted RMSNorm + KV ring-cache store,
//! relative-position-bias sink GQA attention, block-boundary push, and final logit softcap.

/// Block Attention Residuals (`Inkling2BlockAttentionResidual` / `mix_depth`) for `seq >= 1`:
/// computes softmax over `[blocks[0..n_blocks], norm(cur)]` for `n_queries` learned query vectors.
/// `blocks`: `[n_blocks, seq, hidden]`, `cur`: `[seq, hidden]`, `out`: `[n_queries, seq, hidden]`.
/// Grid: `(n_queries, seq, 1)`.
/// `p[0]`: hidden, n_blocks (< 32), n_queries, cur_is_normed (0 or 1); `p[1]`: seq (0 treated as 1).
pub(in crate::chain) const INKLING2_MIX_DEPTH: &str = r#"
@group(0) @binding(0) var<storage, read> blocks: array<f32>;
@group(0) @binding(1) var<storage, read> cur: array<f32>;
@group(0) @binding(2) var<storage, read> query: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 256>;
var<workgroup> scores: array<f32, 32>;
var<workgroup> cur_inv: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let hidden = p[0].x;
    let n_blocks = p[0].y;
    let q_idx = wg.x;
    let seq = max(1u, p[1].x);
    let t = wg.y;
    if (q_idx >= p[0].z || t >= seq) { return; }

    let c_off = t * hidden;
    if (p[0].w == 0u) {
        var sq = 0.0;
        for (var i = li; i < hidden; i += 256u) {
            let v = cur[c_off + i];
            sq += v * v;
        }
        part[li] = sq;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st /= 2u) {
            if (li < st) { part[li] += part[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) { cur_inv = inverseSqrt(part[0] / f32(hidden) + 1e-6); }
    } else {
        if (li == 0u) { cur_inv = 1.0; }
    }
    workgroupBarrier();
    let inv = cur_inv;
    let inv_sqrt_d = inverseSqrt(f32(hidden));
    let q_off = q_idx * hidden;
    let o_off = (q_idx * seq + t) * hidden;

    for (var s = 0u; s < n_blocks; s++) {
        var d = 0.0;
        let b_off = (s * seq + t) * hidden;
        for (var i = li; i < hidden; i += 256u) {
            d += blocks[b_off + i] * query[q_off + i];
        }
        part[li] = d;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st /= 2u) {
            if (li < st) { part[li] += part[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) { scores[s] = part[0] * inv_sqrt_d; }
        workgroupBarrier();
    }

    var dc = 0.0;
    for (var i = li; i < hidden; i += 256u) {
        dc += (cur[c_off + i] * inv) * query[q_off + i];
    }
    part[li] = dc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        scores[n_blocks] = part[0] * inv_sqrt_d;
        var mx = scores[0];
        for (var s = 1u; s <= n_blocks; s++) { mx = max(mx, scores[s]); }
        var sum_e = 0.0;
        for (var s = 0u; s <= n_blocks; s++) {
            let e = exp(scores[s] - mx);
            scores[s] = e;
            sum_e += e;
        }
        let inv_sum = 1.0 / sum_e;
        for (var s = 0u; s <= n_blocks; s++) { scores[s] *= inv_sum; }
    }
    workgroupBarrier();

    let w_cur = scores[n_blocks] * inv;
    for (var i = li; i < hidden; i += 256u) {
        var acc = w_cur * cur[c_off + i];
        for (var s = 0u; s < n_blocks; s++) {
            acc += scores[s] * blocks[(s * seq + t) * hidden + i];
        }
        out[o_off + i] = acc;
    }
}
"#;

/// Subgroup + `vec4<f32>` version of [`INKLING2_MIX_DEPTH`] for 32-lane subgroup devices and
/// `hidden % 4 == 0`: fuses `cur` norm + `cur·query` into one pass and computes all `n_blocks`
/// dot products using `subgroupAdd` with only 2 `workgroupBarrier()`s total (down from up to 89).
pub(in crate::chain) const INKLING2_MIX_DEPTH_SG: &str = r#"
@group(0) @binding(0) var<storage, read> blocks4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cur4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> query4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out4: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wsq: array<f32, 8>;
var<workgroup> wdc: array<f32, 8>;
var<workgroup> wdot: array<f32, 256>;
var<workgroup> scores: array<f32, 32>;
var<workgroup> cur_inv: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let hidden = p[0].x;
    let n_blocks = p[0].y;
    let q_idx = wg.x;
    let seq = max(1u, p[1].x);
    let t = wg.y;
    if (q_idx >= p[0].z || t >= seq) { return; }

    let h4 = hidden / 4u;
    let c_off4 = t * h4;
    let q_off4 = q_idx * h4;
    let o_off4 = (q_idx * seq + t) * h4;
    let inv_sqrt_d = inverseSqrt(f32(hidden));

    var sq = 0.0;
    var dc = 0.0;
    for (var i = li; i < h4; i += 256u) {
        let cv = cur4[c_off4 + i];
        let qv = query4[q_off4 + i];
        sq += dot(cv, cv);
        dc += dot(cv, qv);
    }
    sq = subgroupAdd(sq);
    dc = subgroupAdd(dc);
    if (lane == 0u) {
        wsq[warp] = sq;
        wdc[warp] = dc;
    }

    for (var s = 0u; s < n_blocks; s++) {
        let b_off4 = (s * seq + t) * h4;
        var d = 0.0;
        for (var i = li; i < h4; i += 256u) {
            d += dot(blocks4[b_off4 + i], query4[q_off4 + i]);
        }
        d = subgroupAdd(d);
        if (lane == 0u) { wdot[s * 8u + warp] = d; }
    }
    workgroupBarrier();

    if (li == 0u) {
        let tot_sq = ((wsq[0] + wsq[1]) + (wsq[2] + wsq[3])) + ((wsq[4] + wsq[5]) + (wsq[6] + wsq[7]));
        let inv = select(1.0, inverseSqrt(tot_sq / f32(hidden) + 1e-6), p[0].w == 0u);
        cur_inv = inv;
        let tot_dc = ((wdc[0] + wdc[1]) + (wdc[2] + wdc[3])) + ((wdc[4] + wdc[5]) + (wdc[6] + wdc[7]));
        scores[n_blocks] = (tot_dc * inv) * inv_sqrt_d;
        for (var s = 0u; s < n_blocks; s++) {
            let b = s * 8u;
            let ds = ((wdot[b] + wdot[b + 1u]) + (wdot[b + 2u] + wdot[b + 3u]))
                + ((wdot[b + 4u] + wdot[b + 5u]) + (wdot[b + 6u] + wdot[b + 7u]));
            scores[s] = ds * inv_sqrt_d;
        }
        var mx = scores[0];
        for (var s = 1u; s <= n_blocks; s++) { mx = max(mx, scores[s]); }
        var sum_e = 0.0;
        for (var s = 0u; s <= n_blocks; s++) {
            let e = exp(scores[s] - mx);
            scores[s] = e;
            sum_e += e;
        }
        let inv_sum = 1.0 / sum_e;
        for (var s = 0u; s <= n_blocks; s++) { scores[s] *= inv_sum; }
    }
    workgroupBarrier();

    let w_cur = scores[n_blocks] * cur_inv;
    for (var i = li; i < h4; i += 256u) {
        var acc = cur4[c_off4 + i] * w_cur;
        for (var s = 0u; s < n_blocks; s++) {
            acc += blocks4[(s * seq + t) * h4 + i] * scores[s];
        }
        out4[o_off4 + i] = acc;
    }
}
"#;

/// Concatenated unweighted RMSNorm of `(cur[cur_off + t*H..], msg[msg_off + t*H..])` into `out[t*2H..]`.
/// Grid: `(2, seq, 1)`. `p[0]`: hidden, cur_off, msg_off, 0.
pub(in crate::chain) const INKLING2_CONTROL_CAT: &str = r#"
@group(0) @binding(0) var<storage, read> cur: array<f32>;
@group(0) @binding(1) var<storage, read> msg: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let hidden = p[0].x;
    let is_msg = wg.x == 1u;
    let t = wg.y;
    let src_off = select(p[0].y + t * hidden, p[0].z + t * hidden, is_msg);
    let dst_off = t * 2u * hidden + select(0u, hidden, is_msg);

    var sq = 0.0;
    for (var i = li; i < hidden; i += 256u) {
        let v = select(cur[src_off + i], msg[src_off + i], is_msg);
        sq += v * v;
    }
    part[li] = sq;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(part[0] / f32(hidden) + 1e-6);
    for (var i = li; i < hidden; i += 256u) {
        let v = select(cur[src_off + i], msg[src_off + i], is_msg);
        out[dst_off + i] = v * inv;
    }
}
"#;

/// Single-token static (`n_heads == 0`) or dynamic (`n_heads > 0`) causal depthwise short
/// convolution with residual + scaling, advancing `hist` (`[(kernel - 1) * channels]`) in place.
/// `p[0]`: channels, kernel, n_heads (0 = static), 0.
pub(in crate::chain) const INKLING2_SHORT_CONV_STEP: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> dyn_taps: array<f32>;
@group(0) @binding(6) var<storage, read_write> hist: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = id.x + id.y * 16776960u;
    let c = p[0].x;
    if (ch >= c) { return; }
    let k = p[0].y;
    let past = k - 1u;
    let n_heads = p[0].z;
    let cur_v = x[ch];
    var acc = 0.0;
    if (n_heads == 0u) {
        for (var j = 0u; j < past; j++) {
            acc += hist[j * c + ch] * w[ch * k + j];
        }
        acc += cur_v * w[ch * k + past];
        out[ch] = (acc + cur_v) * inverseSqrt(f32(k + 1u));
    } else {
        let head_dim = c / n_heads;
        let h = ch / head_dim;
        for (var j = 0u; j < past; j++) {
            acc += hist[j * c + ch] * (w[ch * k + j] + dyn_taps[h * k + (k - 1u - j)]);
        }
        acc += cur_v * (w[ch * k + past] + dyn_taps[h * k]);
        out[ch] = (acc + cur_v) * inverseSqrt(f32(2u * k + 1u));
    }
    for (var j = 0u; j + 1u < past; j++) {
        hist[j * c + ch] = hist[(j + 1u) * c + ch];
    }
    if (past > 0u) {
        hist[(past - 1u) * c + ch] = cur_v;
    }
}
"#;

/// Per-head optional learned scale + L2 clipping + optional full-attention log-scale `tau` (read
/// from `step_params[1]` so uniforms remain constant across tokens for 100% bind-group caching).
/// Grid: `(n_heads, 1, 1)`. `p[0]`: n_heads, head_dim, has_scale, threshold bits; `p[1]`: use_tau.
pub(in crate::chain) const INKLING2_L2_CLIP_HEADS: &str = r#"
@group(0) @binding(0) var<storage, read> scale_w: array<f32>;
@group(0) @binding(1) var<storage, read> step_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    if (h >= p[0].x) { return; }
    let head_dim = p[0].y;
    let has_scale = p[0].z != 0u;
    let threshold = bitcast<f32>(p[0].w);
    let extra = select(1.0, bitcast<f32>(step_params[1]), p[1].x != 0u);
    let base = h * head_dim;

    var sq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        var v = x[base + d];
        if (has_scale) {
            v *= scale_w[base + d];
            x[base + d] = v;
        }
        sq += v * v;
    }
    part[li] = sq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let factor = min(1.0, threshold / (sqrt(part[0]) + 1e-6)) * extra;
    for (var d = li; d < head_dim; d += 128u) {
        x[base + d] *= factor;
    }
}
"#;

/// Store clipped `k` and per-head unweighted-RMSNormed `v` into ring-buffered `k_cache`/`v_cache`
/// at row `step_params[0] % kv_cap`.
/// Grid: `(n_kv_heads, 1, 1)`. `p[0]`: n_kv_heads, head_dim, kv_cap, rms_eps bits.
pub(in crate::chain) const INKLING2_V_NORM_KV_STORE: &str = r#"
@group(0) @binding(0) var<storage, read> k_in: array<f32>;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read> step_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_kv = p[0].x;
    let h = wg.x;
    if (h >= n_kv) { return; }
    let head_dim = p[0].y;
    let kv_cap = p[0].z;
    let eps = bitcast<f32>(p[0].w);
    let write_row = step_params[0] % kv_cap;
    let src_base = h * head_dim;
    let dst_base = write_row * (n_kv * head_dim) + src_base;

    var sq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        k_cache[dst_base + d] = k_in[src_base + d];
        let vv = v_in[src_base + d];
        sq += vv * vv;
    }
    part[li] = sq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(part[0] / f32(head_dim) + eps);
    for (var d = li; d < head_dim; d += 128u) {
        v_cache[dst_base + d] = v_in[src_base + d] * inv;
    }
}
"#;

/// Fast 8-warp subgroup GQA attention for `seq == 1`, `head_dim == 128`, `d_rel <= 32`, with
/// on-the-fly relative-position bias dot product and per-query attention sink.
/// `step_params[0]` holds `q_pos` (constant uniforms across all tokens!).
/// Grid: `(n_heads, 1, 1)`. `p[0]`: n_heads, n_kv_heads, head_dim, d_rel; `p[1]`: rel_extent, kv_cap, swa.
pub(in crate::chain) const INKLING2_ATTEND_STEP_SG: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> sinks: array<f32>;
@group(0) @binding(3) var<storage, read> rel_bias: array<f32>;
@group(0) @binding(4) var<storage, read> k_cache: array<f32>;
@group(0) @binding(5) var<storage, read> v_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> step_params: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wm: array<f32, 8>;
var<workgroup> ws: array<f32, 8>;
var<workgroup> wacc: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let n_heads = p[0].x;
    let h = wg.x;
    if (h >= n_heads) { return; }
    let n_kv = p[0].y;
    let d_rel = p[0].w;
    let rel_extent = p[1].x;
    let kv_cap = p[1].y;
    let swa = p[1].z;

    let group = n_heads / n_kv;
    let hk = h / group;
    let kv_dim = n_kv * 128u;
    let q_pos = step_params[0];
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);

    let q_off = h * 128u + lane * 4u;
    let q4 = vec4<f32>(q[q_off], q[q_off + 1u], q[q_off + 2u], q[q_off + 3u]);
    let r_val = select(0.0, r[h * d_rel + lane], lane < d_rel);
    let scale = inverseSqrt(128.0);

    var m = select(-1.0e30, sinks[h], warp == 0u);
    var s = select(0.0, 1.0, warp == 0u);
    var acc = vec4<f32>(0.0);

    for (var k_pos = kv_first + warp; k_pos < kv_end; k_pos += 8u) {
        let row = k_pos % kv_cap;
        let kv_off = row * kv_dim + hk * 128u + lane * 4u;
        let k4 = vec4<f32>(k_cache[kv_off], k_cache[kv_off + 1u], k_cache[kv_off + 2u], k_cache[kv_off + 3u]);
        let dot_qk = subgroupAdd(dot(q4, k4));
        let dist = q_pos - k_pos;
        var rb_term = 0.0;
        if (dist < rel_extent && lane < d_rel) {
            rb_term = r_val * rel_bias[dist * d_rel + lane];
        }
        let bias = subgroupAdd(rb_term);
        let score = dot_qk * scale + bias;
        let v4 = vec4<f32>(v_cache[kv_off], v_cache[kv_off + 1u], v_cache[kv_off + 2u], v_cache[kv_off + 3u]);
        if (s == 0.0) {
            m = score;
            s = 1.0;
            acc = v4;
        } else if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            acc = acc * rs + v4;
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            acc += v4 * w;
        }
    }

    if (lane == 0u) {
        wm[warp] = m;
        ws[warp] = s;
    }
    let wb = warp * 128u + lane * 4u;
    wacc[wb] = acc.x;
    wacc[wb + 1u] = acc.y;
    wacc[wb + 2u] = acc.z;
    wacc[wb + 3u] = acc.w;
    workgroupBarrier();

    if (warp == 0u) {
        var gm = wm[0];
        for (var w = 1u; w < 8u; w++) {
            if (ws[w] > 0.0) { gm = max(gm, wm[w]); }
        }
        var gs = 0.0;
        var gacc = vec4<f32>(0.0);
        for (var w = 0u; w < 8u; w++) {
            if (ws[w] > 0.0) {
                let f = exp(wm[w] - gm);
                gs += ws[w] * f;
                let ob = w * 128u + lane * 4u;
                gacc += vec4<f32>(wacc[ob], wacc[ob + 1u], wacc[ob + 2u], wacc[ob + 3u]) * f;
            }
        }
        let res = gacc * (1.0 / gs);
        out[q_off] = res.x;
        out[q_off + 1u] = res.y;
        out[q_off + 2u] = res.z;
        out[q_off + 3u] = res.w;
    }
}
"#;

/// Portable fallback GQA attention for any `head_dim <= 256` and devices without 32-lane subgroups.
/// Grid: `(n_heads, 1, 1)`. `p[0]`: n_heads, n_kv_heads, head_dim, d_rel; `p[1]`: rel_extent, kv_cap, swa.
pub(in crate::chain) const INKLING2_ATTEND_STEP: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> sinks: array<f32>;
@group(0) @binding(3) var<storage, read> rel_bias: array<f32>;
@group(0) @binding(4) var<storage, read> k_cache: array<f32>;
@group(0) @binding(5) var<storage, read> v_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> step_params: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 128>;
var<workgroup> acc_s: array<f32, 256>;
var<workgroup> cur_score: f32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_heads = p[0].x;
    let h = wg.x;
    if (h >= n_heads) { return; }
    let n_kv = p[0].y;
    let head_dim = p[0].z;
    let d_rel = p[0].w;
    let rel_extent = p[1].x;
    let kv_cap = p[1].y;
    let swa = p[1].z;

    let group = n_heads / n_kv;
    let hk = h / group;
    let kv_dim = n_kv * head_dim;
    let q_pos = step_params[0];
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);
    let scale = inverseSqrt(f32(head_dim));
    let q_base = h * head_dim;

    for (var d = li; d < head_dim; d += 128u) { acc_s[d] = 0.0; }
    var m = sinks[h];
    var s = 1.0;
    workgroupBarrier();

    for (var k_pos = kv_first; k_pos < kv_end; k_pos++) {
        let row = k_pos % kv_cap;
        let kv_base = row * kv_dim + hk * head_dim;
        var dq = 0.0;
        for (var d = li; d < head_dim; d += 128u) {
            dq += q[q_base + d] * k_cache[kv_base + d];
        }
        part[li] = dq;
        workgroupBarrier();
        for (var st = 64u; st > 0u; st /= 2u) {
            if (li < st) { part[li] += part[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) {
            var bias = 0.0;
            let dist = q_pos - k_pos;
            if (dist < rel_extent) {
                let rb = dist * d_rel;
                let rr = h * d_rel;
                for (var j = 0u; j < d_rel; j++) { bias += r[rr + j] * rel_bias[rb + j]; }
            }
            cur_score = part[0] * scale + bias;
        }
        workgroupBarrier();
        let score = cur_score;
        if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            for (var d = li; d < head_dim; d += 128u) {
                acc_s[d] = acc_s[d] * rs + v_cache[kv_base + d];
            }
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            for (var d = li; d < head_dim; d += 128u) {
                acc_s[d] += v_cache[kv_base + d] * w;
            }
        }
        workgroupBarrier();
    }

    let inv_s = 1.0 / s;
    for (var d = li; d < head_dim; d += 128u) {
        out[q_base + d] = acc_s[d] * inv_s;
    }
}
"#;

/// Block-boundary update (batched over `seq` tokens, grid `(seq, 1, 1)`):
/// * `block_slot != 0xffffffffu`: write `rms_norm_no_weight(inp_sa[t], 1e-6)` into `blocks[(block_slot * seq + t) * hidden..]`
///   and copy `attn_conv[t]` into `ffn_inp[t]` (unless `p[0].z != 0u`, which skips writing `ffn_inp`).
/// * `block_slot == 0xffffffffu`: `ffn_inp[t] = inp_sa[t] + attn_conv[t]`.
/// `p[0]`: hidden, block_slot, skip_ffn_inp (0 or 1), seq (0 treated as 1).
pub(in crate::chain) const INKLING2_BLOCK_PUSH: &str = r#"
@group(0) @binding(0) var<storage, read> inp_sa: array<f32>;
@group(0) @binding(1) var<storage, read> attn_conv: array<f32>;
@group(0) @binding(6) var<storage, read_write> blocks: array<f32>;
@group(0) @binding(7) var<storage, read_write> ffn_inp: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let hidden = p[0].x;
    let slot = p[0].y;
    let skip_ffn = p[0].z != 0u;
    let seq = max(1u, p[0].w);
    let t = wg.x;
    if (t >= seq) { return; }
    let t_off = t * hidden;
    if (slot == 0xffffffffu) {
        for (var i = li; i < hidden; i += 256u) {
            ffn_inp[t_off + i] = inp_sa[t_off + i] + attn_conv[t_off + i];
        }
        return;
    }
    var sq = 0.0;
    for (var i = li; i < hidden; i += 256u) {
        let v = inp_sa[t_off + i];
        sq += v * v;
    }
    part[li] = sq;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(part[0] / f32(hidden) + 1e-6);
    let b_off = (slot * seq + t) * hidden;
    for (var i = li; i < hidden; i += 256u) {
        blocks[b_off + i] = inp_sa[t_off + i] * inv;
        if (!skip_ffn) {
            ffn_inp[t_off + i] = attn_conv[t_off + i];
        }
    }
}
"#;

/// Subgroup version of [`INKLING2_BLOCK_PUSH`]: 1 `workgroupBarrier()` instead of 9.
pub(in crate::chain) const INKLING2_BLOCK_PUSH_SG: &str = r#"
@group(0) @binding(0) var<storage, read> inp_sa: array<f32>;
@group(0) @binding(1) var<storage, read> attn_conv: array<f32>;
@group(0) @binding(6) var<storage, read_write> blocks: array<f32>;
@group(0) @binding(7) var<storage, read_write> ffn_inp: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wpart: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let hidden = p[0].x;
    let slot = p[0].y;
    let skip_ffn = p[0].z != 0u;
    let seq = max(1u, p[0].w);
    let t = wg.x;
    if (t >= seq) { return; }
    let t_off = t * hidden;
    if (slot == 0xffffffffu) {
        for (var i = li; i < hidden; i += 256u) {
            ffn_inp[t_off + i] = inp_sa[t_off + i] + attn_conv[t_off + i];
        }
        return;
    }
    var sq = 0.0;
    for (var i = li; i < hidden; i += 256u) {
        let v = inp_sa[t_off + i];
        sq += v * v;
    }
    sq = subgroupAdd(sq);
    if (lane == 0u) { wpart[warp] = sq; }
    workgroupBarrier();
    let total = ((wpart[0] + wpart[1]) + (wpart[2] + wpart[3])) + ((wpart[4] + wpart[5]) + (wpart[6] + wpart[7]));
    let inv = inverseSqrt(total / f32(hidden) + 1e-6);
    let b_off = (slot * seq + t) * hidden;
    for (var i = li; i < hidden; i += 256u) {
        blocks[b_off + i] = inp_sa[t_off + i] * inv;
        if (!skip_ffn) {
            ffn_inp[t_off + i] = attn_conv[t_off + i];
        }
    }
}
"#;

/// Final logit softcapping in place: `logits[i] = cap * tanh(clamp(logits[i] / cap, -20.0, 20.0))`.
/// `p[0]`: len, cap bits.
pub(in crate::chain) const INKLING2_SOFTCAP: &str = r#"
@group(0) @binding(6) var<storage, read_write> logits: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    let cap = bitcast<f32>(p[0].y);
    logits[i] = cap * tanh(clamp(logits[i] / cap, -20.0, 20.0));
}
"#;

/// Fused cross-pass controller up-projection (`ctrl_gain` + `ctrl_forget`, stacked F16 `[2 * hidden, gate_rank]`)
/// and tanh-gated blend across batched `(n_streams, seq)`:
/// `out[(s * seq + t) * hidden + c] = (1 - tanh(forget)) * cur[cur_off + s * cur_stride + t * hidden + c] + tanh(gain) * msg[msg_off + (s * seq + t) * hidden + c]`.
/// Grid: `(ceil(hidden / 256), n_streams, seq)`.
/// `p[0]`: hidden, gate_rank (even, <= 256), cur_off, msg_off; `p[1]`: cur_stride, seq (0 treated as 1), 0, 0.
pub(in crate::chain) const INKLING2_CTRL_UP_BLEND: &str = r#"
@group(0) @binding(0) var<storage, read> w_gf: array<u32>;
@group(0) @binding(1) var<storage, read> mid: array<f32>;
@group(0) @binding(2) var<storage, read> cur: array<f32>;
@group(0) @binding(3) var<storage, read> msg: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> smid: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let hidden = p[0].x;
    let rank = p[0].y;
    let s = wg.y;
    let seq = max(1u, p[1].y);
    let t = wg.z;
    if (t >= seq) { return; }
    let st_idx = s * seq + t;
    if (li < rank) {
        smid[li] = mid[st_idx * rank + li];
    }
    workgroupBarrier();
    let c = wg.x * 256u + li;
    if (c >= hidden) { return; }
    let rw = rank / 2u;
    let g_base = (s * 2u * hidden + c) * rw;
    let f_base = (s * 2u * hidden + hidden + c) * rw;
    var g = 0.0;
    var f = 0.0;
    for (var k = 0u; k < rw; k++) {
        let m0 = smid[2u * k];
        let m1 = smid[2u * k + 1u];
        let wg2 = unpack2x16float(w_gf[g_base + k]);
        let wf2 = unpack2x16float(w_gf[f_base + k]);
        g += wg2.x * m0 + wg2.y * m1;
        f += wf2.x * m0 + wf2.y * m1;
    }
    let gt = tanh(clamp(g, -20.0, 20.0));
    let ft = tanh(clamp(f, -20.0, 20.0));
    out[st_idx * hidden + c] = (1.0 - ft) * cur[p[0].z + s * p[1].x + t * hidden + c] + gt * msg[p[0].w + st_idx * hidden + c];
}
"#;

/// Fused K/V short convolution (static or dynamic) + K L2 clip + V unweighted RMSNorm + ring-buffer
/// KV store into packed `kv_cache` (`[2, kv_cap, kv_dim]`). Replaces 4 dispatches with 1.
/// Grid: `(n_kv_heads, 1, 1)`.
/// `p[0]`: n_kv_heads, head_dim (<= 256), kernel, is_dynamic (0 or 1);
/// `p[1]`: kv_cap, qk_clip bits, rms_eps bits, 0.
pub(in crate::chain) const INKLING2_KV_CONV_CLIP_STORE: &str = r#"
@group(0) @binding(0) var<storage, read> k_raw: array<f32>;
@group(0) @binding(1) var<storage, read> v_raw: array<f32>;
@group(0) @binding(2) var<storage, read> kv_w: array<f32>;
@group(0) @binding(3) var<storage, read> dyn_k: array<f32>;
@group(0) @binding(4) var<storage, read> dyn_v: array<f32>;
@group(0) @binding(5) var<storage, read> step_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> kv_hist: array<f32>;
@group(0) @binding(7) var<storage, read_write> kv_cache: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> pk: array<f32, 128>;
var<workgroup> pv: array<f32, 128>;
var<workgroup> tk: array<f32, 256>;
var<workgroup> tv: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_kv = p[0].x;
    let h = wg.x;
    if (h >= n_kv) { return; }
    let head_dim = p[0].y;
    let k = p[0].z;
    let past = k - 1u;
    let dyn_mode = p[0].w;
    let is_dyn = dyn_mode != 0u;
    let packed_dyn = dyn_mode == 2u;
    let kv_cap = p[1].x;
    let qk_clip = bitcast<f32>(p[1].y);
    let rms_eps = bitcast<f32>(p[1].z);

    let kv_dim = n_kv * head_dim;
    let v_w_off = kv_dim * k;
    let v_h_off = past * kv_dim;
    let conv_scale = select(inverseSqrt(f32(k + 1u)), inverseSqrt(f32(2u * k + 1u)), is_dyn);

    var ksq = 0.0;
    var vsq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        let ch = h * head_dim + d;
        let kc = k_raw[ch];
        let vc = v_raw[ch];
        var ak = 0.0;
        var av = 0.0;
        if (!is_dyn) {
            for (var j = 0u; j < past; j++) {
                ak += kv_hist[j * kv_dim + ch] * kv_w[ch * k + j];
                av += kv_hist[v_h_off + j * kv_dim + ch] * kv_w[v_w_off + ch * k + j];
            }
            ak += kc * kv_w[ch * k + past];
            av += vc * kv_w[v_w_off + ch * k + past];
        } else if (packed_dyn) {
            for (var j = 0u; j < past; j++) {
                let lag = k - 1u - j;
                ak += kv_hist[j * kv_dim + ch] * (kv_w[ch * k + j] + k_raw[kv_dim + h * k + lag]);
                av += kv_hist[v_h_off + j * kv_dim + ch] * (kv_w[v_w_off + ch * k + j] + v_raw[kv_dim + h * k + lag]);
            }
            ak += kc * (kv_w[ch * k + past] + k_raw[kv_dim + h * k]);
            av += vc * (kv_w[v_w_off + ch * k + past] + v_raw[kv_dim + h * k]);
        } else {
            for (var j = 0u; j < past; j++) {
                let lag = k - 1u - j;
                ak += kv_hist[j * kv_dim + ch] * (kv_w[ch * k + j] + dyn_k[h * k + lag]);
                av += kv_hist[v_h_off + j * kv_dim + ch] * (kv_w[v_w_off + ch * k + j] + dyn_v[h * k + lag]);
            }
            ak += kc * (kv_w[ch * k + past] + dyn_k[h * k]);
            av += vc * (kv_w[v_w_off + ch * k + past] + dyn_v[h * k]);
        }
        let ko = (ak + kc) * conv_scale;
        let vo = (av + vc) * conv_scale;
        tk[d] = ko;
        tv[d] = vo;
        ksq += ko * ko;
        vsq += vo * vo;

        for (var j = 0u; j + 1u < past; j++) {
            kv_hist[j * kv_dim + ch] = kv_hist[(j + 1u) * kv_dim + ch];
            kv_hist[v_h_off + j * kv_dim + ch] = kv_hist[v_h_off + (j + 1u) * kv_dim + ch];
        }
        if (past > 0u) {
            kv_hist[(past - 1u) * kv_dim + ch] = kc;
            kv_hist[v_h_off + (past - 1u) * kv_dim + ch] = vc;
        }
    }
    pk[li] = ksq;
    pv[li] = vsq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) {
            pk[li] += pk[li + st];
            pv[li] += pv[li + st];
        }
        workgroupBarrier();
    }
    let k_fac = min(1.0, qk_clip / (sqrt(pk[0]) + 1e-6));
    let v_inv = inverseSqrt(pv[0] / f32(head_dim) + rms_eps);
    let write_row = step_params[0] % kv_cap;
    let dst_k = write_row * kv_dim + h * head_dim;
    let dst_v = kv_cap * kv_dim + dst_k;
    for (var d = li; d < head_dim; d += 128u) {
        kv_cache[dst_k + d] = tk[d] * k_fac;
        kv_cache[dst_v + d] = tv[d] * v_inv;
    }
}
"#;

/// Subgroup version of [`INKLING2_KV_CONV_CLIP_STORE`] (4 warps = 128 threads): 1 `workgroupBarrier()`
/// instead of 8.
pub(in crate::chain) const INKLING2_KV_CONV_CLIP_STORE_SG: &str = r#"
@group(0) @binding(0) var<storage, read> k_raw: array<f32>;
@group(0) @binding(1) var<storage, read> v_raw: array<f32>;
@group(0) @binding(2) var<storage, read> kv_w: array<f32>;
@group(0) @binding(3) var<storage, read> dyn_k: array<f32>;
@group(0) @binding(4) var<storage, read> dyn_v: array<f32>;
@group(0) @binding(5) var<storage, read> step_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> kv_hist: array<f32>;
@group(0) @binding(7) var<storage, read_write> kv_cache: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wpk: array<f32, 4>;
var<workgroup> wpv: array<f32, 4>;
var<workgroup> tk: array<f32, 256>;
var<workgroup> tv: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let n_kv = p[0].x;
    let h = wg.x;
    if (h >= n_kv) { return; }
    let head_dim = p[0].y;
    let k = p[0].z;
    let past = k - 1u;
    let dyn_mode = p[0].w;
    let is_dyn = dyn_mode != 0u;
    let packed_dyn = dyn_mode == 2u;
    let kv_cap = p[1].x;
    let qk_clip = bitcast<f32>(p[1].y);
    let rms_eps = bitcast<f32>(p[1].z);

    let kv_dim = n_kv * head_dim;
    let v_w_off = kv_dim * k;
    let v_h_off = past * kv_dim;
    let conv_scale = select(inverseSqrt(f32(k + 1u)), inverseSqrt(f32(2u * k + 1u)), is_dyn);

    var ksq = 0.0;
    var vsq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        let ch = h * head_dim + d;
        let kc = k_raw[ch];
        let vc = v_raw[ch];
        var ak = 0.0;
        var av = 0.0;
        if (!is_dyn) {
            for (var j = 0u; j < past; j++) {
                ak += kv_hist[j * kv_dim + ch] * kv_w[ch * k + j];
                av += kv_hist[v_h_off + j * kv_dim + ch] * kv_w[v_w_off + ch * k + j];
            }
            ak += kc * kv_w[ch * k + past];
            av += vc * kv_w[v_w_off + ch * k + past];
        } else if (packed_dyn) {
            for (var j = 0u; j < past; j++) {
                let lag = k - 1u - j;
                ak += kv_hist[j * kv_dim + ch] * (kv_w[ch * k + j] + k_raw[kv_dim + h * k + lag]);
                av += kv_hist[v_h_off + j * kv_dim + ch] * (kv_w[v_w_off + ch * k + j] + v_raw[kv_dim + h * k + lag]);
            }
            ak += kc * (kv_w[ch * k + past] + k_raw[kv_dim + h * k]);
            av += vc * (kv_w[v_w_off + ch * k + past] + v_raw[kv_dim + h * k]);
        } else {
            for (var j = 0u; j < past; j++) {
                let lag = k - 1u - j;
                ak += kv_hist[j * kv_dim + ch] * (kv_w[ch * k + j] + dyn_k[h * k + lag]);
                av += kv_hist[v_h_off + j * kv_dim + ch] * (kv_w[v_w_off + ch * k + j] + dyn_v[h * k + lag]);
            }
            ak += kc * (kv_w[ch * k + past] + dyn_k[h * k]);
            av += vc * (kv_w[v_w_off + ch * k + past] + dyn_v[h * k]);
        }
        let ko = (ak + kc) * conv_scale;
        let vo = (av + vc) * conv_scale;
        tk[d] = ko;
        tv[d] = vo;
        ksq += ko * ko;
        vsq += vo * vo;

        for (var j = 0u; j + 1u < past; j++) {
            kv_hist[j * kv_dim + ch] = kv_hist[(j + 1u) * kv_dim + ch];
            kv_hist[v_h_off + j * kv_dim + ch] = kv_hist[v_h_off + (j + 1u) * kv_dim + ch];
        }
        if (past > 0u) {
            kv_hist[(past - 1u) * kv_dim + ch] = kc;
            kv_hist[v_h_off + (past - 1u) * kv_dim + ch] = vc;
        }
    }
    ksq = subgroupAdd(ksq);
    vsq = subgroupAdd(vsq);
    if (lane == 0u) {
        wpk[warp] = ksq;
        wpv[warp] = vsq;
    }
    workgroupBarrier();
    let k_tot = (wpk[0] + wpk[1]) + (wpk[2] + wpk[3]);
    let v_tot = (wpv[0] + wpv[1]) + (wpv[2] + wpv[3]);
    let k_fac = min(1.0, qk_clip / (sqrt(k_tot) + 1e-6));
    let v_inv = inverseSqrt(v_tot / f32(head_dim) + rms_eps);
    let write_row = step_params[0] % kv_cap;
    let dst_k = write_row * kv_dim + h * head_dim;
    let dst_v = kv_cap * kv_dim + dst_k;
    for (var d = li; d < head_dim; d += 128u) {
        kv_cache[dst_k + d] = tk[d] * k_fac;
        kv_cache[dst_v + d] = tv[d] * v_inv;
    }
}
"#;

/// Fully fused 8-warp subgroup GQA attention for `seq == 1`, `head_dim == 128`, `d_rel <= 32`:
/// fuses Q clip + tau, R clip + tau, on-the-fly relative-position bias, per-query sinks,
/// packed `kv_cache`, AND `sigmoid(gate)` output gating into ONE dispatch (replaces 4 dispatches!).
/// `attn_consts` holds `[rel_bias (rel_extent * d_rel), q_clip (n_heads * 128), r_clip (n_heads * d_rel)]`.
/// Grid: `(n_heads, 1, 1)`.
/// `p[0]`: n_heads, n_kv_heads, d_rel, rel_extent;
/// `p[1]`: kv_cap, swa (0 = full attn, >0 = SWA window), qk_clip bits, rel_clip bits.
pub(in crate::chain) const INKLING2_ATTEND_FUSED_SG: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> sinks: array<f32>;
@group(0) @binding(3) var<storage, read> attn_consts: array<f32>;
@group(0) @binding(4) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(5) var<storage, read> gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> step_params: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wm: array<f32, 8>;
var<workgroup> ws: array<f32, 8>;
var<workgroup> wacc: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let n_heads = p[0].x;
    let h = wg.x;
    if (h >= n_heads) { return; }
    let n_kv = p[0].y;
    let d_rel = p[0].z;
    let rel_extent = p[0].w;
    let kv_cap = p[1].x;
    let swa = p[1].y;
    let qk_clip = bitcast<f32>(p[1].z);
    let rel_clip = bitcast<f32>(p[1].w);

    let group = n_heads / n_kv;
    let hk = h / group;
    let kv_dim = n_kv * 128u;
    let v_cache_off = kv_cap * kv_dim;
    let q_clip_off = rel_extent * d_rel;
    let r_clip_off = q_clip_off + n_heads * 128u;

    let q_pos = step_params[0];
    let tau = select(1.0, bitcast<f32>(step_params[1]), swa == 0u);
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);

    // Per-head learned scale + L2 clip + full-attn tau on Q and R in registers (0 extra dispatches!).
    let q_off = h * 128u + lane * 4u;
    let qc_off = q_clip_off + q_off;
    var q4 = vec4<f32>(q[q_off], q[q_off + 1u], q[q_off + 2u], q[q_off + 3u])
        * vec4<f32>(attn_consts[qc_off], attn_consts[qc_off + 1u], attn_consts[qc_off + 2u], attn_consts[qc_off + 3u]);
    let q_sq = subgroupAdd(dot(q4, q4));
    q4 *= min(1.0, qk_clip / (sqrt(q_sq) + 1e-6)) * tau;

    var r_val = 0.0;
    if (lane < d_rel) {
        let ro = h * d_rel + lane;
        r_val = r[ro] * attn_consts[r_clip_off + ro];
    }
    let r_sq = subgroupAdd(r_val * r_val);
    r_val *= min(1.0, rel_clip / (sqrt(r_sq) + 1e-6)) * tau;

    let scale = inverseSqrt(128.0);
    var m = select(-1.0e30, sinks[h], warp == 0u);
    var s = select(0.0, 1.0, warp == 0u);
    var acc = vec4<f32>(0.0);

    for (var k_pos = kv_first + warp; k_pos < kv_end; k_pos += 8u) {
        let row = k_pos % kv_cap;
        let kv_off = row * kv_dim + hk * 128u + lane * 4u;
        let k4 = vec4<f32>(kv_cache[kv_off], kv_cache[kv_off + 1u], kv_cache[kv_off + 2u], kv_cache[kv_off + 3u]);
        let dot_qk = subgroupAdd(dot(q4, k4));
        let dist = q_pos - k_pos;
        var rb_term = 0.0;
        if (dist < rel_extent && lane < d_rel) {
            rb_term = r_val * attn_consts[dist * d_rel + lane];
        }
        let bias = subgroupAdd(rb_term);
        let score = dot_qk * scale + bias;
        let vo = v_cache_off + kv_off;
        let v4 = vec4<f32>(kv_cache[vo], kv_cache[vo + 1u], kv_cache[vo + 2u], kv_cache[vo + 3u]);
        if (s == 0.0) {
            m = score;
            s = 1.0;
            acc = v4;
        } else if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            acc = acc * rs + v4;
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            acc += v4 * w;
        }
    }

    if (lane == 0u) {
        wm[warp] = m;
        ws[warp] = s;
    }
    let wb = warp * 128u + lane * 4u;
    wacc[wb] = acc.x;
    wacc[wb + 1u] = acc.y;
    wacc[wb + 2u] = acc.z;
    wacc[wb + 3u] = acc.w;
    workgroupBarrier();

    if (warp == 0u) {
        var gm = wm[0];
        for (var w = 1u; w < 8u; w++) {
            if (ws[w] > 0.0) { gm = max(gm, wm[w]); }
        }
        var gs = 0.0;
        var gacc = vec4<f32>(0.0);
        for (var w = 0u; w < 8u; w++) {
            if (ws[w] > 0.0) {
                let f = exp(wm[w] - gm);
                gs += ws[w] * f;
                let ob = w * 128u + lane * 4u;
                gacc += vec4<f32>(wacc[ob], wacc[ob + 1u], wacc[ob + 2u], wacc[ob + 3u]) * f;
            }
        }
        let g4 = vec4<f32>(gate[q_off], gate[q_off + 1u], gate[q_off + 2u], gate[q_off + 3u]);
        let sig = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-g4));
        let res = (gacc * (1.0 / gs)) * sig;
        out[q_off] = res.x;
        out[q_off + 1u] = res.y;
        out[q_off + 2u] = res.z;
        out[q_off + 3u] = res.w;
    }
}
"#;

/// Portable fallback fused GQA attention for any `head_dim <= 256` and devices without 32-lane
/// subgroups: fuses Q/R clip+tau, on-the-fly relative-position bias, per-query sinks, packed
/// `kv_cache`, and `sigmoid(gate)` output gating into ONE dispatch.
/// Grid: `(n_heads, 1, 1)`.
/// `p[0]`: n_heads, n_kv_heads, head_dim, d_rel;
/// `p[1]`: rel_extent, kv_cap, swa, qk_clip_bits (and rel_clip passed via `attn_consts`? No: keep `p[1]` clean!).
/// Wait: store `qk_clip` and `rel_clip` at the end of `attn_consts`? Better: `p[1].w` = pack2x16float(qk_clip, rel_clip)!
pub(in crate::chain) const INKLING2_ATTEND_FUSED_PORTABLE: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> sinks: array<f32>;
@group(0) @binding(3) var<storage, read> attn_consts: array<f32>;
@group(0) @binding(4) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(5) var<storage, read> gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> step_params: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 128>;
var<workgroup> q_loc: array<f32, 256>;
var<workgroup> r_loc: array<f32, 64>;
var<workgroup> acc_s: array<f32, 256>;
var<workgroup> cur_score: f32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let packed = (p[0].x & 0x80000000u) != 0u;
    let n_heads = p[0].x & 0x7fffffffu;
    let h = wg.x;
    if (h >= n_heads) { return; }
    let n_kv = p[0].y;
    let head_dim = p[0].z;
    let d_rel = p[0].w;
    let rel_extent = p[1].x;
    let kv_cap = p[1].y;
    let swa = p[1].z;
    let clips = unpack2x16float(p[1].w);
    let qk_clip = clips.x;
    let rel_clip = clips.y;

    let group = n_heads / n_kv;
    let hk = h / group;
    let kv_dim = n_kv * head_dim;
    let v_cache_off = kv_cap * kv_dim;
    let q_clip_off = rel_extent * d_rel;
    let r_clip_off = q_clip_off + n_heads * head_dim;
    let r_buf_off = select(0u, n_heads * head_dim, packed);
    let sink_buf_off = select(0u, n_heads * head_dim + n_heads * d_rel, packed);
    let gate_buf_off = select(0u, n_heads * head_dim + n_heads * d_rel + n_heads, packed);
    let q_pos = step_params[0];
    let tau = select(1.0, bitcast<f32>(step_params[1]), swa == 0u);
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);
    let scale = inverseSqrt(f32(head_dim));
    let q_base = h * head_dim;

    var qsq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        let v = q[q_base + d] * attn_consts[q_clip_off + q_base + d];
        q_loc[d] = v;
        qsq += v * v;
        acc_s[d] = 0.0;
    }
    part[li] = qsq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let q_fac = min(1.0, qk_clip / (sqrt(part[0]) + 1e-6)) * tau;
    for (var d = li; d < head_dim; d += 128u) { q_loc[d] *= q_fac; }

    var rsq = 0.0;
    for (var j = li; j < d_rel; j += 128u) {
        let ro = h * d_rel + j;
        let v = r[r_buf_off + ro] * attn_consts[r_clip_off + ro];
        r_loc[j] = v;
        rsq += v * v;
    }
    part[li] = rsq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let r_fac = min(1.0, rel_clip / (sqrt(part[0]) + 1e-6)) * tau;
    for (var j = li; j < d_rel; j += 128u) { r_loc[j] *= r_fac; }
    workgroupBarrier();

    var m = sinks[sink_buf_off + h];
    var s = 1.0;
    for (var k_pos = kv_first; k_pos < kv_end; k_pos++) {
        let row = k_pos % kv_cap;
        let kv_base = row * kv_dim + hk * head_dim;
        var dq = 0.0;
        for (var d = li; d < head_dim; d += 128u) {
            dq += q_loc[d] * kv_cache[kv_base + d];
        }
        part[li] = dq;
        workgroupBarrier();
        for (var st = 64u; st > 0u; st /= 2u) {
            if (li < st) { part[li] += part[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) {
            var bias = 0.0;
            let dist = q_pos - k_pos;
            if (dist < rel_extent) {
                let rb = dist * d_rel;
                for (var j = 0u; j < d_rel; j++) { bias += r_loc[j] * attn_consts[rb + j]; }
            }
            cur_score = part[0] * scale + bias;
        }
        workgroupBarrier();
        let score = cur_score;
        let vb = v_cache_off + kv_base;
        if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            for (var d = li; d < head_dim; d += 128u) {
                acc_s[d] = acc_s[d] * rs + kv_cache[vb + d];
            }
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            for (var d = li; d < head_dim; d += 128u) {
                acc_s[d] += kv_cache[vb + d] * w;
            }
        }
        workgroupBarrier();
    }

    let inv_s = 1.0 / s;
    for (var d = li; d < head_dim; d += 128u) {
        let g = gate[gate_buf_off + q_base + d];
        out[q_base + d] = (acc_s[d] * inv_s) / (1.0 + exp(-g));
    }
}
"#;

/// Fully fused 8-warp subgroup GQA attention on packed `qrsg` (`[q_dim, r_dim, n_heads, q_dim]` in
/// binding 0) with read-only `step_params` (binding 3, zero false write hazard) AND optional inline
/// Q8_0 quantization of `out` into `out_q8` (binding 7, bit 31 of `p[0].x`), eliminating
/// `chain-q8-quantize-lanes` before `wo` on all 48 layers.
/// Grid: `(n_heads, 1, 1)`.
/// `p[0]`: `n_heads | (write_q8 << 31)`, `n_kv_heads`, `d_rel`, `rel_extent`;
/// `p[1]`: `kv_cap`, `swa`, `qk_clip_bits`, `rel_clip_bits`.
pub(in crate::chain) const INKLING2_ATTEND_FUSED_SG_PACKED: &str = r#"
@group(0) @binding(0) var<storage, read> qrsg: array<f32>;
@group(0) @binding(1) var<storage, read> attn_consts: array<f32>;
@group(0) @binding(2) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(3) var<storage, read> step_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_q8: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wm: array<f32, 8>;
var<workgroup> ws: array<f32, 8>;
var<workgroup> wacc: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let write_q8 = (p[0].x & 0x80000000u) != 0u;
    let n_heads = p[0].x & 0x7fffffffu;
    let h = wg.x;
    if (h >= n_heads) { return; }
    let n_kv = p[0].y;
    let d_rel = p[0].z;
    let rel_extent = p[0].w;
    let kv_cap = p[1].x;
    let swa = p[1].y;
    let qk_clip = bitcast<f32>(p[1].z);
    let rel_clip = bitcast<f32>(p[1].w);

    let group = n_heads / n_kv;
    let hk = h / group;
    let q_dim = n_heads * 128u;
    let r_dim = n_heads * d_rel;
    let r_buf_off = q_dim;
    let sink_buf_off = q_dim + r_dim;
    let gate_buf_off = sink_buf_off + n_heads;
    let kv_dim = n_kv * 128u;
    let v_cache_off = kv_cap * kv_dim;
    let q_clip_off = rel_extent * d_rel;
    let r_clip_off = q_clip_off + q_dim;

    let q_pos = step_params[0];
    let tau = select(1.0, bitcast<f32>(step_params[1]), swa == 0u);
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);

    let q_off = h * 128u + lane * 4u;
    let qc_off = q_clip_off + q_off;
    var q4 = vec4<f32>(qrsg[q_off], qrsg[q_off + 1u], qrsg[q_off + 2u], qrsg[q_off + 3u])
        * vec4<f32>(attn_consts[qc_off], attn_consts[qc_off + 1u], attn_consts[qc_off + 2u], attn_consts[qc_off + 3u]);
    let q_sq = subgroupAdd(dot(q4, q4));
    q4 *= min(1.0, qk_clip / (sqrt(q_sq) + 1e-6)) * tau;

    var r_val = 0.0;
    if (lane < d_rel) {
        let ro = h * d_rel + lane;
        r_val = qrsg[r_buf_off + ro] * attn_consts[r_clip_off + ro];
    }
    let r_sq = subgroupAdd(r_val * r_val);
    r_val *= min(1.0, rel_clip / (sqrt(r_sq) + 1e-6)) * tau;

    let scale = inverseSqrt(128.0);
    var m = select(-1.0e30, qrsg[sink_buf_off + h], warp == 0u);
    var s = select(0.0, 1.0, warp == 0u);
    var acc = vec4<f32>(0.0);

    for (var k_pos = kv_first + warp; k_pos < kv_end; k_pos += 8u) {
        let row = k_pos % kv_cap;
        let kv_off = row * kv_dim + hk * 128u + lane * 4u;
        let k4 = vec4<f32>(kv_cache[kv_off], kv_cache[kv_off + 1u], kv_cache[kv_off + 2u], kv_cache[kv_off + 3u]);
        let dot_qk = subgroupAdd(dot(q4, k4));
        let dist = q_pos - k_pos;
        var rb_term = 0.0;
        if (dist < rel_extent && lane < d_rel) {
            rb_term = r_val * attn_consts[dist * d_rel + lane];
        }
        let bias = subgroupAdd(rb_term);
        let score = dot_qk * scale + bias;
        let vo = v_cache_off + kv_off;
        let v4 = vec4<f32>(kv_cache[vo], kv_cache[vo + 1u], kv_cache[vo + 2u], kv_cache[vo + 3u]);
        if (s == 0.0) {
            m = score;
            s = 1.0;
            acc = v4;
        } else if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            acc = acc * rs + v4;
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            acc += v4 * w;
        }
    }

    if (lane == 0u) {
        wm[warp] = m;
        ws[warp] = s;
    }
    let wb = warp * 128u + lane * 4u;
    wacc[wb] = acc.x;
    wacc[wb + 1u] = acc.y;
    wacc[wb + 2u] = acc.z;
    wacc[wb + 3u] = acc.w;
    workgroupBarrier();

    if (warp == 0u) {
        var gm = wm[0];
        for (var w = 1u; w < 8u; w++) {
            if (ws[w] > 0.0) { gm = max(gm, wm[w]); }
        }
        var gs = 0.0;
        var gacc = vec4<f32>(0.0);
        for (var w = 0u; w < 8u; w++) {
            if (ws[w] > 0.0) {
                let f = exp(wm[w] - gm);
                gs += ws[w] * f;
                let ob = w * 128u + lane * 4u;
                gacc += vec4<f32>(wacc[ob], wacc[ob + 1u], wacc[ob + 2u], wacc[ob + 3u]) * f;
            }
        }
        let go = gate_buf_off + q_off;
        let g4 = vec4<f32>(qrsg[go], qrsg[go + 1u], qrsg[go + 2u], qrsg[go + 3u]);
        let sig = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-g4));
        let res = (gacc * (1.0 / gs)) * sig;
        out[q_off] = res.x;
        out[q_off + 1u] = res.y;
        out[q_off + 2u] = res.z;
        out[q_off + 3u] = res.w;
        if (write_q8) {
            let b = h * 4u + (lane / 8u);
            let part = lane % 8u;
            let xs_at = n_heads * 8u;
            var amax = max(max(abs(res.x), abs(res.y)), max(abs(res.z), abs(res.w)));
            amax = max(amax, subgroupShuffleXor(amax, 1u));
            amax = max(amax, subgroupShuffleXor(amax, 2u));
            amax = max(amax, subgroupShuffleXor(amax, 4u));
            let d = amax / 127.0;
            let inv = select(0.0, 1.0 / d, d > 0.0);
            let qv = clamp(vec4<i32>(round(res * inv)), vec4<i32>(-127), vec4<i32>(127));
            let qb = bitcast<vec4<u32>>(qv) & vec4<u32>(0xffu);
            var hs = qv.x + qv.y + qv.z + qv.w;
            hs += subgroupShuffleXor(hs, 1u);
            hs += subgroupShuffleXor(hs, 2u);
            let other = subgroupShuffleXor(hs, 4u);
            out_q8[b * 8u + part] = qb.x | (qb.y << 8u) | (qb.z << 16u) | (qb.w << 24u);
            if (part == 0u) {
                let at = (xs_at + b) * 4u;
                out_q8[at] = bitcast<u32>(d);
                out_q8[at + 1u] = bitcast<u32>(d * f32(hs));
                out_q8[at + 2u] = bitcast<u32>(d * f32(other));
                out_q8[at + 3u] = 0u;
            }
        }
    }
}
"#;

/// Fused cross-pass controller norm-concat + F16 down-projection across batched `(n_streams, seq)`:
/// `mid[(s * seq + t) * rank + r] = dot(w_down[s, r, 0..H], rms_norm(cur_{s,t})) + dot(w_down[s, r, H..2H], rms_norm(msg_{s,t}))`.
/// Grid: `(gate_rank, n_streams, seq)`.
/// `p[0]`: hidden (even), gate_rank, cur_off, msg_off; `p[1]`: cur_stride, seq (0 treated as 1), 0, 0.
pub(in crate::chain) const INKLING2_CTRL_CAT_DOWN: &str = r#"
@group(0) @binding(0) var<storage, read> w_down: array<u32>;
@group(0) @binding(1) var<storage, read> cur: array<f32>;
@group(0) @binding(2) var<storage, read> msg: array<f32>;
@group(0) @binding(6) var<storage, read_write> mid: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part_c: array<f32, 256>;
var<workgroup> part_m: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let hidden = p[0].x;
    let rank = p[0].y;
    let r = wg.x;
    let seq = max(1u, p[1].y);
    let t = wg.z;
    if (r >= rank || t >= seq) { return; }
    let s = wg.y;
    let st_idx = s * seq + t;
    let cur_off = p[0].z + s * p[1].x + t * hidden;
    let msg_off = p[0].w + st_idx * hidden;
    let half_h = hidden / 2u;

    var sq_c = 0.0;
    var sq_m = 0.0;
    var acc_c = 0.0;
    var acc_m = 0.0;
    let row_base = (s * rank + r) * hidden;
    for (var k = li; k < half_h; k += 256u) {
        let c0 = cur[cur_off + 2u * k];
        let c1 = cur[cur_off + 2u * k + 1u];
        let m0 = msg[msg_off + 2u * k];
        let m1 = msg[msg_off + 2u * k + 1u];
        sq_c += c0 * c0 + c1 * c1;
        sq_m += m0 * m0 + m1 * m1;
        let wc = unpack2x16float(w_down[row_base + k]);
        let wm = unpack2x16float(w_down[row_base + half_h + k]);
        acc_c += wc.x * c0 + wc.y * c1;
        acc_m += wm.x * m0 + wm.y * m1;
    }
    part_c[li] = sq_c;
    part_m[li] = sq_m;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) {
            part_c[li] += part_c[li + st];
            part_m[li] += part_m[li + st];
        }
        workgroupBarrier();
    }
    let inv_c = inverseSqrt(part_c[0] / f32(hidden) + 1e-6);
    let inv_m = inverseSqrt(part_m[0] / f32(hidden) + 1e-6);
    workgroupBarrier();
    part_c[li] = acc_c * inv_c + acc_m * inv_m;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part_c[li] += part_c[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        mid[st_idx * rank + r] = part_c[0];
    }
}
"#;

/// Subgroup version of [`INKLING2_CTRL_CAT_DOWN`]: reduces all 4 sums (`sq_c`, `sq_m`, `acc_c`,
/// `acc_m`) via `subgroupAdd` and combines the 8 warps in a SINGLE `workgroupBarrier()` (down from 17).
pub(in crate::chain) const INKLING2_CTRL_CAT_DOWN_SG: &str = r#"
@group(0) @binding(0) var<storage, read> w_down: array<u32>;
@group(0) @binding(1) var<storage, read> cur: array<f32>;
@group(0) @binding(2) var<storage, read> msg: array<f32>;
@group(0) @binding(6) var<storage, read_write> mid: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> ws_c: array<f32, 8>;
var<workgroup> ws_m: array<f32, 8>;
var<workgroup> wa_c: array<f32, 8>;
var<workgroup> wa_m: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let hidden = p[0].x;
    let rank = p[0].y;
    let r = wg.x;
    let seq = max(1u, p[1].y);
    let t = wg.z;
    if (r >= rank || t >= seq) { return; }
    let s = wg.y;
    let st_idx = s * seq + t;
    let cur_off = p[0].z + s * p[1].x + t * hidden;
    let msg_off = p[0].w + st_idx * hidden;
    let half_h = hidden / 2u;

    var sq_c = 0.0;
    var sq_m = 0.0;
    var acc_c = 0.0;
    var acc_m = 0.0;
    let row_base = (s * rank + r) * hidden;
    for (var k = li; k < half_h; k += 256u) {
        let c0 = cur[cur_off + 2u * k];
        let c1 = cur[cur_off + 2u * k + 1u];
        let m0 = msg[msg_off + 2u * k];
        let m1 = msg[msg_off + 2u * k + 1u];
        sq_c += c0 * c0 + c1 * c1;
        sq_m += m0 * m0 + m1 * m1;
        let wc = unpack2x16float(w_down[row_base + k]);
        let wm = unpack2x16float(w_down[row_base + half_h + k]);
        acc_c += wc.x * c0 + wc.y * c1;
        acc_m += wm.x * m0 + wm.y * m1;
    }
    sq_c = subgroupAdd(sq_c);
    sq_m = subgroupAdd(sq_m);
    acc_c = subgroupAdd(acc_c);
    acc_m = subgroupAdd(acc_m);
    if (lane == 0u) {
        ws_c[warp] = sq_c;
        ws_m[warp] = sq_m;
        wa_c[warp] = acc_c;
        wa_m[warp] = acc_m;
    }
    workgroupBarrier();
    if (li == 0u) {
        let tsq_c = ((ws_c[0] + ws_c[1]) + (ws_c[2] + ws_c[3])) + ((ws_c[4] + ws_c[5]) + (ws_c[6] + ws_c[7]));
        let tsq_m = ((ws_m[0] + ws_m[1]) + (ws_m[2] + ws_m[3])) + ((ws_m[4] + ws_m[5]) + (ws_m[6] + ws_m[7]));
        let tacc_c = ((wa_c[0] + wa_c[1]) + (wa_c[2] + wa_c[3])) + ((wa_c[4] + wa_c[5]) + (wa_c[6] + wa_c[7]));
        let tacc_m = ((wa_m[0] + wa_m[1]) + (wa_m[2] + wa_m[3])) + ((wa_m[4] + wa_m[5]) + (wa_m[6] + wa_m[7]));
        let inv_c = inverseSqrt(tsq_c / f32(hidden) + 1e-6);
        let inv_m = inverseSqrt(tsq_m / f32(hidden) + 1e-6);
        mid[st_idx * rank + r] = tacc_c * inv_c + tacc_m * inv_m;
    }
}
"#;

/// Fused static causal short convolution + residual add: `out[ch] = residual[ch] + static_conv(x[ch])`,
/// updating `hist` in place. Replaces `short_conv_step` + `block_push(None)` with 1 dispatch.
/// `p[0]`: channels, kernel, 0, 0.
pub(in crate::chain) const INKLING2_POST_CONV_ADD: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> hist: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = id.x + id.y * 16776960u;
    let c = p[0].x;
    if (ch >= c) { return; }
    let k = p[0].y;
    let past = k - 1u;
    let cur_v = x[ch];
    var acc = 0.0;
    for (var j = 0u; j < past; j++) {
        acc += hist[j * c + ch] * w[ch * k + j];
    }
    acc += cur_v * w[ch * k + past];
    out[ch] = residual[ch] + (acc + cur_v) * inverseSqrt(f32(k + 1u));
    for (var j = 0u; j + 1u < past; j++) {
        hist[j * c + ch] = hist[(j + 1u) * c + ch];
    }
    if (past > 0u) {
        hist[(past - 1u) * c + ch] = cur_v;
    }
}
"#;

/// Batched `seq`-token K/V short convolution (static or dynamic, read-only `kv_hist`) + K L2 clip
/// + V unweighted RMSNorm + ring-buffer KV cache store into packed `kv_cache` (`[2, kv_cap, kv_dim]`).
/// Follow with [`INKLING2_CONV_HIST_UPDATE`] (`n_streams = 2`) to advance `kv_hist`.
/// Grid: `(n_kv_heads, seq, 1)`.
/// `p[0]`: `n_kv_heads | (head_dim << 16)`, `kernel | (dyn_mode << 16)`, `kv_cap`, `past_len`;
/// `p[1]`: `seq`, `qk_clip bits`, `rms_eps bits`, `0`.
pub(in crate::chain) const INKLING2_KV_CONV_CLIP_STORE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> k_raw: array<f32>;
@group(0) @binding(1) var<storage, read> v_raw: array<f32>;
@group(0) @binding(2) var<storage, read> kv_w: array<f32>;
@group(0) @binding(3) var<storage, read> dyn_k: array<f32>;
@group(0) @binding(4) var<storage, read> dyn_v: array<f32>;
@group(0) @binding(5) var<storage, read> kv_hist: array<f32>;
@group(0) @binding(6) var<storage, read_write> kv_cache: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> pk: array<f32, 128>;
var<workgroup> pv: array<f32, 128>;
var<workgroup> tk: array<f32, 256>;
var<workgroup> tv: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_kv = p[0].x & 0xffffu;
    let head_dim = p[0].x >> 16u;
    let k = p[0].y & 0xffffu;
    let dyn_mode = p[0].y >> 16u;
    let kv_cap = p[0].z;
    let past_len = p[0].w;
    let seq = p[1].x;
    let h = wg.x;
    let t = wg.y;
    if (h >= n_kv || t >= seq) { return; }

    let past = k - 1u;
    let is_dyn = dyn_mode != 0u;
    let packed_dyn = dyn_mode == 2u;
    let qk_clip = bitcast<f32>(p[1].y);
    let rms_eps = bitcast<f32>(p[1].z);

    let kv_dim = n_kv * head_dim;
    let dyn_dim = n_kv * k;
    let kv_stride = kv_dim + select(0u, dyn_dim, packed_dyn);
    let v_w_off = kv_dim * k;
    let v_h_off = past * kv_dim;
    let conv_scale = select(inverseSqrt(f32(k + 1u)), inverseSqrt(f32(2u * k + 1u)), is_dyn);
    let row_off = t * kv_stride;

    var ksq = 0.0;
    var vsq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        let ch = h * head_dim + d;
        let kc = k_raw[row_off + ch];
        let vc = v_raw[row_off + ch];
        var ak = 0.0;
        var av = 0.0;
        for (var j = 0u; j < k; j++) {
            let idx = t + j;
            var xk = 0.0;
            var xv = 0.0;
            if (idx < past) {
                xk = kv_hist[idx * kv_dim + ch];
                xv = kv_hist[v_h_off + idx * kv_dim + ch];
            } else {
                let sr = (idx - past) * kv_stride + ch;
                xk = k_raw[sr];
                xv = v_raw[sr];
            }
            var wk = kv_w[ch * k + j];
            var wv = kv_w[v_w_off + ch * k + j];
            if (packed_dyn) {
                let lag = k - 1u - j;
                wk += k_raw[row_off + kv_dim + h * k + lag];
                wv += v_raw[row_off + kv_dim + h * k + lag];
            } else if (is_dyn) {
                let lag = k - 1u - j;
                let di = (t * n_kv + h) * k + lag;
                wk += dyn_k[di];
                wv += dyn_v[di];
            }
            ak += xk * wk;
            av += xv * wv;
        }
        let ko = (ak + kc) * conv_scale;
        let vo = (av + vc) * conv_scale;
        tk[d] = ko;
        tv[d] = vo;
        ksq += ko * ko;
        vsq += vo * vo;
    }
    pk[li] = ksq;
    pv[li] = vsq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) {
            pk[li] += pk[li + st];
            pv[li] += pv[li + st];
        }
        workgroupBarrier();
    }
    let k_fac = min(1.0, qk_clip / (sqrt(pk[0]) + 1e-6));
    let v_inv = inverseSqrt(pv[0] / f32(head_dim) + rms_eps);
    let write_row = (past_len + t) % kv_cap;
    let dst_k = write_row * kv_dim + h * head_dim;
    let dst_v = kv_cap * kv_dim + dst_k;
    for (var d = li; d < head_dim; d += 128u) {
        kv_cache[dst_k + d] = tk[d] * k_fac;
        kv_cache[dst_v + d] = tv[d] * v_inv;
    }
}
"#;

/// Subgroup version of [`INKLING2_KV_CONV_CLIP_STORE_ROWS`]: 1 `workgroupBarrier()` instead of 8.
pub(in crate::chain) const INKLING2_KV_CONV_CLIP_STORE_ROWS_SG: &str = r#"
@group(0) @binding(0) var<storage, read> k_raw: array<f32>;
@group(0) @binding(1) var<storage, read> v_raw: array<f32>;
@group(0) @binding(2) var<storage, read> kv_w: array<f32>;
@group(0) @binding(3) var<storage, read> dyn_k: array<f32>;
@group(0) @binding(4) var<storage, read> dyn_v: array<f32>;
@group(0) @binding(5) var<storage, read> kv_hist: array<f32>;
@group(0) @binding(6) var<storage, read_write> kv_cache: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wpk: array<f32, 4>;
var<workgroup> wpv: array<f32, 4>;
var<workgroup> tk: array<f32, 256>;
var<workgroup> tv: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let n_kv = p[0].x & 0xffffu;
    let head_dim = p[0].x >> 16u;
    let k = p[0].y & 0xffffu;
    let dyn_mode = p[0].y >> 16u;
    let kv_cap = p[0].z;
    let past_len = p[0].w;
    let seq = p[1].x;
    let h = wg.x;
    let t = wg.y;
    if (h >= n_kv || t >= seq) { return; }

    let past = k - 1u;
    let is_dyn = dyn_mode != 0u;
    let packed_dyn = dyn_mode == 2u;
    let qk_clip = bitcast<f32>(p[1].y);
    let rms_eps = bitcast<f32>(p[1].z);

    let kv_dim = n_kv * head_dim;
    let dyn_dim = n_kv * k;
    let kv_stride = kv_dim + select(0u, dyn_dim, packed_dyn);
    let v_w_off = kv_dim * k;
    let v_h_off = past * kv_dim;
    let conv_scale = select(inverseSqrt(f32(k + 1u)), inverseSqrt(f32(2u * k + 1u)), is_dyn);
    let row_off = t * kv_stride;

    var ksq = 0.0;
    var vsq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        let ch = h * head_dim + d;
        let kc = k_raw[row_off + ch];
        let vc = v_raw[row_off + ch];
        var ak = 0.0;
        var av = 0.0;
        for (var j = 0u; j < k; j++) {
            let idx = t + j;
            var xk = 0.0;
            var xv = 0.0;
            if (idx < past) {
                xk = kv_hist[idx * kv_dim + ch];
                xv = kv_hist[v_h_off + idx * kv_dim + ch];
            } else {
                let sr = (idx - past) * kv_stride + ch;
                xk = k_raw[sr];
                xv = v_raw[sr];
            }
            var wk = kv_w[ch * k + j];
            var wv = kv_w[v_w_off + ch * k + j];
            if (packed_dyn) {
                let lag = k - 1u - j;
                wk += k_raw[row_off + kv_dim + h * k + lag];
                wv += v_raw[row_off + kv_dim + h * k + lag];
            } else if (is_dyn) {
                let lag = k - 1u - j;
                let di = (t * n_kv + h) * k + lag;
                wk += dyn_k[di];
                wv += dyn_v[di];
            }
            ak += xk * wk;
            av += xv * wv;
        }
        let ko = (ak + kc) * conv_scale;
        let vo = (av + vc) * conv_scale;
        tk[d] = ko;
        tv[d] = vo;
        ksq += ko * ko;
        vsq += vo * vo;
    }
    ksq = subgroupAdd(ksq);
    vsq = subgroupAdd(vsq);
    if (lane == 0u) {
        wpk[warp] = ksq;
        wpv[warp] = vsq;
    }
    workgroupBarrier();
    let k_tot = (wpk[0] + wpk[1]) + (wpk[2] + wpk[3]);
    let v_tot = (wpv[0] + wpv[1]) + (wpv[2] + wpv[3]);
    let k_fac = min(1.0, qk_clip / (sqrt(k_tot) + 1e-6));
    let v_inv = inverseSqrt(v_tot / f32(head_dim) + rms_eps);
    let write_row = (past_len + t) % kv_cap;
    let dst_k = write_row * kv_dim + h * head_dim;
    let dst_v = kv_cap * kv_dim + dst_k;
    for (var d = li; d < head_dim; d += 128u) {
        kv_cache[dst_k + d] = tk[d] * k_fac;
        kv_cache[dst_v + d] = tv[d] * v_inv;
    }
}
"#;

/// Advance 1-stream (post-conv) or 2-stream (packed K+V conv) short-convolution history buffer
/// (`hist`: `[n_streams, past, c]`) by `seq` input rows (`x0`/`x1`: `[seq, stride]`).
/// Grid: `(ceil(c / 256), n_streams, 1)`.
/// `p[0]`: `c`, `stride`, `past`, `seq`; `p[1]`: `n_streams`, `0`, `0`, `0`.
pub(in crate::chain) const INKLING2_CONV_HIST_UPDATE: &str = r#"
@group(0) @binding(0) var<storage, read> x0: array<f32>;
@group(0) @binding(1) var<storage, read> x1: array<f32>;
@group(0) @binding(6) var<storage, read_write> hist: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = p[0].x;
    let ch = wg.x * 256u + li;
    let s = wg.y;
    if (ch >= c || s >= p[1].x) { return; }
    let stride = p[0].y;
    let past = p[0].z;
    let seq = p[0].w;
    if (past == 0u || seq == 0u) { return; }
    let h_base = s * past * c;
    if (seq >= past) {
        let start = seq - past;
        for (var j = 0u; j < past; j++) {
            let sr = (start + j) * stride + ch;
            hist[h_base + j * c + ch] = select(x0[sr], x1[sr], s == 1u);
        }
    } else {
        let keep = past - seq;
        for (var j = 0u; j < keep; j++) {
            hist[h_base + j * c + ch] = hist[h_base + (j + seq) * c + ch];
        }
        for (var j = 0u; j < seq; j++) {
            let sr = j * stride + ch;
            hist[h_base + (keep + j) * c + ch] = select(x0[sr], x1[sr], s == 1u);
        }
    }
}
"#;

/// Batched `seq`-token static causal depthwise short convolution (read-only `hist`) with optional
/// residual add (`p[0].w != 0`). Follow with [`INKLING2_CONV_HIST_UPDATE`] (`n_streams = 1`).
/// Grid: `(ceil(c / 256), seq, 1)`.
/// `p[0]`: `c`, `kernel`, `seq`, `add_residual (0 or 1)`.
pub(in crate::chain) const INKLING2_STATIC_CONV_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> hist: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = p[0].x;
    let k = p[0].y;
    let seq = p[0].z;
    let add_res = p[0].w != 0u;
    let ch = wg.x * 256u + li;
    let t = wg.y;
    if (ch >= c || t >= seq) { return; }
    let past = k - 1u;
    let cur_v = x[t * c + ch];
    var acc = 0.0;
    for (var j = 0u; j < k; j++) {
        let idx = t + j;
        var xv = 0.0;
        if (idx < past) {
            xv = hist[idx * c + ch];
        } else {
            xv = x[(idx - past) * c + ch];
        }
        acc += xv * w[ch * k + j];
    }
    var y = (acc + cur_v) * inverseSqrt(f32(k + 1u));
    if (add_res) {
        y += residual[t * c + ch];
    }
    out[t * c + ch] = y;
}
"#;

/// FlashAttention-style Q-tiled (`BQ = 8` query tokens per WG, 1 warp/token) + shared-memory
/// K/V-tiled (`BK = 8`, 8 KB workgroup storage) fused subgroup prefill attention on packed `qrsg`
/// (`[seq, 2*q_dim + r_dim + n_heads]`), `head_dim == 128`, `d_rel <= 32`.
/// Fuses Q/R learned scale + L2 clip + per-token `tau`, on-the-fly relative-position bias,
/// per-query sinks, causal/SWA masking, and `sigmoid(gate)` output gating into ONE dispatch!
/// Grid: `(n_heads, ceil(seq / 8), 1)`.
/// `p[0]`: `n_heads | (n_kv << 16)`, `d_rel | (rel_extent << 16)`, `kv_cap`, `swa`;
/// `p[1]`: `past_len`, `seq`, `0`, `0`.
/// `tau_params`: `[qk_clip, rel_clip, bitcast<f32>(log_scale_floor), log_scale_alpha]`.
pub(in crate::chain) const INKLING2_ATTEND_PREFILL_SG_PACKED: &str = r#"
@group(0) @binding(0) var<storage, read> qrsg: array<f32>;
@group(0) @binding(1) var<storage, read> attn_consts: array<f32>;
@group(0) @binding(2) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(3) var<storage, read> tau_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sk: array<f32, 1024>;
var<workgroup> sv: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let n_heads = p[0].x & 0xffffu;
    let n_kv = p[0].x >> 16u;
    let d_rel = p[0].y & 0xffffu;
    let rel_extent = p[0].y >> 16u;
    let kv_cap = p[0].z;
    let swa = p[0].w;
    let past_len = p[1].x;
    let seq = p[1].y;

    let h = wg.x;
    let t0 = wg.y * 8u;
    if (h >= n_heads || t0 >= seq) { return; }

    let qk_clip = bitcast<f32>(tau_params[0]);
    let rel_clip = bitcast<f32>(tau_params[1]);
    let floor_f = f32(tau_params[2]);
    let alpha = bitcast<f32>(tau_params[3]);

    let t = t0 + warp;
    let is_active = t < seq;
    let t_safe = min(t, seq - 1u);
    let t_end = min(seq, t0 + 8u);

    let group = n_heads / n_kv;
    let hk = h / group;
    let q_dim = n_heads * 128u;
    let r_dim = n_heads * d_rel;
    let row_stride = 2u * q_dim + r_dim + n_heads;
    let r_buf_off = q_dim;
    let sink_buf_off = q_dim + r_dim;
    let gate_buf_off = sink_buf_off + n_heads;
    let kv_dim = n_kv * 128u;
    let v_cache_off = kv_cap * kv_dim;
    let q_clip_off = rel_extent * d_rel;
    let r_clip_off = q_clip_off + q_dim;
    let l4 = lane * 4u;

    let q_pos = past_len + t_safe;
    var tau = 1.0;
    if (swa == 0u) {
        let ratio = max(1.0, f32(q_pos + 1u) / floor_f);
        tau = 1.0 + alpha * log(ratio);
    }

    let row_base = t_safe * row_stride;
    let q_head_off = h * 128u + l4;
    let q_off = row_base + q_head_off;
    let qc_off = q_clip_off + q_head_off;
    var q4 = vec4<f32>(qrsg[q_off], qrsg[q_off + 1u], qrsg[q_off + 2u], qrsg[q_off + 3u])
        * vec4<f32>(attn_consts[qc_off], attn_consts[qc_off + 1u], attn_consts[qc_off + 2u], attn_consts[qc_off + 3u]);
    let q_sq = subgroupAdd(dot(q4, q4));
    q4 *= min(1.0, qk_clip / (sqrt(q_sq) + 1e-6)) * tau;

    var r_val = 0.0;
    if (lane < d_rel) {
        let ro = h * d_rel + lane;
        r_val = qrsg[row_base + r_buf_off + ro] * attn_consts[r_clip_off + ro];
    }
    let r_sq = subgroupAdd(r_val * r_val);
    r_val *= min(1.0, rel_clip / (sqrt(r_sq) + 1e-6)) * tau;

    var m = qrsg[row_base + sink_buf_off + h];
    var s = 1.0;
    var acc = vec4<f32>(0.0);
    let kv_end_t = q_pos + 1u;
    let kv_first_t = select(0u, select(0u, kv_end_t - swa, kv_end_t > swa), swa != 0u);

    let q_pos_min = past_len + t0;
    let kv_end_min = q_pos_min + 1u;
    let kv_first_min = select(0u, select(0u, kv_end_min - swa, kv_end_min > swa), swa != 0u);
    let kv_end_max = past_len + t_end;
    let scale = inverseSqrt(128.0);

    for (var k_block = kv_first_min; k_block < kv_end_max; k_block += 8u) {
        let load_k = k_block + warp;
        let sb = warp * 128u + l4;
        if (load_k < kv_end_max) {
            let row = load_k % kv_cap;
            let kv_off = row * kv_dim + hk * 128u + l4;
            sk[sb] = kv_cache[kv_off];
            sk[sb + 1u] = kv_cache[kv_off + 1u];
            sk[sb + 2u] = kv_cache[kv_off + 2u];
            sk[sb + 3u] = kv_cache[kv_off + 3u];
            let vo = v_cache_off + kv_off;
            sv[sb] = kv_cache[vo];
            sv[sb + 1u] = kv_cache[vo + 1u];
            sv[sb + 2u] = kv_cache[vo + 2u];
            sv[sb + 3u] = kv_cache[vo + 3u];
        }
        workgroupBarrier();

        let tile_n = min(8u, kv_end_max - k_block);
        for (var j = 0u; j < tile_n; j++) {
            let k_pos = k_block + j;
            let valid = is_active && (k_pos >= kv_first_t) && (k_pos <= q_pos);
            let kb = j * 128u + l4;
            let k4 = vec4<f32>(sk[kb], sk[kb + 1u], sk[kb + 2u], sk[kb + 3u]);
            let dot_qk = subgroupAdd(dot(q4, k4));
            let dist = select(rel_extent, q_pos - k_pos, valid);
            var rb_term = 0.0;
            if (dist < rel_extent && lane < d_rel) {
                rb_term = r_val * attn_consts[dist * d_rel + lane];
            }
            let bias = subgroupAdd(rb_term);
            if (valid) {
                let score = dot_qk * scale + bias;
                let v4 = vec4<f32>(sv[kb], sv[kb + 1u], sv[kb + 2u], sv[kb + 3u]);
                if (score > m) {
                    let rs = exp(m - score);
                    s = s * rs + 1.0;
                    acc = acc * rs + v4;
                    m = score;
                } else {
                    let w = exp(score - m);
                    s += w;
                    acc += v4 * w;
                }
            }
        }
        workgroupBarrier();
    }

    if (is_active) {
        let go = row_base + gate_buf_off + q_head_off;
        let g4 = vec4<f32>(qrsg[go], qrsg[go + 1u], qrsg[go + 2u], qrsg[go + 3u]);
        let sig = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-g4));
        let res = (acc * (1.0 / s)) * sig;
        let oo = t * q_dim + q_head_off;
        out[oo] = res.x;
        out[oo + 1u] = res.y;
        out[oo + 2u] = res.z;
        out[oo + 3u] = res.w;
    }
}
"#;

/// Portable fallback batched prefill GQA attention for any `head_dim <= 256`, `d_rel <= 64`,
/// packed (`(p[1].z & 0x80000000u) != 0`) or unpacked, and devices without 32-lane subgroups.
/// Grid: `(n_heads, seq, 1)`.
/// `p[0]`: `n_heads | (n_kv << 16)`, `d_rel | (rel_extent << 16)`, `kv_cap`, `swa`;
/// `p[1]`: `past_len`, `seq`, `head_dim | (packed << 31)`, `0`.
pub(in crate::chain) const INKLING2_ATTEND_PREFILL_PORTABLE: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read> sinks: array<f32>;
@group(0) @binding(3) var<storage, read> attn_consts: array<f32>;
@group(0) @binding(4) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(5) var<storage, read> gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> tau_params: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<f32, 128>;
var<workgroup> q_loc: array<f32, 256>;
var<workgroup> r_loc: array<f32, 64>;
var<workgroup> acc_s: array<f32, 256>;
var<workgroup> cur_score: f32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_heads = p[0].x & 0xffffu;
    let n_kv = p[0].x >> 16u;
    let d_rel = p[0].y & 0xffffu;
    let rel_extent = p[0].y >> 16u;
    let kv_cap = p[0].z;
    let swa = p[0].w;
    let past_len = p[1].x;
    let seq = p[1].y;
    let packed = (p[1].z & 0x80000000u) != 0u;
    let head_dim = p[1].z & 0x7fffffffu;

    let h = wg.x;
    let t = wg.y;
    if (h >= n_heads || t >= seq) { return; }

    let qk_clip = bitcast<f32>(tau_params[0]);
    let rel_clip = bitcast<f32>(tau_params[1]);
    let floor_f = f32(tau_params[2]);
    let alpha = bitcast<f32>(tau_params[3]);

    let group = n_heads / n_kv;
    let hk = h / group;
    let q_dim = n_heads * head_dim;
    let r_dim = n_heads * d_rel;
    let kv_dim = n_kv * head_dim;
    let v_cache_off = kv_cap * kv_dim;
    let q_clip_off = rel_extent * d_rel;
    let r_clip_off = q_clip_off + q_dim;
    let row_stride = 2u * q_dim + r_dim + n_heads;

    let q_row = select(t * q_dim, t * row_stride, packed);
    let r_row = select(t * r_dim, t * row_stride + q_dim, packed);
    let s_row = select(t * n_heads, t * row_stride + q_dim + r_dim, packed);
    let g_row = select(t * q_dim, t * row_stride + q_dim + r_dim + n_heads, packed);

    let q_pos = past_len + t;
    var tau = 1.0;
    if (swa == 0u) {
        let ratio = max(1.0, f32(q_pos + 1u) / floor_f);
        tau = 1.0 + alpha * log(ratio);
    }
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);
    let scale = inverseSqrt(f32(head_dim));
    let q_base = h * head_dim;

    var qsq = 0.0;
    for (var d = li; d < head_dim; d += 128u) {
        let v = q[q_row + q_base + d] * attn_consts[q_clip_off + q_base + d];
        q_loc[d] = v;
        qsq += v * v;
        acc_s[d] = 0.0;
    }
    part[li] = qsq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let q_fac = min(1.0, qk_clip / (sqrt(part[0]) + 1e-6)) * tau;
    for (var d = li; d < head_dim; d += 128u) { q_loc[d] *= q_fac; }

    var rsq = 0.0;
    for (var j = li; j < d_rel; j += 128u) {
        let ro = h * d_rel + j;
        let v = r[r_row + ro] * attn_consts[r_clip_off + ro];
        r_loc[j] = v;
        rsq += v * v;
    }
    part[li] = rsq;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let r_fac = min(1.0, rel_clip / (sqrt(part[0]) + 1e-6)) * tau;
    for (var j = li; j < d_rel; j += 128u) { r_loc[j] *= r_fac; }
    workgroupBarrier();

    var m = sinks[s_row + h];
    var s = 1.0;
    for (var k_pos = kv_first; k_pos < kv_end; k_pos++) {
        let row = k_pos % kv_cap;
        let kv_base = row * kv_dim + hk * head_dim;
        var dq = 0.0;
        for (var d = li; d < head_dim; d += 128u) {
            dq += q_loc[d] * kv_cache[kv_base + d];
        }
        part[li] = dq;
        workgroupBarrier();
        for (var st = 64u; st > 0u; st /= 2u) {
            if (li < st) { part[li] += part[li + st]; }
            workgroupBarrier();
        }
        if (li == 0u) {
            var bias = 0.0;
            let dist = q_pos - k_pos;
            if (dist < rel_extent) {
                let rb = dist * d_rel;
                for (var j = 0u; j < d_rel; j++) { bias += r_loc[j] * attn_consts[rb + j]; }
            }
            cur_score = part[0] * scale + bias;
        }
        workgroupBarrier();
        let score = cur_score;
        let vb = v_cache_off + kv_base;
        if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            for (var d = li; d < head_dim; d += 128u) {
                acc_s[d] = acc_s[d] * rs + kv_cache[vb + d];
            }
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            for (var d = li; d < head_dim; d += 128u) {
                acc_s[d] += kv_cache[vb + d] * w;
            }
        }
        workgroupBarrier();
    }

    let inv_s = 1.0 / s;
    let out_base = t * q_dim + q_base;
    for (var d = li; d < head_dim; d += 128u) {
        let g = gate[g_row + q_base + d];
        out[out_base + d] = (acc_s[d] * inv_s) / (1.0 + exp(-g));
    }
}
"#;

/// Split-K pass 1 for `seq == 1` fused packed GQA attention (`head_dim == 128`, `d_rel <= 32`):
/// partitions `[kv_first, kv_end)` across `n_splits` workgroups per query head so long-context
/// decode saturates all GPU SMs while keeping uniform/grid parameters constant for `ChainRecipe`.
/// Grid: `(n_heads, n_splits, 1)`.
/// `split_buf` layout: `[n_heads, n_splits, 132]` (`0..128` acc, `128` m, `129` s).
/// `p[0]`: `n_heads | (n_splits << 16)`, `n_kv_heads`, `d_rel`, `rel_extent`;
/// `p[1]`: `kv_cap`, `swa`, `qk_clip_bits`, `rel_clip_bits`.
pub(in crate::chain) const INKLING2_ATTEND_SPLIT_PART_SG: &str = r#"
@group(0) @binding(0) var<storage, read> qrsg: array<f32>;
@group(0) @binding(1) var<storage, read> attn_consts: array<f32>;
@group(0) @binding(2) var<storage, read> kv_cache: array<f32>;
@group(0) @binding(3) var<storage, read> step_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> split_buf: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> wm: array<f32, 8>;
var<workgroup> ws: array<f32, 8>;
var<workgroup> wacc: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) warp: u32) {
    let n_heads = p[0].x & 0xffffu;
    let n_splits = p[0].x >> 16u;
    let h = wg.x;
    let sp = wg.y;
    if (h >= n_heads || sp >= n_splits) { return; }
    let n_kv = p[0].y;
    let d_rel = p[0].z;
    let rel_extent = p[0].w;
    let kv_cap = p[1].x;
    let swa = p[1].y;
    let qk_clip = bitcast<f32>(p[1].z);
    let rel_clip = bitcast<f32>(p[1].w);

    let group = n_heads / n_kv;
    let hk = h / group;
    let q_dim = n_heads * 128u;
    let r_buf_off = q_dim;
    let kv_dim = n_kv * 128u;
    let v_cache_off = kv_cap * kv_dim;
    let q_clip_off = rel_extent * d_rel;
    let r_clip_off = q_clip_off + q_dim;

    let q_pos = step_params[0];
    let tau = select(1.0, bitcast<f32>(step_params[1]), swa == 0u);
    let kv_end = q_pos + 1u;
    let kv_first = select(0u, select(0u, kv_end - swa, kv_end > swa), swa != 0u);
    let total = kv_end - kv_first;
    let chunk = max(32u, (total + n_splits - 1u) / n_splits);
    let sp_lo = kv_first + sp * chunk;
    let sp_hi = min(kv_end, sp_lo + chunk);

    let q_off = h * 128u + lane * 4u;
    let qc_off = q_clip_off + q_off;
    var q4 = vec4<f32>(qrsg[q_off], qrsg[q_off + 1u], qrsg[q_off + 2u], qrsg[q_off + 3u])
        * vec4<f32>(attn_consts[qc_off], attn_consts[qc_off + 1u], attn_consts[qc_off + 2u], attn_consts[qc_off + 3u]);
    let q_sq = subgroupAdd(dot(q4, q4));
    q4 *= min(1.0, qk_clip / (sqrt(q_sq) + 1e-6)) * tau;

    var r_val = 0.0;
    if (lane < d_rel) {
        let ro = h * d_rel + lane;
        r_val = qrsg[r_buf_off + ro] * attn_consts[r_clip_off + ro];
    }
    let r_sq = subgroupAdd(r_val * r_val);
    r_val *= min(1.0, rel_clip / (sqrt(r_sq) + 1e-6)) * tau;

    let scale = inverseSqrt(128.0);
    var m = -1.0e30;
    var s = 0.0;
    var acc = vec4<f32>(0.0);

    for (var k_pos = sp_lo + warp; k_pos < sp_hi; k_pos += 8u) {
        let row = k_pos % kv_cap;
        let kv_off = row * kv_dim + hk * 128u + lane * 4u;
        let k4 = vec4<f32>(kv_cache[kv_off], kv_cache[kv_off + 1u], kv_cache[kv_off + 2u], kv_cache[kv_off + 3u]);
        let dot_qk = subgroupAdd(dot(q4, k4));
        let dist = q_pos - k_pos;
        var rb_term = 0.0;
        if (dist < rel_extent && lane < d_rel) {
            rb_term = r_val * attn_consts[dist * d_rel + lane];
        }
        let bias = subgroupAdd(rb_term);
        let score = dot_qk * scale + bias;
        let vo = v_cache_off + kv_off;
        let v4 = vec4<f32>(kv_cache[vo], kv_cache[vo + 1u], kv_cache[vo + 2u], kv_cache[vo + 3u]);
        if (s == 0.0) {
            m = score;
            s = 1.0;
            acc = v4;
        } else if (score > m) {
            let rs = exp(m - score);
            s = s * rs + 1.0;
            acc = acc * rs + v4;
            m = score;
        } else {
            let w = exp(score - m);
            s += w;
            acc += v4 * w;
        }
    }

    if (lane == 0u) {
        wm[warp] = m;
        ws[warp] = s;
    }
    let wb = warp * 128u + lane * 4u;
    wacc[wb] = acc.x;
    wacc[wb + 1u] = acc.y;
    wacc[wb + 2u] = acc.z;
    wacc[wb + 3u] = acc.w;
    workgroupBarrier();

    if (warp == 0u) {
        var gm = -1.0e30;
        var any_w = false;
        for (var w = 0u; w < 8u; w++) {
            if (ws[w] > 0.0) {
                gm = select(wm[w], max(gm, wm[w]), any_w);
                any_w = true;
            }
        }
        var gs = 0.0;
        var gacc = vec4<f32>(0.0);
        if (any_w) {
            for (var w = 0u; w < 8u; w++) {
                if (ws[w] > 0.0) {
                    let f = exp(wm[w] - gm);
                    gs += ws[w] * f;
                    let ob = w * 128u + lane * 4u;
                    gacc += vec4<f32>(wacc[ob], wacc[ob + 1u], wacc[ob + 2u], wacc[ob + 3u]) * f;
                }
            }
        }
        let sb_base = (h * n_splits + sp) * 132u;
        let so = sb_base + lane * 4u;
        split_buf[so] = gacc.x;
        split_buf[so + 1u] = gacc.y;
        split_buf[so + 2u] = gacc.z;
        split_buf[so + 3u] = gacc.w;
        if (lane == 0u) {
            split_buf[sb_base + 128u] = gm;
            split_buf[sb_base + 129u] = gs;
        }
    }
}
"#;

/// Split-K pass 2 for `seq == 1` fused packed GQA attention: combines `n_splits` partials with
/// per-query `sinks[h]`, multiplies `sigmoid(gate)`, writes `out` and optional inline Q8 `out_q8`.
/// Grid: `(n_heads, 1, 1)`, 1 warp (`@workgroup_size(32)`).
/// `p[0]`: `n_heads | (write_q8 << 31)`, `n_splits`, `sink_buf_off`, `gate_buf_off`.
pub(in crate::chain) const INKLING2_ATTEND_SPLIT_JOIN_SG: &str = r#"
@group(0) @binding(0) var<storage, read> qrsg: array<f32>;
@group(0) @binding(1) var<storage, read> split_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_q8: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
    let write_q8 = (p[0].x & 0x80000000u) != 0u;
    let n_heads = p[0].x & 0x7fffffffu;
    let h = wg.x;
    if (h >= n_heads) { return; }
    let n_splits = p[0].y;
    let sink_buf_off = p[0].z;
    let gate_buf_off = p[0].w;

    let sink_v = qrsg[sink_buf_off + h];
    var gm = sink_v;
    for (var sp = 0u; sp < n_splits; sp++) {
        let base = (h * n_splits + sp) * 132u;
        if (split_buf[base + 129u] > 0.0) {
            gm = max(gm, split_buf[base + 128u]);
        }
    }
    var gs = exp(sink_v - gm);
    var gacc = vec4<f32>(0.0);
    for (var sp = 0u; sp < n_splits; sp++) {
        let base = (h * n_splits + sp) * 132u;
        let ps = split_buf[base + 129u];
        if (ps > 0.0) {
            let f = exp(split_buf[base + 128u] - gm);
            gs += ps * f;
            let ob = base + lane * 4u;
            gacc += vec4<f32>(split_buf[ob], split_buf[ob + 1u], split_buf[ob + 2u], split_buf[ob + 3u]) * f;
        }
    }

    let q_off = h * 128u + lane * 4u;
    let go = gate_buf_off + q_off;
    let g4 = vec4<f32>(qrsg[go], qrsg[go + 1u], qrsg[go + 2u], qrsg[go + 3u]);
    let sig = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-g4));
    let res = (gacc * (1.0 / gs)) * sig;
    out[q_off] = res.x;
    out[q_off + 1u] = res.y;
    out[q_off + 2u] = res.z;
    out[q_off + 3u] = res.w;
    if (write_q8) {
        let b = h * 4u + (lane / 8u);
        let part = lane % 8u;
        let xs_at = n_heads * 8u;
        var amax = max(max(abs(res.x), abs(res.y)), max(abs(res.z), abs(res.w)));
        amax = max(amax, subgroupShuffleXor(amax, 1u));
        amax = max(amax, subgroupShuffleXor(amax, 2u));
        amax = max(amax, subgroupShuffleXor(amax, 4u));
        let d = amax / 127.0;
        let inv = select(0.0, 1.0 / d, d > 0.0);
        let qv = clamp(vec4<i32>(round(res * inv)), vec4<i32>(-127), vec4<i32>(127));
        let qb = bitcast<vec4<u32>>(qv) & vec4<u32>(0xffu);
        var hs = qv.x + qv.y + qv.z + qv.w;
        hs += subgroupShuffleXor(hs, 1u);
        hs += subgroupShuffleXor(hs, 2u);
        let other = subgroupShuffleXor(hs, 4u);
        out_q8[b * 8u + part] = qb.x | (qb.y << 8u) | (qb.z << 16u) | (qb.w << 24u);
        if (part == 0u) {
            let at = (xs_at + b) * 4u;
            out_q8[at] = bitcast<u32>(d);
            out_q8[at + 1u] = bitcast<u32>(d * f32(hs));
            out_q8[at + 2u] = bitcast<u32>(d * f32(other));
            out_q8[at + 3u] = 0u;
        }
    }
}
"#;
