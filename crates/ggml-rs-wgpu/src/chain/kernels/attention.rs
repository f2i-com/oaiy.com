//! Attention without tensor cores: RoPE and the cache's store, a prompt's rows (in runs of positions, or tiled), a step's one query (its parts, by head or by a KV head's group, over f32 or f16 halves).

/// RoPE in place on `y` (`[p[0].w rows, p[0].x heads, p[0].y head_dim]`), on the first `rot` (`p[1].x`, 0: all) of
/// each head: row `r`'s pair `k` by the sine `w[r * rot + 2k]` and cosine `w[r * rot + 2k + 1]` (made on the host, as
/// the CPU's rope makes them), the pairs `(2k, 2k + 1)` or with `p[0].z` `(k, k + rot / 2)`.
pub(in crate::chain) const ROPE: &str = r#"
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
pub(in crate::chain) const STORE_ROWS: &str = r#"
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
pub(in crate::chain) const ATTENTION_ROWS_PART: &str = r#"
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
pub(in crate::chain) const ATTENTION_ROWS_PART4: &str = r#"
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
pub(in crate::chain) const ATTENTION_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y4: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

const HD4: u32 = HD4_u;
const TQ: u32 = TQ_u;
const TK: u32 = TK_u;
const KS: u32 = HD4_u + 1u;

var<workgroup> kt: array<vec4<f32>, KT_LEN>;
// the block's scores, then weights, `[key][query]`, read four queries at a time. Scalars, not vec4s: a thread a query
// writes its own, and WGSL lets a write to one component of a vector in memory write the whole vector (Metal does),
// so four threads writing one would race
var<workgroup> pt: array<f32, PT_LEN>;
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
            for (var j = 0u; j < TK; j++) { mb = max(mb, pt[j * TQ + li]); }
            let mn = max(m, mb);
            let a = exp(m - mn);
            var sum = 0.0;
            for (var j = 0u; j < TK; j++) {
                let s = pt[j * TQ + li];
                var e = 0.0;
                if (s > -1.0e38) { e = exp(s - mn); }
                pt[j * TQ + li] = e;
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
pub(in crate::chain) fn attention_tiled(hd: usize) -> String {
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
        scores += "        pt[sj * TQ + 4u * sg] = w0;\n        pt[sj * TQ + 4u * sg + 1u] = w1;\n        pt[sj * TQ + 4u * sg + 2u] = w2;\n        pt[sj * TQ + 4u * sg + 3u] = w3;\n";
    } else {
        assert_eq!(qpt, 1);
        scores += "        pt[sj * TQ + sg] = w0;\n";
    }
    let mut values = String::new();
    for i in 0..qo {
        values += &format!("        o{i} *= al[vg * {qo}u + {i}u];\n");
    }
    values += "        for (var j = 0u; j < TK; j++) {\n            let v = kt[j * KS + vl];\n";
    for c in 0..qo / 4 {
        values += &format!("            let p{c} = vec4<f32>(pt[j * TQ + 4u * (vq4 + {c}u)], pt[j * TQ + 4u * (vq4 + {c}u) + 1u], pt[j * TQ + 4u * (vq4 + {c}u) + 2u], pt[j * TQ + 4u * (vq4 + {c}u) + 3u]);\n");
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
        .replace("PT_LEN", &(tk * tq).to_string())
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
pub(in crate::chain) const ATTENTION_DISPATCH_FLOPS: f64 = (1u64 << 37) as f64;

/// The runs of [`ATTENTION_ROWS_PART`] put together, a workgroup a (head, query); a run with nothing (`l` 0) adds
/// nothing. `p` as for the parts.
pub(in crate::chain) const ATTENTION_ROWS_JOIN: &str = r#"
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

/// Positions a workgroup of [`ATTENTION_PART`] takes.
pub(in crate::chain) const SPLIT: usize = 256;

/// One query's attention over positions split in runs of 256, a workgroup a (query head `h`, run): its scores, their
/// largest `m`, the sum `l` of their exponentials after `m`, and the exponentials' weighted values, into the scratch
/// after the output. `w` the layer's cache (row `t`: K `[n_kv, hd]` then V), `x` the query `[n_h, hd]`, `y` the output
/// `[n_h, hd]` then each (head, run)'s `hd` weighted values and then its `m` and `l`. `p[0]`: `n_h`, `n_kv`, `hd`, the
/// positions' end; `p[1]`: their start, the runs, the bits of the scale. (A workgroup a head walked every position on
/// one thread a dimension: 37 ms of a 3B Llama's decode step at 2,000 positions.)
pub(in crate::chain) const ATTENTION_PART: &str = r#"
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
pub(in crate::chain) const ATTENTION_PART4: &str = r#"
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
pub(in crate::chain) const ATTENTION_JOIN: &str = r#"
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

/// [`ATTENTION_PART4`] for a KV head's whole group of query heads at once, or a whole share of them
/// (`attention_part_group`'s kernel before the group's size is put in), a workgroup a (group, run): each key and each
/// value read once for the group's heads,
/// where a workgroup a query head read them a head each (the 27B's six heads a KV head: at 15,888 positions a step's
/// sixteen layers' parts read the cache's 2 GB six times over, 4.3 ms of its 16.8). The scores a thread a key, a sum a
/// head; the softmaxes together, a head a vec4's component; the values a thread a column of four over a share of the
/// run's keys (256 threads: as many shares as the head's quarter goes into them), the shares' sums put together
/// through the workgroup's memory. The same parts, largest and sum a head for the join.
pub(in crate::chain) const ATTENTION_PART_GROUP: &str = r#"
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
    // the group of query heads (a KV head's, or a whole share of them) and its KV head
    let grp = wg.x;
    let kh = grp * G / (n_h / n_kv);
    let run = wg.y;
    let hd4 = hd / 4u;
    let kvd4 = n_kv * hd4;
    let row4 = 2u * kvd4;
    for (var i = li; i < G * hd4; i += 256u) { buf[i] = q4[grp * G * hd4 + i]; }
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
pub(in crate::chain) fn attention_part_group(g: usize) -> String {
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
            &each(&|i| format!("        {{\n            let at = n_h * hd + ((grp * G + {i}u) * runs + run) * hd + d4 * 4u;\n            y[at] = v{i}.x;\n            y[at + 1u] = v{i}.y;\n            y[at + 2u] = v{i}.z;\n            y[at + 3u] = v{i}.w;\n        }}\n")),
        )
        .replace(
            "LARGEST_AND_SUMS\n",
            &each(&|i| format!("        {{\n            let ml = n_h * hd + n_h * runs * hd + ((grp * G + {i}u) * runs + run) * 2u;\n            y[ml] = m{}.{};\n            y[ml + 1u] = l{}.{};\n        }}\n", i / 4, c(i), i / 4, c(i))),
        )
}

/// [`attention_part_group`] over a cache held as f16 halves ([`HALVE`]'s): each key and value a vec4 of f16, half the
/// bytes (a step's parts over 15,888 positions read 130 MB a layer at half the card's bandwidth).
pub(in crate::chain) fn attention_part_group_halved(g: usize) -> String {
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
pub(in crate::chain) const HALVE: &str = r#"
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

/// [`attention_part_group`] for QSA's entries ([`QSA_ATTENTION_PART`]'s: a query's kept blocks' positions from `list`,
/// then its tail block's), the grid's third axis the queries, the parts laid out as [`ATTENTION_ROWS_PART`]'s for
/// [`ATTENTION_ROWS_JOIN`]: the same sums as the step's kernel over the same entries, so a query that keeps every
/// block gets the dense attention's bits. `p` as [`QSA_ATTENTION_PART`]'s. (Flash-Next past its dense span: 24 query
/// heads over 2 KV heads of 256, some 2,100 entries a query; a workgroup a head read each key and value twelve times
/// over, 74 us a layer for a step's row and 163 for a check's four.)
pub(in crate::chain) fn qsa_attention_part_group(g: usize) -> String {
    let swap = |s: String, old: &str, new: &str, times: usize| {
        assert_eq!(s.matches(old).count(), times, "QSA's grouped parts: {old}");
        s.replace(old, new)
    };
    let s = attention_part_group(g);
    let s = swap(
        s,
        "@group(0) @binding(2) var<storage, read_write> y: array<f32>;\n@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;\n",
        "@group(0) @binding(2) var<storage, read> list: array<u32>;\n@group(0) @binding(6) var<storage, read_write> y: array<f32>;\n@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;\n// each of the run's entries' position\nvar<workgroup> place: array<u32, 256>;\n",
        1,
    );
    let s = swap(s, "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {", "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {", 1);
    let s = swap(
        s,
        "    let hi = p[0].w;\n    let lo = p[1].x;\n",
        "    let ratio = p[1].x;\n    let keep = p[1].w;\n    let rows = nwg.z;\n    let r = wg.z;\n    let pos = p[0].w + r;\n    let visible = (pos + 1u) / ratio;\n    let count = min(visible, keep);\n    let entries = count * ratio + (pos + 1u - visible * ratio);\n",
        1,
    );
    let s = swap(s, "buf[i] = q4[grp * G * hd4 + i];", "buf[i] = q4[(r * n_h + grp * G) * hd4 + i];", 1);
    let s = swap(
        s,
        "    let start = lo + run * 256u;\n    let end = min(start + 256u, hi);\n    let t = start + li;\n    let live = t < end;\n",
        "    let start = run * 256u;\n    let end = min(start + 256u, entries);\n    let e = start + li;\n    let live = e < end;\n    // the entry's position: a kept block's, or the tail's\n    var t = 0u;\n    if (e < count * ratio) {\n        t = list[r * keep + e / ratio] * ratio + e % ratio;\n    } else {\n        t = visible * ratio + (e - count * ratio);\n    }\n    place[li] = t;\n",
        1,
    );
    let s = swap(s, "    let vb = start * row4 + kvd4 + kh * hd4 + d4;\n", "    let vb = kvd4 + kh * hd4 + d4;\n", 1);
    let s = swap(s, "let v = kv4[vb + i * row4];", "let v = kv4[vb + place[i] * row4];", 1);
    let s = swap(s, "let ml = n_h * hd + n_h * runs * hd + ((grp * G + ", "let ml = rows * n_h * hd + rows * n_h * runs * hd + ((r * n_h + grp * G + ", g);
    swap(s, "let at = n_h * hd + ((grp * G + ", "let at = rows * n_h * hd + ((r * n_h + grp * G + ", g)
}

/// The query heads [`ATTENTION_PART_GROUP`] takes at once of `per_kv` a KV head, `head_dim` wide: all of them, else
/// the largest whole share of them it has room for (Flash-Next's 12 of 256 in two sixes); None where none
/// ([`attention_group_for`]). `spare`: workgroup memory the kernel's variant takes besides (QSA's entries' positions).
pub(in crate::chain) fn attention_group_of(per_kv: usize, head_dim: usize, workgroup_bytes: u32, spare: u32) -> Option<usize> {
    (2..=per_kv.min(8)).rev().find(|g| per_kv % g == 0 && attention_group_for(*g, head_dim, workgroup_bytes.saturating_sub(spare)))
}

/// Whether a step's attention parts take a KV head's group of `g` query heads of `head_dim` at once
/// ([`ATTENTION_PART_GROUP`]): 2 to 8 of them, the head 64, 128 or 256 wide, the shares' sums within the kernel's
/// memory (26.6 KB of the workgroup's, where a device allows it).
pub(in crate::chain) fn attention_group_for(g: usize, head_dim: usize, workgroup_bytes: u32) -> bool {
    matches!(head_dim, 64 | 128 | 256) && (2..=8).contains(&g) && g * (256 - head_dim / 4) <= 1152 && workgroup_bytes >= 26_624
}
