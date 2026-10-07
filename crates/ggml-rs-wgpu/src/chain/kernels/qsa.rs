//! QSA's kernels (Qwen3.8-Flash-Next's sparse attention): the pooled block keys, their scores, the selection, the mask and the attention over what was kept.

/// [`QSA_SCORES`] for `heads` index heads (1 to 8) of a width a multiple of 4: each block's pooled key read once, a
/// vec4 at a time, for every head's sum (a sum a load a head: a prompt's chunk of 512 over 3,584 blocks, 4 heads of
/// 128, 8 ms a layer).
pub(in crate::chain) fn qsa_scores4(heads: usize) -> String {
    let each = |f: &dyn Fn(usize) -> String| (0..heads).map(f).collect::<String>();
    let total = (0..heads).map(|h| format!("max(a{h}.x + a{h}.y + a{h}.z + a{h}.w, 0.0)")).collect::<Vec<_>>().join(" + ");
    QSA_SCORES4
        .replace("HEADS", &heads.to_string())
        .replace("SUMS\n", &each(&|h| format!("        var a{h} = vec4<f32>(0.0);\n")))
        .replace("STEPS\n", &each(&|h| format!("            a{h} += qs[{h}u * d4 + i] * v;\n")))
        .replace("TOTAL", &total)
}

pub(in crate::chain) const QSA_SCORES4: &str = r#"
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
pub(in crate::chain) const QSA_MASK: &str = r#"
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

/// QSA's pooled block keys: `pooled[b, i]` the mean of `raw` rows `b ratio ..`, each `/ ratio` added in turn (as
/// `Backend::qsa_pool`), a workgroup a block. `p[0]`: blocks, ratio, d (at most 256).
pub(in crate::chain) const QSA_POOL: &str = r#"
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
pub(in crate::chain) const QSA_SCORES: &str = r#"
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
pub(in crate::chain) const QSA_SELECT: &str = r#"
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
pub(in crate::chain) const QSA_ATTENTION_PART: &str = r#"
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
