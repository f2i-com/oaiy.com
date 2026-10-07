//! A prompt's (and a diffusion's) attention on the tensor cores: the causal kernels, the full ones and QSA's masked ones, each made for its head's width.

/// A prompt's causal attention on the tensor cores ([`attention_coop`]'s kernel before its head's width is put in), a
/// workgroup of 4 subgroups a (head `h`, 32 queries), the keys 32 at a time: each block's scores `q . k` (f16 into
/// f32), a subgroup's 16 queries by 16 keys, stored for the softmax; a first pass over the blocks finds each query's
/// largest score and its sum (a thread a query's 8 keys of a block, its 4 threads' joined), the second its weights
/// (as f16) and the values they weigh, a subgroup's 16 queries by half the head. `q16`: the queries `[rows, n_h, hd]`
/// as f16 (padded to 32 rows), `kv16` the cache's rows as f16 (padded to 32 positions); `y` the output `[rows, n_h,
/// hd]` (the padding's rows past it, in what is scratch).
pub(in crate::chain) const ATTENTION_COOP: &str = r#"enable f16;
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
pub(in crate::chain) const ATTENTION_COOP_SCORES: &str = r#"        {
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
pub(in crate::chain) fn attention_coop(hd: usize) -> String {
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
pub(in crate::chain) const ATTENTION_COOP_WIDE: &str = r#"enable f16;
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
pub(in crate::chain) const ATTENTION_COOP_WIDE_SCORES: &str = r#"        {
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
pub(in crate::chain) fn attention_coop_wide(hd: usize, full: bool) -> String {
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
pub(in crate::chain) const ATTENTION_COOP_ONE: &str = r#"enable f16;
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
pub(in crate::chain) fn attention_coop_one(hd: usize, full: bool) -> String {
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
pub(in crate::chain) fn attention_coop_full(hd: usize) -> String {
    let causal = "    let qpos = p.past + q0 + tr;\n    // the blocks the last query sees\n    let hi = min(p.kv_len, p.past + q0 + 32u);\n";
    let src = attention_coop(hd);
    assert_eq!(src.matches(causal).count(), 1, "the causal limit");
    src.replace(causal, "    // every position, the last query's and the first's alike\n    let qpos = p.kv_len;\n    let hi = p.kv_len;\n")
}

/// [`ATTENTION_COOP`] for QSA's queries past its dense span ([`ChainRecorder::qsa_attention`] of a prompt's rows): every
/// position up to the query's own on the tensor cores, those of a block the query did not keep left out (its blocks
/// of `p._pad0` positions, a bit each in `mask`, `p._pad1` words a query: its incomplete tail block's always in).
pub(in crate::chain) fn attention_coop_masked(hd: usize) -> String {
    attention_coop(hd)
        .replace("@group(0) @binding(2) var<storage, read_write> y: array<f32>;", "@group(0) @binding(2) var<storage, read> mask: array<u32>;\n@group(0) @binding(6) var<storage, read_write> y: array<f32>;")
        .replace("@group(0) @binding(3) var<uniform> p: Params;", "@group(0) @binding(8) var<uniform> p: Params;\n\n// whether query `row` (at `qpos`) attends to position `kp`: its tail block's, or a block it kept\nfn kept(row: u32, qpos: u32, kp: u32) -> bool {\n    let b = kp / p._pad0;\n    if (b >= (qpos + 1u) / p._pad0) { return true; }\n    return ((mask[row * p._pad1 + b / 32u] >> (b % 32u)) & 1u) == 1u;\n}")
        .replace("if (kp <= qpos && kp < p.kv_len) {", "if (kp <= qpos && kp < p.kv_len && kept(q0 + tr, qpos, kp)) {")
}

/// [`attention_coop_masked`] in one pass over the keys, 128 of them a block ([`attention_coop_one`] with the mask: a
/// query's keys of a block it did not keep count for nothing, as those past it; its reference is its first block
/// with a key it attends to). A prompt's chunk of 512 at 14,336 positions (24 heads of 256 over 2 KV heads): its
/// twelve layers' 81 ms with the scores twice and the keys 32 a block.
pub(in crate::chain) fn attention_coop_one_masked(hd: usize) -> String {
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
