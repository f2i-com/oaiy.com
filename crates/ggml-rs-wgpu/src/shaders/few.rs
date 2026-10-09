//! The kernels for a few rows of `x` (a decode step's one, a check of drafts' few): IQ4_XS's, and the K-quants'
//! register-blocked ones, generated unrolled.

use super::*;

/// The most rows of `x` [`iq4_xs_few`] takes.
pub const IQ4_FEW_MAX: usize = 8;

/// IQ4_XS weights against `m` rows of `x` (1 to [`IQ4_FEW_MAX`]: a decode step's row, a check of drafts' few), in f32:
/// a workgroup 8 weight rows, 32 lanes a row, each lane a 32 of the row in every 32 of them (the reach of one six-bit
/// scale), so a row's lanes read side by side; a 32's nibbles are decoded once for every row of `x` by a read of the
/// type's 16 values in the workgroup's memory (its threads make them, one each), `x` four at a load, and the row's
/// sums are the lanes' behind a barrier. A block is 136 bytes of 256 weights: the scale (f16), each 32's six-bit
/// scale in two fields (two bits in the next 16, four in the 32 after), then 128 bytes of nibbles, a byte the weights
/// `j` and `j + 16` of its 32. The bindings and parameters are the generic kernels' ([`COMMON`]).
///
/// The generic one-row kernel took a weight row a workgroup, a byte at a load, and the values from a constant array
/// indexed as it ran, and a few rows of `x` were that kernel once a row: Flash-Next's 45 IQ4_XS matrices were 2.4 ms
/// of a decode step and 8.2 ms of a check of four rows.
pub(crate) fn iq4_xs_few(m: usize) -> String {
    assert!((1..=IQ4_FEW_MAX).contains(&m), "1 to {IQ4_FEW_MAX} rows");
    let each = |f: &dyn Fn(usize) -> String| (0..m).map(f).collect::<String>();
    let sums = each(&|i| format!("    var a{i} = 0.0;\n"));
    let words: String = (0..4)
        .map(|q| {
            format!(
                "            {{\n                let wq = vec4<u32>(w[qw + {q}u]);\n                let lo = (wq >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(15u);\n                let hi = (wq >> vec4<u32>(4u, 12u, 20u, 28u)) & vec4<u32>(15u);\n                let lv = vec4<f32>(lut[lo.x], lut[lo.y], lut[lo.z], lut[lo.w]);\n                let hv = vec4<f32>(lut[hi.x], lut[hi.y], lut[hi.z], lut[hi.w]);\n{}            }}\n",
                each(&|i| format!("                s{i} += dot(lv, x[{i}u * kq + xo + {q}u]) + dot(hv, x[{i}u * kq + xo + {}u]);\n", 4 + q))
            )
        })
        .collect();
    let zero = each(&|i| format!("            var s{i} = 0.0;\n"));
    let scale = each(&|i| format!("            a{i} += dl * s{i};\n"));
    let store = each(&|i| format!("    red[t * {m}u + {i}u] = a{i};\n"));
    format!(
        r#"
struct Params {{
    k: u32,
    n: u32,
    m: u32,
    row0: u32,
    rows: u32,
    row_bytes: u32,
    _pad0: u32,
    _pad1: u32,
}}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const KVALUES = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
var<workgroup> lut: array<f32, 16>;
// each thread's sums, [weight row][lane][row of x]
var<workgroup> red: array<f32, {red_len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    if (t < 16u) {{
        lut[t] = KVALUES[t];
    }}
    workgroupBarrier();
    let slot = t / 32u;
    let lane = t % 32u;
    let r = (wg.x + wg.y * 65535u) * 8u + slot;
    let live = r < p.rows;
    let kq = p.k / 4u;
{sums}    if (live) {{
        let row = r * (p.row_bytes / 4u);
        let subs = p.k / 32u;
        for (var u = lane; u < subs; u += 32u) {{
            let bw = row + (u / 8u) * 34u;
            let sub = u % 8u;
            let head = w[bw];
            let ls = ((w[bw + 1u] >> (4u * sub)) & 15u) | (((head >> (16u + 2u * sub)) & 3u) << 4u);
            let dl = unpack2x16float(head).x * (f32(ls) - 32.0);
            let qw = bw + 2u + sub * 4u;
            let xo = u * 8u;
{zero}{words}{scale}        }}
    }}
{store}    workgroupBarrier();
    // a row of x's sum by a lane of its own: lane i adds row i's, the lanes in order (one lane adding every row's in
    // turn was what a check's kernel waited for)
    if (live && lane < {m}u) {{
        var s = 0.0;
        for (var l = 0u; l < 32u; l++) {{ s += red[(t - lane + l) * {m}u + lane]; }}
        y[lane * p.n + p.row0 + r] = s;
    }}
}}
"#,
        red_len = 256 * m,
    )
}

/// [`iq4_xs_few`] from `m` rows of `x` as int8 ([`QUANT_Q8`]'s; 2 to [`IQ4_FEW_MAX`]: a check of drafts' rows): laid
/// out the same, a 32's nibbles packed as its values' bytes once (the type's values are int8) and four of them against
/// four of a row's in one `dot4I8Packed`, the 32's sum by the weights' scale and the row's for that 32. A row of `x`
/// then costs a 32 three loads and eight such products, where the f32 kernel's cost it eight loads and 32
/// multiply-adds: Flash-Next's IQ4_XS matrices were 7.8 ms of a check of four rows against 2.5 of a step. The
/// parameters as [`rb_kernel_q8`]'s (`xs_at` where the rows' scales start).
pub(crate) fn iq4_xs_few_q8(m: usize) -> String {
    assert!((1..=IQ4_FEW_MAX).contains(&m), "1 to {IQ4_FEW_MAX} rows");
    let each = |f: &dyn Fn(usize) -> String| (0..m).map(f).collect::<String>();
    let sums = each(&|i| format!("    var a{i} = 0.0;\n"));
    let words: String = (0..4)
        .map(|q| {
            format!(
                "            let wq{q} = vec4<u32>(w[qw + {q}u]);\n            let l{q} = (wq{q} >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(15u);\n            let h{q} = (wq{q} >> vec4<u32>(4u, 12u, 20u, 28u)) & vec4<u32>(15u);\n            let lo{q} = lut[l{q}.x] | (lut[l{q}.y] << 8u) | (lut[l{q}.z] << 16u) | (lut[l{q}.w] << 24u);\n            let hi{q} = lut[h{q}.x] | (lut[h{q}.y] << 8u) | (lut[h{q}.z] << 16u) | (lut[h{q}.w] << 24u);\n"
            )
        })
        .collect();
    let rows = each(&|i| {
        format!(
            "            {{\n                let xa = x8[{i}u * k16 + u * 2u];\n                let xb = x8[{i}u * k16 + u * 2u + 1u];\n                let xd = bitcast<f32>(x8[p.xs_at + {i}u * k32 + u].x);\n                let s = dot4I8Packed(lo0, xa.x) + dot4I8Packed(lo1, xa.y) + dot4I8Packed(lo2, xa.z) + dot4I8Packed(lo3, xa.w) + dot4I8Packed(hi0, xb.x) + dot4I8Packed(hi1, xb.y) + dot4I8Packed(hi2, xb.z) + dot4I8Packed(hi3, xb.w);\n                a{i} += dl * xd * f32(s);\n            }}\n"
        )
    });
    let store = each(&|i| format!("    red[t * {m}u + {i}u] = a{i};\n"));
    format!(
        r#"
struct Params {{
    k: u32,
    n: u32,
    m: u32,
    row0: u32,
    rows: u32,
    row_bytes: u32,
    xs_at: u32,
    _pad1: u32,
}}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x8: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// the type's 16 values, each as its int8's byte
const KVALUES = array<u32, 16>(129u, 152u, 173u, 191u, 207u, 221u, 234u, 246u, 1u, 13u, 25u, 38u, 53u, 69u, 89u, 113u);
var<workgroup> lut: array<u32, 16>;
// each thread's sums, [weight row][lane][row of x]
var<workgroup> red: array<f32, {red_len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    if (t < 16u) {{
        lut[t] = KVALUES[t];
    }}
    workgroupBarrier();
    let slot = t / 32u;
    let lane = t % 32u;
    let r = (wg.x + wg.y * 65535u) * 8u + slot;
    let live = r < p.rows;
    let k16 = p.k / 16u;
    let k32 = p.k / 32u;
{sums}    if (live) {{
        let row = r * (p.row_bytes / 4u);
        for (var u = lane; u < k32; u += 32u) {{
            let bw = row + (u / 8u) * 34u;
            let sub = u % 8u;
            let head = w[bw];
            let ls = ((w[bw + 1u] >> (4u * sub)) & 15u) | (((head >> (16u + 2u * sub)) & 3u) << 4u);
            let dl = unpack2x16float(head).x * (f32(ls) - 32.0);
            let qw = bw + 2u + sub * 4u;
{words}{rows}        }}
    }}
{store}    workgroupBarrier();
    // a row of x's sum by a lane of its own: lane i adds row i's, the lanes in order (one lane adding every row's in
    // turn was what a check's kernel waited for)
    if (live && lane < {m}u) {{
        var s = 0.0;
        for (var l = 0u; l < 32u; l++) {{ s += red[(t - lane + l) * {m}u + lane]; }}
        y[lane * p.n + p.row0 + r] = s;
    }}
}}
"#,
        red_len = 256 * m,
    )
}

/// [`rb_kernel`] for a measurement of its shapes.
#[cfg(test)]
pub(crate) fn rb_kernel_for_test(dtype: GgmlType, r: u32, mr: u32, ks: u32) -> Option<String> {
    rb_kernel_ks(dtype, r, mr, ks)
}

/// The K-quants' matmul for few rows of `x` (a decode step's one, a draft's check's few), its weights read wide and
/// every value it loads used more than once: a lane takes a run of a block's quants (Q3_K, Q4_K, Q5_K: 32 weights
/// from vec4 loads; Q6_K: 16 from words, its 210-byte blocks two bytes off in every other one) for each of `r` weight
/// rows, and the same run of `x` for each of `mr` rows; 4 warps a workgroup, `4 r` weight rows. Generated unrolled,
/// every per-row value its own variable, so they stay in registers.
pub(super) fn rb_kernel(dtype: GgmlType, r: u32, mr: u32) -> Option<String> {
    rb_kernel_ks(dtype, r, mr, 1)
}

/// [`rb_kernel`] with `ks` groups of 4 warps a workgroup, each taking every `ks`th run of 32 lanes' tasks along `k` for
/// the same weight rows (more threads in flight for the same rows), their sums added at the end.
fn rb_kernel_ks(dtype: GgmlType, r: u32, mr: u32, ks: u32) -> Option<String> {
    let (vec4_weights, tasks, block) = match dtype {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K => (true, 8u32, 256u32),
        GgmlType::Q6_K => (false, 16u32, 256u32),
        GgmlType::Q4_0 => (false, 1u32, 32u32),
        _ => return None,
    };
    let mut s = String::new();
    let mut l = |line: &str| {
        s.push_str(line);
        s.push('\n');
    };
    l("struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, _pad0: u32, _pad1: u32, }");
    l(if vec4_weights { "@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;" } else { "@group(0) @binding(0) var<storage, read> w: array<u32>;" });
    l("@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;");
    l("@group(0) @binding(2) var<storage, read_write> y: array<f32>;");
    l("@group(0) @binding(3) var<uniform> p: Params;");
    let wgs = 128 * ks;
    l(&format!("var<workgroup> partial: array<f32, {}>;", wgs * r * mr));
    l(RB_HELPERS);
    if !vec4_weights {
        l(RB_Q6K_HELPERS);
    }
    l(&format!("@compute @workgroup_size({wgs})"));
    l("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {");
    l("    let lane = li & 31u;");
    l("    let group = (li >> 5u) % 4u;");
    l("    let kpart = li >> 7u;");
    l(&format!("    let rbase = (wg.y + wg.z * 65535u) * {}u + group * {r}u;", 4 * r));
    l(&format!("    let m0 = wg.x * {mr}u;"));
    l(&format!("    let mn = min({mr}u, p.m - m0);"));
    l("    let k4 = p.k / 4u;");
    l(&format!("    let blocks = p.k / {block}u;"));
    for i in 0..r {
        l(&format!("    let rr{i} = min(rbase + {i}u, p.rows - 1u);"));
    }
    for m in 0..mr {
        l(&format!("    let xr{m} = min(m0 + {m}u, p.m - 1u) * k4;"));
    }
    for i in 0..r {
        for m in 0..mr {
            l(&format!("    var acc{i}_{m} = 0.0;"));
        }
    }
    l(&format!("    for (var c = lane + kpart * 32u; c < blocks * {tasks}u; c += {}u) {{", 32 * ks));
    match dtype {
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let q5 = dtype == GgmlType::Q5_K;
            let (blk_vec4, q_at) = if q5 { (11, 3) } else { (9, 1) };
            l("        let blk = c / 8u;");
            l("        let j = c % 8u;");
            l("        let pair = j / 2u;");
            l("        let half = j % 2u;");
            l("        let sl2 = 2u * pair;");
            l("        let xa = (blk * 256u + 2u * pair * 32u + half * 16u) / 4u;");
            l("        let xb = xa + 8u;");
            for i in 0..r {
                l(&format!("        let b4_{i} = rr{i} * (p.row_bytes / 16u) + blk * {blk_vec4}u;"));
                l(&format!("        let hd{i} = w4[b4_{i}];"));
                l(&format!("        let qd{i} = w4[b4_{i} + {q_at}u + j];"));
                if q5 {
                    l(&format!("        let qh{i} = w4[b4_{i} + 1u + half];"));
                }
            }
            for m in 0..mr {
                l(&format!("        var slo{m} = 0.0;"));
                l(&format!("        var shi{m} = 0.0;"));
            }
            for i in 0..r {
                for m in 0..mr {
                    l(&format!("        var dl{i}_{m} = 0.0;"));
                    l(&format!("        var dh{i}_{m} = 0.0;"));
                }
            }
            for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                for m in 0..mr {
                    l(&format!("        let a{m}_{wi} = x4[xr{m} + xa + {wi}u];"));
                    l(&format!("        let b{m}_{wi} = x4[xr{m} + xb + {wi}u];"));
                    l(&format!("        slo{m} += a{m}_{wi}.x + a{m}_{wi}.y + a{m}_{wi}.z + a{m}_{wi}.w;"));
                    l(&format!("        shi{m} += b{m}_{wi}.x + b{m}_{wi}.y + b{m}_{wi}.z + b{m}_{wi}.w;"));
                }
                for i in 0..r {
                    l(&format!("        let wd{i}_{wi} = qd{i}.{comp};"));
                    let hl = if q5 { format!(" + high4(qh{i}.{comp}, sl2)") } else { String::new() };
                    let hh = if q5 { format!(" + high4(qh{i}.{comp}, sl2 + 1u)") } else { String::new() };
                    l(&format!("        let lo{i}_{wi} = nib_lo(wd{i}_{wi}){hl};"));
                    l(&format!("        let hi{i}_{wi} = nib_hi(wd{i}_{wi}){hh};"));
                    for m in 0..mr {
                        l(&format!("        dl{i}_{m} += dot(lo{i}_{wi}, a{m}_{wi});"));
                        l(&format!("        dh{i}_{m} += dot(hi{i}_{wi}, b{m}_{wi});"));
                    }
                }
            }
            for i in 0..r {
                l(&format!("        let dm{i} = unpack2x16float(hd{i}.x);"));
                l(&format!("        let s{i}a = scale_min(hd{i}, 2u * pair);"));
                l(&format!("        let s{i}b = scale_min(hd{i}, 2u * pair + 1u);"));
                for m in 0..mr {
                    l(&format!("        acc{i}_{m} += dm{i}.x * s{i}a.x * dl{i}_{m} - dm{i}.y * s{i}a.y * slo{m} + dm{i}.x * s{i}b.x * dh{i}_{m} - dm{i}.y * s{i}b.y * shi{m};"));
                }
            }
        }
        GgmlType::Q3_K => {
            l("        let blk = c / 8u;");
            l("        let t = c % 8u;");
            l("        let h = t / 4u;");
            l("        let g = (t / 2u) % 2u;");
            l("        let j0 = 2u * (t % 2u);");
            l("        let xa = (blk * 256u + h * 128u + j0 * 32u + g * 16u) / 4u;");
            for i in 0..r {
                l(&format!("        let b4_{i} = rr{i} * (p.row_bytes / 16u) + blk * 7u;"));
                l(&format!("        let hm{i} = w4[b4_{i} + g];"));
                l(&format!("        let qs{i} = w4[b4_{i} + 2u + 2u * h + g];"));
                l(&format!("        let sd{i} = w4[b4_{i} + 6u];"));
            }
            for m in 0..mr {
                l(&format!("        var sx0_{m} = 0.0;"));
                l(&format!("        var sx1_{m} = 0.0;"));
            }
            for i in 0..r {
                for m in 0..mr {
                    l(&format!("        var dq0_{i}_{m} = 0.0;"));
                    l(&format!("        var dq1_{i}_{m} = 0.0;"));
                }
            }
            for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                for m in 0..mr {
                    l(&format!("        let a{m}_{wi} = x4[xr{m} + xa + {wi}u];"));
                    l(&format!("        let b{m}_{wi} = x4[xr{m} + xa + 8u + {wi}u];"));
                    l(&format!("        sx0_{m} += a{m}_{wi}.x + a{m}_{wi}.y + a{m}_{wi}.z + a{m}_{wi}.w;"));
                    l(&format!("        sx1_{m} += b{m}_{wi}.x + b{m}_{wi}.y + b{m}_{wi}.z + b{m}_{wi}.w;"));
                }
                for i in 0..r {
                    l(&format!("        let v0_{i}_{wi} = q3_pair(qs{i}.{comp}, hm{i}.{comp}, j0, h);"));
                    l(&format!("        let v1_{i}_{wi} = q3_pair(qs{i}.{comp}, hm{i}.{comp}, j0 + 1u, h);"));
                    for m in 0..mr {
                        l(&format!("        dq0_{i}_{m} += dot(v0_{i}_{wi}, a{m}_{wi});"));
                        l(&format!("        dq1_{i}_{m} += dot(v1_{i}_{wi}, b{m}_{wi});"));
                    }
                }
            }
            for i in 0..r {
                l(&format!("        let d{i} = unpack2x16float(sd{i}.w & 0xffffu).x;"));
                l(&format!("        let sc{i} = q3_scales(sd{i}, h, g, j0);"));
                for m in 0..mr {
                    l(&format!("        acc{i}_{m} += d{i} * (sc{i}.x * (dq0_{i}_{m} - 4.0 * sx0_{m}) + sc{i}.y * (dq1_{i}_{m} - 4.0 * sx1_{m}));"));
                }
            }
        }
        GgmlType::Q4_0 => {
            // a block of 32 a task: its scale's word, then four words of nibbles (low: weights 0..16, high: 16..32)
            l("        let blk = c;");
            for i in 0..r {
                l(&format!("        let bw{i} = (rr{i} * p.row_bytes + blk * 20u) / 4u;"));
            }
            for m in 0..mr {
                l(&format!("        var sx{m} = 0.0;"));
                for wi in 0..8 {
                    l(&format!("        let a{m}_{wi} = x4[xr{m} + blk * 8u + {wi}u];"));
                    l(&format!("        sx{m} += a{m}_{wi}.x + a{m}_{wi}.y + a{m}_{wi}.z + a{m}_{wi}.w;"));
                }
            }
            for i in 0..r {
                l(&format!("        let d{i} = unpack2x16float(w[bw{i}] & 0xffffu).x;"));
                for m in 0..mr {
                    l(&format!("        var dq{i}_{m} = 0.0;"));
                }
                for wi in 0..4 {
                    l(&format!("        let qw{i}_{wi} = w[bw{i} + {}u];", wi + 1));
                    l(&format!("        let lo{i}_{wi} = nib_lo(qw{i}_{wi});"));
                    l(&format!("        let hi{i}_{wi} = nib_hi(qw{i}_{wi});"));
                    for m in 0..mr {
                        l(&format!("        dq{i}_{m} += dot(lo{i}_{wi}, a{m}_{wi}) + dot(hi{i}_{wi}, a{m}_{});", wi + 4));
                    }
                }
                for m in 0..mr {
                    l(&format!("        acc{i}_{m} += d{i} * (dq{i}_{m} - 8.0 * sx{m});"));
                }
            }
        }
        _ => {
            // Q6_K
            l("        let blk = c / 16u;");
            l("        let sub = (c % 16u) / 2u;");
            l("        let hf = c % 2u;");
            l("        let h = sub / 4u;");
            l("        let qd = sub % 4u;");
            l("        let lshift = select(0u, 4u, qd >= 2u);");
            l("        let hshift = 2u * qd;");
            l("        let xb = (blk * 256u + sub * 32u + hf * 16u) / 4u;");
            for i in 0..r {
                l(&format!("        let bb{i} = rr{i} * p.row_bytes + blk * 210u;"));
                l(&format!("        let ql{i} = bb{i} + h * 64u + (qd & 1u) * 32u + hf * 16u;"));
                l(&format!("        let qh{i} = bb{i} + 128u + h * 32u + hf * 16u;"));
            }
            for m in 0..mr {
                l(&format!("        var sx{m} = 0.0;"));
            }
            for i in 0..r {
                for m in 0..mr {
                    l(&format!("        var dq{i}_{m} = 0.0;"));
                }
            }
            for wi in 0..4 {
                for m in 0..mr {
                    l(&format!("        let a{m}_{wi} = x4[xr{m} + xb + {wi}u];"));
                    l(&format!("        sx{m} += a{m}_{wi}.x + a{m}_{wi}.y + a{m}_{wi}.z + a{m}_{wi}.w;"));
                }
                for i in 0..r {
                    l(&format!("        let q{i}_{wi} = q6_four(word_at(ql{i} + {}u), word_at(qh{i} + {}u), lshift, hshift);", wi * 4, wi * 4));
                    for m in 0..mr {
                        l(&format!("        dq{i}_{m} += dot(q{i}_{wi}, a{m}_{wi});"));
                    }
                }
            }
            for i in 0..r {
                l(&format!("        let d{i} = unpack2x16float(byte(bb{i} + 208u) | (byte(bb{i} + 209u) << 8u)).x;"));
                l(&format!("        let scb{i} = byte(bb{i} + 192u + h * 8u + 2u * qd + hf);"));
                l(&format!("        let sc{i} = f32(i32(scb{i}) - select(0, 256, scb{i} >= 128u));"));
                for m in 0..mr {
                    l(&format!("        acc{i}_{m} += d{i} * sc{i} * (dq{i}_{m} - 32.0 * sx{m});"));
                }
            }
        }
    }
    l("    }");
    for i in 0..r {
        for m in 0..mr {
            l(&format!("    partial[{}u * {wgs}u + li] = acc{i}_{m};", i * mr + m));
        }
    }
    l("    workgroupBarrier();");
    l("    for (var st = 16u; st > 0u; st /= 2u) {");
    l("        if (lane < st) {");
    l(&format!("            for (var v = 0u; v < {}u; v++) {{ partial[v * {wgs}u + li] += partial[v * {wgs}u + li + st]; }}", r * mr));
    l("        }");
    l("        workgroupBarrier();");
    l("    }");
    l("    if (lane == 0u && kpart == 0u) {");
    for i in 0..r {
        l(&format!("        if (rbase + {i}u < p.rows) {{"));
        let sum: Vec<String> = (0..ks).map(|kp| format!("partial[({i}u * {mr}u + mm) * {wgs}u + {}u + group * 32u]", kp * 128)).collect();
        l(&format!("            for (var mm = 0u; mm < mn; mm++) {{ y[(m0 + mm) * p.n + p.row0 + rbase + {i}u] = {}; }}", sum.join(" + ")));
        l("        }");
    }
    l("    }");
    l("}");
    Some(s)
}

/// The helpers of [`rb_kernel`]'s kernels, each type's decode of four weights.
pub(super) const RB_HELPERS: &str = r#"
fn nib_lo(v: u32) -> vec4<f32> {
    return vec4<f32>(f32(v & 15u), f32((v >> 8u) & 15u), f32((v >> 16u) & 15u), f32((v >> 24u) & 15u));
}
fn nib_hi(v: u32) -> vec4<f32> {
    return vec4<f32>(f32((v >> 4u) & 15u), f32((v >> 12u) & 15u), f32((v >> 20u) & 15u), f32((v >> 28u) & 15u));
}
// The four bytes' bit `s` of `v` (a byte each), as 0 or 16.
fn high4(v: u32, s: u32) -> vec4<f32> {
    return 16.0 * vec4<f32>(f32((v >> s) & 1u), f32((v >> (s + 8u)) & 1u), f32((v >> (s + 16u)) & 1u), f32((v >> (s + 24u)) & 1u));
}
// Byte `b` (0..12) of a block's scales, the header's last three words.
fn sbyte(h: vec4<u32>, b: u32) -> u32 {
    let wd = select(select(h.w, h.z, b < 8u), h.y, b < 4u);
    return (wd >> (8u * (b % 4u))) & 255u;
}
// Sub-block `j`'s 6-bit scale and minimum, as ggml's get_scale_min_k4.
fn scale_min(h: vec4<u32>, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(sbyte(h, j) & 63u), f32(sbyte(h, j + 4u) & 63u));
    }
    let sc = (sbyte(h, j + 4u) & 15u) | ((sbyte(h, j - 4u) >> 6u) << 4u);
    let mn = (sbyte(h, j + 4u) >> 4u) | ((sbyte(h, j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(mn));
}
// Q3_K: four weights of run `j` (their 2 low bits at shift 2j, their high bit j + 4h), each plus 4.
fn q3_pair(qw: u32, hw: u32, j: u32, h: u32) -> vec4<f32> {
    let v = ((qw >> (2u * j)) & 0x03030303u) | (((hw >> (j + 4u * h)) & 0x01010101u) << 2u);
    return vec4<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}
// Q3_K: the scales of runs j0 and j0 + 1 of half h, run g (ggml's kmask unpacking), less 32.
fn q3_scales(s: vec4<u32>, h: u32, g: u32, j0: u32) -> vec2<f32> {
    let sa = ((s.x >> (4u * h)) & 0x0f0f0f0fu) | (((s.z >> (4u * h)) & 0x03030303u) << 4u);
    let sb = ((s.y >> (4u * h)) & 0x0f0f0f0fu) | (((s.z >> (4u * h + 2u)) & 0x03030303u) << 4u);
    let sw = select(sa, sb, j0 == 2u);
    let gs = 8u * g;
    return vec2<f32>(f32((sw >> gs) & 255u) - 32.0, f32((sw >> (gs + 16u)) & 255u) - 32.0);
}
"#;

/// Q6_K's word helpers (its blocks are words, not vec4s), for [`rb_kernel`].
pub(super) const RB_Q6K_HELPERS: &str = r#"
fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
// The four bytes at `o`, an even offset (a word, or the halves of two).
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}
// Four 6-bit weights: low bits from `lo`'s bytes at `ls`, high bits from `hi`'s at `hs`.
fn q6_four(lo: u32, hi: u32, ls: u32, hs: u32) -> vec4<f32> {
    let l = vec4<u32>(lo & 255u, (lo >> 8u) & 255u, (lo >> 16u) & 255u, lo >> 24u);
    let h = vec4<u32>(hi & 255u, (hi >> 8u) & 255u, (hi >> 16u) & 255u, hi >> 24u);
    return vec4<f32>(((l >> vec4<u32>(ls)) & vec4<u32>(15u)) | (((h >> vec4<u32>(hs)) & vec4<u32>(3u)) << vec4<u32>(4u)));
}
"#;
