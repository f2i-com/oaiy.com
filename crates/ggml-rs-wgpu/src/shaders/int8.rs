//! The int8 kernels: rows of `x` quantized to int8 by 32s, and the K-quants' matmuls against them (a few rows'
//! register-blocked, a prompt's tiled).

use super::*;

/// Rows of `x` as int8 for the int8 kernels ([`rb_kernel_q8`]): each 32 of a row by its own scale (its largest
/// magnitude over 127), rounded, four to a word; then each block's scale and its two halves' sums (as `d * sum`), a
/// vec4 a block (`d`, the first 16's, the last 16's, 0) from vec4 `p[0].z` on. A thread a block. `p[0]`: k, rows,
/// where the scales start (in vec4s).
pub const QUANT_Q8: &str = r#"
@group(0) @binding(0) var<storage, read> unused: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = p[0].x;
    let rows = p[0].y;
    let xs_at = p[0].z;
    let b = id.x + id.y * 65535u * 256u;
    if (b >= rows * (k / 32u)) {
        return;
    }
    let base = b * 32u;
    var amax = 0.0;
    for (var i = 0u; i < 32u; i++) {
        amax = max(amax, abs(x[base + i]));
    }
    let d = amax / 127.0;
    let inv = select(0.0, 1.0 / d, d > 0.0);
    var s0 = 0;
    var s1 = 0;
    for (var w = 0u; w < 8u; w++) {
        var word = 0u;
        for (var e = 0u; e < 4u; e++) {
            let qv = clamp(i32(round(x[base + w * 4u + e] * inv)), -127, 127);
            if (w < 4u) {
                s0 += qv;
            } else {
                s1 += qv;
            }
            word |= (bitcast<u32>(qv) & 0xffu) << (8u * e);
        }
        q[b * 8u + w] = word;
    }
    let at = (xs_at + b) * 4u;
    q[at] = bitcast<u32>(d);
    q[at + 1u] = bitcast<u32>(d * f32(s0));
    q[at + 2u] = bitcast<u32>(d * f32(s1));
    q[at + 3u] = 0u;
}
"#;

/// Elements (4-byte) a buffer of [`QUANT_Q8`]'s output takes for `rows` rows of `k`: the packed rows, then the blocks'
/// scales and sums; and where the scales start, in vec4s.
pub fn q8_len(rows: usize, k: usize) -> (usize, usize) {
    (rows * k / 4 + rows * k / 32 * 4, rows * k / 16)
}

/// [`rb_kernel`]'s matmul from rows of `x` as int8 ([`QUANT_Q8`]'s): a weight run's values packed as bytes and each 4 of
/// them against 4 of a row's in one `dot4I8Packed`, a 32-block's scale and its halves' sums for the offsets and the
/// minimums; the same lanes, runs and rows. For several rows of `x` (a check of drafts, a short prompt's chunk) it
/// costs little more than for one, where the f32 kernel's multiply-adds a weight grow with them.
pub fn rb_kernel_q8(dtype: GgmlType, r: u32, mr: u32) -> Option<String> {
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
    l("struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, xs_at: u32, _pad1: u32, }");
    l(if vec4_weights { "@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;" } else { "@group(0) @binding(0) var<storage, read> w: array<u32>;" });
    l("@group(0) @binding(1) var<storage, read> x8: array<vec4<u32>>;");
    l("@group(0) @binding(2) var<storage, read_write> y: array<f32>;");
    l("@group(0) @binding(3) var<uniform> p: Params;");
    l(&format!("var<workgroup> partial: array<f32, {}>;", 128 * r * mr));
    l(RB_HELPERS);
    if !vec4_weights {
        l(RB_Q6K_HELPERS);
    }
    l("// Q3_K: four weights of run `j` (their 2 low bits at shift 2j, their high bit j + 4h) as bytes 0..7 (each plus 4).");
    l("fn q3_bytes(qw: u32, hw: u32, j: u32, h: u32) -> u32 {");
    l("    return ((qw >> (2u * j)) & 0x03030303u) | (((hw >> (j + 4u * h)) & 0x01010101u) << 2u);");
    l("}");
    l("@compute @workgroup_size(128)");
    l("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {");
    l("    let lane = li & 31u;");
    l(&format!("    let rbase = (wg.y + wg.z * 65535u) * {}u + (li >> 5u) * {r}u;", 4 * r));
    l(&format!("    let m0 = wg.x * {mr}u;"));
    l(&format!("    let mn = min({mr}u, p.m - m0);"));
    l("    let k16 = p.k / 16u;");
    l("    let k32 = p.k / 32u;");
    l(&format!("    let blocks = p.k / {block}u;"));
    for i in 0..r {
        l(&format!("    let rr{i} = min(rbase + {i}u, p.rows - 1u);"));
    }
    for m in 0..mr {
        l(&format!("    let xr{m} = min(m0 + {m}u, p.m - 1u);"));
    }
    for i in 0..r {
        for m in 0..mr {
            l(&format!("    var acc{i}_{m} = 0.0;"));
        }
    }
    l(&format!("    for (var c = lane; c < blocks * {tasks}u; c += 32u) {{"));
    let comps = ["x", "y", "z", "w"];
    // a row's 16 values of a 32-block (its half `hf`), the block's scale and that half's sum
    let x_half = |l: &mut dyn FnMut(&str), m: u32, name: &str, block: &str, hf: &str| {
        l(&format!("        let {name}{m} = x8[xr{m} * k16 + ({block}) * 2u + {hf}];"));
        l(&format!("        let s{name}{m} = bitcast<vec4<f32>>(x8[p.xs_at + xr{m} * k32 + ({block})]);"));
        l(&format!("        let h{name}{m} = select(s{name}{m}.y, s{name}{m}.z, {hf} == 1u);"));
    };
    match dtype {
        GgmlType::Q3_K => {
            l("        let blk = c / 8u;");
            l("        let t = c % 8u;");
            l("        let h = t / 4u;");
            l("        let g = (t / 2u) % 2u;");
            l("        let j0 = 2u * (t % 2u);");
            l("        let b0 = blk * 8u + h * 4u + j0;");
            for m in 0..mr {
                x_half(&mut l, m, "xa", "b0", "g");
                x_half(&mut l, m, "xb", "b0 + 1u", "g");
            }
            for i in 0..r {
                l(&format!("        let b4_{i} = rr{i} * (p.row_bytes / 16u) + blk * 7u;"));
                l(&format!("        let hm{i} = w4[b4_{i} + g];"));
                l(&format!("        let qs{i} = w4[b4_{i} + 2u + 2u * h + g];"));
                l(&format!("        let sd{i} = w4[b4_{i} + 6u];"));
                for (wi, comp) in comps.iter().enumerate() {
                    l(&format!("        let v0_{i}_{wi} = q3_bytes(qs{i}.{comp}, hm{i}.{comp}, j0, h);"));
                    l(&format!("        let v1_{i}_{wi} = q3_bytes(qs{i}.{comp}, hm{i}.{comp}, j0 + 1u, h);"));
                }
                l(&format!("        let d{i} = unpack2x16float(sd{i}.w & 0xffffu).x;"));
                l(&format!("        let sc{i} = q3_scales(sd{i}, h, g, j0);"));
                for m in 0..mr {
                    let d0: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(v0_{i}_{wi}, xa{m}.{comp})")).collect();
                    let d1: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(v1_{i}_{wi}, xb{m}.{comp})")).collect();
                    l(&format!("        let p0_{i}_{m} = {};", d0.join(" + ")));
                    l(&format!("        let p1_{i}_{m} = {};", d1.join(" + ")));
                    l(&format!("        acc{i}_{m} += d{i} * (sc{i}.x * (sxa{m}.x * f32(p0_{i}_{m}) - 4.0 * hxa{m}) + sc{i}.y * (sxb{m}.x * f32(p1_{i}_{m}) - 4.0 * hxb{m}));"));
                }
            }
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let q5 = dtype == GgmlType::Q5_K;
            let (blk_vec4, q_at) = if q5 { (11, 3) } else { (9, 1) };
            l("        let blk = c / 8u;");
            l("        let j = c % 8u;");
            l("        let pair = j / 2u;");
            l("        let half = j % 2u;");
            l("        let sl2 = 2u * pair;");
            l("        let ba = blk * 8u + 2u * pair;");
            for m in 0..mr {
                x_half(&mut l, m, "xa", "ba", "half");
                x_half(&mut l, m, "xb", "ba + 1u", "half");
            }
            for i in 0..r {
                l(&format!("        let b4_{i} = rr{i} * (p.row_bytes / 16u) + blk * {blk_vec4}u;"));
                l(&format!("        let hd{i} = w4[b4_{i}];"));
                l(&format!("        let qd{i} = w4[b4_{i} + {q_at}u + j];"));
                if q5 {
                    l(&format!("        let qh{i} = w4[b4_{i} + 1u + half];"));
                }
                for (wi, comp) in comps.iter().enumerate() {
                    let (hl, hh) = if q5 {
                        (format!(" | (((qh{i}.{comp} >> sl2) & 0x01010101u) << 4u)"), format!(" | (((qh{i}.{comp} >> (sl2 + 1u)) & 0x01010101u) << 4u)"))
                    } else {
                        (String::new(), String::new())
                    };
                    l(&format!("        let lo{i}_{wi} = (qd{i}.{comp} & 0x0f0f0f0fu){hl};"));
                    l(&format!("        let hi{i}_{wi} = ((qd{i}.{comp} >> 4u) & 0x0f0f0f0fu){hh};"));
                }
                l(&format!("        let dm{i} = unpack2x16float(hd{i}.x);"));
                l(&format!("        let s{i}a = scale_min(hd{i}, 2u * pair);"));
                l(&format!("        let s{i}b = scale_min(hd{i}, 2u * pair + 1u);"));
                for m in 0..mr {
                    let dl: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(lo{i}_{wi}, xa{m}.{comp})")).collect();
                    let dh: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(hi{i}_{wi}, xb{m}.{comp})")).collect();
                    l(&format!("        let pl{i}_{m} = {};", dl.join(" + ")));
                    l(&format!("        let ph{i}_{m} = {};", dh.join(" + ")));
                    l(&format!("        acc{i}_{m} += dm{i}.x * s{i}a.x * sxa{m}.x * f32(pl{i}_{m}) - dm{i}.y * s{i}a.y * hxa{m} + dm{i}.x * s{i}b.x * sxb{m}.x * f32(ph{i}_{m}) - dm{i}.y * s{i}b.y * hxb{m};"));
                }
            }
        }
        GgmlType::Q4_0 => {
            // a block of 32 a task, its low nibbles against the row's first 16 values and its high ones the last 16
            l("        let blk = c;");
            for m in 0..mr {
                l(&format!("        let xlo{m} = x8[xr{m} * k16 + blk * 2u];"));
                l(&format!("        let xhi{m} = x8[xr{m} * k16 + blk * 2u + 1u];"));
                l(&format!("        let sx{m} = bitcast<vec4<f32>>(x8[p.xs_at + xr{m} * k32 + blk]);"));
            }
            for i in 0..r {
                l(&format!("        let bw{i} = (rr{i} * p.row_bytes + blk * 20u) / 4u;"));
                l(&format!("        let d{i} = unpack2x16float(w[bw{i}] & 0xffffu).x;"));
                for wi in 0..4 {
                    l(&format!("        let qw{i}_{wi} = w[bw{i} + {}u];", wi + 1));
                    l(&format!("        let lo{i}_{wi} = qw{i}_{wi} & 0x0f0f0f0fu;"));
                    l(&format!("        let hi{i}_{wi} = (qw{i}_{wi} >> 4u) & 0x0f0f0f0fu;"));
                }
                for m in 0..mr {
                    let dl: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(lo{i}_{wi}, xlo{m}.{comp})")).collect();
                    let dh: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(hi{i}_{wi}, xhi{m}.{comp})")).collect();
                    l(&format!("        let pq{i}_{m} = {} + {};", dl.join(" + "), dh.join(" + ")));
                    l(&format!("        acc{i}_{m} += d{i} * (sx{m}.x * f32(pq{i}_{m}) - 8.0 * (sx{m}.y + sx{m}.z));"));
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
            l("        let bx = blk * 8u + sub;");
            for m in 0..mr {
                x_half(&mut l, m, "xa", "bx", "hf");
            }
            for i in 0..r {
                l(&format!("        let bb{i} = rr{i} * p.row_bytes + blk * 210u;"));
                l(&format!("        let ql{i} = bb{i} + h * 64u + (qd & 1u) * 32u + hf * 16u;"));
                l(&format!("        let qh{i} = bb{i} + 128u + h * 32u + hf * 16u;"));
                for wi in 0..4 {
                    l(&format!("        let q{i}_{wi} = ((word_at(ql{i} + {o}u) >> lshift) & 0x0f0f0f0fu) | (((word_at(qh{i} + {o}u) >> hshift) & 0x03030303u) << 4u);", o = wi * 4));
                }
                l(&format!("        let d{i} = unpack2x16float(byte(bb{i} + 208u) | (byte(bb{i} + 209u) << 8u)).x;"));
                l(&format!("        let scb{i} = byte(bb{i} + 192u + h * 8u + 2u * qd + hf);"));
                l(&format!("        let sc{i} = f32(i32(scb{i}) - select(0, 256, scb{i} >= 128u));"));
                for m in 0..mr {
                    let dq: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(q{i}_{wi}, xa{m}.{comp})")).collect();
                    l(&format!("        let pq{i}_{m} = {};", dq.join(" + ")));
                    l(&format!("        acc{i}_{m} += d{i} * sc{i} * (sxa{m}.x * f32(pq{i}_{m}) - 32.0 * hxa{m});"));
                }
            }
        }
    }
    l("    }");
    for i in 0..r {
        for m in 0..mr {
            l(&format!("    partial[{}u * 128u + li] = acc{i}_{m};", i * mr + m));
        }
    }
    l("    workgroupBarrier();");
    l("    for (var st = 16u; st > 0u; st /= 2u) {");
    l("        if (lane < st) {");
    l(&format!("            for (var v = 0u; v < {}u; v++) {{ partial[v * 128u + li] += partial[v * 128u + li + st]; }}", r * mr));
    l("        }");
    l("        workgroupBarrier();");
    l("    }");
    l("    if (lane == 0u) {");
    for i in 0..r {
        l(&format!("        if (rbase + {i}u < p.rows) {{"));
        l(&format!("            for (var mm = 0u; mm < mn; mm++) {{ y[(m0 + mm) * p.n + p.row0 + rbase + {i}u] = partial[({i}u * {mr}u + mm) * 128u + li]; }}"));
        l("        }");
    }
    l("    }");
    l("}");
    Some(s)
}

/// Weight rows a workgroup of [`tiled_q8`] takes.
pub const TQ8_ROWS: u32 = 128;
/// Rows of `x` (a prompt's tokens) a workgroup of [`tiled_q8`] takes.
pub const TQ8_TOKENS: u32 = 64;

/// [`tiled_q8`]'s kernel before its loops are unrolled in.
const TQ8_HEAD: &str = r#"struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, xs_at: u32, _pad1: u32, }
@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x8: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
// the step's weights: [word][a group of 4 rows] a vec4 (32 groups), and a row's (scale a, scale b, offset a, offset b)
var<workgroup> wq: array<vec4<u32>, 256>;
var<workgroup> ws: array<vec4<f32>, 128>;
// the step's tokens: [word][a group of 4 tokens] a vec4 (16 groups), and a token's (d, its halves' sums)
var<workgroup> xq: array<vec4<u32>, 128>;
var<workgroup> xs: array<vec4<f32>, 64>;

fn q3_bytes(qw: u32, hw: u32, j: u32, h: u32) -> u32 {
    return ((qw >> (2u * j)) & 0x03030303u) | (((hw >> (j + 4u * h)) & 0x01010101u) << 2u);
}

// Q3_K: the scales of run j's two halves of 16 (sub-blocks 8 h + 2 j and the next), less 32.
fn q3_scales32(s: vec4<u32>, h: u32, j: u32) -> vec2<f32> {
    let sa = ((s.x >> (4u * h)) & 0x0f0f0f0fu) | (((s.z >> (4u * h)) & 0x03030303u) << 4u);
    let sb = ((s.y >> (4u * h)) & 0x0f0f0f0fu) | (((s.z >> (4u * h + 2u)) & 0x03030303u) << 4u);
    let sw = select(sa, sb, j >= 2u);
    let bs = 16u * (j % 2u);
    return vec2<f32>(f32((sw >> bs) & 255u) - 32.0, f32((sw >> (bs + 8u)) & 255u) - 32.0);
}

// an int under 2^22 as f32: its bits added to 1.5 * 2^23's, less that
fn exact(i: i32) -> f32 {
    return bitcast<f32>(bitcast<u32>(i) + 0x4b400000u) - 12582912.0;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 128u;
    let t0 = wg.y * 64u;
    let tx = li % 16u;
    let ty = li / 16u;
    let k16 = p.k / 16u;
    let k32 = p.k / 32u;
    // what this thread loads: weight row lr's words from 4 part; and token lt's half lh (or a token's scales)
    let lr = li / 2u;
    let part = li % 2u;
    let rr = min(r0 + lr, p.rows - 1u);
    let lt = (li % 128u) / 2u;
    let lh = li % 2u;
    let tok = min(t0 + lt, p.m - 1u);
"#;

/// [`tiled_q8`]'s k loop: a step's tile decoded and loaded, its products, then its sums out.
const TQ8_LOOP: &str = r#"    for (var b = 0u; b < k32; b++) {
        let blk = b / 8u;
        let q = b % 8u;
        let h = q / 4u;
        let j = q % 4u;
        let b4 = rr * (p.row_bytes / 16u) + blk * 7u;
        let hm = w4[b4 + part];
        let qs = w4[b4 + 2u + 2u * h + part];
        let g = lr / 4u;
        let c = lr % 4u;
        wq[(part * 4u) * 32u + g][c] = q3_bytes(qs.x, hm.x, j, h);
        wq[(part * 4u + 1u) * 32u + g][c] = q3_bytes(qs.y, hm.y, j, h);
        wq[(part * 4u + 2u) * 32u + g][c] = q3_bytes(qs.z, hm.z, j, h);
        wq[(part * 4u + 3u) * 32u + g][c] = q3_bytes(qs.w, hm.w, j, h);
        if (part == 0u) {
            let sd = w4[b4 + 6u];
            let d = unpack2x16float(sd.w & 0xffffu).x;
            let sc = q3_scales32(sd, h, j) * d;
            ws[lr] = vec4<f32>(sc.x, sc.y, 4.0 * sc.x, 4.0 * sc.y);
        }
        if (li < 128u) {
            let xv = x8[tok * k16 + b * 2u + lh];
            let xg = lt / 4u;
            let xc = lt % 4u;
            xq[(lh * 4u) * 16u + xg][xc] = xv.x;
            xq[(lh * 4u + 1u) * 16u + xg][xc] = xv.y;
            xq[(lh * 4u + 2u) * 16u + xg][xc] = xv.z;
            xq[(lh * 4u + 3u) * 16u + xg][xc] = xv.w;
        } else if (li < 192u) {
            let st = min(t0 + li - 128u, p.m - 1u);
            xs[li - 128u] = bitcast<vec4<f32>>(x8[p.xs_at + st * k32 + b]);
        }
        workgroupBarrier();
PLACEHOLDER_COMPUTE
        workgroupBarrier();
    }
PLACEHOLDER_STORE
}
"#;

/// A prompt's matmul from its rows of `x` as int8 ([`QUANT_Q8`]'s), as llama.cpp's MMQ: a workgroup a tile of
/// [`TQ8_ROWS`] weight rows by [`TQ8_TOKENS`] tokens, `k` a 32-block at a time; each step the tile's weights decoded to
/// int8 (four to a word) with their halves' scales and offsets, and the tokens' int8 values and scales, in the
/// workgroup's memory (a word of 4 rows, or of 4 tokens, a vec4, so a warp's loads take every bank once); a thread 8
/// rows by 4 tokens, each half-block's 4 words of products in one `dot4I8Packed` each, then scaled into f32 sums.
/// Where the f32 tiled kernel waits on the workgroup's memory (two vec4 loads for 16 multiply-adds), this does four
/// int8 multiply-adds an instruction from a quarter of the bytes. None for a type without one.
pub fn tiled_q8(dtype: GgmlType) -> Option<String> {
    if dtype != GgmlType::Q3_K {
        return None;
    }
    let comps = ["x", "y", "z", "w"];
    // a thread's 8 rows: 4 of the group tx, 4 of the group 16 + tx; its 4 tokens the group ty
    let row = |ri: u32| if ri < 4 { format!("tx * 4u + {ri}u") } else { format!("64u + tx * 4u + {}u", ri - 4) };
    let mut acc = String::new();
    for ri in 0..8u32 {
        for c in 0..4 {
            acc.push_str(&format!("    var acc{ri}_{c} = 0.0;\n"));
        }
    }
    let mut compute = String::new();
    for half in 0..2u32 {
        for ri in 0..8u32 {
            for c in 0..4 {
                compute.push_str(&if half == 0 { format!("        var p{ri}_{c}: i32 = 0;\n") } else { format!("        p{ri}_{c} = 0;\n") });
            }
        }
        for w in (half * 4)..(half * 4 + 4) {
            compute.push_str(&format!("        {{\n            let wa = wq[{w}u * 32u + tx];\n            let wb = wq[{w}u * 32u + 16u + tx];\n            let xv = xq[{w}u * 16u + ty];\n"));
            for ri in 0..8u32 {
                let wv = if ri < 4 { format!("wa.{}", comps[ri as usize]) } else { format!("wb.{}", comps[(ri - 4) as usize]) };
                for (c, comp) in comps.iter().enumerate() {
                    compute.push_str(&format!("            p{ri}_{c} += dot4I8Packed({wv}, xv.{comp});\n"));
                }
            }
            compute.push_str("        }\n");
        }
        // the half's sums scaled: its scale * the token's d * the products, less its offset * the token's half sum
        let (sc, off, hx) = if half == 0 { ("x", "z", "y") } else { ("y", "w", "z") };
        for c in 0..4u32 {
            compute.push_str(&format!("        let xs{half}_{c} = xs[ty * 4u + {c}u];\n"));
        }
        for ri in 0..8u32 {
            compute.push_str(&format!("        {{\n            let sw = ws[{}];\n", row(ri)));
            for c in 0..4u32 {
                compute.push_str(&format!("            acc{ri}_{c} += sw.{sc} * xs{half}_{c}.x * exact(p{ri}_{c}) - sw.{off} * xs{half}_{c}.{hx};\n"));
            }
            compute.push_str("        }\n");
        }
    }
    let mut store = String::new();
    for ri in 0..8u32 {
        store.push_str(&format!("    {{\n        let row = r0 + {};\n        if (row < p.rows) {{\n", row(ri)));
        for c in 0..4u32 {
            store.push_str(&format!("            if (t0 + ty * 4u + {c}u < p.m) {{ y[(t0 + ty * 4u + {c}u) * p.n + p.row0 + row] = acc{ri}_{c}; }}\n"));
        }
        store.push_str("        }\n    }\n");
    }
    Some(format!("{TQ8_HEAD}{acc}{}", TQ8_LOOP.replace("PLACEHOLDER_COMPUTE\n", &compute).replace("PLACEHOLDER_STORE\n", &store)))
}
