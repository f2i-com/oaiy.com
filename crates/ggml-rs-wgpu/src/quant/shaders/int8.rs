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

/// [`QUANT_Q8`] for a device whose subgroups are 32 lanes (`Gpu::subgroups32`): 8 lanes a block, each 4 of its values
/// (a vec4, the subgroup's loads side by side), the block's largest magnitude and its halves' sums by
/// `subgroupShuffleXor`; the same words, bit for bit. A thread a block read its 32 values in turn, a step's row of 4,096
/// one workgroup's 128 threads: Qwen3.5 9B's replies quantize 129 rows a token, and on an NVIDIA A2 at 1,200 MHz they ran
/// at 23.2 tokens a second where 24.2 so (24.5 with no quantizing at all).
pub const QUANT_Q8_LANES: &str = r#"
@group(0) @binding(0) var<storage, read> unused: array<u32>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = p[0].x;
    let rows = p[0].y;
    let xs_at = p[0].z;
    let blocks = rows * (k / 32u);
    // the thread's block and its part of it (values 4 part..), a block's 8 lanes side by side in a subgroup
    let t = id.x + id.y * 65535u * 256u;
    let b = t / 8u;
    let part = t % 8u;
    let v = x4[min(b, blocks - 1u) * 8u + part];
    var amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    let d = amax / 127.0;
    let inv = select(0.0, 1.0 / d, d > 0.0);
    let qv = clamp(vec4<i32>(round(v * inv)), vec4<i32>(-127), vec4<i32>(127));
    let qb = bitcast<vec4<u32>>(qv) & vec4<u32>(0xffu);
    // the half's sum (its 4 lanes'), and the other half's
    var s = qv.x + qv.y + qv.z + qv.w;
    s += subgroupShuffleXor(s, 1u);
    s += subgroupShuffleXor(s, 2u);
    let other = subgroupShuffleXor(s, 4u);
    if (b < blocks) {
        q[b * 8u + part] = qb.x | (qb.y << 8u) | (qb.z << 16u) | (qb.w << 24u);
        if (part == 0u) {
            let at = (xs_at + b) * 4u;
            q[at] = bitcast<u32>(d);
            q[at + 1u] = bitcast<u32>(d * f32(s));
            q[at + 2u] = bitcast<u32>(d * f32(other));
            q[at + 3u] = 0u;
        }
    }
}
"#;

/// A step's row normed (`y = x / rms(x) * w`, `p[0].y` the bits of `eps`) and its int8 rows made from it as
/// [`QUANT_Q8_LANES`] makes them (into `q`, its scales from vec4 `p[0].z`), in one pass: each workgroup the row's sum of
/// squares for itself (the row read again from the card's cache), then its 32 blocks, 8 lanes a block. On the wide
/// layout (`x` its first vector, `w` its second, `y` and `q` its two written). A norm then the int8 rows' own pass
/// were two dispatches, a 33B's 105 a step each (the quantizing's 0.54 ms of them).
pub const NORM_Q8_LANES: &str = r#"
@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> y4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> q: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> red: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) sgi: u32) {
    let k = p[0].x;
    let eps = bitcast<f32>(p[0].y);
    let xs_at = p[0].z;
    let k4 = k / 4u;
    var s = 0.0;
    for (var i = li; i < k4; i += 256u) {
        let v = x4[i];
        s += dot(v, v);
    }
    s = subgroupAdd(s);
    if (lane == 0u) { red[sgi] = s; }
    workgroupBarrier();
    let total = ((red[0] + red[1]) + (red[2] + red[3])) + ((red[4] + red[5]) + (red[6] + red[7]));
    let inv = 1.0 / sqrt(total / f32(k) + eps);
    // the thread's block and its part of it (values 4 part..), a block's 8 lanes side by side in a subgroup
    let blocks = k / 32u;
    let b = wg.x * 32u + li / 8u;
    let part = li % 8u;
    let at4 = min(b, blocks - 1u) * 8u + part;
    let v = x4[at4] * inv * w4[at4];
    var amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);
    let qv = clamp(vec4<i32>(round(v * id)), vec4<i32>(-127), vec4<i32>(127));
    let qb = bitcast<vec4<u32>>(qv) & vec4<u32>(0xffu);
    // the half's sum (its 4 lanes'), and the other half's
    var hs = qv.x + qv.y + qv.z + qv.w;
    hs += subgroupShuffleXor(hs, 1u);
    hs += subgroupShuffleXor(hs, 2u);
    let other = subgroupShuffleXor(hs, 4u);
    if (b < blocks) {
        y4[at4] = v;
        q[b * 8u + part] = qb.x | (qb.y << 8u) | (qb.z << 16u) | (qb.w << 24u);
        if (part == 0u) {
            let at = (xs_at + b) * 4u;
            q[at] = bitcast<u32>(d);
            q[at + 1u] = bitcast<u32>(d * f32(hs));
            q[at + 2u] = bitcast<u32>(d * f32(other));
            q[at + 3u] = 0u;
        }
    }
}
"#;

/// Elements (4-byte) a buffer of [`QUANT_Q8`]'s output takes for `rows` rows of `k`: the packed rows, then the blocks'
/// scales and sums; and where the scales start, in vec4s.
pub fn q8_len(rows: usize, k: usize) -> (usize, usize) {
    (rows * k / 4 + rows * k / 32 * 4, rows * k / 16)
}

/// Threads a workgroup of [`rb_kernel_q8`]'s kernel has: 256 where subgroups add its sums (on an NVIDIA A2 at 1,200 MHz
/// Qwen3.5 9B's Q4_K [12288, 4096] in 191 us where 128 threads took 199, its [4096, 4096] in 72 where 76, Q5_K's 1 to
/// 3% sooner, Q6_K's the same; 64 and 512 no better), else 128 (the sums' halving through the workgroup's memory).
pub fn q8_threads(subgroups: bool) -> u32 {
    if subgroups { 256 } else { 128 }
}

/// [`rb_kernel`]'s matmul from rows of `x` as int8 ([`QUANT_Q8`]'s): a weight run's values packed as bytes and each 4 of
/// them against 4 of a row's in one `dot4I8Packed`, a 32-block's scale and its halves' sums for the offsets and the
/// minimums; the same lanes, runs and rows. For several rows of `x` (a check of drafts, a short prompt's chunk) it
/// costs little more than for one, where the f32 kernel's multiply-adds a weight grow with them. With `subgroups` (a
/// device whose subgroups are 32 lanes, [`crate::Gpu::subgroups32`]) each subgroup's lanes' sums are added by
/// `subgroupAdd`, not through the workgroup's memory and five barriers: on an NVIDIA A2 at 1,200 MHz a step's row of
/// Qwen3.5 9B's Q4_K [12288, 4096] in 199 us where 222, its Q5_K [8192, 4096] in 160 where 179, its Q6_K output
/// [248320, 4096] in 5.2 ms where 5.7.
pub fn rb_kernel_q8(dtype: GgmlType, r: u32, mr: u32, subgroups: bool) -> Option<String> {
    rb_kernel_q8_by(dtype, r, mr, subgroups, false, false)
}

/// [`rb_kernel_q8`] for a step's one row on a device whose subgroups are 32 lanes, each workgroup's `r` weight rows
/// along `k` by all its 256 lanes (each subgroup a slice of a row's tasks in turn), the subgroups' sums added through
/// the workgroup's memory: a lane walked a row's tasks 32 at a time, seven turns of a 7,168-wide row one after another
/// (Nanbeige4.3's keys [1024, 7168] 12.7 us a step on an RTX 5090, 330 GB/s; its queries [7168, 7168] 25.4, 1,140).
pub fn rb_kernel_q8_split(dtype: GgmlType, r: u32) -> Option<String> {
    rb_kernel_q8_by(dtype, r, 1, true, true, false)
}

/// [`rb_kernel_q8_split`] with each row's sum added to what `y` holds there (a residual stream: a row's one writer,
/// after its workgroup's sums), the add's own pass not needed.
pub fn rb_kernel_q8_split_add(dtype: GgmlType, r: u32) -> Option<String> {
    rb_kernel_q8_by(dtype, r, 1, true, true, true)
}

fn rb_kernel_q8_by(dtype: GgmlType, r: u32, mr: u32, subgroups: bool, split: bool, add: bool) -> Option<String> {
    let (vec4_weights, tasks, block) = match dtype {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K => (true, 8u32, 256u32),
        // (Q6_K's tasks for one row of `x` 64 weights, a few rows' 16: below)
        GgmlType::Q6_K => (false, if mr == 1 { 4u32 } else { 16u32 }, 256u32),
        GgmlType::Q4_0 | GgmlType::Q8_0 => (false, 1u32, 32u32),
        _ => return None,
    };
    let mut s = String::new();
    let mut l = |line: &str| {
        s.push_str(line);
        s.push('\n');
    };
    // (no `enable subgroups;`: naga 30 has not that directive, and takes subgroups' operations where the device was
    // given them)
    l("struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, xs_at: u32, _pad1: u32, }");
    l(if vec4_weights { "@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;" } else { "@group(0) @binding(0) var<storage, read> w: array<u32>;" });
    l("@group(0) @binding(1) var<storage, read> x8: array<vec4<u32>>;");
    l("@group(0) @binding(2) var<storage, read_write> y: array<f32>;");
    l("@group(0) @binding(3) var<uniform> p: Params;");
    if !subgroups {
        l(&format!("var<workgroup> partial: array<f32, {}>;", 128 * r * mr));
    }
    if split {
        l(&format!("var<workgroup> sums: array<f32, {}>;", 8 * r));
    }
    l(RB_HELPERS);
    if !vec4_weights {
        l(RB_Q6K_HELPERS);
    }
    l("// Q3_K: four weights of run `j` (their 2 low bits at shift 2j, their high bit j + 4h) as bytes 0..7 (each plus 4).");
    l("fn q3_bytes(qw: u32, hw: u32, j: u32, h: u32) -> u32 {");
    l("    return ((qw >> (2u * j)) & 0x03030303u) | (((hw >> (j + 4u * h)) & 0x01010101u) << 2u);");
    l("}");
    l(&format!("@compute @workgroup_size({})", q8_threads(subgroups)));
    if subgroups {
        // (a subgroup's lanes and which of the eight it is as the device has them: its sums are its own lanes')
        l("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_id) sgi: u32) {");
        if split {
            l(&format!("    let rbase = (wg.y + wg.z * 65535u) * {r}u;"));
        } else {
            l(&format!("    let rbase = (wg.y + wg.z * 65535u) * {}u + sgi * {r}u;", q8_threads(subgroups) / 32 * r));
        }
    } else {
        l("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {");
        l("    let lane = li & 31u;");
        l(&format!("    let rbase = (wg.y + wg.z * 65535u) * {}u + (li >> 5u) * {r}u;", 4 * r));
    }
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
    if split {
        l(&format!("    for (var c = sgi * 32u + lane; c < blocks * {tasks}u; c += 256u) {{"));
    } else {
        l(&format!("    for (var c = lane; c < blocks * {tasks}u; c += 32u) {{"));
    }
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
        GgmlType::Q8_0 => {
            // a block of 32 a task (34 bytes: its scale, then its int8s, read as words two bytes off in every other),
            // its int8s against the row's 32
            l("        let blk = c;");
            for m in 0..mr {
                l(&format!("        let xlo{m} = x8[xr{m} * k16 + blk * 2u];"));
                l(&format!("        let xhi{m} = x8[xr{m} * k16 + blk * 2u + 1u];"));
                l(&format!("        let sx{m} = bitcast<vec4<f32>>(x8[p.xs_at + xr{m} * k32 + blk]);"));
            }
            for i in 0..r {
                l(&format!("        let bb{i} = rr{i} * p.row_bytes + blk * 34u;"));
                l(&format!("        let d{i} = unpack2x16float(byte(bb{i}) | (byte(bb{i} + 1u) << 8u)).x;"));
                l(&format!("        let qa{i} = words4_at(bb{i} + 2u);"));
                l(&format!("        let qb{i} = words4_at(bb{i} + 18u);"));
                for m in 0..mr {
                    let dl: Vec<String> = comps.iter().map(|comp| format!("dot4I8Packed(qa{i}.{comp}, xlo{m}.{comp})")).collect();
                    let dh: Vec<String> = comps.iter().map(|comp| format!("dot4I8Packed(qb{i}.{comp}, xhi{m}.{comp})")).collect();
                    l(&format!("        let pq{i}_{m} = {} + {};", dl.join(" + "), dh.join(" + ")));
                    l(&format!("        acc{i}_{m} += d{i} * sx{m}.x * f32(pq{i}_{m});"));
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
        GgmlType::Q6_K if mr > 1 => {
            // Q6_K for a few rows of `x`: a task a 32-block's half, 16 weights (its low bits' 16 bytes and its high
            // bits' 16, each read by the other tasks of the block that share them: the 64-weight tasks below took 2 to
            // 4 rows of [5120, 6144] in 477 to 498 us where these take 404 to 410, their rows' values 4 times as many)
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
                l(&format!("        let lw{i} = words4_at(ql{i});"));
                l(&format!("        let hw{i} = words4_at(qh{i});"));
                for (wi, comp) in comps.iter().enumerate() {
                    l(&format!("        let q{i}_{wi} = ((lw{i}.{comp} >> lshift) & 0x0f0f0f0fu) | (((hw{i}.{comp} >> hshift) & 0x03030303u) << 4u);"));
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
        _ => {
            // Q6_K for one row of `x`: a task a block's half `h` (128 weights) and its 16-byte column `hf` of each run
            // there, 64 weights: the low bits' two runs (each byte's low nibble one 32-block's, its high the one two
            // on) and the high bits' one (each byte 2 bits of the four 32-blocks), every byte read once (a step's row
            // of [5120, 6144] in 275 us where 16-weight tasks, which read each low byte twice and each high byte four
            // times, took 327 on an NVIDIA A2 at 1,200 MHz)
            l("        let blk = c / 4u;");
            l("        let h = (c % 4u) / 2u;");
            l("        let hf = c % 2u;");
            l("        let bx = blk * 8u + h * 4u;");
            for m in 0..mr {
                for qd in 0..4 {
                    x_half(&mut l, m, &format!("x{qd}_"), &format!("bx + {qd}u"), "hf");
                }
            }
            for i in 0..r {
                l(&format!("        let bb{i} = rr{i} * p.row_bytes + blk * 210u;"));
                l(&format!("        let la{i} = words4_at(bb{i} + h * 64u + hf * 16u);"));
                l(&format!("        let lb{i} = words4_at(bb{i} + h * 64u + 32u + hf * 16u);"));
                l(&format!("        let hw{i} = words4_at(bb{i} + 128u + h * 32u + hf * 16u);"));
                l(&format!("        let d{i} = unpack2x16float(byte(bb{i} + 208u) | (byte(bb{i} + 209u) << 8u)).x;"));
                for qd in 0..4u32 {
                    let (lw, lshift) = (if qd % 2 == 0 { "la" } else { "lb" }, if qd >= 2 { 4 } else { 0 });
                    for (wi, comp) in comps.iter().enumerate() {
                        l(&format!("        let q{i}_{qd}_{wi} = ((({lw}{i}.{comp} >> {lshift}u) & 0x0f0f0f0fu) | ((({hw}.{comp} >> {hs}u) & 0x03030303u) << 4u));", hw = format!("hw{i}"), hs = 2 * qd));
                    }
                    l(&format!("        let scb{i}_{qd} = byte(bb{i} + 192u + h * 8u + {}u + hf);", 2 * qd));
                    l(&format!("        let sc{i}_{qd} = f32(i32(scb{i}_{qd}) - select(0, 256, scb{i}_{qd} >= 128u));"));
                }
                for m in 0..mr {
                    let mut terms = Vec::new();
                    for qd in 0..4u32 {
                        let dq: Vec<String> = comps.iter().enumerate().map(|(wi, comp)| format!("dot4I8Packed(q{i}_{qd}_{wi}, x{qd}_{m}.{comp})")).collect();
                        l(&format!("        let pq{i}_{qd}_{m} = {};", dq.join(" + ")));
                        terms.push(format!("sc{i}_{qd} * (sx{qd}_{m}.x * f32(pq{i}_{qd}_{m}) - 32.0 * hx{qd}_{m})"));
                    }
                    l(&format!("        acc{i}_{m} += d{i} * ({});", terms.join(" + ")));
                }
            }
        }
    }
    l("    }");
    if split {
        // each subgroup's sums, then the workgroup's by its first `r` lanes
        for i in 0..r {
            l(&format!("    let t{i} = subgroupAdd(acc{i}_0);"));
        }
        l("    if (lane == 0u) {");
        for i in 0..r {
            l(&format!("        sums[sgi * {r}u + {i}u] = t{i};"));
        }
        l("    }");
        l("    workgroupBarrier();");
        l(&format!("    if (sgi == 0u && lane < {r}u && rbase + lane < p.rows) {{"));
        l("        var t = 0.0;");
        l(&format!("        for (var g = 0u; g < 8u; g++) {{ t += sums[g * {r}u + lane]; }}"));
        l(if add { "        y[m0 * p.n + p.row0 + rbase + lane] += t;" } else { "        y[m0 * p.n + p.row0 + rbase + lane] = t;" });
        l("    }");
        l("}");
        return Some(s);
    }
    if subgroups {
        for i in 0..r {
            for m in 0..mr {
                l(&format!("    let t{i}_{m} = subgroupAdd(acc{i}_{m});"));
            }
        }
        l("    if (lane == 0u) {");
        for i in 0..r {
            l(&format!("        if (rbase + {i}u < p.rows) {{"));
            for m in 0..mr {
                l(&format!("            if ({m}u < mn) {{ y[(m0 + {m}u) * p.n + p.row0 + rbase + {i}u] = t{i}_{m}; }}"));
            }
            l("        }");
        }
        l("    }");
        l("}");
        return Some(s);
    }
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
// the step's weights: [word][128 rows], read a group of 4 rows at a time, and a row's (scale a, scale b, offset a,
// offset b); the step's tokens: [word][64 tokens], read 4 at a time, and a token's (d, its halves' sums). The words
// are scalars, not vec4s of 4 rows or tokens: a thread writes one row's or token's, and WGSL lets a write to one
// component of a vector in memory write the whole vector (Metal does), so four threads writing one would race
var<workgroup> wq: array<u32, 1024>;
var<workgroup> ws: array<vec4<f32>, 128>;
var<workgroup> xq: array<u32, 512>;
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
        wq[(part * 4u) * 128u + lr] = q3_bytes(qs.x, hm.x, j, h);
        wq[(part * 4u + 1u) * 128u + lr] = q3_bytes(qs.y, hm.y, j, h);
        wq[(part * 4u + 2u) * 128u + lr] = q3_bytes(qs.z, hm.z, j, h);
        wq[(part * 4u + 3u) * 128u + lr] = q3_bytes(qs.w, hm.w, j, h);
        if (part == 0u) {
            let sd = w4[b4 + 6u];
            let d = unpack2x16float(sd.w & 0xffffu).x;
            let sc = q3_scales32(sd, h, j) * d;
            ws[lr] = vec4<f32>(sc.x, sc.y, 4.0 * sc.x, 4.0 * sc.y);
        }
        if (li < 128u) {
            let xv = x8[tok * k16 + b * 2u + lh];
            xq[(lh * 4u) * 64u + lt] = xv.x;
            xq[(lh * 4u + 1u) * 64u + lt] = xv.y;
            xq[(lh * 4u + 2u) * 64u + lt] = xv.z;
            xq[(lh * 4u + 3u) * 64u + lt] = xv.w;
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
/// workgroup's memory (a word of 4 rows, or of 4 tokens, read at once, so a warp's loads take every bank once); a thread 8
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
            compute.push_str(&format!(
                "        {{\n            let wi = {w}u * 128u + tx * 4u;\n            let xi = {w}u * 64u + ty * 4u;\n            let wa = vec4<u32>(wq[wi], wq[wi + 1u], wq[wi + 2u], wq[wi + 3u]);\n            let wb = vec4<u32>(wq[wi + 64u], wq[wi + 65u], wq[wi + 66u], wq[wi + 67u]);\n            let xv = vec4<u32>(xq[xi], xq[xi + 1u], xq[xi + 2u], xq[xi + 3u]);\n"
            ));
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
