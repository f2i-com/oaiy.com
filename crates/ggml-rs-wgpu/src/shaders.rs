//! WGSL for the quantized matmul `y[m, n] = sum_k x[m, k] * W[n, k]`, where `W`
//! stays in its GGML block layout on the GPU.
//!
//! One kernel is generated per quant type: a shared body plus that type's
//! `dequant(block_byte, sub)`, which writes the 32 values of sub-block `sub`
//! into `v`. Each port follows `ggml-quants`' `dequantize_block` for that type
//! line by line, so the values are the CPU's exactly; only the summation order
//! of the dot product differs.
//!
//! Blocks are not 4-byte aligned (Q8_0 is 34 bytes, Q4_0 18), so bytes are read
//! out of `array<u32>` words, and f16 scales go through `unpack2x16float` (no
//! shader-f16 feature needed).

use ggml_quants::GgmlType;

/// Output rows of `x` handled per workgroup (the partial sums each thread keeps).
pub const M_TILE: u32 = 8;
/// Rows of `x` from which a call takes the tiled kernel ([`source_many`]): 64 tokens by 64 weight rows a workgroup.
pub const MANY_FROM: usize = 9;
/// Tokens (and weight rows) a workgroup of the tiled kernel takes.
pub const MANY_TILE: u32 = 64;
/// Threads per workgroup: they split one weight row's 32-element sub-blocks.
pub const THREADS: u32 = 64;

/// The bytes a block of `dtype` takes on the GPU where it is not ggml's: Q3_K's 110 padded to 112, so its blocks are
/// seven vec4s ([`rb_kernel`]'s Q3_K reads them so).
pub fn padded_block(dtype: GgmlType) -> Option<(usize, usize, usize)> {
    match dtype {
        GgmlType::Q3_K => Some((110, 112, 110)),
        // its scale, then 2 bytes of gap so the 16 bytes of nibbles are words
        GgmlType::Q4_0 => Some((18, 20, 2)),
        _ => None,
    }
}

/// `bytes` (whole blocks of `from` bytes) with each block padded with zeros to `to` bytes (the gap at byte `at` of it),
/// or back (`to < from`: the gap at `at` taken out).
pub fn pad_blocks(bytes: &[u8], from: usize, to: usize, at: usize) -> Vec<u8> {
    assert!(bytes.len() % from == 0, "pad_blocks: {} bytes are not blocks of {from}", bytes.len());
    let mut out = vec![0u8; bytes.len() / from * to];
    for (src, dst) in bytes.chunks_exact(from).zip(out.chunks_exact_mut(to)) {
        dst[..at].copy_from_slice(&src[..at]);
        if to > from {
            dst[at + (to - from)..].copy_from_slice(&src[at..]);
        } else {
            dst[at..].copy_from_slice(&src[at + (from - to)..]);
        }
    }
    out
}

/// Elements per block, bytes per block on the GPU ([`padded_block`]), and the WGSL `dequant` for a type.
pub fn layout(dtype: GgmlType) -> Option<(u32, u32, &'static str)> {
    Some(match dtype {
        GgmlType::Q4_0 => (32, 20, Q4_0),
        GgmlType::Q4_1 => (32, 20, Q4_1),
        GgmlType::Q5_0 => (32, 22, Q5_0),
        GgmlType::Q5_1 => (32, 24, Q5_1),
        GgmlType::Q8_0 => (32, 34, Q8_0),
        GgmlType::IQ4_NL => (32, 18, IQ4_NL),
        GgmlType::Q2_K => (256, 84, Q2_K),
        GgmlType::Q3_K => (256, 112, Q3_K),
        GgmlType::Q4_K => (256, 144, Q4_K),
        GgmlType::Q5_K => (256, 176, Q5_K),
        GgmlType::Q6_K => (256, 210, Q6_K),
        GgmlType::IQ4_XS => (256, 136, IQ4_XS),
        _ => return None,
    })
}

const COMMON: &str = r#"
struct Params {
    k: u32,          // inputs per row
    n: u32,          // output columns in y (all chunks)
    m: u32,          // rows of x
    row0: u32,       // first weight row of this chunk
    rows: u32,       // weight rows in this chunk
    row_bytes: u32,  // bytes per weight row
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<private> v: array<f32, 32>;
var<workgroup> partial: array<f32, THREADS_X_MTILE>;

fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
fn u16at(o: u32) -> u32 { return byte(o) | (byte(o + 1u) << 8u); }
fn u32at(o: u32) -> u32 { return u16at(o) | (u16at(o + 2u) << 16u); }
fn f16at(o: u32) -> f32 { return unpack2x16float(u16at(o)).x; }
fn i8of(b: u32) -> f32 { return f32(i32(b) - select(0, 256, b >= 128u)); }

const KVALUES = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
                               1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
"#;

const BODY: &str = r#"
@compute @workgroup_size(THREADS)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let r = wg.x + wg.y * 65535u;           // weight row within this chunk
    let m0 = wg.z * MTILE;                  // first x row of this tile
    let valid = r < p.rows;
    var acc: array<f32, MTILE>;
    for (var i = 0u; i < MTILE; i++) { acc[i] = 0.0; }
    if (valid) {
        let units = p.k / 32u;
        let row_base = r * p.row_bytes;
        for (var u = t; u < units; u += THREADS) {
            let block = u / SUBS;
            dequant(row_base + block * BLOCK_BYTES, u % SUBS);
            let xk = u * 32u;
            for (var mi = 0u; mi < MTILE; mi++) {
                let m = m0 + mi;
                if (m >= p.m) { break; }
                let xb = m * p.k + xk;
                var s = 0.0;
                for (var j = 0u; j < 32u; j++) { s += v[j] * x[xb + j]; }
                acc[mi] += s;
            }
        }
    }
    for (var mi = 0u; mi < MTILE; mi++) { partial[mi * THREADS + t] = acc[mi]; }
    workgroupBarrier();
    for (var stride = THREADS / 2u; stride > 0u; stride /= 2u) {
        if (t < stride) {
            for (var mi = 0u; mi < MTILE; mi++) {
                partial[mi * THREADS + t] += partial[mi * THREADS + t + stride];
            }
        }
        workgroupBarrier();
    }
    if (valid && t < MTILE) {
        let m = m0 + t;
        if (m < p.m) { y[m * p.n + p.row0 + r] = partial[t * THREADS]; }
    }
}
"#;

/// The tiled kernel for a prompt (more than [`M_TILE`] rows of `x`): 64 tokens by 64 weight rows a workgroup, a
/// 32-element sub-block of `k` a step. Each of 64 threads decodes one weight row's sub-block with the type's own
/// `dequant` (so the weights are the CPU's exactly) into workgroup memory, kept as vec4s of four rows; then every
/// thread adds 4 tokens by 4 rows, its sums four vec4s. The one-row kernel decodes a row's weights once for every 8
/// tokens and adds them in arrays the compiler keeps in memory: a 4096 x 4096 Q4_K weight against 512 tokens took
/// 24 ms (0.7 TFLOP/s).
const MANY_BODY: &str = r#"
var<workgroup> xs: array<f32, 2048>;
// k step kk, rows 4q..4q+3: wt[kk * 16 + q]
var<workgroup> wt: array<vec4<f32>, 512>;

// Token `tok`'s sums of rows row0..row0+3 of this chunk, those it has and of a token there is.
fn put(v: vec4<f32>, tok: u32, row0: u32) {
    if (tok >= p.m) { return; }
    for (var b = 0u; b < 4u; b++) {
        if (row0 + b < p.rows) { y[tok * p.n + p.row0 + row0 + b] = v[b]; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let units = p.k / 32u;
    for (var u = 0u; u < units; u++) {
        // The tokens' 64 x 32: element e, token e / 32, k e % 32, kept k-major.
        for (var e = li; e < 2048u; e += 256u) {
            let tok = t0 + e / 32u;
            var val = 0.0;
            if (tok < p.m) { val = x[tok * p.k + u * 32u + e % 32u]; }
            xs[(e % 32u) * 64u + e / 32u] = val;
        }
        // The rows' sub-block u, a row a thread.
        if (li < 64u) {
            let r = r0 + li;
            if (r < p.rows) {
                dequant(r * p.row_bytes + (u / SUBS) * BLOCK_BYTES, u % SUBS);
                for (var j = 0u; j < 32u; j++) { wt[j * 16u + li / 4u][li % 4u] = v[j]; }
            } else {
                for (var j = 0u; j < 32u; j++) { wt[j * 16u + li / 4u][li % 4u] = 0.0; }
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let w4 = wt[kk * 16u + tx];
            let xb = kk * 64u + ty * 4u;
            acc0 += xs[xb] * w4;
            acc1 += xs[xb + 1u] * w4;
            acc2 += xs[xb + 2u] * w4;
            acc3 += xs[xb + 3u] * w4;
        }
        workgroupBarrier();
    }
    let tok = t0 + ty * 4u;
    let row0 = r0 + tx * 4u;
    put(acc0, tok, row0);
    put(acc1, tok + 1u, row0);
    put(acc2, tok + 2u, row0);
    put(acc3, tok + 3u, row0);
}
"#;

/// The tiled shader for `dtype` (see [`MANY_BODY`]).
pub fn source_many(dtype: GgmlType) -> Option<String> {
    match dtype {
        GgmlType::Q3_K => return Some(Q3K_TILED.to_string()),
        GgmlType::Q4_K => return Some(Q4K_TILED.to_string()),
        GgmlType::Q5_K => return Some(Q5K_TILED.to_string()),
        GgmlType::Q6_K => return Some(Q6K_TILED.to_string()),
        _ => {}
    }
    let (elems, bytes, dequant) = layout(dtype)?;
    let head = COMMON.replace("THREADS_X_MTILE", &(THREADS * M_TILE).to_string());
    let body = MANY_BODY.replace("SUBS", &format!("{}u", elems / 32)).replace("BLOCK_BYTES", &format!("{bytes}u"));
    let helper = if matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K) { SCALE_MIN_K4 } else { "" };
    Some(format!("{head}\n{helper}\n{dequant}\n{body}"))
}

/// The complete shader for `dtype`.
pub fn source(dtype: GgmlType) -> Option<String> {
    source_rows(dtype, M_TILE)
}

/// The one-row kernel for a decode step's single row of `x`: the same kernel with one row's sums, where eight were
/// kept and reduced whatever the rows (a dense model's decode step on WebGPU is its matmuls one row at a time). The
/// K-quants have register-blocked kernels of their own that read the weights wide ([`rb_kernel`]).
pub fn source_decode(dtype: GgmlType) -> Option<String> {
    match source_rb(dtype, 1) {
        Some(s) => Some(s),
        None => source_rows(dtype, 1),
    }
}

/// A K-quant's register-blocked kernel for `mr` rows of `x` a workgroup ([`rb_kernel`]).
fn source_rb(dtype: GgmlType, mr: u32) -> Option<String> {
    rb_kernel(dtype, rb_rows(dtype, mr), mr)
}

/// Weight rows a lane of `dtype`'s register-blocked kernel takes for `mr` rows of `x` a workgroup, as measured on an
/// RTX 5090 (Qwen3.8 27B's and a 3B Llama's shapes, 1, 2 and 4 weight rows a lane): 4 for Q3_K, Q4_K and Q5_K (one
/// row of `x`: Q3_K [17408, 5120] in 26 us where a row a lane took 35); Q6_K, whose words take more work a weight,
/// 1 for one row of `x` and 2 for several.
pub(crate) fn rb_rows(dtype: GgmlType, mr: u32) -> u32 {
    if dtype == GgmlType::Q4_0 {
        return 4;
    }
    match (dtype, mr) {
        (GgmlType::Q6_K, 1) => 1,
        (GgmlType::Q6_K, _) => 2,
        _ => 4,
    }
}

/// Rows of `x` a workgroup of the multi-row kernels takes ([`source_multi`]).
pub const MULTI_ROWS: u32 = 4;

/// Rows of `x` up to which the multi-row kernels go before the tiled one: they read `x` again for every weight row,
/// where the tiled kernel keeps 64 rows of it in workgroup memory but costs the same for a few rows as for 64
/// (Qwen3.8 27B's chunk of 22 tokens in 186 ms with them and in 199 with it, of 512 in 4.0 s and in 1.3).
pub const MULTI_MAX: usize = 24;

/// `dtype`'s multi-row kernel: the K-quants' register-blocked one, [`MULTI_ROWS`] rows of `x` a workgroup.
pub fn source_multi(dtype: GgmlType) -> Option<String> {
    source_rb(dtype, MULTI_ROWS)
}

/// Which kernel `m` rows of `x` take with `dtype`'s weights: 2 the decode kernel (one row), 3 the multi-row one (the
/// wide K-quants', [`source_multi`]), 1 the tiled one, 0 the one-row kernel for a few rows.
pub fn kind(dtype: GgmlType, m: usize) -> u8 {
    if m == 1 {
        2
    } else if m <= MULTI_MAX && matches!(dtype, GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q4_0) {
        3
    } else if m >= MANY_FROM {
        1
    } else {
        0
    }
}

/// The grid of [`kind`]'s kernel for `m` rows of `x` over `rows` weight rows (beyond 65535 a group of them wraps
/// into the next axis).
pub fn grid(dtype: GgmlType, m: usize, rows: u32) -> (u32, u32, u32) {
    match kind(dtype, m) {
        1 => (rows.div_ceil(MANY_TILE), (m as u32).div_ceil(MANY_TILE), 1),
        // the K-quants' register-blocked kernels: x's rows across, the weight rows' groups down
        2 if decode_rows_per_group(dtype, 1) > 1 => {
            let groups = rows.div_ceil(decode_rows_per_group(dtype, 1));
            (1, groups.min(65535), groups.div_ceil(65535))
        }
        2 => {
            let groups = rows.div_ceil(decode_rows_per_group(dtype, 1));
            (groups.min(65535), groups.div_ceil(65535), 1)
        }
        3 => {
            let groups = rows.div_ceil(decode_rows_per_group(dtype, m));
            ((m as u32).div_ceil(MULTI_ROWS), groups.min(65535), groups.div_ceil(65535))
        }
        _ => (rows.min(65535), rows.div_ceil(65535), (m as u32).div_ceil(M_TILE)),
    }
}

/// Weight rows a workgroup of `dtype`'s kernel for `m` rows of `x` (one, or a few) takes: its grid is the rows over
/// this.
pub fn decode_rows_per_group(dtype: GgmlType, m: usize) -> u32 {
    match dtype {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q4_0 => 4 * rb_rows(dtype, if m == 1 { 1 } else { MULTI_ROWS }),
        _ => 1,
    }
}

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

/// Rows of `x` as f16 for [`coop_tiled`] (two to a word, rounded to nearest), `p[0].z` rows of them (past `p[0].y`,
/// zero: the tile's padding). `p[0]`: k, rows, padded rows.
pub const X_F16: &str = r#"
@group(0) @binding(0) var<storage, read> unused: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k2 = p[0].x / 2u;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= p[0].z * k2) { return; }
    var v = vec2<f32>(0.0);
    if (i / k2 < p[0].y) { v = x2[i]; }
    q[i] = pack2x16float(v);
}
"#;

/// Weight rows (and tokens) a workgroup of [`coop_tiled`] takes.
pub const COOP_TILE: u32 = 128;

/// How many workgroups of 1024 threads run at once (a GPU's SMs or compute units, where one holds one of them; their
/// tensor cores are what a matmul's workgroups share): each works a while (`p[0].x` dependent multiply-adds), and
/// those that start before any has finished are counted (`c[0]`; `c[1]` those finished).
pub const COOP_UNITS_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> unused0: array<u32>;
@group(0) @binding(1) var<storage, read> unused1: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> pad: array<f32, 1024>;

@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    var before = 1u;
    if (li == 0u) { before = atomicLoad(&c[1]); }
    var acc = f32(li) * 1e-3;
    for (var i = 0u; i < p[0].x; i++) { acc = fma(acc, 0.9999, 1e-4); }
    pad[li] = acc;
    workgroupBarrier();
    if (li == 0u) {
        if (before == 0u) { atomicAdd(&c[0], 1u); }
        atomicAdd(&c[1], 1u);
        if (pad[(wg.x * 7u) % 1024u] == 12345.0) { atomicAdd(&c[2], 1u); }
    }
}
"#;

/// The splits of a tensor-core matmul's `steps` (of 64) for `groups` workgroups on a GPU of `units` (its SMs: a
/// workgroup alone on one runs about twice as fast as two together): the fewest of those that take least time, each
/// wave of `units` workgroups taking as long however full, and each split's sums written and read again besides (as
/// much as the matmul's own time `4 s / k` per FLOP of the GPU's per byte, taken as 100), at most 8 splits of 4 steps
/// or more; and how many that is with none empty.
pub fn coop_splits(groups: u32, units: u32, steps: u32) -> u32 {
    let slots = units.max(1);
    let k = (steps * 64) as f64;
    let cost = |s: u32| {
        let w = groups * s;
        let waves = w.div_ceil(slots) * slots;
        waves as f64 / w as f64 + if s > 1 { 400.0 * s as f64 / k } else { 0.0 }
    };
    let most = (steps / 4).clamp(1, 8);
    let s = (1..=most).fold(1, |best, s| if cost(s) < cost(best) - 1e-9 { s } else { best });
    steps.div_ceil(steps.div_ceil(s))
}

/// [`coop_tiled`]'s kernel before its type's decode is put in.
const COOP_KERNEL: &str = r#"enable f16;
enable wgpu_cooperative_matrix;
struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, splits: u32, _pad1: u32, }
WEIGHTS_BINDING
@group(0) @binding(1) var<storage, read> x16: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// two steps' weights [row][k] as f16, four to a vec4, a row 18 vec4s apart (16 bytes past the step's 64, against bank
// conflicts): one decoded while the other's are multiplied; a subgroup's fragment staged at a tile's edge
const STRIDE4: u32 = 18u;
const BUF4: u32 = 2304u;
var<workgroup> wt: array<vec4<f16>, 4608>;
var<workgroup> edge: array<f32, 2048>;

DECODE_HELPERS

// a byte as f32, less `o`: its bits in 2^23's mantissa (exact, no conversion)
fn byte_less(w: u32, o: f32) -> vec4<f32> {
    let b = vec4<u32>(w & 255u, (w >> 8u) & 255u, (w >> 16u) & 255u, w >> 24u);
    return bitcast<vec4<f32>>(b | vec4<u32>(0x4b000000u)) - vec4<f32>(8388608.0 + o);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 128u;
    let t0 = wg.y * 128u;
    let sg = li / 32u;
    let lane = li % 32u;
    // the subgroup's 32 rows by 64 tokens of the tile
    let sr = (sg % 4u) * 32u;
    let st = (sg / 4u) * 64u;
    // what this thread decodes: weight row lr's 32-block lh of a step
    let lr = li / 2u;
    let lh = li % 2u;
    let rr = r0 + lr;
    let kx = p.k;
    // the workgroup's split of the steps (`wg.z` of `p.splits`, none empty), its sums that split's part of `y`
    let all = p.k / 64u;
    let per = (all + p.splits - 1u) / p.splits;
    let s0 = wg.z * per;
    let s1 = min(all, s0 + per);
    let zo = wg.z * p.m * p.n;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c02 = coop_mat16x16<f32, C>();
    var c03 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();
    var c12 = coop_mat16x16<f32, C>();
    var c13 = coop_mat16x16<f32, C>();
    // the first step's weights, then each step's next decoded as it multiplies
    {
        let b = s0;
        let buf = (s0 % 2u) * BUF4;
        DECODE_STEP
    }
    workgroupBarrier();
    for (var b0 = s0; b0 < s1; b0++) {
        if (b0 + 1u < s1) {
            let b = b0 + 1u;
            let buf = (b % 2u) * BUF4;
            DECODE_STEP
        }
        let cur = (b0 % 2u) * BUF4;
        // (every index and stride a `let` of its own: naga's SPIR-V wants a cooperative load's and store's operands
        // emitted before them); the tokens' fragments straight from their f16 rows (padded to the tile)
        for (var kk = 0u; kk < 64u; kk += 16u) {
            let s18 = STRIDE4;
            let ia0 = cur + sr * STRIDE4 + kk / 4u;
            let ia1 = cur + (sr + 16u) * STRIDE4 + kk / 4u;
            let ib0 = (t0 + st) * kx + b0 * 64u + kk;
            let ib1 = ib0 + 16u * kx;
            let ib2 = ib0 + 32u * kx;
            let ib3 = ib0 + 48u * kx;
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia0], s18);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia1], s18);
            let b0f = coopLoad<coop_mat16x16<f16, B>>(&x16[ib0], kx);
            let b1f = coopLoad<coop_mat16x16<f16, B>>(&x16[ib1], kx);
            let b2f = coopLoad<coop_mat16x16<f16, B>>(&x16[ib2], kx);
            let b3f = coopLoad<coop_mat16x16<f16, B>>(&x16[ib3], kx);
            c00 = coopMultiplyAdd(a0, b0f, c00);
            c01 = coopMultiplyAdd(a0, b1f, c01);
            c02 = coopMultiplyAdd(a0, b2f, c02);
            c03 = coopMultiplyAdd(a0, b3f, c03);
            c10 = coopMultiplyAdd(a1, b0f, c10);
            c11 = coopMultiplyAdd(a1, b1f, c11);
            c12 = coopMultiplyAdd(a1, b2f, c12);
            c13 = coopMultiplyAdd(a1, b3f, c13);
        }
        workgroupBarrier();
    }
    // out: y[token, row] is the tile's (row, token) column-major, a token's rows `n` apart
    let ns = p.n;
    let full = r0 + 128u <= p.rows && t0 + 128u <= p.m;
    if (full) {
        let o = zo + (t0 + st) * ns + p.row0 + r0 + sr;
        let o01 = o + 16u * ns;
        let o02 = o + 32u * ns;
        let o03 = o + 48u * ns;
        let o10 = o + 16u;
        let o11 = o + 16u + 16u * ns;
        let o12 = o + 16u + 32u * ns;
        let o13 = o + 16u + 48u * ns;
        coopStore(c00, &y[o], ns);
        coopStore(c01, &y[o01], ns);
        coopStore(c02, &y[o02], ns);
        coopStore(c03, &y[o03], ns);
        coopStore(c10, &y[o10], ns);
        coopStore(c11, &y[o11], ns);
        coopStore(c12, &y[o12], ns);
        coopStore(c13, &y[o13], ns);
    } else {
        // a tile at the edge: each fragment through the subgroup's staging, its rows and tokens in bounds
        EDGE_STORES
    }
}
"#;

/// [`coop_tiled`]'s store of a fragment at a tile's edge.
const COOP_EDGE: &str = r#"        {
            let eo = sg * 256u;
            coopStore(CF, &edge[eo], 16u);
            workgroupBarrier();
            for (var e = lane; e < 256u; e += 32u) {
                let row = r0 + sr + FR + e % 16u;
                let t = t0 + st + FT + e / 16u;
                if (row < p.rows && t < p.m) { y[zo + t * p.n + p.row0 + row] = edge[sg * 256u + e]; }
            }
            workgroupBarrier();
        }
"#;

/// Q3_K's decode for [`coop_tiled`]: its helpers, and a thread's half of a row's 32-block.
const COOP_Q3K_HELPERS: &str = r#"fn q3_bytes(qw: u32, hw: u32, j: u32, h: u32) -> u32 {
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
"#;
/// The other K-quants' decodes for [`coop_tiled`]: Q4_K's and Q5_K's (their 6-bit scales and minimums), Q6_K's
/// (its 210-byte blocks read as words, two scales a 32-block).
const COOP_K_HELPERS: &str = r#"// Byte `b` (0..12) of a block's scales, the header's last three words.
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
"#;
const COOP_Q4K_STEP: &str = r#"let at4 = buf + lr * STRIDE4 + lh * 8u;
        if (rr < p.rows) {
            let bb = 2u * b + lh;
            let blk = bb / 8u;
            let sb = bb % 8u;
            let pr = sb / 2u;
            let sh = 4u * (sb % 2u);
            let b4 = rr * (p.row_bytes / 16u) + blk * 9u;
            let hd = w4[b4];
            let qa = w4[b4 + 1u + 2u * pr];
            let qb = w4[b4 + 2u + 2u * pr];
            let dm = unpack2x16float(hd.x);
            let smn = scale_min(hd, sb);
            let dsc = dm.x * smn.x;
            let dmn = vec4<f32>(dm.y * smn.y);
            wt[at4] = vec4<f16>(dsc * byte_less((qa.x >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 1u] = vec4<f16>(dsc * byte_less((qa.y >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 2u] = vec4<f16>(dsc * byte_less((qa.z >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 3u] = vec4<f16>(dsc * byte_less((qa.w >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 4u] = vec4<f16>(dsc * byte_less((qb.x >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 5u] = vec4<f16>(dsc * byte_less((qb.y >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 6u] = vec4<f16>(dsc * byte_less((qb.z >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 7u] = vec4<f16>(dsc * byte_less((qb.w >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
        } else {
            for (var i = 0u; i < 8u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;
const COOP_Q5K_STEP: &str = r#"let at4 = buf + lr * STRIDE4 + lh * 8u;
        if (rr < p.rows) {
            let bb = 2u * b + lh;
            let blk = bb / 8u;
            let sb = bb % 8u;
            let pr = sb / 2u;
            let sh = 4u * (sb % 2u);
            let b4 = rr * (p.row_bytes / 16u) + blk * 11u;
            let hd = w4[b4];
            let h0 = w4[b4 + 1u];
            let h1 = w4[b4 + 2u];
            let qa = w4[b4 + 3u + 2u * pr];
            let qb = w4[b4 + 4u + 2u * pr];
            let dm = unpack2x16float(hd.x);
            let smn = scale_min(hd, sb);
            let dsc = dm.x * smn.x;
            let dmn = vec4<f32>(dm.y * smn.y);
            wt[at4] = vec4<f16>(dsc * byte_less(((qa.x >> sh) & 0x0f0f0f0fu) | (((h0.x >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 1u] = vec4<f16>(dsc * byte_less(((qa.y >> sh) & 0x0f0f0f0fu) | (((h0.y >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 2u] = vec4<f16>(dsc * byte_less(((qa.z >> sh) & 0x0f0f0f0fu) | (((h0.z >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 3u] = vec4<f16>(dsc * byte_less(((qa.w >> sh) & 0x0f0f0f0fu) | (((h0.w >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 4u] = vec4<f16>(dsc * byte_less(((qb.x >> sh) & 0x0f0f0f0fu) | (((h1.x >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 5u] = vec4<f16>(dsc * byte_less(((qb.y >> sh) & 0x0f0f0f0fu) | (((h1.y >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 6u] = vec4<f16>(dsc * byte_less(((qb.z >> sh) & 0x0f0f0f0fu) | (((h1.z >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 7u] = vec4<f16>(dsc * byte_less(((qb.w >> sh) & 0x0f0f0f0fu) | (((h1.w >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
        } else {
            for (var i = 0u; i < 8u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;
const COOP_Q6K_HELPERS: &str = r#"fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
// The four bytes at `o`, an even offset (a word, or the halves of two).
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}
"#;
const COOP_Q6K_STEP: &str = r#"let at4 = buf + lr * STRIDE4 + lh * 8u;
        if (rr < p.rows) {
            let bb = 2u * b + lh;
            let blk = bb / 8u;
            let sub = bb % 8u;
            let h = sub / 4u;
            let qd = sub % 4u;
            let lshift = select(0u, 4u, qd >= 2u);
            let hshift = 2u * qd;
            let base = rr * p.row_bytes + blk * 210u;
            let d = unpack2x16float(byte(base + 208u) | (byte(base + 209u) << 8u)).x;
            for (var hf = 0u; hf < 2u; hf++) {
                let ql = base + h * 64u + (qd & 1u) * 32u + hf * 16u;
                let qh = base + 128u + h * 32u + hf * 16u;
                let scb = byte(base + 192u + h * 8u + 2u * qd + hf);
                let sc = d * f32(i32(scb) - select(0, 256, scb >= 128u));
                for (var wi = 0u; wi < 4u; wi++) {
                    let q = ((word_at(ql + 4u * wi) >> lshift) & 0x0f0f0f0fu) | (((word_at(qh + 4u * wi) >> hshift) & 0x03030303u) << 4u);
                    wt[at4 + hf * 4u + wi] = vec4<f16>(sc * byte_less(q, 32.0));
                }
            }
        } else {
            for (var i = 0u; i < 8u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;
const COOP_Q3K_STEP: &str = r#"let at4 = buf + lr * STRIDE4 + lh * 8u;
        if (rr < p.rows) {
            let bb = 2u * b + lh;
            let blk = bb / 8u;
            let q = bb % 8u;
            let h = q / 4u;
            let j = q % 4u;
            let b4 = rr * (p.row_bytes / 16u) + blk * 7u;
            let hm0 = w4[b4];
            let hm1 = w4[b4 + 1u];
            let qs0 = w4[b4 + 2u + 2u * h];
            let qs1 = w4[b4 + 3u + 2u * h];
            let sd = w4[b4 + 6u];
            let sc = q3_scales32(sd, h, j) * unpack2x16float(sd.w & 0xffffu).x;
            wt[at4] = vec4<f16>(sc.x * byte_less(q3_bytes(qs0.x, hm0.x, j, h), 4.0));
            wt[at4 + 1u] = vec4<f16>(sc.x * byte_less(q3_bytes(qs0.y, hm0.y, j, h), 4.0));
            wt[at4 + 2u] = vec4<f16>(sc.x * byte_less(q3_bytes(qs0.z, hm0.z, j, h), 4.0));
            wt[at4 + 3u] = vec4<f16>(sc.x * byte_less(q3_bytes(qs0.w, hm0.w, j, h), 4.0));
            wt[at4 + 4u] = vec4<f16>(sc.y * byte_less(q3_bytes(qs1.x, hm1.x, j, h), 4.0));
            wt[at4 + 5u] = vec4<f16>(sc.y * byte_less(q3_bytes(qs1.y, hm1.y, j, h), 4.0));
            wt[at4 + 6u] = vec4<f16>(sc.y * byte_less(q3_bytes(qs1.z, hm1.z, j, h), 4.0));
            wt[at4 + 7u] = vec4<f16>(sc.y * byte_less(q3_bytes(qs1.w, hm1.w, j, h), 4.0));
        } else {
            for (var i = 0u; i < 8u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// A prompt's matmul on the tensor cores (WGSL's cooperative matrices, f16 into f32): a workgroup a tile of
/// [`COOP_TILE`] weight rows by as many tokens, `k` 64 at a time; each step the tile's weights decoded to f16 in the
/// workgroup's memory, then each of its 8 subgroups multiplies its 32 rows by 64 tokens as 2 by 4 fragments of 16x16,
/// the tokens' fragments read straight from their rows as f16 ([`X_F16`]'s, padded to the tile); the sums stored
/// straight into `y` (a tile at the edge through a fragment's staging). None for a type without one.
pub fn coop_tiled(dtype: GgmlType) -> Option<String> {
    let vec4s = "@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;";
    let (binding, helpers, step) = match dtype {
        GgmlType::Q3_K => (vec4s, COOP_Q3K_HELPERS, COOP_Q3K_STEP),
        GgmlType::Q4_K => (vec4s, COOP_K_HELPERS, COOP_Q4K_STEP),
        GgmlType::Q5_K => (vec4s, COOP_K_HELPERS, COOP_Q5K_STEP),
        GgmlType::Q6_K => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_Q6K_HELPERS, COOP_Q6K_STEP),
        _ => return None,
    };
    let frags = [("c00", 0u32, 0u32), ("c01", 0, 16), ("c02", 0, 32), ("c03", 0, 48), ("c10", 16, 0), ("c11", 16, 16), ("c12", 16, 32), ("c13", 16, 48)];
    let edges: String = frags.iter().map(|(cf, fr, ft)| COOP_EDGE.replace("CF", cf).replace("FR", &format!("{fr}u")).replace("FT", &format!("{ft}u"))).collect();
    Some(COOP_KERNEL.replace("WEIGHTS_BINDING", binding).replace("DECODE_HELPERS", helpers).replace("DECODE_STEP", step).replace("EDGE_STORES", &edges))
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
fn rb_kernel(dtype: GgmlType, r: u32, mr: u32) -> Option<String> {
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
const RB_HELPERS: &str = r#"
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
const RB_Q6K_HELPERS: &str = r#"
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

/// The tiled kernel for the K-quants ([`source_many`]'s for them): 64 weight rows by 64 tokens a workgroup as the
/// generic one, but 64 values of `k` a step, each of its 256 threads decoding 16 weights of its row (a quarter of
/// the step) from wide loads, as the CPU dequantizes them, where 64 threads decoded a row's 32 a byte a load while
/// the rest waited; and `x` read four at a time. Qwen3.8 27B's chunk of 512 in 2.4 s with the generic one.
const Q3K_TILED: &str = r#"
struct Params {
    k: u32,
    n: u32,
    m: u32,
    row0: u32,
    rows: u32,
    row_bytes: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// k-major: the step's k index kk, then tokens (xs) or rows (ws) four to a vec4
var<workgroup> xs: array<vec4<f32>, 1024>;
var<workgroup> ws: array<vec4<f32>, 1024>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 16u + row / 4u][row % 4u] = v;
}

fn decode_step(row4: u32, s: u32, wr: u32, wq: u32) {
    let b4 = row4 + (s / 4u) * 7u;
    let h = (s % 4u) / 2u;
    let jp = s % 2u;
    let g = wq / 2u;
    let hm = w4[b4 + g];
    let qs = w4[b4 + 2u + 2u * h + g];
    let sd = w4[b4 + 6u];
    let d = unpack2x16float(sd.w & 0xffffu).x;
    let sa = ((sd.x >> (4u * h)) & 0x0f0f0f0fu) | (((sd.z >> (4u * h)) & 0x03030303u) << 4u);
    let sb = ((sd.y >> (4u * h)) & 0x0f0f0f0fu) | (((sd.z >> (4u * h + 2u)) & 0x03030303u) << 4u);
    let sw = select(sa, sb, jp == 1u);
    let gs = 8u * g;
    // ggml's dl = d * (scale - 32) of runs j0 = 2jp and j0 + 1
    let dl0 = d * (f32((sw >> gs) & 255u) - 32.0);
    let dl1 = d * (f32((sw >> (gs + 16u)) & 255u) - 32.0);
    let j0 = 2u * jp;
    for (var wi = 0u; wi < 2u; wi++) {
        let qw = qs[(wq % 2u) * 2u + wi];
        let hw = hm[(wq % 2u) * 2u + wi];
        for (var b = 0u; b < 4u; b++) {
            let qb = (qw >> (8u * b)) & 255u;
            let hb = (hw >> (8u * b)) & 255u;
            let l = (wq % 2u) * 8u + wi * 4u + b;
            let v0 = i32((qb >> (2u * j0)) & 3u) - select(4, 0, (hb & (1u << (j0 + 4u * h))) != 0u);
            let v1 = i32((qb >> (2u * j0 + 2u)) & 3u) - select(4, 0, (hb & (1u << (j0 + 1u + 4u * h))) != 0u);
            put_w(g * 16u + l, wr, dl0 * f32(v0));
            put_w(32u + g * 16u + l, wr, dl1 * f32(v1));
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    // what this thread loads: x's token li / 4, its 16 of the step's k at (li % 4) * 16; and decodes: row li / 4's
    // quarter li % 4 of the step
    let xt = li / 4u;
    let xq = li % 4u;
    let wr = li / 4u;
    let wq = li % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let k4 = p.k / 4u;
    let steps = p.k / 64u;
    for (var s = 0u; s < steps; s++) {
        let tok = t0 + xt;
        for (var i = 0u; i < 4u; i++) {
            var v = vec4<f32>(0.0);
            if (tok < p.m) { v = x4[tok * k4 + s * 16u + xq * 4u + i]; }
            let kk = xq * 16u + i * 4u;
            xs[kk * 16u + xt / 4u][xt % 4u] = v.x;
            xs[(kk + 1u) * 16u + xt / 4u][xt % 4u] = v.y;
            xs[(kk + 2u) * 16u + xt / 4u][xt % 4u] = v.z;
            xs[(kk + 3u) * 16u + xt / 4u][xt % 4u] = v.w;
        }
        let r = r0 + wr;
        if (r < p.rows) {
            decode_step(r * (p.row_bytes / 16u), s, wr, wq);
        } else {
            for (var i = 0u; i < 8u; i++) {
                put_w(wq * 8u + i, wr, 0.0);
                put_w(32u + wq * 8u + i, wr, 0.0);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 64u; kk++) {
            let wv = ws[kk * 16u + tx];
            let xv = xs[kk * 16u + ty];
            acc0 += xv.x * wv;
            acc1 += xv.y * wv;
            acc2 += xv.z * wv;
            acc3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let row0 = r0 + tx * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let tok = t0 + ty * 4u + i;
        if (tok < p.m) {
            for (var b = 0u; b < 4u; b++) {
                if (row0 + b < p.rows) { y[tok * p.n + p.row0 + row0 + b] = accs[i][b]; }
            }
        }
    }
}
"#;

const Q4K_TILED: &str = r#"
struct Params {
    k: u32,
    n: u32,
    m: u32,
    row0: u32,
    rows: u32,
    row_bytes: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// k-major: the step's k index kk, then tokens (xs) or rows (ws) four to a vec4
var<workgroup> xs: array<vec4<f32>, 1024>;
var<workgroup> ws: array<vec4<f32>, 1024>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 16u + row / 4u][row % 4u] = v;
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

fn decode_step(row4: u32, s: u32, wr: u32, wq: u32) {
    let b4 = row4 + (s / 4u) * 9u;
    let pair = s % 4u;
    let h = w4[b4];
    let qv = w4[b4 + 1u + 2u * pair + wq / 2u];
    let dm = unpack2x16float(h.x);
    let slo = scale_min(h, 2u * pair);
    let shi = scale_min(h, 2u * pair + 1u);
    let d1 = dm.x * slo.x;
    let m1 = dm.y * slo.y;
    let d2 = dm.x * shi.x;
    let m2 = dm.y * shi.y;
    for (var wi = 0u; wi < 2u; wi++) {
        let word = qv[(wq % 2u) * 2u + wi];
        for (var b = 0u; b < 4u; b++) {
            let byte = (word >> (8u * b)) & 255u;
            let l = wq * 8u + wi * 4u + b;
            put_w(l, wr, d1 * f32(byte & 15u) - m1);
            put_w(32u + l, wr, d2 * f32(byte >> 4u) - m2);
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    // what this thread loads: x's token li / 4, its 16 of the step's k at (li % 4) * 16; and decodes: row li / 4's
    // quarter li % 4 of the step
    let xt = li / 4u;
    let xq = li % 4u;
    let wr = li / 4u;
    let wq = li % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let k4 = p.k / 4u;
    let steps = p.k / 64u;
    for (var s = 0u; s < steps; s++) {
        let tok = t0 + xt;
        for (var i = 0u; i < 4u; i++) {
            var v = vec4<f32>(0.0);
            if (tok < p.m) { v = x4[tok * k4 + s * 16u + xq * 4u + i]; }
            let kk = xq * 16u + i * 4u;
            xs[kk * 16u + xt / 4u][xt % 4u] = v.x;
            xs[(kk + 1u) * 16u + xt / 4u][xt % 4u] = v.y;
            xs[(kk + 2u) * 16u + xt / 4u][xt % 4u] = v.z;
            xs[(kk + 3u) * 16u + xt / 4u][xt % 4u] = v.w;
        }
        let r = r0 + wr;
        if (r < p.rows) {
            decode_step(r * (p.row_bytes / 16u), s, wr, wq);
        } else {
            for (var i = 0u; i < 8u; i++) {
                put_w(wq * 8u + i, wr, 0.0);
                put_w(32u + wq * 8u + i, wr, 0.0);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 64u; kk++) {
            let wv = ws[kk * 16u + tx];
            let xv = xs[kk * 16u + ty];
            acc0 += xv.x * wv;
            acc1 += xv.y * wv;
            acc2 += xv.z * wv;
            acc3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let row0 = r0 + tx * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let tok = t0 + ty * 4u + i;
        if (tok < p.m) {
            for (var b = 0u; b < 4u; b++) {
                if (row0 + b < p.rows) { y[tok * p.n + p.row0 + row0 + b] = accs[i][b]; }
            }
        }
    }
}
"#;

const Q6K_TILED: &str = r#"
struct Params {
    k: u32,
    n: u32,
    m: u32,
    row0: u32,
    rows: u32,
    row_bytes: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// k-major: the step's k index kk, then tokens (xs) or rows (ws) four to a vec4
var<workgroup> xs: array<vec4<f32>, 1024>;
var<workgroup> ws: array<vec4<f32>, 1024>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 16u + row / 4u][row % 4u] = v;
}


fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }

// The four bytes at `o`, an even offset (a word, or the halves of two).
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}

// Q6_K (210-byte blocks, every other one two bytes into a word): step s is block s / 4's half h = (s % 4) / 2 and
// its quarters q4 = 2qp and 2qp + 1 (qp = s % 2), 32 weights each; a thread takes l from wq * 8 of both: their low
// bits' bytes (l and 32 + l, a nibble each), their high bits' (l), their scales (a byte for every 16 l).
fn decode_step(row_bytes: u32, s: u32, wr: u32, wq: u32) {
    let bb = row_bytes + (s / 4u) * 210u;
    let h = (s % 4u) / 2u;
    let qp = s % 2u;
    let l0 = wq * 8u;
    let d = unpack2x16float(byte(bb + 208u) | (byte(bb + 209u) << 8u)).x;
    let s0b = byte(bb + 192u + h * 8u + wq / 2u + 4u * qp);
    let s1b = byte(bb + 192u + h * 8u + wq / 2u + 4u * qp + 2u);
    // ggml's d * scale, the scales int8
    let ds0 = d * f32(i32(s0b) - select(0, 256, s0b >= 128u));
    let ds1 = d * f32(i32(s1b) - select(0, 256, s1b >= 128u));
    let lshift = 4u * qp;
    for (var wi = 0u; wi < 2u; wi++) {
        let la = word_at(bb + h * 64u + l0 + wi * 4u);
        let lb = word_at(bb + h * 64u + 32u + l0 + wi * 4u);
        let hb = word_at(bb + 128u + h * 32u + l0 + wi * 4u);
        for (var b = 0u; b < 4u; b++) {
            let l = l0 + wi * 4u + b;
            let hbits = (hb >> (8u * b)) & 255u;
            let qa = (((la >> (8u * b)) >> lshift) & 15u) | (((hbits >> (4u * qp)) & 3u) << 4u);
            let qb = (((lb >> (8u * b)) >> lshift) & 15u) | (((hbits >> (4u * qp + 2u)) & 3u) << 4u);
            put_w(l, wr, ds0 * f32(i32(qa) - 32));
            put_w(32u + l, wr, ds1 * f32(i32(qb) - 32));
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    // what this thread loads: x's token li / 4, its 16 of the step's k at (li % 4) * 16; and decodes: row li / 4's
    // quarter li % 4 of the step
    let xt = li / 4u;
    let xq = li % 4u;
    let wr = li / 4u;
    let wq = li % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let k4 = p.k / 4u;
    let steps = p.k / 64u;
    for (var s = 0u; s < steps; s++) {
        let tok = t0 + xt;
        for (var i = 0u; i < 4u; i++) {
            var v = vec4<f32>(0.0);
            if (tok < p.m) { v = x4[tok * k4 + s * 16u + xq * 4u + i]; }
            let kk = xq * 16u + i * 4u;
            xs[kk * 16u + xt / 4u][xt % 4u] = v.x;
            xs[(kk + 1u) * 16u + xt / 4u][xt % 4u] = v.y;
            xs[(kk + 2u) * 16u + xt / 4u][xt % 4u] = v.z;
            xs[(kk + 3u) * 16u + xt / 4u][xt % 4u] = v.w;
        }
        let r = r0 + wr;
        if (r < p.rows) {
            decode_step(r * p.row_bytes, s, wr, wq);
        } else {
            for (var i = 0u; i < 8u; i++) {
                put_w(wq * 8u + i, wr, 0.0);
                put_w(32u + wq * 8u + i, wr, 0.0);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 64u; kk++) {
            let wv = ws[kk * 16u + tx];
            let xv = xs[kk * 16u + ty];
            acc0 += xv.x * wv;
            acc1 += xv.y * wv;
            acc2 += xv.z * wv;
            acc3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let row0 = r0 + tx * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let tok = t0 + ty * 4u + i;
        if (tok < p.m) {
            for (var b = 0u; b < 4u; b++) {
                if (row0 + b < p.rows) { y[tok * p.n + p.row0 + row0 + b] = accs[i][b]; }
            }
        }
    }
}
"#;

const Q5K_TILED: &str = r#"
struct Params {
    k: u32,
    n: u32,
    m: u32,
    row0: u32,
    rows: u32,
    row_bytes: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// k-major: the step's k index kk, then tokens (xs) or rows (ws) four to a vec4
var<workgroup> xs: array<vec4<f32>, 1024>;
var<workgroup> ws: array<vec4<f32>, 1024>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 16u + row / 4u][row % 4u] = v;
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

fn decode_step(row4: u32, s: u32, wr: u32, wq: u32) {
    let b4 = row4 + (s / 4u) * 11u;
    let pair = s % 4u;
    let h = w4[b4];
    let hv = w4[b4 + 1u + wq / 2u];
    let qv = w4[b4 + 3u + 2u * pair + wq / 2u];
    let dm = unpack2x16float(h.x);
    let slo = scale_min(h, 2u * pair);
    let shi = scale_min(h, 2u * pair + 1u);
    let d1 = dm.x * slo.x;
    let m1 = dm.y * slo.y;
    let d2 = dm.x * shi.x;
    let m2 = dm.y * shi.y;
    for (var wi = 0u; wi < 2u; wi++) {
        let word = qv[(wq % 2u) * 2u + wi];
        let hword = hv[(wq % 2u) * 2u + wi];
        for (var b = 0u; b < 4u; b++) {
            let byte = (word >> (8u * b)) & 255u;
            let hb = (hword >> (8u * b)) & 255u;
            let l = wq * 8u + wi * 4u + b;
            let lo = (byte & 15u) + select(0u, 16u, (hb & (1u << (2u * pair))) != 0u);
            let hi = (byte >> 4u) + select(0u, 16u, (hb & (2u << (2u * pair))) != 0u);
            put_w(l, wr, d1 * f32(lo) - m1);
            put_w(32u + l, wr, d2 * f32(hi) - m2);
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    // what this thread loads: x's token li / 4, its 16 of the step's k at (li % 4) * 16; and decodes: row li / 4's
    // quarter li % 4 of the step
    let xt = li / 4u;
    let xq = li % 4u;
    let wr = li / 4u;
    let wq = li % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let k4 = p.k / 4u;
    let steps = p.k / 64u;
    for (var s = 0u; s < steps; s++) {
        let tok = t0 + xt;
        for (var i = 0u; i < 4u; i++) {
            var v = vec4<f32>(0.0);
            if (tok < p.m) { v = x4[tok * k4 + s * 16u + xq * 4u + i]; }
            let kk = xq * 16u + i * 4u;
            xs[kk * 16u + xt / 4u][xt % 4u] = v.x;
            xs[(kk + 1u) * 16u + xt / 4u][xt % 4u] = v.y;
            xs[(kk + 2u) * 16u + xt / 4u][xt % 4u] = v.z;
            xs[(kk + 3u) * 16u + xt / 4u][xt % 4u] = v.w;
        }
        let r = r0 + wr;
        if (r < p.rows) {
            decode_step(r * (p.row_bytes / 16u), s, wr, wq);
        } else {
            for (var i = 0u; i < 8u; i++) {
                put_w(wq * 8u + i, wr, 0.0);
                put_w(32u + wq * 8u + i, wr, 0.0);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 64u; kk++) {
            let wv = ws[kk * 16u + tx];
            let xv = xs[kk * 16u + ty];
            acc0 += xv.x * wv;
            acc1 += xv.y * wv;
            acc2 += xv.z * wv;
            acc3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let row0 = r0 + tx * 4u;
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let tok = t0 + ty * 4u + i;
        if (tok < p.m) {
            for (var b = 0u; b < 4u; b++) {
                if (row0 + b < p.rows) { y[tok * p.n + p.row0 + row0 + b] = accs[i][b]; }
            }
        }
    }
}
"#;

fn source_rows(dtype: GgmlType, m_tile: u32) -> Option<String> {
    let (elems, bytes, dequant) = layout(dtype)?;
    let head = COMMON
        .replace("THREADS_X_MTILE", &(THREADS * m_tile).to_string());
    let body = BODY
        .replace("THREADS", &format!("{THREADS}u"))
        .replace("MTILE", &format!("{m_tile}u"))
        .replace("SUBS", &format!("{}u", elems / 32))
        .replace("BLOCK_BYTES", &format!("{bytes}u"));
    // `@workgroup_size` takes a plain literal.
    let body = body.replace(&format!("@workgroup_size({THREADS}u)"), &format!("@workgroup_size({THREADS})"));
    let helper = if matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K) { SCALE_MIN_K4 } else { "" };
    Some(format!("{head}\n{helper}\n{dequant}\n{body}"))
}

// Each `dequant(bb, sub)`: `bb` is the block's first byte, `sub` which 32 of its
// values to produce.

const Q4_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    // (on the GPU its nibbles after a 2-byte gap: `padded_block`)
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 4u + i);
        v[i] = (f32(q & 15u) - 8.0) * d;
        v[i + 16u] = (f32(q >> 4u) - 8.0) * d;
    }
}
"#;

const Q4_1: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let mn = f16at(bb + 2u);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 4u + i);
        v[i] = f32(q & 15u) * d + mn;
        v[i + 16u] = f32(q >> 4u) * d + mn;
    }
}
"#;

const Q5_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let qh = u32at(bb + 2u);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 6u + i);
        let lo = (q & 15u) | (((qh >> i) & 1u) << 4u);
        let hi = (q >> 4u) | (((qh >> (i + 16u)) & 1u) << 4u);
        v[i] = (f32(lo) - 16.0) * d;
        v[i + 16u] = (f32(hi) - 16.0) * d;
    }
}
"#;

const Q5_1: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let mn = f16at(bb + 2u);
    let qh = u32at(bb + 4u);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 8u + i);
        let lo = (q & 15u) | (((qh >> i) & 1u) << 4u);
        let hi = (q >> 4u) | (((qh >> (i + 16u)) & 1u) << 4u);
        v[i] = f32(lo) * d + mn;
        v[i + 16u] = f32(hi) * d + mn;
    }
}
"#;

const Q8_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    for (var i = 0u; i < 32u; i++) { v[i] = i8of(byte(bb + 2u + i)) * d; }
}
"#;

const IQ4_NL: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 2u + i);
        v[i] = d * KVALUES[q & 15u];
        v[i + 16u] = d * KVALUES[q >> 4u];
    }
}
"#;

// Q4_K / Q5_K 6-bit scale and min for sub-block j (upstream get_scale_min_k4).
const SCALE_MIN_K4: &str = r#"
fn scale_min(s: u32, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(byte(s + j) & 63u), f32(byte(s + j + 4u) & 63u));
    }
    let sc = (byte(s + j + 4u) & 15u) | ((byte(s + j - 4u) >> 6u) << 4u);
    let mn = (byte(s + j + 4u) >> 4u) | ((byte(s + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(mn));
}
"#;

const Q4_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let dmin = f16at(bb + 2u);
    let sm = scale_min(bb + 4u, sub);
    let d1 = d * sm.x;
    let m1 = dmin * sm.y;
    let q_off = bb + 16u + (sub / 2u) * 32u;
    let shift = (sub & 1u) * 4u;
    for (var l = 0u; l < 32u; l++) {
        v[l] = d1 * f32((byte(q_off + l) >> shift) & 15u) - m1;
    }
}
"#;

const Q5_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let dmin = f16at(bb + 2u);
    let sm = scale_min(bb + 4u, sub);
    let d1 = d * sm.x;
    let m1 = dmin * sm.y;
    let q_off = bb + 48u + (sub / 2u) * 32u;
    let shift = (sub & 1u) * 4u;
    let mask = 1u << sub;
    for (var l = 0u; l < 32u; l++) {
        let high = select(0u, 16u, (byte(bb + 16u + l) & mask) != 0u);
        v[l] = d1 * f32(((byte(q_off + l) >> shift) & 15u) + high) - m1;
    }
}
"#;

// Q6_K: sub-block `sub` is quarter `sub % 4` of half `sub / 4`.
const Q6_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let h = sub / 4u;
    let qd = sub % 4u;
    let d = f16at(bb + 208u);
    let ql = bb + h * 64u + (qd & 1u) * 32u;
    let qh = bb + 128u + h * 32u;
    let lshift = select(0u, 4u, qd >= 2u);
    let hshift = 2u * qd;
    let sc = bb + 192u + h * 8u + 2u * qd;
    for (var l = 0u; l < 32u; l++) {
        let q = ((byte(ql + l) >> lshift) & 15u) | (((byte(qh + l) >> hshift) & 3u) << 4u);
        v[l] = d * i8of(byte(sc + l / 16u)) * (f32(q) - 32.0);
    }
}
"#;

// Q2_K: sub = 4 * outer + j; halves of 16 values carry their own scale byte.
const Q2_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb + 80u);
    let dmin = f16at(bb + 82u);
    let q_base = bb + 16u + (sub / 4u) * 32u;
    let shift = 2u * (sub % 4u);
    for (var half = 0u; half < 2u; half++) {
        let sc = byte(bb + 2u * sub + half);
        let dl = d * f32(sc & 15u);
        let ml = dmin * f32(sc >> 4u);
        for (var l = 0u; l < 16u; l++) {
            let q = (byte(q_base + half * 16u + l) >> shift) & 3u;
            v[half * 16u + l] = dl * f32(q) - ml;
        }
    }
}
"#;

// Q3_K: the 12 packed scale bytes unpack to 16 six-bit scales (upstream's aux
// shuffle); the high bit of each 3-bit value is in hmask.
const Q3_K: &str = r#"
fn q3_scale(s: u32, k: u32) -> f32 {
    let a0 = u32at(s);
    let a1 = u32at(s + 4u);
    let tmp = u32at(s + 8u);
    var word: u32;
    switch (k / 4u) {
        case 0u: { word = (a0 & 0x0f0f0f0fu) | (((tmp >> 0u) & 0x03030303u) << 4u); }
        case 1u: { word = (a1 & 0x0f0f0f0fu) | (((tmp >> 2u) & 0x03030303u) << 4u); }
        case 2u: { word = ((a0 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 4u) & 0x03030303u) << 4u); }
        default: { word = ((a1 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 6u) & 0x03030303u) << 4u); }
    }
    return i8of((word >> ((k % 4u) * 8u)) & 0xffu);
}
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb + 108u);
    let q_base = bb + 32u + (sub / 4u) * 32u;
    let shift = 2u * (sub % 4u);
    let m = 1u << sub;
    for (var half = 0u; half < 2u; half++) {
        let dl = d * (q3_scale(bb + 96u, 2u * sub + half) - 32.0);
        for (var l = 0u; l < 16u; l++) {
            let i = half * 16u + l;
            let raw = i32((byte(q_base + i) >> shift) & 3u);
            let q = raw - select(4, 0, (byte(bb + i) & m) != 0u);
            v[i] = dl * f32(q);
        }
    }
}
"#;

const IQ4_XS: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let scales_h = u16at(bb + 2u);
    let lo4 = (byte(bb + 4u + sub / 2u) >> (4u * (sub & 1u))) & 15u;
    let hi2 = ((scales_h >> (2u * sub)) & 3u) << 4u;
    let dl = d * (f32(lo4 | hi2) - 32.0);
    let q_off = bb + 8u + sub * 16u;
    for (var j = 0u; j < 16u; j++) {
        let q = byte(q_off + j);
        v[j] = dl * KVALUES[q & 15u];
        v[j + 16u] = dl * KVALUES[q >> 4u];
    }
}
"#;
