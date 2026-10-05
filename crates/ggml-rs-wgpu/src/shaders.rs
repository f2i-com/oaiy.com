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
/// seven vec4s ([`Q3K_DECODE`] reads them so).
pub fn padded_block(dtype: GgmlType) -> Option<(usize, usize)> {
    match dtype {
        GgmlType::Q3_K => Some((110, 112)),
        _ => None,
    }
}

/// `bytes` (whole blocks of `from` bytes) with each block padded with zeros to `to` bytes, or back (`to < from`).
pub fn pad_blocks(bytes: &[u8], from: usize, to: usize) -> Vec<u8> {
    assert!(bytes.len() % from == 0, "pad_blocks: {} bytes are not blocks of {from}", bytes.len());
    let keep = from.min(to);
    let mut out = vec![0u8; bytes.len() / from * to];
    for (src, dst) in bytes.chunks_exact(from).zip(out.chunks_exact_mut(to)) {
        dst[..keep].copy_from_slice(&src[..keep]);
    }
    out
}

/// Elements per block, bytes per block on the GPU ([`padded_block`]), and the WGSL `dequant` for a type.
pub fn layout(dtype: GgmlType) -> Option<(u32, u32, &'static str)> {
    Some(match dtype {
        GgmlType::Q4_0 => (32, 18, Q4_0),
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
/// K-quants have kernels of their own that read the weights wide ([`Q3K_DECODE`], [`Q4K_DECODE`], [`Q5K_DECODE`],
/// [`Q6K_DECODE`]).
pub fn source_decode(dtype: GgmlType) -> Option<String> {
    match dtype {
        GgmlType::Q3_K => Some(Q3K_DECODE.to_string()),
        GgmlType::Q4_K => Some(Q4K_DECODE.to_string()),
        GgmlType::Q5_K => Some(Q5K_DECODE.to_string()),
        GgmlType::Q6_K => Some(Q6K_DECODE.to_string()),
        _ => source_rows(dtype, 1),
    }
}

/// Rows of `x` a workgroup of the multi-row kernels takes ([`source_multi`]).
pub const MULTI_ROWS: u32 = 8;

/// Rows of `x` up to which the multi-row kernels go before the tiled one: they read `x` again for every weight row,
/// where the tiled kernel keeps 64 rows of it in workgroup memory but costs the same for a few rows as for 64
/// (Qwen3.8 27B's chunk of 22 tokens in 186 ms with them and in 199 with it, of 512 in 4.0 s and in 1.3).
pub const MULTI_MAX: usize = 24;

/// `dtype`'s multi-row kernel: the wide K-quants'.
pub fn source_multi(dtype: GgmlType) -> Option<String> {
    Some(match dtype {
        GgmlType::Q3_K => Q3K_MULTI,
        GgmlType::Q4_K => Q4K_MULTI,
        GgmlType::Q5_K => Q5K_MULTI,
        GgmlType::Q6_K => Q6K_MULTI,
        _ => return None,
    }.to_string())
}

/// Which kernel `m` rows of `x` take with `dtype`'s weights: 2 the decode kernel (one row), 3 the multi-row one (the
/// wide K-quants', [`source_multi`]), 1 the tiled one, 0 the one-row kernel for a few rows.
pub fn kind(dtype: GgmlType, m: usize) -> u8 {
    if m == 1 {
        2
    } else if m <= MULTI_MAX && matches!(dtype, GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K) {
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
        2 => {
            let groups = rows.div_ceil(decode_rows_per_group(dtype));
            (groups.min(65535), groups.div_ceil(65535), 1)
        }
        3 => {
            let groups = rows.div_ceil(8);
            ((m as u32).div_ceil(MULTI_ROWS), groups.min(65535), groups.div_ceil(65535))
        }
        _ => (rows.min(65535), rows.div_ceil(65535), (m as u32).div_ceil(M_TILE)),
    }
}

/// Weight rows a workgroup of `dtype`'s decode kernel takes (its grid is the rows over this).
pub fn decode_rows_per_group(dtype: GgmlType) -> u32 {
    match dtype {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => 8,
        _ => 1,
    }
}

/// Q3_K for one row of `x`, read wide: its blocks are padded to 112 bytes on the GPU (seven vec4s, [`pad_blocks`]),
/// 16 weight rows a workgroup of 256, 16 lanes a row; a lane takes 64 weights of a block (four of its 16-weight runs,
/// which share their low bits' 16 bytes and their high bits' 16 bytes: a vec4 each) and the block's scales and `d` (a
/// vec4), and `x` four at a time, and adds `d * (scale - 32) * (dot(q + 4, x) - 4 * sum(x))` for each run. Its
/// 110-byte blocks read a word at a time (every other one two bytes into a word) went at 760 GB/s.
const Q3K_DECODE: &str = r#"
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

var<workgroup> partial: array<f32, 256>;

fn bytes4(v: u32) -> vec4<f32> {
    return vec4<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.x + wg.y * 65535u) * 8u + (li >> 5u);
    var acc = 0.0;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row4 = r * (p.row_bytes / 16u);
        for (var c = lane; c < blocks * 8u; c += 32u) {
            let blk = c / 8u;
            let t = c % 8u;
            // the block's half (128 weights), the run within each 32 of it (16 weights), and the pair of shifts
            let h = t / 4u;
            let g = (t / 2u) % 2u;
            let j0 = 2u * (t % 2u);
            let b4 = row4 + blk * 7u;
            let hm = w4[b4 + g];
            let qs = w4[b4 + 2u + 2u * h + g];
            let sd = w4[b4 + 6u];
            let d = unpack2x16float(sd.w & 0xffffu).x;
            let sa = ((sd.x >> (4u * h)) & 0x0f0f0f0fu) | (((sd.z >> (4u * h)) & 0x03030303u) << 4u);
            let sb = ((sd.y >> (4u * h)) & 0x0f0f0f0fu) | (((sd.z >> (4u * h + 2u)) & 0x03030303u) << 4u);
            // runs j0 and j0 + 1: bytes g, g + 2 of sa (j0 0) or of sb (j0 2)
            let sw = select(sa, sb, j0 == 2u);
            let gs = 8u * g;
            let s0 = f32((sw >> gs) & 255u) - 32.0;
            let s1 = f32((sw >> (gs + 16u)) & 255u) - 32.0;
            let xa = (blk * 256u + h * 128u + j0 * 32u + g * 16u) / 4u;
            var dq0 = 0.0;
            var dq1 = 0.0;
            var sx0 = 0.0;
            var sx1 = 0.0;
            for (var wi = 0u; wi < 4u; wi++) {
                let qw = qs[wi];
                let hw = hm[wi];
                let v0 = ((qw >> (2u * j0)) & 0x03030303u) | (((hw >> (j0 + 4u * h)) & 0x01010101u) << 2u);
                let v1 = ((qw >> (2u * j0 + 2u)) & 0x03030303u) | (((hw >> (j0 + 1u + 4u * h)) & 0x01010101u) << 2u);
                let a = x4[xa + wi];
                let b = x4[xa + 8u + wi];
                dq0 += dot(bytes4(v0), a);
                sx0 += a.x + a.y + a.z + a.w;
                dq1 += dot(bytes4(v1), b);
                sx1 += b.x + b.y + b.z + b.w;
            }
            acc += d * (s0 * (dq0 - 4.0 * sx0) + s1 * (dq1 - 4.0 * sx1));
        }
    }
    partial[li] = acc;
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) { partial[li] += partial[li + st]; }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) { y[p.row0 + r] = partial[li]; }
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

/// The wide K-quant kernels for several rows of `x` ([`kind`] 3): each a decode kernel's lanes (the same loads and
/// sums), every weight a lane decodes applied to [`MULTI_ROWS`] rows of `x`, a workgroup an 8-row group of `x` by 8
/// weight rows. A prompt's few rows read the weights once where the one-row kernel read them a byte at a time, and a
/// chunk of hundreds once an 8-row group, where the tiled kernel took 64 tokens and 64 weight rows a workgroup.
const Q3K_MULTI: &str = r#"
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

const MR: u32 = 8u;
var<workgroup> partial: array<f32, 2048>;

fn bytes4(v: u32) -> vec4<f32> {
    return vec4<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.y + wg.z * 65535u) * 8u + (li >> 5u);
    let m0 = wg.x * MR;
    let mn = min(MR, p.m - m0);
    let k4 = p.k / 4u;
    var acc: array<f32, 8>;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row4 = r * (p.row_bytes / 16u);
        for (var c = lane; c < blocks * 8u; c += 32u) {
            let blk = c / 8u;
            let t = c % 8u;
            let h = t / 4u;
            let g = (t / 2u) % 2u;
            let j0 = 2u * (t % 2u);
            let b4 = row4 + blk * 7u;
            let hm = w4[b4 + g];
            let qs = w4[b4 + 2u + 2u * h + g];
            let sd = w4[b4 + 6u];
            let d = unpack2x16float(sd.w & 0xffffu).x;
            let sa = ((sd.x >> (4u * h)) & 0x0f0f0f0fu) | (((sd.z >> (4u * h)) & 0x03030303u) << 4u);
            let sb = ((sd.y >> (4u * h)) & 0x0f0f0f0fu) | (((sd.z >> (4u * h + 2u)) & 0x03030303u) << 4u);
            let sw = select(sa, sb, j0 == 2u);
            let gs = 8u * g;
            let s0 = f32((sw >> gs) & 255u) - 32.0;
            let s1 = f32((sw >> (gs + 16u)) & 255u) - 32.0;
            let xa = (blk * 256u + h * 128u + j0 * 32u + g * 16u) / 4u;
            var q0: array<vec4<f32>, 4>;
            var q1: array<vec4<f32>, 4>;
            for (var wi = 0u; wi < 4u; wi++) {
                let qw = qs[wi];
                let hw = hm[wi];
                q0[wi] = bytes4(((qw >> (2u * j0)) & 0x03030303u) | (((hw >> (j0 + 4u * h)) & 0x01010101u) << 2u));
                q1[wi] = bytes4(((qw >> (2u * j0 + 2u)) & 0x03030303u) | (((hw >> (j0 + 1u + 4u * h)) & 0x01010101u) << 2u));
            }
            for (var mm = 0u; mm < MR; mm++) {
                if (mm < mn) {
                    let xo = (m0 + mm) * k4;
                    var dq0 = 0.0;
                    var dq1 = 0.0;
                    var sx0 = 0.0;
                    var sx1 = 0.0;
                    for (var wi = 0u; wi < 4u; wi++) {
                        let a = x4[xo + xa + wi];
                        let b = x4[xo + xa + 8u + wi];
                        dq0 += dot(q0[wi], a);
                        sx0 += a.x + a.y + a.z + a.w;
                        dq1 += dot(q1[wi], b);
                        sx1 += b.x + b.y + b.z + b.w;
                    }
                    acc[mm] += d * (s0 * (dq0 - 4.0 * sx0) + s1 * (dq1 - 4.0 * sx1));
                }
            }
        }
    }
    for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] = acc[mm]; }
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) {
            for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] += partial[mm * 256u + li + st]; }
        }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) {
        for (var mm = 0u; mm < mn; mm++) { y[(m0 + mm) * p.n + p.row0 + r] = partial[mm * 256u + li]; }
    }
}
"#;

const Q4K_MULTI: &str = r#"
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

const MR: u32 = 8u;
var<workgroup> partial: array<f32, 2048>;

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.y + wg.z * 65535u) * 8u + (li >> 5u);
    let m0 = wg.x * MR;
    let mn = min(MR, p.m - m0);
    let k4 = p.k / 4u;
    var acc: array<f32, 8>;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row4 = r * (p.row_bytes / 16u);
        for (var c = lane; c < blocks * 8u; c += 32u) {
            let blk = c / 8u;
            let j = c % 8u;
            let base4 = row4 + blk * 9u;
            let h = w4[base4];
            let q = w4[base4 + 1u + j];
            let dm = unpack2x16float(h.x);
            let pair = j / 2u;
            let half = j % 2u;
            let slo = scale_min(h, 2u * pair);
            let shi = scale_min(h, 2u * pair + 1u);
            let xa = (blk * 256u + 2u * pair * 32u + half * 16u) / 4u;
            let xb = xa + 8u;
            var lo: array<vec4<f32>, 4>;
            var hi: array<vec4<f32>, 4>;
            for (var wi = 0u; wi < 4u; wi++) {
                let word = q[wi];
                lo[wi] = vec4<f32>(f32(word & 15u), f32((word >> 8u) & 15u), f32((word >> 16u) & 15u), f32((word >> 24u) & 15u));
                hi[wi] = vec4<f32>(f32((word >> 4u) & 15u), f32((word >> 12u) & 15u), f32((word >> 20u) & 15u), f32((word >> 28u) & 15u));
            }
            for (var mm = 0u; mm < MR; mm++) {
                if (mm < mn) {
                    let xo = (m0 + mm) * k4;
                    var dot_lo = 0.0;
                    var sum_lo = 0.0;
                    var dot_hi = 0.0;
                    var sum_hi = 0.0;
                    for (var wi = 0u; wi < 4u; wi++) {
                        let a = x4[xo + xa + wi];
                        let b = x4[xo + xb + wi];
                        dot_lo += dot(lo[wi], a);
                        sum_lo += a.x + a.y + a.z + a.w;
                        dot_hi += dot(hi[wi], b);
                        sum_hi += b.x + b.y + b.z + b.w;
                    }
                    acc[mm] += dm.x * slo.x * dot_lo - dm.y * slo.y * sum_lo + dm.x * shi.x * dot_hi - dm.y * shi.y * sum_hi;
                }
            }
        }
    }
    for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] = acc[mm]; }
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) {
            for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] += partial[mm * 256u + li + st]; }
        }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) {
        for (var mm = 0u; mm < mn; mm++) { y[(m0 + mm) * p.n + p.row0 + r] = partial[mm * 256u + li]; }
    }
}
"#;

const Q5K_MULTI: &str = r#"
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

const MR: u32 = 8u;
var<workgroup> partial: array<f32, 2048>;

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

// The four bytes' bit `s` of `v` (a byte each), as 0 or 16.
fn high4(v: u32, s: u32) -> vec4<f32> {
    return 16.0 * vec4<f32>(f32((v >> s) & 1u), f32((v >> (s + 8u)) & 1u), f32((v >> (s + 16u)) & 1u), f32((v >> (s + 24u)) & 1u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.y + wg.z * 65535u) * 8u + (li >> 5u);
    let m0 = wg.x * MR;
    let mn = min(MR, p.m - m0);
    let k4 = p.k / 4u;
    var acc: array<f32, 8>;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row4 = r * (p.row_bytes / 16u);
        for (var c = lane; c < blocks * 8u; c += 32u) {
            let blk = c / 8u;
            let j = c % 8u;
            let base4 = row4 + blk * 11u;
            let h = w4[base4];
            let pair = j / 2u;
            let half = j % 2u;
            let qh = w4[base4 + 1u + half];
            let q = w4[base4 + 3u + j];
            let dm = unpack2x16float(h.x);
            let slo = scale_min(h, 2u * pair);
            let shi = scale_min(h, 2u * pair + 1u);
            let xa = (blk * 256u + 2u * pair * 32u + half * 16u) / 4u;
            let xb = xa + 8u;
            let sl = 2u * pair;
            var lo: array<vec4<f32>, 4>;
            var hi: array<vec4<f32>, 4>;
            for (var wi = 0u; wi < 4u; wi++) {
                let word = q[wi];
                let hw = qh[wi];
                lo[wi] = vec4<f32>(f32(word & 15u), f32((word >> 8u) & 15u), f32((word >> 16u) & 15u), f32((word >> 24u) & 15u)) + high4(hw, sl);
                hi[wi] = vec4<f32>(f32((word >> 4u) & 15u), f32((word >> 12u) & 15u), f32((word >> 20u) & 15u), f32((word >> 28u) & 15u)) + high4(hw, sl + 1u);
            }
            for (var mm = 0u; mm < MR; mm++) {
                if (mm < mn) {
                    let xo = (m0 + mm) * k4;
                    var dot_lo = 0.0;
                    var sum_lo = 0.0;
                    var dot_hi = 0.0;
                    var sum_hi = 0.0;
                    for (var wi = 0u; wi < 4u; wi++) {
                        let a = x4[xo + xa + wi];
                        let b = x4[xo + xb + wi];
                        dot_lo += dot(lo[wi], a);
                        sum_lo += a.x + a.y + a.z + a.w;
                        dot_hi += dot(hi[wi], b);
                        sum_hi += b.x + b.y + b.z + b.w;
                    }
                    acc[mm] += dm.x * slo.x * dot_lo - dm.y * slo.y * sum_lo + dm.x * shi.x * dot_hi - dm.y * shi.y * sum_hi;
                }
            }
        }
    }
    for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] = acc[mm]; }
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) {
            for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] += partial[mm * 256u + li + st]; }
        }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) {
        for (var mm = 0u; mm < mn; mm++) { y[(m0 + mm) * p.n + p.row0 + r] = partial[mm * 256u + li]; }
    }
}
"#;

const Q6K_MULTI: &str = r#"
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

const MR: u32 = 8u;
var<workgroup> partial: array<f32, 2048>;

fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }

// The four bytes at `o`, an even offset (a word, or the halves of two).
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}

fn bytes4(v: u32) -> vec4<u32> {
    return vec4<u32>(v & 255u, (v >> 8u) & 255u, (v >> 16u) & 255u, v >> 24u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.y + wg.z * 65535u) * 8u + (li >> 5u);
    let m0 = wg.x * MR;
    let mn = min(MR, p.m - m0);
    let k4 = p.k / 4u;
    var acc: array<f32, 8>;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row_base = r * p.row_bytes;
        for (var c = lane; c < blocks * 16u; c += 32u) {
            let blk = c / 16u;
            let sub = (c % 16u) / 2u;
            let hf = c % 2u;
            let bb = row_base + blk * 210u;
            let h = sub / 4u;
            let qd = sub % 4u;
            let ql = bb + h * 64u + (qd & 1u) * 32u + hf * 16u;
            let qh = bb + 128u + h * 32u + hf * 16u;
            let lshift = select(0u, 4u, qd >= 2u);
            let hshift = 2u * qd;
            let d = unpack2x16float(byte(bb + 208u) | (byte(bb + 209u) << 8u)).x;
            let scb = byte(bb + 192u + h * 8u + 2u * qd + hf);
            let sc = f32(i32(scb) - select(0, 256, scb >= 128u));
            let xb = (blk * 256u + sub * 32u + hf * 16u) / 4u;
            var q: array<vec4<f32>, 4>;
            for (var wi = 0u; wi < 4u; wi++) {
                let lo = bytes4(word_at(ql + wi * 4u));
                let hi = bytes4(word_at(qh + wi * 4u));
                q[wi] = vec4<f32>(((lo >> vec4<u32>(lshift)) & vec4<u32>(15u)) | (((hi >> vec4<u32>(hshift)) & vec4<u32>(3u)) << vec4<u32>(4u)));
            }
            for (var mm = 0u; mm < MR; mm++) {
                if (mm < mn) {
                    let xo = (m0 + mm) * k4;
                    var dq = 0.0;
                    var sx = 0.0;
                    for (var wi = 0u; wi < 4u; wi++) {
                        let a = x4[xo + xb + wi];
                        dq += dot(q[wi], a);
                        sx += a.x + a.y + a.z + a.w;
                    }
                    acc[mm] += d * sc * (dq - 32.0 * sx);
                }
            }
        }
    }
    for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] = acc[mm]; }
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) {
            for (var mm = 0u; mm < MR; mm++) { partial[mm * 256u + li] += partial[mm * 256u + li + st]; }
        }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) {
        for (var mm = 0u; mm < mn; mm++) { y[(m0 + mm) * p.n + p.row0 + r] = partial[mm * 256u + li]; }
    }
}
"#;

/// Q5_K for one row of `x`, read wide, as [`Q4K_DECODE`]: 8 weight rows a workgroup of 256, 32 lanes a row; a lane
/// takes a 16-byte run of a block's low bits (32 weights: 16 of a sub-block's low nibbles and 16 of the next one's
/// high), the 16 bytes of high bits beside them and the block's header (its scales), a vec4 load each, and `x` four at
/// a time, and adds `d * dot(q, x) - m * sum(x)` for each half. A block is 176 bytes, 11 vec4s.
const Q5K_DECODE: &str = r#"
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

var<workgroup> partial: array<f32, 256>;

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

// The four bytes' bit `s` of `v` (a byte each), as 0 or 16.
fn high4(v: u32, s: u32) -> vec4<f32> {
    return 16.0 * vec4<f32>(f32((v >> s) & 1u), f32((v >> (s + 8u)) & 1u), f32((v >> (s + 16u)) & 1u), f32((v >> (s + 24u)) & 1u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.x + wg.y * 65535u) * 8u + (li >> 5u);
    var acc = 0.0;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row4 = r * (p.row_bytes / 16u);
        for (var c = lane; c < blocks * 8u; c += 32u) {
            let blk = c / 8u;
            let j = c % 8u;
            let base4 = row4 + blk * 11u;
            let h = w4[base4];
            let pair = j / 2u;
            let half = j % 2u;
            let qh = w4[base4 + 1u + half];
            let q = w4[base4 + 3u + j];
            let dm = unpack2x16float(h.x);
            let slo = scale_min(h, 2u * pair);
            let shi = scale_min(h, 2u * pair + 1u);
            // x: sub-block 2*pair's 16 at half * 16, and sub-block 2*pair + 1's
            let xa = (blk * 256u + 2u * pair * 32u + half * 16u) / 4u;
            let xb = xa + 8u;
            let sl = 2u * pair;
            var dot_lo = 0.0;
            var sum_lo = 0.0;
            var dot_hi = 0.0;
            var sum_hi = 0.0;
            for (var wi = 0u; wi < 4u; wi++) {
                let word = q[wi];
                let hw = qh[wi];
                let a = x4[xa + wi];
                let b = x4[xb + wi];
                let lo = vec4<f32>(f32(word & 15u), f32((word >> 8u) & 15u), f32((word >> 16u) & 15u), f32((word >> 24u) & 15u)) + high4(hw, sl);
                let hi = vec4<f32>(f32((word >> 4u) & 15u), f32((word >> 12u) & 15u), f32((word >> 20u) & 15u), f32((word >> 28u) & 15u)) + high4(hw, sl + 1u);
                dot_lo += dot(lo, a);
                sum_lo += a.x + a.y + a.z + a.w;
                dot_hi += dot(hi, b);
                sum_hi += b.x + b.y + b.z + b.w;
            }
            acc += dm.x * slo.x * dot_lo - dm.y * slo.y * sum_lo + dm.x * shi.x * dot_hi - dm.y * shi.y * sum_hi;
        }
    }
    partial[li] = acc;
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) { partial[li] += partial[li + st]; }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) { y[p.row0 + r] = partial[li]; }
}
"#;

/// Q6_K for one row of `x`, read a word at a time: 8 weight rows a workgroup of 256, 32 lanes a row; a lane takes half
/// a sub-block (16 weights, one scale): their low bits' 16 bytes and high bits' 16 bytes as four words each (a block is
/// 210 bytes, so every other one starts two bytes into a word and its words are put together from two), and `x` four
/// at a time, and adds `d * scale * (dot(q, x) - 32 * sum(x))`.
const Q6K_DECODE: &str = r#"
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

var<workgroup> partial: array<f32, 256>;

fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }

// The four bytes at `o`, an even offset (a word, or the halves of two).
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}

fn bytes4(v: u32) -> vec4<u32> {
    return vec4<u32>(v & 255u, (v >> 8u) & 255u, (v >> 16u) & 255u, v >> 24u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.x + wg.y * 65535u) * 8u + (li >> 5u);
    var acc = 0.0;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row_base = r * p.row_bytes;
        for (var c = lane; c < blocks * 16u; c += 32u) {
            let blk = c / 16u;
            let sub = (c % 16u) / 2u;
            let hf = c % 2u;
            let bb = row_base + blk * 210u;
            let h = sub / 4u;
            let qd = sub % 4u;
            let ql = bb + h * 64u + (qd & 1u) * 32u + hf * 16u;
            let qh = bb + 128u + h * 32u + hf * 16u;
            let lshift = select(0u, 4u, qd >= 2u);
            let hshift = 2u * qd;
            let d = unpack2x16float(byte(bb + 208u) | (byte(bb + 209u) << 8u)).x;
            let scb = byte(bb + 192u + h * 8u + 2u * qd + hf);
            let sc = f32(i32(scb) - select(0, 256, scb >= 128u));
            let xb = (blk * 256u + sub * 32u + hf * 16u) / 4u;
            var dq = 0.0;
            var sx = 0.0;
            for (var wi = 0u; wi < 4u; wi++) {
                let lo = bytes4(word_at(ql + wi * 4u));
                let hi = bytes4(word_at(qh + wi * 4u));
                let q = ((lo >> vec4<u32>(lshift)) & vec4<u32>(15u)) | (((hi >> vec4<u32>(hshift)) & vec4<u32>(3u)) << vec4<u32>(4u));
                let a = x4[xb + wi];
                dq += dot(vec4<f32>(q), a);
                sx += a.x + a.y + a.z + a.w;
            }
            acc += d * sc * (dq - 32.0 * sx);
        }
    }
    partial[li] = acc;
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) { partial[li] += partial[li + st]; }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) { y[p.row0 + r] = partial[li]; }
}
"#;

/// Q4_K for one row of `x`, read wide: 8 weight rows a workgroup of 256, 32 lanes a row; a lane takes a 16-byte run of
/// a block's quants (32 weights: 16 of a sub-block's low nibbles and 16 of the next one's high) with one vec4 load,
/// the block's header (its scales) with another, and `x` four at a time, and adds `d * dot(q, x) - m * sum(x)` for
/// each half. The generic kernel read a byte a load, 13 loads a lane for 16 bytes of quants: 165-230 GB/s.
const Q4K_DECODE: &str = r#"
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

var<workgroup> partial: array<f32, 256>;

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let r = (wg.x + wg.y * 65535u) * 8u + (li >> 5u);
    var acc = 0.0;
    if (r < p.rows) {
        let blocks = p.k / 256u;
        let row4 = r * (p.row_bytes / 16u);
        for (var c = lane; c < blocks * 8u; c += 32u) {
            let blk = c / 8u;
            let j = c % 8u;
            let base4 = row4 + blk * 9u;
            let h = w4[base4];
            let q = w4[base4 + 1u + j];
            let dm = unpack2x16float(h.x);
            let pair = j / 2u;
            let half = j % 2u;
            let slo = scale_min(h, 2u * pair);
            let shi = scale_min(h, 2u * pair + 1u);
            // x: sub-block 2*pair's 16 at half * 16, and sub-block 2*pair + 1's
            let xa = (blk * 256u + 2u * pair * 32u + half * 16u) / 4u;
            let xb = xa + 8u;
            var dot_lo = 0.0;
            var sum_lo = 0.0;
            var dot_hi = 0.0;
            var sum_hi = 0.0;
            for (var wi = 0u; wi < 4u; wi++) {
                let word = q[wi];
                let a = x4[xa + wi];
                let b = x4[xb + wi];
                let lo = vec4<f32>(f32(word & 15u), f32((word >> 8u) & 15u), f32((word >> 16u) & 15u), f32((word >> 24u) & 15u));
                let hi = vec4<f32>(f32((word >> 4u) & 15u), f32((word >> 12u) & 15u), f32((word >> 20u) & 15u), f32((word >> 28u) & 15u));
                dot_lo += dot(lo, a);
                sum_lo += a.x + a.y + a.z + a.w;
                dot_hi += dot(hi, b);
                sum_hi += b.x + b.y + b.z + b.w;
            }
            acc += dm.x * slo.x * dot_lo - dm.y * slo.y * sum_lo + dm.x * shi.x * dot_hi - dm.y * shi.y * sum_hi;
        }
    }
    partial[li] = acc;
    workgroupBarrier();
    for (var st = 16u; st > 0u; st /= 2u) {
        if (lane < st) { partial[li] += partial[li + st]; }
        workgroupBarrier();
    }
    if (r < p.rows && lane == 0u) { y[p.row0 + r] = partial[li]; }
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
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 2u + i);
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
