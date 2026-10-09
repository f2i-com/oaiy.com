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

mod coop;
mod dequant;
mod few;
mod int8;
mod tiled;

// (the kernels the crate takes, and what the files make for each other)
pub use {coop::*, few::*, int8::*};
use {dequant::*, tiled::*};

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
        GgmlType::Q2_0 => (64, 18, Q2_0),
        GgmlType::Q2_K => (256, 84, Q2_K),
        GgmlType::Q3_K => (256, 112, Q3_K),
        GgmlType::Q4_K => (256, 144, Q4_K),
        GgmlType::Q5_K => (256, 176, Q5_K),
        GgmlType::Q6_K => (256, 210, Q6_K),
        GgmlType::IQ4_XS => (256, 136, IQ4_XS),
        // The grid types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M) are NOT on the GPU: `iq_grid_dequant`
        // decodes them rightly (the CPU's values at 1 to 70 rows), but with its tables WGSL constants indexed at run
        // time, which a compiler copies into the function at each call: the cross-type test took 11 s where 0.7, and
        // the same tables in the routed experts' kernel ran one dispatch past Windows' two seconds and reset the
        // driver (2026-10-07). Until the tables are a storage buffer's, a matrix of these types stays on the host
        // (`WgpuBackend::supports` says no; OAIY_GRID_ON_GPU=1 the constants' kernel, for small shapes only).
        GgmlType::IQ2_XXS | GgmlType::IQ2_XS | GgmlType::IQ2_S | GgmlType::IQ3_XXS | GgmlType::IQ3_S | GgmlType::IQ1_S | GgmlType::IQ1_M
            if std::env::var_os("OAIY_GRID_ON_GPU").is_some() =>
        {
            (256, dtype.type_size() as u32, iq_grid_dequant(dtype)?)
        }
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
        GgmlType::Q8_0 => return Some(Q80_TILED.to_string()),
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

/// An attention cache's rows (`[position][K of every KV head, then V]`, f32) as f16 for the tensor cores' one-pass
/// attention, a fragment at a time: its keys, then its values, each `[KV head][16 positions][16 of the head's
/// width]` a run of 256 halves `[position][dim]`, so a fragment's load is one run of 512 bytes (in place its 16
/// positions' pieces were 32 bytes each a row apart, 4 KB: the keys' and values' loads 4.3 of the kernel's 10 ms a
/// layer for 2,048 queries over 15,360 positions, and with these runs the kernel 7.3). Positions from `p[0].z` to the
/// padded `p[0].w` (a multiple of 128) zero. `p[0]`: KV heads, the head's width, positions, padded positions.
pub const KV_F16_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> unused: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n_kv = p[0].x;
    let hd = p[0].y;
    // the fragments a KV head has, and a region (the keys, the values); a fragment 128 words of two halves
    let per_head = (p[0].w / 16u) * (hd / 16u);
    let per_region = n_kv * per_head;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= 2u * per_region * 128u) { return; }
    let frag = i / 128u;
    let within = i % 128u;
    let region = frag / per_region;
    let kh = (frag % per_region) / per_head;
    let rem = frag % per_head;
    let key = (rem / (hd / 16u)) * 16u + within / 8u;
    let dim = (rem % (hd / 16u)) * 16u + (within % 8u) * 2u;
    var v = vec2<f32>(0.0);
    if (key < p[0].z) { v = x2[(key * 2u * n_kv * hd + region * n_kv * hd + kh * hd + dim) / 2u]; }
    q[i] = pack2x16float(v);
}
"#;

/// [`X_F16`] for [`coop_tiled`]: the tokens' rows as f16 a step's 32 of `k` at a time, each step's for every (padded)
/// token together (a token's 32 after the one before's), so a step's loads of a tile's tokens are one run (a warp's
/// 1 KB, where the rows in place made it 16 runs of 64 bytes: the matmul 0.89 ms where 0.96); a last step short of 32
/// (`k` even, not of 32) padded with zeros. `p[0]`: k, rows, padded rows.
pub const X_F16_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> unused: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k2 = p[0].x / 2u;
    let padded = p[0].z;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= padded * ((k2 + 15u) / 16u) * 16u) { return; }
    // pair j of step s's 16, of token t
    let j = i % 16u;
    let t = (i / 16u) % padded;
    let s = i / (16u * padded);
    var v = vec2<f32>(0.0);
    if (t < p[0].y && s * 16u + j < k2) { v = x2[t * k2 + s * 16u + j]; }
    q[i] = pack2x16float(v);
}
"#;

/// The words [`X_F16_TILED`] makes of `rows` tokens of `k` (padded to [`COOP_TILE`] tokens and to a step of 32).
pub fn x_f16_tiled_words(rows: usize, k: usize) -> usize {
    rows.div_ceil(COOP_TILE as usize) * COOP_TILE as usize * k.div_ceil(32) * 16
}

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
