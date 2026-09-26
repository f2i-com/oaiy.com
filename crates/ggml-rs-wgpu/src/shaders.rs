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
/// Threads per workgroup: they split one weight row's 32-element sub-blocks.
pub const THREADS: u32 = 64;

/// Elements per block, bytes per block, and the WGSL `dequant` for a type.
pub fn layout(dtype: GgmlType) -> Option<(u32, u32, &'static str)> {
    Some(match dtype {
        GgmlType::Q4_0 => (32, 18, Q4_0),
        GgmlType::Q4_1 => (32, 20, Q4_1),
        GgmlType::Q5_0 => (32, 22, Q5_0),
        GgmlType::Q5_1 => (32, 24, Q5_1),
        GgmlType::Q8_0 => (32, 34, Q8_0),
        GgmlType::IQ4_NL => (32, 18, IQ4_NL),
        GgmlType::Q2_K => (256, 84, Q2_K),
        GgmlType::Q3_K => (256, 110, Q3_K),
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

/// The complete shader for `dtype`.
pub fn source(dtype: GgmlType) -> Option<String> {
    let (elems, bytes, dequant) = layout(dtype)?;
    let head = COMMON
        .replace("THREADS_X_MTILE", &(THREADS * M_TILE).to_string());
    let body = BODY
        .replace("THREADS", &format!("{THREADS}u"))
        .replace("MTILE", &format!("{M_TILE}u"))
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
