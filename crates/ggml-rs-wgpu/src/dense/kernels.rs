//! The dense kernels' sources: a call's few tokens and a prompt's many, a decode step's one row by kind, and the
//! stages between a call's projections.

use super::*;

const COMMON: &str = r#"
@group(0) @binding(0) var<storage, read> wbuf: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
// k, tokens, the buffer's first row, its rows, the first row asked for, the rows asked for, where its scales start
// (words; bytes for a record), the kind (0 fp8, 1 bf16, 2 mxfp4, 3 a record's mxfp4), and where its weights start
// (words; 0 but in a record).
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 3>;

fn fp8(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    // e4m3fn has no infinity: all ones is NaN, as the reference decodes it.
    if (e == 15u && m == 7u) { return bitcast<f32>(0x7fc00000u); }
    var v: f32;
    if (e == 0u) {
        v = f32(m) * 0.001953125;
    } else {
        v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u));
    }
    return select(v, -v, (b & 128u) != 0u);
}

// e2m1: 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives.
fn fp4(n: u32) -> f32 {
    let m = n & 7u;
    var v = f32(m) * 0.5;
    if (m >= 4u) { v = select(f32(m) - 2.0, 6.0, m == 7u); }
    return select(v, -v, (n & 8u) != 0u);
}

// Weights a word holds: 4 fp8, 2 bf16, 8 e2m1.
fn wide(kind: u32) -> u32 {
    return select(select(4u, 2u, kind == 1u), 8u, kind >= 2u);
}

// The j-th weight of a word.
fn weight(word: u32, kind: u32, j: u32) -> f32 {
    if (kind == 0u) { return fp8((word >> (8u * j)) & 255u); }
    if (kind >= 2u) { return fp4((word >> (4u * j)) & 15u); }
    return select(bitcast<f32>(word & 0xffff0000u), bitcast<f32>(word << 16u), j == 0u);
}

// An e8m0 byte as the reference decodes it: 2^(e - 127), 0 the f32 subnormal 2^-127, 255 NaN.
fn e8m0(e: u32) -> f32 {
    if (e == 255u) { return bitcast<f32>(0x7fc00000u); }
    if (e == 0u) { return bitcast<f32>(0x00400000u); }
    return bitcast<f32>(e << 23u);
}

// The scale of the weights of row `local` (in this buffer) at k column `c`: a 32x32 tile's (fp8), a row's 32 (mxfp4,
// as f32; a record's, as an e8m0 byte), none (bf16).
fn scale(kind: u32, soff: u32, local: u32, c: u32, kb: u32) -> f32 {
    if (kind == 0u) { return bitcast<f32>(wbuf[soff + (local / 32u) * kb + c / 32u]); }
    if (kind == 2u) { return bitcast<f32>(wbuf[soff + local * kb + c / 32u]); }
    if (kind == 3u) {
        let b = soff + local * kb + c / 32u;
        return e8m0((wbuf[b / 4u] >> (8u * (b % 4u))) & 255u);
    }
    return 1.0;
}
"#;

/// The decode kernel: 8 rows a workgroup, 32 lanes a row, each lane a word in 32 of the row, every token at once.
const FEW_KERNEL: &str = r#"
var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k = p[0].x; let t = p[0].y; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z; let kind = p[1].w;
    let woff = p[2].x;
    let lane = li & 31u;
    let slot = li >> 5u;
    // The rows this buffer and the call share: [max(first, lo), min(first + rows, lo + count)).
    let start = max(first, lo);
    let row = start + wg.x * 8u + slot;
    let live = row < min(first + rows, lo + count);
    var acc: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) { acc[i] = 0.0; }
    if (live) {
        let local = row - first;
        let wd = wide(kind);
        let per = k / wd;
        let kb = k / 32u;
        for (var w = lane; w < per; w += 32u) {
            let word = wbuf[woff + local * per + w];
            let c = w * wd;
            let s = scale(kind, soff, local, c, kb);
            for (var tt = 0u; tt < t; tt++) {
                var d = 0.0;
                for (var j = 0u; j < wd; j++) { d += weight(word, kind, j) * x[tt * k + c + j]; }
                acc[tt] += d * s;
            }
        }
    }
    for (var tt = 0u; tt < 8u; tt++) { part[(slot * 32u + lane) * 8u + tt] = acc[tt]; }
    workgroupBarrier();
    if (live && lane == 0u) {
        for (var tt = 0u; tt < t; tt++) {
            var sum = 0.0;
            for (var l = 0u; l < 32u; l++) { sum += part[(slot * 32u + l) * 8u + tt]; }
            y[tt * count + (row - lo)] = sum;
        }
    }
}
"#;

/// The prompt kernel: 64 tokens by 64 rows a workgroup, 32 of k a step; each thread 4 tokens by 4 rows, its sums four
/// vec4s (rows) and the decoded weights kept as vec4s of 4 rows, read a vec4 at a time. (Its sums an array indexed in
/// loops, the compiler kept them in memory, not registers: 64 expert matrices of 31 tokens took 0.15 s.)
const MANY_KERNEL: &str = r#"
var<workgroup> xs: array<f32, 2048>;
// k step kk, rows 4q..4q+3: wt[kk * 16 + q]
var<workgroup> wt: array<vec4<f32>, 512>;

// Token `tok`'s sums of rows row0..row0+3, those in [lo, end) and of a token there is.
fn put(v: vec4<f32>, tok: u32, row0: u32, t: u32, end: u32, lo: u32, count: u32) {
    if (tok >= t) { return; }
    for (var b = 0u; b < 4u; b++) {
        if (row0 + b < end) { y[tok * count + (row0 + b - lo)] = v[b]; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k = p[0].x; let t = p[0].y; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z; let kind = p[1].w;
    let woff = p[2].x;
    let start = max(first, lo);
    let end = min(first + rows, lo + count);
    let r0 = start + wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    let wd = wide(kind);
    let per = k / wd;
    // Words a row holds in a 32-wide step of k.
    let wps = 32u / wd;
    let kb = k / 32u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < k; k0 += 32u) {
        // The tokens' 64 x 32: element e, token e / 32, k e % 32, kept k-major.
        for (var e = li; e < 2048u; e += 256u) {
            let tok = t0 + e / 32u;
            let kk = e % 32u;
            var v = 0.0;
            if (tok < t) { v = x[tok * k + k0 + kk]; }
            xs[kk * 64u + e / 32u] = v;
        }
        // The rows' 64 x 32, decoded and scaled: a word an element.
        for (var e = li; e < 64u * wps; e += 256u) {
            let r = e / wps;
            let q = e % wps;
            let row = r0 + r;
            let live = row < end;
            var word = 0u;
            var s = 0.0;
            if (live) {
                let local = row - first;
                word = wbuf[woff + local * per + k0 / wd + q];
                s = scale(kind, soff, local, k0, kb);
            }
            for (var j = 0u; j < wd; j++) {
                wt[(q * wd + j) * 16u + r / 4u][r % 4u] = select(0.0, weight(word, kind, j) * s, live);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let w = wt[kk * 16u + tx];
            let xb = kk * 64u + ty * 4u;
            acc0 += xs[xb] * w;
            acc1 += xs[xb + 1u] * w;
            acc2 += xs[xb + 2u] * w;
            acc3 += xs[xb + 3u] * w;
        }
        workgroupBarrier();
    }
    let tok = t0 + ty * 4u;
    let row0 = r0 + tx * 4u;
    put(acc0, tok, row0, t, end, lo, count);
    put(acc1, tok + 1u, row0, t, end, lo, count);
    put(acc2, tok + 2u, row0, t, end, lo, count);
    put(acc3, tok + 3u, row0, t, end, lo, count);
}
"#;

pub(super) fn shader(many: bool) -> String {
    format!("{COMMON}{}", if many { MANY_KERNEL } else { FEW_KERNEL })
}

/// A decode step's kernel ([`Arena`]'s): one row of x against a weight of `kind`, written for that kind. 8 rows a
/// workgroup and 32 lanes a row as [`FEW_KERNEL`], but a lane takes 32 of k at a time (the reach of one scale, so a
/// scale is read once for 32 weights where once a word), x four at a load, and an fp8 byte or an e2m1 nibble is a read
/// of the workgroup's table of its values (made by its threads, one each) where it was decoded by its bits with a
/// branch a weight; its one sum is a register. The row of x is at `p[2].y` of the call's inputs and the sums go to
/// `p[2].z` of its results (both in f32).
pub(super) fn one_shader(kind: Kind) -> String {
    const X: &str = "@group(0) @binding(1) var<storage, read> x: array<f32>;";
    assert_eq!(COMMON.matches(X).count(), 1, "the kernels' input binding");
    let common = COMMON.replace(X, "@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;");
    let lanes = ["x", "y", "z", "w"];
    // (the workgroup's table's entries, a thread's part in making it, words a 32 of k, a block's sum, its scale)
    let (entries, fill, words, sum, scale): (usize, &str, usize, String, &str) = match kind {
        Kind::Fp8 => (
            256,
            "    lut[li] = fp8(li);\n",
            8,
            (0..8)
                .map(|q| {
                    let bytes: String = (0..4)
                        .map(|j| {
                            let byte = if j == 3 { "w >> 24u".to_string() } else { format!("(w >> {}u) & 255u", 8 * j) };
                            format!(" d += lut[{byte}] * v.{};", lanes[j])
                        })
                        .collect();
                    format!("            {{ let w = wbuf[at + {q}u]; let v = x4[xb + {q}u];{bytes} }}\n")
                })
                .collect(),
            "bitcast<f32>(wbuf[soff + (local / 32u) * kb + b])",
        ),
        Kind::Bf16 => (
            1,
            "",
            16,
            (0..8)
                .map(|q| {
                    format!(
                        "            {{ let w0 = wbuf[at + {}u]; let w1 = wbuf[at + {}u]; let v = x4[xb + {q}u]; d += bitcast<f32>(w0 << 16u) * v.x; d += bitcast<f32>(w0 & 0xffff0000u) * v.y; d += bitcast<f32>(w1 << 16u) * v.z; d += bitcast<f32>(w1 & 0xffff0000u) * v.w; }}\n",
                        2 * q,
                        2 * q + 1
                    )
                })
                .collect(),
            "1.0",
        ),
        Kind::Mxfp4 | Kind::Record => (
            16,
            "    if (li < 16u) { lut[li] = fp4(li); }\n",
            4,
            (0..4)
                .map(|q| {
                    let nibbles: String = (0..8)
                        .map(|j| {
                            let nibble = if j == 7 { "w >> 28u".to_string() } else { format!("(w >> {}u) & 15u", 4 * j) };
                            format!(" d += lut[{nibble}] * {}.{};", if j < 4 { "va" } else { "vb" }, lanes[j % 4])
                        })
                        .collect();
                    format!("            {{ let w = wbuf[at + {q}u]; let va = x4[xb + {}u]; let vb = x4[xb + {}u];{nibbles} }}\n", 2 * q, 2 * q + 1)
                })
                .collect(),
            if kind == Kind::Mxfp4 { "bitcast<f32>(wbuf[soff + local * kb + b])" } else { "e8m0((wbuf[(soff + local * kb + b) / 4u] >> (8u * ((soff + local * kb + b) % 4u))) & 255u)" },
        ),
    };
    format!(
        r#"{common}
var<workgroup> lut: array<f32, {entries}>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let k = p[0].x; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z;
    let woff = p[2].x;
    let xo = p[2].y / 4u;
    let yo = p[2].z;
    let lane = li & 31u;
    let slot = li >> 5u;
{fill}    workgroupBarrier();
    // The rows this buffer and the call share: [max(first, lo), min(first + rows, lo + count)).
    let start = max(first, lo);
    let row = start + wg.x * 8u + slot;
    let live = row < min(first + rows, lo + count);
    var acc = 0.0;
    if (live) {{
        let local = row - first;
        let kb = k / 32u;
        let wrow = woff + local * kb * {words}u;
        for (var b = lane; b < kb; b += 32u) {{
            let at = wrow + b * {words}u;
            let xb = xo + b * 8u;
            var d = 0.0;
{sum}            acc += d * {scale};
        }}
    }}
    part[li] = acc;
    workgroupBarrier();
    if (live && lane == 0u) {{
        var total = 0.0;
        for (var l = 0u; l < 32u; l++) {{ total += part[slot * 32u + l]; }}
        y[yo + row - lo] = total;
    }}
}}
"#
    )
}

/// The pipeline name of [`one_shader`]'s kernel for `kind`.
pub(super) fn one_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Fp8 => "dense-one-fp8",
        Kind::Bf16 => "dense-one-bf16",
        Kind::Mxfp4 => "dense-one-mxfp4",
        Kind::Record => "dense-one-record",
    }
}

/// What the stages between a call's projections share ([`swiglu_stage`], [`round_stage`]): a workgroup 32 of the row
/// (the reach of one fp8 scale), and the reference's roundings, made of the values' bits (integer steps, products by
/// powers of two) and so the host's to the bit: to bf16, and an activation's quantization to fp8 (a power-of-two
/// scale from the 32's largest, each value rounded to nearest even on the e4m3 grid).
const STAGE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 3>;
var<workgroup> h: array<f32, 32>;

// to bf16 and back: round to nearest even at bit 16
fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

// to the e4m3 grid, |v| <= 448: its step 2^-9 below 2^-6 and 2^(e - 3) in the binade of 2^e
fn e4m3(v: f32) -> f32 {
    let a = abs(v);
    var q = 448.0;
    if (a < 0.015625) {
        q = round(a * 512.0) * 0.001953125;
    } else if (a < 448.0) {
        let e = (bitcast<u32>(a) >> 23u) & 0xffu;
        q = round(a * bitcast<f32>((257u - e) << 23u)) * bitcast<f32>((e - 3u) << 23u);
    }
    return select(q, -q, (bitcast<u32>(v) >> 31u) == 1u);
}

// `v`, one of the workgroup's 32 in `h`, quantized to fp8 with them: the scale 2^ceil(log2(most / 448)), most at
// least 1e-4 (its exponent, one more unless its mantissa is zero)
fn fp8_of(v: f32) -> f32 {
    var most = 0.0;
    for (var l = 0u; l < 32u; l++) { most = max(most, abs(h[l])); }
    let b = bitcast<u32>(max(most, 0.0001) * 0.002232142857142857);
    let e = ((b >> 23u) & 0xffu) + select(0u, 1u, (b & 0x7fffffu) != 0u);
    return e4m3(clamp(v * bitcast<f32>((254u - e) << 23u), -448.0, 448.0)) * bitcast<f32>(e << 23u);
}
"#;

/// The stage between a unit's projections ([`forward_units`]): the gate's and the up's sums rounded to bf16 and
/// clamped at the limit, `silu(gate) * up * weight` rounded to bf16, and that quantized to fp8 ([`STAGE`]), which is
/// what the down projection multiplies. The exponential and the division are the device's own, so a result now and
/// then is a bf16 step from the host's. `p[0]`: where the gate's sums start in the results, the up's, where the
/// activation goes in the inputs; `p[1]`: the weight's bits, the limit's (0: none).
pub(super) fn swiglu_stage() -> String {
    format!(
        "{STAGE}{}",
        r#"
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = wg.x * 32u + li;
    let weight = bitcast<f32>(p[1].x);
    let limit = bitcast<f32>(p[1].y);
    var g = bf16(src[p[0].x + i]);
    var u = bf16(src[p[0].y + i]);
    if (limit > 0.0) {
        u = clamp(u, -limit, limit);
        g = min(g, limit);
    }
    let v = bf16(g / (1.0 + exp(-g)) * u * weight);
    h[li] = v;
    workgroupBarrier();
    dst[p[0].z + i] = fp8_of(v);
}
"#
    )
}

/// The stage between a chain's projections ([`forward_chained`]): the first's sums rounded to bf16, and quantized to
/// fp8 where the last takes its activation so ([`STAGE`]); every step of it the host's to the bit. `p[0]`: where the
/// sums start in the results, (nothing), where the row goes in the inputs; `p[1].x`: 1 where it is quantized.
pub(super) fn round_stage() -> String {
    format!(
        "{STAGE}{}",
        r#"
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = wg.x * 32u + li;
    let v = bf16(src[p[0].x + i]);
    h[li] = v;
    workgroupBarrier();
    let q = fp8_of(v);
    dst[p[0].z + i] = select(v, q, p[1].x == 1u);
}
"#
    )
}
