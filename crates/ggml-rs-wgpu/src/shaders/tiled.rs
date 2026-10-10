//! The K-quants' and Q8_0's tiled kernels for a prompt's rows where there are no tensor cores.

/// The tiled kernel for the K-quants ([`source_many`]'s for them): 64 weight rows by 64 tokens a workgroup as the
/// generic one, but 64 values of `k` a step, each of its 256 threads decoding 16 weights of its row (a quarter of
/// the step) from wide loads, as the CPU dequantizes them, where 64 threads decoded a row's 32 a byte a load while
/// the rest waited; and `x` read four at a time. Qwen3.8 27B's chunk of 512 in 2.4 s with the generic one.
pub(super) const Q3K_TILED: &str = r#"
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

// k-major: the step's k index kk, then its 64 tokens (xs) or rows (ws), read four at a time. Scalars, not vec4s:
// each thread writes its own token's or row's value, and WGSL lets a write to one component of a vector in
// memory write the whole vector (Metal does), so four threads writing a vec4's four would race
var<workgroup> xs: array<f32, 4096>;
var<workgroup> ws: array<f32, 4096>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 64u + row] = v;
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
            xs[kk * 64u + xt] = v.x;
            xs[(kk + 1u) * 64u + xt] = v.y;
            xs[(kk + 2u) * 64u + xt] = v.z;
            xs[(kk + 3u) * 64u + xt] = v.w;
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
            let wi = kk * 64u + tx * 4u;
            let xi = kk * 64u + ty * 4u;
            let wv = vec4<f32>(ws[wi], ws[wi + 1u], ws[wi + 2u], ws[wi + 3u]);
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
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

/// Q8_0's tiled kernel for a prompt without tensor cores: as [`Q4K_TILED`] (64 tokens by 64 weight rows a workgroup,
/// every thread decoding its part of a step: a row's quarter of a block, 8 values) with a block of 32 a step, so its
/// tiles take 16 KB of a workgroup's memory (WebGPU's portable limit; [`MANY_BODY`]'s 18, its decode a thread a row
/// into an array in memory while the rest waited: LTX's steps without tensor cores 55% slower than f16's). A block's
/// 34 bytes (an f16 scale, then 32 int8) start two bytes off a word in every other one.
pub(super) const Q80_TILED: &str = r#"
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

// k-major: the step's k index kk, then its 64 tokens (xs) or rows (ws), read four at a time. Scalars, not vec4s:
// each thread writes its own token's or row's value, and WGSL lets a write to one component of a vector in
// memory write the whole vector (Metal does), so four threads writing a vec4's four would race
var<workgroup> xs: array<f32, 2048>;
var<workgroup> ws: array<f32, 2048>;

// A word's four int8, low byte first.
fn i8x4(v: u32) -> vec4<f32> {
    return vec4<f32>(f32(bitcast<i32>(v << 24u) >> 24u), f32(bitcast<i32>(v << 16u) >> 24u), f32(bitcast<i32>(v << 8u) >> 24u), f32(bitcast<i32>(v) >> 24u));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    // what this thread loads: x's token li / 4, its 8 of the step's k at (li % 4) * 8; and decodes: row li / 4's
    // quarter li % 4 of the step's block
    let xt = li / 4u;
    let xq = li % 4u;
    let wr = li / 4u;
    let wq = li % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let k4 = p.k / 4u;
    let steps = p.k / 32u;
    let r = r0 + wr;
    for (var s = 0u; s < steps; s++) {
        let tok = t0 + xt;
        for (var i = 0u; i < 2u; i++) {
            var v = vec4<f32>(0.0);
            if (tok < p.m) { v = x4[tok * k4 + s * 8u + xq * 2u + i]; }
            let kk = xq * 8u + i * 4u;
            xs[kk * 64u + xt] = v.x;
            xs[(kk + 1u) * 64u + xt] = v.y;
            xs[(kk + 2u) * 64u + xt] = v.z;
            xs[(kk + 3u) * 64u + xt] = v.w;
        }
        var lo = vec4<f32>(0.0);
        var hi = vec4<f32>(0.0);
        if (r < p.rows) {
            // the block's scale, then this quarter's 8 int8 (two bytes off a word where the block starts two off)
            let at = r * p.row_bytes + s * 34u;
            let dw = w[at / 4u];
            let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, at % 4u == 2u);
            let q = at + 2u + wq * 8u;
            let a = w[q / 4u];
            let b = w[q / 4u + 1u];
            var w0 = a;
            var w1 = b;
            if (q % 4u == 2u) {
                let c = w[q / 4u + 2u];
                w0 = (a >> 16u) | (b << 16u);
                w1 = (b >> 16u) | (c << 16u);
            }
            lo = d * i8x4(w0);
            hi = d * i8x4(w1);
        }
        let kw = wq * 8u;
        ws[kw * 64u + wr] = lo.x;
        ws[(kw + 1u) * 64u + wr] = lo.y;
        ws[(kw + 2u) * 64u + wr] = lo.z;
        ws[(kw + 3u) * 64u + wr] = lo.w;
        ws[(kw + 4u) * 64u + wr] = hi.x;
        ws[(kw + 5u) * 64u + wr] = hi.y;
        ws[(kw + 6u) * 64u + wr] = hi.z;
        ws[(kw + 7u) * 64u + wr] = hi.w;
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let wi = kk * 64u + tx * 4u;
            let xi = kk * 64u + ty * 4u;
            let wv = vec4<f32>(ws[wi], ws[wi + 1u], ws[wi + 2u], ws[wi + 3u]);
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
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

pub(super) const Q4K_TILED: &str = r#"
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

// k-major: the step's k index kk, then its 64 tokens (xs) or rows (ws), read four at a time. Scalars, not vec4s:
// each thread writes its own token's or row's value, and WGSL lets a write to one component of a vector in
// memory write the whole vector (Metal does), so four threads writing a vec4's four would race
var<workgroup> xs: array<f32, 4096>;
var<workgroup> ws: array<f32, 4096>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 64u + row] = v;
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
            xs[kk * 64u + xt] = v.x;
            xs[(kk + 1u) * 64u + xt] = v.y;
            xs[(kk + 2u) * 64u + xt] = v.z;
            xs[(kk + 3u) * 64u + xt] = v.w;
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
            let wi = kk * 64u + tx * 4u;
            let xi = kk * 64u + ty * 4u;
            let wv = vec4<f32>(ws[wi], ws[wi + 1u], ws[wi + 2u], ws[wi + 3u]);
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
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

pub(super) const Q6K_TILED: &str = r#"
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

// k-major: the step's k index kk, then its 64 tokens (xs) or rows (ws), read four at a time. Scalars, not vec4s:
// each thread writes its own token's or row's value, and WGSL lets a write to one component of a vector in
// memory write the whole vector (Metal does), so four threads writing a vec4's four would race
var<workgroup> xs: array<f32, 4096>;
var<workgroup> ws: array<f32, 4096>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 64u + row] = v;
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
            xs[kk * 64u + xt] = v.x;
            xs[(kk + 1u) * 64u + xt] = v.y;
            xs[(kk + 2u) * 64u + xt] = v.z;
            xs[(kk + 3u) * 64u + xt] = v.w;
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
            let wi = kk * 64u + tx * 4u;
            let xi = kk * 64u + ty * 4u;
            let wv = vec4<f32>(ws[wi], ws[wi + 1u], ws[wi + 2u], ws[wi + 3u]);
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
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

pub(super) const Q5K_TILED: &str = r#"
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

// k-major: the step's k index kk, then its 64 tokens (xs) or rows (ws), read four at a time. Scalars, not vec4s:
// each thread writes its own token's or row's value, and WGSL lets a write to one component of a vector in
// memory write the whole vector (Metal does), so four threads writing a vec4's four would race
var<workgroup> xs: array<f32, 4096>;
var<workgroup> ws: array<f32, 4096>;

// row `row`'s weight at the step's `kk`
fn put_w(kk: u32, row: u32, v: f32) {
    ws[kk * 64u + row] = v;
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
            xs[kk * 64u + xt] = v.x;
            xs[(kk + 1u) * 64u + xt] = v.y;
            xs[(kk + 2u) * 64u + xt] = v.z;
            xs[(kk + 3u) * 64u + xt] = v.w;
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
            let wi = kk * 64u + tx * 4u;
            let xi = kk * 64u + ty * 4u;
            let wv = vec4<f32>(ws[wi], ws[wi + 1u], ws[wi + 2u], ws[wi + 3u]);
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
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
