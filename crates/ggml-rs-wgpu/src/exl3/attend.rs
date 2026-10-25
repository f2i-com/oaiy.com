//! DeepSeek-V4.1's decode attention as a device program's steps ([`super::ProgramStep`]'s `Rope` to `HcMix`): a step's
//! queries rotated, its window key normed, rotated, quantized and kept, a compressing layer's latent and index key made
//! and kept, the indexer's scores of the compressed positions and its choice of them, the heads' attention over the
//! window and the chosen positions, and the attention's mixing coefficients, each where the host made them between
//! the layer's two calls. Every rounding the host makes is made here the same way, to the bit, but for the
//! exponentials (the attention's softmax, a ratio-2 compressor's pool, the mixing's sigmoids and Sinkhorn), which are
//! the device's own. Numbers that change from step to step (the position, the window's slot, the compressed count) are
//! read from a vector of words the program uploads each run (`pos`, a word each), so its recording is the same at
//! every step and replayed.

/// The host's own roundings as the kernels make them: bf16 (to nearest even), a sign put on, the power of two at or
/// above a number, the e4m3 grid's nearest value (ties to even), and a quotient, a square root and a reciprocal
/// correctly rounded by integer steps (the device's own are within ulps).
macro_rules! exact_fns {
    () => {
        r#"
fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

fn with_sign(m: f32, s: f32) -> f32 {
    return bitcast<f32>((bitcast<u32>(m) & 0x7fffffffu) | (bitcast<u32>(s) & 0x80000000u));
}

fn pow2_ceil(x: f32) -> f32 {
    let b = bitcast<u32>(x);
    var e = i32((b >> 23u) & 0xffu) - 127;
    if ((b & 0x7fffffu) != 0u) { e = e + 1; }
    return bitcast<f32>(u32(e + 127) << 23u);
}

// 1 / s of a normal power of two, exactly
fn inv_pow2(s: f32) -> f32 {
    return bitcast<f32>((254u - ((bitcast<u32>(s) >> 23u) & 0xffu)) << 23u);
}

fn round_e4m3(v: f32) -> f32 {
    let a = abs(v);
    if (a >= 448.0) { return with_sign(448.0, v); }
    var q = 0.0;
    if (a < 0.015625) {
        q = round(a * 512.0) * 0.001953125;
    } else {
        let eb = (bitcast<u32>(a) >> 23u) & 0xffu;
        q = round(a * bitcast<f32>((257u - eb) << 23u)) * bitcast<f32>((eb - 3u) << 23u);
    }
    return with_sign(q, v);
}

// a / b of positive normal numbers (a 0 too), to nearest even: the mantissas' quotient to 26 bits by long division, the
// remainder below b's mantissa, the rest a sticky bit (a quotient past the normal numbers the device's own)
fn div_rn(a: f32, b: f32) -> f32 {
    if (a == 0.0) { return 0.0; }
    let ab = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let mb = (bb & 0x7fffffu) | 0x800000u;
    var rem = (ab & 0x7fffffu) | 0x800000u;
    var q = 0u;
    for (var i = 0u; i < 26u; i++) {
        q = q << 1u;
        if (rem >= mb) {
            rem = rem - mb;
            q = q | 1u;
        }
        rem = rem << 1u;
    }
    var shift = 1u;
    if (q >= 0x2000000u) { shift = 2u; }
    let guard = (q >> (shift - 1u)) & 1u;
    let sticky = rem != 0u || (q & ((1u << (shift - 1u)) - 1u)) != 0u;
    var r = q >> shift;
    if (guard == 1u && (sticky || (r & 1u) == 1u)) { r = r + 1u; }
    var be = i32(shift) - 25 + i32(ab >> 23u) - i32(bb >> 23u) + 150;
    if (r == 0x1000000u) {
        r = 0x800000u;
        be = be + 1;
    }
    if (be <= 0 || be >= 255) { return a / b; }
    return bitcast<f32>((u32(be) << 23u) | (r & 0x7fffffu));
}

// a / b, b a positive normal number: div_rn of a's magnitude with a's sign (a zero, a subnormal or NaN the device's own)
fn div_signed(a: f32, b: f32) -> f32 {
    let m = abs(a);
    if (!(m >= 1.17549435e-38) || m > 3.40282347e38) { return a / b; }
    return with_sign(div_rn(m, b), a);
}

// the square root of a positive normal number, to nearest even: its mantissa's (times 2^26) digit by digit, two bits a
// step, the remainder below twice the root
fn sqrt_rn(a: f32) -> f32 {
    let bits = bitcast<u32>(a);
    var e = i32(bits >> 23u) - 150;
    var m = (bits & 0x7fffffu) | 0x800000u;
    if ((e & 1) != 0) {
        m = m << 1u;
        e = e - 1;
    }
    var rem = 0u;
    var root = 0u;
    for (var i = 25; i >= 0; i--) {
        var pair = 0u;
        if (i >= 13) { pair = (m >> u32(2 * i - 26)) & 3u; }
        rem = (rem << 2u) | pair;
        let trial = (root << 2u) | 1u;
        root = root << 1u;
        if (rem >= trial) {
            rem = rem - trial;
            root = root | 1u;
        }
    }
    var shift = 1u;
    if (root >= 0x2000000u) { shift = 2u; }
    let guard = (root >> (shift - 1u)) & 1u;
    let sticky = rem != 0u || (root & ((1u << (shift - 1u)) - 1u)) != 0u;
    var r = root >> shift;
    if (guard == 1u && (sticky || (r & 1u) == 1u)) { r = r + 1u; }
    var be = e / 2 - 13 + i32(shift) + 150;
    if (r == 0x1000000u) {
        r = 0x800000u;
        be = be + 1;
    }
    return bitcast<f32>((u32(be) << 23u) | (r & 0x7fffffu));
}

// 1 / s of a positive normal number, to nearest even: 2^48 over its mantissa by long division, the remainder below it
fn recip_rn(s: f32) -> f32 {
    let bits = bitcast<u32>(s);
    let es = i32(bits >> 23u);
    let m = (bits & 0x7fffffu) | 0x800000u;
    var rem = 0u;
    var q = 0u;
    for (var i = 0u; i < 49u; i++) {
        rem = rem << 1u;
        if (i == 0u) { rem = rem | 1u; }
        q = q << 1u;
        if (rem >= m) {
            rem = rem - m;
            q = q | 1u;
        }
    }
    if (q == 0x2000000u) {
        return bitcast<f32>(u32(254 - es) << 23u);
    }
    var r = q >> 1u;
    if ((q & 1u) == 1u && (rem != 0u || (r & 1u) == 1u)) { r = r + 1u; }
    var be = 253 - es;
    if (r == 0x1000000u) {
        r = 0x800000u;
        be = be + 1;
    }
    return bitcast<f32>((u32(be) << 23u) | (r & 0x7fffffu));
}
"#
    };
}

/// `Rope`'s kernel: a thread a pair, `rows * half` of them; `p[0]` = (rows, stride, at, half), `p[1]` = (the position's
/// word, inverse, rows a position): with `p[1].z` 0 every row at the word's position, else row `r` at word `p[1].x + r /
/// p[1].z`'s (`RopeRows`: a prompt's tokens, each its heads' rows). Each pair's four products stored to workgroup memory
/// before their sums (so no fused multiply-add takes them): the host's `bf16(a c - b s)`, `bf16(a s + b c)` to the bit.
pub(super) const DS_ROPE: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(0) var<storage, read> cosv: array<f32>;
@group(0) @binding(1) var<storage, read> sinv: array<f32>;
@group(0) @binding(2) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> prod: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let rows = p[0].x;
    let stride = p[0].y;
    let half = p[0].w;
    let i = id.x;
    let live = i < rows * half;
    var base = 0u;
    if (live) {
        let j = i % half;
        base = (i / half) * stride + p[0].z + 2u * j;
        var word = p[1].x;
        if (p[1].z != 0u) { word = p[1].x + (i / half) / p[1].z; }
        let at = words[word] * half + j;
        let c = cosv[at];
        var s = sinv[at];
        if (p[1].y != 0u) { s = -s; }
        let a = x[base];
        let b = x[base + 1u];
        prod[t] = vec4<f32>(a * c, b * s, a * s, b * c);
    }
    workgroupBarrier();
    if (live) {
        let q = prod[t];
        x[base] = bf16(q.x - q.y);
        x[base + 1u] = bf16(q.z + q.w);
    }
}
"#
);

/// `Fp8`'s kernel: a thread a value, a block of 32 eight a workgroup; `p[0]` = (len, 1/448's bits, 1e-4's bits). The
/// host's `fake_quant_fp8_inplace`: a block's scale the power of two at or above its largest magnitude (at least 1e-4)
/// over 448, each value over it on the e4m3 grid (clamped to ±448), times it, rounded to bf16.
pub(super) const DS_FP8: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> am: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let len = p[0].x;
    let i = wg.x * 256u + t;
    var v = 0.0;
    if (i < len) { v = x[i]; }
    am[t] = abs(v);
    workgroupBarrier();
    let b0 = t - t % 32u;
    var m = 0.0;
    for (var j = 0u; j < 32u; j++) { m = max(m, am[b0 + j]); }
    if (i >= len) { return; }
    let s = pow2_ceil(max(m, bitcast<f32>(p[0].z)) * bitcast<f32>(p[0].y));
    x[i] = bf16(round_e4m3(clamp(v * inv_pow2(s), -448.0, 448.0)) * s);
}
"#
);

/// `Fp4`'s kernel: a thread a value; `p[0]` = (len, block, e4m3), `p[1]` = (6 * 2^-126's bits, 6 * 2^-9's, 1/6's). The
/// host's `fake_quant_fp4_inplace`: a block's scale the power of two at or above its largest magnitude over 6 (E8M0),
/// or that over 6 on the e4m3 grid (E4M3, the quotient correctly rounded first), each value over it (correctly
/// rounded) clamped to ±6 and put on the e2m1 grid (ties to the even code), times it, rounded to bf16.
pub(super) const DS_FP4: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> am: array<f32, 256>;

fn round_fp4(v: f32) -> f32 {
    var grid = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    let m = abs(v);
    var hi = 7u;
    for (var i = 0u; i < 8u; i++) {
        if (grid[i] >= m) {
            hi = i;
            break;
        }
    }
    var lo = 0u;
    if (hi > 0u) { lo = hi - 1u; }
    let dl = m - grid[lo];
    let dh = grid[hi] - m;
    var code = lo;
    if (dh < dl || (dh == dl && (hi % 2u) == 0u)) { code = hi; }
    return with_sign(grid[code], v);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let len = p[0].x;
    let block = p[0].y;
    let i = wg.x * 256u + t;
    var v = 0.0;
    if (i < len) { v = x[i]; }
    am[t] = abs(v);
    workgroupBarrier();
    let b0 = t - t % block;
    var m = 0.0;
    for (var j = 0u; j < block; j++) { m = max(m, am[b0 + j]); }
    if (i >= len) { return; }
    var s = 0.0;
    var q = 0.0;
    if (p[0].z != 0u) {
        s = round_e4m3(div_rn(max(m, bitcast<f32>(p[1].y)), 6.0));
        q = div_signed(v, s);
    } else {
        s = pow2_ceil(max(m, bitcast<f32>(p[1].x)) * bitcast<f32>(p[1].z));
        q = v * inv_pow2(s);
    }
    x[i] = bf16(round_fp4(clamp(q, -6.0, 6.0)) * s);
}
"#
);

/// `PutRow`'s kernel: a thread a value; `p[0]` = (width, the row's word).
pub(super) const DS_PUT: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = p[0].x;
    if (id.x >= width) { return; }
    dst[words[p[0].y] * width + id.x] = src[id.x];
}
"#;

/// `Pool`'s kernel: a thread a feature; `p[0]` = (width, ratio). The host's `pool` (each feature's softmax over the
/// group's rows by their scores, the rows' sum by it), rounded to bf16; its exponentials the device's own.
pub(super) const DS_POOL: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(0) var<storage, read> kv: array<f32>;
@group(0) @binding(1) var<storage, read> score: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = p[0].x;
    let r = p[0].y;
    let f = id.x;
    if (f >= w) { return; }
    var mx = bitcast<f32>(0xff800000u);
    for (var i = 0u; i < r; i++) { mx = max(mx, score[i * w + f]); }
    var s = 0.0;
    for (var i = 0u; i < r; i++) { s += exp(score[i * w + f] - mx); }
    var o = 0.0;
    for (var i = 0u; i < r; i++) { o += kv[i * w + f] * (exp(score[i * w + f] - mx) / s); }
    y[f] = bf16(o);
}
"#
);

/// `IndexScores`' kernel: a thread a (key, head), the workgroup's keys `256 / heads` at a time, the workgroups over them;
/// `p[0]` = (heads, dim, the count's word, the weights' scale's bits), `p[1].x` the grid's workgroups. The host's
/// `index_scores`: a key's score `bf16(sum_h bf16(max(bf16(q_h . k), 0) * bf16(w_h * scale)))`, each head's dot product
/// summed in the host's order (one after another; its products, of fp4 values, exact) by its own thread, the queries
/// laid by dimension in workgroup memory (each thread's head beside the next's), then the heads' in theirs by the key's
/// first: to the bit. (A thread a key, its heads one after another, waited on its key's reads 4,096 times: 250 us.)
pub(super) const DS_INDEX_SCORES: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> qt: array<f32, 4096>;
var<workgroup> ws: array<f32, 64>;
var<workgroup> dots: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let heads = p[0].x;
    let dim = p[0].y;
    for (var i = t; i < heads * dim; i += 256u) { qt[(i % dim) * heads + i / dim] = q[i]; }
    if (t < heads) { ws[t] = bf16(w[t] * bitcast<f32>(p[0].w)); }
    workgroupBarrier();
    let n = words[p[0].z];
    let per = 256u / heads;
    let h = t % heads;
    let slot = t / heads;
    for (var base = wg.x * per; base < n; base += p[1].x * per) {
        let k = base + slot;
        var dot = 0.0;
        if (slot < per && k < n) {
            let row = k * dim;
            // (eight of the key's values read before their adds, the adds in order)
            for (var d0 = 0u; d0 < dim; d0 += 8u) {
                var kv: array<f32, 8>;
                for (var j = 0u; j < 8u; j++) { kv[j] = keys[row + d0 + j]; }
                for (var j = 0u; j < 8u; j++) { dot += qt[(d0 + j) * heads + h] * kv[j]; }
            }
        }
        dots[t] = dot;
        workgroupBarrier();
        if (h == 0u && slot < per && k < n) {
            var acc = 0.0;
            for (var j = 0u; j < heads; j++) { acc += bf16(max(bf16(dots[slot * heads + j]), 0.0) * ws[j]); }
            out[k] = bf16(acc);
        }
        workgroupBarrier();
    }
}
"#
);

/// `IndexScoresRows`' kernel: [`DS_INDEX_SCORES`] of a prompt's rows, a workgroup's `y` its row (`q` and the weights
/// its own, `[rows, heads, dim]` and `[rows, heads]`, the weights' scale made already), every one of the first `n` keys
/// (`n` its word's) scored as the host's `index_scores` scores them, to the bit: `[rows, n]` from the vector's start.
/// `p[0]` = (heads, dim, the count's word), `p[1].x` the grid's workgroups along the keys.
pub(super) const DS_INDEX_SCORES_ROWS: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> qt: array<f32, 4096>;
var<workgroup> ws: array<f32, 64>;
var<workgroup> dots: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let heads = p[0].x;
    let dim = p[0].y;
    let n = words[p[0].z];
    let row = wg.y;
    for (var i = t; i < heads * dim; i += 256u) { qt[(i % dim) * heads + i / dim] = q[row * heads * dim + i]; }
    if (t < heads) { ws[t] = w[row * heads + t]; }
    workgroupBarrier();
    let per = 256u / heads;
    let h = t % heads;
    let slot = t / heads;
    for (var base = wg.x * per; base < n; base += p[1].x * per) {
        let k = base + slot;
        var dot = 0.0;
        if (slot < per && k < n) {
            let at = k * dim;
            // (eight of the key's values read before their adds, the adds in order)
            for (var d0 = 0u; d0 < dim; d0 += 8u) {
                var kv: array<f32, 8>;
                for (var j = 0u; j < 8u; j++) { kv[j] = keys[at + d0 + j]; }
                for (var j = 0u; j < 8u; j++) { dot += qt[(d0 + j) * heads + h] * kv[j]; }
            }
        }
        dots[t] = dot;
        workgroupBarrier();
        if (h == 0u && slot < per && k < n) {
            var acc = 0.0;
            for (var j = 0u; j < heads; j++) { acc += bf16(max(bf16(dots[slot * heads + j]), 0.0) * ws[j]); }
            out[row * n + k] = bf16(acc);
        }
        workgroupBarrier();
    }
}
"#
);

/// `TopK`'s kernel, one workgroup: the `k` best of the first `n` scores (`n` its word's), by the host's order (a score's
/// total order, ties to the lower position: bf16 values, so their upper 16 bits), into `dst` in position order (`+
/// offset`), `0xffffffff` past them; all of them where there are no more than `k`. The `k`-th best's key by its upper
/// byte's then its lower byte's counts, then each thread's run of positions: those above it and the first of those at
/// it, their places by the runs' counts before. `p[0]` = (k, the count's word, offset).
pub(super) const DS_TOPK: &str = r#"
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(3) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> part: array<u32, 256>;
var<workgroup> pick: array<u32, 4>;

fn key16(v: f32) -> u32 {
    let b = bitcast<u32>(v) >> 16u;
    if ((b & 0x8000u) != 0u) { return (~b) & 0xffffu; }
    return b | 0x8000u;
}

// the exclusive running sums of `part` in place (thread 0), their total in pick[0]
fn scan(t: u32) {
    if (t == 0u) {
        var run = 0u;
        for (var j = 0u; j < 256u; j++) {
            let c = part[j];
            part[j] = run;
            run += c;
        }
        pick[0] = run;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let k = p[0].x;
    let n = words[p[0].y];
    let chunk = (n + 255u) / 256u;
    let lo = min(t * chunk, n);
    let hi = min(lo + chunk, n);
    let some = n > k;
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    if (some) {
        for (var i = lo; i < hi; i++) { atomicAdd(&hist[key16(scores[i]) >> 8u], 1u); }
    }
    workgroupBarrier();
    if (t == 0u) {
        var b1 = 0u;
        var above = 0u;
        if (some) {
            var cum = 0u;
            for (var b = 255i; b >= 0i; b--) {
                let c = atomicLoad(&hist[u32(b)]);
                if (cum + c >= k) {
                    b1 = u32(b);
                    above = cum;
                    break;
                }
                cum += c;
            }
        }
        pick[1] = b1;
        pick[2] = above;
    }
    workgroupBarrier();
    let b1 = pick[1];
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    if (some) {
        for (var i = lo; i < hi; i++) {
            let kk = key16(scores[i]);
            if ((kk >> 8u) == b1) { atomicAdd(&hist[kk & 255u], 1u); }
        }
    }
    workgroupBarrier();
    if (t == 0u) {
        var thr = 0u;
        var need = 0xffffffffu;
        if (some) {
            var cum = pick[2];
            for (var b = 255i; b >= 0i; b--) {
                let c = atomicLoad(&hist[u32(b)]);
                if (cum + c >= k) {
                    thr = (b1 << 8u) | u32(b);
                    need = k - cum;
                    break;
                }
                cum += c;
            }
        }
        pick[2] = thr;
        pick[3] = need;
    }
    workgroupBarrier();
    let thr = pick[2];
    let need = pick[3];
    var gt = 0u;
    var eq = 0u;
    for (var i = lo; i < hi; i++) {
        let kk = key16(scores[i]);
        if (kk > thr) { gt++; } else if (kk == thr) { eq++; }
    }
    part[t] = eq;
    workgroupBarrier();
    scan(t);
    workgroupBarrier();
    let before = part[t];
    var taken = 0u;
    if (need > before) { taken = min(need - before, eq); }
    workgroupBarrier();
    part[t] = gt + taken;
    workgroupBarrier();
    scan(t);
    workgroupBarrier();
    var at = part[t];
    let total = pick[0];
    var seen = 0u;
    for (var i = lo; i < hi; i++) {
        let kk = key16(scores[i]);
        var take = kk > thr;
        if (kk == thr) {
            take = seen < taken;
            seen++;
        }
        if (take) {
            dst[at] = i + p[0].z;
            at++;
        }
    }
    for (var j = total + t; j < k; j += 256u) { dst[j] = 0xffffffffu; }
}
"#;

/// `BlockKeep`'s first kernel, a thread a block: of the first `n` scores (`n` its word's), each block of `bs`'s best
/// (the host's `candidate_mask`: the largest, from -infinity up), the newest block's +infinity (it is always kept), into
/// `best`; the blocks' count into `blocks[0]` and each block's flag cleared (`blocks[1 + b]`). `p[0]` = (bs, the count's
/// word).
pub(super) const DS_BLOCK_BEST: &str = r#"
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(3) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> best: array<f32>;
@group(0) @binding(7) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bs = p[0].x;
    let n = words[p[0].y];
    let nb = (n + bs - 1u) / bs;
    let b = id.x;
    if (b == 0u) { blocks[0] = nb; }
    if (b >= nb) { return; }
    blocks[1u + b] = 0u;
    var m = bitcast<f32>(0xff800000u);
    for (var i = b * bs; i < min((b + 1u) * bs, n); i++) {
        let v = scores[i];
        if (v > m) { m = v; }
    }
    if (b == nb - 1u) { m = bitcast<f32>(0x7f800000u); }
    best[b] = m;
}
"#;

/// `BlockKeep`'s last kernel, a thread a pick: each block `picks` names (a `TopK` of the blocks' bests, `u32::MAX`
/// none) flagged kept in `blocks` (`blocks[1 + b]`). `p[0].x` = the picks.
pub(super) const DS_BLOCK_FLAGS: &str = r#"
@group(0) @binding(0) var<storage, read> picks: array<u32>;
@group(0) @binding(6) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p[0].x) { return; }
    let b = picks[id.x];
    if (b != 0xffffffffu) { blocks[1u + b] = 1u; }
}
"#;

/// `MaskBlocks`' kernel, a thread a score: of the first `n` scores (`n` its word's), those of a block `blocks` does not
/// keep (`blocks[1 + i / bs]` 0) made -infinity in place, as the host masks a later index source's row by the candidate
/// blocks. `p[0]` = (bs, the count's word).
pub(super) const DS_MASK_BLOCKS: &str = r#"
@group(0) @binding(1) var<storage, read> blocks: array<u32>;
@group(0) @binding(3) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= words[p[0].y]) { return; }
    if (blocks[1u + i / p[0].x] == 0u) { scores[i] = bitcast<f32>(0xff800000u); }
}
"#;

/// The most window and compressed positions a head of `Attend` attends over, and its positions a workgroup (a chunk):
/// a head's positions spread over [`ATTEND_CHUNKS`] workgroups, so the card has some 640 of them at a step and not 64.
pub(super) const ATTEND_MOST: usize = 640;
pub(super) const ATTEND_CHUNK: usize = 64;
pub(super) const ATTEND_CHUNKS: usize = ATTEND_MOST / ATTEND_CHUNK;

/// What `Attend`'s three kernels share: a position's row (`p[0]` = (dim, the window's slots, the compressed picks'
/// count, the position's word)): the window's slots oldest first (those past the position none), then the picked
/// compressed rows (`0xffffffff` none); a window slot its own number, a compressed row with the top bit set.
macro_rules! attend_rows {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> window: array<f32>;
@group(0) @binding(2) var<storage, read> ckv: array<f32>;
@group(0) @binding(3) var<storage, read> idxs: array<u32>;
@group(0) @binding(4) var<storage, read> words: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const NONE: u32 = 0xffffffffu;
const MOST: u32 = 640u;
const CHUNK: u32 = 64u;
const CHUNKS: u32 = 10u;

fn row_of(e: u32) -> u32 {
    let win = p[0].y;
    let pos = words[p[0].w];
    if (e < win) {
        let s = (pos + 1u + e) % win;
        if (s <= pos) { return s; }
        return NONE;
    }
    if (e >= win + p[0].z) { return NONE; }
    let j = idxs[e - win];
    if (j != NONE) { return j | 0x80000000u; }
    return NONE;
}

fn value(r: u32, at: u32) -> f32 {
    if ((r & 0x80000000u) != 0u) { return ckv[(r & 0x7fffffffu) * p[0].x + at]; }
    return window[r * p[0].x + at];
}
"#
    };
}

/// `Attend`'s first kernel, a workgroup a (head, chunk): its positions' scores, each the host's `dot` (sixteen running
/// sums, a thread each, added as a tree; its products exact) times the scale (`p[1].x`'s bits), to the bit; `-inf`
/// for none. Into the scratch's `[heads, 640]`.
pub(super) const DS_ATTEND_SCORES: &str = concat!(
    attend_rows!(),
    r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;

var<workgroup> qs: array<f32, 512>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wg.x;
    let dim = p[0].x;
    for (var i = t; i < dim; i += 256u) { qs[i] = q[h * dim + i]; }
    workgroupBarrier();
    let lane = t % 16u;
    let first = wg.y * CHUNK;
    for (var e0 = first; e0 < first + CHUNK; e0 += 16u) {
        let e = e0 + t / 16u;
        let r = row_of(e);
        var acc = 0.0;
        if (r != NONE) {
            // (a lane's values eight at a time, all read before their adds: read one by one, each add waited for its
            // value, some 57 us a call)
            let row = (r & 0x7fffffffu) * dim;
            let compressed = (r & 0x80000000u) != 0u;
            for (var i0 = lane; i0 < dim; i0 += 128u) {
                var v: array<f32, 8>;
                if (compressed) {
                    for (var j = 0u; j < 8u; j++) { v[j] = ckv[row + i0 + 16u * j]; }
                } else {
                    for (var j = 0u; j < 8u; j++) { v[j] = window[row + i0 + 16u * j]; }
                }
                for (var j = 0u; j < 8u; j++) { acc += qs[i0 + 16u * j] * v[j]; }
            }
        }
        part[t] = acc;
        workgroupBarrier();
        if (lane == 0u) {
            var l = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for (var i = 0u; i < 16u; i++) { l[i] = part[t + i]; }
            for (var width = 8u; width > 0u; width = width / 2u) {
                for (var i = 0u; i < width; i++) { l[i] += l[i + width]; }
            }
            var s = bitcast<f32>(0xff800000u);
            if (r != NONE) { s = l[0] * bitcast<f32>(p[1].x); }
            scratch[h * MOST + e] = s;
        }
        workgroupBarrier();
    }
}
"#
);

/// `Attend`'s second kernel, a workgroup a (head, chunk): the head's largest score (every chunk's, at least -1e30),
/// its positions' weights `exp(s - max)` (the device's own) and their sum in order, and the rows' sum by the weights
/// rounded to bf16, in order: the host's softmax a chunk at a time, into the scratch's sums (`[heads, 10]` after the
/// scores) and parts (`[heads, 10, dim]` after those).
pub(super) const DS_ATTEND_PARTS: &str = concat!(
    exact_fns!(),
    attend_rows!(),
    r#"
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;

var<workgroup> red: array<f32, 256>;
var<workgroup> wt: array<f32, 64>;
var<workgroup> rows: array<u32, 64>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wg.x;
    let c = wg.y;
    let dim = p[0].x;
    let heads = p[1].y;
    var m = -1e30;
    for (var e = t; e < MOST; e += 256u) { m = max(m, scratch[h * MOST + e]); }
    red[t] = m;
    workgroupBarrier();
    for (var width = 128u; width > 0u; width = width / 2u) {
        if (t < width) { red[t] = max(red[t], red[t + width]); }
        workgroupBarrier();
    }
    let mx = red[0];
    if (t < CHUNK) {
        let e = c * CHUNK + t;
        let r = row_of(e);
        rows[t] = r;
        var w = 0.0;
        if (r != NONE) { w = exp(scratch[h * MOST + e] - mx); }
        wt[t] = w;
    }
    workgroupBarrier();
    let sums = heads * MOST;
    if (t == 0u) {
        var s = 0.0;
        for (var j = 0u; j < CHUNK; j++) {
            if (rows[j] != NONE) { s += wt[j]; }
        }
        scratch[sums + h * CHUNKS + c] = s;
    }
    let parts = sums + heads * CHUNKS + (h * CHUNKS + c) * dim;
    for (var d = t; d < dim; d += 256u) {
        var o = 0.0;
        // (eight positions' values read before their adds, the adds in order)
        for (var j0 = 0u; j0 < CHUNK; j0 += 8u) {
            var v: array<f32, 8>;
            for (var j = 0u; j < 8u; j++) {
                let r = rows[j0 + j];
                v[j] = 0.0;
                if (r != NONE) { v[j] = value(r, d); }
            }
            for (var j = 0u; j < 8u; j++) {
                if (rows[j0 + j] != NONE) { o += bf16(wt[j0 + j]) * v[j]; }
            }
        }
        scratch[parts + d] = o;
    }
}
"#
);

/// `Attend`'s last kernel, a workgroup a head: its chunks' sums and parts added in order, the sink's weight in the
/// denominator, each output over it correctly rounded and rounded to bf16 (the host's sums are one after another
/// across every position: these the same weights and rows, added a chunk at a time).
pub(super) const DS_ATTEND_JOIN: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(5) var<storage, read> sink: array<f32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const MOST: u32 = 640u;
const CHUNKS: u32 = 10u;

var<workgroup> red: array<f32, 256>;
var<workgroup> den: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wg.x;
    let dim = p[0].x;
    let heads = p[1].y;
    var m = -1e30;
    for (var e = t; e < MOST; e += 256u) { m = max(m, scratch[h * MOST + e]); }
    red[t] = m;
    workgroupBarrier();
    for (var width = 128u; width > 0u; width = width / 2u) {
        if (t < width) { red[t] = max(red[t], red[t + width]); }
        workgroupBarrier();
    }
    let sums = heads * MOST;
    if (t == 0u) {
        var s = 0.0;
        for (var c = 0u; c < CHUNKS; c++) { s += scratch[sums + h * CHUNKS + c]; }
        den = s + exp(sink[h] - red[0]);
    }
    workgroupBarrier();
    let dn = den;
    let parts = sums + heads * CHUNKS + h * CHUNKS * dim;
    for (var d = t; d < dim; d += 256u) {
        var o = 0.0;
        for (var c = 0u; c < CHUNKS; c++) { o += scratch[parts + c * dim + d]; }
        out[h * dim + d] = bf16(div_signed(o, dn));
    }
}
"#
);

/// Heads a workgroup of `AttendRows` takes, their queries and scores in its memory.
pub(super) const ATTEND_ROWS_HEADS: usize = 8;

/// `AttendRows`' kernel, a workgroup a (group of eight heads, token): a prompt's token's own positions (`idxs`, `[tokens,
/// len]`, rows of `kv`, `u32::MAX` none) gathered in order by a scan of the threads' counts, then each head's
/// `sparse_attend`: the scores each the host's `dot` (sixteen running sums, a lane each, added as a tree; its products
/// exact, a kv row read once for the eight heads) times the scale, their largest, the weights `exp(s - max)` (the
/// device's own) and their sum in order with the sink's, the rows' sum by the weights rounded to bf16 in order, over the
/// sum correctly rounded and rounded to bf16. `p[0]` = (dim, heads, len), `p[1]` = (the scale's bits). Up to 640
/// positions a token and 512 a row (the host's window and picks: 128 and 512).
pub(super) const DS_ATTEND_ROWS: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<f32>;
@group(0) @binding(2) var<storage, read> idxs: array<u32>;
@group(0) @binding(5) var<storage, read> sink: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const NONE: u32 = 0xffffffffu;
const HG: u32 = 8u;
const MOST: u32 = 640u;

var<workgroup> qs: array<f32, 4096>;
var<workgroup> sc: array<f32, 5120>;
var<workgroup> rows: array<u32, 640>;
var<workgroup> cnt: array<u32, 256>;
var<workgroup> part: array<f32, 256>;
var<workgroup> stat: array<f32, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let dim = p[0].x;
    let heads = p[0].y;
    let len = p[0].z;
    let g = wg.x;
    let tok = wg.y + wg.z * 65535u;
    for (var e = t; e < HG * dim; e += 256u) {
        qs[e] = q[(tok * heads + g * HG + e / dim) * dim + e % dim];
    }
    // the token's positions in order: each thread's run of the list counted, the counts scanned, each written at its place
    let per = (len + 255u) / 256u;
    var n = 0u;
    for (var k = 0u; k < per; k++) {
        let e = t * per + k;
        if (e < len && idxs[tok * len + e] != NONE) { n++; }
    }
    cnt[t] = n;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off * 2u) {
        var v = 0u;
        if (t >= off) { v = cnt[t - off]; }
        workgroupBarrier();
        cnt[t] = cnt[t] + v;
        workgroupBarrier();
    }
    var at = cnt[t] - n;
    for (var k = 0u; k < per; k++) {
        let e = t * per + k;
        if (e < len) {
            let r = idxs[tok * len + e];
            if (r != NONE) {
                rows[at] = r;
                at++;
            }
        }
    }
    let total = workgroupUniformLoad(&cnt[255]);
    // the scores: sixteen lanes a position, sixteen positions a pass, each lane's values eight at a time for every head
    let lane = t % 16u;
    let slot = t / 16u;
    for (var e0 = 0u; e0 < total; e0 += 16u) {
        let e = e0 + slot;
        var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if (e < total) {
            let row = rows[e] * dim;
            for (var i0 = lane; i0 < dim; i0 += 128u) {
                var v: array<f32, 8>;
                for (var j = 0u; j < 8u; j++) { v[j] = kv[row + i0 + 16u * j]; }
                for (var h = 0u; h < HG; h++) {
                    for (var j = 0u; j < 8u; j++) { acc[h] += qs[h * dim + i0 + 16u * j] * v[j]; }
                }
            }
        }
        for (var h = 0u; h < HG; h++) {
            part[t] = acc[h];
            workgroupBarrier();
            if (lane == 0u && e < total) {
                var l = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                for (var i = 0u; i < 16u; i++) { l[i] = part[t + i]; }
                for (var width = 8u; width > 0u; width = width / 2u) {
                    for (var i = 0u; i < width; i++) { l[i] += l[i + width]; }
                }
                sc[h * MOST + e] = l[0] * bitcast<f32>(p[1].x);
            }
            workgroupBarrier();
        }
    }
    // each head's largest score (32 threads a head), its weights, and their sum in order with the sink's
    let hh = t / 32u;
    let tl = t % 32u;
    var m = -1e30;
    for (var e = tl; e < total; e += 32u) { m = max(m, sc[hh * MOST + e]); }
    part[t] = m;
    workgroupBarrier();
    if (tl == 0u) {
        var mx = -1e30;
        for (var i = 0u; i < 32u; i++) { mx = max(mx, part[t + i]); }
        stat[hh] = mx;
    }
    workgroupBarrier();
    let mx = stat[hh];
    for (var e = tl; e < total; e += 32u) { sc[hh * MOST + e] = exp(sc[hh * MOST + e] - mx); }
    workgroupBarrier();
    if (tl == 0u) {
        var s = 0.0;
        for (var e = 0u; e < total; e++) { s += sc[hh * MOST + e]; }
        stat[8u + hh] = s + exp(sink[g * HG + hh] - mx);
    }
    workgroupBarrier();
    for (var e = tl; e < total; e += 32u) { sc[hh * MOST + e] = bf16(sc[hh * MOST + e]); }
    workgroupBarrier();
    // the rows' sums by the weights, in order: a thread an output column of every head
    for (var d = t; d < dim; d += 256u) {
        var o = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (var e = 0u; e < total; e++) {
            let v = kv[rows[e] * dim + d];
            for (var h = 0u; h < HG; h++) { o[h] += sc[h * MOST + e] * v; }
        }
        for (var h = 0u; h < HG; h++) {
            out[(tok * heads + g * HG + h) * dim + d] = bf16(div_signed(o[h], stat[8u + h]));
        }
    }
}
"#
);

/// `HcMix`'s kernel, one thread: the host's `mixes_of` of one token's stream from its 24 projections and its sum of
/// squares (the host's, the word `p[1].x` of the step's words): the scale the reciprocal square root of its mean
/// correctly rounded, then the coefficients (pre, post, Sinkhorn's comb by rows), their exponentials and the Sinkhorn's
/// divisions the device's own. `p[0]` = (the stream's width, Sinkhorn's iterations, the norm's eps's bits, the mixing's
/// eps's bits). (Its sum of squares made here, eight threads' running sums, was 400 us a call: their reads one after
/// another.)
pub(super) const DS_HC_MIX: &str = concat!(
    exact_fns!(),
    r#"
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
@group(0) @binding(3) var<storage, read> hs: array<f32>;
@group(0) @binding(4) var<storage, read> words: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
    if (t != 0u) { return; }
    let n = p[0].x;
    let sumsq = bitcast<f32>(words[p[1].x]);
    let eps = bitcast<f32>(p[0].w);
    let r = recip_rn(sqrt_rn(div_rn(sumsq, f32(n)) + bitcast<f32>(p[0].z)));
    var m: array<f32, 24>;
    for (var j = 0u; j < 24u; j++) { m[j] = proj[j] * r; }
    for (var j = 0u; j < 4u; j++) {
        y[j] = sigmoid(m[j] * hs[0] + base[j]) + eps;
        y[4u + j] = 2.0 * sigmoid(m[4u + j] * hs[1] + base[4u + j]);
    }
    var c: array<f32, 16>;
    for (var j = 0u; j < 4u; j++) {
        var mx = bitcast<f32>(0xff800000u);
        for (var k = 0u; k < 4u; k++) {
            c[4u * j + k] = m[8u + 4u * j + k] * hs[2] + base[8u + 4u * j + k];
            mx = max(mx, c[4u * j + k]);
        }
        var s = 0.0;
        for (var k = 0u; k < 4u; k++) {
            c[4u * j + k] = exp(c[4u * j + k] - mx);
            s += c[4u * j + k];
        }
        for (var k = 0u; k < 4u; k++) { c[4u * j + k] = c[4u * j + k] / s + eps; }
    }
    for (var it = 0u; it < max(p[0].y, 1u); it++) {
        if (it > 0u) {
            for (var j = 0u; j < 4u; j++) {
                var s = 0.0;
                for (var k = 0u; k < 4u; k++) { s += c[4u * j + k]; }
                let inv = 1.0 / (s + eps);
                for (var k = 0u; k < 4u; k++) { c[4u * j + k] *= inv; }
            }
        }
        for (var k = 0u; k < 4u; k++) {
            var s = 0.0;
            for (var j = 0u; j < 4u; j++) { s += c[4u * j + k]; }
            let inv = 1.0 / (s + eps);
            for (var j = 0u; j < 4u; j++) { c[4u * j + k] *= inv; }
        }
    }
    for (var i = 0u; i < 16u; i++) { y[8u + i] = c[i]; }
}
"#
);
