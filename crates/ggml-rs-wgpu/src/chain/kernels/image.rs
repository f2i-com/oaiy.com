//! The diffusion transformers', the VAEs' and the UNet's ops: modulated norms, group norms, the shuffles and resamplings, a convolution's f16 input and its f32 tiled kernel.

/// Each row of `x` (rows of `p[0].x`, a workgroup a row: `wg.x + wg.y * 65535`) normed as `p[1].y` says (0 none, 1
/// over its RMS, 2 a layer norm: the mean, then the mean square of the deviations; no weights, `eps` the bits of
/// `p[0].y`) into `y`, times `1 + mods[p[0].z + i]` and plus `mods[p[0].w + i]` unless `p[0].w` is all ones.
/// `p[1].x`: the rows.
pub(in crate::chain) const LAYERNORM_MOD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let r = wg.x + wg.y * 65535u;
    if (r >= p[1].x) { return; }
    let at = r * n;
    let mode = p[1].y & 3u;
    // a clean row (before p[1].z, or from p[1].w on) modulated by the second set
    let second = select(0u, p[1].y >> 2u, r < p[1].z || r >= p[1].w);
    var mean = 0.0;
    if (mode == 2u) {
        var s = 0.0;
        for (var i = li; i < n; i += 256u) { s += x[at + i]; }
        part[li] = s;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
            if (li < stride) { part[li] += part[li + stride]; }
            workgroupBarrier();
        }
        mean = part[0] / f32(n);
        workgroupBarrier();
    }
    var inv = 1.0;
    if (mode != 0u) {
        var q = 0.0;
        for (var i = li; i < n; i += 256u) { let d = x[at + i] - mean; q += d * d; }
        part[li] = q;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
            if (li < stride) { part[li] += part[li + stride]; }
            workgroupBarrier();
        }
        inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    }
    let shifted = p[0].w != 0xffffffffu;
    for (var i = li; i < n; i += 256u) {
        var v = (x[at + i] - mean) * inv * (1.0 + mods[second + p[0].z + i]);
        if (shifted) { v += mods[second + p[0].w + i]; }
        y[at + i] = v;
    }
}
"#;

/// `y[i] *= 2 sigmoid(logits[i / p[0].x])` for `i < p[0].y` (`p[0].x` a head's width): a gated attention's heads.
/// [`ChainRecorder::group_norm_rows`]'s sums: a workgroup a (group, 256 pixels), each thread a pixel's values of the
/// group (their sum and their squares' less the group's first value: sums of what is near zero keep their digits),
/// the workgroup's sums into `stats[(g * chunks + j) * 2..]`. `p[0]`: pixels, channels, groups, chunks.
pub(in crate::chain) const GROUP_NORM_SUMS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<vec2<f32>, 256>;
var<workgroup> some: array<vec2<f32>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let pixels = p[0].x;
    let c = p[0].y;
    let cpg = c / p[0].z;
    let chunks = p[0].w;
    let g = wg.x;
    let j = wg.y;
    let px = j * 256u + li;
    // (the group's first value: what every value is taken from, for the sums' digits)
    let pilot = x[g * cpg];
    var sums = vec2<f32>(0.0);
    if (px < pixels) {
        let at = px * c + g * cpg;
        for (var i = 0u; i < cpg; i++) {
            let v = x[at + i] - pilot;
            sums += vec2<f32>(v, v * v);
        }
    }
    part[li] = sums;
    workgroupBarrier();
    if (li < 16u) {
        var t = vec2<f32>(0.0);
        for (var i = 0u; i < 16u; i++) { t += part[li * 16u + i]; }
        some[li] = t;
    }
    workgroupBarrier();
    if (li == 0u) {
        var t = vec2<f32>(0.0);
        for (var i = 0u; i < 16u; i++) { t += some[i]; }
        let o = (g * chunks + j) * 2u;
        stats[o] = t.x;
        stats[o + 1u] = t.y;
    }
}
"#;

/// [`ChainRecorder::group_norm_rows`]'s mean and scale: a workgroup a group, its chunks' sums added up, the group's
/// mean and `1 / sqrt(variance + eps)` after the chunks' sums (`stats[(groups * chunks + g) * 2..]`). `p[0]`: pixels,
/// channels, groups, chunks; `p[1].x`: eps's bits.
pub(in crate::chain) const GROUP_NORM_STATS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<vec2<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let cpg = p[0].y / p[0].z;
    let chunks = p[0].w;
    let g = wg.x;
    var t = vec2<f32>(0.0);
    for (var j = li; j < chunks; j += 64u) {
        let o = (g * chunks + j) * 2u;
        t += vec2<f32>(stats[o], stats[o + 1u]);
    }
    part[li] = t;
    workgroupBarrier();
    if (li == 0u) {
        var all = vec2<f32>(0.0);
        for (var i = 0u; i < 64u; i++) { all += part[i]; }
        let n = f32(p[0].x) * f32(cpg);
        let m = all.x / n;
        let variance = max(all.y / n - m * m, 0.0);
        let o = (p[0].z * chunks + g) * 2u;
        stats[o] = x[g * cpg] + m;
        stats[o + 1u] = inverseSqrt(variance + bitcast<f32>(p[1].x));
    }
}
"#;

/// [`ChainRecorder::group_norm_rows`]'s rows: each value less its group's mean, times its scale and its channel's
/// weight, plus the channel's bias, through SiLU where `p[1].y` is 1. `p[0]`: pixels, channels, groups, chunks.
pub(in crate::chain) const GROUP_NORM_APPLY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].y;
    if (i >= p[0].x * c) { return; }
    let ch = i % c;
    let o = (p[0].z * p[0].w + ch / (c / p[0].z)) * 2u;
    let v = (x[i] - stats[o]) * stats[o + 1u] * weight[ch] + bias[ch];
    y[i] = select(v, v / (1.0 + exp(-v)), p[1].y == 1u);
}
"#;

/// [`ChainRecorder::geglu_rows`]: `p[0]` rows and `ff`.
pub(in crate::chain) const GEGLU_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].y;
    if (i >= p[0].x * ff) { return; }
    let at = (i / ff) * 2u * ff + i % ff;
    let g = x[at + ff];
    let z = abs(g) * 0.7071067811865476;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign(g) * (1.0 - poly * exp(-z * z));
    y[i] = x[at] * 0.5 * g * (1.0 + erf);
}
"#;

/// `y[(oy, ox), c] = x[(2 oy, 2 ox), c]`: `p[0]` the input's rows, columns and channels.
pub(in crate::chain) const SUBSAMPLE2X_EVEN_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let ow = w / 2u;
    if (i >= (h / 2u) * ow * c) { return; }
    let px = i / c;
    y[i] = x[(2u * (px / ow) * w + 2u * (px % ow)) * c + i % c];
}
"#;

/// [`ChainRecorder::nag_mix`]: a workgroup a row, each thread its share of the row's values four at a time; the row's
/// L1 norms (the guided output's and the plain one's) summed by sixteens through the workgroup's memory. `p[0]`: the
/// row's width (a multiple of 4), rows, and the bits of scale and tau; `p[1].x`: alpha's.
pub(in crate::chain) const NAG_MIX: &str = r#"
@group(0) @binding(0) var<storage, read> neg: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> part: array<vec2<f32>, 256>;
var<workgroup> some: array<vec2<f32>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let w4 = p[0].x / 4u;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].y) { return; }
    let scale = bitcast<f32>(p[0].z);
    let tau = bitcast<f32>(p[0].w);
    let alpha = bitcast<f32>(p[1].x);
    let at = row * w4;
    var sums = vec2<f32>(0.0);
    for (var c = li; c < w4; c += 256u) {
        let a = pos[at + c];
        let g = abs(a * scale - neg[at + c] * (scale - 1.0));
        let q = abs(a);
        sums += vec2<f32>(g.x + g.y + g.z + g.w, q.x + q.y + q.z + q.w);
    }
    part[li] = sums;
    workgroupBarrier();
    if (li < 16u) {
        var s = vec2<f32>(0.0);
        for (var i = 0u; i < 16u; i++) { s += part[li * 16u + i]; }
        some[li] = s;
    }
    workgroupBarrier();
    var total = vec2<f32>(0.0);
    for (var i = 0u; i < 16u; i++) { total += some[i]; }
    // the guided row within tau times the plain one's size
    let factor = clamp(tau * (total.y + 1.0e-6) / max(total.x, 1.0e-30), 0.0, 1.0);
    for (var c = li; c < w4; c += 256u) {
        let a = pos[at + c];
        let g = a * scale - neg[at + c] * (scale - 1.0);
        pos[at + c] = g * (factor * alpha) + a * (1.0 - alpha);
    }
}
"#;

pub(in crate::chain) const HEAD_GATE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].y) { y[i] *= 2.0 / (1.0 + exp(-logits[i / p[0].x])); }
}
"#;

/// `x[i] += y[i] * g` for `i < p[0].x * p[0].y` (rows of `p[0].x`), `g` `mods[p[0].z + i % p[0].x]`, its tanh where
/// `p[0].w` is 1.
/// `y[(oy, ox), c] = x[(2 oy + 1, 2 ox + 1), c]`: `p[0]` the input's rows, columns and channels.
pub(in crate::chain) const SUBSAMPLE2X_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let ow = w / 2u;
    if (i >= (h / 2u) * ow * c) { return; }
    let ch = i % c;
    let px = i / c;
    let oy = px / ow;
    let ox = px % ow;
    y[i] = x[((2u * oy + 1u) * w + 2u * ox + 1u) * c + ch];
}
"#;

/// `y[(oy, ox), co] += mean(group co)` of `x`'s space-to-depth: `p[0]` the input's rows, columns, channels in and out,
/// `p[1]` the time slots and the spatial factor (each output pixel's `cin ft fs fs` values in turn by channel, slot, row
/// and column, the slots before the last zero).
pub(in crate::chain) const SHUFFLE_DOWN_MEAN_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let cin = p[0].z;
    let cout = p[0].w;
    let ft = p[1].x;
    let fs = p[1].y;
    let ow = w / fs;
    if (i >= (h / fs) * ow * cout) { return; }
    let co = i % cout;
    let px = i / cout;
    let oy = px / ow;
    let ox = px % ow;
    let per = fs * fs;
    let g = cin * ft * per / cout;
    var s = 0.0;
    for (var e = co * g; e < (co + 1u) * g; e++) {
        let t = (e / per) % ft;
        if (t == ft - 1u) {
            let c = e / (ft * per);
            let fy = (e / fs) % fs;
            let fx = e % fs;
            s += x[((oy * fs + fy) * w + ox * fs + fx) * cin + c];
        }
    }
    y[i] += s / f32(g);
}
"#;

/// Space to depth, the time slots one frame's: `p[0]` the input's rows, columns and channels, `p[1]` the time slots and
/// the rows' and columns' factors.
pub(in crate::chain) const SPACE_TO_DEPTH_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let w = p[0].y;
    let c = p[0].z;
    let st = p[1].x;
    let sh = p[1].y;
    let sw = p[1].z;
    let co = c * st * sh * sw;
    let ow = w / sw;
    if (i >= (p[0].x / sh) * ow * co) { return; }
    let e = i % co;
    let px = i / co;
    let oy = px / ow;
    let ox = px % ow;
    let ch = e / (st * sh * sw);
    let fy = (e / sw) % sh;
    let fx = e % sw;
    y[i] = x[((oy * sh + fy) * w + ox * sw + fx) * c + ch];
}
"#;

/// `y[r, co] += mean(x[r, co g .. (co + 1) g])`: `p[0]` the rows, the channels in and out.
pub(in crate::chain) const GROUP_MEAN_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let cin = p[0].y;
    let cout = p[0].z;
    if (i >= p[0].x * cout) { return; }
    let g = cin / cout;
    let at = (i / cout) * cin + (i % cout) * g;
    var s = 0.0;
    for (var e = 0u; e < g; e++) { s += x[at + e]; }
    y[i] += s / f32(g);
}
"#;

pub(in crate::chain) const ADD_GATED_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> yv: array<f32>;
@group(0) @binding(1) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    // a clean row (before p[1].x, or from p[1].y on) gated by the second set (p[1].z on)
    let row = i / p[0].x;
    let second = select(0u, p[1].z, row < p[1].x || row >= p[1].y);
    var g = mods[second + p[0].z + i % p[0].x];
    if (p[0].w == 1u) { g = tanh(g); }
    x[i] += yv[i] * g;
}
"#;

/// An image's pixels' rows of `p[0].x` channels f32 (`x`) as f16 padded to `p[0].y` (`q`, two to a word), `p[0].z`
/// pixels, each times `range[0]` ([`F16_RANGE_SET`]'s scale).
pub(in crate::chain) const X_F16_PADDED: &str = r#"
@group(0) @binding(0) var<storage, read> range: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = p[0].x;
    let cp2 = p[0].y / 2u;
    // (a pixel's values `p[0].w` apart: its first `c` taken)
    let xs = p[0].w;
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].z * cp2) { return; }
    let px = i / cp2;
    let j = 2u * (i % cp2);
    var v = vec2<f32>(0.0);
    if (j < c) { v.x = x[px * xs + j]; }
    if (j + 1u < c) { v.y = x[px * xs + j + 1u]; }
    q[i] = pack2x16float(v * range[0]);
}
"#;

/// The largest magnitude of `x` (`p[0].x` values) into `range[2]` (its bits: a non-negative f32's order is its bits'),
/// zeroed first ([`F16_RANGE_CLEAR`]).
pub(in crate::chain) const F16_RANGE_MAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> range: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { atomicMax(&range[2], bitcast<u32>(abs(x[i]))); }
}
"#;

/// `range[2]` zeroed for [`F16_RANGE_MAX`].
pub(in crate::chain) const F16_RANGE_CLEAR: &str = r#"
@group(0) @binding(6) var<storage, read_write> range: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(1)
fn main() {
    range[2] = 0u;
}
"#;

/// A power of two that keeps `x`'s largest (`range[2]`, [`F16_RANGE_MAX`]'s) within 16,384 as f16: `range[0]` it,
/// `range[1]` its inverse (exact both).
pub(in crate::chain) const F16_RANGE_SET: &str = r#"
@group(0) @binding(6) var<storage, read_write> range: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(1)
fn main() {
    let m = bitcast<f32>(range[2]);
    var k = 0.0;
    if (m > 16384.0) { k = ceil(log2(m / 16384.0)); }
    k = min(k, 100.0);
    range[0] = bitcast<u32>(exp2(-k));
    range[1] = bitcast<u32>(exp2(k));
}
"#;

/// `x[i] = x[i] * range[1] + b[i % p[0].x]` for `i < p[0].x * p[0].y`: a scaled convolution's sums back to their
/// range, and its bias.
pub(in crate::chain) const UNSCALE_BIAS_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> b: array<f32>;
@group(0) @binding(1) var<storage, read> range: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x * p[0].y) { x[i] = x[i] * range[1] + b[i % p[0].x]; }
}
"#;

/// [`ChainRecorder::depth_to_space_rows`]: `y`'s voxel (frames of `p[1].y` rows of `p[1].z`) of `p[0].x` channels
/// from `x`'s. `p[0]`: c, st, sh, sw; `p[1]`: the output's voxels, its rows a frame, its row's columns, the frames
/// dropped first.
pub(in crate::chain) const DEPTH_TO_SPACE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = p[0].x;
    let st = p[0].y;
    let sh = p[0].z;
    let sw = p[0].w;
    let oh = p[1].y;
    let ow = p[1].z;
    let i = id.x + id.y * 16776960u;
    if (i >= p[1].x * c) { return; }
    let ch = i % c;
    let v = i / c;
    let ox = v % ow;
    let oy = (v / ow) % oh;
    let ot = v / (ow * oh) + p[1].w;
    let h = oh / sh;
    let w = ow / sw;
    let src = ((ot / st) * h + oy / sh) * w + ox / sw;
    let sc = ((ch * st + ot % st) * sh + oy % sh) * sw + ox % sw;
    y[i] = x[src * (c * st * sh * sw) + sc];
}
"#;

/// `x[i] += b[i % p[0].x]` for `i < p[0].x * p[0].y`.
pub(in crate::chain) const ADD_BIAS_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x * p[0].y) { x[i] += b[i % p[0].x]; }
}
"#;

/// Nearest 2x of an image of `p[0].y` by `p[0].z` pixels of `p[0].x` channels: `y`'s pixel `(oy, ox)` is `x`'s
/// `(oy / 2, ox / 2)`.
pub(in crate::chain) const UPSAMPLE2X_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = p[0].x;
    let h = p[0].y;
    let w = p[0].z;
    let i = id.x + id.y * 16776960u;
    if (i >= 4u * h * w * c) { return; }
    let ch = i % c;
    let px = i / c;
    let ox = px % (2u * w);
    let oy = px / (2u * w);
    y[i] = x[((oy / 2u) * w + ox / 2u) * c + ch];
}
"#;

/// [`ChainRecorder::shuffle_up_add_rows`]: `y` (`[2h * 2w, cout]`) gets `x` (`[h * w, cin]`) shuffled up. `p[0]`:
/// cin, cout, ft, repeats; `p[1]`: h, w.
pub(in crate::chain) const SHUFFLE_UP_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cin = p[0].x;
    let cout = p[0].y;
    let ft = p[0].z;
    let repeats = p[0].w;
    let h = p[1].x;
    let w = p[1].y;
    let i = id.x + id.y * 16776960u;
    if (i >= 4u * h * w * cout) { return; }
    let co = i % cout;
    let px = i / cout;
    let ox = px % (2u * w);
    let oy = px / (2u * w);
    let a = oy % 2u;
    let b = ox % 2u;
    let ci = (4u * co * ft + 4u * (ft - 1u) + 2u * a + b) / repeats;
    y[i] += x[((oy / 2u) * w + ox / 2u) * cin + ci];
}
"#;

/// A convolution without tensor cores ([`ChainRecorder::conv_rows`], [`ChainRecorder::conv3d_rows`]): the f32 tiled
/// matmul's 64 voxels by 64 outputs a workgroup ([`MATMUL_F32_TILED`]), its tokens' tile gathered through the taps
/// as the tensor cores' kernel takes them (a 3x3's pixel `(y + dy - 1, x + dx - 1)`, zeros past the frame's edge; a
/// 3x3x3's from frame `t + dt - 1` clamped to the clip), the weights the same packed f16 (`[cout][taps][cin padded
/// to 32]`, two to a word), the bias added. `p[0]`: `cout`, `cin`, the voxels, the values a voxel of `x` apart (its
/// first `cin`); `p[1]`: the taps, a row's pixels, a frame's rows, the dispatch's first tile of voxels.
pub(in crate::chain) const CONV_F32_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// [16 of k][64 voxels (or outputs)], a row of 17 vec4s (the 17th padding, against bank conflicts)
var<workgroup> xs: array<vec4<f32>, 272>;
var<workgroup> ws: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let cin = p[0].y;
    let m = p[0].z;
    let stride = p[0].w;
    let cp = (cin + 31u) / 32u * 32u;
    let taps = p[1].x;
    let wd = p[1].y;
    let h = p[1].z;
    let plane = h * wd;
    let frames = m / plane;
    let kt = taps * cp;
    let o0 = wg.x * 64u;
    let r0 = (p[1].w + wg.y + wg.z * 65535u) * 64u;
    let tr = t / 16u;
    let to = t % 16u;
    let kk = t % 16u;
    // the voxels this thread loads (rows `t / 16 + 16 q` of the tile): their frame, row and column
    let pa = min(r0 + tr, m - 1u);
    let pb = min(r0 + tr + 16u, m - 1u);
    let pc = min(r0 + tr + 32u, m - 1u);
    let pd = min(r0 + tr + 48u, m - 1u);
    let fa = i32(pa / plane); let ya = i32((pa % plane) / wd); let xa = i32(pa % wd);
    let fb = i32(pb / plane); let yb = i32((pb % plane) / wd); let xb = i32(pb % wd);
    let fc = i32(pc / plane); let yc = i32((pc % plane) / wd); let xc = i32(pc % wd);
    let fd = i32(pd / plane); let yd = i32((pd % plane) / wd); let xd = i32(pd % wd);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var kb = 0u; kb < kt; kb += 16u) {
        let gk = kb + kk;
        let tap = gk / cp;
        let c = gk % cp;
        let live = gk < kt && c < cin;
        // the tap's offsets in time, rows and columns
        var dt = 0;
        var dy = 0;
        var dx = 0;
        if (taps == 27u) {
            dt = i32(tap / 9u) - 1;
            dy = i32((tap / 3u) % 3u) - 1;
            dx = i32(tap % 3u) - 1;
        } else if (taps == 49u) {
            dy = i32(tap / 7u) - 3;
            dx = i32(tap % 7u) - 3;
        } else if (taps == 9u) {
            dy = i32(tap / 3u) - 1;
            dx = i32(tap % 3u) - 1;
        }
        for (var q = 0u; q < 4u; q++) {
            var f = fa;
            var yy = ya;
            var xx = xa;
            var r = r0 + tr;
            if (q == 1u) { f = fb; yy = yb; xx = xb; r = r0 + tr + 16u; }
            if (q == 2u) { f = fc; yy = yc; xx = xc; r = r0 + tr + 32u; }
            if (q == 3u) { f = fd; yy = yd; xx = xd; r = r0 + tr + 48u; }
            let sy = yy + dy;
            let sx = xx + dx;
            let sf = clamp(f + dt, 0, i32(frames) - 1);
            var v = 0.0;
            if (live && r < m && sy >= 0 && sy < i32(h) && sx >= 0 && sx < i32(wd)) {
                v = x[((u32(sf) * h + u32(sy)) * wd + u32(sx)) * stride + c];
            }
            let rr = tr + 16u * q;
            xs[kk * 17u + rr / 4u][rr % 4u] = v;
            var u = 0.0;
            if (o0 + rr < n && gk < kt) {
                let e = (o0 + rr) * kt + gk;
                u = unpack2x16float(w[e / 2u])[e % 2u];
            }
            ws[kk * 17u + rr / 4u][rr % 4u] = u;
        }
        workgroupBarrier();
        for (var j = 0u; j < 16u; j++) {
            let xv = xs[j * 17u + tr];
            let wv = ws[j * 17u + to];
            a0 += xv.x * wv;
            a1 += xv.y * wv;
            a2 += xv.z * wv;
            a3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let o = o0 + to * 4u;
    let r = r0 + tr * 4u;
    let acc = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i = 0u; i < 4u; i++) {
        if (r + i < m) {
            for (var j = 0u; j < 4u; j++) {
                if (o + j < n) {
                    y[(r + i) * n + o + j] = acc[i][j] + bias[o + j];
                }
            }
        }
    }
}
"#;

/// The most work (FLOPs) one dispatch of a convolution without tensor cores takes on: its voxels' tiles in chunks past
/// it, as a prompt's attention's ([`ATTENTION_DISPATCH_FLOPS`]).
pub(in crate::chain) const CONV_DISPATCH_FLOPS: f64 = (1u64 << 37) as f64;
