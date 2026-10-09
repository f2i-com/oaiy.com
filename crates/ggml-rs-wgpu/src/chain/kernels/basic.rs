//! The norms, the residual adds, the gated activations, the copies and the draft's argmax: a language model's small ops.

pub(in crate::chain) const HEAD: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
"#;

/// `y = x / sqrt(mean(x^2) + eps) * w` over `p[0].x` elements (`p[0].y` the bits of `eps`), one workgroup.
pub(in crate::chain) const RMSNORM: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = x[i]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    for (var i = li; i < n; i += 256u) { y[i] = x[i] * inv * bitcast<f32>(w[i]); }
}
"#;

/// [`RMSNORM`] of row `wg.x + 65535 wg.y` of `x` (rows of `p[0].x`, `p[0].w` of them), every row with the same
/// weights `w`, or with `p[0].z` rows of them the row's `row % p[0].z` (a hyper-connection's streams).
pub(in crate::chain) const RMSNORM_ROWS: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].w) { return; }
    let at = row * n;
    var wrows = p[0].z;
    if (wrows == 0u) { wrows = 1u; }
    let wat = (row % wrows) * n;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = x[at + i]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p[0].y));
    for (var i = li; i < n; i += 256u) { y[at + i] = x[at + i] * inv * bitcast<f32>(w[wat + i]); }
}
"#;

/// [`RMSNORM_ROWS`] of rows a multiple of 4 long: vec4 loads, a thread's squares in four running sums (its loads in
/// flight together, where one sum waited on each load in turn). `p[0]`: n, the bits of eps, weight rows, rows.
pub(in crate::chain) const RMSNORM_ROWS4: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y4: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;
var<workgroup> lead: array<f32, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n4 = p[0].x / 4u;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].w) { return; }
    let at = row * n4;
    var wrows = p[0].z;
    if (wrows == 0u) { wrows = 1u; }
    let wat = (row % wrows) * n4;
    var s = vec4<f32>(0.0);
    for (var i = li; i < n4; i += 256u) { let v = x4[at + i]; s += v * v; }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    // the threads' sums in sixteens, then the sixteens by one thread: three barriers, where a tree of halvings took
    // eight (a Flash-Next reply has 16,000 of these norms, 4 us each with the tree). Every thread adding the
    // sixteens for itself was 4,096 reads a row: a prompt's rows by the thousand were the slower for it (the 27B's
    // 15.6K tokens 7.0 to 7.6 s where 6.3 to 6.5).
    if (li < 16u) {
        var t = 0.0;
        for (var j = 0u; j < 16u; j++) { t += part[li * 16u + j]; }
        lead[li] = t;
    }
    workgroupBarrier();
    if (li == 0u) {
        var total = 0.0;
        for (var j = 0u; j < 16u; j++) { total += lead[j]; }
        lead[0] = total;
    }
    workgroupBarrier();
    let inv = 1.0 / sqrt(lead[0] / f32(p[0].x) + bitcast<f32>(p[0].y));
    for (var i = li; i < n4; i += 256u) { y4[at + i] = x4[at + i] * inv * w4[wat + i]; }
}
"#;

/// `x += y`, then each row of `x` RMS-normed into `out` with `w` (`[n]`, one for every row), rows a multiple of 4
/// long: a residual's add and the next norm in one dispatch. `p[0]`: n, the bits of eps.
pub(in crate::chain) const ADD_RMSNORM_ROWS4: &str = r#"
@group(0) @binding(0) var<storage, read> yv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> x4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;
var<workgroup> lead: array<f32, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n4 = p[0].x / 4u;
    let row = wg.x + wg.y * 65535u;
    if (row >= p[0].z) { return; }
    let at = row * n4;
    var s = vec4<f32>(0.0);
    for (var i = li; i < n4; i += 256u) {
        let v = x4[at + i] + yv[at + i];
        x4[at + i] = v;
        s += v * v;
    }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    // the threads' sums in sixteens, then the sixteens by one thread: three barriers, where a tree of halvings took
    // eight (a Flash-Next reply has 16,000 of these norms, 4 us each with the tree). Every thread adding the
    // sixteens for itself was 4,096 reads a row: a prompt's rows by the thousand were the slower for it (the 27B's
    // 15.6K tokens 7.0 to 7.6 s where 6.3 to 6.5).
    if (li < 16u) {
        var t = 0.0;
        for (var j = 0u; j < 16u; j++) { t += part[li * 16u + j]; }
        lead[li] = t;
    }
    workgroupBarrier();
    if (li == 0u) {
        var total = 0.0;
        for (var j = 0u; j < 16u; j++) { total += lead[j]; }
        lead[0] = total;
    }
    workgroupBarrier();
    let inv = 1.0 / sqrt(lead[0] / f32(p[0].x) + bitcast<f32>(p[0].y));
    for (var i = li; i < n4; i += 256u) { out[at + i] = x4[at + i] * inv * w4[i]; }
}
"#;

/// `y[i] += w[p[0].y] * x[i]` for `i < p[0].x`: a weighted sum's term, its weight read from the device.
pub(in crate::chain) const AXPY_AT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[i] = y[i] + bitcast<f32>(w[p[0].y]) * x[i]; }
}
"#;

/// `y += x` over `p[0].x` elements.
pub(in crate::chain) const ADD: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[i] = y[i] + x[i]; }
}
"#;

/// `y[r] = silu(x[r][..ff]) * x[r][ff..]` for each of `p[0].y` rows, `ff = p[0].x`.
pub(in crate::chain) const SILU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].x;
    if (i < ff * p[0].y) {
        let r = i / ff;
        let j = i % ff;
        let g = x[r * 2u * ff + j];
        y[i] = (g / (1.0 + exp(-g))) * x[r * 2u * ff + ff + j];
    }
}
"#;

/// `y[r] = gelu_approx(x[r][..ff]) * x[r][ff..]` (the tanh approximation, as the CPU's) for each of `p[0].y` rows.
pub(in crate::chain) const GELU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].x;
    if (i < ff * p[0].y) {
        let r = i / ff;
        let j = i % ff;
        let g = x[r * 2u * ff + j];
        let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
        y[i] = 0.5 * g * (1.0 + tanh(clamp(inner, -15.0, 15.0))) * x[r * 2u * ff + ff + j];
    }
}
"#;

/// [`ChainRecorder::argmax_rows`]'s first pass over rows of `p[0].x` logits (a row a line of the grid's second axis):
/// a workgroup 4,096 of a row's, a thread 16 (its largest, the first of equals, and the sum of exponentials against it
/// as it goes), then the threads' combined, the lower index of equals taken; row `r`'s workgroup `g`'s largest, its
/// index in the row (as bits) and its sum to `parts[4 (r p[0].y + g)..]` (`p[0].y` the workgroups a row).
///
/// One workgroup took them all at first, 970 of a vocabulary's 248,320 a thread one after another with an
/// exponential each: 160 us a draft, a sixth of what a draft costs.
pub(in crate::chain) const ARGMAX_SOFTMAX_PARTS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> parts: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> mv: array<f32, 256>;
var<workgroup> mi: array<u32, 256>;
var<workgroup> ms: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let row = wg.y;
    var m = -3.4e38;
    var idx = 0xffffffffu;
    var s = 0.0;
    for (var j = 0u; j < 16u; j++) {
        let i = wg.x * 4096u + j * 256u + t;
        if (i < n) {
            let v = x[row * n + i];
            if (v > m) {
                s = s * exp(m - v) + 1.0;
                m = v;
                idx = i;
            } else {
                s += exp(v - m);
            }
        }
    }
    mv[t] = m;
    mi[t] = idx;
    ms[t] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (t < st) {
            let m1 = mv[t];
            let m2 = mv[t + st];
            let i1 = mi[t];
            let i2 = mi[t + st];
            let take = m2 > m1 || (m2 == m1 && i2 < i1);
            let mm = select(m1, m2, take);
            ms[t] = ms[t] * exp(m1 - mm) + ms[t + st] * exp(m2 - mm);
            mv[t] = mm;
            mi[t] = select(i1, i2, take);
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let at = 4u * (row * p[0].y + wg.x);
        parts[at] = mv[0];
        parts[at + 1u] = bitcast<f32>(mi[0]);
        parts[at + 2u] = ms[0];
    }
}
"#;

/// [`ChainRecorder::argmax_rows`]'s second pass: a row's `p[0].x` parts ([`ARGMAX_SOFTMAX_PARTS`]'s) combined the same
/// way, a workgroup a row `r`: the largest's index (as bits), the largest and the sum to `out[4 r..]`.
pub(in crate::chain) const ARGMAX_SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> parts: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> mv: array<f32, 256>;
var<workgroup> mi: array<u32, 256>;
var<workgroup> ms: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let count = p[0].x;
    let row = wg.x;
    var m = -3.4e38;
    var idx = 0xffffffffu;
    var s = 0.0;
    for (var g = t; g < count; g += 256u) {
        let at = 4u * (row * count + g);
        let m2 = parts[at];
        let i2 = bitcast<u32>(parts[at + 1u]);
        let take = m2 > m || (m2 == m && i2 < idx);
        let mm = select(m, m2, take);
        s = s * exp(m - mm) + parts[at + 2u] * exp(m2 - mm);
        m = mm;
        idx = select(idx, i2, take);
    }
    mv[t] = m;
    mi[t] = idx;
    ms[t] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (t < st) {
            let m1 = mv[t];
            let m2 = mv[t + st];
            let i1 = mi[t];
            let i2 = mi[t + st];
            let take = m2 > m1 || (m2 == m1 && i2 < i1);
            let mm = select(m1, m2, take);
            ms[t] = ms[t] * exp(m1 - mm) + ms[t + st] * exp(m2 - mm);
            mv[t] = mm;
            mi[t] = select(i1, i2, take);
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        out[4u * row] = bitcast<f32>(mi[0]);
        out[4u * row + 1u] = mv[0];
        out[4u * row + 2u] = ms[0];
    }
}
"#;

/// `y[p[0].y + i] = x[p[0].z + i]` for `i < p[0].x`.
pub(in crate::chain) const COPY: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[p[0].y + i] = x[p[0].z + i]; }
}
"#;

/// `y[r * p[0].x + i] = x[r * p[0].z + p[0].w + i]` for `r < p[0].y` rows of `p[0].x`: columns of each row.
pub(in crate::chain) const COPY_COLS: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = p[0].x;
    let i = id.x + id.y * 16776960u;
    if (i < width * p[0].y) { y[i] = x[(i / width) * p[0].z + p[0].w + i % width]; }
}
"#;

/// `y[i] = gelu(x[i])` for `i < p[0].x`, the tanh approximation (as [`GELU_MUL_SPLIT`]'s).
/// [`GELU`] exactly: `x (1 + erf(x / sqrt 2)) / 2`, erf as Abramowitz and Stegun's 7.1.26 (within 1.5e-7).
pub(in crate::chain) const GELU_ERF: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) {
        let g = x[i];
        let z = abs(g) * 0.7071067811865476;
        let t = 1.0 / (1.0 + 0.3275911 * z);
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erf = sign(g) * (1.0 - poly * exp(-z * z));
        y[i] = 0.5 * g * (1.0 + erf);
    }
}
"#;

pub(in crate::chain) const GELU: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) {
        let g = x[i];
        let inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
        y[i] = 0.5 * g * (1.0 + tanh(clamp(inner, -15.0, 15.0)));
    }
}
"#;

/// `y[i] = w[i] * sigmoid(x[i])` for `i < p[0].x` (`w` the values, `x` the gate), as the CPU's mul_sigmoid.
pub(in crate::chain) const MUL_SIGMOID: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) { y[i] = bitcast<f32>(w[i]) * (1.0 / (1.0 + exp(-x[i]))); }
}
"#;

/// `y[i] = silu(w[i]) * x[i]` for `i < p[0].x` (`w` the gate, `x` the up projection).
pub(in crate::chain) const SILU_MUL: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i < p[0].x) {
        let g = bitcast<f32>(w[i]);
        y[i] = (g / (1.0 + exp(-g))) * x[i];
    }
}
"#;
