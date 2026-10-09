//! The picture tools' and the sound models' ops: Swin's windows, resizes, a deformable convolution's taps, the 1-D convolutions, Snake, and their small row ops.

/// Swin's windows ([`ChainRecorder::window_rows`]): a thread an output value. `p[0]`: `h`, `w`, `c`, `win`; `p[1]`: the
/// shift, the padded grid's `hp` and `wp`.
pub(in crate::chain) const WINDOW_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let w = p[0].y;
    let c = p[0].z;
    let win = p[0].w;
    let shift = p[1].x;
    let hp = p[1].y;
    let wp = p[1].z;
    if (i >= hp * wp * c) { return; }
    let ch = i % c;
    let tok = i / c;
    let n = win * win;
    let wdx = tok / n;
    let t = tok % n;
    let gw = wp / win;
    let py = ((wdx / gw) * win + t / win + shift) % hp;
    let px = ((wdx % gw) * win + t % win + shift) % wp;
    var v = 0.0;
    if (py < p[0].x && px < w) { v = x[(py * w + px) * c + ch]; }
    out[i] = v;
}
"#;

/// [`WINDOW_ROWS`]' way back, added ([`ChainRecorder::unwindow_add_rows`]): a thread a token's value. `p` as its.
pub(in crate::chain) const UNWINDOW_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> xw: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let w = p[0].y;
    let c = p[0].z;
    let win = p[0].w;
    let shift = p[1].x;
    let hp = p[1].y;
    let wp = p[1].z;
    if (i >= p[0].x * w * c) { return; }
    let ch = i % c;
    let tok = i / c;
    let ry = (tok / w + hp - shift) % hp;
    let rx = (tok % w + wp - shift) % wp;
    let wdx = (ry / win) * (wp / win) + rx / win;
    let t = (ry % win) * win + rx % win;
    out[i] += xw[(wdx * win * win + t) * c + ch];
}
"#;

/// Swin's window attention, heads 32 wide ([`ChainRecorder::window_attention`]): a workgroup a (window, head), a thread
/// a query (its 32 dims and sums in registers), the keys in turn (each a broadcast: every thread reads the same), the
/// softmax online. `p[0]`: the padded grid's `hp` and `wp`, the heads, `win`; `p[1]`: the shift, the scale's bits.
pub(in crate::chain) const WINDOW_ATTENTION: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// a coordinate's region of the rolled grid `n` long (Swin's slices: up to the last window, to the shift, the rest)
fn region(i: u32, n: u32, win: u32, shift: u32) -> u32 {
    if (i < n - win) { return 0u; }
    if (i < n - shift) { return 1u; }
    return 2u;
}

@compute @workgroup_size(N_u)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let hp = p[0].x;
    let wp = p[0].y;
    let heads = p[0].z;
    let win = p[0].w;
    let shift = p[1].x;
    let scale = bitcast<f32>(p[1].y);
    let wdx = wg.x;
    let head = wg.y;
    let n = win * win;
    let c4 = heads * 8u;
    let row4 = 3u * c4;
    let base = wdx * n;
    let qo = (base + t) * row4 + head * 8u;
    let q0 = qkv[qo] * scale;
    let q1 = qkv[qo + 1u] * scale;
    let q2 = qkv[qo + 2u] * scale;
    let q3 = qkv[qo + 3u] * scale;
    let q4 = qkv[qo + 4u] * scale;
    let q5 = qkv[qo + 5u] * scale;
    let q6 = qkv[qo + 6u] * scale;
    let q7 = qkv[qo + 7u] * scale;
    let gw = wp / win;
    let qy = t / win;
    let qx = t % win;
    let gy = (wdx / gw) * win;
    let gx = (wdx % gw) * win;
    var qr = 0u;
    if (shift > 0u) { qr = region(gy + qy, hp, win, shift) * 3u + region(gx + qx, wp, win, shift); }
    var m = -3.0e38;
    var l = 0.0;
    var a0 = vec4<f32>();
    var a1 = vec4<f32>();
    var a2 = vec4<f32>();
    var a3 = vec4<f32>();
    var a4 = vec4<f32>();
    var a5 = vec4<f32>();
    var a6 = vec4<f32>();
    var a7 = vec4<f32>();
    for (var j = 0u; j < n; j++) {
        let ko = (base + j) * row4 + c4 + head * 8u;
        var s = dot(q0, qkv[ko]) + dot(q1, qkv[ko + 1u]) + dot(q2, qkv[ko + 2u]) + dot(q3, qkv[ko + 3u]) + dot(q4, qkv[ko + 4u]) + dot(q5, qkv[ko + 5u]) + dot(q6, qkv[ko + 6u]) + dot(q7, qkv[ko + 7u]);
        let ky = j / win;
        let kx = j % win;
        s += table[((qy + win - 1u - ky) * (2u * win - 1u) + qx + win - 1u - kx) * heads + head];
        if (shift > 0u && region(gy + ky, hp, win, shift) * 3u + region(gx + kx, wp, win, shift) != qr) { s -= 100.0; }
        let mn = max(m, s);
        let a = exp(m - mn);
        let e = exp(s - mn);
        l = l * a + e;
        m = mn;
        let vo = ko + c4;
        a0 = a0 * a + e * qkv[vo];
        a1 = a1 * a + e * qkv[vo + 1u];
        a2 = a2 * a + e * qkv[vo + 2u];
        a3 = a3 * a + e * qkv[vo + 3u];
        a4 = a4 * a + e * qkv[vo + 4u];
        a5 = a5 * a + e * qkv[vo + 5u];
        a6 = a6 * a + e * qkv[vo + 6u];
        a7 = a7 * a + e * qkv[vo + 7u];
    }
    let inv = 1.0 / l;
    let oo = (base + t) * c4 + head * 8u;
    out[oo] = a0 * inv;
    out[oo + 1u] = a1 * inv;
    out[oo + 2u] = a2 * inv;
    out[oo + 3u] = a3 * inv;
    out[oo + 4u] = a4 * inv;
    out[oo + 5u] = a5 * inv;
    out[oo + 6u] = a6 * inv;
    out[oo + 7u] = a7 * inv;
}
"#;

/// A bilinear resize with the corners aligned ([`ChainRecorder::resize_bilinear_rows`]): a thread an output value.
/// `p[0]`: `h`, `w`, `c`, `oh`; `p[1]`: `ow`.
pub(in crate::chain) const RESIZE_BILINEAR_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let oh = p[0].w;
    let ow = p[1].x;
    if (i >= oh * ow * c) { return; }
    let ch = i % c;
    let ox = (i / c) % ow;
    let oy = i / (c * ow);
    var sy = 0.0;
    if (oh > 1u) { sy = f32(h - 1u) / f32(oh - 1u) * f32(oy); }
    var sx = 0.0;
    if (ow > 1u) { sx = f32(w - 1u) / f32(ow - 1u) * f32(ox); }
    let y0 = min(u32(sy), h - 1u);
    let x0 = min(u32(sx), w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let x1 = min(x0 + 1u, w - 1u);
    let fy = sy - f32(y0);
    let fx = sx - f32(x0);
    let top = (1.0 - fx) * x[(y0 * w + x0) * c + ch] + fx * x[(y0 * w + x1) * c + ch];
    let bottom = (1.0 - fx) * x[(y1 * w + x0) * c + ch] + fx * x[(y1 * w + x1) * c + ch];
    out[i] = (1.0 - fy) * top + fy * bottom;
}
"#;

/// A picture as patches ([`ChainRecorder::blocks_to_channels_rows`]): a thread an output value. `p[0]`: the picture's
/// side, `c`, the patches' side.
pub(in crate::chain) const BLOCKS_TO_CHANNELS_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let s = p[0].x;
    let c = p[0].y;
    let size = p[0].z;
    let g = s / size;
    let cc = c * g * g;
    if (i >= size * size * cc) { return; }
    let oc = i % cc;
    let pix = i / cc;
    let ch = oc / (g * g);
    let hg = (oc / g) % g;
    let wg = oc % g;
    out[i] = x[((hg * size + pix / size) * s + wg * size + pix % size) * c + ch];
}
"#;

/// A modulated deformable convolution's taps ([`ChainRecorder::deform_im2col_rows`]): a thread an output value.
/// `p[0]`: `h`, `w`, `c`, `k`; `p[1]`: the first pixel, the pixels.
pub(in crate::chain) const DEFORM_IM2COL_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<f32>;
@group(0) @binding(2) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let h = p[0].x;
    let w = p[0].y;
    let c = p[0].z;
    let k = p[0].w;
    let kk = k * k;
    if (i >= p[1].y * kk * c) { return; }
    let ch = i % c;
    let t = (i / c) % kk;
    let pix = p[1].x + i / (c * kk);
    let pad = f32(k / 2u);
    let y = f32(pix / w) - pad + f32(t / k) + offsets[pix * 2u * kk + 2u * t];
    let xx = f32(pix % w) - pad + f32(t % k) + offsets[pix * 2u * kk + 2u * t + 1u];
    var v = 0.0;
    if (y > -1.0 && y < f32(h) && xx > -1.0 && xx < f32(w)) {
        let y0 = floor(y);
        let x0 = floor(xx);
        let ly = y - y0;
        let lx = xx - x0;
        let iy = i32(y0);
        let ix = i32(x0);
        let hi = i32(h);
        let wi = i32(w);
        if (iy >= 0 && ix >= 0) { v += (1.0 - ly) * (1.0 - lx) * x[(u32(iy) * w + u32(ix)) * c + ch]; }
        if (iy >= 0 && ix + 1 < wi) { v += (1.0 - ly) * lx * x[(u32(iy) * w + u32(ix + 1)) * c + ch]; }
        if (iy + 1 < hi && ix >= 0) { v += ly * (1.0 - lx) * x[(u32(iy + 1) * w + u32(ix)) * c + ch]; }
        if (iy + 1 < hi && ix + 1 < wi) { v += ly * lx * x[(u32(iy + 1) * w + u32(ix + 1)) * c + ch]; }
        v *= mods[pix * kk + t];
    }
    out[i] = v;
}
"#;

/// A 1-D convolution ([`ChainRecorder::conv1d_padded_rows`]): [`CONV_F32_TILED`]'s tiles over steps (64 steps by 64
/// outputs a workgroup), tap `t` of step `s` from step `s + t dilation - pad`. `p[0]`: `cout`, `cin`, the steps, the
/// taps; `p[1]`: the dilation, the dispatch's first tile of steps, `pad`.
pub(in crate::chain) const CONV1D_F32_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> xs: array<vec4<f32>, 272>;
var<workgroup> ws: array<vec4<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let cin = p[0].y;
    let m = p[0].z;
    let taps = p[0].w;
    let dil = i32(p[1].x);
    let cp = (cin + 31u) / 32u * 32u;
    let kt = taps * cp;
    let pad = i32(p[1].z);
    let o0 = wg.x * 64u;
    let r0 = (p[1].y + wg.y) * 64u;
    let tr = t / 16u;
    let to = t % 16u;
    let kk = t % 16u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var kb = 0u; kb < kt; kb += 16u) {
        let gk = kb + kk;
        let tap = i32(gk / cp);
        let c = gk % cp;
        let live = gk < kt && c < cin;
        for (var q = 0u; q < 4u; q++) {
            let rr = tr + 16u * q;
            let s = i32(r0 + rr) + tap * dil - pad;
            var v = 0.0;
            if (live && r0 + rr < m && s >= 0 && s < i32(m)) { v = x[u32(s) * cin + c]; }
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

/// A transposed 1-D convolution ([`ChainRecorder::conv_transpose1d_rows`]): a thread an output value, over the taps
/// that reach it (`(o + pad - j)` a multiple of the stride: `k / stride` of them) and every input channel (vec4s).
/// `p[0]`: `cout`, `cin`, the input's steps, `k`; `p[1]`: the stride, `pad`, the output's steps.
pub(in crate::chain) const CONV_TRANSPOSE1D_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let cout = p[0].x;
    let cin4 = p[0].y / 4u;
    let len = p[0].z;
    let k = p[0].w;
    let stride = p[1].x;
    let pad = p[1].y;
    if (i >= p[1].z * cout) { return; }
    let co = i % cout;
    let o = i / cout;
    var acc = 0.0;
    // the first tap that reaches `o`: `j = (o + pad) mod stride`, then every stride
    for (var j = (o + pad) % stride; j < k; j += stride) {
        if (o + pad < j) { break; }
        let src = (o + pad - j) / stride;
        if (src >= len) { continue; }
        let wb = (co * k + j) * cin4;
        let xb = src * cin4;
        var s = vec4<f32>(0.0);
        for (var c = 0u; c < cin4; c++) { s += w4[wb + c] * x4[xb + c]; }
        acc += s.x + s.y + s.z + s.w;
    }
    y[i] = acc + bias[co];
}
"#;

/// Snake in place ([`ChainRecorder::snake_rows`]): a thread a value. `p[0]`: the rows, `c`.
pub(in crate::chain) const SNAKE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> alpha: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    let a = alpha[i % p[0].y];
    let v = x[i];
    let s = sin(a * v);
    x[i] = v + s * s / (a + 1e-9);
}
"#;

/// Rows gathered ([`ChainRecorder::gather_rows`]): a thread a value, zeros for a missing row. `p[0]`: the rows, `c`,
/// the source's rows, the index's first.
pub(in crate::chain) const GATHER_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].y;
    if (i >= p[0].x * c) { return; }
    let src = index[p[0].w + i / c];
    var v = 0.0;
    if (src < p[0].z) { v = x[src * c + i % c]; }
    out[i] = v;
}
"#;

/// Each channel repeated, added ([`ChainRecorder::repeat_cols_add_rows`]): a thread an output value. `p[0]`: the rows,
/// `c`, the repeats.
pub(in crate::chain) const REPEAT_COLS_ADD_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let width = p[0].y * p[0].z;
    if (i >= p[0].x * width) { return; }
    let r = i / width;
    out[i] += x[r * p[0].y + (i % width) / p[0].z];
}
"#;

/// A depthwise causal 1-D convolution ([`ChainRecorder::depthwise_causal_conv1d_rows`]): a thread an output value.
/// `p[0]`: `c`, `k`, the steps.
pub(in crate::chain) const DEPTHWISE_CAUSAL_CONV1D_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].x;
    let k = p[0].y;
    if (i >= p[0].z * c) { return; }
    let ch = i % c;
    let s = i32(i / c);
    var acc = bias[ch];
    for (var j = 0u; j < k; j++) {
        let src = s + i32(j) - i32(k) + 1;
        if (src >= 0) { acc += w[ch * k + j] * x[u32(src) * c + ch]; }
    }
    y[i] = acc;
}
"#;

/// SnakeBeta in place ([`ChainRecorder::snake_beta_rows`]): a thread a value. `p[0]`: the rows, `c`.
pub(in crate::chain) const SNAKE_BETA_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> freq: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    let ch = i % p[0].y;
    let v = x[i];
    let s = sin(freq[ch] * v);
    x[i] = v + scale[ch] * s * s;
}
"#;

/// [`ChainRecorder::snake_beta_alias_rows`]'s first half: a thread a value of the `2 len` upsampled steps, its six
/// (of `ku` twelve) taps of the steps before it, then SnakeBeta. `p[0]`: len, c, ku.
pub(in crate::chain) const SNAKE_BETA_UP_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> filters: array<f32>;
@group(0) @binding(2) var<storage, read> freq: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(6) var<storage, read_write> mid: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let len = p[0].x;
    let c = p[0].y;
    let ku = p[0].z;
    if (i >= 2u * len * c) { return; }
    let u = i / c;
    let ch = i % c;
    // the transposed convolution's output `n` is the sum over the padded steps `s` of step `s` by tap `n - 2 s`
    let pad = ku / 2u - 1u;
    let n = u + 2u * pad + (ku - 2u) / 2u;
    var acc = 0.0;
    for (var j = n % 2u; j < ku; j += 2u) {
        let s = clamp(i32((n - j) / 2u) - i32(pad), 0, i32(len) - 1);
        acc += x[u32(s) * c + ch] * filters[j];
    }
    let v = 2.0 * acc;
    let w = sin(freq[ch] * v);
    mid[i] = v + scale[ch] * w * w;
}
"#;

/// [`ChainRecorder::snake_beta_alias_rows`]'s second half: a thread a value of the `len` steps kept, its `kd` taps
/// of the upsampled ones. `p[0]`: len, c, where the taps start among the filters, kd.
pub(in crate::chain) const SNAKE_BETA_DOWN_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> mid: array<f32>;
@group(0) @binding(1) var<storage, read> filters: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let len = p[0].x;
    let c = p[0].y;
    let kd = p[0].w;
    if (i >= len * c) { return; }
    let n = i / c;
    let ch = i % c;
    let left = i32(kd / 2u) - i32(1u - kd % 2u);
    var acc = 0.0;
    for (var j = 0u; j < kd; j++) {
        let s = clamp(i32(2u * n + j) - left, 0, i32(2u * len) - 1);
        acc += mid[u32(s) * c + ch] * filters[p[0].z + j];
    }
    y[i] = acc;
}
"#;

/// A clamp in place ([`ChainRecorder::clamp_in_place`]): a thread a value. `p[0]`: the values, the bounds' bits.
pub(in crate::chain) const CLAMP_IN_PLACE: &str = r#"
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    x[i] = clamp(x[i], bitcast<f32>(p[0].y), bitcast<f32>(p[0].z));
}
"#;

/// `tanh` in place ([`ChainRecorder::tanh_in_place`]): a thread a value. `p[0]`: the values.
pub(in crate::chain) const TANH_IN_PLACE: &str = r#"
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    x[i] = tanh(clamp(x[i], -15.0, 15.0));
}
"#;

/// Rows times their gates' sigmoids ([`ChainRecorder::mul_sigmoid_rows`]), in place: a thread a value. `p[0]`: the
/// rows, `c`.
pub(in crate::chain) const MUL_SIGMOID_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x * p[0].y) { return; }
    x[i] = x[i] / (1.0 + exp(-gate[i / p[0].y]));
}
"#;

/// A column's mean ([`ChainRecorder::mean_rows`]): a thread a channel. `p[0]`: the rows, `c`.
pub(in crate::chain) const MEAN_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = id.x;
    let rows = p[0].x;
    let c = p[0].y;
    if (ch >= c) { return; }
    var s = 0.0;
    for (var r = 0u; r < rows; r++) { s += x[r * c + ch]; }
    out[ch] = s / f32(rows);
}
"#;

/// One row into many ([`ChainRecorder::broadcast_rows`]): a thread a value. `p[0]`: the rows, `c`, the stride, `at`.
pub(in crate::chain) const BROADCAST_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let c = p[0].y;
    if (i >= p[0].x * c) { return; }
    dst[(i / c) * p[0].z + p[0].w + i % c] = src[i % c];
}
"#;

/// `out = x` where positive, else `slope x`: `p[0]` the values, the slope's bits.
pub(in crate::chain) const LEAKY_RELU: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    let v = x[i];
    out[i] = select(v * bitcast<f32>(p[0].y), v, v > 0.0);
}
"#;
