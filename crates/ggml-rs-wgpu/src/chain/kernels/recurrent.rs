//! The recurrences: a gated delta net's conv and its recurrence (a step's, and a prompt's in three passes), and the n-gram layer's gate and conv.

/// An n-gram layer's gate (`Backend::ple_gate`), a workgroup a (row, stream): the stream's key and query RMS-normed and
/// scaled by their norms, their dot over `sqrt(d)`, its signed square root's sigmoid the gate; `gated` the gate times
/// the row's value, `conv_in` that RMS-normed, scaled by the conv's norm and rounded to f16. `p[0]`: d, streams, the
/// bits of eps.
pub(in crate::chain) const PLE_GATE: &str = r#"
@group(0) @binding(0) var<storage, read> key: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> nk: array<f32>;
@group(0) @binding(4) var<storage, read> nq: array<f32>;
@group(0) @binding(5) var<storage, read> nc: array<f32>;
@group(0) @binding(6) var<storage, read_write> gated: array<f32>;
@group(0) @binding(7) var<storage, read_write> conv_in: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> red: array<vec2<f32>, 256>;

fn total(v: vec2<f32>, t: u32) -> vec2<f32> {
    red[t] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (t < st) { red[t] += red[t + st]; }
        workgroupBarrier();
    }
    let out = red[0];
    workgroupBarrier();
    return out;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let d = p[0].x;
    let streams = p[0].y;
    let eps = bitcast<f32>(p[0].z);
    let r = wg.x / streams;
    let s = wg.x % streams;
    let o = r * streams * d + s * d;
    var sq = vec2<f32>(0.0);
    for (var j = t; j < d; j += 256u) {
        let kk = key[o + j];
        let qq = x[o + j];
        sq += vec2<f32>(kk * kk, qq * qq);
    }
    let ss = total(sq, t);
    let ik = 1.0 / sqrt(ss.x / f32(d) + eps);
    let iq = 1.0 / sqrt(ss.y / f32(d) + eps);
    var dot = 0.0;
    for (var j = t; j < d; j += 256u) {
        dot += (key[o + j] * ik * nk[s * d + j]) * (x[o + j] * iq * nq[s * d + j]);
    }
    let g = total(vec2<f32>(dot, 0.0), t).x / sqrt(f32(d));
    var sg = 0.0;
    if (g != 0.0) {
        sg = sign(g) * sqrt(max(abs(g), 1e-6));
    }
    let gate = 1.0 / (1.0 + exp(-sg));
    var gs = 0.0;
    for (var j = t; j < d; j += 256u) {
        let gv = gate * value[r * d + j];
        gated[o + j] = gv;
        gs += gv * gv;
    }
    let inv = 1.0 / sqrt(total(vec2<f32>(gs, 0.0), t).x / f32(d) + eps);
    for (var j = t; j < d; j += 256u) {
        conv_in[o + j] = half(gate * value[r * d + j] * inv * nc[s * d + j]);
    }
}
"#;

/// An n-gram layer's dilated causal conv (`Backend::ple_conv`), a thread a channel through the rows: over the stream
/// of the window's rows then `conv_in`'s, `x[r, c] += gated[r, c] + silu(sum over j of w[c, j] stream[r + j dilation,
/// c])`; the window then the stream's last rows. `p[0]`: width, rows, kernel, dilation.
pub(in crate::chain) const PLE_CONV: &str = r#"
@group(0) @binding(0) var<storage, read> gated: array<f32>;
@group(0) @binding(1) var<storage, read> conv_in: array<f32>;
@group(0) @binding(2) var<storage, read> wc: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@group(0) @binding(7) var<storage, read_write> win: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn stream_at(q: u32, c: u32, width: u32, state: u32) -> f32 {
    if (q < state) {
        return win[q * width + c];
    }
    return conv_in[(q - state) * width + c];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = p[0].x;
    let rows = p[0].y;
    let kernel = p[0].z;
    let dil = p[0].w;
    let c = id.x;
    if (c >= width) {
        return;
    }
    let state = (kernel - 1u) * dil;
    for (var r = 0u; r < rows; r++) {
        var acc = 0.0;
        for (var j = 0u; j < kernel; j++) {
            acc += wc[c * kernel + j] * stream_at(r + j * dil, c, width, state);
        }
        x[r * width + c] = x[r * width + c] + (gated[r * width + c] + acc / (1.0 + exp(-acc)));
    }
    // each place read before it is written: the new row i is the stream's row rows + i, later than i
    for (var i = 0u; i < state; i++) {
        win[i * width + c] = stream_at(rows + i, c, width, state);
    }
}
"#;

/// A prompt's conv ([`SSM_CONV`]'s sums), a thread a (channel, token): its `kernel` inputs those up to it, the run's
/// or (before its first) the state's; no output another's input. The state is left to [`SSM_CONV_STATE`].
pub(in crate::chain) const SSM_CONV_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let ch = p[0].x;
    let kern = p[0].z;
    let c = wg.x * 256u + li;
    let t = wg.y;
    if (c >= ch) { return; }
    var acc = 0.0;
    for (var k = 0u; k + 1u < kern; k++) {
        // input t + k - (kern - 1)
        let j = t + k;
        var xv = 0.0;
        if (j + 1u >= kern) { xv = qkv[(j + 1u - kern) * ch + c]; } else { xv = cs[j * ch + c]; }
        acc += xv * cw[c * kern + k];
    }
    acc += qkv[t * ch + c] * cw[c * kern + kern - 1u];
    out[t * ch + c] = acc / (1.0 + exp(-acc));
}
"#;

/// The state after [`SSM_CONV_ROWS`]: the run's last `kernel - 1` inputs (a run as long at least), a thread a channel.
pub(in crate::chain) const SSM_CONV_STATE: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(6) var<storage, read_write> cs: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = p[0].x;
    let rows = p[0].y;
    let kern = p[0].z;
    let c = id.x;
    if (c >= ch) { return; }
    for (var k = 0u; k + 1u < kern; k++) { cs[k * ch + c] = qkv[(rows + 1u + k - kern) * ch + c]; }
}
"#;

/// A gated delta net's causal depthwise conv, a thread a channel through the run's tokens (as the host's): its
/// `kernel - 1` inputs before them from the state, each output through SiLU, the state left with the last inputs.
/// `p[0]`: the channels, the tokens, the kernel.
pub(in crate::chain) const SSM_CONV: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(6) var<storage, read_write> cs: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = p[0].x;
    let rows = p[0].y;
    let kern = p[0].z;
    let c = id.x;
    if (c >= ch) { return; }
    var win: array<f32, 8>;
    for (var k = 0u; k + 1u < kern; k++) { win[k] = cs[k * ch + c]; }
    for (var t = 0u; t < rows; t++) {
        let xin = qkv[t * ch + c];
        var acc = 0.0;
        for (var k = 0u; k + 1u < kern; k++) { acc += win[k] * cw[c * kern + k]; }
        acc += xin * cw[c * kern + kern - 1u];
        out[t * ch + c] = acc / (1.0 + exp(-acc));
        for (var k = 0u; k + 2u < kern; k++) { win[k] = win[k + 1u]; }
        win[kern - 2u] = xin;
    }
    for (var k = 0u; k + 1u < kern; k++) { cs[k * ch + c] = win[k]; }
}
"#;

/// A gated delta net's recurrence through the run's tokens, a workgroup a value head and a thread a row of its state
/// (as the host's `delta_net_step`), the row held in registers from the run's start to its end: each token's q and k
/// (its key head's) L2-normed, the state decayed by `exp(softplus(alpha + dt_bias) * a)`, the delta rule's update by
/// `sigmoid(beta)`, the state read by q, and the result RMS-normed over the head and gated by `silu(z)` (or
/// `sigmoid(z)`). `DK` (the heads' size, `k_dim` = `v_dim`) is made a constant, and the row `DK / 4` vectors of its
/// own ([`delta_net_one`] puts their code in: an array indexed in a loop is the thread's memory, not its registers).
/// `p[0]`: the value heads, the key heads; `p[1]`: the tokens, the bits of the q scale and of eps, the sigmoid gate. (A
/// row read from memory at each token: 43 us a layer for a decode step of Qwen3.8 27B, 30 us a token of a prompt.)
pub(in crate::chain) const DELTA_NET: &str = r#"
@group(0) @binding(0) var<storage, read> cv: array<f32>;
@group(0) @binding(1) var<storage, read> zz: array<f32>;
@group(0) @binding(2) var<storage, read> ba: array<f32>;
@group(0) @binding(3) var<storage, read> sa: array<f32>;
@group(0) @binding(4) var<storage, read> dt: array<f32>;
@group(0) @binding(5) var<storage, read> nm: array<f32>;
@group(0) @binding(6) var<storage, read_write> st: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;
const DK4: u32 = DK_VALUEu / 4u;

// the token's normed q and k, read four at a time. Scalars, not vec4s: a thread writes its own, and WGSL lets a write
// to one component of a vector in memory write the whole vector (Metal does), so four threads writing one would race
var<workgroup> qn: array<f32, DK>;
var<workgroup> kn: array<f32, DK>;
var<workgroup> red: array<f32, 2 * DK>;

@compute @workgroup_size(DK)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let nv = p[0].x;
    let nk = p[0].y;
    let rows = p[1].x;
    let scale_q = bitcast<f32>(p[1].y);
    let eps = bitcast<f32>(p[1].z);
    let sig = p[1].w;
    let h = wg.x;
    let hk = h % nk;
    let ch = 2u * nk * DK + nv * DK;
    let base = (h * DK + i) * DK4;
ROW_LOAD
    for (var t = 0u; t < rows; t++) {
        let r0 = t * ch;
        let q = cv[r0 + hk * DK + i];
        let k = cv[r0 + nk * DK + hk * DK + i];
        red[i] = q * q;
        red[DK + i] = k * k;
        workgroupBarrier();
        for (var s = DK / 2u; s > 0u; s /= 2u) {
            if (i < s) {
                red[i] += red[i + s];
                red[DK + i] += red[DK + i + s];
            }
            workgroupBarrier();
        }
        let inv_q = 1.0 / sqrt(red[0] + eps);
        let inv_k = 1.0 / sqrt(red[DK] + eps);
        qn[i] = q * inv_q * scale_q;
        kn[i] = k * inv_k;
        workgroupBarrier();
        let bt = 1.0 / (1.0 + exp(-ba[t * 2u * nv + h]));
        let ab = ba[t * 2u * nv + nv + h] + dt[h];
        var sp = ab;
        if (ab <= 20.0) {
            let e = exp(ab);
            // ln(1 + e) as ln_1p gives it, where 1 + e would round e away
            sp = select(log(1.0 + e), e * (1.0 - 0.5 * e), e < 1e-4);
        }
        let g = exp(sp * sa[h]);
        // the decayed row read by k, its sums in the host's order
        var kv = 0.0;
ONE_KV
        let delta = (cv[r0 + 2u * nk * DK + h * DK + i] - kv) * bt;
        var core = 0.0;
ONE_UPDATE
        red[i] = core * core;
        workgroupBarrier();
        for (var s = DK / 2u; s > 0u; s /= 2u) {
            if (i < s) { red[i] += red[i + s]; }
            workgroupBarrier();
        }
        let inv = 1.0 / sqrt(red[0] / f32(DK) + eps);
        let zv = zz[t * nv * DK + h * DK + i];
        var gate = zv / (1.0 + exp(-zv));
        if (sig != 0u) { gate = 1.0 / (1.0 + exp(-zv)); }
        out[t * nv * DK + h * DK + i] = core * inv * nm[i] * gate;
        workgroupBarrier();
    }
ROW_STORE
}
"#;

/// [`DELTA_NET`] for heads `dk` wide: the row's `dk / 4` vectors and their code, each sum in the host's order.
pub(in crate::chain) fn delta_net_one(dk: usize) -> String {
    let n = dk / 4;
    let load: String = (0..n).map(|c| format!("    var sr{c} = st[base + {c}u];\n")).collect();
    let kv: String = (0..n)
        .map(|c| format!("        {{\n            let s = sr{c} * g;\n            let kk = vec4<f32>(kn[{a}u], kn[{b}u], kn[{c2}u], kn[{d}u]);\n            kv += s.x * kk.x;\n            kv += s.y * kk.y;\n            kv += s.z * kk.z;\n            kv += s.w * kk.w;\n        }}\n", a = 4 * c, b = 4 * c + 1, c2 = 4 * c + 2, d = 4 * c + 3))
        .collect();
    let update: String = (0..n)
        .map(|c| format!("        {{\n            let s = sr{c} * g + delta * vec4<f32>(kn[{a}u], kn[{b}u], kn[{c2}u], kn[{d}u]);\n            sr{c} = s;\n            let qq = vec4<f32>(qn[{a}u], qn[{b}u], qn[{c2}u], qn[{d}u]);\n            core += s.x * qq.x;\n            core += s.y * qq.y;\n            core += s.z * qq.z;\n            core += s.w * qq.w;\n        }}\n", a = 4 * c, b = 4 * c + 1, c2 = 4 * c + 2, d = 4 * c + 3))
        .collect();
    let store: String = (0..n).map(|c| format!("    st[base + {c}u] = sr{c};\n")).collect();
    DELTA_NET
        .replace("DK_VALUE", &dk.to_string())
        .replace("ROW_LOAD\n", &load)
        .replace("ONE_KV\n", &kv)
        .replace("ONE_UPDATE\n", &update)
        .replace("ROW_STORE\n", &store)
}

/// A prompt's delta net in three passes, the first: each token's q and k (its key heads') L2-normed as
/// [`DELTA_NET`] norms them, a workgroup a (key head, token), into the scratch `qk` (`[token, key head]`: q then k),
/// and the token's `sigmoid(beta)` and decay of each value head the key head's (after them, `[token, value head]`).
pub(in crate::chain) const DELTA_NET_PREP: &str = r#"
@group(0) @binding(0) var<storage, read> cv: array<f32>;
@group(0) @binding(2) var<storage, read> ba: array<f32>;
@group(0) @binding(3) var<storage, read> sa: array<f32>;
@group(0) @binding(4) var<storage, read> dt: array<f32>;
@group(0) @binding(6) var<storage, read_write> qk: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;

var<workgroup> red: array<f32, 2 * DK>;

@compute @workgroup_size(DK)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let nv = p[0].x;
    let nk = p[0].y;
    let rows = p[1].x;
    let scale_q = bitcast<f32>(p[1].y);
    let eps = bitcast<f32>(p[1].z);
    let hk = wg.x;
    let t = wg.y;
    let r0 = t * (2u * nk * DK + nv * DK);
    let q = cv[r0 + hk * DK + i];
    let k = cv[r0 + nk * DK + hk * DK + i];
    red[i] = q * q;
    red[DK + i] = k * k;
    workgroupBarrier();
    for (var s = DK / 2u; s > 0u; s /= 2u) {
        if (i < s) {
            red[i] += red[i + s];
            red[DK + i] += red[DK + i + s];
        }
        workgroupBarrier();
    }
    let inv_q = 1.0 / sqrt(red[0] + eps);
    let inv_k = 1.0 / sqrt(red[DK] + eps);
    let o = (t * nk + hk) * 2u * DK;
    qk[o + i] = q * inv_q * scale_q;
    qk[o + DK + i] = k * inv_k;
    // the value heads that read this key head (h % nk)
    if (i < nv / nk) {
        let h = hk + i * nk;
        let bt = 1.0 / (1.0 + exp(-ba[t * 2u * nv + h]));
        let ab = ba[t * 2u * nv + nv + h] + dt[h];
        var sp = ab;
        if (ab <= 20.0) {
            let e = exp(ab);
            sp = select(log(1.0 + e), e * (1.0 - 0.5 * e), e < 1e-4);
        }
        let bg = rows * nk * 2u * DK + (t * nv + h) * 2u;
        qk[bg] = bt;
        qk[bg + 1u] = exp(sp * sa[h]);
    }
}
"#;

/// The second: the recurrence through the tokens, a workgroup `R` rows of a value head's state and a thread a row
/// held in registers from the first token to the last (as `DK / 4` vectors of its own, the code unrolled: an array
/// indexed in a loop is the thread's memory, not its registers); each token's q and k (the first pass's) in the
/// workgroup's memory, the next token's loaded as this one's are used (one barrier a token); each token's state read
/// by q into `out`, as yet unnormed. ([`delta_net_scan`] puts the rows' code in.)
pub(in crate::chain) const DELTA_NET_SCAN: &str = r#"
@group(0) @binding(0) var<storage, read> qk4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cv: array<f32>;
@group(0) @binding(2) var<storage, read> qk: array<f32>;
@group(0) @binding(6) var<storage, read_write> st: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;
const DK4: u32 = DK_VALUEu / 4u;
const R: u32 = R_VALUEu;

// two tokens' q then k
var<workgroup> qks: array<vec4<f32>, 4 * DK4>;

@compute @workgroup_size(R)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nv = p[0].x;
    let nk = p[0].y;
    let rows = p[1].x;
    let h = wg.x / (DK / R);
    let i = (wg.x % (DK / R)) * R + li;
    let hk = h % nk;
    let ch = 2u * nk * DK + nv * DK;
    let base = (h * DK + i) * DK4;
    let bg0 = rows * nk * 2u * DK;
ROW_LOAD
    for (var c = li; c < 2u * DK4; c += R) { qks[c] = qk4[hk * 2u * DK4 + c]; }
    workgroupBarrier();
    for (var t = 0u; t < rows; t++) {
        let cur = (t % 2u) * 2u * DK4;
        let ks = cur + DK4;
        // the next token's q and k loaded now, stored once this one's are used
        let more = t + 1u < rows;
        let o4 = ((t + 1u) * nk + hk) * 2u * DK4;
        var pre0 = vec4<f32>(0.0);
        var pre1 = vec4<f32>(0.0);
        if (more && li < 2u * DK4) { pre0 = qk4[o4 + li]; }
        if (more && li + R < 2u * DK4) { pre1 = qk4[o4 + li + R]; }
        let bt = qk[bg0 + (t * nv + h) * 2u];
        let g = qk[bg0 + (t * nv + h) * 2u + 1u];
        var ka = vec4<f32>(0.0);
        var kb = vec4<f32>(0.0);
ROW_KV
        let kd = ka + kb;
        let kv = (kd.x + kd.y + kd.z + kd.w) * g;
        let delta = (cv[t * ch + 2u * nk * DK + h * DK + i] - kv) * bt;
        var qa = vec4<f32>(0.0);
        var qb = vec4<f32>(0.0);
ROW_UPDATE
        let qd = qa + qb;
        out[t * nv * DK + h * DK + i] = qd.x + qd.y + qd.z + qd.w;
        let nxt = 2u * DK4 - cur;
        if (more && li < 2u * DK4) { qks[nxt + li] = pre0; }
        if (more && li + R < 2u * DK4) { qks[nxt + li + R] = pre1; }
        workgroupBarrier();
    }
ROW_STORE
}
"#;

/// [`DELTA_NET_SCAN`] for heads `dk` wide, `r` rows a workgroup: the row's `dk / 4` vectors and their code.
pub(in crate::chain) fn delta_net_scan(dk: usize, r: usize) -> String {
    let n = dk / 4;
    let load: String = (0..n).map(|c| format!("    var sr{c} = st[base + {c}u];\n")).collect();
    let kv: String = (0..n).map(|c| format!("        {} += sr{c} * qks[ks + {c}u];\n", if c % 2 == 0 { "ka" } else { "kb" })).collect();
    let update: String = (0..n).map(|c| format!("        sr{c} = sr{c} * g + delta * qks[ks + {c}u];\n        {} += sr{c} * qks[cur + {c}u];\n", if c % 2 == 0 { "qa" } else { "qb" })).collect();
    let store: String = (0..n).map(|c| format!("    st[base + {c}u] = sr{c};\n")).collect();
    DELTA_NET_SCAN
        .replace("DK_VALUE", &dk.to_string())
        .replace("R_VALUE", &r.to_string())
        .replace("ROW_LOAD\n", &load)
        .replace("ROW_KV\n", &kv)
        .replace("ROW_UPDATE\n", &update)
        .replace("ROW_STORE\n", &store)
}

/// The third: each token's state read by q RMS-normed over its head and gated as [`DELTA_NET`]'s, a workgroup a
/// (value head, token), in place.
pub(in crate::chain) const DELTA_NET_NORM: &str = r#"
@group(0) @binding(1) var<storage, read> zz: array<f32>;
@group(0) @binding(5) var<storage, read> nm: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const DK: u32 = DK_VALUEu;

var<workgroup> red: array<f32, DK>;

@compute @workgroup_size(DK)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let nv = p[0].x;
    let eps = bitcast<f32>(p[1].z);
    let sig = p[1].w;
    let o = (wg.y * nv + wg.x) * DK + i;
    let core = out[o];
    red[i] = core * core;
    workgroupBarrier();
    for (var s = DK / 2u; s > 0u; s /= 2u) {
        if (i < s) { red[i] += red[i + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(DK) + eps);
    let zv = zz[o];
    var gate = zv / (1.0 + exp(-zv));
    if (sig != 0u) { gate = 1.0 / (1.0 + exp(-zv)); }
    out[o] = core * inv * nm[i] * gate;
}
"#;
