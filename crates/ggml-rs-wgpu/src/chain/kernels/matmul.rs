//! The matmuls of weights this module holds itself: f32 and f16 matrices (a step's row, a check's few rows, a prompt's tiles), the sums of a split one, W4A8's decode and a LoRA's merge.

/// A split tensor-core matmul's sums put together: `y[i]` the sum of `x`'s `p[0].y` parts of `p[0].x` each.
pub(in crate::chain) const COOP_SUM: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p[0].x;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= n) { return; }
    var acc = x[i];
    for (var z = 1u; z < p[0].y; z++) { acc += x[z * n + i]; }
    y[i] = acc;
}
"#;

/// `w` (f16 pairs, `p[0].x` words) plus `d` (two f32 a word), rounded to f16.
pub(in crate::chain) const ADD_F16: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(6) var<storage, read_write> w: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    if (i >= p[0].x) { return; }
    w[i] = pack2x16float(unpack2x16float(w[i]) + vec2<f32>(d[2u * i], d[2u * i + 1u]));
}
"#;

/// `y[r * n + o] = dot(w[o], x[r])` for f32 weights `w` (`[n, k]`, `p[0]`: `n`, `k`), a workgroup an (output, row).
pub(in crate::chain) const MATMUL_F32: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let o = wg.x;
    let r = wg.y;
    var s = 0.0;
    for (var i = li; i < k; i += 256u) { s += bitcast<f32>(w[o * k + i]) * x[r * k + i]; }
    part[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { y[r * n + o] = part[0]; }
}
"#;

/// [`MATVEC_F32_4`] with f16 weights two to a word (`DeviceChain::vec_f16`'s), `k` a multiple of 4: each output's
/// products summed as it sums them. `p[0]`: n, k.
pub(in crate::chain) const MATVEC_F16: &str = r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k4 = p[0].y / 4u;
    let o = wg.x;
    var s = vec4<f32>(0.0);
    for (var i = li; i < k4; i += 256u) {
        let pr = w2[o * k4 + i];
        let a = unpack2x16float(pr.x);
        let b = unpack2x16float(pr.y);
        s += vec4<f32>(a.x, a.y, b.x, b.y) * x4[i];
    }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { y[o] = part[0]; }
}
"#;

/// Lanes an output row of an f16 matrix `k` wide is shared out to in [`matvec_f16_lanes`]: a power of two from 8 to
/// 256, some ten loads (40 weights) a lane or more. A wide matrix has few rows (a hyper-connection's 324 of 10,240),
/// and it is the lanes that give the GPU its threads: 32 lanes a row left that one 10,000 threads and took three
/// times as long as a workgroup a row.
pub(in crate::chain) fn f16_lanes(k: usize) -> usize {
    // (OAIY_F16_LANES: a measurement's)
    static ASKED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(l) = *ASKED.get_or_init(|| std::env::var("OAIY_F16_LANES").ok().and_then(|v| v.parse().ok()).filter(|l: &usize| l.is_power_of_two() && (8..=256).contains(l))) {
        return l;
    }
    let lanes = (k / 4 / 10).max(1);
    // (the power of two at or under it)
    (1usize << (usize::BITS - 1 - lanes.leading_zeros())).clamp(8, 256)
}

/// An f16 matrix (two weights a word, `k` a multiple of 4) against `rows` rows of x (1 to 8: a decode step's, a check
/// of drafts'): a workgroup `256 / lanes` output rows, `lanes` threads a row ([`f16_lanes`]), each taking four weights
/// in every `4 lanes` of the row with four of x at a load, so a row's threads read side by side; each weight is read
/// once for every row of x, a row's products summed the same way whatever the rows (a lane's in four running sums,
/// then the lanes in order, a row of x's by a lane of its own), so a check's rows are a step's bit for bit. One
/// barrier. `p[0]`: n, k.
///
/// It replaces, for such widths, [`MATVEC_F16`] (a workgroup an output row and eight barriers: Flash-Next's routers'
/// 513 rows of 2,560 gave a thread ten weights) and [`MATVEC_F16_NARROW`] (eight threads an output, a word and two
/// scalar reads of x at a time): a decode step's f16 matrices (its hyper-connections' and, from a GGUF, its shared
/// experts': 1.9 GB) were 3.6 ms of its 11.2.
pub(in crate::chain) fn matvec_f16_lanes(rows: usize, lanes: usize) -> String {
    assert!((1..=8).contains(&rows) && lanes.is_power_of_two() && (8..=256).contains(&lanes), "1 to 8 rows, 8 to 256 lanes");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("            s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let o = (wg.x + wg.y * 65535u) * {per}u + li / {lanes}u;
    let lane = li % {lanes}u;
{regs}
    if (o < n) {{
        for (var i = lane; i < k4; i += {lanes}u) {{
            let pr = w2[o * k4 + i];
            let a = unpack2x16float(pr.x);
            let b = unpack2x16float(pr.y);
            let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    // a row of x's sum by a lane of its own: lane r adds row r's parts, the lanes in order (one lane adding every
    // row's in turn was what a check's kernel waited for)
    if (lane < {rows}u && o < n) {{
        let at = lane * 256u + li - lane;
        var t = 0.0;
        for (var j = 0u; j < {lanes}u; j++) {{ t += part[at + j]; }}
        y[lane * n + o] = t;
    }}
}}
"#,
        len = rows * 256,
        per = 256 / lanes,
    )
}

/// [`matvec_f16_lanes`] of a hyper-connection's down matrix with its gates ([`super::HC_GATES`]) where it stores a
/// sum: output `o` of a row its `silu(sum / streams)` below the rank, else 0 with `post` its `2 sigmoid(sum /
/// streams)`. The sums and the gates as the two kernels make them, bit for bit. `p[0]`: n (the rank and the writes),
/// k, the rank, the streams.
pub(in crate::chain) fn hc_down_gates_lanes(rows: usize, lanes: usize) -> String {
    assert!((1..=8).contains(&rows) && lanes.is_power_of_two() && (8..=256).contains(&lanes), "1 to 8 rows, 8 to 256 lanes");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("            s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> gt: array<f32>;
@group(0) @binding(7) var<storage, read_write> post: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let rank = p[0].z;
    let streams = f32(p[0].w);
    let o = (wg.x + wg.y * 65535u) * {per}u + li / {lanes}u;
    let lane = li % {lanes}u;
{regs}
    if (o < n) {{
        for (var i = lane; i < k4; i += {lanes}u) {{
            let pr = w2[o * k4 + i];
            let a = unpack2x16float(pr.x);
            let b = unpack2x16float(pr.y);
            let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    // (a row of x's sum and its gate by a lane of its own, as [`matvec_f16_lanes`]'s)
    if (lane < {rows}u && o < n) {{
        let at = lane * 256u + li - lane;
        var t = 0.0;
        for (var j = 0u; j < {lanes}u; j++) {{ t += part[at + j]; }}
        let v = t / streams;
        if (o < rank) {{
            gt[lane * n + o] = v / (1.0 + exp(-v));
        }} else {{
            post[lane * (n - rank) + o - rank] = 2.0 / (1.0 + exp(-v));
            gt[lane * n + o] = 0.0;
        }}
    }}
}}
"#,
        len = rows * 256,
        per = 256 / lanes,
    )
}

/// [`matvec_f16_lanes`] of a hyper-connection's up matrix with its mix ([`super::HC_MIX`]): a workgroup's `256 /
/// lanes` output rows are `streams` logits each of some outputs `j` (row `s d + j` stream `s`'s), and where the
/// matmul would store them each stream's slot makes its `normed / (1 + exp(-logit)) / streams` and the first
/// stream's adds them in their order into `out[j]` (a row of x's by a lane of its own in both). The logits' sums and the mix as the two kernels make them, bit for bit. `p[0]`: n (the
/// streams' rows), k, d, the streams (which divide the workgroup's rows).
pub(in crate::chain) fn hc_up_mix_lanes(rows: usize, lanes: usize, streams: usize) -> String {
    assert!((1..=8).contains(&rows) && lanes.is_power_of_two() && (8..=256).contains(&lanes) && streams > 0 && (256 / lanes) % streams == 0, "1 to 8 rows, 8 to 256 lanes, streams that divide a workgroup's rows");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("            s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normed: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let k4 = p[0].y / 4u;
    let d = p[0].z;
    // the workgroup's slots: `streams` in turn for each of its outputs
    let slot = li / {lanes}u;
    let lane = li % {lanes}u;
    let j = (wg.x + wg.y * 65535u) * {each_j}u + slot / {streams}u;
    let s_of = slot % {streams}u;
    let o = s_of * d + j;
{regs}
    if (j < d) {{
        for (var i = lane; i < k4; i += {lanes}u) {{
            let pr = w2[o * k4 + i];
            let a = unpack2x16float(pr.x);
            let b = unpack2x16float(pr.y);
            let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    // row `lane`'s logit of this slot's stream (its lanes' parts in order) and that stream's term of the mix, left
    // where the slot's parts of the row began (no other thread reads them); then the row's lane of each output's first
    // stream adds the streams' terms in their order
    let own = lane * 256u + li - lane;
    if (lane < {rows}u && j < d) {{
        var t = 0.0;
        for (var q = 0u; q < {lanes}u; q++) {{ t += part[own + q]; }}
        part[own] = normed[(lane * {streams}u + s_of) * d + j] / (1.0 + exp(-t)) / f32({streams}u);
    }}
    workgroupBarrier();
    if (lane < {rows}u && s_of == 0u && j < d) {{
        var acc = 0.0;
        for (var s = 0u; s < {streams}u; s++) {{ acc += part[own + s * {lanes}u]; }}
        y[lane * d + j] = acc;
    }}
}}
"#,
        len = rows * 256,
        each_j = 256 / lanes / streams,
    )
}

/// [`matvec_f16_lanes`] of a SwiGLU's gate rows and up rows as one matrix (`[2 ff, k]`, the gate's first) with the
/// SwiGLU ([`super::SILU_MUL_SPLIT`]) where it would store the sums: a workgroup's output rows are pairs, gate `j` then
/// up `j` of some `j`, a row of x's sums by a lane of its own, and the gate's lane writes `silu(gate) * up` to `y[r,
/// j]`. The sums and the SwiGLU as the two kernels make them, bit for bit. `p[0]`: ff, k.
pub(in crate::chain) fn matvec_f16_swiglu_lanes(rows: usize, lanes: usize) -> String {
    assert!((1..=8).contains(&rows) && lanes.is_power_of_two() && (8..=128).contains(&lanes), "1 to 8 rows, 8 to 128 lanes");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("            s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let ff = p[0].x;
    let k4 = p[0].y / 4u;
    // the workgroup's slots: a pair's gate row, then its up row
    let slot = li / {lanes}u;
    let lane = li % {lanes}u;
    let j = (wg.x + wg.y * 65535u) * {pairs}u + slot / 2u;
    let up = slot % 2u;
    let o = up * ff + j;
{regs}
    if (j < ff) {{
        for (var i = lane; i < k4; i += {lanes}u) {{
            let pr = w2[o * k4 + i];
            let a = unpack2x16float(pr.x);
            let b = unpack2x16float(pr.y);
            let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    // row `lane`'s sum of this slot's (its lanes' parts in order), left where they began; then the gate's lane
    let own = lane * 256u + li - lane;
    if (lane < {rows}u && j < ff) {{
        var t = 0.0;
        for (var q = 0u; q < {lanes}u; q++) {{ t += part[own + q]; }}
        part[own] = t;
    }}
    workgroupBarrier();
    if (lane < {rows}u && up == 0u && j < ff) {{
        let g = part[own];
        y[lane * ff + j] = (g / (1.0 + exp(-g))) * part[own + {lanes}u];
    }}
}}
"#,
        len = rows * 256,
        pairs = 256 / lanes / 2,
    )
}

/// [`matvec_f16_lanes`] of a MoE layer's shared expert's down matrix (`[h, ff]`) with the experts' weighted sum into
/// the streams ([`crate::exl3::WSUM_APPLY`]) where it would store a sum: output `c` of row `r` is the shared expert's
/// term, and its lane adds the routed experts' `kk` (`d`, pair `r kk + j`'s outputs, by `wts[r (kk + 1) + j]`), then
/// the shared one's by `wts[r (kk + 1) + kk]`, and adds the sum to row `r`'s streams by `post`. The sums as the two
/// kernels make them, bit for bit. `p[0]`: h, ff, kk, the streams.
pub(in crate::chain) fn shared_down_wsum_lanes(rows: usize, lanes: usize) -> String {
    assert!((1..=8).contains(&rows) && lanes.is_power_of_two() && (8..=256).contains(&lanes), "1 to 8 rows, 8 to 256 lanes");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("            s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> d: array<f32>;
@group(0) @binding(3) var<storage, read> wts: array<f32>;
@group(0) @binding(4) var<storage, read> post: array<f32>;
@group(0) @binding(6) var<storage, read_write> xs: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let kk = p[0].z;
    let streams = p[0].w;
    let o = (wg.x + wg.y * 65535u) * {per}u + li / {lanes}u;
    let lane = li % {lanes}u;
{regs}
    if (o < n) {{
        for (var i = lane; i < k4; i += {lanes}u) {{
            let pr = w2[o * k4 + i];
            let a = unpack2x16float(pr.x);
            let b = unpack2x16float(pr.y);
            let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    // row `lane`'s: the shared expert's output (its lanes' parts in order), then the experts' sum and its streams
    if (lane < {rows}u && o < n) {{
        let at = lane * 256u + li - lane;
        var t = 0.0;
        for (var q = 0u; q < {lanes}u; q++) {{ t += part[at + q]; }}
        var acc = 0.0;
        for (var j = 0u; j < kk; j++) {{ acc += wts[lane * (kk + 1u) + j] * d[(lane * kk + j) * n + o]; }}
        acc += wts[lane * (kk + 1u) + kk] * t;
        for (var s = 0u; s < streams; s++) {{
            let to = (lane * streams + s) * n + o;
            xs[to] = xs[to] + post[lane * streams + s] * acc;
        }}
    }}
}}
"#,
        len = rows * 256,
        per = 256 / lanes,
    )
}

/// [`MATVEC_F16`] (`narrow`: [`MATVEC_F16_NARROW`]) for `rows` rows (2 to 8, a check of drafts): each weight read once
/// for all of them, each row's products summed in the one-row kernel's order (so a row's outputs are its bit for bit).
/// `p[0]`: n, k.
pub(in crate::chain) fn matvec_f16_rows(rows: usize, narrow: bool) -> String {
    assert!((2..=8).contains(&rows), "a few rows (2 to 8)");
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    if narrow {
        let regs = each(&|r| format!("    var s{r} = 0.0;"));
        let sums = each(&|r| format!("            s{r} = s{r} + pr.x * x[{r}u * k + 2u * i];\n            s{r} = s{r} + pr.y * x[{r}u * k + 2u * i + 1u];"));
        let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r};"));
        let outs = each(&|r| format!("        var t{r} = 0.0;\n        for (var j = 0u; j < 8u; j++) {{ t{r} += part[{r}u * 256u + li + j]; }}\n        y[{r}u * n + o] = t{r};"));
        return format!(
            r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let o = wg.x * 32u + li / 8u;
    let q = li % 8u;
    let pairs = k / 2u;
{regs}
    if (o < n) {{
        for (var i = q; i < pairs; i += 8u) {{
            let pr = unpack2x16float(w[o * pairs + i]);
{sums}
        }}
    }}
{parts}
    workgroupBarrier();
    if (q == 0u && o < n) {{
{outs}
    }}
}}
"#,
            len = rows * 256
        );
    }
    let regs = each(&|r| format!("    var s{r} = vec4<f32>(0.0);"));
    let sums = each(&|r| format!("        s{r} += wv * x4[{r}u * k4 + i];"));
    let parts = each(&|r| format!("    part[{r}u * 256u + li] = s{r}.x + s{r}.y + s{r}.z + s{r}.w;"));
    let adds = each(&|r| format!("            part[{r}u * 256u + li] += part[{r}u * 256u + li + st];"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, {len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let o = wg.x;
{regs}
    for (var i = li; i < k4; i += 256u) {{
        let pr = w2[o * k4 + i];
        let a = unpack2x16float(pr.x);
        let b = unpack2x16float(pr.y);
        let wv = vec4<f32>(a.x, a.y, b.x, b.y);
{sums}
    }}
{parts}
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {{
        if (li < st) {{
{adds}
        }}
        workgroupBarrier();
    }}
    if (li < {rows}u) {{ y[li * n + o] = part[li * 256u]; }}
}}
"#,
        len = rows * 256
    )
}

/// [`MATVEC_F16`] for a short row (`k` under 2048): eight threads an output (a word, two weights, at a time), 32 outputs
/// a workgroup. `p[0]`: n, k.
pub(in crate::chain) const MATVEC_F16_NARROW: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let o = wg.x * 32u + li / 8u;
    let q = li % 8u;
    let pairs = k / 2u;
    var s = 0.0;
    if (o < n) {
        for (var i = q; i < pairs; i += 8u) {
            let pr = unpack2x16float(w[o * pairs + i]);
            s = s + pr.x * x[2u * i];
            s = s + pr.y * x[2u * i + 1u];
        }
    }
    part[li] = s;
    workgroupBarrier();
    if (q == 0u && o < n) {
        var t = 0.0;
        for (var j = 0u; j < 8u; j++) { t += part[li + j]; }
        y[o] = t;
    }
}
"#;

/// [`MATMUL_F32`] of one row with `k` a multiple of 4: vec4 loads, a thread's products in four running sums (its
/// loads in flight together). `p[0]`: n, k.
pub(in crate::chain) const MATVEC_F32_4: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k4 = p[0].y / 4u;
    let o = wg.x;
    var s = vec4<f32>(0.0);
    for (var i = li; i < k4; i += 256u) { s += w4[o * k4 + i] * x4[i]; }
    part[li] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { y[o] = part[0]; }
}
"#;

/// [`MATVEC_F32_4`] for 2 to 8 rows (a check of drafts): a workgroup an output, each weight read once for every row,
/// a row's sums in the workgroup's memory a slot each. `p[0]`: n, k, rows.
pub(in crate::chain) const MATVEC_F32_4_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = p[0].x;
    let k4 = p[0].y / 4u;
    let rows = p[0].z;
    let o = wg.x;
    var s: array<vec4<f32>, 8>;
    for (var r = 0u; r < 8u; r++) { s[r] = vec4<f32>(0.0); }
    for (var i = li; i < k4; i += 256u) {
        let wv = w4[o * k4 + i];
        for (var r = 0u; r < rows; r++) { s[r] += wv * x4[r * k4 + i]; }
    }
    for (var r = 0u; r < rows; r++) { part[r * 256u + li] = s[r].x + s[r].y + s[r].z + s[r].w; }
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) {
            for (var r = 0u; r < rows; r++) { part[r * 256u + li] += part[r * 256u + li + st]; }
        }
        workgroupBarrier();
    }
    if (li < rows) { y[li * n + o] = part[li * 256u]; }
}
"#;

/// `y[r, o] = sum over i of x[r, i] w[o, i]` for a prompt's rows ([`MATMUL_F32`]'s, of several): a workgroup a 64x64
/// tile of `y` (64 rows by 64 outputs), a thread 4x4 of it, `k` 16 at a time through the workgroup's memory; split
/// over `k` (the grid's third axis, `p[0].w` of it each), split `s`'s sums to part `s` of `y` (`[splits, rows, n]`)
/// for [`SUM_SPLITS`] to add up. `p[0]`: n, k, rows, k a split.
pub(in crate::chain) const MATMUL_F32_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// [16 of k][64 rows (or outputs)], a row of 68 values (the last 4 padding, against bank conflicts), read four at
// a time. Scalars, not vec4s: each thread writes one row's value, and WGSL lets a write to one component of a
// vector in memory write the whole vector (Metal does), so four threads writing a vec4's four would race
var<workgroup> xs: array<f32, 1088>;
var<workgroup> ws: array<f32, 1088>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let rows = p[0].z;
    let kc = p[0].w;
    let o0 = wg.x * 64u;
    let r0 = wg.y * 64u;
    let k0 = wg.z * kc;
    let k1 = min(k, k0 + kc);
    let tr = t / 16u;
    let to = t % 16u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var kb = k0; kb < k1; kb += 16u) {
        for (var q = 0u; q < 4u; q++) {
            let idx = t + q * 256u;
            let rr = idx / 16u;
            let kk = idx % 16u;
            let gk = kb + kk;
            var v = 0.0;
            if (r0 + rr < rows && gk < k1) {
                v = x[(r0 + rr) * k + gk];
            }
            xs[kk * 68u + rr] = v;
            var u = 0.0;
            if (o0 + rr < n && gk < k1) {
                u = w[(o0 + rr) * k + gk];
            }
            ws[kk * 68u + rr] = u;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let xi = kk * 68u + tr * 4u;
            let wi = kk * 68u + to * 4u;
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
            let wv = vec4<f32>(ws[wi], ws[wi + 1u], ws[wi + 2u], ws[wi + 3u]);
            a0 += xv.x * wv;
            a1 += xv.y * wv;
            a2 += xv.z * wv;
            a3 += xv.w * wv;
        }
        workgroupBarrier();
    }
    let base = wg.z * rows * n;
    let o = o0 + to * 4u;
    let r = r0 + tr * 4u;
    let acc = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i = 0u; i < 4u; i++) {
        if (r + i < rows) {
            for (var j = 0u; j < 4u; j++) {
                if (o + j < n) {
                    y[base + (r + i) * n + o + j] = acc[i][j];
                }
            }
        }
    }
}
"#;

/// ComfyUI's W4A8 decoded to f16 ([`ChainRecorder::w4a8_f16`]), a thread a code byte (two values, an f16 pair's
/// word). `p[0]`: the rows, the columns.
pub(in crate::chain) const W4A8_F16: &str = r#"
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> rel: array<u32>;
@group(0) @binding(2) var<storage, read> channel: array<f32>;
@group(0) @binding(3) var<storage, read> book: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let v = select(bitcast<f32>(((e + 120u) << 23u) | (m << 20u)), f32(m) * 0.001953125, e == 0u);
    return select(v, -v, (b & 128u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let cols = p[0].y;
    let bytes = cols / 2u;
    if (i >= p[0].x * bytes) { return; }
    let row = i / bytes;
    let col = 2u * (i % bytes);
    let b = (codes[i / 4u] >> (8u * (i % 4u))) & 255u;
    let gi = row * (cols / 16u) + col / 16u;
    let s = e4m3((rel[gi / 4u] >> (8u * (gi % 4u))) & 255u);
    let ch = channel[row];
    let lo = clamp(round(book[b & 15u] * s), -127.0, 127.0) * ch;
    let hi = clamp(round(book[b >> 4u] * s), -127.0, 127.0) * ch;
    out[i] = pack2x16float(vec2<f32>(lo, hi));
}
"#;

/// [`W4A8_F16`] with ConvRot's rotation undone (`GS`, a power of four, put in): a workgroup a row's group, decoded
/// into its memory, then the Hadamard matrix's passes of four (a digit of the group's base-4 index each), scaled by
/// the group's square root. `p[0]`: the rows, the columns; a row `wg.y + 65,535 wg.z`, its group `wg.x`.
pub(in crate::chain) const W4A8_F16_ROTATED: &str = r#"
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> rel: array<u32>;
@group(0) @binding(2) var<storage, read> channel: array<f32>;
@group(0) @binding(3) var<storage, read> book: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

const GS: u32 = GS_u;
var<workgroup> g: array<f32, GS_u>;

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let v = select(bitcast<f32>(((e + 120u) << 23u) | (m << 20u)), f32(m) * 0.001953125, e == 0u);
    return select(v, -v, (b & 128u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cols = p[0].y;
    let row = min(wg.y + wg.z * 65535u, p[0].x - 1u);
    let g0 = wg.x * GS;
    let ch = channel[row];
    for (var cb = t; cb < GS / 2u; cb += 64u) {
        let i = row * (cols / 2u) + g0 / 2u + cb;
        let b = (codes[i / 4u] >> (8u * (i % 4u))) & 255u;
        let gi = row * (cols / 16u) + (g0 + 2u * cb) / 16u;
        let s = e4m3((rel[gi / 4u] >> (8u * (gi % 4u))) & 255u);
        g[2u * cb] = clamp(round(book[b & 15u] * s), -127.0, 127.0) * ch;
        g[2u * cb + 1u] = clamp(round(book[b >> 4u] * s), -127.0, 127.0) * ch;
    }
    workgroupBarrier();
    for (var h = 1u; h < GS; h *= 4u) {
        for (var u = t; u < GS / 4u; u += 64u) {
            let j = (u / h) * 4u * h + u % h;
            let a = g[j];
            let b = g[j + h];
            let c = g[j + 2u * h];
            let d = g[j + 3u * h];
            g[j] = a + b + c - d;
            g[j + h] = a + b - c + d;
            g[j + 2u * h] = a - b + c + d;
            g[j + 3u * h] = -a + b + c + d;
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(f32(GS));
    for (var cb = t; cb < GS / 2u; cb += 64u) {
        out[row * (cols / 2u) + g0 / 2u + cb] = pack2x16float(vec2<f32>(g[2u * cb], g[2u * cb + 1u]) * scale);
    }
}
"#;

/// `y[i] = sum over s of part[s * len + i]`, the splits in order. `p[0]`: len, splits.
pub(in crate::chain) const SUM_SPLITS: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let len = p[0].x;
    let i = id.x + id.y * 65535u * 256u;
    if (i >= len) {
        return;
    }
    var acc = 0.0;
    for (var s = 0u; s < p[0].y; s++) {
        acc += part[s * len + i];
    }
    y[i] = acc;
}
"#;
