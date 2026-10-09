//! A MoE layer's routing on the device: a row's top experts and their weights, the weighted sums of their
//! outputs, and a prompt's jobs counted, scanned and scattered into blocks.

use super::*;

/// Each row's experts summed in its own order, each weighted (as `Exl3MoeHost::forward`): `out[r, i] = sum over j < K
/// of w[r, j] * d[r K + j, i]`, then `+ w[r, K] * sh[r, i]` (the shared expert). `p[0]`: hidden, K, rows.
pub(crate) const WSUM_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(1) var<storage, read> sh: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let h = p[0].x;
    let kk = p[0].y;
    let i = id.x;
    if (i >= h * p[0].z) { return; }
    let r = i / h;
    let c = i % h;
    var acc = 0.0;
    for (var j = 0u; j < kk; j++) { acc += w[r * (kk + 1u) + j] * d[(r * kk + j) * h + c]; }
    acc += w[r * (kk + 1u) + kk] * sh[r * h + c];
    out[i] = acc;
}
"#;

/// [`WSUM_ROWS`] with each row's sum added to its streams (`xs[r, s] += post[r, s] * sum`, `p[0].w` streams), as
/// `ChainRecorder::stream_apply` writes it back. `p[0]`: hidden, K, rows, streams.
pub(crate) const WSUM_APPLY: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(1) var<storage, read> sh: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> post: array<f32>;
@group(0) @binding(6) var<storage, read_write> xs: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let h = p[0].x;
    let kk = p[0].y;
    let streams = p[0].w;
    let i = id.x;
    if (i >= h * p[0].z) { return; }
    let r = i / h;
    let c = i % h;
    var acc = 0.0;
    for (var j = 0u; j < kk; j++) { acc += w[r * (kk + 1u) + j] * d[(r * kk + j) * h + c]; }
    acc += w[r * (kk + 1u) + kk] * sh[r * h + c];
    for (var s = 0u; s < streams; s++) {
        let at = (r * streams + s) * h + c;
        xs[at] = xs[at] + post[r * streams + s] * acc;
    }
}
"#;

/// The routing on the GPU, as `ggml_rs::exl3::route` routes, in two kernels ([`record_route`]). This one ranks: the
/// top `p[0].y` of a row's `p[0].x` routed logits by logit, a tie to the lower index (each logit placed by how many
/// beat it), and writes their down jobs by rank, `[e, j]` for pair `j = r k + rank` (the experts' down projections
/// take hidden row `j`). A workgroup 64 of a row's logits (the grid's second axis), four threads a logit, each every
/// fourth of the comparisons' quads: one workgroup a row gave a thread two or three logits' every comparison, some 390
/// turns of four, and a layer's routing was 14.6 us whatever its rows (0.7 ms of a Flash-Next step's 11). `p[0]`:
/// routed (at most 1024), k (at most 32).
pub(crate) const ROUTE_RANK: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(6) var<storage, read_write> jd: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// the row's logits four at a time (the comparisons' reads)
var<workgroup> l4: array<vec4<f32>, 256>;
var<workgroup> beat: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let r = wg.x;
    let lr = r * (n + 1u);
    let quads = (n + 3u) / 4u;
    // past the last logit, -inf (beats none)
    if (t < quads) {
        var v = vec4<f32>(bitcast<f32>(0xff800000u));
        for (var c = 0u; c < 4u; c++) {
            if (4u * t + c < n) { v[c] = logits[lr + 4u * t + c]; }
        }
        l4[t] = v;
    }
    workgroupBarrier();
    let i = wg.y * 64u + t / 4u;
    let live = i < n;
    var above = 0u;
    if (live) {
        // how many beat logit `i` (a higher logit, or as high at a lower index), among this thread's quads
        let v = logits[lr + i];
        for (var q = t % 4u; q < quads; q += 4u) {
            let u = l4[q];
            let j = 4u * q;
            above += select(0u, 1u, u.x > v || (u.x == v && j < i));
            above += select(0u, 1u, u.y > v || (u.y == v && j + 1u < i));
            above += select(0u, 1u, u.z > v || (u.z == v && j + 2u < i));
            above += select(0u, 1u, u.w > v || (u.w == v && j + 3u < i));
        }
    }
    beat[t] = above;
    workgroupBarrier();
    if (live && t % 4u == 0u) {
        let rank = beat[t] + beat[t + 1u] + beat[t + 2u] + beat[t + 3u];
        if (rank < k) {
            let pair = r * k + rank;
            jd[2u * pair] = i;
            jd[2u * pair + 1u] = pair;
        }
    }
}
"#;

/// [`ROUTE_RANK`]'s second kernel, a workgroup a row `r`, a thread an expert of its top k: their weights (a softmax
/// among themselves, their exponentials summed in rank order) and the shared expert's (the logit after the routed
/// ones, by its gate's sigmoid), `w` from `r (k + 1)`: the top k's, the shared one's last; and the row's jobs as the
/// grouped experts take them, gate and up `[2e, r, 2e + 1, r]` an expert (`jobs`, from pair `r k`). `p[0]`: routed, k
/// (at most 32).
pub(crate) const ROUTE_WEIGHTS: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> jd: array<u32>;
@group(0) @binding(6) var<storage, read_write> jobs: array<u32>;
@group(0) @binding(7) var<storage, read_write> w: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> ex: array<f32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let r = wg.x;
    let lr = r * (n + 1u);
    let mx = logits[lr + jd[2u * r * k]];
    var e = 0u;
    var v = 0.0;
    if (j < k) {
        e = jd[2u * (r * k + j)];
        v = exp(logits[lr + e] - mx);
    }
    ex[j] = v;
    workgroupBarrier();
    if (j < k) {
        var sum = 0.0;
        for (var i = 0u; i < k; i++) { sum += ex[i]; }
        w[r * (k + 1u) + j] = v / sum;
        let at = 4u * (r * k + j);
        jobs[at] = 2u * e;
        jobs[at + 1u] = r;
        jobs[at + 2u] = 2u * e + 1u;
        jobs[at + 3u] = r;
    }
    if (j == 0u) {
        w[r * (k + 1u) + k] = 1.0 / (1.0 + exp(-logits[lr + n]));
    }
}
"#;

/// `rows` rows' routing recorded from their router's `logits` (`[rows, routed + 1]`): each row's top `top_k` experts'
/// down jobs by rank into `st` ([`ROUTE_RANK`]), then their weights and gate and up jobs ([`ROUTE_WEIGHTS`]).
pub(crate) fn record_route(rec: &mut crate::chain::Recorder<'_>, logits: &wgpu::Buffer, st: &Step, routed: usize, top_k: usize, rows: usize) {
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
    assert!(routed <= 1024 && top_k <= 32 && top_k <= routed, "moe: the top {top_k} of {routed} experts");
    let d = rec.gpu().dummy().clone();
    let drw = rec.gpu().dummy_rw().clone();
    let words = [routed as u32, top_k as u32];
    let jd = buf(&st.jobs_d);
    rec.dispatch_wide("moe-route-rank", ROUTE_RANK, [logits, &d, &d, &d, &d, &d, &jd, &drw], &words, (rows as u32, (routed as u32).div_ceil(64), 1));
    rec.dispatch_wide("moe-route-weights", ROUTE_WEIGHTS, [logits, &jd, &d, &d, &d, &d, &buf(&st.jobs_gu), &buf(&st.w)], &words, (rows as u32, 1, 1));
}

/// A prompt's routed jobs grouped by expert on the GPU, as [`many_order`] groups them on the host: each expert's down
/// jobs in blocks of `p[0].w` (a block one expert's, its unused places [`NONE`]; the blocks in the experts' order, a
/// job's place among its expert's as the atomics fall), and their gate and up jobs in blocks `2 b` and `2 b + 1`.
/// `od`: the down order (`p[0].y` blocks), then each expert's count, first block and filled places (`p[0].z`
/// experts); `og` the gate and up order. First every place unused and every count 0.
///
/// `p[1].x` not 0: an expert with that many jobs or fewer takes no block of those; its jobs go to a block of its own
/// of `p[1].x` places in a second order, the down one from `od[p[1].y]` (block `e` expert `e`'s) and the gate and up
/// one from `og[p[1].z]` (blocks `2 e` and `2 e + 1`): the few rows' kernel's, where the tensor cores' would decode
/// a matrix for a block of mostly empty rows.
pub(crate) const MANY_CLEAR: &str = r#"
@group(0) @binding(6) var<storage, read_write> og: array<u32>;
@group(0) @binding(7) var<storage, read_write> od: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 65535u * 256u;
    let places = p[0].w * p[0].y;
    if (i < 2u * places) { og[i] = 0xffffffffu; }
    if (i < places) { od[i] = 0xffffffffu; }
    if (i < 3u * p[0].z) { od[places + i] = 0u; }
    let few = p[1].x;
    if (few > 0u) {
        if (i < 2u * p[0].z * few) { og[p[1].z + i] = 0xffffffffu; }
        if (i < p[0].z * few) { od[p[1].y + i] = 0xffffffffu; }
    }
}
"#;

/// [`MANY_CLEAR`]'s second pass: each expert's jobs counted, a thread a pair. `p[0]`: the pairs, the blocks.
pub(crate) const MANY_COUNT: &str = r#"
@group(0) @binding(0) var<storage, read> jd: array<u32>;
@group(0) @binding(7) var<storage, read_write> od: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if (j >= p[0].x) { return; }
    atomicAdd(&od[p[0].w * p[0].y + jd[2u * j]], 1u);
}
"#;

/// [`MANY_CLEAR`]'s third: each expert's first block, the blocks of the experts before it, one workgroup a thread an
/// expert (1024 at most). `p[0]`: the pairs, the blocks, the experts, the jobs a block.
pub(crate) const MANY_SCAN: &str = r#"
@group(0) @binding(7) var<storage, read_write> od: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sums: array<u32, 1024>;

@compute @workgroup_size(1024)
fn main(@builtin(local_invocation_index) e: u32) {
    let bs = p[0].w;
    let at = bs * p[0].y;
    let ne = p[0].z;
    var nb = 0u;
    if (e < ne) { nb = (od[at + e] + bs - 1u) / bs; }
    // (an expert of few jobs: none of these blocks, its own in the second order)
    if (p[1].x > 0u && e < ne && od[at + e] <= p[1].x) { nb = 0u; }
    sums[e] = nb;
    workgroupBarrier();
    for (var st = 1u; st < 1024u; st *= 2u) {
        var v = sums[e];
        if (e >= st) { v += sums[e - st]; }
        workgroupBarrier();
        sums[e] = v;
        workgroupBarrier();
    }
    if (e < ne) { od[at + ne + e] = sums[e] - nb; }
}
"#;

/// [`MANY_CLEAR`]'s last: each pair's down job into its expert's blocks, and its gate and up jobs (`2 j`, `2 j + 1`)
/// into theirs, a thread a pair. `p[0]`: the pairs, the blocks, the experts.
pub(crate) const MANY_SCATTER: &str = r#"
@group(0) @binding(0) var<storage, read> jd: array<u32>;
@group(0) @binding(6) var<storage, read_write> og: array<u32>;
@group(0) @binding(7) var<storage, read_write> od: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if (j >= p[0].x) { return; }
    let bs = p[0].w;
    let at = bs * p[0].y;
    let ne = p[0].z;
    let e = jd[2u * j];
    let pos = atomicAdd(&od[at + 2u * ne + e], 1u);
    let few = p[1].x;
    if (few > 0u && atomicLoad(&od[at + e]) <= few) {
        atomicStore(&od[p[1].y + e * few + pos], j);
        og[p[1].z + 2u * e * few + pos] = 2u * j;
        og[p[1].z + (2u * e + 1u) * few + pos] = 2u * j + 1u;
        return;
    }
    let b = atomicLoad(&od[at + ne + e]) + pos / bs;
    let slot = pos % bs;
    atomicStore(&od[b * bs + slot], j);
    og[2u * b * bs + slot] = 2u * j;
    og[(2u * b + 1u) * bs + slot] = 2u * j + 1u;
}
"#;
