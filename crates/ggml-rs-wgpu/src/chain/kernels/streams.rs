//! A hyper-connection's kernels (Qwen3.8-Flash-Next's four residual streams): its gates, its branch input, its write-back.

/// A hyper-connection's gates, `p[0]`: rank, writes, streams, rows. `t` (binding 6) `[rows, rank + writes]`: its first
/// `rank` become `silu(t / streams)`, the rest 0; `post` (binding 7) `[rows, writes]` gets their `2 sigmoid(t /
/// streams)`. As the host's `hc_gates`.
pub(in crate::chain) const HC_GATES: &str = r#"
@group(0) @binding(6) var<storage, read_write> t: array<f32>;
@group(0) @binding(7) var<storage, read_write> post: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rank = p[0].x;
    let writes = p[0].y;
    let streams = f32(p[0].z);
    let width = rank + writes;
    let i = id.x;
    if (i >= width * p[0].w) { return; }
    let r = i / width;
    let c = i % width;
    let v = t[i] / streams;
    if (c < rank) {
        t[i] = v / (1.0 + exp(-v));
    } else {
        post[r * writes + c - rank] = 2.0 / (1.0 + exp(-v));
        t[i] = 0.0;
    }
}
"#;

/// A hyper-connection's branch input: `y[r, j] = sum over s of x[r, s, j] / (1 + exp(-w[r, s, j])) / streams` (`w` the
/// logits, `x` the normed streams), `p[0]`: d, streams, rows. As the host's `hc_mix`.
pub(in crate::chain) const HC_MIX: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let streams = p[0].y;
    let i = id.x;
    if (i >= d * p[0].z) { return; }
    let r = i / d;
    let j = i % d;
    var acc = 0.0;
    for (var s = 0u; s < streams; s++) {
        let at = (r * streams + s) * d + j;
        acc += x[at] / (1.0 + exp(-bitcast<f32>(w[at]))) / f32(streams);
    }
    y[i] = acc;
}
"#;

/// A hyper-connection site's write-back: `y[r, s, j] += w[r, s] * x[r, j]` (`w` the write weights, `x` the branch
/// output, `y` the streams), `p[0]`: d, streams, rows. As the host's `stream_apply`.
pub(in crate::chain) const STREAM_APPLY: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let streams = p[0].y;
    let i = id.x;
    if (i >= d * streams * p[0].z) { return; }
    let r = i / (d * streams);
    let s = (i / d) % streams;
    let j = i % d;
    y[i] = y[i] + bitcast<f32>(w[r * streams + s]) * x[r * d + j];
}
"#;
