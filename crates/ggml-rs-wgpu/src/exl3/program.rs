//! A model's few steps between its projections, recorded with them as one call ([`run_program`]): a decode step of
//! DeepSeek-V4.1 made a round trip of each sublayer's projections, its hyper-connections' mixing (four copies of the
//! residual stream collapsed into a sublayer's input and spread back after it) and its norms on the host between them.
//! A program is the list of its steps, each making a vector (or writing into one) that later steps take by its step's
//! number, and the vectors read back at the end.

use super::*;

/// A step of a [`run_program`] program. A step's vector is known by the step's number; a step that writes into
/// another's vector ([`ProgramStep::CopyColsBf16`]) makes none of its own.
pub enum ProgramStep<'a> {
    /// The host's values.
    Input(&'a [f32]),
    /// `rows` rows of a vector of `len` values, written by the steps after.
    Scratch(usize),
    /// `m` of `rows` rows of vector `x` (`[rows, k]`): its sums, `[rows, n]`.
    Mul { m: ChainMat<'a>, x: usize, rows: usize },
    /// `jobs` of a group (`(part, row of x)` each, x's rows `k` wide) in one pass: each job's sums, `[jobs, n]`.
    MulGroup { g: &'a Exl3Group, x: usize, jobs: &'a [(u32, u32)] },
    /// `width` columns from `src_at` of `rows` rows of `src` (rows `src_stride` apart), each rounded to bf16, into
    /// `dst` from `dst_at` (rows `dst_stride` apart).
    CopyColsBf16 { src: usize, src_stride: usize, src_at: usize, dst: usize, dst_stride: usize, dst_at: usize, width: usize, rows: usize },
    /// The host's rmsnorm of `rows` rows of `x` (`width` each) by `weight`, its output rounded to bf16, to the bit
    /// (`chained_normed`'s norm).
    Norm { x: usize, weight: &'a [f32], eps: f32, rows: usize, width: usize },
    /// A hyper-connection's spread of a sublayer's output: `out` (`[d]`) and the residual's copies `res` (`[HC, d]`)
    /// by `mix` (`[pre 4, post 4, comb 4 x 4]`): copy `j`'s `bf16(post[j] out + sum_i comb[i][j] res[i])`, `[HC, d]`.
    HcPost { out: usize, res: usize, mix: usize, d: usize },
    /// A hyper-connection's collapse of the copies `h` (`[HC, d]`) into a sublayer's input by `mix`'s pre:
    /// `bf16(sum_i pre[i] h[i])`, `[d]`.
    HcPre { h: usize, mix: usize, d: usize },
    /// A gated unit's activation of `gate` and `up` (`len` each, rounded to bf16 first, as the host rounds a gated
    /// unit's projections): `bf16(silu[bits(g) >> 16] * clamp(u, ±limit))` (`limit` > 0; no clamp else), `silu` the
    /// host's value of every bf16 gate by its bits (`min(g, limit)` taken in them, 65,536 values), so each element is
    /// the host's to the bit, `[len]`.
    SwigluBf16 { gate: usize, up: usize, len: usize, silu: &'a [f32], limit: f32 },
}

/// `x` as an input of `steps`: the step of an input of the same host rows already there (a layer's projections of x
/// take one upload), else a new one. Its step's number.
pub(super) fn input_once<'a>(steps: &mut Vec<ProgramStep<'a>>, x: &'a [f32]) -> usize {
    let same = steps.iter().position(|s| matches!(s, ProgramStep::Input(y) if y.as_ptr() == x.as_ptr() && y.len() == x.len()));
    same.unwrap_or_else(|| {
        steps.push(ProgramStep::Input(x));
        steps.len() - 1
    })
}

/// The copies of a hyper-connected residual stream (DeepSeek-V4.1's `hc_mult`).
const HC: usize = 4;

/// [`ProgramStep::HcPost`]'s kernel: a thread an element of `out`, each copy's sum in the host's order.
const HC_POST: &str = r#"
@group(0) @binding(0) var<storage, read> out: array<f32>;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read> mix: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let k = id.x;
    if (k >= d) { return; }
    let o = out[k];
    let r0 = res[k];
    let r1 = res[d + k];
    let r2 = res[2u * d + k];
    let r3 = res[3u * d + k];
    for (var j = 0u; j < 4u; j++) {
        let r = mix[8u + j] * r0 + mix[12u + j] * r1 + mix[16u + j] * r2 + mix[20u + j] * r3;
        y[j * d + k] = bf16(mix[4u + j] * o + r);
    }
}
"#;

/// [`ProgramStep::SwigluBf16`]'s kernel: a thread an element, `p[0].y` the limit's bits.
const SWIGLU_BF16: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> silu: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p[0].x) { return; }
    let lim = bitcast<f32>(p[0].y);
    var u = bf16(up[i]);
    if (lim > 0.0) { u = clamp(u, -lim, lim); }
    y[i] = bf16(silu[bitcast<u32>(bf16(gate[i])) >> 16u] * u);
}
"#;

/// [`ProgramStep::HcPre`]'s kernel: a thread an element.
const HC_PRE: &str = r#"
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> mix: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let k = id.x;
    if (k >= d) { return; }
    y[k] = bf16(mix[0] * h[k] + mix[1] * h[d + k] + mix[2] * h[2u * d + k] + mix[3] * h[3u * d + k]);
}
"#;

/// Run `steps` on `b` as one recording and one read back: the vectors `reads` names (`(step, first value, values)`),
/// in order. None where a matrix is not one this device's chained recordings take, or a step names a vector it cannot
/// (a later step's, or one of another shape).
pub fn run_program(b: &WgpuBackend, steps: &[ProgramStep<'_>], reads: &[(usize, usize, usize)]) -> Option<Vec<Vec<f32>>> {
    use ggml_rs::ChainRecorder;
    // each step's vector's length (0: it makes none), checked against what the steps after it ask of it
    let mut lens: Vec<usize> = Vec::with_capacity(steps.len());
    let mut rows_one = true;
    for (i, step) in steps.iter().enumerate() {
        let before = |v: usize| (v < i && lens[v] > 0).then(|| lens[v]);
        let len = match step {
            ProgramStep::Input(x) => x.len(),
            ProgramStep::Scratch(len) => *len,
            ProgramStep::Mul { m, x, rows } => {
                let (k, n) = chain_shape(b, m)?;
                rows_one &= *rows == 1;
                (before(*x)? >= rows * k).then_some(rows * n)?
            }
            ProgramStep::MulGroup { g, x, jobs } => {
                let (k, n) = g.kn();
                let rows = jobs.iter().map(|&(_, r)| r as usize + 1).max()?;
                (jobs.iter().all(|&(m, _)| (m as usize) < g.count()) && before(*x)? >= rows * k).then_some(jobs.len() * n)?
            }
            ProgramStep::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => {
                let ok = before(*src)? >= (rows - 1) * src_stride + src_at + width && before(*dst)? >= (rows - 1) * dst_stride + dst_at + width;
                ok.then_some(0)?
            }
            ProgramStep::Norm { x, weight, rows, width, .. } => (before(*x)? >= rows * width && weight.len() == *width && *width <= 1 << 20).then_some(rows * width)?,
            ProgramStep::HcPost { out, res, mix, d } => (before(*out)? >= *d && before(*res)? >= HC * d && before(*mix)? >= 24).then_some(HC * d)?,
            ProgramStep::HcPre { h, mix, d } => (before(*h)? >= HC * d && before(*mix)? >= 24).then_some(*d)?,
            ProgramStep::SwigluBf16 { gate, up, len, silu, .. } => (before(*gate)? >= *len && before(*up)? >= *len && silu.len() == 1 << 16).then_some(*len)?,
        };
        lens.push(len);
    }
    if reads.iter().any(|&(s, at, len)| s >= steps.len() || at + len > lens[s]) {
        return None;
    }
    // The program's vectors kept from its first run (an arena a program: a layer's), so its dispatches' bind groups are
    // the same every step and kept with them; the inputs written each run, the norms' weights and the groups' job lists
    // once. Taken from the recording's pooled scratch, every dispatch made its bind group (and a parameters' buffer)
    // again at every step, some 95 us of the host's work a call, and the kept groups' store started over every few
    // tokens.
    let arena = arena(b, steps, &lens);
    let arena = arena.lock().unwrap_or_else(|p| p.into_inner());
    let mut rec = crate::chain::Recorder::new(b);
    rec.keep_groups(rows_one);
    for (step, v) in steps.iter().zip(&arena.vecs) {
        if let (ProgramStep::Input(x), Some(v)) = (step, v) {
            DeviceChain::upload(b, v, x);
        }
    }
    let (vecs, norm_weights, groups, tables) = (&arena.vecs, &arena.norms, &arena.groups, &arena.tables);
    let v = |i: usize| vecs[i].as_ref().expect("a step's vector");
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector").clone();
    let (dummy, drw) = (rec.gpu().dummy().clone(), rec.gpu().dummy_rw().clone());
    for (i, step) in steps.iter().enumerate() {
        match step {
            ProgramStep::Input(_) | ProgramStep::Scratch(_) => {}
            ProgramStep::Mul { m, x, rows } => {
                let kn = chain_shape(b, m).expect("checked above");
                chain_mul(&mut rec, *m, kn, v(*x), v(i), *rows);
            }
            ProgramStep::MulGroup { g, x, .. } => g.record(&mut rec, groups[i].as_ref().expect("its scratch"), v(*x), v(i)),
            ProgramStep::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => {
                rec.copy_cols_bf16(v(*src), *src_stride, *src_at, v(*dst), *dst_stride, *dst_at, *width, *rows);
            }
            ProgramStep::Norm { x, eps, rows, width, .. } => {
                let w = norm_weights[i].as_ref().expect("its weight");
                rec.dispatch_wide("exl3-rmsnorm-host", NORM_AS_HOST, [&buf(v(*x)), &buf(w), &dummy, &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*width as u32, 0, eps.to_bits()], (*rows as u32, 1, 1));
            }
            ProgramStep::HcPost { out, res, mix, d } => {
                rec.dispatch_wide("ds-hc-post", HC_POST, [&buf(v(*out)), &buf(v(*res)), &buf(v(*mix)), &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*d as u32], ((*d as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::HcPre { h, mix, d } => {
                rec.dispatch_wide("ds-hc-pre", HC_PRE, [&buf(v(*h)), &dummy, &buf(v(*mix)), &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*d as u32], ((*d as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::SwigluBf16 { gate, up, len, limit, .. } => {
                let table = tables[i].as_ref().expect("its table");
                rec.dispatch_wide("ds-swiglu-bf16", SWIGLU_BF16, [&buf(v(*gate)), &buf(v(*up)), &buf(table), &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*len as u32, limit.to_bits()], ((*len as u32).div_ceil(256), 1, 1));
            }
        }
    }
    for &(s, at, len) in reads {
        rec.read_range(v(s), at, len);
    }
    Some(Box::new(rec).finish())
}

/// A program's vectors on its device ([`run_program`]'s arena): each step's, each norm's weight, each group's scratch,
/// each activation's table.
pub(crate) struct ProgramArena {
    vecs: Vec<Option<DeviceVec>>,
    norms: Vec<Option<DeviceVec>>,
    groups: Vec<Option<GroupScratch>>,
    tables: Vec<Option<DeviceVec>>,
}

/// Arenas a device keeps at most before they start over (a program a layer and kind: some two hundred a model).
const ARENAS: usize = 4096;

/// `steps`' arena on `b` (vectors `lens` long), made at the program's first run and kept by the device (let go with it):
/// a program is known by its steps' kinds and shapes, its matrices, its norms' weights and its groups' jobs.
fn arena(b: &WgpuBackend, steps: &[ProgramStep<'_>], lens: &[usize]) -> Arc<Mutex<ProgramArena>> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let mat = |m: &ChainMat<'_>, h: &mut std::collections::hash_map::DefaultHasher| match m {
        ChainMat::Exl3(l) => (0u8, *l as *const dyn PackedLinear as *const () as usize).hash(h),
        ChainMat::F16 { w, n, k } | ChainMat::Bf16 { w, n, k } | ChainMat::F32 { w, n, k } => (1u8, Arc::as_ptr(&w.inner) as *const () as usize, *n, *k).hash(h),
    };
    for (step, &len) in steps.iter().zip(lens) {
        len.hash(&mut h);
        match step {
            ProgramStep::Input(_) => 0u8.hash(&mut h),
            ProgramStep::Scratch(_) => 1u8.hash(&mut h),
            ProgramStep::Mul { m, x, rows } => {
                (2u8, *x, *rows).hash(&mut h);
                mat(m, &mut h);
            }
            ProgramStep::MulGroup { g, x, jobs } => (3u8, *g as *const Exl3Group as usize, *x, *jobs).hash(&mut h),
            ProgramStep::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => (4u8, [*src, *src_stride, *src_at, *dst, *dst_stride, *dst_at, *width, *rows]).hash(&mut h),
            ProgramStep::Norm { x, weight, eps, rows, width } => (5u8, *x, weight.as_ptr() as usize, weight.len(), eps.to_bits(), *rows, *width).hash(&mut h),
            ProgramStep::HcPost { out, res, mix, d } => (6u8, [*out, *res, *mix, *d]).hash(&mut h),
            ProgramStep::HcPre { h: x, mix, d } => (7u8, [*x, *mix, *d]).hash(&mut h),
            ProgramStep::SwigluBf16 { gate, up, len, silu, limit } => (8u8, [*gate, *up, *len], silu.as_ptr() as usize, limit.to_bits()).hash(&mut h),
        }
    }
    let key = h.finish();
    let mut kept = b.gpu.programs.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(a) = kept.get(&key) {
        return Arc::clone(a);
    }
    if kept.len() >= ARENAS {
        kept.clear();
    }
    let up = |values: &[f32]| {
        let v = b.vec(values.len());
        DeviceChain::upload(b, &v, values);
        v
    };
    let made = ProgramArena {
        vecs: lens.iter().map(|&len| (len > 0).then(|| b.vec(len))).collect(),
        norms: steps.iter().map(|s| if let ProgramStep::Norm { weight, .. } = s { Some(up(weight)) } else { None }).collect(),
        groups: steps.iter().map(|s| if let ProgramStep::MulGroup { g, jobs, .. } = s { Some(g.scratch_kept(b, jobs)) } else { None }).collect(),
        tables: steps.iter().map(|s| if let ProgramStep::SwigluBf16 { silu, .. } = s { Some(up(silu)) } else { None }).collect(),
    };
    let made = Arc::new(Mutex::new(made));
    kept.insert(key, Arc::clone(&made));
    made
}