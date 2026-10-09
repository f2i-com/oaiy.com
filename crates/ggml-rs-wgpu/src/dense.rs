//! Block-scaled dense weights on the GPU, the kinds DeepSeek-V4.1 holds: fp8 e4m3 `[n, k]` with one scale per 32x32
//! tile and bf16 `[n, k]` (its trunk: attention projections, shared experts, router, Engram projection, head), and
//! MXFP4 `[n, k]` (e2m1, two to a byte, low nibble first) with one scale per row per 32 of `k` (its routed experts). The
//! caller (dsv41) quantizes the activation as the reference does and rounds the result; here
//! `y[t, r] = Σ_k x[t, k] · W[r, k]` in f32, every weight decoded exactly (a scale is an exact power of two, given as
//! f32 by the caller), so only the order of the sum differs from the CPU's.
//!
//! The rows are kept in buffers below the binding limit, whole 32-row tiles each (their scales beside their weights in
//! the same buffer). A call records a dispatch per buffer its rows touch and the copy back in one submit, and
//! [`forward_batch`] does that for several weights at once (a layer's experts): a kernel for a few tokens (a decode
//! step: 32 lanes a row, 8 rows a workgroup) and a tiled one for more (64 tokens by 64 rows, 32 of `k` a step).
//!
//! A routed expert's record can also be read as stored ([`RecordSlots`]): uploaded whole into a slot made once, its
//! MXFP4 matrices read in place with their e8m0 scales (a byte each), so a prompt's busy experts cost one write each
//! and no conversion on the host.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{chunk_limit, Gpu, WgpuBackend};

/// Tokens the decode kernel takes in one call; more go to the tiled one.
const FEW: usize = 8;

/// A dense weight as stored.
pub enum DenseData {
    /// e4m3 bytes `[n, k]` and one scale per 32x32 tile, `[ceil(n/32), k/32]`, as f32 (an exact power of two).
    Fp8 { w: Vec<u8>, scales: Vec<f32>, n: usize, k: usize },
    /// bf16 bits `[n, k]`.
    Bf16 { w: Vec<u16>, n: usize, k: usize },
    /// e2m1 nibbles `[n, k]`, two to a byte (low first), and one scale per row per 32 of k, `[n, k/32]`, as f32.
    Mxfp4 { w: Vec<u8>, scales: Vec<f32>, n: usize, k: usize },
}

/// Which of the three, and MXFP4 as a record holds it (its scales e8m0 bytes, in the same buffer); also the kernels'
/// `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Fp8 = 0,
    Bf16 = 1,
    Mxfp4 = 2,
    Record = 3,
}

impl DenseData {
    fn nk(&self) -> (usize, usize) {
        match self {
            DenseData::Fp8 { n, k, .. } | DenseData::Bf16 { n, k, .. } | DenseData::Mxfp4 { n, k, .. } => (*n, *k),
        }
    }
}

const COMMON: &str = r#"
@group(0) @binding(0) var<storage, read> wbuf: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
// k, tokens, the buffer's first row, its rows, the first row asked for, the rows asked for, where its scales start
// (words; bytes for a record), the kind (0 fp8, 1 bf16, 2 mxfp4, 3 a record's mxfp4), and where its weights start
// (words; 0 but in a record).
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 3>;

fn fp8(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    // e4m3fn has no infinity: all ones is NaN, as the reference decodes it.
    if (e == 15u && m == 7u) { return bitcast<f32>(0x7fc00000u); }
    var v: f32;
    if (e == 0u) {
        v = f32(m) * 0.001953125;
    } else {
        v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u));
    }
    return select(v, -v, (b & 128u) != 0u);
}

// e2m1: 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives.
fn fp4(n: u32) -> f32 {
    let m = n & 7u;
    var v = f32(m) * 0.5;
    if (m >= 4u) { v = select(f32(m) - 2.0, 6.0, m == 7u); }
    return select(v, -v, (n & 8u) != 0u);
}

// Weights a word holds: 4 fp8, 2 bf16, 8 e2m1.
fn wide(kind: u32) -> u32 {
    return select(select(4u, 2u, kind == 1u), 8u, kind >= 2u);
}

// The j-th weight of a word.
fn weight(word: u32, kind: u32, j: u32) -> f32 {
    if (kind == 0u) { return fp8((word >> (8u * j)) & 255u); }
    if (kind >= 2u) { return fp4((word >> (4u * j)) & 15u); }
    return select(bitcast<f32>(word & 0xffff0000u), bitcast<f32>(word << 16u), j == 0u);
}

// An e8m0 byte as the reference decodes it: 2^(e - 127), 0 the f32 subnormal 2^-127, 255 NaN.
fn e8m0(e: u32) -> f32 {
    if (e == 255u) { return bitcast<f32>(0x7fc00000u); }
    if (e == 0u) { return bitcast<f32>(0x00400000u); }
    return bitcast<f32>(e << 23u);
}

// The scale of the weights of row `local` (in this buffer) at k column `c`: a 32x32 tile's (fp8), a row's 32 (mxfp4,
// as f32; a record's, as an e8m0 byte), none (bf16).
fn scale(kind: u32, soff: u32, local: u32, c: u32, kb: u32) -> f32 {
    if (kind == 0u) { return bitcast<f32>(wbuf[soff + (local / 32u) * kb + c / 32u]); }
    if (kind == 2u) { return bitcast<f32>(wbuf[soff + local * kb + c / 32u]); }
    if (kind == 3u) {
        let b = soff + local * kb + c / 32u;
        return e8m0((wbuf[b / 4u] >> (8u * (b % 4u))) & 255u);
    }
    return 1.0;
}
"#;

/// The decode kernel: 8 rows a workgroup, 32 lanes a row, each lane a word in 32 of the row, every token at once.
const FEW_KERNEL: &str = r#"
var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k = p[0].x; let t = p[0].y; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z; let kind = p[1].w;
    let woff = p[2].x;
    let lane = li & 31u;
    let slot = li >> 5u;
    // The rows this buffer and the call share: [max(first, lo), min(first + rows, lo + count)).
    let start = max(first, lo);
    let row = start + wg.x * 8u + slot;
    let live = row < min(first + rows, lo + count);
    var acc: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) { acc[i] = 0.0; }
    if (live) {
        let local = row - first;
        let wd = wide(kind);
        let per = k / wd;
        let kb = k / 32u;
        for (var w = lane; w < per; w += 32u) {
            let word = wbuf[woff + local * per + w];
            let c = w * wd;
            let s = scale(kind, soff, local, c, kb);
            for (var tt = 0u; tt < t; tt++) {
                var d = 0.0;
                for (var j = 0u; j < wd; j++) { d += weight(word, kind, j) * x[tt * k + c + j]; }
                acc[tt] += d * s;
            }
        }
    }
    for (var tt = 0u; tt < 8u; tt++) { part[(slot * 32u + lane) * 8u + tt] = acc[tt]; }
    workgroupBarrier();
    if (live && lane == 0u) {
        for (var tt = 0u; tt < t; tt++) {
            var sum = 0.0;
            for (var l = 0u; l < 32u; l++) { sum += part[(slot * 32u + l) * 8u + tt]; }
            y[tt * count + (row - lo)] = sum;
        }
    }
}
"#;

/// The prompt kernel: 64 tokens by 64 rows a workgroup, 32 of k a step; each thread 4 tokens by 4 rows, its sums four
/// vec4s (rows) and the decoded weights kept as vec4s of 4 rows, read a vec4 at a time. (Its sums an array indexed in
/// loops, the compiler kept them in memory, not registers: 64 expert matrices of 31 tokens took 0.15 s.)
const MANY_KERNEL: &str = r#"
var<workgroup> xs: array<f32, 2048>;
// k step kk, rows 4q..4q+3: wt[kk * 16 + q]
var<workgroup> wt: array<vec4<f32>, 512>;

// Token `tok`'s sums of rows row0..row0+3, those in [lo, end) and of a token there is.
fn put(v: vec4<f32>, tok: u32, row0: u32, t: u32, end: u32, lo: u32, count: u32) {
    if (tok >= t) { return; }
    for (var b = 0u; b < 4u; b++) {
        if (row0 + b < end) { y[tok * count + (row0 + b - lo)] = v[b]; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k = p[0].x; let t = p[0].y; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z; let kind = p[1].w;
    let woff = p[2].x;
    let start = max(first, lo);
    let end = min(first + rows, lo + count);
    let r0 = start + wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    let wd = wide(kind);
    let per = k / wd;
    // Words a row holds in a 32-wide step of k.
    let wps = 32u / wd;
    let kb = k / 32u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < k; k0 += 32u) {
        // The tokens' 64 x 32: element e, token e / 32, k e % 32, kept k-major.
        for (var e = li; e < 2048u; e += 256u) {
            let tok = t0 + e / 32u;
            let kk = e % 32u;
            var v = 0.0;
            if (tok < t) { v = x[tok * k + k0 + kk]; }
            xs[kk * 64u + e / 32u] = v;
        }
        // The rows' 64 x 32, decoded and scaled: a word an element.
        for (var e = li; e < 64u * wps; e += 256u) {
            let r = e / wps;
            let q = e % wps;
            let row = r0 + r;
            let live = row < end;
            var word = 0u;
            var s = 0.0;
            if (live) {
                let local = row - first;
                word = wbuf[woff + local * per + k0 / wd + q];
                s = scale(kind, soff, local, k0, kb);
            }
            for (var j = 0u; j < wd; j++) {
                wt[(q * wd + j) * 16u + r / 4u][r % 4u] = select(0.0, weight(word, kind, j) * s, live);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let w = wt[kk * 16u + tx];
            let xb = kk * 64u + ty * 4u;
            acc0 += xs[xb] * w;
            acc1 += xs[xb + 1u] * w;
            acc2 += xs[xb + 2u] * w;
            acc3 += xs[xb + 3u] * w;
        }
        workgroupBarrier();
    }
    let tok = t0 + ty * 4u;
    let row0 = r0 + tx * 4u;
    put(acc0, tok, row0, t, end, lo, count);
    put(acc1, tok + 1u, row0, t, end, lo, count);
    put(acc2, tok + 2u, row0, t, end, lo, count);
    put(acc3, tok + 3u, row0, t, end, lo, count);
}
"#;

fn shader(many: bool) -> String {
    format!("{COMMON}{}", if many { MANY_KERNEL } else { FEW_KERNEL })
}

/// A decode step's kernel ([`Arena`]'s): one row of x against a weight of `kind`, written for that kind. 8 rows a
/// workgroup and 32 lanes a row as [`FEW_KERNEL`], but a lane takes 32 of k at a time (the reach of one scale, so a
/// scale is read once for 32 weights where once a word), x four at a load, and an fp8 byte or an e2m1 nibble is a read
/// of the workgroup's table of its values (made by its threads, one each) where it was decoded by its bits with a
/// branch a weight; its one sum is a register. The row of x is at `p[2].y` of the call's inputs and the sums go to
/// `p[2].z` of its results (both in f32).
fn one_shader(kind: Kind) -> String {
    const X: &str = "@group(0) @binding(1) var<storage, read> x: array<f32>;";
    assert_eq!(COMMON.matches(X).count(), 1, "the kernels' input binding");
    let common = COMMON.replace(X, "@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;");
    let lanes = ["x", "y", "z", "w"];
    // (the workgroup's table's entries, a thread's part in making it, words a 32 of k, a block's sum, its scale)
    let (entries, fill, words, sum, scale): (usize, &str, usize, String, &str) = match kind {
        Kind::Fp8 => (
            256,
            "    lut[li] = fp8(li);\n",
            8,
            (0..8)
                .map(|q| {
                    let bytes: String = (0..4)
                        .map(|j| {
                            let byte = if j == 3 { "w >> 24u".to_string() } else { format!("(w >> {}u) & 255u", 8 * j) };
                            format!(" d += lut[{byte}] * v.{};", lanes[j])
                        })
                        .collect();
                    format!("            {{ let w = wbuf[at + {q}u]; let v = x4[xb + {q}u];{bytes} }}\n")
                })
                .collect(),
            "bitcast<f32>(wbuf[soff + (local / 32u) * kb + b])",
        ),
        Kind::Bf16 => (
            1,
            "",
            16,
            (0..8)
                .map(|q| {
                    format!(
                        "            {{ let w0 = wbuf[at + {}u]; let w1 = wbuf[at + {}u]; let v = x4[xb + {q}u]; d += bitcast<f32>(w0 << 16u) * v.x; d += bitcast<f32>(w0 & 0xffff0000u) * v.y; d += bitcast<f32>(w1 << 16u) * v.z; d += bitcast<f32>(w1 & 0xffff0000u) * v.w; }}\n",
                        2 * q,
                        2 * q + 1
                    )
                })
                .collect(),
            "1.0",
        ),
        Kind::Mxfp4 | Kind::Record => (
            16,
            "    if (li < 16u) { lut[li] = fp4(li); }\n",
            4,
            (0..4)
                .map(|q| {
                    let nibbles: String = (0..8)
                        .map(|j| {
                            let nibble = if j == 7 { "w >> 28u".to_string() } else { format!("(w >> {}u) & 15u", 4 * j) };
                            format!(" d += lut[{nibble}] * {}.{};", if j < 4 { "va" } else { "vb" }, lanes[j % 4])
                        })
                        .collect();
                    format!("            {{ let w = wbuf[at + {q}u]; let va = x4[xb + {}u]; let vb = x4[xb + {}u];{nibbles} }}\n", 2 * q, 2 * q + 1)
                })
                .collect(),
            if kind == Kind::Mxfp4 { "bitcast<f32>(wbuf[soff + local * kb + b])" } else { "e8m0((wbuf[(soff + local * kb + b) / 4u] >> (8u * ((soff + local * kb + b) % 4u))) & 255u)" },
        ),
    };
    format!(
        r#"{common}
var<workgroup> lut: array<f32, {entries}>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    let k = p[0].x; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z;
    let woff = p[2].x;
    let xo = p[2].y / 4u;
    let yo = p[2].z;
    let lane = li & 31u;
    let slot = li >> 5u;
{fill}    workgroupBarrier();
    // The rows this buffer and the call share: [max(first, lo), min(first + rows, lo + count)).
    let start = max(first, lo);
    let row = start + wg.x * 8u + slot;
    let live = row < min(first + rows, lo + count);
    var acc = 0.0;
    if (live) {{
        let local = row - first;
        let kb = k / 32u;
        let wrow = woff + local * kb * {words}u;
        for (var b = lane; b < kb; b += 32u) {{
            let at = wrow + b * {words}u;
            let xb = xo + b * 8u;
            var d = 0.0;
{sum}            acc += d * {scale};
        }}
    }}
    part[li] = acc;
    workgroupBarrier();
    if (live && lane == 0u) {{
        var total = 0.0;
        for (var l = 0u; l < 32u; l++) {{ total += part[slot * 32u + l]; }}
        y[yo + row - lo] = total;
    }}
}}
"#
    )
}

/// The pipeline name of [`one_shader`]'s kernel for `kind`.
fn one_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Fp8 => "dense-one-fp8",
        Kind::Bf16 => "dense-one-bf16",
        Kind::Mxfp4 => "dense-one-mxfp4",
        Kind::Record => "dense-one-record",
    }
}

/// OAIY_DENSE_FEW: one row of x through [`FEW_KERNEL`] and a call's own buffers, as two to eight rows are (to compare
/// [`Arena`]'s way with).
fn few_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_DENSE_FEW").is_some())
}

/// What a decode step's dense calls share on a device, made at the first and kept: the call's inputs one after another
/// in one buffer (one write), its dispatches' parameters in another (one write, each dispatch's at an offset its bind
/// group takes), its results in a third and their read-back, and each weight buffer's bind group with those. A call
/// made its inputs' buffers, a result and a parameter buffer and a bind group for every weight, and a read-back, wrote
/// each on its own, and let them all go after: in a reply that was 300 us a call beyond the GPU's 50 (a quarter of a
/// decode step's time over its 380 calls), where the same call alone in a test took 80.
pub(crate) struct Arena {
    pipeline_layout: wgpu::PipelineLayout,
    layout: wgpu::BindGroupLayout,
    x: wgpu::Buffer,
    y: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    /// bytes from one dispatch's parameters to the next's (the device's alignment of a uniform's offset)
    step: u32,
    groups: std::collections::HashMap<wgpu::Buffer, wgpu::BindGroup>,
    /// The stages' bind group ([`STAGE`]): the results read, the inputs written.
    stage: Option<wgpu::BindGroup>,
}

/// The most a call's inputs, and its results, may hold to go through the [`Arena`] (bytes), and its most dispatches.
const ARENA_BYTES: u64 = 4 << 20;
const ARENA_DISPATCHES: usize = 256;
/// A dispatch's parameters (the kernels' `p`).
const PARAM_BYTES: u64 = 48;

impl Arena {
    fn new(gpu: &Gpu) -> Arena {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oaiy-dense-arena"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: std::num::NonZeroU64::new(PARAM_BYTES) }),
            ],
        });
        let pipeline_layout =
            gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("oaiy-dense-arena"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let step = gpu.limits.min_uniform_buffer_offset_alignment.max(PARAM_BYTES as u32).next_multiple_of(16);
        let buffer = |label, size, usage| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        Arena {
            pipeline_layout,
            layout,
            x: buffer("oaiy-dense-arena-x", ARENA_BYTES, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            y: buffer("oaiy-dense-arena-y", ARENA_BYTES, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            staging: buffer("oaiy-dense-arena-read", ARENA_BYTES, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            params: buffer("oaiy-dense-arena-params", ARENA_DISPATCHES as u64 * step as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            step,
            groups: std::collections::HashMap::new(),
            stage: None,
        }
    }

    /// The stages' bind group, made the first time it is asked for.
    fn stage(&mut self, gpu: &Gpu) -> &wgpu::BindGroup {
        let Arena { layout, x, y, params, stage, .. } = self;
        stage.get_or_insert_with(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-dense-arena-stage"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gpu.dummy().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: params, offset: 0, size: std::num::NonZeroU64::new(PARAM_BYTES) }) },
                ],
            })
        })
    }

    /// `weights`' bind group, made the first time it is asked for.
    fn group(&mut self, gpu: &Gpu, weights: &wgpu::Buffer) -> &wgpu::BindGroup {
        let Arena { layout, x, y, params, groups, .. } = self;
        groups.entry(weights.clone()).or_insert_with(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-dense-arena"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: weights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: params, offset: 0, size: std::num::NonZeroU64::new(PARAM_BYTES) }) },
                ],
            })
        })
    }
}

/// The arena's bind groups of `buffers` let go (with the weights they are of: a kept group keeps its buffers).
fn forget(gpu: &Gpu, buffers: &mut dyn Iterator<Item = &wgpu::Buffer>) {
    if let Some(arena) = gpu.dense_arena.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        for buffer in buffers {
            arena.groups.remove(buffer);
        }
    }
}

/// An expert of a decode step for [`forward_units`]: its gate and up projections of `x` (one row, as they take it),
/// and the down projection of their SwiGLU times `weight` (1 where it has none).
pub struct Unit<'a> {
    pub gate: &'a DenseGpu,
    pub up: &'a DenseGpu,
    pub down: &'a DenseGpu,
    pub x: &'a [f32],
    pub weight: f32,
}

/// A projection of other projections' results for [`forward_chained`]: `first`'s sums of one row each (`(weight, x,
/// rows)`), rounded to bf16 and laid end to end, are the row `then` multiplies, quantized to fp8 first where
/// `quantized` (as an fp8 weight takes its activation).
pub struct Chain<'a> {
    pub first: &'a [(&'a DenseGpu, &'a [f32], Range<usize>)],
    pub then: &'a DenseGpu,
    pub quantized: bool,
}

/// What the stages between a call's projections share ([`swiglu_stage`], [`round_stage`]): a workgroup 32 of the row
/// (the reach of one fp8 scale), and the reference's roundings, made of the values' bits (integer steps, products by
/// powers of two) and so the host's to the bit: to bf16, and an activation's quantization to fp8 (a power-of-two
/// scale from the 32's largest, each value rounded to nearest even on the e4m3 grid).
const STAGE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 3>;
var<workgroup> h: array<f32, 32>;

// to bf16 and back: round to nearest even at bit 16
fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

// to the e4m3 grid, |v| <= 448: its step 2^-9 below 2^-6 and 2^(e - 3) in the binade of 2^e
fn e4m3(v: f32) -> f32 {
    let a = abs(v);
    var q = 448.0;
    if (a < 0.015625) {
        q = round(a * 512.0) * 0.001953125;
    } else if (a < 448.0) {
        let e = (bitcast<u32>(a) >> 23u) & 0xffu;
        q = round(a * bitcast<f32>((257u - e) << 23u)) * bitcast<f32>((e - 3u) << 23u);
    }
    return select(q, -q, (bitcast<u32>(v) >> 31u) == 1u);
}

// `v`, one of the workgroup's 32 in `h`, quantized to fp8 with them: the scale 2^ceil(log2(most / 448)), most at
// least 1e-4 (its exponent, one more unless its mantissa is zero)
fn fp8_of(v: f32) -> f32 {
    var most = 0.0;
    for (var l = 0u; l < 32u; l++) { most = max(most, abs(h[l])); }
    let b = bitcast<u32>(max(most, 0.0001) * 0.002232142857142857);
    let e = ((b >> 23u) & 0xffu) + select(0u, 1u, (b & 0x7fffffu) != 0u);
    return e4m3(clamp(v * bitcast<f32>((254u - e) << 23u), -448.0, 448.0)) * bitcast<f32>(e << 23u);
}
"#;

/// The stage between a unit's projections ([`forward_units`]): the gate's and the up's sums rounded to bf16 and
/// clamped at the limit, `silu(gate) * up * weight` rounded to bf16, and that quantized to fp8 ([`STAGE`]), which is
/// what the down projection multiplies. The exponential and the division are the device's own, so a result now and
/// then is a bf16 step from the host's. `p[0]`: where the gate's sums start in the results, the up's, where the
/// activation goes in the inputs; `p[1]`: the weight's bits, the limit's (0: none).
fn swiglu_stage() -> String {
    format!(
        "{STAGE}{}",
        r#"
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = wg.x * 32u + li;
    let weight = bitcast<f32>(p[1].x);
    let limit = bitcast<f32>(p[1].y);
    var g = bf16(src[p[0].x + i]);
    var u = bf16(src[p[0].y + i]);
    if (limit > 0.0) {
        u = clamp(u, -limit, limit);
        g = min(g, limit);
    }
    let v = bf16(g / (1.0 + exp(-g)) * u * weight);
    h[li] = v;
    workgroupBarrier();
    dst[p[0].z + i] = fp8_of(v);
}
"#
    )
}

/// The stage between a chain's projections ([`forward_chained`]): the first's sums rounded to bf16, and quantized to
/// fp8 where the last takes its activation so ([`STAGE`]); every step of it the host's to the bit. `p[0]`: where the
/// sums start in the results, (nothing), where the row goes in the inputs; `p[1].x`: 1 where it is quantized.
fn round_stage() -> String {
    format!(
        "{STAGE}{}",
        r#"
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = wg.x * 32u + li;
    let v = bf16(src[p[0].x + i]);
    h[li] = v;
    workgroupBarrier();
    let q = fp8_of(v);
    dst[p[0].z + i] = select(v, q, p[1].x == 1u);
}
"#
    )
}

/// A decode step's experts on one device in one submit and one read back: each unit's gate and up projections of its
/// row, their SwiGLU quantized on the device ([`swiglu_stage`]) and its down projection (its sums, f32), and with them
/// the sums of `others` (plain weights against one row each, `(weight, x, rows)`). None when it does not go through
/// the device's [`Arena`] (past its sizes, or OAIY_DENSE_FEW): the caller then makes the projections in calls of their
/// own with the SwiGLU on the host between them, which was two round trips an expert's step and is what a prompt's
/// rows still take.
pub fn forward_units(units: &[Unit<'_>], others: &[(&DenseGpu, &[f32], Range<usize>)], limit: f32) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let (downs, _, sums) = begin_units(units, others, limit)?.finish();
    Some((downs, sums))
}

/// [`forward_units`] begun: the call submitted, its results [`Pending`] until they are asked for. What the caller
/// does meanwhile runs beside the device's work: a decode step's experts on the host's cores beside a card's, with no
/// thread started for the card.
pub fn begin_units(units: &[Unit<'_>], others: &[(&DenseGpu, &[f32], Range<usize>)], limit: f32) -> Option<Pending> {
    let first = units.first().map(|u| u.gate).or(others.first().map(|o| o.0))?;
    let gpu = &first.gpu;
    assert!(units.iter().flat_map(|u| [u.gate, u.up, u.down]).chain(others.iter().map(|o| o.0)).all(|w| Arc::ptr_eq(&w.gpu, gpu)), "dense: a call on more than one GPU");
    let _one = first.serial.lock().unwrap_or_else(|p| p.into_inner());
    arena_begin(gpu, others, units, &[], limit)
}

/// A chain's last projection's sums (f32), all of it in one submit and one read back: the first projections, their
/// row rounded and quantized on the device ([`round_stage`]), and the last. Two calls with the rounding on the host
/// between them give the same to the bit. None when it does not go through the device's [`Arena`] (the row not the
/// last's width or not whole 32s, past the arena's sizes, or OAIY_DENSE_FEW): the caller then makes those two calls.
pub fn forward_chained(chain: &Chain<'_>) -> Option<Vec<f32>> {
    let gpu = &chain.then.gpu;
    assert!(chain.first.iter().all(|(w, ..)| Arc::ptr_eq(&w.gpu, gpu)), "dense: a call on more than one GPU");
    let _one = chain.then.serial.lock().unwrap_or_else(|p| p.into_inner());
    arena_begin(gpu, &[], &[], std::slice::from_ref(chain), 0.0)?.finish().1.pop()
}

/// [`forward_batch`] for a call whose every item is one row of x, through the device's [`Arena`]: None when it is not
/// such a call or is past the arena's sizes (the caller then makes its own buffers).
fn forward_in_arena(gpu: &Arc<Gpu>, items: &[(&DenseGpu, &[f32], usize, Range<usize>)]) -> Option<Vec<Vec<f32>>> {
    if items.iter().any(|(_, _, t, rows)| *t != 1 || rows.is_empty()) {
        return None;
    }
    let plain: Vec<(&DenseGpu, &[f32], Range<usize>)> = items.iter().map(|(w, x, _, rows)| (*w, *x, rows.clone())).collect();
    arena_begin(gpu, &plain, &[], &[], 0.0).map(|pending| pending.finish().2)
}

/// A dispatch of an [`arena_begin`]: the weight's buffer it reads, its kernel's kind and grid, and its parameters.
struct Step<'a> {
    buffer: &'a wgpu::Buffer,
    kind: Kind,
    grid: u32,
    params: [u32; 12],
}

/// `w`'s dispatches for rows `rows` of it against the row at `xo` of the inputs, its sums to `yo` of the results.
fn steps_of<'a>(w: &'a DenseGpu, rows: &Range<usize>, xo: u32, yo: u32, steps: &mut Vec<Step<'a>>) {
    assert!(rows.end <= w.n, "dense: rows past the weight");
    for (buffer, first, n_rows, soff, woff) in &w.chunks {
        let (a, b) = ((*first as usize).max(rows.start), (*first as usize + *n_rows as usize).min(rows.end));
        if a < b {
            steps.push(Step {
                buffer,
                kind: w.kind,
                grid: (b - a).div_ceil(8) as u32,
                params: [w.k as u32, 1, *first, *n_rows, rows.start as u32, rows.len() as u32, *soff, w.kind as u32, *woff, xo, yo, 0],
            });
        }
    }
}

/// An [`Arena`]'s call submitted and not yet read back ([`begin_units`]). The arena is the call's own until then:
/// another call on the device meanwhile makes itself one.
pub struct Pending {
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the bytes to read back
    bytes: u64,
    /// how many sums each result holds: the units' down projections', the chains' last projections', the plain weights'
    lens: [Vec<usize>; 3],
}

impl Pending {
    /// The call's results, waited for: each unit's down sums, each chain's last projection's, the plain weights'.
    pub fn finish(self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let Pending { gpu, arena, bytes, lens } = self;
        let waiting = std::time::Instant::now();
        let raw = gpu.map_read_soon(&arena.staging, bytes);
        crate::profile::add(&crate::profile::DENSE_WAIT, waiting);
        let mut at = 0usize;
        let mut take = |n: &usize| -> Vec<f32> {
            let v = raw[at..at + n * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            at += n * 4;
            v
        };
        let [downs, thens, sums] = lens.map(|of| of.iter().map(&mut take).collect::<Vec<Vec<f32>>>());
        // the arena back for the next call (let go if one came back before it)
        gpu.dense_arena.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert(arena);
        (downs, thens, sums)
    }
}

/// One submit through the device's [`Arena`], left [`Pending`]: `plain` weights against one row each, `units`
/// ([`forward_units`]) and `chains` ([`forward_chained`]). The inputs go one after another (each once, however many
/// weights take it), then each unit's activation and each chain's row, which the stages write; the results are each
/// unit's gate and up sums and each chain's first sums, then what is read back: the units' down sums, the chains'
/// last projections' and `plain`'s. Three passes: the gate, up, first and plain projections; the stages; the down and
/// last projections. None when it does not go through an arena.
fn arena_begin(gpu: &Arc<Gpu>, plain: &[(&DenseGpu, &[f32], Range<usize>)], units: &[Unit<'_>], chains: &[Chain<'_>], limit: f32) -> Option<Pending> {
    if few_only() || plain.iter().any(|(_, _, rows)| rows.is_empty()) || chains.iter().any(|c| c.first.iter().any(|(_, _, rows)| rows.is_empty())) {
        return None;
    }
    let making = std::time::Instant::now();
    let mut inputs: Vec<((*const f32, usize), u32)> = Vec::new();
    let mut xs: Vec<u8> = Vec::new();
    let mut place = |x: &[f32], k: usize| -> u32 {
        assert_eq!(x.len(), k, "dense: the input is not [1, k]");
        if let Some((_, at)) = inputs.iter().find(|(key, _)| *key == (x.as_ptr(), x.len())) {
            return *at;
        }
        let at = (xs.len() / 4) as u32;
        inputs.push(((x.as_ptr(), x.len()), at));
        xs.extend(x.iter().flat_map(|v| v.to_le_bytes()));
        at
    };
    let unit_inputs: Vec<u32> = units
        .iter()
        .map(|u| {
            assert!(u.gate.k == u.up.k && u.gate.n == u.up.n && u.gate.n == u.down.k && u.gate.n % 32 == 0, "dense: a unit's projections do not fit each other");
            place(u.x, u.gate.k)
        })
        .collect();
    let chain_inputs: Vec<Vec<u32>> = chains.iter().map(|c| c.first.iter().map(|(w, x, _)| place(x, w.k)).collect()).collect();
    let plain_inputs: Vec<u32> = plain.iter().map(|(w, x, _)| place(x, w.k)).collect();
    let given = (xs.len() / 4) as u32;
    // the inputs' buffer after what is given: each unit's activation and each chain's row; the results: each unit's
    // gate and up sums and each chain's first sums, then (read back) each unit's down sums, each chain's last
    // projection's and the plain weights' sums
    let width: u32 = units.iter().map(|u| u.gate.n as u32).sum();
    let widths: Vec<u32> = chains.iter().map(|c| c.first.iter().map(|(_, _, rows)| rows.len() as u32).sum()).collect();
    if chains.iter().zip(&widths).any(|(c, &w)| w as usize != c.then.k || w % 32 != 0) {
        return None;
    }
    let kept = 2 * width + widths.iter().sum::<u32>();
    // (a stage: whether it is a unit's SwiGLU, else a chain's rounding, and its parameters)
    let (mut first, mut stage, mut last): (Vec<Step<'_>>, Vec<(bool, [u32; 12])>, Vec<Step<'_>>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut act, mut sums, mut out) = (given, 0u32, kept);
    for (u, &xo) in units.iter().zip(&unit_inputs) {
        let inter = u.gate.n as u32;
        steps_of(u.gate, &(0..u.gate.n), xo, sums, &mut first);
        steps_of(u.up, &(0..u.up.n), xo, sums + inter, &mut first);
        stage.push((true, [sums, sums + inter, act, inter, u.weight.to_bits(), limit.to_bits(), 0, 0, 0, 0, 0, 0]));
        steps_of(u.down, &(0..u.down.n), act, out, &mut last);
        act += inter;
        sums += 2 * inter;
        out += u.down.n as u32;
    }
    for ((c, ins), &row) in chains.iter().zip(&chain_inputs).zip(&widths) {
        let from = sums;
        for ((w, _, rows), &xo) in c.first.iter().zip(ins) {
            steps_of(w, rows, xo, sums, &mut first);
            sums += rows.len() as u32;
        }
        stage.push((false, [from, 0, act, row, c.quantized as u32, 0, 0, 0, 0, 0, 0, 0]));
        steps_of(c.then, &(0..c.then.n), act, out, &mut last);
        act += row;
        out += c.then.n as u32;
    }
    for ((w, _, rows), &xo) in plain.iter().zip(&plain_inputs) {
        steps_of(w, rows, xo, out, &mut first);
        out += rows.len() as u32;
    }
    let dispatches = first.len() + stage.len() + last.len();
    if act as u64 * 4 > ARENA_BYTES || out as u64 * 4 > ARENA_BYTES || dispatches > ARENA_DISPATCHES || first.is_empty() {
        return None;
    }
    // (the device's arena taken for this call, or one made: it comes back when the call is read)
    let mut arena = gpu.dense_arena.lock().unwrap_or_else(|p| p.into_inner()).take().unwrap_or_else(|| Arena::new(gpu));
    let step = arena.step as usize;
    let mut table = vec![0u8; dispatches * step];
    for (i, p) in first.iter().map(|s| &s.params).chain(stage.iter().map(|(_, p)| p)).chain(last.iter().map(|s| &s.params)).enumerate() {
        for (j, v) in p.iter().enumerate() {
            table[i * step + 4 * j..i * step + 4 * j + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    gpu.queue().write_buffer(&arena.x, 0, &xs);
    gpu.queue().write_buffer(&arena.params, 0, &table);
    // (the pipelines and the groups first: a pass borrows them)
    let pipelines: Vec<Arc<wgpu::ComputePipeline>> =
        first.iter().chain(&last).map(|s| gpu.named_pipeline_in(one_name(s.kind), &arena.pipeline_layout, || one_shader(s.kind))).collect();
    let swiglu = stage.iter().any(|(unit, _)| *unit).then(|| gpu.named_pipeline_in("dense-swiglu", &arena.pipeline_layout, swiglu_stage));
    let round = stage.iter().any(|(unit, _)| !*unit).then(|| gpu.named_pipeline_in("dense-round", &arena.pipeline_layout, round_stage));
    for s in first.iter().chain(&last) {
        arena.group(gpu, s.buffer);
    }
    if !stage.is_empty() {
        arena.stage(gpu);
    }
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let project = |enc: &mut wgpu::CommandEncoder, steps: &[Step<'_>], pipelines: &[Arc<wgpu::ComputePipeline>], at: usize| {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (i, (s, pipeline)) in steps.iter().zip(pipelines).enumerate() {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &arena.groups[s.buffer], &[((at + i) * step) as u32]);
                pass.dispatch_workgroups(s.grid, 1, 1);
            }
        };
        project(&mut enc, &first, &pipelines[..first.len()], 0);
        if !stage.is_empty() {
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (i, (unit, p)) in stage.iter().enumerate() {
                    let pipeline = match unit {
                        true => swiglu.as_ref(),
                        false => round.as_ref(),
                    };
                    pass.set_pipeline(pipeline.expect("made above"));
                    pass.set_bind_group(0, arena.stage.as_ref().expect("made above"), &[((first.len() + i) * step) as u32]);
                    pass.dispatch_workgroups(p[3] / 32, 1, 1);
                }
            }
            project(&mut enc, &last, &pipelines[first.len()..], first.len() + stage.len());
        }
    }
    let bytes = (out - kept) as u64 * 4;
    enc.copy_buffer_to_buffer(&arena.y, kept as u64 * 4, &arena.staging, 0, bytes);
    crate::profile::add(&crate::profile::DENSE_MAKE, making);
    let read = units
        .iter()
        .flat_map(|u| [(u.gate, u.gate.n), (u.up, u.up.n), (u.down, u.down.n)])
        .chain(chains.iter().flat_map(|c| c.first.iter().map(|(w, _, rows)| (*w, rows.len())).chain([(c.then, c.then.n)])))
        .chain(plain.iter().map(|(w, _, rows)| (*w, rows.len())));
    for (w, rows) in read {
        crate::profile::DENSE_BYTES[0].fetch_add(w.nbytes_of(rows), Ordering::Relaxed);
        crate::profile::DENSE_BYTES[1].fetch_add(1, Ordering::Relaxed);
    }
    let submitting = std::time::Instant::now();
    gpu.queue().submit([enc.finish()]);
    crate::profile::add(&crate::profile::DENSE_SUBMIT, submitting);
    let lens = [units.iter().map(|u| u.down.n).collect(), chains.iter().map(|c| c.then.n).collect(), plain.iter().map(|(_, _, rows)| rows.len()).collect()];
    Some(Pending { gpu: Arc::clone(gpu), arena, bytes, lens })
}

/// A dense weight on the GPU.
pub struct DenseGpu {
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
    /// `(buffer, first row, rows, where its scales start, where its weights start)` (see the kernels' `p`); each
    /// buffer whole 32-row tiles.
    chunks: Vec<(wgpu::Buffer, u32, u32, u32, u32)>,
    n: usize,
    k: usize,
    kind: Kind,
    nbytes: u64,
    used: Arc<AtomicU64>,
}

impl std::fmt::Debug for DenseGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DenseGpu({:?} [{}, {}], {} buffers)", self.kind, self.n, self.k, self.chunks.len())
    }
}

impl Drop for DenseGpu {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes, Ordering::Relaxed);
        // (a record's matrix is its slots' buffer, which the slots let go)
        if self.kind != Kind::Record {
            forget(&self.gpu, &mut self.chunks.iter().map(|c| &c.0));
        }
    }
}

/// One weight's part of a submit: its output buffer and size, read back after.
struct Pass {
    y: wgpu::Buffer,
    size: u64,
}

impl DenseGpu {
    pub fn n(&self) -> usize {
        self.n
    }

    /// The bytes `rows` of its rows hold (for the profile's count of what a call read).
    fn nbytes_of(&self, rows: usize) -> u64 {
        let per = match self.kind {
            Kind::Fp8 => 2,
            Kind::Bf16 => 4,
            Kind::Mxfp4 | Kind::Record => 1,
        };
        (rows * self.k * per / 2) as u64
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Whether `other` is on the same adapter, so the two can go in one [`forward_batch`].
    pub fn same_device(&self, other: &DenseGpu) -> bool {
        Arc::ptr_eq(&self.gpu, &other.gpu)
    }

    /// `[t, rows.len()]`: the sums of `x` (`[t, k]`, as given) against rows `rows` of the weight, in f32.
    pub fn forward(&self, x: &[f32], t: usize, rows: Range<usize>) -> Vec<f32> {
        forward_batch(&[(self, x, t, rows)]).pop().expect("one weight, one result")
    }

    /// Record the dispatches for `xbuf` (`[t, k]`, uploaded) against rows `rows` into `pass`, writing a new output
    /// buffer.
    fn record(&self, enc: &mut wgpu::CommandEncoder, xbuf: &wgpu::Buffer, t: usize, rows: Range<usize>) -> Pass {
        let (k, count) = (self.k, rows.len());
        assert!(rows.end <= self.n, "dense: rows past the weight");
        let gpu = &self.gpu;
        let many = t > FEW;
        let pipeline = gpu.named_pipeline(if many { "dense-many" } else { "dense-few" }, || shader(many));
        let size = (t * count * 4).max(4) as u64;
        let ybuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-dense-y"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut groups = Vec::new();
        for (buffer, first, n_rows, soff, woff) in &self.chunks {
            let (a, b) = ((*first as usize).max(rows.start), (*first as usize + *n_rows as usize).min(rows.end));
            if a >= b {
                continue;
            }
            let params: Vec<u8> = [k as u32, t as u32, *first, *n_rows, rows.start as u32, count as u32, *soff, self.kind as u32, *woff, 0, 0, 0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-dense-params"),
                size: params.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue().write_buffer(&ubuf, 0, &params);
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-dense"),
                layout: &gpu.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ybuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
                ],
            });
            let grid = if many { ((b - a).div_ceil(64) as u32, t.div_ceil(64) as u32) } else { ((b - a).div_ceil(8) as u32, 1) };
            groups.push((group, grid));
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (group, (gx, gy)) in &groups {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        Pass { y: ybuf, size }
    }
}

/// [`DenseGpu::forward`] for several weights of one GPU in one submit and one read back: each `(weight, x, t, rows)`
/// gives its `[t, rows.len()]`, in order. A layer's experts go this way, a round trip a stage rather than a matrix.
pub fn forward_batch(items: &[(&DenseGpu, &[f32], usize, Range<usize>)]) -> Vec<Vec<f32>> {
    let Some(first) = items.first() else { return Vec::new() };
    let gpu = &first.0.gpu;
    assert!(items.iter().all(|(w, ..)| Arc::ptr_eq(&w.gpu, gpu)), "dense: a batch on more than one GPU");
    let _one = first.0.serial.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(out) = forward_in_arena(gpu, items) {
        return out;
    }
    let making = std::time::Instant::now();
    // Each input once, however many weights take it (an expert's gate and up take the same).
    let mut inputs: Vec<((*const f32, usize), wgpu::Buffer)> = Vec::new();
    for (w, x, t, rows) in items {
        assert_eq!(x.len(), t * w.k, "dense: the input is not [t, k]");
        if *t > 0 && !rows.is_empty() && !inputs.iter().any(|(key, _)| *key == (x.as_ptr(), x.len())) {
            inputs.push(((x.as_ptr(), x.len()), upload_f32(gpu, x)));
        }
    }
    let input = |x: &[f32]| &inputs.iter().find(|(key, _)| *key == (x.as_ptr(), x.len())).expect("every input uploaded").1;
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let passes: Vec<Option<Pass>> =
        items.iter().map(|(w, x, t, rows)| (*t > 0 && !rows.is_empty()).then(|| w.record(&mut enc, input(x), *t, rows.clone()))).collect();
    let total: u64 = passes.iter().flatten().map(|p| p.size).sum();
    if total == 0 {
        return items.iter().map(|_| Vec::new()).collect();
    }
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-dense-read"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut at = 0;
    for p in passes.iter().flatten() {
        enc.copy_buffer_to_buffer(&p.y, 0, &staging, at, p.size);
        at += p.size;
    }
    crate::profile::add(&crate::profile::DENSE_MAKE, making);
    for (w, _, t, rows) in items {
        if *t > 0 && !rows.is_empty() {
            crate::profile::DENSE_BYTES[0].fetch_add(w.nbytes_of(rows.len()), Ordering::Relaxed);
            crate::profile::DENSE_BYTES[1].fetch_add(1, Ordering::Relaxed);
        }
    }
    let submitting = std::time::Instant::now();
    gpu.queue().submit([enc.finish()]);
    crate::profile::add(&crate::profile::DENSE_SUBMIT, submitting);
    let waiting = std::time::Instant::now();
    let raw = gpu.map_read(&staging, total);
    crate::profile::add(&crate::profile::DENSE_WAIT, waiting);
    let mut at = 0usize;
    items
        .iter()
        .zip(&passes)
        .map(|((_, _, t, rows), p)| {
            let Some(p) = p else { return Vec::new() };
            let len = t * rows.len();
            let out = raw[at..at + len * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            at += p.size as usize;
            out
        })
        .collect()
}

/// `x` in a new storage buffer.
fn upload_f32(gpu: &Gpu, x: &[f32]) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity(x.len() * 4);
    for v in x {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-dense-x"),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue().write_buffer(&buf, 0, &bytes);
    buf
}

/// Routed experts' records on the GPU, a slot a record, made once and written call after call: a prompt's busy experts
/// pass through them a group at a time. A record goes up as stored, one write, and its MXFP4 matrices are read in place
/// with their e8m0 scales, so nothing is converted on the host. The slots count against the weight budget while they
/// live.
pub struct RecordSlots {
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
    slots: Vec<wgpu::Buffer>,
    record_bytes: usize,
    nbytes: u64,
    used: Arc<AtomicU64>,
}

impl Drop for RecordSlots {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes, Ordering::Relaxed);
        forget(&self.gpu, &mut self.slots.iter());
    }
}

impl RecordSlots {
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// `record` into slot `i`, there before the matmuls of the next call: written, submitted and waited for. A
    /// write's staging memory is let go when its submission has run, so each record's is then the memory the one
    /// before had; queued one behind another with no wait, as they were, every record's staging was new memory,
    /// which the system hands over zeroed. `measure_the_upload_rate`, 32 records (604 MB), an RTX 5090 on eight
    /// lanes: 6.4 ms a record queued together (2.9 GB a second), 0.93 ms each waited for (20 GB a second; on four
    /// lanes 5.2 and 1.8 ms). One core's copy, as `write_buffer` makes it: a record copied on every core
    /// ([`Gpu::write`]) was 1.26 ms, its threads' start more than the copy they shared.
    pub fn write(&self, i: usize, record: &[u8]) {
        assert_eq!(record.len(), self.record_bytes, "dense: a record of another size");
        self.gpu.queue().write_buffer(&self.slots[i], 0, record);
        self.gpu.queue().submit([]);
        self.gpu.wait(None);
    }

    /// Slot `i`'s MXFP4 matrix `[n, k]`: its nibbles from byte `w` (`[n, k]`, two to a byte, low first), its e8m0 scales
    /// from byte `s` (`[n, k/32]`). It reads whatever the slot holds when it is used.
    pub fn mxfp4(&self, i: usize, w: usize, s: usize, n: usize, k: usize) -> DenseGpu {
        assert!(k % 32 == 0 && w % 4 == 0, "dense: a record's matrix off its words");
        assert!(w + n * k / 2 <= self.record_bytes && s + n * (k / 32) <= self.record_bytes, "dense: a matrix past its record");
        DenseGpu {
            gpu: Arc::clone(&self.gpu),
            serial: Arc::clone(&self.serial),
            chunks: vec![(self.slots[i].clone(), 0, n as u32, s as u32, (w / 4) as u32)],
            n,
            k,
            kind: Kind::Record,
            nbytes: 0,
            used: Arc::clone(&self.used),
        }
    }
}

impl WgpuBackend {
    /// Up to `count` slots for records of `record_bytes`, as many as the weight budget has room for; None if not one,
    /// or if a record is not whole words or is past the binding limit.
    pub fn record_slots(&self, count: usize, record_bytes: usize) -> Option<RecordSlots> {
        if record_bytes == 0 || record_bytes % 4 != 0 || record_bytes as u64 > chunk_limit(&self.gpu.limits) {
            return None;
        }
        let free = self.budget.saturating_sub(self.used.load(Ordering::Relaxed));
        let n = count.min((free / record_bytes as u64) as usize);
        if n == 0 {
            return None;
        }
        let nbytes = (n * record_bytes) as u64;
        self.used.fetch_add(nbytes, Ordering::Relaxed);
        let slots = (0..n)
            .map(|_| {
                self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("oaiy-record"),
                    size: record_bytes as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Some(RecordSlots { gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial), slots, record_bytes, nbytes, used: Arc::clone(&self.used) })
    }

    /// A dense weight on the GPU while the weight budget holds it: None beyond it (the caller keeps it on the CPU), or
    /// when `k` is not a multiple of 32 (the tiles' edge).
    pub fn dense(&self, data: DenseData) -> Result<Option<Arc<DenseGpu>>, String> {
        let (n, k) = data.nk();
        if k == 0 || n == 0 || k % 32 != 0 {
            return Ok(None);
        }
        let kb = k / 32;
        // Bytes a row of weights takes, and scale words a 32-row tile takes.
        let (kind, row_bytes, tile_scales) = match &data {
            DenseData::Fp8 { w, scales, .. } => {
                if w.len() != n * k || scales.len() != n.div_ceil(32) * kb {
                    return Err(format!("dense: fp8 [{n}, {k}] with {} bytes and {} scales", w.len(), scales.len()));
                }
                (Kind::Fp8, k, kb)
            }
            DenseData::Bf16 { w, .. } => {
                if w.len() != n * k {
                    return Err(format!("dense: bf16 [{n}, {k}] with {} values", w.len()));
                }
                (Kind::Bf16, k * 2, 0)
            }
            DenseData::Mxfp4 { w, scales, .. } => {
                if w.len() * 2 != n * k || scales.len() != n * kb {
                    return Err(format!("dense: mxfp4 [{n}, {k}] with {} bytes and {} scales", w.len(), scales.len()));
                }
                (Kind::Mxfp4, k / 2, 32 * kb)
            }
        };
        let nbytes = (n * row_bytes + n.div_ceil(32) * tile_scales * 4) as u64;
        let prev = self.used.fetch_add(nbytes, Ordering::Relaxed);
        if prev + nbytes > self.budget {
            self.used.fetch_sub(nbytes, Ordering::Relaxed);
            return Ok(None);
        }
        // Whole 32-row tiles a buffer, as many as the binding limit takes (their scales with them).
        let tile_bytes = 32 * row_bytes + tile_scales * 4;
        let per = ((chunk_limit(&self.gpu.limits) as usize / tile_bytes).max(1)) * 32;
        let gpu = &self.gpu;
        let mut chunks = Vec::new();
        let mut first = 0;
        while first < n {
            let rows = per.min(n - first);
            let mut bytes: Vec<u8> = Vec::with_capacity(rows * row_bytes + rows.div_ceil(32) * tile_scales * 4);
            let soff = match &data {
                DenseData::Fp8 { w, scales, .. } => {
                    bytes.extend_from_slice(&w[first * k..(first + rows) * k]);
                    let soff = (bytes.len() / 4) as u32;
                    for s in &scales[(first / 32) * kb..(first / 32 + rows.div_ceil(32)) * kb] {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                    soff
                }
                DenseData::Bf16 { w, .. } => {
                    bytes.extend(w[first * k..(first + rows) * k].iter().flat_map(|v| v.to_le_bytes()));
                    0
                }
                DenseData::Mxfp4 { w, scales, .. } => {
                    bytes.extend_from_slice(&w[first * k / 2..(first + rows) * k / 2]);
                    let soff = (bytes.len() / 4) as u32;
                    for s in &scales[first * kb..(first + rows) * kb] {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                    soff
                }
            };
            let (buffer, _, _) = gpu.upload_rows(&bytes, bytes.len(), 1).remove(0);
            chunks.push((buffer, first as u32, rows as u32, soff, 0));
            first += rows;
        }
        Ok(Some(Arc::new(DenseGpu { gpu: Arc::clone(gpu), serial: Arc::clone(&self.serial), chunks, n, k, kind, nbytes, used: Arc::clone(&self.used) })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's own decoding of an e4m3 byte (dsv41 formats::fp8_e4m3_to_f32), for the oracle.
    fn e4m3(b: u8) -> f32 {
        let (s, e, m) = ((b >> 7) as i32, ((b >> 3) & 15) as i32, (b & 7) as f32);
        let v = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
        if s == 1 {
            -v
        } else {
            v
        }
    }

    /// dsv41's FP4_VALUES.
    const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

    fn backend() -> Option<WgpuBackend> {
        WgpuBackend::new(Some(1 << 30)).ok()
    }

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn copy(data: &DenseData) -> DenseData {
        match data {
            DenseData::Fp8 { w, scales, n, k } => DenseData::Fp8 { w: w.clone(), scales: scales.clone(), n: *n, k: *k },
            DenseData::Bf16 { w, n, k } => DenseData::Bf16 { w: w.clone(), n: *n, k: *k },
            DenseData::Mxfp4 { w, scales, n, k } => DenseData::Mxfp4 { w: w.clone(), scales: scales.clone(), n: *n, k: *k },
        }
    }

    /// `[t, rows]` in f64: the oracle.
    fn oracle(data: &DenseData, x: &[f32], t: usize, rows: Range<usize>) -> Vec<f64> {
        let (_, k) = data.nk();
        let w = |r: usize, c: usize| -> f64 {
            match data {
                DenseData::Fp8 { w, scales, .. } => e4m3(w[r * k + c]) as f64 * scales[(r / 32) * (k / 32) + c / 32] as f64,
                DenseData::Bf16 { w, .. } => f32::from_bits((w[r * k + c] as u32) << 16) as f64,
                DenseData::Mxfp4 { w, scales, .. } => {
                    let byte = w[(r * k + c) / 2];
                    let nib = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                    E2M1[nib as usize] as f64 * scales[r * (k / 32) + c / 32] as f64
                }
            }
        };
        let mut out = Vec::new();
        for tt in 0..t {
            for r in rows.clone() {
                out.push((0..k).map(|c| x[tt * k + c] as f64 * w(r, c)).sum());
            }
        }
        out
    }

    fn fp8_data(n: usize, k: usize, seed: u64) -> DenseData {
        let mut next = rng(seed);
        // Bytes that are not NaN (0x7f / 0xff), scales 2^-8 .. 2^1.
        let w = (0..n * k).map(|_| { let b = (next() & 255) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
        let scales = (0..n.div_ceil(32) * (k / 32)).map(|_| 2f32.powi((next() % 10) as i32 - 8)).collect();
        DenseData::Fp8 { w, scales, n, k }
    }

    fn bf16_data(n: usize, k: usize, seed: u64) -> DenseData {
        let mut next = rng(seed);
        let w = (0..n * k).map(|_| ((((next() % 2000) as f32 / 1000.0) - 1.0).to_bits() >> 16) as u16).collect();
        DenseData::Bf16 { w, n, k }
    }

    fn mxfp4_data(n: usize, k: usize, seed: u64) -> DenseData {
        let mut next = rng(seed);
        let w = (0..n * k / 2).map(|_| (next() & 255) as u8).collect();
        let scales = (0..n * (k / 32)).map(|_| 2f32.powi((next() % 12) as i32 - 9)).collect();
        DenseData::Mxfp4 { w, scales, n, k }
    }

    fn input(t: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut next = rng(seed);
        (0..t * k).map(|_| ((next() % 2001) as f32 / 1000.0) - 1.0).collect()
    }

    fn close(got: &[f32], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(((*g as f64) - w).abs() <= 2e-5 * scale + 1e-6, "{what}: [{i}] {g} against {w}");
        }
    }

    #[test]
    fn every_fp8_byte_decodes_as_the_reference_decodes_it() {
        let Some(b) = backend() else { return };
        // One row of the 256 bytes (and a second tile row), scale 1: a one-hot input reads each weight back alone.
        let (n, k) = (33, 256);
        // The two NaN bytes (0x7f, 0xff) left out: a NaN weight makes every sum it is in NaN, even times zero.
        let w: Vec<u8> = (0..n * k).map(|i| (i % 256) as u8).map(|b| if b & 0x7f == 0x7f { 0 } else { b }).collect();
        let data = DenseData::Fp8 { w: w.clone(), scales: vec![1.0; 2 * (k / 32)], n, k };
        let g = b.dense(data).unwrap().expect("on the GPU");
        // Every byte but the two NaNs (0x7f, 0xff), which no checkpoint holds.
        for col in (0usize..256).filter(|c| c & 0x7f != 0x7f) {
            let mut x = vec![0.0f32; k];
            x[col] = 1.0;
            let y = g.forward(&x, 1, 0..1);
            let want = e4m3(col as u8);
            // Exactly (by value: the sum of -0 with the +0 products beside it is +0).
            assert_eq!(y[0], want, "byte {col:#04x}");
        }
    }

    #[test]
    fn every_e2m1_nibble_decodes_as_the_reference_decodes_it() {
        let Some(b) = backend() else { return };
        // One row holding the 16 nibbles twice (k = 32), scale 1.
        let w: Vec<u8> = (0..16u8).map(|i| ((2 * i) % 16) | (((2 * i + 1) % 16) << 4)).collect();
        let g = b.dense(DenseData::Mxfp4 { w, scales: vec![1.0], n: 1, k: 32 }).unwrap().expect("on the GPU");
        for col in 0..32 {
            let mut x = vec![0.0f32; 32];
            x[col] = 1.0;
            assert_eq!(g.forward(&x, 1, 0..1)[0], E2M1[col % 16], "nibble {}", col % 16);
        }
    }

    #[test]
    fn each_kind_matches_the_oracle_for_a_decode_step_and_a_prompt() {
        let Some(b) = backend() else { return };
        for (n, k) in [(96, 64), (70, 160), (300, 512)] {
            for (data, kind) in [(fp8_data(n, k, n as u64), "fp8"), (bf16_data(n, k, k as u64), "bf16"), (mxfp4_data(n, k, (n * k) as u64), "mxfp4")] {
                let want_data = copy(&data);
                let g = b.dense(data).unwrap().expect("on the GPU");
                for t in [1usize, 3, 8, 9, 70, 130] {
                    let x = input(t, k, (t * 31 + n) as u64);
                    for rows in [0..n, 5..n.min(77), n - 1..n] {
                        let got = g.forward(&x, t, rows.clone());
                        close(&got, &oracle(&want_data, &x, t, rows.clone()), &format!("{kind} [{n}, {k}] t={t} rows {rows:?}"));
                    }
                }
            }
        }
    }

    #[test]
    fn a_batch_gives_each_weight_its_own_answer_in_one_submit() {
        let Some(b) = backend() else { return };
        let datas = [mxfp4_data(64, 96, 1), mxfp4_data(96, 64, 2), fp8_data(40, 32, 3), bf16_data(33, 64, 4)];
        let wants: Vec<DenseData> = datas.iter().map(copy).collect();
        let gs: Vec<Arc<DenseGpu>> = datas.into_iter().map(|d| b.dense(d).unwrap().unwrap()).collect();
        let shapes = [(5usize, 0..64usize), (12, 10..96), (1, 0..40), (0, 0..33)];
        let xs: Vec<Vec<f32>> = gs.iter().zip(&shapes).map(|(g, (t, _))| input(*t, g.k(), *t as u64 + 7)).collect();
        let items: Vec<(&DenseGpu, &[f32], usize, Range<usize>)> = gs.iter().zip(&xs).zip(&shapes).map(|((g, x), (t, rows))| (&**g, x.as_slice(), *t, rows.clone())).collect();
        let got = forward_batch(&items);
        for (i, ((want, x), (t, rows))) in wants.iter().zip(&xs).zip(&shapes).enumerate() {
            close(&got[i], &oracle(want, x, *t, rows.clone()), &format!("item {i}"));
        }
        assert!(got[3].is_empty(), "no tokens, no rows");
    }

    #[test]
    fn a_weight_beyond_the_budget_or_with_an_odd_k_stays_with_the_caller() {
        let Ok(small) = WgpuBackend::new(Some(1000)) else { return };
        assert!(small.dense(bf16_data(64, 64, 1)).unwrap().is_none());
        assert_eq!(small.usage().0, 0);
        let Some(b) = backend() else { return };
        assert!(b.dense(DenseData::Bf16 { w: vec![0; 2 * 48], n: 2, k: 48 }).unwrap().is_none());
        let g = b.dense(bf16_data(64, 64, 2)).unwrap().unwrap();
        assert!(b.usage().0 > 0);
        drop(g);
        assert_eq!(b.usage().0, 0);
    }

    #[test]
    fn every_e8m0_scale_of_a_record_decodes_as_the_reference_decodes_it() {
        let Some(b) = backend() else { return };
        // A record of one matrix [256, 32]: every weight the nibble 2 (1.0), row r's scale the byte r. A one-hot input
        // reads each row's scale back.
        let (n, k) = (256, 32);
        let w_bytes = n * k / 2;
        let mut record = vec![0x22u8; w_bytes];
        record.extend((0..n).map(|r| r as u8));
        let slots = b.record_slots(1, record.len()).expect("room for a slot");
        slots.write(0, &record);
        let m = slots.mxfp4(0, 0, w_bytes, n, k);
        let mut x = vec![0.0f32; k];
        x[5] = 1.0;
        let y = m.forward(&x, 1, 0..n);
        // Every byte but 255, NaN, which no checkpoint holds (and which WGSL lets a GPU treat as a number).
        for e in 1..255usize {
            assert_eq!(y[e].to_bits(), (e as u32) << 23, "byte {e}");
        }
        // 2^-127 is an f32 subnormal: a GPU may flush it to zero in the product, as no checkpoint's scale comes near.
        assert!(y[0] == f32::from_bits(0x0040_0000) || y[0] == 0.0, "{}", y[0]);
    }

    #[test]
    fn a_records_matrices_read_in_place_give_what_the_same_weights_uploaded_alone_give() {
        let Some(b) = backend() else { return };
        // Two matrices in one record, at word offsets, their scales after them; the same weights as Mxfp4 with f32
        // scales give the same sums, to the bit (the kernels add in the same order).
        let (n1, k1, n2, k2) = (96usize, 64usize, 64usize, 96usize);
        let mut next = rng(11);
        let w1: Vec<u8> = (0..n1 * k1 / 2).map(|_| (next() & 255) as u8).collect();
        let w2: Vec<u8> = (0..n2 * k2 / 2).map(|_| (next() & 255) as u8).collect();
        let s1: Vec<u8> = (0..n1 * k1 / 32).map(|_| 118 + (next() % 8) as u8).collect();
        let s2: Vec<u8> = (0..n2 * k2 / 32).map(|_| 118 + (next() % 8) as u8).collect();
        let record: Vec<u8> = [w1.as_slice(), &w2, &s1, &s2].concat();
        let (o2, os1, os2) = (w1.len(), w1.len() + w2.len(), w1.len() + w2.len() + s1.len());
        let slots = b.record_slots(2, record.len()).expect("room for two slots");
        assert_eq!(slots.len(), 2);
        slots.write(1, &record);
        let f32s = |s: &[u8]| s.iter().map(|&e| f32::from_bits((e as u32) << 23)).collect::<Vec<f32>>();
        let alone1 = b.dense(DenseData::Mxfp4 { w: w1.clone(), scales: f32s(&s1), n: n1, k: k1 }).unwrap().unwrap();
        let alone2 = b.dense(DenseData::Mxfp4 { w: w2.clone(), scales: f32s(&s2), n: n2, k: k2 }).unwrap().unwrap();
        let (m1, m2) = (slots.mxfp4(1, 0, os1, n1, k1), slots.mxfp4(1, o2, os2, n2, k2));
        for t in [1usize, 9, 70] {
            let (x1, x2) = (input(t, k1, t as u64), input(t, k2, t as u64 + 1));
            let got = forward_batch(&[(&m1, &x1, t, 0..n1), (&m2, &x2, t, 3..n2), (&m1, &x1, t, 7..20)]);
            assert_eq!(got[0], alone1.forward(&x1, t, 0..n1), "t={t}");
            assert_eq!(got[1], alone2.forward(&x2, t, 3..n2), "t={t}");
            assert_eq!(got[2], alone1.forward(&x1, t, 7..20), "t={t}");
        }
        let held = b.usage().0;
        drop(slots);
        assert_eq!(b.usage().0, held - 2 * record.len() as u64, "the slots' bytes come back to the budget");
    }

    /// What one dense call costs a decode step, and what of it is the trip itself: a weight the size of an attention
    /// projection against one row, then the same trip with nothing to compute (a copy of 4 KB read back), with the
    /// read-back's buffer kept, and waited for by polling.
    #[test]
    #[ignore = "a timing; run with --nocapture"]
    fn measure_a_round_trip() {
        let Some(b) = backend() else { return };
        let gpu = &b.gpu;
        let (n, k, calls) = (1024usize, 4096usize, 400u32);
        let w = b.dense(fp8_data(n, k, 5)).unwrap().expect("within the budget");
        let x = input(1, k, 6);
        let us = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e6 / calls as f64;
        for _ in 0..20 {
            w.forward(&x, 1, 0..n);
        }
        let t = std::time::Instant::now();
        for _ in 0..calls {
            std::hint::black_box(w.forward(&x, 1, 0..n));
        }
        eprintln!("a dense call of [{n}, {k}] against one row: {:.0} us", us(t));
        let two = [(&*w, &x[..], 1usize, 0..n), (&*w, &x[..], 1usize, 0..n)];
        let t = std::time::Instant::now();
        for _ in 0..calls {
            std::hint::black_box(forward_batch(&two));
        }
        eprintln!("two of them in one call: {:.0} us", us(t));
        // what a kernel reads a second: eight of a weight in one call against one, the seven more over their bytes
        for (name, data, bytes) in [
            ("fp8", fp8_data(4096, 5120, 7), 4096.0 * 5120.0),
            ("bf16", bf16_data(4096, 5120, 8), 4096.0 * 5120.0 * 2.0),
            ("mxfp4", mxfp4_data(4096, 5120, 9), 4096.0 * 5120.0 / 2.0),
        ] {
            let big = b.dense(data).unwrap().expect("within the budget");
            let xb = input(1, 5120, 10);
            let one = [(&*big, &xb[..], 1usize, 0..4096usize)];
            let eight: Vec<(&DenseGpu, &[f32], usize, Range<usize>)> = (0..8).map(|_| one[0].clone()).collect();
            let time = |items: &[(&DenseGpu, &[f32], usize, Range<usize>)]| {
                for _ in 0..10 {
                    forward_batch(items);
                }
                let t = std::time::Instant::now();
                for _ in 0..100 {
                    std::hint::black_box(forward_batch(items));
                }
                t.elapsed().as_secs_f64() / 100.0
            };
            let (a, e) = (time(&one), time(&eight));
            eprintln!("{name} [4096, 5120]: one {:.0} us, eight in a call {:.0} us: {:.0} GB/s", a * 1e6, e * 1e6, 7.0 * bytes / (e - a) / 1e9);
            // as a decode step spaces its calls: the GPU idle between them (the host's work of a layer), by a busy
            // wait (the thread kept) or a sleep (the thread parked)
            for (how, pause_us, sleeps) in [("after 300 us of the host's work", 300u64, false), ("after 2 ms of it", 2000, false), ("after a 2 ms sleep", 2000, true)] {
                let mut spent = 0.0;
                for _ in 0..100 {
                    let pause = std::time::Instant::now();
                    if sleeps {
                        std::thread::sleep(std::time::Duration::from_micros(pause_us));
                    } else {
                        while pause.elapsed() < std::time::Duration::from_micros(pause_us) {
                            std::hint::spin_loop();
                        }
                    }
                    let t = std::time::Instant::now();
                    std::hint::black_box(forward_batch(&eight));
                    spent += t.elapsed().as_secs_f64();
                }
                eprintln!("    eight in a call {how}: {:.0} us", spent / 100.0 * 1e6);
            }
        }
        let t = std::time::Instant::now();
        for _ in 0..calls {
            std::hint::black_box(upload_f32(gpu, &x));
        }
        eprintln!("the input's buffer made and written: {:.0} us", us(t));
        gpu.queue().submit([]);
        gpu.wait(None);
        let size = 4096u64;
        let src = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let staging = || gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let t = std::time::Instant::now();
        for _ in 0..calls {
            let st = staging();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&src, 0, &st, 0, size);
            gpu.queue().submit([enc.finish()]);
            std::hint::black_box(gpu.map_read(&st, size));
        }
        eprintln!("a trip with nothing to compute (4 KB copied and read back, its buffer made each time): {:.0} us", us(t));
        let kept = staging();
        let t = std::time::Instant::now();
        for _ in 0..calls {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
            gpu.queue().submit([enc.finish()]);
            std::hint::black_box(gpu.map_read(&kept, size));
        }
        eprintln!("the same with the read-back's buffer kept: {:.0} us", us(t));
        let t = std::time::Instant::now();
        for _ in 0..calls {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
            gpu.queue().submit([enc.finish()]);
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = done.clone();
            kept.slice(..size).map_async(wgpu::MapMode::Read, move |_| flag.store(true, std::sync::atomic::Ordering::Release));
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                let _ = gpu.device.poll(wgpu::PollType::Poll);
                std::hint::spin_loop();
            }
            std::hint::black_box(kept.slice(..size).get_mapped_range().expect("mapped").to_vec());
            kept.unmap();
        }
        eprintln!("the same, waited for by polling: {:.0} us", us(t));
        let t = std::time::Instant::now();
        for _ in 0..calls {
            gpu.queue().write_buffer(&src, 0, &[0u8; 4096]);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
            gpu.queue().submit([enc.finish()]);
            std::hint::black_box(gpu.map_read(&kept, size));
        }
        eprintln!("the kept trip with 4 KB written first: {:.0} us", us(t));
        // with the device holding what DeepSeek's does (a thousand records' buffers, written once): the same calls
        let record = 18_874_368u64;
        let held: Vec<wgpu::Buffer> = (0..1000)
            .map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: record, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
            .collect();
        gpu.queue().submit([]);
        gpu.wait(None);
        for _ in 0..20 {
            w.forward(&x, 1, 0..n);
        }
        let t = std::time::Instant::now();
        for _ in 0..calls {
            std::hint::black_box(w.forward(&x, 1, 0..n));
        }
        eprintln!("a dense call of [{n}, {k}] with a thousand records' buffers on the device: {:.0} us", us(t));
        let t = std::time::Instant::now();
        for _ in 0..calls {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
            gpu.queue().submit([enc.finish()]);
            std::hint::black_box(gpu.map_read(&kept, size));
        }
        eprintln!("the kept trip then: {:.0} us", us(t));
        drop(held);
    }

    /// What an expert's record costs to put in its slot, by the way: 32 expert-sized writes (600 MB) into buffers made
    /// once, queued one behind another and waited for together (as [`RecordSlots::write`] did: `write_buffer`, one
    /// core's copy); the same each waited for; and each copied on every core and waited for, as it does now. Four
    /// rounds of each.
    #[test]
    #[ignore = "a timing; run with --nocapture"]
    fn measure_the_upload_rate() {
        let Some(b) = backend() else { return };
        let gpu = &b.gpu;
        let size = 18_874_368u64;
        let host: Vec<u8> = (0..size as usize).map(|i| (i * 31 + i / 7) as u8).collect();
        let bufs: Vec<wgpu::Buffer> = (0..32)
            .map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }))
            .collect();
        for (way, what) in ["queued together, one core's copy", "each waited for, one core's copy", "each waited for, every core's copy"].iter().enumerate() {
            for round in 0..4 {
                let t = std::time::Instant::now();
                for buf in &bufs {
                    match way {
                        0 => gpu.queue().write_buffer(buf, 0, &host),
                        1 => {
                            gpu.queue().write_buffer(buf, 0, &host);
                            gpu.queue().submit([]);
                            gpu.wait(None);
                        }
                        _ => {
                            gpu.write(buf, 0, &host);
                            gpu.queue().submit([]);
                            gpu.wait(None);
                        }
                    }
                }
                gpu.queue().submit([]);
                gpu.wait(None);
                let secs = t.elapsed().as_secs_f64();
                eprintln!("{what}, round {round}: {:.0} MB in {secs:.3} s ({:.2} GB/s, {:.2} ms a record)", 32.0 * size as f64 / 1e6, 32.0 * size as f64 / secs / 1e9, secs * 1e3 / 32.0);
            }
            assert_eq!(gpu.read(&bufs[31], size), host, "{what}: the last buffer holds the record");
        }
    }
}
