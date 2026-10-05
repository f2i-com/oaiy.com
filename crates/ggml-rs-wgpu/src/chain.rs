//! A decode step's ops chained on the adapter ([`ggml_rs::chain`]): vectors kept in buffers here, the ops recorded and
//! run in one compute pass of one submit, only what the host asks for read back. The quantized matmuls are the GGUF
//! kernels (`shaders`, one row of `x`); RMSNorm, the residual add, the SwiGLU of a fused gate-up, RoPE, a store into a
//! cache and a decode step's attention have kernels of their own here, on the same bind group layout (weights, a
//! table or a cache at 0, the input at 1, the output at 2, the parameters at 3).

use std::sync::Arc;

use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs::QuantizedTensor;

use crate::{Gpu, WgpuBackend, WgpuQuant};

const HEAD: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;
"#;

/// `y = x / sqrt(mean(x^2) + eps) * w` over `p[0].x` elements (`p[0].y` the bits of `eps`), one workgroup.
const RMSNORM: &str = r#"
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

/// `y += x` over `p[0].x` elements.
const ADD: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < p[0].x) { y[i] = y[i] + x[i]; }
}
"#;

/// `y = silu(x[..ff]) * x[ff..]`, `ff = p[0].x`.
const SILU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let ff = p[0].x;
    if (i < ff) {
        let g = x[i];
        y[i] = (g / (1.0 + exp(-g))) * x[ff + i];
    }
}
"#;

/// RoPE in place on `y` (`[p[0].x heads, p[0].y head_dim]`): pair `k` by the sine `w[2k]` and cosine `w[2k + 1]`
/// (made on the host, as the CPU's rope makes them), the pairs `(2k, 2k + 1)` or with `p[0].z` `(k, k + half)`.
const ROPE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let heads = p[0].x;
    let hd = p[0].y;
    let half = hd / 2u;
    let i = id.x;
    if (i >= heads * half) { return; }
    let h = i / half;
    let k = i % half;
    let s = bitcast<f32>(w[2u * k]);
    let c = bitcast<f32>(w[2u * k + 1u]);
    var ia = h * hd + 2u * k;
    var ib = ia + 1u;
    if (p[0].z != 0u) {
        ia = h * hd + k;
        ib = ia + half;
    }
    let a = y[ia];
    let b = y[ib];
    y[ia] = a * c - b * s;
    y[ib] = a * s + b * c;
}
"#;

/// `y[p[0].y + i] = x[i]` for `i < p[0].x`.
const STORE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < p[0].x) { y[p[0].y + i] = x[i]; }
}
"#;

/// Positions a workgroup of [`ATTENTION_PART`] takes.
const SPLIT: usize = 256;

/// One query's attention over positions split in runs of 256, a workgroup a (query head `h`, run): its scores, their
/// largest `m`, the sum `l` of their exponentials after `m`, and the exponentials' weighted values, into the scratch
/// after the output. `w` the layer's cache (row `t`: K `[n_kv, hd]` then V), `x` the query `[n_h, hd]`, `y` the output
/// `[n_h, hd]` then each (head, run)'s `hd` weighted values and then its `m` and `l`. `p[0]`: `n_h`, `n_kv`, `hd`, the
/// positions' end; `p[1]`: their start, the runs, the bits of the scale. (A workgroup a head walked every position on
/// one thread a dimension: 37 ms of a 3B Llama's decode step at 2,000 positions.)
const ATTENTION_PART: &str = r#"
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let n_kv = p[0].y;
    let hd = p[0].z;
    let hi = p[0].w;
    let lo = p[1].x;
    let runs = p[1].y;
    let scale = bitcast<f32>(p[1].z);
    let h = wg.x;
    let run = wg.y;
    let kh = h / (n_h / n_kv);
    let kvd = n_kv * hd;
    let row = 2u * kvd;
    let start = lo + run * 256u;
    let end = min(start + 256u, hi);
    let t = start + li;
    var s = -3.4e38;
    if (t < end) {
        let kb = t * row + kh * hd;
        var d0 = 0.0;
        for (var d = 0u; d < hd; d++) { d0 += x[h * hd + d] * bitcast<f32>(w[kb + d]); }
        s = d0 * scale;
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] = max(red[li], red[li + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = 0.0;
    if (t < end) { e = exp(s - m); }
    sc[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st /= 2u) {
        if (li < st) { red[li] += red[li + st]; }
        workgroupBarrier();
    }
    let l = red[0];
    let part = n_h * hd + (h * runs + run) * hd;
    let n = end - start;
    for (var d = li; d < hd; d += 256u) {
        var acc = 0.0;
        for (var i = 0u; i < n; i++) { acc += sc[i] * bitcast<f32>(w[(start + i) * row + kvd + kh * hd + d]); }
        y[part + d] = acc;
    }
    if (li == 0u) {
        let ml = n_h * hd + n_h * runs * hd + (h * runs + run) * 2u;
        y[ml] = m;
        y[ml + 1u] = l;
    }
}
"#;

/// The runs of [`ATTENTION_PART`] put together, a workgroup a head: each run's values rescaled from its largest to the
/// head's, over the sum of the rescaled sums. `p` as for the parts.
const ATTENTION_JOIN: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_h = p[0].x;
    let hd = p[0].z;
    let runs = p[1].y;
    let h = wg.x;
    let ml = n_h * hd + n_h * runs * hd + h * runs * 2u;
    var m = -3.4e38;
    for (var r = 0u; r < runs; r++) { m = max(m, y[ml + r * 2u]); }
    var l = 0.0;
    for (var r = 0u; r < runs; r++) { l += exp(y[ml + r * 2u] - m) * y[ml + r * 2u + 1u]; }
    for (var d = li; d < hd; d += 256u) {
        var acc = 0.0;
        for (var r = 0u; r < runs; r++) {
            acc += exp(y[ml + r * 2u] - m) * y[n_h * hd + (h * runs + r) * hd + d];
        }
        y[h * hd + d] = acc / l;
    }
}
"#;

/// A uniform's eight words, the rest of `words` zero.
fn words8(words: &[u32]) -> [u32; 8] {
    let mut all = [0u32; 8];
    all[..words.len()].copy_from_slice(words);
    all
}

fn buffer(v: &DeviceVec) -> &wgpu::Buffer {
    v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector")
}

fn le_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for f in data {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

impl DeviceChain for WgpuBackend {
    fn vec(&self, len: usize) -> DeviceVec {
        let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-vec"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        DeviceVec { len, inner: Arc::new(buf) }
    }

    fn upload_at(&self, v: &DeviceVec, offset: usize, data: &[f32]) {
        assert!(offset + data.len() <= v.len, "chain: {} values at {offset} into a vector of {}", data.len(), v.len);
        if !data.is_empty() {
            self.gpu.queue.write_buffer(buffer(v), (offset * 4) as u64, &le_bytes(data));
        }
    }

    fn resize(&self, v: &DeviceVec, len: usize) -> DeviceVec {
        let grown = self.vec(len);
        let keep = v.len.min(len);
        if keep > 0 {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buffer(v), 0, buffer(&grown), 0, (keep * 4) as u64);
            self.gpu.queue.submit([enc.finish()]);
        }
        grown
    }

    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize {
        n_h * head_dim + n_h * cap.div_ceil(SPLIT).max(1) * (head_dim + 2)
    }

    fn holds(&self, w: &QuantizedTensor) -> bool {
        w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).is_some_and(|q| Arc::ptr_eq(&q.gpu, &self.gpu))
    }

    fn begin(&self) -> Box<dyn ChainRecorder + '_> {
        Box::new(Recorder { backend: self, dispatches: Vec::new(), reads: Vec::new() })
    }
}

/// One dispatch: its pipeline, bind group and grid.
type Dispatch = (Arc<wgpu::ComputePipeline>, wgpu::BindGroup, (u32, u32, u32));

/// A bind group a chain makes again step after step: its pipeline, its three buffers and its parameters.
pub(crate) type GroupKey = (usize, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, [u32; 8]);

/// Bind groups kept before the cache starts over (a cache grown from buffers that were replaced).
const KEEP_GROUPS: usize = 16384;

struct Recorder<'a> {
    backend: &'a WgpuBackend,
    /// Every op's dispatch, in order, run in one compute pass (a pass an op cost more than the ops).
    dispatches: Vec<Dispatch>,
    /// What to read back once they have run: the vector, the element it starts at, a staging buffer, the length.
    reads: Vec<(wgpu::Buffer, usize, wgpu::Buffer, usize)>,
}

impl Recorder<'_> {
    fn gpu(&self) -> &Gpu {
        &self.backend.gpu
    }

    fn uniform(&self, words: &[u32]) -> wgpu::Buffer {
        let all = words8(words);
        let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu().queue.write_buffer(&buf, 0, &bytes);
        buf
    }

    /// A dispatch whose bind group is the same every step (its buffers and parameters): made once and kept.
    fn dispatch_kept(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, words: &[u32], groups: (u32, u32, u32)) {
        let key: GroupKey = (Arc::as_ptr(pipeline) as usize, at0.clone(), at1.clone(), at2.clone(), words8(words));
        let kept = self.gpu().chain_groups.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned();
        let group = match kept {
            Some(group) => group,
            None => {
                let params = self.uniform(words);
                let group = self.group(at0, at1, at2, &params);
                let mut groups = self.gpu().chain_groups.lock().unwrap_or_else(|p| p.into_inner());
                if groups.len() >= KEEP_GROUPS {
                    groups.clear();
                }
                groups.insert(key, group.clone());
                group
            }
        };
        self.dispatches.push((Arc::clone(pipeline), group, groups));
    }

    fn group(&self, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, params: &wgpu::Buffer) -> wgpu::BindGroup {
        self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oaiy-chain"),
            layout: &self.gpu().layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: at0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: at1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        })
    }

    fn dispatch(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, params: &wgpu::Buffer, groups: (u32, u32, u32)) {
        let group = self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oaiy-chain"),
            layout: &self.gpu().layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: at0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: at1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });
        self.dispatches.push((Arc::clone(pipeline), group, groups));
    }

    fn named(&self, name: &'static str, body: &'static str) -> Arc<wgpu::ComputePipeline> {
        self.gpu().named_pipeline(name, || format!("{HEAD}{body}"))
    }
}

impl ChainRecorder for Recorder<'_> {
    fn matmul(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec) {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        assert!(x.len >= k && y.len >= n, "chain: matmul [{n}, {k}] from {} into {}", x.len, y.len);
        let pipeline = self.gpu().pipeline(q.dtype, 1).expect("uploaded weights have a pipeline");
        for (chunk, row0, rows) in &q.chunks {
            // Rows beyond 65535 wrap into the second grid axis.
            let words = [k as u32, n as u32, 1, *row0, *rows, q.row_bytes as u32, 0, 0];
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, ((*rows).min(65535), rows.div_ceil(65535), 1));
        }
    }

    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32) {
        let pipeline = self.named("chain-rmsnorm", RMSNORM);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[x.len as u32, eps.to_bits()], (1, 1, 1));
    }

    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec) {
        let pipeline = self.named("chain-add", ADD);
        self.dispatch_kept(&pipeline, buffer(y), buffer(y), buffer(acc), &[acc.len as u32], ((acc.len as u32).div_ceil(256), 1, 1));
    }

    fn silu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec) {
        let pipeline = self.named("chain-silu-mul-split", SILU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[out.len as u32], ((out.len as u32).div_ceil(256), 1, 1));
    }

    fn rope(&mut self, x: &DeviceVec, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool) {
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (heads * head_dim / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, neox as u32], (pairs.div_ceil(256), 1, 1));
    }

    fn store(&mut self, src: &DeviceVec, dst: &DeviceVec, offset: usize) {
        assert!(offset + src.len <= dst.len, "chain: storing {} at {offset} into {}", src.len, dst.len);
        let pipeline = self.named("chain-store", STORE);
        let params = self.uniform(&[src.len as u32, offset as u32]);
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, ((src.len as u32).div_ceil(256), 1, 1));
    }

    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        let runs = kv_len.saturating_sub(lo).div_ceil(SPLIT).max(1);
        assert!(
            kv.len >= cap * 2 * n_kv * head_dim && kv_len <= cap && out.len >= n_h * head_dim + n_h * runs * (head_dim + 2),
            "chain: attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, kv_len as u32, lo as u32, runs as u32, scale.to_bits()]);
        let part = self.named("chain-attention-part", ATTENTION_PART);
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, runs as u32, 1));
        let join = self.named("chain-attention-join", ATTENTION_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, 1, 1));
    }

    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize) {
        assert!(offset + len <= v.len, "chain: reading {len} at {offset} of {}", v.len);
        let staging = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-read"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.reads.push((buffer(v).clone(), offset, staging, len));
    }

    fn finish(self: Box<Self>) -> Vec<Vec<f32>> {
        let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
        let start = std::time::Instant::now();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (pipeline, group, (x, y, z)) in &self.dispatches {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, *z);
            }
        }
        for (from, offset, staging, len) in &self.reads {
            if *len > 0 {
                enc.copy_buffer_to_buffer(from, (*offset * 4) as u64, staging, 0, (*len * 4) as u64);
            }
        }
        self.gpu().queue.submit([enc.finish()]);
        for (_, _, staging, len) in &self.reads {
            staging.slice(..(*len as u64 * 4).max(4)).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.gpu().device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("webgpu: device lost while waiting for a chain");
        let out = self
            .reads
            .iter()
            .map(|(_, _, staging, len)| {
                let view = staging.slice(..(*len as u64 * 4).max(4)).get_mapped_range().expect("webgpu: mapping a finished buffer");
                let v: Vec<f32> = view.chunks_exact(4).take(*len).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                drop(view);
                staging.unmap();
                v
            })
            .collect();
        crate::profile::add(&crate::profile::LINEAR_WAIT, start);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_quants::GgmlType;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2001) as f32 / 1000.0 - 1.0
        }
    }

    fn close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= 1e-4 * scale, "{what} [{i}]: {x} against {y}");
        }
    }

    /// A chain gives the CPU backend's answer: RMSNorm, a quantized matmul, the fused SwiGLU and an add, read back.
    #[test]
    fn a_chain_matches_the_cpus_ops() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (k, ff) = (256usize, 64usize);
        let mut next = rng(0x1234_5679);
        // a Q8_0 weight [2ff, k]: blocks of a scale and 32 int8s
        let mut bytes = vec![0u8; 2 * ff * (k / 32) * 34];
        for blk in bytes.chunks_mut(34) {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
            for v in blk[2..].iter_mut() {
                *v = ((next() + 1.0) * 127.0) as u8;
            }
        }
        let wq = ggml_rs::QuantizedTensor::from_bytes_cpu(bytes, vec![2 * ff, k], GgmlType::Q8_0);
        let wq = ggml_rs::Backend::to_device_quant(&b, wq);
        assert!(DeviceChain::holds(&b, &wq));
        let xv: Vec<f32> = (0..k).map(|_| next()).collect();
        let nw: Vec<f32> = (0..k).map(|_| next() * 0.5 + 1.0).collect();
        let res: Vec<f32> = (0..ff).map(|_| next()).collect();
        // CPU: rmsnorm, matmul, silu-mul split, add
        let xn = ggml_rs::Backend::rmsnorm(&cpu, &ggml_rs::Tensor::from_vec(xv.clone(), vec![1, k]), &ggml_rs::Tensor::from_vec(nw.clone(), vec![k]), 1e-5);
        let gu = ggml_rs::Backend::linear_q(&cpu, &xn, &ggml_rs::QuantizedTensor::from_bytes_cpu(wq.to_host().bytes().to_vec(), vec![2 * ff, k], GgmlType::Q8_0));
        let act = ggml_rs::Backend::silu_mul_split(&cpu, &gu, ff);
        let want: Vec<f32> = act.data().iter().zip(&res).map(|(a, r)| a + r).collect();
        // the chain
        let (x, n, xnd, gud, actd, acc) = (b.vec(k), b.vec(k), b.vec(k), b.vec(2 * ff), b.vec(ff), b.vec(ff));
        DeviceChain::upload(&b, &x, &xv);
        DeviceChain::upload(&b, &n, &nw);
        DeviceChain::upload(&b, &acc, &res);
        let mut rec = b.begin();
        rec.rmsnorm(&x, &n, &xnd, 1e-5);
        rec.matmul(&wq, &xnd, &gud);
        rec.silu_mul_split(&gud, &actd);
        rec.add(&acc, &actd);
        rec.read(&acc);
        rec.read(&xnd);
        let got = rec.finish();
        assert_eq!(got.len(), 2);
        close(&got[1], xn.data(), "rmsnorm");
        close(&got[0], &want, "the chain");
    }

    /// RoPE, a store into a cache and attention over it give the CPU backend's answer (GQA): a few positions in, and
    /// past 512 (three runs of the split attention put together).
    #[test]
    fn rope_store_and_attention_match_the_cpus() {
        for (cap, past) in [(16usize, 9usize), (640, 530)] {
            rope_store_and_attention_case(cap, past);
        }
    }

    fn rope_store_and_attention_case(cap: usize, past: usize) {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (n_h, n_kv, hd) = (8usize, 2usize, 64usize);
        let (qd, kvd) = (n_h * hd, n_kv * hd);
        let mut next = rng(77);
        // the cache's earlier rows, the new token's q, k and v
        let ks: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
        let vs: Vec<f32> = (0..cap * kvd).map(|_| next()).collect();
        let q: Vec<f32> = (0..qd).map(|_| next()).collect();
        let k: Vec<f32> = (0..kvd).map(|_| next()).collect();
        let v: Vec<f32> = (0..kvd).map(|_| next()).collect();
        let theta = 10000.0f32;
        let table: Vec<f32> = (0..hd / 2)
            .flat_map(|j| {
                let (s, c) = (past as f32 * theta.powf(-2.0 * j as f32 / hd as f32)).sin_cos();
                [s, c]
            })
            .collect();
        for neox in [false, true] {
            let rope_type = if neox { ggml_rs::RopeType::NeoX } else { ggml_rs::RopeType::Normal };
            // CPU
            let mut qc = ggml_rs::Tensor::from_vec(q.clone(), vec![1, n_h, hd]);
            let mut kc = ggml_rs::Tensor::from_vec(k.clone(), vec![1, n_kv, hd]);
            ggml_rs::Backend::rope(&cpu, &mut qc, &[past as u32], hd, rope_type, theta, None);
            ggml_rs::Backend::rope(&cpu, &mut kc, &[past as u32], hd, rope_type, theta, None);
            let mut kcache = ks.clone();
            let mut vcache = vs.clone();
            kcache[past * kvd..(past + 1) * kvd].copy_from_slice(kc.data());
            vcache[past * kvd..(past + 1) * kvd].copy_from_slice(&v);
            let want = ggml_rs::Backend::attention(
                &cpu,
                &qc,
                &ggml_rs::Tensor::from_vec(kcache, vec![cap, n_kv, hd]),
                &ggml_rs::Tensor::from_vec(vcache, vec![cap, n_kv, hd]),
                past + 1,
                0.125,
                past,
                None,
            );
            // the chain: the cache interleaved a row at a time (K then V)
            let mut interleaved = Vec::with_capacity(cap * 2 * kvd);
            for t in 0..cap {
                interleaved.extend_from_slice(&ks[t * kvd..(t + 1) * kvd]);
                interleaved.extend_from_slice(&vs[t * kvd..(t + 1) * kvd]);
            }
            let (qv, kv_, vv, tab, cache, out) = (b.vec(qd), b.vec(kvd), b.vec(kvd), b.vec(hd), b.vec(cap * 2 * kvd), b.vec(b.attention_out_len(n_h, hd, cap)));
            DeviceChain::upload(&b, &qv, &q);
            DeviceChain::upload(&b, &kv_, &k);
            DeviceChain::upload(&b, &vv, &v);
            DeviceChain::upload(&b, &tab, &table);
            DeviceChain::upload(&b, &cache, &interleaved);
            let mut rec = b.begin();
            rec.rope(&qv, n_h, hd, &tab, neox);
            rec.rope(&kv_, n_kv, hd, &tab, neox);
            rec.store(&kv_, &cache, past * 2 * kvd);
            rec.store(&vv, &cache, past * 2 * kvd + kvd);
            rec.attention(&qv, &cache, &out, n_h, n_kv, hd, 0, past + 1, cap, 0.125);
            rec.read_range(&out, 0, qd);
            rec.read_range(&cache, past * 2 * kvd, kvd);
            rec.read(&qv);
            let got = rec.finish();
            close(&got[2], qc.data(), "the rotated query");
            close(&got[1], kc.data(), "the stored key");
            close(&got[0], want.data(), "the attention");
        }
    }
}
