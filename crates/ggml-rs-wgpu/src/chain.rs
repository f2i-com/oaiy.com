//! A decode step's ops chained on the adapter ([`ggml_rs::chain`]): vectors kept in buffers here, the ops recorded
//! into one encoder and submitted once, only what the host asks for read back. The quantized matmuls are the GGUF
//! kernels (`shaders`, one row of `x`); RMSNorm, the residual add and the SwiGLU of a fused gate-up have kernels of
//! their own here, on the same bind group layout (weights or a second input at 0, the input at 1, the output at 2,
//! the parameters at 3).

use std::sync::Arc;

use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs::QuantizedTensor;

use crate::{Gpu, WgpuBackend, WgpuQuant};

const HEAD: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;
"#;

/// `y = x / sqrt(mean(x^2) + eps) * w` over `p.x` elements (`p.y` the bits of `eps`), one workgroup.
const RMSNORM: &str = r#"
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = p.x;
    var s = 0.0;
    for (var i = li; i < n; i += 256u) { let v = x[i]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (li < stride) { part[li] += part[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(n) + bitcast<f32>(p.y));
    for (var i = li; i < n; i += 256u) { y[i] = x[i] * inv * bitcast<f32>(w[i]); }
}
"#;

/// `y += x` over `p.x` elements.
const ADD: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < p.x) { y[i] = y[i] + x[i]; }
}
"#;

/// `y = silu(x[..ff]) * x[ff..]`, `ff = p.x`.
const SILU_MUL_SPLIT: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let ff = p.x;
    if (i < ff) {
        let g = x[i];
        y[i] = (g / (1.0 + exp(-g))) * x[ff + i];
    }
}
"#;

fn buffer(v: &DeviceVec) -> &wgpu::Buffer {
    v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector")
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

    fn upload(&self, v: &DeviceVec, data: &[f32]) {
        assert!(data.len() <= v.len, "chain: {} values into a vector of {}", data.len(), v.len);
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        self.gpu.queue.write_buffer(buffer(v), 0, &bytes);
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

struct Recorder<'a> {
    backend: &'a WgpuBackend,
    /// Every op's dispatch, in order, run in one compute pass (a pass an op cost more than the ops).
    dispatches: Vec<Dispatch>,
    /// What to read back once they have run: the vector, its staging buffer, and its length.
    reads: Vec<(wgpu::Buffer, wgpu::Buffer, usize)>,
}

impl Recorder<'_> {
    fn gpu(&self) -> &Gpu {
        &self.backend.gpu
    }

    fn uniform(&self, words: &[u32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu().queue.write_buffer(&buf, 0, &bytes);
        buf
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
            let params = self.uniform(&[k as u32, n as u32, 1, *row0, *rows, q.row_bytes as u32, 0, 0]);
            // Rows beyond 65535 wrap into the second grid axis.
            self.dispatch(&pipeline, chunk, buffer(x), buffer(y), &params, ((*rows).min(65535), rows.div_ceil(65535), 1));
        }
    }

    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32) {
        let pipeline = self.named("chain-rmsnorm", RMSNORM);
        let params = self.uniform(&[x.len as u32, eps.to_bits(), 0, 0]);
        self.dispatch(&pipeline, buffer(w), buffer(x), buffer(out), &params, (1, 1, 1));
    }

    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec) {
        let pipeline = self.named("chain-add", ADD);
        let params = self.uniform(&[acc.len as u32, 0, 0, 0]);
        self.dispatch(&pipeline, buffer(y), buffer(y), buffer(acc), &params, ((acc.len as u32).div_ceil(256), 1, 1));
    }

    fn silu_mul_split(&mut self, fused: &DeviceVec, out: &DeviceVec) {
        let pipeline = self.named("chain-silu-mul-split", SILU_MUL_SPLIT);
        let params = self.uniform(&[out.len as u32, 0, 0, 0]);
        self.dispatch(&pipeline, buffer(fused), buffer(fused), buffer(out), &params, ((out.len as u32).div_ceil(256), 1, 1));
    }

    fn read(&mut self, v: &DeviceVec) {
        let size = (v.len.max(1) * 4) as u64;
        let staging = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.reads.push((buffer(v).clone(), staging, v.len));
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
        for (from, staging, len) in &self.reads {
            enc.copy_buffer_to_buffer(from, 0, staging, 0, (*len.max(&1) * 4) as u64);
        }
        self.gpu().queue.submit([enc.finish()]);
        for (_, staging, len) in &self.reads {
            staging.slice(..(*len as u64 * 4).max(4)).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.gpu().device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("webgpu: device lost while waiting for a chain");
        let out = self
            .reads
            .iter()
            .map(|(_, staging, len)| {
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

    /// A chain gives the CPU backend's answer: RMSNorm, a quantized matmul, the fused SwiGLU and an add, read back.
    #[test]
    fn a_chain_matches_the_cpus_ops() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let cpu = ggml_rs::CpuBackend::new();
        let (k, ff) = (256usize, 64usize);
        let mut s = 0x1234_5679u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        // a Q8_0 weight [2ff, k]: blocks of a scale and 32 int8s
        let mut bytes = vec![0u8; 2 * ff * (k / 32) * 34];
        for blk in bytes.chunks_mut(34) {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
            for v in blk[2..].iter_mut() {
                *v = (next() % 255) as u8;
            }
        }
        let wq = ggml_rs::QuantizedTensor::from_bytes_cpu(bytes, vec![2 * ff, k], GgmlType::Q8_0);
        let wq = ggml_rs::Backend::to_device_quant(&b, wq);
        assert!(DeviceChain::holds(&b, &wq));
        let xv: Vec<f32> = (0..k).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let nw: Vec<f32> = (0..k).map(|_| (next() % 1001) as f32 / 1000.0 + 0.5).collect();
        let res: Vec<f32> = (0..ff).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
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
        let close = |a: &[f32], b: &[f32], what: &str| {
            let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() <= 1e-4 * scale, "{what} [{i}]: {x} against {y}");
            }
        };
        close(&got[1], xn.data(), "rmsnorm");
        close(&got[0], &want, "the chain");
    }
}
