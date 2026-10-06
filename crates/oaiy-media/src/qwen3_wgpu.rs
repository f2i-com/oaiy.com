//! A Qwen3 decoder's prompt on WebGPU (GQA, per-head q/k RMS norms, SwiGLU, rotate-half RoPE): its final, normed hidden
//! states for a prompt's rows (MOSS-SoundEffect's text encoder, Qwen3-1.7B), the weights f16 from a checkpoint's BF16
//! (or F32), the layers on the device all at once.
use crate::ltx::store::Store;
use crate::wgpu_weights::{f16_words, f16_words_f32};
use candle_core::{Device, Result};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs_wgpu::WgpuBackend;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("Qwen3 on WebGPU: {e}"))
}

/// A matrix `[n, k]` as f16 pairs.
pub(crate) struct Mat {
    pub w: DeviceVec,
    pub n: usize,
    pub k: usize,
}

/// `key`'s matrix as f16 on `gpu` (BF16 converted on every core, F32 rounded).
pub(crate) fn mat(store: &mut Store, gpu: &WgpuBackend, key: &str) -> Result<Mat> {
    let shape = store.index.get(key).ok_or_else(|| err(format!("no {key}")))?.shape.clone();
    let &[n, k] = shape.as_slice() else { return Err(err(format!("{key}: a matrix of {shape:?}"))) };
    let words = match store.bf16_bytes(key)? {
        Some(b) => f16_words(&b),
        None => f16_words_f32(&store.tensor_f32(key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
    }
    .ok_or_else(|| err(format!("{key}: past f16's range")))?;
    let w = gpu.vec(words.len());
    gpu.upload(&w, &words);
    Ok(Mat { w, n, k })
}

/// `key`'s values as f32 on `gpu`.
pub(crate) fn vector(store: &mut Store, gpu: &WgpuBackend, key: &str) -> Result<DeviceVec> {
    let v = store.tensor_f32(key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let d = gpu.vec(v.len());
    gpu.upload(&d, &v);
    Ok(d)
}

struct Layer {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    input_norm: DeviceVec,
    post_norm: DeviceVec,
    gate: Mat,
    up: Mat,
    down: Mat,
}

pub struct WgpuQwen3 {
    layers: Vec<Layer>,
    norm: DeviceVec,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    ff: usize,
    theta: f64,
    eps: f32,
}

impl WgpuQwen3 {
    /// The decoder under `prefix` (`{prefix}.layers.N`, `{prefix}.norm`) of `store`, onto `gpu`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(store: &mut Store, gpu: &WgpuBackend, prefix: &str, layers: usize, heads: usize, kv_heads: usize, head_dim: usize, theta: f64, eps: f32) -> Result<Self> {
        let mut out = Vec::with_capacity(layers);
        for i in 0..layers {
            let l = format!("{prefix}.layers.{i}");
            out.push(Layer {
                q: mat(store, gpu, &format!("{l}.self_attn.q_proj.weight"))?,
                k: mat(store, gpu, &format!("{l}.self_attn.k_proj.weight"))?,
                v: mat(store, gpu, &format!("{l}.self_attn.v_proj.weight"))?,
                o: mat(store, gpu, &format!("{l}.self_attn.o_proj.weight"))?,
                q_norm: vector(store, gpu, &format!("{l}.self_attn.q_norm.weight"))?,
                k_norm: vector(store, gpu, &format!("{l}.self_attn.k_norm.weight"))?,
                input_norm: vector(store, gpu, &format!("{l}.input_layernorm.weight"))?,
                post_norm: vector(store, gpu, &format!("{l}.post_attention_layernorm.weight"))?,
                gate: mat(store, gpu, &format!("{l}.mlp.gate_proj.weight"))?,
                up: mat(store, gpu, &format!("{l}.mlp.up_proj.weight"))?,
                down: mat(store, gpu, &format!("{l}.mlp.down_proj.weight"))?,
            });
        }
        let (hidden, ff) = (out.first().map_or(0, |l| l.q.k), out.first().map_or(0, |l| l.gate.n));
        if out.iter().any(|l| l.q.n != heads * head_dim || l.k.n != kv_heads * head_dim || l.o.n != hidden || l.down.n != hidden) {
            return Err(err("the layers' shapes are not the configuration's"));
        }
        Ok(Self { layers: out, norm: vector(store, gpu, &format!("{prefix}.norm.weight"))?, heads, kv_heads, head_dim, hidden, ff, theta, eps })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// A key-value cache for `capacity` positions (each layer's keys then values a row), from position 0.
    pub fn cache(&self, gpu: &WgpuBackend, capacity: usize) -> Cache {
        let row = 2 * self.kv_heads * self.head_dim;
        Cache { kv: (0..self.layers.len()).map(|_| gpu.vec(capacity * row)).collect(), len: 0, capacity }
    }

    /// The final, normed hidden states (`[t, hidden]`) of a prompt's embeddings `x` (`[t, hidden]`, from position 0),
    /// recorded on `r` into `out` (exactly `[t, hidden]`: its rows' norms take their width from its length).
    pub fn prompt(&self, gpu: &WgpuBackend, r: &mut dyn ChainRecorder, x: &DeviceVec, t: usize, out: &DeviceVec) {
        let mut cache = self.cache(gpu, t);
        self.step(gpu, r, x, t, &mut cache, out);
    }

    /// `t` rows of embeddings `x` (`[t, hidden]`) at the cache's next positions: their final, normed states into `out`
    /// (exactly `[t, hidden]`), their keys and values into `cache` (its length grown by `t`).
    pub fn step(&self, gpu: &WgpuBackend, r: &mut dyn ChainRecorder, x: &DeviceVec, t: usize, cache: &mut Cache, out: &DeviceVec) {
        let (h, hd, nq, nkv) = (self.hidden, self.head_dim, self.heads, self.kv_heads);
        assert_eq!(out.len, t * h, "Qwen3 on WebGPU: a step's states exactly its rows");
        let past = cache.len;
        assert!(past + t <= cache.capacity, "Qwen3 on WebGPU: {} positions past its cache's {}", past + t, cache.capacity);
        // rotate-half RoPE's table for the rows' positions: each position's (sin, cos) a pair
        let table: Vec<f32> = (past..past + t)
            .flat_map(|p| (0..hd / 2).flat_map(move |i| {
                let a = p as f64 / self.theta.powf(2. * i as f64 / hd as f64);
                [a.sin() as f32, a.cos() as f32]
            }))
            .collect();
        let td = gpu.vec(table.len());
        gpu.upload(&td, &table);
        let v = |n: usize| gpu.vec(n);
        let (n, q, k, vv, qn, kn, att, o) = (v(t * h), v(t * nq * hd), v(t * nkv * hd), v(t * nkv * hd), v(t * nq * hd), v(t * nkv * hd), v(gpu.attention_rows_out_len(t, nq, hd, past + t)), v(t * h));
        let (g, u, act) = (v(t * self.ff), v(t * self.ff), v(t * self.ff));
        let one = gpu.vec(1);
        gpu.upload(&one, &[1.0]);
        r.copy(x, 0, out, 0, t * h);
        for (l, kv) in self.layers.iter().zip(&cache.kv) {
            r.rmsnorm_rows(out, &l.input_norm, &n, t, self.eps);
            r.matmul_f16_rows(&l.q.w, l.q.n, l.q.k, &n, &q, t);
            r.matmul_f16_rows(&l.k.w, l.k.n, l.k.k, &n, &k, t);
            r.matmul_f16_rows(&l.v.w, l.v.n, l.v.k, &n, &vv, t);
            r.rmsnorm_rows(&q, &l.q_norm, &qn, t * nq, self.eps);
            r.rmsnorm_rows(&k, &l.k_norm, &kn, t * nkv, self.eps);
            r.rope_rows(&qn, t, nq, hd, &td, true);
            r.rope_rows(&kn, t, nkv, hd, &td, true);
            r.store_rows(&kn, kv, t, nkv * hd, past, 2 * nkv * hd, 0);
            r.store_rows(&vv, kv, t, nkv * hd, past, 2 * nkv * hd, nkv * hd);
            r.attention_rows(&qn, kv, &att, t, nq, nkv, hd, past, None, 1.0 / (hd as f32).sqrt());
            r.matmul_f16_rows(&l.o.w, l.o.n, l.o.k, &att, &o, t);
            r.axpy_at(out, &o, &one, 0, t * h);
            r.rmsnorm_rows(out, &l.post_norm, &n, t, self.eps);
            r.matmul_f16_rows(&l.gate.w, l.gate.n, l.gate.k, &n, &g, t);
            r.matmul_f16_rows(&l.up.w, l.up.n, l.up.k, &n, &u, t);
            r.silu_mul(&g, &u, &act, t * self.ff);
            r.matmul_f16_rows(&l.down.w, l.down.n, l.down.k, &act, &o, t);
            r.axpy_at(out, &o, &one, 0, t * h);
        }
        r.rmsnorm_rows(out, &self.norm, &n, t, self.eps);
        r.copy(&n, 0, out, 0, t * h);
        cache.len += t;
    }
}

/// A [`WgpuQwen3`]'s keys and values so far: a layer's rows (keys, then values) for `capacity` positions.
pub struct Cache {
    kv: Vec<DeviceVec>,
    pub len: usize,
    capacity: usize,
}
