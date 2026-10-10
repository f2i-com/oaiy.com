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
        let (row, hd) = (2 * self.kv_heads * self.head_dim, self.head_dim);
        // rotate-half RoPE's table for every position: each position's (sin, cos) a pair
        let table: Vec<f32> = (0..capacity)
            .flat_map(|p| (0..hd / 2).flat_map(move |i| {
                let a = p as f64 / self.theta.powf(2. * i as f64 / hd as f64);
                [a.sin() as f32, a.cos() as f32]
            }))
            .collect();
        let rope = gpu.vec(table.len());
        gpu.upload(&rope, &table);
        Cache { kv: (0..self.layers.len()).map(|_| gpu.vec(capacity * row)).collect(), len: 0, capacity, rope, scratch: Vec::new() }
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
        // The step's vectors are the cache's, kept for its row count: a step made new ones each time (and new bind
        // groups for them), which was most of a code predictor's step of 1 row. Their values are this step's alone:
        // the rows' RoPE table is copied from the cache's on the GPU, in the recording's order.
        let at = match cache.scratch.iter().position(|s| s.rows == t) {
            Some(at) => at,
            None => {
                let v = |n: usize| gpu.vec(n);
                let one = v(1);
                gpu.upload(&one, &[1.0]);
                let att = v(gpu.attention_rows_out_len(t, nq, hd, cache.capacity));
                cache.scratch.push(Scratch {
                    rows: t,
                    td: v(t * hd),
                    n: v(t * h),
                    q: v(t * nq * hd),
                    k: v(t * nkv * hd),
                    vv: v(t * nkv * hd),
                    qn: v(t * nq * hd),
                    kn: v(t * nkv * hd),
                    att,
                    o: v(t * h),
                    g: v(t * self.ff),
                    u: v(t * self.ff),
                    act: v(t * self.ff),
                    one,
                });
                cache.scratch.len() - 1
            }
        };
        let Scratch { td, n, q, k, vv, qn, kn, att, o, g, u, act, one, .. } = &cache.scratch[at];
        let wider;
        let att = if att.len >= gpu.attention_rows_out_len(t, nq, hd, past + t) {
            att
        } else {
            wider = gpu.vec(gpu.attention_rows_out_len(t, nq, hd, past + t));
            &wider
        };
        r.copy(&cache.rope, past * hd, td, 0, t * hd);
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
    /// RoPE's table for every position (`[capacity, head_dim]`: each position's (sin, cos) pairs).
    rope: DeviceVec,
    /// A step's vectors, kept for each row count it has taken.
    scratch: Vec<Scratch>,
}

/// The vectors a step of `rows` rows works in (its rows' RoPE table, the layers' intermediates, a 1 for the sums).
struct Scratch {
    rows: usize,
    td: DeviceVec,
    n: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    vv: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    att: DeviceVec,
    o: DeviceVec,
    g: DeviceVec,
    u: DeviceVec,
    act: DeviceVec,
    one: DeviceVec,
}
