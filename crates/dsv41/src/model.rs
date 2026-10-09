//! The DeepSeek-V4.1 backbone (reference `Transformer`, `Block`): embed ->
//! 4 hyper-connection copies -> 40 blocks (Engram in front of layers 1 and
//! 14) -> collapse -> norm -> head. CPU reference path: every trunk weight
//! resident in RAM in its stored form, routed experts through the expert
//! cache. MTP/DSpark and vision are not built (docs/DEEPSEEK_V41.md).

use std::path::Path;
use std::sync::Arc;

use oaiy_engine::ecache::Ecache;
use oaiy_engine::{CachePolicy, Error, Result};

use crate::attention::{AttnState, Attention, Shared};
use crate::config::Config;
use crate::engram::{Engram, NgramHasher};
use crate::expert::{SafetensorsExpertStore, RECORD_BYTES};
use crate::hc::{self, HcParams, Mix, HC};
use crate::linear::{load_vec, Out, Weight};
use crate::moe::{Experts, Moe, Route};
use crate::ops::{rmsnorm, Rope};
use crate::safetensors::StIndex;

pub struct ModelOptions {
    /// Longest sequence (prompt + generation); sizes every cache.
    pub max_seq: usize,
    /// RAM for the routed-expert cache (0 = read every expert every time).
    pub expert_cache_bytes: usize,
    /// Bypass the page cache for expert reads.
    pub direct_io: bool,
}

impl Default for ModelOptions {
    fn default() -> Self {
        ModelOptions { max_seq: 4096, expert_cache_bytes: 16 << 30, direct_io: true }
    }
}

struct Layer {
    attn: Attention,
    moe: Moe,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: HcParams,
    hc_ffn: HcParams,
    engram: Option<Engram>,
}

pub struct Model {
    pub cfg: Config,
    embed: Weight,
    norm: Vec<f32>,
    head: Weight,
    layers: Vec<Layer>,
    hasher: NgramHasher,
    states: Vec<AttnState>,
    shared: Shared,
    experts: Experts,
    max_seq: usize,
}

/// Intermediate-value sink for tests and debugging: `(name, values)`.
pub type Trace<'a> = &'a mut dyn FnMut(&str, &[f32]);

/// Teacher forcing for tests: given layer `l >= 1`, the stream and `pre_mix`
/// to feed it (normally the previous layer's output), replacing what this
/// model computed. Isolates each layer's error from everything upstream.
pub type Teacher<'a> = &'a dyn Fn(usize) -> Result<(Vec<f32>, Vec<[f32; HC]>)>;

/// What the golden-file tests drive: the CPU [`Model`] and the GPU model
/// (dsv41-cuda) both implement it, so one test suite validates both.
pub trait Backbone {
    fn config(&self) -> &Config;
    fn config_mut(&mut self) -> &mut Config;
    /// Forward with optional teacher forcing; `trace` receives the same
    /// names as [`Model::forward`] (`layerNN.out`, `.attn_out`, ...).
    fn forward_traced(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Trace<'_>) -> Result<Vec<f32>>;
}

impl Backbone for Model {
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn config_mut(&mut self) -> &mut Config {
        &mut self.cfg
    }
    fn forward_traced(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Trace<'_>) -> Result<Vec<f32>> {
        self.forward_impl(ids, start_pos, teacher, trace)
    }
}

impl Model {
    /// Load the trunk from `model_dir` (the checkpoint, used in place) and the
    /// Engram precompute from `engram_meta`.
    pub fn load(model_dir: &Path, engram_meta: &Path, opts: &ModelOptions) -> Result<Model> {
        let cfg = Config::load(model_dir)?;
        let idx = StIndex::open(model_dir)?;
        if opts.max_seq == 0 || !opts.max_seq.is_multiple_of(2) {
            return Err(Error::Arg("max_seq must be a positive even number".into()));
        }
        let rope_window = Arc::new(Rope::new(cfg.rope_head_dim, opts.max_seq, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow));
        let rope_compress = Arc::new(Rope::new(
            cfg.rope_head_dim,
            opts.max_seq,
            cfg.original_seq_len,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        ));
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let rope = if cfg.ratio(l) > 0 { rope_compress.clone() } else { rope_window.clone() };
            let p = format!("layers.{l}");
            layers.push(Layer {
                attn: Attention::load(&idx, &cfg, l, rope)?,
                moe: Moe::load(&idx, l)?,
                attn_norm: load_vec(&idx, &format!("{p}.attn_norm.weight"))?,
                ffn_norm: load_vec(&idx, &format!("{p}.ffn_norm.weight"))?,
                hc_attn: HcParams::load(&idx, &p, "attn")?,
                hc_ffn: HcParams::load(&idx, &p, "ffn")?,
                engram: if cfg.engram_layer_ids.contains(&l) { Some(Engram::load(&idx, &cfg, l)?) } else { None },
            });
        }
        let states = layers.iter().map(|ly| ly.attn.new_state(&cfg, opts.max_seq)).collect();
        let store = SafetensorsExpertStore::open(&idx, cfg.n_layers as u32, cfg.n_routed_experts as u32, opts.direct_io)?;
        Ok(Model {
            embed: Weight::load(&idx, "embed")?,
            norm: load_vec(&idx, "norm.weight")?,
            head: Weight::load(&idx, "head")?,
            hasher: NgramHasher::load(engram_meta, &cfg, opts.max_seq)?,
            layers,
            states,
            shared: Shared::default(),
            experts: Experts {
                // Every hardware thread for the routed experts (a decode step's at once, a prompt's spread out).
                pool: Some(crate::cpu_experts::CpuExperts::new(0)),
                kernel: crate::cpu_experts::fp4_rows,
                tokens: crate::cpu_experts::fp4_tokens,
                gpu: None,
                store: Arc::new(store),
                cache: Ecache::new(opts.expert_cache_bytes, RECORD_BYTES, CachePolicy::Lfru),
                uses: crate::moe::Uses::new(cfg.n_layers, cfg.n_routed_experts),
            },
            max_seq: opts.max_seq,
            cfg,
        })
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn expert_cache(&self) -> &Ecache {
        &self.experts.cache
    }

    /// Where the routed experts' records are read from.
    pub fn expert_store(&self) -> &Arc<dyn oaiy_engine::store::WeightStore> {
        &self.experts.store
    }

    /// Every routed expert's uses of late ([`crate::moe::Uses`]).
    pub fn expert_uses(&self) -> &crate::moe::Uses {
        &self.experts.uses
    }

    /// Run `ids` at positions `start_pos..` (a prefill at 0, then one token at
    /// a time) and return the last position's logits (f32, `[vocab]`).
    pub fn forward(&mut self, ids: &[u32], start_pos: usize, trace: Trace<'_>) -> Result<Vec<f32>> {
        self.forward_impl(ids, start_pos, None, trace)
    }

    /// [`forward`](Self::forward), but every layer after the first starts
    /// from `teacher`'s stream instead of the previous layer's output.
    pub fn forward_teacher(&mut self, ids: &[u32], start_pos: usize, teacher: Teacher<'_>, trace: Trace<'_>) -> Result<Vec<f32>> {
        self.forward_impl(ids, start_pos, Some(teacher), trace)
    }

    fn forward_impl(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Trace<'_>) -> Result<Vec<f32>> {
        let (d, t) = (self.cfg.dim, ids.len());
        if t == 0 || start_pos + t > self.max_seq {
            return Err(Error::Arg(format!("{t} tokens at {start_pos} do not fit max_seq {}", self.max_seq)));
        }
        let hashes = self.hasher.forward(ids, start_pos)?;
        trace("engram_hashes", &hashes.iter().map(|&v| v as f32).collect::<Vec<_>>());

        let mut emb = vec![0.0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                return Err(Error::Arg(format!("token {id} outside the vocabulary")));
            }
            self.embed.row(id as usize, &mut emb[i * d..(i + 1) * d]);
        }
        trace("embed", &emb);
        let mut h: Vec<f32> = (0..t).flat_map(|i| std::iter::repeat_n(&emb[i * d..(i + 1) * d], HC).flatten().copied()).collect();
        let mut pre_mix: Vec<[f32; HC]> = vec![[1.0, 0.0, 0.0, 0.0]; t];

        let cols = self.hasher.cols();
        let n_eng = self.cfg.engram_layer_ids.len();
        for l in 0..self.cfg.n_layers {
            if let (Some(teach), true) = (teacher, l > 0) {
                let (th, tp) = teach(l)?;
                if th.len() != h.len() || tp.len() != t {
                    return Err(Error::Arg(format!("teacher input for layer {l} has the wrong shape")));
                }
                h = th;
                pre_mix = tp;
            }
            if let Some(eg) = &self.layers[l].engram {
                let hs: Vec<i64> = (0..t)
                    .flat_map(|i| hashes[(i * n_eng + eg.hash_index) * cols..(i * n_eng + eg.hash_index + 1) * cols].iter().copied())
                    .collect();
                let looking_up = std::time::Instant::now();
                h = eg.forward(&self.cfg, &h, t, &hs)?;
                crate::profile::add(crate::profile::Part::Engram, looking_up);
            }
            let (out, next_pre, _routes) = self.block(l, &h, t, start_pos, &pre_mix, trace)?;
            h = out;
            pre_mix = next_pre;
            trace(&format!("layer{l:02}.out"), &h);
            trace(&format!("layer{l:02}.pre_mix"), &pre_mix.iter().flatten().copied().collect::<Vec<_>>());
        }

        if let Some(gpu) = &self.experts.gpu {
            gpu.pass_done(t, &self.experts.cache, self.experts.store.as_ref());
        }
        // The experts' counts of uses age as the tokens go by, and the RAM cache's own with them: both then say what
        // has been used of late.
        if t == 1 && self.experts.uses.step() {
            self.experts.cache.decay();
        }
        let last = t - 1;
        let x = hc::pre(&h[last * HC * d..(last + 1) * HC * d], &pre_mix[last]);
        let x = rmsnorm(&x, &self.norm, self.cfg.norm_eps);
        let logits = self.head.forward(&x, 1, Out::F32);
        trace("logits", &logits);
        Ok(logits)
    }

    /// One `Block`: attention and MoE, each between `hc_pre` and `hc_post`.
    /// The mix a sublayer computes is used by the next one, so this returns
    /// the FFN's `pre` for the next block's attention.
    #[allow(clippy::type_complexity)]
    fn block(
        &mut self,
        l: usize,
        h: &[f32],
        t: usize,
        start_pos: usize,
        pre_mix: &[[f32; HC]],
        trace: Trace<'_>,
    ) -> Result<(Vec<f32>, Vec<[f32; HC]>, Vec<Route>)> {
        let Model { cfg, layers, states, shared, experts, .. } = self;
        let ly = &layers[l];
        let (d, eps) = (cfg.dim, cfg.norm_eps);
        // token i's [HC, dim] stream inside a [t, HC, dim] buffer
        let tok = |i: usize| i * HC * d..(i + 1) * HC * d;
        // Each token's mixing on its own, so spread over the threads: the same results. One by one, a 2,000-token
        // prompt's dot products alone took 29 s (24 of HC * dim a token, twice a layer; `hc`'s timing test).
        let mixes = |x: &[f32], p: &HcParams| -> Vec<Mix> { per_token(t, |i| hc::mixes(&x[tok(i)], p, eps, cfg.hc_sinkhorn_iters, cfg.hc_eps)) };

        // A long prompt's experts of this layer are read while its attention runs, until its router has chosen.
        let experts: &Experts = experts;
        let n_experts = cfg.n_routed_experts as u32;
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            if t >= crate::moe::PREFETCH_TOKENS {
                let stop = &stop;
                scope.spawn(move || experts.prefetch(l as u32, n_experts, stop));
            }
            let mixing = std::time::Instant::now();
            let attn_mix = mixes(h, &ly.hc_attn);
            let x: Vec<f32> = per_token(t, |i| hc::pre(&h[tok(i)], &pre_mix[i])).concat();
            crate::profile::add(crate::profile::Part::Mixing, mixing);
            let x = rmsnorm(&x, &ly.attn_norm, eps);
            let a = ly.attn.forward(cfg, &x, t, start_pos, states, shared);
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let a = a?;
            trace(&format!("layer{l:02}.attn_out"), &a);
            let mixing = std::time::Instant::now();
            let h1: Vec<f32> = per_token(t, |i| hc::post(&a[i * d..(i + 1) * d], &h[tok(i)], &attn_mix[i])).concat();

            let ffn_mix = mixes(&h1, &ly.hc_ffn);
            let x: Vec<f32> = per_token(t, |i| hc::pre(&h1[tok(i)], &attn_mix[i].pre)).concat();
            crate::profile::add(crate::profile::Part::Mixing, mixing);
            let x = rmsnorm(&x, &ly.ffn_norm, eps);
            let (m, routes) = ly.moe.forward(cfg, &x, t, experts)?;
            trace(&format!("layer{l:02}.moe_out"), &m);
            trace(&format!("layer{l:02}.route_ids"), &routes.iter().flat_map(|r| r.experts.iter().map(|&e| e as f32)).collect::<Vec<_>>());
            trace(&format!("layer{l:02}.route_w"), &routes.iter().flat_map(|r| r.weights.iter().copied()).collect::<Vec<_>>());
            let mixing = std::time::Instant::now();
            let h2: Vec<f32> = per_token(t, |i| hc::post(&m[i * d..(i + 1) * d], &h1[tok(i)], &ffn_mix[i])).concat();
            crate::profile::add(crate::profile::Part::Mixing, mixing);
            Ok((h2, ffn_mix.iter().map(|m| m.pre).collect(), routes))
        })
    }
}

/// `f(i)` for each of `t` tokens, in order, spread over the threads (each token's on its own, so the same results as
/// one by one).
fn per_token<T: Send>(t: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let parts: std::sync::Mutex<Vec<(usize, Vec<T>)>> = std::sync::Mutex::new(Vec::new());
    oaiy_engine::backend::parallel_rows(t, 8, &|b, e| {
        let part: Vec<T> = (b..e).map(&f).collect();
        parts.lock().unwrap_or_else(|p| p.into_inner()).push((b, part));
    });
    let mut parts = parts.into_inner().unwrap_or_else(|p| p.into_inner());
    parts.sort_by_key(|(b, _)| *b);
    parts.into_iter().flat_map(|(_, part)| part).collect()
}

/// What the model's state holds after some tokens, to come back to with [`Model::restore`]: a conversation's next turn
/// starts with the prompt the last one read but not with the reply's tokens (its chat template writes the reply its own
/// way), so the state the reply moved on from is the one it can use. Every layer's caches and the n-gram history.
#[derive(Clone)]
pub struct Checkpoint {
    states: Vec<AttnState>,
    history: Vec<i64>,
}

impl Model {
    /// The state as it stands (the caller knows how many tokens it covers).
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint { states: self.states.clone(), history: self.hasher.history().to_vec() }
    }

    /// Back to `c`: the next forward continues from the tokens it covered.
    pub fn restore(&mut self, c: &Checkpoint) {
        self.states.clone_from(&c.states);
        self.hasher.set_cache(&c.history);
    }

    /// Read up to `count` routed experts the RAM cache does not hold into it, from `*cursor` on (every layer's expert
    /// `e` before any layer's `e + 1`: a step uses each layer's alike), while the cache has room: a cold decode step
    /// waits for the drive for every expert it has not met (some 13 ms each from a 1.4 GB/s drive, and a step routes
    /// to 240), so what is read while nothing is asked is a wait a later step does not have. False once every expert
    /// has been looked at or the cache is full (reading past that would only swap one unused record for another).
    /// `elsewhere(layer, expert)`: an expert another tier holds for good (a GPU's), which RAM need not.
    pub fn warm_experts(&self, cursor: &mut usize, count: usize, elsewhere: &dyn Fn(u32, u32) -> bool) -> bool {
        let layers = self.cfg.n_layers;
        self.warm_by(&|i| ((i % layers) as u32, (i / layers) as u32), cursor, count, elsewhere)
    }

    /// [`Self::warm_experts`] in `order` (every expert once: [`crate::moe::Uses::order`]'s, the most used of an
    /// earlier run first), so what RAM holds when it is full is what is most likely to be asked for.
    pub fn warm_experts_in(&self, order: &[(u32, u32)], cursor: &mut usize, count: usize, elsewhere: &dyn Fn(u32, u32) -> bool) -> bool {
        assert_eq!(order.len(), self.cfg.n_layers * self.cfg.n_routed_experts, "an order of every expert");
        self.warm_by(&|i| order[i], cursor, count, elsewhere)
    }

    fn warm_by(&self, at: &dyn Fn(usize) -> (u32, u32), cursor: &mut usize, count: usize, elsewhere: &dyn Fn(u32, u32) -> bool) -> bool {
        let cache = &self.experts.cache;
        let total = self.cfg.n_layers * self.cfg.n_routed_experts;
        let room = cache.n_slots().saturating_sub(cache.len());
        if room == 0 {
            return false;
        }
        let mut picks = Vec::with_capacity(count.min(room));
        while *cursor < total && picks.len() < count.min(room) {
            let (l, e) = at(*cursor);
            *cursor += 1;
            if !cache.probe(l, e) && !elsewhere(l, e) {
                picks.push((l, e));
            }
        }
        self.experts.warm(&picks);
        *cursor < total
    }

    /// Run the routed experts on the CPU through `kernel` (a decode step's pool and a prompt's workers both): the
    /// crate `dsv41-simd` picks this CPU's fastest, which this crate may define but not call (it forbids `unsafe`).
    /// Every kernel gives the same bits.
    pub fn set_expert_row_kernel(&mut self, kernel: crate::cpu_experts::RowKernel) {
        self.experts.pool = Some(crate::cpu_experts::CpuExperts::with_kernel(0, kernel));
        self.experts.kernel = kernel;
    }

    /// Run a prompt's routed experts on the CPU through `kernel` (all of an expert's tokens at once), as
    /// [`Self::set_expert_row_kernel`] sets a decode step's: `dsv41-simd` picks this CPU's. The same bits each.
    pub fn set_expert_tokens_kernel(&mut self, kernel: crate::cpu_experts::TokensKernel) {
        self.experts.tokens = kernel;
    }

    /// Have a prompt's busy routed experts' matmuls made by `kernel` (a GPU's: see [`crate::moe::Experts::gpu`]).
    pub fn set_experts_kernel(&mut self, kernel: Option<Arc<dyn crate::expert::ExpertsKernel>>) {
        self.experts.gpu = kernel;
    }

    /// Hand the trunk's dense weights to `place` (a GPU's), each with its name (`layers.<l>.attn.wq_b`, `head`): the
    /// weight it returns replaces the one it was given (a [`Weight::Device`]), None leaves it here. The embedding stays
    /// (a prompt reads rows of it). How many were placed, and their bytes as stored.
    pub fn offload(&mut self, mut place: impl FnMut(&str, &Weight) -> Option<Weight>) -> (usize, u64) {
        let (mut count, mut bytes) = (0usize, 0u64);
        let mut each = |name: String, w: &mut Weight| {
            let size = match w {
                Weight::Fp8 { w, s, .. } => (w.len() + s.len()) as u64,
                Weight::Bf16 { w, .. } => w.len() as u64 * 2,
                Weight::F32 { w, .. } => w.len() as u64 * 4,
                Weight::Device { .. } => return,
            };
            if let Some(new) = place(&name, w) {
                *w = new;
                count += 1;
                bytes += size;
            }
        };
        for (l, layer) in self.layers.iter_mut().enumerate() {
            for (name, w) in layer.attn.weights_mut() {
                each(format!("layers.{l}.attn.{name}"), w);
            }
            for (name, w) in layer.moe.weights_mut() {
                each(format!("layers.{l}.ffn.{name}"), w);
            }
            if let Some(e) = &mut layer.engram {
                each(format!("layers.{l}.engram.wkv"), e.weight_mut());
            }
        }
        each("head".into(), &mut self.head);
        (count, bytes)
    }
}

/// Index of the largest logit (greedy decoding).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
