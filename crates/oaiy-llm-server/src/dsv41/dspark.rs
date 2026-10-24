//! DSpark: DeepSeek-V4.1's block drafter (`mtp.0..` of the checkpoint; DeepInfer's `DFlash`), whose drafts the trunk
//! checks in one pass of all of them.
//!
//! It reads the trunk: the mean of the four streams entering each of the trunk's layers `targets` (37, 38, 39), laid
//! end to end, through `main_proj` and `main_norm`, gives each of its layers its keys of the trunk's rows, kept in a
//! window of the last 128 positions as the trunk's layers keep theirs. A draft is one pass of `block` rows, the token
//! the trunk chose last and `block - 1` noise tokens at the positions after it, through its layers (blocks like the
//! trunk's, their attention window-only, every row attending to the window and to every row of the block, its experts
//! 128 a layer, 3 a token); their streams are mixed down and normed (`norm`) and given the trunk's head, and each row's
//! logits corrected by a rank-256 term of the token before it (`markov_head`: the chosen token for the first row, each
//! row's draft for the next). A confidence (`confidence_head`, a sigmoid of the row's state and that token's rank-256
//! embedding) says how likely the trunk is to choose each draft.

use std::sync::Arc;

use oaiy_engine::ecache::Ecache;
use oaiy_engine::{CachePolicy, Error, Result};

use crate::dsv41::attention::{AttnState, Attention};
use crate::dsv41::config::{Config, DsparkConfig};
use crate::dsv41::expert::SafetensorsExpertStore;
use crate::dsv41::hc::{self, HcParams, Mix, HC};
use crate::dsv41::linear::{load_vec, Out, Weight};
use crate::dsv41::moe::{Experts, Moe};
use crate::dsv41::ops::{rmsnorm, Rope};
use crate::dsv41::safetensors::StIndex;

/// The largest value of a row of the trunk's streams the drafter's projection is given ([`Drafter::observe`]).
const FEATURE_MAX: f32 = 64.0;

struct Layer {
    attn: Attention,
    moe: Moe,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: HcParams,
    hc_ffn: HcParams,
}

/// A pass's drafts: the tokens, and for each how likely the trunk is to choose it (given it chose those before).
#[derive(Clone, Debug, Default)]
pub struct Draft {
    pub tokens: Vec<u32>,
    pub confidence: Vec<f32>,
}

pub struct Drafter {
    /// Its own shape: the trunk's, with its layers (window-only) and experts.
    cfg: Config,
    spec: DsparkConfig,
    main_proj: Weight,
    main_norm: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    markov_embed: Weight,
    markov_head: Weight,
    /// `[dim + markov_rank]`: the confidence's weights over a row's state and its token's rank-`markov_rank` embedding.
    confidence: Vec<f32>,
    states: Vec<AttnState>,
    pub(crate) experts: Experts,
}

impl Drafter {
    /// The drafter of the checkpoint `idx` of the trunk `trunk`, if it has one; its experts' records through a RAM cache
    /// of `cache_records` of them (all 384, a few GB, where it is to hold them).
    pub fn load(idx: &StIndex, trunk: &Config, max_seq: usize, cache_records: usize, direct_io: bool) -> Result<Option<Drafter>> {
        let Some(spec) = trunk.dspark.clone() else { return Ok(None) };
        let mut cfg = trunk.clone();
        cfg.n_layers = spec.layers;
        cfg.n_routed_experts = spec.n_routed_experts;
        cfg.n_activated_experts = spec.n_activated_experts;
        cfg.compress_ratios = vec![0; spec.layers];
        cfg.kv_source_layers.clear();
        cfg.index_source_layers.clear();
        cfg.candidate_source_layer = None;
        cfg.engram_layer_ids.clear();
        cfg.engram_num_embeddings.clear();
        cfg.dspark = None;
        // (window-only layers: the base theta, no YaRN, as the trunk's window layers)
        let rope = Arc::new(Rope::new(cfg.rope_head_dim, max_seq, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow));
        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let p = format!("mtp.{l}");
            layers.push(Layer {
                attn: Attention::load_named(idx, &cfg, l, &format!("{p}.attn"), Arc::clone(&rope))?,
                moe: Moe::load_named(idx, l, &format!("{p}.ffn"))?,
                attn_norm: load_vec(idx, &format!("{p}.attn_norm.weight"))?,
                ffn_norm: load_vec(idx, &format!("{p}.ffn_norm.weight"))?,
                hc_attn: HcParams::load(idx, &p, "attn")?,
                hc_ffn: HcParams::load(idx, &p, "ffn")?,
            });
        }
        let last = format!("mtp.{}", spec.layers - 1);
        let confidence = {
            let w = Weight::load(idx, &format!("{last}.confidence_head.proj"))?;
            if w.n() != 1 || w.k() != cfg.dim + spec.markov_rank {
                return Err(Error::Format(format!("{last}.confidence_head: [{}, {}], expected [1, {}]", w.n(), w.k(), cfg.dim + spec.markov_rank)));
            }
            let mut v = vec![0.0f32; w.k()];
            w.row(0, &mut v);
            v
        };
        let states = layers.iter().map(|ly| ly.attn.new_state(&cfg, max_seq)).collect();
        let store = SafetensorsExpertStore::open_named(idx, spec.layers as u32, spec.n_routed_experts as u32, direct_io, &|l| format!("mtp.{l}.ffn"))?;
        let format = store.format().clone();
        Ok(Some(Drafter {
            main_proj: Weight::load(idx, "mtp.0.main_proj")?,
            main_norm: load_vec(idx, "mtp.0.main_norm.weight")?,
            norm: load_vec(idx, &format!("{last}.norm.weight"))?,
            markov_embed: Weight::load(idx, &format!("{last}.markov_head.embed"))?,
            markov_head: Weight::load(idx, &format!("{last}.markov_head.head"))?,
            confidence,
            layers,
            states,
            experts: Experts {
                pool: Some(crate::dsv41::cpu_experts::CpuExperts::new(0)),
                kernel: crate::dsv41::simd::row_kernel(),
                tokens: crate::dsv41::simd::tokens_kernel(),
                gpu: None,
                cache: Arc::new(Ecache::new(cache_records * format.record_bytes(), format.record_bytes(), CachePolicy::Lfru)),
                refill: crate::dsv41::moe::Refill::default(),
                store: Arc::new(store),
                uses: crate::dsv41::moe::Uses::new(spec.layers, spec.n_routed_experts),
                format,
            },
            spec,
            cfg,
        }))
    }

    /// The trunk's layers it reads ([`crate::dsv41::model::Model::forward_rows`]'s `targets`).
    pub fn targets(&self) -> &[usize] {
        &self.spec.targets
    }

    /// Drafts a pass makes.
    pub fn block(&self) -> usize {
        self.spec.block
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Its experts: where their records come from, and the kernel that runs them.
    pub fn set_experts_kernel(&mut self, kernel: Option<Arc<dyn crate::dsv41::expert::ExpertsKernel>>) {
        self.experts.gpu = kernel;
    }

    /// Its keys of the trunk as they stand (a few hundred KB a layer), to come back to with [`Drafter::set_keys`] where
    /// the trunk's state is taken back to a checkpoint.
    pub fn keys(&self) -> Vec<AttnState> {
        self.states.clone()
    }

    pub fn set_keys(&mut self, keys: &[AttnState]) {
        self.states.clone_from_slice(keys);
    }

    /// Take in the trunk's rows at `start_pos..`: `features` (`[t, targets * dim]`, from
    /// [`crate::dsv41::model::Model::forward_rows`]), each layer's keys of them kept (only the last window's matter).
    pub fn observe(&mut self, features: &[f32], t: usize, start_pos: usize) {
        let first = t.saturating_sub(self.cfg.window_size);
        let width = features.len() / t.max(1);
        let rows = t - first;
        // Each row scaled by a power of two to no more than FEATURE_MAX: the trunk's streams run to thousands at a few
        // positions (the first), which an EXL3 projection's f16 sums cannot hold (the drafter's keys of them were NaN,
        // and every draft with them). Its norm after makes the scale nothing (and a power of two changes no bits).
        let mut scaled = features[first * width..].to_vec();
        for row in scaled.chunks_exact_mut(width) {
            let most = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if most > FEATURE_MAX {
                let by = (most / FEATURE_MAX).log2().ceil().exp2();
                row.iter_mut().for_each(|v| *v /= by);
            }
        }
        let x = rmsnorm(&self.main_proj.forward(&scaled, rows, Out::Bf16), &self.main_norm, self.cfg.norm_eps);
        // every layer's keys of them in one call
        let of_x: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> = self.layers.iter().map(|ly| (ly.attn.wkv(), &x[..], rows, 0..ly.attn.wkv().n(), Out::Bf16)).collect();
        for ((ly, state), kva) in self.layers.iter().zip(self.states.iter_mut()).zip(crate::dsv41::linear::forward_together(&of_x)) {
            ly.attn.store_keys(&self.cfg, &kva, start_pos + first, state);
        }
    }

    /// Drafts after `golden` (the token the trunk chose, to be at position `at`): a pass of it and `block - 1` noise
    /// tokens, every row's logits (the trunk's head, `embed` its embedding) corrected by the token before it, the best
    /// of each the draft. Greedy: each the most likely.
    pub fn draft(&mut self, embed: &Weight, head: &Weight, golden: u32, at: usize) -> Result<Draft> {
        let (d, b, eps) = (self.cfg.dim, self.spec.block, self.cfg.norm_eps);
        let ids: Vec<u32> = std::iter::once(golden).chain(std::iter::repeat_n(self.spec.noise_token, b - 1)).collect();
        let mut emb = vec![0.0f32; b * d];
        for (i, &id) in ids.iter().enumerate() {
            embed.row(id as usize, &mut emb[i * d..(i + 1) * d]);
        }
        let mut h: Vec<f32> = (0..b).flat_map(|i| std::iter::repeat_n(&emb[i * d..(i + 1) * d], HC).flatten().copied()).collect();
        let mut pre_mix: Vec<[f32; HC]> = vec![[1.0, 0.0, 0.0, 0.0]; b];
        for l in 0..self.layers.len() {
            let (out, next) = self.layer(l, &h, b, at, &pre_mix)?;
            h = out;
            pre_mix = next;
        }
        let state: Vec<f32> = (0..b).flat_map(|i| hc::pre(&h[i * HC * d..(i + 1) * HC * d], &pre_mix[i])).collect();
        let logits = head.forward(&rmsnorm(&state, &self.norm, eps), b, Out::F32);
        let vocab = logits.len() / b;
        let rank = self.spec.markov_rank;
        // Each row's correction is by the token before it, the last row's draft: they are made together for the tokens
        // guessed (each row's best before its correction), and again from the first row whose draft was another, until
        // none is (one call where each sticks, rather than one a row).
        let mut prev: Vec<u32> = std::iter::once(golden).chain((0..b - 1).map(|i| crate::dsv41::model::argmax(&logits[i * vocab..(i + 1) * vocab]))).collect();
        let mut tokens = vec![0u32; b];
        let mut from = 0;
        while from < b {
            let mut m = vec![0.0f32; (b - from) * rank];
            for (i, row) in (from..b).zip(m.chunks_exact_mut(rank)) {
                self.markov_embed.row(prev[i] as usize, row);
            }
            let correction = self.markov_head.forward(&m, b - from, Out::F32);
            let mut again = b;
            for i in from..b {
                let c = &correction[(i - from) * vocab..(i - from + 1) * vocab];
                let row: Vec<f32> = logits[i * vocab..(i + 1) * vocab].iter().zip(c).map(|(a, c)| a + c).collect();
                tokens[i] = crate::dsv41::model::argmax(&row);
                if i + 1 < b && tokens[i] != prev[i + 1] {
                    prev[i + 1] = tokens[i];
                    again = i + 1;
                    break;
                }
            }
            from = again;
        }
        let mut draft = Draft { tokens, confidence: Vec::with_capacity(b) };
        let mut m = vec![0.0f32; rank];
        for i in 0..b {
            self.markov_embed.row(prev[i] as usize, &mut m);
            let s = &state[i * d..(i + 1) * d];
            let z: f32 = s.iter().zip(&self.confidence[..d]).map(|(a, w)| a * w).sum::<f32>() + m.iter().zip(&self.confidence[d..]).map(|(a, w)| a * w).sum::<f32>();
            draft.confidence.push(1.0 / (1.0 + (-z.clamp(-60.0, 60.0)).exp()));
        }
        Ok(draft)
    }

    /// One of its layers over the block: attention and MoE, each between its streams' mixes, as the trunk's blocks.
    fn layer(&mut self, l: usize, h: &[f32], t: usize, start_pos: usize, pre_mix: &[[f32; HC]]) -> Result<(Vec<f32>, Vec<[f32; HC]>)> {
        let Drafter { cfg, layers, states, experts, .. } = self;
        let ly = &layers[l];
        let (d, eps) = (cfg.dim, cfg.norm_eps);
        let tok = |i: usize| i * HC * d..(i + 1) * HC * d;
        let mixes = |x: &[f32], p: &HcParams| -> Vec<Mix> { (0..t).map(|i| hc::mixes(&x[tok(i)], p, eps, cfg.hc_sinkhorn_iters, cfg.hc_eps)).collect() };
        let attn_mix = mixes(h, &ly.hc_attn);
        let x: Vec<f32> = (0..t).flat_map(|i| hc::pre(&h[tok(i)], &pre_mix[i])).collect();
        let a = ly.attn.forward_draft(cfg, &rmsnorm(&x, &ly.attn_norm, eps), t, start_pos, &states[l])?;
        let h1: Vec<f32> = (0..t).flat_map(|i| hc::post(&a[i * d..(i + 1) * d], &h[tok(i)], &attn_mix[i])).collect();
        let ffn_mix = mixes(&h1, &ly.hc_ffn);
        let x: Vec<f32> = (0..t).flat_map(|i| hc::pre(&h1[tok(i)], &attn_mix[i].pre)).collect();
        let (m, _routes) = ly.moe.forward(cfg, &rmsnorm(&x, &ly.ffn_norm, eps), t, experts)?;
        let h2: Vec<f32> = (0..t).flat_map(|i| hc::post(&m[i * d..(i + 1) * d], &h1[tok(i)], &ffn_mix[i])).collect();
        Ok((h2, ffn_mix.iter().map(|m| m.pre).collect()))
    }

    /// Hand its dense weights to `place` (a GPU's), as [`crate::dsv41::model::Model::offload`] does the trunk's.
    pub fn offload(&mut self, mut place: impl FnMut(&str, &Weight) -> Option<Weight>) -> (usize, u64) {
        let mut named: Vec<(String, &mut Weight)> = vec![("mtp.0.main_proj".into(), &mut self.main_proj), ("markov_head.head".into(), &mut self.markov_head)];
        for (l, ly) in self.layers.iter_mut().enumerate() {
            for (name, w) in ly.attn.weights_mut() {
                named.push((format!("mtp.{l}.attn.{name}"), w));
            }
            for (name, w) in ly.moe.weights_mut() {
                named.push((format!("mtp.{l}.ffn.{name}"), w));
            }
        }
        crate::dsv41::model::place_all(named, &mut place)
    }
}
