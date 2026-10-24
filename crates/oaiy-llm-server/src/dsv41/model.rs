//! The DeepSeek-V4.1 backbone (reference `Transformer`, `Block`): embed ->
//! 4 hyper-connection copies -> 40 blocks (Engram in front of layers 1 and
//! 14) -> collapse -> norm -> head. CPU reference path: every trunk weight
//! resident in RAM in its stored form, routed experts through the expert
//! cache. DSpark's drafter is `crate::dsv41::dspark`; vision is not built here (docs/DEEPSEEK_V41.md).

use std::path::Path;
use std::sync::Arc;

use oaiy_engine::ecache::Ecache;
use oaiy_engine::{CachePolicy, Error, Result};

use crate::dsv41::attention::{AttnState, Attention, Shared};
use crate::dsv41::config::Config;
use crate::dsv41::engram::{Engram, NgramHasher};
use crate::dsv41::expert::SafetensorsExpertStore;
use crate::dsv41::hc::{self, HcParams, Mix, HC};
use crate::dsv41::linear::{load_vec, Out, Weight};
use crate::dsv41::moe::{Experts, Moe, Route};
use crate::dsv41::ops::{rmsnorm, Rope};
use crate::dsv41::safetensors::StIndex;

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
    /// Whether a decode step's mixing is made on the device where it holds the projections (true unless
    /// OAIY_DSV41_HOST_MIXING=1, or [`Model::set_device_mixing`]): within roundings of the host's.
    device_mixing: bool,
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
        self.forward_impl(ids, start_pos, teacher, trace, &mut Rows::last())
    }
}

/// What a forward hands back of its rows: the last one's logits or every row's, and for a drafter what it reads of the
/// trunk at the layers `targets`, of the rows from `from` on ([`Model::forward_rows`]).
struct Rows<'a> {
    all: bool,
    targets: &'a [usize],
    from: usize,
    features: Vec<f32>,
}

impl Rows<'_> {
    fn last() -> Self {
        Rows { all: false, targets: &[], from: 0, features: Vec::new() }
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
        // (an EXL3 checkpoint's records hold its projections as stored, the largest layer's size each)
        let format = store.format().clone();
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
                pool: Some(crate::dsv41::cpu_experts::CpuExperts::new(0)),
                kernel: crate::dsv41::cpu_experts::fp4_rows,
                tokens: crate::dsv41::cpu_experts::fp4_tokens,
                gpu: None,
                store: Arc::new(store),
                cache: Arc::new(Ecache::new(opts.expert_cache_bytes, format.record_bytes(), CachePolicy::Lfru)),
                refill: crate::dsv41::moe::Refill::default(),
                uses: crate::dsv41::moe::Uses::new(cfg.n_layers, cfg.n_routed_experts),
                format,
            },
            max_seq: opts.max_seq,
            device_mixing: std::env::var("OAIY_DSV41_HOST_MIXING").as_deref() != Ok("1"),
            cfg,
        })
    }

    /// Make a decode step's mixing on the device where it holds the projections (`on`), or here.
    pub fn set_device_mixing(&mut self, on: bool) {
        self.device_mixing = on;
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

    /// What the routed experts' records hold.
    pub fn expert_format(&self) -> &crate::dsv41::expert::ExpertFormat {
        &self.experts.format
    }

    /// Every routed expert's uses of late ([`crate::dsv41::moe::Uses`]).
    pub fn expert_uses(&self) -> &crate::dsv41::moe::Uses {
        &self.experts.uses
    }

    /// Run `ids` at positions `start_pos..` (a prefill at 0, then one token at
    /// a time) and return the last position's logits (f32, `[vocab]`).
    pub fn forward(&mut self, ids: &[u32], start_pos: usize, trace: Trace<'_>) -> Result<Vec<f32>> {
        self.forward_impl(ids, start_pos, None, trace, &mut Rows::last())
    }

    /// [`forward`](Self::forward) with every row's logits where `all` (`[t, vocab]`: a draft's check, each row's choice
    /// of the next token), and the rows' from `from` on mean of the four streams entering each of the layers `targets`
    /// (after their Engram), rounded to bf16 and laid end to end (`[t - from, targets.len() * dim]`: what DSpark's
    /// drafter reads of the trunk; a long prompt's last rows are all it keeps).
    pub fn forward_rows(&mut self, ids: &[u32], start_pos: usize, targets: &[usize], all: bool, from: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let from = from.min(ids.len());
        let mut rows = Rows { all, targets, from, features: vec![0.0; (ids.len() - from) * targets.len() * self.cfg.dim] };
        let logits = self.forward_impl(ids, start_pos, None, &mut |_, _| {}, &mut rows)?;
        Ok((logits, rows.features))
    }

    /// [`forward`](Self::forward), but every layer after the first starts
    /// from `teacher`'s stream instead of the previous layer's output.
    pub fn forward_teacher(&mut self, ids: &[u32], start_pos: usize, teacher: Teacher<'_>, trace: Trace<'_>) -> Result<Vec<f32>> {
        self.forward_impl(ids, start_pos, Some(teacher), trace, &mut Rows::last())
    }

    fn forward_impl(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Trace<'_>, rows: &mut Rows<'_>) -> Result<Vec<f32>> {
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
        // Every Engram layer's rows asked of its table now: they come from the drive while the layers before it run
        // (a fresh subject's step waited some 5 ms for them at the layers, a familiar one's page cache a fraction).
        let mut begun: Vec<Option<crate::dsv41::engram::Begun>> = Vec::with_capacity(self.layers.len());
        for ly in &self.layers {
            begun.push(match &ly.engram {
                Some(eg) => {
                    let hs: Vec<i64> = (0..t)
                        .flat_map(|i| hashes[(i * n_eng + eg.hash_index) * cols..(i * n_eng + eg.hash_index + 1) * cols].iter().copied())
                        .collect();
                    Some(eg.begin(&hs)?)
                }
                None => None,
            });
        }
        for l in 0..self.cfg.n_layers {
            if let (Some(teach), true) = (teacher, l > 0) {
                let (th, tp) = teach(l)?;
                if th.len() != h.len() || tp.len() != t {
                    return Err(Error::Arg(format!("teacher input for layer {l} has the wrong shape")));
                }
                h = th;
                pre_mix = tp;
            }
            if let (Some(eg), Some(rows)) = (&self.layers[l].engram, begun[l].take()) {
                let looking_up = std::time::Instant::now();
                h = eg.forward_begun(&self.cfg, &h, t, rows)?;
                crate::dsv41::profile::add(crate::dsv41::profile::Part::Engram, looking_up);
            }
            if let Some(at) = rows.targets.iter().position(|&target| target == l) {
                let width = rows.targets.len() * d;
                for i in rows.from..t {
                    let streams = &h[i * HC * d..(i + 1) * HC * d];
                    let r = i - rows.from;
                    let out = &mut rows.features[r * width + at * d..r * width + (at + 1) * d];
                    for (f, o) in out.iter_mut().enumerate() {
                        *o = crate::dsv41::formats::to_bf16((0..HC).map(|s| streams[s * d + f]).sum::<f32>() / HC as f32);
                    }
                }
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
        self.experts.settle_tiers();
        // The experts' counts of uses age as the tokens go by, and the RAM cache's own with them: both then say what
        // has been used of late.
        if t == 1 && self.experts.uses.step() {
            self.experts.cache.decay();
        }
        let first = if rows.all { 0 } else { t - 1 };
        let x: Vec<f32> = (first..t).flat_map(|i| hc::pre(&h[i * HC * d..(i + 1) * HC * d], &pre_mix[i])).collect();
        let x = rmsnorm(&x, &self.norm, self.cfg.norm_eps);
        let logits = self.head.forward(&x, t - first, Out::F32);
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
        let Model { cfg, layers, states, shared, experts, device_mixing, .. } = self;
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
            if t >= crate::dsv41::moe::PREFETCH_TOKENS {
                let stop = &stop;
                scope.spawn(move || experts.prefetch(l as u32, n_experts, stop));
            }
            // A decode step's mixing projections where a device holds them: made in the sublayer's own call there (the
            // attention's projections of x, the router's), the host making only their cheap tail; else all of it here.
            let (side_attn, side_ffn) = if t == 1 && *device_mixing { (ly.hc_attn.on_device(), ly.hc_ffn.on_device()) } else { (None, None) };
            let mixing = std::time::Instant::now();
            let attn_mix_here = side_attn.is_none().then(|| mixes(h, &ly.hc_attn));
            let collapsed: Vec<f32> = per_token(t, |i| hc::pre(&h[tok(i)], &pre_mix[i])).concat();
            crate::dsv41::profile::add(crate::dsv41::profile::Part::Mixing, mixing);
            let x = rmsnorm(&collapsed, &ly.attn_norm, eps);
            // (OAIY_DSV41_FORETELL=1: the layer's router on the attention's input too, in the same call, and how much of
            // what it then routes to that foretold, counted for the log: what an upload begun before attention could know)
            static FORETELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let router = ly.moe.router_and_shared()[0];
            let foretell = t == 1 && matches!(router, crate::dsv41::linear::Weight::Device { .. }) && *FORETELL.get_or_init(|| std::env::var("OAIY_DSV41_FORETELL").as_deref() == Ok("1"));
            // (the stream as the attention takes it, normed as the FFN norms its own input: the router's way of seeing it)
            let ffn_seen = foretell.then(|| rmsnorm(&collapsed, &ly.ffn_norm, eps));
            let mut sides: Vec<(&crate::dsv41::linear::Weight, &[f32])> = side_attn.map(|w| (w, h)).into_iter().collect();
            if let Some(seen) = &ffn_seen {
                sides.push((router, &seen[..]));
            }
            let o = ly.attn.forward_core(cfg, &x, t, start_pos, states, shared, &sides);
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let (o, mut made_sides) = o?;
            let foretold = foretell.then(|| made_sides.pop().expect("the foretelling router's logits"));
            let attn_proj = made_sides.pop();
            let mixing = std::time::Instant::now();
            let attn_mix = match (attn_mix_here, attn_proj) {
                (Some(m), _) => m,
                (None, proj) => vec![hc::mixes_of(&proj.expect("the attention's mixing projection"), h, &ly.hc_attn, eps, cfg.hc_sinkhorn_iters, cfg.hc_eps)],
            };
            crate::dsv41::profile::add(crate::dsv41::profile::Part::Mixing, mixing);
            // A decode step's output projection, the hyper-connection's spread of it and the FFN's input (its collapse and
            // norm), and the router's, the shared expert's and the mixing's projections of those, in one program on the
            // device where it holds them all: the output's call and the router's were a round trip each, and the host's
            // mixing and norm between them. Else each as before.
            let program = side_ffn.and_then(|fw| {
                use crate::dsv41::linear::Step;
                let flat = attn_mix[0].flat();
                let mut steps = Vec::new();
                let a = ly.attn.output_steps(cfg, &o, &mut steps)?;
                let (hi, mi) = (steps.len(), steps.len() + 1);
                steps.push(Step::Input(h));
                steps.push(Step::Input(&flat));
                let h1 = steps.len();
                steps.push(Step::HcPost { out: a, res: hi, mix: mi, d });
                steps.push(Step::HcPre { h: h1, mix: mi, d });
                let xn = steps.len();
                steps.push(Step::Norm { x: h1 + 1, weight: &ly.ffn_norm, eps, rows: 1, width: d });
                let [router, w1, w3, w2] = ly.moe.router_and_shared();
                let made = steps.len();
                for w in [router, w1, w3] {
                    steps.push(Step::Mul { w, x: xn, rows: 1 });
                }
                steps.push(Step::Mul { w: fw, x: h1, rows: 1 });
                // (the shared expert whole: its activation the host's to the bit, its down projection's sums read back)
                let act = steps.len();
                let limit = cfg.swiglu_limit;
                steps.push(Step::SwigluBf16 { gate: made + 1, up: made + 2, len: w1.n(), silu: crate::dsv41::moe::silu_bf16_table(limit), limit });
                steps.push(Step::Mul { w: w2, x: act, rows: 1 });
                let reads = [(a, 0, d), (h1, 0, HC * d), (xn, 0, d), (made, 0, router.n()), (made + 3, 0, fw.n()), (act + 1, 0, w2.n())];
                // The layer's held experts the router likely picks, where the experts' kernel holds them beside the layer's
                // weights: made behind this call, in a recording submitted after it, the host routing and making the rest
                // meanwhile (the call each layer's held experts had of their own is not made for them).
                let k = crate::dsv41::moe::FUSED_CANDIDATES;
                match experts.gpu.as_ref().and_then(|g| g.fused(l as u32)) {
                    Some(held) => {
                        let cands = steps.len();
                        steps.push(Step::Candidates { logits: made, bias: ly.moe.bias(), held, layer: l as u32, k });
                        steps.push(Step::Held { held, layer: l as u32, x: xn, cands, limit });
                        let (now, later) = crate::dsv41::linear::run_program_parts(router, &steps, &reads, cands, &[(cands, 2 * k, 2 * k), (cands + 1, 0, k * d)])?;
                        let later: crate::dsv41::moe::FusedLater<'_> = Box::new(move || {
                            let r = later();
                            crate::dsv41::moe::Fused::from_read(&r[0], &r[1], k, d)
                        });
                        Some((now, Some(later)))
                    }
                    None => crate::dsv41::linear::run_program(&steps, &reads).map(|now| (now, None)),
                }
            });
            let (a, h1, x, made) = match program {
                Some((p, fused)) => {
                    let [a, h1, x, logits, proj, shared]: [Vec<f32>; 6] = p.try_into().expect("six vectors read back");
                    let shared: Vec<f32> = shared.into_iter().map(crate::dsv41::formats::to_bf16).collect();
                    (a, h1, x, Some((logits, shared, proj, fused)))
                }
                None => {
                    let a = ly.attn.output(cfg, &o, t);
                    let mixing = std::time::Instant::now();
                    let h1: Vec<f32> = per_token(t, |i| hc::post(&a[i * d..(i + 1) * d], &h[tok(i)], &attn_mix[i])).concat();
                    let x: Vec<f32> = per_token(t, |i| hc::pre(&h1[tok(i)], &attn_mix[i].pre)).concat();
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::Mixing, mixing);
                    let x = rmsnorm(&x, &ly.ffn_norm, eps);
                    (a, h1, x, None)
                }
            };
            trace(&format!("layer{l:02}.attn_out"), &a);
            let ffn_mix_here = side_ffn.is_none().then(|| mixes(&h1, &ly.hc_ffn));
            let (m, routes, ffn_proj) = match made {
                Some((logits, shared, proj, fused)) => {
                    let (m, routes) = ly.moe.forward_from(cfg, &x, t, experts, logits, None, Some(shared), fused)?;
                    (m, routes, Some(proj))
                }
                None => ly.moe.forward_side(cfg, &x, t, experts, side_ffn.map(|w| (w, &h1[..])))?,
            };
            let ffn_mix = match (ffn_mix_here, ffn_proj) {
                (Some(m), _) => m,
                (None, proj) => vec![hc::mixes_of(&proj.expect("the router's mixing projection"), &h1, &ly.hc_ffn, eps, cfg.hc_sinkhorn_iters, cfg.hc_eps)],
            };
            if let (Some(logits), Some(r)) = (&foretold, routes.first()) {
                crate::dsv41::profile::foretold(&r.experts, &ly.moe.ranked(logits, 8));
            }
            trace(&format!("layer{l:02}.moe_out"), &m);
            trace(&format!("layer{l:02}.route_ids"), &routes.iter().flat_map(|r| r.experts.iter().map(|&e| e as f32)).collect::<Vec<_>>());
            trace(&format!("layer{l:02}.route_w"), &routes.iter().flat_map(|r| r.weights.iter().copied()).collect::<Vec<_>>());
            let mixing = std::time::Instant::now();
            let h2: Vec<f32> = per_token(t, |i| hc::post(&m[i * d..(i + 1) * d], &h1[tok(i)], &ffn_mix[i])).concat();
            crate::dsv41::profile::add(crate::dsv41::profile::Part::Mixing, mixing);
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

    /// What a pass of `t` rows from `start` writes over in every layer's caches ([`Model::rewind`]): a draft's check,
    /// of which only the rows up to the first the model chose otherwise stand. A few slots a layer, where a
    /// [`Checkpoint`] is every cache whole. The n-gram history needs nothing: it is held by position.
    pub fn save(&self, start: usize, t: usize) -> Vec<crate::dsv41::attention::Saved> {
        self.states.iter().map(|s| s.save(self.cfg.head_dim, start, t)).collect()
    }

    /// The caches as they would be had the pass `saved` was taken for read only its first `keep` rows: the next
    /// forward continues from there.
    pub fn rewind(&mut self, saved: &[crate::dsv41::attention::Saved], keep: usize) {
        for (l, (state, saved)) in self.states.iter_mut().zip(saved).enumerate() {
            state.rewind(self.cfg.head_dim, saved, keep, self.cfg.ratio(l));
        }
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

    /// [`Self::warm_experts`] in `order` (every expert once: [`crate::dsv41::moe::Uses::order`]'s, the most used of an
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
    /// crate `dsv41::simd` picks this CPU's fastest, which this crate may define but not call (it forbids `unsafe`).
    /// Every kernel gives the same bits.
    pub fn set_expert_row_kernel(&mut self, kernel: crate::dsv41::cpu_experts::RowKernel) {
        self.experts.pool = Some(crate::dsv41::cpu_experts::CpuExperts::with_kernel(0, kernel));
        self.experts.kernel = kernel;
    }

    /// Run a prompt's routed experts on the CPU through `kernel` (all of an expert's tokens at once), as
    /// [`Self::set_expert_row_kernel`] sets a decode step's: `dsv41::simd` picks this CPU's. The same bits each.
    pub fn set_expert_tokens_kernel(&mut self, kernel: crate::dsv41::cpu_experts::TokensKernel) {
        self.experts.tokens = kernel;
    }

    /// Have a prompt's busy routed experts' matmuls made by `kernel` (a GPU's: see [`crate::dsv41::moe::Experts::gpu`]).
    pub fn set_experts_kernel(&mut self, kernel: Option<Arc<dyn crate::dsv41::expert::ExpertsKernel>>) {
        self.experts.gpu = kernel;
    }

    /// Hand the trunk's dense weights to `place` (a GPU's), each with its name (`layers.<l>.attn.wq_b`, `head`): the
    /// weight it returns replaces the one it was given (a [`Weight::Device`]), None leaves it here. The embedding stays
    /// (a prompt reads rows of it). How many were placed, and their bytes as stored.
    pub fn offload(&mut self, mut place: impl FnMut(&str, &Weight) -> Option<Weight>) -> (usize, u64) {
        let mut named: Vec<(String, &mut Weight)> = Vec::new();
        for (l, layer) in self.layers.iter_mut().enumerate() {
            for (name, w) in layer.attn.weights_mut() {
                named.push((format!("layers.{l}.attn.{name}"), w));
            }
            for (name, w) in layer.moe.weights_mut() {
                named.push((format!("layers.{l}.ffn.{name}"), w));
            }
            if let Some(e) = &mut layer.engram {
                named.push((format!("layers.{l}.engram.wkv"), e.weight_mut()));
            }
            named.push((format!("layers.{l}.hc_attn_fn"), layer.hc_attn.weight_mut()));
            named.push((format!("layers.{l}.hc_ffn_fn"), layer.hc_ffn.weight_mut()));
        }
        named.push(("head".into(), &mut self.head));
        let placed = place_all(named, &mut place);
        // (a mixing projection no device took is made here from the host's own copy: its stored one let go)
        for layer in &mut self.layers {
            for w in [layer.hc_attn.weight_mut(), layer.hc_ffn.weight_mut()] {
                if let Weight::F32 { w, k, .. } = w {
                    *w = Vec::new();
                    *k = 0;
                }
            }
        }
        placed
    }

    /// The embedding (a drafter's rows are made of it too).
    pub fn embed(&self) -> &Weight {
        &self.embed
    }

    /// The output head (a drafter's logits are made by it too).
    pub fn head(&self) -> &Weight {
        &self.head
    }
}

/// [`Model::offload`]'s placing of `named` weights: each given to `place` with its name, the weight it returns put in its
/// place. A stack's parts are placed one by one (an EXL3 checkpoint's `wo_a`, a slice a group), by their own names,
/// after the stack is offered whole (`{name}.group`: a device that takes it so keeps its group, which a step's
/// program then multiplies in one pass). How many were placed, and their bytes as stored.
pub(crate) fn place_all(named: Vec<(String, &mut Weight)>, place: &mut impl FnMut(&str, &Weight) -> Option<Weight>) -> (usize, u64) {
    let (mut count, mut bytes) = (0usize, 0u64);
    for (name, w) in named {
        if matches!(w, Weight::Stacked { group: None, .. }) {
            let made = place(&format!("{name}.group"), w).map(Box::new);
            if let Weight::Stacked { group, .. } = w {
                *group = made;
            }
        }
        let parts: Vec<(String, &mut Weight)> = match w {
            Weight::Stacked { parts, .. } => parts.iter_mut().enumerate().map(|(i, p)| (format!("{name}.slice.{i}"), p)).collect(),
            w => vec![(name, w)],
        };
        for (name, w) in parts {
            let size = match w {
                Weight::Fp8 { w, s, .. } => (w.len() + s.len()) as u64,
                Weight::Bf16 { w, .. } => w.len() as u64 * 2,
                Weight::F32 { w, .. } => w.len() as u64 * 4,
                Weight::Exl3 { data, .. } => data.words.len() as u64 * 4 + (data.suh.len() + data.svh.len()) as u64 * 2,
                Weight::Device { .. } | Weight::Stacked { .. } => continue,
            };
            if let Some(new) = place(&name, w) {
                *w = new;
                count += 1;
                bytes += size;
            }
        }
    }
    (count, bytes)
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
