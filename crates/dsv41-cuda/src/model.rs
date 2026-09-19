//! The DeepSeek-V4.1 backbone on one or more GPUs (Phase C1-C3): the trunk
//! is resident in its stored dtypes (fp8 stays fp8), every matmul, norm,
//! rotation, attention and expert runs on a device, and routed experts live
//! in a per-device VRAM cache fed from nrob's host cache.
//!
//! With several devices the layers are split into contiguous ranges (C3):
//! each layer's weights, KV caches, RoPE tables and experts live on its
//! device, and only the hidden stream (4 x 5120 floats per token) hops
//! between devices, through the host, once per boundary. The split 0-19 /
//! 20-39 keeps every cross-layer hand-off on one device: kv/index sources
//! 2/8/14 and their readers are all below 20, source 20 and the later index
//! sources all above.
//!
//! The host keeps the small decisions, using the very functions the CPU
//! model uses: expert routing (`moe::route`), the indexer's top-k and
//! candidate blocks (`attention::select_compressed`), n-gram hashing and
//! Engram row reads (started on a background thread at the top of each
//! forward). Each costs a few KB of transfer per layer. The hyper-connection
//! mixes, Sinkhorn included, run on the device.
//!
//! Decode is hybrid: routed experts resident in VRAM run in grouped kernels,
//! the others on the CPU (`dsv41::cpu_experts`) straight from the host cache
//! while the GPU works, unless they are used often enough to earn a VRAM
//! slot. A saved usage profile ([`GpuModel::save_usage`] / [`GpuModel::warm`])
//! carries the hot set from one run to the next.
//!
//! Numerics follow the CPU reference step for step (the kernels are tested
//! against it one by one, `tests/kernels.rs`), so the same golden-file suite
//! validates this model (`tests/gpu_model.rs`).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaView};
use nrob::ecache::Ecache;
use nrob::store::WeightStore;
use nrob::{CachePolicy, Error, Result};

use dsv41::attention::{compressed_pos, select_compressed, CandidateRole, Shared};
use dsv41::config::Config;
use dsv41::cpu_experts::CpuExperts;
use dsv41::engram::{Engram, NgramHasher};
use dsv41::expert::{SafetensorsExpertStore, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1, W2, W3};
use dsv41::hc::{HcParams, HC};
use dsv41::linear::{load_vec, Out, Weight};
use dsv41::model::{Backbone, Teacher, Trace};
use dsv41::moe::{route_mixed, Route};
use dsv41::ops::Rope;
use dsv41::safetensors::StIndex;
use dsv41::vision::Prepared;

use crate::expert_cache::DeviceExpertCache;
use crate::gpu::{Gpu, Pos};
use crate::vision::GpuVision;

pub struct GpuOptions {
    /// CUDA device ordinals; layers are split evenly across them in order
    /// (the embedding side on the first, the head on the last).
    pub devices: Vec<usize>,
    /// Longest sequence (prompt + generation); sizes every cache.
    pub max_seq: usize,
    /// Host RAM for the routed-expert cache.
    pub expert_cache_bytes: usize,
    /// Bypass the page cache for expert reads.
    pub direct_io: bool,
    /// VRAM for each device's expert cache; `None` = whatever that device's
    /// share of the trunk leaves free, minus `vram_headroom_bytes`.
    pub vram_expert_bytes: Option<usize>,
    /// VRAM left free for activations when `vram_expert_bytes` is `None`
    /// (1 GiB suits decode and 1k-token prefill chunks; layered prefill of
    /// long prompts wants more).
    pub vram_headroom_bytes: usize,
    /// Hybrid decode: a routed expert that misses VRAM runs on this many CPU
    /// threads straight from the host cache, unless it is used more often
    /// than the VRAM victim (then it is uploaded, at most
    /// [`PROMOTE_PER_LAYER`] per layer per token). `None`: every miss is
    /// uploaded. `Some(0)`: one thread per hardware thread.
    pub cpu_expert_threads: Option<usize>,
    /// Load the vision tower (about 1 GB on the first device, taken before
    /// its expert cache is sized) so prompts can hold images.
    pub vision: bool,
}

/// An image in the sequence: from absolute position `start`, `rows.len() /
/// dim` tokens take these rows as their input embedding in place of their
/// token's ([`GpuModel::encode_image`] makes them), route with the vision
/// bias and take no part in Engram n-grams. A span may begin before a
/// stretch and end after it; each forward uses the part it covers.
#[derive(Clone, Debug)]
pub struct ImageSpan {
    pub start: usize,
    pub rows: Vec<f32>,
}

/// Each token's image row for positions `start..start + t`, where it is one.
fn image_rows(images: &[ImageSpan], start: usize, t: usize, d: usize) -> Result<Vec<Option<&[f32]>>> {
    let mut rows = vec![None; t];
    for sp in images {
        if sp.rows.len() % d != 0 {
            return Err(Error::Arg("image span rows are not whole embeddings".into()));
        }
        let n = sp.rows.len() / d;
        for p in sp.start.max(start)..(sp.start + n).min(start + t) {
            rows[p - start] = Some(&sp.rows[(p - sp.start) * d..(p - sp.start + 1) * d]);
        }
    }
    Ok(rows)
}

/// An Engram layer's rows for a stretch (`hashes` `[t][cols]`) with the
/// image tokens' left zero: the reference shuts their gate, and a zero row
/// makes the gate here add exactly nothing (its key and value are 0).
fn engram_rows(eg: &Engram, hashes: &[i64], head_dim: usize, cols: usize, skip: Option<&[bool]>) -> Result<Vec<f32>> {
    let Some(skip) = skip else { return eg.rows(hashes, head_dim) };
    let text: Vec<i64> = hashes.chunks_exact(cols).zip(skip).filter(|(_, &s)| !s).flat_map(|(h, _)| h.iter().copied()).collect();
    let w = cols * head_dim;
    let mut out = vec![0.0f32; skip.len() * w];
    if text.is_empty() {
        return Ok(out);
    }
    let rows = eg.rows(&text, head_dim)?;
    let mut next = rows.chunks_exact(w);
    for (dst, &s) in out.chunks_exact_mut(w).zip(skip) {
        if !s {
            dst.copy_from_slice(next.next().expect("one row per text token"));
        }
    }
    Ok(out)
}

/// VRAM uploads a hybrid decode step may make per layer: each is a
/// synchronous PCIe copy of 18.8 MB (1.3-2.6 ms on the dev machine).
pub const PROMOTE_PER_LAYER: usize = 1;

/// The per-sequence state after some tokens (see
/// [`GpuModel::checkpoint`]): each layer's window ring and, on
/// ratio-2 compressors, its partial group, held in host memory (~10 MB).
pub struct Checkpoint {
    pos: usize,
    layers: Vec<LayerSnapshot>,
}

/// One layer's window ring, and its compressor's partial group (kv, score).
type LayerSnapshot = (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>);

impl Checkpoint {
    /// Tokens the state covers.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// Wall time per phase, summed over forwards (seconds). Enabled by
/// [`GpuModel::enable_profile`] (the generate example: `DSV41_PROFILE=1`);
/// each phase boundary then synchronizes the device, so profiled runs are a
/// little slower than unprofiled ones.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub forwards: u64,
    /// Hyper-connection mixes, pre and post.
    pub hc: f64,
    pub attn: f64,
    /// Router GEMV, logits download, host top-k.
    pub route: f64,
    /// Expert records to the device (VRAM hit: nothing; miss: host cache or disk, then PCIe).
    pub fetch: f64,
    /// Routed expert kernels.
    pub experts: f64,
    pub shared: f64,
    /// Routed experts on the CPU (hybrid decode), host wall time; overlaps
    /// the GPU's resident experts and shared expert.
    pub cpu: f64,
    /// Routed-expert uses served on the CPU.
    pub cpu_uses: u64,
    pub engram: f64,
    /// Final norm, head GEMV, logits download.
    pub head: f64,
}

/// Tokens over which a saved usage profile's older counts halve (see
/// [`GpuModel::save_usage`]).
pub const USAGE_HALF_LIFE: u64 = 200_000;

/// `layer expert count` lines of a usage profile.
fn parse_usage(text: &str) -> impl Iterator<Item = (usize, usize, u64)> + '_ {
    text.lines().filter(|l| !l.starts_with('#')).filter_map(|line| {
        let v: Vec<u64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        match v[..] {
            [l, e, n] => Some((l as usize, e as usize, n)),
            _ => None,
        }
    })
}

/// Count each token's routed experts of layer `l` into `usage`.
fn note_usage(usage: &mut [u32], n_exp: usize, l: usize, routes: &[Route]) {
    for r in routes {
        for &e in &r.experts {
            let c = &mut usage[l * n_exp + e as usize];
            *c = c.saturating_add(1);
        }
    }
}

/// Where a layered pass's routed experts came from, counted without
/// synchronizing (for benchmarks): experts used (distinct (layer, expert)
/// pairs), how many of them were in VRAM or RAM when the pass began and
/// when their layer ran, and the records read from the drive meanwhile.
#[derive(Clone, Debug, Default)]
pub struct PassStats {
    pub used: usize,
    pub resident_at_start: usize,
    pub resident_at_use: usize,
    pub vram_at_use: usize,
    pub reads: u64,
}

/// A decode step's experts on the CPU: none, still to run on this thread
/// (profiling, so their time is their own), or already handed to the CPU
/// pool, whose rows the reduction reads from the device's hand-off.
enum CpuPart {
    None,
    Blocking(Vec<(usize, nrob::ecache::HostLease, f32)>, Vec<f32>),
    Launched { mask: u32, seq: u32, uses: u64 },
}

/// Routed-expert results before the shared expert is added: one row per
/// expert slot (grouped decode) or already summed per token (prefill).
enum MoeOut {
    Grouped(CudaSlice<f32>, usize),
    Summed(CudaSlice<f32>),
}

/// A dense weight on the device, in its stored form.
enum DW {
    Fp8 { w: CudaSlice<u8>, s: CudaSlice<u8>, n: usize, k: usize },
    Bf16 { w: CudaSlice<u16>, n: usize, k: usize },
}

impl DW {
    fn upload(g: &Gpu, w: &Weight) -> Result<DW> {
        Ok(match w {
            Weight::Fp8 { w, s, n, k } => DW::Fp8 { w: g.upload(w)?, s: g.upload(s)?, n: *n, k: *k },
            Weight::Bf16 { w, n, k } => DW::Bf16 { w: g.upload(w)?, n: *n, k: *k },
            Weight::F32 { .. } => return Err(Error::Unsupported("f32 dense weight on the device".into())),
        })
    }

    fn load(g: &Gpu, idx: &StIndex, prefix: &str) -> Result<DW> {
        Self::upload(g, &Weight::load(idx, prefix)?)
    }

    fn n(&self) -> usize {
        match self {
            DW::Fp8 { n, .. } | DW::Bf16 { n, .. } => *n,
        }
    }

    /// The reference `linear()`: fp8 quantizes the activation and returns bf16.
    fn forward(&self, g: &Gpu, x: &CudaView<'_, f32>, t: usize, out: Out) -> Result<CudaSlice<f32>> {
        // one token: fuse the quantizer, except for very long rows, where
        // every block quantizing its own copy of x costs more than the extra
        // launch (measured: wo_b, k = 8192)
        if let (DW::Fp8 { w, s, n, k }, 1, true) = (self, t, matches!(self, DW::Fp8 { k, .. } if *k < 8192)) {
            if let Some(y) = g.gemv_fp8_token(x, w, s, *n, *k, true, true)? {
                return Ok(y);
            }
        }
        let mut y = g.alloc::<f32>(t * self.n())?;
        match self {
            DW::Fp8 { w, s, n, k } => {
                let xq = g.act_quant_fp8_to(x)?;
                // several tokens (prefill) on tensor cores: a token's result
                // does not depend on how many share the launch
                if t > 1 && k % 32 == 0 {
                    g.gemm_fp8(&xq.as_view(), w, s, &mut y.slice_mut(..), *n, *k, t, true)?;
                } else {
                    g.gemv_fp8(&xq.as_view(), w, s, &mut y.slice_mut(..), *n, *k, t, true)?;
                }
            }
            DW::Bf16 { w, n, k } => g.gemv_bf16(x, w, &mut y.slice_mut(..), *n, *k, t, out == Out::Bf16, 0, 0)?,
        }
        Ok(y)
    }
}

struct GCompressor {
    ratio: usize,
    wkv: DW,
    wgate: Option<DW>,
    norm: CudaSlice<f32>,
}

struct GIndexer {
    role: CandidateRole,
    wq_b: DW,
    weights_proj: DW,
    wk: Option<DW>,
    k_norm: Option<CudaSlice<f32>>,
}

struct GLayer {
    /// Index into `GpuModel::devs`.
    dev: usize,
    ratio: usize,
    wq_a: DW,
    q_norm: CudaSlice<f32>,
    wq_b: DW,
    wkv: DW,
    kv_norm: CudaSlice<f32>,
    /// fp8 as stored, with its 32x32 tile scales (see Gpu::gemv_fp8w).
    wo_a: (CudaSlice<u8>, CudaSlice<u8>),
    wo_b: DW,
    sink: CudaSlice<f32>,
    compressor: Option<GCompressor>,
    indexer: Option<GIndexer>,
    attn_norm: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    hc_attn: HcDev,
    hc_ffn: HcDev,
    gate: DW,
    bias: Vec<f32>,
    /// The routing bias for image tokens (`gate.bias_vl`).
    bias_vl: Option<Vec<f32>>,
    shared: [DW; 3],
    engram: Option<(Arc<Engram>, DW, CudaSlice<f32>)>,
}

struct GState {
    window: CudaSlice<f32>,
    compress_kv: Option<CudaSlice<f32>>,
    kv_state: Option<CudaSlice<f32>>,
    score_state: Option<CudaSlice<f32>>,
    index_k: Option<CudaSlice<f32>>,
}

/// One sublayer's hyper-connection parameters on the device.
struct HcDev {
    proj: CudaSlice<f32>,
    base: CudaSlice<f32>,
    scale: CudaSlice<f32>,
}

/// Per-token mix coefficients stay on the device as `[t][24]` =
/// {pre[4], post[4], comb[16]} (the hc_mix layout); `pre` alone is read with
/// stride 24.
const MIXW: usize = 24;

/// One device and what lives on it besides its layers.
struct Dev {
    /// Declared first so it is freed before the context (jobs share it by Arc).
    handoff: Arc<crate::handoff::Handoff>,
    /// Last hand-off sequence number used on this device.
    cpu_seq: u32,
    /// Router logits and activation, published by the device (decode).
    inbox: crate::handoff::Inbox,
    inbox_seq: u32,
    /// Per-token tickets of hc_project_mix (all zero between launches).
    mix_counter: CudaSlice<u32>,
    g: Gpu,
    rope_window: (CudaSlice<f32>, CudaSlice<f32>),
    rope_compress: (CudaSlice<f32>, CudaSlice<f32>),
    dcache: DeviceExpertCache,
}

pub struct GpuModel {
    pub cfg: Config,
    devs: Vec<Dev>,
    /// Device the hidden stream currently lives on.
    cur: usize,
    embed: Weight,
    norm: CudaSlice<f32>,
    head: DW,
    layers: Vec<GLayer>,
    states: Vec<GState>,
    hasher: NgramHasher,
    max_seq: usize,
    vision: Option<GpuVision>,
    /// The running forward's image tokens (empty: none).
    image_mask: Vec<bool>,
    /// Images for [`Backbone::forward_traced`] (whose signature has none).
    trace_images: Vec<ImageSpan>,
    /// The last layered pass's expert sources.
    pass_stats: PassStats,
    /// Routed-expert uses since the profile was last saved, `[layer][expert]`,
    /// and the tokens they came from.
    usage: Vec<u32>,
    usage_tokens: u64,
    // cross-layer hand-offs (reference SharedAttentionRuntime)
    shared: Shared,
    compress_src: Option<usize>,
    index_k_src: Option<usize>,
    topk: Vec<Vec<i32>>,
    // experts
    store: Arc<dyn WeightStore>,
    cache: Arc<Ecache>,
    cpu: Option<CpuExperts>,
    /// Background fill of the host cache started by [`warm`](Self::warm).
    warming: Option<Warming>,
    /// When the background fill must give way to demand reads.
    demand: Arc<Demand>,
    profile: Option<Profile>,
}

impl GpuModel {
    pub fn load(model_dir: &Path, engram_meta: &Path, opts: &GpuOptions) -> Result<GpuModel> {
        let cfg = Config::load(model_dir)?;
        let idx = StIndex::open(model_dir)?;
        if opts.devices.is_empty() {
            return Err(Error::Arg("no CUDA devices given".into()));
        }
        let gpus: Vec<Gpu> = opts.devices.iter().map(|&o| Gpu::new(o)).collect::<Result<_>>()?;
        let n_dev = gpus.len();
        // contiguous, even split: layer l lives on device l * n_dev / n_layers
        let dev_of = |l: usize| l * n_dev / cfg.n_layers;
        // The cross-layer hand-offs (compressed KV, index keys, top-k lists,
        // candidate blocks) never cross devices only if every device after
        // the first starts at a kv-source layer, which re-publishes all of
        // them before any later layer reads them. 2 devices split at 20: fine.
        for l in 1..cfg.n_layers {
            if dev_of(l) != dev_of(l - 1) && !cfg.kv_source_layers.contains(&l) {
                return Err(Error::Arg(format!(
                    "{n_dev} devices would split the layers at {l}, which is not a kv-source layer {:?}; \
                     the attention hand-offs would cross devices",
                    cfg.kv_source_layers
                )));
            }
        }
        let vec = |g: &Gpu, name: &str| -> Result<CudaSlice<f32>> { g.upload(&load_vec(&idx, name)?) };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut states = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let dev = dev_of(l);
            let g = &gpus[dev];
            let p = format!("layers.{l}");
            let a = format!("{p}.attn");
            let ratio = cfg.ratio(l);
            let kv_source = cfg.kv_source_layers.contains(&l);
            let index_source = cfg.index_source_layers.contains(&l);
            let compressor = if kv_source {
                Some(GCompressor {
                    ratio,
                    wkv: DW::load(g, &idx, &format!("{a}.compressor.wkv"))?,
                    wgate: if ratio > 1 { Some(DW::load(g, &idx, &format!("{a}.compressor.wgate"))?) } else { None },
                    norm: vec(g, &format!("{a}.compressor.norm.weight"))?,
                })
            } else {
                None
            };
            let indexer = if index_source {
                let role = if cfg.candidate_source_layer == Some(l) {
                    CandidateRole::Source
                } else if cfg.candidate_source_layer.is_some_and(|c| c < l) {
                    CandidateRole::User
                } else {
                    CandidateRole::None
                };
                Some(GIndexer {
                    role,
                    wq_b: DW::load(g, &idx, &format!("{a}.indexer.wq_b"))?,
                    weights_proj: DW::load(g, &idx, &format!("{a}.indexer.weights_proj"))?,
                    wk: if kv_source { Some(DW::load(g, &idx, &format!("{a}.indexer.wk"))?) } else { None },
                    k_norm: if kv_source { Some(vec(g, &format!("{a}.indexer.k_norm.weight"))?) } else { None },
                })
            } else {
                None
            };
            let hcp = |which: &str| -> Result<HcDev> {
                let hp = HcParams::load(&idx, &p, which)?;
                let (base, scale) = hp.base_and_scale();
                Ok(HcDev { proj: g.upload(hp.projection())?, base: g.upload(base)?, scale: g.upload(scale)? })
            };
            let wo_a = match Weight::load(&idx, &format!("{a}.wo_a"))? {
                Weight::Fp8 { w, s, .. } => (g.upload(&w)?, g.upload(&s)?),
                _ => return Err(Error::Format("wo_a is not fp8".into())),
            };
            let engram = if cfg.engram_layer_ids.contains(&l) {
                let e = Engram::load(&idx, &cfg, l)?;
                let wkv = DW::upload(g, e.wkv())?;
                let qk = g.upload(e.qk())?;
                Some((Arc::new(e), wkv, qk))
            } else {
                None
            };
            let (hd, r) = (cfg.head_dim, ratio.max(1));
            states.push(GState {
                window: g.zeros(cfg.window_size * hd)?,
                compress_kv: if kv_source { Some(g.zeros(opts.max_seq / r * hd)?) } else { None },
                kv_state: if kv_source && ratio > 1 { Some(g.zeros(ratio * hd)?) } else { None },
                score_state: if kv_source && ratio > 1 { Some(g.upload(&vec![f32::NEG_INFINITY; ratio * hd])?) } else { None },
                index_k: if kv_source && index_source { Some(g.zeros(opts.max_seq / r * cfg.index_head_dim)?) } else { None },
            });
            layers.push(GLayer {
                dev,
                ratio,
                wq_a: DW::load(g, &idx, &format!("{a}.wq_a"))?,
                q_norm: vec(g, &format!("{a}.q_norm.weight"))?,
                wq_b: DW::load(g, &idx, &format!("{a}.wq_b"))?,
                wkv: DW::load(g, &idx, &format!("{a}.wkv"))?,
                kv_norm: vec(g, &format!("{a}.kv_norm.weight"))?,
                wo_a,
                wo_b: DW::load(g, &idx, &format!("{a}.wo_b"))?,
                sink: vec(g, &format!("{a}.attn_sink"))?,
                compressor,
                indexer,
                attn_norm: vec(g, &format!("{p}.attn_norm.weight"))?,
                ffn_norm: vec(g, &format!("{p}.ffn_norm.weight"))?,
                hc_attn: hcp("attn")?,
                hc_ffn: hcp("ffn")?,
                gate: DW::load(g, &idx, &format!("{p}.ffn.gate"))?,
                bias: load_vec(&idx, &format!("{p}.ffn.gate.bias"))?,
                bias_vl: {
                    let name = format!("{p}.ffn.gate.bias_vl");
                    if idx.info(&name).is_ok() {
                        Some(load_vec(&idx, &name)?)
                    } else {
                        None
                    }
                },
                shared: [
                    DW::load(g, &idx, &format!("{p}.ffn.shared_experts.w1"))?,
                    DW::load(g, &idx, &format!("{p}.ffn.shared_experts.w2"))?,
                    DW::load(g, &idx, &format!("{p}.ffn.shared_experts.w3"))?,
                ],
                engram,
            });
        }
        // the output side lives on the last device
        let last = &gpus[n_dev - 1];
        let head = DW::load(last, &idx, "head")?;
        let norm = vec(last, "norm.weight")?;
        let rope = |orig, theta| Rope::new(cfg.rope_head_dim, opts.max_seq, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow);
        let (rw, rc) = (rope(0, cfg.rope_theta), rope(cfg.original_seq_len, cfg.compress_rope_theta));
        // the vision tower before the expert caches size themselves from what is free
        let vision = if opts.vision && cfg.vision.is_some() { Some(GpuVision::load(&gpus[0], &idx, &cfg)?) } else { None };
        let mut devs = Vec::with_capacity(n_dev);
        for g in gpus {
            let up = |r: &Rope| -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
                let (c, s) = r.tables();
                Ok((g.upload(c)?, g.upload(s)?))
            };
            let (rope_window, rope_compress) = (up(&rw)?, up(&rc)?);
            // expert slots from whatever this device's share of the trunk left free
            let vram = match opts.vram_expert_bytes {
                Some(b) => b,
                None => g.mem_info()?.0.saturating_sub(opts.vram_headroom_bytes),
            };
            let dcache = DeviceExpertCache::new(&g, (vram / RECORD_BYTES).max(1), RECORD_BYTES)?;
            let handoff = Arc::new(crate::handoff::Handoff::new(&g, cfg.n_activated_experts, cfg.dim)?);
            let inbox = crate::handoff::Inbox::new(&g, cfg.n_routed_experts + cfg.dim)?;
            let mix_counter = g.zeros::<u32>(opts.max_seq)?;
            devs.push(Dev { handoff, cpu_seq: 0, inbox, inbox_seq: 0, mix_counter, g, rope_window, rope_compress, dcache });
        }
        let store = SafetensorsExpertStore::open(&idx, cfg.n_layers as u32, cfg.n_routed_experts as u32, opts.direct_io)?;
        Ok(GpuModel {
            embed: Weight::load(&idx, "embed")?,
            norm,
            head,
            hasher: NgramHasher::load(engram_meta, &cfg, opts.max_seq)?,
            devs,
            cur: 0,
            profile: None,
            layers,
            states,
            max_seq: opts.max_seq,
            vision,
            image_mask: Vec::new(),
            trace_images: Vec::new(),
            pass_stats: PassStats::default(),
            usage: vec![0; cfg.n_layers * cfg.n_routed_experts],
            usage_tokens: 0,
            shared: Shared::default(),
            compress_src: None,
            index_k_src: None,
            topk: Vec::new(),
            store: Arc::new(store),
            cache: Arc::new(Ecache::new(opts.expert_cache_bytes, RECORD_BYTES, CachePolicy::Lfru)),
            warming: None,
            demand: Arc::new(Demand::new()),
            cpu: opts.cpu_expert_threads.map(|n| CpuExperts::with_kernel(n, crate::cpu::row_kernel())),
            cfg,
        })
    }

    pub fn expert_cache(&self) -> &Ecache {
        &self.cache
    }

    /// Each device's VRAM expert cache, in device order.
    pub fn device_caches(&self) -> impl Iterator<Item = &DeviceExpertCache> {
        self.devs.iter().map(|d| &d.dcache)
    }

    /// Number of devices the layers are split across.
    pub fn devices(&self) -> usize {
        self.devs.len()
    }

    /// Write how often each routed expert has been used (every device's
    /// counts, one `layer expert count` line each) for [`warm`](Self::warm)
    /// to start a later run from.
    pub fn save_usage(&mut self, path: &Path) -> Result<()> {
        // merged, not replaced: one short session must not erase what many
        // longer ones learned (older counts fade over USAGE_HALF_LIFE tokens)
        let n_exp = self.cfg.n_routed_experts;
        let decay = 0.5f64.powf(self.usage_tokens as f64 / USAGE_HALF_LIFE as f64);
        let mut counts: Vec<f64> = self.usage.iter().map(|&c| f64::from(c)).collect();
        if let Ok(old) = std::fs::read_to_string(path) {
            for (l, e, n) in parse_usage(&old) {
                if l < self.cfg.n_layers && e < n_exp {
                    counts[l * n_exp + e] += n as f64 * decay;
                }
            }
        }
        let mut text = String::from("# dsv41 routed-expert usage: layer expert count\n");
        for (i, &c) in counts.iter().enumerate() {
            let n = c.round() as u64;
            if n > 0 {
                text.push_str(&format!("{} {} {n}\n", i / n_exp, i % n_exp));
            }
        }
        std::fs::write(path, text)?;
        self.usage.fill(0);
        self.usage_tokens = 0;
        Ok(())
    }

    /// Warm both expert tiers from a [`save_usage`](Self::save_usage) file.
    ///
    /// In the foreground, each device's VRAM cache gets the hottest experts
    /// of its own layers (read `threads` at a time into the host cache, then
    /// uploaded), and every recorded count seeds that device's frequencies so
    /// the warm set is not the first thing evicted. Then a background thread
    /// keeps filling the host cache with the next-hottest experts used more
    /// than once, while the model runs; demand reads share the drive with it.
    /// Returns (records now in VRAM, records queued for the host cache).
    pub fn warm(&mut self, path: &Path, threads: usize) -> Result<(usize, usize)> {
        let text = std::fs::read_to_string(path)?;
        let (n_layers, n_experts) = (self.cfg.n_layers, self.cfg.n_routed_experts);
        let mut ents: Vec<(u64, u32, u32)> =
            parse_usage(&text).filter(|&(l, e, _)| l < n_layers && e < n_experts).map(|(l, e, n)| (n, l as u32, e as u32)).collect();
        ents.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        // each device's share: the hottest of its layers, up to its slots
        let mut vram_keys: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.devs.len()];
        let mut in_vram = std::collections::HashSet::new();
        for &(_, l, e) in &ents {
            let d = self.layers[l as usize].dev;
            if vram_keys[d].len() < self.devs[d].dcache.slots() {
                vram_keys[d].push((l, e));
                in_vram.insert((l, e));
            }
        }
        let all: Vec<(u32, u32)> = vram_keys.iter().flatten().copied().collect();
        read_into(&self.cache, self.store.as_ref(), &all, threads, None)?;
        let mut vram = 0;
        for (d, dev) in self.devs.iter_mut().enumerate() {
            for &(l, e) in &vram_keys[d] {
                dev.dcache.get(&dev.g, l, e, &self.cache, self.store.as_ref())?;
                vram += 1;
            }
            for &(n, l, e) in ents.iter().filter(|&&(_, l, _)| self.layers[l as usize].dev == d) {
                dev.dcache.seed(l, e, n);
            }
            dev.g.sync()?;
            dev.dcache.stats = Default::default();
        }
        // The tiers are exclusive: what VRAM holds, RAM need not (a VRAM hit
        // never reaches the host cache), so its places go to the next-hottest
        // experts instead: ~2,900 more experts resident than with copies in
        // both. An expert later evicted from VRAM is read again if needed.
        for &(l, e) in &all {
            self.cache.remove(l, e);
        }

        // the rest of the host cache, in the background: experts used more
        // than once, hottest first, leaving room for this run's own misses
        let room = self.cache.n_slots().saturating_sub(self.cache.n_slots() / 20);
        let rest: Vec<(u32, u32)> = ents.iter().filter(|&&(n, l, e)| n > 1 && !in_vram.contains(&(l, e))).map(|&(_, l, e)| (l, e)).take(room).collect();
        let queued = rest.len();
        if let Some(w) = self.warming.take() {
            w.stop();
        }
        self.warming = Some(Warming::spawn(Arc::clone(&self.cache), Arc::clone(&self.store), rest, Arc::clone(&self.demand)));
        Ok((vram, queued))
    }

    /// Keep the background warm-up off the drive (`true`) or let it resume.
    /// A server sets this while it handles a request, so the fill runs only
    /// when idle.
    pub fn set_busy(&self, busy: bool) {
        self.demand.busy.store(busy, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records the background warm-up has read so far, and whether it is done.
    pub fn warming(&self) -> Option<(usize, bool)> {
        self.warming.as_ref().map(|w| (w.done.load(std::sync::atomic::Ordering::Relaxed), w.handle.as_ref().is_none_or(|h| h.is_finished())))
    }

    /// Copy a buffer from device `from` to device `to` through the host (the
    /// consumer cards have no peer access).
    fn relocate(&self, buf: CudaSlice<f32>, from: usize, to: usize) -> Result<CudaSlice<f32>> {
        if from == to {
            return Ok(buf);
        }
        self.devs[to].g.upload(&self.devs[from].g.download(&buf)?)
    }

    /// Mix buffer `[t][24]` holding just the given `pre` vectors.
    fn pre_buffer(&self, pre: &[[f32; HC]]) -> Result<CudaSlice<f32>> {
        let mut host = vec![0.0f32; pre.len() * MIXW];
        for (i, p) in pre.iter().enumerate() {
            host[i * MIXW..i * MIXW + HC].copy_from_slice(p);
        }
        self.devs[self.cur].g.upload(&host)
    }

    fn pre_of(&self, mix: &CudaSlice<f32>) -> Result<Vec<f32>> {
        let host = self.devs[self.cur].g.download(mix)?;
        Ok(host.chunks_exact(MIXW).flat_map(|m| m[..HC].iter().copied()).collect())
    }

    pub fn enable_profile(&mut self) {
        self.profile = Some(Profile::default());
    }

    pub fn profile(&self) -> Option<&Profile> {
        self.profile.as_ref()
    }

    /// Start a profiled phase: synchronize so earlier work is not billed to it.
    fn tick(&self) -> Result<Option<Instant>> {
        if self.profile.is_none() {
            return Ok(None);
        }
        self.devs[self.cur].g.sync()?;
        Ok(Some(Instant::now()))
    }

    /// End a profiled phase started by [`tick`](Self::tick).
    fn tock(&mut self, t0: Option<Instant>, field: fn(&mut Profile) -> &mut f64) -> Result<()> {
        if let Some(t0) = t0 {
            self.devs[self.cur].g.sync()?;
            if let Some(p) = self.profile.as_mut() {
                *field(p) += t0.elapsed().as_secs_f64();
            }
        }
        Ok(())
    }

    /// Last position's logits for `ids` at `start_pos`: a prefill at 0, one
    /// decode token, or a chunk continuing the sequence (positions before
    /// `start_pos` must already have been run, or restored with
    /// [`restore`](Self::restore)).
    pub fn forward(&mut self, ids: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        self.run(ids, start_pos, None, None, &[])
    }

    /// Run `ids` at `start_pos` for their effect on the state only: no final
    /// norm, output head or logits download. A prompt fed token by token
    /// needs logits only from its last token; the head (a 129K-row GEMV and
    /// a device-to-host copy) is wasted on every other one.
    pub fn advance_with(&mut self, ids: &[u32], start_pos: usize, images: &[ImageSpan]) -> Result<()> {
        self.run_as(ids, start_pos, None, None, images, false).map(|_| ())
    }

    /// [`forward`](Self::forward) over a stretch that may hold image tokens
    /// (the spans need not lie inside it, nor start at 0).
    pub fn forward_with(&mut self, ids: &[u32], start_pos: usize, images: &[ImageSpan]) -> Result<Vec<f32>> {
        self.run(ids, start_pos, None, None, images)
    }

    /// Image spans the golden checks' traced forwards
    /// ([`Backbone::forward_traced`]) run with.
    pub fn set_trace_images(&mut self, images: Vec<ImageSpan>) {
        self.trace_images = images;
    }

    /// Where the last [`prefill_layered`](Self::prefill_layered) pass's
    /// routed experts came from.
    pub fn pass_stats(&self) -> &PassStats {
        &self.pass_stats
    }

    /// Whether the vision tower is loaded ([`GpuOptions::vision`] and a
    /// checkpoint that has one).
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// A prepared image's span embeddings, `[n_tokens][dim]` (the delimiters'
    /// learned rows around the aligner's), for an [`ImageSpan`].
    pub fn encode_image(&self, img: &Prepared) -> Result<Vec<f32>> {
        let v = self.vision.as_ref().ok_or_else(|| Error::Unsupported("the vision tower is not loaded".into()))?;
        v.span(&self.devs[0].g, img)
    }

    /// Longest sequence the caches hold.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Prefill `ids` at `start_pos` layer by layer: each layer runs over the
    /// whole stretch before the next starts, with the residual stream held
    /// in host memory. Attention goes in sub-chunks of `sub` tokens, each
    /// continuing the last; the routed experts run over all the tokens at
    /// once, so each expert is fetched once for the stretch instead of once
    /// per chunk. When the RAM tier cannot hold every expert (a chunk of a
    /// few hundred tokens already touches nearly all 15,360), a long prompt
    /// then costs about one pass over the experts. Returns the last token's
    /// logits, as [`forward`](Self::forward) over the same sub-chunks would.
    ///
    /// Memory beyond `forward`'s: two `[len][dim]` f32 buffers on the device
    /// (40 KB a token) and two `[len][4 * dim]` in host RAM (160 KB a token),
    /// so callers split very long prompts into several stretches.
    ///
    /// `cancel`, checked before each layer, abandons the pass with an error;
    /// the state is then partly updated, so restore a checkpoint taken at
    /// or before `start_pos` before running anything else.
    pub fn prefill_layered(&mut self, ids: &[u32], start_pos: usize, sub: usize, cancel: Option<&std::sync::atomic::AtomicBool>) -> Result<Vec<f32>> {
        self.prefill_layered_with(ids, start_pos, sub, cancel, &[])
    }

    /// [`prefill_layered`](Self::prefill_layered) over a stretch that may
    /// hold image tokens.
    pub fn prefill_layered_with(&mut self, ids: &[u32], start_pos: usize, sub: usize, cancel: Option<&std::sync::atomic::AtomicBool>, images: &[ImageSpan]) -> Result<Vec<f32>> {
        self.demand.prefill.store(true, std::sync::atomic::Ordering::Relaxed);
        let out = self.layered_inner(ids, start_pos, sub.max(2), cancel, images);
        self.demand.prefill.store(false, std::sync::atomic::Ordering::Relaxed);
        if out.is_ok() {
            self.admit_prefill_experts()?;
        }
        out
    }

    /// After a prefill, each device takes the experts the prompt used most
    /// that it lacked (see [`DeviceExpertCache::admit_pending`]).
    fn admit_prefill_experts(&mut self) -> Result<()> {
        for d in &mut self.devs {
            d.dcache.admit_pending(&d.g, &self.cache, self.store.as_ref())?;
        }
        Ok(())
    }

    fn layered_inner(&mut self, ids: &[u32], start_pos: usize, sub: usize, cancel: Option<&std::sync::atomic::AtomicBool>, images: &[ImageSpan]) -> Result<Vec<f32>> {
        let (d, t) = (self.cfg.dim, ids.len());
        if t == 0 || start_pos + t > self.max_seq {
            return Err(Error::Arg(format!("{t} tokens at {start_pos}: past max_seq {}", self.max_seq)));
        }
        let img_rows = image_rows(images, start_pos, t, d)?;
        let mask: Vec<bool> = img_rows.iter().map(Option::is_some).collect();
        let any_image = mask.contains(&true);
        self.image_mask.clear();
        self.usage_tokens += t as u64;
        let hashes = self.hasher.forward_masked(ids, start_pos, any_image.then_some(&mask[..]))?;
        let mut h_host = vec![0.0f32; t * HC * d];
        for (i, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                return Err(Error::Arg(format!("token {id} outside the vocabulary")));
            }
            let row = &mut h_host[i * HC * d..(i + 1) * HC * d];
            match img_rows[i] {
                Some(r) => row[..d].copy_from_slice(r),
                None => self.embed.row(id as usize, &mut row[..d]),
            }
            for c in 1..HC {
                row.copy_within(..d, c * d);
            }
        }
        let mut pre_host = vec![0.0f32; t * MIXW];
        for i in 0..t {
            pre_host[i * MIXW] = 1.0;
        }
        // engram rows for every token, read while the first layers run
        let (cols, n_eng, head_dim) = (self.hasher.cols(), self.cfg.engram_layer_ids.len(), self.cfg.engram_head_dim);
        let mut engram_rows: Vec<Option<std::thread::JoinHandle<Result<Vec<f32>>>>> = self
            .layers
            .iter()
            .map(|ly| {
                ly.engram.as_ref().map(|(eg, _, _)| {
                    let eg = Arc::clone(eg);
                    let hs: Vec<i64> = (0..t)
                        .flat_map(|i| hashes[(i * n_eng + eg.hash_index) * cols..(i * n_eng + eg.hash_index + 1) * cols].iter().copied())
                        .collect();
                    let skip = any_image.then(|| mask.clone());
                    std::thread::spawn(move || engram_rows(&eg, &hs, head_dim, cols, skip.as_deref()))
                })
            })
            .collect();

        // which experts VRAM or RAM hold as the pass begins
        let n_exp = self.cfg.n_routed_experts;
        let start_resident: Vec<bool> = (0..self.cfg.n_layers * n_exp)
            .map(|i| {
                let (l, e) = ((i / n_exp) as u32, (i % n_exp) as u32);
                self.devs[self.layers[l as usize].dev].dcache.contains(l, e) || self.cache.probe(l, e)
            })
            .collect();
        let reads0 = self.cache.stats().misses;
        let mut ps = PassStats::default();

        let subs: Vec<(usize, usize)> = (0..t).step_by(sub).map(|a| (a, (a + sub).min(t))).collect();
        // the lists index layers hand to later layers, per sub-chunk
        let mut topk: Vec<Vec<Vec<i32>>> = vec![Vec::new(); subs.len()];
        let mut shared: Vec<Shared> = (0..subs.len()).map(|_| Shared::default()).collect();
        let (lim, eps) = (self.cfg.swiglu_limit, self.cfg.norm_eps);
        let inter = self.cfg.moe_inter_dim;
        let mut h1_host = vec![0.0f32; t * HC * d];
        let mut mix_host = vec![0.0f32; t * MIXW];

        #[allow(clippy::needless_range_loop)] // l indexes layers and the pending engram reads alike
        for l in 0..self.cfg.n_layers {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                return Err(Error::Arg("prefill cancelled".into()));
            }
            self.cur = self.layers[l].dev;
            let rows_host = match engram_rows[l].take() {
                Some(pending) => Some(pending.join().map_err(|_| Error::Io(std::io::Error::other("engram read thread panicked")))??),
                None => None,
            };
            let mut x2_all = self.devs[self.cur].g.alloc::<f32>(t * d)?;
            for (s, &(a, b)) in subs.iter().enumerate() {
                let ts = b - a;
                let mut h = self.devs[self.cur].g.upload(&h_host[a * HC * d..b * HC * d])?;
                let pm = self.devs[self.cur].g.upload(&pre_host[a * MIXW..b * MIXW])?;
                if let (Some((_, wkv, qk)), Some(rows)) = (&self.layers[l].engram, &rows_host) {
                    let g = &self.devs[self.cur].g;
                    let w = rows.len() / t;
                    let r = g.upload(&rows[a * w..b * w])?;
                    let kv = wkv.forward(g, &r.as_view(), ts, Out::Bf16)?;
                    let mut out = g.alloc::<f32>(ts * HC * d)?;
                    g.engram_gate(&h, &kv, qk, &mut out, d, ts, eps, (d as f32).powf(-0.5))?;
                    h = out;
                }
                std::mem::swap(&mut self.topk, &mut topk[s]);
                std::mem::swap(&mut self.shared, &mut shared[s]);
                let attn_mix = self.mixes(&h, l, false, ts)?;
                let x = self.hc_pre(&h, &pm, &self.layers[l].attn_norm, ts)?;
                let attn = self.attention(l, &x, ts, start_pos + a)?;
                let h1 = self.hc_post(&attn, &h, &attn_mix, ts)?;
                let ffn_mix = self.mixes(&h1, l, true, ts)?;
                let x2 = self.hc_pre(&h1, &attn_mix, &self.layers[l].ffn_norm, ts)?;
                std::mem::swap(&mut self.topk, &mut topk[s]);
                std::mem::swap(&mut self.shared, &mut shared[s]);
                let g = &self.devs[self.cur].g;
                g.copy(&x2.as_view(), &mut x2_all.slice_mut(a * d..b * d))?;
                h1_host[a * HC * d..b * HC * d].copy_from_slice(&g.download(&h1)?);
                mix_host[a * MIXW..b * MIXW].copy_from_slice(&g.download(&ffn_mix)?);
            }

            // the routed experts over every token at once
            let ly = &self.layers[l];
            let Dev { g, dcache, .. } = &mut self.devs[self.cur];
            let logits = g.download(&ly.gate.forward(g, &x2_all.as_view(), t, Out::F32)?)?;
            let routes = route_mixed(&self.cfg, &logits, &ly.bias, image_bias(ly, &mask), t);
            note_usage(&mut self.usage, n_exp, l, &routes);
            let mut used: Vec<u32> = routes.iter().flat_map(|r| r.experts.iter().copied()).collect();
            used.sort_unstable();
            used.dedup();
            for &e in &used {
                ps.used += 1;
                ps.resident_at_start += usize::from(start_resident[l * n_exp + e as usize]);
                let in_vram = dcache.contains(l as u32, e);
                ps.vram_at_use += usize::from(in_vram);
                ps.resident_at_use += usize::from(in_vram || self.cache.probe(l as u32, e));
            }
            let prof = self.profile.is_some();
            let (y, fetch, experts) = routed_sum(g, dcache, &self.cache, self.store.as_ref(), l, &x2_all.as_view(), t, &routes, lim, prof)?;
            if let Some(p) = self.profile.as_mut() {
                p.fetch += fetch;
                p.experts += experts;
            }

            // shared expert and the residual update, per sub-chunk
            let [w1, w2, w3] = &ly.shared;
            for &(a, b) in &subs {
                let ts = b - a;
                let xs = x2_all.slice(a * d..b * d);
                let (gate, up) = (w1.forward(g, &xs, ts, Out::Bf16)?, w3.forward(g, &xs, ts, Out::Bf16)?);
                let mut hbuf = g.alloc::<f32>(ts * inter)?;
                g.swiglu(&gate, &up, None, &mut hbuf, inter, ts, lim)?;
                let sh = w2.forward(g, &hbuf.as_view(), ts, Out::Bf16)?;
                let ys = g.dup(&y.slice(a * d..b * d))?;
                let mut m = g.alloc::<f32>(ts * d)?;
                g.add_round(&ys, &sh.as_view(), &mut m, ts * d)?;
                let h1 = g.upload(&h1_host[a * HC * d..b * HC * d])?;
                let mix = g.upload(&mix_host[a * MIXW..b * MIXW])?;
                let mut h2 = g.alloc::<f32>(ts * HC * d)?;
                g.hc_post(&m, &h1, &mix, &mut h2, d, ts)?;
                h_host[a * HC * d..b * HC * d].copy_from_slice(&g.download(&h2)?);
            }
            // the ffn mix is the next layer's pre-mix
            std::mem::swap(&mut pre_host, &mut mix_host);
        }

        ps.reads = self.cache.stats().misses - reads0;
        self.pass_stats = ps;

        // the last token through the final norm and the head
        let out_dev = self.devs.len() - 1;
        self.cur = out_dev;
        let g = &self.devs[out_dev].g;
        let last = t - 1;
        let hl = g.upload(&h_host[last * HC * d..t * HC * d])?;
        let pl = g.upload(&pre_host[last * MIXW..t * MIXW])?;
        let mut x = g.alloc::<f32>(d)?;
        g.hc_pre(&hl, &pl.as_view(), &mut x, d, 1, MIXW)?;
        let mut xn = g.alloc::<f32>(d)?;
        g.rmsnorm(&x.as_view(), &self.norm, &mut xn.slice_mut(..), 1, d, self.cfg.norm_eps)?;
        g.download(&self.head.forward(g, &xn.as_view(), 1, Out::F32)?)
    }

    /// Snapshot the state after `pos` tokens, to come back to with
    /// [`restore`](Self::restore) and continue from `pos` with different
    /// tokens (a new reply to the same prompt, say). Only the sliding-window
    /// rings and the compressors' partial groups need copying (~10 MB, to
    /// host memory): the compressed caches below `pos` are never rewritten
    /// by later tokens, and above it they are rewritten before being read.
    pub fn checkpoint(&self, pos: usize) -> Result<Checkpoint> {
        let mut layers = Vec::with_capacity(self.states.len());
        for (l, st) in self.states.iter().enumerate() {
            let g = &self.devs[self.layers[l].dev].g;
            let copy = |s: &Option<CudaSlice<f32>>| s.as_ref().map(|s| g.download(s)).transpose();
            layers.push((g.download(&st.window)?, copy(&st.kv_state)?, copy(&st.score_state)?));
        }
        Ok(Checkpoint { pos, layers })
    }

    /// Return to a [`checkpoint`](Self::checkpoint): the next forward may
    /// start at its position. Checkpoints taken after it (at a later
    /// position, of other tokens) are no longer valid once new tokens run.
    pub fn restore(&mut self, ck: &Checkpoint) -> Result<()> {
        if ck.layers.len() != self.states.len() {
            return Err(Error::Arg("checkpoint is from another model".into()));
        }
        for (l, (window, kvs, scs)) in ck.layers.iter().enumerate() {
            let g = &self.devs[self.layers[l].dev].g;
            let st = &mut self.states[l];
            g.write(window, &mut st.window.slice_mut(..))?;
            if let (Some(src), Some(dst)) = (kvs, st.kv_state.as_mut()) {
                g.write(src, &mut dst.slice_mut(..))?;
            }
            if let (Some(src), Some(dst)) = (scs, st.score_state.as_mut()) {
                g.write(src, &mut dst.slice_mut(..))?;
            }
        }
        for d in &self.devs {
            d.g.sync()?;
        }
        Ok(())
    }

    fn run(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Option<Trace<'_>>, images: &[ImageSpan]) -> Result<Vec<f32>> {
        self.run_as(ids, start_pos, teacher, trace, images, true)
    }

    /// [`run`](Self::run); `logits: false` stops after the last layer (no
    /// final norm, head or download) and returns nothing.
    fn run_as(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Option<Trace<'_>>, images: &[ImageSpan], logits: bool) -> Result<Vec<f32>> {
        let prefill = ids.len() > 1;
        if prefill {
            self.demand.prefill.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let out = self.run_inner(ids, start_pos, teacher, trace, images, logits);
        if prefill {
            self.demand.prefill.store(false, std::sync::atomic::Ordering::Relaxed);
            if out.is_ok() {
                self.admit_prefill_experts()?;
            }
        } else {
            // decode tokens age the VRAM caches' usage counts
            for d in &mut self.devs {
                d.dcache.tick_token();
            }
        }
        out
    }

    fn run_inner(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, mut trace: Option<Trace<'_>>, images: &[ImageSpan], want_logits: bool) -> Result<Vec<f32>> {
        let (d, t) = (self.cfg.dim, ids.len());
        if t == 0 || start_pos + t > self.max_seq {
            return Err(Error::Arg(format!("{t} tokens at {start_pos}: past max_seq {}", self.max_seq)));
        }
        let img_rows = image_rows(images, start_pos, t, d)?;
        let mask: Vec<bool> = img_rows.iter().map(Option::is_some).collect();
        let any_image = mask.contains(&true);
        // the router (moe) reads it; set before anything can fail half-way
        self.image_mask = if any_image { mask.clone() } else { Vec::new() };
        self.usage_tokens += t as u64;
        let hashes = self.hasher.forward_masked(ids, start_pos, any_image.then_some(&mask[..]))?;
        let mut emb = vec![0.0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                return Err(Error::Arg(format!("token {id} outside the vocabulary")));
            }
            match img_rows[i] {
                Some(r) => emb[i * d..(i + 1) * d].copy_from_slice(r),
                None => self.embed.row(id as usize, &mut emb[i * d..(i + 1) * d]),
            }
        }
        if let Some(tr) = trace.as_mut() {
            tr("engram_hashes", &hashes.iter().map(|&v| v as f32).collect::<Vec<_>>());
            tr("embed", &emb);
        }
        let stream: Vec<f32> = (0..t).flat_map(|i| std::iter::repeat_n(&emb[i * d..(i + 1) * d], HC).flatten().copied()).collect();
        self.cur = self.layers[0].dev;
        let mut h = self.devs[self.cur].g.upload(&stream)?;
        let mut pre_mix = self.pre_buffer(&vec![[1.0, 0.0, 0.0, 0.0]; t])?;

        // Engram rows depend on the token ids alone: start every engram
        // layer's reads now, so the drive works while the layers before it run
        let (cols, n_eng, head_dim) = (self.hasher.cols(), self.cfg.engram_layer_ids.len(), self.cfg.engram_head_dim);
        let mut engram_rows: Vec<Option<std::thread::JoinHandle<Result<Vec<f32>>>>> = self
            .layers
            .iter()
            .map(|ly| {
                ly.engram.as_ref().map(|(eg, _, _)| {
                    let eg = Arc::clone(eg);
                    let hs: Vec<i64> = (0..t)
                        .flat_map(|i| hashes[(i * n_eng + eg.hash_index) * cols..(i * n_eng + eg.hash_index + 1) * cols].iter().copied())
                        .collect();
                    let skip = any_image.then(|| mask.clone());
                    std::thread::spawn(move || engram_rows(&eg, &hs, head_dim, cols, skip.as_deref()))
                })
            })
            .collect();
        #[allow(clippy::needless_range_loop)] // l indexes layers, states and the pending engram reads alike
        for l in 0..self.cfg.n_layers {
            let dev = self.layers[l].dev;
            if dev != self.cur {
                h = self.relocate(h, self.cur, dev)?;
                pre_mix = self.relocate(pre_mix, self.cur, dev)?;
                self.cur = dev;
            }
            if let (Some(teach), true) = (teacher, l > 0) {
                let (th, tp) = teach(l)?;
                if th.len() != t * HC * d || tp.len() != t {
                    return Err(Error::Arg(format!("teacher input for layer {l} has the wrong shape")));
                }
                h = self.devs[self.cur].g.upload(&th)?;
                pre_mix = self.pre_buffer(&tp)?;
            }
            let t0 = self.tick()?;
            if let (Some((_, wkv, qk)), Some(pending)) = (&self.layers[l].engram, engram_rows[l].take()) {
                let host_rows = pending.join().map_err(|_| Error::Io(std::io::Error::other("engram read thread panicked")))??;
                let g = &self.devs[self.cur].g;
                let rows = g.upload(&host_rows)?;
                let kv = wkv.forward(g, &rows.as_view(), t, Out::Bf16)?;
                let mut out = g.alloc::<f32>(t * HC * d)?;
                g.engram_gate(&h, &kv, qk, &mut out, d, t, self.cfg.norm_eps, (d as f32).powf(-0.5))?;
                h = out;
            }
            self.tock(t0, |p| &mut p.engram)?;
            let (out, next_pre) = self.block(l, &h, t, start_pos, &pre_mix, &mut trace)?;
            h = out;
            pre_mix = next_pre;
            if let Some(tr) = trace.as_mut() {
                tr(&format!("layer{l:02}.out"), &self.devs[self.cur].g.download(&h)?);
                tr(&format!("layer{l:02}.pre_mix"), &self.pre_of(&pre_mix)?);
            }
        }
        if !want_logits {
            if let Some(p) = self.profile.as_mut() {
                p.forwards += 1;
            }
            return Ok(Vec::new());
        }

        let out_dev = self.devs.len() - 1;
        let h = self.relocate(h, self.cur, out_dev)?;
        let pre_mix = self.relocate(pre_mix, self.cur, out_dev)?;
        self.cur = out_dev;
        let t0 = self.tick()?;
        let last = t - 1;
        let g = &self.devs[self.cur].g;
        let mut x = g.alloc::<f32>(d)?;
        let hl = g.dup(&h.slice(last * HC * d..(last + 1) * HC * d))?;
        g.hc_pre(&hl, &pre_mix.slice(last * MIXW..(last + 1) * MIXW), &mut x, d, 1, MIXW)?;
        let mut xn = g.alloc::<f32>(d)?;
        g.rmsnorm(&x.as_view(), &self.norm, &mut xn.slice_mut(..), 1, d, self.cfg.norm_eps)?;
        let logits = g.download(&self.head.forward(g, &xn.as_view(), 1, Out::F32)?)?;
        self.tock(t0, |p| &mut p.head)?;
        if let Some(p) = self.profile.as_mut() {
            p.forwards += 1;
        }
        if let Some(tr) = trace.as_mut() {
            tr("logits", &logits);
        }
        Ok(logits)
    }

    /// Mix coefficients `[t][24]` for the stream `h`, computed on the device.
    fn mixes(&mut self, h: &CudaSlice<f32>, l: usize, ffn: bool, t: usize) -> Result<CudaSlice<f32>> {
        let n = HC * self.cfg.dim;
        let (eps, iters, hc_eps) = (self.cfg.norm_eps, self.cfg.hc_sinkhorn_iters, self.cfg.hc_eps);
        let p = if ffn { &self.layers[l].hc_ffn } else { &self.layers[l].hc_attn };
        let Dev { g, mix_counter, .. } = &mut self.devs[self.cur];
        let mut proj = g.alloc::<f32>(t * 25)?;
        let mut mix = g.alloc::<f32>(t * MIXW)?;
        if mix_counter.len() >= t {
            g.hc_project_mix(h, &p.proj, &mut proj, &p.base, &p.scale, &mut mix, mix_counter, t, n, eps, iters, hc_eps)?;
        } else {
            g.hc_project(h, &p.proj, &mut proj, n, t)?;
            g.hc_mix(&proj, &p.base, &p.scale, &mut mix, t, n, eps, iters, hc_eps)?;
        }
        Ok(mix)
    }

    /// Collapse the copies with the `pre` of `mix`, then RMSNorm.
    fn hc_pre(&self, h: &CudaSlice<f32>, mix: &CudaSlice<f32>, norm: &CudaSlice<f32>, t: usize) -> Result<CudaSlice<f32>> {
        let (d, g) = (self.cfg.dim, &self.devs[self.cur].g);
        let mut y = g.alloc::<f32>(t * d)?;
        g.hc_pre_norm(h, &mix.as_view(), MIXW, norm, &mut y, d, t, self.cfg.norm_eps)?;
        Ok(y)
    }

    fn hc_post(&self, out: &CudaSlice<f32>, res: &CudaSlice<f32>, mix: &CudaSlice<f32>, t: usize) -> Result<CudaSlice<f32>> {
        let (d, g) = (self.cfg.dim, &self.devs[self.cur].g);
        let mut y = g.alloc::<f32>(t * HC * d)?;
        g.hc_post(out, res, mix, &mut y, d, t)?;
        Ok(y)
    }

    fn block<'t>(
        &mut self,
        l: usize,
        h: &CudaSlice<f32>,
        t: usize,
        start_pos: usize,
        pre_mix: &CudaSlice<f32>,
        trace: &mut Option<Trace<'t>>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let t0 = self.tick()?;
        let attn_mix = self.mixes(h, l, false, t)?;
        let x = self.hc_pre(h, pre_mix, &self.layers[l].attn_norm, t)?;
        self.tock(t0, |p| &mut p.hc)?;
        let t0 = self.tick()?;
        let a = self.attention(l, &x, t, start_pos)?;
        self.tock(t0, |p| &mut p.attn)?;
        if let Some(tr) = trace.as_mut() {
            tr(&format!("layer{l:02}.attn_out"), &self.devs[self.cur].g.download(&a)?);
        }
        let t0 = self.tick()?;
        let h1 = self.hc_post(&a, h, &attn_mix, t)?;
        let ffn_mix = self.mixes(&h1, l, true, t)?;
        let x = self.hc_pre(&h1, &attn_mix, &self.layers[l].ffn_norm, t)?;
        self.tock(t0, |p| &mut p.hc)?;
        let (m, routes) = self.moe(l, &x, t)?;
        if let Some(tr) = trace.as_mut() {
            tr(&format!("layer{l:02}.moe_out"), &self.devs[self.cur].g.download(&m)?);
            tr(&format!("layer{l:02}.route_ids"), &routes.iter().flat_map(|r| r.experts.iter().map(|&e| e as f32)).collect::<Vec<_>>());
            tr(&format!("layer{l:02}.route_w"), &routes.iter().flat_map(|r| r.weights.iter().copied()).collect::<Vec<_>>());
        }
        let t0 = self.tick()?;
        let h2 = self.hc_post(&m, &h1, &ffn_mix, t)?;
        self.tock(t0, |p| &mut p.hc)?;
        Ok((h2, ffn_mix))
    }

    fn attention(&mut self, l: usize, x: &CudaSlice<f32>, t: usize, start_pos: usize) -> Result<CudaSlice<f32>> {
        let (hd, nh, rd, win) = (self.cfg.head_dim, self.cfg.n_heads, self.cfg.rope_head_dim, self.cfg.window_size);
        let (qlr, eps) = (self.cfg.q_lora_rank, self.cfg.norm_eps);
        let ratio = self.layers[l].ratio;
        let (has_comp, has_idx) = (self.layers[l].compressor.is_some(), self.layers[l].indexer.is_some());
        // queries, sliding-window kv, window cache and index lists; `fresh`
        // is the kv attended directly (a prefill's own, or ring + chunk for a
        // chunk continuing the sequence), None at decode (the window cache is
        // attended where it lives); `offset` is where compressed rows start
        let (q, qr, fresh, mut idxs, offset) = {
            let dv = &self.devs[self.cur];
            let g = &dv.g;
            let ly = &self.layers[l];
            let (cos, sin) = if ratio > 0 { &dv.rope_compress } else { &dv.rope_window };
            let qr0 = ly.wq_a.forward(g, &x.as_view(), t, Out::Bf16)?;
            let mut qr = g.alloc::<f32>(t * qlr)?;
            g.rmsnorm(&qr0.as_view(), &ly.q_norm, &mut qr.slice_mut(..), t, qlr, eps)?;
            let mut q = ly.wq_b.forward(g, &qr.as_view(), t, Out::Bf16)?;
            g.rope(&mut q.slice_mut(..), cos, sin, Pos::Linear { base: start_pos, per: nh }, t * nh, hd, hd - rd, rd / 2, false)?;

            let kv0 = ly.wkv.forward(g, &x.as_view(), t, Out::Bf16)?;
            let window = &mut self.states[l].window;
            if start_pos == 0 {
                let mut kv = g.alloc::<f32>(t * hd)?;
                g.rmsnorm(&kv0.as_view(), &ly.kv_norm, &mut kv.slice_mut(..), t, hd, eps)?;
                g.rope(&mut kv.slice_mut(..), cos, sin, Pos::Linear { base: start_pos, per: 1 }, t, hd, hd - rd, rd / 2, false)?;
                g.act_quant_fp8(&mut kv.slice_mut(..))?;
                if t <= win {
                    g.copy(&kv.as_view(), &mut window.slice_mut(..t * hd))?;
                } else {
                    let cutoff = t % win;
                    g.copy(&kv.slice((t - win) * hd..(t - cutoff) * hd), &mut window.slice_mut(cutoff * hd..))?;
                    g.copy(&kv.slice((t - cutoff) * hd..t * hd), &mut window.slice_mut(..cutoff * hd))?;
                }
                let idxs: Vec<Vec<i32>> = (0..t)
                    .map(|i| {
                        let lo = (i + 1).saturating_sub(win);
                        (0..t.min(win)).map(|j| if lo + j > i { -1 } else { (lo + j) as i32 }).collect()
                    })
                    .collect();
                (q, qr, Some(kv), idxs, t)
            } else if t == 1 {
                // decode: norm, rope, act-quant and the window write in one launch
                let slot = start_pos % win;
                g.kv_finish(&kv0.as_view(), &ly.kv_norm, cos, sin, start_pos, rd / 2, &mut window.slice_mut(slot * hd..(slot + 1) * hd), hd, eps)?;
                let oldest = slot + 1;
                let ring = (oldest..win).chain(0..oldest).map(|s| if s > start_pos { -1 } else { s as i32 }).collect();
                (q, qr, None, vec![ring], win)
            } else {
                // a chunk continuing the sequence: each query's window reaches
                // back into the ring, so attend over [ring ++ this chunk's kv],
                // positions oldest first as decode does
                let mut kv = g.alloc::<f32>(t * hd)?;
                g.rmsnorm(&kv0.as_view(), &ly.kv_norm, &mut kv.slice_mut(..), t, hd, eps)?;
                g.rope(&mut kv.slice_mut(..), cos, sin, Pos::Linear { base: start_pos, per: 1 }, t, hd, hd - rd, rd / 2, false)?;
                g.act_quant_fp8(&mut kv.slice_mut(..))?;
                let mut both = g.alloc::<f32>((win + t) * hd)?;
                g.copy(&window.as_view(), &mut both.slice_mut(..win * hd))?;
                g.copy(&kv.as_view(), &mut both.slice_mut(win * hd..))?;
                // then the ring takes the chunk's last positions (slot = pos % win)
                let m = t.min(win);
                let first = start_pos + t - m;
                let (a, run) = (first % win, m.min(win - first % win));
                g.copy(&kv.slice((t - m) * hd..(t - m + run) * hd), &mut window.slice_mut(a * hd..(a + run) * hd))?;
                if run < m {
                    g.copy(&kv.slice((t - m + run) * hd..t * hd), &mut window.slice_mut(..(m - run) * hd))?;
                }
                let idxs: Vec<Vec<i32>> = (0..t)
                    .map(|i| {
                        let p = start_pos + i;
                        (0..win)
                            .map(|j| match (p + 1 + j).checked_sub(win) {
                                None => -1,
                                Some(q) if q >= start_pos => (win + q - start_pos) as i32,
                                Some(q) => (q % win) as i32,
                            })
                            .collect()
                    })
                    .collect();
                (q, qr, Some(both), idxs, win + t)
            }
        };

        // compressed positions: this layer's own latent, the index hand-off, the shared cache
        let mut comp: Option<(usize, usize)> = None; // (source layer, compressed rows)
        if ratio > 0 {
            let compress_len = (start_pos + t) / ratio;
            let latent = if has_comp {
                self.compress_src = Some(l);
                self.compressor(l, x, t, start_pos)?
            } else {
                None
            };
            let comp_idxs = if has_idx {
                let got = if compress_len == 0 {
                    vec![Vec::new(); t]
                } else {
                    self.indexer(l, x, &qr, latent.as_ref(), t, start_pos, offset)?
                };
                self.topk = got.clone();
                got
            } else {
                self.topk.clone()
            };
            let dv = &self.devs[self.cur];
            let g = &dv.g;
            let (cos, sin) = &dv.rope_compress;
            if let Some(mut lat) = latent {
                let n_c = lat.len() / hd;
                let pos: Vec<i32> = (0..n_c).map(|j| compressed_pos(start_pos, j, ratio) as i32).collect();
                g.rope(&mut lat.slice_mut(..), cos, sin, Pos::Rows(&g.upload(&pos)?), n_c, hd, hd - rd, rd / 2, false)?;
                g.act_quant_fp4(&mut lat.slice_mut(..), 16, true)?;
                let at = start_pos / ratio * hd;
                let cache = self.states[l].compress_kv.as_mut().ok_or_else(|| Error::Format(format!("layer {l}: no compressed cache")))?;
                g.copy(&lat.as_view(), &mut cache.slice_mut(at..at + lat.len()))?;
            }
            let src = self.compress_src.ok_or_else(|| Error::Format(format!("layer {l}: no compressed-KV source")))?;
            let cache = self.states[src].compress_kv.as_ref().ok_or_else(|| Error::Format(format!("layer {src}: no compressed cache")))?;
            if cache.len() < compress_len * hd {
                return Err(Error::Format(format!("layer {src}: compressed cache holds fewer than {compress_len} rows")));
            }
            comp = Some((src, compress_len));
            if comp_idxs.len() != t {
                return Err(Error::Format(format!("layer {l}: index hand-off has {} queries, need {t}", comp_idxs.len())));
            }
            for (w, c) in idxs.iter_mut().zip(comp_idxs) {
                w.extend(c);
            }
        }

        // sparse attention, undo the query rotation, grouped low-rank output
        let dv = &self.devs[self.cur];
        let g = &dv.g;
        let ly = &self.layers[l];
        let (cos, sin) = if ratio > 0 { &dv.rope_compress } else { &dv.rope_window };
        let nidx = idxs.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let flat: Vec<i32> = idxs.iter().flat_map(|r| r.iter().copied().chain(std::iter::repeat_n(-1, nidx - r.len()))).collect();
        let mut o = g.alloc::<f32>(t * nh * hd)?;
        let window = &self.states[l].window;
        let kva = match &fresh {
            Some(kv) => kv.as_view(),
            None => window.as_view(),
        };
        // rows past the window part come from the shared compressed cache
        // (any valid buffer stands in when there are none: never indexed)
        let kvb = match comp {
            Some((src, n)) if n > 0 => self.states[src].compress_kv.as_ref().expect("checked above").slice(..n * hd),
            _ => window.as_view(),
        };
        // the query rotation is undone inside the attention kernel
        g.sparse_attn(&q, &kva, offset, &kvb, &g.upload(&flat)?, &ly.sink, &mut o, t, nh, hd, nidx, (hd as f32).powf(-0.5), Some((cos, sin, rd / 2, start_pos)))?;
        let (groups, orank) = (self.cfg.o_groups, self.cfg.o_lora_rank);
        let gd = nh * hd / groups;
        let mut og = g.alloc::<f32>(t * groups * orank)?;
        if t > 1 && gd % 32 == 0 && orank % 64 == 0 {
            g.gemm_fp8w(&o.as_view(), &ly.wo_a.0, &ly.wo_a.1, &mut og.slice_mut(..), groups * orank, gd, t, true, orank, nh * hd)?;
        } else {
            g.gemv_fp8w(&o.as_view(), &ly.wo_a.0, &ly.wo_a.1, &mut og.slice_mut(..), groups * orank, gd, t, true, orank, nh * hd)?;
        }
        ly.wo_b.forward(g, &og.as_view(), t, Out::Bf16)
    }

    /// Latent(s) completed by these tokens, pre-RoPE, or `None` while a group fills.
    fn compressor(&mut self, l: usize, x: &CudaSlice<f32>, t: usize, start_pos: usize) -> Result<Option<CudaSlice<f32>>> {
        let (g, hd, eps) = (&self.devs[self.cur].g, self.cfg.head_dim, self.cfg.norm_eps);
        let c = self.layers[l].compressor.as_ref().expect("kv source has a compressor");
        let norm = |v: &CudaSlice<f32>, rows: usize| -> Result<CudaSlice<f32>> {
            let mut y = g.alloc::<f32>(rows * hd)?;
            g.rmsnorm(&v.as_view(), &c.norm, &mut y.slice_mut(..), rows, hd, eps)?;
            Ok(y)
        };
        if c.ratio == 1 {
            return Ok(Some(norm(&c.wkv.forward(g, &x.as_view(), t, Out::Bf16)?, t)?));
        }
        let r = c.ratio;
        let kv = c.wkv.forward(g, &x.as_view(), t, Out::F32)?;
        let score = c.wgate.as_ref().expect("ratio > 1 has wgate").forward(g, &x.as_view(), t, Out::F32)?;
        let st = &mut self.states[l];
        let (kvs, scs) = (st.kv_state.as_mut().expect("ratio-2 state"), st.score_state.as_mut().expect("ratio-2 state"));
        // a chunk continuing the sequence: the partial group in the state
        // (start_pos % r rows) comes first, then the chunk's rows
        let (kv, score) = if start_pos > 0 && t > 1 {
            let p = start_pos % r;
            let (mut ckv, mut csc) = (g.alloc::<f32>((p + t) * hd)?, g.alloc::<f32>((p + t) * hd)?);
            if p > 0 {
                g.copy(&kvs.slice(..p * hd), &mut ckv.slice_mut(..p * hd))?;
                g.copy(&scs.slice(..p * hd), &mut csc.slice_mut(..p * hd))?;
            }
            g.copy(&kv.as_view(), &mut ckv.slice_mut(p * hd..))?;
            g.copy(&score.as_view(), &mut csc.slice_mut(p * hd..))?;
            (ckv, csc)
        } else {
            (kv, score)
        };
        let (src_kv, src_sc, groups) = if start_pos == 0 || t > 1 {
            // rows from the first token of a group: complete groups are pooled,
            // the rest waits in the state
            let n = kv.len() / hd;
            let rem = n % r;
            let cutoff = n - rem;
            if rem > 0 {
                g.copy(&kv.slice(cutoff * hd..), &mut kvs.slice_mut(..rem * hd))?;
                g.copy(&score.slice(cutoff * hd..), &mut scs.slice_mut(..rem * hd))?;
            }
            if n < r {
                return Ok(None);
            }
            (kv.slice(..cutoff * hd), score.slice(..cutoff * hd), cutoff / r)
        } else {
            let slot = start_pos % r;
            g.copy(&kv.as_view(), &mut kvs.slice_mut(slot * hd..(slot + 1) * hd))?;
            g.copy(&score.as_view(), &mut scs.slice_mut(slot * hd..(slot + 1) * hd))?;
            if !(start_pos + 1).is_multiple_of(r) {
                return Ok(None);
            }
            (kvs.slice(..), scs.slice(..), 1)
        };
        let mut pooled = g.alloc::<f32>(groups * hd)?;
        g.compress_pool(&src_kv, &src_sc, &mut pooled, groups, r, hd)?;
        Ok(Some(norm(&pooled, groups)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn indexer(
        &mut self,
        l: usize,
        x: &CudaSlice<f32>,
        qr: &CudaSlice<f32>,
        latent: Option<&CudaSlice<f32>>,
        t: usize,
        start_pos: usize,
        offset: usize,
    ) -> Result<Vec<Vec<i32>>> {
        let cfg = &self.cfg;
        let (ihd, inh, rd, hd) = (cfg.index_head_dim, cfg.index_n_heads, cfg.rope_head_dim, cfg.head_dim);
        let ratio = self.layers[l].ratio;
        let end_pos = start_pos + t;
        let dv = &self.devs[self.cur];
        let g = &dv.g;
        let (cos, sin) = &dv.rope_compress;
        let ix = self.layers[l].indexer.as_ref().expect("index source");

        if let Some(wk) = &ix.wk {
            if let Some(lat) = latent {
                let n_c = lat.len() / hd;
                let k0 = wk.forward(g, &lat.as_view(), n_c, Out::Bf16)?;
                let mut k = g.alloc::<f32>(n_c * ihd)?;
                g.rmsnorm(&k0.as_view(), ix.k_norm.as_ref().expect("owner has k_norm"), &mut k.slice_mut(..), n_c, ihd, cfg.norm_eps)?;
                let pos: Vec<i32> = (0..n_c).map(|j| compressed_pos(start_pos, j, ratio) as i32).collect();
                g.rope(&mut k.slice_mut(..), cos, sin, Pos::Rows(&g.upload(&pos)?), n_c, ihd, ihd - rd, rd / 2, false)?;
                g.act_quant_fp4(&mut k.slice_mut(..), 32, false)?;
                let at = start_pos / ratio * ihd;
                let cache = self.states[l].index_k.as_mut().expect("owner has an index cache");
                g.copy(&k.as_view(), &mut cache.slice_mut(at..at + k.len()))?;
            }
            // owners always publish their own keys (see the deviation note in dsv41 attention.rs)
            self.index_k_src = Some(l);
        }

        let mut q = ix.wq_b.forward(g, &qr.as_view(), t, Out::Bf16)?;
        g.rope(&mut q.slice_mut(..), cos, sin, Pos::Linear { base: start_pos, per: inh }, t * inh, ihd, ihd - rd, rd / 2, false)?;
        g.act_quant_fp4(&mut q.slice_mut(..), 32, false)?;
        let wscale = (ihd as f32).powf(-0.5) * (inh as f32).powf(-0.5);
        // bf16(w * wscale) per head happens in the kernel: no round trip here
        let weights = ix.weights_proj.forward(g, &x.as_view(), t, Out::Bf16)?;

        let src = self.index_k_src.ok_or_else(|| Error::Format(format!("layer {l}: no index-key source")))?;
        let n_t = end_pos / ratio;
        let keys = self.states[src].index_k.as_ref().expect("index source has keys");
        let mut scores = g.alloc::<f32>(t * n_t)?;
        g.index_scores(&q, &keys.slice(..n_t * ihd), &weights, &mut scores, inh, ihd, n_t, t, wscale)?;
        let scores = g.download(&scores)?;
        let role = ix.role;
        select_compressed(cfg, &scores, t, n_t, start_pos, ratio, offset, role, &mut self.shared)
            .map_err(|e| Error::Format(format!("layer {l}: {e}")))
    }

    fn moe(&mut self, l: usize, x: &CudaSlice<f32>, t: usize) -> Result<(CudaSlice<f32>, Vec<Route>)> {
        let t0 = self.tick()?;
        let d = self.cfg.dim;
        let Dev { g, dcache, handoff, cpu_seq, inbox, inbox_seq, .. } = &mut self.devs[self.cur];
        if let Some(why) = handoff.take_error() {
            return Err(Error::Format(format!("CPU experts failed: {why}")));
        }
        let (g, ly) = (&*g, &self.layers[l]);
        // decode: the logits and this activation reach the host through the
        // inbox (one kernel and a spin), not two driver downloads
        let n_exp = self.cfg.n_routed_experts;
        let (logits, x_pub) = if t == 1 {
            let lg = ly.gate.forward(g, &x.as_view(), 1, Out::F32)?;
            *inbox_seq += 1;
            g.publish(&lg.as_view(), &x.as_view(), inbox, *inbox_seq)?;
            inbox.wait(g, *inbox_seq)?;
            let (mut lgh, mut xh) = (vec![0.0f32; n_exp], vec![0.0f32; d]);
            inbox.read(0, &mut lgh);
            inbox.read(n_exp, &mut xh);
            (lgh, Some(xh))
        } else {
            (g.download(&ly.gate.forward(g, &x.as_view(), t, Out::F32)?)?, None)
        };
        let routes = route_mixed(&self.cfg, &logits, &ly.bias, image_bias(ly, &self.image_mask), t);
        note_usage(&mut self.usage, n_exp, l, &routes);
        let prof = self.profile.is_some();
        let elapsed = |t0: Option<Instant>| -> Result<f64> {
            match t0 {
                Some(t0) => {
                    g.sync()?;
                    Ok(t0.elapsed().as_secs_f64())
                }
                None => Ok(0.0),
            }
        };
        let (mut fetch, mut compute) = (0.0, 0.0);
        let route_s = elapsed(t0)?;
        let lim = self.cfg.swiglu_limit;
        let (mut xq, out, on_cpu) = if t == 1 {
            // Decode: the routed experts resident in VRAM run in one grouped
            // launch per stage, reading their slots by address (pinned for
            // this batch, so no later upload can evict one). A miss is
            // uploaded (always without a CPU pool; with one, only when it is
            // used more than the VRAM victim) or else computed on the CPU from
            // the host cache while the GPU works. Each result lands in its
            // expert-id row, the order the reference sums in.
            let r = &routes[0];
            let mut order: Vec<usize> = (0..r.experts.len()).collect();
            order.sort_by_key(|&k| r.experts[k]);
            let tf = if prof { g.sync()?; Some(Instant::now()) } else { None };
            dcache.begin_batch();
            let (store, host) = (self.store.as_ref(), &self.cache);
            let (mut recs, mut rows, mut ws) = (Vec::new(), Vec::new(), Vec::new());
            let mut on_cpu = Vec::new();
            let mut promoted = 0;
            for (row, &k) in order.iter().enumerate() {
                let e = r.experts[k];
                let addr = match dcache.lookup(l as u32, e) {
                    Some(rec) => g.addr(&rec),
                    None => {
                        let from_disk = !host.probe(l as u32, e);
                        if from_disk {
                            self.demand.read();
                        }
                        let lease = host.acquire(l as u32, e, store)?;
                        if from_disk {
                            self.demand.read();
                        }
                        let admit = self.cpu.is_none() || (promoted < PROMOTE_PER_LAYER && dcache.worth_admitting(l as u32, e));
                        if !admit {
                            if promoted >= PROMOTE_PER_LAYER {
                                dcache.stats.declined += 1;
                            }
                            on_cpu.push((row, lease, r.weights[k]));
                            continue;
                        }
                        promoted += usize::from(self.cpu.is_some());
                        g.addr(&dcache.insert(g, l as u32, e, &lease)?)
                    }
                };
                recs.push(addr);
                rows.push(row);
                ws.push(r.weights[k]);
            }
            fetch += elapsed(tf)?;
            let tc = if prof { Some(Instant::now()) } else { None };
            // the CPU's copy of x (the GPU is idle: routing just synchronized)
            // start the CPU job first, so it runs while the GPU work below is
            // launched and executed (the GPU is idle here: routing just synced)
            let cpu_part = match self.cpu.as_ref() {
                Some(_) if on_cpu.is_empty() => CpuPart::None,
                Some(_) if prof => CpuPart::Blocking(on_cpu, x_pub.clone().expect("decode publishes x")),
                Some(cpu) => {
                    let x_host = x_pub.clone().expect("decode publishes x");
                    *cpu_seq += 1;
                    let seq = *cpu_seq;
                    let mask = on_cpu.iter().fold(0u32, |m, &(row, _, _)| m | 1 << row);
                    let rows: Vec<usize> = on_cpu.iter().map(|&(row, _, _)| row).collect();
                    let uses = rows.len() as u64;
                    let recs: Vec<Arc<Vec<u8>>> = on_cpu.iter().map(|(_, lease, _)| lease.to_arc()).collect();
                    let ws: Vec<f32> = on_cpu.iter().map(|&(_, _, w)| w).collect();
                    let sink = Arc::clone(handoff);
                    cpu.spawn(
                        recs,
                        ws,
                        x_host,
                        lim,
                        Box::new(move |res| {
                            match res {
                                Ok(ys) => {
                                    for (&row, y) in rows.iter().zip(&ys) {
                                        sink.write_row(row, y);
                                    }
                                }
                                Err(why) => sink.fail(why),
                            }
                            // always, so the device never waits forever
                            sink.release(seq);
                        }),
                    );
                    CpuPart::Launched { mask, seq, uses }
                }
                None => CpuPart::None,
            };
            let nexp = order.len();
            let xq = g.act_quant_fp8_to(&x.as_view())?;
            let mut outs = g.alloc::<f32>(nexp * DIM)?;
            if !recs.is_empty() {
                let ng = recs.len();
                let tab = g.upload(&Gpu::moe_table(&recs, &rows, &ws))?;
                let mut hbuf = g.alloc::<f32>(ng * INTER)?;
                g.moe_gate_up(&xq.as_view(), &tab, &mut hbuf, ng, INTER, DIM, lim)?;
                g.act_quant_fp8(&mut hbuf.slice_mut(..))?;
                g.moe_down(&hbuf, &tab, &mut outs, ng, nexp, INTER, DIM)?;
            }
            compute += elapsed(tc)?;
            (Some(xq), MoeOut::Grouped(outs, nexp), cpu_part)
        } else {
            let (y, f, c) = routed_sum(g, dcache, &self.cache, self.store.as_ref(), l, &x.as_view(), t, &routes, lim, prof)?;
            fetch += f;
            compute += c;
            (None, MoeOut::Summed(y), CpuPart::None)
        };
        let ts = if prof { Some(Instant::now()) } else { None };
        // the fp8 shared expert (decode reuses the routed experts' quantized input)
        let [w1, w2, w3] = &ly.shared;
        let fused = match (&mut xq, w1, w3) {
            (Some(xq), DW::Fp8 { w: a, s: sa, n, k }, DW::Fp8 { w: b, s: sb, .. }) => g.shared_gate_up(&xq.as_view(), a, sa, b, sb, *n, *k, lim)?,
            _ => None,
        };
        let hbuf = match fused {
            Some(h) => h,
            None => {
                let (gate, up) = (w1.forward(g, &x.as_view(), t, Out::Bf16)?, w3.forward(g, &x.as_view(), t, Out::Bf16)?);
                let mut hbuf = g.alloc::<f32>(t * self.cfg.moe_inter_dim)?;
                g.swiglu(&gate, &up, None, &mut hbuf, self.cfg.moe_inter_dim, t, lim)?;
                hbuf
            }
        };
        let sh = w2.forward(g, &hbuf.as_view(), t, Out::Bf16)?;
        let (mut cpu_s, mut cpu_uses) = (0.0, 0);
        let mut out_y = g.alloc::<f32>(t * d)?;
        match out {
            MoeOut::Grouped(mut outs, nexp) => match (on_cpu, self.cpu.as_ref()) {
                (CpuPart::Blocking(jobs, x_host), Some(cpu)) => {
                    let tcpu = Instant::now();
                    let recs: Vec<Arc<Vec<u8>>> = jobs.iter().map(|(_, lease, _)| lease.to_arc()).collect();
                    let ws: Vec<f32> = jobs.iter().map(|&(_, _, w)| w).collect();
                    for ((row, _, _), y) in jobs.iter().zip(cpu.forward(&recs, &ws, &x_host, lim)) {
                        g.write(&y, &mut outs.slice_mut(row * DIM..(row + 1) * DIM))?;
                    }
                    cpu_s = tcpu.elapsed().as_secs_f64();
                    cpu_uses = jobs.len() as u64;
                    g.moe_reduce(&outs, &sh, &mut out_y, nexp, d)?
                }
                (CpuPart::Launched { mask, seq, uses }, _) => {
                    cpu_uses = uses;
                    g.moe_reduce_host(&outs, &sh, &mut out_y, nexp, d, handoff, mask, seq)?
                }
                _ => g.moe_reduce(&outs, &sh, &mut out_y, nexp, d)?,
            },
            MoeOut::Summed(y) => g.add_round(&y, &sh.as_view(), &mut out_y, t * d)?,
        }
        let out = out_y;
        let shared_s = elapsed(ts)?;
        if let Some(p) = self.profile.as_mut() {
            p.route += route_s;
            p.fetch += fetch;
            p.experts += compute;
            p.shared += shared_s - cpu_s; // the CPU experts ran inside that window
            p.cpu += cpu_s;
            p.cpu_uses += cpu_uses;
        }
        Ok((out, routes))
    }
}

/// The vision routing bias and the image mask, when the stretch holds image
/// tokens and the layer has the bias.
fn image_bias<'a>(ly: &'a GLayer, mask: &'a [bool]) -> Option<(&'a [f32], &'a [bool])> {
    match &ly.bias_vl {
        Some(vl) if !mask.is_empty() => Some((vl, mask)),
        _ => None,
    }
}

/// Read `keys` into the host cache, `threads` at a time (skipping resident
/// ones), stopping early when `stop` is raised. Returns how many were read.
/// Prefill's routed experts: each expert used by the `t` tokens of `x` is
/// fetched once (VRAM, else RAM or disk, uploaded) and run over all of its
/// tokens; the weighted results are summed per token (in expert-id order).
/// Returns the sum and the seconds spent fetching and computing (measured
/// only when `prof`, which synchronizes).
#[allow(clippy::too_many_arguments)]
fn routed_sum(
    g: &Gpu,
    dcache: &mut DeviceExpertCache,
    host: &Ecache,
    store: &dyn WeightStore,
    l: usize,
    x: &CudaView<'_, f32>,
    t: usize,
    routes: &[Route],
    lim: f32,
    prof: bool,
) -> Result<(CudaSlice<f32>, f64, f64)> {
    let (mut fetch, mut compute) = (0.0, 0.0);
    // a prefill uses each layer's experts once per pass: the RAM tier makes
    // room from the layers already done, not from the ones coming up
    struct Scan<'a>(&'a Ecache);
    impl Drop for Scan<'_> {
        fn drop(&mut self) {
            self.0.set_scan_layer(None);
        }
    }
    host.set_scan_layer(Some(l as u32));
    let _scan = Scan(host);
    let mut y = g.zeros::<f32>(t * DIM)?;
    let mut used: Vec<u32> = routes.iter().flat_map(|r| r.experts.iter().copied()).collect();
    used.sort_unstable();
    used.dedup();
    // Readers pull the experts VRAM lacks into the host cache, in the order
    // the loop below uses them, so the drive works (several reads deep)
    // while the GPU computes; the loop then finds them in RAM, or waits for
    // the read already under way.
    let ahead: Vec<u32> = used.iter().copied().filter(|&e| !dcache.contains(l as u32, e)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..PREFETCH_READERS.min(ahead.len()) {
            s.spawn(|| {
                use std::sync::atomic::Ordering::Relaxed;
                while !stop.load(Relaxed) {
                    let Some(&e) = ahead.get(next.fetch_add(1, Relaxed)) else { break };
                    // an error here resurfaces when the loop reads it itself
                    let _ = host.acquire(l as u32, e, store);
                }
            });
        }
        let out = compute_experts(g, dcache, host, store, l, x, routes, &used, lim, prof, &mut y, &mut fetch, &mut compute);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        out
    })?;
    Ok((y, fetch, compute))
}

/// Readers fetching a prefill layer's experts ahead of use.
const PREFETCH_READERS: usize = 4;

/// The per-expert loop of [`routed_sum`].
#[allow(clippy::too_many_arguments)]
fn compute_experts(
    g: &Gpu,
    dcache: &mut DeviceExpertCache,
    host: &Ecache,
    store: &dyn WeightStore,
    l: usize,
    x: &CudaView<'_, f32>,
    routes: &[Route],
    used: &[u32],
    lim: f32,
    prof: bool,
    y: &mut CudaSlice<f32>,
    fetch: &mut f64,
    compute: &mut f64,
) -> Result<()> {
    let elapsed = |t0: Option<Instant>| -> Result<f64> {
        match t0 {
            Some(t0) => {
                g.sync()?;
                Ok(t0.elapsed().as_secs_f64())
            }
            None => Ok(0.0),
        }
    };
    // every expert's tokens (in token order) and weights, in one pass
    let mut slot = std::collections::HashMap::with_capacity(used.len());
    for (i, &e) in used.iter().enumerate() {
        slot.insert(e, i);
    }
    let mut buckets: Vec<(Vec<i32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); used.len()];
    for (i, r) in routes.iter().enumerate() {
        for (&e, &w) in r.experts.iter().zip(&r.weights) {
            let b = &mut buckets[slot[&e]];
            b.0.push(i as i32);
            b.1.push(w);
        }
    }
    for (&e, (toks, ws)) in used.iter().zip(buckets) {
        let tf = if prof { g.sync()?; Some(Instant::now()) } else { None };
        // each prefill expert is its own batch: its kernels are issued
        // before the next fetch, and one stream orders that fetch's
        // upload after them, so only the slot in use needs pinning
        dcache.begin_batch();
        let rec = dcache.get_prefill(g, l as u32, e, toks.len() as u64, host, store)?;
        *fetch += elapsed(tf)?;
        let tc = if prof { Some(Instant::now()) } else { None };
        let nt = toks.len();
        let tok_idx = g.upload(&toks)?;
        let mut xs = g.alloc::<f32>(nt * DIM)?;
        g.gather_rows(x, &tok_idx, &mut xs, DIM, nt)?;
        g.act_quant_fp8(&mut xs.slice_mut(..))?;
        let (mut gate, mut up) = (g.alloc::<f32>(nt * INTER)?, g.alloc::<f32>(nt * INTER)?);
        // tensor cores for every prefill expert, however few its tokens: a
        // token's result must not depend on how many share the launch
        g.gemm_fp4(&xs.as_view(), &rec.slice(W1), &rec.slice(S1), &mut gate.slice_mut(..), INTER, DIM, nt, true)?;
        g.gemm_fp4(&xs.as_view(), &rec.slice(W3), &rec.slice(S3), &mut up.slice_mut(..), INTER, DIM, nt, true)?;
        let mut hbuf = g.alloc::<f32>(nt * INTER)?;
        g.swiglu(&gate, &up, Some(&g.upload(&ws)?), &mut hbuf, INTER, nt, lim)?;
        g.act_quant_fp8(&mut hbuf.slice_mut(..))?;
        let mut out = g.alloc::<f32>(nt * DIM)?;
        g.gemm_fp4(&hbuf.as_view(), &rec.slice(W2), &rec.slice(S2), &mut out.slice_mut(..), DIM, INTER, nt, true)?;
        g.scatter_add_rows(y, &out, &tok_idx, DIM, nt)?;
        *compute += elapsed(tc)?;
    }
    Ok(())
}

fn read_into(cache: &Ecache, store: &dyn WeightStore, keys: &[(u32, u32)], threads: usize, stop: Option<&std::sync::atomic::AtomicBool>) -> Result<usize> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (next, read) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let errors: Vec<Error> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.max(1))
            .map(|_| {
                s.spawn(|| -> Result<()> {
                    loop {
                        if stop.is_some_and(|f| f.load(Ordering::Relaxed)) {
                            return Ok(());
                        }
                        let Some(&(l, e)) = keys.get(next.fetch_add(1, Ordering::Relaxed)) else { return Ok(()) };
                        if !cache.probe(l, e) {
                            cache.acquire(l, e, store)?;
                            read.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        workers.into_iter().filter_map(|w| w.join().expect("warm reader").err()).collect()
    });
    match errors.into_iter().next() {
        Some(e) => Err(e),
        None => Ok(read.into_inner()),
    }
}

/// Demand reads of expert records, for the background fill to yield to.
struct Demand {
    /// A prefill is running (it reads many records back to back).
    prefill: std::sync::atomic::AtomicBool,
    /// The owner asked the fill to stay off the drive (a server handling a
    /// request: its decode misses should not queue behind fill reads).
    busy: std::sync::atomic::AtomicBool,
    /// Milliseconds since `epoch` of the latest demand read's start or end.
    last: std::sync::atomic::AtomicU64,
    epoch: Instant,
}

impl Demand {
    /// How long after a demand read the background fill stays off the drive.
    const QUIET_MS: u64 = 50;

    fn new() -> Demand {
        use std::sync::atomic::{AtomicBool, AtomicU64};
        Demand { prefill: AtomicBool::new(false), busy: AtomicBool::new(false), last: AtomicU64::new(0), epoch: Instant::now() }
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64 + Self::QUIET_MS + 1
    }

    fn read(&self) {
        self.last.store(self.now(), std::sync::atomic::Ordering::Relaxed);
    }

    fn recent(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        self.prefill.load(Relaxed) || self.busy.load(Relaxed) || self.now().saturating_sub(self.last.load(Relaxed)) < Self::QUIET_MS
    }
}

/// A background host-cache fill (see [`GpuModel::warm`]).
struct Warming {
    stop: Arc<std::sync::atomic::AtomicBool>,
    done: Arc<std::sync::atomic::AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Warming {
    fn spawn(cache: Arc<Ecache>, store: Arc<dyn WeightStore>, keys: Vec<(u32, u32)>, demand: Arc<Demand>) -> Warming {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let (stop, done) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(0)));
        let (stop2, done2) = (Arc::clone(&stop), Arc::clone(&done));
        let handle = std::thread::Builder::new()
            .name("dsv41-warm".into())
            .spawn(move || {
                // one read at a time, and only while the model is not waiting
                // on the drive itself: a demand read always goes first
                for key in keys.chunks(1) {
                    while demand.recent() && !stop2.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    if stop2.load(Ordering::Relaxed) {
                        return;
                    }
                    // a failed read only leaves that record cold
                    if let Ok(n) = read_into(&cache, store.as_ref(), key, 1, Some(&stop2)) {
                        done2.fetch_add(n, Ordering::Relaxed);
                    }
                }
            })
            .ok();
        Warming { stop, done, handle }
    }

    fn stop(mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Warming {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Backbone for GpuModel {
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn config_mut(&mut self) -> &mut Config {
        &mut self.cfg
    }
    fn forward_traced(&mut self, ids: &[u32], start_pos: usize, teacher: Option<Teacher<'_>>, trace: Trace<'_>) -> Result<Vec<f32>> {
        let images = std::mem::take(&mut self.trace_images);
        let out = self.run(ids, start_pos, teacher, Some(trace), &images);
        self.trace_images = images;
        out
    }
}
