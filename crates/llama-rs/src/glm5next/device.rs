// VENDORED-LOCAL: whole module. GLM-5.3-Flash device forward.
//! Runs the glm5next forward pass with its matrices on a [`Backend`] — CUDA when
//! the `cuda` feature is on, the CPU backend otherwise.
//!
//! ## One implementation, two backends
//!
//! There is no second forward pass here. [`super::forward`] is parameterised over
//! [`super::forward::Mat`], so the same trunk loop runs with host `&[f32]`
//! matrices ([`super::bridge::HostModel`]) or with device `Weight`s (this
//! module). That means the device path cannot drift from the reference, and
//! [`tests::device_and_host_matrices_agree`] checks the two agree numerically.
//!
//! ## What runs where
//!
//! Every matrix multiply — the KDA projections, the MLA projections, the indexer
//! projections, the hyper-connection mixer, the dense FFNs, the shared experts,
//! the routed experts and the LM head — goes through the backend. That is about
//! **99% of the arithmetic**: roughly 16 GMAC per token, of which the bespoke
//! scalar work (the 4x4 Sinkhorn, the KDA recurrence, indexer scoring, the
//! clamps) is around 160 MMAC.
//!
//! Left on the host, deliberately:
//!
//!   * **The KDA recurrence.** `Backend::delta_net_step` exists but decays per
//!     *head*; glm5next decays per *channel* (see [`super::kda`]), so it cannot
//!     be reused. A dedicated kernel is the obvious next step.
//!   * **The hyper-connection tail** — 24 numbers and a 4x4 Sinkhorn per
//!     sublayer. Not worth a kernel.
//!   * **Indexer scoring and top-k**, and the sparse attention mask. The
//!     reference's CUDA path for this has two open upstream bugs
//!     (see [`super::kpool`]), so the host version is the safer starting point.
//!   * **The clamped SwiGLU**, because there is no `clamp` backend op yet.
//!
//! ## Cost of the seam
//!
//! `Mat::Device` copies the activation in and the result out per call. For a
//! `[8192, 4096]` projection that is 16 KB against 34 MMAC, which is the right
//! trade. Keeping activations resident between stages would remove those copies
//! and is the next optimisation; it is not needed for correctness.
//!
//! Routed experts are uploaded **quantised** (about 4.7 MB per projection at
//! Q4_K rather than 33 MB dequantised), so a token moves roughly 113 MB over
//! PCIe for its 8 experts. `expert_stream`'s VRAM expert cache is the way to
//! avoid re-uploading hot experts; wiring this path to it is the follow-up.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use ggml_rs::quantized::QuantizedTensor;
use ggml_rs::{Backend, Tensor};
use gguf::GgufFile;

use super::bridge::PartRef;
use crate::expert_stream::{
    ExpertLayout, GgufExpertStore, LayerStream, ResolvedExpert, StreamShared,
};
use super::forward::{
    self, AttnW, Bat, ExpertFfn, FfnW, HcW, IndexerW, KdaW, LayerW, Mat, MlaW, ModelW, MoeW, Shape,
};
use super::{Glm5NextConfig, LayerKind};
use crate::config::ModelConfig;
use crate::loader::{TensorIndex, Weight};
use crate::{LlamaError, Result};

/// Routed experts, read quantised from the `.gguf` and multiplied on the device.
pub struct DeviceExperts {
    file: GgufFile,
    backend: Arc<dyn Backend>,
    layers: Vec<[PartRef; 3]>,
    n_ff_exp: usize,
    n_embd: usize,
    bytes: std::cell::RefCell<Vec<u8>>,
}

impl DeviceExperts {
    pub fn new(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        first_moe: usize,
        n_moe: usize,
        n_expert: usize,
        n_ff_exp: usize,
        n_embd: usize,
    ) -> Result<Self> {
        let (layers, max_per) = PartRef::collect(g, first_moe, n_moe, n_expert)?;
        Ok(Self {
            file: g.clone(),
            backend,
            layers,
            n_ff_exp,
            n_embd,
            bytes: std::cell::RefCell::new(vec![0u8; max_per]),
        })
    }

    /// Upload one expert slice as a quantised weight and apply it to `x`.
    fn linear_part(&self, p: &PartRef, e: usize, x: &Tensor, shape: Vec<usize>) -> Result<Tensor> {
        let mut buf = self.bytes.borrow_mut();
        let raw = &mut buf[..p.per()];
        p.read_raw(&self.file, e, raw)?;
        // Quantised upload: a Q4_K expert projection is ~4.7 MB, against ~33 MB
        // if it were dequantised first.
        let qt = QuantizedTensor::from_bytes_cpu(raw.to_vec(), shape, p.dtype());
        let w = Weight::Quant(self.backend.to_device_quant(qt));
        Ok(w.linear(&*self.backend, x))
    }
}

impl ExpertFfn for DeviceExperts {
    fn apply(&self, ord: usize, e: usize, x: &[f32], limit: f32, out: &mut [f32]) -> Result<()> {
        let parts = self.layers.get(ord).ok_or_else(|| {
            LlamaError::Config(format!(
                "device: MoE layer {ord} out of range (have {})",
                self.layers.len()
            ))
        })?;
        if x.len() != self.n_embd || out.len() != self.n_embd {
            return Err(LlamaError::Config(format!(
                "device: expert FFN got x {} / out {}, expected n_embd {}",
                x.len(),
                out.len(),
                self.n_embd
            )));
        }
        let (ff, e_dim) = (self.n_ff_exp, self.n_embd);
        let xd = self
            .backend
            .to_device(Tensor::from_vec(x.to_vec(), vec![1, e_dim]));

        let gd = self.linear_part(&parts[0], e, &xd, vec![ff, e_dim])?;
        let ud = self.linear_part(&parts[1], e, &xd, vec![ff, e_dim])?;

        // The clamp has no backend op, so the gate/up pair comes home for it.
        // 2048 floats each.
        let gh = self.backend.to_host(gd);
        let uh = self.backend.to_host(ud);
        let mut h = vec![0.0f32; ff];
        forward::swiglu_clamped(gh.data(), uh.data(), limit, &mut h)?;

        let hd = self.backend.to_device(Tensor::from_vec(h, vec![1, ff]));
        let od = self.linear_part(&parts[2], e, &hd, vec![e_dim, ff])?;
        let oh = self.backend.to_host(od);
        if oh.data().len() != out.len() {
            return Err(LlamaError::Config(format!(
                "device: expert output is {} values, expected {}",
                oh.data().len(),
                out.len()
            )));
        }
        out.copy_from_slice(oh.data());
        Ok(())
    }
}

/// Routed experts through `expert_stream`: an LFRU RAM cache, a batched
/// thread-pool read for a layer's misses, and optionally a VRAM expert cache.
///
/// [`DeviceExperts`] re-reads and re-uploads every expert on every dispatch,
/// which the profiler showed costs about 2.96 s a token: 1008 projections at
/// 4.7 MB each is **4.76 GB read per token**, at 2.74 GB/s because the reads are
/// one-at-a-time with no queue depth. That is essentially the whole token.
///
/// This path exists to remove that. The same weights are reconstructed through
/// `Ecache`, so a re-read only happens on a miss, and
/// [`StreamShared::enable_device_cache`] keeps the hot set in VRAM so a hit does
/// not cross PCIe either.
pub struct StreamExperts {
    shared: Arc<StreamShared>,
    /// Card 0. The trunk lives here and `DeviceModel` reports its name.
    backend: Arc<dyn Backend>,
    /// One per GPU the expert tier spans; `[backend]` when it spans one.
    ///
    /// A kernel reads only the card it was launched on, so a layer whose experts
    /// are cached on card 1 must have its expert FFN launched on card 1. That is
    /// free here: `apply_layer` takes host `x` and returns host `out`, so the
    /// hidden state crosses cards through the host copy the `Mat` seam already
    /// makes, with nothing extra to transfer.
    backends: Vec<Arc<dyn Backend>>,
    /// MoE ordinal -> index into `backends`.
    card_of: Vec<usize>,
    /// The VRAM budget each card's expert cache got, in card order.
    vram_budgets: Vec<usize>,
    /// One handle per MoE ordinal.
    layers: Vec<LayerStream>,
    n_embd: usize,
}

impl StreamExperts {
    /// `cache_budget_bytes` is the RAM the expert cache may use. One record is
    /// gate+up+down for a single expert, about 14 MB here, and the whole model
    /// has `n_moe * n_expert` of them — so the budget sets the hit rate directly.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g: &GgufFile,
        backend: Arc<dyn Backend>,
        first_moe: usize,
        n_moe: usize,
        n_expert: usize,
        n_embd: usize,
        cache_budget_bytes: usize,
    ) -> Result<Self> {
        let layout = ExpertLayout::resolve_range(g, first_moe, n_moe, n_expert)?;
        let expert_bytes = layout.total_expert_bytes();
        let total: u64 = g.tensors().iter().map(|t| t.nbytes()).sum();
        let resident_est = total.saturating_sub(expert_bytes);
        let store = GgufExpertStore::new(g.clone(), layout)?;
        let shared = StreamShared::new(store, cache_budget_bytes, resident_est)?;
        let layers = (0..n_moe).map(|o| shared.layer(o as u32)).collect();
        Ok(Self {
            shared,
            backends: vec![Arc::clone(&backend)],
            card_of: vec![0; n_moe],
            vram_budgets: Vec::new(),
            backend,
            layers,
            n_embd,
        })
    }

    /// The shared cache state, so a caller can turn on the VRAM tier.
    pub fn shared(&self) -> &Arc<StreamShared> {
        &self.shared
    }

    /// VRAM held back on each card: room for the activations, the pinned staging
    /// ring, cuBLAS workspaces and the driver's own overhead.
    ///
    /// Decode activations are tiny -- a 4 x 4096 residual stream is 64 KB, the
    /// widest FFN intermediate 8 KB -- but a matvec still needs somewhere to put
    /// its output, and the pinned staging ring is three records. Override with
    /// GLM5_VRAM_HEADROOM_MB.
    pub const VRAM_HEADROOM: usize = 2 << 30;

    /// Fraction of the remaining VRAM the expert cache may be *charged*.
    ///
    /// The cache counts the bytes it uploads, but each admitted expert is two or
    /// three separate `device_slot` allocations of about 4.7 MB, and CUDA rounds
    /// every one up to its allocation granularity. Measured on a 192-token
    /// document: a cache charged 23 GB had the card at 24.4 GB, about 6% over --
    /// and with only a 1 GB headroom the next staging slot hit
    /// CUDA_ERROR_OUT_OF_MEMORY partway through the prefill. 0.88 leaves room for
    /// that rounding and for the driver's own growth.
    pub const VRAM_CHARGE_FRACTION: f64 = 0.88;

    fn vram_headroom() -> usize {
        std::env::var("GLM5_VRAM_HEADROOM_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb << 20)
            .unwrap_or(Self::VRAM_HEADROOM)
    }

    /// Spread the expert tier over several GPUs, MoE layers dealt out in
    /// contiguous runs, each card's cache sized to what is actually free on it.
    ///
    /// Contiguous rather than round-robin so each card sees the same layers on
    /// every token and its LFRU set converges, and so the hidden state crosses
    /// between cards once per boundary instead of on every layer.
    ///
    /// Sizes differ between cards because their loads do: card 0 carries the
    /// 5.97 GB trunk, the others carry nothing, so asking every card for the same
    /// budget leaves several GB unused on every card but the first. `cap` bounds
    /// each card if a caller wants to hold some back; 0 means take what is free.
    #[cfg(feature = "cuda")]
    pub fn spread_over(
        &mut self,
        cards: Vec<Arc<ggml_rs_cuda::CudaBackend>>,
        cap: usize,
    ) -> Result<()> {
        if cards.is_empty() {
            return Err(LlamaError::Config("device: no GPUs for the expert tier".into()));
        }
        // What each card can actually give, measured on the card.
        let rec = self.shared.record_bytes();
        let head = Self::vram_headroom();
        let mut budgets = Vec::with_capacity(cards.len());
        for c in &cards {
            let free = match c.vram_status() {
                Some((free, total)) => {
                    eprintln!(
                        "  card: {:.1} GB free of {:.1} GB, holding back {:.1} GB",
                        free as f64 / 1e9,
                        total as f64 / 1e9,
                        head as f64 / 1e9
                    );
                    free.saturating_sub(head)
                }
                // No query: fall back to the cap, or a conservative slice.
                None => {
                    if cap > 0 {
                        cap
                    } else {
                        8 << 30
                    }
                }
            };
            let usable = (free as f64 * Self::VRAM_CHARGE_FRACTION) as usize;
            let b = if cap > 0 { usable.min(cap) } else { usable };
            // A cache smaller than a layer's route cannot hold one, so skip it.
            budgets.push(if b / rec.max(1) >= 8 { b } else { 0 });
        }
        let n_moe = self.layers.len();
        let n = cards.len();

        // Contiguous runs, sized in proportion to each card's cache rather than
        // evenly, because the caches are not equal: card 0 carries the trunk.
        //
        // An even 21/21 split over budgets of 1312 and 1635 slots gave each of
        // card 0's layers 62 slots and each of card 1's 78, and the hit rates
        // followed exactly that -- 69.9% against 81.1%. Proportional gives 19/23,
        // so about 69 and 71 slots a layer, which is the point: what a layer needs
        // is slots for *its own* 288 experts, so the thing to equalise is slots per
        // layer, not layers per card.
        let total: usize = budgets.iter().sum();
        let mut counts = vec![0usize; n];
        if total == 0 {
            // No usable cache anywhere: fall back to an even split.
            for (c, slot) in counts.iter_mut().enumerate() {
                *slot = n_moe / n + usize::from(c < n_moe % n);
            }
        } else {
            // Largest-remainder, so the counts sum to n_moe exactly and every card
            // with a cache gets at least one layer.
            let mut rema: Vec<(f64, usize)> = Vec::with_capacity(n);
            let mut used = 0usize;
            for (c, &b) in budgets.iter().enumerate() {
                let exact = n_moe as f64 * b as f64 / total as f64;
                let floor = exact.floor() as usize;
                counts[c] = floor;
                used += floor;
                rema.push((exact - floor as f64, c));
            }
            rema.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for i in 0..(n_moe - used) {
                counts[rema[i % rema.len()].1] += 1;
            }
        }
        let mut card_of = Vec::with_capacity(n_moe);
        for (c, &count) in counts.iter().enumerate() {
            card_of.extend(std::iter::repeat_n(c, count));
        }
        debug_assert_eq!(card_of.len(), n_moe);
        eprintln!(
            "  MoE layers {:?} over budgets {:?} GB -> slots a layer {:?}",
            counts,
            budgets.iter().map(|b| b >> 30).collect::<Vec<_>>(),
            counts
                .iter()
                .zip(&budgets)
                .map(|(&c, &b)| if c > 0 { b / rec.max(1) / c } else { 0 })
                .collect::<Vec<_>>()
        );

        self.shared
            .enable_device_shards(&cards, &budgets, card_of.clone())
            .map_err(|e| LlamaError::Config(format!("device: VRAM tier: {e}")))?;
        self.vram_budgets = budgets;
        self.backends = cards
            .into_iter()
            .map(|c| c as Arc<dyn Backend>)
            .collect();
        self.card_of = card_of;
        Ok(())
    }

    /// Which card runs MoE layer `ord`, and how many layers each card holds.
    pub fn card_layout(&self) -> (&[usize], usize) {
        (&self.card_of, self.backends.len())
    }

    /// The VRAM expert-cache budget each card ended up with, in card order.
    pub fn vram_budgets(&self) -> &[usize] {
        &self.vram_budgets
    }

    pub fn record_bytes(&self) -> usize {
        self.shared.record_bytes()
    }
}

impl StreamExperts {
    fn layer(&self, ord: usize) -> Result<&LayerStream> {
        self.layers.get(ord).ok_or_else(|| {
            LlamaError::Config(format!(
                "device: MoE layer {ord} out of range (have {})",
                self.layers.len()
            ))
        })
    }

    /// One resolved expert's FFN, input already on `be`: the clamp is fused, so
    /// the gate/up pair never leaves the device.
    fn run_expert(&self, be: &dyn Backend, r: &ResolvedExpert, xd: &Tensor, limit: f32) -> Tensor {
        let h = r.pair().swiglu_clamped(be, xd, limit, true);
        r.down().linear(be, &h)
    }

    /// The backend that runs MoE layer `ord`.
    fn card(&self, ord: usize) -> &dyn Backend {
        let i = self.card_of.get(ord).copied().unwrap_or(0);
        &*self.backends[i.min(self.backends.len() - 1)]
    }
}

impl ExpertFfn for StreamExperts {
    /// The whole layer in one host round trip.
    ///
    /// The input is uploaded once, every expert's output is accumulated into a
    /// device-resident sum through the fused
    /// [`Backend::add_to_axis0_range_scaled`], and only that sum comes back. The
    /// per-expert path below did `n_expert_used` uploads and the same number of
    /// synchronising reads; at 32 us a round trip and 336 dispatches a token,
    /// that was ~21 ms of pure latency.
    fn apply_layer(
        &self,
        ord: usize,
        experts: &[(u32, f32)],
        x: &[f32],
        limit: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let ls = self.layer(ord)?;
        if x.len() != self.n_embd || out.len() != self.n_embd {
            return Err(LlamaError::Config(format!(
                "device: expert layer got x {} / out {}, expected n_embd {}",
                x.len(),
                out.len(),
                self.n_embd
            )));
        }
        // The whole route at once, so the VRAM tier is consulted for all of it
        // and the misses are staged in one batch before the first matvec.
        let ids: Vec<u32> = experts.iter().map(|&(e, _)| e).collect();
        let resolved = ls
            .resolve_experts(&ids)
            .map_err(|err| LlamaError::Config(format!("device: MoE layer {ord}: {err}")))?;

        let be = self.card(ord);
        let xd = be.to_device(Tensor::from_vec(x.to_vec(), vec![1, self.n_embd]));
        let mut acc =
            be.to_device(Tensor::from_vec(vec![0.0f32; self.n_embd], vec![1, self.n_embd]));
        for (r, &(_, wt)) in resolved.iter().zip(experts) {
            let o = self.run_expert(be, r, &xd, limit);
            be.add_to_axis0_range_scaled(&mut acc, 0, 1, &o, wt);
        }
        let oh = be.to_host(acc);
        if oh.data().len() != out.len() {
            return Err(LlamaError::Config(format!(
                "device: expert layer sum is {} values, expected {}",
                oh.data().len(),
                out.len()
            )));
        }
        out.copy_from_slice(oh.data());
        Ok(())
    }

    fn apply(&self, ord: usize, e: usize, x: &[f32], limit: f32, out: &mut [f32]) -> Result<()> {
        let ls = self.layer(ord)?;
        if x.len() != self.n_embd || out.len() != self.n_embd {
            return Err(LlamaError::Config(format!(
                "device: expert FFN got x {} / out {}, expected n_embd {}",
                x.len(),
                out.len(),
                self.n_embd
            )));
        }
        // Cached reconstruction: a hit costs no disk read, and with the VRAM tier
        // on, no upload either.
        let resolved = ls
            .resolve_experts(&[e as u32])
            .map_err(|err| LlamaError::Config(format!("device: expert ({ord}, {e}): {err}")))?;
        let be = self.card(ord);
        let xd = be.to_device(Tensor::from_vec(x.to_vec(), vec![1, self.n_embd]));
        let o = self.run_expert(be, &resolved[0], &xd, limit);
        let oh = be.to_host(o);
        if oh.data().len() != out.len() {
            return Err(LlamaError::Config(format!(
                "device: expert output is {} values, expected {}",
                oh.data().len(),
                out.len()
            )));
        }
        out.copy_from_slice(oh.data());
        Ok(())
    }
}

/// A released glm5next model with its matrices on a backend.
pub struct DeviceModel {
    shape: Shape,
    /// Matrices, resident on the backend.
    w: BTreeMap<String, Weight>,
    /// Vectors, host f32: the stages index these directly rather than
    /// multiplying by them.
    v: BTreeMap<String, Tensor>,
    /// The 3-D absorbed-MLA weights, dense f32 on the backend. They are not flat
    /// matrices -- each head takes its own input -- so they go through
    /// [`Bat`] rather than [`Mat`]. 11 MLA layers x 2 x 33.6 MB is 739 MB of
    /// VRAM, against the 61.7 ms a token they cost as host f32.
    b3: BTreeMap<String, Tensor>,
    experts: StreamExperts,
}

impl DeviceModel {
    /// RAM the expert cache gets when a caller does not say. One record is about
    /// 14 MB, so this buys roughly 4 600 of the model's 12 096 experts.
    pub const DEFAULT_EXPERT_CACHE: usize = 64 << 30;

    /// Open a released model. `path` may name any shard of a split GGUF.
    pub fn open(
        path: impl AsRef<Path>,
        max_len: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        Self::open_with_cache(path, max_len, backend, Self::DEFAULT_EXPERT_CACHE)
    }

    /// As [`Self::open`], with an explicit expert-cache budget in bytes.
    pub fn open_with_cache(
        path: impl AsRef<Path>,
        max_len: usize,
        backend: Arc<dyn Backend>,
        cache_budget_bytes: usize,
    ) -> Result<Self> {
        let g = GgufFile::open_streaming(path)?;
        Self::from_gguf(&g, max_len, backend, cache_budget_bytes)
    }

    /// The expert cache, so a caller can enable the VRAM tier.
    pub fn experts(&self) -> &StreamExperts {
        &self.experts
    }

    /// Mutable, for [`StreamExperts::spread_over`].
    pub fn experts_mut(&mut self) -> &mut StreamExperts {
        &mut self.experts
    }

    /// Host RAM left for everything else when the RAM tier sizes itself.
    ///
    /// The tier is a hard cap the cache never exceeds, but the process also holds
    /// the trunk's host copies, the tokenizer, the activations and the OS's own
    /// working set, and Windows will start paging long before it reports no free
    /// memory. Override with GLM5_RAM_RESERVE_GB.
    pub const RAM_RESERVE: usize = 24 << 30;

    /// The RAM-tier budget: everything free except [`Self::RAM_RESERVE`], capped
    /// at what the experts could possibly need (the tier never benefits from
    /// being larger than the whole expert set).
    #[cfg(feature = "cuda")]
    fn auto_ram_budget(expert_bytes: u64) -> usize {
        let reserve = std::env::var("GLM5_RAM_RESERVE_GB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|gb| gb << 30)
            .unwrap_or(Self::RAM_RESERVE);
        let free = match ggml_rs_cuda::host_memory() {
            Some((free, total)) => {
                eprintln!(
                    "  host RAM: {:.1} GB free of {:.1} GB, reserving {:.1} GB",
                    free as f64 / 1e9,
                    total as f64 / 1e9,
                    reserve as f64 / 1e9
                );
                free
            }
            // No reader for this platform: a conservative fixed tier.
            None => {
                eprintln!("  host RAM: unknown, using a 32 GB tier");
                (32 << 30) + reserve
            }
        };
        free.saturating_sub(reserve).min(expert_bytes as usize)
    }

    /// Open the released model with the expert tier spread over every visible
    /// GPU: the trunk on card 0, a VRAM expert cache on each card sized to what
    /// is free there, and a RAM cache behind them.
    ///
    /// This is the full hierarchy: VRAM for the hot experts, RAM for the rest,
    /// the GGUF on the SSD for what neither holds. `ram_budget` of 0 sizes the
    /// RAM tier from free host memory, `vram_cap` of 0 takes what each card has.
    #[cfg(feature = "cuda")]
    pub fn open_tiered(
        path: impl AsRef<Path>,
        max_len: usize,
        n_gpus: usize,
        ram_budget: usize,
        vram_cap: usize,
    ) -> Result<Self> {
        let mut cards = Vec::new();
        for i in 0..n_gpus.max(1) {
            match ggml_rs_cuda::CudaBackend::new(i) {
                Ok(b) => cards.push(Arc::new(b)),
                Err(e) if i == 0 => {
                    return Err(LlamaError::Config(format!("device: no CUDA device 0: {e}")))
                }
                // Fewer cards than asked for is a smaller tier, not a failure.
                Err(_) => break,
            }
        }
        let trunk: Arc<dyn Backend> = Arc::clone(&cards[0]) as Arc<dyn Backend>;

        // The RAM tier wants sizing against the expert bytes, which needs the
        // layout, which needs the file open. One cheap pass for the header.
        let ram = if ram_budget > 0 {
            ram_budget
        } else {
            let g = GgufFile::open_streaming(path.as_ref())?;
            let expert_bytes: u64 = g
                .tensors()
                .iter()
                .filter(|t| t.name.contains("_exps"))
                .map(|t| t.nbytes())
                .sum();
            Self::auto_ram_budget(expert_bytes)
        };

        let mut m = Self::open_with_cache(path, max_len, trunk, ram)?;
        m.experts_mut().spread_over(cards, vram_cap)?;
        Ok(m)
    }

    pub fn from_gguf(
        g: &GgufFile,
        max_len: usize,
        backend: Arc<dyn Backend>,
        cache_budget_bytes: usize,
    ) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(g)?;
        let glm = Glm5NextConfig::from_gguf(g, &cfg)?;
        let idx = TensorIndex::new(g);

        let mut w: BTreeMap<String, Weight> = BTreeMap::new();
        let mut v: BTreeMap<String, Tensor> = BTreeMap::new();
        let mut b3: BTreeMap<String, Tensor> = BTreeMap::new();

        // Leave a little headroom so a tight VRAM budget degrades to host
        // residency rather than failing the load.
        const MARGIN: usize = 512 * 1024 * 1024;
        let mut mat = |name: String| -> Result<()> {
            let weight = idx.take_weight(&name, &[])?;
            w.insert(name, weight.try_to_device(&*backend, MARGIN));
            Ok(())
        };
        let mut vec_ = |name: String| -> Result<()> {
            let t = idx.take(&name, &[])?;
            v.insert(name, t);
            Ok(())
        };
        // A 3-D absorb weight: dequantised once at load, then resident.
        let mut bat_ = |name: String| -> Result<()> {
            let t = idx.take(&name, &[])?;
            b3.insert(name, backend.to_device(t));
            Ok(())
        };

        // The embedding table is indexed by row, so it stays host f32.
        vec_("token_embd.weight".to_string())?;
        vec_("output_norm.weight".to_string())?;
        if g.tensor_by_name("output.weight").is_some() {
            mat("output.weight".to_string())?;
        }

        for il in 0..glm.n_layer {
            for s in ["attn_norm.weight", "ffn_norm.weight"] {
                vec_(format!("blk.{il}.{s}"))?;
            }
            for s in ["hc_attn_fn.weight", "hc_ffn_fn.weight"] {
                mat(format!("blk.{il}.{s}"))?;
            }
            for s in [
                "hc_attn_base.weight",
                "hc_attn_scale.weight",
                "hc_ffn_base.weight",
                "hc_ffn_scale.weight",
            ] {
                vec_(format!("blk.{il}.{s}"))?;
            }

            match glm.layer_kinds[il] {
                LayerKind::Kda => {
                    for s in [
                        "attn_q.weight",
                        "attn_k.weight",
                        "attn_v.weight",
                        "attn_output.weight",
                        "ssm_f_a.weight",
                        "ssm_f_b.weight",
                        "ssm_g_a.weight",
                        "ssm_g_b.weight",
                        "ssm_beta.weight",
                    ] {
                        mat(format!("blk.{il}.{s}"))?;
                    }
                    for s in [
                        "ssm_conv1d_q.weight",
                        "ssm_conv1d_k.weight",
                        "ssm_conv1d_v.weight",
                        "ssm_a",
                        "ssm_dt.bias",
                        "ssm_norm.weight",
                    ] {
                        vec_(format!("blk.{il}.{s}"))?;
                    }
                }
                LayerKind::Mla => {
                    for s in [
                        "attn_q_a.weight",
                        "attn_q_b.weight",
                        "attn_kv_a_mqa.weight",
                        "attn_output.weight",
                        "indexer.attn_k.weight",
                        "indexer.attn_q_b.weight",
                        "indexer.proj.weight",
                        "indexer_compressor_gate.weight",
                    ] {
                        mat(format!("blk.{il}.{s}"))?;
                    }
                    // k_b / v_b take a different input per head, so they are a
                    // stack of matvecs rather than one matrix -- but they are
                    // still multiplied by, and they are the trunk's largest
                    // weights, so they belong on the backend.
                    for s in ["attn_k_b.weight", "attn_v_b.weight"] {
                        bat_(format!("blk.{il}.{s}"))?;
                    }
                    for s in [
                        "attn_q_a_norm.weight",
                        "attn_kv_a_norm.weight",
                        "indexer.k_norm.weight",
                        "indexer.k_norm.bias",
                        "indexer_compressor_ape.weight",
                    ] {
                        vec_(format!("blk.{il}.{s}"))?;
                    }
                }
            }

            if il < glm.n_dense_lead {
                for s in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    mat(format!("blk.{il}.{s}"))?;
                }
            } else {
                for s in [
                    "ffn_gate_inp.weight",
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                ] {
                    mat(format!("blk.{il}.{s}"))?;
                }
                vec_(format!("blk.{il}.exp_probs_b.bias"))?;
            }
        }

        let n_moe = glm.n_layer - glm.n_dense_lead;
        let experts = StreamExperts::new(
            g,
            backend.clone(),
            glm.n_dense_lead,
            n_moe,
            glm.n_expert,
            cfg.embedding_dim,
            cache_budget_bytes,
        )?;

        let shape = super::bridge::shape_from(&cfg, &glm, max_len);
        Ok(Self { shape, w, v, b3, experts })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The backend the trunk lives on (card 0). `State::new_on` wants it, so the
    /// KDA recurrent state is resident on the same card as the matrices feeding it.
    pub fn backend(&self) -> Arc<dyn Backend> {
        Arc::clone(&self.experts.backend)
    }

    pub fn backend_name(&self) -> String {
        // The experts hold the backend handle.
        self.experts.backend.name().to_string()
    }

    fn vec_of(&self, name: &str) -> &[f32] {
        self.v
            .get(name)
            .unwrap_or_else(|| panic!("device: vector {name} was not loaded"))
            .data()
    }

    fn bat_of(&self, name: &str) -> Bat<'_> {
        let t = self
            .b3
            .get(name)
            .unwrap_or_else(|| panic!("device: absorb tensor {name} was not loaded"));
        Bat::Device {
            t,
            backend: &*self.experts.backend,
        }
    }

    fn mat_of(&self, name: &str) -> Mat<'_> {
        let w = self
            .w
            .get(name)
            .unwrap_or_else(|| panic!("device: matrix {name} was not loaded"));
        Mat::Device {
            w,
            backend: &*self.experts.backend,
        }
    }

    /// As [`Self::view`], with the routed experts replaced.
    ///
    /// Measuring the dense trunk on its own needs an [`ExpertFfn`] that costs
    /// nothing; what is left is attention, the shared expert, the
    /// hyper-connections and the head. That number is a floor no amount of
    /// expert tiering can get under.
    pub fn view_with_experts<'a>(&'a self, experts: &'a dyn ExpertFfn) -> ModelW<'a> {
        let mut w = self.view();
        for l in &mut w.layers {
            if let FfnW::Moe(moe) = &mut l.ffn {
                moe.experts = experts;
            }
        }
        w
    }

    /// What actually ended up on the backend.
    ///
    /// `Weight::try_to_device` silently degrades to host residency when the
    /// device is short of room (that is the point of its safety margin), so a
    /// load that succeeds says nothing about where the trunk is. Returns
    /// `(device bytes, host bytes, host-resident names)` over the matrix map and
    /// the 3-D absorb tensors.
    pub fn residency(&self) -> (u64, u64, Vec<String>) {
        let mut on_dev = 0u64;
        let mut on_host = 0u64;
        let mut stragglers = Vec::new();
        let mut note = |name: &str, bytes: u64, dev: bool| {
            if dev {
                on_dev += bytes;
            } else {
                on_host += bytes;
                stragglers.push(format!("{name} ({:.1} MB)", bytes as f64 / 1e6));
            }
        };
        for (name, w) in &self.w {
            let (bytes, dev) = match w {
                Weight::Dense(t) => (t.numel() as u64 * 4, t.is_device()),
                Weight::Quant(qt) => (qt.nbytes() as u64, qt.is_device()),
                // The tied embedding is indexed by row on the host by design.
                Weight::TiedEmbed(t) => (t.numel() as u64 * 4, t.is_device()),
            };
            note(name, bytes, dev);
        }
        for (name, t) in &self.b3 {
            note(name, t.numel() as u64 * 4, t.is_device());
        }
        (on_dev, on_host, stragglers)
    }

    /// Borrow the weights as the view [`super::forward`] takes.
    pub fn view(&self) -> ModelW<'_> {
        let sh = &self.shape;
        let mut layers = Vec::with_capacity(sh.n_layer);
        for il in 0..sh.n_layer {
            let b = |s: &str| self.vec_of(&format!("blk.{il}.{s}"));
            let m = |s: &str| self.mat_of(&format!("blk.{il}.{s}"));
            let attn = match sh.layer_kinds[il] {
                LayerKind::Kda => AttnW::Kda(KdaW {
                    q: m("attn_q.weight"),
                    k: m("attn_k.weight"),
                    v: m("attn_v.weight"),
                    conv_q: b("ssm_conv1d_q.weight"),
                    conv_k: b("ssm_conv1d_k.weight"),
                    conv_v: b("ssm_conv1d_v.weight"),
                    f_a: m("ssm_f_a.weight"),
                    f_b: m("ssm_f_b.weight"),
                    g_a: m("ssm_g_a.weight"),
                    g_b: m("ssm_g_b.weight"),
                    beta: m("ssm_beta.weight"),
                    a: b("ssm_a"),
                    dt_bias: b("ssm_dt.bias"),
                    o_norm: b("ssm_norm.weight"),
                    out: m("attn_output.weight"),
                }),
                LayerKind::Mla => AttnW::Mla(MlaW {
                    q_a: m("attn_q_a.weight"),
                    q_a_norm: b("attn_q_a_norm.weight"),
                    q_b: m("attn_q_b.weight"),
                    kv_a_mqa: m("attn_kv_a_mqa.weight"),
                    kv_a_norm: b("attn_kv_a_norm.weight"),
                    k_b: self.bat_of(&format!("blk.{il}.attn_k_b.weight")),
                    v_b: self.bat_of(&format!("blk.{il}.attn_v_b.weight")),
                    out: m("attn_output.weight"),
                    indexer: IndexerW {
                        attn_k: m("indexer.attn_k.weight"),
                        attn_q_b: m("indexer.attn_q_b.weight"),
                        k_norm: b("indexer.k_norm.weight"),
                        k_norm_bias: b("indexer.k_norm.bias"),
                        proj: m("indexer.proj.weight"),
                        comp_gate: m("indexer_compressor_gate.weight"),
                        comp_ape: b("indexer_compressor_ape.weight"),
                    },
                }),
            };
            let ffn = if il < sh.n_dense_lead {
                FfnW::Dense {
                    gate: m("ffn_gate.weight"),
                    up: m("ffn_up.weight"),
                    down: m("ffn_down.weight"),
                }
            } else {
                FfnW::Moe(MoeW {
                    router: m("ffn_gate_inp.weight"),
                    probs_b: b("exp_probs_b.bias"),
                    experts: &self.experts,
                    ord: il - sh.n_dense_lead,
                    sh_gate: m("ffn_gate_shexp.weight"),
                    sh_up: m("ffn_up_shexp.weight"),
                    sh_down: m("ffn_down_shexp.weight"),
                })
            };
            layers.push(LayerW {
                attn_norm: b("attn_norm.weight"),
                ffn_norm: b("ffn_norm.weight"),
                hc_attn: HcW {
                    fn_: m("hc_attn_fn.weight"),
                    base: b("hc_attn_base.weight"),
                    scale: b("hc_attn_scale.weight"),
                },
                hc_ffn: HcW {
                    fn_: m("hc_ffn_fn.weight"),
                    base: b("hc_ffn_base.weight"),
                    scale: b("hc_ffn_scale.weight"),
                },
                attn,
                ffn,
            });
        }
        ModelW {
            tok_embd: self.vec_of("token_embd.weight"),
            output_norm: self.vec_of("output_norm.weight"),
            output: if self.w.contains_key("output.weight") {
                self.mat_of("output.weight")
            } else {
                Mat::Host(self.vec_of("token_embd.weight"))
            },
            layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::forward::matvec;
    use ggml_quants::q4_k;

    const RELEASED: &str =
        r"D:\glm5.3_flash\Q4_K_M\GLM-5.3-Flash-Q4_K_M-00001-of-00005.gguf";

    /// The seam itself: a `Mat::Device` over a dense `Weight` must give the same
    /// numbers as `Mat::Host` over the same floats. This is what lets one forward
    /// implementation serve both paths, so it is worth pinning without a model.
    #[test]
    fn device_and_host_matrices_agree() {
        let backend = ggml_rs::default_backend();
        let (out_dim, in_dim) = (7usize, 5usize);
        let raw: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 11.0)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 2.0) / 3.0).collect();

        let host = Mat::Host(&raw);
        let mut a = vec![0.0f32; out_dim];
        host.apply(&x, &mut a).expect("host");

        let w = Weight::Dense(
            backend.to_device(Tensor::from_vec(raw.clone(), vec![out_dim, in_dim])),
        );
        let dev = Mat::Device {
            w: &w,
            backend: &*backend,
        };
        let mut b = vec![0.0f32; out_dim];
        dev.apply(&x, &mut b).expect("device");

        for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (p - q).abs() < 1e-5,
                "row {i}: host {p} vs device {q}"
            );
        }
        // And it is not trivially zero.
        assert!(a.iter().any(|&v| v.abs() > 1e-6));
    }

    // --- CUDA -------------------------------------------------------------
    // These are the same checks as above, on a real device. They skip rather
    // than fail when no GPU is reachable, matching how ggml-rs-cuda gates its
    // own tests.

    #[cfg(feature = "cuda")]
    fn cuda_backend() -> Option<Arc<dyn Backend>> {
        match ggml_rs_cuda::CudaBackend::new(0) {
            Ok(b) => Some(Arc::new(b) as Arc<dyn Backend>),
            Err(e) => {
                eprintln!("no CUDA device ({e}); skipping");
                None
            }
        }
    }

    /// The `Mat` seam on an actual GPU: a device matmul must agree with the host
    /// dot product. If this drifts, every glm5next number on CUDA is suspect.
    #[cfg(feature = "cuda")]
    #[test]
    fn matrices_agree_on_cuda() {
        let Some(backend) = cuda_backend() else { return };
        let (out_dim, in_dim) = (64usize, 128usize);
        let raw: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 37 % 199) as f32 - 99.0) / 99.0)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) / 15.0).collect();

        let mut host = vec![0.0f32; out_dim];
        Mat::Host(&raw).apply(&x, &mut host).expect("host");

        let w = Weight::Dense(
            backend.to_device(Tensor::from_vec(raw.clone(), vec![out_dim, in_dim])),
        );
        let mut dev = vec![0.0f32; out_dim];
        Mat::Device { w: &w, backend: &*backend }
            .apply(&x, &mut dev)
            .expect("cuda");

        let mut worst = 0.0f32;
        for (a, b) in host.iter().zip(dev.iter()) {
            worst = worst.max((a - b).abs());
        }
        println!("cuda vs host, {out_dim}x{in_dim}: max diff {worst:.3e}");
        assert!(worst < 1e-3, "cuda disagrees with host by {worst}");
        assert!(host.iter().any(|v| v.abs() > 1e-6));
    }

    /// A quantised expert upload, on the GPU: the path a routed expert takes
    /// every dispatch.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk"]
    fn a_real_expert_runs_on_cuda() {
        let Some(backend) = cuda_backend() else { return };
        let g = GgufFile::open_streaming(RELEASED).expect("open");
        let cfg = ModelConfig::from_gguf(&g).expect("cfg");
        let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm");

        let dev = DeviceExperts::new(
            &g,
            backend,
            glm.n_dense_lead,
            4,
            glm.n_expert,
            glm.n_ff_exp,
            cfg.embedding_dim,
        )
        .expect("device experts");
        let host = super::super::bridge::GgufExperts::new(
            &g,
            glm.n_dense_lead,
            4,
            glm.n_expert,
            glm.n_ff_exp,
            cfg.embedding_dim,
        )
        .expect("host experts");

        let x: Vec<f32> = (0..cfg.embedding_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 80.0)
            .collect();
        let mut a = vec![0.0f32; cfg.embedding_dim];
        let mut b = vec![0.0f32; cfg.embedding_dim];

        for (ord, e) in [(0usize, 0usize), (2, 150)] {
            dev.apply(ord, e, &x, 10.0, &mut a).expect("cuda expert");
            host.apply(ord, e, &x, 10.0, &mut b).expect("host expert");
            let worst = a
                .iter()
                .zip(b.iter())
                .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()));
            let scale = b.iter().fold(0.0f32, |m, q| m.max(q.abs()));
            println!("expert ({ord}, {e}): max diff {worst:.3e}, scale {scale:.3e}");
            assert!(a.iter().all(|v| v.is_finite()));
            assert!(
                worst <= 1e-3 * scale.max(1.0),
                "cuda expert differs from host by {worst} at scale {scale}"
            );
        }
    }

    /// The whole model on the GPU, and its logits against the host reference.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk; loads it twice"]
    fn released_model_agrees_with_host_on_cuda() {
        use super::super::bridge::HostModel;
        let Some(backend) = cuda_backend() else { return };

        let t0 = std::time::Instant::now();
        let dm = DeviceModel::open(RELEASED, 512, backend).expect("cuda load");
        println!(
            "cuda load in {:.1}s, backend {}",
            t0.elapsed().as_secs_f64(),
            dm.backend_name()
        );
        let ds = dm.shape().clone();
        let dv = dm.view();
        // `new_on`, not `new`: the recurrent state goes on the card so this gate
        // compares the CUDA `kda_delta_step` kernel against the host `kda::step`
        // oracle, which is the comparison worth making.
        let mut d_state = forward::State::new_on(&ds, dm.backend()).expect("state");
        assert!(d_state.kda_on_device(), "the KDA state should be resident");
        let t1 = std::time::Instant::now();
        let d = forward::forward_token(&ds, &dv, &mut d_state, 154822).expect("cuda forward");
        let cuda_secs = t1.elapsed().as_secs_f64();
        println!("cuda: one token in {cuda_secs:.1}s");
        drop(dv);
        drop(dm);

        let hm = HostModel::open(RELEASED, 512).expect("host load");
        let hs = hm.shape().clone();
        let hv = hm.view();
        let mut h_state = forward::State::new(&hs).expect("state");
        let t2 = std::time::Instant::now();
        let h = forward::forward_token(&hs, &hv, &mut h_state, 154822).expect("host forward");
        println!("host: one token in {:.1}s", t2.elapsed().as_secs_f64());

        assert_eq!(h.len(), d.len());
        let worst = h
            .iter()
            .zip(d.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let scale = h.iter().fold(0.0f32, |m, a| m.max(a.abs()));
        let n = h.len();
        println!("max |host - cuda| over {n} logits: {worst:.5} (scale {scale:.3})");
        assert!(d.iter().all(|v| v.is_finite()), "cuda logits must be finite");
        // Same dequantised bytes either side; only accumulation order differs.
        assert!(
            worst < 0.05 * scale.max(1.0),
            "cuda and host disagree by {worst} at scale {scale}"
        );
    }

    /// Greedy generation on the GPU, with the real tokenizer and chat format.
    ///
    /// This is the strongest end-to-end signal short of a numerical diff against
    /// llama.cpp: coherent text means the layer map, both attention kinds, the
    /// routing, the hyper-connections and the KDA recurrence are all essentially
    /// right, because almost any error in them degrades into noise.
    ///
    /// Note `max_len` of 512 is below the indexer threshold
    /// (`n_select` = 2051), so attention takes the **dense** path here — the same
    /// choice llama.cpp makes at this context size. The sparse path needs a
    /// longer context to engage.
    ///
    /// **Sampled, not greedy.** This model ships `general.sampling.temp = 1.0`
    /// and `top_p = 0.95`, which `SampleParams::default()` already matches.
    /// Greedy decoding was tried first and cycled: it produced one correct
    /// sentence of reasoning and then repeated it verbatim nine times over 192
    /// tokens. That is a known property of temperature-0 decoding on a reasoning
    /// model with no repetition penalty, not a defect in the trunk — the KDA
    /// decay was measured healthy on real weights at the same time
    /// (`bridge::tests::probe_the_kda_decay_on_real_weights`: median retention
    /// 0.85 to 0.97, so 6 to 30 tokens of memory).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk; generates on the GPU"]
    fn generates_text_on_cuda() {
        use crate::chat::{apply_chat_template, chat_stop_tokens, ChatMessage, Role};
        use crate::config::Architecture;

        let Some(backend) = cuda_backend() else { return };
        let g = GgufFile::open_streaming(RELEASED).expect("open");
        let tok = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
        let t0 = std::time::Instant::now();
        let m = DeviceModel::from_gguf(&g, 512, backend, DeviceModel::DEFAULT_EXPERT_CACHE)
            .expect("load");
        println!("load {:.1}s on {}", t0.elapsed().as_secs_f64(), m.backend_name());

        let sh = m.shape().clone();
        let w = m.view();
        let mut st = forward::State::new_on(&sh, m.backend()).expect("state");

        let msgs = [ChatMessage {
            role: Role::User,
            content: "What is the capital of France? Answer in one short sentence.".to_string(),
        }];
        let prompt = apply_chat_template(&Architecture::Glm5Next, &msgs, true);
        let ids = tok.encode(&prompt, false).expect("encode");
        println!("prompt: {} tokens", ids.len());
        assert!(!ids.is_empty(), "the chat template must tokenize");

        let mut sampler =
            crate::sampler::Sampler::new(crate::sampler::SampleParams::default());

        let t1 = std::time::Instant::now();
        let mut logits = Vec::new();
        for &t in &ids {
            logits = forward::forward_token(&sh, &w, &mut st, t).expect("prefill");
            sampler.observe(t);
        }
        println!(
            "prefill {:.1}s ({:.2}s/token)",
            t1.elapsed().as_secs_f64(),
            t1.elapsed().as_secs_f64() / ids.len() as f64
        );

        let stops: Vec<u32> = chat_stop_tokens(&Architecture::Glm5Next)
            .iter()
            .filter_map(|s| tok.token_id(s))
            .collect();
        println!("stop ids: {stops:?}");

        let t2 = std::time::Instant::now();
        let mut produced: Vec<u32> = Vec::new();
        for _ in 0..128 {
            let lt = Tensor::from_vec(logits.clone(), vec![1, sh.n_vocab]);
            let next = sampler.sample(&lt);
            if stops.contains(&next) {
                println!("(stop token {next})");
                break;
            }
            produced.push(next);
            sampler.observe(next);
            logits = forward::forward_token(&sh, &w, &mut st, next).expect("decode");
        }
        let secs = t2.elapsed().as_secs_f64();
        println!(
            "decode {:.1}s for {} tokens ({:.2}s/token)",
            secs,
            produced.len(),
            secs / produced.len().max(1) as f64
        );

        let text = tok.decode(&produced);
        println!("ids: {produced:?}");
        for &t in &produced {
            print!("[{}]", tok.decode(&[t]).escape_debug());
        }
        println!();
        println!("--- generated ---");
        println!("{text}");
        println!("--- end ---");
        println!("(escaped: {})", text.escape_debug());

        assert!(!produced.is_empty(), "nothing was generated");
        let distinct: std::collections::BTreeSet<u32> = produced.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "generation collapsed to one repeated token: {produced:?}"
        );
        // Coherence, robustly: real text is mostly printable ASCII with spaces
        // between words. Noise from a broken trunk is neither.
        let printable = text
            .chars()
            .filter(|c| c.is_ascii_graphic() || *c == ' ' || *c == '\n')
            .count();
        let spaces = text.chars().filter(|c| *c == ' ').count();
        println!(
            "{} chars, {printable} printable, {spaces} spaces",
            text.chars().count()
        );
        assert!(
            printable * 10 >= text.chars().count() * 9,
            "output should be mostly printable ASCII, got {printable}/{}",
            text.chars().count()
        );
        assert!(
            spaces * 12 >= text.chars().count(),
            "output should have word breaks; {spaces} spaces in {} chars looks like noise",
            text.chars().count()
        );

        // Degeneracy: a cycling decode reuses a handful of tokens. The greedy run
        // that prompted the switch to sampling had 24 distinct tokens in 192 --
        // 12% -- so this threshold separates the two.
        let distinct_frac = distinct.len() as f64 / produced.len() as f64;
        println!(
            "{} distinct of {} tokens ({:.0}%)",
            distinct.len(),
            produced.len(),
            distinct_frac * 100.0
        );
        assert!(
            distinct_frac > 0.25 || produced.len() < 24,
            "only {:.0}% distinct tokens -- the decode is cycling",
            distinct_frac * 100.0
        );
        // Degenerate output is the failure mode worth catching: a single token
        // repeated means the state or the routing is not advancing.
        assert!(!text.is_empty(), "detokenized to nothing");
    }

    /// Where a token actually goes. Splits the routed-expert path into its three
    /// phases so optimisation targets the dominant one rather than a guess.
    ///
    /// A token routes `n_expert_used` experts in each of 42 MoE layers, three
    /// projections each, so the counts below multiply by 8 * 3 * 42 = 1008.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk"]
    fn profile_the_expert_path() {
        let Some(backend) = cuda_backend() else { return };
        let g = GgufFile::open_streaming(RELEASED).expect("open");
        let cfg = ModelConfig::from_gguf(&g).expect("cfg");
        let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm");
        let n_moe = glm.n_layer - glm.n_dense_lead;

        let de = DeviceExperts::new(
            &g,
            backend.clone(),
            glm.n_dense_lead,
            n_moe,
            glm.n_expert,
            glm.n_ff_exp,
            cfg.embedding_dim,
        )
        .expect("experts");

        let per = de.layers[0][0].per();
        let projections_per_token = glm.n_expert_used * 3 * n_moe;
        println!(
            "{} bytes per expert projection, {projections_per_token} projections per token              ({:.2} GB)",
            per,
            (per * projections_per_token) as f64 / 1e9
        );

        let n = 96usize;
        let pick = |i: usize| (i % n_moe, (i * 37) % glm.n_expert);

        // 1. raw ranged reads only -- pure disk I/O.
        let t = std::time::Instant::now();
        for i in 0..n {
            let (ord, e) = pick(i);
            let pr = &de.layers[ord][i % 3];
            let mut buf = de.bytes.borrow_mut();
            let raw = &mut buf[..pr.per()];
            pr.read_raw(&de.file, e, raw).expect("read");
        }
        let read = t.elapsed().as_secs_f64() / n as f64;

        // 2. read + dequantise.
        let mut floats = vec![0.0f32; glm.n_ff_exp * cfg.embedding_dim];
        let t = std::time::Instant::now();
        for i in 0..n {
            let (ord, e) = pick(i);
            let pr = &de.layers[ord][i % 3];
            let mut buf = de.bytes.borrow_mut();
            let raw = &mut buf[..pr.per()];
            pr.read_raw(&de.file, e, raw).expect("read");
            ggml_quants::dequantize(pr.dtype(), raw, &mut floats).expect("dequant");
        }
        let read_dequant = t.elapsed().as_secs_f64() / n as f64;

        // 3. the whole device projection: read, quantised upload, matmul.
        let x = backend.to_device(Tensor::from_vec(
            vec![0.01f32; cfg.embedding_dim],
            vec![1, cfg.embedding_dim],
        ));
        let t = std::time::Instant::now();
        for i in 0..n {
            let (ord, _e) = pick(i);
            let pr = &de.layers[ord][0];
            let _ = de
                .linear_part(pr, (i * 37) % glm.n_expert, &x, vec![glm.n_ff_exp, cfg.embedding_dim])
                .expect("linear");
        }
        backend.synchronize();
        let full = t.elapsed().as_secs_f64() / n as f64;

        let gbs = per as f64 / read / 1e9;
        println!("per projection:");
        println!("  raw read          {:8.3} ms   ({gbs:.2} GB/s)", read * 1e3);
        println!("  + dequantise      {:8.3} ms", read_dequant * 1e3);
        println!("  + upload + matmul {:8.3} ms", full * 1e3);
        println!("extrapolated to one token ({projections_per_token} projections):");
        println!("  read alone        {:8.2} s", read * projections_per_token as f64);
        println!("  whole expert path {:8.2} s", full * projections_per_token as f64);
        println!(
            "  (a token was measured at ~2.7 s end to end, so the expert path is \
             {:.0}% of it)",
            100.0 * full * projections_per_token as f64 / 2.7
        );
        assert!(read > 0.0 && full > 0.0);
    }

    /// What one expert record costs on the CPU today, at the released shapes.
    ///
    /// A routed expert is gate and up of `[n_ff_exp, n_embd]` and down of
    /// `[n_embd, n_ff_exp]`, all Q4_K: 4 718 592 bytes each, 8 388 608 weights
    /// each. The CPU tier of the expert hierarchy only pays off if this beats a
    /// PCIe upload of the same record, which the DeepSeek notes measured at
    /// 1.3 ms over x4 and 2.6 ms over x2.
    ///
    /// Synthetic bytes: Q4_K dequantisation is data-independent, and the
    /// arithmetic and the memory traffic are what is being timed.
    #[test]
    #[ignore = "measures the host expert path"]
    fn measure_host_record_cost() {
        let (n_embd, n_ff) = (4096usize, 2048usize);
        let per = n_embd * n_ff / 256 * 144;
        let raw: Vec<u8> = (0..per).map(|i| (i * 31 % 251) as u8).collect();
        let x: Vec<f32> = (0..n_embd).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let mut w = vec![0.0f32; n_embd * n_ff];
        let mut h = vec![0.0f32; n_ff];
        let mut out = vec![0.0f32; n_embd];

        // one pass to fault the buffers in
        q4_k::dequantize(&raw, &mut w);
        matvec(&w, &x, &mut h).expect("warm");
        std::hint::black_box(&h);

        let n = 5usize;
        let (mut deq, mut mv) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let t = std::time::Instant::now();
            for _ in 0..3 {
                q4_k::dequantize(&raw, &mut w);
            }
            deq += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            matvec(&w, &x, &mut h).expect("gate");
            std::hint::black_box(&h);
            matvec(&w, &x, &mut h).expect("up");
            std::hint::black_box(&h);
            matvec(&w, &h, &mut out).expect("down");
            std::hint::black_box(&out);
            mv += t.elapsed().as_secs_f64();
        }
        let (deq, mv) = (deq / n as f64, mv / n as f64);
        let rec = 3 * per;
        println!();
        println!("one expert record, {:.2} MB, 25.2 M weights, single thread:", rec as f64 / 1e6);
        println!("  dequantise      {:8.2} ms   ({:.1} GB/s of Q4_K in)", deq * 1e3, rec as f64 / deq / 1e9);
        println!("  three matvecs   {:8.2} ms", mv * 1e3);
        println!("  total           {:8.2} ms", (deq + mv) * 1e3);
        println!("  over 32 threads {:8.2} ms   (perfect scaling, which it will not get)", (deq + mv) * 1e3 / 32.0);
        println!();
        println!("The f32 detour is most of it: dequantising writes {:.0} MB and the", 3.0 * (n_embd * n_ff) as f64 * 4.0 / 1e6);
        println!("matvecs read it back. A fused Q4_K dot would touch the {:.1} MB once.", rec as f64 / 1e6);
        println!("to beat: a PCIe upload of the same record, 1.3 ms over x4");
        println!("budget: 20 tok/s over 42 layers is 1.19 ms a layer, for every");
        println!("routed expert of that layer not already resident in VRAM");
        assert!(deq > 0.0 && mv > 0.0);
    }

    /// Why the trunk costs what it does: [`Mat::apply`] round-trips the host on
    /// every matrix.
    ///
    /// `to_device` uploads x, `linear` launches, `to_host` synchronises. If a
    /// 1x64 matvec costs about what a 4096x4096 one costs, the trunk is not
    /// doing arithmetic, it is paying driver latency — the same 35-50 us per
    /// synchronisation on Windows that `dsv41-cuda/src/handoff.rs` was written
    /// to avoid.
    ///
    /// The trunk runs about 710 of these per token (9 per KDA layer, 8 per MLA
    /// layer, 2 hyper-connection mixes, the router and shared expert, the head).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "measures device round-trip latency"]
    fn measure_mat_apply_latency() {
        let Some(backend) = cuda_backend() else { return };
        println!();
        println!("     shape        per apply     implied GB/s of weights");
        for &(o, k) in &[(1usize, 64usize), (64, 64), (512, 512), (2048, 4096), (4096, 4096)] {
            let raw: Vec<f32> = (0..o * k).map(|i| ((i % 19) as f32 - 9.0) / 9.0).collect();
            let w = Weight::Dense(backend.to_device(Tensor::from_vec(raw, vec![o, k])));
            let m = Mat::Device { w: &w, backend: &*backend };
            let x: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
            let mut out = vec![0.0f32; o];
            m.apply(&x, &mut out).expect("warm");

            let n = 200usize;
            let t = std::time::Instant::now();
            for _ in 0..n {
                m.apply(&x, &mut out).expect("apply");
            }
            let per = t.elapsed().as_secs_f64() / n as f64;
            println!(
                "  {:5} x {:5}   {:7.1} us    {:8.1}",
                o,
                k,
                per * 1e6,
                (o * k * 4) as f64 / per / 1e9
            );
        }
        // Split an apply into its three parts, at the hyper-connection shape:
        // hc_attn_fn is [24, 16384] and its input is the whole 4 x 4096 stream.
        println!();
        println!("one apply, taken apart (x of 16384 floats, the hc_mixes shape):");
        let big: Vec<f32> = (0..16384).map(|i| (i % 13) as f32 / 13.0).collect();
        let small = Tensor::from_vec(vec![1.0f32; 24], vec![1, 24]);
        let sd = backend.to_device(small);
        let n = 300usize;

        let t = std::time::Instant::now();
        for _ in 0..n {
            let d = backend.to_device(Tensor::from_vec(big.clone(), vec![1, 16384]));
            std::hint::black_box(&d);
        }
        let up = t.elapsed().as_secs_f64() / n as f64;

        let t = std::time::Instant::now();
        for _ in 0..n {
            let h = backend.to_host(sd.clone());
            std::hint::black_box(&h);
        }
        let down = t.elapsed().as_secs_f64() / n as f64;

        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(big.clone());
        }
        let clone = t.elapsed().as_secs_f64() / n as f64;

        let hw: Vec<f32> = (0..24 * 16384).map(|i| ((i % 17) as f32 - 8.0) / 80.0).collect();
        let hwt = Weight::Dense(backend.to_device(Tensor::from_vec(hw, vec![24, 16384])));
        let hm = Mat::Device { w: &hwt, backend: &*backend };
        let mut hout = vec![0.0f32; 24];
        hm.apply(&big, &mut hout).expect("warm");
        let t = std::time::Instant::now();
        for _ in 0..n {
            hm.apply(&big, &mut hout).expect("hc apply");
        }
        let full = t.elapsed().as_secs_f64() / n as f64;

        println!("  host Vec clone of x   {:7.1} us", clone * 1e6);
        println!("  to_device(x)          {:7.1} us", up * 1e6);
        println!("  to_host(24 floats)    {:7.1} us", down * 1e6);
        println!("  whole apply           {:7.1} us", full * 1e6);
        println!("  unaccounted (matmul)  {:7.1} us", (full - up - down) * 1e6);
        println!("  x 90 calls a token    {:7.2} ms", full * 90.0 * 1e3);

        println!();
        println!("A trunk token is about 710 of these. Multiply the small-matrix");
        println!("number by 710 to see the floor that latency alone imposes.");
    }

    /// The trunk floor: a token with the routed experts stubbed out to zeros.
    ///
    /// 20 tok/s is 50 ms a token. Whatever attention, the shared expert, the
    /// hyper-connections and the head cost comes off that before a single
    /// routed expert is fetched, so this bounds everything else.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk; measures the trunk"]
    fn measure_trunk_floor() {
        struct Zeros;
        impl ExpertFfn for Zeros {
            fn apply(
                &self,
                _ord: usize,
                _e: usize,
                _x: &[f32],
                _limit: f32,
                out: &mut [f32],
            ) -> Result<()> {
                out.fill(0.0);
                Ok(())
            }
        }

        let Some(backend) = cuda_backend() else { return };
        let m = DeviceModel::open_with_cache(RELEASED, 512, backend.clone(), 8 << 30).expect("load");
        let sh = m.shape().clone();

        // Where the trunk actually is. A successful load does not mean the
        // matrices reached the card.
        let (dev, host, stragglers) = m.residency();
        println!();
        println!("trunk residency: {:.2} GB on the backend, {:.2} GB left on the host", dev as f64 / 1e9, host as f64 / 1e9);
        if stragglers.is_empty() {
            println!("  every trunk matrix is on the backend");
        } else {
            println!("  {} host-resident, first few:", stragglers.len());
            for n in stragglers.iter().take(6) {
                println!("    {n}");
            }
        }

        let zeros = Zeros;
        let w = m.view_with_experts(&zeros);
        let mut st = forward::State::new_on(&sh, backend.clone()).expect("state");
        let _ = forward::forward_token(&sh, &w, &mut st, 154822).expect("warmup");

        let n = 8usize;
        let t = std::time::Instant::now();
        for i in 0..n {
            let _ = forward::forward_token(&sh, &w, &mut st, 1000 + i as u32).expect("trunk");
        }
        let trunk = t.elapsed().as_secs_f64() / n as f64;
        let n_moe = sh.n_layer - sh.n_dense_lead;

        // Where it went. One more token, profiled.
        forward::prof::reset();
        let _ = forward::forward_token(&sh, &w, &mut st, 2000).expect("profiled");
        println!();
        println!("one trunk token by phase (routed experts stubbed to zeros):");
        for (name, c) in forward::prof::all() {
            let v = forward::prof::ms(&c);
            println!("  {:32} {:7.1} ms   {:4.1}%", name, v, 100.0 * v / forward::prof::total_ms());
        }
        println!("  {:32} {:7.1} ms", "accounted for", forward::prof::total_ms());
        println!();
        println!("inside the two attention kinds:");
        for (name, c) in forward::prof::inner() {
            println!("  {:32} {:7.1} ms", name, forward::prof::ms(&c));
        }
        let n_kda = sh.layer_kinds[..sh.n_layer].iter().filter(|k| matches!(k, crate::glm5next::LayerKind::Kda)).count();
        let n_mla = sh.n_layer - n_kda;
        println!();
        println!("shapes: n_embd {}, n_head {}, kda_head_dim {}, d_inner {}", sh.n_embd, sh.n_head, sh.kda_head_dim, sh.d_inner());
        println!("        kv_lora {}, qk_head {}, v_head {}, q_lora {}", sh.kv_lora, sh.qk_head, sh.v_head, sh.q_lora);
        println!("        {} KDA layers, {} MLA layers", n_kda, n_mla);
        println!("host f32 weights read per token, per MLA layer:");
        println!("        k_b {:.1} MB, v_b {:.1} MB", (sh.n_head * sh.kv_lora * sh.qk_head * 4) as f64 / 1e6, (sh.n_head * sh.v_head * sh.kv_lora * 4) as f64 / 1e6);
        println!("KDA recurrent state per layer: {:.1} MB", (sh.n_head * sh.kda_head_dim * sh.kda_head_dim * 4) as f64 / 1e6);

        println!();
        println!("trunk only        {:7.1} ms a token   ({:.1} tok/s if experts were free)", trunk * 1e3, 1.0 / trunk);
        println!("budget, 20 tok/s  {:7.1} ms a token", 50.0);
        let left = 0.050 - trunk;
        if left > 0.0 {
            println!("left for experts  {:7.1} ms, over {} MoE layers = {:.2} ms a layer", left * 1e3, n_moe, left * 1e3 / n_moe as f64);
        } else {
            println!("left for experts  none: the trunk alone is over budget by {:.1} ms", -left * 1e3);
        }
        assert!(trunk > 0.0);
    }

    /// Spreading the expert tier over two cards must not change the answer.
    ///
    /// This is the correctness gate on multi-GPU. Half the MoE layers resolve
    /// their experts from card 1 and run the FFN there, so a mistake in the
    /// card/cache pairing -- a kernel launched on one card reading another card's
    /// slot, or a `card_of` off by one -- shows up as garbage logits rather than
    /// as an error. Compared against the same model on one card: both are CUDA,
    /// the arithmetic is identical, so this should be exact.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk"]
    fn two_cards_agree_with_one() {
        let Some(backend) = cuda_backend() else { return };

        // Two tokens in, so the KV cache and the KDA state are both non-trivial.
        let prompt = [154822u32, 6172, 1043];

        let one = DeviceModel::open_with_cache(RELEASED, 512, backend.clone(), 32 << 30)
            .expect("one-card load");
        let sh = one.shape().clone();
        let w1 = one.view();
        let mut s1 = forward::State::new_on(&sh, backend.clone()).expect("state");
        let a = forward::forward_prompt(&sh, &w1, &mut s1, &prompt).expect("one card");
        drop(w1);
        drop(one);

        let two = match DeviceModel::open_tiered(RELEASED, 512, 2, 32 << 30, 8 << 30) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("no second card ({e}); skipping");
                return;
            }
        };
        let (card_of, n_cards) = two.experts().card_layout();
        if n_cards < 2 {
            eprintln!("only {n_cards} card visible; skipping");
            return;
        }
        // The split has to actually split, or this test proves nothing.
        assert!(
            card_of.iter().any(|&c| c == 0) && card_of.iter().any(|&c| c == 1),
            "both cards must own MoE layers, got {card_of:?}"
        );
        let w2 = two.view();
        let mut s2 = forward::State::new_on(&sh, backend.clone()).expect("state");
        let b = forward::forward_prompt(&sh, &w2, &mut s2, &prompt).expect("two cards");

        assert_eq!(a.len(), b.len());
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let mut worst = 0.0f32;
        for (p, q) in a.iter().zip(b.iter()) {
            worst = worst.max((p - q).abs());
        }
        println!(
            "max |one card - two cards| over {} logits: {:.5} (scale {:.3})",
            a.len(),
            worst,
            scale
        );
        assert!(
            worst <= 1e-4 * scale,
            "two-card logits diverge: {worst} at scale {scale}"
        );
        assert!(a.iter().any(|&v| v.abs() > 1e-3), "logits are all zero");
    }

    /// The tier stack against a working set that does not fit one card.
    ///
    /// [`measure_tiered_throughput`] replays a short prefix, so a few hundred
    /// distinct experts serve the whole run and one card's cache holds all of
    /// them -- which makes the second card, the RAM tier and the SSD look free.
    /// They are not; they are idle. DeepSeek measured its comparable 62.7% hit
    /// rate over a 247-token document for the same reason.
    ///
    /// So: prefill a varied prompt first, which routes over a real spread of
    /// experts and fills the tiers the way a prompt would, then time decode. The
    /// numbers that matter are the per-card hit rates and how much came off the
    /// SSD, not just the token time.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk; prefills a document first"]
    fn measure_tiered_on_a_document() {
        // 0 = size the RAM tier from free host memory.
        let ram_gb: usize = std::env::var("GLM5_RAM_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let n_prompt: usize = std::env::var("GLM5_PROMPT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(192);
        let n_gpus: usize = std::env::var("GLM5_GPUS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);

        let m = match DeviceModel::open_tiered(RELEASED, 512, n_gpus, ram_gb << 30, 0) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("tiered open failed ({e}); skipping");
                return;
            }
        };
        let sh = m.shape().clone();
        let rec = m.experts().record_bytes();
        let slots: Vec<usize> = m.experts().vram_budgets().iter().map(|b| b / rec).collect();
        let total: usize = slots.iter().sum();
        let ram = m.experts().shared().cache_budget_bytes();
        println!();
        println!(
            "{} card(s), VRAM slots {:?} = {} total ({:.0}% of 12096 records)",
            slots.len(),
            slots,
            total,
            100.0 * total as f64 / 12096.0
        );
        println!(
            "RAM tier {:.0} GB = {} records ({:.0}% of 12096); experts are 182.4 GB on the SSD",
            ram as f64 / 1e9,
            ram / rec,
            100.0 * (ram / rec) as f64 / 12096.0
        );

        // A prompt that actually varies: a cheap LCG over the vocabulary, so the
        // router sees genuinely different inputs rather than one token repeated.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let prompt: Vec<u32> = (0..n_prompt)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) % sh.n_vocab as u64) as u32
            })
            .collect();

        let w = m.view();
        let mut st = forward::State::new_on(&sh, m.backend()).expect("state");
        let t = std::time::Instant::now();
        let mut logits = forward::forward_prompt(&sh, &w, &mut st, &prompt).expect("prefill");
        let prefill = t.elapsed().as_secs_f64();
        println!(
            "prefill {} tokens in {:.1}s ({:.2} tok/s, one at a time)",
            n_prompt,
            prefill,
            n_prompt as f64 / prefill
        );

        let report = |label: &str| {
            for (i, s) in m.experts().shared().shard_stats().iter().enumerate() {
                let look = s.hits + s.misses;
                println!(
                    "  {label} card {i}: {:.1}% VRAM hits ({} / {}), {:.1} GB uploaded, {} evictions",
                    100.0 * s.hits as f64 / look.max(1) as f64,
                    s.hits,
                    look,
                    s.h2d_bytes as f64 / 1e9,
                    s.evictions
                );
            }
            let r = m.experts().shared().cache_stats();
            println!(
                "  {label} RAM: {:.1}% hits ({} / {}), {:.1} GB read from the SSD",
                100.0 * r.hits as f64 / (r.hits + r.misses).max(1) as f64,
                r.hits,
                r.hits + r.misses,
                r.bytes_read as f64 / 1e9
            );
        };
        report("after prefill,");

        let next = |lg: &Vec<f32>| -> u32 {
            let mut best = f32::NEG_INFINITY;
            let mut bi = 0usize;
            for (i, &x) in lg.iter().enumerate() {
                if x > best {
                    best = x;
                    bi = i;
                }
            }
            bi as u32
        };
        let n = 16usize;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let tk = next(&logits);
            logits = forward::forward_token(&sh, &w, &mut st, tk).expect("decode");
        }
        let dec = t.elapsed().as_secs_f64() / n as f64;
        println!();
        report("after decode, ");

        // Second pass over the same document. The tiers now hold the working set,
        // so this is the steady state -- the number that says what the hierarchy
        // is worth once it is warm, as against the cost of filling it. DeepSeek
        // reports 2-6 tok/s for a fresh process and 29 for its third answer for
        // exactly this reason.
        let before = m.experts().shared().cache_stats();
        let mut st2 = forward::State::new_on(&sh, m.backend()).expect("state");
        let t = std::time::Instant::now();
        let mut lg2 = forward::forward_prompt(&sh, &w, &mut st2, &prompt).expect("reprefill");
        let prefill2 = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        for _ in 0..n {
            let tk = next(&lg2);
            lg2 = forward::forward_token(&sh, &w, &mut st2, tk).expect("decode warm");
        }
        let dec2 = t.elapsed().as_secs_f64() / n as f64;
        let after = m.experts().shared().cache_stats();
        let (h, ms) = (after.hits - before.hits, after.misses - before.misses);
        println!();
        println!(
            "second pass: RAM {:.1}% hits ({h} / {}), {:.1} GB more off the SSD",
            100.0 * h as f64 / (h + ms).max(1) as f64,
            h + ms,
            (after.bytes_read - before.bytes_read) as f64 / 1e9
        );
        report("warm,          ");

        println!();
        println!("prefill  cold {:.1}s -> warm {:.1}s for {} tokens", prefill, prefill2, n_prompt);
        println!("decode   cold {:.3} s/token ({:.2} tok/s)", dec, 1.0 / dec);
        println!("decode   warm {:.3} s/token ({:.2} tok/s)", dec2, 1.0 / dec2);
        println!("target is 20 tok/s = 0.050 s/token");
        println!("the trunk is ~60 ms of that; warm experts are ~{:.0} ms", (dec2 - 0.0604).max(0.0) * 1e3);
        assert!(dec > 0.0 && dec2 > 0.0);
    }

    /// The whole hierarchy: both cards' VRAM, RAM behind them, the GGUF behind
    /// that.
    ///
    /// ```text
    ///   SSD (192.97 GB of GGUF)
    ///     -> RAM cache, LFRU with leases        GLM5_RAM_GB, default 128
    ///        -> VRAM cache per card, LFRU       GLM5_VRAM_GB each, default 26
    ///           -> the matvec, zero H2D on a hit
    /// ```
    ///
    /// Experts are 182.44 GB of the model and a token routes 336 of the 12 096
    /// records, 5.07 GB. At 20 tok/s that is 101 GB/s, which no SSD and no single
    /// DDR5 channel pair can serve -- so the question the tiers answer is not how
    /// fast bytes move but how few of them have to.
    ///
    /// Two 5090s hold 32 GB each. The trunk is 5.97 GB on card 0, so ~26 GB a
    /// card is ~3 250 records, 27% of the model, against ~1 500 on one card.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk; measures the full tier stack"]
    fn measure_tiered_throughput() {
        let ram_gb: usize = std::env::var("GLM5_RAM_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let vram_gb: usize = std::env::var("GLM5_VRAM_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(26);
        let n_gpus: usize = std::env::var("GLM5_GPUS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);

        let t0 = std::time::Instant::now();
        let m = match DeviceModel::open_tiered(
            RELEASED,
            512,
            n_gpus,
            ram_gb << 30,
            vram_gb << 30,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("tiered open failed ({e}); skipping");
                return;
            }
        };
        let (card_of, n_cards) = m.experts().card_layout();
        let mut per_card = vec![0usize; n_cards];
        for &c in card_of {
            per_card[c] += 1;
        }
        println!();
        println!(
            "load {:.1}s: {} card(s), {} GB VRAM each, {} GB RAM, record {:.2} MB",
            t0.elapsed().as_secs_f64(),
            n_cards,
            vram_gb,
            ram_gb,
            m.experts().record_bytes() as f64 / 1e6
        );
        println!("  MoE layers per card: {per_card:?}");
        let (dev, host, _) = m.residency();
        println!(
            "  trunk: {:.2} GB on card 0, {:.2} GB host",
            dev as f64 / 1e9,
            host as f64 / 1e9
        );
        let rec = m.experts().record_bytes();
        let slots: Vec<usize> = m.experts().vram_budgets().iter().map(|b| b / rec).collect();
        let total: usize = slots.iter().sum();
        println!(
            "  VRAM budgets: {:?} GB -> slots {:?}, {} total = {:.0}% of the 12096 records",
            m.experts()
                .vram_budgets()
                .iter()
                .map(|b| b >> 30)
                .collect::<Vec<_>>(),
            slots,
            total,
            100.0 * total as f64 / 12096.0
        );

        let sh = m.shape().clone();
        let w = m.view();
        let mut st = forward::State::new_on(&sh, m.backend()).expect("state");
        let mut logits = forward::forward_token(&sh, &w, &mut st, 154822).expect("first");
        let next = |lg: &Vec<f32>| -> u32 {
            let mut best = f32::NEG_INFINITY;
            let mut bi = 0usize;
            for (i, &x) in lg.iter().enumerate() {
                if x > best {
                    best = x;
                    bi = i;
                }
            }
            bi as u32
        };

        let n = 8usize;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let tk = next(&logits);
            logits = forward::forward_token(&sh, &w, &mut st, tk).expect("cold");
        }
        let cold = t.elapsed().as_secs_f64() / n as f64;

        let mut st2 = forward::State::new_on(&sh, m.backend()).expect("state");
        let mut lg2 = forward::forward_token(&sh, &w, &mut st2, 154822).expect("first");
        let t = std::time::Instant::now();
        for _ in 0..n {
            let tk = next(&lg2);
            lg2 = forward::forward_token(&sh, &w, &mut st2, tk).expect("warm");
        }
        let warm = t.elapsed().as_secs_f64() / n as f64;

        println!();
        for (i, st) in m.experts().shared().shard_stats().iter().enumerate() {
            let look = st.hits + st.misses;
            println!(
                "  card {i}: {} hits / {} lookups = {:.1}%, {:.1} GB resident, {:.1} GB uploaded, {} evictions",
                st.hits,
                look,
                100.0 * st.hits as f64 / look.max(1) as f64,
                st.bytes_hit as f64 / 1e9,
                st.h2d_bytes as f64 / 1e9,
                st.evictions
            );
        }
        let rc = m.experts().shared().cache_stats();
        println!(
            "  RAM tier: {:.1}% hits ({} / {})",
            100.0 * rc.hits as f64 / (rc.hits + rc.misses).max(1) as f64,
            rc.hits,
            rc.hits + rc.misses
        );
        println!();
        println!("cold  {:.3} s/token   ({:.2} tok/s)", cold, 1.0 / cold);
        println!("warm  {:.3} s/token   ({:.2} tok/s)", warm, 1.0 / warm);
        println!("target is 20 tok/s = 0.050 s/token");
        // `measure_trunk_floor`, with the routed experts stubbed to zeros.
        let trunk = 0.0604;
        println!(
            "the trunk alone is {:.0} ms of that warm token; experts are {:.0} ms",
            trunk * 1e3,
            (warm - trunk).max(0.0) * 1e3
        );
        assert!(cold > 0.0 && warm > 0.0);
    }

    /// Decode throughput, cold cache then warm.
    ///
    /// The profiler showed a token is essentially its 1008 expert projections:
    /// 4.76 GB read at 2.74 GB/s, about 2.96 s. Every one of those reads used to
    /// happen on every token. With `Ecache` in front, a repeat is a hit and costs
    /// nothing on disk, so the warm number is the one that matters.
    ///
    /// 20 tok/s needs ~50 ms a token, which needs the expert bytes to come from
    /// RAM or VRAM rather than NVMe -- 4.76 GB a token is 94 GB/s at that rate,
    /// and no SSD does that.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the released model on disk; measures throughput"]
    fn measure_decode_throughput() {
        let Some(backend) = cuda_backend() else { return };

        // A large RAM cache: one record is gate+up+down for one expert, and the
        // model has 42 * 288 = 12096 of them.
        let ram: usize = 96 << 30;
        let t0 = std::time::Instant::now();
        let m = DeviceModel::open_with_cache(RELEASED, 512, backend.clone(), ram).expect("load");

        // VRAM tier. The trunk measures 5.97 GB on the card (see `residency`), so
        // a 32 GB 5090 has ~25 GB spare -- about 1530 slots at the 16.32 MB a
        // record needs, 13% of the 12096. Set GLM5_VRAM_GB to sweep it; the point
        // of the sweep is to find out whether the record uploads are what the
        // warm token is actually spending its time on.
        let vram_gb: usize = std::env::var("GLM5_VRAM_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        match ggml_rs_cuda::CudaBackend::new(0) {
            Ok(dev) => {
                let vram: usize = vram_gb << 30;
                match m.experts().shared().enable_device_cache(Arc::new(dev), vram) {
                    Ok(()) => println!("VRAM expert cache: {} GB", vram >> 30),
                    Err(e) => println!("VRAM expert cache unavailable: {e}"),
                }
            }
            Err(e) => println!("second CUDA handle failed: {e}"),
        }
        println!(
            "load {:.1}s, backend {}, record {:.1} MB, cache budget {} GB",
            t0.elapsed().as_secs_f64(),
            m.backend_name(),
            m.experts().record_bytes() as f64 / 1e6,
            ram >> 30
        );

        let sh = m.shape().clone();
        let w = m.view();
        let mut st = forward::State::new_on(&sh, backend.clone()).expect("state");

        let mut logits = forward::forward_token(&sh, &w, &mut st, 154822).expect("first");
        let next = |lg: &Vec<f32>| -> u32 {
            let mut best = f32::NEG_INFINITY;
            let mut bi = 0usize;
            for (i, &x) in lg.iter().enumerate() {
                if x > best {
                    best = x;
                    bi = i;
                }
            }
            bi as u32
        };

        // Cold: the cache is empty, so most dispatches miss and hit the disk.
        let n_cold = 8usize;
        let t = std::time::Instant::now();
        for _ in 0..n_cold {
            let t_ = next(&logits);
            logits = forward::forward_token(&sh, &w, &mut st, t_).expect("cold");
        }
        let cold = t.elapsed().as_secs_f64() / n_cold as f64;

        // Warm: the same prefix again, so the routed experts are the ones already
        // resident. A fresh State replays the positions; the expert cache carries
        // over because it lives in the model, not the state.
        let mut st2 = forward::State::new_on(&sh, backend.clone()).expect("state");
        let mut lg2 = forward::forward_token(&sh, &w, &mut st2, 154822).expect("first");
        let t = std::time::Instant::now();
        for _ in 0..n_cold {
            let t_ = next(&lg2);
            lg2 = forward::forward_token(&sh, &w, &mut st2, t_).expect("warm");
        }
        let warm = t.elapsed().as_secs_f64() / n_cold as f64;

        println!();
        if let Some(st) = m.experts().shared().device_cache_stats() {
            let n = st.hits + st.misses;
            println!(
                "VRAM tier: {} hits / {} lookups = {:.1}%, {:.1} GB served resident, {:.1} GB uploaded, {} evictions",
                st.hits,
                n,
                100.0 * st.hits as f64 / n.max(1) as f64,
                st.bytes_hit as f64 / 1e9,
                st.h2d_bytes as f64 / 1e9,
                st.evictions
            );
        }
        println!("cold  {:.3} s/token   ({:.2} tok/s)", cold, 1.0 / cold);
        println!("warm  {:.3} s/token   ({:.2} tok/s)", warm, 1.0 / warm);
        println!("speedup from the cache: {:.2}x", cold / warm);
        println!("target is 20 tok/s = 0.050 s/token");
        assert!(cold > 0.0 && warm > 0.0);
    }

    /// The released model, with its matrices on the backend. On a CUDA build this
    /// is the GPU path; on a CPU build it exercises the same plumbing.
    #[test]
    #[ignore = "needs the released model on disk"]
    fn released_model_runs_on_the_backend() {
        let backend = ggml_rs::default_backend();
        let t0 = std::time::Instant::now();
        let m = DeviceModel::open(RELEASED, 512, backend).expect("load");
        println!(
            "device load in {:.1}s, backend {}",
            t0.elapsed().as_secs_f64(),
            m.backend_name()
        );

        let sh = m.shape().clone();
        assert_eq!(sh.n_layer, 45);
        let w = m.view();
        let mut st = forward::State::new_on(&sh, m.backend()).expect("state");

        let t1 = std::time::Instant::now();
        let logits = forward::forward_token(&sh, &w, &mut st, 154822).expect("forward");
        println!("one token in {:.1}s", t1.elapsed().as_secs_f64());

        assert_eq!(logits.len(), sh.n_vocab);
        assert!(logits.iter().all(|x| x.is_finite()));
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                bi = i;
            }
        }
        println!("argmax {bi} logit {best:.4}");
        assert!(logits.iter().any(|&v| v != 0.0));
    }

    /// The device path must agree with the host reference on the real model.
    /// This is the equivalence gate for the backend seam.
    #[test]
    #[ignore = "needs the released model on disk; loads it twice"]
    fn device_agrees_with_host_on_the_real_model() {
        use super::super::bridge::HostModel;

        let hm = HostModel::open(RELEASED, 512).expect("host load");
        let hs = hm.shape().clone();
        let hv = hm.view();
        let mut h_state = forward::State::new(&hs).expect("state");
        let h = forward::forward_token(&hs, &hv, &mut h_state, 154822).expect("host forward");
        drop(hv);
        drop(hm);

        let dm = DeviceModel::open(RELEASED, 512, ggml_rs::default_backend()).expect("dev load");
        let ds = dm.shape().clone();
        let dv = dm.view();
        // On the CPU backend this pits `Backend::kda_delta_step`'s host default
        // against `kda::step`, which is the other half of the recurrence check.
        let mut d_state = forward::State::new_on(&ds, dm.backend()).expect("state");
        let d = forward::forward_token(&ds, &dv, &mut d_state, 154822).expect("device forward");

        assert_eq!(h.len(), d.len());
        let mut worst = 0.0f32;
        for (a, b) in h.iter().zip(d.iter()) {
            worst = worst.max((a - b).abs());
        }
        println!("max |host - device| over {} logits: {worst:.6}", h.len());
        // Both paths dequantise the same bytes; the only differences are
        // accumulation order inside the matmul.
        assert!(worst < 0.05, "device and host disagree by {worst}");
    }
}
