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
use super::cpu_experts::{CpuExperts, CpuJob};
use crate::expert_stream::{
    ExpertLayout, GgufExpertStore, LayerStream, ResolvedExpert, StreamShared,
};
use super::forward::{
    self, AttnW, Bat, ExpertFfn, FfnW, HcW, IndexerW, KdaW, LayerW, Mat, MlaW, ModelW, MoeW, Pair,
    Shape,
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

// VENDORED-LOCAL: GLM-5.3-Flash, from `dsv41-cuda/src/model.rs`.
/// VRAM promotions a hybrid decode step makes per layer.
///
/// Some promotion has to happen or the VRAM tier freezes at whatever it held after
/// the first pass and can never take on a new working set. But each one is a PCIe
/// copy of 14-16 MB -- 1.12 ms on this machine's four-lane card -- that also evicts
/// something, and the CPU can compute the same record in about that time without
/// moving it. So: one a layer, to the miss most likely to be wanted again.
pub const PROMOTE_PER_LAYER: usize = 1;

/// How many of a layer's VRAM misses to promote into the cache, per token.
///
/// A promotion is an upload: 14.16 MB over PCIe, on the critical path, and once
/// the cache is full it also evicts. At a steady 87.5% hit rate the promotions are
/// 347 MB a token -- about 20 ms of the routed experts' 47.8 -- and they buy
/// nothing at the margin, because what they admit is what they evict.
///
/// The alternative for a miss is the CPU tier, which measured 0.64 ms a record
/// against 0.83 ms to upload one, and runs while the GPU works on the resident
/// experts rather than ahead of it. So the right number here is an empirical
/// question, and `GLM5_PROMOTE_PER_LAYER` is how it gets asked: 0 leaves the
/// cache as the prewarm left it and sends every miss to the CPU.
fn promote_per_layer() -> usize {
    std::env::var("GLM5_PROMOTE_PER_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PROMOTE_PER_LAYER)
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
    /// The CPU tier, when it is available. `None` means every miss is uploaded.
    cpu: Option<Arc<CpuExperts>>,
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
            cpu: None,
        })
    }

    /// The shared cache state, so a caller can turn on the VRAM tier.
    pub fn shared(&self) -> &Arc<StreamShared> {
        &self.shared
    }

    /// Turn on the CPU tier: a VRAM miss whose record is in RAM is computed here
    /// rather than uploaded, while the GPU runs the layer's resident experts.
    pub fn enable_cpu_tier(&mut self) {
        self.cpu = Some(Arc::new(CpuExperts::new()));
    }

    /// The CPU tier, if enabled.
    pub fn cpu_tier(&self) -> Option<&Arc<CpuExperts>> {
        self.cpu.as_ref()
    }

    /// VRAM held back on each card: room for the activations, the pinned staging
    /// ring, the kernels' workspaces and the driver's own overhead.
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

    /// Read every expert record once, so the RAM tier is as full as its budget
    /// allows before anything is timed.
    ///
    /// This is what "fully loaded" means for a model whose experts are 182 GB: not
    /// that they are all in memory -- the tier is smaller than that -- but that
    /// nothing is left to discover on the drive that the tier had room for. Layers
    /// are read in order and each layer's experts in one batch, so the reads go out
    /// at the queue depth `fetch_many` was built for.
    ///
    /// Returns how many records were requested. Compare with
    /// [`StreamShared::resident_records`] to see how many the tier kept.
    pub fn prewarm_all(&self) -> Result<usize> {
        let n_expert = self.shared.n_experts();
        let mut asked = 0usize;
        for (ord, ls) in self.layers.iter().enumerate() {
            let ids: Vec<u32> = (0..n_expert as u32).collect();
            ls.prewarm(&ids)
                .map_err(|e| LlamaError::Config(format!("device: prewarm layer {ord}: {e}")))?;
            asked += ids.len();
        }
        Ok(asked)
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

    fn gpu_part(
        &self,
        be: &dyn Backend,
        resolved: &[ResolvedExpert],
        experts: &[(u32, f32)],
        x: &[f32],
        limit: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let any = resolved.iter().any(|r| r.gpu().is_some());
        if !any {
            out.fill(0.0);
            return Ok(());
        }
        let grouped: Option<Tensor> = None;

        // Whether the grouped call consumed the `Device` slots, recorded before
        // `grouped` is moved into the accumulator.
        #[allow(unused_variables)]
        let grouped_ran = grouped.is_some();
        let xd = be.to_device(Tensor::from_vec(x.to_vec(), vec![1, self.n_embd]));
        let mut acc = match grouped {
            Some(sum) => sum,
            None => {
                be.to_device(Tensor::from_vec(vec![0.0f32; self.n_embd], vec![1, self.n_embd]))
            }
        };
        for (r, &(_, wt)) in resolved.iter().zip(experts) {
            let Some((pair, down)) = r.gpu() else { continue };
            let h = pair.swiglu_clamped(be, &xd, limit, true);
            let o = down.linear(be, &h);
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

    /// One resolved expert's FFN, input already on `be`: the clamp is fused, so
    /// the gate/up pair never leaves the device.
    fn run_expert(&self, be: &dyn Backend, r: &ResolvedExpert, xd: &Tensor, limit: f32) -> Tensor {
        let (pair, down) = r
            .gpu()
            .expect("run_expert is only used on the non-hybrid path, which never yields Cpu");
        let h = pair.swiglu_clamped(be, xd, limit, true);
        down.linear(be, &h)
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
    // VENDORED-LOCAL: GLM-5.3-Flash. A chunk of tokens, one read an expert.
    /// The routed half of one MoE layer for several tokens at once.
    ///
    /// Per token this layer costs ~1.2 ms and almost none of it is arithmetic: the
    /// promoted records' pinned copies, the PCIe the compute stream waits on, the
    /// CPU tier. Every one of those is paid **per distinct expert**, so the point of
    /// a chunk is that its tokens share them. Eight experts over 64 tokens is 512
    /// resolutions where the union is nearer 200, and each of those 200 reads its
    /// weights once and applies them to every token that chose it -- a small GEMM
    /// rather than one GEMV a token.
    ///
    /// The route weights are applied on the way out, per (expert, token), because
    /// the same expert carries a different weight for each token that picked it.
    fn apply_batch(
        &self,
        ord: usize,
        routes: &[Vec<(u32, f32)>],
        xs: &[f32],
        n_embd: usize,
        limit: f32,
        outs: &mut [f32],
    ) -> Result<()> {
        let n = routes.len();
        if n * n_embd != xs.len() || xs.len() != outs.len() || n_embd != self.n_embd {
            return Err(LlamaError::Config(format!(
                "device: expert batch has {} routes, {} inputs, {} outputs at n_embd \
                 {n_embd} (model {})",
                n,
                xs.len(),
                outs.len(),
                self.n_embd
            )));
        }
        // One token is decode, and the per-token path is better at it: it overlaps
        // the CPU tier with the GPU and runs the whole route through one grouped
        // kernel, neither of which a chunk of one would gain anything from.
        if n == 1 {
            return self.apply_layer(ord, &routes[0], xs, limit, outs);
        }
        let ls = self.layer(ord)?;

        // The union of the chunk's routes, and for each of them the tokens that
        // chose it with the weight each gave it.
        let mut distinct: Vec<u32> = routes.iter().flatten().map(|&(e, _)| e).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let mut users: Vec<Vec<(usize, f32)>> = vec![Vec::new(); distinct.len()];
        for (t, route) in routes.iter().enumerate() {
            for &(e, wt) in route {
                match distinct.binary_search(&e) {
                    Ok(i) => users[i].push((t, wt)),
                    Err(_) => {
                        return Err(LlamaError::Config(format!(
                            "device: expert {e} of layer {ord} vanished from the union"
                        )))
                    }
                }
            }
        }

        // One resolve for the chunk: this is what is being amortised.
        // Promotions are per resolve call, and a chunk makes one call where the
        // per-token path made `n`. Asking for one would fill the VRAM tier `n` times
        // slower than decode does -- measured as caches stuck at 3.3 and 10.5 GB of
        // a 20.4 and 26.7 GB budget, creeping up 32 MiB at a time, with almost every
        // expert going to the CPU and both cards idle. So the budget scales with the
        // chunk: the same promotions a token as before. `resolve_experts_hybrid`
        // takes the top `promote` of the misses, so this is bounded by how many
        // there actually are.
        let t_resolve = std::time::Instant::now();
        let promote = promote_per_layer().saturating_mul(n);
        let resolved = match &self.cpu {
            Some(_) => ls.resolve_experts_hybrid(&distinct, promote),
            None => ls.resolve_experts(&distinct),
        }
        .map_err(|err| LlamaError::Config(format!("device: MoE layer {ord}: {err}")))?;
        crate::glm5next::forward::prof::add(&crate::glm5next::forward::prof::FFN_RESOLVE, t_resolve);

        let t_dispatch = std::time::Instant::now();
        outs.fill(0.0);
        let be = self.card(ord);

        // The GPU half: one expert, one gather, one pass, one scatter.
        let mut gathered: Vec<f32> = Vec::with_capacity(n * n_embd);
        for (i, r) in resolved.iter().enumerate() {
            let toks = &users[i];
            if toks.is_empty() {
                continue;
            }
            let Some((pair, down)) = r.gpu() else { continue };
            gathered.clear();
            for &(t, _) in toks {
                gathered.extend_from_slice(&xs[t * n_embd..(t + 1) * n_embd]);
            }
            let xg = be.to_device(Tensor::from_vec(gathered.clone(), vec![toks.len(), n_embd]));
            // Both of these take a [rows, k] input and give a [rows, m] result:
            // `swiglu_clamped_split` reads the row count off the tensor.
            let h = pair.swiglu_clamped(be, &xg, limit, true);
            let o = down.linear(be, &h);
            let oh = be.to_host(o);
            let d = oh.data();
            if d.len() != toks.len() * n_embd {
                return Err(LlamaError::Config(format!(
                    "device: expert {} of layer {ord} returned {} values for {} tokens",
                    distinct[i],
                    d.len(),
                    toks.len()
                )));
            }
            for (row, &(t, wt)) in toks.iter().enumerate() {
                let src = &d[row * n_embd..(row + 1) * n_embd];
                for (acc, &v) in outs[t * n_embd..(t + 1) * n_embd].iter_mut().zip(src) {
                    *acc += wt * v;
                }
            }
        }

        // The CPU half, grouped by token because `CpuExperts::run` takes one `x` for
        // the jobs it is given. The leases are already held, so nothing is re-read.
        if let Some(cpu) = &self.cpu {
            let layout = self.shared.layout();
            for t in 0..n {
                let jobs: Vec<CpuJob> = resolved
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let lease = r.cpu_lease()?;
                        let wt = users[i].iter().find(|&&(tt, _)| tt == t).map(|&(_, w)| w)?;
                        Some(CpuJob { lease: lease.clone(), weight: wt, slot: i })
                    })
                    .collect();
                if jobs.is_empty() {
                    continue;
                }
                let done = cpu.run(&jobs, layout, ord, &xs[t * n_embd..(t + 1) * n_embd], limit, n_embd)?;
                for (_, wt, o) in done {
                    for (acc, &v) in outs[t * n_embd..(t + 1) * n_embd].iter_mut().zip(o.iter()) {
                        *acc += wt * v;
                    }
                }
            }
        }
        crate::glm5next::forward::prof::add(
            &crate::glm5next::forward::prof::FFN_DISPATCH,
            t_dispatch,
        );
        Ok(())
    }

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
        // and whatever is going to move is staged in one batch before the first
        // matvec.
        let ids: Vec<u32> = experts.iter().map(|&(e, _)| e).collect();
        let t_resolve = std::time::Instant::now();
        let resolved = match &self.cpu {
            Some(_) => ls.resolve_experts_hybrid(&ids, promote_per_layer()),
            None => ls.resolve_experts(&ids),
        }
        .map_err(|err| LlamaError::Config(format!("device: MoE layer {ord}: {err}")))?;
        crate::glm5next::forward::prof::add(&crate::glm5next::forward::prof::FFN_RESOLVE, t_resolve);
        let t_dispatch = std::time::Instant::now();

        // Whatever the VRAM tier missed and the CPU is taking.
        let cpu_jobs: Vec<CpuJob> = resolved
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r.cpu_lease().map(|l| CpuJob {
                    lease: l.clone(),
                    weight: experts[i].1,
                    slot: i,
                })
            })
            .collect();

        let be = self.card(ord);
        let mut host_sum = vec![0.0f32; self.n_embd];

        // Start the CPU work FIRST, so it runs underneath the GPU launches rather
        // than after them -- the same ordering `dsv41-cuda` uses, and the only
        // reason the tier is free rather than merely cheap.
        let cpu_result: Result<Vec<(usize, f32, Vec<f32>)>> = if cpu_jobs.is_empty() {
            Ok(Vec::new())
        } else {
            let cpu = self.cpu.as_ref().expect("cpu jobs imply a cpu tier");
            let layout = self.shared.layout();
            let (tx, rx) = std::sync::mpsc::channel();
            let n_embd = self.n_embd;
            let mut gpu_err = None;

            rayon::scope(|s| {
                s.spawn(|_| {
                    let _ = tx.send(cpu.run(&cpu_jobs, layout, ord, x, limit, n_embd));
                });
                // ... while this thread drives the GPU.
                if let Err(e) = self.gpu_part(be, &resolved, experts, x, limit, &mut host_sum) {
                    gpu_err = Some(e);
                }
            });
            if let Some(e) = gpu_err {
                return Err(e);
            }
            rx.recv().unwrap_or_else(|_| {
                Err(LlamaError::Config("device: the CPU expert worker vanished".into()))
            })
        };

        // With nothing for the CPU, the GPU part still has to run.
        if cpu_jobs.is_empty() {
            self.gpu_part(be, &resolved, experts, x, limit, &mut host_sum)?;
        }

        for (_, wt, o) in cpu_result? {
            for (t, &v) in host_sum.iter_mut().zip(o.iter()) {
                *t += wt * v;
            }
        }
        out.copy_from_slice(&host_sum);
        crate::glm5next::forward::prof::add(
            &crate::glm5next::forward::prof::FFN_DISPATCH,
            t_dispatch,
        );
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

    pub fn from_gguf(
        g: &GgufFile,
        max_len: usize,
        backend: Arc<dyn Backend>,
        cache_budget_bytes: usize,
    ) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(g)?;
        let glm = Glm5NextConfig::from_gguf(g, &cfg)?;
        let idx = TensorIndex::new(g);

        // Pairs to stack into one matrix. Collected here and built after the
        // per-tensor closures go out of scope, because `Weight::stack_axis0` needs
        // both halves still host-resident and the closures hold `w` mutably.
        //
        // Each pair shares an input and a dtype, so the GGUF rows concatenate and
        // the pair becomes one matvec -- 110 fewer `Mat::apply` calls a token, each
        // of which carries a ~32 us synchronisation whatever its size.
        let mut fuse: Vec<(String, String, String)> = Vec::new();

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
                    // attn_q + attn_k are both Q4_K; attn_v is Q6_K, so it stays
                    // on its own. ssm_f_a + ssm_g_a are both Q8_0 over the layer
                    // input.
                    fuse.push((
                        format!("blk.{il}.attn_qk"),
                        format!("blk.{il}.attn_q.weight"),
                        format!("blk.{il}.attn_k.weight"),
                    ));
                    fuse.push((
                        format!("blk.{il}.ssm_fga"),
                        format!("blk.{il}.ssm_f_a.weight"),
                        format!("blk.{il}.ssm_g_a.weight"),
                    ));
                    for s in [
                        "attn_v.weight",
                        "attn_output.weight",
                        "ssm_f_b.weight",
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
                // The shared expert's gate and up are both Q4_K over the layer
                // input, so they stack into one matvec.
                fuse.push((
                    format!("blk.{il}.ffn_shexp_gate_up"),
                    format!("blk.{il}.ffn_gate_shexp.weight"),
                    format!("blk.{il}.ffn_up_shexp.weight"),
                ));
                for s in ["ffn_gate_inp.weight", "ffn_down_shexp.weight"] {
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

        // Now the closures are out of scope, so `w` is free: stack each collected
        // pair while both halves are still host-resident, then send the result to
        // the card. `stack_axis0` byte-concatenates same-dtype quantised rows, so
        // the fused matrix holds exactly the bytes the two did.
        for (dst, a, b) in fuse {
            let wa = idx.take_weight(&a, &[])?;
            let wb = idx.take_weight(&b, &[])?;
            let fused = Weight::stack_axis0(vec![wa, wb]);
            w.insert(dst, fused.try_to_device(&*backend, MARGIN));
        }

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
                Weight::Packed(w) => (w.nbytes() as u64, true),
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
                    qk: Pair::Fused(m("attn_qk")),
                    v: m("attn_v.weight"),
                    conv_q: b("ssm_conv1d_q.weight"),
                    conv_k: b("ssm_conv1d_k.weight"),
                    conv_v: b("ssm_conv1d_v.weight"),
                    fga: Pair::Fused(m("ssm_fga")),
                    f_b: m("ssm_f_b.weight"),
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
                    sh_gate_up: Pair::Fused(m("ffn_shexp_gate_up")),
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
mod tests;
