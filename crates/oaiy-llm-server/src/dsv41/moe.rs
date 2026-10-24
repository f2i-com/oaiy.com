//! MoE: router, routed experts through the expert cache, shared expert
//! (reference `Gate`, `Expert`, `MoE`).
//!
//! Routing: `sqrt(softplus(x . W_gate))`; the correction bias picks the top
//! 6 but the weights come from the unbiased scores, normalized and scaled by
//! `route_scale`. Outputs accumulate in f32 in ascending expert id order
//! (the reference's loop order), then the shared expert, then one bf16 round.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use oaiy_engine::ecache::{Ecache, HostLease};
use oaiy_engine::store::WeightStore;
use oaiy_engine::Result;

use crate::dsv41::config::Config;
use crate::dsv41::formats::to_bf16;
use crate::dsv41::linear::{load_vec, Out, Weight};
use crate::dsv41::ops::{silu, softplus};
use crate::dsv41::safetensors::StIndex;

/// Where routed expert records come from: a store behind a bounded cache.
pub struct Experts {
    pub store: Arc<dyn WeightStore>,
    pub cache: Ecache,
    /// Workers for a decode step's routed experts, all of a token's at once (the same results as one at a time);
    /// none: one after another on the caller's thread.
    pub pool: Option<crate::dsv41::cpu_experts::CpuExperts>,
    /// Where a prompt's busy experts (those [`GPU_MIN_ROWS`] or more of its tokens chose) have their matmuls made, a
    /// group of [`GPU_GROUP`] a call: a GPU's (`crate::dsv41::expert::ExpertsKernel`). None: all on the CPU.
    pub gpu: Option<Arc<dyn crate::dsv41::expert::ExpertsKernel>>,
    /// The row kernel a prompt's experts on the CPU run through (`pool`'s is its own): the portable one unless a
    /// caller chose this CPU's ([`crate::dsv41::model::Model::set_expert_row_kernel`]).
    pub kernel: crate::dsv41::cpu_experts::RowKernel,
    /// The kernel a prompt's experts on the CPU run through, all of an expert's tokens at once: the portable one
    /// unless a caller chose this CPU's ([`crate::dsv41::model::Model::set_expert_tokens_kernel`]).
    pub tokens: crate::dsv41::cpu_experts::TokensKernel,
    /// Every routed expert's uses of late: what the tiers are ordered by.
    pub uses: Uses,
    /// What a record holds: the original's fp4, or an EXL3 checkpoint's projections (run by `exl3_experts`).
    pub format: crate::dsv41::expert::ExpertFormat,
}

/// Decode steps after which every expert's count of uses halves ([`Uses`], and the RAM cache's own counts with
/// them): some two or three replies. Unaged, an expert a long conversation used ten thousand times would outrank, for
/// as long again, one a new subject uses at every step.
pub const USES_AGE_STEPS: u64 = 512;

/// Every routed expert's uses of late, whatever tier holds it: one for each call that routed to it (a decode step,
/// or a prompt's pass), halved every [`USES_AGE_STEPS`] decode steps. The model's 15,360 experts fit no computer's
/// memory whole, so where each lives is decided by this: the most used where they are computed quickest (a GPU's
/// memory, where there is one), the next in RAM, the rest on the drive. It is also what a usage profile keeps from
/// one run to the next, so a start-up reads the experts in that order.
pub struct Uses {
    counts: std::sync::Mutex<Vec<u32>>,
    per_layer: usize,
    decoded: std::sync::atomic::AtomicU64,
}

impl Uses {
    pub fn new(layers: usize, per_layer: usize) -> Uses {
        Uses { counts: std::sync::Mutex::new(vec![0; layers * per_layer]), per_layer, decoded: std::sync::atomic::AtomicU64::new(0) }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Vec<u32>> {
        self.counts.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Where `(layer, expert)`'s count is, if it is one of a layer's experts.
    fn at(&self, layer: u32, expert: u32) -> Option<usize> {
        ((expert as usize) < self.per_layer).then(|| layer as usize * self.per_layer + expert as usize)
    }

    /// A call routed to `experts` of `layer`.
    pub fn add(&self, layer: u32, experts: &[u32]) {
        let mut counts = self.held();
        for &e in experts {
            if let Some(n) = self.at(layer, e).and_then(|i| counts.get_mut(i)) {
                *n = n.saturating_add(1);
            }
        }
    }

    pub fn of(&self, layer: u32, expert: u32) -> u32 {
        self.at(layer, expert).and_then(|i| self.held().get(i).copied()).unwrap_or(0)
    }

    /// `(layers, experts a layer)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.held().len() / self.per_layer.max(1), self.per_layer)
    }

    /// Every count, layer by layer (`layer * experts a layer + expert`).
    pub fn counts(&self) -> Vec<u32> {
        self.held().clone()
    }

    /// Start from `counts` (another run's, by [`Self::counts`]); false, and nothing changed, if they are not this
    /// model's shape.
    pub fn seed(&self, counts: &[u32]) -> bool {
        let mut held = self.held();
        if counts.len() != held.len() {
            return false;
        }
        held.copy_from_slice(counts);
        true
    }

    /// The experts in the order a start-up should read them: the most used first, and of equals every layer's expert
    /// `e` before any layer's `e + 1` (a step uses each layer's alike).
    pub fn order(&self) -> Vec<(u32, u32)> {
        let counts = self.held();
        let layers = counts.len() / self.per_layer.max(1);
        let mut order: Vec<(u32, u32)> = (0..counts.len()).map(|i| ((i % layers) as u32, (i / layers) as u32)).collect();
        order.sort_by_key(|&(layer, expert)| std::cmp::Reverse(counts[layer as usize * self.per_layer + expert as usize]));
        order
    }

    /// A decode step is done: whether the counts were halved for it.
    pub fn step(&self) -> bool {
        if (self.decoded.fetch_add(1, Ordering::Relaxed) + 1) % USES_AGE_STEPS != 0 {
            return false;
        }
        for n in self.held().iter_mut() {
            *n /= 2;
        }
        true
    }
}

/// Tokens a pass needs before its layers' experts are read while their attention runs: a prompt this long uses most of
/// every layer's experts (a 2,000-token one, 80% of them), and the drive is idle otherwise; a shorter pass would fill
/// the cache with experts it does not use.
pub const PREFETCH_TOKENS: usize = 256;

impl Experts {
    /// Read `layer`'s routed experts into the RAM cache until `stop` (its router has chosen): those not cached already,
    /// in the order the GPU's kernel gives (what it holds left out), on [`READERS`] threads. A record its MoE then asks
    /// for is a hit, or, still being read, waited for rather than read again.
    pub fn prefetch(&self, layer: u32, experts: u32, stop: &std::sync::atomic::AtomicBool) {
        let order: Vec<u32> = match &self.gpu {
            Some(g) => g.prefetch_order(layer, experts),
            None => (0..experts).collect(),
        };
        let order: Vec<u32> = order.into_iter().filter(|&e| !self.cache.probe(layer, e)).collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..READERS.min(order.len()) {
                scope.spawn(|| {
                    while !stop.load(Ordering::Relaxed) {
                        let Some(&e) = order.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                        // the lease dropped at once: a leased record is never evicted
                        let _ = self.cache.acquire(layer, e, self.store.as_ref());
                    }
                });
            }
        });
    }

    /// Read `picks` (`(layer, expert)` each) into the RAM cache on [`READERS`] threads: the drive read ahead of the
    /// steps that will ask for them. One already there is left as it is.
    pub fn warm(&self, picks: &[(u32, u32)]) {
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..READERS.min(picks.len()) {
                scope.spawn(|| {
                    while let Some(&(layer, e)) = picks.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if !self.cache.probe(layer, e) {
                            // the lease dropped at once: a leased record is never evicted
                            let _ = self.cache.acquire(layer, e, self.store.as_ref());
                        }
                    }
                });
            }
        });
    }
}

/// Tokens of a prompt an expert needs before a GPU may take it from the CPU's workers (its record's upload is 18.8 MB):
/// with fewer, a worker's matmuls of a few rows are quicker than the copy.
pub const GPU_MIN_ROWS: usize = 8;

/// [`GPU_MIN_ROWS`], or what OAIY_DSV41_GPU_MIN_ROWS says (a measurement's: where the CPU's workers and a card's
/// uploads balance on a machine). An EXL3 checkpoint's experts go to a GPU from one token: the CPU decodes their
/// trellis at milliseconds of every core an expert, slower than its upload.
fn gpu_min_rows(format: &crate::dsv41::expert::ExpertFormat) -> usize {
    static ROWS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let asked = *ROWS.get_or_init(|| std::env::var("OAIY_DSV41_GPU_MIN_ROWS").ok().and_then(|v| v.parse().ok()));
    asked.unwrap_or(match format {
        crate::dsv41::expert::ExpertFormat::Exl3 { .. } => 1,
        _ => GPU_MIN_ROWS,
    })
}
/// Experts a GPU takes in one call: their records' upload bounded (about 600 MB), the matmuls a stage's round trip.
pub const GPU_GROUP: usize = 32;
/// Rows an EXL3 checkpoint's pass may have and its experts still go a decode step's way on a GPU (a draft's check): the
/// cards' experts begun before any record is read, and what is read kept on them as a step's is. A prompt's way read
/// with threads started for each layer and kept nothing, so a check's experts went up again pass after pass.
const FEW_ROWS: usize = 8;
/// Threads reading a layer's expert records from the drive at once.
const READERS: usize = 8;

pub struct Moe {
    layer: u32,
    gate: Weight,
    bias: Vec<f32>,
    shared: [Weight; 3],
}

impl Moe {
    /// Its dense weights by name (the router and the shared expert), for [`crate::dsv41::model::Model::offload`].
    pub(crate) fn weights_mut(&mut self) -> Vec<(&'static str, &mut Weight)> {
        let [w1, w2, w3] = &mut self.shared;
        vec![("gate", &mut self.gate), ("shared.w1", w1), ("shared.w2", w2), ("shared.w3", w3)]
    }
}

/// `ids`' records, in order: those `there` (in the RAM tier) taken on this thread, the rest read on up to [`READERS`]
/// threads at once. A decode step's are nearly all there, and a thread started for each cost more than taking it (75 us
/// a layer, 3 ms a token); one alone to read is read here too.
fn read_all(ids: &[u32], there: &dyn Fn(u32) -> bool, acquire: &(dyn Fn(u32) -> Result<HostLease> + Sync)) -> Result<Vec<HostLease>> {
    let missing: Vec<usize> = (0..ids.len()).filter(|&i| !there(ids[i])).collect();
    if missing.len() <= 1 {
        return ids.iter().map(|&e| acquire(e)).collect();
    }
    let readers = READERS.min(missing.len());
    let mut got: Vec<Option<Result<HostLease>>> = (0..ids.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let missing = &missing;
        let handles: Vec<_> =
            (0..readers).map(|r| scope.spawn(move || (r..missing.len()).step_by(readers).map(|m| (missing[m], acquire(ids[missing[m]]))).collect::<Vec<_>>())).collect();
        for i in (0..ids.len()).filter(|i| !missing.contains(i)) {
            got[i] = Some(acquire(ids[i]));
        }
        for (i, record) in handles.into_iter().flat_map(|h| h.join().expect("an expert reader panicked")) {
            got[i] = Some(record);
        }
    });
    got.into_iter().map(|record| record.expect("every record asked for")).collect()
}

/// Routed experts one token was sent to, with their weights.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
}

impl Moe {
    pub fn load(idx: &StIndex, layer: usize) -> Result<Moe> {
        Self::load_named(idx, layer, &format!("layers.{layer}.ffn"))
    }

    /// [`Moe::load`] of the weights named `p.*` (DSpark's drafter's `mtp.<i>.ffn`), `layer` its experts' layer in their
    /// store.
    pub fn load_named(idx: &StIndex, layer: usize, p: &str) -> Result<Moe> {
        Ok(Moe {
            layer: layer as u32,
            gate: Weight::load(idx, &format!("{p}.gate"))?,
            bias: load_vec(idx, &format!("{p}.gate.bias"))?,
            shared: [
                Weight::load(idx, &format!("{p}.shared_experts.w1"))?,
                Weight::load(idx, &format!("{p}.shared_experts.w2"))?,
                Weight::load(idx, &format!("{p}.shared_experts.w3"))?,
            ],
        })
    }

    /// Router decision for `t` tokens of `x` (`[t, dim]`, bf16 values).
    pub fn route(&self, cfg: &Config, x: &[f32], t: usize) -> Vec<Route> {
        route(cfg, &self.gate.forward(x, t, Out::F32), &self.bias, t)
    }
}

/// Routing from the gate's f32 logits (`[t, n_experts]`) and correction bias:
/// the bias picks the top experts, the unbiased `sqrt(softplus)` scores
/// weight them (normalized, times `route_scale`). Ties: lower id first.
pub fn route(cfg: &Config, logits: &[f32], bias: &[f32], t: usize) -> Vec<Route> {
    route_mixed(cfg, logits, bias, None, t)
}

/// [`route`] where image tokens (`image_bias.1[i]`) pick their experts with
/// the vision correction bias `image_bias.0` (the checkpoint's `bias_vl`,
/// the reference `Gate`'s `image_mask` path).
pub fn route_mixed(cfg: &Config, logits: &[f32], bias: &[f32], image_bias: Option<(&[f32], &[bool])>, t: usize) -> Vec<Route> {
    let n = bias.len();
    (0..t)
        .map(|i| {
            let bias = match image_bias {
                Some((vl, mask)) if mask[i] => vl,
                _ => bias,
            };
            let scores: Vec<f32> = logits[i * n..(i + 1) * n].iter().map(|&v| softplus(v).sqrt()).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])).then(a.cmp(&b)));
            order.truncate(cfg.n_activated_experts);
            let mut weights: Vec<f32> = order.iter().map(|&e| scores[e]).collect();
            if weights.len() > 1 {
                let s = weights.iter().sum::<f32>() + 1e-20;
                weights.iter_mut().for_each(|w| *w /= s);
            }
            weights.iter_mut().for_each(|w| *w *= cfg.route_scale);
            Route { experts: order.into_iter().map(|e| e as u32).collect(), weights }
        })
        .collect()
}

impl Moe {

    /// `x`: `[t, dim]` (bf16 values); returns `[t, dim]` bf16 and the routes taken.
    pub fn forward(&self, cfg: &Config, x: &[f32], t: usize, experts: &Experts) -> Result<(Vec<f32>, Vec<Route>)> {
        let d = cfg.dim;
        let [w1, w2, w3] = &self.shared;
        // A decode step's shared expert whole and the router's logits in one call, where a device holds them (its
        // down projection was a round trip of its own, a layer); else the router's and the shared expert's
        // projections of x together, and its down projection after the routed experts.
        let whole = (t == 1).then(|| crate::dsv41::linear::gated_together(w1, w3, w2, x, cfg.swiglu_limit, &[(&self.gate, x, 0..self.gate.n(), Out::F32)])).flatten();
        let (logits, shared_parts, shared_whole) = match whole {
            Some((shared, mut logits)) => (logits.pop().expect("the router's logits"), None, Some(shared)),
            None => {
                let [logits, gate, up]: [Vec<f32>; 3] = crate::dsv41::linear::forward_together(&[
                    (&self.gate, x, t, 0..self.gate.n(), Out::F32),
                    (w1, x, t, 0..w1.n(), Out::Bf16),
                    (w3, x, t, 0..w3.n(), Out::Bf16),
                ])
                .try_into()
                .expect("three projections");
                (logits, Some((gate, up)), None)
            }
        };
        let routes = route(cfg, &logits, &self.bias, t);
        let mut y = vec![0.0f32; t * d];
        let mut used: Vec<u32> = routes.iter().flat_map(|r| r.experts.iter().copied()).collect();
        used.sort_unstable();
        used.dedup();
        experts.uses.add(self.layer, &used);
        // Each expert's (tokens, outputs), every expert's computed before any is added, then added in ascending
        // expert order, as one at a time added them: the same sums either way.
        // Their records, read on several threads at once: an SSD serves a queue of reads several times faster than one
        // at a time (the cache reads a record outside its lock, and the store keeps a scratch buffer a read).
        let acquire = |e: u32| experts.cache.acquire(self.layer, e, experts.store.as_ref());
        // An EXL3 checkpoint's decode step. With a GPU's kernel, every expert there: those the cards hold from their
        // slots, the rest read (RAM, else the drive), taken into a free slot while there is one (and let go from RAM),
        // and what is left through the scratch slots (a record's upload is a fraction of a millisecond, its trellis
        // decoded on the CPU some milliseconds of every core). Without one, on every core, each projection's columns in
        // tasks of their own (the pool's workers run the fp4 kernels).
        // (and a pass of a few rows, a draft's check, where a GPU's kernel runs them: each expert on its rows, kept on
        // the cards as a step's are, where a prompt's are not)
        #[cfg(feature = "webgpu")]
        if let (crate::dsv41::expert::ExpertFormat::Exl3 { layers, .. }, true) = (&experts.format, t == 1 || (t <= FEW_ROWS && experts.gpu.is_some())) {
            // every row routed to each expert, in row order: its rows of x and their weights
            let gathered: Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> = used
                .iter()
                .map(|e| {
                    let (mut toks, mut xs, mut ws) = (Vec::new(), Vec::new(), Vec::new());
                    for (i, r) in routes.iter().enumerate() {
                        if let Some(k) = r.experts.iter().position(|x| x == e) {
                            toks.push(i);
                            xs.extend_from_slice(&x[i * d..(i + 1) * d]);
                            ws.push(r.weights[k]);
                        }
                    }
                    (toks, xs, ws)
                })
                .collect();
            let gpu = experts.gpu.as_ref();
            let held = gpu.map_or_else(|| vec![false; used.len()], |g| g.holds(self.layer, &used, &gathered.iter().map(|g| g.0.len()).collect::<Vec<_>>()));
            let ids: Vec<u32> = (0..used.len()).filter(|&j| !held[j]).map(|j| used[j]).collect();
            let mut shared = shared_whole;
            let outs: Vec<Vec<f32>> = match gpu {
                // the held experts begun on their cards before anything is read, which they need not wait for, and the
                // shared expert's down projection beside the routed ones (its own round trip after them before)
                Some(gpu) => {
                    let there: Vec<usize> = (0..used.len()).filter(|&j| held[j]).collect();
                    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); used.len()];
                    // (begun: each card's call submitted here, and waited for once the rest are made)
                    let jobs: Vec<(u32, &[f32], &[f32])> = there.iter().map(|&j| (used[j], &gathered[j].1[..], &gathered[j].2[..])).collect();
                    let cards = (!there.is_empty()).then(|| gpu.begin_held(self.layer, &jobs, cfg.swiglu_limit));
                    std::thread::scope(|scope| -> Result<()> {
                        let down = match (&shared, &shared_parts) {
                            (None, Some((gate, up))) => Some(scope.spawn(move || self.shared_forward(cfg, gate, up, t))),
                            _ => None,
                        };
                        let reading = std::time::Instant::now();
                        let leases = read_all(&ids, &|e| experts.cache.probe(self.layer, e), &acquire)?;
                        crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertRead, reading);
                        let computing = std::time::Instant::now();
                        // what was read through the scratch slots (the cards keep what is used enough, and RAM then
                        // lets it go: an expert in one place)
                        let read: Vec<(u32, &[u8])> = ids.iter().zip(&leases).map(|(&e, l)| (e, &**l)).collect();
                        let at = |e: u32| used.iter().position(|&u| u == e).expect("a used expert");
                        if !read.is_empty() {
                            let jobs: Vec<crate::dsv41::expert::ExpertJob<'_>> =
                                read.iter().map(|&(e, record)| crate::dsv41::expert::ExpertJob { expert: e, record, x: &gathered[at(e)].1, weights: &gathered[at(e)].2 }).collect();
                            for (&(e, _), out) in read.iter().zip(gpu.forward(self.layer, &jobs, cfg.swiglu_limit)) {
                                outs[at(e)] = out;
                            }
                        }
                        let taken = gpu.offer(self.layer, &read);
                        if let Some(finish) = cards {
                            for (&j, out) in there.iter().zip(finish()) {
                                outs[j] = out;
                            }
                        }
                        if let Some(h) = down {
                            shared = Some(h.join().expect("the shared expert panicked"));
                        }
                        crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertGpu, computing);
                        drop(read);
                        drop(leases);
                        for e in taken {
                            experts.cache.remove(self.layer, e);
                        }
                        Ok(())
                    })?;
                    outs
                }
                // (one row: no GPU takes more)
                None => {
                    let reading = std::time::Instant::now();
                    let leases = read_all(&ids, &|e| experts.cache.probe(self.layer, e), &acquire)?;
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertRead, reading);
                    let computing = std::time::Instant::now();
                    let records: Vec<&[u8]> = leases.iter().map(|l| &l[..]).collect();
                    let weights: Vec<f32> = gathered.iter().map(|g| g.2[0]).collect();
                    let outs = crate::dsv41::exl3_experts::forward_step(&records, &layers[self.layer as usize], x, &weights, cfg.swiglu_limit)?;
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertCpu, computing);
                    outs
                }
            };
            // (the sums in ascending expert order, as the other paths add them, each row its own)
            for ((toks, ..), out) in gathered.iter().zip(&outs) {
                for (k, &i) in toks.iter().enumerate() {
                    for (acc, v) in y[i * d..(i + 1) * d].iter_mut().zip(&out[k * d..(k + 1) * d]) {
                        *acc += v;
                    }
                }
            }
            let shared = match (shared, shared_parts) {
                (Some(shared), _) => shared,
                (None, Some((gate, up))) => self.shared_forward(cfg, &gate, &up, t),
                (None, None) => unreachable!("the shared expert is made one way or the other"),
            };
            for (acc, v) in y.iter_mut().zip(shared) {
                *acc = to_bf16(*acc + v);
            }
            return Ok((y, routes));
        }
        let outs: Vec<(Vec<usize>, Vec<f32>)> = match (&experts.pool, t) {
            // A decode step: the token's experts the GPU keeps computed there, while the rest are read and computed at
            // once on the CPU's workers; then the GPU may take in what was read.
            (Some(pool), 1) => {
                let r = &routes[0];
                let weights: Vec<f32> = used.iter().map(|e| r.weights[r.experts.iter().position(|x| x == e).expect("a used expert is routed")]).collect();
                let gpu = experts.gpu.as_ref();
                let held = gpu.map_or_else(|| vec![false; used.len()], |g| g.holds(self.layer, &used, &vec![1; used.len()]));
                let (there, here): (Vec<usize>, Vec<usize>) = (0..used.len()).partition(|&j| held[j]);
                let mut outs: Vec<Vec<f32>> = vec![Vec::new(); used.len()];
                // The device's experts are begun and later taken, the rest read and computed by the workers meanwhile:
                // no thread is started for the device (one was each layer, and one more for each card past the first:
                // some 60 us each to start and as much to join). Which is begun first is what hides the other: with a
                // record to read from the drive, the device's (the read is milliseconds); with all of them in RAM,
                // the workers', and the device's calls are made beside their matmuls.
                let begin = || {
                    gpu.filter(|_| !there.is_empty()).map(|gpu| {
                        let w: Vec<[f32; 1]> = there.iter().map(|&j| [weights[j]]).collect();
                        let jobs: Vec<(u32, &[f32], &[f32])> = there.iter().zip(&w).map(|(&j, w)| (used[j], x, &w[..])).collect();
                        let beginning = std::time::Instant::now();
                        let finish = gpu.begin_held(self.layer, &jobs, cfg.swiglu_limit);
                        (finish, beginning.elapsed())
                    })
                };
                let ids: Vec<u32> = here.iter().map(|&j| used[j]).collect();
                let to_read = ids.iter().any(|&e| !experts.cache.probe(self.layer, e));
                let mut device = if to_read { begin() } else { None };
                let reading = std::time::Instant::now();
                let leases = read_all(&ids, &|e| experts.cache.probe(self.layer, e), &acquire)?;
                crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertRead, reading);
                let computing = std::time::Instant::now();
                let workers = (!here.is_empty()).then(|| {
                    let (tx, rx) = mpsc::channel();
                    let records: Vec<_> = leases.iter().map(|l| l.to_arc()).collect();
                    let w: Vec<f32> = here.iter().map(|&j| weights[j]).collect();
                    pool.spawn(records, w, x.to_vec(), cfg.swiglu_limit, Box::new(move |out| drop(tx.send(out))));
                    rx
                });
                if !to_read {
                    device = begin();
                }
                if let Some(rx) = workers {
                    let made = rx.recv().expect("CPU expert job answered").expect("CPU expert job");
                    for (&j, out) in here.iter().zip(made) {
                        outs[j] = out;
                    }
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertCpu, computing);
                }
                if let Some((finish, begun)) = device {
                    let finishing = std::time::Instant::now();
                    for (&j, out) in there.iter().zip(finish()) {
                        outs[j] = out;
                    }
                    crate::dsv41::profile::add_spent(crate::dsv41::profile::Part::ExpertGpu, begun + finishing.elapsed());
                }
                // What the GPU takes in leaves the RAM tier (once the leases are gone): the two hold different
                // experts, so between them more.
                let taken = gpu.map_or_else(Vec::new, |gpu| {
                    let read: Vec<(u32, &[u8])> = ids.iter().zip(&leases).map(|(&e, l)| (e, &**l)).collect();
                    gpu.offer(self.layer, &read)
                });
                drop(leases);
                for e in taken {
                    experts.cache.remove(self.layer, e);
                }
                outs.into_iter().map(|out| (vec![0], out)).collect()
            }
            // A prompt: each expert's tokens as one batch. The records are handed over as they land, the busy experts'
            // (GPU_MIN_ROWS tokens or more) to the GPU when there is one, a group (GPU_GROUP) as soon as a group's worth
            // has arrived, the rest to the CPU's workers meanwhile: the reads, the GPU and the CPU overlap. Done one
            // after another, the reads were a third of a prompt's MoE and the experts' matmuls the rest.
            _ => {
                // every token routed to each expert, in token order
                let gathered: Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> = used
                    .iter()
                    .map(|e| {
                        let (mut toks, mut xs, mut ws) = (Vec::new(), Vec::new(), Vec::new());
                        for (i, r) in routes.iter().enumerate() {
                            if let Some(k) = r.experts.iter().position(|x| x == e) {
                                toks.push(i);
                                xs.extend_from_slice(&x[i * d..(i + 1) * d]);
                                ws.push(r.weights[k]);
                            }
                        }
                        (toks, xs, ws)
                    })
                    .collect();
                let done = self.prompt_experts(cfg, experts, &used, &gathered, &acquire)?;
                gathered.into_iter().zip(done).map(|((toks, _, _), out)| (toks, out)).collect()
            }
        };
        for (toks, out) in outs {
            for (j, &i) in toks.iter().enumerate() {
                for (acc, v) in y[i * d..(i + 1) * d].iter_mut().zip(&out[j * d..(j + 1) * d]) {
                    *acc += v;
                }
            }
        }
        let shared = match (shared_whole, shared_parts) {
            (Some(shared), _) => shared,
            (None, Some((gate, up))) => self.shared_forward(cfg, &gate, &up, t),
            (None, None) => unreachable!("the shared expert is made one way or the other"),
        };
        for (acc, v) in y.iter_mut().zip(shared) {
            *acc = to_bf16(*acc + v);
        }
        Ok((y, routes))
    }

    /// A prompt's routed experts, each expert `used[j]` on its tokens' rows `gathered[j]`: its outputs, in `used`'s
    /// order. What a GPU holds it computes, and nobody reads. The rest are read (most tokens first) into one queue
    /// kept by their tokens, and taken from both ends as hands come free: the CPU's workers the fewest tokens' (an
    /// expert's tokens all at once through the tokens kernel, each worker its own thread's), the GPU, once it has
    /// made what it holds, the most tokens' a group at a time ([`GPU_MIN_ROWS`] tokens or more: under that the upload
    /// costs more than a worker's matmuls). So the two work at once and share the experts by what each gets
    /// through, whatever the machine: with no GPU the workers take all, and a GPU beside few cores takes most.
    /// Before, the busy ones were the GPU's whatever the workers had to do, and what it held was made before a
    /// worker was given anything: a 276-token prompt's experts were 9.5 s on a card with every core idle. The
    /// outputs are the ones each expert alone gives, whatever makes them and in whatever order.
    fn prompt_experts(
        &self,
        cfg: &Config,
        experts: &Experts,
        used: &[u32],
        gathered: &[(Vec<usize>, Vec<f32>, Vec<f32>)],
        acquire: &(dyn Fn(u32) -> Result<HostLease> + Sync),
    ) -> Result<Vec<Vec<f32>>> {
        /// What has been read and nobody has taken (fewest tokens first), how many records are still to land, and
        /// the first read that failed.
        struct Landed<E> {
            ready: std::collections::VecDeque<(usize, HostLease)>,
            left: usize,
            failed: Option<E>,
        }
        let n = used.len();
        let gpu = experts.gpu.as_ref();
        let tokens: Vec<usize> = gathered.iter().map(|g| g.0.len()).collect();
        let held = gpu.map_or_else(|| vec![false; n], |g| g.holds(self.layer, used, &tokens));
        let mut order: Vec<usize> = (0..n).filter(|&j| !held[j]).collect();
        order.sort_by_key(|&j| std::cmp::Reverse(tokens[j]));
        // (workers on every hardware thread but two where a GPU works beside them: the thread that drives it quantizes
        // its rows and makes their SwiGLU between its calls, and on a core it shared with a worker what the GPU held
        // took 5.8 s of a 276-token prompt where 3.5)
        let all = std::thread::available_parallelism().map_or(1, |n| n.get());
        // (an EXL3 checkpoint's experts all on the GPU where there is one: the CPU decodes their trellis at milliseconds
        // of every core an expert, slower than the upload, and its sums round apart from the GPU's, so a prompt's output
        // hung on which of them each expert happened to go to)
        let exl3 = matches!(experts.format, crate::dsv41::expert::ExpertFormat::Exl3 { .. });
        let threads = if exl3 && gpu.is_some() {
            0
        } else if experts.pool.is_none() {
            1
        } else if gpu.is_some() {
            all.saturating_sub(2).max(1)
        } else {
            all
        };
        let floor = gpu_min_rows(&experts.format);
        let state = Mutex::new(Landed { ready: std::collections::VecDeque::new(), left: order.len(), failed: None });
        let landed = std::sync::Condvar::new();
        let next = AtomicUsize::new(0);
        let mut done: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
        let (out_tx, out_rx) = mpsc::channel::<(usize, Vec<f32>)>();
        std::thread::scope(|scope| {
            let (state, landed, order, next, tokens) = (&state, &landed, &order, &next, &tokens);
            for _ in 0..READERS.min(order.len()) {
                scope.spawn(move || {
                    while let Some(&j) = order.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let record = acquire(used[j]);
                        let mut s = state.lock().unwrap_or_else(|p| p.into_inner());
                        s.left -= 1;
                        match record {
                            Ok(record) => {
                                let at = s.ready.partition_point(|(i, _)| tokens[*i] <= tokens[j]);
                                s.ready.insert(at, (j, record));
                            }
                            Err(e) => {
                                s.failed.get_or_insert(e);
                            }
                        }
                        drop(s);
                        landed.notify_all();
                    }
                });
            }
            let (row_kernel, tokens_kernel) = (experts.kernel, experts.tokens);
            for _ in 0..threads.min(order.len()) {
                let out_tx = out_tx.clone();
                scope.spawn(move || loop {
                    let job = {
                        let mut s = state.lock().unwrap_or_else(|p| p.into_inner());
                        loop {
                            if let Some(job) = s.ready.pop_front() {
                                break Some(job);
                            }
                            if s.left == 0 {
                                break None;
                            }
                            s = landed.wait(s).unwrap_or_else(|p| p.into_inner());
                        }
                    };
                    let Some((j, record)) = job else { break };
                    // (one token: the row kernel, which decodes nothing it does not multiply at once)
                    let (x, w) = (&gathered[j].1, &gathered[j].2);
                    // An EXL3 checkpoint's expert through its own decoder; a record it refuses fails the pass (its
                    // output left empty: the pass returns the error before it is used).
                    #[cfg(feature = "webgpu")]
                    if let crate::dsv41::expert::ExpertFormat::Exl3 { layers, .. } = &experts.format {
                        let out = crate::dsv41::exl3_experts::forward_rows(&record, &layers[self.layer as usize], x, Some(w), cfg.swiglu_limit).unwrap_or_else(|e| {
                            state.lock().unwrap_or_else(|p| p.into_inner()).failed.get_or_insert(e);
                            Vec::new()
                        });
                        if out_tx.send((j, out)).is_err() {
                            break;
                        }
                        continue;
                    }
                    let out = if w.len() == 1 {
                        crate::dsv41::expert::expert_forward_rows(row_kernel, &record, x, Some(w), cfg.swiglu_limit)
                    } else {
                        crate::dsv41::expert::expert_forward_tokens(tokens_kernel, &record, x, Some(w), cfg.swiglu_limit)
                    };
                    if out_tx.send((j, out)).is_err() {
                        break;
                    }
                });
            }
            drop(out_tx);
            if let Some(gpu) = gpu {
                let there: Vec<usize> = (0..n).filter(|&j| held[j]).collect();
                for part in there.chunks(GPU_GROUP) {
                    let jobs: Vec<(u32, &[f32], &[f32])> = part.iter().map(|&j| (used[j], &gathered[j].1[..], &gathered[j].2[..])).collect();
                    let computing = std::time::Instant::now();
                    let got = gpu.forward_held(self.layer, &jobs, cfg.swiglu_limit);
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertGpu, computing);
                    for (&j, out) in part.iter().zip(got) {
                        done[j] = Some(out);
                    }
                }
                // then the most tokens' of what has landed, a group at a time, until nothing of its kind is left
                loop {
                    let group: Vec<(usize, HostLease)> = {
                        let mut s = state.lock().unwrap_or_else(|p| p.into_inner());
                        loop {
                            let mine = s.ready.iter().rev().take_while(|(j, _)| tokens[*j] >= floor).count().min(GPU_GROUP);
                            if mine > 0 {
                                let at = s.ready.len() - mine;
                                break s.ready.split_off(at).into_iter().collect();
                            }
                            if s.left == 0 {
                                break Vec::new();
                            }
                            let waiting = std::time::Instant::now();
                            s = landed.wait(s).unwrap_or_else(|p| p.into_inner());
                            crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertRead, waiting);
                        }
                    };
                    if group.is_empty() {
                        break;
                    }
                    let jobs: Vec<crate::dsv41::expert::ExpertJob<'_>> =
                        group.iter().map(|(j, record)| crate::dsv41::expert::ExpertJob { expert: used[*j], record, x: &gathered[*j].1, weights: &gathered[*j].2 }).collect();
                    let computing = std::time::Instant::now();
                    let got = gpu.forward(self.layer, &jobs, cfg.swiglu_limit);
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertGpu, computing);
                    for ((j, _), out) in group.iter().zip(got) {
                        done[*j] = Some(out);
                    }
                }
            }
            let waiting = std::time::Instant::now();
            let mut made = 0usize;
            for (j, out) in out_rx {
                done[j] = Some(out);
                made += 1;
            }
            if made > 0 {
                crate::dsv41::profile::add(crate::dsv41::profile::Part::ExpertCpu, waiting);
            }
        });
        if let Some(e) = state.into_inner().unwrap_or_else(|p| p.into_inner()).failed {
            return Err(e);
        }
        Ok(done.into_iter().map(|o| o.expect("every expert computed")).collect())
    }

    /// The fp8 shared expert from its gate and up projections: same SwiGLU as a routed one, no route weight.
    fn shared_forward(&self, cfg: &Config, gate: &[f32], up: &[f32], t: usize) -> Vec<f32> {
        let w2 = &self.shared[1];
        let lim = cfg.swiglu_limit;
        let h: Vec<f32> = gate
            .iter()
            .zip(up)
            .map(|(&g, &u)| {
                let (g, u) = if lim > 0.0 { (g.min(lim), u.clamp(-lim, lim)) } else { (g, u) };
                to_bf16(silu(g) * u)
            })
            .collect();
        w2.forward(&h, t, Out::Bf16)
    }
}

#[cfg(test)]
mod uses_tests {
    use super::*;

    /// The counts are a call's worth each, halve once every [`USES_AGE_STEPS`] decode steps, give a start-up its order
    /// (the most used first, then every layer's expert e before any layer's e + 1), and carry over to another run of
    /// the same shape and to no other.
    #[test]
    fn the_counts_age_and_order_a_start_up() {
        let uses = Uses::new(3, 4);
        assert_eq!(uses.shape(), (3, 4));
        for _ in 0..6 {
            uses.add(2, &[1, 3]);
        }
        uses.add(0, &[3]);
        uses.add(0, &[3, 9]);
        assert_eq!((uses.of(2, 1), uses.of(2, 3), uses.of(0, 3), uses.of(1, 0), uses.of(0, 9)), (6, 6, 2, 0, 0));
        assert_eq!(uses.order()[..6], [(2, 1), (2, 3), (0, 3), (0, 0), (1, 0), (2, 0)]);
        assert_eq!(uses.order().len(), 12);
        assert_eq!((0..USES_AGE_STEPS - 1).filter(|_| uses.step()).count(), 0);
        assert!(uses.step(), "the last of the steps halves");
        assert_eq!((uses.of(2, 1), uses.of(0, 3)), (3, 1));
        let next = Uses::new(3, 4);
        assert!(next.seed(&uses.counts()) && next.counts() == uses.counts());
        assert!(!Uses::new(3, 5).seed(&uses.counts()), "another shape's counts are not taken");
    }
}
