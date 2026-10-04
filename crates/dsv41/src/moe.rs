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

use crate::config::Config;
use crate::expert::expert_forward_batch;
use crate::formats::to_bf16;
use crate::linear::{load_vec, Out, Weight};
use crate::ops::{silu, softplus};
use crate::safetensors::StIndex;

/// Where routed expert records come from: a store behind a bounded cache.
pub struct Experts {
    pub store: Arc<dyn WeightStore>,
    pub cache: Ecache,
    /// Workers for a decode step's routed experts, all of a token's at once (the same results as one at a time);
    /// none: one after another on the caller's thread.
    pub pool: Option<crate::cpu_experts::CpuExperts>,
    /// Where a prompt's busy experts (those [`GPU_MIN_ROWS`] or more of its tokens chose) have their matmuls made, a
    /// group of [`GPU_GROUP`] a call: a GPU's (`crate::expert::ExpertsKernel`). None: all on the CPU.
    pub gpu: Option<Arc<dyn crate::expert::ExpertsKernel>>,
}

/// Tokens of a prompt an expert needs before a GPU is worth its record's upload (18.8 MB): with fewer, the CPU's
/// matmuls of a few rows are quicker than the copy.
pub const GPU_MIN_ROWS: usize = 8;
/// Experts a GPU takes in one call: their records' upload bounded (about 600 MB), the matmuls a stage's round trip.
pub const GPU_GROUP: usize = 32;
/// Threads reading a layer's expert records from the drive at once.
const READERS: usize = 8;

pub struct Moe {
    layer: u32,
    gate: Weight,
    bias: Vec<f32>,
    shared: [Weight; 3],
}

impl Moe {
    /// Its dense weights by name (the router and the shared expert), for [`crate::model::Model::offload`].
    pub(crate) fn weights_mut(&mut self) -> Vec<(&'static str, &mut Weight)> {
        let [w1, w2, w3] = &mut self.shared;
        vec![("gate", &mut self.gate), ("shared.w1", w1), ("shared.w2", w2), ("shared.w3", w3)]
    }
}

/// `ids`' records, in order, read on up to [`READERS`] threads at once.
fn read_all(ids: &[u32], acquire: &(dyn Fn(u32) -> Result<HostLease> + Sync)) -> Result<Vec<HostLease>> {
    if ids.len() <= 1 {
        return ids.iter().map(|&e| acquire(e)).collect();
    }
    let readers = READERS.min(ids.len());
    let got: Vec<Result<HostLease>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..readers).map(|r| scope.spawn(move || (r..ids.len()).step_by(readers).map(|i| (i, acquire(ids[i]))).collect::<Vec<_>>())).collect();
        let mut all: Vec<(usize, Result<HostLease>)> = handles.into_iter().flat_map(|h| h.join().expect("an expert reader panicked")).collect();
        all.sort_by_key(|(i, _)| *i);
        all.into_iter().map(|(_, r)| r).collect()
    });
    got.into_iter().collect()
}

/// Routed experts one token was sent to, with their weights.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
}

impl Moe {
    pub fn load(idx: &StIndex, layer: usize) -> Result<Moe> {
        let p = format!("layers.{layer}.ffn");
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
        // The router's and the shared expert's projections of x, together.
        let [w1, _, w3] = &self.shared;
        let [logits, gate, up]: [Vec<f32>; 3] = crate::linear::forward_together(&[
            (&self.gate, x, t, 0..self.gate.n(), Out::F32),
            (w1, x, t, 0..w1.n(), Out::Bf16),
            (w3, x, t, 0..w3.n(), Out::Bf16),
        ])
        .try_into()
        .expect("three projections");
        let routes = route(cfg, &logits, &self.bias, t);
        let mut y = vec![0.0f32; t * d];
        let mut used: Vec<u32> = routes.iter().flat_map(|r| r.experts.iter().copied()).collect();
        used.sort_unstable();
        used.dedup();
        // Each expert's (tokens, outputs), every expert's computed before any is added, then added in ascending
        // expert order, as one at a time added them: the same sums either way.
        // Their records, read on several threads at once: an SSD serves a queue of reads several times faster than one
        // at a time (the cache reads a record outside its lock, and the store keeps a scratch buffer a read).
        let acquire = |e: u32| experts.cache.acquire(self.layer, e, experts.store.as_ref());
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
                std::thread::scope(|scope| -> Result<()> {
                    let device = gpu.filter(|_| !there.is_empty()).map(|gpu| {
                        let (there, weights, used) = (&there, &weights, &used);
                        scope.spawn(move || {
                            let w: Vec<[f32; 1]> = there.iter().map(|&j| [weights[j]]).collect();
                            let jobs: Vec<(u32, &[f32], &[f32])> = there.iter().zip(&w).map(|(&j, w)| (used[j], x, &w[..])).collect();
                            let computing = std::time::Instant::now();
                            let got = gpu.forward_held(self.layer, &jobs, cfg.swiglu_limit);
                            crate::profile::add(crate::profile::Part::ExpertGpu, computing);
                            got
                        })
                    });
                    let ids: Vec<u32> = here.iter().map(|&j| used[j]).collect();
                    let reading = std::time::Instant::now();
                    let leases = read_all(&ids, &acquire)?;
                    crate::profile::add(crate::profile::Part::ExpertRead, reading);
                    if !here.is_empty() {
                        let records: Vec<_> = leases.iter().map(|l| l.to_arc()).collect();
                        let w: Vec<f32> = here.iter().map(|&j| weights[j]).collect();
                        let computing = std::time::Instant::now();
                        for (&j, out) in here.iter().zip(pool.forward(&records, &w, x, cfg.swiglu_limit)) {
                            outs[j] = out;
                        }
                        crate::profile::add(crate::profile::Part::ExpertCpu, computing);
                    }
                    if let Some(device) = device {
                        for (&j, out) in there.iter().zip(device.join().expect("the GPU's experts panicked")) {
                            outs[j] = out;
                        }
                    }
                    if let Some(gpu) = gpu {
                        let read: Vec<(u32, &[u8])> = ids.iter().zip(&leases).map(|(&e, l)| (e, &**l)).collect();
                        gpu.offer(self.layer, &read);
                    }
                    Ok(())
                })?;
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
        let shared = self.shared_forward(cfg, &gate, &up, t);
        for (acc, v) in y.iter_mut().zip(shared) {
            *acc = to_bf16(*acc + v);
        }
        Ok((y, routes))
    }

    /// A prompt's routed experts, each expert `used[j]` on its tokens' rows `gathered[j]`: its outputs, in `used`'s
    /// order. Readers fetch the records (the busy experts' first, so the GPU's groups start while the rest are read) and
    /// hand each over as it lands; this thread sends a group of busy ones to the GPU whenever a group's worth is in, and
    /// the rest to workers on the CPU's threads. The outputs are the ones each expert alone gives, whatever the order
    /// they are made in (a GPU's group or a worker computes each expert by itself).
    fn prompt_experts(
        &self,
        cfg: &Config,
        experts: &Experts,
        used: &[u32],
        gathered: &[(Vec<usize>, Vec<f32>, Vec<f32>)],
        acquire: &(dyn Fn(u32) -> Result<HostLease> + Sync),
    ) -> Result<Vec<Vec<f32>>> {
        let n = used.len();
        let gpu = experts.gpu.as_ref();
        let tokens: Vec<usize> = gathered.iter().map(|g| g.0.len()).collect();
        // What the GPU keeps is computed there and not read; of the rest, the busy experts go to it a group at a time.
        let held = gpu.map_or_else(|| vec![false; n], |g| g.holds(self.layer, used, &tokens));
        let busy: Vec<bool> = (0..n).map(|j| !held[j] && gpu.is_some() && tokens[j] >= GPU_MIN_ROWS).collect();
        let order: Vec<usize> = (0..n).filter(|&j| busy[j]).chain((0..n).filter(|&j| !busy[j] && !held[j])).collect();
        let light = order.iter().filter(|&&j| !busy[j]).count();
        let threads = if experts.pool.is_some() { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { 1 };
        let mut done: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
        let mut failed = None;
        let (got_tx, got_rx) = mpsc::channel::<(usize, Result<HostLease>)>();
        let next = AtomicUsize::new(0);
        let (work_tx, work_rx) = mpsc::channel::<(usize, HostLease)>();
        let work_rx = Mutex::new(work_rx);
        let (out_tx, out_rx) = mpsc::channel::<(usize, Vec<f32>)>();
        std::thread::scope(|scope| {
            for _ in 0..READERS.min(n) {
                let (tx, order, next) = (got_tx.clone(), &order, &next);
                scope.spawn(move || {
                    while let Some(&j) = order.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if tx.send((j, acquire(used[j]))).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(got_tx);
            for _ in 0..threads.min(light) {
                let (work_rx, out_tx) = (&work_rx, out_tx.clone());
                scope.spawn(move || loop {
                    let job = work_rx.lock().unwrap_or_else(|p| p.into_inner()).recv();
                    let Ok((j, record)) = job else { break };
                    let out = expert_forward_batch(&record, &gathered[j].1, Some(&gathered[j].2), cfg.swiglu_limit);
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
                    crate::profile::add(crate::profile::Part::ExpertGpu, computing);
                    for (&j, out) in part.iter().zip(got) {
                        done[j] = Some(out);
                    }
                }
            }
            let mut group: Vec<(usize, HostLease)> = Vec::with_capacity(GPU_GROUP);
            let on_gpu = |group: &mut Vec<(usize, HostLease)>, done: &mut Vec<Option<Vec<f32>>>| {
                let Some(gpu) = gpu else { return };
                let jobs: Vec<crate::expert::ExpertJob<'_>> =
                    group.iter().map(|(j, record)| crate::expert::ExpertJob { record, x: &gathered[*j].1, weights: &gathered[*j].2 }).collect();
                let computing = std::time::Instant::now();
                let got = gpu.forward(&jobs, cfg.swiglu_limit);
                crate::profile::add(crate::profile::Part::ExpertGpu, computing);
                for ((j, _), out) in group.drain(..).zip(got) {
                    done[j] = Some(out);
                }
            };
            loop {
                let waiting = std::time::Instant::now();
                let Ok((j, record)) = got_rx.recv() else { break };
                crate::profile::add(crate::profile::Part::ExpertRead, waiting);
                match record {
                    Err(e) => {
                        failed.get_or_insert(e);
                    }
                    Ok(record) if busy[j] => {
                        group.push((j, record));
                        if group.len() == GPU_GROUP {
                            on_gpu(&mut group, &mut done);
                        }
                    }
                    Ok(record) => {
                        let _ = work_tx.send((j, record));
                    }
                }
            }
            if !group.is_empty() {
                on_gpu(&mut group, &mut done);
            }
            drop(work_tx);
            let waiting = std::time::Instant::now();
            for (j, out) in out_rx {
                done[j] = Some(out);
            }
            if light > 0 {
                crate::profile::add(crate::profile::Part::ExpertCpu, waiting);
            }
        });
        if let Some(e) = failed {
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
