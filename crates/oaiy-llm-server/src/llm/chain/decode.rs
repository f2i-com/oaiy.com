//! a dense model's decode step chained on the backend's device in one submit ([`ggml_rs::chain`]):
//! every layer's norm, q, k and v (and their per-head norms, Qwen3's), RoPE (its sines and cosines made here as the
//! CPU's rope makes them), the K and V stored into a copy of the KV cache kept on the device, attention over it, the
//! output projection, residual, FFN and residual, then the head; only the logits and the step's K and V rows (for the
//! host's cache) come back. The copy is brought up to date first with the rows the host wrote since (a prompt's;
//! [`KvCache::dirty_from`]). On WebGPU a 3B Llama's step, its 113 round trips one, went from 33.6 ms to 16.6.

use std::sync::{Arc, Mutex, OnceLock};

use ggml_rs::{Backend, DeviceVec, QuantizedTensor, Tensor};

use crate::llm::config::ModelConfig;
use crate::llm::kv_cache::KvCache;
use crate::llm::loader::{CommonTensors, FfnPair, Weight};

/// A dense model's tensors as a chained step reads them, and what its layers do that Llama's do not.
pub(crate) struct Dense<'a> {
    pub cfg: &'a ModelConfig,
    pub common: &'a CommonTensors,
    /// Per-head norms of each layer's q and k before RoPE (Qwen3's, Gemma 3's), `[head_dim]` weights each.
    pub qk_norms: Option<Vec<(&'a Tensor, &'a Tensor)>>,
    /// Norms of each layer's attention output and FFN output before their residuals (Gemma 3's).
    pub post_norms: Option<Vec<(&'a Tensor, &'a Tensor)>>,
    /// The FFN's gate through the tanh GELU (Gemma's GeGLU), else SiLU.
    pub gelu: bool,
    /// The embedding times this (Gemma's `sqrt(d)`).
    pub embed_scale: Option<f32>,
    /// Each layer's RoPE base, its per-frequency divisors and its sliding window (Gemma 3's local and global layers);
    /// empty: the config's base for every layer, unscaled, no window.
    pub layers: Vec<(f32, Option<&'a [f32]>, Option<usize>)>,
    /// The logits through `tanh(l / c) * c`.
    pub softcap: Option<f32>,
    /// Qwen3-VL's interleaved M-RoPE: how many rotary pairs follow a row's time, height and width
    /// ([`ModelConfig::mrope_sections`]); None, one position a row (a text row's three are the same anyway).
    pub mrope: Option<[usize; 3]>,
}

impl<'a> Dense<'a> {
    /// A model whose layers are Llama's.
    pub fn plain(cfg: &'a ModelConfig, common: &'a CommonTensors) -> Dense<'a> {
        Dense { cfg, common, qk_norms: None, post_norms: None, gelu: false, embed_scale: None, layers: Vec::new(), softcap: None, mrope: None }
    }

    /// Layer `l`'s RoPE base, divisors and window.
    fn layer(&self, l: usize) -> (f32, Option<&'a [f32]>, Option<usize>) {
        self.layers.get(l).copied().unwrap_or((self.cfg.rope_theta, None, None))
    }

    /// The distinct RoPEs of the layers (base and divisors), and each layer's among them.
    fn ropes(&self) -> (Vec<(f32, Option<&'a [f32]>)>, Vec<usize>) {
        let mut distinct: Vec<(f32, Option<&'a [f32]>)> = Vec::new();
        let which = (0..self.cfg.n_layers)
            .map(|l| {
                let (theta, factors, _) = self.layer(l);
                let same = |(t, f): &(f32, Option<&[f32]>)| t.to_bits() == theta.to_bits() && f.map(|f| f.as_ptr()) == factors.map(|f| f.as_ptr());
                match distinct.iter().position(same) {
                    Some(i) => i,
                    None => {
                        distinct.push((theta, factors));
                        distinct.len() - 1
                    }
                }
            })
            .collect();
        (distinct, which)
    }
}

/// The chained step's state on the device, made at the first step that can use one; None when the backend has no chain
/// or a weight a step reads is not on its device. A model split over two devices ([`ChainDecoder::split`]) has a state
/// on each for its share of the layers, the second's with the head: a step's or a prompt's rows go through the first's
/// layers, their hidden states to the second (a row a step), which runs the rest; each device's copy of the cache holds
/// its own layers' rows. A model too big for one card's memory beside its cache (a 33B's 19 GB and 28 GB of cache at
/// 65,536 positions) so runs on two.
#[derive(Default)]
pub(crate) struct ChainDecoder {
    state: OnceLock<Option<ChainState>>,
    /// The second device and the first of the layers it runs.
    later: OnceLock<(Arc<dyn Backend>, usize)>,
    later_state: OnceLock<Option<ChainState>>,
}

impl std::fmt::Debug for ChainDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let on = match self.state.get() {
            None => "not yet",
            Some(None) => "none",
            Some(Some(_)) => "on the device",
        };
        match self.later.get() {
            Some((_, from)) => write!(f, "ChainDecoder({on}, split at layer {from})"),
            None => write!(f, "ChainDecoder({on})"),
        }
    }
}

/// A chained step's copy of a KV cache on the device, a buffer a layer (row `t`: its K `[n_kv, head_dim]` then its V),
/// `cap` rows of them, and the attention's output and scratch. With `halves` its values are f16, two a word (half the
/// bytes a step's attention reads of a long cache, and of the card's memory it takes): where the device attends such a
/// cache for a step, a prompt's rows and a check's few, and no layer keeps a window (OAIY_KV_F32: f32).
struct ChainKv {
    layers: Vec<DeviceVec>,
    cap: usize,
    out: DeviceVec,
    /// The [`KvCache::id`] the rows are a copy of (0: none).
    owner: u64,
    halves: bool,
}

/// A run's vectors (a prompt's chunk's, a check's), each kept from run to run and made again only where a run needs
/// it longer, lent at the length a run asks for: new memory on a card costs about a third of a second a GiB, and a
/// chunk of 2,048 rows of a 33B's took 1.5 GB of it, 5 s of a 22,492-token prompt's 19.8 (each chunk's its own).
struct RunVecs(Vec<DeviceVec>);

impl RunVecs {
    /// Vector `slot`, `len` long.
    fn get(&mut self, chain: &dyn ggml_rs::DeviceChain, slot: usize, len: usize) -> DeviceVec {
        while self.0.len() <= slot {
            self.0.push(chain.vec(1));
        }
        if self.0[slot].len < len {
            // (the shorter one let go first: its memory the longer one's)
            self.0[slot] = chain.vec(1);
            self.0[slot] = chain.vec(len);
        }
        DeviceVec { len, inner: Arc::clone(&self.0[slot].inner) }
    }
}

/// `values` (pairs) as f16 halves into `words`, two a word, as a cache held as halves keeps them.
fn halves_into(values: &[f32], words: &mut [f32]) {
    for (w, p) in words.iter_mut().zip(values.chunks_exact(2)) {
        *w = f32::from_bits(u32::from(ggml_rs::tensor::f32_to_f16_bits(p[0])) | (u32::from(ggml_rs::tensor::f32_to_f16_bits(p[1])) << 16));
    }
}

/// A cache's rows read back as halves (two a word) as f32 values, for the host's cache, a long run of them on every
/// core (a prompt's chunk of 2,048 rows is 218 million values of a 33B's 52 layers; a step's or a check's few on one,
/// where handing them out cost more than the work).
fn from_halves(words: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    let unpack = |out: &mut [f32], words: &[f32]| {
        for (pair, w) in out.chunks_exact_mut(2).zip(words) {
            let b = w.to_bits();
            pair[0] = ggml_rs::tensor::f16_bits_to_f32(b as u16);
            pair[1] = ggml_rs::tensor::f16_bits_to_f32((b >> 16) as u16);
        }
    };
    let mut out = vec![0f32; 2 * words.len()];
    if words.len() < 1 << 18 {
        unpack(&mut out, words);
    } else {
        out.par_chunks_mut(1 << 16).zip(words.par_chunks(1 << 15)).for_each(|(out, words)| unpack(out, words));
    }
    out
}

struct ChainState {
    /// Its layers, `first..first + count`, and whether it runs the head after them (the last device's state).
    first: usize,
    count: usize,
    head: bool,
    kv: Mutex<ChainKv>,
    /// A prompt's chunk's and a check's vectors, kept from run to run ([`RunVecs`]).
    run_vecs: Mutex<RunVecs>,
    /// A sine-and-cosine table a distinct RoPE of the layers.
    rope_tables: Vec<DeviceVec>,
    /// Gemma 3's norms of a layer's attention and FFN outputs (its layers' each, as the norms below).
    post_norms: Option<Vec<(DeviceVec, DeviceVec)>>,
    attn_norms: Vec<DeviceVec>,
    ffn_norms: Vec<DeviceVec>,
    /// Qwen3's per-head q and k norms, a layer's each.
    qk_norms: Option<Vec<(DeviceVec, DeviceVec)>>,
    output_norm: DeviceVec,
    /// the residual stream, the normed input, q, k, v (and q and k normed), a projection's output, the fused gate-up,
    /// the SwiGLU, the logits
    x: DeviceVec,
    xn: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    proj: DeviceVec,
    /// a projection's output normed (Gemma 3's post norms)
    proj_n: DeviceVec,
    gate_up: DeviceVec,
    act: DeviceVec,
    logits: DeviceVec,
}

/// Room in the device's copy of the cache for `needed` rows of its `layers` layers (grown to a power of two, its rows
/// kept).
fn reserve(chain: &dyn ggml_rs::DeviceChain, g: &mut ChainKv, cfg: &ModelConfig, needed: usize, layers: usize) {
    if g.cap >= needed {
        return;
    }
    let row = 2 * cfg.n_kv_heads * cfg.head_dim;
    let cap = needed.next_power_of_two().max(256);
    let len = if g.halves { cap * row / 2 } else { cap * row };
    g.layers = (0..layers)
        .map(|l| match g.layers.get(l) {
            Some(old) => chain.resize(old, len),
            None => chain.vec(len),
        })
        .collect();
    g.out = chain.vec(chain.attention_out_len(cfg.n_heads, cfg.head_dim, cap));
    g.cap = cap;
}

/// Bring the device's copy of the cache (its layers from `first`) up to `past` rows: the rows the host wrote since (all
/// of them for a cache the copy is not of).
fn sync(chain: &dyn ggml_rs::DeviceChain, g: &mut ChainKv, cfg: &ModelConfig, kv: &KvCache, past: usize, first: usize) {
    let (kvd, row) = (cfg.n_kv_heads * cfg.head_dim, 2 * cfg.n_kv_heads * cfg.head_dim);
    let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
    if from < past {
        for (i, layer) in g.layers.iter().enumerate() {
            let l = first + i;
            // the host's buffers read in place (a copy of one was its whole capacity: 0.3 s for 512 rows)
            let (kh, vh) = (kv.k_buffer(l), kv.v_buffer(l));
            let (kown, vown);
            let kd = if kh.is_device() { kown = kh.to_host(); kown.data() } else { kh.data() };
            let vd = if vh.is_device() { vown = vh.to_host(); vown.data() } else { vh.data() };
            // (each row's K then its V, as halves where the copy is so, made on every core: a conversation brought back
            // whole sends all its rows, 27,000 of them 52 layers of a 33B's)
            use rayon::prelude::*;
            let width = if g.halves { row / 2 } else { row };
            let mut rows = vec![0f32; (past - from) * width];
            rows.par_chunks_mut(width).enumerate().for_each(|(i, out)| {
                let t = from + i;
                let (k, v) = (&kd[t * kvd..(t + 1) * kvd], &vd[t * kvd..(t + 1) * kvd]);
                if g.halves {
                    let (ko, vo) = out.split_at_mut(kvd / 2);
                    halves_into(k, ko);
                    halves_into(v, vo);
                } else {
                    out[..kvd].copy_from_slice(k);
                    out[kvd..].copy_from_slice(v);
                }
            });
            chain.upload_at(layer, from * width, &rows);
        }
    }
    g.owner = kv.id;
}

/// The state of `backend`'s share of `m`'s layers (`layers`, and the head where `head`): None where its chain does not
/// hold one of their weights.
fn make_state(backend: &dyn Backend, m: &Dense<'_>, layers: std::ops::Range<usize>, head: bool) -> Option<ChainState> {
    let chain = backend.chain()?;
    let held = |w: &Weight| match w {
        Weight::Quant(q) if chain.holds(q) => Some(()),
        _ => None,
    };
    let blocks = &m.common.blocks[layers.clone()];
    for b in blocks {
        for w in [&b.attn_q, &b.attn_k, &b.attn_v, &b.attn_output, &b.ffn_down] {
            held(w)?;
        }
        match &b.ffn_pair {
            FfnPair::Fused(w) => held(w)?,
            FfnPair::Split { .. } => return None,
        }
    }
    if head {
        held(&m.common.output)?;
    }
    let cfg = m.cfg;
    if cfg.n_kv_heads == 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
        return None;
    }
    let upload = |t: &Tensor| {
        let t = if t.is_device() { t.to_host() } else { t.clone() };
        let v = chain.vec(t.numel());
        chain.upload(&v, t.data());
        v
    };
    let ff = m.common.blocks[0].ffn_pair.ff();
    let (d, qd, kvd) = (cfg.embedding_dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
    let halves = chain.halves_only(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) && chain.halves_few_rows(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) && (0..cfg.n_layers).all(|l| m.layer(l).2.is_none());
    Some(ChainState {
        first: layers.start,
        count: layers.len(),
        head,
        kv: Mutex::new(ChainKv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0, halves }),
        run_vecs: Mutex::new(RunVecs(Vec::new())),
        rope_tables: m.ropes().0.iter().map(|_| chain.vec(cfg.head_dim)).collect(),
        post_norms: m.post_norms.as_ref().map(|norms| norms[layers.clone()].iter().map(|(a, f)| (upload(a), upload(f))).collect()),
        attn_norms: blocks.iter().map(|b| upload(&b.attn_norm)).collect(),
        ffn_norms: blocks.iter().map(|b| upload(&b.ffn_norm)).collect(),
        qk_norms: m.qk_norms.as_ref().map(|norms| norms[layers.clone()].iter().map(|(q, k)| (upload(q), upload(k))).collect()),
        output_norm: upload(&m.common.output_norm),
        x: chain.vec(d),
        xn: chain.vec(d),
        q: chain.vec(qd),
        k: chain.vec(kvd),
        v: chain.vec(kvd),
        qn: chain.vec(qd),
        kn: chain.vec(kvd),
        proj: chain.vec(d),
        proj_n: chain.vec(d),
        gate_up: chain.vec(2 * ff),
        act: chain.vec(ff),
        logits: chain.vec(cfg.vocab_size),
    })
}

/// Each distinct RoPE's sines and cosines for positions `past..past + rows`, `[rows, head_dim]`, as the CPU's rope
/// makes them.
fn rope_tables(m: &Dense<'_>, past: usize, rows: usize) -> Vec<Vec<f32>> {
    let positions: Vec<[u32; 3]> = (past..past + rows).map(|p| [p as u32; 3]).collect();
    rope_tables_at(m, &positions)
}

/// Which of a row's three positions (time, height, width) rotary pair `j` turns by, under Qwen3-VL's interleaved
/// M-RoPE with `sections` pairs a axis (transformers' `apply_interleaved_mrope`): the height's every third pair from
/// the second while they last, the width's every third from the third, the time's the rest.
pub(crate) fn mrope_axis(j: usize, sections: [usize; 3]) -> usize {
    match j % 3 {
        1 if j < 3 * sections[1] => 1,
        2 if j < 3 * sections[2] => 2,
        _ => 0,
    }
}

/// [`rope_tables`] for rows at `positions` (a row's time, height and width: a picture's rows over its grid, the text
/// after it past its side), each pair turning by the axis [`mrope_axis`] gives it (all by the time's without M-RoPE).
fn rope_tables_at(m: &Dense<'_>, positions: &[[u32; 3]]) -> Vec<Vec<f32>> {
    let hd = m.cfg.head_dim;
    m.ropes()
        .0
        .iter()
        .map(|(theta, factors)| {
            positions
                .iter()
                .flat_map(|at| {
                    (0..hd / 2).flat_map(move |j| {
                        let factor = factors.map(|f| f[j]).unwrap_or(1.0);
                        let pos = at[m.mrope.map_or(0, |s| mrope_axis(j, s))];
                        let (s, c) = (pos as f32 * theta.powf(-2.0 * j as f32 / hd as f32) / factor).sin_cos();
                        [s, c]
                    })
                })
                .collect()
        })
        .collect()
}

fn quant(w: &Weight) -> &QuantizedTensor {
    match w {
        Weight::Quant(q) => q,
        _ => unreachable!("the chain state checked every weight"),
    }
}

impl ChainDecoder {
    /// The layers from `from` on, and the head, run on `other` (whose chain holds their weights): asked before the first
    /// step, else refused (false).
    pub(crate) fn split(&self, other: Arc<dyn Backend>, from: usize) -> bool {
        self.state.get().is_none() && self.later.set((other, from)).is_ok()
    }

    /// Whether its layers run over two devices ([`Self::split`]).
    pub(crate) fn is_split(&self) -> bool {
        self.later.get().is_some()
    }

    /// Each device's part in the order the layers go through them: its backend and its state there.
    fn parts<'s>(&'s self, backend: &'s dyn Backend, m: &Dense<'_>) -> Option<Vec<(&'s dyn Backend, &'s ChainState)>> {
        let to = self.later.get().map_or(m.cfg.n_layers, |(_, from)| *from);
        let first = self.state.get_or_init(|| make_state(backend, m, 0..to, self.later.get().is_none())).as_ref()?;
        let mut parts = vec![(backend, first)];
        if let Some((other, from)) = self.later.get() {
            let st = self.later_state.get_or_init(|| make_state(other.as_ref(), m, *from..m.cfg.n_layers, true)).as_ref()?;
            parts.push((other.as_ref(), st));
        }
        Some(parts)
    }

    /// A prompt's chunk of `tokens`, chained in one submit, if the backend can: the last token's logits `[1, vocab]`;
    /// None leaves it to the model's own path. Every layer on the device as a decode step's is, its rows at once: RoPE
    /// from a table of the chunk's positions, the K and V stored into the device's copy of the cache, the causal
    /// attention over it (Gemma 3's local layers within their window); the logits and the chunk's K and V rows (for
    /// the host's cache) come back. Op by op, every projection's input went up and its output came back: of a 3B
    /// Llama's 2,000-token prompt (8.4 s), 3 s on the host making and reading those; a layer a submit with the
    /// attention on the host, 3.8 s, 2.4 of it the attention.
    pub(crate) fn prompt(&self, backend: &dyn Backend, m: &Dense<'_>, tokens: &[u32], kv: &mut KvCache) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || tokens.len() < 2 {
            return None;
        }
        let emb = backend.embed_lookup(&m.common.tok_embd, tokens, m.cfg.embedding_dim);
        let mut emb = if emb.is_device() { emb.to_host() } else { emb };
        if let Some(s) = m.embed_scale {
            emb.data_mut().iter_mut().for_each(|e| *e *= s);
        }
        self.prompt_embeds(backend, m, emb.data(), tokens.len(), kv, None, &[])
    }

    /// [`Self::prompt`] of `t` rows given as their embeddings (`[t, d]`: a picture's tokens among a prompt's), at
    /// `positions` (each row's time, height and width; None: the rows after what `kv` holds), with `deepstack[i]`
    /// (`[t, d]`, zero where no picture is) added to the states after layer `i`: Qwen3-VL's DeepStack. A split model's
    /// chunk goes through each device's layers in turn, its hidden states read back and given to the next.
    pub(crate) fn prompt_embeds(&self, backend: &dyn Backend, m: &Dense<'_>, emb: &[f32], t: usize, kv: &mut KvCache, positions: Option<&[[u32; 3]]>, deepstack: &[Vec<f32>]) -> Option<Tensor> {
        let out = self.rows(backend, m, emb, t, kv, positions, deepstack, false)?;
        Some(Tensor::from_vec(out, vec![1, m.cfg.vocab_size]))
    }

    /// A check of `tokens` (a token sampled and drafts after it) after what `kv` holds, chained as a prompt's chunk is:
    /// every row's logits (`[1, vocab]` each), the rows kept in `kv` (as a prompt's are) until the caller takes back
    /// those past the drafts taken (`kv.len` less them). None leaves the tokens to the model's own path.
    pub(crate) fn check(&self, backend: &dyn Backend, m: &Dense<'_>, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<Tensor>> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || tokens.is_empty() {
            return None;
        }
        let emb = backend.embed_lookup(&m.common.tok_embd, tokens, m.cfg.embedding_dim);
        let mut emb = if emb.is_device() { emb.to_host() } else { emb };
        if let Some(s) = m.embed_scale {
            emb.data_mut().iter_mut().for_each(|e| *e *= s);
        }
        let out = self.rows(backend, m, emb.data(), tokens.len(), kv, None, &[], true)?;
        Some(out.chunks_exact(m.cfg.vocab_size).map(|row| Tensor::from_vec(row.to_vec(), vec![1, m.cfg.vocab_size])).collect())
    }

    /// [`Self::prompt_embeds`]'s run of `t` rows: the last row's logits, or with `every` every row's (a check's).
    #[allow(clippy::too_many_arguments)]
    fn rows(&self, backend: &dyn Backend, m: &Dense<'_>, emb: &[f32], t: usize, kv: &mut KvCache, positions: Option<&[[u32; 3]]>, deepstack: &[Vec<f32>], every: bool) -> Option<Vec<f32>> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || t == 0 || emb.len() != t * m.cfg.embedding_dim {
            return None;
        }
        if positions.is_some_and(|p| p.len() != t) || deepstack.len() > m.cfg.n_layers || deepstack.iter().any(|d| d.len() != emb.len()) {
            return None;
        }
        let parts = self.parts(backend, m)?;
        if parts.len() > 1 && !deepstack.is_empty() {
            return None;
        }
        let cfg = m.cfg;
        let past = kv.len;
        let tables = match positions {
            Some(at) => rope_tables_at(m, at),
            None => rope_tables(m, past, t),
        };
        // (the rows a part takes: the embeddings, then the part before's hidden states)
        let mut rows_in: Vec<f32> = emb.to_vec();
        let (mut out, mut kv_rows) = (Vec::new(), Vec::with_capacity(cfg.n_layers));
        for &(part, st) in &parts {
            let chain = part.chain()?;
            {
                let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
                reserve(chain, &mut g, cfg, past + t, st.count);
                sync(chain, &mut g, cfg, kv, past, st.first);
            }
            let (rec, _, halves) = record_part(chain, m, st, t, past, &tables, deepstack, every, Some(&rows_in));
            let mut got = rec.finish().into_iter();
            let first = got.next().expect("the logits or the hidden states");
            if st.head {
                out = first;
            } else {
                rows_in = first;
            }
            kv_rows.extend(got.map(|r| if halves { from_halves(&r) } else { r }));
        }
        // the chunk's K and V rows into the host's cache too, which the copy already holds
        kv.append_rows_all(backend, &kv_rows, t);
        kv.commit(t);
        kv.dirty_from = usize::MAX;
        if let Some(c) = m.softcap {
            out.iter_mut().for_each(|v| *v = (*v * (1.0 / c)).tanh() * c);
        }
        Some(out)
    }

    /// A prompt's `chunks` (one after another after what `kv` holds) through a model split over two devices, the
    /// second's part of a chunk running as the first runs its part of the next: each device's own work in its turn,
    /// the hidden states between them, the cache's rows into the host's a chunk at a time (`done(i)` once chunk `i`'s
    /// are). The last chunk's logits `[1, vocab]`; None (nothing run) where the model is on one device, or there is one
    /// chunk, and each goes through [`Self::prompt`]. A chunk at a time, each device waited for the other half the time.
    pub(crate) fn prompt_chunks(&self, backend: &dyn Backend, m: &Dense<'_>, chunks: &[&[u32]], kv: &mut KvCache, done: &mut dyn FnMut(usize)) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || std::env::var_os("OAIY_NO_PROMPT_OVERLAP").is_some() || chunks.len() < 2 || chunks.iter().any(|c| c.is_empty()) {
            return None;
        }
        let parts = self.parts(backend, m)?;
        let &[(da, sa), (db, sb)] = parts.as_slice() else { return None };
        let (ca, cb) = (da.chain()?, db.chain()?);
        let cfg = m.cfg;
        let past0 = kv.len;
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        // room for every chunk's rows first, so no copy of the cache grows under a run going on
        for (chain, st) in [(ca, sa), (cb, sb)] {
            let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
            reserve(chain, &mut g, cfg, past0 + total, st.count);
            sync(chain, &mut g, cfg, kv, past0, st.first);
        }
        let pasts: Vec<usize> = chunks.iter().scan(past0, |at, c| { let p = *at; *at += c.len(); Some(p) }).collect();
        let embed = |c: &[u32]| {
            let emb = backend.embed_lookup(&m.common.tok_embd, c, cfg.embedding_dim);
            let mut emb = if emb.is_device() { emb.to_host() } else { emb };
            if let Some(s) = m.embed_scale {
                emb.data_mut().iter_mut().for_each(|e| *e *= s);
            }
            emb
        };
        let first = embed(chunks[0]);
        let (rec, _, halves) = record_part(ca, m, sa, chunks[0].len(), pasts[0], &rope_tables(m, pasts[0], chunks[0].len()), &[], false, Some(first.data()));
        let mut run_a = Some((rec.submit(), halves));
        let mut run_b: Option<(Box<dyn ggml_rs::chain::ChainSubmitted + '_>, usize, bool)> = None;
        let mut a_rows: Vec<Vec<Vec<f32>>> = Vec::with_capacity(chunks.len());
        let mut logits = Vec::new();
        let mut append = |kv: &mut KvCache, j: usize, mut got: std::vec::IntoIter<Vec<f32>>, halves: bool, a_rows: &mut Vec<Vec<Vec<f32>>>, logits: &mut Vec<f32>| {
            *logits = got.next().expect("the logits");
            let mut rows = std::mem::take(&mut a_rows[j]);
            rows.extend(got.map(|r| if halves { from_halves(&r) } else { r }));
            kv.append_rows_all(backend, &rows, chunks[j].len());
            kv.commit(chunks[j].len());
            done(j);
        };
        for c in 0..chunks.len() {
            let t = chunks[c].len();
            // the second part of this chunk recorded and held as the first device runs its part
            let (rec_b, x_b, halves_b) = record_part(cb, m, sb, t, pasts[c], &rope_tables(m, pasts[c], t), &[], false, None);
            let (run, halves_a) = run_a.take().expect("the first part of every chunk runs");
            let mut got = run.wait().into_iter();
            let hidden = got.next().expect("the hidden states");
            a_rows.push(got.map(|r| if halves_a { from_halves(&r) } else { r }).collect());
            cb.upload(&x_b, &hidden);
            // the first part of the next chunk at once, beside this one's second
            if c + 1 < chunks.len() {
                let next = embed(chunks[c + 1]);
                let (rec, _, halves) = record_part(ca, m, sa, chunks[c + 1].len(), pasts[c + 1], &rope_tables(m, pasts[c + 1], chunks[c + 1].len()), &[], false, Some(next.data()));
                run_a = Some((rec.submit(), halves));
            }
            if let Some((run, j, halves)) = run_b.take() {
                append(kv, j, run.wait().into_iter(), halves, &mut a_rows, &mut logits);
            }
            run_b = Some((rec_b.submit(), c, halves_b));
        }
        let (run, j, halves) = run_b.take().expect("the last chunk's second part");
        append(kv, j, run.wait().into_iter(), halves, &mut a_rows, &mut logits);
        kv.dirty_from = usize::MAX;
        if let Some(c) = m.softcap {
            logits.iter_mut().for_each(|v| *v = (*v * (1.0 / c)).tanh() * c);
        }
        Some(Tensor::from_vec(logits, vec![1, cfg.vocab_size]))
    }
}

/// Part `st`'s run of `t` rows at `past` recorded on `chain`: its vectors (kept from run to run, [`RunVecs`]), the
/// rows' RoPE tables (`tables`, each distinct RoPE's), each of its layers, then the head where it has it (the last
/// row's logits, or with `every` every row's) else its rows' hidden states read, and its layers' cache rows read. The
/// rows it takes (`input`) written now; None: the recording held (nothing of it submitted) for the caller to write
/// them into the vector returned before it submits. And whether its cache is held as halves (its rows read so).
#[allow(clippy::too_many_arguments)]
fn record_part<'c>(chain: &'c dyn ggml_rs::DeviceChain, m: &Dense<'_>, st: &ChainState, t: usize, past: usize, tables: &[Vec<f32>], deepstack: &[Vec<f32>], every: bool, input: Option<&[f32]>) -> (Box<dyn ggml_rs::ChainRecorder + 'c>, DeviceVec, bool) {
    {
        let cfg = m.cfg;
        let (n_h, n_kv, hd, d) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.embedding_dim);
        let (qd, kvd, row) = (n_h * hd, n_kv * hd, 2 * n_kv * hd);
        let ff = m.common.blocks[0].ffn_pair.ff();
        let scale = 1.0 / (hd as f32).sqrt();
        let neox = matches!(cfg.arch.rope_type(), ggml_rs::RopeType::NeoX);
        let rope_of = m.ropes().1;
        {
            let g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
            let tables: Vec<DeviceVec> = tables
                .iter()
                .map(|values| {
                    let table = chain.vec(values.len());
                    chain.upload(&table, values);
                    table
                })
                .collect();
            let mut vecs = st.run_vecs.lock().unwrap_or_else(|p| p.into_inner());
            let lens = [t * d, t * d, t * qd, t * kvd, t * kvd, t * qd, t * kvd, t * d, t * d, t * 2 * ff, t * ff, chain.attention_rows_out_len(t, n_h, hd, past + t), d, if every { t } else { 1 } * cfg.vocab_size];
            let [x, xn, q, k, v, qn, kn, proj, proj_n, gate_up, act, attn, last, logits]: [DeviceVec; 14] = std::array::from_fn(|i| vecs.get(chain, i, lens[i]));
            if let Some(rows) = input {
                chain.upload(&x, rows);
            }
            let deepstack: Vec<DeviceVec> = deepstack
                .iter()
                .map(|rows| {
                    let v = chain.vec(rows.len());
                    chain.upload(&v, rows);
                    v
                })
                .collect();
            let mut rec = chain.begin();
            if input.is_none() {
                rec.hold();
            }
            // the chunk's vectors are its own: no bind groups kept to hold their memory after it
            rec.keep_groups(false);
            for l in st.first..st.first + st.count {
                let (b, i) = (&m.common.blocks[l], l - st.first);
                rec.rmsnorm_rows(&x, &st.attn_norms[i], &xn, t, cfg.rms_eps);
                rec.matmul_rows(quant(&b.attn_q), &xn, &q, t);
                rec.matmul_rows(quant(&b.attn_k), &xn, &k, t);
                rec.matmul_rows(quant(&b.attn_v), &xn, &v, t);
                let (qq, kk) = match &st.qk_norms {
                    Some(norms) => {
                        rec.rmsnorm_rows(&q, &norms[i].0, &qn, t * n_h, cfg.rms_eps);
                        rec.rmsnorm_rows(&k, &norms[i].1, &kn, t * n_kv, cfg.rms_eps);
                        (&qn, &kn)
                    }
                    None => (&q, &k),
                };
                let table = &tables[rope_of[l]];
                rec.rope_rows(qq, t, n_h, hd, table, neox);
                rec.rope_rows(kk, t, n_kv, hd, table, neox);
                if g.halves {
                    rec.store_rows_halved(kk, &g.layers[i], t, kvd, past, row, 0);
                    rec.store_rows_halved(&v, &g.layers[i], t, kvd, past, row, kvd);
                    // (on the tensor cores from 16 rows, a check's few a KV head's query heads together)
                    let attended = rec.attention_rows_halved(qq, &g.layers[i], &attn, t, n_h, n_kv, hd, past, scale) || rec.attention_rows_halved_grouped(qq, &g.layers[i], &attn, t, n_h, n_kv, hd, past, scale);
                    assert!(attended, "chain: {t} rows' attention over a cache held as halves");
                } else {
                    rec.store_rows(kk, &g.layers[i], t, kvd, past, row, 0);
                    rec.store_rows(&v, &g.layers[i], t, kvd, past, row, kvd);
                    rec.attention_rows(qq, &g.layers[i], &attn, t, n_h, n_kv, hd, past, m.layer(l).2, scale);
                }
                rec.matmul_rows(quant(&b.attn_output), &attn, &proj, t);
                match &st.post_norms {
                    Some(norms) => {
                        rec.rmsnorm_rows(&proj, &norms[i].0, &proj_n, t, cfg.rms_eps);
                        rec.add(&x, &proj_n);
                    }
                    None => rec.add(&x, &proj),
                }
                rec.rmsnorm_rows(&x, &st.ffn_norms[i], &xn, t, cfg.rms_eps);
                let FfnPair::Fused(gu) = &b.ffn_pair else { unreachable!("the chain state checked the pair") };
                rec.matmul_rows(quant(gu), &xn, &gate_up, t);
                if m.gelu {
                    rec.gelu_mul_split_rows(&gate_up, &act, t);
                } else {
                    rec.silu_mul_split_rows(&gate_up, &act, t);
                }
                rec.matmul_rows(quant(&b.ffn_down), &act, &proj, t);
                match &st.post_norms {
                    Some(norms) => {
                        rec.rmsnorm_rows(&proj, &norms[i].1, &proj_n, t, cfg.rms_eps);
                        rec.add(&x, &proj_n);
                    }
                    None => rec.add(&x, &proj),
                }
                // (Qwen3-VL: a DeepStack merger's tokens at the picture's rows, after the layer)
                if let Some(ds) = deepstack.get(l) {
                    rec.add(&x, ds);
                }
            }
            if st.head {
                rec.rmsnorm_rows(&x, &st.output_norm, &xn, t, cfg.rms_eps);
                if every {
                    rec.matmul_rows(quant(&m.common.output), &xn, &logits, t);
                } else {
                    // the head of the last token's row only
                    rec.copy(&xn, (t - 1) * d, &last, 0, d);
                    rec.matmul(quant(&m.common.output), &last, &logits);
                }
                rec.read(&logits);
            } else {
                rec.read(&x);
            }
            for layer in g.layers.iter() {
                if g.halves {
                    rec.read_range(layer, past * row / 2, t * row / 2);
                } else {
                    rec.read_range(layer, past * row, t * row);
                }
            }
            (rec, x, g.halves)
        }
    }
}

impl ChainDecoder {
    /// One decode step of `token`, chained, if the backend can: the logits `[1, vocab]`; None leaves the step to the
    /// model's own path.
    pub(crate) fn step(&self, backend: &dyn Backend, m: &Dense<'_>, token: u32, kv: &mut KvCache) -> Option<Tensor> {
        self.step_at(backend, m, token, kv, None)
    }

    /// [`Self::step`] with the token at `position` (its time, height and width: a reply after a picture, which is
    /// shorter in positions than in rows); None: at the row it takes. A split model's step is a submit a device, its
    /// hidden state (one row) read back and given to the next.
    pub(crate) fn step_at(&self, backend: &dyn Backend, m: &Dense<'_>, token: u32, kv: &mut KvCache, position: Option<[u32; 3]>) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() {
            return None;
        }
        let parts = self.parts(backend, m)?;
        let cfg = m.cfg;
        let (n_h, n_kv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let (kvd, row) = (n_kv * hd, 2 * n_kv * hd);
        let past = kv.len;
        let scale = 1.0 / (hd as f32).sqrt();
        // each RoPE's sines and cosines at this position
        let rope_of = m.ropes().1;
        let rope_values = rope_tables_at(m, &[position.unwrap_or([past as u32; 3])]);
        let neox = matches!(cfg.arch.rope_type(), ggml_rs::RopeType::NeoX);
        let emb = backend.embed_lookup(&m.common.tok_embd, &[token], cfg.embedding_dim);
        let mut emb = if emb.is_device() { emb.to_host() } else { emb };
        if let Some(s) = m.embed_scale {
            emb.data_mut().iter_mut().for_each(|v| *v *= s);
        }
        // (the row a part takes: the embedding, then the part before's hidden state)
        let mut row_in: Vec<f32> = emb.data().to_vec();
        let (mut logits, mut kv_rows) = (Vec::new(), Vec::with_capacity(cfg.n_layers));
        // A part after the first is recorded while the one before runs, held (nothing of it submitted) until that one's
        // hidden state is read and given to it: its recording, some 1 ms of a step, beside the other device's work.
        let mut before: Option<(Box<dyn ggml_rs::chain::ChainSubmitted + '_>, bool, bool)> = None;
        for &(part, st) in &parts {
            let chain = part.chain()?;
            let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
            reserve(chain, &mut g, cfg, past + 1, st.count);
            sync(chain, &mut g, cfg, kv, past, st.first);
            for (values, table) in rope_values.iter().zip(&st.rope_tables) {
                chain.upload(table, values);
            }
            let mut rec = chain.begin();
            if before.is_some() {
                rec.hold();
            } else {
                chain.upload(&st.x, &row_in);
            }
            for l in st.first..st.first + st.count {
                let (b, i) = (&m.common.blocks[l], l - st.first);
                rec.rmsnorm(&st.x, &st.attn_norms[i], &st.xn, cfg.rms_eps);
                rec.matmul(quant(&b.attn_q), &st.xn, &st.q);
                rec.matmul(quant(&b.attn_k), &st.xn, &st.k);
                rec.matmul(quant(&b.attn_v), &st.xn, &st.v);
                let (q, k) = match &st.qk_norms {
                    Some(norms) => {
                        rec.rmsnorm_rows(&st.q, &norms[i].0, &st.qn, n_h, cfg.rms_eps);
                        rec.rmsnorm_rows(&st.k, &norms[i].1, &st.kn, n_kv, cfg.rms_eps);
                        (&st.qn, &st.kn)
                    }
                    None => (&st.q, &st.k),
                };
                let table = &st.rope_tables[rope_of[l]];
                rec.rope(q, n_h, hd, table, neox);
                rec.rope(k, n_kv, hd, table, neox);
                // a sliding window sees the last `w` positions, as the CPU's attention does
                let lo = m.layer(l).2.map_or(0, |w| (past + 1).saturating_sub(w));
                if g.halves {
                    rec.store_rows_halved(k, &g.layers[i], 1, kvd, past, row, 0);
                    rec.store_rows_halved(&st.v, &g.layers[i], 1, kvd, past, row, kvd);
                    rec.attention_halved(q, &g.layers[i], &g.out, n_h, n_kv, hd, lo, past + 1, g.cap, scale);
                } else {
                    rec.store(k, &g.layers[i], past * row);
                    rec.store(&st.v, &g.layers[i], past * row + kvd);
                    rec.attention(q, &g.layers[i], &g.out, n_h, n_kv, hd, lo, past + 1, g.cap, scale);
                }
                rec.matmul(quant(&b.attn_output), &g.out, &st.proj);
                match &st.post_norms {
                    Some(norms) => {
                        rec.rmsnorm(&st.proj, &norms[i].0, &st.proj_n, cfg.rms_eps);
                        rec.add(&st.x, &st.proj_n);
                    }
                    None => rec.add(&st.x, &st.proj),
                }
                rec.rmsnorm(&st.x, &st.ffn_norms[i], &st.xn, cfg.rms_eps);
                let FfnPair::Fused(gu) = &b.ffn_pair else { unreachable!("the chain state checked the pair") };
                rec.matmul(quant(gu), &st.xn, &st.gate_up);
                if m.gelu {
                    rec.gelu_mul_split(&st.gate_up, &st.act);
                } else {
                    rec.silu_mul_split(&st.gate_up, &st.act);
                }
                rec.matmul(quant(&b.ffn_down), &st.act, &st.proj);
                match &st.post_norms {
                    Some(norms) => {
                        rec.rmsnorm(&st.proj, &norms[i].1, &st.proj_n, cfg.rms_eps);
                        rec.add(&st.x, &st.proj_n);
                    }
                    None => rec.add(&st.x, &st.proj),
                }
            }
            if st.head {
                rec.rmsnorm(&st.x, &st.output_norm, &st.xn, cfg.rms_eps);
                rec.matmul(quant(&m.common.output), &st.xn, &st.logits);
                rec.read(&st.logits);
            } else {
                rec.read(&st.x);
            }
            for layer in g.layers.iter() {
                if g.halves {
                    rec.read_range(layer, past * row / 2, row / 2);
                } else {
                    rec.read_range(layer, past * row, row);
                }
            }
            // the part before's reads: its hidden state this one's input, written before this one goes
            if let Some((run, head, halves)) = before.take() {
                let mut got = run.wait().into_iter();
                let first = got.next().expect("the logits or the hidden state");
                if head {
                    logits = first;
                } else {
                    row_in = first;
                }
                kv_rows.extend(got.map(|r| if halves { from_halves(&r) } else { r }));
                chain.upload(&st.x, &row_in);
            }
            before = Some((rec.submit(), st.head, g.halves));
        }
        if let Some((run, _, halves)) = before.take() {
            let mut got = run.wait().into_iter();
            logits = got.next().expect("the logits");
            kv_rows.extend(got.map(|r| if halves { from_halves(&r) } else { r }));
        }
        if let Some(c) = m.softcap {
            logits.iter_mut().for_each(|v| *v = (*v * (1.0 / c)).tanh() * c);
        }
        // the step's K and V rows into the host's cache too, which the copy already holds
        for (l, kvrow) in kv_rows.into_iter().enumerate() {
            let k = Tensor::from_vec(kvrow[..kvd].to_vec(), vec![1, n_kv, hd]);
            let v = Tensor::from_vec(kvrow[kvd..].to_vec(), vec![1, n_kv, hd]);
            kv.append(backend, l, &k, &v);
        }
        kv.commit(1);
        kv.dirty_from = usize::MAX;
        Some(Tensor::from_vec(logits, vec![1, cfg.vocab_size]))
    }
}

#[cfg(test)]
mod tests {
    use super::mrope_axis;

    /// Qwen3-VL's interleaved M-RoPE (sections 24, 20, 20 of a 128-wide head's 64 pairs) as transformers'
    /// `apply_interleaved_mrope` assigns pairs: the height's every third from the second below 60, the width's every
    /// third from the third below 60, the time's the rest; Qwen3.5's (11, 11, 10 of 32) the same way.
    #[test]
    fn each_rotary_pair_turns_by_its_axis() {
        let axes: Vec<usize> = (0..64).map(|j| mrope_axis(j, [24, 20, 20])).collect();
        assert_eq!(&axes[..7], &[0, 1, 2, 0, 1, 2, 0]);
        assert_eq!((axes[57], axes[58], axes[59], axes[60], axes[61], axes[62], axes[63]), (0, 1, 2, 0, 0, 0, 0));
        assert_eq!((axes.iter().filter(|&&a| a == 1).count(), axes.iter().filter(|&&a| a == 2).count()), (20, 20));
        let qwen35: Vec<usize> = (0..32).map(|j| mrope_axis(j, [11, 11, 10])).collect();
        assert_eq!((qwen35[31], qwen35[29], qwen35[30]), (1, 2, 0));
        assert_eq!((qwen35.iter().filter(|&&a| a == 1).count(), qwen35.iter().filter(|&&a| a == 2).count()), (11, 10));
    }
}
