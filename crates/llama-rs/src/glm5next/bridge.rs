// VENDORED-LOCAL: whole module. GLM-5.3-Flash real-weights bridge.
//! Loads a released glm5next GGUF into the f32 buffers [`super::forward`] takes,
//! so the host reference can run on real weights.
//!
//! ## Why this is not just "dequantise the model"
//!
//! The routed experts cannot be materialised. 43 MoE layers x 288 experts x 3
//! projections x 2048 x 4096 x 4 bytes is about **1.2 TB** of f32. So this splits
//! the model in two:
//!
//!   * **Everything else** — norms, attention projections, hyper-connection
//!     mixers, the dense FFNs, the shared experts, the embedding and the LM head
//!     — is loaded resident as f32 through `loader::load_tensor_f32`, which
//!     dequantises whatever dtype the GGUF used. That is roughly **34 GB**.
//!   * **The routed experts** are read one at a time, per dispatch, straight out
//!     of the `.gguf` by a ranged read and dequantised into a scratch buffer.
//!     That is [`GgufExperts`], an [`super::forward::ExpertSource`].
//!
//! The expert path needs a **streaming** file (`GgufFile::open_streaming`) so the
//! bodies are served by positioned reads rather than a mapping, and it is
//! split-shard aware: each stacked expert tensor is read through the source for
//! its own shard.
//!
//! ## Cost
//!
//! This is the host reference, so it is slow on purpose: about 17B active
//! parameters per token as scalar f32 dot products. It exists to be *right* —
//! the thing a greedy-token comparison against llama.cpp runs on, and the oracle
//! a device implementation is checked against — not to be fast.

use std::collections::BTreeMap;
use std::path::Path;

use ggml_quants::GgmlType;
use ggml_rs::Tensor;
use gguf::{GgufFile, TensorInfo};

use super::forward::{
    self, AttnW, Bat, ExpertFfn, FfnW, HcW, IndexerW, KdaW, LayerW, Mat, Pair, MlaW, ModelW, MoeW, Shape,
};
use super::{Glm5NextConfig, LayerKind};
use crate::config::ModelConfig;
use crate::loader::TensorIndex;
use crate::{LlamaError, Result};

/// One stacked expert tensor, addressed per expert by a ranged read.
///
/// Shared with [`super::device`], which needs the raw quantised bytes rather than
/// dequantised floats.
pub(crate) struct PartRef {
    shard: usize,
    /// Offset of expert 0 within the shard's data section.
    base: u64,
    /// Bytes per expert slice.
    per: usize,
    dtype: GgmlType,
    /// Floats per expert slice.
    numel: usize,
}

impl PartRef {
    pub(crate) fn per(&self) -> usize {
        self.per
    }
    pub(crate) fn dtype(&self) -> GgmlType {
        self.dtype
    }
    pub(crate) fn numel(&self) -> usize {
        self.numel
    }

    /// Read expert `e`'s packed bytes, without dequantising.
    pub(crate) fn read_raw(&self, file: &GgufFile, e: usize, buf: &mut [u8]) -> Result<()> {
        if buf.len() != self.per {
            return Err(LlamaError::Config(format!(
                "bridge: staging buffer is {} bytes, the slice is {}",
                buf.len(),
                self.per
            )));
        }
        let src = file
            .shard_source_at(self.shard)
            .ok_or_else(|| LlamaError::Config("bridge: shard lost its byte source".into()))?;
        src.read_range(self.base + (e * self.per) as u64, buf)
            .map_err(|err| LlamaError::Config(format!("bridge: expert read failed: {err}")))
    }

    /// Resolve the three stacked expert tensors for each MoE layer, returning
    /// them with the largest per-expert byte count (the staging-buffer size).
    pub(crate) fn collect(
        g: &GgufFile,
        first_moe: usize,
        n_moe: usize,
        n_expert: usize,
    ) -> Result<(Vec<[PartRef; 3]>, usize)> {
        for shard in 0..g.n_shards() {
            if !g.shard_is_source_backed(shard) {
                return Err(LlamaError::Config(format!(
                    "bridge: expert reads need a streaming file (GgufFile::open_streaming); \
                     shard {shard} has no byte source"
                )));
            }
        }
        let mut layers = Vec::with_capacity(n_moe);
        let mut max_per = 0usize;
        for ord in 0..n_moe {
            let il = first_moe + ord;
            let mut parts = Vec::with_capacity(3);
            for part in ["gate", "up", "down"] {
                let name = format!("blk.{il}.ffn_{part}_exps.weight");
                let info = g
                    .tensor_by_name(&name)
                    .ok_or_else(|| LlamaError::MissingTensor(name.clone()))?;
                let r = PartRef::new(g, info, n_expert)?;
                max_per = max_per.max(r.per);
                parts.push(r);
            }
            let mut it = parts.into_iter();
            layers.push([
                it.next().expect("gate"),
                it.next().expect("up"),
                it.next().expect("down"),
            ]);
        }
        Ok((layers, max_per))
    }

    fn new(g: &GgufFile, info: &TensorInfo, n_expert: usize) -> Result<Self> {
        let nbytes = info.nbytes() as usize;
        if n_expert == 0 || nbytes % n_expert != 0 {
            return Err(LlamaError::Config(format!(
                "bridge: {} is {nbytes} bytes, not divisible by {n_expert} experts",
                info.name
            )));
        }
        let numel = info.numel() as usize;
        if numel % n_expert != 0 {
            return Err(LlamaError::Config(format!(
                "bridge: {} has {numel} elements, not divisible by {n_expert} experts",
                info.name
            )));
        }
        Ok(Self {
            shard: g.shard_of(info),
            base: info.offset,
            per: nbytes / n_expert,
            dtype: info.dtype,
            numel: numel / n_expert,
        })
    }
}

/// Routed experts, read one at a time from the `.gguf`.
///
/// Stacked expert tensors are `[n_expert, ...]` with the expert axis outermost,
/// so expert `e`'s bytes are the contiguous range
/// `base + e * per .. base + (e + 1) * per` — one ranged read per projection.
pub struct GgufExperts {
    file: GgufFile,
    /// Per MoE layer, in MoE-ordinal order: gate, up, down.
    layers: Vec<[PartRef; 3]>,
    n_ff_exp: usize,
    n_embd: usize,
    /// Allocated once: the packed bytes of one expert slice, the three
    /// dequantised projections, and the FFN intermediates.
    scratch: std::cell::RefCell<Scratch>,
}

struct Scratch {
    bytes: Vec<u8>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    h: Vec<f32>,
}

impl GgufExperts {
    /// `first_moe` is the block index of MoE layer 0 (`leading_dense_block_count`).
    pub fn new(
        g: &GgufFile,
        first_moe: usize,
        n_moe: usize,
        n_expert: usize,
        n_ff_exp: usize,
        n_embd: usize,
    ) -> Result<Self> {
        let (layers, max_per) = PartRef::collect(g, first_moe, n_moe, n_expert)?;
        let per = n_ff_exp * n_embd;
        Ok(Self {
            file: g.clone(),
            layers,
            n_ff_exp,
            n_embd,
            scratch: std::cell::RefCell::new(Scratch {
                bytes: vec![0u8; max_per],
                gate: vec![0.0; per],
                up: vec![0.0; per],
                down: vec![0.0; per],
                g: vec![0.0; n_ff_exp],
                u: vec![0.0; n_ff_exp],
                h: vec![0.0; n_ff_exp],
            }),
        })
    }

    /// Read and dequantise one expert slice into `out`, using `bytes` as the
    /// staging buffer.
    fn read_part(
        &self,
        p: &PartRef,
        e: usize,
        bytes: &mut [u8],
        out: &mut [f32],
    ) -> Result<()> {
        if out.len() != p.numel() {
            return Err(LlamaError::Config(format!(
                "bridge: expert buffer is {} floats, the slice is {}",
                out.len(),
                p.numel()
            )));
        }
        let staging = &mut bytes[..p.per()];
        p.read_raw(&self.file, e, staging)?;
        ggml_quants::dequantize(p.dtype(), staging, out)?;
        Ok(())
    }
}

impl ExpertFfn for GgufExperts {
    fn apply(
        &self,
        ord: usize,
        e: usize,
        x: &[f32],
        limit: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let parts = self.layers.get(ord).ok_or_else(|| {
            LlamaError::Config(format!(
                "bridge: MoE layer {ord} out of range (have {})",
                self.layers.len()
            ))
        })?;
        if x.len() != self.n_embd || out.len() != self.n_embd {
            return Err(LlamaError::Config(format!(
                "bridge: expert FFN got x {} / out {}, expected n_embd {}",
                x.len(),
                out.len(),
                self.n_embd
            )));
        }
        let sc = &mut *self.scratch.borrow_mut();
        // Split the borrow so the read and the matvec can share the struct.
        let Scratch { bytes, gate, up, down, g, u, h } = sc;
        self.read_part(&parts[0], e, bytes, gate)?;
        self.read_part(&parts[1], e, bytes, up)?;
        self.read_part(&parts[2], e, bytes, down)?;
        let _ = self.n_ff_exp;
        forward::matvec(gate, x, g)?;
        forward::matvec(up, x, u)?;
        forward::swiglu_clamped(g, u, limit, h)?;
        forward::matvec(down, h, out)
    }
}

/// A released glm5next model with its non-expert weights resident as f32.
pub struct HostModel {
    shape: Shape,
    /// Every non-expert tensor, dequantised. Keyed by GGUF name.
    t: BTreeMap<String, Tensor>,
    experts: GgufExperts,
}

impl HostModel {
    /// Open a released model. `path` may name any shard of a split GGUF.
    pub fn open(path: impl AsRef<Path>, max_len: usize) -> Result<Self> {
        let g = GgufFile::open_streaming(path)?;
        Self::from_gguf(&g, max_len)
    }

    pub fn from_gguf(g: &GgufFile, max_len: usize) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(g)?;
        let glm = Glm5NextConfig::from_gguf(g, &cfg)?;
        let idx = TensorIndex::new(g);

        let mut t: BTreeMap<String, Tensor> = BTreeMap::new();
        let mut load = |name: String| -> Result<()> {
            let tensor = idx.take(&name, &[])?;
            t.insert(name, tensor);
            Ok(())
        };

        load("token_embd.weight".to_string())?;
        load("output_norm.weight".to_string())?;
        // A tied head is possible in principle; the released model ships its own.
        if g.tensor_by_name("output.weight").is_some() {
            load("output.weight".to_string())?;
        }

        // Trunk only: the NextN block is a draft head, not part of a decode.
        for il in 0..glm.n_layer {
            for suffix in [
                "attn_norm.weight",
                "ffn_norm.weight",
                "hc_attn_fn.weight",
                "hc_attn_base.weight",
                "hc_attn_scale.weight",
                "hc_ffn_fn.weight",
                "hc_ffn_base.weight",
                "hc_ffn_scale.weight",
            ] {
                load(format!("blk.{il}.{suffix}"))?;
            }
            match glm.layer_kinds[il] {
                LayerKind::Kda => {
                    for suffix in [
                        "attn_q.weight",
                        "attn_k.weight",
                        "attn_v.weight",
                        "attn_output.weight",
                        "ssm_conv1d_q.weight",
                        "ssm_conv1d_k.weight",
                        "ssm_conv1d_v.weight",
                        "ssm_f_a.weight",
                        "ssm_f_b.weight",
                        "ssm_g_a.weight",
                        "ssm_g_b.weight",
                        "ssm_beta.weight",
                        "ssm_a",
                        "ssm_dt.bias",
                        "ssm_norm.weight",
                    ] {
                        load(format!("blk.{il}.{suffix}"))?;
                    }
                }
                LayerKind::Mla => {
                    for suffix in [
                        "attn_q_a.weight",
                        "attn_q_a_norm.weight",
                        "attn_q_b.weight",
                        "attn_kv_a_mqa.weight",
                        "attn_kv_a_norm.weight",
                        "attn_k_b.weight",
                        "attn_v_b.weight",
                        "attn_output.weight",
                        "indexer.attn_k.weight",
                        "indexer.attn_q_b.weight",
                        "indexer.k_norm.weight",
                        "indexer.k_norm.bias",
                        "indexer.proj.weight",
                        "indexer_compressor_gate.weight",
                        "indexer_compressor_ape.weight",
                    ] {
                        load(format!("blk.{il}.{suffix}"))?;
                    }
                }
            }
            if il < glm.n_dense_lead {
                for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    load(format!("blk.{il}.{suffix}"))?;
                }
            } else {
                for suffix in [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                ] {
                    load(format!("blk.{il}.{suffix}"))?;
                }
            }
        }

        // Only the trunk's MoE layers are reachable from a decode, so the expert
        // source covers blocks n_dense_lead..n_layer.
        let n_moe = glm.n_layer - glm.n_dense_lead;
        let experts = GgufExperts::new(
            g,
            glm.n_dense_lead,
            n_moe,
            glm.n_expert,
            glm.n_ff_exp,
            cfg.embedding_dim,
        )?;

        let shape = shape_from(&cfg, &glm, max_len);
        Ok(Self { shape, t, experts })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Resident f32 footprint of the non-expert weights.
    pub fn size_bytes(&self) -> usize {
        self.t.values().map(|x| x.numel() * 4).sum()
    }

    fn g(&self, name: &str) -> &[f32] {
        self.t
            .get(name)
            .unwrap_or_else(|| panic!("bridge: {name} was not loaded"))
            .data()
    }

    /// Borrow the weights as the view [`super::forward`] takes.
    pub fn view(&self) -> ModelW<'_> {
        let sh = &self.shape;
        let mut layers = Vec::with_capacity(sh.n_layer);
        for il in 0..sh.n_layer {
            let b = |s: &str| self.g(&format!("blk.{il}.{s}"));
            let m = |s: &str| Mat::Host(self.g(&format!("blk.{il}.{s}")));
            let attn = match sh.layer_kinds[il] {
                LayerKind::Kda => AttnW::Kda(KdaW {
                    qk: Pair::Split(m("attn_q.weight"), m("attn_k.weight")),
                    v: m("attn_v.weight"),
                    conv_q: b("ssm_conv1d_q.weight"),
                    conv_k: b("ssm_conv1d_k.weight"),
                    conv_v: b("ssm_conv1d_v.weight"),
                    fga: Pair::Split(m("ssm_f_a.weight"), m("ssm_g_a.weight")),
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
                    k_b: Bat::Host(b("attn_k_b.weight")),
                    v_b: Bat::Host(b("attn_v_b.weight")),
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
                    sh_gate_up: Pair::Split(m("ffn_gate_shexp.weight"), m("ffn_up_shexp.weight")),
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
            tok_embd: self.g("token_embd.weight"),
            output_norm: self.g("output_norm.weight"),
            output: Mat::Host(if self.t.contains_key("output.weight") {
                self.g("output.weight")
            } else {
                self.g("token_embd.weight")
            }),
            layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5next::forward;

    /// D: is the faster drive; E: holds an identical copy.
    const RELEASED: &str =
        r"D:\glm5.3_flash\Q4_K_M\GLM-5.3-Flash-Q4_K_M-00001-of-00005.gguf";

    /// How much does a KDA layer actually remember?
    ///
    /// A 192-token generation looped a sentence verbatim, which is what a
    /// recurrent state with ~one token of memory looks like. The decay per step is
    /// `exp(g)` where
    /// `g = gate_lower_bound * sigmoid(-(ssm_a * (f_b(f_a(x)) + dt_bias)))`, so
    /// `g` near the bound (-5) means `exp(-5) = 0.0067` -- near-total forgetting
    /// every token -- and `g` near 0 means full retention.
    ///
    /// This measures it on real weights, which decides whether the repetition is
    /// the recurrence or something else.
    #[test]
    #[ignore = "needs the released model on disk"]
    fn probe_the_kda_decay_on_real_weights() {
        let g = GgufFile::open_streaming(RELEASED).expect("open");
        let cfg = ModelConfig::from_gguf(&g).expect("cfg");
        let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm");
        let idx = TensorIndex::new(&g);

        let n_embd = cfg.embedding_dim;
        let nh = cfg.n_heads;
        let hd = glm.kda_head_dim;
        let di = nh * hd;

        // A plausible post-attn_norm hidden state: a real token embedding, RMS
        // normalised to unit scale, which is what the norm produces.
        let embd = idx.take("token_embd.weight", &[]).expect("embd");
        let tokid = 5000usize;
        let mut x: Vec<f32> = embd.data()[tokid * n_embd..(tokid + 1) * n_embd].to_vec();
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / n_embd as f32 + glm.rms_eps).sqrt();
        for v in x.iter_mut() {
            *v *= inv;
        }

        let mut worst_ret = 1.0f32;
        for il in [0usize, 1, 10, 20, 44] {
            if glm.layer_kinds[il] != LayerKind::Kda {
                continue;
            }
            let f_a = idx.take(&format!("blk.{il}.ssm_f_a.weight"), &[]).expect("f_a");
            let f_b = idx.take(&format!("blk.{il}.ssm_f_b.weight"), &[]).expect("f_b");
            let a = idx.take(&format!("blk.{il}.ssm_a"), &[]).expect("a");
            let dtb = idx.take(&format!("blk.{il}.ssm_dt.bias"), &[]).expect("dt");

            let mut fa = vec![0.0f32; hd];
            forward::matvec(f_a.data(), &x, &mut fa).expect("f_a");
            let mut raw = vec![0.0f32; di];
            forward::matvec(f_b.data(), &fa, &mut raw).expect("f_b");

            let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
            let mut decays = Vec::with_capacity(di);
            for h in 0..nh {
                for i in 0..hd {
                    let k = h * hd + i;
                    let t = (raw[k] + dtb.data()[k]) * a.data()[h];
                    let glog = glm.kda_gate_lower_bound * sig(-t);
                    decays.push(glog.exp());
                }
            }
            decays.sort_by(|p, q| p.partial_cmp(q).unwrap());
            let mean = decays.iter().sum::<f32>() / decays.len() as f32;
            let med = decays[decays.len() / 2];
            let p90 = decays[decays.len() * 9 / 10];
            // Effective memory in tokens: how long until a contribution decays
            // to 1/e.
            let half = if med > 0.0 && med < 1.0 {
                -1.0 / med.ln()
            } else {
                f32::INFINITY
            };
            println!(
                "blk.{il}: ssm_a[0]={:+.4}  decay min {:.4} med {:.4} mean {:.4} p90 {:.4}                   memory ~{:.1} tokens",
                a.data()[0],
                decays[0],
                med,
                mean,
                p90,
                half
            );
            worst_ret = worst_ret.min(med);
        }

        println!();
        println!("gate_lower_bound = {}", glm.kda_gate_lower_bound);
        println!("exp(lower_bound) = {:.5} (total forgetting)", glm.kda_gate_lower_bound.exp());
        println!("median decay across probed layers: {worst_ret:.4}");
        assert!(worst_ret > 0.0, "decay must be positive");
    }

    /// The expert source alone, on real weights: one ranged read per projection
    /// out of a 1.2 TB-if-materialised tensor.
    #[test]
    #[ignore = "needs the released model on disk"]
    fn reads_one_real_expert() {
        let g = GgufFile::open_streaming(RELEASED).expect("open");
        let cfg = ModelConfig::from_gguf(&g).expect("cfg");
        let glm = Glm5NextConfig::from_gguf(&g, &cfg).expect("glm");

        let src = GgufExperts::new(
            &g,
            glm.n_dense_lead,
            4,
            glm.n_expert,
            glm.n_ff_exp,
            cfg.embedding_dim,
        )
        .expect("experts");

        let x: Vec<f32> = (0..cfg.embedding_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 80.0)
            .collect();
        let mut out = vec![0.0f32; cfg.embedding_dim];

        for (ord, e) in [(0usize, 0usize), (0, 287), (3, 100)] {
            src.apply(ord, e, &x, 10.0, &mut out).expect("expert ffn");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "MoE layer {ord} expert {e} produced non-finite output"
            );
            assert!(
                out.iter().any(|&v| v != 0.0),
                "MoE layer {ord} expert {e} produced all zeros"
            );
            let max = out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            assert!(max < 1e3, "MoE layer {ord} expert {e} max |x| = {max}");
        }
        println!(
            "real expert FFNs OK ({} floats per projection)",
            glm.n_ff_exp * cfg.embedding_dim
        );
    }

    /// Does the model actually respond to its input, or is the LM head echoing
    /// the embedding? Feeds a short real prompt and reports the argmax at each
    /// step. If every step predicts its own input token, the trunk is not
    /// contributing and something upstream is wrong.
    #[test]
    #[ignore = "needs the released model on disk; slow"]
    fn responds_to_a_short_prompt() {
        let m = HostModel::open(RELEASED, 512).expect("load");
        let sh = m.shape().clone();
        let w = m.view();
        let mut st = forward::State::new(&sh).expect("state");

        // [gMASK] <sop> then a few ordinary ids.
        let prompt = [154822u32, 154824, 5000, 6000];
        let mut echoes = 0;
        for (i, &tok) in prompt.iter().enumerate() {
            let logits = forward::forward_token(&sh, &w, &mut st, tok).expect("forward");
            let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
            for (j, &v) in logits.iter().enumerate() {
                if v > best {
                    best = v;
                    bi = j;
                }
            }
            let mean = logits.iter().sum::<f32>() / logits.len() as f32;
            println!("step {i}: in {tok} -> argmax {bi} (logit {best:.3}, mean {mean:.3})");
            assert!(logits.iter().all(|x| x.is_finite()));
            if bi as u32 == tok {
                echoes += 1;
            }
        }
        assert_eq!(st.len, prompt.len());
        assert!(
            echoes < prompt.len(),
            "every step predicted its own input token; the trunk is not contributing"
        );
    }

    /// The whole thing: load the released model and generate one real token.
    #[test]
    #[ignore = "needs the released model on disk; loads ~34 GB and is slow"]
    fn generates_a_real_token() {
        let t0 = std::time::Instant::now();
        let m = HostModel::open(RELEASED, 512).expect("load");
        println!(
            "loaded {:.2} GB of non-expert f32 in {:.1}s",
            m.size_bytes() as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        let sh = m.shape().clone();
        assert_eq!(sh.n_layer, 45);
        assert_eq!(sh.n_expert, 288);

        let w = m.view();
        let mut st = forward::State::new(&sh).expect("state");

        // `[gMASK]` is 154822 in this vocabulary.
        let t1 = std::time::Instant::now();
        let logits = forward::forward_token(&sh, &w, &mut st, 154822).expect("forward");
        println!(
            "one token through 45 layers in {:.1}s",
            t1.elapsed().as_secs_f64()
        );

        assert_eq!(logits.len(), sh.n_vocab);
        assert!(logits.iter().all(|x| x.is_finite()), "logits must be finite");
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                bi = i;
            }
        }
        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        println!("argmax {bi} logit {best:.4}, mean {mean:.4}");
        assert!(
            best > mean + 1.0,
            "the argmax should stand out from the mean; {best} vs {mean}"
        );
        assert_eq!(st.len, 1);
    }
}

/// The [`Shape`] a released model implies. Shared by [`HostModel`] and
/// [`super::device::DeviceModel`] so the two cannot describe the same file
/// differently.
pub(crate) fn shape_from(cfg: &ModelConfig, glm: &Glm5NextConfig, max_len: usize) -> Shape {
    Shape {
        n_embd: cfg.embedding_dim,
        n_vocab: cfg.vocab_size,
        n_layer: glm.n_layer,
        n_head: cfg.n_heads,
        kda_head_dim: glm.kda_head_dim,
        d_conv: glm.ssm_conv_kernel,
        q_lora: glm.q_lora_rank,
        kv_lora: glm.kv_lora_rank,
        qk_head: glm.qk_head_dim,
        v_head: glm.v_head_dim,
        d_idx: glm.indexer_head_dim,
        n_ihead: glm.indexer_n_head,
        kpool: glm.indexer_kpool,
        indexer_top_k: glm.indexer_top_k,
        n_expert: glm.n_expert,
        n_expert_used: glm.n_expert_used,
        n_ff_exp: glm.n_ff_exp,
        n_ff_shexp: glm.n_ff_shexp,
        n_ff_dense: cfg.ff_dim,
        n_dense_lead: glm.n_dense_lead,
        layer_kinds: glm.layer_kinds.clone(),
        hc_count: glm.hc_count,
        hc_sinkhorn_iters: glm.hc_sinkhorn_iters,
        hc_eps: glm.hc_eps,
        rms_eps: glm.rms_eps,
        norm_eps: glm.norm_eps,
        kda_gate_lower_bound: glm.kda_gate_lower_bound,
        expert_weights_norm: glm.expert_weights_norm,
        expert_weights_scale: glm.expert_weights_scale,
        swiglu_clamp_exp: glm.swiglu_clamp_exp.clone(),
        swiglu_clamp_shexp: glm.swiglu_clamp_shexp.clone(),
        max_len,
    }
}
