//! The model's shape and a sequence's state (each layer's recurrent state or caches), with the snapshot a state
//! is saved to and restored from.

use super::*;

/// Geometry the loop needs, in host terms. A subset of [`super::super::Glm5NextConfig`]
/// plus the dims it derives, so this module can be exercised at synthetic sizes.
#[derive(Debug, Clone)]
pub struct Shape {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub kda_head_dim: usize,
    pub d_conv: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_head: usize,
    pub v_head: usize,
    pub d_idx: usize,
    pub n_ihead: usize,
    pub kpool: usize,
    pub indexer_top_k: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub n_ff_dense: usize,
    pub n_dense_lead: usize,
    /// `n_layer_all` entries; only the first `n_layer` are run.
    pub layer_kinds: Vec<LayerKind>,
    pub hc_count: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub rms_eps: f32,
    pub norm_eps: f32,
    pub kda_gate_lower_bound: f32,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    pub max_len: usize,
}

impl Shape {
    pub fn d_inner(&self) -> usize {
        self.n_head * self.kda_head_dim
    }
    /// Ordinal of a block among the MLA layers, for the per-layer caches.
    pub(super) fn mla_ordinal(&self, il: usize) -> usize {
        self.layer_kinds[..il]
            .iter()
            .filter(|k| **k == LayerKind::Mla)
            .count()
    }
    /// Ordinal of a block among the KDA layers.
    pub(super) fn kda_ordinal(&self, il: usize) -> usize {
        self.layer_kinds[..il]
            .iter()
            .filter(|k| **k == LayerKind::Kda)
            .count()
    }
}

/// Everything that persists between tokens.
pub struct State {
    pub len: usize,
    /// Per KDA layer, `[n_head * head_dim * head_dim]`. Used when the state is
    /// on the host; empty when [`Self::kda_dev`] holds it instead.
    pub(super) kda: Vec<Vec<f32>>,
    /// VENDORED-LOCAL: GLM-5.3-Flash. The same state, resident on a backend.
    ///
    /// 34 KDA layers x 4.2 MB is 143 MB that the recurrence touches twice per
    /// token. Keeping it here is what makes a kernel worth having: a version that
    /// uploaded and downloaded it each layer would move 285 MB a token, which at
    /// the ~11 GB/s this machine's x4/x8 links manage is slower than the scalar
    /// loop it replaces.
    pub(super) kda_dev: Option<(Vec<Tensor>, Arc<dyn Backend>)>,
    /// Per KDA layer, `[(d_conv - 1) * 3 * d_inner]` — the last `d_conv - 1`
    /// **pre-conv** `q‖k‖v` vectors, which is what the reference's conv state
    /// holds so a rollback restores one block.
    pub(super) conv: Vec<Vec<f32>>,
    /// Per MLA layer, `[max_len * kv_lora]` — the cached latent, serving as both
    /// K and V.
    pub(super) latents: Vec<Vec<f32>>,
    /// VENDORED-LOCAL: GLM-5.3-Flash. The same cache, mirrored on a backend.
    ///
    /// `[max_len, 1, kv_lora]` per MLA layer. The attention over it is O(len) work a
    /// token -- n_head dot products kv_lora wide against every cached row, twice --
    /// which made a prompt O(n^2) and, on the host, 124 ms a token at position 1740
    /// while both cards sat idle. Doing it on the card means the cache has to live
    /// there: uploading `len * kv_lora` a token instead would be the same quadratic
    /// cost over PCIe.
    ///
    /// A mirror, not a move: `latents` stays authoritative, so snapshots and the
    /// host oracle are unchanged, and a row costs one extra device-side copy of
    /// `kv_lora` floats when it is written.
    pub(super) latents_dev: Option<(Vec<Tensor>, Arc<dyn Backend>)>,
    /// Indexer key / gate / pooled, per MLA layer.
    pub(super) kpool: kpool::KpoolCache,
    pub(super) max_len: usize,
}

impl State {
    pub fn new(sh: &Shape) -> Result<Self> {
        let n_kda = sh.layer_kinds[..sh.n_layer]
            .iter()
            .filter(|k| **k == LayerKind::Kda)
            .count();
        let n_mla = sh.n_layer - n_kda;
        let hd = sh.kda_head_dim;
        Ok(Self {
            len: 0,
            kda: vec![vec![0.0f32; sh.n_head * hd * hd]; n_kda],
            kda_dev: None,
            conv: vec![vec![0.0f32; (sh.d_conv - 1) * 3 * sh.d_inner()]; n_kda],
            latents: vec![vec![0.0f32; sh.max_len * sh.kv_lora]; n_mla],
            latents_dev: None,
            kpool: kpool::KpoolCache::new(n_mla, sh.max_len, sh.d_idx)?,
            max_len: sh.max_len,
        })
    }

    /// As [`Self::new`], with the KDA recurrent state resident on `backend`.
    ///
    /// Everything else stays on the host: the conv ring is small, and the latents
    /// and indexer caches are read by host code that has not moved yet.
    pub fn new_on(sh: &Shape, backend: Arc<dyn Backend>) -> Result<Self> {
        let mut st = Self::new(sh)?;
        let hd = sh.kda_head_dim;
        let n = sh.n_head * hd * hd;
        let dev = st
            .kda
            .iter()
            .map(|_| backend.to_device(Tensor::from_vec(vec![0.0f32; n], vec![sh.n_head, hd, hd])))
            .collect();
        // The latent cache is mirrored too, `[max_len, 1, kv_lora]` a layer: one KV
        // head, which is what absorbed MLA is, and what lets `bmm_qkt`/`bmm_av`
        // serve it without broadcasting the rows to every query head.
        let lat = st
            .latents
            .iter()
            .map(|l| {
                backend.to_device(Tensor::from_vec(
                    vec![0.0f32; l.len()],
                    vec![sh.max_len, 1, sh.kv_lora],
                ))
            })
            .collect();
        st.kda = Vec::new();
        st.kda_dev = Some((dev, Arc::clone(&backend)));
        st.latents_dev = Some((lat, backend));
        Ok(st)
    }

    /// How many KDA layers this state covers, whichever side it lives on.
    pub fn n_kda(&self) -> usize {
        match &self.kda_dev {
            Some((d, _)) => d.len(),
            None => self.kda.len(),
        }
    }

    /// Largest magnitude anywhere in the KDA recurrent state. Host state only.
    ///
    /// For watching whether the recurrence is stable as a sequence grows: the state
    /// is scaled by `exp(g_log)` every token, so a positive `g_log` on any channel
    /// shows up here as geometric growth.
    pub fn kda_max_abs(&self) -> f32 {
        // The state is usually on a card, where `kda` is empty -- reading only the
        // host copy reported 0.0 for every device model, which makes the diagnostic
        // useless exactly where it is wanted.
        if let Some((dev, backend)) = &self.kda_dev {
            return dev.iter().fold(0.0f32, |a, t| {
                backend
                    .to_host(t.clone())
                    .data()
                    .iter()
                    .fold(a, |b, v| b.max(v.abs()))
            });
        }
        self.kda
            .iter()
            .flat_map(|s| s.iter())
            .fold(0.0f32, |a, v| a.max(v.abs()))
    }

    /// Largest magnitude in the cached MLA latents.
    pub fn latent_max_abs(&self) -> f32 {
        self.latents
            .iter()
            .flat_map(|s| s.iter())
            .fold(0.0f32, |a, v| a.max(v.abs()))
    }

    /// Whether the recurrent state is resident on a backend.
    pub fn kda_on_device(&self) -> bool {
        self.kda_dev.is_some()
    }

    pub fn reset(&mut self) {
        self.len = 0;
        if let Some((dev, backend)) = &mut self.kda_dev {
            for t in dev.iter_mut() {
                let shape = t.shape().to_vec();
                let n = t.numel();
                *t = backend.to_device(Tensor::from_vec(vec![0.0f32; n], shape));
            }
        }
        for s in self.kda.iter_mut() {
            s.fill(0.0);
        }
        for c in self.conv.iter_mut() {
            c.fill(0.0);
        }
        for l in self.latents.iter_mut() {
            l.fill(0.0);
        }
        // The mirror as well: a stale row past `len` would be a previous
        // conversation's, and the mask only hides rows it knows about.
        if let Some((dev, backend)) = &mut self.latents_dev {
            for t in dev.iter_mut() {
                let shape = t.shape().to_vec();
                let n = t.numel();
                *t = backend.to_device(Tensor::from_vec(vec![0.0f32; n], shape));
            }
        }
        self.kpool.reset();
    }

    pub fn kpool_cache(&self) -> &kpool::KpoolCache {
        &self.kpool
    }
}

/// VENDORED-LOCAL: GLM-5.3-Flash. Everything a prompt leaves behind, as plain f32.
///
/// A prompt is the expensive part of a request and a harness sends the same one
/// every time: coder-cli's system prompt is thousands of tokens, and reading it
/// takes minutes. This is what lets a later process start where an earlier one
/// finished instead of reading it again -- the same trick `dsv41_cuda::Snapshot`
/// does for DeepSeek, whose states `oaiy-llm-server` already keeps on disk.
///
/// The recurrent state is fixed-size (34 layers x 4.2 MB) and has to be kept whole.
/// The caches are not: `latents` and the indexer buffers are allocated for
/// `max_len` and written from the front, so only the first `len` tokens' rows mean
/// anything and the rest would be zeros on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSnapshot {
    /// Tokens this state covers.
    pub len: usize,
    /// Per KDA layer, the delta-rule state.
    pub kda: Vec<Vec<f32>>,
    /// Per KDA layer, the depthwise conv's pre-conv window.
    pub conv: Vec<Vec<f32>>,
    /// Per MLA layer, the cached latent rows for `len` tokens.
    pub latents: Vec<Vec<f32>>,
    /// Per MLA layer, the indexer rows for `len` tokens.
    pub kpool: Vec<Vec<f32>>,
}

impl StateSnapshot {
    /// Size of [`Self::encode`]'s output.
    pub fn encoded_len(&self) -> usize {
        let group = |g: &Vec<Vec<f32>>| 4 + g.iter().map(|v| 8 + 4 * v.len()).sum::<usize>();
        8 + group(&self.kda) + group(&self.conv) + group(&self.latents) + group(&self.kpool)
    }

    /// Append the snapshot's bytes, little-endian.
    ///
    /// Four groups of per-layer f32, each length-prefixed, after the token count.
    /// Plain and self-describing on purpose: a state that cannot be read back is
    /// worse than one that was never kept, and the reader checks every length
    /// against the state it is filling anyway.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.reserve(self.encoded_len());
        out.extend((self.len as u64).to_le_bytes());
        for group in [&self.kda, &self.conv, &self.latents, &self.kpool] {
            out.extend((group.len() as u32).to_le_bytes());
            for v in group {
                out.extend((v.len() as u64).to_le_bytes());
                for &f in v {
                    out.extend(f.to_le_bytes());
                }
            }
        }
    }

    /// Read back what [`Self::encode`] wrote.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let bad = || LlamaError::Config("forward: damaged prompt state".into());
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let end = at.checked_add(n).ok_or_else(bad)?;
            let s = bytes.get(at..end).ok_or_else(bad)?;
            at = end;
            Ok(s)
        };
        let len = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?) as usize;
        let mut groups: Vec<Vec<Vec<f32>>> = Vec::with_capacity(4);
        for _ in 0..4 {
            let n = u32::from_le_bytes(take(4)?.try_into().map_err(|_| bad())?) as usize;
            let mut group = Vec::with_capacity(n);
            for _ in 0..n {
                let count = u64::from_le_bytes(take(8)?.try_into().map_err(|_| bad())?) as usize;
                let raw = take(count.checked_mul(4).ok_or_else(bad)?)?;
                group.push(
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            }
            groups.push(group);
        }
        if at != bytes.len() {
            return Err(LlamaError::Config(format!(
                "forward: prompt state has {} trailing bytes",
                bytes.len() - at
            )));
        }
        let mut it = groups.into_iter();
        Ok(Self {
            len,
            kda: it.next().ok_or_else(bad)?,
            conv: it.next().ok_or_else(bad)?,
            latents: it.next().ok_or_else(bad)?,
            kpool: it.next().ok_or_else(bad)?,
        })
    }
}

impl State {
    /// VENDORED-LOCAL: GLM-5.3-Flash. Copy this state out, for disk or another
    /// process.
    ///
    /// Downloads the recurrent state when it lives on a card: 143 MB, once per
    /// snapshot rather than once per layer per token, which is why the state is
    /// kept there in the first place.
    pub fn snapshot(&self, kv_lora: usize) -> Result<StateSnapshot> {
        let kda = match &self.kda_dev {
            Some((tensors, backend)) => tensors
                .iter()
                .map(|t| backend.to_host(t.clone()).data().to_vec())
                .collect(),
            None => self.kda.clone(),
        };
        let keep = self.len * kv_lora;
        Ok(StateSnapshot {
            len: self.len,
            kda,
            conv: self.conv.clone(),
            latents: self
                .latents
                .iter()
                .map(|l| l.get(..keep.min(l.len())).unwrap_or(l).to_vec())
                .collect(),
            kpool: self.kpool.rows_upto(self.len),
        })
    }

    /// Put a snapshot back, so the next token continues from it.
    ///
    /// Everything past `len` is zeroed rather than left alone: a state being
    /// restored may be shorter than whatever this one held, and the difference has
    /// to read as "not written yet" and not as another prompt's rows.
    pub fn restore(&mut self, snap: &StateSnapshot, kv_lora: usize) -> Result<()> {
        if snap.len > self.max_len {
            return Err(LlamaError::Config(format!(
                "forward: restoring {} tokens into a {}-token state",
                snap.len, self.max_len
            )));
        }
        if snap.kda.len() != self.conv.len()
            || snap.conv.len() != self.conv.len()
            || snap.latents.len() != self.latents.len()
        {
            return Err(LlamaError::Config(format!(
                "forward: snapshot has {} KDA / {} conv / {} MLA layers, state has {} / {}",
                snap.kda.len(),
                snap.conv.len(),
                snap.latents.len(),
                self.conv.len(),
                self.latents.len()
            )));
        }
        match &mut self.kda_dev {
            Some((tensors, backend)) => {
                for (t, src) in tensors.iter_mut().zip(&snap.kda) {
                    let want = t.numel();
                    if src.len() != want {
                        return Err(LlamaError::Config(format!(
                            "forward: restoring {} KDA values into {want}",
                            src.len()
                        )));
                    }
                    *t = backend.to_device(Tensor::from_vec(src.clone(), t.shape().to_vec()));
                }
            }
            None => {
                for (dst, src) in self.kda.iter_mut().zip(&snap.kda) {
                    if src.len() != dst.len() {
                        return Err(LlamaError::Config(format!(
                            "forward: restoring {} KDA values into {}",
                            src.len(),
                            dst.len()
                        )));
                    }
                    dst.copy_from_slice(src);
                }
            }
        }
        for (dst, src) in self.conv.iter_mut().zip(&snap.conv) {
            if src.len() != dst.len() {
                return Err(LlamaError::Config(format!(
                    "forward: restoring {} conv values into {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
        }
        let _ = kv_lora;
        for (dst, src) in self.latents.iter_mut().zip(&snap.latents) {
            if src.len() > dst.len() {
                return Err(LlamaError::Config(format!(
                    "forward: restoring {} latent values into {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst[..src.len()].copy_from_slice(src);
            dst[src.len()..].fill(0.0);
        }
        // And the device mirror, from what was just put back.
        if let Some((dev, backend)) = &mut self.latents_dev {
            for (t, src) in dev.iter_mut().zip(&self.latents) {
                let shape = t.shape().to_vec();
                *t = backend.to_device(Tensor::from_vec(src.clone(), shape));
            }
        }
        self.kpool.restore_rows(&snap.kpool)?;
        self.len = snap.len;
        Ok(())
    }
}
