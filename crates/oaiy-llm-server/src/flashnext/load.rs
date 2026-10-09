//! Loading a Flash-Next checkpoint: its matrices, its experts, its adapters and its prediction layer.

use super::*;

/// The EXL3 matrix `name` (`k -> n`, mul1 codebook) on `backend`, its input and output
/// channels optionally reordered.
/// A dense `rows x cols` matrix: kept as f16 (half the memory and reading) when every value is
/// one, as the checkpoint's f16 weights are; f32 otherwise.
/// Without a card, f32 on `backend`.
pub(super) fn half_or_dense(card: Option<&Arc<Card>>, backend: &Arc<dyn Backend>, values: Vec<f32>, rows: usize, cols: usize) -> Weight {
    let _ = card;
    Weight::Dense(backend.to_device(Tensor::from_vec(values, vec![rows, cols])))
}

/// The EXL3 matrix `name` (`k -> n`), read and checked, on the host.
pub(crate) fn exl3_data(idx: &StIndex, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>) -> Result<Exl3Data> {
    let key = format!("{name}.trellis");
    let info = idx.info(&key)?;
    if info.dtype != Dtype::I16 || info.shape.len() != 3 || info.shape[..2] != [k / 16, n / 16] {
        return Err(bad(format!("{key}: invalid EXL3 tile shape {:?}", info.shape)));
    }
    let mul = idx.read_i64(&format!("{name}.mul1"))?;
    if mul.len() != 1 || mul[0] as u32 != MUL1 as u32 {
        return Err(bad(format!("{name}: unsupported codebook")));
    }
    let suh = idx.read_f32(&format!("{name}.suh"))?;
    let svh = idx.read_f32(&format!("{name}.svh"))?;
    if suh.len() != k || svh.len() != n {
        return Err(bad(format!("{name}: invalid incoherence vector lengths")));
    }
    let bytes = idx.read(&key)?;
    let data = Exl3Data {
        words: bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        suh,
        svh,
        tile_words: info.shape[2],
        input_map: input.unwrap_or_else(|| (0..k as u32).collect()),
        output_map: output.unwrap_or_else(|| (0..n as u32).collect()),
    };
    Ok(data)
}

pub(super) struct Loader<'a> {
    pub(super) idx: &'a StIndex,
    pub(super) backend: Arc<dyn Backend>,
    /// The device's CUDA card (none without CUDA).
    pub(super) card: Option<Arc<Card>>,
    /// LoRA adapters, applied together.
    pub(super) lora: &'a [Adapter],
    /// Makes this device's EXL3 matrices.
    pub(super) packed: Box<dyn Fn(Exl3Data) -> std::result::Result<Arc<dyn PackedLinear>, String> + Send + Sync + 'a>,
}

impl Loader<'_> {
    pub(super) fn host(&self, name: &str, numel: usize, add_one: bool, map: Option<&[u32]>) -> Result<Vec<f32>> {
        let mut data = self.idx.read_f32(name)?;
        if data.len() != numel {
            return Err(bad(format!("{name}: {} values, expected {numel}", data.len())));
        }
        if add_one { for v in &mut data { *v += 1.0; } }
        if let Some(map) = map {
            let width = data.len() / map.len();
            let src = data.clone();
            for (i, &j) in map.iter().enumerate() {
                data[i * width..(i + 1) * width].copy_from_slice(&src[j as usize * width..(j as usize + 1) * width]);
            }
        }
        Ok(data)
    }
    pub(super) fn tensor(&self, name: &str, shape: &[usize], add_one: bool, map: Option<&[u32]>) -> Result<Tensor> {
        let data = self.host(name, shape.iter().product(), add_one, map)?;
        Ok(self.backend.to_device(Tensor::from_vec(data, shape.to_vec())))
    }
    pub(super) fn dense(&self, name: &str, rows: usize, cols: usize) -> Result<Weight> {
        let base = half_or_dense(self.card.as_ref(), &self.backend, self.host(name, rows * cols, false, None)?, rows, cols);
        let target = name.strip_suffix(".weight").unwrap_or(name);
        self.adapt(base, &[Part { name: target.into(), offset: 0, rows, output: None }], cols, rows, None)
    }
    /// `base` (`n x k`) with the adapters' LoRA for `parts` of it, if they have any.
    pub(super) fn adapt(&self, base: Weight, parts: &[Part<'_>], k: usize, n: usize, input: Option<&[u32]>) -> Result<Weight> {
        Ok(match stacked(self.lora, parts, k, n, input)? {
            Some((a, b, rank)) => adapted(base, a, b, rank, self.backend.clone()),
            None => base,
        })
    }
    /// An EXL3 matrix `k -> n`.
    pub(super) fn weight(&self, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>) -> Result<Weight> {
        self.weight_parts(name, k, n, input, output, &[])
    }
    /// As `weight`, with LoRA also for `more` parts of it (Hugging Face matrices llama.cpp
    /// splits it into).
    pub(super) fn weight_parts(&self, name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>, more: &[Part<'_>]) -> Result<Weight> {
        let base = Weight::Packed((self.packed)(exl3_data(self.idx, name, k, n, input.clone(), output.clone())?).map_err(bad)?);
        let mut parts = vec![Part { name: name.into(), offset: 0, rows: n, output: output.as_deref() }];
        parts.extend(more.iter().map(|p| Part { name: p.name.clone(), offset: p.offset, rows: p.rows, output: p.output }));
        self.adapt(base, &parts, k, n, input.as_deref())
    }
    pub(super) fn hyper(&self, p: &str, cfg: &Config, site: bool) -> Result<HyperMix> {
        let width = cfg.streams * cfg.hidden;
        let rank = self.idx.info(&format!("{p}.input_mix_weight_down.weight"))?.shape[0];
        let writes = if site { cfg.streams } else { 0 };
        // The write logits ride along the down projection; the up projection ignores them.
        let mut down = self.host(&format!("{p}.input_mix_weight_down.weight"), rank * width, false, None)?;
        if site {
            down.extend(self.host(&format!("{p}.block_inject_weight.weight"), cfg.streams * width, false, None)?);
        }
        let up = self.host(&format!("{p}.input_mix_weight_up.weight"), width * rank, false, None)?;
        let mut padded = Vec::with_capacity(width * (rank + writes));
        for row in up.chunks_exact(rank) {
            padded.extend_from_slice(row);
            padded.extend(std::iter::repeat_n(0.0, writes));
        }
        // f16 in the checkpoint: kept so on a card, two to a word (half the memory and reading); f32 elsewhere.
        let packed = self.card.is_some();
        let (down, up) = if packed {
            (
                Tensor::from_vec(ggml_rs::tensor::pack_f16(&down), vec![rank + writes, width / 2]),
                Tensor::from_vec(ggml_rs::tensor::pack_f16(&padded), vec![width, (rank + writes) / 2]),
            )
        } else {
            (Tensor::from_vec(down, vec![rank + writes, width]), Tensor::from_vec(padded, vec![width, rank + writes]))
        };
        Ok(HyperMix {
            norm: self.tensor(&format!("{p}.hc_norm.weight"), &[width], true, None)?,
            down: self.backend.to_device(down),
            up: self.backend.to_device(up),
            rank,
            site,
            packed,
        })
    }
}

/// What a LoRA for the checkpoint at `path` must be for.
pub(crate) fn lora_base(path: &Path) -> Result<Base> {
    let cfg = Config::read(path)?;
    Ok(Base::FlashNext(VHeads { k_heads: cfg.nk, v_heads: cfg.nv, k_dim: cfg.kd, v_dim: cfg.vd }))
}

/// A layer's LoRA on projection `which` (0 gate, 1 up, 2 down) of its experts (the shared one
/// last): which have one (`u32::MAX` for none), and their A and B at one rank (the largest,
/// the others padded with zeros).
pub(super) fn expert_lora(lora: &[Adapter], m: &str, cfg: &Config, which: usize) -> Result<Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize)>> {
    let proj = ["gate_proj", "up_proj", "down_proj"][which];
    let (k, n) = if which == 2 { (cfg.moe_ff, cfg.hidden) } else { (cfg.hidden, cfg.moe_ff) };
    let mut found = Vec::new();
    for e in 0..=cfg.experts {
        let name = if e < cfg.experts { format!("{m}.experts.{e}.{proj}") } else { format!("{m}.shared_expert.{proj}") };
        if let Some(pair) = stacked(lora, &[Part { name, offset: 0, rows: n, output: None }], k, n, None)? {
            found.push((e, pair));
        }
    }
    let Some(rank) = found.iter().map(|(_, p)| p.2).max() else { return Ok(None) };
    let mut slot_of = vec![u32::MAX; cfg.experts + 1];
    let (mut a_all, mut b_all) = (Vec::with_capacity(found.len() * rank * k), Vec::with_capacity(found.len() * n * rank));
    for (slot, (e, (a, b, r))) in found.into_iter().enumerate() {
        slot_of[e] = slot as u32;
        a_all.extend_from_slice(&a);
        a_all.resize(a_all.len() + (rank - r) * k, 0.0);
        for row in b.chunks_exact(r) {
            b_all.extend_from_slice(row);
            b_all.resize(b_all.len() + rank - r, 0.0);
        }
    }
    Ok(Some((slot_of, a_all, b_all, rank)))
}

/// A layer's experts (`m`'s, as `make` made them) with the adapters' low-rank updates beside their gate, up and down
/// projections, where the adapters have any: the experts apply them from now on, or the load fails saying they do
/// not (an adapter's target is never read and then left out).
pub(super) fn adapt_experts(lora: &[Adapter], m: &str, cfg: &Config, mut experts: Box<dyn Experts>) -> Result<Box<dyn Experts>> {
    if lora.is_empty() {
        return Ok(experts);
    }
    for (which, proj) in ["gate_proj", "up_proj", "down_proj"].into_iter().enumerate() {
        if let Some((slot_of, a, b, rank)) = expert_lora(lora, m, cfg, which)? {
            experts.low_rank(which, &slot_of, &a, &b, rank).map_err(|e| bad(format!("{m}: a LoRA for its experts' {proj}: {e}")))?;
        }
    }
    Ok(experts)
}

/// The multi-token-prediction layer of the EXL3 checkpoint `idx` indexes (its `mtp.*`: an attention block and a MoE
/// block behind their hyper-connections, the projections either side), on `device` as `last` makes its matrices and
/// `make_experts` its experts: None where the checkpoint has none.
pub(super) fn mtp_layer(idx: &StIndex, cfg: &Config, last: &Loader<'_>, device: usize, make_experts: ExpertMaker<'_>) -> Result<Option<FnMtp>> {
    if idx.get("mtp.fc_embedding.trellis").is_none() {
        return Ok(None);
    }
    let (h, width) = (cfg.hidden, cfg.streams * cfg.hidden);
    let (a, m) = ("mtp.layers.0.self_attn", "mtp.layers.0.mlp");
    let names: Vec<String> = (0..cfg.experts).map(|e| format!("{m}.experts.{e}")).chain([format!("{m}.shared_expert")]).collect();
    let read = |p: &String| -> Result<[Exl3Data; 3]> {
        Ok([
            exl3_data(idx, &format!("{p}.gate_proj"), h, cfg.moe_ff, None, None)?,
            exl3_data(idx, &format!("{p}.up_proj"), h, cfg.moe_ff, None, None)?,
            exl3_data(idx, &format!("{p}.down_proj"), cfg.moe_ff, h, None, None)?,
        ])
    };
    let per = names.len().div_ceil(16);
    let experts: Vec<[Exl3Data; 3]> = std::thread::scope(|scope| {
        let parts: Vec<_> = names.chunks(per).map(|part| scope.spawn(|| part.iter().map(read).collect::<Result<Vec<_>>>())).collect();
        parts.into_iter().map(|t| t.join().expect("expert reader")).collect::<Result<Vec<Vec<_>>>>()
    })?.into_iter().flatten().collect();
    let mut router = last.host(&format!("{m}.gate.weight"), cfg.experts * h, false, None)?;
    router.extend(last.host(&format!("{m}.shared_expert_gate.weight"), h, false, None)?);
    Ok(Some(FnMtp {
        enorm: last.tensor("mtp.pre_fc_norm_embedding.weight", &[h], true, None)?,
        hnorm: last.tensor("mtp.pre_fc_norm_hidden.weight", &[width], true, None)?,
        fc_e: last.weight("mtp.fc_embedding", h, h, None, None)?,
        fc_h: last.weight("mtp.fc_hidden", h, h, None, None)?,
        attn_hc: last.hyper("mtp.layers.0.attn_hyper_connection", cfg, true)?,
        mlp_hc: last.hyper("mtp.layers.0.mlp_hyper_connection", cfg, true)?,
        mixer: last.hyper("mtp.hyper_connection_mixer", cfg, false)?,
        q: last.weight(&format!("{a}.q_proj"), h, 2 * cfg.heads * cfg.head_dim, None, None)?,
        k: last.weight(&format!("{a}.k_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
        v: last.weight(&format!("{a}.v_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
        o: last.weight(&format!("{a}.o_proj"), cfg.heads * cfg.head_dim, h, None, None)?,
        q_norm: last.tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim], true, None)?,
        k_norm: last.tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim], true, None)?,
        router: Tensor::from_vec(router, vec![cfg.experts + 1, h]),
        experts: make_experts(device, m, experts)?,
    }))
}

/// The bytes of Flash-Next's EXL3 matrices outside its experts (attention, delta-net, the head): a portable build keeps
/// that much of the GPU budget for them, where the experts, loaded first, took it all and left the 248k-row head to the
/// CPU.
pub(crate) fn dense_exl3_bytes(path: &Path) -> Result<u64> {
    let idx = StIndex::open(path)?;
    Ok(idx.names().filter(|n| reserved(n)).filter_map(|n| idx.get(n).map(|i| i.nbytes)).sum())
}

/// Whether a tensor is one of the matrices `dense_exl3_bytes` keeps GPU budget for. Not the experts, and not the n-gram
/// table: its rows are trellis-quantized too, but it is read from the disk as needed and never placed on the GPU, and
/// counted (32.6 GB) it left the experts none of a 27 GiB budget.
pub(super) fn reserved(name: &str) -> bool {
    (name.ends_with(".trellis") && name.starts_with("model.language_model.") || name == "lm_head.trellis")
        && !name.contains(".experts.")
        && !name.contains(".shared_expert.")
        && !name.contains(".ngram_embedding.")
}

#[cfg(test)]
mod reserve_tests {
    #[test]
    fn the_reserve_counts_the_dense_matrices_and_the_head_not_the_experts_or_the_ngram_table() {
        let p = "model.language_model.layers.1";
        for name in [format!("{p}.self_attn.q_proj.trellis"), format!("{p}.linear_attn.in_proj_qkv.trellis"), "lm_head.trellis".into()] {
            assert!(super::reserved(&name), "{name}");
        }
        for name in [
            format!("{p}.mlp.experts.7.gate_proj.trellis"),
            format!("{p}.mlp.shared_expert.down_proj.trellis"),
            format!("{p}.ple.ple_embedding.ngram_embedding.shard_0.trellis"),
            format!("{p}.self_attn.q_proj.suh"),
        ] {
            assert!(!super::reserved(&name), "{name}");
        }
    }
}

/// Flash-Next without CUDA: its layers split over `backends` (a whole layer, its experts too, on each; the head on
/// the last), each EXL3 matrix as `packed` makes it on its device and each layer's experts as `experts` does (on any
/// GPU through WebGPU, else on the CPU), no PEFT adapters, a decode step uncaptured. (A CUDA build loads it on the
/// cards.)
/// `mtp`: its multi-token-prediction layer too (on the last device), for drafting.
pub(crate) fn load_portable(path: &Path, backends: Vec<Arc<dyn Backend>>, packed: Packer<'_>, experts: ExpertMaker<'_>, mtp: bool) -> Result<FlashNext> {
    load_portable_with(path, backends, &[], packed, experts, mtp)
}

/// [`load_portable`] with `lora`'s adapters applied, together, as the weights load: beside each dense projection
/// they adapt, on its device, and beside the experts' projections (the shared expert's and the routed ones'), which
/// the experts `experts` makes are then given ([`adapt_experts`]).
pub(crate) fn load_portable_with(path: &Path, backends: Vec<Arc<dyn Backend>>, lora: &[Adapter], packed: Packer<'_>, experts: ExpertMaker<'_>, mtp: bool) -> Result<FlashNext> {
    build(path, backends, Vec::new(), lora, packed, experts, mtp)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build(path: &Path, backends: Vec<Arc<dyn Backend>>, cudas: Vec<Arc<Card>>, lora: &[Adapter], packed: Packer<'_>, make_experts: ExpertMaker<'_>, mtp: bool) -> Result<FlashNext> {
    if !detect(path) {
        return Err(bad("not a Qwen3.8-Flash-Next EXL3 checkpoint"));
    }
    let cfg = Config::read(path)?;
    let tok = tokenizer(path, cfg.vocab)?;
    let idx = StIndex::open(path)?;
    let on = |d: usize| Loader { idx: &idx, backend: backends[d].clone(), card: cudas.get(d).cloned(), lora, packed: packed(d) };
    let p = "model.language_model";
    let h = cfg.hidden;

    let embed = idx.info(&format!("{p}.embed_tokens.weight"))?.clone();
    if embed.shape != [cfg.vocab, h] || !matches!(embed.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32) {
        return Err(bad("invalid embedding table"));
    }

    let (nk, nv, kd, vd, conv) = (cfg.nk, cfg.nv, cfg.kd, cfg.vd, cfg.conv);
    let vm = value_map(nk, nv, vd);
    let hm = value_map(nk, nv, 1);
    let qkv = 2 * nk * kd + nv * vd;
    let qm: Vec<u32> = (0..(2 * nk * kd) as u32).chain(vm.iter().map(|v| v + 2 * (nk * kd) as u32)).collect();
    // A layer is independent of the others: four load at once (the reads, the stacking of its
    // experts and the uploads overlap, on both GPUs).
    let load_layer = |i: usize| -> Result<Layer> {
        let device = i * backends.len() / cfg.layers;
        let l = on(device);
        let lp = format!("{p}.layers.{i}");
        let mixer = if cfg.attention[i] {
            let a = format!("{lp}.self_attn");
            Mixer::Attn(Attn {
                q: l.weight(&format!("{a}.q_proj"), h, 2 * cfg.heads * cfg.head_dim, None, None)?,
                k: l.weight(&format!("{a}.k_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
                v: l.weight(&format!("{a}.v_proj"), h, cfg.kv_heads * cfg.head_dim, None, None)?,
                o: l.weight(&format!("{a}.o_proj"), cfg.heads * cfg.head_dim, h, None, None)?,
                q_norm: l.tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim], true, None)?,
                k_norm: l.tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim], true, None)?,
                // llama.cpp splits it into the indexer's query and key projections.
                index_qk: l.weight_parts(&format!("{a}.indexer.index_qk_proj"), h, (cfg.index_heads + 1) * cfg.index_dim, None, None, &[
                    Part { name: format!("{a}.indexer.index_q_proj"), offset: 0, rows: cfg.index_heads * cfg.index_dim, output: None },
                    Part { name: format!("{a}.indexer.index_k_proj"), offset: cfg.index_heads * cfg.index_dim, rows: cfg.index_dim, output: None },
                ])?,
                index_q_norm: l.tensor(&format!("{a}.indexer.q_layernorm.weight"), &[cfg.index_dim], true, None)?,
                index_k_norm: l.tensor(&format!("{a}.indexer.k_layernorm.weight"), &[cfg.index_dim], true, None)?,
                index_slot: cfg.layers + 1 + cfg.attention[..i].iter().filter(|&&a| a).count(),
            })
        } else {
            let a = format!("{lp}.linear_attn");
            let beta = l.host(&format!("{a}.in_proj_b.weight"), nv * h, false, Some(&hm))?;
            let alpha = l.host(&format!("{a}.in_proj_a.weight"), nv * h, false, Some(&hm))?;
            let ba = Tensor::from_vec([beta, alpha].concat(), vec![2 * nv, h]);
            let mut av = l.host(&format!("{a}.A_log"), nv, false, None)?;
            for v in &mut av { *v = -v.exp(); }
            let av: Vec<f32> = hm.iter().map(|&j| av[j as usize]).collect();
            Mixer::Gdn(Gdn {
                qkv: l.weight(&format!("{a}.in_proj_qkv"), h, qkv, None, Some(qm.clone()))?,
                z: l.weight(&format!("{a}.in_proj_z"), h, nv * vd, None, Some(vm.clone()))?,
                ba: l.adapt(half_or_dense(cudas.get(device), &backends[device], ba.data().to_vec(), 2 * nv, h), &[
                    Part { name: format!("{a}.in_proj_b"), offset: 0, rows: nv, output: Some(&hm) },
                    Part { name: format!("{a}.in_proj_a"), offset: nv, rows: nv, output: Some(&hm) },
                ], h, 2 * nv, None)?,
                a: backends[device].to_device(Tensor::from_vec(av, vec![nv])),
                dt_bias: l.tensor(&format!("{a}.dt_bias"), &[nv], false, Some(&hm))?,
                conv: l.tensor(&format!("{a}.conv1d.weight"), &[qkv, conv], false, Some(&qm))?,
                norm: l.tensor(&format!("{a}.norm.weight"), &[vd], false, None)?,
                out: l.weight(&format!("{a}.out_proj"), nv * vd, h, Some(inverse(&vm)), None)?,
            })
        };
        let m = format!("{lp}.mlp");
        if cfg.shared_ff != cfg.moe_ff {
            return Err(bad("the shared expert must be as wide as the routed ones"));
        }
        // 1539 matrices a layer, read on several threads (each read is small; the latency adds up).
        let names: Vec<String> = (0..cfg.experts).map(|e| format!("{m}.experts.{e}")).chain([format!("{m}.shared_expert")]).collect();
        let read = |p: &String| -> Result<[Exl3Data; 3]> {
            Ok([
                exl3_data(&idx, &format!("{p}.gate_proj"), h, cfg.moe_ff, None, None)?,
                exl3_data(&idx, &format!("{p}.up_proj"), h, cfg.moe_ff, None, None)?,
                exl3_data(&idx, &format!("{p}.down_proj"), cfg.moe_ff, h, None, None)?,
            ])
        };
        let per = names.len().div_ceil(16);
        let experts: Vec<[Exl3Data; 3]> = std::thread::scope(|scope| {
            let parts: Vec<_> = names.chunks(per).map(|part| scope.spawn(|| part.iter().map(read).collect::<Result<Vec<_>>>())).collect();
            parts.into_iter().map(|t| t.join().expect("expert reader")).collect::<Result<Vec<Vec<_>>>>()
        })?.into_iter().flatten().collect();
        let mut router = l.host(&format!("{m}.gate.weight"), cfg.experts * h, false, None)?;
        router.extend(l.host(&format!("{m}.shared_expert_gate.weight"), h, false, None)?);
        let experts = adapt_experts(lora, &m, &cfg, make_experts(device, &m, experts)?)?;
        let moe = Moe {
            router: l.adapt(half_or_dense(cudas.get(device), &backends[device], router, cfg.experts + 1, h), &[
                Part { name: format!("{m}.gate"), offset: 0, rows: cfg.experts, output: None },
                Part { name: format!("{m}.shared_expert_gate"), offset: cfg.experts, rows: 1, output: None },
            ], h, cfg.experts + 1, None)?,
            experts,
        };
        Ok(Layer {
            device,
            attn_hc: l.hyper(&format!("{lp}.attn_hyper_connection"), &cfg, true)?,
            mlp_hc: l.hyper(&format!("{lp}.mlp_hyper_connection"), &cfg, true)?,
            mixer,
            moe,
        })
    };
    const WORKERS: usize = 4;
    let mut loaded: Vec<Option<Layer>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS).map(|w| {
            let load_layer = &load_layer;
            scope.spawn(move || (w..cfg.layers).step_by(WORKERS).map(|i| load_layer(i).map(|l| (i, l))).collect::<Result<Vec<_>>>())
        }).collect();
        let mut out: Vec<Option<Layer>> = (0..cfg.layers).map(|_| None).collect();
        for w in workers {
            for (i, layer) in w.join().expect("layer loader")? { out[i] = Some(layer); }
        }
        Ok::<_, Error>(out)
    })?;
    let layers: Vec<Layer> = loaded.iter_mut().map(|l| l.take().expect("every layer")).collect();

    // The n-gram layer sits with the block it runs before.
    let pd = layers[cfg.ple_layer].device;
    let l = on(pd);
    let pp = format!("{p}.layers.{}.ple", cfg.ple_layer);
    let width = cfg.streams * h;
    let heads = (cfg.ngram - 1) * cfg.heads_per_ngram;
    if heads * ROW_DIM != cfg.ple_dim {
        return Err(bad("n-gram heads do not make up the embedding width"));
    }
    let ple = Ple {
        table: NgramTable::open(&idx, &format!("{pp}.ple_embedding.ngram_embedding"), heads)?,
        key: l.dense(&format!("{pp}.key_proj.weight"), width, cfg.ple_dim)?,
        value: l.dense(&format!("{pp}.value_proj.weight"), h, cfg.ple_dim)?,
        norm_key: l.tensor(&format!("{pp}.norm_key.weight"), &[width], true, None)?,
        norm_query: l.tensor(&format!("{pp}.norm_query.weight"), &[width], true, None)?,
        norm_conv: l.tensor(&format!("{pp}.norm_conv.weight"), &[width], true, None)?,
        conv: l.tensor(&format!("{pp}.conv1d.weight"), &[width, cfg.ple_kernel], false, None)?,
    };

    let last = on(backends.len() - 1);
    let collapse = last.hyper(&format!("{p}.hyper_connection_mixer"), &cfg, false)?;
    let head = last.weight("lm_head", h, cfg.vocab, None, None)?;
    // The multi-token-prediction layer, where asked for and the checkpoint has one: on the last device, with the head.
    let mtp = if mtp { mtp_layer(&idx, &cfg, &last, backends.len() - 1, make_experts)? } else { None };
    // Every target of every adapter found its matrix.
    for adapter in lora { adapter.finish()?; }
    let file = File::open(idx.shard_path(embed.shard))?;
    Ok(FlashNext {
        tokenizer: tok,
        embed: Embed::Plain(file, embed.start, embed.dtype),
        layers,
        ple,
        collapse,
        head,
        devices: backends,
        cudas,
        decoded: Default::default(),
        chain: Default::default(),
        mtp,
        placed: Default::default(),
        pick: Default::default(),
        config: cfg,
    })
}
