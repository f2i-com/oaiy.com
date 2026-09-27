//! Read OrcaSAQ2's original sharded EXL3 checkpoint. No Python runtime/conversion.
use dsv41::safetensors::{Dtype, StIndex};
use ggml_rs::{exl3::Exl3Data, Backend, Tensor};
use ggml_rs_cuda::{exl3::Exl3Matrix, CudaBackend};
use llama_rs::{
    loader::{FfnPair, Weight},
    qwen35::{Qwen35Block, Qwen35Model, SsmConfig},
    Architecture, Model, ModelConfig,
};
use nrob::{json::Json, Error, Result};
use std::{path::Path, sync::Arc};

fn bad(s: impl Into<String>) -> Error {
    Error::Format(s.into())
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    fn model_path() -> std::path::PathBuf {
        std::env::var_os("NROB_TEST_ORCA").map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/OrcaSAQ-2-27B"))
    }
    #[test]
    #[ignore = "manual real-model prefill timing/profiling, requires downloaded weights"]
    fn benchmark_real_model_prefill() {
        let path = model_path();
        let adapter=std::env::var_os("NROB_TEST_LORA").map(|p|crate::lora::Adapter::open(Path::new(&p)).unwrap());
        let model = load_with_adapter(&path, &[0, 1], adapter.as_slice()).unwrap();
        let token = model.tokenizer().encode("Hello", false).unwrap()[0];
        let sizes = std::env::var("NROB_BENCH_PREFILL_SIZES").ok().map(|s|s.split(',').map(|n|n.parse::<usize>().unwrap()).collect::<Vec<_>>())
            .unwrap_or_else(||vec![crate::qwen::PREFILL_CHUNK]);
        for size in sizes {
          let tokens = vec![token; size];
          for run in 0..2 {
            let mut kv = model.new_kv_cache(260000);
            let start = std::time::Instant::now();
            let logits = model.forward(&tokens, &mut kv).to_host();
            assert!(logits.data().iter().all(|v| v.is_finite()));
            eprintln!("prefill run {run}: {} tokens in {:.3}s", tokens.len(), start.elapsed().as_secs_f64());
          }
        }
    }
    #[test]
    #[ignore = "manual real-model decode timing/profiling, requires downloaded weights"]
    fn benchmark_real_model_decode() {
        let path = model_path();
        let devices: &[usize] = if std::env::var_os("NROB_BENCH_LOCAL_KV").is_some() { &[0] } else { &[0, 1] };
        let adapter=std::env::var_os("NROB_TEST_LORA").map(|p|crate::lora::Adapter::open(Path::new(&p)).unwrap());
        let model = load_with_adapter(&path, devices, adapter.as_slice()).unwrap();
        let mut kv = model.new_kv_cache(260000);
        let token = model.tokenizer().encode("Hello", false).unwrap()[0];
        for _ in 0..16 { let _ = model.forward(&[token], &mut kv).to_host(); }
        let start = std::time::Instant::now();
        const STEPS: usize = 128;
        for _ in 0..STEPS { let _ = model.forward(&[token], &mut kv).to_host(); }
        let seconds=start.elapsed().as_secs_f64();
        eprintln!("{STEPS} decode steps: {seconds:.3}s, {:.2} tokens/s", STEPS as f64/seconds);
    }
    #[test]
    #[ignore = "requires downloaded OrcaSAQ and two 32GB CUDA devices"]
    fn real_model_distributed_cache_restores_and_reserves_260000() {
        let path = model_path();
        let adapter=std::env::var_os("NROB_TEST_LORA").map(|p|crate::lora::Adapter::open(Path::new(&p)).unwrap());
        let baseline=adapter.as_ref().map(|_| {
            let base=load(&path,&[0,1]).unwrap();
            let tokens=base.tokenizer().encode("The capital of France is",false).unwrap();
            base.forward(&tokens,&mut base.new_kv_cache(260000)).to_host().data().to_vec()
        });
        let mut model = load_with_adapter(&path, &[0, 1],adapter.as_slice()).unwrap();
        let vision = if std::env::var_os("NROB_TEST_VISION").is_some() {
            let Model::Qwen35(m)=&model else { unreachable!() };
            Some(crate::qwen_vision::load(&path.join("vision"),m.cache_backends.last().unwrap().clone(),5120,None).unwrap())
        } else { None };
        let tokens = model
            .tokenizer()
            .encode("The capital of France is", false)
            .unwrap();
        let mut kv = model.new_kv_cache(260000);
        let distributed = model.forward(&tokens, &mut kv).to_host();
        if let Some(baseline)=baseline {
            let difference=baseline.iter().zip(distributed.data()).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            assert!(difference>1e-3,"nonzero adapter must affect real-model logits");
            eprintln!("LoRA changes real-model logits: max delta {difference}");
        }
        let Model::Qwen35(m) = &mut model else {
            unreachable!()
        };
        let placement = m.cache_backends.clone();
        m.cache_backends.fill(m.backend.clone());
        let mut local = model.new_kv_cache(260000);
        let expected = model.forward(&tokens, &mut local).to_host();
        let error = expected
            .data()
            .iter()
            .zip(distributed.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 1e-4, "distributed attention logits error {error}");
        drop(local);
        let Model::Qwen35(m) = &mut model else {
            unreachable!()
        };
        m.cache_backends = placement;
        let snap =
            crate::qwen_cache::Snapshot::capture(&kv, &m.attention_layers, m.backend.as_ref());
        let recurrent = crate::qwen_cache::RecurrentSnapshot::capture(&kv);
        let next = model.forward(&tokens[..1], &mut kv).to_host();
        let Model::Qwen35(m) = &model else {
            unreachable!()
        };
        snap.restore(&mut kv, &m.attention_layers, m.ssm_cfg, m.backend.as_ref())
            .unwrap();
        let restored = model.forward(&tokens[..1], &mut kv).to_host();
        assert_eq!(
            next.data(),
            restored.data(),
            "restored recurrence and both GPUs must match"
        );
        let Model::Qwen35(m) = &model else { unreachable!() };
        recurrent.restore(&mut kv, snap.pos, &m.attention_layers, m.ssm_cfg, m.backend.as_ref()).unwrap();
        let rewound = model.forward(&tokens[..1], &mut kv).to_host();
        assert_eq!(next.data(), rewound.data(), "recurrent-only rewind must preserve exact real-model logits");
        let Model::Qwen35(m) = &model else {
            unreachable!()
        };
        for (i, &attn) in m.attention_layers.iter().enumerate() {
            if attn {
                let backend = kv.layer_backends[i].clone();
                kv.reserve_layer(backend.as_ref(), i, 260000);
                assert_eq!(kv.k[i].dim(0), 260000);
            }
        }
        assert!(kv.size_bytes() >= 34_078_720_000);
        // Preserved prefix remains usable after growth to the full capacity.
        let logits = model.forward(&tokens[..1], &mut kv).to_host();
        assert!(logits.data().iter().all(|x| x.is_finite()));
        if let Some(vision)=vision {
            let pixels=Tensor::from_vec(vec![0.0;3*768*768],vec![3,768,768]);
            let out=vision.forward(&pixels).unwrap().to_host();
            assert_eq!(out.shape(),[576,5120]);
            assert!(out.data().iter().all(|x|x.is_finite()));
            let Model::Qwen35(m)=&model else { unreachable!() };
            let rows=crate::qwen::PREFILL_CHUNK.min(576);
            let embeds=m.backend.to_device(Tensor::from_vec(out.data()[..rows*5120].to_vec(),vec![rows,5120]));
            let positions:Vec<_>=(0..rows).map(|i|[4,4+(i/24) as u32,4+(i%24) as u32]).collect();
            let logits=m.forward_embeds_positions(&embeds,rows,&mut kv,Some(&positions)).unwrap().to_host();
            assert!(logits.data().iter().all(|x|x.is_finite()));
            eprintln!("{rows}-token image prefill also fits alongside the full KV allocation");
            eprintln!("vision encoding also verified with all 260000 KV slots resident");
        }
        eprintln!(
            "260000-token capacity: {} bytes; distributed/restored logits verified",
            kv.size_bytes()
        );
    }
}
fn json(path: &Path) -> Result<Json> {
    Json::parse(&std::fs::read(path)?)
}
fn number(c: &Json, key: &str) -> Result<usize> {
    let v = c
        .get(key)
        .and_then(Json::as_f64)
        .ok_or_else(|| bad(format!("missing {key}")))?;
    if !v.is_finite() || v < 1.0 || v.fract() != 0.0 || v > 1_048_576.0 {
        return Err(bad(format!("invalid {key}")));
    }
    Ok(v as usize)
}
pub fn detect(path: &Path) -> bool {
    json(&path.join("config.json")).ok().is_some_and(|c| {
        c.get("model_type").and_then(Json::as_str) == Some("qwen3_5")
            && c.get("quantization_config")
                .and_then(|q| q.get("quant_method"))
                .and_then(Json::as_str)
                == Some("exl3")
    })
}

struct Loader<'a> {
    /// Adapters applied together, in order.
    lora: &'a [crate::lora::Adapter],
    idx: StIndex,
    backend: Arc<CudaBackend>,
}
impl Loader<'_> {
    fn tensor(
        &self,
        name: &str,
        shape: &[usize],
        add_one: bool,
        map: Option<&[u32]>,
    ) -> Result<Tensor> {
        let info = self.idx.info(name)?;
        let conv_shape = shape.len() == 2 && info.shape == [shape[0], 1, shape[1]];
        if info.shape != shape && !conv_shape {
            return Err(bad(format!("{name}: incorrect shape {:?}", info.shape)));
        }
        let mut data = self.idx.read_f32(name)?;
        if add_one {
            for v in &mut data {
                *v += 1.0;
            }
        }
        if let Some(map) = map {
            let width = data.len() / map.len();
            let src = data.clone();
            for (i, &j) in map.iter().enumerate() {
                data[i * width..(i + 1) * width]
                    .copy_from_slice(&src[j as usize * width..(j as usize + 1) * width]);
            }
        }
        Ok(self
            .backend
            .to_device(Tensor::from_vec(data, shape.to_vec())))
    }
    fn weight(
        &self,
        name: &str,
        k: usize,
        n: usize,
        input: Option<Vec<u32>>,
        output: Option<Vec<u32>>,
    ) -> Result<Weight> {
        let key = format!("{name}.trellis");
        let info = self.idx.info(&key)?;
        if info.dtype != Dtype::I16 || info.shape.len() != 3 || info.shape[..2] != [k / 16, n / 16]
        {
            return Err(bad(format!("{key}: invalid EXL3 tile shape")));
        }
        let tw = info.shape[2];
        // Published mul1 tensors use this fixed procedural codebook. Reject other codebooks.
        let mul = self.idx.read_i64(&format!("{name}.mul1"))?;
        if mul.len() != 1 || mul[0] as u32 != 0x83dcd12d {
            return Err(bad(format!("{name}: unsupported mul1 codebook")));
        }
        let suh = self.idx.read_f32(&format!("{name}.suh"))?;
        let svh = self.idx.read_f32(&format!("{name}.svh"))?;
        if suh.len() != k || svh.len() != n {
            return Err(bad(format!("{name}: invalid incoherence vector lengths")));
        }
        let bytes = self.idx.read(&key)?;
        let words = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let data = Exl3Data {
            words,
            suh,
            svh,
            tile_words: tw,
            input_map: input.clone().unwrap_or_else(|| (0..k as u32).collect()),
            output_map: output.clone().unwrap_or_else(|| (0..n as u32).collect()),
        };
        let base=Weight::Packed(Arc::new(Exl3Matrix::upload(self.backend.clone(), data).map_err(bad)?));
        // Each adapter wraps what the one before made (an adapter without this weight leaves it).
        self.lora.iter().try_fold(base, |w, adapter| adapter.wrap(name,w,self.backend.clone(),input.as_deref(),output.as_deref()))
    }
}

/// GGML delta-net uses tiled value heads; HF uses grouped heads. This maps
/// each tiled output channel back to its original HF channel, without changing weights.
pub(crate) fn value_map(nk: usize, nv: usize, dim: usize) -> Vec<u32> {
    (0..nv * dim)
        .map(|i| (((i / dim) % nk) * (nv / nk) * dim + (i / dim / nk) * dim + i % dim) as u32)
        .collect()
}
pub(crate) fn inverse(map: &[u32]) -> Vec<u32> {
    let mut out = vec![0; map.len()];
    for (i, &v) in map.iter().enumerate() {
        out[v as usize] = i as u32;
    }
    out
}

pub(crate) fn tokenizer(path: &Path, vocab: usize) -> Result<tokenizer::Tokenizer> {
    let t = json(&path.join("tokenizer.json"))?;
    let model = t.get("model").ok_or_else(|| bad("missing BPE model"))?;
    if t.get("normalizer")
        .and_then(|v| v.get("type"))
        .and_then(Json::as_str)
        != Some("NFC")
    {
        return Err(bad(
            "OrcaSAQ tokenizer requires its published NFC normalizer",
        ));
    }
    if model.get("type").and_then(Json::as_str) != Some("BPE") {
        return Err(bad("OrcaSAQ requires BPE"));
    }
    let mut tokens = vec![String::new(); vocab];
    let mut put = |text: &str, id: usize| -> Result<()> {
        if id >= vocab {
            return Err(bad("token ID outside vocabulary"));
        }
        tokens[id] = text.into();
        Ok(())
    };
    for (text, id) in model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or_else(|| bad("missing vocabulary"))?
    {
        put(
            text,
            id.as_f64().ok_or_else(|| bad("invalid token ID"))? as usize,
        )?;
    }
    let mut special = vec![];
    for a in t
        .get("added_tokens")
        .and_then(Json::as_array)
        .unwrap_or(&[])
    {
        let id = a
            .get("id")
            .and_then(Json::as_f64)
            .ok_or_else(|| bad("missing added token ID"))? as usize;
        put(
            a.get("content")
                .and_then(Json::as_str)
                .ok_or_else(|| bad("missing added token text"))?,
            id,
        )?;
        // HF AddedTokens must all be matched atomically, including non-special ones.
        special.push(id as u32);
    }
    let merges = model
        .get("merges")
        .and_then(Json::as_array)
        .ok_or_else(|| bad("missing merges"))?
        .iter()
        .map(|m| {
            if let Some(a) = m.as_array() {
                if a.len() == 2 {
                    return Ok((
                        a[0].as_str().ok_or_else(|| bad("merge"))?.into(),
                        a[1].as_str().ok_or_else(|| bad("merge"))?.into(),
                    ));
                }
            }
            let (a, b) = m
                .as_str()
                .and_then(|s| s.split_once(' '))
                .ok_or_else(|| bad("invalid merge"))?;
            Ok((a.into(), b.into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let eos = tokens
        .iter()
        .position(|t| t == "<|im_end|>")
        .ok_or_else(|| bad("missing im_end"))?;
    tokenizer::Tokenizer::from_qwen3_bpe_parts(tokens, merges, special, eos as u32)
        .map_err(|e| bad(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_value_heads_map_to_tiled_and_back() {
        let map = value_map(2, 6, 2);
        assert_eq!(map, vec![0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]);
        let inv = inverse(&map);
        for (i, &v) in map.iter().enumerate() {
            assert_eq!(inv[v as usize], i as u32);
        }
    }
    #[test]
    fn exl3_detection_does_not_confuse_other_checkpoints() {
        let dir = std::env::temp_dir().join(format!("nrob-orca-detect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (config, expected) in [
            (
                r#"{"model_type":"qwen3_5","quantization_config":{"quant_method":"exl3"}}"#,
                true,
            ),
            (
                r#"{"model_type":"qwen3_5","quantization_config":{"quant_method":"fp8"}}"#,
                false,
            ),
            (r#"{"model_type":"deepseek_v4"}"#, false),
            ("{", false),
        ] {
            std::fs::write(dir.join("config.json"), config).unwrap();
            assert_eq!(detect(&dir), expected);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    #[ignore = "requires the downloaded original OrcaSAQ tokenizer"]
    fn tokenizer_matches_huggingface_oracle() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tok = tokenizer(&root.join("models/OrcaSAQ-2-27B"), 248320).unwrap();
        let cases = json(&root.join("tools/orcasaq/tokenizer-cases.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let text = case.get("text").unwrap().as_str().unwrap();
            let expected: Vec<u32> = case
                .get("ids")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as u32)
                .collect();
            assert_eq!(tok.encode(text, false).unwrap(), expected, "text: {text:?}");
        }
    }
}

pub fn load(path: &Path, devices: &[usize]) -> Result<Model> {
    load_with_adapter(path,devices,&[])
}
pub(crate) fn load_with_adapter(path: &Path, devices: &[usize], lora:&[crate::lora::Adapter]) -> Result<Model> {
    let raw = json(&path.join("config.json"))?;
    if !detect(path) {
        return Err(bad("not a Qwen3.5-family EXL3 checkpoint"));
    }
    let c = raw.get("text_config").unwrap_or(&raw);
    let (h, ff, layers, heads, kv, hd, vocab) = (
        number(c, "hidden_size")?,
        number(c, "intermediate_size")?,
        number(c, "num_hidden_layers")?,
        number(c, "num_attention_heads")?,
        number(c, "num_key_value_heads")?,
        number(c, "head_dim")?,
        number(c, "vocab_size")?,
    );
    let (nk, nv, kd, vd, conv) = (
        number(c, "linear_num_key_heads")?,
        number(c, "linear_num_value_heads")?,
        number(c, "linear_key_head_dim")?,
        number(c, "linear_value_head_dim")?,
        number(c, "linear_conv_kernel_dim")?,
    );
    if layers > 256
        || heads % kv != 0
        || nv % nk != 0
        || kd != vd
        || kd > 256
        || h > 32768
        || ff > 131072
        || vocab > 1_048_576
    {
        return Err(bad("unsupported Qwen hybrid dimensions"));
    }
    let rope = c
        .get("rope_parameters")
        .ok_or_else(|| bad("missing rope parameters"))?;
    let config = ModelConfig {
        arch: Architecture::Qwen35,
        vocab_size: vocab,
        context_length: number(c, "max_position_embeddings")?,
        embedding_dim: h,
        n_layers: layers,
        n_heads: heads,
        n_kv_heads: kv,
        head_dim: hd,
        ff_dim: ff,
        rms_eps: c.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
        rope_theta: rope.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e7) as f32,
        rope_dim: (hd as f64
            * rope
                .get("partial_rotary_factor")
                .and_then(Json::as_f64)
                .unwrap_or(0.25)) as usize,
        final_logit_softcap: None,
        embedding_scale: false,
        sliding_window: None,
        sliding_window_pattern: 1,
        sliding_window_layers: None,
        ff_dims: None,
        head_dim_swa: None,
        rope_theta_swa: None,
        rope_dim_swa: None,
        activation_sparsity_scale: None,
        recurrent_layers: None,
    };
    let tok = tokenizer(path, vocab)?;
    let device = devices.first().copied().unwrap_or(0);
    let backend = Arc::new(CudaBackend::new(device).map_err(|e| bad(e.to_string()))?);
    let mut cache_devices: Vec<Arc<dyn Backend>> = vec![backend.clone()];
    for &other in devices.iter().skip(1) {
        if other != device {
            cache_devices.push(Arc::new(
                CudaBackend::new(other).map_err(|e| bad(e.to_string()))?,
            ));
        }
    }
    let l = Loader {
        lora,
        idx: StIndex::open(path)?,
        backend: backend.clone(),
    };
    let prefix = "model.language_model";
    let qkey = format!("{prefix}.embed_tokens.qweight");
    if l.idx.info(&qkey)?.dtype != Dtype::I8 || l.idx.info(&qkey)?.shape != [vocab, h] {
        return Err(bad("invalid int8 embedding table"));
    }
    let embedding = l.idx.read(&qkey)?;
    let scales = l.idx.read_f32(&format!("{prefix}.embed_tokens.scales"))?;
    if scales.len() != vocab {
        return Err(bad("invalid embedding scale count"));
    }
    let output = l.weight("lm_head", h, vocab, None, None)?;
    let output_norm = l.tensor(&format!("{prefix}.norm.weight"), &[h], true, None)?;
    let types = c
        .get("layer_types")
        .and_then(Json::as_array)
        .ok_or_else(|| bad("missing layer types"))?;
    if types.len() != layers {
        return Err(bad("layer type count differs from layer count"));
    }
    let mut blocks = Vec::with_capacity(layers);
    let mut attention_layers = vec![];
    let vm = value_map(nk, nv, vd);
    let hm = value_map(nk, nv, 1);
    let qkv = 2 * nk * kd + nv * vd;
    let qm: Vec<_> = (0..(2 * nk * kd) as u32)
        .chain(vm.iter().map(|v| v + 2 * (nk * kd) as u32))
        .collect();
    for (i, typ) in types.iter().enumerate() {
        let p = format!("{prefix}.layers.{i}");
        let attn_norm = l.tensor(&format!("{p}.input_layernorm.weight"), &[h], true, None)?;
        let post_norm = l.tensor(
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            true,
            None,
        )?;
        let ffn_pair = FfnPair::Split {
            gate: l.weight(&format!("{p}.mlp.gate_proj"), h, ff, None, None)?,
            up: l.weight(&format!("{p}.mlp.up_proj"), h, ff, None, None)?,
        };
        let ffn_down = l.weight(&format!("{p}.mlp.down_proj"), ff, h, None, None)?;
        match typ.as_str() {
            Some("full_attention") => {
                attention_layers.push(true);
                let a = format!("{p}.self_attn");
                blocks.push(Qwen35Block::Attention {
                    attn_norm,
                    post_norm,
                    ffn_pair,
                    ffn_down,
                    attn_q: l.weight(&format!("{a}.q_proj"), h, 2 * heads * hd, None, None)?,
                    attn_k: l.weight(&format!("{a}.k_proj"), h, kv * hd, None, None)?,
                    attn_v: l.weight(&format!("{a}.v_proj"), h, kv * hd, None, None)?,
                    attn_output: l.weight(&format!("{a}.o_proj"), heads * hd, h, None, None)?,
                    attn_q_norm: l.tensor(&format!("{a}.q_norm.weight"), &[hd], true, None)?,
                    attn_k_norm: l.tensor(&format!("{a}.k_norm.weight"), &[hd], true, None)?,
                });
            }
            Some("linear_attention") => {
                attention_layers.push(false);
                let a = format!("{p}.linear_attn");
                let dense = |suffix:&str| -> Result<Tensor> {
                    let name=format!("{a}.{suffix}");
                    let base=l.tensor(&format!("{name}.weight"), &[nv,h],false,Some(&hm))?;
                    let merged=lora.iter().try_fold(base, |w, adapter| adapter.merge_dense(&name,w,backend.as_ref(),Some(&hm)))?;
                    Ok(merged.to_host())
                };
                let beta=dense("in_proj_b")?;
                let alpha=dense("in_proj_a")?;
                let ba = Tensor::from_vec([beta.data(), alpha.data()].concat(), vec![2 * nv, h]);
                let mut av = l.idx.read_f32(&format!("{a}.A_log"))?;
                if av.len() != nv {
                    return Err(bad("invalid A_log shape"));
                }
                for v in &mut av {
                    *v = -v.exp();
                }
                let av = hm.iter().map(|&j| av[j as usize]).collect();
                blocks.push(Qwen35Block::Ssm {
                    attn_norm,
                    post_norm,
                    ffn_pair,
                    ffn_down,
                    attn_qkv: l.weight(
                        &format!("{a}.in_proj_qkv"),
                        h,
                        qkv,
                        None,
                        Some(qm.clone()),
                    )?,
                    attn_gate: l.weight(
                        &format!("{a}.in_proj_z"),
                        h,
                        nv * vd,
                        None,
                        Some(vm.clone()),
                    )?,
                    ssm_out: l.weight(
                        &format!("{a}.out_proj"),
                        nv * vd,
                        h,
                        Some(inverse(&vm)),
                        None,
                    )?,
                    ssm_ba: Weight::Dense(backend.to_device(ba)),
                    ssm_a: backend.to_device(Tensor::from_vec(av, vec![nv])),
                    ssm_dt_bias: l.tensor(&format!("{a}.dt_bias"), &[nv], false, Some(&hm))?,
                    ssm_conv1d: l.tensor(
                        &format!("{a}.conv1d.weight"),
                        &[qkv, conv],
                        false,
                        Some(&qm),
                    )?,
                    ssm_norm: l.tensor(&format!("{a}.norm.weight"), &[vd], false, None)?,
                });
            }
            _ => return Err(bad("unsupported layer type")),
        }
    }
    for adapter in lora {adapter.finish()?;}
    let mut attention_index = 0;
    let cache_backends = attention_layers
        .iter()
        .map(|&attn| {
            if attn {
                let b = cache_devices[attention_index % cache_devices.len()].clone();
                attention_index += 1;
                b
            } else {
                cache_devices[0].clone()
            }
        })
        .collect();
    Ok(Model::Qwen35(Qwen35Model {
        cache_backends,
        config,
        ssm_cfg: SsmConfig {
            conv_kernel: conv,
            group_count: nk,
            inner_size: nv * vd,
            state_size: kd,
            time_step_rank: nv,
        },
        attention_layers,
        tokenizer: tok,
        blocks,
        tok_embd: None,
        packed_tok_embd: None,
        int8_tok_embd: Some((embedding, scales)),
        output_norm,
        output,
        backend,
    }))
}
