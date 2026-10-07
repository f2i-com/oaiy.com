//! Read OrcaSAQ2's original sharded EXL3 checkpoint. No Python runtime/conversion.
//!
//! On CUDA (`load`, `load_with_adapter`) its projections stay packed on the card (`Exl3Matrix`), PEFT adapters
//! applied; elsewhere (`load_portable`) they are whatever the caller makes of each `Exl3Data`: packed on any GPU
//! through WebGPU, or decoded on the CPU (`ggml_rs_wgpu::exl3`).
use dsv41::safetensors::{Dtype, StIndex};
use ggml_rs::{exl3::{Exl3Data, PackedLinear}, Backend, Tensor};
use llama_rs::{
    loader::{FfnPair, Weight},
    qwen35::{Qwen35Block, Qwen35Model, SsmConfig},
    Architecture, Model, ModelConfig,
};
use oaiy_engine::{json::Json, Error, Result};
use std::{path::Path, sync::Arc};

fn bad(s: impl Into<String>) -> Error {
    Error::Format(s.into())
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

/// Makes a projection of its packed EXL3 weights: on the card the model runs on.
type Packer<'a> = &'a dyn Fn(Exl3Data) -> std::result::Result<Arc<dyn PackedLinear>, String>;

/// The PEFT adapters applied as the weights load, together and in order: the CUDA build's; a portable build has none.
struct Adapters<'a> {
    list: std::marker::PhantomData<&'a ()>,
}
impl<'a> Adapters<'a> {
    fn none() -> Self {
        Self { list: std::marker::PhantomData }
    }
    /// Each adapter wraps what the one before made (an adapter without this weight leaves it).
    fn wrap(&self, name: &str, w: Weight, backend: &Arc<dyn Backend>, input: Option<&[u32]>, output: Option<&[u32]>) -> Result<Weight> {
        {
            let _ = (name, backend, input, output);
            Ok(w)
        }
    }
    fn merge_dense(&self, name: &str, w: Tensor, backend: &dyn Backend, map: Option<&[u32]>) -> Result<Tensor> {
        {
            let _ = (name, backend, map);
            Ok(w)
        }
    }
    /// Every adapter's targets were found.
    fn finish(&self) -> Result<()> {
        Ok(())
    }
}

struct Loader<'a> {
    adapters: &'a Adapters<'a>,
    idx: StIndex,
    backend: Arc<dyn Backend>,
    packed: Packer<'a>,
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
        let base = Weight::Packed((self.packed)(data).map_err(bad)?);
        self.adapters.wrap(name, base, &self.backend, input.as_deref(), output.as_deref())
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
        let dir = std::env::temp_dir().join(format!("oaiy-orca-detect-{}", std::process::id()));
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

/// OrcaSAQ without CUDA: its tensors on `backend`, each packed projection as `packed` makes it (on any GPU through
/// WebGPU, else on the CPU), no PEFT adapters, its attention cache beside it. (A CUDA build loads it on the card.)
pub(crate) fn load_portable(path: &Path, backend: Arc<dyn Backend>, packed: Packer<'_>) -> Result<Model> {
    build(path, backend.clone(), vec![backend], packed, &Adapters::none())
}

fn build(path: &Path, backend: Arc<dyn Backend>, cache_devices: Vec<Arc<dyn Backend>>, packed: Packer<'_>, adapters: &Adapters<'_>) -> Result<Model> {
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
    let l = Loader {
        adapters,
        idx: StIndex::open(path)?,
        backend: backend.clone(),
        packed,
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
                    let merged=adapters.merge_dense(&name,base,backend.as_ref(),Some(&hm))?;
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
    adapters.finish()?;
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
        chain: Default::default(),
        mtp: None,
    }))
}
