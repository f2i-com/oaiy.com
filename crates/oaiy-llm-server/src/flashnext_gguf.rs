//! Qwen3.8-Flash-Next from a GGUF (`qwen4exp`, as llama.cpp converts it; the GSQ-RCO files): each tensor in the type
//! its file gives it (a size's routed experts all but one type, its dense matrices a mix: Q2_0, the K-quants, Q4_0,
//! Q5_0, IQ4_NL, IQ4_XS, BF16), the hashed n-gram table a second shard of IQ4_NL rows. The tensors are GGML's layout
//! already (the delta nets' value heads tiled, the indexer's projection in its two parts, `ssm_a` its `-exp(A_log)`),
//! which is the layout this engine runs: no channel maps, where the EXL3 loader maps a Hugging Face checkpoint's.
use super::{bad, Config};
use ggml_quants::GgmlType;
use gguf::{GgufFile, Value};
use oaiy_engine::Result;
use std::path::Path;

pub(crate) const ARCH: &str = "qwen4exp";

/// Whether `path` is a Qwen3.8-Flash-Next GGUF (its first shard).
pub fn detect(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "gguf") && GgufFile::open_streaming(path).is_ok_and(|g| g.architecture().is_ok_and(|a| a == ARCH))
}

fn number(g: &GgufFile, key: &str) -> Result<usize> {
    let v = g.get_u64(&format!("{ARCH}.{key}")).map_err(|e| bad(format!("{key}: {e}")))?;
    if v == 0 || v > 1 << 30 {
        return Err(bad(format!("invalid {key}")));
    }
    Ok(v as usize)
}

/// An array of integers in the metadata.
pub(super) fn integers(g: &GgufFile, key: &str) -> Result<Vec<i64>> {
    use gguf::Array as A;
    let Value::Array(a) = g.get(&format!("{ARCH}.{key}")).map_err(|e| bad(format!("{key}: {e}")))? else { return Err(bad(format!("{key} is not a list"))) };
    Ok(match a {
        A::U8(v) => v.iter().map(|&x| x as i64).collect(),
        A::I8(v) => v.iter().map(|&x| x as i64).collect(),
        A::U16(v) => v.iter().map(|&x| x as i64).collect(),
        A::I16(v) => v.iter().map(|&x| x as i64).collect(),
        A::U32(v) => v.iter().map(|&x| x as i64).collect(),
        A::I32(v) => v.iter().map(|&x| x as i64).collect(),
        A::U64(v) => v.iter().map(|&x| x as i64).collect(),
        A::I64(v) => v.clone(),
        _ => return Err(bad(format!("{key} is not a list of integers"))),
    })
}

impl Config {
    /// The dimensions a GGUF's metadata gives (the same the EXL3 checkpoint's `config.json` does).
    pub(super) fn from_gguf(g: &GgufFile) -> Result<Self> {
        let layers = number(g, "block_count")?;
        let interval = number(g, "full_attention_interval")?;
        let ple = integers(g, "ple.layers")?;
        if ple.len() != 1 || ple[0] < 0 {
            return Err(bad("exactly one n-gram layer is supported"));
        }
        let ratios = integers(g, "attention.compress_ratios")?;
        let attention: Vec<bool> = (0..layers).map(|i| (i + 1) % interval == 0).collect();
        let ratio = ratios.iter().copied().find(|&r| r > 0).ok_or_else(|| bad("no attention layer compresses its keys"))? as usize;
        if ratios.len() != layers || ratios.iter().zip(&attention).any(|(&r, &a)| (r as usize) != if a { ratio } else { 0 }) {
            return Err(bad("the indexer's compression is not one ratio at every attention layer"));
        }
        let Value::Array(tokens) = g.get("tokenizer.ggml.tokens").map_err(|e| bad(e.to_string()))? else { return Err(bad("missing vocabulary")) };
        let cfg = Self {
            hidden: number(g, "embedding_length")?,
            layers,
            heads: number(g, "attention.head_count")?,
            kv_heads: number(g, "attention.head_count_kv")?,
            head_dim: number(g, "attention.key_length")?,
            rope_dim: number(g, "rope.dimension_count")?,
            rope_theta: g.get_f32(&format!("{ARCH}.rope.freq_base")).map_err(|e| bad(e.to_string()))?,
            eps: g.get_f32(&format!("{ARCH}.attention.layer_norm_rms_epsilon")).map_err(|e| bad(e.to_string()))?,
            vocab: tokens.len(),
            nk: number(g, "ssm.group_count")?,
            nv: number(g, "ssm.time_step_rank")?,
            kd: number(g, "ssm.state_size")?,
            vd: number(g, "ssm.state_size")?,
            conv: number(g, "ssm.conv_kernel")?,
            experts: number(g, "expert_count")?,
            top_k: number(g, "expert_used_count")?,
            moe_ff: number(g, "expert_feed_forward_length")?,
            shared_ff: number(g, "expert_shared_feed_forward_length")?,
            streams: number(g, "hyper_connection.count")?,
            attention,
            ple_layer: ple[0] as usize,
            // (each n-gram's heads' rows side by side)
            ple_dim: (number(g, "ple.ngram_size")? - 1) * number(g, "ple.heads_per_ngram")? * number(g, "embedding_length_per_layer_input")?,
            ple_kernel: number(g, "ple.conv_kernel")?,
            ngram: number(g, "ple.ngram_size")?,
            heads_per_ngram: number(g, "ple.heads_per_ngram")?,
            ple_eos: number(g, "ple.eos_token_id")? as u32,
            context_length: number(g, "context_length")?,
            index_heads: number(g, "attention.indexer.head_count")?,
            index_dim: number(g, "attention.indexer.key_length")?,
            index_budget: number(g, "attention.indexer.top_k")?,
            index_ratio: ratio,
        };
        if number(g, "ssm.inner_size")? != cfg.nv * cfg.vd || number(g, "attention.value_length")? != cfg.head_dim {
            return Err(bad("unsupported dimensions"));
        }
        cfg.checked()
    }
}

/// A GGUF's tensors by the shapes PyTorch gives them (GGUF's, reversed: `[out, in]` for a matrix, row-major).
pub(super) struct Source<'a> {
    pub g: &'a GgufFile,
}

impl Source<'_> {
    pub fn info(&self, name: &str) -> Result<&gguf::TensorInfo> {
        self.g.tensor_by_name(name).ok_or_else(|| bad(format!("missing {name}")))
    }

    /// `name`'s type, checked to be of `shape`.
    pub fn dtype(&self, name: &str, shape: &[usize]) -> Result<GgmlType> {
        let info = self.info(name)?;
        let stored: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
        if stored != shape {
            return Err(bad(format!("{name}: of shape {stored:?}, not {shape:?}")));
        }
        Ok(info.dtype)
    }

    /// `name`'s values (of `shape`), whatever its type.
    pub fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let dtype = self.dtype(name, shape)?;
        let bytes = self.g.tensor_bytes(self.info(name)?).map_err(|e| bad(format!("{name}: {e}")))?;
        let mut out = vec![0f32; shape.iter().product()];
        // (a row is a whole number of its type's blocks: the tensor's values are its rows', one after another)
        ggml_quants::dequantize(dtype, &bytes, &mut out).map_err(|e| bad(format!("{name}: {e}")))?;
        Ok(out)
    }
}

#[cfg(all(test, feature = "webgpu"))]
mod tests {
    use super::super::*;
    use super::Source;

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += *x as f64 * *y as f64;
            aa += *x as f64 * *x as f64;
            bb += *y as f64 * *y as f64;
        }
        dot / (aa.sqrt() * bb.sqrt()).max(1e-30)
    }

    /// A GGUF's tensors are the EXL3 checkpoint's of the same model (`FLASHNEXT_GGUF` its first shard, `FLASHNEXT_EXL3`
    /// the checkpoint's directory): the dimensions equal; each tensor neither quantizes (the norms, the delta nets'
    /// decays, biases and convolutions, the routers, the hyper-connections) the same values, as this loader reads it
    /// and as the EXL3 loader maps the Hugging Face layout; each quantized matrix's product with a random vector near
    /// the EXL3 one's (two quantizations of one matrix). `FLASHNEXT_GGUF_LAYERS`: how many layers to hold up (4).
    #[test]
    #[ignore = "needs a Flash-Next GGUF (FLASHNEXT_GGUF) and its EXL3 checkpoint (FLASHNEXT_EXL3)"]
    fn a_ggufs_tensors_are_the_exl3_checkpoints() -> Result<()> {
        let (Ok(gguf_path), Ok(exl3)) = (std::env::var("FLASHNEXT_GGUF"), std::env::var("FLASHNEXT_EXL3")) else { return Ok(()) };
        let exl3 = Path::new(&exl3);
        let g = gguf::GgufFile::open(&gguf_path).map_err(|e| bad(e.to_string()))?;
        let src = Source { g: &g };
        let cfg = Config::from_gguf(&g)?;
        let want = Config::read(exl3)?;
        assert_eq!(format!("{cfg:?}"), format!("{want:?}"), "the GGUF's dimensions are the checkpoint's");
        let idx = StIndex::open(exl3)?;
        let host = |name: &str, numel: usize, add_one: bool, map: Option<&[u32]>| -> Result<Vec<f32>> {
            let mut data = idx.read_f32(name)?;
            assert_eq!(data.len(), numel, "{name}");
            if add_one { for v in &mut data { *v += 1.0; } }
            if let Some(map) = map {
                let width = data.len() / map.len();
                let from = data.clone();
                for (i, &j) in map.iter().enumerate() {
                    data[i * width..(i + 1) * width].copy_from_slice(&from[j as usize * width..(j as usize + 1) * width]);
                }
            }
            Ok(data)
        };
        let (h, nk, nv, kd, vd) = (cfg.hidden, cfg.nk, cfg.nv, cfg.kd, cfg.vd);
        let width = cfg.streams * h;
        let vm = value_map(nk, nv, vd);
        let hm = value_map(nk, nv, 1);
        let qkv = 2 * nk * kd + nv * vd;
        let qm: Vec<u32> = (0..(2 * nk * kd) as u32).chain(vm.iter().map(|v| v + 2 * (nk * kd) as u32)).collect();
        let seed = std::cell::Cell::new(0x2545_f491_4f6c_dd1du64);
        let next = || {
            let mut s = seed.get();
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            seed.set(s);
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
        };
        let mut worst_same = 1f64;
        let mut same = |what: &str, got: &[f32], want: &[f32]| {
            let c = cosine(got, want);
            let off = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let size = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            eprintln!("  {what}: cosine {c:.6}, the worst difference {off:.2e} of {size:.2e}");
            worst_same = worst_same.min(c);
            assert!(c > 0.9999, "{what}: cosine {c}");
        };
        // a quantized matrix `k -> n`: its product with a random vector against the EXL3 matrix's (its channels mapped)
        let near = |what: &str, gguf_name: &str, exl3_name: &str, k: usize, n: usize, input: Option<Vec<u32>>, output: Option<Vec<u32>>| -> Result<f64> {
            let w = src.values(gguf_name, &[n, k])?;
            let x: Vec<f32> = (0..k).map(|_| next()).collect();
            let got: Vec<f32> = w.chunks_exact(k).map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
            let packed = ggml_rs_wgpu::exl3::exl3_cpu(exl3_data(&idx, exl3_name, k, n, input, output)?).map_err(bad)?;
            let want = packed.linear(&Tensor::from_vec(x, vec![1, k]));
            let c = cosine(&got, want.data());
            eprintln!("  {what} ({:?}): cosine {c:.4}", src.info(gguf_name)?.dtype);
            Ok(c)
        };
        let p = "model.language_model";
        let layers: usize = std::env::var("FLASHNEXT_GGUF_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let mut least = 1f64;
        for i in 0..layers.min(cfg.layers) {
            let (b, lp) = (format!("blk.{i}"), format!("{p}.layers.{i}"));
            eprintln!("layer {i} ({}):", if cfg.attention[i] { "attention" } else { "a delta net" });
            if cfg.attention[i] {
                let a = format!("{lp}.self_attn");
                for (gn, en, n) in [("attn_q", "q_proj", 2 * cfg.heads * cfg.head_dim), ("attn_k", "k_proj", cfg.kv_heads * cfg.head_dim), ("attn_v", "v_proj", cfg.kv_heads * cfg.head_dim)] {
                    least = least.min(near(gn, &format!("{b}.{gn}.weight"), &format!("{a}.{en}"), h, n, None, None)?);
                }
                least = least.min(near("attn_output", &format!("{b}.attn_output.weight"), &format!("{a}.o_proj"), cfg.heads * cfg.head_dim, h, None, None)?);
                same("attn_q_norm", &src.values(&format!("{b}.attn_q_norm.weight"), &[cfg.head_dim])?, &host(&format!("{a}.q_norm.weight"), cfg.head_dim, true, None)?);
                same("attn_k_norm", &src.values(&format!("{b}.attn_k_norm.weight"), &[cfg.head_dim])?, &host(&format!("{a}.k_norm.weight"), cfg.head_dim, true, None)?);
                // the indexer's projection: its query heads, then its key
                let (iq, ik) = (cfg.index_heads * cfg.index_dim, cfg.index_dim);
                let mut got = src.values(&format!("{b}.indexer.q_proj.weight"), &[iq, h])?;
                got.extend(src.values(&format!("{b}.indexer.k_proj.weight"), &[ik, h])?);
                let x: Vec<f32> = (0..h).map(|_| next()).collect();
                let got: Vec<f32> = got.chunks_exact(h).map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
                let packed = ggml_rs_wgpu::exl3::exl3_cpu(exl3_data(&idx, &format!("{a}.indexer.index_qk_proj"), h, iq + ik, None, None)?).map_err(bad)?;
                let c = cosine(&got, packed.linear(&Tensor::from_vec(x, vec![1, h])).data());
                eprintln!("  indexer q and k (BF16): cosine {c:.4}");
                least = least.min(c);
                same("indexer q_norm", &src.values(&format!("{b}.indexer.q_norm.weight"), &[ik])?, &host(&format!("{a}.indexer.q_layernorm.weight"), ik, true, None)?);
                same("indexer k_norm", &src.values(&format!("{b}.indexer.k_norm.weight"), &[ik])?, &host(&format!("{a}.indexer.k_layernorm.weight"), ik, true, None)?);
            } else {
                let a = format!("{lp}.linear_attn");
                least = least.min(near("attn_qkv", &format!("{b}.attn_qkv.weight"), &format!("{a}.in_proj_qkv"), h, qkv, None, Some(qm.clone()))?);
                least = least.min(near("attn_gate", &format!("{b}.attn_gate.weight"), &format!("{a}.in_proj_z"), h, nv * vd, None, Some(vm.clone()))?);
                least = least.min(near("ssm_out", &format!("{b}.ssm_out.weight"), &format!("{a}.out_proj"), nv * vd, h, Some(inverse(&vm)), None)?);
                same("ssm_beta", &src.values(&format!("{b}.ssm_beta.weight"), &[nv, h])?, &host(&format!("{a}.in_proj_b.weight"), nv * h, false, Some(&hm))?);
                same("ssm_alpha", &src.values(&format!("{b}.ssm_alpha.weight"), &[nv, h])?, &host(&format!("{a}.in_proj_a.weight"), nv * h, false, Some(&hm))?);
                let mut av = host(&format!("{a}.A_log"), nv, false, None)?;
                for v in &mut av { *v = -v.exp(); }
                let av: Vec<f32> = hm.iter().map(|&j| av[j as usize]).collect();
                same("ssm_a", &src.values(&format!("{b}.ssm_a"), &[nv])?, &av);
                same("ssm_dt.bias", &src.values(&format!("{b}.ssm_dt.bias"), &[nv])?, &host(&format!("{a}.dt_bias"), nv, false, Some(&hm))?);
                same("ssm_conv1d", &src.values(&format!("{b}.ssm_conv1d.weight"), &[qkv, cfg.conv])?, &host(&format!("{a}.conv1d.weight"), qkv * cfg.conv, false, Some(&qm))?);
                same("ssm_norm", &src.values(&format!("{b}.ssm_norm.weight"), &[vd])?, &host(&format!("{a}.norm.weight"), vd, false, None)?);
            }
            let m = format!("{lp}.mlp");
            same("ffn_gate_inp", &src.values(&format!("{b}.ffn_gate_inp.weight"), &[cfg.experts, h])?, &host(&format!("{m}.gate.weight"), cfg.experts * h, false, None)?);
            same("ffn_gate_inp_shexp", &src.values(&format!("{b}.ffn_gate_inp_shexp.weight"), &[h])?, &host(&format!("{m}.shared_expert_gate.weight"), h, false, None)?);
            for (gn, en, k, n) in [("ffn_gate_shexp", "gate_proj", h, cfg.shared_ff), ("ffn_up_shexp", "up_proj", h, cfg.shared_ff), ("ffn_down_shexp", "down_proj", cfg.shared_ff, h)] {
                least = least.min(near(gn, &format!("{b}.{gn}.weight"), &format!("{m}.shared_expert.{en}"), k, n, None, None)?);
            }
            // two of the routed experts: an expert's rows of the layer's one tensor
            for e in [0usize, cfg.experts - 1] {
                for (gn, en, k, n) in [("ffn_gate_exps", "gate_proj", h, cfg.moe_ff), ("ffn_up_exps", "up_proj", h, cfg.moe_ff), ("ffn_down_exps", "down_proj", cfg.moe_ff, h)] {
                    let name = format!("{b}.{gn}.weight");
                    let dtype = src.dtype(&name, &[cfg.experts, n, k])?;
                    let bytes = g.tensor_bytes(src.info(&name)?).map_err(|e| bad(e.to_string()))?;
                    let per = n * k / dtype.block_size() * dtype.type_size();
                    let mut w = vec![0f32; n * k];
                    ggml_quants::dequantize(dtype, &bytes[e * per..(e + 1) * per], &mut w).map_err(|e| bad(e.to_string()))?;
                    let x: Vec<f32> = (0..k).map(|_| next()).collect();
                    let got: Vec<f32> = w.chunks_exact(k).map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
                    let packed = ggml_rs_wgpu::exl3::exl3_cpu(exl3_data(&idx, &format!("{m}.experts.{e}.{en}"), k, n, None, None)?).map_err(bad)?;
                    let c = cosine(&got, packed.linear(&Tensor::from_vec(x, vec![1, k])).data());
                    eprintln!("  expert {e}'s {gn} ({dtype:?}): cosine {c:.4}");
                    least = least.min(c);
                }
            }
            for (gn, en) in [("hc_attn", "attn_hyper_connection"), ("hc_ffn", "mlp_hyper_connection")] {
                let (hp, rank) = (format!("{lp}.{en}"), src.info(&format!("{b}.{gn}_down.weight"))?.shape[1] as usize);
                same(&format!("{gn}_down"), &src.values(&format!("{b}.{gn}_down.weight"), &[rank, width])?, &host(&format!("{hp}.input_mix_weight_down.weight"), rank * width, false, None)?);
                same(&format!("{gn}_up"), &src.values(&format!("{b}.{gn}_up.weight"), &[width, rank])?, &host(&format!("{hp}.input_mix_weight_up.weight"), width * rank, false, None)?);
                same(&format!("{gn}_inject"), &src.values(&format!("{b}.{gn}_inject.weight"), &[cfg.streams, width])?, &host(&format!("{hp}.block_inject_weight.weight"), cfg.streams * width, false, None)?);
                same(&format!("{gn}_norm"), &src.values(&format!("{b}.{gn}_norm.weight"), &[width])?, &host(&format!("{hp}.hc_norm.weight"), width, true, None)?);
            }
            if i == cfg.ple_layer {
                let pp = format!("{lp}.ple");
                same("ple_value", &src.values(&format!("{b}.ple_value.weight"), &[h, cfg.ple_dim])?, &host(&format!("{pp}.value_proj.weight"), h * cfg.ple_dim, false, None)?);
                let key = src.values(&format!("{b}.ple_key.weight"), &[width, cfg.ple_dim])?;
                let c = cosine(&key, &host(&format!("{pp}.key_proj.weight"), width * cfg.ple_dim, false, None)?);
                eprintln!("  ple_key ({:?}): cosine {c:.4}", src.info(&format!("{b}.ple_key.weight"))?.dtype);
                least = least.min(c);
                for n in ["norm_key", "norm_query", "norm_conv"] {
                    same(&format!("ple_{n}"), &src.values(&format!("{b}.ple_{n}.weight"), &[width])?, &host(&format!("{pp}.{n}.weight"), width, true, None)?);
                }
                same("ple_conv1d", &src.values(&format!("{b}.ple_conv1d.weight"), &[width, cfg.ple_kernel])?, &host(&format!("{pp}.conv1d.weight"), width * cfg.ple_kernel, false, None)?);
            }
        }
        let rank = src.info("output_hc_down.weight")?.shape[1] as usize;
        same("output_hc_down", &src.values("output_hc_down.weight", &[rank, width])?, &host(&format!("{p}.hyper_connection_mixer.input_mix_weight_down.weight"), rank * width, false, None)?);
        same("output_hc_up", &src.values("output_hc_up.weight", &[width, rank])?, &host(&format!("{p}.hyper_connection_mixer.input_mix_weight_up.weight"), width * rank, false, None)?);
        same("output_hc_norm", &src.values("output_hc_norm.weight", &[width])?, &host(&format!("{p}.hyper_connection_mixer.hc_norm.weight"), width, true, None)?);
        // a few tokens' embeddings (the GGUF's a K-quant of the checkpoint's)
        let embed = idx.info(&format!("{p}.embed_tokens.weight"))?.clone();
        let all = src.dtype("token_embd.weight", &[cfg.vocab, h])?;
        let bytes = g.tensor_bytes(src.info("token_embd.weight")?).map_err(|e| bad(e.to_string()))?;
        let per = h / all.block_size() * all.type_size();
        let file = File::open(idx.shard_path(embed.shard))?;
        for t in [0usize, 1000, 151_645, cfg.vocab - 1] {
            let mut got = vec![0f32; h];
            ggml_quants::dequantize(all, &bytes[t * per..(t + 1) * per], &mut got).map_err(|e| bad(e.to_string()))?;
            let mut buf = vec![0u8; h * embed.dtype.size()];
            read_at(&file, &mut buf, embed.start + (t * h * embed.dtype.size()) as u64)?;
            let want: Vec<f32> = match embed.dtype {
                Dtype::F32 => buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                Dtype::BF16 => buf.chunks_exact(2).map(|c| dsv41::formats::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                _ => buf.chunks_exact(2).map(|c| dsv41::formats::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            };
            let c = cosine(&got, &want);
            eprintln!("token {t}'s embedding ({all:?}): cosine {c:.4}");
            least = least.min(c);
        }
        eprintln!("unquantized tensors: the least cosine {worst_same:.6}; quantized ones against EXL3's: the least {least:.4}");
        assert!(least > 0.8, "a quantized matrix is not its EXL3 one: cosine {least}");
        Ok(())
    }
}
