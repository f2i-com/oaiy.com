//! DeepSeek-V4.1 from a GGUF: llama.cpp's `deepseek41` layout (as the GSQ-RCO quantizations ship it: their routed
//! experts in 1- and 2-bit grid types, the trunk in K-quants and Q8_0, the rest BF16 and F32), its Engram tables a GGUF
//! of their own beside the backbone's shards (`deepseek41-engram`). The tensors are the checkpoint's (its floats the
//! safetensors checkpoint's to the bit, its dimensions reversed), so they are served under the names the safetensors
//! loader asks for ([`open`]: an index of where each lies in the files, nothing read): the trunk's quantized matrices as
//! GGML blocks ([`crate::dsv41::linear::Weight::Ggml`]), each layer's routed experts as one tensor a projection
//! (`layers.{l}.ffn.experts.{w1,w2,w3}`, `[experts, n, k]`) from which a record takes an expert's slices
//! ([`crate::dsv41::expert::ExpertFormat::Gguf`]), the Engram tables' rows as IQ4_NL blocks.

use std::path::{Path, PathBuf};

use gguf::GgufFile;
use oaiy_engine::safetensors::{Dtype, StIndex, TensorInfo};
use oaiy_engine::{Error, Result};

use crate::dsv41::config::Config;
use crate::dsv41::tokenizer::Tokenizer;

/// The backbone's architecture, and the Engram tables' file's.
pub const ARCH: &str = "deepseek41";
pub const ENGRAM_ARCH: &str = "deepseek41-engram";

/// The `.gguf` files at `path`: the file itself, or a directory's, in name order.
fn ggufs(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(path)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf"))).collect())
        .unwrap_or_default();
    found.sort();
    found
}

/// A GGUF file's architecture (its header alone read).
fn architecture(path: &Path) -> Option<String> {
    let g = GgufFile::open_streaming(path).ok()?;
    g.architecture().ok().map(str::to_string)
}

/// The first shard of the DeepSeek-V4.1 GGUF at `path` (that file, or the lowest-named of a directory's whose
/// architecture is `deepseek41`), or None.
pub fn backbone(path: &Path) -> Option<PathBuf> {
    // (a later shard names the first's architecture too, its reader following the splits: the lowest name is the first)
    ggufs(path).into_iter().find(|p| architecture(p).as_deref() == Some(ARCH))
}

/// The Engram tables' GGUF beside the backbone at `first` (in its directory).
fn engram_tables(first: &Path) -> Option<PathBuf> {
    ggufs(first.parent()?).into_iter().find(|p| architecture(p).as_deref() == Some(ENGRAM_ARCH))
}

/// A DeepSeek-V4.1 GGUF opened: its tensors under the safetensors loader's names (the Engram's n-gram hashing under
/// `engram.*`, its pad token `engram_pad`), its hyperparameters and its tokenizer.
pub struct Opened {
    pub index: StIndex,
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub engram_pad: i64,
}

/// The name the safetensors loader gives a GGUF tensor of `deepseek41`'s (llama.cpp's names), or None for one it does
/// not know.
fn renamed(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("embed.weight".into()),
        "output.weight" => return Some("head.weight".into()),
        "output_norm.weight" => return Some("norm.weight".into()),
        // (the Engram's n-gram hashing, which a safetensors checkpoint has from `oracle.py`'s file beside it)
        "engram_token_map.weight" => return Some("engram.token_map".into()),
        "engram_multipliers.weight" => return Some("engram.multipliers".into()),
        "engram_primes.weight" => return Some("engram.primes".into()),
        "engram_offsets.weight" => return Some("engram.offsets".into()),
        _ => {}
    }
    let (layer, part) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let to = match part {
        "attn_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "hc_attn_fn.weight" => "hc_attn_fn",
        "hc_attn_base.weight" => "hc_attn_base",
        "hc_attn_scale.weight" => "hc_attn_scale",
        "hc_ffn_fn.weight" => "hc_ffn_fn",
        "hc_ffn_base.weight" => "hc_ffn_base",
        "hc_ffn_scale.weight" => "hc_ffn_scale",
        "attn_q_a.weight" => "attn.wq_a.weight",
        "attn_q_a_norm.weight" => "attn.q_norm.weight",
        "attn_q_b.weight" => "attn.wq_b.weight",
        "attn_kv.weight" => "attn.wkv.weight",
        "attn_kv_a_norm.weight" => "attn.kv_norm.weight",
        "attn_output_a.weight" => "attn.wo_a.weight",
        "attn_output_b.weight" => "attn.wo_b.weight",
        "attn_sinks.weight" => "attn.attn_sink",
        "attn_compressor_kv.weight" => "attn.compressor.wkv.weight",
        "attn_compressor_gate.weight" => "attn.compressor.wgate.weight",
        "attn_compressor_norm.weight" => "attn.compressor.norm.weight",
        "indexer.attn_q_b.weight" => "attn.indexer.wq_b.weight",
        "indexer.proj.weight" => "attn.indexer.weights_proj.weight",
        "indexer.attn_k.weight" => "attn.indexer.wk.weight",
        "indexer.k_norm.weight" => "attn.indexer.k_norm.weight",
        "ffn_gate_inp.weight" => "ffn.gate.weight",
        "exp_probs_b.bias" => "ffn.gate.bias",
        "ffn_gate_shexp.weight" => "ffn.shared_experts.w1.weight",
        "ffn_up_shexp.weight" => "ffn.shared_experts.w3.weight",
        "ffn_down_shexp.weight" => "ffn.shared_experts.w2.weight",
        "ffn_gate_exps.weight" => "ffn.experts.w1",
        "ffn_up_exps.weight" => "ffn.experts.w3",
        "ffn_down_exps.weight" => "ffn.experts.w2",
        "engram_q.weight" => "engram.q_weight",
        "engram_k.weight" => "engram.k_weight",
        "engram_wkv.weight" => "engram.wkv.weight",
        "engram_embd.weight" => "engram.embed.weight",
        _ => return None,
    };
    Some(format!("layers.{layer}.{to}"))
}

/// `t`'s type as the index keeps it: the float types as themselves, a block type by its number.
fn dtype(t: gguf::GgmlType) -> Dtype {
    use gguf::GgmlType as G;
    match t {
        G::F32 => Dtype::F32,
        G::F16 => Dtype::F16,
        G::BF16 => Dtype::BF16,
        G::I32 => Dtype::I32,
        G::I64 => Dtype::I64,
        other => Dtype::Ggml(other as u32),
    }
}

/// The DeepSeek-V4.1 GGUF at `path` (its first shard, or a directory holding its shards and its Engram tables' file).
pub fn open(path: &Path) -> Result<Opened> {
    let bad = |m: String| Error::Format(format!("{}: {m}", path.display()));
    let first = backbone(path).ok_or_else(|| bad("no DeepSeek-V4.1 GGUF (architecture deepseek41)".into()))?;
    let g = GgufFile::open_streaming(&first).map_err(|e| bad(e.to_string()))?;
    let shards = gguf::reader::split_shard_paths(&first, g.n_shards()).filter(|_| g.n_shards() > 1).unwrap_or_else(|| vec![first.clone()]);
    let tokenizer = Tokenizer::from_gguf(&g)?;
    let config = Config::from_gguf(&g, tokenizer.vocab_size())?;
    let mut tensors = Vec::with_capacity(g.tensors().len() + 2);
    let mut unknown = Vec::new();
    let mut add = |g: &GgufFile, shard_base: usize, tensors: &mut Vec<(String, TensorInfo)>| {
        for t in g.tensors() {
            let Some(name) = renamed(&t.name) else {
                unknown.push(t.name.clone());
                continue;
            };
            let shape: Vec<usize> = t.shape.iter().rev().map(|&d| d as usize).collect();
            let info = TensorInfo { shard: shard_base + g.shard_of(t), dtype: dtype(t.dtype), shape, start: g.tensor_data_offset(t) as u64, nbytes: t.nbytes() };
            tensors.push((name, info));
        }
    };
    add(&g, 0, &mut tensors);
    let mut files = shards;
    if let Some(tables) = engram_tables(&first) {
        let e = GgufFile::open_streaming(&tables).map_err(|e| bad(format!("its Engram tables: {e}")))?;
        let base = files.len();
        add(&e, base, &mut tensors);
        files.push(tables);
    }
    if !unknown.is_empty() {
        return Err(bad(format!("{} tensors of no known part ({})", unknown.len(), unknown.iter().take(4).cloned().collect::<Vec<_>>().join(", "))));
    }
    if !config.engram_layer_ids.is_empty() && !tensors.iter().any(|(n, _)| n.ends_with("engram.embed.weight")) {
        return Err(bad("its Engram tables' GGUF (architecture deepseek41-engram) is not beside it".into()));
    }
    let engram_pad = g.get_u64(&format!("{ARCH}.engram_pad_token_id")).map_or(2, |v| v as i64);
    Ok(Opened { index: StIndex::from_parts(files, tensors, Vec::new()), config, tokenizer, engram_pad })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DeepSeek-V4.1 GGUF (DSV41_GGUF: its directory), its tokenizer against the safetensors checkpoint's
    /// (DSV41_MODEL: a directory with its `tokenizer.json`): the same ids for the tests' long text and for every special
    /// token, so its Engram's n-gram table (keyed by ids) is the checkpoint's.
    #[test]
    #[ignore = "needs a DeepSeek-V4.1 GGUF (DSV41_GGUF) and the safetensors checkpoint's tokenizer.json (DSV41_MODEL)"]
    fn a_ggufs_tokenizer_is_the_checkpoints() {
        let (Some(gguf), Some(model)) = (std::env::var_os("DSV41_GGUF"), std::env::var_os("DSV41_MODEL")) else { return };
        let opened = open(Path::new(&gguf)).expect("the GGUF");
        let want = Tokenizer::load(Path::new(&model)).expect("tokenizer.json");
        assert_eq!(opened.tokenizer.vocab_size(), want.vocab_size());
        let text = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv41-prompt.md")).replace("\r\n", "\n");
        let chat = "<｜begin▁of▁sentence｜><｜User｜>Hello there, 你好! 12345 <think>\n</think>\n<｜Assistant｜>";
        for t in [text.as_str(), chat] {
            assert_eq!(opened.tokenizer.encode(t), want.encode(t), "the GGUF's ids of a text");
        }
        for id in 0..want.vocab_size() as u32 {
            assert_eq!(opened.tokenizer.token_bytes(id), want.token_bytes(id), "token {id}'s bytes");
        }
        let cfg = &opened.config;
        assert_eq!((cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.n_activated_experts), (40, 5120, 64, 512, 6));
        for name in ["layers.0.attn.wq_a.weight", "layers.0.attn.wo_a.weight", "layers.12.ffn.experts.w1", "layers.1.engram.embed.weight", "embed.weight"] {
            assert!(opened.index.get(name).is_some(), "{name} served");
        }
        // its n-gram hashing the one `oracle.py` wrote for the checkpoint (where its file is beside the tokenizer)
        if let Ok(meta) = StIndex::open_file(&Path::new(&model).join("engram_meta.safetensors")) {
            for part in ["token_map", "multipliers", "primes", "offsets"] {
                assert_eq!(opened.index.read_i64(&format!("engram.{part}")).unwrap(), meta.read_i64(part).unwrap(), "the n-gram hashing's {part}");
            }
            assert_eq!(meta.read_i64("pad_id").unwrap(), vec![opened.engram_pad]);
        }
    }
}
