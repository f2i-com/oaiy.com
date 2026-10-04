//! PEFT LoRA/rsLoRA inference alongside the original packed Orca weights, and
//! llama.cpp GGUF LoRAs for Qwen3.8-Flash-Next (read into Hugging Face names and
//! order). Adapter tensors stay separate; only already-dense tiny projections are
//! combined in device memory. Never rewrite or requantize the base checkpoint.
use dsv41::safetensors::{Dtype, StIndex};
use ggml_rs::{exl3::PackedLinear, Backend, Tensor};
use llama_rs::loader::Weight;
use oaiy_engine::{json::Json, Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
    sync::Arc,
};

fn bad(s: impl Into<String>) -> Error {
    Error::Format(format!("LoRA: {}", s.into()))
}

/// The model an adapter is for.
pub(crate) enum Base {
    /// Qwen3.8-27B (OrcaSAQ-2-27B): PEFT adapters.
    Qwen27B,
    /// Qwen3.8-Flash-Next: PEFT adapters, or llama.cpp GGUF LoRAs (whose Gated DeltaNet
    /// value heads llama.cpp stores reordered; these are its head counts and widths).
    FlashNext(VHeads),
}

/// Gated DeltaNet head counts and widths, to undo llama.cpp's value-head order.
pub(crate) struct VHeads {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
}

impl Base {
    fn accepts(&self, base: &str) -> bool {
        match self {
            Self::Qwen27B => base == "Qwen/Qwen3.8-27B" || base.ends_with("/Qwen/Qwen3.8-27B") || base == "orcarouter/OrcaSAQ-2-27B",
            Self::FlashNext(_) => base == "Qwen/Qwen3.8-Flash-Next" || base.ends_with("/Qwen3.8-Flash-Next"),
        }
    }
    fn names(&self) -> &'static str {
        match self {
            Self::Qwen27B => "Qwen/Qwen3.8-27B or orcarouter/OrcaSAQ-2-27B",
            Self::FlashNext(_) => "Qwen/Qwen3.8-Flash-Next",
        }
    }
}

struct Config {
    rank: usize,
    scale: f32,
}
impl Config {
    #[cfg(test)]
    fn parse(c: &Json) -> Result<Self> {
        Self::parse_for(c, &Base::Qwen27B)
    }
    fn parse_for(c: &Json, model: &Base) -> Result<Self> {
        if c.get("peft_type").and_then(Json::as_str) != Some("LORA") {
            return Err(bad("adapter must use PEFT LORA"));
        }
        let base = c
            .get("base_model_name_or_path")
            .and_then(Json::as_str)
            .unwrap_or("")
            .replace('\\', "/");
        if !model.accepts(&base) {
            return Err(bad(format!("adapter must identify {} as its base", model.names())));
        }
        for key in [
            "use_dora",
            "use_qalora",
            "lora_bias",
            "fan_in_fan_out",
            "ensure_weight_tying",
        ] {
            if c.get(key)
                .is_some_and(|v| !matches!(v, Json::Null) && v.as_bool() != Some(false))
            {
                return Err(bad(format!("unsupported {key}")));
            }
        }
        for key in [
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
            "layer_replication",
            "target_parameters",
            "trainable_token_indices",
            "alora_invocation_tokens",
            "use_bdlora",
            "arrow_config",
            "corda_config",
            "lora_ga_config",
            "monteclora_config",
            "velora_config",
            "megatron_config",
        ] {
            if c.get(key).is_some_and(|v| {
                !matches!(v, Json::Null)
                    && !v.as_array().is_some_and(|a| a.is_empty())
                    && !v.as_object().is_some_and(|o| o.is_empty())
            }) {
                return Err(bad(format!("unsupported nonempty {key}")));
            }
        }
        if c.get("bias")
            .and_then(Json::as_str)
            .is_some_and(|v| v != "none")
        {
            return Err(bad("trained base biases are unsupported"));
        }
        let rank = c.get("r").and_then(Json::as_f64).unwrap_or(0.0);
        let alpha = c
            .get("lora_alpha")
            .and_then(Json::as_f64)
            .unwrap_or(f64::NAN);
        if !rank.is_finite()
            || rank.fract() != 0.0
            || !(1.0..=1024.0).contains(&rank)
            || !alpha.is_finite()
            || alpha < 0.0
        {
            return Err(bad("invalid rank or alpha"));
        }
        let rs = match c.get("use_rslora") {
            None | Some(Json::Null) => false,
            Some(v) => v.as_bool().ok_or_else(|| bad("invalid use_rslora"))?,
        };
        let scale = (alpha / if rs { rank.sqrt() } else { rank }) as f32;
        if !scale.is_finite() {
            return Err(bad("scaling overflows FP32"));
        }
        Ok(Self {
            rank: rank as usize,
            scale,
        })
    }
}

/// One adapted matrix's A and B: tensors of a PEFT file, or a GGUF LoRA's, read already
/// (Hugging Face order, A `[rank, k]`, B `[n, rank]`, with their own scale).
enum Pair {
    Keys(String, String),
    Loaded { a: Vec<f32>, b: Vec<f32>, rank: usize, scale: f32 },
}

pub(crate) struct Adapter {
    index: Option<StIndex>,
    config: Config,
    pairs: BTreeMap<String, Pair>,
    /// The targets read so far (a layer loads on several threads).
    used: std::sync::Mutex<BTreeSet<String>>,
    pub fingerprint: u64,
}

/// Rows `offset..offset + rows` of a matrix, adapted as the Hugging Face matrix `name` (its
/// rows reordered by `output`, as the base weight's were).
pub(crate) struct Part<'a> {
    pub name: String,
    pub offset: usize,
    pub rows: usize,
    pub output: Option<&'a [u32]>,
}

/// Every adapter's LoRA for the parts of one `n x k` matrix, stacked into one low-rank
/// pair: A `[R, k]`, B `[n, R]` (scales applied), R the ranks' sum. None if none has any.
pub(crate) fn stacked(adapters: &[Adapter], parts: &[Part<'_>], k: usize, n: usize, input: Option<&[u32]>) -> Result<Option<(Vec<f32>, Vec<f32>, usize)>> {
    let mut pieces = Vec::new();
    for adapter in adapters {
        for part in parts {
            if let Some((a, b, rank)) = adapter.part(&part.name, k, part.rows, input, part.output)? {
                pieces.push((a, b, rank, part.offset));
            }
        }
    }
    if pieces.is_empty() {
        return Ok(None);
    }
    let total: usize = pieces.iter().map(|p| p.2).sum();
    let mut a_all = Vec::with_capacity(total * k);
    let mut b_all = vec![0f32; n * total];
    let mut col = 0;
    for (a, b, rank, offset) in pieces {
        a_all.extend_from_slice(&a);
        for row in 0..b.len() / rank {
            b_all[(offset + row) * total + col..(offset + row) * total + col + rank].copy_from_slice(&b[row * rank..(row + 1) * rank]);
        }
        col += rank;
    }
    Ok(Some((a_all, b_all, total)))
}

/// `base` plus the low-rank `b @ a` (from `stacked`), on `backend`.
pub(crate) fn adapted(base: Weight, a: Vec<f32>, b: Vec<f32>, rank: usize, backend: Arc<dyn Backend>) -> Weight {
    let k = a.len() / rank;
    let n = b.len() / rank;
    let a = backend.to_device(Tensor::from_vec(a, vec![rank, k]));
    let b = backend.to_device(Tensor::from_vec(b, vec![n, rank]));
    Weight::Packed(Arc::new(LoraLinear { base, a, b, backend }))
}

impl Adapter {
    /// Scale its effect: 1 is as trained. The prompt cache keys on it too.
    pub fn with_strength(mut self, strength: f32) -> Self {
        if strength != 1.0 {
            self.config.scale *= strength;
            for pair in self.pairs.values_mut() {
                if let Pair::Loaded { scale, .. } = pair { *scale *= strength; }
            }
            self.fingerprint = crate::disk::fnv(&strength.to_le_bytes(), self.fingerprint);
        }
        self
    }

    /// A PEFT adapter folder for the 27B.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_for(path, &Base::Qwen27B)
    }

    /// An adapter for `model`: a PEFT folder (`adapter_config.json`), or for Flash-Next a
    /// llama.cpp GGUF LoRA (the file, or a folder holding just it).
    pub fn open_for(path: &Path, model: &Base) -> Result<Self> {
        if let Some(gguf) = gguf_lora(path) {
            return match model {
                Base::FlashNext(heads) => Self::open_gguf(&gguf, heads),
                Base::Qwen27B => Err(bad(format!("{}: GGUF LoRAs are read for Qwen3.8-Flash-Next only", gguf.display()))),
            };
        }
        let config_path = path.join("adapter_config.json");
        let config_bytes = std::fs::read(&config_path)
            .map_err(|error| bad(format!("reading {}: {error}", config_path.display())))?;
        let config = Config::parse_for(&Json::parse(&config_bytes)?, model)?;
        let weights = path.join("adapter_model.safetensors");
        let index = StIndex::open_file(&weights)
            .map_err(|error| bad(format!("opening {}: {error}", weights.display())))?;
        let mut pairs = BTreeMap::new();
        for key in index.names() {
            let name = key
                .strip_prefix("base_model.model.")
                .ok_or_else(|| bad(format!("unsupported tensor {key}")))?;
            let (name, side) = if let Some(n) = name.strip_suffix(".lora_A.weight") {
                (n, 0)
            } else if let Some(n) = name.strip_suffix(".lora_B.weight") {
                (n, 1)
            } else {
                return Err(bad(format!("unsupported tensor {key}")));
            };
            let pair = pairs
                .entry(name.to_string())
                .or_insert_with(|| (String::new(), String::new()));
            if side == 0 {
                pair.0 = key.into();
            } else {
                pair.1 = key.into();
            }
        }
        if pairs.is_empty() || pairs.values().any(|(a, b): &(String, String)| a.is_empty() || b.is_empty()) {
            return Err(bad(
                "adapter has no complete A/B pairs or contains an unpaired tensor",
            ));
        }
        // Hash actual contents: replacing a file in place must not resurrect
        // states from another adapter, even when path and byte count match.
        let mut fingerprint = crate::disk::fnv(b"peft-orca-lora-v1", 0);
        fingerprint = crate::disk::fnv(&config_bytes, fingerprint);
        let mut file = std::fs::File::open(weights)?;
        let mut buf = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                fingerprint = (fingerprint ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        Ok(Self {
            index: Some(index),
            config,
            pairs: pairs.into_iter().map(|(name, (a, b))| (name, Pair::Keys(a, b))).collect(),
            used: Default::default(),
            fingerprint,
        })
    }

    /// A llama.cpp LoRA for Qwen3.8-Flash-Next, read into Hugging Face names and order.
    fn open_gguf(path: &Path, heads: &VHeads) -> Result<Self> {
        let g = gguf::GgufFile::open(path).map_err(|e| bad(format!("opening {}: {e}", path.display())))?;
        let text = |key: &str| g.get_str(key).ok().map(str::to_owned);
        if text("general.type").as_deref() != Some("adapter") || text("adapter.type").as_deref() != Some("lora") {
            return Err(bad(format!("{}: not a GGUF LoRA adapter", path.display())));
        }
        if text("general.architecture").as_deref() != Some("qwen4exp") {
            return Err(bad(format!("{}: a LoRA for {}, not Qwen3.8-Flash-Next (qwen4exp)", path.display(), text("general.architecture").unwrap_or_default())));
        }
        let alpha = g.get_f32("adapter.lora.alpha").unwrap_or(0.0);
        let read = |name: &str| -> Result<Tensor> {
            let info = g.tensor_by_name(name).ok_or_else(|| bad(format!("{name} is missing its pair")))?;
            let t = llama_rs::loader::load_tensor_f32(&g, info).map_err(|e| bad(format!("{name}: {e}")))?;
            if t.data().iter().any(|v| !v.is_finite()) {
                return Err(bad("nonfinite adapter weights"));
            }
            Ok(t)
        };
        let mut pairs = BTreeMap::new();
        for info in g.tensors() {
            let Some(stem) = info.name.strip_suffix(".weight.lora_a") else {
                if info.name.ends_with(".weight.lora_b") { continue; }
                return Err(bad(format!("unsupported tensor {}", info.name)));
            };
            let a = read(&info.name)?;
            let b = read(&format!("{stem}.weight.lora_b"))?;
            // A is [.., rank, k] and B [.., n, rank]; llama.cpp scales by alpha / rank.
            let (rank, k) = (a.dim(a.shape().len() - 2), a.dim(a.shape().len() - 1));
            let n = b.dim(b.shape().len() - 2);
            if b.dim(b.shape().len() - 1) != rank || a.numel() / (rank * k) != b.numel() / (n * rank) {
                return Err(bad(format!("{stem}: A {:?} and B {:?} do not pair", a.shape(), b.shape())));
            }
            let scale = if alpha != 0.0 { alpha / rank as f32 } else { 1.0 };
            for (name, a, b) in hf_pairs(stem, a.data(), b.data(), rank, k, n, heads)? {
                pairs.insert(name, Pair::Loaded { a, b, rank, scale });
            }
        }
        if pairs.is_empty() {
            return Err(bad(format!("{}: no LoRA tensors", path.display())));
        }
        let mut fingerprint = crate::disk::fnv(b"gguf-flash-next-lora-v1", 0);
        let mut file = std::fs::File::open(path)?;
        let mut buf = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                fingerprint = (fingerprint ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        Ok(Self { index: None, config: Config { rank: 0, scale: 1.0 }, pairs, used: Default::default(), fingerprint })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }
    pub fn finish(&self) -> Result<()> {
        let used = self.used.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(name) = self.pairs.keys().find(|n| !used.contains(*n)) {
            return Err(bad(format!("unconsumed/unsupported target {name}")));
        }
        Ok(())
    }
    /// The adapter's A and B for `name` (`n x k`), channel-mapped and scaled: A
    /// `[rank, k]`, B `[n, rank]`.
    fn part(
        &self,
        name: &str,
        k: usize,
        n: usize,
        input: Option<&[u32]>,
        output: Option<&[u32]>,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, usize)>> {
        let Some(pair) = self.pairs.get(name) else {
            return Ok(None);
        };
        let (a, b, r, scale) = match pair {
            Pair::Keys(an, bn) => {
                let index = self.index.as_ref().expect("a PEFT adapter's file");
                let r = self.config.rank;
                for (key, shape) in [(an, [r, k]), (bn, [n, r])] {
                    let info = index.info(key)?;
                    if info.shape != shape || !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                        return Err(bad(format!(
                            "{key}: expected floating tensor {shape:?}, got {:?}",
                            info.shape
                        )));
                    }
                }
                let a = index.read_f32(an)?;
                let b = index.read_f32(bn)?;
                if a.iter().chain(&b).any(|v| !v.is_finite()) {
                    return Err(bad("nonfinite adapter weights"));
                }
                (a, b, r, self.config.scale)
            }
            Pair::Loaded { a, b, rank, scale } => {
                if a.len() != rank * k || b.len() != n * rank {
                    return Err(bad(format!("{name}: expected {n} x {k}, got {} x {}", b.len() / rank, a.len() / rank)));
                }
                (a.clone(), b.clone(), *rank, *scale)
            }
        };
        let (a, b) = map_tensors(a, b, r, k, n, input, output, scale)?;
        self.used.lock().unwrap_or_else(|e| e.into_inner()).insert(name.into());
        Ok(Some((a, b, r)))
    }
    fn tensors(
        &self,
        name: &str,
        k: usize,
        n: usize,
        input: Option<&[u32]>,
        output: Option<&[u32]>,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(self.part(name, k, n, input, output)?.map(|(a, b, r)| (Tensor::from_vec(a, vec![r, k]), Tensor::from_vec(b, vec![n, r]))))
    }
    pub fn wrap(
        &self,
        name: &str,
        base: Weight,
        backend: Arc<dyn Backend>,
        input: Option<&[u32]>,
        output: Option<&[u32]>,
    ) -> Result<Weight> {
        let shape = base.shape();
        let Some((a, b)) = self.tensors(name, shape[1], shape[0], input, output)? else {
            return Ok(base);
        };
        let a = backend.to_device(a);
        let b = backend.to_device(b);
        Ok(Weight::Packed(Arc::new(LoraLinear {
            base,
            a,
            b,
            backend,
        })))
    }
    pub fn merge_dense(
        &self,
        name: &str,
        mut base: Tensor,
        backend: &dyn Backend,
        output: Option<&[u32]>,
    ) -> Result<Tensor> {
        let (n, k) = (base.dim(0), base.dim(1));
        let Some((a, b)) = self.tensors(name, k, n, None, output)? else {
            return Ok(base);
        };
        let r = a.dim(0);
        let mut at = vec![0.0; k * r];
        for i in 0..k {
            for j in 0..r {
                at[i * r + j] = a.data()[j * k + i];
            }
        }
        let delta = backend.linear(
            &backend.to_device(b),
            &backend.to_device(Tensor::from_vec(at, vec![k, r])),
        );
        backend.add_inplace(&mut base, &delta);
        Ok(base)
    }
}

/// A GGUF LoRA path: the `.gguf` file, or a folder with one and no PEFT config.
fn gguf_lora(path: &Path) -> Option<std::path::PathBuf> {
    if path.is_file() {
        return path.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")).then(|| path.to_path_buf());
    }
    if !path.is_dir() || path.join("adapter_config.json").exists() {
        return None;
    }
    let files: Vec<_> = std::fs::read_dir(path).ok()?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf"))).collect();
    (files.len() == 1).then(|| files[0].clone())
}

/// A llama.cpp LoRA tensor pair (`stem`, e.g. `blk.3.ssm_out`) as Hugging Face pairs: its
/// name, and llama.cpp's value-head reorder undone (it stores Gated DeltaNet value heads
/// tiled, `[v0 of every key head, v1 of every key head, ..]`, where Hugging Face groups them by
/// key head). Stacked expert pairs split per expert, fused gate/up into both.
fn hf_pairs(stem: &str, a: &[f32], b: &[f32], rank: usize, k: usize, n: usize, heads: &VHeads) -> Result<Vec<(String, Vec<f32>, Vec<f32>)>> {
    let unsupported = || bad(format!("unsupported tensor {stem}"));
    if stem == "output" {
        return Ok(vec![("lm_head".into(), a.to_vec(), b.to_vec())]);
    }
    let rest = stem.strip_prefix("blk.").ok_or_else(unsupported)?;
    let (layer, what) = rest.split_once('.').ok_or_else(unsupported)?;
    let layer: usize = layer.parse().map_err(|_| unsupported())?;
    let p = format!("model.language_model.layers.{layer}");
    // Hugging Face order of llama.cpp's tiled value heads: its row (or column) blocks of
    // `width`, from Hugging Face head g = key head * per + j at llama.cpp head j * k_heads + key head.
    let per = heads.v_heads / heads.k_heads.max(1);
    let untile = |values: &[f32], width: usize, stride: usize, count: usize, at: usize| -> Vec<f32> {
        // `count` lines of `stride` values; the heads are the `v_heads * width` positions from `at`.
        let mut out = values.to_vec();
        for line in 0..count {
            for g in 0..heads.v_heads {
                let t = (g % per) * heads.k_heads + g / per;
                for d in 0..width {
                    out[line * stride + at + g * width + d] = values[line * stride + at + t * width + d];
                }
            }
        }
        out
    };
    // Rows of B (one line of `rank` per row: rows are lines) from row `at`, heads `width` rows each.
    let rows = |b: &[f32], width: usize, at: usize| -> Vec<f32> {
        let mut out = b.to_vec();
        for g in 0..heads.v_heads {
            let t = (g % per) * heads.k_heads + g / per;
            for d in 0..width {
                let (dst, src) = ((at + g * width + d) * rank, (at + t * width + d) * rank);
                out[dst..dst + rank].copy_from_slice(&b[src..src + rank]);
            }
        }
        out
    };
    let one = |name: &str, a: Vec<f32>, b: Vec<f32>| Ok(vec![(format!("{p}.{name}"), a, b)]);
    match what {
        "attn_q" => one("self_attn.q_proj", a.to_vec(), b.to_vec()),
        "attn_k" => one("self_attn.k_proj", a.to_vec(), b.to_vec()),
        "attn_v" => one("self_attn.v_proj", a.to_vec(), b.to_vec()),
        "attn_output" => one("self_attn.o_proj", a.to_vec(), b.to_vec()),
        "indexer.q_proj" => one("self_attn.indexer.index_q_proj", a.to_vec(), b.to_vec()),
        "indexer.k_proj" => one("self_attn.indexer.index_k_proj", a.to_vec(), b.to_vec()),
        // Q and K rows first, then the value heads.
        "attn_qkv" => one("linear_attn.in_proj_qkv", a.to_vec(), rows(b, heads.v_dim, 2 * heads.k_heads * heads.k_dim)),
        "attn_gate" => one("linear_attn.in_proj_z", a.to_vec(), rows(b, heads.v_dim, 0)),
        "ssm_beta" => one("linear_attn.in_proj_b", a.to_vec(), rows(b, 1, 0)),
        "ssm_alpha" => one("linear_attn.in_proj_a", a.to_vec(), rows(b, 1, 0)),
        // The output projection reads the value heads: A's columns.
        "ssm_out" => one("linear_attn.out_proj", untile(a, heads.v_dim, k, rank, 0), b.to_vec()),
        "ffn_gate_inp" => one("mlp.gate", a.to_vec(), b.to_vec()),
        "ffn_gate_inp_shexp" => one("mlp.shared_expert_gate", a.to_vec(), b.to_vec()),
        "ffn_gate_shexp" => one("mlp.shared_expert.gate_proj", a.to_vec(), b.to_vec()),
        "ffn_up_shexp" => one("mlp.shared_expert.up_proj", a.to_vec(), b.to_vec()),
        "ffn_down_shexp" => one("mlp.shared_expert.down_proj", a.to_vec(), b.to_vec()),
        "ple_key" => one("ple.key_proj", a.to_vec(), b.to_vec()),
        "ple_value" => one("ple.value_proj", a.to_vec(), b.to_vec()),
        "ffn_gate_exps" | "ffn_up_exps" | "ffn_down_exps" | "ffn_gate_up_exps" => {
            let experts = a.len() / (rank * k);
            let mut out = Vec::new();
            for e in 0..experts {
                let ae = a[e * rank * k..(e + 1) * rank * k].to_vec();
                let be = &b[e * n * rank..(e + 1) * n * rank];
                let name = |m: &str| format!("{p}.mlp.experts.{e}.{m}");
                match what {
                    "ffn_gate_exps" => out.push((name("gate_proj"), ae, be.to_vec())),
                    "ffn_up_exps" => out.push((name("up_proj"), ae, be.to_vec())),
                    "ffn_down_exps" => out.push((name("down_proj"), ae, be.to_vec())),
                    _ => {
                        // Gate rows, then up rows.
                        let half = n / 2 * rank;
                        out.push((name("gate_proj"), ae.clone(), be[..half].to_vec()));
                        out.push((name("up_proj"), ae, be[half..].to_vec()));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(unsupported()),
    }
}

fn map_tensors(
    a: Vec<f32>,
    b: Vec<f32>,
    r: usize,
    k: usize,
    n: usize,
    input: Option<&[u32]>,
    output: Option<&[u32]>,
    scale: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    for (map, len) in [(input, k), (output, n)] {
        if let Some(map) = map {
            let mut seen = vec![false; len];
            if map.len() != len
                || map
                    .iter()
                    .any(|&i| i as usize >= len || std::mem::replace(&mut seen[i as usize], true))
            {
                return Err(bad("invalid channel permutation"));
            }
        }
    }
    // EXL3 input map: original HF channel -> runtime channel. Its output map
    // is the opposite direction. Apply both to the unquantized LoRA branch.
    let mut mapped_a = vec![0.0; a.len()];
    for row in 0..r {
        for col in 0..k {
            let dst = input.map_or(col, |m| m[col] as usize);
            mapped_a[row * k + dst] = a[row * k + col];
        }
    }
    let mut mapped_b = vec![0.0; b.len()];
    for row in 0..n {
        let src = output.map_or(row, |m| m[row] as usize);
        for col in 0..r {
            mapped_b[row * r + col] = b[src * r + col] * scale;
        }
    }
    if mapped_b.iter().any(|v| !v.is_finite()) {
        return Err(bad("scaled adapter overflows FP32"));
    }
    Ok((mapped_a, mapped_b))
}

struct LoraLinear {
    base: Weight,
    a: Tensor,
    b: Tensor,
    backend: Arc<dyn Backend>,
}
impl std::fmt::Debug for LoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraLinear")
            .field("shape", &self.base.shape())
            .field("rank", &self.a.dim(0))
            .finish()
    }
}
impl PackedLinear for LoraLinear {
    fn shape(&self) -> &[usize] {
        self.base.shape()
    }
    fn nbytes(&self) -> usize {
        (match &self.base {
            Weight::Packed(w) => w.nbytes(),
            Weight::Quant(w) => w.nbytes(),
            Weight::Dense(t) => t.numel() * 4,
            Weight::TiedEmbed(t) => t.numel() * 4,
        }) + (self.a.numel() + self.b.numel()) * 4
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        let mut y = self.base.linear(self.backend.as_ref(), x);
        self.backend.add_lora(&mut y, x, &self.a, &self.b);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggml_rs::CpuBackend;
    use std::sync::atomic::{AtomicU64, Ordering};
    const NAME: &str = "model.language_model.layers.0.mlp.gate_proj";
    const CONFIG: &str = r#"{"peft_type":"LORA","base_model_name_or_path":"Qwen/Qwen3.8-27B","r":2,"lora_alpha":8,"use_rslora":false}"#;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let p = std::env::temp_dir().join(format!(
                "oaiy-lora-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(dir: &Path, bf16: bool) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<_> = (0..6).map(|i| (i as f32 - 3.0) * 0.125).collect();
        let b: Vec<_> = (0..8).map(|i| (i as f32 - 4.0) * 0.25).collect();
        let mut bytes = vec![];
        let mut fields = vec![];
        for (suffix, shape, values) in [("lora_A", [2, 3], &a), ("lora_B", [4, 2], &b)] {
            let start = bytes.len();
            for v in values {
                if bf16 {
                    bytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes())
                } else {
                    bytes.extend_from_slice(&v.to_le_bytes())
                }
            }
            fields.push(format!("\"base_model.model.{NAME}.{suffix}.weight\":{{\"dtype\":\"{}\",\"shape\":[{},{}],\"data_offsets\":[{start},{}]}}",if bf16 {"BF16"} else {"F32"},shape[0],shape[1],bytes.len()));
        }
        let header = format!("{{{}}}", fields.join(","));
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&bytes);
        std::fs::write(dir.join("adapter_config.json"), CONFIG).unwrap();
        std::fs::write(dir.join("adapter_model.safetensors"), file).unwrap();
        (a, b)
    }
    #[test]
    fn missing_adapter_files_report_the_exact_path() {
        let dir = Temp::new();
        let error = Adapter::open(&dir.0).err().unwrap().to_string();
        assert!(error.contains(&dir.0.join("adapter_config.json").display().to_string()));
        std::fs::write(dir.0.join("adapter_config.json"), CONFIG).unwrap();
        let error = Adapter::open(&dir.0).err().unwrap().to_string();
        assert!(error.contains(&dir.0.join("adapter_model.safetensors").display().to_string()));
    }

    #[test]
    fn a_strength_scales_the_adapter_and_keys_the_prompt_cache() {
        let dir = Temp::new();
        fixture(&dir.0, false);
        let plain = Adapter::open(&dir.0).unwrap();
        let (scale, fingerprint) = (plain.config.scale, plain.fingerprint);
        let same = Adapter::open(&dir.0).unwrap().with_strength(1.0);
        assert_eq!((same.config.scale, same.fingerprint), (scale, fingerprint));
        let half = Adapter::open(&dir.0).unwrap().with_strength(0.5);
        assert_eq!(half.config.scale, scale * 0.5);
        assert_ne!(half.fingerprint, fingerprint);
    }

    #[test]
    fn scaling_and_unsupported_variants() {
        let c = Json::parse(CONFIG.as_bytes()).unwrap();
        assert_eq!(Config::parse(&c).unwrap().scale, 4.0);
        let c = Json::parse(CONFIG.replace("false", "true").as_bytes()).unwrap();
        assert!((Config::parse(&c).unwrap().scale - 8.0 / 2f32.sqrt()).abs() < 1e-6);
        for extra in [
            r#""use_dora":true"#,
            r#""fan_in_fan_out":true"#,
            r#""bias":"all""#,
            r#""modules_to_save":["lm_head"]"#,
            r#""rank_pattern":{"q_proj":4}"#,
        ] {
            let c = format!("{},{} }}", CONFIG.trim_end_matches('}'), extra);
            assert!(
                Config::parse(&Json::parse(c.as_bytes()).unwrap()).is_err(),
                "{extra}"
            );
        }
        for config in [
            CONFIG.replace("Qwen3.8-27B", "Qwen3.5-27B"),
            CONFIG.replace("\"r\":2", "\"r\":0"),
            CONFIG.replace("\"r\":2", "\"r\":1.5"),
        ] {
            assert!(Config::parse(&Json::parse(config.as_bytes()).unwrap()).is_err());
        }
    }
    fn compare(backend: Arc<dyn Backend>) {
        for bf16 in [false, true] {
            let dir = Temp::new();
            let (a, b) = fixture(&dir.0, bf16);
            let adapter = Adapter::open(&dir.0).unwrap();
            assert!(adapter.finish().is_err());
            let im = [2, 0, 1];
            let om = [1, 3, 0, 2];
            let weights: Vec<_> = (0..12).map(|i| i as f32 * 0.125).collect();
            let base =
                Weight::Dense(backend.to_device(Tensor::from_vec(weights.clone(), vec![4, 3])));
            let wrapped = adapter
                .wrap(NAME, base, backend.clone(), Some(&im), Some(&om))
                .unwrap();
            adapter.finish().unwrap();
            for seq in [1, 4] {
                let x: Vec<_> = (0..seq * 3).map(|i| (i as f32 - 2.0) * 0.25).collect();
                let got = wrapped
                    .linear(
                        backend.as_ref(),
                        &backend.to_device(Tensor::from_vec(x.clone(), vec![seq, 3])),
                    )
                    .to_host();
                for t in 0..seq {
                    for row in 0..4 {
                        let mut want = 0.0;
                        for col in 0..3 {
                            want += x[t * 3 + col] * weights[row * 3 + col];
                        }
                        // Independent HF-coordinate equation, with a deliberately
                        // non-self-inverse permutation to catch direction errors.
                        for j in 0..2 {
                            for col in 0..3 {
                                want += 4.0
                                    * b[om[row] as usize * 2 + j]
                                    * a[j * 3 + col]
                                    * x[t * 3 + im[col] as usize];
                            }
                        }
                        assert!((got.data()[t * 4 + row] - want).abs() < 1e-5);
                    }
                }
            }
            let base = backend.to_device(Tensor::from_vec(weights.clone(), vec![4, 3]));
            let merged = adapter
                .merge_dense(NAME, base, backend.as_ref(), Some(&om))
                .unwrap()
                .to_host();
            for row in 0..4 {
                for col in 0..3 {
                    let want = weights[row * 3 + col]
                        + (0..2)
                            .map(|j| 4.0 * b[om[row] as usize * 2 + j] * a[j * 3 + col])
                            .sum::<f32>();
                    assert!((merged.data()[row * 3 + col] - want).abs() < 1e-5);
                }
            }
        }
    }
    /// llama.cpp's `_reorder_v_heads`: blocks of `width` along the lines' `at..` span, from
    /// Hugging Face (grouped by key head) to tiled order.
    fn tile(values: &[f32], lines: usize, stride: usize, at: usize, h: &VHeads, width: usize) -> Vec<f32> {
        let per = h.v_heads / h.k_heads;
        let mut out = values.to_vec();
        for line in 0..lines {
            for k in 0..h.k_heads {
                for j in 0..per {
                    for d in 0..width {
                        out[line * stride + at + ((j * h.k_heads + k) * width) + d] = values[line * stride + at + ((k * per + j) * width) + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn gguf_value_heads_come_back_to_hugging_face_order() {
        let h = VHeads { k_heads: 2, v_heads: 6, k_dim: 2, v_dim: 3 };
        let rank = 2;
        let v = h.v_heads * h.v_dim;
        let values = |n: usize| (0..n).map(|i| i as f32).collect::<Vec<f32>>();
        // out_proj: A's columns are the value heads.
        let (a, b) = (values(rank * v), values(4 * rank));
        let got = hf_pairs("blk.5.ssm_out", &tile(&a, rank, v, 0, &h, h.v_dim), &b, rank, v, 4, &h).unwrap();
        assert_eq!(got[0].0, "model.language_model.layers.5.linear_attn.out_proj");
        assert_eq!((&got[0].1, &got[0].2), (&a, &b));
        // in_proj_qkv: B's rows past Q and K are the value heads.
        let n = 2 * h.k_heads * h.k_dim + v;
        let (a, b) = (values(rank * 4), values(n * rank));
        // Each row is a line of `rank`: tile rows, i.e. lines of one, `rank` apart.
        let tiled: Vec<f32> = {
            let mut t = b.clone();
            let at = 2 * h.k_heads * h.k_dim;
            let per = h.v_heads / h.k_heads;
            for k in 0..h.k_heads { for j in 0..per { for d in 0..h.v_dim {
                let (dst, src) = ((at + (j * h.k_heads + k) * h.v_dim + d) * rank, (at + (k * per + j) * h.v_dim + d) * rank);
                t[dst..dst + rank].copy_from_slice(&b[src..src + rank]);
            } } }
            t
        };
        let got = hf_pairs("blk.0.attn_qkv", &a, &tiled, rank, 4, n, &h).unwrap();
        assert_eq!(got[0].0, "model.language_model.layers.0.linear_attn.in_proj_qkv");
        assert_eq!(got[0].2, b);
        // Stacked experts split per expert; fused gate/up into both.
        let (a, b) = (values(3 * rank * 4), values(3 * 6 * rank));
        let got = hf_pairs("blk.2.ffn_gate_up_exps", &a, &b, rank, 4, 6, &h).unwrap();
        assert_eq!(got.len(), 6);
        assert_eq!(got[2].0, "model.language_model.layers.2.mlp.experts.1.gate_proj");
        assert_eq!(got[3].2, b[6 * rank + 3 * rank..2 * 6 * rank].to_vec());
        assert!(hf_pairs("blk.0.hc_attn_down", &a, &b, rank, 4, 6, &h).is_err());
    }

    #[test]
    fn parts_stack_into_one_pair() {
        let dir = Temp::new();
        let (a, b) = fixture(&dir.0, false);
        let adapter = Adapter::open(&dir.0).unwrap();
        // The fixture's 4 x 3 matrix as rows 2..6 of a 7-row one.
        let parts = [Part { name: NAME.into(), offset: 2, rows: 4, output: None }];
        let (sa, sb, rank) = stacked(std::slice::from_ref(&adapter), &parts, 3, 7, None).unwrap().unwrap();
        assert_eq!((rank, sa), (2, a));
        for row in 0..7 {
            for j in 0..2 {
                let want = if (2..6).contains(&row) { 4.0 * b[(row - 2) * 2 + j] } else { 0.0 };
                assert_eq!(sb[row * 2 + j], want);
            }
        }
        adapter.finish().unwrap();
    }

    #[test]
    fn adapter_math_and_channel_maps_match_dense_equation() {
        compare(Arc::new(CpuBackend::new()));
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_adapter_math_matches_dense_equation() {
        if let Ok(backend) = ggml_rs_cuda::CudaBackend::new(0) {
            compare(Arc::new(backend));
        }
    }
    #[test]
    fn shape_errors_and_cache_identity_are_explicit() {
        let dir = Temp::new();
        fixture(&dir.0, false);
        let adapter = Adapter::open(&dir.0).unwrap();
        let base = Weight::Dense(Tensor::zeros(vec![4, 4]));
        assert!(adapter
            .wrap(NAME, base, Arc::new(CpuBackend::new()), None, None)
            .is_err());
        let before = adapter.fingerprint;
        std::fs::write(
            dir.0.join("adapter_config.json"),
            CONFIG.replace("alpha\":8", "alpha\":4"),
        )
        .unwrap();
        let changed = Adapter::open(&dir.0).unwrap();
        assert_ne!(before, changed.fingerprint);
        let p = dir.0.join("adapter_model.safetensors");
        let mut bytes = std::fs::read(&p).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(p, bytes).unwrap();
        assert_ne!(
            changed.fingerprint,
            Adapter::open(&dir.0).unwrap().fingerprint
        );
        assert!(map_tensors(
            vec![0.0; 6],
            vec![0.0; 8],
            2,
            3,
            4,
            Some(&[0, 0, 1]),
            None,
            1.0
        )
        .is_err());
    }
}
