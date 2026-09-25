//! PEFT LoRA/rsLoRA inference alongside the original packed Orca weights.
//! Adapter tensors stay separate; only already-dense tiny projections are
//! combined in device memory. Never rewrite or requantize the base checkpoint.
use dsv41::safetensors::{Dtype, StIndex};
use ggml_rs::{exl3::PackedLinear, Backend, Tensor};
use llama_rs::loader::Weight;
use nrob::{json::Json, Error, Result};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
    sync::Arc,
};

fn bad(s: impl Into<String>) -> Error {
    Error::Format(format!("LoRA: {}", s.into()))
}

struct Config {
    rank: usize,
    scale: f32,
}
impl Config {
    fn parse(c: &Json) -> Result<Self> {
        if c.get("peft_type").and_then(Json::as_str) != Some("LORA") {
            return Err(bad("adapter must use PEFT LORA"));
        }
        let base = c
            .get("base_model_name_or_path")
            .and_then(Json::as_str)
            .unwrap_or("")
            .replace('\\', "/");
        if base != "Qwen/Qwen3.8-27B"
            && !base.ends_with("/Qwen/Qwen3.8-27B")
            && base != "orcarouter/OrcaSAQ-2-27B"
        {
            return Err(bad(
                "adapter must identify Qwen/Qwen3.8-27B or orcarouter/OrcaSAQ-2-27B as its base",
            ));
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

pub(crate) struct Adapter {
    index: StIndex,
    config: Config,
    pairs: BTreeMap<String, (String, String)>,
    used: RefCell<BTreeSet<String>>,
    pub fingerprint: u64,
}
impl Adapter {
    pub fn open(path: &Path) -> Result<Self> {
        let config_path = path.join("adapter_config.json");
        let config_bytes = std::fs::read(&config_path)
            .map_err(|error| bad(format!("reading {}: {error}", config_path.display())))?;
        let config = Config::parse(&Json::parse(&config_bytes)?)?;
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
        if pairs.is_empty() || pairs.values().any(|(a, b)| a.is_empty() || b.is_empty()) {
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
            index,
            config,
            pairs,
            used: RefCell::new(BTreeSet::new()),
            fingerprint,
        })
    }
    pub fn len(&self) -> usize {
        self.pairs.len()
    }
    pub fn finish(&self) -> Result<()> {
        let used = self.used.borrow();
        if let Some(name) = self.pairs.keys().find(|n| !used.contains(*n)) {
            return Err(bad(format!("unconsumed/unsupported target {name}")));
        }
        Ok(())
    }
    fn tensors(
        &self,
        name: &str,
        k: usize,
        n: usize,
        input: Option<&[u32]>,
        output: Option<&[u32]>,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let Some((an, bn)) = self.pairs.get(name) else {
            return Ok(None);
        };
        let r = self.config.rank;
        for (key, shape) in [(an, [r, k]), (bn, [n, r])] {
            let info = self.index.info(key)?;
            if info.shape != shape || !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                return Err(bad(format!(
                    "{key}: expected floating tensor {shape:?}, got {:?}",
                    info.shape
                )));
            }
        }
        let a = self.index.read_f32(an)?;
        let b = self.index.read_f32(bn)?;
        if a.iter().chain(&b).any(|v| !v.is_finite()) {
            return Err(bad("nonfinite adapter weights"));
        }
        let (a, b) = map_tensors(a, b, r, k, n, input, output, self.config.scale)?;
        self.used.borrow_mut().insert(name.into());
        Ok(Some((
            Tensor::from_vec(a, vec![r, k]),
            Tensor::from_vec(b, vec![n, r]),
        )))
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
        let low = self.backend.linear(x, &self.a);
        let delta = self.backend.linear(&low, &self.b);
        self.backend.add_inplace(&mut y, &delta);
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
                "nrob-lora-{}-{}",
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
    #[test]
    fn adapter_math_and_channel_maps_match_dense_equation() {
        compare(Arc::new(CpuBackend::new()));
    }
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
