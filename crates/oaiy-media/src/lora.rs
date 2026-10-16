//! LoRA adapters for the image transformers (Qwen Image's, FLUX.2 Klein's, FLUX 3 Image Klein's), applied at run time
//! beside each projection or merged into its weights as they load.
//!
//! Several adapters stack: the turbo adapter and any number of style or
//! character LoRAs, each with a strength. Both common key layouts are read:
//! diffusers/PEFT (`transformer.{module}.lora_A.weight`, `.lora_B.weight`,
//! optional `.alpha`) and kohya/ComfyUI (`diffusion_model.{module}.lora_down`,
//! `.lora_up`, `.alpha`, or `lora_unet_{module with _}.lora_down`). The
//! effective scale is strength × alpha / rank (alpha = rank when absent; a diffusers file's `lora_alpha` from its
//! metadata, where it keeps no alpha tensors), folded into the up factor. A model whose modules another trainer
//! names otherwise (diffusers' FLUX 3) gives their other names ([`Alias`]).

use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

/// One adapter: its tensors, strength, and which modules it has and used.
struct Adapter {
    path: PathBuf,
    weights: Weights,
    strength: f64,
    /// The modules it adapts, as the model names them.
    modules: BTreeSet<String>,
    used: BTreeSet<String>,
    peft: Option<Peft>,
}

/// A diffusers (PEFT) file's scale, from its `lora_adapter_metadata` (it keeps no alpha tensors): its `lora_alpha`, a
/// module's own in `alpha_pattern`, over the rank (rank-stabilized LoRA's: over the rank's square root).
struct Peft {
    alpha: f64,
    pattern: Vec<(String, f64)>,
    rslora: bool,
}

impl Peft {
    fn of(weights: &Weights) -> Option<Self> {
        let j = Json::parse(weights.metadata("lora_adapter_metadata")?.as_bytes()).ok()?;
        // (diffusers names each part of a pipeline's: the transformer's `transformer.lora_alpha`)
        let field = |k: &str| j.get(&format!("transformer.{k}")).or_else(|| j.get(k));
        let alpha = field("lora_alpha")?.as_f64()?;
        let pattern = match field("alpha_pattern") {
            Some(Json::Obj(m)) => m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64()?))).collect(),
            _ => Vec::new(),
        };
        Some(Self { alpha, pattern, rslora: field("use_rslora").and_then(Json::as_bool).unwrap_or(false) })
    }

    /// The scale of `module`'s factors of `rank` (a pattern names the module, or its end).
    fn scale(&self, module: &str, rank: usize) -> f64 {
        let alpha = self.pattern.iter().find(|(p, _)| module == p || module.ends_with(&format!(".{p}"))).map_or(self.alpha, |(_, a)| *a);
        if self.rslora { alpha / (rank as f64).sqrt() } else { alpha / rank as f64 }
    }
}

/// Another name an adapter may give a module (another trainer's naming of the same model), and whether its up
/// factor's two halves of rows are the other way round there (diffusers' FLUX 3 `norm_out.linear`: the scale then the
/// shift, where BFL's final modulation is the shift then the scale).
pub struct Alias {
    pub name: String,
    pub swap_halves: bool,
}

/// Which of a stored up factor's rows are a projection's.
#[derive(Clone, Copy)]
enum Rows {
    All,
    /// Half `i` of a fused one (Comfy's fused MLP: the gate's rows, then the up's).
    Half(usize),
    /// All, its two halves the other way round.
    Swapped,
}

/// The adapters for one transformer.
#[derive(Default)]
pub struct Loras {
    adapters: Vec<Adapter>,
    /// The model they are for, as messages name it (None: Qwen Image 2.1).
    model: Option<&'static str>,
    /// Another naming of the model's modules an adapter may use.
    alias: Option<fn(&str) -> Option<Alias>>,
}

/// Where a module's factors live in an adapter.
struct Keys {
    down: String,
    up: String,
    alpha: String,
}

/// The key layouts to try for a module, most common first.
fn layouts(module: &str) -> Vec<Keys> {
    let flat = format!("lora_unet_{}", module.replace('.', "_"));
    let mut out = Vec::new();
    for prefix in ["transformer.", "", "base_model.model."] {
        let base = format!("{prefix}{module}");
        out.push(Keys { down: format!("{base}.lora_A.weight"), up: format!("{base}.lora_B.weight"), alpha: format!("{base}.alpha") });
        out.push(Keys { down: format!("{base}.lora_down.weight"), up: format!("{base}.lora_up.weight"), alpha: format!("{base}.alpha") });
    }
    out.push(Keys { down: format!("{flat}.lora_down.weight"), up: format!("{flat}.lora_up.weight"), alpha: format!("{flat}.alpha") });
    out
}

/// The module a down-factor key adapts, in the model's own naming.
fn module_of(key: &str) -> Option<String> {
    let base = key
        .strip_suffix(".lora_A.weight")
        .or_else(|| key.strip_suffix(".lora_down.weight"))?;
    let base = ["model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer."]
        .iter()
        .find_map(|p| base.strip_prefix(p))
        .unwrap_or(base);
    Some(base.to_owned())
}

impl Adapter {
    fn open(path: &Path, strength: f64) -> Result<Self> {
        let weights = Weights::open(path)?;
        let modules: BTreeSet<String> = weights.names().iter().filter_map(|k| module_of(k)).collect();
        if modules.is_empty() {
            candle_core::bail!("{} holds no LoRA factors (lora_A/lora_B or lora_down/lora_up)", path.display());
        }
        let peft = Peft::of(&weights);
        Ok(Self { path: path.to_owned(), weights, strength, modules, used: BTreeSet::new(), peft })
    }

    /// Whether it adapts `module` (kohya's flat names count too).
    fn adapts(&self, module: &str) -> bool {
        self.modules.contains(module) || self.modules.contains(&format!("lora_unet_{}", module.replace('.', "_")))
    }

    /// Its tensors that are neither LoRA factors nor their alphas: whole weights or changes of other kinds (a
    /// distillation extract's norm changes and new layers, a DoRA's magnitudes), which are not applied.
    fn extras(&self) -> Vec<String> {
        let factor = |k: &str| [".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight", ".alpha"].iter().any(|s| k.ends_with(s));
        self.weights.names().into_iter().filter(|k| !factor(k)).collect()
    }

    /// The (down, up × scale) factors for `module`, for a projection of shape [out, input]: under its own name, or
    /// (where the adapter has none there) `alias`'s.
    #[allow(clippy::too_many_arguments)]
    fn factors(&mut self, module: &str, alias: Option<&Alias>, out: usize, input: usize, dev: &Device, dtype: DType) -> Result<Option<(Tensor, Tensor)>> {
        if let Some(f) = self.factors_as(module, module, out, Rows::All, input, dev, dtype)? {
            return Ok(Some(f));
        }
        // Comfy's fused MLP: one `img_mlp.gate_up` factor pair whose up rows are
        // the gate rows then the up rows, as the fused base weight stores them.
        for (suffix, half) in [(".img_mlp.gate_layer", 0), (".img_mlp.proj", 1)] {
            if let Some(block) = module.strip_suffix(suffix) {
                let fused = format!("{block}.img_mlp.gate_up");
                return self.factors_as(&fused, module, out, Rows::Half(half), input, dev, dtype);
            }
        }
        if let Some(a) = alias {
            return self.factors_as(&a.name, module, out, if a.swap_halves { Rows::Swapped } else { Rows::All }, input, dev, dtype);
        }
        Ok(None)
    }

    /// The factors stored for `stored`, for the projection `module`, the up factor's `rows` ours.
    #[allow(clippy::too_many_arguments)]
    fn factors_as(&mut self, stored: &str, module: &str, out: usize, rows: Rows, input: usize, dev: &Device, dtype: DType) -> Result<Option<(Tensor, Tensor)>> {
        for keys in layouts(stored) {
            if !self.weights.has(&keys.down) {
                continue;
            }
            let down = self.weights.tensor(&keys.down, dev, DType::F32)?;
            let mut up = self.weights.tensor(&keys.up, dev, DType::F32)?;
            let (rank, down_in) = down.dims2()?;
            match rows {
                Rows::All => {}
                Rows::Half(half) => {
                    let rows = up.dim(0)?;
                    if rows != 2 * out {
                        candle_core::bail!("LoRA {} does not fit {module}: its fused {stored} has {rows} rows, not 2 × {out}", self.path.display());
                    }
                    up = up.narrow(0, half * out, out)?;
                }
                // (a shape that does not fit is refused below)
                Rows::Swapped if up.dim(0)? == out && out % 2 == 0 => {
                    up = Tensor::cat(&[up.narrow(0, out / 2, out / 2)?, up.narrow(0, 0, out / 2)?], 0)?;
                }
                Rows::Swapped => {}
            }
            let (up_out, up_rank) = up.dims2()?;
            if up_rank != rank || down_in != input || up_out != out {
                candle_core::bail!(
                    "LoRA {} does not fit {module}: its factors are [{up_out}, {up_rank}]·[{rank}, {down_in}], the projection is [{out}, {input}] (is it for another model?)",
                    self.path.display()
                );
            }
            let alpha = if self.weights.has(&keys.alpha) {
                self.weights.tensor(&keys.alpha, dev, DType::F32)?.flatten_all()?.to_vec1::<f32>()?.first().copied().map(f64::from)
            } else {
                None
            };
            let scale = self.strength
                * match (alpha, &self.peft) {
                    (Some(a), _) => a / rank as f64,
                    (None, Some(peft)) => peft.scale(stored, rank),
                    (None, None) => 1.,
                };
            self.used.insert(stored.to_owned());
            return Ok(Some((down.to_dtype(dtype)?, (up * scale)?.to_dtype(dtype)?)));
        }
        Ok(None)
    }
}

impl Loras {
    /// The adapters to apply: each path with its strength.
    pub fn open(list: &[(PathBuf, f64)]) -> Result<Self> {
        let adapters = list.iter().map(|(p, s)| Adapter::open(p, *s)).collect::<Result<_>>()?;
        Ok(Self { adapters, model: None, alias: None })
    }

    /// For `model` (as messages name it), whose modules an adapter may also name as `alias` gives them.
    pub fn for_model(self, model: &'static str, alias: Option<fn(&str) -> Option<Alias>>) -> Self {
        Self { model: Some(model), alias, ..self }
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Strict architecture preflight without reading adapter payloads: every adapter's modules among `expected` (each
    /// projection's name, rows and columns; Klein's the original fused BFL projection names, including fused
    /// QKV/MLP) and of their shapes, under their own names or their aliases', and nothing in it but LoRA factors.
    pub fn validate_modules(&self, expected: &[(String, usize, usize)]) -> Result<()> {
        let model = self.model.unwrap_or("the selected architecture");
        for adapter in &self.adapters {
            if !adapter.strength.is_finite() || !(-4. ..=4.).contains(&adapter.strength) {
                candle_core::bail!("LoRA strength must be finite and between -4 and 4");
            }
            let extras = adapter.extras();
            if !extras.is_empty() {
                candle_core::bail!(
                    "LoRA {} holds {} tensors besides LoRA factors (e.g. {}): whole weights or changes of another kind, as a distillation extract or a DoRA keeps, which OAIY does not apply",
                    adapter.path.display(),
                    extras.len(),
                    extras.iter().take(2).map(String::as_str).collect::<Vec<_>>().join(", ")
                );
            }
            let mut found = BTreeSet::new();
            for (module, out, input) in expected {
                let alias = self.alias.and_then(|f| f(module));
                for stored in std::iter::once(module.as_str()).chain(alias.as_ref().map(|a| a.name.as_str())) {
                    let Some(keys) = layouts(stored).into_iter().find(|k| adapter.weights.has(&k.down)) else { continue };
                    let down = adapter.weights.shape(&keys.down)?;
                    let up = adapter.weights.shape(&keys.up)?;
                    if down.len() != 2 || up.len() != 2 || down[0] == 0 || down[1] != *input || up[0] != *out || up[1] != down[0] {
                        candle_core::bail!("LoRA {} does not fit {module}: {up:?} x {down:?}; expected [{out}, {input}]", adapter.path.display());
                    }
                    found.insert(stored.to_owned());
                    break;
                }
            }
            if found.is_empty() {
                candle_core::bail!("LoRA {} fits none of {model}'s projections (is it for another model?)", adapter.path.display());
            }
            if let Some(m) = adapter.modules.iter().find(|m| !found.contains(*m) && !found.iter().any(|u| format!("lora_unet_{}", u.replace('.', "_")) == **m)) {
                candle_core::bail!("LoRA {} adapts {m}, which is not in {model}: it contains projections outside it (is it for another model?)", adapter.path.display());
            }
        }
        Ok(())
    }

    /// Whether the first adapter (the turbo one, when there is one) adapts `module`.
    pub fn first_adapts(&self, module: &str) -> bool {
        self.adapters.first().is_some_and(|a| a.adapts(module))
    }

    /// Every adapter's factors for `module`, a projection of shape [out, input].
    pub fn factors(&mut self, module: &str, out: usize, input: usize, dev: &Device, dtype: DType) -> Result<Vec<(Tensor, Tensor)>> {
        let alias = self.alias.and_then(|f| f(module));
        let mut all = Vec::new();
        for adapter in &mut self.adapters {
            if let Some(f) = adapter.factors(module, alias.as_ref(), out, input, dev, dtype)? {
                all.push(f);
            }
        }
        Ok(all)
    }

    /// After loading: an adapter that fits none of the model's projections is
    /// for another model. Returns a note for each one that fit only in part.
    pub fn check(&self) -> Result<Vec<String>> {
        let mut notes = Vec::new();
        for a in &self.adapters {
            if a.used.is_empty() {
                match self.model {
                    Some(model) => candle_core::bail!("LoRA {} fits none of {model}'s projections (it may be for another model)", a.path.display()),
                    None => candle_core::bail!(
                        "LoRA {} fits none of this Qwen Image 2.1 model's projections (it may be for another model, such as the two-stream Qwen-Image or Qwen-Image-Edit)",
                        a.path.display()
                    ),
                }
            }
            let unused: Vec<&String> = a.modules.iter().filter(|m| !a.used.contains(*m) && !a.used.iter().any(|u| format!("lora_unet_{}", u.replace('.', "_")) == **m)).collect();
            if !unused.is_empty() {
                notes.push(format!(
                    "LoRA {}: {} of its {} modules are not in this model and were left out (e.g. {})",
                    a.path.display(),
                    unused.len(),
                    a.modules.len(),
                    unused.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ));
            }
        }
        Ok(notes)
    }
}

/// A safetensors file of F32 tensors, with `metadata` (string pairs): a test's model or adapter.
#[cfg(test)]
pub(crate) fn write_f32(path: &Path, tensors: &[(&str, &[usize], &[f32])], metadata: &[(&str, &str)]) {
    let (mut header, mut data) = (Vec::new(), Vec::new());
    if !metadata.is_empty() {
        header.push(("__metadata__".to_owned(), Json::Obj(metadata.iter().map(|(k, v)| ((*k).to_owned(), Json::str(*v))).collect())));
    }
    for (name, shape, values) in tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|x| x.to_le_bytes()));
        header.push(((*name).to_owned(), Json::obj([
            ("dtype", Json::str("F32")),
            ("shape", Json::Arr(shape.iter().map(|&n| Json::Int(n as i64)).collect())),
            ("data_offsets", Json::Arr(vec![Json::Int(start as i64), Json::Int(data.len() as i64)])),
        ])));
    }
    let header = Json::Obj(header).to_json();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file of F32 tensors.
    fn write(path: &Path, tensors: &[(&str, &[usize], &[f32])]) {
        write_f32(path, tensors, &[]);
    }

    #[test]
    fn adapters_in_either_layout_stack_on_a_projection_with_their_scales() {
        let dir = std::env::temp_dir().join(format!("oaiy-image-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, peft, kohya, other) = (dir.join("model.safetensors"), dir.join("peft.safetensors"), dir.join("kohya.safetensors"), dir.join("other.safetensors"));
        let q = "transformer_blocks.0.attn.to_q";
        // A 2×2 identity projection.
        write(&model, &[("transformer_blocks.0.attn.to_q.weight", &[2, 2], &[1., 0., 0., 1.])]);
        // Diffusers/PEFT, rank 1, no alpha (scale 1): adds [[0,0],[1,0]]·x, i.e. y1 += x0.
        write(&peft, &[
            ("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", &[1, 2], &[1., 0.]),
            ("transformer.transformer_blocks.0.attn.to_q.lora_B.weight", &[2, 1], &[0., 1.]),
        ]);
        // Kohya, rank 2 with alpha 1 (scale 1/2), strength 2: adds x1 to y0.
        write(&kohya, &[
            ("lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight", &[2, 2], &[0., 1., 0., 0.]),
            ("lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight", &[2, 2], &[1., 0., 0., 0.]),
            ("lora_unet_transformer_blocks_0_attn_to_q.alpha", &[], &[1.]),
            ("lora_unet_transformer_blocks_0_img_mod.lora_down.weight", &[1, 2], &[1., 1.]),
            ("lora_unet_transformer_blocks_0_img_mod.lora_up.weight", &[2, 1], &[1., 1.]),
        ]);
        let mut loras = Loras::open(&[(peft.clone(), 1.), (kohya.clone(), 2.)]).unwrap();
        let mut w = Weights::open(&model).unwrap();
        let linear = w.linear(q, &Device::Cpu, DType::F32, &mut loras).unwrap();
        let y = linear.forward(&Tensor::new(&[[3f32, 5.]], &Device::Cpu).unwrap()).unwrap();
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![3. + 5., 5. + 3.]);
        // The kohya file's other module is not in this model: a note, not an error.
        let notes = loras.check().unwrap();
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("1 of its 2 modules are not in this model"), "{notes:?}");
        // A LoRA that fits nothing, or has the wrong shapes, is refused.
        write(&other, &[("transformer.add_k_proj.lora_A.weight", &[1, 2], &[1., 0.]), ("transformer.add_k_proj.lora_B.weight", &[2, 1], &[1., 0.])]);
        let mut none = Loras::open(&[(other.clone(), 1.)]).unwrap();
        w.linear(q, &Device::Cpu, DType::F32, &mut none).unwrap();
        assert!(none.check().unwrap_err().to_string().contains("fits none of this Qwen Image 2.1 model's projections"));
        write(&other, &[("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", &[1, 3], &[1., 0., 0.]), ("transformer.transformer_blocks.0.attn.to_q.lora_B.weight", &[2, 1], &[1., 0.])]);
        let mut wrong = Loras::open(&[(other, 1.)]).unwrap();
        assert!(w.linear(q, &Device::Cpu, DType::F32, &mut wrong).err().unwrap().to_string().contains("does not fit transformer_blocks.0.attn.to_q"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_fused_gate_up_lora_is_split_between_the_gate_and_up_projections() {
        let dir = std::env::temp_dir().join(format!("oaiy-image-lora-fused-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("comfy.safetensors");
        // Rank 1, input 2, fused output 2 × 2: gate rows [1, 2], up rows [3, 4].
        write(&file, &[
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight", &[1, 2], &[1., 1.]),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight", &[4, 1], &[1., 2., 3., 4.]),
        ]);
        let mut loras = Loras::open(&[(file, 1.)]).unwrap();
        let x = Tensor::new(&[[1f32, 1.]], &Device::Cpu).unwrap();
        for (module, want) in [("transformer_blocks.0.img_mlp.gate_layer", [2f32, 4.]), ("transformer_blocks.0.img_mlp.proj", [6., 8.])] {
            let f = loras.factors(module, 2, 2, &Device::Cpu, DType::F32).unwrap();
            assert_eq!(f.len(), 1);
            let (a, b) = &f[0];
            let y = x.matmul(&a.t().unwrap()).unwrap().matmul(&b.t().unwrap()).unwrap();
            assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), want.to_vec(), "{module}");
        }
        assert!(loras.check().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The Qwen Image 2.1 LoRAs in OAIY_QWEN_LORAS (a folder) fit every block's projections, whole.
    #[test]
    #[ignore = "needs OAIY_QWEN_LORAS, a folder of Qwen Image 2.1 LoRAs"]
    fn real_qwen_image_loras_fit_every_block() {
        let dir = PathBuf::from(std::env::var("OAIY_QWEN_LORAS").expect("OAIY_QWEN_LORAS"));
        let files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
        assert!(!files.is_empty());
        for file in files {
            let mut loras = Loras::open(&[(file.clone(), 1.)]).unwrap();
            let mut fitted = 0;
            for i in 0..32 {
                for (name, out, input) in [("attn.to_q", 4096, 4096), ("attn.to_k", 4096, 4096), ("attn.to_v", 4096, 4096), ("attn.to_out.0", 4096, 4096), ("img_mlp.gate_layer", 12288, 4096), ("img_mlp.proj", 12288, 4096), ("img_mlp.out", 4096, 12288)] {
                    fitted += loras.factors(&format!("transformer_blocks.{i}.{name}"), out, input, &Device::Cpu, DType::F32).unwrap().len();
                }
            }
            assert_eq!(fitted, 32 * 7, "{}", file.display());
            assert!(loras.check().unwrap().is_empty(), "{}", file.display());
        }
    }

    /// A diffusers file keeps its scale in its metadata, not as alpha tensors: `lora_alpha` over the rank, or a
    /// module's own alpha from `alpha_pattern`.
    #[test]
    fn a_diffusers_files_scale_is_its_metadatas() {
        let dir = std::env::temp_dir().join(format!("oaiy-image-lora-peft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("peft.safetensors");
        let meta = r#"{"transformer.r": 2, "transformer.lora_alpha": 8, "transformer.alpha_pattern": {"to_k": 1}, "transformer.use_rslora": false}"#;
        write_f32(&file, &[
            ("transformer.blocks.0.to_q.lora_A.weight", &[2, 2], &[1., 0., 0., 1.]),
            ("transformer.blocks.0.to_q.lora_B.weight", &[2, 2], &[1., 0., 0., 1.]),
            ("transformer.blocks.0.to_k.lora_A.weight", &[2, 2], &[1., 0., 0., 1.]),
            ("transformer.blocks.0.to_k.lora_B.weight", &[2, 2], &[1., 0., 0., 1.]),
        ], &[("lora_adapter_metadata", meta), ("format", "pt")]);
        let mut loras = Loras::open(&[(file, 0.5)]).unwrap();
        for (module, scale) in [("blocks.0.to_q", 0.5 * 8. / 2.), ("blocks.0.to_k", 0.5 * 1. / 2.)] {
            let f = loras.factors(module, 2, 2, &Device::Cpu, DType::F32).unwrap();
            let up = f[0].1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(up, vec![scale as f32, 0., 0., scale as f32], "{module}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn renamed(module: &str) -> Option<Alias> {
        match module {
            "block.q" => Some(Alias { name: "block.attn.to_q".into(), swap_halves: false }),
            "final.modulation" => Some(Alias { name: "norm_out.linear".into(), swap_halves: true }),
            _ => None,
        }
    }

    /// A module another trainer names otherwise is found by that name, the halves of its up factor's rows put the
    /// other way round where the model keeps them so; the preflight knows the names too.
    #[test]
    fn an_alias_finds_a_module_by_another_name_and_may_swap_its_halves() {
        let dir = std::env::temp_dir().join(format!("oaiy-image-lora-alias-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("renamed.safetensors");
        write_f32(&file, &[
            ("transformer.block.attn.to_q.lora_A.weight", &[1, 2], &[1., 0.]),
            ("transformer.block.attn.to_q.lora_B.weight", &[2, 1], &[3., 4.]),
            ("transformer.norm_out.linear.lora_A.weight", &[1, 2], &[0., 1.]),
            ("transformer.norm_out.linear.lora_B.weight", &[4, 1], &[1., 2., 3., 4.]),
        ], &[]);
        let expected = vec![("block.q".to_owned(), 2, 2), ("final.modulation".to_owned(), 4, 2)];
        let plain = Loras::open(&[(file.clone(), 1.)]).unwrap();
        assert!(plain.validate_modules(&expected).unwrap_err().to_string().contains("fits none of the selected architecture's projections"));
        let mut loras = Loras::open(&[(file, 1.)]).unwrap().for_model("a test model", Some(renamed));
        loras.validate_modules(&expected).unwrap();
        let up = |f: Vec<(Tensor, Tensor)>| f[0].1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(up(loras.factors("block.q", 2, 2, &Device::Cpu, DType::F32).unwrap()), vec![3., 4.]);
        assert_eq!(up(loras.factors("final.modulation", 4, 2, &Device::Cpu, DType::F32).unwrap()), vec![3., 4., 1., 2.]);
        assert!(loras.check().unwrap().is_empty());
        // (a projection it does not name: nothing)
        assert!(loras.factors("block.k", 2, 2, &Device::Cpu, DType::F32).unwrap().is_empty());
        let fewer = vec![expected[0].clone()];
        assert!(loras.validate_modules(&fewer).unwrap_err().to_string().contains("adapts norm_out.linear, which is not in a test model"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file that keeps more than LoRA factors (a distillation extract's norm changes and new layers) would be
    /// applied only in part: the preflight refuses it.
    #[test]
    fn a_file_with_more_than_lora_factors_is_refused() {
        let dir = std::env::temp_dir().join(format!("oaiy-image-lora-extras-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("extract.safetensors");
        write_f32(&file, &[
            ("block.q.lora_A.weight", &[1, 2], &[1., 0.]),
            ("block.q.lora_B.weight", &[2, 1], &[1., 0.]),
            ("block.q.alpha", &[], &[1.]),
            ("delta.block.norm.scale", &[2], &[0., 0.]),
        ], &[]);
        let err = Loras::open(&[(file, 1.)]).unwrap().validate_modules(&[("block.q".to_owned(), 2, 2)]).unwrap_err().to_string();
        assert!(err.contains("holds 1 tensors besides LoRA factors (e.g. delta.block.norm.scale)"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn modules_are_named_as_the_model_names_them() {
        assert_eq!(module_of("transformer.transformer_blocks.0.attn.to_q.lora_A.weight").as_deref(), Some("transformer_blocks.0.attn.to_q"));
        assert_eq!(module_of("diffusion_model.transformer_blocks.3.img_mlp.proj.lora_down.weight").as_deref(), Some("transformer_blocks.3.img_mlp.proj"));
        assert_eq!(module_of("transformer_blocks.1.attn.to_out.0.lora_B.weight"), None);
        let keys = layouts("transformer_blocks.0.attn.to_q");
        assert!(keys.iter().any(|k| k.down == "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight"));
        assert!(keys.iter().any(|k| k.alpha == "transformer.transformer_blocks.0.attn.to_q.alpha"));
    }
}
