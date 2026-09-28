//! LoRA adapters for the Qwen Image transformer, applied at run time beside
//! each projection (the base weights stay as they are, quantized or not).
//!
//! Several adapters stack: the turbo adapter and any number of style or
//! character LoRAs, each with a strength. Both common key layouts are read:
//! diffusers/PEFT (`transformer.{module}.lora_A.weight`, `.lora_B.weight`,
//! optional `.alpha`) and kohya/ComfyUI (`diffusion_model.{module}.lora_down`,
//! `.lora_up`, `.alpha`, or `lora_unet_{module with _}.lora_down`). The
//! effective scale is strength × alpha / rank (alpha = rank when absent), folded
//! into the up factor.

use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
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
}

/// The adapters for one transformer.
#[derive(Default)]
pub struct Loras {
    adapters: Vec<Adapter>,
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
        Ok(Self { path: path.to_owned(), weights, strength, modules, used: BTreeSet::new() })
    }

    /// Whether it adapts `module` (kohya's flat names count too).
    fn adapts(&self, module: &str) -> bool {
        self.modules.contains(module) || self.modules.contains(&format!("lora_unet_{}", module.replace('.', "_")))
    }

    /// The (down, up × scale) factors for `module`, for a projection of shape [out, input].
    fn factors(&mut self, module: &str, out: usize, input: usize, dev: &Device, dtype: DType) -> Result<Option<(Tensor, Tensor)>> {
        if let Some(f) = self.factors_as(module, module, out, None, input, dev, dtype)? {
            return Ok(Some(f));
        }
        // Comfy's fused MLP: one `img_mlp.gate_up` factor pair whose up rows are
        // the gate rows then the up rows, as the fused base weight stores them.
        for (suffix, half) in [(".img_mlp.gate_layer", 0), (".img_mlp.proj", 1)] {
            if let Some(block) = module.strip_suffix(suffix) {
                let fused = format!("{block}.img_mlp.gate_up");
                return self.factors_as(&fused, module, out, Some(half), input, dev, dtype);
            }
        }
        Ok(None)
    }

    /// The factors stored for `stored`, for the projection `module`; with
    /// `half`, the up factor is fused and only that half of its rows is ours.
    #[allow(clippy::too_many_arguments)]
    fn factors_as(&mut self, stored: &str, module: &str, out: usize, half: Option<usize>, input: usize, dev: &Device, dtype: DType) -> Result<Option<(Tensor, Tensor)>> {
        for keys in layouts(stored) {
            if !self.weights.has(&keys.down) {
                continue;
            }
            let down = self.weights.tensor(&keys.down, dev, DType::F32)?;
            let mut up = self.weights.tensor(&keys.up, dev, DType::F32)?;
            let (rank, down_in) = down.dims2()?;
            if let Some(half) = half {
                let rows = up.dim(0)?;
                if rows != 2 * out {
                    candle_core::bail!("LoRA {} does not fit {module}: its fused {stored} has {rows} rows, not 2 × {out}", self.path.display());
                }
                up = up.narrow(0, half * out, out)?;
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
            let scale = self.strength * alpha.map_or(1., |a| a / rank as f64);
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
        Ok(Self { adapters })
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Whether the first adapter (the turbo one, when there is one) adapts `module`.
    pub fn first_adapts(&self, module: &str) -> bool {
        self.adapters.first().is_some_and(|a| a.adapts(module))
    }

    /// Every adapter's factors for `module`, a projection of shape [out, input].
    pub fn factors(&mut self, module: &str, out: usize, input: usize, dev: &Device, dtype: DType) -> Result<Vec<(Tensor, Tensor)>> {
        let mut all = Vec::new();
        for adapter in &mut self.adapters {
            if let Some(f) = adapter.factors(module, out, input, dev, dtype)? {
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
                candle_core::bail!(
                    "LoRA {} fits none of this Qwen Image 2.1 model's projections (it may be for another model, such as the two-stream Qwen-Image or Qwen-Image-Edit)",
                    a.path.display()
                );
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

#[cfg(test)]
mod tests {
    use super::*;
    use oaiy_engine::json::Json;

    /// A safetensors file of F32 tensors.
    fn write(path: &Path, tensors: &[(&str, &[usize], &[f32])]) {
        let (mut header, mut data) = (Vec::new(), Vec::new());
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
