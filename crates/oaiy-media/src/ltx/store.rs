//! Bounded weight residency. SSD reads use the published safetensors directly.
//!
//! Besides BF16/F16/F32 weights, fp8 (F8_E4M3) and int8 (I8) checkpoints load
//! as published: each weight may carry a per-tensor (or per-row)
//! `<name>.weight_scale`, as ComfyUI's scaled checkpoints store it. They stay at
//! their stored size on disk and in the RAM tier, and become BF16 on the device.
//! NVFP4 checkpoints (ComfyUI's, and Lightricks' own `-nvfp4` releases) load too:
//! two E2M1 values per byte, high nibble first, an fp8 `weight_scale` per block
//! of 16 in the cuBLAS tiled layout, and a tensor-wide `weight_scale_2`.
//! Transformers saved with bare names (`patchify_proj.weight`) are served under
//! the `model.diffusion_model.` prefix the loader uses.
//!
//! A LoRA (`<module>.lora_A.weight` / `lora_B.weight`, as ID-LoRA and the LTX
//! trainer save them) is added to its weights as they are read: `W + s B A`.
use candle_core::{DType, Device, Result, Tensor};
use dsv41::safetensors::{Dtype, StIndex};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn int8_rows_round_trip_within_half_a_step() -> Result<()> {
        let dev = Device::Cpu;
        let values: Vec<f32> = (0..4 * 300).map(|i| ((i * 37 % 101) as f32 - 50.) * if i < 300 { 0.01 } else { 3. }).collect();
        let w = Tensor::from_vec(values.clone(), (4, 300), &dev)?;
        let (codes, scale) = quantize_rows(&w)?;
        assert_eq!(codes.dtype(), DType::U8);
        let back = codes.to_dtype(DType::F32)?.affine(1., -128.)?.broadcast_mul(&scale.to_dtype(DType::F32)?)?.to_vec2::<f32>()?;
        let scales = scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for (r, row) in back.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                // Half a quantization step, plus BF16's rounding of the scale.
                assert!((v - values[r * 300 + c]).abs() <= scales[r] * 0.51 + values[r * 300 + c].abs() * 0.008, "{r} {c}");
            }
        }
        Ok(())
    }
    #[test]
    fn comfy_convrot_int8_decodes_as_the_reference_does() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("oaiy-ltx-convrot-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let file = dir.join("m.safetensors");
        let key = "model.diffusion_model.blk.to_q.weight";
        // 2 rows × 16 columns of INT8 codes, a per-row scale, rotated in groups of 16.
        let codes: Vec<u8> = (0..32i32).map(|i| ((i * 37 % 200) - 100) as i8 as u8).collect();
        let scales: Vec<u8> = [0.5f32, 0.25].iter().flat_map(|x| x.to_le_bytes()).collect();
        let note = br#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 16}"#.to_vec();
        let mut header = Vec::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in [
            (key.to_owned(), "I8", vec![2, 16], codes),
            ("model.diffusion_model.blk.to_q.weight_scale".to_owned(), "F32", vec![2, 1], scales),
            ("model.diffusion_model.blk.to_q.comfy_quant".to_owned(), "U8", vec![note.len()], note),
        ] {
            header.push(format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{},{}]}}", data.len(), data.len() + bytes.len()));
            data.extend(bytes);
        }
        let header = format!("{{{}}}", header.join(","));
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header.as_bytes());
        out.extend(data);
        std::fs::write(&file, out)?;
        let mut store = Store::open(&file, 0)?;
        let ours = store.tensor(key, &Device::Cpu, false)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let reference = crate::comfy_quant::load(&store.index, key, &Device::Cpu, DType::F32)?.unwrap().flatten_all()?.to_vec1::<f32>()?;
        for (a, b) in ours.iter().zip(&reference) {
            // BF16 storage of the result.
            assert!((a - b).abs() <= b.abs() * 0.008 + 1e-3, "{a} vs {b}");
        }
        // Without undoing the rotation the values would differ well beyond that.
        let raw = decode(key, Dtype::I8, &[2, 16], &store.index.read(key).unwrap(), Some(&[0.5, 0.25]), &Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(raw.iter().zip(&reference).any(|(a, b)| (a - b).abs() > 1.));
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn lora_adds_scaled_b_times_a_to_its_weight() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("oaiy-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let write = |path: &Path, tensors: &[(&str, &[usize], Vec<u8>)]| -> Result<()> {
            let mut header = String::from("{");
            let mut data = Vec::new();
            for (i, (name, shape, bytes)) in tensors.iter().enumerate() {
                if i > 0 { header.push(','); }
                header.push_str(&format!("\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{},{}]}}", data.len(), data.len() + bytes.len()));
                data.extend_from_slice(bytes);
            }
            header.push('}');
            let mut out = (header.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(header.as_bytes());
            out.extend(data);
            std::fs::write(path, out)?;
            Ok(())
        };
        let (model, lora) = (dir.join("m.safetensors"), dir.join("l.safetensors"));
        // W (2x3) = 1; A (1x3) = [1,2,3]; B (2x1) = [1,-1]; scale 0.5.
        write(&model, &[("model.diffusion_model.blk.to_q.weight", &[2, 3], f32s(&[1.; 6]))])?;
        write(&lora, &[
            ("diffusion_model.blk.to_q.lora_A.weight", &[1, 3], f32s(&[1., 2., 3.])),
            ("diffusion_model.blk.to_q.lora_B.weight", &[2, 1], f32s(&[1., -1.])),
        ])?;
        let mut store = Store::open(&model, 0)?;
        assert_eq!(store.add_lora(&lora, 0.5)?, 1);
        let w = store.tensor("model.diffusion_model.blk.to_q.weight", &Device::Cpu, false)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(w, vec![vec![1.5, 2.0, 2.5], vec![0.5, 0.0, -0.5]]);
        // A transposed pair is refused, not merged.
        write(&lora, &[
            ("diffusion_model.blk.to_q.lora_A.weight", &[3, 1], f32s(&[1., 2., 3.])),
            ("diffusion_model.blk.to_q.lora_B.weight", &[1, 2], f32s(&[1., -1.])),
        ])?;
        assert!(Store::open(&model, 0)?.add_lora(&lora, 1.).is_err());
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
    #[test]
    fn nvfp4_unpacks_high_nibble_first_with_tiled_block_scales() -> Result<()> {
        // 128 rows x 64 values: 32 bytes a row, 4 blocks of 16 a row.
        let (rows, cols) = (128usize, 64usize);
        let bytes: Vec<u8> = (0..rows * cols / 2).map(|i| if i % 32 == 0 { 0x2F } else { 0x00 }).collect();
        // Row r, block c: scale 1 + r (as fp8 would round it, so keep r small) for block 0.
        let mut plain = vec![0x38u8; rows * cols / 16]; // fp8 1.0
        plain[5 * 4] = 0x40; // row 5, block 0: fp8 2.0
        // Tile them as cuBLAS stores them.
        let mut tiled = vec![0u8; plain.len()];
        for (at, t) in tiled.iter_mut().enumerate() {
            let (k, j, i) = (at % 4, at / 4 % 4, at / 16 % 32);
            *t = plain[(j * 32 + i) * 4 + k];
        }
        assert_eq!(untile_scales(&tiled, rows, cols / 16)?, plain);
        let scales: Vec<f32> = plain.iter().map(|&b| e4m3_table()[b as usize] * 0.5).collect();
        let t = decode_as("w", Dtype::U8, &[rows, cols / 2], &bytes, Some(&scales), &Device::Cpu, DType::F32)?;
        let v = t.to_vec2::<f32>()?;
        // 0x2F: high nibble 2 (1.0) first, then low nibble 15 (-6.0); tensor scale 0.5.
        assert_eq!((v[0][0], v[0][1], v[0][2]), (0.5, -3.0, 0.0));
        assert_eq!((v[5][0], v[5][1]), (1.0, -6.0));
        Ok(())
    }
    /// A real NVFP4 layer against the same layer of a BF16 relative (the
    /// distilled model): the right layout correlates near 1, a wrong one near 0.
    #[test]
    #[ignore = "needs an NVFP4 LTX checkpoint and a BF16 relative; OAIY_LTX_NVFP4, OAIY_LTX_BF16"]
    fn nvfp4_layer_matches_its_bf16_relative() -> Result<()> {
        let (Some(q), Some(b)) = (std::env::var_os("OAIY_LTX_NVFP4"), std::env::var_os("OAIY_LTX_BF16")) else { return Ok(()) };
        let name = "model.diffusion_model.transformer_blocks.10.attn1.to_q.weight";
        let a = Store::open(Path::new(&q), 0)?.tensor(name, &Device::Cpu, false)?.to_dtype(DType::F32)?.flatten_all()?;
        let r = Store::open(Path::new(&b), 0)?.tensor(name, &Device::Cpu, false)?.to_dtype(DType::F32)?.flatten_all()?;
        let centre = |t: &Tensor| -> Result<Tensor> { t.broadcast_sub(&t.mean_all()?) };
        let (a, r) = (centre(&a)?, centre(&r)?);
        let dot = (&a * &r)?.sum_all()?.to_scalar::<f32>()?;
        let corr = dot / (a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() * r.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt());
        eprintln!("NVFP4 vs BF16 correlation {corr}");
        if let Some(dump) = std::env::var_os("OAIY_LTX_NVFP4_REF") {
            let expected = candle_core::safetensors::load(Path::new(&dump), &Device::Cpu)?.remove("w").unwrap().flatten_all()?;
            let mine = Store::open(Path::new(&q), 0)?.tensor(name, &Device::Cpu, false)?.to_dtype(DType::F32)?.flatten_all()?;
            let err = (&mine - &expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("NVFP4 decode vs reference dequantization: relative RMS {err:e}");
            assert!(err < 5e-3, "{err}");
        }
        assert!(corr > 0.99, "{corr}");
        Ok(())
    }
    #[test]
    fn embedding_rows_preserve_order_and_check_bounds() -> Result<()> {
        let path = std::env::temp_dir().join(format!("oaiy-rows-{}-{}.safetensors", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let header = br#"{"embedding":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header);
        file.extend((1..=6).flat_map(|n| (n as f32).to_le_bytes()));
        std::fs::write(&path, file)?;
        let mut store = Store::open(&path, 0)?;
        let selected = store.rows("embedding", &[2, 0, 2], &Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(selected, vec![5., 6., 1., 2., 5., 6.]);
        assert!(store.rows("embedding", &[3], &Device::Cpu).is_err());
        assert_eq!(store.disk_bytes, 24);
        std::fs::remove_file(path)?;
        Ok(())
    }

    fn write_st(name: &str, tensors: &[(&str, &str, &[usize], Vec<u8>)], meta: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("oaiy-{name}-{}.safetensors", std::process::id()));
        let (mut header, mut data) = (format!("{{\"__metadata__\":{meta}"), Vec::new());
        for (k, dtype, shape, bytes) in tensors {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header += &format!(",\"{k}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", dims.join(","), data.len(), data.len() + bytes.len());
            data.extend_from_slice(bytes);
        }
        header.push('}');
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend(data);
        std::fs::write(&path, file).unwrap();
        path
    }

    #[test]
    #[ignore = "needs a CUDA device and --features cuda"]
    fn quantized_weights_decode_the_same_on_the_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let bytes: Vec<u8> = (0..=255u8).collect();
        for (dtype, scale) in [(Dtype::F8E4M3, Some(&[0.37f32][..])), (Dtype::I8, Some(&[0.01f32][..])), (Dtype::F8E4M3, None)] {
            let cpu = decode("x.weight", dtype, &[16, 16], &bytes, scale, &Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let gpu = decode("x.weight", dtype, &[16, 16], &bytes, scale, &dev)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
                assert!(a == b || (a.is_nan() && b.is_nan()), "{dtype:?} byte {i}: cpu {a} gpu {b}");
            }
        }
        Ok(())
    }

    #[test]
    fn the_e4m3_table_matches_the_reference_conversion() -> Result<()> {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let reference = Tensor::from_raw_buffer(&bytes, DType::F8E4M3, &[256], &Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for (b, (want, got)) in reference.iter().zip(e4m3_table()).enumerate() {
            assert!(want == got || (want.is_nan() && got.is_nan()), "byte {b:#04x}: {want} vs {got}");
        }
        Ok(())
    }

    #[test]
    fn scaled_fp8_and_int8_weights_decode_to_bf16() -> Result<()> {
        // fp8 e4m3: 0x38 = 1.0, 0x40 = 2.0, 0xB8 = -1.0, 0x00 = 0.
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let path = write_st("quant", &[
            ("patchify_proj.weight", "F8_E4M3", &[2, 2], vec![0x38, 0x40, 0xB8, 0x00]),
            ("patchify_proj.weight_scale", "F32", &[1], f32s(&[0.5])),
            ("emb.weight", "I8", &[3, 2], vec![1, 255, 127, 128, 0, 2]),
            ("emb.weight_scale", "F32", &[1], f32s(&[0.25])),
            ("rows.weight", "I8", &[2, 2], vec![2, 4, 2, 4]),
            ("rows.weight_scale", "F32", &[2], f32s(&[1.0, 0.5])),
        ], "{\"model_version\":\"2.5.0\"}");
        let mut store = Store::open(&path, 1 << 20)?;
        let get = |t: Tensor| t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Bare transformer names are served under the loader's prefix.
        let w = store.tensor("model.diffusion_model.patchify_proj.weight", &Device::Cpu, true)?;
        assert_eq!(get(w), vec![0.5, 1.0, -0.5, 0.0]);
        // The RAM tier keeps the stored bytes and decodes the same way.
        let again = store.tensor("model.diffusion_model.patchify_proj.weight", &Device::Cpu, true)?;
        assert_eq!(get(again), vec![0.5, 1.0, -0.5, 0.0]);
        assert_eq!(store.host_bytes, 4);
        let rows = store.rows("model.diffusion_model.emb.weight", &[1, 0], &Device::Cpu)?;
        assert_eq!(get(rows), vec![31.75, -32.0, 0.25, -0.25]);
        // Every int8 byte, 0xFF included.
        let all: Vec<f32> = (0..=255u8).map(|b| b as i8 as f32 * 0.25).collect();
        let decoded = decode("all.weight", Dtype::I8, &[256], &(0..=255u8).collect::<Vec<_>>(), Some(&[0.25]), &Device::Cpu)?;
        assert_eq!(get(decoded), all);
        let per_row = store.tensor("model.diffusion_model.rows.weight", &Device::Cpu, false)?;
        assert_eq!(get(per_row), vec![2.0, 4.0, 1.0, 2.0]);
        // Scales are applied, not loaded as weights of their own.
        let g = store.group("model.diffusion_model.", &Device::Cpu, false, |_| true)?;
        assert!(g.tensors.keys().all(|k| !k.ends_with("weight_scale")));
        std::fs::remove_file(path)?;
        Ok(())
    }
}

/// A LoRA's two factors for one weight, on the host.
struct LoraPair {
    a: Tensor,
    b: Tensor,
}

pub struct Store {
    pub index: StIndex,
    /// LoRA factors by the weight they adapt, and their scale.
    lora: HashMap<String, LoraPair>,
    lora_scale: f64,
    host: HashMap<String, Vec<u8>>,
    /// `weight_scale` values already read, by weight name.
    scales: HashMap<String, Option<Vec<f32>>>,
    /// Comfy ConvRot group sizes already read, by weight name.
    rotations: HashMap<String, Option<usize>>,
    pub host_bytes: u64,
    pub disk_bytes: u64,
    budget: u64,
}
impl Store {
    pub fn open(path: &Path, budget: u64) -> Result<Self> {
        let index = if path.is_dir() {
            StIndex::open(path)
        } else {
            StIndex::open_file(path)
        }
        .map_err(candle_core::Error::wrap)?;
        let mut index = index;
        let prefix = super::transformer::PREFIX;
        if index.get("patchify_proj.weight").is_some() && index.get(&format!("{prefix}patchify_proj.weight")).is_none() {
            index.prefix_names(prefix);
        }
        Ok(Self {
            index,
            lora: HashMap::new(),
            lora_scale: 0.,
            host: HashMap::new(),
            scales: HashMap::new(),
            rotations: HashMap::new(),
            host_bytes: 0,
            disk_bytes: 0,
            budget,
        })
    }
    /// Add a LoRA: every `<module>.lora_A.weight` / `lora_B.weight` pair adapts
    /// that module's weight by `scale * B A` whenever it is read. Returns how
    /// many weights it adapts.
    pub fn add_lora(&mut self, path: &Path, scale: f64) -> Result<usize> {
        let mut file = Store::open(path, 0)?;
        let names: Vec<String> = file.index.names().map(str::to_owned).collect();
        let prefix = super::transformer::PREFIX;
        for name in &names {
            let Some(module) = name.strip_suffix(".lora_A.weight") else { continue };
            let b_name = format!("{module}.lora_B.weight");
            if !names.contains(&b_name) {
                candle_core::bail!("LoRA {} lacks {b_name}", path.display());
            }
            // `diffusion_model.x`, `model.diffusion_model.x` or bare `x`.
            let bare = module.strip_prefix("model.diffusion_model.").or_else(|| module.strip_prefix("diffusion_model.")).unwrap_or(module);
            let weight = format!("{prefix}{bare}.weight");
            let info = self.index.info(&weight).map_err(|_| candle_core::Error::Msg(format!("LoRA {}: the model has no {weight}", path.display())))?.clone();
            let a = file.tensor(name, &Device::Cpu, false)?;
            let b = file.tensor(&b_name, &Device::Cpu, false)?;
            // A is (rank, in) and B (out, rank); B A must be the weight's shape
            // (an NVFP4 weight stores half its columns).
            let columns = if self.nvfp4(&weight) { info.shape[1] * 2 } else { info.shape[1] };
            let (out, rank) = b.dims2()?;
            let (rank_a, inner) = a.dims2()?;
            if rank != rank_a || info.shape.len() != 2 || out != info.shape[0] || inner != columns {
                candle_core::bail!("LoRA {} for {weight}: B {:?} A {:?} do not make its {:?}", path.display(), b.dims(), a.dims(), info.shape);
            }
            self.lora.insert(weight, LoraPair { a, b });
        }
        self.lora_scale = scale;
        Ok(self.lora.len())
    }

    pub fn tensor(&mut self, key: &str, dev: &Device, cache: bool) -> Result<Tensor> {
        let t = self.tensor_plain(key, dev, cache)?;
        match self.lora.get(key) {
            None => Ok(t),
            Some(l) => {
                let delta = l.b.to_device(dev)?.to_dtype(DType::F32)?.matmul(&l.a.to_device(dev)?.to_dtype(DType::F32)?)?;
                (t.to_dtype(DType::F32)? + (delta * self.lora_scale)?)?.to_dtype(t.dtype())
            }
        }
    }

    fn tensor_plain(&mut self, key: &str, dev: &Device, cache: bool) -> Result<Tensor> {
        let info = self
            .index
            .info(key)
            .map_err(candle_core::Error::wrap)?
            .clone();
        let scale = self.scale(key)?;
        let rotation = self.rotation(key)?;
        if let Some(bytes) = self.host.get(key) {
            let t = decode(key, info.dtype, &info.shape, bytes, scale.as_deref(), dev)?;
            return unrotated(t, rotation, DType::BF16);
        }
        let bytes = self.index.read(key).map_err(candle_core::Error::wrap)?;
        self.disk_bytes += bytes.len() as u64;
        let t = decode(key, info.dtype, &info.shape, &bytes, scale.as_deref(), dev)?;
        if cache && self.host_bytes + bytes.len() as u64 <= self.budget {
            self.host_bytes += bytes.len() as u64;
            self.host.insert(key.to_owned(), bytes);
        }
        unrotated(t, rotation, DType::BF16)
    }

    /// A Comfy ConvRot weight's group size (its rows were Hadamard-rotated before
    /// quantization), read once from its `.comfy_quant` note.
    fn rotation(&mut self, key: &str) -> Result<Option<usize>> {
        if let Some(r) = self.rotations.get(key) {
            return Ok(*r);
        }
        let r = crate::comfy_quant::rotation(&self.index, key)?;
        self.rotations.insert(key.to_owned(), r);
        Ok(r)
    }
    /// A tensor at full precision (the audio decoder and vocoder run in F32).
    pub fn tensor_f32(&mut self, key: &str, dev: &Device) -> Result<Tensor> {
        let info = self.index.info(key).map_err(candle_core::Error::wrap)?.clone();
        let scale = self.scale(key)?;
        let rotation = self.rotation(key)?;
        let bytes = self.index.read(key).map_err(candle_core::Error::wrap)?;
        self.disk_bytes += bytes.len() as u64;
        let t = decode_as(key, info.dtype, &info.shape, &bytes, scale.as_deref(), dev, DType::F32)?;
        unrotated(t, rotation, DType::F32)
    }
    /// Fetch only prompt vocabulary rows, avoiding a multi-gigabyte embedding upload.
    pub fn rows(&mut self, key: &str, ids: &[u32], dev: &Device) -> Result<Tensor> {
        use std::io::{Read, Seek, SeekFrom};
        let info = self.index.info(key).map_err(candle_core::Error::wrap)?;
        if info.shape.len() != 2 || ids.iter().any(|&i| i as usize >= info.shape[0]) {
            candle_core::bail!("invalid embedding row selection for {key}");
        }
        let stored = info.dtype;
        if !matches!(stored, Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F8E4M3 | Dtype::I8) {
            candle_core::bail!("unsupported embedding dtype {stored:?}");
        }
        let width = info.shape[1];
        let row_bytes = width
            .checked_mul(stored.size())
            .ok_or_else(|| candle_core::Error::Msg("embedding row overflow".into()))?;
        let size = ids
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| candle_core::Error::Msg("embedding selection overflow".into()))?;
        let mut bytes = vec![0u8; size];
        let mut file = std::fs::File::open(self.index.shard_path(info.shard))?;
        for (i, &id) in ids.iter().enumerate() {
            let offset = (id as u64)
                .checked_mul(row_bytes as u64)
                .ok_or_else(|| candle_core::Error::Msg("embedding offset overflow".into()))?;
            if offset + row_bytes as u64 > info.nbytes {
                candle_core::bail!("embedding row outside tensor");
            }
            file.seek(SeekFrom::Start(info.start.checked_add(offset).ok_or_else(
                || candle_core::Error::Msg("embedding file offset overflow".into()),
            )?))?;
            file.read_exact(&mut bytes[i * row_bytes..(i + 1) * row_bytes])?;
        }
        self.disk_bytes += size as u64;
        let scale = match self.scale(key)? {
            // A per-row scale follows the selected rows.
            Some(s) if s.len() > 1 => Some(ids.iter().map(|&i| s[i as usize]).collect()),
            other => other,
        };
        decode(key, stored, &[ids.len(), width], &bytes, scale.as_deref(), dev)
    }

    /// The `weight_scale` stored beside a quantized weight, if any.
    fn scale(&mut self, key: &str) -> Result<Option<Vec<f32>>> {
        let Some(base) = key.strip_suffix(".weight") else { return Ok(None) };
        if let Some(s) = self.scales.get(key) {
            return Ok(s.clone());
        }
        let name = format!("{base}.weight_scale");
        let found = match self.index.get(&name).cloned() {
            None => None,
            Some(info) => {
                let bytes = self.index.read(&name).map_err(candle_core::Error::wrap)?;
                let values: Vec<f32> = match info.dtype {
                    Dtype::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                    Dtype::BF16 => bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
                    // NVFP4: one fp8 scale per 16 weights, tiled, times the tensor's own.
                    Dtype::F8E4M3 if self.nvfp4(key) => {
                        let global_name = format!("{base}.weight_scale_2");
                        let g = self.index.read(&global_name).map_err(candle_core::Error::wrap)?;
                        let global = f32::from_le_bytes(g.get(..4).and_then(|b| b.try_into().ok()).ok_or_else(|| candle_core::Error::Msg(format!("{global_name} is not one F32")))?);
                        let (rows, cols) = match info.shape.as_slice() {
                            [r, c] => (*r, *c),
                            _ => candle_core::bail!("LTX tensor {name}: NVFP4 scales must be 2D"),
                        };
                        untile_scales(&bytes, rows, cols)?.into_iter().map(|b| e4m3_table()[b as usize] * global).collect()
                    }
                    other => candle_core::bail!("LTX tensor {name}: unsupported scale dtype {other:?}"),
                };
                Some(values)
            }
        };
        self.scales.insert(key.to_owned(), found.clone());
        Ok(found)
    }
    /// A weight packed as NVFP4 (two values per stored byte).
    fn nvfp4(&self, key: &str) -> bool {
        key.strip_suffix(".weight").is_some_and(|base| {
            self.index.get(&format!("{base}.weight_scale_2")).is_some()
                && self.index.get(key).is_some_and(|i| i.dtype == Dtype::U8)
        })
    }

    pub fn group_bytes(&self, prefix: &str, select: impl Fn(&str) -> bool) -> Result<u64> {
        let mut bytes = 0;
        for key in self.index.names() {
            let Some(short) = key.strip_prefix(prefix).filter(|s| select(s) && !is_scale(s)) else {
                continue;
            };
            let info = self.index.info(key).map_err(candle_core::Error::wrap)?;
            // All selected floating-point weights become BF16 on the device
            // (an NVFP4 byte holds two).
            let unpacked = if self.nvfp4(key) { 2 } else { 1 };
            bytes += info.shape.iter().product::<usize>() as u64 * 2 * unpacked;
            if info.shape.len() == 2 {
                if let Some(base) = short.strip_suffix(".weight") {
                    let bias = format!("{base}.bias");
                    if select(&bias) && self.index.get(&format!("{prefix}{bias}")).is_some() {
                        bytes += info.shape[0] as u64 * 8 * 2;
                    }
                }
            }
        }
        Ok(bytes)
    }
    pub fn group(
        &mut self,
        prefix: &str,
        dev: &Device,
        cache: bool,
        select: impl Fn(&str) -> bool,
    ) -> Result<Group> {
        let mut names: Vec<_> = self
            .index
            .names()
            .filter_map(|k| {
                k.strip_prefix(prefix)
                    .filter(|s| select(s) && !is_scale(s))
                    .map(|s| (k.to_owned(), s.to_owned()))
            })
            .collect();
        names.sort();
        let mut tensors = HashMap::new();
        let mut bytes = 0;
        for (key, short) in names {
            let t = self.tensor(&key, dev, cache)?;
            bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            tensors.insert(short, t);
        }
        // Include the bias in the GEMM accumulator before its BF16 rounding.
        // Eight extra channels keep the contraction dimension tensor-core aligned.
        let mut biased = HashSet::new();
        let biases: Vec<_> = tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".bias").map(str::to_owned))
            .collect();
        for key in biases {
            let name = format!("{key}.weight");
            let Some(w) = tensors.get(&name) else {
                continue;
            };
            if w.rank() != 2 {
                continue;
            }
            let out = w.dim(0)?;
            let b = tensors.get(&format!("{key}.bias")).unwrap().unsqueeze(1)?;
            let zero = Tensor::zeros((out, 7), w.dtype(), dev)?;
            let packed = Tensor::cat(&[w, &b, &zero], 1)?;
            bytes += (out * 8 * w.dtype().size_in_bytes()) as u64;
            tensors.insert(name, packed);
            biased.insert(key);
        }
        Ok(Group {
            tensors,
            bytes,
            biased,
            int8: HashMap::new(),
        })
    }

    /// A group kept on the device with its large 2-D weights as INT8, one
    /// scale per output row (symmetric, round to nearest): half the BF16
    /// size, so a whole LTX transformer (about 22 GB) stays on one 32 GB card
    /// rather than streaming half of itself from RAM on every pass. Each
    /// weight is read as usual first (LoRA merged, fp8 or NVFP4 decoded).
    pub fn group_int8(&mut self, prefix: &str, dev: &Device, select: impl Fn(&str) -> bool) -> Result<Group> {
        let mut names: Vec<_> = self
            .index
            .names()
            .filter_map(|k| k.strip_prefix(prefix).filter(|s| select(s) && !is_scale(s)).map(|s| (k.to_owned(), s.to_owned())))
            .collect();
        names.sort();
        let mut tensors = HashMap::new();
        let mut int8 = HashMap::new();
        let mut biased = HashSet::new();
        let mut bytes = 0;
        for (key, short) in &names {
            let t = self.tensor(key, dev, false)?;
            match short.strip_suffix(".weight").filter(|_| t.rank() == 2 && t.elem_count() >= INT8_MIN_ELEMENTS) {
                Some(linear) => {
                    let (codes, scale) = quantize_rows(&t)?;
                    bytes += (codes.elem_count() + scale.elem_count() * 2) as u64;
                    if names.iter().any(|(_, s)| *s == format!("{linear}.bias")) {
                        biased.insert(linear.to_owned());
                    }
                    int8.insert(linear.to_owned(), scale);
                    tensors.insert(short.clone(), codes);
                }
                None => {
                    bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
                    tensors.insert(short.clone(), t);
                }
            }
        }
        // Small weights stay dense; with a bias they are packed as `group` does.
        let dense: Vec<String> = tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".bias").map(str::to_owned))
            .filter(|k| !int8.contains_key(k))
            .collect();
        for key in dense {
            let name = format!("{key}.weight");
            let Some(w) = tensors.get(&name).filter(|w| w.rank() == 2) else { continue };
            let out = w.dim(0)?;
            let b = tensors.get(&format!("{key}.bias")).ok_or_else(|| candle_core::Error::Msg("bias".into()))?.unsqueeze(1)?;
            let packed = Tensor::cat(&[w, &b, &Tensor::zeros((out, 7), w.dtype(), dev)?], 1)?;
            bytes += (out * 8 * w.dtype().size_in_bytes()) as u64;
            tensors.insert(name, packed);
            biased.insert(key);
        }
        Ok(Group { tensors, bytes, biased, int8 })
    }

    /// What `group_int8` would keep on the device, in bytes.
    pub fn group_int8_bytes(&self, prefix: &str, select: impl Fn(&str) -> bool) -> Result<u64> {
        let mut bytes = 0;
        for key in self.index.names() {
            let Some(short) = key.strip_prefix(prefix).filter(|s| select(s) && !is_scale(s)) else { continue };
            let info = self.index.info(key).map_err(candle_core::Error::wrap)?;
            let unpacked = if self.nvfp4(key) { 2 } else { 1 };
            let n = info.shape.iter().product::<usize>() as u64 * unpacked;
            bytes += if short.ends_with(".weight") && info.shape.len() == 2 && n >= INT8_MIN_ELEMENTS as u64 { n + info.shape[0] as u64 * 2 } else { n * 2 };
        }
        Ok(bytes)
    }
}
/// Weights smaller than this stay in BF16 when a group is kept as INT8.
const INT8_MIN_ELEMENTS: usize = 1 << 16;

/// Symmetric per-row INT8: codes offset by 128 into U8, and each row's scale
/// (its largest magnitude over 127) as a BF16 column.
fn quantize_rows(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let w = w.to_dtype(DType::F32)?;
    let scale = (w.abs()?.max_keepdim(1)? / 127.)?.clamp(1e-12f32, f32::MAX)?;
    let codes = w.broadcast_div(&scale)?.round()?.clamp(-127f32, 127f32)?.affine(1., 128.)?.to_dtype(DType::U8)?;
    Ok((codes, scale.to_dtype(DType::BF16)?))
}

/// Quantization side tensors, applied with their weight rather than loaded.
fn is_scale(name: &str) -> bool {
    name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_2")
        || name.ends_with(".input_scale")
        // ComfyUI's per-layer quantization note (JSON bytes), not a weight.
        || name.ends_with(".comfy_quant")
}

/// NVFP4 block scales from the cuBLAS tiled layout (128 x 4 tiles, stored as
/// 32 x 4 x 4) to row-major `rows x cols`.
fn untile_scales(tiled: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if rows % 128 != 0 || cols % 4 != 0 || tiled.len() != rows * cols {
        candle_core::bail!("NVFP4 scales of {rows} x {cols} are not whole 128 x 4 tiles");
    }
    let mut out = vec![0u8; rows * cols];
    let col_blocks = cols / 4;
    for (at, &b) in tiled.iter().enumerate() {
        // Stored order: (row block, column block, i, j, k).
        let k = at % 4;
        let j = at / 4 % 4;
        let i = at / 16 % 32;
        let cb = at / 512 % col_blocks;
        let rb = at / 512 / col_blocks;
        out[(rb * 128 + j * 32 + i) * cols + cb * 4 + k] = b;
    }
    Ok(out)
}

/// E2M1: 1 sign, 2 exponent and 1 mantissa bits.
const E2M1: [f32; 16] = [0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.];

/// Every fp8 E4M3 (fn) byte's value: 1 sign, 4 exponent (bias 7) and 3
/// mantissa bits; no infinities, and 0x7F/0xFF are NaN.
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0f32; 256];
        for (b, v) in t.iter_mut().enumerate() {
            let (sign, exp, man) = (if b & 0x80 != 0 { -1.0 } else { 1.0 }, (b >> 3) & 0xF, (b & 7) as f32);
            *v = sign * match (exp, man as u8) {
                (0xF, 7) => f32::NAN,
                (0, _) => man / 8.0 * 2f32.powi(-6),
                _ => (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7),
            };
        }
        t
    })
}

/// Stored bytes as a BF16 tensor on `dev`. fp8 converts on the device; int8
/// (which candle has no type for) goes up as bytes and is re-signed there.
/// `scale` is one value for the whole tensor or one per row.
/// A decoded weight with Comfy ConvRot undone (when it was rotated), as `out`.
fn unrotated(t: Tensor, rotation: Option<usize>, out: DType) -> Result<Tensor> {
    match rotation {
        None => Ok(t),
        Some(gs) => crate::comfy_quant::unrotate(&t, gs)?.to_dtype(out),
    }
}

fn decode(key: &str, dtype: Dtype, shape: &[usize], bytes: &[u8], scale: Option<&[f32]>, dev: &Device) -> Result<Tensor> {
    decode_as(key, dtype, shape, bytes, scale, dev, DType::BF16)
}

fn decode_as(key: &str, dtype: Dtype, shape: &[usize], bytes: &[u8], scale: Option<&[f32]>, dev: &Device, out: DType) -> Result<Tensor> {
    // NVFP4: `rows x cols/2` bytes and one scale per 16 values.
    if let (Dtype::U8, Some(scales), [rows, packed]) = (dtype, scale, shape) {
        let (rows, cols) = (*rows, *packed * 2);
        if cols % 16 == 0 && scales.len() == rows * cols / 16 {
            // Each byte to its two values, high nibble first.
            let pairs: Vec<f32> = (0..256usize).flat_map(|b| [E2M1[b >> 4], E2M1[b & 15]]).collect();
            let table = Tensor::from_vec(pairs, (256, 2), dev)?;
            let codes = Tensor::from_raw_buffer(bytes, DType::U8, &[bytes.len()], dev)?.to_dtype(DType::U32)?;
            let values = table.index_select(&codes, 0)?.reshape((rows, cols / 16, 16))?;
            let s = Tensor::from_slice(scales, (rows, cols / 16, 1), dev)?;
            return values.broadcast_mul(&s)?.reshape((rows, cols))?.to_dtype(out);
        }
    }
    let quantized = matches!(dtype, Dtype::F8E4M3 | Dtype::I8);
    if quantized && scale.is_none_or(|s| s.len() == 1) {
        // Each byte is looked up in a 256-entry table that already carries the
        // tensor's scale, rounded once to the output type: exact, the same on
        // every GPU (candle's own fp8 casts are not built for sm_120), and with
        // no full-precision copy of the weight on the device.
        let s = scale.map_or(1., |s| s[0]);
        let values: Vec<f32> = match dtype {
            Dtype::F8E4M3 => e4m3_table().iter().map(|v| v * s).collect(),
            _ => (0..=255u8).map(|b| b as i8 as f32 * s).collect(),
        };
        let table = Tensor::from_vec(values, 256, dev)?.to_dtype(out)?;
        // Widened to u32: candle reads an index equal to its type's maximum
        // (255 for u8) as padding and yields zero, which would erase every
        // int8 -1 and fp8 0xFF.
        let codes = Tensor::from_raw_buffer(bytes, DType::U8, &[bytes.len()], dev)?.to_dtype(DType::U32)?;
        return table.index_select(&codes, 0)?.reshape(shape);
    }
    let t = match dtype {
        Dtype::BF16 => Tensor::from_raw_buffer(bytes, DType::BF16, shape, dev)?,
        Dtype::F16 => Tensor::from_raw_buffer(bytes, DType::F16, shape, dev)?,
        Dtype::F32 => Tensor::from_raw_buffer(bytes, DType::F32, shape, dev)?,
        Dtype::U8 if scale.is_none() => Tensor::from_raw_buffer(bytes, DType::U8, shape, dev)?,
        // Per-row scales: the unscaled table, then the rows' scales in F32.
        Dtype::F8E4M3 | Dtype::I8 => decode_as(key, dtype, shape, bytes, None, dev, DType::F32)?,
        other => candle_core::bail!("LTX tensor {key}: unsupported dtype {other:?}"),
    };
    let t = match scale {
        None => t,
        Some(rows) if shape.first() == Some(&rows.len()) => {
            let mut dims = vec![1; shape.len()];
            dims[0] = rows.len();
            let s = Tensor::from_slice(rows, dims.as_slice(), dev)?;
            t.to_dtype(DType::F32)?.broadcast_mul(&s)?
        }
        Some(s) => candle_core::bail!("LTX tensor {key}: {} scales for shape {shape:?}", s.len()),
    };
    t.to_dtype(out)
}

pub struct Group {
    pub tensors: HashMap<String, Tensor>,
    pub bytes: u64,
    biased: HashSet<String>,
    /// Weights kept as INT8 (their `tensors` entry holds the codes, offset by
    /// 128 in U8), with one scale per output row, by linear name.
    int8: HashMap<String, Tensor>,
}
impl Group {
    pub fn get(&self, k: &str) -> Result<&Tensor> {
        self.tensors
            .get(k)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing LTX tensor {k}")))
    }
    pub fn linear(&self, key: &str, x: &Tensor) -> Result<Tensor> {
        let expanded;
        let w = match self.int8.get(key) {
            // An INT8 weight back to BF16 for this product only: codes - 128,
            // times the row's scale, then its bias column as the packed form.
            Some(scale) => {
                let w = self.get(&format!("{key}.weight"))?.to_dtype(DType::BF16)?.affine(1., -128.)?.broadcast_mul(scale)?;
                expanded = match self.tensors.get(&format!("{key}.bias")) {
                    Some(b) if self.biased.contains(key) => {
                        let out = w.dim(0)?;
                        Tensor::cat(&[&w, &b.to_dtype(DType::BF16)?.unsqueeze(1)?, &Tensor::zeros((out, 7), DType::BF16, w.device())?], 1)?
                    }
                    _ => w,
                };
                &expanded
            }
            None => self.get(&format!("{key}.weight"))?,
        };
        let input = x.dim(candle_core::D::Minus1)?;
        let rows = x.elem_count() / input;
        let flat = x.reshape((rows, input))?;
        let flat = if self.biased.contains(key) {
            Tensor::cat(
                &[
                    &flat,
                    &Tensor::ones((rows, 1), x.dtype(), x.device())?,
                    &Tensor::zeros((rows, 7), x.dtype(), x.device())?,
                ],
                1,
            )?
        } else {
            flat
        };
        let y = flat.matmul(&w.t()?)?;
        let mut shape = x.dims().to_vec();
        let last = shape
            .last_mut()
            .ok_or_else(|| candle_core::Error::Msg("linear requires a channel axis".into()))?;
        *last = w.dim(0)?;
        y.reshape(shape)
    }
}
