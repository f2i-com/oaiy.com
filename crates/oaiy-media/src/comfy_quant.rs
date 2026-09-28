//! Direct Comfy Kitchen ConvRot checkpoint decoding, in memory only.
//! Storage semantics follow comfy_kitchen/backends/eager/w4a8_int8.py and
//! tensor/int8_utils.py (Comfy-Org/comfy-kitchen). Compute uses dense BF16;
//! this is compatibility loading, not a fused W4A8 activation kernel.
use candle_core::{DType, Device, Result, Tensor};
use dsv41::safetensors::{Dtype, StIndex};
use oaiy_engine::json::Json;

pub fn load(s: &StIndex, name: &str, dev: &Device, dtype: DType) -> Result<Option<Tensor>> {
    let Some(prefix) = name.strip_suffix(".weight") else {
        return Ok(None);
    };
    let metadata = format!("{prefix}.comfy_quant");
    if s.get(&metadata).is_none() {
        return Ok(None);
    }
    let config = Json::parse(&s.read(&metadata).map_err(candle_core::Error::wrap)?)
        .map_err(candle_core::Error::wrap)?;
    let format = config.get("format").and_then(Json::as_str).unwrap_or("");
    let info = s.info(name).map_err(candle_core::Error::wrap)?;
    if info.dtype != Dtype::I8 || info.shape.len() != 2 {
        candle_core::bail!("{name}: ConvRot expects a matrix of I8 storage");
    }
    let (rows, packed_cols) = (info.shape[0], info.shape[1]);
    let mut cols = packed_cols;
    let bytes = s.read(name).map_err(candle_core::Error::wrap)?;
    let mut data = match format {
        "int8_tensorwise" => {
            let scale = s
                .read_f32(&format!("{prefix}.weight_scale"))
                .map_err(candle_core::Error::wrap)?;
            if scale.len() != 1 && scale.len() != rows {
                candle_core::bail!("{name}: expected scalar or per-row INT8 scale");
            }
            let mut data = vec![0.; bytes.len()];
            parallel_rows(&mut data, cols, |start, chunk| {
                for (r, row) in chunk.chunks_mut(cols).enumerate() {
                    let row_id = start + r;
                    let scale = scale[if scale.len() == 1 { 0 } else { row_id }];
                    for (out, &b) in row
                        .iter_mut()
                        .zip(&bytes[row_id * cols..(row_id + 1) * cols])
                    {
                        *out = b as i8 as f32 * scale;
                    }
                }
            });
            data
        }
        "asym_w4a8_int8" => {
            let group = config.get("group_size").and_then(Json::as_i64).unwrap_or(0);
            if group != 16 {
                candle_core::bail!("{name}: supported W4A8 group_size is 16");
            }
            cols = cols
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::Msg("W4A8 shape overflow".into()))?;
            let rel_name = format!("{prefix}.weight_s_rel");
            let rel_info = s.info(&rel_name).map_err(candle_core::Error::wrap)?;
            if cols % 16 != 0
                || rel_info.shape != [rows, cols / 16]
                || rel_info.dtype != Dtype::F8E4M3
            {
                candle_core::bail!("{name}: invalid W4A8 FP8 group scales");
            }
            if s.get(&format!("{prefix}.weight_correction")).is_some() {
                candle_core::bail!("{name}: asymmetric W4A8 correction is not supported");
            }
            let rel = s.read(&rel_name).map_err(candle_core::Error::wrap)?;
            let channel = s
                .read_f32(&format!("{prefix}.weight_s_channel"))
                .map_err(candle_core::Error::wrap)?;
            let book_name = format!("{prefix}.weight_codebook");
            let book = if s.get(&book_name).is_some() {
                s.read_f32(&book_name).map_err(candle_core::Error::wrap)?
            } else {
                (0..16).map(|n| (n - 8) as f32).collect()
            };
            if channel.len() != rows || book.len() != 16 {
                candle_core::bail!("{name}: invalid W4A8 codebook/channel scale");
            }
            let mut data = vec![0.; bytes.len() * 2];
            parallel_rows(&mut data, cols, |start, chunk| {
                for (r, row) in chunk.chunks_mut(cols).enumerate() {
                    let row_id = start + r;
                    let channel = channel[row_id];
                    for (g, group) in row.chunks_mut(16).enumerate() {
                        let scale = dsv41::formats::fp8_e4m3_to_f32(rel[row_id * (cols / 16) + g]);
                        let packed = &bytes
                            [row_id * packed_cols + g * 8..row_id * packed_cols + (g + 1) * 8];
                        for (out, &b) in group.chunks_mut(2).zip(packed) {
                            // Low nibble is the even column; round to the INT8
                            // grid before channel scaling and inverse rotation.
                            out[0] = (book[(b & 15) as usize] * scale)
                                .round_ties_even()
                                .clamp(-127., 127.)
                                * channel;
                            out[1] = (book[(b >> 4) as usize] * scale)
                                .round_ties_even()
                                .clamp(-127., 127.)
                                * channel;
                        }
                    }
                }
            });
            data
        }
        _ => candle_core::bail!("{name}: unsupported comfy_quant format {format}"),
    };
    if rows == 0 || cols == 0 || data.iter().any(|v| !v.is_finite()) {
        candle_core::bail!("{name}: empty or nonfinite quantized tensor");
    }
    let rotation = match config.get("convrot") {
        Some(Json::Bool(true)) => config
            .get("convrot_groupsize")
            .and_then(Json::as_i64)
            .filter(|&n| n >= 4)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("{name}: missing or invalid ConvRot group size"))
            })?,
        None | Some(Json::Bool(false)) => 0,
        _ => candle_core::bail!("{name}: invalid convrot flag"),
    };
    if rotation != 0 {
        if rotation < 4
            || rotation > 4096
            || !(rotation as usize).is_power_of_two()
            || (rotation as usize).trailing_zeros() % 2 != 0
            || cols % rotation as usize != 0
        {
            candle_core::bail!("{name}: invalid ConvRot group size {rotation}");
        }
        let gs = rotation as usize;
        // GPU matmul amortizes inverse rotation across all rows. W4A8 rounds
        // the channel-scaled grid to compute dtype before rotating, like upstream.
        let t = Tensor::from_vec(std::mem::take(&mut data), (rows * cols / gs, gs), dev)?;
        let t = if format == "asym_w4a8_int8" {
            t.to_dtype(dtype)?.to_dtype(DType::F32)?
        } else {
            t
        };
        let h = Tensor::from_vec(hadamard(gs), (gs, gs), dev)?;
        return Ok(Some(t.matmul(&h)?.reshape((rows, cols))?.to_dtype(dtype)?));
    }
    Ok(Some(
        Tensor::from_vec(data, (rows, cols), dev)?.to_dtype(dtype)?,
    ))
}

/// The ConvRot group size of a Comfy `int8_tensorwise` weight, when its note
/// says it was rotated (the stored rows are then Hadamard-rotated in groups).
/// None for weights without a note, or not rotated.
pub fn rotation(s: &StIndex, name: &str) -> Result<Option<usize>> {
    let Some(prefix) = name.strip_suffix(".weight") else { return Ok(None) };
    let metadata = format!("{prefix}.comfy_quant");
    if s.get(&metadata).is_none() {
        return Ok(None);
    }
    let config = Json::parse(&s.read(&metadata).map_err(candle_core::Error::wrap)?).map_err(candle_core::Error::wrap)?;
    match config.get("format").and_then(Json::as_str) {
        Some("int8_tensorwise") => {}
        other => candle_core::bail!("{name}: unsupported comfy_quant format {other:?}"),
    }
    if !matches!(config.get("convrot"), Some(Json::Bool(true))) {
        return Ok(None);
    }
    let gs = config.get("convrot_groupsize").and_then(Json::as_i64).unwrap_or(0);
    let valid = gs >= 4 && gs <= 4096 && (gs as usize).is_power_of_two() && (gs as usize).trailing_zeros() % 2 == 0;
    if !valid {
        candle_core::bail!("{name}: invalid ConvRot group size {gs}");
    }
    Ok(Some(gs as usize))
}

/// Undo ConvRot on a decoded `[rows, cols]` weight: each group of `gs` columns
/// times the (symmetric, orthonormal) Hadamard matrix, in F32.
pub fn unrotate(t: &Tensor, gs: usize) -> Result<Tensor> {
    let (rows, cols) = t.dims2()?;
    if cols % gs != 0 {
        candle_core::bail!("ConvRot group size {gs} does not divide {cols} columns");
    }
    let h = Tensor::from_vec(hadamard(gs), (gs, gs), t.device())?;
    t.to_dtype(DType::F32)?.reshape((rows * cols / gs, gs))?.matmul(&h)?.reshape((rows, cols))
}

fn parallel_rows(data: &mut [f32], cols: usize, decode: impl Fn(usize, &mut [f32]) + Sync) {
    if cols == 0 || data.is_empty() {
        return;
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let rows_per_chunk = (data.len() / cols).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in data.chunks_mut(rows_per_chunk * cols).enumerate() {
            let decode = &decode;
            scope.spawn(move || decode(i * rows_per_chunk, chunk));
        }
    });
}

fn hadamard(size: usize) -> Vec<f32> {
    const H: [[f32; 4]; 4] = [
        [1., 1., 1., -1.],
        [1., 1., -1., 1.],
        [1., -1., 1., 1.],
        [-1., 1., 1., 1.],
    ];
    let mut out = vec![0.; size * size];
    for r in 0..size {
        for c in 0..size {
            let (mut a, mut b, mut value) = (r, c, 1. / (size as f32).sqrt());
            while a != 0 || b != 0 {
                value *= H[a % 4][b % 4];
                a /= 4;
                b /= 4;
            }
            out[r * size + c] = value;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_codes_scales_rounding_and_rotation_match_reference() {
        let root = std::env::temp_dir().join(format!("oaiy-convrot-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("weights.safetensors");
        let mut data = Vec::new();
        let mut header = Vec::new();
        let mut add = |name: &str, ty: &str, shape: &[i64], bytes: Vec<u8>| {
            let start = data.len();
            data.extend(bytes);
            header.push((
                name.to_owned(),
                Json::obj([
                    ("dtype", Json::str(ty)),
                    (
                        "shape",
                        Json::Arr(shape.iter().map(|&n| Json::Int(n)).collect()),
                    ),
                    (
                        "data_offsets",
                        Json::Arr(vec![Json::Int(start as i64), Json::Int(data.len() as i64)]),
                    ),
                ]),
            ));
        };
        let f32s = |xs: &[f32]| xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let meta =
            br#"{"format":"asym_w4a8_int8","group_size":16,"convrot":true,"convrot_groupsize":4}"#;
        add("a.comfy_quant", "U8", &[meta.len() as i64], meta.to_vec());
        add(
            "a.weight",
            "I8",
            &[1, 8],
            vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe],
        );
        add("a.weight_s_rel", "F8_E4M3", &[1, 1], vec![0x30]); // 0.5
        add("a.weight_s_channel", "F32", &[1], f32s(&[2.]));
        add(
            "a.weight_codebook",
            "F32",
            &[16],
            f32s(&(0..16).map(|n| (n - 8) as f32).collect::<Vec<_>>()),
        );
        let meta = br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":4}"#;
        add("b.comfy_quant", "U8", &[meta.len() as i64], meta.to_vec());
        add("b.weight", "I8", &[1, 4], vec![1, 2, 4, 8]);
        add("b.weight_scale", "F32", &[1, 1], f32s(&[1.]));
        add(
            "transformer_blocks.0.img_mlp.gate_up.weight",
            "F32",
            &[4, 2],
            f32s(&[1., 2., 3., 4., 5., 6., 7., 8.]),
        );
        let header = Json::Obj(header).to_json();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        std::fs::write(&file, bytes).unwrap();
        let mut weights = crate::weights::Weights::open(&file).unwrap();
        let a = weights
            .tensor("a.weight", &Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            a,
            vec![-9., -7., -5., -5., -5., -3., -1., -1., -1., 1., 3., 3., 3., 5., 7., 7.]
        );
        let b = weights
            .tensor("b.weight", &Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(b, vec![-0.5, 3.5, 5.5, 6.5]);
        let input = Tensor::new(&[[1f32, 1.]], &Device::Cpu).unwrap();
        for (name, expected) in [("gate_layer", vec![3., 7.]), ("proj", vec![11., 15.])] {
            let linear = weights
                .linear(
                    &format!("transformer_blocks.0.img_mlp.{name}"),
                    &Device::Cpu,
                    DType::F32,
                    &mut crate::lora::Loras::default(),
                )
                .unwrap();
            assert_eq!(
                linear
                    .forward(&input)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                expected
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn regular_rotation_is_symmetric_and_self_inverse() {
        let h = Tensor::from_vec(hadamard(256), (256, 256), &Device::Cpu).unwrap();
        let identity = h.matmul(&h).unwrap().to_vec2::<f32>().unwrap();
        for (i, row) in identity.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!((v - if i == j { 1. } else { 0. }).abs() < 1e-6);
            }
        }
        assert_eq!(
            hadamard(4),
            vec![
                0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5
            ]
        );
    }
}
