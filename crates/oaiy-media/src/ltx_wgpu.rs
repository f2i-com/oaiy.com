//! LTX 2.3's transformer on WebGPU: its weights on the GPU as they are stored where the chain has a kernel for them
//! (NVFP4 packed, the tensor cores decoding it as they multiply: Lightricks' `-nvfp4` release's 44 blocks of 48), else
//! f16 (BF16 rounded).
use crate::ltx::store::{untile_scales, Store};
use candle_core::{Device, Result};
use dsv41::safetensors::Dtype;
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A linear layer on the GPU: its weight NVFP4 (its words and its scale's vector) or f16, and its bias.
pub enum Weight {
    Nvfp4 { w: DeviceVec, scale: DeviceVec },
    F16(DeviceVec),
}

pub struct Linear {
    pub weight: Weight,
    pub bias: DeviceVec,
    pub n: usize,
    pub k: usize,
}

impl Linear {
    /// `name`'s weight and bias (`{name}.weight`, `{name}.bias`; zeros where none) from `store` onto `gpu`.
    pub fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<Self> {
        let key = format!("{name}.weight");
        let info = store.index.get(&key).cloned().ok_or_else(|| err(format!("missing LTX tensor {key}")))?;
        let global_name = format!("{name}.weight_scale_2");
        let (weight, n, k) = if info.dtype == Dtype::U8 && store.index.get(&global_name).is_some() {
            let (n, half) = match info.shape.as_slice() {
                [n, h] => (*n, *h),
                _ => candle_core::bail!("{key}: an NVFP4 weight of shape {:?}", info.shape),
            };
            let k = 2 * half;
            let packed = store.index.read(&key).map_err(err)?;
            let scale_name = format!("{name}.weight_scale");
            let sinfo = store.index.get(&scale_name).cloned().ok_or_else(|| err(format!("missing {scale_name}")))?;
            let (sr, sc) = match sinfo.shape.as_slice() {
                [r, c] => (*r, *c),
                _ => candle_core::bail!("{scale_name}: scales of shape {:?}", sinfo.shape),
            };
            // (the scales' tiles cover rows to 128's: a layer of fewer rows keeps its own)
            let tiled = store.index.read(&scale_name).map_err(err)?;
            let all = untile_scales(&tiled, sr, sc)?;
            if sc != k / 16 || sr < n {
                candle_core::bail!("{scale_name}: {sr} x {sc} scales for a weight of {n} x {k}");
            }
            let scales = &all[..n * sc];
            let g = store.index.read(&global_name).map_err(err)?;
            let global = f32::from_le_bytes(g.get(..4).and_then(|b| b.try_into().ok()).ok_or_else(|| err(format!("{global_name} is not one F32")))?);
            match gpu.nvfp4_weights(&packed, scales, global, n, k) {
                Some((w, scale)) => (Weight::Nvfp4 { w, scale }, n, k),
                None => {
                    let t = store.tensor_f32(&key, &Device::Cpu)?;
                    let v = gpu.vec_f16_rounded(&t.flatten_all()?.to_vec1::<f32>()?).ok_or_else(|| err(format!("{key}: past f16's range")))?;
                    (Weight::F16(v), n, k)
                }
            }
        } else {
            let t = store.tensor_f32(&key, &Device::Cpu)?;
            let (n, k) = t.dims2()?;
            let v = gpu.vec_f16_rounded(&t.flatten_all()?.to_vec1::<f32>()?).ok_or_else(|| err(format!("{key}: past f16's range")))?;
            (Weight::F16(v), n, k)
        };
        let bias_name = format!("{name}.bias");
        let bias: Vec<f32> = if store.index.get(&bias_name).is_some() { store.tensor_f32(&bias_name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()? } else { vec![0.0; n] };
        let b = gpu.vec(n);
        gpu.upload(&b, &bias);
        Ok(Self { weight, bias: b, n, k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows.
    pub fn forward(&self, rec: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match &self.weight {
            Weight::Nvfp4 { w, scale } => rec.matmul_nvfp4_rows(w, scale, &self.bias, self.n, self.k, x, y, rows),
            Weight::F16(w) => {
                rec.matmul_f16_rows(w, self.n, self.k, x, y, rows);
                rec.add_bias_rows(y, &self.bias, rows, self.n);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// A real NVFP4 layer of Lightricks' release (`OAIY_LTX_NVFP4`) on the tensor cores gives what its store's own
    /// decode does on the CPU (in f32), 300 rows.
    #[test]
    #[ignore = "needs LTX 2.3's NVFP4 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
    fn an_nvfp4_layer_on_the_gpu_is_the_stores() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = ggml_rs_wgpu::WgpuBackend::new(None).map_err(err)?;
        // (not a gate's logits: 32 rows, their scales a tile of 128's, which the store's own decode does not take)
        for name in ["model.diffusion_model.transformer_blocks.10.attn1.to_q", "model.diffusion_model.transformer_blocks.20.ff.net.2", "model.diffusion_model.transformer_blocks.30.audio_ff.net.0.proj"] {
            let l = Linear::load(&mut store, &gpu, name)?;
            assert!(matches!(l.weight, Weight::Nvfp4 { .. }), "{name} NVFP4");
            let rows = 300;
            let mut seed = 0x9e37_79b9u64;
            let x: Vec<f32> = (0..rows * l.k)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    half::f16::from_f32((seed % 2001) as f32 / 1000.0 - 1.0).to_f32()
                })
                .collect();
            let (xd, yd) = (gpu.vec(x.len()), gpu.vec(rows * l.n));
            gpu.upload(&xd, &x);
            let mut rec = gpu.begin();
            l.forward(rec.as_mut(), &xd, &yd, rows);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let w = store.tensor_f32(&format!("{name}.weight"), &Device::Cpu)?;
            let b = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?;
            let xt = candle_core::Tensor::from_vec(x, (rows, l.k), &Device::Cpu)?;
            let want = xt.matmul(&w.t()?)?.broadcast_add(&b)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
            let worst = got.iter().zip(&want).map(|(a, e)| (*a as f64 - *e as f64).abs()).fold(0.0, f64::max);
            eprintln!("{name} [{}, {}]: the worst error {worst:.3e} of an RMS {rms:.3e}", l.n, l.k);
            assert!(worst <= 1e-3 * rms.max(1e-6) * 10.0, "{name}: worst {worst} of an RMS {rms}");
        }
        Ok(())
    }
}
