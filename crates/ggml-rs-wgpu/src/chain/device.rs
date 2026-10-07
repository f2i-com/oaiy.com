//! The backend as a chain's device: its vectors, what it holds, and a recording's start.
use super::*;

impl DeviceChain for WgpuBackend {
    fn pieces_in_flight_at_most(&self, pieces: usize) {
        WgpuBackend::pieces_in_flight_at_most(self, pieces);
    }

    fn has_room(&self, bytes: u64) -> bool {
        // (the adapter's own count of its memory in use against its budget; a gigabyte left after)
        self.memory_budget().is_some_and(|(budget, used)| used.saturating_add(bytes).saturating_add(1 << 30) <= budget)
    }

    fn attention_halves(&self, n_h: usize, n_kv: usize, head_dim: usize) -> bool {
        // (OAIY_ATTENTION_F32: a step's attention over the cache as it is)
        static F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let g = if n_kv > 0 && n_h % n_kv == 0 { n_h / n_kv } else { 0 };
        !*F32.get_or_init(|| std::env::var_os("OAIY_ATTENTION_F32").is_some() || std::env::var_os("OAIY_ATTENTION_PART4").is_some())
            && self.gpu.device.features().contains(wgpu::Features::SHADER_F16)
            && attention_group_for(g, head_dim, self.gpu.limits.max_compute_workgroup_storage_size)
    }

    fn vec(&self, len: usize) -> DeviceVec {
        DeviceVec { len, inner: Arc::new(vec_buffer(&self.gpu, len)) }
    }

    fn vec_f16(&self, values: &[f32]) -> Option<DeviceVec> {
        use rayon::prelude::*;
        // checked and packed on every core (a model's hyper-connections are some 700 million values)
        let exact = values.len() % 2 == 0 && values.par_chunks(1 << 16).all(|c| c.iter().all(|&v| half::f16::from_f32(v).to_f32().to_bits() == v.to_bits()));
        if !exact {
            return None;
        }
        let words: Vec<f32> = values.par_chunks_exact(2).map(|p| f32::from_bits(half::f16::from_f32(p[0]).to_bits() as u32 | (half::f16::from_f32(p[1]).to_bits() as u32) << 16)).collect();
        let v = self.vec(words.len());
        DeviceChain::upload(self, &v, &words);
        Some(v)
    }

    fn conv3d_weights(&self, w: &[f32], cout: usize, cin: usize) -> Option<DeviceVec> {
        // (the tensor cores' kernel's layout, the f32 one's too: CONV_F32_TILED)
        if w.len() != cout * cin * 27 {
            return None;
        }
        let cp = cin.div_ceil(32) * 32;
        let mut packed = vec![0f32; cout * 27 * cp];
        for co in 0..cout {
            for c in 0..cin {
                for tap in 0..27 {
                    packed[(co * 27 + tap) * cp + c] = w[(co * cin + c) * 27 + tap];
                }
            }
        }
        self.vec_f16_rounded(&packed)
    }

    fn conv1d_weights(&self, w: &[f32], cout: usize, cin: usize, k: usize) -> Option<DeviceVec> {
        if w.len() != cout * cin * k {
            return None;
        }
        let cp = cin.div_ceil(32) * 32;
        let mut packed = vec![0f32; cout * k * cp];
        for co in 0..cout {
            for c in 0..cin {
                for tap in 0..k {
                    packed[(co * k + tap) * cp + c] = w[(co * cin + c) * k + tap];
                }
            }
        }
        self.vec_f16_rounded(&packed)
    }

    fn conv_weights(&self, w: &[f32], cout: usize, cin: usize, k: usize) -> Option<DeviceVec> {
        let taps = k * k;
        if !matches!(k, 1 | 3 | 7) || w.len() != cout * cin * taps {
            return None;
        }
        let cp = cin.div_ceil(32) * 32;
        let mut packed = vec![0f32; cout * taps * cp];
        for co in 0..cout {
            for c in 0..cin {
                for tap in 0..taps {
                    packed[(co * taps + tap) * cp + c] = w[(co * cin + c) * taps + tap];
                }
            }
        }
        self.vec_f16_rounded(&packed)
    }

    fn attention_rows_full_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
        // the tensor cores' kernel writes its rows padded to 32 and keeps nothing else there
        if rows >= 16 && matches!(head_dim, 64 | 128 | 256) && self.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            rows.div_ceil(32) * 32 * n_h * head_dim
        } else {
            self.attention_rows_out_len(rows, n_h, head_dim, kv_len)
        }
    }

    fn nvfp4_weights(&self, packed: &[u8], scales: &[u8], global: f32, rows: usize, cols: usize) -> Option<(DeviceVec, DeviceVec)> {
        if cols % 64 != 0 || packed.len() != rows * cols / 2 || scales.len() != rows * cols / 16 || !self.gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return None;
        }
        // a row's nibbles' words, then its scales' (four a word)
        let (nb, sb) = (cols / 2, cols / 16);
        let mut words: Vec<f32> = Vec::with_capacity(rows * (nb + sb) / 4);
        for r in 0..rows {
            let row = packed[r * nb..(r + 1) * nb].iter().chain(&scales[r * sb..(r + 1) * sb]);
            let bytes: Vec<u8> = row.copied().collect();
            words.extend(bytes.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))));
        }
        let w = self.vec(words.len());
        DeviceChain::upload(self, &w, &words);
        let s = self.vec(4);
        DeviceChain::upload(self, &s, &[1.0, global, 0.0, 0.0]);
        Some((w, s))
    }

    fn vec_f16_rounded(&self, values: &[f32]) -> Option<DeviceVec> {
        use rayon::prelude::*;
        if values.len() % 2 != 0 || values.par_chunks(1 << 16).any(|c| c.iter().any(|v| !v.is_finite() || v.abs() > 65504.0)) {
            return None;
        }
        let words: Vec<f32> = values.par_chunks_exact(2).map(|p| f32::from_bits(half::f16::from_f32(p[0]).to_bits() as u32 | (half::f16::from_f32(p[1]).to_bits() as u32) << 16)).collect();
        let v = self.vec(words.len());
        DeviceChain::upload(self, &v, &words);
        Some(v)
    }

    fn zero(&self, v: &DeviceVec) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(buffer(v), 0, None);
        self.gpu.queue().submit([enc.finish()]);
    }

    fn alias(&self, v: &DeviceVec, shape: Vec<usize>) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), v.len, "chain: an alias of {} values as {shape:?}", v.len);
        let storage = Aliased { v: v.clone(), gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial) };
        Tensor::from_device(Box::new(storage), shape)
    }

    fn aliased(&self, t: &Tensor) -> Option<DeviceVec> {
        let a = t.device_storage()?.as_any().downcast_ref::<Aliased>()?;
        Arc::ptr_eq(&a.gpu, &self.gpu).then(|| a.v.clone())
    }

    fn upload_at(&self, v: &DeviceVec, offset: usize, data: &[f32]) {
        assert!(offset + data.len() <= v.len, "chain: {} values at {offset} into a vector of {}", data.len(), v.len);
        if !data.is_empty() {
            // (the values' own bytes: every target wgpu runs on is little-endian)
            self.gpu.write(buffer(v), (offset * 4) as u64, bytemuck::cast_slice(data));
            // a large write's staging (device memory, with Resizable BAR) let go now: a model's weights uploaded in
            // turn held it all until the next submit (Qwen Image's 14 GB took 31 of a 32 GB card)
            if data.len() >= 16 << 20 {
                self.gpu.queue().submit([]);
                self.gpu.wait(None);
            }
        }
    }

    fn resize(&self, v: &DeviceVec, len: usize) -> DeviceVec {
        let grown = self.vec(len);
        let keep = v.len.min(len);
        if keep > 0 {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buffer(v), 0, buffer(&grown), 0, (keep * 4) as u64);
            self.gpu.queue().submit([enc.finish()]);
        }
        grown
    }

    fn attention_out_len(&self, n_h: usize, head_dim: usize, cap: usize) -> usize {
        n_h * head_dim + n_h * cap.div_ceil(SPLIT).max(1) * (head_dim + 2)
    }

    fn attention_rows_out_len(&self, rows: usize, n_h: usize, head_dim: usize, kv_len: usize) -> usize {
        // (one pass, tiled, or on the tensor cores, their rows padded to 32: the output alone; else its runs' parts)
        let padded = rows.div_ceil(32) * 32 * n_h * head_dim;
        if attention_tiled_for(rows, head_dim) {
            padded
        } else {
            padded.max(attention_runs_out_len(rows, n_h, head_dim, kv_len))
        }
    }

    fn qsa_attention_out_len(&self, rows: usize, n_h: usize, head_dim: usize, keep: usize, ratio: usize) -> usize {
        // the selection sorts 4096 blocks' keys and indices in a workgroup's memory (32 KB, past WebGPU's default 16)
        if self.gpu.limits.max_compute_workgroup_storage_size < 32768 {
            return 0;
        }
        rows * n_h * head_dim + rows * n_h * (keep * ratio + ratio).div_ceil(256) * (head_dim + 2)
    }

    fn holds_exl3(&self, w: &dyn ggml_rs::exl3::PackedLinear) -> bool {
        // (a projection with a low-rank update beside it, a LoRA adapter's: held when its three matrices are)
        if let Some(l) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::LowRank>()) {
            return self.holds_exl3(&*l.base) && self.holds_exl3(&*l.a) && self.holds_exl3(&*l.b);
        }
        // (a GGUF's matrix where a packed projection is asked for: held as a quantized weight is)
        if let Some(q) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::QuantLinear>()) {
            return self.holds(&q.w);
        }
        if let Some(f) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::HalfLinear>()) {
            return f.is_on(&self.gpu);
        }
        w.as_any()
            .and_then(|a| a.downcast_ref::<crate::exl3::Exl3Gpu>())
            .is_some_and(|g| g.is_on(&self.gpu) && g.single_chunk().is_some())
    }

    fn holds_experts(&self, e: &dyn ggml_rs::exl3::Experts) -> bool {
        // (EXL3's groups, or a GGUF's quant blocks)
        e.as_any().is_some_and(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>().is_some_and(|g| g.is_on(&self.gpu)) || a.downcast_ref::<crate::quant_moe::QuantMoe>().is_some_and(|g| g.is_on(&self.gpu)))
    }

    fn holds(&self, w: &QuantizedTensor) -> bool {
        w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).is_some_and(|q| Arc::ptr_eq(&q.gpu, &self.gpu))
    }

    fn copy_weight(&self, w: &QuantizedTensor) -> Option<QuantizedTensor> {
        WgpuBackend::copy_weight(self, w)
    }

    fn begin(&self) -> Box<dyn ChainRecorder + '_> {
        Box::new(Recorder { backend: self, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 })
    }
}
