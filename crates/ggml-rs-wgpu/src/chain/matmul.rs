//! The recorder's matmuls of a prompt's rows: a quantized weight's (f32, on the tensor cores, from int8 activations) and an f16 matrix's.
use super::*;

impl Recorder<'_> {
    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] through the f32 kernels alone: the decode kernel for one row, the
    /// one-row kernel for a few, the tiled one for a prompt.
    pub(crate) fn matmul_rows_f32(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        assert!(m > 0 && x.len >= m * k && y.len >= m * n, "chain: matmul [{n}, {k}] of {m} rows from {} into {}", x.len, y.len);
        let pipeline = self.gpu().pipeline(q.dtype, m).expect("uploaded weights have a pipeline");
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0];
            let groups = crate::quant::shaders::grid(q.dtype, m, *rows);
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, groups);
        }
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for one to eight rows against IQ4_XS weights
    /// ([`crate::quant::shaders::iq4_xs_few`]). False for another type, more rows, or a width that is not whole blocks.
    pub(crate) fn matmul_rows_iq4_xs(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::IQ4_XS || m == 0 || m > crate::quant::shaders::IQ4_FEW_MAX || k % 256 != 0 || q.row_bytes != k / 256 * 136 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an IQ4_XS matmul [{n}, {k}] of {m} rows");
        const NAMES: [&str; crate::quant::shaders::IQ4_FEW_MAX] =
            ["chain-iq4xs-few-1", "chain-iq4xs-few-2", "chain-iq4xs-few-3", "chain-iq4xs-few-4", "chain-iq4xs-few-5", "chain-iq4xs-few-6", "chain-iq4xs-few-7", "chain-iq4xs-few-8"];
        let pipeline = self.gpu().named_pipeline(NAMES[m - 1], || crate::quant::shaders::iq4_xs_few(m));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0];
            let groups = rows.div_ceil(8);
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, (groups.min(65535), groups.div_ceil(65535), 1));
        }
        true
    }

    /// [`Self::matmul_rows_iq4_xs`] from `x`'s rows as int8 ([`crate::quant::shaders::iq4_xs_few_q8`]; quantized once, as for
    /// [`Self::matmul_rows_q8`]): a check of drafts' rows. False as that one is.
    pub(crate) fn matmul_rows_iq4_xs_q8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::IQ4_XS || m == 0 || m > crate::quant::shaders::IQ4_FEW_MAX || k % 256 != 0 || q.row_bytes != k / 256 * 136 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an IQ4_XS int8 matmul [{n}, {k}] of {m} rows");
        let (xq, xs_at) = self.x_q8(x, m, k);
        const NAMES: [&str; crate::quant::shaders::IQ4_FEW_MAX] =
            ["chain-iq4xs-q8-1", "chain-iq4xs-q8-2", "chain-iq4xs-q8-3", "chain-iq4xs-q8-4", "chain-iq4xs-q8-5", "chain-iq4xs-q8-6", "chain-iq4xs-q8-7", "chain-iq4xs-q8-8"];
        let pipeline = self.gpu().named_pipeline(NAMES[m - 1], || crate::quant::shaders::iq4_xs_few_q8(m));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = rows.div_ceil(8);
            self.dispatch_kept(&pipeline, chunk, buffer(&xq), buffer(y), &words, (groups.min(65535), groups.div_ceil(65535), 1));
        }
        true
    }

    /// A SwiGLU's gate and up matmul with its SwiGLU, one dispatch ([`matvec_f16_swiglu_lanes`]): `w` the f16 matrix
    /// `[2 ff, k]` (the gate's rows, then the up's), `out[r, j] = silu(gate) * up` for `rows` rows of `x`. False
    /// (nothing recorded) where the rows are more than a check's, or the width is not the lanes' kernel's, or
    /// OAIY_MOE_UNFUSED is set: the caller's matmul and SwiGLU then.
    pub(crate) fn matmul_f16_swiglu_rows(&mut self, w: &DeviceVec, ff: usize, k: usize, x: &DeviceVec, out: &DeviceVec, rows: usize) -> bool {
        let lanes = f16_lanes(k);
        if !(1..=8).contains(&rows) || k % 4 != 0 || lanes > 128 || !moe_fused() {
            return false;
        }
        assert!(w.len * 2 >= 2 * ff * k && x.len >= rows * k && out.len >= rows * ff && 2 * ff <= 65535, "chain: a SwiGLU's matmul [{}, {k}] of {rows} rows", 2 * ff);
        let (name, source) = super::ops::fused_kernel("chain-matvec-f16-swiglu", rows, lanes, 0, || matvec_f16_swiglu_lanes(rows, lanes));
        let pipeline = self.gpu().named_pipeline(name, || source.to_string());
        let groups = (ff as u32).div_ceil((256 / lanes / 2) as u32);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[ff as u32, k as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.weigh(2.0 * (rows * 2 * ff) as f64 * k as f64);
        true
    }

    /// A MoE layer's shared expert's down matmul with the experts' weighted sum into the streams, one dispatch
    /// ([`shared_down_wsum_lanes`]): `w` the f16 matrix `[h, ff]`, `x` the shared expert's SwiGLU (`[rows, ff]`), `d`
    /// the routed experts' outputs, `wts` their weights and the shared one's, `xs` and `post` the streams and their
    /// write weights. False (nothing recorded) as [`Self::matmul_f16_swiglu_rows`]: the caller's matmul and sum then.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shared_down_wsum(&mut self, w: &DeviceVec, h: usize, ff: usize, x: &DeviceVec, d: &DeviceVec, wts: &DeviceVec, post: &DeviceVec, xs: &DeviceVec, rows: usize, top_k: usize, streams: usize) -> bool {
        if !(1..=8).contains(&rows) || ff % 4 != 0 || !moe_fused() {
            return false;
        }
        assert!(
            w.len * 2 >= h * ff && x.len >= rows * ff && d.len >= rows * top_k * h && wts.len >= rows * (top_k + 1) && post.len >= rows * streams && xs.len >= rows * streams * h && h <= 65535,
            "chain: a shared expert's down matmul [{h}, {ff}] and {rows} rows' sums"
        );
        let lanes = f16_lanes(ff);
        let (name, source) = super::ops::fused_kernel("chain-shared-down-wsum", rows, lanes, 0, || shared_down_wsum_lanes(rows, lanes));
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let groups = (h as u32).div_ceil((256 / lanes) as u32);
        self.dispatch_wide(name, source, [buffer(w), buffer(x), buffer(d), buffer(wts), buffer(post), &dd, buffer(xs), &drw], &[h as u32, ff as u32, top_k as u32, streams as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.weigh(2.0 * (rows * h) as f64 * ff as f64);
        true
    }

    /// `x`'s `m` rows of `k` as int8 ([`crate::quant::shaders::QUANT_Q8`]), and where their scales start: made once for every
    /// int8 matmul that reads them until something writes `x`.
    pub(super) fn x_q8(&mut self, x: &DeviceVec, m: usize, k: usize) -> (DeviceVec, usize) {
        let (len, xs_at) = crate::quant::shaders::q8_len(m, k);
        let xb = buffer(x).clone();
        if let Some((.., xq)) = self.q8.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            return (xq.clone(), xs_at);
        }
        // (a kept recording's own buffer for them, the same every run, so the recording is replayed; else scratch)
        let xq = if self.keeps() { DeviceVec { len, inner: std::sync::Arc::new(self.gpu().q8_kept(&xb, m, k, (len * 4) as u64)) } } else { self.scratch(len) };
        let blocks = (m * k / 32) as u32;
        // (a device whose subgroups are 32 lanes: 8 lanes a block, [`crate::quant::shaders::QUANT_Q8_LANES`])
        let (quant, groups) = if self.gpu().subgroups32() {
            (self.gpu().named_pipeline("chain-q8-quantize-lanes", || crate::quant::shaders::QUANT_Q8_LANES.to_string()), (blocks * 8).div_ceil(256))
        } else {
            (self.gpu().named_pipeline("chain-q8-quantize", || crate::quant::shaders::QUANT_Q8.to_string()), blocks.div_ceil(256))
        };
        let d = self.gpu().dummy().clone();
        self.dispatch_kept(&quant, &d, buffer(x), buffer(&xq), &[k as u32, m as u32, xs_at as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.q8.push((xb, m, k, xq.clone()));
        (xq, xs_at)
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for a prompt's rows on the tensor cores
    /// ([`crate::quant::shaders::coop_tiled`]: f16 weights and tokens into f32 sums). False where the device has no cooperative
    /// matrices or the type no such kernel.
    pub(crate) fn matmul_rows_coop(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        self.matmul_rows_coop_split(w, x, y, m, None)
    }

    /// [`Self::matmul_rows_coop`], split along k as given (else as [`crate::quant::shaders::coop_splits`] chooses).
    pub(crate) fn matmul_rows_coop_split(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        use ggml_quants::GgmlType as T;
        // (each type's kernel in 16 x 16 fragments, and in Metal's 8 x 8: `Gpu::coop_tile`)
        let names = match q.dtype {
            T::Q3_K => ("chain-coop-Q3_K", "chain-coop8-Q3_K"),
            T::Q4_K => ("chain-coop-Q4_K", "chain-coop8-Q4_K"),
            T::Q5_K => ("chain-coop-Q5_K", "chain-coop8-Q5_K"),
            T::Q6_K => ("chain-coop-Q6_K", "chain-coop8-Q6_K"),
            T::Q8_0 => ("chain-coop-Q8_0", "chain-coop8-Q8_0"),
            T::Q2_0 => ("chain-coop-Q2_0", "chain-coop8-Q2_0"),
            T::Q4_0 => ("chain-coop-Q4_0", "chain-coop8-Q4_0"),
            T::Q5_0 => ("chain-coop-Q5_0", "chain-coop8-Q5_0"),
            T::IQ4_NL => ("chain-coop-IQ4_NL", "chain-coop8-IQ4_NL"),
            T::IQ4_XS => ("chain-coop-IQ4_XS", "chain-coop8-IQ4_XS"),
            _ => return false,
        };
        let eight = match self.gpu().coop_tile() {
            16 => false,
            8 => true,
            _ => return false,
        };
        if k % 256 != 0 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: a tensor-core matmul [{n}, {k}] of {m} rows");
        let x16 = self.x16_tiled(x, m, k);
        let tile = crate::quant::shaders::COOP_TILE;
        let dtype = q.dtype;
        let pipeline = if eight {
            self.gpu().named_pipeline(names.1, || crate::quant::shaders::coop8_tiled(dtype).expect("a K-quant's kernel in 8 x 8 fragments"))
        } else if let Some(name) = split_name(dtype).filter(|_| coop_split_for(self.backend, dtype)) {
            self.gpu().named_pipeline(name, || crate::quant::shaders::coop_tiled_split(dtype).expect("a K-quant's tensor-core kernel, its work apart"))
        } else {
            self.gpu().named_pipeline(names.0, || crate::quant::shaders::coop_tiled(dtype).expect("a K-quant's tensor-core kernel"))
        };
        let tiles = q.chunks.iter().map(|(_, _, rows)| rows.div_ceil(tile)).max().unwrap_or(1) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, m, n, y, split);
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, splits, 0];
            self.dispatch_kept(&pipeline, chunk, buffer(&x16), &out, &words, (rows.div_ceil(tile), (m as u32).div_ceil(tile), splits));
        }
        self.coop_sum(parts, m, n, y, splits);
        true
    }

    /// The tokens' rows `x` (`m` of `k`) as [`crate::quant::shaders::X_F16_TILED`] gives them (f16, padded to the tile and to
    /// a step of 32): once for every tensor-core matmul that reads them until something writes `x`.
    pub(super) fn x16_tiled(&mut self, x: &DeviceVec, m: usize, k: usize) -> DeviceVec {
        let tile = crate::quant::shaders::COOP_TILE;
        let padded = (m as u32).div_ceil(tile) as usize * tile as usize;
        let xb = buffer(x).clone();
        if let Some((.., v)) = self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            return v.clone();
        }
        let words = crate::quant::shaders::x_f16_tiled_words(m, k);
        let v = self.scratch(words);
        let conv = self.gpu().named_pipeline("chain-x-f16-tiled", || crate::quant::shaders::X_F16_TILED.to_string());
        let groups = (words as u32).div_ceil(256);
        let d = self.gpu().dummy().clone();
        self.dispatch_kept(&conv, &d, buffer(x), buffer(&v), &[k as u32, m as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.x16.push((xb, m, k, v.clone()));
        v
    }

    /// A tensor-core matmul's splits along k (as given, else as [`crate::quant::shaders::coop_splits`] chooses for `tiles`
    /// workgroups), where its sums go (`y`, or a part of scratch a split, added into `y` after), and the parts.
    pub(super) fn coop_parts(&mut self, tiles: u32, k: usize, m: usize, n: usize, y: &DeviceVec, split: Option<u32>) -> (u32, wgpu::Buffer, Option<DeviceVec>) {
        // a matmul of too few tiles to fill the GPU's last wave split along k: each split's sums into a part of
        // scratch, then the parts added into y
        let units = self.gpu().coop_units();
        let steps = k.div_ceil(32) as u32;
        let splits = split.unwrap_or_else(|| crate::quant::shaders::coop_splits(tiles, units, steps));
        // (one buffer of parts a recording, grown as it needs: its matmuls run in turn)
        let parts = if splits > 1 {
            let len = splits as usize * m * n;
            let v = match self.parts.take() {
                Some(v) if v.len >= len => v,
                _ => self.scratch(len),
            };
            self.parts = Some(v.clone());
            Some(v)
        } else {
            None
        };
        let out = parts.as_ref().map_or(buffer(y), buffer).clone();
        (splits, out, parts)
    }

    /// A split tensor-core matmul's parts added into `y`.
    pub(super) fn coop_sum(&mut self, parts: Option<DeviceVec>, m: usize, n: usize, y: &DeviceVec, splits: u32) {
        if let Some(parts) = parts {
            let sum = self.named("chain-coop-sum", COOP_SUM);
            let groups = ((m * n) as u32).div_ceil(256);
            let d = self.gpu().dummy().clone();
            self.dispatch_kept(&sum, &d, buffer(&parts), buffer(y), &[(m * n) as u32, splits], (groups.min(65535), groups.div_ceil(65535), 1));
        }
    }

    /// `y[r] = W x[r]` for a prompt's rows of f16 weights (`[n, k]` two to a word) through the f32 tiled kernel (the
    /// weights read as f32), split along k where its tiles are few.
    pub(crate) fn matmul_f16_tiled(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.matmul_half_tiled(w, n, k, x, y, rows, false)
    }

    /// [`Self::matmul_f16_tiled`], the weights f16 or (`bf16`) BF16 as a checkpoint's bytes lie, widened by a shift.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_half_tiled(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize, bf16: bool) {
        let tiles = n.div_ceil(64) * rows.div_ceil(64);
        let want = 1024usize.div_ceil(tiles).min(k / 256).max(1);
        let kc = k.div_ceil(want).div_ceil(16) * 16;
        let splits = k.div_ceil(kc);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let grid = (n.div_ceil(64) as u32, rows.div_ceil(64) as u32, splits as u32);
        let words = [n as u32, k as u32, rows as u32, kc as u32];
        let (name, value) = if bf16 {
            ("chain-matmul-bf16-tiled", "fn wv(e: u32) -> f32 {\n    let u = w[e / 2u];\n    return bitcast<f32>(select(u << 16u, u & 0xffff0000u, (e & 1u) == 1u));\n}")
        } else {
            ("chain-matmul-f16-tiled", "fn wv(e: u32) -> f32 {\n    let pr = unpack2x16float(w[e / 2u]);\n    return select(pr.x, pr.y, (e & 1u) == 1u);\n}")
        };
        let tiled = MATMUL_F32_TILED.replace("var<storage, read> w: array<f32>;", &format!("var<storage, read> w: array<u32>;\n{value}")).replace("u = w[(o0 + rr) * k + gk];", "u = wv((o0 + rr) * k + gk);");
        if splits == 1 {
            self.dispatch_wide(name, &tiled, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(y), &drw], &words, grid);
        } else {
            let part = self.scratch(splits * rows * n);
            self.dispatch_wide(name, &tiled, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(&part), &drw], &words, grid);
            let len = (rows * n) as u32;
            self.dispatch_wide("chain-sum-splits", SUM_SPLITS, [buffer(&part), &d, &d, &d, &d, &d, buffer(y), &drw], &[len, splits as u32], (len.div_ceil(256).min(65535), len.div_ceil(256 * 65535), 1));
            // (its parts read: spare for the next split's, where each had its own to the recording's end, a sound
            // step's 360 of 18 MB without tensor cores)
            self.spare.push((buffer(&part).size(), buffer(&part).clone()));
        }
    }

    /// `y[r] = W x[r]` for a prompt's rows of f16 weights (`[n, k]` two to a word, `k` of 4) on the tensor cores
    /// ([`crate::quant::shaders::coop_tiled_f16`]), split along k as given (else as chosen). False where the device has no
    /// cooperative matrices.
    pub(crate) fn matmul_f16_coop(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) -> bool {
        if !self.gpu().coop16() || k % 4 != 0 || m.div_ceil(crate::quant::shaders::COOP_TILE as usize) > 65535 {
            return false;
        }
        let x16 = self.x16_tiled(x, m, k);
        self.matmul_f16_coop_x16(w, n, k, &x16, y, m, split);
        true
    }

    /// [`Self::matmul_f16_coop`] of `x16`, `m` rows already as [`crate::quant::shaders::X_F16_TILED`] gives them.
    #[allow(clippy::too_many_arguments)]
    fn matmul_f16_coop_x16(&mut self, w: &DeviceVec, n: usize, k: usize, x16: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) {
        let tile = crate::quant::shaders::COOP_TILE;
        let pipeline = if split_f16_on(self.gpu()) {
            self.gpu().named_pipeline("chain-coop-split-f16", || crate::quant::shaders::coop_split_f16(0))
        } else {
            self.gpu().named_pipeline("chain-coop-f16", crate::quant::shaders::coop_tiled_f16)
        };
        let tiles = (n as u32).div_ceil(tile) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, m, n, y, split);
        let words = [k as u32, n as u32, m as u32, 0, n as u32, 0, splits, 0];
        self.dispatch_kept(&pipeline, buffer(w), buffer(x16), &out, &words, ((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), splits));
        self.coop_sum(parts, m, n, y, splits);
    }

    /// [`ChainRecorder::matmul_f16_rows_ranged`] on the tensor cores: `x`'s largest found on the device, `x` as tiled f16
    /// scaled by the power of two that keeps it within 16,384 (the copy and its scale kept for every matmul that reads
    /// `x` until something writes it: a layer's q, k and v from one norm), [`Self::matmul_f16_coop`]'s kernel, its sums
    /// scaled back. False where the device has no cooperative matrices.
    pub(crate) fn matmul_f16_scaled(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        if !self.gpu().coop16() || k % 4 != 0 || m.div_ceil(crate::quant::shaders::COOP_TILE as usize) > 65535 {
            return false;
        }
        // (kept with the plain copies, a width of its own: k with the top bit, and its scale beside it with the next)
        let (key, scale_key) = (k | 1 << 31, k | 3 << 30);
        let xb = buffer(x).clone();
        let kept = |rec: &Self, key: usize| rec.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == key).map(|(.., v)| v.clone());
        let (x16, range) = match (kept(self, key), kept(self, scale_key)) {
            (Some(x16), Some(range)) => (x16, range),
            _ => {
                let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
                let range = self.scratch(4);
                let len = (m * k) as u32;
                self.dispatch_wide("chain-f16-range-clear", F16_RANGE_CLEAR, [&d, &d, &d, &d, &d, &d, buffer(&range), &drw], &[0], (1, 1, 1));
                self.dispatch_wide("chain-f16-range-max", F16_RANGE_MAX, [buffer(x), &d, &d, &d, &d, &d, buffer(&range), &drw], &[len], grid(len.div_ceil(256)));
                self.dispatch_wide("chain-f16-range-set", F16_RANGE_SET, [&d, &d, &d, &d, &d, &d, buffer(&range), &drw], &[0], (1, 1, 1));
                let tile = crate::quant::shaders::COOP_TILE;
                let padded = (m as u32).div_ceil(tile) * tile;
                let words = crate::quant::shaders::x_f16_tiled_words(m, k);
                let x16 = self.scratch(words);
                let conv = self.gpu().named_pipeline("chain-x-f16-tiled-scaled", crate::quant::shaders::x_f16_tiled_scaled);
                let groups = (words as u32).div_ceil(256);
                self.dispatch_kept(&conv, buffer(&range), buffer(x), buffer(&x16), &[k as u32, m as u32, padded], (groups.min(65535), groups.div_ceil(65535), 1));
                self.x16.push((xb.clone(), m, key, x16.clone()));
                self.x16.push((xb, m, scale_key, range.clone()));
                (x16, range)
            }
        };
        self.matmul_f16_coop_x16(w, n, k, &x16, y, m, None);
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let len = (m * n) as u32;
        self.dispatch_wide("chain-unscale-rows", UNSCALE_ROWS, [buffer(&range), &d, &d, &d, &d, &d, buffer(y), &drw], &[len], grid(len.div_ceil(256)));
        true
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for a prompt's rows, from `x`'s rows as int8 (quantized once,
    /// as for [`Self::matmul_rows_q8`]) through [`crate::quant::shaders::tiled_q8`]. False for a type without that kernel.
    pub(crate) fn matmul_rows_tq8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::Q3_K || k % 256 != 0 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an int8 tiled matmul [{n}, {k}] of {m} rows");
        let (xq, xs_at) = self.x_q8(x, m, k);
        let pipeline = self.gpu().named_pipeline("chain-tq8-Q3_K", || crate::quant::shaders::tiled_q8(ggml_quants::GgmlType::Q3_K).expect("Q3_K's int8 tiled kernel"));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = (rows.div_ceil(crate::quant::shaders::TQ8_ROWS), (m as u32).div_ceil(crate::quant::shaders::TQ8_TOKENS), 1);
            self.dispatch_kept(&pipeline, chunk, buffer(&xq), buffer(y), &words, groups);
        }
        true
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for several rows, from `x`'s rows as int8 (quantized once,
    /// [`crate::quant::shaders::QUANT_Q8`], for every matmul that reads them until something writes `x`): the K-quants' int8
    /// kernels, a check of drafts' rows in about 1.3 of a step's time where the f32 kernels take 1.7. False for a type
    /// without one (nothing recorded).
    pub(super) fn matmul_rows_q8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        let block = if matches!(q.dtype, ggml_quants::GgmlType::Q4_0 | ggml_quants::GgmlType::Q8_0) { 32 } else { 256 };
        if !matches!(q.dtype, ggml_quants::GgmlType::Q3_K | ggml_quants::GgmlType::Q4_K | ggml_quants::GgmlType::Q5_K | ggml_quants::GgmlType::Q6_K | ggml_quants::GgmlType::Q4_0 | ggml_quants::GgmlType::Q8_0) || k % block != 0 {
            return false;
        }
        // (Q8_0 not where a recording's rows are alike: its rows by f32 there, as before. A few rows' int8 kernel for
        // it too: Music 3's language model steps its guided and unguided rows together, and by f32 its [12288, 4096]
        // matrices took 9.5 ms each on an NVIDIA A2, a frame 1.4 s)
        if q.dtype == ggml_quants::GgmlType::Q8_0 && self.alike {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an int8 matmul [{n}, {k}] of {m} rows");
        let (xq, xs_at) = self.x_q8(x, m, k);
        // (one row by the several rows' kernel where the recording's rows are alike: the one-row int8 kernel sums a
        // row in another order)
        let mr = if m == 1 && !self.alike { 1 } else { crate::quant::shaders::MULTI_ROWS };
        let name = match (q.dtype, mr == 1) {
            (ggml_quants::GgmlType::Q3_K, true) => "chain-q8-Q3_K-decode",
            (ggml_quants::GgmlType::Q3_K, false) => "chain-q8-Q3_K-multi",
            (ggml_quants::GgmlType::Q4_K, true) => "chain-q8-Q4_K-decode",
            (ggml_quants::GgmlType::Q4_K, false) => "chain-q8-Q4_K-multi",
            (ggml_quants::GgmlType::Q5_K, true) => "chain-q8-Q5_K-decode",
            (ggml_quants::GgmlType::Q5_K, false) => "chain-q8-Q5_K-multi",
            (ggml_quants::GgmlType::Q4_0, true) => "chain-q8-Q4_0-decode",
            (ggml_quants::GgmlType::Q4_0, false) => "chain-q8-Q4_0-multi",
            (ggml_quants::GgmlType::Q8_0, true) => "chain-q8-Q8_0-decode",
            (ggml_quants::GgmlType::Q8_0, false) => "chain-q8-Q8_0-multi",
            (_, true) => "chain-q8-Q6_K-decode",
            (_, false) => "chain-q8-Q6_K-multi",
        };
        let subgroups = self.gpu().subgroups32();
        let pipeline = self.gpu().named_pipeline(name, || crate::quant::shaders::rb_kernel_q8(q.dtype, crate::quant::shaders::rb_rows_q8(q.dtype, mr), mr, subgroups).expect("a K-quant's int8 kernel"));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = crate::quant::shaders::grid_q8(q.dtype, m, mr, *rows, subgroups);
            self.dispatch_kept(&pipeline, chunk, buffer(&xq), buffer(y), &words, groups);
        }
        true
    }
}

/// Whether the shared expert's fused kernels are taken (OAIY_MOE_UNFUSED: its matmuls, SwiGLU and sum each their own).
fn moe_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_MOE_UNFUSED").is_none())
}

/// Whether a decode step's one row of `w`'s type takes the K-quants' int8 kernel on this device (its activations
/// rounded to int8 once and multiplied with packed int8 dot products, llama.cpp's MMVQ arithmetic) rather than the
/// f32 one. Neither is the faster everywhere. The f32 kernels were chosen on an RTX 5090, where the weights' reads
/// decide (a 27B step's matmuls at 1,090 GB/s); on an NVIDIA A2 held to 45 W, ten SMs at their power cap's clocks, the
/// work of unpacking each weight decides, and the int8 kernel made Qwen3.5 9B's replies 19.1 tokens a second where
/// the f32 one made 12.5 (the same replies but where a near-tie went the other way, as a check's rows from int8 do).
/// So it is timed, once a type on each device, at the first step's row of it: [`time_decode`]. The int8 kernel is
/// taken where it is 5% faster or more; a tie keeps f32's exact sums. `OAIY_Q8_DECODE=1` or `0` decides without
/// timing; `OAIY_NO_Q8` (no int8 kernels) and a device whose packed dot products are wrapped (`Gpu::int8_rows`) never
/// get here. Q8_0 too, a step's one row alone (its weights int8 already; a few rows' by f32 as before): on the A2 its
/// 4096 x 4096 in 116 us where the f32 kernel took 1,867, Qwen3.5 9B's delta nets' gates [64, 4096] in 13 where 50.
pub(crate) fn decode_in_int8(backend: &WgpuBackend, w: &QuantizedTensor) -> bool {
    use ggml_quants::GgmlType;
    let Some(dtype) = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).map(|q| q.dtype) else { return false };
    if !matches!(dtype, GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q4_0 | GgmlType::Q8_0) {
        return false;
    }
    let gpu = &backend.gpu;
    if let Some(known) = gpu.decode_int8.lock().unwrap_or_else(|p| p.into_inner()).get(&dtype) {
        return *known;
    }
    static ASKED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let asked = *ASKED.get_or_init(|| match std::env::var("OAIY_Q8_DECODE").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    });
    let timed = if asked.is_some() { None } else { time_decode(backend, dtype) };
    let int8 = match (asked, timed) {
        (Some(asked), _) => asked,
        (None, Some((f32_s, int8_s))) => {
            let int8 = int8_s < f32_s * 0.95;
            eprintln!(
                "{dtype:?}: a step's row in {:.0} us from int8 activations, {:.0} us in f32 (a 4096 x 4096 weight): {}",
                int8_s * 1e6,
                f32_s * 1e6,
                if int8 { "int8" } else { "f32" }
            );
            int8
        }
        // (no room for the timing's weight on the device: the f32 kernel, as before)
        (None, None) => false,
    };
    gpu.decode_int8.lock().unwrap_or_else(|p| p.into_inner()).insert(dtype, int8);
    int8
}

/// [`crate::quant::shaders::coop_tiled_split`]'s pipeline's name for `dtype`, where it has one.
fn split_name(dtype: ggml_quants::GgmlType) -> Option<&'static str> {
    use ggml_quants::GgmlType as T;
    Some(match dtype {
        T::Q3_K => "chain-coop-split-Q3_K",
        T::Q4_K => "chain-coop-split-Q4_K",
        T::Q5_K => "chain-coop-split-Q5_K",
        T::Q6_K => "chain-coop-split-Q6_K",
        T::Q8_0 => "chain-coop-split-Q8_0",
        _ => return None,
    })
}

/// Whether a prompt's tensor-core matmul of `dtype` takes [`crate::quant::shaders::coop_tiled_split`] (subgroups that decode
/// apart from those that multiply) on this device rather than [`crate::quant::shaders::coop_tiled`] (each subgroup both, in
/// turn). The split kernel wants a device whose subgroups are 32 lanes (`Gpu::subgroups32`; every other device takes
/// the other); there, which is the faster is timed, once a type, at its first matmul ([`time_coop`]): an NVIDIA A2 at
/// 1,200 MHz ran Q4_K's [12288, 4096] by 512 tokens at 6.8 TFLOPS split and 6.2 together, Q6_K's at 5.7 and 5.1, Q5_K's
/// at 5.9 and 5.7 (Qwen3.5 9B's 6,933-token prompt 322 tokens a second together, 361 split), its f32 sums the nearer
/// the host's. The split kernel is taken unless the other is 2% faster or more (its sums the nearer, and Q5_K's the two
/// within a few percent: timed one after the other by 512 tokens in a server's warm-up, the split one came out 3% the
/// slower where by itself, and in the 9B's prompt, it was 1 to 4% the faster). `OAIY_COOP_SPLIT=1` or `0` decides
/// without timing (as profiling wants: `OAIY_CHAIN_PROFILE`'s pass a dispatch slowed the split kernel's timing to a tie).
/// Whether f16 weights' tensor-core matmuls and convolutions take [`crate::quant::shaders::coop_split_f16`]'s kernel (its
/// subgroups' decode a copy): a device whose subgroups are 32 lanes and whose workgroups take 512 threads, unless
/// `OAIY_COOP_SPLIT=0`.
pub(crate) fn split_f16_on(gpu: &Gpu) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    gpu.subgroups32() && gpu.limits.max_compute_invocations_per_workgroup >= 512 && gpu.limits.max_compute_workgroup_size_x >= 512 && !*OFF.get_or_init(|| std::env::var("OAIY_COOP_SPLIT").is_ok_and(|v| v == "0"))
}

pub(crate) fn coop_split_for(backend: &WgpuBackend, dtype: ggml_quants::GgmlType) -> bool {
    let gpu = &backend.gpu;
    // (512 threads a workgroup)
    if !gpu.subgroups32() || split_name(dtype).is_none() || gpu.limits.max_compute_invocations_per_workgroup < 512 || gpu.limits.max_compute_workgroup_size_x < 512 {
        return false;
    }
    if let Some(known) = gpu.coop_split.lock().unwrap_or_else(|p| p.into_inner()).get(&dtype) {
        return *known;
    }
    static ASKED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let asked = *ASKED.get_or_init(|| match std::env::var("OAIY_COOP_SPLIT").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    });
    let timed = if asked.is_some() { None } else { time_coop(backend, dtype) };
    let split = match (asked, timed) {
        (Some(asked), _) => asked,
        (None, Some((together, split))) => {
            let faster = together >= split * 0.98;
            eprintln!(
                "{dtype:?}: a prompt's matmul in {:.2} ms with its decode apart, {:.2} ms together (a 4096 x 4096 weight by 1,024 tokens): {}",
                split * 1e3,
                together * 1e3,
                if faster { "apart" } else { "together" }
            );
            faster
        }
        (None, None) => false,
    };
    gpu.coop_split.lock().unwrap_or_else(|p| p.into_inner()).insert(dtype, split);
    split
}

/// A prompt's matmul of a 4096 x 4096 `dtype` weight by 1,024 tokens (a prompt's chunk) on the tensor cores, each
/// kernel's (together, apart): the better of three chains of 2 each, the two kernels' chains in turn (a drift of the
/// clocks the same for both). None where the weight finds no room on the device.
fn time_coop(backend: &WgpuBackend, dtype: ggml_quants::GgmlType) -> Option<(f64, f64)> {
    use ggml_quants::GgmlType as T;
    let (n, k, m) = (4096usize, 4096usize, 1024usize);
    let (block, bytes) = match dtype {
        T::Q3_K => (256, 110),
        T::Q4_K => (256, 144),
        T::Q5_K => (256, 176),
        T::Q6_K => (256, 210),
        T::Q8_0 => (32, 34),
        _ => return None,
    };
    // (any bytes will do for a time: the kernels do the same work whatever the weights are)
    let raw: Vec<u8> = (0..n * (k / block) * bytes).map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_shr(13) as u8).collect();
    let w = backend.upload_on_card(QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
    let q = w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?;
    let padded = m.div_ceil(crate::quant::shaders::COOP_TILE as usize) * crate::quant::shaders::COOP_TILE as usize;
    let (x16, y) = (backend.vec(crate::quant::shaders::x_f16_tiled_words(m, k).max(padded * k / 2)), backend.vec(m * n));
    let gpu = &backend.gpu;
    // (the names [`Recorder::matmul_rows_coop_split`]'s pipelines go by)
    let name = match dtype {
        T::Q3_K => "chain-coop-Q3_K",
        T::Q4_K => "chain-coop-Q4_K",
        T::Q5_K => "chain-coop-Q5_K",
        T::Q6_K => "chain-coop-Q6_K",
        _ => "chain-coop-Q8_0",
    };
    let together = gpu.named_pipeline(name, || crate::quant::shaders::coop_tiled(dtype).expect("a K-quant's tensor-core kernel"));
    let apart = gpu.named_pipeline(split_name(dtype)?, || crate::quant::shaders::coop_tiled_split(dtype).expect("a K-quant's tensor-core kernel, its work apart"));
    let run = |pipeline: &std::sync::Arc<wgpu::ComputePipeline>| -> f64 {
        let t = std::time::Instant::now();
        let mut rec = Recorder::new(backend);
        for _ in 0..2 {
            for (chunk, row0, rows) in &q.chunks {
                let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 1, 0];
                rec.dispatch_kept(pipeline, chunk, buffer(&x16), buffer(&y), &words, (rows.div_ceil(128), (m as u32).div_ceil(128), 1));
            }
        }
        rec.read_range(&y, 0, 1);
        Box::new(rec).finish();
        t.elapsed().as_secs_f64() / 2.0
    };
    run(&together);
    run(&apart);
    let (mut a, mut b) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        a = a.min(run(&together));
        b = b.min(run(&apart));
    }
    Some((a, b))
}

/// What [`decode_in_int8`] decided for `dtype` on this device, once it has (false before): for a test's tolerance.
#[cfg(test)]
pub(crate) fn decoded_in_int8(backend: &WgpuBackend, dtype: ggml_quants::GgmlType) -> bool {
    backend.gpu.decode_int8.lock().unwrap_or_else(|p| p.into_inner()).get(&dtype).copied().unwrap_or(false)
}

/// A step's row of a 4096 x 4096 `dtype` weight on this device, in f32 and from int8 activations: the better of three
/// chains of 16 matmuls each way, a matmul's time (the int8 one's quantizing of its row once a chain, as a step
/// quantizes a row once for the matmuls that read it). None where the weight finds no room on the device.
fn time_decode(backend: &WgpuBackend, dtype: ggml_quants::GgmlType) -> Option<(f64, f64)> {
    use ggml_quants::GgmlType;
    let (n, k) = (4096usize, 4096usize);
    let (block, bytes) = match dtype {
        GgmlType::Q4_0 => (32, 18),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::Q3_K => (256, 110),
        GgmlType::Q4_K => (256, 144),
        GgmlType::Q5_K => (256, 176),
        GgmlType::Q6_K => (256, 210),
        _ => return None,
    };
    // (any bytes will do for a time: the kernels do the same work whatever the weights are)
    let raw: Vec<u8> = (0..n * (k / block) * bytes).map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_shr(13) as u8).collect();
    let w = backend.upload_on_card(QuantizedTensor::from_bytes_cpu(raw, vec![n, k], dtype));
    w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?;
    let (x, y) = (backend.vec(k), backend.vec(n));
    DeviceChain::upload(backend, &x, &(0..k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect::<Vec<_>>());
    let time = |int8: bool| -> f64 {
        let run = || {
            let mut rec = Recorder::new(backend);
            for _ in 0..16 {
                if !(int8 && rec.matmul_rows_q8(&w, &x, &y, 1)) {
                    rec.matmul_rows_f32(&w, &x, &y, 1);
                }
            }
            rec.read_range(&y, 0, 1);
            Box::new(rec).finish();
        };
        run();
        (0..3)
            .map(|_| {
                let t = std::time::Instant::now();
                run();
                t.elapsed().as_secs_f64() / 16.0
            })
            .fold(f64::MAX, f64::min)
    };
    Some((time(false), time(true)))
}
