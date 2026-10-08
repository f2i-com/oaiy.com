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
            let groups = crate::shaders::grid(q.dtype, m, *rows);
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, groups);
        }
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for one to eight rows against IQ4_XS weights
    /// ([`crate::shaders::iq4_xs_few`]). False for another type, more rows, or a width that is not whole blocks.
    pub(crate) fn matmul_rows_iq4_xs(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::IQ4_XS || m == 0 || m > crate::shaders::IQ4_FEW_MAX || k % 256 != 0 || q.row_bytes != k / 256 * 136 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an IQ4_XS matmul [{n}, {k}] of {m} rows");
        const NAMES: [&str; crate::shaders::IQ4_FEW_MAX] =
            ["chain-iq4xs-few-1", "chain-iq4xs-few-2", "chain-iq4xs-few-3", "chain-iq4xs-few-4", "chain-iq4xs-few-5", "chain-iq4xs-few-6", "chain-iq4xs-few-7", "chain-iq4xs-few-8"];
        let pipeline = self.gpu().named_pipeline(NAMES[m - 1], || crate::shaders::iq4_xs_few(m));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0];
            let groups = rows.div_ceil(8);
            self.dispatch_kept(&pipeline, chunk, buffer(x), buffer(y), &words, (groups.min(65535), groups.div_ceil(65535), 1));
        }
        true
    }

    /// [`Self::matmul_rows_iq4_xs`] from `x`'s rows as int8 ([`crate::shaders::iq4_xs_few_q8`]; quantized once, as for
    /// [`Self::matmul_rows_q8`]): a check of drafts' rows. False as that one is.
    pub(crate) fn matmul_rows_iq4_xs_q8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::IQ4_XS || m == 0 || m > crate::shaders::IQ4_FEW_MAX || k % 256 != 0 || q.row_bytes != k / 256 * 136 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an IQ4_XS int8 matmul [{n}, {k}] of {m} rows");
        let (xq, xs_at) = self.x_q8(x, m, k);
        const NAMES: [&str; crate::shaders::IQ4_FEW_MAX] =
            ["chain-iq4xs-q8-1", "chain-iq4xs-q8-2", "chain-iq4xs-q8-3", "chain-iq4xs-q8-4", "chain-iq4xs-q8-5", "chain-iq4xs-q8-6", "chain-iq4xs-q8-7", "chain-iq4xs-q8-8"];
        let pipeline = self.gpu().named_pipeline(NAMES[m - 1], || crate::shaders::iq4_xs_few_q8(m));
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

    /// `x`'s `m` rows of `k` as int8 ([`crate::shaders::QUANT_Q8`]), and where their scales start: made once for every
    /// int8 matmul that reads them until something writes `x`.
    pub(super) fn x_q8(&mut self, x: &DeviceVec, m: usize, k: usize) -> (DeviceVec, usize) {
        let (len, xs_at) = crate::shaders::q8_len(m, k);
        let xb = buffer(x).clone();
        if let Some((.., xq)) = self.q8.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            return (xq.clone(), xs_at);
        }
        let xq = self.scratch(len);
        let quant = self.gpu().named_pipeline("chain-q8-quantize", || crate::shaders::QUANT_Q8.to_string());
        let blocks = (m * k / 32) as u32;
        let groups = blocks.div_ceil(256);
        let d = self.gpu().dummy().clone();
        self.dispatch_kept(&quant, &d, buffer(x), buffer(&xq), &[k as u32, m as u32, xs_at as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.q8.push((xb, m, k, xq.clone()));
        (xq, xs_at)
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for a prompt's rows on the tensor cores
    /// ([`crate::shaders::coop_tiled`]: f16 weights and tokens into f32 sums). False where the device has no cooperative
    /// matrices or the type no such kernel.
    pub(crate) fn matmul_rows_coop(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        self.matmul_rows_coop_split(w, x, y, m, None)
    }

    /// [`Self::matmul_rows_coop`], split along k as given (else as [`crate::shaders::coop_splits`] chooses).
    pub(crate) fn matmul_rows_coop_split(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        use ggml_quants::GgmlType as T;
        let name = match q.dtype {
            T::Q3_K => "chain-coop-Q3_K",
            T::Q4_K => "chain-coop-Q4_K",
            T::Q5_K => "chain-coop-Q5_K",
            T::Q6_K => "chain-coop-Q6_K",
            T::Q8_0 => "chain-coop-Q8_0",
            T::Q2_0 => "chain-coop-Q2_0",
            T::Q4_0 => "chain-coop-Q4_0",
            T::Q5_0 => "chain-coop-Q5_0",
            T::IQ4_NL => "chain-coop-IQ4_NL",
            T::IQ4_XS => "chain-coop-IQ4_XS",
            _ => return false,
        };
        if !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) || k % 256 != 0 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: a tensor-core matmul [{n}, {k}] of {m} rows");
        let x16 = self.x16_tiled(x, m, k);
        let tile = crate::shaders::COOP_TILE;
        let dtype = q.dtype;
        let pipeline = self.gpu().named_pipeline(name, || crate::shaders::coop_tiled(dtype).expect("a K-quant's tensor-core kernel"));
        let tiles = q.chunks.iter().map(|(_, _, rows)| rows.div_ceil(tile)).max().unwrap_or(1) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, m, n, y, split);
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, splits, 0];
            self.dispatch_kept(&pipeline, chunk, buffer(&x16), &out, &words, (rows.div_ceil(tile), (m as u32).div_ceil(tile), splits));
        }
        self.coop_sum(parts, m, n, y, splits);
        true
    }

    /// The tokens' rows `x` (`m` of `k`) as [`crate::shaders::X_F16_TILED`] gives them (f16, padded to the tile and to
    /// a step of 32): once for every tensor-core matmul that reads them until something writes `x`.
    pub(super) fn x16_tiled(&mut self, x: &DeviceVec, m: usize, k: usize) -> DeviceVec {
        let tile = crate::shaders::COOP_TILE;
        let padded = (m as u32).div_ceil(tile) as usize * tile as usize;
        let xb = buffer(x).clone();
        if let Some((.., v)) = self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == k) {
            return v.clone();
        }
        let words = crate::shaders::x_f16_tiled_words(m, k);
        let v = self.scratch(words);
        let conv = self.gpu().named_pipeline("chain-x-f16-tiled", || crate::shaders::X_F16_TILED.to_string());
        let groups = (words as u32).div_ceil(256);
        let d = self.gpu().dummy().clone();
        self.dispatch_kept(&conv, &d, buffer(x), buffer(&v), &[k as u32, m as u32, padded as u32], (groups.min(65535), groups.div_ceil(65535), 1));
        self.x16.push((xb, m, k, v.clone()));
        v
    }

    /// A tensor-core matmul's splits along k (as given, else as [`crate::shaders::coop_splits`] chooses for `tiles`
    /// workgroups), where its sums go (`y`, or a part of scratch a split, added into `y` after), and the parts.
    pub(super) fn coop_parts(&mut self, tiles: u32, k: usize, m: usize, n: usize, y: &DeviceVec, split: Option<u32>) -> (u32, wgpu::Buffer, Option<DeviceVec>) {
        // a matmul of too few tiles to fill the GPU's last wave split along k: each split's sums into a part of
        // scratch, then the parts added into y
        let units = self.gpu().coop_units();
        let steps = k.div_ceil(32) as u32;
        let splits = split.unwrap_or_else(|| crate::shaders::coop_splits(tiles, units, steps));
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
    /// ([`crate::shaders::coop_tiled_f16`]), split along k as given (else as chosen). False where the device has no
    /// cooperative matrices.
    pub(crate) fn matmul_f16_coop(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, m: usize, split: Option<u32>) -> bool {
        if !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) || k % 4 != 0 || m.div_ceil(crate::shaders::COOP_TILE as usize) > 65535 {
            return false;
        }
        let x16 = self.x16_tiled(x, m, k);
        let tile = crate::shaders::COOP_TILE;
        let pipeline = self.gpu().named_pipeline("chain-coop-f16", crate::shaders::coop_tiled_f16);
        let tiles = (n as u32).div_ceil(tile) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, m, n, y, split);
        let words = [k as u32, n as u32, m as u32, 0, n as u32, 0, splits, 0];
        self.dispatch_kept(&pipeline, buffer(w), buffer(&x16), &out, &words, ((n as u32).div_ceil(tile), (m as u32).div_ceil(tile), splits));
        self.coop_sum(parts, m, n, y, splits);
        true
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for a prompt's rows, from `x`'s rows as int8 (quantized once,
    /// as for [`Self::matmul_rows_q8`]) through [`crate::shaders::tiled_q8`]. False for a type without that kernel.
    pub(crate) fn matmul_rows_tq8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if q.dtype != ggml_quants::GgmlType::Q3_K || k % 256 != 0 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an int8 tiled matmul [{n}, {k}] of {m} rows");
        let (xq, xs_at) = self.x_q8(x, m, k);
        let pipeline = self.gpu().named_pipeline("chain-tq8-Q3_K", || crate::shaders::tiled_q8(ggml_quants::GgmlType::Q3_K).expect("Q3_K's int8 tiled kernel"));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = (rows.div_ceil(crate::shaders::TQ8_ROWS), (m as u32).div_ceil(crate::shaders::TQ8_TOKENS), 1);
            self.dispatch_kept(&pipeline, chunk, buffer(&xq), buffer(y), &words, groups);
        }
        true
    }

    /// `y[r] = W x[r]` as [`ChainRecorder::matmul_rows`] for several rows, from `x`'s rows as int8 (quantized once,
    /// [`crate::shaders::QUANT_Q8`], for every matmul that reads them until something writes `x`): the K-quants' int8
    /// kernels, a check of drafts' rows in about 1.3 of a step's time where the f32 kernels take 1.7. False for a type
    /// without one (nothing recorded).
    pub(super) fn matmul_rows_q8(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) -> bool {
        let q = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()).expect("a weight this adapter holds");
        let (n, k) = (w.shape()[0], w.shape()[1]);
        let block = if q.dtype == ggml_quants::GgmlType::Q4_0 { 32 } else { 256 };
        if !matches!(q.dtype, ggml_quants::GgmlType::Q3_K | ggml_quants::GgmlType::Q4_K | ggml_quants::GgmlType::Q5_K | ggml_quants::GgmlType::Q6_K | ggml_quants::GgmlType::Q4_0) || k % block != 0 {
            return false;
        }
        assert!(x.len >= m * k && y.len >= m * n, "chain: an int8 matmul [{n}, {k}] of {m} rows");
        let (xq, xs_at) = self.x_q8(x, m, k);
        // (one row by the several rows' kernel where the recording's rows are alike: the one-row int8 kernel sums a
        // row in another order)
        let mr = if m == 1 && !self.alike { 1 } else { crate::shaders::MULTI_ROWS };
        let name = match (q.dtype, mr == 1) {
            (ggml_quants::GgmlType::Q3_K, true) => "chain-q8-Q3_K-decode",
            (ggml_quants::GgmlType::Q3_K, false) => "chain-q8-Q3_K-multi",
            (ggml_quants::GgmlType::Q4_K, true) => "chain-q8-Q4_K-decode",
            (ggml_quants::GgmlType::Q4_K, false) => "chain-q8-Q4_K-multi",
            (ggml_quants::GgmlType::Q5_K, true) => "chain-q8-Q5_K-decode",
            (ggml_quants::GgmlType::Q5_K, false) => "chain-q8-Q5_K-multi",
            (ggml_quants::GgmlType::Q4_0, true) => "chain-q8-Q4_0-decode",
            (ggml_quants::GgmlType::Q4_0, false) => "chain-q8-Q4_0-multi",
            (_, true) => "chain-q8-Q6_K-decode",
            (_, false) => "chain-q8-Q6_K-multi",
        };
        let pipeline = self.gpu().named_pipeline(name, || crate::shaders::rb_kernel_q8(q.dtype, crate::shaders::rb_rows(q.dtype, mr), mr).expect("a K-quant's int8 kernel"));
        for (chunk, row0, rows) in &q.chunks {
            let words = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, xs_at as u32, 0];
            let groups = crate::shaders::grid(q.dtype, m, *rows);
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
