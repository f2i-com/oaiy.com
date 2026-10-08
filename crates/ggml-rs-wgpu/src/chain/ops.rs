//! The ops a recording takes ([`ChainRecorder`]): each one's checks and its kernels' dispatches.
use super::*;

impl ChainRecorder for Recorder<'_> {
    fn matmul_rows(&mut self, w: &QuantizedTensor, x: &DeviceVec, y: &DeviceVec, m: usize) {
        // several rows (a check of drafts, a short chunk) from int8 activations, where the type has kernels for them
        // (OAIY_NO_Q8: the f32 ones)
        static Q8: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let q8 = *Q8.get_or_init(|| std::env::var_os("OAIY_NO_Q8").is_none());
        // (a prompt's rows: the tensor cores where the device has them (f16 into f32), else the int8 tiled kernel,
        // llama.cpp's MMQ's arithmetic, where the type has one)
        // (IQ4_XS against a step's row or a check's few: a kernel of its own; OAIY_NO_IQ4_FEW: the generic one)
        static IQ4: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let iq4 = *IQ4.get_or_init(|| std::env::var_os("OAIY_NO_IQ4_FEW").is_none());
        let done = (iq4 && self.matmul_rows_iq4_xs(w, x, y, m))
            || ((2..=crate::shaders::MULTI_MAX).contains(&m) && q8 && self.matmul_rows_q8(w, x, y, m))
            || (m > crate::shaders::MULTI_MAX && self.matmul_rows_coop(w, x, y, m))
            || (m > crate::shaders::MULTI_MAX && q8 && self.matmul_rows_tq8(w, x, y, m));
        if !done {
            self.matmul_rows_f32(w, x, y, m);
        }
        self.weigh(2.0 * m as f64 * w.shape().iter().product::<usize>() as f64);
    }

    fn rmsnorm(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, eps: f32) {
        let pipeline = self.named("chain-rmsnorm", RMSNORM);
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[x.len as u32, eps.to_bits()], (1, 1, 1));
    }

    fn rmsnorm_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        assert!(rows > 0 && x.len % rows == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: rmsnorm of {rows} rows of {}", x.len);
        let n = x.len / rows;
        // rows a multiple of 4 long (a model's width, a head's): vec4 loads
        let pipeline = if n % 4 == 0 { self.gpu().named_pipeline("chain-rmsnorm-rows4", || RMSNORM_ROWS4.to_string()) } else { self.named("chain-rmsnorm-rows", RMSNORM_ROWS) };
        let r = rows as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), 0, r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn rmsnorm_heads_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, heads: usize, eps: f32) {
        let r = rows * heads;
        assert!(r > 0 && x.len % r == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: a multi-head rmsnorm of {rows} rows of {heads} heads ({})", x.len);
        let n = x.len / r;
        // (the norms' kernels take the weight's rows in turn: row r of x by w's row r % heads)
        let pipeline = if n % 4 == 0 { self.gpu().named_pipeline("chain-rmsnorm-rows4", || RMSNORM_ROWS4.to_string()) } else { self.named("chain-rmsnorm-rows", RMSNORM_ROWS) };
        let rr = r as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), heads as u32, rr], (rr.min(65535), rr.div_ceil(65535), 1));
    }

    fn rmsnorm_silu_rows(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        assert!(rows > 0 && x.len % rows == 0 && w.len >= x.len / rows && out.len >= x.len, "chain: rmsnorm and SiLU of {rows} rows of {}", x.len);
        let n = x.len / rows;
        let silu = |body: &str| {
            let at = "y4[at + i] = x4[at + i] * inv * w4[wat + i];";
            assert_eq!(body.matches(at).count(), 1, "the norm's store");
            body.replace(at, "let v = x4[at + i] * inv * w4[wat + i];\n        y4[at + i] = v / (vec4<f32>(1.0) + exp(-v));")
        };
        assert!(n % 4 == 0, "chain: rmsnorm and SiLU of rows {n} long (a multiple of 4)");
        let pipeline = self.gpu().named_pipeline("chain-rmsnorm-silu-rows4", || silu(RMSNORM_ROWS4));
        let r = rows as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), 0, r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn add_rmsnorm_rows(&mut self, x: &DeviceVec, y: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, eps: f32) {
        let n = x.len / rows.max(1);
        if rows == 0 || n % 4 != 0 || n * rows != x.len || y.len < x.len || w.len < n || out.len < x.len || Arc::ptr_eq(&x.inner, &out.inner) {
            self.add(x, y);
            self.rmsnorm_rows(x, w, out, rows, eps);
            return;
        }
        let d = self.gpu().dummy().clone();
        let r = rows as u32;
        self.dispatch_wide("chain-add-rmsnorm-rows4", ADD_RMSNORM_ROWS4, [buffer(y), buffer(w), &d, &d, &d, &d, buffer(x), buffer(out)], &[n as u32, eps.to_bits(), r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn add(&mut self, acc: &DeviceVec, y: &DeviceVec) {
        let pipeline = self.named("chain-add", ADD);
        self.dispatch_kept(&pipeline, buffer(y), buffer(y), buffer(acc), &[acc.len as u32], grid((acc.len as u32).div_ceil(256)));
    }

    fn silu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        let ff = out.len / rows;
        assert!(rows > 0 && out.len == rows * ff && fused.len >= 2 * out.len, "chain: SwiGLU of {rows} rows");
        let pipeline = self.named("chain-silu-mul-split", SILU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[ff as u32, rows as u32], grid((out.len as u32).div_ceil(256)));
    }

    fn gelu_mul_split_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        let ff = out.len / rows;
        assert!(rows > 0 && out.len == rows * ff && fused.len >= 2 * out.len, "chain: GeGLU of {rows} rows");
        let pipeline = self.named("chain-gelu-mul-split", GELU_MUL_SPLIT);
        self.dispatch_kept(&pipeline, buffer(fused), buffer(fused), buffer(out), &[ff as u32, rows as u32], grid((out.len as u32).div_ceil(256)));
    }

    fn keep_groups(&mut self, keep: bool) {
        self.keep = keep;
    }

    fn hold(&mut self) {
        self.hold = true;
    }

    fn flush(&mut self) {
        // let go: what is held, then what is left, then the reads so far copied out after them
        self.hold = false;
        let held = std::mem::take(&mut self.held);
        if !held.is_empty() {
            self.gpu().submit_piece(held);
        }
        for list in std::mem::take(&mut self.lists) {
            self.gpu().feed(list);
        }
        self.submit_piece();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        for (from, offset, staging, len) in &self.reads[self.copied..] {
            if *len > 0 {
                enc.copy_buffer_to_buffer(from, (*offset * 4) as u64, staging, 0, (*len * 4) as u64);
            }
        }
        self.copied = self.reads.len();
        if let Some(staging) = self.resolve_pieces(&mut enc) {
            self.stamped = Some(staging);
        }
        self.flushed = Some(self.gpu().submit_after(vec![enc.finish()]));
    }

    fn exl3_rows(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.exl3_rows_of(w, x, None, y, rows);
    }

    fn rmsnorm_streams(&mut self, x: &DeviceVec, w: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, eps: f32) {
        let n = x.len / (rows * streams).max(1);
        assert!(rows > 0 && streams > 0 && n * rows * streams == x.len && w.len >= streams * n && out.len >= x.len, "chain: a norm of {rows} rows of {streams} streams");
        let pipeline = self.named("chain-rmsnorm-rows", RMSNORM_ROWS);
        let r = (rows * streams) as u32;
        self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(out), &[n as u32, eps.to_bits(), streams as u32, r], (r.min(65535), r.div_ceil(65535), 1));
    }

    fn hc_gates(&mut self, t: &DeviceVec, post: &DeviceVec, rows: usize, rank: usize, writes: usize, streams: usize) {
        assert!(t.len >= rows * (rank + writes) && post.len >= rows * writes.max(1), "chain: a hyper-connection's gates");
        let len = (rows * (rank + writes)) as u32;
        // nothing read: a dummy at the read bindings (a buffer written may not be bound as read too)
        let d = self.gpu().dummy().clone();
        self.dispatch_wide("chain-hc-gates", HC_GATES, [&d, &d, &d, &d, &d, &d, buffer(t), buffer(post)], &[rank as u32, writes as u32, streams as u32, rows as u32], (len.div_ceil(256), 1, 1));
    }

    fn hc_mix(&mut self, logits: &DeviceVec, normed: &DeviceVec, out: &DeviceVec, rows: usize, streams: usize, d: usize) {
        assert!(logits.len >= rows * streams * d && normed.len >= rows * streams * d && out.len >= rows * d, "chain: a hyper-connection's mix");
        let pipeline = self.named("chain-hc-mix", HC_MIX);
        self.dispatch_kept(&pipeline, buffer(logits), buffer(normed), buffer(out), &[d as u32, streams as u32, rows as u32], (((rows * d) as u32).div_ceil(256), 1, 1));
    }

    fn stream_apply(&mut self, x: &DeviceVec, y: &DeviceVec, post: &DeviceVec, rows: usize, streams: usize, d: usize) {
        assert!(x.len >= rows * streams * d && y.len >= rows * d && post.len >= rows * streams, "chain: a hyper-connection's write-back");
        let pipeline = self.named("chain-stream-apply", STREAM_APPLY);
        self.dispatch_kept(&pipeline, buffer(post), buffer(y), buffer(x), &[d as u32, streams as u32, rows as u32], (((rows * streams * d) as u32).div_ceil(256), 1, 1));
    }

    fn moe_rows(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]) {
        if let Some(q) = experts.as_any().and_then(|a| a.downcast_ref::<crate::quant_moe::QuantMoe>()) {
            return q.record(self, x, out, assign);
        }
        let g = experts.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()).expect("experts this adapter holds as groups");
        g.record(self, x, out, assign);
    }

    fn moe_routed(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize) -> bool {
        if let Some(q) = experts.as_any().and_then(|a| a.downcast_ref::<crate::quant_moe::QuantMoe>()) {
            return q.record_routed(self, x, out, logits, top_k, rows, None);
        }
        let Some(g) = experts.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()) else { return false };
        g.record_routed(self, x, out, logits, top_k, rows, None)
    }

    fn moe_routed_into(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, streams_x: &DeviceVec, post: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, streams: usize) -> bool {
        // no vector for the sums: they go into the streams
        let none = self.gpu().dummy_rw().clone();
        let out = DeviceVec { len: 0, inner: Arc::new(none) };
        if let Some(q) = experts.as_any().and_then(|a| a.downcast_ref::<crate::quant_moe::QuantMoe>()) {
            return q.record_routed(self, x, &out, logits, top_k, rows, Some((streams_x, post, streams)));
        }
        let Some(g) = experts.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3MoeGrouped>()) else { return false };
        g.record_routed(self, x, &out, logits, top_k, rows, Some((streams_x, post, streams)))
    }

    fn moe_shared(&mut self, experts: &dyn ggml_rs::exl3::Experts, x: &DeviceVec, out: &DeviceVec, rows: usize) -> bool {
        let Some(host) = experts.as_any().and_then(|a| a.downcast_ref::<crate::quant_host::QuantMoeHost>()) else { return false };
        host.record_shared(self, x, out, rows)
    }

    fn axpy_at(&mut self, acc: &DeviceVec, y: &DeviceVec, weights: &DeviceVec, at: usize, len: usize) {
        assert!(acc.len >= len && y.len >= len && weights.len > at, "chain: a weighted term of {len}");
        let pipeline = self.named("chain-axpy-at", AXPY_AT);
        self.dispatch_kept(&pipeline, buffer(weights), buffer(y), buffer(acc), &[len as u32, at as u32], grid((len as u32).div_ceil(256)));
    }

    fn copy_cols(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, width: usize, stride: usize, at: usize) {
        assert!(rows > 0 && at + width <= stride && src.len >= rows * stride && dst.len >= rows * width, "chain: {rows} rows' columns {at}..{} of {stride}", at + width);
        let pipeline = self.named("chain-copy-cols", COPY_COLS);
        self.dispatch_kept(&pipeline, buffer(src), buffer(src), buffer(dst), &[width as u32, rows as u32, stride as u32, at as u32], grid(((rows * width) as u32).div_ceil(256)));
    }

    fn rope_partial_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, rot: usize, table: &DeviceVec) {
        assert!(rot > 0 && rot % 2 == 0 && rot <= head_dim && x.len >= rows * heads * head_dim && table.len >= rows * rot, "chain: RoPE of {rot} of {head_dim}");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * rot / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, 1, rows as u32, rot as u32], grid(pairs.div_ceil(256)));
    }

    fn matmul_f16_rows_f32(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        if rows <= 8 {
            self.matmul_f16_rows(w, n, k, x, y, rows);
        } else {
            assert!(k % 2 == 0 && w.len * 2 >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows.div_ceil(64) <= 65535, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
            self.matmul_f16_tiled(w, n, k, x, y, rows);
            self.weigh(2.0 * (rows * n) as f64 * k as f64);
        }
    }

    fn norm_mod_rows_clean(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, scale_at: usize, shift_at: Option<usize>, norm: ggml_rs::RowNorm, eps: f32, clean: ggml_rs::CleanRows) {
        let set = if clean == ggml_rs::CleanRows::NONE { 0 } else { clean.offset };
        assert!(rows > 0 && x.len >= rows * n && out.len >= rows * n && mods.len >= set + scale_at + n && shift_at.is_none_or(|s| mods.len >= set + s + n) && set < 1 << 30, "chain: a modulated norm of {rows} rows of {n}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let mode = match norm {
            ggml_rs::RowNorm::None => 0u32,
            ggml_rs::RowNorm::Rms => 1,
            ggml_rs::RowNorm::Layer => 2,
        };
        let bound = |v: usize| v.min(u32::MAX as usize) as u32;
        let words = [n as u32, eps.to_bits(), scale_at as u32, shift_at.map_or(u32::MAX, |s| s as u32), rows as u32, mode | (set as u32) << 2, bound(clean.before), bound(clean.from)];
        let r = rows as u32;
        self.dispatch_wide("chain-layernorm-mod-rows", LAYERNORM_MOD_ROWS, [buffer(x), buffer(mods), &d, &d, &d, &d, buffer(out), &drw], &words, (r.min(65535), r.div_ceil(65535), 1));
    }

    fn add_gated_rows_clean(&mut self, x: &DeviceVec, y: &DeviceVec, rows: usize, n: usize, mods: &DeviceVec, gate_at: usize, tanh: bool, clean: ggml_rs::CleanRows) {
        let set = if clean == ggml_rs::CleanRows::NONE { 0 } else { clean.offset };
        assert!(x.len >= rows * n && y.len >= rows * n && mods.len >= set + gate_at + n, "chain: a gated residual of {rows} rows of {n}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let bound = |v: usize| v.min(u32::MAX as usize) as u32;
        let words = [n as u32, rows as u32, gate_at as u32, tanh as u32, bound(clean.before), bound(clean.from), set as u32, 0];
        self.dispatch_wide("chain-add-gated-rows", ADD_GATED_ROWS, [buffer(y), buffer(mods), &d, &d, &d, &d, buffer(x), &drw], &words, grid(((rows * n) as u32).div_ceil(256)));
    }

    fn subsample2x_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        assert!(h % 2 == 0 && w % 2 == 0 && x.len >= h * w * c && out.len >= h * w * c / 4, "chain: subsampling {h}x{w} pixels of {c}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w * c / 4) as u32;
        self.dispatch_wide("chain-subsample2x-rows", SUBSAMPLE2X_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn shuffle_down_mean_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, cin: usize, cout: usize, ft: usize, fs: usize) {
        assert!(fs >= 1 && ft >= 1 && h % fs == 0 && w % fs == 0 && (cin * ft * fs * fs) % cout == 0 && x.len >= h * w * cin && out.len >= h * w / (fs * fs) * cout, "chain: a shuffled mean of {h}x{w} pixels of {cin} into {cout}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w / (fs * fs) * cout) as u32;
        self.dispatch_wide("chain-shuffle-down-mean-add-rows", SHUFFLE_DOWN_MEAN_ADD_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, cin as u32, cout as u32, ft as u32, fs as u32], grid(n.div_ceil(256)));
    }

    fn space_to_depth_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, st: usize, sh: usize, sw: usize) {
        let vol = st * sh * sw;
        assert!(vol > 0 && h % sh == 0 && w % sw == 0 && x.len >= h * w * c && out.len >= h * w / (sh * sw) * c * vol, "chain: space to depth of {h}x{w} pixels of {c} by ({st}, {sh}, {sw})");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w / (sh * sw) * c * vol) as u32;
        self.dispatch_wide("chain-space-to-depth-rows", SPACE_TO_DEPTH_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, c as u32, 0, st as u32, sh as u32, sw as u32], grid(n.div_ceil(256)));
    }

    fn group_mean_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, cin: usize, cout: usize) {
        assert!(cout > 0 && cin % cout == 0 && x.len >= rows * cin && out.len >= rows * cout, "chain: a group mean of {rows} rows of {cin} into {cout}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (rows * cout) as u32;
        self.dispatch_wide("chain-group-mean-add-rows", GROUP_MEAN_ADD_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[rows as u32, cin as u32, cout as u32], grid(n.div_ceil(256)));
    }

    fn add_f16(&mut self, w: &DeviceVec, d: &DeviceVec, len: usize) {
        assert!(len % 2 == 0 && w.len * 2 >= len && d.len >= len, "chain: {len} f16 values plus f32");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let words = (len / 2) as u32;
        self.dispatch_wide("chain-add-f16", ADD_F16, [buffer(d), &dm, &dm, &dm, &dm, &dm, buffer(w), &drw], &[words], grid(words.div_ceil(256)));
    }

    fn conv3d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, x: &DeviceVec, frames: usize, h: usize, wd: usize, y: &DeviceVec) {
        self.conv_taps(w, b, cout, cin, 27, x, cin, frames, h, wd, y);
    }

    fn depth_to_space_rows(&mut self, x: &DeviceVec, out: &DeviceVec, frames: usize, h: usize, w: usize, c: usize, st: usize, sh: usize, sw: usize, drop: usize) {
        let ot = frames * st - drop;
        let voxels = ot * h * sh * w * sw;
        assert!(drop < frames * st && x.len >= frames * h * w * c * st * sh * sw && out.len >= voxels * c, "chain: depth to space of {frames}x{h}x{w} voxels of {c} channels by ({st}, {sh}, {sw})");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-depth-to-space-rows", DEPTH_TO_SPACE_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[c as u32, st as u32, sh as u32, sw as u32, voxels as u32, (h * sh) as u32, (w * sw) as u32, drop as u32], grid(((voxels * c) as u32).div_ceil(256)));
    }

    fn conv_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, x: &DeviceVec, h: usize, wd: usize, y: &DeviceVec) {
        assert!(matches!(k, 1 | 3 | 7), "chain: a {k}x{k} convolution");
        self.conv_taps(w, b, cout, cin, k * k, x, cin, 1, h, wd, y);
    }

    fn conv_rows_strided(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, x: &DeviceVec, xs: usize, h: usize, wd: usize, y: &DeviceVec) {
        assert!(matches!(k, 1 | 3 | 7), "chain: a {k}x{k} convolution");
        self.conv_taps(w, b, cout, cin, k * k, x, xs, 1, h, wd, y);
    }

    fn w4a8_f16(&mut self, codes: &DeviceVec, rel: &DeviceVec, channel: &DeviceVec, book: &DeviceVec, rows: usize, cols: usize, rotation: usize, out: &DeviceVec) {
        assert!(
            rows > 0 && cols % 16 == 0 && codes.len * 8 >= rows * cols && rel.len * 4 >= rows * cols / 16 && channel.len >= rows && book.len >= 16 && out.len * 2 >= rows * cols,
            "chain: a W4A8 matrix of {rows} by {cols}"
        );
        assert!(rotation == 0 || (rotation.is_power_of_two() && rotation.trailing_zeros() % 2 == 0 && (4..=4096).contains(&rotation) && cols % rotation == 0), "chain: W4A8's rotation of {rotation} over {cols} columns");
        let drw = self.gpu().dummy_rw().clone();
        let d = self.gpu().dummy().clone();
        let bufs = [buffer(codes), buffer(rel), buffer(channel), buffer(book), &d, &d, buffer(out), &drw];
        let words = [rows as u32, cols as u32];
        if rotation == 0 {
            let n = (rows * cols / 2) as u32;
            self.dispatch_wide("chain-w4a8-f16", W4A8_F16, bufs, &words, grid(n.div_ceil(256)));
        } else {
            let name: &'static str = match rotation {
                4 => "chain-w4a8-f16-rotated-4",
                16 => "chain-w4a8-f16-rotated-16",
                64 => "chain-w4a8-f16-rotated-64",
                256 => "chain-w4a8-f16-rotated-256",
                1024 => "chain-w4a8-f16-rotated-1024",
                _ => "chain-w4a8-f16-rotated-4096",
            };
            let body = W4A8_F16_ROTATED.replace("GS_u", &format!("{rotation}u"));
            let r = rows as u32;
            self.dispatch_wide(name, &body, bufs, &words, ((cols / rotation) as u32, r.min(65535), r.div_ceil(65535)));
        }
    }

    fn window_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, win: usize, shift: usize) {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        assert!(shift < win && x.len >= h * w * c && out.len >= hp * wp * c, "chain: windows of {h}x{w} tokens of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (hp * wp * c) as u32;
        self.dispatch_wide("chain-window-rows", WINDOW_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[h as u32, w as u32, c as u32, win as u32, shift as u32, hp as u32, wp as u32], grid(n.div_ceil(256)));
    }

    fn unwindow_add_rows(&mut self, windows: &DeviceVec, acc: &DeviceVec, h: usize, w: usize, c: usize, win: usize, shift: usize) {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        assert!(shift < win && windows.len >= hp * wp * c && acc.len >= h * w * c, "chain: windows of {h}x{w} tokens of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (h * w * c) as u32;
        self.dispatch_wide("chain-unwindow-add-rows", UNWINDOW_ADD_ROWS, [buffer(windows), &d, &d, &d, &d, &d, buffer(acc), &drw], &[h as u32, w as u32, c as u32, win as u32, shift as u32, hp as u32, wp as u32], grid(n.div_ceil(256)));
    }

    fn window_attention(&mut self, qkv: &DeviceVec, table: &DeviceVec, out: &DeviceVec, h: usize, w: usize, heads: usize, win: usize, shift: usize, scale: f32) {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        let (n, c) = (win * win, heads * 32);
        assert!(n <= 256 && shift < win && qkv.len >= hp * wp * 3 * c && table.len >= (2 * win - 1) * (2 * win - 1) * heads && out.len >= hp * wp * c, "chain: window attention of {h}x{w} tokens, {heads} heads");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        // (a pipeline a window's size: its threads, a query each)
        let name: &'static str = match n {
            144 => "chain-window-attention-144",
            49 => "chain-window-attention-49",
            16 => "chain-window-attention-16",
            _ => panic!("chain: window attention of {win}x{win} windows"),
        };
        let body = WINDOW_ATTENTION.replace("N_u", &format!("{n}u"));
        self.dispatch_wide(name, &body, [buffer(qkv), buffer(table), &d, &d, &d, &d, buffer(out), &drw], &[hp as u32, wp as u32, heads as u32, win as u32, shift as u32, scale.to_bits()], (((hp / win) * (wp / win)) as u32, heads as u32, 1));
        self.weigh(4.0 * (hp * wp * n) as f64 * c as f64);
    }

    fn resize_bilinear_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, oh: usize, ow: usize) {
        assert!(h > 0 && w > 0 && x.len >= h * w * c && out.len >= oh * ow * c, "chain: a resize of {h}x{w} to {oh}x{ow}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (oh * ow * c) as u32;
        self.dispatch_wide("chain-resize-bilinear-rows", RESIZE_BILINEAR_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[h as u32, w as u32, c as u32, oh as u32, ow as u32], grid(n.div_ceil(256)));
    }

    fn blocks_to_channels_rows(&mut self, x: &DeviceVec, out: &DeviceVec, s: usize, c: usize, size: usize) {
        assert!(size > 0 && s % size == 0 && x.len >= s * s * c && out.len >= s * s * c, "chain: patches of {size} of a {s}-pixel picture");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (s * s * c) as u32;
        self.dispatch_wide("chain-blocks-to-channels-rows", BLOCKS_TO_CHANNELS_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[s as u32, c as u32, size as u32], grid(n.div_ceil(256)));
    }

    fn deform_im2col_rows(&mut self, x: &DeviceVec, offsets: &DeviceVec, modulators: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize, k: usize, first: usize, pixels: usize) {
        let kk = k * k;
        assert!(k % 2 == 1 && first + pixels <= h * w && x.len >= h * w * c && offsets.len >= h * w * 2 * kk && modulators.len >= h * w * kk && out.len >= pixels * kk * c, "chain: a deformable convolution's taps of {pixels} pixels");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (pixels * kk * c) as u32;
        self.dispatch_wide("chain-deform-im2col-rows", DEFORM_IM2COL_ROWS, [buffer(x), buffer(offsets), buffer(modulators), &d, &d, &d, buffer(out), &drw], &[h as u32, w as u32, c as u32, k as u32, first as u32, pixels as u32], grid(n.div_ceil(256)));
    }

    fn conv1d_padded_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, dilation: usize, pad: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        let cp = cin.div_ceil(32) * 32;
        assert!(dilation > 0 && len > 0 && pad <= (k - 1) * dilation && w.len * 2 >= cout * k * cp && b.len >= cout && x.len >= len * cin && y.len >= len * cout, "chain: a 1-D convolution of {len} steps, {cin} channels to {cout}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let tiles = len.div_ceil(64);
        let per_tile = 2.0 * 64.0 * (cout * cin * k) as f64;
        let chunk = ((CONV_DISPATCH_FLOPS / per_tile) as usize).clamp(1, 65535);
        let mut first = 0;
        while first < tiles {
            let n = chunk.min(tiles - first);
            let words = [cout as u32, cin as u32, len as u32, k as u32, dilation as u32, first as u32, pad as u32];
            self.dispatch_wide("chain-conv1d-f32-tiled", CONV1D_F32_TILED, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &words, ((cout as u32).div_ceil(64), n as u32, 1));
            self.weigh(per_tile * n as f64);
            first += n;
        }
    }

    fn conv_transpose1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, k: usize, stride: usize, pad: usize, x: &DeviceVec, len: usize, out: usize, y: &DeviceVec) {
        assert!(out + pad <= (len - 1) * stride + k + stride && cin % 4 == 0 && stride > 0 && w.len >= cout * k * cin && b.len >= cout && x.len >= len * cin && y.len >= out * cout, "chain: a transposed 1-D convolution of {len} steps, {cin} channels to {cout}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (out * cout) as u32;
        self.dispatch_wide("chain-conv-transpose1d-rows", CONV_TRANSPOSE1D_ROWS, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &[cout as u32, cin as u32, len as u32, k as u32, stride as u32, pad as u32, out as u32], grid(n.div_ceil(256)));
        self.weigh(2.0 * (out * cout) as f64 * (cin * k.div_ceil(stride)) as f64);
    }

    fn gather_rows(&mut self, x: &DeviceVec, index: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, first: usize, src_rows: usize) {
        assert!(index.len >= first + rows && out.len >= rows * c && x.len >= src_rows * c && rows * c < 1 << 32, "chain: a gather of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-gather-rows", GATHER_ROWS, [buffer(x), buffer(index), &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, c as u32, src_rows as u32, first as u32], grid(n.div_ceil(256)));
    }

    fn repeat_cols_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, repeat: usize) {
        assert!(x.len >= rows * c && out.len >= rows * c * repeat && repeat > 0, "chain: {rows} rows of {c} repeated {repeat} times");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c * repeat) as u32;
        self.dispatch_wide("chain-repeat-cols-add-rows", REPEAT_COLS_ADD_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, c as u32, repeat as u32], grid(n.div_ceil(256)));
    }

    fn depthwise_causal_conv1d_rows(&mut self, w: &DeviceVec, b: &DeviceVec, c: usize, k: usize, x: &DeviceVec, len: usize, y: &DeviceVec) {
        assert!(w.len >= c * k && b.len >= c && x.len >= len * c && y.len >= len * c, "chain: a depthwise convolution of {len} steps of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (len * c) as u32;
        self.dispatch_wide("chain-depthwise-causal-conv1d-rows", DEPTHWISE_CAUSAL_CONV1D_ROWS, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &[c as u32, k as u32, len as u32], grid(n.div_ceil(256)));
    }

    fn snake_beta_rows(&mut self, x: &DeviceVec, freq: &DeviceVec, scale: &DeviceVec, rows: usize, c: usize) {
        assert!(x.len >= rows * c && freq.len >= c && scale.len >= c, "chain: SnakeBeta of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-snake-beta-rows", SNAKE_BETA_ROWS, [buffer(freq), buffer(scale), &d, &d, &d, &d, buffer(x), &drw], &[rows as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn clamp_in_place(&mut self, x: &DeviceVec, len: usize, lo: f32, hi: f32) {
        assert!(x.len >= len, "chain: a clamp of {len}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = len as u32;
        self.dispatch_wide("chain-clamp-in-place", CLAMP_IN_PLACE, [&d, &d, &d, &d, &d, &d, buffer(x), &drw], &[n, lo.to_bits(), hi.to_bits()], grid(n.div_ceil(256)));
    }

    fn snake_rows(&mut self, x: &DeviceVec, alpha: &DeviceVec, rows: usize, c: usize) {
        assert!(x.len >= rows * c && alpha.len >= c, "chain: Snake of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-snake-rows", SNAKE_ROWS, [buffer(alpha), &d, &d, &d, &d, &d, buffer(x), &drw], &[rows as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn tanh_in_place(&mut self, x: &DeviceVec, len: usize) {
        assert!(x.len >= len, "chain: tanh of {len}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = len as u32;
        self.dispatch_wide("chain-tanh-in-place", TANH_IN_PLACE, [&d, &d, &d, &d, &d, &d, buffer(x), &drw], &[n], grid(n.div_ceil(256)));
    }

    fn mul_sigmoid_rows(&mut self, x: &DeviceVec, gate: &DeviceVec, rows: usize, c: usize) {
        assert!(x.len >= rows * c && gate.len >= rows, "chain: a gate of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-mul-sigmoid-rows", MUL_SIGMOID_ROWS, [buffer(gate), &d, &d, &d, &d, &d, buffer(x), &drw], &[rows as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn mean_rows(&mut self, x: &DeviceVec, out: &DeviceVec, rows: usize, c: usize) {
        assert!(rows > 0 && x.len >= rows * c && out.len >= c, "chain: a mean of {rows} rows of {c}");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        self.dispatch_wide("chain-mean-rows", MEAN_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, c as u32], ((c as u32).div_ceil(64), 1, 1));
    }

    fn broadcast_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, c: usize, stride: usize, at: usize) {
        assert!(src.len >= c && at + c <= stride && dst.len >= rows * stride, "chain: a broadcast of {c} into {rows} rows");
        let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
        let n = (rows * c) as u32;
        self.dispatch_wide("chain-broadcast-rows", BROADCAST_ROWS, [buffer(src), &d, &d, &d, &d, &d, buffer(dst), &drw], &[rows as u32, c as u32, stride as u32, at as u32], grid(n.div_ceil(256)));
    }

    fn leaky_relu(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize, slope: f32) {
        assert!(x.len >= len && out.len >= len, "chain: a leaky ReLU of {len}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = len as u32;
        if Arc::ptr_eq(&x.inner, &out.inner) {
            // (in place: the one buffer bound once, written where it is read)
            let body = LEAKY_RELU.replace("@group(0) @binding(0) var<storage, read> x: array<f32>;
", "").replace("let v = x[i];", "let v = out[i];");
            self.dispatch_wide("chain-leaky-relu-in-place", &body, [&d, &d, &d, &d, &d, &d, buffer(out), &drw], &[n, slope.to_bits()], grid(n.div_ceil(256)));
        } else {
            self.dispatch_wide("chain-leaky-relu", LEAKY_RELU, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[n, slope.to_bits()], grid(n.div_ceil(256)));
        }
    }

    fn matmul_nvfp4_rows(&mut self, w: &DeviceVec, scale: &DeviceVec, b: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(k % 64 == 0 && w.len >= n * (k / 8 + k / 64) && scale.len >= 2 && b.len >= n && x.len >= rows * k && y.len >= rows * n && rows.div_ceil(crate::shaders::COOP_TILE as usize) <= 65535, "chain: an NVFP4 matmul [{n}, {k}] of {rows} rows");
        let x16 = self.x16_tiled(x, rows, k);
        let tile = crate::shaders::COOP_TILE;
        let pipeline = self.gpu().named_pipeline("chain-coop-nvfp4", crate::shaders::coop_tiled_nvfp4);
        let tiles = (n as u32).div_ceil(tile) * (rows as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, k, rows, n, y, None);
        let words = [k as u32, n as u32, rows as u32, 0, n as u32, (k / 8 + k / 64) as u32, splits, 0];
        self.dispatch_kept(&pipeline, buffer(w), buffer(&x16), &out, &words, ((n as u32).div_ceil(tile), (rows as u32).div_ceil(tile), splits));
        self.coop_sum(parts, rows, n, y, splits);
        // the tensor's own scale, and the bias
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-unscale-bias-rows", UNSCALE_BIAS_ROWS, [buffer(b), buffer(scale), &d, &d, &d, &d, buffer(y), &drw], &[n as u32, rows as u32], grid(((rows * n) as u32).div_ceil(256)));
        self.weigh(2.0 * (rows * n) as f64 * k as f64);
    }

    fn add_bias_rows(&mut self, y: &DeviceVec, b: &DeviceVec, rows: usize, n: usize) {
        assert!(y.len >= rows * n && b.len >= n, "chain: a bias on {rows} rows of {n}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-add-bias-rows", ADD_BIAS_ROWS, [buffer(b), &d, &d, &d, &d, &d, buffer(y), &drw], &[n as u32, rows as u32], grid(((rows * n) as u32).div_ceil(256)));
    }

    fn upsample2x_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        assert!(x.len >= h * w * c && out.len >= 4 * h * w * c, "chain: upsampling {h}x{w} pixels of {c}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-upsample2x-rows", UPSAMPLE2X_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[c as u32, h as u32, w as u32], grid(((4 * h * w * c) as u32).div_ceil(256)));
    }

    fn shuffle_up_add_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, cin: usize, cout: usize, ft: usize) {
        let repeats = cout * ft * 4 / cin.max(1);
        assert!(ft > 0 && repeats > 0 && repeats * cin == cout * ft * 4 && x.len >= h * w * cin && out.len >= 4 * h * w * cout, "chain: an upsampling shortcut of {cin} channels to {cout}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-shuffle-up-add-rows", SHUFFLE_UP_ADD_ROWS, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[cin as u32, cout as u32, ft as u32, repeats as u32, h as u32, w as u32], grid(((4 * h * w * cout) as u32).div_ceil(256)));
    }

    fn gelu_erf(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && out.len >= len, "chain: an exact GELU of {len}");
        let pipeline = self.named("chain-gelu-erf", GELU_ERF);
        self.dispatch_kept(&pipeline, buffer(x), buffer(x), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn gelu(&mut self, x: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && out.len >= len, "chain: a GELU of {len}");
        let pipeline = self.named("chain-gelu", GELU);
        self.dispatch_kept(&pipeline, buffer(x), buffer(x), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn attention_rows_full(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, kv_len: usize, scale: f32) {
        assert!(rows > 0 && kv_len > 0, "chain: a full attention of {rows} queries over {kv_len} positions");
        // (a full attention's `past` is its positions: no query's own place among them)
        if !self.attention_rows_coop_masked(q, kv, out, rows, n_h, n_kv, head_dim, kv_len, None, scale, true) {
            self.attention_rows_f32_masked(q, kv, out, rows, n_h, n_kv, head_dim, kv_len, None, scale, true);
        }
    }

    fn rope_split_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec) {
        assert!(x.len >= rows * heads * head_dim && table.len >= rows * heads * head_dim, "chain: a split RoPE of {rows} rows");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * head_dim / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, 1, rows as u32, 0, 1], grid(pairs.div_ceil(256)));
    }

    fn group_norm_rows(&mut self, x: &DeviceVec, weight: &DeviceVec, bias: &DeviceVec, out: &DeviceVec, stats: &DeviceVec, pixels: usize, c: usize, groups: usize, eps: f32, silu: bool) {
        let chunks = pixels.div_ceil(256);
        assert!(
            groups > 0 && c % groups == 0 && pixels > 0 && chunks <= 65535 && x.len >= pixels * c && out.len >= pixels * c && weight.len >= c && bias.len >= c && stats.len >= groups * (chunks + 1) * 2,
            "chain: a group norm of {pixels} pixels of {c} in {groups} groups"
        );
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let words = [pixels as u32, c as u32, groups as u32, chunks as u32, eps.to_bits(), silu as u32];
        self.dispatch_wide("chain-group-norm-sums", GROUP_NORM_SUMS, [buffer(x), &d, &d, &d, &d, &d, buffer(stats), &drw], &words, (groups as u32, chunks as u32, 1));
        self.dispatch_wide("chain-group-norm-stats", GROUP_NORM_STATS, [buffer(x), &d, &d, &d, &d, &d, buffer(stats), &drw], &words, (groups as u32, 1, 1));
        self.dispatch_wide("chain-group-norm-apply", GROUP_NORM_APPLY, [buffer(x), buffer(weight), buffer(bias), buffer(stats), &d, &d, buffer(out), &drw], &words, grid(((pixels * c) as u32).div_ceil(256)));
    }

    fn geglu_rows(&mut self, fused: &DeviceVec, out: &DeviceVec, rows: usize, ff: usize) {
        assert!(fused.len >= rows * 2 * ff && out.len >= rows * ff, "chain: GEGLU of {rows} rows of {ff}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-geglu-rows", GEGLU_ROWS, [buffer(fused), &d, &d, &d, &d, &d, buffer(out), &drw], &[rows as u32, ff as u32], grid(((rows * ff) as u32).div_ceil(256)));
    }

    fn subsample2x_even_rows(&mut self, x: &DeviceVec, out: &DeviceVec, h: usize, w: usize, c: usize) {
        assert!(h % 2 == 0 && w % 2 == 0 && x.len >= h * w * c && out.len >= h * w * c / 4, "chain: subsampling {h}x{w} pixels of {c}");
        let dm = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let n = (h * w * c / 4) as u32;
        self.dispatch_wide("chain-subsample2x-even-rows", SUBSAMPLE2X_EVEN_ROWS, [buffer(x), &dm, &dm, &dm, &dm, &dm, buffer(out), &drw], &[h as u32, w as u32, c as u32], grid(n.div_ceil(256)));
    }

    fn nag_mix(&mut self, pos: &DeviceVec, neg: &DeviceVec, rows: usize, width: usize, scale: f32, tau: f32, alpha: f32) {
        assert!(width % 4 == 0 && pos.len >= rows * width && neg.len >= rows * width, "chain: a guidance mix of {rows} rows of {width}");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let rows32 = rows as u32;
        self.dispatch_wide("chain-nag-mix", NAG_MIX, [buffer(neg), &d, &d, &d, &d, &d, buffer(pos), &drw], &[width as u32, rows32, scale.to_bits(), tau.to_bits(), alpha.to_bits()], (rows32.min(65535), rows32.div_ceil(65535), 1));
    }

    fn head_gate_rows(&mut self, y: &DeviceVec, logits: &DeviceVec, rows: usize, heads: usize, head_dim: usize) {
        assert!(y.len >= rows * heads * head_dim && logits.len >= rows * heads, "chain: a head gate of {rows} rows");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let len = (rows * heads * head_dim) as u32;
        self.dispatch_wide("chain-head-gate-rows", HEAD_GATE_ROWS, [buffer(logits), &d, &d, &d, &d, &d, buffer(y), &drw], &[head_dim as u32, len], grid(len.div_ceil(256)));
    }

    fn mul_sigmoid(&mut self, x: &DeviceVec, gate: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(x.len >= len && gate.len >= len && out.len >= len && !Arc::ptr_eq(&x.inner, &out.inner), "chain: a gate of {len}");
        let pipeline = self.named("chain-mul-sigmoid", MUL_SIGMOID);
        self.dispatch_kept(&pipeline, buffer(x), buffer(gate), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn silu_mul(&mut self, gate: &DeviceVec, up: &DeviceVec, out: &DeviceVec, len: usize) {
        assert!(gate.len >= len && up.len >= len && out.len >= len, "chain: a SwiGLU of {len}");
        let pipeline = self.named("chain-silu-mul", SILU_MUL);
        self.dispatch_kept(&pipeline, buffer(gate), buffer(up), buffer(out), &[len as u32], grid((len as u32).div_ceil(256)));
    }

    fn ple_gate(&mut self, key: &DeviceVec, x: &DeviceVec, value: &DeviceVec, norm_key: &DeviceVec, norm_query: &DeviceVec, norm_conv: &DeviceVec, gated: &DeviceVec, conv_in: &DeviceVec, rows: usize, streams: usize, d: usize, eps: f32) {
        let width = streams * d;
        assert!(key.len >= rows * width && x.len >= rows * width && value.len >= rows * d && norm_key.len >= width && norm_query.len >= width && norm_conv.len >= width && gated.len >= rows * width && conv_in.len >= rows * width, "chain: an n-gram gate of {rows} rows of {streams} streams of {d}");
        let body = format!("{}{PLE_GATE}", crate::exl3::HALF);
        self.dispatch_wide("chain-ple-gate", &body, [buffer(key), buffer(x), buffer(value), buffer(norm_key), buffer(norm_query), buffer(norm_conv), buffer(gated), buffer(conv_in)], &[d as u32, streams as u32, eps.to_bits()], ((rows * streams) as u32, 1, 1));
    }

    fn ple_conv(&mut self, x: &DeviceVec, gated: &DeviceVec, conv_in: &DeviceVec, window: &DeviceVec, weight: &DeviceVec, rows: usize, width: usize, kernel: usize, dilation: usize) {
        assert!(kernel >= 1 && x.len >= rows * width && gated.len >= rows * width && conv_in.len >= rows * width && window.len >= (kernel - 1) * dilation * width && weight.len >= width * kernel, "chain: an n-gram conv of {rows} rows of {width}");
        let d = self.gpu().dummy().clone();
        self.dispatch_wide("chain-ple-conv", PLE_CONV, [buffer(gated), buffer(conv_in), buffer(weight), &d, &d, &d, buffer(x), buffer(window)], &[width as u32, rows as u32, kernel as u32, dilation as u32], ((width as u32).div_ceil(256), 1, 1));
    }

    fn matmul_f16_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(k % 2 == 0 && w.len * 2 >= n * k && x.len >= rows * k && y.len >= rows * n, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
        // (the tiled kernels take a grid's 65,535 tiles of rows: 64 rows each, the tensor cores' 32)
        assert!(n <= 65535 && rows.div_ceil(64) <= 65535, "chain: an f16 matmul [{n}, {k}] of {rows} rows");
        // a step's row or a check's few against a width of whole fours: an output row's lanes side by side
        // (OAIY_F16_FIRST: the first kernels, a workgroup an output row or eight threads an output)
        static LANES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if rows <= 8 && k % 4 == 0 && *LANES.get_or_init(|| std::env::var_os("OAIY_F16_FIRST").is_none()) {
            type Made = ((usize, usize), &'static str);
            static NAMES: Mutex<Vec<Made>> = Mutex::new(Vec::new());
            let lanes = f16_lanes(k);
            let name = {
                let mut names = NAMES.lock().unwrap_or_else(|p| p.into_inner());
                match names.iter().find(|(key, _)| *key == (rows, lanes)) {
                    Some((_, name)) => *name,
                    None => {
                        let name: &'static str = Box::leak(format!("chain-matvec-f16-{rows}-{lanes}").into_boxed_str());
                        names.push(((rows, lanes), name));
                        name
                    }
                }
            };
            let pipeline = self.gpu().named_pipeline(name, || matvec_f16_lanes(rows, lanes));
            let groups = (n as u32).div_ceil((256 / lanes) as u32);
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (groups.min(65535), groups.div_ceil(65535), 1));
            self.weigh(2.0 * (rows * n) as f64 * k as f64);
            return;
        }
        if rows == 1 {
            // a long row a workgroup (as the f32 one sums it); short ones eight threads each, 32 a workgroup
            if k >= 2048 && k % 4 == 0 {
                let pipeline = self.gpu().named_pipeline("chain-matvec-f16-4", || MATVEC_F16.to_string());
                self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (n as u32, 1, 1));
            } else {
                let pipeline = self.named("chain-matvec-f16-narrow", MATVEC_F16_NARROW);
                self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], ((n as u32).div_ceil(32), 1, 1));
            }
            return;
        }
        // a few rows (a check of drafts): each weight read once for all of them, each row summed as one row is
        if rows <= 8 {
            const NAMES: [[&str; 7]; 2] = [
                ["chain-matvec-f16-rows-2", "chain-matvec-f16-rows-3", "chain-matvec-f16-rows-4", "chain-matvec-f16-rows-5", "chain-matvec-f16-rows-6", "chain-matvec-f16-rows-7", "chain-matvec-f16-rows-8"],
                ["chain-matvec-f16-narrow-2", "chain-matvec-f16-narrow-3", "chain-matvec-f16-narrow-4", "chain-matvec-f16-narrow-5", "chain-matvec-f16-narrow-6", "chain-matvec-f16-narrow-7", "chain-matvec-f16-narrow-8"],
            ];
            let wide = k >= 2048 && k % 4 == 0;
            let pipeline = self.gpu().named_pipeline(NAMES[!wide as usize][rows - 2], || matvec_f16_rows(rows, !wide));
            let groups = if wide { n as u32 } else { (n as u32).div_ceil(32) };
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (groups, 1, 1));
            return;
        }
        // a prompt's rows on the tensor cores (f16 tokens, f32 sums: Qwen3.8-Flash-Next's hyper-connections' 324 by
        // 10,240 and back, its routers' and its delta nets' `ba` some 14 ms of a chunk of 512 where 33; OAIY_NO_COOP_F16
        // the f32 tiled kernel)
        if std::env::var_os("OAIY_NO_COOP_F16").is_some() || !self.matmul_f16_coop(w, n, k, x, y, rows, None) {
            self.matmul_f16_tiled(w, n, k, x, y, rows);
        }
        self.weigh(2.0 * (rows * n) as f64 * k as f64);
    }

    fn matmul_f32_rows(&mut self, w: &DeviceVec, n: usize, k: usize, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        assert!(w.len >= n * k && x.len >= rows * k && y.len >= rows * n && n <= 65535 && rows.div_ceil(64) <= 65535, "chain: an f32 matmul [{n}, {k}] of {rows} rows");
        if rows == 1 {
            let pipeline = if k % 4 == 0 { self.gpu().named_pipeline("chain-matvec-f32-4", || MATVEC_F32_4.to_string()) } else { self.named("chain-matmul-f32", MATMUL_F32) };
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32], (n as u32, 1, 1));
            return;
        }
        // a few rows (a check of drafts): each weight read once for all of them, where the tiled kernel's 64-row tiles
        // and its split sums cost a prompt's
        if rows <= 8 && k % 4 == 0 {
            let pipeline = self.gpu().named_pipeline("chain-matvec-f32-4-rows", || MATVEC_F32_4_ROWS.to_string());
            self.dispatch_kept(&pipeline, buffer(w), buffer(x), buffer(y), &[n as u32, k as u32, rows as u32], (n as u32, 1, 1));
            return;
        }
        // a prompt's rows in 64x64 tiles; few tiles (few outputs) split k for a second pass to add up, enough
        // workgroups to fill the GPU, 256 of k a split at least
        let tiles = n.div_ceil(64) * rows.div_ceil(64);
        let want = 1024usize.div_ceil(tiles).min(k / 256).max(1);
        let kc = k.div_ceil(want).div_ceil(16) * 16;
        let splits = k.div_ceil(kc);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let grid = (n.div_ceil(64) as u32, rows.div_ceil(64) as u32, splits as u32);
        let words = [n as u32, k as u32, rows as u32, kc as u32];
        if splits == 1 {
            self.dispatch_wide("chain-matmul-f32-tiled", MATMUL_F32_TILED, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(y), &drw], &words, grid);
        } else {
            let part = self.scratch(splits * rows * n);
            self.dispatch_wide("chain-matmul-f32-tiled", MATMUL_F32_TILED, [buffer(w), buffer(x), &d, &d, &d, &d, buffer(&part), &drw], &words, grid);
            let len = (rows * n) as u32;
            self.dispatch_wide("chain-sum-splits", SUM_SPLITS, [buffer(&part), &d, &d, &d, &d, &d, buffer(y), &drw], &[len, splits as u32], (len.div_ceil(256).min(65535), len.div_ceil(256 * 65535), 1));
            // (its parts read: spare for the next split's, where each had its own to the recording's end, a sound
            // step's 360 of 18 MB without tensor cores)
            self.spare.push((buffer(&part).size(), buffer(&part).clone()));
        }
        self.weigh(2.0 * (rows * n) as f64 * k as f64);
    }

    fn ssm_conv(&mut self, qkv: &DeviceVec, weight: &DeviceVec, state: &DeviceVec, out: &DeviceVec, rows: usize, channels: usize, kernel: usize) {
        assert!((2..=8).contains(&kernel) && qkv.len >= rows * channels && weight.len >= channels * kernel && state.len >= (kernel - 1) * channels && out.len >= rows * channels, "chain: a conv of {kernel} over {rows} rows of {channels}");
        let q = buffer(qkv);
        let words = [channels as u32, rows as u32, kernel as u32];
        let groups = (channels as u32).div_ceil(256);
        // a prompt's a thread a (channel, token) where the thread a channel walked its tokens (0.26 ms of a 27B's
        // layer for 512), then the state; a step's and a check's as they were
        if rows >= 16 && rows <= 65535 {
            let (d, drw) = (self.gpu().dummy().clone(), self.gpu().dummy_rw().clone());
            self.dispatch_wide("chain-ssm-conv-rows", SSM_CONV_ROWS, [q, buffer(weight), buffer(state), &d, &d, &d, &drw, buffer(out)], &words, (groups, rows as u32, 1));
            self.dispatch_wide("chain-ssm-conv-state", SSM_CONV_STATE, [q, &d, &d, &d, &d, &d, buffer(state), &drw], &words, (groups, 1, 1));
            return;
        }
        self.dispatch_wide("chain-ssm-conv", SSM_CONV, [q, buffer(weight), q, q, q, q, buffer(state), buffer(out)], &words, (groups, 1, 1));
    }

    fn delta_net(&mut self, conv: &DeviceVec, z: &DeviceVec, beta_alpha: &DeviceVec, ssm_a: &DeviceVec, dt_bias: &DeviceVec, norm: &DeviceVec, state: &DeviceVec, out: &DeviceVec, d: DeltaNet) {
        let ch = 2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim;
        // a pipeline a head size, the size a constant of it
        let name: &'static str = match d.k_dim {
            16 => "chain-delta-net-16",
            32 => "chain-delta-net-32",
            64 => "chain-delta-net-64",
            128 => "chain-delta-net-128",
            other => panic!("chain: a delta net of heads of {other} (16, 32, 64 or 128)"),
        };
        assert!(d.k_dim == d.v_dim && d.k_heads > 0, "chain: a delta net of heads of {} and {} (one size)", d.k_dim, d.v_dim);
        assert!(
            conv.len >= d.rows * ch && z.len >= d.rows * d.v_heads * d.v_dim && beta_alpha.len >= d.rows * 2 * d.v_heads && ssm_a.len >= d.v_heads && dt_bias.len >= d.v_heads
                && norm.len >= d.v_dim && state.len >= d.v_heads * d.k_dim * d.v_dim && out.len >= d.rows * d.v_heads * d.v_dim,
            "chain: a delta net's buffers"
        );
        let words = [d.v_heads as u32, d.k_heads as u32, d.k_dim as u32, d.v_dim as u32, d.rows as u32, d.scale_q.to_bits(), d.eps.to_bits(), d.sigmoid_gate as u32];
        // a prompt's in three passes (the recurrence's alone in turn); a step's and a check's few rows the one kernel
        // (a check's rows a step's bit for bit)
        if d.rows >= 16 && d.k_dim >= 32 && d.v_heads % d.k_heads == 0 {
            self.delta_net_rows(conv, z, beta_alpha, ssm_a, dt_bias, norm, state, out, &d, &words);
            return;
        }
        self.dispatch_wide(name, &delta_net_one(d.k_dim), [buffer(conv), buffer(z), buffer(beta_alpha), buffer(ssm_a), buffer(dt_bias), buffer(norm), buffer(state), buffer(out)], &words, (d.v_heads as u32, 1, 1));
    }

    fn rope_rows(&mut self, x: &DeviceVec, rows: usize, heads: usize, head_dim: usize, table: &DeviceVec, neox: bool) {
        assert!(x.len >= rows * heads * head_dim && table.len >= rows * head_dim, "chain: RoPE of {rows} rows");
        let pipeline = self.named("chain-rope", ROPE);
        let pairs = (rows * heads * head_dim / 2) as u32;
        self.dispatch_kept(&pipeline, buffer(table), buffer(table), buffer(x), &[heads as u32, head_dim as u32, neox as u32, rows as u32], grid(pairs.div_ceil(256)));
    }

    fn store_rows(&mut self, src: &DeviceVec, dst: &DeviceVec, rows: usize, len: usize, start: usize, stride: usize, at: usize) {
        assert!(src.len >= rows * len && at + len <= stride && dst.len >= (start + rows) * stride, "chain: storing {rows} rows");
        let pipeline = self.named("chain-store-rows", STORE_ROWS);
        let params = self.uniform(&[len as u32, start as u32, stride as u32, at as u32, rows as u32]);
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, grid(((rows * len) as u32).div_ceil(256)));
    }

    fn attention_rows(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, past: usize, window: Option<usize>, scale: f32) {
        if !self.attention_rows_coop(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale) {
            self.attention_rows_f32(q, kv, out, rows, n_h, n_kv, head_dim, past, window, scale);
        }
    }

    fn copy(&mut self, src: &DeviceVec, src_at: usize, dst: &DeviceVec, dst_at: usize, len: usize) {
        assert!(src_at + len <= src.len && dst_at + len <= dst.len, "chain: copying {len} from {src_at} of {} to {dst_at} of {}", src.len, dst.len);
        let pipeline = self.named("chain-copy", COPY);
        let params = self.uniform(&[len as u32, dst_at as u32, src_at as u32]);
        // (past 16.8 million values the grid's second dimension: a 1024x1024 reference image's tokens in a prefix)
        self.dispatch(&pipeline, buffer(src), buffer(src), buffer(dst), &params, grid((len as u32).div_ceil(256)));
    }

    fn attention(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        // (OAIY_ATTENTION_PART4: a workgroup a query head, as before the groups' kernel)
        static HEADS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let group = !*HEADS.get_or_init(|| std::env::var_os("OAIY_ATTENTION_PART4").is_some());
        self.attention_by(q, kv, out, n_h, n_kv, head_dim, lo, kv_len, cap, scale, group);
    }

    fn halve(&mut self, src: &DeviceVec, dst: &DeviceVec, at: usize, len: usize) {
        assert!(at % 2 == 0 && len % 2 == 0 && at + len <= src.len && (at + len) / 2 <= dst.len, "chain: halves of {len} values at {at} of {} into {}", src.len, dst.len);
        if len == 0 {
            return;
        }
        let pipeline = self.gpu().named_pipeline("chain-halve", || HALVE.to_string());
        let params = self.uniform(&[(at / 2) as u32, (len / 2) as u32]);
        let d = self.gpu().dummy().clone();
        let groups = ((len / 2) as u32).div_ceil(256);
        self.dispatch(&pipeline, &d, buffer(src), buffer(dst), &params, (groups.min(65535), groups.div_ceil(65535), 1));
    }

    fn attention_halved(&mut self, q: &DeviceVec, kv: &DeviceVec, out: &DeviceVec, n_h: usize, n_kv: usize, head_dim: usize, lo: usize, kv_len: usize, cap: usize, scale: f32) {
        let runs = kv_len.saturating_sub(lo).div_ceil(SPLIT).max(1);
        assert!(
            self.backend.attention_halves(n_h, n_kv, head_dim) && kv.len >= cap * n_kv * head_dim && kv_len <= cap && out.len >= n_h * head_dim + n_h * runs * (head_dim + 2),
            "chain: attention's buffers"
        );
        let params = self.uniform(&[n_h as u32, n_kv as u32, head_dim as u32, kv_len as u32, lo as u32, runs as u32, scale.to_bits()]);
        const NAMES: [&str; 9] = ["", "", "chain-attention-halves-2", "chain-attention-halves-3", "chain-attention-halves-4", "chain-attention-halves-5", "chain-attention-halves-6", "chain-attention-halves-7", "chain-attention-halves-8"];
        let g = n_h / n_kv;
        let part = self.gpu().named_pipeline(NAMES[g], || attention_part_group_halved(g));
        self.dispatch(&part, buffer(kv), buffer(q), buffer(out), &params, (n_kv as u32, runs as u32, 1));
        let join = self.named("chain-attention-join", ATTENTION_JOIN);
        self.dispatch(&join, buffer(kv), buffer(q), buffer(out), &params, (n_h as u32, 1, 1));
    }

    fn qsa_pool(&mut self, raw: &DeviceVec, pooled: &DeviceVec, blocks: usize, ratio: usize, d: usize) {
        assert!(d <= 256 && raw.len >= blocks * ratio * d && pooled.len >= blocks * d, "chain: QSA's pool of {blocks} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let b = blocks as u32;
        self.dispatch_wide("chain-qsa-pool", QSA_POOL, [buffer(raw), &dd, &dd, &dd, &dd, &dd, buffer(pooled), &drw], &[b, ratio as u32, d as u32], (b.min(65535), b.div_ceil(65535), 1));
    }

    fn qsa_scores(&mut self, q: &DeviceVec, pooled: &DeviceVec, scores: &DeviceVec, rows: usize, heads: usize, d: usize, nb: usize, first: usize, ratio: usize, scale: f32) {
        assert!(heads * d <= 2048 && q.len >= rows * heads * d && pooled.len >= nb * d && scores.len >= rows * nb, "chain: QSA's scores of {rows} rows over {nb} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        // a head a multiple of 4 wide, 8 heads at most: each pooled key read once for them all, a vec4 at a time
        if d % 4 == 0 && (1..=8).contains(&heads) && std::env::var_os("OAIY_QSA_SCORES_F1").is_none() {
            const NAMES: [&str; 9] = ["", "chain-qsa-scores4-1", "chain-qsa-scores4-2", "chain-qsa-scores4-3", "chain-qsa-scores4-4", "chain-qsa-scores4-5", "chain-qsa-scores4-6", "chain-qsa-scores4-7", "chain-qsa-scores4-8"];
            let src = qsa_scores4(heads);
            self.dispatch_wide(NAMES[heads], &src, [buffer(q), buffer(pooled), &dd, &dd, &dd, &dd, buffer(scores), &drw], &[rows as u32, heads as u32, d as u32, nb as u32, first as u32, ratio as u32, scale.to_bits()], ((nb as u32).div_ceil(256), rows as u32, 1));
            return;
        }
        self.dispatch_wide("chain-qsa-scores", QSA_SCORES, [buffer(q), buffer(pooled), &dd, &dd, &dd, &dd, buffer(scores), &drw], &[rows as u32, heads as u32, d as u32, nb as u32, first as u32, ratio as u32, scale.to_bits()], ((nb as u32).div_ceil(256), rows as u32, 1));
    }

    fn qsa_select(&mut self, scores: &DeviceVec, list: &DeviceVec, rows: usize, nb: usize, first: usize, ratio: usize, keep: usize) {
        assert!(nb <= 4096 && keep > 0 && scores.len >= rows * nb && list.len >= rows * keep, "chain: QSA's selection of {keep} of {nb} blocks");
        let dd = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-qsa-select", QSA_SELECT, [buffer(scores), &dd, &dd, &dd, &dd, &dd, buffer(list), &drw], &[rows as u32, nb as u32, first as u32, ratio as u32, keep as u32], (rows as u32, 1, 1));
    }

    fn qsa_attention(&mut self, q: &DeviceVec, kv: &DeviceVec, list: &DeviceVec, out: &DeviceVec, rows: usize, n_h: usize, n_kv: usize, head_dim: usize, first: usize, ratio: usize, keep: usize, scale: f32) {
        // a prompt's rows on the tensor cores where the device has them (a check's few as a step's)
        if rows > 8 && self.qsa_attention_coop(q, kv, list, out, rows, n_h, n_kv, head_dim, first, ratio, keep, scale) {
            return;
        }
        self.qsa_attention_f32(q, kv, list, out, rows, n_h, n_kv, head_dim, first, ratio, keep, scale);
    }

    fn argmax_softmax(&mut self, x: &DeviceVec, out: &DeviceVec) {
        assert!(x.len > 0 && out.len >= 3, "chain: a draft's token of {} logits", x.len);
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        self.dispatch_wide("chain-argmax-softmax", ARGMAX_SOFTMAX, [buffer(x), &d, &d, &d, &d, &d, buffer(out), &drw], &[x.len as u32], (1, 1, 1));
    }

    fn read_range(&mut self, v: &DeviceVec, offset: usize, len: usize) {
        assert!(offset + len <= v.len, "chain: reading {len} at {offset} of {}", v.len);
        let staging = self.gpu().staging(((len.max(1) * 4) as u64).next_power_of_two().max(256));
        self.reads.push((buffer(v).clone(), offset, staging, len));
    }

    fn finish(mut self: Box<Self>) -> Vec<Vec<f32>> {
        // (profiled: what is left a piece of its own, timed as the others)
        let profiled = crate::profile::chain_on();
        // (by the feed: what is left a piece of its own too, encoded at its turn as the others)
        if profiled || self.fed() {
            self.submit_piece();
        }
        let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
        let start = std::time::Instant::now();
        let mut enc = self.gpu().device.create_command_encoder(&Default::default());
        if !self.dispatches.is_empty() {
            // what the pieces submitted while recording left (`push`), with the reads
            let i = self.next_stamp();
            let timestamp_writes = self.stamp_writes(i);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
            for (pipeline, group, (x, y, z)) in &self.dispatches {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, *z);
            }
        }
        // (the pieces since a flush, if any, resolved with this; else the flush's)
        let late = self.resolve_pieces(&mut enc);
        let pieces = late.clone().or_else(|| self.stamped.take());
        for (from, offset, staging, len) in &self.reads[self.copied..] {
            if *len > 0 {
                enc.copy_buffer_to_buffer(from, (*offset * 4) as u64, staging, 0, (*len * 4) as u64);
            }
        }
        // nothing recorded since a flush: its submission the one waited for (profiled: this one, after every timed piece)
        let since = !self.dispatches.is_empty() || !self.held.is_empty() || !self.lists.is_empty() || self.copied < self.reads.len() || profiled || late.is_some();
        let command = enc.finish();
        crate::profile::add(&crate::profile::CHAIN_ENCODE, start);
        let submitted = std::time::Instant::now();
        let last = match self.flushed.take() {
            Some(piece) if !since => piece,
            _ => {
                // (a held recording's pieces first, in turn)
                let held = std::mem::take(&mut self.held);
                if !held.is_empty() {
                    self.gpu().submit_piece(held);
                }
                for list in std::mem::take(&mut self.lists) {
                    self.gpu().feed(list);
                }
                self.gpu().submit_after(vec![command])
            }
        };
        // (gone to the queue before its staging buffers are mapped: a submission may not copy into a mapped one)
        let index = self.gpu().gone(last);
        for (_, _, staging, len) in &self.reads {
            staging.slice(..(*len as u64 * 4).max(4)).map_async(wgpu::MapMode::Read, |_| {});
        }
        for (_, staging) in &self.timed {
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        if let Some(staging) = &pieces {
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        // this recording's work waited for, not what was submitted after it (the next chunk's, as this one's rows
        // are read)
        self.gpu().wait(Some(index));
        crate::profile::add(&crate::profile::CHAIN_WAIT, submitted);
        let pooled = std::mem::take(&mut self.pooled);
        self.gpu().unpool(pooled);
        if let Some(staging) = pieces {
            let period = self.gpu().queue().get_timestamp_period() as f64;
            let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping the pieces' times");
            let ticks: Vec<u64> = view.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
            drop(view);
            staging.unmap();
            let busy: u64 = ticks.chunks_exact(2).map(|p| p[1].saturating_sub(p[0])).sum();
            let span = ticks.last().copied().unwrap_or(0).saturating_sub(ticks.first().copied().unwrap_or(0));
            use std::sync::atomic::Ordering as O;
            crate::profile::PIECES[0].fetch_add((busy as f64 * period) as u64, O::Relaxed);
            crate::profile::PIECES[1].fetch_add((span as f64 * period) as u64, O::Relaxed);
            crate::profile::PIECES[2].fetch_add(ticks.len() as u64 / 2, O::Relaxed);
            if std::env::var("OAIY_PIECE_STAMPS").is_ok_and(|v| v == "2") {
                // each piece: its start after the first's, and how long (ms)
                static BASE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let base = *BASE.get_or_init(|| ticks[0]);
                let at = |t: u64| t.saturating_sub(base) as f64 * period / 1e6;
                eprintln!("    a recording's {} pieces, {:.1} ms busy over {:.1}, on the GPU's clock {:.1}..{:.1}", ticks.len() / 2, busy as f64 * period / 1e6, span as f64 * period / 1e6, at(ticks[0]), at(*ticks.last().expect("a piece")));
            }
        }
        for (pipelines, staging) in std::mem::take(&mut self.timed) {
            let period = self.gpu().queue().get_timestamp_period() as f64;
            let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping the profile");
            let ticks: Vec<u64> = view.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
            drop(view);
            staging.unmap();
            let mut k = crate::profile::KERNELS.lock().unwrap_or_else(|p| p.into_inner());
            for (i, pipeline) in pipelines.iter().enumerate() {
                let ns = (ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period) as u64;
                let e = k.entry(self.gpu().name_of(pipeline)).or_default();
                e.0 += ns;
                e.1 += 1;
            }
        }
        // copied as bytes (a 512-row chunk's cache rows, 64 MB: a value at a time some 15 ms), 8 MB or more of them on
        // every core, the host's pages first touched there (a checkpoint's 96 recurrent states, 149 MB: 34 ms on one)
        let copied = |staging: &wgpu::Buffer, len: usize| {
            use rayon::prelude::*;
            let view = staging.slice(..(len as u64 * 4).max(4)).get_mapped_range().expect("webgpu: mapping a finished buffer");
            let mut v = vec![0f32; len];
            let (to, from) = (bytemuck::cast_slice_mut::<f32, u8>(&mut v), &view[..len * 4]);
            if from.len() >= 8 << 20 {
                to.par_chunks_mut(1 << 20).zip(from.par_chunks(1 << 20)).for_each(|(t, f)| t.copy_from_slice(f));
            } else {
                to.copy_from_slice(from);
            }
            drop(view);
            staging.unmap();
            v
        };
        let out = if self.reads.iter().map(|r| r.3 * 4).sum::<usize>() >= 8 << 20 && self.reads.len() > 1 {
            use rayon::prelude::*;
            self.reads.par_iter().map(|(_, _, staging, len)| copied(staging, *len)).collect()
        } else {
            self.reads.iter().map(|(_, _, staging, len)| copied(staging, *len)).collect()
        };
        let staged: Vec<(u64, wgpu::Buffer)> = std::mem::take(&mut self.reads).into_iter().map(|(_, _, staging, _)| (staging.size(), staging)).collect();
        self.gpu().unstage(staged);
        crate::profile::add(&crate::profile::LINEAR_WAIT, start);
        out
    }
}
