//! The recorder's packed projections: an EXL3 one's transforms and matmul, a GGUF's matrix where one is asked for.
use super::*;

impl Recorder<'_> {
    /// [`ChainRecorder::exl3_rows`], the input `x`, or (`up` given) the SwiGLU `silu(x) * up` computed as the input
    /// transform reads it (a shared expert's down projection: a dispatch fewer).
    pub(crate) fn exl3_rows_of(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, x: &DeviceVec, up: Option<&DeviceVec>, y: &DeviceVec, rows: usize) {
        // a GGUF's matrix: the quantized matmul of its rows as they are (no transform either side, no channel map),
        // a SwiGLU's product made first where one is asked for
        if let Some(q) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::QuantLinear>()) {
            let (k, n) = q.kn();
            assert!(rows > 0 && x.len >= rows * k && y.len >= rows * n, "chain: a quantized projection [{n}, {k}] of {rows} rows");
            match up {
                Some(u) => {
                    let t = self.scratch(rows * k);
                    self.silu_mul(x, u, &t, rows * k);
                    self.matmul_rows(&q.w, &t, y, rows);
                }
                None => self.matmul_rows(&q.w, x, y, rows),
            }
            return;
        }
        // (and its matrices of floats, as f16)
        if let Some(f) = w.as_any().and_then(|a| a.downcast_ref::<crate::quant_linear::HalfLinear>()) {
            let (k, n) = f.kn();
            assert!(rows > 0 && x.len >= rows * k && y.len >= rows * n, "chain: an f16 projection [{n}, {k}] of {rows} rows");
            match up {
                Some(u) => {
                    let t = self.scratch(rows * k);
                    self.silu_mul(x, u, &t, rows * k);
                    self.matmul_f16_rows(&f.w, n, k, &t, y, rows);
                }
                None => self.matmul_f16_rows(&f.w, n, k, x, y, rows),
            }
            return;
        }
        let g = w.as_any().and_then(|a| a.downcast_ref::<crate::exl3::Exl3Gpu>()).expect("an EXL3 projection this adapter holds");
        assert!(g.is_on(&self.backend.gpu), "chain: an EXL3 projection of another adapter");
        let (words, splits) = g.single_chunk().expect("an EXL3 projection in one buffer");
        let (k, n) = g.kn();
        assert!(rows > 0 && x.len >= rows * k && y.len >= rows * n && rows <= 65535, "chain: an EXL3 [{n}, {k}] of {rows} rows");
        let c = g.chain(self.backend);
        // a step's one row: the projection's own scratch (its bind groups kept); a check's few rows the device's shared
        // few-rows scratch (kept too); else this call's
        let few = (2..=crate::exl3::FEW_MAX).contains(&rows) && self.keep && crate::exl3::FewScratch::fits(k, n, splits as usize);
        let (xh, part, yt, jobs) = if rows == 1 && self.keep {
            (c.xh.clone(), c.part.clone(), c.yt.clone(), c.jobs1.clone())
        } else if few {
            let f = self.gpu().few(self.backend);
            (f.xh.clone(), f.part.clone(), f.yt.clone(), f.jobs.clone())
        } else {
            let list: Vec<u32> = (0..rows as u32).flat_map(|r| [0, r]).collect();
            let jobs = self.scratch(list.len());
            crate::exl3::upload_u32(self.backend, &jobs, &list);
            let lens = [rows * k, rows * (splits as usize).max(COOP_SPLITS_MAX) * n, rows * n];
            let [xh, part, yt] = match self.exl3_tmp.take() {
                Some(t) if t.iter().zip(lens).all(|(v, len)| v.len >= len) => t,
                Some([a, b, c]) => {
                    let grow = |v: DeviceVec, len: usize, r: &mut Self| if v.len >= len { v } else { r.scratch(len.max(v.len)) };
                    [grow(a, lens[0], self), grow(b, lens[1], self), grow(c, lens[2], self)]
                }
                None => [self.scratch(lens[0]), self.scratch(lens[1]), self.scratch(lens[2])],
            };
            self.exl3_tmp = Some([xh.clone(), part.clone(), yt.clone()]);
            (xh, part, yt, jobs)
        };
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        let imap = c.imap.as_ref().map_or(&d, buffer).clone();
        match up {
            Some(u) => {
                assert!(x.len >= rows * k && u.len >= rows * k, "chain: a SwiGLU's {rows} rows of {k}");
                self.dispatch_wide("exl3-pre-swiglu", crate::exl3::chain_shader("pre-swiglu"), [buffer(x), buffer(&c.suh), &imap, buffer(&jobs), buffer(u), &d, buffer(&xh), &drw], &[k as u32, c.imap.is_none() as u32, 1, 0, 1, 0], ((k / 128) as u32, rows as u32, 1));
            }
            None => self.dispatch_wide("exl3-pre", crate::exl3::chain_shader("pre"), [buffer(x), buffer(&c.suh), &imap, buffer(&jobs), &d, &d, buffer(&xh), &drw], &[k as u32, c.imap.is_none() as u32], ((k / 128) as u32, rows as u32, 1)),
        }
        let ntiles = (n / 16) as u32;
        let grid = |z: usize| (ntiles.min(65535), ntiles.div_ceil(65535), z as u32 * splits);
        let mm = crate::exl3::chain_shader("mm");
        // a prompt's rows on the tensor cores, in blocks of 128, split along k where its workgroups are too few to fill
        // the GPU twice over (a block of 128 rows of a projection 2,048 wide is 16 of them)
        let coop = rows > crate::exl3::FEW_MAX && crate::exl3::coop_on(self.gpu());
        let mut coop_splits = 1;
        if coop {
            const BLOCK: usize = 128;
            let many: Vec<u32> = (0..rows as u32).collect::<Vec<_>>().chunks(BLOCK).flat_map(|b| b.iter().copied().chain(std::iter::repeat(crate::exl3::NONE)).take(BLOCK)).collect();
            let order = self.scratch(many.len());
            crate::exl3::upload_u32(self.backend, &order, &many);
            let blocks = many.len() / BLOCK;
            let groups = ntiles.div_ceil(8) as usize * blocks;
            let kts = k / 16;
            let want = (2 * self.gpu().coop_units() as usize).div_ceil(groups).clamp(1, (kts / 8).clamp(1, COOP_SPLITS_MAX));
            coop_splits = kts.div_ceil(kts.div_ceil(want));
            let src = crate::exl3::g_coop(BLOCK);
            let per = 65535 / coop_splits;
            for first in (0..blocks).step_by(per) {
                let these = (per.min(blocks - first) * coop_splits) as u32;
                self.dispatch_wide(crate::exl3::coop_name(BLOCK), &src, [words, buffer(&xh), buffer(&jobs), buffer(&order), &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, coop_splits as u32, 0, first as u32], (ntiles.div_ceil(8), 1, these));
            }
        } else if rows == 1 {
            self.dispatch_wide("exl3-mm", mm, [words, buffer(&xh), buffer(&jobs), &d, &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0], grid(1));
        } else if rows <= crate::exl3::FEW_MAX {
            // a few rows (a check of drafts): each tile decoded once for all of them, each row summed as one row is
            let order = if few {
                self.gpu().few(self.backend).order.clone()
            } else {
                let order = self.scratch(rows);
                crate::exl3::upload_u32(self.backend, &order, &(0..rows as u32).collect::<Vec<_>>());
                order
            };
            self.dispatch_wide(crate::exl3::few_name(rows), crate::exl3::g_few(rows), [words, buffer(&xh), buffer(&jobs), buffer(&order), &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0, 0], grid(1));
        } else {
            // a prompt's rows summed as the projection's own passes sum them (each tile decoded once for 64 of them here),
            // and a last lone row of a pass as one row is
            const BLOCK: usize = 64;
            let lone = rows % 32 == 1;
            let many: Vec<u32> = (0..(rows - lone as usize) as u32).collect::<Vec<_>>().chunks(BLOCK).flat_map(|b| b.iter().copied().chain(std::iter::repeat(crate::exl3::NONE)).take(BLOCK)).collect();
            let order = self.scratch(many.len());
            crate::exl3::upload_u32(self.backend, &order, &many);
            let per = 65535 / splits as usize;
            let blocks = many.len() / BLOCK;
            let kernel = crate::exl3::g_many(BLOCK);
            for first in (0..blocks).step_by(per) {
                self.dispatch_wide(crate::exl3::many_name(BLOCK), &kernel, [words, buffer(&xh), buffer(&jobs), buffer(&order), &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0, first as u32], grid(per.min(blocks - first)));
            }
            if lone {
                self.dispatch_wide("exl3-mm", mm, [words, buffer(&xh), buffer(&jobs), &d, &d, &d, buffer(&part), &drw], &[n as u32, k as u32, g.tile_words() as u32, splits, 0, rows as u32 - 1], grid(1));
            }
        }
        let post = crate::exl3::chain_shader("post");
        let post_out = if c.omap.is_some() { buffer(&yt) } else { buffer(y) };
        let parts = if coop { coop_splits as u32 } else { splits };
        self.dispatch_wide("exl3-post", post, [buffer(&part), buffer(&c.svh), buffer(&jobs), &d, &d, &d, post_out, &drw], &[n as u32, parts], ((n / 128) as u32, rows as u32, 1));
        if let Some(omap) = &c.omap {
            let gather = crate::exl3::chain_shader("gather");
            self.dispatch_wide("exl3-gather", gather, [buffer(&yt), buffer(omap), buffer(&jobs), &d, &d, &d, buffer(y), &drw], &[n as u32], ((n as u32).div_ceil(256), rows as u32, 1));
        }
    }

    /// [`Self::exl3_rows_of`] of a SwiGLU: `silu(gate) * up`'s rows.
    pub(crate) fn exl3_rows_swiglu(&mut self, w: &dyn ggml_rs::exl3::PackedLinear, gate: &DeviceVec, up: &DeviceVec, y: &DeviceVec, rows: usize) {
        self.exl3_rows_of(w, gate, Some(up), y, rows);
    }
}
