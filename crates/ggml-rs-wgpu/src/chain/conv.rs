//! The recorder's convolutions: a kernel's taps over an image's or a clip's voxels.
use super::*;

impl Recorder<'_> {
    /// [`ChainRecorder::conv_rows`] and [`ChainRecorder::conv3d_rows`]: `taps` 1, 9 (3x3) or 27 (3x3x3), `frames` of
    /// `h` rows of `wd` (one frame for a picture), `x`'s voxels `xs` values apart (its first `cin` each).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv_taps(&mut self, w: &DeviceVec, b: &DeviceVec, cout: usize, cin: usize, taps: usize, x: &DeviceVec, xs: usize, frames: usize, h: usize, wd: usize, y: &DeviceVec) {
        let cp = cin.div_ceil(32) * 32;
        let m = frames * h * wd;
        assert!(m > 0 && xs >= cin && w.len * 2 >= cout * taps * cp && b.len >= cout && x.len >= (m - 1) * xs + cin && y.len >= m * cout, "chain: a convolution of {taps} taps of {frames}x{h}x{wd} voxels, {cin} channels ({xs} apart) to {cout}");
        assert!(cp < 1 << 16 && xs < 1 << 14, "chain: a convolution's {cin} channels {xs} apart");
        let d = self.gpu().dummy().clone();
        let drw = self.gpu().dummy_rw().clone();
        if !self.gpu().device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            // no tensor cores: the f32 tiled kernel, the voxels' tiles in chunks of bounded work (the inputs as they
            // are, no f16 copy or range to keep)
            let tiles = m.div_ceil(64);
            let per_tile = 2.0 * 64.0 * (cout * cin * taps) as f64;
            let chunk = ((CONV_DISPATCH_FLOPS / per_tile) as usize).clamp(1, 65535);
            let mut first = 0;
            while first < tiles {
                let n = chunk.min(tiles - first);
                let words = [cout as u32, cin as u32, m as u32, xs as u32, taps as u32, wd as u32, h as u32, first as u32];
                self.dispatch_wide("chain-conv-f32-tiled", CONV_F32_TILED, [buffer(w), buffer(x), buffer(b), &d, &d, &d, buffer(y), &drw], &words, ((cout as u32).div_ceil(64), n as u32, 1));
                self.weigh(per_tile * n as f64);
                first += n;
            }
            return;
        }
        // the input as f16, each pixel's channels padded to 32's: once for every convolution that reads it until
        // something writes `x` (kept with the matmuls' tiled copies, a width of its own)
        let key = cp | xs << 16 | 1 << 31;
        let xb = buffer(x).clone();
        let x16 = match self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == key) {
            Some((.., v)) => v.clone(),
            None => {
                // `x`'s scale for f16 (its largest within 16,384) set on the device, then the copy scaled; the scale kept
                // beside the copy (a width of its own) for the sums' way back
                let range = self.scratch(4);
                // (over every value of the pixels: a strided input's others too, its range no smaller)
                let len = ((m - 1) * xs + cin) as u32;
                self.dispatch_wide("chain-f16-range-clear", F16_RANGE_CLEAR, [&d, &d, &d, &d, &d, &d, buffer(&range), &drw], &[0], (1, 1, 1));
                self.dispatch_wide("chain-f16-range-max", F16_RANGE_MAX, [buffer(x), &d, &d, &d, &d, &d, buffer(&range), &drw], &[len], grid(len.div_ceil(256)));
                self.dispatch_wide("chain-f16-range-set", F16_RANGE_SET, [&d, &d, &d, &d, &d, &d, buffer(&range), &drw], &[0], (1, 1, 1));
                let v = self.scratch(m * cp / 2);
                let conv = self.gpu().named_pipeline("chain-x-f16-padded", || X_F16_PADDED.to_string());
                let words = (m * cp / 2) as u32;
                self.dispatch_kept(&conv, buffer(&range), buffer(x), buffer(&v), &[cin as u32, cp as u32, m as u32, xs as u32], grid(words.div_ceil(256)));
                self.x16.push((xb.clone(), m, key, v.clone()));
                self.x16.push((xb, m, key ^ (3 << 30), range));
                v
            }
        };
        let xb = buffer(x).clone();
        let range = self.x16.iter().find(|(b, rows, width, _)| *b == xb && *rows == m && *width == key ^ (3 << 30)).map(|(.., v)| v.clone()).expect("a convolution's input's scale beside its copy");
        let tile = crate::shaders::COOP_TILE;
        let pipeline = match taps {
            27 => self.gpu().named_pipeline("chain-coop-conv3d", || crate::shaders::coop_conv(27)),
            49 => self.gpu().named_pipeline("chain-coop-conv7x7", || crate::shaders::coop_conv(49)),
            9 => self.gpu().named_pipeline("chain-coop-conv3x3", || crate::shaders::coop_conv(9)),
            _ => self.gpu().named_pipeline("chain-coop-conv1x1", || crate::shaders::coop_conv(1)),
        };
        let kk = taps * cp;
        let tiles = (cout as u32).div_ceil(tile) * (m as u32).div_ceil(tile);
        let (splits, out, parts) = self.coop_parts(tiles, kk, m, cout, y, None);
        let words = [kk as u32, cout as u32, m as u32, 0, cout as u32, wd as u32, splits, h as u32];
        self.dispatch_kept(&pipeline, buffer(w), buffer(&x16), &out, &words, ((cout as u32).div_ceil(tile), (m as u32).div_ceil(tile), splits));
        self.coop_sum(parts, m, cout, y, splits);
        // the sums back to x's range, and the bias
        self.dispatch_wide("chain-unscale-bias-rows", UNSCALE_BIAS_ROWS, [buffer(b), buffer(&range), &d, &d, &d, &d, buffer(y), &drw], &[cout as u32, m as u32], grid(((m * cout) as u32).div_ceil(256)));
        self.weigh(2.0 * (m * cout) as f64 * (cin * taps) as f64);
    }
}
