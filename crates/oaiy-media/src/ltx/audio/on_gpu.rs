//! The audio VAE's two generators on WebGPU: BigVGAN's (the vocoder at 16 kHz, then the bandwidth extension's at
//! 48 kHz) as chains of the device's 1-D convolutions and its SnakeBeta without aliasing
//! ([`ChainRecorder::snake_beta_alias_rows`]). They are nearly all of a clip's sound's decoding: on the CPU a 2 s
//! clip's took 16 s and the largest clip's 38 s. The VAE's own decoder (a latent to its log-mel), the STFT between the
//! generators and the skip path's resampling stay Candle's on the host.
//!
//! The convolutions' weights are f16 on the device (the kernel's), their sums f32: the reference keeps the vocoder in
//! f32 because BF16 throughout loses it, and what f16 weights alone cost is what the test below measures.
use super::{msg, AudioDecoder, Generator};
use candle_core::{DType, Result, Tensor};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs_wgpu::WgpuBackend;

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn upload(gpu: &WgpuBackend, v: &[f32]) -> DeviceVec {
    let d = gpu.vec(v.len());
    gpu.upload(&d, v);
    d
}

/// A 1-D convolution's packed weights and bias.
struct Conv1 {
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
    k: usize,
}

/// An activation without aliasing: its filters (the upsampling one's taps, then the low-pass's) and SnakeBeta's
/// frequency and scale a channel.
struct Act {
    filters: DeviceVec,
    ku: usize,
    kd: usize,
    freq: DeviceVec,
    scale: DeviceVec,
}

/// A residual block's unit: an activation and a dilated convolution, then another of each.
struct Unit {
    act1: Act,
    conv1: Conv1,
    dilation: usize,
    act2: Act,
    conv2: Conv1,
}

/// A stage: its transposed convolution (the weight as `[cout, k, cin]`), then its residual blocks, each a few units.
struct Stage {
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    blocks: Vec<Vec<Unit>>,
}

/// One generator on the device.
struct OnGpu {
    pre: Conv1,
    stages: Vec<Stage>,
    post_act: Act,
    post: Conv1,
    clamp: bool,
    /// 1, and what a stage's blocks' sum is added to itself by to be their mean (`1 / blocks - 1`).
    weights: DeviceVec,
}

impl AudioDecoder {
    fn gpu_conv(&self, gpu: &WgpuBackend, prefix: &str) -> Result<Conv1> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let (cout, cin, k) = w.dims3()?;
        if k % 2 == 0 {
            candle_core::bail!("{prefix}: a convolution of {k} taps");
        }
        let packed = gpu.conv1d_weights(&values(w)?, cout, cin, k).ok_or_else(|| msg(format!("{prefix}: a weight past f16's range")))?;
        let b = match self.w.get(&format!("{prefix}.bias")) {
            Some(b) => values(b)?,
            None => vec![0.; cout],
        };
        Ok(Conv1 { w: packed, b: upload(gpu, &b), cin, cout, k })
    }

    fn gpu_act(&self, gpu: &WgpuBackend, prefix: &str) -> Result<Act> {
        let up = values(self.get(&format!("{prefix}.upsample.filter"))?)?;
        let down = values(self.get(&format!("{prefix}.downsample.lowpass.filter"))?)?;
        let freq: Vec<f32> = values(self.get(&format!("{prefix}.act.alpha"))?)?.iter().map(|a| a.exp()).collect();
        let scale: Vec<f32> = values(self.get(&format!("{prefix}.act.beta"))?)?.iter().map(|b| 1.0 / (b.exp() + 1e-9)).collect();
        let (ku, kd) = (up.len(), down.len());
        if ku < 2 || ku % 2 != 0 || kd == 0 {
            candle_core::bail!("{prefix}: filters of {ku} and {kd} taps");
        }
        let filters: Vec<f32> = up.into_iter().chain(down).collect();
        Ok(Act { filters: upload(gpu, &filters), ku, kd, freq: upload(gpu, &freq), scale: upload(gpu, &scale) })
    }

    fn gpu_generator(&self, gpu: &WgpuBackend, g: &Generator) -> Result<OnGpu> {
        let p = g.prefix;
        let blocks = g.resblock_kernels.len();
        let mut stages = Vec::new();
        for (i, (&stride, &kernel)) in g.upsample_rates.iter().zip(&g.upsample_kernels).enumerate() {
            let w = self.get(&format!("{p}.ups.{i}.weight"))?;
            let (cin, cout, k) = w.dims3()?;
            if k != kernel || k < stride || (k - stride) % 2 != 0 || cin % 4 != 0 {
                candle_core::bail!("{p}.ups.{i}: a transposed convolution of {k} taps at stride {stride}, {cin} channels");
            }
            // (PyTorch's `[cin, cout, k]` as the kernel reads it)
            let v = values(w)?;
            let packed: Vec<f32> = (0..cout * k * cin)
                .map(|at| {
                    let (co, tap, c) = (at / (k * cin), (at / cin) % k, at % cin);
                    v[(c * cout + co) * k + tap]
                })
                .collect();
            let b = values(self.get(&format!("{p}.ups.{i}.bias"))?)?;
            let mut stage = Vec::new();
            for (j, d) in g.dilations.iter().enumerate() {
                let rp = format!("{p}.resblocks.{}", i * blocks + j);
                let mut units = Vec::new();
                for (u, &dilation) in d.iter().enumerate() {
                    units.push(Unit {
                        act1: self.gpu_act(gpu, &format!("{rp}.acts1.{u}"))?,
                        conv1: self.gpu_conv(gpu, &format!("{rp}.convs1.{u}"))?,
                        dilation,
                        act2: self.gpu_act(gpu, &format!("{rp}.acts2.{u}"))?,
                        conv2: self.gpu_conv(gpu, &format!("{rp}.convs2.{u}"))?,
                    });
                }
                stage.push(units);
            }
            stages.push(Stage { w: upload(gpu, &packed), b: upload(gpu, &b), cin, cout, k, stride, blocks: stage });
        }
        Ok(OnGpu {
            pre: self.gpu_conv(gpu, &format!("{p}.conv_pre"))?,
            stages,
            post_act: self.gpu_act(gpu, &format!("{p}.act_post"))?,
            post: self.gpu_conv(gpu, &format!("{p}.conv_post"))?,
            clamp: g.clamp,
            weights: upload(gpu, &[1.0, 1.0 / blocks as f32 - 1.0]),
        })
    }

    /// [`Self::waveform`] with the two generators on `gpu`: a log-mel `(1, 2, frames, 64)` to the stereo waveform at
    /// `sample_rate`, `(1, 2, samples)`.
    pub fn waveform_webgpu(&self, gpu: &WgpuBackend, mel: &Tensor) -> Result<Tensor> {
        let low = self.gpu_generator(gpu, &self.vocoder)?.generate(gpu, mel)?;
        let len = low.dim(2)?;
        let low = low.pad_with_zeros(2, 0, (self.hop - len % self.hop) % self.hop)?;
        let residual = self.gpu_generator(gpu, &self.bwe)?.generate(gpu, &self.log_mel(&low)?)?;
        let skip = self.resample(&low)?;
        (residual + skip)?.narrow(2, 0, len * self.ratio)
    }

    /// [`Self::decode`] with the generators on WebGPU device `device`.
    pub fn decode_webgpu(&self, device: usize, latent: &Tensor) -> Result<Tensor> {
        let gpu = WgpuBackend::nth(device, None).map_err(msg)?;
        self.waveform_webgpu(&gpu, &self.mel(latent)?)
    }
}

impl Act {
    /// The activation of `x`'s `len` steps of `c` into `y` (`mid` its `2 len` upsampled steps).
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, len: usize, c: usize, mid: &DeviceVec, y: &DeviceVec) {
        r.snake_beta_alias_rows(x, &self.filters, self.ku, self.kd, &self.freq, &self.scale, len, c, mid, y);
    }
}

impl OnGpu {
    /// A mel `(1, streams, frames, bins)` to the generator's waveform `(1, channels, samples)`, as
    /// [`AudioDecoder::generate`]: a step's channels are its streams' bins in turn.
    fn generate(&self, gpu: &WgpuBackend, mel: &Tensor) -> Result<Tensor> {
        let (b, s, frames, bins) = mel.dims4()?;
        if b != 1 || s * bins != self.pre.cin {
            candle_core::bail!("a mel of {b} by {s} by {bins} bins for a generator of {} channels", self.pre.cin);
        }
        let x0 = upload(gpu, &values(&mel.permute((0, 2, 1, 3))?.contiguous()?)?);
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let mut len = frames;
        let mut x = gpu.vec(len * self.pre.cout);
        r.conv1d_rows(&self.pre.w, &self.pre.b, self.pre.cout, self.pre.cin, self.pre.k, 1, &x0, len, &x);
        for st in &self.stages {
            let (out, c) = (len * st.stride, st.cout);
            let y = gpu.vec(out * c);
            r.conv_transpose1d_rows(&st.w, &st.b, c, st.cin, st.k, st.stride, (st.k - st.stride) / 2, &x, len, out, &y);
            len = out;
            // a block's stream, an activation's output, a convolution's, the activations' upsampled steps, and the
            // blocks' sum
            let (a, h, g, mid, sum) = (gpu.vec(len * c), gpu.vec(len * c), gpu.vec(len * c), gpu.vec(2 * len * c), gpu.vec(len * c));
            for (j, units) in st.blocks.iter().enumerate() {
                // (the blocks all read the stage's steps; their mean moves on)
                r.copy(&y, 0, &a, 0, len * c);
                for u in units {
                    u.act1.run(r, &a, len, c, &mid, &h);
                    r.conv1d_rows(&u.conv1.w, &u.conv1.b, c, c, u.conv1.k, u.dilation, &h, len, &g);
                    u.act2.run(r, &g, len, c, &mid, &h);
                    r.conv1d_rows(&u.conv2.w, &u.conv2.b, c, c, u.conv2.k, 1, &h, len, &g);
                    r.axpy_at(&a, &g, &self.weights, 0, len * c);
                }
                if j == 0 {
                    r.copy(&a, 0, &sum, 0, len * c);
                } else {
                    r.axpy_at(&sum, &a, &self.weights, 0, len * c);
                }
            }
            r.copy(&sum, 0, &y, 0, len * c);
            r.axpy_at(&y, &sum, &self.weights, 1, len * c);
            x = y;
        }
        let c = self.post.cin;
        let (mid, h, out) = (gpu.vec(2 * len * c), gpu.vec(len * c), gpu.vec(len * self.post.cout));
        self.post_act.run(r, &x, len, c, &mid, &h);
        r.conv1d_rows(&self.post.w, &self.post.b, self.post.cout, c, self.post.k, 1, &h, len, &out);
        if self.clamp {
            r.clamp_in_place(&out, len * self.post.cout, -1.0, 1.0);
        }
        r.read(&out);
        let wave = rec.finish().pop().ok_or_else(|| msg("the waveform was not read"))?;
        let n = len * self.post.cout;
        if wave.len() < n {
            candle_core::bail!("a waveform of {} values where {n}", wave.len());
        }
        Tensor::from_vec(wave[..n].to_vec(), (1, len, self.post.cout), mel.device())?.transpose(1, 2)?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The generators on WebGPU give Candle's waveform on the CPU from the same mel (the checkpoint's audio VAE,
    /// `OAIY_LTX_CHECKPOINT`; a latent of twelve frames at random, half a second): the vocoder alone and the whole
    /// decode, each's error against Candle's (relative RMS) and what each took.
    #[test]
    #[ignore = "needs the checkpoint's audio VAE (OAIY_LTX_CHECKPOINT) and a WebGPU adapter"]
    fn the_webgpu_generators_are_candles() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_CHECKPOINT") else { return Ok(()) };
        let decoder = AudioDecoder::load(std::path::Path::new(&path), &Device::Cpu)?;
        let gpu = WgpuBackend::nth(0, None).map_err(msg)?;
        let frames: usize = std::env::var("OAIY_LTX_AUDIO_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let latent: Vec<f32> = (0..frames * 128)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
            })
            .collect();
        let latent = Tensor::from_vec(latent, (1, frames, 128), &Device::Cpu)?;
        let t = std::time::Instant::now();
        let mel = decoder.mel(&latent)?;
        let mel_seconds = t.elapsed().as_secs_f64();
        let relative = |got: &Tensor, want: &Tensor| -> Result<f64> {
            let (g, w) = (values(got)?, values(want)?);
            assert_eq!(g.len(), w.len());
            let e: f64 = g.iter().zip(&w).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
            Ok((e / w.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt())
        };
        let t = std::time::Instant::now();
        let want_low = decoder.vocoder_only(&mel)?;
        let low_seconds = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let got_low = decoder.gpu_generator(&gpu, &decoder.vocoder)?.generate(&gpu, &mel)?;
        let low_gpu = t.elapsed().as_secs_f64();
        let low = relative(&got_low, &want_low)?;
        eprintln!("{frames} latent frames: the VAE's decoder {mel_seconds:.2} s (the host); the vocoder {low_seconds:.2} s on the host, {low_gpu:.2} s on WebGPU, {low:.5} from the host's (relative RMS)");
        let t = std::time::Instant::now();
        let want = decoder.waveform(&mel)?;
        let whole_seconds = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let got = decoder.waveform_webgpu(&gpu, &mel)?;
        let whole_gpu = t.elapsed().as_secs_f64();
        let whole = relative(&got, &want)?;
        eprintln!("the whole waveform ({} samples): {whole_seconds:.2} s on the host, {whole_gpu:.2} s with the generators on WebGPU, {whole:.5} from the host's", want.dim(2)?);
        assert!(low < 0.02 && whole < 0.02, "the generators on WebGPU: {low} and {whole} from Candle's");
        Ok(())
    }
}
