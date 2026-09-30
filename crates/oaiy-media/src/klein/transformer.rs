//! Native BFL-layout Klein 4B transformer. Shared modulation is evaluated
//! once per step; five double streams precede twenty fused single streams.
use super::math;
use crate::{
    lora::Loras,
    math::{attention, heads, layer_norm, unheads},
    residency::{Budget, Resident, Tiered},
    weights::{bytes_of, Linear, Weights},
};
use candle_core::{DType, Device, Result, Tensor, D};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub double: usize,
    pub single: usize,
    pub context: usize,
    pub channels: usize,
    pub axes: [usize; 4],
}
impl Default for Config {
    fn default() -> Self {
        Self {
            hidden: 3072,
            heads: 24,
            double: 5,
            single: 20,
            context: 7680,
            channels: 128,
            axes: [32; 4],
        }
    }
}
impl Config {
    pub fn projections(self) -> Vec<(String, usize, usize)> {
        let h = self.hidden;
        let mut v = vec![
            ("img_in".into(), h, self.channels),
            ("txt_in".into(), h, self.context),
            ("time_in.in_layer".into(), h, 256),
            ("time_in.out_layer".into(), h, h),
            ("double_stream_modulation_img.lin".into(), 6 * h, h),
            ("double_stream_modulation_txt.lin".into(), 6 * h, h),
            ("single_stream_modulation.lin".into(), 3 * h, h),
            ("final_layer.adaLN_modulation.1".into(), 2 * h, h),
            ("final_layer.linear".into(), self.channels, h),
        ];
        for i in 0..self.double {
            for stream in ["img", "txt"] {
                for (name, out, input) in [
                    ("attn.qkv", 3 * h, h),
                    ("attn.proj", h, h),
                    ("mlp.0", 6 * h, h),
                    ("mlp.2", h, 3 * h),
                ] {
                    v.push((format!("double_blocks.{i}.{stream}_{name}"), out, input));
                }
            }
        }
        for i in 0..self.single {
            v.push((format!("single_blocks.{i}.linear1"), 9 * h, h));
            v.push((format!("single_blocks.{i}.linear2"), h, 4 * h));
        }
        v
    }
    pub fn validate(self, w: &Weights) -> Result<()> {
        if self.heads == 0
            || self.hidden % self.heads != 0
            || self.axes.iter().sum::<usize>() != self.hidden / self.heads
        {
            candle_core::bail!("invalid Klein architecture");
        }
        for (name, out, input) in self.projections() {
            let shape = w.shape(&format!("{name}.weight"))?;
            if shape != [out, input] {
                candle_core::bail!("Klein 4B tensor {name} has {shape:?}, expected [{out}, {input}]; use the original BFL-layout checkpoint");
            }
        }
        Ok(())
    }
}

struct Stream {
    qkv: Linear,
    proj: Linear,
    mlp_in: Linear,
    mlp_out: Linear,
    qn: Tensor,
    kn: Tensor,
}
impl Stream {
    fn load(w: &mut Weights, p: &str, dev: &Device, dtype: DType, l: &mut Loras) -> Result<Self> {
        Ok(Self {
            qkv: w.linear(&format!("{p}_attn.qkv"), dev, dtype, l)?,
            proj: w.linear(&format!("{p}_attn.proj"), dev, dtype, l)?,
            mlp_in: w.linear(&format!("{p}_mlp.0"), dev, dtype, l)?,
            mlp_out: w.linear(&format!("{p}_mlp.2"), dev, dtype, l)?,
            qn: w.tensor(&format!("{p}_attn.norm.query_norm.scale"), dev, dtype)?,
            kn: w.tensor(&format!("{p}_attn.norm.key_norm.scale"), dev, dtype)?,
        })
    }
    fn qkv(&self, x: &Tensor, m: &[Tensor], cfg: Config) -> Result<(Tensor, Tensor, Tensor)> {
        qkv(
            &self.qkv.forward(&modulate(x, &m[0], &m[1])?)?,
            &self.qn,
            &self.kn,
            cfg,
        )
    }
    fn residual(&self, x: &Tensor, attn: &Tensor, m: &[Tensor]) -> Result<Tensor> {
        let x = (x + self.proj.forward(attn)?.broadcast_mul(&m[2])?)?;
        let ff = gated(&self.mlp_in.forward(&modulate(&x, &m[3], &m[4])?)?)?;
        &x + self.mlp_out.forward(&ff)?.broadcast_mul(&m[5])?
    }
    fn bytes(&self) -> u64 {
        [&self.qkv, &self.proj, &self.mlp_in, &self.mlp_out]
            .iter()
            .map(|l| l.bytes())
            .sum::<u64>()
            + bytes_of(&self.qn)
            + bytes_of(&self.kn)
    }
    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            qkv: self.qkv.to_device(d)?,
            proj: self.proj.to_device(d)?,
            mlp_in: self.mlp_in.to_device(d)?,
            mlp_out: self.mlp_out.to_device(d)?,
            qn: self.qn.to_device(d)?,
            kn: self.kn.to_device(d)?,
        })
    }
}
struct Double {
    img: Stream,
    txt: Stream,
}
impl Resident for Double {
    fn bytes(&self) -> u64 {
        self.img.bytes() + self.txt.bytes()
    }
    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            img: self.img.to_device(d)?,
            txt: self.txt.to_device(d)?,
        })
    }
}
fn load_double(w: &mut Weights, i: usize, d: &Device, t: DType, l: &mut Loras) -> Result<Double> {
    Ok(Double {
        img: Stream::load(w, &format!("double_blocks.{i}.img"), d, t, l)?,
        txt: Stream::load(w, &format!("double_blocks.{i}.txt"), d, t, l)?,
    })
}
struct Single {
    input: Linear,
    output: Linear,
    qn: Tensor,
    kn: Tensor,
}
impl Resident for Single {
    fn bytes(&self) -> u64 {
        self.input.bytes() + self.output.bytes() + bytes_of(&self.qn) + bytes_of(&self.kn)
    }
    fn to_device(&self, d: &Device) -> Result<Self> {
        Ok(Self {
            input: self.input.to_device(d)?,
            output: self.output.to_device(d)?,
            qn: self.qn.to_device(d)?,
            kn: self.kn.to_device(d)?,
        })
    }
}
fn load_single(w: &mut Weights, i: usize, d: &Device, t: DType, l: &mut Loras) -> Result<Single> {
    let p = format!("single_blocks.{i}");
    Ok(Single {
        input: w.linear(&format!("{p}.linear1"), d, t, l)?,
        output: w.linear(&format!("{p}.linear2"), d, t, l)?,
        qn: w.tensor(&format!("{p}.norm.query_norm.scale"), d, t)?,
        kn: w.tensor(&format!("{p}.norm.key_norm.scale"), d, t)?,
    })
}
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    layer_norm(x)?
        .broadcast_mul(&(scale + 1.)?)?
        .broadcast_add(shift)
}
fn gated(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    if d % 2 != 0 {
        candle_core::bail!("invalid gated MLP width");
    }
    candle_nn::ops::silu(&x.narrow(D::Minus1, 0, d / 2)?)? * x.narrow(D::Minus1, d / 2, d / 2)?
}
fn qkv(x: &Tensor, qn: &Tensor, kn: &Tensor, cfg: Config) -> Result<(Tensor, Tensor, Tensor)> {
    let h = cfg.hidden;
    Ok((
        rms(&heads(&x.narrow(2, 0, h)?, cfg.heads)?, qn)?,
        rms(&heads(&x.narrow(2, h, h)?, cfg.heads)?, kn)?,
        heads(&x.narrow(2, 2 * h, h)?, cfg.heads)?,
    ))
}
// BFL rounds the normalized activation back to its dtype before the scale.
fn rms(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?
        .to_dtype(x.dtype())?
        .broadcast_mul(scale)
}
fn chunks(x: Tensor, n: usize) -> Result<Vec<Tensor>> {
    let h = x.dim(D::Minus1)? / n;
    (0..n)
        .map(|i| x.narrow(D::Minus1, i * h, h)?.unsqueeze(1))
        .collect()
}

pub struct Transformer {
    cfg: Config,
    weights: Weights,
    loras: Loras,
    dev: Device,
    dtype: DType,
    img: Linear,
    txt: Linear,
    time_in: Linear,
    time_out: Linear,
    img_mod: Linear,
    txt_mod: Linear,
    single_mod: Linear,
    final_mod: Linear,
    final_proj: Linear,
    double: Tiered<Double>,
    single: Tiered<Single>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oaiy_engine::json::Json;
    #[test]
    fn full_tiny_transformer_matches_published_bfl_forward_on_every_residency_tier() -> Result<()> {
        check_published_forward(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an explicitly selected idle OAIY_KLEIN_TEST_CUDA_DEVICE"]
    fn cuda_tiny_transformer_matches_published_bfl_forward_on_every_residency_tier() -> Result<()> {
        let device = std::env::var("OAIY_KLEIN_TEST_CUDA_DEVICE")
            .map_err(candle_core::Error::wrap)?
            .parse::<usize>()
            .map_err(candle_core::Error::wrap)?;
        check_published_forward(&Device::new_cuda(device)?)
    }

    fn check_published_forward(device: &Device) -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/klein");
        let j = Json::parse(&std::fs::read(root.join("expected.json"))?)
            .map_err(candle_core::Error::wrap)?;
        let values = |key: &str| -> Result<Vec<f32>> {
            j.get(key)
                .and_then(Json::as_array)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {key}")))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|n| n as f32)
                        .ok_or_else(|| candle_core::Error::Msg("invalid reference number".into()))
                })
                .collect()
        };
        let cfg = Config {
            hidden: 8,
            heads: 1,
            double: 1,
            single: 1,
            context: 6,
            channels: 8,
            axes: [2; 4],
        };
        let x = Tensor::from_vec(values("input")?, (1, 4, 8), device)?;
        let context = Tensor::from_vec(values("context")?, (1, 3, 6), device)?;
        let expected = values("output")?;
        for memory in [
            crate::residency::Memory::Gpu,
            crate::residency::Memory::Ram,
            crate::residency::Memory::Ssd,
        ] {
            let budget = Budget {
                memory,
                ..Budget::default()
            };
            let mut model = Transformer::load_config(
                &root.join("tiny.safetensors"),
                &[],
                device,
                DType::F32,
                &budget,
                cfg,
            )?;
            let actual = model
                .predict(&x, &context, 0.625, 2, 2)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(error < 2e-5, "BFL forward error {error}, tier {memory:?}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires OAIY_KLEIN_LORA; reads adapter headers and factors on CPU only"]
    fn supplied_adapter_fits_all_klein_projections() -> Result<()> {
        let p = std::env::var("OAIY_KLEIN_LORA").map_err(candle_core::Error::wrap)?;
        let w = Weights::open(Path::new(&p))?;
        assert_eq!(w.names().len(), 160);
        let mut l = Loras::open(&[(PathBuf::from(p), 1.0)])?;
        l.validate_modules(&Config::default().projections())?;
        // Other FLUX widths/architectures must fail before loading payloads.
        let wrong = Config {
            hidden: 4096,
            heads: 32,
            ..Config::default()
        };
        assert!(l.validate_modules(&wrong.projections()).is_err());
        let mut pairs = 0;
        for (name, out, input) in Config::default().projections() {
            for (a, b) in l.factors(&name, out, input, &Device::Cpu, DType::F32)? {
                assert_eq!(a.dims(), [32, input]);
                assert_eq!(b.dims(), [out, 32]);
                assert!(a
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite()));
                assert!(b
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite()));
                pairs += 1;
            }
        }
        assert_eq!(pairs, 80);
        assert!(l.check()?.is_empty());
        Ok(())
    }
}
impl Transformer {
    pub fn load(
        path: &Path,
        adapters: &[(PathBuf, f64)],
        dev: &Device,
        dtype: DType,
        budget: &Budget,
    ) -> Result<Self> {
        Self::load_config(path, adapters, dev, dtype, budget, Config::default())
    }
    fn load_config(
        path: &Path,
        adapters: &[(PathBuf, f64)],
        dev: &Device,
        dtype: DType,
        budget: &Budget,
        cfg: Config,
    ) -> Result<Self> {
        let mut w = Weights::open(path)?;
        cfg.validate(&w)?;
        let mut l = Loras::open(adapters)?;
        l.validate_modules(&cfg.projections())?;
        let img = w.linear("img_in", dev, dtype, &mut l)?;
        let txt = w.linear("txt_in", dev, dtype, &mut l)?;
        let time_in = w.linear("time_in.in_layer", dev, dtype, &mut l)?;
        let time_out = w.linear("time_in.out_layer", dev, dtype, &mut l)?;
        let img_mod = w.linear("double_stream_modulation_img.lin", dev, dtype, &mut l)?;
        let txt_mod = w.linear("double_stream_modulation_txt.lin", dev, dtype, &mut l)?;
        let single_mod = w.linear("single_stream_modulation.lin", dev, dtype, &mut l)?;
        let final_mod = w.linear("final_layer.adaLN_modulation.1", dev, dtype, &mut l)?;
        let final_proj = w.linear("final_layer.linear", dev, dtype, &mut l)?;
        let double = Tiered::load(
            cfg.double,
            budget,
            dev,
            |i| load_double(&mut w, i, dev, dtype, &mut l),
            |_| {},
        )?;
        // Both groups share the user's budget, rather than each claiming its entirety.
        let mut remaining = *budget;
        remaining.vram_bytes = budget
            .vram_bytes
            .map(|limit| limit.saturating_sub(double.gpu_bytes));
        remaining.ram_bytes = remaining.ram_bytes.saturating_sub(double.host_bytes);
        let single = Tiered::load(
            cfg.single,
            &remaining,
            dev,
            |i| load_single(&mut w, i, dev, dtype, &mut l),
            |_| {},
        )?;
        Ok(Self {
            cfg,
            weights: w,
            loras: l,
            dev: dev.clone(),
            dtype,
            img,
            txt,
            time_in,
            time_out,
            img_mod,
            txt_mod,
            single_mod,
            final_mod,
            final_proj,
            double,
            single,
        })
    }
    pub fn predict(
        &mut self,
        x: &Tensor,
        context: &Tensor,
        t: f64,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let nt = context.dim(1)?;
        if x.dims() != [1, h * w, self.cfg.channels] || context.dims() != [1, nt, self.cfg.context]
        {
            candle_core::bail!("invalid Klein input shape");
        }
        let vec = self.time_out.forward(&candle_nn::ops::silu(
            &self
                .time_in
                .forward(&math::timestep(t, &self.dev, self.dtype)?)?,
        )?)?;
        let activated = candle_nn::ops::silu(&vec)?;
        let mi = chunks(self.img_mod.forward(&activated)?, 6)?;
        let mt = chunks(self.txt_mod.forward(&activated)?, 6)?;
        let ms = chunks(self.single_mod.forward(&activated)?, 3)?;
        let (cos, sin) = math::rotary(&math::ids(nt, h, w), self.cfg.axes, 2000., &self.dev)?;
        let mut img = self.img.forward(x)?;
        let mut txt = self.txt.forward(context)?;
        let cfg = self.cfg;
        let dev = &self.dev;
        let dtype = self.dtype;
        let weights = &mut self.weights;
        let loras = &mut self.loras;
        for i in 0..cfg.double {
            (img, txt) = self.double.with(
                i,
                |i| load_double(weights, i, dev, dtype, loras),
                |b| {
                    let (iq, ik, iv) = b.img.qkv(&img, &mi, cfg)?;
                    let (tq, tk, tv) = b.txt.qkv(&txt, &mt, cfg)?;
                    let q = math::rotate(&Tensor::cat(&[tq, iq], 2)?, &cos, &sin)?;
                    let k = math::rotate(&Tensor::cat(&[tk, ik], 2)?, &cos, &sin)?;
                    let a = unheads(&attention(&q, &k, &Tensor::cat(&[tv, iv], 2)?, 0)?)?;
                    Ok((
                        b.img.residual(&img, &a.narrow(1, nt, h * w)?, &mi)?,
                        b.txt.residual(&txt, &a.narrow(1, 0, nt)?, &mt)?,
                    ))
                },
            )?;
        }
        let mut all = Tensor::cat(&[txt, img], 1)?;
        for i in 0..cfg.single {
            all = self.single.with(
                i,
                |i| load_single(weights, i, dev, dtype, loras),
                |b| {
                    let projected = b.input.forward(&modulate(&all, &ms[0], &ms[1])?)?;
                    let (q, k, v) = qkv(&projected, &b.qn, &b.kn, cfg)?;
                    let a = unheads(&attention(
                        &math::rotate(&q, &cos, &sin)?,
                        &math::rotate(&k, &cos, &sin)?,
                        &v,
                        0,
                    )?)?;
                    let ff = gated(&projected.narrow(2, 3 * cfg.hidden, 6 * cfg.hidden)?)?;
                    &all + b
                        .output
                        .forward(&Tensor::cat(&[a, ff], 2)?)?
                        .broadcast_mul(&ms[2])?
                },
            )?;
        }
        let mf = chunks(self.final_mod.forward(&activated)?, 2)?;
        self.final_proj
            .forward(&modulate(&all.narrow(1, nt, h * w)?, &mf[0], &mf[1])?)
    }
    pub fn residency(&self) -> oaiy_engine::json::Json {
        oaiy_engine::json::Json::obj([
            ("double", self.double.report()),
            ("single", self.single.report()),
        ])
    }
}
