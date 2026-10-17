//! A distillation of MOSS-Image to a few passes without guidance, as data-space DMD makes one (arXiv 2610.12102): the
//! transformer as it is, with low-rank factors beside each block's six projections (q, k, v, o, `ffn_up_gate` and
//! `ffn_down`), the time embedding and the two modulations' biases in place of its own, and a sampler of its own. From
//! fixed noise `z`, each pass runs the transformer at the next of its step codes on `z + Q x0` (`Q` its latent mix, `x0`
//! the clean estimate so far, zero at first), and the estimate becomes `z - v`. Fewer passes than it has codes are its
//! first ones (each was trained). Its factors' change to a weight is a fraction of the weight's own rounding (bf16's,
//! f16's or Q8_0's), so they are kept beside the weights and added as the transformer runs, never merged into them.
//!
//! Read from one file holding all of it (`layers.{i}.proj.{module}.lora_A` `[rank, in]` and `.lora_B` `[out, rank]`,
//! the replacements, `latent_mix.Q` and `step_codes`), or from a folder of three: `lora.safetensors` (the factors
//! kohya's way: `diffusion_model.layers.{i}.proj.{module}.lora_down.weight`, `.lora_up.weight` and `.alpha`),
//! `time_embedding.safetensors` and `dataspace.safetensors`.
use crate::common::weights::Weights;
use candle_core::{bail, DType, Device, Result};
use std::path::{Path, PathBuf};

/// The projections it adapts in each block.
pub const MODULES: [&str; 6] = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"];
/// The transformer's tensors it replaces: the time embedding's two layers and the two modulations' biases.
pub const REPLACES: [&str; 6] = ["time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias", "shared_block_modulation.bias", "video_head_modulation.bias"];
/// The one file that holds all of it, in a folder that has it; else the folder's three.
const WHOLE: &str = "generator.safetensors";
const PARTS: [&str; 3] = ["lora.safetensors", "time_embedding.safetensors", "dataspace.safetensors"];

/// A block's projection's factors: `a` (`[rank, input]`) and `b` (`[out, rank]`, its scale in it).
pub struct Factors {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub out: usize,
    pub input: usize,
}

pub struct Distillation {
    files: Vec<(PathBuf, Weights)>,
    /// The transformer's time at each pass (0 to 1000).
    pub codes: Vec<f32>,
    /// The latent mix `Q` (`[channels, channels]`, an output channel a row).
    pub mix: Vec<f32>,
    pub channels: usize,
}

impl Distillation {
    /// The distillation at `path`: a file holding all of it, or a folder with that file or the three parts.
    pub fn open(path: &Path) -> Result<Self> {
        let paths: Vec<PathBuf> = if path.is_dir() {
            if path.join(WHOLE).is_file() {
                vec![path.join(WHOLE)]
            } else {
                let parts: Vec<PathBuf> = PARTS.iter().map(|p| path.join(p)).collect();
                if let Some(gone) = parts.iter().find(|p| !p.is_file()) {
                    bail!("{} holds no MOSS-Image distillation: no {WHOLE}, and no {}", path.display(), gone.display());
                }
                parts
            }
        } else {
            vec![path.to_path_buf()]
        };
        let files = paths.into_iter().map(|p| Weights::open(&p).map(|w| (p, w))).collect::<Result<Vec<_>>>()?;
        let mut d = Self { files, codes: Vec::new(), mix: Vec::new(), channels: 0 };
        let (Some(mix), Some(codes)) = (d.values("latent_mix.Q")?, d.values("step_codes")?) else {
            bail!("{} is not a MOSS-Image distillation: it has no latent_mix.Q and step_codes", path.display());
        };
        let channels = (mix.0.len() as f64).sqrt() as usize;
        if mix.1 != [channels, channels] || codes.0.is_empty() || codes.0.iter().any(|c| !(0. ..=1000.).contains(c)) {
            bail!("{}: a latent mix of shape {:?} and step codes {:?}", path.display(), mix.1, codes.0);
        }
        (d.mix, d.codes, d.channels) = (mix.0, codes.0, channels);
        Ok(d)
    }

    /// The file holding `name`.
    fn holding(&self, name: &str) -> Option<usize> {
        self.files.iter().position(|(_, w)| w.has(name))
    }

    /// `name`'s values and shape, where a file holds it.
    fn values(&mut self, name: &str) -> Result<Option<(Vec<f32>, Vec<usize>)>> {
        let Some(i) = self.holding(name) else { return Ok(None) };
        let w = &mut self.files[i].1;
        let shape = w.shape(name)?;
        Ok(Some((w.tensor(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?, shape)))
    }

    /// The file holding `name` (one of [`REPLACES`]), to read it from in place of the transformer's.
    pub fn replacing(&mut self, name: &str) -> Option<&mut Weights> {
        self.holding(name).map(|i| &mut self.files[i].1)
    }

    /// Block `block`'s factors for `module` (one of [`MODULES`]), `b` scaled by alpha over the rank where the file says
    /// an alpha (kohya's); None where it adapts no such projection.
    pub fn factors(&mut self, block: usize, module: &str) -> Result<Option<Factors>> {
        let (one, kohya) = (format!("layers.{block}.proj.{module}"), format!("diffusion_model.layers.{block}.proj.{module}"));
        let layouts = [
            (format!("{one}.lora_A"), format!("{one}.lora_B"), None),
            (format!("{one}.lora_A.weight"), format!("{one}.lora_B.weight"), Some(format!("{one}.alpha"))),
            (format!("{kohya}.lora_down.weight"), format!("{kohya}.lora_up.weight"), Some(format!("{kohya}.alpha"))),
        ];
        for (a, b, alpha) in layouts {
            let (Some((a, a_shape)), Some((mut b, b_shape))) = (self.values(&a)?, self.values(&b)?) else { continue };
            let (&[rank, input], &[out, rank_b]) = (a_shape.as_slice(), b_shape.as_slice()) else {
                bail!("layers.{block}.proj.{module}: factors of shapes {a_shape:?} and {b_shape:?}");
            };
            if rank != rank_b || rank == 0 {
                bail!("layers.{block}.proj.{module}: factors of ranks {rank} and {rank_b}");
            }
            if let Some(alpha) = alpha.map(|k| self.values(&k)).transpose()?.flatten() {
                let scale = alpha.0.first().copied().unwrap_or(rank as f32) / rank as f32;
                b.iter_mut().for_each(|v| *v *= scale);
            }
            return Ok(Some(Factors { a, b, rank, out, input }));
        }
        Ok(None)
    }

    /// The sampler: from noise `z` (`[tokens, channels]`, a token a row), `passes` of `predict(pass, x, time)` (the
    /// transformer's velocity at its time), and the last clean estimate, the picture's latent.
    pub fn sample(&self, z: &[f32], passes: usize, mut predict: impl FnMut(usize, &[f32], f32) -> Result<Vec<f32>>) -> Result<Vec<f32>> {
        let mut x0 = vec![0.; z.len()];
        for (pass, &time) in self.codes[..passes.min(self.codes.len())].iter().enumerate() {
            let v = predict(pass, &self.mix_in(z, &x0), time)?;
            if v.len() != z.len() {
                bail!("a pass's velocity of {} values for {}", v.len(), z.len());
            }
            x0 = z.iter().zip(&v).map(|(z, v)| z - v).collect();
        }
        Ok(x0)
    }

    /// A pass's input: each token's `z + Q x0` (`[tokens, channels]`, a token a row).
    pub fn mix_in(&self, z: &[f32], x0: &[f32]) -> Vec<f32> {
        let c = self.channels;
        let mut x = z.to_vec();
        for (row, est) in x.chunks_exact_mut(c).zip(x0.chunks_exact(c)) {
            for (o, v) in row.iter_mut().enumerate() {
                *v += self.mix[o * c..(o + 1) * c].iter().zip(est).map(|(q, e)| q * e).sum::<f32>();
            }
        }
        x
    }
}
