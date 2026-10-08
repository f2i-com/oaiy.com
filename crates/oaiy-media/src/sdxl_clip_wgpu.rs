//! SDXL's two CLIP text encoders on WebGPU: CLIP-L (OpenAI's, Hugging Face's names, QuickGELU) and CLIP-G (OpenCLIP's
//! bigG, its q, k and v one matrix, exact GELU), each a pre-norm transformer with causal attention over the prompt's
//! 77 tokens, as chains of the device's ops. A prompt's encoding took Candle 2.5 s on the CPU (the prompt and the
//! negative one through both), and the encoders 1.9 s to load there: more than a picture's steps take the GPU.
//!
//! A layer's q, k and v are one matrix here (CLIP-G's is stored so; CLIP-L's three are put one below another): its
//! output's columns are a token's q, then its k and v, which is the row the device's attention takes of a cache.
//! The token and position tables, the last norm and CLIP-G's projection of its pooled token stay on the host.
use crate::sdxl::config::CLIPConfig;
use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs_wgpu::WgpuBackend;

/// A prompt's tokens.
const TOKENS: usize = 77;
const EPS: f32 = 1e-5;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

fn values(w: &mut Weights, key: &str) -> Result<Vec<f32>> {
    w.tensor(key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn upload(gpu: &WgpuBackend, v: &[f32]) -> DeviceVec {
    let d = gpu.vec(v.len());
    gpu.upload(&d, v);
    d
}

/// `keys`' matrices one below another as f16 (`[n, k]`), with their biases one after another.
struct Lin {
    w: DeviceVec,
    b: DeviceVec,
    n: usize,
    k: usize,
}

impl Lin {
    /// `weights` (each `[_, k]`, or one matrix of several) with `biases`, from their F16 bytes where the file keeps
    /// them so.
    fn load(w: &mut Weights, gpu: &WgpuBackend, weights: &[String], biases: &[String]) -> Result<Self> {
        let raw: Vec<Option<(Vec<u8>, Vec<usize>)>> = weights.iter().map(|key| w.raw_f16(key)).collect::<Result<_>>()?;
        let (words, n, k) = if raw.iter().all(|r| r.is_some()) {
            let (mut bytes, mut n, mut k) = (Vec::new(), 0, 0);
            for (key, (b, dims)) in weights.iter().zip(raw.into_iter().flatten()) {
                let &[rows, cols] = dims.as_slice() else { candle_core::bail!("{key}: a matrix of shape {dims:?}") };
                if k != 0 && cols != k {
                    candle_core::bail!("{key}: {cols} wide beside {k}");
                }
                (n, k) = (n + rows, cols);
                bytes.extend(b);
            }
            (crate::wgpu_weights::f16_words_halves(&bytes), n, k)
        } else {
            let (mut all, mut n, mut k) = (Vec::new(), 0, 0);
            for key in weights {
                let t = w.tensor(key, &Device::Cpu, DType::F32)?;
                let (rows, cols) = t.dims2()?;
                if k != 0 && cols != k {
                    candle_core::bail!("{key}: {cols} wide beside {k}");
                }
                (n, k) = (n + rows, cols);
                all.extend(t.flatten_all()?.to_vec1::<f32>()?);
            }
            (crate::wgpu_weights::f16_words_f32(&all), n, k)
        };
        let words = words.ok_or_else(|| err(format!("{}: a weight past f16's range", weights[0])))?;
        let mut bias = Vec::new();
        for key in biases {
            bias.extend(values(w, key)?);
        }
        if bias.len() != n {
            candle_core::bail!("{}: {} biases for {n} rows", weights[0], bias.len());
        }
        Ok(Lin { w: upload(gpu, &words), b: upload(gpu, &bias), n, k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows (`x`'s values as they are: a text encoder's states may pass f16's range).
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows_f32(&self.w, self.n, self.k, x, y, rows);
        r.add_bias_rows(y, &self.b, rows, self.n);
    }
}

/// A layer norm's weight and bias as the modulation a modulated norm takes: `[weight - 1, bias]`.
fn norm(w: &mut Weights, gpu: &WgpuBackend, name: &str) -> Result<DeviceVec> {
    let mut v = values(w, &format!("{name}.weight"))?;
    v.iter_mut().for_each(|x| *x -= 1.0);
    v.extend(values(w, &format!("{name}.bias"))?);
    Ok(upload(gpu, &v))
}

struct Layer {
    ln1: DeviceVec,
    qkv: Lin,
    out: Lin,
    ln2: DeviceVec,
    fc1: Lin,
    fc2: Lin,
}

/// One encoder on the device.
struct Clip {
    /// The token table (`[vocab, h]`) and the positions' (`[77, h]`), on the host: a prompt takes 77 rows of one.
    tokens: Vec<f32>,
    positions: Vec<f32>,
    /// The layers it runs: all `total` of an encoder with a pooled output (CLIP-G), all but the last of one without
    /// (CLIP-L: SDXL's context is the states at least a layer before its last, and nothing reads the last layer's;
    /// some checkpoints keep values there that are no numbers, which Candle loads and never multiplies by).
    layers: Vec<Layer>,
    total: usize,
    /// The last norm's weight and bias, and CLIP-G's projection (`[h, h]`, applied as `x @ P`), on the host.
    last: (Vec<f32>, Vec<f32>),
    projection: Option<Vec<f32>>,
    h: usize,
    heads: usize,
    ff: usize,
    /// QuickGELU (`x sigmoid(1.702 x)`: CLIP-L's) where set, else the exact GELU (CLIP-G's).
    quick: bool,
    /// 0.702: what `x` is added to itself by for QuickGELU's `1.702 x`.
    more: DeviceVec,
}

impl Clip {
    fn load(w: &mut Weights, gpu: &WgpuBackend, cfg: &CLIPConfig, prefix: &str, open_clip: bool) -> Result<Self> {
        let h = cfg.hidden_size;
        let p = |key: &str| format!("{prefix}{key}");
        let run = cfg.num_hidden_layers - usize::from(!open_clip);
        let mut layers = Vec::with_capacity(run);
        for i in 0..run {
            let layer = if open_clip {
                let b = p(&format!("transformer.resblocks.{i}"));
                Layer {
                    ln1: norm(w, gpu, &format!("{b}.ln_1"))?,
                    qkv: Lin::load(w, gpu, &[format!("{b}.attn.in_proj_weight")], &[format!("{b}.attn.in_proj_bias")])?,
                    out: Lin::load(w, gpu, &[format!("{b}.attn.out_proj.weight")], &[format!("{b}.attn.out_proj.bias")])?,
                    ln2: norm(w, gpu, &format!("{b}.ln_2"))?,
                    fc1: Lin::load(w, gpu, &[format!("{b}.mlp.c_fc.weight")], &[format!("{b}.mlp.c_fc.bias")])?,
                    fc2: Lin::load(w, gpu, &[format!("{b}.mlp.c_proj.weight")], &[format!("{b}.mlp.c_proj.bias")])?,
                }
            } else {
                let b = p(&format!("text_model.encoder.layers.{i}"));
                let three = |what: &str| ["q_proj", "k_proj", "v_proj"].map(|n| format!("{b}.self_attn.{n}.{what}")).to_vec();
                Layer {
                    ln1: norm(w, gpu, &format!("{b}.layer_norm1"))?,
                    qkv: Lin::load(w, gpu, &three("weight"), &three("bias"))?,
                    out: Lin::load(w, gpu, &[format!("{b}.self_attn.out_proj.weight")], &[format!("{b}.self_attn.out_proj.bias")])?,
                    ln2: norm(w, gpu, &format!("{b}.layer_norm2"))?,
                    fc1: Lin::load(w, gpu, &[format!("{b}.mlp.fc1.weight")], &[format!("{b}.mlp.fc1.bias")])?,
                    fc2: Lin::load(w, gpu, &[format!("{b}.mlp.fc2.weight")], &[format!("{b}.mlp.fc2.bias")])?,
                }
            };
            if layer.qkv.n != 3 * h || layer.qkv.k != h || layer.out.n != h || layer.fc1.n != cfg.intermediate_size || layer.fc2.n != h {
                candle_core::bail!("{prefix}: layer {i}'s matrices are not a CLIP of {h}'s");
            }
            layers.push(layer);
        }
        let (tokens, positions, last, projection) = if open_clip {
            (p("token_embedding.weight"), p("positional_embedding"), p("ln_final"), Some(values(w, &p("text_projection"))?))
        } else {
            (p("text_model.embeddings.token_embedding.weight"), p("text_model.embeddings.position_embedding.weight"), p("text_model.final_layer_norm"), None)
        };
        let positions = values(w, &positions)?;
        if positions.len() < TOKENS * h || cfg.max_position_embeddings < TOKENS || h % cfg.num_attention_heads != 0 {
            candle_core::bail!("{prefix}: positions for fewer than {TOKENS} tokens of {h}");
        }
        Ok(Clip {
            tokens: values(w, &tokens)?,
            positions,
            layers,
            total: cfg.num_hidden_layers,
            last: (values(w, &format!("{last}.weight"))?, values(w, &format!("{last}.bias"))?),
            projection,
            h,
            heads: cfg.num_attention_heads,
            ff: cfg.intermediate_size,
            quick: !open_clip,
            more: upload(gpu, &[0.702]),
        })
    }

    /// `ids` (77 of them) through the encoder: the states after all but its last `skip` layers (`[77, h]`: SDXL's
    /// context is the last but one's), and, for an encoder with a pooled output, the last layer's (`[77, h]`, before
    /// the last norm; else the same states again).
    fn forward(&self, gpu: &WgpuBackend, ids: &[u32], skip: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let (h, n) = (self.h, self.total);
        if ids.len() != TOKENS || ids.iter().any(|&t| (t as usize + 1) * h > self.tokens.len()) {
            candle_core::bail!("a prompt of {} tokens for an encoder of {TOKENS}", ids.len());
        }
        let x0: Vec<f32> = ids.iter().enumerate().flat_map(|(at, &t)| (0..h).map(move |c| (at, t as usize, c))).map(|(at, t, c)| self.tokens[t * h + c] + self.positions[at * h + c]).collect();
        let keep_at = n - skip.clamp(1, n);
        let hd = h / self.heads;
        let v = |len: usize| gpu.vec(len);
        let (x, normed, fused, q, kv, y, f, fa, kept) = (upload(gpu, &x0), v(TOKENS * h), v(TOKENS * 3 * h), v(TOKENS * h), v(TOKENS * 2 * h), v(TOKENS * h), v(TOKENS * self.ff), v(TOKENS * self.ff), v(TOKENS * h));
        // (QuickGELU's `1.702 x`: a dispatch may not read the buffer it writes)
        let scaled = v(if self.quick { TOKENS * self.ff } else { 1 });
        let att = v(gpu.attention_rows_out_len(TOKENS, self.heads, hd, TOKENS));
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        // (an encoder without a pooled output stops at the states it is asked for)
        let run = if self.projection.is_some() { self.layers.len() } else { keep_at };
        for (i, l) in self.layers.iter().enumerate().take(run) {
            if i == keep_at {
                r.copy(&x, 0, &kept, 0, TOKENS * h);
            }
            // each token over itself and those before it
            r.layernorm_mod_rows(&x, &normed, TOKENS, h, &l.ln1, 0, Some(h), EPS);
            l.qkv.run(r, &normed, &fused, TOKENS);
            r.copy_cols(&fused, &q, TOKENS, h, 3 * h, 0);
            r.copy_cols(&fused, &kv, TOKENS, 2 * h, 3 * h, h);
            r.attention_rows(&q, &kv, &att, TOKENS, self.heads, self.heads, hd, 0, None, 1.0 / (hd as f32).sqrt());
            l.out.run(r, &att, &y, TOKENS);
            r.add(&x, &y);
            r.layernorm_mod_rows(&x, &normed, TOKENS, h, &l.ln2, 0, Some(h), EPS);
            l.fc1.run(r, &normed, &f, TOKENS);
            if self.quick {
                r.copy(&f, 0, &scaled, 0, TOKENS * self.ff);
                r.axpy_at(&scaled, &f, &self.more, 0, TOKENS * self.ff);
                r.mul_sigmoid(&f, &scaled, &fa, TOKENS * self.ff);
            } else {
                r.gelu_erf(&f, &fa, TOKENS * self.ff);
            }
            l.fc2.run(r, &fa, &y, TOKENS);
            r.add(&x, &y);
        }
        if keep_at >= run {
            r.copy(&x, 0, &kept, 0, TOKENS * h);
        }
        r.read(&kept);
        r.read(&x);
        let mut got = rec.finish().into_iter();
        let (kept, last) = (got.next().ok_or_else(|| err("the context was not read"))?, got.next().ok_or_else(|| err("the last states were not read"))?);
        Ok((kept[..TOKENS * h].to_vec(), last[..TOKENS * h].to_vec()))
    }

    /// The last states' token `at` through the last norm and the projection: CLIP-G's pooled output.
    fn pooled(&self, last: &[f32], at: usize) -> Result<Vec<f32>> {
        let h = self.h;
        let Some(projection) = &self.projection else { candle_core::bail!("an encoder without a projection") };
        let row = &last[at * h..(at + 1) * h];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / h as f64;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / h as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        let normed: Vec<f64> = row.iter().zip(&self.last.0).zip(&self.last.1).map(|((v, w), b)| (*v as f64 - mean) * inv * *w as f64 + *b as f64).collect();
        // (`x @ P`: output j the sum over i of x[i] P[i][j])
        Ok((0..h).map(|j| normed.iter().enumerate().map(|(i, x)| x * projection[i * h + j] as f64).sum::<f64>() as f32).collect())
    }
}

/// SDXL's two text encoders on a WebGPU device.
pub struct WgpuClips {
    gpu: WgpuBackend,
    l: Clip,
    g: Clip,
}

impl WgpuClips {
    /// CLIP-L under `l_prefix` and CLIP-G under `g_prefix` of the checkpoint `w`, on `gpu`.
    pub fn load(w: &mut Weights, l_prefix: &str, g_prefix: &str, gpu: WgpuBackend) -> Result<Self> {
        let l = Clip::load(w, &gpu, &CLIPConfig::clip_l_14_336(), l_prefix, false)?;
        let g = Clip::load(w, &gpu, &CLIPConfig::open_clip_g_14_laion2b(), g_prefix, true)?;
        Ok(Self { gpu, l, g })
    }

    /// A prompt as both encoders tokenize it (`ids_l`, `ids_g`: 77 each; `eot` CLIP-G's end token's place), as
    /// `text_encoder::dual_encode` gives it: the context `(1, 77, 2048)` (CLIP-L's 768 then CLIP-G's 1,280, each the
    /// states `skip` layers from its last) and CLIP-G's pooled output `(1, 1280)`.
    pub fn encode(&self, ids_l: &[u32], ids_g: &[u32], eot: usize, skip: usize) -> Result<(Tensor, Tensor)> {
        let (cl, _) = self.l.forward(&self.gpu, ids_l, skip)?;
        let (cg, last) = self.g.forward(&self.gpu, ids_g, skip)?;
        if eot >= TOKENS {
            candle_core::bail!("an end token at {eot} of {TOKENS}");
        }
        let (hl, hg) = (self.l.h, self.g.h);
        let context: Vec<f32> = (0..TOKENS).flat_map(|t| cl[t * hl..(t + 1) * hl].iter().chain(&cg[t * hg..(t + 1) * hg]).copied()).collect();
        let pooled = self.g.pooled(&last, eot)?;
        Ok((Tensor::from_vec(context, (1, TOKENS, hl + hg), &Device::Cpu)?, Tensor::from_vec(pooled, (1, hg), &Device::Cpu)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    /// The encoders on WebGPU give Candle's on the CPU in f32 (an SDXL checkpoint, `OAIY_SDXL_CHECKPOINT`; a prompt's
    /// tokens made up, CLIP-L's padded with its end token and CLIP-G's with zeros, as the worker pads them): the
    /// context and the pooled output, each's error against Candle's (relative RMS), at the default skip and one more,
    /// and what each takes.
    #[test]
    #[ignore = "needs an SDXL checkpoint (OAIY_SDXL_CHECKPOINT) and a WebGPU adapter"]
    fn the_webgpu_encoders_are_candles() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_SDXL_CHECKPOINT") else { return Ok(()) };
        let (lp, gp) = ("conditioner.embedders.0.transformer.", "conditioner.embedders.1.model.");
        let mut w = Weights::open(std::path::Path::new(&path))?;
        let gpu = WgpuBackend::nth(0, None).map_err(err)?;
        let t = std::time::Instant::now();
        let clips = WgpuClips::load(&mut w, lp, gp, gpu)?;
        let loaded = t.elapsed().as_secs_f64();
        // Candle's, from the same tensors
        let t = std::time::Instant::now();
        let mut part = |prefix: &str| -> Result<std::collections::HashMap<String, Tensor>> {
            let mut all = std::collections::HashMap::new();
            for name in w.names() {
                if let Some(key) = name.strip_prefix(prefix) {
                    if !key.ends_with("position_ids") && key != "logit_scale" {
                        all.insert(key.to_owned(), w.tensor(&name, &Device::Cpu, DType::F32)?);
                    }
                }
            }
            Ok(all)
        };
        let cl = crate::sdxl::text_encoder::ClipL::load(&CLIPConfig::clip_l_14_336(), VarBuilder::from_tensors(part(lp)?, DType::F32, &Device::Cpu))?;
        let cg = crate::sdxl::text_encoder::ClipG::load(&CLIPConfig::open_clip_g_14_laion2b(), VarBuilder::from_tensors(part(gp)?, DType::F32, &Device::Cpu))?;
        let candle_loaded = t.elapsed().as_secs_f64();
        let words: Vec<u32> = (0..23u32).map(|i| 1000 + (i * 7919) % 40000).collect();
        let ids = |pad: u32| -> Vec<u32> {
            let mut v = vec![49406];
            v.extend(&words);
            v.push(49407);
            v.resize(TOKENS, pad);
            v
        };
        let (il, ig, eot) = (ids(49407), ids(0), words.len() + 1);
        let relative = |got: &Tensor, want: &Tensor| -> Result<f64> {
            let (g, w) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
            assert_eq!(g.len(), w.len());
            let e: f64 = g.iter().zip(&w).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
            Ok((e / w.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt())
        };
        let mut worst = 0f64;
        for skip in [1usize, 2] {
            let t = std::time::Instant::now();
            let want = crate::sdxl::text_encoder::dual_encode(&cl, &cg, &Tensor::from_vec(il.clone(), (1, TOKENS), &Device::Cpu)?, &Tensor::from_vec(ig.clone(), (1, TOKENS), &Device::Cpu)?, &[eot], skip)?;
            let host = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let (context, pooled) = clips.encode(&il, &ig, eot, skip)?;
            let ours = t.elapsed().as_secs_f64();
            let (ec, ep) = (relative(&context, &want.context)?, relative(&pooled, &want.pooled)?);
            eprintln!("skip {skip}: the context {ec:.5} from Candle's (relative RMS), the pooled output {ep:.5}; {host:.2} s on the host, {ours:.3} s on WebGPU");
            worst = worst.max(ec).max(ep);
        }
        eprintln!("loaded in {loaded:.2} s on WebGPU, {candle_loaded:.2} s by Candle");
        assert!(worst < 0.01, "the encoders on WebGPU: {worst} from Candle's");
        Ok(())
    }
}
