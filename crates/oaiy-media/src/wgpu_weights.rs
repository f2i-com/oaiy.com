//! Weights onto a WebGPU device fast: a checkpoint's stored bytes (BF16, GGUF's blocks) converted to f16 or Q8_0 on
//! every core, straight from those bytes (Candle's decode to f32 on one core and the copies after it were most of a
//! model's load), and written into the card's staging memory from every core (the chain's upload).
use crate::lora::Loras;
use crate::weights::{Raw, Weights};
use candle_core::{quantized::GgmlDType, DType, Device, Result};
use ggml_rs::{DeviceChain, DeviceVec};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// `f` over `src` and `dst` in matching runs on every core: `per` of `src`'s items to `per_out` of `dst`'s.
pub(crate) fn on_cores<S: Sync, T: Send>(src: &[S], per: usize, dst: &mut [T], per_out: usize, f: impl Fn(&[S], &mut [T]) + Sync) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let each = src.len().div_ceil(per).div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (a, b) in src.chunks(each * per).zip(dst.chunks_mut(each * per_out)) {
            let f = &f;
            s.spawn(move || f(a, b));
        }
    });
}

pub(crate) fn bf16(b: &[u8]) -> f32 {
    f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
}

/// A weight's values as Q8_0's blocks (32 values a block).
pub(crate) fn q8_0(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() / 32 * 34];
    on_cores(values, 32, &mut out, 34, ggml_quants::q8_0::quantize);
    out
}

/// A BF16 weight's bytes as Q8_0's blocks.
pub(crate) fn q8_0_bf16(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bytes.len() / 64 * 34];
    on_cores(bytes, 64, &mut out, 34, |src, dst| {
        let mut x = [0f32; 32];
        for (b, q) in src.chunks_exact(64).zip(dst.chunks_exact_mut(34)) {
            for (v, two) in x.iter_mut().zip(b.chunks_exact(2)) {
                *v = bf16(two);
            }
            ggml_quants::q8_0::quantize(&x, q);
        }
    });
    out
}

/// A pair's f16 word (the first value low), as the chain's f16 matrices hold them; false where a value is past f16's
/// range.
fn f16_word(lo: f32, hi: f32, ok: &mut bool) -> f32 {
    *ok &= lo.abs() <= 65504.0 && hi.abs() <= 65504.0;
    f32::from_bits(half::f16::from_f32(lo).to_bits() as u32 | (half::f16::from_f32(hi).to_bits() as u32) << 16)
}

/// Runs `f` (each run's words, its range kept) on every core: None where a value was past f16's range.
fn checked<S: Sync>(src: &[S], per: usize, words: usize, per_out: usize, f: impl Fn(&[S], &mut [f32], &mut bool) + Sync) -> Option<Vec<f32>> {
    let mut out = vec![0f32; words];
    let ok = std::sync::atomic::AtomicBool::new(true);
    on_cores(src, per, &mut out, per_out, |s, d| {
        let mut fine = true;
        f(s, d, &mut fine);
        if !fine {
            ok.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    ok.into_inner().then_some(out)
}

/// A BF16 weight's bytes as f16 words, each value rounded to the nearest f16; None where one is past its range.
pub fn f16_words(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    checked(bytes, 4, bytes.len() / 4, 1, |src, dst, fine| {
        for (b, w) in src.chunks_exact(4).zip(dst.iter_mut()) {
            *w = f16_word(bf16(&b[..2]), bf16(&b[2..]), fine);
        }
    })
}

/// [`f16_words`] of values already f32.
pub fn f16_words_f32(values: &[f32]) -> Option<Vec<f32>> {
    if values.len() % 2 != 0 {
        return None;
    }
    checked(values, 2, values.len() / 2, 1, |src, dst, fine| {
        for (p, w) in src.chunks_exact(2).zip(dst.iter_mut()) {
            *w = f16_word(p[0], p[1], fine);
        }
    })
}

/// GGUF's type as the block decoders name it.
fn ggml(t: GgmlDType) -> Option<ggml_quants::GgmlType> {
    use ggml_quants::GgmlType as G;
    Some(match t {
        GgmlDType::F32 => G::F32,
        GgmlDType::F16 => G::F16,
        GgmlDType::BF16 => G::BF16,
        GgmlDType::Q4_0 => G::Q4_0,
        GgmlDType::Q4_1 => G::Q4_1,
        GgmlDType::Q5_0 => G::Q5_0,
        GgmlDType::Q5_1 => G::Q5_1,
        GgmlDType::Q8_0 => G::Q8_0,
        GgmlDType::Q2K => G::Q2_K,
        GgmlDType::Q3K => G::Q3_K,
        GgmlDType::Q4K => G::Q4_K,
        GgmlDType::Q5K => G::Q5_K,
        GgmlDType::Q6K => G::Q6_K,
        _ => return None,
    })
}

/// A GGUF matrix's blocks (`rows` rows of `k`) and one row's bytes, where the block decoders take its type.
fn rows_of(t: GgmlDType, bytes: &[u8], rows: usize, k: usize) -> Option<(ggml_quants::GgmlType, usize)> {
    let g = ggml(t).filter(|g| ggml_quants::is_supported(*g) && k % g.block_size() == 0)?;
    let row = k / g.block_size() * g.type_size();
    (bytes.len() == rows * row && k % 2 == 0).then_some((g, row))
}

/// A GGUF matrix (`rows` rows of `k`) as f16 words: each row decoded and rounded on one of every core.
pub fn f16_words_ggml(t: GgmlDType, bytes: &[u8], rows: usize, k: usize) -> Option<Vec<f32>> {
    let (g, row) = rows_of(t, bytes, rows, k)?;
    checked(bytes, row, rows * k / 2, k / 2, |src, dst, fine| {
        let mut values = vec![0f32; k];
        for (r, w) in src.chunks_exact(row).zip(dst.chunks_exact_mut(k / 2)) {
            if ggml_quants::dequantize(g, r, &mut values).is_err() {
                *fine = false;
                return;
            }
            for (p, word) in values.chunks_exact(2).zip(w.iter_mut()) {
                *word = f16_word(p[0], p[1], fine);
            }
        }
    })
}

/// `name.weight` (`[n, k]`) from `w` as an f16 matrix on `gpu` (its vector, `n`, `k`), from its stored bytes where
/// [`Weights::raw`] gives them, else as [`Weights::tensor`] reads it; each of `lora`'s factors for `name` (`B A`) added
/// on the GPU (the f16 matmul makes `B A`, rounded into the weight as the host's merge rounds it: on the host it took
/// three quarters of Qwen Image's load).
pub fn f16_matrix(w: &mut Weights, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str, lora: &mut Loras) -> Result<(DeviceVec, usize, usize)> {
    let key = format!("{name}.weight");
    let shape = w.shape(&key)?;
    let &[n, k] = shape.as_slice() else { candle_core::bail!("{key}: a matrix of shape {shape:?}") };
    let words = match w.raw(&key)? {
        Some(Raw::Bf16(b)) => f16_words(&b),
        Some(Raw::Ggml(t, b)) => f16_words_ggml(t, &b, n, k),
        None => f16_words_f32(&w.tensor(&key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?),
    }
    .ok_or_else(|| err(format!("{name}: a weight past f16's range")))?;
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
    if !lora.is_empty() {
        for (a, b) in lora.factors(name, n, k, &Device::Cpu, DType::F32)? {
            let rank = a.dim(0)?;
            // `A` transposed (the matmul's weight, `[k, rank]`) as f16, `B`'s rows its tokens
            let at = f16_words_f32(&a.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?).ok_or_else(|| err(format!("{name}: a LoRA factor past f16's range")))?;
            let (ad, bd, delta) = (gpu.vec(at.len()), gpu.vec(n * rank), gpu.vec(n * k));
            gpu.upload(&ad, &at);
            gpu.upload(&bd, &b.flatten_all()?.to_vec1::<f32>()?);
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            rec.matmul_f16_rows(&ad, k, rank, &bd, &delta, n);
            rec.add_f16(&v, &delta, n * k);
            rec.finish();
        }
    }
    Ok((v, n, k))
}
