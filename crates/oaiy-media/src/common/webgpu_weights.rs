//! Weights onto a WebGPU device fast: a checkpoint's stored bytes (BF16, GGUF's blocks, fp8 and int8 codes) converted to f16 or Q8_0 on
//! every core, straight from those bytes (Candle's decode to f32 on one core and the copies after it were most of a
//! model's load), and written into the card's staging memory from every core (the chain's upload).
use crate::common::lora::Loras;
use crate::common::weights::{Raw, Weights};
use candle_core::{quantized::GgmlDType, DType, Device, Result};
use ggml_rs::{DeviceChain, DeviceVec};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// The GPU a picture model's text encoder runs on: the computer's own where `OAIY_TEXT_ENCODER_GPU` is `own` (a Mac's,
/// beside the eGPU's card the worker makes its pictures on: the encoder's layers are not sent over the cable one at a
/// time, and the card's memory is left to the pictures' models, kept there between pictures), else GPU `device` as the
/// worker's other models take it. The states it makes go to the picture's model through the host either way.
pub fn text_encoder_gpu(device: usize) -> Result<ggml_rs_wgpu::WgpuBackend> {
    let own = std::env::var("OAIY_TEXT_ENCODER_GPU").is_ok_and(|v| v.trim().eq_ignore_ascii_case("own"));
    if own { ggml_rs_wgpu::WgpuBackend::own(None) } else { ggml_rs_wgpu::WgpuBackend::nth(device, None) }.map_err(err)
}

/// `f` over `src` and `dst` in matching runs on every core: `per` of `src`'s items to `per_out` of `dst`'s.
pub(crate) fn on_cores<S: Sync, T: Send>(src: &[S], per: usize, dst: &mut [T], per_out: usize, f: impl Fn(&[S], &mut [T]) + Sync) {
    on_cores_at(src, per, dst, per_out, |_, a, b| f(a, b));
}

/// [`on_cores`], `f` told where its run starts too (the index in `src` of its first item).
pub(crate) fn on_cores_at<S: Sync, T: Send>(src: &[S], per: usize, dst: &mut [T], per_out: usize, f: impl Fn(usize, &[S], &mut [T]) + Sync) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let each = src.len().div_ceil(per).div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (i, (a, b)) in src.chunks(each * per).zip(dst.chunks_mut(each * per_out)).enumerate() {
            let f = &f;
            s.spawn(move || f(i * each * per, a, b));
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

/// An f16 weight as the card keeps it (two halves a word, the first low) as Q8_0's blocks.
pub(crate) fn q8_0_f16_words(words: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; words.len() / 16 * 34];
    on_cores(words, 16, &mut out, 34, |src, dst| {
        let mut x = [0f32; 32];
        for (w, q) in src.chunks_exact(16).zip(dst.chunks_exact_mut(34)) {
            for (pair, two) in x.chunks_exact_mut(2).zip(w) {
                let bits = two.to_bits();
                pair[0] = half::f16::from_bits(bits as u16).to_f32();
                pair[1] = half::f16::from_bits((bits >> 16) as u16).to_f32();
            }
            ggml_quants::q8_0::quantize(&x, q);
        }
    });
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

/// A weight kept as a code a value (a byte: fp8, int8) as Q8_0's blocks, made from its codes on every core, `group`
/// of them at a time (a multiple of 32 that divides them): `values(at, codes, x)` puts in `x` the values of `codes`,
/// the first of them at `at` (Candle's decode to f32 on one core and the copies after it took most of an fp8
/// transformer's load).
pub(crate) fn q8_0_codes(codes: &[u8], group: usize, values: impl Fn(usize, &[u8], &mut [f32]) + Sync) -> Vec<u8> {
    let mut out = vec![0u8; codes.len() / 32 * 34];
    on_cores_at(codes, group, &mut out, group / 32 * 34, |at, src, dst| {
        let mut x = vec![0f32; group];
        for (i, (c, q)) in src.chunks_exact(group).zip(dst.chunks_exact_mut(group / 32 * 34)).enumerate() {
            values(at + i * group, c, &mut x);
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

/// Runs `f` (each run's start in `src`, its words, its range kept) on every core: None where a value was past f16's
/// range.
fn checked<S: Sync>(src: &[S], per: usize, words: usize, per_out: usize, f: impl Fn(usize, &[S], &mut [f32], &mut bool) + Sync) -> Option<Vec<f32>> {
    let mut out = vec![0f32; words];
    let ok = std::sync::atomic::AtomicBool::new(true);
    on_cores_at(src, per, &mut out, per_out, |at, s, d| {
        let mut fine = true;
        f(at, s, d, &mut fine);
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
    checked(bytes, 4, bytes.len() / 4, 1, |_, src, dst, fine| {
        for (b, w) in src.chunks_exact(4).zip(dst.iter_mut()) {
            *w = f16_word(bf16(&b[..2]), bf16(&b[2..]), fine);
        }
    })
}

/// F16 bytes (little-endian halves) as f16 words: the bytes as they are, two halves a word; None where a half is an
/// infinity or not a number (past f16's range, as the other ways in say it) or the halves are odd in number.
pub fn f16_words_halves(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    checked(bytes, 4, bytes.len() / 4, 1, |_, src, dst, fine| {
        for (b, w) in src.chunks_exact(4).zip(dst.iter_mut()) {
            let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            *fine &= word & 0x7c00 != 0x7c00 && word & 0x7c00_0000 != 0x7c00_0000;
            *w = f32::from_bits(word);
        }
    })
}

/// [`f16_words`] of values already f32.
pub fn f16_words_f32(values: &[f32]) -> Option<Vec<f32>> {
    if values.len() % 2 != 0 {
        return None;
    }
    checked(values, 2, values.len() / 2, 1, |_, src, dst, fine| {
        for (p, w) in src.chunks_exact(2).zip(dst.iter_mut()) {
            *w = f16_word(p[0], p[1], fine);
        }
    })
}

/// [`f16_words`] of a weight kept as a code a value ([`q8_0_codes`]'s `values`; `group` even).
pub fn f16_words_codes(codes: &[u8], group: usize, values: impl Fn(usize, &[u8], &mut [f32]) + Sync) -> Option<Vec<f32>> {
    if codes.len() % group != 0 || group % 2 != 0 {
        return None;
    }
    checked(codes, group, codes.len() / 2, group / 2, |at, src, dst, fine| {
        let mut x = vec![0f32; group];
        for (i, (c, w)) in src.chunks_exact(group).zip(dst.chunks_exact_mut(group / 2)).enumerate() {
            values(at + i * group, c, &mut x);
            for (p, w) in x.chunks_exact(2).zip(w.iter_mut()) {
                *w = f16_word(p[0], p[1], fine);
            }
        }
    })
}

/// GGUF's type as the block decoders name it.
pub(crate) fn ggml(t: GgmlDType) -> Option<ggml_quants::GgmlType> {
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
    checked(bytes, row, rows * k / 2, k / 2, |_, src, dst, fine| {
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
    let v = match w.w4a8(&key)? {
        // ComfyUI's W4A8 decoded on the device (on the CPU some 8 s of a Qwen3-VL 8B's load)
        Some(q) => {
            let words = |bytes: &[u8]| -> Vec<f32> { bytes.chunks(4).map(|c| { let mut b = [0u8; 4]; b[..c.len()].copy_from_slice(c); f32::from_bits(u32::from_le_bytes(b)) }).collect() };
            let (codes, rel) = (words(&q.codes), words(&q.rel));
            let (cd, rd, ch, bk, v) = (gpu.vec(codes.len()), gpu.vec(rel.len()), gpu.vec(q.rows), gpu.vec(16), gpu.vec(q.rows * q.cols / 2));
            gpu.upload(&cd, &codes);
            gpu.upload(&rd, &rel);
            gpu.upload(&ch, &q.channel);
            gpu.upload(&bk, &q.book);
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            rec.w4a8_f16(&cd, &rd, &ch, &bk, q.rows, q.cols, q.rotation, &v);
            rec.finish();
            v
        }
        None => {
            let words = match w.raw(&key)? {
                Some(Raw::Bf16(b)) => f16_words(&b),
                Some(Raw::Ggml(t, b)) => f16_words_ggml(t, &b, n, k),
                None => match w.raw_f16(&key)? {
                    // (an F16 file's: its bytes as they are)
                    Some((b, _)) => f16_words_halves(&b),
                    None => f16_words_f32(&w.tensor(&key, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?),
                },
            }
            .ok_or_else(|| err(format!("{name}: a weight past f16's range")))?;
            let v = gpu.vec(words.len());
            gpu.upload(&v, &words);
            v
        }
    };
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

#[cfg(test)]
mod f16_tests {
    use super::*;

    /// F16 bytes as words are what the same values give as f32 (every half's bits kept, the subnormals and both
    /// zeros among them), and an infinity or a NaN is refused as a value past f16's range is.
    #[test]
    fn f16_bytes_are_their_values_words() {
        let halves: Vec<u16> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 16) as u16).filter(|h| h & 0x7c00 != 0x7c00).chain([0, 0x8000, 1, 0x8001, 0x03ff, 0x7bff, 0xfbff]).collect();
        let halves = &halves[..halves.len() / 2 * 2];
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let values: Vec<f32> = halves.iter().map(|h| half::f16::from_bits(*h).to_f32()).collect();
        let (got, want) = (f16_words_halves(&bytes).unwrap(), f16_words_f32(&values).unwrap());
        assert_eq!(got.iter().map(|w| w.to_bits()).collect::<Vec<_>>(), want.iter().map(|w| w.to_bits()).collect::<Vec<_>>());
        for bad in [0x7c00u16, 0xfc00, 0x7e00] {
            let bytes: Vec<u8> = [1u16, bad].iter().flat_map(|h| h.to_le_bytes()).collect();
            assert!(f16_words_halves(&bytes).is_none(), "{bad:#x}");
        }
    }
}
