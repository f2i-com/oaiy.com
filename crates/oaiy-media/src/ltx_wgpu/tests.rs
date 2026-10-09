//! `ltx_wgpu`'s tests.

use super::*;
use candle_core::DType;

/// How long a step of the WebGPU video stream takes (`--ignored --nocapture`, `OAIY_LTX_NVFP4`): a 768x512 clip of
/// 121 frames (16 latent frames of 16 by 24: 6,144 tokens) over the connector's 1,024 context rows.
#[test]
#[ignore = "a timing; needs LTX 2.3's NVFP4 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
fn measure_a_video_step() -> Result<()> {
    let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
    let (frames, h, w, fps, lc) = (16usize, 16usize, 24usize, 24usize, 1024usize);
    let tokens = frames * h * w;
    let latent: Vec<f32> = (0..tokens * 128).map(|i| ((i * 7919 % 2001) as f32 / 1000.0 - 1.0) * 1.5).collect();
    let context: Vec<f32> = (0..lc * D).map(|i| (i * 104729 % 2001) as f32 / 1000.0 - 1.0).collect();
    let mut store = Store::open(std::path::Path::new(&path), 0)?;
    let gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
    let table = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, false), &[20., 2048., 2048.], D, HEADS));
    for i in 0..3 {
        let t = std::time::Instant::now();
        let v = gpu.forward(&latent, tokens, &context, lc, 0.7 - 0.1 * i as f64, &table, (0, 0), h * w, None)?;
        eprintln!("step {i}: {:.2} s ({} values)", t.elapsed().as_secs_f64(), v.len());
    }
    Ok(())
}

/// `a`'s cosine with `b`, and its distance from `b` over `b`'s size.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let diff = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb), diff / nb)
}

/// `t` (BF16) with every value one BF16 step off, alternately up and down: inputs as near as BF16 holds them.
fn nudged(t: &candle_core::Tensor) -> Result<candle_core::Tensor> {
    let v: Vec<half::bf16> = t.flatten_all()?.to_vec1::<half::bf16>()?.iter().enumerate().map(|(i, x)| half::bf16::from_bits(if i % 2 == 0 { x.to_bits().wrapping_add(1) } else { x.to_bits().wrapping_sub(1) })).collect();
    candle_core::Tensor::from_vec(v, t.dims(), t.device())
}

/// The WebGPU video stream gives the Candle one's velocity (on CUDA, BF16) for the checkpoint `OAIY_LTX_NVFP4` names
/// (Lightricks' NVFP4 release, or a BF16 one's Q8_0): a video of 2 latent frames by 4 by 6 and a context of 40 rows
/// (random, as the connector's are near unit RMS) at two sigmas and spatio-temporal guidance's pass. As near as the
/// reference is to itself, its inputs one BF16 step off, or to a tenth: the distilled release's last blocks take
/// random inputs' BF16 noise to a fifth of the velocity ([`trace_the_video_blocks_against_candles`]: its blocks
/// agree to 0.9999 through block 36).
#[test]
#[ignore = "needs an LTX 2.3 checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter, CUDA (the cuda feature) and some 60 GB of RAM"]
fn the_webgpu_video_stream_is_the_candle_one() -> Result<()> {
    let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
    let path = std::path::Path::new(&path);
    let (frames, h, w, fps, lc) = (2usize, 4usize, 6usize, 24usize, 40usize);
    let tokens = frames * h * w;
    let mut seed = 0x51ed_270bu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    };
    let latent: Vec<f32> = (0..tokens * 128).map(|_| (next() * 1.7) as f32).collect();
    let context: Vec<f32> = (0..lc * D).map(|_| (next() * 1.7) as f32).collect();
    // (a negative prompt's context, of another length)
    let negative_rows = 24usize;
    let negative: Vec<f32> = (0..negative_rows * D).map(|_| (next() * 1.7) as f32).collect();
    let t = std::time::Instant::now();
    let mut store = Store::open(path, 0)?;
    let mut gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
    eprintln!("WebGPU video stream loaded in {:.1} s", t.elapsed().as_secs_f64());
    let table = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, false), &[20., 2048., 2048.], D, HEADS));
    let got: Vec<Vec<f32>> = [0.8, 0.25].iter().map(|&sigma| gpu.forward(&latent, tokens, &context, lc, sigma, &table, (0, 0), h * w, None)).collect::<Result<_>>()?;
    // and spatio-temporal guidance's pass, block 28's self-attention passed through
    let stg = gpu.forward(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, Some(28))?;
    // and a starting image's and an end image's clean tokens (the first frame's, an appended frame's)
    let ends = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, true), &[20., 2048., 2048.], D, HEADS));
    let appended: Vec<f32> = latent.iter().chain(&latent[..h * w * 128]).map(|v| v * 0.9).collect();
    let conditioned = gpu.forward(&appended, tokens + h * w, &context, lc, 0.8, &ends, (h * w, h * w), h * w, None)?;
    // and a negative prompt without CFG: every block's text attention guided away from it (the reference's
    // defaults for video: scale 11, tau 2.5, alpha 0.25)
    gpu.nag = Some(Nag { context: negative.clone(), rows: negative_rows, scale: 11.0, tau: 2.5, alpha: 0.25 });
    let guided = gpu.forward(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, None)?;
    drop(gpu);
    let dev = Device::Cpu;
    // (a budget under the blocks' INT8 size: they stream as they are stored; a BF16 checkpoint's 28 GB over a
    // budget its INT8 rows fit would be those rows, not the reference)
    let mut cpu = crate::ltx::transformer::Transformer::new(Store::open(path, 0)?, &dev, 8 << 30, false, false)?;
    assert!(!cpu.int8, "the reference's blocks as INT8");
    let rope = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, false, &dev)?;
    let lt = candle_core::Tensor::from_vec(latent, (1, tokens, 128), &dev)?.to_dtype(DType::BF16)?;
    let ct = candle_core::Tensor::from_vec(context, (1, lc, D), &dev)?.to_dtype(DType::BF16)?;
    let (ln, cn) = (nudged(&lt)?, nudged(&ct)?);
    let rope_ends = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, true, &dev)?;
    let la = candle_core::Tensor::from_vec(appended, (1, tokens + h * w, 128), &dev)?.to_dtype(DType::BF16)?;
    let lan = nudged(&la)?;
    let nt = candle_core::Tensor::from_vec(negative, (1, negative_rows, D), &dev)?.to_dtype(DType::BF16)?;
    let passes_plain = got[0].clone();
    let mut want_plain: Option<Vec<f32>> = None;
    let passes = [(0.8, None, false, false, &got[0]), (0.25, None, false, false, &got[1]), (0.8, Some(28), false, false, &stg), (0.8, None, true, false, &conditioned), (0.8, None, false, true, &guided)];
    for (sigma, skip, ends, nag, got) in passes {
        let t = std::time::Instant::now();
        cpu.skip_video_self_attn = skip;
        cpu.nag = nag.then(|| crate::ltx::transformer::Nag { context: nt.clone(), scale: 11., tau: 2.5, alpha: 0.25 });
        let (rope, clean) = if ends { (&rope_ends, h * w) } else { (&rope, 0) };
        let velocity = |cpu: &mut crate::ltx::transformer::Transformer, l: &candle_core::Tensor, c: &candle_core::Tensor| -> Result<Vec<f32>> {
            cpu.forward(l, c, sigma, rope, clean, clean, None, |_| {})?.0.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };
        let (l, n) = if ends { (&la, &lan) } else { (&lt, &ln) };
        let want = velocity(&mut cpu, l, &ct)?;
        let seconds = t.elapsed().as_secs_f64();
        let nudged = velocity(&mut cpu, n, &cn)?;
        let (_, spread) = compare(&nudged, &want);
        let (cos, err) = compare(got, &want);
        if ends {
            // the clean tokens' rows (the first frame's, the appended one's) and the noisy ones' apart
            let row = 128;
            let (lo, hi) = (clean * row, want.len() - clean * row);
            let part = |v: &[f32], noisy: bool| -> Vec<f32> { if noisy { v[lo..hi].to_vec() } else { v[..lo].iter().chain(&v[hi..]).copied().collect() } };
            for noisy in [false, true] {
                let (c, e) = compare(&part(got, noisy), &part(&want, noisy));
                let (_, sp) = compare(&part(&nudged, noisy), &part(&want, noisy));
                eprintln!("  the {} rows: cosine {c:.6}, relative error {e:.2e} (the reference's own spread {sp:.2e})", if noisy { "noisy" } else { "clean" });
            }
        }
        let what = match (skip, ends, nag) {
            (Some(b), _, _) => format!("sigma {sigma}, block {b}'s self-attention passed through"),
            (None, true, _) => format!("sigma {sigma}, a starting and an end image's tokens clean"),
            (None, false, true) => format!("sigma {sigma}, a negative prompt by NAG"),
            (None, false, false) => format!("sigma {sigma}"),
        };
        if !nag && !ends && skip.is_none() && want_plain.is_none() {
            want_plain = Some(want.clone());
        }
        if nag {
            // What the guidance changes, here and in the reference: the velocity's difference from the pass with no
            // negative prompt (a pair's passes through the same arithmetic, so its difference is the guidance's).
            let (cos, moved) = compare(got, &passes_plain);
            eprintln!("  NAG against no negative prompt: cosine {cos:.6}, relative difference {moved:.2e}");
            assert!(moved > 1e-3, "the negative prompt changes the velocity");
            let plain = want_plain.as_ref().expect("the pass with no negative prompt first");
            let ours: Vec<f32> = got.iter().zip(&passes_plain).map(|(a, b)| a - b).collect();
            let theirs: Vec<f32> = want.iter().zip(plain).map(|(a, b)| a - b).collect();
            let (dc, de) = compare(&ours, &theirs);
            // (the change is some 4% of the velocity, as the reference's own spread is: the two changes agree as
            // far as that lets them; the blocks' trace with OAIY_LTX_TRACE_NAG holds each block to four places)
            eprintln!("  the guidance's change against the reference's: cosine {dc:.6}, relative error {de:.2e}");
            assert!(dc > 0.6, "the guidance's change is the reference's: cosine {dc}");
        }
        eprintln!("{what}: cosine {cos:.6}, relative error {err:.2e} (the reference's own spread {spread:.2e}; its step {seconds:.1} s)");
        // (its BF16 rounding at every op beside one step off at the inputs: within three times that)
        assert!(err <= (3.0 * spread).max(0.1), "{what}: relative error {err} where the reference's own spread is {spread}");
    }
    Ok(())
}

/// The two streams' first two blocks on WebGPU give the reference's own velocities: its golden tensors
/// (`tools/ltx/audio_reference.py --part transformer`, in `OAIY_LTX_GOLDEN`; the checkpoint `OAIY_LTX_CHECKPOINT`),
/// as [`crate::ltx::transformer`]'s `audio_video_blocks_match_reference` holds Candle to them: a video of one
/// latent frame of 4 by 4, its first four tokens a starting image's (clean: their velocities are the sampler's
/// to discard), six audio frames, each stream over eight context rows, at sigma 0.725; then the audio frozen
/// (its sigma 0, the video's as it was). The reference is BF16 throughout and this is Q8_0 weights (f16 with
/// OAIY_LTX_WEBGPU_WEIGHTS=f16) under f32 activations: held to a twentieth, where Candle's BF16 is to 0.03.
#[test]
#[ignore = "needs the reference's golden tensors (OAIY_LTX_GOLDEN), their checkpoint (OAIY_LTX_CHECKPOINT) and a WebGPU adapter"]
fn the_two_streams_blocks_are_the_references() -> Result<()> {
    let (Some(root), Some(weights)) = (std::env::var_os("OAIY_LTX_GOLDEN").map(std::path::PathBuf::from), std::env::var_os("OAIY_LTX_CHECKPOINT")) else { return Ok(()) };
    let read = |name: &str, len: usize| -> Result<Vec<f32>> {
        let bytes = std::fs::read(root.join(name))?;
        if bytes.len() != 4 * len {
            candle_core::bail!("{name}: {} bytes, not {len} floats", bytes.len());
        }
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let (frames, h, w, fps, lc) = (1usize, 4usize, 4usize, 24usize, 8usize);
    let (tokens, audio_frames, la, clean) = (frames * h * w, 6usize, 8usize, 4usize);
    let (x, context) = (read("av-video-input.f32", tokens * 128)?, read("av-video-context.f32", lc * D)?);
    let (ax, actx) = (read("av-audio-input.f32", audio_frames * 128)?, read("av-audio-context.f32", la * A)?);
    let mut store = Store::open(std::path::Path::new(&weights), 0)?;
    let mut gpu = WgpuLtx::load_streams(&mut store, 0, true, |_| {})?;
    // (the reference ran the first two blocks)
    gpu.blocks.truncate(2);
    let positions = video_positions(frames, h, w, fps, false);
    let table = gpu.upload(&rope_table(&positions, &[20., 2048., 2048.], D, HEADS));
    let times: Vec<Vec<f32>> = positions.iter().map(|p| vec![p[0]]).collect();
    let video_cross = gpu.upload(&rope_table(&times, &[20.], A, HEADS));
    let middles: Vec<Vec<f32>> = crate::ltx::transformer::audio_spans(audio_frames).iter().map(|&(s, e)| vec![((s + e) / 2.) as f32]).collect();
    let sound_table = gpu.upload(&rope_table(&middles, &[20.], A, HEADS));
    let relative = |got: &[f32], want: &[f32]| -> f64 {
        let e: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
        (e / want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt()
    };
    let mut worst = 0f64;
    for (sound_sigma, video_name, audio_name, what) in [(0.725, "av-video-output.f32", "av-audio-output.f32", "both denoised"), (0., "av-frozen-video-output.f32", "av-frozen-audio-output.f32", "the audio frozen")] {
        // (a set of goldens made without the frozen pass has no such pair)
        if sound_sigma == 0. && !root.join(video_name).is_file() {
            continue;
        }
        let pass = AudioPass { latent: &ax, tokens: audio_frames, context: &actx, rows: la, sigma: sound_sigma, table: &sound_table, video_cross: &video_cross, skip_self: None };
        let (video, audio) = gpu.forward_av(&x, tokens, &context, lc, 0.725, &table, (clean, 0), h * w, None, &pass)?;
        let (want_video, want_audio) = (read(video_name, tokens * 128)?, read(audio_name, audio_frames * 128)?);
        let (ev, ea) = (relative(&video[clean * 128..], &want_video[clean * 128..]), relative(&audio, &want_audio));
        eprintln!("{what}: the video's velocity {ev:.4} from the reference's (relative RMS), the audio's {ea:.4}");
        worst = worst.max(ev).max(ea);
    }
    assert!(worst < 0.05, "the two streams' blocks: {worst} from the reference's");
    Ok(())
}

/// The audio stream's text connector on WebGPU (2,048 wide: [`connector`] at its blocks' width) gives the
/// reference's own context: its golden tensors (`OAIY_LTX_GOLDEN`'s `audio-connector-input.f32`, twelve prompt
/// rows, and `-output.f32`, the 1,024 rows; the checkpoint `OAIY_LTX_CHECKPOINT`), as
/// [`crate::ltx::transformer`]'s `audio_connector_matches_reference` holds Candle to them.
#[test]
#[ignore = "needs the reference's golden tensors (OAIY_LTX_GOLDEN), their checkpoint (OAIY_LTX_CHECKPOINT) and a WebGPU adapter"]
fn the_audio_connector_is_the_references() -> Result<()> {
    let (Some(root), Some(weights)) = (std::env::var_os("OAIY_LTX_GOLDEN").map(std::path::PathBuf::from), std::env::var_os("OAIY_LTX_CHECKPOINT")) else { return Ok(()) };
    let read = |name: &str, len: usize| -> Result<Vec<f32>> {
        let bytes = std::fs::read(root.join(name))?;
        if bytes.len() != 4 * len {
            candle_core::bail!("{name}: {} bytes, not {len} floats", bytes.len());
        }
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let (rows, s) = (1024usize, 12usize);
    let (features, want) = (read("audio-connector-input.f32", s * A)?, read("audio-connector-output.f32", rows * A)?);
    let gpu = ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(err)?;
    let mut store = Store::open(std::path::Path::new(&weights), 0)?;
    let p = format!("{PREFIX}audio_embeddings_connector");
    let registers = store.tensor_f32(&format!("{p}.learnable_registers"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let blocks: Vec<ConnectorBlock> = (0..8).map(|i| ConnectorBlock::load(&mut store, &gpu, &format!("{p}.transformer_1d_blocks.{i}"))).collect::<Result<_>>()?;
    // the prompt's rows, then the learned registers (register i % 128 at row i)
    let x0: Vec<f32> = features.iter().copied().chain((s..rows).flat_map(|i| registers[(i % 128) * A..(i % 128 + 1) * A].iter().copied())).collect();
    let x = gpu.vec(rows * A);
    gpu.upload(&x, &x0);
    let rope = rope_table(&(0..rows).map(|i| vec![i as f32]).collect::<Vec<_>>(), &[4096.], A, HEADS);
    let table = gpu.vec(rope.len());
    gpu.upload(&table, &rope);
    let mut rec = gpu.begin();
    rec.keep_groups(false);
    let out = connector(&gpu, rec.as_mut(), &blocks, &x, rows, &table);
    rec.read(&out);
    let got = rec.finish().pop().ok_or_else(|| err("the context was not read"))?;
    let (cos, e) = compare(&got, &want);
    eprintln!("the audio connector's context: cosine {cos:.6}, {e:.4} from the reference's (relative)");
    assert!(e < 0.03, "the audio connector: {e} from the reference's");
    Ok(())
}

/// The two streams on WebGPU give the Candle ones' velocities (CPU, BF16) for the checkpoint `OAIY_LTX_NVFP4`
/// names (one with its audio stream: LTX 2.3's single file): a video of 2 latent frames by 4 by 6 and 9 audio
/// frames, each over its own context (random, near unit RMS), at two sigmas, the audio's its own. Each stream as
/// near as the reference is to itself with its inputs one BF16 step off, or to a tenth (see
/// [`the_webgpu_video_stream_is_the_candle_one`]).
#[test]
#[ignore = "needs an LTX 2.3 checkpoint with its audio stream (OAIY_LTX_NVFP4), a WebGPU adapter, some 60 GB of RAM and a Candle that multiplies BF16 (CUDA's: the CPU's does not)"]
fn the_webgpu_audio_and_video_streams_are_the_candle_ones() -> Result<()> {
    use crate::ltx::transformer::{AudioInput, Rope, Transformer};
    let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
    let path = std::path::Path::new(&path);
    let (frames, h, w, fps, lc) = (2usize, 4usize, 6usize, 24usize, 40usize);
    let (audio_frames, la) = (9usize, 24usize);
    let tokens = frames * h * w;
    let mut seed = 0x51ed_270bu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    };
    let latent: Vec<f32> = (0..tokens * 128).map(|_| (next() * 1.7) as f32).collect();
    let context: Vec<f32> = (0..lc * D).map(|_| (next() * 1.7) as f32).collect();
    let sound: Vec<f32> = (0..audio_frames * 128).map(|_| (next() * 1.7) as f32).collect();
    let sound_context: Vec<f32> = (0..la * A).map(|_| (next() * 1.7) as f32).collect();
    let t = std::time::Instant::now();
    let mut store = Store::open(path, 0)?;
    let gpu = WgpuLtx::load_streams(&mut store, 0, true, |_| {})?;
    eprintln!("WebGPU video and audio streams loaded in {:.1} s", t.elapsed().as_secs_f64());
    let positions = video_positions(frames, h, w, fps, false);
    let table = gpu.upload(&rope_table(&positions, &[20., 2048., 2048.], D, HEADS));
    let times: Vec<Vec<f32>> = positions.iter().map(|p| vec![p[0]]).collect();
    let video_cross = gpu.upload(&rope_table(&times, &[20.], A, HEADS));
    let middles: Vec<Vec<f32>> = crate::ltx::transformer::audio_spans(audio_frames).iter().map(|&(s, e)| vec![((s + e) / 2.) as f32]).collect();
    let sound_table = gpu.upload(&rope_table(&middles, &[20.], A, HEADS));
    // (the audio's sigma its own: the gates and the rows between the streams go by the right one each)
    let sigmas = [(0.8, 0.725), (0.25, 0.25)];
    let mut got = Vec::new();
    for &(sigma, sound_sigma) in &sigmas {
        let t = std::time::Instant::now();
        let pass = AudioPass { latent: &sound, tokens: audio_frames, context: &sound_context, rows: la, sigma: sound_sigma, table: &sound_table, video_cross: &video_cross, skip_self: None };
        got.push(gpu.forward_av(&latent, tokens, &context, lc, sigma, &table, (0, 0), h * w, None, &pass)?);
        eprintln!("a step of both streams: {:.2} s", t.elapsed().as_secs_f64());
    }
    // and the video alone from the model loaded with both: its stream as it is without sound
    let alone = gpu.forward(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, None)?;
    drop(gpu);
    let dev = Device::Cpu;
    let mut cpu = Transformer::new(Store::open(path, 0)?, &dev, 8 << 30, false, true)?;
    assert!(!cpu.int8, "the reference's blocks as INT8");
    let rope = Rope::video_with_end(frames, h, w, fps, false, &dev)?;
    let (sound_rope, cross_rope) = (Rope::audio(audio_frames, &dev)?, Rope::video_cross(frames, h, w, fps, false, &dev)?);
    let bf16 = |v: &[f32], rows: usize, width: usize| candle_core::Tensor::from_vec(v.to_vec(), (1, rows, width), &dev)?.to_dtype(DType::BF16);
    let (lt, ct, st, sct) = (bf16(&latent, tokens, 128)?, bf16(&context, lc, D)?, bf16(&sound, audio_frames, 128)?, bf16(&sound_context, la, A)?);
    let (ln, cn, sn, scn) = (nudged(&lt)?, nudged(&ct)?, nudged(&st)?, nudged(&sct)?);
    let host = |t: candle_core::Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
    for (i, &(sigma, sound_sigma)) in sigmas.iter().enumerate() {
        let t = std::time::Instant::now();
        let mut both = |l: &candle_core::Tensor, c: &candle_core::Tensor, s: &candle_core::Tensor, sc: &candle_core::Tensor| -> Result<(Vec<f32>, Vec<f32>)> {
            let audio = AudioInput { latent: s, context: sc, rope: &sound_rope, video_cross: &cross_rope, sigma: sound_sigma, isolated: false, clean_tokens: 0 };
            let (video, sound) = cpu.forward(l, c, sigma, &rope, 0, 0, Some(audio), |_| {})?;
            Ok((host(video)?, host(sound.ok_or_else(|| err("the reference gave no audio velocity"))?)?))
        };
        let want = both(&lt, &ct, &st, &sct)?;
        let seconds = t.elapsed().as_secs_f64();
        let near = both(&ln, &cn, &sn, &scn)?;
        for (what, got, want, near) in [("video", &got[i].0, &want.0, &near.0), ("audio", &got[i].1, &want.1, &near.1)] {
            let (_, spread) = compare(near, want);
            let (cos, e) = compare(got, want);
            eprintln!("sigma {sigma} (the audio's {sound_sigma}), the {what}'s velocity: cosine {cos:.6}, relative error {e:.2e} (the reference's own spread {spread:.2e}; its step {seconds:.1} s)");
            assert!(e <= (3.0 * spread).max(0.1), "the {what}'s velocity at sigma {sigma}: relative error {e} where the reference's own spread is {spread}");
        }
    }
    // (the video alone: against the reference with no audio, through the same blocks)
    let (video, _) = cpu.forward(&lt, &ct, 0.8, &rope, 0, 0, None, |_| {})?;
    let (cos, e) = compare(&alone, &host(video)?);
    eprintln!("the video alone from the model with both: cosine {cos:.6}, relative error {e:.2e}");
    assert!(e <= 0.3, "the video alone: relative error {e}");
    Ok(())
}

/// A layer a LoRA adapts is loaded with it (`Store::add_lora`): its product the weight's plus the LoRA's scaled
/// `B A`, as f16 (a layer 8 wide) and as Q8_0 (256 wide, to its quantization), where a layer the LoRA does not
/// name is its weight alone.
#[test]
fn a_layer_a_lora_adapts_is_loaded_with_it() -> Result<()> {
    let Ok(gpu) = ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)) else { return Ok(()) };
    let dir = std::env::temp_dir().join(format!("oaiy-wgpu-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let write = |path: &std::path::Path, tensors: &[(String, Vec<usize>, Vec<u8>)]| -> Result<()> {
        let mut header = String::from("{");
        let mut data = Vec::new();
        for (i, (name, shape, bytes)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!("\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{},{}]}}", data.len(), data.len() + bytes.len()));
            data.extend_from_slice(bytes);
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend(data);
        std::fs::write(path, out)?;
        Ok(())
    };
    let mut seed = 0x9e37_79b9u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
    };
    let (n, rank, rows, scale) = (6usize, 2usize, 5usize, 0.7f32);
    for k in [8usize, 256] {
        let w: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let (a, b): (Vec<f32>, Vec<f32>) = ((0..rank * k).map(|_| next()).collect(), (0..n * rank).map(|_| next()).collect());
        let bias: Vec<f32> = (0..n).map(|_| next()).collect();
        let (model, lora) = (dir.join(format!("m{k}.safetensors")), dir.join(format!("l{k}.safetensors")));
        write(&model, &[
            ("model.diffusion_model.blk.to_q.weight".into(), vec![n, k], f32s(&w)),
            ("model.diffusion_model.blk.to_q.bias".into(), vec![n], f32s(&bias)),
            ("model.diffusion_model.blk.to_k.weight".into(), vec![n, k], f32s(&w)),
        ])?;
        write(&lora, &[("diffusion_model.blk.to_q.lora_A.weight".into(), vec![rank, k], f32s(&a)), ("diffusion_model.blk.to_q.lora_B.weight".into(), vec![n, rank], f32s(&b))])?;
        let mut store = Store::open(&model, 0)?;
        assert_eq!(store.add_lora(&lora, scale as f64)?, 1);
        assert!(store.adapted("model.diffusion_model.blk.to_q.weight") && !store.adapted("model.diffusion_model.blk.to_k.weight"));
        let x: Vec<f32> = (0..rows * k).map(|_| next()).collect();
        for (name, adapted) in [("model.diffusion_model.blk.to_q", true), ("model.diffusion_model.blk.to_k", false)] {
            let layer = Linear::load(&mut store, &gpu, name)?;
            let (xv, yv) = (gpu.vec(rows * k), gpu.vec(rows * n));
            gpu.upload(&xv, &x);
            let mut rec = gpu.begin();
            layer.forward(rec.as_mut(), &xv, &yv, rows);
            rec.read(&yv);
            let got = rec.finish().pop().unwrap();
            // (against the weight with the LoRA's part, and against the weight alone: how far the LoRA moves it)
            let (mut worst, mut size, mut alone) = (0f64, 0f64, 0f64);
            for r in 0..rows {
                for i in 0..n {
                    let part = |j: usize| scale as f64 * (0..rank).map(|q| b[i * rank + q] as f64 * a[q * k + j] as f64).sum::<f64>();
                    let plain = (0..k).map(|j| w[i * k + j] as f64 * x[r * k + j] as f64).sum::<f64>() + if adapted { bias[i] as f64 } else { 0. };
                    let want = plain + if adapted { (0..k).map(|j| part(j) * x[r * k + j] as f64).sum::<f64>() } else { 0. };
                    worst = worst.max((got[r * n + i] as f64 - want).abs());
                    alone = alone.max((got[r * n + i] as f64 - plain).abs());
                    size = size.max(want.abs());
                }
            }
            eprintln!("{k} wide, {}: the worst error {worst:.2e} of {size:.2} (from the weight alone {alone:.2e})", if adapted { "with the LoRA" } else { "no LoRA on it" });
            // (f16's rounding of the weights, or Q8_0's quantization of them)
            assert!(worst <= size * if k % 256 == 0 { 0.03 } else { 5e-3 }, "{name}, {k} wide: {worst} of {size}");
            assert!(!adapted || alone > 20. * worst, "{name}, {k} wide: the LoRA's part is in the product");
        }
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

/// Each block's output of the WebGPU video stream against Candle's (on CUDA, BF16; `OAIY_LTX_NVFP4`), the inputs
/// as [`the_webgpu_video_stream_is_the_candle_one`]'s at sigma 0.8: where the two part.
#[test]
#[ignore = "a trace; needs an LTX 2.3 checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter and CUDA (the cuda feature)"]
fn trace_the_video_blocks_against_candles() -> Result<()> {
    let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
    let path = std::path::Path::new(&path);
    let (frames, h, w, fps, lc) = (2usize, 4usize, 6usize, 24usize, 40usize);
    let tokens = frames * h * w;
    let mut seed = 0x51ed_270bu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    };
    let latent: Vec<f32> = (0..tokens * 128).map(|_| (next() * 1.7) as f32).collect();
    let context: Vec<f32> = (0..lc * D).map(|_| (next() * 1.7) as f32).collect();
    // (OAIY_LTX_TRACE_NAG: a negative prompt by NAG in both, its context another 24 rows: every block's text
    // attention guided, and the first blocks, which agree to four places, held to the reference's)
    let negative: Option<Vec<f32>> = std::env::var_os("OAIY_LTX_TRACE_NAG").map(|_| (0..24 * D).map(|_| (next() * 1.7) as f32).collect());
    let mut store = Store::open(path, 0)?;
    let mut gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
    gpu.nag = negative.as_ref().map(|n| Nag { context: n.clone(), rows: 24, scale: 11.0, tau: 2.5, alpha: 0.25 });
    let table = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, false), &[20., 2048., 2048.], D, HEADS));
    let got = gpu.pass(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, None, true, None)?;
    let trail = &got[0];
    drop(gpu);
    let dev = Device::Cpu;
    let mut cpu = crate::ltx::transformer::Transformer::new(Store::open(path, 0)?, &dev, 8 << 30, false, false)?;
    cpu.hiddens = Some(Vec::new());
    if let Some(n) = &negative {
        let context = candle_core::Tensor::from_vec(n.clone(), (1, 24, D), &dev)?.to_dtype(DType::BF16)?;
        cpu.nag = Some(crate::ltx::transformer::Nag { context, scale: 11., tau: 2.5, alpha: 0.25 });
    }
    let rope = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, false, &dev)?;
    let lt = candle_core::Tensor::from_vec(latent, (1, tokens, 128), &dev)?.to_dtype(DType::BF16)?;
    let ct = candle_core::Tensor::from_vec(context, (1, lc, D), &dev)?.to_dtype(DType::BF16)?;
    let (v, _) = cpu.forward(&lt, &ct, 0.8, &rope, 0, 0, None, |_| {})?;
    let hiddens = cpu.hiddens.take().unwrap_or_default();
    for (i, hidden) in hiddens.iter().enumerate() {
        let want = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (cos, rel) = compare(&trail[i * tokens * D..(i + 1) * tokens * D], &want);
        let rms = (want.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        let peak = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        eprintln!("block {i:2}: cosine {cos:.6}, relative error {rel:.2e} (the reference's RMS {rms:.3e}, its largest {peak:.3e})");
        if negative.is_some() && i < 8 {
            assert!(cos > 0.999, "block {i} with a negative prompt by NAG: cosine {cos}");
        }
    }
    let want = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let (cos, rel) = compare(got.last().unwrap(), &want);
    eprintln!("velocity: cosine {cos:.6}, relative error {rel:.2e}");
    Ok(())
}

/// A BF16 checkpoint's layers (`OAIY_LTX_NVFP4` naming a BF16 one: LTX 2.3's distilled release) as Q8_0 on the
/// tensor cores give their quantization's product (in f64, the kernel's own error), 300 rows; and how far that is
/// from the BF16 weights' (the quantization's).
#[test]
#[ignore = "needs a BF16 LTX 2.3 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter with tensor cores"]
fn a_q8_layer_on_the_gpu_is_its_quantizations() -> Result<()> {
    let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
    let mut store = Store::open(std::path::Path::new(&path), 0)?;
    let gpu = ggml_rs_wgpu::WgpuBackend::new(None).map_err(err)?;
    for name in ["model.diffusion_model.transformer_blocks.10.attn1.to_q", "model.diffusion_model.transformer_blocks.10.attn1.to_gate_logits", "model.diffusion_model.transformer_blocks.10.ff.net.0.proj", "model.diffusion_model.transformer_blocks.10.ff.net.2"] {
        let l = Linear::load(&mut store, &gpu, name)?;
        if !matches!(l.weight, Weight::Q8(_)) {
            eprintln!("{name}: not Q8_0 (an NVFP4 checkpoint's, or no tensor cores)");
            return Ok(());
        }
        let rows = 300;
        let mut seed = 0x9e37_79b9u64;
        let x: Vec<f32> = (0..rows * l.k)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        let (xd, yd) = (gpu.vec(x.len()), gpu.vec(rows * l.n));
        gpu.upload(&xd, &x);
        let mut rec = gpu.begin();
        l.forward(rec.as_mut(), &xd, &yd, rows);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        let w = store.tensor_f32(&format!("{name}.weight"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let b = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let mut wq = vec![0f32; w.len()];
        ggml_quants::q8_0::dequantize(&q8_0(&w), &mut wq);
        let product = |w: &[f32]| -> Vec<f64> {
            let mut y = vec![0f64; rows * l.n];
            for r in 0..rows {
                let xr = &x[r * l.k..(r + 1) * l.k];
                for o in 0..l.n {
                    let wr = &w[o * l.k..(o + 1) * l.k];
                    y[r * l.n + o] = b[o] as f64 + xr.iter().zip(wr).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>();
                }
            }
            y
        };
        let (exact, bf16) = (product(&wq), product(&w));
        let rms = (bf16.iter().map(|v| v * v).sum::<f64>() / bf16.len() as f64).sqrt();
        let rel = |a: &[f64]| (got.iter().zip(a).map(|(g, e)| (*g as f64 - e).powi(2)).sum::<f64>() / a.len() as f64).sqrt() / rms;
        let quant = (exact.iter().zip(&bf16).map(|(e, f)| (e - f).powi(2)).sum::<f64>() / bf16.len() as f64).sqrt() / rms;
        eprintln!("{name} [{}, {}]: the kernel's error {:.2e} of the RMS, the GPU's from BF16's {:.2e} (Q8_0's own {quant:.2e})", l.n, l.k, rel(&exact), rel(&bf16));
        assert!(rel(&exact) < 2e-3, "{name}: the kernel's error {} of the RMS", rel(&exact));
    }
    Ok(())
}

/// A real NVFP4 layer of Lightricks' release (`OAIY_LTX_NVFP4`) on the tensor cores gives what its store's own
/// decode does on the CPU (in f32), 300 rows.
#[test]
#[ignore = "needs LTX 2.3's NVFP4 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
fn an_nvfp4_layer_on_the_gpu_is_the_stores() -> Result<()> {
    let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
    let mut store = Store::open(std::path::Path::new(&path), 0)?;
    let gpu = ggml_rs_wgpu::WgpuBackend::new(None).map_err(err)?;
    // (not a gate's logits: 32 rows, their scales a tile of 128's, which the store's own decode does not take)
    for name in ["model.diffusion_model.transformer_blocks.10.attn1.to_q", "model.diffusion_model.transformer_blocks.20.ff.net.2", "model.diffusion_model.transformer_blocks.30.audio_ff.net.0.proj"] {
        let l = Linear::load(&mut store, &gpu, name)?;
        assert!(matches!(l.weight, Weight::Nvfp4 { .. }), "{name} NVFP4");
        let rows = 300;
        let mut seed = 0x9e37_79b9u64;
        let x: Vec<f32> = (0..rows * l.k)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                half::f16::from_f32((seed % 2001) as f32 / 1000.0 - 1.0).to_f32()
            })
            .collect();
        let (xd, yd) = (gpu.vec(x.len()), gpu.vec(rows * l.n));
        gpu.upload(&xd, &x);
        let mut rec = gpu.begin();
        l.forward(rec.as_mut(), &xd, &yd, rows);
        rec.read(&yd);
        let got = rec.finish().pop().unwrap();
        let w = store.tensor_f32(&format!("{name}.weight"), &Device::Cpu)?;
        let b = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?;
        let xt = candle_core::Tensor::from_vec(x, (rows, l.k), &Device::Cpu)?;
        let want = xt.matmul(&w.t()?)?.broadcast_add(&b)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        let worst = got.iter().zip(&want).map(|(a, e)| (*a as f64 - *e as f64).abs()).fold(0.0, f64::max);
        eprintln!("{name} [{}, {}]: the worst error {worst:.3e} of an RMS {rms:.3e}", l.n, l.k);
        assert!(worst <= 1e-3 * rms.max(1e-6) * 10.0, "{name}: worst {worst} of an RMS {rms}");
    }
    Ok(())
}
