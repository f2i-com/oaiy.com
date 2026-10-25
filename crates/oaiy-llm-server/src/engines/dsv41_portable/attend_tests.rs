//! A decode step's attention on the device (`ggml_rs_wgpu::exl3::ProgramStep`'s `Rope` to `HcMix`) against the CPU
//! model's own: each rounding to the bit, the exponentials near.

use super::*;
use crate::dsv41::formats::{fake_quant_fp4_inplace, fake_quant_fp8_inplace, to_bf16, Fp4Scale};
use crate::dsv41::ops::Rope;
use ggml_rs::chain::{DeviceChain, DeviceVec};
use ggml_rs_wgpu::exl3::{run_program, ProgramStep as G};

fn gpu() -> WgpuBackend {
    WgpuBackend::new(Some(1 << 30)).expect("a WebGPU adapter")
}

/// A generator of values of mixed signs and sizes.
struct Values(u64);

impl Values {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// `n` bf16 values in `±scale`, a few of them much larger.
    fn bf16(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let r = self.next();
                let v = ((r % 20_001) as f32 / 10_000.0 - 1.0) * scale * if r >> 40 & 63 == 0 { 30.0 } else { 1.0 };
                to_bf16(v)
            })
            .collect()
    }
}

fn kept(b: &WgpuBackend, values: &[f32]) -> DeviceVec {
    let v = DeviceChain::vec(b, values.len());
    DeviceChain::upload(b, &v, values);
    v
}

fn words(w: &[usize]) -> Vec<f32> {
    w.iter().map(|&v| f32::from_bits(v as u32)).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The rotary rotation, the fp8 and the fp4 round trips (each scale kind) on the device are the host's to the bit: rows
/// of queries and keys at several positions of both of DeepSeek-V4.1's rotary flavours, forward and back, and values
/// of mixed sizes, among them some on the e2m1 grid's midpoints (ties) of their block's scale.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn the_rotary_and_quantizers_on_the_device_are_the_hosts() {
    let b = gpu();
    let mut r = Values(0x9e37_79b9_7f4a_7c15);
    let (rd, max_pos) = (64usize, 4096usize);
    for rope in [Rope::new(rd, max_pos, 0, 10000.0, 16.0, 32.0, 1.0), Rope::new(rd, max_pos, 65536, 160000.0, 16.0, 32.0, 1.0)] {
        let (cos, sin) = rope.tables();
        let (cos, sin) = (kept(&b, cos), kept(&b, sin));
        for (rows, width) in [(64usize, 512usize), (32, 128), (1, 512)] {
            for pos in [0usize, 1, 127, 128, 2047, 4095] {
                for inverse in [false, true] {
                    let x = r.bf16(rows * width, 4.0);
                    let mut want = x.clone();
                    for row in want.chunks_mut(width) {
                        rope.apply(&mut row[width - rd..], pos, inverse);
                    }
                    let w = words(&[7, pos]);
                    let steps = [G::Input(&x), G::State(&cos), G::State(&sin), G::Input(&w), G::Rope { x: 0, rows, stride: width, at: width - rd, half: rd / 2, cos: 1, sin: 2, pos: 3, word: 1, inverse }];
                    let got = run_program(&b, &steps, &[(0, 0, x.len())]).expect("a program");
                    assert_eq!(bits(&got[0]), bits(&want), "{rows} rows of {width} at {pos}, back {inverse}");
                }
            }
        }
    }
    for (len, scale) in [(512usize, 1.0f32), (512, 1e-3), (4096, 30.0), (100, 0.25)] {
        let x = r.bf16(len, scale);
        let mut fp8 = x.clone();
        fake_quant_fp8_inplace(&mut fp8);
        // (an fp4 block's ties: each block's largest at 6 times a scale, the rest on the grid's midpoints of it)
        let mut ties = x.clone();
        for (i, v) in ties.iter_mut().enumerate() {
            let mid = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0][i % 7];
            *v = if i % 16 == 0 { 6.0 * scale } else if i % 3 == 0 { to_bf16(mid * scale) * if i % 2 == 0 { 1.0 } else { -1.0 } } else { *v };
        }
        for (block, kind) in [(32usize, Fp4Scale::E8M0), (16, Fp4Scale::E4M3)] {
            for input in [&x, &ties] {
                let mut want = input.clone();
                fake_quant_fp4_inplace(&mut want, block, kind);
                let steps = [G::Input(input), G::Fp4 { x: 0, len, block, e4m3: kind == Fp4Scale::E4M3 }];
                let got = run_program(&b, &steps, &[(0, 0, len)]).expect("a program");
                assert_eq!(bits(&got[0]), bits(&want), "fp4 of {len} values of {scale}, blocks of {block}");
            }
        }
        let steps = [G::Input(&x), G::Fp8 { x: 0, len }];
        let got = run_program(&b, &steps, &[(0, 0, len)]).expect("a program");
        assert_eq!(bits(&got[0]), bits(&fp8), "fp8 of {len} values of {scale}");
    }
}

/// The indexer's scores on the device are the host's to the bit, and its picks the host's (`select_compressed` of a
/// decode step: the 512 best, ties to the lower position, in order): queries and keys fp4 as the indexer makes them,
/// counts below, at and past the picks' number, and keys of few values so that scores tie often.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn the_indexers_scores_and_picks_on_the_device_are_the_hosts() {
    let b = gpu();
    let mut r = Values(0x2545_f491_4f6c_dd1d);
    let (heads, dim, k, cap) = (32usize, 128usize, 512usize, 4096usize);
    for (n, coarse) in [(1usize, false), (100, false), (512, false), (513, true), (3000, false), (4096, true)] {
        let mut q = r.bf16(heads * dim, 2.0);
        fake_quant_fp4_inplace(&mut q, 32, Fp4Scale::E8M0);
        let mut keys = r.bf16(cap * dim, if coarse { 0.01 } else { 1.0 });
        if coarse {
            // (a few distinct keys over and over: many equal scores)
            for i in 0..cap * dim {
                keys[i] = keys[(i / dim % 5) * dim + i % dim];
            }
        }
        fake_quant_fp4_inplace(&mut keys, 32, Fp4Scale::E8M0);
        let w = r.bf16(heads, 1.0);
        let scale = (dim as f32).powf(-0.5) * (heads as f32).powf(-0.5);
        let weights: Vec<f32> = w.iter().map(|v| to_bf16(v * scale)).collect();
        let want = crate::dsv41::attention::index_scores(&q, &keys[..n * dim], &weights, heads, dim);
        // (a decode step's `select_compressed`: the best of them by the host's order, in position order)
        let mut picks: Vec<i32> = crate::dsv41::attention::topk_indices(&want, k).into_iter().map(|p| p as i32).collect();
        picks.sort_unstable();
        let key_v = kept(&b, &keys);
        let picked = DeviceChain::vec(&b, k);
        let wd = words(&[n]);
        let steps = [G::Input(&q), G::State(&key_v), G::Input(&w), G::Input(&wd), G::IndexScores { q: 0, keys: 1, weights: 2, heads, dim, scale, pos: 3, word: 0 }, G::State(&picked), G::TopK { scores: 4, dst: 5, k, pos: 3, word: 0 }];
        let got = run_program(&b, &steps, &[(4, 0, n), (5, 0, k)]).expect("a program");
        assert_eq!(bits(&got[0]), bits(&want), "the scores of {n} keys");
        let device: Vec<i32> = got[1].iter().map(|v| v.to_bits()).filter(|&p| p != u32::MAX).map(|p| p as i32).collect();
        assert_eq!(device, picks, "the picks of {n} keys (coarse {coarse})");
    }
}

/// A prompt's index scores on the device (`IndexScoresRows`) are the host's (`index_scores` of every row against every
/// key) to the bit: queries and keys fp4 as the indexer makes them, each row its own weights, keys of few values.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn a_prompts_index_scores_on_the_device_are_the_hosts() {
    let b = gpu();
    let mut r = Values(0x94d0_49bb_1331_11eb);
    let (heads, dim) = (32usize, 128usize);
    for (rows, n, coarse) in [(37usize, 3000usize, false), (600, 700, true), (5, 9000, false)] {
        let mut q = r.bf16(rows * heads * dim, 2.0);
        fake_quant_fp4_inplace(&mut q, 32, Fp4Scale::E8M0);
        let mut keys = r.bf16(n * dim, if coarse { 0.01 } else { 1.0 });
        if coarse {
            for i in 0..n * dim {
                keys[i] = keys[(i / dim % 5) * dim + i % dim];
            }
        }
        fake_quant_fp4_inplace(&mut keys, 32, Fp4Scale::E8M0);
        let scale = (dim as f32).powf(-0.5) * (heads as f32).powf(-0.5);
        let weights: Vec<f32> = r.bf16(rows * heads, 1.0).iter().map(|v| to_bf16(v * scale)).collect();
        let want = crate::dsv41::attention::index_scores(&q, &keys, &weights, heads, dim);
        // (a cache with room past the count, as a layer's whole cache is)
        let mut cache = keys.clone();
        cache.resize((n + 1000) * dim, 7.0);
        let wd = words(&[n]);
        let steps = [G::Input(&q), G::Input(&cache), G::Input(&weights), G::Input(&wd), G::IndexScoresRows { q: 0, keys: 1, weights: 2, rows, heads, dim, n: n + 1000, pos: 3, word: 0 }];
        let got = run_program(&b, &steps, &[(4, 0, rows * n)]).expect("a program");
        assert_eq!(bits(&got[0]), bits(&want), "{rows} rows of {n} keys (coarse {coarse})");
    }
}

/// The candidate blocks on the device (`BlockKeep` of the candidate source's scores, `MaskBlocks` of a later source's,
/// then its `TopK`) are the host's (`candidate_mask`, `select_compressed`'s masking and picks): counts up to the 2,048
/// blocks of 8 that keep every position, at them and past them, scores of few values so that blocks tie often.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn the_candidate_blocks_on_the_device_are_the_hosts() {
    let b = gpu();
    let mut r = Values(0x7fb5_d329_728e_a185);
    let (blocks_k, bs, k, cap) = (2048usize, 8usize, 512usize, 65536usize);
    let blocks = DeviceChain::vec(&b, 1 + cap.div_ceil(bs));
    let picked = DeviceChain::vec(&b, k);
    for (n, coarse) in [(100usize, false), (16384, false), (16385, true), (16391, false), (40000, true), (65536, false)] {
        let source = r.bf16(n, 1.0);
        let mut later = r.bf16(n, 1.0);
        let source: Vec<f32> = if coarse { source.iter().map(|v| to_bf16((v * 4.0).round() / 4.0)).collect() } else { source };
        if coarse {
            later.iter_mut().for_each(|v| *v = to_bf16((*v * 8.0).round() / 8.0));
        }
        let mask = crate::dsv41::attention::candidate_mask(&source, n, blocks_k, bs);
        let masked: Vec<f32> = later.iter().zip(&mask).map(|(&v, &keep)| if keep { v } else { f32::NEG_INFINITY }).collect();
        let mut picks: Vec<i32> = crate::dsv41::attention::topk_indices(&masked, k).into_iter().map(|p| p as i32).collect();
        picks.sort_unstable();
        let wd = words(&[n]);
        let steps = [
            G::Input(&source),
            G::Input(&later),
            G::Input(&wd),
            G::State(&blocks),
            G::BlockKeep { scores: 0, blocks: 3, k: blocks_k, bs, pos: 2, word: 0 },
            G::MaskBlocks { scores: 1, blocks: 3, bs, pos: 2, word: 0 },
            G::State(&picked),
            G::TopK { scores: 1, dst: 6, k, pos: 2, word: 0 },
        ];
        let nb = n.div_ceil(bs);
        let got = run_program(&b, &steps, &[(3, 0, 1 + nb), (1, 0, n), (6, 0, k)]).expect("a program");
        let flags = bits(&got[0]);
        assert_eq!(flags[0] as usize, nb, "the blocks of {n}");
        let kept: Vec<bool> = (0..n).map(|p| flags[1 + p / bs] == 1).collect();
        assert_eq!(kept, mask, "the blocks kept of {n} (coarse {coarse})");
        assert_eq!(bits(&got[1]), bits(&masked), "the later source's scores of {n} masked");
        let device: Vec<i32> = got[2].iter().map(|v| v.to_bits()).filter(|&p| p != u32::MAX).map(|p| p as i32).collect();
        assert_eq!(device, picks, "the picks of {n} (coarse {coarse})");
    }
}

/// A prompt's chunk's attention on the device (`RopeRows`, `AttendRows`, `RopeRows` back) is the host's but for its
/// exponentials: tokens at their own positions, each over its own positions (window rows and compressed ones laid end
/// to end, some none, as the host gathers them), the queries rotated to the bit first; nearly every output the host's
/// bf16 value and none far from it.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn a_prompts_attention_on_the_device_is_the_hosts_but_for_its_exponentials() {
    let b = gpu();
    let mut r = Values(0x5851_f42d_4c95_7f2d);
    let (heads, dim, rd, len, rows) = (64usize, 512usize, 64usize, 640usize, 2000usize);
    let rope = Rope::new(rd, 8192, 0, 10000.0, 16.0, 32.0, 1.0);
    let (cos, sin) = rope.tables();
    let (cos_v, sin_v) = (kept(&b, cos), kept(&b, sin));
    let mut kv = r.bf16(rows * dim, 2.0);
    for (i, row) in kv.chunks_mut(dim).enumerate() {
        if i < 1000 {
            fake_quant_fp8_inplace(row);
        } else {
            fake_quant_fp4_inplace(row, 16, Fp4Scale::E4M3);
        }
    }
    let sink = r.bf16(heads, 2.0);
    let scale = (dim as f32).powf(-0.5);
    for (tokens, start) in [(37usize, 0usize), (100, 3000)] {
        let mut q = r.bf16(tokens * heads * dim / 2, 3.0);
        q.extend(r.bf16(tokens * heads * dim / 2, 0.02));
        let idxs: Vec<Vec<i32>> = (0..tokens).map(|_| (0..len).map(|i| if i % 29 == 3 { -1 } else { ((r.next() as usize) % rows) as i32 }).collect()).collect();
        let mut want = vec![0.0f32; tokens * heads * dim];
        let mut rotated = q.clone();
        for i in 0..tokens {
            for h in 0..heads {
                let at = (i * heads + h) * dim;
                rope.apply(&mut rotated[at + dim - rd..at + dim], start + i, false);
                let out = &mut want[at..at + dim];
                crate::dsv41::attention::sparse_attend(&rotated[at..at + dim], &kv, dim, &idxs[i], sink[h], scale, out);
                rope.apply(&mut out[dim - rd..], start + i, true);
            }
        }
        let pos = words(&(start..start + tokens).collect::<Vec<_>>());
        let picks: Vec<f32> = idxs.iter().flatten().map(|&j| f32::from_bits(if j < 0 { u32::MAX } else { j as u32 })).collect();
        let rot = |x: usize, inverse: bool| G::RopeRows { x, rows: tokens * heads, stride: dim, at: dim - rd, half: rd / 2, cos: 1, sin: 2, pos: 3, group: heads, inverse };
        let steps = [G::Input(&q), G::State(&cos_v), G::State(&sin_v), G::Input(&pos), rot(0, false), G::Input(&kv), G::Input(&picks), G::AttendRows { q: 0, kv: 5, idxs: 6, tokens, heads, dim, len, sink: &sink, scale }, rot(7, true)];
        let got = run_program(&b, &steps, &[(0, 0, q.len()), (7, 0, want.len())]).expect("a program");
        assert_eq!(bits(&got[0]), bits(&rotated), "{tokens} tokens' queries rotated");
        let same = got[1].iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let rel = |g: &[f32], w: &[f32]| g.iter().zip(w).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / w.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
        let worst = got[1].chunks(dim).zip(want.chunks(dim)).map(|(g, w)| rel(g, w)).fold(0.0f32, f32::max);
        eprintln!("{tokens} tokens from {start}: {same} of {} outputs the host's to the bit, the farthest {worst:.2e} off", want.len());
        assert!(same * 2 >= want.len() && worst < 1e-2, "{tokens} tokens: {same} of {} the same, the farthest {worst}", want.len());
    }
}

/// A decode step's attention on the device is the host's but for its exponentials: every head over a window ring (the
/// step early in a sequence, its later slots none, and later) and 512 picked compressed rows (some none), the keys
/// quantized as the caches hold them; nearly every output the host's bf16 value and none far from it. And a pool and
/// a stream's mixing coefficients near the host's.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn a_steps_attention_on_the_device_is_the_hosts_but_for_its_exponentials() {
    let b = gpu();
    let mut r = Values(0x0123_4567_89ab_cdefu64);
    let (heads, dim, win, k, rows) = (64usize, 512usize, 128usize, 512usize, 3000usize);
    // (queries of two sizes: a softmax with one position far ahead, and one spread over many)
    let mut q = r.bf16(heads * dim / 2, 3.0);
    q.extend(r.bf16(heads * dim / 2, 0.02));
    let mut window = r.bf16(win * dim, 2.0);
    for row in window.chunks_mut(dim) {
        fake_quant_fp8_inplace(row);
    }
    let mut compressed = r.bf16(rows * dim, 2.0);
    for row in compressed.chunks_mut(dim) {
        fake_quant_fp4_inplace(row, 16, Fp4Scale::E4M3);
    }
    let sink = r.bf16(heads, 2.0);
    let scale = (dim as f32).powf(-0.5);
    let (win_v, rows_v) = (kept(&b, &window), kept(&b, &compressed));
    for pos in [5usize, 127, 1000] {
        let mut picks: Vec<i32> = (0..k).map(|i| if i % 37 == 5 { -1 } else { ((r.next() as usize) % rows) as i32 }).collect();
        picks.sort_unstable();
        let pick_words: Vec<f32> = picks.iter().map(|&p| f32::from_bits(if p < 0 { u32::MAX } else { p as u32 })).collect();
        let picks_v = kept(&b, &pick_words);
        // the host's: the ring oldest first, then the picks, over the window and the compressed rows laid end to end
        let oldest = pos % win + 1;
        let mut idxs: Vec<i32> = (oldest..win).chain(0..oldest).map(|s| if s > pos { -1 } else { s as i32 }).collect();
        idxs.extend(picks.iter().map(|&p| if p < 0 { -1 } else { p + win as i32 }));
        let mut all = window.clone();
        all.extend_from_slice(&compressed);
        let mut want = vec![0.0f32; heads * dim];
        for (h, out) in want.chunks_mut(dim).enumerate() {
            crate::dsv41::attention::sparse_attend(&q[h * dim..(h + 1) * dim], &all, dim, &idxs, sink[h], scale, out);
        }
        let wd = words(&[pos]);
        let steps = [G::Input(&q), G::State(&win_v), G::State(&rows_v), G::State(&picks_v), G::Input(&wd), G::Attend { q: 0, heads, dim, window: 1, win, compressed: Some((2, 3, k)), sink: &sink, scale, pos: 4, word: 0 }];
        let got = run_program(&b, &steps, &[(5, 0, heads * dim)]).expect("a program");
        let same = got[0].iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        // (off by at most a part in a hundred of the head's largest output)
        let rel = |g: &[f32], w: &[f32]| g.iter().zip(w).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / w.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
        let worst = got[0].chunks(dim).zip(want.chunks(dim)).map(|(g, w)| rel(g, w)).fold(0.0f32, f32::max);
        eprintln!("attention at {pos}: {same} of {} outputs the host's to the bit, the farthest {worst:.2e} off", want.len());
        assert!(same * 2 >= want.len() && worst < 1e-2, "attention at {pos}: {same} of {} the same, the farthest {worst}", want.len());
    }
    // a ratio-2 compressor's pool
    let kv: Vec<f32> = (0..2 * dim).map(|_| (r.next() % 2001) as f32 / 1000.0 - 1.0).collect();
    let score: Vec<f32> = (0..2 * dim).map(|_| (r.next() % 2001) as f32 / 300.0 - 3.0).collect();
    let want: Vec<f32> = crate::dsv41::attention::pool(&kv, &score, 2, dim).into_iter().map(to_bf16).collect();
    let steps = [G::Input(&kv), G::Input(&score), G::Pool { kv: 0, score: 1, ratio: 2, width: dim }];
    let got = run_program(&b, &steps, &[(2, 0, dim)]).expect("a program");
    let worst = got[0].iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 1e-2, "the pool at most {worst} off");
    // a stream's mixing coefficients from its projections
    let n = 4 * 5120;
    let h = r.bf16(n, 1.5);
    let proj: Vec<f32> = (0..24).map(|_| (r.next() % 2001) as f32 / 10.0 - 100.0).collect();
    let p = crate::dsv41::hc::HcParams::new(vec![0.0; 24 * n], (0..24).map(|i| i as f32 * 0.1 - 1.0).collect(), vec![0.5, 0.7, 1.3]);
    let want = crate::dsv41::hc::mixes_of(&proj, &h, &p, 1e-20, 20, 1e-6).flat();
    let (base, sc) = p.base_scale();
    let sumsq = [crate::dsv41::hc::sumsq(&h)];
    let steps = [G::Input(&proj), G::Input(&sumsq), G::HcMix { proj: 0, pos: 1, word: 0, base, scale: sc, n, norm_eps: 1e-20, iters: 20, hc_eps: 1e-6 }];
    let got = run_program(&b, &steps, &[(2, 0, 24)]).expect("a program");
    let worst = got[0].iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 1e-5, "the mixing at most {worst} off: {:?} vs {:?}", got[0], want);
}
