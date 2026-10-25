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
