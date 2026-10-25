//! `exl3`'s tests.

use super::*;
use ggml_rs::exl3::mul1;

/// An independent packing oracle: each weight's bits written MSB first into its tile's stream, and the dense
/// matrix they decode to (the same construction as ggml-rs-cuda's EXL3 test, which checks the CUDA kernels).
fn oracle(tw: usize, k: usize, n: usize) -> (Exl3Data, Vec<f32>) {
    let nw = tw / 2;
    let mut words = vec![0u32; k / 16 * n / 16 * nw];
    let mut dense = vec![0f32; k * n];
    for kt in 0..k / 16 {
        for nt in 0..n / 16 {
            let mut stream = vec![];
            for i in 0..256 {
                let bits = tw / 16 + if tw % 16 == 8 { i % 2 } else { 0 };
                let v = (i * 17 + kt * 43 + nt * 11 + 3) as u32 & ((1 << bits) - 1);
                for bit in (0..bits).rev() {
                    stream.push((v >> bit) & 1);
                }
            }
            let base = (kt * (n / 16) + nt) * nw;
            for (i, &bit) in stream.iter().enumerate() {
                words[base + i / 32] |= bit << (31 - i % 32);
            }
            let mut end = 0;
            for i in 0..256 {
                end += tw / 16 + if tw % 16 == 8 { i % 2 } else { 0 };
                let mut code = 0;
                for j in 0..16 {
                    code = (code << 1) | stream[(end + stream.len() - 16 + j) % stream.len()];
                }
                let lane = i / 8;
                let j = i % 8;
                let r = (lane % 4) * 2 + (j % 2) + if j & 2 != 0 { 8 } else { 0 };
                let c = lane / 4 + if j & 4 != 0 { 8 } else { 0 };
                dense[(kt * 16 + r) * n + nt * 16 + c] = mul1(code);
            }
        }
    }
    let data = Exl3Data {
        words,
        suh: (0..k).map(|i| if i % 3 == 0 { -0.5 } else { 0.5 }).collect(),
        svh: (0..n).map(|i| if i % 5 == 0 { -0.25 } else { 0.25 }).collect(),
        tile_words: tw,
        input_map: (0..k as u32).rev().collect(),
        output_map: (0..n as u32).rev().collect(),
    };
    (data, dense)
}

fn inputs(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k).map(|i| ((i % k * 7 + i / k * 13) % 23) as f32 / 32.0 - 0.25).collect()
}

/// The projection the oracle's dense matrix gives, step by step as exllamav3 rounds.
fn expected(data: &Exl3Data, dense: &[f32], x: &[f32]) -> Vec<f32> {
    let (k, n) = (data.suh.len(), data.svh.len());
    let mut out = vec![];
    for row in x.chunks_exact(k) {
        let mut xh: Vec<f32> = data.input_map.iter().enumerate().map(|(i, &j)| half(row[j as usize]) * data.suh[i]).collect();
        had(&mut xh);
        for v in &mut xh {
            *v = half(*v * ISQRT128);
        }
        let mut y: Vec<f32> = (0..n).map(|j| half((0..k).map(|i| xh[i] * dense[i * n + j]).sum())).collect();
        had(&mut y);
        out.extend(data.output_map.iter().map(|&j| half(y[j as usize] * ISQRT128 * data.svh[j as usize])));
    }
    out
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 0.003, "{what} [{i}]: {a} != {e}");
    }
}

const RATES: [usize; 11] = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128];

#[test]
fn the_cpu_projection_matches_the_independent_oracle_at_every_rate() {
    for tw in RATES {
        let (k, n) = (128, 256);
        let (data, dense) = oracle(tw, k, n);
        for i in 0..k {
            for j in 0..n {
                assert_eq!(data.value(i, j), dense[i * n + j], "the oracle and Exl3Data::value disagree: tw={tw} k={i} n={j}");
                let (w0, w1, sh) = positions(tw)[(i % 16) * 16 + j % 16];
                let base = ((i / 16) * (n / 16) + j / 16) * (tw / 2);
                let pair = ((data.words[base + w0] as u64) << 32) | data.words[base + w1] as u64;
                assert_eq!(decode((pair >> sh) as u32 & 0xffff), dense[i * n + j], "decode: tw={tw} k={i} n={j}");
            }
        }
        for rows in [1, 5] {
            let x = inputs(rows, k);
            let want = expected(&data, &dense, &x);
            let cpu = Exl3Cpu::new(Exl3Data { words: data.words.clone(), suh: data.suh.clone(), svh: data.svh.clone(), tile_words: tw, input_map: data.input_map.clone(), output_map: data.output_map.clone() }).unwrap();
            let y = cpu.linear(&Tensor::from_vec(x, vec![rows, k]));
            assert_eq!(y.shape(), &[rows, n]);
            close(y.data(), &want, &format!("cpu tw={tw} rows={rows}"));
        }
    }
}

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(Some(1 << 30)) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping WebGPU EXL3 tests: {e}");
            None
        }
    }
}

#[test]
fn the_gpu_projection_matches_the_independent_oracle_at_every_rate() {
    let Some(b) = backend() else { return };
    for tw in RATES {
        let (k, n) = (128, 256);
        let (data, dense) = oracle(tw, k, n);
        // More rows than one pass takes (32), and fewer.
        for rows in [1, 5, 37] {
            let x = inputs(rows, k);
            let want = expected(&data, &dense, &x);
            let w = b.exl3(Exl3Data { words: data.words.clone(), suh: data.suh.clone(), svh: data.svh.clone(), tile_words: tw, input_map: data.input_map.clone(), output_map: data.output_map.clone() }).unwrap();
            assert!(format!("{w:?}").starts_with("Exl3Gpu"), "on the GPU: {w:?}");
            let y = w.linear(&Tensor::from_vec(x, vec![rows, k]));
            close(y.data(), &want, &format!("gpu tw={tw} rows={rows}"));
        }
    }
}

#[test]
fn a_projection_over_several_buffers_and_splits_agrees_with_the_cpu() {
    let Some(b) = backend() else { return };
    // A model-like width, buffers of 3 tile rows (as a small binding limit would make them), split work.
    let (k, n, tw) = (1024, 512, 48);
    let mut seed = 13_234_567u32;
    let words: Vec<u32> = (0..k / 16 * n / 16 * (tw / 2))
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        })
        .collect();
    let data = || Exl3Data {
        words: words.clone(),
        suh: (0..k).map(|i| if i % 3 == 0 { -0.25 } else { 0.25 }).collect(),
        svh: vec![0.125; n],
        tile_words: tw,
        input_map: (0..k as u32).rev().collect(),
        output_map: (0..n as u32).rev().collect(),
    };
    let gpu = b.exl3_with(data(), Some(3)).unwrap();
    assert!(format!("{gpu:?}").contains("chunk(s)") && !format!("{gpu:?}").contains(" 1 chunk"), "{gpu:?}");
    let cpu = exl3_cpu(data()).unwrap();
    let x = Tensor::from_vec((0..3 * k).map(|i| ((i * 17 % 73) as f32 - 36.0) / 37.0).collect(), vec![3, k]);
    let (g, c) = (gpu.linear(&x), cpu.linear(&x));
    close(g.data(), c.data(), "gpu vs cpu");
}

/// Projections of several shapes and bitrates (one over several buffers) in one call, a decode step's row and a prompt's
/// rows past a pass's, are each the CPU's projection of its rows; one on the CPU is not taken.
#[test]
fn several_projections_in_one_submit_are_each_the_cpus() {
    let Some(b) = backend() else { return };
    let shapes = [(256, 512, 48), (512, 256, 32), (1024, 512, 64), (256, 128, 24)];
    let lins: Vec<_> = shapes.iter().enumerate().map(|(i, &(k, n, tw))| b.exl3_with(random_exl3(k, n, tw, 70 + i as u32), (i == 2).then_some(3)).unwrap()).collect();
    let cpus: Vec<_> = shapes.iter().enumerate().map(|(i, &(k, n, tw))| exl3_cpu(random_exl3(k, n, tw, 70 + i as u32)).unwrap()).collect();
    for m in [1, 3, ROWS + 5] {
        let xs: Vec<Vec<f32>> = shapes.iter().enumerate().map(|(i, &(k, ..))| (0..m * k).map(|j| (((j + i) * 29 % 83) as f32 - 41.0) / 43.0).collect()).collect();
        let items: Vec<(&dyn PackedLinear, &[f32])> = lins.iter().zip(&xs).map(|(l, x)| (&**l, &x[..])).collect();
        let got = exl3_many(&items).expect("all on one GPU");
        for (i, ((c, x), y)) in cpus.iter().zip(&xs).zip(&got).enumerate() {
            let want = c.linear(&Tensor::from_vec(x.clone(), vec![m, shapes[i].0]));
            close(y, want.data(), &format!("projection {i}, {m} rows"));
        }
    }
    let cpu = exl3_cpu(random_exl3(256, 128, 24, 9)).unwrap();
    let x = vec![0.5f32; 256];
    assert!(exl3_many(&[(&*lins[3], &x[..]), (&*cpu, &x[..])]).is_none(), "one on the CPU is not taken");
}

/// An f16 matrix's scratch in a chained recording is never a later item's input: DeepSeek-V4.1's router (384 x 5120)
/// over a prompt's 15 rows, then an EXL3 projection of the same 5120 inputs (its input the size of the router's
/// partial sums), each the host's sums.
#[test]
fn an_f16_matmuls_scratch_is_not_a_later_items_input() {
    let Some(b) = backend() else { return };
    let (n, k, rows) = (384, 5120, 15);
    let w: Vec<f32> = (0..n * k).map(|i| ((i * 29 % 61) as f32 - 30.0) / 64.0).collect();
    let wv = ggml_rs::DeviceChain::vec_f16(&b, &w).expect("f16 values exactly");
    let lin = b.exl3_with(random_exl3(k, 256, 32, 5), None).unwrap();
    let cpu = exl3_cpu(random_exl3(k, 256, 32, 5)).unwrap();
    let x: Vec<f32> = (0..rows * k).map(|i| ((i * 13 % 37) as f32 - 18.0) / 32.0).collect();
    let got = chained(&b, &[(ChainMat::F16 { w: &wv, n, k }, &x[..]), (ChainMat::Exl3(&*lin), &x[..])]).expect("both kinds chained");
    let want: Vec<f32> = (0..rows).flat_map(|r| (0..n).map(|o| (0..k).map(|i| w[o * k + i] * x[r * k + i]).sum::<f32>()).collect::<Vec<_>>()).collect();
    close(&got[0], &want, "the router's sums");
    close(&got[1], cpu.linear(&Tensor::from_vec(x.clone(), vec![rows, k])).data(), "the projection after it");
}

/// What a decode step's call of experts held in slots costs (`--ignored --nocapture`), at DeepSeek-V4.1's shape (5120 to
/// 2304, 3 bits): three experts of one row in a call of their own, then sixteen such sets in one call (the GPU's share
/// of a set, the call's own costs spread over them), and one expert alone.
#[test]
#[ignore = "a measurement"]
fn measure_slots_decode_call() {
    let Some(b) = backend() else { return };
    let (hidden, ff, tw) = (5120, 2304, 48);
    let (layout, records, _) = as_records(experts(4, hidden, ff, tw));
    let slots = b.exl3_slots(4, hidden, ff, [tw, tw]).expect("room for four slots");
    for s in 0..4 {
        slots.put(s, &records.0[s], &layout);
    }
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    let three: Vec<(usize, &[f32])> = (0..3).map(|s| (s, &x[..])).collect();
    let time = |what: &str, f: &dyn Fn()| {
        for _ in 0..5 {
            f();
        }
        let t = std::time::Instant::now();
        for _ in 0..50 {
            f();
        }
        eprintln!("{what}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 50.0);
    };
    time("3 experts of a row, a call", &|| drop(slots.forward(&three, 10.0)));
    time("1 expert of a row, a call", &|| drop(slots.forward(&three[..1], 10.0)));
    let parts: Vec<(&Exl3Slots, &[(usize, &[f32])])> = (0..16).map(|_| (&slots, &three[..])).collect();
    time("16 sets of 3 in one call (/16 for a set)", &|| drop(Exl3Slots::forward_many(&parts, 10.0)));
    // a step's held experts as a card has them: six of a layer in three pools
    let pools: Vec<Exl3Slots> = (0..3).map(|_| b.exl3_slots(2, hidden, ff, [tw, tw]).expect("room for two slots")).collect();
    for p in &pools {
        p.put_many(0, &[&records.0[0][..], &records.0[1][..]], &layout);
    }
    let two: Vec<(usize, &[f32])> = (0..2).map(|s| (s, &x[..])).collect();
    let parts: Vec<(&Exl3Slots, &[(usize, &[f32])])> = pools.iter().map(|p| (p, &two[..])).collect();
    time("6 experts in 3 pools, a call", &|| drop(Exl3Slots::forward_many(&parts, 10.0)));
}

/// A program run again and again (its later runs replayed from its first's recording)
/// reads each run's own inputs: each run's two parts the definitions of that run's values, the second read after the
/// first.
#[test]
fn a_program_run_again_reads_each_runs_inputs() {
    let Some(b) = backend() else { return };
    let d = 1024;
    let bf = |v: f32| f32::from_bits((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff_0000);
    for run in 0..5u32 {
        let h: Vec<f32> = (0..4 * d).map(|i| bf(((i as u32 * 37 + run * 101) % 97) as f32 / 48.0 - 1.0)).collect();
        let mix: Vec<f32> = (0..24).map(|i| ((i as u32 * 7 + run * 3) % 11) as f32 / 16.0 + 0.1).collect();
        let steps = [ProgramStep::Input(&h), ProgramStep::Input(&mix), ProgramStep::HcPre { h: 0, mix: 1, d }, ProgramStep::HcPre { h: 0, mix: 1, d }];
        let (now, later) = run_program_parts(&b, &steps, &[(2, 0, d)], 3, &[(3, 0, d)]).expect("the program runs");
        let later = later.wait();
        let want: Vec<f32> = (0..d).map(|k| bf(mix[0] * h[k] + mix[1] * h[d + k] + mix[2] * h[2 * d + k] + mix[3] * h[3 * d + k])).collect();
        let near = |got: &[f32]| got.iter().zip(&want).all(|(a, e)| a == e || (a - e).abs() <= e.abs() / 128.0);
        assert!(near(&now[0]), "run {run}: the first part's");
        assert!(near(&later[0]), "run {run}: the second part's");
    }
}

/// A program's held experts ([`ProgramStep::Candidates`], then [`ProgramStep::Held`]) are the `k` best of the router's
/// scores, those the table holds made as the slots make them alone (to the bit) and those it does not skipped, the
/// experts and their slots read back as the table has them; at DeepSeek-V4.1's shape, 3 bits a weight.
#[test]
fn a_programs_held_experts_are_the_slots_own() {
    let Some(b) = backend() else { return };
    let (hidden, ff, tw, limit, n, k) = (5120, 2304, 48, 10.0f32, 16usize, 4usize);
    let (layout, records, _) = as_records(experts(3, hidden, ff, tw));
    let slots = b.exl3_slots(4, hidden, ff, [tw, tw]).expect("room for four slots");
    // experts 3, 7 and 11 of the layer in slots 2, 0 and 1
    let held = [(3usize, 2u32), (7, 0), (11, 1)];
    for (r, &(_, s)) in held.iter().enumerate() {
        slots.put(s as usize, &records.0[r], &layout);
    }
    let mut words = vec![u32::MAX; n];
    for &(e, s) in &held {
        words[e] = s;
    }
    let table = b.vec_of_bytes(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    // logits that put experts 7, 5, 3 and 12 first by score (the bias moving 12 above 11), 11 fifth
    let mut logits = vec![-3.0f32; n];
    for (e, v) in [(7, 4.0), (5, 3.0), (3, 2.0), (11, 1.5), (12, 1.0)] {
        logits[e] = v;
    }
    let mut bias = vec![0.0f32; n];
    bias[12] = 0.2;
    let steps = [
        ProgramStep::Input(&logits),
        ProgramStep::Input(&x),
        ProgramStep::Candidates { logits: 0, bias: &bias, table: &table, k },
        ProgramStep::Held { slots: HeldSlots::Exl3(&slots), x: 1, cands: 2, limit },
    ];
    let got = run_program(&b, &steps, &[(2, 0, 4 * k), (3, 0, k * hidden)]).expect("the program runs");
    let bits: Vec<u32> = got[0].iter().map(|v| v.to_bits()).collect();
    assert_eq!(&bits[2 * k..3 * k], &[7, 5, 3, 12], "the candidates, best first");
    assert_eq!(&bits[3 * k..], &[0, u32::MAX, 2, u32::MAX], "their slots");
    for (r, slot) in [(0usize, 0usize), (2, 2)] {
        let alone = slots.forward(&[(slot, &x[..])], limit).pop().expect("one output");
        let mine = &got[1][r * hidden..(r + 1) * hidden];
        assert!(mine.iter().zip(&alone).all(|(a, b)| a.to_bits() == b.to_bits()), "candidate {r}: the slots' own output");
    }
}

/// A decode step's expert made on the host ([`step_expert_host`], and several at once on its cores:
/// [`step_experts_host`]) is a card's ([`Exl3Slots`]' job of one row) to the bit, at DeepSeek-V4.1's shape (5120 to
/// 2304) at 3 bits a weight and at 2, on small inputs and large: the transforms, the one-row kernel's sums in its order
/// (each lane's products added by multiply-adds, as the card's compiler makes them), and the SwiGLU from the table both
/// read. (It says how many values the sums added by a multiply and an add would differ in.)
#[test]
fn a_steps_expert_on_the_host_is_the_cards_to_the_bit() {
    let Some(b) = backend() else { return };
    let (hidden, ff, limit) = (5120, 2304, 10.0);
    let table = silu_table(limit);
    for tw in [48, 32] {
        let (layout, records, _) = as_records(experts(2, hidden, ff, tw));
        let slots = b.exl3_slots(2, hidden, ff, [tw, tw]).expect("room for two slots");
        slots.put_many(0, &[&records.0[0][..], &records.0[1][..]], &layout);
        let (mut fused_off, mut plain_off) = (0, 0);
        for (n, scale) in [0.05f32, 3.0, 0.4, 1.0, 7.0, 0.2, 2.0, 0.7].into_iter().enumerate() {
            let i = n % 2;
            let mut s = 0x9e37_79b9u32 ^ (n as u32 * 7919);
            let x: Vec<f32> = (0..hidden)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    ((s % 20_001) as f32 / 10_000.0 - 1.0) * scale
                })
                .collect();
            let card = slots.forward(&[(i, &x[..])], limit).pop().unwrap();
            let differ = |v: &[f32]| v.iter().zip(&card).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let fused = step_expert_host(&records.0[i], &layout, &x, limit, &table, true);
            let plain = step_expert_host(&records.0[i], &layout, &x, limit, &table, false);
            fused_off += differ(&fused);
            plain_off += differ(&plain);
            assert_eq!(differ(&fused), 0, "tw={tw} scale={scale}: the host's step is the card's");
            if n < 2 {
                let both = step_experts_host(&[(&records.0[i][..], &x[..]), (&records.0[1 - i][..], &x[..])], &layout, limit, true);
                assert_eq!(differ(&both[0]), 0, "tw={tw} scale={scale}: on the host's cores, beside another");
            }
        }
        eprintln!("tw={tw}: of 8 inputs' 5120 values, the host's with multiply-adds differs in {fused_off}, with a multiply and an add in {plain_off}");
    }
}

/// A group of projections of one shape ([`WgpuBackend::exl3_group`], DeepSeek-V4.1's `wo_a` slices) in a program, each
/// part on its own row of the input, gives each part's own projection's sums to the bit (as a chained recording makes
/// them alone), at 3 bits a weight and at 2.
#[test]
fn a_groups_parts_in_a_program_are_each_parts_own_projection() {
    let Some(b) = backend() else { return };
    let (k, n, parts) = (512usize, 256usize, 4usize);
    for tw in [48, 32] {
        let datas: Vec<Exl3Data> = (0..parts).map(|p| random_exl3(k, n, tw, 500 + tw as u32 + p as u32)).collect();
        let group = b.exl3_group(&datas.iter().collect::<Vec<_>>()).expect("a group");
        let alone: Vec<Arc<dyn PackedLinear>> = datas.iter().map(|d| b.exl3_with(Exl3Data { words: d.words.clone(), suh: d.suh.clone(), svh: d.svh.clone(), tile_words: d.tile_words, input_map: d.input_map.clone(), output_map: d.output_map.clone() }, None).unwrap()).collect();
        let x: Vec<f32> = (0..parts * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        let jobs: Vec<(u32, u32)> = (0..parts as u32).map(|p| (p, p)).collect();
        let got = run_program(&b, &[ProgramStep::Input(&x), ProgramStep::MulGroup { g: GroupMat::Exl3(&group), x: 0, jobs: &jobs }], &[(1, 0, parts * n)]).expect("the program runs").pop().unwrap();
        for p in 0..parts {
            let want = chained(&b, &[(ChainMat::Exl3(&*alone[p]), &x[p * k..(p + 1) * k])]).expect("alone").pop().unwrap();
            assert!(got[p * n..(p + 1) * n].iter().zip(&want).all(|(a, e)| a.to_bits() == e.to_bits()), "tw={tw}: part {p} in the group is its own projection");
        }
    }
}

/// A program's hyper-connection steps ([`run_program`]) are their definitions within a rounding: the post (copy `j`
/// of `bf16(post[j] out + sum_i comb[i][j] res[i])`, `comb[i][j]` at `8 + 4 i + j` of the mix), the pre
/// (`bf16(sum_i pre[i] h[i])`), and the host's norm of that to the bit; an f32 matrix's sums of the stream beside
/// them; on random values of the model's width. (The card may fuse a product into a sum, so a value is the
/// definition's or one bf16 step from it.)
#[test]
fn a_programs_hyper_connection_steps_are_their_definitions() {
    let Some(b) = backend() else { return };
    let d = 5120;
    let mut s = 0x9e37_79b9u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 20_001) as f32 / 10_000.0 - 1.0
    };
    let bf = |v: f32| f32::from_bits((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff_0000);
    let out: Vec<f32> = (0..d).map(|_| bf(next() * 3.0)).collect();
    let h: Vec<f32> = (0..4 * d).map(|_| bf(next() * 2.0)).collect();
    let mix: Vec<f32> = (0..24).map(|i| if i < 8 { next().abs() + 0.2 } else { next().abs() / 4.0 }).collect();
    let norm: Vec<f32> = (0..d).map(|i| 0.5 + (i % 9) as f32 / 8.0).collect();
    let fnw: Vec<f32> = (0..24 * 4 * d).map(|_| next() * 0.01).collect();
    let fnv = b.vec(fnw.len());
    DeviceChain::upload(&b, &fnv, &fnw);
    let steps = [
        ProgramStep::Input(&out),
        ProgramStep::Input(&h),
        ProgramStep::Input(&mix),
        ProgramStep::HcPost { out: 0, res: 1, mix: 2, d },
        ProgramStep::HcPre { h: 3, mix: 2, d },
        ProgramStep::Norm { x: 4, weight: &norm, eps: 1e-20, rows: 1, width: d },
        ProgramStep::Mul { m: ChainMat::F32 { w: &fnv, n: 24, k: 4 * d }, x: 3, rows: 1 },
    ];
    let got = run_program(&b, &steps, &[(3, 0, 4 * d), (4, 0, d), (5, 0, d), (6, 0, 24)]).expect("the program runs");
    let near = |a: f32, e: f32| a == e || (a - e).abs() <= e.abs() / 128.0;
    // the post, then the pre of the card's post, then the host's norm of the card's pre
    for j in 0..4 {
        for k in 0..d {
            let r = (0..4).map(|i| mix[8 + 4 * i + j] * h[i * d + k]).sum::<f32>();
            let want = bf(mix[4 + j] * out[k] + r);
            assert!(near(got[0][j * d + k], want), "post copy {j} at {k}: {} where {want}", got[0][j * d + k]);
        }
    }
    let h1 = &got[0];
    for k in 0..d {
        let want = bf((0..4).map(|i| mix[i] * h1[i * d + k]).sum::<f32>());
        assert!(near(got[1][k], want), "pre at {k}: {} where {want}", got[1][k]);
    }
    let x = &got[1];
    let var = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
    let r = 1.0 / (var + 1e-20).sqrt();
    let want: Vec<f32> = x.iter().zip(&norm).map(|(v, g)| bf(g * (v * r))).collect();
    assert_eq!(got[2], want, "the norm of the card's pre, to the bit");
    for (row, &g) in got[3].iter().enumerate() {
        let want: f64 = fnw[row * 4 * d..(row + 1) * 4 * d].iter().zip(h1.iter()).map(|(w, v)| *w as f64 * *v as f64).sum();
        assert!((g as f64 - want).abs() <= 1e-3 * (1.0 + want.abs()), "the stream's projection {row}: {g} where {want}");
    }
}

/// A program's gated activation is the host's to the bit: its projections rounded to bf16, the gate's silu by the table,
/// the up clamped (none with no limit), their product rounded to bf16; with the model's limit and with none.
#[test]
fn a_programs_gated_activation_is_the_hosts_to_the_bit() {
    let Some(b) = backend() else { return };
    let len = 3072;
    let mut s = 0x2545_f491u32;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 40_001) as f32 / 1_000.0 - 20.0
    };
    let bf = |v: f32| f32::from_bits((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff_0000);
    let silu = |g: f32| g / (1.0 + (-g).exp());
    // (sums as a projection leaves them: not yet rounded)
    let gate: Vec<f32> = (0..len).map(|_| next() * 1.000_123).collect();
    let up: Vec<f32> = (0..len).map(|_| next() * 0.999_877).collect();
    for limit in [10.0f32, 0.0] {
        let table: Vec<f32> = (0..1u32 << 16).map(|bits| f32::from_bits(bits << 16)).map(|g| silu(if limit > 0.0 { g.min(limit) } else { g })).collect();
        let steps = [ProgramStep::Input(&gate), ProgramStep::Input(&up), ProgramStep::SwigluBf16 { gate: 0, up: 1, len, silu: &table, limit }];
        let got = run_program(&b, &steps, &[(2, 0, len)]).expect("the program runs");
        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let (g, u) = (bf(g), bf(u));
                let (g, u) = if limit > 0.0 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
                bf(silu(g) * u)
            })
            .collect();
        assert_eq!(got[0], want, "the activation with limit {limit}");
    }
}

/// The one-row kernel's sums on a card, before any rounding, are the host's emulation's ([`ProjAt::step_sums`]'s) with
/// each lane's products added by multiply-adds, as the card's compiler makes them (the rounding to f16 after hides
/// which in all but a few values, so the raw sums say it), at 3 bits a weight and at 2.
#[test]
fn the_one_row_kernels_sums_are_the_hosts_multiply_adds() {
    let Some(b) = backend() else { return };
    let (k, n) = (5120, 2304);
    for tw in [48, 32] {
        let data = random_exl3(k, n, tw, 900 + tw as u32);
        let words = data.words.clone();
        let g = Exl3Gpu::upload(&b, data, None);
        let mut s = 0x2545_f491u32;
        let xh: Vec<f32> = (0..k)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                half((s % 20_001) as f32 / 10_000.0 - 1.0)
            })
            .collect();
        let mut enc = g.gpu.device.create_command_encoder(&Default::default());
        let pass = g.record(&mut enc, &xh, 1);
        let card = read_back(&g.gpu, enc, &[pass]).pop().unwrap();
        let off = |v: &[f32]| v.iter().zip(&card).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let differ = |fused: bool| off(&one_row_sums(&words[..], k, n, tw, &xh, step_splits(k, n), 0..n / 16, fused));
        let (fused, plain) = (differ(true), differ(false));
        eprintln!("tw={tw}: of {n} sums, the host's with multiply-adds differ in {fused}, with a multiply and an add in {plain}");
        assert_eq!(fused, 0, "tw={tw}: the card's sums are multiply-adds");
        // and the host's own (its vectors' where it has them), of every column and of a run of them
        let record: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ProjAt { suh: 0, svh: 0, mul1: 0, trellis: 0, k, n, tile_words: tw };
        assert_eq!(off(&p.step_sums(&record, &xh, 0..n / 16, true)), 0, "tw={tw}: the host's step sums");
        let part = p.step_sums(&record, &xh, 5..13, true);
        assert!(part.iter().zip(&card[5 * 16..13 * 16]).all(|(a, b)| a.to_bits() == b.to_bits()), "tw={tw}: a run of columns");
    }
}

/// The host's vector sums of a step's row ([`ProjAt::step_sums`], AVX-512's where the CPU has it) are its emulation's
/// with multiply-adds ([`one_row_sums`], which the test above holds to a card's) to the bit, at 2, 3 and 4 bits a weight,
/// of every column and of a run of them: on the CPU alone.
#[test]
fn the_hosts_vector_sums_are_its_emulations() {
    let (k, n) = (5120, 2304);
    for tw in [32, 48, 64] {
        let data = random_exl3(k, n, tw, 300 + tw as u32);
        let record: Vec<u8> = data.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut s = 0x9e37_79b9u32;
        let xh: Vec<f32> = (0..k)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                half((s % 20_001) as f32 / 10_000.0 - 1.0) * if s >> 28 == 0 { 40.0 } else { 1.0 }
            })
            .collect();
        let want = one_row_sums(&data.words[..], k, n, tw, &xh, step_splits(k, n), 0..n / 16, true);
        let p = ProjAt { suh: 0, svh: 0, mul1: 0, trellis: 0, k, n, tile_words: tw };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p.step_sums(&record, &xh, 0..n / 16, true)), bits(&want), "tw={tw}: every column");
        assert_eq!(bits(&p.step_sums(&record, &xh, 5..13, true)), bits(&want[5 * 16..13 * 16]), "tw={tw}: a run of columns");
    }
}

/// A step's experts on the host's cores together ([`step_experts_host`]'s tasks of a split's rows) are each as the host
/// makes it alone ([`step_expert_host`], which the test above holds to a card's) to the bit: one, two and three a call,
/// at 3 bits a weight and at 2: on the CPU alone.
#[test]
fn the_hosts_step_experts_together_are_each_alone() {
    let (hidden, ff, limit) = (5120, 2304, 10.0);
    let table = silu_table(limit);
    for tw in [48, 32] {
        let (layout, records, _) = as_records(experts(3, hidden, ff, tw));
        let xs: Vec<Vec<f32>> = (0..3u32)
            .map(|e| {
                let mut s = 0x2545_f491u32 ^ (e * 7919 + tw as u32);
                (0..hidden)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        ((s % 20_001) as f32 / 10_000.0 - 1.0) * [0.3f32, 2.0, 7.0][e as usize]
                    })
                    .collect()
            })
            .collect();
        let alone: Vec<Vec<u32>> = (0..3).map(|e| step_expert_host(&records.0[e], &layout, &xs[e], limit, &table, true).iter().map(|v| v.to_bits()).collect()).collect();
        for n in 1..=3 {
            let jobs: Vec<(&[u8], &[f32])> = (0..n).map(|e| (&records.0[e][..], &xs[e][..])).collect();
            for (e, y) in step_experts_host(&jobs, &layout, limit, true).iter().enumerate() {
                assert_eq!(y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), alone[e], "tw={tw}: expert {e} of {n} a call");
            }
        }
    }
}

/// What a decode step's experts cost on the host's cores (`--ignored --nocapture`) at DeepSeek-V4.1's shape (5120 to
/// 2304, 3 bits): one, two and six experts a call, each its own record, records in turn past the caches as a step's are.
#[test]
#[ignore = "a measurement"]
fn measure_step_experts_on_the_host() {
    let (hidden, ff, tw, limit) = (5120, 2304, 48, 10.0);
    let (layout, records, _) = as_records(experts(12, hidden, ff, tw));
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    for n in [1usize, 2, 6] {
        let mut at = 0;
        let mut call = || {
            let jobs: Vec<(&[u8], &[f32])> = (0..n).map(|j| (&records.0[(at + j) % 12][..], &x[..])).collect();
            at += n;
            drop(step_experts_host(&jobs, &layout, limit, true));
        };
        for _ in 0..3 {
            call();
        }
        let t = std::time::Instant::now();
        for _ in 0..60 {
            call();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 60.0;
        eprintln!("{n} expert(s) a call: {ms:.2} ms ({:.2} ms an expert)", ms / n as f64);
    }
    // the same six records every call (in the caches): what the reads from memory cost
    let jobs: Vec<(&[u8], &[f32])> = (0..6).map(|j| (&records.0[j][..], &x[..])).collect();
    drop(step_experts_host(&jobs, &layout, limit, true));
    let t = std::time::Instant::now();
    for _ in 0..60 {
        drop(step_experts_host(&jobs, &layout, limit, true));
    }
    eprintln!("6 experts a call, the same each time: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 60.0);
    let t = std::time::Instant::now();
    for _ in 0..60 {
        drop(step_experts_host(&jobs[..1], &layout, limit, true));
    }
    eprintln!("1 expert a call, the same each time: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 60.0);
    // one projection's sums on this thread alone (a gate's: 5120 by 2304), its weights a nanosecond
    let data = random_exl3(hidden, ff, tw, 77);
    let record: Vec<u8> = data.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ProjAt { suh: 0, svh: 0, mul1: 0, trellis: 0, k: hidden, n: ff, tile_words: tw };
    let xh: Vec<f32> = (0..hidden).map(|i| half(((i * 37 % 101) as f32 - 50.0) / 60.0)).collect();
    drop(p.step_sums(&record, &xh, 0..ff / 16, true));
    let t = std::time::Instant::now();
    for _ in 0..20 {
        drop(p.step_sums(&record, &xh, 0..ff / 16, true));
    }
    let s = t.elapsed().as_secs_f64() / 20.0;
    eprintln!("a projection's sums on one thread: {:.2} ms, {:.2} weights a ns", s * 1e3, (hidden * ff) as f64 / s / 1e9);
}

/// What putting experts into mapped (scratch) slots costs (`--ignored --nocapture`), at DeepSeek-V4.1's shape (13.3 MB a
/// record): one, two and eight records a call, each then computed; the call and a put alone; a put, then a call of
/// another slot; a put, a wait, then its call. Measured on an RTX 5090 on four Gen5 lanes: a record 2.8-3.0 ms put and
/// computed, 1.4 the put alone (the host's writes at 11 GB/s), 0.15 the call alone, the call after the put no slower for
/// another slot or 3 ms later: the card takes the writes some 1.3 ms after the host has made them.
#[test]
#[ignore = "a measurement"]
fn measure_mapped_puts() {
    let Some(b) = backend() else { return };
    let (hidden, ff, tw) = (5120, 2304, 48);
    let (layout, records, _) = as_records(experts(8, hidden, ff, tw));
    let slots = b.exl3_slots_mapped(8, hidden, ff, [tw, tw]).expect("room for eight mapped slots");
    let all: Vec<&[u8]> = records.0.iter().map(|r| &r[..]).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    for n in [1, 2, 8] {
        for _ in 0..3 {
            slots.put_many(0, &all[..n], &layout);
        }
        let t = std::time::Instant::now();
        for _ in 0..20 {
            slots.put_many(0, &all[..n], &layout);
            // (a call on them after each, as a pass makes: the next put waits for it)
            drop(slots.forward(&[(0, &x[..])], 10.0));
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
        eprintln!("{n} record(s) a call: {ms:.2} ms ({:.2} GB/s of records)", (n * layout.bytes) as f64 / ms / 1e6);
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        drop(slots.forward(&[(0, &x[..])], 10.0));
    }
    eprintln!("the call alone: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    let t = std::time::Instant::now();
    for _ in 0..20 {
        slots.put_many(0, &all[..1], &layout);
    }
    eprintln!("a record's put alone: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    let t = std::time::Instant::now();
    for _ in 0..20 {
        slots.put_many(0, &all[..1], &layout);
        drop(slots.forward(&[(1, &x[..])], 10.0));
    }
    eprintln!("a put, then a call of another slot: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    let mut call = 0.0;
    for _ in 0..20 {
        slots.put_many(0, &all[..1], &layout);
        std::thread::sleep(std::time::Duration::from_millis(3));
        let s = std::time::Instant::now();
        drop(slots.forward(&[(0, &x[..])], 10.0));
        call += s.elapsed().as_secs_f64() * 1e3 / 20.0;
    }
    eprintln!("a put, 3 ms, then its call: the call {call:.2} ms");
}

/// Projections in one chained recording ([`exl3_chained`]: the transforms on the device), a step's row, a few rows and
/// a prompt's (past the few-rows kernel: the tensor cores' blocks where the device has them), are each the CPU's
/// projection of its rows; one over several buffers is not taken.
#[test]
fn several_projections_chained_are_each_the_cpus() {
    let Some(b) = backend() else { return };
    let shapes = [(256, 512, 48), (512, 256, 32), (256, 128, 24), (1024, 384, 64)];
    let lins: Vec<_> = shapes.iter().enumerate().map(|(i, &(k, n, tw))| b.exl3_with(random_exl3(k, n, tw, 90 + i as u32), None).unwrap()).collect();
    let cpus: Vec<_> = shapes.iter().enumerate().map(|(i, &(k, n, tw))| exl3_cpu(random_exl3(k, n, tw, 90 + i as u32)).unwrap()).collect();
    for m in [1, 3, 200] {
        let xs: Vec<Vec<f32>> = shapes.iter().enumerate().map(|(i, &(k, ..))| (0..m * k).map(|j| (((j + 3 * i) * 37 % 89) as f32 - 44.0) / 45.0).collect()).collect();
        let items: Vec<(&dyn PackedLinear, &[f32])> = lins.iter().zip(&xs).map(|(l, x)| (&**l, &x[..])).collect();
        let got = exl3_chained(&b, &items).expect("all on the device, each in one buffer");
        for (i, ((c, x), y)) in cpus.iter().zip(&xs).zip(&got).enumerate() {
            let want = c.linear(&Tensor::from_vec(x.clone(), vec![m, shapes[i].0]));
            close(y, want.data(), &format!("chained projection {i}, {m} rows"));
        }
    }
    let split = b.exl3_with(random_exl3(1024, 512, 64, 7), Some(3)).unwrap();
    let x = vec![0.25f32; 1024];
    assert!(exl3_chained(&b, &[(&*split, &x[..])]).is_none(), "a projection over several buffers is not chained");
    // an f16 matrix (a router's) in the same recording as an EXL3 projection: each the host's sums of its rows, at
    // DeepSeek-V4.1's shapes too (its router 384 x 5120, its indexer's weights 32 x 5120, its Engram's 4 x 5120)
    for (n, k) in [(96, 256), (384, 5120), (32, 5120), (4, 5120)] {
    let w: Vec<f32> = (0..n * k).map(|i| ((i * 29 % 61) as f32 - 30.0) / 64.0).collect();
    let wv = ggml_rs::DeviceChain::vec_f16(&b, &w).expect("f16 values exactly");
    for m in [1usize, 3, 40] {
        let xf: Vec<f32> = (0..m * k).map(|i| ((i * 13 % 37) as f32 - 18.0) / 32.0).collect();
        let xe: Vec<f32> = (0..m * shapes[0].0).map(|i| ((i * 7 % 23) as f32 - 11.0) / 12.0).collect();
        // (the same values as BF16, as a checkpoint's bytes lie: each is one)
        let wb = b.vec_of_bytes(&w.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<u8>>());
        let got = chained(&b, &[(ChainMat::F16 { w: &wv, n, k }, &xf[..]), (ChainMat::Exl3(&*lins[0]), &xe[..]), (ChainMat::Bf16 { w: &wb, n, k }, &xf[..])]).expect("every kind chained");
        let (w, xf_) = (&w, &xf);
        let want: Vec<f32> = (0..m).flat_map(|r| (0..n).map(move |o| (0..k).map(|i| w[o * k + i] * xf_[r * k + i]).sum::<f32>())).collect();
        close(&got[0], &want, &format!("the f16 matrix [{n}, {k}], {m} rows"));
        close(&got[1], cpus[0].linear(&Tensor::from_vec(xe.clone(), vec![m, shapes[0].0])).data(), &format!("the EXL3 projection beside it, {m} rows"));
        close(&got[2], &want, &format!("the BF16 matrix [{n}, {k}], {m} rows"));
    }
    }
}

/// A step's row, and a check's few, through grouped projections and one after them in one recording ([`chained_after`]:
/// DeepSeek-V4.1's `wo_a` groups then `wo_b`, EXL3 or BF16) is, to the bit, the two calls made apart with the rows
/// rounded to bf16 on the host between them; rows that do not make the second's input are not taken.
#[test]
fn rows_chained_after_grouped_projections_are_the_two_calls_apart() {
    let Some(b) = backend() else { return };
    let (k, n) = (256, 128);
    let groups: Vec<_> = (0..4).map(|g| b.exl3_with(random_exl3(k, n, 32, 40 + g), None).unwrap()).collect();
    let cols = |g: usize| g * 32..g * 32 + 32;
    let after_exl3 = b.exl3_with(random_exl3(128, 512, 48, 9), None).unwrap();
    let w: Vec<f32> = (0..512 * 128).map(|i| ((i * 29 % 61) as f32 - 30.0) / 64.0).collect();
    let after_bf16 = b.vec_of_bytes(&w.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<u8>>());
    let bf16 = |v: f32| f32::from_bits((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff_0000);
    for t in [1usize, 5] {
        let xs: Vec<Vec<f32>> = (0..4).map(|g| (0..t * k).map(|i| (((i + 5 * g) * 31 % 97) as f32 - 48.0) / 50.0).collect()).collect();
        let firsts: Vec<(ChainMat<'_>, &[f32])> = groups.iter().zip(&xs).map(|(l, x)| (ChainMat::Exl3(&**l), &x[..])).collect();
        let apart = chained(&b, &firsts).expect("the groups chained");
        let rows: Vec<f32> = (0..t).flat_map(|r| (0..4).flat_map(|g| apart[g][r * n..(r + 1) * n][cols(g)].iter().map(|&v| bf16(v)).collect::<Vec<_>>()).collect::<Vec<_>>()).collect();
        let with: Vec<(ChainMat<'_>, &[f32], std::ops::Range<usize>)> = groups.iter().zip(&xs).enumerate().map(|(g, (l, x))| (ChainMat::Exl3(&**l), &x[..], cols(g))).collect();
        for then in [ChainMat::Exl3(&*after_exl3), ChainMat::Bf16 { w: &after_bf16, n: 512, k: 128 }] {
            let want = chained(&b, &[(then, &rows[..])]).expect("the second chained").pop().unwrap();
            let got = chained_after(&b, &with, then).expect("the two chained");
            assert_eq!(got, want, "{t} rows chained after the groups");
        }
        assert!(chained_after(&b, &with[..3], ChainMat::Exl3(&*after_exl3)).is_none(), "rows short of the second's input");
    }
}

/// [`chained_normed`]'s square root and reciprocal are correctly rounded (`1 / sqrt(a)` of a million values over sixty
/// binades, each the host's to the bit), and the rows it norms between two stages are the host's norm of the first's
/// output to the bit: the first stage's sums those of [`chained`], the second's those of [`chained`] of the host's norm.
#[test]
fn a_norm_between_chained_stages_is_the_hosts() {
    use ggml_rs::ChainRecorder;
    let Some(b) = backend() else { return };
    let mut s = 7u32;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    };
    let a: Vec<f32> = (0..1 << 20).map(|_| f32::from_bits(((100 + next() % 60) << 23) | (next() >> 9))).collect();
    // 1 / sqrt(a), a / 1280 and a over its neighbour, each of every value
    let functions = NORM_AS_HOST.split("@compute").next().expect("the functions").to_string();
    for (name, expr, host) in [
        ("test-rsqrt-rn", "recip_rn(sqrt_rn(x[i]))", &(|a: &[f32], i: usize| 1.0 / a[i].sqrt()) as &dyn Fn(&[f32], usize) -> f32),
        ("test-div-rn", "div_rn(x[i], 1280.0)", &|a: &[f32], i: usize| a[i] / 1280.0),
        ("test-div-rn-any", "div_rn(x[i], x[(i + 1u) % p[0].x])", &|a: &[f32], i: usize| a[i] / a[(i + 1) % a.len()]),
    ] {
        let body = format!("{functions}@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let i = id.x;\n    if (i < p[0].x) {{ y[i] = {expr}; }}\n}}\n");
        let (av, yv) = (b.vec(a.len()), b.vec(a.len()));
        DeviceChain::upload(&b, &av, &a);
        let mut rec = crate::chain::Recorder::new(&b);
        let (d, drw) = (rec.gpu().dummy().clone(), rec.gpu().dummy_rw().clone());
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().unwrap().clone();
        rec.dispatch_wide(name, &body, [&buf(&av), &d, &d, &d, &d, &d, &buf(&yv), &drw], &[a.len() as u32], ((a.len() / 256) as u32, 1, 1));
        rec.read(&yv);
        let got = Box::new(rec).finish().pop().unwrap();
        let wrong = (0..a.len()).filter(|&i| host(&a, i).to_bits() != got[i].to_bits()).count();
        assert_eq!(wrong, 0, "{expr} of {} values, {wrong} not the host's", a.len());
    }
    // the norm between two stages, of a width a power of two and of DeepSeek-V4.1's (1280)
    let bf16 = |v: f32| f32::from_bits((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff_0000);
    for dn in [512usize, 1280] {
    let k = 256;
    let q_a = b.exl3_with(random_exl3(k, dn, 32, 21), None).unwrap();
    let w: Vec<f32> = (0..64 * k).map(|i| ((i * 29 % 61) as f32 - 30.0) / 64.0).collect();
    let beside = b.vec_of_bytes(&w.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<u8>>());
    let q_b = b.exl3_with(random_exl3(dn, 256, 48, 22), None).unwrap();
    let w2: Vec<f32> = (0..128 * dn).map(|i| ((i * 17 % 53) as f32 - 26.0) / 64.0).collect();
    let after = b.vec_of_bytes(&w2.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<u8>>());
    let norm: Vec<f32> = (0..dn).map(|i| bf16(0.5 + (i * 37 % 101) as f32 / 64.0)).collect();
    let eps = 1e-6f32;
    for rows in [1usize, 3] {
        let x: Vec<f32> = (0..rows * k).map(|i| ((i * 31 % 97) as f32 - 48.0) / 20.0).collect();
        let first = [(ChainMat::Exl3(&*q_a), &x[..]), (ChainMat::Bf16 { w: &beside, n: 64, k }, &x[..])];
        let then = [ChainMat::Exl3(&*q_b), ChainMat::Bf16 { w: &after, n: 128, k: dn }];
        let (got_first, got_then) = chained_normed(&b, &first, 0, &norm, eps, &then).expect("both stages chained");
        let want_first = chained(&b, &first).expect("the first stage alone");
        assert_eq!(got_first, want_first, "the first stage, {rows} rows");
        let normed: Vec<f32> = want_first[0]
            .chunks_exact(dn)
            .flat_map(|row| {
                let v: Vec<f32> = row.iter().map(|&a| bf16(a)).collect();
                let r = 1.0 / (v.iter().map(|v| v * v).sum::<f32>() / dn as f32 + eps).sqrt();
                v.iter().zip(&norm).map(|(v, g)| bf16(g * (v * r))).collect::<Vec<_>>()
            })
            .collect();
        let want_then = chained(&b, &[(then[0], &normed[..]), (then[1], &normed[..])]).expect("the second stage alone");
        assert_eq!(got_then, want_then, "the second stage of the host's norm, {rows} rows of {dn}");
    }
    }
}

/// What a decode step's call of a layer's projections of x and its queries costs (`--ignored --nocapture`), at
/// DeepSeek-V4.1-Flash's shapes (5120 wide, its queries' row 1280, 64 heads of 512, 3 bits): back to back, the call
/// alone and the GPU's share of it (the same work recorded 16 times in one call), and each projection in a call of
/// its own.
#[test]
#[ignore = "a measurement"]
fn measure_a_layers_query_call() {
    let Some(b) = backend() else { return };
    let h = 5120;
    let first: Vec<_> = [(h, 1280), (h, 512), (h, 1024), (h, 1024)].iter().enumerate().map(|(i, &(k, n))| b.exl3_with(random_exl3(k, n, 48, 300 + i as u32), None).unwrap()).collect();
    let then: Vec<_> = [(1280, 32768), (1280, 4096)].iter().enumerate().map(|(i, &(k, n))| b.exl3_with(random_exl3(k, n, 48, 310 + i as u32), None).unwrap()).collect();
    let w: Vec<f32> = (0..32 * h).map(|i| ((i * 29 % 61) as f32 - 30.0) / 64.0).collect();
    let router = b.vec_of_bytes(&w.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<u8>>());
    let x: Vec<f32> = (0..h).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    let norm = vec![1.0f32; 1280];
    let mut items: Vec<(ChainMat<'_>, &[f32])> = first.iter().map(|l| (ChainMat::Exl3(&**l), &x[..])).collect();
    items.push((ChainMat::Bf16 { w: &router, n: 32, k: h }, &x[..]));
    let thens: Vec<ChainMat<'_>> = then.iter().map(|l| ChainMat::Exl3(&**l)).collect();
    let time = |what: &str, f: &dyn Fn()| {
        for _ in 0..5 {
            f();
        }
        let t = std::time::Instant::now();
        for _ in 0..40 {
            f();
        }
        eprintln!("{what}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 40.0);
    };
    time("a layer's queries' call", &|| drop(chained_normed(&b, &items, 0, &norm, 1e-20, &thens)));
    // as a decode step spaces its calls: the card idle between them for the host's work (a busy wait) or a sleep, the
    // call itself timed
    for (how, pause_us, sleeps) in [("after 0.5 ms of the host's work", 500u64, false), ("after 2 ms of it", 2000, false), ("after a 2 ms sleep", 2000, true)] {
        let mut spent = 0.0;
        for _ in 0..40 {
            let pause = std::time::Instant::now();
            if sleeps {
                std::thread::sleep(std::time::Duration::from_micros(pause_us));
            } else {
                while pause.elapsed() < std::time::Duration::from_micros(pause_us) {
                    std::hint::spin_loop();
                }
            }
            let t = std::time::Instant::now();
            drop(chained_normed(&b, &items, 0, &norm, 1e-20, &thens));
            spent += t.elapsed().as_secs_f64();
        }
        eprintln!("  the call {how}: {:.3} ms", spent * 1e3 / 40.0);
    }
    let many: Vec<(ChainMat<'_>, &[f32])> = (0..16).flat_map(|_| items.iter().copied()).collect();
    time("its first stage 16 times in one call (/16)", &|| drop(chained(&b, &many)));
    for (i, item) in items.iter().enumerate() {
        time(&format!("projection {i} alone"), &|| drop(chained(&b, &[*item])));
    }
}

fn random_exl3(k: usize, n: usize, tw: usize, seed: u32) -> Exl3Data {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    };
    let words = (0..k / 16 * n / 16 * (tw / 2)).map(|_| next()).collect();
    let mut scale = |len: usize, mag: f32| (0..len).map(|_| if next() & 1 == 0 { mag } else { -mag }).collect::<Vec<f32>>();
    Exl3Data { suh: scale(k, 0.25), svh: scale(n, 0.125), words, tile_words: tw, input_map: (0..k as u32).collect(), output_map: (0..n as u32).collect() }
}

/// `count` experts plus the shared one, each `[gate, up, down]`.
fn experts(count: usize, hidden: usize, ff: usize, tw: usize) -> Vec<[Exl3Data; 3]> {
    (0..=count as u32).map(|e| [random_exl3(hidden, ff, tw, 11 + e * 3), random_exl3(hidden, ff, tw, 12 + e * 3), random_exl3(ff, hidden, tw, 13 + e * 3)]).collect()
}

/// The MoE as its definition reads, from standalone projections: each row's top-k routed experts by logit (ties to
/// the lower index), softmax-weighted, plus the shared expert by the sigmoid of its gate.
fn reference(experts: Vec<[Exl3Data; 3]>, x: &[f32], logits: &[f32], top_k: usize) -> Vec<f32> {
    let p: Vec<[Exl3Cpu; 3]> = experts.into_iter().map(|[g, u, d]| [Exl3Cpu::new(g).unwrap(), Exl3Cpu::new(u).unwrap(), Exl3Cpu::new(d).unwrap()]).collect();
    let (h, width) = (p[0][0].t.k, p.len());
    let mut out = vec![];
    for (r, row) in x.chunks_exact(h).enumerate() {
        let l = &logits[r * width..(r + 1) * width];
        let mut idx: Vec<usize> = (0..width - 1).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let top = &idx[..top_k];
        let z: f32 = top.iter().map(|&e| (l[e] - l[top[0]]).exp()).sum();
        let mut picks: Vec<(usize, f32)> = top.iter().map(|&e| (e, (l[e] - l[top[0]]).exp() / z)).collect();
        picks.push((width - 1, 1.0 / (1.0 + (-l[width - 1]).exp())));
        let mut acc = vec![0f32; h];
        let xr = Tensor::from_vec(row.to_vec(), vec![1, h]);
        for (e, w) in picks {
            let g = p[e][0].linear(&xr);
            let u = p[e][1].linear(&xr);
            let hid: Vec<f32> = g.data().iter().zip(u.data()).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let f = hid.len();
            let d = p[e][2].linear(&Tensor::from_vec(hid, vec![1, f]));
            for (a, v) in acc.iter_mut().zip(d.data()) {
                *a += w * v;
            }
        }
        out.extend(acc);
    }
    out
}

#[test]
fn a_moe_layer_routes_and_mixes_as_its_definition_on_the_gpu_on_the_cpu_and_split_between_them() {
    let (count, hidden, ff, top_k) = (8, 256, 128, 3);
    for tw in [48, 80] {
        for rows in [1, 7] {
            let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
            let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29) as f32 - 14.0) / 7.0).collect();
            let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
            let xt = Tensor::from_vec(x.clone(), vec![rows, hidden]);
            let lt = Tensor::from_vec(logits.clone(), vec![rows, count + 1]);
            let cpu = exl3_experts_cpu(experts(count, hidden, ff, tw)).unwrap();
            close(cpu.forward(&xt, &lt, top_k).data(), &want, &format!("cpu moe tw={tw} rows={rows}"));
            let Some(b) = backend() else { continue };
            // the routed experts as groups (one shape and bitrate), their transforms on the GPU
            let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
            assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
            close(gpu.forward(&xt, &lt, top_k).data(), &want, &format!("grouped moe tw={tw} rows={rows}"));
            if rows == 1 {
                // chained, a step's kept scratch, twice
                let assign = vec![route(&logits, top_k)];
                let (xd, out) = (b.vec(hidden), b.vec(hidden));
                DeviceChain::upload(&b, &xd, &x);
                for _ in 0..2 {
                    let mut rec = b.begin();
                    rec.moe_rows(gpu.as_ref(), &xd, &out, &assign);
                    rec.read(&out);
                    close(&rec.finish().pop().unwrap(), &want, &format!("chained moe tw={tw}"));
                }
            }
            drop(gpu);
            // projections a dispatch each, as a layer whose experts differ in shape or bitrate goes (made so here: the
            // environment's OAIY_EXL3_UNGROUPED is the process's, and set here it sent the other tests' layers that way)
            let each: Vec<[Proj; 3]> = experts(count, hidden, ff, tw).into_iter().map(|[g, u, d]| [b.proj(g, 0).unwrap(), b.proj(u, 0).unwrap(), b.proj(d, 0).unwrap()]).collect();
            let host: Box<dyn ggml_rs::exl3::Experts> = Box::new(Exl3MoeHost::new(each, Some((Arc::clone(&b.gpu), Arc::clone(&b.serial)))).unwrap());
            assert!(format!("{host:?}").contains(&format!("{} of {} projections on the GPU", 3 * (count + 1), 3 * (count + 1))), "{host:?}");
            close(host.forward(&xt, &lt, top_k).data(), &want, &format!("gpu moe tw={tw} rows={rows}"));
            drop(host);
            // A budget for some of the experts: the rest decode on the CPU, and the layer is the same.
            let one = (hidden * ff * tw / 128) as u64;
            let Ok(small) = WgpuBackend::new(Some(one * 10)) else { continue };
            let split = small.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
            assert!(format!("{split:?}").contains("10 of 27 projections on the GPU"), "{split:?}");
            close(split.forward(&xt, &lt, top_k).data(), &want, &format!("split moe tw={tw} rows={rows}"));
        }
    }
}

#[test]
fn routing_takes_the_top_k_ties_to_the_lower_index_and_the_shared_expert_by_its_gate() {
    let r = route(&[1.0, 3.0, 3.0, 2.0, 0.0], 2);
    assert_eq!(r.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 4]);
    assert!((r[0].1 - 0.5).abs() < 1e-6 && (r[1].1 - 0.5).abs() < 1e-6, "{r:?}");
    assert!((r[2].1 - 0.5).abs() < 1e-6, "sigmoid(0) for the shared expert: {r:?}");
    let r = route(&[0.0, 0.0, 0.0, 5.0], 1);
    assert_eq!(r[0], (0, 1.0), "a three-way tie goes to the lowest index");
}

#[test]
fn the_cpu_projection_spreads_any_width_over_any_threads() {
    // 40 tile columns over 32 threads: two each, so the last threads have none (it panicked on a prompt's last row).
    let (k, n, tw) = (256, 640, 48);
    let data = random_exl3(k, n, tw, 7);
    let x: Vec<f32> = (0..9 * k).map(|i| ((i * 13 % 37) as f32 - 18.0) / 20.0).collect();
    let cpu = Exl3Cpu::new(data).unwrap();
    let mut xh = vec![0f32; 9 * k];
    for (row, out) in x.chunks_exact(k).zip(xh.chunks_exact_mut(k)) {
        cpu.t.pre(row, out);
    }
    let one = cpu.matmul(&xh, 9, 1);
    for threads in [2, 3, 7, 32, 64] {
        let many = cpu.matmul(&xh, 9, threads);
        close(&many, &one, &format!("{threads} threads"));
    }
}

/// How long a decode step's projection takes, and how much of it is waiting for the GPU (`--ignored --nocapture`).
#[test]
#[ignore = "timing"]
fn time_a_decode_projection() {
    let Some(b) = backend() else { return };
    for (k, n) in [(2560, 10240), (6144, 2560), (2560, 640)] {
        let w = b.exl3(random_exl3(k, n, 48, 5)).unwrap();
        let x = Tensor::from_vec(vec![0.1; k], vec![1, k]);
        for _ in 0..5 {
            w.linear(&x);
        }
        let t = std::time::Instant::now();
        for _ in 0..50 {
            w.linear(&x);
        }
        let each = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
        // The same with nothing to compute: an empty submit and wait.
        let t = std::time::Instant::now();
        for _ in 0..50 {
            b.gpu.queue().submit([]);
            b.gpu.wait(None);
        }
        let idle = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
        // As in a model: the host computes between calls (here 3 ms of spinning), so the GPU waits idle between them.
        let spin = |ms: f64| {
            let t = std::time::Instant::now();
            while t.elapsed().as_secs_f64() * 1000.0 < ms {
                std::hint::spin_loop();
            }
        };
        let mut gapped = 0.0;
        for _ in 0..50 {
            spin(3.0);
            let t = std::time::Instant::now();
            w.linear(&x);
            gapped += t.elapsed().as_secs_f64() * 1000.0;
        }
        // And with every other core busy, as a model's host threads keep them.
        let stop = std::sync::atomic::AtomicBool::new(false);
        let busy = std::thread::scope(|s| {
            for _ in 1..std::thread::available_parallelism().map_or(4, |n| n.get()) {
                s.spawn(|| {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                });
            }
            let t = std::time::Instant::now();
            for _ in 0..50 {
                w.linear(&x);
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            t.elapsed().as_secs_f64() * 1000.0 / 50.0
        });
        eprintln!("{k}x{n}: {each:.3} ms a call back to back; {:.3} ms after 3 ms of host work; {busy:.3} ms with every core busy; an empty submit and wait {idle:.3} ms", gapped / 50.0);
    }
}

#[test]
fn a_weight_beyond_the_budget_runs_on_the_cpu_and_the_budget_is_returned() {
    let Ok(b) = WgpuBackend::new(Some(4096)) else { return };
    let (data, _) = oracle(48, 128, 256);
    let w = b.exl3(data).unwrap();
    assert!(format!("{w:?}").starts_with("Exl3Cpu"), "{w:?}");
    assert_eq!(b.usage().0, 0);
    let Some(b) = backend() else { return };
    let (data, _) = oracle(48, 128, 256);
    let w = b.exl3(data).unwrap();
    assert!(b.usage().0 > 0);
    drop(w);
    assert_eq!(b.usage().0, 0, "dropping the weight returns its bytes");
}

/// A projection chained on the GPU, its transforms there too (the maps gathered, the Hadamard transforms and their
/// f16 roundings), gives the projection's own answer (its transforms on the host) bit for bit: one row (a step's,
/// its scratch kept), a few rows (a check of drafts: each row as one row alone), and a prompt's rows as its passes
/// take them (32 at a time, a last lone row as one row), with and without maps, at 3 and 5 bits.
#[test]
fn a_chained_projection_matches_the_projection() {
    let Some(b) = backend() else { return };
    let (k, n) = (512usize, 384usize);
    // (every rate a checkpoint may hold: a step's kernel is written for each, and at 7 bits a weight and more a
    // lane's last codes lie in its third and fourth words)
    assert_eq!(RATES, [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128]);
    for (tw, maps) in RATES.into_iter().zip([false, false, true, false, false, false, true, true, false, true, false]) {
        let mut data = random_exl3(k, n, tw, 77 + tw as u32);
        if maps {
            data.input_map = (0..k as u32).map(|i| (i * 7 + 3) % k as u32).collect();
            data.output_map = (0..n as u32).rev().collect();
        }
        let w = b.exl3(data).unwrap();
        assert!(b.holds_exl3(w.as_ref()), "the adapter holds it");
        for rows in [1usize, 2, 3, 5, 8, 9, 32, 33, 70, 129] {
            let xs: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect();
            let want = if (2..=FEW_MAX).contains(&rows) {
                let each: Vec<f32> = xs.chunks_exact(k).flat_map(|r| w.linear(&Tensor::from_vec(r.to_vec(), vec![1, k])).data().to_vec()).collect();
                Tensor::from_vec(each, vec![rows, n])
            } else {
                w.linear(&Tensor::from_vec(xs.clone(), vec![rows, k]))
            };
            let (x, y) = (b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &x, &xs);
            // twice: the second from the kept scratch
            for _ in 0..2 {
                let mut rec = b.begin();
                rec.exl3_rows(w.as_ref(), &x, &y, rows);
                rec.read(&y);
                let got = rec.finish().pop().unwrap();
                if rows > FEW_MAX && coop_on(&b.gpu) {
                    // a prompt's rows on the tensor cores: a matmul's sums (the same products, f16 exactly)
                    let dot: f64 = got.iter().zip(want.data()).map(|(a, e)| *a as f64 * *e as f64).sum();
                    let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                    let cos = dot / (norm(&got) * norm(want.data()));
                    let top = want.data().iter().fold(0f32, |m, v| m.max(v.abs()));
                    let worst = got.iter().zip(want.data()).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
                    assert!(cos > 0.99999 && worst <= 2e-3 * top, "tw={tw} maps={maps} rows={rows}: cosine {cos}, worst {worst} of {top}");
                    continue;
                }
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&got), bits(want.data()), "tw={tw} maps={maps} rows={rows}");
            }
        }
    }
}
/// A chained projection's time for a few rows (`--ignored --nocapture`): Qwen3.8-Flash-Next's delta nets' qkv, z and
/// output, its attention's q, and an expert's gate and down, at 3 bits, matrices in turn past the L2 as a model's
/// layers are.
#[test]
#[ignore = "a measurement"]
fn measure_chained_projections() {
    let Some(b) = backend() else { return };
    for (k, n, count) in [(2560usize, 10240usize, 8usize), (2560, 6144, 12), (6144, 2560, 12), (2560, 12288, 8), (2560, 640, 96), (640, 2560, 96)] {
        let ws: Vec<_> = (0..count as u32).map(|i| b.exl3(random_exl3(k, n, 48, 900 + i)).unwrap()).collect();
        let bytes = (k / 16 * n / 16 * 24 * 4) as f64;
        let mut line = format!("[{n}, {k}]:");
        let mut one = 0.0;
        for rows in [1usize, 2, 3, 4, 8] {
            let (x, y) = (b.vec(rows * k), b.vec(rows * n));
            DeviceChain::upload(&b, &x, &(0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
            let reps = 4 * count;
            let run = || {
                let mut rec = b.begin();
                for i in 0..reps {
                    rec.exl3_rows(ws[i % count].as_ref(), &x, &y, rows);
                }
                rec.read_range(&y, 0, 1);
                rec.finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let each = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
            if rows == 1 {
                one = each;
            }
            line += &format!(" {rows} rows {:.1} us ({:.2}x, {:.0} GB/s);", each * 1e6, each / one, bytes / each / 1e9);
        }
        eprintln!("{line}");
    }
}

/// The few-rows kernel's cost by how many of a block's places hold jobs, against the one-job kernel's
/// (`--ignored --nocapture`): an expert's gate (640 x 2560) and a delta net's qkv (10240 x 2560) at 3 bits, matrices
/// in turn past the L2; and a dispatch of empty blocks.
#[test]
#[ignore = "a measurement"]
fn measure_few_blocks() {
    let Some(b) = backend() else { return };
    for (k, n, count) in [(2560usize, 640usize, 96usize), (2560, 10240, 8)] {
        let ws: Vec<_> = (0..count as u32).map(|i| b.exl3(random_exl3(k, n, 48, 700 + i)).unwrap()).collect();
        let g: Vec<&Exl3Gpu> = ws.iter().map(|w| w.as_any().unwrap().downcast_ref::<Exl3Gpu>().unwrap()).collect();
        let splits = g[0].single_chunk().unwrap().1;
        let (xh, part) = (b.vec(8 * k), b.vec(8 * splits as usize * n));
        DeviceChain::upload(&b, &xh, &(0..8 * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
        let jobs = u32_vec(&b, &(0..8u32).flat_map(|r| [0, r]).collect::<Vec<_>>());
        let ntiles = (n / 16) as u32;
        let reps = 4 * count;
        let time = |name: &'static str, body: &str, order: Option<&DeviceVec>, blocks: u32| {
            let run = || {
                use ggml_rs::ChainRecorder;
                let mut rec = crate::chain::Recorder::new(&b);
                let d = rec.gpu().dummy().clone();
                let drw = rec.gpu().dummy_rw().clone();
                let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().unwrap().clone();
                for i in 0..reps {
                    let (words, _) = g[i % count].single_chunk().unwrap();
                    let ob = order.map(buf).unwrap_or_else(|| d.clone());
                    rec.dispatch_wide(name, body, [words, &buf(&xh), &buf(&jobs), &ob, &d, &d, &buf(&part), &drw], &[n as u32, k as u32, 48, splits, 0, 0], (ntiles, 1, blocks * splits));
                }
                rec.read_range(&part, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            t.elapsed().as_secs_f64() / 5.0 / reps as f64 * 1e6
        };
        let one = time("exl3-mm", &chain_shader("mm"), None, 1);
        let mut line = format!("[{n}, {k}]: a job {one:.1} us;");
        for (rows, used) in [(4usize, 1usize), (4, 2), (4, 4), (2, 1), (2, 2), (8, 1)] {
            let order: Vec<u32> = (0..rows as u32).map(|r| if (r as usize) < used { r } else { NONE }).collect();
            let ov = u32_vec(&b, &order);
            line += &format!(" few-{rows} with {used} {:.1} us;", time(few_kernel(rows, 48).0, few_kernel(rows, 48).1, Some(&ov), 1));
        }
        let empty = u32_vec(&b, &vec![NONE; 4 * 40]);
        line += &format!(" 40 empty blocks of 4 {:.1} us", time(few_kernel(4, 48).0, few_kernel(4, 48).1, Some(&empty), 40));
        eprintln!("{line}");
    }
}

/// A check's experts as the GPU takes them in one pass (`--ignored --nocapture`): 4 rows of 10 of 128 experts
/// (Qwen3.8-Flash-Next's 2560 x 640 at 3 bits), their gate and up jobs a job a workgroup set, or grouped by matrix
/// in blocks of 4 (no empty blocks), as the rows share none, some or all of them.
#[test]
#[ignore = "a measurement"]
fn measure_grouped_check_experts() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, rows, k) = (128usize, 2560usize, 640usize, 4usize, 10usize);
    let moe = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
    let g = moe.as_any().unwrap().downcast_ref::<Exl3MoeGrouped>().unwrap();
    let st = g.scratch(&mut |n| b.vec(n), rows, k, true);
    let x = b.vec(rows * hidden);
    DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
    for (what, shared) in [("none shared", 0usize), ("3 of 10 shared", 3), ("all shared", 10)] {
        // row r's experts: the shared ones, then its own
        let picks: Vec<Vec<usize>> = (0..rows).map(|r| (0..k).map(|j| if j < shared { j } else { 10 + r * 25 + j }).collect()).collect();
        let jobs: Vec<u32> = picks.iter().enumerate().flat_map(|(r, p)| p.iter().flat_map(move |&e| [2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32])).collect();
        upload_u32(&b, &st.jobs_gu, &jobs);
        let n = jobs.len() / 2;
        // blocks of a matrix's jobs, as GROUP makes them
        let mut seen: Vec<(u32, Vec<u32>)> = Vec::new();
        for q in 0..n {
            let m = jobs[2 * q];
            match seen.iter_mut().find(|(mm, _)| *mm == m) {
                Some((_, list)) => list.push(q as u32),
                None => seen.push((m, vec![q as u32])),
            }
        }
        let order: Vec<u32> = seen.iter().flat_map(|(_, list)| (0..rows).map(|i| list.get(i).copied().unwrap_or(NONE))).collect();
        upload_u32(&b, &st.order_gu, &order);
        let blocks = seen.len();
        let time = |few: bool| {
            let run = || {
                use ggml_rs::ChainRecorder;
                let mut rec = crate::chain::Recorder::new(&b);
                for _ in 0..48 {
                    let o = if few { Order::Few(&st.order_gu, blocks, rows) } else { Order::Jobs };
                    g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, o, &st.xh_gu, &st.part_gu, &st.out_gu);
                }
                rec.read_range(&st.out_gu, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            t.elapsed().as_secs_f64() / 5.0 / 48.0 * 1e6
        };
        eprintln!("{what}: {n} jobs, {blocks} matrices: a job each {:.1} us a pass, grouped {:.1} us", time(false), time(true));
    }
}

/// What a prompt's grouped experts' gate and up matmul takes on the tensor cores (`--ignored --nocapture`, with
/// OAIY_CHAIN_PROFILE for each kernel's GPU time): 128 of Qwen3.8-Flash-Next's experts (2,560 by 640, 3 bits), 10,
/// 20 or 40 jobs each (a chunk of 512, 1,024 or 2,048 rows over its 512), in blocks of 16 to 128; then the same
/// blocks over 8 matrices (their words in the L2 throughout). On an RTX 5090 a block's time is its decode's, its
/// rows nearly free: 10 jobs in blocks of 32 0.47 ms, 20 0.51; but blocks of 64 or 128 0.83 and 0.94 for the same
/// blocks (their registers leave a workgroup an SM where 32's leave two: 32's padded to one an SM is 0.80), and the
/// words in the L2 0.42 where 0.50. Neither skipping a block's empty fragments' multiply-adds nor the decode's
/// conversion done in bits changed it.
#[test]
#[ignore = "a measurement"]
fn measure_prompt_expert_blocks() {
    let Some(b) = backend() else { return };
    if !coop_on(&b.gpu) {
        return;
    }
    let (count, hidden, ff, tw, top_k) = (128usize, 2560usize, 640usize, 48usize, 10usize);
    let moe = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    let g = moe.as_any().unwrap().downcast_ref::<Exl3MoeGrouped>().unwrap();
    for per in [10usize, 20, 40] {
        let rows = per * count / top_k;
        let st = g.scratch(&mut |n| b.vec(n), rows, top_k, false);
        let x = b.vec(rows * hidden);
        DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
        // row r's experts r k to r k + k - 1 (of the count, around): `per` rows each
        let jobs: Vec<u32> = (0..rows)
            .flat_map(|r| {
                (0..top_k).flat_map(move |j| {
                    let e = ((r * top_k + j) % count) as u32;
                    [2 * e, r as u32, 2 * e + 1, r as u32]
                })
            })
            .collect();
        upload_u32(&b, &st.jobs_gu, &jobs);
        let n = jobs.len() / 2;
        for bs in [16usize, 32, 64, 128] {
            let order = many_order(&jobs, bs);
            let ob = b.vec(order.len());
            upload_u32(&b, &ob, &order);
            let blocks = order.len() / bs;
            let run = || {
                let mut rec = crate::chain::Recorder::new(&b);
                for _ in 0..8 {
                    g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, Order::Many(&ob, blocks, bs), &st.xh_gu, &st.part_gu, &st.out_gu);
                }
                use ggml_rs::ChainRecorder;
                rec.read_range(&st.out_gu, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let _ = crate::profile::take_kernels();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                run();
            }
            let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
            let k = crate::profile::take_kernels();
            let mm: f64 = k.iter().filter(|e| e.0.starts_with("exl3-coop")).map(|e| e.1).sum::<f64>() / 24.0;
            eprintln!("{per} jobs an expert ({rows} rows), blocks of {bs} ({blocks}): a pass {ms:.3} ms, its matmul {mm:.3} ms on the GPU");
        }
    }
    // the same blocks (256 of 10 jobs in places for 32) over 4 experts' matrices (their words in the L2 throughout)
    let (rows, bs, blocks, per) = (256usize, 32usize, 256usize, 10usize);
    let st = g.scratch(&mut |n| b.vec(n), rows, top_k, false);
    let x = b.vec(rows * hidden);
    DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
    for (what, spread) in [("256 matrices", 256u32), ("8 matrices", 8)] {
        let jobs: Vec<u32> = (0..blocks * per).flat_map(|j| [((j / per) as u32) % spread, (j % rows) as u32]).collect();
        upload_u32(&b, &st.jobs_gu, &jobs);
        let order: Vec<u32> = (0..blocks).flat_map(|blk| (0..bs).map(move |i| if i < per { (blk * per + i) as u32 } else { NONE })).collect();
        let ob = b.vec(order.len());
        upload_u32(&b, &ob, &order);
        let n = blocks * per;
        let run = || {
            let mut rec = crate::chain::Recorder::new(&b);
            for _ in 0..8 {
                g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, Order::Many(&ob, blocks, bs), &st.xh_gu, &st.part_gu, &st.out_gu);
            }
            use ggml_rs::ChainRecorder;
            rec.read_range(&st.out_gu, 0, 1);
            Box::new(rec).finish();
        };
        run();
        let _ = crate::profile::take_kernels();
        for _ in 0..3 {
            run();
        }
        let k = crate::profile::take_kernels();
        let mm: f64 = k.iter().filter(|e| e.0.starts_with("exl3-coop")).map(|e| e.1).sum::<f64>() / 24.0;
        eprintln!("{blocks} blocks of {per} jobs over {what}: the matmul {mm:.3} ms on the GPU");
    }
}

/// A prompt's rows through grouped experts, each expert's in blocks of 32 (a tile decoded once a block): an expert
/// with more rows than a block (two blocks), one with a single row, the rest a few each, as the definition gives.
#[test]
fn grouped_experts_take_a_prompts_rows_in_blocks_as_the_definition() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, rows) = (6, 256, 128, 2, 40);
    // expert 0 in the top two of rows 0..35, expert 5 only in row 39's, the others by a spread
    let logits: Vec<f32> = (0..rows)
        .flat_map(|r| {
            (0..=count).map(move |e| match e {
                0 if r < 35 => 5.0,
                5 if r == 39 => 5.0,
                5 => -5.0,
                e if e == count => 0.3,
                e => ((r * 7 + e * 3) % 11) as f32 / 11.0,
            })
        })
        .collect();
    let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
    let on = |e: usize| assign.iter().filter(|a| a[..top_k].iter().any(|p| p.0 == e)).count();
    assert!(on(0) > 32 && on(5) == 1, "expert 0 on {} rows, expert 5 on {}", on(0), on(5));
    let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    for tw in [48, 80] {
        let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
        let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
        assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
        let got = gpu.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k);
        close(got.data(), &want, &format!("grouped moe of a prompt tw={tw}"));
    }
}

#[test]
fn a_prompts_jobs_go_by_matrix_in_blocks_of_32() {
    // matrix 1 on 33 jobs, matrix 0 on 2, matrix 7 on 1: in their list's order within a matrix
    let mut jobs = vec![];
    for j in 0..36u32 {
        let m = match j {
            3 | 20 => 0,
            35 => 7,
            _ => 1,
        };
        jobs.extend([m, j]);
    }
    let order = many_order(&jobs, 32);
    assert_eq!(order.len(), 4 * 32, "a block for matrix 0, two for matrix 1, one for matrix 7");
    assert_eq!(&order[..3], &[3, 20, NONE]);
    assert!(order[2..32].iter().all(|&j| j == NONE));
    let ones: Vec<u32> = (0..35).filter(|&j| j != 3 && j != 20).collect();
    assert_eq!(&order[32..64], &ones[..32]);
    assert_eq!(&order[64..65], &ones[32..]);
    assert!(order[65..96].iter().all(|&j| j == NONE));
    assert_eq!(order[96], 35);
    assert!(order[97..].iter().all(|&j| j == NONE));
}

/// A prompt's experts routed and grouped by expert on the GPU (where the tensor cores take a prompt's blocks) give
/// what routing and grouping them on the host gives, and the MoE's definition: rows more than a check's, experts
/// on more rows than a block's 16 and on none, the sums into the streams as well.
#[test]
fn a_prompts_experts_routed_and_grouped_on_the_gpu_are_the_hosts() {
    let Some(b) = backend() else { return };
    if !coop_on(&b.gpu) {
        return;
    }
    let (count, hidden, ff, top_k) = (40, 256, 128, 6);
    let gpu = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
    for rows in [9usize, 40, 130] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        // expert 3 on every row, expert 39 on none, the rest spread
        let ls: Vec<f32> = (0..rows * (count + 1))
            .map(|i| match i % (count + 1) {
                3 => 4.0,
                39 => -9.0,
                e => (((i / (count + 1)) * 7 + e * 13) % 31) as f32 / 10.0 - 1.5,
            })
            .collect();
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
        let (xr, lr, on_gpu, on_host) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        let mut rec = b.begin();
        rec.keep_groups(false);
        assert!(rec.moe_routed(gpu.as_ref(), &xr, &on_gpu, &lr, top_k, rows), "{rows} rows route on the GPU");
        rec.moe_rows(gpu.as_ref(), &xr, &on_host, &assign);
        rec.read(&on_gpu);
        rec.read(&on_host);
        let mut got = rec.finish();
        let (h, g) = (got.pop().unwrap(), got.pop().unwrap());
        let scale = h.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        for (i, (a, e)) in g.iter().zip(&h).enumerate() {
            assert!((a - e).abs() <= 1e-5 * scale, "{rows} rows [{i}]: {a} against {e}");
        }
        close(&g, &reference(experts(count, hidden, ff, 48), &xs, &ls, top_k), &format!("{rows} rows routed on the GPU against the definition"));
    }
}

/// A step's experts routed on the GPU from the router's logits give what routing on the host gives: the same
/// experts (ties to the lower index; logits with few distinct values, so ties are common), their weights within an
/// ulp of exp; and the MoE's definition.
#[test]
fn experts_routed_on_the_gpu_are_the_hosts() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k) = (40, 256, 128, 6);
    let gpu = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    let xd = b.vec(hidden);
    DeviceChain::upload(&b, &xd, &x);
    let (on_host, on_gpu, ld) = (b.vec(hidden), b.vec(hidden), b.vec(count + 1));
    let mut seed = 7u32;
    for case in 0..12 {
        let levels = [4u32, 1 << 16][case % 2];
        let logits: Vec<f32> = (0..=count)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) % levels) as f32 / levels as f32 * 6.0 - 3.0
            })
            .collect();
        DeviceChain::upload(&b, &ld, &logits);
        for keep in [true, false] {
            let mut rec = b.begin();
            rec.keep_groups(keep);
            rec.moe_rows(gpu.as_ref(), &xd, &on_host, &[route(&logits, top_k)]);
            assert!(rec.moe_routed(gpu.as_ref(), &xd, &on_gpu, &ld, top_k, 1), "the grouped experts route on the GPU");
            rec.read(&on_host);
            rec.read(&on_gpu);
            let mut got = rec.finish();
            let (g, h) = (got.pop().unwrap(), got.pop().unwrap());
            let scale = h.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for (i, (a, e)) in g.iter().zip(&h).enumerate() {
                assert!((a - e).abs() <= 1e-5 * scale, "case {case} keep {keep} [{i}]: {a} against {e}");
            }
            if case == 0 {
                close(&g, &reference(experts(count, hidden, ff, 48), &x, &logits, top_k), "routed on the GPU against the definition");
            }
        }
    }
    // a check's few rows, each routed on its own: each row's sums its own step's
    for rows in [2usize, 3, 5] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        let ls: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 89) as f32 - 44.0) / 15.0).collect();
        let (xr, lr, yr) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        for keep in [true, false] {
            let mut rec = b.begin();
            rec.keep_groups(keep);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &yr, &lr, top_k, rows), "{rows} rows route on the GPU");
            rec.read(&yr);
            let got = rec.finish().pop().unwrap();
            for r in 0..rows {
                let (x1, l1, y1) = (b.vec(hidden), b.vec(count + 1), b.vec(hidden));
                DeviceChain::upload(&b, &x1, &xs[r * hidden..(r + 1) * hidden]);
                DeviceChain::upload(&b, &l1, &ls[r * (count + 1)..(r + 1) * (count + 1)]);
                let mut rec = b.begin();
                rec.keep_groups(keep);
                assert!(rec.moe_routed(gpu.as_ref(), &x1, &y1, &l1, top_k, 1));
                rec.read(&y1);
                let want = rec.finish().pop().unwrap();
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&got[r * hidden..(r + 1) * hidden]), bits(&want), "{rows} rows keep {keep}: row {r}");
            }
            // the sums added to each row's streams as a site's write-back adds them: the same bits as the sums,
            // then the write-back
            let streams = 3;
            let base: Vec<f32> = (0..rows * streams * hidden).map(|i| ((i * 13 % 47) as f32 - 23.0) / 9.0).collect();
            let post: Vec<f32> = (0..rows * streams).map(|i| (i as f32 - 2.0) / 3.0).collect();
            let (xa, xb, pd, sums) = (b.vec(base.len()), b.vec(base.len()), b.vec(post.len()), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xa, &base);
            DeviceChain::upload(&b, &xb, &base);
            DeviceChain::upload(&b, &pd, &post);
            let mut rec = b.begin();
            rec.keep_groups(keep);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &sums, &lr, top_k, rows));
            rec.stream_apply(&xa, &sums, &pd, rows, streams, hidden);
            assert!(rec.moe_routed_into(gpu.as_ref(), &xr, &xb, &pd, &lr, top_k, rows, streams));
            rec.read(&xa);
            rec.read(&xb);
            let mut got = rec.finish();
            let (into, apart) = (got.pop().unwrap(), got.pop().unwrap());
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
            assert_eq!(bits(&into), bits(&apart), "{rows} rows keep {keep}: into the streams");
        }
    }
}

/// A LoRA adapter's pairs for a layer of `count` routed experts and the shared one (the last), as `Experts::low_rank`
/// takes them: the experts `with` have one of `rank` beside a projection `k -> n` (a slot each, in that order), values
/// a seed makes, large beside the made experts' own outputs (so a wrong update shows).
fn lora_pairs(count: usize, with: &[usize], rank: usize, k: usize, n: usize, seed: u32) -> Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize)> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) % 2001) as f32 / 1000.0 - 1.0
    };
    let mut slot_of = vec![u32::MAX; count + 1];
    for (slot, &e) in with.iter().enumerate() {
        slot_of[e] = slot as u32;
    }
    let a = (0..with.len() * rank * k).map(|_| next() * 0.3).collect();
    let b = (0..with.len() * n * rank).map(|_| next() * 2.0).collect();
    Some((slot_of, a, b, rank))
}

type Pairs = [Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize)>; 3];

/// [`reference`] with low-rank updates beside the experts' gate, up and down projections (`lora[which]`): each
/// projection's output plus `b (a x)` of its input, the sums in f64.
fn reference_adapted(experts: Vec<[Exl3Data; 3]>, lora: &Pairs, x: &[f32], logits: &[f32], top_k: usize) -> Vec<f32> {
    let p: Vec<[Exl3Cpu; 3]> = experts.into_iter().map(|[g, u, d]| [Exl3Cpu::new(g).unwrap(), Exl3Cpu::new(u).unwrap(), Exl3Cpu::new(d).unwrap()]).collect();
    let (h, width) = (p[0][0].t.k, p.len());
    let update = |which: usize, e: usize, x: &[f32], y: &mut [f32]| {
        let Some((slot_of, a, b, rank)) = &lora[which] else { return };
        if slot_of[e] == u32::MAX {
            return;
        }
        let (s, k, n) = (slot_of[e] as usize, x.len(), y.len());
        let low: Vec<f64> = (0..*rank).map(|r| (0..k).map(|c| a[(s * rank + r) * k + c] as f64 * x[c] as f64).sum()).collect();
        for (i, yv) in y.iter_mut().enumerate() {
            *yv += (0..*rank).map(|r| b[(s * n + i) * rank + r] as f64 * low[r]).sum::<f64>() as f32;
        }
    };
    let mut out = vec![];
    for (r, row) in x.chunks_exact(h).enumerate() {
        let xr = Tensor::from_vec(row.to_vec(), vec![1, h]);
        let mut acc = vec![0f32; h];
        for (e, w) in route(&logits[r * width..(r + 1) * width], top_k) {
            let mut g = p[e][0].linear(&xr).data().to_vec();
            update(0, e, row, &mut g);
            let mut u = p[e][1].linear(&xr).data().to_vec();
            update(1, e, row, &mut u);
            let hid: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let f = hid.len();
            let mut d = p[e][2].linear(&Tensor::from_vec(hid.clone(), vec![1, f])).data().to_vec();
            update(2, e, &hid, &mut d);
            for (a, v) in acc.iter_mut().zip(&d) {
                *a += w * v;
            }
        }
        out.extend(acc);
    }
    out
}

/// A LoRA adapter's low-rank updates beside a layer's experts (`Experts::low_rank`) are the definition's sums: the
/// routed experts' in their groups' kernels and the shared expert's beside its projections on the GPU, and on the
/// host's path with its projections on the CPU or a dispatch each. A step's (its kept scratch, twice), a check's few
/// rows routed on the GPU (each row its own step's, to the bit), a prompt's rows by blocks; a gate and an up update of
/// different ranks in one group; experts with no update beside ones with; and an update whose B is zeros leaves the
/// layer as it was. Made experts and made pairs: 256 wide, 8 or 40 experts.
#[test]
fn a_low_rank_update_beside_the_experts_is_its_definition_on_the_gpu_and_the_host() {
    let (count, hidden, ff, top_k, tw) = (8, 256, 128, 3, 48);
    let lora: Pairs = [
        lora_pairs(count, &[2, count], 2, hidden, ff, 5),
        lora_pairs(count, &[2, 5], 4, hidden, ff, 6),
        lora_pairs(count, &[1, 3, 4, 6, count], 2, ff, hidden, 7),
    ];
    let none: Pairs = [None, None, None];
    // (by the sums' size: the made updates take them to tens, in terms that partly cancel, and the shared expert's
    // update goes through f16, its pairs as they are held and a prompt's rows as the tensor cores read them: a part in
    // 2,000 of a term at most. An update left out, or another expert's, is wrong by the sums' own size.)
    let near = |actual: &[f32], expected: &[f32], what: &str| {
        assert_eq!(actual.len(), expected.len(), "{what}");
        let size = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 0.003 + 1e-3 * size, "{what} [{i}]: {a} != {e} (sums up to {size})");
        }
    };
    let adapt = |e: &mut Box<dyn ggml_rs::exl3::Experts>, lora: &Pairs| {
        for (which, l) in lora.iter().enumerate() {
            let (s, a, b, r) = l.as_ref().unwrap();
            e.low_rank(which, s, a, b, *r).unwrap();
        }
    };
    for rows in [1usize, 7] {
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29) as f32 - 14.0) / 7.0).collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &x, &logits, top_k);
        let plain = reference_adapted(experts(count, hidden, ff, tw), &none, &x, &logits, top_k);
        close(&plain, &reference(experts(count, hidden, ff, tw), &x, &logits, top_k), "the definition with no update");
        let moved = want.iter().zip(&plain).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let size = plain.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(moved > 10.0 * size.max(0.01), "the updates are most of the layer's sums: they move them by {moved} at most, where the sums without are up to {size}");
        let xt = Tensor::from_vec(x.clone(), vec![rows, hidden]);
        let lt = Tensor::from_vec(logits.clone(), vec![rows, count + 1]);
        let mut cpu = exl3_experts_cpu(experts(count, hidden, ff, tw)).unwrap();
        adapt(&mut cpu, &lora);
        near(cpu.forward(&xt, &lt, top_k).data(), &want, &format!("adapted cpu moe rows={rows}"));
        let Some(b) = backend() else { continue };
        let mut gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
        assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
        adapt(&mut gpu, &lora);
        near(gpu.forward(&xt, &lt, top_k).data(), &want, &format!("adapted grouped moe rows={rows}"));
        if rows == 1 {
            let assign = vec![route(&logits, top_k)];
            let (xd, out) = (b.vec(hidden), b.vec(hidden));
            DeviceChain::upload(&b, &xd, &x);
            for _ in 0..2 {
                let mut rec = b.begin();
                rec.moe_rows(gpu.as_ref(), &xd, &out, &assign);
                rec.read(&out);
                near(&rec.finish().pop().unwrap(), &want, "adapted chained moe");
            }
        }
        drop(gpu);
        // each projection a dispatch of its own on the GPU, the sums the host's
        let each: Vec<[Proj; 3]> = experts(count, hidden, ff, tw).into_iter().map(|[g, u, d]| [b.proj(g, 0).unwrap(), b.proj(u, 0).unwrap(), b.proj(d, 0).unwrap()]).collect();
        let mut host: Box<dyn ggml_rs::exl3::Experts> = Box::new(Exl3MoeHost::new(each, Some((Arc::clone(&b.gpu), Arc::clone(&b.serial)))).unwrap());
        adapt(&mut host, &lora);
        near(host.forward(&xt, &lt, top_k).data(), &want, &format!("adapted gpu moe rows={rows}"));
    }
    let Some(b) = backend() else { return };
    // routed on the GPU: a step, and a check's rows, each row its own step's bits; kept scratch and not
    let (count, top_k) = (40, 6);
    let lora: Pairs = [
        lora_pairs(count, &[3, 7, 21, count], 2, hidden, ff, 15),
        lora_pairs(count, &[3, 8], 4, hidden, ff, 16),
        lora_pairs(count, &[0, 3, 5, 9, 17, 22, 31, 39, count], 2, ff, hidden, 17),
    ];
    let mut gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    adapt(&mut gpu, &lora);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    let width = count + 1;
    for rows in [1usize, 2, 5] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        // (expert 3, which has every update, on every row)
        let ls: Vec<f32> = (0..rows * width).map(|i| if i % width == 3 { 4.0 } else { ((i * 53 % 89) as f32 - 44.0) / 15.0 }).collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &xs, &ls, top_k);
        let (xr, lr, yr) = (b.vec(rows * hidden), b.vec(rows * width), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        for keep in [true, false] {
            let began = std::time::Instant::now();
            let mut rec = b.begin();
            rec.keep_groups(keep);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &yr, &lr, top_k, rows), "{rows} rows route on the GPU");
            rec.read(&yr);
            let got = rec.finish().pop().unwrap();
            eprintln!("    adapted experts routed on the GPU: {rows} rows, keep {keep}: {:.2} ms", began.elapsed().as_secs_f64() * 1e3);
            near(&got, &want, &format!("adapted {rows} rows routed on the GPU, keep {keep}"));
            for r in 0..rows {
                let (x1, l1, y1) = (b.vec(hidden), b.vec(width), b.vec(hidden));
                DeviceChain::upload(&b, &x1, &xs[r * hidden..(r + 1) * hidden]);
                DeviceChain::upload(&b, &l1, &ls[r * width..(r + 1) * width]);
                let mut rec = b.begin();
                rec.keep_groups(keep);
                assert!(rec.moe_routed(gpu.as_ref(), &x1, &y1, &l1, top_k, 1));
                rec.read(&y1);
                assert_eq!(bits(&got[r * hidden..(r + 1) * hidden]), bits(&rec.finish().pop().unwrap()), "adapted {rows} rows keep {keep}: row {r} is its step");
            }
        }
    }
    // a prompt's rows: routed and grouped by expert on the GPU where the tensor cores take them, and the host's routing
    for rows in [9usize, 40, 130] {
        let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
        let ls: Vec<f32> = (0..rows * width)
            .map(|i| match i % width {
                3 => 4.0,
                39 => -9.0,
                e => (((i / width) * 7 + e * 13) % 31) as f32 / 10.0 - 1.5,
            })
            .collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &xs, &ls, top_k);
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * width..(r + 1) * width], top_k)).collect();
        let (xr, lr, on_gpu, on_host) = (b.vec(rows * hidden), b.vec(rows * width), b.vec(rows * hidden), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xr, &xs);
        DeviceChain::upload(&b, &lr, &ls);
        let began = std::time::Instant::now();
        let mut rec = b.begin();
        rec.keep_groups(false);
        let routed = coop_on(&b.gpu) && rec.moe_routed(gpu.as_ref(), &xr, &on_gpu, &lr, top_k, rows);
        rec.moe_rows(gpu.as_ref(), &xr, &on_host, &assign);
        if routed {
            rec.read(&on_gpu);
        }
        rec.read(&on_host);
        let mut got = rec.finish();
        eprintln!("    adapted experts of a prompt's {rows} rows (routed on the GPU: {routed}): {:.2} ms", began.elapsed().as_secs_f64() * 1e3);
        near(&got.pop().unwrap(), &want, &format!("adapted {rows} rows routed on the host"));
        if routed {
            near(&got.pop().unwrap(), &want, &format!("adapted {rows} rows routed on the GPU"));
        }
    }
    // an update whose B is zeros adds nothing
    let (slot_of, a, bm, rank) = lora[2].clone().unwrap();
    let mut zeros = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    zeros.low_rank(2, &slot_of, &a, &vec![0.0; bm.len()], rank).unwrap();
    let plain = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
    let xs: Vec<f32> = (0..3 * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
    let ls: Vec<f32> = (0..3 * width).map(|i| ((i * 53 % 89) as f32 - 44.0) / 15.0).collect();
    let (xt, lt) = (Tensor::from_vec(xs, vec![3, hidden]), Tensor::from_vec(ls, vec![3, width]));
    assert_eq!(zeros.forward(&xt, &lt, top_k).data(), plain.forward(&xt, &lt, top_k).data(), "an update of zeros");
    // and an update the experts cannot take is refused, not dropped
    assert!(zeros.low_rank(2, &slot_of[..count], &a, &bm, rank).is_err(), "pairs for another count of experts");
    assert!(zeros.low_rank(2, &slot_of, &a, &bm, rank).is_err(), "a second update of the same matrices");
}

/// How long the CPU takes for one decode row of a Flash-Next expert's projections (`--ignored --nocapture`).
#[test]
#[ignore = "timing"]
fn time_cpu_expert_rows() {
    let gate = Exl3Cpu::new(random_exl3(2560, 640, 48, 5)).unwrap();
    let down = Exl3Cpu::new(random_exl3(640, 2560, 48, 6)).unwrap();
    // a projection's words as a record holds them, a tile's columns a task
    for (name, p) in [("gate", &gate), ("down", &down)] {
        let bytes: Vec<u8> = p.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let xh = inputs(1, p.t.k);
        let t = std::time::Instant::now();
        for _ in 0..20 {
            tile_columns(&LeWords(&bytes), p.t.k, p.t.n, 48, &xh, 1, 0..p.t.n / 16);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
        eprintln!("{name} from bytes, 1 row, 1 thread: {ms:.3} ms ({:.2} G weights/s)", (p.t.k * p.t.n) as f64 / ms / 1e6);
    }
    for threads in [1, 2, 5, 15] {
        for (name, p) in [("gate", &gate), ("down", &down)] {
            let xh = inputs(1, p.t.k);
            for _ in 0..3 {
                p.matmul(&xh, 1, threads);
            }
            let t = std::time::Instant::now();
            for _ in 0..20 {
                p.matmul(&xh, 1, threads);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
            eprintln!("{name} 1 row, {threads} threads: {ms:.3} ms ({:.2} G weights/s)", (p.t.k * p.t.n) as f64 / ms / 1e6);
        }
    }
}

/// Records held in memory, as [`FileRecords`] reads them from files.
#[derive(Debug)]
struct MemRecords(Vec<Vec<u8>>);

impl ExpertSource for MemRecords {
    fn experts(&self) -> usize {
        self.0.len()
    }

    fn read(&self, e: usize, into: &mut Vec<u8>) -> std::io::Result<()> {
        into.clear();
        into.extend_from_slice(&self.0[e]);
        Ok(())
    }
}

/// `experts`' routed ones as records (each projection's scales as f16, then its words as a file stores them), and their
/// layout; the shared expert (the last) apart.
fn as_records(mut experts: Vec<[Exl3Data; 3]>) -> (RecordLayout, MemRecords, [Exl3Data; 3]) {
    let shared = experts.pop().unwrap();
    let mut layout = None;
    let records = experts
        .iter()
        .map(|e| {
            let mut bytes = Vec::new();
            let proj = [0, 1, 2].map(|w| {
                let d = &e[w];
                let half = |v: &f32| half::f16::from_f32(*v).to_bits().to_le_bytes();
                let suh = bytes.len();
                bytes.extend(d.suh.iter().flat_map(half));
                let svh = bytes.len();
                bytes.extend(d.svh.iter().flat_map(half));
                let mul1 = bytes.len();
                bytes.extend(0x83dc_d12du32.to_le_bytes());
                let trellis = bytes.len();
                bytes.extend(d.words.iter().flat_map(|w| w.to_le_bytes()));
                ProjAt { suh, svh, mul1, trellis, k: d.suh.len(), n: d.svh.len(), tile_words: d.tile_words }
            });
            layout = Some(RecordLayout { bytes: bytes.len(), proj });
            bytes
        })
        .collect();
    (layout.unwrap(), MemRecords(records), shared)
}

/// Experts read as records and computed on the host, the shared one on the GPU, are the definition's sums: alone, with
/// an adapter's low-rank updates, and in a chain that runs the shared expert on the GPU and reads its output back.
#[test]
fn experts_read_as_records_are_the_definitions_sums() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, tw) = (8, 256, 128, 3, 48);
    let lora: Pairs = [
        lora_pairs(count, &[2, count], 2, hidden, ff, 5),
        lora_pairs(count, &[2, 5], 4, hidden, ff, 6),
        lora_pairs(count, &[1, 3, 4, 6, count], 2, ff, hidden, 7),
    ];
    let none: Pairs = [None, None, None];
    for rows in [1usize, 7] {
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29) as f32 - 14.0) / 7.0).collect();
        let xt = Tensor::from_vec(x.clone(), vec![rows, hidden]);
        let lt = Tensor::from_vec(logits.clone(), vec![rows, count + 1]);
        for (adapted, l) in [(false, &none), (true, &lora)] {
            let want = reference_adapted(experts(count, hidden, ff, tw), l, &x, &logits, top_k);
            let size = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let (layout, records, shared) = as_records(experts(count, hidden, ff, tw));
            let mut moe = b.exl3_experts_tiered(layout, Arc::new(records), shared, 0, 0).unwrap();
            if adapted {
                for (which, p) in l.iter().enumerate() {
                    let (s, a, bb, r) = p.as_ref().unwrap();
                    moe.low_rank(which, s, a, bb, *r).unwrap();
                }
            }
            assert!(moe.on_host());
            let what = format!("tiered moe rows={rows} adapted={adapted}");
            let got = moe.forward(&xt, &lt, top_k);
            for (i, (a, e)) in got.data().iter().zip(&want).enumerate() {
                assert!((a - e).abs() <= 0.003 + 1e-3 * size, "{what} [{i}]: {a} != {e}");
            }
            // the shared expert chained, its output read back and given
            let (xd, sd) = (b.vec(rows * hidden), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xd, &x);
            let mut rec = b.begin();
            assert!(rec.moe_shared(moe.as_ref(), &xd, &sd, rows), "{what}: the shared expert chained");
            rec.read(&sd);
            let shared = rec.finish().pop().unwrap();
            let given = moe.forward_given(&xt, &lt, top_k, &shared);
            for (i, (a, e)) in given.data().iter().zip(&want).enumerate() {
                assert!((a - e).abs() <= 0.003 + 1e-3 * size, "{what}, the shared expert given [{i}]: {a} != {e}");
            }
        }
    }
}

/// Experts a GPU holds some of in slots: a prompt's rows planned on the host (what they use taken in, the rest summed
/// there) and a step's row routed on the GPU to the slots (the rest the host's) add up to the definition's sums, before
/// and after slots change hands.
#[test]
fn experts_held_in_slots_and_the_rest_on_the_host_are_the_definitions_sums() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, tw) = (8, 256, 128, 3, 48);
    let near = |got: &[f32], want: &[f32], what: &str| {
        let size = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (a, e)) in got.iter().zip(want).enumerate() {
            assert!((a - e).abs() <= 0.003 + 1e-3 * size, "{what} [{i}]: {a} != {e}");
        }
    };
    let (layout, records, shared) = as_records(experts(count, hidden, ff, tw));
    let moe = b.exl3_experts_tiered(layout, Arc::new(records), shared, 3, 0).unwrap();
    assert!(moe.host_rest() && !moe.on_host(), "{moe:?}");
    for round in 0..4 {
        // a prompt's rows, planned
        let rows = 5;
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101 + round * 7) as f32 - 50.0) / 60.0).collect();
        let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29 + round * 11) as f32 % 29.0 - 14.0) / 7.0).collect();
        let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
        let (xt, lt) = (Tensor::from_vec(x.clone(), vec![rows, hidden]), Tensor::from_vec(logits.clone(), vec![rows, count + 1]));
        let (assign, rest) = moe.plan_rows(&xt, &lt, top_k).unwrap();
        let (xd, out) = (b.vec(rows * hidden), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.moe_rows(moe.as_ref(), &xd, &out, &assign);
        rec.read(&out);
        let mut got = rec.finish().pop().unwrap();
        if let Some(r) = rest {
            for (g, v) in got.iter_mut().zip(r.data()) {
                *g += v;
            }
        }
        near(&got, &want, &format!("round {round}: a prompt's rows planned"));
        // a step's row routed on the GPU, the rest the host's
        for step in 0..3 {
            let x: Vec<f32> = (0..hidden).map(|i| ((i * 29 % 97 + step * 13 + round) as f32 - 48.0) / 50.0).collect();
            let logits: Vec<f32> = (0..count + 1).map(|i| ((i * 31 % 23 + step * 5 + round * 3) as f32 % 23.0 - 11.0) / 5.0).collect();
            let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
            let (xd, out, ld) = (b.vec(hidden), b.vec(hidden), b.vec(count + 1));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &ld, &logits);
            let mut rec = b.begin();
            rec.keep_groups(true);
            assert!(rec.moe_routed(moe.as_ref(), &xd, &out, &ld, top_k, 1), "a step routed on the GPU");
            rec.read(&out);
            let mut got = rec.finish().pop().unwrap();
            let rest = moe.forward_rest(&Tensor::from_vec(x.clone(), vec![1, hidden]), &Tensor::from_vec(logits.clone(), vec![1, count + 1]), top_k);
            if let Some(r) = rest {
                for (g, v) in got.iter_mut().zip(r.data()) {
                    *g += v;
                }
            }
            near(&got, &want, &format!("round {round} step {step}: routed to the slots"));
        }
    }
    let (held, not, taken) = moe.rest_counts().unwrap();
    assert!(held > 0 && not > 0 && taken > 3, "the slots were used and changed hands: {held} held, {not} not, {taken} taken in");
}

/// How long the host takes for a step's row through 1 to 10 of Flash-Next's experts read as records (`--ignored
/// --nocapture`).
#[test]
#[ignore = "timing"]
fn time_host_experts_of_a_step() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, tw) = (16, 2560, 640, 48);
    let (layout, records, shared) = as_records(experts(count, hidden, ff, tw));
    let moe = b.exl3_experts_tiered(layout, Arc::new(records), shared, 0, 0).unwrap();
    let t = moe.as_any().unwrap().downcast_ref::<Exl3MoeTiered>().unwrap();
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    for used in [1usize, 2, 4, 10] {
        let assign = vec![(0..used).map(|e| (e, 0.1)).chain([(count, 0.5)]).collect::<Vec<_>>()];
        for _ in 0..3 {
            t.time_sum_host(&x, &assign);
        }
        let n = 30;
        let start = std::time::Instant::now();
        for _ in 0..n {
            t.time_sum_host(&x, &assign);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / n as f64;
        eprintln!("{used} experts, one row: {ms:.3} ms ({:.3} ms an expert)", ms / used as f64);
    }
}

/// Arm's vectors give every code mul1's value, to the bit.
#[cfg(target_arch = "aarch64")]
#[test]
fn the_vectors_decode_every_code_as_mul1() {
    for first in (0..65536u32).step_by(4) {
        let codes = [first, first + 1, first + 2, first + 3];
        let got = host::neon::values_of(codes);
        for (c, v) in codes.iter().zip(got) {
            assert_eq!(v.to_bits(), mul1(*c).to_bits(), "code {c}: {v} where {}", mul1(*c));
        }
    }
}

/// A prompt's rows over many experts read as records (each expert a task of its own: its record read, then run) are
/// the definition's sums, with an adapter's low-rank updates beside some of them.
#[test]
fn many_experts_read_as_records_each_a_task_are_the_definitions_sums() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, tw, rows) = (40, 256, 128, 6, 48, 30);
    let lora: Pairs = [
        lora_pairs(count, &[3, 7, 21, count], 2, hidden, ff, 15),
        lora_pairs(count, &[3, 8], 4, hidden, ff, 16),
        lora_pairs(count, &[1, 3, 4, 26, count], 2, ff, hidden, 17),
    ];
    let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
    let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 67) as f32 - 33.0) / 9.0).collect();
    let (xt, lt) = (Tensor::from_vec(x.clone(), vec![rows, hidden]), Tensor::from_vec(logits.clone(), vec![rows, count + 1]));
    let used: std::collections::BTreeSet<usize> = (0..rows).flat_map(|r| route(&logits[r * (count + 1)..(r + 1) * (count + 1)], top_k)).map(|(e, _)| e).filter(|&e| e < count).collect();
    assert!(used.len() >= 24, "the rows route to {} experts", used.len());
    let none: Pairs = [None, None, None];
    for (adapted, l) in [(false, &none), (true, &lora)] {
        let want = reference_adapted(experts(count, hidden, ff, tw), l, &x, &logits, top_k);
        let size = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (layout, records, shared) = as_records(experts(count, hidden, ff, tw));
        let mut moe = b.exl3_experts_tiered(layout, Arc::new(records), shared, 0, 0).unwrap();
        if adapted {
            for (which, p) in l.iter().enumerate() {
                let (s, a, bb, r) = p.as_ref().unwrap();
                moe.low_rank(which, s, a, bb, *r).unwrap();
            }
        }
        let got = moe.forward(&xt, &lt, top_k);
        for (i, (a, e)) in got.data().iter().zip(&want).enumerate() {
            assert!((a - e).abs() <= 0.003 + 1e-3 * size, "many experts, adapted={adapted} [{i}]: {a} != {e}");
        }
    }
}

/// An adapter's low-rank updates follow experts into the slots a GPU holds them in: a prompt's planned rows and a
/// step's row routed to the slots, the rest on the host, add up to the adapted definition's sums as slots change hands.
#[test]
fn experts_held_in_slots_take_their_low_rank_updates_with_them() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, tw) = (8, 256, 128, 3, 48);
    let lora: Pairs = [
        lora_pairs(count, &[2, 6, count], 2, hidden, ff, 5),
        lora_pairs(count, &[2, 5], 4, hidden, ff, 6),
        lora_pairs(count, &[0, 1, 3, 4, 6, count], 2, ff, hidden, 7),
    ];
    let near = |got: &[f32], want: &[f32], what: &str| {
        let size = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (a, e)) in got.iter().zip(want).enumerate() {
            assert!((a - e).abs() <= 0.003 + 2e-3 * size, "{what} [{i}]: {a} != {e} (sums up to {size})");
        }
    };
    let (layout, records, shared) = as_records(experts(count, hidden, ff, tw));
    let mut moe = b.exl3_experts_tiered(layout, Arc::new(records), shared, 3, 0).unwrap();
    for (which, p) in lora.iter().enumerate() {
        let (s, a, bb, r) = p.as_ref().unwrap();
        moe.low_rank(which, s, a, bb, *r).unwrap();
    }
    for round in 0..4 {
        let rows = 5;
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101 + round * 7) as f32 - 50.0) / 60.0).collect();
        let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29 + round * 11) as f32 % 29.0 - 14.0) / 7.0).collect();
        let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &x, &logits, top_k);
        let (xt, lt) = (Tensor::from_vec(x.clone(), vec![rows, hidden]), Tensor::from_vec(logits.clone(), vec![rows, count + 1]));
        let (assign, rest) = moe.plan_rows(&xt, &lt, top_k).unwrap();
        let (xd, out) = (b.vec(rows * hidden), b.vec(rows * hidden));
        DeviceChain::upload(&b, &xd, &x);
        let mut rec = b.begin();
        rec.moe_rows(moe.as_ref(), &xd, &out, &assign);
        rec.read(&out);
        let mut got = rec.finish().pop().unwrap();
        if let Some(r) = rest {
            for (g, v) in got.iter_mut().zip(r.data()) {
                *g += v;
            }
        }
        near(&got, &want, &format!("round {round}: a prompt's rows planned, adapted"));
        for step in 0..3 {
            let x: Vec<f32> = (0..hidden).map(|i| ((i * 29 % 97 + step * 13 + round) as f32 - 48.0) / 50.0).collect();
            let logits: Vec<f32> = (0..count + 1).map(|i| ((i * 31 % 23 + step * 5 + round * 3) as f32 % 23.0 - 11.0) / 5.0).collect();
            let want = reference_adapted(experts(count, hidden, ff, tw), &lora, &x, &logits, top_k);
            let (xd, out, ld) = (b.vec(hidden), b.vec(hidden), b.vec(count + 1));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &ld, &logits);
            let mut rec = b.begin();
            rec.keep_groups(true);
            assert!(rec.moe_routed(moe.as_ref(), &xd, &out, &ld, top_k, 1), "a step routed on the GPU");
            rec.read(&out);
            let mut got = rec.finish().pop().unwrap();
            if let Some(r) = moe.forward_rest(&Tensor::from_vec(x.clone(), vec![1, hidden]), &Tensor::from_vec(logits.clone(), vec![1, count + 1]), top_k) {
                for (g, v) in got.iter_mut().zip(r.data()) {
                    *g += v;
                }
            }
            near(&got, &want, &format!("round {round} step {step}: routed to the slots, adapted"));
        }
    }
    let (held, not, taken) = moe.rest_counts().unwrap();
    assert!(held > 0 && not > 0 && taken > 3, "the slots were used and changed hands: {held} held, {not} not, {taken} taken in");
}

/// A few rows routed on the GPU to the slots (a prompt's last tokens after a kept state, a check's rows), several of
/// a row's experts not held: each row's sum with the host's rest is the definition's. (An expert the card lacks goes
/// by slot 0's place, a row's several at once: they must not be taken for one expert's rows, a row each.)
#[test]
fn a_few_rows_routed_to_slots_with_several_experts_not_held_are_the_definitions_sums() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, top_k, tw) = (12, 256, 128, 5, 48);
    let (layout, records, shared) = as_records(experts(count, hidden, ff, tw));
    let moe = b.exl3_experts_tiered(layout, Arc::new(records), shared, 3, 0).unwrap();
    for round in 0..3 {
        for rows in [2usize, 3, 5, 8] {
            let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101 + round * 7 + rows) as f32 - 50.0) / 60.0).collect();
            let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29 + round * 11 + rows * 3) as f32 % 29.0 - 14.0) / 7.0).collect();
            let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
            let size = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let (xd, out, ld) = (b.vec(rows * hidden), b.vec(rows * hidden), b.vec(rows * (count + 1)));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &ld, &logits);
            let mut rec = b.begin();
            rec.keep_groups(true);
            assert!(rec.moe_routed(moe.as_ref(), &xd, &out, &ld, top_k, rows), "{rows} rows routed on the GPU");
            rec.read(&out);
            let mut got = rec.finish().pop().unwrap();
            if let Some(r) = moe.forward_rest(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k) {
                for (g, v) in got.iter_mut().zip(r.data()) {
                    *g += v;
                }
            }
            for (i, (a, e)) in got[..rows * hidden].iter().zip(&want).enumerate() {
                assert!((a - e).abs() <= 0.003 + 1e-3 * size, "round {round}, {rows} rows [{i}]: {a} != {e}");
            }
        }
    }
    let (held, not, _) = moe.rest_counts().unwrap();
    assert!(held > 0 && not > held, "most routings were to experts not held: {held} held, {not} not");
}

/// An expert of a record as its definition reads, from its projections run on the host ([`ProjAt::forward_host`]): gate
/// and up of `x`'s `rows`, the SwiGLU with up clamped to `limit` either way and gate from above (0: none), then down.
fn slot_reference(record: &[u8], layout: &RecordLayout, x: &[f32], rows: usize, limit: f32) -> Vec<f32> {
    let [g, u, d] = layout.proj;
    let (gate, up) = (g.forward_host(record, x, rows), u.forward_host(record, x, rows));
    let hidden: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let (g, u) = if limit > 0.0 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
            g / (1.0 + (-g).exp()) * u
        })
        .collect();
    d.forward_host(record, &hidden, rows)
}

/// Experts put into slots from their records ([`Exl3Slots`]) are each the definition's expert on its rows, whatever
/// slot holds it: a job of one row (a step's) and of several (a prompt's), with no limit and with one that clamps.
#[test]
fn experts_in_slots_are_each_the_definitions() {
    let Some(b) = backend() else { return };
    let (count, hidden, ff, tw) = (6, 256, 128, 48);
    let (layout, records, _) = as_records(experts(count, hidden, ff, tw));
    // slots written through the queue one at a time, and slots the host writes in place several at once (a second
    // round over the first: what the slots held before is all written over)
    let queued = b.exl3_slots(4, hidden, ff, [tw, tw]).expect("room for four slots");
    let mapped = b.exl3_slots_mapped(4, hidden, ff, [tw, tw]).expect("room for four more");
    assert!(queued.fits(&layout) && queued.slots() == 4 && mapped.slots() == 4);
    // experts 5, 2 and 0 in slots 0 to 2 (slot 3 left empty)
    let held = [5usize, 2, 0];
    for (s, &e) in held.iter().enumerate() {
        queued.put(s, &records.0[e], &layout);
    }
    mapped.put_many(0, &[&records.0[1], &records.0[3], &records.0[4]], &layout);
    mapped.put_many(0, &held.map(|e| &records.0[e][..]), &layout);
    for slots in [&queued, &mapped] {
        for limit in [0.0f32, 0.5] {
            for rows in [1usize, 3] {
                let xs: Vec<Vec<f32>> = (0..held.len()).map(|j| (0..rows * hidden).map(|i| (((i + 7 * j) * 37 % 101) as f32 - 50.0) / 40.0).collect()).collect();
                let jobs: Vec<(usize, &[f32])> = xs.iter().enumerate().map(|(s, x)| (s, &x[..])).collect();
                let got = slots.forward(&jobs, limit);
                for (j, &e) in held.iter().enumerate() {
                    let want = slot_reference(&records.0[e], &layout, &xs[j], rows, limit);
                    close(&got[j], &want, &format!("{slots:?}: expert {e} in slot {j}, {rows} rows, limit {limit}"));
                }
            }
        }
    }
    // two sets of slots in one call: each its own jobs'
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 13 % 29) as f32 - 14.0) / 15.0).collect();
    let both = Exl3Slots::forward_many(&[(&queued, &[(1, &x[..])]), (&mapped, &[(2, &x[..]), (0, &x[..])])], 0.0);
    close(&both[0][0], &slot_reference(&records.0[2], &layout, &x, 1, 0.0), "the first set's slot 1");
    close(&both[1][0], &slot_reference(&records.0[0], &layout, &x, 1, 0.0), "the second set's slot 2");
    close(&both[1][1], &slot_reference(&records.0[5], &layout, &x, 1, 0.0), "the second set's slot 0");
    // a call of jobs of one row and of three: each the same bits as in a call of its own kind
    let x3: Vec<f32> = (0..3 * hidden).map(|i| ((i * 11 % 31) as f32 - 15.0) / 16.0).collect();
    let mixed = queued.forward(&[(0, &x[..]), (1, &x3[..]), (2, &x[..])], 0.5);
    let ones = queued.forward(&[(2, &x[..]), (0, &x[..])], 0.5);
    let threes = queued.forward(&[(1, &x3[..])], 0.5);
    assert!(mixed[0] == ones[1] && mixed[2] == ones[0] && mixed[1] == threes[0], "a job's output hangs on the jobs beside it");
    // a slot copied into the other set (the mapped set's slot 0, expert 5, over the queued set's slot 1, expert 2),
    // then the mapped set written over: the copy is the expert it was
    mapped.copy_slot(0, &queued, 1);
    mapped.put_many(0, &[&records.0[3]], &layout);
    let got = queued.forward(&[(1, &x[..]), (0, &x[..])], 0.0);
    close(&got[0], &slot_reference(&records.0[5], &layout, &x, 1, 0.0), "the copied slot");
    close(&got[1], &slot_reference(&records.0[5], &layout, &x, 1, 0.0), "the slot beside it, as it was");
}
