//! The DeepSeek worker's tests: the experts on a GPU against the CPU's, the tiers, and the model against the CPU model.

use super::experts::MARGIN;
use super::trunk::WgpuDense;
use super::*;
use dsv41::model::{Model, ModelOptions};
use std::path::PathBuf;
use std::time::Instant;

fn checkpoint() -> PathBuf {
    std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"D:\deepseek\model"))
}

fn prompt(dir: &std::path::Path, tokens: usize) -> Vec<u32> {
    let tok = dsv41::tokenizer::Tokenizer::load(dir).unwrap();
    // A text kept with the tests (docs/WEBGPU.md as it was on 2026-10-05, some 3,400 tokens), LF whatever the
    // checkout wrote: a CRLF copy is another prompt (other tokens, other experts). The prompt was the docs as they
    // stood, and every edit of them was another prompt: by 2026-10-09 the GPU's logits after its first 400 tokens
    // were the CPU's to a cosine of 0.9908 where this text's are to 0.9990 (the same greedy tokens after both), and
    // the check that asks for 0.998 failed with nothing in the model changed. DSV41_PROMPT_FILE: another text.
    let text: String = match std::env::var("DSV41_PROMPT_FILE") {
        Ok(file) => std::fs::read_to_string(file).expect("DSV41_PROMPT_FILE"),
        Err(_) => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv41-prompt.md")).to_string(),
    }
    .replace("\r\n", "\n");
    let mut ids = vec![0u32];
    ids.extend(tok.encode(&text));
    ids.truncate(tokens);
    ids
}

/// An expert on the GPU gives the CPU's output but for the order of its sums: on a random record (realistic scales)
/// and 33 rows, nearly every output is the same bf16 value, and the rest a few bf16 steps away (an fp8 rounding of
/// the SwiGLU's output can go the other way), never a different answer.
#[test]
fn an_expert_on_the_gpu_gives_the_cpus_output_but_for_its_sums_order() {
    use dsv41::expert::{expert_forward_batch, ExpertJob, ExpertsKernel, DIM, RECORD_BYTES, S1, W1};
    let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut record = vec![0u8; RECORD_BYTES];
    for (i, byte) in record.iter_mut().enumerate() {
        // Weights: any nibbles. Scales (e8m0): 2^-9 .. 2^-6, as a trained expert's are.
        *byte = if i < S1.start { (next() & 255) as u8 } else { 118 + (next() % 4) as u8 };
    }
    assert!(W1.start == 0);
    let rows = 33;
    let x: Vec<f32> = (0..rows * DIM).map(|_| (((next() % 2001) as f32 / 1000.0) - 1.0) * 0.5).collect();
    let weights: Vec<f32> = (0..rows).map(|i| 0.05 + 0.01 * i as f32).collect();
    let cpu = expert_forward_batch(&record, &x, Some(&weights), 10.0);
    let gpu = WgpuExperts::new(&b).unwrap().forward(&[ExpertJob { record: &record, x: &x, weights: &weights }], 10.0).pop().unwrap();
    assert_eq!(cpu.len(), gpu.len());
    let same = cpu.iter().zip(&gpu).filter(|(a, b)| a == b).count();
    let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
    eprintln!("{same} of {} outputs the same; the largest difference {worst:.2e} of the largest output", cpu.len());
    assert!(same as f64 >= 0.9 * cpu.len() as f64, "{same} of {}", cpu.len());
    assert!(worst < 0.05, "{worst}");
}

/// Where a call of a prompt's busy experts on the GPU spends its time: 32 experts of 31 rows (a 2,000-token prompt's
/// average), the whole call and each of its steps.
#[test]
#[ignore = "a timing; needs a WebGPU adapter; run with --nocapture"]
fn measure_a_group_of_experts_on_the_gpu() {
    use dsv41::expert::{ExpertJob, ExpertsKernel, BLOCK, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1, W2, W3};
    use dsv41::formats::fake_quant_fp8;
    let b = gpu();
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let (n, rows) = (32, 31);
    let records: Vec<Vec<u8>> = (0..n)
        .map(|_| (0..RECORD_BYTES).map(|i| if i < S1.start { (next() & 255) as u8 } else { 118 + (next() % 4) as u8 }).collect())
        .collect();
    let x: Vec<f32> = (0..rows * DIM).map(|_| (((next() % 2001) as f32 / 1000.0) - 1.0) * 0.5).collect();
    let weights: Vec<f32> = (0..rows).map(|i| 0.05 + 0.01 * i as f32).collect();
    let jobs: Vec<ExpertJob<'_>> = records.iter().map(|r| ExpertJob { record: r, x: &x, weights: &weights }).collect();
    let kernel = WgpuExperts::new(&b).unwrap();
    kernel.forward(&jobs[..2], 10.0);
    for round in 0..3 {
        let t = Instant::now();
        kernel.forward(&jobs, 10.0);
        eprintln!("round {round}: {n} experts of {rows} rows in {:.3} s", t.elapsed().as_secs_f64());
    }
    drop(kernel);
    // The same steps one at a time.
    let slots = b.record_slots(n, RECORD_BYTES).unwrap();
    let t = Instant::now();
    for (i, r) in records.iter().enumerate() {
        slots.write(i, r);
    }
    let upload = t.elapsed().as_secs_f64();
    let placed: Vec<(DenseGpu, DenseGpu, DenseGpu)> = (0..n)
        .map(|i| (slots.mxfp4(i, W1.start, S1.start, INTER, DIM), slots.mxfp4(i, W3.start, S3.start, INTER, DIM), slots.mxfp4(i, W2.start, S2.start, DIM, INTER)))
        .collect();
    let t = Instant::now();
    let xq = fake_quant_fp8(&x, BLOCK);
    let quant = t.elapsed().as_secs_f64();
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
        placed.iter().flat_map(|(g, u, _)| [(g, xq.as_slice(), rows, 0..INTER), (u, xq.as_slice(), rows, 0..INTER)]).collect();
    let t = Instant::now();
    let sums = ggml_rs_wgpu::dense::forward_batch(&items);
    let gate_up = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let hq: Vec<Vec<f32>> = sums.chunks(2).map(|p| fake_quant_fp8(&dsv41::expert::swiglu(&p[0], &p[1], Some(&weights), 10.0), BLOCK)).collect();
    let host = t.elapsed().as_secs_f64();
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> = placed.iter().zip(&hq).map(|((_, _, d), h)| (d, h.as_slice(), rows, 0..DIM)).collect();
    let t = Instant::now();
    ggml_rs_wgpu::dense::forward_batch(&items);
    let down = t.elapsed().as_secs_f64();
    eprintln!(
        "the records' writes queued {upload:.3} s, the input's quantization {quant:.3} s, gate and up (the uploads landing first) {gate_up:.3} s, the SwiGLU on the host {host:.3} s, down {down:.3} s"
    );
    // Gate and up again, the records already there: the matmuls and the round trip alone.
    let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> =
        placed.iter().flat_map(|(g, u, _)| [(g, xq.as_slice(), rows, 0..INTER), (u, xq.as_slice(), rows, 0..INTER)]).collect();
    let t = Instant::now();
    ggml_rs_wgpu::dense::forward_batch(&items);
    eprintln!("gate and up with the records in place {:.3} s", t.elapsed().as_secs_f64());
}

fn gpu() -> WgpuBackend {
    let gb: u64 = std::env::var("DSV41_WEBGPU_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(27);
    WgpuBackend::new(Some(gb << 30)).expect("a WebGPU adapter")
}

/// A store whose record for `(layer, expert)` is made from the two numbers: nibbles from a generator they seed,
/// scales of a real record's size.
struct Made;

impl oaiy_engine::store::WeightStore for Made {
    fn record_bytes(&self) -> usize {
        dsv41::expert::RECORD_BYTES
    }
    fn shape(&self) -> (u32, u32) {
        (2, 64)
    }
    fn fetch(&self, layer: u32, expert: u32, dst: &mut [u8]) -> oaiy_engine::Result<()> {
        let mut s = (0x9E37_79B9_7F4A_7C15u64 ^ ((layer as u64) << 32 | expert as u64)).wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
        let scales = dsv41::expert::S1.start;
        for (i, chunk) in dst.chunks_mut(8).enumerate() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            for (j, b) in chunk.iter_mut().enumerate() {
                let byte = (s >> (8 * j)) as u8;
                *b = if i * 8 + j < scales { byte } else { 118 + byte % 4 };
            }
        }
        Ok(())
    }
}

/// A decode step's experts in one submit, the SwiGLU between an expert's projections on the device, give what the
/// two calls with the SwiGLU on the host give: on six records and a row each, nearly every one of the 30,720
/// outputs the same bf16 value, and the rest a few bf16 steps away (the device's exponential is not the host's to
/// the bit, and a rounding of the activation can then go the other way), none far off.
#[test]
#[ignore = "needs a WebGPU adapter (some 700 MB of it)"]
fn a_steps_experts_in_one_submit_give_the_two_calls_outputs_but_for_a_rounding() {
    use dsv41::expert::{BLOCK, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1, W2, W3};
    use dsv41::formats::{fake_quant_fp8, to_bf16};
    use ggml_rs_wgpu::dense::{forward_batch, forward_units, Unit};
    use oaiy_engine::store::WeightStore;
    let b = WgpuBackend::new(Some(1 << 30)).expect("a WebGPU adapter");
    let slots = b.record_slots(6, RECORD_BYTES).unwrap();
    let mut record = vec![0u8; RECORD_BYTES];
    for e in 0..6 {
        Made.fetch(1, e as u32, &mut record).unwrap();
        slots.write(e, &record);
    }
    let mats: Vec<[DenseGpu; 3]> =
        (0..6).map(|i| [slots.mxfp4(i, W1.start, S1.start, INTER, DIM), slots.mxfp4(i, W3.start, S3.start, INTER, DIM), slots.mxfp4(i, W2.start, S2.start, DIM, INTER)]).collect();
    let xs: Vec<Vec<f32>> = (0..6).map(|e| fake_quant_fp8(&(0..DIM).map(|i| (((i * 37 + e * 11) % 101) as f32 - 50.0) / 60.0).collect::<Vec<f32>>(), BLOCK)).collect();
    let weights: Vec<f32> = (0..6).map(|e| 0.05 + 0.07 * e as f32).collect();
    for limit in [10.0f32, 0.0] {
        // the two calls, the SwiGLU on the host between them
        let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> = (0..6).flat_map(|i| [(&mats[i][0], &xs[i][..], 1, 0..INTER), (&mats[i][1], &xs[i][..], 1, 0..INTER)]).collect();
        let sums = forward_batch(&items);
        let hq: Vec<Vec<f32>> = sums.chunks(2).zip(&weights).map(|(p, w)| fake_quant_fp8(&dsv41::expert::swiglu(&p[0], &p[1], Some(&[*w]), limit), BLOCK)).collect();
        let items: Vec<(&DenseGpu, &[f32], usize, std::ops::Range<usize>)> = (0..6).map(|i| (&mats[i][2], &hq[i][..], 1, 0..DIM)).collect();
        let want: Vec<Vec<f32>> = forward_batch(&items).into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect();
        // one submit
        let units: Vec<Unit<'_>> = (0..6).map(|i| Unit { gate: &mats[i][0], up: &mats[i][1], down: &mats[i][2], x: &xs[i], weight: weights[i] }).collect();
        let (got, none) = forward_units(&units, &[], limit).expect("six experts fit the arena");
        assert!(none.is_empty());
        let got: Vec<Vec<f32>> = got.into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect();
        let steps = |v: f32| (v.to_bits() >> 16) as i64 * if v < 0.0 { -1 } else { 1 };
        let (mut same, mut worst, mut far) = (0usize, 0i64, 0.0f32);
        let scale = want.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            if g.to_bits() == w.to_bits() {
                same += 1;
            } else {
                worst = worst.max((steps(*g) - steps(*w)).abs());
                far = far.max((g - w).abs());
            }
        }
        eprintln!("limit {limit}: {same} of {} outputs the same; the rest at most {worst} bf16 steps away ({far:.3e} of a largest {scale:.3e})", 6 * DIM);
        assert!(scale > 0.0 && same * 100 >= 6 * DIM * 97, "{same} the same");
        assert!(far <= scale * 0.01, "an output {far} away of a largest {scale}");
    }
}

/// While the server idles the card takes RAM's most used experts in place of its least used, each in one place
/// before and after: on a tier of four slots, filled as a decode step's misses fill it, two experts used ten
/// times as much as two of the card's change places with them, the displaced two are in RAM again with their own
/// records, one used a little stays where it is, and what the card then computes for one it took in is what that
/// expert's record gives. Nothing comes in on a request's path once the tier is full.
#[test]
#[ignore = "needs a WebGPU adapter (some 700 MB of it)"]
fn an_idle_rebalance_gives_the_card_the_most_used_experts_and_ram_the_displaced() {
    use dsv41::expert::{ExpertJob, ExpertsKernel, DIM, RECORD_BYTES};
    use oaiy_engine::store::WeightStore;
    let record = RECORD_BYTES as u64;
    let b = WgpuBackend::new(Some(dsv41::moe::GPU_GROUP as u64 * record + MARGIN + 4 * record + record / 2)).expect("a WebGPU adapter");
    let k = WgpuExperts::new(&b).unwrap();
    assert_eq!(k.tier().0, 4, "a tier of four slots");
    k.set_exclusive(true);
    let (store, cache) = (Made, oaiy_engine::ecache::Ecache::new(8 * RECORD_BYTES, RECORD_BYTES, oaiy_engine::CachePolicy::Lfru));
    // the model's counts of uses, as its MoE keeps them: a call that routed to an expert one
    let uses = dsv41::moe::Uses::new(2, 64);
    // experts 0 to 3 come in as a decode step's misses do (each used twice), and leave RAM
    for e in 0..4u32 {
        let lease = cache.acquire(0, e, &store).unwrap();
        assert_eq!(k.holds(0, &[e], &[1]), [false]);
        assert_eq!(k.offer(0, &[(e, &lease[..])]), [e]);
        drop(lease);
        assert!(cache.remove(0, e));
        uses.add(0, &[e, e]);
    }
    assert_eq!(k.holds(0, &[0, 1, 2, 3], &[1; 4]), [true; 4]);
    // a fifth does not, the tier full: a swap on a request's path would leave what it let go in no tier
    let lease = cache.acquire(0, 4, &store).unwrap();
    assert_eq!(k.holds(0, &[4], &[1]), [false]);
    assert!(k.offer(0, &[(4, &lease[..])]).is_empty());
    drop(lease);
    // 4 and 5 are then used some twenty times (and 1, which the card holds), 6 three times; 5 and 6 are in RAM too
    uses.add(0, &[4, 4, 4]);
    for _ in 0..20 {
        uses.add(0, &[1, 4, 5]);
    }
    for _ in 0..3 {
        uses.add(0, &[6]);
    }
    drop(cache.acquire(0, 5, &store).unwrap());
    drop(cache.acquire(0, 6, &store).unwrap());
    assert_eq!(k.rebalance(&store, &cache, &uses, 8), 2, "4 and 5 in place of two of the card's least used");
    assert_eq!(k.holds(0, &[1, 4, 5, 6], &[1; 4]), [true, true, true, false]);
    // the idle reading leaves out of RAM what the card holds now, and no longer what it held before
    assert!(k.elsewhere(0, 1) && k.elsewhere(0, 4) && k.elsewhere(0, 5) && !k.elsewhere(0, 6));
    assert_eq!([0u32, 2, 3].into_iter().filter(|&e| k.elsewhere(0, e)).count(), 1, "of the three it held, the one it kept");
    assert!(!cache.probe(0, 4) && !cache.probe(0, 5) && cache.probe(0, 6), "what the card took in is not in RAM too");
    let out: Vec<u32> = [0u32, 2, 3].into_iter().filter(|&e| !k.holds(0, &[e], &[1])[0]).collect();
    assert_eq!(out.len(), 2, "two of the three used least gave way: {out:?}");
    for &e in &out {
        assert!(cache.probe(0, e), "a displaced expert is in RAM again");
        let mut own = vec![0u8; RECORD_BYTES];
        store.fetch(0, e, &mut own).unwrap();
        assert!(cache.acquire(0, e, &store).unwrap()[..] == own[..], "with its own record");
    }
    // what the card computes for one it took in is what its record gives
    let x: Vec<f32> = (0..DIM).map(|i| ((i * 37 % 101) as f32 - 50.0) / 80.0).collect();
    let weights = [0.3f32];
    let held = k.forward_held(0, &[(4, &x[..], &weights[..])], 10.0).pop().unwrap();
    let mut own = vec![0u8; RECORD_BYTES];
    store.fetch(0, 4, &mut own).unwrap();
    let fresh = k.forward(&[ExpertJob { record: &own, x: &x, weights: &weights }], 10.0).pop().unwrap();
    assert!(held == fresh && held.iter().any(|v| *v != 0.0), "the slot holds expert 4's record");
    assert_eq!(k.rebalance(&store, &cache, &uses, 8), 0, "and then there is nothing to move");
    assert_eq!(k.moved(), 2);
    // The counts age (the model halves them every so many decode steps). Experts 6 and 7 are then used thirty
    // times: 5, which the card took in after twenty uses, would keep its slot against thirty (twice as much and
    // four more is forty-four) but for the halving.
    assert_eq!((0..dsv41::moe::USES_AGE_STEPS).filter(|_| uses.step()).count(), 1);
    drop(cache.acquire(0, 7, &store).unwrap());
    for _ in 0..30 {
        uses.add(0, &[6, 7]);
    }
    assert_eq!(k.rebalance(&store, &cache, &uses, 8), 2, "6 and 7 in place of the card's two least used, one of them 5");
    assert_eq!(k.holds(0, &[1, 4, 5, 6, 7], &[1; 5]), [true, true, false, true, true]);
    assert!(cache.probe(0, 5) && !cache.probe(0, 6) && !cache.probe(0, 7));
    // 5 came back to RAM with the count it had, not as a newcomer: of RAM's records it is not the next to go
    for e in 8..14u32 {
        drop(cache.acquire(0, e, &store).unwrap());
    }
    assert!(cache.probe(0, 5), "a displaced expert keeps its place in RAM against records used once");
}

/// A usage profile written and read again gives the counts it was written from, and the order a start-up reads
/// the experts in is the most used first, then by number (every layer's expert e before any layer's e + 1);
/// another model's file, or one cut short, changes nothing. With a profile the first card's tier and the other
/// cards' share are planned from that order: the tier the most used, and the idle reading leaves both out of RAM.
#[test]
#[ignore = "needs a WebGPU adapter (some 700 MB of it)"]
fn a_usage_profile_orders_a_start_ups_tiers() {
    use dsv41::expert::{ExpertsKernel, RECORD_BYTES};
    let dir = std::env::temp_dir().join(format!("oaiy-usage-{}", std::process::id()));
    let path = dir.join("profile.bin");
    let uses = dsv41::moe::Uses::new(2, 64);
    assert!(!read_usage(&path, &uses), "no profile yet");
    for _ in 0..9 {
        uses.add(1, &[40]);
    }
    for _ in 0..5 {
        uses.add(0, &[7, 3]);
    }
    uses.add(1, &[0]);
    write_usage(&path, &uses).unwrap();
    let again = dsv41::moe::Uses::new(2, 64);
    assert!(read_usage(&path, &again) && again.counts() == uses.counts());
    assert!(!read_usage(&path, &dsv41::moe::Uses::new(2, 32)), "another model's shape");
    std::fs::write(&path, &std::fs::read(&path).unwrap()[..40]).unwrap();
    assert!(!read_usage(&path, &dsv41::moe::Uses::new(2, 64)), "a file cut short");
    let _ = std::fs::remove_dir_all(&dir);
    let order = again.order();
    assert_eq!(order.len(), 128);
    assert_eq!(order[..6], [(1, 40), (0, 3), (0, 7), (1, 0), (0, 0), (0, 1)]);
    // a tier of four slots takes the first four of the order, read from the drive while idle, not RAM's
    let record = RECORD_BYTES as u64;
    let b = WgpuBackend::new(Some(dsv41::moe::GPU_GROUP as u64 * record + MARGIN + 4 * record + record / 2)).expect("a WebGPU adapter");
    let mut k = WgpuExperts::new(&b).unwrap();
    k.set_exclusive(true);
    k.start_from(&order);
    let (store, cache) = (Made, oaiy_engine::ecache::Ecache::new(8 * RECORD_BYTES, RECORD_BYTES, oaiy_engine::CachePolicy::Lfru));
    assert!(k.elsewhere(1, 40) && k.elsewhere(0, 3) && k.elsewhere(0, 7) && k.elsewhere(1, 0) && !k.elsewhere(0, 0), "a slot waits for each of the four");
    assert!(k.pin_some(&store, &cache, 3), "three read, one to go");
    assert!(!k.pin_some(&store, &cache, 3), "and then the tier is as planned");
    assert!(k.elsewhere(1, 40) && k.elsewhere(1, 0) && !k.elsewhere(0, 0), "and holds them");
    assert_eq!(k.holds(1, &[40, 0], &[1; 2]), [true, true]);
    assert_eq!(k.holds(0, &[3, 7, 0], &[1; 3]), [true, true, false]);
    assert!(cache.is_empty(), "none of it through RAM");
}

/// The trunk and a prompt's busy experts on the GPU give the CPU model's answer: on a prompt long enough for experts
/// to be busy (400 tokens) the last position's logits agree closely, and the next greedy tokens are the same.
#[test]
#[ignore = "loads the 510 GB checkpoint twice; needs a WebGPU adapter"]
fn the_trunk_and_the_experts_on_the_gpu_answer_as_the_cpu_model_does() {
    let dir = checkpoint();
    let meta = dir.join("engram_meta.safetensors");
    let ids = prompt(&dir, 400);
    let opts = ModelOptions { max_seq: 512, expert_cache_bytes: 24 << 30, direct_io: true };
    let run = |model: &mut Model| {
        let mut logits = model.forward(&ids, 0, &mut |_, _| {}).unwrap();
        let first = logits.clone();
        let mut tokens = Vec::new();
        for step in 0..6 {
            let next = dsv41::model::argmax(&logits);
            tokens.push(next);
            logits = model.forward(&[next], ids.len() + step, &mut |_, _| {}).unwrap();
        }
        (first, tokens)
    };
    let (cpu_logits, cpu_tokens) = run(&mut Model::load(&dir, &meta, &opts).unwrap());
    let backend: Arc<dyn ggml_rs::Backend> = Arc::new(gpu());
    let mut model = Model::load(&dir, &meta, &opts).unwrap();
    let (count, bytes) = offload(&mut model, backend.as_any().downcast_ref::<WgpuBackend>().unwrap());
    let kernel = Arc::new(WgpuExperts::new(backend.as_any().downcast_ref::<WgpuBackend>().unwrap()).unwrap());
    model.set_experts_kernel(Some(Arc::clone(&kernel) as Arc<dyn dsv41::expert::ExpertsKernel>));
    eprintln!("{count} dense matrices on the GPU, {:.2} GB, and the experts", bytes as f64 / 1e9);
    let (gpu_logits, gpu_tokens) = run(&mut model);
    let (slots, hits, misses, admitted) = kernel.tier();
    eprintln!("the experts kept on the GPU: {hits} hits / {misses} misses, {admitted} taken in, {slots} slots");
    // The decode steps computed some of their experts there, so the same tokens cover those too.
    assert!(hits > 0, "no expert was computed from the GPU's tier");
    let dot: f64 = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| *a as f64 * *b as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (norm(&cpu_logits) * norm(&gpu_logits));
    let max_diff = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("logits: cosine {cosine:.6}, largest difference {max_diff:.4}; tokens CPU {cpu_tokens:?}, GPU {gpu_tokens:?}");
    assert!(count > 0);
    // The sums' order differs on the GPU: an expert's outputs are the CPU's to the bit but for about one in 170,000
    // (`an_expert_on_the_gpu_gives_the_cpus_output_but_for_its_sums_order`), and those few, through the fp8
    // roundings of 40 layers and a 400-token prompt's experts, leave the logits close, not equal (measured 2026-10-05:
    // cosine 0.99882, the same six tokens; the trunk alone, cosine 1.000000).
    assert!(cosine > 0.998, "cosine {cosine}");
    assert_eq!(cpu_tokens, gpu_tokens);
}

/// The CPU model's measurement (dsv41/tests/cpu_speed.rs) with the trunk on the GPU.
#[test]
#[ignore = "loads the 510 GB checkpoint and prefills a 2,000-token prompt; needs a WebGPU adapter"]
fn measure_prefill_and_warm_decode_with_the_trunk_on_the_gpu() {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    let dir = checkpoint();
    let meta = dir.join("engram_meta.safetensors");
    let tokens: usize = std::env::var("DSV41_PROMPT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let steps: usize = std::env::var("DSV41_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let gb: usize = std::env::var("DSV41_CACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
    let ids = prompt(&dir, tokens);
    let opts = ModelOptions { max_seq: ids.len() + steps + 8, expert_cache_bytes: gb << 30, direct_io: true };
    let mut model = Model::load(&dir, &meta, &opts).unwrap();
    let backend: Arc<dyn ggml_rs::Backend> = Arc::new(gpu());
    let t = Instant::now();
    let (count, bytes) = offload(&mut model, backend.as_any().downcast_ref::<WgpuBackend>().unwrap());
    eprintln!("{count} dense matrices on the GPU, {:.2} GB, in {:.1} s", bytes as f64 / 1e9, t.elapsed().as_secs_f64());
    let mut kernel = None;
    if std::env::var("DSV41_GPU_EXPERTS").map_or(true, |v| v != "0") {
        let k = Arc::new(WgpuExperts::new(backend.as_any().downcast_ref::<WgpuBackend>().unwrap()).expect("room for a record"));
        eprintln!("{} record slots for a prompt's experts, {} kept between passes", k.slots(), k.tier().0);
        if std::env::var("DSV41_TIER_EXCLUSIVE").is_ok_and(|v| v == "1") {
            k.set_exclusive(true);
            eprintln!("what the tier takes in leaves RAM");
        }
        if std::env::var("DSV41_TIER_ADMIT").is_ok_and(|v| v == "0") {
            k.set_admit_on_decode(false);
            eprintln!("a decode step's misses kept out of the tier");
        }
        model.set_experts_kernel(Some(Arc::clone(&k) as Arc<dyn dsv41::expert::ExpertsKernel>));
        kernel = Some(k);
        eprintln!("a prompt's busy experts on the GPU");
    }
    let spans: RefCell<BTreeMap<&'static str, f64>> = RefCell::new(BTreeMap::new());
    let last = RefCell::new(Instant::now());
    let mut trace = |name: &str, _: &[f32]| {
        let now = Instant::now();
        let what = if name.ends_with(".attn_out") { "attention" } else if name.ends_with(".moe_out") { "moe" } else { "other" };
        *spans.borrow_mut().entry(what).or_default() += now.duration_since(*last.borrow()).as_secs_f64();
        *last.borrow_mut() = now;
    };
    let report = |label: &str, secs: f64, before: oaiy_engine::ecache::CacheStats, after: oaiy_engine::ecache::CacheStats, spans: &BTreeMap<&'static str, f64>| {
        let read = after.bytes_read - before.bytes_read;
        let tier = kernel.as_ref().map_or(String::new(), |k| {
            let (slots, hits, misses, admitted) = k.tier();
            format!("; on the GPU so far: {hits} hits / {misses} misses, {admitted} taken in, {slots} slots")
        });
        eprintln!(
            "{label}: {secs:.2} s; read {:.2} GB, {} hits / {} misses{tier}; {}
{}
{}",
            read as f64 / 1e9,
            after.hits - before.hits,
            after.misses - before.misses,
            spans.iter().map(|(k, v)| format!("{k} {v:.2} s")).collect::<Vec<_>>().join(", "),
            dsv41::profile::take_line(),
            ggml_rs_wgpu::profile::take_dense_line()
        );
    };
    let before = model.expert_cache().stats();
    dsv41::profile::take();
    *last.borrow_mut() = Instant::now();
    let t = Instant::now();
    let mut logits = model.forward(&ids, 0, &mut trace).unwrap();
    report(&format!("prompt of {} tokens", ids.len()), t.elapsed().as_secs_f64(), before, model.expert_cache().stats(), &spans.borrow());
    let mut times = Vec::new();
    for step in 0..steps {
        let next = dsv41::model::argmax(&logits);
        spans.borrow_mut().clear();
        let before = model.expert_cache().stats();
        *last.borrow_mut() = Instant::now();
        let t = Instant::now();
        logits = model.forward(&[next], ids.len() + step, &mut trace).unwrap();
        let secs = t.elapsed().as_secs_f64();
        times.push(secs);
        report(&format!("step {step} (token {next})"), secs, before, model.expert_cache().stats(), &spans.borrow());
    }
    let tail = &times[times.len() / 2..];
    eprintln!("last {} steps: {:.2} s a token on average", tail.len(), tail.iter().sum::<f64>() / tail.len() as f64);
}

/// A projection of projections in one call ([`dsv41::linear::chained_together`]) is the two calls to the bit:
/// eight groups of a bf16 weight (a layer's `wo_a`) whose row an fp8 weight (`wo_b`) takes quantized, and a bf16
/// weight after a bf16 weight (the row rounded and not quantized), each on rows of several sizes of value.
#[test]
#[ignore = "needs a WebGPU adapter"]
fn a_chained_projection_is_the_two_calls_to_the_bit() {
    use dsv41::linear::{chained_together, forward_together, Out};
    let b = WgpuBackend::new(Some(1 << 30)).expect("a WebGPU adapter");
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let (groups, gd, rank, n) = (8usize, 512usize, 128usize, 1024usize);
    let bf16 = |n: usize, k: usize, next: &mut dyn FnMut() -> u64| DenseData::Bf16 { w: (0..n * k).map(|_| ((((next() % 2000) as f32 / 1000.0) - 1.0).to_bits() >> 16) as u16).collect(), n, k };
    let fp8 = |n: usize, k: usize, next: &mut dyn FnMut() -> u64| DenseData::Fp8 {
        w: (0..n * k).map(|_| { let b = (next() & 255) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect(),
        scales: (0..n.div_ceil(32) * (k / 32)).map(|_| 2f32.powi((next() % 10) as i32 - 8)).collect(),
        n,
        k,
    };
    let device = |data: DenseData, n: usize, k: usize, fp8: bool| Weight::Device { kernel: Arc::new(WgpuDense(b.dense(data).unwrap().expect("on the GPU"))), n, k, fp8 };
    let a = device(bf16(groups * rank, gd, &mut next), groups * rank, gd, false);
    let thens = [(device(fp8(n, groups * rank, &mut next), n, groups * rank, true), "an fp8 weight after"), (device(bf16(n, groups * rank, &mut next), n, groups * rank, false), "a bf16 weight after")];
    for (then, what) in thens {
        for scale in [1.0f32, 1e-3, 40.0] {
            let xs: Vec<Vec<f32>> = (0..groups).map(|_| (0..gd).map(|_| (((next() % 2001) as f32 / 1000.0) - 1.0) * scale).collect()).collect();
            let two: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> = xs.iter().enumerate().map(|(g, x)| (&a, x.as_slice(), 1, g * rank..(g + 1) * rank, Out::Bf16)).collect();
            let row: Vec<f32> = forward_together(&two).concat();
            let want = then.forward(&row, 1, Out::Bf16);
            let first: Vec<(&Weight, &[f32], std::ops::Range<usize>)> = xs.iter().enumerate().map(|(g, x)| (&a, x.as_slice(), g * rank..(g + 1) * rank)).collect();
            let got = chained_together(&first, &then, Out::Bf16).expect("the chain goes through the arena");
            assert_eq!(got.len(), n);
            assert!(want.iter().any(|v| *v != 0.0), "{what}: sums that are something");
            let differing = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
            assert_eq!(differing, 0, "{what}, inputs of {scale}: {differing} of {n} sums differ");
        }
    }
}
