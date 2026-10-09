//! The dense weights' tests: each kind against an oracle, the records, the arena's calls, and their timings.

use super::*;

/// The reference's own decoding of an e4m3 byte (dsv41 formats::fp8_e4m3_to_f32), for the oracle.
fn e4m3(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) as i32, ((b >> 3) & 15) as i32, (b & 7) as f32);
    let v = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if s == 1 {
        -v
    } else {
        v
    }
}

/// dsv41's FP4_VALUES.
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

fn backend() -> Option<WgpuBackend> {
    WgpuBackend::new(Some(1 << 30)).ok()
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn copy(data: &DenseData) -> DenseData {
    match data {
        DenseData::Fp8 { w, scales, n, k } => DenseData::Fp8 { w: w.clone(), scales: scales.clone(), n: *n, k: *k },
        DenseData::Bf16 { w, n, k } => DenseData::Bf16 { w: w.clone(), n: *n, k: *k },
        DenseData::Mxfp4 { w, scales, n, k } => DenseData::Mxfp4 { w: w.clone(), scales: scales.clone(), n: *n, k: *k },
    }
}

/// `[t, rows]` in f64: the oracle.
fn oracle(data: &DenseData, x: &[f32], t: usize, rows: Range<usize>) -> Vec<f64> {
    let (_, k) = data.nk();
    let w = |r: usize, c: usize| -> f64 {
        match data {
            DenseData::Fp8 { w, scales, .. } => e4m3(w[r * k + c]) as f64 * scales[(r / 32) * (k / 32) + c / 32] as f64,
            DenseData::Bf16 { w, .. } => f32::from_bits((w[r * k + c] as u32) << 16) as f64,
            DenseData::Mxfp4 { w, scales, .. } => {
                let byte = w[(r * k + c) / 2];
                let nib = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                E2M1[nib as usize] as f64 * scales[r * (k / 32) + c / 32] as f64
            }
        }
    };
    let mut out = Vec::new();
    for tt in 0..t {
        for r in rows.clone() {
            out.push((0..k).map(|c| x[tt * k + c] as f64 * w(r, c)).sum());
        }
    }
    out
}

fn fp8_data(n: usize, k: usize, seed: u64) -> DenseData {
    let mut next = rng(seed);
    // Bytes that are not NaN (0x7f / 0xff), scales 2^-8 .. 2^1.
    let w = (0..n * k).map(|_| { let b = (next() & 255) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
    let scales = (0..n.div_ceil(32) * (k / 32)).map(|_| 2f32.powi((next() % 10) as i32 - 8)).collect();
    DenseData::Fp8 { w, scales, n, k }
}

fn bf16_data(n: usize, k: usize, seed: u64) -> DenseData {
    let mut next = rng(seed);
    let w = (0..n * k).map(|_| ((((next() % 2000) as f32 / 1000.0) - 1.0).to_bits() >> 16) as u16).collect();
    DenseData::Bf16 { w, n, k }
}

fn mxfp4_data(n: usize, k: usize, seed: u64) -> DenseData {
    let mut next = rng(seed);
    let w = (0..n * k / 2).map(|_| (next() & 255) as u8).collect();
    let scales = (0..n * (k / 32)).map(|_| 2f32.powi((next() % 12) as i32 - 9)).collect();
    DenseData::Mxfp4 { w, scales, n, k }
}

fn input(t: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut next = rng(seed);
    (0..t * k).map(|_| ((next() % 2001) as f32 / 1000.0) - 1.0).collect()
}

fn close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(((*g as f64) - w).abs() <= 2e-5 * scale + 1e-6, "{what}: [{i}] {g} against {w}");
    }
}

#[test]
fn every_fp8_byte_decodes_as_the_reference_decodes_it() {
    let Some(b) = backend() else { return };
    // One row of the 256 bytes (and a second tile row), scale 1: a one-hot input reads each weight back alone.
    let (n, k) = (33, 256);
    // The two NaN bytes (0x7f, 0xff) left out: a NaN weight makes every sum it is in NaN, even times zero.
    let w: Vec<u8> = (0..n * k).map(|i| (i % 256) as u8).map(|b| if b & 0x7f == 0x7f { 0 } else { b }).collect();
    let data = DenseData::Fp8 { w: w.clone(), scales: vec![1.0; 2 * (k / 32)], n, k };
    let g = b.dense(data).unwrap().expect("on the GPU");
    // Every byte but the two NaNs (0x7f, 0xff), which no checkpoint holds.
    for col in (0usize..256).filter(|c| c & 0x7f != 0x7f) {
        let mut x = vec![0.0f32; k];
        x[col] = 1.0;
        let y = g.forward(&x, 1, 0..1);
        let want = e4m3(col as u8);
        // Exactly (by value: the sum of -0 with the +0 products beside it is +0).
        assert_eq!(y[0], want, "byte {col:#04x}");
    }
}

#[test]
fn every_e2m1_nibble_decodes_as_the_reference_decodes_it() {
    let Some(b) = backend() else { return };
    // One row holding the 16 nibbles twice (k = 32), scale 1.
    let w: Vec<u8> = (0..16u8).map(|i| ((2 * i) % 16) | (((2 * i + 1) % 16) << 4)).collect();
    let g = b.dense(DenseData::Mxfp4 { w, scales: vec![1.0], n: 1, k: 32 }).unwrap().expect("on the GPU");
    for col in 0..32 {
        let mut x = vec![0.0f32; 32];
        x[col] = 1.0;
        assert_eq!(g.forward(&x, 1, 0..1)[0], E2M1[col % 16], "nibble {}", col % 16);
    }
}

#[test]
fn each_kind_matches_the_oracle_for_a_decode_step_and_a_prompt() {
    let Some(b) = backend() else { return };
    for (n, k) in [(96, 64), (70, 160), (300, 512)] {
        for (data, kind) in [(fp8_data(n, k, n as u64), "fp8"), (bf16_data(n, k, k as u64), "bf16"), (mxfp4_data(n, k, (n * k) as u64), "mxfp4")] {
            let want_data = copy(&data);
            let g = b.dense(data).unwrap().expect("on the GPU");
            for t in [1usize, 3, 8, 9, 70, 130] {
                let x = input(t, k, (t * 31 + n) as u64);
                for rows in [0..n, 5..n.min(77), n - 1..n] {
                    let got = g.forward(&x, t, rows.clone());
                    close(&got, &oracle(&want_data, &x, t, rows.clone()), &format!("{kind} [{n}, {k}] t={t} rows {rows:?}"));
                }
            }
        }
    }
}

#[test]
fn a_batch_gives_each_weight_its_own_answer_in_one_submit() {
    let Some(b) = backend() else { return };
    let datas = [mxfp4_data(64, 96, 1), mxfp4_data(96, 64, 2), fp8_data(40, 32, 3), bf16_data(33, 64, 4)];
    let wants: Vec<DenseData> = datas.iter().map(copy).collect();
    let gs: Vec<Arc<DenseGpu>> = datas.into_iter().map(|d| b.dense(d).unwrap().unwrap()).collect();
    let shapes = [(5usize, 0..64usize), (12, 10..96), (1, 0..40), (0, 0..33)];
    let xs: Vec<Vec<f32>> = gs.iter().zip(&shapes).map(|(g, (t, _))| input(*t, g.k(), *t as u64 + 7)).collect();
    let items: Vec<(&DenseGpu, &[f32], usize, Range<usize>)> = gs.iter().zip(&xs).zip(&shapes).map(|((g, x), (t, rows))| (&**g, x.as_slice(), *t, rows.clone())).collect();
    let got = forward_batch(&items);
    for (i, ((want, x), (t, rows))) in wants.iter().zip(&xs).zip(&shapes).enumerate() {
        close(&got[i], &oracle(want, x, *t, rows.clone()), &format!("item {i}"));
    }
    assert!(got[3].is_empty(), "no tokens, no rows");
}

#[test]
fn a_weight_beyond_the_budget_or_with_an_odd_k_stays_with_the_caller() {
    let Ok(small) = WgpuBackend::new(Some(1000)) else { return };
    assert!(small.dense(bf16_data(64, 64, 1)).unwrap().is_none());
    assert_eq!(small.usage().0, 0);
    let Some(b) = backend() else { return };
    assert!(b.dense(DenseData::Bf16 { w: vec![0; 2 * 48], n: 2, k: 48 }).unwrap().is_none());
    let g = b.dense(bf16_data(64, 64, 2)).unwrap().unwrap();
    assert!(b.usage().0 > 0);
    drop(g);
    assert_eq!(b.usage().0, 0);
}

#[test]
fn every_e8m0_scale_of_a_record_decodes_as_the_reference_decodes_it() {
    let Some(b) = backend() else { return };
    // A record of one matrix [256, 32]: every weight the nibble 2 (1.0), row r's scale the byte r. A one-hot input
    // reads each row's scale back.
    let (n, k) = (256, 32);
    let w_bytes = n * k / 2;
    let mut record = vec![0x22u8; w_bytes];
    record.extend((0..n).map(|r| r as u8));
    let slots = b.record_slots(1, record.len()).expect("room for a slot");
    slots.write(0, &record);
    let m = slots.mxfp4(0, 0, w_bytes, n, k);
    let mut x = vec![0.0f32; k];
    x[5] = 1.0;
    let y = m.forward(&x, 1, 0..n);
    // Every byte but 255, NaN, which no checkpoint holds (and which WGSL lets a GPU treat as a number).
    for e in 1..255usize {
        assert_eq!(y[e].to_bits(), (e as u32) << 23, "byte {e}");
    }
    // 2^-127 is an f32 subnormal: a GPU may flush it to zero in the product, as no checkpoint's scale comes near.
    assert!(y[0] == f32::from_bits(0x0040_0000) || y[0] == 0.0, "{}", y[0]);
}

#[test]
fn a_records_matrices_read_in_place_give_what_the_same_weights_uploaded_alone_give() {
    let Some(b) = backend() else { return };
    // Two matrices in one record, at word offsets, their scales after them; the same weights as Mxfp4 with f32
    // scales give the same sums, to the bit (the kernels add in the same order).
    let (n1, k1, n2, k2) = (96usize, 64usize, 64usize, 96usize);
    let mut next = rng(11);
    let w1: Vec<u8> = (0..n1 * k1 / 2).map(|_| (next() & 255) as u8).collect();
    let w2: Vec<u8> = (0..n2 * k2 / 2).map(|_| (next() & 255) as u8).collect();
    let s1: Vec<u8> = (0..n1 * k1 / 32).map(|_| 118 + (next() % 8) as u8).collect();
    let s2: Vec<u8> = (0..n2 * k2 / 32).map(|_| 118 + (next() % 8) as u8).collect();
    let record: Vec<u8> = [w1.as_slice(), &w2, &s1, &s2].concat();
    let (o2, os1, os2) = (w1.len(), w1.len() + w2.len(), w1.len() + w2.len() + s1.len());
    let slots = b.record_slots(2, record.len()).expect("room for two slots");
    assert_eq!(slots.len(), 2);
    slots.write(1, &record);
    let f32s = |s: &[u8]| s.iter().map(|&e| f32::from_bits((e as u32) << 23)).collect::<Vec<f32>>();
    let alone1 = b.dense(DenseData::Mxfp4 { w: w1.clone(), scales: f32s(&s1), n: n1, k: k1 }).unwrap().unwrap();
    let alone2 = b.dense(DenseData::Mxfp4 { w: w2.clone(), scales: f32s(&s2), n: n2, k: k2 }).unwrap().unwrap();
    let (m1, m2) = (slots.mxfp4(1, 0, os1, n1, k1), slots.mxfp4(1, o2, os2, n2, k2));
    for t in [1usize, 9, 70] {
        let (x1, x2) = (input(t, k1, t as u64), input(t, k2, t as u64 + 1));
        let got = forward_batch(&[(&m1, &x1, t, 0..n1), (&m2, &x2, t, 3..n2), (&m1, &x1, t, 7..20)]);
        assert_eq!(got[0], alone1.forward(&x1, t, 0..n1), "t={t}");
        assert_eq!(got[1], alone2.forward(&x2, t, 3..n2), "t={t}");
        assert_eq!(got[2], alone1.forward(&x1, t, 7..20), "t={t}");
    }
    let held = b.usage().0;
    drop(slots);
    assert_eq!(b.usage().0, held - 2 * record.len() as u64, "the slots' bytes come back to the budget");
}

/// What one dense call costs a decode step, and what of it is the trip itself: a weight the size of an attention
/// projection against one row, then the same trip with nothing to compute (a copy of 4 KB read back), with the
/// read-back's buffer kept, and waited for by polling.
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_a_round_trip() {
    let Some(b) = backend() else { return };
    let gpu = &b.gpu;
    let (n, k, calls) = (1024usize, 4096usize, 400u32);
    let w = b.dense(fp8_data(n, k, 5)).unwrap().expect("within the budget");
    let x = input(1, k, 6);
    let us = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e6 / calls as f64;
    for _ in 0..20 {
        w.forward(&x, 1, 0..n);
    }
    let t = std::time::Instant::now();
    for _ in 0..calls {
        std::hint::black_box(w.forward(&x, 1, 0..n));
    }
    eprintln!("a dense call of [{n}, {k}] against one row: {:.0} us", us(t));
    let two = [(&*w, &x[..], 1usize, 0..n), (&*w, &x[..], 1usize, 0..n)];
    let t = std::time::Instant::now();
    for _ in 0..calls {
        std::hint::black_box(forward_batch(&two));
    }
    eprintln!("two of them in one call: {:.0} us", us(t));
    // what a kernel reads a second: eight of a weight in one call against one, the seven more over their bytes
    for (name, data, bytes) in [
        ("fp8", fp8_data(4096, 5120, 7), 4096.0 * 5120.0),
        ("bf16", bf16_data(4096, 5120, 8), 4096.0 * 5120.0 * 2.0),
        ("mxfp4", mxfp4_data(4096, 5120, 9), 4096.0 * 5120.0 / 2.0),
    ] {
        let big = b.dense(data).unwrap().expect("within the budget");
        let xb = input(1, 5120, 10);
        let one = [(&*big, &xb[..], 1usize, 0..4096usize)];
        let eight: Vec<(&DenseGpu, &[f32], usize, Range<usize>)> = (0..8).map(|_| one[0].clone()).collect();
        let time = |items: &[(&DenseGpu, &[f32], usize, Range<usize>)]| {
            for _ in 0..10 {
                forward_batch(items);
            }
            let t = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(forward_batch(items));
            }
            t.elapsed().as_secs_f64() / 100.0
        };
        let (a, e) = (time(&one), time(&eight));
        eprintln!("{name} [4096, 5120]: one {:.0} us, eight in a call {:.0} us: {:.0} GB/s", a * 1e6, e * 1e6, 7.0 * bytes / (e - a) / 1e9);
        // as a decode step spaces its calls: the GPU idle between them (the host's work of a layer), by a busy
        // wait (the thread kept) or a sleep (the thread parked)
        for (how, pause_us, sleeps) in [("after 300 us of the host's work", 300u64, false), ("after 2 ms of it", 2000, false), ("after a 2 ms sleep", 2000, true)] {
            let mut spent = 0.0;
            for _ in 0..100 {
                let pause = std::time::Instant::now();
                if sleeps {
                    std::thread::sleep(std::time::Duration::from_micros(pause_us));
                } else {
                    while pause.elapsed() < std::time::Duration::from_micros(pause_us) {
                        std::hint::spin_loop();
                    }
                }
                let t = std::time::Instant::now();
                std::hint::black_box(forward_batch(&eight));
                spent += t.elapsed().as_secs_f64();
            }
            eprintln!("    eight in a call {how}: {:.0} us", spent / 100.0 * 1e6);
        }
    }
    let t = std::time::Instant::now();
    for _ in 0..calls {
        std::hint::black_box(upload_f32(gpu, &x));
    }
    eprintln!("the input's buffer made and written: {:.0} us", us(t));
    gpu.queue().submit([]);
    gpu.wait(None);
    let size = 4096u64;
    let src = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let staging = || gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let t = std::time::Instant::now();
    for _ in 0..calls {
        let st = staging();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &st, 0, size);
        gpu.queue().submit([enc.finish()]);
        std::hint::black_box(gpu.map_read(&st, size));
    }
    eprintln!("a trip with nothing to compute (4 KB copied and read back, its buffer made each time): {:.0} us", us(t));
    let kept = staging();
    let t = std::time::Instant::now();
    for _ in 0..calls {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
        gpu.queue().submit([enc.finish()]);
        std::hint::black_box(gpu.map_read(&kept, size));
    }
    eprintln!("the same with the read-back's buffer kept: {:.0} us", us(t));
    let t = std::time::Instant::now();
    for _ in 0..calls {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
        gpu.queue().submit([enc.finish()]);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        kept.slice(..size).map_async(wgpu::MapMode::Read, move |_| flag.store(true, std::sync::atomic::Ordering::Release));
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            std::hint::spin_loop();
        }
        std::hint::black_box(kept.slice(..size).get_mapped_range().expect("mapped").to_vec());
        kept.unmap();
    }
    eprintln!("the same, waited for by polling: {:.0} us", us(t));
    let t = std::time::Instant::now();
    for _ in 0..calls {
        gpu.queue().write_buffer(&src, 0, &[0u8; 4096]);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
        gpu.queue().submit([enc.finish()]);
        std::hint::black_box(gpu.map_read(&kept, size));
    }
    eprintln!("the kept trip with 4 KB written first: {:.0} us", us(t));
    // with the device holding what DeepSeek's does (a thousand records' buffers, written once): the same calls
    let record = 18_874_368u64;
    let held: Vec<wgpu::Buffer> = (0..1000)
        .map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: record, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
        .collect();
    gpu.queue().submit([]);
    gpu.wait(None);
    for _ in 0..20 {
        w.forward(&x, 1, 0..n);
    }
    let t = std::time::Instant::now();
    for _ in 0..calls {
        std::hint::black_box(w.forward(&x, 1, 0..n));
    }
    eprintln!("a dense call of [{n}, {k}] with a thousand records' buffers on the device: {:.0} us", us(t));
    let t = std::time::Instant::now();
    for _ in 0..calls {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &kept, 0, size);
        gpu.queue().submit([enc.finish()]);
        std::hint::black_box(gpu.map_read(&kept, size));
    }
    eprintln!("the kept trip then: {:.0} us", us(t));
    drop(held);
}

/// What an expert's record costs to put in its slot, by the way: 32 expert-sized writes (600 MB) into buffers made
/// once, queued one behind another and waited for together (as [`RecordSlots::write`] did: `write_buffer`, one
/// core's copy); the same each waited for; and each copied on every core and waited for, as it does now. Four
/// rounds of each.
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_the_upload_rate() {
    let Some(b) = backend() else { return };
    let gpu = &b.gpu;
    let size = 18_874_368u64;
    let host: Vec<u8> = (0..size as usize).map(|i| (i * 31 + i / 7) as u8).collect();
    let bufs: Vec<wgpu::Buffer> = (0..32)
        .map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }))
        .collect();
    for (way, what) in ["queued together, one core's copy", "each waited for, one core's copy", "each waited for, every core's copy"].iter().enumerate() {
        for round in 0..4 {
            let t = std::time::Instant::now();
            for buf in &bufs {
                match way {
                    0 => gpu.queue().write_buffer(buf, 0, &host),
                    1 => {
                        gpu.queue().write_buffer(buf, 0, &host);
                        gpu.queue().submit([]);
                        gpu.wait(None);
                    }
                    _ => {
                        gpu.write(buf, 0, &host);
                        gpu.queue().submit([]);
                        gpu.wait(None);
                    }
                }
            }
            gpu.queue().submit([]);
            gpu.wait(None);
            let secs = t.elapsed().as_secs_f64();
            eprintln!("{what}, round {round}: {:.0} MB in {secs:.3} s ({:.2} GB/s, {:.2} ms a record)", 32.0 * size as f64 / 1e6, 32.0 * size as f64 / secs / 1e9, secs * 1e3 / 32.0);
        }
        assert_eq!(gpu.read(&bufs[31], size), host, "{what}: the last buffer holds the record");
    }
}
