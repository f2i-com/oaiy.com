//! `device`'s tests.

use super::*;
use super::super::forward::matvec;
use ggml_quants::q4_k;

use crate::glm5next::test_paths::released;

/// The seam itself: a `Mat::Device` over a dense `Weight` must give the same
/// numbers as `Mat::Host` over the same floats. This is what lets one forward
/// implementation serve both paths, so it is worth pinning without a model.
#[test]
fn device_and_host_matrices_agree() {
    let backend = ggml_rs::default_backend();
    let (out_dim, in_dim) = (7usize, 5usize);
    let raw: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 11.0)
        .collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 2.0) / 3.0).collect();

    let host = Mat::Host(&raw);
    let mut a = vec![0.0f32; out_dim];
    host.apply(&x, &mut a).expect("host");

    let w = Weight::Dense(
        backend.to_device(Tensor::from_vec(raw.clone(), vec![out_dim, in_dim])),
    );
    let dev = Mat::Device {
        w: &w,
        backend: &*backend,
    };
    let mut b = vec![0.0f32; out_dim];
    dev.apply(&x, &mut b).expect("device");

    for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (p - q).abs() < 1e-5,
            "row {i}: host {p} vs device {q}"
        );
    }
    // And it is not trivially zero.
    assert!(a.iter().any(|&v| v.abs() > 1e-6));
}

// --- CUDA -------------------------------------------------------------
// These are the same checks as above, on a real device. They skip rather
// than fail when no GPU is reachable, matching how ggml-rs-cuda gates its
// own tests.

/// What one expert record costs on the CPU today, at the released shapes.
///
/// A routed expert is gate and up of `[n_ff_exp, n_embd]` and down of
/// `[n_embd, n_ff_exp]`, all Q4_K: 4 718 592 bytes each, 8 388 608 weights
/// each. The CPU tier of the expert hierarchy only pays off if this beats a
/// PCIe upload of the same record, which the DeepSeek notes measured at
/// 1.3 ms over x4 and 2.6 ms over x2.
///
/// Synthetic bytes: Q4_K dequantisation is data-independent, and the
/// arithmetic and the memory traffic are what is being timed.
#[test]
#[ignore = "measures the host expert path"]
fn measure_host_record_cost() {
    let (n_embd, n_ff) = (4096usize, 2048usize);
    let per = n_embd * n_ff / 256 * 144;
    let raw: Vec<u8> = (0..per).map(|i| (i * 31 % 251) as u8).collect();
    let x: Vec<f32> = (0..n_embd).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let mut w = vec![0.0f32; n_embd * n_ff];
    let mut h = vec![0.0f32; n_ff];
    let mut out = vec![0.0f32; n_embd];

    // one pass to fault the buffers in
    q4_k::dequantize(&raw, &mut w);
    matvec(&w, &x, &mut h).expect("warm");
    std::hint::black_box(&h);

    let n = 5usize;
    let (mut deq, mut mv) = (0.0f64, 0.0f64);
    for _ in 0..n {
        let t = std::time::Instant::now();
        for _ in 0..3 {
            q4_k::dequantize(&raw, &mut w);
        }
        deq += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        matvec(&w, &x, &mut h).expect("gate");
        std::hint::black_box(&h);
        matvec(&w, &x, &mut h).expect("up");
        std::hint::black_box(&h);
        matvec(&w, &h, &mut out).expect("down");
        std::hint::black_box(&out);
        mv += t.elapsed().as_secs_f64();
    }
    let (deq, mv) = (deq / n as f64, mv / n as f64);
    let rec = 3 * per;
    println!();
    println!("one expert record, {:.2} MB, 25.2 M weights, single thread:", rec as f64 / 1e6);
    println!("  dequantise      {:8.2} ms   ({:.1} GB/s of Q4_K in)", deq * 1e3, rec as f64 / deq / 1e9);
    println!("  three matvecs   {:8.2} ms", mv * 1e3);
    println!("  total           {:8.2} ms", (deq + mv) * 1e3);
    println!("  over 32 threads {:8.2} ms   (perfect scaling, which it will not get)", (deq + mv) * 1e3 / 32.0);
    println!();
    println!("The f32 detour is most of it: dequantising writes {:.0} MB and the", 3.0 * (n_embd * n_ff) as f64 * 4.0 / 1e6);
    println!("matvecs read it back. A fused Q4_K dot would touch the {:.1} MB once.", rec as f64 / 1e6);
    println!("to beat: a PCIe upload of the same record, 1.3 ms over x4");
    println!("budget: 20 tok/s over 42 layers is 1.19 ms a layer, for every");
    println!("routed expert of that layer not already resident in VRAM");
    assert!(deq > 0.0 && mv > 0.0);
}

/// The sampling the model itself recommends, read out of the file.
///
/// GLM-5.3-Flash ships `general.sampling.temp = 1.0` and
/// `general.sampling.top_p = 0.95`. This pins that we use those rather than a
/// number someone picked -- including me, who lowered the temperature to 0.6 on a
/// hunch and then found the file had said 1.0 all along.
#[test]
#[ignore = "needs the released model on disk"]
fn sampling_comes_from_the_model() {
    let g = GgufFile::open_streaming(released()).expect("open");
    let p = crate::sampler::SampleParams::from_gguf(&g);
    println!();
    println!("general.sampling.temp  -> temperature {}", p.temperature);
    println!("general.sampling.top_p -> top_p       {:?}", p.top_p);
    println!("top_k {:?}, min_p {:?}, repeat {:?}", p.top_k, p.min_p, p.repeat_penalty);
    assert!(
        (p.temperature - 1.0).abs() < 1e-6,
        "expected the file temp of 1.0, got {}",
        p.temperature
    );
    assert!(
        p.top_p.is_some_and(|v| (v - 0.95).abs() < 1e-3),
        "expected the file top_p of 0.95, got {:?}",
        p.top_p
    );
    // Nothing records a repetition penalty, so it must stay off.
    assert!(p.repeat_penalty.is_none(), "a penalty appeared from nowhere");
}

/// What the chat template actually hands the model.
///
/// Printed rather than asserted, because a generation that ignored its prompt
/// while raw token ids clearly did not is most easily explained by the prompt
/// never containing the question.
#[test]
#[ignore = "needs the released model on disk (for the tokenizer)"]
fn show_the_templated_prompt() {
    use crate::chat::{apply_chat_template, ChatMessage, Role};
    use crate::config::Architecture;

    let g = GgufFile::open_streaming(released()).expect("open");
    let tok = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");

    for content in [
        "What is the capital of France? Answer in one short sentence.",
        "Write a Python function to sort a list.",
    ] {
        let msgs = [ChatMessage { role: Role::User, content: content.to_string() }];
        let p = apply_chat_template(&Architecture::Glm5Next, &msgs, true);
        let ids = tok.encode(&p, false).expect("encode");
        println!();
        println!("content: {content:?}");
        println!("template ({} chars): {p:?}", p.len());
        println!("ids ({}): {:?}", ids.len(), ids);
        println!("round trip: {:?}", tok.decode(&ids));
        assert!(
            p.contains("France") || p.contains("Python"),
            "the templated prompt does not contain the user content at all"
        );
    }
}

/// The released model, with its matrices on the backend. On a CUDA build this
/// is the GPU path; on a CPU build it exercises the same plumbing.
#[test]
#[ignore = "needs the released model on disk"]
fn released_model_runs_on_the_backend() {
    let backend = ggml_rs::default_backend();
    let t0 = std::time::Instant::now();
    let m = DeviceModel::open(released(), 512, backend).expect("load");
    println!(
        "device load in {:.1}s, backend {}",
        t0.elapsed().as_secs_f64(),
        m.backend_name()
    );

    let sh = m.shape().clone();
    assert_eq!(sh.n_layer, 45);
    let w = m.view();
    let mut st = forward::State::new_on(&sh, m.backend()).expect("state");

    let t1 = std::time::Instant::now();
    let logits = forward::forward_token(&sh, &w, &mut st, 154822).expect("forward");
    println!("one token in {:.1}s", t1.elapsed().as_secs_f64());

    assert_eq!(logits.len(), sh.n_vocab);
    assert!(logits.iter().all(|x| x.is_finite()));
    let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    println!("argmax {bi} logit {best:.4}");
    assert!(logits.iter().any(|&v| v != 0.0));
}

/// The device path must agree with the host reference on the real model.
/// This is the equivalence gate for the backend seam.
#[test]
#[ignore = "needs the released model on disk; loads it twice"]
fn device_agrees_with_host_on_the_real_model() {
    use super::super::bridge::HostModel;

    let hm = HostModel::open(released(), 512).expect("host load");
    let hs = hm.shape().clone();
    let hv = hm.view();
    let mut h_state = forward::State::new(&hs).expect("state");
    let h = forward::forward_token(&hs, &hv, &mut h_state, 154822).expect("host forward");
    drop(hv);
    drop(hm);

    let dm = DeviceModel::open(released(), 512, ggml_rs::default_backend()).expect("dev load");
    let ds = dm.shape().clone();
    let dv = dm.view();
    // On the CPU backend this pits `Backend::kda_delta_step`'s host default
    // against `kda::step`, which is the other half of the recurrence check.
    let mut d_state = forward::State::new_on(&ds, dm.backend()).expect("state");
    let d = forward::forward_token(&ds, &dv, &mut d_state, 154822).expect("device forward");

    assert_eq!(h.len(), d.len());
    let mut worst = 0.0f32;
    for (a, b) in h.iter().zip(d.iter()) {
        worst = worst.max((a - b).abs());
    }
    println!("max |host - device| over {} logits: {worst:.6}", h.len());
    // Both paths dequantise the same bytes; the only differences are
    // accumulation order inside the matmul.
    assert!(worst < 0.05, "device and host disagree by {worst}");
}
