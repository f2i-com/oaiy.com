//! A chunk of tokens continuing a sequence gives what decoding it a token at a time gives: on the real checkpoint
//! (`DSV41_MODEL`), a prompt, then 150 more tokens in one forward against the same 150 a token at a time (past the
//! 128-token window, from a position no compression ratio divides). `#[ignore]`d; run it with
//! `cargo test --release -p dsv41 --test chunk_continuation -- --ignored --nocapture`.
use dsv41::model::{argmax, Model, ModelOptions};
use std::path::PathBuf;

#[test]
#[ignore = "loads the 510 GB checkpoint twice and decodes 150 tokens on the CPU"]
fn a_chunk_continuing_a_sequence_is_read_as_its_tokens_one_at_a_time_are() {
    let dir = std::env::var_os("DSV41_MODEL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"D:\deepseek\model"));
    let meta = dir.join("engram_meta.safetensors");
    let tok = dsv41::tokenizer::Tokenizer::load(&dir).unwrap();
    let docs = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs");
    let text = std::fs::read_to_string(format!("{docs}/WEBGPU.md")).unwrap();
    let mut ids = vec![0u32];
    ids.extend(tok.encode(&text));
    let (head, chunk) = (70usize, 150usize);
    assert!(ids.len() > head + chunk + 1);
    let opts = ModelOptions { max_seq: 512, expert_cache_bytes: 24 << 30, direct_io: true };

    let mut chunked = Model::load(&dir, &meta, &opts).unwrap();
    chunked.forward(&ids[..head], 0, &mut |_, _| {}).unwrap();
    let a = chunked.forward(&ids[head..head + chunk], head, &mut |_, _| {}).unwrap();

    let mut stepped = Model::load(&dir, &meta, &opts).unwrap();
    stepped.forward(&ids[..head], 0, &mut |_, _| {}).unwrap();
    let mut b = Vec::new();
    for (i, &id) in ids[head..head + chunk].iter().enumerate() {
        b = stepped.forward(&[id], head + i, &mut |_, _| {}).unwrap();
    }

    let report = |what: &str, a: &[f32], b: &[f32]| {
        let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(a) * norm(b));
        let exact = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        eprintln!("{what}: exact {exact}, cosine {cosine:.8}, largest difference {diff:.6}, argmax {} / {}", argmax(a), argmax(b));
        (cosine, exact)
    };
    let (cosine, _) = report("after the chunk", &a, &b);
    assert_eq!(argmax(&a), argmax(&b));
    assert!(cosine > 0.99999, "cosine {cosine}");
    // And the states it left: the next token decodes the same from both.
    let next = ids[head + chunk];
    let a2 = chunked.forward(&[next], head + chunk, &mut |_, _| {}).unwrap();
    let b2 = stepped.forward(&[next], head + chunk, &mut |_, _| {}).unwrap();
    let (cosine, _) = report("a step after it", &a2, &b2);
    assert_eq!(argmax(&a2), argmax(&b2));
    assert!(cosine > 0.99999, "cosine {cosine}");
}
