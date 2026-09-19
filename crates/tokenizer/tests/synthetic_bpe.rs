//! Build a tiny GGUF-in-memory with a hand-crafted vocab + merges, then check
//! that BPE encode/decode round-trips correctly.

use std::collections::BTreeMap;

use gguf::value::{Array, Value};
use gguf::{GgmlType, GgufFile, TensorInfo};
use tokenizer::byte_encoder::byte_to_char;
use tokenizer::Tokenizer;

fn make_gguf(model: &str, tokens: Vec<String>, merges: Vec<String>) -> Vec<u8> {
    let mut md: BTreeMap<String, Value> = BTreeMap::new();
    md.insert("general.architecture".into(), Value::String("test".into()));
    md.insert("tokenizer.ggml.model".into(), Value::String(model.into()));
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(tokens)));
    md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(merges)));
    // Need at least one tensor for the file to be valid; an empty F32.
    let info = TensorInfo {
        name: "noop".into(),
        shape: vec![1],
        dtype: GgmlType::F32,
        offset: 0,
    };
    let data = vec![0u8; 4];
    gguf::reader::write_to_vec(&md, &[(info, data)], 32).unwrap()
}

#[test]
fn bpe_encodes_via_simple_merges() {
    // A 4-byte word "abcd". Build vocab so that all single bytes are tokens,
    // plus a few merges.
    let mut tokens = Vec::new();
    for b in 0u8..=255u8 { tokens.push(byte_to_char(b).to_string()); }
    // Add merged tokens at the end so vocab index is unique.
    let ab: String = format!("{}{}", byte_to_char(b'a'), byte_to_char(b'b')); // "ab"
    let cd: String = format!("{}{}", byte_to_char(b'c'), byte_to_char(b'd')); // "cd"
    let abcd: String = format!("{}{}", ab, cd);                                // "abcd"
    tokens.push(ab.clone());
    tokens.push(cd.clone());
    tokens.push(abcd.clone());

    // Merges, low-rank first.
    let merges = vec![
        format!("{} {}", byte_to_char(b'a'), byte_to_char(b'b')), // a b
        format!("{} {}", byte_to_char(b'c'), byte_to_char(b'd')), // c d
        format!("{} {}", ab, cd),                                  // ab cd
    ];

    let bytes = make_gguf("gpt2", tokens.clone(), merges);
    let f = GgufFile::from_bytes(bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();

    let ids = tok.encode("abcd", false).unwrap();

    // After BPE: a+b -> ab, c+d -> cd, ab+cd -> abcd. Result is one token.
    assert_eq!(ids.len(), 1, "expected single merged token, got {ids:?}");
    let tok_str = tok.token(ids[0]).unwrap();
    assert_eq!(tok_str, abcd);

    // Decode round-trips.
    let text = tok.decode(&ids);
    assert_eq!(text, "abcd");
}

#[test]
fn bpe_with_no_merges_emits_per_byte() {
    let mut tokens = Vec::new();
    for b in 0u8..=255u8 { tokens.push(byte_to_char(b).to_string()); }
    let bytes = make_gguf("gpt2", tokens, vec![]);
    let f = GgufFile::from_bytes(bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();

    let ids = tok.encode("hi", false).unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(tok.decode(&ids), "hi");
}

#[test]
fn bpe_decode_arbitrary_bytes() {
    let mut tokens = Vec::new();
    for b in 0u8..=255u8 { tokens.push(byte_to_char(b).to_string()); }
    let bytes = make_gguf("gpt2", tokens, vec![]);
    let f = GgufFile::from_bytes(bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();

    let original = "Hello, 世界! 👋";
    let ids = tok.encode(original, false).unwrap();
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, original);
}

#[test]
fn spm_greedy_longest_match() {
    // Synthetic SPM vocab: ▁hello, ▁world, h, e, l, o, w, r, d, ▁, plus byte fallbacks.
    let tokens: Vec<String> = vec![
        "▁hello".into(),
        "▁world".into(),
        "▁".into(),
        "h".into(), "e".into(), "l".into(), "o".into(),
        "w".into(), "r".into(), "d".into(),
    ]
    .into_iter()
    .chain((0u8..=255).map(|b| format!("<0x{:02X}>", b)))
    .collect();

    let bytes = make_gguf("llama", tokens, vec![]);
    let f = GgufFile::from_bytes(bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();

    // SPM encoder prefixes with ▁; "hello world" -> ▁hello, ▁world (length 2).
    let ids = tok.encode("hello world", false).unwrap();
    assert_eq!(ids.len(), 2, "expected 2 tokens, got {} : {:?}",
               ids.len(),
               ids.iter().map(|id| tok.token(*id).unwrap_or("?")).collect::<Vec<_>>());
    assert_eq!(tok.token(ids[0]).unwrap(), "▁hello");
    assert_eq!(tok.token(ids[1]).unwrap(), "▁world");

    // Note: SPM decode preserves the leading SPM space marker ▁ (which
    // becomes " "). Caller is responsible for trimming when doing
    // full-sequence decoding. For streaming token-at-a-time decoding, the
    // leading space is exactly the inter-word space.
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, " hello world");
}
