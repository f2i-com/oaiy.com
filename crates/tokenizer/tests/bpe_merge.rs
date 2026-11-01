//! Byte-level BPE's merge: the ids it gave when each step rescanned every pair, and how long a word it can take. A
//! request's text reaches it on the request thread, before any job there is to cancel: one word of 1 MB took that loop
//! hours.

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;
use std::time::Duration;

use gguf::value::{Array, Value};
use gguf::{GgmlType, GgufFile, TensorInfo};
use tokenizer::byte_encoder::{byte_to_char, bytes_to_visible};
use tokenizer::pretokenizer::split_llama3;
use tokenizer::Tokenizer;

/// GPT-2's first merges in their order, as its `merges.txt` has them: enough real overlap (`Ġ t`, `Ġt he`, `in g`,
/// `at ion`) to make the order they apply in matter.
const MERGES: &[(&str, &str)] = &[
    ("Ġ", "t"), ("Ġ", "a"), ("h", "e"), ("i", "n"), ("r", "e"), ("o", "n"), ("Ġt", "he"), ("e", "r"), ("Ġ", "s"), ("a", "t"),
    ("Ġ", "w"), ("Ġ", "o"), ("e", "n"), ("Ġ", "c"), ("i", "t"), ("i", "s"), ("a", "n"), ("o", "r"), ("e", "s"), ("Ġ", "b"),
    ("e", "d"), ("Ġ", "f"), ("in", "g"), ("Ġ", "p"), ("o", "u"), ("Ġa", "n"), ("a", "l"), ("a", "r"), ("Ġt", "o"), ("Ġ", "m"),
    ("Ġo", "f"), ("Ġ", "in"), ("Ġ", "d"), ("Ġ", "h"), ("Ġan", "d"), ("i", "c"), ("a", "s"), ("l", "e"), ("Ġt", "h"), ("i", "on"),
    ("o", "m"), ("l", "l"), ("en", "t"), ("Ġ", "n"), ("Ġ", "l"), ("s", "t"), ("Ġ", "re"), ("v", "e"), ("Ġ", "e"), ("r", "o"),
    ("l", "y"), ("Ġb", "e"), ("Ġ", "g"), ("Ġ", "T"), ("c", "t"), ("Ġ", "S"), ("i", "d"), ("o", "t"), ("Ġ", "I"), ("u", "t"),
    ("e", "t"), ("Ġ", "A"), ("Ġ", "is"), ("Ġ", "on"), ("i", "m"), ("a", "m"), ("o", "w"), ("a", "y"), ("a", "d"), ("s", "e"),
    ("Ġth", "at"), ("Ġ", "C"), ("i", "g"), ("Ġf", "or"), ("a", "c"), ("Ġ", "y"), ("v", "er"), ("u", "r"), ("Ġ", "u"), ("l", "d"),
    ("Ġs", "t"), ("Ġ", "M"), ("'", "s"), ("Ġ", "he"), ("Ġ", "it"), ("at", "ion"), ("it", "h"), ("i", "r"), ("c", "e"), ("Ġy", "ou"),
    ("i", "l"), ("Ġ", "B"), ("Ġw", "h"), ("o", "l"), ("Ġ", "P"), ("Ġw", "ith"), ("Ġ", "1"), ("t", "er"), ("c", "h"), ("Ġa", "s"),
    ("Ġw", "e"), ("Ġ", "("), ("n", "d"), ("i", "ll"), ("Ġ", "D"), ("i", "e"), ("Ġ", "2"), ("a", "g"), ("Ġc", "on"), ("Ġ", "H"),
    ("u", "s"), ("Ġ", "r"), ("Ġp", "ro"), ("Ġh", "a"), ("Ġ", "v"), ("Ġ", "an"), ("p", "e"), ("Ġ", "F"), ("Ġ", "W"), ("o", "o"),
];

/// A tokenizer of every byte, the merges and what they make, and `specials` (type 3, control) at the end.
fn tokenizer(merges: &[(&str, &str)], specials: &[&str]) -> Tokenizer {
    let mut tokens: Vec<String> = (0u8..=255).map(|b| byte_to_char(b).to_string()).collect();
    for (l, r) in merges {
        let merged = format!("{l}{r}");
        if !tokens.contains(&merged) {
            tokens.push(merged);
        }
    }
    let mut types = vec![1i32; tokens.len()];
    for s in specials {
        tokens.push((*s).to_string());
        types.push(3);
    }
    let mut md: BTreeMap<String, Value> = BTreeMap::new();
    md.insert("general.architecture".into(), Value::String("test".into()));
    md.insert("tokenizer.ggml.model".into(), Value::String("gpt2".into()));
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(tokens)));
    md.insert("tokenizer.ggml.token_type".into(), Value::Array(Array::I32(types)));
    md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(merges.iter().map(|(l, r)| format!("{l} {r}")).collect())));
    let info = TensorInfo { name: "noop".into(), shape: vec![1], dtype: GgmlType::F32, offset: 0 };
    Tokenizer::from_gguf(&GgufFile::from_bytes(gguf::reader::write_to_vec(&md, &[(info, vec![0u8; 4])], 32).unwrap()).unwrap()).unwrap()
}

/// The merge as it was: every pair of the piece looked at for each one merged, the lowest rank first and of equal ones
/// the leftmost. What the heap has to agree with.
fn rescanning(tok: &Tokenizer, ranks: &HashMap<(String, String), u32>, text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for piece in split_llama3(text) {
        let mut tokens: Vec<String> = bytes_to_visible(piece.as_bytes()).chars().map(|c| c.to_string()).collect();
        loop {
            let (mut best_rank, mut best_idx) = (u32::MAX, None);
            for i in 0..tokens.len().saturating_sub(1) {
                if let Some(&r) = ranks.get(&(tokens[i].clone(), tokens[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(i) = best_idx else { break };
            tokens[i] = format!("{}{}", tokens[i], tokens[i + 1]);
            tokens.remove(i + 1);
        }
        out.extend(tokens.iter().map(|t| tok.token_id(t).expect("every merge's result is in the vocabulary")));
    }
    out
}

fn ranks(merges: &[(&str, &str)]) -> HashMap<(String, String), u32> {
    merges.iter().enumerate().map(|(i, (l, r))| ((l.to_string(), r.to_string()), i as u32)).collect()
}

/// Ids written down from the loop this replaced, run on these strings before it went.
const WRITTEN_DOWN: &[(&str, &[u32])] = &[
    ("The quick brown fox jumps over the lazy dog.", &[84, 258, 32, 113, 117, 291, 107, 275, 305, 119, 110, 277, 111, 120, 32, 106, 117, 109, 112, 115, 267, 332, 262, 300, 97, 122, 121, 288, 111, 103, 46]),
    ("It is the end of the beginning, and we will stand with them for a good reason.", &[73, 116, 318, 262, 32, 268, 100, 286, 262, 307, 103, 259, 110, 278, 44, 290, 356, 266, 359, 336, 272, 100, 351, 262, 109, 329, 257, 308, 374, 100, 302, 292, 261, 46]),
    ("   indented    text\twith\n\nmixed  whitespace\r\n", &[32, 32, 287, 100, 298, 276, 32, 32, 32, 256, 101, 120, 116, 9, 119, 342, 10, 10, 109, 105, 120, 276, 32, 348, 270, 274, 112, 330, 101, 13, 10]),
    ("don't, won't, it's: he'll say they've gone; I'm sure we'd go", &[100, 261, 39, 116, 44, 266, 261, 39, 116, 44, 340, 338, 58, 339, 39, 297, 264, 323, 262, 121, 39, 303, 308, 261, 101, 59, 314, 39, 109, 264, 117, 260, 356, 39, 100, 308, 111]),
    ("naïve café — 世界, 12345 and 3.14", &[110, 97, 195, 175, 303, 269, 97, 102, 195, 169, 32, 226, 128, 148, 32, 228, 184, 150, 231, 149, 140, 44, 32, 49, 50, 51, 52, 53, 290, 32, 51, 46, 49, 52]),
    ("fn main() { println!(\"hello, {}\", 1 + 2); }", &[102, 110, 285, 97, 259, 40, 41, 32, 123, 279, 114, 259, 116, 108, 110, 33, 40, 34, 258, 297, 111, 44, 32, 123, 125, 34, 44, 32, 49, 32, 43, 32, 50, 41, 59, 32, 125]),
];

#[test]
fn the_ids_are_the_ones_the_rescanning_loop_gave() {
    let tok = tokenizer(MERGES, &[]);
    for (text, ids) in WRITTEN_DOWN {
        assert_eq!(tok.encode(text, false).unwrap(), *ids, "{text:?}");
    }
}

/// The two on text that pushes at the order: runs of a letter (equal pairs, one after another, whose leftmost goes
/// first), pairs that form only after others have merged, and characters of several bytes.
#[test]
fn the_merge_agrees_with_the_rescanning_loop_on_random_text() {
    let tok = tokenizer(MERGES, &[]);
    let ranks = ranks(MERGES);
    let alphabet: Vec<char> = "the in at ion er ing oo ll ss aaa tt   \n'.,1(é世".chars().collect();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    for _ in 0..3000 {
        let len = rand() % 120;
        let text: String = (0..len).map(|_| alphabet[rand() % alphabet.len()]).collect();
        assert_eq!(tok.encode(&text, false).unwrap(), rescanning(&tok, &ranks, &text), "{text:?}");
    }
}

/// Merges whose result is no token are still made, and what is left over is out of the vocabulary, as it was.
#[test]
fn a_merge_out_of_the_vocabulary_is_still_an_error_with_no_unk() {
    let tok = tokenizer(&[("a", "b")], &[]);
    assert_eq!(tok.encode("ab", false).unwrap().len(), 1);
    let mut tokens: Vec<String> = (0u8..=255).map(|b| byte_to_char(b).to_string()).collect();
    tokens.push("unused".into());
    let mut md: BTreeMap<String, Value> = BTreeMap::new();
    md.insert("general.architecture".into(), Value::String("test".into()));
    md.insert("tokenizer.ggml.model".into(), Value::String("gpt2".into()));
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(tokens)));
    md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(vec!["a b".into()])));
    let info = TensorInfo { name: "noop".into(), shape: vec![1], dtype: GgmlType::F32, offset: 0 };
    let tok = Tokenizer::from_gguf(&GgufFile::from_bytes(gguf::reader::write_to_vec(&md, &[(info, vec![0u8; 4])], 32).unwrap()).unwrap()).unwrap();
    assert!(tok.encode("ab", false).is_err());
    assert_eq!(tok.encode("ba", false).unwrap().len(), 2, "not merged: both are tokens");
}

/// `tok.encode(text)`, given `within` to finish: a loop that takes hours must fail this test, not hang it.
fn encoded_within(tok: Tokenizer, text: String, within: Duration) -> Vec<u32> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(tok.encode(&text, false).unwrap());
    });
    rx.recv_timeout(within).expect("the text is still being encoded")
}

#[test]
fn one_long_word_is_encoded_in_good_time() {
    // (each doubling a merge: every step of the old loop had hundreds of thousands of pairs to look at)
    const DOUBLING: &[(&str, &str)] = &[("a", "a"), ("aa", "aa"), ("aaaa", "aaaa"), ("aaaaaaaa", "aaaaaaaa")];
    let n = 1 << 18;
    let ids = encoded_within(tokenizer(DOUBLING, &[]), "a".repeat(n), Duration::from_secs(60));
    assert_eq!(ids.len(), n / 16);
    assert!(ids.iter().all(|&id| id == ids[0]));
}

#[test]
fn a_word_past_the_cap_is_still_all_there() {
    let tok = tokenizer(&[("a", "a")], &[]);
    let n = (1 << 20) + 3;
    let ids = encoded_within(tok.clone(), "a".repeat(n), Duration::from_secs(60));
    assert_eq!(tok.decode(&ids), "a".repeat(n));
}

/// Each special found looked for every special again, to the end of the text for those that are not in it.
#[test]
fn a_prompt_full_of_special_tokens_is_cut_in_good_time() {
    let specials: Vec<String> = (0..64).map(|i| format!("<|absent_{i}|>")).chain(["<|x|>".to_string()]).collect();
    let refs: Vec<&str> = specials.iter().map(String::as_str).collect();
    let n = 200_000;
    let tok = tokenizer(&[], &refs);
    let x = tok.token_id("<|x|>").unwrap();
    let ids = encoded_within(tok, format!("{}z", "<|x|>".repeat(n)), Duration::from_secs(60));
    assert_eq!(ids.len(), n + 1);
    assert!(ids[..n].iter().all(|&id| id == x));
}

/// Of specials at the same place the longest, and between them text, as ever.
#[test]
fn specials_are_still_cut_earliest_and_longest_first() {
    let tok = tokenizer(&[], &["<|a|>", "<|a|><|b|>", "<|b|>"]);
    let id = |s: &str| tok.token_id(s).unwrap();
    let ids = tok.encode("x<|b|>y<|a|><|b|><|a|>", false).unwrap();
    let byte = |c: char| tok.token_id(&c.to_string()).unwrap();
    assert_eq!(ids, [byte('x'), id("<|b|>"), byte('y'), id("<|a|><|b|>"), id("<|a|>")]);
}
