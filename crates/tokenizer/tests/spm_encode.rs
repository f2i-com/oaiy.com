//! The SentencePiece encoder as llama.cpp encodes: special tokens (USER_DEFINED ones too, such as Gemma's newlines)
//! split out first, a leading space only where the vocabulary asks for one, then pairs merged by score (by merge rank
//! for Gemma 4's `"gemma4"`).

use std::collections::BTreeMap;

use gguf::value::{Array, Value};
use gguf::{GgmlType, GgufFile, TensorInfo};
use tokenizer::Tokenizer;

/// An in-memory SPM vocabulary: `(piece, score, type)` (1 normal, 3 control, 4 user-defined, 6 byte), and
/// `add_space_prefix` when given.
fn spm(pieces: &[(&str, f32, i32)], add_space_prefix: Option<bool>) -> Tokenizer {
    spm_model("llama", pieces, &[], add_space_prefix)
}

/// [`spm`] for another `tokenizer.ggml.model`, with merges.
fn spm_model(model: &str, pieces: &[(&str, f32, i32)], merges: &[&str], add_space_prefix: Option<bool>) -> Tokenizer {
    let mut tokens: Vec<String> = vec!["<unk>".into()];
    let mut scores = vec![0.0f32];
    let mut types = vec![2i32];
    for b in 0u8..=255 {
        tokens.push(format!("<0x{b:02X}>"));
        scores.push(0.0);
        types.push(6);
    }
    for (p, s, t) in pieces {
        tokens.push(p.to_string());
        scores.push(*s);
        types.push(*t);
    }
    let mut md: BTreeMap<String, Value> = BTreeMap::new();
    md.insert("general.architecture".into(), Value::String("test".into()));
    md.insert("tokenizer.ggml.model".into(), Value::String(model.into()));
    if !merges.is_empty() {
        md.insert("tokenizer.ggml.merges".into(), Value::Array(Array::String(merges.iter().map(|m| m.to_string()).collect())));
    }
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Array::String(tokens)));
    md.insert("tokenizer.ggml.scores".into(), Value::Array(Array::F32(scores)));
    md.insert("tokenizer.ggml.token_type".into(), Value::Array(Array::I32(types)));
    md.insert("tokenizer.ggml.unknown_token_id".into(), Value::U32(0));
    if let Some(a) = add_space_prefix {
        md.insert("tokenizer.ggml.add_space_prefix".into(), Value::Bool(a));
    }
    let info = TensorInfo { name: "noop".into(), shape: vec![1], dtype: GgmlType::F32, offset: 0 };
    let bytes = gguf::reader::write_to_vec(&md, &[(info, vec![0u8; 4])], 32).unwrap();
    Tokenizer::from_gguf(&GgufFile::from_bytes(bytes).unwrap()).unwrap()
}

fn pieces(tok: &Tokenizer, ids: &[u32]) -> Vec<String> {
    ids.iter().map(|&i| tok.token(i).unwrap().to_string()).collect()
}

const VOCAB: &[(&str, f32, i32)] = &[
    ("<start_of_turn>", 0.0, 3),
    ("\n", 0.0, 4),
    ("\n\n", 0.0, 4),
    ("  ", 0.0, 4),
    ("u", -20.0, 1),
    ("s", -20.0, 1),
    ("e", -20.0, 1),
    ("r", -20.0, 1),
    ("t", -20.0, 1),
    ("h", -20.0, 1),
    ("c", -20.0, 1),
    ("a", -20.0, 1),
    ("▁", -20.0, 1),
    ("us", -9.0, 1),
    ("er", -9.0, 1),
    ("user", -3.0, 1),
    ("▁user", -2.0, 1),
    ("th", -9.0, 1),
    ("the", -4.0, 1),
    ("▁t", -8.0, 1),
    ("▁th", -7.0, 1),
    ("▁the", -1.0, 1),
    ("ca", -9.0, 1),
    ("cat", -5.0, 1),
    ("▁c", -8.0, 1),
    ("▁ca", -7.0, 1),
    ("▁cat", -1.0, 1),
];

#[test]
fn a_user_defined_newline_is_one_token_and_the_line_after_it_takes_no_space() {
    // Gemma's GGUFs say add_space_prefix = false and mark "\n" USER_DEFINED: the text after a newline (or a chat
    // marker) is a line's start, not a word's. A `▁` there made Gemma 3 answer markdown prompts with fragments of them.
    let tok = spm(VOCAB, Some(false));
    let ids = tok.encode("<start_of_turn>user\nthe cat\n\nthe  cat", false).unwrap();
    assert_eq!(pieces(&tok, &ids), ["<start_of_turn>", "user", "\n", "the", "▁cat", "\n\n", "the", "  ", "cat"]);
    // And they decode as the text they are, the newlines too.
    assert_eq!(tok.decode(&ids), "user\nthe cat\n\nthe  cat");
}

#[test]
fn a_vocabulary_that_does_not_say_takes_the_leading_space_after_specials() {
    // llama.cpp's default for SentencePiece (Llama 1/2): a space before the text, and after each special token.
    let tok = spm(VOCAB, None);
    let ids = tok.encode("<start_of_turn>user\nthe cat", false).unwrap();
    assert_eq!(pieces(&tok, &ids), ["<start_of_turn>", "▁user", "\n", "▁the", "▁cat"]);
}

#[test]
fn pairs_merge_by_score_not_by_length() {
    // "abc": greedy longest-match from the left takes "ab" then "c"; SentencePiece merges the best-scoring pair first,
    // "bc", and "a" + "bc" cannot merge further.
    let tok = spm(&[("a", -20.0, 1), ("b", -20.0, 1), ("c", -20.0, 1), ("ab", -5.0, 1), ("bc", -1.0, 1)], Some(false));
    let ids = tok.encode("abc", false).unwrap();
    assert_eq!(pieces(&tok, &ids), ["a", "bc"]);
    // A character that is no piece falls back to its bytes.
    let ids = tok.encode("aé", false).unwrap();
    assert_eq!(pieces(&tok, &ids), ["a", "<0xC3>", "<0xA9>"]);
}

#[test]
fn gemma4_merges_by_rank() {
    // Gemma 4's pieces all score the same and its merges carry the order: "b c" is listed first, so "abc" is "a" +
    // "bc", where merging by score (all equal, leftmost first) would give "ab" + "c".
    let flat = [("a", -1000.0, 1), ("b", -1000.0, 1), ("c", -1000.0, 1), ("ab", -1000.0, 1), ("bc", -1000.0, 1)];
    let tok = spm_model("gemma4", &flat, &["b c", "a b"], Some(false));
    let ids = tok.encode("abc", false).unwrap();
    assert_eq!(pieces(&tok, &ids), ["a", "bc"]);
    // Only listed pairs merge, whatever the vocabulary holds.
    let tok = spm_model("gemma4", &flat, &["a b"], Some(false));
    let ids = tok.encode("abc", false).unwrap();
    assert_eq!(pieces(&tok, &ids), ["ab", "c"]);
}

/// A piece of markdown that Gemma 3 4B answered with fragments of, and the ids llama.cpp's encoder gives it under
/// Gemma 3's and Gemma 4's vocabularies (the same ids: their pieces share them). Checked against llama.cpp's own
/// server's `/tokenize` on these and longer texts (a 44,592-token one among them).
const EXCERPT: &str = "without a folder (`\"server\": \"oaiy-llm-server\"`) are found beside\n`oaiy-studio`, then beside the configuration, then on `PATH`. Relative paths in\nthe configuration resolve against the configuration's folder. Generated media\ngoes to `outputs/` there, and prompt states to `cache/`. Copy the folder to\nanother machine and it runs there.\n\n```sh\noaiy-studio                      # opens the UI in a browser window without tabs\noaiy-studio --open browser       # a normal browser tab\noaiy-studio --headless           # console only; the UI still serves on its port\noaiy-studio --config D:/ai/studio.json -";
const WANT: [u32; 177] = [
        2, 105, 2364, 107, 31048, 496, 11709, 20442, 236775, 6458, 1083, 623, 236748, 1389, 236762, 236772, 859, 236757,
        236772, 6458, 236775, 18833, 659, 1765, 28383, 107, 236929, 236748, 1389, 236762, 236772, 45134, 8347, 1299,
        28383, 506, 9831, 236764, 1299, 580, 2165, 15573, 21233, 58162, 16412, 528, 107, 1437, 9831, 14245, 2342, 506,
        9831, 236789, 236751, 11709, 236761, 60257, 4009, 107, 166219, 531, 2165, 52256, 236786, 236929, 993, 236764, 532,
        11172, 5022, 531, 2165, 12655, 236786, 21233, 11371, 506, 11709, 531, 107, 51208, 5464, 532, 625, 8784, 993,
        236761, 108, 2717, 1179, 107, 236748, 1389, 236762, 236772, 45134, 158, 236865, 19015, 506, 9711, 528, 496, 11180,
        4771, 2180, 42364, 107, 236748, 1389, 236762, 236772, 45134, 2617, 5265, 11180, 143, 236865, 496, 3867, 11180,
        6937, 107, 236748, 1389, 236762, 236772, 45134, 2617, 200179, 147, 236865, 3554, 1186, 236793, 506, 9711, 2036,
        14736, 580, 1061, 2411, 107, 236748, 1389, 236762, 236772, 45134, 2617, 3762, 622, 23932, 1389, 236786, 45134,
        236761, 3723, 753, 108, 3689, 563, 506, 5279, 529, 7001, 236881, 25685, 528, 886, 3658, 236761, 106, 107, 105,
        4368, 107,
];

/// The model's tokenizer, if its GGUF is on disk (`var` to point at it).
fn tokenizer_of(var: &str, default: &str) -> Option<Tokenizer> {
    let path = std::env::var(var).unwrap_or_else(|_| default.into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    Some(Tokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap())
}

#[test]
fn gemma3_encodes_markdown_as_llama_cpp_does() {
    let Some(tok) = tokenizer_of("GEMMA3_GGUF", r"E:\models\gemma-3-4b-it-q4_k_m.gguf") else { return };
    let prompt = format!("<bos><start_of_turn>user\n{EXCERPT}\n\nWhat is the capital of France? Answer in one word.<end_of_turn>\n<start_of_turn>model\n");
    assert_eq!(tok.encode(&prompt, false).unwrap(), WANT);
}

#[test]
fn gemma4_encodes_markdown_as_llama_cpp_does() {
    let Some(tok) = tokenizer_of("GEMMA4_GGUF", r"E:\models\gemma-4-e2b-q4_k_m.gguf") else { return };
    let prompt = format!("<bos><|turn>user\n{EXCERPT}\n\nWhat is the capital of France? Answer in one word.<turn|>\n<|turn>model\n");
    assert_eq!(tok.encode(&prompt, false).unwrap(), WANT);
}
