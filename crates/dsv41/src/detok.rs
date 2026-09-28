//! Token ids back to text, from the checkpoint's `tokenizer.json` (byte-level
//! BPE). Decoding only: every token maps to a fixed byte string, so no merges
//! or pre-tokenizer are needed. (Encoding needs the pre-tokenizer's Unicode
//! regex; until that lands, `tools/dsv41/encode.py` produces prompt ids.)
//!
//! Special tokens (`added_tokens`) decode to their literal content. A token
//! can end mid UTF-8 sequence, so streaming callers work in bytes
//! ([`Detokenizer::bytes`]) and print once a sequence completes.

use std::path::Path;

use oaiy_engine::json::Json;
use oaiy_engine::{Error, Result};

pub struct Detokenizer {
    /// Byte string of every token id (empty for unused ids).
    tokens: Vec<Vec<u8>>,
}

impl Detokenizer {
    pub fn load(model_dir: &Path) -> Result<Detokenizer> {
        let src = std::fs::read(model_dir.join("tokenizer.json"))?;
        let root = Json::parse(&src)?;
        let model = root.get("model").ok_or_else(|| Error::Format("tokenizer.json: no model".into()))?;
        let vocab = model
            .get("vocab")
            .filter(|v| v.as_object().is_some())
            .ok_or_else(|| Error::Format("tokenizer.json: no model.vocab".into()))?;
        let unicode_to_byte = byte_decoder();
        let mut tokens: Vec<Vec<u8>> = Vec::new();
        let mut put = |id: i64, bytes: Vec<u8>| {
            if id >= 0 {
                let id = id as usize;
                if tokens.len() <= id {
                    tokens.resize(id + 1, Vec::new());
                }
                tokens[id] = bytes;
            }
        };
        let id = |v: Option<&Json>| v.and_then(Json::as_i64).unwrap_or(-1);
        for (k, v) in vocab.members() {
            // every char of a byte-level token stands for exactly one byte
            let bytes = k.chars().map(|c| unicode_to_byte.get(&c).copied().unwrap_or(b'?')).collect();
            put(id(Some(v)), bytes);
        }
        for a in root.get("added_tokens").and_then(Json::as_array).unwrap_or(&[]) {
            let content = a.get("content").and_then(Json::as_str).unwrap_or("");
            put(id(a.get("id")), content.as_bytes().to_vec());
        }
        Ok(Detokenizer { tokens })
    }

    /// The bytes token `id` stands for (empty if unknown).
    pub fn bytes(&self, id: u32) -> &[u8] {
        self.tokens.get(id as usize).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Decode a whole sequence (invalid UTF-8 replaced).
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.bytes(id).iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's byte-to-unicode table, inverted: printable bytes map to
/// themselves, the rest to U+0100 upwards, in byte order.
fn byte_decoder() -> std::collections::HashMap<char, u8> {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut map = std::collections::HashMap::with_capacity(256);
    let mut next = 256u32;
    for b in 0..=255u8 {
        let c = if printable(b) {
            b as u32
        } else {
            next += 1;
            next - 1
        };
        map.insert(char::from_u32(c).expect("valid scalar"), b);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_decoder_is_a_bijection_with_gpt2_space() {
        let m = byte_decoder();
        assert_eq!(m.len(), 256);
        assert_eq!(m[&'Ġ'], b' '); // U+0120 is the space byte
        assert_eq!(m[&'Ċ'], b'\n');
        assert_eq!(m[&'A'], b'A');
    }

    #[test]
    fn decodes_the_golden_answer_when_present() {
        let dir = std::path::PathBuf::from(std::env::var("DSV41_MODEL").unwrap_or_else(|_| r"E:\deepseek\model".into()));
        if !dir.join("tokenizer.json").exists() {
            return;
        }
        let d = Detokenizer::load(&dir).unwrap();
        assert_eq!(d.decode(&[671, 6102, 294, 8760, 344, 2619, 51119, 42499]), "The capital of France is **Paris**.");
    }
}
