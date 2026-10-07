//! BPE / SPM tokenizer fed by GGUF metadata.
//!
//! Two encoders are supported:
//!   * **GPT2 / Llama-3 / Qwen / Gemma** (`tokenizer.ggml.model = "gpt2"`):
//!     byte-level BPE with merges loaded from the GGUF.
//!   * **SentencePiece** (`tokenizer.ggml.model = "llama"`, Gemma 4's `"gemma4"`):
//!     llama.cpp's encoder — special tokens split out first, then pairs of
//!     symbols merged by score (see [`spm`]).
//!
//! Decoding is straightforward in both: look up token strings, concatenate,
//! and (for byte-level) reverse the byte→Unicode mapping.
//!
//! Limitations vs. upstream tokenizers (called out so you don't get surprised):
//!   - Pretokenizer regex (Llama-3's pattern) is **not** implemented; we feed
//!     the whole prompt to BPE in one go, which yields slightly different
//!     tokens for inputs that contain digits, punctuation runs, or unusual
//!     whitespace. For typical English prompts the output is identical or
//!     nearly so. To be safe, validate against `transformers` or `tiktoken`
//!     for a given vocab before shipping production text.
//!   - Special-token handling is minimal — BOS/EOS are honored, but
//!     instruction-tuned models with `<|im_start|>` / `<|begin_of_text|>` etc.
//!     are *not* injected automatically. The caller wires those in.

#![deny(rust_2018_idioms)]

pub mod bpe;
pub mod byte_encoder;
pub mod error;
pub mod pretokenizer;
pub mod spm;

use std::collections::HashMap;
use std::sync::Arc;

use gguf::value::Array;
use gguf::GgufFile;

pub use error::{Result, TokenizerError};

/// Which encoder family to use. Determined by `tokenizer.ggml.model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerModel {
    /// Byte-level BPE (Llama 3, Qwen, Gemma, GPT-2 family).
    Gpt2,
    /// SentencePiece BPE (Llama 1/2). Uses scores for tie-breaking.
    Llama,
    /// WordPiece (BERT-style). Not yet supported for encoding; decode-only.
    Bert,
    /// RWKV. Not yet supported.
    Rwkv,
    /// Unknown / custom.
    Other(Arc<str>),
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    // VENDORED-LOCAL: HF Qwen3.8 uses NFC, combining marks and single digits.
    qwen3_pre: bool,
    model: TokenizerModel,

    /// Token strings, indexed by token id.
    tokens: Vec<String>,

    /// Reverse lookup: token string -> token id.
    token_to_id: HashMap<String, u32>,

    /// SentencePiece scores (one per token), if present: the SPM encoder merges
    /// the pair whose piece scores highest first.
    scores: Option<Vec<f32>>,

    /// Whether SPM text after a special token (or at the start) takes a leading
    /// space: `tokenizer.ggml.add_space_prefix`, true when absent (llama.cpp's
    /// default for SentencePiece); Gemma's GGUFs say false.
    add_space_prefix: bool,

    /// Whether the SPM encoder merges by merge rank rather than by score:
    /// Gemma 4 (`"gemma4"`), whose pieces all score the same and whose merges
    /// carry the order.
    spm_by_merge_rank: bool,

    /// Token type per id (1=normal, 2=unknown, 3=control, 4=user-defined,
    /// 5=unused, 6=byte). Used for special-token detection (TODO).
    #[allow(dead_code)]
    token_types: Option<Vec<i32>>,

    /// BPE merges, keyed by `(left, right)` -> rank (lower = applied first).
    merges_rank: HashMap<(String, String), u32>,

    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
    pad: Option<u32>,

    /// Pre-computed list of "special" token strings + ids — anything with
    /// `token_type` 3 (CONTROL) or 4 (USER_DEFINED). The encoder substitutes
    /// these by literal-string match before running BPE merges, so e.g.
    /// `<|im_end|>` resolves to its single token id rather than getting
    /// chopped into `<`, `|`, `im`, `_end`, `|`, `>`.
    ///
    /// Sorted by length descending — see [`Tokenizer::special_tokens`].
    special_tokens: Vec<(String, u32)>,
}

impl Tokenizer {
    /// VENDORED-LOCAL: construct the same byte BPE from a HF checkpoint.
    pub fn from_qwen3_bpe_parts(tokens: Vec<String>, merges: Vec<(String,String)>,
        special_ids: Vec<u32>, eos: u32) -> Result<Self> {
        if eos as usize >= tokens.len() || special_ids.iter().any(|&i|i as usize>=tokens.len()) {
            return Err(TokenizerError::MissingMetadata("valid special token IDs"));
        }
        let token_to_id=tokens.iter().enumerate().map(|(i,t)|(t.clone(),i as u32)).collect();
        let merges_rank=merges.into_iter().enumerate().map(|(i,m)|(m,i as u32)).collect();
        let mut special_tokens:Vec<_>=special_ids.into_iter().map(|i|(tokens[i as usize].clone(),i)).collect();
        special_tokens.sort_by(|a,b|b.0.len().cmp(&a.0.len()));
        Ok(Self{qwen3_pre:true,model:TokenizerModel::Gpt2,tokens,token_to_id,scores:None,add_space_prefix:false,spm_by_merge_rank:false,token_types:None,
            merges_rank,bos:None,eos:Some(eos),unk:None,pad:None,special_tokens})
    }
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model_name = gguf.get_str("tokenizer.ggml.model").unwrap_or("llama");
        let model = match model_name {
            "gpt2" => TokenizerModel::Gpt2,
            // "llama" is the SentencePiece-based llama.cpp identifier; "gemma4"
            // ships with both scores (all equal) and merges (BPE in
            // SentencePiece's alphabet): the SPM path takes it, merging by rank
            // (`spm_by_merge_rank`).
            "llama" | "gemma4" => TokenizerModel::Llama,
            "bert" => TokenizerModel::Bert,
            "rwkv" => TokenizerModel::Rwkv,
            other => TokenizerModel::Other(other.into()),
        };

        let tokens = match gguf.get("tokenizer.ggml.tokens")?.as_array() {
            Some(Array::String(v)) => v.clone(),
            _ => return Err(TokenizerError::MissingMetadata("tokenizer.ggml.tokens (string array)")),
        };

        let scores = match gguf.metadata().get("tokenizer.ggml.scores").and_then(|v| v.as_array()) {
            Some(Array::F32(v)) => Some(v.clone()),
            _ => None,
        };

        let token_types = match gguf.metadata().get("tokenizer.ggml.token_type").and_then(|v| v.as_array()) {
            Some(Array::I32(v)) => Some(v.clone()),
            _ => None,
        };

        let merges = match gguf.metadata().get("tokenizer.ggml.merges").and_then(|v| v.as_array()) {
            Some(Array::String(v)) => v.clone(),
            _ => Vec::new(),
        };

        let mut merges_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            // Each merge is "left right" — split on first space.
            let mut split = m.splitn(2, ' ');
            let l = split.next().unwrap_or("");
            let r = split.next().unwrap_or("");
            if l.is_empty() || r.is_empty() { continue; }
            merges_rank.insert((l.to_string(), r.to_string()), rank as u32);
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let bos = gguf.get_u64("tokenizer.ggml.bos_token_id").ok().map(|v| v as u32);
        let eos = gguf.get_u64("tokenizer.ggml.eos_token_id").ok().map(|v| v as u32);
        let unk = gguf.get_u64("tokenizer.ggml.unknown_token_id").ok().map(|v| v as u32);
        let pad = gguf.get_u64("tokenizer.ggml.padding_token_id").ok().map(|v| v as u32);
        let add_space_prefix = gguf.get_bool("tokenizer.ggml.add_space_prefix").unwrap_or(true);
        let spm_by_merge_rank = model_name == "gemma4" && !merges_rank.is_empty();

        // Build the special-token table: any token with type 3 (CONTROL) or 4
        // (USER_DEFINED). These need literal-string matching before BPE so multi-
        // char markers like `<|im_end|>` aren't shredded by byte-level BPE.
        let mut special_tokens: Vec<(String, u32)> = Vec::new();
        if let Some(types) = &token_types {
            for (i, &t) in types.iter().enumerate() {
                if (t == 3 || t == 4) && i < tokens.len() {
                    let s = &tokens[i];
                    if !s.is_empty() {
                        special_tokens.push((s.clone(), i as u32));
                    }
                }
            }
        }
        // Longest-first so prefix-matching the wrong (shorter) token doesn't
        // win when a longer special is also a prefix.
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // (a Qwen GGUF's own pre-tokenizer, `tokenizer.ggml.pre` qwen2 or qwen35: its letters with their marks, its
        // digits one by one and NFC first, as its Hugging Face tokenizer has them; Llama's splitter otherwise)
        let qwen3_pre = model == TokenizerModel::Gpt2 && matches!(gguf.get_str("tokenizer.ggml.pre"), Ok("qwen2" | "qwen35"));
        Ok(Self {
            qwen3_pre,
            model,
            tokens,
            token_to_id,
            scores,
            add_space_prefix,
            spm_by_merge_rank,
            token_types,
            merges_rank,
            bos,
            eos,
            unk,
            pad,
            special_tokens,
        })
    }

    pub fn model(&self) -> &TokenizerModel { &self.model }
    pub fn vocab_size(&self) -> usize { self.tokens.len() }
    pub fn bos(&self) -> Option<u32> { self.bos }
    pub fn eos(&self) -> Option<u32> { self.eos }
    pub fn unk(&self) -> Option<u32> { self.unk }
    pub fn pad(&self) -> Option<u32> { self.pad }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Reverse lookup: token literal -> id. Useful for finding special tokens
    /// like `<|im_end|>` / `<end_of_turn>` to use as stop signals during chat
    /// generation.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    /// Encode `text` into token ids. If `add_bos` and a BOS token is defined,
    /// it's prepended.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(b) = self.bos { ids.push(b); }
        }
        match self.model {
            TokenizerModel::Gpt2 => bpe::encode(self, text, &mut ids)?,
            TokenizerModel::Llama => spm::encode(self, text, &mut ids)?,
            TokenizerModel::Bert | TokenizerModel::Rwkv | TokenizerModel::Other(_) => {
                return Err(TokenizerError::UnsupportedEncoder(format!("{:?}", self.model)));
            }
        }
        Ok(ids)
    }

    /// Decode token ids back into text. Best-effort: unknown ids are skipped.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self.model {
            TokenizerModel::Gpt2 => bpe::decode(self, ids),
            TokenizerModel::Llama => spm::decode(self, ids),
            TokenizerModel::Bert | TokenizerModel::Rwkv | TokenizerModel::Other(_) => {
                ids.iter().filter_map(|&id| self.token(id)).collect()
            }
        }
    }

    pub(crate) fn id_of(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    /// Special tokens (CONTROL/USER_DEFINED), sorted longest-first.
    /// Used by encoders to substitute multi-char markers before BPE/SPM.
    pub(crate) fn special_tokens(&self) -> &[(String, u32)] {
        &self.special_tokens
    }

    pub(crate) fn merges_rank(&self) -> &HashMap<(String, String), u32> {
        &self.merges_rank
    }

    pub(crate) fn scores(&self) -> Option<&[f32]> {
        self.scores.as_deref()
    }

    pub(crate) fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    pub(crate) fn spm_by_merge_rank(&self) -> bool {
        self.spm_by_merge_rank
    }

    #[allow(dead_code)]
    pub(crate) fn token_types(&self) -> Option<&[i32]> {
        self.token_types.as_deref()
    }
}
