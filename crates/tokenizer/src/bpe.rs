//! Byte-level BPE encoder / decoder.
//!
//! Algorithm:
//!   1. Pretokenize via [`pretokenizer::split_llama3`] — splits on word /
//!      punctuation / digit boundaries so BPE doesn't merge across them.
//!   2. For each piece: UTF-8 encode, map each byte to a "visible" char.
//!   3. Apply BPE merges in priority order (lowest rank first) within the
//!      piece's char sequence.
//!   4. Look up each final piece in the vocab.
//!
//! Decoding is the reverse: concat token strings, reverse the byte mapping.

use crate::byte_encoder::{bytes_to_visible, visible_to_bytes};
use crate::error::{Result, TokenizerError};
use crate::pretokenizer;
use crate::Tokenizer;

pub fn encode(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() { return Ok(()); }

    // First pass: peel off special tokens (chat markers, BOS/EOS literals, etc.)
    // by literal-string match. Any text between specials gets BPE-encoded normally.
    // Without this, byte-level BPE would shred `<|im_end|>` into 6 separate tokens
    // and the model would never see the chat structure.
    let specials = tok.special_tokens();
    let mut cursor = 0usize;
    while cursor < text.len() {
        // Find the earliest occurrence of any special token from `cursor` on.
        // Ties at the same position go to the longest match — `specials` is
        // sorted longest-first, so the first hit at a given offset is also the
        // longest.
        let mut best: Option<(usize, usize, u32)> = None;
        for (s, id) in specials {
            if let Some(off) = text[cursor..].find(s.as_str()) {
                let pos = cursor + off;
                let end = pos + s.len();
                match best {
                    None => best = Some((pos, end, *id)),
                    Some((bp, be, _)) => {
                        if pos < bp || (pos == bp && end > be) {
                            best = Some((pos, end, *id));
                        }
                    }
                }
            }
        }

        match best {
            Some((start, end, id)) => {
                if start > cursor {
                    encode_text_segment(tok, &text[cursor..start], out)?;
                }
                out.push(id);
                cursor = end;
            }
            None => {
                encode_text_segment(tok, &text[cursor..], out)?;
                break;
            }
        }
    }

    Ok(())
}

/// BPE-encode a contiguous run of normal (non-special) text. Same algorithm
/// as before the special-token pre-pass: pretokenize into word/punct pieces,
/// byte-encode each piece, then apply BPE merges in priority order.
fn encode_text_segment(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() { return Ok(()); }
    let merges = tok.merges_rank();

    // VENDORED-LOCAL: preserve added-token matching before NFC normalization.
    use unicode_normalization::UnicodeNormalization;
    let normalized;
    let text = if tok.qwen3_pre { normalized=text.nfc().collect::<String>(); &normalized } else {text};
    let pieces=if tok.qwen3_pre {pretokenizer::split_qwen3(text)} else {pretokenizer::split_llama3(text)};
    for piece in pieces {
        let visible = bytes_to_visible(piece.as_bytes());
        let mut tokens: Vec<String> = visible.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx: Option<usize> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let key = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&r) = merges.get(&key) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(i) = best_idx else { break; };
            let merged = format!("{}{}", tokens[i], tokens[i + 1]);
            tokens[i] = merged;
            tokens.remove(i + 1);
        }

        for p in &tokens {
            if let Some(id) = tok.id_of(p) {
                out.push(id);
            } else if let Some(unk) = tok.unk() {
                out.push(unk);
            } else {
                return Err(TokenizerError::MissingMetadata(
                    "BPE produced an out-of-vocab piece and no unk token is defined",
                ));
            }
        }
    }

    Ok(())
}

pub fn decode(tok: &Tokenizer, ids: &[u32]) -> String {
    let mut visible = String::new();
    for &id in ids {
        if let Some(s) = tok.token(id) {
            visible.push_str(s);
        }
    }
    let bytes = visible_to_bytes(&visible);
    String::from_utf8_lossy(&bytes).into_owned()
}
