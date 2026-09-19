//! SentencePiece-ish encoder for Llama-1/2 vocabularies.
//!
//! Not a true Viterbi unigram decoder — that's the upstream-faithful answer
//! and is on the roadmap. For now we use **greedy longest-match**:
//!
//!   1. Replace U+0020 (space) with U+2581 ('▁') per SPM convention.
//!   2. Walk the input, at each position pick the longest vocabulary token
//!      that matches.
//!   3. If nothing matches, emit a single byte fallback (`<0xXX>`) if
//!      available, otherwise the unk token.
//!
//! For typical English prompts this matches the Viterbi result; for unusual
//! inputs (multilingual punctuation, code, etc.) it can diverge slightly.

use crate::error::{Result, TokenizerError};
use crate::Tokenizer;

const SPM_SPACE: char = '▁';

pub fn encode(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() { return Ok(()); }

    // First pass: peel off special tokens (chat markers like `<start_of_turn>`,
    // `<bos>`, etc.) by literal-string match. Without this, SPM's longest-match
    // walk would silently fail on multi-char markers — they're not in the
    // vocabulary as a single piece for SPM-style models, and the byte-fallback
    // path would emit them as raw bytes.
    let specials = tok.special_tokens();
    if !specials.is_empty() {
        let mut cursor = 0usize;
        while cursor < text.len() {
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
        return Ok(());
    }

    encode_text_segment(tok, text, out)
}

/// SPM-encode a contiguous run of normal (non-special) text. Same algorithm
/// as before the special-token pre-pass: SPM-normalize spaces, then greedy
/// longest-match walk through the vocabulary.
fn encode_text_segment(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() { return Ok(()); }

    // Normalize spaces. SPM also prefixes with a space — many Llama vocabs
    // expect that, so we add a leading ▁ if the input doesn't start with one.
    let normalized: String = std::iter::once(SPM_SPACE)
        .chain(text.chars().map(|c| if c == ' ' { SPM_SPACE } else { c }))
        .collect();

    let bytes = normalized.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Try to match the longest token starting at byte i. We walk forward in
        // increasing UTF-8-safe boundaries; SPM tokens are usually short.
        let mut best_end: Option<usize> = None;
        let mut end = i;
        // Cap: scan up to 64 bytes ahead. Tokens >64 bytes are pathological
        // for natural-language inputs.
        let limit = (i + 64).min(bytes.len());
        while end < limit {
            // Advance to next UTF-8 char boundary.
            let next = utf8_next(bytes, end);
            if next == end { break; }
            end = next;
            // Try this slice.
            let slice = std::str::from_utf8(&bytes[i..end]).unwrap_or("");
            if !slice.is_empty() && tok.id_of(slice).is_some() {
                best_end = Some(end);
            }
        }

        if let Some(e) = best_end {
            let slice = std::str::from_utf8(&bytes[i..e]).unwrap();
            let id = tok.id_of(slice).unwrap();
            out.push(id);
            i = e;
        } else {
            // Single-char fallback. Emit byte fallback `<0xXX>` if vocab has it,
            // else unk.
            let next = utf8_next(bytes, i);
            let n = (next - i).max(1);
            for b in &bytes[i..i + n] {
                let byte_tok = format!("<0x{:02X}>", b);
                if let Some(id) = tok.id_of(&byte_tok) {
                    out.push(id);
                } else if let Some(unk) = tok.unk() {
                    out.push(unk);
                } else {
                    return Err(TokenizerError::MissingMetadata(
                        "SPM byte-fallback token not in vocab and no unk defined",
                    ));
                }
            }
            i += n;
        }
    }

    Ok(())
}

pub fn decode(tok: &Tokenizer, ids: &[u32]) -> String {
    let mut out = String::new();
    for &id in ids {
        let Some(s) = tok.token(id) else { continue };

        // Skip control / unknown / user-defined special tokens — these are
        // markers like `<s>`, `<unk>`, `<|im_start|>` that shouldn't render.
        if let Some(types) = tok.token_types() {
            if let Some(&ty) = types.get(id as usize) {
                // 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED
                if matches!(ty, 2 | 3 | 4 | 5) { continue; }
            }
        }

        // Byte-fallback tokens "<0xXX>" decode to a single byte.
        if let Some(byte) = parse_byte_fallback(s) {
            out.push_str(&String::from_utf8_lossy(&[byte]));
        } else {
            out.push_str(s);
        }
    }
    // Convert SPM space marker back to ASCII space. We deliberately do NOT
    // strip a leading space — for streaming token-at-a-time decoding, the
    // leading ▁ on each word IS the inter-word space. Callers doing
    // full-sequence decoding can `.trim_start()` themselves.
    out.replace(SPM_SPACE, " ")
}

fn parse_byte_fallback(s: &str) -> Option<u8> {
    let s = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(s, 16).ok()
}

/// Find the next byte index that's a UTF-8 char boundary, given that `i` is
/// already on a boundary.
fn utf8_next(bytes: &[u8], i: usize) -> usize {
    if i >= bytes.len() { return i; }
    let lead = bytes[i];
    let n = if lead < 0x80 { 1 }
            else if lead < 0xC0 { 1 }   // continuation byte (corrupt input); advance 1
            else if lead < 0xE0 { 2 }
            else if lead < 0xF0 { 3 }
            else { 4 };
    (i + n).min(bytes.len())
}
