//! SentencePiece encoder for `tokenizer.ggml.model = "llama"` vocabularies (Llama 1/2, Mistral, Gemma 3 and 3n) and
//! Gemma 4's `"gemma4"`, as llama.cpp encodes them (llama-vocab.cpp: `tokenizer_st_partition`, then
//! `llm_tokenizer_spm_session`):
//!
//!   1. The text is split on special tokens (CONTROL and USER_DEFINED pieces), the longest first. Besides chat markers
//!      such as `<start_of_turn>`, Gemma's vocabulary marks its runs of newlines (`"\n"` .. 31 of them) and of spaces
//!      (2 .. 31) USER_DEFINED, so they are taken out here as single tokens.
//!   2. Each piece of plain text: a leading space when the vocabulary asks for one (`tokenizer.ggml.add_space_prefix`,
//!      true unless the GGUF says otherwise; Gemma's says false) and the piece opens the text or follows a special
//!      token; spaces become U+2581 (`▁`).
//!   3. Its characters are merged pairwise, always the neighbouring pair whose merge is the highest-scoring vocabulary
//!      piece (the leftmost of equals), until no pair merges; a symbol left that is not a piece becomes byte-fallback
//!      tokens (`<0xXX>`), else unk. Gemma 4's vocabulary gives every piece the same score and its order as merges
//!      (BPE in SentencePiece's alphabet): there the pair merged first is the earliest merge, and only listed pairs
//!      merge (its ids then are llama.cpp's on every text tried, a 44,592-token one among them).
//!
//! This replaced a greedy longest-match walk that also put a `▁` before every piece of text, Gemma's too: every line
//! after a newline began with a space the model never saw in training, and a prompt of markdown with a few dozen
//! newlines made Gemma 3 answer with fragments of it.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{Result, TokenizerError};
use crate::Tokenizer;

const SPM_SPACE: char = '▁';

pub fn encode(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let mut prev_special = true;
    for frag in partition(text, tok.special_tokens()) {
        match frag {
            Frag::Special(id) => {
                out.push(id);
                prev_special = true;
            }
            Frag::Text(piece) => {
                let mut escaped = String::with_capacity(piece.len() + 3);
                if tok.add_space_prefix() && prev_special {
                    escaped.push(SPM_SPACE);
                }
                escaped.extend(piece.chars().map(|c| if c == ' ' { SPM_SPACE } else { c }));
                merge(tok, &escaped, out)?;
                prev_special = false;
            }
        }
    }
    Ok(())
}

/// A piece of the text: plain text, or a special token found in it.
enum Frag<'a> {
    Text(&'a str),
    Special(u32),
}

/// `text` split on `specials` (longest first, as given), each special taken out of the plain text that is left
/// wherever it occurs. A special whose first two bytes never occur next to each other in the text (its first byte, for
/// one of a single byte) cannot be in it and is skipped without a search: Gemma's vocabulary has 6,415 of them.
fn partition<'a>(text: &'a str, specials: &[(String, u32)]) -> Vec<Frag<'a>> {
    let bytes = text.as_bytes();
    let mut firsts = [false; 256];
    let mut pairs = vec![false; 1 << 16];
    for (i, &b) in bytes.iter().enumerate() {
        firsts[b as usize] = true;
        if let Some(&n) = bytes.get(i + 1) {
            pairs[(b as usize) << 8 | n as usize] = true;
        }
    }
    let mut frags = vec![Frag::Text(text)];
    for (s, id) in specials {
        let sb = s.as_bytes();
        let possible = match sb {
            [] => false,
            [b] => firsts[*b as usize],
            [a, b, ..] => pairs[(*a as usize) << 8 | *b as usize],
        };
        if !possible || !frags.iter().any(|f| matches!(f, Frag::Text(t) if t.contains(s.as_str()))) {
            continue;
        }
        let mut next = Vec::with_capacity(frags.len() + 8);
        for frag in frags {
            match frag {
                Frag::Text(t) => {
                    let mut rest = t;
                    while let Some(at) = rest.find(s.as_str()) {
                        if at > 0 {
                            next.push(Frag::Text(&rest[..at]));
                        }
                        next.push(Frag::Special(*id));
                        rest = &rest[at + s.len()..];
                    }
                    if !rest.is_empty() {
                        next.push(Frag::Text(rest));
                    }
                }
                special => next.push(special),
            }
        }
        frags = next;
    }
    frags
}

/// One symbol of the text being merged: a byte range, and its neighbours in what is left of the list (`usize::MAX`
/// for none).
struct Symbol {
    start: usize,
    len: usize,
    prev: usize,
    next: usize,
}

const NONE: usize = usize::MAX;

/// A pair of neighbouring symbols that may merge, ordered so the heap gives the highest priority first (a piece's score,
/// or minus a merge's rank) and, of equal ones, the leftmost.
struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    len: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).then_with(|| other.left.cmp(&self.left))
    }
}

/// SPM-encode one piece of escaped plain text (the module's step 3).
fn merge(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let mut syms: Vec<Symbol> = text
        .char_indices()
        .map(|(start, c)| Symbol { start, len: c.len_utf8(), prev: NONE, next: NONE })
        .collect();
    let n = syms.len();
    for i in 0..n {
        syms[i].prev = if i == 0 { NONE } else { i - 1 };
        syms[i].next = if i + 1 == n { NONE } else { i + 1 };
    }
    // A vocabulary without scores (none seen in a GGUF) prefers its earlier pieces, as SentencePiece orders them.
    let score = |id: u32| tok.scores().and_then(|s| s.get(id as usize).copied()).unwrap_or(-(id as f32));
    let by_rank = tok.spm_by_merge_rank();
    let mut heap = BinaryHeap::new();
    let try_add = |heap: &mut BinaryHeap<Bigram>, syms: &[Symbol], left: usize, right: usize| {
        if left == NONE || right == NONE {
            return;
        }
        let (l, r) = (&syms[left], &syms[right]);
        let len = l.len + r.len;
        let priority = if by_rank {
            let pair = (text[l.start..l.start + l.len].to_string(), text[r.start..r.start + r.len].to_string());
            tok.merges_rank().get(&pair).map(|&rank| -(rank as f32))
        } else {
            tok.id_of(&text[l.start..l.start + len]).map(score)
        };
        if let Some(score) = priority {
            heap.push(Bigram { score, left, right, len });
        }
    };
    for i in 1..n {
        try_add(&mut heap, &syms, i - 1, i);
    }
    while let Some(b) = heap.pop() {
        let (l, r) = (b.left, b.right);
        // a pair one of whose symbols has since been merged into another is stale
        if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != b.len {
            continue;
        }
        syms[l].len += syms[r].len;
        syms[r].len = 0;
        syms[l].next = syms[r].next;
        if syms[r].next != NONE {
            let after = syms[r].next;
            syms[after].prev = l;
        }
        try_add(&mut heap, &syms, syms[l].prev, l);
        try_add(&mut heap, &syms, l, syms[l].next);
    }
    let mut i = 0;
    while i != NONE {
        let piece = &text[syms[i].start..syms[i].start + syms[i].len];
        match tok.id_of(piece) {
            Some(id) => out.push(id),
            None => {
                for b in piece.bytes() {
                    match tok.id_of(&format!("<0x{b:02X}>")).or(tok.unk()) {
                        Some(id) => out.push(id),
                        None => return Err(TokenizerError::MissingMetadata("SPM byte-fallback token not in vocab and no unk defined")),
                    }
                }
            }
        }
        i = syms[i].next;
    }
    Ok(())
}

pub fn decode(tok: &Tokenizer, ids: &[u32]) -> String {
    let mut out = String::new();
    for &id in ids {
        let Some(s) = tok.token(id) else { continue };

        // Skip unknown / control / unused tokens — markers like `<s>`, `<unk>`, `<start_of_turn>` that shouldn't
        // render. USER_DEFINED pieces are text, written as they are (as llama.cpp's `token_to_piece` does): Gemma's
        // newlines and runs of spaces are, and skipping them took every newline out of its replies.
        if let Some(types) = tok.token_types() {
            if let Some(&ty) = types.get(id as usize) {
                // 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED
                if matches!(ty, 2 | 3 | 5) {
                    continue;
                }
                if ty == 4 {
                    out.push_str(s);
                    continue;
                }
            }
        }

        // Byte-fallback tokens "<0xXX>" decode to a single byte.
        if let Some(byte) = parse_byte_fallback(s) {
            out.push_str(&String::from_utf8_lossy(&[byte]));
        } else {
            out.push_str(&s.replace(SPM_SPACE, " "));
        }
    }
    // We deliberately do NOT strip a leading space — for streaming token-at-a-time decoding, the leading ▁ on each
    // word IS the inter-word space. Callers doing full-sequence decoding can `.trim_start()` themselves.
    out
}

fn parse_byte_fallback(s: &str) -> Option<u8> {
    let s = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(s, 16).ok()
}
