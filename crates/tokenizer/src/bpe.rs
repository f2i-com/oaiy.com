//! Byte-level BPE encoder / decoder.
//!
//! Algorithm:
//!   1. Pretokenize via [`pretokenizer::split_llama3`] — splits on word /
//!      punctuation / digit boundaries so BPE doesn't merge across them.
//!   2. For each piece: UTF-8 encode, map each byte to a "visible" char.
//!   3. Apply BPE merges in priority order (lowest rank first) within the
//!      piece's char sequence, as `spm.rs` merges and DeepSeek's tokenizer
//!      does: the symbols a linked list, the pairs that may merge a heap.
//!      This was a loop over every pair for each pair merged, a `String` cloned
//!      for each look: hours on one word of 1 MB, on the request's thread, before
//!      there was a job to cancel.
//!   4. Look up each final piece in the vocab.
//!
//! Decoding is the reverse: concat token strings, reverse the byte mapping.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::byte_encoder::{bytes_to_visible, visible_to_bytes};
use crate::error::{Result, TokenizerError};
use crate::literal;
use crate::pretokenizer;
use crate::Tokenizer;

/// The longest run of bytes merged as one. No token is near this long (a piece past it is a long run of one letter or
/// of punctuation, or text with no spaces), and what a merge holds is some 60 bytes a byte: a 64 MiB word whole
/// would have asked each connection for gigabytes. Past it a piece is merged a part at a time, its ids the same but
/// where a part ends.
const MAX_PIECE: usize = 64 << 10;

pub fn encode(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    if text.is_empty() { return Ok(()); }

    // First pass: peel off special tokens (chat markers, BOS/EOS literals, etc.)
    // by literal-string match. Any text between specials gets BPE-encoded normally.
    // Without this, byte-level BPE would shred `<|im_end|>` into 6 separate tokens
    // and the model would never see the chat structure.
    //
    // Each special's next occurrence waits in a heap, earliest first and of those at the same place the longest, and
    // is looked for again only once the cursor has passed it: the text is searched once for each special, however
    // many times one occurs. (Every special searched for again at each one found was quadratic in a prompt of them.)
    let specials = tok.special_tokens();
    let next = |i: usize, from: usize| {
        let s = &specials[i].0;
        text[from..].find(s.as_str()).map(|at| Reverse((from + at, Reverse(s.len()), i)))
    };
    let mut heap: BinaryHeap<_> = (0..specials.len()).filter_map(|i| next(i, 0)).collect();
    let mut cursor = 0usize;
    while let Some(Reverse((pos, Reverse(len), i))) = heap.pop() {
        if pos >= cursor {
            if pos > cursor {
                encode_text_segment(tok, &text[cursor..pos], out)?;
            }
            out.push(specials[i].1);
            cursor = pos + len;
        }
        heap.extend(next(i, cursor));
    }
    encode_text_segment(tok, &text[cursor..], out)
}

/// BPE-encode a contiguous run of normal (non-special) text. Same algorithm
/// as before the special-token pre-pass: pretokenize into word/punct pieces,
/// byte-encode each piece, then apply BPE merges in priority order.
fn encode_text_segment(tok: &Tokenizer, text: &str, out: &mut Vec<u32>) -> Result<()> {
    let text = literal::unmark(text);
    let text = &*text;
    if text.is_empty() { return Ok(()); }

    // preserve added-token matching before NFC normalization.
    use unicode_normalization::UnicodeNormalization;
    let normalized;
    let text = if tok.qwen3_pre { normalized=text.nfc().collect::<String>(); &normalized } else {text};
    let pieces=if tok.qwen3_pre {pretokenizer::split_qwen3(text)} else {pretokenizer::split_llama3(text)};
    for piece in pieces {
        for part in piece.as_bytes().chunks(MAX_PIECE) {
            let visible = bytes_to_visible(part);
            if tok.ignore_merges {
                if let Some(id) = tok.id_of(&visible).filter(|&id| !tok.is_control(id)) {
                    out.push(id);
                    continue;
                }
            }
            merge(tok, &visible, out)?;
        }
    }

    Ok(())
}

/// One symbol of a piece being merged: a byte range of its visible text, and its neighbours in what is left of the
/// list ([`NONE`] for none).
struct Symbol {
    start: usize,
    len: usize,
    prev: usize,
    next: usize,
}

const NONE: usize = usize::MAX;

/// BPE-merge the non-empty `visible` text of one piece (the module's step 3): the pair of the lowest rank first, of
/// equal pairs the leftmost, each merge putting the pairs it makes in the heap. A pair in the heap is `(rank, left,
/// right, its length)`; one whose symbols have changed since it was queued is passed over when it comes up.
///
/// By the symbols' text, not ids, for the same answer in a vocabulary that lacks a byte or a merge's result: such a
/// piece still merges, and only then is out of the vocabulary.
fn merge(tok: &Tokenizer, visible: &str, out: &mut Vec<u32>) -> Result<()> {
    let ranks = tok.merges_rank();
    let mut syms: Vec<Symbol> = visible.char_indices().map(|(start, c)| Symbol { start, len: c.len_utf8(), prev: NONE, next: NONE }).collect();
    let n = syms.len();
    for i in 0..n {
        syms[i].prev = if i == 0 { NONE } else { i - 1 };
        syms[i].next = if i + 1 == n { NONE } else { i + 1 };
    }
    let text = |s: &Symbol| &visible[s.start..s.start + s.len];
    // (one key, filled again for each look: nothing is allocated once it has held the longest pair)
    let mut key = (String::new(), String::new());
    let mut try_add = |heap: &mut BinaryHeap<Reverse<(u32, usize, usize, usize)>>, syms: &[Symbol], left: usize, right: usize| {
        if left == NONE || right == NONE {
            return;
        }
        let (l, r) = (&syms[left], &syms[right]);
        key.0.clear();
        key.0.push_str(text(l));
        key.1.clear();
        key.1.push_str(text(r));
        if let Some(&rank) = ranks.get(&key) {
            heap.push(Reverse((rank, left, right, l.len + r.len)));
        }
    };
    let mut heap = BinaryHeap::with_capacity(n);
    for i in 1..n {
        try_add(&mut heap, &syms, i - 1, i);
    }
    while let Some(Reverse((_, l, r, len))) = heap.pop() {
        // a pair one of whose symbols has since merged with another is stale
        if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != len {
            continue;
        }
        syms[l].len += syms[r].len;
        syms[r].len = 0;
        let after = syms[r].next;
        syms[l].next = after;
        if after != NONE {
            syms[after].prev = l;
        }
        try_add(&mut heap, &syms, syms[l].prev, l);
        try_add(&mut heap, &syms, l, after);
    }
    let mut i = 0;
    while i != NONE {
        if let Some(id) = tok.id_of(text(&syms[i])) {
            out.push(id);
        } else if let Some(unk) = tok.unk() {
            out.push(unk);
        } else {
            return Err(TokenizerError::MissingMetadata(
                "BPE produced an out-of-vocab piece and no unk token is defined",
            ));
        }
        i = syms[i].next;
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
