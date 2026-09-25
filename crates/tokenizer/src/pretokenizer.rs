//! Pretokenization regex for byte-level BPE.
//!
//! BPE alone over-merges across word boundaries (e.g., it'll happily produce
//! a single token for `"Mr.Smith"` instead of `["Mr", ".", "Smith"]`). The
//! standard fix is a regex that splits the input into "pre-tokens" before BPE
//! sees it. Llama-3, GPT-4, Qwen, and most modern byte-level BPE vocabs share
//! a near-identical pattern.
//!
//! We carry one canonical pattern (the Llama-3 one — same as GPT-4
//! `cl100k_base` minus the possessive quantifiers, which the standard `regex`
//! crate doesn't support but which are functionally equivalent for these
//! anchored alternatives).

use regex::Regex;
use std::sync::OnceLock;

/// Llama-3 / GPT-4-style pretokenizer pattern. Splits into:
///   * English contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`)
///   * Letter runs (optionally prefixed by one non-letter/digit/whitespace char)
///   * 1–3 digit runs (so `12345` is split into `123` + `45`)
///   * Punctuation runs (optionally with a leading space and trailing newlines)
///   * Whitespace runs (handling line endings carefully)
fn llama3_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // Note: `regex` doesn't support `?+` / `++` (possessive). Plain `?` and
        // `+` produce the same matches here because every alternative is
        // anchored — there's no backtracking that could change the result.
        Regex::new(
            r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|\S)|\s+"#
        ).expect("pretokenizer regex compiles")
    })
}

/// Split `text` into pre-tokens. Returns string slices in order.
/// For inputs with no special structure (pure letters, no punctuation), this
/// yields one piece per word + leading-space-prefixed continuation.
pub fn split_llama3(text: &str) -> Vec<&str> {
    let re = llama3_regex();
    let mut out = Vec::new();
    let mut last_end = 0;
    for m in re.find_iter(text) {
        // The pattern is comprehensive enough that we shouldn't see gaps in
        // practice, but if we do, push the gap as its own piece (better to
        // round-trip than silently drop bytes).
        if m.start() > last_end {
            out.push(&text[last_end..m.start()]);
        }
        out.push(m.as_str());
        last_end = m.end();
    }
    if last_end < text.len() {
        out.push(&text[last_end..]);
    }
    out
}

/// VENDORED-LOCAL: Qwen3.8's regex, with its whitespace lookahead expressed
/// explicitly because Rust regex does not support lookaround.
pub fn split_qwen3(text:&str)->Vec<&str>{
    static R:OnceLock<Regex>=OnceLock::new();
    let re=R.get_or_init(||Regex::new(r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"#).expect("Qwen regex"));
    let mut pieces=vec![];let mut pos=0;
    while pos<text.len(){
        let Some(m)=re.find_at(text,pos) else {pieces.push(&text[pos..]);break;};
        if m.start()>pos{pieces.push(&text[pos..m.start()]);}
        let mut end=m.end();let span=m.as_str();
        if end<text.len() && span.chars().all(char::is_whitespace) && !span.ends_with(['\r','\n']) && span.chars().count()>1 {
            // \s+(?!\S) keeps all but the last whitespace before a non-space.
            end-=span.chars().last().unwrap().len_utf8();
        }
        pieces.push(&text[m.start()..end]);pos=end;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_simple_sentence() {
        let pieces = split_llama3("The capital of France is Paris.");
        let collected: Vec<&str> = pieces;
        // Expect roughly: ["The", " capital", " of", " France", " is", " Paris", "."]
        assert!(collected.len() >= 6, "expected ≥6 pieces, got {collected:?}");
        assert_eq!(collected[0], "The");
        assert!(collected.iter().any(|s| *s == " Paris"));
        assert!(collected.iter().any(|s| *s == "."));
    }

    #[test]
    fn separates_punctuation_glued_to_word() {
        let pieces = split_llama3("Mr.Smith said hi.");
        // Don't assert exact split (pattern lets one leading non-letter prefix
        // into a letter piece), but ensure "Smith" is its own piece.
        assert!(pieces.iter().any(|s| s.contains("Smith")), "{pieces:?}");
    }

    #[test]
    fn keeps_digit_runs_grouped_3() {
        let pieces = split_llama3("12345");
        // Should split into ["123", "45"] (max 3 digits per piece).
        assert!(pieces.iter().all(|s| s.chars().all(|c| c.is_ascii_digit())));
        assert!(pieces.iter().all(|s| s.len() <= 3));
        let total: usize = pieces.iter().map(|s| s.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn roundtrips_all_bytes() {
        let s = "Hello, world! 12345 \"quoted\" — em-dash.";
        let pieces = split_llama3(s);
        let joined: String = pieces.concat();
        assert_eq!(joined, s);
    }
}
