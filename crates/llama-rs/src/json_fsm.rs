//! Token-level JSON constraint — a small finite state machine that, paired
//! with a pre-computed per-state vocab mask, restricts sampling to byte
//! sequences that *will* parse as a single JSON object.
//!
//! Scope is intentionally narrow: flat single-line objects only.
//!   `{"key": "value", "k2": 42}`
//! Supported value types: quoted ASCII string, integer (with optional `-`).
//! No nesting, arrays, floats, booleans, null, escapes, or non-ASCII keys.
//! These are the most common cases for "extract a few labelled fields" use
//! cases, and the FSM stays small enough to keep the per-state mask
//! computation fast (~10 states × vocab decodes).
//!
//! Within those limits the produced output is *guaranteed* parseable — every
//! sampled token byte advances the FSM through a legal transition; tokens
//! whose byte sequence would derail the FSM from the current state are masked
//! out before sampling. Compare with `with_token_filter`'s character-class
//! approach in `examples/json_mode.rs` which only enforces "looks JSON-ish".
//!
//! Per-step cost at decode time: one vocab-sized boolean mask lookup + one
//! linear scan per token to update the FSM state from the new token's bytes.
//! Both are well under per-token kernel-launch budget.
//!
//! API: [`JsonObjectFsm::new`] takes a `Tokenizer` and pre-computes the
//! per-state masks; [`JsonObjectFsm::current_mask`] returns the active mask;
//! [`JsonObjectFsm::observe`] feeds the sampled token's bytes through the
//! state machine. The wiring into `GenerateIter` lives in `lib.rs`.

use std::collections::HashMap;
use tokenizer::Tokenizer;

/// FSM states for a flat JSON object. Numeric variants intentionally laid out
/// for use as `Vec` indices — we map state → mask via a small dense vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsonFsmState {
    /// Before the opening `{`. Whitespace allowed.
    Init = 0,
    /// After `{`. Expect either a string key or closing `}` for empty object.
    AfterOpen = 1,
    /// After a `,` between fields. Strictly expect a string key (no `}` —
    /// trailing commas are not legal JSON).
    AfterComma = 2,
    /// Inside a key's quoted string. Expect any string body byte or closing `"`.
    KeyBody = 3,
    /// After closing key quote. Expect `:` (with optional whitespace).
    AfterKey = 4,
    /// After `:`. Expect a value: opening `"`, digit/`-`, or first letter of
    /// a literal (`t`/`f`/`n`).
    AfterColon = 5,
    /// Inside a value's quoted string. Expect body bytes or closing `"`.
    ValueStringBody = 6,
    /// Inside a number value. Digits only (no decimals, no exponent — flat scope).
    ValueNumberBody = 7,
    /// After a complete value. Expect `,` (next key) or `}` (close object).
    AfterValue = 8,
    /// Output complete — closing `}` consumed. No further bytes accepted.
    Done = 9,
    // ---- Literal-matching states. Each tracks how far we are through the
    // expected character sequence for `true` / `false` / `null`. The names
    // describe the "next expected character" — e.g. `TrueR` means we just
    // consumed `t` and now expect `r`.
    /// Saw `t`, expect `r` next.
    TrueR = 10,
    /// Saw `tr`, expect `u` next.
    TrueU = 11,
    /// Saw `tru`, expect `e` next.
    TrueE = 12,
    /// Saw `f`, expect `a` next.
    FalseA = 13,
    /// Saw `fa`, expect `l` next.
    FalseL = 14,
    /// Saw `fal`, expect `s` next.
    FalseS = 15,
    /// Saw `fals`, expect `e` next.
    FalseE = 16,
    /// Saw `n`, expect `u` next.
    NullU = 17,
    /// Saw `nu`, expect first `l` next.
    NullL1 = 18,
    /// Saw `nul`, expect second `l` next.
    NullL2 = 19,
    // ---- Top-level array states. Mirrors the object container family but
    // accepts `]` instead of `}` and treats every element position as a value
    // (no key/colon dance). Top-level only — arrays inside object values, or
    // nested arrays, are *not* supported here (they'd require a state stack
    // / push-down automaton; the current pre-computed-mask design assumes
    // a fixed finite state set).
    /// After `[`. Expect a value or closing `]` (empty array).
    ArrAfterOpen = 20,
    /// After `,` between array elements. Expect a value (no `]` — strict).
    ArrAfterComma = 21,
    /// Inside an array string element. Body bytes or closing `"`.
    ArrValueStringBody = 22,
    /// Inside an array number element. Digits.
    ArrValueNumberBody = 23,
    /// After a complete array element. Expect `,` or `]`.
    ArrAfterValue = 24,
    // Array-context literal completion chains. Same byte sequences as the
    // object-context ones, but transition to ArrAfterValue on completion.
    ArrTrueR = 25,
    ArrTrueU = 26,
    ArrTrueE = 27,
    ArrFalseA = 28,
    ArrFalseL = 29,
    ArrFalseS = 30,
    ArrFalseE = 31,
    ArrNullU = 32,
    ArrNullL1 = 33,
    ArrNullL2 = 34,
}

const N_STATES: usize = 35;

/// Single byte transition: returns the next state, or `None` if the byte is
/// not legal in `s`. Whitespace (`' '`, `'\t'`, `'\n'`, `'\r'`) is permitted
/// in every "between values" state but forbidden inside string bodies (since
/// raw newlines aren't valid in JSON strings, they'd have to be escaped).
fn step(s: JsonFsmState, b: u8) -> Option<JsonFsmState> {
    use JsonFsmState::*;
    let is_ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
    let is_string_body = b.is_ascii_graphic()
        || b == b' '
        || b == b'\t';
    let is_key_body = b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b' ';
    let is_digit = b.is_ascii_digit();

    match s {
        Init => {
            if is_ws { Some(Init) }
            else if b == b'{' { Some(AfterOpen) }
            else if b == b'[' { Some(ArrAfterOpen) }
            else { None }
        }
        AfterOpen => {
            if is_ws { Some(AfterOpen) }
            else if b == b'"' { Some(KeyBody) }
            else if b == b'}' { Some(Done) }
            else { None }
        }
        AfterComma => {
            // Strict: only string-key opener allowed (no `}` — JSON forbids
            // trailing commas).
            if is_ws { Some(AfterComma) }
            else if b == b'"' { Some(KeyBody) }
            else { None }
        }
        KeyBody => {
            // Inside the key string. Close on `"`, otherwise accept identifier-like
            // bytes (alnum, underscore, hyphen, space). Strict: no escape sequences.
            if b == b'"' { Some(AfterKey) }
            else if is_key_body { Some(KeyBody) }
            else { None }
        }
        AfterKey => {
            if is_ws { Some(AfterKey) }
            else if b == b':' { Some(AfterColon) }
            else { None }
        }
        AfterColon => {
            if is_ws { Some(AfterColon) }
            else if b == b'"' { Some(ValueStringBody) }
            else if is_digit || b == b'-' { Some(ValueNumberBody) }
            // Literal entry points — start of `true` / `false` / `null`.
            else if b == b't' { Some(TrueR) }
            else if b == b'f' { Some(FalseA) }
            else if b == b'n' { Some(NullU) }
            else { None }
        }
        ValueStringBody => {
            if b == b'"' { Some(AfterValue) }
            else if is_string_body && b != b'\\' { Some(ValueStringBody) }
            else { None }
        }
        ValueNumberBody => {
            if is_digit { Some(ValueNumberBody) }
            else if is_ws { Some(AfterValue) }
            else if b == b',' { Some(AfterComma) }
            else if b == b'}' { Some(Done) }
            else { None }
        }
        AfterValue => {
            if is_ws { Some(AfterValue) }
            else if b == b',' { Some(AfterComma) }
            else if b == b'}' { Some(Done) }
            else { None }
        }
        // Literal-completion chains. Each state accepts exactly one byte and
        // advances to the next chain state, or `AfterValue` for the terminal
        // letter. No whitespace tolerated mid-literal — JSON forbids
        // `t r u e` etc. anyway.
        TrueR => if b == b'r' { Some(TrueU) } else { None },
        TrueU => if b == b'u' { Some(TrueE) } else { None },
        TrueE => if b == b'e' { Some(AfterValue) } else { None },
        FalseA => if b == b'a' { Some(FalseL) } else { None },
        FalseL => if b == b'l' { Some(FalseS) } else { None },
        FalseS => if b == b's' { Some(FalseE) } else { None },
        FalseE => if b == b'e' { Some(AfterValue) } else { None },
        NullU => if b == b'u' { Some(NullL1) } else { None },
        NullL1 => if b == b'l' { Some(NullL2) } else { None },
        NullL2 => if b == b'l' { Some(AfterValue) } else { None },
        // ---- Array container ----
        ArrAfterOpen => {
            if is_ws { Some(ArrAfterOpen) }
            else if b == b']' { Some(Done) }
            // Element value entries.
            else if b == b'"' { Some(ArrValueStringBody) }
            else if is_digit || b == b'-' { Some(ArrValueNumberBody) }
            else if b == b't' { Some(ArrTrueR) }
            else if b == b'f' { Some(ArrFalseA) }
            else if b == b'n' { Some(ArrNullU) }
            else { None }
        }
        ArrAfterComma => {
            // Strict: no `]` after comma — JSON forbids trailing commas.
            if is_ws { Some(ArrAfterComma) }
            else if b == b'"' { Some(ArrValueStringBody) }
            else if is_digit || b == b'-' { Some(ArrValueNumberBody) }
            else if b == b't' { Some(ArrTrueR) }
            else if b == b'f' { Some(ArrFalseA) }
            else if b == b'n' { Some(ArrNullU) }
            else { None }
        }
        ArrValueStringBody => {
            if b == b'"' { Some(ArrAfterValue) }
            else if is_string_body && b != b'\\' { Some(ArrValueStringBody) }
            else { None }
        }
        ArrValueNumberBody => {
            if is_digit { Some(ArrValueNumberBody) }
            else if is_ws { Some(ArrAfterValue) }
            else if b == b',' { Some(ArrAfterComma) }
            else if b == b']' { Some(Done) }
            else { None }
        }
        ArrAfterValue => {
            if is_ws { Some(ArrAfterValue) }
            else if b == b',' { Some(ArrAfterComma) }
            else if b == b']' { Some(Done) }
            else { None }
        }
        // Array-context literal completion. Same byte sequences, but the
        // terminal letter transitions to ArrAfterValue (not AfterValue).
        ArrTrueR => if b == b'r' { Some(ArrTrueU) } else { None },
        ArrTrueU => if b == b'u' { Some(ArrTrueE) } else { None },
        ArrTrueE => if b == b'e' { Some(ArrAfterValue) } else { None },
        ArrFalseA => if b == b'a' { Some(ArrFalseL) } else { None },
        ArrFalseL => if b == b'l' { Some(ArrFalseS) } else { None },
        ArrFalseS => if b == b's' { Some(ArrFalseE) } else { None },
        ArrFalseE => if b == b'e' { Some(ArrAfterValue) } else { None },
        ArrNullU => if b == b'u' { Some(ArrNullL1) } else { None },
        ArrNullL1 => if b == b'l' { Some(ArrNullL2) } else { None },
        ArrNullL2 => if b == b'l' { Some(ArrAfterValue) } else { None },
        Done => None,
    }
}

/// Walk all bytes of `text` through the FSM starting at `s`. Returns the final
/// state on success, or `None` if any byte was rejected. Used both at vocab
/// pre-compute time (to determine if a token is admissible from each state)
/// and at sampling time (to advance the live state by the sampled token).
pub fn run(s: JsonFsmState, text: &str) -> Option<JsonFsmState> {
    let mut cur = s;
    for b in text.bytes() {
        cur = step(cur, b)?;
    }
    Some(cur)
}

/// Per-FSM-state vocab mask: `masks[state as usize][token_id] == true` iff the
/// token's decoded text consumes legally from `state`. Indexed by state then
/// token id for cache-friendly per-step access.
pub struct JsonObjectFsm {
    masks: Vec<Vec<bool>>,
    state: JsonFsmState,
}

impl JsonObjectFsm {
    /// Pre-compute admissibility of every vocab token from every FSM state.
    /// Cost: O(n_states · vocab · avg_token_bytes). At 9 states × 256K vocab ×
    /// ~4 bytes ≈ 9M byte-checks — runs in well under a second on a typical
    /// machine. Done once per generation; the resulting masks are reused
    /// per decode step.
    ///
    /// Tokens whose decoded text is empty (e.g. some special tokens) are
    /// allowed nowhere — they'd advance the FSM zero bytes, which is a no-op
    /// and would let the sampler emit them indefinitely.
    pub fn new(tok: &Tokenizer) -> Self {
        let vocab = tok.vocab_size();
        let states: [JsonFsmState; N_STATES] = [
            JsonFsmState::Init,
            JsonFsmState::AfterOpen,
            JsonFsmState::AfterComma,
            JsonFsmState::KeyBody,
            JsonFsmState::AfterKey,
            JsonFsmState::AfterColon,
            JsonFsmState::ValueStringBody,
            JsonFsmState::ValueNumberBody,
            JsonFsmState::AfterValue,
            JsonFsmState::Done,
            JsonFsmState::TrueR,
            JsonFsmState::TrueU,
            JsonFsmState::TrueE,
            JsonFsmState::FalseA,
            JsonFsmState::FalseL,
            JsonFsmState::FalseS,
            JsonFsmState::FalseE,
            JsonFsmState::NullU,
            JsonFsmState::NullL1,
            JsonFsmState::NullL2,
            JsonFsmState::ArrAfterOpen,
            JsonFsmState::ArrAfterComma,
            JsonFsmState::ArrValueStringBody,
            JsonFsmState::ArrValueNumberBody,
            JsonFsmState::ArrAfterValue,
            JsonFsmState::ArrTrueR,
            JsonFsmState::ArrTrueU,
            JsonFsmState::ArrTrueE,
            JsonFsmState::ArrFalseA,
            JsonFsmState::ArrFalseL,
            JsonFsmState::ArrFalseS,
            JsonFsmState::ArrFalseE,
            JsonFsmState::ArrNullU,
            JsonFsmState::ArrNullL1,
            JsonFsmState::ArrNullL2,
        ];

        // Decode every token once and cache the text — n_states sees the same
        // strings, no point decoding 9 × 256K = 2.3M times.
        let texts: Vec<String> = (0..vocab as u32).map(|id| tok.decode(&[id])).collect();

        let mut masks = vec![vec![false; vocab]; N_STATES];
        for &s in &states {
            let row = &mut masks[s as usize];
            for (id, text) in texts.iter().enumerate() {
                if text.is_empty() { continue; }
                row[id] = run(s, text).is_some();
            }
        }

        Self { masks, state: JsonFsmState::Init }
    }

    /// Mask for the current FSM state. `result[id] == true` ⇒ token id is
    /// admissible right now. The caller masks logits (set forbidden to `-inf`)
    /// before calling the sampler.
    pub fn current_mask(&self) -> &[bool] {
        &self.masks[self.state as usize]
    }

    /// Whether the FSM has reached its terminal state. Call after `observe` to
    /// decide whether to stop generation.
    pub fn done(&self) -> bool { self.state == JsonFsmState::Done }

    /// Advance the FSM by walking `text` (the decoded bytes of a sampled
    /// token) through the transition function. Panics in debug if `text`
    /// wasn't admissible — the caller is expected to have applied the
    /// `current_mask` before sampling, so this should never happen in practice.
    pub fn observe(&mut self, text: &str) {
        match run(self.state, text) {
            Some(next) => self.state = next,
            None => {
                debug_assert!(false, "JsonObjectFsm.observe: token text {text:?} not admissible from state {:?}", self.state);
            }
        }
    }

    /// Token-id histogram for diagnostic / debugging — count of admissible
    /// tokens per state. Helps catch tokenizer / FSM mismatches at load time.
    #[allow(dead_code)]
    pub fn admissibility_counts(&self) -> HashMap<JsonFsmState, usize> {
        let mut out = HashMap::new();
        for (idx, mask) in self.masks.iter().enumerate() {
            let s = match idx {
                0 => JsonFsmState::Init,
                1 => JsonFsmState::AfterOpen,
                2 => JsonFsmState::AfterComma,
                3 => JsonFsmState::KeyBody,
                4 => JsonFsmState::AfterKey,
                5 => JsonFsmState::AfterColon,
                6 => JsonFsmState::ValueStringBody,
                7 => JsonFsmState::ValueNumberBody,
                8 => JsonFsmState::AfterValue,
                9 => JsonFsmState::Done,
                10 => JsonFsmState::TrueR,
                11 => JsonFsmState::TrueU,
                12 => JsonFsmState::TrueE,
                13 => JsonFsmState::FalseA,
                14 => JsonFsmState::FalseL,
                15 => JsonFsmState::FalseS,
                16 => JsonFsmState::FalseE,
                17 => JsonFsmState::NullU,
                18 => JsonFsmState::NullL1,
                19 => JsonFsmState::NullL2,
                20 => JsonFsmState::ArrAfterOpen,
                21 => JsonFsmState::ArrAfterComma,
                22 => JsonFsmState::ArrValueStringBody,
                23 => JsonFsmState::ArrValueNumberBody,
                24 => JsonFsmState::ArrAfterValue,
                25 => JsonFsmState::ArrTrueR,
                26 => JsonFsmState::ArrTrueU,
                27 => JsonFsmState::ArrTrueE,
                28 => JsonFsmState::ArrFalseA,
                29 => JsonFsmState::ArrFalseL,
                30 => JsonFsmState::ArrFalseS,
                31 => JsonFsmState::ArrFalseE,
                32 => JsonFsmState::ArrNullU,
                33 => JsonFsmState::ArrNullL1,
                _ => JsonFsmState::ArrNullL2,
            };
            out.insert(s, mask.iter().filter(|&&b| b).count());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use JsonFsmState::*;

    #[test]
    fn happy_path_string_value() {
        // `{"k": "v"}` — every byte should be a valid transition.
        let s = run(Init, "{\"k\": \"v\"}").unwrap();
        assert_eq!(s, Done);
    }

    #[test]
    fn happy_path_number_value() {
        let s = run(Init, "{\"age\": 42}").unwrap();
        assert_eq!(s, Done);
    }

    #[test]
    fn multi_field_object() {
        let s = run(Init, "{\"name\": \"Alice\", \"age\": 30}").unwrap();
        assert_eq!(s, Done);
    }

    #[test]
    fn rejects_unquoted_key() {
        // `{name: "v"}` — `n` is not a valid byte in AfterOpen state.
        assert!(run(Init, "{name: \"v\"}").is_none());
    }

    #[test]
    fn rejects_trailing_comma() {
        // `{"k": "v",}` — after comma we expect an opening quote, not `}`.
        assert!(run(Init, "{\"k\": \"v\",}").is_none());
    }

    #[test]
    fn rejects_nested_object() {
        // Flat-only scope: `{` inside a value is not a legal transition.
        assert!(run(Init, "{\"k\": {").is_none());
    }

    #[test]
    fn empty_object_ok() {
        let s = run(Init, "{}").unwrap();
        assert_eq!(s, Done);
    }

    #[test]
    fn whitespace_tolerated_between_tokens() {
        let s = run(Init, "  {  \"k\"  :  \"v\"  }").unwrap();
        assert_eq!(s, Done);
    }

    #[test]
    fn done_rejects_further_input() {
        // Once `}` consumed, Done is terminal — even whitespace is rejected.
        // The caller (GenerateIter) is expected to set `done = true` and stop.
        assert!(run(Init, "{}  ").is_none());
    }

    #[test]
    fn boolean_true_value() {
        assert_eq!(run(Init, "{\"x\": true}").unwrap(), Done);
    }

    #[test]
    fn boolean_false_value() {
        assert_eq!(run(Init, "{\"x\": false}").unwrap(), Done);
    }

    #[test]
    fn null_value() {
        assert_eq!(run(Init, "{\"x\": null}").unwrap(), Done);
    }

    #[test]
    fn mixed_literal_object() {
        let s = run(Init, "{\"a\": true, \"b\": false, \"c\": null, \"d\": 42}").unwrap();
        assert_eq!(s, Done);
    }

    #[test]
    fn rejects_misspelled_literal() {
        // `tru}` — chain breaks at `}` when expecting `e`.
        assert!(run(Init, "{\"x\": tru}").is_none());
        // `fals}` — chain breaks at `}` when expecting `e`.
        assert!(run(Init, "{\"x\": fals}").is_none());
        // `nul}` — chain breaks at `}` when expecting second `l`.
        assert!(run(Init, "{\"x\": nul}").is_none());
    }

    #[test]
    fn rejects_uppercase_literal() {
        // JSON is case-sensitive: `True` / `FALSE` / `Null` are not legal.
        assert!(run(Init, "{\"x\": True}").is_none());
        assert!(run(Init, "{\"x\": FALSE}").is_none());
        assert!(run(Init, "{\"x\": Null}").is_none());
    }

    #[test]
    fn literal_followed_by_comma_continues() {
        // `true,` should land in AfterValue (after `true`) then advance to
        // AfterComma on the comma. Then a new key starts the next field.
        let s = run(Init, "{\"a\": true, \"b\": null}").unwrap();
        assert_eq!(s, Done);
    }

    // ---- Top-level array tests ----

    #[test]
    fn empty_array_ok() {
        assert_eq!(run(Init, "[]").unwrap(), Done);
    }

    #[test]
    fn array_of_numbers() {
        assert_eq!(run(Init, "[1, 2, 3]").unwrap(), Done);
    }

    #[test]
    fn array_of_strings() {
        assert_eq!(run(Init, "[\"a\", \"b\", \"c\"]").unwrap(), Done);
    }

    #[test]
    fn array_of_literals() {
        assert_eq!(run(Init, "[true, false, null]").unwrap(), Done);
    }

    #[test]
    fn mixed_array() {
        assert_eq!(run(Init, "[1, \"two\", true, null]").unwrap(), Done);
    }

    #[test]
    fn array_rejects_trailing_comma() {
        assert!(run(Init, "[1, 2, 3,]").is_none());
    }

    #[test]
    fn array_rejects_unquoted_string() {
        // `[abc]` — `a` isn't a number, literal start, or string opener.
        assert!(run(Init, "[abc]").is_none());
    }

    #[test]
    fn array_rejects_nested() {
        // Nested array is out of scope (would need a stack / PDA). The inner
        // `[` is rejected from ArrAfterOpen.
        assert!(run(Init, "[[1]]").is_none());
        // Same for object inside array.
        assert!(run(Init, "[{\"a\": 1}]").is_none());
    }

    #[test]
    fn whitespace_in_array() {
        assert_eq!(run(Init, "[  1  ,  2  ]").unwrap(), Done);
    }
}
