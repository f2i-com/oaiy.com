//! Text to token ids for DeepSeek-V4.1, from the checkpoint's
//! `tokenizer.json` (byte-level BPE). It reproduces the Hugging Face
//! pipeline the checkpoint declares, exactly:
//!
//! 1. **Added tokens** (`<｜User｜>`, `<think>`, …) are cut out of the text
//!    first as literal strings, leftmost-longest: those marked
//!    `normalized: false` in one pass, then the `normalized: true` ones in
//!    what is left. (The normalizer is empty, so no text changes.)
//! 2. **Pre-tokenization** splits each remaining span three times, each
//!    split keeping its matches and the gaps between them as separate
//!    pieces: runs of up to 3 digits, then runs of CJK ideographs and kana,
//!    then the main word/punctuation/whitespace pattern (see [`main_match`]).
//! 3. **BPE**: each piece's UTF-8 bytes become byte-level symbols, merged
//!    pair by pair in rank order (the lowest-ranked pair first; of equal
//!    pairs, the leftmost).
//!
//! The pre-tokenizer patterns use Unicode general categories, from the
//! generated table in [`crate::unicode`].

use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use nrob::json::Json;
use nrob::{Error, Result};

use crate::unicode::{class, Class};

/// A trie node: children as (byte, node), and the token id if one ends here.
type Node = (Vec<(u8, u32)>, Option<u32>);

/// A byte-trie over added-token strings, for leftmost-longest matching.
#[derive(Default)]
struct Trie {
    nodes: Vec<Node>,
}

impl Trie {
    fn new() -> Trie {
        Trie { nodes: vec![(Vec::new(), None)] }
    }

    fn insert(&mut self, s: &str, id: u32) {
        let mut n = 0usize;
        for &b in s.as_bytes() {
            n = match self.nodes[n].0.iter().find(|(c, _)| *c == b) {
                Some(&(_, next)) => next as usize,
                None => {
                    self.nodes.push((Vec::new(), None));
                    let next = self.nodes.len() - 1;
                    self.nodes[n].0.push((b, next as u32));
                    next
                }
            };
        }
        self.nodes[n].1 = Some(id);
    }

    /// The longest token starting at `bytes[0]`: (length, id).
    fn longest(&self, bytes: &[u8]) -> Option<(usize, u32)> {
        let (mut n, mut best) = (0usize, None);
        for (i, &b) in bytes.iter().enumerate() {
            match self.nodes[n].0.iter().find(|(c, _)| *c == b) {
                Some(&(_, next)) => n = next as usize,
                None => break,
            }
            if let Some(id) = self.nodes[n].1 {
                best = Some((i + 1, id));
            }
        }
        best
    }

    fn is_empty(&self) -> bool {
        self.nodes[0].0.is_empty()
    }
}

/// A stretch of the input: plain text, or an added token already resolved.
enum Span<'a> {
    Text(&'a str),
    Token(u32),
}

/// Cut `text` at every added token in `trie`, leftmost-longest.
fn cut<'a>(text: &'a str, trie: &Trie, out: &mut Vec<Span<'a>>) {
    if trie.is_empty() {
        out.push(Span::Text(text));
        return;
    }
    let bytes = text.as_bytes();
    let (mut pos, mut plain) = (0usize, 0usize);
    while pos < bytes.len() {
        match trie.longest(&bytes[pos..]) {
            Some((len, id)) => {
                if plain < pos {
                    out.push(Span::Text(&text[plain..pos]));
                }
                out.push(Span::Token(id));
                pos += len;
                plain = pos;
            }
            None => pos += 1,
        }
    }
    if plain < bytes.len() {
        out.push(Span::Text(&text[plain..]));
    }
}

pub struct Tokenizer {
    /// Byte-level symbol string -> id.
    vocab: HashMap<String, u32>,
    /// (left, right) -> (rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Id of each single byte's symbol.
    byte_id: [u32; 256],
    /// Added tokens matched before normalization, then after.
    raw_added: Trie,
    normalized_added: Trie,
    /// Bytes each id stands for (decoding).
    bytes: Vec<Vec<u8>>,
    /// Added-token ids by content.
    added_ids: HashMap<String, u32>,
    /// Pieces seen before, and their ids.
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

/// GPT-2's byte-to-character table: printable bytes stand for themselves,
/// the rest map to U+0100 upwards in byte order.
pub fn byte_chars() -> [char; 256] {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u8 {
        let cp = if printable(b) {
            b as u32
        } else {
            next += 1;
            next - 1
        };
        table[b as usize] = char::from_u32(cp).expect("valid scalar");
    }
    table
}

impl Tokenizer {
    /// Load `<model_dir>/tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Tokenizer> {
        Self::parse(&std::fs::read(model_dir.join("tokenizer.json"))?)
    }

    pub fn parse(src: &[u8]) -> Result<Tokenizer> {
        let bad = |m: &str| Error::Format(format!("tokenizer.json: {m}"));
        let root = Json::parse(src)?;
        let model = root.get("model").ok_or_else(|| bad("no model"))?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            return Err(bad("model is not BPE"));
        }
        let vocab_json = model.get("vocab").and_then(Json::as_object).ok_or_else(|| bad("no model.vocab"))?;
        let mut vocab = HashMap::with_capacity(vocab_json.len());
        let mut size = 0usize;
        for (k, v) in vocab_json {
            let id = v.as_i64().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| bad("bad vocab id"))?;
            vocab.insert(k.clone(), id);
            size = size.max(id as usize + 1);
        }

        let chars = byte_chars();
        let mut byte_id = [0u32; 256];
        for (b, c) in chars.iter().enumerate() {
            byte_id[b] = *vocab.get(&c.to_string()).ok_or_else(|| bad("a byte has no token"))?;
        }

        let merges_json = model.get("merges").and_then(Json::as_array).ok_or_else(|| bad("no model.merges"))?;
        let mut merges = HashMap::with_capacity(merges_json.len());
        for (rank, m) in merges_json.iter().enumerate() {
            let (l, r) = match m {
                Json::Str(s) => s.split_once(' ').ok_or_else(|| bad("merge without a space"))?,
                Json::Arr(pair) if pair.len() == 2 => (
                    pair[0].as_str().ok_or_else(|| bad("bad merge"))?,
                    pair[1].as_str().ok_or_else(|| bad("bad merge"))?,
                ),
                _ => return Err(bad("bad merge")),
            };
            let id = |s: &str| vocab.get(s).copied().ok_or_else(|| bad(&format!("merge part {s:?} not in the vocabulary")));
            let merged = vocab.get(&format!("{l}{r}")).copied().ok_or_else(|| bad("merge result not in the vocabulary"))?;
            // the first (lowest-ranked) occurrence of a pair wins
            merges.entry((id(l)?, id(r)?)).or_insert((rank as u32, merged));
        }

        let mut raw_added = Trie::new();
        let mut normalized_added = Trie::new();
        let mut added_ids = HashMap::new();
        let mut added_bytes = Vec::new();
        for a in root.get("added_tokens").and_then(Json::as_array).unwrap_or(&[]) {
            let content = a.get("content").and_then(Json::as_str).ok_or_else(|| bad("added token without content"))?;
            let id = a.get("id").and_then(Json::as_i64).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| bad("bad added token id"))?;
            if content.is_empty() {
                continue;
            }
            if a.get("normalized").and_then(Json::as_bool).unwrap_or(true) {
                normalized_added.insert(content, id);
            } else {
                raw_added.insert(content, id);
            }
            added_ids.insert(content.to_string(), id);
            added_bytes.push((id, content.as_bytes().to_vec()));
            size = size.max(id as usize + 1);
        }

        let unchar: HashMap<char, u8> = chars.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let mut bytes = vec![Vec::new(); size];
        for (s, &id) in &vocab {
            bytes[id as usize] = s.chars().map(|c| unchar.get(&c).copied().unwrap_or(b'?')).collect();
        }
        for (id, b) in added_bytes {
            bytes[id as usize] = b;
        }

        Ok(Tokenizer {
            vocab,
            merges,
            byte_id,
            raw_added,
            normalized_added,
            bytes,
            added_ids,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Ids a model with this tokenizer can see.
    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }

    /// The id of an added (special) token such as `<｜User｜>`.
    pub fn special(&self, content: &str) -> Option<u32> {
        self.added_ids.get(content).copied()
    }

    /// The bytes token `id` stands for (empty if unknown). A token can end
    /// inside a UTF-8 sequence.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Decode a whole sequence (invalid UTF-8 replaced).
    pub fn decode(&self, ids: &[u32]) -> String {
        let b: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id).iter().copied()).collect();
        String::from_utf8_lossy(&b).into_owned()
    }

    /// Token ids for `text`. No BOS is added: the chat encoding writes it
    /// out as text.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut raw = Vec::new();
        cut(text, &self.raw_added, &mut raw);
        let mut spans = Vec::new();
        for s in raw {
            match s {
                Span::Text(t) => cut(t, &self.normalized_added, &mut spans),
                tok => spans.push(tok),
            }
        }
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut pieces = Vec::new();
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        for span in spans {
            match span {
                Span::Token(id) => out.push(id),
                Span::Text(t) => {
                    pieces.clear();
                    pretokenize(t, &mut pieces);
                    for &p in &pieces {
                        if let Some(ids) = cache.get(p) {
                            out.extend_from_slice(ids);
                            continue;
                        }
                        let ids = self.bpe(p);
                        out.extend_from_slice(&ids);
                        if cache.len() < 200_000 {
                            cache.insert(p.to_string(), ids);
                        }
                    }
                }
            }
        }
        out
    }

    /// Merge one piece's bytes, the lowest-ranked pair first.
    fn bpe(&self, piece: &str) -> Vec<u32> {
        #[derive(Clone, Copy)]
        struct Sym {
            id: u32,
            prev: usize,
            next: usize,
            live: bool,
        }
        const NONE: usize = usize::MAX;
        let bytes = piece.as_bytes();
        if bytes.len() == 1 {
            return vec![self.byte_id[bytes[0] as usize]];
        }
        let mut syms: Vec<Sym> = bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| Sym {
                id: self.byte_id[b as usize],
                prev: if i == 0 { NONE } else { i - 1 },
                next: if i + 1 == bytes.len() { NONE } else { i + 1 },
                live: true,
            })
            .collect();
        // min-heap on (rank, position) via Reverse ordering
        let mut heap: BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<_>, syms: &[Sym], i: usize| {
            let j = syms[i].next;
            if j != NONE {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i].id, syms[j].id)) {
                    heap.push(std::cmp::Reverse((rank, i, merged)));
                }
            }
        };
        for i in 0..syms.len() {
            push(&mut heap, &syms, i);
        }
        while let Some(std::cmp::Reverse((_, i, merged))) = heap.pop() {
            if !syms[i].live {
                continue;
            }
            let j = syms[i].next;
            if j == NONE {
                continue;
            }
            // stale if the pair changed since it was queued
            match self.merges.get(&(syms[i].id, syms[j].id)) {
                Some(&(_, m)) if m == merged => {}
                _ => continue,
            }
            syms[i].id = merged;
            syms[j].live = false;
            let after = syms[j].next;
            syms[i].next = after;
            if after != NONE {
                syms[after].prev = i;
            }
            let before = syms[i].prev;
            if before != NONE {
                push(&mut heap, &syms, before);
            }
            push(&mut heap, &syms, i);
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i != NONE {
            out.push(syms[i].id);
            i = syms[i].next;
        }
        out
    }

    /// Whether `s` is a single vocabulary symbol (byte-level form).
    pub fn has_symbol(&self, s: &str) -> bool {
        self.vocab.contains_key(s)
    }
}

/// Split `s` by `matcher` (an anchored match at a byte offset, returning
/// its end), keeping both the matches and the gaps between them.
fn split_isolated<'a>(s: &'a str, matcher: fn(&str, usize) -> Option<usize>, out: &mut Vec<&'a str>) {
    let (mut pos, mut gap) = (0usize, 0usize);
    while pos < s.len() {
        match matcher(s, pos) {
            Some(end) if end > pos => {
                if gap < pos {
                    out.push(&s[gap..pos]);
                }
                out.push(&s[pos..end]);
                pos = end;
                gap = end;
            }
            _ => pos += s[pos..].chars().next().map_or(1, char::len_utf8),
        }
    }
    if gap < s.len() {
        out.push(&s[gap..]);
    }
}

fn pretokenize<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    split_isolated(text, digits_match, &mut a);
    for p in a {
        split_isolated(p, cjk_match, &mut b);
    }
    for p in b {
        split_isolated(p, main_match, out);
    }
}

/// Run of chars satisfying `f` starting at byte `pos`: its end.
fn run(s: &str, pos: usize, f: impl Fn(char) -> bool) -> usize {
    let mut end = pos;
    for c in s[pos..].chars() {
        if !f(c) {
            break;
        }
        end += c.len_utf8();
    }
    end
}

/// `\p{N}{1,3}`
fn digits_match(s: &str, pos: usize) -> Option<usize> {
    let mut end = pos;
    for c in s[pos..].chars().take(3) {
        if class(c) != Class::Number {
            break;
        }
        end += c.len_utf8();
    }
    (end > pos).then_some(end)
}

/// `[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+`
fn cjk_match(s: &str, pos: usize) -> Option<usize> {
    let end = run(s, pos, |c| matches!(c, '\u{4e00}'..='\u{9fa5}' | '\u{3040}'..='\u{309f}' | '\u{30a0}'..='\u{30ff}'));
    (end > pos).then_some(end)
}

fn is_letter_or_mark(c: char) -> bool {
    matches!(class(c), Class::Letter | Class::Mark)
}

fn is_punct_or_symbol(c: char) -> bool {
    matches!(class(c), Class::Punct | Class::Symbol)
}

/// The main pattern, alternatives tried in order at `pos`:
///
/// ```text
/// [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+   one ASCII punct + ASCII letters
/// |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+             a word, with one optional lead char
/// | ?[\p{P}\p{S}]+[\r\n]*                          punctuation, then line breaks
/// |\s*[\r\n]+                                      whitespace through the last line break
/// |\s+(?!\S)                                       whitespace not followed by text
/// |\s+                                             any other whitespace
/// ```
fn main_match(s: &str, pos: usize) -> Option<usize> {
    let mut it = s[pos..].chars();
    let c0 = it.next()?;
    let c1 = it.next();
    let after0 = pos + c0.len_utf8();

    // one ASCII punctuation mark, then ASCII letters
    if c0.is_ascii_punctuation() && c1.is_some_and(|c| c.is_ascii_alphabetic()) {
        return Some(run(s, after0, |c| c.is_ascii_alphabetic()));
    }
    // a word: optional lead char that is not a line break, letter,
    // punctuation or symbol; the lead is dropped if no word follows it
    if !matches!(c0, '\r' | '\n') && !matches!(class(c0), Class::Letter | Class::Punct | Class::Symbol) {
        let end = run(s, after0, is_letter_or_mark);
        if end > after0 {
            return Some(end);
        }
    }
    let end = run(s, pos, is_letter_or_mark);
    if end > pos {
        return Some(end);
    }
    // punctuation/symbols, optionally after one space, then line breaks
    let ps_from = |start: usize| {
        let end = run(s, start, is_punct_or_symbol);
        (end > start).then(|| run(s, end, |c| matches!(c, '\r' | '\n')))
    };
    if c0 == ' ' {
        if let Some(end) = ps_from(after0) {
            return Some(end);
        }
    }
    if let Some(end) = ps_from(pos) {
        return Some(end);
    }
    // whitespace
    let ws_end = run(s, pos, char::is_whitespace);
    if ws_end == pos {
        return None;
    }
    // through the last line break in the run
    if let Some(i) = s[pos..ws_end].rfind(['\r', '\n']) {
        return Some(pos + i + 1);
    }
    // not followed by text: the whole run at the end, else all but its
    // last char (which then leads the next word)
    if ws_end == s.len() {
        return Some(ws_end);
    }
    let last = s[pos..ws_end].chars().next_back().map_or(0, char::len_utf8);
    if ws_end - last > pos {
        return Some(ws_end - last);
    }
    Some(ws_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        pretokenize(s, &mut out);
        out
    }

    #[test]
    fn pretokenizer_shapes() {
        assert_eq!(pieces("Hello world"), ["Hello", " world"]);
        assert_eq!(pieces("hello   123"), ["hello", "   ", "123"]);
        assert_eq!(pieces("12345"), ["123", "45"]);
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("foo_bar.baz()"), ["foo", "_bar", ".baz", "()"]);
        assert_eq!(pieces("  \n\n  x"), ["  \n\n", " ", " x"]);
        assert_eq!(pieces("trailing   "), ["trailing", "   "]);
        assert_eq!(pieces("中文abc"), ["中文", "abc"]);
        assert_eq!(pieces("...!!!\n\nok"), ["...!!!\n\n", "ok"]);
        assert_eq!(pieces("\thi"), ["\thi"]);
    }

    fn model_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var("DSV41_MODEL").unwrap_or_else(|_| r"E:\deepseek\model".into()));
        dir.join("tokenizer.json").exists().then_some(dir)
    }

    /// Every case of the reference tokenizer's golden file encodes to the
    /// same ids (tools/dsv41/tokenizer_golden.py writes it).
    #[test]
    fn matches_reference_tokenizer() {
        let Some(dir) = model_dir() else { return };
        let golden = std::env::var("DSV41_TOKENIZER_GOLDEN").unwrap_or_else(|_| r"E:\deepseek\golden\tokenizer_cases.json".into());
        let Ok(src) = std::fs::read(&golden) else { return };
        let tok = Tokenizer::load(&dir).unwrap();
        let cases = Json::parse(&src).unwrap();
        let mut failures = Vec::new();
        for case in cases.as_array().unwrap() {
            let text = case.get("text").and_then(Json::as_str).unwrap();
            let want: Vec<u32> = case.get("ids").and_then(Json::as_array).unwrap().iter().map(|v| v.as_i64().unwrap() as u32).collect();
            let got = tok.encode(text);
            if got != want {
                failures.push(format!("{text:?}\n  want {want:?}\n  got  {got:?}"));
            }
            assert_eq!(tok.decode(&want), text, "decode round trip");
        }
        let n = cases.len();
        assert!(n > 3000, "golden file has only {n} cases");
        assert!(failures.is_empty(), "{} of {n} cases differ:\n{}", failures.len(), failures[..failures.len().min(8)].join("\n"));
        eprintln!("{n} reference cases match");
    }
}
