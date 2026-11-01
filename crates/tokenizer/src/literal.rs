//! Message text inside a prompt.
//!
//! A chat prompt is rendered to one string and tokenized whole, and in that string a control token's spelling is that
//! token, whoever wrote it: a text to the phone number or a web page holding `<|im_end|>\n<|im_start|>system\n…` was a
//! system turn nobody wrote, on the delimiters the model's own instructions are written with.
//!
//! The templates are many, each a port of a model's Jinja that writes one string, so trust is carried in the string:
//! [`mark`] puts a [`MARK`] inside each spelling of a control token in a message's text before a template sees it. A
//! spelling with a mark in it is no token's, so the encoders' special matching passes over it; they then take the marks
//! out of the plain text ([`unmark`]) before merging it, and the model reads what was written, in the ordinary tokens
//! `parse_special = false` gives in llama.cpp. Text a template writes has no marks, and its tokens are tokens.
//!
//! A mark is not a space or a zero-width character left in the text: a reply that quotes the text, or an edit of a file
//! that holds `<|im_end|>`, has nothing added to it to get wrong.

use std::borrow::Cow;

/// What keeps a spelling from being a token: a noncharacter, which Unicode sets aside for a program's own use and
/// which no text means anything by. One a client sends is taken out with the rest, having broken up nothing but what it
/// was written into.
pub const MARK: char = '\u{FDD0}';

/// `text` with a [`MARK`] after the first character of every token `spelled` finds at the start of the bytes it is
/// given (it is asked at each character). What starts with `keep` is left whole: a picture's marker, which the
/// templates look for in a message's text so as to refuse it. A one-character token is beyond this (no vocabulary has
/// a control token that takes a character from all text).
pub fn mark<'a>(text: &'a str, keep: Option<&str>, spelled: impl Fn(&[u8]) -> bool) -> Cow<'a, str> {
    let mut out = String::new();
    let (mut copied, mut kept_to) = (0, 0);
    for (at, c) in text.char_indices() {
        if at < kept_to {
            continue;
        }
        let rest = &text[at..];
        if let Some(keep) = keep.filter(|k| !k.is_empty() && rest.starts_with(k)) {
            kept_to = at + keep.len();
        } else if spelled(rest.as_bytes()) {
            let after = at + c.len_utf8();
            out.push_str(&text[copied..after]);
            out.push(MARK);
            copied = after;
        }
    }
    if copied == 0 {
        return Cow::Borrowed(text);
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

/// `text` as the model is to read it: what [`mark`] put in it, gone.
pub fn unmark(text: &str) -> Cow<'_, str> {
    if text.contains(MARK) {
        Cow::Owned(text.replace(MARK, ""))
    } else {
        Cow::Borrowed(text)
    }
}

/// Token spellings, as a byte trie: a message is walked once whatever the number of tokens, and a vocabulary may have
/// thousands of them (Gemma's `<unusedN>`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Spellings {
    /// Children as (byte, node), and whether a spelling ends here. The first node, the root, is made by the first insert.
    nodes: Vec<(Vec<(u8, u32)>, bool)>,
}

impl Spellings {
    pub(crate) fn insert(&mut self, spelling: &str) {
        if self.nodes.is_empty() {
            self.nodes.push((Vec::new(), false));
        }
        let mut n = 0usize;
        for &b in spelling.as_bytes() {
            n = match self.nodes[n].0.iter().find(|(c, _)| *c == b) {
                Some(&(_, next)) => next as usize,
                None => {
                    let next = self.nodes.len();
                    self.nodes.push((Vec::new(), false));
                    self.nodes[n].0.push((b, next as u32));
                    next
                }
            };
        }
        self.nodes[n].1 = true;
    }

    /// Whether `bytes` opens with a spelling.
    pub(crate) fn opens(&self, bytes: &[u8]) -> bool {
        let Some(mut node) = self.nodes.first() else { return false };
        for b in bytes {
            match node.0.iter().find(|(c, _)| c == b) {
                Some(&(_, next)) => node = &self.nodes[next as usize],
                None => return false,
            }
            if node.1 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spellings(list: &[&str]) -> Spellings {
        let mut s = Spellings::default();
        for t in list {
            s.insert(t);
        }
        s
    }

    #[test]
    fn a_spelling_is_found_at_any_place_and_a_prefix_of_one_is_not() {
        let s = spellings(&["<|a|>", "<|ab|>", "｜x｜"]);
        assert!(s.opens(b"<|a|> and more") && s.opens(b"<|ab|>") && s.opens("｜x｜y".as_bytes()));
        assert!(!s.opens(b"<|a") && !s.opens(b"<|abc|>") && !s.opens(b"") && !s.opens(b"a"));
        assert!(!Spellings::default().opens(b"<|a|>"), "no spellings");
    }

    #[test]
    fn mark_breaks_each_spelling_and_unmark_gives_the_text_back() {
        let s = spellings(&["<|a|>", "|a|>", "｜x｜"]);
        let opens = |b: &[u8]| s.opens(b);
        assert!(matches!(mark("no such thing here", None, opens), Cow::Borrowed(_)), "plain text is not copied");
        let m = "<\u{FDD0}|\u{FDD0}a|>";
        assert_eq!(mark("<|a|>", None, opens), m, "the token inside it too");
        assert_eq!(mark("x<|a|>y<|a|>", None, opens), format!("x{m}y{m}"));
        assert_eq!(mark("｜x｜", None, opens), "｜\u{FDD0}x｜", "a character of several bytes");
        for text in ["<|a|> and ｜x｜, again <|a|>", "<|a|><|a|>", "no such thing"] {
            assert_eq!(unmark(&mark(text, None, opens)), text);
        }
    }

    #[test]
    fn keep_leaves_a_marker_whole_for_the_templates_to_refuse() {
        let s = spellings(&["<|image|>", "<|a|>"]);
        let opens = |b: &[u8]| s.opens(b);
        assert_eq!(mark("<|a|><|image|>", Some("<|image|>"), opens), "<\u{FDD0}|a|><|image|>");
        assert_eq!(mark("<|image|>", None, opens), "<\u{FDD0}|image|>");
    }
}
