//! Text that is a secret (a recovery phrase, a kit code) and how it is built without leaving copies behind.
//!
//! A `String` that grows by pushing moves to a bigger block when it is full, and the block it leaves is freed, not wiped, with the first part of the text in
//! it (review M-4: `bip39::decode` left blocks of 19 and 38 bytes that began 'legal winner thank year wave sausage w'; `encode` left blocks of 16, 32 and 64
//! bytes, the last holding ten of the twelve words; the kit left `FLRK1-CEIR-...`). Two rules keep a secret text in one block: reserve the whole size up
//! front, and where the size is not known exactly, grow by hand with [`push_zeroizing`], which wipes the block it leaves.

use zeroize::Zeroizing;

/// JavaScript's `\s`: the **one** set of white space that the phrase decoder splits on and the kit decoder strips, so that nothing a browser's `split(/\s+/)` and
/// `replace(/[\s-]+/g, "")` accept is refused here, and nothing they refuse is accepted (review L-8). It is the ECMAScript `WhiteSpace` and `LineTerminator` productions:
/// TAB, LF, VT, FF, CR, SPACE, NBSP, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000 and the byte-order mark U+FEFF. It is **not** Unicode's
/// `White_Space` (which also has U+0085 and, before Unicode 6.3, U+180E) and not Rust's `char::is_whitespace`: U+0085 is not white space to JavaScript, and the
/// Rust decoders used to split phrases on it. `tests/vectors/text-corpus.json` holds the set as Node computes it, and the test below compares every code point.
pub(crate) fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\u{9}'..='\u{d}' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
    )
}

/// Appends `c` to `text`. If `text` is full it moves to a bigger buffer by hand, and the old one is wiped as it is dropped (a `String` that grows on its own leaves
/// the old block behind, unwiped). Reserve enough capacity up front and this never has to grow.
pub(crate) fn push_zeroizing(text: &mut Zeroizing<String>, c: char) {
    if text.len() + c.len_utf8() > text.capacity() {
        let mut bigger = Zeroizing::new(String::with_capacity((text.capacity() * 2).max(64)));
        bigger.push_str(text);
        *text = bigger; // the old buffer is dropped here, and `Zeroizing` wipes it
    }
    text.push(c);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The set is JavaScript's `\s` exactly: all 1,112,064 scalar values are compared with the list Node produced (`tests/vectors/scripts/text_corpora.mjs`).
    #[test]
    fn the_white_space_set_is_javascripts_backslash_s_for_every_code_point() {
        let corpus = include_str!("../tests/vectors/text-corpus.json");
        let start = corpus.find("\"js_space\": [").expect("the corpus lists the code points") + "\"js_space\": [".len();
        let end = start + corpus[start..].find(']').expect("the list ends");
        let listed: Vec<u32> = corpus[start..end].split(',').map(|n| n.trim().parse().expect("a code point")).collect();
        assert_eq!(listed.len(), 25, "JavaScript's \\s has 25 members");
        let ours: Vec<u32> = (0..=0x10ffffu32).filter_map(char::from_u32).filter(|c| is_js_space(*c)).map(u32::from).collect();
        assert_eq!(ours, listed);
        // and what it is not: Unicode's White_Space has U+0085, which JavaScript does not count
        assert!('\u{85}'.is_whitespace() && !is_js_space('\u{85}'));
        assert!(is_js_space('\u{feff}') && !'\u{feff}'.is_whitespace());
    }

    /// Growing by hand reaches the same text as pushing into a `String`, from nothing, from a full buffer, and across multi-byte characters.
    #[test]
    fn pushing_builds_the_same_text_as_a_string_across_growth_and_multibyte_characters() {
        let mut ours = Zeroizing::new(String::new());
        let mut plain = String::new();
        for (i, c) in "Wörter mit Ümläuten, 日本語 and 🙂 too, and more than sixty-four bytes of it in all".chars().cycle().take(500).enumerate()
        {
            push_zeroizing(&mut ours, c);
            plain.push(c);
            assert_eq!(*ours, plain, "after {i} characters");
        }
        let mut exact = Zeroizing::new(String::with_capacity(3));
        for c in ['a', 'b', 'c', 'd'] {
            push_zeroizing(&mut exact, c);
        }
        assert_eq!(*exact, "abcd");
        assert!(exact.capacity() >= 64, "a full buffer moved to a bigger one");
    }
}
