//! Text that is a secret (a recovery phrase, a kit code) and how it is built without leaving copies behind.
//!
//! A `String` that grows by pushing moves to a bigger block when it is full, and the block it leaves is freed, not wiped, with the first part of the text in
//! it (review M-4: `bip39::decode` left blocks of 19 and 38 bytes that began 'legal winner thank year wave sausage w'; `encode` left blocks of 16, 32 and 64
//! bytes, the last holding ten of the twelve words; the kit left `FLRK1-CEIR-...`). Two rules keep a secret text in one block: reserve the whole size up
//! front, and where the size is not known exactly, grow by hand with [`push_zeroizing`], which wipes the block it leaves.

use zeroize::Zeroizing;

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
