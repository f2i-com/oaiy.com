//! The BIP-39 English word list: 2,048 words, and the draw of a passphrase from it.
//!
//! Source: `bip-0039/english.txt` of <https://github.com/bitcoin/bips>, unchanged. BIP-39 is by Marek
//! Palatinus, Pavol Rusnak, Aaron Voisine and Sean Bowe and is published under the MIT licence (the
//! `License: MIT` line of the BIP's own header); the list is vendored in `bip39_english.txt`, and
//! `NOTICE` says so. The list is not used to derive keys here: it is a set of 2,048 short, distinct,
//! easy-to-type words, which is what a generated password needs (six of them are 66 bits).
//!
//! A word is an 11-bit number, so a draw takes 11 bits of randomness and never rejects: two random
//! bytes masked to 11 bits are uniform over 0..2048 (65,536 is a multiple of 2,048), which is what a
//! draw by `random % 2048` would also be, and what a draw by `random % 2000` would not.

use std::sync::OnceLock;

/// The list as the file holds it (one word a line).
const ENGLISH_TEXT: &str = include_str!("bip39_english.txt");

/// The number of words: 11 bits each.
pub const WORDS: usize = 2048;

/// The words, in the list's order.
pub fn words() -> &'static [&'static str] {
    static LIST: OnceLock<Vec<&'static str>> = OnceLock::new();
    LIST.get_or_init(|| {
        ENGLISH_TEXT
            .lines()
            .map(str::trim)
            .filter(|w| !w.is_empty())
            .collect()
    })
}

/// A source of random bytes.
pub type Fill<'a> = &'a mut dyn FnMut(&mut [u8]) -> Result<(), super::token::MintError>;

/// `count` words drawn uniformly, joined by `separator`. Fails only when the random source does.
pub fn draw(
    count: usize,
    separator: &str,
    fill: Fill<'_>,
) -> Result<String, super::token::MintError> {
    let list = words();
    debug_assert_eq!(list.len(), WORDS);
    let mut chosen: Vec<&str> = Vec::with_capacity(count);
    for _ in 0..count {
        let mut two = [0u8; 2];
        fill(&mut two)?;
        let index = (usize::from(two[0]) << 8 | usize::from(two[1])) & (WORDS - 1);
        chosen.push(list[index]);
    }
    Ok(chosen.join(separator))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn the_list_is_the_bip39_english_list_byte_for_byte() {
        // The published file: 2,048 lines, each ending in a line feed, SHA-256 as below. The hash is taken over
        // the words rejoined, so a checkout that turned line feeds into CRLF does not change it.
        let list = words();
        assert_eq!(list.len(), WORDS);
        let mut text = list.join("\n");
        text.push('\n');
        assert_eq!(
            crate::auth::token::to_hex(&Sha256::digest(text.as_bytes())),
            "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"
        );
        assert_eq!((list[0], list[2047]), ("abandon", "zoo"));
    }

    #[test]
    fn the_list_has_the_properties_a_typed_passphrase_needs() {
        let list = words();
        assert!(list.windows(2).all(|w| w[0] < w[1]), "sorted, no repeats");
        for w in list {
            assert!(
                (3..=8).contains(&w.len()) && w.bytes().all(|b| b.is_ascii_lowercase()),
                "{w}"
            );
        }
        // The first four letters name a word: a passphrase can be typed by its prefixes.
        let mut prefixes: Vec<&str> = list.iter().map(|w| &w[..w.len().min(4)]).collect();
        prefixes.sort_unstable();
        prefixes.dedup();
        assert_eq!(prefixes.len(), WORDS);
    }

    #[test]
    fn a_draw_takes_eleven_bits_without_bias() {
        // Every one of the 2,048 values of the low 11 bits is one word, and the high bits change nothing.
        let list = words();
        for i in 0u16..=u16::MAX {
            let bytes = i.to_be_bytes();
            let mut fed = false;
            let mut fill = |buf: &mut [u8]| {
                buf.copy_from_slice(&bytes);
                fed = true;
                Ok(())
            };
            let one = draw(1, "-", &mut fill).unwrap();
            assert_eq!(one, list[usize::from(i) % WORDS]);
            assert!(fed);
        }
    }

    #[test]
    fn a_draw_joins_the_words_and_reports_a_failure_of_the_random_source() {
        let mut n = 0u8;
        let mut fill = |buf: &mut [u8]| {
            buf[0] = 0;
            buf[1] = n;
            n += 1;
            Ok(())
        };
        assert_eq!(
            draw(3, " ", &mut fill).unwrap(),
            format!("{} {} {}", words()[0], words()[1], words()[2])
        );
        let mut broken = |_: &mut [u8]| Err(super::super::token::MintError::NoRandomness);
        assert_eq!(
            draw(6, "-", &mut broken),
            Err(super::super::token::MintError::NoRandomness)
        );
    }
}
