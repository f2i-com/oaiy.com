//! `bip39` (design 4.1.1, 4.3): the twelve-word recovery phrase, English list, 16 bytes of entropy and 4 check bits.
//!
//! The secret is the 16 entropy bytes; the twelve words are their checksummed spelling. The BIP-39 seed (PBKDF2) is never used,
//! so a phrase made here is not a wallet. `wk = kdf(1, "flphras1", argon2id13(entropy16, salt16, 3, 64 MiB))`
//! ([`phrase_wrap_key`]) turns the words into the key that wraps the vault's UMK.
//!
//! **Decoding** follows design 4.3 exactly: NFKD, lower case, split on white space, **exactly 12 words**, every word in the list,
//! and the checksum verified, in that order, with `Error::PhraseLength`, `PhraseWord`, `PhraseChecksum`. The checksum is
//! verified before any key derivation, and the type system keeps it so: the derivation functions take an [`Entropy`], and an
//! `Entropy` comes only from [`decode`] (or from random bytes or 16 bytes the caller holds), never from unchecked words.
//! Input over 1024 bytes is refused as `PhraseLength` before it is normalised. **White space is JavaScript's `\s`** (review L-8): TAB, LF, VT, FF, CR, SPACE,
//! NBSP, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000 and the byte-order mark U+FEFF, and not Unicode's `White_Space`, which also has U+0085
//! (the set is `text::is_js_space`, the same one the kit decoder strips, compared with Node code point by code point). The browser's decoder (V-04) is `normalize('NFKD').
//! toLowerCase().split(/\s+/)`, so a phrase that one accepts the other accepts **except above the 1024-byte cap**, which this decoder has (design 4.3) and that code, as
//! it stands, does not (V-04 must pin the cap): `tests/vectors/text-corpus.json` has 177 phrases, with the verdict of exactly that code in Node, with and without the cap,
//! and this decoder agrees with every one; the five that differ are class `stricter`, all over the cap. A phrase a user made up is never accepted: there is no path from text to entropy but the checksummed words.
//!
//! The list is embedded (`bip39_english.txt`, SHA-256 [`WORDLIST_SHA256`], the official list): the first four letters of a word
//! identify it, which is what lets the entry window resolve a word by autocomplete.

use std::sync::OnceLock;

use unicode_normalization::UnicodeNormalization;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::argon;
use crate::error::Error;
use crate::kdf::{self, sha256, Purpose};
use crate::text::{is_js_space, push_zeroizing};
use crate::zeroize::{scrub_stack, Secret, SecretString};

/// The list, one word per line, LF-terminated.
const WORDLIST: &str = include_str!("bip39_english.txt");

/// SHA-256 of the file `bip39_english.txt`, the official English list (design 4.1.1).
pub const WORDLIST_SHA256: &str = "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda";

/// Words in a phrase.
pub const WORD_COUNT: usize = 12;

/// The longest text `decode` looks at.
pub const MAX_INPUT_BYTES: usize = 1024;

/// The 2048 words, sorted.
pub fn wordlist() -> &'static [&'static str] {
    static WORDS: OnceLock<Vec<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| WORDLIST.lines().filter(|line| !line.is_empty()).collect()).as_slice()
}

/// The 16 secret bytes a phrase spells.
pub struct Entropy(Secret<16>);

impl Entropy {
    /// From 16 bytes the caller holds.
    pub const fn from_bytes(bytes: [u8; 16]) -> Entropy {
        Entropy(Secret::new(bytes))
    }

    /// Sixteen bytes from the operating system's random generator: how a phrase is made.
    pub fn random() -> Result<Entropy, Error> {
        let mut secret = Secret::<16>::zeroed();
        secret.fill_random()?;
        scrub_stack();
        Ok(Entropy(secret))
    }

    /// The bytes.
    pub const fn expose(&self) -> &[u8; 16] {
        self.0.expose()
    }
}

impl ZeroizeOnDrop for Entropy {}

impl core::fmt::Debug for Entropy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("bip39::Entropy(redacted)")
    }
}

impl PartialEq for Entropy {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Entropy {}

/// The twelve words of `entropy`, separated by single spaces.
pub fn encode(entropy: &Entropy) -> SecretString {
    let list = wordlist();
    let mut bits = [0u8; 19];
    bits[..16].copy_from_slice(entropy.expose());
    bits[16] = (sha256(entropy.expose())[0] >> 4) << 4;
    // reserved up front (the longest word has eight letters, so twelve words and eleven spaces fit): a `String` that grows moves to a new block and leaves the
    // old one, with part of the phrase in it, unwiped (review M-4)
    let mut text = String::with_capacity(WORD_COUNT * 9);
    for i in 0..WORD_COUNT {
        let start = 11 * i;
        let byte = start / 8;
        let window = (u32::from(bits[byte]) << 16) | (u32::from(bits[byte + 1]) << 8) | u32::from(bits[byte + 2]);
        let index = ((window >> (24 - 11 - (start % 8))) & 0x7ff) as usize;
        if i > 0 {
            text.push(' ');
        }
        text.push_str(list.get(index).copied().unwrap_or(""));
    }
    bits.zeroize();
    SecretString::new(text)
}

/// Reads a phrase: `Error::PhraseLength`, `PhraseWord` or `PhraseChecksum`, in that order of precedence, or the entropy.
pub fn decode(input: &str) -> Result<Entropy, Error> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(Error::PhraseLength);
    }
    // Both copies of the text are built into buffers of a size that is not outgrown in the usual case (NFKD may lengthen a character, so there is room), and by
    // hand when it is (`push_zeroizing`), so that no block of the phrase is ever freed without being wiped (review M-4).
    let mut decomposed: Zeroizing<String> = Zeroizing::new(String::with_capacity(input.len() * 2 + 16));
    for c in input.nfkd() {
        push_zeroizing(&mut decomposed, c);
    }
    let mut lowered: Zeroizing<String> = Zeroizing::new(String::with_capacity(decomposed.len() * 2 + 16));
    for c in decomposed.chars() {
        for lower in c.to_lowercase() {
            push_zeroizing(&mut lowered, lower);
        }
    }
    let words: Vec<&str> = lowered.split(is_js_space).filter(|w| !w.is_empty()).collect();
    if words.len() != WORD_COUNT {
        return Err(Error::PhraseLength);
    }
    let list = wordlist();
    let mut bits = [0u8; 17];
    for (i, word) in words.iter().enumerate() {
        let index = list.binary_search(word).map_err(|_| Error::PhraseWord)?;
        for bit in 0..11 {
            if (index >> (10 - bit)) & 1 == 1 {
                let position = 11 * i + bit;
                bits[position / 8] |= 0x80 >> (position % 8);
            }
        }
    }
    let mut entropy = [0u8; 16];
    entropy.copy_from_slice(&bits[..16]);
    let expected = sha256(&entropy)[0] >> 4;
    let given = bits[16] >> 4;
    bits.zeroize();
    if expected != given {
        entropy.zeroize();
        return Err(Error::PhraseChecksum);
    }
    let result = Entropy::from_bytes(entropy);
    entropy.zeroize();
    Ok(result)
}

/// `wk = kdf(1, "flphras1", argon2id13(entropy16, salt16, ops, mem))`, the key that wraps the UMK (design 4.3). Argon2's bounds are
/// checked before it runs (`KdfParamsOutOfRange`), and it takes the entropy, which exists only after the checksum was verified.
pub fn wrap_key_into(entropy: &Entropy, salt: &[u8], ops: u64, mem_bytes: u64, out: &mut Secret<32>) -> Result<(), Error> {
    let ikm = argon::argon2id13(entropy.expose(), salt, ops, mem_bytes)?;
    kdf::derive_into(&ikm, Purpose::PhraseWrap, out)
}

/// [`wrap_key_into`] by value (which leaves a copy of the wrap key in the frame that made it: see `kdf::derive`).
pub fn wrap_key(entropy: &Entropy, salt: &[u8], ops: u64, mem_bytes: u64) -> Result<Secret<32>, Error> {
    let ikm = argon::argon2id13(entropy.expose(), salt, ops, mem_bytes)?;
    kdf::derive(&ikm, Purpose::PhraseWrap)
}

/// The words to the wrap key, written into `out` in place: [`decode`] (the checksum, before any KDF work), then [`wrap_key_into`].
pub fn phrase_wrap_key_into(phrase: &str, salt: &[u8], ops: u64, mem_bytes: u64, out: &mut Secret<32>) -> Result<(), Error> {
    let entropy = decode(phrase)?;
    wrap_key_into(&entropy, salt, ops, mem_bytes, out)
}

/// The words to the wrap key: [`decode`] (the checksum, before any KDF work), then [`wrap_key`].
pub fn phrase_wrap_key(phrase: &str, salt: &[u8], ops: u64, mem_bytes: u64) -> Result<Secret<32>, Error> {
    let entropy = decode(phrase)?;
    wrap_key(&entropy, salt, ops, mem_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Design 4.3: no key derivation runs for a phrase that fails its length, its words or its checksum.
    #[test]
    fn checksum_before_kdf_no_argon2_runs_for_a_bad_phrase() {
        let salt = [0xa0u8; 16];
        let before = crate::argon::calls();
        let bad = [
            "",
            "abandon",
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon",
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon",
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abouu",
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon",
            "legal winner thank year wave sausage worth useful legal winner thank zoo",
        ];
        for phrase in bad {
            assert!(phrase_wrap_key(phrase, &salt, 3, argon::MEM_MIN).is_err(), "{phrase}");
        }
        assert_eq!(crate::argon::calls(), before, "Argon2 ran for a phrase that is not valid");
        // and it does run for a valid one, so the counter is not stuck
        assert!(phrase_wrap_key(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
            &salt,
            3,
            argon::MEM_MIN
        )
        .is_ok());
        assert_eq!(crate::argon::calls(), before + 1);
    }
}
