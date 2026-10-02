//! base64url without padding (RFC 4648 section 5), read strictly (README section 1 and Interpretation 12).
//!
//! The relay's reader (`B64::dec`) accepts exactly what its writer produces: the alphabet `A-Za-z0-9_-`, no padding, no whitespace, no `+` or `/`, and a last character
//! whose unused low bits are zero, so that one byte string has one spelling (a token whose last secret character is `9` and not `8` is another spelling of the same bytes,
//! and is refused). This module is the same rule, for every value, fixed-length or not: the two ends of a signature or a MAC must never be able to disagree about which
//! text a byte string is.

use crate::error::B64Error;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// The value of each byte of the alphabet, 255 for any other byte.
const DECODE: [u8; 256] = {
    let mut table = [255u8; 256];
    let mut i = 0;
    while i < 64 {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// How many characters `n` bytes take.
pub const fn encoded_len(n: usize) -> usize {
    (n / 3) * 4
        + match n % 3 {
            0 => 0,
            1 => 2,
            _ => 3,
        }
}

/// Encodes `bytes` as base64url with no padding.
pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(encoded_len(bytes.len()));
    let mut chunks = bytes.chunks_exact(3);
    for c in &mut chunks {
        let n = (u32::from(c[0]) << 16) | (u32::from(c[1]) << 8) | u32::from(c[2]);
        for shift in [18, 12, 6, 0] {
            out.push(char::from(ALPHABET[((n >> shift) & 63) as usize]));
        }
    }
    match chunks.remainder() {
        [a] => {
            let n = u32::from(*a) << 16;
            for shift in [18, 12] {
                out.push(char::from(ALPHABET[((n >> shift) & 63) as usize]));
            }
        }
        [a, b] => {
            let n = (u32::from(*a) << 16) | (u32::from(*b) << 8);
            for shift in [18, 12, 6] {
                out.push(char::from(ALPHABET[((n >> shift) & 63) as usize]));
            }
        }
        _ => {}
    }
    out
}

/// Decodes canonical base64url. An empty text is refused (no value of the protocol is zero bytes of base64url).
pub fn decode(text: &str) -> Result<Vec<u8>, B64Error> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Err(B64Error::Length);
    }
    let mut values = Vec::with_capacity(bytes.len());
    for &b in bytes {
        let v = DECODE[usize::from(b)];
        if v == 255 {
            return Err(B64Error::Alphabet);
        }
        values.push(v);
    }
    let rem = values.len() % 4;
    if rem == 1 {
        return Err(B64Error::Length);
    }
    let mut out = Vec::with_capacity(values.len() / 4 * 3 + 2);
    let mut groups = values.chunks_exact(4);
    for g in &mut groups {
        let n = (u32::from(g[0]) << 18) | (u32::from(g[1]) << 12) | (u32::from(g[2]) << 6) | u32::from(g[3]);
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    }
    match groups.remainder() {
        [a, b] => {
            // 12 bits carry one byte: the low four bits of the second character are unused and must be zero.
            if b & 0x0f != 0 {
                return Err(B64Error::NonCanonical);
            }
            out.push((a << 2) | (b >> 4));
        }
        [a, b, c] => {
            // 18 bits carry two bytes: the low two bits of the third character are unused and must be zero.
            if c & 0x03 != 0 {
                return Err(B64Error::NonCanonical);
            }
            out.push((a << 2) | (b >> 4));
            out.push((b << 4) | (c >> 2));
        }
        _ => {}
    }
    Ok(out)
}

/// Decodes canonical base64url that must be exactly `N` bytes (a key, a thumbprint, a signature, a nonce, a `pid`).
pub fn decode_exact<const N: usize>(text: &str) -> Result<[u8; N], B64Error> {
    // The length of the text is checked first so that a hostile value of any size costs nothing to refuse.
    if text.len() != encoded_len(N) {
        return Err(if text.bytes().all(|b| DECODE[usize::from(b)] != 255) { B64Error::WrongSize } else { B64Error::Alphabet });
    }
    let bytes = decode(text)?;
    <[u8; N]>::try_from(bytes.as_slice()).map_err(|_| B64Error::WrongSize)
}

/// True when `text` is the canonical spelling of some byte string.
pub fn is_canonical(text: &str) -> bool {
    decode(text).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4648_test_vectors_without_padding() {
        // RFC 4648 section 10, standard alphabet, padding removed: none of these has a `+` or a `/`.
        for (plain, coded) in
            [("", ""), ("f", "Zg"), ("fo", "Zm8"), ("foo", "Zm9v"), ("foob", "Zm9vYg"), ("fooba", "Zm9vYmE"), ("foobar", "Zm9vYmFy")]
        {
            assert_eq!(encode(plain.as_bytes()), coded);
            if !plain.is_empty() {
                assert_eq!(decode(coded).unwrap(), plain.as_bytes());
            }
        }
        // The two characters that differ from the standard alphabet.
        assert_eq!(encode(&[0xfb, 0xff, 0xbf]), "-_-_");
        assert_eq!(decode("-_-_").unwrap(), vec![0xfb, 0xff, 0xbf]);
    }

    #[test]
    fn every_length_round_trips() {
        for n in 0..=70usize {
            let bytes: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let text = encode(&bytes);
            assert_eq!(text.len(), encoded_len(n));
            if n > 0 {
                assert_eq!(decode(&text).unwrap(), bytes);
            }
        }
    }

    #[test]
    fn refuses_padding_whitespace_other_alphabets_and_impossible_lengths() {
        assert_eq!(decode("Zg=="), Err(B64Error::Alphabet));
        assert_eq!(decode("Zg="), Err(B64Error::Alphabet));
        assert_eq!(decode("Zm9v "), Err(B64Error::Alphabet));
        assert_eq!(decode(" Zm9v"), Err(B64Error::Alphabet));
        assert_eq!(decode("Zm9v\n"), Err(B64Error::Alphabet));
        assert_eq!(decode("Zm+v"), Err(B64Error::Alphabet));
        assert_eq!(decode("Zm/v"), Err(B64Error::Alphabet));
        assert_eq!(decode("Zm9\u{e9}"), Err(B64Error::Alphabet));
        assert_eq!(decode(""), Err(B64Error::Length));
        assert_eq!(decode("Z"), Err(B64Error::Length));
        assert_eq!(decode("Zm9vY"), Err(B64Error::Length));
    }

    #[test]
    fn refuses_a_spelling_whose_unused_low_bits_are_not_zero() {
        // One byte takes two characters and the second keeps four unused bits: only A, Q, g and w (6-bit values 0, 16, 32, 48) have them clear. "Zg" is the byte 'f';
        // "Zh" ... are other spellings of it, and "Zw" is a different byte ('g'), not another spelling.
        assert_eq!(decode("Zg").unwrap(), b"f");
        assert_eq!(decode("Zw").unwrap(), b"g");
        for last in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".chars() {
            let text = format!("Z{last}");
            assert_eq!(decode(&text).is_ok(), "AQgw".contains(last), "{text}");
        }
        // Two bytes: the third character keeps two unused bits; 'M' (12) is canonical after "Zm", 'N' (13) is another spelling.
        assert_eq!(decode("Zm8").unwrap(), b"fo");
        assert_eq!(decode("Zm9"), Err(B64Error::NonCanonical));
        // The relay's own vector: the last character of a 43-character secret must be one of A E I M Q U Y c g k o s w 0 4 8.
        for c in "AEIMQUYcgkosw048".chars() {
            assert!(decode(&format!("ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj{c}")).is_ok(), "{c}");
        }
        for c in "BCDFGHJKLNOPRSTVWXZabdefhijlmnpqrtuvxyz1235679-_".chars() {
            assert_eq!(decode(&format!("ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj{c}")), Err(B64Error::NonCanonical), "{c}");
        }
    }

    #[test]
    fn decode_exact_checks_the_size() {
        let key: [u8; 32] = decode_exact("_RckOFqgx1tk-3jNYC-h2ZH96_drE8WO1wLqyDXp9hg").unwrap();
        assert_eq!(key[0], 0xfd);
        assert_eq!(decode_exact::<32>("_RckOFqgx1tk-3jNYC-h2ZH96_drE8WO1wLqyDXp9"), Err(B64Error::WrongSize));
        assert_eq!(decode_exact::<16>("_RckOFqgx1tk-3jNYC-h2ZH96_drE8WO1wLqyDXp9hg"), Err(B64Error::WrongSize));
        assert_eq!(decode_exact::<32>("_RckOFqgx1tk-3jNYC-h2ZH96_drE8WO1wLqyDXp9h="), Err(B64Error::Alphabet));
        assert_eq!(decode_exact::<8>("AQIDBAUGBwh"), Err(B64Error::NonCanonical));
        assert_eq!(decode_exact::<8>("AQIDBAUGBwg").unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
