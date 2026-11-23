//! GPT-2 byte ⇄ unicode mapping.
//!
//! GPT-2 / Llama-3 / Qwen / Gemma byte-level BPE represents each input byte as
//! a "visible" Unicode character so the tokenizer can split on whitespace
//! without losing information. Most printable ASCII bytes map to themselves;
//! control characters, space (0x20), DEL, and a handful of others get mapped
//! to characters in U+0100..U+0143 (e.g. space → 'Ġ' = U+0120).

use std::sync::OnceLock;

/// `BYTE_TO_CHAR[b]` is the Unicode char representing byte `b`.
fn byte_to_char_table() -> &'static [char; 256] {
    static T: OnceLock<[char; 256]> = OnceLock::new();
    T.get_or_init(|| {
        // Bytes that map to themselves.
        let kept: Vec<u8> = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut table = ['\0'; 256];
        for &b in &kept { table[b as usize] = b as char; }

        // The remaining 68 bytes get mapped to U+0100 + n.
        let mut n: u32 = 0;
        for b in 0u8..=255u8 {
            if !kept.contains(&b) {
                table[b as usize] = char::from_u32(0x100 + n).unwrap();
                n += 1;
            }
        }
        debug_assert_eq!(n, 68);
        table
    })
}

#[inline]
pub fn byte_to_char(b: u8) -> char {
    byte_to_char_table()[b as usize]
}

/// Reverse mapping: `CHAR_TO_BYTE[c]` -> byte. Returns None for chars outside
/// the byte-level alphabet. All 256 byte-level BPE symbols lie in `0..0x144`,
/// so a direct flat table gives O(1) lookup without hashing.
#[inline]
pub fn char_to_byte(c: char) -> Option<u8> {
    const TABLE_LEN: usize = 0x144;
    const NONE: u16 = u16::MAX;
    static REV: OnceLock<[u16; TABLE_LEN]> = OnceLock::new();
    let tbl = REV.get_or_init(|| {
        let t = byte_to_char_table();
        let mut arr = [NONE; TABLE_LEN];
        for (i, &ch) in t.iter().enumerate() {
            let code = ch as usize;
            debug_assert!(code < TABLE_LEN);
            arr[code] = i as u16;
        }
        arr
    });
    let code = c as usize;
    if code < TABLE_LEN {
        let v = tbl[code];
        if v != NONE {
            return Some(v as u8);
        }
    }
    None
}

/// Encode a byte string as a chain of "visible" chars.
pub fn bytes_to_visible(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| byte_to_char(b)).collect()
}

/// Append raw bytes decoded from a visible-char string into `out`.
#[inline]
pub fn push_visible_bytes(s: &str, out: &mut Vec<u8>) {
    out.reserve(s.len());
    for c in s.chars() {
        if let Some(b) = char_to_byte(c) {
            out.push(b);
        }
    }
}

/// Reverse: turn a visible-char string back into raw bytes. Drops unknown
/// chars rather than panicking — robust to corrupt inputs.
pub fn visible_to_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    push_visible_bytes(s, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_maps_to_g_dot() {
        // GPT-2's signature: 0x20 (space) -> 'Ġ' (U+0120).
        assert_eq!(byte_to_char(b' '), 'Ġ');
    }

    #[test]
    fn printable_ascii_passthrough() {
        for b in b'!'..=b'~' {
            assert_eq!(byte_to_char(b), b as char);
        }
    }

    #[test]
    fn roundtrip_arbitrary_bytes() {
        let s: Vec<u8> = (0..=255u8).collect();
        let visible = bytes_to_visible(&s);
        let back = visible_to_bytes(&visible);
        assert_eq!(back, s);
    }

    #[test]
    fn roundtrip_text() {
        let text = "Hello, world! 👋";
        let v = bytes_to_visible(text.as_bytes());
        let back = visible_to_bytes(&v);
        assert_eq!(back, text.as_bytes());
    }
}
