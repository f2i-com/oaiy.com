//! HMAC-SHA256 (review M04): `hmac_sha256_verify` must accept a whole tag and nothing else. The reviewer's mutant that made it accept a truncated tag
//! (`verify_truncated_left`: a tag of 1 to 32 bytes that is a prefix of the right one) passed the suite, because the suite had no HMAC test at all. Truncation is the
//! dangerous one: a verifier that accepts a 4-byte prefix can be satisfied by guessing 2^32 times, not 2^256.
//!
//! The known answers are the seven cases of RFC 4231 for HMAC-SHA-256 (the values are the RFC's, and were also recomputed with Node's `crypto.createHmac`): a key
//! shorter than, equal to and longer than the block (131 bytes, which is hashed first), data shorter and longer than the block, and the truncation case whose
//! RFC output is the first 128 bits of the tag (here the full tag, and the 128-bit prefix must be refused).

mod common;

use common::{hex, unhex};
use oaiy_crypto::kdf::{hmac_sha256, hmac_sha256_verify};

/// (name, key, data, the full HMAC-SHA-256 tag), all hex.
const RFC_4231: [(&str, &str, &str, &str); 7] = [
    (
        "case 1",
        "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
        "4869205468657265",
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
    ),
    (
        "case 2",
        "4a656665",
        "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
    ),
    (
        "case 3",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
    ),
    (
        "case 4",
        "0102030405060708090a0b0c0d0e0f10111213141516171819",
        "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
        "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
    ),
    (
        "case 5 (the RFC truncates to 128 bits: a 16-byte prefix of this)",
        "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c",
        "546573742057697468205472756e636174696f6e",
        "a3b6167473100ee06e0c796c2955552bfa6f7c0a6a8aef8b93f860aab0cd20c5",
    ),
    (
        "case 6 (a key of 131 bytes, hashed first)",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "54657374205573696e67204c6172676572205468616e20426c6f636b2d53697a65204b6579202d2048617368204b6579204669727374",
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
    ),
    (
        "case 7 (a key of 131 bytes and data longer than a block)",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "5468697320697320612074657374207573696e672061206c6172676572207468616e20626c6f636b2d73697a65206b657920616e642061206c6172676572207468616e20626c6f636b2d73697a6520646174612e20546865206b6579206e6565647320746f20626520686173686564206265666f7265206265696e6720757365642062792074686520484d414320616c676f726974686d2e",
        "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
    ),
];
#[test]
fn hmac_sha256_gives_the_rfc_4231_tags_and_verify_accepts_them() {
    for (name, key, data, tag) in RFC_4231 {
        let (key, data, tag) = (unhex(key), unhex(data), unhex(tag));
        assert_eq!(hex(&hmac_sha256(&key, &data).unwrap()), hex(&tag), "{name}");
        assert!(hmac_sha256_verify(&key, &data, &tag), "{name}: the right tag");
    }
}

#[test]
fn verify_accepts_the_whole_tag_and_no_truncation_of_it_and_no_extension() {
    for (name, key, data, tag) in RFC_4231 {
        let (key, data, tag) = (unhex(key), unhex(data), unhex(tag));
        // every prefix, the empty one included: the RFC's 128-bit truncation is among them and is not a valid tag here
        for length in 0..tag.len() {
            assert!(!hmac_sha256_verify(&key, &data, &tag[..length]), "{name}: a tag truncated to {length} bytes was accepted");
        }
        // a suffix is not a tag either, and neither is the tag with something after it
        for start in 1..tag.len() {
            assert!(!hmac_sha256_verify(&key, &data, &tag[start..]), "{name}: the last {} bytes were accepted", tag.len() - start);
        }
        for extra in [&[0u8][..], &[0u8; 32][..], &tag[..]] {
            let longer = [tag.as_slice(), extra].concat();
            assert!(!hmac_sha256_verify(&key, &data, &longer), "{name}: a tag with {} bytes after it was accepted", extra.len());
        }
        // and every one of the 256 bits of the tag matters
        for bit in 0..256 {
            let mut flipped = tag.clone();
            flipped[bit / 8] ^= 1 << (bit % 8);
            assert!(!hmac_sha256_verify(&key, &data, &flipped), "{name}: bit {bit} of the tag did not matter");
        }
    }
}

#[test]
fn verify_rejects_another_key_or_another_message() {
    let (_, key, data, tag) = RFC_4231[1];
    let (key, data, tag) = (unhex(key), unhex(data), unhex(tag));
    assert!(hmac_sha256_verify(&key, &data, &tag));
    assert!(!hmac_sha256_verify(b"Jeff", &data, &tag));
    assert!(!hmac_sha256_verify(&[], &data, &tag));
    assert!(!hmac_sha256_verify(&key, b"what do ya want for nothing", &tag));
    assert!(!hmac_sha256_verify(&key, &[], &tag));
    // a key of any length, the empty one included, and a message of any length
    assert_eq!(hmac_sha256(&[], &[]).unwrap().len(), 32);
    assert!(hmac_sha256_verify(&[], &[], &hmac_sha256(&[], &[]).unwrap()));
}
