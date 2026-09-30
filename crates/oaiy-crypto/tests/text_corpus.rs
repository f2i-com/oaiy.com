//! The text of a recovery kit code and of a recovery phrase, against what JavaScript does with it (review L-8, and the survivors M53 and M33).
//!
//! `vectors/text-corpus.json` is made by `vectors/scripts/text_corpora.mjs` in Node: the white space that `\s` means, a verbatim port of FormLogic's
//! `decodeRecoveryKey` for 274 kit inputs, and the decoder of design 4.3 written the way the browser will write it (`normalize('NFKD').toLowerCase().split(/\s+/)`)
//! for 172 phrases. Every entry has the verdict that JavaScript gave. Nothing here is computed by the Rust code under test.
//!
//! **Kit:** an entry of class `same` must get exactly JavaScript's verdict and, when accepted, the same key (this is where a white space that one decoder knows
//! and the other does not would show: 25 characters as separators, at both ends, after every hyphen and between every character, and 23 characters that look
//! like white space or a hyphen and are neither). An entry of class `stricter` is accepted by JavaScript and must be **refused** here, on purpose: four
//! trailing bits that are not zero, a letter that JavaScript's upper-casing turns into `A` to `Z`, a code over the length cap. So no code this decoder accepts
//! is refused by JavaScript. **Phrase:** every entry must get JavaScript's verdict (`ok` with the same entropy, `length`, `word` or `checksum`), which includes
//! the 25 characters of `\s` between the words and the characters that are not (U+0085 among them).

mod common;

use common::{hex, json};
use oaiy_crypto::bip39;
use oaiy_crypto::kit::{self, RecoveryKit};
use oaiy_crypto::Error;

const CORPUS: &str = include_str!("vectors/text-corpus.json");

#[test]
fn the_kit_decoder_agrees_with_formlogics_javascript_where_it_should_and_is_stricter_only_where_it_is_meant_to_be() {
    let corpus = json(CORPUS);
    let entries = corpus["kit"].as_array().unwrap();
    let (mut same_ok, mut same_err, mut stricter) = (0, 0, 0);
    for entry in entries {
        let input = entry["input"].as_str().unwrap();
        let (class, note) = (entry["class"].as_str().unwrap(), entry["note"].as_str().unwrap());
        let js_ok = entry["js_ok"].as_bool().unwrap();
        let ours = RecoveryKit::decode(input);
        match class {
            "same" => {
                assert_eq!(ours.is_ok(), js_ok, "{note}: {input:?}: JavaScript says {js_ok}, this says {:?}", ours.as_ref().err());
                if let Ok(kit) = &ours {
                    assert_eq!(hex(kit.key().expose()), entry["key"].as_str().unwrap(), "{note}: the same code, another key");
                    same_ok += 1;
                } else {
                    same_err += 1;
                }
            }
            "stricter" => {
                assert!(js_ok, "{note}: a `stricter` entry is one that JavaScript accepts");
                assert!(ours.is_err(), "{note}: {input:?}: accepted here, and meant to be refused");
                stricter += 1;
            }
            other => panic!("unknown class {other}"),
        }
    }
    // the corpus has every kind in it, so a decoder that refuses everything or accepts everything cannot pass
    assert!(same_ok >= 100 && same_err >= 60 && stricter >= 50, "{same_ok} accepted, {same_err} refused, {stricter} stricter of {}", entries.len());
}

/// Every one of the 25 characters of `\s` separates the groups of a code, and only those: the 8 that Unicode has and JavaScript does not, and the look-alikes, do not.
#[test]
fn the_kit_decoder_strips_exactly_the_white_space_of_javascript() {
    let key = oaiy_crypto::zeroize::Secret::new(std::array::from_fn::<u8, 32, _>(|i| i as u8));
    let code = RecoveryKit::from_bytes(key).encode().expose().to_string();
    let js_space =
        json(CORPUS)["js_space"].as_array().unwrap().iter().map(|v| char::from_u32(v.as_u64().unwrap() as u32).unwrap()).collect::<Vec<_>>();
    assert_eq!(js_space.len(), 25);
    for c in &js_space {
        assert!(RecoveryKit::decode(&code.replace('-', &c.to_string())).is_ok(), "U+{:04X} as the separator", *c as u32);
        assert!(RecoveryKit::decode(&format!("{c}{code}{c}")).is_ok(), "U+{:04X} at both ends", *c as u32);
    }
    // the ones the two languages disagree about: Unicode White_Space without U+0085... and the other way round, the BOM
    for c in ['\u{85}', '\u{180e}', '\u{200b}', '\u{2060}', '\u{ad}', '\u{1c}', '\u{1f}', '\u{7f}', '\u{0}'] {
        assert!(!js_space.contains(&c));
        assert_eq!(
            RecoveryKit::decode(&code.replace('-', &c.to_string())).unwrap_err(),
            Error::KitFormat,
            "U+{:04X} is not white space to JavaScript",
            c as u32
        );
    }
    // and no Unicode white space at all is white space here that is not JavaScript's
    for c in (0..=0x10ffffu32).filter_map(char::from_u32).filter(|c| c.is_whitespace() && !js_space.contains(c)) {
        assert!(RecoveryKit::decode(&format!("{c}{code}")).is_err(), "U+{:04X}", c as u32);
    }
}

/// The length cap (M33: a mutant that removed it survived): 256 bytes are read, 257 are not, counted in bytes and not in characters.
#[test]
fn a_kit_code_longer_than_the_cap_is_refused_and_one_of_exactly_the_cap_is_read() {
    assert_eq!(kit::MAX_INPUT_BYTES, 256);
    let code = RecoveryKit::from_bytes(oaiy_crypto::zeroize::Secret::new([0x42; 32])).encode().expose().to_string();
    assert_eq!(code.len(), 75);
    let padded = |extra: usize| format!("{code}{}", " ".repeat(extra));
    assert!(RecoveryKit::decode(&padded(256 - 75)).is_ok(), "exactly 256 bytes");
    assert_eq!(RecoveryKit::decode(&padded(257 - 75)).unwrap_err(), Error::KitFormat, "257 bytes");
    assert_eq!(RecoveryKit::decode(&padded(5000)).unwrap_err(), Error::KitFormat, "far over");
    // bytes, not characters: an ideographic space is three bytes
    let wide = |n: usize| format!("{code}{}", "\u{3000}".repeat(n));
    assert!(RecoveryKit::decode(&wide(60)).is_ok(), "75 + 180 = 255 bytes (61 characters fewer than the cap in characters)");
    assert_eq!(RecoveryKit::decode(&wide(61)).unwrap_err(), Error::KitFormat, "75 + 183 = 258 bytes");
}

#[test]
fn the_phrase_decoder_agrees_with_the_browser_code_on_every_entry_of_the_corpus() {
    let corpus = json(CORPUS);
    let entries = corpus["phrase"].as_array().unwrap();
    let mut verdicts = std::collections::BTreeMap::new();
    let mut stricter = 0;
    for entry in entries {
        let (input, note) = (entry["input"].as_str().unwrap(), entry["note"].as_str().unwrap());
        let js = entry["verdict"].as_str().unwrap();
        let (class, pure) = (entry["class"].as_str().unwrap(), entry["js_verdict"].as_str().unwrap());
        // `same`: the browser code and this decoder say the same. `stricter`: they differ, and only above the byte cap, which this decoder has (design 4.3) and the browser's pure
        // code does not: the entry says what each says.
        match class {
            "same" => assert_eq!(pure, js, "{note}: a `same` entry whose verdicts differ"),
            "stricter" => {
                assert_ne!(pure, js, "{note}: a `stricter` entry whose verdicts agree");
                assert!(input.len() > bip39::MAX_INPUT_BYTES, "{note}: a difference below the cap ({} bytes)", input.len());
                stricter += 1;
            }
            other => panic!("unknown class {other}"),
        }
        let ours = match bip39::decode(input) {
            Ok(entropy) => {
                assert_eq!(Some(hex(entropy.expose()).as_str()), entry["entropy"].as_str(), "{note}: the same phrase, other entropy");
                "ok"
            }
            Err(Error::PhraseLength) => "length",
            Err(Error::PhraseWord) => "word",
            Err(Error::PhraseChecksum) => "checksum",
            Err(other) => panic!("{note}: {other:?}"),
        };
        assert_eq!(ours, js, "{note}: {input:?}");
        *verdicts.entry(js).or_insert(0) += 1;
    }
    assert!(verdicts["ok"] >= 100 && verdicts["length"] >= 30 && verdicts["word"] >= 10 && verdicts["checksum"] >= 5, "{verdicts:?}");
    assert!(stricter >= 4, "the corpus has the entries above the cap that the browser code would read ({stricter})");
}

/// U+0085 (next line) is white space to Unicode and to Rust and not to JavaScript: it used to split a phrase here, and does not now (review M52: the mutant that
/// removed it from the set survived). A phrase whose only separators are U+0085 is one word.
#[test]
fn next_line_does_not_separate_the_words_of_a_phrase() {
    let words = "legal winner thank year wave sausage worth useful legal winner thank yellow";
    assert!(bip39::decode(words).is_ok());
    assert_eq!(bip39::decode(&words.replace(' ', "\u{85}")).unwrap_err(), Error::PhraseLength);
    assert_eq!(bip39::decode(&words.replace(' ', " \u{85} ")).unwrap_err(), Error::PhraseLength);
    // the ones JavaScript does count are separators
    for c in ['\u{b}', '\u{c}', '\u{a0}', '\u{1680}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{feff}'] {
        assert!(bip39::decode(&words.replace(' ', &c.to_string())).is_ok(), "U+{:04X}", c as u32);
    }
}
