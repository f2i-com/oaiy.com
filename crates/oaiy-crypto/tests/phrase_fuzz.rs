//! A deterministic fuzz of the recovery phrase decoder: two hundred thousand generated inputs (the test asserts at least 100,000), checked against a model and against an independent
//! implementation (the `bip39` crate 2.2.2, `rust-bitcoin`'s), never panicking. (`cargo-fuzz` is not needed for a decoder this size; the
//! generator is seeded, so every run is the same run.)
//!
//! Three families: (1) random entropy written as a phrase in a random disguise (case, white space of every Unicode kind, full-width letters,
//! the `fi` ligature) which must decode to that entropy; (2) valid phrases mutated at the word level (swap, drop, duplicate, insert, replace,
//! typo), whose outcome is computed by the model from the words alone; (3) garbage of every code point class, which must never panic and, in the
//! astronomically unlikely case that it decodes, must round-trip.

mod common;

use common::Rng;
use oaiy_crypto::bip39::{self as ours, Entropy};
use oaiy_crypto::Error;

/// What a word-level model says a list of words decodes to, computed with the reference crate.
#[derive(Debug, PartialEq, Eq)]
enum Model {
    Ok([u8; 16]),
    Length,
    Word,
    Checksum,
}

fn reference(words: &[String]) -> Model {
    if words.len() != 12 {
        return Model::Length;
    }
    let list = ours::wordlist();
    if words.iter().any(|w| list.binary_search(&w.as_str()).is_err()) {
        return Model::Word;
    }
    match ::bip39::Mnemonic::parse_in_normalized(::bip39::Language::English, &words.join(" ")) {
        Ok(m) => {
            let entropy = m.to_entropy();
            let mut out = [0u8; 16];
            out.copy_from_slice(&entropy);
            Model::Ok(out)
        }
        Err(::bip39::Error::InvalidChecksum) => Model::Checksum,
        Err(other) => panic!("the reference crate refused twelve known words for another reason: {other:?}"),
    }
}

fn ours_as_model(input: &str) -> Model {
    match ours::decode(input) {
        Ok(entropy) => Model::Ok(*entropy.expose()),
        Err(Error::PhraseLength) => Model::Length,
        Err(Error::PhraseWord) => Model::Word,
        Err(Error::PhraseChecksum) => Model::Checksum,
        Err(other) => panic!("decode returned {other:?}"),
    }
}

const SEPARATORS: [&str; 12] = [" ", " ", "  ", "\t", "\n", "\r\n", " \n ", "\u{a0}", "\u{3000}", "\u{2003}", "\u{feff} ", "\u{2028}"];

fn fullwidth(text: &str) -> String {
    text.chars().map(|c| if c.is_ascii_lowercase() { char::from_u32(0xff41 + (c as u32 - 'a' as u32)).unwrap() } else { c }).collect()
}

fn disguise(words: &[String], rng: &mut Rng) -> String {
    let style = rng.below(6);
    let mut text = String::new();
    if rng.below(4) == 0 {
        text.push_str(SEPARATORS[rng.below(SEPARATORS.len())]);
    }
    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            text.push_str(SEPARATORS[rng.below(SEPARATORS.len())]);
        }
        let rendered = match style {
            0 => word.clone(),
            1 => word.to_uppercase(),
            2 => word.chars().map(|c| if rng.below(2) == 0 { c.to_ascii_uppercase() } else { c }).collect(),
            3 => fullwidth(word),
            4 => word.replace("fi", "\u{fb01}"),
            _ => {
                // a capital first letter, as a phone keyboard writes it
                let mut chars = word.chars();
                chars.next().map(|f| f.to_ascii_uppercase().to_string() + chars.as_str()).unwrap_or_default()
            }
        };
        text.push_str(&rendered);
    }
    if rng.below(4) == 0 {
        text.push_str(SEPARATORS[rng.below(SEPARATORS.len())]);
    }
    text
}

fn words_of(entropy: &Entropy) -> Vec<String> {
    ours::encode(entropy).expose().split(' ').map(str::to_string).collect()
}

fn mutate(words: &[String], rng: &mut Rng) -> Vec<String> {
    let list = ours::wordlist();
    let mut w = words.to_vec();
    match rng.below(8) {
        0 => {
            let (i, j) = (rng.below(w.len()), rng.below(w.len()));
            w.swap(i, j);
        }
        1 => {
            w.remove(rng.below(w.len()));
        }
        2 => {
            let i = rng.below(w.len());
            let dup = w[i].clone();
            w.insert(i, dup);
        }
        3 => {
            let i = rng.below(w.len() + 1);
            w.insert(i, list[rng.below(2048)].to_string());
        }
        4 => {
            let i = rng.below(w.len());
            w[i] = list[rng.below(2048)].to_string();
        }
        5 => {
            // a typo: one letter changed, one dropped, or one added
            let i = rng.below(w.len());
            let mut chars: Vec<char> = w[i].chars().collect();
            let p = rng.below(chars.len());
            match rng.below(3) {
                0 => chars[p] = char::from(b'a' + rng.below(26) as u8),
                1 => {
                    chars.remove(p);
                }
                _ => chars.insert(p, char::from(b'a' + rng.below(26) as u8)),
            }
            w[i] = chars.into_iter().collect();
        }
        6 => {
            let n = 1 + rng.below(3);
            for _ in 0..n {
                w.push(list[rng.below(2048)].to_string());
            }
        }
        _ => {
            let keep = rng.below(w.len() + 1);
            w.truncate(keep);
        }
    }
    w
}

fn garbage(rng: &mut Rng) -> String {
    const POOL: &[&str] = &[
        "a",
        "b",
        "z",
        "abandon",
        "zoo",
        "about",
        " ",
        "  ",
        "\t",
        "\n",
        "\0",
        "\u{a0}",
        "\u{200b}",
        "\u{200d}",
        "\u{feff}",
        "\u{301}",
        "\u{130}",
        "\u{df}",
        "\u{3a3}",
        "\u{fb01}",
        "\u{fdfa}",
        "\u{1f434}",
        "\u{202e}",
        "\u{ff5a}",
        "é",
        "日本語",
        "\u{10ffff}",
        "\u{d7ff}",
        "\u{e000}",
        "-",
        ",",
        ".",
        "|",
        "1",
        "9",
    ];
    let n = rng.below(60);
    (0..n).map(|_| POOL[rng.below(POOL.len())]).collect()
}

#[test]
fn two_hundred_thousand_generated_inputs_never_panic_and_agree_with_the_model_and_the_reference_crate() {
    let mut rng = Rng(0x0a17_9050_beef);
    let mut inputs = 0usize;
    let (mut ok_disguised, mut ok_mutated, mut checksum, mut length, mut word) = (0usize, 0usize, 0usize, 0usize, 0usize);

    // family 1: random entropy, random disguise, must come back
    for _ in 0..50_000 {
        let entropy = Entropy::from_bytes(rng.array::<16>());
        let words = words_of(&entropy);
        // the reference crate gives the same words for the same entropy
        assert_eq!(words.join(" "), ::bip39::Mnemonic::from_entropy(entropy.expose()).unwrap().to_string());
        let input = disguise(&words, &mut rng);
        inputs += 1;
        assert_eq!(ours::decode(&input).unwrap(), entropy, "{input:?}");
        ok_disguised += 1;
        // the plain phrase too
        inputs += 1;
        assert_eq!(ours::decode(&words.join(" ")).unwrap(), entropy);
    }

    // family 2: word-level mutations of valid phrases, outcome from the model
    for _ in 0..60_000 {
        let entropy = Entropy::from_bytes(rng.array::<16>());
        let words = mutate(&words_of(&entropy), &mut rng);
        let expected = reference(&words);
        let input = disguise(&words, &mut rng);
        inputs += 1;
        let got = ours_as_model(&input);
        assert_eq!(got, expected, "{input:?}");
        match got {
            Model::Ok(_) => ok_mutated += 1,
            Model::Checksum => checksum += 1,
            Model::Length => length += 1,
            Model::Word => word += 1,
        }
    }

    // family 3: garbage, no panic, and a decoded result must round-trip
    for _ in 0..40_000 {
        let input = garbage(&mut rng);
        inputs += 1;
        if let Ok(entropy) = ours::decode(&input) {
            assert_eq!(ours::decode(ours::encode(&entropy).expose()).unwrap(), entropy);
        }
    }
    // over-long input is refused for its length before it is looked at
    for extra in [1usize, 2, 100, 5000] {
        let long = format!("{} {}", "abandon ".repeat(11) + "about", " ".repeat(extra + 1024));
        inputs += 1;
        assert_eq!(ours::decode(&long).unwrap_err(), Error::PhraseLength);
    }

    assert!(inputs >= 100_000, "{inputs} inputs");
    // the run exercised every outcome (a fuzz that never reaches an error path proves nothing about it)
    assert!(ok_disguised >= 50_000);
    assert!(ok_mutated > 0, "mutations that are still valid phrases (swapping two equal words, for example)");
    assert!(checksum > 5_000 && length > 5_000 && word > 5_000, "checksum {checksum}, length {length}, word {word}");
}

#[test]
fn precedence_length_then_word_then_checksum() {
    // eleven words, one of which is not a word: length first
    let eleven_bad = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abouu";
    assert_eq!(ours::decode(eleven_bad).unwrap_err(), Error::PhraseLength);
    // twelve words, one unknown: word before checksum
    let unknown = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abouu";
    assert_eq!(ours::decode(unknown).unwrap_err(), Error::PhraseWord);
    // twelve known words with a bad checksum
    let bad_sum = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon";
    assert_eq!(ours::decode(bad_sum).unwrap_err(), Error::PhraseChecksum);
    // an empty string, whitespace only, and a BOM
    for empty in ["", " ", "\n\t ", "\u{feff}"] {
        assert_eq!(ours::decode(empty).unwrap_err(), Error::PhraseLength, "{empty:?}");
    }
}

#[test]
fn nfkd_and_case_and_unicode_white_space() {
    let entropy = Entropy::from_bytes([0x5c; 16]);
    let words = ours::encode(&entropy);
    let plain = words.expose().to_string();
    // full-width letters normalise to ASCII (NFKD), as does a no-break space between the words
    let wide: String = plain
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                char::from_u32(0xff41 + (c as u32 - 'a' as u32)).unwrap()
            } else if c == ' ' {
                '\u{3000}'
            } else {
                c
            }
        })
        .collect();
    assert_eq!(ours::decode(&wide).unwrap(), entropy);
    assert_eq!(ours::decode(&plain.to_uppercase()).unwrap(), entropy);
    assert_eq!(ours::decode(&plain.replace(' ', "\u{a0}")).unwrap(), entropy);
    // the ligature fi (U+FB01) is "fi" after NFKD: "find" style words survive it
    let with_fi = Entropy::from_bytes([0x11; 16]);
    let text = ours::encode(&with_fi).expose().to_string();
    assert_eq!(ours::decode(&text.replace("fi", "\u{fb01}")).unwrap(), with_fi);
    // the same decomposed and composed forms of a non-ASCII letter are both refused (no word has one), and neither panics
    assert_eq!(
        ours::decode("e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301} e\u{301}").unwrap_err(),
        Error::PhraseWord
    );
    // a 13-word phrase that is a valid 12-word phrase plus a word, and a 24-word one, are refused for their length
    assert_eq!(ours::decode(&format!("{plain} abandon")).unwrap_err(), Error::PhraseLength);
    assert_eq!(ours::decode(&format!("{plain} {plain}")).unwrap_err(), Error::PhraseLength);
    // 12 words of another checksum-valid phrase of the reference crate's 15-word kind is not a 12-word phrase
    let fifteen = ::bip39::Mnemonic::from_entropy(&[7u8; 20]).unwrap().to_string();
    assert_eq!(ours::decode(&fifteen).unwrap_err(), Error::PhraseLength);
}
