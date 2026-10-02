//! Shared helpers of the integration tests: where the protocol package is, and a small way to walk a JSON document that panics, with the path, when it is not there.

#![allow(dead_code)]

use std::path::PathBuf;

use oaiy_relay_core::json::{self, Json};

/// `platform/protocol/relay/v1` of this repository: the contract, read in place so that a change to it shows up as a failing test here.
pub fn protocol_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../platform/protocol/relay/v1")
}

/// A file of the protocol package, as text.
pub fn read_text(relative: &str) -> String {
    let path = protocol_dir().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// A JSON file of the protocol package, parsed by this crate's own parser (which refuses duplicate members, so a fixture that has one fails to load).
pub fn load(relative: &str) -> Json {
    json::parse(read_text(relative).as_bytes()).unwrap_or_else(|e| panic!("{relative}: {e}"))
}

/// Walking a document by a dotted path (`A3.expected.pid`); a number in the path indexes an array.
pub trait At {
    /// The value at `path`, or a panic naming the path.
    fn at(&self, path: &str) -> &Json;
    /// The string at `path`.
    fn s(&self, path: &str) -> &str {
        self.at(path).as_str().unwrap_or_else(|| panic!("{path}: not a string"))
    }
    /// The integer at `path`.
    fn n(&self, path: &str) -> u64 {
        self.at(path).as_u64().unwrap_or_else(|| panic!("{path}: not an integer"))
    }
}

impl At for Json {
    fn at(&self, path: &str) -> &Json {
        let mut cur = self;
        for part in path.split('.') {
            cur = match (cur, part.parse::<usize>()) {
                (Json::Arr(items), Ok(i)) => items.get(i),
                (v, _) => v.get(part),
            }
            .unwrap_or_else(|| panic!("no {part} in {path}"));
        }
        cur
    }
}

/// A small seeded generator (SplitMix64): the tests that generate input are reproducible from their seed, which is printed when they fail.
pub struct Rng(pub u64);

impl Rng {
    /// The next 64 bits.
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `n` (`n` > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// True once in `n`.
    pub fn one_in(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }

    /// `len` bytes.
    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }

    /// One of `items`.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

/// Hex to bytes.
pub fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2), "odd hex: {text}");
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap_or_else(|_| panic!("hex: {text}"))).collect()
}

/// Hex to a 32-byte array.
pub fn unhex32(text: &str) -> [u8; 32] {
    unhex(text).try_into().unwrap_or_else(|_| panic!("not 32 bytes: {text}"))
}

/// Bytes to lower-case hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
