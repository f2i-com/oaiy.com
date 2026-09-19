//! A small, strict JSON parser: bytes in, a [`Json`] tree out.
//!
//! Used for safetensors headers, `config.json` and `tokenizer.json`. It
//! follows RFC 8259: strings are decoded (escapes and surrogate pairs
//! included), numbers must match the JSON grammar, and anything after the
//! top-level value is an error. The input is untrusted, so nesting is capped
//! at [`MAX_DEPTH`] and every failure is an `Error::Format` naming the byte
//! offset, never a panic.
//!
//! Numbers are held as `f64`, which is exact for every integer up to 2^53.
//! That is far past any tensor offset or shape a checkpoint can hold.

use crate::error::{Error, Result};

/// Deepest nesting accepted. Real files nest a handful of levels; the cap
/// keeps a hostile file from overflowing the stack of the recursive parser.
pub const MAX_DEPTH: usize = 128;

/// A parsed JSON value. Objects keep their members in file order.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parse one complete JSON document.
    pub fn parse(src: &[u8]) -> Result<Json> {
        let mut p = Parser { src, pos: 0 };
        p.skip_ws();
        let v = p.value(0)?;
        p.skip_ws();
        if p.pos != src.len() {
            return Err(p.fail("unexpected data after the value"));
        }
        Ok(v)
    }

    /// Member `key` of an object (the first, if the key repeats); `None`
    /// for a missing key or a non-object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Element `i` of an array; `None` out of range or for a non-array.
    pub fn at(&self, i: usize) -> Option<&Json> {
        self.as_array()?.get(i)
    }

    /// An object's `(key, value)` pairs in file order; empty for anything
    /// else.
    pub fn members(&self) -> impl Iterator<Item = (&str, &Json)> {
        self.as_object().unwrap_or(&[]).iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Elements of an array, or members of an object; 0 for a scalar.
    pub fn len(&self) -> usize {
        match self {
            Json::Arr(a) => a.len(),
            Json::Obj(o) => o.len(),
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }

    /// A number with no fractional part that `f64` holds exactly (|v| up to
    /// 2^53). `1e3` is 1000; `1.5` and `1e300` are `None`.
    pub fn as_i64(&self) -> Option<i64> {
        const EXACT: f64 = 9_007_199_254_740_992.0; // 2^53
        match self {
            Json::Num(v) if v.fract() == 0.0 && v.abs() <= EXACT => Some(*v as i64),
            _ => None,
        }
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn fail(&self, what: &str) -> Error {
        Error::Format(format!("JSON, byte {}: {what}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn eat(&mut self, want: u8) -> Result<()> {
        match self.bump() {
            Some(b) if b == want => Ok(()),
            Some(_) => {
                self.pos -= 1;
                Err(self.fail(&format!("expected '{}'", want as char)))
            }
            None => Err(self.fail(&format!("expected '{}', found the end", want as char))),
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > MAX_DEPTH {
            return Err(self.fail("nested too deeply"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.word(b"true", Json::Bool(true)),
            Some(b'f') => self.word(b"false", Json::Bool(false)),
            Some(b'n') => self.word(b"null", Json::Null),
            Some(b'-' | b'0'..=b'9') => Ok(Json::Num(self.number()?)),
            Some(_) => Err(self.fail("expected a value")),
            None => Err(self.fail("expected a value, found the end")),
        }
    }

    fn word(&mut self, w: &[u8], v: Json) -> Result<Json> {
        if self.src[self.pos..].starts_with(w) {
            self.pos += w.len();
            Ok(v)
        } else {
            Err(self.fail("unknown literal"))
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json> {
        self.eat(b'{')?;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(members));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.fail("expected a string key"));
            }
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            let v = self.value(depth + 1)?;
            members.push((key, v));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(members));
                }
                _ => return Err(self.fail("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.fail("expected ',' or ']'")),
            }
        }
    }

    /// A string body, escapes decoded. Plain runs are copied in bulk; the
    /// result must be valid UTF-8.
    fn string(&mut self) -> Result<String> {
        self.eat(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let run = self.src[self.pos..]
                .iter()
                .position(|&b| b == b'"' || b == b'\\' || b < 0x20)
                .ok_or_else(|| self.fail("unterminated string"))?;
            out.extend_from_slice(&self.src[self.pos..self.pos + run]);
            self.pos += run;
            match self.bump() {
                Some(b'"') => break,
                Some(b'\\') => self.escape(&mut out)?,
                _ => {
                    self.pos -= 1;
                    return Err(self.fail("control character in a string"));
                }
            }
        }
        String::from_utf8(out).map_err(|_| self.fail("string is not valid UTF-8"))
    }

    fn escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let c = match self.bump() {
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{8}',
            Some(b'f') => '\u{c}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(b'u') => {
                let first = self.hex4()?;
                let code = match first {
                    0xD800..=0xDBFF => {
                        // a high surrogate must be followed by a low one
                        if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                            return Err(self.fail("unpaired surrogate"));
                        }
                        let second = self.hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&second) {
                            return Err(self.fail("unpaired surrogate"));
                        }
                        0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                    }
                    0xDC00..=0xDFFF => return Err(self.fail("unpaired surrogate")),
                    c => c,
                };
                char::from_u32(code).ok_or_else(|| self.fail("bad \\u escape"))?
            }
            _ => return Err(self.fail("bad escape")),
        };
        let mut buf = [0u8; 4];
        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .src
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.fail("short \\u escape"))?;
        let mut v = 0u32;
        for &d in digits {
            let n = (d as char).to_digit(16).ok_or_else(|| self.fail("bad \\u escape"))?;
            v = v * 16 + n;
        }
        self.pos += 4;
        Ok(v)
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`
    fn number(&mut self) -> Result<f64> {
        let start = self.pos;
        let digits = |p: &mut Self| {
            let n = p.src[p.pos..].iter().take_while(|b| b.is_ascii_digit()).count();
            p.pos += n;
            n
        };
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return Err(self.fail("bad number")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if digits(self) == 0 {
                return Err(self.fail("bad number: no digits after '.'"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if digits(self) == 0 {
                return Err(self.fail("bad number: no exponent digits"));
            }
        }
        // the slice is ASCII by construction
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        text.parse::<f64>().map_err(|_| self.fail("bad number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Json {
        Json::parse(s.as_bytes()).unwrap_or_else(|e| panic!("{s}: {e}"))
    }

    #[test]
    fn scalars() {
        assert_eq!(p("null"), Json::Null);
        assert_eq!(p(" true "), Json::Bool(true));
        assert_eq!(p("false"), Json::Bool(false));
        assert_eq!(p("-12.5e1"), Json::Num(-125.0));
        assert_eq!(p("0"), Json::Num(0.0));
        assert_eq!(p("\"hi\""), Json::Str("hi".into()));
    }

    #[test]
    fn nested_access() {
        let v = p(r#"{"a": {"x": [1, {"y": 2}]}, "b": [3], "c": "s", "a": 9}"#);
        let keys: Vec<&str> = v.members().map(|(k, _)| k).collect();
        assert_eq!(keys, ["a", "b", "c", "a"]);
        // the first of a repeated key wins
        let y = v.get("a").and_then(|a| a.get("x")).and_then(|x| x.at(1)).and_then(|o| o.get("y"));
        assert_eq!(y.and_then(Json::as_i64), Some(2));
        assert_eq!(v.get("b").map(Json::len), Some(1));
        assert_eq!(v.get("c").and_then(Json::as_str), Some("s"));
        assert!(v.get("missing").is_none());
        // wrong-type access is None, not a panic
        assert!(v.get("c").and_then(|c| c.get("x")).is_none());
        assert!(v.get("b").and_then(|b| b.at(5)).is_none());
        assert_eq!(v.get("c").map(|c| c.members().count()), Some(0));
    }

    #[test]
    fn empty_containers() {
        let v = p(r#"{"o": {}, "a": [ ]}"#);
        assert!(v.get("o").is_some_and(Json::is_empty));
        assert!(v.get("a").is_some_and(Json::is_empty));
    }

    #[test]
    fn strings_are_decoded() {
        assert_eq!(p(r#""a\"b\\c\/d\n\t""#).as_str(), Some("a\"b\\c/d\n\t"));
        assert_eq!(p(r#""\u00e9\u0120""#).as_str(), Some("éĠ"));
        // a surrogate pair is one character
        assert_eq!(p(r#""\ud83d\ude00""#).as_str(), Some("😀"));
        // raw UTF-8 passes through
        assert_eq!(p("\"日本\"").as_str(), Some("日本"));
    }

    #[test]
    fn integers() {
        assert_eq!(p("1e3").as_i64(), Some(1000));
        assert_eq!(p("-7").as_i64(), Some(-7));
        assert_eq!(p("12884901888").as_i64(), Some(12_884_901_888)); // > 4 GB offset
        assert_eq!(p("1.5").as_i64(), None);
        assert_eq!(p("1e300").as_i64(), None);
        assert_eq!(p("\"3\"").as_i64(), None);
    }

    #[test]
    fn malformed_input_is_an_error() {
        for bad in [
            "", "{", "[1, 2", "{\"a\" 1}", "{\"a\": 1,}", "[1,]", "{a: 1}", "tru", "nul",
            "01", "1.", "-", "1e", ".5", "+1", "\"open", "\"a\\x\"", "\"\\u12\"",
            "\"\\ud800\"", "\"\\udc00\"", "\"\\ud800\\u0041\"", "\"tab\there\"", "1 2", "[] x",
        ] {
            assert!(Json::parse(bad.as_bytes()).is_err(), "{bad:?} parsed");
        }
        assert!(Json::parse(b"\"\xff\"").is_err(), "invalid UTF-8 accepted");
    }

    #[test]
    fn depth_is_capped() {
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(Json::parse(ok.as_bytes()).is_ok());
        let deep = "[".repeat(100_000);
        assert!(Json::parse(deep.as_bytes()).is_err());
    }
}
