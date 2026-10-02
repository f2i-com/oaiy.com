//! Strict JSON, and the canonical form the protocol needs in exactly one family (README section 1, "Canonical JSON"): pairing.
//!
//! Why a parser of its own and not `serde_json`: what the protocol refuses is not what a general parser refuses, and the difference is the whole point of a canonical
//! form. The parser here
//!
//! - refuses a duplicate member name (a general parser keeps the last, so two readers can disagree about which value a signed text holds);
//! - tells an integer from a number that is not one by its spelling (`60.0`, `6e1` and `-0` are not integers: Interpretations 2 and 23), and keeps an integer exact
//!   up to 38 digits, so that a cursor of 2^53 is an integer that is out of range and not a float that was rounded;
//! - refuses invalid UTF-8, an unescaped control character, a lone surrogate in an escape, a leading byte-order mark and anything after the value;
//! - never recurses past [`MAX_DEPTH`] (64, the relay's own limit: 65 levels are refused), so that no input can overflow the stack.
//!
//! In **canonical mode** ([`parse_canonical`]) it also refuses what the canonical form has no spelling for: a number that is not an integer, and an integer outside
//! -2^63 to 2^64-1, "so that every integer has exactly one spelling". [`Json::to_canonical`] writes the canonical text: members sorted bytewise by their UTF-8 bytes (the
//! keys of the protocol are ASCII, so this is also the order of RFC 8785's UTF-16 code units), no whitespace, strings escaped as JSON with only quote, backslash and
//! the control characters escaped (`\b \t \n \f \r`, otherwise `\u00xx` in lower case), `/` and everything above ASCII written as it is.
//!
//! **What is never re-serialised.** A MAC or a signature is checked over the exact bytes that were received and only then parsed. The one place the protocol
//! re-serialises is the pairing response's `claims`, which the phone signs in canonical form and the desktop must therefore rebuild from what it parsed.

use core::fmt;

/// The deepest nesting the protocol reads (the relay refuses 65 levels, accepts 64).
pub const MAX_DEPTH: usize = 64;

/// The integers the canonical form can spell: -2^63 to 2^64-1.
pub const CANONICAL_MIN: i128 = -(1i128 << 63);
/// The largest integer the canonical form can spell.
pub const CANONICAL_MAX: i128 = (1i128 << 64) - 1;

/// Why a text was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonError {
    /// The bytes are not valid UTF-8.
    Utf8,
    /// Not JSON (a missing comma, a bare word, a trailing comma, a leading byte-order mark ...).
    Syntax,
    /// Something follows the value.
    TrailingData,
    /// More than [`MAX_DEPTH`] levels of arrays and objects.
    Depth,
    /// An object with the same member name twice.
    DuplicateKey,
    /// A `\u` escape that is not four hex digits, or a surrogate that is not half of a pair.
    BadEscape,
    /// An unescaped control character in a string.
    Control,
    /// Canonical mode: a number that is not an integer (a fraction, an exponent, `-0`).
    NotAnInteger,
    /// Canonical mode: an integer outside -2^63 to 2^64-1.
    IntegerRange,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            JsonError::Utf8 => "not valid UTF-8",
            JsonError::Syntax => "not JSON",
            JsonError::TrailingData => "data after the value",
            JsonError::Depth => "nested too deeply",
            JsonError::DuplicateKey => "a member name is repeated",
            JsonError::BadEscape => "a bad \\u escape",
            JsonError::Control => "an unescaped control character",
            JsonError::NotAnInteger => "a number that is not an integer",
            JsonError::IntegerRange => "an integer outside the canonical range",
        })
    }
}

impl std::error::Error for JsonError {}

/// A JSON number, classified by its spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Number {
    /// An integer spelling (`-?(0|[1-9][0-9]*)`, other than `-0`) of at most 38 digits.
    Int(i128),
    /// An integer spelling of more than 38 digits.
    Big(String),
    /// Any other spelling: a fraction, an exponent, or `-0`. The text is kept as it was written.
    Other(String),
}

/// A JSON value. Objects keep their members in the order they were written (and refuse a repeated name), so that a text can be built in the order a signature covers.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` or `false`
    Bool(bool),
    /// A number.
    Num(Number),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object, in written order.
    Obj(Vec<(String, Json)>),
}

/// Parses `input` (general mode: any spelling of a number is read, and kept as written).
pub fn parse(input: &[u8]) -> Result<Json, JsonError> {
    Parser::run(input, false)
}

/// Parses `input` for the canonical form: a number that is not an integer, a `-0` and an integer outside -2^63 to 2^64-1 are refused.
pub fn parse_canonical(input: &[u8]) -> Result<Json, JsonError> {
    Parser::run(input, true)
}

/// `parse_canonical` then `to_canonical`: the canonical text of `input`.
pub fn canonicalize(input: &[u8]) -> Result<String, JsonError> {
    parse_canonical(input)?.to_canonical()
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    canonical: bool,
}

impl<'a> Parser<'a> {
    fn run(input: &'a [u8], canonical: bool) -> Result<Json, JsonError> {
        std::str::from_utf8(input).map_err(|_| JsonError::Utf8)?;
        let mut p = Parser { b: input, i: 0, canonical };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(JsonError::TrailingData);
        }
        Ok(v)
    }

    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), JsonError> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(JsonError::Syntax)
        }
    }

    fn word(&mut self, w: &[u8]) -> Result<(), JsonError> {
        if self.b[self.i..].starts_with(w) {
            self.i += w.len();
            Ok(())
        } else {
            Err(JsonError::Syntax)
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        match self.peek().ok_or(JsonError::Syntax)? {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.word(b"true").map(|_| Json::Bool(true)),
            b'f' => self.word(b"false").map(|_| Json::Bool(false)),
            b'n' => self.word(b"null").map(|_| Json::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(JsonError::Syntax),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth >= MAX_DEPTH {
            return Err(JsonError::Depth);
        }
        self.eat(b'{')?;
        let mut members: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(members));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value(depth + 1)?;
            members.push((key, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(JsonError::Syntax),
            }
        }
        // A repeated name is refused, in n log n: no object of any size makes this quadratic.
        let mut order: Vec<usize> = (0..members.len()).collect();
        order.sort_by(|&a, &b| members[a].0.as_bytes().cmp(members[b].0.as_bytes()));
        if order.windows(2).any(|w| members[w[0]].0 == members[w[1]].0) {
            return Err(JsonError::DuplicateKey);
        }
        Ok(Json::Obj(members))
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth >= MAX_DEPTH {
            return Err(JsonError::Depth);
        }
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(JsonError::Syntax),
            }
        }
        Ok(Json::Arr(items))
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let digits = self.b.get(self.i..self.i + 4).ok_or(JsonError::BadEscape)?;
        let mut v = 0u32;
        for &d in digits {
            let n = match d {
                b'0'..=b'9' => d - b'0',
                b'a'..=b'f' => d - b'a' + 10,
                b'A'..=b'F' => d - b'A' + 10,
                _ => return Err(JsonError::BadEscape),
            };
            v = (v << 4) | u32::from(n);
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            // A run of ordinary bytes is copied whole: the input was checked to be UTF-8 and a quote, a backslash or a control byte is never part of a multi-byte character.
            let start = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            if self.i > start {
                out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| JsonError::Utf8)?);
            }
            match self.peek().ok_or(JsonError::Syntax)? {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = self.peek().ok_or(JsonError::BadEscape)?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xd800..0xdc00).contains(&hi) {
                                // A high surrogate must be followed by `\u` and a low one.
                                if self.b.get(self.i) != Some(&b'\\') || self.b.get(self.i + 1) != Some(&b'u') {
                                    return Err(JsonError::BadEscape);
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return Err(JsonError::BadEscape);
                                }
                                0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                            } else if (0xdc00..0xe000).contains(&hi) {
                                return Err(JsonError::BadEscape);
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or(JsonError::BadEscape)?);
                        }
                        _ => return Err(JsonError::BadEscape),
                    }
                }
                _ => return Err(JsonError::Control),
            }
        }
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return Err(JsonError::Syntax),
        }
        let int_end = self.i;
        let mut fractional = false;
        if self.peek() == Some(b'.') {
            self.i += 1;
            fractional = true;
            let d = self.i;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
            if self.i == d {
                return Err(JsonError::Syntax);
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            fractional = true;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            let d = self.i;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
            if self.i == d {
                return Err(JsonError::Syntax);
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| JsonError::Syntax)?;
        let negative = text.starts_with('-');
        let digits = &text[usize::from(negative)..int_end - start];
        if fractional || text == "-0" {
            if self.canonical {
                return Err(JsonError::NotAnInteger);
            }
            return Ok(Json::Num(Number::Other(text.to_string())));
        }
        if digits.len() > 38 {
            if self.canonical {
                return Err(JsonError::IntegerRange);
            }
            return Ok(Json::Num(Number::Big(text.to_string())));
        }
        let magnitude: i128 = digits.parse().map_err(|_| JsonError::Syntax)?;
        let v = if negative { -magnitude } else { magnitude };
        if self.canonical && !(CANONICAL_MIN..=CANONICAL_MAX).contains(&v) {
            return Err(JsonError::IntegerRange);
        }
        Ok(Json::Num(Number::Int(v)))
    }
}

impl Json {
    /// An object from members, in the order given.
    pub fn obj<K: Into<String>, const N: usize>(members: [(K, Json); N]) -> Json {
        Json::Obj(members.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A string value.
    pub fn str(text: impl Into<String>) -> Json {
        Json::Str(text.into())
    }

    /// An integer value.
    pub fn int(n: impl Into<i128>) -> Json {
        Json::Num(Number::Int(n.into()))
    }

    /// The member `key` of an object (the first, and only: names are unique), `None` for another kind of value or an absent name.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The text of a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The value of an integer spelling.
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Json::Num(Number::Int(n)) => Some(*n),
            _ => None,
        }
    }

    /// An integer from 0 to 2^64-1.
    pub fn as_u64(&self) -> Option<u64> {
        self.as_int().and_then(|n| u64::try_from(n).ok())
    }

    /// An integer from 0 to 2^53-1 (`uint53` of the schemas: a cursor, a sequence number, a time).
    pub fn as_uint53(&self) -> Option<u64> {
        self.as_u64().filter(|n| *n <= MAX_SAFE_INT)
    }

    /// A boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The elements of an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// The members of an object, in written order.
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    /// True for an object.
    pub fn is_object(&self) -> bool {
        matches!(self, Json::Obj(_))
    }

    /// The text of the string member `key`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Json::as_str)
    }

    /// The `uint53` member `key`.
    pub fn get_uint53(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Json::as_uint53)
    }

    /// The canonical text (see the module): members sorted bytewise, no whitespace. Refuses a number that is not an integer or is out of range, and a tree deeper than
    /// [`MAX_DEPTH`].
    pub fn to_canonical(&self) -> Result<String, JsonError> {
        let mut out = String::new();
        self.write(&mut out, true, 0)?;
        Ok(out)
    }

    /// Compact text in the order the object was built (no sorting, no whitespace): for a request whose member order a signature covers. A number kept as written
    /// (a fraction, an exponent) is written as written.
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        // A tree deeper than the limit cannot be built by the parser; a hand-built one is cut off rather than overflowing the stack.
        let _ = self.write(&mut out, false, 0);
        out
    }

    fn write(&self, out: &mut String, canonical: bool, depth: usize) -> Result<(), JsonError> {
        if depth >= MAX_DEPTH && matches!(self, Json::Arr(_) | Json::Obj(_)) {
            return Err(JsonError::Depth);
        }
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Num(Number::Int(n)) => {
                if canonical && !(CANONICAL_MIN..=CANONICAL_MAX).contains(n) {
                    return Err(JsonError::IntegerRange);
                }
                out.push_str(&n.to_string());
            }
            Json::Num(Number::Big(t)) => {
                if canonical {
                    return Err(JsonError::IntegerRange);
                }
                out.push_str(t);
            }
            Json::Num(Number::Other(t)) => {
                if canonical {
                    return Err(JsonError::NotAnInteger);
                }
                out.push_str(t);
            }
            Json::Str(s) => write_string(out, s),
            Json::Arr(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out, canonical, depth + 1)?;
                }
                out.push(']');
            }
            Json::Obj(members) => {
                out.push('{');
                let mut order: Vec<usize> = (0..members.len()).collect();
                if canonical {
                    order.sort_by(|&a, &b| members[a].0.as_bytes().cmp(members[b].0.as_bytes()));
                }
                for (n, &i) in order.iter().enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    write_string(out, &members[i].0);
                    out.push(':');
                    members[i].1.write(out, canonical, depth + 1)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
}

/// The largest integer a `uint53` holds: 2^53 - 1.
pub const MAX_SAFE_INT: u64 = 9_007_199_254_740_991;

/// A JSON string literal: quote, backslash and control characters escaped (`\b \t \n \f \r`, otherwise `\u00xx` in lower case); everything else as it is.
pub fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A JSON string literal as a new `String`.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    write_string(&mut out, s);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(text: &str) -> Result<String, JsonError> {
        canonicalize(text.as_bytes())
    }

    #[test]
    fn the_vectors_canonical_cases() {
        assert_eq!(canon(r#"{"b":1,"a":2}"#).unwrap(), r#"{"a":2,"b":1}"#);
        assert_eq!(canon(r#"{ "z" : [ 3 , 1 , {"y":true,"x":null} ] , "a" : { } }"#).unwrap(), r#"{"a":{},"z":[3,1,{"x":null,"y":true}]}"#);
        assert_eq!(canon(r#"{"B":1,"a":2,"_":3,"1":4}"#).unwrap(), r#"{"1":4,"B":1,"_":3,"a":2}"#);
        assert_eq!(
            canon("{\"z\":\"é\",\"a\":\"日本\",\"k\":\"\\u0001\\n\\\"\\\\\\/\"}").unwrap(),
            "{\"a\":\"日本\",\"k\":\"\\u0001\\n\\\"\\\\/\",\"z\":\"é\"}"
        );
        let limits = r#"{"a":9007199254740991,"b":9223372036854775807,"c":18446744073709551615,"d":-9223372036854775808,"e":0}"#;
        assert_eq!(canon(limits).unwrap(), limits);
        assert_eq!(canon(r#"{"a":0,"b":-1,"c":-10,"d":10}"#).unwrap(), r#"{"a":0,"b":-1,"c":-10,"d":10}"#);
    }

    #[test]
    fn the_vectors_refusals() {
        for text in [r#"{"n":0.5}"#, r#"{"n":1.0}"#, r#"{"n":1e2}"#, r#"{"a":[1,2,3.5]}"#, r#"{"n":-0}"#, r#"{"a":[1,-0,3]}"#] {
            assert_eq!(canon(text), Err(JsonError::NotAnInteger), "{text}");
        }
        assert_eq!(canon(r#"{"n":18446744073709551616}"#), Err(JsonError::IntegerRange));
        assert_eq!(canon(r#"{"n":-9223372036854775809}"#), Err(JsonError::IntegerRange));
    }

    #[test]
    fn general_mode_keeps_what_canonical_mode_refuses() {
        let v = parse(br#"{"bulkShare":0.75,"n":-0,"huge":123456789012345678901234567890123456789012,"ok":5}"#).unwrap();
        assert_eq!(v.get("bulkShare"), Some(&Json::Num(Number::Other("0.75".into()))));
        assert_eq!(v.get("n"), Some(&Json::Num(Number::Other("-0".into()))));
        assert!(matches!(v.get("huge"), Some(Json::Num(Number::Big(_)))));
        assert_eq!(v.get("ok").and_then(Json::as_int), Some(5));
        // A float is never an integer, whatever its value; 2^53 is an integer and not a uint53.
        assert_eq!(v.get("bulkShare").and_then(Json::as_int), None);
        let big = parse(b"9007199254740992").unwrap();
        assert_eq!(big.as_int(), Some(9_007_199_254_740_992));
        assert_eq!(big.as_uint53(), None);
        assert_eq!(parse(b"9007199254740991").unwrap().as_uint53(), Some(9_007_199_254_740_991));
        assert_eq!(parse(b"-1").unwrap().as_uint53(), None);
    }

    #[test]
    fn duplicate_names_are_refused_at_any_depth_and_any_size() {
        assert_eq!(parse(br#"{"a":1,"a":2}"#), Err(JsonError::DuplicateKey));
        assert_eq!(parse(br#"{"x":{"a":1,"b":2,"a":3}}"#), Err(JsonError::DuplicateKey));
        assert_eq!(parse(br#"[{"k":1,"k":1}]"#), Err(JsonError::DuplicateKey));
        // Names that differ only by an escape spelling are the same name.
        assert_eq!(parse(br#"{"a":1,"\u0061":2}"#), Err(JsonError::DuplicateKey));
        let mut big = String::from("{");
        for i in 0..20_000 {
            big.push_str(&format!("\"k{i}\":{i},"));
        }
        big.push_str("\"k7\":0}");
        assert_eq!(parse(big.as_bytes()), Err(JsonError::DuplicateKey));
    }

    #[test]
    fn nesting_is_limited_to_64_levels() {
        let ok = format!("{}1{}", "[".repeat(64), "]".repeat(64));
        assert!(parse(ok.as_bytes()).is_ok());
        let too_deep = format!("{}1{}", "[".repeat(65), "]".repeat(65));
        assert_eq!(parse(too_deep.as_bytes()), Err(JsonError::Depth));
        // An input far deeper than any stack could take is refused at the 65th level, not parsed.
        let bomb = "[".repeat(1_000_000);
        assert_eq!(parse(bomb.as_bytes()), Err(JsonError::Depth));
        let bomb = "{\"a\":".repeat(100_000);
        assert_eq!(parse(bomb.as_bytes()), Err(JsonError::Depth));
    }

    #[test]
    fn strings_utf8_and_escapes() {
        assert_eq!(parse(br#""\ud83d\ude00""#).unwrap(), Json::Str("\u{1f600}".into()));
        assert_eq!(parse("\"\u{1f600}\"".as_bytes()).unwrap(), Json::Str("\u{1f600}".into()));
        for bad in [r#""\ud83d""#, r#""\ud83dx""#, r#""\ude00""#, r#""\ud83d\u0041""#, r#""\u12""#, r#""\u12G4""#, r#""\x41""#, r#""\ ""#] {
            assert_eq!(parse(bad.as_bytes()), Err(JsonError::BadEscape), "{bad}");
        }
        assert_eq!(parse(b"\"a\x01b\""), Err(JsonError::Control));
        assert_eq!(parse(b"\"a\nb\""), Err(JsonError::Control));
        assert_eq!(parse(b"\"a\x7fb\"").unwrap(), Json::Str("a\u{7f}b".into()));
        assert_eq!(parse(b"\"\xff\""), Err(JsonError::Utf8));
        assert_eq!(parse(b"\"\xc0\xaf\""), Err(JsonError::Utf8), "an overlong slash");
        assert_eq!(parse(b"\"\xed\xa0\x80\""), Err(JsonError::Utf8), "a surrogate written as UTF-8");
        assert_eq!(parse(b"\xef\xbb\xbf{}"), Err(JsonError::Syntax), "a byte-order mark");
    }

    #[test]
    fn syntax_that_is_not_json_is_refused() {
        for bad in [
            "",
            " ",
            "{",
            "}",
            "[1,]",
            "[,1]",
            "{\"a\":1,}",
            "{\"a\"}",
            "{a:1}",
            "'a'",
            "01",
            "1.",
            ".5",
            "+1",
            "1e",
            "--1",
            "nul",
            "True",
            "[1 2]",
            "{} {}",
            "{}x",
            "[]]",
        ] {
            assert!(parse(bad.as_bytes()).is_err(), "{bad:?}");
        }
        for good in ["null", "true", "false", "0", "-1", "[]", "{}", " [ 1 , 2 ] ", "\"\"", "1E5", "-0.0e-1"] {
            assert!(parse(good.as_bytes()).is_ok(), "{good:?}");
        }
    }

    #[test]
    fn the_canonical_writer_escapes_only_what_the_protocol_says() {
        let v = Json::obj([("k", Json::str("a\u{0}b\u{1f}c\u{7f}d\"e\\f/g\u{8}\u{c}\n\r\t \u{2028}é日"))]);
        assert_eq!(v.to_canonical().unwrap(), "{\"k\":\"a\\u0000b\\u001fc\u{7f}d\\\"e\\\\f/g\\b\\f\\n\\r\\t \u{2028}é日\"}");
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn canonical_text_parses_back_to_itself() {
        let text = r#"{"a":[1,2,{"c":"d","b":null}],"B":true,"z":-9223372036854775808}"#;
        let once = canon(text).unwrap();
        assert_eq!(canon(&once).unwrap(), once);
        assert_eq!(parse(once.as_bytes()).unwrap().to_canonical().unwrap(), once);
    }

    #[test]
    fn compact_text_keeps_the_order_and_the_spelling() {
        let v = parse(br#"{ "z" : 1, "a" : [ 1.50 , 2 ] }"#).unwrap();
        assert_eq!(v.to_compact(), r#"{"z":1,"a":[1.50,2]}"#);
    }

    #[test]
    fn a_hand_built_tree_that_is_too_deep_is_an_error_and_not_a_stack_overflow() {
        let mut v = Json::Null;
        for _ in 0..200 {
            v = Json::Arr(vec![v]);
        }
        assert_eq!(v.to_canonical(), Err(JsonError::Depth));
        let _ = v.to_compact();
    }
}
