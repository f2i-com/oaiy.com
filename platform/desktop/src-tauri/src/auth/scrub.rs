//! Scrubbed logs (design 4.12): every line any log route returns, and every line the `logs_tail`
//! control tool hands the model, passes through [`scrub_line`], because service, plugin, node, python
//! and engine logs carry captured child output.
//!
//! It replaces with `[redacted]`:
//!
//! 1. the token grammar (`oaiy(pat|ses|dsk|run|con|dev)_<id>_<secret>`, and any longer-than-a-prefix
//!    start of one, in case a line was cut) and the legacy `oaiypat_<64 hex>`, also when its underscores
//!    are percent-encoded (`%5F`) or JSON-escaped (`\u005f`);
//! 2. `Bearer <token>`, and the value after `Authorization:`, `Cookie:` and `Set-Cookie:` (to the end
//!    of the line);
//! 3. the shape of a setup code or an invite (`K7QX-4M2P`, `M6SC-7N75-YR3H`; in lower case only when it
//!    has a digit; with any of the dashes a keyboard or a word processor makes);
//! 4. the value of a `key=value`, `key: value`, `key => value`, JSON `"key":"value"` or `--key value`
//!    whose key contains `key`, `token`, `secret`, `password`, `passphrase`, `phrase`, `mnemonic`,
//!    `authorization`, `cookie` or `csrf`. An unquoted value runs to the next whitespace (to the end of the
//!    line for a password, a passphrase, a secret, an authorization or a cookie, whose values may hold
//!    spaces and any punctuation); in JSON, to the next `,`, `}` or `]`. A quoted value runs to its closing
//!    quote, whatever it holds, and knows the escapes: `\"password\":\"x\"` (JSON inside a JSON string)
//!    is a pair too;
//! 5. a word that starts like a secret value (`sk-`, `sk_live_`, `ghp_`, `hf_`, and the token prefixes:
//!    the list of `control::audit::secret_value`), an AWS access key id (`AKIA` and 16 more) and a JSON
//!    web token (`eyJ…​.…​.…`);
//! 6. the password of a URL (`https://user:password@host`), and the whole userinfo when it has no name
//!    and is long enough to be a token.
//!
//! It over-redacts on purpose: a log line that lost a harmless word is a smaller loss than one that
//! kept a credential. Nothing here is a regular expression: the scanners are small, every one is
//! linear in the line, and they fail toward redaction.
//!
//! What it does not do, and the test corpus says so by name: a code typed with spaces (`K7QX 4M2P`), a
//! `-p <password>` (a short flag is a port, a path or a count as often as a password), a key or a
//! certificate on lines of its own, and a secret with no shape and no key.

use crate::control::audit::{secret_value, REDACTED};

#[cfg(test)]
mod corpus;

const KEY_WORDS: [&str; 10] = [
    "key",
    "token",
    "secret",
    "password",
    "passphrase",
    "phrase",
    "mnemonic",
    "authorization",
    "cookie",
    "csrf",
];
/// Keys whose unquoted value may hold spaces and punctuation: it runs to the end of the line.
const LONG_VALUE_WORDS: [&str; 7] = [
    "password",
    "passphrase",
    "phrase",
    "mnemonic",
    "secret",
    "authorization",
    "cookie",
];
const HEADERS: [&str; 3] = ["authorization:", "set-cookie:", "cookie:"];
const KINDS: [&str; 6] = ["pat", "ses", "dsk", "run", "con", "dev"];
/// The alphabet of a setup code or an invite (Crockford base32).
const CROCKFORD: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
/// The most of a token-looking run that is looked at (a token is 68 bytes, a legacy one 72).
const SCAN: usize = 96;
/// The characters of a token that one look at a candidate reads: `oaiy`, the kind, `_`, a run, `_`, a run.
const WINDOW: usize = 4 + 3 + 1 + SCAN + 1 + SCAN;
/// The shortest word after `Bearer` that is hidden (a real one is far longer; a shorter word is prose).
const MIN_BEARER: usize = 4;

fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// The longest line scrubbed whole; a longer one is cut (a log line that long is not a log line, and
/// every scan below is linear in it).
pub const MAX_LINE: usize = 32 * 1024;

/// `line` with what looks like a credential replaced by `[redacted]`.
pub fn scrub_line(line: &str) -> String {
    let cut;
    let line = if line.len() > MAX_LINE {
        let mut end = MAX_LINE;
        while !line.is_char_boundary(end) {
            end -= 1;
        }
        cut = format!("{} ... [line cut]", &line[..end]);
        cut.as_str()
    } else {
        line
    };
    scrub_whole(line)
}

/// [`scrub_line`] with no cut: every stage is linear in the line, and the tests hold them to it.
fn scrub_whole(line: &str) -> String {
    let s = header_values(line);
    let s = tokens(&s);
    let s = bearers(&s);
    let s = urls(&s);
    let s = pairs(&s);
    let s = secret_words(&s);
    codes(&s)
}

/// Rule 2 (headers): from `Authorization:`, `Cookie:` or `Set-Cookie:` to the end of the line.
fn header_values(s: &str) -> String {
    let lower = s.to_ascii_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    'outer: while i < s.len() {
        for h in HEADERS {
            let at_boundary = i == 0 || !lower.as_bytes()[i - 1].is_ascii_alphanumeric();
            if at_boundary && lower[i..].starts_with(h) {
                let after = i + h.len();
                let value_start = after
                    + s[after..]
                        .bytes()
                        .take_while(|b| *b == b' ' || *b == b'\t')
                        .count();
                if s[value_start..].is_empty() || &s[value_start..] == REDACTED {
                    // Nothing to hide, or already hidden.
                    out.push_str(&s[i..]);
                } else {
                    out.push_str(&s[i..value_start]);
                    out.push_str(REDACTED);
                }
                i = s.len();
                continue 'outer;
            }
        }
        // Copy one character (a header name is ASCII, the rest may not be).
        let ch = s[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Rule 1: tokens, whole or cut short, and legacy pairing tokens.
fn tokens(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let mut copied = 0;
    while i < b.len() {
        if b[i] == b'o' && s[i..].starts_with("oaiy") {
            if let Some(end) = token_end(&s[i..]) {
                out.push_str(&s[copied..i]);
                out.push_str(REDACTED);
                i += end;
                copied = i;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&s[copied..]);
    out
}

/// The byte a character of the token grammar stands for, and how many bytes of the line it takes: a
/// percent-encoded underscore (`%5F`) and a JSON-escaped one (`\u005f`) are the `_` of a token.
fn logical(b: &[u8], i: usize) -> Option<(u8, usize)> {
    let c = *b.get(i)?;
    if c == b'%' && b.get(i + 1) == Some(&b'5') && matches!(b.get(i + 2), Some(b'F' | b'f')) {
        return Some((b'_', 3));
    }
    if c == b'\\'
        && b.get(i + 1) == Some(&b'u')
        && b.get(i + 2) == Some(&b'0')
        && b.get(i + 3) == Some(&b'0')
        && b.get(i + 4) == Some(&b'5')
        && matches!(b.get(i + 5), Some(b'F' | b'f'))
    {
        return Some((b'_', 6));
    }
    Some((c, 1))
}

/// If `s` starts with a token (or a start of one worth hiding), how many bytes of `s` it is.
fn token_end(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    // The line as the grammar reads it, for as far as the grammar can reach, and where each character ends.
    let mut win = [0u8; WINDOW];
    let mut ends = [0usize; WINDOW];
    let (mut n, mut at) = (0, 0);
    while n < WINDOW {
        let Some((c, len)) = logical(b, at) else {
            break;
        };
        at += len;
        win[n] = c;
        ends[n] = at;
        n += 1;
    }
    let w = &win[..n];
    let kind = KINDS.iter().find(|k| w[4..].starts_with(k.as_bytes()))?;
    let after_kind = 4 + kind.len();
    if w.get(after_kind) != Some(&b'_') {
        return None;
    }
    let body = after_kind + 1;
    // Every run is bounded, so that a line made of near-misses costs a little for each and not a scan
    // of the rest of the line for each.
    let run = w[body..]
        .iter()
        .take(SCAN)
        .take_while(|c| is_token_char(**c))
        .count();
    // The grammar: 16 hex, `_`, 43 base64url. A cut one: the same start and at least 8 secret characters.
    let id_len = w[body..]
        .iter()
        .take(SCAN)
        .take_while(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
        .count();
    if id_len >= 16 && w.get(body + 16) == Some(&b'_') {
        let secret = w[body + 17..]
            .iter()
            .take(SCAN)
            .take_while(|c| is_token_char(**c))
            .count();
        if secret >= 8 {
            return Some(ends[body + 17 + secret - 1]);
        }
    }
    // A legacy pairing token (`oaiypat_` and 64 hex), or the start of one, and whatever is glued to it (a
    // suffix of letters or `_x`): the run it sits in is hidden with it.
    if *kind == "pat" && id_len >= 8 {
        return Some(ends[body + run - 1]);
    }
    None
}

/// Rule 2 (bearer): `Bearer <word>`.
fn bearers(s: &str) -> String {
    let lower = s.to_ascii_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let boundary = i == 0 || !lower.as_bytes()[i - 1].is_ascii_alphanumeric();
        if boundary && lower[i..].starts_with("bearer") {
            let mut j = i + "bearer".len();
            let spaces = s[j..]
                .bytes()
                .take_while(|b| *b == b' ' || *b == b'\t')
                .count();
            if spaces > 0 {
                j += spaces;
                let word = s[j..]
                    .bytes()
                    .take_while(|b| {
                        !b.is_ascii_whitespace()
                            && !matches!(b, b'"' | b'\'' | b',' | b';' | b')' | b'}' | b']')
                    })
                    .count();
                if word >= MIN_BEARER && &s[j..j + word] != REDACTED {
                    out.push_str(&s[i..j]);
                    out.push_str(REDACTED);
                    i = j + word;
                    continue;
                }
            }
        }
        let ch = s[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Rule 6: the password of `scheme://user:password@host`, wherever a URL is in the line.
fn urls(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut copied = 0;
    let mut from = 0;
    while let Some(p) = s[from..].find("://") {
        let a = from + p + 3;
        from = a;
        // The authority ends at the first character that cannot be in one.
        let mut e = a;
        while e < b.len()
            && !b[e].is_ascii_whitespace()
            && !matches!(
                b[e],
                b'/' | b'?' | b'#' | b'"' | b'\'' | b'<' | b'>' | b'\\'
            )
        {
            e += 1;
        }
        let authority = &s[a..e];
        // The last `@` is the one that ends the userinfo (a password may hold one).
        let Some(at) = authority.rfind('@') else {
            continue;
        };
        let user = &authority[..at];
        let (hide_from, hide_to) = match user.find(':') {
            Some(colon) => (a + colon + 1, a + at),
            // A userinfo with no password that is long enough to be a token itself.
            None if user.len() >= 20 => (a, a + at),
            None => continue,
        };
        if hide_from >= hide_to || &s[hide_from..hide_to] == REDACTED {
            continue;
        }
        out.push_str(&s[copied..hide_from]);
        out.push_str(REDACTED);
        copied = hide_to;
    }
    out.push_str(&s[copied..]);
    out
}

fn key_word(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    KEY_WORDS.iter().any(|k| lower.contains(k))
}

fn long_value_key(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    LONG_VALUE_WORDS.iter().any(|k| lower.contains(k))
}

fn is_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.')
}

fn skip_spaces(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ' | b'\t')) {
        i += 1;
    }
    i
}

/// The separator between a key and its value at `b[i]` and how long it is: `=`, `:`, `=>`, `:=` and the
/// full-width colon.
fn separator(b: &[u8], i: usize) -> Option<usize> {
    match *b.get(i)? {
        b'=' if b.get(i + 1) == Some(&b'>') => Some(2),
        b':' if b.get(i + 1) == Some(&b'=') => Some(2),
        b'=' | b':' => Some(1),
        0xEF if b.get(i + 1) == Some(&0xBC) && b.get(i + 2) == Some(&0x9A) => Some(3),
        _ => None,
    }
}

/// Where a quoted value ends: after the closing quote, when it has one (`true`), else at the end of the
/// line. `from` is the first byte after the opening quote; the quote is `\`×`escapes` then `q`. With no
/// backslashes a `\` escapes what follows; with some (JSON inside a JSON string), the value ends at that
/// many backslashes and the quote, and a quote with fewer or more is part of it.
fn quoted_end(b: &[u8], from: usize, escapes: usize, q: u8) -> (usize, bool) {
    let mut j = from;
    while j < b.len() {
        if b[j] == b'\\' {
            let run = b[j..].iter().take_while(|c| **c == b'\\').count();
            if b.get(j + run) == Some(&q) {
                let closes = if escapes == 0 {
                    run % 2 == 0
                } else {
                    run == escapes
                };
                if closes {
                    return (j + run + 1, true);
                }
                j += run + 1;
            } else {
                j += run;
            }
            continue;
        }
        if b[j] == q && escapes == 0 {
            return (j + 1, true);
        }
        j += 1;
    }
    (b.len(), false)
}

/// A value found after a key: `s[keep_to..end]` is the value, with its quotes.
struct Span {
    keep_to: usize,
    end: usize,
    /// The quote that opens it: how many backslashes come before it, and which quote.
    quote: Option<(usize, u8)>,
    closed: bool,
}

/// The value of the pair whose key ends at `i`, if there is one.
fn value_span(s: &str, i: usize, name: &str) -> Option<Span> {
    let b = s.as_bytes();
    // An (escaped) closing quote of a quoted key: `"password"`, `\"password\"`.
    let mut q = i;
    while b.get(q) == Some(&b'\\') {
        q += 1;
    }
    let quoted_key = matches!(b.get(q), Some(b'"' | b'\''));
    let after_key = if quoted_key { q + 1 } else { i };
    let sep_at = skip_spaces(b, after_key);
    let Some(sep_len) = separator(b, sep_at) else {
        return flag_value(s, i, name, quoted_key);
    };
    let k = skip_spaces(b, sep_at + sep_len);
    if k >= b.len() {
        return None;
    }
    let escapes = b[k..].iter().take_while(|c| **c == b'\\').count();
    if let Some(&quote @ (b'"' | b'\'')) = b.get(k + escapes) {
        let (end, closed) = quoted_end(b, k + escapes + 1, escapes, quote);
        return Some(Span {
            keep_to: k,
            end,
            quote: Some((escapes, quote)),
            closed,
        });
    }
    if let open @ (b'{' | b'[') = b[k] {
        let close = if open == b'{' { b'}' } else { b']' };
        let mut depth = 0i32;
        let mut end = b.len();
        for (j, c) in b.iter().enumerate().skip(k) {
            if *c == open {
                depth += 1;
            } else if *c == close {
                depth -= 1;
                if depth == 0 {
                    end = j + 1;
                    break;
                }
            }
        }
        return Some(Span {
            keep_to: k,
            end,
            quote: None,
            closed: false,
        });
    }
    // Not quoted. A password, a secret and the like run to the end of the line (a JSON value ends at its
    // comma or its closing bracket, since the key was quoted); any other key, to the next whitespace.
    let json = quoted_key;
    let mut end = k;
    if long_value_key(name) && !json {
        end = b.len();
    } else {
        while end < b.len()
            && !b[end].is_ascii_whitespace()
            && !(json && matches!(b[end], b',' | b'}' | b']'))
        {
            end += 1;
        }
    }
    (end > k).then_some(Span {
        keep_to: k,
        end,
        quote: None,
        closed: false,
    })
}

/// `--password hunter2`: a long flag whose name is a secret's, and the word after it.
fn flag_value(s: &str, i: usize, name: &str, quoted_key: bool) -> Option<Span> {
    let b = s.as_bytes();
    if quoted_key || !name.starts_with("--") || name.len() <= 2 {
        return None;
    }
    let k = skip_spaces(b, i);
    // A space, then a word that is not another flag.
    if k == i || k >= b.len() || b[k] == b'-' {
        return None;
    }
    if let quote @ (b'"' | b'\'') = b[k] {
        let (end, closed) = quoted_end(b, k + 1, 0, quote);
        return Some(Span {
            keep_to: k,
            end,
            quote: Some((0, quote)),
            closed,
        });
    }
    let end = k + b[k..]
        .iter()
        .take_while(|c| !c.is_ascii_whitespace())
        .count();
    Some(Span {
        keep_to: k,
        end,
        quote: None,
        closed: false,
    })
}

/// Rule 4: `key=value`, `key: value`, `"key":"value"` and `--key value` where the key names a secret.
fn pairs(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if !is_name_byte(b[i]) {
            let ch = s[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        // A key is a run of name characters.
        let start = i;
        while i < b.len() && is_name_byte(b[i]) {
            i += 1;
        }
        out.push_str(&s[start..i]);
        let name = &s[start..i];
        if !key_word(name) {
            continue;
        }
        let Some(span) = value_span(s, i, name) else {
            continue;
        };
        let value = &s[span.keep_to..span.end];
        // A value that is already hidden (`[redacted]` is a bracketed value, so it ends at its `]`, and what
        // follows it is not taken for part of it).
        if value == REDACTED {
            continue;
        }
        // Keep the key, the separator, the quotes; hide what is between them.
        out.push_str(&s[i..span.keep_to]);
        match span.quote {
            Some((escapes, q)) => {
                let quote: String = "\\".repeat(escapes) + &(q as char).to_string();
                let inner_start = span.keep_to + quote.len();
                let inner_end = if span.closed {
                    span.end - quote.len()
                } else {
                    span.end
                };
                if s[inner_start..inner_end] == *REDACTED {
                    out.push_str(&s[span.keep_to..span.end]);
                } else {
                    out.push_str(&quote);
                    out.push_str(REDACTED);
                    if span.closed {
                        out.push_str(&quote);
                    }
                }
            }
            None => out.push_str(REDACTED),
        }
        i = span.end;
    }
    out
}

/// Whether `word` is a JSON web token: three base64url parts, the first two at least a few characters, the
/// first starting `eyJ` (the base64 of `{"`).
fn is_jwt(word: &str) -> bool {
    let word = word.trim_end_matches(['.', ':', '!']);
    let parts: Vec<&str> = word.split('.').collect();
    word.starts_with("eyJ")
        && parts.len() == 3
        && parts[0].len() >= 8
        && parts[1].len() >= 8
        && !parts[2].is_empty()
        && parts.iter().all(|p| {
            p.bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        })
}

/// `AKIA` or `ASIA` and 16 upper-case letters and digits: an AWS access key id.
fn is_aws_key_id(word: &str) -> bool {
    let b = word.as_bytes();
    b.len() >= 20
        && (b.starts_with(b"AKIA") || b.starts_with(b"ASIA"))
        && b[4..20]
            .iter()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

/// A Stripe key: `sk_live_`, `sk_test_`, `rk_live_` or `rk_test_` and a run.
fn is_stripe_key(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    ["sk_live_", "sk_test_", "rk_live_", "rk_test_"]
        .iter()
        .any(|p| lower.starts_with(p) && word.len() > p.len() + 8)
}

/// Rule 5: a word that starts like a secret value, wherever it is.
fn secret_words(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if !word.is_empty() {
            if secret_value(word) || is_stripe_key(word) || is_aws_key_id(word) || is_jwt(word) {
                out.push_str(REDACTED);
            } else {
                out.push_str(word);
            }
            word.clear();
        }
    };
    for ch in s.chars() {
        if ch.is_whitespace()
            || matches!(
                ch,
                '"' | '\''
                    | ','
                    | ';'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '='
                    | '<'
                    | '>'
                    | '&'
                    | '?'
            )
        {
            flush(&mut word, &mut out);
            out.push(ch);
        } else {
            word.push(ch);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// A dash: the hyphen and the ones a keyboard, a phone or a word processor puts in its place.
fn is_dash(c: char) -> bool {
    matches!(
        c,
        '-' | '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}'
    )
}

/// Rule 3: a setup code or an invite (`K7QX-4M2P`, `M6SC-7N75-YR3H`).
fn codes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if !word.is_empty() {
            if is_code(word) {
                out.push_str(REDACTED);
            } else {
                out.push_str(word);
            }
            word.clear();
        }
    };
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || is_dash(ch) {
            word.push(ch);
        } else {
            flush(&mut word, &mut out);
            out.push(ch);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// `[0-9A-HJKMNP-TV-Z]{4}(-[0-9A-HJKMNP-TV-Z]{4}){1,2}`, the whole word. In lower case (or mixed) it is
/// hidden only with a digit in it: nearly every real code has one, and a word pair such as `sync-data` has
/// none.
fn is_code(word: &str) -> bool {
    let normal: String = word
        .chars()
        .map(|c| if is_dash(c) { '-' } else { c })
        .collect();
    let groups: Vec<&str> = normal.split('-').collect();
    let shaped = (2..=3).contains(&groups.len())
        && groups.iter().all(|g| {
            g.len() == 4
                && g.bytes()
                    .all(|b| CROCKFORD.contains(&b.to_ascii_uppercase()))
        });
    shaped
        && (!normal.bytes().any(|b| b.is_ascii_lowercase())
            || normal.bytes().any(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str = "oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    const LEGACY: &str = "oaiypat_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn scrubbed(line: &str) -> String {
        scrub_line(line)
    }

    // One test per rule, on lines that only that rule can hide, so that each rule is held by a test
    // of its own (the corpus holds them together).

    #[test]
    fn rule_1_a_token_is_hidden_even_inside_a_word_or_cut_short() {
        for (kind, what) in [
            ("pat", "a"),
            ("ses", "b"),
            ("dsk", "c"),
            ("run", "d"),
            ("con", "e"),
            ("dev", "f"),
        ] {
            let token = TOKEN.replace("oaiypat", &format!("oaiy{kind}"));
            assert_eq!(
                scrubbed(&format!("x{what}{token}")),
                format!("x{what}[redacted]"),
                "{kind}"
            );
        }
        assert_eq!(
            scrubbed("id=abc-oaiypat_0123456789abcdef_AAECAwQFBgcI"),
            "id=abc-[redacted]"
        );
        assert_eq!(scrubbed(&format!("_{LEGACY}")), "_[redacted]");
        // What is glued to a legacy token is hidden with it: the run it sits in.
        assert_eq!(
            scrubbed(&format!("paired {LEGACY}xyz ok")),
            "paired [redacted] ok"
        );
        assert_eq!(
            scrubbed(&format!("paired {LEGACY}_x ok")),
            "paired [redacted] ok"
        );
        // What is not the start of a token stays.
        assert_eq!(
            scrubbed("oaiypat_xyz oaiyses_ oaiy"),
            "oaiypat_xyz oaiyses_ oaiy"
        );
    }

    #[test]
    fn rule_1_a_token_with_a_percent_encoded_or_json_escaped_underscore_is_hidden() {
        let secret = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
        for line in [
            format!("GET /x?y=oaiypat%5F0123456789abcdef%5F{secret}"),
            format!("GET /x?y=oaiypat%5f0123456789abcdef%5f{secret}"),
            format!("{{\"a\":\"oaiypat\\u005f0123456789abcdef\\u005f{secret}\"}}"),
            format!("oaiyses\\u005F0123456789abcdef_{secret}"),
            format!("oaiypat%5F0123456789abcdef_{secret}"),
        ] {
            let out = scrubbed(&line);
            assert!(!out.contains(secret), "{line} -> {out}");
            assert!(out.contains(REDACTED), "{line} -> {out}");
        }
        assert_eq!(
            scrubbed(&format!("a oaiypat%5F0123456789abcdef%5F{secret}%20b")),
            "a [redacted]%20b"
        );
        // The start of a legacy token, encoded.
        assert_eq!(
            scrubbed("x oaiypat%5F0123456789abcdef0123 y"),
            "x [redacted] y"
        );
        // What is not a token stays.
        assert_eq!(scrubbed("oaiypat%5Fxyz"), "oaiypat%5Fxyz");
        assert_eq!(scrubbed("oaiypat%5"), "oaiypat%5");
        assert_eq!(scrubbed("oaiypat\\u005"), "oaiypat\\u005");
    }

    #[test]
    fn rule_2_a_bearer_and_the_value_of_a_credential_header_are_hidden() {
        assert_eq!(
            scrubbed("got Bearer abcdefghijkl end"),
            "got Bearer [redacted] end"
        );
        assert_eq!(
            scrubbed("Authorization: Bearer abcdefghijkl"),
            "Authorization: [redacted]"
        );
        assert_eq!(scrubbed("Cookie: a=b; c=d"), "Cookie: [redacted]");
        assert_eq!(
            scrubbed("Set-Cookie: sid=abc; Path=/; HttpOnly"),
            "Set-Cookie: [redacted]"
        );
        assert_eq!(
            scrubbed("authorization:Basic dXNlcjpwYXNz"),
            "authorization:[redacted]"
        );
    }

    #[test]
    fn rule_2_a_bearer_followed_by_a_short_word_is_hidden_and_prose_is_not() {
        assert_eq!(scrubbed("Bearer abc12"), "Bearer [redacted]");
        assert_eq!(scrubbed("Bearer abcd."), "Bearer [redacted]");
        assert_eq!(scrubbed("the bearer of bad news"), "the bearer of bad news");
        assert_eq!(scrubbed("Bearer abc"), "Bearer abc");
    }

    #[test]
    fn rule_3_a_setup_code_or_invite_is_hidden() {
        assert_eq!(scrubbed("code M6SC-7N75-YR3H."), "code [redacted].");
        assert_eq!(scrubbed("(K7QX-4M2P)"), "([redacted])");
        assert_eq!(
            scrubbed("K7QX-4M2P-XXXX-YYYY stays"),
            "K7QX-4M2P-XXXX-YYYY stays",
            "four groups is not the shape"
        );
        assert_eq!(scrubbed("K7QX stays"), "K7QX stays");
    }

    #[test]
    fn rule_3_a_lower_case_code_is_hidden_when_it_has_a_digit_and_a_pair_of_words_is_not() {
        assert_eq!(scrubbed("setup code k7qx-4m2p"), "setup code [redacted]");
        assert_eq!(scrubbed("invite m6sc-7n75-yr3h."), "invite [redacted].");
        assert_eq!(scrubbed("mixed K7qx-4M2p"), "mixed [redacted]");
        assert_eq!(
            scrubbed("sync-data and send-data"),
            "sync-data and send-data"
        );
        // Upper case needs no digit, as before.
        assert_eq!(scrubbed("DASH-BACK"), "[redacted]");
    }

    #[test]
    fn rule_3_a_code_typed_with_another_dash_is_hidden() {
        for dash in [
            '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}',
        ] {
            assert_eq!(
                scrubbed(&format!("setup code K7QX{dash}4M2P ok")),
                "setup code [redacted] ok",
                "U+{:04X}",
                dash as u32
            );
        }
    }

    #[test]
    fn rule_4_the_value_of_a_pair_named_for_a_secret_is_hidden() {
        assert_eq!(
            scrubbed("api_key=hunter2 host=x"),
            "api_key=[redacted] host=x"
        );
        assert_eq!(scrubbed("token: abc123 more"), "token: [redacted] more");
        assert_eq!(
            scrubbed(r#"{"cookie":"a=b","n":1}"#),
            r#"{"cookie":"[redacted]","n":1}"#
        );
        assert_eq!(scrubbed("csrf=Zm9v"), "csrf=[redacted]");
    }

    #[test]
    fn rule_4_a_password_a_passphrase_and_a_secret_run_to_the_end_of_the_line() {
        assert_eq!(scrubbed("password=hunter2 host=x"), "password=[redacted]");
        // What is already hidden is not hidden again, and what follows it is not taken for its value.
        assert_eq!(
            scrubbed("password=[redacted] host=x"),
            "password=[redacted] host=x"
        );
        assert_eq!(scrubbed("secret: abc123 more"), "secret: [redacted]");
        assert_eq!(scrubbed("phrase=one two"), "phrase=[redacted]");
        assert_eq!(
            scrubbed("Authorization=Basic dXNlcjpw more"),
            "Authorization=[redacted]"
        );
        assert_eq!(
            scrubbed("password: correct horse battery staple"),
            "password: [redacted]"
        );
        // An unquoted value, whatever punctuation it holds.
        for text in [
            "Sup3r;Secret!",
            "Sup3r,Secret!",
            "Sup3r&Secret!",
            "Sup3r)Secret!",
            "Sup3r]Secret!",
            "Sup3r}Secret!",
            "Sup3r\"Secret!",
            "Sup3r'Secret!",
            "Sup3r Secret!",
        ] {
            assert_eq!(
                scrubbed(&format!("password={text}")),
                "password=[redacted]",
                "{text}"
            );
        }
    }

    #[test]
    fn rule_4_any_other_key_runs_to_the_whitespace_and_takes_what_is_in_it() {
        for text in ["a;b", "a,b", "a&b", "a)b", "a]b", "a}b", "a\"b", "a'b"] {
            assert_eq!(
                scrubbed(&format!("token={text} next")),
                "token=[redacted] next",
                "{text}"
            );
        }
        assert_eq!(
            scrubbed("GET /x?token=abc123&y=1"),
            "GET /x?token=[redacted]"
        );
    }

    #[test]
    fn rule_4_a_quoted_value_runs_to_its_closing_quote_whatever_it_holds() {
        assert_eq!(
            scrubbed(r#"password="correct horse; battery, staple" next=1"#),
            r#"password="[redacted]" next=1"#
        );
        assert_eq!(
            scrubbed("password='p;w x' next=1"),
            "password='[redacted]' next=1"
        );
        // An escaped quote does not close it.
        assert_eq!(
            scrubbed(r#"password="ab\"cd" next=1"#),
            r#"password="[redacted]" next=1"#
        );
        assert_eq!(
            scrubbed(r#"{"password":"ab\"cd","user":"a"}"#),
            r#"{"password":"[redacted]","user":"a"}"#
        );
        // A backslash before the closing quote, itself escaped, does.
        assert_eq!(
            scrubbed(r#"password="ab\\" next=1"#),
            r#"password="[redacted]" next=1"#
        );
    }

    #[test]
    fn rule_4_json_inside_a_json_string_is_a_pair_too() {
        assert_eq!(
            scrubbed(r#"{"body":"{\"password\":\"hunter2\",\"a\":1}"}"#),
            r#"{"body":"{\"password\":\"[redacted]\",\"a\":1}"}"#
        );
        assert_eq!(
            scrubbed(r#"{"body":"{\"token\":\"abcdef123456\"}"}"#),
            r#"{"body":"{\"token\":\"[redacted]\"}"}"#
        );
        // Two levels down.
        assert_eq!(
            scrubbed(r#"{"a":"{\"b\":\"{\\\"password\\\":\\\"hunter2\\\"}\"}"}"#),
            r#"{"a":"{\"b\":\"{\\\"password\\\":\\\"[redacted]\\\"}\"}"}"#
        );
        // A quote in the password, one level down, is three backslashes and a quote: not the end of the value.
        assert_eq!(
            scrubbed(r#"{"body":"{\"password\":\"hunter\\\"2\"}"}"#),
            r#"{"body":"{\"password\":\"[redacted]\"}"}"#
        );
        // And the same text with the quotes escaped on their own.
        assert_eq!(
            scrubbed(r#"log: \"password\": \"hunter2\""#),
            r#"log: \"password\": \"[redacted]\""#
        );
    }

    #[test]
    fn rule_4_a_json_scalar_ends_at_its_comma_and_the_rest_of_the_object_is_kept() {
        assert_eq!(
            scrubbed(r#"{"password":12345,"user":"a"}"#),
            r#"{"password":[redacted],"user":"a"}"#
        );
        assert_eq!(
            scrubbed(r#"{"user":"a","token":null}"#),
            r#"{"user":"a","token":[redacted]}"#
        );
    }

    #[test]
    fn rule_4_the_other_separators_and_the_long_flag() {
        assert_eq!(scrubbed("password => hunter2"), "password => [redacted]");
        assert_eq!(scrubbed("password := hunter2"), "password := [redacted]");
        assert_eq!(
            scrubbed("password\u{ff1a}hunter2"),
            "password\u{ff1a}[redacted]"
        );
        assert_eq!(scrubbed("password : hunter2"), "password : [redacted]");
        assert_eq!(
            scrubbed("spawn: tool --password hunter2 --x"),
            "spawn: tool --password [redacted] --x"
        );
        assert_eq!(
            scrubbed("tool --token=abc123 --x"),
            "tool --token=[redacted] --x"
        );
        assert_eq!(
            scrubbed("tool --api-key \"my key\" --x"),
            "tool --api-key \"[redacted]\" --x"
        );
        // A flag that is followed by another flag, and a short one, hide nothing.
        assert_eq!(scrubbed("tool --password --x"), "tool --password --x");
        assert_eq!(
            scrubbed("mysql -u root -p hunter2"),
            "mysql -u root -p hunter2"
        );
    }

    #[test]
    fn rule_5_a_word_that_starts_like_a_secret_is_hidden_wherever_it_is() {
        assert_eq!(
            scrubbed("saw sk-abcdefghijklmnop in env"),
            "saw [redacted] in env"
        );
        assert_eq!(
            scrubbed("ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
            "[redacted]"
        );
        assert_eq!(scrubbed("(hf_abcdefghijklmnop)"), "([redacted])");
        assert_eq!(
            scrubbed("sk-short"),
            "sk-short",
            "a prefix alone is not a secret"
        );
    }

    #[test]
    fn rule_5_a_stripe_key_an_aws_key_id_and_a_json_web_token_are_hidden() {
        let stripe = ["sk", "_live_", "abcdefghijklmnopqrstuvwx"].concat();
        let test = ["rk", "_test_", "abcdefghijklmnopqrstuvwx"].concat();
        let aws = ["AK", "IA", "IOSFODNN7EXAMPLE"].concat();
        let jwt = [
            "eyJhbGciOiJIUzI1NiJ9",
            ".",
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            ".",
            "c2lnbmF0dXJlLWJ5dGVz",
        ]
        .concat();
        assert_eq!(
            scrubbed(&format!("stripe {stripe} ok")),
            "stripe [redacted] ok"
        );
        assert_eq!(scrubbed(&format!("stripe {test}")), "stripe [redacted]");
        assert_eq!(scrubbed(&format!("aws {aws}.")), "aws [redacted]");
        assert_eq!(
            scrubbed(&format!("user sent {jwt}.")),
            "user sent [redacted]"
        );
        // What is not one stays.
        assert_eq!(scrubbed("sk_live_short"), "sk_live_short");
        assert_eq!(scrubbed("AKIAshort"), "AKIAshort");
        assert_eq!(scrubbed("eyJabc.def"), "eyJabc.def");
        assert_eq!(scrubbed("version 1.2.3"), "version 1.2.3");
    }

    #[test]
    fn rule_6_the_password_of_a_url_is_hidden_and_the_rest_of_it_is_not() {
        assert_eq!(
            scrubbed("connecting to https://admin:Sup3rS3cret@db.example.com/x"),
            "connecting to https://admin:[redacted]@db.example.com/x"
        );
        assert_eq!(
            scrubbed("DATABASE_URL: postgres://app:hunter22@localhost:5432/app"),
            "DATABASE_URL: postgres://app:[redacted]@localhost:5432/app"
        );
        // A password with an `@` in it: the last one ends the userinfo.
        assert_eq!(scrubbed("x://u:p@ss@host/"), "x://u:[redacted]@host/");
        // A userinfo with no password: hidden only when it is long enough to be a token.
        assert_eq!(
            scrubbed("git clone ssh://git@github.com/x/y"),
            "git clone ssh://git@github.com/x/y"
        );
        assert_eq!(
            scrubbed("https://abcdefghijklmnopqrstuvwxyz@host/"),
            "https://[redacted]@host/"
        );
        // Not userinfo: an `@` after the path, a port, an IPv6 host.
        for line in [
            "https://example.com/a@b.c",
            "http://localhost:8080/api",
            "http://[::1]:8080/x?email=a@b.c",
            "see https://example.com/x?y=z#a@b",
        ] {
            assert_eq!(scrubbed(line), line);
        }
    }

    #[test]
    fn ordinary_log_lines_pass_through_unchanged() {
        for line in [
            "",
            "service comfyui started on port 8188",
            "[INFO] OAIY API listening on http://127.0.0.1:17972",
            "downloading model.gguf 45% (1.2 GB / 2.6 GB)",
            "GET /api/health 200 in 3ms",
            "plugin aokie.phone: call 8f3a answered by the agent",
            "error: could not open file C:\\Users\\someone\\data\\x.json",
            "the bearer of bad news",
            "oaiypat_",
            "oaiypat_x",
            "oaiy is the product name, oaiyses is not a token",
            "UUID 3F2A9B1C-1234-5678-9ABC-DEF012345678 finished",
            "keyboard layout us",
            "lookup done in 12ms",
            "ID 0491 570 006",
            "PASS-FAIL and TEST OK",
            "héllo wörld ✓ 日本語",
            "sync-data finished; send-data next",
            "fetching https://example.com:8443/a/b?c=d",
            "version 1.2.3-beta and tool --verbose --port 8080",
            "date 2026-09-30 time 12:30",
        ] {
            assert_eq!(scrubbed(line), line, "{line:?} must not change");
        }
    }

    #[test]
    fn the_scrub_hides_the_vectors_of_the_design() {
        // The token and the setup code of Appendix A never survive a log line, wherever they sit.
        for secret in [TOKEN, "M6SC-7N75-YR3H"] {
            let line = format!("value={secret} and again {secret}");
            let out = scrubbed(&line);
            assert!(!out.contains(secret), "{out}");
        }
        // The session-link code (43 base64url characters, printed on the console and carried in a
        // URL fragment) has no marker and is never logged, so the scrub does not look for one; a
        // pair that names it as a secret is still hidden.
        let out = scrubbed("token: __79_Pv6-fj39vX08_Lx8O_u7ezr6uno5-bl5OPi4eA");
        assert!(!out.contains("__79_Pv6"), "{out}");
    }

    #[test]
    fn scrubbing_twice_is_scrubbing_once() {
        for line in [
            format!("Authorization: Bearer {TOKEN}"),
            format!("password=hunter2 {TOKEN} M6SC-7N75-YR3H"),
            r#"{"csrf":"abc","token":"def"}"#.to_string(),
            "already [redacted] here".to_string(),
            "key=[redacted]".to_string(),
            "Cookie: [redacted]".to_string(),
            "password=[redacted]".to_string(),
            r#"{"body":"{\"password\":\"hunter2\"}"}"#.to_string(),
            "https://u:pw@host tool --token abc123 sk-abcdefghijklmnop".to_string(),
            "setup code k7qx-4m2p and oaiypat%5F0123456789abcdef%5FAAECAwQFBgcICQoL".to_string(),
        ] {
            let once = scrubbed(&line);
            assert_eq!(scrubbed(&once), once, "{line:?} -> {once:?}");
        }
    }

    #[test]
    fn a_secret_in_the_middle_of_a_multibyte_line_is_found_and_the_rest_is_kept() {
        let line = format!("héllo ✓ api_key=hunter2 ✓ {TOKEN} wörld");
        assert_eq!(
            scrubbed(&line),
            "héllo ✓ api_key=[redacted] ✓ [redacted] wörld"
        );
        // A password takes the rest of the line, multibyte or not.
        assert_eq!(
            scrubbed("héllo ✓ password=hünter2 ✓ wörld"),
            "héllo ✓ password=[redacted]"
        );
        assert_eq!(
            scrubbed("héllo password=\"hünter2 ✓\" wörld ✓"),
            "héllo password=\"[redacted]\" wörld ✓"
        );
    }

    #[test]
    fn a_pair_whose_value_never_closes_hides_the_rest_of_the_line() {
        assert_eq!(
            scrubbed(r#"{"password":"never closed and more"#),
            r#"{"password":"[redacted]"#
        );
        assert_eq!(scrubbed("password="), "password=");
        assert_eq!(scrubbed("password"), "password");
        assert_eq!(scrubbed("token:"), "token:");
        assert_eq!(scrubbed("password=\"abc def ghi"), "password=\"[redacted]");
        assert_eq!(scrubbed("token={a:{b:c}"), "token=[redacted]");
    }

    #[test]
    fn a_key_that_merely_contains_a_secret_word_is_treated_as_one() {
        // Over-redaction is the intended direction.
        assert_eq!(scrubbed("monkey=1"), "monkey=[redacted]");
        assert_eq!(scrubbed("X-Api-Key: abc"), "X-Api-Key: [redacted]");
        assert_eq!(scrubbed("TOKEN_LIMIT=4096"), "TOKEN_LIMIT=[redacted]");
    }

    #[test]
    fn the_scrub_never_panics_on_odd_input() {
        for line in [
            "\u{0}",
            "\"",
            "'",
            "{",
            "}",
            "[",
            "=",
            ":",
            "Authorization",
            "Authorization:",
            "Bearer",
            "Bearer ",
            "oaiy",
            "oaiypat",
            "oaiypat_0123456789abcdef_",
            "oaiy\u{e9}",
            "password:\"",
            "password:{",
            "password:[[[",
            "\u{1F600}password=\u{1F600}",
            "-----",
            "ABCD-",
            "-ABCD",
            "ABCD--EFGH",
            "://",
            "x://@",
            "x://:@",
            "x://a:b@",
            "--",
            "--password",
            "--password ",
            "password\u{ff}",
            "password\u{ef}\u{bc}",
            "password\u{ff1a}",
            "\\",
            "\\\\\\",
            "password=\\",
            "password=\\\"",
            "\\\"password\\\":\\\"",
            "eyJ..",
            "oaiypat%",
            "oaiypat%5",
            "oaiypat\\u005",
            "oaiypat\\u005f",
            "oaiypat_%5F",
            "\u{2013}",
            "ABCD\u{2013}",
        ] {
            let _ = scrubbed(line);
        }
    }

    /// Lines built so that each scan of the scrub has something to start on at every few bytes.
    fn adversarial(n: usize) -> Vec<(&'static str, String)> {
        vec![
            ("oaiypat_", "oaiypat_".repeat(n)),
            ("oaiy", "oaiy".repeat(n)),
            ("oaiypat%5F", "oaiypat%5F".repeat(n)),
            ("oaiypat\\u005f", "oaiypat\\u005f".repeat(n)),
            ("key:{", "key:{".repeat(n)),
            ("password=", "password=".repeat(n)),
            ("password=\"", "password=\"".repeat(n)),
            ("\\\"password\\\":\\\"", "\\\"password\\\":\\\"".repeat(n)),
            ("password=\\", "password=\\".repeat(n)),
            ("Bearer ", "Bearer ".repeat(n)),
            ("Authorization:", "Authorization: ".repeat(n)),
            ("AB12-", "AB12-".repeat(n)),
            ("k7qx-", "k7qx-".repeat(n)),
            ("://", "://".repeat(n)),
            ("a:b@://", "http://a:b@".repeat(n)),
            ("--password ", "--password ".repeat(n)),
            ("token=x ", "token=x ".repeat(n)),
            ("eyJ.", "eyJhbGciOiJ.".repeat(n)),
            ("AKIA", "AKIA".repeat(n)),
            ("sk_live_", "sk_live_".repeat(n)),
            ("a-b-c-", "a-b-c-".repeat(n)),
            ("key: ", "key: ".repeat(n)),
            ("A", "A".repeat(n * 8)),
            ("spaces", format!("bearer{}", " ".repeat(n * 8))),
        ]
    }

    #[test]
    fn a_long_line_of_near_misses_costs_little_and_is_cut_at_the_limit() {
        // Every scan is bounded, so a line built to make each one start over does not take a scan of
        // the rest of the line per occurrence (which is minutes for a line like these).
        for (name, long) in adversarial(20_000) {
            let started = std::time::Instant::now();
            let out = scrubbed(&long);
            // Cut at the limit; a replacement can be longer than the word it hides (`key=1` is 5 bytes,
            // `key=[redacted]` 14).
            assert!(out.len() <= 4 * MAX_LINE, "{name}: {} bytes", out.len());
            // A linear scan of 32 KiB is a few milliseconds even in a debug build; a scan of the rest of
            // the line for each start is many seconds.
            eprintln!(
                "scrub of a 32 KiB line of {name:?}: {:?}",
                started.elapsed()
            );
            assert!(
                started.elapsed() < std::time::Duration::from_millis(600),
                "{name}: took {:?}",
                started.elapsed()
            );
        }
        // A line over the limit is cut on a character boundary, and says so.
        let mut line = "é".repeat(MAX_LINE);
        line.push_str(" password=hunter2");
        let out = scrubbed(&line);
        assert!(out.ends_with("... [line cut]") && !out.contains("hunter2"));
        // A line at the limit is not cut.
        assert!(!scrubbed(&"x".repeat(MAX_LINE)).contains("[line cut]"));
    }

    #[test]
    fn every_stage_is_linear_in_the_line_and_not_only_the_cut_one() {
        // The cut is a limit and not the reason it is fast: without it, a line twice as long takes about
        // twice as long. (The bound is generous: a scan that starts over at every occurrence, on lines this
        // long, takes seconds to minutes in a debug build.)
        for (name, long) in adversarial(16_000) {
            let started = std::time::Instant::now();
            let _ = scrub_whole(&long);
            eprintln!(
                "scrub of {} bytes of {name:?}, uncut: {:?}",
                long.len(),
                started.elapsed()
            );
            assert!(
                started.elapsed() < std::time::Duration::from_millis(1500),
                "{name}: {} bytes took {:?}",
                long.len(),
                started.elapsed()
            );
        }
    }
}
