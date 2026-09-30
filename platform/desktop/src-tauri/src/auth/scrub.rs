//! Scrubbed logs (design 4.12): every line any log route returns, and every line the `logs_tail`
//! control tool hands the model, passes through [`scrub_line`], because service, plugin, node, python
//! and engine logs carry captured child output.
//!
//! It replaces with `[redacted]`:
//!
//! 1. the token grammar (`oaiy(pat|ses|dsk|run|con|dev)_<id>_<secret>`, and any longer-than-a-prefix
//!    start of one, in case a line was cut) and the legacy `oaiypat_<64 hex>`;
//! 2. `Bearer <token>`, and the value after `Authorization:`, `Cookie:` and `Set-Cookie:` (to the end
//!    of the line);
//! 3. the shape of a setup code or an invite (`K7QX-4M2P`, `M6SC-7N75-YR3H`);
//! 4. the value of a `key=value`, `key: value` or JSON `"key":"value"` pair whose key contains `key`,
//!    `token`, `secret`, `password`, `passphrase`, `phrase`, `mnemonic`, `authorization`, `cookie` or
//!    `csrf`;
//! 5. a word that starts like a secret value (`sk-`, `ghp_`, `hf_`, and the token prefixes: the list
//!    of `control::audit::secret_value`).
//!
//! It over-redacts on purpose: a log line that lost a harmless word is a smaller loss than one that
//! kept a credential. Nothing here is a regular expression: the scanners are small and fail toward
//! redaction.

use crate::control::audit::{secret_value, REDACTED};

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
const HEADERS: [&str; 3] = ["authorization:", "set-cookie:", "cookie:"];
const KINDS: [&str; 6] = ["pat", "ses", "dsk", "run", "con", "dev"];
/// The alphabet of a setup code or an invite (Crockford base32).
const CROCKFORD: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
/// The most of a token-looking run that is looked at (a token is 68 bytes, a legacy one 72).
const SCAN: usize = 96;

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
    let s = header_values(line);
    let s = tokens(&s);
    let s = bearers(&s);
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

/// If `s` starts with a token (or a start of one worth hiding), how long it is.
fn token_end(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let kind = KINDS.iter().find(|k| s[4..].starts_with(*k))?;
    let after_kind = 4 + kind.len();
    if b.get(after_kind) != Some(&b'_') {
        return None;
    }
    let body = after_kind + 1;
    // Every run is bounded, so that a line made of near-misses costs a little for each and not a scan
    // of the rest of the line for each.
    let run = b[body..]
        .iter()
        .take(SCAN)
        .take_while(|c| is_token_char(**c))
        .count();
    // The grammar: 16 hex, `_`, 43 base64url. A cut one: the same start and at least 8 secret characters.
    let id_len = b[body..]
        .iter()
        .take(SCAN)
        .take_while(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
        .count();
    if id_len >= 16 && b.get(body + 16) == Some(&b'_') {
        let secret = b[body + 17..]
            .iter()
            .take(SCAN)
            .take_while(|c| is_token_char(**c))
            .count();
        if secret >= 8 {
            return Some(body + 17 + secret);
        }
    }
    // A legacy pairing token (`oaiypat_` and 64 hex), or the start of one.
    if *kind == "pat" && id_len >= 8 && id_len == run {
        return Some(body + id_len);
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
                if word >= 6 && &s[j..j + word] != REDACTED {
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

fn key_word(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    KEY_WORDS.iter().any(|k| lower.contains(k))
}

/// Rule 4: `key=value`, `key: value` and `"key":"value"` where the key names a secret.
fn pairs(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        // A key is a run of name characters.
        if b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'-' || b[i] == b'.' {
            let start = i;
            while i < b.len()
                && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'-' || b[i] == b'.')
            {
                i += 1;
            }
            out.push_str(&s[start..i]);
            if !key_word(&s[start..i]) {
                continue;
            }
            // An optional closing quote of the key, then a separator.
            let mut j = i;
            if matches!(b.get(j), Some(b'"') | Some(b'\'')) {
                j += 1;
            }
            let mut k = j + b[j.min(b.len())..]
                .iter()
                .take_while(|c| **c == b' ' || **c == b'\t')
                .count();
            if !matches!(b.get(k), Some(b'=') | Some(b':')) {
                continue;
            }
            k += 1;
            k += b[k.min(b.len())..]
                .iter()
                .take_while(|c| **c == b' ' || **c == b'\t')
                .count();
            if k >= b.len() {
                continue;
            }
            let value_end = value_end(s, k);
            if value_end == k {
                continue;
            }
            let value = &s[k..value_end];
            if value == REDACTED
                || value == format!("\"{REDACTED}\"")
                || value == format!("'{REDACTED}'")
            {
                continue;
            }
            // Keep the key, the separator and any quotes; hide what is between them.
            out.push_str(&s[i..k]);
            match b[k] {
                q @ (b'"' | b'\'') => {
                    out.push(q as char);
                    out.push_str(REDACTED);
                    // A closing quote is part of the value's extent when there is one.
                    if value.len() >= 2 && value.as_bytes()[value.len() - 1] == q {
                        out.push(q as char);
                    }
                }
                _ => out.push_str(REDACTED),
            }
            i = value_end;
            continue;
        }
        let ch = s[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Where the value that starts at `k` ends: a quoted string (with its escapes), a balanced `{…}` or
/// `[…]`, or a run up to whitespace or a delimiter.
fn value_end(s: &str, k: usize) -> usize {
    let b = s.as_bytes();
    match b[k] {
        q @ (b'"' | b'\'') => {
            let mut j = k + 1;
            while j < b.len() {
                if b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == q {
                    return j + 1;
                }
                j += 1;
            }
            // Never closed: everything to the end of the line is the value.
            b.len()
        }
        open @ (b'{' | b'[') => {
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0i32;
            let mut j = k;
            while j < b.len() {
                if b[j] == open {
                    depth += 1;
                } else if b[j] == close {
                    depth -= 1;
                    if depth == 0 {
                        return j + 1;
                    }
                }
                j += 1;
            }
            b.len()
        }
        _ => {
            let mut j = k;
            while j < b.len()
                && !b[j].is_ascii_whitespace()
                && !matches!(b[j], b'&' | b',' | b';' | b')' | b'}' | b']' | b'"' | b'\'')
            {
                j += 1;
            }
            j
        }
    }
}

/// Rule 5: a word that starts like a secret value, wherever it is.
fn secret_words(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if !word.is_empty() {
            if secret_value(word) {
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
        if ch.is_ascii_alphanumeric() || ch == '-' {
            word.push(ch);
        } else {
            flush(&mut word, &mut out);
            out.push(ch);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// `[0-9A-HJKMNP-TV-Z]{4}(-[0-9A-HJKMNP-TV-Z]{4}){1,2}`, the whole word.
fn is_code(word: &str) -> bool {
    let groups: Vec<&str> = word.split('-').collect();
    (2..=3).contains(&groups.len())
        && groups
            .iter()
            .all(|g| g.len() == 4 && g.bytes().all(|b| CROCKFORD.contains(&b)))
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
    // of its own (the corpus below holds them together).

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
        // What is not the start of a token stays.
        assert_eq!(
            scrubbed("oaiypat_xyz oaiyses_ oaiy"),
            "oaiypat_xyz oaiyses_ oaiy"
        );
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
    fn rule_4_the_value_of_a_pair_named_for_a_secret_is_hidden() {
        assert_eq!(
            scrubbed("password=hunter2 host=x"),
            "password=[redacted] host=x"
        );
        assert_eq!(scrubbed("secret: abc123 more"), "secret: [redacted] more");
        assert_eq!(
            scrubbed(r#"{"cookie":"a=b","n":1}"#),
            r#"{"cookie":"[redacted]","n":1}"#
        );
        assert_eq!(scrubbed("csrf=Zm9v"), "csrf=[redacted]");
        assert_eq!(scrubbed("phrase=one two"), "phrase=[redacted] two");
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
    fn a_corpus_of_lines_that_carry_a_secret() {
        for (line, want) in [
            // The token grammar, every kind, wherever it sits.
            (
                format!("paired app used {TOKEN} from 203.0.113.9"),
                "paired app used [redacted] from 203.0.113.9".to_string(),
            ),
            (TOKEN.to_string(), "[redacted]".to_string()),
            (format!("token={TOKEN}"), "token=[redacted]".to_string()),
            (format!("\"{TOKEN}\""), "\"[redacted]\"".to_string()),
            (format!("x{TOKEN}y"), "x[redacted]".to_string()),
            (
                format!("legacy {LEGACY} imported"),
                "legacy [redacted] imported".to_string(),
            ),
            (
                "oaiyses_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8".into(),
                "[redacted]".to_string(),
            ),
            (
                "got oaiydsk_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8 ok"
                    .into(),
                "got [redacted] ok".to_string(),
            ),
            (
                "oaiyrun_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8".into(),
                "[redacted]".to_string(),
            ),
            (
                "oaiycon_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8".into(),
                "[redacted]".to_string(),
            ),
            (
                "oaiydev_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8".into(),
                "[redacted]".to_string(),
            ),
            // A line cut short.
            (
                "cut oaiypat_0123456789abcdef_AAECAwQF".into(),
                "cut [redacted]".to_string(),
            ),
            // Bearer and headers.
            (
                "Authorization: Bearer abcdefghijkl".into(),
                "Authorization: [redacted]".to_string(),
            ),
            (
                "authorization:   Basic dXNlcjpwYXNz".into(),
                "authorization:   [redacted]".to_string(),
            ),
            (
                "> Cookie: __Host-oaiy_dash=abc; other=1".into(),
                "> Cookie: [redacted]".to_string(),
            ),
            (
                "Set-Cookie: oaiy_dash_17972=abcdef; HttpOnly".into(),
                "Set-Cookie: [redacted]".to_string(),
            ),
            (
                "curl -H 'X: y' with Bearer sk-abcdefghijklmnop end".into(),
                "curl -H 'X: y' with Bearer [redacted] end".to_string(),
            ),
            (
                "BEARER abcdefghijkl".into(),
                "BEARER [redacted]".to_string(),
            ),
            // Codes.
            (
                "setup code is M6SC-7N75-YR3H, use it".into(),
                "setup code is [redacted], use it".to_string(),
            ),
            ("invite K7QX-4M2P".into(), "invite [redacted]".to_string()),
            // key=value, key: value, JSON.
            (
                "connecting password=hunter2 host=x".into(),
                "connecting password=[redacted] host=x".to_string(),
            ),
            (
                "api_key: 12345 next".into(),
                "api_key: [redacted] next".to_string(),
            ),
            (
                "GET /x?token=abc123&y=1".into(),
                "GET /x?token=[redacted]&y=1".to_string(),
            ),
            (
                r#"{"user":"a","password":"hunter2","n":1}"#.into(),
                r#"{"user":"a","password":"[redacted]","n":1}"#.to_string(),
            ),
            (
                r#"{"csrf": "abcdefgh"}"#.into(),
                r#"{"csrf": "[redacted]"}"#.to_string(),
            ),
            (
                r#"{"passphrase":"correct horse battery staple"}"#.into(),
                r#"{"passphrase":"[redacted]"}"#.to_string(),
            ),
            (
                r#"{"mnemonic":"abandon ability able","x":1}"#.into(),
                r#"{"mnemonic":"[redacted]","x":1}"#.to_string(),
            ),
            (
                r#"{"secret":{"a":1,"b":[2,3]},"ok":true}"#.into(),
                r#"{"secret":[redacted],"ok":true}"#.to_string(),
            ),
            (
                r#"{"cookie":"a=b; c=d"}"#.into(),
                r#"{"cookie":"[redacted]"}"#.to_string(),
            ),
            (
                "client_secret=abc phrase=fox jumps".into(),
                "client_secret=[redacted] phrase=[redacted] jumps".to_string(),
            ),
            (
                "Password: hunter2".into(),
                "Password: [redacted]".to_string(),
            ),
            (
                "HF_TOKEN=hf_abcdefghijklmnop".into(),
                "HF_TOKEN=[redacted]".to_string(),
            ),
            // Value prefixes, anywhere.
            (
                "saw sk-abcdefghijklmnop in env".into(),
                "saw [redacted] in env".to_string(),
            ),
            (
                "gh ghp_abcdefghijklmnopqrstuvwxyz0123456789 ok".into(),
                "gh [redacted] ok".to_string(),
            ),
            (
                "file github_pat_11AAAAAAAAAAAAAAAAAAAAAA end".into(),
                "file [redacted] end".to_string(),
            ),
            ("hf_abcdefghijklmnop".into(), "[redacted]".to_string()),
            (
                "xoxb-1234567890-abcdefghij".into(),
                "[redacted]".to_string(),
            ),
        ] {
            assert_eq!(scrubbed(&line), want, "for {line:?}");
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
            "the code a1b2-c3d4 is lower case",
            "UUID 3F2A9B1C-1234-5678-9ABC-DEF012345678 finished",
            "keyboard layout us",
            "lookup done in 12ms",
            "ID 0491 570 006",
            "PASS-FAIL and TEST OK",
            "héllo wörld ✓ 日本語",
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
        ] {
            let once = scrubbed(&line);
            assert_eq!(scrubbed(&once), once, "{line:?} -> {once:?}");
        }
    }

    #[test]
    fn a_secret_in_the_middle_of_a_multibyte_line_is_found_and_the_rest_is_kept() {
        let line = format!("héllo ✓ password=hunter2 ✓ {TOKEN} wörld");
        assert_eq!(
            scrubbed(&line),
            "héllo ✓ password=[redacted] ✓ [redacted] wörld"
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
        ] {
            let _ = scrubbed(line);
        }
    }

    #[test]
    fn a_long_line_of_near_misses_costs_little_and_is_cut_at_the_limit() {
        // Every scan is bounded, so a line built to make each one start over does not take a scan of
        // the rest of the line per occurrence (which is minutes for a line like these).
        let started = std::time::Instant::now();
        for long in [
            "oaiypat_".repeat(20_000),
            "key:{".repeat(20_000),
            "password=\"".repeat(20_000),
            "Bearer ".repeat(20_000),
            "AB12-".repeat(40_000),
            "A".repeat(200_000),
        ] {
            let out = scrubbed(&long);
            // Cut at the limit; a replacement can be longer than the word it hides.
            assert!(out.len() <= 2 * MAX_LINE, "{} bytes", out.len());
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "took {:?}",
            started.elapsed()
        );
        // A line over the limit is cut on a character boundary, and says so.
        let mut line = "é".repeat(MAX_LINE);
        line.push_str(" password=hunter2");
        let out = scrubbed(&line);
        assert!(out.ends_with("... [line cut]") && !out.contains("hunter2"));
        // A line at the limit is not cut.
        assert!(!scrubbed(&"x".repeat(MAX_LINE)).contains("[line cut]"));
    }
}
