//! The corpora of `scrub_line`: lines that carry a secret, with what the scrub must make of each, and the
//! reviewer's 81 lines (a security review of the first version of the scrub found 24 of them leaking), each
//! with the substrings that must not survive.
//!
//! Fixtures that look like a provider's key are put together from pieces, so that no secret scanner takes
//! this file for a leak: none of them is a real credential.

use super::*;

const TOKEN: &str = "oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
const LEGACY: &str = "oaiypat_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
/// The 43-character secret of the token above.
const S: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
/// The first 32 characters of a legacy token's hex.
const LEGACY_HEX: &str = "0123456789abcdef0123456789abcdef";

fn tok(kind: &str) -> String {
    format!("oaiy{kind}_0123456789abcdef_{S}")
}

#[test]
fn a_corpus_of_lines_that_carry_a_secret() {
    let ghp = ["gh", "p_", "abcdefghijklmnopqrstuvwxyz0123456789"].concat();
    let xoxb = ["xo", "xb-", "1234567890-abcdefghij"].concat();
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
        (tok("ses"), "[redacted]".to_string()),
        (
            format!("got {} ok", tok("dsk")),
            "got [redacted] ok".to_string(),
        ),
        (tok("run"), "[redacted]".to_string()),
        (tok("con"), "[redacted]".to_string()),
        (tok("dev"), "[redacted]".to_string()),
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
        (
            "the code a1b2-c3d4 is lower case".into(),
            "the code [redacted] is lower case".to_string(),
        ),
        // key=value, key: value, JSON.
        (
            "connecting password=hunter2 host=x".into(),
            "connecting password=[redacted]".to_string(),
        ),
        (
            "api_key: 12345 next".into(),
            "api_key: [redacted] next".to_string(),
        ),
        (
            "GET /x?token=abc123&y=1".into(),
            "GET /x?token=[redacted]".to_string(),
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
            "client_secret=[redacted]".to_string(),
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
        (format!("gh {ghp} ok"), "gh [redacted] ok".to_string()),
        (
            "file github_pat_11AAAAAAAAAAAAAAAAAAAAAA end".into(),
            "file [redacted] end".to_string(),
        ),
        ("hf_abcdefghijklmnop".into(), "[redacted]".to_string()),
        (xoxb, "[redacted]".to_string()),
    ] {
        assert_eq!(scrub_line(&line), want, "for {line:?}");
    }
}

/// The reviewer's lines: `(what it is, the line, substrings that must not survive)`.
fn reviewers_lines() -> Vec<(&'static str, String, Vec<String>)> {
    let pat = tok("pat");
    let ses = tok("ses");
    let legacy = LEGACY.to_string();
    let enc = pat.replacen('_', "%5F", 2);
    let long_pre = "x".repeat(40_000);
    let csrf = "A-wCyU91xe9jXHeH2QyQhZgv00DTWUwk1H9x_7RSRmg";
    let jwt_head = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
    let jwt_body = "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ";
    let jwt_sig = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
    let stripe = ["sk", "_live_", "abcdefghijklmnopqrstuvwx"].concat();
    let ghp = ["gh", "p_", "abcdefghijklmnopqrstuvwxyz0123"].concat();
    let aws_id = ["AK", "IAIOSFODNN7EXAMPLE"].concat();
    let aws_secret = ["wJalrXUtnFEMI/K7MDENG", "/bPxRfiCYEXAMPLEKEY"].concat();
    let v = |s: &str| vec![s.to_string()];
    vec![
        ("plain token", format!("started run with {pat} ok"), v(S)),
        (
            "token in query token=",
            format!("GET /api/x?token={pat}&a=1"),
            v(S),
        ),
        (
            "token in query access_token=",
            format!("GET /api/x?access_token={pat}"),
            v(S),
        ),
        ("token in query t=", format!("GET /api/x?t={pat}"), v(S)),
        (
            "token in fragment",
            format!("redirect http://x/#access_token={pat}"),
            v(S),
        ),
        (
            "Authorization header",
            format!("Authorization: Bearer {pat}"),
            v(S),
        ),
        (
            "authorization= form",
            format!("authorization=Bearer {pat}"),
            v(S),
        ),
        (
            "curl -H",
            format!("curl -H \"Authorization: Bearer {pat}\" http://x"),
            v(S),
        ),
        (
            "JSON Authorization",
            format!("{{\"Authorization\":\"Bearer {pat}\"}}"),
            v(S),
        ),
        (
            "Cookie header",
            format!("Cookie: __Host-oaiy_dash={ses}; other=1"),
            v(S),
        ),
        (
            "Set-Cookie header",
            format!("Set-Cookie: oaiy_dash={ses}; Path=/; HttpOnly"),
            v(S),
        ),
        (
            "cookie k=v in log (no header word)",
            format!("cookies: {{ oaiy_dash: '{ses}' }}"),
            v(S),
        ),
        ("JSON token", format!("{{\"token\":\"{pat}\"}}"), v(S)),
        ("array of tokens", format!("[\"{pat}\",\"{ses}\"]"), v(S)),
        ("id= with a token", format!("id={pat}"), v(S)),
        ("token glued to letters after", format!("{pat}abc"), v(S)),
        (
            "token glued to letters before",
            format!("xoaiy{}", &pat[4..]),
            v(S),
        ),
        ("token followed by _more", format!("{pat}_more"), v(S)),
        ("url-encoded underscore", format!("GET /x?y={enc}"), v(S)),
        (
            "JSON unicode-escaped underscore",
            format!("{{\"a\":\"oaiypat\\u005f0123456789abcdef\\u005f{S}\"}}"),
            v(S),
        ),
        ("legacy token", format!("paired {legacy} ok"), v(LEGACY_HEX)),
        (
            "legacy token glued to letters",
            format!("paired {legacy}xyz"),
            v(LEGACY_HEX),
        ),
        (
            "legacy token then _suffix",
            format!("paired {legacy}_x"),
            v(LEGACY_HEX),
        ),
        ("csrf header", format!("X-OAIY-CSRF: {csrf}"), v(csrf)),
        ("csrf key=value", format!("x-oaiy-csrf={csrf}"), v(csrf)),
        (
            "JWT bare",
            format!("user sent {jwt_head}.{jwt_body}.{jwt_sig}"),
            v(jwt_sig),
        ),
        (
            "JWT under token key",
            "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcDEF123".to_string(),
            v("abcDEF123"),
        ),
        (
            "URL with credentials",
            "connecting to https://admin:Sup3rS3cret@db.example.com/x".to_string(),
            v("Sup3rS3cret"),
        ),
        (
            "postgres URL",
            "DATABASE_URL: postgres://app:hunter22@localhost:5432/app".to_string(),
            v("hunter22"),
        ),
        (
            "password= plain",
            "password=hunter2".to_string(),
            v("hunter2"),
        ),
        (
            "password= with ;",
            "password=Sup3r;Secret!".to_string(),
            v("Secret!"),
        ),
        (
            "password= with ,",
            "password=Sup3r,Secret!".to_string(),
            v("Secret!"),
        ),
        (
            "password= with &",
            "password=Sup3r&Secret!".to_string(),
            v("Secret!"),
        ),
        (
            "password= with )",
            "password=Sup3r)Secret!".to_string(),
            v("Secret!"),
        ),
        (
            "password= with ]",
            "password=Sup3r]Secret!".to_string(),
            v("Secret!"),
        ),
        (
            "password= with a quote inside",
            "password=Sup3r\"Secret!".to_string(),
            v("Secret!"),
        ),
        (
            "password double-quoted with space",
            "password=\"correct horse battery\"".to_string(),
            vec!["horse".into(), "battery".into()],
        ),
        (
            "password single-quoted",
            "password='p;w x'".to_string(),
            v("p;w"),
        ),
        (
            "password: yaml",
            "password: hunter2".to_string(),
            v("hunter2"),
        ),
        (
            "password : spaced",
            "password : hunter2".to_string(),
            v("hunter2"),
        ),
        (
            "JSON password",
            "{\"password\":\"hunter2\",\"user\":\"a\"}".to_string(),
            v("hunter2"),
        ),
        (
            "JSON in a JSON string",
            "{\"body\":\"{\\\"password\\\":\\\"hunter2\\\"}\"}".to_string(),
            v("hunter2"),
        ),
        (
            "JSON in a JSON string, token key",
            "{\"body\":\"{\\\"token\\\":\\\"abcdef123456\\\"}\"}".to_string(),
            v("abcdef123456"),
        ),
        (
            "escaped quotes password",
            "log: \\\"password\\\": \\\"hunter2\\\"".to_string(),
            v("hunter2"),
        ),
        (
            "Password upper",
            "PASSWORD=hunter2".to_string(),
            v("hunter2"),
        ),
        (
            "Api-Key header form",
            "Api-Key: abcdef123456".to_string(),
            v("abcdef123456"),
        ),
        (
            "x-api-key",
            "x-api-key: abcdef123456".to_string(),
            v("abcdef123456"),
        ),
        (
            "X-Auth-Token",
            "X-Auth-Token: abcdef123456".to_string(),
            v("abcdef123456"),
        ),
        (
            "Proxy-Authorization",
            "Proxy-Authorization: Basic dXNlcjpwYXNz".to_string(),
            v("dXNlcjpwYXNz"),
        ),
        (
            "Authorization Basic",
            "Authorization: Basic dXNlcjpwYXNz".to_string(),
            v("dXNlcjpwYXNz"),
        ),
        (
            "--password flag",
            "spawn: tool --password hunter2 --x".to_string(),
            v("hunter2"),
        ),
        (
            "--token flag",
            "spawn: tool --token abcdef123456".to_string(),
            v("abcdef123456"),
        ),
        (
            "setup code",
            "setup code K7QX-4M2P".to_string(),
            v("K7QX-4M2P"),
        ),
        (
            "invite code",
            "invite M6SC-7N75-YR3H".to_string(),
            v("M6SC-7N75-YR3H"),
        ),
        (
            "lowercase setup code",
            "setup code k7qx-4m2p".to_string(),
            v("k7qx-4m2p"),
        ),
        (
            "code with en dash",
            "setup code K7QX\u{2013}4M2P".to_string(),
            v("4M2P"),
        ),
        (
            "sk- key",
            "key sk-abcdefghijklmnopqrstuvwx".to_string(),
            v("abcdefghijklmnop"),
        ),
        (
            "sk_live stripe key",
            format!("stripe {stripe}"),
            v("abcdefghijklmnop"),
        ),
        ("ghp_", ghp, v("abcdefghijklmnop")),
        (
            "hf_",
            "downloading with hf_abcdefghijklmnopqrstuvwx".to_string(),
            v("abcdefghijklmnop"),
        ),
        ("AWS access key id", format!("aws {aws_id}"), v(&aws_id)),
        (
            "AWS secret under a secret key",
            format!("aws_secret_access_key={aws_secret}"),
            v("wJalrXUtnFEMI"),
        ),
        (
            "private key block",
            "-----BEGIN PRIVATE KEY-----".to_string(),
            vec![],
        ),
        ("bearer lower", format!("bearer {pat}"), v(S)),
        ("Bearer with tab", format!("Bearer\t{pat}"), v(S)),
        ("Bearer with nbsp", format!("Bearer\u{a0}{pat}"), v(S)),
        (
            "fullwidth colon",
            "password\u{ff1a}hunter2".to_string(),
            v("hunter2"),
        ),
        (
            "unicode key",
            "\u{30d1}\u{30b9}\u{30ef}\u{30fc}\u{30c9} password: h\u{fc}nter2".to_string(),
            v("h\u{fc}nter2"),
        ),
        (
            "emoji then token",
            format!("\u{1f600} {pat} \u{1f600}"),
            v(S),
        ),
        ("NUL in line", format!("a\0b {pat}"), v(S)),
        (
            "CR in line",
            format!("a\rb Authorization: Bearer {pat}"),
            v(S),
        ),
        ("token after 40k chars", format!("{long_pre} {pat}"), v(S)),
        (
            "token at 32760",
            format!("{} {pat}", "y".repeat(32_760)),
            v(S),
        ),
        (
            "cut mid multibyte",
            format!("{}\u{20ac}{pat}", "z".repeat(32_766)),
            v(S),
        ),
        (
            "bearer word too short",
            "Bearer abc12".to_string(),
            v("abc12"),
        ),
        ("key= at end", "password=".to_string(), vec![]),
        (
            "unclosed quote password",
            "password=\"abc def ghi".to_string(),
            vec!["abc".into(), "ghi".into()],
        ),
        (
            "nested braces token",
            "token={a:{b:c}}".to_string(),
            v("b:c"),
        ),
    ]
}

#[test]
fn the_reviewers_lines_leak_nothing() {
    let cases = reviewers_lines();
    assert_eq!(
        cases.len(),
        78,
        "the reviewer's 80 lines, less the two that are known limits (below)"
    );
    let mut leaks = Vec::new();
    for (label, line, forbidden) in &cases {
        let out = scrub_line(line);
        let bad: Vec<&String> = forbidden
            .iter()
            .filter(|f| out.contains(f.as_str()))
            .collect();
        if !bad.is_empty() {
            let shown: String = out.chars().take(140).collect();
            leaks.push(format!("{label}: still contains {bad:?} -> {shown}"));
        }
    }
    assert!(
        leaks.is_empty(),
        "{} of {} lines leak:\n{}",
        leaks.len(),
        cases.len(),
        leaks.join("\n")
    );
}

#[test]
fn the_known_limits_are_pinned_so_that_a_change_to_one_is_deliberate() {
    // `-p hunter2`: a short flag is a port, a path or a count as often as it is a password, so the scrub
    // does not take the word after it.
    assert_eq!(
        scrub_line("mysql -u root -p hunter2"),
        "mysql -u root -p hunter2"
    );
    assert_eq!(scrub_line("ssh -p 2222 host"), "ssh -p 2222 host");
    // A code typed with spaces is not the shape the design gives (`K7QX-4M2P`).
    assert_eq!(scrub_line("setup code K7QX 4M2P"), "setup code K7QX 4M2P");
    // A secret with no shape and no key.
    assert_eq!(scrub_line("the answer is hunter2"), "the answer is hunter2");
}
