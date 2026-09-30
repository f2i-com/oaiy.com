//! Rules about the auth code itself, checked by reading it (a "grep test", done on tokens so that a
//! comment or a string cannot fool it):
//!
//! - **Every secret comparison is constant time.** No `==` or `!=` with a secret-looking value on
//!   either side; the comparison helpers of `token.rs` must be built on `subtle`'s `ct_eq`.
//! - **No secret is logged.** No `log::`, `println!` or `eprintln!` names a secret-looking value.
//! - **Nothing is written except through the private, atomic writers.** No `fs::write` or
//!   `File::create` in the auth code.
//! - **No `unsafe`.**
//!
//! The rules read the non-test code of `src/auth/*.rs` (test blocks and test files are taken out with
//! the same lexer as the route-coverage check).

use std::collections::BTreeSet;
use std::path::Path;

use super::route_coverage::{strip_test_items, tokenize, Tok, Token};

/// Words in a name that say the value is, or is derived from, a secret.
const SECRET_WORDS: [&str; 12] = [
    "hash",
    "secret",
    "csrf",
    "password",
    "digest",
    "mac",
    "code",
    "token",
    "cookie",
    "presented",
    "want",
    "expected",
];
/// Endings that say the value is about a secret, not a secret: a length, a count, a kind.
const HARMLESS_ENDINGS: [&str; 7] = ["len", "size", "count", "kind", "epoch", "ms", "id"];

fn secret_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    SECRET_WORDS.iter().any(|w| lower.contains(w))
        && !HARMLESS_ENDINGS.iter().any(|e| lower.ends_with(e))
}

fn is_punct(t: &[Token], i: usize, c: char) -> bool {
    matches!(t.get(i), Some(Token { tok: Tok::Punct(p), .. }) if *p == c)
}

/// The index just before the group that closes at `t[close]`.
fn before_group(t: &[Token], close: usize) -> usize {
    let mut depth = 0i32;
    let mut k = close as i64;
    while k >= 0 {
        match t[k as usize].tok {
            Tok::Punct(')') | Tok::Punct(']') | Tok::Punct('}') => depth += 1,
            Tok::Punct('(') | Tok::Punct('[') | Tok::Punct('{') => {
                depth -= 1;
                if depth == 0 {
                    return k as usize;
                }
            }
            _ => {}
        }
        k -= 1;
    }
    0
}

/// The name of the value on the left of the operator at `t[op]`: the last identifier of the chain
/// before it (`x.hash` -> `hash`, `x.hash()` -> `hash`, `secret.len()` -> `len`). `None` for a literal.
fn left_operand(t: &[Token], op: usize) -> Option<String> {
    let mut j = op.checked_sub(1)?;
    loop {
        match &t[j].tok {
            Tok::Punct(')') | Tok::Punct(']') => {
                let open = before_group(t, j);
                match open.checked_sub(1) {
                    // A call or an index: the name before the brackets.
                    Some(p) if matches!(t[p].tok, Tok::Ident(_)) => j = p,
                    // A parenthesised expression: its last identifier.
                    _ => {
                        return t[open..j].iter().rev().find_map(|x| {
                            if let Tok::Ident(n) = &x.tok {
                                Some(n.clone())
                            } else {
                                None
                            }
                        })
                    }
                }
            }
            Tok::Ident(n) => return Some(n.clone()),
            _ => return None,
        }
    }
}

/// The name of the value on the right: the last identifier of the chain that starts after the operator.
fn right_operand(t: &[Token], op_end: usize) -> Option<String> {
    let mut k = op_end;
    while matches!(
        t.get(k).map(|x| &x.tok),
        Some(Tok::Punct('&' | '*' | '!' | '('))
    ) {
        k += 1;
    }
    let mut last = match &t.get(k)?.tok {
        Tok::Ident(n) => n.clone(),
        _ => return None,
    };
    k += 1;
    loop {
        if is_punct(t, k, '(') || is_punct(t, k, '[') {
            let mut depth = 0i32;
            while k < t.len() {
                match t[k].tok {
                    Tok::Punct('(') | Tok::Punct('[') => depth += 1,
                    Tok::Punct(')') | Tok::Punct(']') => {
                        depth -= 1;
                        if depth == 0 {
                            k += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            continue;
        }
        let dot = is_punct(t, k, '.');
        let path = is_punct(t, k, ':') && is_punct(t, k + 1, ':');
        if dot || path {
            let next = k + if dot { 1 } else { 2 };
            if let Some(Token {
                tok: Tok::Ident(n), ..
            }) = t.get(next)
            {
                last = n.clone();
                k = next + 1;
                continue;
            }
        }
        break;
    }
    Some(last)
}

/// Comparisons with `==` or `!=` where a side is a secret-looking value and neither side is a literal.
pub(super) fn comparison_violations(file: &str, t: &[Token]) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..t.len() {
        let equals = is_punct(t, i, '=')
            && is_punct(t, i + 1, '=')
            && !(i > 0
                && (is_punct(t, i - 1, '=')
                    || is_punct(t, i - 1, '<')
                    || is_punct(t, i - 1, '>')
                    || is_punct(t, i - 1, '!')));
        let differs = is_punct(t, i, '!') && is_punct(t, i + 1, '=') && !is_punct(t, i + 2, '=');
        if !(equals || differs) {
            continue;
        }
        let (Some(l), Some(r)) = (left_operand(t, i), right_operand(t, i + 2)) else {
            continue;
        };
        if secret_name(&l) || secret_name(&r) {
            out.push(format!("{file}:{}: `{l}` {} `{r}` compares a secret with an ordinary operator: use token::hashes_equal or secrets_equal", t[i].line, if equals { "==" } else { "!=" }));
        }
    }
    out
}

/// A macro call whose arguments name a secret-looking value, for the macros that print or log.
pub(super) fn logging_violations(file: &str, t: &[Token]) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..t.len() {
        let Tok::Ident(name) = &t[i].tok else {
            continue;
        };
        if !is_punct(t, i + 1, '!') || !is_punct(t, i + 2, '(') {
            continue;
        }
        // `log::warn!(`, `println!(`, `eprintln!(`, `format!` is not logging by itself.
        let logging = matches!(
            name.as_str(),
            "println" | "eprintln" | "print" | "eprint" | "dbg"
        ) || (matches!(name.as_str(), "error" | "warn" | "info" | "debug" | "trace")
            && i >= 3
            && is_punct(t, i - 1, ':')
            && matches!(&t[i - 3].tok, Tok::Ident(l) if l == "log"));
        if !logging {
            continue;
        }
        let end = super::route_coverage::group_end(t, i + 2);
        for tok in &t[i + 3..end.saturating_sub(1).max(i + 3)] {
            if let Tok::Ident(arg) = &tok.tok {
                if secret_name(arg) {
                    out.push(format!(
                        "{file}:{}: `{name}!` is given `{arg}`, which looks like a secret",
                        tok.line
                    ));
                }
            }
        }
    }
    out
}

/// `fs::write(` and `File::create(` anywhere in the code.
pub(super) fn write_violations(file: &str, t: &[Token]) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..t.len() {
        let call = |a: &str, b: &str| {
            matches!(&t[i].tok, Tok::Ident(x) if x == a)
                && is_punct(t, i + 1, ':')
                && is_punct(t, i + 2, ':')
                && matches!(t.get(i + 3), Some(Token { tok: Tok::Ident(y), .. }) if y == b)
        };
        if call("fs", "write") || call("File", "create") {
            out.push(format!("{file}:{}: a direct write: use secret_file::write (atomic, private from the first byte)", t[i].line));
        }
    }
    out
}

/// The non-test tokens of every `.rs` file directly under `src/auth`, by file name.
fn auth_code() -> Vec<(String, Vec<Token>)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("auth");
    let mod_rs = std::fs::read_to_string(dir.join("mod.rs")).unwrap();
    let (_, test_mods) = strip_test_items(&tokenize(&mod_rs));
    let tests: BTreeSet<String> = test_mods.iter().map(|m| format!("{m}.rs")).collect();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".rs") || tests.contains(&name) {
            continue;
        }
        let src = std::fs::read_to_string(entry.path()).unwrap();
        out.push((name, strip_test_items(&tokenize(&src)).0));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn the_auth_code_is_found_and_its_test_files_are_left_out() {
    let names: Vec<String> = auth_code().into_iter().map(|(n, _)| n).collect();
    for expected in [
        "token.rs",
        "store.rs",
        "chain.rs",
        "scopes.rs",
        "presets.rs",
        "principal.rs",
        "audit.rs",
        "scrub.rs",
        "lock.rs",
        "routes.rs",
        "export.rs",
        "mod.rs",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "{expected} missing from {names:?}"
        );
    }
    for test_file in ["store_tests.rs", "route_coverage.rs", "conformance.rs"] {
        assert!(
            !names.contains(&test_file.to_string()),
            "{test_file} is test code"
        );
    }
}

#[test]
fn every_secret_comparison_in_the_auth_code_is_constant_time() {
    let mut all = Vec::new();
    for (name, t) in auth_code() {
        all.extend(comparison_violations(&name, &t));
    }
    assert!(all.is_empty(), "{}", all.join("\n"));
}

#[test]
fn the_comparison_helpers_are_built_on_subtles_ct_eq() {
    let (_, t) = auth_code()
        .into_iter()
        .find(|(n, _)| n == "token.rs")
        .unwrap();
    for helper in ["hashes_equal", "secrets_equal"] {
        let at = (0..t.len())
            .find(|i| {
                matches!(&t[*i].tok, Tok::Ident(n) if n == "fn")
                    && matches!(&t[i + 1].tok, Tok::Ident(n) if n == helper)
            })
            .unwrap_or_else(|| panic!("{helper} not found"));
        let open = (at..t.len()).find(|i| is_punct(&t, *i, '{')).unwrap();
        let body = &t[open..super::route_coverage::group_end(&t, open)];
        assert!(
            body.iter()
                .any(|x| matches!(&x.tok, Tok::Ident(n) if n == "ct_eq")),
            "{helper} must use ct_eq"
        );
        assert!(
            !body.iter().any(|x| matches!(x.tok, Tok::Punct('='))),
            "{helper} must not compare with =="
        );
    }
    // And the store uses them for the stored hash.
    let (_, store) = auth_code()
        .into_iter()
        .find(|(n, _)| n == "store.rs")
        .unwrap();
    assert!(
        store
            .iter()
            .any(|x| matches!(&x.tok, Tok::Ident(n) if n == "hashes_equal")),
        "the store must compare hashes with hashes_equal"
    );
}

#[test]
fn the_guard_compares_the_environment_token_in_constant_time() {
    let (_, t) = auth_code()
        .into_iter()
        .find(|(n, _)| n == "guard.rs")
        .unwrap();
    assert!(
        t.iter()
            .any(|x| matches!(&x.tok, Tok::Ident(n) if n == "secrets_equal")),
        "guard.rs must compare the static token with secrets_equal"
    );
}

#[test]
fn nothing_in_the_auth_code_logs_a_secret_looking_value() {
    let mut all = Vec::new();
    for (name, t) in auth_code() {
        all.extend(logging_violations(&name, &t));
    }
    assert!(all.is_empty(), "{}", all.join("\n"));
}

/// `export.rs` writes `routes.json`, which holds no secret, from a test.
const MAY_WRITE_DIRECTLY: [&str; 1] = ["export.rs"];

#[test]
fn the_auth_code_writes_only_through_the_private_atomic_writers() {
    let mut all = Vec::new();
    for (name, t) in auth_code() {
        if !MAY_WRITE_DIRECTLY.contains(&name.as_str()) {
            all.extend(write_violations(&name, &t));
        }
    }
    assert!(all.is_empty(), "{}", all.join("\n"));
}

#[test]
fn the_auth_code_has_no_unsafe() {
    for (name, t) in auth_code() {
        assert!(
            !t.iter()
                .any(|x| matches!(&x.tok, Tok::Ident(n) if n == "unsafe")),
            "{name} uses unsafe"
        );
    }
}

// ---- the crates the auth code adds ------------------------------------------------------------------

/// The `[[package]]` entries of a `Cargo.lock`: `(name, version, dependencies)`, the dependencies as the lock
/// names them (`digest`, or `digest 0.10.7` when the lock has to tell two versions apart).
fn lock_packages(lock: &str) -> Vec<(String, String, Vec<String>)> {
    let quoted = |line: &str| line.split('"').nth(1).map(str::to_string);
    let mut out: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut in_dependencies = false;
    for line in lock.lines() {
        if line.starts_with("[[package]]") {
            out.push((String::new(), String::new(), Vec::new()));
            in_dependencies = false;
        } else if let Some(entry) = out.last_mut() {
            if let Some(rest) = line.strip_prefix("name = ") {
                entry.0 = quoted(rest).unwrap_or_default();
            } else if let Some(rest) = line.strip_prefix("version = ") {
                entry.1 = quoted(rest).unwrap_or_default();
            } else if line.starts_with("dependencies = [") {
                in_dependencies = true;
            } else if line.starts_with(']') {
                in_dependencies = false;
            } else if in_dependencies {
                if let Some(dep) = quoted(line) {
                    entry.2.push(dep);
                }
            }
        }
    }
    out
}

#[test]
fn the_lock_holds_one_subtle_and_one_hmac_that_brings_no_other_crate() {
    // `Cargo.toml` says `subtle` was in the lock already and `hmac` is the one package this adds, depending
    // only on `digest`. A change to that is a change to a dependency: it fails here until it is on purpose.
    let lock = include_str!("../../Cargo.lock");
    let packages = lock_packages(lock);
    let named = |n: &str| -> Vec<&(String, String, Vec<String>)> {
        packages.iter().filter(|p| p.0 == n).collect()
    };
    let hmac = named("hmac");
    assert_eq!(hmac.len(), 1, "one hmac in the lock: {hmac:?}");
    assert!(hmac[0].1.starts_with("0.12."), "{}", hmac[0].1);
    assert_eq!(
        hmac[0].2,
        ["digest"],
        "hmac depends on digest and nothing else"
    );
    let subtle = named("subtle");
    assert_eq!(subtle.len(), 1, "one subtle in the lock: {subtle:?}");
    assert!(subtle[0].1.starts_with("2."), "{}", subtle[0].1);
    // The desktop names both, and so nothing else needs to.
    let ours = named("oaiy-desktop");
    assert_eq!(ours.len(), 1);
    assert!(ours[0].2.contains(&"hmac".to_string()) && ours[0].2.contains(&"subtle".to_string()));
}

#[test]
fn the_lock_reader_finds_a_package_its_version_and_its_dependencies() {
    let lock = r#"
version = 4

[[package]]
name = "alpha"
version = "1.2.3"
dependencies = [
 "beta",
 "gamma 0.1.0",
]

[[package]]
name = "beta"
version = "0.9.0"

[[package]]
name = "gamma"
version = "0.1.0"
dependencies = [
 "beta",
]
"#;
    assert_eq!(
        lock_packages(lock),
        [
            (
                "alpha".to_string(),
                "1.2.3".to_string(),
                vec!["beta".to_string(), "gamma 0.1.0".to_string()]
            ),
            ("beta".to_string(), "0.9.0".to_string(), vec![]),
            (
                "gamma".to_string(),
                "0.1.0".to_string(),
                vec!["beta".to_string()]
            ),
        ]
    );
}

// ---- the checkers themselves, on source they have not seen ----------------------------------------

fn code(src: &str) -> Vec<Token> {
    strip_test_items(&tokenize(src)).0
}

#[test]
fn the_comparison_check_flags_a_secret_on_either_side_and_lets_the_rest_through() {
    for bad in [
        "fn f(r: R, p: &str) -> bool { r.hash == p }",
        "fn f(r: R, p: &str) -> bool { p != r.hash }",
        "fn f(a: &str, b: &str) -> bool { a.csrf() == b }",
        "fn f(secret: &str, given: &str) -> bool { secret == given }",
        "fn f(x: &X) -> bool { x.password == y.password }",
        "fn f(x: &X) -> bool { x.setup_code != *y }",
        "fn f(presented_hash: String) -> bool { presented_hash == DUMMY }",
        "fn f(a: &str) -> bool { a == DUMMY_HASH }",
        "fn f(a: &str) -> bool { (a.token) == other }",
    ] {
        assert_eq!(comparison_violations("t.rs", &code(bad)).len(), 1, "{bad}");
    }
    for fine in [
        "fn f(r: R, k: K) -> bool { r.kind == k }",
        "fn f(secret: &str) -> bool { secret.len() == 43 }",
        "fn f(secret: &str) -> bool { secret.len() != SECRET_LEN }",
        "fn f(r: R) -> bool { r.hash == \"literal\" }",
        "fn f(x: u8) -> bool { x == 3 }",
        "fn f(a: u64, b: u64) -> bool { a >= b && a <= b }",
        "fn f(t: Tok) -> bool { matches!(t, Tok::Ident(n) if n == \"x\") }",
        "fn f(r: R) -> bool { r.token_count == r.max }",
        "fn f() -> u8 { let hash = 5; hash }",
        "fn f(a: A, b: B) -> bool { a.id == b.id }",
        "fn f(s: S) -> Option<()> { let hash = s.hash; if s.ok => { Some(()) } else { None } }",
    ] {
        assert!(
            comparison_violations("t.rs", &code(fine)).is_empty(),
            "{fine}"
        );
    }
    // A comparison in test code is not the auth code's.
    assert!(comparison_violations(
        "t.rs",
        &code("#[cfg(test)] mod tests { fn f(r: R, p: &str) -> bool { r.hash == p } }")
    )
    .is_empty());
}

#[test]
fn the_logging_check_flags_a_macro_that_names_a_secret() {
    for bad in [
        "fn f(token: &str) { log::warn!(\"bad {}\", token); }",
        "fn f(t: &T) { eprintln!(\"x {}\", t.secret); }",
        "fn f(hash: &str) { println!(\"{}\", hash); }",
        "fn f(c: C) { log::error!(\"{}\", c.password); }",
        "fn f(c: C) { dbg!(c.csrf); }",
    ] {
        assert_eq!(logging_violations("t.rs", &code(bad)).len(), 1, "{bad}");
    }
    for fine in [
        "fn f(e: E) { log::warn!(\"auth: could not write: {e}\"); }",
        "fn f(id: &str) { eprintln!(\"oaiy-audit event={}\", id); }",
        "fn f(token: &str) -> String { format!(\"{}\", token) }",
        "fn f(a: A) { log::info!(\"{}\", a.count); }",
    ] {
        assert!(logging_violations("t.rs", &code(fine)).is_empty(), "{fine}");
    }
}

#[test]
fn the_write_check_flags_a_direct_write() {
    assert_eq!(
        write_violations("t.rs", &code("fn f() { std::fs::write(p, b); }")).len(),
        1
    );
    assert_eq!(
        write_violations("t.rs", &code("fn f() { let f = File::create(p); }")).len(),
        1
    );
    assert!(write_violations(
        "t.rs",
        &code("fn f() { secret_file::write(p, b); std::fs::rename(a, b); }")
    )
    .is_empty());
}
