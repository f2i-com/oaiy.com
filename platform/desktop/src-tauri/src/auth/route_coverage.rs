//! The route-coverage check (design sections 4.3 and 9.2), run by plain `cargo test`.
//!
//! It reads every `.rs` file under `src/`, takes out what only exists in tests, finds every axum
//! `.route("…", get(..).post(..))` call, and compares the `(method, path)` pairs with [`ROUTES`]:
//!
//! - a route with no row fails the test (a route added without a classification is unusable in
//!   `scoped` mode and must not be merged), and
//! - a row with no route fails the test unless it is `since: 2` (a route another step of the access
//!   model adds, or another design announced).
//!
//! The walker has a small lexer of its own, so a brace in a string, a comment or a raw string cannot
//! end a `#[cfg(test)]` block early, and it fails loudly on what it cannot follow (a route whose path
//! is not a literal, a routing helper it does not know, `.nest(`, `.fallback(`) instead of skipping it.
//! Its fixtures pin those behaviours.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::routes::{Only, Route, Verb, ROUTES};

// ---------------------------------------------------------------------------------------------
// A small Rust lexer: identifiers, string contents and punctuation, with comments and the insides
// of character literals dropped.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Tok {
    Ident(String),
    Str(String),
    Punct(char),
    Other,
}

#[derive(Clone, Debug)]
pub(super) struct Token {
    pub tok: Tok,
    pub line: usize,
}

/// If a string literal (plain, byte, raw, raw byte) starts at `i`: where its content starts, how many
/// `#` close it, and whether it is raw.
fn string_start(c: &[char], i: usize) -> Option<(usize, usize, bool)> {
    let at = |k: usize| c.get(k).copied().unwrap_or('\0');
    let mut j = i;
    if at(j) == 'b' {
        j += 1;
    }
    if at(j) == 'r' {
        let mut k = j + 1;
        let mut hashes = 0;
        while at(k) == '#' {
            hashes += 1;
            k += 1;
        }
        return (at(k) == '"').then_some((k + 1, hashes, true));
    }
    (at(j) == '"').then_some((j + 1, 0, false))
}

pub(super) fn tokenize(src: &str) -> Vec<Token> {
    let c: Vec<char> = src.chars().collect();
    let n = c.len();
    let at = |k: usize| c.get(k).copied().unwrap_or('\0');
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = 1usize;
    while i < n {
        let ch = c[i];
        if ch == '\n' {
            line += 1;
            i += 1;
        } else if ch.is_whitespace() {
            i += 1;
        } else if ch == '/' && at(i + 1) == '/' {
            while i < n && c[i] != '\n' {
                i += 1;
            }
        } else if ch == '/' && at(i + 1) == '*' {
            let mut depth = 1;
            i += 2;
            while i < n && depth > 0 {
                if c[i] == '/' && at(i + 1) == '*' {
                    depth += 1;
                    i += 2;
                } else if c[i] == '*' && at(i + 1) == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    if c[i] == '\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
        } else if let Some((start, hashes, raw)) =
            string_start(&c, i).filter(|_| ch == '"' || ch == 'b' || ch == 'r')
        {
            let first_line = line;
            let mut j = start;
            let mut content = String::new();
            loop {
                if j >= n {
                    break;
                }
                if raw {
                    if c[j] == '"' && (0..hashes).all(|h| at(j + 1 + h) == '#') {
                        j += 1 + hashes;
                        break;
                    }
                } else if c[j] == '\\' {
                    content.push(c[j]);
                    if let Some(&next) = c.get(j + 1) {
                        content.push(next);
                        if next == '\n' {
                            line += 1;
                        }
                    }
                    j += 2;
                    continue;
                } else if c[j] == '"' {
                    j += 1;
                    break;
                }
                if c[j] == '\n' {
                    line += 1;
                }
                content.push(c[j]);
                j += 1;
            }
            out.push(Token {
                tok: Tok::Str(content),
                line: first_line,
            });
            i = j;
        } else if ch == '\'' || (ch == 'b' && at(i + 1) == '\'') {
            // A character literal, or a lifetime / loop label.
            let q = if ch == 'b' { i + 1 } else { i };
            if at(q + 1) == '\\' {
                let mut j = q + 3;
                while j < n && c[j] != '\'' {
                    j += 1;
                }
                i = j + 1;
                out.push(Token {
                    tok: Tok::Other,
                    line,
                });
            } else if at(q + 2) == '\'' {
                i = q + 3;
                out.push(Token {
                    tok: Tok::Other,
                    line,
                });
            } else {
                // A lifetime: the quote is dropped and the name that follows is an identifier.
                i = q + 1;
                out.push(Token {
                    tok: Tok::Other,
                    line,
                });
            }
        } else if ch.is_alphabetic() || ch == '_' {
            let mut j = i;
            // A raw identifier: `r#type`.
            if ch == 'r' && at(i + 1) == '#' && (at(i + 2).is_alphabetic() || at(i + 2) == '_') {
                j = i + 2;
            }
            let start = j;
            while j < n && (c[j].is_alphanumeric() || c[j] == '_') {
                j += 1;
            }
            out.push(Token {
                tok: Tok::Ident(c[start..j].iter().collect()),
                line,
            });
            i = j;
        } else if ch.is_ascii_digit() {
            while i < n && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Token {
                tok: Tok::Other,
                line,
            });
        } else {
            out.push(Token {
                tok: Tok::Punct(ch),
                line,
            });
            i += 1;
        }
    }
    out
}

fn is_ident(t: &[Token], i: usize, name: &str) -> bool {
    matches!(t.get(i), Some(Token { tok: Tok::Ident(n), .. }) if n == name)
}

fn is_punct(t: &[Token], i: usize, p: char) -> bool {
    matches!(t.get(i), Some(Token { tok: Tok::Punct(q), .. }) if *q == p)
}

/// The index just after the bracketed group that opens at `t[i]` (any of `( [ {`).
pub(super) fn group_end(t: &[Token], i: usize) -> usize {
    skip_group(t, i)
}

fn skip_group(t: &[Token], i: usize) -> usize {
    let mut depth = 0i32;
    let mut k = i;
    while k < t.len() {
        match t[k].tok {
            Tok::Punct('(') | Tok::Punct('[') | Tok::Punct('{') => depth += 1,
            Tok::Punct(')') | Tok::Punct(']') | Tok::Punct('}') => {
                depth -= 1;
                if depth == 0 {
                    return k + 1;
                }
            }
            _ => {}
        }
        k += 1;
    }
    t.len()
}

/// `#[cfg(test)]` at `t[i]`: the index after it.
fn cfg_test_attribute(t: &[Token], i: usize) -> Option<usize> {
    let ok = is_punct(t, i, '#')
        && is_punct(t, i + 1, '[')
        && is_ident(t, i + 2, "cfg")
        && is_punct(t, i + 3, '(')
        && is_ident(t, i + 4, "test")
        && is_punct(t, i + 5, ')')
        && is_punct(t, i + 6, ']');
    ok.then_some(i + 7)
}

/// What is left of a file once everything `#[cfg(test)]` is taken out, and the names of the
/// `#[cfg(test)] mod name;` declarations (whose files are test files).
pub(super) fn strip_test_items(t: &[Token]) -> (Vec<Token>, Vec<String>) {
    // `#![cfg(test)]`: the whole file is test code.
    if is_punct(t, 0, '#')
        && is_punct(t, 1, '!')
        && is_punct(t, 2, '[')
        && is_ident(t, 3, "cfg")
        && is_punct(t, 4, '(')
        && is_ident(t, 5, "test")
    {
        return (Vec::new(), Vec::new());
    }
    let mut kept = Vec::with_capacity(t.len());
    let mut test_mods = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let Some(mut j) = cfg_test_attribute(t, i) else {
            kept.push(t[i].clone());
            i += 1;
            continue;
        };
        // Any attributes that follow belong to the same item.
        while is_punct(t, j, '#') && is_punct(t, j + 1, '[') {
            j = skip_group(t, j + 1);
        }
        // `pub(crate) mod name;` is a declaration of a file.
        let mut m = j;
        if is_ident(t, m, "pub") {
            m += 1;
            if is_punct(t, m, '(') {
                m = skip_group(t, m);
            }
        }
        if is_ident(t, m, "mod") && is_punct(t, m + 2, ';') {
            if let Some(Token {
                tok: Tok::Ident(name),
                ..
            }) = t.get(m + 1)
            {
                test_mods.push(name.clone());
            }
        }
        // The item ends at its `;` or at the end of its first `{ … }` block.
        let mut k = j;
        while k < t.len() {
            match t[k].tok {
                Tok::Punct('(') | Tok::Punct('[') => k = skip_group(t, k),
                Tok::Punct('{') => {
                    k = skip_group(t, k);
                    break;
                }
                Tok::Punct(';') => {
                    k += 1;
                    break;
                }
                _ => k += 1,
            }
        }
        i = k;
    }
    (kept, test_mods)
}

// ---------------------------------------------------------------------------------------------
// Finding the routes.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Found {
    pub file: String,
    pub line: usize,
    pub verb: Verb,
    pub pattern: String,
}

/// The value `&format!("{GATEWAY_PREFIX}/*path")` builds, the one dynamic path the API has.
const GATEWAY_PREFIX: &str = "/api/ai/engine/gateway";

/// Split the tokens of a call's arguments (`t[open]` is the `(`) at top-level commas.
fn call_args(t: &[Token], open: usize) -> (Vec<&[Token]>, usize) {
    let end = skip_group(t, open);
    let inner = &t[open + 1..end.saturating_sub(1).max(open + 1)];
    let mut args = Vec::new();
    let mut start = 0;
    let mut k = 0;
    while k < inner.len() {
        match inner[k].tok {
            Tok::Punct('(') | Tok::Punct('[') | Tok::Punct('{') => {
                k = skip_group(inner, k);
                continue;
            }
            Tok::Punct(',') => {
                args.push(&inner[start..k]);
                start = k + 1;
            }
            _ => {}
        }
        k += 1;
    }
    if start < inner.len() {
        args.push(&inner[start..]);
    }
    (args, end)
}

/// The path a `.route(` call's first argument spells, or why it cannot be known here.
fn route_path(arg: &[Token]) -> Result<String, String> {
    match arg {
        [Token {
            tok: Tok::Str(s), ..
        }] => Ok(s.clone()),
        // `&format!("{GATEWAY_PREFIX}/*path")`
        [Token {
            tok: Tok::Punct('&'),
            ..
        }, Token {
            tok: Tok::Ident(f), ..
        }, Token {
            tok: Tok::Punct('!'),
            ..
        }, Token {
            tok: Tok::Punct('('),
            ..
        }, Token {
            tok: Tok::Str(s), ..
        }, ..]
            if f == "format" =>
        {
            let path = s.replace("{GATEWAY_PREFIX}", GATEWAY_PREFIX);
            if path.contains('{') {
                Err(format!(
                    "a formatted route path with a placeholder this check does not know: {s:?}"
                ))
            } else {
                Ok(path)
            }
        }
        _ => Err("a route path that is not a string literal".into()),
    }
}

/// The methods a `.route(` call's second argument routes: `get(a).post(b)`, `axum::routing::any(c)`.
fn route_verbs(arg: &[Token]) -> Result<Vec<Verb>, String> {
    let mut verbs = Vec::new();
    let mut k = 0;
    while k < arg.len() {
        match &arg[k].tok {
            Tok::Punct('(') | Tok::Punct('[') | Tok::Punct('{') => {
                k = skip_group(arg, k);
                continue;
            }
            Tok::Ident(name) if is_punct(arg, k + 1, '(') => {
                let verb = match name.as_str() {
                    "get" => Some(Verb::Get),
                    "post" => Some(Verb::Post),
                    "put" => Some(Verb::Put),
                    "patch" => Some(Verb::Patch),
                    "delete" => Some(Verb::Delete),
                    "any" => Some(Verb::Any),
                    "head" | "options" | "trace" | "on" | "on_service" | "get_service"
                    | "post_service" | "put_service" | "patch_service" | "delete_service"
                    | "any_service" | "head_service" | "options_service" | "trace_service" => {
                        return Err(format!(
                            "the routing helper `{name}`, which this check does not follow"
                        ));
                    }
                    _ => None,
                };
                verbs.extend(verb);
            }
            _ => {}
        }
        k += 1;
    }
    if verbs.is_empty() {
        Err("a route with no method this check can read".into())
    } else {
        Ok(verbs)
    }
}

/// Everything the scan found and everything it could not follow.
#[derive(Debug, Default)]
pub(crate) struct Scan {
    pub found: Vec<Found>,
    pub problems: Vec<String>,
}

/// The module files that `#[cfg(test)] mod name;` in `file` declares.
fn declared_test_files(file: &str, names: &[String]) -> Vec<String> {
    let p = Path::new(file);
    let file_name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parent = p
        .parent()
        .and_then(|d| d.to_str())
        .unwrap_or("")
        .replace('\\', "/");
    let base = if matches!(file_name, "mod.rs" | "lib.rs" | "main.rs") {
        parent
    } else {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if parent.is_empty() {
            stem.to_string()
        } else {
            format!("{parent}/{stem}")
        }
    };
    let join = |rest: String| {
        if base.is_empty() {
            rest
        } else {
            format!("{base}/{rest}")
        }
    };
    names
        .iter()
        .flat_map(|n| [join(format!("{n}.rs")), join(format!("{n}/mod.rs"))])
        .collect()
}

/// Scan `files` (`(path relative to src, source)`).
pub(super) fn scan(files: &[(String, String)]) -> Scan {
    let stripped: Vec<(String, Vec<Token>, Vec<String>)> = files
        .iter()
        .map(|(path, src)| {
            let (kept, mods) = strip_test_items(&tokenize(src));
            (path.clone(), kept, mods)
        })
        .collect();
    let test_files: BTreeSet<String> = stripped
        .iter()
        .flat_map(|(path, _, mods)| declared_test_files(path, mods))
        .collect();

    let mut scan = Scan::default();
    for (path, t, _) in &stripped {
        if test_files.contains(path) {
            continue;
        }
        for i in 0..t.len() {
            if !is_punct(t, i, '.') {
                continue;
            }
            let line = t[i].line;
            let Some(Token {
                tok: Tok::Ident(method),
                ..
            }) = t.get(i + 1)
            else {
                continue;
            };
            if !is_punct(t, i + 2, '(') {
                continue;
            }
            match method.as_str() {
                "nest" | "nest_service" | "route_service" | "fallback" | "fallback_service" => {
                    scan.problems.push(format!("{path}:{line}: `.{method}(` changes what `MatchedPath` is, so the guard's keying does not cover it"));
                }
                "route" => {
                    let (args, _) = call_args(t, i + 2);
                    // A method named `route` with one argument is not axum's `Router::route`.
                    if args.len() < 2 {
                        continue;
                    }
                    let pattern = route_path(args[0]);
                    let verbs = route_verbs(args[1]);
                    match (pattern, verbs) {
                        (Ok(pattern), Ok(verbs)) => {
                            for verb in verbs {
                                scan.found.push(Found {
                                    file: path.clone(),
                                    line,
                                    verb,
                                    pattern: pattern.clone(),
                                });
                            }
                        }
                        (p, v) => {
                            for why in [p.err(), v.err()].into_iter().flatten() {
                                scan.problems.push(format!("{path}:{line}: {why}"));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    scan
}

// ---------------------------------------------------------------------------------------------
// Comparing with the table.
// ---------------------------------------------------------------------------------------------

/// Routes the scan finds that are not on the main router: the voice gateway's own listener
/// (`voice/mod.rs`, port 17872, its own token) answers its own `/api/health` and realtime stream.
pub(super) const NOT_ON_MAIN_ROUTER: &[(&str, &str)] = &[
    ("voice/mod.rs", "/api/health"),
    ("voice/mod.rs", "/api/ai/providers/:id/v1/realtime/stream"),
];

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Comparison {
    /// Routes with no row.
    pub unclassified: Vec<String>,
    /// Rows with `since: 1` and no route.
    pub stale: Vec<String>,
}

fn key(verb: Verb, pattern: &str) -> String {
    format!("{} {}", verb.as_str(), pattern)
}

pub(super) fn compare(found: &[Found], table: &[Route]) -> Comparison {
    let rows: BTreeSet<(Verb, &str)> = table.iter().map(|r| (r.method, r.pattern)).collect();
    let routes: BTreeSet<(Verb, &str)> =
        found.iter().map(|f| (f.verb, f.pattern.as_str())).collect();
    let mut out = Comparison::default();
    for f in found {
        if !rows.contains(&(f.verb, f.pattern.as_str())) {
            out.unclassified.push(format!(
                "{} ({}:{})",
                key(f.verb, &f.pattern),
                f.file,
                f.line
            ));
        }
    }
    out.unclassified.sort();
    out.unclassified.dedup();
    for r in table {
        if r.since == 1 && !routes.contains(&(r.method, r.pattern)) {
            out.stale.push(r.key());
        }
    }
    out
}

/// The `.rs` files under `src/`, as `(path relative to src with `/`, source)`.
fn source_files() -> Vec<(String, String)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .flatten()
            .collect();
        entries.sort_by_key(|e| e.path());
        for e in entries {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let rel = p
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let src =
                    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                out.push((rel, src));
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    out
}

/// The scan of the real source, with the routes that are not on the main router taken out.
pub(crate) fn scan_main_router() -> Scan {
    let mut s = scan(&source_files());
    let listed: BTreeSet<(&str, &str)> = NOT_ON_MAIN_ROUTER.iter().copied().collect();
    let mut seen = BTreeSet::new();
    s.found.retain(|f| {
        let hit = listed.contains(&(f.file.as_str(), f.pattern.as_str()));
        if hit {
            seen.insert((f.file.clone(), f.pattern.clone()));
        }
        !hit
    });
    for (file, pattern) in &listed {
        assert!(
            seen.contains(&(file.to_string(), pattern.to_string())),
            "NOT_ON_MAIN_ROUTER lists {pattern} in {file}, which is not there any more: remove the entry"
        );
    }
    s
}

// ---------------------------------------------------------------------------------------------
// Tests on the real source.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_scan_of_the_source_follows_every_route_it_meets() {
    let s = scan_main_router();
    assert!(
        s.problems.is_empty(),
        "routes this check cannot follow:\n{}",
        s.problems.join("\n")
    );
}

#[test]
fn every_route_of_the_main_router_is_under_api_so_that_the_guard_covers_it() {
    // The guard judges what `axum` matched, and a path outside `/api/` that no route answers is not under it
    // (the static UI of a later step does its own checks). A route added outside `/api/` would be served
    // with no credential: it must be a deliberate change here.
    let s = scan_main_router();
    let outside: Vec<String> = s
        .found
        .iter()
        .filter(|f| !f.pattern.starts_with("/api/"))
        .map(|f| format!("{} {} ({}:{})", f.verb.as_str(), f.pattern, f.file, f.line))
        .collect();
    assert!(
        outside.is_empty(),
        "routes outside /api/ are not behind the guard:\n{}",
        outside.join("\n")
    );
}

#[test]
fn every_route_in_the_source_has_a_row_in_the_table() {
    let s = scan_main_router();
    let c = compare(&s.found, ROUTES);
    assert!(
        c.unclassified.is_empty(),
        "routes with no row in auth/routes.rs (add one; a route without a row is refused in scoped mode):\n{}",
        c.unclassified.join("\n")
    );
}

#[test]
fn every_row_of_a_route_that_existed_before_the_access_model_has_a_route() {
    let s = scan_main_router();
    let c = compare(&s.found, ROUTES);
    assert!(
        c.stale.is_empty(),
        "rows with since 1 and no route in the source (remove the row or mark it since 2):\n{}",
        c.stale.join("\n")
    );
}

#[test]
fn the_walker_finds_the_routes_the_design_counted_and_the_ones_added_since() {
    let s = scan_main_router();
    let pairs: BTreeSet<(Verb, String)> = s
        .found
        .iter()
        .map(|f| (f.verb, f.pattern.clone()))
        .collect();
    // The design's Appendix B counted 161 pairs on the main router; the four below are routes the
    // code has gained since (the plugin trust route and three update routes), the eleven of the
    // receptionist's transfers and messages (the four the design reserved, `GET|PATCH /api/messages`
    // and `GET|PUT /api/ring/settings`, and seven it did not have), and the three routes of `/api/auth/`
    // that the access model has built so far are the rows with `since: 2` that exist.
    assert_eq!(
        pairs.len(),
        165 + 11 + 3,
        "main-router (method, path) pairs"
    );
    let mut built: Vec<String> = ROUTES
        .iter()
        .filter(|r| r.since == 2 && pairs.contains(&(r.method, r.pattern.to_string())))
        .map(|r| r.key())
        .collect();
    built.sort();
    assert_eq!(
        built,
        [
            "GET /api/auth/info",
            "GET /api/auth/whoami",
            "POST /api/auth/derive"
        ],
        "the routes of the model that exist"
    );
    for (verb, pattern) in [
        (Verb::Get, "/api/health"),
        (Verb::Post, "/api/plugins/:id/trust"),
        (Verb::Get, "/api/update/status"),
        (Verb::Post, "/api/update/check"),
        (Verb::Post, "/api/update/agent-flushed"),
        (Verb::Any, "/api/ai/engine/gateway/*path"),
        (Verb::Get, "/api/plugins/:id/ui/:screen/*path"),
    ] {
        assert!(
            pairs.contains(&(verb, pattern.to_string())),
            "{} {pattern} not found",
            verb.as_str()
        );
    }
    // A test-only router is not the main router.
    assert!(!pairs.iter().any(|(_, p)| p == "/latest.json"));
}

#[test]
fn the_table_has_a_row_for_every_route_the_boot_test_probes_and_the_counts_the_design_gives() {
    let existing = ROUTES.iter().filter(|r| r.since == 1).count();
    let new_or_reserved = ROUTES.iter().filter(|r| r.since == 2).count();
    // 161 in the design, and the eleven routes the code gained since (see above).
    assert_eq!(existing, 165 + 11);
    // 41 new and 60 reserved in the design; `GET /api/update` (reserved) is gone, replaced by the
    // real `GET /api/update/status`, and four reserved rows (`GET|PATCH /api/messages`,
    // `GET|PUT /api/ring/settings`) are `since: 1` now that their routes exist.
    assert_eq!(new_or_reserved, 100 - 4);
    assert_eq!(ROUTES.iter().filter(|r| r.only == Only::Server).count(), 28);
}

// ---------------------------------------------------------------------------------------------
// Fixtures: what the walker must do with source it has not seen.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod fixtures {
    use super::*;
    use crate::auth::routes::{Class, Route, Verb};

    fn table(rows: &[(Verb, &'static str, u8)]) -> Vec<Route> {
        rows.iter()
            .map(|(m, p, since)| Route {
                method: *m,
                pattern: p,
                class: Class::Public,
                since: *since,
                only: Only::Everywhere,
            })
            .collect()
    }

    fn scan_one(src: &str) -> Scan {
        scan(&[("router.rs".to_string(), src.to_string())])
    }

    fn keys(s: &Scan) -> Vec<String> {
        let mut k: Vec<String> = s.found.iter().map(|f| key(f.verb, &f.pattern)).collect();
        k.sort();
        k
    }

    #[test]
    fn a_route_with_no_row_is_reported() {
        let s = scan_one(
            r#"fn r() -> Router { Router::new().route("/api/new", get(list)).route("/api/old", post(make)) }"#,
        );
        let c = compare(&s.found, &table(&[(Verb::Post, "/api/old", 1)]));
        assert_eq!(c.unclassified.len(), 1, "{c:?}");
        assert!(c.unclassified[0].starts_with("GET /api/new "), "{c:?}");
        assert!(c.stale.is_empty());
    }

    #[test]
    fn a_stale_row_is_reported_unless_it_is_since_2() {
        let s = scan_one(r#"fn r() -> Router { Router::new().route("/api/old", get(list)) }"#);
        let c = compare(
            &s.found,
            &table(&[
                (Verb::Get, "/api/old", 1),
                (Verb::Delete, "/api/gone", 1),
                (Verb::Post, "/api/future", 2),
            ]),
        );
        assert_eq!(c.stale, ["DELETE /api/gone"]);
        assert!(c.unclassified.is_empty());
    }

    #[test]
    fn a_method_missing_from_the_table_is_a_route_with_no_row() {
        let s = scan_one(r#"fn r() -> Router { Router::new().route("/api/x", get(a).post(b)) }"#);
        let c = compare(&s.found, &table(&[(Verb::Get, "/api/x", 1)]));
        assert_eq!(c.unclassified.len(), 1, "{c:?}");
        assert!(c.unclassified[0].starts_with("POST /api/x "), "{c:?}");
    }

    #[test]
    fn a_test_module_declared_pub_crate_is_not_the_main_router() {
        // The prototype of this check missed `pub(crate) mod tests {`, and reported `/latest.json`.
        let s = scan_one(
            r#"
            pub fn router() -> Router { Router::new().route("/api/real", get(h)) }
            #[cfg(test)]
            pub(crate) mod tests {
                fn feed() { let app = Router::new().route("/latest.json", axum::routing::get(f)); }
            }
            "#,
        );
        assert_eq!(keys(&s), ["GET /api/real"]);
        assert!(s.problems.is_empty(), "{:?}", s.problems);
    }

    #[test]
    fn every_kind_of_test_only_item_is_skipped() {
        let s = scan_one(
            r#"
            pub fn a() { r().route("/api/a", get(h)); }
            #[cfg(test)] mod plain { fn t() { r().route("/api/t1", get(h)); } }
            #[cfg(test)]
            #[allow(dead_code)]
            pub(crate) fn only_in_tests() -> Router { Router::new().route("/api/t2", get(h)) }
            #[cfg(test)] impl Foo { fn f() { r().route("/api/t3", get(h)); } }
            #[cfg(test)] const C: [u8; 2] = [1, 2];
            pub fn b() { r().route("/api/b", post(h)); }
            "#,
        );
        assert_eq!(keys(&s), ["GET /api/a", "POST /api/b"]);
    }

    #[test]
    fn a_cfg_not_test_item_is_not_skipped() {
        let s = scan_one(
            r#"#[cfg(not(test))] fn a() { r().route("/api/a", get(h)); } #[cfg(all(unix, test))] fn b() { r().route("/api/b", get(h)); }"#,
        );
        assert_eq!(
            keys(&s),
            ["GET /api/a", "GET /api/b"],
            "only exactly cfg(test) is test code to this check"
        );
    }

    #[test]
    fn a_whole_file_declared_as_a_test_module_is_skipped() {
        let files = vec![
            ("control/mod.rs".to_string(), r#"pub fn router() { r().route("/api/mcp", post(h)); } #[cfg(test)] mod tests;"#.to_string()),
            ("control/tests.rs".to_string(), r#"fn t() { r().route("/api/only-in-tests", get(h)); }"#.to_string()),
            ("bridge.rs".to_string(), r#"pub fn f() { r().route("/api/bridge", get(h)); } #[cfg(test)] pub(crate) mod support;"#.to_string()),
            ("bridge/support.rs".to_string(), r#"fn t() { r().route("/api/bridge-support", get(h)); }"#.to_string()),
            ("other/tests.rs".to_string(), r#"fn t() { r().route("/api/not-declared-so-it-is-scanned", get(h)); }"#.to_string()),
        ];
        let s = scan(&files);
        let mut k = keys(&s);
        k.dedup();
        assert_eq!(
            k,
            [
                "GET /api/bridge",
                "GET /api/not-declared-so-it-is-scanned",
                "POST /api/mcp"
            ]
        );
    }

    #[test]
    fn a_file_that_is_test_code_from_its_first_line_is_skipped() {
        let s = scan_one("#![cfg(test)]\nfn t() { r().route(\"/api/t\", get(h)); }\n");
        assert!(s.found.is_empty());
    }

    #[test]
    fn braces_in_strings_comments_raw_strings_and_characters_do_not_end_a_test_module_early() {
        let s = scan_one(
            r##"
            #[cfg(test)]
            mod tests {
                const A: &str = "}}} unbalanced";
                const B: &str = r#"}"# ;
                const C: char = '}';
                const D: char = '{';
                // a comment with a } in it
                /* and a block } comment /* nested } */ still } */
                fn t<'a>(x: &'a str) -> &'a str { x }
                fn f() { r().route("/api/inside", get(h)); }
            }
            pub fn after() { r().route("/api/after", get(h)); }
            "##,
        );
        assert_eq!(keys(&s), ["GET /api/after"]);
    }

    #[test]
    fn a_route_split_over_lines_with_a_qualified_method_and_a_chain_is_read() {
        let s = scan_one(
            r#"
            fn r() -> Router {
                Router::new()
                    .route(
                        "/api/voice/voices",
                        get(voices_list).post(voice_add).layer(DefaultBodyLimit::max(1 << 20)),
                    )
                    .route("/api/services/:id", axum::routing::delete(remove))
                    .route("/api/x", put(|Json(b): Json<Value>| async move { post_body(b) }))
            }
            "#,
        );
        assert_eq!(
            keys(&s),
            [
                "DELETE /api/services/:id",
                "GET /api/voice/voices",
                "POST /api/voice/voices",
                "PUT /api/x"
            ]
        );
        assert!(s.problems.is_empty(), "{:?}", s.problems);
    }

    #[test]
    fn the_gateway_prefix_format_is_expanded_and_the_route_is_any() {
        let s = scan_one(
            r#"fn r() -> Router { Router::new().route(&format!("{GATEWAY_PREFIX}/*path"), any(forward).layer(L)) }"#,
        );
        assert_eq!(keys(&s), ["ANY /api/ai/engine/gateway/*path"]);
        assert!(s.problems.is_empty(), "{:?}", s.problems);
    }

    #[test]
    fn what_the_walker_cannot_follow_is_a_problem_and_never_a_silent_skip() {
        for src in [
            r#"fn r(p: &str) { Router::new().route(p, get(h)); }"#,
            r#"fn r() { Router::new().route(&format!("/api/{}", name), get(h)); }"#,
            r#"fn r() { Router::new().route("/api/x", on(MethodFilter::GET, h)); }"#,
            r#"fn r() { Router::new().route("/api/x", handler_router()); }"#,
            r#"fn r() { Router::new().route("/api/x", head(h)); }"#,
            r#"fn r() { Router::new().nest("/api/x", inner()); }"#,
            r#"fn r() { Router::new().fallback(handler); }"#,
            r#"fn r() { Router::new().route_service("/api/x", svc); }"#,
        ] {
            let s = scan_one(src);
            assert!(!s.problems.is_empty(), "no problem reported for: {src}");
        }
    }

    #[test]
    fn a_method_called_route_that_is_not_axums_is_ignored() {
        let s = scan_one(r##"fn f(w: W) { w.route(text); w.route(r#"{"op":"result"}"#); }"##);
        assert!(s.found.is_empty() && s.problems.is_empty(), "{s:?}");
    }

    #[test]
    fn the_lexer_reads_lifetimes_byte_strings_escapes_and_raw_identifiers() {
        let toks = tokenize(
            r##"fn f<'a>(x: &'a [u8]) { let _ = (b"a}b", b'{', '\'', '\\', '\u{7d}', r#type, br#"}"#, "a\"}"); }"##,
        );
        let braces: Vec<char> = toks
            .iter()
            .filter_map(|t| {
                if let Tok::Punct(c @ ('{' | '}')) = t.tok {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            braces,
            ['{', '}'],
            "only the function body's own braces are code: {toks:?}"
        );
    }

    #[test]
    fn line_numbers_follow_the_source() {
        let s =
            scan_one("fn a() {}\n\n\nfn r() { Router::new()\n  .route(\"/api/x\", get(h)); }\n");
        assert_eq!(s.found[0].line, 5);
    }

    #[test]
    fn the_not_on_main_router_list_is_a_fixed_shape() {
        let mut names: BTreeMap<&str, usize> = BTreeMap::new();
        for (file, _) in NOT_ON_MAIN_ROUTER {
            *names.entry(file).or_default() += 1;
        }
        assert_eq!(names.len(), 1);
        assert_eq!(names["voice/mod.rs"], 2);
    }
}
