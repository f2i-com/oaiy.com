#![cfg(test)]
//! Reading the crate's own sources in a test without reading its test code.
//!
//! Several tests guard a property of the code that runs by reading the source files (every name a store
//! is kept under is classified for the backup, a converted lane puts no credential on a request itself,
//! no lane builds a client of its own). Each used to cut the file at its first `#[cfg(test)]`, or at the
//! exact text `#[cfg(test)]\nmod tests`. Both are wrong the same way: a `#[cfg(test)]` on a `mod x;` that
//! lives in another file, or on one helper, or on a test module in the middle of the file, ends the
//! "production code" there, and every line after it is never looked at (about 550 lines of `link/mod.rs`
//! were hidden by `#[cfg(test)] pub(crate) mod testkit;` on line 34).
//!
//! [`production_code`] removes the test ITEMS instead: whatever follows a `#[cfg(test)]` attribute up to
//! the end of the item it is on, found with a small lexer that knows comments (nested), strings (and raw
//! strings), characters and lifetimes, so that a brace in any of them does not end an item early. What is
//! removed is blanked, not deleted: the newlines stay, so a line number in the result is the line number
//! in the file.
//!
//! Only the attribute `cfg(test)` itself counts. `cfg(not(test))` is production code, and so is
//! `cfg(all(test, windows))`: a scan that sees a little too much finds a thing; one that sees too little
//! finds nothing.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Space,
    Comment,
    Str,
    Char,
    Lifetime,
    Ident,
    Punct,
}

struct Tok {
    kind: Kind,
    start: usize,
    end: usize,
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The tokens of a Rust source, with the text of comments, strings and characters each in one token.
fn tokenize(b: &[char]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        let c = b[i];
        let kind = if c.is_whitespace() {
            while i < b.len() && b[i].is_whitespace() {
                i += 1;
            }
            Kind::Space
        } else if c == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            Kind::Comment
        } else if c == '/' && b.get(i + 1) == Some(&'*') {
            let mut depth = 1;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            Kind::Comment
        } else if c == '"' {
            i = end_of_string(b, i);
            Kind::Str
        } else if c == '\'' {
            if b.get(i + 1) == Some(&'\\') {
                // A character with an escape: `'\n'`, `'\''`, `'\u{1F600}'`.
                i += 3;
                while i < b.len() && b[i] != '\'' {
                    i += 1;
                }
                i += 1;
                Kind::Char
            } else if b.get(i + 2) == Some(&'\'') {
                i += 3;
                Kind::Char
            } else {
                i += 1;
                while i < b.len() && is_ident_char(b[i]) {
                    i += 1;
                }
                Kind::Lifetime
            }
        } else if is_ident_start(c) {
            while i < b.len() && is_ident_char(b[i]) {
                i += 1;
            }
            let word: String = b[start..i].iter().collect();
            // `r"..."`, `r#"..."#`, `br#"..."#`, `cr"..."`: a raw string, whose end is a quote and as many hashes.
            let mut hashes = 0;
            while matches!(word.as_str(), "r" | "br" | "cr") && b.get(i + hashes) == Some(&'#') {
                hashes += 1;
            }
            if matches!(word.as_str(), "r" | "br" | "cr") && b.get(i + hashes) == Some(&'"') {
                i = end_of_raw_string(b, i + hashes, hashes);
                Kind::Str
            } else {
                Kind::Ident
            }
        } else if c.is_ascii_digit() {
            while i < b.len() && (is_ident_char(b[i]) || b[i] == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
                i += 1;
            }
            Kind::Ident
        } else {
            i += 1;
            Kind::Punct
        };
        out.push(Tok { kind, start, end: i });
    }
    out
}

/// The index after the string that starts with the quote at `i`.
fn end_of_string(b: &[char], i: usize) -> usize {
    let mut j = i + 1;
    while j < b.len() && b[j] != '"' {
        j += if b[j] == '\\' { 2 } else { 1 };
    }
    (j + 1).min(b.len())
}

/// The index after the raw string whose opening quote is at `quote` and which opened with `hashes` hashes.
fn end_of_raw_string(b: &[char], quote: usize, hashes: usize) -> usize {
    let mut j = quote + 1;
    while j < b.len() {
        if b[j] == '"' && (0..hashes).all(|k| b.get(j + 1 + k) == Some(&'#')) {
            return j + 1 + hashes;
        }
        j += 1;
    }
    b.len()
}

/// The tokens that are neither space nor comment, as indexes into `toks`.
struct Sig<'a> {
    b: &'a [char],
    toks: &'a [Tok],
    at: Vec<usize>,
}

impl<'a> Sig<'a> {
    fn new(b: &'a [char], toks: &'a [Tok]) -> Self {
        let at = (0..toks.len()).filter(|&i| !matches!(toks[i].kind, Kind::Space | Kind::Comment)).collect();
        Self { b, toks, at }
    }

    fn len(&self) -> usize {
        self.at.len()
    }

    fn tok(&self, k: usize) -> &Tok {
        &self.toks[self.at[k]]
    }

    fn punct(&self, k: usize) -> Option<char> {
        if k >= self.len() {
            return None;
        }
        let t = self.tok(k);
        (t.kind == Kind::Punct).then(|| self.b[t.start])
    }

    fn is_punct(&self, k: usize, c: char) -> bool {
        k < self.len() && self.punct(k) == Some(c)
    }

    fn word(&self, k: usize) -> Option<String> {
        if k >= self.len() {
            return None;
        }
        let t = self.tok(k);
        (t.kind == Kind::Ident).then(|| self.b[t.start..t.end].iter().collect())
    }

    fn is_word(&self, k: usize, w: &str) -> bool {
        self.word(k).as_deref() == Some(w)
    }

    /// The index of the token that closes the bracket opened at `k` (`(`, `[` or `{`), or the last token.
    fn closing(&self, k: usize) -> usize {
        let mut depth = 0i32;
        for j in k..self.len() {
            match self.punct(j) {
                Some('(' | '[' | '{') => depth += 1,
                Some(')' | ']' | '}') => {
                    depth -= 1;
                    if depth == 0 {
                        return j;
                    }
                }
                _ => {}
            }
        }
        self.len().saturating_sub(1)
    }

    /// If an attribute (`#[...]` or `#![...]`) starts at `k`: whether it is an inner one, and the index of its `]`.
    fn attribute(&self, k: usize) -> Option<(bool, usize)> {
        if !self.is_punct(k, '#') {
            return None;
        }
        let inner = self.is_punct(k + 1, '!');
        let open = k + 1 + usize::from(inner);
        self.is_punct(open, '[').then(|| (inner, self.closing(open)))
    }

    /// Whether the attribute that ends at `close` (and starts at `k`) is exactly `cfg(test)`.
    fn is_cfg_test(&self, k: usize, close: usize) -> bool {
        let open = k + 1 + usize::from(self.is_punct(k + 1, '!'));
        close == open + 5
            && self.is_word(open + 1, "cfg")
            && self.is_punct(open + 2, '(')
            && self.is_word(open + 3, "test")
            && self.is_punct(open + 4, ')')
    }

    /// Past the attributes that start at `k`: the index of the first token that is not one.
    fn past_attributes(&self, mut k: usize) -> usize {
        while let Some((_, close)) = self.attribute(k) {
            k = close + 1;
        }
        k
    }

    /// Past the modifiers of the item that starts at `k` (`pub`, `pub(crate)`, `unsafe`, `async`, `default`,
    /// `extern "C"`, the `const` of a `const fn`), to the word that says what it is.
    fn past_modifiers(&self, k: usize) -> usize {
        let mut w = k;
        loop {
            if self.is_word(w, "pub") {
                w += 1;
                if self.is_punct(w, '(') {
                    w = self.closing(w) + 1;
                }
            } else if ["unsafe", "async", "default", "move"].iter().any(|m| self.is_word(w, m)) {
                w += 1;
            } else if self.is_word(w, "extern") && self.tok_is_str(w + 1) {
                w += 2;
            } else if self.is_word(w, "const") && ["fn", "unsafe", "async", "extern"].iter().any(|m| self.is_word(w + 1, m)) {
                w += 1;
            } else {
                return w;
            }
        }
    }

    /// The name of the module if the item that starts at `k` is `mod name;`, which lives in a file of its own.
    fn declared_module(&self, k: usize) -> Option<String> {
        let w = self.past_modifiers(k);
        (self.is_word(w, "mod") && self.is_punct(w + 2, ';')).then(|| self.word(w + 1)).flatten()
    }

    /// The index of the last token of the item, statement, field or arm that starts at `k` (after its
    /// attributes), or `k - 1` when there is nothing there (an attribute at the end of a scope).
    ///
    /// What ends it depends on what it is. An item with a body (`fn`, `impl`, `mod`...), a block, and a statement
    /// that is a braced construct (`if`, `match`, `loop`, `while`, `for`, with a label or not) end at the `}` that
    /// closes their body (an `if` at the end of its last `else`), with no `;` after it. A macro invoked with braces
    /// (`thread_local! { .. }`) ends at its `}` too. A match arm whose body is a block ends there. Everything else
    /// (`let`, `const`, a call, a field, a variant) ends at its `;` or `,`, outside any brackets.
    fn item_end(&self, k: usize) -> usize {
        let w = self.past_modifiers(k);
        let label = w + 2 <= self.len() && self.tok(w).kind == Kind::Lifetime && self.is_punct(w + 1, ':');
        let at = if label { w + 2 } else { w };
        if ["if", "match", "loop", "while", "for"].iter().any(|m| self.is_word(at, m)) {
            return self.brace_statement_end(at);
        }
        let block = self.is_punct(w, '{');
        let body_item = ["mod", "fn", "impl", "struct", "enum", "trait", "union", "macro_rules", "extern"].iter().any(|m| self.is_word(w, m));
        // `let`, `const`, `static`, `type` and `use` end at their `;` alone: a `,` in `HashMap<A, B>` or in `use a::{b, c}` is not their end.
        let semicolon_only = ["let", "const", "static", "type", "use"].iter().any(|m| self.is_word(w, m));

        let mut depth = 0i32;
        let mut j = k;
        while j < self.len() {
            match self.punct(j) {
                Some('{') if depth == 0 && (block || body_item) => return self.closing(j),
                // A macro invoked with braces is an item or a statement of its own: it ends at its `}`, with a `;` after it or not.
                Some('{') if depth == 0 && !semicolon_only && j > k && self.is_punct(j - 1, '!') => {
                    let close = self.closing(j);
                    return if self.is_punct(close + 1, ';') { close + 1 } else { close };
                }
                // A match arm whose body is a block ends at the block (and the comma after it, if there is one).
                Some('=') if depth == 0 && !semicolon_only && self.is_punct(j + 1, '>') && self.is_punct(j + 2, '{') => {
                    let close = self.closing(j + 2);
                    return if self.is_punct(close + 1, ',') { close + 1 } else { close };
                }
                Some('(' | '[' | '{') => depth += 1,
                Some(')' | ']' | '}') => {
                    if depth == 0 {
                        return j.wrapping_sub(1);
                    }
                    depth -= 1;
                }
                Some(';') if depth == 0 => return j,
                Some(',') if depth == 0 && !body_item && !semicolon_only && !block => return j,
                _ => {}
            }
            j += 1;
        }
        self.len() - 1
    }

    /// The last token of a statement that is a braced construct, which starts with the keyword at `at`: the `}` that closes
    /// its body, or the last of the `else` blocks of an `if`. A `{` in the header of an `if let`, a `while let` or a `for` is
    /// a struct pattern (`if let Foo { x } = y {`), not the body; one inside brackets is part of the expression.
    fn brace_statement_end(&self, at: usize) -> usize {
        let mut keyword = self.word(at).unwrap_or_default();
        let mut j = at + 1;
        loop {
            let mut depth = 0i32;
            let mut pattern = keyword == "for" || (matches!(keyword.as_str(), "if" | "while") && self.is_word(j, "let"));
            let body = loop {
                if j >= self.len() {
                    return self.len() - 1;
                }
                match self.punct(j) {
                    Some('(' | '[') => depth += 1,
                    Some('{') if depth > 0 => depth += 1,
                    Some('{') if pattern => {
                        j = self.closing(j);
                    }
                    Some('{') => break j,
                    Some(')' | ']' | '}') => {
                        if depth == 0 {
                            return j.wrapping_sub(1);
                        }
                        depth -= 1;
                    }
                    Some(';') if depth == 0 => return j,
                    Some('=') if depth == 0 && pattern && keyword != "for" && !self.is_punct(j + 1, '=') && !self.is_punct(j + 1, '>') && !self.is_punct(j.wrapping_sub(1), '=') => pattern = false,
                    _ if depth == 0 && pattern && keyword == "for" && self.is_word(j, "in") => pattern = false,
                    _ => {}
                }
                j += 1;
            };
            let close = self.closing(body);
            if keyword == "if" && self.is_word(close + 1, "else") {
                j = close + 2;
                keyword = if self.is_word(j, "if") {
                    j += 1;
                    "if".to_string()
                } else {
                    "else".to_string()
                };
                continue;
            }
            return close;
        }
    }
    fn tok_is_str(&self, k: usize) -> bool {
        k < self.len() && self.tok(k).kind == Kind::Str
    }
}

/// A source without its test code, and the modules it declares as `#[cfg(test)] mod name;`.
pub(crate) struct Stripped {
    /// The source with the test items blanked out (see the module's note): the same number of lines as the
    /// source, and the code that is left is on the line it was.
    pub code: String,
    /// The `name` of each `#[cfg(test)] mod name;`: a file of test code of its own.
    pub test_mods: Vec<String>,
}

pub(crate) fn strip_tests(source: &str) -> Stripped {
    let b: Vec<char> = source.chars().collect();
    let toks = tokenize(&b);
    let sig = Sig::new(&b, &toks);

    // A file that begins with `#![cfg(test)]` is a test module through and through.
    let mut k = 0;
    while let Some((true, close)) = sig.attribute(k) {
        if sig.is_cfg_test(k, close) {
            let blank = b.iter().map(|&c| if c == '\n' { '\n' } else { ' ' }).collect();
            return Stripped { code: blank, test_mods: Vec::new() };
        }
        k = close + 1;
    }

    let mut blank: Vec<(usize, usize)> = Vec::new();
    let mut test_mods = Vec::new();
    let mut k = 0;
    while k < sig.len() {
        match sig.attribute(k) {
            Some((false, close)) if sig.is_cfg_test(k, close) => {
                let first = sig.past_attributes(close + 1);
                test_mods.extend(sig.declared_module(first));
                let last = sig.item_end(first);
                let last = if last == usize::MAX || last < first { close } else { last };
                blank.push((sig.tok(k).start, sig.tok(last).end));
                k = last + 1;
            }
            _ => k += 1,
        }
    }

    let mut out = b.clone();
    for (from, to) in blank {
        for c in &mut out[from..to] {
            if *c != '\n' {
                *c = ' ';
            }
        }
    }
    Stripped { code: out.into_iter().collect(), test_mods }
}

/// `source` without its test code: [`strip_tests`]'s `code`.
pub(crate) fn production_code(source: &str) -> String {
    strip_tests(source).code
}

/// [`production_code`] without the comments either (doc comments too), for a guard on what the code DOES: a
/// comment may say "credential" or "Bearer" about a lane that applies none.
pub(crate) fn code_without_comments(source: &str) -> String {
    let code = production_code(source);
    let b: Vec<char> = code.chars().collect();
    let mut out = b.clone();
    for t in tokenize(&b).iter().filter(|t| t.kind == Kind::Comment) {
        for c in &mut out[t.start..t.end] {
            if *c != '\n' {
                *c = ' ';
            }
        }
    }
    out.into_iter().collect()
}

/// The files, among `sources` (a path under `src/` with `/` in it, and the text), that are test code from
/// their first line: those a `#[cfg(test)] mod name;` declares, and everything under them. A test helper kept
/// in a file of its own has `#[test]` functions and fixtures at its top level, which no attribute in the file
/// says are tests.
pub(crate) fn test_only_files(sources: &[(String, String)]) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for (path, text) in sources {
        let (dir, file) = path.rsplit_once('/').unwrap_or(("", path.as_str()));
        let stem = file.trim_end_matches(".rs");
        // The folder the modules a file declares are in: beside `mod.rs`, `lib.rs` and `main.rs`, else in a folder named for it.
        let modules_in = if matches!(stem, "mod" | "lib" | "main") {
            dir.to_string()
        } else if dir.is_empty() {
            stem.to_string()
        } else {
            format!("{dir}/{stem}")
        };
        for name in strip_tests(text).test_mods {
            let base = if modules_in.is_empty() { name.clone() } else { format!("{modules_in}/{name}") };
            out.insert(format!("{base}.rs"));
            out.insert(format!("{base}/mod.rs"));
            let under = format!("{base}/");
            out.extend(sources.iter().map(|(p, _)| p.clone()).filter(|p| p.starts_with(&under)));
        }
    }
    out.retain(|p| sources.iter().any(|(s, _)| s == p));
    out
}

/// How many lines of `text` have something in them.
pub(crate) fn lines_with_code(text: &str) -> usize {
    text.lines().filter(|l| !l.trim().is_empty()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What is left of `source`, with the blanks squeezed out so the assertions can say what stays.
    fn kept(source: &str) -> String {
        production_code(source).split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn a_test_module_in_the_middle_of_a_file_hides_only_itself() {
        let source = "fn before() {}\n#[cfg(test)]\nmod tests {\n    fn t() { let s = \"x\"; }\n}\nfn after() { let name = \"link/account.json\"; }\n";
        assert_eq!(kept(source), "fn before() {} fn after() { let name = \"link/account.json\"; }");
    }

    #[test]
    fn an_out_of_line_test_module_is_one_line_and_hides_nothing_after_it() {
        // `#[cfg(test)] pub(crate) mod testkit;` ended the "production code" of link/mod.rs on its 34th line.
        let source = "pub mod a;\n#[cfg(test)]\npub(crate) mod testkit;\n#[cfg(test)]\nmod more;\nconst X: &str = \"kept.json\";\nfn f() {}\n";
        assert_eq!(kept(source), "pub mod a; const X: &str = \"kept.json\"; fn f() {}");
    }

    #[test]
    fn one_function_one_impl_one_const_one_use_and_one_static_are_each_only_themselves() {
        let source = "fn a() {}\n#[cfg(test)]\nfn only_in_tests() { call(\"x.json\"); }\nfn b() {}\n#[cfg(test)]\nimpl S { fn t(&self) {} }\nfn c() {}\n#[cfg(test)]\nconst T: &[(&str, &str)] = &[(\"a\", \"b\")];\nfn d() {}\n#[cfg(test)]\nuse std::collections::{HashMap, HashSet};\nfn e() {}\n#[cfg(test)]\nstatic S: Mutex<Option<u8>> = Mutex::new(None);\nfn f() {}\n";
        assert_eq!(kept(source), "fn a() {} fn b() {} fn c() {} fn d() {} fn e() {} fn f() {}");
    }

    #[test]
    fn braces_in_strings_characters_comments_and_raw_strings_do_not_end_a_module_early() {
        let source = concat!(
            "fn before() {}\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    fn t() {\n",
            "        let a = \"}}}\"; let b = '}'; let c = '\\'';\n",
            "        /* } /* } */ } */\n",
            "        // }\n",
            "        let d = r#\"}\"#; let e = r##\"\"# }\"##; let lifetime: &'static str = \"\";\n",
            "        let f = b\"}\";\n",
            "    }\n",
            "}\n",
            "fn after() { \"link/x.json\"; }\n",
        );
        assert_eq!(kept(source), "fn before() {} fn after() { \"link/x.json\"; }");
    }

    #[test]
    fn nested_braces_and_a_nested_test_attribute_are_matched() {
        let source = "#[cfg(test)]\nmod tests {\n    #[cfg(test)]\n    fn inner() { if x { y { } } }\n    mod deeper { fn f() { {} } }\n}\nfn after() {}\n";
        assert_eq!(kept(source), "fn after() {}");
    }

    #[test]
    fn only_the_attribute_cfg_test_is_a_test_attribute() {
        // `not(test)` is production code; so is `all(test, windows)`, which a scan is better off seeing.
        let source = "#[cfg(not(test))]\nfn real() {}\n#[cfg(all(test, windows))]\nfn windows_tests() {}\n#[cfg(windows)]\nfn windows() {}\n#[cfg(test)]\nfn gone() {}\n";
        assert_eq!(kept(source), "#[cfg(not(test))] fn real() {} #[cfg(all(test, windows))] fn windows_tests() {} #[cfg(windows)] fn windows() {}");
    }

    #[test]
    fn other_attributes_comments_and_visibility_between_the_attribute_and_the_item_belong_to_it() {
        let source = "#[cfg(test)]\n#[allow(dead_code)]\n/// A helper.\n// and a note\npub(crate) fn helper() { let x = 1; }\nfn kept() {}\n#[cfg(test)]\nconst unsafe fn odd() {}\nfn kept_too() {}\n";
        assert_eq!(kept(source), "fn kept() {} fn kept_too() {}");
    }

    #[test]
    fn a_test_item_inside_an_impl_a_struct_or_a_function_body_is_blanked_alone() {
        let source = "impl S {\n    #[cfg(test)]\n    fn t(&self) {}\n    fn p(&self) {}\n}\nstruct A {\n    x: u8,\n    #[cfg(test)]\n    y: u8,\n    z: u8,\n}\nfn f() {\n    #[cfg(test)]\n    let t = 1;\n    #[cfg(test)]\n    { println!(\"}\"); }\n    let p = 2;\n}\n";
        assert_eq!(kept(source), "impl S { fn p(&self) {} } struct A { x: u8, z: u8, } fn f() { let p = 2; }");
    }

    #[test]
    fn a_file_that_is_a_test_module_from_its_first_line_is_all_test_code() {
        assert_eq!(kept("#![cfg(test)]\n//! Docs.\nfn all() { \"x.json\"; }\n"), "");
        assert_eq!(kept("//! Docs.\n#![allow(dead_code)]\n#![cfg(test)]\nfn all() {}\n"), "");
    }

    #[test]
    fn the_attribute_in_a_comment_or_a_string_is_nothing() {
        let source = "/// Not behind `#[cfg(test)]` on the module.\nfn a() { let s = \"#[cfg(test)] fn gone() {}\"; }\n// #[cfg(test)]\nfn b() {}\n";
        let code = production_code(source);
        assert!(code.contains("fn a()") && code.contains("fn b()"));
        assert!(code.contains("fn gone() {}"), "inside a string, so it is a string: {code}");
        assert_eq!(code, source, "nothing was blanked");
    }

    #[test]
    fn line_numbers_are_those_of_the_file() {
        let source = "fn a() {}\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn b() {}\n";
        let code = production_code(source);
        assert_eq!(code.lines().count(), source.lines().count());
        assert_eq!(code.lines().position(|l| l.contains("fn b()")), source.lines().position(|l| l.contains("fn b()")));
        assert_eq!(lines_with_code(&code), 2);
    }

    #[test]
    fn the_modules_declared_for_tests_are_named_and_a_file_of_theirs_is_all_test_code() {
        let stripped = strip_tests("pub mod a;\n#[cfg(test)]\npub(crate) mod testkit;\n#[cfg(test)] mod inline { fn t() {} }\n#[cfg(test)]\nmod routes_check;\n#[cfg(not(test))]\nmod real;\n");
        assert_eq!(stripped.test_mods, ["testkit", "routes_check"], "an inline module has no file of its own");

        let src = |p: &str, t: &str| (p.to_string(), t.to_string());
        let sources = vec![
            src("lib.rs", "pub mod link;\n#[cfg(test)]\nmod source_scan;\n"),
            src("source_scan.rs", "fn t() {}"),
            src("link/mod.rs", "#[cfg(test)]\npub(crate) mod testkit;\npub mod net;\n"),
            src("link/testkit.rs", "#[test] fn t() {}"),
            src("link/net.rs", "fn n() {}"),
            src("auth/mod.rs", "#[cfg(test)] mod route_coverage;\n#[cfg(test)] mod fixtures;\n"),
            src("auth/route_coverage.rs", "#[test] fn t() { \"/latest.json\"; }"),
            src("auth/route_coverage/deeper.rs", "fn d() {}"),
            src("auth/fixtures/mod.rs", "fn f() {}"),
            src("auth/fixtures/more.rs", "fn m() {}"),
            src("auth/real.rs", "fn r() {}"),
            src("ring/mod.rs", "pub mod settings;\n"),
            // A module declared in a file that is not `mod.rs` lives in a folder of that name.
            src("ring/limits.rs", "#[cfg(test)] mod checks;\n"),
            src("ring/limits/checks.rs", "fn c() {}"),
            src("ring/settings.rs", "fn s() {}"),
        ];
        let only: Vec<String> = test_only_files(&sources).into_iter().collect();
        assert_eq!(
            only,
            ["auth/fixtures/mod.rs", "auth/fixtures/more.rs", "auth/route_coverage.rs", "auth/route_coverage/deeper.rs", "link/testkit.rs", "ring/limits/checks.rs", "source_scan.rs"]
        );
    }

    #[test]
    fn comments_can_be_taken_out_too_and_a_string_with_slashes_in_it_is_not_one() {
        let source = "// Bearer in a comment\nfn f() { let url = \"https://x.test/a\"; /* credential */ g(url) } /// doc credential\n#[cfg(test)]\nmod tests { fn t() { \"credential\"; } }\n";
        let code = code_without_comments(source);
        assert!(!code.to_lowercase().contains("bearer") && !code.contains("credential"), "{code}");
        assert!(code.contains("\"https://x.test/a\"") && code.contains("g(url)"), "{code}");
        assert_eq!(code.lines().count(), source.lines().count());
    }

    /// What the reviewer of this scanner tried against it, one case to a line: (what it is, a source, what of it is code that must
    /// stay in the result, what of it is test code that must be gone). Each case that the first version got wrong is here: a braced
    /// macro item, and a braced statement (`if`, `match`, loop) or a block arm under `#[cfg(test)]`, which ran on to the next `;`
    /// and hid the production code after it.
    #[test]
    fn the_cases_that_tried_to_hide_production_code_from_the_scan_or_leave_test_code_in_it() {
        let cases: Vec<(&str, &str, Vec<&str>, Vec<&str>)> = vec![
        ("braced item macro, no semicolon", "#[cfg(test)]\nthread_local! { static X: u8 = 1; }\nfn real() { \"hidden.json\"; }\nconst C: u8 = 1;\n", vec!["fn real()", "hidden.json", "const C"], vec!["thread_local"]),
        ("lazy_static style braced macro", "#[cfg(test)]\nlazy_static! { static ref X: u8 = 1; }\nfn real() { \"hidden.json\"; }\nstatic S: u8 = 2;\n", vec!["fn real()", "hidden.json"], vec!["lazy_static"]),
        ("cfg(test) if statement then production statement", "fn f() {\n    #[cfg(test)]\n    if probe() { return; }\n    real_call(\"x.json\");\n    other();\n}\n", vec!["real_call", "other()"], vec!["probe"]),
        ("cfg(test) if then tail expression", "fn f() -> u8 {\n    #[cfg(test)]\n    if probe() { return 1; }\n    tail_expr()\n}\n", vec!["tail_expr()"], vec!["probe"]),
        ("cfg(test) if let then tail", "fn f() -> bool {\n    #[cfg(test)]\n    if let Some(on) = gate() { return on; }\n    snapshot().on()\n}\n", vec!["snapshot().on()"], vec!["gate()"]),
        ("cfg(test) match statement", "fn f() {\n    #[cfg(test)]\n    match x { _ => {} }\n    real_call();\n}\n", vec!["real_call()"], vec!["match x"]),
        ("cfg(test) loop / while / for", "fn f() {\n    #[cfg(test)]\n    loop { break; }\n    a_real();\n    #[cfg(test)]\n    for i in 0..3 { t(i); }\n    b_real();\n    #[cfg(test)]\n    while c() { }\n    c_real();\n}\n", vec!["a_real()", "b_real()", "c_real()"], vec!["loop", "for i", "while c"]),
        ("cfg(test) unsafe block then code", "fn f() {\n    #[cfg(test)]\n    unsafe { t(); }\n    real();\n}\n", vec!["real()"], vec!["unsafe"]),
        ("block arm without a comma", "fn f(v: u8) {\n    match v {\n        #[cfg(test)]\n        1 => { t() }\n        2 => { real_arm() }\n        _ => {}\n    }\n}\n", vec!["real_arm()"], vec!["t()"]),
        ("arm with a comma", "fn f(v: u8) {\n    match v {\n        #[cfg(test)]\n        1 => t(),\n        2 => real_arm(),\n        _ => {}\n    }\n}\n", vec!["real_arm()"], vec!["t()"]),
        ("cfg_attr(test) is not a test item", "#[cfg_attr(test, derive(Debug))]\nstruct S { a: u8 }\nfn after() {}\n", vec!["struct S", "fn after()"], vec![]),
        ("cfg(not(test)) kept", "#[cfg(not(test))]\nfn real() { \"x.json\"; }\n", vec!["fn real()", "x.json"], vec![]),
        ("cfg(all(test, x)) kept", "#[cfg(all(test, windows))]\nfn w() { \"w.json\"; }\n", vec!["w.json"], vec![]),
        ("cfg(any(test, x)) kept", "#[cfg(any(test, windows))]\nfn w() { \"w.json\"; }\n", vec!["w.json"], vec![]),
        ("cfg(test) with spaces and a comment inside", "#[ cfg ( /* c */ test ) ]\nfn gone() {}\nfn kept() {}\n", vec!["fn kept()"], vec!["fn gone"]),
        ("attribute in a doc comment", "/// #[cfg(test)]\nfn kept() {}\n/** #[cfg(test)] */\nfn kept2() {}\n", vec!["fn kept()", "fn kept2()"], vec![]),
        ("attribute in a string", "fn kept() { let s = \"#[cfg(test)] fn x() {}\"; }\nfn also() {}\n", vec!["fn also()"], vec![]),
        ("attribute in a raw string with hashes", "fn kept() { let s = r##\"#[cfg(test)]\nfn x() {\"##; }\nfn also() {}\n", vec!["fn also()"], vec![]),
        ("raw string with hashes in a test mod", "#[cfg(test)]\nmod t { fn f() { let a = r##\"} \"# }\"##; let b = br#\"}\"#; let c = b\"}\"; let d = c\"}\"; } }\nfn after() {}\n", vec!["fn after()"], vec!["mod t"]),
        ("char literals with brace and quote", "#[cfg(test)]\nmod t { fn f() { let a = '}'; let b = '\"'; let c = '\\''; let d = '\\u{7d}'; let e = b'}'; } }\nfn after() {}\n", vec!["fn after()"], vec!["mod t"]),
        ("lifetimes and labels in a test mod", "#[cfg(test)]\nmod t { fn f<'a>(x: &'a str) { 'outer: loop { break 'outer; } let c = 'a'; } }\nfn after() {}\n", vec!["fn after()"], vec!["mod t"]),
        ("lifetime in production before a char", "fn f<'a>(x: &'a str) -> char { 'z' }\n#[cfg(test)]\nfn t() {}\nfn after() {}\n", vec!["fn f<'a>", "fn after()"], vec!["fn t"]),
        ("nested block comment with brace", "#[cfg(test)]\nmod t { /* } /* } */ } */ fn f() {} }\nfn after() {}\n", vec!["fn after()"], vec!["mod t"]),
        ("line comment with brace and apostrophe", "#[cfg(test)]\nmod t { // don't } me\n fn f() {} }\nfn after() {}\n", vec!["fn after()"], vec!["mod t"]),
        ("stacked attributes", "#[cfg(test)]\n#[allow(dead_code)]\n#[derive(Debug)]\nstruct S;\nfn after() {}\n", vec!["fn after()"], vec!["struct S"]),
        ("cfg(test) then other attributes then item with generics and where", "#[cfg(test)]\nimpl<T: Clone> Foo<T> for Bar where T: Send, { fn f(&self) {} }\nfn after() {}\n", vec!["fn after()"], vec!["impl<T"]),
        ("cfg(test) const with array", "#[cfg(test)]\nconst A: [(&str, u8); 2] = [(\"a\", 1), (\"b\", 2)];\nfn after() {}\n", vec!["fn after()"], vec!["const A"]),
        ("cfg(test) macro_rules", "#[cfg(test)]\nmacro_rules! m { () => { 1 } }\nfn after() {}\n", vec!["fn after()"], vec!["macro_rules"]),
        ("cfg(test) paren macro item", "#[cfg(test)]\nfoo!(a, b);\nfn after() {}\n", vec!["fn after()"], vec!["foo!"]),
        ("cfg(test) extern crate / use group", "#[cfg(test)]\nextern crate foo;\n#[cfg(test)]\nuse a::{b, c};\nfn after() {}\n", vec!["fn after()"], vec!["extern", "use a"]),
        ("cfg(test) struct field with a generic comma", "struct S {\n    #[cfg(test)]\n    m: HashMap<String, u8>,\n    real: u8,\n}\n", vec!["real: u8"], vec!["HashMap"]),
        ("cfg(test) enum variant with data", "enum E {\n    #[cfg(test)]\n    A { x: u8 },\n    B,\n}\n", vec!["B,"], vec!["A {"]),
        ("cfg(test) trait method declaration", "trait T {\n    #[cfg(test)]\n    fn t(&self);\n    fn p(&self);\n}\n", vec!["fn p("], vec!["fn t("]),
        ("cfg(test) fn with a return type that has braces in an array length", "#[cfg(test)]\nfn f() -> [u8; { 3 }] { [0; 3] }\nfn after() {}\n", vec!["fn after()"], vec!["fn f()"]),
        ("cfg(test) fn with a const generic default in braces", "#[cfg(test)]\nfn f<const N: usize = { 3 }>() {}\nfn after() {}\n", vec!["fn after()"], vec![]),
        ("inner attribute later in the file", "fn a() {}\nmod m {\n    #![cfg(test)]\n    fn t() {}\n}\nfn after() {}\n", vec!["fn after()"], vec![]),
        ("a file that starts with an inner cfg(test)", "//! doc\n#![cfg(test)]\nfn t() { \"x.json\"; }\n", vec![], vec!["x.json"]),
        ("cfg(test) in front of a closing brace", "fn f() {\n    real();\n    #[cfg(test)]\n}\nfn after() {}\n", vec!["real()", "fn after()"], vec![]),
        ("cfg(test) on an expression statement with a try", "fn f() -> R {\n    #[cfg(test)]\n    inject()?;\n    real()?;\n    Ok(())\n}\n", vec!["real()?", "Ok(())"], vec!["inject"]),
        ("cfg(test) on a let with a closure block", "fn f() {\n    #[cfg(test)]\n    let t = || { 1 };\n    real();\n}\n", vec!["real()"], vec!["let t"]),
        ("cfg(test) on a statement macro with braces", "fn f() {\n    #[cfg(test)]\n    println! { \"x\" }\n    real();\n}\n", vec!["real()"], vec![]),
        ("cfg(test) mod with a string containing a closing brace and quotes", "#[cfg(test)]\nmod t { fn f() { let s = \"\\\"}\"; } }\nfn after() {}\n", vec!["fn after()"], vec!["mod t"]),
        ("crlf line endings", "fn a() {}\r\n#[cfg(test)]\r\nmod t {\r\n  fn f() {}\r\n}\r\nfn after() {}\r\n", vec!["fn after()"], vec!["mod t"]),
        ("non-ascii identifiers and chars", "#[cfg(test)]\nmod t { fn f() { let é = 'é'; let e = '😀'; } }\nfn après() {}\n", vec!["fn après()"], vec!["mod t"]),
        ("unterminated block comment at the end", "fn a() {}\n/* never closed\n#[cfg(test)]\nfn x() {}\n", vec!["fn a()"], vec![]),
        ("unterminated string at the end", "fn a() {}\nconst S: &str = \"never closed;\n#[cfg(test)]\nfn x() {}\n", vec!["fn a()"], vec![]),
        ("cfg(test) with nothing after it at end of file", "fn a() {}\n#[cfg(test)]\n", vec!["fn a()"], vec![]),
        ("cfg(test) mod declared with a path attribute", "#[cfg(test)]\n#[path = \"x.rs\"]\nmod tests;\nfn after() {}\n", vec!["fn after()"], vec!["mod tests"]),
        ("a doc comment after the cfg attribute", "#[cfg(test)]\n/// docs\nfn t() {}\nfn after() {}\n", vec!["fn after()"], vec!["fn t"]),
        ("cfg(test) pub(in crate::x) fn", "#[cfg(test)]\npub(in crate::x) fn t() { }\nfn after() {}\n", vec!["fn after()"], vec!["fn t"]),
        ("cfg(test) async fn / unsafe impl / extern fn", "#[cfg(test)]\nasync fn a() {}\n#[cfg(test)]\nunsafe impl Send for X {}\n#[cfg(test)]\nextern \"C\" fn c() {}\nfn after() {}\n", vec!["fn after()"], vec!["async", "unsafe impl", "extern"]),
        ("cfg(test) extern block", "#[cfg(test)]\nextern \"C\" { fn x(); }\nfn after() {}\n", vec!["fn after()"], vec!["extern"]),
        ("cfg(test) static mut with a block initialiser", "#[cfg(test)]\nstatic X: Mutex<u8> = Mutex::new({ 1 });\nfn after() {}\n", vec!["fn after()"], vec!["static X"]),
        ("cfg(test) union / trait / type alias with where", "#[cfg(test)]\nunion U { a: u8 }\n#[cfg(test)]\ntrait Tr: Sized { fn t(); }\nfn after() {}\n", vec!["fn after()"], vec!["union", "trait"]),
        ("cfg(test) impl in the middle of a generic production impl", "impl<T> Real<T> {\n    #[cfg(test)]\n    fn t(&self) -> Vec<(u8, u8)> { vec![] }\n    fn p(&self) {}\n}\n", vec!["fn p("], vec!["fn t("]),
        ];
        let mut wrong = Vec::new();
        for (name, source, visible, gone) in &cases {
            let out = kept(source);
            for v in visible {
                if !out.contains(v) {
                    wrong.push(format!("{name}: production text {v:?} is hidden; kept: {out}"));
                }
            }
            for g in gone {
                if out.contains(g) {
                    wrong.push(format!("{name}: test text {g:?} is left in; kept: {out}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{} of {} cases: {wrong:#?}", wrong.len(), cases.len());
    }
    #[test]
    fn a_comma_in_the_generics_of_a_test_const_static_type_or_let_does_not_end_it() {
        // The `,` of `HashMap<String, Vec<u8>>` is outside every bracket, and is not the end of what it is in: these end at their `;`.
        let source = "#[cfg(test)]\nconst M: HashMap<String, Vec<u8>> = HashMap::new();\nfn a() {}\n#[cfg(test)]\nstatic S: Mutex<HashMap<u8, u8>> = Mutex::new(HashMap::new());\nfn b() {}\n#[cfg(test)]\ntype T<A, B> = (A, B);\nfn c() {}\nfn f() {\n    #[cfg(test)]\n    let m: HashMap<String, u8> = HashMap::new();\n    real();\n}\n";
        assert_eq!(kept(source), "fn a() {} fn b() {} fn c() {} fn f() { real(); }");
        // A field has no `;`: it ends at its first comma outside a bracket, and what is left of its generics after that (`u8>,`) names nothing.
        let fields = kept("struct S {\n    #[cfg(test)]\n    m: HashMap<String, u8>,\n    real: u8,\n}\n");
        assert!(fields.contains("real: u8") && !fields.contains("HashMap"), "{fields}");
    }
    #[test]
    fn a_lifetime_is_not_the_start_of_a_character() {
        let source = "fn f<'a>(x: &'a str) -> &'a str { x }\n#[cfg(test)]\nmod tests { fn t<'a>() { let c = '{'; } }\nfn g() {}\n";
        assert_eq!(kept(source), "fn f<'a>(x: &'a str) -> &'a str { x } fn g() {}");
    }
}
