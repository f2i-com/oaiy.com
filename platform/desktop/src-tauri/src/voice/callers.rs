//! The names callers are greeted by.
//!
//! Kept in `<data>/callers.json`, beside the voices folder, by the last nine
//! digits of the caller's number: +61 491 570 006 and 0491 570 006 are one
//! caller. A call's greeting says the first name when it is known.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

static FILE: OnceLock<PathBuf> = OnceLock::new();
/// One change at a time: the file is read, changed and written whole.
static CHANGING: Mutex<()> = Mutex::new(());

/// The longest name kept.
const MAX_NAME: usize = 80;

/// Where the names are: `<data>/callers.json`.
pub fn init(data_dir: &Path) {
    let _ = FILE.set(data_dir.join("callers.json"));
}

/// A number's key: its last nine digits, or None when it has too few digits to be a phone's.
fn key(number: &str) -> Option<String> {
    let digits: Vec<char> = number.chars().filter(char::is_ascii_digit).collect();
    if digits.len() < 8 {
        return None;
    }
    Some(digits[digits.len().saturating_sub(9)..].iter().collect())
}

fn read(file: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(file).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

/// The name kept for a caller's number.
pub fn name_of(number: &str) -> Option<String> {
    name_in(FILE.get()?, number)
}

fn name_in(file: &Path, number: &str) -> Option<String> {
    read(file).remove(&key(number)?).filter(|n| !n.is_empty())
}

/// Keep the name a caller is greeted by; an empty name forgets it. Answers with the name kept.
pub fn remember(number: &str, name: &str) -> Result<String, String> {
    let file = FILE.get().ok_or("the callers file is not set up")?;
    remember_in(file, number, name)
}

fn remember_in(file: &Path, number: &str, name: &str) -> Result<String, String> {
    let key = key(number).ok_or("give the caller's phone number, with at least 8 digits")?;
    let name: String = name.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_NAME).collect();
    let _one = CHANGING.lock().unwrap_or_else(|e| e.into_inner());
    let mut names = read(file);
    if name.is_empty() {
        names.remove(&key);
    } else {
        names.insert(key, name.clone());
    }
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&names).map_err(|e| e.to_string())?;
    std::fs::write(file, text).map_err(|e| e.to_string())?;
    Ok(name)
}

/// Whether the phone's name for a caller is a person's: not a number, and not "Unknown" or the like.
pub fn looks_like_name(name: &str) -> bool {
    let name = name.trim();
    let first = first_name(name).to_lowercase();
    name.chars().any(char::is_alphabetic)
        && !name.chars().any(|c| c.is_ascii_digit())
        && !matches!(first.as_str(), "unknown" | "private" | "anonymous" | "withheld" | "restricted" | "unavailable" | "blocked" | "no" | "caller" | "spam")
}

/// The first word of a name, without the punctuation around it.
fn first_name(name: &str) -> &str {
    name.split_whitespace().next().unwrap_or("").trim_matches(|c: char| !c.is_alphanumeric())
}

/// Whether `text` has `word` in it as a whole word, in any case.
fn has_word(text: &str, word: &str) -> bool {
    let word = word.to_lowercase();
    text.split(|c: char| !c.is_alphanumeric()).any(|w| w.to_lowercase() == word)
}

/// `text` without a leading `word` (in any case), when `text` starts with it as a whole word:
/// (the word as written, the rest).
fn strip_word<'a>(text: &'a str, word: &str) -> Option<(&'a str, &'a str)> {
    let head = text.get(..word.len())?;
    let rest = &text[word.len()..];
    (head.eq_ignore_ascii_case(word) && !rest.starts_with(char::is_alphanumeric)).then_some((head, rest))
}

/// The greeting, said to someone by their first name: kept as it is when it
/// has the name, the name put after a "Hi" or "Hello" it starts with, or else
/// "Hi Lance! " before it.
pub fn personal_greeting(greeting: &str, name: &str) -> String {
    let first = first_name(name);
    if first.is_empty() || greeting.trim().is_empty() || has_word(greeting, first) {
        return greeting.to_string();
    }
    let text = greeting.trim_start();
    for hello in ["hi", "hello", "hey", "g'day", "g\u{2019}day"] {
        if let Some((head, rest)) = strip_word(text, hello) {
            // "Hi there!" is "Hi Lance!", not "Hi Lance there!".
            let rest = rest.strip_prefix(' ').and_then(|r| strip_word(r, "there")).map_or(rest, |(_, after)| after);
            return format!("{head} {first}{rest}");
        }
    }
    format!("Hi {first}! {text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oaiy-desktop-callers-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("callers.json")
    }

    #[test]
    fn a_caller_is_known_by_the_last_nine_digits_of_their_number() {
        let f = file("digits");
        assert_eq!(remember_in(&f, "+61 491 570 006", "  Lance   Smith ").unwrap(), "Lance Smith");
        assert_eq!(name_in(&f, "0491570006").as_deref(), Some("Lance Smith"));
        assert_eq!(name_in(&f, "+61491570006").as_deref(), Some("Lance Smith"));
        assert_eq!(name_in(&f, "0491570157"), None);
        // Too short to be a phone's number: not kept, not found.
        assert!(remember_in(&f, "1234567", "Short").is_err());
        assert_eq!(name_in(&f, "1234567"), None);
        assert_eq!(name_in(&f, ""), None);
        // Eight digits is enough.
        remember_in(&f, "9876 5432", "Landline").unwrap();
        assert_eq!(name_in(&f, "(98) 765 432").as_deref(), Some("Landline"));
        let kept: BTreeMap<String, String> = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        assert_eq!(kept.get("491570006").map(String::as_str), Some("Lance Smith"));
        let _ = std::fs::remove_dir_all(f.parent().unwrap());
    }

    #[test]
    fn an_empty_name_forgets_the_caller() {
        let f = file("forget");
        remember_in(&f, "0491570006", "Lance").unwrap();
        remember_in(&f, "0400000001", "Sam").unwrap();
        assert_eq!(remember_in(&f, "+61491570006", "  ").unwrap(), "");
        assert_eq!(name_in(&f, "0491570006"), None);
        assert_eq!(name_in(&f, "0400000001").as_deref(), Some("Sam"), "the others are kept");
        // Renamed: the new name replaces the old.
        remember_in(&f, "0400000001", "Samantha").unwrap();
        assert_eq!(name_in(&f, "+61400000001").as_deref(), Some("Samantha"));
        let _ = std::fs::remove_dir_all(f.parent().unwrap());
    }

    #[test]
    fn a_greeting_says_the_first_name() {
        assert_eq!(personal_greeting("Hi! Thanks for calling", "Lance Smith"), "Hi Lance! Thanks for calling");
        assert_eq!(personal_greeting("Hello, thanks for calling", "Lance"), "Hello Lance, thanks for calling");
        assert_eq!(personal_greeting("hey there, how can I help?", "Lance"), "hey Lance, how can I help?");
        assert_eq!(personal_greeting("G'day! OAIY here.", "Lance"), "G'day Lance! OAIY here.");
        assert_eq!(personal_greeting("Hi", "Lance"), "Hi Lance");
        assert_eq!(personal_greeting("Thanks for calling OAIY.", "Lance"), "Hi Lance! Thanks for calling OAIY.");
        // "Hiya" and "Hello" in the middle are not a "Hi" it starts with.
        assert_eq!(personal_greeting("Hiya, OAIY here.", "Lance"), "Hi Lance! Hiya, OAIY here.");
        assert_eq!(personal_greeting("Thanks for calling, hello!", "Lance"), "Hi Lance! Thanks for calling, hello!");
        // Already said: kept as it is.
        assert_eq!(personal_greeting("Hi Lance, thanks for calling", "Lance Smith"), "Hi Lance, thanks for calling");
        assert_eq!(personal_greeting("Welcome back, lance!", "Lance"), "Welcome back, lance!");
        // No name, or no greeting: nothing changes.
        assert_eq!(personal_greeting("Hi! Thanks for calling", ""), "Hi! Thanks for calling");
        assert_eq!(personal_greeting("", "Lance"), "");
    }

    #[test]
    fn a_name_from_the_phone_is_a_name_only_when_it_looks_like_one() {
        assert!(looks_like_name("Lance Smith"));
        assert!(looks_like_name("Zoë"));
        assert!(!looks_like_name("+61491570006"));
        assert!(!looks_like_name("0491 570 006"));
        assert!(!looks_like_name(""));
        assert!(!looks_like_name("Unknown"));
        assert!(!looks_like_name("No caller ID"));
        assert!(!looks_like_name("Private number"));
    }
}
