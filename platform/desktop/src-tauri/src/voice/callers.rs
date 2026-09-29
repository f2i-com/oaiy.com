//! The names callers are greeted by.
//!
//! The names are kept with the contacts ([`super::contacts`], in
//! `<data>/callers.json`), by the last nine digits of the caller's number:
//! +61 491 570 006 and 0491 570 006 are one caller. A call's greeting says the
//! first name when it is known.

/// The name kept for a caller's number.
pub fn name_of(number: &str) -> Option<String> {
    super::contacts::name_of(number)
}

/// The receptionist's name for a caller (`PUT /api/voice/callers`): kept
/// unless the person named them in Contacts, whose name then stays; an empty
/// name clears only the receptionist's. Answers with the name kept.
pub fn remember(number: &str, name: &str) -> Result<String, String> {
    super::contacts::shared().remember_name(number, name).map_err(|e| e.message)
}

/// Whether the phone's name for a caller is a person's: not a number, and not
/// "Unknown" or the like. One word is a name ("Lance").
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

    #[test]
    fn a_greeting_says_the_first_name() {
        assert_eq!(personal_greeting("Hi! Thanks for calling", "Liam Smith"), "Hi Liam! Thanks for calling");
        assert_eq!(personal_greeting("Hello, thanks for calling", "Liam"), "Hello Liam, thanks for calling");
        assert_eq!(personal_greeting("hey there, how can I help?", "Liam"), "hey Liam, how can I help?");
        assert_eq!(personal_greeting("G'day! OAIY here.", "Liam"), "G'day Liam! OAIY here.");
        assert_eq!(personal_greeting("Hi", "Liam"), "Hi Liam");
        assert_eq!(personal_greeting("Thanks for calling OAIY.", "Liam"), "Hi Liam! Thanks for calling OAIY.");
        // "Hiya" and "Hello" in the middle are not a "Hi" it starts with.
        assert_eq!(personal_greeting("Hiya, OAIY here.", "Liam"), "Hi Liam! Hiya, OAIY here.");
        assert_eq!(personal_greeting("Thanks for calling, hello!", "Liam"), "Hi Liam! Thanks for calling, hello!");
        // Already said: kept as it is.
        assert_eq!(personal_greeting("Hi Liam, thanks for calling", "Liam Smith"), "Hi Liam, thanks for calling");
        assert_eq!(personal_greeting("Welcome back, liam!", "Liam"), "Welcome back, liam!");
        // No name, or no greeting: nothing changes.
        assert_eq!(personal_greeting("Hi! Thanks for calling", ""), "Hi! Thanks for calling");
        assert_eq!(personal_greeting("", "Liam"), "");
    }

    #[test]
    fn a_name_from_the_phone_is_a_name_only_when_it_looks_like_one() {
        assert!(looks_like_name("Liam Smith"));
        // One word is a whole name.
        assert!(looks_like_name("Liam"));
        assert!(looks_like_name("liam"));
        assert!(looks_like_name("Zoë"));
        assert!(!looks_like_name("+61491570006"));
        assert!(!looks_like_name("0491 570 006"));
        assert!(!looks_like_name(""));
        assert!(!looks_like_name("Unknown"));
        assert!(!looks_like_name("No caller ID"));
        assert!(!looks_like_name("Private number"));
    }
}
