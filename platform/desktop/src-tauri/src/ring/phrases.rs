//! Did the caller ask for a person? The receptionist's model says so when it asks to
//! transfer a call, but the model can be talked into saying anything, so what the
//! caller actually said is checked here, on the words this desktop heard and
//! transcribed itself.
//!
//! The rules are the design's Appendix A.9 (`docs/contracts/transfer/phrases.json`
//! has its sixteen known answers). Text is normalised (lower case, every run of
//! characters outside `a-z 0-9 '` becomes one space), and of the caller's last
//! three turns one must match a rule and none of the block patterns may match
//! that same turn. Two block patterns are this desktop's own, on top of the
//! reference (`phrases-oaiy.json`): "talk to you" (the caller is speaking to the
//! receptionist, not asking for someone else) and a refusal ("don't transfer me").
//!
//! This is a gate, not a promise: it lets the request through only when the caller's
//! words could be a request for a person; the limits, quiet hours and settings still decide.

use std::sync::OnceLock;

use regex::Regex;

/// How many of the caller's last turns are read.
pub const TURNS_READ: usize = 3;
/// The longest a turn is read to (the rest is dropped).
pub const TURN_CHARS: usize = 300;

const PERSON: &str = "(?:the |your |a |an )?(?:owner|manager|boss|proprietor|person|human|real person|actual person|someone|somebody|staff|member of staff|representative)";

fn rules() -> &'static [Regex] {
    static RULES: OnceLock<Vec<Regex>> = OnceLock::new();
    RULES.get_or_init(|| {
        [
            format!(r"\b(?:speak|talk|chat)(?:ing)? (?:to|with) {PERSON}\b"),
            r"\b(?:put|patch) me through\b".to_string(),
            r"\btransfer me\b".to_string(),
            format!(r"\b(?:get|find|fetch) (?:me )?{PERSON}\b"),
            r"\b(?:is|are) (?:the |your )?(?:owner|manager|boss|anyone|anybody|somebody|someone) (?:there|available|in|around|free)\b".to_string(),
            r"\b(?:real|actual) (?:person|human)\b".to_string(),
            r"\b(?:i )?(?:want|need|would like|d like|wanna) (?:to )?(?:speak|talk) (?:to|with)\b".to_string(),
        ]
        .iter()
        .map(|p| Regex::new(p).expect("a phrase rule is a valid pattern"))
        .collect()
    })
}

fn blocks() -> &'static [Regex] {
    static BLOCKS: OnceLock<Vec<Regex>> = OnceLock::new();
    BLOCKS.get_or_init(|| {
        [
            r"\b(?:my|our|his|her|their) (?:owner|manager|boss)\b",
            r"\bowner of\b",
            r"\btalk to you later\b",
            r"\bspeak to you (?:later|soon)\b",
            r"\bdon't (?:want|need) (?:to )?(?:speak|talk)\b",
            r"\bno need to (?:speak|talk|transfer)\b",
            // This desktop's own: they are talking to the receptionist ("I want to talk to you about Tuesday")...
            r"\b(?:speak|talk|chat)(?:ing)? (?:to|with) (?:you|u)\b",
            // ...or refusing ("don't transfer me", "do not put me through").
            r"\b(?:don't|do not|dont|never|not) (?:transfer|put|patch) me\b",
        ]
        .iter()
        .map(|p| Regex::new(p).expect("a block pattern is a valid pattern"))
        .collect()
    })
}

/// `text` as the rules read it: lower case, apostrophes straight, every run of anything but
/// `a-z 0-9 '` and space one space, and trimmed.
pub fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut gap = false;
    for c in text.chars() {
        let c = match c {
            '\u{2019}' | '\u{2018}' | '\u{02bc}' => '\'',
            c => c,
        };
        let lower = c.to_lowercase().next().unwrap_or(c);
        if lower.is_ascii_lowercase() || lower.is_ascii_digit() || lower == '\'' {
            if gap && !out.is_empty() {
                out.push(' ');
            }
            gap = false;
            out.push(lower);
        } else {
            gap = true;
        }
    }
    out
}

/// The turns that are read: the last [`TURNS_READ`], each cut to [`TURN_CHARS`].
pub fn recent<S: AsRef<str>>(turns: &[S]) -> Vec<String> {
    let start = turns.len().saturating_sub(TURNS_READ);
    turns[start..].iter().map(|t| t.as_ref().chars().take(TURN_CHARS).collect()).collect()
}

/// Whether one turn asks for a person.
fn turn_asks(turn: &str) -> bool {
    let text = normalise(turn);
    !text.is_empty() && !blocks().iter().any(|b| b.is_match(&text)) && rules().iter().any(|r| r.is_match(&text))
}

/// Whether the caller, in their last three turns, asked for the owner or a person.
pub fn caller_asked<S: AsRef<str>>(turns: &[S]) -> bool {
    recent(turns).iter().any(|t| turn_asks(t))
}

/// Whether the caller, in their last three turns, said one of the owner's urgent phrases (whole words).
pub fn urgent<S: AsRef<str>>(turns: &[S], phrases: &[String]) -> bool {
    let wanted: Vec<String> = phrases.iter().map(|p| normalise(p)).filter(|p| !p.is_empty()).collect();
    if wanted.is_empty() {
        return false;
    }
    recent(turns).iter().any(|turn| {
        let text = format!(" {} ", normalise(turn));
        wanted.iter().any(|p| text.contains(&format!(" {p} ")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const REFERENCE: &str = include_str!("../../../../../docs/contracts/transfer/phrases.json");
    const OAIY_EXTRA: &str = include_str!("../../../../../docs/contracts/transfer/phrases-oaiy.json");

    fn cases(file: &str, key: &str) -> Vec<Vec<String>> {
        let v: Value = serde_json::from_str(file).unwrap();
        v[key].as_array().unwrap().iter().map(|case| case["turns"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect()).collect()
    }

    #[test]
    fn the_reference_positives_are_requests_for_a_person() {
        let positives = cases(REFERENCE, "positive");
        assert_eq!(positives.len(), 8);
        for turns in positives {
            assert!(caller_asked(&turns), "{turns:?} asks for a person");
        }
    }

    #[test]
    fn the_reference_negatives_are_not() {
        let negatives = cases(REFERENCE, "negative");
        assert_eq!(negatives.len(), 8);
        for turns in negatives {
            assert!(!caller_asked(&turns), "{turns:?} does not ask for a person");
        }
    }

    #[test]
    fn this_desktops_own_cases() {
        for turns in cases(OAIY_EXTRA, "positive") {
            assert!(caller_asked(&turns), "{turns:?} asks for a person");
        }
        for turns in cases(OAIY_EXTRA, "negative") {
            assert!(!caller_asked(&turns), "{turns:?} does not");
        }
    }

    #[test]
    fn text_is_normalised_the_way_the_rules_read_it() {
        assert_eq!(normalise("  Can I SPEAK to the owner?!  "), "can i speak to the owner");
        assert_eq!(normalise("I\u{2019}d like a real person, please."), "i'd like a real person please");
        assert_eq!(normalise("Speak\u{2014}to\u{00a0}the\tmanager"), "speak to the manager");
        assert_eq!(normalise("...?!"), "");
        assert_eq!(normalise("caf\u{e9} 24/7"), "caf 24 7");
    }

    #[test]
    fn only_the_last_three_turns_count_and_each_is_cut() {
        let ask = "Can I speak to the manager?".to_string();
        let chatter = "What are your opening hours".to_string();
        assert!(caller_asked(&[ask.clone(), chatter.clone(), chatter.clone()]));
        assert!(!caller_asked(&[ask.clone(), chatter.clone(), chatter.clone(), chatter.clone()]), "the ask is four turns back");
        assert!(!caller_asked::<String>(&[]));
        // A turn is read to 300 characters: an ask that starts after them is not read.
        let long = format!("{} can I speak to the owner", "blah ".repeat(80));
        assert!(!caller_asked(&[long]));
    }

    #[test]
    fn a_block_pattern_beats_a_rule_in_the_same_turn_only() {
        // The refusal is in the same turn as the words that would ask.
        assert!(!caller_asked(&["No need to speak to anyone, transfer me if you must".to_string()]));
        assert!(!caller_asked(&["My manager said speak to the owner".to_string()]));
        // A later turn that asks is not undone by an earlier one that refused.
        assert!(caller_asked(&["No need to transfer me".to_string(), "Actually, put me through to the owner".to_string()]));
    }

    #[test]
    fn urgent_phrases_are_the_owners_own_and_whole() {
        let phrases = vec!["gas leak".to_string(), "burst pipe".to_string()];
        assert!(urgent(&["There's a gas leak!".to_string()], &phrases));
        assert!(urgent(&["a BURST   pipe in the yard".to_string()], &phrases));
        assert!(!urgent(&["a gas leaky tap".to_string()], &phrases), "whole words");
        assert!(!urgent(&["There's a gas leak!".to_string()], &[]), "no phrases: nothing is urgent");
        assert!(!urgent(&["It is urgent".to_string()], &phrases), "the model's word for it is not the caller's");
    }
}
