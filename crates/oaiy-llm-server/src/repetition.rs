//! Bound degenerate prose/reasoning generation without inspecting user text.
pub(crate) const CODE: &str = "generation_repetition";

#[derive(Default)]
pub(crate) struct Guard { tokens: Vec<u32> }

impl Guard {
    /// Four short blocks, or two long blocks followed by 128 repeated tokens.
    /// Catch a third copy even if its ending changes. Retain at most 4224
    /// tokens to cover a 2048-token period. This is an exact-copy heuristic,
    /// not a semantic judge of reasoning progress. Tool payloads may legitimately
    /// contain repeated data;
    /// do not apply the heuristic there, or carry prose history across them.
    pub fn push(&mut self, token: u32, tool_payload: bool) -> bool {
        if tool_payload { self.tokens.clear(); return false; }
        self.tokens.push(token);
        if self.tokens.len() > 4224 { self.tokens.remove(0); }
        let n = self.tokens.len();
        for period in 48..=2048.min(n / 2) {
            let required = if period <= 256 { period * 4 } else { period * 2 + 128 };
            if required > n || self.tokens[n - 1] != self.tokens[n - 1 - period] {
                continue;
            }
            let tail = &self.tokens[n - required..];
            if tail[period..].iter().zip(tail.iter()).all(|(a,b)| a == b) { return true; }
        }
        false
    }
}

/// Tiny exact cycles such as repeated intentions to call a tool. Only used
/// inside reasoning: prose and code may legitimately repeat short sequences.
#[derive(Default)]
pub(crate) struct ShortReasoningGuard { tokens: Vec<u32> }
impl ShortReasoningGuard {
    pub fn push(&mut self, token: u32) -> bool {
        self.tokens.push(token);
        if self.tokens.len() > 376 { self.tokens.remove(0); }
        let n = self.tokens.len();
        for period in 2usize..=47 {
            let required = period * 8.max(64usize.div_ceil(period));
            if required > n { continue; }
            let tail = &self.tokens[n - required..];
            if tail[period..].iter().zip(tail.iter()).all(|(a,b)| a == b) { return true; }
        }
        false
    }
}


/// Compare completed prose sentences, not token IDs: equivalent visible text
/// can be sampled with different token boundaries. Code spans/fences are exempt.
/// This deliberately catches exact loops, not paraphrases or factual mistakes.
#[derive(Default)]
pub(crate) struct ProseGuard {
    sentence: String,
    previous: String,
    copies: usize,
    overflow: bool,
    code: Option<(char, usize)>,
    run: Option<(char, usize)>,
    escaped: bool,
}
impl ProseGuard {
    pub fn clear(&mut self) { *self = Self::default(); }
    fn clear_sentences(&mut self) {
        self.sentence.clear(); self.previous.clear(); self.copies = 0; self.overflow = false;
    }
    pub fn push(&mut self, text: &str) -> bool {
        for ch in text.chars() {
            if self.escaped { self.escaped = false; continue; }
            if matches!(ch, '`' | '~') {
                self.clear_sentences();
                if self.run.is_some_and(|(old, _)| old != ch) { self.finish_run(); }
                self.run.get_or_insert((ch, 0)).1 += 1;
                continue;
            }
            self.finish_run();
            if self.code.is_some() { continue; }
            if ch == '\\' { self.escaped = true; self.clear_sentences(); continue; }
            if ch.is_whitespace() {
                if !self.sentence.is_empty() && !self.sentence.ends_with(' ') { self.sentence.push(' '); }
            } else if !self.overflow {
                self.sentence.push(ch);
            }
            if self.sentence.len() > 2048 {
                self.clear_sentences(); self.overflow = true;
            }
            if matches!(ch, '.' | '!' | '?') {
                let sentence = self.sentence.trim();
                if !self.overflow && sentence.len() >= 24 && sentence.chars().filter(|c| c.is_alphabetic()).count() >= 16 {
                    if sentence == self.previous { self.copies += 1; }
                    else { self.previous = sentence.to_owned(); self.copies = 1; }
                } else { self.previous.clear(); self.copies = 0; }
                self.sentence.clear(); self.overflow = false;
                if self.copies >= 4 { return true; }
            }
        }
        false
    }
    fn finish_run(&mut self) {
        if let Some((ch, n)) = self.run.take() {
            match self.code {
                None if ch == '`' || n >= 3 => self.code = Some((ch, n)),
                Some((old, len)) if old == ch && (n == len || len >= 3 && n >= len) => self.code = None,
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires OAIY_LOOP_EVENTS; replays local saved deltas without model inference"]
    fn recorded_prose_loop_is_stopped() {
        let path = std::env::var("OAIY_LOOP_EVENTS").expect("set OAIY_LOOP_EVENTS");
        let events = std::fs::read_to_string(path).unwrap();
        let mut guard = ProseGuard::default();
        for (line, event) in events.lines().enumerate() {
            let value = oaiy_engine::json::Json::parse(event.as_bytes()).unwrap();
            if value.get("UserMessageAppended").is_some() { guard.clear(); }
            if let Some(delta) = value.get("AssistantDelta") {
                if delta.get("kind").and_then(oaiy_engine::json::Json::as_str) == Some("text") {
                    if guard.push(delta.get("text").and_then(oaiy_engine::json::Json::as_str).unwrap()) {
                        println!("recorded prose loop stopped at event line {}", line + 1);
                        return;
                    }
                } else { guard.clear(); }
            }
        }
        panic!("recording did not trigger the prose guard");
    }

    #[test]
    fn prose_loop_is_independent_of_stream_chunk_boundaries() {
        let sentence = "Let me inspect the workspace first, then build the machine in one chosen folder.";
        for chunk in [1, 3, 13, 1000] {
            let text = sentence.repeat(4);
            let mut g = ProseGuard::default();
            let mut hit = false;
            for piece in text.as_bytes().chunks(chunk) { hit |= g.push(std::str::from_utf8(piece).unwrap()); }
            assert!(hit, "chunk size {chunk}");
        }
        let mut g = ProseGuard::default();
        for _ in 0..3 { assert!(!g.push(sentence)); }
        assert!(g.push("Let me inspect the workspace first,\n then build the machine in one chosen folder."));
    }
    #[test]
    fn prose_guard_preserves_code_progress_and_channel_boundaries() {
        let sentence = "Let me inspect the workspace first, then build the machine in one chosen folder.";
        for delimiter in ["`", "```", "~~~"] {
            let mut g = ProseGuard::default();
            let text = format!("{delimiter}\n{}{delimiter}\n", sentence.repeat(8));
            for ch in text.chars() { assert!(!g.push(&ch.to_string())); }
            assert!(g.push(&sentence.repeat(4)));
        }
        let mut g = ProseGuard::default();
        for i in 0..100 { assert!(!g.push(&format!("I finished workspace inspection number {i}."))); }
        assert!(!g.push(&sentence.repeat(3)));
        g.clear();
        assert!(!g.push(sentence));
        assert!(!g.push(&"x".repeat(10000)));
        assert!(g.sentence.len() <= 2048);
    }

    #[test]
    fn short_thought_cycles_need_sustained_repetition() {
        for period in [2usize, 7, 13, 47] {
            let mut g = ShortReasoningGuard::default();
            let required = period * 8.max(64usize.div_ceil(period));
            for i in 0..required - 1 { assert!(!g.push((i % period) as u32)); }
            assert!(g.push(((required - 1) % period) as u32));
        }
        let mut g = ShortReasoningGuard::default();
        for i in 0..2000 { assert!(!g.push(i)); }
        assert_eq!(g.tokens.len(), 376);
    }

    #[test]
    fn four_cycles_stop_but_progress_and_three_cycles_do_not() {
        let mut g = Guard::default();
        for _ in 0..3 { for token in 0..61 { assert!(!g.push(token, false)); } }
        for token in 0..60 { assert!(!g.push(token, false)); }
        assert!(g.push(60, false));
        let mut progressing = Guard::default();
        for token in 0..12000 { assert!(!progressing.push(token, false)); }
        assert_eq!(progressing.tokens.len(), 4224);
    }
    #[test]
    fn repeated_long_plans_and_medium_blocks_stop() {
        for period in [257, 511, 512, 617, 2048] {
            let mut g = Guard::default();
            for i in 0..period * 2 + 127 { assert!(!g.push(i % period, false)); }
            assert!(g.push(127, false), "period {period}");
        }
    }
    #[test]
    fn similar_plans_with_new_evidence_are_not_exact_loops() {
        let mut g = Guard::default();
        for revision in 0..6 {
            for i in 0..617 {
                let token = if i % 100 == 0 { 10000 + revision } else { i };
                assert!(!g.push(token, false));
            }
        }
    }
    #[test]
    fn long_tool_payload_clears_prose_history() {
        let mut g = Guard::default();
        for token in 0..617 { assert!(!g.push(token, false)); }
        for token in 0..10000 { assert!(!g.push(token % 617, true)); }
        for token in 0..617 { assert!(!g.push(token, false)); }
    }
    #[test]
    fn repetitive_tool_payloads_are_not_truncated() {
        let mut g = Guard::default();
        for _ in 0..3000 { assert!(!g.push(7, true)); }
        for token in 0..180 { assert!(!g.push(token % 60, false)); }
        assert!(!g.push(7, true));
        for token in 0..180 { assert!(!g.push(token % 60, false)); }
    }
}
