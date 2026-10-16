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

/// A run of tokens that repeats without end, wherever it is: prose, reasoning, or a tool call's payload, where the
/// guards above do not look. The 27B, testing a game it had written, filled a tool call's list of actions with
/// `{"key":"ArrowRight"},` until it was stopped (a list no penalty on prose reaches, and nothing else ended it before
/// the reply's 64,000 tokens).
///
/// A run is the reply's last tokens being the same every `period` tokens, 64 at most. What a reply means to write
/// repeats too (a map's rows of zeros, a table's rules), so a run is let be until it is longer than such things are:
/// [`CycleGuard::limit`]. Past that, [`CycleGuard::over`] says of the token sampled next whether it is the run's
/// next, and the caller has the model pick another, which ends a list where its next item was the same again.
pub(crate) struct CycleGuard {
    /// The last [`CycleGuard::MOST`] tokens, the newest last.
    last: std::collections::VecDeque<u32>,
    /// For each period, how many of the newest tokens are each the token that period before it.
    run: [usize; CycleGuard::MOST + 1],
}

/// A run past its limit that the token sampled would go on: its period, its length with that token, and how far
/// past the limit that is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cycle {
    pub period: usize,
    pub span: usize,
    pub past: usize,
}

impl Default for CycleGuard {
    fn default() -> Self {
        Self { last: Default::default(), run: [0; Self::MOST + 1] }
    }
}

impl CycleGuard {
    /// The longest period watched.
    pub const MOST: usize = 64;

    /// The tokens a run of `period` may be before it is taken for one without end: 128 of its turns and 2,048 tokens
    /// at least, or 4,096 tokens of a run of one to three tokens. What a reply means to write repeats for that long:
    /// a CSV of 200 rows the same (six tokens a row: 1,200; at 512 tokens, the limit this had first, it was refused
    /// after 85 of them), a map's 41-token row 128 times over, `0,` in a map of 2,000 cells. A run without end (the
    /// 27B's `{"key":"ArrowRight"},` toward the reply's 64,000 tokens) is still turned within the first few thousand.
    pub fn limit(period: usize) -> usize {
        if period <= 3 { 4096 } else { (128 * period).max(2048) }
    }

    /// Whether a run is within `margin` tokens of its limit, so that [`Self::over`] may soon say so of a token: a
    /// greedy request's rows are then read back whole, as the model's pick alone (made on its GPU) leaves no other
    /// token to take, and the reply would be stopped where it could have been turned. `margin` covers the rows a
    /// check of drafts made before this was asked.
    pub fn near(&self, turned: usize, margin: usize) -> bool {
        let n = self.last.len();
        (1..=Self::MOST.min(n)).any(|period| {
            let limit = (Self::limit(period) >> turned.min(8)).max(8 * period);
            self.run[period] > 0 && self.run[period] + 1 + period + margin >= limit
        })
    }

    /// The token is the reply's next.
    pub fn push(&mut self, token: u32) {
        let n = self.last.len();
        for period in 1..=Self::MOST {
            self.run[period] = if period <= n && self.last[n - period] == token { self.run[period] + 1 } else { 0 };
        }
        if n == Self::MOST {
            self.last.pop_front();
        }
        self.last.push_back(token);
    }

    /// Whether `next` would go on a run that is at its limit or past it (the run's own period: one that is a run of
    /// a shorter period too is that period's, by its limit). `turned`: the times this reply was turned from a run
    /// already, each of which halves the limit (eight turns of the run at least): a model turned from a list's item
    /// that goes on with the same item spelt another way has shown what it is doing, and is not given as long again.
    pub fn over(&self, next: u32, turned: usize) -> Option<Cycle> {
        let n = self.last.len();
        (1..=Self::MOST.min(n)).find_map(|period| {
            if self.last[n - period] != next {
                return None;
            }
            let span = self.run[period] + 1 + period;
            let shorter = (1..period).any(|d| period % d == 0 && self.last[n - d] == next && self.run[d] + 1 + d >= span);
            let limit = (Self::limit(period) >> turned.min(8)).max(8 * period);
            (!shorter && span >= limit).then(|| Cycle { period, span, past: span - limit })
        })
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

    /// The tokens of `unit` over and over, pushed until [`CycleGuard::over`] says the next is one too many: how many
    /// were pushed, and what it said.
    fn until_over(guard: &mut CycleGuard, unit: &[u32], most: usize) -> Option<(usize, Cycle)> {
        for i in 0..most {
            let next = unit[i % unit.len()];
            if let Some(cycle) = guard.over(next, 0) {
                return Some((i, cycle));
            }
            guard.push(next);
        }
        None
    }

    #[test]
    fn a_list_item_repeated_without_end_is_a_cycle_after_its_limit() {
        // `{"key":"ArrowRight"},` as seven tokens, after a call's opening
        let mut guard = CycleGuard::default();
        for t in [900, 901, 902] {
            guard.push(t);
        }
        let (pushed, cycle) = until_over(&mut guard, &[1, 2, 3, 4, 5, 6, 7], 10_000).expect("a run without end");
        assert_eq!((cycle.period, cycle.span, cycle.past), (7, 2048, 0));
        assert_eq!(pushed, 2047, "the 2,048th token of the run is the one refused: 292 items written, not 64,000 tokens");
        // the model picks another token: the run is over, and what follows is judged anew
        guard.push(77);
        assert_eq!(guard.over(1, 0), None);
        assert!(until_over(&mut guard, &[1, 2, 3, 4, 5, 6, 7], 1600).is_none());
        // refused again and again where the caller lets it go on: how far past the limit is said
        let mut guard = CycleGuard::default();
        for i in 0..2100 {
            guard.push([1, 2, 3, 4, 5, 6, 7][i % 7]);
        }
        assert_eq!(guard.over([1, 2, 3, 4, 5, 6, 7][2100 % 7], 0).map(|c| (c.period, c.past)), Some((7, 53)));
        // a reply turned from a run before is given half as long at the next, and half again, down to eight turns
        // of the run and no less
        for (turned, limit) in [(1, 1024), (2, 512), (3, 256), (4, 128), (5, 64), (6, 56), (9, 56)] {
            let mut guard = CycleGuard::default();
            let mut written = 0;
            while guard.over([1, 2, 3, 4, 5, 6, 7][written % 7], turned).is_none() {
                guard.push([1, 2, 3, 4, 5, 6, 7][written % 7]);
                written += 1;
            }
            assert_eq!(written + 1, limit, "turned {turned} times");
        }
    }

    #[test]
    fn what_data_is_made_of_is_let_be_far_longer() {
        // a map's cells: `0` and `,`, 2,000 of them
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &[10, 11], 10_000).unwrap();
        assert_eq!((cycle.period, pushed), (2, 4095));
        // one token over and over (a rule of dashes, a run of spaces): its own period's limit, not four's
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &[5], 10_000).unwrap();
        assert_eq!((cycle.period, pushed), (1, 4095));
        // a map's row of twenty cells, forty tokens: 128 rows the same
        let row: Vec<u32> = (0..20).flat_map(|_| [10, 11]).chain([12]).collect();
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &row, 10_000).unwrap();
        assert_eq!((cycle.period, pushed), (41, 128 * 41 - 1));
        // a CSV's 200 rows the same, `0,ok,0` and its line end as six tokens (asked for through OAIY: refused after
        // 513 tokens at the limit this had first)
        let mut guard = CycleGuard::default();
        for t in [20, 21, 22, 23] {
            guard.push(t); // its header
        }
        assert!(until_over(&mut guard, &[30, 31, 32, 31, 30, 33], 200 * 6).is_none());
    }

    #[test]
    fn a_run_is_near_its_limit_a_margin_before_it_is_over() {
        let unit = [1, 2, 3, 4, 5, 6, 7];
        let mut guard = CycleGuard::default();
        let mut written = 0;
        while !guard.near(0, 16) {
            assert!(guard.over(unit[written % 7], 0).is_none(), "over before near, at {written}");
            guard.push(unit[written % 7]);
            written += 1;
        }
        // near 16 tokens before the 2,048th is refused, and near from there until it is
        assert_eq!(written, 2048 - 1 - 16);
        while guard.over(unit[written % 7], 0).is_none() {
            assert!(guard.near(0, 16));
            guard.push(unit[written % 7]);
            written += 1;
        }
        assert_eq!(written, 2047);
        // a turned reply's next run is near at its own, halved, limit; one that does not repeat never is
        let mut guard = CycleGuard::default();
        (0..1007).for_each(|i| guard.push(unit[i % 7]));
        assert!(guard.near(1, 16) && !guard.near(0, 16));
        let mut guard = CycleGuard::default();
        (0..10_000u32).for_each(|i| guard.push(i));
        assert!(!guard.near(0, 16));
    }

    #[test]
    fn a_reply_that_does_not_repeat_is_never_a_cycle() {
        let mut guard = CycleGuard::default();
        let mut x = 12345u32;
        for _ in 0..20_000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let next = (x >> 16) % 50;
            assert_eq!(guard.over(next, 0), None);
            guard.push(next);
        }
        // nor one whose repeats are longer than it watches (a verse sung again: the other guards' to judge)
        let verse: Vec<u32> = (0..80).collect();
        let mut guard = CycleGuard::default();
        assert!(until_over(&mut guard, &verse, 8_000).is_none());
    }

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
