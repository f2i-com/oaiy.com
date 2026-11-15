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
/// A run is the reply's last tokens being the same every `period` tokens, 8,192 at most. What a reply means to write
/// repeats too (a map's rows of zeros, a table's rules), so a run is let be until it is longer than such things are:
/// [`CycleGuard::limit`]. Past that, [`CycleGuard::over`] says of the token sampled next whether it is the run's
/// next, and the caller has the model pick another, which ends a list where its next item was the same again. A
/// block longer than [`CycleGuard::LONG`] written again whole ([`Cycle::whole`]) is not turned from but stopped:
/// another token in the middle of a file's second copy would leave the copy in the file, broken. A short run that
/// opens more brackets than it closes (`f(f(f(`, [`CycleGuard::nested`]) has a limit of its own, far shorter: no data
/// is made of it. (The engine stops, rather than turns, any run in a tool call: what the call writes would keep it.)
pub(crate) struct CycleGuard {
    /// The last [`CycleGuard::MOST`] tokens, the newest last.
    last: std::collections::VecDeque<u32>,
    /// For each period, how many of the newest tokens are each the token that period before it.
    run: Vec<usize>,
}

/// A run past its limit that the token sampled would go on: its period, its length with that token, and how far
/// past the limit that is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cycle {
    pub period: usize,
    pub span: usize,
    pub past: usize,
}

impl Cycle {
    /// A block of more than [`CycleGuard::LONG`] tokens written again whole (a file, a call): stopped, not turned.
    pub fn whole(&self) -> bool {
        self.period > CycleGuard::LONG
    }
}

impl Default for CycleGuard {
    fn default() -> Self {
        Self { last: std::collections::VecDeque::with_capacity(Self::MOST), run: vec![0; Self::MOST + 1] }
    }
}

impl CycleGuard {
    /// The longest period watched: a file or a call written again whole, a few thousand tokens a turn. (A function or
    /// a test written again and again is a few hundred: a 33B writing code into a tool call at 0.6 did so until its
    /// reply's 8,192 tokens, where 64 was watched; with 1,024 watched, a file written again went on to the reply's
    /// limit as before.)
    pub const MOST: usize = 8192;

    /// The longest period [`Self::limit`] gives its limit alone; a longer one's is [`Self::limit_of`]'s.
    pub const SHORT: usize = 64;

    /// The longest period judged by its turns ([`Self::limit_of`]: eight of a turn of many tokens). A longer one of many
    /// tokens is a block written again whole, which nothing a reply means to write is: let be its second copy and
    /// [`Self::INTO`] tokens of a third.
    pub const LONG: usize = 1024;

    /// How far into a long block's third copy it is let go.
    pub const INTO: usize = 128;

    /// The turns a run that opens more brackets than it closes is let be ([`Self::nested`]), and 128 tokens at least.
    pub const NESTED: usize = 32;

    /// The tokens a run of `period` may be before it is taken for one without end: 128 of its turns and 2,048 tokens
    /// at least, or 4,096 tokens of a run of one to three tokens. What a reply means to write repeats for that long:
    /// a CSV of 200 rows the same (six tokens a row: 1,200; at 512 tokens, the limit this had first, it was refused
    /// after 85 of them), a map's 41-token row 128 times over, `0,` in a map of 2,000 cells. A run without end (the
    /// 27B's `{"key":"ArrowRight"},` toward the reply's 64,000 tokens) is still turned within the first few thousand.
    pub fn limit(period: usize) -> usize {
        if period <= 3 { 4096 } else { (128 * period).max(2048) }
    }

    /// The limit of a run of `period` whose turn is `turn`: [`Self::limit`] up to [`Self::SHORT`] tokens a turn; past
    /// that, by what the turn is made of. A few tokens over and over (a map's row of cells, `0` and `,`) is data, let be
    /// 32 turns and 4,096 tokens at least; a turn of many (a function written again) is a run without end after eight
    /// turns and 1,024 tokens: no code means the same block of 65 tokens or more eight times in a row. A turn of many
    /// longer than [`Self::LONG`] (a file written again whole) is one after its second copy and [`Self::INTO`] tokens:
    /// eight such turns would be more than a reply holds.
    pub fn limit_of(period: usize, turn: &[u32]) -> usize {
        if period <= Self::SHORT {
            return Self::limit(period);
        }
        Self::limit_by(period, Self::many(turn.iter().copied()))
    }

    /// [`Self::limit_of`] past [`Self::SHORT`] tokens a turn, of a turn of `many` tokens (more than eight) or of few.
    fn limit_by(period: usize, many: bool) -> usize {
        match (many, period > Self::LONG) {
            (true, true) => 2 * period + Self::INTO,
            (true, false) => (8 * period).max(1024),
            (false, _) => (32 * period).max(4096),
        }
    }

    /// Whether a turn is made of more than eight tokens.
    fn many(turn: impl IntoIterator<Item = u32>) -> bool {
        let mut seen: Vec<u32> = Vec::with_capacity(9);
        for t in turn {
            if !seen.contains(&t) {
                seen.push(t);
                if seen.len() > 8 {
                    return true;
                }
            }
        }
        false
    }

    /// The limit of the run of `period` the newest tokens are in ([`Self::limit_of`], its turn read where it lies), in a
    /// reply turned from a run `turned` times already: halved each time, down to eight turns of the run. (A long
    /// block's is never turned from, so never halved.)
    fn limit_here(&self, period: usize, turned: usize) -> usize {
        if period > Self::LONG {
            let n = self.last.len();
            return Self::limit_by(period, Self::many(self.last.range(n - period..).copied()));
        }
        let limit = if period <= Self::SHORT {
            Self::limit(period)
        } else {
            let n = self.last.len();
            Self::limit_by(period, Self::many(self.last.range(n - period..).copied()))
        };
        (limit >> turned.min(8)).max(8 * period)
    }

    /// The least [`Self::limit_here`] a run of `period` can have, whatever its turn is made of: a run shorter is none,
    /// and what it is made of (and its divisors) is read only past that.
    fn least_at(period: usize, turned: usize) -> usize {
        if period > Self::LONG {
            return 2 * period + Self::INTO;
        }
        let least = if period <= Self::SHORT { Self::limit(period) } else { (8 * period).max(1024) };
        (least >> turned.min(8)).max(8 * period)
    }

    /// Whether a run is within `margin` tokens of its limit, so that [`Self::over`] may soon say so of a token: a
    /// greedy request's rows are then read back whole, as the model's pick alone (made on its GPU) leaves no other
    /// token to take, and the reply would be stopped where it could have been turned. `margin` covers the rows a
    /// check of drafts made before this was asked. (A long block's run is stopped, not turned: not asked of.)
    pub fn near(&self, turned: usize, margin: usize) -> bool {
        let n = self.last.len();
        (1..=Self::LONG.min(n)).any(|period| {
            if self.run[period] == 0 || self.run[period] + 1 + period + margin < Self::least_at(period, turned) {
                return false;
            }
            // (a run of a shorter period too is that period's, by its limit, as in `over`: a run of one token was near
            // at its multiples' 2,048 tokens, 2,000 before its own limit of 4,096)
            if (1..period).any(|d| period % d == 0 && self.run[d] + d >= self.run[period] + period) {
                return false;
            }
            self.run[period] + 1 + period + margin >= self.limit_here(period, turned)
        })
    }

    /// The token is the reply's next.
    pub fn push(&mut self, token: u32) {
        // (each period's run against the token that far back, the newest first; a period longer than the reply so far
        // has had no run, and has none)
        for (run, &before) in self.run[1..].iter_mut().zip(self.last.iter().rev()) {
            *run = if before == token { *run + 1 } else { 0 };
        }
        if self.last.len() == Self::MOST {
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
        (1..=Self::MOST.min(n)).zip(self.last.iter().rev()).find_map(|(period, &before)| {
            if before != next {
                return None;
            }
            let span = self.run[period] + 1 + period;
            if span < Self::least_at(period, turned) {
                return None;
            }
            // (a run of a shorter period too is that period's: asked first, as a run of one token is a run of every
            // period, and what each is made of would be read for nothing)
            if (1..period).any(|d| period % d == 0 && self.last[n - d] == next && self.run[d] + 1 + d >= span) {
                return None;
            }
            let limit = self.limit_here(period, turned);
            (span >= limit).then(|| Cycle { period, span, past: span - limit })
        })
    }

    /// The limit of a run of `period` that opens more than it closes: [`Self::NESTED`] turns, and 128 tokens at least
    /// (a parser's test nesting arrays 32 deep is let be).
    fn nested_limit(period: usize) -> usize {
        (Self::NESTED * period).max(128)
    }

    /// Whether `next` would go on a run of a short turn (up to [`Self::SHORT`] tokens) that opens more brackets than it
    /// closes, at [`Self::nested_limit`] or past it: a call or an index written inside itself again and again (`f(f(f(`,
    /// `a[b[a[b[`), which no code or data means past a few levels, and which nothing after it can make whole. Its own
    /// limit, under [`Self::limit`]'s: a 33B with its thinking off wrote `dirs.indexOf(` 513 times (2,052 tokens, that
    /// limit) into a file, twice. `opens` says of a turn whether it opens more than it closes ([`opens_more`] of it
    /// written out); it is asked only of a run that long.
    pub fn nested(&self, next: u32, opens: impl Fn(&[u32]) -> bool) -> Option<Cycle> {
        let n = self.last.len();
        (1..=Self::SHORT.min(n)).zip(self.last.iter().rev()).find_map(|(period, &before)| {
            if before != next {
                return None;
            }
            let span = self.run[period] + 1 + period;
            let limit = Self::nested_limit(period);
            if span < limit {
                return None;
            }
            // (a run of a shorter period too is that period's, as in `over`)
            if (1..period).any(|d| period % d == 0 && self.last[n - d] == next && self.run[d] + 1 + d >= span) {
                return None;
            }
            let turn: Vec<u32> = self.last.range(n - period..).copied().collect();
            opens(&turn).then(|| Cycle { period, span, past: span - limit })
        })
    }

    /// Whether a run that opens more than it closes is within `margin` tokens of its limit: [`Self::near`] for
    /// [`Self::nested`]'s runs.
    pub fn nested_near(&self, margin: usize, opens: impl Fn(&[u32]) -> bool) -> bool {
        let n = self.last.len();
        (1..=Self::SHORT.min(n)).any(|period| {
            if self.run[period] == 0 || self.run[period] + 1 + period + margin < Self::nested_limit(period) {
                return false;
            }
            opens(&self.last.range(n - period..).copied().collect::<Vec<_>>())
        })
    }
}

/// Whether text opens more brackets (`(`, `[`, `{`) than it closes.
pub(crate) fn opens_more(text: &str) -> bool {
    let depth: i64 = text.chars().map(|c| match c {
        '(' | '[' | '{' => 1,
        ')' | ']' | '}' => -1,
        _ => 0,
    }).sum();
    depth > 0
}

/// What a reply the guard stopped says: [`CODE`], how many tokens the run's turn is, the turn written out (quoted and
/// escaped as JSON writes a string, its middle left out past 80 characters), how many times it came, and whether that
/// was in a tool call (where nothing the call wrote is used). OAIY's Agent quotes it to the model when it asks again.
pub(crate) fn stopped(cycle: Cycle, turn: &str, in_call: bool) -> String {
    let chars: Vec<char> = turn.chars().collect();
    let shown: String = if chars.len() > 80 { chars[..60].iter().chain(&['…']).chain(&chars[chars.len() - 19..]).collect() } else { turn.to_owned() };
    let mut quoted = String::from('"');
    for c in shown.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            c if (c as u32) < 0x20 => quoted.push_str(&format!("\\u{:04x}", c as u32)),
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    let what = if cycle.period == 1 { "token".to_owned() } else { format!("{} tokens", cycle.period) };
    format!("{CODE}: the reply went on repeating the same {what} ({quoted}, {} times: {} tokens){} and was stopped", cycle.span / cycle.period.max(1), cycle.span, if in_call { " in a tool call" } else { "" })
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
    fn a_function_written_again_and_again_is_a_cycle_after_eight_turns() {
        // a 150-token function of many tokens, written over and over (inside a tool call, where the prose guards do
        // not look): eight of it, 1,200 tokens, then the next token of the ninth is refused
        let function: Vec<u32> = (0..150).map(|i| 1000 + (i * 7919) % 400).collect();
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &function, 20_000).expect("a function written without end");
        assert_eq!((cycle.period, pushed), (150, 1199));
        // a longer one, the same: eight turns
        let block: Vec<u32> = (0..400).map(|i| 3000 + (i * 104_729) % 900).collect();
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &block, 20_000).unwrap();
        assert_eq!((cycle.period, pushed), (400, 3199));
        // a map's row of 50 cells (`0` and `,`, 100 tokens, then a newline): data, let be 32 rows and more
        let row: Vec<u32> = (0..50).flat_map(|_| [10, 11]).chain([12]).collect();
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &row, 20_000).unwrap();
        assert_eq!((cycle.period, pushed), (101, 4095));
        // code that goes on (each block another) is no run at all
        let mut guard = CycleGuard::default();
        let flowing: Vec<u32> = (0..6000).map(|i| 5000 + (i * 7919) % 3000).collect();
        assert!(until_over(&mut guard, &flowing, 6000).is_none());
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
    fn a_run_of_one_or_two_tokens_is_near_only_its_own_limit() {
        // (its multiples, periods 4 to 15 at 2,048 tokens, were taken for it: near from 2,031, over at 4,095)
        for unit in [vec![5u32], vec![10, 11]] {
            let mut guard = CycleGuard::default();
            let mut written = 0;
            while !guard.near(0, 16) {
                assert!(guard.over(unit[written % unit.len()], 0).is_none(), "{unit:?}: over before near, at {written}");
                guard.push(unit[written % unit.len()]);
                written += 1;
            }
            assert_eq!(written, 4096 - 1 - 16, "{unit:?}");
        }
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
        // nor one whose repeats are longer than it watches (a book written again)
        let book: Vec<u32> = (0..CycleGuard::MOST as u32 + 8).collect();
        let mut guard = CycleGuard::default();
        assert!(until_over(&mut guard, &book, 2 * book.len() + 1000).is_none());
    }

    /// `n` tokens of many kinds with no shorter period among them: a stretch of code or prose, as the guard sees one.
    fn stretch(n: usize, seed: u32) -> Vec<u32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                1000 + x % 5000
            })
            .collect()
    }

    #[test]
    fn a_long_block_written_again_whole_is_a_cycle_into_its_third_copy() {
        // a file of 1,100 tokens written again into the call that writes it: let be its second copy and 128 tokens of a
        // third, where eight turns (8,800 tokens) would have let it go on to the reply's limit
        let file = stretch(1100, 7);
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &file, 20_000).expect("a file written again");
        assert_eq!((cycle.period, pushed, cycle.past), (1100, 2 * 1100 + CycleGuard::INTO - 1, 0));
        assert!(cycle.whole(), "stopped, not turned: another token would leave the copy in the file");
        // after what came before it, the same: the run is the copy's alone
        let mut guard = CycleGuard::default();
        stretch(3000, 11).into_iter().for_each(|t| guard.push(t));
        let (pushed, cycle) = until_over(&mut guard, &stretch(5000, 13), 20_000).unwrap();
        assert_eq!((cycle.period, pushed), (5000, 2 * 5000 + CycleGuard::INTO - 1));
        // a reply turned from runs before is given no less (a long block's limit is not halved)
        let mut guard = CycleGuard::default();
        let mut written = 0;
        while guard.over(file[written % file.len()], 3).is_none() {
            guard.push(file[written % file.len()]);
            written += 1;
        }
        assert_eq!(written, 2 * 1100 + CycleGuard::INTO - 1);
        // a block of LONG tokens keeps its eight turns, and is turned from, not stopped
        let block = stretch(CycleGuard::LONG, 17);
        let mut guard = CycleGuard::default();
        let (pushed, cycle) = until_over(&mut guard, &block, 20_000).unwrap();
        assert_eq!((cycle.period, pushed), (CycleGuard::LONG, 8 * CycleGuard::LONG - 1));
        assert!(!cycle.whole());
        // a long row of data (a map's 750 cells, `0` and `,`, and its line end) is let be, row after row
        let row: Vec<u32> = (0..750).flat_map(|_| [10, 11]).chain([12]).collect();
        let mut guard = CycleGuard::default();
        assert!(until_over(&mut guard, &row, 10 * row.len()).is_none());
        // and a long block that is not the same again (the file's copy with a line changed) is none: the run starts over
        let mut changed = file.clone();
        changed[600] = 99;
        let mut guard = CycleGuard::default();
        file.iter().chain(&changed).for_each(|&t| guard.push(t));
        for &t in &file[..700] {
            assert_eq!(guard.over(t, 0), None);
            guard.push(t);
        }
    }

    #[test]
    fn a_call_nested_in_itself_is_a_cycle_after_32_turns() {
        // `dirs.indexOf(` as four tokens, the last an opening bracket: what a 33B with its thinking off wrote 513 times
        // into a file, at the run's own limit of 2,048 tokens
        let opens = |turn: &[u32]| turn.contains(&4);
        let unit = [1, 2, 3, 4];
        let mut guard = CycleGuard::default();
        guard.push(900); // what came before it
        let mut written = 0;
        let cycle = loop {
            let next = unit[written % 4];
            assert_eq!(guard.over(next, 0), None, "under its run's own limit throughout");
            if let Some(cycle) = guard.nested(next, opens) {
                break cycle;
            }
            guard.push(next);
            written += 1;
        };
        assert_eq!((cycle.period, cycle.span, cycle.past, written), (4, 128, 0, 127));
        assert_eq!(CycleGuard::limit(4), 2048);
        // near it a margin before, as `near` is of the runs `over` judges
        let mut guard = CycleGuard::default();
        (0..127 - 16).for_each(|i| guard.push(unit[i % 4]));
        assert!(guard.nested_near(16, opens) && !guard.nested_near(15, opens));
        // one opening token over and over: 128 of it (a parser's test nesting arrays 32 deep is let be)
        let mut guard = CycleGuard::default();
        for _ in 0..127 {
            assert_eq!(guard.nested(9, |t: &[u32]| t == [9]), None);
            guard.push(9);
        }
        assert_eq!(guard.nested(9, |t: &[u32]| t == [9]).map(|c| (c.period, c.span)), Some((1, 128)));
        // a turn that closes what it opens (a list's item, `{"key":"ArrowRight"},`) is none: its limit is `over`'s
        let item = [1, 2, 3, 4, 5, 6, 7];
        let mut guard = CycleGuard::default();
        for i in 0..3000 {
            assert_eq!(guard.nested(item[i % 7], |_: &[u32]| false), None);
            assert!(!guard.nested_near(16, |_: &[u32]| false));
            guard.push(item[i % 7]);
        }
        // a run's turn is asked whole, and once a shorter period's run holds it, as that period's
        let mut guard = CycleGuard::default();
        let asked = std::cell::RefCell::new(Vec::new());
        for i in 0..300 {
            let _ = guard.nested(unit[i % 4], |t: &[u32]| {
                asked.borrow_mut().push(t.len());
                false
            });
            guard.push(unit[i % 4]);
        }
        let asked = asked.into_inner();
        assert!(!asked.is_empty() && asked.iter().all(|&len| len == 4), "{asked:?}");
        // a turn longer than SHORT tokens is not judged so
        let long = stretch(CycleGuard::SHORT + 1, 5);
        let mut guard = CycleGuard::default();
        for i in 0..40 * long.len() {
            assert_eq!(guard.nested(long[i % long.len()], |_: &[u32]| true), None);
            guard.push(long[i % long.len()]);
        }
    }

    #[test]
    fn a_turn_opens_more_than_it_closes() {
        for text in ["dirs.indexOf(", "dirs.indexOf(dirsInv[", "f(", "[[", "{ a: {"] {
            assert!(opens_more(text), "{text}");
        }
        for text in ["{\"key\":\"ArrowRight\"},", "0,", "f(x)", ")", "]);", "", "abc"] {
            assert!(!opens_more(text), "{text}");
        }
    }

    #[test]
    fn a_stopped_reply_says_what_it_repeated_and_where() {
        let said = stopped(Cycle { period: 4, span: 128, past: 0 }, "dirs.indexOf(", true);
        assert_eq!(said, "generation_repetition: the reply went on repeating the same 4 tokens (\"dirs.indexOf(\", 32 times: 128 tokens) in a tool call and was stopped");
        // quotes and line ends as JSON writes them
        let said = stopped(Cycle { period: 3, span: 4096, past: 0 }, "say \"hi\"\n", false);
        assert!(said.ends_with("(\"say \\\"hi\\\"\\n\", 1365 times: 4096 tokens) and was stopped"), "{said}");
        // a long turn by its start and end
        let turn = format!("{}{}", "a".repeat(100), "b".repeat(100));
        let said = stopped(Cycle { period: 1100, span: 2328, past: 0 }, &turn, true);
        assert!(said.contains(&format!("(\"{}…{}\", 2 times: 2328 tokens) in a tool call", "a".repeat(60), "b".repeat(19))), "{said}");
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
