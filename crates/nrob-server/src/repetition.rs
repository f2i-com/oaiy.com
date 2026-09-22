//! Bound degenerate prose/reasoning generation without inspecting user text.
pub(crate) const CODE: &str = "generation_repetition";

#[derive(Default)]
pub(crate) struct Guard { tokens: Vec<u32> }

impl Guard {
    /// Four consecutive equal blocks of 48..=256 tokens. Keep only the
    /// relevant suffix. Tool payloads may legitimately contain repeated data;
    /// do not apply the heuristic there, or carry prose history across them.
    pub fn push(&mut self, token: u32, tool_payload: bool) -> bool {
        if tool_payload { self.tokens.clear(); return false; }
        self.tokens.push(token);
        if self.tokens.len() > 1024 { self.tokens.remove(0); }
        let n = self.tokens.len();
        for period in 48..=256.min(n / 4) {
            let tail = &self.tokens[n - period * 4..];
            if tail[period..].iter().zip(tail.iter()).all(|(a,b)| a == b) { return true; }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_cycles_stop_but_progress_and_three_cycles_do_not() {
        let mut g = Guard::default();
        for _ in 0..3 { for token in 0..61 { assert!(!g.push(token, false)); } }
        for token in 0..60 { assert!(!g.push(token, false)); }
        assert!(g.push(60, false));
        let mut progressing = Guard::default();
        for token in 0..4096 { assert!(!progressing.push(token, false)); }
        assert_eq!(progressing.tokens.len(), 1024);
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
