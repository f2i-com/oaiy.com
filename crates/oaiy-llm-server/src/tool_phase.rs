//! Precision controller only; tool execution still requires the DSML validator.
pub(crate) struct ToolPhase {
    text: String,
    original: bool,
    closed: bool,
    parser: dsv41::chat::StreamParser,
}
impl Default for ToolPhase {
    fn default() -> Self {
        Self {text: String::new(), original: false, closed: false,
            parser: dsv41::chat::StreamParser::new(dsv41::chat::Mode::Chat)}
    }
}
impl ToolPhase {
    pub fn push(&mut self, piece: &str) {
        self.text.push_str(piece);
        let _ = self.parser.push(piece);
        if !self.original && !self.closed {
            if let Some((_, rest)) = self.text.split_once("<｜DSML｜") {
                if let Some((tag, _)) = rest.split_once('>') {
                    self.original = tag.trim() == "calls";
                }
            }
        }
        // A closing tag embedded inside a string is not a completed call.
        if self.parser.tool_calls_ready() {
            self.original = false;
            self.closed = true;
        }
    }
    pub fn original(&self) -> bool { self.original }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn switches_at_complete_open_and_valid_close() {
        let mut p=ToolPhase::default();
        for s in ["<", "｜DSML｜", " ca", "lls"] {p.push(s);assert!(!p.original());}
        p.push(">\n"); assert!(p.original());
        p.push("<｜DSML｜ invoke name=\"workspace_info\">\n</｜DSML｜ invoke>\n");
        assert!(p.original());
        p.push("</｜DSML｜ calls>"); assert!(!p.original());
    }
    #[test]
    fn invalid_call_does_not_signal_valid_completion() {
        let mut p=ToolPhase::default();
        p.push("<｜DSML｜ calls><｜DSML｜ invoke name=\"x\"><parameter name=\"bad\">x</parameter></invoke></｜DSML｜ calls>");
        assert!(p.original()); // request cleanup always restores ternary
        let mut plain=ToolPhase::default(); plain.push("ordinary reply <calls>hello</calls>"); assert!(!plain.original());
    }
}
