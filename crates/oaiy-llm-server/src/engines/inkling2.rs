//! Thinking Machines Lab Inkling-2 / Inkling-2-Mini engine, chat template, and
//! streaming output translator (thinking, content, multi-tool calls, vision).
//!
//! Prompt format matches `thinkingmachines/Inkling-2-Mini/chat_template.jinja`:
//!   * Optional tool declarations at the start:
//!     `<|message_system|>tool_declare<|content_xml|>[...]<|end_message|>`
//!   * Thinking effort line emitted before the first non-system message (or at the end):
//!     `<|message_system|><|content_text|>Thinking effort level: {num}<|end_message|>`
//!   * Turns use `<|message_{system,user,model,tool}|>[name]<|content_{text,thinking,image,invoke_tool_json}|>...<|end_message|>`
//!     with `<|content_model_end_sampling|>` terminating assistant turns in history.
//!   * No trailing assistant opener is emitted; the model itself opens `<|message_model|>`.
//!
//! Output translation converts Inkling-2's native structured output into OAIY's shared
//! `<think>...</think>` + DSML `<｜DSML｜ calls>` stream so `api.rs` handles SSE streaming,
//! `reasoning_content`, `content`, `tool_calls`, stop strings, and contract validation uniformly.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use oaiy_engine::json::Json;
use oaiy_engine::Result;
use tokenizer::Tokenizer;

use crate::dsv41::chat::{Encoded, Mode, Options, DSML};
use crate::engines::disk::DiskCache;
use crate::engines::glm::ThinkBudget;
use crate::engines::job::{penalize_reply, sample, Event, Finish, ImagePrep, Job};
use crate::llm::arch::inkling2::state::{Inkling2ConvSnapshot, Inkling2StateSnapshot};
use crate::llm::arch::inkling2::vision::Inkling2Vision;
use crate::llm::arch::inkling2::Inkling2Model;

pub const MSG_USER: &str = "<|message_user|>";
pub const MSG_MODEL: &str = "<|message_model|>";
pub const MSG_SYSTEM: &str = "<|message_system|>";
pub const MSG_TOOL: &str = "<|message_tool|>";
pub const CONTENT_TEXT: &str = "<|content_text|>";
pub const CONTENT_THINKING: &str = "<|content_thinking|>";
pub const CONTENT_IMAGE: &str = "<|content_image|>";
pub const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const END_MESSAGE: &str = "<|end_message|>";
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";
/// Special placeholder token expanded to `rows * cols` vision embeddings.
pub const IMAGE_PLACEHOLDER: &str = "<|unused_200054|>";

/// Map OAIY's `(Mode, effort 1..=100)` onto Inkling-2's `[0, 0.99]` effort string
/// (matching `effort_map` in `chat_template.jinja`:
/// `none=0, minimal=0.1, low=0.2, medium=0.7, high=0.9, xhigh/max=0.99`).
pub fn effort_level_str(mode: Mode, effort: u32) -> &'static str {
    if mode != Mode::Thinking {
        return "0";
    }
    match effort {
        0 => "0",
        1..=20 => "0.1",
        21..=50 => "0.2",
        51..=74 => "0.7",
        75..=94 => "0.9",
        _ => "0.99",
    }
}

/// Serialize a `Json` value with object keys sorted lexicographically (case-sensitive)
/// and compact `(",", ":")` separators, matching Jinja's `tojson(sort_keys=true, separators=(",", ":"))`.
fn to_sorted_compact_json(v: &Json) -> String {
    match v {
        Json::Obj(members) => {
            let mut sorted: BTreeMap<&str, &Json> = BTreeMap::new();
            for (k, val) in members {
                sorted.insert(k.as_str(), val);
            }
            let mut out = String::from("{");
            for (i, (k, val)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Json::str(k).to_json());
                out.push(':');
                out.push_str(&to_sorted_compact_json(val));
            }
            out.push('}');
            out
        }
        Json::Arr(items) => {
            let mut out = String::from("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&to_sorted_compact_json(item));
            }
            out.push(']');
            out
        }
        other => other.to_json(),
    }
}

fn role_token(role: &str) -> std::result::Result<&'static str, String> {
    match role {
        "user" => Ok(MSG_USER),
        "assistant" => Ok(MSG_MODEL),
        "system" | "developer" => Ok(MSG_SYSTEM),
        "tool" | "function" => Ok(MSG_TOOL),
        other => Err(format!("Unknown message role: {other}")),
    }
}

/// Format chat messages into an Inkling-2 prompt string + extracted image records.
pub fn chat_prompt(msgs: &[Json], opts: &Options, allow_images: bool) -> std::result::Result<Encoded, String> {
    let mut prompt = String::with_capacity(512);
    let mut images = Vec::new();

    // 1. Tool declarations (attached to the first system message by `api::Server::messages`).
    let tools: &[Json] = msgs
        .first()
        .filter(|m| matches!(m.get("role").and_then(Json::as_str), Some("system" | "developer")))
        .and_then(|m| m.get("tools"))
        .and_then(Json::as_array)
        .unwrap_or(&[]);

    if !tools.is_empty() {
        let mut specs = Vec::with_capacity(tools.len());
        for tool in tools {
            let func = tool.get("function").unwrap_or(tool);
            let name = func
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| "tool missing function.name".to_string())?;
            let description = func
                .get("description")
                .and_then(Json::as_str)
                .unwrap_or("");
            let parameters = func
                .get("parameters")
                .cloned()
                .filter(|v| !matches!(v, Json::Null))
                .unwrap_or_else(|| Json::Obj(Vec::new()));
            let typ = tool
                .get("type")
                .and_then(Json::as_str)
                .unwrap_or("function");
            specs.push(Json::obj([
                ("description", Json::str(description)),
                ("name", Json::str(name)),
                ("parameters", parameters),
                ("type", Json::str(typ)),
            ]));
        }
        prompt.push_str(MSG_SYSTEM);
        prompt.push_str("tool_declare<|content_xml|>");
        prompt.push_str(&to_sorted_compact_json(&Json::Arr(specs)));
        prompt.push_str(END_MESSAGE);
    }

    let effort_str = effort_level_str(opts.mode, opts.effort);
    let emit_effort = |out: &mut String| {
        out.push_str(MSG_SYSTEM);
        out.push_str(CONTENT_TEXT);
        out.push_str("Thinking effort level: ");
        out.push_str(effort_str);
        out.push_str(END_MESSAGE);
    };

    let mut effort_emitted = false;
    let last_user = msgs
        .iter()
        .rposition(|m| m.get("role").and_then(Json::as_str) == Some("user"));

    for (idx, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Json::as_str).unwrap_or("user");
        let r_base = role_token(role)?;

        if !effort_emitted && !matches!(role, "system" | "developer") {
            emit_effort(&mut prompt);
            effort_emitted = true;
        }

        // Skip synthetic empty system message added solely to carry `tools`.
        if matches!(role, "system" | "developer")
            && idx == 0
            && !tools.is_empty()
            && m.get("content").and_then(Json::as_str).is_some_and(str::is_empty)
        {
            continue;
        }

        // Header: `<|message_{role}|>` + optional name (resolved from `tool_call_id` for tool role).
        let mut rtok = String::from(r_base);
        if matches!(role, "tool" | "function") {
            if let Some(name) = m.get("name").and_then(Json::as_str).filter(|s| !s.is_empty()) {
                rtok.push_str(name);
            } else if let Some(tc_id) = m.get("tool_call_id").and_then(Json::as_str).filter(|s| !s.is_empty()) {
                'find_tc: for prev in msgs {
                    if prev.get("role").and_then(Json::as_str) == Some("assistant") {
                        if let Some(tcs) = prev.get("tool_calls").and_then(Json::as_array) {
                            for tc in tcs {
                                if tc.get("id").and_then(Json::as_str) == Some(tc_id) {
                                    if let Some(fn_name) = tc
                                        .get("function")
                                        .and_then(|f| f.get("name"))
                                        .and_then(Json::as_str)
                                    {
                                        rtok.push_str(fn_name);
                                        break 'find_tc;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if role == "assistant" {
            if let Some(name) = m.get("name").and_then(Json::as_str).filter(|s| !s.is_empty()) {
                rtok.push_str(name);
            }
        }

        let tool_calls = m
            .get("tool_calls")
            .and_then(Json::as_array)
            .filter(|a| !a.is_empty());

        // Reasoning content on assistant turns (kept after last user message, or when drop_thinking is false).
        let keep_thinking = !opts.drop_thinking || last_user.is_none_or(|u| idx > u);
        let reasoning = if role == "assistant" && keep_thinking {
            ["reasoning_content", "reasoning", "thinking"]
                .iter()
                .find_map(|k| m.get(k).and_then(Json::as_str).filter(|s| !s.is_empty()))
        } else {
            None
        };
        if let Some(rc) = reasoning {
            prompt.push_str(&rtok);
            prompt.push_str(CONTENT_THINKING);
            prompt.push_str(rc);
            prompt.push_str(END_MESSAGE);
        }

        // Message content (string or multimodal content parts).
        let content_val = m.get("content_blocks").or_else(|| m.get("content"));
        match content_val {
            Some(Json::Str(s)) => {
                if allow_images && s.contains(IMAGE_PLACEHOLDER) {
                    return Err(format!(
                        "message {}: send pictures as image_url parts, not as {IMAGE_PLACEHOLDER}",
                        idx + 1
                    ));
                }
                if !s.is_empty() || !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) {
                    prompt.push_str(&rtok);
                    prompt.push_str(CONTENT_TEXT);
                    prompt.push_str(s);
                    prompt.push_str(END_MESSAGE);
                }
            }
            Some(Json::Arr(parts)) => {
                for part in parts {
                    match part {
                        Json::Str(s) => {
                            if allow_images && s.contains(IMAGE_PLACEHOLDER) {
                                return Err(format!(
                                    "message {}: send pictures as image_url parts, not as {IMAGE_PLACEHOLDER}",
                                    idx + 1
                                ));
                            }
                            prompt.push_str(&rtok);
                            prompt.push_str(CONTENT_TEXT);
                            prompt.push_str(s);
                            prompt.push_str(END_MESSAGE);
                        }
                        Json::Obj(_) => match part.get("type").and_then(Json::as_str) {
                            None | Some("text" | "input_text") => {
                                let t = part.get("text").and_then(Json::as_str).unwrap_or("");
                                if allow_images && t.contains(IMAGE_PLACEHOLDER) {
                                    return Err(format!(
                                        "message {}: send pictures as image_url parts, not as {IMAGE_PLACEHOLDER}",
                                        idx + 1
                                    ));
                                }
                                prompt.push_str(&rtok);
                                prompt.push_str(CONTENT_TEXT);
                                prompt.push_str(t);
                                prompt.push_str(END_MESSAGE);
                            }
                            Some("image" | "input_image" | "image_url") => {
                                if !matches!(role, "user" | "tool" | "function") {
                                    return Err(
                                        "Image content is only supported for user and tool messages".into(),
                                    );
                                }
                                if !allow_images {
                                    return Err(
                                        "this model has no vision projector loaded; images are not supported".into(),
                                    );
                                }
                                let rec = match part.get("image_url") {
                                    Some(url) => Json::obj([("url", url.get("url").unwrap_or(url).clone())]),
                                    None => part.clone(),
                                };
                                images.push(rec);
                                prompt.push_str(&rtok);
                                prompt.push_str(CONTENT_IMAGE);
                                prompt.push_str(IMAGE_PLACEHOLDER);
                                prompt.push_str(END_MESSAGE);
                            }
                            Some("audio" | "input_audio" | "audio_url") => {
                                return Err("Inkling 2 does not accept audio input".into());
                            }
                            Some(other) => {
                                return Err(format!("Unsupported content part type: {other}"));
                            }
                        },
                        _ => return Err("Invalid content part".into()),
                    }
                }
            }
            None | Some(Json::Null) => {
                if !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) {
                    prompt.push_str(&rtok);
                    prompt.push_str(CONTENT_TEXT);
                    prompt.push_str(END_MESSAGE);
                }
            }
            Some(_) => return Err(format!("message {}: content must be text or a list of parts", idx + 1)),
        }

        // Assistant tool calls.
        if role == "assistant" {
            if let Some(calls) = tool_calls {
                for tc in calls {
                    let func = tc.get("function").unwrap_or(tc);
                    let fn_name = func
                        .get("name")
                        .and_then(Json::as_str)
                        .ok_or_else(|| "tool call function name must be a string".to_string())?;
                    let args_obj = match func.get("arguments") {
                        None | Some(Json::Null) => Json::Obj(Vec::new()),
                        Some(Json::Obj(m)) => Json::Obj(m.clone()),
                        Some(Json::Str(s)) if s.trim().is_empty() => Json::Obj(Vec::new()),
                        Some(Json::Str(s)) => {
                            let parsed = Json::parse(s.as_bytes())
                                .map_err(|e| format!("invalid tool call arguments JSON: {e}"))?;
                            if parsed.as_object().is_none() {
                                return Err("tool call arguments must be an object".into());
                            }
                            parsed
                        }
                        Some(_) => return Err("tool call arguments must be an object".into()),
                    };
                    prompt.push_str(MSG_MODEL);
                    prompt.push_str(fn_name);
                    prompt.push_str(CONTENT_INVOKE_TOOL_JSON);
                    prompt.push_str(r#"{"name":"#);
                    prompt.push_str(&Json::str(fn_name).to_json());
                    prompt.push_str(r#","args":"#);
                    prompt.push_str(&to_sorted_compact_json(&args_obj));
                    prompt.push('}');
                    prompt.push_str(END_MESSAGE);
                }
            }
            prompt.push_str(END_SAMPLING);
        }
    }

    if !effort_emitted {
        emit_effort(&mut prompt);
    }

    Ok(Encoded { prompt, images })
}

/// Token IDs for Inkling-2's structural control markers.
#[derive(Clone, Copy, Debug)]
pub struct SpecialIds {
    pub msg_user: Option<u32>,
    pub msg_model: Option<u32>,
    pub msg_tool: Option<u32>,
    pub content_text: Option<u32>,
    pub content_thinking: Option<u32>,
    pub content_invoke_tool: Option<u32>,
    pub end_message: Option<u32>,
    pub end_sampling: Option<u32>,
}

impl SpecialIds {
    pub fn from_tokenizer(tok: &Tokenizer) -> Self {
        Self {
            msg_user: tok.token_id(MSG_USER),
            msg_model: tok.token_id(MSG_MODEL),
            msg_tool: tok.token_id(MSG_TOOL),
            content_text: tok.token_id(CONTENT_TEXT),
            content_thinking: tok.token_id(CONTENT_THINKING),
            content_invoke_tool: tok.token_id(CONTENT_INVOKE_TOOL_JSON),
            end_message: tok.token_id(END_MESSAGE),
            end_sampling: tok.token_id(END_SAMPLING),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Segment {
    Idle,
    Header,
    Thinking,
    Text,
    ToolJson,
}

/// Translates Inkling-2's native token stream (`<|message_model|>...<|end_message|>`)
/// into OAIY's `<think>...</think>` + DSML `<｜DSML｜ calls>` event stream (or plain-text
/// `<tool_call>` blocks when no OpenAI `tools` catalog was declared).
pub struct Inkling2Translator {
    ids: SpecialIds,
    has_tools: bool,
    seg: Segment,
    header_buf: String,
    tool_buf: String,
    pending_calls: Vec<String>,
    tool_error: Option<String>,
    seen_thinking: bool,
    closed_thinking: bool,
    emitted_text: bool,
    seen_output: bool,
    pending_bytes: Vec<u8>,
}

impl Inkling2Translator {
    pub fn new(ids: SpecialIds) -> Self {
        Self::with_tools(ids, true)
    }

    pub fn with_tools(ids: SpecialIds, has_tools: bool) -> Self {
        Self {
            ids,
            has_tools,
            seg: Segment::Idle,
            header_buf: String::new(),
            tool_buf: String::new(),
            pending_calls: Vec::new(),
            tool_error: None,
            seen_thinking: false,
            closed_thinking: false,
            emitted_text: false,
            seen_output: false,
            pending_bytes: Vec::new(),
        }
    }

    pub fn tool_error(&self) -> Option<&str> {
        self.tool_error.as_deref()
    }

    pub fn in_thinking(&self) -> bool {
        self.seg == Segment::Thinking
    }

    pub fn in_text(&self) -> bool {
        self.seg == Segment::Text
    }

    pub fn in_header(&self) -> bool {
        self.seg == Segment::Header
    }

    pub fn in_tool(&self) -> bool {
        self.seg == Segment::ToolJson
    }

    pub fn seen_thinking(&self) -> bool {
        self.seen_thinking
    }

    pub fn closed_thinking(&self) -> bool {
        self.closed_thinking
    }

    pub fn has_output(&self) -> bool {
        self.seen_output
    }

    /// Feed one generated token ID. Returns translated UTF-8 chunks to emit as `Event::Text`
    /// and whether generation should stop (`<|content_model_end_sampling|>`).
    pub fn push(&mut self, tok_id: u32, tok: &Tokenizer) -> (Vec<String>, bool) {
        let mut out = Vec::new();

        if Some(tok_id) == self.ids.end_sampling {
            self.flush_pending_utf8(&mut out);
            self.flush_segment_close();
            self.flush_tool_calls(&mut out);
            return (out, true);
        }

        if Some(tok_id) == self.ids.msg_model {
            self.flush_pending_utf8(&mut out);
            self.flush_segment_close();
            self.seg = Segment::Header;
            self.header_buf.clear();
            return (out, false);
        }

        if Some(tok_id) == self.ids.content_thinking {
            self.flush_pending_utf8(&mut out);
            self.flush_segment_close();
            self.flush_tool_calls(&mut out);
            self.seg = Segment::Thinking;
            self.seen_thinking = true;
            self.closed_thinking = false;
            out.push("<think>".into());
            return (out, false);
        }

        if Some(tok_id) == self.ids.content_text {
            self.flush_pending_utf8(&mut out);
            self.flush_segment_close();
            self.flush_tool_calls(&mut out);
            if !self.closed_thinking {
                out.push("</think>".into());
                self.closed_thinking = true;
            }
            self.seg = Segment::Text;
            return (out, false);
        }

        if Some(tok_id) == self.ids.content_invoke_tool {
            self.flush_pending_utf8(&mut out);
            self.flush_segment_close();
            if !self.closed_thinking {
                out.push("</think>".into());
                self.closed_thinking = true;
            }
            self.seg = Segment::ToolJson;
            self.tool_buf.clear();
            return (out, false);
        }

        if Some(tok_id) == self.ids.end_message {
            self.flush_pending_utf8(&mut out);
            self.flush_segment_close();
            self.seg = Segment::Idle;
            return (out, false);
        }

        // Normal token inside the active segment (incrementally decode UTF-8 in O(1) amortized time
        // so invalid UTF-8 bytes emit U+FFFD and never stall streaming).
        let mut decoded = String::new();
        tok.decode_token_incremental(tok_id, &mut self.pending_bytes, &mut decoded);
        if !decoded.is_empty() {
            self.push_decoded(&decoded, &mut out);
        }
        (out, false)
    }

    /// Flush any remaining buffered text or tool calls at end of generation (e.g., `max_tokens`).
    pub fn finish(&mut self, _tok: &Tokenizer) -> Vec<String> {
        let mut out = Vec::new();
        self.flush_pending_utf8(&mut out);
        self.flush_segment_close();
        self.flush_tool_calls(&mut out);
        out
    }

    fn flush_pending_utf8(&mut self, out: &mut Vec<String>) {
        if self.pending_bytes.is_empty() {
            return;
        }
        let mut s = String::new();
        Tokenizer::flush_incremental(&mut self.pending_bytes, &mut s);
        if !s.is_empty() {
            self.push_decoded(&s, out);
        }
    }

    fn push_decoded(&mut self, text: &str, out: &mut Vec<String>) {
        match self.seg {
            Segment::Header => {
                self.header_buf.push_str(text);
                // Valid Inkling-2 headers are either empty (before `<|content_text|>`) or a short
                // tool identifier (before `<|content_invoke_tool_json|>`). If prose or `<tool_call>`
                // markup arrives here without `<|content_text|>`, promote it to `Segment::Text` so
                // the reply is never swallowed silently into `header_buf`.
                if self.header_buf.len() > 64
                    || self.header_buf.bytes().any(|b| b.is_ascii_whitespace() || b == b'<' || b == b'{')
                {
                    if !self.closed_thinking {
                        out.push("</think>".into());
                        self.closed_thinking = true;
                    }
                    let promoted = std::mem::take(&mut self.header_buf);
                    self.seg = Segment::Text;
                    self.emitted_text = true;
                    if !promoted.trim().is_empty() {
                        self.seen_output = true;
                    }
                    out.push(promoted);
                }
            }
            Segment::ToolJson => self.tool_buf.push_str(text),
            Segment::Thinking => {
                if !text.is_empty() {
                    out.push(text.to_string());
                }
            }
            Segment::Text => {
                if !text.is_empty() {
                    self.emitted_text = true;
                    if !text.trim().is_empty() {
                        self.seen_output = true;
                    }
                    out.push(text.to_string());
                }
            }
            Segment::Idle => {
                // Fallback if a model outputs plain text before any `<|content_*|>` tag.
                if !text.is_empty() {
                    if !self.closed_thinking {
                        out.push("</think>".into());
                        self.closed_thinking = true;
                    }
                    self.emitted_text = true;
                    if !text.trim().is_empty() {
                        self.seen_output = true;
                    }
                    out.push(text.to_string());
                }
            }
        }
    }

    fn flush_segment_close(&mut self) {
        match self.seg {
            Segment::Thinking => {
                // Defer emitting `</think>` until a non-thinking segment (`content_text`,
                // `content_invoke_tool`, or `Segment::Idle` fallback text) actually begins.
                // Closing `</think>` here on `<|end_message|>` / `<|content_model_end_sampling|>`
                // / `finish()` when the model emitted only thinking would transition `StreamParser`
                // out of `Phase::Reasoning` and defeat `answer_left_in_thought` recovery.
            }
            Segment::ToolJson => {
                if self.has_tools {
                    match format_tool_call_as_dsml(&self.header_buf, &self.tool_buf) {
                        Ok(dsml_invoke) => {
                            self.pending_calls.push(dsml_invoke);
                            self.seen_output = true;
                        }
                        Err(e) => {
                            if self.tool_error.is_none() {
                                self.tool_error = Some(e);
                            }
                        }
                    }
                } else {
                    // Text-protocol callers (no declared OpenAI `tools`, e.g. FormLogic Site AI
                    // on OAIY) instruct the model in prose to emit `<tool_call>` blocks. Never
                    // convert Inkling-2's native tool channel into DSML (which would fail
                    // `validate_tools` with `missing tool catalog`) or reject non-JSON payloads
                    // as `tool_contract_error` — emit plain-text `<tool_call>` markup instead.
                    match format_tool_call_as_text(&self.header_buf, &self.tool_buf) {
                        Ok(Some(text_call)) => {
                            self.pending_calls.push(text_call);
                            self.seen_output = true;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            if self.tool_error.is_none() {
                                self.tool_error = Some(e);
                            }
                        }
                    }
                }
                self.tool_buf.clear();
                self.header_buf.clear();
            }
            _ => {}
        }
    }

    fn flush_tool_calls(&mut self, out: &mut Vec<String>) {
        if self.tool_error.is_some() {
            self.pending_calls.clear();
            return;
        }
        if self.pending_calls.is_empty() {
            return;
        }
        let body = self.pending_calls.join("\n");
        self.pending_calls.clear();
        if self.has_tools {
            out.push(format!("\n\n<{DSML} calls>\n{body}\n</{DSML} calls>"));
        } else {
            if self.emitted_text {
                out.push(format!("\n\n{body}"));
            } else {
                out.push(body);
            }
            self.emitted_text = true;
        }
    }
}

/// Mask structural stop/re-think tokens that would otherwise let Inkling-2 terminate with only
/// reasoning (no text or tool call) or reopen an unbudgeted second thinking block in the same turn.
fn mask_post_thinking_logits(
    logits: &mut [f32],
    ids: SpecialIds,
    translator: &Inkling2Translator,
    room_left: bool,
    think_budget_exhausted: bool,
) {
    if room_left && !translator.has_output() && !translator.in_tool() {
        if translator.seen_thinking() {
            if let Some(es) = ids.end_sampling {
                if (es as usize) < logits.len() {
                    logits[es as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if !translator.in_thinking() {
            if let Some(em) = ids.end_message {
                if (em as usize) < logits.len() {
                    logits[em as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if translator.in_text() || translator.in_header() {
            if let Some(mm) = ids.msg_model {
                if (mm as usize) < logits.len() {
                    logits[mm as usize] = f32::NEG_INFINITY;
                }
            }
        }
    }
    // `ThinkBudget` only budgets a single thinking block (`seen` stays true once the first block
    // closes). Never allow Inkling-2 to open a second unbudgeted `<|content_thinking|>` block in
    // the same turn — whether the first closed naturally or via `think_budget`.
    let forbid_rethink =
        think_budget_exhausted || (translator.seen_thinking() && !translator.in_thinking());
    if forbid_rethink && !translator.in_tool() {
        if let Some(ct) = ids.content_thinking {
            if (ct as usize) < logits.len() {
                logits[ct as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Queue the Inkling-2 message opener tokens that must immediately follow a budget-forced
/// `<|end_message|>` so the model enters the answer channel (just like `</think>` in Qwen/DeepSeek)
/// instead of remaining idle where it can sample `<|content_model_end_sampling|>` or reopen thinking.
fn enqueue_forced_reply_opener(queue: &mut VecDeque<u32>, ids: SpecialIds, has_tools: bool) {
    if let Some(mm) = ids.msg_model {
        queue.push_back(mm);
    }
    if !has_tools {
        if let Some(ct) = ids.content_text {
            queue.push_back(ct);
        }
    }
}

/// Convert one Inkling-2 `<|content_invoke_tool_json|>` segment into plain-text `<tool_call>`
/// markup when no OpenAI `tools` catalog was declared by the caller.
fn format_tool_call_as_text(header_name: &str, raw: &str) -> std::result::Result<Option<String>, String> {
    let header = header_name.trim();
    let trimmed = raw.trim();
    if header.contains(DSML) || trimmed.contains(DSML) {
        return Err("unexpected DSML in Inkling-2 output".into());
    }
    if trimmed.is_empty() && header.is_empty() {
        return Ok(None);
    }
    // Model already wrote text-protocol `<tool_call ...>...</tool_call>` inside the tool channel.
    if trimmed.contains("<tool_call") {
        return Ok(Some(trimmed.to_string()));
    }
    if let Ok(parsed) = Json::parse(trimmed.as_bytes()) {
        if let Json::Obj(members) = &parsed {
            let explicit_name = parsed.get("name").and_then(Json::as_str).map(str::trim).filter(|s| !s.is_empty());
            let name = explicit_name.or_else(|| (!header.is_empty()).then_some(header));
            if let Some(name) = name {
                let raw_args = parsed
                    .get("args")
                    .or_else(|| parsed.get("arguments"))
                    .or_else(|| parsed.get("parameters"));
                let arguments = match raw_args {
                    Some(Json::Obj(obj)) => Json::Obj(obj.clone()),
                    Some(Json::Str(s)) => {
                        let t = s.trim();
                        if t.is_empty() {
                            Json::Obj(Vec::new())
                        } else if let Ok(Json::Obj(obj)) = Json::parse(t.as_bytes()) {
                            Json::Obj(obj)
                        } else {
                            Json::Str(s.clone())
                        }
                    }
                    None | Some(Json::Null) => {
                        let flat: Vec<(String, Json)> = members
                            .iter()
                            .filter(|(k, _)| !matches!(k.as_str(), "name" | "args" | "arguments" | "parameters"))
                            .cloned()
                            .collect();
                        Json::Obj(flat)
                    }
                    Some(other) => other.clone(),
                };
                let call_json = Json::obj([
                    ("name", Json::str(name)),
                    ("arguments", arguments),
                ])
                .to_json();
                if call_json.contains(DSML) {
                    return Err("unexpected DSML in Inkling-2 output".into());
                }
                return Ok(Some(format!("<tool_call>\n{call_json}\n</tool_call>")));
            }
        }
    }
    if !header.is_empty() && valid_dsml_attr(header) {
        Ok(Some(format!("<tool_call name=\"{header}\">\n{trimmed}\n</tool_call>")))
    } else {
        Ok(Some(format!("<tool_call>\n{trimmed}\n</tool_call>")))
    }
}

fn valid_dsml_attr(s: &str) -> bool {
    !s.is_empty()
        && !s.contains(DSML)
        && !s
            .chars()
            .any(|c| matches!(c, '"' | '\'' | '<' | '>' | '\n' | '\r' | '\0'))
}

/// Convert one Inkling-2 JSON tool invocation (`{"name":"fn","args":{...}}`) into a single
/// OAIY DSML `<｜DSML｜ invoke>` block.
fn format_tool_call_as_dsml(header_name: &str, json_str: &str) -> std::result::Result<String, String> {
    let trimmed = json_str.trim();
    if trimmed.is_empty() {
        return Err("empty Inkling-2 tool call".into());
    }
    let parsed = Json::parse(trimmed.as_bytes()).map_err(|e| format!("invalid JSON tool call: {e}"))?;
    if !matches!(parsed, Json::Obj(_)) {
        return Err("Inkling-2 tool call must be a JSON object".into());
    }
    let name = parsed
        .get("name")
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let h = header_name.trim();
            (!h.is_empty()).then_some(h)
        })
        .ok_or_else(|| "missing Inkling-2 tool name".to_string())?;
    if !valid_dsml_attr(name) {
        return Err("invalid Inkling-2 tool name".into());
    }
    let args: Vec<(String, Json)> = match parsed.get("args").or_else(|| parsed.get("arguments")) {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::Obj(obj)) => obj.clone(),
        Some(Json::Str(s)) => {
            let trimmed_args = s.trim();
            if trimmed_args.is_empty() {
                Vec::new()
            } else {
                match Json::parse(trimmed_args.as_bytes()) {
                    Ok(Json::Obj(obj)) => obj,
                    Ok(_) => return Err("Inkling-2 tool arguments must be a JSON object".into()),
                    Err(e) => return Err(format!("invalid JSON tool arguments: {e}")),
                }
            }
        }
        Some(_) => return Err("Inkling-2 tool arguments must be a JSON object".into()),
    };

    let mut seen = std::collections::BTreeSet::new();
    let mut params = Vec::with_capacity(args.len());
    for (k, v) in args {
        if !valid_dsml_attr(&k) {
            return Err("invalid Inkling-2 parameter name".into());
        }
        if !seen.insert(k.clone()) {
            return Err("duplicate Inkling-2 parameter".into());
        }
        let (is_str, rendered) = match v {
            Json::Str(s) => ("true", s),
            other => ("false", other.to_json()),
        };
        if rendered.contains(DSML) {
            return Err("unexpected DSML in Inkling-2 tool argument".into());
        }
        params.push(format!(
            "<{DSML} parameter name=\"{k}\" string=\"{is_str}\">{rendered}</{DSML} parameter>"
        ));
    }
    let inner = params.join("\n");
    Ok(format!("<{DSML} invoke name=\"{name}\">\n{inner}\n</{DSML} invoke>"))
}

const SOFT_KEPT: usize = 4;
/// Maximum lightweight conv-only checkpoints retained for the active live conversation (~3.7 MiB each).
const MAX_LIVE_CHECKPOINTS: usize = 8;
/// Minimum prompt prefix length worth snapshotting into host RAM or persistent disk cache.
const MIN_SNAPSHOT_TOKENS: usize = 64;
/// Avoid re-writing conversation disk checkpoints when an earlier non-base disk entry already
/// sits within this many tokens (keeps SSD writes low during short multi-step tool turns).
const DISK_NEAR_TOKENS: usize = 256;

/// Token position of the immutable system+tools+effort base prefix: the index of the first
/// `<|message_user|>` token (strictly inside `0..prompt.len()`).
pub(crate) fn base_boundary(prompt: &[u32], specials: SpecialIds) -> Option<usize> {
    let msg_user = specials.msg_user?;
    prompt
        .iter()
        .position(|&t| t == msg_user)
        .filter(|&p| p >= MIN_SNAPSHOT_TOKENS && p < prompt.len())
}

/// Natural message-boundary checkpoint positions inside `prompt`:
///   * `base`: right before the first `<|message_user|>` (system prompt + tools + effort),
///   * last `<|message_user|>` and last `<|message_tool|>` boundaries,
///   * `prompt.len() - 1` (so exact retries still leave 1 token for logits).
pub(crate) fn checkpoint_positions(prompt: &[u32], specials: SpecialIds) -> Vec<usize> {
    let mut stops = Vec::with_capacity(4);
    if let Some(u) = specials.msg_user {
        if let Some(first_u) = prompt.iter().position(|&t| t == u) {
            stops.push(first_u);
        }
        if let Some(last_u) = prompt.iter().rposition(|&t| t == u) {
            stops.push(last_u);
        }
    }
    if let Some(tool) = specials.msg_tool {
        if let Some(last_t) = prompt.iter().rposition(|&t| t == tool) {
            stops.push(last_t);
        }
    }
    if prompt.len() > 1 {
        stops.push(prompt.len() - 1);
    }
    stops.retain(|&p| p > 0 && p < prompt.len());
    stops.sort_unstable();
    stops.dedup();
    stops
}

struct RamEntry {
    keys: Vec<u64>,
    snap: Arc<Inkling2StateSnapshot>,
    bytes: usize,
    base: bool,
    used: u64,
}

/// Byte-budgeted host-RAM prefix snapshot cache (LRU; protects `base` system-prompt entries
/// against eviction by ordinary conversation snapshots).
#[derive(Default)]
pub(crate) struct RamCache {
    entries: Vec<RamEntry>,
    budget: usize,
    held: usize,
    clock: u64,
}

impl RamCache {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            held: 0,
            clock: 0,
        }
    }

    pub(crate) fn has(&self, keys: &[u64]) -> bool {
        self.entries.iter().any(|e| e.keys == keys)
    }

    pub(crate) fn best(&mut self, prompt: &[u64], limit: usize, after: usize) -> Option<(Vec<u64>, Arc<Inkling2StateSnapshot>)> {
        let best_idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.keys.len() > after
                    && e.keys.len() <= limit
                    && prompt.starts_with(&e.keys)
            })
            .max_by_key(|(_, e)| (e.keys.len(), e.used))
            .map(|(i, _)| i)?;
        self.clock += 1;
        let entry = &mut self.entries[best_idx];
        entry.used = self.clock;
        Some((entry.keys.clone(), Arc::clone(&entry.snap)))
    }

    pub(crate) fn insert(&mut self, keys: Vec<u64>, snap: Arc<Inkling2StateSnapshot>, base: bool) -> bool {
        if self.budget == 0 || keys.is_empty() {
            return false;
        }
        let bytes = snap.bytes().saturating_add(keys.len() * 8);
        if bytes > self.budget {
            return false;
        }
        if let Some(existing) = self.entries.iter_mut().find(|e| e.keys == keys) {
            self.clock += 1;
            existing.used = self.clock;
            existing.base |= base;
            return true;
        }
        // Evict LRU non-base entries first; evict oldest base entry only if required.
        while self.held.saturating_add(bytes) > self.budget && !self.entries.is_empty() {
            let victim = (0..self.entries.len())
                .min_by_key(|&i| (self.entries[i].base, self.entries[i].used))
                .expect("non-empty entries");
            let removed = self.entries.remove(victim);
            self.held = self.held.saturating_sub(removed.bytes);
        }
        if self.held.saturating_add(bytes) > self.budget {
            return false;
        }
        self.clock += 1;
        self.held = self.held.saturating_add(bytes);
        self.entries.push(RamEntry {
            keys,
            snap,
            bytes,
            base,
            used: self.clock,
        });
        true
    }
}

/// Dedicated worker engine serving an `Inkling2Model` (GGUF or Safetensors) with optional vision
/// and 3-tier prompt prefix caching (live GPU conv checkpoints, host RAM LRU snapshots, and
/// persistent on-disk `DiskCache`).
pub struct Inkling2Engine {
    model: Inkling2Model,
    vision: Option<Arc<Inkling2Vision>>,
    tok: Arc<Tokenizer>,
    specials: SpecialIds,
    max_seq: usize,
    covered: Vec<u64>,
    checkpoints: Vec<(Vec<u64>, Inkling2ConvSnapshot, bool)>,
    ram_cache: RamCache,
    pub(crate) disk: Option<DiskCache<Arc<Inkling2StateSnapshot>>>,
    image_disk_cache: bool,
    soft: VecDeque<(u64, Arc<Vec<f32>>)>,
    private_session: Option<Option<String>>,
    private_at: std::time::Instant,
    log: bool,
}

impl Inkling2Engine {
    pub fn new(
        model: Inkling2Model,
        vision: Option<Arc<Inkling2Vision>>,
        tok: Arc<Tokenizer>,
        max_seq: usize,
        log: bool,
    ) -> Self {
        let specials = SpecialIds::from_tokenizer(&tok);
        Self {
            model,
            vision,
            tok,
            specials,
            max_seq,
            covered: Vec::new(),
            checkpoints: Vec::new(),
            ram_cache: RamCache::default(),
            disk: None,
            image_disk_cache: false,
            soft: VecDeque::new(),
            private_session: None,
            private_at: std::time::Instant::now(),
            log,
        }
    }

    pub fn with_ram_budget(mut self, bytes: usize) -> Self {
        self.ram_cache = RamCache::new(bytes);
        self
    }

    pub fn with_disk(
        mut self,
        disk: Option<DiskCache<Arc<Inkling2StateSnapshot>>>,
        image_disk_cache: bool,
    ) -> Self {
        self.disk = disk;
        self.image_disk_cache = image_disk_cache;
        self
    }

    fn reset(&mut self) {
        self.private_session = None;
        self.model.reset();
        self.covered.clear();
        self.checkpoints.clear();
        self.soft.clear();
    }

    fn push_live_checkpoint(&mut self, keys: Vec<u64>, ckpt: Inkling2ConvSnapshot, base: bool) {
        if let Some(existing) = self.checkpoints.iter_mut().find(|(k, _, _)| *k == keys) {
            existing.1 = ckpt;
            existing.2 |= base;
            return;
        }
        self.checkpoints.push((keys, ckpt, base));
        while self.checkpoints.len() > MAX_LIVE_CHECKPOINTS {
            if let Some(idx) = self.checkpoints.iter().position(|(_, _, is_base)| !*is_base) {
                self.checkpoints.remove(idx);
            } else {
                self.checkpoints.remove(0);
            }
        }
    }

    /// Park the current live conversation into host RAM (and optionally disk) before switching
    /// to an unrelated prompt so returning to it later restores in ~15 ms instead of re-prefilling.
    fn park_displaced_live(&mut self, incoming: &[u64], incoming_forget: bool) {
        if incoming_forget || self.private_session.is_some() || self.covered.len() < MIN_SNAPSHOT_TOKENS {
            return;
        }
        let common = self
            .covered
            .iter()
            .zip(incoming)
            .take_while(|(a, b)| a == b)
            .count();
        // Only park when the incoming prompt diverges substantially (< 50% shared).
        if common * 2 >= self.covered.len() {
            return;
        }
        if self.ram_cache.has(&self.covered) {
            return;
        }
        let est = self.model.estimate_snapshot_bytes(self.covered.len());
        if est as usize > self.ram_cache.budget {
            return;
        }
        if let Ok(snap) = self.model.capture_state_snapshot() {
            if snap.pos == self.covered.len() {
                let snap = Arc::new(snap);
                self.ram_cache.insert(self.covered.clone(), snap, false);
            }
        }
    }

    fn store_snapshot_tiers(
        &mut self,
        keys: Vec<u64>,
        snap: Arc<Inkling2StateSnapshot>,
        base: bool,
        base_pos: Option<usize>,
        allow_disk: bool,
        forget: bool,
    ) {
        if forget || keys.len() < MIN_SNAPSHOT_TOKENS {
            return;
        }
        self.ram_cache.insert(keys.clone(), Arc::clone(&snap), base);
        if allow_disk {
            if let Some(disk) = self.disk.as_mut() {
                let est = snap.encoded_len() as u64;
                let near_base = !base
                    && base_pos.is_some_and(|bp| {
                        keys.len() >= bp
                            && keys.len() - bp < DISK_NEAR_TOKENS
                            && disk.has(&keys[..bp])
                    });
                let near_skip = near_base || (!base && disk.near(&keys, DISK_NEAR_TOKENS));
                if !near_skip
                    && !disk.has(&keys)
                    && disk.takes(est)
                    && disk.worth(&keys, est, base, &self.covered)
                {
                    disk.save(keys, snap, base);
                }
            }
        }
    }

    pub fn run(mut self, jobs: Receiver<Job>) {
        loop {
            if self.private_session.is_some()
                && self.private_at.elapsed() >= crate::engines::job::INCOGNITO_KEPT
            {
                self.reset();
            }
            let job = match self.private_session.as_ref().map(|_| {
                crate::engines::job::INCOGNITO_KEPT.saturating_sub(self.private_at.elapsed())
            }) {
                Some(left) => match jobs.recv_timeout(left) {
                    Ok(j) => j,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                },
                None => match jobs.recv() {
                    Ok(j) => j,
                    Err(_) => break,
                },
            };

            if job.wipe {
                if self.private_session.as_ref().is_some_and(|s| *s == job.session) {
                    self.reset();
                }
                let _ = job.events.send(Event::Done {
                    finish: Finish::Stop,
                    completion_tokens: 0,
                });
                continue;
            }
            if self
                .private_session
                .as_ref()
                .is_some_and(|s| !(job.forget && *s == job.session))
            {
                self.reset();
            }

            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.generate(&job)));
            let err = match res {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(e.to_string()),
                Err(p) => {
                    let said = p
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| p.downcast_ref::<&str>().copied())
                        .unwrap_or("a panic");
                    Some(format!("the Inkling-2 run failed: {said}"))
                }
            };
            if let Some(e) = err {
                if self.log {
                    eprintln!("  {e}");
                }
                self.reset();
                let _ = job.events.send(Event::Error(e));
            }
            if job.forget {
                self.private_session = Some(job.session.clone());
                self.private_at = std::time::Instant::now();
            }
        }
    }

    fn encode_pictures(&mut self, job: &Job, start: usize) -> Result<Vec<Option<Arc<Vec<f32>>>>> {
        let mut out = Vec::with_capacity(job.images.len());
        for (i, img) in job.images.iter().enumerate() {
            if img.start + img.prep.n_tokens() <= start {
                out.push(None);
                continue;
            }
            if !job.forget {
                if let Some((_, cached)) = self.soft.iter().find(|(h, _)| *h == img.hash) {
                    out.push(Some(Arc::clone(cached)));
                    continue;
                }
            }
            let (Some(tower), ImagePrep::Inkling2(prep)) = (&self.vision, &img.prep) else {
                return Err(oaiy_engine::Error::Arg(
                    "this model reads text only: no Inkling-2 vision tower is loaded".into(),
                ));
            };
            let t = std::time::Instant::now();
            let soft = tower
                .encode(prep)
                .map_err(|e| oaiy_engine::Error::Arg(format!("image {}: {e}", i + 1)))?;
            let expected = prep.n_tokens() * self.model.cfg.hidden;
            if soft.len() != expected {
                return Err(oaiy_engine::Error::Arg(format!(
                    "image {}: vision tower gave {} values, expected {expected}",
                    i + 1,
                    soft.len()
                )));
            }
            if self.log {
                eprintln!(
                    "  picture {} ({}x{} tokens) read in {:.2}s",
                    i + 1,
                    prep.cols,
                    prep.rows,
                    t.elapsed().as_secs_f64()
                );
            }
            let soft = Arc::new(soft);
            if !job.forget {
                self.soft.push_back((img.hash, Arc::clone(&soft)));
                if self.soft.len() > SOFT_KEPT {
                    self.soft.pop_front();
                }
            }
            out.push(Some(soft));
        }
        Ok(out)
    }

    fn generate(&mut self, job: &Job) -> Result<()> {
        if job.prompt.is_empty() {
            return Err(oaiy_engine::Error::Arg("inkling2: empty prompt".into()));
        }
        if !job.images.is_empty() && self.vision.is_none() {
            return Err(oaiy_engine::Error::Arg(
                "this model reads text only: no Inkling-2 vision tower is loaded".into(),
            ));
        }

        let keyed: Vec<u64> = {
            let mut k: Vec<u64> = job.prompt.iter().map(|&t| u64::from(t)).collect();
            for img in &job.images {
                for (off, id) in k[img.start..img.start + img.prep.n_tokens()]
                    .iter_mut()
                    .enumerate()
                {
                    *id = (1u64 << 63) ^ img.hash.rotate_left(17) ^ (off as u64);
                }
            }
            k
        };

        let cache_clock = std::time::Instant::now();
        let max_prefix = keyed.len().saturating_sub(1);
        let disk_limit = match job.images.iter().map(|i| i.start).min() {
            Some(first_img) if !self.image_disk_cache => first_img.min(max_prefix),
            _ => max_prefix,
        };

        // 1. Evaluate Tier-1 live GPU/CPU state & live multi-boundary conv checkpoints.
        let common = self
            .covered
            .iter()
            .zip(&keyed)
            .take_while(|(a, b)| a == b)
            .count();
        let swa = self.model.cfg.sliding_window;
        let swa_valid_from = self.model.swa_valid_from();
        // Prune live checkpoints whose SWA rows were already overwritten or dropped during SWA grow/resync.
        self.checkpoints
            .retain(|(_, ckpt, _)| ckpt.pos.saturating_sub(swa) >= swa_valid_from);
        let live_exact = if common == self.covered.len() && common < keyed.len() {
            common
        } else {
            0
        };
        let best_live_ckpt = self
            .checkpoints
            .iter()
            .enumerate()
            .filter(|(_, (saved, ckpt, _))| {
                saved.len() > live_exact
                    && saved.len() <= common
                    && saved.len() < keyed.len()
                    && ckpt.pos.saturating_sub(swa) >= swa_valid_from
                    && keyed.starts_with(saved)
            })
            .max_by_key(|(_, (saved, _, _))| saved.len())
            .map(|(idx, (saved, _, _))| (idx, saved.len()));
        let live_potential = best_live_ckpt.map_or(live_exact, |(_, len)| len);

        // 2. Check Tier-2 Host RAM cache & Tier-3 persistent DiskCache for a strictly longer match.
        let ram_candidate = if !job.forget {
            self.ram_cache.best(&keyed, max_prefix, live_potential)
        } else {
            None
        };
        let after_ram = ram_candidate
            .as_ref()
            .map_or(live_potential, |(k, _)| k.len());
        let disk_candidate = if !job.forget && disk_limit > after_ram {
            self.disk
                .as_mut()
                .and_then(|d| d.load_best(&keyed, disk_limit, after_ram, self.log))
        } else {
            None
        };

        let (mut start, mut source) = (0usize, "none");

        if let Some((saved_keys, snap)) = disk_candidate {
            self.park_displaced_live(&keyed, job.forget);
            match self.model.restore_state_snapshot(&snap) {
                Ok(()) => {
                    let conv = snap.to_conv_snapshot();
                    let is_base = base_boundary(&job.prompt, self.specials) == Some(snap.pos);
                    self.checkpoints.clear();
                    self.push_live_checkpoint(saved_keys.clone(), conv, is_base);
                    self.ram_cache
                        .insert(saved_keys.clone(), Arc::clone(&snap), is_base);
                    self.covered = saved_keys;
                    start = snap.pos;
                    source = "disk";
                }
                Err(e) => {
                    if self.log {
                        eprintln!("  Inkling-2 disk prompt state skipped: {e}");
                    }
                }
            }
        }

        if start == 0 {
            if let Some((saved_keys, snap)) = ram_candidate {
                self.park_displaced_live(&keyed, job.forget);
                match self.model.restore_state_snapshot(&snap) {
                    Ok(()) => {
                        let conv = snap.to_conv_snapshot();
                        let is_base = base_boundary(&job.prompt, self.specials) == Some(snap.pos);
                        self.checkpoints.clear();
                        self.push_live_checkpoint(saved_keys.clone(), conv, is_base);
                        self.covered = saved_keys;
                        start = snap.pos;
                        source = "ram";
                    }
                    Err(e) => {
                        if self.log {
                            eprintln!("  Inkling-2 RAM prompt state skipped: {e}");
                        }
                    }
                }
            }
        }

        if start == 0 {
            if let Some((idx, _)) = best_live_ckpt {
                let (saved_keys, ckpt, _) = self.checkpoints[idx].clone();
                if self.model.restore_conv_checkpoint(&ckpt) {
                    self.covered = saved_keys;
                    self.checkpoints
                        .retain(|(k, _, _)| k.len() <= self.covered.len() && keyed.starts_with(k));
                    start = self.covered.len();
                    source = "checkpoint";
                }
            } else if live_exact > 0 {
                self.checkpoints
                    .retain(|(k, _, _)| k.len() <= live_exact && keyed.starts_with(k));
                start = live_exact;
                source = "memory";
            }
        }

        if start == 0 {
            self.park_displaced_live(&keyed, job.forget);
            self.model.reset();
            self.covered.clear();
            self.checkpoints.clear();
        }

        let _ = job.events.send(Event::CacheReuse {
            cached: start,
            source,
            common,
        });
        if self.log && (start > 0 || self.disk.is_some()) {
            eprintln!(
                "  Inkling-2 cache: {start}/{} tokens from {source} (common {common}) in {:.3}s",
                keyed.len(),
                cache_clock.elapsed().as_secs_f64()
            );
        }

        // Encode only images that extend past the cached prefix (cached image spans are skipped!).
        let pictures = self.encode_pictures(job, start)?;

        let started = std::time::Instant::now();
        let total = keyed.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });

        let base_pos = base_boundary(&job.prompt, self.specials);
        let need_base_snapshot = base_pos.is_some_and(|bp| {
            bp > start
                && !job.forget
                && (!self.ram_cache.has(&keyed[..bp])
                    || self.disk.as_ref().is_some_and(|d| {
                        bp <= disk_limit
                            && !d.has(&keyed[..bp])
                            && d.takes(self.model.estimate_snapshot_bytes(bp))
                    }))
        });
        let stops = checkpoint_positions(&job.prompt, self.specials);
        let keep_at = keyed.len().saturating_sub(1);

        let mut pos = start;
        let mut logits = Vec::new();
        let d = self.model.cfg.hidden;
        let chunk = self.model.preferred_prefill_chunk().max(1);
        let mut keep_at_ckpt: Option<Inkling2ConvSnapshot> = None;

        while pos < keyed.len() {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done {
                    finish: Finish::Stop,
                    completion_tokens: 0,
                });
                return Ok(());
            }

            // Stop cleanly at `base_pos` (only on first encounter, so system+tools gets pinned)
            // and at `keep_at = keyed.len() - 1` (so exact retries & post-turn disk snapshots work).
            let mut next_stop = (pos + chunk).min(keyed.len());
            if need_base_snapshot {
                if let Some(bp) = base_pos {
                    if bp > pos && bp < next_stop {
                        next_stop = bp;
                    }
                }
            }
            if keep_at > pos && keep_at < next_stop {
                next_stop = keep_at;
            }
            // Also record lightweight conv checkpoints on turn boundaries when crossing them
            // without breaking large chunks into tiny fragments (only when chunk is already small
            // or boundary matches a stop).
            let take = next_stop - pos;
            let ids = &job.prompt[pos..pos + take];
            let soft: Vec<(usize, &[f32])> = job
                .images
                .iter()
                .zip(&pictures)
                .filter_map(|(img, rows_opt)| {
                    let rows = rows_opt.as_ref()?;
                    let lo = img.start.max(pos);
                    let hi = (img.start + img.prep.n_tokens()).min(pos + take);
                    (lo < hi).then(|| (lo - pos, &rows[(lo - img.start) * d..(hi - img.start) * d]))
                })
                .collect();
            let is_final = pos + take >= keyed.len();
            if is_final {
                logits = self
                    .model
                    .forward_last(ids, &soft)
                    .map_err(|e| oaiy_engine::Error::Arg(e.to_string()))?;
            } else {
                self.model
                    .prefill_only(ids, &soft)
                    .map_err(|e| oaiy_engine::Error::Arg(e.to_string()))?;
            }
            self.covered.extend_from_slice(&keyed[pos..pos + take]);
            pos += take;

            // Pin the base system+tools prefix snapshot the very first time we cross it.
            if need_base_snapshot && base_pos == Some(pos) {
                if let Ok(snap) = self.model.capture_state_snapshot() {
                    if snap.pos == pos {
                        let conv = snap.to_conv_snapshot();
                        let keys_prefix = keyed[..pos].to_vec();
                        self.push_live_checkpoint(keys_prefix.clone(), conv, true);
                        self.store_snapshot_tiers(
                            keys_prefix,
                            Arc::new(snap),
                            true,
                            base_pos,
                            pos <= disk_limit,
                            job.forget,
                        );
                    }
                }
            }

            // Capture lightweight (~0.4 ms) conv checkpoint at `keep_at = keyed.len() - 1` and
            // at the end of prefill (`keyed.len()`).
            if pos == keep_at && keep_at > 0 {
                if let Some(conv) = self.model.capture_conv_checkpoint() {
                    self.push_live_checkpoint(keyed[..pos].to_vec(), conv.clone(), base_pos == Some(pos));
                    keep_at_ckpt = Some(conv);
                }
            } else if stops.contains(&pos) && pos != keyed.len() && base_pos != Some(pos) {
                if let Some(conv) = self.model.capture_conv_checkpoint() {
                    self.push_live_checkpoint(keyed[..pos].to_vec(), conv, false);
                }
            }

            let _ = job.events.send(Event::Progress {
                done: pos - start,
                total,
            });
        }

        let prefill_s = started.elapsed().as_secs_f64();
        if let Some(conv) = self.model.capture_conv_checkpoint() {
            self.push_live_checkpoint(keyed.clone(), conv, false);
        }
        let _ = job.events.send(Event::Prefilled { cached: start });

        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        let has_tools = !job.tools.is_empty();
        let mut translator = Inkling2Translator::with_tools(self.specials, has_tools);
        let mut budget = ThinkBudget::new(false, self.specials.end_message, job.think_budget)
            .closing(&job.think_closing)
            .opens_with(self.specials.content_thinking);
        let mut forced_after_think: VecDeque<u32> = VecDeque::new();
        let mut think_budget_exhausted = false;
        let mut thinking_done_pending: Option<(usize, Option<usize>)> = None;
        let mut reply_hist: VecDeque<u32> = VecDeque::new();
        let max_seq = self.model.cfg.context_length.min(self.max_seq);

        let mut n = 0usize;
        let mut keep_at_persisted = false;
        let decode_start = std::time::Instant::now();
        let finish = loop {
            if translator.in_text() && !reply_hist.is_empty() {
                penalize_reply(&mut logits, &reply_hist, &job.sampling);
            }
            let next = if let Some(forced_tok) = forced_after_think.pop_front() {
                forced_tok
            } else {
                let room_left = n + 4 < job.max_tokens && pos + 4 < max_seq;
                mask_post_thinking_logits(
                    &mut logits,
                    self.specials,
                    &translator,
                    room_left,
                    think_budget_exhausted,
                );
                let sampled = sample(&logits, &job.sampling, &mut rng);
                let was_thinking = budget.thinking;
                let (tok_next, forced) = budget.pass(sampled);
                if forced && !budget.thinking {
                    think_budget_exhausted = true;
                    if self.log {
                        eprintln!(
                            "  reasoning ended at its budget ({} tokens)",
                            job.think_budget.unwrap_or(0)
                        );
                    }
                    enqueue_forced_reply_opener(&mut forced_after_think, self.specials, has_tools);
                }
                if budget.reports(was_thinking) {
                    if !budget.thinking {
                        // Defer `Event::Thinking { done: true }` until a non-thinking segment
                        // actually opens (matching `Inkling2Translator`'s deferred `</think>`),
                        // because `api.rs` calls `parser.end_reasoning()` on `done: true`.
                        thinking_done_pending = Some((budget.used, budget.budget));
                    } else {
                        let _ = job.events.send(Event::Thinking {
                            used: budget.used,
                            budget: budget.budget,
                            done: false,
                        });
                    }
                }
                tok_next
            };

            let (chunks, stop) = translator.push(next, &self.tok);
            let mut disconnected = false;
            for chunk in chunks {
                if job.events.send(Event::Text(chunk)).is_err() {
                    disconnected = true;
                    break;
                }
            }
            if let Some((used, b)) = thinking_done_pending {
                if translator.closed_thinking() {
                    thinking_done_pending = None;
                    let _ = job.events.send(Event::Thinking {
                        used,
                        budget: b,
                        done: true,
                    });
                }
            }
            if stop || disconnected {
                break Finish::Stop;
            }
            n += 1;
            if translator.in_text() && job.sampling.repeat_last_n > 0 {
                reply_hist.push_back(next);
                while reply_hist.len() > job.sampling.repeat_last_n {
                    reply_hist.pop_front();
                }
            }

            if n >= job.max_tokens || pos + 1 >= max_seq {
                break Finish::Length;
            }
            if job.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }

            // Before decode wraps the SWA ring buffer past `keep_at_ckpt`, snapshot `keep_at`
            // into host RAM & DiskCache so long thinking turns NEVER lose their prompt cache.
            if !keep_at_persisted && !job.forget && keep_at >= MIN_SNAPSHOT_TOKENS {
                if let Some(ref ckpt) = keep_at_ckpt {
                    let slack = self.model.swa_checkpoint_slack();
                    if pos.saturating_sub(ckpt.pos) + 2 >= slack {
                        if let Some(snap) = self.model.capture_snapshot_at_conv_checkpoint(ckpt) {
                            let is_base = base_pos == Some(keep_at);
                            self.store_snapshot_tiers(
                                keyed[..keep_at].to_vec(),
                                Arc::new(snap),
                                is_base,
                                base_pos,
                                keep_at <= disk_limit,
                                false,
                            );
                        }
                        keep_at_persisted = true;
                    }
                }
            }

            logits = self
                .model
                .forward_last(&[next], &[])
                .map_err(|e| oaiy_engine::Error::Arg(e.to_string()))?;
            self.covered.push(u64::from(next));
            pos += 1;
        };

        for chunk in translator.finish(&self.tok) {
            let _ = job.events.send(Event::Text(chunk));
        }
        if let Some(err) = translator.tool_error() {
            if !job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Error(format!(
                    "tool_contract_error: {err}; no tool from this batch was executed"
                )));
                return Ok(());
            }
        }

        let decode_s = decode_start.elapsed().as_secs_f64();
        if self.log {
            eprintln!(
                "  {} prompt tokens ({start} reused from {source}) in {prefill_s:.1}s; {n} generated in {decode_s:.1}s ({:.1} tok/s)",
                keyed.len(),
                n as f64 / decode_s.max(1e-9)
            );
            #[cfg(feature = "webgpu")]
            if ggml_rs_wgpu::profile::chain_on() {
                let kernels = ggml_rs_wgpu::profile::take_kernels();
                let gpu_ms: f64 = kernels.iter().map(|(_, ms, _)| *ms).sum();
                let line: Vec<String> = kernels
                    .iter()
                    .take(20)
                    .map(|(name, ms, c)| format!("{name} {ms:.1}ms ({c})"))
                    .collect();
                eprintln!("  gpu kernels ({gpu_ms:.1}ms): {}", line.join(", "));
            }
        }

        let _ = job.events.send(Event::Done {
            finish,
            completion_tokens: n,
        });

        // Persist `keep_at` into RAM & DiskCache AFTER sending `Event::Done` so streaming
        // TTFT and token generation have zero added latency.
        if !keep_at_persisted
            && !job.forget
            && keep_at >= MIN_SNAPSHOT_TOKENS
            && keep_at > start
        {
            if let Some(ref ckpt) = keep_at_ckpt {
                let need_ram = !self.ram_cache.has(&keyed[..keep_at]);
                    let need_disk = keep_at <= disk_limit
                    && self.disk.as_ref().is_some_and(|d| {
                        let is_base = base_pos == Some(keep_at);
                        let est = self.model.estimate_snapshot_bytes(keep_at);
                        let near_base = !is_base
                            && base_pos.is_some_and(|bp| {
                                keep_at >= bp
                                    && keep_at - bp < DISK_NEAR_TOKENS
                                    && d.has(&keyed[..bp])
                            });
                        let near_skip = near_base || (!is_base && d.near(&keyed[..keep_at], DISK_NEAR_TOKENS));
                        !near_skip && !d.has(&keyed[..keep_at]) && d.takes(est)
                    });
                if need_ram || need_disk {
                    if let Some(snap) = self.model.capture_snapshot_at_conv_checkpoint(ckpt) {
                        let is_base = base_pos == Some(keep_at);
                        self.store_snapshot_tiers(
                            keyed[..keep_at].to_vec(),
                            Arc::new(snap),
                            is_base,
                            base_pos,
                            keep_at <= disk_limit,
                            false,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::chat::{Delta, StreamParser};

    fn test_tokenizer() -> Tokenizer {
        // Build the standard 256-entry GPT-2 byte-level BPE alphabet so every ASCII/UTF-8 byte encodes cleanly.
        let mut byte_to_char = ['\0'; 256];
        let mut extra = 256u32;
        for b in 0..=255u8 {
            let direct = (b'!'..=b'~').contains(&b)
                || (0xA1..=0xAC).contains(&b)
                || (0xAE..=0xFF).contains(&b);
            if direct {
                byte_to_char[b as usize] = b as char;
            } else {
                byte_to_char[b as usize] = char::from_u32(extra).unwrap();
                extra += 1;
            }
        }
        let mut toks: Vec<String> = byte_to_char.iter().map(|c| c.to_string()).collect();
        let specials_str = [
            MSG_USER,
            MSG_MODEL,
            MSG_SYSTEM,
            MSG_TOOL,
            CONTENT_TEXT,
            CONTENT_THINKING,
            CONTENT_IMAGE,
            CONTENT_INVOKE_TOOL_JSON,
            END_MESSAGE,
            END_SAMPLING,
            IMAGE_PLACEHOLDER,
        ];
        let mut special_ids = Vec::new();
        for s in specials_str {
            let id = toks.len() as u32;
            toks.push(s.to_string());
            special_ids.push(id);
        }
        let eos = toks.iter().position(|t| t == END_SAMPLING).unwrap() as u32;
        Tokenizer::from_gpt4o_bpe_parts(toks, vec![], special_ids, eos, None).unwrap()
    }

    #[test]
    fn chat_prompt_matches_official_jinja_template() {
        let tools = Json::Arr(vec![Json::obj([
            ("type", Json::str("function")),
            (
                "function",
                Json::obj([
                    ("name", Json::str("get_weather")),
                    ("description", Json::str("Look up weather")),
                    ("parameters", Json::obj([("type", Json::str("object"))])),
                ]),
            ),
        ])]);
        let msgs = vec![
            Json::obj([
                ("role", Json::str("system")),
                ("content", Json::str("Be helpful.")),
                ("tools", tools),
            ]),
            Json::obj([("role", Json::str("user")), ("content", Json::str("Weather in Paris?"))]),
            Json::obj([
                ("role", Json::str("assistant")),
                ("content", Json::Null),
                (
                    "tool_calls",
                    Json::Arr(vec![Json::obj([
                        ("id", Json::str("call_1")),
                        ("type", Json::str("function")),
                        (
                            "function",
                            Json::obj([
                                ("name", Json::str("get_weather")),
                                ("arguments", Json::str(r#"{"unit":"c","city":"Paris"}"#)),
                            ]),
                        ),
                    ])]),
                ),
            ]),
            Json::obj([
                ("role", Json::str("tool")),
                ("tool_call_id", Json::str("call_1")),
                ("content", Json::str("18C sunny")),
            ]),
        ];

        let enc = chat_prompt(
            &msgs,
            &Options {
                mode: Mode::Thinking,
                effort: 75,
                drop_thinking: true,
            },
            false,
        )
        .unwrap();

        let expected = concat!(
            "<|message_system|>tool_declare<|content_xml|>",
            r#"[{"description":"Look up weather","name":"get_weather","parameters":{"type":"object"},"type":"function"}]"#,
            "<|end_message|>",
            "<|message_system|><|content_text|>Be helpful.<|end_message|>",
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
            "<|message_user|><|content_text|>Weather in Paris?<|end_message|>",
            "<|message_model|>get_weather<|content_invoke_tool_json|>",
            r#"{"name":"get_weather","args":{"city":"Paris","unit":"c"}}"#,
            "<|end_message|><|content_model_end_sampling|>",
            "<|message_tool|>get_weather<|content_text|>18C sunny<|end_message|>",
        );
        assert_eq!(enc.prompt, expected);
    }

    #[test]
    fn translator_streams_thinking_text_and_parallel_tool_calls_through_stream_parser() {
        let tok = test_tokenizer();
        let ids = SpecialIds::from_tokenizer(&tok);

        // 1. Thinking + Answer in Thinking mode
        let raw = format!(
            "{MSG_MODEL}{CONTENT_THINKING}Let me check.{END_MESSAGE}{MSG_MODEL}{CONTENT_TEXT}Paris.{END_MESSAGE}{END_SAMPLING}"
        );
        let tokens = tok.encode(&raw, false).unwrap();
        let mut tr = Inkling2Translator::new(ids);
        let mut parser = StreamParser::new(Mode::Thinking);
        let (mut reasoning, mut content) = (String::new(), String::new());
        for id in tokens {
            let (chunks, stop) = tr.push(id, &tok);
            for c in chunks {
                for d in parser.push(&c) {
                    match d {
                        Delta::Reasoning(s) => reasoning.push_str(&s),
                        Delta::Content(s) => content.push_str(&s),
                    }
                }
            }
            if stop {
                break;
            }
        }
        let (rest, calls) = parser.finish();
        for d in rest {
            match d {
                Delta::Reasoning(s) => reasoning.push_str(&s),
                Delta::Content(s) => content.push_str(&s),
            }
        }
        assert_eq!(reasoning, "Let me check.");
        assert_eq!(content, "Paris.");
        assert!(calls.is_empty());

        // 2. Direct answer without thinking in Thinking mode (effort 0 or model skips thinking)
        let raw_direct = format!("{MSG_MODEL}{CONTENT_TEXT}42{END_MESSAGE}{END_SAMPLING}");
        let mut tr2 = Inkling2Translator::new(ids);
        let mut p2 = StreamParser::new(Mode::Thinking);
        let (mut r2, mut c2) = (String::new(), String::new());
        for id in tok.encode(&raw_direct, false).unwrap() {
            let (chunks, stop) = tr2.push(id, &tok);
            for c in chunks {
                for d in p2.push(&c) {
                    match d {
                        Delta::Reasoning(s) => r2.push_str(&s),
                        Delta::Content(s) => c2.push_str(&s),
                    }
                }
            }
            if stop {
                break;
            }
        }
        let (rest, _) = p2.finish();
        for d in rest {
            if let Delta::Content(s) = d {
                c2.push_str(&s);
            }
        }
        assert_eq!(r2, "");
        assert_eq!(c2, "42");

        // 3. Parallel tool calls
        let raw_tools = format!(
            "{MSG_MODEL}add<|content_invoke_tool_json|>{{\"name\":\"add\",\"args\":{{\"a\":1,\"b\":2}}}}{END_MESSAGE}\
             {MSG_MODEL}say<|content_invoke_tool_json|>{{\"name\":\"say\",\"args\":{{\"msg\":\"hi\"}}}}{END_MESSAGE}{END_SAMPLING}"
        );
        let mut tr3 = Inkling2Translator::new(ids);
        let mut p3 = StreamParser::new(Mode::Chat);
        for id in tok.encode(&raw_tools, false).unwrap() {
            let (chunks, stop) = tr3.push(id, &tok);
            for c in chunks {
                p3.push(&c);
            }
            if stop {
                break;
            }
        }
        let (_, calls3) = p3.finish();
        assert_eq!(calls3.len(), 2);
        assert_eq!(calls3[0].name, "add");
        assert_eq!(
            Json::parse(calls3[0].arguments.as_bytes()).unwrap(),
            Json::parse(br#"{"a":1,"b":2}"#).unwrap()
        );
        assert_eq!(calls3[1].name, "say");
        assert_eq!(
            Json::parse(calls3[1].arguments.as_bytes()).unwrap(),
            Json::parse(br#"{"msg":"hi"}"#).unwrap()
        );

        // 4. Thinking-only stop keeps StreamParser in reasoning so api.rs can recover an answer
        // written inside `<|content_thinking|>`, even when initialized in Chat mode (since
        // Inkling-2 prompts do not open `<think>`).
        let raw_thought_only = format!(
            "{MSG_MODEL}{CONTENT_THINKING}Only thinking text.{END_MESSAGE}{END_SAMPLING}"
        );
        let mut tr4 = Inkling2Translator::new(ids);
        let mut p4 = StreamParser::new(Mode::Chat);
        let (mut r4, mut c4) = (String::new(), String::new());
        for id in tok.encode(&raw_thought_only, false).unwrap() {
            let (chunks, stop) = tr4.push(id, &tok);
            for c in chunks {
                for d in p4.push(&c) {
                    match d {
                        Delta::Reasoning(s) => r4.push_str(&s),
                        Delta::Content(s) => c4.push_str(&s),
                    }
                }
            }
            if stop {
                break;
            }
        }
        assert!(p4.is_reasoning(), "thinking-only completion must remain in reasoning phase");
        let (rest4, calls4) = p4.finish();
        for d in rest4 {
            match d {
                Delta::Reasoning(s) => r4.push_str(&s),
                Delta::Content(s) => c4.push_str(&s),
            }
        }
        assert_eq!(r4, "Only thinking text.");
        assert_eq!(c4, "");
        assert!(calls4.is_empty());

        // 5. Thinking + Text when StreamParser starts in Chat mode (matching Flavour::opens_thought == false)
        let mut tr5 = Inkling2Translator::new(ids);
        let mut p5 = StreamParser::new(Mode::Chat);
        let (mut r5, mut c5) = (String::new(), String::new());
        for id in tok.encode(&raw, false).unwrap() {
            let (chunks, stop) = tr5.push(id, &tok);
            for c in chunks {
                for d in p5.push(&c) {
                    match d {
                        Delta::Reasoning(s) => r5.push_str(&s),
                        Delta::Content(s) => c5.push_str(&s),
                    }
                }
            }
            if stop {
                break;
            }
        }
        assert!(!p5.is_reasoning());
        let (rest5, _) = p5.finish();
        for d in rest5 {
            match d {
                Delta::Reasoning(s) => r5.push_str(&s),
                Delta::Content(s) => c5.push_str(&s),
            }
        }
        assert_eq!(r5, "Let me check.");
        assert_eq!(c5, "Paris.");
    }

    #[test]
    fn ram_cache_pins_base_entries_and_selects_longest_prefix() {
        let make_snap = |pos: usize, floats: usize| {
            Arc::new(Inkling2StateSnapshot {
                pos,
                hidden: 4,
                conv_kernel: 2,
                sliding_window: 4,
                slots: vec![crate::llm::arch::inkling2::state::Inkling2SlotSnapshot {
                    is_swa: false,
                    kv_dim: 2,
                    base_pos: 0,
                    stored_rows: floats,
                    k_cache: vec![0.0; floats * 2],
                    v_cache: vec![0.0; floats * 2],
                    k_conv: vec![0.0; 2],
                    v_conv: vec![0.0; 2],
                    attn_post_conv: vec![0.0; 4],
                    ffn_post_conv: vec![0.0; 4],
                }],
            })
        };
        // Budget holds ~2 entries (each ~224 bytes).
        let s1 = make_snap(3, 10);
        let s2 = make_snap(5, 10);
        let s3 = make_snap(6, 10);
        let unit = s1.bytes() + 6 * 8;
        let mut cache = RamCache::new(unit * 2 + 32);

        // Insert base prefix [1,2,3] and conversation [1,2,3,4,5].
        assert!(cache.insert(vec![1, 2, 3], s1, true));
        assert!(cache.insert(vec![1, 2, 3, 4, 5], s2, false));

        // Longest matching prefix for [1,2,3,4,5,99] is [1,2,3,4,5].
        let (k, snap) = cache.best(&[1, 2, 3, 4, 5, 99], 5, 0).unwrap();
        assert_eq!(k, vec![1, 2, 3, 4, 5]);
        assert_eq!(snap.pos, 5);

        // Inserting a third conversation [1,2,3,7,8,9] evicts the non-base entry [1,2,3,4,5],
        // NOT the protected base entry [1,2,3]!
        assert!(cache.insert(vec![1, 2, 3, 7, 8, 9], s3, false));
        assert!(cache.has(&[1, 2, 3]), "base system-prompt entry must survive conversation eviction");
        assert!(!cache.has(&[1, 2, 3, 4, 5]), "older non-base conversation entry should evict first");
        assert!(cache.has(&[1, 2, 3, 7, 8, 9]));

        // A brand-new conversation sharing [1,2,3] still hits the pinned base entry!
        let (k_new, _) = cache.best(&[1, 2, 3, 42, 43], 4, 0).unwrap();
        assert_eq!(k_new, vec![1, 2, 3]);
    }

    #[test]
    fn base_and_checkpoint_boundaries_find_system_and_turn_markers() {
        let tok = test_tokenizer();
        let ids = SpecialIds::from_tokenizer(&tok);
        let msg_u = ids.msg_user.unwrap();
        let msg_t = ids.msg_tool.unwrap();

        let mut prompt = vec![10u32; 80]; // 80-token system+tools prefix
        prompt.push(msg_u); // index 80: first user turn
        prompt.extend_from_slice(&[11, 12, 13]);
        prompt.push(msg_t); // index 84: tool result
        prompt.extend_from_slice(&[14, 15]);
        prompt.push(msg_u); // index 87: second user turn
        prompt.extend_from_slice(&[16, 17, 18]);

        assert_eq!(base_boundary(&prompt, ids), Some(80));
        let stops = checkpoint_positions(&prompt, ids);
        assert_eq!(stops, vec![80, 84, 87, prompt.len() - 1]);
    }

    #[test]
    fn translator_recovers_from_invalid_utf8_and_streams_split_multibyte_chars() {
        let tok = test_tokenizer();
        let ids = SpecialIds::from_tokenizer(&tok);
        let mut tr = Inkling2Translator::new(ids);

        // Open text segment.
        let _ = tr.push(ids.msg_model.unwrap(), &tok);
        let _ = tr.push(ids.content_text.unwrap(), &tok);

        // Token 0xFF is an invalid UTF-8 lead byte in the byte-level BPE alphabet (256-entry table).
        let (bad_chunks, stop) = tr.push(0xFFu32, &tok);
        assert!(!stop);
        assert_eq!(bad_chunks.join(""), "\u{FFFD}");

        // Subsequent valid ASCII tokens must stream immediately (no freeze).
        let mut streamed = String::new();
        for b in b"ok" {
            let (chunks, stop) = tr.push(u32::from(*b), &tok);
            assert!(!stop);
            for c in chunks {
                streamed.push_str(&c);
            }
        }
        assert_eq!(streamed, "ok");

        // Split 4-byte emoji ("🚀" = F0 9F 9A 80) across individual byte tokens.
        let emoji = "🚀".as_bytes();
        let mut emoji_out = String::new();
        for (idx, &b) in emoji.iter().enumerate() {
            let (chunks, stop) = tr.push(u32::from(b), &tok);
            assert!(!stop);
            for c in chunks {
                emoji_out.push_str(&c);
            }
            if idx < 3 {
                assert!(emoji_out.is_empty(), "incomplete UTF-8 must be buffered until complete");
            }
        }
        assert_eq!(emoji_out, "🚀");

        // Trailing incomplete sequence flushed on finish().
        let (c1, _) = tr.push(0xF0u32, &tok);
        assert!(c1.is_empty());
        let tail = tr.finish(&tok);
        assert_eq!(tail.join(""), "\u{FFFD}");
    }

    #[test]
    fn translator_handles_missing_end_message_string_arguments_dsml_rejection_and_malformed_json() {
        let tok = test_tokenizer();
        let ids = SpecialIds::from_tokenizer(&tok);

        // 1. Back-to-back tool calls without `<|end_message|>` plus JSON-encoded string `"arguments"`.
        let raw = format!(
            "{MSG_MODEL}add{CONTENT_INVOKE_TOOL_JSON}{{\"name\":\"add\",\"arguments\":\"{{\\\"a\\\":1,\\\"b\\\":2}}\"}}\
             {CONTENT_INVOKE_TOOL_JSON}{{\"name\":\"say\",\"args\":{{\"msg\":\"hi\"}}}}{CONTENT_TEXT}Done.{END_SAMPLING}"
        );
        let mut tr = Inkling2Translator::new(ids);
        let mut p = StreamParser::new(Mode::Chat);
        for id in tok.encode(&raw, false).unwrap() {
            let (chunks, stop) = tr.push(id, &tok);
            for c in chunks {
                p.push(&c);
            }
            if stop {
                break;
            }
        }
        assert_eq!(tr.tool_error(), None);
        let (_, calls) = p.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "add");
        assert_eq!(
            Json::parse(calls[0].arguments.as_bytes()).unwrap(),
            Json::parse(br#"{"a":1,"b":2}"#).unwrap()
        );
        assert_eq!(calls[1].name, "say");
        assert_eq!(
            Json::parse(calls[1].arguments.as_bytes()).unwrap(),
            Json::parse(br#"{"msg":"hi"}"#).unwrap()
        );

        // 2. Malformed tool JSON must surface `tool_error()` and suppress emitting partial calls.
        let malformed = format!(
            "{MSG_MODEL}add{CONTENT_INVOKE_TOOL_JSON}{{\"name\":\"add\",\"args\":{{\"a\":1}}}}{END_MESSAGE}\
             {MSG_MODEL}bad{CONTENT_INVOKE_TOOL_JSON}{{not valid json{END_MESSAGE}{END_SAMPLING}"
        );
        let mut tr_bad = Inkling2Translator::new(ids);
        let mut out_bad = String::new();
        for id in tok.encode(&malformed, false).unwrap() {
            let (chunks, stop) = tr_bad.push(id, &tok);
            for c in chunks {
                out_bad.push_str(&c);
            }
            if stop {
                break;
            }
        }
        assert!(tr_bad.tool_error().is_some(), "malformed tool JSON must set tool_error");
        assert!(!out_bad.contains(DSML), "broken batch must not emit partial DSML calls: {out_bad}");

        // 3. Reject DSML (including Unicode-escaped `\uFF5C`) and quotes in name/key/rendered.
        for (hdr, payload) in [
            ("", r#"{"name":"evil\" x=\"1","args":{}}"#),
            ("", r#"{"name":"ok","args":{"bad\"k":"v"}}"#),
            ("", r#"{"name":"ok","args":{"x":"\uFF5CDSML\uFF5C"}}"#),
            ("", r#"{"name":"ok","arguments":"not json"}"#),
            ("", r#"{"name":"ok","args":[1,2]}"#),
            ("evil\"hdr", r#"{"args":{}}"#),
        ] {
            assert!(
                format_tool_call_as_dsml(hdr, payload).is_err(),
                "expected error for header={hdr:?} payload={payload}"
            );
        }
    }

    #[test]
    fn translator_emits_plain_text_tool_calls_when_no_tools_catalog_is_declared() {
        let tok = test_tokenizer();
        let ids = SpecialIds::from_tokenizer(&tok);

        // 1. Valid Inkling-2 JSON tool call with no declared OpenAI tools must emit plain-text
        // `<tool_call>` markup (never DSML, which would fail `validate_tools` with `missing tool catalog`).
        let raw_valid = format!(
            "{MSG_MODEL}{CONTENT_THINKING}Let me inspect index.html.{END_MESSAGE}\
             {MSG_MODEL}read_file{CONTENT_INVOKE_TOOL_JSON}{{\"name\":\"read_file\",\"args\":{{\"path\":\"index.html\"}}}}{END_MESSAGE}{END_SAMPLING}"
        );
        let mut tr = Inkling2Translator::with_tools(ids, false);
        let mut p = StreamParser::new(Mode::Chat);
        let (mut reasoning, mut content) = (String::new(), String::new());
        for id in tok.encode(&raw_valid, false).unwrap() {
            let (chunks, stop) = tr.push(id, &tok);
            for c in chunks {
                assert!(!c.contains(DSML), "text-protocol output must never emit DSML: {c}");
                for d in p.push(&c) {
                    match d {
                        Delta::Reasoning(s) => reasoning.push_str(&s),
                        Delta::Content(s) => content.push_str(&s),
                    }
                }
            }
            if stop {
                break;
            }
        }
        assert_eq!(tr.tool_error(), None);
        assert!(!p.tool_calls_ready());
        let (rest, calls) = p.finish();
        for d in rest {
            match d {
                Delta::Reasoning(s) => reasoning.push_str(&s),
                Delta::Content(s) => content.push_str(&s),
            }
        }
        assert!(calls.is_empty());
        assert_eq!(reasoning, "Let me inspect index.html.");
        assert_eq!(
            content,
            "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"index.html\"}}\n</tool_call>"
        );

        // 2. Malformed JSON in `<|content_invoke_tool_json|>` without a tool catalog must pass
        // through as plain-text `<tool_call>` instead of failing with `tool_contract_error`.
        let raw_bad = format!(
            "{MSG_MODEL}edit_file{CONTENT_INVOKE_TOOL_JSON}{{\"name\":\"edit_file\" broken}}{END_MESSAGE}{END_SAMPLING}"
        );
        let mut tr_bad = Inkling2Translator::with_tools(ids, false);
        let mut out_bad = String::new();
        for id in tok.encode(&raw_bad, false).unwrap() {
            let (chunks, stop) = tr_bad.push(id, &tok);
            for c in chunks {
                out_bad.push_str(&c);
            }
            if stop {
                break;
            }
        }
        assert_eq!(tr_bad.tool_error(), None);
        assert!(!out_bad.contains(DSML));
        assert!(out_bad.contains("<tool_call name=\"edit_file\">"), "{out_bad}");
        assert!(out_bad.contains("{\"name\":\"edit_file\" broken}"), "{out_bad}");

        // 3. Header-only tool name + arguments object, flat args alongside "name", and already-wrapped `<tool_call>` markup.
        assert_eq!(
            format_tool_call_as_text("read_file", r#"{"path":"main.js"}"#).unwrap().as_deref(),
            Some("<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"main.js\"}}\n</tool_call>")
        );
        assert_eq!(
            format_tool_call_as_text("", r#"{"name":"read_file","path":"main.js"}"#).unwrap().as_deref(),
            Some("<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"main.js\"}}\n</tool_call>")
        );
        assert_eq!(
            format_tool_call_as_text("", "<tool_call name=\"list_files\" />").unwrap().as_deref(),
            Some("<tool_call name=\"list_files\" />")
        );

        // 4. Raw DSML remains rejected even without a tool catalog.
        assert!(format_tool_call_as_text("", &format!("{{\"name\":\"x\",\"args\":{{\"a\":\"{DSML}\"}}}}")).is_err());
        assert!(format_tool_call_as_text("", r#"{"name":"x","args":{"a":"\uFF5CDSML\uFF5C"}}"#).is_err());
    }

    #[test]
    fn post_thinking_guards_block_empty_stop_and_force_reply_opener_at_budget() {
        let tok = test_tokenizer();
        let ids = SpecialIds::from_tokenizer(&tok);
        let vocab_len = tok.vocab_size();

        // After `<|content_thinking|>...<|end_message|>`, translator has seen thinking with no
        // output yet, and `closed_thinking()` stays false so `Event::Thinking { done: true }`
        // stays deferred. `end_sampling` must be masked to prevent an empty reasoning-only reply.
        let mut tr = Inkling2Translator::with_tools(ids, false);
        let raw_thought = format!("{MSG_MODEL}{CONTENT_THINKING}Planning the edit.{END_MESSAGE}");
        for id in tok.encode(&raw_thought, false).unwrap() {
            let _ = tr.push(id, &tok);
        }
        assert!(tr.seen_thinking());
        assert!(!tr.closed_thinking());
        assert!(!tr.has_output());

        let mut logits = vec![1.0f32; vocab_len];
        mask_post_thinking_logits(&mut logits, ids, &tr, true, false);
        assert!(logits[ids.end_sampling.unwrap() as usize].is_infinite());
        assert!(logits[ids.end_message.unwrap() as usize].is_infinite());
        assert!(
            logits[ids.content_thinking.unwrap() as usize].is_infinite(),
            "reopening a second unbudgeted thinking block must be masked even after natural close"
        );

        // Once think_budget closes thinking, `enqueue_forced_reply_opener` opens
        // `<|message_model|><|content_text|>` on text protocol (and `<|message_model|>` when
        // native tools are declared), and `content_thinking` is also masked.
        let mut q_text = VecDeque::new();
        enqueue_forced_reply_opener(&mut q_text, ids, false);
        assert_eq!(q_text, VecDeque::from([ids.msg_model.unwrap(), ids.content_text.unwrap()]));
        let mut q_tools = VecDeque::new();
        enqueue_forced_reply_opener(&mut q_tools, ids, true);
        assert_eq!(q_tools, VecDeque::from([ids.msg_model.unwrap()]));

        // Feeding `<|message_model|><|content_text|>` closes thinking (releasing deferred
        // `Event::Thinking { done: true }`), while `end_message`, `end_sampling`, and `msg_model`
        // stay masked until at least one non-whitespace token is generated.
        for tok_id in q_text {
            let _ = tr.push(tok_id, &tok);
        }
        assert!(tr.closed_thinking());
        assert!(tr.in_text());
        assert!(!tr.has_output());
        let mut logits2 = vec![1.0f32; vocab_len];
        mask_post_thinking_logits(&mut logits2, ids, &tr, true, true);
        assert!(logits2[ids.end_sampling.unwrap() as usize].is_infinite());
        assert!(logits2[ids.end_message.unwrap() as usize].is_infinite());
        assert!(logits2[ids.msg_model.unwrap() as usize].is_infinite());
        assert!(logits2[ids.content_thinking.unwrap() as usize].is_infinite());

        // Once non-whitespace text is emitted, `end_message`, `end_sampling`, and `msg_model` are unmasked.
        for id in tok.encode("Done.", false).unwrap() {
            let _ = tr.push(id, &tok);
        }
        assert!(tr.has_output());
        let mut logits3 = vec![1.0f32; vocab_len];
        mask_post_thinking_logits(&mut logits3, ids, &tr, true, true);
        assert!(logits3[ids.end_sampling.unwrap() as usize].is_finite());
        assert!(logits3[ids.end_message.unwrap() as usize].is_finite());
        assert!(logits3[ids.msg_model.unwrap() as usize].is_finite());
        assert!(logits3[ids.content_thinking.unwrap() as usize].is_infinite());

        // If the model writes prose or `<tool_call>` directly after `<|message_model|>` without
        // emitting `<|content_text|>`, `Segment::Header` promotes it to `Segment::Text`.
        let mut tr_hdr = Inkling2Translator::with_tools(ids, false);
        let raw_hdr = format!("{MSG_MODEL}{CONTENT_THINKING}Plan.{END_MESSAGE}{MSG_MODEL}Hello world!{END_MESSAGE}{END_SAMPLING}");
        let mut out_hdr = String::new();
        for id in tok.encode(&raw_hdr, false).unwrap() {
            let (chunks, stop) = tr_hdr.push(id, &tok);
            for c in chunks {
                out_hdr.push_str(&c);
            }
            if stop {
                break;
            }
        }
        assert!(tr_hdr.has_output());
        assert_eq!(out_hdr, "<think>Plan.</think>Hello world!");
    }
}
