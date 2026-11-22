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
use crate::engines::glm::ThinkBudget;
use crate::engines::job::{penalize_reply, sample, Event, Finish, ImagePrep, Job};
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
    pub msg_model: Option<u32>,
    pub content_text: Option<u32>,
    pub content_thinking: Option<u32>,
    pub content_invoke_tool: Option<u32>,
    pub end_message: Option<u32>,
    pub end_sampling: Option<u32>,
}

impl SpecialIds {
    pub fn from_tokenizer(tok: &Tokenizer) -> Self {
        Self {
            msg_model: tok.token_id(MSG_MODEL),
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
/// into OAIY's `<think>...</think>` + DSML `<｜DSML｜ calls>` event stream.
pub struct Inkling2Translator {
    ids: SpecialIds,
    seg: Segment,
    header_buf: String,
    tool_buf: String,
    pending_calls: Vec<String>,
    closed_thinking: bool,
    pending_bytes: Vec<u32>,
}

impl Inkling2Translator {
    pub fn new(ids: SpecialIds) -> Self {
        Self {
            ids,
            seg: Segment::Idle,
            header_buf: String::new(),
            tool_buf: String::new(),
            pending_calls: Vec::new(),
            closed_thinking: false,
            pending_bytes: Vec::new(),
        }
    }

    pub fn in_thinking(&self) -> bool {
        self.seg == Segment::Thinking
    }

    pub fn in_text(&self) -> bool {
        self.seg == Segment::Text
    }

    /// Feed one generated token ID. Returns translated UTF-8 chunks to emit as `Event::Text`
    /// and whether generation should stop (`<|content_model_end_sampling|>`).
    pub fn push(&mut self, tok_id: u32, tok: &Tokenizer) -> (Vec<String>, bool) {
        let mut out = Vec::new();

        if Some(tok_id) == self.ids.end_sampling {
            self.flush_pending_utf8(tok, &mut out);
            self.flush_segment_close(&mut out);
            self.flush_tool_calls(&mut out);
            return (out, true);
        }

        if Some(tok_id) == self.ids.msg_model {
            self.flush_pending_utf8(tok, &mut out);
            self.flush_segment_close(&mut out);
            self.seg = Segment::Header;
            self.header_buf.clear();
            return (out, false);
        }

        if Some(tok_id) == self.ids.content_thinking {
            self.flush_pending_utf8(tok, &mut out);
            self.flush_tool_calls(&mut out);
            self.seg = Segment::Thinking;
            self.closed_thinking = false;
            out.push("<think>".into());
            return (out, false);
        }

        if Some(tok_id) == self.ids.content_text {
            self.flush_pending_utf8(tok, &mut out);
            self.flush_tool_calls(&mut out);
            if !self.closed_thinking {
                out.push("</think>".into());
                self.closed_thinking = true;
            }
            self.seg = Segment::Text;
            return (out, false);
        }

        if Some(tok_id) == self.ids.content_invoke_tool {
            self.flush_pending_utf8(tok, &mut out);
            if !self.closed_thinking {
                out.push("</think>".into());
                self.closed_thinking = true;
            }
            self.seg = Segment::ToolJson;
            self.tool_buf.clear();
            return (out, false);
        }

        if Some(tok_id) == self.ids.end_message {
            self.flush_pending_utf8(tok, &mut out);
            self.flush_segment_close(&mut out);
            self.seg = Segment::Idle;
            return (out, false);
        }

        // Normal token inside the active segment (buffer incomplete UTF-8 across multi-byte tokens).
        self.pending_bytes.push(tok_id);
        let decoded = tok.decode(&self.pending_bytes);
        if !decoded.contains('\u{FFFD}') {
            self.pending_bytes.clear();
            self.push_decoded(&decoded, &mut out);
        }
        (out, false)
    }

    /// Flush any remaining buffered text or tool calls at end of generation (e.g., `max_tokens`).
    pub fn finish(&mut self, tok: &Tokenizer) -> Vec<String> {
        let mut out = Vec::new();
        self.flush_pending_utf8(tok, &mut out);
        self.flush_segment_close(&mut out);
        self.flush_tool_calls(&mut out);
        out
    }

    fn flush_pending_utf8(&mut self, tok: &Tokenizer, out: &mut Vec<String>) {
        if self.pending_bytes.is_empty() {
            return;
        }
        let s = tok.decode(&self.pending_bytes).replace('\u{FFFD}', "");
        self.pending_bytes.clear();
        if !s.is_empty() {
            self.push_decoded(&s, out);
        }
    }

    fn push_decoded(&mut self, text: &str, out: &mut Vec<String>) {
        match self.seg {
            Segment::Header => self.header_buf.push_str(text),
            Segment::ToolJson => self.tool_buf.push_str(text),
            Segment::Thinking | Segment::Text => {
                if !text.is_empty() {
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
                    out.push(text.to_string());
                }
            }
        }
    }

    fn flush_segment_close(&mut self, out: &mut Vec<String>) {
        match self.seg {
            Segment::Thinking => {
                out.push("</think>".into());
                self.closed_thinking = true;
            }
            Segment::ToolJson => {
                if let Some(dsml_invoke) = format_tool_call_as_dsml(&self.header_buf, &self.tool_buf) {
                    self.pending_calls.push(dsml_invoke);
                }
                self.tool_buf.clear();
            }
            _ => {}
        }
    }

    fn flush_tool_calls(&mut self, out: &mut Vec<String>) {
        if self.pending_calls.is_empty() {
            return;
        }
        let body = self.pending_calls.join("\n");
        self.pending_calls.clear();
        out.push(format!("\n\n<{DSML} calls>\n{body}\n</{DSML} calls>"));
    }
}

/// Convert one Inkling-2 JSON tool invocation (`{"name":"fn","args":{...}}`) into a single
/// OAIY DSML `<｜DSML｜ invoke>` block.
fn format_tool_call_as_dsml(header_name: &str, json_str: &str) -> Option<String> {
    let trimmed = json_str.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parsed = Json::parse(trimmed.as_bytes()).ok()?;
    let name = parsed
        .get("name")
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let h = header_name.trim();
            (!h.is_empty()).then_some(h)
        })?;
    let args = parsed
        .get("args")
        .or_else(|| parsed.get("arguments"))
        .and_then(Json::as_object)
        .unwrap_or(&[]);

    let mut params = Vec::with_capacity(args.len());
    for (k, v) in args {
        let (is_str, rendered) = match v {
            Json::Str(s) => ("true", s.clone()),
            other => ("false", other.to_json()),
        };
        params.push(format!(
            "<{DSML} parameter name=\"{k}\" string=\"{is_str}\">{rendered}</{DSML} parameter>"
        ));
    }
    let inner = params.join("\n");
    Some(format!("<{DSML} invoke name=\"{name}\">\n{inner}\n</{DSML} invoke>"))
}

const SOFT_KEPT: usize = 4;
const PREFILL_CHUNK: usize = 64;

/// Dedicated worker engine serving an `Inkling2Model` (GGUF or Safetensors) with optional vision.
pub struct Inkling2Engine {
    model: Inkling2Model,
    vision: Option<Arc<Inkling2Vision>>,
    tok: Arc<Tokenizer>,
    specials: SpecialIds,
    max_seq: usize,
    covered: Vec<u32>,
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
            soft: VecDeque::new(),
            private_session: None,
            private_at: std::time::Instant::now(),
            log,
        }
    }

    fn reset(&mut self) {
        self.private_session = None;
        self.model.reset();
        self.covered.clear();
        self.soft.clear();
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

    fn encode_pictures(&mut self, job: &Job) -> Result<Vec<Arc<Vec<f32>>>> {
        let mut out = Vec::with_capacity(job.images.len());
        for (i, img) in job.images.iter().enumerate() {
            if !job.forget {
                if let Some((_, cached)) = self.soft.iter().find(|(h, _)| *h == img.hash) {
                    out.push(Arc::clone(cached));
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
            out.push(soft);
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

        let pictures = self.encode_pictures(job)?;
        let keyed: Vec<u32> = {
            let mut k = job.prompt.clone();
            for img in &job.images {
                for (off, id) in k[img.start..img.start + img.prep.n_tokens()]
                    .iter_mut()
                    .enumerate()
                {
                    *id = 0x8000_0000
                        | ((img.hash.rotate_left(17) ^ off as u64) as u32 & 0x7fff_ffff);
                }
            }
            k
        };

        let started = std::time::Instant::now();
        let common = self
            .covered
            .iter()
            .zip(&keyed)
            .take_while(|(a, b)| a == b)
            .count();
        let start = if common == self.covered.len() && common < keyed.len() {
            common
        } else {
            self.model.reset();
            self.covered.clear();
            0
        };
        let source = if start > 0 { "memory" } else { "none" };
        let _ = job.events.send(Event::CacheReuse {
            cached: start,
            source,
            common,
        });

        let total = keyed.len() - start;
        let _ = job.events.send(Event::Progress { done: 0, total });

        let mut pos = start;
        let mut logits = Vec::new();
        let d = self.model.cfg.hidden;
        while pos < keyed.len() {
            if job.cancel.load(Ordering::Relaxed) {
                let _ = job.events.send(Event::Done {
                    finish: Finish::Stop,
                    completion_tokens: 0,
                });
                return Ok(());
            }
            let take = PREFILL_CHUNK.min(keyed.len() - pos);
            let ids = &job.prompt[pos..pos + take];
            let soft: Vec<(usize, &[f32])> = job
                .images
                .iter()
                .zip(&pictures)
                .filter_map(|(img, rows)| {
                    let lo = img.start.max(pos);
                    let hi = (img.start + img.prep.n_tokens()).min(pos + take);
                    (lo < hi).then(|| (lo - pos, &rows[(lo - img.start) * d..(hi - img.start) * d]))
                })
                .collect();
            logits = self
                .model
                .forward_last(ids, &soft)
                .map_err(|e| oaiy_engine::Error::Arg(e.to_string()))?;
            self.covered.extend_from_slice(&keyed[pos..pos + take]);
            pos += take;
            let _ = job.events.send(Event::Progress {
                done: pos - start,
                total,
            });
        }

        let prefill_s = started.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });

        let mut rng = job.sampling.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut translator = Inkling2Translator::new(self.specials);
        let mut budget = ThinkBudget::new(false, self.specials.end_message, job.think_budget)
            .closing(&job.think_closing)
            .opens_with(self.specials.content_thinking);
        let mut reply_hist: VecDeque<u32> = VecDeque::new();
        let max_seq = self.model.cfg.context_length.min(self.max_seq);

        let mut n = 0usize;
        let decode_start = std::time::Instant::now();
        let finish = loop {
            if translator.in_text() && !reply_hist.is_empty() {
                penalize_reply(&mut logits, &reply_hist, &job.sampling);
            }
            let sampled = sample(&logits, &job.sampling, &mut rng);
            let was_thinking = budget.thinking;
            let (next, forced) = budget.pass(sampled);
            if forced && !budget.thinking && self.log {
                eprintln!(
                    "  reasoning ended at its budget ({} tokens)",
                    job.think_budget.unwrap_or(0)
                );
            }
            if budget.reports(was_thinking) {
                let _ = job.events.send(Event::Thinking {
                    used: budget.used,
                    budget: budget.budget,
                    done: !budget.thinking,
                });
            }

            let (chunks, stop) = translator.push(next, &self.tok);
            let mut disconnected = false;
            for chunk in chunks {
                if job.events.send(Event::Text(chunk)).is_err() {
                    disconnected = true;
                    break;
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

            logits = self
                .model
                .forward_last(&[next], &[])
                .map_err(|e| oaiy_engine::Error::Arg(e.to_string()))?;
            self.covered.push(next);
            pos += 1;
        };

        for chunk in translator.finish(&self.tok) {
            let _ = job.events.send(Event::Text(chunk));
        }

        let decode_s = decode_start.elapsed().as_secs_f64();
        if self.log {
            eprintln!(
                "  {} prompt tokens ({start} reused) in {prefill_s:.1}s; {n} generated in {decode_s:.1}s ({:.1} tok/s)",
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
    }
}
