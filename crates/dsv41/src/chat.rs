//! The DeepSeek-V4.1 chat format: OpenAI-style messages to prompt text, and
//! completions back to assistant messages.
//!
//! This ports DeepSeek's reference `encoding/encoding.py` (MIT License; see
//! NOTICE) rule for rule, templates included (the golden tests compare
//! against it), with its V4.1 specifics:
//!
//! - roles `system`, `user`, `assistant`, `tool` (folded into the next user
//!   turn as `<tool_result>` blocks, in the order of the calls) and
//!   `latest_reminder`; a system message after the first is a
//!   mid-conversation `<｜System｜>` turn;
//! - `chat` mode closes the thinking block at once (`<｜Assistant｜></think>`);
//!   `thinking` mode opens it, prefixes the conversation with a numeric
//!   reasoning effort (1–100), and drops earlier turns' reasoning unless the
//!   conversation has tools;
//! - tool calls are DSML blocks (`<｜DSML｜ calls>` … `</｜DSML｜ calls>`),
//!   with string parameters written raw and everything else as JSON;
//! - images are `<｜deepseek_image｜>` placeholders, returned in prompt
//!   order for the vision tower.
//!
//! JSON embedded in the prompt (tool schemas, response formats, non-string
//! arguments) is written as Python's `json.dumps` would, since the model
//! was trained on that.

use nrob::json::{write_str, Json};
use nrob::{Error, Result};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
pub const USER: &str = "<｜User｜>";
pub const ASSISTANT: &str = "<｜Assistant｜>";
pub const SYSTEM: &str = "<｜System｜>";
pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub const IMAGE: &str = "<｜deepseek_image｜>";

/// Where a completion's tool calls begin.
pub const TOOL_CALLS_START: &str = "\n\n<｜DSML｜ calls";

const TASKS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

const EFFORT_TEMPLATE: (&str, &str) =
    ("Reasoning Effort: ", " (range 1-100, the higher the value, the more thorough the reasoning)\n\n");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Answer directly: the thinking block is closed before the reply.
    Chat,
    /// Reason inside `<think>…</think>` first.
    Thinking,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub mode: Mode,
    /// Reasoning budget 1–100 (thinking mode only).
    pub effort: u32,
    /// Drop reasoning from turns before the last user turn (ignored, as
    /// if false, when the conversation has tools).
    pub drop_thinking: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { mode: Mode::Chat, effort: 75, drop_thinking: true }
    }
}

/// `"low"`, `"high"`, `"max"` or 1–100, as the reference accepts; OpenAI's
/// `"medium"` maps to the default. `None` for anything else.
pub fn parse_effort(v: &Json) -> Option<u32> {
    match v {
        Json::Str(s) => match s.as_str() {
            "low" => Some(50),
            "medium" | "high" => Some(75),
            "max" => Some(100),
            _ => None,
        },
        Json::Int(i) if (1..=100).contains(i) => Some(*i as u32),
        _ => None,
    }
}

/// A prompt and the images it refers to, in placeholder order. Each image
/// record is `{"type": "image", "url"|"source"|"data": …}`.
#[derive(Debug)]
pub struct Encoded {
    pub prompt: String,
    pub images: Vec<Json>,
}

#[derive(Clone, Debug)]
enum Block {
    Text(String),
    ToolResult { id: String, content: ToolContent },
    /// A block type the format has no rendering for.
    Other(String),
}

#[derive(Clone, Debug)]
enum ToolContent {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Clone, Debug, Default)]
struct Msg {
    role: String,
    content: Option<String>,
    blocks: Option<Vec<Block>>,
    tools: Vec<Json>,
    response_format: Option<Json>,
    tool_calls: Vec<Json>,
    reasoning: Option<String>,
    wo_eos: bool,
    task: Option<String>,
    tool_call_id: String,
}

fn fail(m: impl Into<String>) -> Error {
    Error::Arg(m.into())
}

fn text_of(v: Option<&Json>) -> Option<String> {
    v.and_then(Json::as_str).map(str::to_string)
}

/// Content blocks from a list, images replaced by placeholder text and
/// collected in order.
fn blocks_from(list: &[Json], images: &mut Vec<Json>) -> Result<Vec<Block>> {
    let mut out = Vec::with_capacity(list.len());
    for b in list {
        // Python renders a missing type as "None"
        let ty = b.get("type").and_then(Json::as_str).unwrap_or("None");
        match ty {
            "image" | "image_url" => {
                out.push(Block::Text(IMAGE.into()));
                images.push(image_record(b)?);
            }
            "text" => {
                let text = b.get("text").and_then(Json::as_str).unwrap_or("");
                if text.contains(IMAGE) {
                    return Err(fail(format!("text block contains the image placeholder {IMAGE}; send images as image blocks")));
                }
                out.push(Block::Text(text.into()));
            }
            "tool_result" => {
                let content = match b.get("content") {
                    Some(Json::Arr(list)) => ToolContent::Blocks(blocks_from(list, images)?),
                    Some(Json::Str(s)) => ToolContent::Text(s.clone()),
                    _ => ToolContent::Text(String::new()),
                };
                let id = b.get("tool_use_id").and_then(Json::as_str).unwrap_or("").to_string();
                out.push(Block::ToolResult { id, content });
            }
            other => out.push(Block::Other(other.to_string())),
        }
    }
    Ok(out)
}

fn image_record(b: &Json) -> Result<Json> {
    let mut rec = vec![("type".to_string(), Json::str("image"))];
    if b.get("type").and_then(Json::as_str) == Some("image_url") {
        let url = match b.get("image_url") {
            Some(Json::Str(s)) => s.clone(),
            Some(o) => o.get("url").and_then(Json::as_str).unwrap_or("").to_string(),
            None => String::new(),
        };
        rec.push(("url".into(), Json::Str(url)));
    } else {
        for key in ["source", "url", "data"] {
            if let Some(v) = b.get(key) {
                rec.push((key.into(), v.clone()));
            }
        }
    }
    let valid = rec[1..].iter().any(|(_, v)| truthy(v));
    if !valid {
        return Err(fail("image block has no source"));
    }
    Ok(Json::Obj(rec))
}

/// Python truthiness of a JSON value.
fn truthy(v: &Json) -> bool {
    match v {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Int(i) => *i != 0,
        Json::Num(f) => *f != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Arr(a) => !a.is_empty(),
        Json::Obj(o) => !o.is_empty(),
    }
}

/// OpenAI-style message JSON to the internal form, images extracted.
fn read_message(m: &Json, images: &mut Vec<Json>) -> Result<Msg> {
    let role = m.get("role").and_then(Json::as_str).ok_or_else(|| fail("message without a role"))?.to_string();
    let mut msg = Msg { role, ..Msg::default() };
    let content = m.get("content");
    if let Some(Json::Str(s)) = content {
        if s.contains(IMAGE) {
            return Err(fail(format!("message content contains the image placeholder {IMAGE}; send images as image blocks")));
        }
    }
    msg.reasoning = text_of(m.get("reasoning_content"));
    if msg.reasoning.as_deref().is_some_and(|r| r.contains(IMAGE)) {
        return Err(fail(format!("reasoning_content contains the image placeholder {IMAGE}")));
    }
    let list = match (m.get("content_blocks"), content) {
        (Some(Json::Arr(b)), _) => Some(b.as_slice()),
        (None, Some(Json::Arr(b))) => Some(b.as_slice()),
        _ => None,
    };
    match list {
        Some(list) if !list.is_empty() => {
            let blocks = blocks_from(list, images)?;
            msg.content = match content {
                Some(Json::Str(s)) => Some(s.clone()),
                _ => Some(
                    blocks
                        .iter()
                        .filter_map(|b| if let Block::Text(t) = b { Some(t.as_str()) } else { None })
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                ),
            };
            msg.blocks = Some(blocks);
        }
        Some(_) => msg.blocks = Some(Vec::new()),
        None => msg.content = text_of(content),
    }
    if let Some(Json::Arr(t)) = m.get("tools") {
        msg.tools = t.clone();
    }
    msg.response_format = m.get("response_format").filter(|v| truthy(v)).cloned();
    if let Some(Json::Arr(c)) = m.get("tool_calls") {
        msg.tool_calls = c.clone();
    }
    msg.wo_eos = m.get("wo_eos").and_then(Json::as_bool).unwrap_or(false);
    msg.task = text_of(m.get("task"));
    msg.tool_call_id = m.get("tool_call_id").and_then(Json::as_str).unwrap_or("").to_string();
    Ok(msg)
}

/// Tool messages become `tool_result` blocks in a user turn; consecutive
/// user turns merge (unless the earlier one carries a task).
fn merge_tool_messages(messages: Vec<Msg>) -> Vec<Msg> {
    let mut merged: Vec<Msg> = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg.role.as_str() {
            "tool" => {
                let block = Block::ToolResult { id: msg.tool_call_id.clone(), content: ToolContent::Text(msg.content.clone().unwrap_or_default()) };
                match merged.last_mut() {
                    Some(last) if last.role == "user" && last.blocks.is_some() => last.blocks.as_mut().expect("checked").push(block),
                    _ => merged.push(Msg { role: "user".into(), blocks: Some(vec![block]), ..Msg::default() }),
                }
            }
            "user" => {
                let blocks = msg.blocks.clone().unwrap_or_else(|| vec![Block::Text(msg.content.clone().unwrap_or_default())]);
                match merged.last_mut() {
                    Some(last) if last.role == "user" && last.blocks.is_some() && last.task.is_none() => {
                        last.blocks.as_mut().expect("checked").extend(blocks)
                    }
                    _ => merged.push(Msg { blocks: Some(blocks), ..msg }),
                }
            }
            _ => merged.push(msg),
        }
    }
    merged
}

fn call_id(tc: &Json) -> String {
    tc.get("id")
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| tc.get("function").and_then(|f| f.get("id")).and_then(Json::as_str))
        .unwrap_or("")
        .to_string()
}

/// Order a user turn's tool results as the preceding assistant called them.
fn sort_tool_results(messages: &mut [Msg]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        if msg.role == "assistant" && !msg.tool_calls.is_empty() {
            order = msg.tool_calls.iter().enumerate().map(|(i, tc)| (call_id(tc), i)).filter(|(id, _)| !id.is_empty()).collect();
        } else if msg.role == "user" {
            let Some(blocks) = msg.blocks.as_mut() else { continue };
            let n_results = blocks.iter().filter(|b| matches!(b, Block::ToolResult { .. })).count();
            if n_results < 2 || order.is_empty() {
                continue;
            }
            let rank = |b: &Block| match b {
                Block::ToolResult { id, .. } => order.iter().find(|(o, _)| o == id).map_or(0, |&(_, i)| i),
                _ => 0,
            };
            let mut results: Vec<Block> = blocks.iter().filter(|b| matches!(b, Block::ToolResult { .. })).cloned().collect();
            results.sort_by_key(rank); // stable, like Python's sorted
            let mut it = results.into_iter();
            for b in blocks.iter_mut() {
                if matches!(b, Block::ToolResult { .. }) {
                    *b = it.next().expect("same count");
                }
            }
        }
    }
}

/// The last user turn (a system turn after the first counts as one).
fn last_user_index(messages: &[Msg]) -> Option<usize> {
    (0..messages.len()).rev().find(|&i| messages[i].role == "user" || (messages[i].role == "system" && i > 0))
}

fn drop_thinking_messages(messages: Vec<Msg>) -> Vec<Msg> {
    let last = last_user_index(&messages).map_or(-1, |i| i as i64);
    let keep = ["user", "system", "tool", "latest_reminder", "direct_search_results"];
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(i, mut m)| {
            if keep.contains(&m.role.as_str()) || i as i64 >= last {
                Some(m)
            } else if m.role == "assistant" {
                m.reasoning = None;
                Some(m)
            } else {
                None
            }
        })
        .collect()
}

/// Split `name` at `::` into (namespace, name), checking it against an
/// explicit namespace.
fn split_tool_name(name: &str, namespace: Option<&str>) -> Result<(Option<String>, String)> {
    let (ns, bare) = match name.split_once("::") {
        Some((prefix, bare)) => {
            if namespace.is_some_and(|n| n != prefix) {
                return Err(fail(format!("conflicting tool namespaces: {} != {prefix}", namespace.unwrap_or(""))));
            }
            (Some(prefix.to_string()), bare.to_string())
        }
        None => (namespace.map(str::to_string), name.to_string()),
    };
    if bare.contains("::") {
        return Err(fail(format!("tool name must not contain '::': {bare}")));
    }
    if ns.as_deref().is_some_and(|n| n.contains("::")) {
        return Err(fail("tool namespace must not contain '::'"));
    }
    Ok((ns, bare))
}

fn qualified(ns: Option<&str>, name: &str) -> String {
    match ns {
        Some(ns) => format!("{ns}::{name}"),
        None => name.to_string(),
    }
}

fn namespace_name(v: Option<&Json>) -> Option<&str> {
    match v? {
        Json::Str(s) => Some(s.as_str()),
        o @ Json::Obj(_) => o.get("name").and_then(Json::as_str),
        _ => None,
    }
}

/// OpenAI tool definitions to the function schemas the prompt lists.
fn tool_schemas(tools: &[Json]) -> Result<Vec<Json>> {
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let Some(Json::Obj(function)) = tool.get("function") else {
            return Err(fail("tool without a function"));
        };
        let mut function = function.clone();
        // the tool-level namespace goes on the function (last, if new)
        if let Some(ns) = tool.get("namespace").filter(|v| !matches!(v, Json::Null)) {
            match function.iter_mut().find(|(k, _)| k == "namespace") {
                Some((_, v)) => *v = ns.clone(),
                None => function.push(("namespace".into(), ns.clone())),
            }
        }
        let f = Json::Obj(function.clone());
        let ns_value = f.get("namespace").cloned();
        let name = f.get("name").and_then(Json::as_str).ok_or_else(|| fail("tool function without a name"))?;
        let (ns, bare) = split_tool_name(name, namespace_name(ns_value.as_ref()))?;
        let full = qualified(ns.as_deref(), &bare);
        for (k, v) in function.iter_mut() {
            if k == "name" {
                *v = Json::Str(full.clone());
            }
        }
        function.retain(|(k, _)| k != "namespace");
        if let Some(desc) = ns_value.as_ref().and_then(|n| n.get("description")).and_then(Json::as_str).filter(|d| !d.is_empty()) {
            let old = function.iter().find(|(k, _)| k == "description").and_then(|(_, v)| v.as_str()).unwrap_or("").to_string();
            let new = Json::Str(format!("{desc}\n{old}"));
            match function.iter_mut().find(|(k, _)| k == "description") {
                Some((_, v)) => *v = new,
                None => function.push(("description".into(), new)),
            }
        }
        out.push(Json::Obj(function));
    }
    Ok(out)
}

struct Call {
    name: String,
    namespace: Option<String>,
    arguments: Json,
}

fn tool_calls_from_openai(calls: &[Json]) -> Result<Vec<Call>> {
    calls
        .iter()
        .map(|tc| {
            let f = tc.get("function").ok_or_else(|| fail("tool call without a function"))?;
            let name = f.get("name").and_then(Json::as_str).ok_or_else(|| fail("tool call without a name"))?;
            let ns = tc.get("namespace").filter(|v| truthy(v)).or_else(|| f.get("namespace"));
            let (namespace, name) = split_tool_name(name, ns.and_then(Json::as_str))?;
            let arguments = f.get("arguments").cloned().unwrap_or(Json::Null);
            Ok(Call { name, namespace, arguments })
        })
        .collect()
}

fn tools_prompt(tools: &[Json]) -> String {
    let d = DSML;
    let schemas: Vec<String> = tools.iter().map(Json::to_python_json).collect();
    format!(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{d} calls>\" block like the following:\n\n\
<{d} calls>\n<{d} invoke name=\"$TOOL_NAME\">\n<{d} parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{d} parameter>\n...\n</{d} invoke>\n\
<{d} invoke name=\"$TOOL_NAME2\">\n...\n</{d} invoke>\n</{d} calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by {THINK_START}), you MUST output your complete reasoning inside {THINK_START}...{THINK_END} BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after {THINK_END} with tool calls or final response.\n\n### Available Tool Schemas\n\n{}\n\n\
You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
        schemas.join("\n")
    )
}

/// A call's arguments as DSML parameters. Arguments may be an object, a JSON
/// string of one (even encoded twice), or anything else (then passed as a
/// single `arguments` parameter).
fn dsml_arguments(call: &Call) -> String {
    let mut args = call.arguments.clone();
    for _ in 0..2 {
        match &args {
            Json::Str(s) => match Json::parse(s.as_bytes()) {
                Ok(v) => args = v,
                Err(_) => break,
            },
            _ => break,
        }
    }
    let members = match args {
        Json::Obj(m) => m,
        _ => vec![("arguments".to_string(), call.arguments.clone())],
    };
    members
        .iter()
        .map(|(k, v)| {
            let (is_str, value) = match v {
                Json::Str(s) => ("true", s.clone()),
                other => ("false", other.to_python_json()),
            };
            format!("<{DSML} parameter name=\"{k}\" string=\"{is_str}\">{value}</{DSML} parameter>")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_message(index: usize, messages: &[Msg], opt: &Options, drop_thinking: bool) -> Result<String> {
    let msg = &messages[index];
    let last_user = last_user_index(messages).map_or(-1, |i| i as i64);
    let thinking = opt.mode == Mode::Thinking;

    let effort = if index == 0 && thinking { format!("{}{}{}", EFFORT_TEMPLATE.0, opt.effort, EFFORT_TEMPLATE.1) } else { String::new() };
    let mut p = String::new();
    if index == 0 && (!effort.is_empty() || msg.role == "system") {
        p.push_str(SYSTEM);
    }
    p.push_str(&effort);

    match msg.role.as_str() {
        "system" => {
            if index > 0 {
                p.push_str(SYSTEM);
            }
            p.push_str(msg.content.as_deref().unwrap_or(""));
            if !msg.tools.is_empty() {
                p.push_str("\n\n");
                p.push_str(&tools_prompt(&tool_schemas(&msg.tools)?));
            }
            if let Some(rf) = &msg.response_format {
                p.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                p.push_str(&rf.to_python_json());
            }
        }
        "user" => {
            p.push_str(USER);
            match msg.blocks.as_deref() {
                Some(blocks) if !blocks.is_empty() => {
                    let parts: Vec<String> = blocks
                        .iter()
                        .map(|b| match b {
                            Block::Text(t) => t.clone(),
                            Block::ToolResult { content, .. } => {
                                let text = match content {
                                    ToolContent::Text(t) => t.clone(),
                                    ToolContent::Blocks(bs) => bs
                                        .iter()
                                        .map(|b| match b {
                                            Block::Text(t) => t.clone(),
                                            Block::ToolResult { .. } => "[Unsupported tool_result]".into(),
                                            Block::Other(ty) => format!("[Unsupported {ty}]"),
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n\n"),
                                };
                                format!("<tool_result>{text}</tool_result>")
                            }
                            Block::Other(ty) => format!("[Unsupported {ty}]"),
                        })
                        .collect();
                    p.push_str(&parts.join("\n\n"));
                }
                _ => p.push_str(msg.content.as_deref().unwrap_or("")),
            }
        }
        "latest_reminder" => {
            p.push_str(LATEST_REMINDER);
            p.push_str(msg.content.as_deref().unwrap_or(""));
        }
        "assistant" => {
            let mut calls = String::new();
            if !msg.tool_calls.is_empty() {
                let list: Vec<String> = tool_calls_from_openai(&msg.tool_calls)?
                    .iter()
                    .map(|c| {
                        format!(
                            "<{DSML} invoke name=\"{}\">\n{}\n</{DSML} invoke>",
                            qualified(c.namespace.as_deref(), &c.name),
                            dsml_arguments(c)
                        )
                    })
                    .collect();
                calls = format!("\n\n<{DSML} calls>\n{}\n</{DSML} calls>", list.join("\n"));
            }
            let prev_has_task = index > 0 && messages[index - 1].task.is_some();
            let mut reasoning = String::new();
            if thinking && !prev_has_task && (!drop_thinking || index as i64 > last_user) {
                reasoning = format!("{}{THINK_END}", msg.reasoning.as_deref().unwrap_or(""));
            }
            p.push_str(&reasoning);
            p.push_str(msg.content.as_deref().unwrap_or(""));
            p.push_str(&calls);
            if !msg.wo_eos {
                p.push_str(EOS);
            }
        }
        "tool" => return Err(fail("tool messages are folded into user turns before rendering")),
        other => return Err(fail(format!("unknown role {other:?}"))),
    }

    // what follows the message
    if index + 1 < messages.len() && !matches!(messages[index + 1].role.as_str(), "assistant" | "latest_reminder") {
        return Ok(p);
    }
    if let Some(task) = &msg.task {
        let token = TASKS.iter().find(|(t, _)| t == task).map(|(_, tok)| *tok).ok_or_else(|| fail(format!("invalid task {task:?}")))?;
        if task != "action" {
            p.push_str(token);
        } else {
            p.push_str(ASSISTANT);
            p.push_str(if thinking { THINK_START } else { THINK_END });
            p.push_str(token);
        }
    } else if msg.role == "user" || (msg.role == "system" && index > 0) {
        p.push_str(ASSISTANT);
        let open = thinking && (!drop_thinking || index as i64 >= last_user);
        p.push_str(if open { THINK_START } else { THINK_END });
    }
    Ok(p)
}

/// Render `messages` (OpenAI-style JSON objects) as a prompt that ends
/// where the assistant's reply begins.
pub fn encode(messages: &[Json], opt: &Options) -> Result<Encoded> {
    let mut images = Vec::new();
    let msgs: Vec<Msg> = messages.iter().map(|m| read_message(m, &mut images)).collect::<Result<_>>()?;
    let mut msgs = merge_tool_messages(msgs);
    sort_tool_results(&mut msgs);
    let drop_thinking = opt.drop_thinking && !msgs.iter().any(|m| !m.tools.is_empty());
    if opt.mode == Mode::Thinking && drop_thinking {
        msgs = drop_thinking_messages(msgs);
    }
    let mut prompt = String::from(BOS);
    for i in 0..msgs.len() {
        prompt.push_str(&render_message(i, &msgs, opt, drop_thinking)?);
    }
    Ok(Encoded { prompt, images })
}

// ---- parsing completions ------------------------------------------------------

/// A tool call the model made.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub namespace: Option<String>,
    /// JSON object text, as the reference writes it.
    pub arguments: String,
}

/// An assistant reply, parsed.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Reply {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Text from `i` up to the first of `stops`: (end of the stop, the text,
/// which stop), or (the end, the rest, None).
fn read_until<'a>(text: &'a str, i: usize, stops: &[&str]) -> (usize, &'a str, Option<usize>) {
    let mut best: Option<(usize, usize)> = None;
    for (k, s) in stops.iter().enumerate() {
        if let Some(p) = text[i..].find(s) {
            if best.is_none_or(|(bp, _)| i + p < bp) {
                best = Some((i + p, k));
            }
        }
    }
    match best {
        Some((p, k)) => (p + stops[k].len(), &text[i..p], Some(k)),
        None => (text.len(), &text[i..], None),
    }
}

fn format_err(m: impl Into<String>) -> Error {
    Error::Format(m.into())
}

/// Parse a DSML calls block from `i` (just after `<｜DSML｜ calls`):
/// (position after `</｜DSML｜ calls>`, calls).
fn parse_tool_calls(text: &str, mut i: usize) -> Result<(usize, Vec<ToolCall>)> {
    let calls_end = format!("</{DSML} calls>");
    let invoke_start = format!("<{DSML} invoke");
    let invoke_end = format!("</{DSML} invoke");
    let param_start = format!("<{DSML} parameter");
    let param_end = format!("/{DSML} parameter");
    let mut calls = Vec::new();
    while i < text.len() {
        let (next, between, stop) = read_until(text, i, &[&invoke_start, &calls_end]);
        i = next;
        if between != ">\n" {
            return Err(format_err(format!("tool call format: expected '>\\n', got {between:?}")));
        }
        match stop {
            Some(1) => return Ok((i, calls)),
            None => return Err(format_err("tool calls block is not closed")),
            _ => {}
        }
        let (next, head, mut stop) = read_until(text, i, &[&param_start, &invoke_end]);
        i = next;
        let name = head
            .trim_start()
            .strip_prefix("name=\"")
            .and_then(|r| r.strip_suffix("\">\n"))
            .ok_or_else(|| format_err(format!("tool name format: {head:?}")))?;
        let mut params: Vec<(String, String, bool)> = Vec::new();
        while stop == Some(0) {
            let (next, body, _) = read_until(text, i, &[&param_end]);
            i = next;
            let rest = body.strip_prefix(" name=\"").ok_or_else(|| format_err(format!("parameter format: {body:?}")))?;
            let t = rest.find("\" string=\"true\">");
            let f = rest.find("\" string=\"false\">");
            let (at, is_str, marker) = match (t, f) {
                (Some(t), Some(f)) if f < t => (f, false, "\" string=\"false\">".len()),
                (Some(t), _) => (t, true, "\" string=\"true\">".len()),
                (None, Some(f)) => (f, false, "\" string=\"false\">".len()),
                (None, None) => return Err(format_err(format!("parameter format: {body:?}"))),
            };
            let pname = &rest[..at];
            let value = rest[at + marker..].strip_suffix('<').ok_or_else(|| format_err(format!("parameter format: {body:?}")))?;
            if params.iter().any(|(n, _, _)| n == pname) {
                return Err(format_err(format!("duplicate parameter {pname:?}")));
            }
            params.push((pname.to_string(), value.to_string(), is_str));
            let (next, between, s) = read_until(text, i, &[&param_start, &invoke_end]);
            i = next;
            stop = s;
            if between != ">\n" {
                return Err(format_err(format!("parameter format: expected '>\\n', got {between:?}")));
            }
        }
        let mut args = String::from("{");
        for (k, (pname, value, is_str)) in params.iter().enumerate() {
            if k > 0 {
                args.push_str(", ");
            }
            write_str(pname, &mut args);
            args.push_str(": ");
            if *is_str {
                write_str(value, &mut args);
            } else {
                args.push_str(value);
            }
        }
        args.push('}');
        let (namespace, name) = split_tool_name(name, None).map_err(|e| format_err(e.to_string()))?;
        calls.push(ToolCall { name, namespace, arguments: args });
    }
    Ok((i, calls))
}

/// Parse a whole completion (which ends with EOS, or where generation
/// stopped at EOS), as the reference does, rejecting malformed output.
pub fn parse_reply(text: &str, mode: Mode) -> Result<Reply> {
    let mut reply = Reply::default();
    let mut i = 0;
    if mode == Mode::Thinking {
        let (next, r, stop) = read_until(text, i, &[THINK_END, TOOL_CALLS_START]);
        if stop != Some(0) {
            return Err(format_err("missing </think>"));
        }
        reply.reasoning = r.to_string();
        i = next;
    }
    let (next, content, stop) = read_until(text, i, &[EOS, TOOL_CALLS_START]);
    reply.content = content.to_string();
    i = next;
    match stop {
        Some(1) => {
            let (next, calls) = parse_tool_calls(text, i)?;
            reply.tool_calls = calls;
            let (next, after, _) = read_until(text, next, &[EOS]);
            if !after.is_empty() {
                return Err(format_err("text after the tool calls"));
            }
            i = next;
        }
        Some(_) => {}
        None => return Err(format_err("missing EOS")),
    }
    if i != text.len() {
        return Err(format_err("text after the end of the reply"));
    }
    for sp in [BOS, EOS, THINK_START, THINK_END, DSML] {
        if reply.content.contains(sp) || reply.reasoning.contains(sp) {
            return Err(format_err(format!("special token {sp:?} inside the reply")));
        }
    }
    Ok(reply)
}

// ---- streaming ------------------------------------------------------------------

/// A piece of a reply as it streams.
#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Reasoning,
    Content,
    ToolCalls,
}

/// Splits streamed completion text into reasoning and content, holding back
/// any tail that might be the start of `</think>` or a tool-calls block,
/// and collects the tool-calls block for [`finish`](Self::finish).
pub struct StreamParser {
    phase: Phase,
    pending: String,
    calls: String,
}

impl StreamParser {
    /// `mode` is the prompt's: thinking mode starts inside the reasoning.
    pub fn new(mode: Mode) -> StreamParser {
        let phase = if mode == Mode::Thinking { Phase::Reasoning } else { Phase::Content };
        StreamParser { phase, pending: String::new(), calls: String::new() }
    }

    /// Feed more text; returns what can be shown now.
    pub fn push(&mut self, text: &str) -> Vec<Delta> {
        let mut out = Vec::new();
        match self.phase {
            Phase::ToolCalls => self.calls.push_str(text),
            _ => self.pending.push_str(text),
        }
        loop {
            match self.phase {
                Phase::Reasoning => match self.pending.find(THINK_END) {
                    Some(p) => {
                        emit(&mut out, Delta::Reasoning(self.pending[..p].to_string()));
                        self.pending.drain(..p + THINK_END.len());
                        self.phase = Phase::Content;
                    }
                    None => {
                        let keep = held(&self.pending, THINK_END);
                        let shown: String = self.pending.drain(..self.pending.len() - keep).collect();
                        emit(&mut out, Delta::Reasoning(shown));
                        return out;
                    }
                },
                Phase::Content => match self.pending.find(TOOL_CALLS_START) {
                    Some(p) => {
                        emit(&mut out, Delta::Content(self.pending[..p].to_string()));
                        self.calls = self.pending[p + 2..].to_string();
                        self.pending.clear();
                        self.phase = Phase::ToolCalls;
                        return out;
                    }
                    None => {
                        let keep = held(&self.pending, TOOL_CALLS_START);
                        let shown: String = self.pending.drain(..self.pending.len() - keep).collect();
                        emit(&mut out, Delta::Content(shown));
                        return out;
                    }
                },
                Phase::ToolCalls => return out,
            }
        }
    }

    /// End of generation: the held-back text, and the tool calls if the
    /// reply made any. A tool-calls block that does not parse is returned
    /// as content, so nothing the model wrote is lost.
    pub fn finish(mut self) -> (Vec<Delta>, Vec<ToolCall>) {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        match self.phase {
            Phase::Reasoning => emit(&mut out, Delta::Reasoning(rest)),
            Phase::Content => emit(&mut out, Delta::Content(rest)),
            Phase::ToolCalls => {
                let block = self.calls.trim_end_matches(EOS);
                let opening = format!("<{DSML} calls");
                let parsed = block.strip_prefix(&opening).map(|_| parse_tool_calls(block, opening.len()));
                match parsed {
                    Some(Ok((end, calls))) if block[end..].trim().is_empty() => return (out, calls),
                    _ => emit(&mut out, Delta::Content(format!("\n\n{block}"))),
                }
            }
        }
        (out, Vec::new())
    }
}

fn emit(out: &mut Vec<Delta>, d: Delta) {
    let empty = match &d {
        Delta::Reasoning(s) | Delta::Content(s) => s.is_empty(),
    };
    if !empty {
        out.push(d);
    }
}

/// Bytes at the end of `s` that could begin `marker` (kept back until the
/// next text decides).
fn held(s: &str, marker: &str) -> usize {
    (1..marker.len().min(s.len() + 1))
        .rev()
        .find(|&k| s.is_char_boundary(s.len() - k) && marker.starts_with(&s[s.len() - k..]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_parser_splits_reasoning_content_and_calls() {
        let full = format!(
            "let me think</think>Sure.\n\n<{DSML} calls>\n<{DSML} invoke name=\"read_file\">\n<{DSML} parameter name=\"path\" string=\"true\">/a/b</{DSML} parameter>\n</{DSML} invoke>\n</{DSML} calls>"
        );
        // any split into chunks gives the same result
        for step in [1, 2, 3, 7, 1000] {
            let mut p = StreamParser::new(Mode::Thinking);
            let (mut reasoning, mut content) = (String::new(), String::new());
            let chars: Vec<char> = full.chars().collect();
            for chunk in chars.chunks(step) {
                for d in p.push(&chunk.iter().collect::<String>()) {
                    match d {
                        Delta::Reasoning(s) => reasoning.push_str(&s),
                        Delta::Content(s) => content.push_str(&s),
                    }
                }
            }
            let (rest, calls) = p.finish();
            assert!(rest.is_empty(), "{rest:?}");
            assert_eq!(reasoning, "let me think");
            assert_eq!(content, "Sure.");
            assert_eq!(calls, vec![ToolCall { name: "read_file".into(), namespace: None, arguments: r#"{"path": "/a/b"}"#.into() }]);
        }
        // a partial marker at the end is released as text
        let mut p = StreamParser::new(Mode::Chat);
        assert!(p.push("a\n\n<").iter().all(|d| d == &Delta::Content("a".into())));
        assert_eq!(p.finish().0, vec![Delta::Content("\n\n<".into())]);
    }

    fn golden() -> Option<Json> {
        let path = std::env::var("DSV41_CHAT_GOLDEN").unwrap_or_else(|_| r"E:\deepseek\golden\chat_cases.json".into());
        Json::parse(&std::fs::read(path).ok()?).ok()
    }

    /// Every case from tools/dsv41/chat_golden.py (the reference encoder).
    #[test]
    fn matches_reference_encoding() {
        let Some(cases) = golden() else { return };
        let mut failures = Vec::new();
        let cases = cases.as_array().unwrap();
        for (n, case) in cases.iter().enumerate() {
            let mode = if case.get("mode").and_then(Json::as_str) == Some("thinking") { Mode::Thinking } else { Mode::Chat };
            let want_err = case.get("error").is_some();
            match case.get("kind").and_then(Json::as_str) {
                Some("encode") => {
                    let effort = match case.get("effort") {
                        None | Some(Json::Null) => 75,
                        Some(v) => parse_effort(v).unwrap(),
                    };
                    let drop_thinking = case.get("drop_thinking").and_then(Json::as_bool).unwrap_or(true);
                    let opt = Options { mode, effort, drop_thinking };
                    let got = encode(case.get("messages").and_then(Json::as_array).unwrap(), &opt);
                    match (got, want_err) {
                        (Ok(e), false) => {
                            let want = case.get("prompt").and_then(Json::as_str).unwrap();
                            if e.prompt != want {
                                failures.push(format!("case {n} prompt\n want {want:?}\n got  {:?}", e.prompt));
                            } else if Json::Arr(e.images.clone()) != *case.get("images").unwrap() {
                                failures.push(format!("case {n} images {:?}", e.images));
                            }
                        }
                        (Err(_), true) => {}
                        (Ok(e), true) => failures.push(format!("case {n}: expected an error, got {:?}", e.prompt)),
                        (Err(e), false) => failures.push(format!("case {n}: {e}")),
                    }
                }
                Some("parse") => {
                    let text = case.get("text").and_then(Json::as_str).unwrap();
                    match (parse_reply(text, mode), want_err) {
                        (Ok(r), false) => {
                            let m = case.get("message").unwrap();
                            let calls: Vec<ToolCall> = m
                                .get("tool_calls")
                                .and_then(Json::as_array)
                                .unwrap()
                                .iter()
                                .map(|c| ToolCall {
                                    name: c.get("function").and_then(|f| f.get("name")).and_then(Json::as_str).unwrap().into(),
                                    namespace: c.get("namespace").and_then(Json::as_str).map(str::to_string),
                                    arguments: c.get("function").and_then(|f| f.get("arguments")).and_then(Json::as_str).unwrap().into(),
                                })
                                .collect();
                            let want = Reply {
                                content: m.get("content").and_then(Json::as_str).unwrap().into(),
                                reasoning: m.get("reasoning_content").and_then(Json::as_str).unwrap().into(),
                                tool_calls: calls,
                            };
                            if r != want {
                                failures.push(format!("case {n} parse\n want {want:?}\n got  {r:?}"));
                            }
                        }
                        (Err(_), true) => {}
                        (Ok(r), true) => failures.push(format!("case {n}: expected a parse error, got {r:?}")),
                        (Err(e), false) => failures.push(format!("case {n} parse: {e}")),
                    }
                }
                _ => panic!("unknown case kind"),
            }
        }
        assert!(cases.len() > 400);
        assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.len(), failures[..failures.len().min(6)].join("\n"));
        eprintln!("{} reference chat cases match", cases.len());
    }
}
