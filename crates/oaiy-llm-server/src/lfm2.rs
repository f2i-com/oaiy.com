//! LiquidAI's LFM2 chat format, as LFM2.5-24B-A2B's `chat_template.jinja` writes it: ChatML after
//! `<|startoftext|>`, the tools listed in the system message as JSON (`List of tools: [...]`), a reply that opens on
//! `<think>` (LFM2.5 always thinks first), and calls written as a Python list between `<|tool_call_start|>` and
//! `<|tool_call_end|>` (`[get_weather(city='Paris', days=3)]`), read into the server's tool markup (DSML) as Qwen's XML
//! calls are (`qwen::normalize`). The model runs as a Qwen3.5 hybrid (`crate::llm::Qwen35Model`), so its engine is
//! Qwen's; only its prompts and its calls are its own.
use crate::dsv41::chat::{Encoded, Mode, Options, DSML};
use oaiy_engine::json::Json;

pub const CALL_START: &str = "<|tool_call_start|>";
pub const CALL_END: &str = "<|tool_call_end|>";

/// A message's text, as the template's `parse_content` makes it: a string as it is, a list's text parts joined (any
/// other part as its JSON); this model reads no pictures.
fn text(value: Option<&Json>) -> Result<String, String> {
    match value {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => {
            let mut out = String::new();
            for p in parts {
                match (p, p.get("type").and_then(Json::as_str)) {
                    (Json::Str(s), _) => out.push_str(s),
                    (_, Some("text")) => out.push_str(p.get("text").and_then(Json::as_str).unwrap_or("")),
                    (_, Some("image_url" | "image" | "input_image")) => return Err("LFM2 reads no pictures".into()),
                    _ => out.push_str(&p.to_python_json()),
                }
            }
            Ok(out)
        }
        Some(other) => Ok(other.to_python_json()),
    }
}

fn role(m: &Json) -> &str {
    m.get("role").and_then(Json::as_str).unwrap_or("user")
}

/// A call's argument as the template's `format_arg_value` writes it: a string single-quoted (its backslashes,
/// quotes and line ends escaped), a list or object as JSON, anything else as Python's `str` has it.
fn py_arg(v: &Json) -> String {
    match v {
        Json::Str(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'").replace('\n', "\\n").replace('\r', "\\r")),
        Json::Null => "None".into(),
        Json::Bool(b) => if *b { "True" } else { "False" }.into(),
        _ => v.to_python_json(),
    }
}

fn identifier(s: &str) -> Result<(), String> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || "_-.:".contains(c)) { Err("invalid tool/parameter name".into()) } else { Ok(()) }
}

/// An assistant message's calls, as the template's `render_tool_calls` writes them.
fn render_calls(calls: &[Json]) -> Result<String, String> {
    let mut each = Vec::new();
    for call in calls {
        let f = call.get("function").unwrap_or(call);
        let name = f.get("name").and_then(Json::as_str).ok_or("tool call without name")?;
        identifier(name)?;
        let args = match f.get("arguments") {
            Some(Json::Str(s)) if matches!(s.trim(), "" | "{}" | "null") => Vec::new(),
            Some(Json::Str(s)) => match Json::parse(s.as_bytes()).map_err(|e| e.to_string())? {
                Json::Obj(a) => a,
                _ => return Err("tool arguments must be an object".into()),
            },
            Some(Json::Obj(a)) => a.clone(),
            None | Some(Json::Null) => Vec::new(),
            Some(_) => return Err("tool arguments must be an object".into()),
        };
        let args: Vec<String> = args.iter().map(|(k, v)| format!("{k}={}", py_arg(v))).collect();
        each.push(format!("{name}({})", args.join(", ")));
    }
    Ok(format!("{CALL_START}[{}]{CALL_END}", each.join(", ")))
}

/// The prompt for a chat request: the template's, with the reply opened on `<think>` (and, where the request does not
/// think, closed at once). The tools are the first message's (`tools`, where the server puts them, on a system message
/// it adds if there is none).
pub fn chat_prompt(msgs: &[Json], opts: &Options) -> Result<Encoded, String> {
    let mut prompt = String::from("<|startoftext|>");
    let (mut system, mut tools, mut rest) = (String::new(), &[][..], msgs);
    if let Some(first) = msgs.first().filter(|m| role(m) == "system") {
        system = text(first.get("content_blocks").or_else(|| first.get("content")))?;
        tools = first.get("tools").and_then(Json::as_array).unwrap_or(&[]);
        rest = &msgs[1..];
    }
    if !tools.is_empty() {
        if !system.is_empty() {
            system.push('\n');
        }
        let listed: Vec<String> = tools.iter().map(|t| t.as_str().map_or_else(|| t.to_python_json(), str::to_owned)).collect();
        system.push_str(&format!("List of tools: [{}]", listed.join(", ")));
    }
    if !system.is_empty() {
        prompt.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }
    // a reply's thinking is kept after the last user message (a turn of calls and their results), dropped before
    let last_user = rest.iter().rposition(|m| role(m) == "user");
    for (i, m) in rest.iter().enumerate() {
        let r = role(m);
        if !matches!(r, "system" | "user" | "assistant" | "tool") {
            return Err("unsupported LFM2 message role".into());
        }
        prompt.push_str(&format!("<|im_start|>{r}\n"));
        let content = text(m.get("content_blocks").or_else(|| m.get("content")))?;
        if r == "assistant" {
            let keep = last_user.is_none_or(|u| i > u);
            let thinking = ["thinking", "reasoning", "reasoning_content"].iter().find_map(|k| m.get(k).and_then(Json::as_str).filter(|s| !s.is_empty()));
            if let (true, Some(t)) = (keep, thinking) {
                prompt.push_str(&format!("<think>{t}</think>"));
            }
            let content = match content.rsplit_once("</think>") {
                Some((_, answer)) if !keep => answer.trim().to_owned(),
                _ => content,
            };
            prompt.push_str(&content);
            if let Some(calls) = m.get("tool_calls").and_then(Json::as_array).filter(|c| !c.is_empty()) {
                prompt.push_str(&render_calls(calls)?);
            }
        } else {
            prompt.push_str(&content);
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>");
    if opts.mode != Mode::Thinking {
        prompt.push_str("</think>");
    }
    Ok(Encoded { prompt, images: Vec::new() })
}

/// A Python literal's reader (the values of LFM2's calls): strings in either quote (Python's escapes), numbers,
/// `True`, `False`, `None` (JSON's `true`, `false`, `null` too), lists, tuples and dicts.
struct Py<'a> {
    s: &'a [u8],
    at: usize,
}

impl<'a> Py<'a> {
    fn ws(&mut self) {
        while self.at < self.s.len() && self.s[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.at).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at {} of an LFM2 call", c as char, self.at))
        }
    }

    fn name(&mut self) -> Result<String, String> {
        self.ws();
        let from = self.at;
        while self.at < self.s.len() && (self.s[self.at].is_ascii_alphanumeric() || b"_-.:".contains(&self.s[self.at])) {
            self.at += 1;
        }
        if from == self.at {
            return Err(format!("expected a name at {from} of an LFM2 call"));
        }
        Ok(String::from_utf8_lossy(&self.s[from..self.at]).into_owned())
    }

    fn string(&mut self) -> Result<String, String> {
        let q = self.s[self.at];
        let triple = self.s[self.at..].starts_with(&[q, q, q]);
        self.at += if triple { 3 } else { 1 };
        let mut out = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.at) else { return Err("an unterminated string in an LFM2 call".into()) };
            if c == q && (!triple || self.s[self.at..].starts_with(&[q, q, q])) {
                self.at += if triple { 3 } else { 1 };
                break;
            }
            self.at += 1;
            if c != b'\\' {
                out.push(c);
                continue;
            }
            let Some(&e) = self.s.get(self.at) else { return Err("an unterminated string in an LFM2 call".into()) };
            self.at += 1;
            let mut ch = |v: u32| -> Result<(), String> {
                let c = char::from_u32(v).ok_or("an invalid escape in an LFM2 call")?;
                out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                Ok(())
            };
            match e {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'b' => out.push(8),
                b'f' => out.push(12),
                b'0' => out.push(0),
                b'\n' => {}
                b'x' | b'u' | b'U' => {
                    let n = match e { b'x' => 2, b'u' => 4, _ => 8 };
                    let hex = std::str::from_utf8(self.s.get(self.at..self.at + n).ok_or("a short escape in an LFM2 call")?).map_err(|e| e.to_string())?;
                    self.at += n;
                    ch(u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?)?;
                }
                // `\\`, `\'`, `\"` and anything else: the character itself
                other => out.push(other),
            }
        }
        String::from_utf8(out).map_err(|e| e.to_string())
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek().ok_or("a value missing in an LFM2 call")? {
            b'\'' | b'"' => Ok(Json::Str(self.string()?)),
            open @ (b'[' | b'(') => {
                let close = if open == b'[' { b']' } else { b')' };
                self.at += 1;
                let mut items = Vec::new();
                while self.peek() != Some(close) {
                    items.push(self.value()?);
                    if self.peek() == Some(b',') {
                        self.at += 1;
                    } else {
                        break;
                    }
                }
                self.eat(close)?;
                Ok(Json::Arr(items))
            }
            b'{' => {
                self.at += 1;
                let mut members = Vec::new();
                while self.peek() != Some(b'}') {
                    let key = match self.value()? {
                        Json::Str(s) => s,
                        other => other.to_json(),
                    };
                    self.eat(b':')?;
                    members.push((key, self.value()?));
                    if self.peek() == Some(b',') {
                        self.at += 1;
                    } else {
                        break;
                    }
                }
                self.eat(b'}')?;
                Ok(Json::Obj(members))
            }
            _ => {
                let from = self.at;
                while self.at < self.s.len() && (self.s[self.at].is_ascii_alphanumeric() || b"+-._".contains(&self.s[self.at])) {
                    self.at += 1;
                }
                let word = std::str::from_utf8(&self.s[from..self.at]).map_err(|e| e.to_string())?;
                match word {
                    "True" | "true" => Ok(Json::Bool(true)),
                    "False" | "false" => Ok(Json::Bool(false)),
                    "None" | "null" => Ok(Json::Null),
                    w => {
                        let w = w.replace('_', "");
                        if let Ok(i) = w.parse::<i64>() {
                            Ok(Json::Int(i))
                        } else {
                            w.parse::<f64>().ok().filter(|v| v.is_finite()).map(Json::Num).ok_or_else(|| format!("{word:?} is no value an LFM2 call can hold"))
                        }
                    }
                }
            }
        }
    }

    /// `name(key=value, ...)`.
    fn call(&mut self) -> Result<(String, Vec<(String, Json)>), String> {
        let name = self.name()?;
        self.eat(b'(')?;
        let mut args = Vec::new();
        while self.peek() != Some(b')') {
            let key = self.name()?;
            self.eat(b'=')?;
            args.push((key, self.value()?));
            if self.peek() == Some(b',') {
                self.at += 1;
            } else {
                break;
            }
        }
        self.eat(b')')?;
        Ok((name, args))
    }
}

/// The calls between the tags: a Python list of calls (or a call, or calls, without the list), or (where the system
/// prompt asked for JSON) a JSON list of `{"name", "arguments"}`.
fn parse_calls(body: &str) -> Result<Vec<(String, Vec<(String, Json)>)>, String> {
    let body = body.trim();
    if let Ok(Json::Arr(items)) = Json::parse(body.as_bytes()) {
        if !items.is_empty() && items.iter().all(|i| i.get("name").and_then(Json::as_str).is_some()) {
            return items
                .iter()
                .map(|i| {
                    let name = i.get("name").and_then(Json::as_str).unwrap_or_default().to_owned();
                    match i.get("arguments").or_else(|| i.get("parameters")) {
                        Some(Json::Obj(a)) => Ok((name, a.clone())),
                        None | Some(Json::Null) => Ok((name, Vec::new())),
                        Some(_) => Err("an LFM2 call's arguments are not an object".into()),
                    }
                })
                .collect();
        }
    }
    let mut p = Py { s: body.as_bytes(), at: 0 };
    let listed = p.peek() == Some(b'[');
    if listed {
        p.at += 1;
    }
    let mut calls = Vec::new();
    loop {
        if listed && p.peek() == Some(b']') {
            break;
        }
        calls.push(p.call()?);
        if p.peek() == Some(b',') {
            p.at += 1;
        } else {
            break;
        }
    }
    if listed {
        p.eat(b']')?;
    }
    p.ws();
    if p.at != body.len() {
        return Err(format!("unexpected text after LFM2's calls: {:?}", &body[p.at..]));
    }
    if calls.is_empty() {
        return Err("an LFM2 call list with no calls".into());
    }
    Ok(calls)
}

/// A complete reply's calls (each block between `<|tool_call_start|>` and `<|tool_call_end|>`) in the server's tool
/// markup, after the reply's text (the text around them, in order); each call a declared function's, its arguments
/// typed as the model wrote them (a string where the function's parameter is a string and the model wrote a number or
/// a boolean). A reply with no calls is as it is; one whose call is cut short or malformed is refused.
pub fn normalize(text: &str, tools: &[Json]) -> Result<String, String> {
    if text.contains(DSML) {
        return Err("unexpected DSML in LFM2 output".into());
    }
    if !text.contains(CALL_START) {
        if text.contains(CALL_END) {
            return Err("an LFM2 tool call's end with no start".into());
        }
        return Ok(text.to_owned());
    }
    let (mut prose, mut calls, mut rest) = (String::new(), Vec::new(), text);
    while let Some((before, after)) = rest.split_once(CALL_START) {
        prose.push_str(before);
        let (body, tail) = after.split_once(CALL_END).ok_or("incomplete LFM2 tool call")?;
        calls.extend(parse_calls(body)?);
        rest = tail;
    }
    prose.push_str(rest);
    let mut out = format!("{prose}<{DSML} calls>");
    for (name, args) in calls {
        identifier(&name)?;
        let schema = tools.iter().filter_map(|t| t.get("function")).find(|f| f.get("name").and_then(Json::as_str) == Some(name.as_str())).ok_or("LFM2 called an undeclared function")?;
        out.push_str(&format!("<{DSML} invoke name=\"{name}\">"));
        let mut seen = std::collections::BTreeSet::new();
        for (key, value) in args {
            identifier(&key)?;
            if !seen.insert(key.clone()) {
                return Err("duplicate LFM2 parameter".into());
            }
            let typ = schema.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.get(&key)).and_then(|p| p.get("type"));
            let string_only = typ.and_then(Json::as_str) == Some("string");
            let (is_string, raw) = match &value {
                Json::Str(s) => (true, s.clone()),
                Json::Int(_) | Json::Num(_) | Json::Bool(_) if string_only => (true, value.to_json()),
                other => (false, other.to_json()),
            };
            out.push_str(&format!("<{DSML} parameter name=\"{key}\" string=\"{is_string}\">{raw}</{DSML} parameter>"));
        }
        out.push_str(&format!("</{DSML} invoke>"));
    }
    out.push_str(&format!("</{DSML} calls>"));
    Ok(out)
}

#[cfg(test)]
mod tests;
