//! Native Qwen dense hybrid inference, image embeddings and XML tool protocol.

// Host checkpoints of the hybrid attention/recurrent cache.
pub(crate) mod cache;
// Conversations Qwen3.8-Flash-Next sets aside in host RAM.
pub(crate) mod park;
// The vision tower an EXL3 checkpoint comes with (OrcaSAQ's, Flash-Next's), on the model's backend.
#[cfg(feature = "webgpu")]
pub(crate) mod vision;
#[cfg(test)]
mod real;

use std::sync::{mpsc::Receiver, atomic::Ordering};
use crate::dsv41::chat::{Encoded, Mode, Options, DSML};
use ggml_rs::Tensor;
use std::sync::Arc;
use crate::engines::qwen::cache::{Snapshot, RecurrentSnapshot, WholeBuilder};
use crate::engines::qwen::park::{Held, Parking};
use crate::llm::{Model, MmProj, MmProjConfig, KvCache};
use oaiy_engine::json::Json;
use crate::engines::job::{Event, Finish, ImagePrep, Job, JobImage, sample};

const IMAGE: &str = "<|vision_start|><|image_pad|><|vision_end|>";
// Amortize EXL3 weight reconstruction across more image/prompt rows. Bounded
// to keep activations small alongside the configured long-context KV cache.
pub(crate) const PREFILL_CHUNK: usize = 512;

/// How a picture stands in a Qwen prompt: Qwen3.5's own `<|vision_start|><|image_pad|><|vision_end|>` beside its
/// text, or MiniCPM-V 4.7's bare `<|image_pad|>`, a message's parts each on a line of its own and the whole trimmed (its
/// template's `join('\n')` and `trim`). Qwen3-VL's stand as Qwen3.5's, in a template with no thought: an instruct
/// model's turns open and close without the `<think>` block Qwen3.5's carry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pictures { Qwen, MiniCpmV, Qwen3Vl }

fn content(value: Option<&Json>, images: &mut Vec<Json>, style: Pictures) -> Result<String, String> {
    match value {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => {
            if s.contains("<|image_pad|>") { return Err("send images as image_url blocks, not literal placeholders".into()); }
            Ok(s.clone())
        }
        Some(Json::Arr(parts)) => {
            let mut out = Vec::new();
            for p in parts {
                match p.get("type").and_then(Json::as_str) {
                    Some("text" | "input_text") => out.push(content(p.get("text"), images, style)?),
                    Some("image_url" | "input_image" | "image") => {
                        let rec = if let Some(url) = p.get("image_url") {
                            Json::obj([("url", url.get("url").unwrap_or(url).clone())])
                        } else { p.clone() };
                        images.push(rec);
                        out.push(match style { Pictures::Qwen | Pictures::Qwen3Vl => IMAGE, Pictures::MiniCpmV => "<|image_pad|>" }.to_owned());
                    }
                    _ => return Err("unsupported Qwen content block".into()),
                }
            }
            Ok(match style { Pictures::Qwen | Pictures::Qwen3Vl => out.concat(), Pictures::MiniCpmV => out.join("\n").trim().to_owned() })
        }
        _ => Err("message content must be text or blocks".into()),
    }
}

pub fn chat_prompt(msgs: &[Json], opts: &Options) -> Result<Encoded, String> {
    chat_prompt_as(msgs, opts, Pictures::Qwen)
}

/// [`chat_prompt`], its pictures standing as `style` has them.
pub fn chat_prompt_as(msgs: &[Json], opts: &Options, style: Pictures) -> Result<Encoded, String> {
    let mut prompt = String::new();
    let mut images = Vec::new();
    for (i,m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Json::as_str).unwrap_or("user");
        if !matches!(role, "system" | "user" | "assistant" | "tool") { return Err("unsupported Qwen message role".into()); }
        let before = images.len();
        let text = content(m.get("content_blocks").or_else(|| m.get("content")), &mut images, style)?;
        if role == "system" && images.len() != before { return Err("Qwen system messages cannot contain images".into()); }
        if role == "tool" {
            if i == 0 || msgs[i-1].get("role").and_then(Json::as_str) != Some("tool") { prompt.push_str("<|im_start|>user"); }
            prompt.push_str(&format!("\n<tool_response>\n{text}\n</tool_response>"));
            if i+1 == msgs.len() || msgs[i+1].get("role").and_then(Json::as_str) != Some("tool") { prompt.push_str("<|im_end|>\n"); }
            continue;
        }
        prompt.push_str(&format!("<|im_start|>{role}\n"));
        // (Qwen3-VL's own template: the system's text, then its tools as JSON, each call asked for as JSON in the call's
        // tags, the form it was trained on; a past call shown the same way)
        if style == Pictures::Qwen3Vl {
            prompt.push_str(&text);
            if role == "system" {
                if let Some(tools) = m.get("tools").and_then(Json::as_array).filter(|t| !t.is_empty()) {
                    if !text.is_empty() { prompt.push_str("\n\n"); }
                    prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
                    for t in tools { prompt.push('\n'); prompt.push_str(&t.to_json()); }
                    prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
                }
            }
            if let Some(calls) = m.get("tool_calls").and_then(Json::as_array) {
                for (n, call) in calls.iter().enumerate() {
                    let f = call.get("function").unwrap_or(call);
                    let name = f.get("name").and_then(Json::as_str).ok_or("tool call without name")?;
                    identifier(name)?;
                    let args = match f.get("arguments") {
                        Some(Json::Str(s)) => s.clone(),
                        Some(v) => v.to_json(),
                        None => "{}".into(),
                    };
                    if n > 0 || !text.is_empty() { prompt.push('\n'); }
                    prompt.push_str(&format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"));
                }
            }
            prompt.push_str("<|im_end|>\n");
            continue;
        }
        if role == "system" {
            if let Some(tools) = m.get("tools").and_then(Json::as_array).filter(|t| !t.is_empty()) {
                prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
                for t in tools { prompt.push_str(&t.to_json()); prompt.push('\n'); }
                prompt.push_str("</tools>\n\nCall functions with <tool_call>\n<function=FUNCTION_NAME>\n<parameter=PARAMETER_NAME>\nvalue\n</parameter>\n</function>\n</tool_call>. Supply every required parameter. Objects, arrays, numbers and booleans must be valid JSON; strings are literal text. No suffix after a function call.\n\n");
            }
            if opts.mode == Mode::Thinking && opts.effort <= 33 { prompt.push_str("Keep your thinking brief and focused.\n\n"); }
        }
        if role == "assistant" && style != Pictures::Qwen3Vl { prompt.push_str("<think>\n\n</think>\n\n"); }
        prompt.push_str(&text);
        if let Some(calls) = m.get("tool_calls").and_then(Json::as_array) {
            for call in calls {
                let f = call.get("function").unwrap_or(call);
                let name = f.get("name").and_then(Json::as_str).ok_or("tool call without name")?;
                identifier(name)?;
                let args = match f.get("arguments") {
                    Some(Json::Str(s)) => Json::parse(s.as_bytes()).map_err(|e| e.to_string())?,
                    Some(v) => v.clone(), None => Json::Obj(vec![]),
                };
                let Json::Obj(args) = args else { return Err("tool arguments must be an object".into()); };
                prompt.push_str(&format!("<tool_call>\n<function={name}>\n"));
                for (k,v) in args {
                    identifier(&k)?;
                    let value = v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_json());
                    prompt.push_str(&format!("<parameter={k}>\n{value}\n</parameter>\n"));
                }
                prompt.push_str("</function>\n</tool_call>");
            }
        }
        prompt.push_str("<|im_end|>\n");
    }
    if style == Pictures::Qwen3Vl {
        prompt.push_str("<|im_start|>assistant\n");
        return Ok(Encoded { prompt, images });
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n");
    if opts.mode != Mode::Thinking { prompt.push_str("\n</think>\n\n"); }
    Ok(Encoded { prompt, images })
}

fn identifier(s: &str) -> Result<(), String> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || "_-.:".contains(c)) { Err("invalid tool/parameter name".into()) } else { Ok(()) }
}

/// Qwen sometimes opens a call with the parameter tag where the function tag
/// belongs: `<parameter=update_plan>` then the parameters. When the name is a
/// declared function, read it as `<function=update_plan>`, and set right the
/// closing tag that leaves (one `</parameter>` too many, or no `</function>`).
/// Anything else stays as written, to be refused.
fn repair_function_tag(block: &str, tools: &[Json]) -> Option<String> {
    let block = block.trim();
    if block.starts_with("<function=") {
        return None;
    }
    // `<parameter=NAME>` in place of the function tag leaves one `</parameter>`
    // too many; `function=NAME>` behind a stray special token (`<|im_start|>`)
    // or with no `<` at all leaves the rest as it should be.
    let (name, body, parameter_tag) = match block.strip_prefix("<parameter=") {
        Some(rest) => {
            let (name, body) = rest.split_once('>')?;
            (name, body, true)
        }
        None => {
            let (name, body) = without_special_token(block).strip_prefix("function=")?.split_once('>')?;
            (name, body, false)
        }
    };
    tools.iter().filter_map(|t| t.get("function")).find(|f| f.get("name").and_then(Json::as_str) == Some(name))?;
    let mut body = body.trim_end();
    if let Some(inner) = body.strip_suffix("</function>") {
        body = inner.trim_end();
    }
    let (opens, closes) = (body.matches("<parameter=").count(), body.matches("</parameter>").count());
    let body = if parameter_tag && closes == opens + 1 {
        body[..body.rfind("</parameter>")?].trim_end()
    } else if closes == opens {
        body
    } else {
        return None;
    };
    Some(format!("<function={name}>{body}\n</function>"))
}

/// Text after a leading chat-template token such as `<|im_start|>`, or the text as it is.
fn without_special_token(s: &str) -> &str {
    s.strip_prefix("<|")
        .and_then(|rest| rest.split_once("|>"))
        .filter(|(token, _)| !token.is_empty() && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .map_or(s, |(_, rest)| rest.trim_start())
}

/// Convert a complete, schema-typed native call to the server's existing
/// strict tool parser. Partial/malformed calls never become executable calls.
pub fn normalize(text: &str, tools: &[Json]) -> Result<String, String> {
    if text.contains(DSML) { return Err("unexpected DSML in Qwen output".into()); }
    let Some((prefix, rest)) = text.split_once("<tool_call>") else {
        if text.contains("<tool_call") { return Err("incomplete Qwen tool call".into()); }
        return Ok(text.to_owned());
    };
    let mut rest = rest;
    let mut out = format!("{prefix}<{DSML} calls>");
    loop {
        let (block, tail) = rest.split_once("</tool_call>").ok_or("incomplete Qwen tool call")?;
        // (Qwen3's and Qwen3-VL's own form: `{"name": ..., "arguments": {...}}` in the call's tags)
        if block.trim().starts_with('{') {
            out.push_str(&json_call(block, tools)?);
            if tail.trim().is_empty() { break; }
            rest = tail.trim().strip_prefix("<tool_call>").ok_or("unexpected suffix after tool call")?;
            continue;
        }
        let repaired = repair_function_tag(block, tools);
        let block = repaired.as_deref().unwrap_or(block);
        let f = block.trim().strip_prefix("<function=").ok_or("missing function tag")?;
        let (name, params) = f.split_once('>').ok_or("incomplete function tag")?;
        identifier(name)?;
        let schema = tools.iter().filter_map(|t| t.get("function")).find(|f| f.get("name").and_then(Json::as_str) == Some(name)).ok_or("Qwen called an undeclared function")?;
        let mut params = params.trim().strip_suffix("</function>").ok_or("missing closing function tag")?.trim();
        out.push_str(&format!("<{DSML} invoke name=\"{name}\">"));
        let mut seen = std::collections::BTreeSet::new();
        while !params.is_empty() {
            let p = params.strip_prefix("<parameter=").ok_or("unexpected content in function call")?;
            let (key, value) = p.split_once('>').ok_or("incomplete parameter tag")?;
            identifier(key)?;
            if !seen.insert(key) { return Err("duplicate Qwen parameter".into()); }
            let (value, tail) = value.split_once("</parameter>").ok_or("incomplete parameter value")?;
            // Remove only the template's framing newline, preserving string whitespace.
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            let typ = schema.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.get(key)).and_then(|p|p.get("type"));
            let is_string = typ.and_then(Json::as_str) == Some("string") || typ.and_then(Json::as_array).is_some_and(|a| a.iter().any(|t| t.as_str() == Some("string")));
            let raw = if is_string { value.to_owned() } else { Json::parse(value.trim().as_bytes()).map_err(|e| format!("invalid JSON parameter {key}: {e}"))?.to_json() };
            out.push_str(&format!("<{DSML} parameter name=\"{key}\" string=\"{is_string}\">{raw}</{DSML} parameter>"));
            params = tail.trim();
        }
        out.push_str(&format!("</{DSML} invoke>"));
        if tail.trim().is_empty() { break; }
        rest = tail.trim().strip_prefix("<tool_call>").ok_or("unexpected suffix after tool call")?;
    }
    out.push_str(&format!("</{DSML} calls>"));
    Ok(out)
}

/// A call in Qwen3's JSON form (`{"name": ..., "arguments": {...}}`, the arguments an object or that object as a
/// string) in the server's markup, each argument a string where the function's schema says it is one.
fn json_call(block: &str, tools: &[Json]) -> Result<String, String> {
    let call = Json::parse(block.trim().as_bytes()).map_err(|e| format!("invalid JSON tool call: {e}"))?;
    let name = call.get("name").and_then(Json::as_str).ok_or("a JSON tool call without its name")?;
    identifier(name)?;
    let schema = tools.iter().filter_map(|t| t.get("function")).find(|f| f.get("name").and_then(Json::as_str) == Some(name)).ok_or("Qwen called an undeclared function")?;
    let args = match call.get("arguments").or_else(|| call.get("parameters")) {
        Some(Json::Str(s)) => Json::parse(s.as_bytes()).map_err(|e| format!("invalid JSON arguments: {e}"))?,
        Some(v) => v.clone(),
        None => Json::Obj(vec![]),
    };
    let Json::Obj(args) = args else { return Err("tool arguments must be an object".into()) };
    let mut out = format!("<{DSML} invoke name=\"{name}\">");
    for (key, value) in args {
        identifier(&key)?;
        let typ = schema.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.get(&key)).and_then(|p| p.get("type"));
        let is_string = typ.and_then(Json::as_str) == Some("string") || typ.and_then(Json::as_array).is_some_and(|a| a.iter().any(|t| t.as_str() == Some("string")));
        let raw = match (&value, is_string) {
            (Json::Str(s), true) => s.clone(),
            _ => value.to_json(),
        };
        out.push_str(&format!("<{DSML} parameter name=\"{key}\" string=\"{is_string}\">{raw}</{DSML} parameter>"));
    }
    out.push_str(&format!("</{DSML} invoke>"));
    Ok(out)
}

/// How far into a prompt of `len` keys its state may be kept on (and taken from) the disk: all of it, but for a
/// prompt with pictures where the disk's namespace does not fingerprint the vision pipeline, the text before its first
/// picture alone.
fn text_until(job: &Job, image_disk_cache: bool, len: usize) -> usize {
    match job.images.iter().map(|i| i.start).min() {
        Some(first) if !image_disk_cache => first,
        _ => len,
    }
}

/// The start of a rejected tool call, as the model wrote it, for the error.
fn call_excerpt(raw: &str) -> String {
    let from = raw.find("<tool_call").unwrap_or(0);
    let call = &raw[from..];
    let mut end = call.len().min(300);
    while !call.is_char_boundary(end) { end -= 1; }
    format!("{:?}{}", &call[..end], if end < call.len() { "…" } else { "" })
}

/// Stream prose and reasoning as tokens arrive. Native tool XML stays buffered
/// until the existing strict parser validates the complete call. A token may
/// end halfway through a UTF-8 character or the opening tool tag.
#[derive(Default)]
struct NativeStream { sent: String, tool_sent: String, calls: Calls }

impl NativeStream {
    fn tool_preview(&mut self, decoded: &str) -> Result<Option<(String, bool)>, String> {
        let Some(at) = decoded.find(self.calls.open().0) else { return Ok(None) };
        let draft = decoded[at..].trim_end_matches('\u{fffd}');
        let delta = draft.strip_prefix(&self.tool_sent).ok_or("Qwen tool preview changed after streaming")?;
        if delta.is_empty() { return Ok(None); }
        let update = (delta.to_owned(), self.tool_sent.is_empty());
        self.tool_sent = draft.to_owned();
        Ok(Some(update))
    }

    fn push(&mut self, decoded: &str, tools: &[Json], complete: bool) -> Result<String, String> {
        let text = if complete { self.calls.normalize(decoded, tools)? } else {
            if decoded.contains(DSML) { return Err("unexpected DSML in Qwen output".into()); }
            let (tag, start) = self.calls.open();
            let end = decoded.find(start).unwrap_or_else(|| {
                let held = (1..tag.len()).rev().find(|&n| decoded.ends_with(&tag[..n])).unwrap_or(0);
                decoded.len()-held
            });
            decoded[..end].trim_end_matches('\u{fffd}').to_owned()
        };
        let delta = text.strip_prefix(&self.sent).ok_or("Qwen decoded text changed after streaming")?.to_owned();
        self.sent = text;
        Ok(delta)
    }
}

pub fn prepare_images(records: &[Json], prompt: Vec<u32>, image_id: u32, cfg: &MmProjConfig, local: bool, max_seq: usize) -> Result<(Vec<u32>, Vec<JobImage>), String> {
    let count = prompt.iter().filter(|&&t| t == image_id).count();
    if count != records.len() { return Err("image placeholder count does not match image blocks".into()); }
    let side = cfg.image_size / cfg.patch_size / 2;
    if prompt.len().saturating_add(count.saturating_mul(side*side-1)) >= max_seq { return Err("images exceed configured context".into()); }
    let vc = crate::llm::VisionConfig { image_size: cfg.image_size, patch_size: cfg.patch_size, mean: cfg.mean, std: cfg.std };
    let mut ids = Vec::new(); let mut images = Vec::new(); let mut records = records.iter();
    for id in prompt {
        if id != image_id { ids.push(id); continue; }
        let bytes = crate::dsv41::vision::image_bytes(records.next().unwrap(), local).map_err(|e| e.to_string())?;
        let pixels = crate::llm::vision::preprocess_image_letterboxed(&bytes, &vc).map_err(|e| e.to_string())?;
        let hash=crate::engines::disk::fnv(&bytes,0);
        images.push(JobImage { start: ids.len(), prep: ImagePrep::Qwen { pixels, side }, hash });
        ids.extend(std::iter::repeat_n(image_id, side*side));
    }
    Ok((ids, images))
}

/// A MiniCPM-V 4.7 model's pictures: its markers' token ids, the patches a side each token is (4, or 2 in 4x), the
/// most slices a picture is cut into, and its tower's mean and spread.
#[derive(Clone, Debug)]
pub struct MiniCpmV {
    pub markers: crate::llm::vision::mmproj::minicpmv::Markers,
    pub merge: usize,
    pub max_slices: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// A MiniCPM-V 4.7 prompt's pictures, as its processor lays them out: each `<|image_pad|>` becomes
/// `<image_id>N</image_id>` (`encode`'s tokens; N the picture's count in the prompt), then `<image>`, the overview's
/// tokens and `</image>`, then each row of slices, every slice's tokens between `<slice>` and `</slice>`, the rows apart
/// by a newline. Each tile is an image of the job, its tokens in a run of their own.
pub fn prepare_minicpmv(records: &[Json], prompt: Vec<u32>, image_id: u32, m: &MiniCpmV, encode: impl Fn(&str) -> Vec<u32>, local: bool, max_seq: usize) -> Result<(Vec<u32>, Vec<JobImage>), String> {
    use crate::llm::vision::mmproj::minicpmv;
    let count = prompt.iter().filter(|&&t| t == image_id).count();
    if count != records.len() { return Err("image placeholder count does not match image blocks".into()); }
    let (mut ids, mut images, mut records) = (Vec::new(), Vec::new(), records.iter());
    for (picture, id) in prompt.into_iter().scan(0usize, |n, id| { let at = *n; *n += (id == image_id) as usize; Some((at, id)) }) {
        if id != image_id { ids.push(id); continue; }
        let bytes = crate::dsv41::vision::image_bytes(records.next().unwrap(), local).map_err(|e| e.to_string())?;
        let t = minicpmv::tiles_from_bytes(&bytes, m.max_slices, m.mean, m.std).map_err(|e| e.to_string())?;
        let sizes = t.plan.target_sizes();
        let mut tiles = t.tiles.into_iter().zip(sizes).enumerate();
        let mut tile = |ids: &mut Vec<u32>, images: &mut Vec<JobImage>| {
            let (k, (pixels, size)) = tiles.next().expect("a tile for each place");
            let prep = ImagePrep::MiniCpmV { pixels, size, merge: m.merge };
            let n = prep.n_tokens();
            images.push(JobImage { start: ids.len(), prep, hash: crate::engines::disk::fnv(&bytes, k as u64 + 1) });
            ids.extend(std::iter::repeat_n(image_id, n));
        };
        ids.extend(encode(&format!("<image_id>{picture}</image_id>")));
        ids.push(m.markers.image_start);
        tile(&mut ids, &mut images);
        ids.push(m.markers.image_end);
        if let Some((rows, cols)) = t.plan.grid {
            for r in 0..rows {
                if r > 0 { ids.push(m.markers.newline); }
                for _ in 0..cols {
                    ids.push(m.markers.slice_start);
                    tile(&mut ids, &mut images);
                    ids.push(m.markers.slice_end);
                }
            }
        }
        if ids.len() >= max_seq { return Err("images exceed configured context".into()); }
    }
    Ok((ids, images))
}

/// What the engine runs: a Qwen3.5 hybrid (`llm`), or Qwen3.8-Flash-Next.
pub enum Hybrid {
    Qwen35(Model),
    #[cfg(feature = "webgpu")]
    Flash(Box<crate::engines::flashnext::FlashNext>),
    #[cfg(test)]
    Fake(Box<crate::engines::qwen::real::FakeModel>),
}

impl From<Model> for Hybrid {
    fn from(m: Model) -> Self { Self::Qwen35(m) }
}

impl Hybrid {
    /// The Qwen3.5 model: its checkpoints and disk states are Qwen3.5's shape.
    fn qwen35(&self) -> Option<&crate::llm::Qwen35Model> {
        match self { Self::Qwen35(Model::Qwen35(m)) => Some(m), _ => None }
    }
    /// Qwen3.8-Flash-Next's cache: which of its slots hold keys and values a token (`FlashNext::row_slots`), for a
    /// whole state of it on the disk. None for the other models (the Qwen3.5 hybrid's states are its own shape).
    fn row_slots(&self) -> Option<Vec<bool>> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => Some(f.row_slots()),
            _ => None,
        }
    }
    fn tokenizer(&self) -> Result<&tokenizer::Tokenizer, String> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => Ok(&m.tokenizer),
            Self::Qwen35(Model::Qwen3(m)) => Ok(&m.tokenizer),
            Self::Qwen35(_) => Err("not a dense Qwen hybrid".into()),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => Ok(&f.tokenizer),
            #[cfg(test)]
            Self::Fake(f) => Ok(&f.tok),
        }
    }
    fn width(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.config().embedding_dim,
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.config.hidden,
            #[cfg(test)]
            Self::Fake(f) => f.width,
        }
    }
    /// Whether the engine may set conversations aside in RAM: Flash-Next, which keeps one
    /// conversation in memory and nothing on disk. The Qwen3.5 hybrid has its checkpoints and
    /// disk states and goes on as it did.
    fn parks(&self) -> bool {
        match self {
            Self::Qwen35(_) => false,
            #[cfg(feature = "webgpu")]
            Self::Flash(_) => true,
            #[cfg(test)]
            Self::Fake(_) => true,
        }
    }
    fn new_kv_cache(&self, max_seq: usize) -> KvCache {
        match self {
            Self::Qwen35(m) => m.new_kv_cache(max_seq),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.new_kv_cache(max_seq),
            #[cfg(test)]
            Self::Fake(_) => crate::engines::qwen::real::new_kv(max_seq),
        }
    }
    /// Token embeddings on the host.
    fn embed(&self, tokens: &[u32]) -> Result<Tensor, String> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => Ok(m.embed_text(tokens).to_host()),
            Self::Qwen35(Model::Qwen3(m)) => Ok(m.embed_text(tokens)),
            Self::Qwen35(_) => Err("not a dense Qwen hybrid".into()),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.embed_text(tokens).map_err(|e| e.to_string()),
            #[cfg(test)]
            Self::Fake(f) => Ok(Tensor::zeros(vec![tokens.len(), f.width])),
        }
    }
    /// Whether the model drafts tokens a check takes (its multi-token-prediction layer, chained).
    fn drafts(&self) -> bool {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.drafts(),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.drafts(),
            _ => false,
        }
    }
    /// The tokens before a draft's `next` it may need (its layer catches up on the last run's rows): at most this many.
    fn draft_window(&self) -> usize {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Flash(_) => crate::engines::flashnext::CHECK_ROWS,
            _ => crate::llm::SPEC_ROWS,
        }
    }
    /// Up to `k` drafts after the last of `recent` (the token sampled for position `kv.len`); None or none: a step.
    fn draft(&self, kv: &KvCache, recent: &[u32], k: usize) -> Option<Vec<u32>> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.draft(kv, recent, k),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.draft(kv, recent, k),
            _ => None,
        }
    }
    /// A check of `rows` (the token sampled, then its drafts): every row's logits, undoable ([`Self::rollback`]).
    fn check(&self, rows: &[u32], kv: &mut KvCache) -> Option<Vec<Tensor>> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.check(rows, kv),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.check(rows, kv),
            _ => None,
        }
    }
    /// [`Self::check`] for a request that samples greedily, where the model picks each row's token on its GPU
    /// (Flash-Next chained): the tokens alone. None: nothing run, [`Self::check`] the caller's.
    fn check_picks(&self, rows: &[u32], kv: &mut KvCache) -> Option<Vec<u32>> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.check_picks(rows, kv),
            _ => {
                let _ = (rows, kv);
                None
            }
        }
    }
    /// A decode step's next token for a request that samples greedily, as [`Self::check_picks`]. None: nothing run,
    /// [`Self::forward`] the caller's.
    fn step_pick(&self, token: u32, kv: &mut KvCache) -> Option<u32> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => {
                let embeds = f.embed_text(&[token]).ok()?;
                f.step_pick(token, &embeds, kv)
            }
            _ => {
                let _ = (token, kv);
                None
            }
        }
    }
    /// Undo the last check's `rows` past its first `keep`.
    fn rollback(&self, kv: &mut KvCache, rows: usize, keep: usize) {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.rollback(kv, rows, keep),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => {
                let _ = rows;
                f.rollback(kv, keep)
            }
            _ => {}
        }
    }
    /// A prompt's text chunks (each embedded, on the host) after what `kv` holds, as many at once as the model runs
    /// so (Qwen3.8-Flash-Next: each chunk's first devices' layers as its last device runs the chunk before's): the
    /// last chunk's logits, `done(i)` once each chunk is in. Another model's, or one Flash-Next cannot chain, a chunk
    /// at a time.
    fn forward_chunks(&self, chunks: &[(&[u32], Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize)) -> Result<Tensor, String> {
        #[cfg(feature = "webgpu")]
        if let Self::Flash(f) = self {
            let len = kv.len;
            let refs: Vec<(&[u32], &Tensor)> = chunks.iter().map(|(t, e)| (*t, e)).collect();
            if let Some(l) = f.forward_chunks(&refs, kv, done) {
                return Ok(l);
            }
            // (what ran before a chunk it could not chain stands: the rest a chunk at a time)
            let ran: usize = chunks.iter().scan(len, |at, (t, _)| { *at += t.len(); Some(*at) }).take_while(|&at| at <= kv.len).count();
            let mut last = None;
            for (i, (t, e)) in chunks.iter().enumerate().skip(ran) {
                last = Some(self.forward(t, e.clone(), kv, None)?);
                done(i);
            }
            return last.ok_or_else(|| "no chunk to run".to_string());
        }
        // the Qwen3.5 hybrid's as one run (its chain records each chunk as the one before runs)
        if let (Self::Qwen35(Model::Qwen35(_)), true) = (self, chunks.len() > 1) {
            let (tokens, emb) = self.joined(chunks);
            let out = self.forward(&tokens, emb, kv, None)?;
            for i in 0..chunks.len() {
                done(i);
            }
            return Ok(out);
        }
        let mut last = None;
        for (i, (t, e)) in chunks.iter().enumerate() {
            last = Some(self.forward(t, e.clone(), kv, None)?);
            done(i);
        }
        last.ok_or_else(|| "no chunk to run".to_string())
    }
    /// The tokens a prompt's chunk has at most ([`PREFILL_CHUNK`]; Flash-Next's own where its devices take more).
    fn prompt_rows(&self) -> usize {
        #[cfg(feature = "webgpu")]
        if let Self::Flash(f) = self {
            return f.prompt_rows();
        }
        PREFILL_CHUNK
    }
    /// Whether the model is chained on GPUs, its kernels' pipelines made as each first runs
    /// ([`QwenEngine::warm_up`]): Flash-Next, a chained Qwen3.5 hybrid.
    /// What the model keeps on its GPU between requests let go ([`crate::llm::Qwen35Chain::let_go_of_spare`]); with
    /// `cache`, its attention cache's copy there too. Other models keep nothing of the kind.
    fn let_go_of_spare(&self, cache: bool) {
        if let Self::Qwen35(Model::Qwen35(m)) = self {
            m.chain.let_go_of_spare(m, cache);
        }
    }

    fn warms(&self) -> bool {
        // (Flash-Next with experts read as records: two chunks of tokens of no meaning would read some 30 GB of
        // records and fill the card's free slots with experts no conversation routes to; its load ran a short
        // prompt and a step already, which made the steps' kernels)
        #[cfg(feature = "webgpu")]
        if let Self::Flash(f) = self {
            return f.rest_layers().0 == 0;
        }
        matches!(self, Self::Qwen35(Model::Qwen35(m)) if m.backend.chain().is_some())
    }
    /// Whether a prompt's chunks go to [`Self::forward_chunks`] together: Flash-Next over several GPUs, a chained
    /// Qwen3.5 hybrid. (One card that runs every layer's experts gains nothing by it: 3.1 to 3.5 s for 4,086 tokens
    /// either way, and two chunks' vectors and scratch at once are 1.5 GiB more of a card that has none to spare.)
    fn pipelines(&self) -> bool {
        #[cfg(feature = "webgpu")]
        if let Self::Flash(f) = self {
            return f.devices_len() > 1;
        }
        matches!(self, Self::Qwen35(Model::Qwen35(m)) if m.backend.chain().is_some())
    }
    /// Run `tokens` (embedded as `embeds`, on the host) after what `kv` holds: the last logits, on the host.
    /// `chunks`' tokens one after another and their embeddings side by side (copied on every core: 64 MB a run of
    /// 3,150 tokens, 10 ms on one).
    fn joined(&self, chunks: &[(&[u32], Tensor)]) -> (Vec<u32>, Tensor) {
        use rayon::prelude::*;
        let width = self.width();
        let rows: usize = chunks.iter().map(|(t, _)| t.len()).sum();
        let mut emb = vec![0f32; rows * width];
        let mut parts: Vec<&mut [f32]> = Vec::with_capacity(chunks.len());
        let mut rest = &mut emb[..];
        for (t, _) in chunks {
            let (part, after) = rest.split_at_mut(t.len() * width);
            parts.push(part);
            rest = after;
        }
        parts.into_par_iter().zip(chunks.par_iter()).for_each(|(part, (_, e))| part.copy_from_slice(e.data()));
        let mut tokens = Vec::with_capacity(rows);
        for (t, _) in chunks {
            tokens.extend_from_slice(t);
        }
        (tokens, Tensor::from_vec(emb, vec![rows, width]))
    }
    /// Whether a run of `rows` after what `kv` holds can keep its recurrent states from inside it once each of `taps`
    /// rows is in ([`Self::forward_chunks_tapped`]): a chained Qwen3.5 hybrid's, where its chain says so.
    fn can_tap(&self, rows: usize, kv: &KvCache, taps: &[usize]) -> bool {
        #[cfg(feature = "webgpu")]
        if let Self::Flash(f) = self {
            return f.can_tap(rows, kv, taps);
        }
        matches!(self, Self::Qwen35(Model::Qwen35(m)) if m.can_tap(rows, kv, taps))
    }
    /// [`Self::forward_chunks`] with the recurrent states as they are once each of `taps` rows of the chunks is in
    /// (what a checkpoint there holds, the run not stopping for it), where [`Self::can_tap`] said it can.
    /// [`Self::forward_chunks`] and [`Self::forward_chunks_tapped`] (no `taps`: the former) for a dense Qwen hybrid's
    /// rows at `at`, as [`Self::forward`] places them: the text after a picture the cache holds already, in one
    /// chained run as any text is. Another model's: Err (it runs such text a chunk at a time).
    fn forward_chunks_at(&self, chunks: &[(&[u32], Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize], at: &[[u32; 3]]) -> Result<(Tensor, Vec<crate::llm::Tapped>), String> {
        let Self::Qwen35(Model::Qwen35(m)) = self else { return Err("not a dense Qwen hybrid".into()) };
        let (tokens, emb) = self.joined(chunks);
        let embeds = m.backend.to_device(emb);
        let (logits, kept) = m.forward_embeds_tapped_at(&embeds, tokens.len(), kv, taps, at).ok_or("the run at its positions could not be chained")?;
        for i in 0..chunks.len() {
            done(i);
        }
        Ok((logits.to_host(), kept))
    }

    fn forward_chunks_tapped(&self, chunks: &[(&[u32], Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize]) -> Result<(Tensor, Vec<crate::llm::Tapped>), String> {
        #[cfg(feature = "webgpu")]
        if let Self::Flash(f) = self {
            let len = kv.len;
            let refs: Vec<(&[u32], &Tensor)> = chunks.iter().map(|(t, e)| (*t, e)).collect();
            let (logits, kept) = f.forward_chunks_tapped(&refs, kv, done, taps);
            if let Some(l) = logits {
                return Ok((l, kept));
            }
            // (as [`Self::forward_chunks`]: what ran before a chunk it could not chain stands, with the states it
            // kept; the rest a chunk at a time, their checkpoints not kept)
            let ran: usize = chunks.iter().scan(len, |at, (t, _)| { *at += t.len(); Some(*at) }).take_while(|&at| at <= kv.len).count();
            let mut last = None;
            for (i, (t, e)) in chunks.iter().enumerate().skip(ran) {
                last = Some(self.forward(t, e.clone(), kv, None)?);
                done(i);
            }
            return last.map(|l| (l, kept)).ok_or_else(|| "no chunk to run".to_string());
        }
        let Self::Qwen35(Model::Qwen35(m)) = self else { return Err("not a dense Qwen hybrid".into()) };
        let (tokens, emb) = self.joined(chunks);
        let embeds = m.backend.to_device(emb);
        let (logits, kept) = m.forward_embeds_tapped(&embeds, tokens.len(), kv, taps).ok_or("the run could not keep its states from inside")?;
        for i in 0..chunks.len() {
            done(i);
        }
        Ok((logits.to_host(), kept))
    }
    fn forward(&self, tokens: &[u32], embeds: Tensor, kv: &mut KvCache, positions: Option<&[[u32; 3]]>) -> Result<Tensor, String> {
        self.forward_deepstack(tokens, embeds, kv, positions, &[])
    }

    /// [`Self::forward`] with `deepstack[i]` (`[tokens, width]`, zero where no picture is) added to the states after
    /// layer `i`: Qwen3-VL's DeepStack, its pictures' rows read from its tower's mergers. Only Qwen3-VL has it.
    fn forward_deepstack(&self, tokens: &[u32], embeds: Tensor, kv: &mut KvCache, positions: Option<&[[u32; 3]]>, deepstack: &[Vec<f32>]) -> Result<Tensor, String> {
        if !deepstack.is_empty() && !matches!(self, Self::Qwen35(Model::Qwen3(_))) {
            return Err("DeepStack tokens for a model without DeepStack".into());
        }
        match self {
            Self::Qwen35(Model::Qwen35(m)) => {
                let embeds = m.backend.to_device(embeds);
                Ok(m.forward_embeds_positions(&embeds, tokens.len(), kv, positions).map_err(|e| e.to_string())?.to_host())
            }
            Self::Qwen35(Model::Qwen3(m)) => Ok(m.forward_embeds_positions(tokens, &embeds, kv, positions, deepstack).map_err(|e| e.to_string())?.to_host()),
            Self::Qwen35(_) => Err("not a dense Qwen hybrid".into()),
            #[cfg(feature = "webgpu")]
            Self::Flash(f) => f.forward(tokens, &embeds, kv, positions).map_err(|e| e.to_string()),
            #[cfg(test)]
            Self::Fake(f) => f.forward(tokens, kv),
        }
    }
}

pub struct QwenEngine {
    model: Hybrid,
    projector: Option<MmProj>,
    kv: KvCache,
    covered: Vec<u64>,
    /// Pictures' soft tokens by their hash, and their DeepStack tokens (Qwen3-VL's; none for the other towers).
    vision_cache: std::collections::VecDeque<(u64, Tensor, Vec<Tensor>)>,
    log: bool,
    checkpoints: Vec<(Vec<u64>, RecurrentSnapshot, bool)>,
    pub disk: Option<crate::engines::disk::DiskCache<Arc<Snapshot>>>,
    /// Only enable when the disk namespace fingerprints the vision pipeline.
    pub image_disk_cache: bool,
    /// The incognito session whose prompt state is held (in memory only), if any.
    private_session: Option<String>,
    /// Conversations set aside in host RAM (Flash-Next only; see `park`). Lives and dies
    /// with the engine, so a model reload (the only way its weights or LoRA change) starts it
    /// empty, and `kv` is never rebuilt for another context length.
    parking: Parking,
    /// A run that repeats without end is turned from, and stopped where it goes on all the same
    /// (`--repetition-guard`, on unless said): [`crate::guards::repetition::CycleGuard`].
    cycles: bool,
    /// The request has begun its reply (its prompt is read): a failure from here on is not run again.
    replying: bool,
    /// Prompts whose whole states go to the disk between requests ([`QwenEngine::save_some`]), the newest last.
    unsaved: Vec<Vec<u64>>,
    /// The state being made for the disk, a layer at a time: its prompt, whether it is a system prompt's, and when
    /// it began.
    building: Option<(Vec<u64>, bool, WholeBuilder, std::time::Instant)>,
    /// A MiniCPM-V 4.7 model's picture layout (its pictures' positions are its canvas's).
    minicpmv: Option<MiniCpmV>,
    /// Its model's calls: Qwen's XML, or LFM2's Python list.
    calls: Calls,
}

/// How a model writes its tool calls: Qwen's `<tool_call>` XML, or LFM2's Python list between
/// `<|tool_call_start|>` and `<|tool_call_end|>` (`crate::engines::lfm2`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Calls {
    #[default]
    Qwen,
    Lfm2,
}

impl Calls {
    /// What opens a call (its first characters enough to hold back text at).
    fn open(self) -> (&'static str, &'static str) {
        match self {
            Calls::Qwen => ("<tool_call>", "<tool_call"),
            Calls::Lfm2 => (crate::engines::lfm2::CALL_START, "<|tool_call_start"),
        }
    }

    /// A complete reply's calls in the server's tool markup.
    fn normalize(self, text: &str, tools: &[Json]) -> Result<String, String> {
        match self {
            Calls::Qwen => normalize(text, tools),
            Calls::Lfm2 => crate::engines::lfm2::normalize(text, tools),
        }
    }
}

/// The times a reply is turned from a run without end before it is stopped: a model turned from one may write
/// another, and a reply that has written three is not going anywhere.
const TURNS: usize = 3;

impl QwenEngine {
    pub fn new(model: impl Into<Hybrid>, projector: Option<MmProj>, max_seq: usize, log: bool) -> Self {
        let model = model.into();
        let kv = model.new_kv_cache(max_seq);
        Self { model, projector, kv, covered: Vec::new(), vision_cache: Default::default(), log, checkpoints: Vec::new(), disk: None, image_disk_cache:false, private_session: None, parking: Parking::new(0, log), cycles: true, replying: false, unsaved: Vec::new(), building: None, minicpmv: None, calls: Calls::Qwen }
    }
    /// A MiniCPM-V 4.7 model: its pictures laid out on their canvases ([`prepare_minicpmv`], [`positions`]).
    pub fn minicpmv(mut self, layout: Option<MiniCpmV>) -> Self {
        self.minicpmv = layout;
        self
    }
    /// Its model's tool calls' format (Qwen's unless this says otherwise).
    pub fn calls(mut self, calls: Calls) -> Self {
        self.calls = calls;
        self
    }
    /// Whether a run that repeats without end is turned from and stopped (it is, unless this says not).
    pub fn repetition_guard(mut self, on: bool) -> Self {
        self.cycles = on;
        self
    }
    /// Let the engine set conversations aside in host RAM, up to `bytes` (0: not at all), when
    /// its model can: only Qwen3.8-Flash-Next, which has no disk states of its own.
    pub fn park_up_to(mut self, bytes: usize) -> Self {
        if self.model.parks() { self.parking.set_budget(bytes); }
        self
    }
    pub fn run(mut self, jobs: Receiver<Job>) {
        self.warm_up();
        loop {
            // (between requests the states for the disk are made, a layer at a time, a request taken between layers:
            // made after a reply, one held the next request back while its rows were copied, 2 s of an agent's step
            // at 26,000 tokens)
            let job = if self.unsaved.is_empty() && self.building.is_none() {
                match jobs.recv() { Ok(job) => job, Err(_) => break }
            } else {
                match jobs.try_recv() {
                    Ok(job) => job,
                    Err(std::sync::mpsc::TryRecvError::Empty) => { self.save_guarded(); continue; }
                    // (the engine let go: what it had to write, written first)
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => { while self.save_guarded() {} break; }
                }
            };
            // The end of an incognito session: wipe what is held of it.
            if job.wipe {
                if self.private_session.is_some() && self.private_session == job.session { self.forget_state(); }
                let _ = job.events.send(Event::Done { finish: Finish::Stop, completion_tokens: 0 });
                continue;
            }
            // A session's private state is reused by that session only: anything
            // else wipes it first.
            if self.private_session.is_some() && !(job.forget && job.session.is_some() && job.session == self.private_session) {
                self.forget_state();
            }
            self.replying = false;
            let mut result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.generate(&job)));
            // A GPU that had no room while the prompt was read (no word of the reply sent): what the model keeps on it
            // between requests let go, its cache's copy too, and the request run once more from the start. Left as it
            // was, every request after failed the same way until the engine was started again.
            if result.as_ref().is_err_and(|p| ggml_rs::chain::no_room(p.as_ref())) && !self.replying {
                if self.log { eprintln!("  Qwen: the GPU had no room for this request; what the model keeps there between requests let go, and the request run again"); }
                self.kv.reset(); self.covered.clear();
                self.model.let_go_of_spare(true);
                result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.generate(&job)));
            }
            // (a picture's prompt keeps vectors of its own on the GPU: let go, for the text prompts that come after)
            if !job.images.is_empty() || result.is_err() {
                self.model.let_go_of_spare(result.is_err());
            }
            let error = match result { Ok(Ok(())) => None, Ok(Err(e)) => Some(e), Err(_) => Some("native Qwen inference failed".into()) };
            if let Some(e) = error { self.kv.reset(); self.covered.clear(); let _ = job.events.send(Event::Error(e)); }
            if job.forget {
                // An incognito session keeps its state for its next request, in
                // memory only (nothing of it goes to disk); otherwise nothing of
                // this request is reused or kept.
                if job.session.is_some() { self.private_session = job.session.clone(); } else { self.forget_state(); }
            }
        }
    }

    /// The model run once before the first request, on tokens of no meaning, and what that left let go
    /// (OAIY_NO_WARMUP: not). A chained model's kernels are made as each first runs (a shader compiled, a pipeline
    /// built), so a server's first request paid for them: the 27B's first 15.6K-token prompt took 7.6 s where the
    /// next took 6.3, its first reply ran at 62.6 tokens a second where 67.5. llama.cpp's server runs its model once
    /// at load too.
    fn warm_up(&mut self) {
        if std::env::var_os("OAIY_NO_WARMUP").is_some() || !self.model.warms() {
            return;
        }
        let clock = std::time::Instant::now();
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.warm()));
        self.kv.reset();
        self.covered.clear();
        self.checkpoints.clear();
        if self.log {
            eprintln!("  Qwen warm-up: {:.2}s{}", clock.elapsed().as_secs_f64(), if matches!(ran, Ok(Ok(()))) { "" } else { " (not all of it ran)" });
        }
    }

    /// [`Self::warm_up`]'s run: two of a prompt's chunks as a prompt's go, a decode step (its token picked on the GPU
    /// and its logits read), and, a model that drafts, a round of drafts and a check of each size both ways, undone.
    fn warm(&mut self) -> Result<(), String> {
        let hybrid = &self.model;
        let rows = hybrid.prompt_rows();
        if self.kv.max_len < 2 * rows + 64 {
            return Ok(());
        }
        let token = |i: usize| 1000 + (i as u32 * 7919) % 20000;
        let prompt: Vec<u32> = (0..2 * rows).map(token).collect();
        let spans = [(0, rows), (rows, 2 * rows)];
        if hybrid.pipelines() {
            let chunks = spans.iter().map(|&(a, b)| Ok((&prompt[a..b], hybrid.embed(&prompt[a..b])?))).collect::<Result<Vec<(&[u32], Tensor)>, String>>()?;
            hybrid.forward_chunks(&chunks, &mut self.kv, &mut |_: usize| {})?;
        } else {
            for (a, b) in spans {
                let embeds = hybrid.embed(&prompt[a..b])?;
                hybrid.forward(&prompt[a..b], embeds, &mut self.kv, None)?;
            }
        }
        let next = token(2 * rows);
        let _ = hybrid.step_pick(next, &mut self.kv);
        let embeds = hybrid.embed(&[next])?;
        hybrid.forward(&[next], embeds, &mut self.kv, None)?;
        if hybrid.drafts() {
            let window = hybrid.draft_window().min(prompt.len());
            let recent: Vec<u32> = prompt[prompt.len() - window..].iter().copied().chain([next]).collect();
            let _ = hybrid.draft(&self.kv, &recent, drafts_most());
            for n in 2..=1 + drafts_most() {
                let checked: Vec<u32> = (0..n).map(|i| token(3 * rows + i)).collect();
                if hybrid.check_picks(&checked, &mut self.kv).is_some() {
                    hybrid.rollback(&mut self.kv, n, 1);
                }
                if hybrid.check(&checked, &mut self.kv).is_some() {
                    hybrid.rollback(&mut self.kv, n, 1);
                }
            }
        }
        Ok(())
    }

    /// Forget everything held of a private request or session. Whatever was set aside in RAM
    /// goes too: the engine holds nothing afterwards, as it never did (the states set aside are
    /// never a private session's, but an incognito request ends with an empty engine).
    fn forget_state(&mut self) {
        self.unsaved.clear();
        self.building = None;
        self.kv.reset();
        self.covered.clear();
        self.checkpoints.clear();
        self.vision_cache.clear();
        self.parking.clear();
        self.private_session = None;
    }
    fn generate(&mut self, job: &Job) -> Result<(), String> {
        if job.prompt.is_empty() { return Err("empty Qwen prompt".into()); }
        let (positions, mut next_position) = positions(job, self.minicpmv.as_ref())?;
        let mut keys: Vec<_> = job.prompt.iter().map(|&t| t as u64).collect();
        for image in &job.images { for (offset,k) in keys[image.start..image.start+image.prep.n_tokens()].iter_mut().enumerate() { *k = image.hash.rotate_left(17) ^ (offset as u64) ^ (1<<63); } }
        // (checkpoints a cancelled or failed request left on their device: to the host before anything is set aside)
        let device = self.model.qwen35().map(|m| &m.backend);
        for (_, snap, _) in &mut self.checkpoints {
            snap.settle(&self.kv, device);
        }
        let cache_clock = std::time::Instant::now();
        // Flash-Next: a conversation set aside in RAM that this prompt continues comes back
        // first (the live one is set aside in its place), and the cache logic below goes on from
        // it as from any live state. A big live state that this prompt shares little with is set
        // aside too, now, while all its checkpoints are there: the logic below drops the ones the
        // prompt does not share. An incognito request neither takes from the RAM nor gives to it.
        let swapped = !job.forget && self.parking.swap_in(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: self.private_session.is_some() }, &keys);
        if !job.forget {
            self.parking.park_displaced(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: self.private_session.is_some() }, &keys);
        }
        let hybrid = &self.model;
        // Checkpoints and disk states are the Qwen3.5 hybrid's (Flash-Next continues from memory only).
        let model = hybrid.qwen35();
        let stops = checkpoint_positions(&job.prompt, hybrid.tokenizer()?.token_id("<|im_start|>"));
        let to_disk = disk_positions(&stops, keys.len());
        let common = self.covered.iter().zip(&keys).take_while(|(a,b)| a==b).count();
        // An attention suffix overwritten by a different branch cannot support
        // a recurrent-only checkpoint, even if a later request matches its keys.
        self.checkpoints.retain(|(saved,_,_)| saved.len() <= common && keys.starts_with(saved));
        let mut start = if common == self.covered.len() && common < keys.len() { common } else { 0 };
        let mut source = if start > 0 { "memory" } else { "none" };
        if let Some((saved, snap, _)) = self.checkpoints.iter().filter(|(saved, _, _)|
            saved.len() > start && saved.len() < keys.len() && keys.starts_with(saved)
        ).max_by_key(|(saved, _, _)| saved.len()) {
            match model {
                Some(model) => snap.restore(&mut self.kv, common, &model.attention_layers, model.ssm_cfg, model.backend.as_ref())?,
                None => snap.restore_slots(&mut self.kv, common)?,
            }
            self.covered = saved.clone(); start = saved.len(); source = "checkpoint";
            self.checkpoints.retain(|(saved,_,_)| saved.len() <= start);
        }
        // Persist images only when the caller identifies both the projector and
        // preprocessing version. The prompt keys also include the image bytes.
        // The text before a prompt's first picture is text like any other (a
        // picture-reviewing step's long instructions): its state comes from
        // the disk whatever follows it.
        let text_until = text_until(job, self.image_disk_cache, keys.len());
        if let Some(model) = model.filter(|_| text_until > 0) {
            if let Some((saved, snap)) = self.disk.as_mut().and_then(|d| d.load_best(&keys, text_until.min(keys.len()-1), start, self.log)) {
                match snap.restore(&mut self.kv, &model.attention_layers, model.ssm_cfg, model.backend.as_ref()) {
                    Ok(()) => {
                        // A disk restore replaces the attention buffers. Even
                        // matching token keys may have different rounding from
                        // another batching history, so old recurrent-only states
                        // must not be mixed with this attention snapshot.
                        self.checkpoints.clear();
                        self.checkpoints.push((saved.clone(), RecurrentSnapshot::from_snapshot(&snap), stops.first() == Some(&snap.pos)));
                        self.covered = saved; start = snap.pos; source = "disk";
                    }
                    Err(e) => { if self.log { eprintln!("Qwen checkpoint skipped: {e}"); } }
                }
            }
        }
        // Qwen3.8-Flash-Next's, a whole state of its cache's slots (no images' yet: their positions are not keyed).
        if let Some(rows) = hybrid.row_slots().filter(|_| model.is_none() && job.images.is_empty()) {
            if let Some((saved, snap)) = self.disk.as_mut().and_then(|d| d.load_best(&keys, keys.len()-1, start, self.log)) {
                match snap.restore_whole(&mut self.kv, &rows) {
                    Ok(()) => {
                        // (as above: the attention's rows are the disk's now, so no older checkpoint goes with them)
                        self.checkpoints.clear();
                        self.checkpoints.push((saved.clone(), RecurrentSnapshot::from_snapshot(&snap), stops.first() == Some(&snap.pos)));
                        self.covered = saved; start = snap.pos; source = "disk";
                    }
                    Err(e) => { if self.log { eprintln!("Qwen checkpoint skipped: {e}"); } }
                }
            }
        }
        // What came back from RAM is not "memory" or a checkpoint of this process's own making.
        if swapped && start > 0 { source = "ram"; }
        if start == 0 { self.kv.reset(); self.covered.clear(); self.checkpoints.clear(); }
        let _ = job.events.send(Event::CacheReuse { cached: start, source, common });
        if self.log { eprintln!("  Qwen cache: {start}/{} tokens from {source} (common {common}) in {:.3}s", keys.len(), cache_clock.elapsed().as_secs_f64()); }
        let total = keys.len()-start;
        // (the prompt's length, for the devices' copies of the attention cache: room for it made once)
        self.kv.expect = keys.len();
        let _ = job.events.send(Event::Progress { done: 0, total });
        #[cfg(feature = "webgpu")]
        let cold_at_start = ggml_rs_wgpu::quant::moe::cached_experts();
        #[cfg(feature = "webgpu")]
        let tiered_at_start = ggml_rs_wgpu::exl3::tiered_experts();
        #[cfg(feature = "webgpu")]
        let scratch_at_start = ggml_rs_wgpu::scratch_made();
        let clock = std::time::Instant::now();
        let mut soft = Vec::new();
        for image in &job.images {
            if image.start + image.prep.n_tokens() <= start { continue; }
            if job.cancel.load(Ordering::Relaxed) { return Ok(()); }
            let (t, ds) = if let Some((_,t,ds)) = self.vision_cache.iter().find(|(hash,_,_)| *hash == image.hash) { (t.clone(), ds.clone()) } else {
                let (ImagePrep::Qwen { pixels, .. } | ImagePrep::MiniCpmV { pixels, .. }) = &image.prep else { return Err("wrong image format for Qwen".into()); };
                let (t, ds) = self.projector.as_ref().ok_or("Qwen vision projector is not configured")?.forward_deepstack(pixels).map_err(|e| e.to_string())?;
                let (t, ds): (Tensor, Vec<Tensor>) = (t.to_host(), ds.into_iter().map(|d| d.to_host()).collect());
                self.vision_cache.push_back((image.hash,t.clone(),ds.clone()));
                while self.vision_cache.len() > 8 { self.vision_cache.pop_front(); }
                (t, ds)
            };
            soft.push((image.start,t,ds));
        }
        let mut logits = None;
        let mut pos = start;
        // (OAIY_PREFILL_LOG: each run's rows and time, its embedding's and its checkpoint's)
        let said = std::env::var_os("OAIY_PREFILL_LOG").is_some();
        while pos < keys.len() {
            if job.cancel.load(Ordering::Relaxed) { return Ok(()); }
            let (from, began) = (pos, std::time::Instant::now());
            // a text prompt's chunks up to the next checkpoint together, where the model runs them so (some at a time,
            // between cancellations' looks)
            // (text after pictures the cache holds already goes that way too, at its positions: a dense hybrid's; a
            // picture still to be read, or another model's text after one, goes a chunk at a time below)
            let placed = !job.images.is_empty();
            let cached = job.images.iter().all(|image| image.start + image.prep.n_tokens() <= pos);
            if hybrid.pipelines() && (!placed || (cached && hybrid.qwen35().is_some())) {
                // the checkpoints this run would reach that are not kept yet: kept from inside it where the model can
                // (its states copied as the run goes: a prompt's last two, before the assistant's header and before
                // its last token, were a run of 6 rows and a run of 1 of their own), else a run's end each as before
                // (the disk's whole states are made from them once the reply is out: `Snapshot::from_checkpoint`)
                let far = (pos + 8 * PREFILL_CHUNK).min(keys.len());
                let due: Vec<usize> = stops.iter().copied().filter(|&s| s > pos && s <= far && !self.checkpoints.iter().any(|(saved, _, _)| saved == &keys[..s])).collect();
                let inside = !due.is_empty() && self.kv.len == pos && hybrid.can_tap(far - pos, &self.kv, &due.iter().map(|s| s - pos).collect::<Vec<_>>());
                let stop = if inside { far } else { stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()).min(keys.len()).min(pos + 8 * PREFILL_CHUNK) };
                let rows = hybrid.prompt_rows();
                let spans: Vec<(usize, usize)> = (pos..stop).step_by(rows).map(|a| (a, (a + rows).min(stop))).collect();
                // (a dense hybrid's chunks' embeddings on every core: its table's rows, 16 ms a run of 3,150 on one)
                let embeds = if hybrid.qwen35().is_some() {
                    use rayon::prelude::*;
                    spans.par_iter().map(|&(a, b)| hybrid.embed(&job.prompt[a..b])).collect::<Result<Vec<_>, _>>()?
                } else {
                    spans.iter().map(|&(a, b)| hybrid.embed(&job.prompt[a..b])).collect::<Result<Vec<_>, _>>()?
                };
                if said { eprintln!("  prefill: {} rows embedded in {:.1} ms", stop - pos, began.elapsed().as_secs_f64() * 1e3); }
                let chunks: Vec<(&[u32], Tensor)> = spans.iter().zip(embeds).map(|(&(a, b), e)| (&job.prompt[a..b], e)).collect();
                let events = &job.events;
                let mut done = |i: usize| { let _ = events.send(Event::Progress { done: spans[i].1 - start, total }); };
                let taps: Vec<usize> = if inside { due.iter().map(|s| s - pos).collect() } else { Vec::new() };
                if placed {
                    let (out, kept) = hybrid.forward_chunks_at(&chunks, &mut self.kv, &mut done, &taps, &positions[pos..stop])?;
                    logits = Some(out);
                    for tap in kept {
                        if self.log { eprintln!("  Qwen checkpoint: {} tokens; from inside the run (after a picture)", tap.at); }
                        let base = stops.first() == Some(&tap.at);
                        self.checkpoints.push((keys[..tap.at].to_vec(), RecurrentSnapshot::tapped(tap.at, tap.states, tap.convs), base));
                        trim_checkpoints(&mut self.checkpoints);
                    }
                } else if inside {
                    let (out, kept) = hybrid.forward_chunks_tapped(&chunks, &mut self.kv, &mut done, &taps)?;
                    logits = Some(out);
                    for tap in kept {
                        if self.log { eprintln!("  Qwen checkpoint: {} tokens; from inside the run", tap.at); }
                        let base = stops.first() == Some(&tap.at);
                        self.checkpoints.push((keys[..tap.at].to_vec(), RecurrentSnapshot::tapped(tap.at, tap.states, tap.convs), base));
                        trim_checkpoints(&mut self.checkpoints);
                    }
                } else {
                    logits = Some(hybrid.forward_chunks(&chunks, &mut self.kv, &mut done)?);
                }
                self.covered.extend_from_slice(&keys[pos..stop]);
                pos = stop;
            } else {
                // (the model's own chunk where it has one: Flash-Next's on one card, which runs no pipeline)
                let end = (pos + hybrid.prompt_rows()).min(keys.len()).min(stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()));
                let mut embeds = hybrid.embed(&job.prompt[pos..end])?;
                let width = hybrid.width();
                // (Qwen3-VL's DeepStack: each merger's tokens at the pictures' rows of the chunk, zero elsewhere)
                let layers = soft.iter().filter(|(at, t, _)| *at < end && at + t.dim(0) > pos).map(|(_, _, ds)| ds.len()).max().unwrap_or(0);
                let mut deepstack = vec![vec![0f32; (end - pos) * width]; layers];
                for (at,t,ds) in &soft {
                    let from = pos.max(*at); let to = end.min(*at+t.dim(0));
                    if from < to {
                        embeds.data_mut()[(from-pos)*width..(to-pos)*width].copy_from_slice(&t.data()[(from-at)*width..(to-at)*width]);
                        for (rows, d) in deepstack.iter_mut().zip(ds) {
                            rows[(from-pos)*width..(to-pos)*width].copy_from_slice(&d.data()[(from-at)*width..(to-at)*width]);
                        }
                    }
                }
                let out = hybrid.forward_deepstack(&job.prompt[pos..end], embeds, &mut self.kv, (!job.images.is_empty()).then_some(&positions[pos..end]), &deepstack)?;
                logits = Some(out);
                self.covered.extend_from_slice(&keys[pos..end]);
                let _ = job.events.send(Event::Progress { done: end-start, total });
                pos = end;
            }
            if said { eprintln!("  prefill: rows {from}..{pos} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3); }
            let began = std::time::Instant::now();
            if stops.contains(&pos) && !self.checkpoints.iter().any(|(saved, _, _)| saved == &keys[..pos]) {
                if self.log { eprintln!("  Qwen checkpoint: {pos} tokens; disk={}", self.disk.is_some()); }
                let base = stops.first() == Some(&pos);
                // (the disk's state of it is made once the reply is out, from this and the attention cache)
                let snap = RecurrentSnapshot::capture_later(&self.kv, model.map(|m| &m.backend));
                self.checkpoints.push((keys[..pos].to_vec(), snap, base));
                // Keep the system prefix plus recent conversation boundaries.
                // Host storage avoids competing with media for VRAM.
                trim_checkpoints(&mut self.checkpoints);
                if said { eprintln!("  prefill: the checkpoint at {pos} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3); }
            }
        }
        let prefill_secs = clock.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });
        let decode_clock = std::time::Instant::now();
        let tok = hybrid.tokenizer()?;
        let eos = tok.token_id("<|im_end|>").ok_or("Qwen tokenizer lacks im_end")?;
        let think_end = tok.token_id("</think>");
        let mut thinking = job.think_budget.is_some();
        let mut generated = Vec::new(); let mut think_used = 0usize;
        let mut stream = NativeStream { calls: self.calls, ..Default::default() };
        let mut rng = job.sampling.seed ^ 0x9E3779B97F4A7C15;
        // (OAIY_CHAIN_PROFILE: the prompt's kernels said here, the reply's alone after)
        #[cfg(feature = "webgpu")]
        let profiled = self.log && std::env::var_os("OAIY_CHAIN_PROFILE").is_some();
        #[cfg(feature = "webgpu")]
        if profiled {
            let kernels = ggml_rs_wgpu::profile::take_kernels();
            eprintln!("  Qwen kernels: the prompt's {:.1} ms of the GPU's in {} dispatches (each timed in a pass of its own)", kernels.iter().map(|k| k.1).sum::<f64>(), kernels.iter().map(|k| k.2 as u64).sum::<u64>());
            for (name, ms, count) in kernels.iter().take(24) {
                eprintln!("    {name:<34} {ms:>9.2} ms {count:>7}");
            }
        }
        // (a card holding only some of a layer's experts: what it read from the host's memory and brought in, for
        // the request's line)
        #[cfg(feature = "webgpu")]
        let (cold_before, cold_prompt) = (cold_at_start, ggml_rs_wgpu::quant::moe::cached_experts());
        #[cfg(feature = "webgpu")]
        let (tiered_before, tiered_prompt) = (tiered_at_start, ggml_rs_wgpu::exl3::tiered_experts());
        #[cfg(feature = "webgpu")]
        let scratch_prompt = ggml_rs_wgpu::scratch_made();
        let mut logits = Row::Logits(logits.unwrap());
        // (a greedy request's rows: the token alone where the model picks it on its GPU, megabytes of logits a check
        // not read back)
        let greedy = job.sampling.temperature <= 0.0 && job.images.is_empty() && !top_log();
        let mut finish = Finish::Length;
        // Where decoding's time goes, for the log: sampling, the text (decode and stream), the model.
        let (mut t_sample, mut t_text, mut t_model) = (0f64, 0f64, 0f64);
        // Drafting (a model with its multi-token-prediction layer, chained: a Qwen3.5 GGUF's, Flash-Next's; text only):
        // each token sampled is run with the drafts after it in one check, and while the sampler picks what was drafted
        // its logits are the check's (sampling the model's distribution and taking a draft only where it is the token
        // sampled is that distribution's sampling still); where it picks another, the check's rows from there are undone.
        let drafter = (job.images.is_empty() && hybrid.drafts()).then_some(hybrid);
        let mut pending: std::collections::VecDeque<(u32, Row)> = Default::default();
        let (mut check_rows, mut drafts_checked, mut drafts_taken) = (0usize, 0usize, 0usize);
        // (for the log: the checks made and their rows, the time making drafts, in checks, and undoing rows)
        let (mut checks, mut rows_checked, mut t_draft, mut t_check, mut t_undo) = (0usize, 0usize, 0f64, 0f64, 0f64);
        // (OAIY_DECODE_LOG: the model's time at every fiftieth token)
        let mut by_fifty: Vec<f64> = Vec::new();
        let (mut cycles, mut turned) = (crate::guards::repetition::CycleGuard::default(), 0usize);
        self.replying = true;
        while generated.len() < job.max_tokens && self.kv.len < self.kv.max_len {
            if job.cancel.load(Ordering::Relaxed) {
                if let (Some(m), false) = (drafter, pending.is_empty()) { m.rollback(&mut self.kv, check_rows, check_rows - pending.len()); }
                return Ok(());
            }
            let clock = std::time::Instant::now();
            let mut next = logits.sample(&job.sampling, &mut rng);
            // A run that repeats without end (a list's same item again and again): the model picks another token
            // than the run's next, at the run's comma or line end where it has one (a list's place to close) and
            // anywhere once a whole turn has passed with none. Each time halves what the next run is let be. A reply
            // turned so `TURNS` times is stopped. A greedy request's rows near a run's limit are read back whole
            // (`CycleGuard::near`, below), so it is turned as another is; one whose row is still the model's pick
            // alone (no other token to take) is stopped.
            if let Some(cycle) = cycles.over(next, turned).filter(|_| self.cycles) {
                if cycle.past >= cycle.period || tok.decode(&[next]).contains([',', ';', '\n']) {
                    let other = if turned < TURNS { logits.sample_but(next, &job.sampling, &mut rng) } else { None };
                    let Some(other) = other else {
                        if let (Some(m), false) = (drafter, pending.is_empty()) { m.rollback(&mut self.kv, check_rows, check_rows - pending.len()); }
                        return Err(format!("{}: the reply went on repeating the same {} tokens ({} tokens of them) and was stopped", crate::guards::repetition::CODE, cycle.period, cycle.span));
                    };
                    turned += 1;
                    if self.log { eprintln!("  Qwen: the reply repeated the same {} tokens for {} tokens: turned from it ({turned} of {TURNS})", cycle.period, cycle.span); }
                    next = other;
                }
            }
            t_sample += clock.elapsed().as_secs_f64();
            if thinking && job.think_budget.is_some_and(|n|think_used >= n) { if let Some(end) = think_end { next = end; } }
            let stop = next == eos || Some(next) == tok.eos();
            // the check ran this token already where it is the next draft; else its rows from here on are undone
            let ran = !stop && pending.front().is_some_and(|(d, _)| *d == next);
            if !ran && !pending.is_empty() {
                let m = drafter.expect("drafts are a drafter's");
                let clock = std::time::Instant::now();
                m.rollback(&mut self.kv, check_rows, check_rows - pending.len());
                t_undo += clock.elapsed().as_secs_f64();
                pending.clear();
            }
            if stop { finish = Finish::Stop; break; }
            generated.push(next);
            cycles.push(next);
            let clock = std::time::Instant::now();
            let decoded = tok.decode(&generated);
            let delta = stream.push(&decoded, &job.tools, false)?;
            if !delta.is_empty() { let _ = job.events.send(Event::Text(delta)); }
            if let Some((text,start)) = stream.tool_preview(&decoded)? {
                let _ = job.events.send(Event::ToolPreview { text, start });
            }
            if thinking {
                think_used += 1;
                if Some(next)==think_end {
                    thinking=false;
                    let _ = job.events.send(Event::Thinking { used: think_used, budget: job.think_budget, done: true });
                }
            }
            if thinking && generated.len() % 16 == 0 { let _ = job.events.send(Event::Thinking { used: think_used, budget: job.think_budget, done: false }); }
            t_text += clock.elapsed().as_secs_f64();
            let clock = std::time::Instant::now();
            if ran {
                let (_, after) = pending.pop_front().expect("the draft just matched");
                logits = after;
                drafts_taken += 1;
            } else {
                // (a greedy request's next rows read back whole where a run nears its limit, for the guard to have
                // another token to take than the run's: the model's pick alone has none)
                let pick = greedy && !(self.cycles && cycles.near(turned, 16));
                // its drafts, then the token and them in one check; else the token alone
                let checked = drafter.filter(|_| self.kv.len + 1 + drafts_most() <= self.kv.max_len).and_then(|m| {
                    let recent: Vec<u32> = self.covered[self.covered.len().saturating_sub(m.draft_window())..].iter().map(|&k| k as u32).chain([next]).collect();
                    // none the layer is sure enough of: a step of the token alone
                    let began = std::time::Instant::now();
                    let drafts = m.draft(&self.kv, &recent, drafts_most()).filter(|d| !d.is_empty());
                    t_draft += began.elapsed().as_secs_f64();
                    let drafts = drafts?;
                    let rows: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
                    let began = std::time::Instant::now();
                    let picked = if pick { m.check_picks(&rows, &mut self.kv) } else { None };
                    let checked: Option<Vec<Row>> = match picked {
                        Some(tokens) => Some(tokens.into_iter().map(Row::Pick).collect()),
                        None => m.check(&rows, &mut self.kv).map(|l| l.into_iter().map(Row::Logits).collect()),
                    };
                    t_check += began.elapsed().as_secs_f64();
                    Some((drafts, checked?))
                });
                match checked {
                    Some((drafts, mut rows)) => {
                        check_rows = rows.len();
                        checks += 1;
                        rows_checked += check_rows;
                        drafts_checked += drafts.len();
                        logits = rows.remove(0);
                        pending = drafts.into_iter().zip(rows).collect();
                    }
                    None => {
                        logits = match if pick { hybrid.step_pick(next, &mut self.kv) } else { None } {
                            Some(token) => Row::Pick(token),
                            None => {
                                let embeds = hybrid.embed(&[next])?;
                                Row::Logits(hybrid.forward(&[next], embeds, &mut self.kv, (!job.images.is_empty()).then_some(&[[next_position;3]]))?)
                            }
                        };
                    }
                }
            }
            t_model += clock.elapsed().as_secs_f64();
            if generated.len() % 50 == 0 {
                by_fifty.push(t_model);
            }
            next_position += 1;
            self.covered.push(next as u64);
        }
        // a check's rows no token reached
        if let (Some(m), false) = (drafter, pending.is_empty()) {
            m.rollback(&mut self.kv, check_rows, check_rows - pending.len());
        }
        #[cfg(feature = "webgpu")]
        if profiled {
            let kernels = ggml_rs_wgpu::profile::take_kernels();
            eprintln!("  Qwen kernels: the reply's {:.1} ms of the GPU's in {} dispatches (each timed in a pass of its own)", kernels.iter().map(|k| k.1).sum::<f64>(), kernels.iter().map(|k| k.2 as u64).sum::<u64>());
            for (name, ms, count) in kernels.iter().take(48) {
                eprintln!("    {name:<34} {ms:>9.2} ms {count:>7}");
            }
        }
        if self.log && std::env::var_os("OAIY_DECODE_LOG").is_some() {
            let each: Vec<String> = by_fifty.iter().scan(0f64, |before, &t| { let d = t - *before; *before = t; Some(format!("{d:.3}")) }).collect();
            eprintln!("  Qwen decode: the model's seconds by fifties of tokens: {}", each.join(" "));
        }
        let raw = tok.decode(&generated);
        // The client is told what the model wrote, so a rejected call can be read
        // and fixed; the server log never holds it.
        let text = stream.push(&raw, &job.tools, true)
            .map_err(|e| format!("tool_contract_error: {e}; no tool from this batch was executed. The model wrote: {}", call_excerpt(&raw)))?;
        if self.log { eprintln!("  Qwen: {} prompt tokens ({} cached) in {:.2}s; {} generated in {:.2}s (sampling {t_sample:.2}s, text {t_text:.2}s, model {t_model:.2}s){}", job.prompt.len(),start,prefill_secs,generated.len(),decode_clock.elapsed().as_secs_f64(), if drafts_checked > 0 { format!("; {drafts_taken} of {drafts_checked} drafts taken ({checks} checks of {rows_checked} rows {t_check:.2}s, drafting {t_draft:.2}s, undoing {t_undo:.2}s)") } else { String::new() }); }
        #[cfg(feature = "webgpu")]
        if self.log {
            let now = ggml_rs_wgpu::quant::moe::cached_experts();
            if now.0 > cold_before.0 {
                eprintln!("  Qwen experts: the prompt read {} from the host's memory and brought {} to the card, the reply {} and {}", cold_prompt.0 - cold_before.0, cold_prompt.1 - cold_before.1, now.0 - cold_prompt.0, now.1 - cold_prompt.1);
            }
            // (experts read as records, a GPU holding some: how many of the routings it held, what it took in, what was read)
            let t = ggml_rs_wgpu::exl3::tiered_experts();
            if t[0] + t[1] > tiered_before[0] + tiered_before[1] {
                let part = |a: [u64; 10], b: [u64; 10]| {
                    let (held, not) = (b[0] - a[0], b[1] - a[1]);
                    let s = |i: usize| (b[i] - a[i]) as f64 / 1e6;
                    format!("{:.1}% of {} routings held on the GPU, {} taken in, {} records read ({:.2} GB; {} read ahead used, {} not); reading {:.2} s, running {:.2} s, taking in {:.2} s", 100.0 * held as f64 / (held + not).max(1) as f64, held + not, b[2] - a[2], b[3] - a[3], (b[4] - a[4]) as f64 / 1e9, b[8] - a[8], b[9] - a[9], s(5), s(6), s(7))
                };
                eprintln!("  Qwen experts: the prompt {}; the reply {}", part(tiered_before, tiered_prompt), part(tiered_prompt, t));
            }
            // (scratch its recordings found none of in the pool: new memory, which the system zeroes as it gives it)
            let made = ggml_rs_wgpu::scratch_made();
            if made > scratch_at_start {
                eprintln!("  Qwen scratch: {} MiB made new for the prompt, {} for the reply", (scratch_prompt - scratch_at_start) >> 20, (made - scratch_prompt) >> 20);
            }
        }
        if !text.is_empty() { let _ = job.events.send(Event::Text(text)); }
        let _ = job.events.send(Event::Done { finish, completion_tokens: generated.len() });
        // the reply is out: this prompt's checkpoints' states, copied on their devices as it ran, to the host now
        let device = self.model.qwen35().map(|m| &m.backend);
        for (_, snap, _) in &mut self.checkpoints {
            snap.settle(&self.kv, device);
        }
        // and the disk's whole states of them, once the engine is idle (disk states are the Qwen3.5 hybrid's and
        // Flash-Next's, of text; memory checkpoints any model's): `save_unsaved`
        // (of a prompt with pictures, without the vision pipeline's own namespace: what is before the first, text,
        // as `text_until` was above)
        let stored = self.model.qwen35().is_some() || (self.model.row_slots().is_some() && job.images.is_empty());
        if let Some(disk) = self.disk.as_ref().filter(|_| stored && !job.forget) {
            for &at in to_disk.iter().filter(|&&at| at <= text_until) {
                let saved = keys[..at].to_vec();
                if !disk.has(&saved) && !self.unsaved.contains(&saved) {
                    self.unsaved.push(saved);
                }
            }
            // (the newest few: the conversation's latest, and its system prompt's)
            let extra = self.unsaved.len().saturating_sub(4);
            self.unsaved.drain(..extra);
        }
        Ok(())
    }

    /// [`Self::save_some`] as a request's run is, its panic caught: a step of a disk state runs between requests, in
    /// the model's own thread, and a panic there ended that thread (nothing starts it again), and every request
    /// after failed. A step that panics gives up the states still to make, and says so.
    fn save_guarded(&mut self) -> bool {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.save_some())) {
            Ok(more) => more,
            Err(_) => {
                eprintln!("  Qwen: making a prompt's state for the disk failed; the states still to make are not made");
                self.building = None;
                self.unsaved.clear();
                false
            }
        }
    }

    /// A step of the disk's states ([`QwenEngine::unsaved`]): the next attention layer of the one being made, or
    /// the newest still to make begun (where it is still what the cache holds: its rows up to there unchanged, and
    /// its checkpoint kept). Whether there is more to do.
    fn save_some(&mut self) -> bool {
        let Some(disk) = self.disk.as_mut() else { self.unsaved.clear(); self.building = None; return false };
        if let Some((saved, base, builder, began)) = &mut self.building {
            // (a request between steps that went on from it leaves its rows as they were; one that went back does not)
            if !self.covered.starts_with(saved) || builder.pos() > self.kv.len {
                self.building = None;
            } else if let Some(whole) = builder.step(&self.kv) {
                if self.log { eprintln!("  Qwen: the prompt's state at {} tokens made for the disk between requests in {:.2}s", saved.len(), began.elapsed().as_secs_f64()); }
                disk.save(std::mem::take(saved), Arc::new(whole), *base);
                self.building = None;
            }
            return self.building.is_some() || !self.unsaved.is_empty();
        }
        while let Some(saved) = self.unsaved.pop() {
            let at = saved.len();
            if disk.has(&saved) || !self.covered.starts_with(&saved) || at > self.kv.len {
                continue;
            }
            let Some((_, snap, base)) = self.checkpoints.iter().find(|(kept, _, _)| *kept == saved) else { continue };
            // (Qwen3.8-Flash-Next's: its cache's slots, the keys and values of those that hold them up to there)
            let (rows, backend) = match (self.model.qwen35(), self.model.row_slots()) {
                (Some(model), _) => (model.attention_layers.clone(), Arc::clone(&model.backend)),
                (None, Some(rows)) => match self.kv.layer_backends.first().cloned() {
                    Some(backend) => (rows, backend),
                    None => continue,
                },
                (None, None) => continue,
            };
            let bytes = Snapshot::whole_bytes(at, &self.kv, &rows);
            if (!*base && disk.near(&saved, NEAR)) || too_large(disk, at, bytes, self.log) || !disk.worth(&saved, bytes, *base, &self.covered) {
                continue;
            }
            if let Some(builder) = Snapshot::builder(snap, &self.kv, &rows, backend) {
                self.building = Some((saved, *base, builder, std::time::Instant::now()));
                break;
            }
        }
        self.building.is_some() || !self.unsaved.is_empty()
    }
}

/// A conversation's state goes to the disk again once it is this many tokens past the one there: a restart reads
/// at most these again (some 4 s of the 27B's on the eGPU), where a state at every step of an agent's was 3.5 GB
/// written every few seconds at 27,000 tokens.
const NEAR: usize = 4096;

/// Tokens drafted a check, where a model drafts (its multi-token-prediction layer): Qwen3.8 27B's take 0.86, 0.73
/// and 0.65 in turn (each where the ones before it were).
/// What a row of the model gave the decode loop: its logits, or (a greedy request's, where the model picks on its
/// GPU) its largest logit's token alone.
enum Row {
    Logits(Tensor),
    Pick(u32),
}

impl Row {
    /// The token sampled from the row.
    fn sample(&self, s: &crate::engines::job::Sampling, rng: &mut u64) -> u32 {
        match self {
            Row::Logits(l) => {
                if top_log() {
                    let mut top: Vec<(usize, f32)> = l.data().iter().copied().enumerate().collect();
                    top.select_nth_unstable_by(5, |a, b| b.1.total_cmp(&a.1));
                    top.truncate(5);
                    top.sort_by(|a, b| b.1.total_cmp(&a.1));
                    eprintln!("  top: {}", top.iter().map(|(t, v)| format!("{t}:{v:.3}")).collect::<Vec<_>>().join(" "));
                }
                sample(l.data(), s, rng)
            }
            Row::Pick(token) => *token,
        }
    }

    /// A token sampled from the row with `banned` out of it; None where the row is the model's pick alone.
    fn sample_but(&self, banned: u32, s: &crate::engines::job::Sampling, rng: &mut u64) -> Option<u32> {
        match self {
            Row::Logits(l) => {
                let mut logits = l.data().to_vec();
                if let Some(x) = logits.get_mut(banned as usize) {
                    *x = f32::NEG_INFINITY;
                }
                Some(sample(&logits, s, rng))
            }
            Row::Pick(_) => None,
        }
    }
}

/// OAIY_TOP_LOG: each row's five largest logits and their tokens said as it is sampled (a greedy request's rows read
/// back for it, where the model would pick on its GPU): two runs of one prompt held against each other, by their
/// numbers and not by their text alone.
fn top_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_TOP_LOG").is_some())
}

const DRAFTS: usize = 3;

/// [`DRAFTS`], or OAIY_DRAFTS's (1 to 7: a check is 8 rows at most). More than three loses on Flash-Next, whose check
/// is some 1.9 steps (its 66 tokens after 3,780: 0.84 s with three, 42 of 51 drafts taken; 0.96 with four, 45 of 59;
/// 0.94 with five; 0.99 with six).
fn drafts_most() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("OAIY_DRAFTS").ok().and_then(|v| v.parse().ok()).filter(|n| (1..=7).contains(n)).unwrap_or(DRAFTS))
}

/// Checkpoints kept in memory: the base and the last two requests' two (each request's before the assistant's header
/// and a token short of its prompt). Three kept only the last request's, and an agent's next prompt that renders the
/// reply before a tool's answer otherwise (a call's, twice in one call) went back to the base: 768 and 1,010 tokens
/// read again on the eGPU's 27B, 0.8 and 1.0 s, where they were 30 or so. Its recurrent states are some 150 MB each.
const KEPT_CHECKPOINTS: usize = 5;

/// Whether a prompt's state at `at` tokens, `bytes` of keys and values, is more than the disk's cache is allowed in all
/// (said, with `log`): it is then not copied out of the cache and written, where a long conversation's was after
/// every reply (the 27B's at 80,000 tokens 10 GB, on a computer with 24). The memory's checkpoints still serve the
/// conversation while the engine runs; a larger `--prompt-cache-gb` keeps such states too.
fn too_large<S: crate::engines::disk::PromptState>(disk: &crate::engines::disk::DiskCache<S>, at: usize, bytes: u64, log: bool) -> bool {
    let large = !disk.takes(bytes);
    if large && log {
        eprintln!("  Qwen: the prompt's state at {at} tokens ({:.1} GB) is more than the prompt cache's whole allowance: not kept on the disk", bytes as f64 / 1e9);
    }
    large
}

fn trim_checkpoints(checkpoints: &mut Vec<(Vec<u64>, RecurrentSnapshot, bool)>) {
    while checkpoints.len() > 1 && (checkpoints.len() > KEPT_CHECKPOINTS ||
        checkpoints.iter().map(|(_, s, _)| s.bytes()).sum::<usize>() > 1024 * 1024 * 1024) {
        let remove = (0..checkpoints.len()-1).min_by_key(|&i| (checkpoints[i].2, i)).unwrap();
        checkpoints.remove(remove);
    }
}

/// Of a prompt's checkpoints (`stops`), those its disk keeps: all but the one a token short of the whole prompt where
/// another comes a few tokens before it (the assistant's header: an identical retry from the disk runs those few tokens
/// more, and a disk state is a whole one, the attention cache's keys and values too).
fn disk_positions(stops: &[usize], len: usize) -> Vec<usize> {
    let last = len.saturating_sub(1);
    let near = stops.iter().any(|&s| s < last && last - s <= 16);
    stops.iter().copied().filter(|&s| !(s == last && near)).collect()
}

/// Save before the first user message, before the current assistant header
/// (thinking/tool rendering changes its suffix), and one token short of the
/// entire prompt so an identical retry still runs a token to recover logits.
fn checkpoint_positions(prompt: &[u32], im_start: Option<u32>) -> Vec<usize> {
    let boundaries: Vec<_> = prompt.iter().enumerate().filter_map(|(i, &t)|
        (i > 0 && Some(t) == im_start).then_some(i)).collect();
    let mut stops = vec![prompt.len().saturating_sub(1)];
    stops.extend(boundaries.first().copied());
    stops.extend(boundaries.last().copied());
    stops.retain(|&p| p > 0 && p < prompt.len());
    stops.sort_unstable(); stops.dedup(); stops
}

/// Each prompt token's M-RoPE position and the one the reply starts at: text one by one, a Qwen picture's tokens over
/// its grid from where it starts (the text after it past its side); a MiniCPM-V 4.7 prompt's laid out on its pictures'
/// canvases (`minicpmv::canvas_positions`, as transformers' `get_rope_index`).
fn positions(job: &Job, minicpmv: Option<&MiniCpmV>) -> Result<(Vec<[u32;3]>,u32),String> {
    if job.images.iter().any(|i| matches!(i.prep, ImagePrep::MiniCpmV { .. })) {
        let m = minicpmv.ok_or("MiniCPM-V pictures for a model without their layout")?;
        let (mut mask, mut sizes) = (vec![false; job.prompt.len()], Vec::new());
        for image in &job.images {
            let ImagePrep::MiniCpmV { size, merge, .. } = image.prep else { return Err("wrong image format".into()); };
            if merge != m.merge { return Err("a picture cut for another mode".into()); }
            let end = image.start + image.prep.n_tokens();
            if end > mask.len() { return Err("an image past its prompt".into()); }
            mask[image.start..end].iter_mut().for_each(|t| *t = true);
            sizes.push(size);
        }
        let (canvas, next) = crate::llm::vision::mmproj::minicpmv::canvas_positions(&job.prompt, &mask, &sizes, m.merge, m.markers);
        let canvas = canvas.into_iter().map(|p| [p[0] as u32, p[1] as u32, p[2] as u32]).collect();
        return Ok((canvas, next as u32));
    }
    let mut positions = Vec::with_capacity(job.prompt.len()); let mut at=0usize; let mut next=0u32;
    for image in &job.images {
        if image.start < at { return Err("overlapping image spans".into()); }
        while at < image.start { positions.push([next;3]); at+=1; next+=1; }
        let ImagePrep::Qwen { side, .. } = image.prep else { return Err("wrong image format".into()); };
        for y in 0..side { for x in 0..side { positions.push([next,next+y as u32,next+x as u32]); } }
        at += side*side; next += side as u32;
    }
    while at < job.prompt.len() { positions.push([next;3]); at+=1; next+=1; }
    Ok((positions,next))
}

#[cfg(test)]
mod tests;
