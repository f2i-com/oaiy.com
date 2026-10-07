//! Native Qwen dense hybrid inference, image embeddings and XML tool protocol.
use std::sync::{mpsc::Receiver, atomic::Ordering};
use dsv41::chat::{Encoded, Mode, Options, DSML};
use ggml_rs::Tensor;
use std::sync::Arc;
use crate::qwen_cache::{Snapshot, RecurrentSnapshot};
use crate::qwen_park::{Held, Parking};
use llama_rs::{Model, MmProj, MmProjConfig, KvCache};
use oaiy_engine::json::Json;
use crate::engine::{Event, Finish, ImagePrep, Job, JobImage, sample};

const IMAGE: &str = "<|vision_start|><|image_pad|><|vision_end|>";
// Amortize EXL3 weight reconstruction across more image/prompt rows. Bounded
// to keep activations small alongside the configured long-context KV cache.
pub(crate) const PREFILL_CHUNK: usize = 512;

fn content(value: Option<&Json>, images: &mut Vec<Json>) -> Result<String, String> {
    match value {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => {
            if s.contains("<|image_pad|>") { return Err("send images as image_url blocks, not literal placeholders".into()); }
            Ok(s.clone())
        }
        Some(Json::Arr(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(Json::as_str) {
                    Some("text" | "input_text") => out.push_str(&content(p.get("text"), images)?),
                    Some("image_url" | "input_image" | "image") => {
                        let rec = if let Some(url) = p.get("image_url") {
                            Json::obj([("url", url.get("url").unwrap_or(url).clone())])
                        } else { p.clone() };
                        images.push(rec);
                        out.push_str(IMAGE);
                    }
                    _ => return Err("unsupported Qwen content block".into()),
                }
            }
            Ok(out)
        }
        _ => Err("message content must be text or blocks".into()),
    }
}

pub fn chat_prompt(msgs: &[Json], opts: &Options) -> Result<Encoded, String> {
    let mut prompt = String::new();
    let mut images = Vec::new();
    for (i,m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Json::as_str).unwrap_or("user");
        if !matches!(role, "system" | "user" | "assistant" | "tool") { return Err("unsupported Qwen message role".into()); }
        let before = images.len();
        let text = content(m.get("content_blocks").or_else(|| m.get("content")), &mut images)?;
        if role == "system" && images.len() != before { return Err("Qwen system messages cannot contain images".into()); }
        if role == "tool" {
            if i == 0 || msgs[i-1].get("role").and_then(Json::as_str) != Some("tool") { prompt.push_str("<|im_start|>user"); }
            prompt.push_str(&format!("\n<tool_response>\n{text}\n</tool_response>"));
            if i+1 == msgs.len() || msgs[i+1].get("role").and_then(Json::as_str) != Some("tool") { prompt.push_str("<|im_end|>\n"); }
            continue;
        }
        prompt.push_str(&format!("<|im_start|>{role}\n"));
        if role == "system" {
            if let Some(tools) = m.get("tools").and_then(Json::as_array).filter(|t| !t.is_empty()) {
                prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
                for t in tools { prompt.push_str(&t.to_json()); prompt.push('\n'); }
                prompt.push_str("</tools>\n\nCall functions with <tool_call>\n<function=FUNCTION_NAME>\n<parameter=PARAMETER_NAME>\nvalue\n</parameter>\n</function>\n</tool_call>. Supply every required parameter. Objects, arrays, numbers and booleans must be valid JSON; strings are literal text. No suffix after a function call.\n\n");
            }
            if opts.mode == Mode::Thinking && opts.effort <= 33 { prompt.push_str("Keep your thinking brief and focused.\n\n"); }
        }
        if role == "assistant" { prompt.push_str("<think>\n\n</think>\n\n"); }
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
struct NativeStream { sent: String, tool_sent: String }

impl NativeStream {
    fn tool_preview(&mut self, decoded: &str) -> Result<Option<(String, bool)>, String> {
        let Some(at) = decoded.find("<tool_call>") else { return Ok(None) };
        let draft = decoded[at..].trim_end_matches('\u{fffd}');
        let delta = draft.strip_prefix(&self.tool_sent).ok_or("Qwen tool preview changed after streaming")?;
        if delta.is_empty() { return Ok(None); }
        let update = (delta.to_owned(), self.tool_sent.is_empty());
        self.tool_sent = draft.to_owned();
        Ok(Some(update))
    }

    fn push(&mut self, decoded: &str, tools: &[Json], complete: bool) -> Result<String, String> {
        let text = if complete { normalize(decoded, tools)? } else {
            if decoded.contains(DSML) { return Err("unexpected DSML in Qwen output".into()); }
            const TAG: &str = "<tool_call>";
            let end = decoded.find("<tool_call").unwrap_or_else(|| {
                let held = (1..TAG.len()).rev().find(|&n| decoded.ends_with(&TAG[..n])).unwrap_or(0);
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
    let vc = llama_rs::VisionConfig { image_size: cfg.image_size, patch_size: cfg.patch_size, mean: cfg.mean, std: cfg.std };
    let mut ids = Vec::new(); let mut images = Vec::new(); let mut records = records.iter();
    for id in prompt {
        if id != image_id { ids.push(id); continue; }
        let bytes = dsv41::vision::image_bytes(records.next().unwrap(), local).map_err(|e| e.to_string())?;
        let pixels = llama_rs::vision::preprocess_image_letterboxed(&bytes, &vc).map_err(|e| e.to_string())?;
        let hash=crate::disk::fnv(&bytes,0);
        images.push(JobImage { start: ids.len(), prep: ImagePrep::Qwen { pixels, side }, hash });
        ids.extend(std::iter::repeat_n(image_id, side*side));
    }
    Ok((ids, images))
}

/// What the engine runs: a Qwen3.5 hybrid (llama-rs), or Qwen3.8-Flash-Next.
pub enum Hybrid {
    Qwen35(Model),
    #[cfg(any(feature = "cuda", feature = "webgpu"))]
    Flash(Box<crate::flashnext::FlashNext>),
    #[cfg(test)]
    Fake(Box<crate::qwen_real::FakeModel>),
}

impl From<Model> for Hybrid {
    fn from(m: Model) -> Self { Self::Qwen35(m) }
}

impl Hybrid {
    /// The Qwen3.5 model: its checkpoints and disk states are Qwen3.5's shape.
    fn qwen35(&self) -> Option<&llama_rs::Qwen35Model> {
        match self { Self::Qwen35(Model::Qwen35(m)) => Some(m), _ => None }
    }
    fn tokenizer(&self) -> Result<&tokenizer::Tokenizer, String> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => Ok(&m.tokenizer),
            Self::Qwen35(_) => Err("not a dense Qwen hybrid".into()),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(f) => Ok(&f.tokenizer),
            #[cfg(test)]
            Self::Fake(f) => Ok(&f.tok),
        }
    }
    fn width(&self) -> usize {
        match self {
            Self::Qwen35(m) => m.config().embedding_dim,
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
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
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(_) => true,
            #[cfg(test)]
            Self::Fake(_) => true,
        }
    }
    fn new_kv_cache(&self, max_seq: usize) -> KvCache {
        match self {
            Self::Qwen35(m) => m.new_kv_cache(max_seq),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(f) => f.new_kv_cache(max_seq),
            #[cfg(test)]
            Self::Fake(_) => crate::qwen_real::new_kv(max_seq),
        }
    }
    /// Token embeddings on the host.
    fn embed(&self, tokens: &[u32]) -> Result<Tensor, String> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => Ok(m.embed_text(tokens).to_host()),
            Self::Qwen35(_) => Err("not a dense Qwen hybrid".into()),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(f) => f.embed_text(tokens).map_err(|e| e.to_string()),
            #[cfg(test)]
            Self::Fake(f) => Ok(Tensor::zeros(vec![tokens.len(), f.width])),
        }
    }
    /// Whether the model drafts tokens a check takes (its multi-token-prediction layer, chained).
    fn drafts(&self) -> bool {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.drafts(),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(f) => f.drafts(),
            _ => false,
        }
    }
    /// The tokens before a draft's `next` it may need (its layer catches up on the last run's rows): at most this many.
    fn draft_window(&self) -> usize {
        match self {
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(_) => crate::flashnext::CHECK_ROWS,
            _ => llama_rs::SPEC_ROWS,
        }
    }
    /// Up to `k` drafts after the last of `recent` (the token sampled for position `kv.len`); None or none: a step.
    fn draft(&self, kv: &KvCache, recent: &[u32], k: usize) -> Option<Vec<u32>> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.draft(kv, recent, k),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(f) => f.draft(kv, recent, k),
            _ => None,
        }
    }
    /// A check of `rows` (the token sampled, then its drafts): every row's logits, undoable ([`Self::rollback`]).
    fn check(&self, rows: &[u32], kv: &mut KvCache) -> Option<Vec<Tensor>> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.check(rows, kv),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
            Self::Flash(f) => f.check(rows, kv),
            _ => None,
        }
    }
    /// Undo the last check's `rows` past its first `keep`.
    fn rollback(&self, kv: &mut KvCache, rows: usize, keep: usize) {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => m.rollback(kv, rows, keep),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
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
        #[cfg(any(feature = "cuda", feature = "webgpu"))]
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
            let width = self.width();
            let rows: usize = chunks.iter().map(|(t, _)| t.len()).sum();
            let mut emb = Vec::with_capacity(rows * width);
            let mut tokens = Vec::with_capacity(rows);
            for (t, e) in chunks {
                emb.extend_from_slice(e.data());
                tokens.extend_from_slice(t);
            }
            let out = self.forward(&tokens, Tensor::from_vec(emb, vec![rows, width]), kv, None)?;
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
    /// Whether a prompt's chunks go to [`Self::forward_chunks`] together: Flash-Next over several GPUs, a chained
    /// Qwen3.5 hybrid.
    fn pipelines(&self) -> bool {
        #[cfg(any(feature = "cuda", feature = "webgpu"))]
        if let Self::Flash(f) = self {
            return f.devices_len() > 1;
        }
        matches!(self, Self::Qwen35(Model::Qwen35(m)) if m.backend.chain().is_some())
    }
    /// Run `tokens` (embedded as `embeds`, on the host) after what `kv` holds: the last logits, on the host.
    fn forward(&self, tokens: &[u32], embeds: Tensor, kv: &mut KvCache, positions: Option<&[[u32; 3]]>) -> Result<Tensor, String> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => {
                let embeds = m.backend.to_device(embeds);
                Ok(m.forward_embeds_positions(&embeds, tokens.len(), kv, positions).map_err(|e| e.to_string())?.to_host())
            }
            Self::Qwen35(_) => Err("not a dense Qwen hybrid".into()),
            #[cfg(any(feature = "cuda", feature = "webgpu"))]
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
    vision_cache: std::collections::VecDeque<(u64, Tensor)>,
    log: bool,
    checkpoints: Vec<(Vec<u64>, RecurrentSnapshot, bool)>,
    pub disk: Option<crate::disk::DiskCache<Arc<Snapshot>>>,
    /// Only enable when the disk namespace fingerprints the vision pipeline.
    pub image_disk_cache: bool,
    /// The incognito session whose prompt state is held (in memory only), if any.
    private_session: Option<String>,
    /// Conversations set aside in host RAM (Flash-Next only; see `qwen_park`). Lives and dies
    /// with the engine, so a model reload (the only way its weights or LoRA change) starts it
    /// empty, and `kv` is never rebuilt for another context length.
    parking: Parking,
}

impl QwenEngine {
    pub fn new(model: impl Into<Hybrid>, projector: Option<MmProj>, max_seq: usize, log: bool) -> Self {
        let model = model.into();
        let kv = model.new_kv_cache(max_seq);
        Self { model, projector, kv, covered: Vec::new(), vision_cache: Default::default(), log, checkpoints: Vec::new(), disk: None, image_disk_cache:false, private_session: None, parking: Parking::new(0, log) }
    }
    /// Let the engine set conversations aside in host RAM, up to `bytes` (0: not at all), when
    /// its model can: only Qwen3.8-Flash-Next, which has no disk states of its own.
    pub fn park_up_to(mut self, bytes: usize) -> Self {
        if self.model.parks() { self.parking.set_budget(bytes); }
        self
    }
    pub fn run(mut self, jobs: Receiver<Job>) {
        for job in jobs {
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
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.generate(&job)));
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

    /// Forget everything held of a private request or session. Whatever was set aside in RAM
    /// goes too: the engine holds nothing afterwards, as it never did (the states set aside are
    /// never a private session's, but an incognito request ends with an empty engine).
    fn forget_state(&mut self) {
        self.kv.reset();
        self.covered.clear();
        self.checkpoints.clear();
        self.vision_cache.clear();
        self.parking.clear();
        self.private_session = None;
    }
    fn generate(&mut self, job: &Job) -> Result<(), String> {
        if job.prompt.is_empty() { return Err("empty Qwen prompt".into()); }
        let (positions, mut next_position) = positions(job)?;
        let mut keys: Vec<_> = job.prompt.iter().map(|&t| t as u64).collect();
        for image in &job.images { for (offset,k) in keys[image.start..image.start+image.prep.n_tokens()].iter_mut().enumerate() { *k = image.hash.rotate_left(17) ^ (offset as u64) ^ (1<<63); } }
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
        if let Some(model) = model.filter(|_| job.images.is_empty() || self.image_disk_cache) {
            if let Some((saved, snap)) = self.disk.as_mut().and_then(|d| d.load_best(&keys, keys.len()-1, start, self.log)) {
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
        // What came back from RAM is not "memory" or a checkpoint of this process's own making.
        if swapped && start > 0 { source = "ram"; }
        if start == 0 { self.kv.reset(); self.covered.clear(); self.checkpoints.clear(); }
        let _ = job.events.send(Event::CacheReuse { cached: start, source, common });
        if self.log { eprintln!("  Qwen cache: {start}/{} tokens from {source} (common {common}) in {:.3}s", keys.len(), cache_clock.elapsed().as_secs_f64()); }
        let total = keys.len()-start;
        let _ = job.events.send(Event::Progress { done: 0, total });
        let clock = std::time::Instant::now();
        let mut soft = Vec::new();
        for image in &job.images {
            if image.start + image.prep.n_tokens() <= start { continue; }
            if job.cancel.load(Ordering::Relaxed) { return Ok(()); }
            let t = if let Some((_,t)) = self.vision_cache.iter().find(|(hash,_)| *hash == image.hash) { t.clone() } else {
                let ImagePrep::Qwen { pixels, .. } = &image.prep else { return Err("wrong image format for Qwen".into()); };
                let t = self.projector.as_ref().ok_or("Qwen vision projector is not configured")?.forward(pixels).map_err(|e| e.to_string())?.to_host();
                self.vision_cache.push_back((image.hash,t.clone()));
                while self.vision_cache.len() > 8 { self.vision_cache.pop_front(); }
                t
            };
            soft.push((image.start,t));
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
            if hybrid.pipelines() && job.images.is_empty() {
                let stop = stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()).min(keys.len()).min(pos + 8 * PREFILL_CHUNK);
                let spans: Vec<(usize, usize)> = (pos..stop).step_by(PREFILL_CHUNK).map(|a| (a, (a + PREFILL_CHUNK).min(stop))).collect();
                let embeds = spans.iter().map(|&(a, b)| hybrid.embed(&job.prompt[a..b])).collect::<Result<Vec<_>, _>>()?;
                if said { eprintln!("  prefill: {} rows embedded in {:.1} ms", stop - pos, began.elapsed().as_secs_f64() * 1e3); }
                let chunks: Vec<(&[u32], Tensor)> = spans.iter().zip(embeds).map(|(&(a, b), e)| (&job.prompt[a..b], e)).collect();
                let events = &job.events;
                let mut done = |i: usize| { let _ = events.send(Event::Progress { done: spans[i].1 - start, total }); };
                logits = Some(hybrid.forward_chunks(&chunks, &mut self.kv, &mut done)?);
                self.covered.extend_from_slice(&keys[pos..stop]);
                pos = stop;
            } else {
                let end = (pos+PREFILL_CHUNK).min(keys.len()).min(stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()));
                let mut embeds = hybrid.embed(&job.prompt[pos..end])?;
                let width = hybrid.width();
                for (at,t) in &soft {
                    let from = pos.max(*at); let to = end.min(*at+t.dim(0));
                    if from < to { embeds.data_mut()[(from-pos)*width..(to-pos)*width].copy_from_slice(&t.data()[(from-at)*width..(to-at)*width]); }
                }
                let out = hybrid.forward(&job.prompt[pos..end], embeds, &mut self.kv, (!job.images.is_empty()).then_some(&positions[pos..end]))?;
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
                // Disk states are the Qwen3.5 hybrid's; memory checkpoints any model's.
                if let Some(model) = model.filter(|_| job.images.is_empty() || self.image_disk_cache) {
                    if let Some(disk) = self.disk.as_mut().filter(|_| !job.forget) {
                        if !disk.has(&keys[..pos]) {
                            let snap = Arc::new(Snapshot::capture(&self.kv, &model.attention_layers, model.backend.as_ref()));
                            disk.save(keys[..pos].to_vec(), snap, base);
                        }
                    }
                }
                let snap = RecurrentSnapshot::capture(&self.kv);
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
        let mut stream = NativeStream::default();
        let mut rng = job.sampling.seed ^ 0x9E3779B97F4A7C15;
        let mut logits = logits.unwrap();
        let mut finish = Finish::Length;
        // Where decoding's time goes, for the log: sampling, the text (decode and stream), the model.
        let (mut t_sample, mut t_text, mut t_model) = (0f64, 0f64, 0f64);
        // Drafting (a model with its multi-token-prediction layer, chained: a Qwen3.5 GGUF's, Flash-Next's; text only):
        // each token sampled is run with the drafts after it in one check, and while the sampler picks what was drafted
        // its logits are the check's (sampling the model's distribution and taking a draft only where it is the token
        // sampled is that distribution's sampling still); where it picks another, the check's rows from there are undone.
        let drafter = (job.images.is_empty() && hybrid.drafts()).then_some(hybrid);
        let mut pending: std::collections::VecDeque<(u32, Tensor)> = Default::default();
        let (mut check_rows, mut drafts_checked, mut drafts_taken) = (0usize, 0usize, 0usize);
        while generated.len() < job.max_tokens && self.kv.len < self.kv.max_len {
            if job.cancel.load(Ordering::Relaxed) {
                if let (Some(m), false) = (drafter, pending.is_empty()) { m.rollback(&mut self.kv, check_rows, check_rows - pending.len()); }
                return Ok(());
            }
            let clock = std::time::Instant::now();
            let mut next = sample(logits.data(), &job.sampling, &mut rng);
            t_sample += clock.elapsed().as_secs_f64();
            if thinking && job.think_budget.is_some_and(|n|think_used >= n) { if let Some(end) = think_end { next = end; } }
            let stop = next == eos || Some(next) == tok.eos();
            // the check ran this token already where it is the next draft; else its rows from here on are undone
            let ran = !stop && pending.front().is_some_and(|(d, _)| *d == next);
            if !ran && !pending.is_empty() {
                let m = drafter.expect("drafts are a drafter's");
                m.rollback(&mut self.kv, check_rows, check_rows - pending.len());
                pending.clear();
            }
            if stop { finish = Finish::Stop; break; }
            generated.push(next);
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
                // its drafts, then the token and them in one check; else the token alone
                let checked = drafter.filter(|_| self.kv.len + 1 + DRAFTS <= self.kv.max_len).and_then(|m| {
                    let recent: Vec<u32> = self.covered[self.covered.len().saturating_sub(m.draft_window())..].iter().map(|&k| k as u32).chain([next]).collect();
                    // none the layer is sure enough of: a step of the token alone
                    let drafts = m.draft(&self.kv, &recent, DRAFTS).filter(|d| !d.is_empty())?;
                    let rows: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
                    Some((drafts, m.check(&rows, &mut self.kv)?))
                });
                match checked {
                    Some((drafts, mut rows)) => {
                        check_rows = rows.len();
                        drafts_checked += drafts.len();
                        logits = rows.remove(0);
                        pending = drafts.into_iter().zip(rows).collect();
                    }
                    None => {
                        let embeds = hybrid.embed(&[next])?;
                        logits = hybrid.forward(&[next], embeds, &mut self.kv, (!job.images.is_empty()).then_some(&[[next_position;3]]))?;
                    }
                }
            }
            t_model += clock.elapsed().as_secs_f64();
            next_position += 1;
            self.covered.push(next as u64);
        }
        // a check's rows no token reached
        if let (Some(m), false) = (drafter, pending.is_empty()) {
            m.rollback(&mut self.kv, check_rows, check_rows - pending.len());
        }
        let raw = tok.decode(&generated);
        // The client is told what the model wrote, so a rejected call can be read
        // and fixed; the server log never holds it.
        let text = stream.push(&raw, &job.tools, true)
            .map_err(|e| format!("tool_contract_error: {e}; no tool from this batch was executed. The model wrote: {}", call_excerpt(&raw)))?;
        if self.log { eprintln!("  Qwen: {} prompt tokens ({} cached) in {:.2}s; {} generated in {:.2}s (sampling {t_sample:.2}s, text {t_text:.2}s, model {t_model:.2}s){}", job.prompt.len(),start,prefill_secs,generated.len(),decode_clock.elapsed().as_secs_f64(), if drafts_checked > 0 { format!("; {drafts_taken} of {drafts_checked} drafts taken") } else { String::new() }); }
        if !text.is_empty() { let _ = job.events.send(Event::Text(text)); }
        let _ = job.events.send(Event::Done { finish, completion_tokens: generated.len() });
        Ok(())
    }
}

/// Tokens drafted a check, where a model drafts (its multi-token-prediction layer): Qwen3.8 27B's take 0.86, 0.73
/// and 0.65 in turn (each where the ones before it were).
const DRAFTS: usize = 3;

fn trim_checkpoints(checkpoints: &mut Vec<(Vec<u64>, RecurrentSnapshot, bool)>) {
    while checkpoints.len() > 1 && (checkpoints.len() > 3 ||
        checkpoints.iter().map(|(_, s, _)| s.bytes()).sum::<usize>() > 1024 * 1024 * 1024) {
        let remove = (0..checkpoints.len()-1).min_by_key(|&i| (checkpoints[i].2, i)).unwrap();
        checkpoints.remove(remove);
    }
}

/// Save before the first user message, before the current assistant header
/// (thinking/tool rendering changes its suffix), and one token short of the
/// entire prompt so an identical retry still runs a token to recover logits.
/// A checkpoint this close before the prompt's last token serves the same prompt asked again as one at that token
/// would: the tokens between are run from it.
const NEAR_THE_END: usize = 16;

fn checkpoint_positions(prompt: &[u32], im_start: Option<u32>) -> Vec<usize> {
    let boundaries: Vec<_> = prompt.iter().enumerate().filter_map(|(i, &t)|
        (i > 0 && Some(t) == im_start).then_some(i)).collect();
    // The position before the last (the same prompt asked again runs one token), unless the last boundary is a few
    // tokens back (a chat's: the assistant's header): a checkpoint is the recurrent states to the host (the 27B's
    // 149 MB, 30 ms) and a run cut in two there, each prompt, for the header's few tokens' 30 ms when one is asked again.
    let last = prompt.len().saturating_sub(1);
    let mut stops = Vec::new();
    if boundaries.last().map_or(true, |&b| last.saturating_sub(b) > NEAR_THE_END) { stops.push(last); }
    stops.extend(boundaries.first().copied());
    stops.extend(boundaries.last().copied());
    stops.retain(|&p| p > 0 && p < prompt.len());
    stops.sort_unstable(); stops.dedup(); stops
}

fn positions(job: &Job) -> Result<(Vec<[u32;3]>,u32),String> {
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
mod tests {
    use super::*;
    #[test]
    fn native_file_call_round_trip_preserves_literal_content() {
        let tools = vec![Json::parse(br#"{"function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}"#).unwrap()];
        let content = "\n<style>\n  :root { --bg: #000; }\n  body::before { background: var(--bg); content: 'λ 😀'; }\n</style>\n<script>const x = a < b && b > 0;</script>\n\n";
        let raw = format!("<tool_call>\n<function=write_file>\n<parameter=path>\nsite/index.html\n</parameter>\n<parameter=content>\n{content}\n</parameter>\n</function>\n</tool_call>");
        let mut stream = NativeStream::default();
        let mut parser = dsv41::chat::StreamParser::new(Mode::Chat);
        for end in raw.char_indices().map(|(at,c)| at+c.len_utf8()) {
            parser.push(&stream.push(&raw[..end], &tools, false).unwrap());
            assert!(!parser.tool_calls_ready());
        }
        parser.push(&stream.push(&raw, &tools, true).unwrap());
        assert!(parser.tool_call_error().is_none());
        let (_, calls) = parser.finish();
        assert_eq!(calls.len(), 1);
        let args = Json::parse(calls[0].arguments.as_bytes()).unwrap();
        assert_eq!(args.get("content").and_then(Json::as_str), Some(content));
        assert_eq!(args.get("path").and_then(Json::as_str), Some("site/index.html"));
    }

    #[test]
    fn native_stream_releases_reasoning_and_prose_before_completion() {
        let mut stream = NativeStream::default();
        let mut parser = dsv41::chat::StreamParser::new(Mode::Thinking);
        let first = stream.push("Let me check", &[], false).unwrap();
        assert!(matches!(&parser.push(&first)[..], [dsv41::chat::Delta::Reasoning(t)] if t == "Let me check"));
        let next = stream.push("Let me check</think>Hello", &[], false).unwrap();
        assert!(parser.push(&next).iter().any(|d| matches!(d,dsv41::chat::Delta::Content(t) if t == "Hello")));
        assert!(stream.push("Let me check</think>Hello", &[], true).unwrap().is_empty());
        let mut utf8 = NativeStream::default();
        assert_eq!(utf8.push("Hi \u{fffd}", &[], false).unwrap(), "Hi ");
        assert_eq!(utf8.push("Hi 😀", &[], false).unwrap(), "😀");
    }

    #[test]
    fn native_stream_never_exposes_partial_or_invalid_tool_calls() {
        let raw = "Checking. <tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n<parameter=keys>\n[]\n</parameter>\n</function>\n</tool_call>";
        let mut stream = NativeStream::default();
        let mut emitted = String::new();
        for end in 1..=raw.len() { emitted.push_str(&stream.push(&raw[..end], &tools(), false).unwrap()); }
        assert_eq!(emitted, "Checking. ");
        let (preview,start) = stream.tool_preview(raw).unwrap().unwrap();
        assert!(start && preview.starts_with("<tool_call>"));
        assert!(stream.tool_preview(raw).unwrap().is_none());
        emitted.push_str(&stream.push(raw, &tools(), true).unwrap());
        assert_eq!(emitted, normalize(raw, &tools()).unwrap());
        for invalid in [raw.replace("function=computer", "function=unknown"), raw.replace("</tool_call>", "")] {
            let mut stream = NativeStream::default();
            assert_eq!(stream.push(&invalid, &tools(), false).unwrap(), "Checking. ");
            assert!(stream.push(&invalid, &tools(), true).is_err());
        }
    }

    #[test]
    fn checkpoints_survive_changed_assistant_suffix_and_leave_logits_token() {
        // 1 is im_start; later reasoning and tool XML need not match.
        let first = [1, 10, 11, 1, 20, 21, 1, 30, 31, 32];
        let next = [1, 10, 11, 1, 20, 21, 1, 30, 40, 41, 1, 50];
        let stops = checkpoint_positions(&first, Some(1));
        // (the last boundary three tokens from the end: none at the token before the last)
        assert_eq!(stops, [3, 6]);
        // a boundary far from the end leaves the one before the last token
        let mut long = vec![7u32; 40];
        long[3] = 1;
        assert_eq!(checkpoint_positions(&long, Some(1)), [3, 39]);
        assert_eq!(stops.iter().copied().filter(|&p| next.starts_with(&first[..p])).max(), Some(6));
        assert_eq!(checkpoint_positions(&[1], Some(1)), []);
        assert_eq!(checkpoint_positions(&[7, 8, 9], None), [2]);
    }
    fn tools() -> Vec<Json> { vec![Json::parse(br#"{"function":{"name":"computer","parameters":{"type":"object","properties":{"action":{"type":"string"},"keys":{"type":"array"}},"required":["action"]}}}"#).unwrap()] }
    #[test]
    fn native_calls_preserve_types_and_reject_partial_or_unknown_calls() {
        let call = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n<parameter=keys>\n[{\"x\":42,\"key\":\"h\"}]\n</parameter>\n</function>\n</tool_call>";
        let converted=normalize(call,&tools()).unwrap();
        let mut parser=dsv41::chat::StreamParser::new(Mode::Chat); parser.push(&converted);
        assert!(parser.tool_call_error().is_none()); let (_,calls)=parser.finish();
        let args=Json::parse(calls[0].arguments.as_bytes()).unwrap(); assert!(args.get("keys").unwrap().as_array().is_some());
        assert!(normalize(&call.replace("</tool_call>",""),&tools()).is_err());
        assert!(normalize(&call.replace("function=computer","function=unknown"),&tools()).is_err());
        assert!(normalize(&format!("{call} unwanted suffix"),&tools()).is_err());
    }
    #[test]
    fn a_function_named_with_the_parameter_tag_is_read_as_the_function() {
        let call = "<tool_call>\n<parameter=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n</parameter>\n</tool_call>";
        let fixed = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(normalize(call, &tools()).unwrap(), normalize(fixed, &tools()).unwrap());
        // No stray closing tag, or a proper </function>: read the same.
        assert!(normalize(&call.replace("</parameter>\n</parameter>", "</parameter>"), &tools()).is_ok());
        assert!(normalize(&call.replace("</parameter>\n</parameter>", "</parameter>\n</function>"), &tools()).is_ok());
        // Only for a declared function, and only when the tags balance.
        assert!(normalize(&call.replace("parameter=computer", "parameter=unknown"), &tools()).is_err());
        assert!(normalize(&call.replace("</parameter>\n</parameter>", "</parameter>\n</parameter>\n</parameter>"), &tools()).is_err());
    }
    #[test]
    fn a_function_tag_behind_a_stray_template_token_is_read_as_the_function() {
        let fixed = "<tool_call>\n<function=computer>\n<parameter=action>\nkeyboard_sequence\n</parameter>\n</function>\n</tool_call>";
        let want = normalize(fixed, &tools()).unwrap();
        // `<|im_start|>` where the `<` belongs, or no `<` at all.
        assert_eq!(normalize(&fixed.replace("<function=", "<|im_start|>function="), &tools()).unwrap(), want);
        assert_eq!(normalize(&fixed.replace("<function=", "function="), &tools()).unwrap(), want);
        assert_eq!(normalize(&fixed.replace("<function=", "<|im_start|>function=").replace("\n</function>", ""), &tools()).unwrap(), want);
        // Still only a declared function with balanced tags, and only a template-like token.
        assert!(normalize(&fixed.replace("<function=computer", "<|im_start|>function=unknown"), &tools()).is_err());
        assert!(normalize(&fixed.replace("<function=", "<|im_start|>function=").replace("</parameter>", ""), &tools()).is_err());
        assert!(normalize(&fixed.replace("<function=", "<|a b|>function="), &tools()).is_err());
    }
    #[test]
    fn rejected_calls_are_quoted_from_their_start() {
        assert_eq!(call_excerpt("prose <tool_call>\n{\"name\":\"x\"}"), "\"<tool_call>\\n{\\\"name\\\":\\\"x\\\"}\"");
        let long = format!("<tool_call>{}", "é".repeat(400));
        assert!(call_excerpt(&long).ends_with('…'));
    }
    #[test]
    fn template_keeps_tool_images_and_non_thinking_prefix() {
        let msgs=Json::parse(br#"[{"role":"user","content":"look"},{"role":"tool","content":[{"type":"text","text":"screen"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]"#).unwrap();
        let encoded=chat_prompt(msgs.as_array().unwrap(),&Options {mode:Mode::Chat,effort:25,drop_thinking:true}).unwrap();
        assert_eq!(encoded.images.len(),1); assert!(encoded.prompt.contains(IMAGE));
        assert!(encoded.prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    // ---- Two conversations taking turns on Qwen3.8-Flash-Next: what the owner's call test did
    // between 11:22 and 11:26, a runner of about 21,000 tokens and a call's sub-agent of about
    // 5,000, sharing a 520-token system prompt. Every switch read the whole prompt again (36 s).

    use crate::qwen_park::fixtures::{dump, kv_like_flash, ROWS};
    use crate::qwen_park::Checkpoint;

    const IM: u32 = 1;

    fn mix(t: u32, pos: u32, salt: u32, j: u32) -> f32 {
        let mut h = ((t as u64) << 32) ^ (pos as u64);
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (((salt as u64) << 8) | j as u64);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
        (h % 2000) as f32 / 16.0 - 62.5
    }

    /// A fake model: what it holds is a pure function of the token sequence, as a transformer's
    /// cache is (a row for each token and position, recurrent states that fold every token in), so
    /// a cache that holds a prefix and one built from nothing agree to the last bit.
    fn forward(kv: &mut KvCache, tokens: &[u32]) {
        for &t in tokens {
            let pos = kv.len as u32;
            for (slot, heads, dim) in ROWS {
                let k: Vec<f32> = (0..heads * dim).map(|j| mix(t, pos, slot as u32, j as u32)).collect();
                let v: Vec<f32> = k.iter().map(|x| -x * 0.5).collect();
                let backend = kv.layer_backends[slot].clone();
                kv.append(backend.as_ref(), slot, &Tensor::from_vec(k, vec![1, heads, dim]), &Tensor::from_vec(v, vec![1, heads, dim]));
            }
            kv.commit(1);
            let fold = |old: Option<Tensor>, shape: Vec<usize>, salt: u32| -> Tensor {
                let mut data = old.map_or_else(|| vec![0.0; shape.iter().product()], |o| o.data().to_vec());
                for (j, x) in data.iter_mut().enumerate() { *x = *x * 0.75 + mix(t, 0, salt, j as u32); }
                Tensor::from_vec(data, shape)
            };
            kv.ssm_state[1] = Some(fold(kv.ssm_state[1].take(), vec![2, 3, 3], 11));
            kv.ssm_conv[1] = Some(fold(kv.ssm_conv[1].take(), vec![3, 7], 12));
            kv.ssm_state[3] = Some(fold(kv.ssm_state[3].take(), vec![2], 13));
            kv.ssm_conv[3] = Some(fold(kv.ssm_conv[3].take(), vec![4, 5], 14));
        }
    }

    /// What a cache holds after `prompt` and `reply`, read from nothing.
    fn cold(prompt: &[u32], reply: &[u32]) -> Vec<Option<Vec<u32>>> {
        let mut kv = kv_like_flash(65536);
        forward(&mut kv, prompt);
        forward(&mut kv, reply);
        dump(&kv)
    }

    struct Served { source: &'static str, start: usize, swapped: bool, read: usize }

    /// The engine's state, and `QwenEngine::generate`'s cache logic for Flash-Next as written there
    /// (no disk, no Qwen3.5 checkpoint restore), with the fake model's prefill and decode.
    struct Mini { kv: KvCache, covered: Vec<u64>, checkpoints: Vec<Checkpoint>, parking: Parking }

    impl Mini {
        fn new(budget: usize, log: bool) -> Self {
            Mini { kv: kv_like_flash(65536), covered: Vec::new(), checkpoints: Vec::new(), parking: Parking::new(budget, log) }
        }
        fn serve(&mut self, prompt: &[u32], reply: &[u32], forget: bool) -> Served {
            let keys: Vec<u64> = prompt.iter().map(|&t| t as u64).collect();
            let stops = checkpoint_positions(prompt, Some(IM));
            let swapped = !forget && self.parking.swap_in(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: false }, &keys);
            if !forget {
                self.parking.park_displaced(Held { kv: &mut self.kv, covered: &mut self.covered, checkpoints: &mut self.checkpoints, private: false }, &keys);
            }
            let common = self.covered.iter().zip(&keys).take_while(|(a, b)| a == b).count();
            self.checkpoints.retain(|(saved, _, _)| saved.len() <= common && keys.starts_with(saved));
            let mut start = if common == self.covered.len() && common < keys.len() { common } else { 0 };
            let mut source = if start > 0 { "memory" } else { "none" };
            if let Some((saved, snap, _)) = self.checkpoints.iter().filter(|(saved, _, _)| saved.len() > start && saved.len() < keys.len() && keys.starts_with(saved)).max_by_key(|(saved, _, _)| saved.len()) {
                snap.restore_slots(&mut self.kv, common).unwrap();
                self.covered = saved.clone(); start = saved.len(); source = "checkpoint";
                self.checkpoints.retain(|(saved, _, _)| saved.len() <= start);
            }
            if swapped && start > 0 { source = "ram"; }
            if start == 0 { self.kv.reset(); self.covered.clear(); self.checkpoints.clear(); }
            let mut pos = start;
            while pos < keys.len() {
                let end = (pos + PREFILL_CHUNK).min(keys.len()).min(stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()));
                forward(&mut self.kv, &prompt[pos..end]);
                self.covered.extend_from_slice(&keys[pos..end]);
                pos = end;
                if stops.contains(&pos) && !self.checkpoints.iter().any(|(saved, _, _)| saved == &keys[..pos]) {
                    let base = stops.first() == Some(&pos);
                    self.checkpoints.push((keys[..pos].to_vec(), RecurrentSnapshot::capture(&self.kv), base));
                    trim_checkpoints(&mut self.checkpoints);
                }
            }
            for &t in reply { forward(&mut self.kv, &[t]); self.covered.push(t as u64); }
            Served { source, start, swapped, read: keys.len() - start }
        }
    }

    /// A conversation: the system prompt every conversation starts with (520 tokens), a system
    /// prompt of its own after it (300 tokens, so that the first message boundary lies past what
    /// they share), and the messages that follow.
    struct Chat { id: u32, messages: Vec<Vec<u32>>, shared_system: bool }

    impl Chat {
        fn new(id: u32, first: usize) -> Self {
            Chat { id, messages: vec![Self::message(id, 1, first)], shared_system: false }
        }
        /// As `new`, but its own part of the system prompt is the same for every conversation: all
        /// of them share it up to the first message boundary.
        fn sharing_its_system_prompt(id: u32, first: usize) -> Self {
            Chat { shared_system: true, ..Self::new(id, first) }
        }
        /// A message of `n` tokens of its own; it begins at a message boundary.
        fn message(id: u32, turn: u32, n: usize) -> Vec<u32> {
            std::iter::once(IM).chain((1..n as u32).map(|i| (id << 20) + (turn << 16) + i)).collect()
        }
        /// The assistant's turn as a later prompt renders it: its header, the reply, its end.
        fn answered(&mut self, reply: &[u32], rendered_with: Option<u32>, next: usize) {
            let mut turn = vec![IM, 2];
            turn.extend(rendered_with);
            turn.extend(reply);
            turn.push(3);
            self.messages.push(turn);
            let n = self.messages.len() as u32;
            self.messages.push(Self::message(self.id, n, next));
        }
        /// What a request sends: the system prompts, the messages and the assistant's header.
        fn prompt(&self) -> Vec<u32> {
            let mut p: Vec<u32> = std::iter::once(IM).chain(5..524).collect();
            assert_eq!(p.len(), 520);
            let own = if self.shared_system { 0 } else { self.id };
            p.extend((0..300).map(|i| (own << 20) + (1 << 19) + i));
            for m in &self.messages { p.extend(m); }
            p.extend([IM, 2]);
            p
        }
    }

    fn lengths(m: &Mini) -> Vec<usize> {
        m.parking.keys_held().iter().map(Vec::len).collect()
    }

    #[test]
    fn two_conversations_taking_turns_swap_in_with_full_reuse_after_the_first_round() {
        let reply = |id: u32, n: u32| -> Vec<u32> { (0..n).map(|i| 900_000 + id * 1_000 + i).collect() };
        let (ra1, ra2, ra3, rb1, rb2, rb3) = (reply(1, 40), reply(2, 25), reply(3, 60), reply(4, 30), reply(5, 20), reply(6, 35));
        let mut m = Mini::new(1 << 40, true);
        // The runner (A) and the call's sub-agent (B).
        let (mut a, mut b) = (Chat::new(10, 16_180), Chat::new(20, 3_680));

        // The first round reads everything: nothing was set aside yet.
        let a1 = a.prompt();
        let s = m.serve(&a1, &ra1, false);
        assert_eq!((s.source, s.start, s.swapped, s.read), ("none", 0, false, a1.len()));
        let b1 = b.prompt();
        let s = m.serve(&b1, &rb1, false);
        assert_eq!((s.source, s.start, s.swapped, s.read), ("none", 0, false, b1.len()), "520 tokens in common are no checkpoint");
        // A1 was big and mostly unrelated: set aside as B1 displaced it.
        assert_eq!(lengths(&m), [a1.len() + ra1.len()]);

        // From here on every request is a swap-in that reads only what is new.
        a.answered(&ra1, None, 2_000);
        let a2 = a.prompt();
        let s = m.serve(&a2, &ra2, false);
        assert_eq!((s.source, s.swapped), ("ram", true));
        assert_eq!(s.start, a1.len() + ra1.len(), "all of A1 and its reply");
        assert_eq!(s.read, a2.len() - (a1.len() + ra1.len()));
        assert!(s.read < 2_100, "only the new message, the end of the reply and the header");
        assert_eq!(dump(&m.kv), cold(&a2, &ra2), "the restored cache goes on bit for bit as a cold read of the whole prompt would");
        assert_eq!(lengths(&m), [b1.len() + rb1.len()], "B1 waits now");

        b.answered(&rb1, None, 400);
        let b2 = b.prompt();
        let s = m.serve(&b2, &rb2, false);
        assert_eq!((s.source, s.swapped, s.start), ("ram", true, b1.len() + rb1.len()));
        assert_eq!(s.read, b2.len() - (b1.len() + rb1.len()));
        assert_eq!(dump(&m.kv), cold(&b2, &rb2));
        assert_eq!(lengths(&m), [a2.len() + ra2.len()]);

        // The runner's next prompt renders its last reply with something more in it: it goes back
        // to the checkpoint taken before the reply, which was set aside with the state.
        a.answered(&ra2, Some(9), 1_500);
        let a3 = a.prompt();
        let s = m.serve(&a3, &ra3, false);
        assert_eq!((s.source, s.swapped), ("ram", true));
        assert_eq!(s.start, a2.len() - 1, "the checkpoint one token short of A2's prompt");
        assert_eq!(s.read, a3.len() - (a2.len() - 1));
        assert_eq!(dump(&m.kv), cold(&a3, &ra3));

        b.answered(&rb2, None, 300);
        let b3 = b.prompt();
        let s = m.serve(&b3, &rb3, false);
        assert_eq!((s.source, s.swapped, s.start), ("ram", true, b2.len() + rb2.len()));
        assert_eq!(dump(&m.kv), cold(&b3, &rb3));
    }

    #[test]
    fn a_reply_rendered_differently_still_finds_its_checkpoint_after_a_conversation_was_discarded() {
        // The runner's A1 is set aside as B1 displaces it (nothing of it is shared but 520 tokens),
        // and its next prompt renders the last reply with something more in it, so that it
        // can only go back to a checkpoint of A1: the checkpoints went into the stash with it,
        // though the engine drops the ones a prompt does not share as soon as B1 is read.
        let (ra1, ra2, rb1) = (vec![902_000; 40], vec![903_000; 25], vec![904_000; 30]);
        let mut m = Mini::new(1 << 40, false);
        let (mut a, b) = (Chat::new(10, 16_180), Chat::new(20, 3_680));
        let a1 = a.prompt();
        m.serve(&a1, &ra1, false);
        m.serve(&b.prompt(), &rb1, false);
        assert_eq!(lengths(&m), [a1.len() + ra1.len()]);
        a.answered(&ra1, Some(9), 2_000);
        let a2 = a.prompt();
        let s = m.serve(&a2, &ra2, false);
        assert_eq!((s.source, s.swapped), ("ram", true));
        assert_eq!(s.start, a1.len() - 1, "the checkpoint one token short of A1's prompt");
        assert_eq!(s.read, a2.len() - (a1.len() - 1));
        assert_eq!(dump(&m.kv), cold(&a2, &ra2));
    }

    #[test]
    fn a_conversation_rolled_back_to_a_shared_system_prompt_is_set_aside_as_a_copy() {
        // The two conversations share their whole system prompt, which ends at a message boundary:
        // B1 goes back to A1's checkpoint there and writes over the rest of A1. A1 has to be set
        // aside before that, and the checkpoint it leaves B1 is still B1's to use.
        let (ra1, ra2, rb1, rb2) = (vec![902_000; 40], vec![903_000; 25], vec![904_000; 30], vec![905_000; 20]);
        let mut m = Mini::new(1 << 40, false);
        let (mut a, mut b) = (Chat::sharing_its_system_prompt(10, 16_180), Chat::sharing_its_system_prompt(20, 3_680));
        let a1 = a.prompt();
        m.serve(&a1, &ra1, false);
        let b1 = b.prompt();
        let s = m.serve(&b1, &rb1, false);
        assert_eq!((s.source, s.start, s.swapped), ("checkpoint", 820, false), "B1 reads on from A1's checkpoint at the end of the system prompt");
        assert_eq!(dump(&m.kv), cold(&b1, &rb1));
        assert_eq!(lengths(&m), [a1.len() + ra1.len()], "and A1 waits");
        a.answered(&ra1, None, 2_000);
        let a2 = a.prompt();
        let s = m.serve(&a2, &ra2, false);
        assert_eq!((s.source, s.swapped, s.start), ("ram", true, a1.len() + ra1.len()));
        assert_eq!(dump(&m.kv), cold(&a2, &ra2));
        b.answered(&rb1, None, 400);
        let b2 = b.prompt();
        let s = m.serve(&b2, &rb2, false);
        assert_eq!((s.source, s.swapped, s.start), ("ram", true, b1.len() + rb1.len()));
        assert_eq!(dump(&m.kv), cold(&b2, &rb2));
    }

    #[test]
    fn without_it_every_switch_reads_the_whole_prompt_again_and_with_it_only_what_is_new() {
        let reply = |id: u32| -> Vec<u32> { (0..30).map(|i| 900_000 + id * 1_000 + i).collect() };
        // (prompt length, tokens read) of each request, three rounds of the two conversations.
        let rounds = |budget: usize| -> Vec<(usize, usize)> {
            let mut m = Mini::new(budget, false);
            let (mut a, mut b) = (Chat::new(10, 16_180), Chat::new(20, 3_680));
            let mut out = Vec::new();
            for round in 0..3u32 {
                for (chat, id) in [(&mut a, 1u32), (&mut b, 2)] {
                    if round > 0 { chat.answered(&reply(id * 10 + round - 1), None, 500); }
                    let p = chat.prompt();
                    out.push((p.len(), m.serve(&p, &reply(id * 10 + round), false).read));
                }
            }
            out
        };
        let plain = rounds(0);
        assert!(plain.iter().all(|&(len, read)| read == len), "what ran on 1 October: {plain:?}");
        let parked = rounds(1 << 40);
        assert_eq!(parked[..2], plain[..2]);
        // One token (the end of the reply), the message of 500 and the header: 503.
        assert!(parked[2..].iter().all(|&(_, read)| read == 503), "{parked:?}");
        // A budget that holds one conversation and not both keeps the one displaced last, and the
        // other is read whole.
        let tight = rounds(2_000_000);
        let total = |r: &[(usize, usize)]| r.iter().map(|x| x.1).sum::<usize>();
        assert!(total(&parked) < total(&tight) && total(&tight) <= total(&plain), "{} {} {}", total(&parked), total(&tight), total(&plain));
    }

    #[test]
    fn an_incognito_request_neither_takes_from_the_stash_nor_adds_to_it() {
        let mut m = Mini::new(1 << 40, false);
        let (a, b) = (Chat::new(10, 6_000), Chat::new(20, 3_000));
        m.serve(&a.prompt(), &[7, 8], false);
        m.serve(&b.prompt(), &[7, 8], false);
        assert_eq!(m.parking.held().0, 1, "A waits");
        // An incognito request that continues A takes nothing from the stash, and B, which it
        // displaces, is not set aside for it.
        let mut again = Chat { id: 10, messages: a.messages.clone(), shared_system: false };
        again.answered(&[7, 8], None, 100);
        let s = m.serve(&again.prompt(), &[], true);
        assert_eq!((s.source, s.start, s.swapped), ("none", 0, false));
        assert_eq!(lengths(&m).len(), 1, "A is still the only one");
        // And the engine forgets everything when it ends (QwenEngine::forget_state).
        m.kv.reset(); m.covered.clear(); m.checkpoints.clear(); m.parking.clear();
        assert_eq!(m.parking.held(), (0, 0));
    }
}
