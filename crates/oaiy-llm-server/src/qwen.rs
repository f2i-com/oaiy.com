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
    #[cfg(feature = "webgpu")]
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
            Self::Fake(_) => crate::qwen_real::new_kv(max_seq),
        }
    }
    /// Token embeddings on the host.
    fn embed(&self, tokens: &[u32]) -> Result<Tensor, String> {
        match self {
            Self::Qwen35(Model::Qwen35(m)) => Ok(m.embed_text(tokens).to_host()),
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
            Self::Flash(_) => crate::flashnext::CHECK_ROWS,
            _ => llama_rs::SPEC_ROWS,
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
    fn warms(&self) -> bool {
        #[cfg(feature = "webgpu")]
        if let Self::Flash(_) = self {
            return true;
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
    fn forward_chunks_tapped(&self, chunks: &[(&[u32], Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize]) -> Result<(Tensor, Vec<llama_rs::Tapped>), String> {
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
        match self {
            Self::Qwen35(Model::Qwen35(m)) => {
                let embeds = m.backend.to_device(embeds);
                Ok(m.forward_embeds_positions(&embeds, tokens.len(), kv, positions).map_err(|e| e.to_string())?.to_host())
            }
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
        self.warm_up();
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
        // (the prompt's length, for the devices' copies of the attention cache: room for it made once)
        self.kv.expect = keys.len();
        let _ = job.events.send(Event::Progress { done: 0, total });
        #[cfg(feature = "webgpu")]
        let cold_at_start = ggml_rs_wgpu::quant_moe::cached_experts();
        #[cfg(feature = "webgpu")]
        let scratch_at_start = ggml_rs_wgpu::scratch_made();
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
                // the checkpoints this run would reach that are not kept yet: kept from inside it where the model can
                // (its states copied as the run goes: a prompt's last two, before the assistant's header and before
                // its last token, were a run of 6 rows and a run of 1 of their own), else a run's end each as before
                // (a disk's checkpoints are whole states: a run's end too)
                let far = (pos + 8 * PREFILL_CHUNK).min(keys.len());
                let due: Vec<usize> = stops.iter().copied().filter(|&s| s > pos && s <= far && !self.checkpoints.iter().any(|(saved, _, _)| saved == &keys[..s])).collect();
                let inside = !due.is_empty() && (self.disk.is_none() || job.forget) && self.kv.len == pos && hybrid.can_tap(far - pos, &self.kv, &due.iter().map(|s| s - pos).collect::<Vec<_>>());
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
                if inside {
                    let (out, kept) = hybrid.forward_chunks_tapped(&chunks, &mut self.kv, &mut done, &due.iter().map(|s| s - pos).collect::<Vec<_>>())?;
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
        let mut stream = NativeStream::default();
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
        let (cold_before, cold_prompt) = (cold_at_start, ggml_rs_wgpu::quant_moe::cached_experts());
        #[cfg(feature = "webgpu")]
        let scratch_prompt = ggml_rs_wgpu::scratch_made();
        let mut logits = Row::Logits(logits.unwrap());
        // (a greedy request's rows: the token alone where the model picks it on its GPU, megabytes of logits a check
        // not read back)
        let greedy = job.sampling.temperature <= 0.0 && job.images.is_empty();
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
        while generated.len() < job.max_tokens && self.kv.len < self.kv.max_len {
            if job.cancel.load(Ordering::Relaxed) {
                if let (Some(m), false) = (drafter, pending.is_empty()) { m.rollback(&mut self.kv, check_rows, check_rows - pending.len()); }
                return Ok(());
            }
            let clock = std::time::Instant::now();
            let mut next = logits.sample(&job.sampling, &mut rng);
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
                let checked = drafter.filter(|_| self.kv.len + 1 + drafts_most() <= self.kv.max_len).and_then(|m| {
                    let recent: Vec<u32> = self.covered[self.covered.len().saturating_sub(m.draft_window())..].iter().map(|&k| k as u32).chain([next]).collect();
                    // none the layer is sure enough of: a step of the token alone
                    let began = std::time::Instant::now();
                    let drafts = m.draft(&self.kv, &recent, drafts_most()).filter(|d| !d.is_empty());
                    t_draft += began.elapsed().as_secs_f64();
                    let drafts = drafts?;
                    let rows: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
                    let began = std::time::Instant::now();
                    let picked = if greedy { m.check_picks(&rows, &mut self.kv) } else { None };
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
                        logits = match if greedy { hybrid.step_pick(next, &mut self.kv) } else { None } {
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
            let now = ggml_rs_wgpu::quant_moe::cached_experts();
            if now.0 > cold_before.0 {
                eprintln!("  Qwen experts: the prompt read {} from the host's memory and brought {} to the card, the reply {} and {}", cold_prompt.0 - cold_before.0, cold_prompt.1 - cold_before.1, now.0 - cold_prompt.0, now.1 - cold_prompt.1);
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
        Ok(())
    }
}

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
    fn sample(&self, s: &crate::job::Sampling, rng: &mut u64) -> u32 {
        match self {
            Row::Logits(l) => sample(l.data(), s, rng),
            Row::Pick(token) => *token,
        }
    }
}

const DRAFTS: usize = 3;

/// [`DRAFTS`], or OAIY_DRAFTS's (1 to 7: a check is 8 rows at most). More than three loses on Flash-Next, whose check
/// is some 1.9 steps (its 66 tokens after 3,780: 0.84 s with three, 42 of 51 drafts taken; 0.96 with four, 45 of 59;
/// 0.94 with five; 0.99 with six).
fn drafts_most() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("OAIY_DRAFTS").ok().and_then(|v| v.parse().ok()).filter(|n| (1..=7).contains(n)).unwrap_or(DRAFTS))
}

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
fn checkpoint_positions(prompt: &[u32], im_start: Option<u32>) -> Vec<usize> {
    let boundaries: Vec<_> = prompt.iter().enumerate().filter_map(|(i, &t)|
        (i > 0 && Some(t) == im_start).then_some(i)).collect();
    let mut stops = vec![prompt.len().saturating_sub(1)];
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
mod tests;
