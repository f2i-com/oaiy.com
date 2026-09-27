//! Native Qwen dense hybrid inference, image embeddings and XML tool protocol.
use std::sync::{mpsc::Receiver, atomic::Ordering};
use dsv41::chat::{Encoded, Mode, Options, DSML};
use ggml_rs::Tensor;
use std::sync::Arc;
use crate::qwen_cache::{Snapshot, RecurrentSnapshot};
use llama_rs::{Model, MmProj, MmProjConfig, KvCache};
use nrob::json::Json;
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
    let (name, body) = block.trim().strip_prefix("<parameter=")?.split_once('>')?;
    tools.iter().filter_map(|t| t.get("function")).find(|f| f.get("name").and_then(Json::as_str) == Some(name))?;
    let mut body = body.trim_end();
    if let Some(inner) = body.strip_suffix("</function>") {
        body = inner.trim_end();
    }
    let (opens, closes) = (body.matches("<parameter=").count(), body.matches("</parameter>").count());
    let body = if closes == opens + 1 {
        body[..body.rfind("</parameter>")?].trim_end()
    } else if closes == opens {
        body
    } else {
        return None;
    };
    Some(format!("<function={name}>{body}\n</function>"))
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

pub struct QwenEngine {
    model: Model,
    projector: Option<MmProj>,
    kv: KvCache,
    covered: Vec<u64>,
    vision_cache: std::collections::VecDeque<(u64, Tensor)>,
    log: bool,
    checkpoints: Vec<(Vec<u64>, RecurrentSnapshot, bool)>,
    pub disk: Option<crate::disk::DiskCache<Arc<Snapshot>>>,
    /// Only enable when the disk namespace fingerprints the vision pipeline.
    pub image_disk_cache: bool,
}

impl QwenEngine {
    pub fn new(model: Model, projector: Option<MmProj>, max_seq: usize, log: bool) -> Self {
        let kv = model.new_kv_cache(max_seq);
        Self { model, projector, kv, covered: Vec::new(), vision_cache: Default::default(), log, checkpoints: Vec::new(), disk: None, image_disk_cache:false }
    }
    pub fn run(mut self, jobs: Receiver<Job>) {
        for job in jobs {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.generate(&job)));
            let error = match result { Ok(Ok(())) => None, Ok(Err(e)) => Some(e), Err(_) => Some("native Qwen inference failed".into()) };
            if let Some(e) = error { self.kv.reset(); self.covered.clear(); let _ = job.events.send(Event::Error(e)); }
            if job.forget {
                // Incognito: nothing of this request is reused or kept.
                self.kv.reset();
                self.covered.clear();
                self.checkpoints.clear();
                self.vision_cache.clear();
            }
        }
    }
    fn generate(&mut self, job: &Job) -> Result<(), String> {
        if job.prompt.is_empty() { return Err("empty Qwen prompt".into()); }
        let Model::Qwen35(model) = &self.model else { return Err("not a dense Qwen hybrid".into()); };
        let (positions, mut next_position) = positions(job)?;
        let mut keys: Vec<_> = job.prompt.iter().map(|&t| t as u64).collect();
        for image in &job.images { for (offset,k) in keys[image.start..image.start+image.prep.n_tokens()].iter_mut().enumerate() { *k = image.hash.rotate_left(17) ^ (offset as u64) ^ (1<<63); } }
        let stops = checkpoint_positions(&job.prompt, model.tokenizer.token_id("<|im_start|>"));
        let cache_clock = std::time::Instant::now();
        let common = self.covered.iter().zip(&keys).take_while(|(a,b)| a==b).count();
        // An attention suffix overwritten by a different branch cannot support
        // a recurrent-only checkpoint, even if a later request matches its keys.
        self.checkpoints.retain(|(saved,_,_)| saved.len() <= common && keys.starts_with(saved));
        let mut start = if common == self.covered.len() && common < keys.len() { common } else { 0 };
        let mut source = if start > 0 { "memory" } else { "none" };
        if let Some((saved, snap, _)) = self.checkpoints.iter().filter(|(saved, _, _)|
            saved.len() > start && saved.len() < keys.len() && keys.starts_with(saved)
        ).max_by_key(|(saved, _, _)| saved.len()) {
            snap.restore(&mut self.kv, common, &model.attention_layers, model.ssm_cfg, model.backend.as_ref())?;
            self.covered = saved.clone(); start = saved.len(); source = "checkpoint";
            self.checkpoints.retain(|(saved,_,_)| saved.len() <= start);
        }
        // Persist images only when the caller identifies both the projector and
        // preprocessing version. The prompt keys also include the image bytes.
        if job.images.is_empty() || self.image_disk_cache {
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
        while pos < keys.len() {
            if job.cancel.load(Ordering::Relaxed) { return Ok(()); }
            let end = (pos+PREFILL_CHUNK).min(keys.len()).min(stops.iter().copied().find(|&s| s > pos).unwrap_or(keys.len()));
            let mut embeds = model.embed_text(&job.prompt[pos..end]).to_host();
            let width = model.config.embedding_dim;
            for (at,t) in &soft {
                let from = pos.max(*at); let to = end.min(*at+t.dim(0));
                if from < to { embeds.data_mut()[(from-pos)*width..(to-pos)*width].copy_from_slice(&t.data()[(from-at)*width..(to-at)*width]); }
            }
            let embeds = model.backend.to_device(embeds);
            let out = model.forward_embeds_positions(&embeds, end-pos, &mut self.kv, (!job.images.is_empty()).then_some(&positions[pos..end])).map_err(|e|e.to_string())?;
            logits = Some(out.to_host());
            self.covered.extend_from_slice(&keys[pos..end]);
            let _ = job.events.send(Event::Progress { done: end-start, total });
            pos = end;
            if stops.contains(&pos) && !self.checkpoints.iter().any(|(saved, _, _)| saved == &keys[..pos]) {
                if self.log { eprintln!("  Qwen checkpoint: {pos} tokens; disk={}", self.disk.is_some()); }
                let base = stops.first() == Some(&pos);
                if job.images.is_empty() || self.image_disk_cache {
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
            }
        }
        let prefill_secs = clock.elapsed().as_secs_f64();
        let _ = job.events.send(Event::Prefilled { cached: start });
        let decode_clock = std::time::Instant::now();
        let tok = &model.tokenizer;
        let eos = tok.token_id("<|im_end|>").ok_or("Qwen tokenizer lacks im_end")?;
        let think_end = tok.token_id("</think>");
        let mut thinking = job.think_budget.is_some();
        let mut generated = Vec::new(); let mut think_used = 0usize;
        let mut stream = NativeStream::default();
        let mut rng = job.sampling.seed ^ 0x9E3779B97F4A7C15;
        let mut logits = logits.unwrap();
        let mut finish = Finish::Length;
        while generated.len() < job.max_tokens && self.kv.len < self.kv.max_len {
            if job.cancel.load(Ordering::Relaxed) { return Ok(()); }
            let mut next = sample(logits.data(), &job.sampling, &mut rng);
            if thinking && job.think_budget.is_some_and(|n|think_used >= n) { if let Some(end) = think_end { next = end; } }
            if next == eos || Some(next) == tok.eos() { finish = Finish::Stop; break; }
            generated.push(next);
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
            let embeds = model.embed_text(&[next]);
            logits = model.forward_embeds_positions(&embeds,1,&mut self.kv,(!job.images.is_empty()).then_some(&[[next_position;3]])).map_err(|e|e.to_string())?.to_host();
            next_position += 1;
            self.covered.push(next as u64);
        }
        let raw = tok.decode(&generated);
        // The client is told what the model wrote, so a rejected call can be read
        // and fixed; the server log never holds it.
        let text = stream.push(&raw, &job.tools, true)
            .map_err(|e| format!("tool_contract_error: {e}; no tool from this batch was executed. The model wrote: {}", call_excerpt(&raw)))?;
        if self.log { eprintln!("  Qwen: {} prompt tokens ({} cached) in {:.2}s; {} generated in {:.2}s", job.prompt.len(),start,prefill_secs,generated.len(),decode_clock.elapsed().as_secs_f64()); }
        if !text.is_empty() { let _ = job.events.send(Event::Text(text)); }
        let _ = job.events.send(Event::Done { finish, completion_tokens: generated.len() });
        Ok(())
    }
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
        assert_eq!(stops, [3, 6, 9]);
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
}
