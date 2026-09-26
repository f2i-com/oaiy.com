//! Work out what a model file or folder is, from its own headers.
//!
//! Nothing is loaded: a GGUF's metadata key/values and first tensor names, a
//! safetensors JSON header (with its `__metadata__`), or a folder's layout
//! (`model_index.json`, `config.json`, shard indexes) decide it. The result
//! names the role (an LLM, an image model, a video model, or a component one of
//! those needs) and a ready configuration entry, so the UI can add a picked file
//! to the right section -- and therefore the right endpoint -- in one step.

use crate::util::str_or;
use nrob::json::Json;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// Largest safetensors header read (real ones are a few MB).
const MAX_HEADER: u64 = 128 << 20;

#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    /// A language model nrob-server can serve (GGUF, EXL3, DeepSeek checkpoint).
    Llm,
    /// A text-to-image model; `architecture` is qwen-image or sdxl.
    Image { architecture: &'static str },
    /// A text/image-to-video model of an LTX family.
    Video { family: &'static str },
    /// A part another model needs: `vision_projector`, `adapter`, `text_encoder`,
    /// `video_text_encoder`, `vae`, `tokenizer`, `clip_tokenizer`, `image_base`.
    Component { kind: &'static str },
}

#[derive(Clone, Debug)]
pub struct Detected {
    pub role: Role,
    pub format: &'static str,
    /// A human summary ("Qwen3.5 GGUF (qwen35)").
    pub summary: String,
    /// Configuration fields the file fills (path keys and defaults).
    pub fields: Vec<(String, Json)>,
    /// Fields the entry still needs before it can run.
    pub missing: Vec<String>,
}

impl Detected {
    pub fn kind(&self) -> &'static str {
        match self.role {
            Role::Llm => "llm",
            Role::Image { .. } => "image",
            Role::Video { .. } => "video",
            Role::Component { kind } => kind,
        }
    }

    pub fn to_json(&self) -> Json {
        Json::obj([
            ("kind", Json::str(self.kind())),
            ("section", Json::str(match self.role {
                Role::Llm => "llm",
                Role::Image { .. } => "image",
                Role::Video { .. } => "video",
                Role::Component { .. } => "component",
            })),
            ("format", Json::str(self.format)),
            ("summary", Json::str(&self.summary)),
            ("fields", Json::Obj(self.fields.clone())),
            ("missing", Json::Arr(self.missing.iter().map(Json::str).collect())),
        ])
    }
}

fn detected(role: Role, format: &'static str, summary: String, fields: Vec<(&str, Json)>) -> Detected {
    Detected { role, format, summary, fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), missing: Vec::new() }
}

/// A path as configuration text. On Windows a path picked with `/` keeps `/`
/// throughout, rather than gaining `\` where folders were joined onto it.
pub(crate) fn path_json(p: &Path) -> Json {
    let s = p.to_string_lossy();
    if cfg!(windows) && s.contains('/') {
        return Json::str(s.replace('\\', "/"));
    }
    Json::str(s)
}

/// What `path` is. Errors name why a file is not recognised.
pub fn detect(path: &Path) -> Result<Detected, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_dir() {
        return directory(path);
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "gguf" => gguf_file(path),
        "safetensors" => safetensors_file(path),
        "json" if name == "tokenizer.json" => tokenizer_file(path),
        "json" if name.ends_with(".safetensors.index.json") || name == "model_index.json" || name == "config.json" => {
            path.parent().map_or(Err("no folder".into()), directory)
        }
        "exe" | "" if name.contains("ffmpeg") => Ok(detected(Role::Component { kind: "ffmpeg" }, "program", "FFmpeg (writes the video container)".into(), vec![("ffmpeg", path_json(path))])),
        "bin" | "pt" | "pth" | "ckpt" => Err(format!(
            "{name}: pickled PyTorch weights are not supported (they can run code when loaded); use the .safetensors or .gguf release"
        )),
        _ => Err(format!("{name}: not a model format nrob reads (GGUF, safetensors, a tokenizer.json, or a model folder)")),
    }
}

// ---------------------------------------------------------------- GGUF

struct Gguf {
    kv: Vec<(String, Json)>,
    tensors: Vec<String>,
}

/// The metadata key/values (strings and numbers; arrays summarised as their
/// length) and the first `max_tensors` tensor names.
fn read_gguf(path: &Path, max_tensors: usize) -> Result<Gguf, String> {
    let bad = |m: &str| format!("{}: {m}", path.display());
    let mut r = BufReader::with_capacity(1 << 20, File::open(path).map_err(|e| bad(&e.to_string()))?);
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    let mut u32_ = |r: &mut BufReader<File>| -> Result<u32, String> { r.read_exact(&mut b4).map_err(|_| bad("truncated"))?; Ok(u32::from_le_bytes(b4)) };
    let mut u64_ = |r: &mut BufReader<File>| -> Result<u64, String> { r.read_exact(&mut b8).map_err(|_| bad("truncated"))?; Ok(u64::from_le_bytes(b8)) };
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|_| bad("truncated"))?;
    if &magic != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = u32_(&mut r)?;
    if !(2..=3).contains(&version) {
        return Err(bad(&format!("GGUF version {version} is not supported")));
    }
    let (n_tensors, n_kv) = (u64_(&mut r)?, u64_(&mut r)?);
    if n_kv > 1 << 20 || n_tensors > 1 << 24 {
        return Err(bad("implausible GGUF header"));
    }
    fn string(r: &mut BufReader<File>, keep: bool) -> Result<String, String> {
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8).map_err(|_| "truncated string".to_string())?;
        let n = u64::from_le_bytes(b8);
        if n > 1 << 24 {
            return Err("implausible string".into());
        }
        if !keep {
            r.seek_relative(n as i64).map_err(|e| e.to_string())?;
            return Ok(String::new());
        }
        let mut s = vec![0; n as usize];
        r.read_exact(&mut s).map_err(|_| "truncated string".to_string())?;
        Ok(String::from_utf8_lossy(&s).into_owned())
    }
    fn size(t: u32) -> Option<i64> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }
    fn scalar(r: &mut BufReader<File>, t: u32) -> Result<Json, String> {
        let n = size(t).ok_or("unknown GGUF value type")? as usize;
        let mut b = [0u8; 8];
        r.read_exact(&mut b[..n]).map_err(|_| "truncated value".to_string())?;
        Ok(match t {
            0 => Json::Int(b[0] as i64),
            1 => Json::Int(b[0] as i8 as i64),
            2 => Json::Int(u16::from_le_bytes([b[0], b[1]]) as i64),
            3 => Json::Int(i16::from_le_bytes([b[0], b[1]]) as i64),
            4 => Json::Int(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
            5 => Json::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
            6 => Json::Num(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
            7 => Json::Bool(b[0] != 0),
            10 => Json::Int(u64::from_le_bytes(b) as i64),
            11 => Json::Int(i64::from_le_bytes(b)),
            _ => Json::Num(f64::from_le_bytes(b)),
        })
    }
    let mut kv = Vec::new();
    for _ in 0..n_kv {
        let key = string(&mut r, true).map_err(|e| bad(&e))?;
        let t = u32_(&mut r)?;
        let value = match t {
            8 => Json::str(string(&mut r, true).map_err(|e| bad(&e))?),
            9 => {
                let item = u32_(&mut r)?;
                let n = u64_(&mut r)?;
                if item == 8 {
                    for _ in 0..n {
                        string(&mut r, false).map_err(|e| bad(&e))?;
                    }
                } else {
                    let s = size(item).ok_or_else(|| bad("nested GGUF arrays are not expected in metadata"))?;
                    let skip = i64::try_from(n).ok().and_then(|n| n.checked_mul(s)).ok_or_else(|| bad("implausible GGUF array"))?;
                    r.seek_relative(skip).map_err(|e| bad(&e.to_string()))?;
                }
                Json::obj([("array_len", Json::Int(n as i64))])
            }
            t => scalar(&mut r, t).map_err(|e| bad(&e))?,
        };
        kv.push((key, value));
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors.min(max_tensors as u64) {
        tensors.push(string(&mut r, true).map_err(|e| bad(&e))?);
        let dims = u32_(&mut r)?;
        r.seek_relative(8 * dims as i64 + 12).map_err(|e| bad(&e.to_string()))?;
    }
    Ok(Gguf { kv, tensors })
}

fn gguf_file(path: &Path) -> Result<Detected, String> {
    let g = read_gguf(path, 64)?;
    let get = |k: &str| g.kv.iter().find(|(key, _)| key == k).and_then(|(_, v)| v.as_str()).unwrap_or("").to_string();
    let arch = get("general.architecture");
    let name = get("general.name");
    let label = if name.is_empty() { stem(path) } else { name };
    if arch == "clip" || get("general.type") == "mmproj" {
        return Ok(detected(Role::Component { kind: "vision_projector" }, "gguf",
            format!("Vision projector for an LLM ({label})"), vec![("vision_projector", path_json(path))]));
    }
    let image_tensors = g.tensors.iter().any(|t| t.contains("img_in.") || t.contains("transformer_blocks.0.img_mlp"));
    if arch.starts_with("qwen_image") || image_tensors {
        let mut d = detected(Role::Image { architecture: "qwen-image" }, "gguf",
            format!("Qwen Image transformer, quantized GGUF ({arch})"),
            vec![("architecture", Json::str("qwen-image")), ("transformer", path_json(path)), ("weights", Json::str("gguf"))]);
        d.missing.push("base".into());
        return Ok(d);
    }
    if arch.is_empty() {
        return Err(format!("{}: GGUF without general.architecture", path.display()));
    }
    if arch == "music3-lm" {
        return Ok(detected(Role::Component { kind: "music_lm" }, "gguf",
            format!("MiniMax Music 3 language model, {} ({label})", get("music3.quant")), vec![("language_model", path_json(path))]));
    }
    Ok(detected(Role::Llm, "gguf", format!("{label} — GGUF LLM ({arch})"), vec![("path", path_json(path))]))
}

// ---------------------------------------------------------- safetensors

/// A safetensors header: tensor names with their shapes, and `__metadata__`.
struct Header {
    tensors: Vec<(String, Vec<i64>)>,
    metadata: Json,
}

fn read_header(path: &Path) -> Result<Header, String> {
    let bad = |m: &str| format!("{}: {m}", path.display());
    let mut f = File::open(path).map_err(|e| bad(&e.to_string()))?;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).map_err(|_| bad("truncated"))?;
    let n = u64::from_le_bytes(b8);
    if n == 0 || n > MAX_HEADER {
        return Err(bad("not a safetensors file (bad header length)"));
    }
    let mut header = vec![0; n as usize];
    f.read_exact(&mut header).map_err(|_| bad("truncated header"))?;
    let j = Json::parse(&header).map_err(|e| bad(&format!("header: {e}")))?;
    let mut tensors = Vec::new();
    let mut metadata = Json::Obj(Vec::new());
    for (k, v) in j.members() {
        if k == "__metadata__" {
            metadata = v.clone();
        } else {
            let shape = v.get("shape").and_then(Json::as_array).map(|s| s.iter().filter_map(Json::as_i64).collect()).unwrap_or_default();
            tensors.push((k.to_string(), shape));
        }
    }
    Ok(Header { tensors, metadata })
}

fn has(h: &Header, needle: &str) -> bool {
    h.tensors.iter().any(|(k, _)| k.contains(needle))
}

fn shape_of<'a>(h: &'a Header, suffix: &str) -> Option<&'a [i64]> {
    h.tensors.iter().find(|(k, _)| k.ends_with(suffix)).map(|(_, s)| s.as_slice())
}

/// LTX 2.x metadata: `model_version` and the embedded architecture config.
fn ltx_family(h: &Header) -> Option<&'static str> {
    let version = h.metadata.get("model_version").and_then(Json::as_str)?;
    if version.starts_with("2.5.") {
        Some("ltx-2.5")
    } else if version.starts_with("2.3.") {
        Some("ltx-2.3")
    } else {
        None
    }
}

fn safetensors_file(path: &Path) -> Result<Detected, String> {
    let h = read_header(path)?;
    // A Qwen3-TTS weight file stands for its folder (config, tokenizer, codec).
    if has(&h, "talker.codec_head.") {
        if let Some(dir) = path.parent() {
            return directory(dir);
        }
    }
    classify_header(&h, path, "safetensors")
}

fn classify_header(h: &Header, path: &Path, format: &'static str) -> Result<Detected, String> {
    let p = path_json(path);
    let label = stem(path);
    let spec = h.metadata.get("modelspec.architecture").and_then(Json::as_str).unwrap_or("");
    let config = h.metadata.get("config").and_then(Json::as_str).unwrap_or("");
    if has(h, ".lora_A.") || has(h, ".lora_down.") || h.metadata.get("lora_adapter_metadata").is_some() {
        return Ok(detected(Role::Component { kind: "adapter" }, format,
            format!("LoRA adapter ({label}); attach it to a Qwen Image model as its turbo adapter"), vec![("adapter", p)]));
    }
    if spec.contains("stable-diffusion-xl") || (has(h, "conditioner.embedders.1.model.") && has(h, "model.diffusion_model.input_blocks.")) {
        let mut d = detected(Role::Image { architecture: "sdxl" }, format, format!("SDXL checkpoint ({label})"),
            vec![("architecture", Json::str("sdxl")), ("checkpoint", p), ("steps", Json::Int(20)), ("cfg", Json::Num(4.0))]);
        d.missing.push("tokenizer".into());
        return Ok(d);
    }
    if let Some(family) = ltx_family(h) {
        // Parts of an LTX release that the worker has no use for.
        if config.contains("\"audio_vae\"") || has(h, "audio_vae.decoder.") {
            // The vocoder must come with its bandwidth extension (48 kHz output).
            if has(h, "vocoder.bwe_generator.") {
                return Ok(detected(Role::Component { kind: "audio_vae" }, format,
                    format!("LTX {} audio VAE and vocoder ({label}); gives LTX clips a soundtrack", &family[4..]), vec![("audio_vae", p)]));
            }
            return Err(format!("{label}: an LTX audio VAE without the bandwidth-extension vocoder nrob needs"));
        }
        if config.contains("CausalDiffusionVAE") || has(h, "decoder.diff_blocks.") {
            return Err(format!("{label}: LTX {} diffusion-decoder VAE, which nrob does not run; use the conv VAE (ltx-2.5-video-vae-conv) instead", &family[4..]));
        }
        if config.contains("upsampler") || config.contains("upscaler") {
            return Err(format!("{label}: LTX latent upscaler; nrob renders at the requested size in one pass"));
        }
        let transformer = config.contains("\"transformer\"") || has(h, "transformer_blocks.");
        let vae = config.contains("\"vae\"") || has(h, "decoder.conv_in.");
        if transformer {
            // LTX 2.3 releases (and Sulphur) carry the VAE in the same file.
            // Sulphur runs the LTX 2.3 path; its own label keeps catalogs readable.
            let family = if family == "ltx-2.3" && label.to_ascii_lowercase().contains("sulphur") { "sulphur-2" } else { family };
            let mut fields = vec![("family", Json::str(family)), ("transformer", p.clone())];
            if vae {
                fields.push(("vae", p.clone()));
            }
            // LTX 2.3 releases (and Sulphur) also carry the audio VAE and vocoder.
            if has(h, "audio_vae.decoder.") && has(h, "vocoder.bwe_generator.") {
                fields.push(("audio_vae", p));
            }
            let title = if family == "sulphur-2" { "Sulphur 2 (LTX 2.3)".to_string() } else { format!("LTX {}", &family[4..]) };
            let mut d = detected(Role::Video { family }, format, format!("{title} video transformer ({label})"), fields);
            if !vae {
                d.missing.push("vae".into());
            }
            d.missing.push("text_encoder".into());
            if family != "ltx-2.5" {
                d.missing.push("tokenizer".into());
            }
            return Ok(d);
        }
        if vae {
            return Ok(detected(Role::Component { kind: "vae" }, format, format!("LTX {} video VAE ({label})", &family[4..]), vec![("vae", p)]));
        }
    }
    if has(h, "img_in.") && has(h, "transformer_blocks.") {
        let mut d = detected(Role::Image { architecture: "qwen-image" }, format, format!("Qwen Image transformer ({label})"),
            vec![("architecture", Json::str("qwen-image")), ("safetensors_transformer", p), ("weights", Json::str("safetensors"))]);
        d.missing.push("base".into());
        return Ok(d);
    }
    // Gemma 4 for LTX 2.5: its config in the metadata, or (in re-quantized
    // copies that drop it) the embedded tokenizer and LTX projection.
    if h.metadata.get("gemma_config").is_some() || (has(h, "tokenizer_json") && has(h, "text_embedding_projection.video_aggregate_embed.")) {
        return Ok(detected(Role::Component { kind: "video_text_encoder" }, format,
            format!("Gemma text encoder for LTX 2.5 ({label})"), vec![("text_encoder", p)]));
    }
    if has(h, "layers.0.self_attn.q_proj") || has(h, "layers.0.mlp.down_proj") {
        // Text encoders are decoder LLMs; the vocabulary tells which one.
        let vocab = shape_of(h, "embed_tokens.weight").and_then(|s| s.first().copied()).unwrap_or(0);
        if vocab >= 256_000 {
            return Ok(detected(Role::Component { kind: "video_text_encoder" }, format,
                format!("Gemma 3 text encoder for LTX 2.3 / Sulphur ({label})"), vec![("text_encoder", p)]));
        }
        if (150_000..160_000).contains(&vocab) {
            return Ok(detected(Role::Component { kind: "text_encoder" }, format,
                format!("Qwen3-VL text encoder for Qwen Image ({label})"), vec![("text_encoder", p)]));
        }
        return Err(format!("{label}: a single-file decoder LLM; serve language models as GGUF (or an EXL3/DeepSeek folder)"));
    }
    if has(h, "decoder.conv_in.") && has(h, "encoder.conv_in.") {
        return Ok(detected(Role::Component { kind: "vae" }, format, format!("VAE ({label})"), vec![("vae", p)]));
    }
    Err(format!("{label}: safetensors with {} tensors, but not a layout nrob recognises", h.tensors.len()))
}

fn tokenizer_file(path: &Path) -> Result<Detected, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let mut head = vec![0; meta.len().min(4 << 20) as usize];
    File::open(path).and_then(|mut f| f.read_exact(&mut head)).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&head);
    if text.contains("<|startoftext|>") {
        return Ok(detected(Role::Component { kind: "clip_tokenizer" }, "tokenizer", "CLIP tokenizer (for SDXL)".into(), vec![("tokenizer", path_json(path))]));
    }
    if text.contains("<start_of_turn>") || text.contains("<start_of_image>") {
        return Ok(detected(Role::Component { kind: "tokenizer" }, "tokenizer", "Gemma 3 tokenizer (for LTX 2.3 / Sulphur)".into(), vec![("tokenizer", path_json(path))]));
    }
    Err("tokenizer.json: neither the CLIP nor the Gemma 3 tokenizer".into())
}

// --------------------------------------------------------------- folders

fn directory(dir: &Path) -> Result<Detected, String> {
    let label = stem(dir);
    let read = |name: &str| std::fs::read(dir.join(name)).ok().and_then(|b| Json::parse(&b).ok());
    if let Some(index) = read("model_index.json") {
        let class = str_or(&index, "_class_name", "");
        if class.starts_with("QwenImage") {
            let transformer = dir.join("transformer");
            if !(transformer.is_dir() && has_ext(&transformer, "safetensors")) {
                // Encoder, VAE and processor without transformer weights: the base
                // a GGUF or single-file transformer runs on.
                return Ok(detected(Role::Component { kind: "image_base" }, "diffusers",
                    format!("Qwen Image base folder ({class}) without transformer weights"), vec![("base", path_json(dir))]));
            }
            let fields = vec![
                ("architecture", Json::str("qwen-image")),
                ("base", path_json(dir)),
                ("safetensors_transformer", path_json(&transformer)),
                ("weights", Json::str("safetensors")),
            ];
            return Ok(detected(Role::Image { architecture: "qwen-image" }, "diffusers", format!("Qwen Image pipeline folder ({class})"), fields));
        }
        return Err(format!("{label}: diffusers pipeline {class:?} is not one nrob runs"));
    }
    // An LTX 2.5 release laid out as ComfyUI folders. Each part is the first
    // file in its folder that is that part (a folder may also hold an audio
    // VAE, upscalers or GGUF variants nrob does not use).
    if dir.join("diffusion_models").is_dir() && dir.join("text_encoders").is_dir() {
        let first = |sub: &str| first_with_ext(&dir.join(sub), "safetensors");
        let part = |sub: &str, kind: &'static str| {
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join(sub)).ok()?.flatten().map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))).collect();
            files.sort();
            files.into_iter().find(|p| safetensors_file(p).is_ok_and(|d| d.role == Role::Component { kind }))
        };
        if let Some(t) = first("diffusion_models") {
            let mut d = safetensors_file(&t)?;
            if let Role::Video { .. } = d.role {
                for (sub, key, kind) in [("text_encoders", "text_encoder", "video_text_encoder"), ("vae", "vae", "vae"), ("vae", "audio_vae", "audio_vae")] {
                    if let Some(p) = part(sub, kind) {
                        d.fields.retain(|(k, _)| k != key);
                        d.fields.push((key.into(), path_json(&p)));
                        d.missing.retain(|m| m != key);
                    }
                }
                d.summary = format!("{} (folder)", d.summary);
                return Ok(d);
            }
        }
    }
    if let Some(config) = read("config.json") {
        let quant = config.get("quantization_config").map(|q| str_or(q, "quant_method", "")).unwrap_or("");
        let arch = config.get("architectures").and_then(|a| a.at(0)).and_then(Json::as_str).unwrap_or("").to_string();
        let model_type = str_or(&config, "model_type", "");
        if quant == "exl3" || arch.starts_with("Deepseek") || model_type.starts_with("deepseek") {
            return Ok(detected(Role::Llm, "checkpoint", format!("{label} — {} checkpoint folder ({arch})", if quant == "exl3" { "EXL3" } else { "safetensors" }),
                vec![("path", path_json(dir))]));
        }
        // MiniMax Music 3: the whole pipeline in one folder.
        if model_type == "minimax_music3" {
            for part in ["language_model", "rvq_depth_decoder", "condition_encoder", "transformer", "vocoder", "tokenizer"] {
                if !dir.join(part).is_dir() {
                    return Err(format!("{label}: a MiniMax Music 3 folder needs its {part}/ folder"));
                }
            }
            return Ok(detected(Role::Component { kind: "music_model" }, "safetensors", format!("MiniMax Music 3: songs from lyrics and a description ({label})"), vec![("path", path_json(dir))]));
        }
        // Qwen3-TTS: VoiceDesign speaks in voices described in words; Base
        // speaks in saved voices. One speech model pairs the two.
        if model_type == "qwen3_tts" {
            let (kind, field, what) = match str_or(&config, "tts_model_type", "") {
                "voice_design" => ("speech_design", "design", "VoiceDesign: speaks in any voice you describe"),
                "base" => ("speech_base", "base", "Base: speaks in saved voices"),
                other => return Err(format!("{label}: Qwen3-TTS {other} models are not supported; use VoiceDesign or Base")),
            };
            if !dir.join("speech_tokenizer").join("model.safetensors").is_file() {
                return Err(format!("{label}: a Qwen3-TTS folder needs its speech_tokenizer/ folder"));
            }
            return Ok(detected(Role::Component { kind }, "safetensors", format!("Qwen3-TTS {what} ({label})"), vec![(field, path_json(dir))]));
        }
        if model_type == "qwen3_vl" && has_ext(dir, "safetensors") {
            return Ok(detected(Role::Component { kind: "text_encoder" }, "safetensors", format!("Qwen3-VL text encoder folder ({label})"), vec![("text_encoder", path_json(dir))]));
        }
    }
    if let Some(g) = first_with_ext(dir, "gguf") {
        // A folder of GGUF shards (or one GGUF and its projector).
        let mut d = gguf_file(&g)?;
        if d.role == Role::Llm {
            d.fields = vec![("path".into(), path_json(dir))];
            if let Some(proj) = std::fs::read_dir(dir).ok().into_iter().flatten().flatten().map(|e| e.path())
                .find(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) && p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj")))
            {
                d.fields.push(("vision_projector".into(), path_json(&proj)));
            }
        }
        return Ok(d);
    }
    if let Some(s) = first_with_ext(dir, "safetensors") {
        return safetensors_file(&s);
    }
    Err(format!("{label}: no model files nrob recognises in this folder"))
}

fn has_ext(dir: &Path, ext: &str) -> bool {
    first_with_ext(dir, ext).is_some()
}

/// The first file with `ext` (sorted, so shard 1 of N comes first).
fn first_with_ext(dir: &Path, ext: &str) -> Option<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext))
            && !p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj")))
        .collect();
    files.sort();
    files.into_iter().next()
}

fn stem(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    match name.rsplit_once('.') {
        Some((s, ext)) if !s.is_empty() && ext.len() <= 12 && !path.is_dir() => s.to_string(),
        _ => name,
    }
}

/// A configuration name from a file name: lowercase, `-` between words, short.
pub fn model_name(path: &Path) -> String {
    let mut out = String::new();
    for c in stem(path).chars() {
        if c.is_ascii_alphanumeric() || c == '.' {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed: String = out.trim_matches('-').chars().take(48).collect();
    if trimmed.is_empty() { "model".into() } else { trimmed.trim_end_matches('-').to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tmp(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("nrob-studio-detect-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn safetensors(path: &Path, header: &str) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    fn gguf(path: &Path, kv: &[(&str, &str)], tensors: &[&str]) {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((kv.len() as u64 + 1).to_le_bytes());
        let s = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        };
        for (k, v) in kv {
            s(&mut b, k);
            b.extend(8u32.to_le_bytes());
            s(&mut b, v);
        }
        // An array of strings, as a tokenizer vocabulary is stored: skipped.
        s(&mut b, "tokenizer.ggml.tokens");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        s(&mut b, "a");
        s(&mut b, "b");
        for t in tensors {
            s(&mut b, t);
            b.extend(2u32.to_le_bytes());
            b.extend([0u8; 16 + 12]);
        }
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn gguf_llms_projectors_and_image_transformers_are_told_apart() {
        let d = tmp("gguf");
        let llm = d.0.join("Qwen3-8B-Q4_K_M.gguf");
        gguf(&llm, &[("general.architecture", "qwen3"), ("general.name", "Qwen3 8B")], &["token_embd.weight"]);
        let r = detect(&llm).unwrap();
        assert_eq!(r.role, Role::Llm);
        assert!(r.summary.contains("qwen3"));
        let proj = d.0.join("mmproj-F16.gguf");
        gguf(&proj, &[("general.architecture", "clip"), ("general.type", "mmproj")], &[]);
        assert_eq!(detect(&proj).unwrap().kind(), "vision_projector");
        let image = d.0.join("qwen-image-Q4.gguf");
        gguf(&image, &[("general.architecture", "qwen_image21")], &["img_in.weight"]);
        let r = detect(&image).unwrap();
        assert_eq!(r.role, Role::Image { architecture: "qwen-image" });
        assert_eq!(r.missing, vec!["base".to_string()]);
        // The folder holding the LLM (and its projector) is the LLM, projector attached.
        std::fs::remove_file(&image).unwrap();
        let folder = detect(&d.0).unwrap();
        assert_eq!(folder.role, Role::Llm);
        assert!(folder.fields.iter().any(|(k, _)| k == "vision_projector"));
        assert_eq!(model_name(&llm), "qwen3-8b-q4-k-m");
    }

    #[test]
    fn safetensors_headers_decide_the_role() {
        let d = tmp("st");
        let cases = [
            ("sdxl.safetensors", r#"{"__metadata__":{"modelspec.architecture":"stable-diffusion-xl-v1-base"},"model.diffusion_model.input_blocks.0.0.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#, "image"),
            ("ltx.safetensors", r#"{"__metadata__":{"model_version":"2.3.0","config":"{\"transformer\":{},\"vae\":{}}"},"transformer_blocks.0.x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#, "video"),
            ("vae.safetensors", r#"{"__metadata__":{"model_version":"2.5.0","config":"{\"vae\":{}}"},"decoder.conv_in.conv.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#, "vae"),
            ("lora.safetensors", r#"{"transformer.img_in.lora_A.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#, "adapter"),
            ("qwen.safetensors", r#"{"img_in.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"transformer_blocks.0.attn.to_q.weight":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#, "image"),
            ("te.safetensors", r#"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[151936,4096],"data_offsets":[0,2]},"model.layers.0.self_attn.q_proj.weight":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#, "text_encoder"),
            ("gemma.safetensors", r#"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[262208,3840],"data_offsets":[0,2]},"model.layers.0.mlp.down_proj.weight":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#, "video_text_encoder"),
        ];
        for (name, header, kind) in cases {
            let p = d.0.join(name);
            safetensors(&p, header);
            assert_eq!(detect(&p).unwrap().kind(), kind, "{name}");
        }
        let ltx = detect(&d.0.join("ltx.safetensors")).unwrap();
        assert_eq!(ltx.role, Role::Video { family: "ltx-2.3" });
        assert!(ltx.fields.iter().any(|(k, _)| k == "vae"));
        assert!(ltx.missing.contains(&"tokenizer".to_string()));
        assert!(!ltx.fields.iter().any(|(k, _)| k == "audio_vae"), "no audio parts, no audio VAE");
        // An LTX 2.3 release with its audio VAE and vocoder is its own audio VAE too.
        let av = d.0.join("ltx-av.safetensors");
        safetensors(&av, r#"{"__metadata__":{"model_version":"2.3.0","config":"{\"transformer\":{},\"vae\":{}}"},"transformer_blocks.0.x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"audio_vae.decoder.conv_in.conv.bias":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]},"vocoder.bwe_generator.conv_pre.weight":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}"#);
        let r = detect(&av).unwrap();
        assert!(r.fields.iter().any(|(k, v)| k == "audio_vae" && v.as_str().is_some_and(|s| s.ends_with("ltx-av.safetensors"))));
        let bin = d.0.join("model.bin");
        std::fs::write(&bin, b"x").unwrap();
        assert!(detect(&bin).unwrap_err().contains("pickled"));
        std::fs::write(d.0.join("junk.safetensors"), b"\xff\xff\xff\xff\xff\xff\xff\xff").unwrap();
        assert!(detect(&d.0.join("junk.safetensors")).is_err());
    }

    #[test]
    fn folders_and_tokenizers_are_recognised() {
        let d = tmp("dirs");
        let pipeline = d.0.join("Qwen-Image");
        std::fs::create_dir_all(pipeline.join("transformer")).unwrap();
        std::fs::write(pipeline.join("model_index.json"), r#"{"_class_name":"QwenImage21Pipeline"}"#).unwrap();
        assert_eq!(detect(&pipeline).unwrap().kind(), "image_base");
        safetensors(&pipeline.join("transformer").join("a.safetensors"), "{}");
        let r = detect(&pipeline).unwrap();
        assert_eq!(r.role, Role::Image { architecture: "qwen-image" });
        assert!(r.fields.iter().any(|(k, _)| k == "safetensors_transformer"));
        let exl3 = d.0.join("orca");
        std::fs::create_dir_all(&exl3).unwrap();
        std::fs::write(exl3.join("config.json"), r#"{"quantization_config":{"quant_method":"exl3"},"architectures":["Qwen3ForCausalLM"]}"#).unwrap();
        assert_eq!(detect(&exl3).unwrap().role, Role::Llm);
        let tok = d.0.join("tokenizer.json");
        std::fs::write(&tok, r#"{"added_tokens":[{"content":"<|startoftext|>"}]}"#).unwrap();
        assert_eq!(detect(&tok).unwrap().kind(), "clip_tokenizer");
    }

    #[test]
    fn a_quantized_ltx_release_folder_picks_the_parts_nrob_runs() {
        let d = tmp("ltx-folder");
        let root = d.0.join("LTX-2.5-finetune");
        for sub in ["diffusion_models", "text_encoders", "vae"] {
            std::fs::create_dir_all(root.join(sub)).unwrap();
        }
        let t = |name: &str, dtype: &str| format!(r#""{name}":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,1]}}"#);
        // A bare-named fp8 transformer with its scales.
        safetensors(&root.join("diffusion_models").join("ltx25-fp8_scaled.safetensors"), &format!(
            r#"{{"__metadata__":{{"model_version":"2.5.0","config":"{{\"transformer\":{{}}}}"}},{},{}}}"#,
            t("transformer_blocks.0.attn1.to_q.weight", "F8_E4M3"), t("transformer_blocks.0.attn1.to_q.weight_scale", "F32")));
        // An int8 Gemma 4 without its gemma_config, known by tokenizer and projection.
        let gemma = root.join("text_encoders").join("gemma4-int8.safetensors");
        safetensors(&gemma, &format!("{{{},{},{}}}", t("tokenizer_json", "U8"), t("text_embedding_projection.video_aggregate_embed.weight", "I8"), t("model.layers.0.mlp.down_proj.weight", "I8")));
        assert_eq!(detect(&gemma).unwrap().kind(), "video_text_encoder");
        // Sorted first in vae/, but neither is a VAE nrob can use.
        let audio = root.join("vae").join("a_audio_vae.safetensors");
        safetensors(&audio, &format!(r#"{{"__metadata__":{{"model_version":"2.5.0","config":"{{\"audio_vae\":{{}}}}"}},{},{}}}"#,
            t("audio_vae.decoder.conv_in.conv.bias", "BF16"), t("vocoder.bwe_generator.conv_pre.weight", "BF16")));
        assert_eq!(detect(&audio).unwrap().kind(), "audio_vae");
        let diffusion = root.join("vae").join("b_video_vae.safetensors");
        safetensors(&diffusion, &format!(r#"{{"__metadata__":{{"model_version":"2.5.0","config":"{{\"vae\":{{\"_class_name\":\"CausalDiffusionVAE\"}}}}"}},{},{}}}"#, t("decoder.conv_in.weight", "BF16"), t("decoder.diff_blocks.0.x", "BF16")));
        assert!(detect(&diffusion).unwrap_err().contains("diffusion-decoder"));
        let r = detect(&root).unwrap();
        assert_eq!(r.role, Role::Video { family: "ltx-2.5" });
        let field = |k: &str| r.fields.iter().find(|(f, _)| f == k).map(|(_, v)| v.as_str().unwrap_or("").to_string());
        assert!(field("text_encoder").unwrap().ends_with("gemma4-int8.safetensors"));
        assert!(field("audio_vae").unwrap().ends_with("a_audio_vae.safetensors"));
        assert_eq!(field("vae"), None, "no usable video VAE in the folder");
        assert_eq!(r.missing, vec!["vae".to_string()]);
    }
}
