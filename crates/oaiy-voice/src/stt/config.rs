//! What a Parakeet checkpoint is: its front end, encoder shape and decoding
//! head, read from a `.nemo`'s `model_config.yaml` or a Hugging Face
//! `config.json` (plus `processor_config.json`). Sizes that the weights
//! themselves carry (widths, layer counts, vocabulary) come from the tensors;
//! the configuration supplies only what they cannot say.

use std::collections::HashMap;

use oaiy_engine::json::Json;

use super::mel::MelConfig;
use super::{bad, Result};

/// How the joint's output is decoded.
#[derive(Clone, Debug, PartialEq)]
pub enum Head {
    /// Plain RNN-T: blank or one token per step, at most `max_symbols` a frame.
    Rnnt,
    /// Token-and-duration transducer: the last `durations.len()` joint outputs
    /// score how many frames to advance.
    Tdt { durations: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub mel: MelConfig,
    pub head: Head,
    pub max_symbols: usize,
    /// Multiply the subsampled input by sqrt(d_model) (`xscaling`).
    pub xscaling: bool,
    /// Limited attention context `[left, right]` in encoder frames; -1 is unlimited.
    pub att_context: [i64; 2],
}

impl ModelConfig {
    /// The defaults of the Parakeet 0.6B models, for what a config leaves out.
    pub fn parakeet() -> Self {
        Self { mel: MelConfig::parakeet(128), head: Head::Tdt { durations: vec![0, 1, 2, 3, 4] }, max_symbols: 10, xscaling: false, att_context: [-1, -1] }
    }

    /// From a `.nemo`'s `model_config.yaml`.
    pub fn from_nemo_yaml(text: &str) -> Result<Self> {
        let y = Yaml::parse(text);
        let mut cfg = Self::parakeet();
        let num = |k: &str| y.scalar(k).and_then(|v| v.parse::<f64>().ok());
        let sr = num("preprocessor.sample_rate").or(num("sample_rate")).unwrap_or(16_000.0) as usize;
        let win = num("preprocessor.window_size").unwrap_or(0.025);
        let hop = num("preprocessor.window_stride").unwrap_or(0.01);
        let win_length = (win * sr as f64).round() as usize;
        let n_fft = num("preprocessor.n_fft").map(|v| v as usize).unwrap_or_else(|| win_length.next_power_of_two());
        let preemph = match y.scalar("preprocessor.preemph") {
            None => Some(0.97),
            Some("null") | Some("None") => None,
            Some(v) => Some(v.parse::<f32>().map_err(|_| bad(format!("preprocessor.preemph: {v:?}")))?),
        };
        cfg.mel = MelConfig {
            sample_rate: sr,
            n_fft,
            win_length,
            hop_length: (hop * sr as f64).round() as usize,
            n_mels: num("preprocessor.features").map(|v| v as usize).unwrap_or(128),
            preemph,
            log_guard: num("preprocessor.log_zero_guard_value").unwrap_or(2f64.powi(-24)),
        };
        check_front_end(&y)?;
        let durations = y.list("model_defaults.tdt_durations").or_else(|| y.list("decoding.durations")).unwrap_or_default();
        let model_type = y.scalar("decoding.model_type").unwrap_or("");
        cfg.head = if model_type == "tdt" || !durations.is_empty() {
            let d = durations.iter().map(|v| v.parse::<usize>().map_err(|_| bad(format!("a TDT duration {v:?}")))).collect::<Result<Vec<_>>>()?;
            if d.is_empty() {
                return Err(bad("a TDT model without durations"));
            }
            Head::Tdt { durations: d }
        } else {
            Head::Rnnt
        };
        if let Some(m) = num("decoding.greedy.max_symbols") {
            cfg.max_symbols = m as usize;
        }
        cfg.xscaling = y.scalar("encoder.xscaling").is_some_and(|v| v == "true");
        match y.scalar("encoder.self_attention_model") {
            None | Some("rel_pos") => {}
            Some(other) => return Err(bad(format!("encoder.self_attention_model {other} is not supported (rel_pos only)"))),
        }
        match y.scalar("encoder.subsampling") {
            None | Some("dw_striding") => {}
            Some(other) => return Err(bad(format!("encoder.subsampling {other} is not supported (dw_striding only)"))),
        }
        if let Some(ctx) = y.list("encoder.att_context_size") {
            // A list of [left, right] pairs (multi-lookahead models) keeps the first.
            let v: Vec<i64> = ctx.iter().filter_map(|s| s.trim_matches(|c| c == '[' || c == ']').split(',').next().and_then(|x| x.trim().parse().ok())).collect();
            if ctx.len() == 2 && v.len() == 2 {
                cfg.att_context = [v[0], v[1]];
            }
        }
        Ok(cfg)
    }

    /// From a Hugging Face `config.json` and, when present, `processor_config.json`.
    pub fn from_hf(config: &[u8], processor: Option<&[u8]>) -> Result<Self> {
        let c = Json::parse(config).map_err(|e| bad(format!("config.json: {e}")))?;
        let mut cfg = Self::parakeet();
        match c.get("model_type").and_then(Json::as_str) {
            Some("parakeet_tdt") | None => {
                if let Some(d) = c.get("durations").and_then(Json::as_array) {
                    cfg.head = Head::Tdt { durations: d.iter().map(|x| x.as_i64().map(|v| v as usize).ok_or_else(|| bad("config.json: a bad duration"))).collect::<Result<Vec<_>>>()? };
                }
            }
            Some("parakeet_rnnt") => cfg.head = Head::Rnnt,
            Some(other) => return Err(bad(format!("config.json: model_type {other} is not a Parakeet transducer"))),
        }
        if let Some(m) = c.get("max_symbols_per_step").and_then(Json::as_i64) {
            cfg.max_symbols = m.max(1) as usize;
        }
        if let Some(enc) = c.get("encoder_config") {
            cfg.xscaling = enc.get("scale_input").and_then(Json::as_bool).unwrap_or(false);
            if let Some(n) = enc.get("num_mel_bins").and_then(Json::as_i64) {
                cfg.mel.n_mels = n as usize;
            }
        }
        if let Some(p) = processor {
            let p = Json::parse(p).map_err(|e| bad(format!("processor_config.json: {e}")))?;
            let fe = p.get("feature_extractor").unwrap_or(&p);
            let int = |k: &str| fe.get(k).and_then(Json::as_i64).map(|v| v as usize);
            if let Some(v) = int("feature_size") {
                cfg.mel.n_mels = v;
            }
            if let Some(v) = int("sampling_rate") {
                cfg.mel.sample_rate = v;
            }
            if let Some(v) = int("n_fft") {
                cfg.mel.n_fft = v;
            }
            if let Some(v) = int("win_length") {
                cfg.mel.win_length = v;
            }
            if let Some(v) = int("hop_length") {
                cfg.mel.hop_length = v;
            }
            if let Some(v) = fe.get("preemphasis") {
                cfg.mel.preemph = v.as_f64().map(|x| x as f32);
            }
        }
        Ok(cfg)
    }
}

/// Refuse front-end settings this implementation does not reproduce, rather
/// than transcribe with the wrong features.
fn check_front_end(y: &Yaml) -> Result<()> {
    let want = |k: &str, ok: &[&str]| -> Result<()> {
        match y.scalar(k) {
            Some(v) if !ok.contains(&v) => Err(bad(format!("{k}: {v} is not supported"))),
            _ => Ok(()),
        }
    };
    want("preprocessor.normalize", &["per_feature"])?;
    want("preprocessor.window", &["hann"])?;
    want("preprocessor.log", &["true"])?;
    want("preprocessor.frame_splicing", &["1"])?;
    want("preprocessor.mag_power", &["2.0", "2"])?;
    want("preprocessor.exact_pad", &["false"])?;
    want("preprocessor.log_zero_guard_type", &["add"])?;
    want("preprocessor.mel_norm", &["slaney"])?;
    want("preprocessor.lowfreq", &["0", "0.0"])?;
    if let Some(h) = y.scalar("preprocessor.highfreq") {
        if h != "null" {
            return Err(bad(format!("preprocessor.highfreq {h} is not supported")));
        }
    }
    Ok(())
}

/// Just enough YAML for an OmegaConf dump: nested `key: value` maps with
/// two-space indentation and block lists (`- item`), read into dotted paths.
/// Flow lists (`[a, b]`) are kept as one scalar.
pub struct Yaml {
    scalars: HashMap<String, String>,
    lists: HashMap<String, Vec<String>>,
}

impl Yaml {
    pub fn parse(text: &str) -> Self {
        let mut scalars = HashMap::new();
        let mut lists: HashMap<String, Vec<String>> = HashMap::new();
        // (indent, key) of the open maps.
        let mut stack: Vec<(usize, String)> = Vec::new();
        let path = |stack: &[(usize, String)], key: &str| {
            let mut p: Vec<&str> = stack.iter().map(|(_, k)| k.as_str()).collect();
            p.push(key);
            p.join(".")
        };
        for raw in text.lines() {
            let line = raw.trim_end();
            let body = line.trim_start();
            if body.is_empty() || body.starts_with('#') {
                continue;
            }
            let indent = line.len() - body.len();
            if let Some(item) = body.strip_prefix("- ").or(if body == "-" { Some("") } else { None }) {
                // A list item belongs to the innermost open key at or left of it.
                while stack.last().is_some_and(|(i, _)| *i > indent) {
                    stack.pop();
                }
                if let Some((_, key)) = stack.last() {
                    let owner = path(&stack[..stack.len() - 1], key);
                    lists.entry(owner).or_default().push(unquote(item.trim()));
                }
                continue;
            }
            let Some((key, value)) = split_key(body) else { continue };
            while stack.last().is_some_and(|(i, _)| *i >= indent) {
                stack.pop();
            }
            let value = value.trim();
            if value.is_empty() {
                stack.push((indent, key.to_string()));
            } else {
                scalars.insert(path(&stack, key), unquote(value));
            }
        }
        Self { scalars, lists }
    }

    pub fn scalar(&self, path: &str) -> Option<&str> {
        self.scalars.get(path).map(String::as_str)
    }

    pub fn list(&self, path: &str) -> Option<Vec<String>> {
        if let Some(l) = self.lists.get(path) {
            return Some(l.clone());
        }
        // A flow list on one line.
        let s = self.scalars.get(path)?;
        let inner = s.strip_prefix('[')?.strip_suffix(']')?;
        Some(inner.split(',').map(|x| unquote(x.trim())).filter(|x| !x.is_empty()).collect())
    }
}

/// `key: value` or `key:`, with a plain or quoted key.
fn split_key(body: &str) -> Option<(&str, &str)> {
    if let Some(rest) = body.strip_prefix('\'').or_else(|| body.strip_prefix('"')) {
        let q = body.as_bytes()[0] as char;
        let end = rest.find(q)?;
        let after = rest[end + 1..].strip_prefix(':')?;
        return Some((&rest[..end], after));
    }
    let at = body.find(": ").or_else(|| body.strip_suffix(':').map(|b| b.len()))?;
    Some((&body[..at], body.get(at + 1..).unwrap_or("")))
}

fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() >= 2 && ((b[0] == b'\'' && b[b.len() - 1] == b'\'') || (b[0] == b'"' && b[b.len() - 1] == b'"')) {
        let inner = &s[1..s.len() - 1];
        return if b[0] == b'\'' { inner.replace("''", "'") } else { inner.to_string() };
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEMO: &str = "sample_rate: 16000\nmodel_defaults:\n  enc_hidden: 1024\n  tdt_durations:\n  - 0\n  - 1\n  - 2\n  - 3\n  - 4\n  num_tdt_durations: 5\npreprocessor:\n  _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor\n  sample_rate: 16000\n  normalize: per_feature\n  window_size: 0.025\n  window_stride: 0.01\n  window: hann\n  features: 128\n  n_fft: 512\n  log: true\n  frame_splicing: 1\n  dither: 1.0e-05\n  pad_to: 0\nencoder:\n  xscaling: false\n  att_context_size:\n  - -1\n  - -1\n  self_attention_model: rel_pos\njoint:\n  vocabulary:\n  - <unk>\n  - 'on'\n  - ▁t\ndecoding:\n  strategy: greedy_batch\n  model_type: tdt\n  greedy:\n    max_symbols: 10\n";

    #[test]
    fn nemo_yaml_paths_and_lists() {
        let y = Yaml::parse(NEMO);
        assert_eq!(y.scalar("preprocessor.window_size"), Some("0.025"));
        assert_eq!(y.scalar("decoding.greedy.max_symbols"), Some("10"));
        assert_eq!(y.list("model_defaults.tdt_durations").unwrap(), vec!["0", "1", "2", "3", "4"]);
        assert_eq!(y.scalar("model_defaults.num_tdt_durations"), Some("5"));
        assert_eq!(y.list("joint.vocabulary").unwrap(), vec!["<unk>", "on", "▁t"]);
        assert_eq!(y.list("encoder.att_context_size").unwrap(), vec!["-1", "-1"]);
    }

    #[test]
    fn nemo_config_reads_the_front_end_and_head() {
        let c = ModelConfig::from_nemo_yaml(NEMO).unwrap();
        assert_eq!(c.mel, MelConfig::parakeet(128));
        assert_eq!(c.head, Head::Tdt { durations: vec![0, 1, 2, 3, 4] });
        assert_eq!((c.max_symbols, c.xscaling, c.att_context), (10, false, [-1, -1]));
        let rnnt = NEMO.replace("  model_type: tdt\n", "  model_type: rnnt\n").replace("  tdt_durations:\n  - 0\n  - 1\n  - 2\n  - 3\n  - 4\n", "");
        assert_eq!(ModelConfig::from_nemo_yaml(&rnnt).unwrap().head, Head::Rnnt);
        assert!(ModelConfig::from_nemo_yaml(&NEMO.replace("normalize: per_feature", "normalize: all_features")).is_err());
    }

    #[test]
    fn hf_config_reads_durations_and_features() {
        let config = br#"{"architectures":["ParakeetForTDT"],"durations":[0,1,2,3,4],"encoder_config":{"scale_input":false,"num_mel_bins":128},"max_symbols_per_step":10,"model_type":"parakeet_tdt"}"#;
        let processor = br#"{"feature_extractor":{"feature_size":128,"hop_length":160,"n_fft":512,"preemphasis":0.97,"sampling_rate":16000,"win_length":400}}"#;
        let c = ModelConfig::from_hf(config, Some(processor)).unwrap();
        assert_eq!(c.mel, MelConfig::parakeet(128));
        assert_eq!(c.head, Head::Tdt { durations: vec![0, 1, 2, 3, 4] });
    }
}
