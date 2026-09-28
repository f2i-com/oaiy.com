//! Text to token ids: the Qwen2 byte-level BPE the talker reads.
use candle_core::Result;
use std::path::Path;

/// Qwen2 byte-level BPE from `vocab.json` and `merges.txt`.
pub fn tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer> {
    use tokenizers::pre_tokenizers::{byte_level::ByteLevel, sequence::Sequence, split::Split, split::SplitPattern};
    let (vocab, merges) = (dir.join("vocab.json"), dir.join("merges.txt"));
    let bpe = tokenizers::models::bpe::BPE::from_file(&vocab.to_string_lossy(), &merges.to_string_lossy())
        .build()
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer ({}): {e}", dir.display())))?;
    let mut tok = tokenizers::Tokenizer::new(bpe);
    tok.with_normalizer(Some(tokenizers::normalizers::unicode::NFC));
    const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(SplitPattern::Regex(PATTERN.into()), tokenizers::SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?;
    tok.with_pre_tokenizer(Some(Sequence::new(vec![split.into(), ByteLevel::new(false, false, false).into()])));
    Ok(tok)
}

pub fn encode(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(text, false).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))?.get_ids().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 0.6B folder, if it is on this machine (`OAIY_TTS_MODEL` or the usual place).
    fn model_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var_os("OAIY_TTS_MODEL").map(Into::into).unwrap_or_else(|| std::path::PathBuf::from("E:/models/Qwen3-TTS-12Hz-0.6B-Base"));
        dir.join("vocab.json").exists().then_some(dir)
    }

    #[test]
    fn text_becomes_qwen_token_ids() -> Result<()> {
        let Some(dir) = model_dir() else {
            eprintln!("skipped: no Qwen3-TTS model folder");
            return Ok(());
        };
        let tok = tokenizer(&dir)?;
        // "Hello" and " world" are single Qwen2 tokens; numbers split per digit.
        assert_eq!(encode(&tok, "Hello world")?, vec![9707, 1879]);
        assert_eq!(encode(&tok, "42")?, vec![19, 17]);
        Ok(())
    }
}
