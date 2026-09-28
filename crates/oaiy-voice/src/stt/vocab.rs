//! Token ids to text. Parakeet's tokenizers are SentencePiece BPE models
//! (1024 pieces for the English ones, 8192 for v3), shipped as a
//! SentencePiece `tokenizer.model` inside a `.nemo` or as a Hugging Face
//! `tokenizer.json`. Decoding needs only the pieces: `▁` marks a word start,
//! `<0xNN>` pieces are raw bytes (v3 has byte fallback), and control pieces
//! (`<unk>`, `<pad>`, `<|...|>`) print nothing.

use oaiy_engine::json::Json;

use super::{bad, Result};

#[derive(Clone, Debug, PartialEq)]
enum Piece {
    Text(String),
    Byte(u8),
    Control,
}

#[derive(Clone, Debug)]
pub struct Vocab {
    pieces: Vec<Piece>,
}

fn classify(s: &str, control: bool) -> Piece {
    if control {
        return Piece::Control;
    }
    if let Some(hex) = s.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')) {
        if let Ok(b) = u8::from_str_radix(hex, 16) {
            if hex.len() == 2 {
                return Piece::Byte(b);
            }
        }
    }
    if (s.starts_with("<|") && s.ends_with("|>")) || s == "<unk>" || s == "<pad>" || s == "<s>" || s == "</s>" {
        return Piece::Control;
    }
    Piece::Text(s.to_string())
}

impl Vocab {
    /// From piece strings in id order (tests, and configs that list them).
    pub fn from_pieces<S: AsRef<str>>(pieces: &[S]) -> Self {
        Self { pieces: pieces.iter().map(|p| classify(p.as_ref(), false)).collect() }
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// A SentencePiece `ModelProto` (`tokenizer.model`): field 1 holds the
    /// pieces, each with its string (1), score (2) and type (3; 2 unknown,
    /// 3 control, 6 byte).
    pub fn from_sentencepiece(bytes: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut r = Proto { b: bytes, i: 0 };
        while let Some((field, wire)) = r.key()? {
            if field == 1 && wire == 2 {
                let body = r.bytes()?;
                let mut p = Proto { b: body, i: 0 };
                let (mut text, mut kind) = (String::new(), 1u64);
                while let Some((f, w)) = p.key()? {
                    match (f, w) {
                        (1, 2) => text = String::from_utf8_lossy(p.bytes()?).into_owned(),
                        (3, 0) => kind = p.varint()?,
                        _ => p.skip(w)?,
                    }
                }
                pieces.push(match kind {
                    6 => classify(&text, false),
                    2..=5 => Piece::Control,
                    _ => classify(&text, false),
                });
            } else {
                r.skip(wire)?;
            }
        }
        if pieces.is_empty() {
            return Err(bad("tokenizer.model holds no pieces"));
        }
        Ok(Self { pieces })
    }

    /// A Hugging Face `tokenizer.json`: `model.vocab` (piece to id) and the
    /// added tokens, which are control pieces when `special`.
    pub fn from_tokenizer_json(bytes: &[u8]) -> Result<Self> {
        let json = Json::parse(bytes).map_err(|e| bad(format!("tokenizer.json: {e}")))?;
        let vocab = json.get("model").and_then(|m| m.get("vocab")).ok_or_else(|| bad("tokenizer.json has no model.vocab"))?;
        let mut pieces: Vec<Option<Piece>> = Vec::new();
        let mut put = |id: i64, piece: Piece| -> Result<()> {
            let id = usize::try_from(id).map_err(|_| bad("tokenizer.json: a negative id"))?;
            if id > 1 << 20 {
                return Err(bad("tokenizer.json: an id past a million"));
            }
            if pieces.len() <= id {
                pieces.resize(id + 1, None);
            }
            pieces[id] = Some(piece);
            Ok(())
        };
        match vocab {
            Json::Obj(members) => {
                for (piece, id) in members {
                    put(id.as_i64().ok_or_else(|| bad("tokenizer.json: a non-integer id"))?, classify(piece, false))?;
                }
            }
            // Unigram models list [piece, score] pairs in id order.
            Json::Arr(items) => {
                for (id, item) in items.iter().enumerate() {
                    let piece = item.at(0).and_then(Json::as_str).ok_or_else(|| bad("tokenizer.json: a bad vocab entry"))?;
                    put(id as i64, classify(piece, false))?;
                }
            }
            _ => return Err(bad("tokenizer.json: model.vocab is neither a map nor a list")),
        }
        if let Some(added) = json.get("added_tokens").and_then(Json::as_array) {
            for t in added {
                let (Some(id), Some(content)) = (t.get("id").and_then(Json::as_i64), t.get("content").and_then(Json::as_str)) else { continue };
                let special = t.get("special").and_then(Json::as_bool).unwrap_or(false);
                put(id, classify(content, special))?;
            }
        }
        Ok(Self { pieces: pieces.into_iter().map(|p| p.unwrap_or(Piece::Control)).collect() })
    }

    /// The text of a token sequence: pieces joined, `▁` as a space, byte
    /// pieces gathered into UTF-8, leading and trailing spaces trimmed (as
    /// SentencePiece's `decode` does).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            match self.pieces.get(id as usize) {
                Some(Piece::Text(s)) => bytes.extend_from_slice(s.replace('\u{2581}', " ").as_bytes()),
                Some(Piece::Byte(b)) => bytes.push(*b),
                Some(Piece::Control) | None => {}
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

/// Just enough protobuf to walk a `ModelProto`.
struct Proto<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Proto<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.b.get(self.i).ok_or_else(|| bad("tokenizer.model: truncated"))?;
            self.i += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(bad("tokenizer.model: a varint too long"))
    }

    fn key(&mut self) -> Result<Option<(u64, u64)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let k = self.varint()?;
        Ok(Some((k >> 3, k & 7)))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = usize::try_from(self.varint()?).map_err(|_| bad("tokenizer.model: a field too long"))?;
        let s = self.b.get(self.i..self.i.checked_add(n).ok_or_else(|| bad("tokenizer.model: a field too long"))?).ok_or_else(|| bad("tokenizer.model: truncated"))?;
        self.i += n;
        Ok(s)
    }

    fn skip(&mut self, wire: u64) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.i += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.i += 4,
            _ => return Err(bad(format!("tokenizer.model: wire type {wire}"))),
        }
        if self.i > self.b.len() {
            return Err(bad("tokenizer.model: truncated"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_joins_pieces_and_bytes() {
        let v = Vocab::from_pieces(&["<unk>", "\u{2581}Hel", "lo", "\u{2581}w", "orld", "<0xC3>", "<0xA9>", "<|endoftext|>", "."]);
        assert_eq!(v.decode(&[1, 2, 3, 4, 8]), "Hello world.");
        assert_eq!(v.decode(&[3, 5, 6, 7, 0]), "w\u{e9}");
        assert_eq!(v.decode(&[]), "");
        assert_eq!(v.decode(&[1, 99]), "Hel");
    }

    #[test]
    fn sentencepiece_model_pieces_are_read() {
        // ModelProto { pieces: [{piece: "<unk>", type: UNKNOWN}, {piece: "▁a", score: -1.0}] }
        fn piece(text: &str, kind: Option<u8>) -> Vec<u8> {
            let mut body = vec![0x0a, text.len() as u8];
            body.extend_from_slice(text.as_bytes());
            body.extend_from_slice(&[0x15, 0, 0, 0x80, 0xbf]);
            if let Some(k) = kind {
                body.extend_from_slice(&[0x18, k]);
            }
            let mut out = vec![0x0a, body.len() as u8];
            out.extend(body);
            out
        }
        let mut model = piece("<unk>", Some(2));
        model.extend(piece("\u{2581}a", None));
        // A trainer_spec (field 2) to skip.
        model.extend_from_slice(&[0x12, 0x02, 0x08, 0x01]);
        let v = Vocab::from_sentencepiece(&model).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v.decode(&[0, 1, 1]), "a a");
    }

    #[test]
    fn tokenizer_json_vocab_and_added_tokens() {
        let json = r#"{"added_tokens":[{"id":0,"content":"<unk>","special":true},{"id":3,"content":"<|endoftext|>","special":false}],
            "model":{"type":"BPE","vocab":{"<unk>":0,"▁hi":1,"!":2,"<|endoftext|>":3}}}"#;
        let v = Vocab::from_tokenizer_json(json.as_bytes()).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v.decode(&[1, 2, 3]), "hi!");
    }
}
