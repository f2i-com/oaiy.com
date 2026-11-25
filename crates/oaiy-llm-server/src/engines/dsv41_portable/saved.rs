//! A DeepSeek-V4.1 state kept on disk ([`crate::engines::disk`]): the state a conversation's system turn leaves, so a
//! later process (OAIY started again, the model loaded again) reads on from it rather than reading the turn again (an
//! agent's 19,000-21,000 tokens, eleven minutes of the 2-bit GGUF's).
//!
//! A state is every layer's caches whole, sized for the context (some 1 GB at 65,536 tokens), most of it as a fresh
//! state left it: each vector is written up to its last value that is not its fresh fill, in bf16 where every value
//! written is one (the compressed rows and index keys, fp4 values on an e4m3 or power-of-two scale, are), and read back
//! into a vector of its whole length filled as a fresh one is. The layout says each vector's length and fill, so a
//! state reads back without the model; the fingerprint (the model's files and its context) says whose it is.

use std::io::{Read, Write};
use std::sync::Arc;

use crate::dsv41::attention::AttnState;
use crate::dsv41::model::Checkpoint;

/// A state on its way to disk or read from it: the tokens it covers and the state (shared with the engine's own copy,
/// [`super::Engine`]'s `system`, so a write copies nothing).
pub(crate) struct SavedState {
    pub pos: usize,
    pub state: Arc<Checkpoint>,
    /// Each vector's values written and whether as bf16, in order (made once, for its length and its write).
    plan: Vec<(usize, bool)>,
}

/// How much of `v` a state keeps on disk: up to its last value that is not `fill` (as bits), and whether all of those
/// are bf16 values (their low 16 bits 0).
fn kept(v: &[f32], fill: f32) -> (usize, bool) {
    let n = v.iter().rposition(|x| x.to_bits() != fill.to_bits()).map_or(0, |i| i + 1);
    (n, v[..n].iter().all(|x| x.to_bits() & 0xffff == 0))
}

/// A vector's header: its length, its values written, its fill's bits, whether as bf16.
const VECTOR_HEAD: usize = 8 + 8 + 4 + 1;
const MAX_POS: u64 = 1 << 24;
const MAX_LAYERS: u64 = 1024;
const MAX_VEC_LEN: u64 = 1 << 27;
const MAX_TOTAL_FLOATS: u64 = 1 << 30;
const MAX_HISTORY: u64 = 1 << 24;

impl SavedState {
    pub fn new(pos: usize, state: Arc<Checkpoint>) -> SavedState {
        let plan = state.parts().0.iter().flat_map(|s| s.parts().map(|(v, fill)| kept(v, fill))).collect();
        SavedState { pos, state, plan }
    }

    fn read(input: &mut dyn Read, length: u64) -> std::io::Result<SavedState> {
        const CHUNK_ELEMS: usize = 16_384;
        let bad = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("a damaged DeepSeek-V4.1 state: {what}"));
        let mut remaining = length.checked_sub(16).ok_or_else(|| bad("truncated header"))?;
        let mut word = [0u8; 8];
        let mut u64_of = |input: &mut dyn Read| -> std::io::Result<u64> {
            input.read_exact(&mut word)?;
            Ok(u64::from_le_bytes(word))
        };
        let pos_u64 = u64_of(input)?;
        let layers_u64 = u64_of(input)?;
        if pos_u64 == 0 || pos_u64 > MAX_POS {
            return Err(bad("its position"));
        }
        let min_layer_bytes = layers_u64
            .checked_mul(7 * VECTOR_HEAD as u64)
            .and_then(|b| b.checked_add(8));
        if layers_u64 > MAX_LAYERS || min_layer_bytes.is_none_or(|need| need > remaining) {
            return Err(bad("its layers"));
        }
        let pos = usize::try_from(pos_u64).map_err(|_| bad("its position"))?;
        let layers = usize::try_from(layers_u64).map_err(|_| bad("its layers"))?;
        let mut states = Vec::with_capacity(layers);
        let mut plan = Vec::with_capacity(layers * 7);
        let mut total_floats = 0u64;
        let mut scratch = vec![0u8; CHUNK_ELEMS * 4];
        for _ in 0..layers {
            let mut parts = Vec::with_capacity(7);
            for _ in 0..7 {
                remaining = remaining.checked_sub(VECTOR_HEAD as u64).ok_or_else(|| bad("truncated vector header"))?;
                let len_u64 = u64_of(input)?;
                let n_u64 = u64_of(input)?;
                let mut fill = [0u8; 4];
                input.read_exact(&mut fill)?;
                let mut kind = [0u8; 1];
                input.read_exact(&mut kind)?;
                if n_u64 > len_u64 || len_u64 > MAX_VEC_LEN || kind[0] > 1 {
                    return Err(bad("a vector's length"));
                }
                total_floats = total_floats
                    .checked_add(len_u64)
                    .filter(|&t| t <= MAX_TOTAL_FLOATS)
                    .ok_or_else(|| bad("its total size"))?;
                let bf16 = kind[0] == 1;
                let elem_bytes = if bf16 { 2u64 } else { 4u64 };
                let data_bytes = n_u64.checked_mul(elem_bytes).ok_or_else(|| bad("a vector's length"))?;
                remaining = remaining.checked_sub(data_bytes).ok_or_else(|| bad("truncated vector data"))?;
                let len = usize::try_from(len_u64).map_err(|_| bad("a vector's length"))?;
                let n = usize::try_from(n_u64).map_err(|_| bad("a vector's length"))?;
                let fill_f32 = f32::from_bits(u32::from_le_bytes(fill));
                let mut v = Vec::with_capacity(n.min(CHUNK_ELEMS));
                let mut left = n;
                while left > 0 {
                    let take = left.min(CHUNK_ELEMS);
                    let buf = &mut scratch[..take * elem_bytes as usize];
                    input.read_exact(buf)?;
                    if bf16 {
                        for b in buf.chunks_exact(2) {
                            v.push(f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16));
                        }
                    } else {
                        for b in buf.chunks_exact(4) {
                            v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                        }
                    }
                    left -= take;
                }
                v.resize(len, fill_f32);
                plan.push((n, bf16));
                parts.push(v);
            }
            states.push(AttnState::from_parts(parts).ok_or_else(|| bad("a layer"))?);
        }
        remaining = remaining.checked_sub(8).ok_or_else(|| bad("truncated history header"))?;
        let hist_u64 = u64_of(input)?;
        let hist_bytes = hist_u64.checked_mul(8).ok_or_else(|| bad("its history"))?;
        if hist_u64 > MAX_HISTORY || hist_bytes > remaining {
            return Err(bad("its history"));
        }
        remaining -= hist_bytes;
        if remaining != 0 {
            return Err(bad("trailing bytes"));
        }
        let n = usize::try_from(hist_u64).map_err(|_| bad("its history"))?;
        let mut history = Vec::with_capacity(n.min(CHUNK_ELEMS));
        let mut left = n;
        while left > 0 {
            let take = left.min(CHUNK_ELEMS / 2);
            let buf = &mut scratch[..take * 8];
            input.read_exact(buf)?;
            for b in buf.chunks_exact(8) {
                history.push(i64::from_le_bytes(b.try_into().expect("eight bytes")));
            }
            left -= take;
        }
        let mut extra = [0u8; 1];
        if input.read(&mut extra)? != 0 {
            return Err(bad("trailing bytes"));
        }
        Ok(SavedState { pos, state: Arc::new(Checkpoint::from_parts(states, history)), plan })
    }
}

impl crate::engines::disk::PromptState for SavedState {
    fn pos(&self) -> usize {
        self.pos
    }

    fn encoded_len(&self) -> usize {
        let history = self.state.parts().1;
        16 + self.plan.iter().map(|&(n, bf16)| VECTOR_HEAD + n * if bf16 { 2 } else { 4 }).sum::<usize>() + 8 + history.len() * 8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.reserve(self.encoded_len());
        self.write_to(out).expect("a write to memory");
    }

    fn write_to(&self, out: &mut dyn Write) -> std::io::Result<()> {
        let (states, history) = self.state.parts();
        out.write_all(&(self.pos as u64).to_le_bytes())?;
        out.write_all(&(states.len() as u64).to_le_bytes())?;
        let mut plan = self.plan.iter();
        // (a vector's values a few MB at a time, as bytes)
        let mut bytes: Vec<u8> = Vec::new();
        for s in states {
            for (v, fill) in s.parts() {
                let &(n, bf16) = plan.next().expect("a plan of every vector");
                out.write_all(&(v.len() as u64).to_le_bytes())?;
                out.write_all(&(n as u64).to_le_bytes())?;
                out.write_all(&fill.to_bits().to_le_bytes())?;
                out.write_all(&[u8::from(bf16)])?;
                for chunk in v[..n].chunks(1 << 20) {
                    bytes.clear();
                    if bf16 {
                        bytes.extend(chunk.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()));
                    } else {
                        bytes.extend(chunk.iter().flat_map(|x| x.to_le_bytes()));
                    }
                    out.write_all(&bytes)?;
                }
            }
        }
        out.write_all(&(history.len() as u64).to_le_bytes())?;
        for h in history {
            out.write_all(&h.to_le_bytes())?;
        }
        Ok(())
    }

    fn decode(bytes: &[u8]) -> oaiy_engine::Result<SavedState> {
        let mut input = bytes;
        SavedState::read(&mut input, bytes.len() as u64).map_err(oaiy_engine::Error::Io)
    }

    fn read_from(input: &mut dyn Read, length: u64) -> oaiy_engine::Result<SavedState> {
        let mut input = std::io::BufReader::with_capacity(1 << 22, input.take(length));
        SavedState::read(&mut input, length).map_err(oaiy_engine::Error::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::disk::PromptState;

    /// A state of two layers, its vectors partly filled (the rest as fresh), some of them bf16 values and some not.
    fn state() -> SavedState {
        let ninf = f32::NEG_INFINITY;
        let layer = |seed: f32| {
            let mut compressed = vec![0.0f32; 4096];
            for (i, v) in compressed.iter_mut().take(1000).enumerate() {
                *v = f32::from_bits(((seed + i as f32) * 0.37).to_bits() & 0xffff_0000);
            }
            let mut window = vec![0.0f32; 512];
            for (i, v) in window.iter_mut().enumerate() {
                *v = (seed - i as f32) * 0.001;
            }
            let mut scores = vec![ninf; 64];
            scores[3] = 1.5;
            AttnState::from_parts(vec![window, compressed, vec![0.25; 64], scores, vec![0.0; 256], vec![], vec![]]).expect("seven vectors")
        };
        SavedState::new(1234, Arc::new(Checkpoint::from_parts(vec![layer(1.0), layer(7.0)], vec![5, -3, 99])))
    }

    #[test]
    fn a_saved_state_reads_back_as_it_was() {
        let s = state();
        let mut bytes = Vec::new();
        s.encode(&mut bytes);
        assert_eq!(bytes.len(), s.encoded_len());
        let back = SavedState::read_from(&mut &bytes[..], bytes.len() as u64).expect("read back");
        assert_eq!(back.pos(), 1234);
        let ((a, ha), (b, hb)) = (s.state.parts(), back.state.parts());
        assert_eq!(ha, hb);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            for (p, q) in x.parts().iter().zip(y.parts()) {
                assert_eq!(p.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), q.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
        // (the compressed rows' 1,000 bf16 values in half the bytes, the rest of them not written)
        assert!(bytes.len() < 2 * (512 * 4 + 1000 * 2 + 64 * 4 + 64 * 4 + 7 * VECTOR_HEAD) + 64, "{} bytes", bytes.len());
    }

    #[test]
    fn a_damaged_saved_state_is_refused() {
        let s = state();
        let mut bytes = Vec::new();
        s.encode(&mut bytes);
        assert!(SavedState::read_from(&mut &bytes[..bytes.len() - 9], (bytes.len() - 9) as u64).is_err(), "cut short");
        let mut wrong = bytes.clone();
        wrong[8..16].copy_from_slice(&5000u64.to_le_bytes());
        assert!(SavedState::read_from(&mut &wrong[..], wrong.len() as u64).is_err(), "its layers' count");

        // pos == 0 rejected
        let mut zero_pos = bytes.clone();
        zero_pos[0..8].copy_from_slice(&0u64.to_le_bytes());
        assert!(SavedState::decode(&zero_pos).is_err(), "pos == 0");

        // trailing bytes rejected
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(SavedState::decode(&trailing).is_err(), "trailing bytes");

        // huge vector len / n in truncated buffer must fail cleanly without OOM
        let mut huge_vec = Vec::new();
        huge_vec.extend_from_slice(&1u64.to_le_bytes()); // pos
        huge_vec.extend_from_slice(&1u64.to_le_bytes()); // layers
        huge_vec.extend_from_slice(&(1u64 << 31).to_le_bytes()); // len
        huge_vec.extend_from_slice(&(1u64 << 30).to_le_bytes()); // n
        huge_vec.extend_from_slice(&0u32.to_le_bytes()); // fill
        huge_vec.push(0); // f32
        assert!(SavedState::decode(&huge_vec).is_err(), "huge vector counts");

        // huge history count in truncated buffer must fail cleanly without OOM
        let mut huge_hist = Vec::new();
        huge_hist.extend_from_slice(&1u64.to_le_bytes()); // pos
        huge_hist.extend_from_slice(&0u64.to_le_bytes()); // 0 layers
        huge_hist.extend_from_slice(&(1u64 << 30).to_le_bytes()); // huge history len
        assert!(SavedState::decode(&huge_hist).is_err(), "huge history count");
    }
}
