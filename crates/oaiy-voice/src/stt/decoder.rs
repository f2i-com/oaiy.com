//! The transducer's prediction network and joint, and greedy decoding of
//! both heads Parakeet ships: RNN-T and TDT (token-and-duration).
//!
//! The decoding loops follow NeMo's `GreedyRNNTInfer._greedy_decode` and
//! `GreedyTDTInfer._greedy_decode` step for step, including how the
//! per-frame symbol cap interacts with durations, so that the same logits give
//! the same tokens. They ask a [`Scorer`] for the best token (and duration) at
//! a frame; [`DeviceScorer`] answers from the joint on the model's device.
//!
//! The prediction network only changes when a token is emitted, so between
//! emissions the joint depends on the encoder frame alone: the scorer
//! evaluates a block of upcoming frames in one product and serves the
//! following blank steps from it, which takes one device round trip per
//! block instead of one per frame.

use candle_core::{DType, Device, Module, Tensor};

use super::encoder::Linear;
use super::weights::Weights;
use super::{bad, Result};

/// The best token and the best duration index at one frame.
pub trait Scorer {
    fn best(&mut self, frame: usize) -> Result<(u32, usize)>;
    /// Feed an emitted token to the prediction network.
    fn emit(&mut self, token: u32) -> Result<()>;
}

/// Greedy RNN-T: at each frame, emit tokens until blank or `max_symbols`.
pub fn greedy_rnnt(frames: usize, blank: u32, max_symbols: usize, s: &mut dyn Scorer) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for t in 0..frames {
        let mut symbols = 0;
        while symbols < max_symbols {
            let (k, _) = s.best(t)?;
            if k == blank {
                break;
            }
            out.push(k);
            s.emit(k)?;
            symbols += 1;
        }
    }
    Ok(out)
}

/// Greedy TDT: a step emits a token (or blank) and advances by its predicted
/// duration; a duration of 0 stays on the frame, up to `max_symbols` steps,
/// after which the frame is left anyway.
pub fn greedy_tdt(frames: usize, blank: u32, durations: &[usize], max_symbols: usize, s: &mut dyn Scorer) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut t = 0;
    while t < frames {
        let mut symbols = 0;
        let mut need_loop = true;
        while need_loop && symbols < max_symbols {
            let (k, d) = s.best(t)?;
            let skip = *durations.get(d).ok_or_else(|| bad(format!("duration index {d} out of range")))?;
            if k != blank {
                out.push(k);
                s.emit(k)?;
            }
            symbols += 1;
            t += skip;
            need_loop = skip == 0;
        }
        if symbols == max_symbols {
            t += 1;
        }
    }
    Ok(out)
}

struct Lstm {
    /// `[W_ih W_hh]`: `(4 hidden, input + hidden)`, gates i, f, g, o.
    w: Linear,
    hidden: usize,
}

impl Lstm {
    /// One step: `x` `(1, input)`, state `(h, c)` each `(1, hidden)`.
    fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let gates = self.w.forward(&Tensor::cat(&[x, h], 1)?)?;
        let n = self.hidden;
        let i = candle_nn::ops::sigmoid(&gates.narrow(1, 0, n)?)?;
        let f = candle_nn::ops::sigmoid(&gates.narrow(1, n, n)?)?;
        let g = gates.narrow(1, 2 * n, n)?.tanh()?;
        let o = candle_nn::ops::sigmoid(&gates.narrow(1, 3 * n, n)?)?;
        let c = ((f * c)? + (i * g)?)?;
        let h = (o * c.tanh()?)?;
        Ok((h, c))
    }
}

/// Sizes read from the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderShape {
    /// Tokens including blank (the embedding rows).
    pub tokens: usize,
    pub hidden: usize,
    pub layers: usize,
    pub joint: usize,
    /// Joint outputs: tokens, blank and any durations.
    pub outputs: usize,
}

pub struct Decoder {
    embed: Tensor,
    lstm: Vec<Lstm>,
    /// `joint.pred`: prediction output to the joint's width.
    pred: Linear,
    /// `joint.enc`: encoder output to the joint's width.
    enc: Linear,
    /// `joint.joint_net.2`: after ReLU, to the outputs.
    out: Linear,
    pub shape: DecoderShape,
    dev: Device,
}

/// The prediction network's state after the tokens so far: each layer's
/// `(h, c)` and the projected output the joint adds to an encoder frame.
#[derive(Clone)]
pub struct PredState {
    layers: Vec<(Tensor, Tensor)>,
    projected: Tensor,
}

impl Decoder {
    pub fn load(w: &mut Weights, d_model: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let e = w.shape("decoder.prediction.embed.weight")?;
        let [tokens, hidden] = e[..] else { return Err(bad("decoder.prediction.embed.weight is not 2-D")) };
        let embed = w.tensor("decoder.prediction.embed.weight", &[tokens, hidden], DType::F32, dev)?;
        let mut lstm = Vec::new();
        while w.has(&format!("decoder.prediction.dec_rnn.lstm.weight_ih_l{}", lstm.len())) {
            let l = lstm.len();
            let p = "decoder.prediction.dec_rnn.lstm";
            let (wih, s1) = w.f32(&format!("{p}.weight_ih_l{l}"))?;
            let (whh, s2) = w.f32(&format!("{p}.weight_hh_l{l}"))?;
            let (bih, _) = w.f32(&format!("{p}.bias_ih_l{l}"))?;
            let (bhh, _) = w.f32(&format!("{p}.bias_hh_l{l}"))?;
            if s1 != [4 * hidden, hidden] || s2 != [4 * hidden, hidden] || bih.len() != 4 * hidden || bhh.len() != 4 * hidden {
                return Err(bad(format!("LSTM layer {l}: shapes {s1:?} {s2:?}")));
            }
            let mut cat = Vec::with_capacity(8 * hidden * hidden);
            for r in 0..4 * hidden {
                cat.extend_from_slice(&wih[r * hidden..(r + 1) * hidden]);
                cat.extend_from_slice(&whh[r * hidden..(r + 1) * hidden]);
            }
            let weight = Tensor::from_vec(cat, (4 * hidden, 2 * hidden), &Device::Cpu)?.to_dtype(dtype)?.to_device(dev)?;
            let bias: Vec<f32> = bih.iter().zip(&bhh).map(|(a, b)| a + b).collect();
            lstm.push(Lstm { w: Linear::from_parts(weight, Some(Tensor::from_vec(bias, 4 * hidden, dev)?)), hidden });
        }
        if lstm.is_empty() {
            return Err(bad("no prediction LSTM in the checkpoint"));
        }
        let joint = w.shape("joint.pred.weight")?[0];
        let outputs = w.shape("joint.joint_net.2.weight")?[0];
        let pred = Linear::load(w, "joint.pred", joint, hidden, dtype, dev)?;
        let enc = Linear::load(w, "joint.enc", joint, d_model, dtype, dev)?;
        let out = Linear::load(w, "joint.joint_net.2", outputs, joint, dtype, dev)?;
        let shape = DecoderShape { tokens, hidden, layers: lstm.len(), joint, outputs };
        Ok(Self { embed, lstm, pred, enc, out, shape, dev: dev.clone() })
    }

    /// The blank token: the last embedding row (`blank_as_pad`).
    pub fn blank(&self) -> u32 {
        (self.shape.tokens - 1) as u32
    }

    /// Run the prediction network on `input` `(1, hidden)` from `state`
    /// (zeros when `None`).
    fn predict(&self, input: &Tensor, state: Option<&PredState>) -> Result<PredState> {
        let zeros = Tensor::zeros((1, self.shape.hidden), DType::F32, &self.dev)?;
        let mut x = input.clone();
        let mut layers = Vec::with_capacity(self.lstm.len());
        for (l, cell) in self.lstm.iter().enumerate() {
            let (h, c) = match state {
                Some(s) => (s.layers[l].0.clone(), s.layers[l].1.clone()),
                None => (zeros.clone(), zeros.clone()),
            };
            let (h, c) = cell.step(&x, &h, &c)?;
            x = h.clone();
            layers.push((h, c));
        }
        Ok(PredState { layers, projected: self.pred.forward(&x)? })
    }

    /// The state before any token: NeMo feeds a zero vector (start of
    /// sequence) into a zero state.
    pub fn start(&self) -> Result<PredState> {
        self.predict(&Tensor::zeros((1, self.shape.hidden), DType::F32, &self.dev)?, None)
    }

    /// The state after `token`.
    pub fn next(&self, state: &PredState, token: u32) -> Result<PredState> {
        let x = self.embed.index_select(&Tensor::new(&[token], &self.dev)?, 0)?;
        self.predict(&x, Some(state))
    }

    /// Encoder frames `(T, d_model)` projected to the joint's width.
    pub fn project_encoder(&self, enc: &Tensor) -> Result<Tensor> {
        self.enc.forward(enc)
    }

    /// Joint outputs for projected encoder frames `(n, joint)` and a state.
    pub fn joint(&self, enc_proj: &Tensor, state: &PredState) -> Result<Tensor> {
        self.out.forward(&enc_proj.broadcast_add(&state.projected)?.relu()?)
    }
}

/// Answers [`Scorer`] queries from the joint on the model's device, a block
/// of frames at a time.
pub struct DeviceScorer<'a> {
    dec: &'a Decoder,
    enc_proj: Tensor,
    frames: usize,
    state: PredState,
    /// Token outputs (with blank) before the duration outputs.
    token_outputs: usize,
    block: usize,
    cache_start: usize,
    cache: Vec<(u32, usize)>,
    /// Device round trips made (for measurement).
    pub trips: usize,
}

impl<'a> DeviceScorer<'a> {
    pub fn new(dec: &'a Decoder, enc: &Tensor, token_outputs: usize, block: usize) -> Result<Self> {
        let enc_proj = dec.project_encoder(enc)?;
        let frames = enc_proj.dim(0)?;
        Ok(Self { dec, enc_proj, frames, state: dec.start()?, token_outputs, block: block.max(1), cache_start: 0, cache: Vec::new(), trips: 0 })
    }
}

impl Scorer for DeviceScorer<'_> {
    fn best(&mut self, frame: usize) -> Result<(u32, usize)> {
        if frame < self.cache_start || frame >= self.cache_start + self.cache.len() {
            let n = self.block.min(self.frames - frame);
            let logits = self.dec.joint(&self.enc_proj.narrow(0, frame, n)?, &self.state)?;
            let outputs = logits.dim(1)?;
            let tokens = logits.narrow(1, 0, self.token_outputs)?.argmax_keepdim(1)?;
            let best = if outputs > self.token_outputs {
                let durations = logits.narrow(1, self.token_outputs, outputs - self.token_outputs)?.argmax_keepdim(1)?;
                Tensor::cat(&[&tokens, &durations], 1)?
            } else {
                Tensor::cat(&[&tokens, &tokens.zeros_like()?], 1)?
            };
            let best = best.to_vec2::<u32>()?;
            self.trips += 1;
            self.cache_start = frame;
            self.cache = best.into_iter().map(|r| (r[0], r[1] as usize)).collect();
        }
        Ok(self.cache[frame - self.cache_start])
    }

    fn emit(&mut self, token: u32) -> Result<()> {
        self.state = self.dec.next(&self.state, token)?;
        self.cache.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scorer from a table: per (frame, number of tokens emitted so far).
    struct Table {
        rows: Vec<Vec<(u32, usize)>>,
        emitted: usize,
        calls: usize,
    }

    impl Scorer for Table {
        fn best(&mut self, frame: usize) -> Result<(u32, usize)> {
            self.calls += 1;
            let row = &self.rows[frame];
            Ok(row[self.emitted.min(row.len() - 1)])
        }
        fn emit(&mut self, _token: u32) -> Result<()> {
            self.emitted += 1;
            Ok(())
        }
    }

    const BLANK: u32 = 9;

    #[test]
    fn rnnt_emits_until_blank_then_advances() {
        // Frame 0: token 1 then blank; frame 1: blank; frame 2: tokens 2, 3 then blank.
        let mut s = Table { rows: vec![vec![(1, 0), (BLANK, 0)], vec![(BLANK, 0)], vec![(BLANK, 0), (2, 0), (3, 0), (BLANK, 0)]], emitted: 0, calls: 0 };
        // Frame 2 is reached with one token emitted: row index 1 is token 2.
        assert_eq!(greedy_rnnt(3, BLANK, 10, &mut s).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn rnnt_caps_symbols_per_frame() {
        let mut s = Table { rows: vec![vec![(4, 0)], vec![(BLANK, 0)]], emitted: 0, calls: 0 };
        assert_eq!(greedy_rnnt(2, BLANK, 3, &mut s).unwrap(), vec![4, 4, 4]);
    }

    #[test]
    fn tdt_advances_by_durations() {
        let durations = [0, 1, 2, 3, 4];
        // Frame 0: token 5 staying (duration index 0), then token 6 moving 2
        // frames; frame 2: blank moving 3; frame 5: past the end (5 frames).
        let mut s = Table { rows: vec![vec![(5, 0), (6, 2)], vec![], vec![(BLANK, 3), (BLANK, 3), (BLANK, 3)], vec![], vec![]], emitted: 0, calls: 0 };
        assert_eq!(greedy_tdt(5, BLANK, &durations, 10, &mut s).unwrap(), vec![5, 6]);
        assert_eq!(s.calls, 3);
    }

    #[test]
    fn tdt_blank_with_zero_duration_leaves_after_the_cap() {
        // NeMo loops on the frame until max_symbols, then moves one frame on.
        let mut s = Table { rows: vec![vec![(BLANK, 0)], vec![(7, 1)]], emitted: 0, calls: 0 };
        assert_eq!(greedy_tdt(2, BLANK, &[0, 1], 4, &mut s).unwrap(), vec![7]);
        assert_eq!(s.calls, 5);
    }

    #[test]
    fn tdt_cap_adds_a_frame_even_after_a_move() {
        // max_symbols 2: token (stay), token (move 1) hits the cap, so the
        // frame index moves 1 + 1 and frame 1 is never scored.
        let mut s = Table { rows: vec![vec![(1, 0), (2, 1)], vec![(3, 1), (3, 1), (3, 1)], vec![(BLANK, 1), (BLANK, 1), (BLANK, 1)]], emitted: 0, calls: 0 };
        assert_eq!(greedy_tdt(3, BLANK, &[0, 1], 2, &mut s).unwrap(), vec![1, 2]);
    }
}
