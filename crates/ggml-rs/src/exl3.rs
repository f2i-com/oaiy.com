//! VENDORED-LOCAL: original EXL3 mul1 trellis storage (including half bitrates).
//! Layout follows exllamav3's pack/reconstruct kernels; no requantization.
use crate::Tensor;

pub trait PackedLinear: std::fmt::Debug + Send + Sync {
    fn shape(&self) -> &[usize];
    fn linear(&self, x: &Tensor) -> Tensor;
    fn nbytes(&self) -> usize;
    /// VENDORED-LOCAL: the concrete projection, for a backend that runs it in a chain of its own (none by default).
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// A MoE layer's EXL3 experts, the shared one last: `forward(x, logits, top_k)` routes each row of `x` (`[rows,
/// hidden]`) by its router logits (`[rows, experts + 1]`, the shared expert's gate last) to its `top_k` experts,
/// softmax-weighted, plus the shared expert weighted by the sigmoid of its gate, and sums what they give (`[rows,
/// hidden]`). Each expert is `down(silu(gate(x)) * up(x))`. On CUDA (`ggml_rs_cuda::exl3::Exl3Experts`), or on any
/// GPU through WebGPU and the CPU (`ggml_rs_wgpu::exl3`).
pub trait Experts: std::fmt::Debug + Send + Sync {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor;
    /// VENDORED-LOCAL: the concrete experts, for a backend that runs them in a chain of its own (none by default).
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
    /// VENDORED-LOCAL: whether these run on the host (a layer no GPU had room for): `forward` then takes host tensors,
    /// touches no device, and gives a row the same sum whatever rows are beside it, so a chain may call it between a
    /// layer's submits (the layer's router and input read back, the sum uploaded). Not by default.
    fn on_host(&self) -> bool {
        false
    }
    /// VENDORED-LOCAL: whether the device holds only some of these and reads the others from the host's memory as
    /// it runs (`ggml_rs_wgpu::quant_moe::Cache`): a prompt's chunk then copies each expert it uses that the device
    /// lacks, once a chunk. Not by default.
    fn part_held(&self) -> bool {
        false
    }
    /// VENDORED-LOCAL: [`Self::forward`] of experts on the host whose shared expert a device ran
    /// (`ChainRecorder::moe_shared`): `shared` its outputs (`[rows, hidden]`, not yet weighted), added by its gate to
    /// the routed experts' sums. By default the shared expert is run again here.
    fn forward_given(&self, x: &Tensor, logits: &Tensor, top_k: usize, shared: &[f32]) -> Tensor {
        let _ = shared;
        self.forward(x, logits, top_k)
    }
    /// VENDORED-LOCAL: a low-rank update (a LoRA adapter's `B (A x)`) beside projection `which` (0 gate, 1 up, 2
    /// down) of these experts, from now on: `slot_of[e]` is the place of expert `e`'s pair (the shared expert last),
    /// `u32::MAX` where it has none, in `a` (`[slots, rank, k]`) and `b` (`[slots, n, rank]`, the adapter's scale in
    /// it). The base weights are not rewritten. An error where these experts take none (the default): an adapter's
    /// update must be applied or refused, never dropped.
    fn low_rank(&mut self, which: usize, slot_of: &[u32], a: &[f32], b: &[f32], rank: usize) -> Result<(), String> {
        let _ = (which, slot_of, a, b, rank);
        Err("these experts take no low-rank update".into())
    }
}

/// VENDORED-LOCAL: one row's experts and their weights, as the CUDA routing gives them: the `top_k` of the routed
/// experts by logit (a tie to the lower index), softmax-weighted among themselves, then the shared expert (index
/// `logits.len() - 1`) weighted by the sigmoid of its gate.
pub fn route(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let routed = logits.len() - 1;
    let k = top_k.min(routed);
    // the top k in one pass (a prompt routes each of its rows at every layer): the list kept by logit, a tie after the
    // lower index already in it, as the experts come in order
    let mut top: Vec<usize> = Vec::with_capacity(k + 1);
    for (e, &v) in logits[..routed].iter().enumerate() {
        if top.len() == k && (k == 0 || v <= logits[top[k - 1]]) {
            continue;
        }
        let at = top.partition_point(|&o| logits[o] >= v);
        top.insert(at, e);
        top.truncate(k);
    }
    let max = logits[top[0]];
    let sum: f32 = top.iter().map(|&e| (logits[e] - max).exp()).sum();
    let mut out: Vec<(usize, f32)> = top.iter().map(|&e| (e, (logits[e] - max).exp() / sum)).collect();
    out.push((routed, 1.0 / (1.0 + (-logits[routed]).exp())));
    out
}

#[derive(Debug)]
pub struct Exl3Data {
    pub words: Vec<u32>,
    pub suh: Vec<f32>,
    pub svh: Vec<f32>,
    /// Number of uint16 words in a 16x16 tile: 48=3, 56=3.5, 64=4.
    pub tile_words: usize,
    /// Original input channel -> caller's input channel.
    pub input_map: Vec<u32>,
    /// Caller's output channel -> original output channel.
    pub output_map: Vec<u32>,
}

impl Exl3Data {
    pub fn validate(&self) -> Result<(), String> {
        let (k, n) = (self.suh.len(), self.svh.len());
        if k.checked_mul(n).is_none_or(|v| v > u32::MAX as usize) {
            return Err("EXL3 projection exceeds CUDA indexing capacity".into());
        }
        if k == 0
            || n == 0
            || k % 128 != 0
            || n % 128 != 0
            || !matches!(
                self.tile_words,
                16 | 24 | 32 | 40 | 48 | 56 | 64 | 80 | 96 | 112 | 128
            )
        {
            return Err("unsupported EXL3 dimensions or bitrate".into());
        }
        let len = (k / 16)
            .checked_mul(n / 16)
            .and_then(|v| v.checked_mul(self.tile_words / 2));
        if len != Some(self.words.len()) {
            return Err("EXL3 packed byte count disagrees with shape".into());
        }
        for (map, dim) in [(&self.input_map, k), (&self.output_map, n)] {
            let mut sorted = map.clone();
            sorted.sort_unstable();
            if sorted.len() != dim || sorted.iter().enumerate().any(|(i, &v)| i != v as usize) {
                return Err("EXL3 channel map is not a permutation".into());
            }
        }
        if self.suh.iter().chain(&self.svh).any(|v| !v.is_finite()) {
            return Err("nonfinite EXL3 scale".into());
        }
        Ok(())
    }

    /// Reference scalar reconstruction in the Hadamard basis, W[k,n].
    pub fn value(&self, k: usize, n: usize) -> f32 {
        let (r, c) = (k % 16, n % 16);
        let lane = (r % 8 / 2) + 4 * (c % 8);
        let j = (r % 2) + 2 * (r / 8) + 4 * (c / 8);
        let i = lane * 8 + j;
        let bits = self.tile_words / 16;
        let end = (i + 1) * bits
            + if self.tile_words % 16 == 8 {
                (i + 1) / 2
            } else {
                0
            };
        let nw = self.tile_words / 2;
        let start = (end + nw * 32 - 16) % (nw * 32);
        let base = ((k / 16) * (self.svh.len() / 16) + n / 16) * nw;
        let a = self.words[base + start / 32] as u64;
        let b = self.words[base + (start / 32 + 1) % nw] as u64;
        let idx = (((a << 32) | b) >> (48 - start % 32)) as u32 & 65535;
        mul1(idx)
    }
}

pub fn mul1(index: u32) -> f32 {
    let x = index.wrapping_mul(0x83dcd12d);
    let sum = x.to_le_bytes().iter().map(|&v| v as u16).sum::<u16>();
    let a = half::f16::from_bits(0x6400 + sum).to_f32();
    let inv = half::f16::from_bits(0x1eee).to_f32();
    let bias = half::f16::from_bits(0xc931).to_f32();
    half::f16::from_f32(a.mul_add(inv, bias)).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_packed_storage_and_maps_are_errors() {
        let mut d = Exl3Data {
            words: vec![0; 128 / 16 * 128 / 16 * 24],
            suh: vec![1.0; 128],
            svh: vec![1.0; 128],
            tile_words: 48,
            input_map: (0..128).collect(),
            output_map: (0..128).collect(),
        };
        assert!(d.validate().is_ok());
        d.words.pop();
        assert!(d.validate().is_err());
        d.words.push(0);
        d.input_map[1] = 0;
        assert!(d.validate().is_err());
        d.input_map[1] = 1;
        d.suh[0] = f32::NAN;
        assert!(d.validate().is_err());
        d.suh[0] = 1.0;
        d.tile_words = 52;
        assert!(d.validate().is_err());
    }
}

#[cfg(test)]
mod route_tests {
    use super::route;

    /// The routing's top k against a full sort's (by logit, a tie to the lower index), ties and all.
    #[test]
    fn routing_picks_what_a_full_sort_picks() {
        let mut s = 12_345u32;
        for case in 0..400 {
            let n = [9usize, 65, 513][case % 3];
            // few distinct values in some cases, so ties are common
            let levels = [3u32, 50, 1 << 20][case % 3];
            let logits: Vec<f32> = (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (s >> 8) as f32 % levels as f32 / levels as f32 * 8.0 - 4.0
                })
                .collect();
            for top_k in [1, 2, 8, 10] {
                let mut order: Vec<usize> = (0..n - 1).collect();
                order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
                let want: Vec<usize> = order[..top_k.min(n - 1)].to_vec();
                let got: Vec<usize> = route(&logits, top_k).iter().map(|p| p.0).collect();
                assert_eq!(&got[..got.len() - 1], &want[..], "case {case}, top {top_k}");
                assert_eq!(got[got.len() - 1], n - 1, "the shared expert last");
            }
        }
    }
}
