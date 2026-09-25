//! VENDORED-LOCAL: original EXL3 mul1 trellis storage (including half bitrates).
//! Layout follows exllamav3's pack/reconstruct kernels; no requantization.
use crate::Tensor;

pub trait PackedLinear: std::fmt::Debug + Send + Sync {
    fn shape(&self) -> &[usize];
    fn linear(&self, x: &Tensor) -> Tensor;
    fn nbytes(&self) -> usize;
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
