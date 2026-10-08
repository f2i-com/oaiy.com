//! A MoE layer's experts as a GGUF holds them (Qwen3.8-Flash-Next's GSQ-RCO files: 512 routed experts a layer, their
//! gate, up and down matrices each one tensor of quant blocks), beside [`crate::exl3`]'s EXL3 ones and run as those
//! are: a row's experts a job list (job `j` matrix `jobs[2j]` on input row `jobs[2j + 1]`, its result row `j`), routed
//! on the host or on the GPU ([`crate::exl3::record_route`]), grouped by expert for a prompt's rows, each row's weighted sum
//! made on the GPU. What differs is the matmul: a GGUF's weights are the weights (no Hadamard transforms, no channel
//! maps). The tensor cores' kernel decodes them by one function a type ([`Kind::wgsl`]'s `w8`: eight weights of a
//! row); the kernel for a step's row and a check's few ([`few_source`]) is written for its type, a word's weights as
//! vectors. Another type is its decode function, its rows' layout on the GPU and, for a lookup-table type, its grid.
//!
//! Q2_0 (ggml type 42: 64 weights a block of 18 bytes, an f16 scale then 2 bits a weight, a weight `(code - 1) *
//! scale`) is held a row at a time: its codes' words (16 weights a word, weight `i` its bits `2 i`), then its blocks'
//! scales, f16, two a word: the GGUF's bytes, no more, each word aligned.
//!
//! The grid types the GSQ-RCO IQ2_XS file keeps its experts' gate and up matrices in (IQ2_S, IQ2_XXS, IQ1_M: 256
//! weights a block, a group of eight an entry of ggml's grid, with its signs and a scale of its own) are held a block
//! in whole words, its fields each on a word's edge (a 32's four index bytes one word, its signs another: the GGUF's
//! bytes in another order, and a pad). The grid is a storage buffer of the layer's (binding 4), two bits a weight (a
//! grid's bytes take four values at most), so a group is one load of it: as WGSL constants such tables are copied at
//! each call, which once ran a dispatch past Windows' two seconds and reset the driver (`crate::shaders::layout`'s
//! note).
use crate::exl3::{coop_on, many_order, moe_rows_for, record_route, Step, FEW_MAX, GROUP, MANY_CLEAR, MANY_COUNT, MANY_SCAN, MANY_SCATTER, WSUM_APPLY, WSUM_ROWS};
use crate::{chunk_limit, Gpu, WgpuBackend};
use ggml_quants::GgmlType;
use ggml_rs::exl3::{route, Experts};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, Tensor};
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// One MoE layer's experts as a GGUF holds them.
pub struct QuantExpertsData {
    pub hidden: usize,
    pub ff: usize,
    /// The routed experts.
    pub experts: usize,
    /// Routed gate and up: `[experts][ff rows][hidden cols]`; down: `[experts][hidden rows][ff cols]`. Raw block
    /// bytes, row-major, each row a whole number of blocks (a GGUF's `ffn_gate_exps` of 2560 x 640 x 512 and so on).
    pub gate: (GgmlType, Vec<u8>),
    pub up: (GgmlType, Vec<u8>),
    pub down: (GgmlType, Vec<u8>),
    /// The shared expert, dequantised by the loader: gate and up `[ff, hidden]`, down `[hidden, ff]`, f32 row-major.
    pub shared: [Vec<f32>; 3],
}

/// The bytes of a row of `cols` weights of type `t`.
fn row_bytes(t: GgmlType, cols: usize) -> usize {
    cols / t.block_size() * t.type_size()
}

impl QuantExpertsData {
    /// Each tensor's bytes are its shape's (a row a whole number of its type's blocks), of a type `ggml_quants` decodes.
    pub fn validate(&self) -> Result<(), String> {
        let (h, f, e) = (self.hidden, self.ff, self.experts);
        if h == 0 || f == 0 || e == 0 {
            return Err("experts of no size".into());
        }
        for (what, (t, bytes), rows, cols) in [("gate", &self.gate, f, h), ("up", &self.up, f, h), ("down", &self.down, h, f)] {
            if !ggml_quants::is_supported(*t) {
                return Err(format!("the experts' {what}: no decoder for {}", t.name()));
            }
            if cols % t.block_size() != 0 {
                return Err(format!("the experts' {what}: rows of {cols} are not whole {} blocks", t.name()));
            }
            let want = e * rows * row_bytes(*t, cols);
            if bytes.len() != want {
                return Err(format!("the experts' {what}: {} bytes, {want} for {e} of {rows} x {cols} in {}", bytes.len(), t.name()));
            }
        }
        for (what, m) in ["gate", "up", "down"].iter().zip(&self.shared) {
            if m.len() != h * f {
                return Err(format!("the shared expert's {what}: {} values, not {}", m.len(), h * f));
            }
        }
        Ok(())
    }
}

/// `y = W x`, `W` `[n, k]` row-major, summed in f64 (the reference's).
fn matvec(w: &[f32], k: usize, x: &[f32], y: &mut [f32]) {
    for (row, out) in w.chunks_exact(k).zip(y.iter_mut()) {
        *out = row.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// The experts on the host: the reference the GPU's are held against, and where they run without a GPU (or past its
/// budget). Each expert a call's rows share is dequantised once for them.
pub struct QuantMoeCpu {
    data: QuantExpertsData,
}

impl std::fmt::Debug for QuantMoeCpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantMoeCpu({} routed experts of {}x{} in {}, and the shared one, on the host)", self.data.experts, self.data.hidden, self.data.ff, self.data.gate.0.name())
    }
}

impl QuantMoeCpu {
    /// Matrix `e` of a tensor of `rows` by `cols` matrices, dequantised.
    fn matrix((t, bytes): &(GgmlType, Vec<u8>), e: usize, rows: usize, cols: usize) -> Vec<f32> {
        let each = rows * row_bytes(*t, cols);
        let mut out = vec![0f32; rows * cols];
        ggml_quants::dequantize(*t, &bytes[e * each..(e + 1) * each], &mut out).expect("a validated tensor");
        out
    }
}

impl Experts for QuantMoeCpu {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let d = &self.data;
        let (h, f, n) = (d.hidden, d.ff, d.experts);
        let x = x.to_host();
        let logits = logits.to_host();
        let (xs, ls) = (x.data(), logits.data());
        let rows = xs.len() / h;
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * (n + 1)..(r + 1) * (n + 1)], top_k)).collect();
        // each routed expert's rows (and their weights)
        let mut by: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (r, a) in assign.iter().enumerate() {
            for &(e, w) in &a[..a.len() - 1] {
                by[e].push((r, w));
            }
        }
        let expert = |g: &[f32], u: &[f32], dn: &[f32], xr: &[f32], w: f32| -> Vec<f32> {
            let (mut a, mut b, mut y) = (vec![0f32; f], vec![0f32; f], vec![0f32; h]);
            matvec(g, h, xr, &mut a);
            matvec(u, h, xr, &mut b);
            for (a, b) in a.iter_mut().zip(&b) {
                *a = silu(*a) * b;
            }
            matvec(dn, f, &a, &mut y);
            y.iter_mut().for_each(|v| *v *= w);
            y
        };
        let routed: Vec<(usize, Vec<f32>)> = by
            .par_iter()
            .enumerate()
            .filter(|(_, rs)| !rs.is_empty())
            .flat_map_iter(|(e, rs)| {
                let (g, u, dn) = (Self::matrix(&d.gate, e, f, h), Self::matrix(&d.up, e, f, h), Self::matrix(&d.down, e, h, f));
                rs.iter().map(|&(r, w)| (r, expert(&g, &u, &dn, &xs[r * h..(r + 1) * h], w))).collect::<Vec<_>>()
            })
            .collect();
        let mut out = vec![0f32; rows * h];
        for (r, y) in routed {
            out[r * h..(r + 1) * h].iter_mut().zip(&y).for_each(|(o, v)| *o += v);
        }
        // the shared expert on every row, weighted by its gate's sigmoid
        out.par_chunks_mut(h).enumerate().for_each(|(r, o)| {
            let y = expert(&d.shared[0], &d.shared[1], &d.shared[2], &xs[r * h..(r + 1) * h], assign[r].last().expect("the shared expert").1);
            o.iter_mut().zip(&y).for_each(|(o, v)| *o += v);
        });
        Tensor::from_vec(out, vec![rows, h])
    }

    fn on_host(&self) -> bool {
        true
    }
}

/// The experts of `data` on the host, the reference: [`crate::quant_host::quant_experts_host`] is what a model runs.
pub fn quant_experts_cpu(data: QuantExpertsData) -> Result<Box<dyn Experts>, String> {
    data.validate()?;
    Ok(Box::new(QuantMoeCpu { data }))
}

/// A weight type the GPU's kernels decode: its rows' layout there and its `w8`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Q2_0,
    IQ2_S,
    IQ2_XXS,
    IQ1_M,
}

impl Kind {
    fn of(t: GgmlType) -> Option<Kind> {
        match t {
            GgmlType::Q2_0 => Some(Kind::Q2_0),
            GgmlType::IQ2_S => Some(Kind::IQ2_S),
            GgmlType::IQ2_XXS => Some(Kind::IQ2_XXS),
            GgmlType::IQ1_M => Some(Kind::IQ1_M),
            _ => None,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Kind::Q2_0 => "q2_0",
            Kind::IQ2_S => "iq2_s",
            Kind::IQ2_XXS => "iq2_xxs",
            Kind::IQ1_M => "iq1_m",
        }
    }

    /// A grid type's block of 256 weights: its bytes in the GGUF, and the words it takes here (padded to whole ones).
    fn block(self) -> Option<(usize, usize)> {
        match self {
            Kind::Q2_0 => None,
            Kind::IQ2_S => Some((82, 21)),
            Kind::IQ2_XXS => Some((66, 17)),
            Kind::IQ1_M => Some((56, 14)),
        }
    }

    /// Whether a row of `k` weights is whole blocks of the type.
    fn fits(self, k: usize) -> bool {
        k % if self.block().is_some() { 256 } else { 64 } == 0
    }

    /// The words a row of `k` weights takes on the GPU.
    fn row_words(self, k: usize) -> usize {
        match self.block() {
            // a word 16 weights' codes, then a word two blocks' scales
            None => k / 16 + (k / 64).div_ceil(2),
            Some((_, words)) => k / 256 * words,
        }
    }

    /// A row's blocks (`src`, `k` weights as the GGUF has them) as the GPU holds it (`dst`, [`Self::row_words`] long).
    fn pack_row(self, src: &[u8], k: usize, dst: &mut [u32]) {
        match self.block() {
            None => {
                let kw = k / 16;
                dst[kw..].fill(0);
                for (b, block) in src.chunks_exact(18).enumerate() {
                    for (i, c) in block[2..].chunks_exact(4).enumerate() {
                        dst[4 * b + i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                    dst[kw + b / 2] |= (u16::from_le_bytes([block[0], block[1]]) as u32) << (16 * (b % 2));
                }
            }
            Some((bytes, words)) => {
                let le = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                for (block, out) in src.chunks_exact(bytes).zip(dst.chunks_exact_mut(words)) {
                    match self {
                        // the GGUF's: the scale (f16), 32 index bytes, 32 sign bytes, 8 bytes of the indices' high
                        // bits, 8 of scales; here the eight 32s' index words, their sign words, the high bits', the
                        // scales', the scale
                        Kind::IQ2_S => {
                            for s in 0..8 {
                                out[s] = le(&block[2 + 4 * s..]);
                                out[8 + s] = le(&block[34 + 4 * s..]);
                            }
                            (out[16], out[17], out[18], out[19]) = (le(&block[66..]), le(&block[70..]), le(&block[74..]), le(&block[78..]));
                            out[20] = u16::from_le_bytes([block[0], block[1]]) as u32;
                        }
                        // the GGUF's: the scale, then each 32's four index bytes and its word of signs and scale;
                        // here the index words, those words, the scale
                        Kind::IQ2_XXS => {
                            for s in 0..8 {
                                out[s] = le(&block[2 + 8 * s..]);
                                out[8 + s] = le(&block[6 + 8 * s..]);
                            }
                            out[16] = u16::from_le_bytes([block[0], block[1]]) as u32;
                        }
                        // (IQ1_M's fields lie on words' edges as they are: its bytes, little-endian)
                        _ => {
                            for (o, c) in out.iter_mut().zip(block.chunks_exact(4)) {
                                *o = le(c);
                            }
                        }
                    }
                }
            }
        }
    }

    /// A grid type's grid: ggml's, and whether its bytes are int8 (IQ1's -1, 0 and 1; the others' are magnitudes).
    fn grid(self) -> Option<(&'static [u64], bool)> {
        use ggml_quants::iq_tables as t;
        match self {
            Kind::Q2_0 => None,
            Kind::IQ2_S => Some((&t::IQ2S_GRID, false)),
            Kind::IQ2_XXS => Some((&t::IQ2XXS_GRID, false)),
            Kind::IQ1_M => Some((&t::IQ1S_GRID, true)),
        }
    }

    /// The table a grid type's kernels read (its storage buffer's words): the grid's entries, a word each
    /// ([`grid_codes`]).
    fn table(self) -> Option<Vec<u32>> {
        self.grid().map(|(g, signed)| grid_codes(g, signed).0)
    }

    /// The type's decoder. Q2_0: `fn w8(rb: u32, kt: u32, hf: u32, kw: u32) -> array<f32, 8>`, weights `16 kt + 8 hf
    /// ..` (eight of them) of the row whose words start at `rb` of `words`, a row of `16 kw` weights. A grid type:
    /// `fn sub_of(bw: u32, sub: u32) -> Sub`, what the 32 weights `32 sub ..` of the block at `bw` share, and `fn
    /// grp(h: Sub, l: u32) -> array<f32, 8>`, its group `l` of four; `BLOCK_WORDS`; and the grid's binding.
    fn wgsl(self) -> String {
        let Some((grid, signed)) = self.grid() else { return W8_Q2_0.to_string() };
        let helpers = GRID_HELPERS.replace("LEVELS", &levels_wgsl(grid_codes(grid, signed).1));
        match self {
            Kind::IQ2_S => format!("{helpers}{GRID_MAGS}{SUB_IQ2_S}"),
            Kind::IQ2_XXS => format!("{helpers}{GRID_MAGS}{SUB_IQ2_XXS}"),
            _ => format!("{helpers}{SUB_IQ1_M}"),
        }
    }
}

/// A grid's entries as the kernels read them: two bits a weight, eight weights a word (weight `j` its bits `2 j`),
/// each one of the grid's values, which are four at most (returned in order: a code is its value's place among them;
/// the last repeated where they are fewer). `signed`: the grid's bytes are int8.
fn grid_codes(grid: &[u64], signed: bool) -> (Vec<u32>, [f32; 4]) {
    let value = |b: u8| if signed { b as i8 as f32 } else { b as f32 };
    let mut seen: Vec<u8> = grid.iter().flat_map(|e| e.to_le_bytes()).collect();
    seen.sort_unstable_by(|a, b| value(*a).total_cmp(&value(*b)));
    seen.dedup();
    assert!(seen.len() <= 4, "a grid of {} values: two bits a weight hold four", seen.len());
    let code = |b: &u8| seen.iter().position(|s| s == b).expect("a value of the grid's") as u32;
    let codes = grid.iter().map(|e| e.to_le_bytes().iter().enumerate().fold(0u32, |w, (j, b)| w | code(b) << (2 * j))).collect();
    let mut levels = [value(*seen.last().expect("a grid has values")); 4];
    for (l, b) in levels.iter_mut().zip(&seen) {
        *l = value(*b);
    }
    (codes, levels)
}

/// A grid's values from their codes `c` (a `vec4<f32>` of them), as WGSL: `l0 + c (a + b c)` where that gives each
/// value exactly in f32 (ggml's grids: 8, 25, 43 and 62 are `8 + c (16.5 + c / 2)`, IQ1's -1, 0 and 1 `c - 1`), two
/// multiply-adds a weight; else by comparing (five steps a weight, which was a third of the few-row kernel's time).
fn levels_wgsl(l: [f32; 4]) -> String {
    let b = (l[2] - 2.0 * l[1] + l[0]) / 2.0;
    let a = l[1] - l[0] - b;
    if (0..4).all(|c| l[0] + c as f32 * (a + b * c as f32) == l[c]) {
        format!("vec4<f32>({:?}) + c * (vec4<f32>({a:?}) + {b:?} * c)", l[0])
    } else {
        format!("select(select(vec4<f32>({:?}), vec4<f32>({:?}), c == vec4<f32>(1.0)), select(vec4<f32>({:?}), vec4<f32>({:?}), c == vec4<f32>(3.0)), c >= vec4<f32>(2.0))", l[0], l[1], l[2], l[3])
    }
}

/// What the grid types' decoders share: the grid's buffer, what a 32 of a block's weights share, and a grid entry's
/// values (`LEVELS` the grid's four from a code, written in by [`Kind::wgsl`]: [`levels_wgsl`]).
const GRID_HELPERS: &str = r#"
@group(0) @binding(4) var<storage, read> grid: array<u32>;

// what a block's 32 weights share: their four groups' grid indices' low bytes, their signs, the scale of each half
// (groups 0 and 1, 2 and 3), and the type's own bits
struct Sub {
    idx: u32,
    sg: u32,
    s0: f32,
    s1: f32,
    x: u32,
}

// four of a grid entry's weights: two bits each (`e` shifted by `sh`), a weight one of the grid's four values
fn levels(e: u32, sh: vec4<u32>) -> vec4<f32> {
    let c = vec4<f32>((vec4<u32>(e) >> sh) & vec4<u32>(3u));
    return LEVELS;
}
"#;

/// A group's eight weights of the types whose grids are magnitudes (IQ2_S, IQ2_XXS).
const GRID_MAGS: &str = r#"
// a group's eight weights: its grid entry's values, each times `s`, negative where its bit of `signs` is set
// (a sign is its bit moved into the float's own: the product negated, exactly)
fn mags(e: u32, signs: u32, s: f32) -> array<f32, 8> {
    let m0 = bitcast<vec4<u32>>(s * levels(e, vec4<u32>(0u, 2u, 4u, 6u)));
    let m1 = bitcast<vec4<u32>>(s * levels(e, vec4<u32>(8u, 10u, 12u, 14u)));
    let v0 = bitcast<vec4<f32>>(m0 ^ ((vec4<u32>(signs) << vec4<u32>(31u, 30u, 29u, 28u)) & vec4<u32>(0x80000000u)));
    let v1 = bitcast<vec4<f32>>(m1 ^ ((vec4<u32>(signs) << vec4<u32>(27u, 26u, 25u, 24u)) & vec4<u32>(0x80000000u)));
    return array<f32, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}
"#;

const SUB_IQ2_S: &str = r#"
// IQ2_S, a block 21 words here: the eight 32s' index words (a group's grid index's low byte each), their sign words
// (a byte a group), two words of the indices' high bits (a byte a 32: two bits a group), two of scales (a byte a 32:
// a nibble for each half), and the block's f16 scale.
const BLOCK_WORDS: u32 = 21u;
fn sub_of(bw: u32, sub: u32) -> Sub {
    let sh = 8u * (sub % 4u);
    let sc = (words[bw + 18u + sub / 4u] >> sh) & 255u;
    let d = unpack2x16float(words[bw + 20u]).x;
    return Sub(words[bw + sub], words[bw + 8u + sub], d * (0.5 + f32(sc & 15u)) * 0.25, d * (0.5 + f32(sc >> 4u)) * 0.25, (words[bw + 16u + sub / 4u] >> sh) & 255u);
}
fn grp(h: Sub, l: u32) -> array<f32, 8> {
    let i = ((h.idx >> (8u * l)) & 255u) | ((h.x << (8u - 2u * l)) & 0x300u);
    return mags(grid[i], (h.sg >> (8u * l)) & 255u, select(h.s0, h.s1, l >= 2u));
}
"#;

const SUB_IQ2_XXS: &str = r#"
// IQ2_XXS, a block 17 words here: the eight 32s' index words (a group's grid index each), then for each 32 a word of
// its groups' sign patterns (seven bits each: a group's eighth sign makes its negatives even) under the 32's scale in
// its top four bits, and the block's f16 scale.
const BLOCK_WORDS: u32 = 17u;
fn sub_of(bw: u32, sub: u32) -> Sub {
    let aux = words[bw + 8u + sub];
    let s = unpack2x16float(words[bw + 16u]).x * (0.5 + f32(aux >> 28u)) * 0.25;
    return Sub(words[bw + sub], aux, s, s, 0u);
}
fn grp(h: Sub, l: u32) -> array<f32, 8> {
    let s7 = (h.sg >> (7u * l)) & 127u;
    return mags(grid[(h.idx >> (8u * l)) & 255u], s7 | ((countOneBits(s7) & 1u) << 7u), h.s0);
}
"#;

const SUB_IQ1_M: &str = r#"
// IQ1_M, a block 14 words (the GGUF's 56 bytes as they lie): the eight 32s' index words (a group's grid index's low
// byte each), four words of the indices' high bits and the offsets' signs (a nibble a group), and four 16-bit words:
// twelve bits of scales each (three for each 16 weights) under a nibble of the block's f16 scale. A weight is its grid
// value (-1, 0 or 1) plus or minus an eighth, times its scale.
const BLOCK_WORDS: u32 = 14u;
fn sub_of(bw: u32, sub: u32) -> Sub {
    let s0 = words[bw + 12u] & 0xffffu;
    let s1 = words[bw + 12u] >> 16u;
    let s2 = words[bw + 13u] & 0xffffu;
    let s3 = words[bw + 13u] >> 16u;
    let d = unpack2x16float((s0 >> 12u) | ((s1 >> 8u) & 0x00f0u) | ((s2 >> 4u) & 0x0f00u) | (s3 & 0xf000u)).x;
    let pair = sub / 2u;
    let sc = select(select(s0, s1, pair == 1u), select(s2, s3, pair == 3u), pair >= 2u) >> (6u * (sub % 2u));
    return Sub(words[bw + sub], 0u, d * f32(2u * (sc & 7u) + 1u), d * f32(2u * ((sc >> 3u) & 7u) + 1u), (words[bw + 8u + sub / 2u] >> (16u * (sub % 2u))) & 0xffffu);
}
fn grp(h: Sub, l: u32) -> array<f32, 8> {
    let hn = (h.x >> (4u * l)) & 15u;
    let e = grid[((h.idx >> (8u * l)) & 255u) | ((hn & 7u) << 8u)];
    let delta = select(0.125, -0.125, (hn & 8u) != 0u);
    let dl = select(h.s0, h.s1, l >= 2u);
    let v0 = dl * (levels(e, vec4<u32>(0u, 2u, 4u, 6u)) + delta);
    let v1 = dl * (levels(e, vec4<u32>(8u, 10u, 12u, 14u)) + delta);
    return array<f32, 8>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
}
"#;

const W8_Q2_0: &str = r#"
// Q2_0: a row's codes (16 weights a word, weight i its bits 2 i: -1, 0, 1 or 2 times its block's scale), then its
// blocks' scales (64 weights a block), f16, two a word.
fn w8(rb: u32, kt: u32, hf: u32, kw: u32) -> array<f32, 8> {
    let bits = words[rb + kt] >> (16u * hf);
    let d2 = unpack2x16float(words[rb + kw + kt / 8u]);
    let d = select(d2.x, d2.y, ((kt >> 2u) & 1u) == 1u);
    return array<f32, 8>(
        (f32(bits & 3u) - 1.0) * d,
        (f32((bits >> 2u) & 3u) - 1.0) * d,
        (f32((bits >> 4u) & 3u) - 1.0) * d,
        (f32((bits >> 6u) & 3u) - 1.0) * d,
        (f32((bits >> 8u) & 3u) - 1.0) * d,
        (f32((bits >> 10u) & 3u) - 1.0) * d,
        (f32((bits >> 12u) & 3u) - 1.0) * d,
        (f32((bits >> 14u) & 3u) - 1.0) * d
    );
}
"#;

/// Lanes an output row is shared out to in [`few_source`]'s kernel, for rows of `k` weights: 8, 16 or 32, some five
/// words (80 weights) a lane or more. A lane's work must outweigh what every thread does whatever its share (its
/// jobs' numbers, the barrier, its part in the sum): the experts' down projection is 640 wide, 40 words, and 32 lanes
/// of it had a word or two each.
///
/// A grid type's lanes take a group of eight weights at a turn, ten such a lane or more (the same 80 weights; 64 and
/// 128 lanes of Flash-Next's 2,560 were slower: a check's four rows 1.78 ms a step's layers with 64 where 1.53).
fn few_lanes(kind: Kind, k: usize) -> usize {
    let turn = if kind.block().is_some() { 8 } else { 16 };
    (k / turn / (80 / turn)).next_power_of_two().clamp(8, 32)
}

/// The matmul of blocks of `rows` (1 to [`FEW_MAX`]) jobs of one matrix, in f32: a workgroup a block's `256 / lanes`
/// output rows, `lanes` threads a row ([`few_lanes`]), each taking a word (16 weights) in every `lanes` of the row, so
/// the threads of a row read side by side; a word's weights decoded once for the block's rows as four vectors, x four
/// at a load, the row's sum the lanes' behind a barrier. A block's jobs come from `order` (its unused places
/// `0xffffffff`, a block with none ends at once), or with `p[0].w` block `b` is job `b` alone (a step's jobs, no
/// order). Job `j`'s sums to `y[j * n..]`. `p[0]`: n, k, words a row, whether the order is the identity; `p[1]`: words
/// a matrix, the pass's first block.
///
/// The first kernel gave an output row to 16 threads by sixteenths of its width, each decoding eight weights at a
/// call into an array and reading x one at a time: a Flash-Next layer's ten experts for a step's one row took 77 us
/// (1.3 TFLOPS), and this one 0.049 with 32 lanes for every width.
fn few_source(kind: Kind, rows: usize, lanes: usize) -> String {
    assert!((1..=FEW_MAX).contains(&rows), "a block of 1 to {FEW_MAX} rows");
    assert!(matches!(lanes, 8 | 16 | 32), "8, 16 or 32 lanes a row");
    if kind != Kind::Q2_0 {
        return few_source_grid(kind, rows, lanes);
    }
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<String>();
    let ids = each(&|i| {
        format!(
            "    var j{i} = blk;\n    if (!identity) {{ j{i} = order[blk * {rows}u + {i}u]; }}\n    let on{i} = j{i} != 0xffffffffu;\n    let xb{i} = jobs[2u * select(j{i}, 0u, !on{i}) + 1u] * (k / 4u);\n    var a{i} = 0.0;\n"
        )
    });
    let sums = each(&|i| {
        format!("            if (on{i}) {{ a{i} += d * (dot(c0, x[xb{i} + at]) + dot(c1, x[xb{i} + at + 1u]) + dot(c2, x[xb{i} + at + 2u]) + dot(c3, x[xb{i} + at + 3u])); }}\n")
    });
    let store = each(&|i| format!("    red[t * {rows}u + {i}u] = a{i};\n"));
    // (a row of the block's sum by a lane of its own: lane i adds row i's, the lanes in order)
    let out = each(&|i| format!("        if (lane == {i}u && on{i}) {{\n            var s = 0.0;\n            for (var q = 0u; q < {lanes}u; q++) {{ s += red[(t - {i}u + q) * {rows}u + {i}u]; }}\n            y[j{i} * n + row] = s;\n        }}\n"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// each thread's sums, [output row][lane][row of x]
var<workgroup> red: array<f32, {red_len}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let rw = p[0].z;
    let identity = p[0].w == 1u;
    let blk = p[1].y + wg.z;
    let slot = t / {lanes}u;
    let lane = t % {lanes}u;
{ids}    let row = (wg.x + wg.y * 65535u) * {per}u + slot;
    let live = on0 && row < n;
    let rb = jobs[2u * select(j0, 0u, !on0)] * p[1].x + min(row, n - 1u) * rw;
    let kw = k / 16u;
    if (live) {{
        for (var wi = lane; wi < kw; wi += {lanes}u) {{
            // Q2_0: a word 16 weights' codes (weight i its bits 2 i: -1, 0, 1 or 2 times its block's scale), a block
            // 64 weights, the blocks' scales f16 after the codes, two a word
            let w = vec4<u32>(words[rb + wi]);
            let d2 = unpack2x16float(words[rb + kw + wi / 8u]);
            let d = select(d2.x, d2.y, ((wi >> 2u) & 1u) == 1u);
            let c0 = vec4<f32>((w >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u)) - 1.0;
            let c1 = vec4<f32>((w >> vec4<u32>(8u, 10u, 12u, 14u)) & vec4<u32>(3u)) - 1.0;
            let c2 = vec4<f32>((w >> vec4<u32>(16u, 18u, 20u, 22u)) & vec4<u32>(3u)) - 1.0;
            let c3 = vec4<f32>((w >> vec4<u32>(24u, 26u, 28u, 30u)) & vec4<u32>(3u)) - 1.0;
            let at = wi * 4u;
{sums}        }}
    }}
{store}    workgroupBarrier();
    if (live) {{
{out}    }}
}}
"#,
        red_len = 256 * rows,
        per = 256 / lanes,
    )
}

/// [`few_source`] for a grid type: laid out the same, a lane taking a group of eight weights in every `lanes` of the
/// row (one grid entry, decoded once for the block's rows), x two fours at a load. What a group's decoding costs does
/// not show in this kernel's time: a Flash-Next layer's twenty IQ2_S matrices for a step's row take 18 us whether a
/// group is seven loads (bytes off words' edges, a grid entry two words) or six with less arithmetic, as now. A lane
/// taking 32 weights at a turn (their fields read once, their four groups written out one after another) was slower
/// for every count of lanes tried, 25 us and a check's four rows 86 where 45: the loop's body four times as long.
fn few_source_grid(kind: Kind, rows: usize, lanes: usize) -> String {
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<String>();
    let ids = each(&|i| {
        format!(
            "    var j{i} = blk;\n    if (!identity) {{ j{i} = order[blk * {rows}u + {i}u]; }}\n    let on{i} = j{i} != 0xffffffffu;\n    let xb{i} = jobs[2u * select(j{i}, 0u, !on{i}) + 1u] * (k / 4u);\n    var a{i} = 0.0;\n"
        )
    });
    let sums = each(&|i| format!("            if (on{i}) {{ a{i} += dot(c0, x[xb{i} + at]) + dot(c1, x[xb{i} + at + 1u]); }}\n"));
    let store = each(&|i| format!("    red[t * {rows}u + {i}u] = a{i};\n"));
    // (a row of the block's sum by a lane of its own: lane i adds row i's, the lanes in order)
    let out = each(&|i| format!("        if (lane == {i}u && on{i}) {{\n            var s = 0.0;\n            for (var q = 0u; q < {lanes}u; q++) {{ s += red[(t - {i}u + q) * {rows}u + {i}u]; }}\n            y[j{i} * n + row] = s;\n        }}\n"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// each thread's sums, [output row][lane][row of x]
var<workgroup> red: array<f32, {red_len}>;
{decoder}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let rw = p[0].z;
    let identity = p[0].w == 1u;
    let blk = p[1].y + wg.z;
    let slot = t / {lanes}u;
    let lane = t % {lanes}u;
{ids}    let row = (wg.x + wg.y * 65535u) * {per}u + slot;
    let live = on0 && row < n;
    let rb = jobs[2u * select(j0, 0u, !on0)] * p[1].x + min(row, n - 1u) * rw;
    if (live) {{
        for (var gi = lane; gi < k / 8u; gi += {lanes}u) {{
            let g = gi % 32u;
            let v = grp(sub_of(rb + (gi / 32u) * BLOCK_WORDS, g / 4u), g % 4u);
            let c0 = vec4<f32>(v[0], v[1], v[2], v[3]);
            let c1 = vec4<f32>(v[4], v[5], v[6], v[7]);
            let at = gi * 2u;
{sums}        }}
    }}
{store}    workgroupBarrier();
    if (live) {{
{out}    }}
}}
"#,
        red_len = 256 * rows,
        per = 256 / lanes,
        decoder = kind.wgsl(),
    )
}

/// [`few_source`]'s matmul on the tensor cores (WGSL's cooperative matrices, f16 into f32), for blocks of `rows` (16,
/// 32, 64 or 128) jobs of one matrix, as [`crate::exl3::g_coop`] is laid out: a workgroup 8 tile columns (128
/// outputs) of a block, `k` 16 at a time; each step a lane decodes eight weights of its warp's 16 outputs into the
/// workgroup's memory as f16 (a Q2_0 weight is one exactly) and the block's inputs beside them (rounded to f16), the
/// next step's loaded as this one's are multiplied; each warp its 16 outputs by the block's rows. `p` as
/// [`few_source`]'s (no identity order). A grid type's lane reads its 32's fields once for the two steps its groups
/// are in, and a group is one load of the grid (it read the fields a group, bytes off words' edges, and a grid entry
/// of two words: seven loads where Q2_0 takes two, and a Flash-Next layer's IQ2_S experts were 4.9 ms of a prompt's
/// 512 rows; they are 1.8).
fn coop_source(kind: Kind, rows: usize) -> String {
    assert!(matches!(rows, 16 | 32 | 64 | 128), "a block of 16, 32, 64 or 128 rows");
    // the decoder's state, the first step's eight weights and a later step `kn`'s
    let (state, first, next) = if kind.block().is_some() {
        ("    var h = sub_of(rb, 0u);\n", "var v = grp(h, hf);", "if (kn % 2u == 0u) { h = sub_of(rb + (kn / 16u) * BLOCK_WORDS, (kn / 2u) % 8u); }\n        var v = grp(h, 2u * (kn % 2u) + hf);")
    } else {
        ("", "var v = w8(rb, kn, hf, kw);", "var v = w8(rb, kn, hf, kw);")
    };
    let f = rows / 16;
    let pairs = rows * 8;
    let each = |g: &dyn Fn(usize) -> String| (0..f).map(g).collect::<String>();
    let decl = each(&|i| format!("    var c{i} = coop_mat16x16<f32, C>();\n"));
    let mma = each(&|i| format!("        {{\n            let bf = coopLoad<coop_mat16x16<f16, B>>(&xt[curx + {}u * S2], s2);\n            c{i} = coopMultiplyAdd(af, bf, c{i});\n        }}\n", i * 16));
    let out = each(&|i| {
        format!(
            "    {{\n        let so = warp * 256u;\n        coopStore(c{i}, &stage[so], 16u);\n        workgroupBarrier();\n        for (var e = l; e < 256u; e += 32u) {{\n            let id = ids[{}u + e / 16u];\n            if (id != 0xffffffffu && live) {{ y[id * n + tc * 16u + e % 16u] = stage[so + e]; }}\n        }}\n        workgroupBarrier();\n    }}\n",
            i * 16
        )
    });
    // the inputs a thread loads a step: pairs `t + 256 i` of the block's rows by 16 of k
    let per = pairs.div_ceil(256);
    let xdecl: String = (0..per).map(|i| format!("    var xp{i} = vec2<f32>(0.0);\n")).collect();
    let xload: String = (0..per)
        .map(|i| format!("        {{\n            let q = t + {}u;\n            if (q < {pairs}u) {{\n                xp{i} = vec2<f32>(0.0);\n                if (ids[q / 8u] != 0xffffffffu) {{ xp{i} = x2[(xb[q / 8u] + kn * 16u) / 2u + q % 8u]; }}\n            }}\n        }}\n", 256 * i))
        .collect();
    let xstore: String = (0..per).map(|i| format!("        {{\n            let q = t + {}u;\n            if (q < {pairs}u) {{ xt[nbx + (q / 8u) * S2 + q % 8u] = vec2<f16>(xp{i}); }}\n        }}\n", 256 * i)).collect();
    let wstore = "        wt[nbw + wo] = vec2<f16>(f16(v[0]), f16(v[1]));\n        wt[nbw + wo + 1u] = vec2<f16>(f16(v[2]), f16(v[3]));\n        wt[nbw + wo + 2u] = vec2<f16>(f16(v[4]), f16(v[5]));\n        wt[nbw + wo + 3u] = vec2<f16>(f16(v[6]), f16(v[7]));\n";
    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// a row's stride in the tiles, as f16 pairs: 16 of k and 8 against bank conflicts
const S2: u32 = 12u;
// two steps' weights [output][k] and inputs [row][k]
var<workgroup> wt: array<vec2<f16>, {wt_len}>;
var<workgroup> xt: array<vec2<f16>, {xt_len}>;
var<workgroup> stage: array<f32, 2048>;
// the block's jobs, and where each one's input row starts
var<workgroup> ids: array<u32, {rows}>;
var<workgroup> xb: array<u32, {rows}>;
{w8}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let rw = p[0].z;
    let ntiles = n / 16u;
    let blk = p[1].y + wg.z;
    if (t < {rows}u) {{
        let id = order[blk * {rows}u + t];
        var at = 0u;
        if (id != 0xffffffffu) {{ at = jobs[2u * id + 1u] * k; }}
        ids[t] = id;
        xb[t] = at;
    }}
    workgroupBarrier();
    // a block the order left unused (a GPU's grouping sizes the grid for the most blocks it could fill)
    let head = workgroupUniformLoad(&ids[0]);
    if (head == 0xffffffffu) {{
        return;
    }}
    let warp = t / 32u;
    let l = t % 32u;
    // the warp's tile column (past the matrix: the last, its sums not stored); the lane's output of its 16 and half
    // of the step's 16 weights
    let tc = wg.x * 8u + warp;
    let live = tc < ntiles;
    let o = l / 2u;
    let hf = l % 2u;
    let rb = jobs[2u * head] * p[1].x + (min(tc, ntiles - 1u) * 16u + o) * rw;
    let kw = k / 16u;
    let wo = (warp * 16u + o) * S2 + hf * 4u;
    let s2 = S2;
{decl}{xdecl}{state}    // the first step's
    {{
        let kn = 0u;
        {first}
{xload}        let nbw = 0u;
        let nbx = 0u;
{wstore}{xstore}    }}
    workgroupBarrier();
    for (var kt = 0u; kt < kw; kt++) {{
        // the next step (the last's own again, into the buffer no one reads after): loaded before this one's are
        // multiplied, stored after
        let kn = min(kt + 1u, kw - 1u);
        {next}
{xload}        let curx = (kt % 2u) * {half_xt}u;
        let af = coopLoadT<coop_mat16x16<f16, A>>(&wt[(kt % 2u) * {half_wt}u + warp * 16u * S2], s2);
{mma}        let nbw = ((kt + 1u) % 2u) * {half_wt}u;
        let nbx = ((kt + 1u) % 2u) * {half_xt}u;
{wstore}{xstore}        workgroupBarrier();
    }}
{out}}}
"#,
        wt_len = 2 * 128 * 12,
        xt_len = 2 * rows * 12,
        half_wt = 128 * 12,
        half_xt = rows * 12,
        w8 = kind.wgsl(),
    )
}

/// Each pair's SwiGLU, as its down projection reads it: `act[j] = silu(y[2 j]) * y[2 j + 1]`, rows of `p[0].x`,
/// `p[0].y` pairs.
const SWIGLU_PAIRS: &str = r#"
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(6) var<storage, read_write> act: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 16776960u;
    let ff = p[0].x;
    if (i >= p[0].y * ff) { return; }
    let pair = i / ff;
    let c = i % ff;
    let g = y[2u * pair * ff + c];
    act[i] = (g / (1.0 + exp(-g))) * y[(2u * pair + 1u) * ff + c];
}
"#;

/// A kernel's pipeline name and source, made once (a pipeline is named for good).
/// `mats`: 0 where the card holds every expert of the layer, else the group's matrices an expert, for the kernel that
/// reads an expert the card does not hold from the host's memory, or (`staged`) from where a prompt's were copied
/// ([`cached_source`]).
fn kernel(kind: Kind, coop: bool, rows: usize, k: usize, mats: usize, staged: bool) -> (&'static str, &'static str) {
    type Made = ((Kind, bool, usize, usize, usize, bool), (&'static str, &'static str));
    static MADE: Mutex<Vec<Made>> = Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|p| p.into_inner());
    // (the tensor cores' kernel is one for every width; the few-row one is written for its lanes a row)
    let lanes = if coop { 0 } else { few_lanes(kind, k) };
    let key = (kind, coop, rows, lanes, mats, staged);
    if let Some((_, k)) = made.iter().find(|(k, _)| *k == key) {
        return *k;
    }
    let part = if mats > 0 { format!("-{}{mats}", if staged { "staged" } else { "part" }) } else { String::new() };
    let name: &'static str = Box::leak(if coop { format!("quant-moe-{}-coop-{rows}{part}", kind.tag()) } else { format!("quant-moe-{}-few-{rows}-{lanes}{part}", kind.tag()) }.into_boxed_str());
    let source = if coop { coop_source(kind, rows) } else { few_source(kind, rows, lanes) };
    let source: &'static str = Box::leak(if mats > 0 { cached_source(source, mats, staged) } else { source }.into_boxed_str());
    made.push((key, (name, source)));
    (name, source)
}

/// [`few_source`]'s or [`coop_source`]'s kernel for a layer a card holds only some of whose experts ([`Cache`]): a
/// job's matrix read from its expert's slot on the card or, where it has none, from the host's memory, which holds
/// every expert's (binding 5: a thread's block is one matrix, so the one or the other for all it reads). `mats`: the
/// group's matrices an expert (2: gate and up; 1: down). The layer's state (binding 7) is each expert's slot, then
/// the pass each was last used in, then the passes so far: the gate and up kernel counts a pass, and the down one,
/// run after it, writes that count for its block's expert (a step's or a check's kernels the steps' and checks'
/// passes; a prompt's, `staged`, the prompts' own, kept after them: [`Cache`]'s state).
///
/// `staged`: binding 5 is the card's scratch, where [`EXPERTS_IN`] put the experts the run uses that the card has no
/// slot for, each at the place the state gives it ([`ADMIT`]). A prompt's kernel (the tensor
/// cores') takes that: it reads a word a thread between barriers, and from the host's memory each such read waits
/// for the bus (Flash-Next's prompt of 4,086 tokens read 16,600 experts so in 15.9 s, 0.95 ms each; a stream
/// brings one in 57 us).
fn cached_source(source: String, mats: usize, staged: bool) -> String {
    let once = |s: String, old: &str, new: String| -> String {
        assert_eq!(s.matches(old).count(), 1, "a kernel's `{old}`");
        s.replace(old, &new)
    };
    let coop = source.contains("jobs[2u * head] * p[1].x");
    let (job, first, expert) = if coop { ("jobs[2u * head]", "t == 0u && wg.x == 0u", "jobs[2u * head]") } else { ("jobs[2u * select(j0, 0u, !on0)]", "t == 0u && wg.x == 0u && wg.y == 0u && on0", "jobs[2u * j0]") };
    let mut s = once(source, &format!("{job} * p[1].x"), format!("mat_at({job}) * p[1].x"));
    // every read of the weights by where they are
    let mut out = String::with_capacity(s.len() + 2048);
    while let Some(at) = s.find("words[") {
        let end = at + s[at..].find(']').expect("an index's end");
        assert!(!s[at + 6..end].contains('['), "a kernel's read of its weights by another read");
        out.push_str(&s[..at]);
        out.push_str("wd(");
        out.push_str(&s[at + 6..end]);
        out.push(')');
        s = s[end + 1..].to_string();
    }
    out.push_str(&s);
    let helpers = format!(
        r#"@group(0) @binding(5) var<storage, read> coldw: array<u32>;
@group(0) @binding(7) var<storage, read_write> state: array<u32>;
// whether this thread's matrix is read from the host's memory (its expert has no slot on the card)
var<private> cold_src: bool;
fn wd(i: u32) -> u32 {{
    if (cold_src) {{
        return coldw[i];
    }}
    return words[i];
}}
// where a job's matrix `m` is: its expert's slot's on the card, or its own in the host's memory (or where the
// run's were copied from there)
fn mat_at(m: u32) -> u32 {{
    let e = m / {mats}u;
    let slot = state[e];
    cold_src = slot == 0xffffffffu;
    return select(slot, {other}, cold_src) * {mats}u + m % {mats}u;
}}
@compute"#,
        other = if staged { "state[3u * p[1].w + 4u + e]" } else { "e" },
    );
    let out = once(out, "@compute", helpers);
    let main = "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {\n";
    // (a prompt's rows count their own passes and say their own use: the state's fourth word after the steps' and
    // checks' passes, and the experts' words after it)
    let (count, used) = if staged { ("2u * p[1].w + 3u", "2u * p[1].w + 4u") } else { ("2u * p[1].w", "p[1].w") };
    if mats == 2 {
        // a pass counted (the pass's first dispatch's first thread)
        once(out, main, format!("{main}    if (t == 0u && wg.x == 0u && wg.y == 0u && wg.z == 0u && p[1].y == 0u) {{ state[{count}] = state[{count}] + 1u; }}\n"))
    } else {
        // the block's expert used in this pass (one thread a block says so)
        once(out, "    let rb = mat_at(", format!("    if ({first}) {{ state[{used} + {expert}] = state[{count}]; }}\n    let rb = mat_at("))
    }
}

/// An expert the card has no slot for, in a layer's state.
const MISS: u32 = u32::MAX;

/// The experts a pass takes in at most a layer: the victims the host lists, and the copies a pass makes.
const TAKE: usize = 32;

/// A pass's experts that the card has no slot for, seen to before the experts' kernels run. Each is given the slot of
/// one the host listed as least recently used (the state's victims, from `4 experts + 4`: [`TAKE`] experts, [`MISS`]
/// none; one this pass uses, or that has no slot any more, is passed over): the slots' map is changed here, and the
/// taking in listed for [`EXPERTS_IN`] (how many at `2 experts + 1`; each one's expert and slot after the victims).
/// When the victims are out, a prompt's (`p[0].z` 1) are numbered for the card's scratch (the state from `3 experts
/// + 4`: an expert's place, [`MISS`] none) and a few rows' are left for their kernels to read from the host's
/// memory. One workgroup: its threads mark the experts the pass's pairs name (`jobs`: a pair's expert its even
/// word) and say which have no slot; where any has none, one of them then goes through those pairs (a few rows':
/// the first 256 pairs') or the experts (a prompt's). Most layers of a step have none, and the one thread's loads are
/// one after another: going through every pair was 10 us a layer a pass. The experts not taken in are counted at
/// `2 experts + 2` (a prompt's numbered ones; a few rows' pairs left). `p[0]`: the pairs, the experts (1,024 at
/// most), whether the rest are numbered, the victims.
const ADMIT: &str = r#"
@group(0) @binding(0) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> state: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> mark: array<atomic<u32>, 1024>;
// the pairs whose expert has no slot: how many, and which of the first 256
var<workgroup> cold: atomic<u32>;
var<workgroup> none: array<u32, 256>;
// the victims tried, and the experts taken in
var<private> tried: u32;
var<private> taken: u32;

// expert `e` (no slot) given the next victim's that can go: whether there was one
fn take(e: u32) -> bool {
    let experts = p[0].y;
    let most = p[0].w;
    let victims = 4u * experts + 4u;
    while (tried < most) {
        let v = state[victims + tried];
        tried++;
        if (v < experts) {
            let slot = state[v];
            if (slot != 0xffffffffu && atomicLoad(&mark[v]) == 0u) {
                state[v] = 0xffffffffu;
                state[e] = slot;
                state[victims + most + 2u * taken] = e;
                state[victims + most + 2u * taken + 1u] = slot;
                taken++;
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let pairs = p[0].x;
    let experts = p[0].y;
    for (var e = t; e < experts; e += 256u) {
        atomicStore(&mark[e], 0u);
    }
    if (t == 0u) {
        atomicStore(&cold, 0u);
    }
    workgroupBarrier();
    for (var q = t; q < pairs; q += 256u) {
        let e = jobs[2u * q];
        atomicStore(&mark[e], 1u);
        let lacks = state[e] == 0xffffffffu;
        if (lacks) {
            atomicAdd(&cold, 1u);
        }
        if (q < 256u) {
            none[q] = u32(lacks);
        }
    }
    workgroupBarrier();
    if (t != 0u) {
        return;
    }
    tried = 0u;
    taken = 0u;
    var left = 0u;
    if (atomicLoad(&cold) != 0u) {
        if (p[0].z == 1u) {
            for (var e = 0u; e < experts; e++) {
                var at = 0xffffffffu;
                if (atomicLoad(&mark[e]) == 1u && state[e] == 0xffffffffu) {
                    if (!take(e)) {
                        at = left;
                        left++;
                    }
                }
                state[3u * experts + 4u + e] = at;
            }
        } else {
            for (var q = 0u; q < min(pairs, 256u); q++) {
                if (none[q] == 1u) {
                    let e = jobs[2u * q];
                    if (state[e] == 0xffffffffu) {
                        if (!take(e)) {
                            left++;
                        }
                    }
                }
            }
        }
    }
    state[2u * experts + 1u] = taken;
    state[2u * experts + 2u] = left;
}
"#;

/// Experts copied from the host's memory to the card: their gate and up matrices (`cg` to `hg`) and their down ones
/// (`cd` to `hd`), a workgroup 16,384 words, its 256 threads 256 consecutive words at each of 64 turns. A load from
/// the host's memory is the bus's fetch of its 64-byte line, kept for no later load: the threads that load at once
/// must share lines (each thread its own 64 words in turn had a warp's 32 loads in 32 lines, a line fetched for 4
/// bytes of it: 2.1 GB a second where the bus gives 26.6). `p[1].y` 1: the experts [`ADMIT`] took in, into their
/// slots (a workgroup's third index an entry of its list; past the list it ends at once); 0: the ones it numbered,
/// into the card's scratch at their places (the third index the expert; one with no place ends at once). `p[0]`: an
/// expert's words in the first group and in the second, the experts, the workgroups an expert's first group takes;
/// `p[1].x`: the victims ([`TAKE`]).
const EXPERTS_IN: &str = r#"
@group(0) @binding(0) var<storage, read> cg: array<u32>;
@group(0) @binding(1) var<storage, read> cd: array<u32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read_write> hg: array<u32>;
@group(0) @binding(7) var<storage, read_write> hd: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let experts = p[0].z;
    var e = wg.z;
    var at = 0xffffffffu;
    if (p[1].y == 1u) {
        if (wg.z < state[2u * experts + 1u]) {
            let entry = 4u * experts + 4u + p[1].x + 2u * wg.z;
            e = state[entry];
            at = state[entry + 1u];
        }
    } else {
        at = state[3u * experts + 4u + e];
    }
    if (at == 0xffffffffu) {
        return;
    }
    let gw = p[0].x;
    let dw = p[0].y;
    if (wg.x < p[0].w) {
        let first = wg.x * 16384u + t;
        for (var i = first; i < min(first + 16384u, gw); i += 256u) {
            hg[at * gw + i] = cg[e * gw + i];
        }
    } else {
        let first = (wg.x - p[0].w) * 16384u + t;
        for (var i = first; i < min(first + 16384u, dw); i += 256u) {
            hd[at * dw + i] = cd[e * dw + i];
        }
    }
}
"#;

/// A layer's routed experts where the card holds only some of them ([`QuantMoe::make`]'s `slots`): every expert's
/// matrices are in the host's memory ([`Gpu::host_buffer`]), which the card reads over the bus with no word from the
/// host between a layer's router and its experts. A pass takes the experts it uses that have no slot into the slots
/// of the least recently used before its experts' kernels run ([`ADMIT`], [`EXPERTS_IN`]: an expert crosses the bus
/// once, 57 us on eight lanes of PCIe 5), up to [`TAKE`] a layer; past that a few rows' kernels read an expert from
/// the host's memory where it is ([`cached_source`]) and a prompt's read it from the card's scratch, copied there.
/// The kernels write the pass each expert was last used in; a recording that ran the layer reads its state as it
/// finishes ([`Self::watch`]) and the host lists the next victims from it ([`Self::settle`]).
///
/// Flash-Next's GSQ-RCO IQ2_XS file is 37 GB of experts, 1.5 MB each, and one RTX 5090 holds 313 of a layer's 512.
/// Strata's request (4,086 tokens of code, then 256 of prose) there reads 24 experts a token from the host's
/// memory: whole layers' experts on the host's cores gave 46 tokens a second and a prompt of 24.5 s. A fixed set of
/// experts would read some 140 a token (the model's routing replayed: what a reply uses is not what its prompt did).
pub(crate) struct Cache {
    gpu: Arc<Gpu>,
    experts: usize,
    /// The experts the card holds.
    slots: usize,
    /// Each expert's slot on the card ([`MISS`]: none); the step or check each was last used in; four words (the
    /// steps and checks so far, the experts the last pass took in and those it did not, the prompts' passes so far);
    /// the prompt's pass each was last used in; each expert's place in a prompt's scratch; the victims ([`TAKE`]
    /// experts); and the last pass's taking in (an expert and its slot each).
    state: wgpu::Buffer,
    /// The gate and up group's, and the down group's: every expert's in the host's memory, the card's slots, and an
    /// expert's bytes there.
    groups: [(wgpu::Buffer, wgpu::Buffer, u64); 2],
    host: Mutex<Held>,
    /// The experts read from the host's memory (taken in or not), and those taken in, so far.
    counts: [std::sync::atomic::AtomicU64; 2],
}

/// What the host knows of a part-held layer: the experts' slots when it last looked, the passes then (the steps' and
/// checks', and the prompts'), and the victims it listed.
struct Held {
    map: Vec<u32>,
    seen: [u32; 2],
    victims: Vec<u32>,
}

/// The most words a prompt's scratch of experts takes on a device (by its address): the largest of its part-held
/// layers' two groups', so a recording takes one pair of vectors for all of them ([`Cache::admit`]).
static STAGE_MOST: Mutex<Vec<(usize, [usize; 2])>> = Mutex::new(Vec::new());

/// Every such layer's experts read from the host's memory and brought to a card since the process began.
static CACHED: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The experts cards have read from the host's memory, and those of them taken into a card's slots, since the process
/// began (every layer's that a card holds only some of): what a request's cost of them was is the difference over it.
pub fn cached_experts() -> (u64, u64) {
    (CACHED[0].load(Ordering::Relaxed), CACHED[1].load(Ordering::Relaxed))
}

impl Cache {
    /// Before a pass's experts' kernels (`jobs`: its `pairs`, a pair's expert its even word): its experts with no
    /// slot taken in, and (`staged`: a prompt's rows) those there was no victim for copied into the recording's
    /// scratch, where the tensor cores' kernels read them: the gate and up group's vector and the down group's
    /// (each layer's in turn the same two).
    fn admit(&self, rec: &mut crate::chain::Recorder<'_>, jobs: &DeviceVec, pairs: usize, staged: bool) -> Option<[DeviceVec; 2]> {
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let (d, drw) = (rec.gpu().dummy().clone(), rec.gpu().dummy_rw().clone());
        // an expert's words in each group; a workgroup 16,384 of them
        let words = [self.groups[0].2 as usize / 4, self.groups[1].2 as usize / 4];
        let per = |w: usize| w.div_ceil(16384) as u32;
        let wide = per(words[0]) + per(words[1]);
        let params = |listed: u32| [words[0] as u32, words[1] as u32, self.experts as u32, per(words[0]), TAKE as u32, listed];
        rec.dispatch_wide("moe-admit", ADMIT, [&buf(jobs), &d, &d, &d, &d, &d, &self.state, &drw], &[pairs as u32, self.experts as u32, staged as u32, TAKE as u32], (1, 1, 1));
        rec.dispatch_wide(
            "moe-take-in",
            EXPERTS_IN,
            [&self.groups[0].0, &self.groups[1].0, &self.state, &d, &d, &d, &self.groups[0].1, &self.groups[1].1],
            &params(1),
            (wide, 1, TAKE.min(pairs) as u32),
        );
        if !staged {
            return None;
        }
        // room for every expert the card has no slot for: the recording's one pair of vectors, as long as the
        // device's largest such layer needs (a layer of another type a pair of its own would hold them all to the
        // recording's end)
        let lens = words.map(|w| (self.experts - self.slots) * w);
        let (held, stage) = match rec.moe_stage.take() {
            Some((l, s)) if l[0] >= lens[0] && l[1] >= lens[1] => (l, s),
            _ => {
                let device = Arc::as_ptr(&self.gpu) as usize;
                let most = STAGE_MOST.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|(d, _)| *d == device).map_or(lens, |(_, m)| [m[0].max(lens[0]), m[1].max(lens[1])]);
                (most, most.map(|l| rec.scratch(l)))
            }
        };
        rec.moe_stage = Some((held, stage.clone()));
        rec.dispatch_wide(
            "moe-stage-copy",
            EXPERTS_IN,
            [&self.groups[0].0, &self.groups[1].0, &self.state, &d, &d, &d, &buf(&stage[0]), &buf(&stage[1])],
            &params(0),
            (wide, 1, self.experts as u32),
        );
        Some(stage)
    }

    /// `rec` reads the layer's state as it finishes (once a recording, however often it runs the layer).
    fn watch(self: &Arc<Self>, rec: &mut crate::chain::Recorder<'_>) {
        if rec.settles.iter().any(|(c, _)| Arc::ptr_eq(c, self)) {
            return;
        }
        let at = rec.read_of(&self.state, 0, 3 * self.experts + 4);
        rec.settles.push((Arc::clone(self), at));
    }

    /// What a recording read of the layer's state once it had run (`read`: its first three parts): the card's
    /// experts that no pass since the last look used are listed as the next victims ([`TAKE`] of them), the ones
    /// steps and checks used longest ago first (then the ones prompts did), where that list is another than the card
    /// has. So a prompt's chunks take in what they use and leave each other's, and the slots they take are the ones
    /// a reply would miss least: a reply's experts are not its prompt's, and by one count of use a prompt put out
    /// the experts replies had just used before any other.
    pub(crate) fn settle(&self, read: &[f32]) {
        let e = self.experts;
        let word = |i: usize| read[i].to_bits();
        let mut host = self.host.lock().unwrap_or_else(|p| p.into_inner());
        let seen = host.seen;
        host.seen = [word(2 * e), word(2 * e + 3)];
        for x in 0..e {
            host.map[x] = word(x);
        }
        // (when a step or a check last used expert `x`, and when a prompt's rows did)
        let used = |x: usize| (word(e + x), word(2 * e + 4 + x));
        // (the layer's last pass: the experts it took in, and those it read where they are or from a prompt's scratch)
        let (taken, left) = (read[2 * e + 1].to_bits() as u64, read[2 * e + 2].to_bits() as u64);
        for (count, n) in [left + taken, taken].into_iter().enumerate() {
            self.counts[count].fetch_add(n, Ordering::Relaxed);
            CACHED[count].fetch_add(n, Ordering::Relaxed);
        }
        let mut victims: Vec<u32> = (0..e).filter(|&x| word(x) != MISS && used(x).0 <= seen[0] && used(x).1 <= seen[1]).map(|x| x as u32).collect();
        victims.sort_by_key(|&x| used(x as usize));
        victims.resize(TAKE, MISS);
        if victims != host.victims {
            self.gpu.write(&self.state, ((4 * e + 4) * 4) as u64, bytemuck::cast_slice(&victims[..]));
            host.victims = victims;
        }
    }
}

/// One kind of a layer's routed experts' projection as a group (their gate and up matrices, or their down ones): their
/// rows in one buffer, a matrix's after another.
struct Group {
    words: wgpu::Buffer,
    kind: Kind,
    k: usize,
    n: usize,
    /// Words a row, and a matrix.
    rw: usize,
    mwords: usize,
    /// A grid type's table ([`Kind::table`]), which its kernels read at binding 4.
    table: Option<wgpu::Buffer>,
    /// Where the card holds only some of the layer's experts ([`Cache`]): every expert's rows in the host's memory
    /// (`words` then the card's slots').
    cold: Option<wgpu::Buffer>,
    /// The group's matrices an expert (2: gate and up; 1: down).
    mats: usize,
}

/// How a group's jobs are taken: each a block of its own (a step's, no order), or in blocks of one matrix (the order,
/// its blocks, the jobs a block): up to [`FEW_MAX`] a block in f32, 16 and more on the tensor cores.
#[derive(Clone, Copy)]
enum Order<'a> {
    Jobs,
    Blocks(&'a DeviceVec, usize, usize),
}

/// The shared expert's matrix: f16 where every value is one exactly (a Q2_0 matrix's are), else f32.
pub(crate) struct Dense {
    w: DeviceVec,
    half: bool,
    n: usize,
    k: usize,
}

impl Dense {
    pub(crate) fn new(b: &WgpuBackend, values: &[f32], n: usize, k: usize) -> Self {
        match DeviceChain::vec_f16(b, values) {
            Some(w) => Dense { w, half: true, n, k },
            None => {
                let w = b.vec(values.len());
                DeviceChain::upload(b, &w, values);
                Dense { w, half: false, n, k }
            }
        }
    }

    /// [`Self::rows`] of a SwiGLU's gate rows then its up rows (`[2 ff, k]`), and the SwiGLU, into `out` (`[rows,
    /// ff]`): one dispatch where the recorder has the two as one kernel (`fused` then unwritten), else the matmul into
    /// `fused` and the SwiGLU of it.
    pub(crate) fn rows_swiglu(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, fused: &DeviceVec, out: &DeviceVec, rows: usize) {
        if self.half && rec.matmul_f16_swiglu_rows(&self.w, self.n / 2, self.k, x, out, rows) {
            return;
        }
        self.rows(rec, x, fused, rows);
        rec.silu_mul_split_rows(fused, out, rows);
    }

    /// `first`'s rows then `second`'s as one matrix (`[n, k]`).
    pub(crate) fn stacked(b: &WgpuBackend, first: &[f32], second: &[f32], n: usize, k: usize) -> Self {
        let both: Vec<f32> = first.iter().chain(second).copied().collect();
        Dense::new(b, &both, n, k)
    }

    pub(crate) fn rows(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        if self.half {
            rec.matmul_f16_rows(&self.w, self.n, self.k, x, y, rows);
        } else {
            rec.matmul_f32_rows(&self.w, self.n, self.k, x, y, rows);
        }
    }
}

/// A MoE layer's GGUF experts on the GPU as groups, as [`crate::exl3::Exl3MoeGrouped`] holds EXL3's: the routed ones'
/// gate and up matrices in one buffer (matrix `2e` expert `e`'s gate, `2e + 1` its up), their down matrices in
/// another, the shared expert dense.
pub struct QuantMoe {
    b: WgpuBackend,
    routed: usize,
    hidden: usize,
    ff: usize,
    gu: Group,
    down: Group,
    /// The shared expert: its gate and up matrices one below the other (one matmul gives both, a dispatch fewer a
    /// layer), and its down one.
    shared: [Dense; 2],
    /// Whether a prompt's blocks go to the tensor cores (where the device has them; OAIY_QUANT_MOE_NO_COOP: no).
    coop: bool,
    /// The bytes counted against the backend's budget, given back when the layer goes.
    bytes: u64,
    /// Where the card holds only some of the routed experts: which, and the rest's place in the host's memory.
    cache: Option<Arc<Cache>>,
}

impl std::fmt::Debug for QuantMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantMoe({} routed experts of {}x{} in {} as two groups, and the shared one)", self.routed, self.hidden, self.ff, self.gu.kind.tag())
    }
}

impl Drop for QuantMoe {
    fn drop(&mut self) {
        self.b.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// What tells this module's scratch from EXL3's of the same shape, in the caches the two share.
const QUANT: usize = 1 << 40;

impl QuantMoe {
    /// The layer's experts as groups on `b`, if they can be: their types ones the kernels decode, each group within a
    /// binding, and all of them within the budget less `reserve`. Else `data` back, for the host.
    fn try_new(b: &WgpuBackend, data: QuantExpertsData, reserve: u64) -> Result<Self, QuantExpertsData> {
        Self::make(b, data, reserve, None)
    }

    /// [`Self::try_new`], the card holding `slots` of the routed experts where that is fewer than all of them (the
    /// first so many to begin with), every expert's matrices in the host's memory for the kernels to read the others
    /// from ([`Cache`]); `data` back too where the device's kernels cannot read the host's memory.
    fn make(b: &WgpuBackend, data: QuantExpertsData, reserve: u64, slots: Option<usize>) -> Result<Self, QuantExpertsData> {
        let (h, f, e) = (data.hidden, data.ff, data.experts);
        let (Some(kg), Some(ku), Some(kd)) = (Kind::of(data.gate.0), Kind::of(data.up.0), Kind::of(data.down.0)) else { return Err(data) };
        // (the tensor cores' tiles are 16 by 16, a scale's block 64; a grid type's block 256)
        if kg != ku || h % 64 != 0 || f % 64 != 0 || !kg.fits(h) || !kd.fits(f) {
            return Err(data);
        }
        let (gw, dw) = (kg.row_words(h), kd.row_words(f));
        let gu_bytes = (2 * e * f * gw * 4) as u64;
        let d_bytes = (e * h * dw * 4) as u64;
        let limit = chunk_limit(&b.gpu.limits);
        // (a job's matrix and row index its words in 32 bits)
        if gu_bytes > limit || d_bytes > limit || gu_bytes / 4 > u32::MAX as u64 || d_bytes / 4 > u32::MAX as u64 {
            return Err(data);
        }
        let held = slots.map_or(e, |s| s.clamp(1, e));
        let part = held < e;
        if part && !b.host_weights() {
            return Err(data);
        }
        let shared_bytes = (3 * h * f * 4) as u64;
        let total = (gu_bytes + d_bytes) / e as u64 * held as u64 + shared_bytes + if part { ((4 * e + 4 + 3 * TAKE) * 4) as u64 } else { 0 };
        let prev = b.used.fetch_add(total, Ordering::Relaxed);
        if prev + total > b.budget.saturating_sub(reserve) {
            b.used.fetch_sub(total, Ordering::Relaxed);
            return Err(data);
        }
        // a group's rows packed on every core, a matrix's after another (`which`: the tensors a matrix comes from in turn)
        let group = |which: &[&(GgmlType, Vec<u8>)], kind: Kind, n: usize, k: usize| -> Group {
            let (rw, rb) = (kind.row_words(k), row_bytes(which[0].0, k));
            let mwords = n * rw;
            let mut words = vec![0u32; e * which.len() * mwords];
            words.par_chunks_mut(mwords).enumerate().for_each(|(m, out)| {
                let src = &which[m % which.len()].1[(m / which.len()) * n * rb..];
                for (r, row) in out.chunks_exact_mut(rw).enumerate() {
                    kind.pack_row(&src[r * rb..(r + 1) * rb], k, row);
                }
            });
            // (the words' bytes as they lie: little-endian, as the kernels read them)
            let bytes: &[u8] = bytemuck::cast_slice(&words);
            let table = kind.table().map(|t| {
                let bytes: &[u8] = bytemuck::cast_slice(&t);
                b.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0
            });
            // (the card's slots: the first experts' to begin with; all of them in the host's memory beside)
            let hot = &bytes[..held * which.len() * mwords * 4];
            let cold = part.then(|| b.gpu.host_buffer(bytes).expect("a storage buffer in the host's memory"));
            Group { words: b.gpu.upload_rows(hot, hot.len(), 1).remove(0).0, kind, k, n, rw, mwords, table, cold, mats: which.len() }
        };
        let gu = group(&[&data.gate, &data.up], kg, f, h);
        let down = group(&[&data.down], kd, h, f);
        let shared = [Dense::stacked(b, &data.shared[0], &data.shared[1], 2 * f, h), Dense::new(b, &data.shared[2], h, f)];
        let coop = coop_on(&b.gpu) && std::env::var_os("OAIY_QUANT_MOE_NO_COOP").is_none();
        let cache = part.then(|| {
            // each expert's slot (the first `held` their own), no pass yet, no place in a prompt's scratch; the first
            // victims the last of the card's experts (none has been used)
            let mut state = vec![0u32; 4 * e + 4 + 3 * TAKE];
            for (x, s) in state[..e].iter_mut().enumerate() {
                *s = if x < held { x as u32 } else { MISS };
            }
            state[3 * e + 4..4 * e + 4].fill(MISS);
            let victims: Vec<u32> = (0..TAKE).map(|i| if i < held { (held - 1 - i) as u32 } else { MISS }).collect();
            state[4 * e + 4..4 * e + 4 + TAKE].copy_from_slice(&victims);
            let bytes: &[u8] = bytemuck::cast_slice(&state);
            let of = |g: &Group| (g.cold.clone().expect("a part-held group's rows in the host's memory"), g.words.clone(), (g.mats * g.mwords * 4) as u64);
            // (what a prompt's scratch of this layer's experts takes, for the device's largest)
            let lens = [(e - held) * gu.mats * gu.mwords, (e - held) * down.mats * down.mwords];
            let device = Arc::as_ptr(&b.gpu) as usize;
            let mut most = STAGE_MOST.lock().unwrap_or_else(|p| p.into_inner());
            match most.iter_mut().find(|(d, _)| *d == device) {
                Some((_, m)) => *m = [m[0].max(lens[0]), m[1].max(lens[1])],
                None => most.push((device, lens)),
            }
            drop(most);
            Arc::new(Cache {
                gpu: Arc::clone(&b.gpu),
                experts: e,
                slots: held,
                state: b.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0,
                groups: [of(&gu), of(&down)],
                host: Mutex::new(Held { map: state[..e].to_vec(), seen: [0, 0], victims }),
                counts: [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)],
            })
        });
        Ok(QuantMoe { b: b.clone(), routed: e, hidden: h, ff: f, gu, down, shared, coop, bytes: total, cache })
    }

    pub(crate) fn is_on(&self, gpu: &Arc<Gpu>) -> bool {
        Arc::ptr_eq(&self.b.gpu, gpu)
    }

    /// Scratch for `rows` rows of `top_k` experts, from `vec` (EXL3's [`Step`], what its caches keep: the vectors its
    /// transforms take here the SwiGLUs', the shared expert's in `part_gu` and each pair's in `xh_d`). `few`: a check's
    /// rows, whose jobs [`GROUP`] orders.
    fn scratch(&self, vec: &mut dyn FnMut(usize) -> DeviceVec, rows: usize, top_k: usize, few: bool) -> Step {
        let (h, f) = (self.hidden, self.ff);
        let pairs = rows * top_k;
        Step {
            top_k,
            jobs_gu: vec(4 * pairs),
            jobs_d: vec(2 * pairs),
            w: vec(rows * (top_k + 1)),
            xh_gu: vec(1),
            part_gu: vec(rows * f),
            out_gu: vec(2 * pairs * f),
            xh_d: vec(pairs * f),
            part_d: vec(1),
            out_d: vec(pairs * h),
            // (the shared expert's gate and up outputs side by side)
            sg: vec(rows * 2 * f),
            su: vec(1),
            sd: vec(rows * h),
            order_gu: vec(if few { 2 * pairs * rows } else { 1 }),
            order_d: vec(if few { pairs * rows } else { 1 }),
        }
    }

    /// The kept scratch of `rows` routed rows (a step's one, a check's few): one for every layer of this shape on the
    /// device (a layer's experts are done before the next layer's start), its bind groups kept.
    fn step(&self, rows: usize, top_k: usize) -> Arc<Step> {
        let key = [rows, top_k, self.hidden, self.ff, QUANT, QUANT];
        let mut s = self.b.gpu.moe_steps.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, st)) = s.iter().find(|(k, _)| *k == key) {
            return Arc::clone(st);
        }
        let st = Arc::new(self.scratch(&mut |n| self.b.vec(n), rows, top_k, rows > 1));
        s.push((key, Arc::clone(&st)));
        st
    }

    /// One group's jobs (`jobs`, `count` of them) on `x`'s rows into `y`'s (job `j`'s row `j`).
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn group_pass(&self, rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, jobs: &DeviceVec, count: usize, order: Order<'_>, y: &DeviceVec) {
        self.group_pass_from(rec, g, x, jobs, count, order, y, None)
    }

    /// [`Self::group_pass`]; `stage`: where a part-held layer's experts with no slot on the card were copied for this
    /// run ([`Cache::stage`]: a prompt's), else its kernels read them from the host's memory.
    #[allow(clippy::too_many_arguments)]
    fn group_pass_from(&self, rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, jobs: &DeviceVec, count: usize, order: Order<'_>, y: &DeviceVec, stage: Option<&DeviceVec>) {
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let (xb, jb, yb) = (buf(x), buf(jobs), buf(y));
        let ntiles = (g.n / 16) as u32;
        let (ob, blocks, rows, identity) = match order {
            Order::Jobs => (d.clone(), count, 1, 1),
            Order::Blocks(o, blocks, rows) => (buf(o), blocks, rows, 0),
        };
        let coop = rows > FEW_MAX;
        let (name, source) = kernel(g.kind, coop, rows, g.k, if self.cache.is_some() { g.mats } else { 0 }, stage.is_some());
        // (a card holding some of the experts: the rest's rows in the host's memory or where this run's were copied,
        // and the layer's state)
        let staged = stage.map(|s| buf(s));
        let (cold, state) = match (&self.cache, &staged) {
            (Some(c), Some(s)) => (s, &c.state),
            (Some(c), None) => (g.cold.as_ref().expect("a part-held group's rows in the host's memory"), &c.state),
            (None, _) => (&d, &drw),
        };
        // as many blocks a pass as the grid's third axis takes
        for first in (0..blocks).step_by(65535) {
            let these = 65535.min(blocks - first) as u32;
            let words = [g.n as u32, g.k as u32, g.rw as u32, identity, g.mwords as u32, first as u32, 0, self.routed as u32];
            let groups = (g.n as u32).div_ceil((256 / few_lanes(g.kind, g.k)) as u32);
            let grid = if coop { (ntiles.div_ceil(8), 1, these) } else { (groups.min(65535), groups.div_ceil(65535), these) };
            rec.dispatch_wide(name, source, [&g.words, &xb, &jb, &ob, g.table.as_ref().unwrap_or(&d), cold, &yb, state], &words, grid);
        }
        rec.weigh(2.0 * count as f64 * (g.n * g.k) as f64);
    }

    /// Record `assign`'s experts for each row of `x` into `out` (see `ChainRecorder::moe_rows`).
    pub(crate) fn record(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]) {
        let rows = assign.len();
        let h = self.hidden;
        let top_k = assign.first().map_or(0, |a| a.len().saturating_sub(1));
        assert!(rows > 0 && top_k > 0 && assign.iter().all(|a| a.len() == top_k + 1 && a[top_k].0 == self.routed), "moe: each row's routed experts, then the shared one");
        assert!(x.len >= rows * h && out.len >= rows * h, "moe: {rows} rows of {h}");
        // the jobs: gate and up of each (row, expert) on its row of x, then down of each on its hidden row
        let mut jobs_gu = Vec::with_capacity(4 * rows * top_k);
        let mut jobs_d = Vec::with_capacity(2 * rows * top_k);
        let mut w = Vec::with_capacity(rows * (top_k + 1));
        for (r, a) in assign.iter().enumerate() {
            for (j, &(e, wt)) in a[..top_k].iter().enumerate() {
                assert!(e < self.routed, "moe: expert {e} of {}", self.routed);
                jobs_gu.extend([2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32]);
                jobs_d.extend([e as u32, (r * top_k + j) as u32]);
                w.push(wt);
            }
            w.push(a[top_k].1);
        }
        let b = rec.backend().clone();
        // a step's one row: the kept scratch (its bind groups kept); else this call's (from the pool, given back when
        // the recording has run)
        let st = if rows == 1 && rec.keeps() { self.step(1, top_k) } else { Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, false)) };
        let up = |v: &DeviceVec, data: &[u32]| DeviceChain::upload(&b, v, &data.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>());
        up(&st.jobs_gu, &jobs_gu);
        up(&st.jobs_d, &jobs_d);
        DeviceChain::upload(&b, &st.w, &w);
        // a prompt's rows: each expert's in blocks, its weights decoded once a block (16 on the tensor cores)
        let block = if self.coop { 16 } else { FEW_MAX };
        let mut order = |jobs: &[u32]| {
            let o = many_order(jobs, block);
            let v = rec.scratch(o.len());
            up(&v, &o);
            (v, o.len() / block)
        };
        let orders = (rows > 1).then(|| (order(&jobs_gu), order(&jobs_d)));
        let (ogu, od) = match &orders {
            Some(((g, gn), (d, dn))) => (Order::Blocks(g, *gn, block), Order::Blocks(d, *dn, block)),
            None => (Order::Jobs, Order::Jobs),
        };
        self.run(rec, &st, x, out, rows, ogu, od, None);
    }

    /// `rows` rows' experts routed on the GPU from the router's `logits` (`[rows, routed + 1]`) and recorded into
    /// `out` (see `ChainRecorder::moe_routed`), as [`crate::exl3::Exl3MoeGrouped::record_routed`] records EXL3's:
    /// [`record_route`] writes the jobs and weights where [`Self::record`] uploads them. `into`: each row's sum added to its
    /// streams (the streams, their write weights, how many) where it would be `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_routed(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, into: Option<(&DeviceVec, &DeviceVec, usize)>) -> bool {
        // a prompt's rows (more than a check's) grouped by expert on the GPU, where the tensor cores take its blocks
        let many = rows > FEW_MAX && self.coop;
        if self.routed > 1024 || top_k == 0 || top_k > 32.min(self.routed) || rows == 0 || (rows > 64 && !many) || rows > 65535 || logits.len < rows * (self.routed + 1) {
            return false;
        }
        assert!(x.len >= rows * self.hidden && (into.is_some() || out.len >= rows * self.hidden), "moe: {rows} rows of {}", self.hidden);
        let pairs = rows * top_k;
        // a check's few rows: an expert the rows share decoded once for them (its jobs one block)
        let grouped = (2..=FEW_MAX).contains(&rows) && pairs <= 256;
        // (a prompt's scratch the recording's, each layer's in turn)
        let st = if rec.keeps() && !many && (rows == 1 || grouped) {
            self.step(rows, top_k)
        } else if many {
            let key = [rows, top_k, self.hidden, self.ff + QUANT];
            match rec.moe_tmp.take() {
                Some((k, st)) if k == key => {
                    rec.moe_tmp = Some((k, Arc::clone(&st)));
                    st
                }
                _ => {
                    let st = Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, false));
                    rec.moe_tmp = Some((key, Arc::clone(&st)));
                    st
                }
            }
        } else {
            Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, grouped))
        };
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        record_route(rec, &buf(logits), &st, self.routed, top_k, rows);
        if many {
            // blocks of some three times the jobs an expert has on average; the most blocks the experts could fill (a
            // part-filled one each at most), the grid that wide
            let bs = moe_rows_for(pairs, self.routed);
            let blocks = pairs.div_ceil(bs) + self.routed;
            let (og, od) = (rec.scratch(2 * bs * blocks), rec.scratch(bs * blocks + 3 * self.routed));
            let words = [pairs as u32, blocks as u32, self.routed as u32, bs as u32];
            let groups = ((2 * bs * blocks) as u32).div_ceil(256);
            let jd = buf(&st.jobs_d);
            rec.dispatch_wide("moe-many-clear", MANY_CLEAR, [&d, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, (groups.min(65535), groups.div_ceil(65535), 1));
            rec.dispatch_wide("moe-many-count", MANY_COUNT, [&jd, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            rec.dispatch_wide("moe-many-scan", MANY_SCAN, [&d, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, (1, 1, 1));
            rec.dispatch_wide("moe-many-scatter", MANY_SCATTER, [&jd, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            self.run(rec, &st, x, out, rows, Order::Blocks(&og, 2 * blocks, bs), Order::Blocks(&od, blocks, bs), into);
            return true;
        }
        let (ogu, od) = if grouped {
            rec.dispatch_wide("moe-group", GROUP, [&buf(&st.jobs_d), &d, &d, &d, &d, &d, &buf(&st.order_gu), &buf(&st.order_d)], &[pairs as u32, rows as u32], (1, 1, 1));
            (Order::Blocks(&st.order_gu, 2 * pairs, rows), Order::Blocks(&st.order_d, pairs, rows))
        } else {
            (Order::Jobs, Order::Jobs)
        };
        self.run(rec, &st, x, out, rows, ogu, od, into);
        true
    }

    /// The experts' work once `st` holds the jobs and weights: gate and up, each pair's SwiGLU, down, the shared
    /// expert on every row, and each row's weighted sum (into `out`, or added to the streams `into` names).
    #[allow(clippy::too_many_arguments)]
    fn run(&self, rec: &mut crate::chain::Recorder<'_>, st: &Step, x: &DeviceVec, out: &DeviceVec, rows: usize, ogu: Order<'_>, od: Order<'_>, into: Option<(&DeviceVec, &DeviceVec, usize)>) {
        let (h, f, top_k) = (self.hidden, self.ff, st.top_k);
        let pairs = rows * top_k;
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        // (a part-held layer: its state read as the recording finishes; the pass's experts with no slot taken in
        // first, and a prompt's rows' other such copied into the card's scratch for the tensor cores' kernels)
        let stage = self.cache.as_ref().and_then(|c| {
            c.watch(rec);
            c.admit(rec, &st.jobs_d, pairs, matches!(ogu, Order::Blocks(_, _, rows) if rows > FEW_MAX))
        });
        self.group_pass_from(rec, &self.gu, x, &st.jobs_gu, 2 * pairs, ogu, &st.out_gu, stage.as_ref().map(|s| &s[0]));
        let groups = ((pairs * f) as u32).div_ceil(256);
        rec.dispatch_wide("quant-moe-swiglu", SWIGLU_PAIRS, [&buf(&st.out_gu), &d, &d, &d, &d, &d, &buf(&st.xh_d), &drw], &[f as u32, pairs as u32], (groups.min(65535), groups.div_ceil(65535).max(1), 1));
        self.group_pass_from(rec, &self.down, &st.xh_d, &st.jobs_d, pairs, od, &st.out_d, stage.as_ref().map(|s| &s[1]));
        // the shared expert on every row: its gate and up with their SwiGLU, then its down projection with the
        // experts' sum into the streams, each one dispatch for a step's row or a check's few
        self.shared[0].rows_swiglu(rec, x, &st.sg, &st.part_gu, rows);
        if let Some((xs, post, streams)) = into {
            assert!(xs.len >= rows * streams * h && post.len >= rows * streams, "moe: {rows} rows' {streams} streams");
            let down = &self.shared[1];
            if down.half && rec.shared_down_wsum(&down.w, h, f, &st.part_gu, &st.out_d, &st.w, post, xs, rows, top_k, streams) {
                return;
            }
        }
        self.shared[1].rows(rec, &st.part_gu, &st.sd, rows);
        let groups = (((rows * h) as u32).div_ceil(256), 1, 1);
        match into {
            Some((xs, post, streams)) => {
                rec.dispatch_wide("moe-wsum-apply", WSUM_APPLY, [&buf(&st.out_d), &buf(&st.sd), &buf(&st.w), &buf(post), &d, &d, &buf(xs), &drw], &[h as u32, top_k as u32, rows as u32, streams as u32], groups);
            }
            None => rec.dispatch_wide("moe-wsum-rows", WSUM_ROWS, [&buf(&st.out_d), &buf(&st.sd), &buf(&st.w), &d, &d, &d, &buf(out), &drw], &[h as u32, top_k as u32, rows as u32], groups),
        }
    }
}

impl Experts for QuantMoe {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let h = self.hidden;
        let x = x.to_host();
        let logits = logits.to_host();
        let rows = x.numel() / h;
        let width = self.routed + 1;
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect();
        let b = &self.b;
        let (xd, out) = (b.vec(rows * h), b.vec(rows * h));
        DeviceChain::upload(b, &xd, x.data());
        let mut rec = b.begin();
        rec.keep_groups(false);
        rec.moe_rows(self, &xd, &out, &assign);
        rec.read(&out);
        let y = rec.finish().pop().expect("the experts' sum");
        Tensor::from_vec(y, vec![rows, h])
    }
}

impl WgpuBackend {
    /// A MoE layer's experts as a GGUF holds them: on the GPU as groups ([`QuantMoe`]) where the kernels decode their
    /// types and the weight budget holds them, else on the host, their shared expert here
    /// ([`crate::quant_host::quant_experts_host_beside`]).
    pub fn quant_experts(&self, data: QuantExpertsData) -> Result<Box<dyn Experts>, String> {
        self.quant_experts_leaving(data, 0)
    }

    /// As [`Self::quant_experts`] where the card has room for `slots` of the layer's routed experts only: it holds
    /// those (the ones last used, in time) and its kernels read the rest from the host's memory, which holds them
    /// all ([`Cache`]). On the host as [`Self::quant_experts`]'s where the device cannot ([`Self::host_weights`]).
    pub fn quant_experts_cached(&self, data: QuantExpertsData, slots: usize) -> Result<Box<dyn Experts>, String> {
        data.validate()?;
        match QuantMoe::make(self, data, 0, Some(slots)) {
            Ok(g) => Ok(Box::new(g)),
            Err(data) => crate::quant_host::quant_experts_host_beside(self, data),
        }
    }

    /// As [`Self::quant_experts`], leaving `reserve` bytes of the budget for the model's other matrices (loaded after).
    pub fn quant_experts_leaving(&self, data: QuantExpertsData, reserve: u64) -> Result<Box<dyn Experts>, String> {
        data.validate()?;
        match QuantMoe::try_new(self, data, reserve) {
            Ok(g) => Ok(Box::new(g)),
            Err(data) => crate::quant_host::quant_experts_host_beside(self, data),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn rng(mut seed: u64) -> impl FnMut() -> f32 {
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
        }
    }

    /// `matrices` of `rows` by `cols` in Q2_0: every code at random, each block a scale of its own (some negative).
    fn random_q2_0(next: &mut impl FnMut() -> f32, matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(matrices * rows * cols / 64 * 18);
        for _ in 0..matrices * rows * cols / 64 {
            let d = (0.015 + 0.01 * next()) * if next() > 0.8 { -1.0 } else { 1.0 };
            bytes.extend(half::f16::from_f32(d).to_bits().to_le_bytes());
            bytes.extend((0..16).map(|_| ((next() + 1.0) * 127.99) as u8));
        }
        bytes
    }

    /// `matrices` of `rows` by `cols` in a grid type (IQ2_S, IQ2_XXS or IQ1_M): every index, sign and scale at random,
    /// each block's own scale a small f16 (IQ1_M's in the top nibbles of its four words of scales).
    fn random_grid(next: &mut impl FnMut() -> f32, t: GgmlType, matrices: usize, rows: usize, cols: usize) -> Vec<u8> {
        let blocks = matrices * rows * cols / 256;
        let mut bytes = Vec::with_capacity(blocks * t.type_size());
        for _ in 0..blocks {
            let d = half::f16::from_f32(0.004 + 0.002 * next()).to_bits();
            if t == GgmlType::IQ1_M {
                bytes.extend((0..48).map(|_| ((next() + 1.0) * 127.99) as u8));
                for i in 0..4 {
                    let scales = ((next() + 1.0) * 2047.99) as u16;
                    bytes.extend((((d >> (4 * i)) & 15) << 12 | scales).to_le_bytes());
                }
            } else {
                bytes.extend(d.to_le_bytes());
                bytes.extend((0..t.type_size() - 2).map(|_| ((next() + 1.0) * 127.99) as u8));
            }
        }
        bytes
    }

    /// Experts at random: the routed ones Q2_0; the shared one's values f16's where `half` (as a Q2_0 matrix's are).
    pub(crate) fn experts(seed: u64, hidden: usize, ff: usize, count: usize, half: bool) -> QuantExpertsData {
        experts_of(seed, hidden, ff, count, half, GgmlType::Q2_0)
    }

    /// [`experts`] with the routed ones' gate and up matrices of type `gu` (their down ones Q2_0, as the GSQ-RCO
    /// IQ2_XS file has them).
    pub(crate) fn experts_of(seed: u64, hidden: usize, ff: usize, count: usize, half: bool, gu: GgmlType) -> QuantExpertsData {
        let mut next = rng(seed);
        if gu != GgmlType::Q2_0 {
            let gate = random_grid(&mut next, gu, count, ff, hidden);
            let up = random_grid(&mut next, gu, count, ff, hidden);
            let down = random_q2_0(&mut next, count, hidden, ff);
            let mut dense = |n: usize| -> Vec<f32> { (0..n).map(|_| if half { half::f16::from_f32(0.03 * next()).to_f32() } else { 0.03 * next() }).collect() };
            let shared = [dense(ff * hidden), dense(ff * hidden), dense(hidden * ff)];
            return QuantExpertsData { hidden, ff, experts: count, gate: (gu, gate), up: (gu, up), down: (GgmlType::Q2_0, down), shared };
        }
        let gate = random_q2_0(&mut next, count, ff, hidden);
        let up = random_q2_0(&mut next, count, ff, hidden);
        let down = random_q2_0(&mut next, count, hidden, ff);
        let mut dense = |n: usize| -> Vec<f32> { (0..n).map(|_| if half { half::f16::from_f32(0.03 * next()).to_f32() } else { 0.03 * next() }).collect() };
        let shared = [dense(ff * hidden), dense(ff * hidden), dense(hidden * ff)];
        QuantExpertsData { hidden, ff, experts: count, gate: (GgmlType::Q2_0, gate), up: (GgmlType::Q2_0, up), down: (GgmlType::Q2_0, down), shared }
    }

    pub(crate) fn copy(d: &QuantExpertsData) -> QuantExpertsData {
        QuantExpertsData { hidden: d.hidden, ff: d.ff, experts: d.experts, gate: d.gate.clone(), up: d.up.clone(), down: d.down.clone(), shared: d.shared.clone() }
    }

    /// `rows` rows of inputs and of router logits (no two alike).
    pub(crate) fn inputs(seed: u64, rows: usize, hidden: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
        let mut next = rng(seed);
        ((0..rows * hidden).map(|_| next()).collect(), (0..rows * (count + 1)).map(|_| 3.0 * next()).collect())
    }

    /// The worst difference between `got` and `want`, over `want`'s RMS.
    pub(crate) fn worst(got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len());
        let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        got.iter().zip(want).map(|(g, w)| (*g as f64 - *w as f64).abs()).fold(0f64, f64::max) / rms.max(1e-30)
    }

    /// The host's experts are the definition's: each row's top `k` routed experts by logit, softmax-weighted among
    /// themselves, and the shared one by its gate's sigmoid, each `down(silu(gate x) * up x)`, from every tensor
    /// dequantised whole and summed in three loops.
    #[test]
    fn the_hosts_experts_are_three_plain_loops() {
        let (hidden, ff, count, top_k, rows) = (256usize, 128usize, 12usize, 3usize, 5usize);
        let data = experts(11, hidden, ff, count, false);
        let (x, logits) = inputs(12, rows, hidden, count);
        let whole = |(t, bytes): &(GgmlType, Vec<u8>), n: usize| {
            let mut out = vec![0f32; n];
            ggml_quants::dequantize(*t, bytes, &mut out).unwrap();
            out
        };
        let (g, u, d) = (whole(&data.gate, count * ff * hidden), whole(&data.up, count * ff * hidden), whole(&data.down, count * hidden * ff));
        let mut want = vec![0f64; rows * hidden];
        for r in 0..rows {
            let l = &logits[r * (count + 1)..(r + 1) * (count + 1)];
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
            let top = &order[..top_k];
            let sum: f64 = top.iter().map(|&e| (l[e] as f64).exp()).sum();
            let mut parts: Vec<(f64, &[f32], &[f32], &[f32])> = top.iter().map(|&e| ((l[e] as f64).exp() / sum, &g[e * ff * hidden..(e + 1) * ff * hidden], &u[e * ff * hidden..(e + 1) * ff * hidden], &d[e * hidden * ff..(e + 1) * hidden * ff])).collect();
            parts.push((1.0 / (1.0 + (-l[count] as f64).exp()), &data.shared[0], &data.shared[1], &data.shared[2]));
            for (w, g, u, d) in parts {
                let act: Vec<f64> = (0..ff)
                    .map(|j| {
                        let dot = |m: &[f32]| (0..hidden).map(|c| m[j * hidden + c] as f64 * x[r * hidden + c] as f64).sum::<f64>();
                        let (a, b) = (dot(g), dot(u));
                        a / (1.0 + (-a).exp()) * b
                    })
                    .collect();
                for i in 0..hidden {
                    want[r * hidden + i] += w * (0..ff).map(|j| d[i * ff + j] as f64 * act[j]).sum::<f64>();
                }
            }
        }
        let want: Vec<f32> = want.iter().map(|v| *v as f32).collect();
        let host = quant_experts_cpu(data).unwrap();
        let got = host.forward(&Tensor::from_vec(x, vec![rows, hidden]), &Tensor::from_vec(logits, vec![rows, count + 1]), top_k).to_host();
        let e = worst(got.data(), &want);
        eprintln!("the host's experts against three loops: the worst error {e:.2e} of the RMS");
        assert!(e < 1e-4, "the host's experts: {e}");
    }

    /// The GPU's experts' sums by each of its ways in: routed on the host (`Experts::forward`, `moe_rows`), routed on
    /// the GPU (`moe_routed`), and those added to the streams (`moe_routed_into`; four streams, their sums the row's
    /// weights' times it). None where the device routes no such rows.
    #[allow(clippy::type_complexity)]
    fn gpu_sums(b: &WgpuBackend, moe: &dyn Experts, x: &[f32], logits: &[f32], rows: usize, hidden: usize, count: usize, top_k: usize) -> (Vec<f32>, Option<Vec<f32>>, Option<(Vec<f32>, Vec<f32>)>) {
        let hosted = moe.forward(&Tensor::from_vec(x.to_vec(), vec![rows, hidden]), &Tensor::from_vec(logits.to_vec(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
        let streams = 4;
        let (xd, ld, out, xs, post) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden), b.vec(rows * streams * hidden), b.vec(rows * streams));
        DeviceChain::upload(b, &xd, x);
        DeviceChain::upload(b, &ld, logits);
        let posts: Vec<f32> = (0..rows * streams).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
        let before: Vec<f32> = (0..rows * streams * hidden).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
        DeviceChain::upload(b, &post, &posts);
        DeviceChain::upload(b, &xs, &before);
        let mut rec = b.begin();
        rec.keep_groups(false);
        if !rec.moe_routed(moe, &xd, &out, &ld, top_k, rows) {
            return (hosted, None, None);
        }
        rec.read(&out);
        assert!(rec.moe_routed_into(moe, &xd, &xs, &post, &ld, top_k, rows, streams));
        rec.read(&xs);
        let mut read = rec.finish();
        let after = read.pop().unwrap();
        let routed = read.pop().unwrap();
        // what the streams should hold, from the routed sums
        let want: Vec<f32> = (0..rows * streams * hidden).map(|i| before[i] + posts[i / hidden] * routed[(i / (streams * hidden)) * hidden + i % hidden]).collect();
        (hosted, Some(routed), Some((after, want)))
    }

    /// The GPU's Q2_0 experts are the host's: a small layer (hidden 256, 12 experts, 3 a row) and Flash-Next's shape
    /// (2560 by 640, 24 experts, 10 a row), for a step's one row, a check's three, and a prompt's 40, 70 and 512, routed on
    /// the host and on the GPU, in f32 and (where the adapter has them) on the tensor cores, whose inputs are f16's.
    #[test]
    fn the_gpus_q2_0_experts_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        for (hidden, ff, count, top_k) in [(256usize, 128usize, 12usize, 3usize), (2560, 640, 24, 10)] {
            for half in [true, false] {
                let data = experts(31 + hidden as u64, hidden, ff, count, half);
                let host = quant_experts_cpu(copy(&data)).unwrap();
                let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("room for the experts");
                f32s.coop = false;
                let cores = QuantMoe::try_new(&b, data, 0).ok().expect("room for the experts");
                for rows in [1usize, 3, 40, 70, 512] {
                    let (x, logits) = inputs(7 + rows as u64, rows, hidden, count);
                    let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
                    for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
                        if !moe.coop && what != "f32" {
                            continue;
                        }
                        let (hosted, routed, into) = gpu_sums(&b, moe, &x, &logits, rows, hidden, count, top_k);
                        let e = worst(&hosted, &want);
                        let mut line = format!("{hidden} by {ff}, {count} experts ({} shared), {rows} rows, {what}: routed on the host {e:.2e}", if half { "an f16" } else { "an f32" });
                        // f32's sums to their rounding; the tensor cores' inputs are rounded to f16 (a block of 16
                        // jobs: the host's routing of any rows but a step's one, the GPU's of a prompt's), as the
                        // shared expert's f16 matmul rounds a prompt's
                        let shared = half && rows > 8 && coop_on(&b.gpu);
                        let bound = |cores: bool| if cores || shared { 6e-3 } else { 2e-5 };
                        assert!(e < bound(moe.coop && rows > 1), "{line}");
                        if let Some(routed) = routed {
                            let e = worst(&routed, &want);
                            line += &format!(", on the GPU {e:.2e}");
                            assert!(e < bound(moe.coop && rows > FEW_MAX), "{line}");
                            let (after, expect) = into.unwrap();
                            let e = worst(&after, &expect);
                            line += &format!(", into the streams {e:.2e}");
                            assert!(e < 1e-4, "{line}");
                        }
                        eprintln!("{line}");
                    }
                }
            }
        }
    }

    /// A layer a card holds some of whose experts is the layer it holds all of, bit for bit: a small layer (hidden
    /// 256, 12 experts, 3 a row) with 5 slots against the same all on the card, Q2_0 and IQ2_S, over passes of a
    /// step's row, a check's three and a prompt's 40 and 70, in f32 and on the tensor cores, by every way in; and
    /// its slots took in experts the passes used. Each pass's time is printed (new kernels: a slow one a stop sign).
    #[test]
    fn a_cards_share_of_a_layers_experts_is_all_of_them() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        if !b.host_weights() {
            eprintln!("this device's kernels do not read the host's memory");
            return;
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let (hidden, ff, count, top_k) = (256usize, 128usize, 12usize, 3usize);
        for gu in [GgmlType::Q2_0, GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
            for cores in [false, true] {
                if cores && !coop_on(&b.gpu) {
                    continue;
                }
                let data = experts_of(77, hidden, ff, count, true, gu);
                let mut whole = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("room for the experts");
                let mut part = QuantMoe::make(&b, data, 0, Some(5)).ok().expect("room for five of the experts");
                (whole.coop, part.coop) = (cores, cores);
                let cache = Arc::clone(part.cache.as_ref().expect("a layer of 12 experts in 5 slots is part-held"));
                assert!(whole.cache.is_none());
                for (pass, rows) in [1usize, 3, 1, 40, 70, 1, 3, 40, 1].into_iter().enumerate() {
                    let (x, logits) = inputs(100 + pass as u64, rows, hidden, count);
                    let want = gpu_sums(&b, &whole, &x, &logits, rows, hidden, count, top_k);
                    let clock = std::time::Instant::now();
                    let got = gpu_sums(&b, &part, &x, &logits, rows, hidden, count, top_k);
                    let ms = clock.elapsed().as_secs_f64() * 1e3;
                    let what = format!("{gu:?}, {} pass {pass} of {rows} rows ({ms:.0} ms)", if cores { "the tensor cores," } else { "f32," });
                    assert_eq!(bits(&got.0), bits(&want.0), "{what}: routed on the host");
                    assert_eq!(got.1.as_deref().map(bits), want.1.as_deref().map(bits), "{what}: routed on the GPU");
                    assert_eq!(got.2.as_ref().map(|(a, _)| bits(a)), want.2.as_ref().map(|(a, _)| bits(a)), "{what}: into the streams");
                }
                let (read, brought) = (cache.counts[0].load(Ordering::Relaxed), cache.counts[1].load(Ordering::Relaxed));
                // (the slots as the card left them: each still one expert's)
                let mut held: Vec<u32> = cache.host.lock().unwrap().map.iter().copied().filter(|s| *s != MISS).collect();
                held.sort_unstable();
                eprintln!("{gu:?}, {}: {read} experts read from the host's memory, {brought} of them taken into the card's {} slots", if cores { "the tensor cores" } else { "f32" }, held.len());
                assert!(read > 0 && brought > 0 && held == [0, 1, 2, 3, 4], "{gu:?}: the card's slots took in what the passes used: {held:?}");
            }
        }
    }

    /// What the grid types' kernels take for granted of ggml's tables: a grid's bytes are one of four values at most,
    /// so an entry is two bits a weight (each grid given back from its codes and values), and IQ2_XXS's sign pattern
    /// `i` is `i` with an eighth sign that makes its set bits even (the kernel computes it).
    #[test]
    fn the_grids_are_two_bits_a_weight() {
        for kind in [Kind::IQ2_S, Kind::IQ2_XXS, Kind::IQ1_M] {
            let (grid, signed) = kind.grid().unwrap();
            let (codes, levels) = grid_codes(grid, signed);
            assert_eq!(codes.len(), grid.len());
            for (e, c) in grid.iter().zip(&codes) {
                for (j, b) in e.to_le_bytes().iter().enumerate() {
                    let want = if signed { *b as i8 as f32 } else { *b as f32 };
                    assert_eq!(levels[((c >> (2 * j)) & 3) as usize], want, "{kind:?}: weight {j} of entry {e:#x}");
                }
            }
        }
        for (i, s) in ggml_quants::iq_tables::KSIGNS_IQ2XS.iter().enumerate() {
            assert_eq!(*s as u32, i as u32 | ((i as u32).count_ones() & 1) << 7, "sign pattern {i}");
        }
    }

    /// The GPU's experts whose gate and up matrices are a grid type's (IQ2_S, IQ2_XXS, IQ1_M; their down ones Q2_0: the
    /// GSQ-RCO IQ2_XS file's layers) are the host's reference: a small layer (hidden 256, 12 experts, 3 a row) for a
    /// step's row, a check's three and a prompt's 40 and 70, routed on the host and on the GPU, in f32 and (where the
    /// adapter has them) on the tensor cores, whose weights and inputs are rounded to f16; with
    /// OAIY_GRID_EXPERTS_FULL Flash-Next's shape too (2560 by 640, 24 experts, 10 a row, and 512 rows). Each case's
    /// time is printed: these kernels read their grids from a buffer, and a slow one is a stop sign.
    #[test]
    fn the_gpus_grid_experts_are_the_hosts() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let full = std::env::var_os("OAIY_GRID_EXPERTS_FULL").is_some();
        let shapes: &[(usize, usize, usize, usize)] = if full { &[(256, 128, 12, 3), (2560, 640, 24, 10)] } else { &[(256, 128, 12, 3)] };
        for gu in [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
            for &(hidden, ff, count, top_k) in shapes {
                let data = experts_of(53 + hidden as u64, hidden, ff, count, true, gu);
                let host = quant_experts_cpu(copy(&data)).unwrap();
                let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("the grid type's experts on the GPU");
                f32s.coop = false;
                let cores = QuantMoe::try_new(&b, data, 0).ok().expect("the grid type's experts on the GPU");
                let all_rows: &[usize] = if full && hidden > 256 { &[1, 3, 40, 512] } else { &[1, 3, 40, 70] };
                for &rows in all_rows {
                    let (x, logits) = inputs(7 + rows as u64, rows, hidden, count);
                    let want = host.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k).to_host().data().to_vec();
                    for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
                        if !moe.coop && what != "f32" {
                            continue;
                        }
                        let clock = std::time::Instant::now();
                        let (hosted, routed, into) = gpu_sums(&b, moe, &x, &logits, rows, hidden, count, top_k);
                        let ms = clock.elapsed().as_secs_f64() * 1e3;
                        let e = worst(&hosted, &want);
                        let mut line = format!("{gu:?} gate and up, {hidden} by {ff}, {count} experts, {rows} rows, {what} ({ms:.0} ms): routed on the host {e:.2e}");
                        // f32's sums to their rounding; the tensor cores' weights and inputs are rounded to f16, as
                        // the shared expert's f16 matmul rounds a prompt's
                        let shared = rows > 8 && coop_on(&b.gpu);
                        let bound = |cores: bool| if cores || shared { 8e-3 } else { 5e-5 };
                        assert!(e < bound(moe.coop && rows > 1), "{line}");
                        if let Some(routed) = routed {
                            let e = worst(&routed, &want);
                            line += &format!(", on the GPU {e:.2e}");
                            assert!(e < bound(moe.coop && rows > FEW_MAX), "{line}");
                            let (after, expect) = into.unwrap();
                            let e = worst(&after, &expect);
                            line += &format!(", into the streams {e:.2e}");
                            assert!(e < 1e-4, "{line}");
                        }
                        eprintln!("{line}");
                    }
                }
            }
        }
    }

    /// The tensor cores' matmul is the f32 kernel's where the inputs are f16's (a Q2_0 weight is one exactly, so the
    /// tiles lose nothing): a group's jobs in blocks of 16, 32, 64 and 128 (some places unused, some blocks none)
    /// against a job each, for the gate and up group and the down one.
    #[test]
    fn the_tensor_cores_matmul_is_f32s_on_f16_inputs() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        if !coop_on(&b.gpu) {
            return;
        }
        let (hidden, ff, count) = (2560usize, 640usize, 6usize);
        let moe = QuantMoe::try_new(&b, experts(41, hidden, ff, count, true), 0).ok().expect("room for the experts");
        let mut next = rng(42);
        for (g, matrices) in [(&moe.gu, 2 * count), (&moe.down, count)] {
            // 150 jobs over the group's matrices (unevenly), each on one of 40 input rows
            let inputs = 40usize;
            let x: Vec<f32> = (0..inputs * g.k).map(|_| half::f16::from_f32(2.0 * next()).to_f32()).collect();
            let jobs: Vec<u32> = (0..150u32).flat_map(|j| [(j * j % 7 + j % 3) % matrices as u32, (j * 11) % inputs as u32]).collect();
            let count = jobs.len() / 2;
            let (xd, jd) = (b.vec(x.len()), crate::exl3::u32_vec(&b, &jobs));
            DeviceChain::upload(&b, &xd, &x);
            let run = |order: Option<(Vec<u32>, usize)>| -> Vec<f32> {
                let y = b.vec(count * g.n);
                let mut rec = crate::chain::Recorder::new(&b);
                rec.keep_groups(false);
                match &order {
                    Some((o, rows)) => {
                        let ov = crate::exl3::u32_vec(&b, o);
                        moe.group_pass(&mut rec, g, &xd, &jd, count, Order::Blocks(&ov, o.len() / rows, *rows), &y);
                    }
                    None => moe.group_pass(&mut rec, g, &xd, &jd, count, Order::Jobs, &y),
                }
                rec.read(&y);
                Box::new(rec).finish().pop().unwrap()
            };
            let want = run(None);
            for rows in [16usize, 32, 64, 128] {
                let mut order = many_order(&jobs, rows);
                // (a block of none between the others, as a GPU's grouping leaves them)
                order.splice(rows..rows, vec![crate::exl3::NONE; rows]);
                let got = run(Some((order, rows)));
                let e = worst(&got, &want);
                eprintln!("[{}, {}], blocks of {rows} on the tensor cores against a job each in f32: the worst error {e:.2e} of the RMS", g.n, g.k);
                // (the sums of 2,560 products in another order)
                assert!(e < 1e-4, "blocks of {rows}: {e}");
            }
            // and the f32 kernel's blocks of 2 to 8
            for rows in 2..=FEW_MAX {
                let got = run(Some((many_order(&jobs, rows), rows)));
                let e = worst(&got, &want);
                assert!(e < 2e-6, "[{}, {}], blocks of {rows} in f32: {e}", g.n, g.k);
            }
        }
    }

    /// A step's experts with their bind groups kept are the same sums a second time, and a layer that goes gives its
    /// bytes back to the budget.
    #[test]
    fn a_kept_step_repeats_and_a_layer_gives_its_bytes_back() {
        let Ok(b) = WgpuBackend::new(Some(1 << 30)) else { return };
        let (hidden, ff, count, top_k) = (256usize, 128usize, 12usize, 3usize);
        let used = b.used.load(Ordering::Relaxed);
        let data = experts(5, hidden, ff, count, true);
        let host = quant_experts_cpu(copy(&data)).unwrap();
        let moe = b.quant_experts(data).unwrap();
        assert!(DeviceChain::holds_experts(&b, moe.as_ref()) && b.used.load(Ordering::Relaxed) > used, "the experts on the GPU, counted");
        let (xd, ld, out) = (b.vec(hidden), b.vec(count + 1), b.vec(hidden));
        for step in 0..3u64 {
            let (x, logits) = inputs(90 + step, 1, hidden, count);
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &ld, &logits);
            let mut rec = b.begin();
            assert!(rec.moe_routed(moe.as_ref(), &xd, &out, &ld, top_k, 1));
            rec.read(&out);
            let got = rec.finish().pop().unwrap();
            let want = host.forward(&Tensor::from_vec(x, vec![1, hidden]), &Tensor::from_vec(logits, vec![1, count + 1]), top_k).to_host();
            let e = worst(&got, want.data());
            assert!(e < 2e-5, "step {step}: {e}");
        }
        drop(moe);
        assert_eq!(b.used.load(Ordering::Relaxed), used, "the layer's bytes given back");
        // experts of a type the kernels do not decode run on the host
        let mut other = experts(6, hidden, ff, count, true);
        let q4 = |n: usize| (ggml_quants::GgmlType::Q4_0, vec![0u8; n / 32 * 18]);
        (other.gate, other.up, other.down) = (q4(count * ff * hidden), q4(count * ff * hidden), q4(count * hidden * ff));
        let hosted = b.quant_experts(other).unwrap();
        assert!(!DeviceChain::holds_experts(&b, hosted.as_ref()), "Q4_0 experts are the host's");
    }

    /// The experts' kernels' time (`--ignored --nocapture`): Flash-Next's layer (512 experts of 2560 by 640 in Q2_0,
    /// 10 a row), a prompt's 512 rows in f32 and on the tensor cores, routed on the host and on the GPU, and a step's
    /// one row.
    #[test]
    #[ignore = "a measurement"]
    fn measure_q2_0_experts() {
        let Ok(b) = WgpuBackend::new(Some(3 << 30)) else { return };
        let (hidden, ff, count, top_k) = (2560usize, 640usize, 512usize, 10usize);
        let data = experts(77, hidden, ff, count, true);
        let mut f32s = QuantMoe::try_new(&b, copy(&data), 0).ok().expect("room for the experts");
        f32s.coop = false;
        let cores = QuantMoe::try_new(&b, data, 0).ok().expect("room for the experts");
        for rows in [512usize, 1] {
            let (x, logits) = inputs(3, rows, hidden, count);
            let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
            let (xd, ld, out) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xd, &x);
            DeviceChain::upload(&b, &ld, &logits);
            for (what, moe) in [("f32", &f32s), ("the tensor cores", &cores)] {
                if what != "f32" && (!moe.coop || rows == 1) {
                    continue;
                }
                for device in [false, true] {
                    let reps = if rows == 1 { 200 } else { 8 };
                    let run = || {
                        let mut rec = b.begin();
                        rec.keep_groups(rows == 1);
                        for _ in 0..reps {
                            if device {
                                if !rec.moe_routed(moe, &xd, &out, &ld, top_k, rows) {
                                    return false;
                                }
                            } else {
                                rec.moe_rows(moe, &xd, &out, &assign);
                            }
                        }
                        rec.read_range(&out, 0, 1);
                        rec.finish();
                        true
                    };
                    if !run() {
                        eprintln!("{rows} rows, {what}, routed on the GPU: not this device's");
                        continue;
                    }
                    let t = std::time::Instant::now();
                    for _ in 0..3 {
                        run();
                    }
                    let each = t.elapsed().as_secs_f64() / 3.0 / reps as f64;
                    let flops = 2.0 * (rows * top_k) as f64 * (3 * hidden * ff) as f64;
                    eprintln!("{rows} rows of {top_k} of {count} experts, {what}, routed on the {}: {:.3} ms a layer ({:.1} TFLOPS)", if device { "GPU" } else { "host" }, each * 1e3, flops / each / 1e12);
                }
            }
        }
    }
}
