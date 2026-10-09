//! The experts' kernels: a quant type's decode (`Kind`), and with it the matmul of a check's few rows and of a
//! prompt's blocks by tensor cores, the SwiGLU between, and the forms that read a card's cache of experts.

use super::*;

/// A weight type the GPU's kernels decode: its rows' layout there and its `w8`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    Q2_0,
    IQ2_S,
    IQ2_XXS,
    IQ1_M,
}

impl Kind {
    pub(super) fn of(t: GgmlType) -> Option<Kind> {
        match t {
            GgmlType::Q2_0 => Some(Kind::Q2_0),
            GgmlType::IQ2_S => Some(Kind::IQ2_S),
            GgmlType::IQ2_XXS => Some(Kind::IQ2_XXS),
            GgmlType::IQ1_M => Some(Kind::IQ1_M),
            _ => None,
        }
    }

    pub(super) fn tag(self) -> &'static str {
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
    pub(super) fn fits(self, k: usize) -> bool {
        k % if self.block().is_some() { 256 } else { 64 } == 0
    }

    /// The words a row of `k` weights takes on the GPU.
    pub(super) fn row_words(self, k: usize) -> usize {
        match self.block() {
            // a word 16 weights' codes, then a word two blocks' scales
            None => k / 16 + (k / 64).div_ceil(2),
            Some((_, words)) => k / 256 * words,
        }
    }

    /// A row's blocks (`src`, `k` weights as the GGUF has them) as the GPU holds it (`dst`, [`Self::row_words`] long).
    pub(super) fn pack_row(self, src: &[u8], k: usize, dst: &mut [u32]) {
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
    pub(super) fn grid(self) -> Option<(&'static [u64], bool)> {
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
    pub(super) fn table(self) -> Option<Vec<u32>> {
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
pub(super) fn grid_codes(grid: &[u64], signed: bool) -> (Vec<u32>, [f32; 4]) {
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
pub(super) fn few_lanes(kind: Kind, k: usize) -> usize {
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
pub(super) const SWIGLU_PAIRS: &str = r#"
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
///
/// `lanes`: the few rows' kernel's lanes a row where not [`few_lanes`]'s (0: those).
pub(super) fn kernel(kind: Kind, coop: bool, rows: usize, k: usize, mats: usize, staged: bool, lanes: usize) -> (&'static str, &'static str) {
    type Made = ((Kind, bool, usize, usize, usize, bool), (&'static str, &'static str));
    static MADE: Mutex<Vec<Made>> = Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|p| p.into_inner());
    // (the tensor cores' kernel is one for every width; the few-row one is written for its lanes a row)
    let lanes = if coop { 0 } else if lanes > 0 { lanes } else { few_lanes(kind, k) };
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
